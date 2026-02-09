use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Program, SELF_SLOT, VarSlot};
use crate::simulator::checker::state::{Budget, SearchNode};
use crate::simulator::core::{
    Continuation, Env, LogEntry, Logger, NodeId, OpKind, Operation, Record, Runnable, State,
    UpdatePolicy, Value, exec, exec_sync_on_node, make_local_env,
};
use crate::simulator::coverage::{GlobalState, LocalCoverage};
use crate::simulator::hash_utils::HashPolicy;
use ecow::EcoString;
use log::{info, warn};
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

mod state;

#[derive(Clone, Default, Debug)]
pub struct SearchLogger {
    pub logs: Vec<LogEntry>,
}

impl Logger for SearchLogger {
    fn log(&mut self, entry: LogEntry) {
        self.logs.push(entry);
    }
}

pub struct CheckerConfig {
    pub max_depth: u32,
    pub num_servers: usize,
    pub num_clients: usize,
    pub initial_budget: Budget,
    pub keys: Vec<EcoString>,
}

struct VisitedState<H: HashPolicy> {
    budget: Budget,
    steps: u32,
    shortest_history: Vec<Operation<H>>,
}

pub struct ModelChecker<H: HashPolicy> {
    program: Arc<Program>,
    queue: BinaryHeap<SearchNode<H>>,
    /// Maps state hash -> Metadata about the best visit to this state.
    visited: HashMap<u64, VisitedState<H>>,
    global_state: Arc<GlobalState>,
    config: CheckerConfig,
}

impl<H: HashPolicy> ModelChecker<H> {
    pub fn new(program: Program, global_state: Arc<GlobalState>, config: CheckerConfig) -> Self {
        Self {
            program: Arc::new(program),
            queue: BinaryHeap::new(),
            visited: HashMap::new(),
            global_state,
            config,
        }
    }

    fn hash_state(state: &State<H>) -> u64 {
        state.signature()
    }

    fn should_visit(&mut self, node: &SearchNode<H>) -> bool {
        let h = Self::hash_state(&node.state);

        if let Some(prev) = self.visited.get(&h) {
            // Criteria for re-visiting:
            // 1. More resources (Budget) allows finding bugs we couldn't before.
            let better_budget = node.best_budget.crashes > prev.budget.crashes
                || node.best_budget.writes > prev.budget.writes
                || node.best_budget.reads > prev.budget.reads;

            // 2. Shorter path (Steps) to the same state.
            let shorter_path = node.steps < prev.steps;

            if !better_budget && !shorter_path {
                return false;
            }
        }

        // Update visited with current best path
        self.visited.insert(
            h,
            VisitedState {
                budget: node.best_budget.clone(),
                steps: node.steps,
                shortest_history: node.history.clone(),
            },
        );
        true
    }

    pub fn explore(&mut self, initial_state: State<H>) {
        info!("Starting State-Based Exploration");

        let root = SearchNode {
            state: initial_state,
            cost: 0.0,
            steps: 0,
            best_budget: self.config.initial_budget.clone(),
            history: Vec::new(),
            logs: Vec::new(),
            next_op_id: 0,
        };

        self.queue.push(root);

        let mut iterations = 0;

        while let Some(node) = self.queue.pop() {
            iterations += 1;
            if iterations % 1000 == 0 {
                info!(
                    "Checker: {} steps, queue size {}, visited {}",
                    iterations,
                    self.queue.len(),
                    self.visited.len()
                );
            }

            if node.steps >= self.config.max_depth {
                continue;
            }

            if !self.should_visit(&node) {
                continue;
            }

            // Generate successors
            let successors = self.expand(node);

            for succ in successors {
                self.queue.push(succ);
            }
        }

        info!(
            "Exploration finished. Visited {} unique states.",
            self.visited.len()
        );
    }

    fn expand(&self, node: SearchNode<H>) -> Vec<SearchNode<H>> {
        let mut successors = Vec::new();

        // Runnable items
        for i in 0..node.state.runnable_tasks.len() {
            let mut next_state = node.state.clone();
            let runnable = next_state.runnable_tasks.remove(i);

            // Handle timers - drop if node is crashed, otherwise execute
            let record = match runnable {
                Runnable::Timer(timer) => {
                    if next_state
                        .crash_info
                        .currently_crashed
                        .contains(&timer.node)
                    {
                        // Timer is dropped for crashed node
                        continue;
                    }
                    // Execute timer: send unit value to its channel
                    if let Some(mut chan) = next_state.channels.get(&timer.channel).cloned() {
                        if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                            let mut r_node_env = next_state.nodes[reader.node.index].clone();
                            if let Err(e) = crate::simulator::core::eval::store(
                                &lhs,
                                Value::<H>::unit(),
                                &mut reader.env,
                                &mut r_node_env,
                            ) {
                                log::warn!("Store failed in timer completion: {}", e);
                            }
                            next_state.nodes[reader.node.index] = r_node_env;
                            next_state
                                .runnable_tasks
                                .push_back(Runnable::Record(reader));
                        } else {
                            chan.buffer.push_back(Value::<H>::unit());
                        }
                        next_state.channels.insert(timer.channel, chan);
                    }
                    let succ = self.make_successor(&node, next_state, 0.5);
                    successors.push(succ);
                    continue;
                }
                Runnable::Record(r) => r,
                Runnable::ChannelSend {
                    target,
                    channel,
                    message,
                    origin_node,
                    x,
                    policy,
                    pc,
                } => {
                    if next_state.crash_info.currently_crashed.contains(&target) {
                        continue;
                    }

                    // Execute send: check for waiting readers
                    if let Some(mut chan) = next_state.channels.get(&channel).cloned() {
                        if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                            let mut r_node_env = next_state.nodes[reader.node.index].clone();
                            if let Err(e) = crate::simulator::core::eval::store(
                                &lhs,
                                message,
                                &mut reader.env,
                                &mut r_node_env,
                            ) {
                                log::warn!("Store failed in channel send completion: {}", e);
                            }
                            next_state.nodes[reader.node.index] = r_node_env;
                            next_state
                                .runnable_tasks
                                .push_back(Runnable::Record(reader));
                        } else {
                            chan.buffer.push_back(message);
                        }
                        next_state.channels.insert(channel, chan);
                    }
                    // Create successor
                    let succ = self.make_successor(&node, next_state, x);
                    successors.push(succ);
                    continue;
                }
            };

            // Handle crashed nodes (drop or queue message)
            if next_state
                .crash_info
                .currently_crashed
                .contains(&record.node)
            {
                if record.origin_node != record.node {
                    next_state
                        .crash_info
                        .queued_messages
                        .push_back((record.node, record));
                    // Valid state transition for queuing
                    successors.push(self.make_successor(&node, next_state, 0.0));
                }
                continue;
            }

            let mut logger = SearchLogger::default();
            let mut coverage = LocalCoverage::new();

            let result = exec(
                &mut next_state,
                &mut logger,
                &self.program,
                record,
                None,
                &mut coverage,
            );

            match result {
                Ok(None) => {
                    let cost_delta = 1.0 - coverage.novelty_score();
                    let mut succ = self.make_successor(&node, next_state, cost_delta);
                    succ.logs.extend(logger.logs);
                    successors.push(succ);
                }
                Ok(Some(client_res)) => {
                    let cost_delta = 1.0 - coverage.novelty_score();
                    let mut succ = self.make_successor(&node, next_state, cost_delta);
                    succ.history.push(Operation::<H> {
                        client_id: client_res.client_id,
                        op_action: client_res.op_name,
                        kind: OpKind::Response,
                        payload: vec![client_res.value],
                        unique_id: client_res.unique_id,
                    });
                    succ.logs.extend(logger.logs);
                    successors.push(succ);
                }
                Err(e) => {
                    warn!("Runtime error in path: {}", e);
                }
            }
        }

        // Crashes
        if node.best_budget.crashes > 0 {
            let server_role = self
                .program
                .roles
                .iter()
                .find(|(_, name)| name == "Node")
                .map(|(id, _)| *id)
                .unwrap_or(NameId(0));
            for i in 0..self.config.num_servers {
                let node_id = NodeId {
                    role: server_role,
                    index: i,
                };
                if !node.state.crash_info.currently_crashed.contains(&node_id) {
                    let mut next_state = node.state.clone();
                    self.apply_crash(&mut next_state, node_id);

                    let mut succ = self.make_successor(&node, next_state, 5.0);
                    succ.history.push(Operation::<H> {
                        client_id: -1,
                        op_action: "System.Crash".to_string(),
                        kind: OpKind::Crash,
                        payload: vec![Value::<H>::node(node_id)],
                        unique_id: -1,
                    });
                    succ.best_budget.crashes -= 1;
                    successors.push(succ);
                }
            }
        }

        // Recovery
        if node.best_budget.recovers > 0 {
            let crashed: Vec<NodeId> = node
                .state
                .crash_info
                .currently_crashed
                .iter()
                .copied()
                .collect();
            for node_id in crashed {
                let mut next_state = node.state.clone();
                let mut logger = SearchLogger::default();
                let mut coverage = LocalCoverage::new();

                self.apply_recover(&mut next_state, &mut logger, &mut coverage, node_id);

                let mut succ = self.make_successor(&node, next_state, 3.0);
                succ.history.push(Operation::<H> {
                    client_id: -1,
                    op_action: "System.Recover".to_string(),
                    kind: OpKind::Recover,
                    payload: vec![Value::<H>::node(node_id)],
                    unique_id: -1,
                });
                succ.logs.extend(logger.logs);
                succ.best_budget.recovers -= 1;
                successors.push(succ);
            }
        }

        // Writes
        if node.best_budget.writes > 0 {
            let server_role = self
                .program
                .roles
                .iter()
                .find(|(_, name)| name == "Node")
                .map(|(id, _)| *id)
                .unwrap_or(NameId(0));
            let client_role = self
                .program
                .roles
                .iter()
                .find(|(_, name)| name == "ClientInterface")
                .map(|(id, _)| *id)
                .unwrap_or(NameId(1));
            for client_idx in
                self.config.num_servers..self.config.num_servers + self.config.num_clients
            {
                let client_node_id = NodeId {
                    role: client_role,
                    index: client_idx,
                };
                let server_node_id = NodeId {
                    role: server_role,
                    index: 0,
                };
                for key in &self.config.keys {
                    let mut next_state = node.state.clone();
                    let op_id = node.next_op_id;

                    if self.schedule_client_write(
                        &mut next_state,
                        client_node_id,
                        op_id,
                        server_node_id,
                        key,
                        "val1",
                    ) {
                        let mut succ = self.make_successor(&node, next_state, 1.0);
                        succ.history.push(Operation::<H> {
                            client_id: client_idx as i32,
                            op_action: "ClientInterface.Write".to_string(),
                            kind: OpKind::Invocation,
                            payload: vec![
                                Value::<H>::node(server_node_id),
                                Value::<H>::string(key.clone()),
                                Value::<H>::string(EcoString::from("val1")),
                            ],
                            unique_id: op_id,
                        });
                        succ.next_op_id += 1;
                        succ.best_budget.writes -= 1;
                        successors.push(succ);
                    }
                }
            }
        }

        // Reads
        if node.best_budget.reads > 0 {
            let server_role = self
                .program
                .roles
                .iter()
                .find(|(_, name)| name == "Node")
                .map(|(id, _)| *id)
                .unwrap_or(NameId(0));
            let client_role = self
                .program
                .roles
                .iter()
                .find(|(_, name)| name == "ClientInterface")
                .map(|(id, _)| *id)
                .unwrap_or(NameId(1));
            for client_idx in
                self.config.num_servers..self.config.num_servers + self.config.num_clients
            {
                let client_node_id = NodeId {
                    role: client_role,
                    index: client_idx,
                };
                let server_node_id = NodeId {
                    role: server_role,
                    index: 0,
                };
                for key in &self.config.keys {
                    let mut next_state = node.state.clone();
                    let op_id = node.next_op_id;

                    if self.schedule_client_read(
                        &mut next_state,
                        client_node_id,
                        op_id,
                        server_node_id,
                        key,
                    ) {
                        let mut succ = self.make_successor(&node, next_state, 1.0);
                        succ.history.push(Operation::<H> {
                            client_id: client_idx as i32,
                            op_action: "ClientInterface.Read".to_string(),
                            kind: OpKind::Invocation,
                            payload: vec![
                                Value::<H>::node(server_node_id),
                                Value::<H>::string(key.clone()),
                            ],
                            unique_id: op_id,
                        });
                        succ.next_op_id += 1;
                        succ.best_budget.reads -= 1;
                        successors.push(succ);
                    }
                }
            }
        }

        successors
    }

    fn make_successor(
        &self,
        parent: &SearchNode<H>,
        state: State<H>,
        cost_delta: f64,
    ) -> SearchNode<H> {
        SearchNode::<H> {
            state,
            cost: parent.cost + cost_delta + 0.1,
            steps: parent.steps + 1,
            best_budget: parent.best_budget.clone(),
            history: parent.history.clone(),
            logs: parent.logs.clone(),
            next_op_id: parent.next_op_id,
        }
    }

    fn apply_crash(&self, state: &mut State<H>, node_id: NodeId) {
        state.crash_info.currently_crashed.insert(node_id);
        let tasks = std::mem::take(&mut state.runnable_tasks);
        for task in tasks {
            match task {
                Runnable::Timer(timer) => {
                    // Drop timers for crashed nodes
                    if timer.node != node_id {
                        state.runnable_tasks.push_back(Runnable::Timer(timer));
                    }
                }
                Runnable::Record(record) => {
                    if record.node == node_id {
                        if record.origin_node != record.node {
                            state
                                .crash_info
                                .queued_messages
                                .push_back((node_id, record));
                        }
                    } else {
                        state.runnable_tasks.push_back(Runnable::Record(record));
                    }
                }
                Runnable::ChannelSend { target, .. } => {
                    // If target is the crashed node, drop it.
                    // If target is another node, keep it.
                    if target != node_id {
                        state.runnable_tasks.push_back(task);
                    }
                }
            }
        }
    }

    fn apply_recover(
        &self,
        state: &mut State<H>,
        logger: &mut SearchLogger,
        coverage: &mut LocalCoverage,
        node_id: NodeId,
    ) {
        state.crash_info.currently_crashed.remove(&node_id);
        state.nodes[node_id.index] = Env::<H>::default();

        if let Some(init_fn) = self.program.get_func_by_name("Node.BASE_NODE_INIT") {
            let mut env = make_local_env(
                init_fn,
                vec![],
                &Env::<H>::default(),
                &state.nodes[node_id.index],
            );
            if let VarSlot::Local(self_idx, _) = SELF_SLOT {
                env.set(self_idx, Value::<H>::node(node_id));
            }
            let _ = exec_sync_on_node(
                state,
                logger,
                &self.program,
                &mut env,
                node_id,
                init_fn.entry,
                None,
                coverage,
            );
        }

        // Schedule Node.RecoverInit if exists
        if let Some(recover_fn) = self.program.get_func_by_name("Node.RecoverInit") {
            let server_role = node_id.role;
            let actuals = vec![
                Value::<H>::int(node_id.index as i64),
                Value::<H>::list(
                    (0..self.config.num_servers)
                        .map(|i| {
                            Value::<H>::node(NodeId {
                                role: server_role,
                                index: i,
                            })
                        })
                        .collect(),
                ),
            ];
            let env = make_local_env(
                recover_fn,
                actuals,
                &Env::<H>::default(),
                &state.nodes[node_id.index],
            );
            let record = Record {
                pc: recover_fn.entry,
                node: node_id,
                origin_node: node_id,
                continuation: Continuation::Recover,
                env,
                x: 0.0,
                policy: UpdatePolicy::Identity,
            };
            state.runnable_tasks.push_back(Runnable::Record(record));
        }

        // Replay queued messages
        let queued = std::mem::take(&mut state.crash_info.queued_messages);
        for (dest, record) in queued {
            if dest == node_id {
                state.runnable_tasks.push_back(Runnable::Record(record));
            } else {
                state.crash_info.queued_messages.push_back((dest, record));
            }
        }
    }

    fn schedule_client_write(
        &self,
        state: &mut State<H>,
        client_node_id: NodeId,
        op_id: i32,
        target_server: NodeId,
        key: &EcoString,
        val: &str,
    ) -> bool {
        if let Some(func) = self.program.get_func_by_name("ClientInterface.Write") {
            let actuals = vec![
                Value::<H>::node(target_server),
                Value::<H>::string(key.clone()),
                Value::<H>::string(EcoString::from(val)),
            ];
            let env = make_local_env(
                func,
                actuals,
                &Env::<H>::default(),
                &state.nodes[client_node_id.index],
            );
            let record = Record {
                pc: func.entry,
                node: client_node_id,
                origin_node: client_node_id,
                continuation: Continuation::ClientOp {
                    client_id: client_node_id.index as i32,
                    op_name: "Write".to_string(),
                    unique_id: op_id,
                },
                env,
                x: 0.5,
                policy: UpdatePolicy::Identity,
            };
            state.runnable_tasks.push_back(Runnable::Record(record));
            true
        } else {
            false
        }
    }

    fn schedule_client_read(
        &self,
        state: &mut State<H>,
        client_node_id: NodeId,
        op_id: i32,
        target_server: NodeId,
        key: &EcoString,
    ) -> bool {
        if let Some(func) = self.program.get_func_by_name("ClientInterface.Read") {
            let actuals = vec![
                Value::<H>::node(target_server),
                Value::<H>::string(key.clone()),
            ];
            let env = make_local_env(
                func,
                actuals,
                &Env::<H>::default(),
                &state.nodes[client_node_id.index],
            );
            let record = Record {
                pc: func.entry,
                node: client_node_id,
                origin_node: client_node_id,
                continuation: Continuation::ClientOp {
                    client_id: client_node_id.index as i32,
                    op_name: "Read".to_string(),
                    unique_id: op_id,
                },
                env,
                x: 0.5,
                policy: UpdatePolicy::Identity,
            };
            state.runnable_tasks.push_back(Runnable::Record(record));
            true
        } else {
            false
        }
    }
}
