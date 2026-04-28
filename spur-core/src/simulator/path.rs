use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::{
    Continuation, Env, LogEntry, Logger, NodeId, OpKind, Operation, PurgatoryConfig,
    QueuePolicyConfig, QueueSelector, Record, Reservation, Runnable, RunnableCategory,
    RuntimeError, SchedulePolicy, ScheduleResult, State, TraceEntry, Value, WithinQueueSelector,
    make_local_env, schedule_runnable,
};
use crate::simulator::coverage::{GlobalState, LocalCoverage, VertexMap};
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::plan::{
    ClientOpSpec, DeliverSpec, EventAction, ExecutionPlan, PlanEngine, PlannedEvent,
};
use ecow::EcoString;
use log::{info, warn};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

pub mod generator;
pub mod plan;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Full,
}

#[derive(Clone, Debug)]
pub struct TopologyInfo {
    pub topology: Topology,
    pub num_servers: i32,
}

/// Newtype wrapper for log and trace entries that implements Logger.
#[derive(Debug, Default)]
pub struct Logs {
    pub entries: Vec<LogEntry>,
    pub traces: Vec<TraceEntry>,
}

impl Logger for Logs {
    fn log(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }
    fn log_trace(&mut self, entry: TraceEntry) {
        self.traces.push(entry);
    }
}

/// A pool that dynamically creates client nodes on demand and recycles them.
#[derive(Debug)]
pub struct ClientPool {
    free_clients: Vec<NodeId>,
    client_role: NameId,
    node_slot_count: usize,
}

impl ClientPool {
    pub fn new(client_role: NameId, node_slot_count: usize) -> Self {
        Self {
            free_clients: Vec::new(),
            client_role,
            node_slot_count,
        }
    }

    /// Get a client node — reuses a free one or creates a new one.
    /// Returns (NodeId, bool) where the boolean is true if the node was newly created.
    pub fn get<H: HashPolicy>(&mut self, state: &mut State<H>) -> (NodeId, bool) {
        if let Some(node_id) = self.free_clients.pop() {
            (node_id, false)
        } else {
            (state.add_node(self.client_role, self.node_slot_count), true)
        }
    }

    /// Return a client node to the pool for reuse.
    pub fn release(&mut self, node_id: NodeId) {
        self.free_clients.push(node_id);
    }
}

/// Wrapper around State that adds path-execution tracking fields.
#[derive(Debug)]
pub struct PathState<H: HashPolicy> {
    pub state: State<H>,
    pub coverage: LocalCoverage,
    pub logs: Logs,
    pub history: Vec<Operation<H>>,
    pub client_pool: ClientPool,
}

impl<H: HashPolicy> PathState<H> {
    pub fn new(
        role_node_counts: &[(NameId, usize)],
        node_slot_count: usize,
        client_role: NameId,
    ) -> Self {
        Self {
            state: State::<H>::new(role_node_counts, node_slot_count),
            coverage: LocalCoverage::new(),
            logs: Logs::default(),
            history: Vec::new(),
            client_pool: ClientPool::new(client_role, node_slot_count),
        }
    }
}

fn schedule_client_op<H: HashPolicy>(
    state: &mut State<H>,
    history: &mut Vec<Operation<H>>,
    prog: &Program,
    op_id: i32,
    op_spec: &ClientOpSpec,
    client_node_id: NodeId,
    server_role: NameId,
    policy: &SchedulePolicy,
) -> Result<(), RuntimeError> {
    let client_id = client_node_id.index as i32;
    let (op_name, actuals) = match op_spec {
        ClientOpSpec::Write(target, key) => (
            "ClientInterface.Write",
            vec![
                Value::<H>::node(NodeId {
                    role: server_role,
                    index: *target as usize,
                }),
                Value::<H>::string(EcoString::from(key.as_str())),
                Value::<H>::int(op_id as i64),
            ],
        ),
        ClientOpSpec::Read(target, key) => (
            "ClientInterface.Read",
            vec![
                Value::<H>::node(NodeId {
                    role: server_role,
                    index: *target as usize,
                }),
                Value::<H>::string(EcoString::from(key.as_str())),
            ],
        ),
    };

    let op_func = prog
        .get_func_by_name(op_name)
        .ok_or_else(|| RuntimeError::MissingRequiredFunction(op_name.to_string()))?;
    let env = make_local_env(
        op_func,
        actuals.clone(),
        &Env::default(),
        &state.nodes[client_node_id.index],
        &prog.id_to_name,
    );

    history.push(Operation {
        client_id,
        op_action: op_name.to_string(),
        kind: OpKind::Invocation,
        payload: actuals,
        unique_id: op_id,
        step: state.crash_info.current_step,
    });

    let mut rng = rand::rng();
    state.push_runnable(Runnable::Record(Record {
        pc: op_func.entry,
        node: client_node_id,
        origin_node: client_node_id,
        continuation: Continuation::ClientOp {
            client_id,
            op_name: op_name.to_string(),
            unique_id: op_id,
        },
        entry_pc: op_func.entry,
        initial_env: env.clone(),
        env,
        priority: policy.sample(&mut rng, RunnableCategory::Record),
        causal_operation_id: Some(op_id),
        trace_id: None,
        link_seq: None,
    }));
    Ok(())
}

pub fn exec_plan<H: HashPolicy>(
    path_state: &mut PathState<H>,
    program: Program,
    plan: ExecutionPlan,
    max_iterations: i32,
    topology: TopologyInfo,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    run_id: i64,
    policy: &SchedulePolicy,
    strict_timers: bool,
    queue_policy: &QueuePolicyConfig,
    within_queue: &WithinQueueSelector,
    quick_fire_multiplier: f64,
    purgatory_config: &PurgatoryConfig,
) -> Result<(), RuntimeError> {
    let mut selector = queue_policy.into_selector();
    let mut op_id_counter = 0i32;
    let mut in_progress: HashMap<i32, NodeIndex> = HashMap::new();
    // Map from node_id index to the plan engine NodeIndex for pending crash/recover events
    let mut pending_crash_recover: HashMap<usize, NodeIndex> = HashMap::new();
    // Map from (node_index, label) to the plan engine NodeIndex for pending AllowTimer events
    let mut pending_allow_timer: HashMap<(usize, String), NodeIndex> = HashMap::new();
    let mut pending_partition: Option<NodeIndex> = None;
    let mut pending_heal: Option<NodeIndex> = None;

    // Build name-to-entry-pc map for resolving deliver specs.
    // Resolution happens once upfront, not per scheduler call.
    let name_to_entry: HashMap<&str, Vertex> = program
        .func_name_to_id
        .iter()
        .filter_map(|(name, name_id)| {
            program.rpc.get(name_id).map(|fi| (name.as_str(), fi.entry))
        })
        .collect();

    // Reverse map for matching RecordExecuted results back to function names
    let entry_to_name: HashMap<Vertex, &str> = name_to_entry
        .iter()
        .map(|(&name, &entry)| (entry, name))
        .collect();

    // Collect all deliver events from the plan DAG before PlanEngine::new consumes it.
    let all_delivers: HashMap<NodeIndex, DeliverSpec> = plan
        .node_indices()
        .filter_map(|idx| match &plan[idx].action {
            EventAction::Deliver(spec) => Some((idx, spec.clone())),
            _ => None,
        })
        .collect();

    // Track deliver states: ready (unlocked) vs completed
    let mut ready_delivers: HashSet<NodeIndex> = HashSet::new();
    let mut completed_delivers: HashSet<NodeIndex> = HashSet::new();

    let mut engine = PlanEngine::new(plan);

    // Starvation detection: track consecutive no-progress iterations
    let mut no_progress_count: i32 = 0;
    const STARVATION_WARN_THRESHOLD: i32 = 500;

    // Look up role NameIds from the program
    let server_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "Node")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("Node".to_string()))?;

    let validate_node = |state: &State<H>,
                         index: usize,
                         expected_role: NameId,
                         _role_name: &str|
     -> Result<NodeId, RuntimeError> {
        if index >= state.nodes.len() {
            return Err(RuntimeError::IndexOutOfBounds {
                index,
                len: state.nodes.len(),
            });
        }
        let node_val = state.nodes[index].get(0);
        let node_id = node_val.as_node()?;
        if node_id.role != expected_role {
            return Err(RuntimeError::TypeError {
                expected: "node with correct role",
                got: "node with incorrect role",
            });
        }
        Ok(node_id)
    };

    for step in 0..max_iterations {
        if engine.is_complete() {
            info!("Plan {} completed in {} steps", run_id, step);
            return Ok(());
        }

        path_state.state.crash_info.current_step = step;

        // Release delayed messages whose time has come
        path_state.state.release_from_purgatory(step);

        // Dispatch ready events
        let ready_events: Vec<(NodeIndex, PlannedEvent)> = engine
            .get_ready_events()
            .into_iter()
            .map(|(idx, e)| (idx, e.clone()))
            .collect();

        for (node_idx, event) in ready_events {
            match &event.action {
                EventAction::ClientRequest(op_spec) => {
                    op_id_counter += 1;
                    in_progress.insert(op_id_counter, node_idx);

                    // Get a client node from the pool (creates one if needed)
                    let (client_node_id, is_new) =
                        path_state.client_pool.get(&mut path_state.state);

                    if is_new {
                        if let Some(init_fn) =
                            program.get_func_by_name("ClientInterface.BASE_NODE_INIT")
                        {
                            let mut env = make_local_env(
                                init_fn,
                                vec![],
                                &Env::<H>::default(),
                                &path_state.state.nodes[client_node_id.index],
                                &program.id_to_name,
                            );
                            if let Err(e) = crate::simulator::core::exec_sync_on_node(
                                &mut path_state.state,
                                &mut path_state.logs,
                                &program,
                                &mut env,
                                client_node_id,
                                init_fn.entry,
                                global_snapshot,
                                &mut path_state.coverage,
                                policy,
                                purgatory_config,
                            ) {
                                log::warn!(
                                    "Failed to initialize dynamic client node {}: {}",
                                    client_node_id,
                                    e
                                );
                            }
                        }
                    }

                    // Validate target server in op_spec
                    let target_idx = match op_spec {
                        ClientOpSpec::Write(t, _) => *t as usize,
                        ClientOpSpec::Read(t, _) => *t as usize,
                    };
                    validate_node(&path_state.state, target_idx, server_role, "Node")?;

                    schedule_client_op(
                        &mut path_state.state,
                        &mut path_state.history,
                        &program,
                        op_id_counter,
                        op_spec,
                        client_node_id,
                        server_role,
                        policy,
                    )?;
                }
                EventAction::CrashNode(node_id) => {
                    let nid =
                        validate_node(&path_state.state, *node_id as usize, server_role, "Node")?;
                    let mut rng = rand::rng();
                    path_state.state.push_runnable(Runnable::Crash {
                        node_id: nid,
                        priority: policy.sample(&mut rng, RunnableCategory::Crash),
                    });
                    pending_crash_recover.insert(nid.index, node_idx);
                }
                EventAction::RecoverNode(node_id) => {
                    let nid =
                        validate_node(&path_state.state, *node_id as usize, server_role, "Node")?;
                    let mut rng = rand::rng();
                    path_state.state.push_runnable(Runnable::Recover {
                        node_id: nid,
                        priority: policy.sample(&mut rng, RunnableCategory::Recover),
                    });
                    pending_crash_recover.insert(nid.index, node_idx);
                }
                EventAction::AllowTimer(node_id, label) => {
                    let key = (*node_id as usize, label.clone());
                    path_state.state.allowed_timers.insert(key.clone());
                    pending_allow_timer.insert(key, node_idx);
                }
                EventAction::Partition(spec) => {
                    let partition_type = spec.to_partition_type(
                        server_role,
                        topology.num_servers,
                    );
                    let mut rng = rand::rng();
                    path_state.state.push_runnable(Runnable::Partition {
                        partition_type,
                        priority: policy.sample(&mut rng, RunnableCategory::Partition),
                    });
                    pending_partition = Some(node_idx);
                }
                EventAction::Heal => {
                    let mut rng = rand::rng();
                    path_state.state.push_runnable(Runnable::Heal {
                        priority: policy.sample(&mut rng, RunnableCategory::Heal),
                    });
                    pending_heal = Some(node_idx);
                }
                EventAction::Deliver(_) => {
                    // Deliver events are constraints, not actions.
                    // When ready, lift the reservation so the scheduler can pick the match.
                    ready_delivers.insert(node_idx);
                }
            }
        }

        // Build reservations from delivers that are NOT yet ready and NOT completed.
        // These constrain the scheduler from picking their matching runnables early.
        let reservations: Vec<Reservation> = all_delivers
            .iter()
            .filter(|(idx, _)| !ready_delivers.contains(idx) && !completed_delivers.contains(idx))
            .filter_map(|(_, spec)| {
                name_to_entry.get(spec.function.as_str()).map(|&entry_pc| Reservation {
                    entry_pc,
                    from: spec.from.map(|f| f as usize),
                    to: spec.to.map(|t| t as usize),
                })
            })
            .collect();

        let history_start_len = path_state.history.len();

        if !path_state.state.all_queues_empty() {
            let result = schedule_runnable(
                &mut path_state.state,
                &mut path_state.logs,
                &program,
                false,
                global_snapshot,
                &mut path_state.coverage,
                &topology,
                global_state,
                policy,
                strict_timers,
                &mut selector,
                within_queue,
                quick_fire_multiplier,
                purgatory_config,
                &reservations,
            )?;

            match result {
                ScheduleResult::None => {}
                ScheduleResult::ClientOp(result) => {
                    path_state.client_pool.release(NodeId {
                        role: path_state.client_pool.client_role,
                        index: result.client_id as usize,
                    });
                    path_state.history.push(Operation {
                        client_id: result.client_id,
                        op_action: result.op_name,
                        kind: OpKind::Response,
                        payload: vec![result.value],
                        unique_id: result.unique_id,
                        step: path_state.state.crash_info.current_step,
                    });
                }
                ScheduleResult::Crash { node_id } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Crash".to_string(),
                        kind: OpKind::Crash,
                        payload: vec![Value::<H>::node(node_id)],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    if let Some(plan_node) = pending_crash_recover.remove(&node_id.index) {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::Recover { node_id } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Recover".to_string(),
                        kind: OpKind::Recover,
                        payload: vec![Value::<H>::node(node_id)],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    if let Some(plan_node) = pending_crash_recover.remove(&node_id.index) {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::TimerFired { node_id, label } => {
                    let key = (node_id.index, label);
                    if let Some(plan_node) = pending_allow_timer.remove(&key) {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::Partition { partition_type: _ } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Partition".to_string(),
                        kind: OpKind::Partition,
                        payload: vec![],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    if let Some(plan_node) = pending_partition.take() {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::Heal => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Heal".to_string(),
                        kind: OpKind::Heal,
                        payload: vec![],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    if let Some(plan_node) = pending_heal.take() {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::RecordExecuted {
                    entry_pc,
                    origin_node,
                    dest_node,
                } => {
                    // Check if this record delivery matches any ready deliver event.
                    if let Some(&func_name) = entry_to_name.get(&entry_pc) {
                        let matched = ready_delivers
                            .iter()
                            .find(|idx| {
                                if let Some(spec) = all_delivers.get(idx) {
                                    spec.function == func_name
                                        && spec
                                            .to
                                            .map_or(true, |t| dest_node.index == t as usize)
                                        && spec
                                            .from
                                            .map_or(true, |f| origin_node.index == f as usize)
                                } else {
                                    false
                                }
                            })
                            .copied();

                        if let Some(idx) = matched {
                            ready_delivers.remove(&idx);
                            completed_delivers.insert(idx);
                            engine.mark_event_completed(idx);
                        }
                    }
                }
            }
        }

        // Only scan new history entries added during this step
        let completed: Vec<i32> = path_state.history[history_start_len..]
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Response))
            .filter_map(|op| {
                in_progress.get(&op.unique_id).map(|&node_idx| {
                    engine.mark_event_completed(node_idx);
                    op.unique_id
                })
            })
            .collect();

        for id in completed {
            in_progress.remove(&id);
        }

        // Starvation detection: if nothing happened this iteration, increment counter.
        // Helps catch typos in deliver function names.
        if path_state.history.len() == history_start_len {
            no_progress_count += 1;
            if no_progress_count == STARVATION_WARN_THRESHOLD {
                let pending_deliver_names: Vec<&str> = ready_delivers
                    .iter()
                    .filter_map(|idx| all_delivers.get(idx).map(|s| s.function.as_str()))
                    .collect();
                let blocked_deliver_names: Vec<&str> = all_delivers
                    .iter()
                    .filter(|(idx, _)| {
                        !ready_delivers.contains(idx) && !completed_delivers.contains(idx)
                    })
                    .map(|(_, s)| s.function.as_str())
                    .collect();
                warn!(
                    "Plan {} stalled for {} iterations. Ready delivers waiting: {:?}. Blocked delivers: {:?}",
                    run_id, no_progress_count, pending_deliver_names, blocked_deliver_names
                );
            }
        } else {
            no_progress_count = 0;
        }
    }

    warn!(
        "Hit max iterations ({}) before plan {} completion",
        max_iterations, run_id
    );
    Ok(())
}
