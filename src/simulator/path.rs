use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Program, SELF_SLOT, VarSlot};
use crate::simulator::core::{
    Continuation, Env, LogEntry, Logger, NodeId, OpKind, Operation, Record, Runnable, RuntimeError,
    State, UpdatePolicy, Value, exec, exec_sync_on_node, make_local_env, schedule_runnable,
};
use crate::simulator::coverage::{GlobalState, LocalCoverage, VertexMap};
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::plan::{
    ClientOpSpec, EventAction, ExecutionPlan, PlanEngine, PlannedEvent,
};
use ecow::EcoString;
use log::{info, warn};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

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

/// Newtype wrapper for log entries that implements Logger.
#[derive(Debug, Default)]
pub struct Logs(pub Vec<LogEntry>);

impl Logger for Logs {
    fn log(&mut self, entry: LogEntry) {
        self.0.push(entry);
    }
}

/// Wrapper around State that adds path-execution tracking fields.
#[derive(Debug)]
pub struct PathState<H: HashPolicy> {
    pub state: State<H>,
    pub coverage: LocalCoverage,
    pub logs: Logs,
    pub history: Vec<Operation<H>>,
    pub free_clients: Vec<i32>,
}

impl<H: HashPolicy> PathState<H> {
    pub fn new(role_node_counts: &[(NameId, usize)], node_slot_count: usize) -> Self {
        Self {
            state: State::<H>::new(role_node_counts, node_slot_count),
            coverage: LocalCoverage::new(),
            logs: Logs::default(),
            history: Vec::new(),
            free_clients: Vec::new(),
        }
    }
}

fn recover_node<H: HashPolicy, L: Logger>(
    topology: &TopologyInfo,
    state: &mut State<H>,
    logger: &mut L,
    prog: &Program,
    node_id: NodeId,
    _global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<(), RuntimeError> {
    let Some(recover_fn) = prog.get_func_by_name("Node.RecoverInit") else {
        return Ok(());
    };

    let actuals = match topology.topology {
        Topology::Full => vec![
            Value::<H>::int(node_id.index as i64),
            Value::<H>::list(
                (0..topology.num_servers)
                    .map(|j| {
                        Value::<H>::node(NodeId {
                            role: node_id.role,
                            index: j as usize,
                        })
                    })
                    .collect(),
            ),
        ],
    };

    let node_env = &state.nodes[node_id.index];
    let env = make_local_env(
        recover_fn,
        actuals,
        &Env::<H>::default(),
        node_env,
        &prog.id_to_name,
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

    exec(state, logger, prog, record, global_snapshot, local_coverage)?;
    Ok(())
}

fn reinit_node<H: HashPolicy, L: Logger>(
    topology: &TopologyInfo,
    state: &mut State<H>,
    logger: &mut L,
    prog: &Program,
    node_id: NodeId,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<(), RuntimeError> {
    let init_fn = prog
        .get_func_by_name("Node.BASE_NODE_INIT")
        .ok_or_else(|| RuntimeError::MissingRequiredFunction("Node.BASE_NODE_INIT".to_string()))?;

    let node_env = &state.nodes[node_id.index];
    let mut env = make_local_env(
        init_fn,
        vec![],
        &Env::<H>::default(),
        node_env,
        &prog.id_to_name,
    );
    if let VarSlot::Node(self_idx, _) = SELF_SLOT {
        env.set(self_idx, Value::<H>::node(node_id));
    }

    exec_sync_on_node(
        state,
        logger,
        prog,
        &mut env,
        node_id,
        init_fn.entry,
        global_snapshot,
        local_coverage,
    )?;

    recover_node(
        topology,
        state,
        logger,
        prog,
        node_id,
        global_state,
        global_snapshot,
        local_coverage,
    )
}

fn crash_node<H: HashPolicy>(
    state: &mut State<H>,
    history: &mut Vec<Operation<H>>,
    node_id: NodeId,
) {
    if state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is already crashed", node_id);
        return;
    }
    state.crash_info.currently_crashed.insert(node_id);

    history.push(Operation {
        client_id: -1,
        op_action: "System.Crash".to_string(),
        kind: OpKind::Crash,
        payload: vec![Value::<H>::node(node_id)],
        unique_id: -1,
    });

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
                    let is_external = record.origin_node != record.node;
                    let origin_alive = !state
                        .crash_info
                        .currently_crashed
                        .contains(&record.origin_node);
                    if is_external && origin_alive {
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

fn recover_crashed_node<H: HashPolicy, L: Logger>(
    state: &mut State<H>,
    logger: &mut L,
    history: &mut Vec<Operation<H>>,
    prog: &Program,
    topology: &TopologyInfo,
    node_id: NodeId,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<(), RuntimeError> {
    if !state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is not crashed", node_id);
        return Ok(());
    }
    state.crash_info.currently_crashed.remove(&node_id);

    history.push(Operation {
        client_id: -1,
        op_action: "System.Recover".to_string(),
        kind: OpKind::Recover,
        payload: vec![Value::<H>::node(node_id)],
        unique_id: -1,
    });

    state.nodes[node_id.index] = Env::<H>::default();
    reinit_node(
        topology,
        state,
        logger,
        prog,
        node_id,
        global_state,
        global_snapshot,
        local_coverage,
    )?;

    let queued = std::mem::take(&mut state.crash_info.queued_messages);
    for (dest, record) in queued {
        if dest == node_id {
            state.runnable_tasks.push_back(Runnable::Record(record));
        } else {
            state.crash_info.queued_messages.push_back((dest, record));
        }
    }
    Ok(())
}

fn schedule_client_op<H: HashPolicy>(
    state: &mut State<H>,
    history: &mut Vec<Operation<H>>,
    prog: &Program,
    op_id: i32,
    op_spec: &ClientOpSpec,
    client_node_id: NodeId,
    server_role: NameId,
) -> Result<(), RuntimeError> {
    let client_id = client_node_id.index as i32;
    let (op_name, actuals) = match op_spec {
        ClientOpSpec::Write(target, key, val) => (
            "ClientInterface.Write",
            vec![
                Value::<H>::node(NodeId {
                    role: server_role,
                    index: *target as usize,
                }),
                Value::<H>::string(EcoString::from(key.as_str())),
                Value::<H>::string(EcoString::from(val.as_str())),
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
        ClientOpSpec::SimulateTimeout(target) => (
            "ClientInterface.SimulateTimeout",
            vec![Value::<H>::node(NodeId {
                role: server_role,
                index: *target as usize,
            })],
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
    });

    state.runnable_tasks.push_back(Runnable::Record(Record {
        pc: op_func.entry,
        node: client_node_id,
        origin_node: client_node_id,
        continuation: Continuation::ClientOp {
            client_id,
            op_name: op_name.to_string(),
            unique_id: op_id,
        },
        env,
        x: 0.4,
        policy: UpdatePolicy::Identity,
    }));
    Ok(())
}

pub fn exec_plan<H: HashPolicy>(
    path_state: &mut PathState<H>,
    program: Program,
    plan: ExecutionPlan,
    max_iterations: i32,
    topology: TopologyInfo,
    randomly_delay_msgs: bool,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    run_id: i64,
) -> Result<(), RuntimeError> {
    let mut engine = PlanEngine::new(plan);
    let mut op_id_counter = 0i32;
    let mut in_progress: HashMap<i32, NodeIndex> = HashMap::new();

    // Look up role NameIds from the program
    let server_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "Node")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("Node".to_string()))?;
    let client_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "ClientInterface")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("ClientInterface".to_string()))?;

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

        // Dispatch ready events
        let ready_events: Vec<(NodeIndex, PlannedEvent)> = engine
            .get_ready_events()
            .into_iter()
            .map(|(idx, e)| (idx, e.clone()))
            .collect();

        for (node_idx, event) in ready_events {
            match &event.action {
                EventAction::ClientRequest(op_spec) => {
                    if let Some(client_id) = path_state.free_clients.pop() {
                        op_id_counter += 1;
                        in_progress.insert(op_id_counter, node_idx);

                        // Validate client node
                        let client_node_id = validate_node(
                            &path_state.state,
                            client_id as usize,
                            client_role,
                            "ClientInterface",
                        )?;

                        // Validate target server in op_spec
                        let target_idx = match op_spec {
                            ClientOpSpec::Write(t, _, _) => *t as usize,
                            ClientOpSpec::Read(t, _) => *t as usize,
                            ClientOpSpec::SimulateTimeout(t) => *t as usize,
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
                        )?;
                    } else {
                        if let Err(e) = engine.mark_as_ready(node_idx) {
                            warn!("Failed to mark event as ready: {}", e);
                        }
                    }
                }
                EventAction::CrashNode(node_id) => {
                    let nid =
                        validate_node(&path_state.state, *node_id as usize, server_role, "Node")?;
                    crash_node(&mut path_state.state, &mut path_state.history, nid);
                    engine.mark_event_completed(node_idx);
                }
                EventAction::RecoverNode(node_id) => {
                    let nid =
                        validate_node(&path_state.state, *node_id as usize, server_role, "Node")?;
                    recover_crashed_node(
                        &mut path_state.state,
                        &mut path_state.logs,
                        &mut path_state.history,
                        &program,
                        &topology,
                        nid,
                        global_state,
                        global_snapshot,
                        &mut path_state.coverage,
                    )?;
                    engine.mark_event_completed(node_idx);
                }
            }
        }

        let history_start_len = path_state.history.len();

        // Execute one simulation step
        if !path_state.state.runnable_tasks.is_empty() {
            let result = schedule_runnable(
                &mut path_state.state,
                &mut path_state.logs,
                &program,
                false,
                false,
                false,
                &[],
                randomly_delay_msgs,
                global_snapshot,
                &mut path_state.coverage,
            )?;

            if let Some(result) = result {
                path_state.free_clients.push(result.client_id);
                path_state.history.push(Operation {
                    client_id: result.client_id,
                    op_action: result.op_name,
                    kind: OpKind::Response,
                    payload: vec![result.value],
                    unique_id: result.unique_id,
                });
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
    }

    warn!(
        "Hit max iterations ({}) before plan completion",
        max_iterations
    );
    Ok(())
}
