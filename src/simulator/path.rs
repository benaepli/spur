use crate::analysis::resolver::NameId;
use crate::compiler::cfg::Program;
use crate::simulator::core::{
    Continuation, Env, LogEntry, Logger, NodeId, OpKind, Operation, Record, Runnable, RuntimeError,
    ScheduleResult, State, UpdatePolicy, Value, make_local_env, schedule_runnable,
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
        entry_pc: op_func.entry,
        initial_env: env.clone(),
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
    // Map from node_id index to the plan engine NodeIndex for pending crash/recover events
    let mut pending_crash_recover: HashMap<usize, NodeIndex> = HashMap::new();

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
                    path_state
                        .state
                        .runnable_tasks
                        .push_back(Runnable::Crash { node_id: nid });
                    pending_crash_recover.insert(nid.index, node_idx);
                }
                EventAction::RecoverNode(node_id) => {
                    let nid =
                        validate_node(&path_state.state, *node_id as usize, server_role, "Node")?;
                    path_state
                        .state
                        .runnable_tasks
                        .push_back(Runnable::Recover { node_id: nid });
                    pending_crash_recover.insert(nid.index, node_idx);
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
                &topology,
                global_state,
            )?;

            match result {
                ScheduleResult::None => {}
                ScheduleResult::ClientOp(result) => {
                    path_state.free_clients.push(result.client_id);
                    path_state.history.push(Operation {
                        client_id: result.client_id,
                        op_action: result.op_name,
                        kind: OpKind::Response,
                        payload: vec![result.value],
                        unique_id: result.unique_id,
                    });
                }
                ScheduleResult::Crash { node_id } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Crash".to_string(),
                        kind: OpKind::Crash,
                        payload: vec![Value::<H>::node(node_id)],
                        unique_id: -1,
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
                    });
                    if let Some(plan_node) = pending_crash_recover.remove(&node_id.index) {
                        engine.mark_event_completed(plan_node);
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
    }

    warn!(
        "Hit max iterations ({}) before plan {} completion",
        max_iterations, run_id
    );
    Ok(())
}
