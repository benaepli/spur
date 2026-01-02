use std::collections::HashMap;

use crate::compiler::cfg::{Program, SELF_SLOT, VarSlot};
use crate::simulator::core::{
    Continuation, Env, LogEntry, Logger, OpKind, Operation, Record, RuntimeError, State,
    UpdatePolicy, Value, exec, exec_sync_on_node, make_local_env, schedule_record,
};
use crate::simulator::coverage::{GlobalState, LocalCoverage, VertexMap};
use crate::simulator::path::plan::{
    ClientOpSpec, EventAction, ExecutionPlan, PlanEngine, PlannedEvent,
};
use ecow::EcoString;
use log::{info, warn};
use petgraph::graph::NodeIndex;

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
pub struct PathState {
    pub state: State,
    pub coverage: LocalCoverage,
    pub logs: Logs,
    pub history: Vec<Operation>,
    pub free_clients: Vec<i32>,
}

impl PathState {
    pub fn new(node_count: usize, node_slot_count: usize) -> Self {
        Self {
            state: State::new(node_count, node_slot_count),
            coverage: LocalCoverage::new(),
            logs: Logs::default(),
            history: Vec::new(),
            free_clients: Vec::new(),
        }
    }
}

fn recover_node<L: Logger>(
    topology: &TopologyInfo,
    state: &mut State,
    logger: &mut L,
    prog: &Program,
    node_id: usize,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<(), RuntimeError> {
    let Some(recover_fn) = prog.get_func_by_name("Node.RecoverInit") else {
        return Ok(());
    };

    let actuals = match topology.topology {
        Topology::Full => vec![
            Value::int(node_id as i64),
            Value::list(
                (0..topology.num_servers)
                    .map(|j| Value::node(j as usize))
                    .collect(),
            ),
        ],
    };

    let node_env = &state.nodes[node_id];
    let env = make_local_env(recover_fn, actuals, &Env::default(), node_env);

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

fn reinit_node<L: Logger>(
    topology: &TopologyInfo,
    state: &mut State,
    logger: &mut L,
    prog: &Program,
    node_id: usize,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<(), RuntimeError> {
    let init_fn = prog
        .get_func_by_name("Node.BASE_NODE_INIT")
        .ok_or_else(|| RuntimeError::MissingRequiredFunction("Node.BASE_NODE_INIT".to_string()))?;

    let node_env = &state.nodes[node_id];
    let mut env = make_local_env(init_fn, vec![], &Env::default(), node_env);
    if let VarSlot::Local(self_idx, _) = SELF_SLOT {
        env.set(self_idx, Value::node(node_id));
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

fn crash_node(state: &mut State, history: &mut Vec<Operation>, node_id: usize) {
    if state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is already crashed", node_id);
        return;
    }
    state.crash_info.currently_crashed.insert(node_id);

    history.push(Operation {
        client_id: -1,
        op_action: "System.Crash".to_string(),
        kind: OpKind::Crash,
        payload: vec![Value::node(node_id)],
        unique_id: -1,
    });

    let records = std::mem::take(&mut state.runnable_records);
    for record in records {
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
            state.runnable_records.push_back(record);
        }
    }
}

fn recover_crashed_node<L: Logger>(
    state: &mut State,
    logger: &mut L,
    history: &mut Vec<Operation>,
    prog: &Program,
    topology: &TopologyInfo,
    node_id: usize,
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
        payload: vec![Value::node(node_id)],
        unique_id: -1,
    });

    state.nodes[node_id] = Env::default();
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
            state.runnable_records.push_back(record);
        } else {
            state.crash_info.queued_messages.push_back((dest, record));
        }
    }
    Ok(())
}

fn schedule_client_op(
    state: &mut State,
    history: &mut Vec<Operation>,
    prog: &Program,
    op_id: i32,
    op_spec: &ClientOpSpec,
    client_id: i32,
) -> Result<(), RuntimeError> {
    let (op_name, actuals) = match op_spec {
        ClientOpSpec::Write(target, key, val) => (
            "ClientInterface.Write",
            vec![
                Value::node(*target as usize),
                Value::string(EcoString::from(key.as_str())),
                Value::string(EcoString::from(val.as_str())),
            ],
        ),
        ClientOpSpec::Read(target, key) => (
            "ClientInterface.Read",
            vec![
                Value::node(*target as usize),
                Value::string(EcoString::from(key.as_str())),
            ],
        ),
        ClientOpSpec::SimulateTimeout(target) => (
            "ClientInterface.SimulateTimeout",
            vec![Value::node(*target as usize)],
        ),
    };

    let op_func = prog
        .get_func_by_name(op_name)
        .ok_or_else(|| RuntimeError::MissingRequiredFunction(op_name.to_string()))?;
    let env = make_local_env(
        op_func,
        actuals.clone(),
        &Env::default(),
        &state.nodes[client_id as usize],
    );

    history.push(Operation {
        client_id,
        op_action: op_name.to_string(),
        kind: OpKind::Invocation,
        payload: actuals,
        unique_id: op_id,
    });

    state.runnable_records.push_back(Record {
        pc: op_func.entry,
        node: client_id as usize,
        origin_node: client_id as usize,
        continuation: Continuation::ClientOp {
            client_id,
            op_name: op_name.to_string(),
            unique_id: op_id,
        },
        env,
        x: 0.4,
        policy: UpdatePolicy::Identity,
    });
    Ok(())
}

pub fn exec_plan(
    path_state: &mut PathState,
    program: Program,
    plan: ExecutionPlan,
    max_iterations: i32,
    topology: TopologyInfo,
    randomly_delay_msgs: bool,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
) -> Result<(), RuntimeError> {
    let mut engine = PlanEngine::new(plan);
    let mut op_id_counter = 0i32;
    let mut in_progress: HashMap<i32, NodeIndex> = HashMap::new();

    for step in 0..max_iterations {
        if engine.is_complete() {
            info!("Plan completed in {} steps", step);
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
                        schedule_client_op(
                            &mut path_state.state,
                            &mut path_state.history,
                            &program,
                            op_id_counter,
                            op_spec,
                            client_id,
                        )?;
                    } else {
                        if let Err(e) = engine.mark_as_ready(node_idx) {
                            warn!("Failed to mark event as ready: {}", e);
                        }
                    }
                }
                EventAction::CrashNode(node_id) => {
                    crash_node(
                        &mut path_state.state,
                        &mut path_state.history,
                        *node_id as usize,
                    );
                    engine.mark_event_completed(node_idx);
                }
                EventAction::RecoverNode(node_id) => {
                    recover_crashed_node(
                        &mut path_state.state,
                        &mut path_state.logs,
                        &mut path_state.history,
                        &program,
                        &topology,
                        *node_id as usize,
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
        if !path_state.state.runnable_records.is_empty() {
            let result = schedule_record(
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
