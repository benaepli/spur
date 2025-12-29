use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::analysis::resolver::NameId;
use crate::compiler::cfg::Program;
use crate::simulator::core::{
    Continuation, Env, OpKind, Operation, Record, RuntimeError, SELF_NAME_ID, State, UpdatePolicy,
    Value, eval, exec, exec_sync_on_node, schedule_record,
};
use crate::simulator::plan::{ClientOpSpec, EventAction, ExecutionPlan, PlanEngine};
use log::{info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Full,
}

#[derive(Clone, Debug)]
pub struct TopologyInfo {
    pub topology: Topology,
    pub num_servers: i32,
}

fn create_env(
    node_env: &Rc<RefCell<Env>>,
    formals: &[NameId],
    actuals: Vec<Value>,
    locals: &[(NameId, crate::compiler::cfg::Expr)],
) -> Result<Env, RuntimeError> {
    assert_eq!(
        formals.len(),
        actuals.len(),
        "Argument count mismatch: expected {}, got {}",
        formals.len(),
        actuals.len()
    );

    let mut env: Env = formals.iter().copied().zip(actuals).collect();

    let node_env_ref = node_env.borrow();
    for (name, default_expr) in locals {
        env.insert(*name, eval(&env, &node_env_ref, default_expr)?);
    }

    Ok(env)
}

#[derive(Debug)]
struct RecoverContinuation;

impl Continuation for RecoverContinuation {
    fn call(&self, _state: &mut State, _val: Value) {}
}

#[derive(Debug)]
struct ClientOpContinuation {
    client_id: i32,
    op_name: String,
    unique_id: i32,
}

impl Continuation for ClientOpContinuation {
    fn call(&self, state: &mut State, value: Value) {
        state.free_clients.push(self.client_id);
        state.history.push(Operation {
            client_id: self.client_id,
            op_action: self.op_name.clone(),
            kind: OpKind::Response,
            payload: vec![value],
            unique_id: self.unique_id,
        });
    }
}

fn recover_node(
    topology: &TopologyInfo,
    state: &mut State,
    prog: &Program,
    node_id: usize,
) -> Result<(), RuntimeError> {
    let Some(recover_fn) = prog.get_func_by_name("Node.RecoverInit") else {
        return Ok(());
    };

    let actuals = match topology.topology {
        Topology::Full => vec![
            Value::Int(node_id as i64),
            Value::List(
                (0..topology.num_servers)
                    .map(|j| Value::Node(j as usize))
                    .collect(),
            ),
        ],
    };

    let mut env = Env::default();
    for (i, formal) in recover_fn.formals.iter().enumerate() {
        env.insert(*formal, actuals[i].clone());
    }
    let node_env = state.nodes[node_id].borrow();
    for (name, expr) in &recover_fn.locals {
        env.insert(*name, eval(&env, &node_env, expr)?);
    }
    drop(node_env);

    let record = Record {
        pc: recover_fn.entry,
        node: node_id,
        origin_node: node_id,
        continuation: Rc::new(RecoverContinuation),
        env,
        id: -1,
        x: 0.0,
        policy: UpdatePolicy::Identity,
    };

    exec(state, &prog, record)
}

fn reinit_node(
    topology: &TopologyInfo,
    state: &mut State,
    prog: &Program,
    node_id: usize,
) -> Result<(), RuntimeError> {
    let init_fn = prog
        .get_func_by_name("Node.BASE_NODE_INIT")
        .expect("BASE_NODE_INIT not found");

    let mut env = Env::default();
    env.insert(SELF_NAME_ID, Value::Node(node_id));
    let node_env = state.nodes[node_id].borrow();
    for (name, expr) in &init_fn.locals {
        env.insert(*name, eval(&env, &node_env, expr)?);
    }
    drop(node_env);

    exec_sync_on_node(state, prog, &mut env, node_id, init_fn.entry)?;

    recover_node(topology, state, prog, node_id)
}

fn crash_node(state: &mut State, node_id: usize) {
    let ci = &mut state.crash_info;

    if !ci.currently_crashed.insert(node_id) {
        warn!("Node {} is already crashed", node_id);
        return;
    }

    let (crashed, alive): (Vec<_>, Vec<_>) = state
        .runnable_records
        .drain(..)
        .partition(|r| r.node == node_id);

    state.runnable_records = alive;

    for record in crashed {
        let is_external = record.origin_node != record.node;
        let origin_alive = !ci.currently_crashed.contains(&record.origin_node);

        if is_external && origin_alive {
            ci.queued_messages.push((node_id, record));
        }
    }
}

fn recover_crashed_node(
    state: &mut State,
    prog: &Program,
    topology: &TopologyInfo,
    node_id: usize,
) -> Result<(), RuntimeError> {
    if !state.crash_info.currently_crashed.remove(&node_id) {
        warn!("Node {} is not crashed", node_id);
        return Ok(());
    }

    state.nodes[node_id] = Rc::new(RefCell::new(Env::default()));
    reinit_node(topology, state, prog, node_id)?;

    let mut queued_for_node = Vec::new();
    let mut remaining = Vec::new();
    for (dest, record) in std::mem::take(&mut state.crash_info.queued_messages) {
        if dest == node_id {
            queued_for_node.push(record);
        } else {
            remaining.push((dest, record));
        }
    }
    state.crash_info.queued_messages = remaining;
    state.runnable_records.extend(queued_for_node);
    Ok(())
}

fn schedule_client_op(
    state: &mut State,
    prog: &Program,
    op_id: i32,
    op_spec: &ClientOpSpec,
    client_id: i32,
) -> Result<(), RuntimeError> {
    let (op_name, actuals) = match op_spec {
        ClientOpSpec::Write(target, key, val) => (
            "ClientInterface.Write",
            vec![
                Value::Node(*target as usize),
                Value::String(key.clone()),
                Value::String(val.clone()),
            ],
        ),
        ClientOpSpec::Read(target, key) => (
            "ClientInterface.Read",
            vec![Value::Node(*target as usize), Value::String(key.clone())],
        ),
        ClientOpSpec::SimulateTimeout(target) => (
            "ClientInterface.SimulateTimeout",
            vec![Value::Node(*target as usize)],
        ),
    };

    let op_func = prog
        .get_func_by_name(op_name)
        .expect("Client op function not found");
    let env = create_env(
        &state.nodes[client_id as usize],
        &op_func.formals,
        actuals.clone(),
        &op_func.locals,
    )?;

    state.history.push(Operation {
        client_id,
        op_action: op_name.to_string(),
        kind: OpKind::Invocation,
        payload: actuals,
        unique_id: op_id,
    });

    state.runnable_records.push(Record {
        pc: op_func.entry,
        node: client_id as usize,
        origin_node: client_id as usize,
        continuation: Rc::new(ClientOpContinuation {
            client_id,
            op_name: op_name.to_string(),
            unique_id: op_id,
        }),
        env,
        id: op_id,
        x: 0.4,
        policy: UpdatePolicy::Identity,
    });
    Ok(())
}

pub fn exec_plan(
    state: &mut State,
    mut program: Program,
    plan: ExecutionPlan,
    max_iterations: i32,
    topology: TopologyInfo,
    randomly_delay_msgs: bool,
) -> Result<(), RuntimeError> {
    let mut engine = PlanEngine::new(plan);
    let mut op_id_counter = 0i32;
    let mut in_progress: HashMap<i32, String> = HashMap::new();

    for step in 0..max_iterations {
        if engine.is_complete() {
            info!("Plan completed in {} steps", step);
            return Ok(());
        }

        state.crash_info.current_step = step;

        // 1. Dispatch ready events
        for event in engine.get_ready_events().expect("Plan error") {
            match &event.action {
                EventAction::ClientRequest(op_spec) => {
                    if let Some(client_id) = state.free_clients.pop() {
                        op_id_counter += 1;
                        in_progress.insert(op_id_counter, event.id.clone());
                        schedule_client_op(state, &program, op_id_counter, op_spec, client_id)?;
                    } else {
                        engine
                            .mark_as_ready(&event.id)
                            .expect("Failed to defer event");
                    }
                }
                EventAction::CrashNode(node_id) => {
                    crash_node(state, *node_id as usize);
                    engine
                        .mark_event_completed(&event.id)
                        .expect("Failed to complete crash");
                }
                EventAction::RecoverNode(node_id) => {
                    recover_crashed_node(state, &program, &topology, *node_id as usize)?;
                    engine
                        .mark_event_completed(&event.id)
                        .expect("Failed to complete recover");
                }
            }
        }

        let history_start_len = state.history.len();

        // Execute one simulation step
        if !state.runnable_records.is_empty() {
            schedule_record(
                state,
                &program,
                false,
                false,
                false,
                &[],
                randomly_delay_msgs,
            )?;
        }

        // Only scan new history entries added during this step
        let completed: Vec<i32> = state.history[history_start_len..]
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Response))
            .filter_map(|op| {
                in_progress.get(&op.unique_id).map(|event_id| {
                    let _ = engine.mark_event_completed(event_id);
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
