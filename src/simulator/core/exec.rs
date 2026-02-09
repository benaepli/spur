use crate::compiler::cfg::{Instr, Label, Lhs, Program, VarSlot};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::{eval, make_local_env, store};
use crate::simulator::core::state::{
    ChannelState, ClientOpResult, Continuation, LogEntry, Logger, NodeId, Record, Runnable, State,
    Timer, UpdatePolicy,
};
use crate::simulator::core::values::{ChannelId, Env, Value, ValueKind};
use crate::simulator::coverage::{LocalCoverage, VertexMap};
use crate::simulator::hash_utils::HashPolicy;

pub fn exec_sync_on_node<H: HashPolicy, L: Logger>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_id: NodeId,
    start_pc: usize,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Value<H>, RuntimeError> {
    let mut node_env = state.nodes[node_id.index].clone();
    let result = exec_sync_inner(
        state,
        logger,
        program,
        local_env,
        &mut node_env,
        start_pc,
        node_id,
        global_snapshot,
        local_coverage,
    );
    state.nodes[node_id.index] = node_env;
    result
}

enum StepOutcome<H: HashPolicy> {
    Continue(usize),
    Return(Value<H>),
}

fn execute_common_label<H: HashPolicy, L: Logger>(
    label: &Label,
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    node_id: NodeId,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Option<StepOutcome<H>>, RuntimeError> {
    match label {
        Label::Instr(instr, next) => match instr {
            Instr::Assign(lhs, rhs) | Instr::Copy(lhs, rhs) => {
                let v = eval(local_env, node_env, rhs, &program.id_to_name)?;
                store(lhs, v, local_env, node_env)?;
                Ok(Some(StepOutcome::Continue(*next)))
            }
            Instr::SyncCall(lhs, func_name, args) => {
                let arg_vals: Result<Vec<Value<H>>, _> = args
                    .iter()
                    .map(|a| eval(local_env, node_env, a, &program.id_to_name))
                    .collect();
                let arg_vals = arg_vals?;
                let func_name_id = program
                    .func_name_to_id
                    .get(func_name)
                    .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                let func_info = program
                    .rpc
                    .get(func_name_id)
                    .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;

                if !func_info.is_sync {
                    return Err(RuntimeError::SyncCallToAsyncFunction(func_name.clone()));
                }

                let mut callee_local = make_local_env(
                    func_info,
                    arg_vals,
                    local_env,
                    node_env,
                    &program.id_to_name,
                );

                let val = exec_sync_inner(
                    state,
                    logger,
                    program,
                    &mut callee_local,
                    node_env,
                    func_info.entry,
                    node_id,
                    global_snapshot,
                    local_coverage,
                )?;

                store(lhs, val, local_env, node_env)?;
                Ok(Some(StepOutcome::Continue(*next)))
            }
            Instr::Async(lhs, node_expr, func_name, args) => {
                let target_node =
                    eval(local_env, node_env, node_expr, &program.id_to_name)?.as_node()?;
                let arg_vals: Result<Vec<Value<H>>, _> = args
                    .iter()
                    .map(|a| eval(local_env, node_env, a, &program.id_to_name))
                    .collect();
                let arg_vals = arg_vals?;

                let chan_id = ChannelId {
                    node: node_id,
                    id: state.alloc_channel_id(),
                };

                state.channels.insert(chan_id, ChannelState::new());
                store(lhs, Value::channel(chan_id), local_env, node_env)?;

                let func_name_id = program
                    .func_name_to_id
                    .get(func_name)
                    .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                let func_info = program
                    .rpc
                    .get(func_name_id)
                    .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                let callee_locals = make_local_env(
                    func_info,
                    arg_vals,
                    local_env,
                    node_env,
                    &program.id_to_name,
                );

                let new_record = Record {
                    pc: func_info.entry,
                    node: target_node,
                    origin_node: node_id,
                    continuation: Continuation::Async { chan_id },
                    env: callee_locals,
                    x: 0.5,
                    policy: UpdatePolicy::Identity,
                };

                if state.crash_info.currently_crashed.contains(&target_node) {
                    state
                        .crash_info
                        .queued_messages
                        .push_back((target_node, new_record));
                } else {
                    state.runnable_tasks.push_back(Runnable::Record(new_record));
                }
                Ok(Some(StepOutcome::Continue(*next)))
            }
        },
        Label::MakeChannel(lhs, _, next) => {
            let cid = ChannelId {
                node: node_id,
                id: state.alloc_channel_id(),
            };
            state.channels.insert(cid, ChannelState::new());
            store(lhs, Value::channel(cid), local_env, node_env)?;
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::SetTimer(lhs, next) => {
            let cid = ChannelId {
                node: node_id,
                id: state.alloc_channel_id(),
            };
            state.channels.insert(cid, ChannelState::new());
            store(lhs, Value::channel(cid), local_env, node_env)?;

            // Create a timer that will fire when scheduled
            let timer = Timer {
                pc: *next,
                node: node_id,
                channel: cid,
            };
            state.runnable_tasks.push_back(Runnable::Timer(timer));
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::UniqueId(lhs, next) => {
            let id = state.alloc_unique_id();
            store(lhs, Value::int(id as i64), local_env, node_env)?;
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::Cond(cond, bthen, belse) => {
            if eval(local_env, node_env, cond, &program.id_to_name)?.as_bool()? {
                Ok(Some(StepOutcome::Continue(*bthen)))
            } else {
                Ok(Some(StepOutcome::Continue(*belse)))
            }
        }
        Label::Return(expr) => {
            let val = eval(local_env, node_env, expr, &program.id_to_name)?;
            Ok(Some(StepOutcome::Return(val)))
        }
        Label::Print(expr, next) => {
            let val = eval(local_env, node_env, expr, &program.id_to_name)?;
            logger.log(LogEntry {
                node: node_id,
                content: val.to_string(),
                step: state.crash_info.current_step,
            });
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::Break(target) => Ok(Some(StepOutcome::Continue(*target))),
        Label::Continue(target) => Ok(Some(StepOutcome::Continue(*target))),
        Label::ForLoopIn(lhs, expr, iter_state_slot, body, next) => {
            let iter_slot_idx = match iter_state_slot {
                VarSlot::Local(idx, _) => *idx,
                VarSlot::Node(_, _) => return Err(RuntimeError::InvalidIteratorState),
            };

            let col_val = {
                let current = local_env.get(iter_slot_idx).clone();
                if matches!(current.kind, ValueKind::Unit) {
                    let original_collection = eval(local_env, node_env, expr, &program.id_to_name)?;
                    local_env.set(iter_slot_idx, original_collection.clone());
                    original_collection
                } else {
                    current
                }
            };

            match col_val.kind {
                ValueKind::List(l) => {
                    if l.is_empty() {
                        local_env.set(iter_slot_idx, Value::unit());
                        Ok(Some(StepOutcome::Continue(*next)))
                    } else {
                        let item = l.head().ok_or(RuntimeError::EmptyCollection)?.clone();
                        let new_l = Value::list(l.skip(1));
                        local_env.set(iter_slot_idx, new_l);

                        store(lhs, item, local_env, node_env)?;
                        Ok(Some(StepOutcome::Continue(*body)))
                    }
                }
                ValueKind::Map(m) => {
                    if m.is_empty() {
                        local_env.set(iter_slot_idx, Value::unit());
                        Ok(Some(StepOutcome::Continue(*next)))
                    } else {
                        let (k, v) = m.iter().next().ok_or(RuntimeError::EmptyCollection)?;
                        let k = k.clone();
                        let v = v.clone();

                        let new_m = m.without(&k);
                        local_env.set(iter_slot_idx, Value::map(new_m));

                        match lhs {
                            Lhs::Tuple(vars) if vars.len() == 2 => {
                                store(&Lhs::Var(vars[0]), k, local_env, node_env)?;
                                store(&Lhs::Var(vars[1]), v, local_env, node_env)?;
                                Ok(Some(StepOutcome::Continue(*body)))
                            }
                            _ => Err(RuntimeError::ForLoopMapExpectsTupleLhs),
                        }
                    }
                }
                _ => Err(RuntimeError::ForLoopNotCollection {
                    got: col_val.type_name(),
                }),
            }
        }
        _ => Ok(None),
    }
}

fn exec_sync_inner<H: HashPolicy, L: Logger>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    start_pc: usize,
    node_id: NodeId,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Value<H>, RuntimeError> {
    let mut pc = start_pc;
    let mut prev_pc = pc;
    loop {
        if pc != prev_pc {
            let rarity = global_snapshot.map_or(1.0, |s| s.novelty_score(pc));
            local_coverage.record_with_rarity(prev_pc, pc, rarity);
            prev_pc = pc;
        }

        let label = program.cfg.get_label(pc);
        if let Some(outcome) = execute_common_label(
            label,
            state,
            logger,
            program,
            local_env,
            node_env,
            node_id,
            global_snapshot,
            local_coverage,
        )? {
            match outcome {
                StepOutcome::Continue(next) => {
                    pc = next;
                    continue;
                }
                StepOutcome::Return(val) => return Ok(val),
            }
        }

        return Err(RuntimeError::UnsupportedSyncInstruction(format!(
            "{:?}",
            label
        )));
    }
}

pub fn exec<H: HashPolicy, L: Logger>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    mut record: Record<H>,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Option<ClientOpResult<H>>, RuntimeError> {
    let mut local_env = record.env;
    let mut node_env = state.nodes[record.node.index].clone();

    let mut prev_pc = record.pc;

    loop {
        let current_pc = record.pc;
        if current_pc != prev_pc {
            let rarity = global_snapshot.map_or(1.0, |s| s.novelty_score(current_pc));
            local_coverage.record_with_rarity(prev_pc, current_pc, rarity);
            prev_pc = current_pc;
        }

        let label = program.cfg.get_label(record.pc);

        if let Some(outcome) = execute_common_label(
            label,
            state,
            logger,
            program,
            &mut local_env,
            &mut node_env,
            record.node,
            global_snapshot,
            local_coverage,
        )? {
            match outcome {
                StepOutcome::Continue(next) => {
                    record.pc = next;
                    continue;
                }
                StepOutcome::Return(val) => {
                    let node_id = record.node;
                    state.nodes[node_id.index] = node_env;
                    let result = record.continuation.call(state, val);
                    return Ok(result);
                }
            }
        }

        match label {
            Label::Send(chan_expr, val_expr, next) => {
                let cid =
                    eval(&local_env, &node_env, chan_expr, &program.id_to_name)?.as_channel()?;
                let val = eval(&local_env, &node_env, val_expr, &program.id_to_name)?;
                if cid.node != record.node {
                    // Remote Send
                    state.runnable_tasks.push_back(Runnable::ChannelSend {
                        target: cid.node,
                        channel: cid,
                        message: val,
                        origin_node: record.node,
                        x: record.x,
                        policy: record.policy.clone(),
                        pc: *next,
                    });
                    // Non-blocking, proceed
                    record.pc = *next;
                } else {
                    // Local Send
                    let mut chan = state
                        .channels
                        .get(&cid)
                        .ok_or(RuntimeError::ChannelNotFound(cid.id))?
                        .clone();

                    if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                        // Wakeup reader
                        let mut r_node_env = state.nodes[reader.node.index].clone();
                        store(&lhs, val, &mut reader.env, &mut r_node_env)?;
                        state.nodes[reader.node.index] = r_node_env;
                        state.runnable_tasks.push_back(Runnable::Record(reader));
                    } else {
                        chan.buffer.push_back(val);
                    }
                    state.channels.insert(cid, chan);
                    record.pc = *next;
                }
            }
            Label::Recv(lhs, chan_expr, next) => {
                let cid =
                    eval(&local_env, &node_env, chan_expr, &program.id_to_name)?.as_channel()?;
                if cid.node != record.node {
                    return Err(RuntimeError::RemoteChannelRead);
                }

                let mut chan = state
                    .channels
                    .get(&cid)
                    .ok_or(RuntimeError::ChannelNotFound(cid.id))?
                    .clone();

                if let Some(val) = chan.buffer.pop_front() {
                    store(lhs, val, &mut local_env, &mut node_env)?;
                    record.pc = *next;
                } else {
                    // Block Reader
                    let node_id = record.node;
                    record.env = local_env;
                    record.pc = *next; // When woke, proceed to next
                    chan.waiting_readers.push_back((record, lhs.clone()));
                    state.channels.insert(cid, chan);
                    state.nodes[node_id.index] = node_env;
                    return Ok(None); // Stop execution
                }
                state.channels.insert(cid, chan);
            }
            Label::Pause(next) => {
                let node_id = record.node;
                record.env = local_env;
                record.pc = *next;
                state.runnable_tasks.push_back(Runnable::Record(record));
                state.nodes[node_id.index] = node_env;
                return Ok(None); // Yield
            }
            Label::SpinAwait(expr, next) => {
                if eval(&local_env, &node_env, expr, &program.id_to_name)?.as_bool()? {
                    record.pc = *next;
                } else {
                    let node_id = record.node;
                    record.env = local_env;
                    state.runnable_tasks.push_back(Runnable::Record(record));
                    state.nodes[node_id.index] = node_env;
                    return Ok(None); // Yield
                }
            }
            Label::Instr(_, _)
            | Label::MakeChannel(_, _, _)
            | Label::SetTimer(_, _)
            | Label::UniqueId(_, _)
            | Label::Cond(_, _, _)
            | Label::Return(_)
            | Label::Print(_, _)
            | Label::Break(_)
            | Label::ForLoopIn(_, _, _, _, _) => {
                unreachable!(
                    "Label {:?} should have been handled by execute_common_label or is missing implementation in exec loop",
                    label
                )
            }
            Label::Continue(_) => {
                unreachable!(
                    "Label::Continue should have been handled by execute_common_label or is missing implementation in exec loop"
                )
            }
        }
    }
}

#[cfg(test)]
mod test;
