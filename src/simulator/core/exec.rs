use crate::compiler::cfg::{Instr, Label, Lhs, Program, SELF_SLOT, VarSlot};
use crate::simulator::coverage::{LocalCoverage, VertexMap};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::{eval, make_local_env, store};
use crate::simulator::core::state::{
    ChannelState, ClientOpResult, Continuation, LogEntry, Logger, Record, State, UpdatePolicy,
};
use crate::simulator::core::values::{ChannelId, Env, Value, ValueKind};

pub fn exec_sync_on_node<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env,
    node_id: usize,
    start_pc: usize,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Value, RuntimeError> {
    let mut node_env = state.nodes[node_id].clone();
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
    state.nodes[node_id] = node_env;
    result
}

fn exec_sync_inner<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env,
    node_env: &mut Env,
    start_pc: usize,
    node_id: usize,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Value, RuntimeError> {
    let mut pc = start_pc;
    let mut prev_pc = pc;
    loop {
        if pc != prev_pc {
            let rarity = global_snapshot.map_or(1.0, |s| s.novelty_score(pc));
            local_coverage.record_with_rarity(prev_pc, pc, rarity);
            prev_pc = pc;
        }

        let label = program.cfg.get_label(pc).clone();
        match label {
            Label::Instr(instr, next) => {
                pc = next;
                match instr {
                    Instr::Assign(lhs, rhs) => {
                        let v = eval(local_env, node_env, &rhs)?;
                        store(&lhs, v, local_env, node_env)?;
                    }
                    Instr::Copy(lhs, rhs) => {
                        let v = eval(local_env, node_env, &rhs)?;
                        store(&lhs, v, local_env, node_env)?;
                    }
                    Instr::SyncCall(lhs, func_name, args) => {
                        let arg_vals: Result<Vec<Value>, _> =
                            args.iter().map(|a| eval(local_env, node_env, a)).collect();
                        let arg_vals = arg_vals?;
                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;

                        if !func_info.is_sync {
                            return Err(RuntimeError::SyncCallToAsyncFunction(func_name.clone()));
                        }

                        let arg_vals: Vec<Value> = arg_vals;
                        let mut callee_local =
                            make_local_env(func_info, arg_vals, local_env, node_env);

                        // Pass node_env directly (it's already mutable borrowed from caller)
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

                        store(&lhs, val, local_env, node_env)?;
                    }
                    Instr::Async(lhs, node_expr, func_name, args) => {
                        // Similar to exec Async but we are inside sync.
                        let target_node = eval(local_env, node_env, &node_expr)?.as_node()?;
                        let arg_vals: Result<Vec<Value>, _> =
                            args.iter().map(|a| eval(local_env, node_env, a)).collect();
                        let arg_vals = arg_vals?;

                        let origin_node = match SELF_SLOT {
                            VarSlot::Local(idx, _) => local_env.get(idx).clone().as_node()?,
                            VarSlot::Node(idx, _) => node_env.get(idx).clone().as_node()?,
                        };

                        let chan_id = ChannelId {
                            node: origin_node,
                            id: state.alloc_channel_id(),
                        };

                        state.channels.insert(chan_id, ChannelState::new(1));
                        store(&lhs, Value::channel(chan_id), local_env, node_env)?;

                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let callee_locals =
                            make_local_env(func_info, arg_vals, local_env, node_env);

                        let new_record = Record {
                            pc: func_info.entry,
                            node: target_node,
                            origin_node,
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
                            state.runnable_records.push_back(new_record);
                        }
                    }
                }
            }
            Label::MakeChannel(lhs, cap, next) => {
                let origin_node = match SELF_SLOT {
                    VarSlot::Local(idx, _) => local_env.get(idx).clone().as_node()?,
                    VarSlot::Node(idx, _) => node_env.get(idx).clone().as_node()?,
                };
                let cid = ChannelId {
                    node: origin_node,
                    id: state.alloc_channel_id(),
                };
                state.channels.insert(cid, ChannelState::new(cap as i32));
                store(&lhs, Value::channel(cid), local_env, node_env)?;
                pc = next;
            }
            Label::Cond(cond, bthen, belse) => {
                if eval(local_env, node_env, &cond)?.as_bool()? {
                    pc = bthen;
                } else {
                    pc = belse;
                }
            }
            Label::Return(expr) => {
                return eval(local_env, node_env, &expr);
            }
            Label::Print(expr, next) => {
                let val = eval(local_env, node_env, &expr)?;
                logger.log(LogEntry {
                    node: node_id,
                    content: val.to_string(),
                    step: state.crash_info.current_step,
                });
                pc = next;
            }
            Label::Break(target) => {
                pc = target;
            }
            Label::ForLoopIn(lhs, expr, iter_state_slot, body, next) => {
                // Check if we need to initialize the iterator state
                let iter_slot_idx = match iter_state_slot {
                    VarSlot::Local(idx, _) => idx,
                    VarSlot::Node(_, _) => panic!("Iterator state must be local"),
                };

                let col_val = {
                    let current = local_env.get(iter_slot_idx).clone();
                    if matches!(current.kind, ValueKind::Unit) {
                        // First iteration: initialize with collection
                        let original_collection = eval(local_env, node_env, &expr)?;
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
                            pc = next;
                        } else {
                            let item = l.head().unwrap().clone();
                            let new_l = Value::list(l.skip(1));
                            local_env.set(iter_slot_idx, new_l);

                            store(&lhs, item, local_env, node_env)?;
                            pc = body;
                        }
                    }
                    ValueKind::Map(m) => {
                        if m.is_empty() {
                            local_env.set(iter_slot_idx, Value::unit());
                            pc = next;
                        } else {
                            let (k, v) = m.iter().next().unwrap();
                            let k = k.clone();
                            let v = v.clone();

                            let new_m = m.without(&k);
                            local_env.set(iter_slot_idx, Value::map(new_m));

                            match lhs {
                                Lhs::Tuple(vars) if vars.len() == 2 => {
                                    store(&Lhs::Var(vars[0]), k, local_env, node_env)?;
                                    store(&Lhs::Var(vars[1]), v, local_env, node_env)?;
                                    pc = body;
                                }
                                _ => return Err(RuntimeError::ForLoopMapExpectsTupleLhs),
                            }
                        }
                    }
                    _ => {
                        return Err(RuntimeError::ForLoopNotCollection {
                            got: col_val.type_name(),
                        });
                    }
                }
            }
            other => {
                return Err(RuntimeError::UnsupportedSyncInstruction(format!(
                    "{:?}",
                    other
                )));
            }
        }
    }
}

pub fn exec<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    mut record: Record,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Option<ClientOpResult>, RuntimeError> {
    let mut local_env = record.env;
    let mut node_env = state.nodes[record.node].clone();

    let mut prev_pc = record.pc;

    loop {
        let current_pc = record.pc;
        if current_pc != prev_pc {
            let rarity = global_snapshot.map_or(1.0, |s| s.novelty_score(current_pc));
            local_coverage.record_with_rarity(prev_pc, current_pc, rarity);
            prev_pc = current_pc;
        }

        let label = program.cfg.get_label(record.pc).clone();

        match label {
            Label::Instr(instr, next) => {
                record.pc = next;
                match instr {
                    Instr::Assign(lhs, rhs) => {
                        let v = eval(&local_env, &node_env, &rhs)?;
                        store(&lhs, v, &mut local_env, &mut node_env)?;
                    }
                    Instr::Copy(lhs, rhs) => {
                        let v = eval(&local_env, &node_env, &rhs)?;
                        store(&lhs, v, &mut local_env, &mut node_env)?;
                    }
                    Instr::SyncCall(lhs, func_name, args) => {
                        let arg_vals: Result<Vec<Value>, _> = args
                            .iter()
                            .map(|a| eval(&local_env, &node_env, a))
                            .collect();
                        let arg_vals = arg_vals?;
                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;

                        if !func_info.is_sync {
                            return Err(RuntimeError::SyncCallToAsyncFunction(func_name.clone()));
                        }

                        let arg_vals: Vec<Value> = arg_vals;
                        let mut callee_local =
                            make_local_env(func_info, arg_vals, &local_env, &node_env);

                        // Pass node_env directly
                        let ret_val = exec_sync_inner(
                            state,
                            logger,
                            program,
                            &mut callee_local,
                            &mut node_env,
                            func_info.entry,
                            record.node,
                            global_snapshot,
                            local_coverage,
                        )?;

                        store(&lhs, ret_val, &mut local_env, &mut node_env)?;
                    }
                    Instr::Async(lhs, node_expr, func_name, args) => {
                        // Async logic
                        let target_node = eval(&local_env, &node_env, &node_expr)?.as_node()?;
                        let arg_vals: Result<Vec<Value>, _> = args
                            .iter()
                            .map(|a| eval(&local_env, &node_env, a))
                            .collect();
                        let arg_vals = arg_vals?;

                        // Make channel
                        let chan_id = ChannelId {
                            node: record.node,
                            id: state.alloc_channel_id(),
                        };
                        state.channels.insert(chan_id, ChannelState::new(1));
                        store(&lhs, Value::channel(chan_id), &mut local_env, &mut node_env)?;

                        // Setup Callee Record
                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let callee_locals =
                            make_local_env(func_info, arg_vals, &local_env, &node_env);

                        let new_record = Record {
                            pc: func_info.entry,
                            node: target_node,
                            origin_node: record.node,
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
                            state.runnable_records.push_back(new_record);
                        }
                    }
                }
            }
            Label::MakeChannel(lhs, cap, next) => {
                let cid = ChannelId {
                    node: record.node,
                    id: state.alloc_channel_id(),
                };
                state.channels.insert(cid, ChannelState::new(cap as i32));
                store(&lhs, Value::channel(cid), &mut local_env, &mut node_env)?;
                record.pc = next;
            }
            Label::Send(chan_expr, val_expr, next) => {
                let cid = eval(&local_env, &node_env, &chan_expr)?.as_channel()?;
                let val = eval(&local_env, &node_env, &val_expr)?;
                if cid.node != record.node {
                    return Err(RuntimeError::NetworkedChannelUnsupported);
                }
                // Local Send
                let mut chan = state.channels.get(&cid).unwrap().clone();
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    // Wakeup reader
                    let mut r_node_env = state.nodes[reader.node].clone();
                    store(&lhs, val, &mut reader.env, &mut r_node_env)?;
                    state.nodes[reader.node] = r_node_env;
                    state.runnable_records.push_back(reader);
                    record.pc = next;
                } else if (chan.buffer.len() as i32) < chan.capacity {
                    chan.buffer.push_back(val);
                    record.pc = next;
                } else {
                    let node_id = record.node;
                    record.env = local_env;
                    record.pc = next;
                    chan.waiting_writers.push_back((record, val));
                    state.channels.insert(cid, chan);
                    state.nodes[node_id] = node_env;
                    return Ok(None);
                }
                state.channels.insert(cid, chan);
            }
            Label::Recv(lhs, chan_expr, next) => {
                let cid = eval(&local_env, &node_env, &chan_expr)?.as_channel()?;
                if cid.node != record.node {
                    return Err(RuntimeError::RemoteChannelRead);
                }

                let mut chan = state.channels.get(&cid).unwrap().clone();

                if let Some(val) = chan.buffer.pop_front() {
                    store(&lhs, val, &mut local_env, &mut node_env)?;
                    // Wake writer if any
                    if let Some((writer, w_val)) = chan.waiting_writers.pop_front() {
                        chan.buffer.push_back(w_val);
                        state.runnable_records.push_back(writer);
                    }
                    record.pc = next;
                } else if let Some((writer, w_val)) = chan.waiting_writers.pop_front() {
                    store(&lhs, w_val, &mut local_env, &mut node_env)?;
                    state.runnable_records.push_back(writer);
                    record.pc = next;
                } else {
                    // Block Reader
                    let node_id = record.node;
                    record.env = local_env;
                    record.pc = next; // When woke, proceed to next
                    chan.waiting_readers.push_back((record, lhs));
                    state.channels.insert(cid, chan);
                    state.nodes[node_id] = node_env;
                    return Ok(None); // Stop execution
                }
                state.channels.insert(cid, chan);
            }
            Label::Return(expr) => {
                let val = eval(&local_env, &node_env, &expr)?;
                // Call continuation and return the result
                let node_id = record.node;
                state.nodes[node_id] = node_env;
                let result = record.continuation.call(state, val);
                return Ok(result);
            }
            Label::Pause(next) => {
                let node_id = record.node;
                record.env = local_env;
                record.pc = next;
                state.runnable_records.push_back(record);
                state.nodes[node_id] = node_env;
                return Ok(None); // Yield
            }
            Label::Cond(cond, bthen, belse) => {
                if eval(&local_env, &node_env, &cond)?.as_bool()? {
                    record.pc = bthen;
                } else {
                    record.pc = belse;
                }
            }
            Label::Print(expr, next) => {
                let val = eval(&local_env, &node_env, &expr)?;
                logger.log(LogEntry {
                    node: record.node,
                    content: val.to_string(),
                    step: state.crash_info.current_step,
                });
                record.pc = next;
            }
            Label::SpinAwait(expr, next) => {
                if eval(&local_env, &node_env, &expr)?.as_bool()? {
                    record.pc = next;
                } else {
                    let node_id = record.node;
                    record.env = local_env;
                    state.runnable_records.push_back(record);
                    state.nodes[node_id] = node_env;
                    return Ok(None); // Yield
                }
            }
            Label::ForLoopIn(lhs, expr, iter_state_slot, body_pc, next_pc) => {
                // Check if we need to initialize the iterator state
                let iter_slot_idx = match iter_state_slot {
                    VarSlot::Local(idx, _) => idx,
                    VarSlot::Node(_, _) => panic!("Iterator state must be local"),
                };

                let col_val = {
                    let current = local_env.get(iter_slot_idx).clone();
                    if matches!(current.kind, ValueKind::Unit) {
                        // First iteration: initialize with collection
                        let original_collection = eval(&local_env, &node_env, &expr)?;
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
                            record.pc = next_pc;
                        } else {
                            let item = l.head().unwrap().clone();
                            let new_l = Value::list(l.skip(1));
                            local_env.set(iter_slot_idx, new_l);

                            store(&lhs, item, &mut local_env, &mut node_env)?;
                            record.pc = body_pc;
                        }
                    }
                    ValueKind::Map(m) => {
                        if m.is_empty() {
                            local_env.set(iter_slot_idx, Value::unit());
                            record.pc = next_pc;
                        } else {
                            let (k, v) = m.iter().next().unwrap();
                            let k = k.clone();
                            let v = v.clone();

                            let new_m = m.without(&k);
                            local_env.set(iter_slot_idx, Value::map(new_m));

                            match lhs {
                                Lhs::Tuple(vars) if vars.len() == 2 => {
                                    store(&Lhs::Var(vars[0]), k, &mut local_env, &mut node_env)?;
                                    store(&Lhs::Var(vars[1]), v, &mut local_env, &mut node_env)?;
                                    record.pc = body_pc;
                                }
                                _ => return Err(RuntimeError::ForLoopMapExpectsTupleLhs),
                            }
                        }
                    }
                    _ => {
                        return Err(RuntimeError::ForLoopNotCollection {
                            got: col_val.type_name(),
                        });
                    }
                }
            }
            Label::Break(target) => {
                record.pc = target;
            }
        }
    }
}
