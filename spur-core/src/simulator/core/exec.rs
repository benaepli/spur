use crate::analysis::resolver::NameId;
use crate::compiler::cfg::compiled::{
    AsyncOp, Dest, ForLoopInOp, Op, TraceDispatchOp, TraceEnterOp, TraceExitOp,
};
use crate::compiler::cfg::{Expr, FunctionInfo, Instr, Label, Program, VarSlot};
use crate::simulator::core::compiled_eval::{borrowed, coperand, cvalue};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::{
    FrameBuilder, Operand, build_frame, eval, eval_operand, set_local, store,
};
use crate::simulator::core::state::{
    ClientOpResult, Continuation, LogEntry, Logger, NodeId, PurgatoryConfig, Record,
    Runnable, RunnableCategory, SchedulePolicy, State, Timer, TraceEntry, TraceKind,
};
use crate::simulator::core::values::{ChannelId, Env, Value, ValueKind, ValueSeq};
use ecow::EcoVec;
use rand::Rng;
use crate::simulator::rng::{Stream, StreamRng};
use crate::simulator::feedback::Feedback;
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::text_buffer::TextBuffer;
use crate::simulator::util_stats;
use crate::simulator::util_stats::{DeliveryBias, InterpreterTally};
use std::cell::RefCell;
use std::collections::HashMap;

/// Scratch for building one trace payload: the parameter texts laid end to
/// end, and where each one ends. Kept per thread so formatting a parameter
/// allocates nothing after the first row on that thread.
#[derive(Default)]
struct TraceScratch {
    text: String,
    ends: Vec<usize>,
}

thread_local! {
    static TRACE_SCRATCH: RefCell<TraceScratch> = RefCell::new(TraceScratch::default());
}

/// Formats each evaluated parameter and appends the JSON array of the texts
/// to `out`, byte for byte what `serde_json` writes for a `Vec<String>` of
/// the same items. A parameter whose evaluation failed reads "<error>".
fn write_trace_payload<H: HashPolicy>(
    values: impl Iterator<Item = Result<Value<H>, RuntimeError>>,
    out: &mut TextBuffer,
) {
    TRACE_SCRATCH.with(|scratch| {
        let scratch = &mut *scratch.borrow_mut();
        scratch.text.clear();
        scratch.ends.clear();
        for value in values {
            match value {
                Ok(v) => {
                    let _ = v.write_to(&mut scratch.text);
                }
                Err(_) => scratch.text.push_str("<error>"),
            }
            scratch.ends.push(scratch.text.len());
        }
        json_string_array(&scratch.text, &scratch.ends, out)
    })
}

/// Appends the JSON array whose elements are the pieces of `text` ending at
/// each offset in `ends`, in order. `ends` must be non-decreasing and bounded
/// by `text.len()`, with every offset on a character boundary.
pub(crate) fn json_string_array(text: &str, ends: &[usize], out: &mut TextBuffer) {
    out.reserve(text.len() + 2 + 4 * ends.len());
    out.push_str("[");
    let mut start = 0;
    for (i, &end) in ends.iter().enumerate() {
        if i > 0 {
            out.push_str(",");
        }
        out.push_json_str(&text[start..end]);
        start = end;
    }
    out.push_str("]");
}

/// Appends the printed text of `val` to `out` and returns where it ends. A
/// string prints as its text between two quote characters.
fn print_into<H: HashPolicy>(val: &Value<H>, out: &mut TextBuffer) -> usize {
    if let ValueKind::String(s) = &val.kind {
        util_stats::record_print_presized();
        out.reserve(s.len() + 2);
    }
    let _ = val.write_to(out);
    out.len()
}

pub fn exec_sync_on_node<H: HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_id: NodeId,
    start_pc: usize,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl StreamRng,
) -> Result<Value<H>, RuntimeError> {
    let mut node_env = state.nodes[node_id.index].clone();
    let result = if program.compiled.covers(program.cfg.graph.len()) {
        let mut tally = InterpreterTally::new();
        let result = run_sync_ops::<H, L, F>(
            state,
            logger,
            program,
            local_env,
            &mut node_env,
            start_pc,
            node_id,
            snapshot,
            feedback,
            policy,
            purgatory_config,
            None, // top-level sync calls have no causal client op
            rng,
            &mut tally,
        );
        tally.flush();
        result
    } else {
        exec_sync_inner::<H, L, F>(
            state,
            logger,
            program,
            local_env,
            &mut node_env,
            start_pc,
            node_id,
            snapshot,
            feedback,
            policy,
            purgatory_config,
            None, // top-level sync calls have no causal client op
            rng,
        )
    };
    state.nodes[node_id.index] = node_env;
    result
}

/// Evaluates one expression of a label run from the graph.
fn legacy_eval<H: HashPolicy>(
    local_env: &Env<H>,
    node_env: &Env<H>,
    expr: &Expr,
    role_names: &HashMap<NameId, String>,
) -> Result<Value<H>, RuntimeError> {
    util_stats::record_legacy_eval();
    eval(local_env, node_env, expr, role_names)
}

/// Evaluates one read-only expression of a label run from the graph.
fn legacy_operand<'a, H: HashPolicy>(
    local_env: &'a Env<H>,
    node_env: &'a Env<H>,
    expr: &Expr,
    role_names: &HashMap<NameId, String>,
) -> Result<Operand<'a, H>, RuntimeError> {
    util_stats::record_legacy_eval();
    eval_operand(local_env, node_env, expr, role_names)
}

enum StepOutcome<H: HashPolicy> {
    Continue(usize),
    Return(Value<H>),
}

/// Decide how long a remote send waits in purgatory before it is enqueued.
/// `None` means enqueue it now. The selection roll and the duration draw are
/// made before the destination's liveness is consulted, so suppressing a hold
/// for one destination leaves the draws seen by every other send unchanged.
fn purgatory_hold_steps<H: HashPolicy>(
    state: &State<H>,
    purgatory_config: &PurgatoryConfig,
    dest: NodeId,
    rng: &mut impl StreamRng,
) -> Option<i32> {
    rng.use_stream(Stream::SendDelay);
    if purgatory_config.delay_probability <= 0.0
        || rng.random::<f64>() >= purgatory_config.delay_probability
    {
        return None;
    }
    let (min, max) = purgatory_config.delay_duration_range;
    let duration = if min >= max {
        min
    } else {
        let ln_min = (min as f64).ln();
        let ln_max = (max as f64).ln();
        rng.random_range(ln_min..=ln_max).exp().round() as i32
    };
    let receiver_down = state.crash_info.currently_crashed.contains(&dest);
    if receiver_down && !purgatory_config.hold_down_receivers {
        util_stats::record_purgatory_passthrough_down_receiver();
        return None;
    }
    util_stats::record_purgatory_delay(receiver_down);
    Some(duration)
}

fn execute_common_label<H: HashPolicy, L: Logger, F: Feedback>(
    label: &Label,
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    node_id: NodeId,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    causal_operation_id: Option<i32>,
    pending_trace_id: &mut Option<i64>,
    pending_trace_payload: &mut Option<Box<str>>,
    rng: &mut impl StreamRng,
) -> Result<Option<StepOutcome<H>>, RuntimeError> {
    match label {
        Label::Instr(instr, next) => match instr {
            Instr::Assign(lhs, rhs) | Instr::Copy(lhs, rhs) => {
                let v = legacy_eval(local_env, node_env, rhs, &program.id_to_name)?;
                store(lhs, v, local_env, node_env)?;
                Ok(Some(StepOutcome::Continue(*next)))
            }
            Instr::SyncCall(lhs, func_name, args) => {
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

                let mut builder = FrameBuilder::<H>::new(func_info);
                for a in args {
                    builder.push(legacy_eval(local_env, node_env, a, &program.id_to_name)?);
                }
                let mut callee_local = builder.finish(func_info);

                let val = exec_sync_inner::<H, L, F>(
                    state,
                    logger,
                    program,
                    &mut callee_local,
                    node_env,
                    func_info.entry,
                    node_id,
                    snapshot,
                    feedback,
                    policy,
                    purgatory_config,
                    causal_operation_id,
                    rng,
                )?;

                store(lhs, val, local_env, node_env)?;
                Ok(Some(StepOutcome::Continue(*next)))
            }
            Instr::Async(lhs, node_expr, func_name, args) => {
                let (target_node, link_id) =
                    legacy_operand(local_env, node_env, node_expr, &program.id_to_name)?
                        .as_rpc_target()?;
                let mut arg_vals: EcoVec<Value<H>> = EcoVec::with_capacity(args.len());
                for a in args {
                    arg_vals.push(legacy_eval(local_env, node_env, a, &program.id_to_name)?);
                }

                let chan_id = ChannelId {
                    node: node_id,
                    id: state.alloc_channel_id(),
                };

                state.insert_channel(chan_id);
                store(lhs, Value::channel(chan_id), local_env, node_env)?;

                let func_name_id = program
                    .func_name_to_id
                    .get(func_name)
                    .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                let func_info = program
                    .rpc
                    .get(func_name_id)
                    .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                let callee_locals = build_frame(func_info, &arg_vals);

                // Tag with FIFO sequence number if routed through a FIFO link.
                // Allocates the next sender-side seq for this link.
                let link_seq = link_id.map(|lid| {
                    let next = state.link_send_seq.get(&lid).copied().unwrap_or(0);
                    state.link_send_seq.insert(lid, next + 1);
                    (lid, next)
                });

                let send_ordinal = state.next_send_ordinal(node_id);
                let drawn_priority = policy.sample(rng, RunnableCategory::Record);
                let mut new_record = Record {
                    pc: func_info.entry,
                    node: target_node,
                    origin_node: node_id,
                    continuation: Continuation::Async { chan_id },
                    entry_pc: func_info.entry,
                    initial_args: arg_vals,
                    entry_func: func_info.name,
                    env: callee_locals,
                    priority: state.record_priority(causal_operation_id, drawn_priority),
                    causal_operation_id,
                    trace_id: pending_trace_id.take(),
                    trace_payload: pending_trace_payload.take(),
                    link_seq,
                    origin_incarnation: state.incarnation(node_id),
                    bias: DeliveryBias::NONE,
                    timer_entry: None,
                    send_ordinal,
                    receiver_token_at_send: state.node_state_token(target_node),
                };

                match purgatory_hold_steps(state, purgatory_config, target_node, rng) {
                    Some(duration) => {
                        let release_step = state.crash_info.current_step + duration;
                        new_record.bias.insert(DeliveryBias::DELAYED);
                        state.delay_runnable(release_step, Runnable::Record(new_record));
                    }
                    None => state.push_runnable(Runnable::Record(new_record)),
                }
                Ok(Some(StepOutcome::Continue(*next)))
            }
        },
        Label::MakeChannel(lhs, _, next) => {
            let cid = ChannelId {
                node: node_id,
                id: state.alloc_channel_id(),
            };
            state.insert_channel(cid);
            store(lhs, Value::channel(cid), local_env, node_env)?;
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::MakeFifoLink(lhs, peer_expr, next) => {
            let peer =
                legacy_operand(local_env, node_env, peer_expr, &program.id_to_name)?.as_node()?;
            let link_id = state.alloc_link_id();
            state.link_meta.insert(link_id, (node_id, peer));
            state.link_send_seq.insert(link_id, 0);
            state.link_deliver_seq.insert(link_id, 0);
            store(lhs, Value::fifo_link(link_id, peer), local_env, node_env)?;
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::SetTimer(lhs, next, label) => {
            let cid = ChannelId {
                node: node_id,
                id: state.alloc_channel_id(),
            };
            state.insert_channel(cid);
            store(lhs, Value::channel(cid), local_env, node_env)?;

            // Create a timer that will fire when scheduled
            let timer = Timer {
                pc: *next,
                node: node_id,
                channel: cid,
                priority: policy.sample(rng, RunnableCategory::Timer),
                label: label.clone(),
            };
            state.push_runnable(Runnable::Timer(timer));
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::UniqueId(lhs, next) => {
            let id = state.alloc_unique_id();
            store(lhs, Value::int(id as i64), local_env, node_env)?;
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::Cond(cond, bthen, belse) => {
            if legacy_operand(local_env, node_env, cond, &program.id_to_name)?.as_bool()? {
                Ok(Some(StepOutcome::Continue(*bthen)))
            } else {
                Ok(Some(StepOutcome::Continue(*belse)))
            }
        }
        Label::Return(expr) => {
            let val = legacy_eval(local_env, node_env, expr, &program.id_to_name)?;
            Ok(Some(StepOutcome::Return(val)))
        }
        Label::Print(expr, next) => {
            let val = legacy_operand(local_env, node_env, expr, &program.id_to_name)?;
            let content_end = print_into(&val, logger.log_text());
            logger.log(LogEntry {
                node: node_id,
                content_end,
                step: state.crash_info.current_step,
            });
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::Break(target) => Ok(Some(StepOutcome::Continue(*target))),
        Label::Continue(target) => Ok(Some(StepOutcome::Continue(*target))),
        Label::PersistData(type_id, expr, next) => {
            let val = legacy_eval(local_env, node_env, expr, &program.id_to_name)?;
            state.persisted_data.insert(node_id.index, (*type_id, val));
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::RetrieveData(type_id, lhs, next) => {
            let result = match state.persisted_data.get(&node_id.index) {
                Some((stored_tid, val)) => {
                    if stored_tid != type_id {
                        return Err(RuntimeError::PersistTypeMismatch {
                            stored: stored_tid.0,
                            expected: type_id.0,
                        });
                    }
                    Value::option_some(val.clone())
                }
                None => Value::option_none(),
            };
            store(lhs, result, local_env, node_env)?;
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::DiscardData(next) => {
            state.persisted_data.remove(&node_id.index);
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::ForLoopIn(lhs, expr, iter_state_slot, body, next) => {
            let iter_slot_idx = match iter_state_slot {
                VarSlot::Local(idx, _) => *idx,
                VarSlot::Node(_, _) => return Err(RuntimeError::InvalidIteratorState),
            };

            let col_val = {
                let current = local_env.get(iter_slot_idx).clone();
                if matches!(current.kind, ValueKind::Unit) {
                    let original_collection = legacy_eval(local_env, node_env, expr, &program.id_to_name)?;
                    set_local(local_env, iter_slot_idx, original_collection.clone());
                    original_collection
                } else {
                    current
                }
            };
            let col_val = col_val.into_map_form();

            match col_val.kind {
                ValueKind::List(l) => {
                    if l.is_empty() {
                        set_local(local_env, iter_slot_idx, Value::unit());
                        Ok(Some(StepOutcome::Continue(*next)))
                    } else {
                        let item = l.first().ok_or(RuntimeError::EmptyCollection)?.clone();
                        let new_l = Value::list(ValueSeq::from(&l[1..]));
                        set_local(local_env, iter_slot_idx, new_l);

                        store(lhs, item, local_env, node_env)?;
                        Ok(Some(StepOutcome::Continue(*body)))
                    }
                }
                ValueKind::Map(m) => {
                    if m.is_empty() {
                        set_local(local_env, iter_slot_idx, Value::unit());
                        Ok(Some(StepOutcome::Continue(*next)))
                    } else {
                        let (k, v) = m.iter().next().ok_or(RuntimeError::EmptyCollection)?;
                        let k = k.clone();
                        let v = v.clone();

                        let new_m = m.without(&k);
                        set_local(local_env, iter_slot_idx, Value::map(new_m));

                        let pair = Value::tuple(ValueSeq::from([k, v]));
                        store(lhs, pair, local_env, node_env)?;
                        Ok(Some(StepOutcome::Continue(*body)))
                    }
                }
                _ => Err(RuntimeError::ForLoopNotCollection {
                    got: col_val.type_name(),
                }),
            }
        }
        Label::TraceEnter(func_name, param_exprs, trace_id_lhs, next) => {
            let pending_id = pending_trace_id.take();
            let carried = pending_trace_payload.take();
            let id = pending_id.unwrap_or_else(|| state.alloc_unique_id() as i64);
            store(trace_id_lhs, Value::int(id), local_env, node_env)?;
            // A carried payload was formatted by the dispatch that allocated
            // the pending id, from the argument values this frame's
            // parameters were built from, so it is the text formatting them
            // here would produce.
            let out = logger.trace_text();
            match (pending_id, carried) {
                (Some(_), Some(text)) => {
                    util_stats::record_trace_enter_payload(true);
                    out.push_str(&text);
                }
                _ => {
                    util_stats::record_trace_enter_payload(false);
                    write_trace_payload(
                        param_exprs
                            .iter()
                            .map(|e| legacy_eval(local_env, node_env, e, &program.id_to_name)),
                        out,
                    );
                }
            }
            let payload_end = out.len();
            logger.log_trace(TraceEntry {
                node: node_id,
                function_name: func_name.clone(),
                kind: TraceKind::Enter,
                payload_end,
                schedulable_count: state.total_runnable_count(),
                step: state.crash_info.current_step,
                trace_id: id,
                causal_operation_id: causal_operation_id.map(|id| id as i64),
            });
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::TraceExit(func_name, trace_id_expr, return_val_expr, next) => {
            let trace_id =
                legacy_eval(local_env, node_env, trace_id_expr, &program.id_to_name)?.as_int()?;
            let return_val = legacy_eval(local_env, node_env, return_val_expr, &program.id_to_name)?;
            let out = logger.trace_text();
            write_trace_payload(std::iter::once(Ok(return_val)), out);
            let payload_end = out.len();
            logger.log_trace(TraceEntry {
                node: node_id,
                function_name: func_name.clone(),
                kind: TraceKind::Exit,
                payload_end,
                schedulable_count: state.total_runnable_count(),
                step: state.crash_info.current_step,
                trace_id,
                causal_operation_id: causal_operation_id.map(|id| id as i64),
            });
            Ok(Some(StepOutcome::Continue(*next)))
        }
        Label::TraceDispatch(func_name, param_exprs, next) => {
            let id = state.alloc_unique_id() as i64;
            *pending_trace_id = Some(id);
            let out = logger.trace_text();
            let start = out.len();
            write_trace_payload(
                param_exprs
                    .iter()
                    .map(|e| legacy_eval(local_env, node_env, e, &program.id_to_name)),
                out,
            );
            *pending_trace_payload = Some(Box::from(out.str_from(start)));
            let payload_end = out.len();
            logger.log_trace(TraceEntry {
                node: node_id,
                function_name: func_name.clone(),
                kind: TraceKind::Dispatch,
                payload_end,
                schedulable_count: state.total_runnable_count(),
                step: state.crash_info.current_step,
                trace_id: id,
                causal_operation_id: causal_operation_id.map(|id| id as i64),
            });
            Ok(Some(StepOutcome::Continue(*next)))
        }
        _ => Ok(None),
    }
}

fn exec_sync_inner<H: HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    start_pc: usize,
    node_id: NodeId,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    causal_operation_id: Option<i32>,
    rng: &mut impl StreamRng,
) -> Result<Value<H>, RuntimeError> {
    let mut pc = start_pc;
    let mut prev_pc = pc;
    let mut pending_trace_id = None;
    let mut pending_trace_payload = None;
    loop {
        if pc != prev_pc {
            F::record_transition(feedback, prev_pc, pc, snapshot);
            prev_pc = pc;
        }

        let label = program.cfg.get_label(pc);
        util_stats::record_legacy_label();
        if let Some(outcome) = execute_common_label::<H, L, F>(
            label,
            state,
            logger,
            program,
            local_env,
            node_env,
            node_id,
            snapshot,
            feedback,
            policy,
            purgatory_config,
            causal_operation_id,
            &mut pending_trace_id,
            &mut pending_trace_payload,
            rng,
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

/// Runs a record from the graph's labels, for a program without a decoded
/// form.
fn exec_legacy<H: HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    mut record: Record<H>,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl StreamRng,
) -> Result<Option<ClientOpResult<H>>, RuntimeError> {
    let causal_operation_id = record.causal_operation_id;
    let mut pending_trace_id = record.trace_id;
    // The payload moves out rather than being copied: only the entry row
    // reads it, so a record stored again after that row carries none, and a
    // re-delivery of it formats its entry row from the parameters.
    let mut pending_trace_payload = record.trace_payload.take();
    let mut local_env = record.env;
    let mut node_env = state.nodes[record.node.index].clone();

    let mut prev_pc = record.pc;

    // Capture the first-entry handler delivery for timeline coverage. The whole
    // block is const-folded away for strategies that do not track timelines.
    // `record.pc == record.entry_pc` is true only on the first exec entry of a
    // delivery; Recv/Pause re-push with pc advanced past entry.
    if F::CAPTURES_TIMELINE && record.pc == record.entry_pc {
        let server_role = program
            .roles
            .iter()
            .find(|(_, n)| n == "Node")
            .map(|(id, _)| *id);
        if server_role == Some(record.node.role) {
            F::note_delivery(feedback, record.node, record.entry_pc);
        }
    }

    loop {
        let current_pc = record.pc;
        if current_pc != prev_pc {
            F::record_transition(feedback, prev_pc, current_pc, snapshot);
            prev_pc = current_pc;
        }

        let label = program.cfg.get_label(record.pc);
        util_stats::record_legacy_label();

        if let Some(outcome) = execute_common_label::<H, L, F>(
            label,
            state,
            logger,
            program,
            &mut local_env,
            &mut node_env,
            record.node,
            snapshot,
            feedback,
            policy,
            purgatory_config,
            causal_operation_id,
            &mut pending_trace_id,
            &mut pending_trace_payload,
            rng,
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
                let cid = legacy_operand(&local_env, &node_env, chan_expr, &program.id_to_name)?
                    .as_channel()?;
                let val = legacy_eval(&local_env, &node_env, val_expr, &program.id_to_name)?;
                if cid.node != record.node {
                    let cs = Runnable::ChannelSend {
                        target: cid.node,
                        channel: cid,
                        message: val,
                        origin_node: record.node,
                        pc: *next,
                        priority: policy.sample(rng, RunnableCategory::ChannelSend),
                    };
                    match purgatory_hold_steps(state, purgatory_config, cid.node, rng) {
                        Some(duration) => {
                            let release_step = state.crash_info.current_step + duration;
                            state.delay_runnable(release_step, cs);
                        }
                        None => state.push_runnable(cs),
                    }
                    // Non-blocking, proceed
                    record.pc = *next;
                } else {
                    // Local Send
                    let chan = state
                        .channels
                        .get_mut(&cid)
                        .ok_or(RuntimeError::ChannelNotFound(cid.id))?;

                    match chan.pop_waiting_reader() {
                        None => chan.buffer.push_back(val),
                        Some((mut reader, lhs)) => {
                            let node_index = reader.node.index;
                            let mut r_node_env = state.nodes[node_index].clone();
                            store(&lhs, val, &mut reader.env, &mut r_node_env)?;
                            state.nodes[node_index] = r_node_env;
                            state.push_to_local(node_index, Runnable::Record(reader));
                        }
                    }
                    record.pc = *next;
                }
            }
            Label::Recv(lhs, chan_expr, next) => {
                let cid = legacy_operand(&local_env, &node_env, chan_expr, &program.id_to_name)?
                    .as_channel()?;
                if cid.node != record.node {
                    return Err(RuntimeError::RemoteChannelRead);
                }

                let chan = state
                    .channels
                    .get_mut(&cid)
                    .ok_or(RuntimeError::ChannelNotFound(cid.id))?;

                if let Some(val) = chan.buffer.pop_front() {
                    store(lhs, val, &mut local_env, &mut node_env)?;
                    record.pc = *next;
                } else {
                    // Block Reader
                    let node_id = record.node;
                    record.env = local_env;
                    record.pc = *next; // When woke, proceed to next
                    chan.push_waiting_reader(record, lhs.clone());
                    state.nodes[node_id.index] = node_env;
                    return Ok(None); // Stop execution
                }
            }
            Label::Pause(next) => {
                let node_id = record.node;
                record.env = local_env;
                record.pc = *next;
                state.push_to_local(node_id.index, Runnable::Record(record));
                state.nodes[node_id.index] = node_env;
                return Ok(None); // Yield
            }
            Label::SpinAwait(expr, next) => {
                if legacy_operand(&local_env, &node_env, expr, &program.id_to_name)?.as_bool()? {
                    record.pc = *next;
                } else {
                    let node_id = record.node;
                    record.env = local_env;
                    state.push_to_local(node_id.index, Runnable::Record(record));
                    state.nodes[node_id.index] = node_env;
                    return Ok(None); // Yield
                }
            }
            Label::Instr(_, _)
            | Label::MakeChannel(_, _, _)
            | Label::SetTimer(_, _, _)
            | Label::MakeFifoLink(_, _, _)
            | Label::UniqueId(_, _)
            | Label::Cond(_, _, _)
            | Label::Return(_)
            | Label::Print(_, _)
            | Label::PersistData(_, _, _)
            | Label::RetrieveData(_, _, _)
            | Label::DiscardData(_)
            | Label::Break(_)
            | Label::ForLoopIn(_, _, _, _, _)
            | Label::TraceEnter(_, _, _, _)
            | Label::TraceExit(_, _, _, _)
            | Label::TraceDispatch(_, _, _) => {
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

pub fn exec<H: HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    record: Record<H>,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl StreamRng,
) -> Result<Option<ClientOpResult<H>>, RuntimeError> {
    if !program.compiled.covers(program.cfg.graph.len()) {
        return exec_legacy::<H, L, F>(
            state,
            logger,
            program,
            record,
            snapshot,
            feedback,
            policy,
            purgatory_config,
            rng,
        );
    }
    let mut tally = InterpreterTally::new();
    let result = exec_ops::<H, L, F>(
        state,
        logger,
        program,
        record,
        snapshot,
        feedback,
        policy,
        purgatory_config,
        rng,
        &mut tally,
    );
    tally.flush();
    result
}

/// What an operation left the loop to do.
enum Flow<H: HashPolicy> {
    Next(usize),
    Return(Value<H>),
    /// A channel or yield operation, which only a record's loop runs.
    RecordOnly,
}

#[inline(always)]
fn store_dest<H: HashPolicy>(
    dest: Dest,
    val: Value<H>,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
) {
    match dest {
        Dest::Local(idx) => set_local(local_env, idx, val),
        Dest::Node(idx) => node_env.set(idx, val),
    }
}

/// The callee of a call whose name was not resolved when the program was
/// decoded, with the errors the name lookup gives.
fn resolve_by_name<'p>(
    program: &'p Program,
    func_name: &str,
) -> Result<&'p FunctionInfo, RuntimeError> {
    let func_name_id = program
        .func_name_to_id
        .get(func_name)
        .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.to_string()))?;
    program
        .rpc
        .get(func_name_id)
        .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.to_string()))
}

#[inline(always)]
fn callee<'p>(
    program: &'p Program,
    index: u32,
    func_name: &str,
    t: &mut InterpreterTally,
) -> Result<&'p FunctionInfo, RuntimeError> {
    match program.compiled.call_functions.get(index as usize) {
        Some(info) => {
            t.call_targets_indexed += 1;
            Ok(info)
        }
        None => {
            t.call_targets_fallback += 1;
            resolve_by_name(program, func_name)
        }
    }
}

/// Runs every operation a synchronous function and an asynchronous record
/// share. Effects happen in the order the matching label arm of
/// `execute_common_label` performs them.
#[inline(always)]
fn run_common_op<H: HashPolicy, L: Logger, F: Feedback>(
    op: &Op,
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    node_id: NodeId,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    causal_operation_id: Option<i32>,
    pending_trace_id: &mut Option<i64>,
    pending_trace_payload: &mut Option<Box<str>>,
    rng: &mut impl StreamRng,
    t: &mut InterpreterTally,
) -> Result<Flow<H>, RuntimeError> {
    let names = &program.id_to_name;
    match op {
        Op::AssignLocal { slot, next, rhs } => {
            let v = cvalue(local_env, node_env, rhs, names, t)?;
            set_local(local_env, *slot, v);
            Ok(Flow::Next(*next as usize))
        }
        Op::AssignNode { slot, next, rhs } => {
            let v = cvalue(local_env, node_env, rhs, names, t)?;
            node_env.set(*slot, v);
            Ok(Flow::Next(*next as usize))
        }
        Op::CondLocal { slot, then, els } => {
            let taken = borrowed(local_env.get(*slot), t).as_bool()?;
            Ok(Flow::Next(if taken { *then } else { *els } as usize))
        }
        Op::CondNode { slot, then, els } => {
            let taken = borrowed(node_env.get(*slot), t).as_bool()?;
            Ok(Flow::Next(if taken { *then } else { *els } as usize))
        }
        Op::Cond { cond, then, els } => {
            let taken = coperand(local_env, node_env, cond, names, t)?.as_bool()?;
            Ok(Flow::Next(if taken { *then } else { *els } as usize))
        }
        Op::Goto(target) => Ok(Flow::Next(*target as usize)),
        Op::Return(rhs) => Ok(Flow::Return(cvalue(local_env, node_env, rhs, names, t)?)),
        Op::SyncCall(call) => {
            let func_info = callee(program, call.callee, &call.name, t)?;
            if !func_info.is_sync {
                return Err(RuntimeError::SyncCallToAsyncFunction(call.name.clone()));
            }
            let mut builder = FrameBuilder::<H>::new(func_info);
            for a in &call.args {
                builder.push(cvalue(local_env, node_env, a, names, t)?);
            }
            let mut callee_local = builder.finish(func_info);
            let val = run_sync_ops::<H, L, F>(
                state,
                logger,
                program,
                &mut callee_local,
                node_env,
                func_info.entry,
                node_id,
                snapshot,
                feedback,
                policy,
                purgatory_config,
                causal_operation_id,
                rng,
                t,
            )?;
            store_dest(call.dest, val, local_env, node_env);
            Ok(Flow::Next(call.next as usize))
        }
        Op::Async(call) => run_async_op(
            call,
            state,
            program,
            local_env,
            node_env,
            node_id,
            policy,
            purgatory_config,
            causal_operation_id,
            pending_trace_id,
            pending_trace_payload,
            rng,
            t,
        )
        .map(Flow::Next),
        Op::MakeChannel { dest, next } => {
            let cid = ChannelId {
                node: node_id,
                id: state.alloc_channel_id(),
            };
            state.insert_channel(cid);
            store_dest(*dest, Value::channel(cid), local_env, node_env);
            Ok(Flow::Next(*next as usize))
        }
        Op::MakeFifoLink { dest, next, peer } => {
            let peer = coperand(local_env, node_env, peer, names, t)?.as_node()?;
            let link_id = state.alloc_link_id();
            state.link_meta.insert(link_id, (node_id, peer));
            state.link_send_seq.insert(link_id, 0);
            state.link_deliver_seq.insert(link_id, 0);
            store_dest(*dest, Value::fifo_link(link_id, peer), local_env, node_env);
            Ok(Flow::Next(*next as usize))
        }
        Op::SetTimer { dest, next, label } => {
            let cid = ChannelId {
                node: node_id,
                id: state.alloc_channel_id(),
            };
            state.insert_channel(cid);
            store_dest(*dest, Value::channel(cid), local_env, node_env);
            let timer = Timer {
                pc: *next as usize,
                node: node_id,
                channel: cid,
                priority: policy.sample(rng, RunnableCategory::Timer),
                label: label.clone(),
            };
            state.push_runnable(Runnable::Timer(timer));
            Ok(Flow::Next(*next as usize))
        }
        Op::UniqueId { dest, next } => {
            let id = state.alloc_unique_id();
            store_dest(*dest, Value::int(id as i64), local_env, node_env);
            Ok(Flow::Next(*next as usize))
        }
        Op::Print { value, next } => {
            let val = coperand(local_env, node_env, value, names, t)?;
            let content_end = print_into(&val, logger.log_text());
            logger.log(LogEntry {
                node: node_id,
                content_end,
                step: state.crash_info.current_step,
            });
            Ok(Flow::Next(*next as usize))
        }
        Op::PersistData {
            type_id,
            next,
            value,
        } => {
            let val = cvalue(local_env, node_env, value, names, t)?;
            state.persisted_data.insert(node_id.index, (*type_id, val));
            Ok(Flow::Next(*next as usize))
        }
        Op::RetrieveData {
            type_id,
            dest,
            next,
        } => {
            let result = match state.persisted_data.get(&node_id.index) {
                Some((stored_tid, val)) => {
                    if stored_tid != type_id {
                        return Err(RuntimeError::PersistTypeMismatch {
                            stored: stored_tid.0,
                            expected: type_id.0,
                        });
                    }
                    Value::option_some(val.clone())
                }
                None => Value::option_none(),
            };
            store_dest(*dest, result, local_env, node_env);
            Ok(Flow::Next(*next as usize))
        }
        Op::DiscardData(next) => {
            state.persisted_data.remove(&node_id.index);
            Ok(Flow::Next(*next as usize))
        }
        Op::ForLoopIn(fl) => run_for_loop_in(fl, local_env, node_env, names, t).map(Flow::Next),
        Op::TraceEnter(te) => run_trace_enter(
            te,
            state,
            logger,
            local_env,
            node_env,
            node_id,
            names,
            causal_operation_id,
            pending_trace_id,
            pending_trace_payload,
            t,
        )
        .map(Flow::Next),
        Op::TraceExit(tx) => run_trace_exit(
            tx,
            state,
            logger,
            local_env,
            node_env,
            node_id,
            names,
            causal_operation_id,
            t,
        )
        .map(Flow::Next),
        Op::TraceDispatch(td) => Ok(Flow::Next(run_trace_dispatch(
            td,
            state,
            logger,
            local_env,
            node_env,
            node_id,
            names,
            causal_operation_id,
            pending_trace_id,
            pending_trace_payload,
            t,
        ))),
        Op::Send(_) | Op::Recv(_) | Op::Pause(_) | Op::SpinAwait { .. } => Ok(Flow::RecordOnly),
    }
}

#[inline(never)]
fn run_async_op<H: HashPolicy>(
    call: &AsyncOp,
    state: &mut State<H>,
    program: &Program,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    node_id: NodeId,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    causal_operation_id: Option<i32>,
    pending_trace_id: &mut Option<i64>,
    pending_trace_payload: &mut Option<Box<str>>,
    rng: &mut impl StreamRng,
    t: &mut InterpreterTally,
) -> Result<usize, RuntimeError> {
    let names = &program.id_to_name;
    let (target_node, link_id) =
        coperand(local_env, node_env, &call.target, names, t)?.as_rpc_target()?;
    let mut arg_vals: EcoVec<Value<H>> = EcoVec::with_capacity(call.args.len());
    for a in &call.args {
        arg_vals.push(cvalue(local_env, node_env, a, names, t)?);
    }

    let chan_id = ChannelId {
        node: node_id,
        id: state.alloc_channel_id(),
    };

    state.insert_channel(chan_id);
    store_dest(call.dest, Value::channel(chan_id), local_env, node_env);

    let func_info = callee(program, call.callee, &call.name, t)?;
    let callee_locals = build_frame(func_info, &arg_vals);

    // Tag with FIFO sequence number if routed through a FIFO link.
    // Allocates the next sender-side seq for this link.
    let link_seq = link_id.map(|lid| {
        let next = state.link_send_seq.get(&lid).copied().unwrap_or(0);
        state.link_send_seq.insert(lid, next + 1);
        (lid, next)
    });

    let send_ordinal = state.next_send_ordinal(node_id);
    let drawn_priority = policy.sample(rng, RunnableCategory::Record);
    let mut new_record = Record {
        pc: func_info.entry,
        node: target_node,
        origin_node: node_id,
        continuation: Continuation::Async { chan_id },
        entry_pc: func_info.entry,
        initial_args: arg_vals,
        entry_func: func_info.name,
        env: callee_locals,
        priority: state.record_priority(causal_operation_id, drawn_priority),
        causal_operation_id,
        trace_id: pending_trace_id.take(),
        trace_payload: pending_trace_payload.take(),
        link_seq,
        origin_incarnation: state.incarnation(node_id),
        bias: DeliveryBias::NONE,
        timer_entry: None,
        send_ordinal,
        receiver_token_at_send: state.node_state_token(target_node),
    };

    match purgatory_hold_steps(state, purgatory_config, target_node, rng) {
        Some(duration) => {
            let release_step = state.crash_info.current_step + duration;
            new_record.bias.insert(DeliveryBias::DELAYED);
            state.delay_runnable(release_step, Runnable::Record(new_record));
        }
        None => state.push_runnable(Runnable::Record(new_record)),
    }
    Ok(call.next as usize)
}

#[inline(never)]
fn run_for_loop_in<H: HashPolicy>(
    fl: &ForLoopInOp,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    names: &HashMap<NameId, String>,
    t: &mut InterpreterTally,
) -> Result<usize, RuntimeError> {
    let Some(iter_slot_idx) = fl.iter_slot else {
        return Err(RuntimeError::InvalidIteratorState);
    };

    let col_val = {
        let current = local_env.get(iter_slot_idx).clone();
        if matches!(current.kind, ValueKind::Unit) {
            let original_collection = cvalue(local_env, node_env, &fl.collection, names, t)?;
            set_local(local_env, iter_slot_idx, original_collection.clone());
            original_collection
        } else {
            current
        }
    };
    let col_val = col_val.into_map_form();

    match col_val.kind {
        ValueKind::List(l) => {
            if l.is_empty() {
                set_local(local_env, iter_slot_idx, Value::unit());
                Ok(fl.next as usize)
            } else {
                let item = l.first().ok_or(RuntimeError::EmptyCollection)?.clone();
                let new_l = Value::list(ValueSeq::from(&l[1..]));
                set_local(local_env, iter_slot_idx, new_l);

                store_dest(fl.dest, item, local_env, node_env);
                Ok(fl.body as usize)
            }
        }
        ValueKind::Map(m) => {
            if m.is_empty() {
                set_local(local_env, iter_slot_idx, Value::unit());
                Ok(fl.next as usize)
            } else {
                let (k, v) = m.iter().next().ok_or(RuntimeError::EmptyCollection)?;
                let k = k.clone();
                let v = v.clone();

                let new_m = m.without(&k);
                set_local(local_env, iter_slot_idx, Value::map(new_m));

                let pair = Value::tuple(ValueSeq::from([k, v]));
                store_dest(fl.dest, pair, local_env, node_env);
                Ok(fl.body as usize)
            }
        }
        _ => Err(RuntimeError::ForLoopNotCollection {
            got: col_val.type_name(),
        }),
    }
}

#[inline(never)]
fn run_trace_enter<H: HashPolicy, L: Logger>(
    te: &TraceEnterOp,
    state: &mut State<H>,
    logger: &mut L,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    node_id: NodeId,
    names: &HashMap<NameId, String>,
    causal_operation_id: Option<i32>,
    pending_trace_id: &mut Option<i64>,
    pending_trace_payload: &mut Option<Box<str>>,
    t: &mut InterpreterTally,
) -> Result<usize, RuntimeError> {
    let pending_id = pending_trace_id.take();
    let carried = pending_trace_payload.take();
    let id = pending_id.unwrap_or_else(|| state.alloc_unique_id() as i64);
    store_dest(te.dest, Value::int(id), local_env, node_env);
    // A carried payload was formatted by the dispatch that allocated the
    // pending id, from the argument values this frame's parameters were
    // built from, so it is the text formatting them here would produce.
    let out = logger.trace_text();
    match (pending_id, carried) {
        (Some(_), Some(text)) => {
            util_stats::record_trace_enter_payload(true);
            out.push_str(&text);
        }
        _ => {
            util_stats::record_trace_enter_payload(false);
            write_trace_payload(
                te.params
                    .iter()
                    .map(|e| cvalue(local_env, node_env, e, names, t)),
                out,
            );
        }
    }
    let payload_end = out.len();
    logger.log_trace(TraceEntry {
        node: node_id,
        function_name: te.func_name.clone(),
        kind: TraceKind::Enter,
        payload_end,
        schedulable_count: state.total_runnable_count(),
        step: state.crash_info.current_step,
        trace_id: id,
        causal_operation_id: causal_operation_id.map(|id| id as i64),
    });
    Ok(te.next as usize)
}

#[inline(never)]
fn run_trace_exit<H: HashPolicy, L: Logger>(
    tx: &TraceExitOp,
    state: &mut State<H>,
    logger: &mut L,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    node_id: NodeId,
    names: &HashMap<NameId, String>,
    causal_operation_id: Option<i32>,
    t: &mut InterpreterTally,
) -> Result<usize, RuntimeError> {
    let trace_id = cvalue(local_env, node_env, &tx.trace_id, names, t)?.as_int()?;
    let return_val = cvalue(local_env, node_env, &tx.return_value, names, t)?;
    let out = logger.trace_text();
    write_trace_payload(std::iter::once(Ok(return_val)), out);
    let payload_end = out.len();
    logger.log_trace(TraceEntry {
        node: node_id,
        function_name: tx.func_name.clone(),
        kind: TraceKind::Exit,
        payload_end,
        schedulable_count: state.total_runnable_count(),
        step: state.crash_info.current_step,
        trace_id,
        causal_operation_id: causal_operation_id.map(|id| id as i64),
    });
    Ok(tx.next as usize)
}

#[inline(never)]
fn run_trace_dispatch<H: HashPolicy, L: Logger>(
    td: &TraceDispatchOp,
    state: &mut State<H>,
    logger: &mut L,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    node_id: NodeId,
    names: &HashMap<NameId, String>,
    causal_operation_id: Option<i32>,
    pending_trace_id: &mut Option<i64>,
    pending_trace_payload: &mut Option<Box<str>>,
    t: &mut InterpreterTally,
) -> usize {
    let id = state.alloc_unique_id() as i64;
    *pending_trace_id = Some(id);
    let out = logger.trace_text();
    let start = out.len();
    write_trace_payload(
        td.params
            .iter()
            .map(|e| cvalue(local_env, node_env, e, names, t)),
        out,
    );
    *pending_trace_payload = Some(Box::from(out.str_from(start)));
    let payload_end = out.len();
    logger.log_trace(TraceEntry {
        node: node_id,
        function_name: td.func_name.clone(),
        kind: TraceKind::Dispatch,
        payload_end,
        schedulable_count: state.total_runnable_count(),
        step: state.crash_info.current_step,
        trace_id: id,
        causal_operation_id: causal_operation_id.map(|id| id as i64),
    });
    td.next as usize
}

/// Runs a synchronous function from its decoded operations until it returns.
fn run_sync_ops<H: HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
    start_pc: usize,
    node_id: NodeId,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    causal_operation_id: Option<i32>,
    rng: &mut impl StreamRng,
    t: &mut InterpreterTally,
) -> Result<Value<H>, RuntimeError> {
    let ops = &program.compiled.ops;
    let mut pc = start_pc;
    let mut prev_pc = pc;
    let mut pending_trace_id = None;
    let mut pending_trace_payload = None;
    loop {
        if pc != prev_pc {
            F::record_transition(feedback, prev_pc, pc, snapshot);
            prev_pc = pc;
        }
        t.label_execs += 1;
        match run_common_op::<H, L, F>(
            &ops[pc],
            state,
            logger,
            program,
            local_env,
            node_env,
            node_id,
            snapshot,
            feedback,
            policy,
            purgatory_config,
            causal_operation_id,
            &mut pending_trace_id,
            &mut pending_trace_payload,
            rng,
            t,
        )? {
            Flow::Next(next) => pc = next,
            Flow::Return(val) => return Ok(val),
            Flow::RecordOnly => {
                return Err(RuntimeError::UnsupportedSyncInstruction(format!(
                    "{:?}",
                    program.cfg.get_label(pc)
                )));
            }
        }
    }
}

/// Runs a record from its decoded operations until it returns or yields.
fn exec_ops<H: HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    mut record: Record<H>,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl StreamRng,
    t: &mut InterpreterTally,
) -> Result<Option<ClientOpResult<H>>, RuntimeError> {
    let causal_operation_id = record.causal_operation_id;
    let mut pending_trace_id = record.trace_id;
    // The payload moves out rather than being copied: only the entry row
    // reads it, so a record stored again after that row carries none, and a
    // re-delivery of it formats its entry row from the parameters.
    let mut pending_trace_payload = record.trace_payload.take();
    let mut local_env = record.env;
    let mut node_env = state.nodes[record.node.index].clone();

    let mut prev_pc = record.pc;

    // `record.pc == record.entry_pc` is true only on the first exec entry of a
    // delivery; Recv/Pause re-push with pc advanced past entry.
    if F::CAPTURES_TIMELINE
        && record.pc == record.entry_pc
        && program.compiled.server_role == Some(record.node.role)
    {
        F::note_delivery(feedback, record.node, record.entry_pc);
    }

    let ops = &program.compiled.ops;
    let names = &program.id_to_name;
    loop {
        let current_pc = record.pc;
        if current_pc != prev_pc {
            F::record_transition(feedback, prev_pc, current_pc, snapshot);
            prev_pc = current_pc;
        }
        t.label_execs += 1;

        match &ops[current_pc] {
            Op::Send(send) => {
                let cid = coperand(&local_env, &node_env, &send.chan, names, t)?.as_channel()?;
                let val = cvalue(&local_env, &node_env, &send.value, names, t)?;
                let next = send.next as usize;
                if cid.node != record.node {
                    let cs = Runnable::ChannelSend {
                        target: cid.node,
                        channel: cid,
                        message: val,
                        origin_node: record.node,
                        pc: next,
                        priority: policy.sample(rng, RunnableCategory::ChannelSend),
                    };
                    match purgatory_hold_steps(state, purgatory_config, cid.node, rng) {
                        Some(duration) => {
                            let release_step = state.crash_info.current_step + duration;
                            state.delay_runnable(release_step, cs);
                        }
                        None => state.push_runnable(cs),
                    }
                    record.pc = next;
                } else {
                    let chan = state
                        .channels
                        .get_mut(&cid)
                        .ok_or(RuntimeError::ChannelNotFound(cid.id))?;

                    match chan.pop_waiting_reader() {
                        None => chan.buffer.push_back(val),
                        Some((mut reader, lhs)) => {
                            let node_index = reader.node.index;
                            let mut r_node_env = state.nodes[node_index].clone();
                            store(&lhs, val, &mut reader.env, &mut r_node_env)?;
                            state.nodes[node_index] = r_node_env;
                            state.push_to_local(node_index, Runnable::Record(reader));
                        }
                    }
                    record.pc = next;
                }
            }
            Op::Recv(recv) => {
                let cid = coperand(&local_env, &node_env, &recv.chan, names, t)?.as_channel()?;
                if cid.node != record.node {
                    return Err(RuntimeError::RemoteChannelRead);
                }

                let chan = state
                    .channels
                    .get_mut(&cid)
                    .ok_or(RuntimeError::ChannelNotFound(cid.id))?;

                if let Some(val) = chan.buffer.pop_front() {
                    store_dest(recv.dest, val, &mut local_env, &mut node_env);
                    record.pc = recv.next as usize;
                } else {
                    let node_id = record.node;
                    record.env = local_env;
                    record.pc = recv.next as usize;
                    chan.push_waiting_reader(record, recv.lhs.clone());
                    state.nodes[node_id.index] = node_env;
                    return Ok(None);
                }
            }
            Op::Pause(next) => {
                let node_id = record.node;
                record.env = local_env;
                record.pc = *next as usize;
                state.push_to_local(node_id.index, Runnable::Record(record));
                state.nodes[node_id.index] = node_env;
                return Ok(None);
            }
            Op::SpinAwait { cond, next } => {
                if coperand(&local_env, &node_env, cond, names, t)?.as_bool()? {
                    record.pc = *next as usize;
                } else {
                    let node_id = record.node;
                    record.env = local_env;
                    state.push_to_local(node_id.index, Runnable::Record(record));
                    state.nodes[node_id.index] = node_env;
                    return Ok(None);
                }
            }
            op => match run_common_op::<H, L, F>(
                op,
                state,
                logger,
                program,
                &mut local_env,
                &mut node_env,
                record.node,
                snapshot,
                feedback,
                policy,
                purgatory_config,
                causal_operation_id,
                &mut pending_trace_id,
                &mut pending_trace_payload,
                rng,
                t,
            )? {
                Flow::Next(next) => record.pc = next,
                Flow::Return(val) => {
                    let node_id = record.node;
                    state.nodes[node_id.index] = node_env;
                    return Ok(record.continuation.call(state, val));
                }
                Flow::RecordOnly => {
                    unreachable!("channel and yield operations are matched before the shared ones")
                }
            },
        }
    }
}

#[cfg(test)]
mod test;
