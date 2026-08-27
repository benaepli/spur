use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::make_local_env;
use crate::simulator::core::exec::{exec, exec_sync_on_node};
use crate::simulator::core::partition::{activate_partition, heal_partition};
use crate::simulator::core::queue_selector::{
    QueueInfo, QueueSelection, QueueSelector, WithinQueueSelector,
};
use crate::simulator::core::state::{
    Continuation, Logger, NodeId, PurgatoryConfig, Record, Runnable, RunnableCategory,
    SchedulePolicy, ScheduleResult, State,
};
use crate::simulator::core::values::{Env, Value};
use crate::simulator::coverage::GlobalState;
use crate::simulator::feedback::Feedback;
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::Topology;
use crate::simulator::path::TopologyInfo;
use crate::simulator::rng::{Stream, StreamRng};
use crate::simulator::util_stats;
use crate::simulator::util_stats::DeliveryBias;
use imbl::OrdSet;
use log::warn;
use rand::Rng;

/// A resolved deliver reservation. Runnables matching this are excluded from scheduling
/// until the deliver's DAG dependencies are met.
#[derive(Debug, Clone)]
pub struct Reservation {
    pub entry_pc: Vertex,
    pub from: Option<usize>,
    pub to: Option<usize>,
}

impl Reservation {
    pub fn matches<H: HashPolicy>(&self, runnable: &Runnable<H>) -> bool {
        match runnable {
            Runnable::Record(r) => {
                r.entry_pc == self.entry_pc
                    && self.to.is_none_or(|t| r.node.index == t)
                    && self.from.is_none_or(|f| r.origin_node.index == f)
            }
            // ChannelSend runnables are not matchable by delivers.
            // All VR inter-node messages are RPCs (Record runnables).
            _ => false,
        }
    }
}

/// A FIFO-tagged runnable is only deliverable when its sequence number matches
/// the link's next-expected-deliver counter. All other runnables (non-FIFO or
/// non-Record) are always eligible.
fn is_fifo_blocked<H: HashPolicy>(
    runnable: &Runnable<H>,
    link_deliver_seq: &imbl::HashMap<crate::simulator::core::values::LinkId, u32>,
) -> bool {
    if let Runnable::Record(r) = runnable
        && let Some((link_id, seq)) = r.link_seq {
            let expected = link_deliver_seq.get(&link_id).copied().unwrap_or(0);
            return seq != expected;
        }
    false
}

/// Per-node flag: the node originated a remote message that has not been
/// delivered yet, counting sends still waiting out a purgatory delay. Indexed
/// by node index; nodes added after the queues were sized read as false.
fn nodes_with_in_flight_send<H: HashPolicy>(state: &State<H>) -> Vec<bool> {
    let mut senders = vec![false; state.local_queues.len()];
    let mut mark = |r: &Runnable<H>| {
        let origin = match r {
            Runnable::Record(rec) if rec.origin_node != rec.node => rec.origin_node.index,
            Runnable::ChannelSend {
                origin_node,
                target,
                ..
            } if origin_node != target => origin_node.index,
            _ => return,
        };
        if let Some(slot) = senders.get_mut(origin) {
            *slot = true;
        }
    };
    for r in &state.network_queue {
        mark(r);
    }
    for (_, r) in &state.purgatory {
        mark(r);
    }
    senders
}

/// Score a runnable in [0, 1] by combining novelty and priority. For Recover
/// events targeting a currently-crashed node, `quick_fire_multiplier` increases
/// the weight of priority relative to novelty while keeping the result in [0, 1].
fn score_runnable<H: HashPolicy, F: Feedback>(
    r: &Runnable<H>,
    feedback: &F::Local,
    snapshot: &F::Snapshot,
    currently_crashed: &OrdSet<NodeId>,
    quick_fire_multiplier: f64,
) -> f64 {
    let novelty = F::runnable_novelty(feedback, r, snapshot);
    let priority = r.priority();
    let is_quick_fire =
        matches!(r, Runnable::Recover { node_id, .. } if currently_crashed.contains(node_id));
    if is_quick_fire {
        let w = 0.75 * quick_fire_multiplier;
        (0.25 * novelty + w * priority) / (0.25 + w)
    } else {
        0.25 * novelty + 0.75 * priority
    }
}

/// The priority-only share of `score_runnable` (novelty zeroed out, same
/// quick-fire weighting). Only used by the opt-in utilization probe below.
fn priority_component<H: HashPolicy>(
    r: &Runnable<H>,
    currently_crashed: &OrdSet<NodeId>,
    quick_fire_multiplier: f64,
) -> f64 {
    let priority = r.priority();
    let is_quick_fire =
        matches!(r, Runnable::Recover { node_id, .. } if currently_crashed.contains(node_id));
    if is_quick_fire {
        let w = 0.75 * quick_fire_multiplier;
        (w * priority) / (0.25 + w)
    } else {
        0.75 * priority
    }
}

/// Where a runnable sits across the three kinds of queue, so the one the
/// scoring function ranks first can be compared against the one a step runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum QueueSlot {
    Local(usize, usize),
    Network(usize),
    Timer(usize),
}

impl QueueSlot {
    fn same_queue(self, other: Self) -> bool {
        match (self, other) {
            (QueueSlot::Local(a, _), QueueSlot::Local(b, _)) => a == b,
            (QueueSlot::Network(_), QueueSlot::Network(_)) => true,
            (QueueSlot::Timer(_), QueueSlot::Timer(_)) => true,
            _ => false,
        }
    }
}

/// The top-ranked runnable at one scheduling point, and whether anything is
/// keeping it from being run.
enum PreferredSlot {
    NoneEligible,
    Blocked(util_stats::SteerOutcome),
    Eligible(QueueSlot),
}

/// Observation-only picture of what the scoring function wanted at one
/// scheduling point, resolved against what the point went on to do.
struct SteerPreference {
    expressed: bool,
    preferred: PreferredSlot,
}

impl SteerPreference {
    fn outcome(&self, chosen: Option<QueueSlot>) -> util_stats::SteerOutcome {
        match self.preferred {
            PreferredSlot::NoneEligible => util_stats::SteerOutcome::NoEligibleCandidates,
            PreferredSlot::Blocked(reason) => reason,
            PreferredSlot::Eligible(slot) => match chosen {
                None => util_stats::SteerOutcome::NoEligibleCandidates,
                Some(c) if c == slot => util_stats::SteerOutcome::Honored,
                Some(c) if c.same_queue(slot) => util_stats::SteerOutcome::SamplerChoseOther,
                Some(_) => util_stats::SteerOutcome::OtherQueue,
            },
        }
    }
}

/// A timer whose label is currently not permitted to fire; other runnables and
/// unlabeled timers are never gated this way.
fn timer_gate_blocks<H: HashPolicy>(state: &State<H>, r: &Runnable<H>) -> bool {
    match r {
        Runnable::Timer(t) => t
            .label
            .as_ref()
            .is_some_and(|l| !state.allowed_timers.contains(&(t.node.index, l.clone()))),
        _ => false,
    }
}

/// Rank every runnable in every queue by the same score the selectors use, and
/// report where the top-ranked one sits and what is in its way. Consumes no
/// RNG and touches no state; the cost is one pass over all queues, which is
/// why it is behind its own switch.
fn audit_steer_preference<H: HashPolicy, F: Feedback>(
    state: &State<H>,
    feedback: &F::Local,
    snapshot: &F::Snapshot,
    quick_fire_multiplier: f64,
    strict_timers: bool,
    is_ineligible: &impl Fn(&Runnable<H>) -> bool,
) -> SteerPreference {
    let currently_crashed = &state.crash_info.currently_crashed;
    let mut candidates: u64 = 0;
    let mut any_eligible = false;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_slot: Option<(QueueSlot, Option<util_stats::SteerOutcome>)> = None;
    let mut best_priority = f64::NEG_INFINITY;
    let mut best_priority_slot: Option<QueueSlot> = None;

    {
        let mut visit = |slot: QueueSlot, r: &Runnable<H>, timer_gated: bool| {
            let blocked = if is_ineligible(r) {
                Some(util_stats::SteerOutcome::BlockedByOrder)
            } else if timer_gated {
                Some(util_stats::SteerOutcome::BlockedByTimerGate)
            } else {
                None
            };
            candidates += 1;
            any_eligible |= blocked.is_none();
            let score = score_runnable::<H, F>(
                r,
                feedback,
                snapshot,
                currently_crashed,
                quick_fire_multiplier,
            );
            if score > best_score {
                best_score = score;
                best_slot = Some((slot, blocked));
            }
            let priority = priority_component(r, currently_crashed, quick_fire_multiplier);
            if priority > best_priority {
                best_priority = priority;
                best_priority_slot = Some(slot);
            }
        };

        for (node_idx, queue) in state.local_queues.iter().enumerate() {
            for (i, r) in queue.iter().enumerate() {
                visit(QueueSlot::Local(node_idx, i), r, false);
            }
        }
        for (i, r) in state.network_queue.iter().enumerate() {
            visit(QueueSlot::Network(i), r, false);
        }
        for (i, r) in state.timer_queue.iter().enumerate() {
            let gated = strict_timers && timer_gate_blocks(state, r);
            visit(QueueSlot::Timer(i), r, gated);
        }
    }

    let expressed = candidates > 1 && best_slot.map(|(s, _)| s) != best_priority_slot;
    let preferred = match best_slot {
        _ if !any_eligible => PreferredSlot::NoneEligible,
        Some((_, Some(reason))) => PreferredSlot::Blocked(reason),
        Some((slot, None)) => PreferredSlot::Eligible(slot),
        None => PreferredSlot::NoneEligible,
    };
    SteerPreference {
        expressed,
        preferred,
    }
}

/// Select an eligible item from a single queue.
///
/// `Tournament` samples `k` indices uniformly and takes the highest-scoring
/// (near-greedy for typical k). `Proportional` uses Efraimidis-Spirakis weighted
/// reservoir sampling with weight `score^exponent`, giving exact proportional
/// selection in a single O(eligible) pass.
fn select_within_queue<H: HashPolicy, F: Feedback>(
    queue: &[Runnable<H>],
    eligible: &[usize],
    feedback: &F::Local,
    snapshot: &F::Snapshot,
    currently_crashed: &OrdSet<NodeId>,
    quick_fire_multiplier: f64,
    selector: &WithinQueueSelector,
    rng: &mut impl StreamRng,
) -> usize {
    if eligible.len() <= 1 {
        return eligible[0];
    }
    rng.use_stream(Stream::WithinQueue);

    // Observation-only utilization probe: would the greedy pick change if the
    // novelty/steer term were dropped? Compares the blended-score argmax with
    // the priority-only argmax (first index wins ties). Consumes no RNG and
    // does not influence the selection below.
    if util_stats::enabled() {
        let mut best_blend = f64::NEG_INFINITY;
        let mut best_blend_idx = eligible[0];
        let mut best_prio = f64::NEG_INFINITY;
        let mut best_prio_idx = eligible[0];
        for &i in eligible {
            let blend = score_runnable::<H, F>(
                &queue[i],
                feedback,
                snapshot,
                currently_crashed,
                quick_fire_multiplier,
            );
            let prio = priority_component(&queue[i], currently_crashed, quick_fire_multiplier);
            if blend > best_blend {
                best_blend = blend;
                best_blend_idx = i;
            }
            if prio > best_prio {
                best_prio = prio;
                best_prio_idx = i;
            }
        }
        util_stats::record_steer_evaluation(best_blend_idx != best_prio_idx);
    }

    match selector {
        WithinQueueSelector::Tournament { k } => {
            let k = (*k).max(1);
            let mut best_idx = eligible[rng.random_range(0..eligible.len())];
            let mut best_score = score_runnable::<H, F>(
                &queue[best_idx],
                feedback,
                snapshot,
                currently_crashed,
                quick_fire_multiplier,
            );
            for _ in 1..k.min(eligible.len()) {
                let i = eligible[rng.random_range(0..eligible.len())];
                let s = score_runnable::<H, F>(
                    &queue[i],
                    feedback,
                    snapshot,
                    currently_crashed,
                    quick_fire_multiplier,
                );
                if s > best_score {
                    best_idx = i;
                    best_score = s;
                }
            }
            best_idx
        }
        WithinQueueSelector::Proportional { exponent } => {
            // Efraimidis-Spirakis: argmax of (ln(u_i) / w_i) is exact weighted
            // sampling proportional to w_i. Both ln(u) (u in (0,1)) and w are
            // negative/positive respectively, so the largest key wins.
            //
            // Floor weight to keep zero-score items reachable; without this,
            // a score of exactly 0 would have 0 selection probability and a
            // score of 0 with exponent 0 would produce 0/0.
            let mut best_idx = eligible[0];
            let mut best_key = f64::NEG_INFINITY;
            for &i in eligible {
                let s = score_runnable::<H, F>(
                    &queue[i],
                    feedback,
                    snapshot,
                    currently_crashed,
                    quick_fire_multiplier,
                );
                let weight = s.powf(*exponent).max(1e-9);
                let u: f64 = rng.random();
                // u is in (0, 1); ln(u) is negative; key = ln(u) / weight is negative.
                // Higher weight means a key closer to 0 (larger), so argmax is correct.
                let key = u.ln() / weight;
                if key > best_key {
                    best_key = key;
                    best_idx = i;
                }
            }
            best_idx
        }
    }
}

pub fn schedule_runnable<H: HashPolicy, L: Logger, Q: QueueSelector, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    topology: &TopologyInfo,
    global_state: &GlobalState<F>,
    policy: &SchedulePolicy,
    strict_timers: bool,
    selector: &mut Q,
    within_queue: &WithinQueueSelector,
    quick_fire_multiplier: f64,
    purgatory_config: &PurgatoryConfig,
    reservations: &[Reservation],
    rng: &mut impl StreamRng,
) -> Result<ScheduleResult<H>, RuntimeError> {
    if state.all_queues_empty() {
        return Ok(ScheduleResult::None);
    }

    // Helper: check if a runnable is reserved OR FIFO-blocked. Both exclude the
    // item from scheduling via the same plumbing, so combine them here.
    let link_deliver_seq = state.link_deliver_seq.clone();
    let is_ineligible = |r: &Runnable<H>| {
        reservations.iter().any(|res| res.matches(r)) || is_fifo_blocked(r, &link_deliver_seq)
    };

    // Observation-only crash-anchor probe: is there a schedulable crash for a
    // node whose own message is still in flight, and does the step take it?
    // The senders vector is kept for the crash arm below so both sides of the
    // ratio are measured against the same queue contents.
    let in_flight_senders: Option<Vec<bool>> = if util_stats::enabled() {
        let crash_nodes: Vec<usize> = state
            .local_queues
            .iter()
            .enumerate()
            .filter(|(_, q)| {
                q.iter()
                    .any(|r| matches!(r, Runnable::Crash { .. }) && !is_ineligible(r))
            })
            .map(|(idx, _)| idx)
            .collect();
        if crash_nodes.is_empty() {
            util_stats::record_crash_anchor_offer(false, false);
            None
        } else {
            let senders = nodes_with_in_flight_send(state);
            let anchored = crash_nodes
                .iter()
                .any(|&idx| senders.get(idx).copied().unwrap_or(false));
            util_stats::record_crash_anchor_offer(true, anchored);
            Some(senders)
        }
    } else {
        None
    };

    // Observation-only steer-authority audit: what the scoring function ranks
    // first here, resolved below against what this step actually runs.
    let audit = util_stats::steer_audit_enabled().then(|| {
        audit_steer_preference::<H, F>(
            state,
            feedback,
            snapshot,
            quick_fire_multiplier,
            strict_timers,
            &is_ineligible,
        )
    });

    // Build QueueInfo, accounting for strict_timers eligibility AND reservations.
    // Subtract reserved items so the QueueSelector doesn't route to queues
    // where all items are reserved (wastes iterations in fully-constrained plans).
    let timer_queue_size = if strict_timers {
        state
            .timer_queue
            .iter()
            .filter(|r| {
                if is_ineligible(r) {
                    return false;
                }
                if let Runnable::Timer(t) = r {
                    t.label.as_ref().is_none_or(|l| {
                        state.allowed_timers.contains(&(t.node.index, l.clone()))
                    })
                } else {
                    true
                }
            })
            .count()
    } else {
        state.timer_queue.iter().filter(|r| !is_ineligible(r)).count()
    };

    let info = QueueInfo {
        local_queue_sizes: state
            .local_queues
            .iter()
            .map(|q| q.iter().filter(|r| !is_ineligible(r)).count())
            .collect(),
        network_queue_size: state.network_queue.iter().filter(|r| !is_ineligible(r)).count(),
        timer_queue_size,
        step: state.crash_info.current_step,
    };

    rng.use_stream(Stream::QueueChoice);
    let record_unscheduled = |audit: &Option<SteerPreference>| {
        if let Some(a) = audit {
            util_stats::record_steer_authority(a.expressed, a.outcome(None));
        }
    };

    let selection = match selector.select(&info, rng) {
        Some(s) => s,
        None => {
            record_unscheduled(&audit);
            return Ok(ScheduleResult::None);
        }
    };

    let (runnable, chosen_slot) = match selection {
        QueueSelection::Local(node_idx) => {
            let queue = &state.local_queues[node_idx];
            let eligible: Vec<usize> = (0..queue.len())
                .filter(|&i| !is_ineligible(&queue[i]))
                .collect();
            if eligible.is_empty() {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
            let idx = select_within_queue::<H, F>(
                queue,
                &eligible,
                feedback,
                snapshot,
                &state.crash_info.currently_crashed,
                quick_fire_multiplier,
                within_queue,
                rng,
            );
            (
                state.local_queues[node_idx].remove(idx),
                QueueSlot::Local(node_idx, idx),
            )
        }
        QueueSelection::Network => {
            let queue = &state.network_queue;
            let eligible: Vec<usize> = (0..queue.len())
                .filter(|&i| !is_ineligible(&queue[i]))
                .collect();
            if eligible.is_empty() {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
            let idx = select_within_queue::<H, F>(
                queue,
                &eligible,
                feedback,
                snapshot,
                &state.crash_info.currently_crashed,
                quick_fire_multiplier,
                within_queue,
                rng,
            );
            (state.network_queue.remove(idx), QueueSlot::Network(idx))
        }
        QueueSelection::Timer => {
            let queue = &state.timer_queue;
            let eligible: Vec<usize> = if strict_timers {
                (0..queue.len())
                    .filter(|&i| {
                        if is_ineligible(&queue[i]) {
                            return false;
                        }
                        if let Runnable::Timer(t) = &queue[i] {
                            t.label.as_ref().is_none_or(|l| {
                                state.allowed_timers.contains(&(t.node.index, l.clone()))
                            })
                        } else {
                            true
                        }
                    })
                    .collect()
            } else {
                (0..queue.len())
                    .filter(|&i| !is_ineligible(&queue[i]))
                    .collect()
            };
            if eligible.is_empty() {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
            let idx = select_within_queue::<H, F>(
                queue,
                &eligible,
                feedback,
                snapshot,
                &state.crash_info.currently_crashed,
                quick_fire_multiplier,
                within_queue,
                rng,
            );
            (state.timer_queue.remove(idx), QueueSlot::Timer(idx))
        }
    };

    if let Some(a) = &audit {
        util_stats::record_steer_authority(a.expressed, a.outcome(Some(chosen_slot)));
    }

    match runnable {
        Runnable::Crash { node_id, .. } => {
            if let Some(senders) = &in_flight_senders {
                util_stats::record_crash_anchor_apply(
                    senders.get(node_id.index).copied().unwrap_or(false),
                );
            }
            crash_node(state, node_id);
            Ok(ScheduleResult::Crash { node_id })
        }
        Runnable::Recover { node_id, .. } => {
            recover_crashed_node::<H, L, F>(
                state,
                logger,
                program,
                topology,
                node_id,
                global_state,
                snapshot,
                feedback,
                policy,
                purgatory_config,
                rng,
            )?;
            Ok(ScheduleResult::Recover { node_id })
        }
        Runnable::Partition { partition_type, .. } => {
            activate_partition(state, partition_type.clone());
            Ok(ScheduleResult::Partition { partition_type })
        }
        Runnable::Heal { .. } => {
            heal_partition(state);
            Ok(ScheduleResult::Heal)
        }
        Runnable::Timer(timer) => {
            if state.crash_info.currently_crashed.contains(&timer.node) {
                return Ok(ScheduleResult::None);
            }

            if let Some(mut chan) = state.channels.get(&timer.channel).cloned() {
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    let mut r_node_env = state.nodes[reader.node.index].clone();
                    if let Err(e) = crate::simulator::core::eval::store(
                        &lhs,
                        Value::<H>::unit(),
                        &mut reader.env,
                        &mut r_node_env,
                    ) {
                        log::warn!("Store failed in timer completion: {}", e);
                    }
                    let node_index = reader.node.index;
                    state.nodes[node_index] = r_node_env;
                    state.push_to_local(node_index, Runnable::Record(reader));
                } else {
                    chan.buffer.push_back(Value::<H>::unit());
                }
                state.channels.insert(timer.channel, chan);
            }
            if let Some(label) = timer.label {
                state
                    .allowed_timers
                    .remove(&(timer.node.index, label.clone()));
                Ok(ScheduleResult::TimerFired {
                    node_id: timer.node,
                    label,
                })
            } else {
                Ok(ScheduleResult::None)
            }
        }
        other => {
            let (src_node, dest_node) = match &other {
                Runnable::Record(r) => (r.origin_node, r.node),
                Runnable::ChannelSend {
                    origin_node,
                    target,
                    ..
                } => (*origin_node, *target),
                _ => unreachable!(),
            };

            if state.crash_info.currently_crashed.contains(&dest_node) {
                if let Runnable::Record(r) = other
                    && src_node != dest_node {
                        let mut r = r;
                        r.reset();
                        state.crash_info.queued_messages.push_back((dest_node, r));
                    }
                return Ok(ScheduleResult::None);
            }

            if state.partition_info.is_blocked(src_node, dest_node) {
                match other {
                    Runnable::Record(r) => {
                        let mut r = r;
                        r.reset();
                        state.partition_info.buffer_record(dest_node, r);
                    }
                    Runnable::ChannelSend {
                        channel,
                        message,
                        origin_node,
                        pc,
                        priority,
                        ..
                    } => {
                        state.partition_info.buffer_channel_send(
                            dest_node,
                            channel,
                            message,
                            origin_node,
                            pc,
                            priority,
                        );
                    }
                    _ => unreachable!(),
                }
                return Ok(ScheduleResult::None);
            }

            match other {
                Runnable::Record(r) => {
                    let record_entry_pc = r.entry_pc;
                    let record_origin = r.origin_node;
                    let record_dest = r.node;
                    // Bump the link's deliver counter so the next FIFO message
                    // in this link becomes schedulable.
                    if let Some((link_id, seq)) = r.link_seq {
                        state.link_deliver_seq.insert(link_id, seq + 1);
                    }
                    // Measure the effect of a message on its receiver: only a
                    // first entry into a handler for a message from another
                    // node counts, not the continuations it is re-queued as.
                    let message_entry = record_origin != record_dest && r.pc == record_entry_pc;
                    let entry_step = state.crash_info.current_step;
                    let probe = (util_stats::acted_fraction_enabled() && message_entry).then(|| {
                        let mut bias = r.bias;
                        if r.origin_incarnation != state.incarnation(record_origin) {
                            bias.insert(DeliveryBias::SENDER_RESTARTED);
                        }
                        (bias, state.node_state_token(record_dest))
                    });
                    let result = exec::<H, L, F>(
                        state,
                        logger,
                        program,
                        r,
                        snapshot,
                        feedback,
                        policy,
                        purgatory_config,
                        rng,
                    )?;
                    if let Some((bias, before)) = probe {
                        let acted = state.node_state_token(record_dest) != before;
                        util_stats::record_delivery(bias, acted);
                    }
                    if message_entry {
                        util_stats::record_message_entry(record_dest.index, entry_step);
                    }
                    match result {
                        Some(client_op) => Ok(ScheduleResult::ClientOp(client_op)),
                        None => Ok(ScheduleResult::RecordExecuted {
                            entry_pc: record_entry_pc,
                            origin_node: record_origin,
                            dest_node: record_dest,
                        }),
                    }
                }
                Runnable::ChannelSend {
                    channel, message, ..
                } => {
                    if let Some(mut chan) = state.channels.get(&channel).cloned() {
                        if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                            let mut r_node_env = state.nodes[reader.node.index].clone();
                            if let Err(e) = crate::simulator::core::eval::store(
                                &lhs,
                                message,
                                &mut reader.env,
                                &mut r_node_env,
                            ) {
                                log::warn!("Store failed in remote channel delivery: {}", e);
                            }
                            let node_index = reader.node.index;
                            state.nodes[node_index] = r_node_env;
                            state.push_to_local(node_index, Runnable::Record(reader));
                        } else {
                            chan.buffer.push_back(message);
                        }
                        state.channels.insert(channel, chan);
                    }
                    Ok(ScheduleResult::None)
                }
                _ => unreachable!(),
            }
        }
    }
}

fn crash_node<H: HashPolicy>(state: &mut State<H>, node_id: NodeId) {
    if state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is already crashed", node_id);
        return;
    }
    state.crash_info.currently_crashed.insert(node_id);

    let mut held: u64 = 0;
    let mut dropped: u64 = 0;

    // 1. Process local queue for crashed node: save external records, drop the rest
    let local = std::mem::take(&mut state.local_queues[node_id.index]);
    for task in local {
        if let Runnable::Record(record) = task
            && record.origin_node != record.node {
                let mut record = record;
                record.reset();
                held += 1;
                state
                    .crash_info
                    .queued_messages
                    .push_back((node_id, record));
            } else {
                dropped += 1;
            }
    }

    // 2. Filter network queue: remove items targeting the crashed node
    let net = std::mem::take(&mut state.network_queue);
    for task in net {
        match &task {
            Runnable::Record(r) if r.node == node_id => {
                if r.origin_node != r.node {
                    let mut r = r.clone();
                    r.reset();
                    held += 1;
                    state.crash_info.queued_messages.push_back((node_id, r));
                } else {
                    dropped += 1;
                }
            }
            Runnable::ChannelSend { target, .. } if *target == node_id => dropped += 1,
            Runnable::Crash { node_id: nid, .. } | Runnable::Recover { node_id: nid, .. }
                if *nid == node_id => {}
            _ => state.network_queue.push(task),
        }
    }

    util_stats::record_crash(node_id.index, held, dropped);

    // 3. Filter timer queue: remove timers for the crashed node
    let timers = std::mem::take(&mut state.timer_queue);
    for task in timers {
        if let Runnable::Timer(ref t) = task
            && t.node == node_id {
                continue;
            }
        state.timer_queue.push(task);
    }
}

fn recover_crashed_node<H: HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    topology: &TopologyInfo,
    node_id: NodeId,
    global_state: &GlobalState<F>,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl StreamRng,
) -> Result<(), RuntimeError> {
    if !state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is not crashed", node_id);
        return Ok(());
    }
    state.crash_info.currently_crashed.remove(&node_id);
    if let Some(inc) = state.incarnations.get_mut(node_id.index) {
        *inc = inc.saturating_add(1);
    }
    util_stats::record_recover(node_id.index, state.crash_info.current_step);
    F::note_recovery(feedback, node_id);

    state.nodes[node_id.index] = Env::<H>::default();
    reinit_node::<H, L, F>(
        topology,
        state,
        logger,
        program,
        node_id,
        global_state,
        snapshot,
        feedback,
        policy,
        purgatory_config,
        rng,
    )?;

    let queued = std::mem::take(&mut state.crash_info.queued_messages);
    for (dest, record) in queued {
        if dest == node_id {
            let mut record = record;
            record.bias.insert(DeliveryBias::RECEIVER_RESTARTED);
            state.push_runnable(Runnable::Record(record));
        } else {
            state.crash_info.queued_messages.push_back((dest, record));
        }
    }
    Ok(())
}

fn reinit_node<H: HashPolicy, L: Logger, F: Feedback>(
    topology: &TopologyInfo,
    state: &mut State<H>,
    logger: &mut L,
    prog: &Program,
    node_id: NodeId,
    global_state: &GlobalState<F>,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl StreamRng,
) -> Result<(), RuntimeError> {
    use crate::compiler::cfg::{SELF_SLOT, VarSlot};

    let init_fn = prog
        .get_func_by_name("Node.BASE_NODE_INIT")
        .ok_or_else(|| RuntimeError::MissingRequiredFunction("Node.BASE_NODE_INIT".to_string()))?;

    if let VarSlot::Node(self_idx, _) = SELF_SLOT {
        state.nodes[node_id.index].set(self_idx, Value::<H>::node(node_id));
    }

    let node_env = &state.nodes[node_id.index];
    let mut env = make_local_env(
        init_fn,
        vec![],
        &Env::<H>::default(),
        node_env,
        &prog.id_to_name,
    );

    exec_sync_on_node::<H, L, F>(
        state,
        logger,
        prog,
        &mut env,
        node_id,
        init_fn.entry,
        snapshot,
        feedback,
        policy,
        purgatory_config,
        rng,
    )?;

    recover_node::<H, L, F>(
        topology,
        state,
        logger,
        prog,
        node_id,
        global_state,
        snapshot,
        feedback,
        policy,
        purgatory_config,
        rng,
    )
}

fn recover_node<H: HashPolicy, L: Logger, F: Feedback>(
    topology: &TopologyInfo,
    state: &mut State<H>,
    logger: &mut L,
    prog: &Program,
    node_id: NodeId,
    _global_state: &GlobalState<F>,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl StreamRng,
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
        entry_pc: recover_fn.entry,
        initial_env: env.clone(),
        env,
        priority: policy.sample(rng, RunnableCategory::Record),
        causal_operation_id: None,
        trace_id: None,
        link_seq: None,
        origin_incarnation: state.incarnation(node_id),
        bias: DeliveryBias::NONE,
    };

    exec::<H, L, F>(
        state,
        logger,
        prog,
        record,
        snapshot,
        feedback,
        policy,
        purgatory_config,
        rng,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::feedback::NoFeedback;
    use crate::simulator::hash_utils::NoHashing;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn heal(priority: f64) -> Runnable<NoHashing> {
        Runnable::Heal { priority }
    }

    /// Score under default `score_runnable` parameters (no novelty signal,
    /// no quick-fire boost) is `0.25 + 0.75 * priority`.
    fn expected_score(priority: f64) -> f64 {
        0.25 + 0.75 * priority
    }

    #[test]
    fn proportional_selection_matches_expected_distribution() {
        let queue: Vec<Runnable<NoHashing>> = vec![
            heal(0.0), // score 0.25
            heal(0.5), // score 0.625
            heal(1.0), // score 1.00
        ];
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let crashed = OrdSet::new();
        let selector = WithinQueueSelector::Proportional { exponent: 1.0 };

        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        let trials = 50_000usize;
        let mut counts = [0usize; 3];
        for _ in 0..trials {
            let idx = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &crashed,
                1.0,
                &selector,
                &mut rng,
            );
            counts[idx] += 1;
        }

        let total_score: f64 = (0..3)
            .map(|i| expected_score(queue[i].priority()))
            .sum();
        for i in 0..3 {
            let expected = expected_score(queue[i].priority()) / total_score;
            let observed = counts[i] as f64 / trials as f64;
            // Binomial std error ≈ sqrt(p(1-p)/n); with n=50k and p ~0.5 that's
            // ~0.0022. Allow 0.015 (≈7σ) to keep the test robust.
            assert!(
                (observed - expected).abs() < 0.015,
                "bucket {}: expected ~{:.3}, observed {:.3} (n={})",
                i,
                expected,
                observed,
                trials,
            );
        }
    }

    #[test]
    fn proportional_with_zero_exponent_is_uniform() {
        let queue: Vec<Runnable<NoHashing>> = vec![heal(0.0), heal(0.5), heal(1.0)];
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let crashed = OrdSet::new();
        let selector = WithinQueueSelector::Proportional { exponent: 0.0 };

        let mut rng = StdRng::seed_from_u64(42);
        let trials = 30_000usize;
        let mut counts = [0usize; 3];
        for _ in 0..trials {
            let idx = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &crashed,
                1.0,
                &selector,
                &mut rng,
            );
            counts[idx] += 1;
        }
        for (i, &count) in counts.iter().enumerate() {
            let observed = count as f64 / trials as f64;
            assert!(
                (observed - 1.0 / 3.0).abs() < 0.02,
                "bucket {} should be ~uniform 0.333, got {:.3}",
                i,
                observed,
            );
        }
    }

    #[test]
    fn tournament_default_preserves_existing_behavior() {
        // Default selector is Tournament { k: 10 }. With sampling-with-replacement,
        // the top-scoring item should dominate but not deterministically.
        let queue: Vec<Runnable<NoHashing>> = vec![heal(0.1), heal(0.9)];
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let crashed = OrdSet::new();
        let selector = WithinQueueSelector::default();
        assert!(matches!(selector, WithinQueueSelector::Tournament { k: 10 }));

        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = [0usize; 2];
        for _ in 0..4_000 {
            let idx = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &crashed,
                1.0,
                &selector,
                &mut rng,
            );
            counts[idx] += 1;
        }
        // P(top wins) = 1 - (1/2)^k.min(2) = 0.75. Allow a wide margin.
        assert!(
            counts[1] > counts[0] * 2,
            "tournament should favor higher-score index 1: got {:?}",
            counts
        );
    }

    #[test]
    fn select_within_queue_handles_singleton() {
        let queue: Vec<Runnable<NoHashing>> = vec![heal(0.5)];
        let eligible = vec![0];
        let crashed = OrdSet::new();
        let mut rng = StdRng::seed_from_u64(1);

        let tournament = WithinQueueSelector::Tournament { k: 10 };
        let proportional = WithinQueueSelector::Proportional { exponent: 1.0 };

        for selector in [&tournament, &proportional] {
            let idx = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &crashed,
                1.0,
                selector,
                &mut rng,
            );
            assert_eq!(idx, 0);
        }
    }
}
