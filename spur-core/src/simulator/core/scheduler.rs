use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::make_local_env;
use crate::simulator::core::exec::{exec, exec_sync_on_node};
use crate::simulator::core::partition::{activate_partition, heal_partition};
use crate::simulator::core::queue_selector::{
    QueueInfo, QueueSelection, QueueSelector, WithinQueueSelector,
};
use crate::simulator::core::state::{
    Continuation, HandlerTrigger, Logger, NodeId, PurgatoryConfig, Record, Runnable,
    RunnableCategory, SchedulePolicy, ScheduleResult, State,
};
use crate::simulator::core::steer_terms::{ResolvedTerms, Term, TERMS};
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

/// A runnable the quick-fire weighting applies to: bringing back a node that is
/// down right now.
fn is_quick_fire<H: HashPolicy>(r: &Runnable<H>, currently_crashed: &OrdSet<NodeId>) -> bool {
    matches!(r, Runnable::Recover { node_id, .. } if currently_crashed.contains(node_id))
}

/// Combine the score terms into [0, 1]. Where the quick-fire weighting
/// applies, `recover_crashed` multiplies the priority weight, which raises
/// priority relative to novelty; a multiplier of 1 makes both branches
/// identical. `bonus` is the summed weight of the predicates true of the
/// runnable and enters both the numerator and the denominator, so the score
/// stays in [0, 1] and a runnable with every predicate true scores 1 only
/// when its other terms do.
fn blend(terms: &ResolvedTerms, novelty: f64, priority: f64, quick_fire: bool, bonus: f64) -> f64 {
    let w = if quick_fire {
        terms.priority * terms.recover_crashed
    } else {
        terms.priority
    };
    let num = terms.novelty * novelty + w * priority;
    let den = terms.novelty + w;
    if bonus > 0.0 {
        (num + bonus) / (den + bonus)
    } else {
        num / den
    }
}

/// The summed weight of the predicates in `mask`.
#[inline]
fn bonus_of(terms: &ResolvedTerms, mask: u8) -> f64 {
    let mut bonus = 0.0;
    for t in Term::ALL {
        if mask & (1u8 << t.index()) != 0 {
            bonus += terms.weight(t);
        }
    }
    bonus
}

/// Score a runnable in [0, 1] and report which predicates were true of it.
/// The predicates are read from the run's state only when a predicate
/// carries weight or the counters want them (`want_mask`); otherwise the
/// score is the novelty and priority terms alone and no state is consulted.
fn score_with_terms<H: HashPolicy, F: Feedback>(
    r: &Runnable<H>,
    feedback: &F::Local,
    snapshot: &F::Snapshot,
    state: &State<H>,
    terms: &ResolvedTerms,
    want_mask: bool,
) -> (f64, u8) {
    let novelty = F::runnable_novelty(feedback, r, snapshot);
    let priority = r.priority();
    let quick_fire = is_quick_fire(r, &state.crash_info.currently_crashed);
    let mask = if want_mask || terms.any_predicate() {
        state.term_mask(r)
    } else {
        0
    };
    let bonus = if terms.any_predicate() {
        bonus_of(terms, mask)
    } else {
        0.0
    };
    (blend(terms, novelty, priority, quick_fire, bonus), mask)
}

/// Score a runnable in [0, 1]. For a recover of a node that is down,
/// `terms.recover_crashed` raises the weight of priority relative to
/// novelty; a predicate that carries weight raises the score of a runnable
/// it is true of.
fn score_runnable<H: HashPolicy, F: Feedback>(
    r: &Runnable<H>,
    feedback: &F::Local,
    snapshot: &F::Snapshot,
    state: &State<H>,
    terms: &ResolvedTerms,
) -> f64 {
    score_with_terms::<H, F>(r, feedback, snapshot, state, terms, false).0
}

/// Rank the eligible candidates once per swept magnitude and report how often
/// the top-ranked one moves away from what the identity weighting ranks first.
/// This asks whether reweighting the priority term can outvote the random draw
/// priority itself is sampled from, which is a property of the scoring function
/// and not of any one magnitude being configured. Consumes no RNG and does not
/// influence the selection.
///
/// A selection with a single eligible candidate is still counted, so a zero
/// flip rate can be told apart from a weighting that was never given a
/// competitor to rank against.
fn audit_multiplier_authority<H: HashPolicy, F: Feedback>(
    queue: &[Runnable<H>],
    eligible: &[usize],
    feedback: &F::Local,
    snapshot: &F::Snapshot,
    state: &State<H>,
    terms: &ResolvedTerms,
) {
    let currently_crashed = &state.crash_info.currently_crashed;
    let sweep = util_stats::MULTIPLIER_SWEEP;
    let present = eligible
        .iter()
        .any(|&i| is_quick_fire(&queue[i], currently_crashed));
    let contested = eligible.len() > 1;
    util_stats::record_multiplier_decision(contested, present);
    if !present || !contested {
        return;
    }

    let mut best = [(f64::NEG_INFINITY, usize::MAX); util_stats::MULTIPLIER_SWEEP.len()];
    let mut best_configured = (f64::NEG_INFINITY, usize::MAX);
    for &i in eligible {
        let novelty = F::runnable_novelty(feedback, &queue[i], snapshot);
        let priority = queue[i].priority();
        let quick_fire = is_quick_fire(&queue[i], currently_crashed);
        let bonus = if terms.any_predicate() {
            bonus_of(terms, state.term_mask(&queue[i]))
        } else {
            0.0
        };
        for (slot, &m) in best.iter_mut().zip(sweep.iter()) {
            let s = blend(&terms.with_recover_crashed(m), novelty, priority, quick_fire, bonus);
            if s > slot.0 {
                *slot = (s, i);
            }
        }
        let s = blend(terms, novelty, priority, quick_fire, bonus);
        if s > best_configured.0 {
            best_configured = (s, i);
        }
    }

    let baseline = best[0].1;
    let mut flipped = [false; util_stats::MULTIPLIER_SWEEP.len()];
    for (f, slot) in flipped.iter_mut().zip(best.iter()) {
        *f = slot.1 != baseline;
    }
    util_stats::record_multiplier_flips(
        terms.recover_crashed,
        &flipped,
        best_configured.1 != baseline,
    );
}

/// The priority-only share of `score_runnable` (novelty zeroed out, same
/// quick-fire weighting). Only used by the opt-in utilization probe below.
fn priority_component<H: HashPolicy>(
    r: &Runnable<H>,
    currently_crashed: &OrdSet<NodeId>,
    terms: &ResolvedTerms,
) -> f64 {
    let priority = r.priority();
    let w = if is_quick_fire(r, currently_crashed) {
        terms.priority * terms.recover_crashed
    } else {
        terms.priority
    };
    (w * priority) / (terms.novelty + w)
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
    terms: &ResolvedTerms,
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
            let score = score_runnable::<H, F>(r, feedback, snapshot, state, terms);
            if score > best_score {
                best_score = score;
                best_slot = Some((slot, blocked));
            }
            let priority = priority_component(r, currently_crashed, terms);
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

    util_stats::record_audit_candidates(candidates as usize);
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

/// Select an eligible item from a single queue, returning its index and
/// the predicates true of it.
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
    state: &State<H>,
    terms: &ResolvedTerms,
    selector: &WithinQueueSelector,
    rng: &mut impl StreamRng,
) -> (usize, u8) {
    let stats = util_stats::enabled();
    util_stats::record_preference_consultation(terms.any_predicate());
    if util_stats::multiplier_audit_enabled() {
        audit_multiplier_authority::<H, F>(queue, eligible, feedback, snapshot, state, terms);
    }
    // The term counters can only separate candidates a predicate is true of
    // from the rest when some predicate carries weight. With none carrying
    // weight, the state reads and the extra ranking below report a constant, so
    // they are skipped and each skip is counted.
    let count_terms = stats && terms.any_predicate();
    if stats && !count_terms {
        util_stats::record_empty_slice_skip(util_stats::EmptySliceStage::CandidateMask);
    }
    // The predicates true of any eligible candidate, for the counters only.
    let present = if count_terms {
        eligible.iter().fold(0u8, |m, &i| m | state.term_mask(&queue[i]))
    } else {
        0
    };
    if eligible.len() <= 1 {
        let (_, mask) =
            score_with_terms::<H, F>(&queue[eligible[0]], feedback, snapshot, state, terms, count_terms);
        if count_terms {
            let mut evaluated = [0u64; TERMS];
            for t in Term::ALL {
                evaluated[t.index()] = u64::from(mask & (1u8 << t.index()) != 0);
            }
            util_stats::record_term_decision(eligible.len(), present, &evaluated, mask, false);
        }
        return (eligible[0], mask);
    }
    rng.use_stream(Stream::WithinQueue);

    // Observation-only utilization probe: would the greedy pick change if the
    // novelty/steer term were dropped? Compares the blended-score argmax with
    // the priority-only argmax (first index wins ties). Consumes no RNG and
    // does not influence the selection below.
    if count_terms {
        let currently_crashed = &state.crash_info.currently_crashed;
        let mut best_blend = f64::NEG_INFINITY;
        let mut best_blend_idx = eligible[0];
        let mut best_prio = f64::NEG_INFINITY;
        let mut best_prio_idx = eligible[0];
        for &i in eligible {
            let blend = score_runnable::<H, F>(&queue[i], feedback, snapshot, state, terms);
            let prio = priority_component(&queue[i], currently_crashed, terms);
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
    } else if stats {
        util_stats::record_empty_slice_skip(util_stats::EmptySliceStage::RankingPass);
    }

    let mut evaluated = [0u64; TERMS];
    let mut count_mask = |mask: u8| {
        for t in Term::ALL {
            if mask & (1u8 << t.index()) != 0 {
                evaluated[t.index()] += 1;
            }
        }
    };
    let (best_idx, best_mask, flipped) = match selector {
        WithinQueueSelector::Tournament { k } => {
            let k = (*k).max(1);
            // The choice the score makes without predicate weights, kept
            // beside the real one so a flip can be counted.
            let unweighted = ResolvedTerms {
                weights: [0.0; TERMS],
                ..*terms
            };
            let mut best_idx = eligible[rng.random_range(0..eligible.len())];
            let (mut best_score, mut best_mask) =
                score_with_terms::<H, F>(&queue[best_idx], feedback, snapshot, state, terms, count_terms);
            count_mask(best_mask);
            let mut plain_idx = best_idx;
            let mut plain_score = if terms.any_predicate() {
                score_runnable::<H, F>(&queue[best_idx], feedback, snapshot, state, &unweighted)
            } else {
                best_score
            };
            for _ in 1..k.min(eligible.len()) {
                let i = eligible[rng.random_range(0..eligible.len())];
                let (s, mask) =
                    score_with_terms::<H, F>(&queue[i], feedback, snapshot, state, terms, count_terms);
                count_mask(mask);
                if s > best_score {
                    best_idx = i;
                    best_score = s;
                    best_mask = mask;
                }
                if terms.any_predicate() {
                    let s0 = score_runnable::<H, F>(&queue[i], feedback, snapshot, state, &unweighted);
                    if s0 > plain_score {
                        plain_idx = i;
                        plain_score = s0;
                    }
                }
            }
            (best_idx, best_mask, terms.any_predicate() && plain_idx != best_idx)
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
            let mut best_mask = 0u8;
            let mut best_key = f64::NEG_INFINITY;
            for &i in eligible {
                let (s, mask) =
                    score_with_terms::<H, F>(&queue[i], feedback, snapshot, state, terms, count_terms);
                count_mask(mask);
                let weight = s.powf(*exponent).max(1e-9);
                let u: f64 = rng.random();
                // u is in (0, 1); ln(u) is negative; key = ln(u) / weight is negative.
                // Higher weight means a key closer to 0 (larger), so argmax is correct.
                let key = u.ln() / weight;
                if key > best_key {
                    best_key = key;
                    best_idx = i;
                    best_mask = mask;
                }
            }
            (best_idx, best_mask, false)
        }
    };
    if count_terms {
        util_stats::record_term_decision(eligible.len(), present, &evaluated, best_mask, flipped);
    }
    (best_idx, best_mask)
}

/// Where a predicated candidate would send the step. With a predicate
/// weight W among the terms' total, the step goes to a queue holding such a
/// candidate with probability W / (W + novelty + priority), the share the
/// predicate holds of the score; otherwise the ordinary queue roll decides.
/// Reads only the ledger, so the cost is one pass over the nodes, and
/// consumes exactly one draw when a candidate exists and none otherwise.
fn route_by_terms<H: HashPolicy>(
    state: &State<H>,
    info: &QueueInfo,
    terms: &ResolvedTerms,
    rng: &mut impl StreamRng,
) -> Option<QueueSelection> {
    util_stats::record_preference_consultation(terms.any_predicate());
    if !terms.any_predicate() {
        return None;
    }
    let mut queues: Vec<(QueueSelection, f64)> = Vec::new();
    for (n, ledger) in state.send_ledger.iter().enumerate() {
        if ledger.crash_pending == 0 || info.local_queue_sizes.get(n).copied().unwrap_or(0) == 0 {
            continue;
        }
        if let Some(t) = state.crash_after_sends_term(n) {
            let w = terms.weight(t);
            if w > 0.0 {
                queues.push((QueueSelection::Local(n), w));
            }
        }
    }
    if info.network_queue_size > 0 && state.net_stale_records > 0 {
        let w_stale = terms.weight(Term::StaleLate);
        let w_request = if state.net_requests > 0 {
            terms.weight(Term::RequestBeforeStale)
        } else {
            0.0
        };
        let w = w_stale.max(w_request);
        if w > 0.0 {
            queues.push((QueueSelection::Network, w));
        }
    }
    let total: f64 = queues.iter().map(|(_, w)| w).sum();
    if total <= 0.0 {
        return None;
    }
    rng.use_stream(Stream::QueueChoice);
    let u: f64 = rng.random();
    let share = total / (total + terms.novelty + terms.priority);
    let routed = if u < share {
        let mut x = (u / share) * total;
        let mut pick = queues[queues.len() - 1].0;
        for (q, w) in &queues {
            if x < *w {
                pick = *q;
                break;
            }
            x -= w;
        }
        Some(pick)
    } else {
        None
    };
    util_stats::record_term_authority(routed.is_some());
    routed
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
    terms: &ResolvedTerms,
    purgatory_config: &PurgatoryConfig,
    reservations: &[Reservation],
    rng: &mut impl StreamRng,
) -> Result<ScheduleResult<H>, RuntimeError> {
    util_stats::record_steer_step();
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
    // A crash is never withheld by a reservation or a link order, so a
    // pending crash is a schedulable one.
    if util_stats::enabled() {
        let mut crash_eligible = false;
        let mut anchored = false;
        for ledger in &state.send_ledger {
            if ledger.crash_pending > 0 {
                crash_eligible = true;
                anchored |= ledger.in_flight > 0;
            }
        }
        util_stats::record_crash_anchor_offer(crash_eligible, anchored);
    }

    // Observation-only steer-authority audit: what the scoring function ranks
    // first here, resolved below against what this step actually runs. It ranks
    // every runnable in every queue, and with no predicate carrying weight the
    // ranking is novelty and priority alone, which is worth resolving only when
    // the session asks for it; otherwise it is skipped and the skip is counted.
    util_stats::record_preference_consultation(terms.any_predicate());
    let audit_wanted = util_stats::steer_audit_enabled();
    let resolvable = terms.any_predicate() || util_stats::steer_audit_always();
    if audit_wanted && !resolvable {
        util_stats::record_empty_slice_skip(util_stats::EmptySliceStage::QueueAudit);
    }
    let audit = (audit_wanted && resolvable).then(|| {
        audit_steer_preference::<H, F>(
            state,
            feedback,
            snapshot,
            terms,
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

    let record_unscheduled = |audit: &Option<SteerPreference>| {
        if let Some(a) = audit {
            util_stats::record_steer_authority(a.expressed, a.outcome(None));
        }
    };

    let routed = route_by_terms(state, &info, terms, rng);
    rng.use_stream(Stream::QueueChoice);
    let selection = match routed {
        Some(s) => s,
        None => match selector.select(&info, rng) {
            Some(s) => s,
            None => {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
        },
    };

    let (runnable, chosen_slot, chosen_mask) = match selection {
        QueueSelection::Local(node_idx) => {
            let queue = &state.local_queues[node_idx];
            let eligible: Vec<usize> = (0..queue.len())
                .filter(|&i| !is_ineligible(&queue[i]))
                .collect();
            if eligible.is_empty() {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
            let (idx, mask) = select_within_queue::<H, F>(
                queue,
                &eligible,
                feedback,
                snapshot,
                state,
                terms,
                within_queue,
                rng,
            );
            (
                state.take_local(node_idx, idx),
                QueueSlot::Local(node_idx, idx),
                mask,
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
            let (idx, mask) = select_within_queue::<H, F>(
                queue,
                &eligible,
                feedback,
                snapshot,
                state,
                terms,
                within_queue,
                rng,
            );
            (state.take_network(idx), QueueSlot::Network(idx), mask)
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
            let (idx, mask) = select_within_queue::<H, F>(
                queue,
                &eligible,
                feedback,
                snapshot,
                state,
                terms,
                within_queue,
                rng,
            );
            (state.timer_queue.remove(idx), QueueSlot::Timer(idx), mask)
        }
    };

    if let Some(a) = &audit {
        util_stats::record_steer_authority(a.expressed, a.outcome(Some(chosen_slot)));
    }

    // Observation-only timer-admission probe: when a timer and a message
    // delivery are both schedulable, which of the two the step runs. Nothing
    // else in the counters observes that ordering.
    if util_stats::enabled() && info.timer_queue_size > 0 && info.network_queue_size > 0 {
        util_stats::record_timer_admission(matches!(chosen_slot, QueueSlot::Timer(_)));
    }

    match runnable {
        Runnable::Crash { node_id, .. } => {
            if util_stats::enabled() {
                let ledger = state.send_ledger.get(node_id.index).copied().unwrap_or_default();
                util_stats::record_crash_anchor_apply(ledger.in_flight > 0);
                util_stats::record_term_acted(chosen_mask, ledger.recent > 0);
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

            if let Some(chan) = state.channels.get_mut(&timer.channel) {
                match chan.pop_waiting_reader() {
                    None => chan.buffer.push_back(Value::<H>::unit()),
                    Some((mut reader, lhs)) => {
                        let node_index = reader.node.index;
                        let mut r_node_env = state.nodes[node_index].clone();
                        if let Err(e) = crate::simulator::core::eval::store(
                            &lhs,
                            Value::<H>::unit(),
                            &mut reader.env,
                            &mut r_node_env,
                        ) {
                            log::warn!("Store failed in timer completion: {}", e);
                        }
                        state.nodes[node_index] = r_node_env;
                        reader.timer_entry = Some(reader.pc);
                        state.push_to_local(node_index, Runnable::Record(reader));
                    }
                }
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
                        (
                            bias,
                            state.node_state_token(record_dest),
                            state.entries_since_restart(record_dest.index),
                        )
                    });
                    // The segment a timer firing woke, measured the same way
                    // as a delivery: the token counts state writes, so a
                    // segment that only sends reads as inert.
                    let timer_entry = r.timer_entry == Some(r.pc);
                    let timer_probe = (util_stats::acted_fraction_enabled() && timer_entry).then(|| {
                        let inflight = state.pending_deliveries_to(record_dest) > 0;
                        let key = util_stats::TimerKey::new(
                            r.pc,
                            inflight,
                            state.incarnation(record_dest),
                            state.timer_inert_streak(record_dest.index, r.pc),
                        );
                        (r.pc, key, inflight, state.node_state_token(record_dest))
                    });
                    if message_entry {
                        state.note_handler_entry(record_dest.index, HandlerTrigger::Delivery);
                    } else if timer_entry {
                        state.note_handler_entry(record_dest.index, HandlerTrigger::Timer);
                    }
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
                    if let Some((bias, before, distance)) = probe {
                        let acted = state.node_state_token(record_dest) != before;
                        util_stats::record_delivery(bias, acted, distance);
                        util_stats::record_term_acted(chosen_mask, acted);
                    }
                    if let Some((pc, key, inflight, before)) = timer_probe {
                        let acted = state.node_state_token(record_dest) != before;
                        state.note_timer_effect(record_dest.index, pc, inflight, acted);
                        util_stats::record_timer(key, acted);
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
                    if let Some(chan) = state.channels.get_mut(&channel) {
                        match chan.pop_waiting_reader() {
                            None => chan.buffer.push_back(message),
                            Some((mut reader, lhs)) => {
                                let node_index = reader.node.index;
                                let mut r_node_env = state.nodes[node_index].clone();
                                if let Err(e) = crate::simulator::core::eval::store(
                                    &lhs,
                                    message,
                                    &mut reader.env,
                                    &mut r_node_env,
                                ) {
                                    log::warn!("Store failed in remote channel delivery: {}", e);
                                }
                                state.nodes[node_index] = r_node_env;
                                reader.timer_entry = None;
                                state.push_to_local(node_index, Runnable::Record(reader));
                            }
                        }
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
    state.note_handler_entry(node_id.index, HandlerTrigger::None);

    let mut held: u64 = 0;
    let mut dropped: u64 = 0;

    // 1. Process local queue for crashed node: save external records, drop the rest
    let local = std::mem::take(&mut state.local_queues[node_id.index]);
    for task in local {
        if let Runnable::Crash { .. } = &task
            && let Some(l) = state.send_ledger.get_mut(node_id.index)
        {
            l.crash_pending = l.crash_pending.saturating_sub(1);
        }
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
                state.flight_leave(&task);
                state.net_leave(&task);
                if r.origin_node != r.node {
                    let mut r = r.clone();
                    r.reset();
                    held += 1;
                    state.crash_info.queued_messages.push_back((node_id, r));
                } else {
                    dropped += 1;
                }
            }
            Runnable::ChannelSend { target, .. } if *target == node_id => {
                state.flight_leave(&task);
                dropped += 1
            }
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
    let own_sends_inflight = state
        .send_ledger
        .get(node_id.index)
        .is_some_and(|l| l.in_flight > 0);
    state.note_incarnation_bump(node_id.index);
    state.note_handler_entry(node_id.index, HandlerTrigger::None);
    util_stats::record_recover(
        node_id.index,
        state.crash_info.current_step,
        own_sends_inflight,
    );
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
        timer_entry: None,
        send_ordinal: state.next_send_ordinal(node_id),
        receiver_token_at_send: state.node_state_token(node_id),
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
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    fn heal(priority: f64) -> Runnable<NoHashing> {
        Runnable::Heal { priority }
    }

    fn terms_with(multiplier: f64) -> ResolvedTerms {
        ResolvedTerms::default().with_recover_crashed(multiplier)
    }

    fn empty_state() -> State<NoHashing> {
        State::new(&[(crate::analysis::resolver::NameId(0), 1)], 1)
    }

    /// The fixed blend the scheduler scored with before its terms had
    /// names, kept as the oracle the named form is checked against.
    fn blend_score(novelty: f64, priority: f64, quick_fire: bool, multiplier: f64) -> f64 {
        if quick_fire {
            let w = 0.75 * multiplier;
            (0.25 * novelty + w * priority) / (0.25 + w)
        } else {
            0.25 * novelty + 0.75 * priority
        }
    }

    /// The priority-only share of `blend_score`, the same way.
    fn legacy_priority_component(priority: f64, quick_fire: bool, multiplier: f64) -> f64 {
        if quick_fire {
            let w = 0.75 * multiplier;
            (w * priority) / (0.25 + w)
        } else {
            0.75 * priority
        }
    }

    /// At the default weights the named blend is the fixed blend bit for
    /// bit: the weights are dyadic, `0.25 + 0.75` is exactly `1.0`, and a
    /// division by `1.0` is the identity, so the extra normalisation cannot
    /// move a single bit. The quick-fire branch performs the same operations
    /// in the same order as the fixed form.
    #[test]
    fn default_terms_reproduce_blend_score_bitwise() {
        let crashed_node = NodeId {
            role: crate::analysis::resolver::NameId(0),
            index: 0,
        };
        let mut crashed = OrdSet::new();
        crashed.insert(crashed_node);
        for &novelty in &[0.0, 0.3, 0.4, 1.0] {
            for &priority in &[0.0, 0.15, 0.3, 0.65, 0.95, 1.0] {
                for &m in &[1.0, 3.0, 5.0, 8.0, 1000.0] {
                    let terms = terms_with(m);
                    for &quick_fire in &[false, true] {
                        assert_eq!(
                            blend(&terms, novelty, priority, quick_fire, 0.0).to_bits(),
                            blend_score(novelty, priority, quick_fire, m).to_bits(),
                            "novelty {novelty} priority {priority} m {m} quick_fire {quick_fire}"
                        );
                    }
                    let recover = Runnable::<NoHashing>::Recover {
                        node_id: crashed_node,
                        priority,
                    };
                    assert_eq!(
                        priority_component(&recover, &crashed, &terms).to_bits(),
                        legacy_priority_component(priority, true, m).to_bits()
                    );
                    assert_eq!(
                        priority_component(&heal(priority), &crashed, &terms).to_bits(),
                        legacy_priority_component(priority, false, m).to_bits()
                    );
                }
            }
        }
    }

    /// Score under default `score_runnable` parameters (no novelty signal,
    /// no quick-fire boost) is `0.25 + 0.75 * priority`.
    fn expected_score(priority: f64) -> f64 {
        0.25 + 0.75 * priority
    }

    #[test]
    fn proportional_selection_matches_expected_distribution() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let queue: Vec<Runnable<NoHashing>> = vec![
            heal(0.0), // score 0.25
            heal(0.5), // score 0.625
            heal(1.0), // score 1.00
        ];
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let state = empty_state();
        let selector = WithinQueueSelector::Proportional { exponent: 1.0 };

        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        let trials = 50_000usize;
        let mut counts = [0usize; 3];
        for _ in 0..trials {
            let (idx, _) = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &state,
                &terms_with(1.0),
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
        let _serial = crate::simulator::config_override::exclusive_session();
        let queue: Vec<Runnable<NoHashing>> = vec![heal(0.0), heal(0.5), heal(1.0)];
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let state = empty_state();
        let selector = WithinQueueSelector::Proportional { exponent: 0.0 };

        let mut rng = StdRng::seed_from_u64(42);
        let trials = 30_000usize;
        let mut counts = [0usize; 3];
        for _ in 0..trials {
            let (idx, _) = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &state,
                &terms_with(1.0),
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
        let _serial = crate::simulator::config_override::exclusive_session();
        // Default selector is Tournament { k: 10 }. With sampling-with-replacement,
        // the top-scoring item should dominate but not deterministically.
        let queue: Vec<Runnable<NoHashing>> = vec![heal(0.1), heal(0.9)];
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let state = empty_state();
        let selector = WithinQueueSelector::default();
        assert!(matches!(selector, WithinQueueSelector::Tournament { k: 10 }));

        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = [0usize; 2];
        for _ in 0..4_000 {
            let (idx, _) = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &state,
                &terms_with(1.0),
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

    /// A multiplier of 1 collapses the two branches of `blend_score`, which is
    /// what makes it usable as the unweighted baseline the sweep compares
    /// against.
    #[test]
    fn identity_multiplier_makes_the_quick_fire_branch_a_no_op() {
        for &novelty in &[0.0, 0.4, 1.0] {
            for &priority in &[0.0, 0.3, 1.0] {
                assert_eq!(
                    blend_score(novelty, priority, true, 1.0),
                    blend_score(novelty, priority, false, 1.0),
                );
            }
        }
    }

    /// However large the multiplier grows, the quick-fire score converges on
    /// the priority draw itself and never exceeds 1, so it cannot outrank a
    /// competitor whose own blended score is already higher than that draw.
    #[test]
    fn quick_fire_score_is_bounded_by_the_priority_draw() {
        let priority = 0.4;
        for &m in &[3.0, 10.0, 100.0, 1000.0] {
            let s = blend_score(1.0, priority, true, m);
            assert!(s > priority && s <= 1.0, "m={} gave {}", m, s);
        }
        assert!(blend_score(1.0, priority, true, 1000.0) < blend_score(1.0, priority, true, 3.0));
    }

    /// With every predicate weight at zero the router draws nothing, so the
    /// queue roll sees exactly the sequence it saw before terms existed.
    #[test]
    fn zero_weights_consume_no_queue_choice_draw() {
        let mut state = empty_state();
        state.send_ledger[0].crash_pending = 1;
        state.send_ledger[0].recent = 1;
        state.send_ledger[0].trigger = HandlerTrigger::Delivery;
        state.net_stale_records = 1;
        state.net_requests = 1;
        let info = QueueInfo {
            local_queue_sizes: vec![1],
            network_queue_size: 4,
            timer_queue_size: 0,
            step: 0,
        };
        let mut rng = StdRng::seed_from_u64(5);
        let mut untouched = StdRng::seed_from_u64(5);
        assert!(route_by_terms(&state, &info, &ResolvedTerms::default(), &mut rng).is_none());
        assert_eq!(rng.next_u64(), untouched.next_u64(), "a draw was consumed");
    }

    /// A crash of a node with sends in flight from a delivery-triggered
    /// handler is routed to that node's queue with the predicate's share of
    /// the score, and nothing is routed when no predicate holds.
    #[test]
    fn authority_routes_with_the_configured_share() {
        let mut state = empty_state();
        state.send_ledger[0].crash_pending = 1;
        state.send_ledger[0].recent = 1;
        state.send_ledger[0].trigger = HandlerTrigger::Delivery;
        let info = QueueInfo {
            local_queue_sizes: vec![1],
            network_queue_size: 40,
            timer_queue_size: 0,
            step: 0,
        };
        let terms = ResolvedTerms {
            weights: [0.0, 2.33, 0.0, 0.0],
            ..ResolvedTerms::default()
        };
        let mut rng = StdRng::seed_from_u64(11);
        let trials = 20_000;
        let mut local = 0usize;
        for _ in 0..trials {
            match route_by_terms(&state, &info, &terms, &mut rng) {
                Some(QueueSelection::Local(0)) => local += 1,
                Some(other) => panic!("routed to {other:?}"),
                None => {}
            }
        }
        let share = 2.33 / (2.33 + 1.0);
        let observed = local as f64 / trials as f64;
        assert!(
            (observed - share).abs() < 0.02,
            "expected share {share:.3}, observed {observed:.3}"
        );
        state.send_ledger[0].trigger = HandlerTrigger::Timer;
        assert!(route_by_terms(&state, &info, &terms, &mut rng).is_none(), "the timer term carries no weight");
        state.send_ledger[0].trigger = HandlerTrigger::Delivery;
        state.send_ledger[0].recent = 0;
        assert!(route_by_terms(&state, &info, &terms, &mut rng).is_none(), "no sends in flight");
    }

    #[test]
    fn select_within_queue_handles_singleton() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let queue: Vec<Runnable<NoHashing>> = vec![heal(0.5)];
        let eligible = vec![0];
        let state = empty_state();
        let mut rng = StdRng::seed_from_u64(1);

        let tournament = WithinQueueSelector::Tournament { k: 10 };
        let proportional = WithinQueueSelector::Proportional { exponent: 1.0 };

        for selector in [&tournament, &proportional] {
            let (idx, _) = select_within_queue::<NoHashing, NoFeedback>(
                &queue,
                &eligible,
                &(),
                &(),
                &state,
                &terms_with(1.0),
                selector,
                &mut rng,
            );
            assert_eq!(idx, 0);
        }
    }
}
