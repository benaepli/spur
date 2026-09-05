use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::make_local_env;
use crate::simulator::core::exec::{exec, exec_sync_on_node};
use crate::simulator::core::partition::{activate_partition, heal_partition};
use crate::simulator::core::queue_selector::{
    QueueInfo, QueueSelection, QueueSelector, WithinQueueSelector,
};
use crate::simulator::core::state::{
    Continuation, HandlerTrigger, Logger, NodeId, PurgatoryConfig, Record, ReplayCut, Runnable,
    RunnableCategory, SchedulePolicy, ScheduleResult, State,
};
use crate::simulator::core::steer_terms::{ResolvedTerms, Term, TERMS};
use crate::simulator::core::values::{Env, Value};
use crate::simulator::crash_phase;
use crate::simulator::fresh_first;
use crate::simulator::ghost_absorber;
use crate::simulator::pair_order;
use crate::simulator::coverage::GlobalState;
use crate::simulator::feedback::Feedback;
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::Topology;
use crate::simulator::path::TopologyInfo;
use crate::simulator::rng::{Stream, StreamRng};
use crate::simulator::timer_context;
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

/// The factor the identity-weighted recovery term applies to a candidate its
/// predicate holds of. Fixed at the identity, so multiplying by it cannot move
/// a single bit of a candidate's score and the term cannot change any ranking.
const RECOVERY_PLACEBO_FACTOR: f64 = 1.0;

/// A cheap integer mix, used only to turn two small numbers into a parity that
/// neither of them biases.
#[inline]
fn mix(a: u64, b: u64) -> u64 {
    let mut x = a
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(b.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x
}

/// A predicate over a candidate the recovery weighting applies to that carries
/// no information about the run: the parity of a mix of the node's index, how
/// many times it has come back, and how much undelivered traffic the run is
/// holding. All three are read only to make the parity vary from one selection
/// to the next, so the predicate holds of roughly half the candidates and which
/// half is unrelated to anything the protocol did.
fn uninformative_recovery_predicate<H: HashPolicy>(r: &Runnable<H>, state: &State<H>) -> bool {
    match r {
        Runnable::Recover { node_id, .. } => {
            let incarnation = state
                .incarnations
                .get(node_id.index)
                .copied()
                .unwrap_or_default();
            let undelivered = state.network_queue.len() as u64;
            mix(node_id.index as u64, mix(u64::from(incarnation), undelivered)) & 1 == 1
        }
        _ => false,
    }
}

/// Walk the eligible candidates the way a recovery-reweighting term would,
/// evaluating its predicate, re-ranking under its factor and reporting what it
/// preferred, with the factor held at the identity so nothing it computes can
/// reach the selection. This separates what such a term costs in code path and
/// throughput from what its reweighting does. Consumes no RNG.
fn walk_recovery_placebo<H: HashPolicy, F: Feedback>(
    queue: &[Runnable<H>],
    eligible: &[usize],
    feedback: &F::Local,
    snapshot: &F::Snapshot,
    state: &State<H>,
    terms: &ResolvedTerms,
) {
    if eligible.is_empty() {
        return;
    }
    let currently_crashed = &state.crash_info.currently_crashed;
    let weighted_terms = terms.with_recover_crashed(terms.recover_crashed * RECOVERY_PLACEBO_FACTOR);
    let mut evaluated = 0u64;
    let mut best = (f64::NEG_INFINITY, usize::MAX);
    let mut best_carries = false;
    let mut plain = (f64::NEG_INFINITY, usize::MAX);
    for &i in eligible {
        let novelty = F::runnable_novelty(feedback, &queue[i], snapshot);
        let priority = queue[i].priority();
        let quick_fire = is_quick_fire(&queue[i], currently_crashed);
        let bonus = if terms.any_predicate() {
            bonus_of(terms, state.term_mask(&queue[i]))
        } else {
            0.0
        };
        let carries = quick_fire && uninformative_recovery_predicate(&queue[i], state);
        if carries {
            evaluated += 1;
        }
        let with_term = if carries { &weighted_terms } else { terms };
        let s = blend(with_term, novelty, priority, quick_fire, bonus);
        if s > best.0 {
            best = (s, i);
            best_carries = carries;
        }
        let without_term = blend(terms, novelty, priority, quick_fire, bonus);
        if without_term > plain.0 {
            plain = (without_term, i);
        }
    }
    let present = evaluated > 0;
    util_stats::record_recovery_placebo(
        evaluated,
        present,
        present && eligible.len() > 1,
        best_carries,
        best.1 != plain.1,
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
    util_stats::record_steer_reach(if candidates <= 1 {
        util_stats::SteerReach::SingleCandidate
    } else if expressed {
        util_stats::SteerReach::PreferenceExpressed
    } else {
        util_stats::SteerReach::RankingAgreedWithPriority
    });
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
    if util_stats::recovery_weight_placebo_enabled() {
        walk_recovery_placebo::<H, F>(queue, eligible, feedback, snapshot, state, terms);
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

/// Bitmask over the first 64 nodes of pending crashes a crash-placement hold
/// withholds from this step. The drawn target step expires the hold
/// unconditionally; on a run that anchors, the crash then waits further for a
/// drawn phase of its victim's own fan-out, bounded by a window. Past both,
/// the ordinary crash admission applies unchanged. Each offer the target step
/// excludes is counted per node per step; the anchor's waiting is counted in
/// its own block, so the placement counters keep meaning the step hold alone.
/// The fan-out is read on the node `phase_read_node` names, which is the
/// victim itself on every run that does not retarget its crashes.
fn crash_hold_mask<H: HashPolicy>(
    state: &mut State<H>,
    servers: usize,
    rng: &mut impl StreamRng,
) -> u64 {
    let step_now = state.crash_info.current_step;
    let mut mask = 0u64;
    let width = state.crash_hold_until.len().min(u64::BITS as usize);
    for n in 0..width {
        let Some(ledger) = state.send_ledger.get(n) else {
            continue;
        };
        if ledger.crash_pending == 0 {
            continue;
        }
        if step_now < state.crash_hold_until[n] {
            mask |= 1u64 << n;
            util_stats::record_crash_place_hold();
            continue;
        }
        let read = phase_read_node(state, n, servers);
        let ledger = state.send_ledger.get(read).copied().unwrap_or(*ledger);
        let fanout = crash_phase::Fanout {
            segment_sends: ledger.issued.saturating_sub(ledger.floor),
            undelivered: ledger.recent,
            in_flight: ledger.in_flight,
        };
        if state
            .crash_phase
            .hold_read_on(n, read, fanout, step_now, rng)
        {
            mask |= 1u64 << n;
        }
    }
    mask
}

/// The node whose ledger the fan-out phase of `n`'s waiting crash is read
/// on. This is `n` itself unless the run retargets its crashes, still has a
/// release ahead for `n`, and the absorber ranking would move the crash to
/// another node at this step. The ranking is the one `retarget_crash`
/// consults at apply, on the same ledgers, so the node read and the node
/// crashed can differ only by what moves between the last read and the
/// apply.
fn phase_read_node<H: HashPolicy>(state: &State<H>, n: usize, servers: usize) -> usize {
    if !state.retarget.enabled || !state.crash_phase.awaits_release(n) {
        return n;
    }
    let planned = state.local_queues.get(n).and_then(|queue| {
        queue.iter().find_map(|r| match r {
            Runnable::Crash { node_id, .. } => Some(*node_id),
            _ => None,
        })
    });
    let Some(planned) = planned else {
        return n;
    };
    match absorber_decision(state, planned, servers).choice {
        ghost_absorber::Choice::Retarget { node, .. } => node,
        ghost_absorber::Choice::SameVictim | ghost_absorber::Choice::NoAbsorber => n,
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
    terms: &ResolvedTerms,
    purgatory_config: &PurgatoryConfig,
    partial_fanout_crash_bias: f64,
    timer_ctx_mode: timer_context::RunMode,
    reservations: &[Reservation],
    rng: &mut impl StreamRng,
) -> Result<ScheduleResult<H>, RuntimeError> {
    if state.all_queues_empty() {
        util_stats::record_steer_reach(util_stats::SteerReach::NoScheduleAttempt);
        return Ok(ScheduleResult::None);
    }

    // Crash-timing bias: a pending crash for a node that is not in the middle of
    // its own fan-out - it has no undelivered send of its own, or more than one -
    // may be withheld from this step, so the crash is more likely to be taken at
    // a point where exactly one of the node's peers is still uninformed of its
    // last action. Withholding is not cancelling: the crash stays queued and is
    // offered again at the next step, which keeps the number of crashes a run
    // takes the same and moves only when they land. Nodes past the width of the
    // mask are never withheld.
    let crash_hold_mask = crash_hold_mask(state, topology.num_servers.max(0) as usize, rng);

    let crash_defer_mask: u64 = if partial_fanout_crash_bias > 0.0 {
        let mut mask = 0u64;
        for (n, ledger) in state.send_ledger.iter().enumerate().take(u64::BITS as usize) {
            if ledger.crash_pending == 0 || ledger.in_flight == 1 {
                continue;
            }
            rng.use_stream(Stream::FaultPriority);
            let withhold = rng.random::<f64>() < partial_fanout_crash_bias;
            util_stats::record_crash_timing_bias(withhold);
            if withhold {
                mask |= 1u64 << n;
            }
        }
        mask
    } else {
        0
    };

    // Helper: check if a runnable is reserved OR FIFO-blocked. Both exclude the
    // item from scheduling via the same plumbing, so combine them here.
    let link_deliver_seq = state.link_deliver_seq.clone();
    let crash_block_mask = crash_defer_mask | crash_hold_mask;
    // A planned crash whose victim is already down waits for that node to
    // come back. Only a retargeted crash can leave a plan in that position,
    // so the check is confined to runs that retarget; it holds the crash in
    // the queue rather than dropping it, so the plan's pair still completes.
    let crashed_victims: Vec<NodeId> = if state.retarget.enabled {
        state.crash_info.currently_crashed.iter().copied().collect()
    } else {
        Vec::new()
    };
    let is_ineligible = |r: &Runnable<H>| {
        if crash_block_mask != 0
            && let Runnable::Crash { node_id, .. } = r
            && node_id.index < u64::BITS as usize
            && crash_block_mask & (1u64 << node_id.index) != 0
        {
            return true;
        }
        if !crashed_victims.is_empty()
            && let Runnable::Crash { node_id, .. } = r
            && crashed_victims.contains(node_id)
        {
            util_stats::record_victim_crashed_hold();
            return true;
        }
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

    // Whether any node whose crash was schedulable here was holding an
    // undelivered message of its own. Read before the pick, while the crash
    // this step may take is still counted among the candidates.
    let crash_candidate_with_inflight = util_stats::crash_census_enabled().then(|| {
        state
            .send_ledger
            .iter()
            .any(|l| l.crash_pending > 0 && l.in_flight > 0)
    });

    // Observation-only steer-authority audit: what the scoring function ranks
    // first here, resolved below against what this step actually runs. It ranks
    // every runnable in every queue, and with no predicate carrying weight the
    // ranking is novelty and priority alone, which is worth resolving only when
    // the session asks for it; otherwise it is skipped and the skip is counted.
    util_stats::record_steer_step();
    util_stats::record_preference_consultation(terms.any_predicate());
    let audit_wanted = util_stats::steer_audit_enabled();
    let resolvable = terms.any_predicate() || util_stats::steer_audit_always();
    if audit_wanted && !resolvable {
        util_stats::record_empty_slice_skip(util_stats::EmptySliceStage::QueueAudit);
    }
    if !audit_wanted {
        util_stats::record_steer_reach(util_stats::SteerReach::AuditDisabled);
    } else if !resolvable {
        util_stats::record_steer_reach(util_stats::SteerReach::NoWeightedPredicate);
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
        None => {
            // Steer-off probe runs and steps with no eligible timer take the
            // stock roll. The learned multiplier is read for the context of
            // the first eligible timer in queue order; the features are
            // copied into owned locals because the selector call below must
            // not overlap a borrow of `state`.
            let bias: Option<f64> = if timer_ctx_mode == timer_context::RunMode::Steered
                && info.timer_queue_size > 0
            {
                let head_timer_node = state.timer_queue.iter().find_map(|r| {
                    if is_ineligible(r) {
                        return None;
                    }
                    if let Runnable::Timer(t) = r {
                        if strict_timers
                            && let Some(l) = &t.label
                            && !state.allowed_timers.contains(&(t.node.index, l.clone()))
                        {
                            return None;
                        }
                        Some(t.node)
                    } else {
                        None
                    }
                });
                head_timer_node.and_then(|node| {
                    let ledger = state.send_ledger.get(node.index).copied().unwrap_or_default();
                    let cell = timer_context::cell_key(
                        state.pending_deliveries_to(node),
                        ledger.in_flight,
                        state.max_timer_inert_streak(node.index),
                        state.incarnation(node) > 0
                            && state.entries_since_restart(node.index) < 8,
                    );
                    timer_context::multiplier(cell)
                })
            } else {
                None
            };
            let picked = match bias {
                Some(m) if selector.supports_timer_bias() => {
                    util_stats::record_timer_context_bias(m);
                    selector.select_timer_biased(&info, m, rng)
                }
                Some(_) => {
                    util_stats::record_timer_context_excluded();
                    selector.select(&info, rng)
                }
                None => selector.select(&info, rng),
            };
            match picked {
                Some(s) => s,
                None => {
                    record_unscheduled(&audit);
                    return Ok(ScheduleResult::None);
                }
            }
        }
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
            let (drawn, drawn_mask) = select_within_queue::<H, F>(
                queue,
                &eligible,
                feedback,
                snapshot,
                state,
                terms,
                within_queue,
                rng,
            );
            let idx = fresh_first_dispatch(state, &eligible, drawn);
            let idx = pair_order_dispatch(state, &eligible, idx);
            observe_rush_dispatch(state, &eligible, idx);
            let delivered_op = state.network_queue[idx].causal_operation_id();
            note_first_delivery(state, delivered_op);
            // The mask names the predicates true of the record the step
            // runs, under the same condition the selection computed it.
            let mask = if idx != drawn && util_stats::enabled() && terms.any_predicate() {
                state.term_mask(&state.network_queue[idx])
            } else {
                drawn_mask
            };
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
            // Every census of what a crash lands on reads the node the crash
            // is applied to, which may differ from the planned victim. Only
            // the phase arm is keyed on the planned node, the node the hold
            // was armed on.
            let victim = retarget_crash(state, node_id, topology.num_servers.max(0) as usize);
            let ledger = state.send_ledger.get(victim.index).copied().unwrap_or_default();
            if util_stats::enabled() {
                util_stats::record_crash_anchor_apply(ledger.in_flight > 0);
                util_stats::record_term_acted(chosen_mask, ledger.recent > 0);
            }
            if let Some(any_candidate) = crash_candidate_with_inflight {
                util_stats::record_crash_census(ledger.in_flight, any_candidate);
            }
            if let Some(arm) = state.crash_phase.arm_of(node_id.index) {
                util_stats::record_crash_phase_apply(
                    arm,
                    ledger.in_flight,
                    victim.index != node_id.index,
                );
                state.crash_phase.landing_applied(node_id.index, victim.index);
            }
            if util_stats::enabled() {
                util_stats::record_victim_swap_census(
                    state.retarget.enabled,
                    ledger.last_ghost_step >= 0,
                    ledger.in_flight > 0,
                );
            }
            crash_node(state, victim);
            Ok(ScheduleResult::Crash {
                node_id: victim,
                planned: node_id,
            })
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
                    // Whether a ghost reached a destination that had already
                    // heard from the sender's current incarnation, read off
                    // the per-destination table before this entry is added.
                    if message_entry && util_stats::enabled() {
                        let current = state.incarnation(record_origin);
                        if r.origin_incarnation != current {
                            let overtaken = state.fresh_first.heard_from(
                                record_dest.index,
                                record_origin.index,
                                current,
                            );
                            util_stats::record_fresh_first_ghost_entry(
                                state.fresh_first.enabled,
                                overtaken,
                            );
                        }
                        state.fresh_first.note_entry(
                            record_dest.index,
                            record_origin.index,
                            r.origin_incarnation,
                        );
                    }
                    // Whether a message entry from a sender that has crashed
                    // at least once has a sibling of its class - same sender,
                    // same destination, same sending incarnation - already
                    // entered or still in the network queue, and whether a
                    // queued sibling carries a lower send ordinal. Read on
                    // census runs only, and the queue only when the sender's
                    // ledger says it holds a remote record of the sender at
                    // all.
                    if message_entry
                        && state.pair_order.census
                        && (state.incarnation(record_origin) > 0
                            || state.crash_info.currently_crashed.contains(&record_origin))
                    {
                        let entered_before = state.pair_order.note_entry(
                            record_origin.index,
                            record_dest.index,
                            r.origin_incarnation,
                        );
                        let queued = state
                            .send_ledger
                            .get(record_origin.index)
                            .is_some_and(|l| l.net_records > 0);
                        let scanned: &[Runnable<H>] = if queued { &state.network_queue } else { &[] };
                        let siblings = scanned.iter().filter_map(|q| match q {
                            Runnable::Record(s)
                                if s.origin_node == record_origin
                                    && s.node == record_dest
                                    && s.origin_incarnation == r.origin_incarnation =>
                            {
                                Some(s.send_ordinal)
                            }
                            _ => None,
                        });
                        if let Some(inverted) =
                            pair_order::classify_entry(siblings, r.send_ordinal, entered_before)
                        {
                            util_stats::record_pair_order_entry(
                                state.pair_order.enabled,
                                state.fault_crossing(record_origin, r.origin_incarnation),
                                inverted,
                            );
                        }
                    }
                    // A delivery from a sender that is down, or that restarted
                    // since sending, marks its receiver as having absorbed
                    // state from an incarnation that no longer exists.
                    let ghost = (message_entry
                        && state.fault_crossing(record_origin, r.origin_incarnation))
                    .then(|| state.node_state_token(record_dest));
                    if ghost.is_some()
                        && !state.retarget.signal_counted
                        && state
                            .send_ledger
                            .get(record_dest.index)
                            .is_some_and(|l| l.crash_pending > 0)
                    {
                        state.retarget.signal_counted = true;
                        state.replay_cut = Some(ReplayCut {
                            step: entry_step,
                            tape_pos: rng.position(),
                        });
                        util_stats::record_ghost_signal_run();
                    }
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
                        let pending = state.pending_deliveries_to(record_dest);
                        let inflight = pending > 0;
                        let key = util_stats::TimerKey::new(
                            r.pc,
                            inflight,
                            state.incarnation(record_dest),
                            state.timer_inert_streak(record_dest.index, r.pc),
                        );
                        let ledger = state
                            .send_ledger
                            .get(record_dest.index)
                            .copied()
                            .unwrap_or_default();
                        let cell = timer_context::cell_key(
                            pending,
                            ledger.in_flight,
                            state.max_timer_inert_streak(record_dest.index),
                            state.incarnation(record_dest) > 0
                                && state.entries_since_restart(record_dest.index) < 8,
                        );
                        (r.pc, key, inflight, cell, state.node_state_token(record_dest))
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
                    if let Some(before) = ghost {
                        state.note_ghost_delivery(record_dest.index, entry_step, before);
                    }
                    if let Some((bias, before, distance)) = probe {
                        let acted = state.node_state_token(record_dest) != before;
                        util_stats::record_delivery(bias, acted, distance);
                        util_stats::record_term_acted(chosen_mask, acted);
                    }
                    if let Some((pc, key, inflight, cell, before)) = timer_probe {
                        let acted = state.node_state_token(record_dest) != before;
                        state.note_timer_effect(record_dest.index, pc, inflight, acted);
                        util_stats::record_timer(key, acted);
                        // Only steer-off probe runs feed the learner, so the
                        // learned rates carry no imprint of the bias.
                        if timer_ctx_mode == timer_context::RunMode::Probe {
                            timer_context::record_firing(cell, acted);
                        }
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

/// Where a released planned crash of `planned` lands. On a run that does
/// not retarget this is `planned` itself and nothing is read or counted. On
/// a treated run the crash moves to the live server that most recently took
/// a fault-crossing delivery, unless the plan still has a crash or recover
/// outstanding on that server or the best absorber is `planned` itself.
/// Candidates are the servers, which are the first `servers` nodes; the
/// clients that follow them are never crashed.
fn retarget_crash<H: HashPolicy>(state: &State<H>, planned: NodeId, servers: usize) -> NodeId {
    if !state.retarget.enabled {
        return planned;
    }
    let decision = absorber_decision(state, planned, servers);
    let (outcome, victim) = match decision.choice {
        ghost_absorber::Choice::Retarget { node, acted } => (
            util_stats::VictimSwap::Applied { acted },
            NodeId {
                role: planned.role,
                index: node,
            },
        ),
        ghost_absorber::Choice::SameVictim => (util_stats::VictimSwap::SameVictim, planned),
        ghost_absorber::Choice::NoAbsorber => (util_stats::VictimSwap::NoAbsorber, planned),
    };
    util_stats::record_victim_swap(outcome, decision.skipped_pending_pair);
    victim
}

/// The absorber ranking's choice for a crash planned on `planned`, read on
/// the current ledgers. A node that is down cannot be crashed, and a node
/// the plan still has a crash or recover outstanding on, or whose own
/// crash is still queued, is passed over. Both the hold-time read and the
/// apply-time retarget consult this, so they agree on the same ledgers.
fn absorber_decision<H: HashPolicy>(
    state: &State<H>,
    planned: NodeId,
    servers: usize,
) -> ghost_absorber::Decision {
    ghost_absorber::choose(
        planned.index,
        &state.send_ledger,
        servers,
        |n| {
            !state.crash_info.currently_crashed.contains(&NodeId {
                role: planned.role,
                index: n,
            })
        },
        |n| {
            state.retarget.has_pending_pair(n)
                || state.send_ledger.get(n).is_some_and(|l| l.crash_pending > 0)
        },
    )
}

/// The network-queue index a step takes once the within-queue draw has
/// settled on `drawn`. This is `drawn` itself on a run that does not prefer
/// fresh records, and at any step where the drawn item is not a remote
/// record or has no eligible rival of the opposite incarnation class from
/// the same sender to the same destination. On a treated run whose draw
/// fell on a ghost - a record whose sender restarted since sending - with a
/// live destination and such a fresh rival, it is the rival with the highest
/// priority, the lowest index among equals. The sender's ledger says in
/// O(1) whether it has both a fresh and a stale record in the queue, and
/// nothing else is read when it does not. The displaced ghost stays in the
/// queue and stays eligible. No random draw is taken here, so a treated
/// step reads the same random sequence as an untreated one.
fn fresh_first_dispatch<H: HashPolicy>(
    state: &mut State<H>,
    eligible: &[usize],
    drawn: usize,
) -> usize {
    let stats = util_stats::enabled();
    let treated = state.fresh_first.enabled;
    if !treated && !stats {
        return drawn;
    }
    let queue = &state.network_queue;
    let chosen = 'pick: {
        let Some(Runnable::Record(rec)) = queue.get(drawn) else {
            break 'pick drawn;
        };
        if rec.origin_node == rec.node {
            break 'pick drawn;
        }
        let origin = rec.origin_node;
        let dest = rec.node;
        let Some(ledger) = state.send_ledger.get(origin.index) else {
            break 'pick drawn;
        };
        if ledger.net_fresh == 0 || ledger.net_records <= ledger.net_fresh {
            break 'pick drawn;
        }
        let current = state.incarnation(origin);
        let stale_drawn = rec.origin_incarnation != current;
        let candidates = eligible.iter().map(|&i| {
            let item = match &queue[i] {
                Runnable::Record(r)
                    if r.origin_node == origin && r.node == dest && r.origin_node != r.node =>
                {
                    Some((r.origin_incarnation, r.priority))
                }
                _ => None,
            };
            (i, item)
        });
        let (contested, best_fresh) = fresh_first::rival(candidates, drawn, stale_drawn, current);
        if !contested {
            break 'pick drawn;
        }
        let down = state.crash_info.currently_crashed.contains(&dest);
        util_stats::record_fresh_first_contest(treated, stale_drawn, down);
        if !treated || !stale_drawn || down {
            break 'pick drawn;
        }
        let Some(fresh) = best_fresh else {
            break 'pick drawn;
        };
        let times = state.fresh_first.displace((origin.index, rec.send_ordinal));
        util_stats::record_fresh_first_swap(times > 1);
        fresh
    };
    if treated
        && let Some(Runnable::Record(r)) = queue.get(chosen)
        && r.origin_node != r.node
        && let Some(count) = state.fresh_first.take((r.origin_node.index, r.send_ordinal))
    {
        util_stats::record_fresh_first_taken(count);
    }
    chosen
}

/// Steps between two reads of the rushed-dispatch split.
const RUSH_DISPATCH_STRIDE: i32 = 64;

/// Split the network steps that had a record of a rushed client operation
/// among their candidates by whether the step dispatched one. The layers
/// that run ahead of the score can take another record even when a rushed
/// one carries the top priority, and only this split says how often they do.
/// Every step would pay a scan of the candidates, so one step in
/// `RUSH_DISPATCH_STRIDE` is read and the split is that sample's.
fn observe_rush_dispatch<H: HashPolicy>(state: &State<H>, eligible: &[usize], pick: usize) {
    if state.crash_info.current_step % RUSH_DISPATCH_STRIDE != 0 {
        return;
    }
    if state.client_anchor.rushed_ops.is_empty() || !util_stats::enabled() {
        return;
    }
    let rushed = |i: usize| {
        state
            .client_anchor
            .rushes(state.network_queue[i].causal_operation_id())
    };
    if eligible.iter().any(|&i| rushed(i)) {
        util_stats::record_client_anchor_rush_dispatch(rushed(pick));
    }
}

/// A record of a client operation that became ready after the run's first
/// crash is about to be delivered. The first such delivery of an operation
/// gives the steps it spent between issue and arrival, which every
/// direction of the request-timing axis reports.
fn note_first_delivery<H: HashPolicy>(state: &mut State<H>, op: Option<i32>) {
    if state.client_anchor.awaiting_delivery.is_empty() {
        return;
    }
    let Some(op) = op else {
        return;
    };
    if let Some(invoked_step) = state.client_anchor.awaiting_delivery.remove(&op) {
        let steps = (state.crash_info.current_step - invoked_step).max(0) as u64;
        util_stats::record_client_anchor_first_delivery(state.client_anchor.arm, steps);
    }
}

/// After the draw and the fresh-incarnation swap have settled on `pick`: on
/// a treated run, a remote record whose sending incarnation is not the one
/// running now - which covers every record of a sender that is down - gives
/// way to the eligible record from the same sender to the same destination,
/// sent by the same incarnation, that carries the lowest send ordinal, when
/// that ordinal is below the pick's. The scan is gated on the sender's
/// ledger holding at least one other remote record in the queue. A pick of
/// the sender's current incarnation keeps the order the draw gave it, and
/// the replacement it would otherwise have taken is counted as suppressed.
/// Records of a sender that never crashed, and anything but a remote
/// record, keep the pick. A pick with an eligible rival of its class is
/// counted as a contest, with whether the pick already carried the lowest
/// ordinal: on every treated run, where the scan happens anyway, and on
/// census runs of the control half; any other untreated run returns without
/// reading the queue. No random draw is taken.
fn pair_order_dispatch<H: HashPolicy>(state: &State<H>, eligible: &[usize], pick: usize) -> usize {
    let treated = state.pair_order.enabled;
    if !treated && !state.pair_order.census {
        return pick;
    }
    let queue = &state.network_queue;
    let Some(Runnable::Record(rec)) = queue.get(pick) else {
        return pick;
    };
    if rec.origin_node == rec.node {
        return pick;
    }
    let origin = rec.origin_node;
    let dest = rec.node;
    if state.incarnation(origin) == 0 && !state.crash_info.currently_crashed.contains(&origin) {
        return pick;
    }
    if !state
        .send_ledger
        .get(origin.index)
        .is_some_and(|l| l.net_records >= 2)
    {
        return pick;
    }
    let candidates = eligible.iter().map(|&i| {
        let item = match &queue[i] {
            Runnable::Record(r)
                if r.origin_node == origin && r.node == dest && r.origin_node != r.node =>
            {
                Some((r.origin_incarnation, r.send_ordinal))
            }
            _ => None,
        };
        (i, item)
    });
    let contest = pair_order::contest(candidates, pick, rec.origin_incarnation, rec.send_ordinal);
    if !contest.rivals {
        return pick;
    }
    util_stats::record_pair_order_contest(treated, contest.earliest.is_none());
    if !treated {
        return pick;
    }
    let ghost = state.fault_crossing(origin, rec.origin_incarnation);
    util_stats::record_pair_order_contest_class(ghost);
    match contest.earliest {
        Some(earliest) if ghost => {
            util_stats::record_pair_order_correction(ghost);
            earliest
        }
        Some(_) => {
            util_stats::record_pair_order_fresh_suppressed();
            pick
        }
        None => pick,
    }
}

fn crash_node<H: HashPolicy>(state: &mut State<H>, node_id: NodeId) {
    if state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is already crashed", node_id);
        return;
    }
    state.crash_info.currently_crashed.insert(node_id);
    state.note_handler_entry(node_id.index, HandlerTrigger::None);
    state.clear_ghost_mark(node_id.index);

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
    state.fresh_first.clear_origin(node_id.index);
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

    /// Four crashed nodes with distinct incarnations, each with a recover
    /// waiting for it, plus a runnable the recovery weighting does not apply
    /// to.
    fn crashed_state_with_recovers() -> (State<NoHashing>, Vec<Runnable<NoHashing>>) {
        let mut state = State::<NoHashing>::new(&[(crate::analysis::resolver::NameId(0), 4)], 1);
        let mut queue = Vec::new();
        for index in 0..4usize {
            let node_id = NodeId {
                role: crate::analysis::resolver::NameId(0),
                index,
            };
            state.crash_info.currently_crashed.insert(node_id);
            state.incarnations[index] = index as u32;
            queue.push(Runnable::Recover {
                node_id,
                priority: 0.2 * (index as f64 + 1.0),
            });
        }
        queue.push(heal(0.5));
        (state, queue)
    }

    /// A pending crash is withheld while its node's hold has not expired,
    /// counted once per withheld offer, and released the moment the target
    /// step arrives; a hold on a node with no queued crash withholds
    /// nothing.
    #[test]
    fn a_crash_hold_withholds_a_pending_crash_until_its_target_step() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let mut state = State::<NoHashing>::new(&[(crate::analysis::resolver::NameId(0), 2)], 1);
        let node = NodeId {
            role: crate::analysis::resolver::NameId(0),
            index: 1,
        };
        state.push_runnable(Runnable::Crash {
            node_id: node,
            priority: 0.5,
        });
        state.crash_hold_until[1] = 10;
        let mut rng = StdRng::seed_from_u64(5);

        state.crash_info.current_step = 4;
        assert_eq!(crash_hold_mask(&mut state, 2, &mut rng), 1u64 << 1);
        state.crash_info.current_step = 10;
        assert_eq!(
            crash_hold_mask(&mut state, 2, &mut rng),
            0,
            "the hold expires at its target step"
        );

        state.crash_info.current_step = 4;
        state.crash_hold_until[0] = 10;
        assert_eq!(
            crash_hold_mask(&mut state, 2, &mut rng),
            1u64 << 1,
            "node 0 has no pending crash"
        );

        let holds = util_stats::snapshot().crash_place.holds;
        util_stats::set_enabled(false);
        assert_eq!(holds, 2, "one count per withheld offer");
    }

    /// On a run that anchors, the expiring step hold hands the crash to the
    /// fan-out wait, which keeps withholding it until the victim's segment
    /// shows the drawn phase. The withheld steps stay out of the placement
    /// counters, whose equality is what says the step hold is the only gate
    /// they describe.
    #[test]
    fn an_anchored_crash_keeps_waiting_past_its_target_step() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place;
        let mut state = State::<NoHashing>::new(&[(crate::analysis::resolver::NameId(0), 2)], 1);
        let node = NodeId {
            role: crate::analysis::resolver::NameId(0),
            index: 1,
        };
        state.push_runnable(Runnable::Crash {
            node_id: node,
            priority: 0.5,
        });
        state.crash_hold_until[1] = 10;
        state.crash_phase.arm_node(1, 0);
        // A segment with nothing issued meets no waiting arm's phase.
        state.send_ledger[1].floor = state.send_ledger[1].issued;
        let mut rng = StdRng::seed_from_u64(9);

        state.crash_info.current_step = 10;
        let mut waited = 0;
        for step in 10..10 + crash_phase::WINDOW {
            state.crash_info.current_step = step;
            if crash_hold_mask(&mut state, 2, &mut rng) != 0 {
                waited += 1;
            }
        }
        let arm = state.crash_phase.arm_of(1);
        let place = util_stats::snapshot().crash_place;
        util_stats::set_enabled(false);
        match arm {
            Some(util_stats::CrashPhaseArm::Stock) => {
                assert_eq!(waited, 0, "a stock draw must not withhold anything")
            }
            Some(_) => assert_eq!(
                waited,
                crash_phase::WINDOW,
                "a waiting arm must withhold every step of its window"
            ),
            None => panic!("an armed node drew no arm"),
        }
        assert_eq!(
            place.holds - before.holds,
            0,
            "the fan-out wait must not count as a placement hold"
        );
    }

    /// The identity-weighted recovery term reads its predicate and reports
    /// what it preferred without being able to reach the selection: the same
    /// seed draws the same candidates with the term on and off, and the term
    /// never reports a flip.
    #[test]
    fn the_recovery_placebo_walk_cannot_change_a_selection() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let (state, queue) = crashed_state_with_recovers();
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let selector = WithinQueueSelector::Proportional { exponent: 1.0 };
        let terms = terms_with(5.0);
        let trials = 500usize;

        let carriers = eligible
            .iter()
            .filter(|&&i| {
                is_quick_fire(&queue[i], &state.crash_info.currently_crashed)
                    && uninformative_recovery_predicate(&queue[i], &state)
            })
            .count() as u64;
        assert!(carriers > 0, "the predicate holds of no candidate");

        let draw = |placebo: bool| {
            util_stats::set_enabled(true);
            util_stats::set_recovery_weight_placebo(placebo);
            let mut rng = StdRng::seed_from_u64(7);
            (0..trials)
                .map(|_| {
                    select_within_queue::<NoHashing, NoFeedback>(
                        &queue,
                        &eligible,
                        &(),
                        &(),
                        &state,
                        &terms,
                        &selector,
                        &mut rng,
                    )
                    .0
                })
                .collect::<Vec<usize>>()
        };
        let with_term = draw(true);
        let stats = util_stats::snapshot().recovery_weight_placebo;
        let without_term = draw(false);
        util_stats::set_recovery_weight_placebo(false);
        util_stats::set_enabled(false);

        assert_eq!(with_term, without_term);
        assert_eq!(stats.decisions, trials as u64);
        assert_eq!(stats.evaluated, carriers * trials as u64);
        assert_eq!(stats.present, trials as u64);
        assert_eq!(stats.contested, trials as u64);
        assert_eq!(stats.flipped, 0);
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

    /// The node a waiting crash's phase is read on is the node the retarget
    /// would crash on the same ledgers, and the planned victim itself when
    /// the run does not retarget, has no release ahead for the node, or the
    /// ranking keeps or cannot move the crash.
    #[test]
    fn the_phase_is_read_where_the_retarget_would_land_and_falls_back_to_the_victim() {
        let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        let planned = node(0);
        state.push_runnable(Runnable::Crash {
            node_id: planned,
            priority: 0.5,
        });
        state.crash_phase.arm_node(0, 0);
        state.send_ledger[1].last_ghost_step = 40;
        state.send_ledger[1].last_ghost_acted = true;
        state.send_ledger[2].last_ghost_step = 9;

        assert_eq!(phase_read_node(&state, 0, 3), 0, "a run that does not retarget");
        state.retarget.enabled = true;
        assert_eq!(phase_read_node(&state, 0, 3), 1);
        assert_eq!(phase_read_node(&state, 0, 3), retarget_crash(&state, planned, 3).index);
        state.retarget.pending_pair_mask = 1 << 1;
        assert_eq!(phase_read_node(&state, 0, 3), 2, "an outstanding pair is passed over");
        assert_eq!(phase_read_node(&state, 0, 3), retarget_crash(&state, planned, 3).index);
        state.retarget.pending_pair_mask = 0;
        crash_node(&mut state, node(1));
        assert_eq!(phase_read_node(&state, 0, 3), 2, "a node that is down is not read");
        assert_eq!(phase_read_node(&state, 0, 3), retarget_crash(&state, planned, 3).index);

        state.send_ledger[0].last_ghost_step = 99;
        state.send_ledger[0].last_ghost_acted = true;
        assert_eq!(phase_read_node(&state, 0, 3), 0, "the victim at the top of the ranking");
        assert_eq!(retarget_crash(&state, planned, 3), planned);
        for l in state.send_ledger.iter_mut() {
            l.last_ghost_step = -1;
            l.last_ghost_acted = false;
        }
        assert_eq!(phase_read_node(&state, 0, 3), 0, "no mark anywhere");
        assert_eq!(retarget_crash(&state, planned, 3), planned);

        state.send_ledger[0].last_ghost_step = 5;
        state.push_runnable(Runnable::Crash {
            node_id: node(2),
            priority: 0.5,
        });
        assert_eq!(phase_read_node(&state, 2, 3), 2, "a crash with no release ahead is not read");
        state.crash_phase.arm_node(2, 0);
        assert_eq!(
            phase_read_node(&state, 2, 3),
            2,
            "a node whose own crash is still queued is passed over"
        );
        assert_eq!(phase_read_node(&state, 2, 3), retarget_crash(&state, node(2), 3).index);
        state.send_ledger[0].crash_pending = 0;
        assert_eq!(phase_read_node(&state, 2, 3), 0);
        assert_eq!(phase_read_node(&state, 2, 3), retarget_crash(&state, node(2), 3).index);
    }

    /// Two twins differing only in whether the run retargets draw the same
    /// arm and leave their streams at the same position; the retargeting
    /// twin releases when the absorber's segment shows the arm's phase
    /// while the other waits out its window on the victim's silent segment,
    /// and the landing counters count each crash once.
    #[test]
    fn a_retargeting_run_waits_on_the_absorber_s_fanout_and_draws_like_its_twin() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_phase.landing;
        let build = |retarget: bool| {
            let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
            state.push_runnable(Runnable::Crash {
                node_id: node(0),
                priority: 0.5,
            });
            state.crash_hold_until[0] = 10;
            state.crash_phase.arm_node(0, 0);
            state.retarget.enabled = retarget;
            state.send_ledger[2].last_ghost_step = 40;
            state.send_ledger[2].last_ghost_acted = true;
            state
        };
        let meet = |ledger: &mut crate::simulator::core::state::SendLedger,
                    arm: util_stats::CrashPhaseArm| {
            ledger.issued = 2;
            ledger.floor = 0;
            ledger.recent = match arm {
                util_stats::CrashPhaseArm::Early => 2,
                _ => 1,
            };
            ledger.in_flight = ledger.recent;
        };
        let seeds = 40u64;
        let mut waiting = 0u64;
        for seed in 0..seeds {
            let mut control = build(false);
            let mut retargeting = build(true);
            let mut rng_c = StdRng::seed_from_u64(seed);
            let mut rng_r = StdRng::seed_from_u64(seed);
            let mut held_c = 0;
            let mut held_r = 0;
            for step in 10..=10 + crash_phase::WINDOW {
                control.crash_info.current_step = step;
                retargeting.crash_info.current_step = step;
                if step == 11 {
                    let arm = retargeting.crash_phase.arm_of(0).expect("the retargeting twin drew");
                    assert_eq!(control.crash_phase.arm_of(0), Some(arm), "the twins drew apart");
                    meet(&mut control.send_ledger[2], arm);
                    meet(&mut retargeting.send_ledger[2], arm);
                }
                held_c += (crash_hold_mask(&mut control, 3, &mut rng_c) != 0) as i32;
                held_r += (crash_hold_mask(&mut retargeting, 3, &mut rng_r) != 0) as i32;
            }
            assert_eq!(
                rng_c.next_u64(),
                rng_r.next_u64(),
                "seed {seed}: the twins consumed different streams"
            );
            match retargeting.crash_phase.arm_of(0) {
                Some(util_stats::CrashPhaseArm::Stock) => {
                    assert_eq!((held_c, held_r), (0, 0), "seed {seed}: a stock draw held");
                }
                Some(_) => {
                    waiting += 1;
                    assert_eq!(held_c, crash_phase::WINDOW, "seed {seed}: the control twin");
                    assert_eq!(held_r, 1, "seed {seed}: the retargeting twin");
                }
                None => panic!("seed {seed}: no arm drawn"),
            }
        }
        let after = util_stats::snapshot().crash_phase.landing;
        util_stats::set_enabled(false);
        assert!(waiting > 0, "no seed drew a waiting arm");
        assert_eq!(after.evaluated_on_other_node - before.evaluated_on_other_node, seeds);
        assert_eq!(after.condition_on_other_node - before.condition_on_other_node, waiting);
        assert_eq!(after.expired_on_other_node - before.expired_on_other_node, 0);
    }

    /// A crash wipes what its node absorbed, so a node that comes back never
    /// ranks on a mark from a state it no longer holds; a run that does not
    /// retarget keeps the plan's victim without consulting the marks.
    #[test]
    fn a_crash_clears_the_node_s_ghost_mark_and_only_treated_runs_retarget() {
        let role = crate::analysis::resolver::NameId(0);
        let mut state = State::<NoHashing>::new(&[(role, 3)], 1);
        state.send_ledger[1].last_ghost_step = 40;
        state.send_ledger[1].last_ghost_acted = true;
        state.send_ledger[2].last_ghost_step = 9;
        let planned = NodeId { role, index: 0 };
        assert_eq!(retarget_crash(&state, planned, 3), planned, "an untreated run keeps the victim");

        state.retarget.enabled = true;
        assert_eq!(retarget_crash(&state, planned, 3), NodeId { role, index: 1 });
        state.retarget.pending_pair_mask = 1 << 1;
        assert_eq!(retarget_crash(&state, planned, 3), NodeId { role, index: 2 });
        state.retarget.pending_pair_mask = 0;
        state.send_ledger[2].crash_pending = 1;
        assert_eq!(retarget_crash(&state, NodeId { role, index: 1 }, 3), NodeId { role, index: 1 });

        crash_node(&mut state, NodeId { role, index: 1 });
        assert_eq!(state.send_ledger[1].last_ghost_step, -1);
        assert!(!state.send_ledger[1].last_ghost_acted);
        assert_eq!(state.send_ledger[2].last_ghost_step, 9, "another node's mark stays");
        state.send_ledger[2].crash_pending = 0;
        assert_eq!(
            retarget_crash(&state, planned, 3),
            NodeId { role, index: 2 },
            "a node that is down is not a candidate"
        );
    }

    const ROLE: crate::analysis::resolver::NameId = crate::analysis::resolver::NameId(0);

    fn node(index: usize) -> NodeId {
        NodeId { role: ROLE, index }
    }

    /// A remote record from `origin` to `dest` sent at `incarnation`, queued
    /// through the ledger hooks so the sender's counts stay exact.
    fn queue_record(
        state: &mut State<NoHashing>,
        origin: usize,
        dest: usize,
        incarnation: u32,
        priority: f64,
    ) -> usize {
        let env = Env::<NoHashing>::with_slots(1);
        let rec = Record {
            pc: 0,
            node: node(dest),
            origin_node: node(origin),
            continuation: Continuation::Recover,
            entry_pc: 0,
            initial_env: env.clone(),
            env,
            priority,
            causal_operation_id: None,
            trace_id: None,
            link_seq: None,
            origin_incarnation: incarnation,
            bias: DeliveryBias::NONE,
            timer_entry: None,
            send_ordinal: state.next_send_ordinal(node(origin)),
            receiver_token_at_send: state.node_state_token(node(dest)),
        };
        state.push_runnable(Runnable::Record(rec));
        state.network_queue.len() - 1
    }

    fn queue_channel_send(state: &mut State<NoHashing>, origin: usize, dest: usize) -> usize {
        state.push_runnable(Runnable::ChannelSend {
            target: node(dest),
            channel: crate::simulator::core::values::ChannelId {
                node: node(dest),
                id: 0,
            },
            message: Value::<NoHashing>::unit(),
            origin_node: node(origin),
            pc: 0,
            priority: 0.5,
        });
        state.network_queue.len() - 1
    }

    /// Three nodes; node 0 has restarted once, so its incarnation is 1. The
    /// queue holds, in order: a ghost 0->1, a fresh 0->1 at priority 0.3, a
    /// channel send 0->1, a fresh 0->2, a fresh 0->1 at priority 0.8, and a
    /// second fresh 0->1 at priority 0.8.
    fn contested_state() -> (State<NoHashing>, Vec<usize>) {
        let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        let ghost = queue_record(&mut state, 0, 1, 0, 0.9);
        state.incarnations[0] = 1;
        state.note_incarnation_bump(0);
        let low = queue_record(&mut state, 0, 1, 1, 0.3);
        let chan = queue_channel_send(&mut state, 0, 1);
        let other_dest = queue_record(&mut state, 0, 2, 1, 0.95);
        let high = queue_record(&mut state, 0, 1, 1, 0.8);
        let high_later = queue_record(&mut state, 0, 1, 1, 0.8);
        (state, vec![ghost, low, chan, other_dest, high, high_later])
    }

    #[test]
    fn a_treated_step_takes_the_best_fresh_rival_in_place_of_a_drawn_ghost() {
        let (mut state, slots) = contested_state();
        let [ghost, low, chan, other_dest, high, high_later] = slots[..] else { unreachable!() };
        let eligible: Vec<usize> = (0..state.network_queue.len()).collect();
        assert_eq!(fresh_first_dispatch(&mut state, &eligible, ghost), ghost, "an untreated run keeps the draw");

        state.fresh_first.enabled = true;
        assert_eq!(
            fresh_first_dispatch(&mut state, &eligible, ghost),
            high,
            "the highest-priority fresh rival wins, the lowest index among equals"
        );
        assert_eq!(state.fresh_first.displaced_len(), 1, "the ghost carries a displaced count");
        for drawn in [low, chan, other_dest, high, high_later] {
            assert_eq!(fresh_first_dispatch(&mut state, &eligible, drawn), drawn, "a fresh draw or a channel send is never displaced");
        }
        assert_eq!(state.network_queue.len(), 6, "nothing was taken or masked");

        let only_high = vec![ghost, low, chan, high_later];
        assert_eq!(fresh_first_dispatch(&mut state, &only_high, ghost), high_later, "an ineligible rival is not chosen");
        let ghost_ordinal = match &state.network_queue[ghost] {
            Runnable::Record(r) => r.send_ordinal,
            _ => unreachable!(),
        };
        let mut peek = state.fresh_first.clone();
        assert_eq!(peek.take((0, ghost_ordinal)), Some(fresh_first::DisplacedCount::Twice));

        // A step that keeps the ghost takes it, which closes its count.
        state.crash_info.currently_crashed.insert(node(1));
        assert_eq!(fresh_first_dispatch(&mut state, &eligible, ghost), ghost, "a crashed destination sees no swap");
        state.crash_info.currently_crashed.remove(&node(1));
        assert_eq!(state.fresh_first.displaced_len(), 0, "the taken ghost keeps no count");

        assert_eq!(fresh_first_dispatch(&mut state, &eligible, ghost), high);
        let ghost_only: Vec<usize> = vec![ghost, chan, other_dest];
        assert_eq!(fresh_first_dispatch(&mut state, &ghost_only, ghost), ghost, "no eligible fresh rival keeps the ghost");
        assert_eq!(state.fresh_first.displaced_len(), 0);
    }

    #[test]
    fn the_ledger_gate_is_consulted_before_the_queue_is_read() {
        let (mut state, slots) = contested_state();
        let ghost = slots[0];
        let eligible: Vec<usize> = (0..state.network_queue.len()).collect();
        state.fresh_first.enabled = true;
        let ledger = state.send_ledger[0];
        assert!(ledger.net_fresh > 0 && ledger.net_records > ledger.net_fresh, "the fixture holds both classes");

        state.send_ledger[0].net_fresh = 0;
        assert_eq!(fresh_first_dispatch(&mut state, &eligible, ghost), ghost, "a ledger with no fresh record closes the gate");
        state.send_ledger[0].net_fresh = ledger.net_records;
        assert_eq!(fresh_first_dispatch(&mut state, &eligible, ghost), ghost, "a ledger with no stale record closes the gate");
        state.send_ledger[0] = ledger;
        assert_ne!(fresh_first_dispatch(&mut state, &eligible, ghost), ghost);

        let mut fresh_only = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        let a = queue_record(&mut fresh_only, 0, 1, 0, 0.9);
        queue_record(&mut fresh_only, 0, 1, 0, 0.1);
        fresh_only.fresh_first.enabled = true;
        assert_eq!(fresh_first_dispatch(&mut fresh_only, &[a, a + 1], a), a, "two fresh records are not a contest");
    }

    /// The swap takes no draw: after the within-queue selection a treated
    /// step and an untreated one leave the generator at the same position.
    #[test]
    fn the_treated_step_reads_the_same_random_sequence_as_the_untreated_one() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let (state, slots) = contested_state();
        let ghost = slots[0];
        let eligible: Vec<usize> = (0..state.network_queue.len()).collect();
        let selector = WithinQueueSelector::Tournament { k: 3 };
        let mut positions = Vec::new();
        for treated in [false, true, false, true] {
            let mut state = state.clone();
            state.fresh_first.enabled = treated;
            let mut rng = StdRng::seed_from_u64(77);
            let (drawn, _) = select_within_queue::<NoHashing, NoFeedback>(
                &state.network_queue,
                &eligible,
                &(),
                &(),
                &state,
                &ResolvedTerms::default(),
                &selector,
                &mut rng,
            );
            let taken = fresh_first_dispatch(&mut state, &eligible, drawn);
            if drawn == ghost {
                assert_eq!(taken != drawn, treated, "only the treated step swaps");
            } else {
                assert_eq!(taken, drawn);
            }
            positions.push((drawn, rng.next_u64(), rng.next_u64()));
        }
        assert!(positions.windows(2).all(|w| w[0] == w[1]), "the halves diverged: {positions:?}");
        let mut untouched = StdRng::seed_from_u64(77);
        let mut probe = StdRng::seed_from_u64(77);
        let mut state = state.clone();
        state.fresh_first.enabled = true;
        assert_ne!(fresh_first_dispatch(&mut state, &eligible, ghost), ghost);
        assert_eq!(probe.next_u64(), untouched.next_u64(), "the swap itself took a draw");
    }

    /// Three nodes; node 0 has restarted once, so its incarnation is 1. The
    /// queue holds, in order: two ghosts 0->1 in send order, a fresh 0->1, a
    /// ghost 0->2, a second fresh 0->1, a channel send 0->1, and two records
    /// 2->1 from a node that never crashed, in send order.
    fn ordered_state() -> (State<NoHashing>, Vec<usize>) {
        let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        let ghost_a = queue_record(&mut state, 0, 1, 0, 0.9);
        let ghost_b = queue_record(&mut state, 0, 1, 0, 0.4);
        state.incarnations[0] = 1;
        state.note_incarnation_bump(0);
        let fresh_a = queue_record(&mut state, 0, 1, 1, 0.3);
        let ghost_other = queue_record(&mut state, 0, 2, 0, 0.5);
        let fresh_b = queue_record(&mut state, 0, 1, 1, 0.8);
        let chan = queue_channel_send(&mut state, 0, 1);
        let quiet_a = queue_record(&mut state, 2, 1, 0, 0.6);
        let quiet_b = queue_record(&mut state, 2, 1, 0, 0.7);
        (
            state,
            vec![ghost_a, ghost_b, fresh_a, ghost_other, fresh_b, chan, quiet_a, quiet_b],
        )
    }

    #[test]
    fn a_treated_step_takes_the_earliest_send_of_a_dead_incarnation_s_class() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let (mut state, slots) = ordered_state();
        let [ghost_a, ghost_b, fresh_a, ghost_other, fresh_b, chan, quiet_a, quiet_b] = slots[..] else {
            unreachable!()
        };
        let eligible: Vec<usize> = (0..state.network_queue.len()).collect();
        for pick in &eligible {
            assert_eq!(pair_order_dispatch(&state, &eligible, *pick), *pick, "an untreated run keeps the pick");
        }
        state.pair_order.census = true;
        for pick in &eligible {
            assert_eq!(pair_order_dispatch(&state, &eligible, *pick), *pick, "an untreated census run keeps the pick");
        }
        state.pair_order.census = false;

        state.pair_order.enabled = true;
        assert_eq!(pair_order_dispatch(&state, &eligible, ghost_b), ghost_a, "the later ghost gives way to the earlier one");
        assert_eq!(
            pair_order_dispatch(&state, &eligible, fresh_b),
            fresh_b,
            "a pick of the incarnation running now keeps the order the draw gave it"
        );
        for pick in [ghost_a, fresh_a] {
            assert_eq!(pair_order_dispatch(&state, &eligible, pick), pick, "the earliest of its class is kept");
        }
        assert_eq!(pair_order_dispatch(&state, &eligible, ghost_other), ghost_other, "another destination has no rival");
        assert_eq!(pair_order_dispatch(&state, &eligible, chan), chan, "a channel send is never replaced");
        assert_eq!(pair_order_dispatch(&state, &eligible, quiet_b), quiet_b, "a sender that never crashed keeps the pick");
        assert_eq!(pair_order_dispatch(&state, &eligible, quiet_a), quiet_a);
        assert_eq!(state.network_queue.len(), 8, "nothing was taken or masked");

        let without_earliest: Vec<usize> = eligible.iter().copied().filter(|&i| i != ghost_a).collect();
        assert_eq!(pair_order_dispatch(&state, &without_earliest, ghost_b), ghost_b, "an ineligible earlier sibling is not chosen");
        let only_fresh_b = vec![ghost_a, fresh_b, chan];
        assert_eq!(pair_order_dispatch(&state, &only_fresh_b, fresh_b), fresh_b, "no eligible rival keeps the pick");

        state.crash_info.currently_crashed.insert(node(2));
        assert_eq!(
            pair_order_dispatch(&state, &eligible, quiet_b),
            quiet_a,
            "a sender that is down has no incarnation running, so its records are the dead class"
        );
        state.crash_info.currently_crashed.remove(&node(2));

        let ledger = state.send_ledger[0];
        state.send_ledger[0].net_records = 1;
        assert_eq!(pair_order_dispatch(&state, &eligible, ghost_b), ghost_b, "a ledger with a single record closes the gate");
        state.send_ledger[0] = ledger;
        assert_eq!(pair_order_dispatch(&state, &eligible, ghost_b), ghost_a);
    }

    /// The dead class is counted and replaced, the current incarnation's
    /// class is counted and left where the draw put it.
    #[test]
    fn a_treated_step_prices_the_replacement_it_declines_to_make() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let (mut state, slots) = ordered_state();
        let [ghost_a, ghost_b, _, _, fresh_b, _, _, _] = slots[..] else {
            unreachable!()
        };
        let eligible: Vec<usize> = (0..state.network_queue.len()).collect();
        state.pair_order.enabled = true;
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().pair_order;
        pair_order_dispatch(&state, &eligible, ghost_b);
        pair_order_dispatch(&state, &eligible, fresh_b);
        pair_order_dispatch(&state, &eligible, ghost_a);
        let after = util_stats::snapshot().pair_order;
        util_stats::set_enabled(false);

        assert_eq!(after.corrections_ghost - before.corrections_ghost, 1);
        assert_eq!(
            after.corrections_fresh - before.corrections_fresh,
            0,
            "a pick of the incarnation running now was replaced"
        );
        assert_eq!(
            after.corrections_fresh_suppressed - before.corrections_fresh_suppressed,
            1
        );
        assert_eq!(after.contests_by_class.ghost - before.contests_by_class.ghost, 2);
        assert_eq!(after.contests_by_class.fresh - before.contests_by_class.fresh, 1);
        assert_eq!(after.corrected - before.corrected, 1);
    }

    /// The replacement takes no draw: after the within-queue selection a
    /// treated step and an untreated one leave the generator at the same
    /// position.
    #[test]
    fn the_pair_order_step_reads_the_same_random_sequence_on_both_halves() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let (state, slots) = ordered_state();
        let later = [slots[1], slots[4]];
        let later_ghost = slots[1];
        let eligible: Vec<usize> = (0..state.network_queue.len()).collect();
        let selector = WithinQueueSelector::Tournament { k: 3 };
        let mut positions = Vec::new();
        for seed in 0..32u64 {
            for treated in [false, true] {
                let mut state = state.clone();
                state.pair_order.enabled = treated;
                let mut rng = StdRng::seed_from_u64(seed);
                let (drawn, _) = select_within_queue::<NoHashing, NoFeedback>(
                    &state.network_queue,
                    &eligible,
                    &(),
                    &(),
                    &state,
                    &ResolvedTerms::default(),
                    &selector,
                    &mut rng,
                );
                let taken = pair_order_dispatch(&state, &eligible, drawn);
                let replaces = treated && drawn == later_ghost;
                if later.contains(&drawn) {
                    assert_eq!(
                        taken != drawn,
                        replaces,
                        "a later send of the wrong class was replaced"
                    );
                } else {
                    assert_eq!(taken, drawn);
                }
                positions.push((seed, drawn, rng.next_u64(), rng.next_u64()));
            }
        }
        assert!(
            positions.chunks(2).all(|w| w[0] == w[1]),
            "the halves diverged: {positions:?}"
        );
        assert!(
            positions.iter().any(|p| p.1 == later_ghost),
            "no draw fell on a later send of the dead class, so the replacement was never exercised"
        );
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
