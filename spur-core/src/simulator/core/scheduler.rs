use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::build_frame;
use crate::simulator::core::exec::{exec, exec_sync_on_node};
use crate::simulator::core::partition::{activate_partition, heal_partition};
use crate::simulator::core::queue_selector::{
    QueueInfo, QueueSelection, QueueSelector, TimerBiasUse, WithinQueueSelector,
    select_with_timer_context,
};
use crate::simulator::core::state::{
    Continuation, HandlerTrigger, Logger, NodeId, PurgatoryConfig, Record, ReplayCut, Runnable,
    RunnableCategory, SchedulePolicy, ScheduleResult, State,
};
use crate::simulator::core::steer_terms::{ResolvedTerms, Term, TERMS};
use crate::simulator::core::values::{Env, Value};
use ecow::EcoVec;
use crate::simulator::crash_phase;
use crate::simulator::fresh_first;
use crate::simulator::ghost_absorber;
use crate::simulator::pair_order;
use crate::simulator::ghost_release;
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

/// Every index below its length, lent as the eligible list of a queue whose
/// eligibility pass admitted all of its runnables.
static IDENTITY_INDICES: [usize; 128] = {
    let mut indices = [0usize; 128];
    let mut i = 0;
    while i < indices.len() {
        indices[i] = i;
        i += 1;
    }
    indices
};

/// The eligible indices of one queue, in queue order.
enum EligibleList {
    /// Every index below the queue length.
    Known(usize),
    /// The indices a filter over the queue kept.
    Built(Vec<usize>),
}

impl EligibleList {
    /// The eligible list of a queue holding `len` runnables, of which the
    /// eligibility pass that sized the queue for selection admitted
    /// `admitted`. `keep` must be that same predicate taken by index, and
    /// nothing it reads may have changed since that pass: a queue admitted
    /// in full is then known to keep every index without a second pass.
    #[inline(always)]
    fn new(len: usize, admitted: usize, keep: impl FnMut(&usize) -> bool) -> Self {
        if admitted == len && len <= IDENTITY_INDICES.len() {
            EligibleList::Known(len)
        } else {
            let mut built = Vec::with_capacity(admitted);
            built.extend((0..len).filter(keep));
            EligibleList::Built(built)
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[usize] {
        match self {
            EligibleList::Known(len) => &IDENTITY_INDICES[..*len],
            EligibleList::Built(built) => built,
        }
    }
}

/// Counts how the non-empty eligible list of a queue holding `len`
/// runnables, `admitted` of them eligible, was obtained.
#[inline(always)]
fn record_eligible_list(len: usize, admitted: usize) {
    if util_stats::enabled() {
        let full = admitted == len;
        util_stats::record_eligible_list(
            full && len <= IDENTITY_INDICES.len(),
            full && len > IDENTITY_INDICES.len(),
        );
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
    expire_release_trigger(state, step_now);
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

/// The trigger a restart armed runs out at the bound: nothing is released
/// after it and every held crash keeps its target.
fn expire_release_trigger<H: HashPolicy>(state: &mut State<H>, step_now: i32) {
    if let Some(t) = state.ghost_release.trigger
        && step_now >= t.expires
    {
        state.ghost_release.trigger = None;
        util_stats::record_ghost_release_trigger(util_stats::GhostReleaseTrigger::Expired);
    }
}

/// A restart of `origin` at `restart_step` on a releasing run arms the
/// trigger when some other node's planned crash is still held; a trigger
/// still armed from an earlier restart is replaced. No hold moves here, and
/// below the learner's floor nothing is armed. Nothing here reads a random
/// stream.
fn arm_release_trigger<H: HashPolicy>(state: &mut State<H>, origin: usize, restart_step: i32) {
    if !state.ghost_release.releases() {
        return;
    }
    if !ghost_release::any_other_held(
        &state.crash_hold_until,
        &state.send_ledger,
        origin,
        restart_step,
    ) {
        return;
    }
    util_stats::record_ghost_release_restart_with_held_crash();
    let gr = &mut state.ghost_release;
    let Some(bound) = gr.bound else {
        return;
    };
    if gr.trigger.is_some() {
        util_stats::record_ghost_release_trigger(util_stats::GhostReleaseTrigger::Superseded);
    }
    gr.trigger = Some(ghost_release::Trigger {
        origin,
        restart_step,
        expires: restart_step.saturating_add(bound),
    });
    util_stats::record_ghost_release_trigger(util_stats::GhostReleaseTrigger::Armed);
}

/// What one firing released: how many holds moved and, over the first 64
/// nodes, whose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Released {
    count: u64,
    mask: u64,
}

fn node_bit(n: usize) -> u64 {
    if n < u64::BITS as usize {
        1u64 << n
    } else {
        0
    }
}

/// A message entry at `dest` sent by `origin` at incarnation `sent_at`
/// landed at `step` and `acted` says whether it wrote `dest`'s state. An
/// entry from a dead incarnation of a live origin at a live destination is
/// a ghost lag sample - the step less the origin's last restart step - fed
/// to the learner on the runs that feed it. On a releasing run the first
/// such entry after the armed restart, from the restarted node and writing
/// state, fires the trigger: every other held crash goes at the next step,
/// or on the single half the one crash `release_one` names.
fn note_ghost_release_entry<H: HashPolicy>(
    state: &mut State<H>,
    origin: NodeId,
    sent_at: u32,
    dest: NodeId,
    step: i32,
    acted: bool,
    servers: usize,
) {
    if origin == dest || sent_at == state.incarnation(origin) {
        return;
    }
    let down = &state.crash_info.currently_crashed;
    if down.contains(&dest) {
        return;
    }
    let last_restart = state
        .send_ledger
        .get(origin.index)
        .map_or(-1, |l| l.last_restart_step);
    if !down.contains(&origin) && last_restart >= 0 && state.ghost_release.feeds_learner {
        ghost_release::merge_probe_lag(state.ghost_release.scope, step - last_restart);
    }
    let Some(t) = state.ghost_release.trigger else {
        return;
    };
    if !acted || t.origin != origin.index || step <= t.restart_step {
        return;
    }
    let released = if state.ghost_release.single() {
        match release_one(state, t.origin, dest, step, servers) {
            Some(r) => r,
            None => {
                util_stats::record_ghost_release_single(util_stats::SingleRelease::NoCase);
                return;
            }
        }
    } else {
        release_all(state, t.origin, step)
    };
    let gr = &mut state.ghost_release;
    gr.released_mask |= released.mask;
    gr.ghost_node = Some(dest.index);
    gr.trigger = None;
    util_stats::record_ghost_release_trigger(if released.count > 0 {
        util_stats::GhostReleaseTrigger::Fired {
            steps_from_restart: (step - t.restart_step).max(0) as u64,
            released: released.count,
        }
    } else {
        util_stats::GhostReleaseTrigger::FiredNothingHeld
    });
}

/// Every node other than `origin` whose planned crash is still held goes
/// at the step after `step`.
fn release_all<H: HashPolicy>(state: &mut State<H>, origin: usize, step: i32) -> Released {
    let mut out = Released::default();
    for n in 0..state.crash_hold_until.len() {
        if n == origin
            || !ghost_release::held(&state.crash_hold_until, &state.send_ledger, n, step)
        {
            continue;
        }
        if ghost_release::release(&mut state.crash_hold_until, n, step) {
            out.count += 1;
            out.mask |= node_bit(n);
        }
    }
    out
}

/// The one crash a firing on `v` releases on the single half: `v`'s own
/// held crash; else the first held crash by index that the absorber ranking
/// would move onto `v`; else, when `v` has no pending pair, the first held
/// crash by index, which is then applied to `v` instead of its planned
/// victim. None when no case holds, and the trigger stays armed. The
/// restarted node's own crash is never the one released.
fn release_one<H: HashPolicy>(
    state: &mut State<H>,
    origin: usize,
    v: NodeId,
    step: i32,
    servers: usize,
) -> Option<Released> {
    let is_held = |state: &State<H>, n: usize| {
        n != origin && ghost_release::held(&state.crash_hold_until, &state.send_ledger, n, step)
    };
    let nodes = 0..state.crash_hold_until.len();
    let via_ranking = |state: &State<H>| {
        nodes.clone().filter(|&n| is_held(state, n)).find(|&n| {
            let planned = NodeId {
                role: v.role,
                index: n,
            };
            matches!(
                absorber_decision(state, planned, servers).choice,
                ghost_absorber::Choice::Retarget { node, .. } if node == v.index
            )
        })
    };
    let (n, case) = if is_held(state, v.index) {
        (v.index, util_stats::SingleRelease::OwnCrash)
    } else if let Some(n) = state.retarget.enabled.then(|| via_ranking(state)).flatten() {
        (n, util_stats::SingleRelease::ViaRanking)
    } else if !state.retarget.has_pending_pair(v.index) {
        let n = nodes.filter(|&n| is_held(state, n)).next()?;
        (n, util_stats::SingleRelease::Forced)
    } else {
        return None;
    };
    if !ghost_release::release(&mut state.crash_hold_until, n, step) {
        return Some(Released::default());
    }
    if case == util_stats::SingleRelease::Forced {
        state.ghost_release.forced_victim = Some((n, v.index));
    }
    util_stats::record_ghost_release_single(case);
    Some(Released {
        count: 1,
        mask: node_bit(n),
    })
}

/// The node a planned crash of `planned` is applied to when a firing named
/// one: the node whose entry fired the trigger, if it is still live and has
/// no pending pair. None hands the crash to the stock path. The naming is
/// consumed either way.
fn forced_victim<H: HashPolicy>(state: &mut State<H>, planned: NodeId) -> Option<NodeId> {
    let (n, v) = state.ghost_release.forced_victim?;
    if n != planned.index {
        return None;
    }
    state.ghost_release.forced_victim = None;
    let victim = NodeId {
        role: planned.role,
        index: v,
    };
    if state.crash_info.currently_crashed.contains(&victim) || state.retarget.has_pending_pair(v) {
        return None;
    }
    util_stats::record_victim_swap(util_stats::VictimSwap::ForcedOntoAbsorber, false);
    Some(victim)
}

/// A planned crash of `planned` is being applied on `victim`: count where
/// it landed against the run's most recent acted ghost entry and, for a
/// crash a firing released, against the node whose entry fired it, the
/// victim's sends in flight, and the released crash applied just before.
fn note_ghost_release_apply<H: HashPolicy>(state: &mut State<H>, planned: NodeId, victim: NodeId) {
    let step = state.crash_info.current_step;
    let in_flight = state
        .send_ledger
        .get(victim.index)
        .map_or(0, |l| l.in_flight);
    let down = &state.crash_info.currently_crashed;
    let gr = &mut state.ghost_release;
    let later = gr.crashes_applied > 0;
    gr.crashes_applied = gr.crashes_applied.saturating_add(1);
    let bit = node_bit(planned.index);
    let fired = gr.released_mask & bit != 0;
    gr.released_mask &= !bit;
    let double = gr.last_released_apply.take().is_some_and(|(v0, s0)| {
        step - s0 <= 8
            && down.contains(&NodeId {
                role: victim.role,
                index: v0,
            })
    });
    if fired {
        gr.last_released_apply = Some((victim.index, step));
    }
    if !util_stats::enabled() {
        return;
    }
    let last_acted_ghost = state.last_acted_ghost_step;
    util_stats::record_ghost_release_apply(
        gr.cell,
        util_stats::GhostReleaseApply {
            later,
            within_3_of_acted_ghost: last_acted_ghost >= 0 && step - last_acted_ghost <= 3,
            fired,
            anchored: gr.anchored,
            retarget: state.retarget.enabled,
            on_ghost_node: gr.ghost_node == Some(victim.index),
            in_flight,
            double_after_release: double,
        },
    );
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

/// A planned crash whose victim is already down waits for that node to come
/// back. Only a retargeted crash can leave a plan in that position, so the
/// check is confined to runs that retarget; it holds the crash in the queue
/// rather than dropping it, so the plan's pair still completes.
fn crashed_victims<H: HashPolicy>(state: &State<H>) -> Option<&OrdSet<NodeId>> {
    state
        .retarget
        .enabled
        .then_some(&state.crash_info.currently_crashed)
}

/// What withholds a runnable from this step: a planned crash in
/// `crash_block_mask` or waiting on a victim that is down, a runnable a
/// reservation matches, or one a FIFO link has not reached yet.
struct Ineligibility<'a> {
    crash_block_mask: u64,
    crashed_victims: Option<&'a OrdSet<NodeId>>,
    reservations: &'a [Reservation],
    link_deliver_seq: &'a imbl::HashMap<crate::simulator::core::values::LinkId, u32>,
}

impl Ineligibility<'_> {
    /// Whether `r` is withheld. A withheld crash on a down victim is counted
    /// each time it is tested.
    #[inline(always)]
    fn rejects<H: HashPolicy>(&self, r: &Runnable<H>) -> bool {
        if self.crash_block_mask != 0
            && let Runnable::Crash { node_id, .. } = r
            && node_id.index < u64::BITS as usize
            && self.crash_block_mask & (1u64 << node_id.index) != 0
        {
            return true;
        }
        if let Some(crashed) = self.crashed_victims
            && let Runnable::Crash { node_id, .. } = r
            && crashed.contains(node_id)
        {
            util_stats::record_victim_crashed_hold();
            return true;
        }
        self.reservations.iter().any(|res| res.matches(r))
            || is_fifo_blocked(r, self.link_deliver_seq)
    }
}

/// With no reservation, no FIFO link and no strict timer gate, the only
/// runnable `Ineligibility` can reject is a planned crash. A runnable carries
/// a link tag only after its link was created, and creating a link enters it
/// in `link_deliver_seq`, which never loses an entry within a run.
fn only_crashes_can_be_ineligible<H: HashPolicy>(
    state: &State<H>,
    reservations: &[Reservation],
    strict_timers: bool,
) -> bool {
    reservations.is_empty() && state.link_deliver_seq.is_empty() && !strict_timers
}

/// The eligible sizes of the network and timer queues, with the eligible
/// size of every local queue written into `local_queue_sizes`: the counts a
/// filter of every queue by `is_ineligible`, and by the strict timer gate for
/// the timer queue, would give. `is_ineligible` must test an `Ineligibility`
/// over `crash_block_mask`, the run's crashed victims, the step's reservations
/// and the state's links, and `only_crashes_ineligible` must be
/// `only_crashes_can_be_ineligible` for the same reservations.
///
/// When only a planned crash can be rejected, a queue is filtered only where
/// such a crash can sit rejected, and every other queue's size is its length.
/// A planned crash is queued only on its own node's local queue and counted
/// there by `crash_pending`, so a local queue is filtered only when its node
/// has a pending crash that is in `crash_block_mask` or whose node is down on
/// a retargeting run, or when the node has no ledger. Filtering those queues
/// with the same predicate keeps every count it makes of a withheld crash.
#[inline(always)]
fn eligible_counts<H: HashPolicy>(
    state: &State<H>,
    only_crashes_ineligible: bool,
    strict_timers: bool,
    crash_block_mask: u64,
    is_ineligible: &impl Fn(&Runnable<H>) -> bool,
    local_queue_sizes: &mut Vec<usize>,
) -> (usize, usize) {
    local_queue_sizes.clear();
    if !only_crashes_ineligible {
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
        local_queue_sizes.extend(
            state
                .local_queues
                .iter()
                .map(|q| q.iter().filter(|r| !is_ineligible(r)).count()),
        );
        let network_queue_size = state.network_queue.iter().filter(|r| !is_ineligible(r)).count();
        if util_stats::enabled() {
            let elements = state.local_queues.iter().map(Vec::len).sum::<usize>()
                + state.network_queue.len()
                + state.timer_queue.len();
            util_stats::record_eligibility_pass(true, true, elements as u64);
        }
        return (network_queue_size, timer_queue_size);
    }

    let mut walked = false;
    let mut walked_elements = 0usize;
    local_queue_sizes.extend(state.local_queues.iter().enumerate().map(|(n, q)| {
        let filtered = match state.send_ledger.get(n) {
            None => true,
            Some(ledger) => {
                ledger.crash_pending > 0
                    && ((n < u64::BITS as usize && crash_block_mask & (1u64 << n) != 0)
                        || (state.retarget.enabled
                            && state
                                .crash_info
                                .currently_crashed
                                .iter()
                                .any(|down| down.index == n)))
            }
        };
        if filtered {
            walked = true;
            walked_elements += q.len();
            q.iter().filter(|r| !is_ineligible(r)).count()
        } else {
            q.len()
        }
    }));
    if util_stats::enabled() {
        util_stats::record_eligibility_pass(walked, false, walked_elements as u64);
    }
    (state.network_queue.len(), state.timer_queue.len())
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
    local_queue_sizes: &mut Vec<usize>,
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

    let crash_block_mask = crash_defer_mask | crash_hold_mask;
    let only_crashes_ineligible =
        only_crashes_can_be_ineligible(state, reservations, strict_timers);
    let ineligibility = Ineligibility {
        crash_block_mask,
        crashed_victims: crashed_victims(state),
        reservations,
        link_deliver_seq: &state.link_deliver_seq,
    };
    let is_ineligible = |r: &Runnable<H>| ineligibility.rejects(r);

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
    let (network_queue_size, timer_queue_size) = eligible_counts(
        state,
        only_crashes_ineligible,
        strict_timers,
        crash_block_mask,
        &is_ineligible,
        local_queue_sizes,
    );
    let info = QueueInfo {
        local_queue_sizes,
        network_queue_size,
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
            // the first eligible timer in queue order. The multiplier read
            // draws nothing, so reading it only where the roll compares it
            // leaves every draw in place.
            let multiplier = || {
                let head_timer_node = state.timer_queue.iter().find_map(|r| {
                    if !only_crashes_ineligible && is_ineligible(r) {
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
            };
            let picked = if timer_ctx_mode == timer_context::RunMode::Steered
                && info.timer_queue_size > 0
            {
                let (picked, bias_use) =
                    select_with_timer_context(selector, &info, multiplier, rng);
                match bias_use {
                    TimerBiasUse::Applied(m) => util_stats::record_timer_context_bias(m),
                    TimerBiasUse::Excluded => util_stats::record_timer_context_excluded(),
                    TimerBiasUse::Unused => {}
                }
                picked
            } else {
                selector.select(&info, rng)
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
            let admitted = info.local_queue_sizes.get(node_idx).copied().unwrap_or(0);
            let eligible = EligibleList::new(queue.len(), admitted, |&i| !is_ineligible(&queue[i]));
            let eligible = eligible.as_slice();
            if eligible.is_empty() {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
            record_eligible_list(queue.len(), admitted);
            let (idx, mask) = select_within_queue::<H, F>(
                queue,
                eligible,
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
            let admitted = info.network_queue_size;
            let eligible = EligibleList::new(queue.len(), admitted, |&i| !is_ineligible(&queue[i]));
            let eligible = eligible.as_slice();
            if eligible.is_empty() {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
            record_eligible_list(queue.len(), admitted);
            let (drawn, drawn_mask) = select_within_queue::<H, F>(
                queue,
                eligible,
                feedback,
                snapshot,
                state,
                terms,
                within_queue,
                rng,
            );
            let idx = fresh_first_dispatch(state, eligible, drawn);
            let idx = pair_order_dispatch(state, eligible, idx);
            observe_rush_dispatch(state, eligible, idx);
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
            let admitted = info.timer_queue_size;
            let eligible = if strict_timers {
                EligibleList::new(queue.len(), admitted, |&i| {
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
            } else {
                EligibleList::new(queue.len(), admitted, |&i| !is_ineligible(&queue[i]))
            };
            let eligible = eligible.as_slice();
            if eligible.is_empty() {
                record_unscheduled(&audit);
                return Ok(ScheduleResult::None);
            }
            record_eligible_list(queue.len(), admitted);
            let (idx, mask) = select_within_queue::<H, F>(
                queue,
                eligible,
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
            let servers = topology.num_servers.max(0) as usize;
            let victim = forced_victim(state, node_id)
                .unwrap_or_else(|| retarget_crash(state, node_id, servers));
            note_ghost_release_apply(state, node_id, victim);
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
            crash_node(state, program, victim);
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
            activate_partition(state, program, partition_type.clone());
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
            let token_before = state.node_state_token(timer.node);

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
                    acted: state.node_state_token(timer.node) != token_before,
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
                        r.reset(program);
                        state.crash_info.queued_messages.push_back((dest_node, r));
                    }
                return Ok(ScheduleResult::None);
            }

            if state.partition_info.is_blocked(src_node, dest_node) {
                match other {
                    Runnable::Record(r) => {
                        let mut r = r;
                        r.reset(program);
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
                    let record_sent_at = r.origin_incarnation;
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
                    // Read at every entry, not only on probes: the compare
                    // after the segment is the progress mark the stall clock
                    // takes, and the token every probe compares against.
                    let token_before = state.node_state_token(record_dest);
                    // Whether a ghost reached a destination that had already
                    // heard from the sender's current incarnation, read off
                    // the per-destination table before this entry is added.
                    // The table is kept on every run: the overtaken-ghost
                    // reward reads it.
                    let mut overtaken_at_restarted = false;
                    if message_entry {
                        let current = state.incarnation(record_origin);
                        if r.origin_incarnation != current {
                            let overtaken = state.fresh_first.heard_from(
                                record_dest.index,
                                record_origin.index,
                                current,
                            );
                            overtaken_at_restarted = state.overtaken_ghost_at_restarted(
                                record_origin,
                                r.origin_incarnation,
                                record_dest,
                            );
                            util_stats::record_fresh_first_ghost_entry(
                                state.fresh_first.enabled,
                                overtaken,
                                state.incarnation(record_dest) > 0,
                            );
                        }
                        state.note_entry_at_absorber(
                            record_origin,
                            r.origin_incarnation,
                            record_dest,
                            entry_step,
                        );
                        let servers = topology.num_servers.max(0) as usize;
                        if record_dest.index < servers
                            && state.client_anchor.caused_post_fault(r.causal_operation_id)
                        {
                            state.note_post_fault_request_entry(record_dest.index, entry_step, servers);
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
                    .then_some(token_before);
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
                        (bias, state.entries_since_restart(record_dest.index))
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
                        (r.pc, key, inflight, cell)
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
                    let acted = state.node_state_token(record_dest) != token_before;
                    if message_entry {
                        note_ghost_release_entry(
                            state,
                            record_origin,
                            record_sent_at,
                            record_dest,
                            entry_step,
                            acted,
                            topology.num_servers.max(0) as usize,
                        );
                    }
                    if let Some(before) = ghost {
                        state.note_ghost_delivery(record_dest.index, entry_step, before);
                        if acted {
                            state.last_acted_ghost_step = entry_step;
                        }
                        // A dead-incarnation record is a ghost, so the
                        // token taken for the mark serves the reward too.
                        if overtaken_at_restarted && acted {
                            state.overtaken_ghost_acted = true;
                        }
                    }
                    if let Some((bias, distance)) = probe {
                        util_stats::record_delivery(bias, acted, distance);
                        util_stats::record_term_acted(chosen_mask, acted);
                    }
                    if let Some((pc, key, inflight, cell)) = timer_probe {
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
                            acted,
                            timer_entry,
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
/// queue and stays eligible. A destination that has never restarted in the
/// run keeps the drawn ghost. No random draw is taken here, so a treated
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
        if state.incarnation(dest) == 0 {
            util_stats::record_fresh_first_skipped_never_restarted_dest();
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

fn crash_node<H: HashPolicy>(state: &mut State<H>, program: &Program, node_id: NodeId) {
    if state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is already crashed", node_id);
        return;
    }
    state.crash_info.currently_crashed.insert(node_id);
    state.note_handler_entry(node_id.index, HandlerTrigger::None);
    state.note_crash_of_acted_absorber(node_id.index);
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
                record.reset(program);
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
                    r.reset(program);
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
    let restart_step = state.crash_info.current_step;
    if let Some(l) = state.send_ledger.get_mut(node_id.index) {
        l.last_restart_step = restart_step;
    }
    arm_release_trigger(state, node_id.index, restart_step);
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

    let mut env = build_frame::<H>(init_fn, &[]);

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

    let initial_args: EcoVec<Value<H>> = actuals.into_iter().collect();
    let env = build_frame(recover_fn, &initial_args);

    let record = Record {
        pc: recover_fn.entry,
        node: node_id,
        origin_node: node_id,
        continuation: Continuation::Recover,
        entry_pc: recover_fn.entry,
        initial_args,
        entry_func: recover_fn.name,
        env,
        priority: policy.sample(rng, RunnableCategory::Record),
        causal_operation_id: None,
        trace_id: None,
        trace_payload: None,
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

    /// A state with `n` nodes, a queued crash on every node in `crashes`,
    /// the given holds, and the ghost release in `cell` with `bound`.
    fn release_state(
        n: usize,
        crashes: &[usize],
        holds: Vec<i32>,
        cell: util_stats::GhostReleaseCell,
        bound: Option<i32>,
    ) -> State<NoHashing> {
        let mut state = State::<NoHashing>::new(&[(ROLE, n)], 1);
        for &c in crashes {
            state.push_runnable(Runnable::Crash {
                node_id: node(c),
                priority: 0.5,
            });
        }
        state.crash_hold_until = holds;
        state.ghost_release.cell = cell;
        state.ghost_release.bound = bound;
        state.crash_info.current_step = 100;
        state
    }

    /// A restart that finds another node's crash still held arms the
    /// trigger on the releasing cells and moves no hold; a restart that
    /// finds none, a run outside the releasing cells, and a run below the
    /// learner's floor arm nothing. A restart while armed replaces the
    /// trigger.
    #[test]
    fn a_restart_with_another_node_s_crash_held_arms_the_trigger_and_moves_nothing() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place.ghost_release;
        let holds = vec![900, 500, 431, 110];
        let armed = ghost_release::Trigger {
            origin: 0,
            restart_step: 100,
            expires: 120,
        };
        for cell in [
            util_stats::GhostReleaseCell::ReleaseAll,
            util_stats::GhostReleaseCell::Single,
        ] {
            let mut state = release_state(4, &[0, 1, 2, 3], holds.clone(), cell, Some(20));
            arm_release_trigger(&mut state, 0, 100);
            assert_eq!(state.ghost_release.trigger, Some(armed), "{cell:?}");
            assert_eq!(state.crash_hold_until, holds, "{cell:?}: arming moved a hold");
            assert_eq!(state.ghost_release.released_mask, 0);
        }
        let mut control = release_state(
            4,
            &[0, 1, 2, 3],
            holds.clone(),
            util_stats::GhostReleaseCell::Untreated,
            Some(20),
        );
        arm_release_trigger(&mut control, 0, 100);
        assert_eq!(control.ghost_release.trigger, None, "the untreated cell armed");
        let mut unplaced = release_state(
            4,
            &[0, 1, 2, 3],
            holds.clone(),
            util_stats::GhostReleaseCell::Unplaced,
            Some(20),
        );
        arm_release_trigger(&mut unplaced, 0, 100);
        assert_eq!(unplaced.ghost_release.trigger, None, "an unplaced run armed");
        let mut unlearned = release_state(
            4,
            &[0, 1, 2, 3],
            holds.clone(),
            util_stats::GhostReleaseCell::ReleaseAll,
            None,
        );
        arm_release_trigger(&mut unlearned, 0, 100);
        assert_eq!(unlearned.ghost_release.trigger, None, "below the floor armed");
        assert_eq!(unlearned.crash_hold_until, holds);

        // Only the restarted node's own crash is held, or every other hold
        // has passed: nothing to release, nothing armed.
        let mut own = release_state(
            3,
            &[0, 1],
            vec![900, 90, 0],
            util_stats::GhostReleaseCell::ReleaseAll,
            Some(20),
        );
        arm_release_trigger(&mut own, 0, 100);
        assert_eq!(own.ghost_release.trigger, None, "nothing held, something armed");
        // A held crash whose target lies inside the bound still arms: the
        // arming asks only whether a hold is ahead.
        let mut near = release_state(
            3,
            &[1],
            vec![0, 105, 0],
            util_stats::GhostReleaseCell::ReleaseAll,
            Some(20),
        );
        arm_release_trigger(&mut near, 0, 100);
        assert_eq!(near.ghost_release.trigger, Some(armed));
        assert_eq!(near.crash_hold_until, vec![0, 105, 0]);

        // A restart while armed replaces the trigger.
        let mut twice = release_state(
            3,
            &[0, 1, 2],
            vec![900, 500, 431],
            util_stats::GhostReleaseCell::Single,
            Some(20),
        );
        arm_release_trigger(&mut twice, 0, 100);
        arm_release_trigger(&mut twice, 2, 105);
        assert_eq!(
            twice.ghost_release.trigger,
            Some(ghost_release::Trigger {
                origin: 2,
                restart_step: 105,
                expires: 125
            })
        );
        assert_eq!(twice.crash_hold_until, vec![900, 500, 431]);

        let after = util_stats::snapshot().crash_place.ghost_release;
        util_stats::set_enabled(false);
        assert_eq!(after.armed - before.armed, 5);
        assert_eq!(after.superseded - before.superseded, 1);
        assert_eq!(after.restarts_with_held_crash - before.restarts_with_held_crash, 6);
        assert_eq!(after.fired, before.fired);
        assert_eq!(after.expired, before.expired);
    }

    /// Entries that are not from the restarted node's dead incarnation,
    /// that land on the restarted node, that leave the receiver's state
    /// unchanged, or that land at the restart step do not fire the trigger.
    /// The first entry that does moves every other held crash to the next
    /// step, leaves the restarted node's own hold and a hold already passed
    /// alone, and disarms; a later entry moves nothing.
    #[test]
    fn the_first_acted_dead_incarnation_entry_at_a_live_peer_releases_every_other_held_crash() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place.ghost_release;
        let mut state = release_state(
            5,
            &[0, 1, 2, 3, 4],
            vec![900, 500, 431, 90, 700],
            util_stats::GhostReleaseCell::ReleaseAll,
            Some(20),
        );
        state.incarnations = vec![1, 1, 0, 0, 0];
        state.send_ledger[0].last_restart_step = 100;
        state.send_ledger[1].last_restart_step = 50;
        arm_release_trigger(&mut state, 0, 100);
        let armed = state.ghost_release.trigger;
        assert!(armed.is_some());
        // Another node's dead incarnation entering the restarted node.
        note_ghost_release_entry(&mut state, node(1), 0, node(0), 103, true, 5);
        // The restarted node's live incarnation entering a peer.
        note_ghost_release_entry(&mut state, node(0), 1, node(2), 104, true, 5);
        // The dead incarnation entering a peer without writing its state.
        note_ghost_release_entry(&mut state, node(0), 0, node(2), 105, false, 5);
        // The dead incarnation entering a peer at the restart step itself.
        note_ghost_release_entry(&mut state, node(0), 0, node(2), 100, true, 5);
        // The dead incarnation entering a peer that is down.
        state.crash_info.currently_crashed.insert(node(4));
        note_ghost_release_entry(&mut state, node(0), 0, node(4), 105, true, 5);
        state.crash_info.currently_crashed.remove(&node(4));
        assert_eq!(state.ghost_release.trigger, armed, "a non-firing entry disarmed");
        assert_eq!(state.crash_hold_until, vec![900, 500, 431, 90, 700]);

        note_ghost_release_entry(&mut state, node(0), 0, node(2), 106, true, 5);
        assert_eq!(
            state.crash_hold_until,
            vec![900, 107, 107, 90, 107],
            "the firing releases every other held crash at the next step"
        );
        assert_eq!(state.ghost_release.trigger, None);
        assert_eq!(state.ghost_release.released_mask, 0b10110);
        assert_eq!(state.ghost_release.ghost_node, Some(2));
        note_ghost_release_entry(&mut state, node(0), 0, node(1), 108, true, 5);
        assert_eq!(
            state.crash_hold_until,
            vec![900, 107, 107, 90, 107],
            "a second entry moves nothing"
        );
        assert_eq!(state.ghost_release.ghost_node, Some(2));

        let after = util_stats::snapshot().crash_place.ghost_release;
        util_stats::set_enabled(false);
        assert_eq!(after.fired - before.fired, 1);
        assert_eq!(after.fired_nothing_held, before.fired_nothing_held);
        assert_eq!(after.released_crashes - before.released_crashes, 3);
        assert_eq!(after.steps_from_restart_sum - before.steps_from_restart_sum, 6);
        assert_eq!(after.single.releases, before.single.releases);
    }

    /// A firing that finds every other hold already at or before the next
    /// step moves nothing, disarms, and counts apart from the firings that
    /// released something.
    #[test]
    fn a_firing_that_shortens_nothing_counts_under_fired_nothing_held() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place.ghost_release;
        let mut state = release_state(
            3,
            &[0, 1, 2],
            vec![900, 500, 431],
            util_stats::GhostReleaseCell::ReleaseAll,
            Some(20),
        );
        state.incarnations = vec![1, 0, 0];
        state.send_ledger[0].last_restart_step = 100;
        arm_release_trigger(&mut state, 0, 100);
        // Every other hold is at the next step already, or has passed.
        state.crash_hold_until = vec![900, 107, 101];
        note_ghost_release_entry(&mut state, node(0), 0, node(2), 106, true, 3);
        assert_eq!(state.crash_hold_until, vec![900, 107, 101]);
        assert_eq!(state.ghost_release.trigger, None, "the firing did not disarm");
        assert_eq!(state.ghost_release.released_mask, 0);
        assert_eq!(state.ghost_release.ghost_node, Some(2));
        let after = util_stats::snapshot().crash_place.ghost_release;
        util_stats::set_enabled(false);
        assert_eq!(after.fired_nothing_held - before.fired_nothing_held, 1);
        assert_eq!(after.fired, before.fired);
        assert_eq!(after.released_crashes, before.released_crashes);
    }

    /// With no firing by the bound the trigger expires at the top of the
    /// hold mask and every crash keeps its drawn target; an entry after
    /// the expiry moves nothing.
    #[test]
    fn expiry_at_the_bound_leaves_every_target_untouched() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place.ghost_release;
        let mut idle = release_state(
            3,
            &[1, 2],
            vec![0, 500, 431],
            util_stats::GhostReleaseCell::ReleaseAll,
            Some(20),
        );
        idle.incarnations = vec![1, 0, 0];
        arm_release_trigger(&mut idle, 0, 100);
        let armed = idle.ghost_release.trigger;
        assert!(armed.is_some());
        let mut rng = StdRng::seed_from_u64(1);
        for step in 101..120 {
            idle.crash_info.current_step = step;
            assert_eq!(crash_hold_mask(&mut idle, 3, &mut rng), 0b110, "step {step}");
            assert_eq!(idle.ghost_release.trigger, armed, "step {step}: expired early");
        }
        idle.crash_info.current_step = 120;
        assert_eq!(crash_hold_mask(&mut idle, 3, &mut rng), 0b110);
        assert_eq!(idle.ghost_release.trigger, None, "the bound expires the trigger");
        assert_eq!(idle.crash_hold_until, vec![0, 500, 431], "expiry moved a hold");
        note_ghost_release_entry(&mut idle, node(0), 0, node(2), 121, true, 3);
        assert_eq!(idle.crash_hold_until, vec![0, 500, 431]);
        let after = util_stats::snapshot().crash_place.ghost_release;
        util_stats::set_enabled(false);
        assert_eq!(after.expired - before.expired, 1);
        assert_eq!(after.fired, before.fired);
    }

    /// A releasing twin and an untreated twin take the same restart and the
    /// same firing entry: only the releasing twin's holds move, the release
    /// takes no random value, and the twins' streams stay in step over the
    /// offers that follow.
    #[test]
    fn the_untreated_twin_never_releases_and_keeps_its_streams() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let holds = vec![900, 500, 431, 110];
        let build = |cell| {
            let mut state = release_state(4, &[0, 1, 2, 3], holds.clone(), cell, Some(20));
            state.incarnations = vec![1, 0, 0, 0];
            state.send_ledger[0].last_restart_step = 100;
            state
        };
        let mut treated = build(util_stats::GhostReleaseCell::ReleaseAll);
        let mut control = build(util_stats::GhostReleaseCell::Untreated);
        for s in [&mut treated, &mut control] {
            arm_release_trigger(s, 0, 100);
            note_ghost_release_entry(s, node(0), 0, node(2), 106, true, 4);
        }
        assert_eq!(treated.crash_hold_until, vec![900, 107, 107, 107]);
        assert_eq!(control.crash_hold_until, holds, "the untreated twin");
        assert_eq!(control.ghost_release.trigger, None);
        assert_eq!(control.ghost_release.released_mask, 0);

        let mut rng_t = StdRng::seed_from_u64(3);
        let mut rng_c = StdRng::seed_from_u64(3);
        for step in 100..=200 {
            treated.crash_info.current_step = step;
            control.crash_info.current_step = step;
            let held_t = crash_hold_mask(&mut treated, 4, &mut rng_t);
            let held_c = crash_hold_mask(&mut control, 4, &mut rng_c);
            let mut want_c = 1u64;
            for (n, t) in [(1, 500), (2, 431), (3, 110)] {
                if step < t {
                    want_c |= 1 << n;
                }
            }
            assert_eq!(held_c, want_c, "step {step}: the control twin");
            let want_t = if step < 107 { 0b1111 } else { 0b0001 };
            assert_eq!(held_t, want_t, "step {step}: the releasing twin");
        }
        assert_eq!(rng_t.next_u64(), rng_c.next_u64(), "the twins' streams parted");
    }

    /// On an anchored run a released crash's phase slot stays pending until
    /// its new target and draws its arm at the first offer past it, as an
    /// unreleased hold does at its drawn target.
    #[test]
    fn a_released_crash_s_phase_slot_draws_at_the_first_offer_past_its_new_target() {
        let _serial = crate::simulator::config_override::exclusive_session();
        let mut state = release_state(
            2,
            &[1],
            vec![0, 500],
            util_stats::GhostReleaseCell::ReleaseAll,
            Some(20),
        );
        state.crash_phase.arm_node(1, 0);
        // A segment with nothing issued meets no waiting arm's phase.
        state.send_ledger[1].floor = state.send_ledger[1].issued;
        state.incarnations = vec![1, 0];
        arm_release_trigger(&mut state, 0, 100);
        note_ghost_release_entry(&mut state, node(0), 0, node(1), 103, true, 2);
        let target = state.crash_hold_until[1];
        assert_eq!(target, 104);
        let mut rng = StdRng::seed_from_u64(9);
        for step in 100..target {
            state.crash_info.current_step = step;
            assert_eq!(crash_hold_mask(&mut state, 2, &mut rng), 1 << 1, "step {step}");
            assert_eq!(state.crash_phase.arm_of(1), None, "the slot drew before its target");
        }
        state.crash_info.current_step = target;
        let held = crash_hold_mask(&mut state, 2, &mut rng) != 0;
        match state.crash_phase.arm_of(1) {
            Some(util_stats::CrashPhaseArm::Stock) => assert!(!held, "a stock draw held"),
            Some(_) => assert!(held, "a waiting arm released at once on a silent segment"),
            None => panic!("the slot did not draw at the first offer past the new target"),
        }
    }

    /// On the single half a firing on v releases exactly one crash: v's own
    /// held crash when it has one; else the first held crash by index the
    /// absorber ranking would move onto v; else, when v has no pending
    /// pair, the first held crash by index, which is then to be applied to
    /// v; and nothing when none of the three applies, the trigger staying
    /// armed for a later entry.
    #[test]
    fn the_single_half_releases_exactly_the_crash_its_case_names() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place.ghost_release;
        let build = |crashes: &[usize], holds: Vec<i32>| {
            let mut state = release_state(
                4,
                crashes,
                holds,
                util_stats::GhostReleaseCell::Single,
                Some(20),
            );
            state.incarnations = vec![1, 0, 0, 0];
            state.send_ledger[0].last_restart_step = 100;
            arm_release_trigger(&mut state, 0, 100);
            assert!(state.ghost_release.trigger.is_some());
            state
        };
        let fire_on = |state: &mut State<NoHashing>, v: usize| {
            note_ghost_release_entry(state, node(0), 0, node(v), 106, true, 4);
        };

        // (1) v's own held crash goes alone.
        let mut own = build(&[0, 1, 2, 3], vec![900, 500, 431, 300]);
        own.retarget.enabled = true;
        own.send_ledger[1].last_ghost_step = 95;
        own.send_ledger[1].last_ghost_acted = true;
        fire_on(&mut own, 2);
        assert_eq!(own.crash_hold_until, vec![900, 500, 107, 300]);
        assert_eq!(own.ghost_release.trigger, None);
        assert_eq!(own.ghost_release.released_mask, 0b0100);
        assert_eq!(own.ghost_release.forced_victim, None);

        // (2) v has no held crash; the ranking would move the crashes of
        // nodes 1 and 3 onto v, so the first by index goes alone.
        let mut ranked = build(&[0, 1, 3], vec![900, 500, 0, 300]);
        ranked.retarget.enabled = true;
        ranked.send_ledger[2].last_ghost_step = 95;
        ranked.send_ledger[2].last_ghost_acted = true;
        fire_on(&mut ranked, 2);
        assert_eq!(ranked.crash_hold_until, vec![900, 107, 0, 300]);
        assert_eq!(ranked.ghost_release.trigger, None);
        assert_eq!(ranked.ghost_release.released_mask, 0b0010);
        assert_eq!(ranked.ghost_release.forced_victim, None);

        // (2) passes a held crash the ranking keeps where it is: node 1's
        // own mark outranks v, so its crash stays on node 1, and node 3's
        // crash is the first the ranking moves onto v.
        let mut elsewhere = build(&[0, 1, 3], vec![900, 500, 0, 300]);
        elsewhere.retarget.enabled = true;
        elsewhere.send_ledger[1].last_ghost_step = 97;
        elsewhere.send_ledger[1].last_ghost_acted = true;
        elsewhere.send_ledger[2].last_ghost_step = 95;
        elsewhere.send_ledger[2].last_ghost_acted = true;
        fire_on(&mut elsewhere, 2);
        assert_eq!(elsewhere.crash_hold_until, vec![900, 500, 0, 107]);
        assert_eq!(elsewhere.ghost_release.released_mask, 0b1000);

        // (3) no ranking reaches v and v has no pending pair: the first held
        // crash by index goes alone and is to be applied to v. The
        // restarted node's own crash, first by index, is never the one.
        let mut forced = build(&[0, 1, 3], vec![900, 500, 0, 300]);
        fire_on(&mut forced, 2);
        assert_eq!(forced.crash_hold_until, vec![900, 107, 0, 300]);
        assert_eq!(forced.ghost_release.trigger, None);
        assert_eq!(forced.ghost_release.released_mask, 0b0010);
        assert_eq!(forced.ghost_release.forced_victim, Some((1, 2)));
        // The same on a retarget run whose ranking finds no absorber.
        let mut unmarked = build(&[0, 1, 3], vec![900, 500, 0, 300]);
        unmarked.retarget.enabled = true;
        fire_on(&mut unmarked, 2);
        assert_eq!(unmarked.crash_hold_until, vec![900, 107, 0, 300]);
        assert_eq!(unmarked.ghost_release.forced_victim, Some((1, 2)));

        // (4) v has a pending pair and no case applies: nothing moves and
        // the trigger stays armed; an entry at another peer then fires.
        let mut none = build(&[0, 1, 3], vec![900, 500, 0, 300]);
        none.retarget.enabled = true;
        none.retarget.pending_pair_mask = 1 << 2;
        let armed = none.ghost_release.trigger;
        fire_on(&mut none, 2);
        assert_eq!(none.crash_hold_until, vec![900, 500, 0, 300]);
        assert_eq!(none.ghost_release.trigger, armed, "no case, yet disarmed");
        assert_eq!(none.ghost_release.released_mask, 0);
        assert_eq!(none.ghost_release.forced_victim, None);
        note_ghost_release_entry(&mut none, node(0), 0, node(3), 108, true, 4);
        assert_eq!(none.crash_hold_until, vec![900, 500, 0, 109]);
        assert_eq!(none.ghost_release.trigger, None);
        assert_eq!(none.ghost_release.ghost_node, Some(3));

        // A named crash whose hold is at the next step already: the firing
        // disarms, moves nothing, and names no forced victim.
        let mut at_next = build(&[0, 1], vec![900, 500, 0, 0]);
        at_next.crash_hold_until[1] = 107;
        fire_on(&mut at_next, 2);
        assert_eq!(at_next.crash_hold_until, vec![900, 107, 0, 0]);
        assert_eq!(at_next.ghost_release.trigger, None);
        assert_eq!(at_next.ghost_release.forced_victim, None);
        assert_eq!(at_next.ghost_release.released_mask, 0);

        let after = util_stats::snapshot().crash_place.ghost_release;
        util_stats::set_enabled(false);
        assert_eq!(after.single.released_own_crash - before.single.released_own_crash, 2);
        assert_eq!(after.single.released_via_ranking - before.single.released_via_ranking, 2);
        assert_eq!(after.single.released_forced - before.single.released_forced, 2);
        assert_eq!(after.single.releases - before.single.releases, 6);
        assert_eq!(after.single.no_release_case - before.single.no_release_case, 1);
        assert_eq!(after.fired - before.fired, 6);
        assert_eq!(after.fired_nothing_held - before.fired_nothing_held, 1);
        assert_eq!(after.released_crashes - before.released_crashes, 6);
    }

    /// At the apply site a planned crash a firing named for the peer whose
    /// entry fired it lands there when that peer is still live with no
    /// pending pair, and counts as a forced victim swap; otherwise, or for
    /// any other planned crash, the stock path runs. The naming is consumed
    /// by the crash it names.
    #[test]
    fn a_forced_victim_lands_the_named_crash_on_the_ghost_node() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().victim_swap;
        let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        state.ghost_release.forced_victim = Some((1, 2));
        assert_eq!(forced_victim(&mut state, node(0)), None, "another crash took the naming");
        assert_eq!(state.ghost_release.forced_victim, Some((1, 2)));
        assert_eq!(forced_victim(&mut state, node(1)), Some(node(2)));
        assert_eq!(state.ghost_release.forced_victim, None, "the naming was not consumed");
        assert_eq!(forced_victim(&mut state, node(1)), None);

        state.ghost_release.forced_victim = Some((1, 2));
        state.crash_info.currently_crashed.insert(node(2));
        assert_eq!(forced_victim(&mut state, node(1)), None, "a down peer took the crash");
        assert_eq!(state.ghost_release.forced_victim, None);
        state.crash_info.currently_crashed.remove(&node(2));

        state.ghost_release.forced_victim = Some((1, 2));
        state.retarget.enabled = true;
        state.retarget.pending_pair_mask = 1 << 2;
        assert_eq!(forced_victim(&mut state, node(1)), None, "a pending pair took the crash");
        assert_eq!(state.ghost_release.forced_victim, None);

        let after = util_stats::snapshot().victim_swap;
        util_stats::set_enabled(false);
        assert_eq!(after.forced_onto_absorber - before.forced_onto_absorber, 1);
        assert_eq!(after.applied - before.applied, 1);
        assert_eq!(after.acted_absorber - before.acted_absorber, 1);
    }

    /// A ghost lag is sampled at an entry from a dead incarnation of a live
    /// origin at a live destination, on a run that feeds the learner, and
    /// nowhere else.
    #[test]
    fn only_dead_incarnation_entries_between_live_nodes_on_a_feeding_run_are_lag_samples() {
        let _serial = crate::simulator::config_override::exclusive_session();
        ghost_release::reset();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place.ghost_release.lag_samples;
        let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        state.incarnations[0] = 1;
        state.send_ledger[0].last_restart_step = 40;
        state.ghost_release.feeds_learner = true;
        state.ghost_release.scope = 6000;
        for _ in 0..199 {
            note_ghost_release_entry(&mut state, node(0), 0, node(1), 52, true, 3);
        }
        note_ghost_release_entry(&mut state, node(0), 1, node(1), 52, true, 3);
        state.crash_info.currently_crashed.insert(node(1));
        note_ghost_release_entry(&mut state, node(0), 0, node(1), 52, true, 3);
        state.crash_info.currently_crashed.remove(&node(1));
        state.crash_info.currently_crashed.insert(node(0));
        note_ghost_release_entry(&mut state, node(0), 0, node(2), 52, true, 3);
        state.crash_info.currently_crashed.remove(&node(0));
        state.ghost_release.feeds_learner = false;
        note_ghost_release_entry(&mut state, node(0), 0, node(1), 52, true, 3);
        state.ghost_release.feeds_learner = true;
        assert_eq!(ghost_release::bound(6000), None, "a non-sample reached the learner");
        note_ghost_release_entry(&mut state, node(0), 0, node(2), 52, false, 3);
        assert_eq!(ghost_release::bound(6000), Some(12), "an inert entry is still a lag");
        let after = util_stats::snapshot().crash_place.ghost_release.lag_samples;
        util_stats::set_enabled(false);
        assert_eq!(after - before, 200);
        ghost_release::reset();
    }

    /// A crash apply counts under its run's cell, marks a later crash that
    /// lands within three steps of the run's most recent acted ghost
    /// entry, and counts a crash a firing released by where it landed: on
    /// the node whose entry fired it, split by the retarget arm; within
    /// three steps, split by the phase arm; by the victim's sends in
    /// flight; and as a double crash when it follows a released crash
    /// whose victim is still down. Enabling the counters again clears the
    /// block.
    #[test]
    fn a_crash_apply_counts_its_landing() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_place.ghost_release;
        let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        state.ghost_release.cell = util_stats::GhostReleaseCell::ReleaseAll;
        state.ghost_release.anchored = false;
        state.retarget.enabled = false;
        state.last_acted_ghost_step = 108;
        // The first crash: never a later one.
        state.crash_info.current_step = 110;
        note_ghost_release_apply(&mut state, node(1), node(1));
        // A later crash within three steps of the acted ghost.
        state.crash_info.current_step = 111;
        note_ghost_release_apply(&mut state, node(2), node(2));
        // A later crash past three steps.
        state.crash_info.current_step = 112;
        note_ghost_release_apply(&mut state, node(1), node(1));
        assert_eq!(state.ghost_release.crashes_applied, 3);

        // Two released crashes: node 1's lands on the ghost node with two
        // sends in flight, one step past the acted ghost, on the stock
        // retarget arm; node 2's lands elsewhere, within eight steps of the
        // first while node 2 is still down, on the retarget arm of an
        // anchored run, and four steps past the acted ghost.
        state.ghost_release.released_mask = 0b110;
        state.ghost_release.ghost_node = Some(2);
        state.send_ledger[2].in_flight = 2;
        state.last_acted_ghost_step = 119;
        state.crash_info.current_step = 120;
        note_ghost_release_apply(&mut state, node(1), node(2));
        assert_eq!(state.ghost_release.released_mask, 0b100);
        assert_eq!(state.ghost_release.last_released_apply, Some((2, 120)));
        state.crash_info.currently_crashed.insert(node(2));
        state.retarget.enabled = true;
        state.ghost_release.anchored = true;
        state.crash_info.current_step = 123;
        note_ghost_release_apply(&mut state, node(2), node(1));
        assert_eq!(state.ghost_release.released_mask, 0);
        assert_eq!(state.ghost_release.last_released_apply, Some((1, 123)));
        // A crash more than eight steps after the released one is no
        // double.
        state.crash_info.currently_crashed.insert(node(1));
        state.crash_info.current_step = 140;
        note_ghost_release_apply(&mut state, node(0), node(0));
        assert_eq!(state.ghost_release.last_released_apply, None);

        let mut other = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        other.ghost_release.cell = util_stats::GhostReleaseCell::Untreated;
        note_ghost_release_apply(&mut other, node(0), node(0));
        other.ghost_release.cell = util_stats::GhostReleaseCell::Single;
        note_ghost_release_apply(&mut other, node(0), node(0));
        other.ghost_release.cell = util_stats::GhostReleaseCell::Unplaced;
        note_ghost_release_apply(&mut other, node(0), node(0));

        let after = util_stats::snapshot().crash_place.ghost_release;
        let d = |f: fn(&util_stats::GhostReleaseCellStats) -> u64| {
            (
                f(&after.cells.untreated) - f(&before.cells.untreated),
                f(&after.cells.release_all) - f(&before.cells.release_all),
                f(&after.cells.single) - f(&before.cells.single),
                f(&after.cells.treated) - f(&before.cells.treated),
            )
        };
        assert_eq!(d(|c| c.crashes_applied), (1, 6, 1, 7));
        assert_eq!(d(|c| c.later_crashes_applied), (0, 5, 1, 6));
        assert_eq!(d(|c| c.applied_within_3_of_acted_ghost), (0, 2, 0, 2));
        assert_eq!(d(|c| c.fired_crashes_applied), (0, 2, 0, 2));
        assert_eq!(d(|c| c.fired_crashes_applied_within_3), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_on_ghost_node), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_applied_stock), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_on_ghost_node_stock), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_applied_retarget), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_on_ghost_node_retarget), (0, 0, 0, 0));
        assert_eq!(d(|c| c.fired_crashes_applied_unanchored), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_applied_within_3_unanchored), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_applied_anchored), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_crashes_applied_within_3_anchored), (0, 0, 0, 0));
        assert_eq!(d(|c| c.fired_inflight_bucket_0), (0, 1, 0, 1));
        assert_eq!(d(|c| c.fired_inflight_bucket_2), (0, 1, 0, 1));
        assert_eq!(d(|c| c.double_crash_after_release), (0, 1, 0, 1));
        let s = (&before.single.cells, &after.single.cells);
        assert_eq!(s.1.release_all.crashes_applied - s.0.release_all.crashes_applied, 6);
        assert_eq!(
            s.1.release_all.double_crash_after_release - s.0.release_all.double_crash_after_release,
            1
        );
        assert_eq!(s.1.single.crashes_applied - s.0.single.crashes_applied, 1);

        util_stats::set_enabled(true);
        let zero = util_stats::snapshot().crash_place.ghost_release;
        util_stats::set_enabled(false);
        assert_eq!(zero.armed, 0);
        assert_eq!(zero.fired, 0);
        assert_eq!(zero.released_crashes, 0);
        assert_eq!(zero.cells.treated.crashes_applied, 0);
        assert_eq!(zero.cells.untreated.crashes_applied, 0);
        assert_eq!(zero.cells.release_all.fired_crashes_applied_stock, 0);
        assert_eq!(zero.cells.single.double_crash_after_release, 0);
        assert_eq!(zero.single.releases, 0);
        assert_eq!(zero.single.no_release_case, 0);
        assert_eq!(zero.lag_samples, 0);
        assert_eq!((zero.lag_p50, zero.lag_p75, zero.lag_p90, zero.scopes_engaged), (0, 0, 0, 0));
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
            local_queue_sizes: &[1],
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
            local_queue_sizes: &[1],
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
        crash_node(&mut state, &Program::default(), node(1));
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

        crash_node(&mut state, &Program::default(), NodeId { role, index: 1 });
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
            initial_args: EcoVec::new(),
            entry_func: crate::analysis::resolver::NameId(0),
            env,
            priority,
            causal_operation_id: None,
            trace_id: None,
            trace_payload: None,
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

    /// Three nodes; node 0 has restarted once, so its incarnation is 1, and
    /// so have the two destinations, which the preference requires. The
    /// queue holds, in order: a ghost 0->1, a fresh 0->1 at priority 0.3, a
    /// channel send 0->1, a fresh 0->2, a fresh 0->1 at priority 0.8, and a
    /// second fresh 0->1 at priority 0.8.
    fn contested_state() -> (State<NoHashing>, Vec<usize>) {
        let mut state = State::<NoHashing>::new(&[(ROLE, 3)], 1);
        for dest in [1, 2] {
            state.incarnations[dest] = 1;
            state.note_incarnation_bump(dest);
        }
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

    /// The lent list equals the filtered list for every queue length,
    /// including lengths past the lendable range, and is lent exactly when
    /// the queue was admitted in full within that range.
    #[test]
    fn eligible_list_matches_the_filtered_queue_at_every_length() {
        let mut rng = StdRng::seed_from_u64(17);
        let range = IDENTITY_INDICES.len();
        let mut lent = 0;
        let mut long = 0;
        for len in 0..=(3 * range) {
            for density in [0u64, 1, 8, 64] {
                let rejected: Vec<bool> = (0..len)
                    .map(|_| density != 0 && rng.next_u64() % density == 0)
                    .collect();
                let keep = |&i: &usize| !rejected[i];
                let reference: Vec<usize> = (0..len).filter(keep).collect();
                let admitted = rejected.iter().filter(|r| !**r).count();
                let list = EligibleList::new(len, admitted, keep);
                assert_eq!(list.as_slice(), &reference[..], "len {len} density {density}");
                let full = admitted == len;
                match &list {
                    EligibleList::Known(n) => {
                        assert!(full && len <= range, "len {len} lent while not admitted in full");
                        assert_eq!(*n, len);
                        lent += 1;
                    }
                    EligibleList::Built(built) => {
                        assert!(!full || len > range, "len {len} built while lendable");
                        assert!(built.capacity() >= admitted);
                        if full {
                            long += 1;
                        }
                    }
                }
            }
        }
        assert!(lent > range && long > range, "lent {lent}, long {long}");
    }

    fn eligibility_record(
        state: &mut State<NoHashing>,
        origin: usize,
        dest: usize,
        entry_pc: usize,
        link_seq: Option<(crate::simulator::core::values::LinkId, u32)>,
    ) -> Runnable<NoHashing> {
        Runnable::Record(Record {
            pc: 0,
            node: node(dest),
            origin_node: node(origin),
            continuation: Continuation::Recover,
            entry_pc,
            initial_args: EcoVec::new(),
            entry_func: crate::analysis::resolver::NameId(0),
            env: Env::<NoHashing>::with_slots(1),
            priority: 0.5,
            causal_operation_id: None,
            trace_id: None,
            trace_payload: None,
            link_seq,
            origin_incarnation: 0,
            bias: DeliveryBias::NONE,
            timer_entry: None,
            send_ordinal: state.next_send_ordinal(node(origin)),
            receiver_token_at_send: state.node_state_token(node(dest)),
        })
    }

    fn below(rng: &mut StdRng, n: u64) -> u64 {
        rng.next_u64() % n
    }

    /// Four nodes with planned crashes, recovers and local records on their
    /// own queues, remote records, and labelled and unlabelled timers, under
    /// a drawn mix of FIFO links and tags, a reservation, the strict timer
    /// gate and its allowed labels, a retargeting run with nodes down, and a
    /// crash block mask. Returns the state, the reservations, whether timers
    /// are strict, and the mask.
    fn eligibility_case(rng: &mut StdRng) -> (State<NoHashing>, Vec<Reservation>, bool, u64) {
        use crate::simulator::core::state::Timer;
        use crate::simulator::core::values::{ChannelId, LinkId};
        const NODES: usize = 4;
        let mut state = State::<NoHashing>::new(&[(ROLE, NODES)], 1);
        let fifo = below(rng, 3) == 0;
        if fifo {
            state.link_deliver_seq.insert(LinkId(0), 1);
        }
        let tag = |rng: &mut StdRng| {
            (fifo && below(rng, 2) == 0).then(|| (LinkId(0), below(rng, 3) as u32))
        };
        for n in 0..NODES {
            for _ in 0..below(rng, 4) {
                let r = match below(rng, 4) {
                    0 => Runnable::Crash {
                        node_id: node(n),
                        priority: 0.5,
                    },
                    1 => Runnable::Recover {
                        node_id: node(n),
                        priority: 0.5,
                    },
                    _ => {
                        let entry = [0, 7][below(rng, 2) as usize];
                        let link = tag(rng);
                        eligibility_record(&mut state, n, n, entry, link)
                    }
                };
                state.push_runnable(r);
            }
        }
        for _ in 0..below(rng, 6) {
            let origin = below(rng, NODES as u64) as usize;
            let dest = (origin + 1 + below(rng, NODES as u64 - 1) as usize) % NODES;
            let entry = [0, 7][below(rng, 2) as usize];
            let link = tag(rng);
            let r = eligibility_record(&mut state, origin, dest, entry, link);
            state.push_runnable(r);
        }
        for _ in 0..below(rng, 5) {
            let n = below(rng, NODES as u64) as usize;
            let label = [None, Some("a".to_string()), Some("b".to_string())]
                [below(rng, 3) as usize]
                .clone();
            state.push_runnable(Runnable::Timer(Timer {
                pc: 0,
                node: node(n),
                channel: ChannelId { node: node(n), id: 0 },
                priority: 0.5,
                label,
            }));
        }
        let strict_timers = below(rng, 3) == 0;
        for n in 0..NODES {
            if below(rng, 2) == 0 {
                state.allowed_timers.insert((n, "a".to_string()));
            }
        }
        let reservations = if below(rng, 3) == 0 {
            vec![Reservation {
                entry_pc: 7,
                from: None,
                to: (below(rng, 2) == 0).then(|| below(rng, NODES as u64) as usize),
            }]
        } else {
            Vec::new()
        };
        state.retarget.enabled = below(rng, 2) == 0;
        for n in 0..NODES {
            if below(rng, 3) == 0 {
                state.crash_info.currently_crashed.insert(node(n));
            }
        }
        let mask = rng.next_u64() & 0xf;
        (state, reservations, strict_timers, mask)
    }

    /// Every queue filtered runnable by runnable, the eligible sizes
    /// `eligible_counts` must reproduce.
    fn filtered_counts(
        state: &State<NoHashing>,
        strict_timers: bool,
        is_ineligible: &impl Fn(&Runnable<NoHashing>) -> bool,
    ) -> (Vec<usize>, usize, usize) {
        let timer = state
            .timer_queue
            .iter()
            .filter(|r| {
                if is_ineligible(r) {
                    return false;
                }
                match r {
                    Runnable::Timer(t) if strict_timers => t.label.as_ref().is_none_or(|l| {
                        state.allowed_timers.contains(&(t.node.index, l.clone()))
                    }),
                    _ => true,
                }
            })
            .count();
        let local = state
            .local_queues
            .iter()
            .map(|q| q.iter().filter(|r| !is_ineligible(r)).count())
            .collect();
        let network = state.network_queue.iter().filter(|r| !is_ineligible(r)).count();
        (local, network, timer)
    }

    /// The eligible sizes equal a filter of every queue, with the same counts
    /// of crashes held on a down victim, under reservations, FIFO links,
    /// strict timers, crash block masks and retargeting. A queue is filtered
    /// exactly when its node's queued crash is masked or its node is down on
    /// a retargeting run, and every queue is filtered when something other
    /// than a planned crash can be rejected.
    #[test]
    fn eligible_counts_match_a_filter_of_every_queue() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let mut rng = StdRng::seed_from_u64(23);
        let mut general = [0u32; 3];
        let mut masked_walks = 0;
        let mut victim_walks = 0;
        let mut victim_holds = 0;
        let mut pending_not_withheld = 0;
        let mut counted = 0;
        for case in 0..6000 {
            let (state, reservations, strict_timers, mask) = eligibility_case(&mut rng);
            let ineligibility = Ineligibility {
                crash_block_mask: mask,
                crashed_victims: crashed_victims(&state),
                reservations: &reservations,
                link_deliver_seq: &state.link_deliver_seq,
            };
            let is_ineligible = |r: &Runnable<NoHashing>| ineligibility.rejects(r);
            let s0 = util_stats::snapshot();
            let reference = filtered_counts(&state, strict_timers, &is_ineligible);
            let s1 = util_stats::snapshot();
            let only = only_crashes_can_be_ineligible(&state, &reservations, strict_timers);
            let mut sizes = Vec::new();
            let (network, timer) =
                eligible_counts(&state, only, strict_timers, mask, &is_ineligible, &mut sizes);
            let s2 = util_stats::snapshot();
            assert_eq!((sizes, network, timer), reference, "case {case}");

            let holds_reference = s1.victim_swap.victim_crashed_holds - s0.victim_swap.victim_crashed_holds;
            let holds = s2.victim_swap.victim_crashed_holds - s1.victim_swap.victim_crashed_holds;
            assert_eq!(holds, holds_reference, "case {case}: crashes held on a down victim");
            victim_holds += holds;

            let walked_steps = s2.sched.eligibility_walked_steps - s1.sched.eligibility_walked_steps;
            let counted_steps = s2.sched.eligibility_counted_steps - s1.sched.eligibility_counted_steps;
            let general_steps = s2.sched.eligibility_general_steps - s1.sched.eligibility_general_steps;
            let elements =
                s2.sched.eligibility_walked_elements - s1.sched.eligibility_walked_elements;
            assert_eq!(walked_steps + counted_steps, 1, "case {case}");
            if !only {
                assert_eq!((walked_steps, general_steps), (1, 1), "case {case}");
                let all = state.local_queues.iter().map(Vec::len).sum::<usize>()
                    + state.network_queue.len()
                    + state.timer_queue.len();
                assert_eq!(elements, all as u64, "case {case}");
                general[if !reservations.is_empty() {
                    0
                } else if !state.link_deliver_seq.is_empty() {
                    1
                } else {
                    2
                }] += 1;
                continue;
            }
            assert_eq!(general_steps, 0, "case {case}");
            let mut expected = 0u64;
            let mut walked = false;
            for (n, q) in state.local_queues.iter().enumerate() {
                let has_crash = q.iter().any(|r| matches!(r, Runnable::Crash { .. }));
                let masked = mask & (1u64 << n) != 0;
                let down = state.retarget.enabled
                    && state.crash_info.currently_crashed.contains(&node(n));
                if has_crash && (masked || down) {
                    walked = true;
                    expected += q.len() as u64;
                    masked_walks += u32::from(masked);
                    victim_walks += u32::from(down && !masked);
                } else if has_crash {
                    pending_not_withheld += 1;
                }
            }
            assert_eq!(walked_steps == 1, walked, "case {case}");
            assert_eq!(elements, expected, "case {case}");
            counted += counted_steps;
        }
        util_stats::set_enabled(false);
        assert!(
            general.iter().all(|&g| g > 0)
                && masked_walks > 0
                && victim_walks > 0
                && victim_holds > 0
                && pending_not_withheld > 0
                && counted > 0,
            "general {general:?} masked {masked_walks} victim {victim_walks} holds {victim_holds} \
             pending {pending_not_withheld} counted {counted}"
        );
    }
}
