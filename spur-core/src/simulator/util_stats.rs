//! Opt-in, process-wide utilization counters for explorer mechanisms.
//!
//! Enabled per explorer session via `ExplorerConfig::stats` and dumped by the
//! CLI to `<output_dir>/utilization.json`. Counters are observation-only: they
//! never affect scheduling, scoring, or RNG consumption. When disabled, every
//! probe is a single relaxed atomic load.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static ACTED_ENABLED: AtomicBool = AtomicBool::new(true);
static STEER_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);

static RNG_ISOLATED_RUNS: AtomicU64 = AtomicU64::new(0);
static RNG_SHARED_RUNS: AtomicU64 = AtomicU64::new(0);
static STEER_EVALUATIONS: AtomicU64 = AtomicU64::new(0);
static STEER_DIVERGENT_PICKS: AtomicU64 = AtomicU64::new(0);
static SA_STEPS: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_EXPRESSED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_HONORED: AtomicU64 = AtomicU64::new(0);
static SA_HONORED: AtomicU64 = AtomicU64::new(0);
static SA_NO_ELIGIBLE: AtomicU64 = AtomicU64::new(0);
static SA_BLOCKED_BY_ORDER: AtomicU64 = AtomicU64::new(0);
static SA_BLOCKED_BY_TIMER_GATE: AtomicU64 = AtomicU64::new(0);
static SA_OTHER_QUEUE: AtomicU64 = AtomicU64::new(0);
static SA_SAMPLER_CHOSE_OTHER: AtomicU64 = AtomicU64::new(0);
static PURGATORY_DELAYED_SENDS: AtomicU64 = AtomicU64::new(0);
static AOS_TAPE_WINS: AtomicU64 = AtomicU64::new(0);
static AOS_CONFIG_WINS: AtomicU64 = AtomicU64::new(0);
static DEDUP_CHECKS: AtomicU64 = AtomicU64::new(0);
static DEDUP_HITS: AtomicU64 = AtomicU64::new(0);
/// Reserved for a size-capped dedup bypass; no such path exists today, so this
/// stays 0 but is kept in the output shape for downstream consumers.
static DEDUP_SKIPPED_LARGE: AtomicU64 = AtomicU64::new(0);
// f64 sums stored as bit patterns (0u64 == 0.0f64), updated via CAS.
static FEEDBACK_TIMELINE_SCORE_SUM: AtomicU64 = AtomicU64::new(0);
static FEEDBACK_CFG_SCORE_SUM: AtomicU64 = AtomicU64::new(0);
static FEEDBACK_SCORED_RUNS: AtomicU64 = AtomicU64::new(0);
static CURRICULUM_LOWERED_RUNS: AtomicU64 = AtomicU64::new(0);
static CURRICULUM_CRASHES_SUM: AtomicU64 = AtomicU64::new(0);
static CURRICULUM_SERVERS_SUM: AtomicU64 = AtomicU64::new(0);
static CR_RUNS: AtomicU64 = AtomicU64::new(0);
static CR_CRASHES: AtomicU64 = AtomicU64::new(0);
static CR_RECOVERS: AtomicU64 = AtomicU64::new(0);
static CR_HELD_AT_CRASH: AtomicU64 = AtomicU64::new(0);
static CR_DROPPED_AT_CRASH: AtomicU64 = AtomicU64::new(0);
static CR_CROSSING_DELIVERIES: AtomicU64 = AtomicU64::new(0);
static CR_RUNS_WITH_CROSSING: AtomicU64 = AtomicU64::new(0);
static CA_STEPS_WITH_CRASH_ELIGIBLE: AtomicU64 = AtomicU64::new(0);
static CA_OFFERED: AtomicU64 = AtomicU64::new(0);
static CA_CRASHES_TAKEN: AtomicU64 = AtomicU64::new(0);
static CA_APPLIED: AtomicU64 = AtomicU64::new(0);
static RW_CLOSED: AtomicU64 = AtomicU64::new(0);
static RW_WIDTH_SUM: AtomicU64 = AtomicU64::new(0);
static RW_MAX: AtomicU64 = AtomicU64::new(0);
static RW_UNCLOSED: AtomicU64 = AtomicU64::new(0);
static OH3_RUNS: AtomicU64 = AtomicU64::new(0);
static OH3_WITH_H3: AtomicU64 = AtomicU64::new(0);
static OH3_WITH_OVERLAP: AtomicU64 = AtomicU64::new(0);
static PFO_PAIRS_SEEN: AtomicU64 = AtomicU64::new(0);
static PFO_EDGES_ADDED: AtomicU64 = AtomicU64::new(0);
static PFO_OPS_AFTER_LAST_RECOVER: AtomicU64 = AtomicU64::new(0);
static NOVELTY_ABLATED_RUNS: AtomicU64 = AtomicU64::new(0);

/// Recovery-window widths are tallied into a histogram so percentiles can be
/// read without keeping every sample. Widths at or above the cap fold into the
/// last slot, which the percentile reader reports as the cap itself.
const RW_WIDTH_CAP: usize = 1024;
static RW_WIDTHS: Mutex<[u64; RW_WIDTH_CAP + 1]> = Mutex::new([0; RW_WIDTH_CAP + 1]);

/// Crash and recover events applied in one run, clamped, as the bucket index of
/// the cross-tab. This is the only ordinal measure of how much fault activity a
/// run saw that the simulator itself observes.
const FAULT_EVENT_BUCKETS: usize = 9;
static OH3_BY_FAULT_EVENTS: Mutex<[[u64; 3]; FAULT_EVENT_BUCKETS]> =
    Mutex::new([[0; 3]; FAULT_EVENT_BUCKETS]);

/// Delivery-effect counters, laid out as (total, acted) pairs. Index 0 is every
/// delivery, index 1 every delivery that carried at least one bias, and the
/// remaining indices split that by which bias the message carried (a message
/// can carry more than one, so those three do not sum to index 1).
const DELIVERY_ALL: usize = 0;
const DELIVERY_BIASED: usize = 1;
const DELIVERY_DELAYED: usize = 2;
const DELIVERY_SENDER_RESTARTED: usize = 3;
const DELIVERY_RECEIVER_RESTARTED: usize = 4;
const DELIVERY_BUCKETS: usize = 5;

static DELIVERIES: [AtomicU64; DELIVERY_BUCKETS] =
    [const { AtomicU64::new(0) }; DELIVERY_BUCKETS];
static DELIVERIES_ACTED: [AtomicU64; DELIVERY_BUCKETS] =
    [const { AtomicU64::new(0) }; DELIVERY_BUCKETS];

static TERMINATION: Mutex<TerminationStats> = Mutex::new(TerminationStats::new());
static TIMELINE_KEYS: Mutex<TimelineKeyGrowth> = Mutex::new(TimelineKeyGrowth::new());

/// Runs per point on the timeline-key growth curve, and the most points a
/// session will report. Once the cap is reached every later run folds into the
/// final point, so the curve keeps its shape at the head where growth is fast
/// and stays bounded for a long session.
const TIMELINE_BUCKET_RUNS: u64 = 100;
const TIMELINE_MAX_BUCKETS: usize = 512;

/// Enable or disable recording for this explorer session. Enabling resets all
/// counters so repeated sessions in one process don't bleed into each other.
pub fn set_enabled(on: bool) {
    if on {
        for c in [
            &STEER_EVALUATIONS,
            &STEER_DIVERGENT_PICKS,
            &SA_STEPS,
            &SA_PREFERENCE_EXPRESSED,
            &SA_PREFERENCE_HONORED,
            &SA_HONORED,
            &SA_NO_ELIGIBLE,
            &SA_BLOCKED_BY_ORDER,
            &SA_BLOCKED_BY_TIMER_GATE,
            &SA_OTHER_QUEUE,
            &SA_SAMPLER_CHOSE_OTHER,
            &PURGATORY_DELAYED_SENDS,
            &AOS_TAPE_WINS,
            &AOS_CONFIG_WINS,
            &DEDUP_CHECKS,
            &DEDUP_HITS,
            &DEDUP_SKIPPED_LARGE,
            &FEEDBACK_TIMELINE_SCORE_SUM,
            &FEEDBACK_CFG_SCORE_SUM,
            &FEEDBACK_SCORED_RUNS,
            &CURRICULUM_LOWERED_RUNS,
            &CURRICULUM_CRASHES_SUM,
            &CURRICULUM_SERVERS_SUM,
            &CR_RUNS,
            &CR_CRASHES,
            &CR_RECOVERS,
            &CR_HELD_AT_CRASH,
            &CR_DROPPED_AT_CRASH,
            &CR_CROSSING_DELIVERIES,
            &CR_RUNS_WITH_CROSSING,
            &CA_STEPS_WITH_CRASH_ELIGIBLE,
            &CA_OFFERED,
            &CA_CRASHES_TAKEN,
            &CA_APPLIED,
            &RW_CLOSED,
            &RW_WIDTH_SUM,
            &RW_MAX,
            &RW_UNCLOSED,
            &OH3_RUNS,
            &OH3_WITH_H3,
            &OH3_WITH_OVERLAP,
            &PFO_PAIRS_SEEN,
            &PFO_EDGES_ADDED,
            &PFO_OPS_AFTER_LAST_RECOVER,
            &NOVELTY_ABLATED_RUNS,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for c in DELIVERIES.iter().chain(DELIVERIES_ACTED.iter()) {
            c.store(0, Ordering::Relaxed);
        }
        if let Ok(mut w) = RW_WIDTHS.lock() {
            *w = [0; RW_WIDTH_CAP + 1];
        }
        if let Ok(mut x) = OH3_BY_FAULT_EVENTS.lock() {
            *x = [[0; 3]; FAULT_EVENT_BUCKETS];
        }
        if let Ok(mut t) = TERMINATION.lock() {
            *t = TerminationStats::new();
        }
        if let Ok(mut g) = TIMELINE_KEYS.lock() {
            *g = TimelineKeyGrowth::new();
        }
    }
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether recording is enabled. Callers with a non-trivial probe (e.g. the
/// steer argmax comparison) should guard the computation behind this.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the delivery-effect counters for this session. They also
/// require `set_enabled(true)`; this switch exists so the per-delivery probe
/// can be turned off on its own.
pub fn set_acted_fraction_enabled(on: bool) {
    ACTED_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a delivery's effect should be measured. Callers must check this
/// before doing the before/after comparison the measurement needs.
#[inline]
pub fn acted_fraction_enabled() -> bool {
    enabled() && ACTED_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the steer-authority audit for this session. It also
/// requires `set_enabled(true)`; the audit walks every queue at every
/// scheduling point, so it is a separate switch from the cheap counters.
pub fn set_steer_audit_enabled(on: bool) {
    STEER_AUDIT_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a scheduling point should be audited. Callers must check this
/// before scoring the queues, which is the expensive part.
#[inline]
pub fn steer_audit_enabled() -> bool {
    enabled() && STEER_AUDIT_ENABLED.load(Ordering::Relaxed)
}

/// What stood between the highest-scoring runnable and the one a scheduling
/// point actually ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerOutcome {
    /// The step ran the highest-scoring runnable.
    Honored,
    /// Every runnable was filtered out, or the queue the step routed to held
    /// nothing it was allowed to run, so no event was scheduled at all.
    NoEligibleCandidates,
    /// The highest-scoring runnable was withheld by a deliver reservation or
    /// by the per-link delivery order.
    BlockedByOrder,
    /// The highest-scoring runnable was a timer whose label was not currently
    /// permitted to fire.
    BlockedByTimerGate,
    /// The highest-scoring runnable was eligible but sat in a queue the
    /// queue-level router did not choose.
    OtherQueue,
    /// The highest-scoring runnable was eligible and in the chosen queue, but
    /// the randomized within-queue sampler took a different one.
    SamplerChoseOther,
}

/// One scheduling point was audited. `expressed` means the score ranking put a
/// different runnable on top than priority alone would have, i.e. the steering
/// term changed what "preferred" means at this point; `outcome` says what
/// happened to that preferred runnable.
#[inline]
pub fn record_steer_authority(expressed: bool, outcome: SteerOutcome) {
    if !steer_audit_enabled() {
        return;
    }
    SA_STEPS.fetch_add(1, Ordering::Relaxed);
    if expressed {
        SA_PREFERENCE_EXPRESSED.fetch_add(1, Ordering::Relaxed);
        if outcome == SteerOutcome::Honored {
            SA_PREFERENCE_HONORED.fetch_add(1, Ordering::Relaxed);
        }
    }
    let counter = match outcome {
        SteerOutcome::Honored => &SA_HONORED,
        SteerOutcome::NoEligibleCandidates => &SA_NO_ELIGIBLE,
        SteerOutcome::BlockedByOrder => &SA_BLOCKED_BY_ORDER,
        SteerOutcome::BlockedByTimerGate => &SA_BLOCKED_BY_TIMER_GATE,
        SteerOutcome::OtherQueue => &SA_OTHER_QUEUE,
        SteerOutcome::SamplerChoseOther => &SA_SAMPLER_CHOSE_OTHER,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Which scheduler perturbations a message was carrying when it was delivered.
/// A message can carry several at once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeliveryBias(u8);

impl DeliveryBias {
    /// Delivered as sent: no delay, and neither end restarted in between.
    pub const NONE: Self = Self(0);
    /// Held back by the message-delay mechanism before being made schedulable.
    pub const DELAYED: Self = Self(1);
    /// The sending node crashed and recovered between the send and this
    /// delivery, so the message comes from an incarnation that no longer exists.
    pub const SENDER_RESTARTED: Self = Self(2);
    /// The message was kept across the receiver's own crash and handed to a
    /// later incarnation of it.
    pub const RECEIVER_RESTARTED: Self = Self(4);

    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A message reached a handler entry on another node. `acted` means the
/// handler changed that node's persistent state rather than falling through a
/// guard, which is the difference between a hazard the protocol saw and one it
/// ignored. Only the handler's first execution segment is observed: a write
/// that happens after the handler blocks on a channel is attributed to nothing.
#[inline]
pub fn record_delivery(bias: DeliveryBias, acted: bool) {
    if !acted_fraction_enabled() {
        return;
    }
    let mut buckets = [DELIVERY_ALL; DELIVERY_BUCKETS];
    let mut len = 1;
    if !bias.is_empty() {
        buckets[len] = DELIVERY_BIASED;
        len += 1;
    }
    for (bit, bucket) in [
        (DeliveryBias::DELAYED, DELIVERY_DELAYED),
        (DeliveryBias::SENDER_RESTARTED, DELIVERY_SENDER_RESTARTED),
        (
            DeliveryBias::RECEIVER_RESTARTED,
            DELIVERY_RECEIVER_RESTARTED,
        ),
    ] {
        if bias.contains(bit) {
            buckets[len] = bucket;
            len += 1;
        }
    }
    for &b in &buckets[..len] {
        DELIVERIES[b].fetch_add(1, Ordering::Relaxed);
        if acted {
            DELIVERIES_ACTED[b].fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn add_f64(cell: &AtomicU64, v: f64) {
    let mut cur = cell.load(Ordering::Relaxed);
    loop {
        let next = (f64::from_bits(cur) + v).to_bits();
        match cell.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// One within-queue selection over >1 eligible items was evaluated;
/// `divergent` means the blended-score argmax differed from the
/// priority-only argmax (i.e. novelty/steer changed the greedy pick).
#[inline]
pub fn record_steer_evaluation(divergent: bool) {
    if !enabled() {
        return;
    }
    STEER_EVALUATIONS.fetch_add(1, Ordering::Relaxed);
    if divergent {
        STEER_DIVERGENT_PICKS.fetch_add(1, Ordering::Relaxed);
    }
}

/// A run built its scheduling random source; `isolated` means the run drew
/// each decision kind from its own generator rather than from one shared
/// stream. `isolated_runs` equal to the number of runs is what "the mechanism
/// was on for the whole session" looks like.
#[inline]
pub fn record_rng_isolation(isolated: bool) {
    if !enabled() {
        return;
    }
    if isolated {
        RNG_ISOLATED_RUNS.fetch_add(1, Ordering::Relaxed);
    } else {
        RNG_SHARED_RUNS.fetch_add(1, Ordering::Relaxed);
    }
}

/// A record/ChannelSend was moved into purgatory instead of being enqueued.
#[inline]
pub fn record_purgatory_delay() {
    if !enabled() {
        return;
    }
    PURGATORY_DELAYED_SENDS.fetch_add(1, Ordering::Relaxed);
}

/// The AOS bandit chose an arm (`tape` = TapeMutate, else ConfigMutate).
#[inline]
pub fn record_aos_pick(tape: bool) {
    if !enabled() {
        return;
    }
    if tape {
        AOS_TAPE_WINS.fetch_add(1, Ordering::Relaxed);
    } else {
        AOS_CONFIG_WINS.fetch_add(1, Ordering::Relaxed);
    }
}

/// A candidate was checked against the dedup set; `hit` means it was rejected
/// as a duplicate.
#[inline]
pub fn record_dedup_check(hit: bool) {
    if !enabled() {
        return;
    }
    DEDUP_CHECKS.fetch_add(1, Ordering::Relaxed);
    if hit {
        DEDUP_HITS.fetch_add(1, Ordering::Relaxed);
    }
}

/// One run was scored by a feedback's `plan_score`; each component that the
/// feedback mode computes contributes to its sum (None = not tracked).
#[inline]
pub fn record_feedback_scores(timeline: Option<f64>, cfg: Option<f64>) {
    if !enabled() {
        return;
    }
    FEEDBACK_SCORED_RUNS.fetch_add(1, Ordering::Relaxed);
    if let Some(t) = timeline {
        add_f64(&FEEDBACK_TIMELINE_SCORE_SUM, t);
    }
    if let Some(c) = cfg {
        add_f64(&FEEDBACK_CFG_SCORE_SUM, c);
    }
}

/// Per-run bookkeeping for crash/recover crossings. Runs execute one at a time
/// per thread, so this lives in thread-local storage and needs no locking.
struct RunCrossingState {
    /// Node index -> records addressed to that node before it crashed that are
    /// being held for redelivery when it comes back.
    held: HashMap<usize, u64>,
    counted_in_runs_with_crossing: bool,
    /// Node index -> the scheduler step at which the node restarted, for nodes
    /// whose new incarnation has not yet been handed a message from anyone
    /// else. The recovery window is the span in which the protocol's restart
    /// work and the traffic aimed at it can still interleave.
    recovery_open: HashMap<usize, i32>,
    /// Nodes that completed a crash-and-recover cycle in this run.
    recovered: HashSet<usize>,
    /// Crash and recover events applied in this run.
    fault_events: u32,
    /// Some node crashed while another node's recovery window was open.
    crash_inside_recovery_window: bool,
    /// Client operations invoked since the most recent recover, or 0 while no
    /// node has recovered yet. Whatever is left here when the run ends is the
    /// client work that outlived every fault.
    client_ops_since_last_recover: u64,
    any_recover: bool,
    finalized: bool,
}

impl Default for RunCrossingState {
    fn default() -> Self {
        Self {
            held: HashMap::new(),
            counted_in_runs_with_crossing: false,
            recovery_open: HashMap::new(),
            recovered: HashSet::new(),
            fault_events: 0,
            crash_inside_recovery_window: false,
            client_ops_since_last_recover: 0,
            any_recover: false,
            // Nothing has been recorded yet, so there is no run to flush.
            finalized: true,
        }
    }
}

thread_local! {
    static RUN_CROSSING: RefCell<RunCrossingState> = RefCell::new(RunCrossingState::default());
}

/// Fold this thread's finished run into the per-run tallies. Idempotent, so it
/// can be called both from the normal end-of-run hook and from the start of the
/// next run for runs that ended without reaching it.
fn finish_run() {
    let Some((unclosed, h3, ordered, bucket, post_recover_ops)) = RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        if c.finalized {
            return None;
        }
        c.finalized = true;
        let unclosed = c.recovery_open.len() as u64;
        c.recovery_open.clear();
        let h3 = c.recovered.len() >= 2;
        Some((
            unclosed,
            h3,
            h3 && c.crash_inside_recovery_window,
            (c.fault_events as usize).min(FAULT_EVENT_BUCKETS - 1),
            c.client_ops_since_last_recover,
        ))
    }) else {
        return;
    };
    PFO_OPS_AFTER_LAST_RECOVER.fetch_add(post_recover_ops, Ordering::Relaxed);
    RW_UNCLOSED.fetch_add(unclosed, Ordering::Relaxed);
    OH3_RUNS.fetch_add(1, Ordering::Relaxed);
    if h3 {
        OH3_WITH_H3.fetch_add(1, Ordering::Relaxed);
    }
    if ordered {
        OH3_WITH_OVERLAP.fetch_add(1, Ordering::Relaxed);
    }
    if let Ok(mut x) = OH3_BY_FAULT_EVENTS.lock() {
        x[bucket][0] += 1;
        if h3 {
            x[bucket][1] += 1;
        }
        if ordered {
            x[bucket][2] += 1;
        }
    }
}

/// One plan execution is starting on this thread. Held-message bookkeeping is
/// per-run, so anything left over from a run that ended while a node was still
/// down is discarded here rather than leaking into the next run.
pub fn begin_run() {
    if !enabled() {
        return;
    }
    finish_run();
    CR_RUNS.fetch_add(1, Ordering::Relaxed);
    RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        c.held.clear();
        c.counted_in_runs_with_crossing = false;
        c.recovery_open.clear();
        c.recovered.clear();
        c.fault_events = 0;
        c.crash_inside_recovery_window = false;
        c.client_ops_since_last_recover = 0;
        c.any_recover = false;
        c.finalized = false;
    });
}

/// The plan generator examined one crash-and-recover pair for post-fault client
/// work and added `edges_added` mandatory recover-before-request edges for it.
#[inline]
pub fn record_post_fault_ops(pairs_seen: u64, edges_added: u64) {
    if !enabled() {
        return;
    }
    PFO_PAIRS_SEEN.fetch_add(pairs_seen, Ordering::Relaxed);
    PFO_EDGES_ADDED.fetch_add(edges_added, Ordering::Relaxed);
}

/// A planned client operation was handed to a client node. Only invocations
/// that follow every recover in the run are reported, so this is the execution
/// side of the same question the generator edges are meant to force.
#[inline]
pub fn record_client_op_invoked() {
    if !enabled() {
        return;
    }
    RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        if c.any_recover {
            c.client_ops_since_last_recover += 1;
        }
    });
}

/// A node crashed: `held` messages addressed to it were kept for redelivery
/// after it recovers, `dropped` were discarded.
pub fn record_crash(node_index: usize, held: u64, dropped: u64) {
    if !enabled() {
        return;
    }
    CR_CRASHES.fetch_add(1, Ordering::Relaxed);
    CR_HELD_AT_CRASH.fetch_add(held, Ordering::Relaxed);
    CR_DROPPED_AT_CRASH.fetch_add(dropped, Ordering::Relaxed);
    let closed_unclosed = RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        if held > 0 {
            *c.held.entry(node_index).or_insert(0) += held;
        }
        c.fault_events += 1;
        // A node's own crash ends its recovery window without a width: no
        // message ever reached the incarnation that came back.
        let own = c.recovery_open.remove(&node_index).is_some();
        if c.recovery_open.keys().any(|&n| n != node_index) {
            c.crash_inside_recovery_window = true;
        }
        own
    });
    if closed_unclosed {
        RW_UNCLOSED.fetch_add(1, Ordering::Relaxed);
    }
}

/// A message from another node entered a handler on `node_index` at scheduler
/// step `step`. This closes the node's recovery window if one is open.
pub fn record_message_entry(node_index: usize, step: i32) {
    if !enabled() {
        return;
    }
    let opened_at = RUN_CROSSING.with(|c| c.borrow_mut().recovery_open.remove(&node_index));
    let Some(opened_at) = opened_at else {
        return;
    };
    let width = step.saturating_sub(opened_at).max(0) as u64;
    RW_CLOSED.fetch_add(1, Ordering::Relaxed);
    RW_WIDTH_SUM.fetch_add(width, Ordering::Relaxed);
    RW_MAX.fetch_max(width, Ordering::Relaxed);
    if let Ok(mut w) = RW_WIDTHS.lock() {
        w[(width as usize).min(RW_WIDTH_CAP)] += 1;
    }
}

/// A node recovered. Every message held from before its crash is requeued to
/// it at this point, so each one is a delivery that crosses the node's own
/// crash/recover boundary. Messages that arrived while the node was down are
/// requeued too but are not counted: they were sent to an already-dead node.
pub fn record_recover(node_index: usize, step: i32) {
    if !enabled() {
        return;
    }
    CR_RECOVERS.fetch_add(1, Ordering::Relaxed);
    let first_crossing_of_run = RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        c.fault_events += 1;
        c.recovered.insert(node_index);
        c.recovery_open.insert(node_index, step);
        c.any_recover = true;
        c.client_ops_since_last_recover = 0;
        let crossings = c.held.remove(&node_index).unwrap_or(0);
        if crossings == 0 {
            return None;
        }
        let first = !c.counted_in_runs_with_crossing;
        c.counted_in_runs_with_crossing = true;
        Some((crossings, first))
    });
    let Some((crossings, first)) = first_crossing_of_run else {
        return;
    };
    CR_CROSSING_DELIVERIES.fetch_add(crossings, Ordering::Relaxed);
    if first {
        CR_RUNS_WITH_CROSSING.fetch_add(1, Ordering::Relaxed);
    }
}

/// One scheduling step was examined for the crash-after-send condition:
/// `crash_eligible` means some node had a schedulable crash waiting, and
/// `anchored` means at least one such node also had a message it sent still
/// undelivered, so crashing it there would leave that message orphaned.
#[inline]
pub fn record_crash_anchor_offer(crash_eligible: bool, anchored: bool) {
    if !enabled() {
        return;
    }
    if crash_eligible {
        CA_STEPS_WITH_CRASH_ELIGIBLE.fetch_add(1, Ordering::Relaxed);
    }
    if anchored {
        CA_OFFERED.fetch_add(1, Ordering::Relaxed);
    }
}

/// A crash was executed; `anchored` means the crashing node had a message it
/// sent still undelivered at that moment.
#[inline]
pub fn record_crash_anchor_apply(anchored: bool) {
    if !enabled() {
        return;
    }
    CA_CRASHES_TAKEN.fetch_add(1, Ordering::Relaxed);
    if anchored {
        CA_APPLIED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Why a single plan execution stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunEnd {
    /// Every planned event reached Completed.
    PlanComplete,
    /// The per-run step budget ran out while planned events were outstanding.
    IterationsExhausted,
    /// No runnable work and no planned event able to become ready.
    Deadlock,
}

/// Termination counts and running sums over one bucket of runs. `steps_used`
/// and the queue depths are summed rather than averaged so buckets can be
/// merged; divide by `runs` to read a mean.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TerminationTally {
    pub runs: u64,
    pub plan_complete: u64,
    /// Plan finished while messages, timers or delayed sends were still
    /// queued, so the remaining protocol traffic never ran.
    pub plan_complete_with_pending_work: u64,
    pub iterations_exhausted: u64,
    pub deadlock: u64,
    pub steps_used_sum: u64,
    pub step_budget_sum: u64,
    pub pending_work_at_exit_sum: u64,
    pub planned_events_outstanding_sum: u64,
}

impl TerminationTally {
    const fn new() -> Self {
        Self {
            runs: 0,
            plan_complete: 0,
            plan_complete_with_pending_work: 0,
            iterations_exhausted: 0,
            deadlock: 0,
            steps_used_sum: 0,
            step_budget_sum: 0,
            pending_work_at_exit_sum: 0,
            planned_events_outstanding_sum: 0,
        }
    }

    fn add(&mut self, end: RunEnd, s: &RunTermination) {
        self.runs += 1;
        match end {
            RunEnd::PlanComplete => {
                self.plan_complete += 1;
                if s.pending_work_at_exit > 0 {
                    self.plan_complete_with_pending_work += 1;
                }
            }
            RunEnd::IterationsExhausted => self.iterations_exhausted += 1,
            RunEnd::Deadlock => self.deadlock += 1,
        }
        self.steps_used_sum += s.steps_used;
        self.step_budget_sum += s.step_budget;
        self.pending_work_at_exit_sum += s.pending_work_at_exit;
        self.planned_events_outstanding_sum += s.planned_events_outstanding;
    }
}

/// One run's termination facts.
pub struct RunTermination {
    pub end: RunEnd,
    pub steps_used: u64,
    pub step_budget: u64,
    /// Runnables still queued (including delayed sends) when the run stopped.
    pub pending_work_at_exit: u64,
    pub planned_events_outstanding: u64,
    /// Distinct nodes that both crashed and recovered during the run. Deep
    /// fault interleavings need at least two; the bucketed tallies show
    /// whether those runs stop for a different reason than shallow ones.
    pub recovered_nodes: usize,
}

/// Termination tallies over all runs and split by how many distinct nodes
/// completed a crash-and-recover cycle (index 0, 1, and 2-or-more).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TerminationStats {
    pub all: TerminationTally,
    pub by_recovered_nodes: [TerminationTally; 3],
}

impl TerminationStats {
    const fn new() -> Self {
        Self {
            all: TerminationTally::new(),
            by_recovered_nodes: [TerminationTally::new(); 3],
        }
    }
}

/// One plan execution finished. Called once per run, off the scheduling hot
/// path.
pub fn record_run_termination(s: &RunTermination) {
    if !enabled() {
        return;
    }
    finish_run();
    let bucket = s.recovered_nodes.min(2);
    if let Ok(mut t) = TERMINATION.lock() {
        t.all.add(s.end, s);
        t.by_recovered_nodes[bucket].add(s.end, s);
    }
}

/// One point on the timeline-key growth curve, covering `runs` consecutive
/// runs starting at `first_run`. `cumulative_distinct` is the running total of
/// keys ever inserted as of the last run in the point.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TimelineGrowthBucket {
    pub first_run: u64,
    pub runs: u64,
    pub keys_in_run_sum: u64,
    pub new_keys: u64,
    pub cumulative_distinct: u64,
}

/// Running totals over the coverage keys a session's runs produce.
struct TimelineKeyGrowth {
    runs: u64,
    keys_in_run_sum: u64,
    new_keys_sum: u64,
    distinct_keys_live: u64,
    buckets: Vec<TimelineGrowthBucket>,
}

impl TimelineKeyGrowth {
    const fn new() -> Self {
        Self {
            runs: 0,
            keys_in_run_sum: 0,
            new_keys_sum: 0,
            distinct_keys_live: 0,
            buckets: Vec::new(),
        }
    }
}

/// A run built its coverage keys with novelty turned off, so every ordering it
/// saw collapsed onto one key. Recorded once per run.
#[inline]
pub fn record_novelty_ablated_run() {
    if !enabled() {
        return;
    }
    NOVELTY_ABLATED_RUNS.fetch_add(1, Ordering::Relaxed);
}

/// One run's coverage keys were folded into the shared store: `keys_in_run` is
/// how many keys the run produced, `new_keys` how many of those the store had
/// never seen, and `distinct_keys_live` the store's size afterwards (which is
/// smaller than the cumulative total whenever the store has been decayed).
pub fn record_timeline_keys(keys_in_run: u64, new_keys: u64, distinct_keys_live: u64) {
    if !enabled() {
        return;
    }
    let Ok(mut g) = TIMELINE_KEYS.lock() else {
        return;
    };
    g.runs += 1;
    g.keys_in_run_sum += keys_in_run;
    g.new_keys_sum += new_keys;
    g.distinct_keys_live = distinct_keys_live;
    let first_run = g.runs;
    let cumulative = g.new_keys_sum;
    let start_new = match g.buckets.last() {
        Some(b) => b.runs >= TIMELINE_BUCKET_RUNS,
        None => true,
    };
    if start_new && g.buckets.len() < TIMELINE_MAX_BUCKETS {
        g.buckets.push(TimelineGrowthBucket {
            first_run,
            runs: 0,
            keys_in_run_sum: 0,
            new_keys: 0,
            cumulative_distinct: 0,
        });
    }
    if let Some(b) = g.buckets.last_mut() {
        b.runs += 1;
        b.keys_in_run_sum += keys_in_run;
        b.new_keys += new_keys;
        b.cumulative_distinct = cumulative;
    }
}

/// How fast the coverage-key space is still growing. `saturation_run_index` is
/// the run at which growth first fell below one new key per run over a whole
/// curve point, or 0 if it never did.
#[derive(Serialize)]
pub struct TimelineKeyStats {
    pub runs: u64,
    pub keys_in_run_sum: u64,
    pub cumulative_distinct_keys: u64,
    pub distinct_keys_live: u64,
    pub saturation_run_index: u64,
    pub bucket_runs: u64,
    /// Runs whose keys were built with novelty off. Nonzero means the coverage
    /// channel was ablated, and `distinct_keys_live` should then be 1.
    pub novelty_ablated_runs: u64,
    pub growth_curve: Vec<TimelineGrowthBucket>,
}

impl TimelineKeyStats {
    fn read() -> Self {
        let g = TIMELINE_KEYS
            .lock()
            .map(|g| {
                (
                    g.runs,
                    g.keys_in_run_sum,
                    g.new_keys_sum,
                    g.distinct_keys_live,
                    g.buckets.clone(),
                )
            })
            .unwrap_or_else(|p| {
                let g = p.into_inner();
                (
                    g.runs,
                    g.keys_in_run_sum,
                    g.new_keys_sum,
                    g.distinct_keys_live,
                    g.buckets.clone(),
                )
            });
        let (runs, keys_in_run_sum, new_keys_sum, distinct_keys_live, buckets) = g;
        let saturation_run_index = buckets
            .iter()
            .find(|b| b.runs >= TIMELINE_BUCKET_RUNS && b.new_keys < b.runs)
            .map(|b| b.first_run + b.runs - 1)
            .unwrap_or(0);
        Self {
            runs,
            keys_in_run_sum,
            cumulative_distinct_keys: new_keys_sum,
            distinct_keys_live,
            saturation_run_index,
            bucket_runs: TIMELINE_BUCKET_RUNS,
            novelty_ablated_runs: NOVELTY_ABLATED_RUNS.load(Ordering::Relaxed),
            growth_curve: buckets,
        }
    }
}

/// The curriculum lowered its knobs into one concrete run config. Zero here
/// means the curriculum was not on the path that produced these runs.
#[inline]
pub fn record_curriculum_lowering(num_crashes: i32, num_servers: i32) {
    if !enabled() {
        return;
    }
    CURRICULUM_LOWERED_RUNS.fetch_add(1, Ordering::Relaxed);
    CURRICULUM_CRASHES_SUM.fetch_add(num_crashes.max(0) as u64, Ordering::Relaxed);
    CURRICULUM_SERVERS_SUM.fetch_add(num_servers.max(0) as u64, Ordering::Relaxed);
}

#[derive(Serialize)]
pub struct CurriculumStats {
    pub lowered_runs: u64,
    pub crashes_sum: u64,
    pub servers_sum: u64,
}

/// How many runs drew their schedule from per-decision generators, and how
/// many drew from a single shared one.
#[derive(Serialize)]
pub struct RngStreamStats {
    pub isolated_runs: u64,
    pub shared_runs: u64,
}

#[derive(Serialize)]
pub struct SteerStats {
    pub evaluations: u64,
    pub divergent_picks: u64,
}

/// How often the runnable the scoring function ranked first is the one the
/// scheduling point ran, and what took precedence when it was not. The buckets
/// after `honored` partition the remaining steps by the single constraint that
/// stood in the way, so all six sum to `steps`.
#[derive(Serialize)]
pub struct SteerAuthorityStats {
    pub steps: u64,
    /// Steps where the steering term put a different runnable on top than
    /// priority alone would have. The denominator for `preference_honored`:
    /// on the other steps the audit cannot tell steer's choice from the
    /// choice the scheduler would have made without it.
    pub preference_expressed: u64,
    pub preference_honored: u64,
    pub honored: u64,
    pub no_eligible_candidates: u64,
    pub blocked_by_order: u64,
    pub blocked_by_timer_gate: u64,
    pub other_queue: u64,
    pub sampler_chose_other: u64,
}

#[derive(Serialize)]
pub struct PurgatoryStats {
    pub delayed_sends: u64,
}

#[derive(Serialize)]
pub struct AosStats {
    pub tape_wins: u64,
    pub config_wins: u64,
}

#[derive(Serialize)]
pub struct DedupStats {
    pub checks: u64,
    pub hits: u64,
    pub skipped_large: u64,
}

#[derive(Serialize)]
pub struct FeedbackStats {
    pub timeline_score_sum: f64,
    pub cfg_score_sum: f64,
    pub scored_runs: u64,
}

/// How dense crash/recover activity is, and how often a message actually
/// survives a receiver's crash to be delivered after it comes back.
#[derive(Serialize)]
pub struct CrashRecoveryStats {
    /// Plan executions observed, i.e. the denominator for `runs_with_crossing`.
    pub runs: u64,
    pub crashes: u64,
    pub recovers: u64,
    /// Messages from another node that were queued to a node when it crashed
    /// and were kept for redelivery.
    pub messages_held_at_crash: u64,
    /// Work queued to a node at crash time that was thrown away instead:
    /// the node's own in-progress continuations and channel sends to it.
    pub messages_dropped_at_crash: u64,
    /// Held messages that were requeued to their target when it recovered,
    /// so they were sent to a live node, survived its downtime, and are
    /// handled by a different incarnation than the one they were sent to.
    pub crossing_deliveries: u64,
    pub runs_with_crossing: u64,
}

/// Deliveries and the share of them that changed the receiving node's state,
/// for one bias bucket.
#[derive(Serialize)]
pub struct DeliveryEffect {
    pub deliveries: u64,
    pub acted: u64,
    pub acted_fraction: f64,
}

impl DeliveryEffect {
    fn read(bucket: usize) -> Self {
        let deliveries = DELIVERIES[bucket].load(Ordering::Relaxed);
        let acted = DELIVERIES_ACTED[bucket].load(Ordering::Relaxed);
        Self {
            deliveries,
            acted,
            acted_fraction: if deliveries == 0 {
                0.0
            } else {
                acted as f64 / deliveries as f64
            },
        }
    }
}

/// How often a message that reached a handler actually changed the receiver's
/// state, split by which perturbation the message was carrying. A bias whose
/// `acted_fraction` is near zero is being delivered but ignored, which is a
/// different failure than one whose `deliveries` is near zero.
#[derive(Serialize)]
pub struct DeliveryEffectStats {
    pub all: DeliveryEffect,
    pub biased: DeliveryEffect,
    pub delayed: DeliveryEffect,
    pub sender_restarted: DeliveryEffect,
    pub receiver_restarted: DeliveryEffect,
}

/// How often a crash could be, and actually was, scheduled at the moment its
/// node had an undelivered message in the network. `offered` over
/// `steps_with_crash_eligible` says how often the situation arises at all;
/// `applied` over `crashes_taken` says how often the scheduler lands on it
/// without being pushed.
#[derive(Serialize)]
pub struct CrashAnchorStats {
    pub steps_with_crash_eligible: u64,
    pub offered: u64,
    pub crashes_taken: u64,
    pub applied: u64,
}

/// How wide the interval between a node's restart and the first message handed
/// to the incarnation that came back is, in scheduler steps. A window that is
/// only a step or two wide is one nothing else can be scheduled into.
#[derive(Serialize)]
pub struct RecoveryWindowStats {
    /// Windows that closed because a message reached the restarted node.
    pub count: u64,
    /// Restarts whose window never closed: the node crashed again, or the run
    /// ended, before anything was handed to it.
    pub unclosed: u64,
    pub mean_events_open: f64,
    pub p50: u64,
    pub p90: u64,
    pub max: u64,
    /// Widths at or above this are reported as this value.
    pub width_cap: u64,
}

impl RecoveryWindowStats {
    fn read() -> Self {
        let count = RW_CLOSED.load(Ordering::Relaxed);
        let hist = RW_WIDTHS
            .lock()
            .map(|w| *w)
            .unwrap_or_else(|p| *p.into_inner());
        Self {
            count,
            unclosed: RW_UNCLOSED.load(Ordering::Relaxed),
            mean_events_open: if count == 0 {
                0.0
            } else {
                RW_WIDTH_SUM.load(Ordering::Relaxed) as f64 / count as f64
            },
            p50: histogram_quantile(&hist, count, 0.5),
            p90: histogram_quantile(&hist, count, 0.9),
            max: RW_MAX.load(Ordering::Relaxed),
            width_cap: RW_WIDTH_CAP as u64,
        }
    }
}

fn histogram_quantile(hist: &[u64], total: u64, q: f64) -> u64 {
    if total == 0 {
        return 0;
    }
    let target = ((total as f64) * q).ceil().max(1.0) as u64;
    let mut seen = 0u64;
    for (width, &n) in hist.iter().enumerate() {
        seen += n;
        if seen >= target {
            return width as u64;
        }
    }
    (hist.len() - 1) as u64
}

/// Runs holding two distinct nodes' crash-and-recover cycles, split by whether
/// those cycles were merely both present or actually interleaved.
#[derive(Serialize)]
pub struct OrderedH3Stats {
    /// Runs folded into these tallies, the denominator for both rates.
    pub runs: u64,
    /// Runs in which two or more distinct nodes each crashed and came back.
    pub runs_with_h3: u64,
    /// The subset of those in which a crash landed while some other node's
    /// recovery window was still open, so the two cycles overlapped rather than
    /// running one after the other.
    pub runs_with_overlap: u64,
    /// The same three counts split by how many crash and recover events the run
    /// applied, index 0..8 with 8 meaning eight or more. The simulator has no
    /// view of the grader's prefix depth, so this is the severity axis it can
    /// report on its own.
    pub by_fault_events: Vec<OrderedH3Bucket>,
}

#[derive(Serialize)]
pub struct OrderedH3Bucket {
    pub runs: u64,
    pub runs_with_h3: u64,
    pub runs_with_overlap: u64,
}

impl OrderedH3Stats {
    fn read() -> Self {
        let table = OH3_BY_FAULT_EVENTS
            .lock()
            .map(|x| *x)
            .unwrap_or_else(|p| *p.into_inner());
        Self {
            runs: OH3_RUNS.load(Ordering::Relaxed),
            runs_with_h3: OH3_WITH_H3.load(Ordering::Relaxed),
            runs_with_overlap: OH3_WITH_OVERLAP.load(Ordering::Relaxed),
            by_fault_events: table
                .iter()
                .map(|b| OrderedH3Bucket {
                    runs: b[0],
                    runs_with_h3: b[1],
                    runs_with_overlap: b[2],
                })
                .collect(),
        }
    }
}

/// Whether the generator managed to reserve client work for after a fault, and
/// whether the reservation survived into execution. `edges_added` well below
/// `pairs_seen` times the configured count means the graph had no client
/// request left that could be ordered after a recover without closing a cycle;
/// `ops_invoked_after_last_recover` at zero means no run ever got there.
#[derive(Serialize)]
pub struct PostFaultOpsStats {
    pub pairs_seen: u64,
    pub edges_added: u64,
    pub ops_invoked_after_last_recover: u64,
}

/// A point-in-time copy of all counters, serializable to `utilization.json`.
#[derive(Serialize)]
pub struct UtilizationSnapshot {
    pub rng_streams: RngStreamStats,
    pub steer: SteerStats,
    pub steer_authority: SteerAuthorityStats,
    pub purgatory: PurgatoryStats,
    pub aos: AosStats,
    pub dedup: DedupStats,
    pub feedback: FeedbackStats,
    pub curriculum: CurriculumStats,
    pub crash_recovery: CrashRecoveryStats,
    pub recovery_window: RecoveryWindowStats,
    pub ordered_h3: OrderedH3Stats,
    pub post_fault_ops: PostFaultOpsStats,
    pub delivery_effects: DeliveryEffectStats,
    pub crash_anchor: CrashAnchorStats,
    pub termination: TerminationStats,
    pub timeline_keys: TimelineKeyStats,
}

pub fn snapshot() -> UtilizationSnapshot {
    UtilizationSnapshot {
        rng_streams: RngStreamStats {
            isolated_runs: RNG_ISOLATED_RUNS.load(Ordering::Relaxed),
            shared_runs: RNG_SHARED_RUNS.load(Ordering::Relaxed),
        },
        steer: SteerStats {
            evaluations: STEER_EVALUATIONS.load(Ordering::Relaxed),
            divergent_picks: STEER_DIVERGENT_PICKS.load(Ordering::Relaxed),
        },
        steer_authority: SteerAuthorityStats {
            steps: SA_STEPS.load(Ordering::Relaxed),
            preference_expressed: SA_PREFERENCE_EXPRESSED.load(Ordering::Relaxed),
            preference_honored: SA_PREFERENCE_HONORED.load(Ordering::Relaxed),
            honored: SA_HONORED.load(Ordering::Relaxed),
            no_eligible_candidates: SA_NO_ELIGIBLE.load(Ordering::Relaxed),
            blocked_by_order: SA_BLOCKED_BY_ORDER.load(Ordering::Relaxed),
            blocked_by_timer_gate: SA_BLOCKED_BY_TIMER_GATE.load(Ordering::Relaxed),
            other_queue: SA_OTHER_QUEUE.load(Ordering::Relaxed),
            sampler_chose_other: SA_SAMPLER_CHOSE_OTHER.load(Ordering::Relaxed),
        },
        purgatory: PurgatoryStats {
            delayed_sends: PURGATORY_DELAYED_SENDS.load(Ordering::Relaxed),
        },
        aos: AosStats {
            tape_wins: AOS_TAPE_WINS.load(Ordering::Relaxed),
            config_wins: AOS_CONFIG_WINS.load(Ordering::Relaxed),
        },
        dedup: DedupStats {
            checks: DEDUP_CHECKS.load(Ordering::Relaxed),
            hits: DEDUP_HITS.load(Ordering::Relaxed),
            skipped_large: DEDUP_SKIPPED_LARGE.load(Ordering::Relaxed),
        },
        feedback: FeedbackStats {
            timeline_score_sum: f64::from_bits(FEEDBACK_TIMELINE_SCORE_SUM.load(Ordering::Relaxed)),
            cfg_score_sum: f64::from_bits(FEEDBACK_CFG_SCORE_SUM.load(Ordering::Relaxed)),
            scored_runs: FEEDBACK_SCORED_RUNS.load(Ordering::Relaxed),
        },
        curriculum: CurriculumStats {
            lowered_runs: CURRICULUM_LOWERED_RUNS.load(Ordering::Relaxed),
            crashes_sum: CURRICULUM_CRASHES_SUM.load(Ordering::Relaxed),
            servers_sum: CURRICULUM_SERVERS_SUM.load(Ordering::Relaxed),
        },
        crash_recovery: CrashRecoveryStats {
            runs: CR_RUNS.load(Ordering::Relaxed),
            crashes: CR_CRASHES.load(Ordering::Relaxed),
            recovers: CR_RECOVERS.load(Ordering::Relaxed),
            messages_held_at_crash: CR_HELD_AT_CRASH.load(Ordering::Relaxed),
            messages_dropped_at_crash: CR_DROPPED_AT_CRASH.load(Ordering::Relaxed),
            crossing_deliveries: CR_CROSSING_DELIVERIES.load(Ordering::Relaxed),
            runs_with_crossing: CR_RUNS_WITH_CROSSING.load(Ordering::Relaxed),
        },
        recovery_window: RecoveryWindowStats::read(),
        ordered_h3: OrderedH3Stats::read(),
        post_fault_ops: PostFaultOpsStats {
            pairs_seen: PFO_PAIRS_SEEN.load(Ordering::Relaxed),
            edges_added: PFO_EDGES_ADDED.load(Ordering::Relaxed),
            ops_invoked_after_last_recover: PFO_OPS_AFTER_LAST_RECOVER.load(Ordering::Relaxed),
        },
        delivery_effects: DeliveryEffectStats {
            all: DeliveryEffect::read(DELIVERY_ALL),
            biased: DeliveryEffect::read(DELIVERY_BIASED),
            delayed: DeliveryEffect::read(DELIVERY_DELAYED),
            sender_restarted: DeliveryEffect::read(DELIVERY_SENDER_RESTARTED),
            receiver_restarted: DeliveryEffect::read(DELIVERY_RECEIVER_RESTARTED),
        },
        crash_anchor: CrashAnchorStats {
            steps_with_crash_eligible: CA_STEPS_WITH_CRASH_ELIGIBLE.load(Ordering::Relaxed),
            offered: CA_OFFERED.load(Ordering::Relaxed),
            crashes_taken: CA_CRASHES_TAKEN.load(Ordering::Relaxed),
            applied: CA_APPLIED.load(Ordering::Relaxed),
        },
        termination: TERMINATION
            .lock()
            .map(|t| *t)
            .unwrap_or_else(|p| *p.into_inner()),
        timeline_keys: TimelineKeyStats::read(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    #[test]
    fn delivery_effects_split_by_bias() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);

        record_delivery(DeliveryBias::NONE, true);
        record_delivery(DeliveryBias::DELAYED, false);
        let mut both = DeliveryBias::DELAYED;
        both.insert(DeliveryBias::SENDER_RESTARTED);
        record_delivery(both, true);

        let s = snapshot().delivery_effects;
        set_enabled(false);

        assert_eq!(s.all.deliveries, 3);
        assert_eq!(s.all.acted, 2);
        assert_eq!(s.biased.deliveries, 2);
        assert_eq!(s.biased.acted, 1);
        assert_eq!(s.delayed.deliveries, 2);
        assert_eq!(s.delayed.acted, 1);
        assert_eq!(s.sender_restarted.deliveries, 1);
        assert_eq!(s.receiver_restarted.deliveries, 0);
        assert_eq!(s.receiver_restarted.acted_fraction, 0.0);
        assert!((s.all.acted_fraction - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn steer_authority_outcomes_partition_the_steps() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(true);

        record_steer_authority(true, SteerOutcome::Honored);
        record_steer_authority(false, SteerOutcome::Honored);
        record_steer_authority(true, SteerOutcome::SamplerChoseOther);
        record_steer_authority(true, SteerOutcome::OtherQueue);
        record_steer_authority(false, SteerOutcome::BlockedByTimerGate);
        record_steer_authority(false, SteerOutcome::BlockedByOrder);
        record_steer_authority(false, SteerOutcome::NoEligibleCandidates);

        let s = snapshot().steer_authority;
        set_steer_audit_enabled(false);
        set_enabled(false);

        assert_eq!(s.steps, 7);
        assert_eq!(s.preference_expressed, 3);
        assert_eq!(s.preference_honored, 1);
        assert_eq!(
            s.honored
                + s.no_eligible_candidates
                + s.blocked_by_order
                + s.blocked_by_timer_gate
                + s.other_queue
                + s.sampler_chose_other,
            s.steps
        );
    }

    #[test]
    fn steer_authority_records_nothing_when_the_audit_is_off() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(false);

        record_steer_authority(true, SteerOutcome::Honored);

        let s = snapshot().steer_authority;
        set_enabled(false);

        assert_eq!(s.steps, 0);
    }

    #[test]
    fn recovery_windows_close_on_the_first_message_to_the_restarted_node() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        begin_run();
        record_crash(0, 0, 0);
        record_recover(0, 10);
        // A message to a node with no open window leaves the tallies alone.
        record_message_entry(1, 12);
        record_message_entry(0, 14);
        record_crash(1, 0, 0);
        record_recover(1, 20);

        begin_run();

        let s = snapshot();
        set_enabled(false);

        assert_eq!(s.recovery_window.count, 1);
        assert_eq!(s.recovery_window.max, 4);
        assert_eq!(s.recovery_window.p50, 4);
        assert_eq!(s.recovery_window.mean_events_open, 4.0);
        // Node 1 recovered and the run ended with nothing delivered to it.
        assert_eq!(s.recovery_window.unclosed, 1);
        assert_eq!(s.ordered_h3.runs, 1);
        assert_eq!(s.ordered_h3.runs_with_h3, 1);
        // Node 1 crashed after node 0's window had already closed.
        assert_eq!(s.ordered_h3.runs_with_overlap, 0);
        assert_eq!(s.ordered_h3.by_fault_events[4].runs, 1);
    }

    #[test]
    fn a_crash_inside_an_open_recovery_window_is_an_overlap() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        begin_run();
        record_crash(0, 0, 0);
        record_recover(0, 5);
        record_crash(1, 0, 0);
        record_recover(1, 9);
        record_message_entry(0, 11);
        begin_run();

        let s = snapshot();
        set_enabled(false);

        assert_eq!(s.ordered_h3.runs_with_h3, 1);
        assert_eq!(s.ordered_h3.runs_with_overlap, 1);
        assert_eq!(s.recovery_window.count, 1);
        assert_eq!(s.recovery_window.max, 6);
    }

    #[test]
    fn timeline_key_growth_accumulates_into_one_curve_point() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        record_timeline_keys(10, 10, 10);
        record_timeline_keys(12, 3, 13);

        let s = snapshot().timeline_keys;
        set_enabled(false);

        assert_eq!(s.runs, 2);
        assert_eq!(s.keys_in_run_sum, 22);
        assert_eq!(s.cumulative_distinct_keys, 13);
        assert_eq!(s.distinct_keys_live, 13);
        assert_eq!(s.growth_curve.len(), 1);
        assert_eq!(s.growth_curve[0].first_run, 1);
        assert_eq!(s.growth_curve[0].runs, 2);
        assert_eq!(s.growth_curve[0].cumulative_distinct, 13);
        // Saturation needs a full point to be judged, so a short session
        // reports none rather than declaring saturation early.
        assert_eq!(s.saturation_run_index, 0);
    }
}
