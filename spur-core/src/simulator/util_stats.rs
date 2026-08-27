//! Opt-in, process-wide utilization counters for explorer mechanisms.
//!
//! Enabled per explorer session via `ExplorerConfig::stats` and dumped by the
//! CLI to `<output_dir>/utilization.json`. Counters are observation-only: they
//! never affect scheduling, scoring, or RNG consumption. When disabled, every
//! probe is a single relaxed atomic load.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static ACTED_ENABLED: AtomicBool = AtomicBool::new(true);

static STEER_EVALUATIONS: AtomicU64 = AtomicU64::new(0);
static STEER_DIVERGENT_PICKS: AtomicU64 = AtomicU64::new(0);
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

/// Runs per point on the timeline-key growth curve. Small enough that a short
/// session still produces several points, large enough that the per-point
/// new-key rate is not dominated by single-run variance.
const GROWTH_BUCKET_RUNS: u64 = 100;
/// Points kept on the growth curve. Once the curve is this long the totals keep
/// accumulating but no further points are appended, so the serialized snapshot
/// stays bounded on very long sessions.
const MAX_GROWTH_BUCKETS: usize = 4096;

static TIMELINE_GROWTH: Mutex<TimelineGrowth> = Mutex::new(TimelineGrowth::new());
static TL_DISTINCT_LIVE: AtomicU64 = AtomicU64::new(0);

/// Enable or disable recording for this explorer session. Enabling resets all
/// counters so repeated sessions in one process don't bleed into each other.
pub fn set_enabled(on: bool) {
    if on {
        for c in [
            &STEER_EVALUATIONS,
            &STEER_DIVERGENT_PICKS,
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
            &TL_DISTINCT_LIVE,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for c in DELIVERIES.iter().chain(DELIVERIES_ACTED.iter()) {
            c.store(0, Ordering::Relaxed);
        }
        if let Ok(mut t) = TERMINATION.lock() {
            *t = TerminationStats::new();
        }
        if let Ok(mut g) = TIMELINE_GROWTH.lock() {
            *g = TimelineGrowth::new();
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
#[derive(Default)]
struct RunCrossingState {
    /// Node index -> records addressed to that node before it crashed that are
    /// being held for redelivery when it comes back.
    held: HashMap<usize, u64>,
    counted_in_runs_with_crossing: bool,
}

thread_local! {
    static RUN_CROSSING: RefCell<RunCrossingState> = RefCell::new(RunCrossingState::default());
}

/// One plan execution is starting on this thread. Held-message bookkeeping is
/// per-run, so anything left over from a run that ended while a node was still
/// down is discarded here rather than leaking into the next run.
pub fn begin_run() {
    if !enabled() {
        return;
    }
    CR_RUNS.fetch_add(1, Ordering::Relaxed);
    RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        c.held.clear();
        c.counted_in_runs_with_crossing = false;
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
    if held > 0 {
        RUN_CROSSING.with(|c| {
            *c.borrow_mut().held.entry(node_index).or_insert(0) += held;
        });
    }
}

/// A node recovered. Every message held from before its crash is requeued to
/// it at this point, so each one is a delivery that crosses the node's own
/// crash/recover boundary. Messages that arrived while the node was down are
/// requeued too but are not counted: they were sent to an already-dead node.
pub fn record_recover(node_index: usize) {
    if !enabled() {
        return;
    }
    CR_RECOVERS.fetch_add(1, Ordering::Relaxed);
    let first_crossing_of_run = RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
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
    let bucket = s.recovered_nodes.min(2);
    if let Ok(mut t) = TERMINATION.lock() {
        t.all.add(s.end, s);
        t.by_recovered_nodes[bucket].add(s.end, s);
    }
}

/// One point on the timeline-key growth curve, covering `runs` consecutive
/// merges in the order they reached the global store.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TimelineGrowthBucket {
    /// 1-based merge index of this point's first run.
    pub first_run: u64,
    pub runs: u64,
    pub keys_in_run_sum: u64,
    /// Keys these runs contributed that no earlier run had produced.
    pub new_keys: u64,
    /// Distinct keys accumulated from the first run through this point.
    pub cumulative_distinct: u64,
}

#[derive(Debug)]
struct TimelineGrowth {
    runs: u64,
    keys_in_run_sum: u64,
    cumulative_distinct: u64,
    buckets: Vec<TimelineGrowthBucket>,
}

impl TimelineGrowth {
    const fn new() -> Self {
        Self {
            runs: 0,
            keys_in_run_sum: 0,
            cumulative_distinct: 0,
            buckets: Vec::new(),
        }
    }
}

/// One run's timeline keys were merged into the global store: `keys_in_run`
/// distinct keys, of which `new_keys` had never been produced by any earlier
/// run, leaving `distinct_keys_live` keys in the store. The live count differs
/// from the cumulative count when the store is decayed, which drops keys that
/// a later run can then rediscover.
pub fn record_timeline_keys(keys_in_run: u64, new_keys: u64, distinct_keys_live: u64) {
    if !enabled() {
        return;
    }
    TL_DISTINCT_LIVE.fetch_max(distinct_keys_live, Ordering::Relaxed);
    let Ok(mut g) = TIMELINE_GROWTH.lock() else {
        return;
    };
    g.runs += 1;
    g.keys_in_run_sum += keys_in_run;
    g.cumulative_distinct += new_keys;

    let index = g.runs;
    let bucket = ((index - 1) / GROWTH_BUCKET_RUNS) as usize;
    if bucket >= MAX_GROWTH_BUCKETS {
        return;
    }
    if bucket == g.buckets.len() {
        g.buckets.push(TimelineGrowthBucket {
            first_run: index,
            runs: 0,
            keys_in_run_sum: 0,
            new_keys: 0,
            cumulative_distinct: 0,
        });
    }
    let cumulative = g.cumulative_distinct;
    let b = &mut g.buckets[bucket];
    b.runs += 1;
    b.keys_in_run_sum += keys_in_run;
    b.new_keys += new_keys;
    b.cumulative_distinct = cumulative;
}

/// How fast the timeline coverage key space is still growing. A key space that
/// keeps yielding new keys at a constant rate is not saturated, so refining the
/// key would split an already-unexhausted space; one whose growth has flattened
/// is exhausted and refinement has somewhere to go.
#[derive(Debug, Serialize)]
pub struct TimelineKeyStats {
    pub runs: u64,
    /// Sum of per-run distinct key counts; divide by `runs` for the mean.
    pub keys_in_run_sum: u64,
    pub cumulative_distinct_keys: u64,
    /// High-water mark of keys held in the global store at a merge.
    pub distinct_keys_live: u64,
    /// Merge index of the last run of the first full point on the curve whose
    /// growth fell below one new key per run. Zero means it never fell below.
    pub saturation_run_index: u64,
    pub bucket_runs: u64,
    pub growth_curve: Vec<TimelineGrowthBucket>,
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

#[derive(Serialize)]
pub struct SteerStats {
    pub evaluations: u64,
    pub divergent_picks: u64,
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

/// A point-in-time copy of all counters, serializable to `utilization.json`.
#[derive(Serialize)]
pub struct UtilizationSnapshot {
    pub steer: SteerStats,
    pub purgatory: PurgatoryStats,
    pub aos: AosStats,
    pub dedup: DedupStats,
    pub feedback: FeedbackStats,
    pub curriculum: CurriculumStats,
    pub crash_recovery: CrashRecoveryStats,
    pub delivery_effects: DeliveryEffectStats,
    pub crash_anchor: CrashAnchorStats,
    pub termination: TerminationStats,
    pub timeline_keys: TimelineKeyStats,
}

fn timeline_key_stats() -> TimelineKeyStats {
    let g = TIMELINE_GROWTH
        .lock()
        .map(|g| TimelineGrowth {
            runs: g.runs,
            keys_in_run_sum: g.keys_in_run_sum,
            cumulative_distinct: g.cumulative_distinct,
            buckets: g.buckets.clone(),
        })
        .unwrap_or_else(|_| TimelineGrowth::new());
    let saturation_run_index = g
        .buckets
        .iter()
        .find(|b| b.runs == GROWTH_BUCKET_RUNS && b.new_keys < b.runs)
        .map(|b| b.first_run + b.runs - 1)
        .unwrap_or(0);
    TimelineKeyStats {
        runs: g.runs,
        keys_in_run_sum: g.keys_in_run_sum,
        cumulative_distinct_keys: g.cumulative_distinct,
        distinct_keys_live: TL_DISTINCT_LIVE.load(Ordering::Relaxed),
        saturation_run_index,
        bucket_runs: GROWTH_BUCKET_RUNS,
        growth_curve: g.buckets,
    }
}

pub fn snapshot() -> UtilizationSnapshot {
    UtilizationSnapshot {
        steer: SteerStats {
            evaluations: STEER_EVALUATIONS.load(Ordering::Relaxed),
            divergent_picks: STEER_DIVERGENT_PICKS.load(Ordering::Relaxed),
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
        timeline_keys: timeline_key_stats(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are process-wide and enabling resets them, so tests that
    /// enable them cannot run concurrently.
    static COUNTERS: Mutex<()> = Mutex::new(());

    #[test]
    fn delivery_effects_split_by_bias() {
        let _guard = COUNTERS.lock().unwrap_or_else(|p| p.into_inner());
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
    fn timeline_growth_curve_finds_saturation_point() {
        let _guard = COUNTERS.lock().unwrap_or_else(|p| p.into_inner());
        set_enabled(true);

        // First bucket grows at 5 new keys/run, second at 0.5.
        for _ in 0..GROWTH_BUCKET_RUNS {
            record_timeline_keys(20, 5, 100);
        }
        for i in 0..GROWTH_BUCKET_RUNS {
            record_timeline_keys(20, i % 2, 100);
        }
        // A partial third bucket must not be read as saturated.
        record_timeline_keys(20, 0, 100);

        let s = snapshot().timeline_keys;
        set_enabled(false);

        assert_eq!(s.runs, 2 * GROWTH_BUCKET_RUNS + 1);
        assert_eq!(s.keys_in_run_sum, 20 * (2 * GROWTH_BUCKET_RUNS + 1));
        assert_eq!(
            s.cumulative_distinct_keys,
            5 * GROWTH_BUCKET_RUNS + GROWTH_BUCKET_RUNS / 2
        );
        assert_eq!(s.distinct_keys_live, 100);
        assert_eq!(s.growth_curve.len(), 3);
        assert_eq!(s.growth_curve[0].first_run, 1);
        assert_eq!(s.growth_curve[1].first_run, GROWTH_BUCKET_RUNS + 1);
        assert_eq!(s.growth_curve[2].runs, 1);
        assert_eq!(s.saturation_run_index, 2 * GROWTH_BUCKET_RUNS);
    }
}
