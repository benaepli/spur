//! Opt-in, process-wide utilization counters for explorer mechanisms.
//!
//! Enabled per explorer session via `ExplorerConfig::stats` and dumped by the
//! CLI to `<output_dir>/utilization.json`. Counters are observation-only: they
//! never affect scheduling, scoring, or RNG consumption. When disabled, every
//! probe is a single relaxed atomic load.

use serde::Serialize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

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

static TERMINATION: Mutex<TerminationStats> = Mutex::new(TerminationStats::new());

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
        ] {
            c.store(0, Ordering::Relaxed);
        }
        if let Ok(mut t) = TERMINATION.lock() {
            *t = TerminationStats::new();
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

/// A point-in-time copy of all counters, serializable to `utilization.json`.
#[derive(Serialize)]
pub struct UtilizationSnapshot {
    pub steer: SteerStats,
    pub purgatory: PurgatoryStats,
    pub aos: AosStats,
    pub dedup: DedupStats,
    pub feedback: FeedbackStats,
    pub curriculum: CurriculumStats,
    pub termination: TerminationStats,
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
        termination: TERMINATION
            .lock()
            .map(|t| *t)
            .unwrap_or_else(|p| *p.into_inner()),
    }
}
