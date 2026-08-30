//! Opt-in, process-wide utilization counters for explorer mechanisms.
//!
//! Enabled per explorer session via `ExplorerConfig::stats` and dumped by the
//! CLI to `<output_dir>/utilization.json`. Counters are observation-only: they
//! never affect scheduling, scoring, or RNG consumption. When disabled, every
//! probe is a single relaxed atomic load.

use crate::simulator::core::steer_terms::{Term, TERMS};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static ACTED_ENABLED: AtomicBool = AtomicBool::new(true);
static STEER_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
static STEER_AUDIT_ALWAYS: AtomicBool = AtomicBool::new(false);
static MULTIPLIER_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);

static RNG_ISOLATED_RUNS: AtomicU64 = AtomicU64::new(0);
static RNG_SHARED_RUNS: AtomicU64 = AtomicU64::new(0);
static STEER_EVALUATIONS: AtomicU64 = AtomicU64::new(0);
static STEER_DIVERGENT_PICKS: AtomicU64 = AtomicU64::new(0);
static ES_CANDIDATE_MASK: AtomicU64 = AtomicU64::new(0);
static ES_RANKING_PASS: AtomicU64 = AtomicU64::new(0);
static ES_QUEUE_AUDIT: AtomicU64 = AtomicU64::new(0);
static SA_STEPS: AtomicU64 = AtomicU64::new(0);
static SA_STEPS_TOTAL: AtomicU64 = AtomicU64::new(0);
static SA_AUDITED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_EXPRESSED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_HONORED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_CONSULTED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_SOURCE_ABSENT: AtomicU64 = AtomicU64::new(0);
static SA_HONORED: AtomicU64 = AtomicU64::new(0);
static SA_NO_ELIGIBLE: AtomicU64 = AtomicU64::new(0);
static SA_BLOCKED_BY_ORDER: AtomicU64 = AtomicU64::new(0);
static SA_BLOCKED_BY_TIMER_GATE: AtomicU64 = AtomicU64::new(0);
static SA_OTHER_QUEUE: AtomicU64 = AtomicU64::new(0);
static SA_SAMPLER_CHOSE_OTHER: AtomicU64 = AtomicU64::new(0);
static PURGATORY_DELAYED_SENDS: AtomicU64 = AtomicU64::new(0);
static PURGATORY_HOLDS_DOWN_RECEIVER: AtomicU64 = AtomicU64::new(0);
static PURGATORY_HOLDS_UP_RECEIVER: AtomicU64 = AtomicU64::new(0);
static PURGATORY_PASSTHROUGH_DOWN_RECEIVER: AtomicU64 = AtomicU64::new(0);
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
static CR_RECOVERS_WITH_INFLIGHT: AtomicU64 = AtomicU64::new(0);
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
static MA_DECISIONS: AtomicU64 = AtomicU64::new(0);
static MA_CONTESTED_DECISIONS: AtomicU64 = AtomicU64::new(0);
static MA_QUICK_FIRE_OFFERS: AtomicU64 = AtomicU64::new(0);
static MA_QUICK_FIRE_DECISIONS: AtomicU64 = AtomicU64::new(0);
static MA_FLIPPED_CONFIGURED: AtomicU64 = AtomicU64::new(0);
static MA_CONFIGURED_SUM: AtomicU64 = AtomicU64::new(0);

/// Multiplier magnitudes the authority probe ranks candidates under. Index 0 is
/// the identity weighting, which every other entry is compared against, so its
/// own flip count is zero by construction and reads as a self-check.
pub const MULTIPLIER_SWEEP: [f64; 5] = [1.0, 3.0, 10.0, 100.0, 1000.0];

static MA_FLIPPED: [AtomicU64; MULTIPLIER_SWEEP.len()] =
    [const { AtomicU64::new(0) }; MULTIPLIER_SWEEP.len()];

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

/// The same (total, acted) pairs again, split by how far the receiving node
/// had progressed past its own most recent restart when the message reached
/// it, counted in handler entries. A receiver that has not restarted counts
/// from the start of the run, so its distance says where in the run the
/// delivery landed rather than where in a recovery it landed.
const ACCEPT_DIST_BUCKETS: usize = 7;
const ACCEPT_DIST_LABELS: [&str; ACCEPT_DIST_BUCKETS] =
    ["0", "1", "2", "3-4", "5-8", "9-16", "17+"];

/// Delivery populations the distance census is kept for. A message that
/// carries both restarts is counted in both rows.
const ACCEPT_PATH_ALL: usize = 0;
const ACCEPT_PATH_SENDER_RESTARTED: usize = 1;
const ACCEPT_PATH_RECEIVER_RESTARTED: usize = 2;
const ACCEPT_PATHS: usize = 3;

static ACCEPT_DIST: [[AtomicU64; ACCEPT_DIST_BUCKETS]; ACCEPT_PATHS] =
    [const { [const { AtomicU64::new(0) }; ACCEPT_DIST_BUCKETS] }; ACCEPT_PATHS];
static ACCEPT_DIST_ACTED: [[AtomicU64; ACCEPT_DIST_BUCKETS]; ACCEPT_PATHS] =
    [const { [const { AtomicU64::new(0) }; ACCEPT_DIST_BUCKETS] }; ACCEPT_PATHS];
static ACCEPT_DIST_ENABLED: AtomicBool = AtomicBool::new(false);

static TERMINATION: Mutex<TerminationStats> = Mutex::new(TerminationStats::new());

static PREFIX_EXTENSION: Mutex<PrefixExtensionStats> = Mutex::new(PrefixExtensionStats::new());
static PREFIX_EXTENSION_ENABLED: AtomicBool = AtomicBool::new(false);

/// Per-term counters, laid out as `TERM_*` rows by `Term::index`: how many
/// within-queue candidates were scored with the term true (`evaluated`),
/// how many selections had such a candidate at all (`present`) and among
/// more than one eligible candidate (`contested`), how many selections
/// chose one (`won`), how many of those choices differ from what the score
/// without predicate weights would have chosen (`flipped`), and how many
/// chosen candidates were measured for their effect (`measured`) and had
/// one (`acted`).
const TERM_EVALUATED: usize = 0;
const TERM_PRESENT: usize = 1;
const TERM_CONTESTED: usize = 2;
const TERM_WON: usize = 3;
const TERM_FLIPPED: usize = 4;
const TERM_MEASURED: usize = 5;
const TERM_ACTED: usize = 6;
const TERM_COUNTERS: usize = 7;
static TERM: [[AtomicU64; TERM_COUNTERS]; TERMS] =
    [const { [const { AtomicU64::new(0) }; TERM_COUNTERS] }; TERMS];
static TERM_DECISIONS: AtomicU64 = AtomicU64::new(0);
static TERM_AUTHORITY_DRAWS: AtomicU64 = AtomicU64::new(0);
static TERM_AUTHORITY_ROUTED: AtomicU64 = AtomicU64::new(0);

/// Log2 buckets of eligible-candidate and audited-candidate counts per
/// selection: 0, 1, 2, 3-4, 5-8, ... up to 2^14 and above.
const HIST_BUCKETS: usize = 16;
static ELIGIBLE_HIST: [AtomicU64; HIST_BUCKETS] = [const { AtomicU64::new(0) }; HIST_BUCKETS];
static CANDIDATES_HIST: [AtomicU64; HIST_BUCKETS] = [const { AtomicU64::new(0) }; HIST_BUCKETS];

fn hist_bucket(n: usize) -> usize {
    match n {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => ((usize::BITS - (n - 1).leading_zeros()) as usize + 1).min(HIST_BUCKETS - 1),
    }
}

/// Context of one timer firing: the vertex the woken record resumes at (so a
/// spec's timer handlers are told apart without naming them), whether a
/// delivery to the node was pending, the node's incarnation and how many
/// firings at that vertex on that node had changed nothing before this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct TimerKey {
    pub vertex: usize,
    pub inflight: bool,
    /// 0, 1, or 2 for two or more recoveries.
    pub incarnation: u8,
    /// 0 for none, 1 for 1-2, 2 for 3-7, 3 for 8 or more.
    pub inert_streak: u8,
}

impl TimerKey {
    pub fn new(vertex: usize, inflight: bool, incarnation: u32, inert_streak: u32) -> Self {
        Self {
            vertex,
            inflight,
            incarnation: incarnation.min(2) as u8,
            inert_streak: match inert_streak {
                0 => 0,
                1..=2 => 1,
                3..=7 => 2,
                _ => 3,
            },
        }
    }
}

/// Keys are a small product of vertices and buckets; the cap only guards
/// against a spec with an unexpected number of timer resume points.
const TIMER_KEY_CAP: usize = 4096;
static TIMER_EFFECTS: LazyLock<Mutex<HashMap<TimerKey, (u64, u64)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static TIMERS_FIRED: AtomicU64 = AtomicU64::new(0);
static TIMERS_ACTED: AtomicU64 = AtomicU64::new(0);
static TIMERS_INFLIGHT_FIRED: AtomicU64 = AtomicU64::new(0);
static TIMERS_INFLIGHT_ACTED: AtomicU64 = AtomicU64::new(0);

/// One slot per `TimerKey::inert_streak` bucket. The per-key table is a list
/// and readers that difference the snapshot keep only integer leaves, so the
/// same split is also kept as named counters.
const STREAK_BUCKETS: usize = 4;
static TIMER_STREAK_FIRED: [AtomicU64; STREAK_BUCKETS] =
    [const { AtomicU64::new(0) }; STREAK_BUCKETS];
static TIMER_STREAK_ACTED: [AtomicU64; STREAK_BUCKETS] =
    [const { AtomicU64::new(0) }; STREAK_BUCKETS];

static TIMER_STEER_EVALUATED: AtomicU64 = AtomicU64::new(0);
static TIMER_STEER_RAISED: AtomicU64 = AtomicU64::new(0);
static TIMER_STEER_LOWERED: AtomicU64 = AtomicU64::new(0);
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
            &SA_STEPS_TOTAL,
            &SA_AUDITED,
            &SA_PREFERENCE_EXPRESSED,
            &SA_PREFERENCE_HONORED,
            &SA_PREFERENCE_CONSULTED,
            &SA_PREFERENCE_SOURCE_ABSENT,
            &SA_HONORED,
            &SA_NO_ELIGIBLE,
            &SA_BLOCKED_BY_ORDER,
            &SA_BLOCKED_BY_TIMER_GATE,
            &SA_OTHER_QUEUE,
            &SA_SAMPLER_CHOSE_OTHER,
            &PURGATORY_DELAYED_SENDS,
            &PURGATORY_HOLDS_DOWN_RECEIVER,
            &PURGATORY_HOLDS_UP_RECEIVER,
            &PURGATORY_PASSTHROUGH_DOWN_RECEIVER,
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
            &CR_RECOVERS_WITH_INFLIGHT,
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
            &MA_DECISIONS,
            &MA_CONTESTED_DECISIONS,
            &MA_QUICK_FIRE_OFFERS,
            &MA_QUICK_FIRE_DECISIONS,
            &MA_FLIPPED_CONFIGURED,
            &MA_CONFIGURED_SUM,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for c in DELIVERIES
            .iter()
            .chain(DELIVERIES_ACTED.iter())
            .chain(MA_FLIPPED.iter())
            .chain(ACCEPT_DIST.iter().flatten())
            .chain(ACCEPT_DIST_ACTED.iter().flatten())
        {
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
        if let Ok(mut p) = PREFIX_EXTENSION.lock() {
            *p = PrefixExtensionStats::new();
        }
        if let Ok(mut g) = TIMELINE_KEYS.lock() {
            *g = TimelineKeyGrowth::new();
        }
        for c in [
            &TIMERS_FIRED,
            &TIMERS_ACTED,
            &TIMERS_INFLIGHT_FIRED,
            &TIMERS_INFLIGHT_ACTED,
            &TIMER_STEER_EVALUATED,
            &TIMER_STEER_RAISED,
            &TIMER_STEER_LOWERED,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for c in TIMER_STREAK_FIRED.iter().chain(TIMER_STREAK_ACTED.iter()) {
            c.store(0, Ordering::Relaxed);
        }
        if let Ok(mut t) = TIMER_EFFECTS.lock() {
            t.clear();
        }
        for row in TERM.iter() {
            for c in row {
                c.store(0, Ordering::Relaxed);
            }
        }
        for c in [&TERM_DECISIONS, &TERM_AUTHORITY_DRAWS, &TERM_AUTHORITY_ROUTED] {
            c.store(0, Ordering::Relaxed);
        }
        for c in [&ES_CANDIDATE_MASK, &ES_RANKING_PASS, &ES_QUEUE_AUDIT] {
            c.store(0, Ordering::Relaxed);
        }
        for c in ELIGIBLE_HIST.iter().chain(CANDIDATES_HIST.iter()) {
            c.store(0, Ordering::Relaxed);
        }
    }
    ENABLED.store(on, Ordering::Relaxed);
}

/// One within-queue selection was scored with terms. `present` and
/// `chosen` are term masks over the eligible candidates and the chosen one;
/// `evaluated` counts, per term, the sampled candidates the term was true
/// of; `flipped` says whether the choice differs from the one the score
/// without predicate weights would have made.
#[inline]
pub fn record_term_decision(
    eligible: usize,
    present: u8,
    evaluated: &[u64; TERMS],
    chosen: u8,
    flipped: bool,
) {
    if !enabled() {
        return;
    }
    TERM_DECISIONS.fetch_add(1, Ordering::Relaxed);
    ELIGIBLE_HIST[hist_bucket(eligible)].fetch_add(1, Ordering::Relaxed);
    for t in Term::ALL {
        let i = t.index();
        let bit = 1u8 << i;
        let row = &TERM[i];
        row[TERM_EVALUATED].fetch_add(evaluated[i], Ordering::Relaxed);
        if present & bit != 0 {
            row[TERM_PRESENT].fetch_add(1, Ordering::Relaxed);
            if eligible > 1 {
                row[TERM_CONTESTED].fetch_add(1, Ordering::Relaxed);
            }
        }
        if chosen & bit != 0 {
            row[TERM_WON].fetch_add(1, Ordering::Relaxed);
            if flipped {
                row[TERM_FLIPPED].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The candidate chosen with terms `mask` true was measured for its effect.
#[inline]
pub fn record_term_acted(mask: u8, acted: bool) {
    if !enabled() || mask == 0 {
        return;
    }
    for t in Term::ALL {
        if mask & (1u8 << t.index()) != 0 {
            let row = &TERM[t.index()];
            row[TERM_MEASURED].fetch_add(1, Ordering::Relaxed);
            if acted {
                row[TERM_ACTED].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The queue router drew for a predicated candidate; `routed` says whether
/// the draw sent the step to that candidate's queue.
#[inline]
pub fn record_term_authority(routed: bool) {
    if !enabled() {
        return;
    }
    TERM_AUTHORITY_DRAWS.fetch_add(1, Ordering::Relaxed);
    if routed {
        TERM_AUTHORITY_ROUTED.fetch_add(1, Ordering::Relaxed);
    }
}

/// The steer audit ranked `candidates` runnables at one scheduling point.
#[inline]
pub fn record_audit_candidates(candidates: usize) {
    if !enabled() {
        return;
    }
    CANDIDATES_HIST[hist_bucket(candidates)].fetch_add(1, Ordering::Relaxed);
}

/// The counters of one term.
#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermCounters {
    pub evaluated: u64,
    pub present: u64,
    pub contested: u64,
    pub won: u64,
    pub flipped: u64,
    pub measured: u64,
    pub acted: u64,
}

impl TermCounters {
    fn read(i: usize) -> Self {
        let row = &TERM[i];
        let get = |k: usize| row[k].load(Ordering::Relaxed);
        Self {
            evaluated: get(TERM_EVALUATED),
            present: get(TERM_PRESENT),
            contested: get(TERM_CONTESTED),
            won: get(TERM_WON),
            flipped: get(TERM_FLIPPED),
            measured: get(TERM_MEASURED),
            acted: get(TERM_ACTED),
        }
    }
}

/// The score terms' counters for one session.
#[derive(Serialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct SteerTermStats {
    pub decisions: u64,
    pub authority_draws: u64,
    pub authority_routed: u64,
    pub crash_after_timer_sends: TermCounters,
    pub crash_after_delivery_sends: TermCounters,
    pub stale_late: TermCounters,
    pub request_before_stale: TermCounters,
    pub eligible_hist: Vec<u64>,
    pub candidates_hist: Vec<u64>,
}

impl SteerTermStats {
    fn read() -> Self {
        Self {
            decisions: TERM_DECISIONS.load(Ordering::Relaxed),
            authority_draws: TERM_AUTHORITY_DRAWS.load(Ordering::Relaxed),
            authority_routed: TERM_AUTHORITY_ROUTED.load(Ordering::Relaxed),
            crash_after_timer_sends: TermCounters::read(Term::CrashAfterTimerSends.index()),
            crash_after_delivery_sends: TermCounters::read(Term::CrashAfterDeliverySends.index()),
            stale_late: TermCounters::read(Term::StaleLate.index()),
            request_before_stale: TermCounters::read(Term::RequestBeforeStale.index()),
            eligible_hist: ELIGIBLE_HIST.iter().map(|c| c.load(Ordering::Relaxed)).collect(),
            candidates_hist: CANDIDATES_HIST.iter().map(|c| c.load(Ordering::Relaxed)).collect(),
        }
    }
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

/// Enable or disable the acceptance-distance census. It rides the
/// delivery-effect probe, so it also requires `set_acted_fraction_enabled(true)`.
pub fn set_acceptance_distance_enabled(on: bool) {
    ACCEPT_DIST_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a delivery's position relative to its receiver's restart should be
/// recorded.
#[inline]
pub fn acceptance_distance_enabled() -> bool {
    acted_fraction_enabled() && ACCEPT_DIST_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the per-run record of how runs stop extending their
/// schedule. It also requires `set_enabled(true)`.
pub fn set_prefix_extension_enabled(on: bool) {
    PREFIX_EXTENSION_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a finished run should be classified by how it stopped extending.
#[inline]
pub fn prefix_extension_enabled() -> bool {
    enabled() && PREFIX_EXTENSION_ENABLED.load(Ordering::Relaxed)
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

/// Extend the steer-authority audit to sessions where no predicate carries
/// weight, whose ranking is novelty and priority alone. Off leaves those
/// sessions counting only the skip.
pub fn set_steer_audit_always(on: bool) {
    STEER_AUDIT_ALWAYS.store(on, Ordering::Relaxed);
}

/// Whether a scheduling point should be audited even with an unweighted
/// ranking.
#[inline]
pub fn steer_audit_always() -> bool {
    STEER_AUDIT_ALWAYS.load(Ordering::Relaxed)
}

/// Enable or disable the multiplier-authority probe for this session. It also
/// requires `set_enabled(true)`; the probe re-ranks the eligible candidates
/// once per swept magnitude, so it is a separate switch from the cheap counters.
pub fn set_multiplier_audit_enabled(on: bool) {
    MULTIPLIER_AUDIT_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a within-queue selection should be re-ranked under each swept
/// magnitude. Callers must check this before doing the extra ranking.
#[inline]
pub fn multiplier_audit_enabled() -> bool {
    enabled() && MULTIPLIER_AUDIT_ENABLED.load(Ordering::Relaxed)
}

/// One within-queue selection was seen by the multiplier-authority probe.
/// `contested` means more than one candidate was eligible, `quick_fire_present`
/// that at least one of them is a candidate the multiplier applies to. Only a
/// selection that is both can have its ranking changed by any magnitude, so
/// splitting the two says whether a zero flip rate means the weighting lost or
/// means it was never handed a competitor.
#[inline]
pub fn record_multiplier_decision(contested: bool, quick_fire_present: bool) {
    if !multiplier_audit_enabled() {
        return;
    }
    MA_DECISIONS.fetch_add(1, Ordering::Relaxed);
    if contested {
        MA_CONTESTED_DECISIONS.fetch_add(1, Ordering::Relaxed);
    }
    if quick_fire_present {
        MA_QUICK_FIRE_OFFERS.fetch_add(1, Ordering::Relaxed);
        if contested {
            MA_QUICK_FIRE_DECISIONS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The outcome of re-ranking one selection. `flipped[i]` means the top-ranked
/// candidate under `MULTIPLIER_SWEEP[i]` differs from the one the identity
/// weighting ranks first; `configured_flipped` is the same question asked of
/// the magnitude the session is actually running with.
#[inline]
pub fn record_multiplier_flips(
    configured_multiplier: f64,
    flipped: &[bool; MULTIPLIER_SWEEP.len()],
    configured_flipped: bool,
) {
    if !multiplier_audit_enabled() {
        return;
    }
    add_f64(&MA_CONFIGURED_SUM, configured_multiplier);
    if configured_flipped {
        MA_FLIPPED_CONFIGURED.fetch_add(1, Ordering::Relaxed);
    }
    for (counter, &f) in MA_FLIPPED.iter().zip(flipped.iter()) {
        if f {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
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

/// One step of the run's budget was taken. Counted once per budget step
/// whether or not that step went on to select anything, so it is the
/// denominator `steps` is a subset of, and a session that ran steps can never
/// report zero.
#[inline]
pub fn record_steer_step_total() {
    if !steer_audit_enabled() {
        return;
    }
    SA_STEPS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

/// One budget step reached the point where the run's preference is read and
/// the ranking can change what runs. Counted whatever the scoring weights are,
/// so `steps` is the denominator the audited steps are a subset of. Short of
/// `steps_total` by the steps that stop before that point, which is what
/// separates an independent count from an alias of the budget.
#[inline]
pub fn record_steer_step() {
    if !steer_audit_enabled() {
        return;
    }
    SA_STEPS.fetch_add(1, Ordering::Relaxed);
}

/// A scheduling decision asked what the run prefers. Recorded at every site
/// that reads a preference source, before the site decides whether the answer
/// is worth using, so a session that scheduled anything can never report zero
/// consultations. `source_present` is false when no preference source is
/// configured, i.e. no predicate carries weight, which is the reading that
/// separates "the site never ran" from "the site ran and had nothing to say".
///
/// A single scheduling point reaches several such sites - queue routing, the
/// within-queue ranking, and the audit - so this is not a count of steps.
#[inline]
pub fn record_preference_consultation(source_present: bool) {
    if !steer_audit_enabled() {
        return;
    }
    SA_PREFERENCE_CONSULTED.fetch_add(1, Ordering::Relaxed);
    if !source_present {
        SA_PREFERENCE_SOURCE_ABSENT.fetch_add(1, Ordering::Relaxed);
    }
}

/// One scheduling point had its preference resolved. `expressed` means the
/// score ranking put a different runnable on top than priority alone would
/// have, i.e. the steering term changed what "preferred" means at this point;
/// `outcome` says what happened to that preferred runnable.
#[inline]
pub fn record_steer_authority(expressed: bool, outcome: SteerOutcome) {
    if !steer_audit_enabled() {
        return;
    }
    SA_AUDITED.fetch_add(1, Ordering::Relaxed);
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

/// Which distance bucket `distance` handler entries past a restart falls in.
#[inline]
fn acceptance_distance_bucket(distance: u32) -> usize {
    match distance {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        _ => 6,
    }
}

/// A message reached a handler entry on another node. `acted` means the
/// handler changed that node's persistent state rather than falling through a
/// guard, which is the difference between a hazard the protocol saw and one it
/// ignored. Only the handler's first execution segment is observed: a write
/// that happens after the handler blocks on a channel is attributed to nothing.
/// `receiver_distance` is how many handler entries the receiving node had
/// taken since its own most recent restart, not counting this one.
#[inline]
pub fn record_delivery(bias: DeliveryBias, acted: bool, receiver_distance: u32) {
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
    if !ACCEPT_DIST_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let slot = acceptance_distance_bucket(receiver_distance);
    let mut paths = [ACCEPT_PATH_ALL; ACCEPT_PATHS];
    let mut plen = 1;
    for (bit, path) in [
        (DeliveryBias::SENDER_RESTARTED, ACCEPT_PATH_SENDER_RESTARTED),
        (
            DeliveryBias::RECEIVER_RESTARTED,
            ACCEPT_PATH_RECEIVER_RESTARTED,
        ),
    ] {
        if bias.contains(bit) {
            paths[plen] = path;
            plen += 1;
        }
    }
    for &p in &paths[..plen] {
        ACCEPT_DIST[p][slot].fetch_add(1, Ordering::Relaxed);
        if acted {
            ACCEPT_DIST_ACTED[p][slot].fetch_add(1, Ordering::Relaxed);
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

/// A stage of the scoring path that reads the run's state, or ranks a set of
/// runnables, only so the term counters have something to report. With no
/// predicate carrying weight there is nothing for those counters to separate,
/// so each stage is skipped and the skip is counted here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptySliceStage {
    /// The predicate reads over the eligible set of one within-queue selection.
    CandidateMask,
    /// The second ranking of the eligible set that compares the blended argmax
    /// with the priority-only one.
    RankingPass,
    /// The ranking of every runnable in every queue at one scheduling point.
    QueueAudit,
}

/// One occurrence of a stage being skipped. `candidate_mask_skipped` and
/// `ranking_pass_skipped` count within-queue selections, `queue_audit_skipped`
/// counts scheduling points, so a session in which the mechanism applied
/// everywhere has all three at the same magnitude as the corresponding
/// unskipped counters (`steer_terms.decisions`, `steer.evaluations`,
/// `steer_authority.audited`) reach when a predicate does carry weight.
#[inline]
pub fn record_empty_slice_skip(stage: EmptySliceStage) {
    if !enabled() {
        return;
    }
    let counter = match stage {
        EmptySliceStage::CandidateMask => &ES_CANDIDATE_MASK,
        EmptySliceStage::RankingPass => &ES_RANKING_PASS,
        EmptySliceStage::QueueAudit => &ES_QUEUE_AUDIT,
    };
    counter.fetch_add(1, Ordering::Relaxed);
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
/// `receiver_down` splits the held population by whether the destination node
/// was crashed at the moment of the send.
#[inline]
pub fn record_purgatory_delay(receiver_down: bool) {
    if !enabled() {
        return;
    }
    PURGATORY_DELAYED_SENDS.fetch_add(1, Ordering::Relaxed);
    if receiver_down {
        PURGATORY_HOLDS_DOWN_RECEIVER.fetch_add(1, Ordering::Relaxed);
    } else {
        PURGATORY_HOLDS_UP_RECEIVER.fetch_add(1, Ordering::Relaxed);
    }
}

/// A send that purgatory selected for a hold was enqueued undelayed because its
/// destination node was crashed and holds into crashed receivers are disabled.
#[inline]
pub fn record_purgatory_passthrough_down_receiver() {
    if !enabled() {
        return;
    }
    PURGATORY_PASSTHROUGH_DOWN_RECEIVER.fetch_add(1, Ordering::Relaxed);
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
///
/// `own_sends_inflight` means the node still had messages of its own
/// undelivered when it came back, so those messages are now stale-incarnation
/// deliveries waiting to happen.
pub fn record_recover(node_index: usize, step: i32, own_sends_inflight: bool) {
    if !enabled() {
        return;
    }
    CR_RECOVERS.fetch_add(1, Ordering::Relaxed);
    if own_sends_inflight {
        CR_RECOVERS_WITH_INFLIGHT.fetch_add(1, Ordering::Relaxed);
    }
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
    debug_assert!(
        !(steer_audit_enabled() && s.steps_used > 0 && SA_STEPS_TOTAL.load(Ordering::Relaxed) == 0),
        "a run took {} scheduling steps and none was counted; the steer-authority \
         counters are not reaching the scheduler",
        s.steps_used
    );
    finish_run();
    let bucket = s.recovered_nodes.min(2);
    if let Ok(mut t) = TERMINATION.lock() {
        t.all.add(s.end, s);
        t.by_recovered_nodes[bucket].add(s.end, s);
    }
}

/// A budget-ended run whose last steps released nothing was not short of
/// budget, it was short of releasable work. The threshold is long enough that
/// waiting out one delayed message does not read as stopped.
const STALLED_TAIL_STEPS: u64 = 100;

/// How one run stopped extending its schedule. A run extends for as long as the
/// scheduler keeps releasing queued work; it stops when the plan has no event
/// left to complete, when the step budget ends it, or when nothing that is
/// queued can be released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixStop {
    /// The plan finished and nothing was left queued.
    PlanCompleteQuiescent,
    /// The plan finished while protocol work was still queued, so the plan,
    /// not the protocol, is what ended the run.
    PlanCompletePending,
    /// The budget ended a run that was still releasing work.
    BudgetReleasing,
    /// The budget ended a run whose queued work had stopped being releasable.
    BudgetBlocked,
    /// The budget ended a run with nothing queued to release.
    BudgetIdle,
    Deadlock,
}

/// One run's extension facts. `steps_blocked` counts the steps where queued
/// work existed and the scheduler released none of it, which is an extension
/// offered at the frontier and refused; `steps_idle` counts the steps that had
/// nothing queued to offer.
pub struct RunExtension {
    pub end: RunEnd,
    pub steps: u64,
    pub steps_released: u64,
    pub steps_blocked: u64,
    pub steps_idle: u64,
    /// Steps at the end of the run, up to termination, that released nothing.
    pub tail_without_release: u64,
    /// Runnables still queued (including delayed sends) when the run stopped.
    pub pending_at_exit: u64,
    pub recovered_nodes: usize,
}

impl RunExtension {
    fn stop(&self) -> PrefixStop {
        match self.end {
            RunEnd::Deadlock => PrefixStop::Deadlock,
            RunEnd::PlanComplete if self.pending_at_exit == 0 => PrefixStop::PlanCompleteQuiescent,
            RunEnd::PlanComplete => PrefixStop::PlanCompletePending,
            RunEnd::IterationsExhausted => {
                if self.pending_at_exit == 0 {
                    PrefixStop::BudgetIdle
                } else if self.tail_without_release >= STALLED_TAIL_STEPS {
                    PrefixStop::BudgetBlocked
                } else {
                    PrefixStop::BudgetReleasing
                }
            }
        }
    }
}

/// Stop counts and running sums over one bucket of runs. The sums are summed
/// rather than averaged so buckets can be merged; divide by `runs` to read a
/// mean.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PrefixExtensionTally {
    pub runs: u64,
    pub plan_complete_quiescent: u64,
    pub plan_complete_pending: u64,
    pub budget_releasing: u64,
    pub budget_blocked: u64,
    pub budget_idle: u64,
    pub deadlock: u64,
    pub steps_sum: u64,
    pub steps_released_sum: u64,
    pub steps_blocked_sum: u64,
    pub steps_idle_sum: u64,
    pub tail_without_release_sum: u64,
    pub pending_at_exit_sum: u64,
}

impl PrefixExtensionTally {
    const fn new() -> Self {
        Self {
            runs: 0,
            plan_complete_quiescent: 0,
            plan_complete_pending: 0,
            budget_releasing: 0,
            budget_blocked: 0,
            budget_idle: 0,
            deadlock: 0,
            steps_sum: 0,
            steps_released_sum: 0,
            steps_blocked_sum: 0,
            steps_idle_sum: 0,
            tail_without_release_sum: 0,
            pending_at_exit_sum: 0,
        }
    }

    fn add(&mut self, x: &RunExtension) {
        self.runs += 1;
        match x.stop() {
            PrefixStop::PlanCompleteQuiescent => self.plan_complete_quiescent += 1,
            PrefixStop::PlanCompletePending => self.plan_complete_pending += 1,
            PrefixStop::BudgetReleasing => self.budget_releasing += 1,
            PrefixStop::BudgetBlocked => self.budget_blocked += 1,
            PrefixStop::BudgetIdle => self.budget_idle += 1,
            PrefixStop::Deadlock => self.deadlock += 1,
        }
        self.steps_sum += x.steps;
        self.steps_released_sum += x.steps_released;
        self.steps_blocked_sum += x.steps_blocked;
        self.steps_idle_sum += x.steps_idle;
        self.tail_without_release_sum += x.tail_without_release;
        self.pending_at_exit_sum += x.pending_at_exit;
    }
}

/// Extension tallies over all runs and split by how many distinct nodes
/// completed a crash-and-recover cycle (index 0, 1, and 2-or-more), so the
/// runs that carry the deepest fault interleavings can be read apart from the
/// shallow ones that dominate the total.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PrefixExtensionStats {
    pub all: PrefixExtensionTally,
    pub by_recovered_nodes: [PrefixExtensionTally; 3],
}

impl PrefixExtensionStats {
    const fn new() -> Self {
        Self {
            all: PrefixExtensionTally::new(),
            by_recovered_nodes: [PrefixExtensionTally::new(); 3],
        }
    }
}

/// One plan execution finished. Called once per run, off the scheduling hot
/// path.
pub fn record_run_extension(x: &RunExtension) {
    if !prefix_extension_enabled() {
        return;
    }
    let bucket = x.recovered_nodes.min(2);
    if let Ok(mut p) = PREFIX_EXTENSION.lock() {
        p.all.add(x);
        p.by_recovered_nodes[bucket].add(x);
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

/// How often each stage of the scoring path was skipped because no predicate
/// carried weight. All zero means every stage ran, which is what a session with
/// a nonzero weight looks like.
#[derive(Serialize, Debug)]
pub struct EmptySliceStats {
    pub candidate_mask_skipped: u64,
    pub ranking_pass_skipped: u64,
    pub queue_audit_skipped: u64,
}

/// How often the runnable the scoring function ranked first is the one the
/// scheduling point ran, and what took precedence when it was not. The buckets
/// after `honored` partition the audited steps by the single constraint that
/// stood in the way, so all six sum to `audited`.
#[derive(Serialize)]
pub struct SteerAuthorityStats {
    /// Every budget step the session took, whether or not it went on to
    /// select anything. Zero here with steps used in the session means the
    /// counters are not wired up.
    pub steps_total: u64,
    /// The subset of `steps_total` that reached the point where the run's
    /// preference is read, audited or not. Equal to `steps_total` means no
    /// step stops before that point, so the decision site is on every step.
    pub steps: u64,
    /// The subset of `steps` where the ranking was resolved against what the
    /// step ran. Short of `steps` by the points the audit skipped, which
    /// `steer_empty_slice.queue_audit_skipped` counts.
    pub audited: u64,
    /// Steps where the steering term put a different runnable on top than
    /// priority alone would have. The denominator for `preference_honored`:
    /// on the other steps the audit cannot tell steer's choice from the
    /// choice the scheduler would have made without it.
    pub preference_expressed: u64,
    pub preference_honored: u64,
    /// Every read of a preference source, counted before the reader decides
    /// what to do with the answer. Several per step, so it is not comparable
    /// with `steps`; it is the denominator that says whether the decision
    /// sites execute at all, which `preference_expressed` alone cannot.
    pub preference_consulted: u64,
    /// The subset of `preference_consulted` where nothing was configured to
    /// have a preference. Equal to `preference_consulted` means the sites all
    /// ran and every one of them had no source to read.
    pub preference_source_absent: u64,
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
    /// Held sends whose destination node was crashed when the send was made.
    pub holds_down_receiver: u64,
    /// Held sends whose destination node was running when the send was made.
    pub holds_up_receiver: u64,
    /// Sends selected for a hold into a crashed destination and let through.
    pub passthrough_down_receiver: u64,
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

/// Record one timer firing that woke a waiting record and whether the woken
/// segment changed the node's state. Same switch as the delivery probe.
#[inline]
pub fn record_timer(key: TimerKey, acted: bool) {
    if !acted_fraction_enabled() {
        return;
    }
    TIMERS_FIRED.fetch_add(1, Ordering::Relaxed);
    if acted {
        TIMERS_ACTED.fetch_add(1, Ordering::Relaxed);
    }
    if key.inflight {
        TIMERS_INFLIGHT_FIRED.fetch_add(1, Ordering::Relaxed);
        if acted {
            TIMERS_INFLIGHT_ACTED.fetch_add(1, Ordering::Relaxed);
        }
    }
    let bucket = usize::from(key.inert_streak).min(STREAK_BUCKETS - 1);
    TIMER_STREAK_FIRED[bucket].fetch_add(1, Ordering::Relaxed);
    if acted {
        TIMER_STREAK_ACTED[bucket].fetch_add(1, Ordering::Relaxed);
    }
    if let Ok(mut t) = TIMER_EFFECTS.lock() {
        if let Some(e) = t.get_mut(&key) {
            e.0 += 1;
            if acted {
                e.1 += 1;
            }
        } else if t.len() < TIMER_KEY_CAP {
            t.insert(key, (1, u64::from(acted)));
        }
    }
}

/// Timer firings and the share that changed the node's state, for one slice.
#[derive(Serialize)]
pub struct TimerEffect {
    pub fired: u64,
    pub acted: u64,
    pub acted_fraction: f64,
}

impl TimerEffect {
    fn of(fired: u64, acted: u64) -> Self {
        Self {
            fired,
            acted,
            acted_fraction: if fired == 0 { 0.0 } else { acted as f64 / fired as f64 },
        }
    }
}

/// One context key of the timer effect table with its counts.
#[derive(Serialize)]
pub struct TimerKeyEffect {
    pub vertex: usize,
    pub inflight: bool,
    pub incarnation: u8,
    pub inert_streak: u8,
    pub fired: u64,
    pub acted: u64,
}

/// Timer firings in the session: overall, with a delivery to the node in
/// flight, on an idle node, and per context key. Only firings that woke a
/// waiting record are counted; a firing whose channel had no reader is
/// consumed later by a receive and is not attributed.
#[derive(Serialize)]
pub struct TimerEffectStats {
    pub all: TimerEffect,
    pub with_inflight: TimerEffect,
    pub idle: TimerEffect,
    pub inert_streak: InertStreakHistogram,
    pub by_key: Vec<TimerKeyEffect>,
}

/// Timer firings grouped by how many firings at the same resume point on the
/// same node had changed nothing before this one: none, one or two, three to
/// seven, eight or more.
#[derive(Serialize)]
pub struct InertStreakHistogram {
    pub none: TimerEffect,
    pub short: TimerEffect,
    pub medium: TimerEffect,
    pub long: TimerEffect,
}

impl InertStreakHistogram {
    fn read() -> Self {
        let b = |i: usize| {
            TimerEffect::of(
                TIMER_STREAK_FIRED[i].load(Ordering::Relaxed),
                TIMER_STREAK_ACTED[i].load(Ordering::Relaxed),
            )
        };
        Self {
            none: b(0),
            short: b(1),
            medium: b(2),
            long: b(3),
        }
    }
}

/// Steps where admitting a timer was an actual choice, i.e. a timer and a
/// message delivery were both schedulable, and which of the two the step ran.
/// A mechanism that reweights timers against deliveries moves `raised` and
/// `lowered` against this denominator; with none configured the split is
/// whatever the queue selector draws.
#[derive(Serialize)]
pub struct TimerSteerStats {
    pub evaluated: u64,
    pub raised: u64,
    pub lowered: u64,
}

impl TimerSteerStats {
    fn read() -> Self {
        Self {
            evaluated: TIMER_STEER_EVALUATED.load(Ordering::Relaxed),
            raised: TIMER_STEER_RAISED.load(Ordering::Relaxed),
            lowered: TIMER_STEER_LOWERED.load(Ordering::Relaxed),
        }
    }
}

/// Record one step at which a timer and a delivery were both schedulable.
/// `chose_timer` says which one the step ran.
#[inline]
pub fn record_timer_admission(chose_timer: bool) {
    if !enabled() {
        return;
    }
    TIMER_STEER_EVALUATED.fetch_add(1, Ordering::Relaxed);
    if chose_timer {
        TIMER_STEER_RAISED.fetch_add(1, Ordering::Relaxed);
    } else {
        TIMER_STEER_LOWERED.fetch_add(1, Ordering::Relaxed);
    }
}

impl TimerEffectStats {
    fn read() -> Self {
        let fired = TIMERS_FIRED.load(Ordering::Relaxed);
        let acted = TIMERS_ACTED.load(Ordering::Relaxed);
        let inflight = TIMERS_INFLIGHT_FIRED.load(Ordering::Relaxed);
        let inflight_acted = TIMERS_INFLIGHT_ACTED.load(Ordering::Relaxed);
        let mut by_key: Vec<TimerKeyEffect> = TIMER_EFFECTS
            .lock()
            .map(|t| {
                t.iter()
                    .map(|(k, (f, a))| TimerKeyEffect {
                        vertex: k.vertex,
                        inflight: k.inflight,
                        incarnation: k.incarnation,
                        inert_streak: k.inert_streak,
                        fired: *f,
                        acted: *a,
                    })
                    .collect()
            })
            .unwrap_or_default();
        by_key.sort_by_key(|e| (e.vertex, e.inflight, e.incarnation, e.inert_streak));
        Self {
            all: TimerEffect::of(fired, acted),
            with_inflight: TimerEffect::of(inflight, inflight_acted),
            idle: TimerEffect::of(fired - inflight, acted - inflight_acted),
            inert_streak: InertStreakHistogram::read(),
            by_key,
        }
    }
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

/// Deliveries and the share of them that changed the receiving node's state,
/// for one distance-from-restart bucket.
#[derive(Serialize)]
pub struct AcceptanceDistanceBucket {
    pub distance: &'static str,
    pub deliveries: u64,
    pub acted: u64,
    pub acted_fraction: f64,
}

/// The acted fraction as a function of how far the receiver had moved past its
/// own restart. A curve that falls off with distance means arrival position
/// decides whether a stale message is taken or absorbed; a flat curve means it
/// does not.
#[derive(Serialize)]
pub struct AcceptanceDistanceStats {
    pub all: Vec<AcceptanceDistanceBucket>,
    pub sender_restarted: Vec<AcceptanceDistanceBucket>,
    pub receiver_restarted: Vec<AcceptanceDistanceBucket>,
}

impl AcceptanceDistanceStats {
    fn read_path(path: usize) -> Vec<AcceptanceDistanceBucket> {
        (0..ACCEPT_DIST_BUCKETS)
            .map(|b| {
                let deliveries = ACCEPT_DIST[path][b].load(Ordering::Relaxed);
                let acted = ACCEPT_DIST_ACTED[path][b].load(Ordering::Relaxed);
                AcceptanceDistanceBucket {
                    distance: ACCEPT_DIST_LABELS[b],
                    deliveries,
                    acted,
                    acted_fraction: if deliveries == 0 {
                        0.0
                    } else {
                        acted as f64 / deliveries as f64
                    },
                }
            })
            .collect()
    }

    fn read() -> Self {
        Self {
            all: Self::read_path(ACCEPT_PATH_ALL),
            sender_restarted: Self::read_path(ACCEPT_PATH_SENDER_RESTARTED),
            receiver_restarted: Self::read_path(ACCEPT_PATH_RECEIVER_RESTARTED),
        }
    }
}

/// How often a message that reached a handler actually changed the receiver's
/// state, split by which perturbation the message was carrying. A bias whose
/// `acted_fraction` is near zero is being delivered but ignored, which is a
/// different failure than one whose `deliveries` is near zero.
///
/// The five flat fault tallies are the base rates of the crash-and-recover
/// predicates the delivery buckets are read against: how often a crash lands on
/// a node that still has a message of its own undelivered, how often a node
/// comes back with such a message still outstanding, and how many deliveries
/// then arrive from an incarnation that no longer exists. They are carried here
/// rather than beside the other fault counters so that a reader holding only
/// this block can compute them.
#[derive(Serialize)]
pub struct DeliveryEffectStats {
    pub all: DeliveryEffect,
    pub biased: DeliveryEffect,
    pub delayed: DeliveryEffect,
    pub sender_restarted: DeliveryEffect,
    pub receiver_restarted: DeliveryEffect,
    pub acceptance_distance: AcceptanceDistanceStats,
    pub crashes_total: u64,
    pub crashes_with_own_sends_inflight: u64,
    pub recoveries_total: u64,
    pub recoveries_with_own_prior_sends_inflight: u64,
    pub stale_sender_deliveries_after_recovery: u64,
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

/// Whether raising the weight of the priority term can outvote the random draw
/// priority is sampled from. `decisions` is every within-queue selection the
/// probe saw, `contested_decisions` those with more than one eligible
/// candidate, `quick_fire_offers` those holding a candidate the multiplier
/// applies to, and `quick_fire_decisions` the intersection, which is the only
/// place any magnitude can change the ranking and the denominator for every
/// flip count. `flipped_configured` is the probe's own firing count: the
/// selections the session's own magnitude reordered.
#[derive(Serialize)]
pub struct MultiplierAuthorityStats {
    pub decisions: u64,
    pub contested_decisions: u64,
    pub quick_fire_offers: u64,
    pub quick_fire_decisions: u64,
    pub mean_configured_multiplier: f64,
    pub flipped_configured: u64,
    pub sweep: Vec<MultiplierFlip>,
}

/// Selections whose top-ranked candidate under `multiplier` differs from the
/// one the identity weighting ranks first, out of `quick_fire_decisions`.
#[derive(Serialize)]
pub struct MultiplierFlip {
    pub multiplier: f64,
    pub flipped: u64,
}

impl MultiplierAuthorityStats {
    fn read() -> Self {
        let quick_fire_decisions = MA_QUICK_FIRE_DECISIONS.load(Ordering::Relaxed);
        Self {
            decisions: MA_DECISIONS.load(Ordering::Relaxed),
            contested_decisions: MA_CONTESTED_DECISIONS.load(Ordering::Relaxed),
            quick_fire_offers: MA_QUICK_FIRE_OFFERS.load(Ordering::Relaxed),
            quick_fire_decisions,
            mean_configured_multiplier: if quick_fire_decisions == 0 {
                0.0
            } else {
                f64::from_bits(MA_CONFIGURED_SUM.load(Ordering::Relaxed))
                    / quick_fire_decisions as f64
            },
            flipped_configured: MA_FLIPPED_CONFIGURED.load(Ordering::Relaxed),
            sweep: MULTIPLIER_SWEEP
                .iter()
                .zip(MA_FLIPPED.iter())
                .map(|(&multiplier, flipped)| MultiplierFlip {
                    multiplier,
                    flipped: flipped.load(Ordering::Relaxed),
                })
                .collect(),
        }
    }
}

/// A point-in-time copy of all counters, serializable to `utilization.json`.
#[derive(Serialize)]
pub struct UtilizationSnapshot {
    pub rng_streams: RngStreamStats,
    pub steer: SteerStats,
    pub steer_empty_slice: EmptySliceStats,
    pub steer_authority: SteerAuthorityStats,
    pub multiplier_authority: MultiplierAuthorityStats,
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
    pub timer_effects: TimerEffectStats,
    pub timer_steer: TimerSteerStats,
    pub crash_anchor: CrashAnchorStats,
    pub termination: TerminationStats,
    pub prefix_extension: PrefixExtensionStats,
    pub timeline_keys: TimelineKeyStats,
    pub steer_terms: SteerTermStats,
}

/// The snapshot as JSON, for readers that difference or accumulate it.
pub fn snapshot_value() -> serde_json::Value {
    serde_json::to_value(snapshot()).unwrap_or(serde_json::Value::Null)
}

/// `after - before` on every integer leaf, keeping the object structure.
/// Floats, arrays and strings are ratios, curves and labels, not counts, so
/// they are dropped; a reader recomputes ratios from the integer leaves.
pub fn delta(before: &serde_json::Value, after: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match (before, after) {
        (Value::Object(b), Value::Object(a)) => {
            let mut out = serde_json::Map::new();
            for (k, av) in a {
                let bv = b.get(k).unwrap_or(&Value::Null);
                match delta(bv, av) {
                    Value::Null => {}
                    v => {
                        out.insert(k.clone(), v);
                    }
                }
            }
            Value::Object(out)
        }
        (_, Value::Number(a)) if a.is_u64() || a.is_i64() => {
            let av = a.as_i64().unwrap_or_else(|| a.as_u64().unwrap_or(0) as i64);
            let bv = match before {
                Value::Number(b) => b.as_i64().unwrap_or_else(|| b.as_u64().unwrap_or(0) as i64),
                _ => 0,
            };
            Value::from(av - bv)
        }
        _ => Value::Null,
    }
}

/// Adds every integer leaf of `delta` into `acc`, creating what is missing.
pub fn add(acc: &mut serde_json::Value, delta: &serde_json::Value) {
    use serde_json::Value;
    if !acc.is_object() {
        *acc = Value::Object(serde_json::Map::new());
    }
    let Value::Object(d) = delta else { return };
    let acc_map = acc.as_object_mut().expect("made an object above");
    for (k, dv) in d {
        match dv {
            Value::Object(_) => {
                let slot = acc_map.entry(k.clone()).or_insert_with(|| Value::Object(serde_json::Map::new()));
                add(slot, dv);
            }
            Value::Number(n) => {
                let cur = acc_map.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
                acc_map.insert(k.clone(), Value::from(cur + n.as_i64().unwrap_or(0)));
            }
            _ => {}
        }
    }
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
        steer_empty_slice: EmptySliceStats {
            candidate_mask_skipped: ES_CANDIDATE_MASK.load(Ordering::Relaxed),
            ranking_pass_skipped: ES_RANKING_PASS.load(Ordering::Relaxed),
            queue_audit_skipped: ES_QUEUE_AUDIT.load(Ordering::Relaxed),
        },
        steer_authority: SteerAuthorityStats {
            steps_total: SA_STEPS_TOTAL.load(Ordering::Relaxed),
            steps: SA_STEPS.load(Ordering::Relaxed),
            audited: SA_AUDITED.load(Ordering::Relaxed),
            preference_expressed: SA_PREFERENCE_EXPRESSED.load(Ordering::Relaxed),
            preference_honored: SA_PREFERENCE_HONORED.load(Ordering::Relaxed),
            preference_consulted: SA_PREFERENCE_CONSULTED.load(Ordering::Relaxed),
            preference_source_absent: SA_PREFERENCE_SOURCE_ABSENT.load(Ordering::Relaxed),
            honored: SA_HONORED.load(Ordering::Relaxed),
            no_eligible_candidates: SA_NO_ELIGIBLE.load(Ordering::Relaxed),
            blocked_by_order: SA_BLOCKED_BY_ORDER.load(Ordering::Relaxed),
            blocked_by_timer_gate: SA_BLOCKED_BY_TIMER_GATE.load(Ordering::Relaxed),
            other_queue: SA_OTHER_QUEUE.load(Ordering::Relaxed),
            sampler_chose_other: SA_SAMPLER_CHOSE_OTHER.load(Ordering::Relaxed),
        },
        multiplier_authority: MultiplierAuthorityStats::read(),
        purgatory: PurgatoryStats {
            delayed_sends: PURGATORY_DELAYED_SENDS.load(Ordering::Relaxed),
            holds_down_receiver: PURGATORY_HOLDS_DOWN_RECEIVER.load(Ordering::Relaxed),
            holds_up_receiver: PURGATORY_HOLDS_UP_RECEIVER.load(Ordering::Relaxed),
            passthrough_down_receiver: PURGATORY_PASSTHROUGH_DOWN_RECEIVER
                .load(Ordering::Relaxed),
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
            acceptance_distance: AcceptanceDistanceStats::read(),
            crashes_total: CA_CRASHES_TAKEN.load(Ordering::Relaxed),
            crashes_with_own_sends_inflight: CA_APPLIED.load(Ordering::Relaxed),
            recoveries_total: CR_RECOVERS.load(Ordering::Relaxed),
            recoveries_with_own_prior_sends_inflight: CR_RECOVERS_WITH_INFLIGHT
                .load(Ordering::Relaxed),
            stale_sender_deliveries_after_recovery: DELIVERIES[DELIVERY_SENDER_RESTARTED]
                .load(Ordering::Relaxed),
        },
        timer_effects: TimerEffectStats::read(),
        timer_steer: TimerSteerStats::read(),
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
        prefix_extension: PREFIX_EXTENSION
            .lock()
            .map(|p| *p)
            .unwrap_or_else(|p| *p.into_inner()),
        timeline_keys: TimelineKeyStats::read(),
        steer_terms: SteerTermStats::read(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    #[test]
    fn term_counters_reset_and_snapshot() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        let stale = 1u8 << Term::StaleLate.index();
        let crash = 1u8 << Term::CrashAfterTimerSends.index();
        record_term_decision(3, stale | crash, &[0, 0, 2, 0], stale, true);
        record_term_decision(1, crash, &[1, 0, 0, 0], 0, false);
        record_term_acted(stale, true);
        record_term_acted(stale, false);
        record_term_authority(true);
        record_term_authority(false);
        record_audit_candidates(7);
        let s = snapshot().steer_terms;
        assert_eq!(s.decisions, 2);
        assert_eq!(s.stale_late.evaluated, 2);
        assert_eq!(s.stale_late.present, 1);
        assert_eq!(s.stale_late.contested, 1);
        assert_eq!(s.stale_late.won, 1);
        assert_eq!(s.stale_late.flipped, 1);
        assert_eq!(s.stale_late.measured, 2);
        assert_eq!(s.stale_late.acted, 1);
        assert_eq!(s.crash_after_timer_sends.present, 2);
        assert_eq!(s.crash_after_timer_sends.contested, 1);
        assert_eq!(s.crash_after_timer_sends.won, 0);
        assert_eq!(s.authority_draws, 2);
        assert_eq!(s.authority_routed, 1);
        assert_eq!(s.eligible_hist[hist_bucket(3)], 1);
        assert_eq!(s.candidates_hist[hist_bucket(7)], 1);
        set_enabled(true);
        let z = snapshot().steer_terms;
        assert_eq!(z, SteerTermStats { eligible_hist: vec![0; HIST_BUCKETS], candidates_hist: vec![0; HIST_BUCKETS], ..SteerTermStats::default() });
        set_enabled(false);
    }

    #[test]
    fn hist_buckets_are_log2_with_small_counts_exact() {
        assert_eq!(hist_bucket(0), 0);
        assert_eq!(hist_bucket(1), 1);
        assert_eq!(hist_bucket(2), 2);
        assert_eq!(hist_bucket(3), 3);
        assert_eq!(hist_bucket(4), 3);
        assert_eq!(hist_bucket(5), 4);
        assert_eq!(hist_bucket(8), 4);
        assert_eq!(hist_bucket(9), 5);
        assert_eq!(hist_bucket(1 << 20), HIST_BUCKETS - 1);
    }

    #[test]
    fn delivery_effects_split_by_bias() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);

        record_delivery(DeliveryBias::NONE, true, 0);
        record_delivery(DeliveryBias::DELAYED, false, 0);
        let mut both = DeliveryBias::DELAYED;
        both.insert(DeliveryBias::SENDER_RESTARTED);
        record_delivery(both, true, 0);

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
        assert!(
            s.acceptance_distance.all.iter().all(|b| b.deliveries == 0),
            "census stays empty while its own switch is off"
        );
    }

    #[test]
    fn fault_tallies_ride_along_with_the_delivery_effects() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);

        begin_run();
        record_crash_anchor_apply(true);
        record_crash(0, 0, 0);
        record_recover(0, 4, true);
        record_crash_anchor_apply(false);
        record_crash(1, 0, 0);
        record_recover(1, 9, false);
        record_delivery(DeliveryBias::SENDER_RESTARTED, true, 1);
        record_delivery(DeliveryBias::NONE, true, 1);

        let s = snapshot().delivery_effects;
        set_enabled(false);

        assert_eq!(s.crashes_total, 2);
        assert_eq!(s.crashes_with_own_sends_inflight, 1);
        assert_eq!(s.recoveries_total, 2);
        assert_eq!(s.recoveries_with_own_prior_sends_inflight, 1);
        assert_eq!(s.stale_sender_deliveries_after_recovery, 1);
        assert_eq!(
            s.stale_sender_deliveries_after_recovery,
            s.sender_restarted.deliveries
        );
    }

    #[test]
    fn acceptance_distance_splits_by_receiver_progress() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);
        set_acceptance_distance_enabled(true);

        record_delivery(DeliveryBias::SENDER_RESTARTED, true, 0);
        record_delivery(DeliveryBias::SENDER_RESTARTED, false, 20);
        record_delivery(DeliveryBias::RECEIVER_RESTARTED, false, 3);
        record_delivery(DeliveryBias::NONE, true, 6);

        let s = snapshot().delivery_effects.acceptance_distance;
        set_acceptance_distance_enabled(false);
        set_enabled(false);

        assert_eq!(s.sender_restarted[0].distance, "0");
        assert_eq!(s.sender_restarted[0].deliveries, 1);
        assert_eq!(s.sender_restarted[0].acted, 1);
        assert_eq!(s.sender_restarted[6].distance, "17+");
        assert_eq!(s.sender_restarted[6].deliveries, 1);
        assert_eq!(s.sender_restarted[6].acted, 0);
        assert_eq!(s.receiver_restarted[3].distance, "3-4");
        assert_eq!(s.receiver_restarted[3].deliveries, 1);
        assert_eq!(s.all[4].distance, "5-8");
        assert_eq!(s.all[4].deliveries, 1);
        assert_eq!(s.all.iter().map(|b| b.deliveries).sum::<u64>(), 4);
    }

    #[test]
    fn timer_effects_split_by_context_and_reset_the_streak_when_acted() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);

        record_timer(TimerKey::new(7, false, 0, 0), false);
        record_timer(TimerKey::new(7, false, 0, 1), false);
        record_timer(TimerKey::new(7, false, 0, 2), true);
        record_timer(TimerKey::new(7, true, 1, 0), false);
        record_timer(TimerKey::new(9, true, 3, 12), true);

        let s = snapshot().timer_effects;
        set_enabled(false);

        assert_eq!(s.all.fired, 5);
        assert_eq!(s.all.acted, 2);
        assert_eq!(s.with_inflight.fired, 2);
        assert_eq!(s.with_inflight.acted, 1);
        assert_eq!(s.idle.fired, 3);
        assert_eq!(s.idle.acted, 1);
        assert_eq!(s.by_key.len(), 4, "streaks 1 and 2 share a bucket");
        let short = s.by_key.iter().find(|e| e.vertex == 7 && e.inert_streak == 1).expect("streak bucket 1 keyed");
        assert_eq!((short.fired, short.acted), (2, 1));
        let deep = s.by_key.iter().find(|e| e.vertex == 9).expect("vertex 9 keyed");
        assert_eq!((deep.incarnation, deep.inert_streak), (2, 3));
        assert!(s.by_key.windows(2).all(|w| (w[0].vertex, w[0].inflight) <= (w[1].vertex, w[1].inflight)));

        let h = &s.inert_streak;
        assert_eq!((h.none.fired, h.none.acted), (2, 0));
        assert_eq!((h.short.fired, h.short.acted), (2, 1));
        assert_eq!((h.medium.fired, h.medium.acted), (0, 0));
        assert_eq!((h.long.fired, h.long.acted), (1, 1));
        assert_eq!(
            h.none.fired + h.short.fired + h.medium.fired + h.long.fired,
            s.all.fired
        );

        set_enabled(true);
        assert_eq!(snapshot().timer_effects.all.fired, 0, "enabling resets the table");
        assert_eq!(snapshot().timer_effects.inert_streak.short.fired, 0);
        set_enabled(false);
    }

    #[test]
    fn timer_admission_splits_contested_steps() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        record_timer_admission(true);
        record_timer_admission(false);
        record_timer_admission(false);

        let s = snapshot().timer_steer;
        set_enabled(false);

        assert_eq!(s.evaluated, 3);
        assert_eq!(s.raised, 1);
        assert_eq!(s.lowered, 2);
        assert_eq!(s.raised + s.lowered, s.evaluated);

        record_timer_admission(true);
        assert_eq!(snapshot().timer_steer.evaluated, 3, "records nothing when off");
    }

    #[test]
    fn a_timer_streak_counts_inert_firings_and_resets_on_an_effect() {
        use crate::analysis::resolver::NameId;
        use crate::simulator::core::state::State;
        use crate::simulator::hash_utils::NoHashing;
        let mut st = State::<NoHashing>::new(&[(NameId(0), 1)], 4);
        st.note_timer_effect(0, 5, false, false);
        st.note_timer_effect(0, 5, false, false);
        assert_eq!(st.timer_inert_streak(0, 5), 2);
        st.note_timer_effect(0, 5, true, true);
        assert_eq!(st.timer_inert_streak(0, 5), 0);
        assert_eq!(st.timer_stats.max_inert_streak, 2);
        assert_eq!((st.timer_stats.fired, st.timer_stats.acted), (3, 1));
        assert_eq!((st.timer_stats.idle_fired, st.timer_stats.inflight_fired), (2, 1));
    }

    #[test]
    fn steer_authority_outcomes_partition_the_audited_steps() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(true);

        for _ in 0..12 {
            record_steer_step_total();
        }
        for _ in 0..9 {
            record_steer_step();
        }
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

        assert_eq!((s.steps_total, s.steps), (12, 9));
        assert_eq!(s.audited, 7);
        assert_eq!(s.preference_expressed, 3);
        assert_eq!(s.preference_honored, 1);
        assert_eq!(
            s.honored
                + s.no_eligible_candidates
                + s.blocked_by_order
                + s.blocked_by_timer_gate
                + s.other_queue
                + s.sampler_chose_other,
            s.audited
        );
    }

    #[test]
    fn steer_authority_records_nothing_when_the_audit_is_off() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(false);

        record_steer_step_total();
        record_steer_step();
        record_steer_authority(true, SteerOutcome::Honored);
        record_preference_consultation(true);

        let s = snapshot().steer_authority;
        set_enabled(false);

        assert_eq!(
            (s.steps_total, s.steps, s.audited, s.preference_consulted),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn preference_consultations_split_into_sourced_and_sourceless_reads() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(true);

        record_preference_consultation(false);
        record_preference_consultation(false);
        record_preference_consultation(true);

        let s = snapshot().steer_authority;
        set_steer_audit_enabled(false);
        set_enabled(false);

        assert_eq!(s.preference_consulted, 3);
        assert_eq!(s.preference_source_absent, 2);
    }

    #[test]
    fn runs_are_split_by_what_stopped_them_from_scheduling_further_work() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_prefix_extension_enabled(true);

        let run = |end, tail_without_release, pending_at_exit, recovered_nodes| RunExtension {
            end,
            steps: 1_000,
            steps_released: 900,
            steps_blocked: 60,
            steps_idle: 40,
            tail_without_release,
            pending_at_exit,
            recovered_nodes,
        };
        record_run_extension(&run(RunEnd::IterationsExhausted, 0, 7, 0));
        record_run_extension(&run(RunEnd::IterationsExhausted, STALLED_TAIL_STEPS, 7, 2));
        record_run_extension(&run(RunEnd::IterationsExhausted, 500, 0, 1));
        record_run_extension(&run(RunEnd::PlanComplete, 0, 3, 2));
        record_run_extension(&run(RunEnd::PlanComplete, 0, 0, 0));

        let s = snapshot().prefix_extension;
        set_prefix_extension_enabled(false);
        // The switch alone gates the block: recording stays off with stats on.
        record_run_extension(&run(RunEnd::Deadlock, 0, 0, 0));
        let after_off = snapshot().prefix_extension;
        set_enabled(false);

        assert_eq!(s.all.runs, 5);
        assert_eq!(s.all.budget_releasing, 1);
        assert_eq!(s.all.budget_blocked, 1);
        assert_eq!(s.all.budget_idle, 1);
        assert_eq!(s.all.plan_complete_pending, 1);
        assert_eq!(s.all.plan_complete_quiescent, 1);
        assert_eq!(s.all.steps_sum, 5_000);
        assert_eq!(s.all.steps_blocked_sum, 300);
        assert_eq!(s.by_recovered_nodes[2].runs, 2);
        assert_eq!(s.by_recovered_nodes[2].budget_blocked, 1);
        assert_eq!(s.by_recovered_nodes[2].plan_complete_pending, 1);
        assert_eq!(after_off.all.runs, 5);
        assert_eq!(after_off.all.deadlock, 0);
    }

    #[test]
    fn recovery_windows_close_on_the_first_message_to_the_restarted_node() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        begin_run();
        record_crash(0, 0, 0);
        record_recover(0, 10, true);
        // A message to a node with no open window leaves the tallies alone.
        record_message_entry(1, 12);
        record_message_entry(0, 14);
        record_crash(1, 0, 0);
        record_recover(1, 20, false);

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
        record_recover(0, 5, false);
        record_crash(1, 0, 0);
        record_recover(1, 9, false);
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
