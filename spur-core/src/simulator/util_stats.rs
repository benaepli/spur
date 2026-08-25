//! Opt-in, process-wide utilization counters for explorer mechanisms.
//!
//! Enabled per explorer session via `ExplorerConfig::stats` and dumped by the
//! CLI to `<output_dir>/utilization.json`. Counters are observation-only: they
//! never affect scheduling, scoring, or RNG consumption. When disabled, every
//! probe is a single relaxed atomic load.

use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);

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

static AUDIT_DECISIONS: AtomicU64 = AtomicU64::new(0);
static AUDIT_DIVERGENT_PICKS: AtomicU64 = AtomicU64::new(0);
static AUDIT_TIEBREAKS: AtomicU64 = AtomicU64::new(0);
static AUDIT_FLAT_NOVELTY: AtomicU64 = AtomicU64::new(0);
static AUDIT_SCORE_VARIANCE_SUM: AtomicU64 = AtomicU64::new(0);
static AUDIT_NOVELTY_VARIANCE_SUM: AtomicU64 = AtomicU64::new(0);
static AUDIT_PLAN_RUNS: AtomicU64 = AtomicU64::new(0);
static AUDIT_PLAN_SUM: AtomicU64 = AtomicU64::new(0);
static AUDIT_PLAN_SQ_SUM: AtomicU64 = AtomicU64::new(0);

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
        ] {
            c.store(0, Ordering::Relaxed);
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

/// Enable or disable the scoring-authority audit for this session. Enabling
/// resets the audit counters.
pub fn set_audit_enabled(on: bool) {
    if on {
        for c in [
            &AUDIT_DECISIONS,
            &AUDIT_DIVERGENT_PICKS,
            &AUDIT_TIEBREAKS,
            &AUDIT_FLAT_NOVELTY,
            &AUDIT_SCORE_VARIANCE_SUM,
            &AUDIT_NOVELTY_VARIANCE_SUM,
            &AUDIT_PLAN_RUNS,
            &AUDIT_PLAN_SUM,
            &AUDIT_PLAN_SQ_SUM,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
    AUDIT_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether the scoring-authority audit is recording.
#[inline]
pub fn audit_enabled() -> bool {
    AUDIT_ENABLED.load(Ordering::Relaxed)
}

/// Whether any counter group is recording, i.e. whether a snapshot is worth
/// writing out.
#[inline]
pub fn any_enabled() -> bool {
    enabled() || audit_enabled()
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

/// One within-queue selection over >1 eligible items, described by how much
/// authority the score had over it: whether the blended argmax differed from
/// the priority-only argmax (`divergent`), whether that difference resolved a
/// priority tie (`tiebreak`), whether every candidate had the same novelty so
/// the score could not discriminate at all (`flat_novelty`), and the spread of
/// the blended score and of the raw novelty term across the candidates.
#[inline]
pub fn record_decision_audit(
    divergent: bool,
    tiebreak: bool,
    flat_novelty: bool,
    score_variance: f64,
    novelty_variance: f64,
) {
    if !audit_enabled() {
        return;
    }
    AUDIT_DECISIONS.fetch_add(1, Ordering::Relaxed);
    if divergent {
        AUDIT_DIVERGENT_PICKS.fetch_add(1, Ordering::Relaxed);
    }
    if tiebreak {
        AUDIT_TIEBREAKS.fetch_add(1, Ordering::Relaxed);
    }
    if flat_novelty {
        AUDIT_FLAT_NOVELTY.fetch_add(1, Ordering::Relaxed);
    }
    add_f64(&AUDIT_SCORE_VARIANCE_SUM, score_variance);
    add_f64(&AUDIT_NOVELTY_VARIANCE_SUM, novelty_variance);
}

/// The final fitness a completed run was ranked by. Its spread across runs is
/// the plan-selection counterpart to the per-decision spread above: a near-zero
/// variance means ranking plans by this score is close to ranking them at
/// random.
#[inline]
pub fn record_plan_score(score: f64) {
    if !audit_enabled() {
        return;
    }
    AUDIT_PLAN_RUNS.fetch_add(1, Ordering::Relaxed);
    add_f64(&AUDIT_PLAN_SUM, score);
    add_f64(&AUDIT_PLAN_SQ_SUM, score * score);
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

/// How much authority the scoring layer had over what actually got scheduled.
/// All zero when the audit was not enabled.
#[derive(Serialize)]
pub struct AuditStats {
    pub decisions: u64,
    pub divergent_picks: u64,
    pub decision_divergence_frac: f64,
    pub tiebreak_decisions: u64,
    pub tiebreak_frac: f64,
    pub flat_novelty_decisions: u64,
    pub flat_novelty_frac: f64,
    pub score_variance_mean: f64,
    pub novelty_variance_mean: f64,
    pub plan_scored_runs: u64,
    pub plan_score_mean: f64,
    pub plan_score_variance: f64,
}

/// A point-in-time copy of all counters, serializable to `utilization.json`.
#[derive(Serialize)]
pub struct UtilizationSnapshot {
    pub steer: SteerStats,
    pub purgatory: PurgatoryStats,
    pub aos: AosStats,
    pub dedup: DedupStats,
    pub feedback: FeedbackStats,
    pub score_authority: AuditStats,
}

fn audit_snapshot() -> AuditStats {
    let decisions = AUDIT_DECISIONS.load(Ordering::Relaxed);
    let plan_runs = AUDIT_PLAN_RUNS.load(Ordering::Relaxed);
    let per_decision = |v: u64| {
        if decisions == 0 {
            0.0
        } else {
            v as f64 / decisions as f64
        }
    };
    let per_decision_f64 = |cell: &AtomicU64| {
        if decisions == 0 {
            0.0
        } else {
            f64::from_bits(cell.load(Ordering::Relaxed)) / decisions as f64
        }
    };
    let plan_sum = f64::from_bits(AUDIT_PLAN_SUM.load(Ordering::Relaxed));
    let plan_sq_sum = f64::from_bits(AUDIT_PLAN_SQ_SUM.load(Ordering::Relaxed));
    let (plan_mean, plan_variance) = if plan_runs == 0 {
        (0.0, 0.0)
    } else {
        let n = plan_runs as f64;
        let mean = plan_sum / n;
        ((mean), (plan_sq_sum / n - mean * mean).max(0.0))
    };
    AuditStats {
        decisions,
        divergent_picks: AUDIT_DIVERGENT_PICKS.load(Ordering::Relaxed),
        decision_divergence_frac: per_decision(AUDIT_DIVERGENT_PICKS.load(Ordering::Relaxed)),
        tiebreak_decisions: AUDIT_TIEBREAKS.load(Ordering::Relaxed),
        tiebreak_frac: per_decision(AUDIT_TIEBREAKS.load(Ordering::Relaxed)),
        flat_novelty_decisions: AUDIT_FLAT_NOVELTY.load(Ordering::Relaxed),
        flat_novelty_frac: per_decision(AUDIT_FLAT_NOVELTY.load(Ordering::Relaxed)),
        score_variance_mean: per_decision_f64(&AUDIT_SCORE_VARIANCE_SUM),
        novelty_variance_mean: per_decision_f64(&AUDIT_NOVELTY_VARIANCE_SUM),
        plan_scored_runs: plan_runs,
        plan_score_mean: plan_mean,
        plan_score_variance: plan_variance,
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
        score_authority: audit_snapshot(),
    }
}
