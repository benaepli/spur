//! A step cap for runs, learned across a session and shared by every worker.
//! Runs are bucketed into scopes by their configured step budget. One run in
//! every `PROBE_PERIOD` is a probe: it always runs to the full budget and is
//! the only kind of run that feeds the learner. Every other run is capped at
//! a high quantile of the lengths of probes that completed, with headroom,
//! so step budget is not spent on runs whose plans would have finished far
//! earlier or never. The cap is recomputed only when a scope's completed
//! count crosses a doubling checkpoint (200, 400, 800, ...) and is constant
//! in between, so it is a deterministic function of the sample sequence
//! rather than of when run starts happen to read the histogram.

use crate::simulator::run_phase;
use crate::simulator::util_stats;
use dashmap::DashMap;
use std::sync::LazyLock;

/// One run in this many is a probe that runs to the full budget.
pub const PROBE_PERIOD: i64 = 32;

/// Quantile of the completed-probe length distribution the cap is read from.
const QUANTILE: f64 = 0.99;

/// Multiplier applied to the quantile so a run slightly longer than the
/// quantile still completes.
const HEADROOM: f64 = 1.5;

/// Completed probes a scope must accumulate before its cap takes effect.
/// Below the floor every run gets the full budget.
const MIN_COMPLETED_SAMPLES: u64 = 200;

/// Histogram cells per scope. Lengths are bucketed by a per-scope width so
/// the full budget fits; a completed length is always below the budget, so
/// the last cell never saturates.
const HIST_CELLS: usize = 256;

/// How a probe run ended. Only completed probes carry a length the cap can
/// be read from; exhausted and deadlocked probes are counted and discarded,
/// because a truncated length would bias the quantile downward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Exhausted,
    Deadlocked,
}

struct ScopeAccum {
    /// Steps per histogram cell, fixed when the scope is created.
    bucket_width: u32,
    /// Cell `c` counts the completed probes whose length fell in
    /// `[c * bucket_width, (c + 1) * bucket_width)`.
    hist: [u32; HIST_CELLS],
    /// Completed probes folded into the histogram.
    completed: u64,
    /// Completed probes whose length exceeded the cap in effect when they
    /// merged. A large share means the cap is cutting off completions.
    over_cap: u64,
    /// The cap set at the last checkpoint, governing every run until the
    /// next one; None until the first checkpoint is crossed.
    current: Option<i32>,
    /// Completed count at which the cap is next recomputed; doubles after
    /// each recompute.
    next_checkpoint: u64,
}

impl ScopeAccum {
    fn new(backup: i32) -> Self {
        Self {
            bucket_width: ((backup.max(1) + HIST_CELLS as i32 - 1) / HIST_CELLS as i32).max(1)
                as u32,
            hist: [0; HIST_CELLS],
            completed: 0,
            over_cap: 0,
            current: None,
            next_checkpoint: MIN_COMPLETED_SAMPLES,
        }
    }

    /// The cap governing this scope's runs, or None while no checkpoint has
    /// been crossed.
    fn cap(&self) -> Option<i32> {
        self.current
    }

    /// The cap the histogram supports right now, or None while it is below
    /// the sample floor. The quantile is Laplace-smoothed and read at the
    /// winning cell's upper edge, so bucketing only ever rounds the cap up.
    fn estimate(&self, backup: i32) -> Option<i32> {
        if self.completed < MIN_COMPLETED_SAMPLES {
            return None;
        }
        let samples: u64 = self.hist.iter().map(|&n| n as u64).sum();
        if samples == 0 {
            return None;
        }
        let denom = (samples + 2) as f64;
        let mut cum: u64 = 0;
        for (c, &n) in self.hist.iter().enumerate() {
            cum += n as u64;
            if (cum + 1) as f64 / denom >= QUANTILE {
                let upper = (c as i64 + 1) * self.bucket_width as i64 - 1;
                let capped = (HEADROOM * upper as f64).ceil() as i64;
                return Some(capped.min(backup as i64) as i32);
            }
        }
        None
    }
}

static TABLE: LazyLock<DashMap<i32, ScopeAccum>> = LazyLock::new(DashMap::new);

/// Whether this run is a probe that must run to the full budget. Read from
/// the run's mixed phase, not the id itself: a phase taken straight off the
/// id shares the id's factors with the configuration grid's width, and the
/// probe stream then misses whole configurations (`run_phase`).
pub fn is_probe(run_id: i64) -> bool {
    run_phase::phase(run_id, PROBE_PERIOD) == 0
}

/// The step cap for a run whose configured budget is `backup`: the learned
/// cap when the scope is past the floor, otherwise the budget unchanged.
/// Read once at run start so the bound is frozen for the whole run.
pub fn effective_cap(backup: i32) -> i32 {
    TABLE.get(&backup).and_then(|acc| acc.cap()).unwrap_or(backup)
}

/// Fold one probe run into its scope. Only a completed probe contributes a
/// length; the cap it is checked against is the one in effect before it
/// merges, so a completion the cap would have cut off is visible.
pub fn merge_probe(backup: i32, outcome: Outcome, steps: i32) {
    let completed = outcome == Outcome::Completed;
    util_stats::record_run_cap_probe(completed);
    if !completed {
        return;
    }
    {
        let mut acc = TABLE.entry(backup).or_insert_with(|| ScopeAccum::new(backup));
        if let Some(cap) = acc.cap() {
            if cap < backup && steps > cap {
                acc.over_cap += 1;
                util_stats::record_run_cap_over_cap_completion();
            }
        }
        let cell = ((steps.max(0) as u32) / acc.bucket_width).min(HIST_CELLS as u32 - 1);
        acc.hist[cell as usize] = acc.hist[cell as usize].saturating_add(1);
        acc.completed += 1;
        if acc.completed >= acc.next_checkpoint {
            acc.current = acc.estimate(backup);
            acc.next_checkpoint = acc.next_checkpoint.saturating_mul(2);
            util_stats::record_run_cap_recompute();
        }
    }
    publish_gauges();
}

/// Scale every scope's mass by `factor`, dropping scopes that reach zero,
/// so stale phases of a long exploration lose their vote. The cap and the
/// next checkpoint are left alone: shrinking the completed count delays the
/// next crossing, and the recompute there reads the decay-weighted
/// histogram, so the cap leans toward recent phases.
pub fn decay(factor: f64) {
    let factor = factor.clamp(0.0, 1.0);
    TABLE.retain(|_, acc| {
        for n in acc.hist.iter_mut() {
            *n = ((*n as f64) * factor).floor() as u32;
        }
        acc.completed = ((acc.completed as f64) * factor).floor() as u64;
        acc.over_cap = ((acc.over_cap as f64) * factor).floor() as u64;
        acc.current.is_some() || acc.completed > 0 || acc.hist.iter().any(|&n| n > 0)
    });
    publish_gauges();
}

/// Clear the table so explorer sessions in one process do not share caps.
pub fn reset() {
    TABLE.clear();
    util_stats::set_run_cap_learned(0, 0);
}

fn publish_gauges() {
    let mut learned: u64 = 0;
    let mut max_scope: Option<(i32, i32)> = None;
    for e in TABLE.iter() {
        let backup = *e.key();
        if let Some(cap) = e.value().cap() {
            learned += 1;
            if max_scope.map_or(true, |(k, _)| backup > k) {
                max_scope = Some((backup, cap));
            }
        }
    }
    util_stats::set_run_cap_learned(learned, max_scope.map_or(0, |(_, c)| c.max(0) as u64));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    #[test]
    fn identity_below_the_sample_floor() {
        let _serial = config_override::exclusive_session();
        reset();
        for _ in 0..199 {
            merge_probe(6000, Outcome::Completed, 1200);
        }
        assert_eq!(effective_cap(6000), 6000, "199 samples are under the floor");
        merge_probe(6000, Outcome::Completed, 1200);
        assert!(effective_cap(6000) < 6000, "the 200th sample engages the cap");
        reset();
    }

    #[test]
    fn cap_is_the_quantile_upper_edge_with_headroom_and_scopes_are_separate() {
        let _serial = config_override::exclusive_session();
        reset();
        for _ in 0..200 {
            merge_probe(6000, Outcome::Completed, 1200);
        }
        // Width 24, so 1200 lands in cell 50 with upper edge 1223; the cap
        // is ceil(1.5 * 1223).
        assert_eq!(effective_cap(6000), 1835);
        assert_eq!(effective_cap(1500), 1500, "another scope stays identity");
        reset();
    }

    #[test]
    fn a_completion_longer_than_the_engaged_cap_counts_and_still_folds_in() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        for _ in 0..200 {
            merge_probe(6000, Outcome::Completed, 100);
        }
        let cap = effective_cap(6000);
        assert!(cap < 6000);
        let before = util_stats::snapshot().run_cap;
        merge_probe(6000, Outcome::Completed, cap + 500);
        let after = util_stats::snapshot().run_cap;
        util_stats::set_enabled(false);
        assert_eq!(after.over_cap_completions, before.over_cap_completions + 1);
        assert_eq!(after.probe_completions, before.probe_completions + 1);
        reset();
    }

    #[test]
    fn exhausted_and_deadlocked_probes_feed_neither_histogram_nor_floor() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        for _ in 0..150 {
            merge_probe(6000, Outcome::Completed, 100);
        }
        for _ in 0..100 {
            merge_probe(6000, Outcome::Exhausted, 6000);
            merge_probe(6000, Outcome::Deadlocked, 40);
        }
        assert_eq!(effective_cap(6000), 6000, "non-completions must not reach the floor");
        let s = util_stats::snapshot().run_cap;
        util_stats::set_enabled(false);
        assert_eq!(s.probes, 350);
        assert_eq!(s.probe_completions, 150);
        reset();
    }

    #[test]
    fn decay_delays_the_next_checkpoint_and_reset_empties() {
        let _serial = config_override::exclusive_session();
        reset();
        for _ in 0..200 {
            merge_probe(6000, Outcome::Completed, 100);
        }
        let engaged = effective_cap(6000);
        assert!(engaged < 6000);
        decay(0.5);
        assert_eq!(effective_cap(6000), engaged, "the cap survives decay unchanged");
        assert!(!TABLE.is_empty(), "an engaged scope is retained through decay");
        // Decay halved the completed count to 100; 300 more completions
        // re-cross the 400 checkpoint and the recompute reads the mixed,
        // decay-weighted histogram.
        for _ in 0..300 {
            merge_probe(6000, Outcome::Completed, 1200);
        }
        assert_eq!(effective_cap(6000), 1835, "the recompute follows the fresher lengths");
        reset();
        assert!(TABLE.is_empty());
        assert_eq!(effective_cap(6000), 6000);
    }

    #[test]
    fn cap_is_constant_between_checkpoints() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        for _ in 0..200 {
            merge_probe(6000, Outcome::Completed, 1200);
        }
        assert_eq!(effective_cap(6000), 1835);
        let before = util_stats::snapshot().run_cap;
        for _ in 0..150 {
            merge_probe(6000, Outcome::Completed, 4000);
        }
        let after = util_stats::snapshot().run_cap;
        assert_eq!(effective_cap(6000), 1835, "350 completions sit between checkpoints");
        assert_eq!(
            after.over_cap_completions,
            before.over_cap_completions + 150,
            "completions above the standing cap are counted while it holds"
        );
        for _ in 0..50 {
            merge_probe(6000, Outcome::Completed, 4000);
        }
        util_stats::set_enabled(false);
        // The p99 of the mixture lands in the 4000-length cell, whose upper
        // edge with headroom exceeds the budget, so the cap opens back up.
        assert_eq!(effective_cap(6000), 6000);
        reset();
    }

    #[test]
    fn recompute_fires_once_per_checkpoint() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        let base = util_stats::snapshot().run_cap.cap_recomputes;
        for _ in 0..200 {
            merge_probe(6000, Outcome::Completed, 1200);
        }
        assert_eq!(util_stats::snapshot().run_cap.cap_recomputes, base + 1);
        for _ in 0..200 {
            merge_probe(6000, Outcome::Completed, 1200);
        }
        assert_eq!(util_stats::snapshot().run_cap.cap_recomputes, base + 2);
        for _ in 0..400 {
            merge_probe(6000, Outcome::Completed, 1200);
        }
        assert_eq!(util_stats::snapshot().run_cap.cap_recomputes, base + 3);
        util_stats::set_enabled(false);
        reset();
    }

    #[test]
    fn disengaged_scope_recomputes_at_backup() {
        let _serial = config_override::exclusive_session();
        reset();
        for _ in 0..200 {
            merge_probe(1500, Outcome::Completed, 1400);
        }
        assert_eq!(effective_cap(1500), 1500, "the estimate clamps to the budget");
        for _ in 0..200 {
            merge_probe(1500, Outcome::Completed, 1400);
        }
        assert_eq!(effective_cap(1500), 1500);
        reset();
    }

    #[test]
    fn probes_are_one_run_in_the_period_and_spread_over_every_grid_width() {
        let _serial = config_override::exclusive_session();
        let n = 64_000i64;
        let probes: Vec<i64> = (0..n).filter(|&id| is_probe(id)).collect();
        let want = n / PROBE_PERIOD;
        assert!(
            (probes.len() as i64 - want).abs() < want / 10,
            "{} probes in {n} runs, expected about {want}",
            probes.len()
        );
        // The reason the phase is mixed: against a grid walked in order, a
        // probe stream that misses a residue never sees those configs.
        for width in [2i64, 6, 27, 54] {
            let mut seen = vec![false; width as usize];
            for &id in &probes {
                seen[id.rem_euclid(width) as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "probes miss a position of a {width}-wide grid");
        }
        assert!((-64_000..0).any(is_probe), "negative ids designate probes too");
    }
}
