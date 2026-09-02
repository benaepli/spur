//! Learned crash placement, shared across a session. Runs are split by
//! posture: a settable share of them draw a target step for each planned
//! crash uniformly over the span completed runs actually cover and hold the
//! crash until that step; the rest stay byte-identical to stock behavior, so
//! the two populations form an internal placed-versus-stock contrast.
//!
//! The span's upper bound is the median completed-run length, learned per
//! backup-budget scope from the run-cap probes. Probes are exempt from
//! placement at every fraction, so every probe feeds the learner and the
//! lengths it reads carry neither a cap nor a placement imprint. The bound
//! is also kept under three quarters of the run's frozen step cap,
//! reserving the last quarter for the recovery tail.

use crate::simulator::rng::{Stream, StreamRng};
use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::util_stats;
use dashmap::DashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicI64, Ordering};

/// Run ids repeat their posture-and-probe phase with this period: twice the
/// run-cap probe period, so the probe population splits evenly across the
/// two postures.
const POSTURE_PERIOD: i64 = 2 * run_cap::PROBE_PERIOD;

/// Quantile of the completed-length distribution the span bound is read from.
const QUANTILE: f64 = 0.5;

/// Share of the run's frozen step cap the span bound may not exceed.
const CAP_RESERVE_NUM: i64 = 3;
const CAP_RESERVE_DEN: i64 = 4;

/// Completed stock-posture probes a scope must accumulate before its median
/// takes effect. Below the floor no hold is ever drawn.
const MIN_COMPLETED_SAMPLES: u64 = 200;

/// Histogram cells per scope, bucketed by a per-scope width so the full
/// budget fits.
const HIST_CELLS: usize = 256;

/// Share of runs the placed posture covers when nothing sets it. Held below
/// one so ordinary capped runs stay stock: run-cap probes are exempt
/// anyway, and a session whose only unplaced runs were probes would have no
/// control that differs from the rest by placement alone.
pub const DEFAULT_FRACTION: f64 = 0.9;

/// Lowest phase in the placed posture: phases at or above it are placed,
/// phases below it are stock. Set from the fraction, so the placed set grows
/// downward from the top phase and phase 0 is the last one it would reach.
static PLACED_FROM: AtomicI64 = AtomicI64::new(run_cap::PROBE_PERIOD);

/// The phase a fraction admits. Never zero, so a fraction of one still
/// leaves one posture phase stock; run-cap probes are exempt separately,
/// which is what keeps the learner's feed clear of placed lengths.
fn placed_from(fraction: f64) -> i64 {
    let covered = (fraction.clamp(0.0, 1.0) * POSTURE_PERIOD as f64).round() as i64;
    (POSTURE_PERIOD - covered).clamp(1, POSTURE_PERIOD)
}

/// Set the share of runs the placed posture covers. The default reproduces
/// the phase split the mechanism shipped with, so a session that sets
/// nothing behaves as before.
pub fn set_fraction(fraction: f64) {
    PLACED_FROM.store(placed_from(fraction), Ordering::Relaxed);
}

/// Whether this run draws crash holds. The complementary runs are left
/// exactly stock, including their random-stream draw counts. Run-cap
/// probes are never placed: their lengths are what the span is learned
/// from, so a placement imprint on them would feed back into the bound.
pub fn is_placed(run_id: i64) -> bool {
    !run_cap::is_probe(run_id)
        && run_phase::phase(run_id, POSTURE_PERIOD) >= PLACED_FROM.load(Ordering::Relaxed)
}

/// Whether this run's completed length may feed the learner. Every run-cap
/// probe does, since none of them is placed - so the feed is one run in
/// `run_cap::PROBE_PERIOD` rather than one in `POSTURE_PERIOD`, and the
/// scope crosses its sample floor in half the runs it used to take.
fn feeds_learner(run_id: i64) -> bool {
    run_cap::is_probe(run_id)
}

struct ScopeAccum {
    /// Steps per histogram cell, fixed when the scope is created.
    bucket_width: u32,
    /// Cell `c` counts the completed probes whose length fell in
    /// `[c * bucket_width, (c + 1) * bucket_width)`.
    hist: [u32; HIST_CELLS],
    /// Completed stock-posture probes folded into the histogram.
    completed: u64,
    /// The median set at the last checkpoint, governing every draw until the
    /// next one; None until the first checkpoint is crossed.
    current: Option<i32>,
    /// Completed count at which the median is next recomputed; doubles after
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
            current: None,
            next_checkpoint: MIN_COMPLETED_SAMPLES,
        }
    }

    /// The median the histogram supports right now, or None while it is
    /// below the sample floor. Laplace-smoothed and read at the winning
    /// cell's upper edge, so bucketing only ever rounds the median up.
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
                return Some(upper.min(backup as i64) as i32);
            }
        }
        None
    }
}

static TABLE: LazyLock<DashMap<i32, ScopeAccum>> = LazyLock::new(DashMap::new);

/// The learned median completed-run length for a scope, or None while the
/// scope is below its sample floor. Constant between checkpoints.
pub fn median(backup: i32) -> Option<i32> {
    TABLE.get(&backup).and_then(|acc| acc.current)
}

/// Fold one run-cap probe's ending into the learner. Only a completed probe
/// in the stock posture contributes a length; every other call returns
/// without touching the table, so the sites mirror `run_cap::merge_probe`.
pub fn merge_stock_probe(run_id: i64, backup: i32, outcome: run_cap::Outcome, steps: i32) {
    if !feeds_learner(run_id) || outcome != run_cap::Outcome::Completed {
        return;
    }
    let mut acc = TABLE.entry(backup).or_insert_with(|| ScopeAccum::new(backup));
    let cell = ((steps.max(0) as u32) / acc.bucket_width).min(HIST_CELLS as u32 - 1);
    acc.hist[cell as usize] = acc.hist[cell as usize].saturating_add(1);
    acc.completed += 1;
    if acc.completed >= acc.next_checkpoint {
        acc.current = acc.estimate(backup);
        acc.next_checkpoint = acc.next_checkpoint.saturating_mul(2);
    }
}

/// The step past which a run's crash timing must not reach: the share of the
/// frozen step cap left for the recovery tail.
pub fn cap_reserve(effective_cap: i32) -> i32 {
    (effective_cap as i64 * CAP_RESERVE_NUM / CAP_RESERVE_DEN) as i32
}

/// Draw the step a placed run holds a crash until: uniform over
/// `[t_ready, U)` where `U` is the learned median bounded by three quarters
/// of the run's frozen step cap. Returns None, drawing nothing from any
/// random stream, when the run is in the stock posture, the scope is below
/// its floor, or the span is already spent.
pub fn draw_hold(
    run_id: i64,
    backup: i32,
    effective_cap: i32,
    t_ready: i32,
    rng: &mut impl StreamRng,
) -> Option<i32> {
    if !is_placed(run_id) {
        return None;
    }
    let l50 = median(backup)?;
    let reserve = cap_reserve(effective_cap);
    let capped = reserve < l50;
    let upper = l50.min(reserve);
    if t_ready >= upper {
        return None;
    }
    rng.use_stream(Stream::FaultPriority);
    let span = (upper - t_ready) as u64;
    let target = t_ready + (rng.next_u64() % span) as i32;
    util_stats::record_crash_place_draw(capped, (target - t_ready) as u64);
    Some(target)
}

/// Scale every scope's mass by `factor`, dropping scopes that reach zero,
/// so stale phases of a long exploration lose their vote. The median and
/// the next checkpoint are left alone: shrinking the completed count delays
/// the next crossing, and the recompute there reads the decay-weighted
/// histogram.
pub fn decay(factor: f64) {
    let factor = factor.clamp(0.0, 1.0);
    TABLE.retain(|_, acc| {
        for n in acc.hist.iter_mut() {
            *n = ((*n as f64) * factor).floor() as u32;
        }
        acc.completed = ((acc.completed as f64) * factor).floor() as u64;
        acc.current.is_some() || acc.completed > 0 || acc.hist.iter().any(|&n| n > 0)
    });
}

/// Clear the table so explorer sessions in one process do not share spans,
/// and restore the default fraction. A session that forgets to push its own
/// then inherits the default rather than the previous session's value.
pub fn reset() {
    TABLE.clear();
    set_fraction(DEFAULT_FRACTION);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// Counts draws so a test can assert a path consumed none.
    struct CountingRng {
        inner: SmallRng,
        draws: u64,
    }

    impl CountingRng {
        fn new(seed: u64) -> Self {
            Self {
                inner: SmallRng::seed_from_u64(seed),
                draws: 0,
            }
        }
    }

    impl RngCore for CountingRng {
        fn next_u32(&mut self) -> u32 {
            self.draws += 1;
            self.inner.next_u32()
        }
        fn next_u64(&mut self) -> u64 {
            self.draws += 1;
            self.inner.next_u64()
        }
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            self.draws += 1;
            self.inner.fill_bytes(dst)
        }
    }

    impl StreamRng for CountingRng {}

    fn feed(n: usize, backup: i32, steps: i32) {
        let feeder = id_at_phase(0);
        debug_assert!(run_cap::is_probe(feeder), "the feed must come from a probe");
        for _ in 0..n {
            merge_stock_probe(feeder, backup, run_cap::Outcome::Completed, steps);
        }
    }

    /// A run id whose posture phase is `want`.
    fn id_at_phase(want: i64) -> i64 {
        (0..1_000_000i64)
            .find(|&id| run_phase::phase(id, POSTURE_PERIOD) == want)
            .expect("every phase is reachable")
    }

    /// Share of a long id range in the placed posture.
    fn placed_share() -> f64 {
        let n = 64_000i64;
        (-n..n).filter(|&id| is_placed(id)).count() as f64 / (2 * n) as f64
    }

    /// The share a fraction should produce: the phases it admits, less the
    /// one probe phase among them, since probes are never placed.
    fn expected_share(fraction: f64) -> f64 {
        let from = placed_from(fraction);
        let phases = POSTURE_PERIOD - from;
        let probe_phases = (from..POSTURE_PERIOD)
            .filter(|p| p.rem_euclid(run_cap::PROBE_PERIOD) == 0)
            .count() as i64;
        (phases - probe_phases) as f64 / POSTURE_PERIOD as f64
    }

    #[test]
    fn the_default_places_its_share_and_leaves_ordinary_runs_stock() {
        let _serial = config_override::exclusive_session();
        reset();
        let want = expected_share(DEFAULT_FRACTION);
        let share = placed_share();
        assert!((share - want).abs() < 0.01, "placed share {share} is not about {want}");
        assert!(share < 1.0 - 1.0 / POSTURE_PERIOD as f64, "the default leaves no stock control");
        // The stock remainder must contain runs that are not probes, or the
        // only control differs from the rest by capping as well as placement.
        let ordinary_stock = (0..64_000i64)
            .filter(|&id| !is_placed(id) && !run_cap::is_probe(id))
            .count();
        assert!(ordinary_stock > 0, "every stock run is a probe");
        assert!(is_placed(id_at_phase(POSTURE_PERIOD - 1)), "the top phase is placed");
        assert!(!is_placed(id_at_phase(0)), "phase 0 is stock");
        assert!((-64_000..0).any(is_placed), "negative ids reach the placed posture");
    }

    #[test]
    fn no_run_cap_probe_is_ever_placed_at_any_fraction() {
        let _serial = config_override::exclusive_session();
        reset();
        for f in [0.0, 0.5, 0.97, 1.0] {
            set_fraction(f);
            let placed_probes = (-64_000..64_000i64)
                .filter(|&id| run_cap::is_probe(id) && is_placed(id))
                .count();
            assert_eq!(placed_probes, 0, "a probe was placed at fraction {f}");
        }
        reset();
    }

    #[test]
    fn the_fraction_grows_the_placed_set_downward_and_never_takes_phase_zero() {
        let _serial = config_override::exclusive_session();
        reset();
        set_fraction(0.0);
        assert_eq!(placed_share(), 0.0, "zero places nothing");
        set_fraction(1.0);
        assert!(!is_placed(id_at_phase(0)), "phase 0 stays stock");
        for phase in 1..POSTURE_PERIOD {
            let id = id_at_phase(phase);
            assert_eq!(is_placed(id), !run_cap::is_probe(id), "phase {phase} at fraction one");
        }
        set_fraction(0.97);
        // 0.97 of 64 phases rounds to 62, so the two lowest stay stock.
        assert!(!is_placed(id_at_phase(0)) && !is_placed(id_at_phase(1)));
        assert!(is_placed(id_at_phase(2)));
        let share = placed_share();
        let want = expected_share(0.97);
        assert!((share - want).abs() < 0.01, "share {share} is not about {want}");
        reset();
        let d = expected_share(DEFAULT_FRACTION);
        assert!((placed_share() - d).abs() < 0.01, "reset restores the default share");
    }

    #[test]
    fn only_stock_posture_completed_probes_feed_the_learner() {
        let _serial = config_override::exclusive_session();
        reset();
        let ordinary = id_at_phase(1);
        assert!(!run_cap::is_probe(ordinary), "phase 1 of the posture period is no probe");
        let feeder = id_at_phase(0);
        assert!(run_cap::is_probe(feeder));
        for _ in 0..200 {
            // A non-probe run and every non-completed outcome stay out of
            // the histogram.
            merge_stock_probe(ordinary, 6000, run_cap::Outcome::Completed, 1200);
            merge_stock_probe(feeder, 6000, run_cap::Outcome::Exhausted, 6000);
            merge_stock_probe(feeder, 6000, run_cap::Outcome::Deadlocked, 40);
        }
        assert_eq!(median(6000), None, "nothing above reaches the floor");
        feed(200, 6000, 1200);
        assert!(median(6000).is_some(), "stock completions engage the scope");
        reset();
    }

    #[test]
    fn median_is_the_upper_cell_edge_and_constant_between_checkpoints() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(199, 6000, 1200);
        assert_eq!(median(6000), None, "199 samples are under the floor");
        feed(1, 6000, 1200);
        // Width 24, so 1200 lands in cell 50 with upper edge 1223.
        assert_eq!(median(6000), Some(1223));
        feed(150, 6000, 100);
        assert_eq!(median(6000), Some(1223), "350 samples sit between checkpoints");
        feed(50, 6000, 100);
        // At 400 the recompute reads the mixture; its median falls in the
        // 100-length cell, upper edge 119.
        assert_eq!(median(6000), Some(119));
        assert_eq!(median(1500), None, "another scope stays unengaged");
        reset();
    }

    #[test]
    fn decay_delays_the_next_checkpoint_and_reset_empties() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(200, 6000, 1200);
        assert_eq!(median(6000), Some(1223));
        decay(0.5);
        assert_eq!(median(6000), Some(1223), "the median survives decay unchanged");
        feed(300, 6000, 100);
        assert_eq!(median(6000), Some(119), "the delayed recompute follows fresher lengths");
        reset();
        assert_eq!(median(6000), None);
    }

    #[test]
    fn a_stock_run_draws_no_hold_and_consumes_no_randomness() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(200, 6000, 1200);
        let mut rng = CountingRng::new(7);
        let stock_phases = placed_from(DEFAULT_FRACTION);
        for phase in 0..stock_phases {
            let id = id_at_phase(phase);
            assert!(!is_placed(id), "phase {phase} should be below the placed threshold");
            assert_eq!(draw_hold(id, 6000, 6000, 10, &mut rng), None, "run {id} is stock");
        }
        // A probe is stock at every fraction, wherever its phase falls.
        let probe = (0..1_000_000i64).find(|&id| run_cap::is_probe(id)).unwrap();
        assert_eq!(draw_hold(probe, 6000, 6000, 10, &mut rng), None, "a probe is never placed");
        assert_eq!(rng.draws, 0, "stock posture must not touch the stream");
        reset();
    }

    #[test]
    fn a_placed_run_draws_inside_the_span_and_not_past_it() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(200, 6000, 1200);
        let mut rng = CountingRng::new(7);
        let placed = id_at_phase(POSTURE_PERIOD - 1);
        for _ in 0..100 {
            let t = draw_hold(placed, 6000, 6000, 10, &mut rng).expect("engaged scope draws");
            assert!((10..1223).contains(&t), "target {t} escapes [t_ready, U)");
        }
        assert_eq!(
            draw_hold(placed, 6000, 6000, 1223, &mut rng),
            None,
            "a spent span draws nothing"
        );
        assert_eq!(draw_hold(placed, 1500, 6000, 10, &mut rng), None, "unengaged scope");
        reset();
    }

    #[test]
    fn the_cap_reserve_bounds_the_span_and_counts_as_capped() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        feed(200, 6000, 1200);
        let before = util_stats::snapshot().crash_place;
        let mut rng = CountingRng::new(3);
        let placed = id_at_phase(POSTURE_PERIOD - 1);
        // Three quarters of a 400-step cap is 300, under the 1223 median.
        for _ in 0..50 {
            let t = draw_hold(placed, 6000, 400, 0, &mut rng).expect("capped span still draws");
            assert!((0..300).contains(&t), "target {t} escapes the reserve bound");
        }
        let t = draw_hold(placed, 6000, 6000, 0, &mut rng).expect("uncapped draw");
        assert!((0..1223).contains(&t));
        let after = util_stats::snapshot().crash_place;
        util_stats::set_enabled(false);
        assert_eq!(after.draws, before.draws + 51);
        assert_eq!(after.capped_draws, before.capped_draws + 50);
        assert!(after.held_steps_sum > before.held_steps_sum);
        reset();
    }
}
