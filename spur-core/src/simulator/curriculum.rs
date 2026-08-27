//! Semantic-knob curriculum for the continuous explorer.
//!
//! A knob is a single `f64` in `[0, 1]` whose meaning is defined entirely by a
//! lowering function ([`lower`]) that fans it out to many concrete
//! `SingleRunConfig` fields. The curriculum moves the knobs over the course of a
//! session via [`knob_value`].
//! The lowering interpolates within the config-provided ranges,
//! so a knob only chooses where inside an allowed range a run sits.

use std::f64::consts::PI;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::simulator::core::{QueuePolicyConfig, SchedulePolicy, WithinQueueSelector};
use crate::simulator::explorer::{ExplorerConfig, Range, SingleRunConfig};

/// Hard cap on concurrent write-like ops.
pub const MAX_CONCURRENT_WRITES_CAP: i32 = 3;

/// The three semantic knobs, each in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Knobs {
    /// Cluster + workload size. Low = small/obvious; high = large.
    pub scale: f64,
    /// Interleaving freedom. Low = near-sequential; high = many concurrent ops.
    pub concurrency: f64,
    /// How many faults, and how aggressively they bite. Low = few/loose; high =
    /// many crashes recovered promptly.
    pub fault_tightness: f64,
}

/// Knob identity, used to give each knob a desynchronized trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Knob {
    Scale,
    Concurrency,
    FaultTightness,
}

impl Knob {
    /// Phase offset so the knobs do not all move in lockstep. This lets
    /// combinations like loose faults with high concurrency show up too.
    fn phase(self) -> f64 {
        match self {
            Knob::Scale => 0.0,
            Knob::Concurrency => 0.33,
            Knob::FaultTightness => 0.66,
        }
    }
}

/// Linear interpolation, clamping `t` to `[0, 1]`.
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Interpolate within an integer range at parameter `t`, snapped to the grid
/// `{min, min+step, ...}` (never exceeding `max`).
pub fn lerp_range(r: &Range, t: f64) -> i32 {
    let t = t.clamp(0.0, 1.0);
    let step = r.step.max(1);
    let steps = ((r.max - r.min) / step).max(0);
    r.min + (steps as f64 * t).round() as i32 * step
}

/// Pick a dependency-density value from the config's discrete list nearest the
/// target implied by `concurrency` (higher concurrency picks a lower density,
/// which allows more interleaving). Falls back to a continuous sweep if the
/// list is empty.
fn pick_density(values: &[f64], concurrency: f64) -> f64 {
    if values.is_empty() {
        return lerp(0.6, 0.1, concurrency);
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let target = lerp(max, min, concurrency);
    values
        .iter()
        .copied()
        .min_by(|a, b| {
            (a - target)
                .abs()
                .partial_cmp(&(b - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0.5)
}

/// Round `n` toward the nearest odd value inside `[lo, hi]`, preferring upward.
/// Quorum protocols want odd cluster sizes; if no odd value fits, returns `n`.
fn prefer_odd(n: i32, lo: i32, hi: i32) -> i32 {
    if n % 2 != 0 {
        return n;
    }
    let up = n + 1;
    let down = n - 1;
    if up <= hi {
        up
    } else if down >= lo {
        down
    } else {
        n
    }
}

/// Build a `Shaped` schedule policy from the default, overriding the `recover`
/// band center (a higher center means higher recover priority at the
/// within-queue draw). Loose faults keep recover low; tight faults push it
/// toward 1.0.
fn shaped_with_recover_center(center: f64) -> SchedulePolicy {
    match SchedulePolicy::default() {
        SchedulePolicy::Shaped {
            alpha,
            beta,
            record,
            timer,
            channel_send,
            crash,
            mut recover,
            partition,
            heal,
        } => {
            recover.center = center.clamp(0.0, 1.0);
            SchedulePolicy::Shaped {
                alpha,
                beta,
                record,
                timer,
                channel_send,
                crash,
                recover,
                partition,
                heal,
            }
        }
        other => other,
    }
}

/// Lower the knobs into a concrete run config. `constraints` supplies the range
/// envelope. Always caps `max_concurrent_writes`, and keeps the within-queue
/// selector greedy when `fault_tightness` is high so the recover-priority boost
/// is not inert.
pub fn lower(knobs: &Knobs, constraints: &ExplorerConfig, _rng: &mut impl Rng) -> SingleRunConfig {
    // scale -> sizes
    let num_servers = prefer_odd(
        lerp_range(&constraints.num_servers_range, knobs.scale),
        constraints.num_servers_range.min,
        constraints.num_servers_range.max,
    );
    let num_write_ops = lerp_range(&constraints.num_write_ops_range, knobs.scale);
    let num_read_ops = lerp_range(&constraints.num_read_ops_range, knobs.scale);
    let num_rmw_ops = lerp_range(&constraints.num_rmw_ops_range, knobs.scale);
    let num_keys = lerp_range(&constraints.num_keys_range, knobs.scale);

    // concurrency -> density + write cap + routing
    let dependency_density =
        pick_density(&constraints.dependency_density_values, knobs.concurrency);

    let mcw_range = constraints
        .max_concurrent_writes_range
        .clone()
        .unwrap_or(Range {
            min: 1,
            max: MAX_CONCURRENT_WRITES_CAP,
            step: 1,
        });
    let mcw_lo = mcw_range.min.clamp(1, MAX_CONCURRENT_WRITES_CAP);
    let mcw_hi = mcw_range.max.clamp(mcw_lo, MAX_CONCURRENT_WRITES_CAP);
    let max_concurrent_writes =
        Some(lerp(mcw_lo as f64, mcw_hi as f64, knobs.concurrency).round() as i32);

    // fault_tightness -> counts + scheduling timing
    let num_crashes = lerp_range(&constraints.num_crashes_range, knobs.fault_tightness);
    let num_partitions = lerp_range(&constraints.num_partitions_range, knobs.fault_tightness);

    // p_local drops as concurrency rises (more network reordering), but stays
    // high under tight faults so the recover's (typically size-1) local queue
    // can still be drawn at the queue-selection step.
    let mut p_local = lerp(0.92, 0.75, knobs.concurrency);
    if knobs.fault_tightness > 0.6 {
        p_local = p_local.max(0.85);
    }
    let queue_policy = QueuePolicyConfig::Probabilistic {
        p_local,
        p_timer: 0.03,
    };

    // recover band rises with tightness; quick-fire weight rises with tightness.
    let schedule_policy = shaped_with_recover_center(lerp(0.5, 1.0, knobs.fault_tightness));
    let quick_fire_multiplier = lerp(1.0, 8.0, knobs.fault_tightness);

    // Keep the selector greedy when we lean on priority; else honor config.
    let within_queue_selector = if knobs.fault_tightness > 0.5 {
        WithinQueueSelector::Tournament { k: 10 }
    } else {
        constraints.within_queue_selector.clone()
    };

    crate::simulator::util_stats::record_curriculum_lowering(num_crashes, num_servers);

    SingleRunConfig {
        num_servers,
        num_write_ops,
        num_read_ops,
        num_rmw_ops,
        num_keys,
        num_crashes,
        num_partitions,
        max_concurrent_writes,
        dependency_density,
        post_fault_client_ops: constraints.post_fault_client_ops,
        use_coverage_scheduling: constraints.use_coverage_scheduling,
        max_iterations: constraints.max_iterations,
        schedule_policy,
        queue_policy,
        within_queue_selector,
        quick_fire_multiplier,
        purgatory: constraints.purgatory.clone(),
        timeline_key_granularity: constraints.feedback.key_granularity(),
        rng_stream_isolation: constraints.rng_stream_isolation,
    }
}

/// Smooth trajectory for a knob: a knob-specific linear trend plus a
/// desynchronized sinusoidal oscillation, clamped to `[0, 1]`.
fn drift_curve(knob: Knob, progress: f64, phase: f64) -> f64 {
    let trend = match knob {
        // Start small and grow over the session.
        Knob::Scale => progress,
        // Centered; rely on oscillation to wander.
        Knob::Concurrency => 0.5,
        // Start tight, loosen toward ~0.2.
        Knob::FaultTightness => 1.0 - 0.8 * progress,
    };
    let osc = 0.25 * (2.0 * PI * (1.5 * progress + phase)).sin();
    (trend + osc).clamp(0.0, 1.0)
}

/// Evaluate one knob's trajectory. `progress` (0 to 1) is session/slice
/// progress; `stagnation` (0 to 1) is the optional flatlined-novelty signal;
/// `rng` supplies per-knob jitter so the knob wanders rather than ramping
/// monotonically.
pub fn knob_value(knob: Knob, progress: f64, stagnation: f64, rng: &mut impl Rng) -> f64 {
    let drift = drift_curve(knob, progress, knob.phase());
    let jitter = rng.random_range(-0.15..=0.15);
    // When novelty flatlines, kick toward the more intense end to break out.
    let kick = stagnation.clamp(0.0, 1.0) * 0.3;
    (drift + jitter + kick).clamp(0.0, 1.0)
}

/// Stateful curriculum: tracks a run clock and the current stagnation signal,
/// and samples a fresh `Knobs` per run.
#[derive(Debug)]
pub struct Curriculum {
    rng: SmallRng,
    elapsed: u64,
    /// Number of runs over which `progress` sweeps from 0 to 1.
    horizon: u64,
    stagnation: f64,
}

impl Curriculum {
    pub fn new(seed: u64, horizon: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            elapsed: 0,
            horizon: horizon.max(1),
            stagnation: 0.0,
        }
    }

    /// Current session progress in `[0, 1]`.
    pub fn progress(&self) -> f64 {
        (self.elapsed as f64 / self.horizon as f64).min(1.0)
    }

    /// Sample a fresh set of knob values at the current progress/stagnation.
    pub fn sample(&mut self) -> Knobs {
        let progress = self.progress();
        let stagnation = self.stagnation;
        Knobs {
            scale: knob_value(Knob::Scale, progress, stagnation, &mut self.rng),
            concurrency: knob_value(Knob::Concurrency, progress, stagnation, &mut self.rng),
            fault_tightness: knob_value(Knob::FaultTightness, progress, stagnation, &mut self.rng),
        }
    }

    /// Advance the run clock after a batch of `runs`.
    pub fn advance(&mut self, runs: u64) {
        self.elapsed = self.elapsed.saturating_add(runs);
    }

    /// Update the stagnation signal (clamped to `[0, 1]`).
    pub fn set_stagnation(&mut self, s: f64) {
        self.stagnation = s.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::explorer::ExplorerConfig;

    /// A representative envelope mirroring a typical falsification config.
    fn test_constraints() -> ExplorerConfig {
        let json = r#"{
            "num_servers": {"min": 3, "max": 5, "step": 2},
            "num_write_ops": {"min": 2, "max": 6},
            "num_read_ops": {"min": 1, "max": 4},
            "num_keys": {"min": 1, "max": 2},
            "num_crashes": {"min": 0, "max": 3},
            "num_partitions": {"min": 0, "max": 1},
            "max_concurrent_writes": {"min": 1, "max": 5},
            "dependency_density": [0.1, 0.3, 0.6],
            "num_runs_per_config": 1,
            "max_iterations": 1000
        }"#;
        serde_json::from_str(json).expect("valid test config")
    }

    #[test]
    fn lerp_range_hits_endpoints_and_snaps_to_step() {
        let r = Range {
            min: 3,
            max: 9,
            step: 2,
        };
        assert_eq!(lerp_range(&r, 0.0), 3);
        assert_eq!(lerp_range(&r, 1.0), 9);
        // Midpoint snaps to a valid step value within [3, 9].
        let mid = lerp_range(&r, 0.5);
        assert!([3, 5, 7, 9].contains(&mid), "got {mid}");
    }

    #[test]
    fn lower_caps_writes_and_emits_shaped_policy() {
        let c = test_constraints();
        let mut rng = SmallRng::seed_from_u64(7);
        // Max everything; the write cap must still hold at 3 even though the
        // range allows 5.
        let knobs = Knobs {
            scale: 1.0,
            concurrency: 1.0,
            fault_tightness: 1.0,
        };
        let cfg = lower(&knobs, &c, &mut rng);
        assert_eq!(
            cfg.max_concurrent_writes,
            Some(MAX_CONCURRENT_WRITES_CAP),
            "write concurrency must be capped at {MAX_CONCURRENT_WRITES_CAP}"
        );
        assert!(matches!(cfg.schedule_policy, SchedulePolicy::Shaped { .. }));
        // High fault_tightness must keep the selector greedy (priority not inert).
        assert!(matches!(
            cfg.within_queue_selector,
            WithinQueueSelector::Tournament { .. }
        ));
        // Sizes stay within the envelope.
        assert!((3..=5).contains(&cfg.num_servers));
        assert!((2..=6).contains(&cfg.num_write_ops));
    }

    #[test]
    fn lower_write_cap_holds_when_config_min_exceeds_cap() {
        let mut c = test_constraints();
        c.max_concurrent_writes_range = Some(Range {
            min: 5,
            max: 8,
            step: 1,
        });
        let mut rng = SmallRng::seed_from_u64(7);
        let knobs = Knobs {
            scale: 1.0,
            concurrency: 1.0,
            fault_tightness: 0.0,
        };
        let cfg = lower(&knobs, &c, &mut rng);
        assert!(cfg.max_concurrent_writes.unwrap() <= MAX_CONCURRENT_WRITES_CAP);
    }

    #[test]
    fn lower_low_knobs_are_small_and_loose() {
        let c = test_constraints();
        let mut rng = SmallRng::seed_from_u64(7);
        let knobs = Knobs {
            scale: 0.0,
            concurrency: 0.0,
            fault_tightness: 0.0,
        };
        let cfg = lower(&knobs, &c, &mut rng);
        assert_eq!(cfg.num_servers, 3); // range min, already odd
        assert_eq!(cfg.num_crashes, 0); // loose, so no crashes
        assert_eq!(cfg.max_concurrent_writes, Some(1)); // near-sequential
        // Low concurrency picks the highest density offered.
        assert_eq!(cfg.dependency_density, 0.6);
    }

    #[test]
    fn knob_value_stays_in_unit_interval() {
        let mut rng = SmallRng::seed_from_u64(1);
        for step in 0..=20 {
            let p = step as f64 / 20.0;
            for knob in [Knob::Scale, Knob::Concurrency, Knob::FaultTightness] {
                for stag in [0.0, 0.5, 1.0] {
                    let v = knob_value(knob, p, stag, &mut rng);
                    assert!((0.0..=1.0).contains(&v), "knob {knob:?} p={p} -> {v}");
                }
            }
        }
    }

    #[test]
    fn curriculum_scale_trends_upward_over_session() {
        let mut cur = Curriculum::new(42, 100);
        let early = cur.sample().scale;
        cur.advance(100); // jump to end of horizon
        let late = cur.sample().scale;
        // Not strictly monotone (jitter), but the trend should lift scale.
        assert!(
            late >= early - 0.2,
            "scale should trend up: early={early}, late={late}"
        );
        assert!((cur.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn knobs_are_desynchronized() {
        // At a fixed progress, the three knobs should not collapse to one value.
        let mut cur = Curriculum::new(5, 100);
        cur.advance(50);
        let k = cur.sample();
        let all_equal = (k.scale - k.concurrency).abs() < 1e-9
            && (k.concurrency - k.fault_tightness).abs() < 1e-9;
        assert!(!all_equal, "knobs collapsed: {k:?}");
    }
}
