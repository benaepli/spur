//! Learned per-context odds that a timer firing changes its node's state,
//! shared across a session and read at the queue-selection roll. Runs are
//! split by id into steer-off probes and steered runs: probes apply no bias
//! and are the only runs that feed the learner, so the learned rates are
//! free of the bias they drive. The learner keys each firing by structural
//! features of the timer's owner node - pending deliveries, in-flight sends,
//! the node's largest inert-firing streak, and restart recency - and a
//! steered run multiplies its configured timer probability by the cell's
//! acted rate over the global acted rate, clamped.
//!
//! Learning piggybacks on the per-firing acted probe, so it only accumulates
//! while the session records acted fractions (`emit_acted_fraction`).

use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::util_stats;
use std::sync::atomic::{AtomicU64, Ordering};

/// The phase, modulo `run_cap::PROBE_PERIOD`, that marks a steer-off probe
/// run. Sharing run_cap's period with a different phase keeps the two probe
/// populations disjoint. The phase is read from the run's mixed id
/// (`run_phase`), so it does not align with the configuration grid's width.
pub const PROBE_PHASE: i64 = 16;

/// Firings a cell must accumulate before its multiplier engages.
const MIN_CELL_SAMPLES: u64 = 200;

/// Bounds on the cell-rate over global-rate ratio a steered roll applies.
const CLAMP_LO: f64 = 0.25;
const CLAMP_HI: f64 = 4.0;

/// Product of the feature buckets: pending {3} x in-flight {2} x streak {4}
/// x restarted {2}.
const CELLS: usize = 48;

/// Whether a run feeds the learner unbiased or reads it to steer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    Probe,
    Steered,
}

/// The mode a run id designates. Probe runs take the stock roll and are the
/// only runs whose firings the learner counts.
pub fn run_mode(run_id: i64) -> RunMode {
    if run_phase::phase(run_id, run_cap::PROBE_PERIOD) == PROBE_PHASE {
        RunMode::Probe
    } else {
        RunMode::Steered
    }
}

/// Index of one context cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellKey(u8);

/// The cell for a node's structural features. The streak edges mirror the
/// per-firing probe's inert-streak buckets, and the restart-recency window
/// mirrors that bucketing's longest edge, so a cell here names the same
/// contexts the probe's own histogram splits by.
pub fn cell_key(
    pending_deliveries: usize,
    in_flight_sends: u32,
    node_max_inert_streak: u32,
    recently_restarted: bool,
) -> CellKey {
    let pending: usize = match pending_deliveries {
        0 => 0,
        1..=2 => 1,
        _ => 2,
    };
    let inflight: usize = usize::from(in_flight_sends > 0);
    let streak: usize = match node_max_inert_streak {
        0 => 0,
        1..=2 => 1,
        3..=7 => 2,
        _ => 3,
    };
    let restarted = usize::from(recently_restarted);
    CellKey((pending * 16 + inflight * 8 + streak * 2 + restarted) as u8)
}

static CELL_FIRED: [AtomicU64; CELLS] = [const { AtomicU64::new(0) }; CELLS];
static CELL_ACTED: [AtomicU64; CELLS] = [const { AtomicU64::new(0) }; CELLS];
// Dedicated global tallies so a multiplier read is four relaxed loads, not
// a sum over every cell.
static GLOBAL_FIRED: AtomicU64 = AtomicU64::new(0);
static GLOBAL_ACTED: AtomicU64 = AtomicU64::new(0);

/// Fold one probe-run firing into its cell. The engaged-cell gauge is
/// republished only when this firing lifts the cell over the sample floor,
/// which keeps the per-firing cost flat.
pub fn record_firing(cell: CellKey, acted: bool) {
    util_stats::record_timer_context_probe(acted);
    let i = cell.0 as usize;
    let prev = CELL_FIRED[i].fetch_add(1, Ordering::Relaxed);
    if acted {
        CELL_ACTED[i].fetch_add(1, Ordering::Relaxed);
    }
    GLOBAL_FIRED.fetch_add(1, Ordering::Relaxed);
    if acted {
        GLOBAL_ACTED.fetch_add(1, Ordering::Relaxed);
    }
    if prev + 1 == MIN_CELL_SAMPLES {
        publish_gauges();
    }
}

/// The factor a steered roll multiplies its timer probability by for a
/// timer in `cell`, or None while the cell is under the sample floor or the
/// global rate is degenerate. Pure read: no locks, no rng.
pub fn multiplier(cell: CellKey) -> Option<f64> {
    let fired = CELL_FIRED[cell.0 as usize].load(Ordering::Relaxed);
    if fired < MIN_CELL_SAMPLES {
        return None;
    }
    let global_fired = GLOBAL_FIRED.load(Ordering::Relaxed);
    let global_acted = GLOBAL_ACTED.load(Ordering::Relaxed);
    if global_fired == 0 || global_acted == 0 {
        return None;
    }
    let acted = CELL_ACTED[cell.0 as usize].load(Ordering::Relaxed);
    let cell_rate = acted as f64 / fired as f64;
    let global_rate = global_acted as f64 / global_fired as f64;
    Some((cell_rate / global_rate).clamp(CLAMP_LO, CLAMP_HI))
}

/// Scale every cell and the globals by `factor`, so stale phases of a long
/// exploration lose their vote. Runs only between batches, with no run in
/// flight, so the non-atomic load/store scaling cannot race a merge.
pub fn decay(factor: f64) {
    let factor = factor.clamp(0.0, 1.0);
    for c in CELL_FIRED
        .iter()
        .chain(CELL_ACTED.iter())
        .chain([&GLOBAL_FIRED, &GLOBAL_ACTED])
    {
        let v = c.load(Ordering::Relaxed);
        c.store(((v as f64) * factor).floor() as u64, Ordering::Relaxed);
    }
    publish_gauges();
}

/// Clear the table so explorer sessions in one process do not share odds.
pub fn reset() {
    for c in CELL_FIRED
        .iter()
        .chain(CELL_ACTED.iter())
        .chain([&GLOBAL_FIRED, &GLOBAL_ACTED])
    {
        c.store(0, Ordering::Relaxed);
    }
    util_stats::set_timer_context_cells_engaged(0);
}

fn publish_gauges() {
    let engaged = CELL_FIRED
        .iter()
        .filter(|c| c.load(Ordering::Relaxed) >= MIN_CELL_SAMPLES)
        .count() as u64;
    util_stats::set_timer_context_cells_engaged(engaged);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    #[test]
    fn multiplier_engages_at_the_sample_floor() {
        let _serial = config_override::exclusive_session();
        reset();
        let cell = cell_key(0, 0, 0, false);
        for _ in 0..199 {
            record_firing(cell, true);
        }
        assert_eq!(multiplier(cell), None, "199 samples are under the floor");
        record_firing(cell, true);
        let m = multiplier(cell).expect("the 200th sample engages the cell");
        assert!((m - 1.0).abs() < 1e-9, "the only cell sits at the global rate");
        reset();
    }

    #[test]
    fn ratio_is_clamped_and_the_global_rate_is_the_unit() {
        let _serial = config_override::exclusive_session();
        reset();
        let hot = cell_key(3, 1, 0, false);
        let cold = cell_key(0, 0, 8, false);
        let par = cell_key(1, 0, 1, false);
        for _ in 0..200 {
            record_firing(hot, true);
        }
        for i in 0..1000 {
            record_firing(cold, i < 10);
        }
        for i in 0..200 {
            record_firing(par, i % 40 < 7);
        }
        // Global rate is 245/1400 = 0.175; the hot cell's 1.0 and the cold
        // cell's 0.01 both fall outside the clamp.
        assert_eq!(multiplier(hot), Some(CLAMP_HI));
        assert_eq!(multiplier(cold), Some(CLAMP_LO));
        let m = multiplier(par).expect("engaged");
        assert!((m - 1.0).abs() < 1e-9, "a cell at the global rate is the identity");
        reset();
    }

    #[test]
    fn zero_global_acted_yields_no_multiplier() {
        let _serial = config_override::exclusive_session();
        reset();
        let cell = cell_key(0, 0, 0, false);
        for _ in 0..200 {
            record_firing(cell, false);
        }
        assert_eq!(multiplier(cell), None);
        reset();
    }

    #[test]
    fn decay_disengages_and_reset_empties() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        let cell = cell_key(0, 0, 0, false);
        for _ in 0..200 {
            record_firing(cell, true);
        }
        assert!(multiplier(cell).is_some());
        assert_eq!(util_stats::snapshot().timer_context.probe_firings, 200);
        assert_eq!(util_stats::snapshot().timer_context.probe_acted, 200);
        decay(0.5);
        assert_eq!(multiplier(cell), None, "100 samples fall under the floor");
        assert_eq!(util_stats::snapshot().timer_context.cells_engaged, 0);
        for _ in 0..20 {
            decay(0.5);
        }
        assert_eq!(GLOBAL_FIRED.load(Ordering::Relaxed), 0, "repeated decay empties");

        for _ in 0..200 {
            record_firing(cell, true);
        }
        assert!(multiplier(cell).is_some());
        reset();
        util_stats::set_enabled(false);
        assert_eq!(multiplier(cell), None);
        assert_eq!(GLOBAL_FIRED.load(Ordering::Relaxed), 0);
        assert_eq!(util_stats::snapshot().timer_context.cells_engaged, 0);
    }

    #[test]
    fn probe_phase_is_disjoint_from_run_cap_probes() {
        let _serial = config_override::exclusive_session();
        let n = 64_000i64;
        let mut probes = 0i64;
        for id in -n..n {
            let both = run_mode(id) == RunMode::Probe && run_cap::is_probe(id);
            assert!(!both, "run {id} is a probe for both learners");
            if run_mode(id) == RunMode::Probe {
                probes += 1;
            }
        }
        let want = 2 * n / run_cap::PROBE_PERIOD;
        assert!(
            (probes - want).abs() < want / 10,
            "{probes} steer-off probes in {} runs, expected about {want}",
            2 * n
        );
    }

    #[test]
    fn engaged_cells_gauge_counts_cells_over_the_floor() {
        let _serial = config_override::exclusive_session();
        reset();
        let a = cell_key(0, 0, 0, false);
        let b = cell_key(0, 0, 0, true);
        assert_eq!(util_stats::snapshot().timer_context.cells_engaged, 0);
        for _ in 0..200 {
            record_firing(a, true);
        }
        assert_eq!(util_stats::snapshot().timer_context.cells_engaged, 1);
        for _ in 0..200 {
            record_firing(b, false);
        }
        assert_eq!(util_stats::snapshot().timer_context.cells_engaged, 2);
        reset();
    }

    #[test]
    fn cell_key_packs_every_bucket_distinctly() {
        let mut seen = [false; CELLS];
        for pending in [0usize, 1, 2, 3, 100] {
            for inflight in [0u32, 1, 5] {
                for streak in [0u32, 1, 2, 3, 7, 8, 50] {
                    for restarted in [false, true] {
                        let CellKey(i) = cell_key(pending, inflight, streak, restarted);
                        seen[i as usize] = true;
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every cell index is reachable");
    }
}
