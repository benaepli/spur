//! A restart arms a trigger; the first entry from the restarted node's dead
//! incarnation that changes a live peer's state releases the other nodes'
//! still-held planned crashes to the next step.
//!
//! A placed crash draws its target step on a clock that knows nothing about
//! the other nodes' restarts, so a crash landing right after a peer has
//! reacted to a record from a dead incarnation is reached by chance. On a
//! salted half of the placed runs, a restart of node p at step s that finds
//! some other node's planned crash still held arms a trigger on p. Nothing
//! moves at arming. The first message entry after s at a live node v other
//! than p, sent by a dead incarnation of p, that writes v's state sets every
//! other held crash's target to the next step and disarms the trigger. If
//! no such entry arrives by s + Q the trigger expires and every crash keeps
//! its target. A hold only ever shortens, nothing is drawn, and the
//! untreated half keeps every random stream in step with the treated one.
//!
//! Q is a quantile of the ghost lag learned per backup-budget scope: at
//! every message entry from a dead incarnation of a live origin, at a live
//! destination other than the origin, the lag is the entry step less the
//! origin's last restart step. Only run-cap probes feed the learner, which
//! are never placed. The bound is read at the winning cell's upper edge
//! once a scope holds enough samples, recomputed at doubling checkpoints.
//! Below the floor no trigger is armed.
//!
//! Half of the releasing runs, under a salt of their own, release exactly
//! one crash at a firing on v: v's own held crash if it has one, else the
//! first held crash the absorber ranking would move onto v, else the first
//! held crash by index, which is then applied to v. When none of the three
//! applies nothing is released and the trigger stays armed.

use crate::simulator::core::state::SendLedger;
use crate::simulator::fault_timing;
use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::util_stats::{self, GhostReleaseCell, LagGauges};
use dashmap::DashMap;
use std::sync::LazyLock;

/// Salt for the half of the placed runs that release. Distinct from every
/// other split of a session.
const RELEASE_SALT: u64 = 0x_4748_4F53_5452_454C; // "GHOSTREL"

/// Salt for the half of the releasing runs that release one crash.
const SINGLE_SALT: u64 = 0x_4748_4F53_544F_4E45; // "GHOSTONE"

/// Quantile of the lag distribution the run's bound is read from.
pub const BOUND_QUANTILE: f64 = 0.75;

/// Lag samples a scope must accumulate before its bound takes effect.
const MIN_SAMPLES: u64 = 200;

/// Histogram cells per scope. A lag is a count of steps between a restart
/// and an entry from the restarted node's dead incarnation, so a cell is one
/// step wide and the last cell takes every lag past the range.
const HIST_CELLS: usize = 256;

/// The coin that puts a placed run in the releasing half.
fn release_coin(run_id: i64) -> bool {
    run_phase::salted_phase(run_id, RELEASE_SALT, 2) == 1
}

/// The coin that puts a releasing run in the single-release half.
fn single_coin(run_id: i64) -> bool {
    run_phase::salted_phase(run_id, SINGLE_SALT, 2) == 1
}

/// Whether this run's restarts arm the release trigger, read from the run
/// id's placement coin. Drawn over the placed runs, so run-cap probes are
/// exempt.
pub fn is_treated(run_id: i64) -> bool {
    fault_timing::is_placed(run_id) && release_coin(run_id)
}

/// Whether a firing on this run releases one crash, read from the run id's
/// placement coin. Never set without `is_treated`.
pub fn is_single(run_id: i64) -> bool {
    is_treated(run_id) && single_coin(run_id)
}

/// The cell a run occupies given whether it is placed - by its crash arm,
/// which a learner may have chosen over the id's coin - and its own coins:
/// the releasing halves, the placed runs that do not release, or outside
/// the placed posture. A run that draws no holds has nothing a firing could
/// release, so it is outside the cells whatever its coins say.
pub fn cell_of(placed: bool, run_id: i64) -> GhostReleaseCell {
    if !placed {
        GhostReleaseCell::Unplaced
    } else if !release_coin(run_id) {
        GhostReleaseCell::Untreated
    } else if single_coin(run_id) {
        GhostReleaseCell::Single
    } else {
        GhostReleaseCell::ReleaseAll
    }
}

/// `cell_of` with placement read from the run id's coin.
pub fn cell(run_id: i64) -> GhostReleaseCell {
    cell_of(fault_timing::is_placed(run_id), run_id)
}

/// Whether this run's ghost lags may feed the learner: run-cap probes only,
/// which are in the stock posture at every placement fraction.
pub fn feeds_learner(run_id: i64) -> bool {
    run_cap::is_probe(run_id)
}

/// The quantile of a lag histogram, Laplace-smoothed and read at the
/// winning cell's upper edge so bucketing only ever rounds up, bounded by
/// the scope's budget. None on an empty histogram.
fn estimate(hist: &[u32; HIST_CELLS], quantile: f64, backup: i32) -> Option<i32> {
    let samples: u64 = hist.iter().map(|&n| n as u64).sum();
    if samples == 0 {
        return None;
    }
    let denom = (samples + 2) as f64;
    let mut cum: u64 = 0;
    for (c, &n) in hist.iter().enumerate() {
        cum += n as u64;
        if (cum + 1) as f64 / denom >= quantile {
            return Some((c as i64).min(backup as i64) as i32);
        }
    }
    None
}

struct ScopeAccum {
    /// Cell `c` counts the lags equal to `c`, the last cell those at or past
    /// it.
    hist: [u32; HIST_CELLS],
    /// Lags folded into the histogram.
    samples: u64,
    /// The histogram as it stood at the last checkpoint, which every read
    /// between checkpoints is taken from; None until the first checkpoint
    /// is crossed.
    frozen: Option<[u32; HIST_CELLS]>,
    /// Sample count at which the histogram is next frozen; doubles after
    /// each freeze.
    next_checkpoint: u64,
}

impl ScopeAccum {
    fn new() -> Self {
        Self {
            hist: [0; HIST_CELLS],
            samples: 0,
            frozen: None,
            next_checkpoint: MIN_SAMPLES,
        }
    }

    /// The quantile the frozen histogram supports, or None while the scope
    /// has not crossed its first checkpoint.
    fn read(&self, quantile: f64, backup: i32) -> Option<i32> {
        self.frozen
            .as_ref()
            .and_then(|hist| estimate(hist, quantile, backup))
    }
}

static TABLE: LazyLock<DashMap<i32, ScopeAccum>> = LazyLock::new(DashMap::new);

/// The learned lag quantile for a scope, or None while the scope is below
/// its sample floor. Constant between checkpoints.
pub fn lag_quantile(backup: i32, quantile: f64) -> Option<i32> {
    TABLE
        .get(&backup)
        .and_then(|acc| acc.read(quantile, backup))
}

/// The bound a run in this scope waits under: the learned lag at
/// `BOUND_QUANTILE`.
pub fn bound(backup: i32) -> Option<i32> {
    lag_quantile(backup, BOUND_QUANTILE)
}

/// Fold one ghost lag from a run that feeds the learner into its scope.
pub fn merge_probe_lag(backup: i32, lag: i32) {
    util_stats::record_ghost_release_lag_sample();
    let mut acc = TABLE.entry(backup).or_insert_with(ScopeAccum::new);
    let cell = (lag.max(0) as usize).min(HIST_CELLS - 1);
    acc.hist[cell] = acc.hist[cell].saturating_add(1);
    acc.samples += 1;
    if acc.samples >= acc.next_checkpoint {
        acc.frozen = Some(acc.hist);
        acc.next_checkpoint = acc.next_checkpoint.saturating_mul(2);
        drop(acc);
        publish_gauges();
    }
}

/// Scale every scope's mass by `factor`, dropping scopes that reach zero,
/// so stale phases of a long exploration lose their vote. The frozen
/// histogram and the next checkpoint are left alone: shrinking the sample
/// count delays the next crossing, and the freeze there reads the
/// decay-weighted histogram.
pub fn decay(factor: f64) {
    let factor = factor.clamp(0.0, 1.0);
    TABLE.retain(|_, acc| {
        for n in acc.hist.iter_mut() {
            *n = ((*n as f64) * factor).floor() as u32;
        }
        acc.samples = ((acc.samples as f64) * factor).floor() as u64;
        acc.frozen.is_some() || acc.samples > 0 || acc.hist.iter().any(|&n| n > 0)
    });
    publish_gauges();
}

/// Clear the table so explorer sessions in one process do not share
/// bounds.
pub fn reset() {
    TABLE.clear();
    util_stats::set_ghost_release_learned(0, LagGauges::default());
}

fn publish_gauges() {
    let mut engaged: u64 = 0;
    let mut widest: Option<(i32, LagGauges)> = None;
    for e in TABLE.iter() {
        let backup = *e.key();
        let acc = e.value();
        let Some(p75) = acc.read(0.75, backup) else {
            continue;
        };
        engaged += 1;
        if widest.is_none_or(|(k, _)| backup > k) {
            let gauge = |q: f64| acc.read(q, backup).map_or(0, |v| v.max(0) as u64);
            widest = Some((
                backup,
                LagGauges {
                    p50: gauge(0.5),
                    p75: p75.max(0) as u64,
                    p90: gauge(0.9),
                },
            ));
        }
    }
    util_stats::set_ghost_release_learned(engaged, widest.map_or(LagGauges::default(), |(_, g)| g));
}

/// The trigger a restart arms on a releasing run: the restarted node, the
/// step it restarted at, and the step past which no entry releases
/// anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trigger {
    pub origin: usize,
    pub restart_step: i32,
    pub expires: i32,
}

/// One run's release state. `released_mask` has a bit per node, up to the
/// first 64, whose planned crash a firing released and that has not yet
/// been applied, with `ghost_node` the node whose entry fired the trigger
/// last. `forced_victim` names a released crash and the node it is to be
/// applied to instead of its planned victim.
#[derive(Clone, Debug, Default)]
pub struct RunState {
    pub cell: GhostReleaseCell,
    /// The scope's learned bound at run start, or None below the floor.
    pub bound: Option<i32>,
    /// The run is a run-cap probe, so its ghost lags feed the learner.
    pub feeds_learner: bool,
    /// The backup-budget scope the run's lags are folded into.
    pub scope: i32,
    /// The run's placed crashes wait for a fan-out phase past their hold.
    pub anchored: bool,
    pub trigger: Option<Trigger>,
    pub released_mask: u64,
    pub ghost_node: Option<usize>,
    pub forced_victim: Option<(usize, usize)>,
    /// Crashes applied in this run so far.
    pub crashes_applied: u32,
    /// The node and step of the most recent applied crash a firing had
    /// released.
    pub last_released_apply: Option<(usize, i32)>,
}

impl RunState {
    /// The state a run starts with: its cell given whether its crash arm
    /// draws holds, the bound its scope has learned by now, whether it
    /// feeds the learner, and whether its crashes wait for a fan-out phase.
    pub fn at_run_start(run_id: i64, backup: i32, placed: bool, anchored: bool) -> Self {
        Self {
            cell: cell_of(placed, run_id),
            bound: bound(backup),
            feeds_learner: feeds_learner(run_id),
            scope: backup,
            anchored,
            ..Self::default()
        }
    }

    /// Whether restarts on this run arm the release trigger.
    pub fn releases(&self) -> bool {
        matches!(
            self.cell,
            GhostReleaseCell::ReleaseAll | GhostReleaseCell::Single
        )
    }

    /// Whether a firing on this run releases one crash.
    pub fn single(&self) -> bool {
        self.cell == GhostReleaseCell::Single
    }
}

/// Whether node `n` has a planned crash whose hold is still ahead of `step`.
pub fn held(hold_until: &[i32], ledgers: &[SendLedger], n: usize, step: i32) -> bool {
    ledgers.get(n).is_some_and(|l| l.crash_pending > 0)
        && hold_until.get(n).is_some_and(|&h| h > step)
}

/// Whether some node other than `origin` has a held crash at `step`.
pub fn any_other_held(
    hold_until: &[i32],
    ledgers: &[SendLedger],
    origin: usize,
    step: i32,
) -> bool {
    (0..hold_until.len()).any(|n| n != origin && held(hold_until, ledgers, n, step))
}

/// Set node `n`'s hold to the step after `step` unless it is already there
/// or earlier; true when the hold moved.
pub fn release(hold_until: &mut [i32], n: usize, step: i32) -> bool {
    let next = step.saturating_add(1);
    match hold_until.get_mut(n) {
        Some(h) if *h > next => {
            *h = next;
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    fn feed(n: usize, backup: i32, lag: i32) {
        for _ in 0..n {
            merge_probe_lag(backup, lag);
        }
    }

    #[test]
    fn the_releasing_half_is_half_the_placed_runs_and_the_single_half_of_that() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        let n = 64_000i64;
        let placed = (0..n).filter(|&id| fault_timing::is_placed(id)).count();
        let treated = (0..n).filter(|&id| is_treated(id)).count();
        let single = (0..n).filter(|&id| is_single(id)).count();
        let share = treated as f64 / placed as f64;
        assert!(
            (share - 0.5).abs() < 0.02,
            "the releasing share of placed runs is {share}"
        );
        let nested = single as f64 / treated as f64;
        assert!(
            (nested - 0.5).abs() < 0.02,
            "the single share of releasing runs is {nested}"
        );
        assert_eq!(
            (0..n)
                .filter(|&id| is_treated(id) && !fault_timing::is_placed(id))
                .count(),
            0,
            "an unplaced run must never release"
        );
        assert_eq!(
            (0..n)
                .filter(|&id| is_treated(id) && run_cap::is_probe(id))
                .count(),
            0,
            "a run-cap probe must never release"
        );
        assert_eq!(
            (0..n)
                .filter(|&id| is_single(id) && !is_treated(id))
                .count(),
            0,
            "the single half lies inside the releasing half"
        );
        // Neither split may follow another split of the same session.
        let with_anchor = (0..n)
            .filter(|&id| fault_timing::is_placed(id))
            .filter(|&id| is_treated(id) == crate::simulator::crash_phase::is_anchored(id))
            .count();
        let overlap = with_anchor as f64 / placed as f64;
        assert!(
            (overlap - 0.5).abs() < 0.03,
            "releasing follows the anchor split at {overlap}"
        );
        let with_retarget = (0..n)
            .filter(|&id| fault_timing::is_placed(id))
            .filter(|&id| is_treated(id) == crate::simulator::ghost_absorber::is_treated(id))
            .count();
        let overlap = with_retarget as f64 / placed as f64;
        assert!(
            (overlap - 0.5).abs() < 0.03,
            "releasing follows the retarget split at {overlap}"
        );
        let with_release = (0..n)
            .filter(|&id| is_treated(id))
            .filter(|&id| is_single(id) == (run_phase::phase(id, 2) == 1))
            .count();
        let overlap = with_release as f64 / treated as f64;
        assert!(
            (overlap - 0.5).abs() < 0.03,
            "the single half follows the plain phase at {overlap}"
        );
        for id in 0..n {
            let c = cell(id);
            assert_eq!(c == GhostReleaseCell::Single, is_single(id), "run {id}");
            assert_eq!(
                c == GhostReleaseCell::ReleaseAll,
                is_treated(id) && !is_single(id)
            );
            assert_eq!(
                c == GhostReleaseCell::Untreated,
                fault_timing::is_placed(id) && !is_treated(id)
            );
            assert_eq!(
                c == GhostReleaseCell::Unplaced,
                !fault_timing::is_placed(id)
            );
            assert_eq!(feeds_learner(id), run_cap::is_probe(id));
        }
    }

    #[test]
    fn the_bound_is_the_upper_cell_edge_at_the_quantile_and_constant_between_checkpoints() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(199, 6000, 12);
        assert_eq!(bound(6000), None, "199 samples are under the floor");
        feed(1, 6000, 12);
        assert_eq!(bound(6000), Some(12));
        assert_eq!(lag_quantile(6000, 0.5), Some(12));
        assert_eq!(lag_quantile(6000, 0.9), Some(12));
        // Between checkpoints the bound does not move.
        feed(150, 6000, 40);
        assert_eq!(bound(6000), Some(12), "350 samples sit between checkpoints");
        // At 400 the freeze reads the mixture: 200 at 12 and 200 at 40,
        // whose median is the 12 cell and whose 0.75 and 0.9 quantiles are
        // the 40 cell.
        feed(50, 6000, 40);
        assert_eq!(lag_quantile(6000, 0.5), Some(12));
        assert_eq!(bound(6000), Some(40));
        assert_eq!(lag_quantile(6000, 0.9), Some(40));
        assert_eq!(bound(1500), None, "another scope stays unengaged");
        // Three in four at 5 and one in four at 30: the smoothed 0.75
        // quantile reaches the 30 cell only once its mass passes the line.
        reset();
        feed(140, 6000, 5);
        feed(60, 6000, 30);
        assert_eq!(bound(6000), Some(30));
        assert_eq!(lag_quantile(6000, 0.5), Some(5));
        reset();
        feed(160, 6000, 5);
        feed(40, 6000, 30);
        assert_eq!(bound(6000), Some(5));
        assert_eq!(lag_quantile(6000, 0.9), Some(30));
        reset();
    }

    #[test]
    fn a_lag_past_the_range_lands_in_the_last_cell_and_the_bound_is_bounded_by_the_budget() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(200, 6000, 5_000);
        assert_eq!(bound(6000), Some(HIST_CELLS as i32 - 1));
        reset();
        feed(200, 100, 5_000);
        assert_eq!(
            bound(100),
            Some(100),
            "the bound never exceeds the scope's budget"
        );
        reset();
    }

    #[test]
    fn decay_delays_the_next_checkpoint_and_reset_empties() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(200, 6000, 12);
        assert_eq!(bound(6000), Some(12));
        decay(0.5);
        assert_eq!(bound(6000), Some(12), "the bound survives decay unchanged");
        feed(300, 6000, 40);
        assert_eq!(
            bound(6000),
            Some(40),
            "the delayed freeze follows fresher lags"
        );
        reset();
        assert_eq!(bound(6000), None);
    }

    #[test]
    fn the_gauges_follow_the_widest_engaged_scope() {
        let _serial = config_override::exclusive_session();
        reset();
        let zero = util_stats::snapshot().crash_place.ghost_release;
        assert_eq!(
            (
                zero.scopes_engaged,
                zero.lag_p50,
                zero.lag_p75,
                zero.lag_p90
            ),
            (0, 0, 0, 0)
        );
        feed(200, 1500, 8);
        feed(100, 6000, 21);
        feed(60, 6000, 33);
        feed(40, 6000, 47);
        let s = util_stats::snapshot().crash_place.ghost_release;
        assert_eq!(
            (s.scopes_engaged, s.lag_p50, s.lag_p75, s.lag_p90),
            (2, 21, 33, 47),
            "the gauges read the widest scope"
        );
        reset();
        let z = util_stats::snapshot().crash_place.ghost_release;
        assert_eq!(
            (z.scopes_engaged, z.lag_p50, z.lag_p75, z.lag_p90),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn a_run_below_the_floor_starts_with_no_bound_and_the_cells_are_the_ids() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let treated = (0..100_000i64).find(|&id| is_treated(id)).unwrap();
        let s = RunState::at_run_start(treated, 6000, true, true);
        assert_eq!(s.bound, None);
        assert!(s.releases());
        assert!(s.anchored);
        assert_eq!(s.cell, cell(treated));
        assert_eq!(s.scope, 6000);
        assert!(!s.feeds_learner);
        assert_eq!(s.trigger, None);
        assert_eq!(s.forced_victim, None);
        // A crash arm that draws no holds leaves the run outside the cells
        // whatever its coins say.
        let stock_arm = RunState::at_run_start(treated, 6000, false, false);
        assert_eq!(stock_arm.cell, GhostReleaseCell::Unplaced);
        assert!(!stock_arm.releases());
        feed(200, 6000, 9);
        let s = RunState::at_run_start(treated, 6000, true, false);
        assert_eq!(s.bound, Some(9));
        assert!(!s.anchored);
        let probe = (0..100_000i64).find(|&id| run_cap::is_probe(id)).unwrap();
        let p = RunState::at_run_start(probe, 6000, false, false);
        assert!(p.feeds_learner && !p.releases() && !p.single());
        assert_eq!(p.cell, GhostReleaseCell::Unplaced);
        let unplaced_coin = (0..100_000i64)
            .find(|&id| !fault_timing::is_placed(id) && !run_cap::is_probe(id))
            .unwrap();
        assert_eq!(cell(unplaced_coin), GhostReleaseCell::Unplaced);
        assert_ne!(
            cell_of(true, unplaced_coin),
            GhostReleaseCell::Unplaced,
            "a learner-placed run takes the cell its coins name"
        );
        reset();
    }

    #[test]
    fn a_release_moves_a_hold_to_the_next_step_and_never_later() {
        let mut ledgers = [SendLedger::default(); 4];
        ledgers[1].crash_pending = 1;
        ledgers[2].crash_pending = 1;
        // Node 3 has a far target but no queued crash; node 2's hold has
        // already passed.
        let mut holds = [0, 500, 90, 800];
        assert!(held(&holds, &ledgers, 1, 100));
        assert!(!held(&holds, &ledgers, 2, 100), "a passed hold is not held");
        assert!(
            !held(&holds, &ledgers, 3, 100),
            "no queued crash, nothing held"
        );
        assert!(
            !held(&holds, &ledgers, 1, 500),
            "a hold at the step is not ahead of it"
        );
        assert!(any_other_held(&holds, &ledgers, 0, 100));
        assert!(
            !any_other_held(&holds, &ledgers, 1, 100),
            "the origin's own hold is not another's"
        );
        assert!(release(&mut holds, 1, 100));
        assert_eq!(holds, [0, 101, 90, 800]);
        assert!(
            !release(&mut holds, 1, 100),
            "a hold at the next step stays"
        );
        assert!(
            !release(&mut holds, 2, 100),
            "a passed hold is never lengthened"
        );
        assert_eq!(holds, [0, 101, 90, 800]);
        assert!(
            !release(&mut holds, 9, 100),
            "a node past the table is left alone"
        );
    }
}
