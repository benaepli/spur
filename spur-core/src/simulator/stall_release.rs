//! A release for runs the stall cap treats. Half of them, drawn by a salt of
//! their own, do not end at their first stall: the client operations still
//! in progress are settled for the plan's dependency purposes, so every
//! planned event ordered behind them becomes ready and issues on the steps
//! that follow, and the settlement itself is a progress mark that re-arms
//! the clock. The run then ends by plan completion or at its next stall,
//! which ends it as the stall cap ends any treated run. The invocation of a
//! settled operation stays in the history without a response, and a real
//! response that arrives later is recorded as it would have been, so the
//! ground truth a linearizability check reads is untouched.
//!
//! The other half of the treated runs, and every run the stall cap does not
//! treat, behave as they do without the release. The cell is a pure
//! function of the run id.

use crate::simulator::run_phase;
use crate::simulator::stall_cap;
use crate::simulator::util_stats::StallReleaseCell;

/// Domain salt for the release split, so it is independent of the stall
/// cap's own split and of every other salted split.
pub const STALL_RELEASE_SALT: u64 = 0x_5354_4C52_454C_5345; // "STLRELSE"

/// Phases the release split is drawn over; phase zero is the cut half.
const RELEASE_PERIOD: i64 = 2;

/// The cell a run id lands in. Only a run the stall cap treats can be
/// released, so both probe streams are exempt as they are from the cap.
pub fn cell(run_id: i64) -> StallReleaseCell {
    if !stall_cap::is_treated(run_id) {
        StallReleaseCell::Exempt
    } else if run_phase::salted_phase(run_id, STALL_RELEASE_SALT, RELEASE_PERIOD) != 0 {
        StallReleaseCell::Release
    } else {
        StallReleaseCell::Cut
    }
}

/// Whether the run's first stall releases it instead of ending it.
pub fn is_treated(run_id: i64) -> bool {
    cell(run_id) == StallReleaseCell::Release
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::run_cap;
    use crate::simulator::timer_context;

    #[test]
    fn the_release_cell_is_half_of_the_stall_caps_treated_runs_and_nothing_else() {
        let n = 64_000i64;
        let mut treated = 0i64;
        let mut released = 0i64;
        for id in -n / 2..n / 2 {
            let c = cell(id);
            assert_eq!(is_treated(id), c == StallReleaseCell::Release);
            let probe = run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe;
            if !stall_cap::is_treated(id) {
                assert_eq!(c, StallReleaseCell::Exempt, "run {id}: only a treated run is split");
                continue;
            }
            assert!(!probe, "run {id}: a probe is never in the stall cap's treated cell");
            treated += 1;
            released += (c == StallReleaseCell::Release) as i64;
        }
        let share = released as f64 / treated as f64;
        assert!((share - 0.5).abs() < 0.02, "the release cell takes {share} of the treated runs");
    }
}
