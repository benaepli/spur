//! Per-run tag naming the session-global mechanisms that selected the run
//! and whether the placed-crash mechanism acted on it.
//!
//! Several mechanisms treat only part of a session's runs, which makes the
//! treated and untreated halves of one process an internal contrast under
//! identical conditions. Reading that contrast off run-id arithmetic works
//! only until a phase constant moves, and cannot express whether a selected
//! run was inert. The tag records both, so a consumer groups by the column
//! and never by the id.
//!
//! Bits are independent: a run can be selected by more than one mechanism.
//! A timer-context probe is placed like any other run, and whether a placed
//! run acted is a separate fact from whether it was selected, so no single
//! label could name what a run was.

use crate::simulator::fault_timing;
use crate::simulator::run_cap;
use crate::simulator::timer_context;

/// The run draws crash holds over the learned completed-run span.
pub const CRASH_PLACED: i32 = 1 << 0;
/// The run is a run-cap probe: it runs to the full budget uncapped and is
/// the only kind of run that feeds the cap learner.
pub const RUN_CAP_PROBE: i32 = 1 << 1;
/// The run is a timer-context probe: admission runs unsteered, so the cell
/// learner reads odds that carry no bias imprint.
pub const TIMER_STEER_OFF: i32 = 1 << 2;
/// A crash hold was actually drawn. A placed run below its scope's sample
/// floor, or one whose span is already spent, is selected and inert; only
/// this bit separates the two.
pub const CRASH_HOLD_DRAWN: i32 = 1 << 3;

/// The whole tag: what the run id selected, plus what the run did.
pub fn of(run_id: i64, crash_hold_drawn: bool) -> i32 {
    from_run_id(run_id) | if crash_hold_drawn { CRASH_HOLD_DRAWN } else { 0 }
}

/// The bits that follow from the run id alone.
pub fn from_run_id(run_id: i64) -> i32 {
    let mut v = 0;
    if fault_timing::is_placed(run_id) {
        v |= CRASH_PLACED;
    }
    if run_cap::is_probe(run_id) {
        v |= RUN_CAP_PROBE;
    }
    if timer_context::run_mode(run_id) == timer_context::RunMode::Probe {
        v |= TIMER_STEER_OFF;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    #[test]
    fn the_id_bits_agree_with_the_mechanisms_that_own_them() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        for id in [0i64, 1, 16, 32, 48, 64, -1, -16, -32] {
            let v = from_run_id(id);
            assert_eq!(v & CRASH_PLACED != 0, fault_timing::is_placed(id));
            assert_eq!(v & RUN_CAP_PROBE != 0, run_cap::is_probe(id));
            assert_eq!(
                v & TIMER_STEER_OFF != 0,
                timer_context::run_mode(id) == timer_context::RunMode::Probe
            );
            assert_eq!(v & CRASH_HOLD_DRAWN, 0, "the acted bit is not an id bit");
            assert_eq!(of(id, true), v | CRASH_HOLD_DRAWN);
            assert_eq!(of(id, false), v);
        }
    }

    #[test]
    fn the_populations_overlap_so_a_single_label_could_not_name_them() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        let mut timer_probe_placed = 0;
        let mut timer_probe_stock = 0;
        let mut cap_probe_placed = 0;
        for id in 0..64_000i64 {
            let v = from_run_id(id);
            if v & RUN_CAP_PROBE != 0 && v & CRASH_PLACED != 0 {
                cap_probe_placed += 1;
            }
            if v & TIMER_STEER_OFF == 0 {
                continue;
            }
            if v & CRASH_PLACED != 0 { timer_probe_placed += 1 } else { timer_probe_stock += 1 }
        }
        assert!(timer_probe_placed > 0, "no timer probe is placed, so the bits never overlap");
        assert!(timer_probe_stock > 0, "every timer probe is placed");
        assert_eq!(cap_probe_placed, 0, "a run-cap probe must never be placed");
    }
}
