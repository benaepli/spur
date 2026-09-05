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

use crate::simulator::client_anchor;
use crate::simulator::crash_phase;
use crate::simulator::fault_timing;
use crate::simulator::fresh_first;
use crate::simulator::ghost_absorber;
use crate::simulator::pair_order;
use crate::simulator::replay_corpus;
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
/// The run's placed crashes wait for a drawn phase of the victim's own
/// fan-out once their step hold expires, instead of competing at once.
pub const CRASH_PHASE: i32 = 1 << 9;
/// At a network step whose pick is a record from a sender that has crashed
/// at least once, the run takes instead the eligible record from that
/// sender to the same destination, sent by the same incarnation, with the
/// lowest send ordinal.
pub const PAIR_SEND_ORDER: i32 = 1 << 15;
/// The run holds client requests that become ready after its first crash
/// and issues each once it has waited a fixed number of steps, or earlier
/// when nothing else in the run can move.
pub const CLIENT_FANOUT_RELEASE: i32 = 1 << 18;
/// The run issues client requests that become ready after its first crash
/// at their ready step, and gives every record those requests cause the top
/// of the priority range.
pub const CLIENT_RUSH_PRIORITY: i32 = 1 << 14;
/// The run's planned crashes move to the live node that last took a
/// delivery whose sender was down or had restarted since sending.
pub const GHOST_ABSORBER_RETARGET: i32 = 1 << 19;
/// The run is a replay slot of a grid arm: it ran as a child of a corpus
/// parent when the arm held one, and fresh otherwise. Only a grid arm runs
/// slots, so the arm joins this bit through the run's attribution rather
/// than `from_run_id`; the bit is still a pure function of the run id.
pub const REPLAY_SLOT: i32 = 1 << 20;
/// A replay slot whose child replays the parent's schedule prefix rather
/// than only its plan. Joined by the arm like `REPLAY_SLOT`.
pub const REPLAY_PREFIX: i32 = 1 << 21;
/// At a network step whose draw fell on a record from a sender's dead
/// incarnation, the run takes instead an eligible record from that sender's
/// current incarnation to the same destination.
pub const FRESH_FIRST_PAIR: i32 = 1 << 24;

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
    if crash_phase::is_anchored(run_id) {
        v |= CRASH_PHASE;
    }
    if ghost_absorber::is_treated(run_id) {
        v |= GHOST_ABSORBER_RETARGET;
    }
    if fresh_first::is_treated(run_id) {
        v |= FRESH_FIRST_PAIR;
    }
    if pair_order::is_treated(run_id) {
        v |= PAIR_SEND_ORDER;
    }
    if client_anchor::is_treated(run_id) {
        v |= CLIENT_FANOUT_RELEASE;
    }
    if client_anchor::is_rushed(run_id) {
        v |= CLIENT_RUSH_PRIORITY;
    }
    v
}

/// The bits a grid arm joins to the tag of every run it issues. A slot whose
/// corpus held no parent ran fresh and still carries its bits, so the halves
/// are compared by id, never by whether a parent was available.
pub fn grid_arm_bits(run_id: i64) -> i32 {
    let mut v = 0;
    if replay_corpus::is_slot(run_id) {
        v |= REPLAY_SLOT;
    }
    if replay_corpus::is_prefix(run_id) {
        v |= REPLAY_PREFIX;
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
            assert_eq!(v & CRASH_PHASE != 0, crash_phase::is_anchored(id));
            assert_eq!(v & GHOST_ABSORBER_RETARGET != 0, ghost_absorber::is_treated(id));
            assert_eq!(v & FRESH_FIRST_PAIR != 0, fresh_first::is_treated(id));
            assert_eq!(v & PAIR_SEND_ORDER != 0, pair_order::is_treated(id));
            assert_eq!(v & CLIENT_FANOUT_RELEASE != 0, client_anchor::is_treated(id));
            assert_eq!(v & CLIENT_RUSH_PRIORITY != 0, client_anchor::is_rushed(id));
            assert_ne!(
                v & (CLIENT_FANOUT_RELEASE | CLIENT_RUSH_PRIORITY),
                CLIENT_FANOUT_RELEASE | CLIENT_RUSH_PRIORITY,
                "a run took both directions of the request-timing axis"
            );
            assert_eq!(v & CRASH_HOLD_DRAWN, 0, "the acted bit is not an id bit");
            assert_eq!(of(id, true), v | CRASH_HOLD_DRAWN);
            assert_eq!(of(id, false), v);
            assert_eq!(v & (REPLAY_SLOT | REPLAY_PREFIX), 0, "the slot bits are the arm's");
            let g = grid_arm_bits(id);
            assert_eq!(g & REPLAY_SLOT != 0, replay_corpus::is_slot(id));
            assert_eq!(g & REPLAY_PREFIX != 0, replay_corpus::is_prefix(id));
            assert_eq!(g & !(REPLAY_SLOT | REPLAY_PREFIX), 0, "the arm sets only its bits");
        }
    }

    #[test]
    fn the_slot_bits_spare_every_probe_and_leave_a_contrast() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        let n = 64_000i64;
        let mut slots = 0;
        let mut prefixes = 0;
        for id in 0..n {
            let v = from_run_id(id) | grid_arm_bits(id);
            if v & (RUN_CAP_PROBE | TIMER_STEER_OFF) != 0 {
                assert_eq!(v & REPLAY_SLOT, 0, "run {id}: a probe is a replay slot");
            }
            if v & REPLAY_PREFIX != 0 {
                assert_ne!(v & REPLAY_SLOT, 0, "run {id}: a prefix child outside a slot");
            }
            slots += (v & REPLAY_SLOT != 0) as i64;
            prefixes += (v & REPLAY_PREFIX != 0) as i64;
        }
        assert!(slots > 0 && slots < n, "the slot split leaves no contrast");
        assert!(prefixes > 0 && prefixes < slots, "the prefix split leaves no contrast");
    }

    #[test]
    fn the_populations_overlap_so_a_single_label_could_not_name_them() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        let mut timer_probe_placed = 0;
        let mut cap_probe_placed = 0;
        let mut ordinary_stock = 0;
        for id in 0..64_000i64 {
            let v = from_run_id(id);
            if v & TIMER_STEER_OFF != 0 && v & CRASH_PLACED != 0 {
                timer_probe_placed += 1;
            }
            if v & RUN_CAP_PROBE != 0 && v & CRASH_PLACED != 0 {
                cap_probe_placed += 1;
            }
            if v & RUN_CAP_PROBE == 0 && v & CRASH_PLACED == 0 {
                ordinary_stock += 1;
            }
        }
        let anchored_placed = (0..64_000i64)
            .filter(|&id| from_run_id(id) & (CRASH_PHASE | CRASH_PLACED) == CRASH_PHASE | CRASH_PLACED)
            .count();
        let placed = (0..64_000i64).filter(|&id| from_run_id(id) & CRASH_PLACED != 0).count();
        assert!(
            anchored_placed > 0 && anchored_placed < placed,
            "the anchor took {anchored_placed} of {placed} placed runs, leaving no contrast"
        );
        assert!(timer_probe_placed > 0, "no timer probe is placed, so the bits never overlap");
        assert_eq!(cap_probe_placed, 0, "a run-cap probe must never be placed");
        let retargeted = (0..64_000i64)
            .filter(|&id| from_run_id(id) & GHOST_ABSORBER_RETARGET != 0)
            .count();
        let retargeted_probes = (0..64_000i64)
            .filter(|&id| from_run_id(id) & (GHOST_ABSORBER_RETARGET | RUN_CAP_PROBE) == GHOST_ABSORBER_RETARGET | RUN_CAP_PROBE)
            .count();
        assert!(retargeted > 0 && retargeted < 64_000, "the retarget split leaves no contrast");
        assert_eq!(retargeted_probes, 0, "a run-cap probe must never be retargeted");
        let fresh_first_runs = (0..64_000i64)
            .filter(|&id| from_run_id(id) & FRESH_FIRST_PAIR != 0)
            .count();
        let fresh_first_probes = (0..64_000i64)
            .filter(|&id| {
                from_run_id(id) & (FRESH_FIRST_PAIR | RUN_CAP_PROBE) == FRESH_FIRST_PAIR | RUN_CAP_PROBE
            })
            .count();
        assert!(
            fresh_first_runs > 0 && fresh_first_runs < 64_000,
            "the fresh-first split leaves no contrast"
        );
        assert_eq!(fresh_first_probes, 0, "a run-cap probe must never prefer fresh records");
        let pair_order_runs = (0..64_000i64)
            .filter(|&id| from_run_id(id) & PAIR_SEND_ORDER != 0)
            .count();
        let pair_order_probes = (0..64_000i64)
            .filter(|&id| {
                let v = from_run_id(id);
                v & PAIR_SEND_ORDER != 0 && v & (RUN_CAP_PROBE | TIMER_STEER_OFF) != 0
            })
            .count();
        assert!(
            pair_order_runs > 0 && pair_order_runs < 64_000,
            "the pair-order split leaves no contrast"
        );
        assert_eq!(pair_order_probes, 0, "a probe must never take records in send order");
        let anchor_runs = (0..64_000i64)
            .filter(|&id| from_run_id(id) & CLIENT_FANOUT_RELEASE != 0)
            .count();
        let anchor_probes = (0..64_000i64)
            .filter(|&id| {
                let v = from_run_id(id);
                v & CLIENT_FANOUT_RELEASE != 0 && v & (RUN_CAP_PROBE | TIMER_STEER_OFF) != 0
            })
            .count();
        assert!(
            anchor_runs > 0 && anchor_runs < 64_000,
            "the client-anchor split leaves no contrast"
        );
        assert_eq!(anchor_probes, 0, "a probe must never hold client requests");
        let rush_runs = (0..64_000i64)
            .filter(|&id| from_run_id(id) & CLIENT_RUSH_PRIORITY != 0)
            .count();
        let rush_probes = (0..64_000i64)
            .filter(|&id| {
                let v = from_run_id(id);
                v & CLIENT_RUSH_PRIORITY != 0 && v & (RUN_CAP_PROBE | TIMER_STEER_OFF) != 0
            })
            .count();
        let rush_and_hold = (0..64_000i64)
            .filter(|&id| {
                from_run_id(id) & (CLIENT_RUSH_PRIORITY | CLIENT_FANOUT_RELEASE)
                    == CLIENT_RUSH_PRIORITY | CLIENT_FANOUT_RELEASE
            })
            .count();
        assert!(
            rush_runs > 0 && rush_runs < anchor_runs,
            "the rush split leaves no contrast against the hold"
        );
        assert_eq!(rush_probes, 0, "a probe must never rush client requests");
        assert_eq!(rush_and_hold, 0, "the two directions must never meet on a run");
        assert!(ordinary_stock > 0, "no stock run that is not a probe, so there is no control");
    }
}
