//! Score damping for timer firings that have been changing nothing.
//!
//! A timer whose woken segment left its node's state alone, again and again at
//! the same resume point, is unlikely to change anything the next time either.
//! Once a run has spent a configured fraction of its step budget, such a
//! timer's score is multiplied down, so a timer that does change something -
//! or a candidate in another queue - is more likely to be taken. Below that
//! fraction every score is left exactly as it was, so the order a run starts
//! with is the unsteered one and only the tail of a run heading for budget
//! exhaustion is steered.
//!
//! The estimate is a Beta posterior mean per (node, resume vertex, pending
//! delivery) key over the firings this thread has already seen, so nothing
//! here needs to know what a timer means to the spec under test. A key with no
//! observations has posterior mean 1/2 and is left undamped; a key that never
//! acts decays towards zero as evidence accumulates.

use crate::compiler::cfg::Vertex;
use crate::simulator::util_stats;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// The fraction of the step budget at or past which damping applies. 1.0 is
/// "never", which is the same scoring as having no mechanism at all.
const OFF: f64 = 1.0;

/// Guards the table against a spec with an unexpected number of timer resume
/// points; the key space is nodes times resume vertices times two.
const KEY_CAP: usize = 4096;

static LATE_FRACTION: AtomicU64 = AtomicU64::new(OFF.to_bits());

thread_local! {
    static EFFECTS: RefCell<HashMap<(usize, Vertex, bool), (u32, u32)>> =
        RefCell::new(HashMap::new());
    static STEP_BUDGET: Cell<i32> = const { Cell::new(0) };
}

/// Set the budget fraction for this session. Anything at or above 1.0 turns
/// the mechanism off.
pub fn set_late_fraction(fraction: f64) {
    LATE_FRACTION.store(fraction.to_bits(), Ordering::Relaxed);
}

#[inline]
fn late_fraction() -> f64 {
    f64::from_bits(LATE_FRACTION.load(Ordering::Relaxed))
}

/// Whether timer firings should be learned from at all.
#[inline]
pub fn enabled() -> bool {
    late_fraction() < OFF
}

/// One run is starting on this thread with `step_budget` steps to spend. The
/// learned table is kept across the runs of a session on purpose: the estimate
/// a run is steered by is the evidence from the runs before it.
pub fn begin_run(step_budget: i32) {
    STEP_BUDGET.set(step_budget);
}

/// Account for one timer firing that woke a record at `vertex` on `node`, and
/// whether the woken segment changed the node's state.
pub fn note_firing(node: usize, vertex: Vertex, inflight: bool, acted: bool) {
    if !enabled() {
        return;
    }
    EFFECTS.with_borrow_mut(|table| {
        if let Some(e) = table.get_mut(&(node, vertex, inflight)) {
            e.0 = e.0.saturating_add(1);
            if acted {
                e.1 = e.1.saturating_add(1);
            }
        } else if table.len() < KEY_CAP {
            table.insert((node, vertex, inflight), (1, u32::from(acted)));
        }
    });
}

/// Whether the run at `step` is far enough into its budget for damping to
/// apply. Callers check this before reading anything else off the state, so a
/// disabled session pays one atomic load per timer candidate.
#[inline]
pub fn armed(step: i32) -> bool {
    let fraction = late_fraction();
    if fraction >= OFF {
        return false;
    }
    let budget = STEP_BUDGET.get();
    let late = budget > 0 && f64::from(step) >= fraction * f64::from(budget);
    if !late {
        util_stats::record_timer_damping_gated();
    }
    late
}

/// The factor a timer candidate's score is multiplied by, in (0, 1]. Only a
/// timer whose node has no delivery pending is damped: with a delivery in
/// flight the firing races it, and the race is the interesting one whatever
/// the timer did on its own before.
pub fn score_multiplier(node: usize, vertex: Vertex, inflight: bool) -> f64 {
    if inflight {
        util_stats::record_timer_damping(false);
        return 1.0;
    }
    let (fired, acted) = EFFECTS
        .with_borrow(|table| table.get(&(node, vertex, inflight)).copied())
        .unwrap_or((0, 0));
    let acts = f64::from(acted + 1) / f64::from(fired + 2);
    let multiplier = (2.0 * acts).min(1.0);
    util_stats::record_timer_damping(multiplier < 1.0);
    multiplier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    fn clear() {
        EFFECTS.with_borrow_mut(|t| t.clear());
    }

    #[test]
    fn off_by_default_and_armed_only_late() {
        let _serial = config_override::exclusive_session();
        clear();
        begin_run(100);
        assert!(!enabled());
        assert!(!armed(99));

        set_late_fraction(0.5);
        assert!(enabled());
        assert!(!armed(49));
        assert!(armed(50));
        set_late_fraction(OFF);
    }

    #[test]
    fn an_inert_key_is_damped_and_an_acting_one_is_not() {
        let _serial = config_override::exclusive_session();
        clear();
        set_late_fraction(0.5);

        assert_eq!(score_multiplier(0, 7, false), 1.0, "no evidence, no damping");
        for _ in 0..8 {
            note_firing(0, 7, false, false);
            note_firing(0, 9, false, true);
        }
        assert!(score_multiplier(0, 7, false) < 0.25);
        assert_eq!(score_multiplier(0, 9, false), 1.0);
        assert_eq!(score_multiplier(0, 7, true), 1.0, "a pending delivery is never damped");

        set_late_fraction(OFF);
        clear();
    }

    #[test]
    fn nothing_is_learned_while_off() {
        let _serial = config_override::exclusive_session();
        clear();
        set_late_fraction(OFF);
        note_firing(1, 3, false, false);
        set_late_fraction(0.5);
        assert_eq!(score_multiplier(1, 3, false), 1.0);
        set_late_fraction(OFF);
        clear();
    }
}
