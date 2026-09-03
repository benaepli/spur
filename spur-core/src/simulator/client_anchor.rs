//! Deferral of client requests that become ready after a run's first crash.
//!
//! A delivery crosses a fault when its sender is down at delivery time or
//! has come back from a crash since sending. When the node that takes such
//! a delivery writes state and answers with a message to every peer, the
//! network holds a full fan-out none of which has landed; a client request
//! issued at that moment competes with that fan-out for the next steps. The
//! plan decides when its client requests become ready, and on most runs a
//! request that becomes ready after the first crash is issued long before or
//! long after such a window.
//!
//! On the treated half of the runs a client request that becomes ready after
//! the run's first crash is held instead of issued, and is issued once it has
//! waited a fixed number of steps, or earlier when nothing else in the run
//! can move, so no run stalls or ends on a request that was never issued.
//! Requests ready before the first crash are issued when they become ready
//! on both halves. Fan-out windows are detected and counted on both halves
//! and issue nothing.
//!
//! The treated half is drawn under a salt of its own, so the split is
//! independent of every other split of a session. Probes are never treated:
//! run-cap probes feed the length learners, and timer-context probes must
//! run unsteered.

use crate::simulator::core::state::{HandlerTrigger, SendLedger};
use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::timer_context;
use std::collections::VecDeque;

/// Salt for the treated half. Distinct from every other split of a session.
pub const CLIENT_ANCHOR_SALT: u64 = 0x_434C_4E54_4143_4852; // "CLNTACHR"

/// A held request is issued once its ready step lies this many steps or
/// more behind the current step.
pub const EXPIRY_STEPS: i32 = 32;

/// Whether this run holds its post-crash client requests.
pub fn is_treated(run_id: i64) -> bool {
    !run_cap::is_probe(run_id)
        && timer_context::run_mode(run_id) != timer_context::RunMode::Probe
        && run_phase::salted_phase(run_id, CLIENT_ANCHOR_SALT, 2) == 1
}

/// Whether a window opened at `step`: some server among the first `servers`
/// ledgers took a fault-crossing delivery at this step that wrote its state,
/// and the handler it woke sent to every peer with all of those sends still
/// in the air. `servers` includes the node itself, so a full fan-out is
/// `servers - 1` sends; a single server has no peers and never opens one.
pub fn fanout_window(ledgers: &[SendLedger], servers: usize, step: i32) -> bool {
    let peers = servers.saturating_sub(1) as u32;
    if peers == 0 {
        return false;
    }
    ledgers.iter().take(servers).any(|l| {
        l.trigger == HandlerTrigger::Delivery
            && l.last_ghost_step == step
            && l.last_ghost_acted
            && l.recent >= peers
            && l.in_flight >= l.recent
    })
}

/// One run's switch. `enabled` is set on the treated half of generated
/// runs and never on a plan run from a file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunState {
    pub enabled: bool,
}

/// Why a held request left the queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Release {
    /// Waited past `EXPIRY_STEPS`.
    Expiry,
    /// Nothing else in the run could move.
    DryQueue,
}

/// A request handed back for issue, with the step it became ready at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Released<T> {
    pub item: T,
    pub ready_step: i32,
    pub release: Release,
}

/// What a window found in the queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Firing {
    /// This was the run's first window.
    pub first: bool,
    /// Requests held when the window opened.
    pub held: usize,
}

/// The held requests of one run in ready order. A request is pending work
/// from the moment it is held until it is taken for issue, so a run must
/// not be read as finished or stuck while the queue holds anything.
#[derive(Clone, Debug)]
pub struct HoldQueue<T> {
    held: VecDeque<(T, i32)>,
    firings: u32,
}

impl<T> Default for HoldQueue<T> {
    fn default() -> Self {
        Self {
            held: VecDeque::new(),
            firings: 0,
        }
    }
}

impl<T> HoldQueue<T> {
    /// Hold a request that became ready at `ready_step`.
    pub fn hold(&mut self, item: T, ready_step: i32) {
        self.held.push_back((item, ready_step));
    }

    /// Requests held.
    pub fn held(&self) -> usize {
        self.held.len()
    }

    /// The work the run still owes.
    pub fn pending(&self) -> usize {
        self.held.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    /// Windows seen so far in this run.
    pub fn firings(&self) -> u32 {
        self.firings
    }

    /// A window opened: count it and leave every held request where it is.
    pub fn fire(&mut self) -> Firing {
        self.firings += 1;
        Firing {
            first: self.firings == 1,
            held: self.held.len(),
        }
    }

    /// The requests to issue at `step`: every request whose wait has run
    /// past `EXPIRY_STEPS`, in ready order.
    pub fn take_due(&mut self, step: i32) -> Vec<Released<T>> {
        let mut out = Vec::new();
        while let Some((_, ready_step)) = self.held.front()
            && ready_step + EXPIRY_STEPS < step
        {
            let (item, ready_step) = self.held.pop_front().expect("front was present");
            out.push(Released {
                item,
                ready_step,
                release: Release::Expiry,
            });
        }
        out
    }

    /// Nothing else in the run can move: hand out the earliest held request
    /// for issue now.
    pub fn take_dry(&mut self) -> Option<Released<T>> {
        let (item, ready_step) = self.held.pop_front()?;
        Some(Released {
            item,
            ready_step,
            release: Release::DryQueue,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    #[test]
    fn the_treated_half_is_about_half_and_never_a_probe() {
        let _serial = config_override::exclusive_session();
        let n = 40_000i64;
        let mut treated = 0i64;
        let mut eligible = 0i64;
        for id in 0..n {
            let t = is_treated(id);
            assert_eq!(t, is_treated(id), "the split must not vary between reads");
            let probe = run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe;
            if probe {
                assert!(!t, "probe {id} is treated");
            } else {
                eligible += 1;
                treated += t as i64;
            }
        }
        let share = treated as f64 / eligible as f64;
        assert!(
            (share - 0.5).abs() < 0.02,
            "the treated half takes {share} of the eligible runs"
        );
    }

    fn ledger(step: i32, acted: bool, recent: u32, in_flight: u32) -> SendLedger {
        SendLedger {
            trigger: HandlerTrigger::Delivery,
            last_ghost_step: step,
            last_ghost_acted: acted,
            recent,
            in_flight,
            ..SendLedger::default()
        }
    }

    #[test]
    fn an_acted_ghost_with_a_full_fanout_in_flight_opens_a_window() {
        let ledgers = [SendLedger::default(), ledger(7, true, 2, 2), SendLedger::default()];
        assert!(fanout_window(&ledgers, 3, 7));
        assert!(!fanout_window(&ledgers, 3, 8), "the window is the step of the delivery");
        let more = [SendLedger::default(), ledger(7, true, 2, 5), SendLedger::default()];
        assert!(fanout_window(&more, 3, 7), "older sends in flight do not close it");
    }

    #[test]
    fn a_partial_fanout_or_an_inert_ghost_opens_nothing() {
        let partial = [SendLedger::default(), ledger(7, true, 1, 1), SendLedger::default()];
        assert!(!fanout_window(&partial, 3, 7));
        let inert = [SendLedger::default(), ledger(7, false, 2, 2), SendLedger::default()];
        assert!(!fanout_window(&inert, 3, 7));
        let timer = [SendLedger {
            trigger: HandlerTrigger::Timer,
            ..ledger(7, true, 2, 2)
        }];
        assert!(!fanout_window(&timer, 3, 7), "a timer handler is not a delivery");
        let landed = [SendLedger::default(), ledger(7, true, 2, 1), SendLedger::default()];
        assert!(!fanout_window(&landed, 3, 7), "in_flight below recent is not a full fan-out");
    }

    #[test]
    fn a_client_ledger_and_a_lone_server_never_open_a_window() {
        let ledgers = [SendLedger::default(), SendLedger::default(), ledger(7, true, 2, 2)];
        assert!(!fanout_window(&ledgers, 2, 7), "the third ledger is a client's");
        assert!(!fanout_window(&[ledger(7, true, 0, 0)], 1, 7));
    }

    #[test]
    fn a_window_counts_what_is_held_and_releases_nothing() {
        let mut q: HoldQueue<&str> = HoldQueue::default();
        q.hold("a", 3);
        q.hold("b", 5);
        assert_eq!(q.held(), 2);
        assert!(q.take_due(6).is_empty(), "nothing is due before expiry");
        assert_eq!(q.fire(), Firing { first: true, held: 2 });
        assert_eq!(q.held(), 2, "a window leaves the queue as it was");
        assert_eq!(q.pending(), 2);
        assert!(!q.is_empty());
        assert!(q.take_due(7).is_empty(), "a window sets nothing aside");
        assert_eq!(q.fire(), Firing { first: false, held: 2 });
        assert_eq!(q.firings(), 2);
        let due = q.take_due(3 + EXPIRY_STEPS + 1);
        assert_eq!(
            due,
            vec![Released {
                item: "a",
                ready_step: 3,
                release: Release::Expiry
            }]
        );
        assert_eq!(q.fire(), Firing { first: false, held: 1 });
    }

    #[test]
    fn a_request_expires_after_the_fixed_wait_in_ready_order() {
        let mut q: HoldQueue<u32> = HoldQueue::default();
        q.hold(1, 10);
        q.hold(2, 11);
        assert!(q.take_due(10 + EXPIRY_STEPS).is_empty());
        let due = q.take_due(10 + EXPIRY_STEPS + 1);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].item, 1);
        assert_eq!(due[0].release, Release::Expiry);
        assert_eq!(q.held(), 1);
        let due = q.take_due(11 + EXPIRY_STEPS + 1);
        assert_eq!(due[0].item, 2);
        assert!(q.is_empty());
    }

    #[test]
    fn a_dry_queue_hands_out_the_earliest_request_at_once() {
        let mut q: HoldQueue<u32> = HoldQueue::default();
        assert_eq!(q.take_dry(), None);
        q.hold(1, 2);
        q.hold(2, 3);
        let r = q.take_dry().expect("one is held");
        assert_eq!((r.item, r.ready_step, r.release), (1, 2, Release::DryQueue));
        assert_eq!(q.held(), 1);
        assert!(!q.is_empty(), "a held request is still pending work");
        q.take_dry();
        assert!(q.is_empty(), "nothing is held at exit once every request left");
    }
}
