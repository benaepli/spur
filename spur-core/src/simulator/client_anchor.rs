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
//! The opposite direction on the same axis is the rush: the request is
//! issued at its ready step, as an untreated run does, and the record its
//! invocation pushes takes the top of the priority range, as does every
//! record later built while that operation is the ambient cause. A rushed
//! run therefore lands the request ahead of the fan-out instead of behind
//! it. A quarter of the runs that do not hold are rushed, so the axis
//! carries a hold half, a rush quarter and a stock quarter.
//!
//! Each direction is drawn under a salt of its own, so the split is
//! independent of every other split of a session. Probes take neither
//! direction: run-cap probes feed the length learners, and timer-context
//! probes must run unsteered.

use crate::simulator::core::state::{HandlerTrigger, SendLedger};
use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::timer_context;
use std::collections::{HashMap, HashSet, VecDeque};

/// Salt for the treated half. Distinct from every other split of a session.
pub const CLIENT_ANCHOR_SALT: u64 = 0x_434C_4E54_4143_4852; // "CLNTACHR"

/// Salt for the rushed quarter. Distinct from every other split of a
/// session, so the direction a run draws does not follow from the hold.
pub const RUSH_SALT: u64 = 0x_434C_4E54_5255_5348; // "CLNTRUSH"

/// A held request is issued once its ready step lies this many steps or
/// more behind the current step.
pub const EXPIRY_STEPS: i32 = 64;

/// The priority a rushed operation's records take. The scoring blend reads
/// priority on this scale, where one is the top of the range.
pub const RUSH_PRIORITY: f64 = 1.0;

/// One operation in this many reports the steps between its issue and its
/// first delivery. Every dispatch would otherwise consult the table of
/// operations still awaiting one.
pub const DISTANCE_STRIDE: i32 = 8;

/// Whether the run id names a run that either direction may act on.
fn is_eligible(run_id: i64) -> bool {
    !run_cap::is_probe(run_id)
        && timer_context::run_mode(run_id) != timer_context::RunMode::Probe
}

/// Whether this run holds its post-crash client requests.
pub fn is_treated(run_id: i64) -> bool {
    is_eligible(run_id) && run_phase::salted_phase(run_id, CLIENT_ANCHOR_SALT, 2) == 1
}

/// Whether this run rushes its post-crash client requests. Drawn over the
/// runs that do not hold, so the two directions never meet on one run.
pub fn is_rushed(run_id: i64) -> bool {
    is_eligible(run_id)
        && !is_treated(run_id)
        && run_phase::salted_phase(run_id, RUSH_SALT, 2) == 1
}

/// A direction on the post-crash request-timing axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Arm {
    /// The request is issued at its ready step and contends as it is drawn.
    #[default]
    Stock,
    /// The request waits and is issued once its hold runs out.
    Hold,
    /// The request is issued at its ready step and its records take the top
    /// of the priority range.
    Rush,
}

impl Arm {
    /// The index this direction occupies in a per-arm counter.
    pub fn index(self) -> usize {
        match self {
            Arm::Hold => 0,
            Arm::Rush => 1,
            Arm::Stock => 2,
        }
    }
}

/// The direction this run draws.
pub fn arm(run_id: i64) -> Arm {
    if is_treated(run_id) {
        Arm::Hold
    } else if is_rushed(run_id) {
        Arm::Rush
    } else {
        Arm::Stock
    }
}

/// A single-member group has no peers and cannot open a fan-out window.
pub fn fanout_window(ledgers: &[SendLedger], widths: &[u32], step: i32) -> bool {
    ledgers.iter().zip(widths).any(|(l, &peers)| {
        peers > 0
            && l.trigger == HandlerTrigger::Delivery
            && l.last_ghost_step == step
            && l.last_ghost_acted
            && l.recent >= peers
            && l.in_flight >= l.recent
    })
}

/// One run's direction and the operations it is following. `arm` is drawn
/// from the run id for a generated run and is always `Stock` for a plan run
/// from a file. `rushed_ops` names the client operations whose records take
/// the top priority; `awaiting_delivery` holds the invocation step of every
/// post-crash operation whose first delivery has not been counted yet, on
/// all three directions, so the distance is comparable across them, and is
/// filled only while the counters are on, since nothing else reads it.
/// `first_post_fault_op` is the id of the first client operation invoked
/// after the run's first crash; operation ids are assigned in increasing
/// order, so every operation at or above it was invoked after that crash.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunState {
    pub arm: Arm,
    pub rushed_ops: HashSet<i32>,
    pub awaiting_delivery: HashMap<i32, i32>,
    pub first_post_fault_op: Option<i32>,
}

impl RunState {
    /// Whether the run holds its post-crash client requests.
    pub fn holds(&self) -> bool {
        self.arm == Arm::Hold
    }

    /// Whether a record caused by `causal_operation_id` takes the top of the
    /// priority range.
    pub fn rushes(&self, causal_operation_id: Option<i32>) -> bool {
        causal_operation_id.is_some_and(|op| self.rushed_ops.contains(&op))
    }

    /// Whether a record caused by `causal_operation_id` was caused by a
    /// client operation invoked after the run's first crash.
    pub fn caused_post_fault(&self, causal_operation_id: Option<i32>) -> bool {
        match (causal_operation_id, self.first_post_fault_op) {
            (Some(op), Some(first)) => op >= first,
            _ => false,
        }
    }
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
    fn the_three_directions_partition_the_eligible_runs() {
        let _serial = config_override::exclusive_session();
        let n = 40_000i64;
        let mut counts = [0i64; 3];
        let mut eligible = 0i64;
        for id in 0..n {
            let a = arm(id);
            assert_eq!(a, arm(id), "the direction must not vary between reads");
            assert_eq!(is_treated(id), a == Arm::Hold);
            assert_eq!(is_rushed(id), a == Arm::Rush);
            assert!(
                !(is_treated(id) && is_rushed(id)),
                "run {id} took both directions"
            );
            let probe = run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe;
            if probe {
                assert_eq!(a, Arm::Stock, "probe {id} took a direction");
            } else {
                eligible += 1;
                counts[a.index()] += 1;
            }
        }
        assert_eq!(counts.iter().sum::<i64>(), eligible);
        let hold = counts[Arm::Hold.index()] as f64 / eligible as f64;
        let rush = counts[Arm::Rush.index()] as f64 / eligible as f64;
        let stock = counts[Arm::Stock.index()] as f64 / eligible as f64;
        assert!((hold - 0.5).abs() < 0.02, "the hold takes {hold} of the eligible runs");
        assert!((rush - 0.25).abs() < 0.02, "the rush takes {rush} of the eligible runs");
        assert!((stock - 0.25).abs() < 0.02, "stock takes {stock} of the eligible runs");
    }

    #[test]
    fn a_run_state_rushes_only_the_operations_it_was_given() {
        let mut st = RunState::default();
        assert!(!st.holds());
        assert!(!st.rushes(Some(1)));
        assert!(!st.rushes(None));
        st.arm = Arm::Hold;
        assert!(st.holds());
        st.arm = Arm::Rush;
        st.rushed_ops.insert(7);
        assert!(!st.holds());
        assert!(st.rushes(Some(7)));
        assert!(!st.rushes(Some(8)));
        assert!(!st.rushes(None), "a record with no cause is never rushed");
    }

    #[test]
    fn an_operation_at_or_above_the_first_post_fault_id_was_caused_post_fault() {
        let mut st = RunState::default();
        assert!(!st.caused_post_fault(Some(3)), "no crash has happened");
        st.first_post_fault_op = Some(3);
        assert!(!st.caused_post_fault(Some(2)));
        assert!(st.caused_post_fault(Some(3)));
        assert!(st.caused_post_fault(Some(9)));
        assert!(!st.caused_post_fault(None), "a record with no cause was not caused by a request");
    }

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
        assert!(fanout_window(&ledgers, &[2; 3], 7));
        assert!(!fanout_window(&ledgers, &[2; 3], 8), "the window is the step of the delivery");
        let more = [SendLedger::default(), ledger(7, true, 2, 5), SendLedger::default()];
        assert!(fanout_window(&more, &[2; 3], 7), "older sends in flight do not close it");
    }

    #[test]
    fn a_partial_fanout_or_an_inert_ghost_opens_nothing() {
        let partial = [SendLedger::default(), ledger(7, true, 1, 1), SendLedger::default()];
        assert!(!fanout_window(&partial, &[2; 3], 7));
        let inert = [SendLedger::default(), ledger(7, false, 2, 2), SendLedger::default()];
        assert!(!fanout_window(&inert, &[2; 3], 7));
        let timer = [SendLedger {
            trigger: HandlerTrigger::Timer,
            ..ledger(7, true, 2, 2)
        }];
        assert!(!fanout_window(&timer, &[2; 3], 7), "a timer handler is not a delivery");
        let landed = [SendLedger::default(), ledger(7, true, 2, 1), SendLedger::default()];
        assert!(!fanout_window(&landed, &[2; 3], 7), "in_flight below recent is not a full fan-out");
    }

    #[test]
    fn a_client_ledger_and_a_lone_server_never_open_a_window() {
        let ledgers = [SendLedger::default(), SendLedger::default(), ledger(7, true, 2, 2)];
        assert!(!fanout_window(&ledgers, &[1; 2], 7), "the third ledger is a client's");
        assert!(!fanout_window(&[ledger(7, true, 0, 0)], &[0; 1], 7));
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
        assert_eq!(EXPIRY_STEPS, 64, "the wait is fixed at 64 steps");
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
