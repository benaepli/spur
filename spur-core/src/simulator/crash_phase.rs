//! Two-stage release for a placed crash: after the drawn step arrives, the
//! crash waits for a drawn phase of its victim's own fan-out.
//!
//! Crash placement answers when a crash lands on a clock that knows nothing
//! about what the victim is doing, so the stratum where a crash strands part
//! of a fan-out - some peers told, some not - is reached by accident. The
//! second stage turns that stratum into an assignment: one of three
//! equal-mass arms is drawn per placed crash, two of which keep the crash
//! withheld until the victim's current handler segment shows the phase the
//! arm names, and the third releases at once and is the control.
//!
//! Only half the placed runs anchor, under a salt of their own, so the other
//! half stays the randomized control it already was. A wait is bounded: the
//! mask comes off after a fixed window of steps whatever the fan-out does,
//! and never past the step reserve a run keeps for its recovery tail, so an
//! unmet phase costs a bounded stretch of a run and never the run.

use crate::simulator::fault_timing;
use crate::simulator::rng::{Stream, StreamRng};
use crate::simulator::run_phase;
use crate::simulator::util_stats::{self, CrashPhaseArm, CrashPhaseRelease};

/// Salt for the half of the placed runs that anchor. Distinct from every
/// other split of a session, so a run's anchoring is independent of its
/// posture, its probe role and its timer-steer mode.
const ANCHOR_SALT: u64 = 0x_4641_4E4F_5554_5048; // "FANOUTPH"

/// Steps an armed crash may stay withheld waiting for its phase. Wide
/// enough to cover the handler segments around a fan-out, short against a
/// run of a few thousand steps.
pub const WINDOW: i32 = 96;

/// Whether this run's placed crashes wait for a fan-out phase. Every other
/// run - stock posture, run-cap probe, and the unanchored half of the placed
/// posture - is untouched, down to the number of values it draws.
pub fn is_anchored(run_id: i64) -> bool {
    fault_timing::is_placed(run_id) && run_phase::salted_phase(run_id, ANCHOR_SALT, 2) == 1
}

/// The victim's fan-out as the arms read it. `segment_sends` is what the
/// node's current handler segment has issued and `undelivered` how many of
/// those nobody has received yet; `in_flight` counts every send of the node
/// still undelivered, which is what the census asks about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fanout {
    pub segment_sends: u32,
    pub undelivered: u32,
    pub in_flight: u32,
}

/// Whether the victim's fan-out is in the phase `arm` waits for.
pub fn phase_reached(arm: CrashPhaseArm, f: Fanout) -> bool {
    match arm {
        CrashPhaseArm::Early => f.segment_sends >= 1 && f.undelivered == f.segment_sends,
        CrashPhaseArm::Mid => {
            f.segment_sends >= 2 && f.undelivered >= 1 && f.undelivered < f.segment_sends
        }
        CrashPhaseArm::Stock => true,
    }
}

/// Where one node's placed crash stands in the second stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Slot {
    /// Not anchored: nothing here reads or writes a random stream.
    #[default]
    Off,
    /// A hold was drawn for this node on an anchored run; the arm is drawn
    /// at the step that hold expires.
    Pending,
    /// Withheld until the arm's phase or the window's end.
    Waiting { arm: CrashPhaseArm, armed_at: i32 },
    /// The mask is off for good; the arm is kept for the crash's census.
    Released { arm: CrashPhaseArm },
}

/// One run's second stage: a slot per node, the arms already seen in this
/// run, and the step past which no wait may run.
#[derive(Clone, Debug, Default)]
pub struct RunAnchor {
    slots: Vec<Slot>,
    seen: [bool; CrashPhaseArm::ALL.len()],
    reserve: i32,
}

impl RunAnchor {
    pub fn with_nodes(num_nodes: usize) -> Self {
        Self {
            slots: vec![Slot::Off; num_nodes],
            seen: [false; CrashPhaseArm::ALL.len()],
            reserve: 0,
        }
    }

    pub fn push_node(&mut self) {
        self.slots.push(Slot::Off);
    }

    /// This node's queued crash is placed on an anchored run: draw its arm
    /// when its step hold expires, and hold no wait past `reserve`.
    pub fn arm_node(&mut self, node: usize, reserve: i32) {
        self.reserve = reserve;
        if let Some(slot) = self.slots.get_mut(node) {
            *slot = Slot::Pending;
        }
    }

    /// The arm a node's crash carries, or None when the node is not
    /// anchored. Read where the crash is applied, so the census can split
    /// what a crash landed on by the phase it was made to wait for.
    pub fn arm_of(&self, node: usize) -> Option<CrashPhaseArm> {
        match self.slots.get(node) {
            Some(Slot::Waiting { arm, .. }) | Some(Slot::Released { arm }) => Some(*arm),
            _ => None,
        }
    }

    /// Whether `node`'s crash stays withheld at `step_now`, past the step
    /// its placement hold expired. Draws exactly one value the first time it
    /// is asked about a node armed by `arm_node`, and none at all otherwise.
    pub fn hold(
        &mut self,
        node: usize,
        f: Fanout,
        step_now: i32,
        rng: &mut impl StreamRng,
    ) -> bool {
        let slot = match self.slots.get(node) {
            Some(s) => *s,
            None => return false,
        };
        let next = match slot {
            Slot::Off | Slot::Released { .. } => return false,
            Slot::Pending => {
                let arm = self.draw(rng);
                if arm == CrashPhaseArm::Stock {
                    util_stats::record_crash_phase_release(
                        arm,
                        CrashPhaseRelease::Immediate,
                        0,
                        f.in_flight,
                    );
                    Slot::Released { arm }
                } else {
                    self.settle(arm, step_now, step_now, f)
                }
            }
            Slot::Waiting { arm, armed_at } => self.settle(arm, armed_at, step_now, f),
        };
        self.slots[node] = next;
        matches!(next, Slot::Waiting { .. })
    }

    /// One equal-mass arm, and the run's first draw of it.
    fn draw(&mut self, rng: &mut impl StreamRng) -> CrashPhaseArm {
        rng.use_stream(Stream::CrashPhase);
        let i = (rng.next_u64() % CrashPhaseArm::ALL.len() as u64) as usize;
        let arm = CrashPhaseArm::ALL[i];
        let first = !self.seen[arm.index()];
        self.seen[arm.index()] = true;
        util_stats::record_crash_phase_arm(arm, first);
        arm
    }

    /// Whether a waiting arm keeps waiting at `step_now`, and what to record
    /// when it stops.
    fn settle(&self, arm: CrashPhaseArm, armed_at: i32, step_now: i32, f: Fanout) -> Slot {
        let waited = (step_now - armed_at).max(0) as u64;
        if phase_reached(arm, f) {
            util_stats::record_crash_phase_release(
                arm,
                CrashPhaseRelease::Condition,
                waited,
                f.in_flight,
            );
            return Slot::Released { arm };
        }
        if step_now >= self.deadline(armed_at) {
            util_stats::record_crash_phase_release(
                arm,
                CrashPhaseRelease::Expired,
                waited,
                f.in_flight,
            );
            return Slot::Released { arm };
        }
        Slot::Waiting { arm, armed_at }
    }

    /// The last step a wait armed at `armed_at` may still withhold: the
    /// window's end, or the run's step reserve if that comes first.
    fn deadline(&self, armed_at: i32) -> i32 {
        let window_end = armed_at.saturating_add(WINDOW);
        if self.reserve > 0 {
            window_end.min(self.reserve)
        } else {
            window_end
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;
    use crate::simulator::run_cap;
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

    fn fanout(segment_sends: u32, undelivered: u32) -> Fanout {
        Fanout {
            segment_sends,
            undelivered,
            in_flight: undelivered,
        }
    }

    #[test]
    fn the_anchored_half_is_half_the_placed_runs_and_no_other_posture() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        let n = 64_000i64;
        let placed = (0..n).filter(|&id| fault_timing::is_placed(id)).count();
        let anchored = (0..n).filter(|&id| is_anchored(id)).count();
        let share = anchored as f64 / placed as f64;
        assert!((share - 0.5).abs() < 0.02, "the anchored share of placed runs is {share}");
        assert_eq!(
            (0..n).filter(|&id| is_anchored(id) && !fault_timing::is_placed(id)).count(),
            0,
            "an unplaced run must never anchor"
        );
        assert_eq!(
            (0..n).filter(|&id| is_anchored(id) && run_cap::is_probe(id)).count(),
            0,
            "a run-cap probe must never anchor"
        );
        // Anchoring must not track the other splits of the same session, or
        // the halves would differ by more than this mechanism.
        let both = (0..n)
            .filter(|&id| fault_timing::is_placed(id))
            .filter(|&id| is_anchored(id) == (run_phase::phase(id, 2) == 1))
            .count();
        let overlap = both as f64 / placed as f64;
        assert!((overlap - 0.5).abs() < 0.03, "anchoring follows the plain phase at {overlap}");
    }

    #[test]
    fn an_unanchored_node_holds_nothing_and_draws_nothing() {
        let mut anchor = RunAnchor::with_nodes(3);
        let mut rng = CountingRng::new(7);
        for step in 0..200 {
            for node in 0..3 {
                assert!(!anchor.hold(node, fanout(2, 2), step, &mut rng), "node {node} held");
            }
        }
        assert!(!anchor.hold(9, fanout(1, 1), 0, &mut rng), "an unknown node held");
        assert_eq!(anchor.arm_of(0), None, "an unanchored node carries no arm");
        assert_eq!(rng.draws, 0, "an unanchored run must not touch the stream");
    }

    #[test]
    fn the_early_predicate_reads_the_whole_segment_undelivered() {
        for (segment, undelivered, want) in [
            (0, 0, false),
            (1, 0, false),
            (1, 1, true),
            (2, 1, false),
            (2, 2, true),
            (3, 2, false),
            (5, 5, true),
        ] {
            assert_eq!(
                phase_reached(CrashPhaseArm::Early, fanout(segment, undelivered)),
                want,
                "early on {segment} issued, {undelivered} undelivered"
            );
        }
    }

    #[test]
    fn the_mid_predicate_reads_a_segment_part_delivered() {
        for (segment, undelivered, want) in [
            (0, 0, false),
            (1, 0, false),
            (1, 1, false),
            (2, 0, false),
            (2, 1, true),
            (2, 2, false),
            (3, 1, true),
            (3, 2, true),
            (3, 3, false),
        ] {
            assert_eq!(
                phase_reached(CrashPhaseArm::Mid, fanout(segment, undelivered)),
                want,
                "mid on {segment} issued, {undelivered} undelivered"
            );
        }
    }

    #[test]
    fn a_stock_arm_waits_for_nothing() {
        for f in [fanout(0, 0), fanout(1, 0), fanout(4, 2)] {
            assert!(phase_reached(CrashPhaseArm::Stock, f), "stock waited on {f:?}");
        }
    }

    /// An anchor whose node 0 is armed and whose draw has been forced to
    /// `arm`, with the draws it consumed and the step it was armed at.
    fn armed_at(arm: CrashPhaseArm, reserve: i32, step: i32) -> (RunAnchor, Fanout) {
        let mut anchor = RunAnchor::with_nodes(1);
        anchor.arm_node(0, reserve);
        anchor.slots[0] = Slot::Waiting { arm, armed_at: step };
        (anchor, fanout(0, 0))
    }

    #[test]
    fn a_wait_ends_at_the_phase_it_named() {
        let mut rng = CountingRng::new(1);
        for (arm, met) in [
            (CrashPhaseArm::Early, fanout(2, 2)),
            (CrashPhaseArm::Mid, fanout(2, 1)),
        ] {
            let (mut anchor, unmet) = armed_at(arm, 0, 10);
            for step in 10..40 {
                assert!(anchor.hold(0, unmet, step, &mut rng), "{arm:?} released early");
            }
            assert!(!anchor.hold(0, met, 40, &mut rng), "{arm:?} missed its phase");
            assert_eq!(anchor.arm_of(0), Some(arm), "the arm outlives the wait");
            assert!(!anchor.hold(0, unmet, 41, &mut rng), "a released crash held again");
        }
        assert_eq!(rng.draws, 0, "an already-armed slot draws nothing");
    }

    #[test]
    fn a_wait_ends_at_the_window_and_at_the_reserve() {
        let mut rng = CountingRng::new(1);
        let unmet = fanout(0, 0);
        let (mut anchor, _) = armed_at(CrashPhaseArm::Early, 0, 5);
        for step in 5..5 + WINDOW {
            assert!(anchor.hold(0, unmet, step, &mut rng), "the window ended at step {step}");
        }
        assert!(!anchor.hold(0, unmet, 5 + WINDOW, &mut rng), "the window did not expire");

        // A reserve short of the window ends the wait first.
        let (mut anchor, _) = armed_at(CrashPhaseArm::Mid, 12, 5);
        for step in 5..12 {
            assert!(anchor.hold(0, unmet, step, &mut rng), "the reserve ended at step {step}");
        }
        assert!(!anchor.hold(0, unmet, 12, &mut rng), "the reserve did not end the wait");
        assert_eq!(rng.draws, 0);
    }

    #[test]
    fn the_three_arms_are_drawn_at_equal_mass_and_cost_one_draw_each() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().crash_phase;
        let mut rng = CountingRng::new(99);
        let n = 3_000;
        let mut anchor = RunAnchor::with_nodes(n);
        for node in 0..n {
            anchor.arm_node(node, 0);
            anchor.hold(node, fanout(0, 0), 0, &mut rng);
        }
        let after = util_stats::snapshot().crash_phase;
        util_stats::set_enabled(false);
        assert_eq!(rng.draws, n as u64, "an arm draw must cost exactly one value");
        let counts = [
            after.early.armed - before.early.armed,
            after.mid.armed - before.mid.armed,
            after.stock.armed - before.stock.armed,
        ];
        assert_eq!(counts.iter().sum::<u64>(), n as u64, "every crash drew an arm");
        for (i, c) in counts.iter().enumerate() {
            let share = *c as f64 / n as f64;
            assert!((share - 1.0 / 3.0).abs() < 0.05, "arm {i} took {share} of the draws");
        }
        assert_eq!(
            after.armed - before.armed,
            counts[0] + counts[1],
            "the headline count is the two waiting arms"
        );
        assert_eq!(after.stock_releases - before.stock_releases, counts[2]);
        assert_eq!(after.stock.expired, before.stock.expired, "stock never expires");
    }

    #[test]
    fn a_stock_draw_releases_at_once_and_a_waiting_one_does_not() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        let mut rng = CountingRng::new(4);
        let n = 600;
        let mut anchor = RunAnchor::with_nodes(n);
        let mut held = 0;
        for node in 0..n {
            anchor.arm_node(node, 0);
            // A fan-out no waiting arm accepts, so only stock releases here.
            if anchor.hold(node, fanout(0, 0), 0, &mut rng) {
                held += 1;
                assert_ne!(anchor.arm_of(node), Some(CrashPhaseArm::Stock));
            } else {
                assert_eq!(anchor.arm_of(node), Some(CrashPhaseArm::Stock));
            }
        }
        util_stats::set_enabled(false);
        assert!(held > 0 && held < n, "the arms did not separate: {held} of {n} held");
    }
}
