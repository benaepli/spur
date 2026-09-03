//! Crash retargeting keyed on fault-crossing deliveries.
//!
//! A delivery crosses a fault when its sender is down at delivery time or
//! has come back from a crash since sending. The node that takes such a
//! delivery has absorbed state from an incarnation that no longer exists,
//! and a crash of that node - rather than of whichever node the plan named -
//! is what strands the absorbed state before the node can spread it. Every
//! run records, per node, the step of the last fault-crossing delivery it
//! took and whether that delivery wrote anything. On the treated half of the
//! runs a planned crash, once released, moves from the plan's victim to the
//! live node with the most recent such mark; the plan's crash-recover pair
//! follows the crash to the node it landed on, so the pair stays a pair.
//!
//! The treated half is drawn under a salt of its own, so the split is
//! independent of every other split of a session. Run-cap probes are never
//! treated: their completed lengths feed the length learners, which must not
//! carry an imprint of the retarget.

use crate::simulator::core::state::SendLedger;
use crate::simulator::run_cap;
use crate::simulator::run_phase;

/// Salt for the treated half. Distinct from every other split of a session.
const RETARGET_SALT: u64 = 0x_4748_4F53_5441_4253; // "GHOSTABS"

/// Whether this run's planned crashes move to the node that last absorbed a
/// fault-crossing delivery.
pub fn is_treated(run_id: i64) -> bool {
    !run_cap::is_probe(run_id) && run_phase::salted_phase(run_id, RETARGET_SALT, 2) == 1
}

/// One run's retargeting state. `pending_pair_mask` has a bit per node, up
/// to the first 64, that holds a planned crash or recover the plan is still
/// waiting on; nodes past that width are never excluded by it.
#[derive(Clone, Debug, Default)]
pub struct RunState {
    /// The run is on the treated half and the plan is a generated one.
    pub enabled: bool,
    pub pending_pair_mask: u64,
    /// The once-per-run signal has been counted already.
    pub signal_counted: bool,
}

impl RunState {
    pub fn has_pending_pair(&self, node: usize) -> bool {
        node < u64::BITS as usize && self.pending_pair_mask & (1u64 << node) != 0
    }
}

/// Where a released crash lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// No live node carries a usable mark; the crash stays on its victim.
    NoAbsorber,
    /// The best absorber is the planned victim itself.
    SameVictim,
    /// The crash moves to `node`; `acted` is that node's mark.
    Retarget { node: usize, acted: bool },
}

/// The choice together with whether a better-ranked node had to be passed
/// over because the plan still has a crash or recover outstanding on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub choice: Choice,
    pub skipped_pending_pair: bool,
}

/// Rank the marked nodes among the first `servers` ledgers and pick the
/// crash's target. A mark that wrote state outranks one that did not; among
/// equals the later step wins, then the lower index, so the choice is a pure
/// function of the ledgers. A node that is down cannot be crashed and is not
/// ranked; the victim is always live here. A node the plan still has a
/// crash or recover outstanding on is passed over, so the retarget never
/// creates a second pair on one node.
pub fn choose(
    victim: usize,
    ledgers: &[SendLedger],
    servers: usize,
    is_live: impl Fn(usize) -> bool,
    has_pending_pair: impl Fn(usize) -> bool,
) -> Decision {
    let mut ranked: Vec<(bool, i32, usize)> = ledgers
        .iter()
        .take(servers)
        .enumerate()
        .filter(|(n, l)| l.last_ghost_step >= 0 && (*n == victim || is_live(*n)))
        .map(|(n, l)| (l.last_ghost_acted, l.last_ghost_step, n))
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    let mut skipped_pending_pair = false;
    for (acted, _, node) in ranked {
        if node == victim {
            return Decision {
                choice: Choice::SameVictim,
                skipped_pending_pair,
            };
        }
        if has_pending_pair(node) {
            skipped_pending_pair = true;
            continue;
        }
        return Decision {
            choice: Choice::Retarget { node, acted },
            skipped_pending_pair,
        };
    }
    Decision {
        choice: Choice::NoAbsorber,
        skipped_pending_pair,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    fn marked(step: i32, acted: bool) -> SendLedger {
        SendLedger {
            last_ghost_step: step,
            last_ghost_acted: acted,
            ..SendLedger::default()
        }
    }

    fn pick(victim: usize, ledgers: &[SendLedger]) -> Decision {
        choose(victim, ledgers, ledgers.len(), |_| true, |_| false)
    }

    #[test]
    fn an_acted_mark_beats_a_later_mark_that_did_not_act() {
        let ledgers = [SendLedger::default(), marked(10, true), marked(50, false)];
        assert_eq!(
            pick(0, &ledgers).choice,
            Choice::Retarget {
                node: 1,
                acted: true
            }
        );
        let ledgers = [SendLedger::default(), marked(10, true), marked(50, true)];
        assert_eq!(
            pick(0, &ledgers).choice,
            Choice::Retarget {
                node: 2,
                acted: true
            }
        );
        let ledgers = [SendLedger::default(), marked(10, false), marked(50, false)];
        assert_eq!(
            pick(0, &ledgers).choice,
            Choice::Retarget {
                node: 2,
                acted: false
            }
        );
    }

    #[test]
    fn a_node_with_an_outstanding_pair_is_passed_over_and_counted() {
        let ledgers = [SendLedger::default(), marked(50, true), marked(10, true)];
        let d = choose(0, &ledgers, 3, |_| true, |n| n == 1);
        assert_eq!(
            d.choice,
            Choice::Retarget {
                node: 2,
                acted: true
            }
        );
        assert!(d.skipped_pending_pair);
        let d = choose(0, &ledgers, 3, |_| true, |n| n == 2);
        assert_eq!(
            d.choice,
            Choice::Retarget {
                node: 1,
                acted: true
            }
        );
        assert!(
            !d.skipped_pending_pair,
            "the best absorber was not passed over"
        );
        let d = choose(0, &ledgers, 3, |_| true, |_| true);
        assert_eq!(d.choice, Choice::NoAbsorber);
        assert!(d.skipped_pending_pair);
    }

    #[test]
    fn no_mark_anywhere_keeps_the_victim() {
        let ledgers = [SendLedger::default(); 3];
        let d = pick(1, &ledgers);
        assert_eq!(d.choice, Choice::NoAbsorber);
        assert!(!d.skipped_pending_pair);
    }

    #[test]
    fn the_victim_at_the_top_of_the_ranking_stays_the_victim() {
        let ledgers = [marked(9, false), marked(90, true), marked(30, true)];
        assert_eq!(pick(1, &ledgers).choice, Choice::SameVictim);
        assert_eq!(
            pick(0, &ledgers).choice,
            Choice::Retarget {
                node: 1,
                acted: true
            }
        );
    }

    #[test]
    fn a_node_that_is_down_and_a_client_ledger_are_never_chosen() {
        let ledgers = [
            SendLedger::default(),
            marked(90, true),
            marked(30, true),
            marked(99, true),
        ];
        let d = choose(0, &ledgers, 3, |n| n != 1, |_| false);
        assert_eq!(
            d.choice,
            Choice::Retarget {
                node: 2,
                acted: true
            }
        );
        assert!(!d.skipped_pending_pair, "a node that is down is not a skip");
    }

    #[test]
    fn the_treated_half_is_about_half_pure_and_never_a_probe() {
        let _serial = config_override::exclusive_session();
        let n = 40_000i64;
        let mut treated = 0i64;
        for id in 0..n {
            let t = is_treated(id);
            assert_eq!(t, is_treated(id), "the split must not vary between reads");
            if run_cap::is_probe(id) {
                assert!(!t, "run-cap probe {id} is treated");
            }
            treated += t as i64;
        }
        let probes = (0..n).filter(|&id| run_cap::is_probe(id)).count() as f64;
        let share = treated as f64 / (n as f64 - probes);
        assert!(
            (share - 0.5).abs() < 0.02,
            "the treated half takes {share} of the non-probe runs"
        );
    }
}
