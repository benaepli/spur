//! Same-step dispatch preference for send order between a sender that has
//! crashed and one destination.
//!
//! A sender's records to one destination carry a send ordinal that grows
//! with each send, but the network step draws among them freely, so a
//! destination routinely takes a later send before an earlier one. Around
//! a crash of the sender that freedom decides which of the dead
//! incarnation's messages a peer acts on first. On the treated half of the
//! runs, a network step whose pick is a remote record whose sending
//! incarnation is not the one running now - which covers every record of a
//! sender that is down - takes instead the eligible record from the same
//! sender to the same destination, sent by the same incarnation, with the
//! lowest send ordinal, when one is below the pick's. A pick of the
//! sender's current incarnation, and a pick from a sender that never
//! crashed, keep the order the draw gave them. Nothing is masked or held:
//! the displaced record stays in the queue and stays eligible. The
//! replacement runs after the draw and after the fresh-incarnation swap,
//! and consumes no random draw, so a treated and an untreated run read the
//! same random sequence at every step.
//!
//! The firing is counted by the incarnation class of the pick: contests and
//! replacements for the dead class, and, for the sender's current
//! incarnation, the replacements the preference declines to make. The
//! entry census names the dead class's share of the pair entries and of the
//! inversions on both halves, so the class the preference acts on can be
//! read against the class it leaves alone.
//!
//! The treated half is drawn under a salt of its own, so the split is
//! independent of every other split of a session. Run-cap probes and
//! timer-context probes are never treated: their completed lengths and
//! firing odds feed learners that must not carry an imprint of the
//! preference.
//!
//! The census that reads the preference - contests at the network step on
//! the control half, and pair entries and inversions at the message entry
//! on both halves - scans the network queue, so it runs on a salted
//! sixteenth of the runs, drawn independently of the treated split so both
//! halves are sampled alike. The treated half counts its contests on every
//! treated run, since the preference scans the eligible records there
//! anyway, and `corrected` counts every replacement. Outside the sample an
//! untreated run pays one branch at each site.

use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::timer_context;
use std::collections::HashMap;

/// Salt for the treated half. Distinct from every other split of a session.
pub const PAIR_ORDER_SALT: u64 = 0x_5041_4952_4F52_4452; // "PAIRORDR"

/// Whether this run takes a crashed sender's records to one destination in
/// send order at a contested network step.
pub fn is_treated(run_id: i64) -> bool {
    !run_cap::is_probe(run_id)
        && timer_context::run_mode(run_id) != timer_context::RunMode::Probe
        && run_phase::salted_phase(run_id, PAIR_ORDER_SALT, 2) == 1
}

/// Salt for the census sample. Distinct from the treated split and from
/// every other split of a session.
pub const CENSUS_SALT: u64 = 0x_5041_4952_4345_4E53; // "PAIRCENS"

/// One run in this many is a census run.
pub const CENSUS_PERIOD: i64 = 16;

/// Whether this run reads the census: the contest count on the control half
/// and the pair-entry and inversion counts on both halves.
pub fn is_census_run(run_id: i64) -> bool {
    run_phase::salted_phase(run_id, CENSUS_SALT, CENSUS_PERIOD) == 0
}

/// A class of records: one sender, one destination, and the sender's
/// incarnation at the send. Ordinals never repeat within a class.
pub type ClassKey = (usize, usize, u32);

/// One run's preference state. `entered` counts, per class, the message
/// entries a destination has taken, so an entry can tell whether a sibling
/// of its class arrived before it. It is maintained only on a census run.
#[derive(Clone, Debug, Default)]
pub struct RunState {
    /// The run is on the treated half and the plan is a generated one.
    pub enabled: bool,
    /// The run is a census run and the session collects statistics.
    pub census: bool,
    entered: HashMap<ClassKey, u32>,
}

impl RunState {
    /// `dest` took a message entry of the class. Returns whether an entry of
    /// the same class had been taken before.
    pub fn note_entry(&mut self, origin: usize, dest: usize, incarnation: u32) -> bool {
        let n = self.entered.entry((origin, dest, incarnation)).or_insert(0);
        *n += 1;
        *n > 1
    }

    /// Classes that have taken at least one entry.
    pub fn classes_entered(&self) -> usize {
        self.entered.len()
    }
}

/// What the eligible candidates hold against the picked record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contest {
    /// An eligible record of the pick's class other than the pick exists.
    pub rivals: bool,
    /// The queue index of the eligible record of the pick's class with the
    /// lowest send ordinal, when that ordinal is below the pick's; `None`
    /// when the pick already carries the lowest ordinal of its class among
    /// the eligible records, or when it has no rival.
    pub earliest: Option<usize>,
}

/// Among the eligible candidates other than the one at `pick`, the records
/// of the pick's class: each candidate is its queue index with
/// `(incarnation, send ordinal)` for a remote record from the same sender to
/// the same destination and `None` for anything else, which never competes.
/// A record sent by another incarnation of the sender is not a rival.
pub fn contest<I>(candidates: I, pick: usize, incarnation: u32, ordinal: u32) -> Contest
where
    I: IntoIterator<Item = (usize, Option<(u32, u32)>)>,
{
    let mut rivals = false;
    let mut best: Option<(u32, usize)> = None;
    for (i, item) in candidates {
        if i == pick {
            continue;
        }
        let Some((inc, ord)) = item else { continue };
        if inc != incarnation {
            continue;
        }
        rivals = true;
        if ord < ordinal && best.is_none_or(|(o, _)| ord < o) {
            best = Some((ord, i));
        }
    }
    Contest {
        rivals,
        earliest: best.map(|(_, i)| i),
    }
}

/// The census of one message entry from a sender that has crashed at least
/// once in the run. `siblings` are the send ordinals of the records of the
/// entry's class still in the network queue at the entry, and
/// `entered_before` says an entry of the class was taken earlier. Returns
/// `None` when the entry has no sibling either way and so is not a pair
/// entry; otherwise whether it is an inversion, which is an entry taken
/// while a sibling with a lower send ordinal is still in the queue, so the
/// destination acts on the later send first.
pub fn classify_entry<I>(siblings: I, ordinal: u32, entered_before: bool) -> Option<bool>
where
    I: IntoIterator<Item = u32>,
{
    let mut any = entered_before;
    let mut inverted = false;
    for s in siblings {
        any = true;
        if s < ordinal {
            inverted = true;
        }
    }
    any.then_some(inverted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;
    use crate::simulator::fresh_first;
    use crate::simulator::ghost_absorber;
    use crate::simulator::replay_corpus;

    #[test]
    fn the_treated_half_is_about_half_pure_and_never_a_probe() {
        let _serial = config_override::exclusive_session();
        let n = 40_000i64;
        let mut treated = 0i64;
        let mut exempt = 0i64;
        for id in 0..n {
            let t = is_treated(id);
            assert_eq!(t, is_treated(id), "the split must not vary between reads");
            let probe = run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe;
            if probe {
                assert!(!t, "probe {id} is treated");
                exempt += 1;
            }
            treated += t as i64;
        }
        assert!(exempt > 0, "no run id is a probe");
        let share = treated as f64 / (n - exempt) as f64;
        assert!(
            (share - 0.5).abs() < 0.02,
            "the treated half takes {share} of the non-probe runs"
        );
    }

    #[test]
    fn the_split_is_independent_of_the_other_salted_splits() {
        let _serial = config_override::exclusive_session();
        let n = 40_000i64;
        let ids: Vec<i64> = (0..n)
            .filter(|&id| {
                !run_cap::is_probe(id)
                    && timer_context::run_mode(id) != timer_context::RunMode::Probe
            })
            .collect();
        for (name, other) in [
            ("fresh first", &(|id| fresh_first::is_treated(id)) as &dyn Fn(i64) -> bool),
            ("retarget", &(|id| ghost_absorber::is_treated(id))),
            ("replay slot", &(|id| replay_corpus::is_slot(id))),
            ("replay prefix", &(|id| replay_corpus::is_prefix(id))),
        ] {
            let within = ids.iter().filter(|&&id| other(id)).count() as f64;
            if within == 0.0 {
                continue;
            }
            let both = ids.iter().filter(|&&id| other(id) && is_treated(id)).count() as f64;
            let share = both / within;
            assert!(
                (share - 0.5).abs() < 0.05,
                "the treated half takes {share} of the {name} runs"
            );
        }
    }

    #[test]
    fn the_census_sample_is_a_sixteenth_of_each_half() {
        let _serial = config_override::exclusive_session();
        let n = 64_000i64;
        let expected = 1.0 / CENSUS_PERIOD as f64;
        for treated in [false, true] {
            let half: Vec<i64> = (0..n).filter(|&id| is_treated(id) == treated).collect();
            let sampled = half.iter().filter(|&&id| is_census_run(id)).count() as f64;
            let share = sampled / half.len() as f64;
            assert!(
                (share - expected).abs() < 0.01,
                "the census takes {share} of the half with treated = {treated}"
            );
        }
        for id in 0..n {
            assert_eq!(is_census_run(id), is_census_run(id), "the sample must not vary between reads");
        }
    }

    fn items(v: &[(usize, Option<(u32, u32)>)]) -> Vec<(usize, Option<(u32, u32)>)> {
        v.to_vec()
    }

    #[test]
    fn the_lowest_ordinal_of_the_pick_s_class_wins_and_other_classes_never_compete() {
        let queue = items(&[
            (0, Some((1, 9))),
            (1, None),
            (2, Some((1, 4))),
            (3, Some((0, 1))),
            (4, Some((1, 6))),
        ]);
        assert_eq!(
            contest(queue.clone(), 0, 1, 9),
            Contest { rivals: true, earliest: Some(2) },
            "the lowest ordinal among the eligible same-class records wins"
        );
        assert_eq!(
            contest(queue.clone(), 2, 1, 4),
            Contest { rivals: true, earliest: None },
            "a pick that already carries the lowest ordinal is kept"
        );
        assert_eq!(
            contest(queue.clone(), 3, 0, 1),
            Contest { rivals: false, earliest: None },
            "a record of another incarnation is not a rival"
        );
        assert_eq!(
            contest(items(&[(0, Some((1, 9))), (1, None)]), 0, 1, 9),
            Contest { rivals: false, earliest: None },
            "the pick is not its own rival and a channel send never competes"
        );
        assert_eq!(
            contest(items(&[(0, Some((1, 9))), (4, Some((1, 6)))]), 0, 1, 9),
            Contest { rivals: true, earliest: Some(4) }
        );
    }

    #[test]
    fn the_census_names_pair_entries_and_inversions() {
        assert_eq!(classify_entry([], 3, false), None, "a lone entry is not a pair entry");
        assert_eq!(classify_entry([], 3, true), Some(false), "an entered sibling makes a pair entry in order");
        assert_eq!(classify_entry([5, 7], 3, false), Some(false), "later siblings still queued keep the order");
        assert_eq!(classify_entry([5, 1], 3, false), Some(true), "an earlier sibling still queued is an inversion");
        assert_eq!(classify_entry([1], 3, true), Some(true));
    }

    #[test]
    fn the_entry_table_answers_per_class() {
        let mut s = RunState::default();
        assert!(!s.note_entry(0, 1, 0), "the first entry of a class has no predecessor");
        assert!(s.note_entry(0, 1, 0));
        assert!(!s.note_entry(0, 1, 1), "another incarnation is another class");
        assert!(!s.note_entry(0, 2, 0), "another destination is another class");
        assert!(!s.note_entry(2, 1, 0), "another sender is another class");
        assert_eq!(s.classes_entered(), 4);
    }
}
