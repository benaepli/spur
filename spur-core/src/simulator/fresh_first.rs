//! Same-step dispatch preference for a sender's current incarnation.
//!
//! When a node crashes with messages in the air and, once back, sends again
//! to the same peers, the network queue holds two classes of record from
//! that sender to one destination: ghosts from the incarnation that died and
//! fresh records from the one that is running. Which class a destination
//! takes first decides whether the ghost lands on a peer that has already
//! heard from the restarted sender. On the treated half of the runs, a
//! network step whose within-queue draw fell on a ghost takes instead an
//! eligible fresh record from the same sender to the same destination when
//! one exists, the highest-priority such record and the lowest queue index
//! among equals. The ghost is not masked or held: it stays in the queue and
//! stays eligible, and is taken whenever a later draw falls on it without a
//! fresh rival present. The swap runs after the draw and consumes no random
//! draw itself, so a treated and an untreated run read the same random
//! sequence at every step.
//!
//! The preference acts only at a destination that has itself restarted in
//! the run. A destination that never went down cannot have rebuilt its
//! state from a peer, so the order in which it takes the two classes
//! decides much less there, and the drawn ghost stays.
//!
//! The treated half is drawn under a salt of its own, so the split is
//! independent of every other split of a session. Run-cap probes are never
//! treated: their completed lengths feed the length learners, which must not
//! carry an imprint of the preference.

use crate::simulator::run_cap;
use crate::simulator::run_phase;
use std::collections::HashMap;

/// Salt for the treated half. Distinct from every other split of a session.
pub const FRESH_FIRST_SALT: u64 = 0x_4652_4553_4846_5354; // "FRESHFST"

/// Whether this run prefers a sender's current incarnation at a contested
/// network step.
pub fn is_treated(run_id: i64) -> bool {
    !run_cap::is_probe(run_id) && run_phase::salted_phase(run_id, FRESH_FIRST_SALT, 2) == 1
}

/// A record's identity within a run: the sending node and the position of
/// the send among that node's sends, which never repeats within a run and
/// survives a re-delivery after the destination's crash.
pub type RecordKey = (usize, u32);

/// One run's preference state. `displaced` holds, per ghost still in the
/// queue, how many times a fresh rival was taken in its place. `heard` is
/// the per-destination table of the highest incarnation of each origin the
/// destination has taken a message entry from, stored plus one so zero
/// means none; it is sized on first use and grown when a node is added.
#[derive(Clone, Debug, Default)]
pub struct RunState {
    /// The run is on the treated half and the plan is a generated one.
    pub enabled: bool,
    displaced: HashMap<RecordKey, u32>,
    heard: Vec<Vec<u32>>,
}

/// How many times a ghost was displaced before it was taken, bucketed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplacedCount {
    Once,
    Twice,
    Thrice,
    FourOrMore,
}

impl DisplacedCount {
    fn of(n: u32) -> Self {
        match n {
            0 | 1 => DisplacedCount::Once,
            2 => DisplacedCount::Twice,
            3 => DisplacedCount::Thrice,
            _ => DisplacedCount::FourOrMore,
        }
    }
}

impl RunState {
    /// A fresh rival was taken in place of the ghost `key`. Returns how many
    /// times that has now happened to this ghost.
    pub fn displace(&mut self, key: RecordKey) -> u32 {
        let n = self.displaced.entry(key).or_insert(0);
        *n += 1;
        *n
    }

    /// The record `key` is being taken. Returns the bucket of the number of
    /// times it had been displaced, or `None` when it never was.
    pub fn take(&mut self, key: RecordKey) -> Option<DisplacedCount> {
        self.displaced.remove(&key).map(DisplacedCount::of)
    }

    /// Ghosts currently carrying a displaced count.
    pub fn displaced_len(&self) -> usize {
        self.displaced.len()
    }

    fn ensure(&mut self, dest: usize, origin: usize) {
        let n = dest.max(origin) + 1;
        if self.heard.len() < n {
            self.heard.resize(n, Vec::new());
        }
        let row = &mut self.heard[dest];
        if row.len() < n {
            row.resize(n, 0);
        }
    }

    /// `dest` took a message entry sent by `origin` at `incarnation`.
    pub fn note_entry(&mut self, dest: usize, origin: usize, incarnation: u32) {
        self.ensure(dest, origin);
        let slot = &mut self.heard[dest][origin];
        *slot = (*slot).max(incarnation.saturating_add(1));
    }

    /// Whether `dest` has taken a message entry from `origin` sent at
    /// `incarnation` or a later one.
    pub fn heard_from(&self, dest: usize, origin: usize, incarnation: u32) -> bool {
        self.heard
            .get(dest)
            .and_then(|row| row.get(origin))
            .is_some_and(|&h| h >= incarnation.saturating_add(1))
    }

    /// `origin` came back from a crash: what any destination heard from its
    /// earlier incarnations no longer says anything about the current one.
    pub fn clear_origin(&mut self, origin: usize) {
        for row in &mut self.heard {
            if let Some(slot) = row.get_mut(origin) {
                *slot = 0;
            }
        }
    }
}

/// Among the eligible candidates other than the one at `drawn`, the records
/// of the class opposite the drawn record's: `want_fresh` asks for records
/// of the current incarnation, otherwise for ghosts. Each candidate is its
/// queue index with `(incarnation, priority)` for a remote record from the
/// same sender to the same destination and `None` for anything else, which
/// never competes. Returns whether any rival exists and, for a fresh
/// search, the index of the one with the highest priority, the lowest index
/// among equals.
pub fn rival<I>(candidates: I, drawn: usize, want_fresh: bool, current: u32) -> (bool, Option<usize>)
where
    I: IntoIterator<Item = (usize, Option<(u32, f64)>)>,
{
    let mut any = false;
    let mut best: Option<(f64, usize)> = None;
    for (i, item) in candidates {
        if i == drawn {
            continue;
        }
        let Some((incarnation, priority)) = item else { continue };
        if (incarnation == current) != want_fresh {
            continue;
        }
        any = true;
        if !want_fresh {
            break;
        }
        match best {
            Some((p, j)) if p > priority || (p == priority && j < i) => {}
            _ => best = Some((priority, i)),
        }
    }
    (any, best.map(|(_, i)| i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;
    use crate::simulator::ghost_absorber;
    use crate::simulator::replay_corpus;

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

    #[test]
    fn the_split_is_independent_of_the_other_salted_splits() {
        let _serial = config_override::exclusive_session();
        let n = 40_000i64;
        let ids: Vec<i64> = (0..n).filter(|&id| !run_cap::is_probe(id)).collect();
        for (name, other) in [
            ("retarget", &(|id| ghost_absorber::is_treated(id)) as &dyn Fn(i64) -> bool),
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
    fn a_ghost_counts_its_displacements_until_it_is_taken() {
        let mut s = RunState::default();
        assert_eq!(s.take((0, 7)), None, "a record never displaced has no count");
        assert_eq!(s.displace((0, 7)), 1);
        assert_eq!(s.displace((0, 7)), 2);
        assert_eq!(s.displace((1, 7)), 1, "another origin's record is another ghost");
        assert_eq!(s.displaced_len(), 2);
        assert_eq!(s.take((0, 7)), Some(DisplacedCount::Twice));
        assert_eq!(s.take((0, 7)), None, "taking clears the count");
        assert_eq!(s.take((1, 7)), Some(DisplacedCount::Once));
        for _ in 0..3 {
            s.displace((2, 0));
        }
        assert_eq!(s.take((2, 0)), Some(DisplacedCount::Thrice));
        for _ in 0..9 {
            s.displace((2, 1));
        }
        assert_eq!(s.take((2, 1)), Some(DisplacedCount::FourOrMore));
        assert_eq!(s.displaced_len(), 0);
    }

    #[test]
    fn the_heard_table_answers_per_pair_and_forgets_an_origin_at_its_restart() {
        let mut s = RunState::default();
        assert!(!s.heard_from(1, 0, 0), "nothing heard yet");
        s.note_entry(1, 0, 0);
        assert!(s.heard_from(1, 0, 0));
        assert!(!s.heard_from(1, 0, 1), "an entry from incarnation 0 is not one from 1");
        assert!(!s.heard_from(2, 0, 0), "another destination heard nothing");
        assert!(!s.heard_from(1, 2, 0), "another origin was not heard");
        s.note_entry(1, 0, 1);
        assert!(s.heard_from(1, 0, 1));
        s.note_entry(1, 0, 0);
        assert!(s.heard_from(1, 0, 1), "a late ghost entry does not lower the table");
        s.note_entry(1, 3, 0);
        s.clear_origin(0);
        assert!(!s.heard_from(1, 0, 0));
        assert!(!s.heard_from(1, 0, 1));
        assert!(s.heard_from(1, 3, 0), "another origin's row is untouched");
        s.note_entry(7, 9, 2);
        assert!(s.heard_from(7, 9, 2), "the table grows to any node index");
    }

    fn items(v: &[(usize, Option<(u32, f64)>)]) -> Vec<(usize, Option<(u32, f64)>)> {
        v.to_vec()
    }

    #[test]
    fn the_rival_search_takes_the_highest_priority_fresh_record_and_the_lowest_index_among_equals() {
        let queue = items(&[
            (0, Some((0, 0.9))),
            (1, None),
            (2, Some((1, 0.2))),
            (3, Some((1, 0.8))),
            (4, Some((1, 0.8))),
        ]);
        assert_eq!(rival(queue.clone(), 0, true, 1), (true, Some(3)));
        let none_fresh = items(&[(0, Some((0, 0.9))), (1, None), (5, Some((0, 0.1)))]);
        assert_eq!(rival(none_fresh, 0, true, 1), (false, None));
        assert_eq!(rival(queue.clone(), 3, false, 1), (true, None), "a ghost rival is only detected");
        let only_fresh = items(&[(3, Some((1, 0.8))), (4, Some((1, 0.8)))]);
        assert_eq!(rival(only_fresh, 3, false, 1), (false, None));
        assert_eq!(rival(items(&[(0, Some((0, 0.9)))]), 0, true, 1), (false, None), "the drawn record is not its own rival");
    }
}
