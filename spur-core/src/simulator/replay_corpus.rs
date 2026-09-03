//! A corpus of run prefixes that reached the fault-crossing signal, replayed
//! by later runs of the same arm so more of the arm's runs are spent past
//! the point where a crash is queued on a node that has taken a delivery
//! from a dead incarnation.
//!
//! A fresh run that fires the signal is admitted as a parent: its scheduling
//! draws up to the signal, its workload seed and its configuration. A later
//! run in a replay slot becomes a child of the next parent in turn. A prefix
//! child replays the parent's draws and then continues on its own seed; a
//! plan-only child takes the parent's workload and configuration but draws
//! its whole schedule fresh, which tells apart what the schedule prefix buys
//! from what the plan alone buys. Children are never admitted, so the corpus
//! does not feed on itself.
//!
//! Slots are a pure function of the run id, so the treated and untreated
//! halves are a contrast by id and not by outcome: a slot whose corpus is
//! empty runs fresh and stays on the treated half. Run-cap probes and
//! steer-off timer probes are never slots; their runs feed learners that
//! must not carry a replay imprint.

use crate::simulator::rng::Recording;
use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::timer_context;
use std::collections::VecDeque;

/// Salt for the replay-slot half. Distinct from every other split of a session.
const REPLAY_SALT: u64 = 0x_5245_504C_4159_5354; // "REPLAYST"
/// Salt for the prefix half of the slots.
const PREFIX_SALT: u64 = 0x_5052_4546_4958_5450; // "PREFIXTP"

/// Parents kept per corpus; the oldest is dropped to admit a new one.
pub const CAPACITY: usize = 64;
/// Children a parent seeds before it is dropped.
pub const CHILDREN_PER_PARENT: u32 = 8;

/// Whether this run is a replay slot: it runs as a child when its corpus
/// holds a parent and fresh otherwise, and carries the slot bit either way.
pub fn is_slot(run_id: i64) -> bool {
    !run_cap::is_probe(run_id)
        && timer_context::run_mode(run_id) != timer_context::RunMode::Probe
        && run_phase::salted_phase(run_id, REPLAY_SALT, 2) == 1
}

/// Whether this slot's child replays the parent's schedule prefix rather
/// than only its plan. Implies `is_slot`.
pub fn is_prefix(run_id: i64) -> bool {
    is_slot(run_id) && run_phase::salted_phase(run_id, PREFIX_SALT, 2) == 1
}

/// What a child takes from its parent: the draws up to the signal, the
/// workload seed and configuration that make the plan, the parent's place in
/// the configuration grid, and the step the signal fired at, which is what a
/// faithful prefix child fires at too.
#[derive(Clone, Debug)]
pub struct Seed<C> {
    pub tape: Recording,
    pub workload_seed: u64,
    pub cfg: C,
    pub config_index: usize,
    pub cut_step: i32,
}

struct Parent<C> {
    seed: Seed<C>,
    children: u32,
}

/// The ring of parents of one arm. Parents are served round-robin; one that
/// has seeded `CHILDREN_PER_PARENT` children leaves the ring.
pub struct Corpus<C> {
    parents: VecDeque<Parent<C>>,
    cursor: usize,
}

impl<C> Default for Corpus<C> {
    fn default() -> Self {
        Self {
            parents: VecDeque::new(),
            cursor: 0,
        }
    }
}

impl<C: Clone> Corpus<C> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.parents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    /// Admit a parent, dropping the oldest when the ring is full. The
    /// round-robin cursor keeps pointing at the parent it was on.
    pub fn admit(&mut self, seed: Seed<C>) {
        if self.parents.len() == CAPACITY {
            self.parents.pop_front();
            self.cursor = self.cursor.saturating_sub(1);
        }
        self.parents.push_back(Parent { seed, children: 0 });
    }

    /// The seed for the next child, from the next parent in turn; `None`
    /// when the ring is empty.
    pub fn next_child(&mut self) -> Option<Seed<C>> {
        if self.parents.is_empty() {
            return None;
        }
        let idx = self.cursor % self.parents.len();
        let parent = &mut self.parents[idx];
        parent.children += 1;
        let seed = parent.seed.clone();
        if parent.children >= CHILDREN_PER_PARENT {
            self.parents.remove(idx);
            self.cursor = idx;
        } else {
            self.cursor = idx + 1;
        }
        Some(seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(tag: u64) -> Seed<u64> {
        Seed {
            tape: vec![tag; 3].into(),
            workload_seed: tag,
            cfg: tag,
            config_index: tag as usize,
            cut_step: tag as i32,
        }
    }

    #[test]
    fn an_empty_corpus_seeds_no_child() {
        let mut c = Corpus::<u64>::new();
        assert!(c.is_empty());
        assert!(c.next_child().is_none());
    }

    #[test]
    fn parents_are_served_in_turn_and_leave_after_their_last_child() {
        let mut c = Corpus::new();
        c.admit(seed(1));
        c.admit(seed(2));
        let mut served = Vec::new();
        while let Some(s) = c.next_child() {
            served.push(s.cfg);
            assert!(served.len() <= 100, "parents never leave the ring");
        }
        assert_eq!(served.len(), 2 * CHILDREN_PER_PARENT as usize);
        assert_eq!(
            served.iter().filter(|&&t| t == 1).count() as u32,
            CHILDREN_PER_PARENT
        );
        assert_eq!(&served[..4], &[1, 2, 1, 2], "parents alternate");
        assert!(c.is_empty(), "a parent that seeded its last child stays");
    }

    #[test]
    fn a_child_carries_its_parent_whole() {
        let mut c = Corpus::new();
        c.admit(seed(7));
        let s = c.next_child().expect("one parent");
        assert_eq!(&*s.tape, &[7, 7, 7]);
        assert_eq!(s.workload_seed, 7);
        assert_eq!(s.cfg, 7);
        assert_eq!(s.config_index, 7);
        assert_eq!(s.cut_step, 7);
        assert_eq!(c.len(), 1, "one child does not use a parent up");
    }

    #[test]
    fn the_ring_drops_its_oldest_parent_at_capacity() {
        let mut c = Corpus::new();
        for i in 0..(CAPACITY as u64 + 3) {
            c.admit(seed(i));
        }
        assert_eq!(c.len(), CAPACITY);
        let first = c.next_child().expect("a full ring");
        assert_eq!(first.cfg, 3, "the three oldest parents were dropped");
    }

    #[test]
    fn a_parent_leaving_mid_ring_does_not_skip_its_successor() {
        let mut c = Corpus::new();
        c.admit(seed(1));
        c.admit(seed(2));
        c.admit(seed(3));
        for _ in 0..(3 * (CHILDREN_PER_PARENT - 1)) {
            c.next_child();
        }
        let tail: Vec<u64> = std::iter::from_fn(|| c.next_child()).map(|s| s.cfg).collect();
        assert_eq!(tail, vec![1, 2, 3], "each parent seeds exactly one more child");
    }

    #[test]
    fn slots_are_a_pure_function_of_the_id_and_spare_the_probes() {
        let n = 40_000i64;
        let mut slots = 0;
        let mut prefixes = 0;
        for id in 0..n {
            let slot = is_slot(id);
            let prefix = is_prefix(id);
            assert_eq!(slot, is_slot(id), "id {id}: the slot changed between reads");
            assert!(!prefix || slot, "id {id}: a prefix child outside a slot");
            if run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe
            {
                assert!(!slot, "id {id}: a probe is a slot");
            }
            slots += slot as i64;
            prefixes += prefix as i64;
        }
        let slot_share = slots as f64 / n as f64;
        let prefix_share = prefixes as f64 / slots.max(1) as f64;
        assert!(
            (0.42..0.5).contains(&slot_share),
            "slots take {slot_share} of the runs; half less the probes was expected"
        );
        assert!(
            (0.47..0.53).contains(&prefix_share),
            "prefix children take {prefix_share} of the slots"
        );
    }
}
