use crate::compiler::cfg::Vertex;
use crate::simulator::path::plan::PlannedEvent;
use dashmap::DashMap;
use imbl::HashMap as ImMap;
use imbl::shared_ptr::ArcK;
use nauty_pet::prelude::CanonGraph;
use rand::SeedableRng;
use rand::prelude::SmallRng;
use scalable_cuckoo_filter::{
    DefaultHasher as CuckooHasher, ScalableCuckooFilter, ScalableCuckooFilterBuilder,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

#[derive(Debug, Default, Clone)]
pub struct VertexMap {
    vertices: ImMap<Vertex, u64>,
    total: u64,
}

impl VertexMap {
    pub fn new() -> Self {
        Self {
            vertices: ImMap::new(),
            total: 0,
        }
    }

    pub fn novelty_score(&self, vertex: Vertex) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        let count = self.vertices.get(&vertex).copied().unwrap_or(0) as f64;
        1.0 - (count / self.total as f64)
    }

    pub fn get(&self, vertex: &Vertex) -> Option<u64> {
        self.vertices.get(vertex).copied()
    }

    pub fn merge_from(&mut self, other: &HashMap<Vertex, u64>) {
        for (v, count) in other {
            self.vertices
                .entry(*v)
                .and_modify(|e| *e += count)
                .or_insert(*count);
            self.total += count;
        }
    }
}

impl IntoIterator for VertexMap {
    type Item = (Vertex, u64);
    type IntoIter = imbl::hashmap::ConsumingIter<(Vertex, u64), ArcK>;

    fn into_iter(self) -> Self::IntoIter {
        self.vertices.into_iter()
    }
}

#[derive(Debug, Default, Clone)]
pub struct LocalCoverage {
    edges: HashMap<(Vertex, Vertex), u64>,
    vertices: HashMap<Vertex, u64>,
    total: u64,
    novelty_score: f64,
}

impl LocalCoverage {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            vertices: HashMap::new(),
            total: 0,
            novelty_score: 0.0,
        }
    }

    /// Records a transition and accumulates novelty score using rarity^3.
    /// `rarity` should come from GlobalCoverage::novelty_score().
    pub fn record_with_rarity(&mut self, from: Vertex, to: Vertex, rarity: f64) -> bool {
        self.novelty_score += rarity.powi(3);
        self.record(from, to)
    }

    /// Returns the normalized plan score: mean novelty per step.
    /// Returns a value in [0, 1] where 1 means every transition was maximally novel.
    pub fn plan_score(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.novelty_score / self.total as f64
    }

    /// Returns the raw novelty score.
    pub fn novelty_score(&self) -> f64 {
        self.novelty_score
    }

    /// Records a transition between two vertices in the CFG.
    /// Returns true if this is the first time this edge has been seen.
    pub fn record(&mut self, from: Vertex, to: Vertex) -> bool {
        self.total += 1;

        let count = self.edges.entry((from, to)).or_insert(0);
        *count += 1;
        {
            let count = self.vertices.entry(to).or_insert(0);
            *count += 1;
        }
        *count == 1
    }

    /// Returns the total number of unique edges visited.
    pub fn unique_edges(&self) -> usize {
        self.edges.len()
    }

    /// Access to edges for merging into GlobalCoverage
    pub fn edges(&self) -> &HashMap<(Vertex, Vertex), u64> {
        &self.edges
    }

    /// Access to vertices for merging into GlobalCoverage
    pub fn vertices(&self) -> &HashMap<Vertex, u64> {
        &self.vertices
    }
}

#[derive(Debug, Default)]
pub struct GlobalCoverage {
    edges: DashMap<(usize, usize), u64>,
    vertices: RwLock<VertexMap>,
    total: AtomicU64,
}

impl GlobalCoverage {
    pub fn new() -> Self {
        Self {
            edges: DashMap::new(),
            vertices: RwLock::new(VertexMap::new()),
            total: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> VertexMap {
        self.vertices
            .read()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock")
            .clone()
    }

    pub fn merge(&self, local: &LocalCoverage) {
        let _ = self.total.fetch_add(local.total, Ordering::Relaxed);
        for ((from, to), count) in local.edges() {
            *self.edges.entry((*from, *to)).or_default() += count;
        }

        let mut vertices = self
            .vertices
            .write()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock");
        vertices.merge_from(local.vertices());
    }

    /// Returns the total number of vertex visits across all merged runs.
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Calculates the novelty score for a vertex using global stats.
    /// Returns 1.0 if never seen, otherwise 1.0 - (count/total).
    pub fn novelty_score(&self, vertex: Vertex) -> f64 {
        let total = self.total();
        if total == 0 {
            return 1.0;
        }
        self.vertices
            .read()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock")
            .novelty_score(vertex)
    }

    /// Access to vertices for coverage visualization.
    pub fn vertices_snapshot(&self) -> VertexMap {
        self.vertices
            .read()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock")
            .clone()
    }
}

const FALSE_POSITIVE_PROBABILITY: f64 = 0.001;

type Canonical = CanonGraph<PlannedEvent, ()>;
type CuckooFilter = ScalableCuckooFilter<Canonical, CuckooHasher, SmallRng>;

/// Global state shared across all simulation runs.
#[derive(Debug)]
pub struct GlobalState {
    pub coverage: GlobalCoverage,
    seen_states: Mutex<CuckooFilter>,
}

impl GlobalState {
    pub fn new(expected_runs: usize) -> Self {
        Self {
            coverage: GlobalCoverage::new(),
            seen_states: Mutex::new(
                ScalableCuckooFilterBuilder::new()
                    .rng(SmallRng::from_os_rng())
                    .initial_capacity(expected_runs)
                    .false_positive_probability(FALSE_POSITIVE_PROBABILITY)
                    .finish(),
            ),
        }
    }

    pub fn insert(&self, item: &Canonical) {
        self.seen_states
            .lock()
            .expect("Mutex poisoned - this indicates a panic occurred while holding the lock")
            .insert(item)
    }

    pub fn contains(&self, item: &Canonical) -> bool {
        self.seen_states
            .lock()
            .expect("Mutex poisoned - this indicates a panic occurred while holding the lock")
            .contains(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_score_empty_coverage() {
        let coverage = LocalCoverage::new();
        assert_eq!(coverage.plan_score(), 0.0);
    }

    #[test]
    fn test_plan_score_normalized_range() {
        let mut coverage = LocalCoverage::new();
        // Record transitions with maximum rarity (1.0)
        coverage.record_with_rarity(0, 1, 1.0);
        coverage.record_with_rarity(1, 2, 1.0);
        coverage.record_with_rarity(2, 3, 0.5);

        let score = coverage.plan_score();
        assert!(score >= 0.0, "plan_score should be >= 0, got {}", score);
        assert!(score <= 1.0, "plan_score should be <= 1, got {}", score);
    }

    #[test]
    fn test_plan_score_independent_of_length() {
        // Two runs with identical per-step rarity but different lengths
        let mut short = LocalCoverage::new();
        short.record_with_rarity(0, 1, 0.8);
        short.record_with_rarity(1, 2, 0.8);

        let mut long = LocalCoverage::new();
        for i in 0..100 {
            long.record_with_rarity(i, i + 1, 0.8);
        }

        let diff = (short.plan_score() - long.plan_score()).abs();
        assert!(
            diff < 1e-10,
            "Same per-step rarity should give same plan_score: short={}, long={}, diff={}",
            short.plan_score(),
            long.plan_score(),
            diff
        );
    }

    #[test]
    fn test_plan_score_higher_for_rarer_transitions() {
        let mut novel = LocalCoverage::new();
        novel.record_with_rarity(0, 1, 1.0);
        novel.record_with_rarity(1, 2, 1.0);

        let mut common = LocalCoverage::new();
        common.record_with_rarity(0, 1, 0.1);
        common.record_with_rarity(1, 2, 0.1);

        assert!(
            novel.plan_score() > common.plan_score(),
            "Novel transitions should score higher: novel={}, common={}",
            novel.plan_score(),
            common.plan_score()
        );
    }
}
