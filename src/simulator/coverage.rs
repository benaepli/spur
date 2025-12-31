use crate::compiler::cfg::Vertex;
use crate::simulator::plan::PlannedEvent;
use crossbeam::epoch::Atomic;
use dashmap::DashMap;
use nauty_pet::prelude::CanonGraph;
use rand::SeedableRng;
use rand::prelude::SmallRng;
use scalable_cuckoo_filter::{DefaultHasher, ScalableCuckooFilter, ScalableCuckooFilterBuilder};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const STEP_PENALTY: f64 = 0.0001;

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

    /// Returns the final plan score: novelty minus step penalty.
    pub fn plan_score(&self) -> f64 {
        self.novelty_score - (self.total as f64 * STEP_PENALTY)
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
    vertices: DashMap<usize, u64>,
    total: AtomicU64,
}

impl GlobalCoverage {
    pub fn new() -> Self {
        Self {
            edges: DashMap::new(),
            vertices: DashMap::new(),
            total: AtomicU64::new(0),
        }
    }

    pub fn merge(&self, local: &LocalCoverage) {
        let _ = self.total.fetch_add(local.total, Ordering::SeqCst);
        for ((from, to), count) in local.edges() {
            *self.edges.entry((*from, *to)).or_default() += count;
        }
        for (v, count) in local.vertices() {
            *self.vertices.entry(*v).or_default() += count;
        }
    }

    /// Returns the total number of vertex visits across all merged runs.
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::SeqCst)
    }

    /// Calculates the novelty score for a vertex using global stats.
    /// Returns 1.0 if never seen, otherwise 1.0 - (count/total).
    pub fn novelty_score(&self, vertex: Vertex) -> f64 {
        let count = match self.vertices.get(&vertex) {
            None => return 1.0,
            Some(count) => *count as f64,
        };
        let total = self.total();
        if total == 0 {
            return 1.0;
        }
        1.0 - (count / total as f64)
    }

    /// Access to vertices for coverage visualization.
    pub fn vertices(&self) -> &DashMap<usize, u64> {
        &self.vertices
    }
}

const FALSE_POSITIVE_PROBABILITY: f64 = 0.001;

type Canonical = CanonGraph<PlannedEvent, ()>;
type CuckooFilter = ScalableCuckooFilter<Canonical, DefaultHasher, SmallRng>;

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
        self.seen_states.lock().unwrap().insert(item)
    }

    pub fn contains(&self, item: &Canonical) -> bool {
        self.seen_states.lock().unwrap().contains(item)
    }
}
