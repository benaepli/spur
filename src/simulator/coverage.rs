use crate::compiler::cfg::Vertex;
use dashmap::DashMap;
use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct LocalCoverage {
    edges: HashMap<(Vertex, Vertex), u64>,
    vertices: HashMap<Vertex, u64>,
    total: u64,
}

impl LocalCoverage {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            vertices: HashMap::new(),
            total: 0,
        }
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

#[derive(Debug, Default, Clone)]
pub struct GlobalCoverage {
    edges: DashMap<(usize, usize), u64>,
    vertices: DashMap<usize, u64>,
}

impl GlobalCoverage {
    pub fn new() -> Self {
        Self {
            edges: DashMap::new(),
            vertices: DashMap::new(),
        }
    }

    pub fn merge(&self, local: &LocalCoverage) {
        for ((from, to), count) in local.edges() {
            *self.edges.entry((*from, *to)).or_default() += count;
        }
        for (v, count) in local.vertices() {
            *self.vertices.entry(*v).or_default() += count;
        }
    }

    /// Returns the total number of vertex visits across all merged runs.
    pub fn total(&self) -> u64 {
        self.vertices.iter().map(|e| *e.value()).sum()
    }

    /// Calculates the novelty score for a vertex using global stats.
    /// Returns 1.0 if never seen, otherwise 1.0 - (count/total).
    pub fn novelty_score(&self, vertex: Vertex) -> f64 {
        let total = self.total();
        if total == 0 {
            return 1.0;
        }
        match self.vertices.get(&vertex) {
            None => 1.0,
            Some(count) => 1.0 - (*count as f64 / total as f64),
        }
    }
}

/// Global state shared across all simulation runs.
#[derive(Debug, Default)]
pub struct GlobalState {
    pub coverage: GlobalCoverage,
}

impl GlobalState {
    pub fn new() -> Self {
        Self {
            coverage: GlobalCoverage::new(),
        }
    }
}
