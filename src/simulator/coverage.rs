use crate::compiler::cfg::Vertex;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Default, Clone)]
pub struct LocalCoverage {
    edges: HashMap<(Vertex, Vertex), u64>,
}

impl LocalCoverage {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Records a transition between two vertices in the CFG.
    /// Returns true if this is the first time this edge has been seen.
    pub fn record(&mut self, from: Vertex, to: Vertex) -> bool {
        let count = self.edges.entry((from, to)).or_insert(0);
        *count += 1;
        *count == 1
    }

    /// Returns the total number of unique edges visited.
    pub fn unique_edges(&self) -> usize {
        self.edges.len()
    }
}


#[derive(Debug, Default, Clone)]
pub struct GlobalCoverage {
    frequencies: DashMap<(usize, usize), u64>,
}

impl GlobalCoverage {
    pub fn new() -> Self {
        Self {
            frequencies: DashMap::new()
        }
    }


}