use crate::simulator::core::State;
use std::cmp::Ordering;

#[derive(Debug)]
struct Budget {
    crashes: u32,
    writes: u32,
    reads: u32,
}

#[derive(Debug)]
struct SearchNode {
    state: State,
    cost: f64, // Lower is better
    steps: u32,

    best_budget: Budget,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for SearchNode {}
impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost)
    }
}
impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
