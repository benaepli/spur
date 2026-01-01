use crate::simulator::core::{LogEntry, Operation, State};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Budget {
    pub crashes: u32,
    pub recovers: u32,
    pub writes: u32,
    pub reads: u32,
}

#[derive(Debug, Clone)]
pub struct SearchNode {
    pub state: State,
    pub cost: f64,
    pub steps: u32,
    pub best_budget: Budget,
    pub history: Vec<Operation>,
    pub logs: Vec<LogEntry>,
    pub next_op_id: i32,
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
