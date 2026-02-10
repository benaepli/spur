use crate::simulator::core::{LogEntry, Operation, State};
use crate::simulator::hash_utils::HashPolicy;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Budget {
    pub crashes: u32,
    pub recovers: u32,
    pub writes: u32,
    pub reads: u32,
}

#[derive(Debug, Clone)]
pub struct SearchNode<H: HashPolicy> {
    pub state: State<H>,
    pub cost: f64,
    pub steps: u32,
    pub best_budget: Budget,
    pub history: Vec<Operation<H>>,
    pub logs: Vec<LogEntry>,
    pub next_op_id: i32,
}
impl<H: HashPolicy> PartialEq for SearchNode<H> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<H: HashPolicy> Eq for SearchNode<H> {}
impl<H: HashPolicy> PartialOrd for SearchNode<H> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost)
    }
}
impl<H: HashPolicy> Ord for SearchNode<H> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
