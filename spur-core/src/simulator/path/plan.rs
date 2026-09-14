use crate::simulator::plan_config::PartitionSpec;
use ecow::EcoString;
use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum PlanError {
    #[error("event not found: {0:?}")]
    EventNotFound(NodeIndex),
    #[error("event {0:?} is not in progress")]
    NotInProgress(NodeIndex),
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, Ord, PartialOrd)]
pub enum ClientOpSpec {
    Write(i32, EcoString),
    Read(i32, EcoString),
    Rmw(i32, EcoString),
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, Ord, PartialOrd)]
pub struct DeliverSpec {
    pub function: String,
    pub from: Option<i32>,
    pub to: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, Ord, PartialOrd)]
pub enum EventAction {
    ClientRequest(ClientOpSpec),
    CrashNode(i32),
    RecoverNode(i32),
    AllowTimer(i32, String),
    Partition(PartitionSpec),
    Heal,
    Deliver(DeliverSpec),
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, Ord, PartialOrd)]
pub struct PlannedEvent {
    pub action: EventAction,
}

pub type ExecutionPlan = DiGraph<PlannedEvent, ()>;

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum EventStatus {
    Pending,
    Ready,
    InProgress,
    Completed,
}

/// `ready` and `open` must equal the number of statuses that are Ready and
/// the number that are not Completed after every public call, since the
/// release scan and the completion check answer from them alone.
#[derive(Debug, Clone)]
pub struct PlanEngine {
    graph: DiGraph<PlannedEvent, ()>,
    statuses: HashMap<NodeIndex, EventStatus>,
    ready: usize,
    open: usize,
}

impl PlanEngine {
    pub fn new(graph: ExecutionPlan) -> Self {
        let statuses: HashMap<NodeIndex, EventStatus> = graph
            .node_indices()
            .map(|idx| {
                let status = if graph.neighbors_directed(idx, Direction::Incoming).count() == 0 {
                    EventStatus::Ready
                } else {
                    EventStatus::Pending
                };
                (idx, status)
            })
            .collect();
        let ready = statuses
            .values()
            .filter(|s| **s == EventStatus::Ready)
            .count();
        let open = statuses.len();

        PlanEngine {
            graph,
            statuses,
            ready,
            open,
        }
    }

    /// Whether some event is Ready, so that `get_ready_events` would release
    /// at least one.
    pub fn has_ready(&self) -> bool {
        self.ready > 0
    }

    /// Returns a list of all events that are currently ready, marking them as InProgress.
    /// Events are released in index order: the statuses map has no order of
    /// its own, and the release order decides which operation gets which id
    /// and which client node, so a run is only a function of its seed when
    /// this order is fixed.
    pub fn get_ready_events(&mut self) -> Vec<(NodeIndex, &PlannedEvent)> {
        if self.ready == 0 {
            return Vec::new();
        }
        let mut ready: Vec<_> = self
            .statuses
            .iter()
            .filter(|(_, s)| **s == EventStatus::Ready)
            .map(|(idx, _)| *idx)
            .collect();
        ready.sort_unstable();

        for idx in &ready {
            self.statuses.insert(*idx, EventStatus::InProgress);
        }
        self.ready = 0;

        ready
            .into_iter()
            .map(|idx| (idx, &self.graph[idx]))
            .collect()
    }

    /// Marks an event as completed and updates dependencies, returning the
    /// dependents this completion made ready.
    pub fn mark_event_completed(&mut self, idx: NodeIndex) -> Vec<NodeIndex> {
        match self.statuses.insert(idx, EventStatus::Completed) {
            None | Some(EventStatus::Completed) => {}
            Some(previous) => {
                self.open -= 1;
                if previous == EventStatus::Ready {
                    self.ready -= 1;
                }
            }
        }

        // Notify dependents
        let children: Vec<_> = self
            .graph
            .neighbors_directed(idx, Direction::Outgoing)
            .collect();

        let mut released = Vec::new();
        for child in children {
            let all_deps_done = self
                .graph
                .neighbors_directed(child, Direction::Incoming)
                .all(|dep| self.statuses.get(&dep) == Some(&EventStatus::Completed));

            if all_deps_done && self.statuses.get(&child) == Some(&EventStatus::Pending) {
                self.statuses.insert(child, EventStatus::Ready);
                self.ready += 1;
                released.push(child);
            }
        }
        released
    }

    pub fn event(&self, idx: NodeIndex) -> &PlannedEvent {
        &self.graph[idx]
    }

    pub fn is_complete(&self) -> bool {
        self.open == 0
    }

    /// Planned events that have not reached Completed.
    pub fn outstanding_count(&self) -> usize {
        self.statuses
            .values()
            .filter(|s| **s != EventStatus::Completed)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::path::generator::{GeneratorConfig, generate_plan};
    use crate::simulator::recover_deps::RecoverDeps;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn assert_counts_exact(engine: &PlanEngine) {
        let ready = engine
            .statuses
            .values()
            .filter(|s| **s == EventStatus::Ready)
            .count();
        let all_completed = engine
            .statuses
            .values()
            .all(|s| *s == EventStatus::Completed);
        assert_eq!(engine.ready, ready, "ready count");
        assert_eq!(engine.has_ready(), ready > 0);
        assert_eq!(engine.open, engine.outstanding_count(), "open count");
        assert_eq!(engine.is_complete(), all_completed);
    }

    fn scan_reference(engine: &PlanEngine) -> Vec<NodeIndex> {
        let mut ready: Vec<_> = engine
            .statuses
            .iter()
            .filter(|(_, s)| **s == EventStatus::Ready)
            .map(|(idx, _)| *idx)
            .collect();
        ready.sort_unstable();
        ready
    }

    fn complete(engine: &mut PlanEngine, idx: NodeIndex) {
        engine.mark_event_completed(idx);
        assert_counts_exact(engine);
    }

    /// Drives a plan the way the run loop does: every step releases what is
    /// ready, and in-flight events complete in a random order. Some steps
    /// settle every in-flight client request at once, some complete an event
    /// a second time, and some runs stop before the plan finishes.
    fn drive(plan: ExecutionPlan, rng: &mut SmallRng) {
        let nodes = plan.node_count();
        let mut engine = PlanEngine::new(plan);
        assert_counts_exact(&engine);
        let mut in_flight: Vec<NodeIndex> = Vec::new();
        let mut completed: Vec<NodeIndex> = Vec::new();
        let early_end = rng
            .random_bool(0.25)
            .then(|| rng.random_range(0..4 * nodes + 1));
        let mut step = 0;
        while !engine.is_complete() && early_end != Some(step) {
            assert!(step < 100 * nodes + 100, "the plan finishes");
            let expected = scan_reference(&engine);
            let released: Vec<NodeIndex> = engine
                .get_ready_events()
                .into_iter()
                .map(|(idx, _)| idx)
                .collect();
            assert_eq!(released, expected, "the scan releases exactly the ready events");
            assert_counts_exact(&engine);
            in_flight.extend(released);
            match rng.random_range(0..10) {
                0 => {
                    let mut clients: Vec<NodeIndex> = in_flight
                        .iter()
                        .copied()
                        .filter(|idx| {
                            matches!(engine.event(*idx).action, EventAction::ClientRequest(_))
                        })
                        .collect();
                    clients.sort_unstable();
                    for idx in clients {
                        complete(&mut engine, idx);
                        in_flight.retain(|i| *i != idx);
                        completed.push(idx);
                    }
                }
                1 if !completed.is_empty() => {
                    let idx = completed[rng.random_range(0..completed.len())];
                    complete(&mut engine, idx);
                }
                _ if !in_flight.is_empty() => {
                    let idx = in_flight.swap_remove(rng.random_range(0..in_flight.len()));
                    complete(&mut engine, idx);
                    completed.push(idx);
                }
                _ => {}
            }
            step += 1;
        }
        assert_counts_exact(&engine);
        if early_end.is_none() {
            assert!(engine.is_complete());
            assert_eq!(engine.outstanding_count(), 0);
        }
    }

    #[test]
    fn counts_stay_exact_over_generated_plans_with_crashes_and_partitions() {
        let mut rng = SmallRng::seed_from_u64(0x9e37_79b9);
        for seed in 0..600u64 {
            let config = GeneratorConfig {
                num_servers: 3,
                num_write_ops: rng.random_range(0..5),
                num_read_ops: rng.random_range(0..9),
                num_rmw_ops: rng.random_range(0..3),
                num_keys: rng.random_range(1..3),
                num_crashes: rng.random_range(0..4),
                num_partitions: rng.random_range(0..3),
                dependency_density: [0.0, 0.3, 0.7, 1.0][(seed % 4) as usize],
                max_concurrent_writes: [None, Some(1), Some(2)][(seed % 3) as usize],
                post_fault_client_ops: rng.random_range(0..3),
                recover_deps: RecoverDeps::Stock,
            };
            let plan = generate_plan(config, &mut SmallRng::seed_from_u64(seed));
            drive(plan, &mut rng);
        }
    }

    #[test]
    fn counts_stay_exact_over_timer_and_deliver_events() {
        let mut rng = SmallRng::seed_from_u64(17);
        for _ in 0..300 {
            let mut plan = ExecutionPlan::new();
            let event = |action| PlannedEvent { action };
            let timer = plan.add_node(event(EventAction::AllowTimer(1, "tick".to_string())));
            let deliver = plan.add_node(event(EventAction::Deliver(DeliverSpec {
                function: "Prepare".to_string(),
                from: Some(0),
                to: None,
            })));
            let partition =
                plan.add_node(event(EventAction::Partition(PartitionSpec::MajoritiesRing)));
            let heal = plan.add_node(event(EventAction::Heal));
            let crash = plan.add_node(event(EventAction::CrashNode(2)));
            let recover = plan.add_node(event(EventAction::RecoverNode(2)));
            let write = plan.add_node(event(EventAction::ClientRequest(ClientOpSpec::Write(
                0,
                "k".into(),
            ))));
            let read = plan.add_node(event(EventAction::ClientRequest(ClientOpSpec::Read(
                1,
                "k".into(),
            ))));
            plan.add_edge(timer, deliver, ());
            plan.add_edge(partition, heal, ());
            plan.add_edge(crash, recover, ());
            plan.add_edge(deliver, write, ());
            plan.add_edge(heal, write, ());
            plan.add_edge(recover, read, ());
            plan.add_edge(write, read, ());
            drive(plan, &mut rng);
        }
    }

    #[test]
    fn completing_an_unreleased_event_keeps_the_counts() {
        let mut plan = ExecutionPlan::new();
        let event = |action| PlannedEvent { action };
        let a = plan.add_node(event(EventAction::CrashNode(0)));
        let b = plan.add_node(event(EventAction::RecoverNode(0)));
        let c = plan.add_node(event(EventAction::Heal));
        let d = plan.add_node(event(EventAction::CrashNode(1)));
        plan.add_edge(a, b, ());
        plan.add_edge(c, d, ());
        let mut engine = PlanEngine::new(plan);
        assert_counts_exact(&engine);
        assert_eq!((engine.ready, engine.open), (2, 4));

        complete(&mut engine, a);
        assert_eq!((engine.ready, engine.open), (2, 3), "a Ready event completed, b released");
        complete(&mut engine, d);
        assert_eq!((engine.ready, engine.open), (2, 2), "a Pending event completed");
        complete(&mut engine, d);
        assert_eq!((engine.ready, engine.open), (2, 2), "a second completion changes nothing");
        assert_eq!(engine.get_ready_events().len(), 2);
        assert_counts_exact(&engine);
        assert!(engine.get_ready_events().is_empty());
        complete(&mut engine, b);
        complete(&mut engine, c);
        assert!(engine.is_complete());
        assert_counts_exact(&engine);

        let empty = PlanEngine::new(ExecutionPlan::new());
        assert!(empty.is_complete() && !empty.has_ready());
    }
}
