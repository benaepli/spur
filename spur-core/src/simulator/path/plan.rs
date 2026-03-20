use ecow::EcoString;
use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("event not found: {0:?}")]
    EventNotFound(NodeIndex),
    #[error("event {0:?} is not in progress")]
    NotInProgress(NodeIndex),
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, Ord, PartialOrd)]
pub enum ClientOpSpec {
    Write(i32, EcoString, EcoString),
    Read(i32, EcoString),
    SimulateTimeout(i32),
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, Ord, PartialOrd)]
pub enum EventAction {
    ClientRequest(ClientOpSpec),
    CrashNode(i32),
    RecoverNode(i32),
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

#[derive(Debug, Clone)]
pub struct PlanEngine {
    graph: DiGraph<PlannedEvent, ()>,
    statuses: HashMap<NodeIndex, EventStatus>,
}

impl PlanEngine {
    pub fn new(graph: ExecutionPlan) -> Self {
        let statuses = graph
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

        PlanEngine { graph, statuses }
    }

    /// Returns a list of all events that are currently ready, marking them as InProgress.
    pub fn get_ready_events(&mut self) -> Vec<(NodeIndex, &PlannedEvent)> {
        let ready: Vec<_> = self
            .statuses
            .iter()
            .filter(|(_, s)| **s == EventStatus::Ready)
            .map(|(idx, _)| *idx)
            .collect();

        for idx in &ready {
            self.statuses.insert(*idx, EventStatus::InProgress);
        }

        ready
            .into_iter()
            .map(|idx| (idx, &self.graph[idx]))
            .collect()
    }

    /// Marks an event as completed and updates dependencies.
    pub fn mark_event_completed(&mut self, idx: NodeIndex) {
        self.statuses.insert(idx, EventStatus::Completed);

        // Notify dependents
        let children: Vec<_> = self
            .graph
            .neighbors_directed(idx, Direction::Outgoing)
            .collect();

        for child in children {
            let all_deps_done = self
                .graph
                .neighbors_directed(child, Direction::Incoming)
                .all(|dep| self.statuses.get(&dep) == Some(&EventStatus::Completed));

            if all_deps_done && self.statuses.get(&child) == Some(&EventStatus::Pending) {
                self.statuses.insert(child, EventStatus::Ready);
            }
        }
    }

    /// Reverts an InProgress event back to Ready.
    pub fn mark_as_ready(&mut self, idx: NodeIndex) -> Result<(), PlanError> {
        match self.statuses.get(&idx) {
            Some(EventStatus::InProgress) => {
                self.statuses.insert(idx, EventStatus::Ready);
                Ok(())
            }
            Some(_) => Err(PlanError::NotInProgress(idx)),
            None => Err(PlanError::EventNotFound(idx)),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.statuses.values().all(|s| *s == EventStatus::Completed)
    }
}
