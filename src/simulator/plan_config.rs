use ecow::EcoString;
use petgraph::graph::DiGraph;
use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

use crate::simulator::core::SchedulePolicy;
use crate::simulator::path::plan::{ClientOpSpec, EventAction, PlannedEvent};

#[derive(Debug, Error)]
pub enum PlanConfigError {
    #[error("unknown event id in dependencies: {0}")]
    UnknownEventId(String),
    #[error("duplicate event id: {0}")]
    DuplicateEventId(String),
    #[error("invalid target node {target} for event '{event_id}': must be < num_servers ({num_servers})")]
    TargetOutOfBounds {
        event_id: String,
        target: i32,
        num_servers: i32,
    },
    #[error("num_servers must be >= 1, got {0}")]
    InvalidNumServers(i32),
    #[error("num_runs must be >= 1, got {0}")]
    InvalidNumRuns(i32),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSpec {
    Write(i32, String, String),
    Read(i32, String),
    Crash(i32),
    Recover(i32),
    Timeout(i32),
}

impl EventSpec {
    fn target(&self) -> i32 {
        match self {
            EventSpec::Write(t, _, _) => *t,
            EventSpec::Read(t, _) => *t,
            EventSpec::Crash(t) => *t,
            EventSpec::Recover(t) => *t,
            EventSpec::Timeout(t) => *t,
        }
    }

    fn to_event_action(&self) -> EventAction {
        match self {
            EventSpec::Write(t, k, v) => EventAction::ClientRequest(ClientOpSpec::Write(
                *t,
                EcoString::from(k.as_str()),
                EcoString::from(v.as_str()),
            )),
            EventSpec::Read(t, k) => EventAction::ClientRequest(ClientOpSpec::Read(
                *t,
                EcoString::from(k.as_str()),
            )),
            EventSpec::Crash(t) => EventAction::CrashNode(*t),
            EventSpec::Recover(t) => EventAction::RecoverNode(*t),
            EventSpec::Timeout(t) => {
                EventAction::ClientRequest(ClientOpSpec::SimulateTimeout(*t))
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlanFileConfig {
    pub num_servers: i32,
    pub num_runs: i32,
    pub max_iterations: i32,

    #[serde(default)]
    pub schedule_policy: SchedulePolicy,

    pub events: HashMap<String, EventSpec>,
    #[serde(default)]
    pub dependencies: Vec<(String, String)>,
}

impl PlanFileConfig {
    pub fn validate(&self) -> Result<(), PlanConfigError> {
        if self.num_servers < 1 {
            return Err(PlanConfigError::InvalidNumServers(self.num_servers));
        }
        if self.num_runs < 1 {
            return Err(PlanConfigError::InvalidNumRuns(self.num_runs));
        }

        // Validate target indices
        for (id, spec) in &self.events {
            let target = spec.target();
            if target < 0 || target >= self.num_servers {
                return Err(PlanConfigError::TargetOutOfBounds {
                    event_id: id.clone(),
                    target,
                    num_servers: self.num_servers,
                });
            }
        }

        // Validate dependency references
        for (from, to) in &self.dependencies {
            if !self.events.contains_key(from) {
                return Err(PlanConfigError::UnknownEventId(from.clone()));
            }
            if !self.events.contains_key(to) {
                return Err(PlanConfigError::UnknownEventId(to.clone()));
            }
        }

        Ok(())
    }

    pub fn to_execution_plan(&self) -> Result<DiGraph<PlannedEvent, ()>, PlanConfigError> {
        self.validate()?;

        let mut graph = DiGraph::new();
        let mut id_to_node = HashMap::new();

        for (id, spec) in &self.events {
            let node_idx = graph.add_node(PlannedEvent {
                action: spec.to_event_action(),
            });
            id_to_node.insert(id.clone(), node_idx);
        }

        for (from, to) in &self.dependencies {
            let from_idx = id_to_node[from];
            let to_idx = id_to_node[to];
            graph.add_edge(from_idx, to_idx, ());
        }

        Ok(graph)
    }
}
