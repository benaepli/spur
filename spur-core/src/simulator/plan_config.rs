use crate::simulator::core::partition::PartitionType;
use crate::simulator::core::state::NodeId;
use ecow::EcoString;
use imbl::OrdSet;
use petgraph::graph::DiGraph;
use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

use crate::analysis::resolver::NameId;
use crate::simulator::core::{PurgatoryConfig, QueuePolicyConfig, SchedulePolicy};
use crate::simulator::path::plan::{ClientOpSpec, DeliverSpec, EventAction, PlannedEvent};

#[derive(Debug, Error)]
pub enum PlanConfigError {
    #[error("unknown event id in dependencies: {0}")]
    UnknownEventId(String),
    #[error("duplicate event id: {0}")]
    DuplicateEventId(String),
    #[error(
        "invalid target node {target} for event '{event_id}': must be < num_servers ({num_servers})"
    )]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartitionSpec {
    IsolateOne { node: i32 },
    Halves { side_a: Vec<i32> },
    MajoritiesRing,
    Bridge { bridge: i32 },
}

impl PartitionSpec {
    /// Convert a plan config PartitionSpec into a runtime PartitionType.
    pub fn to_partition_type(&self, server_role: NameId, num_servers: i32) -> PartitionType {
        let make_node = |idx: i32| NodeId {
            role: server_role,
            index: idx as usize,
        };
        let n = num_servers as usize;
        match self {
            PartitionSpec::IsolateOne { node } => PartitionType::IsolateOne(make_node(*node)),
            PartitionSpec::Halves { side_a } => {
                let a: OrdSet<NodeId> = side_a.iter().map(|&i| make_node(i)).collect();
                let b: OrdSet<NodeId> = (0..num_servers)
                    .filter(|i| !side_a.contains(i))
                    .map(|i| make_node(i))
                    .collect();
                PartitionType::Halves { side_a: a, side_b: b }
            }
            PartitionSpec::MajoritiesRing => PartitionType::MajoritiesRing { num_nodes: n },
            PartitionSpec::Bridge { bridge } => {
                let bridge_node = make_node(*bridge);
                let mid = n / 2;
                let side_a: OrdSet<NodeId> = (0..mid)
                    .filter(|&i| i != *bridge as usize)
                    .map(|i| make_node(i as i32))
                    .collect();
                let side_b: OrdSet<NodeId> = (mid..n)
                    .filter(|&i| i != *bridge as usize)
                    .map(|i| make_node(i as i32))
                    .collect();
                PartitionType::Bridge {
                    bridge: bridge_node,
                    side_a,
                    side_b,
                }
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSpec {
    Write(i32, String),
    Read(i32, String),
    Crash(i32),
    Recover(i32),
    AllowTimer(i32, String),
    Partition(PartitionSpec),
    Heal,
    Deliver {
        function: String,
        #[serde(default)]
        from: Option<i32>,
        #[serde(default)]
        to: Option<i32>,
    },
}

impl EventSpec {
    fn validate_nodes(&self, num_servers: i32, event_id: &str) -> Result<(), PlanConfigError> {
        let check = |node: i32| -> Result<(), PlanConfigError> {
            if node < 0 || node >= num_servers {
                Err(PlanConfigError::TargetOutOfBounds {
                    event_id: event_id.to_string(),
                    target: node,
                    num_servers,
                })
            } else {
                Ok(())
            }
        };

        match self {
            EventSpec::Write(t, _)
            | EventSpec::Read(t, _)
            | EventSpec::Crash(t)
            | EventSpec::Recover(t)
            | EventSpec::AllowTimer(t, _) => check(*t),
            EventSpec::Partition(spec) => match spec {
                PartitionSpec::IsolateOne { node } => check(*node),
                PartitionSpec::Halves { side_a } => {
                    for n in side_a {
                        check(*n)?;
                    }
                    Ok(())
                }
                PartitionSpec::Bridge { bridge } => check(*bridge),
                PartitionSpec::MajoritiesRing => Ok(()),
            },
            EventSpec::Heal => Ok(()),
            EventSpec::Deliver { from, to, .. } => {
                if let Some(f) = from {
                    check(*f)?;
                }
                if let Some(t) = to {
                    check(*t)?;
                }
                Ok(())
            }
        }
    }

    fn to_event_action(&self) -> EventAction {
        match self {
            EventSpec::Write(t, k) => EventAction::ClientRequest(ClientOpSpec::Write(
                *t,
                EcoString::from(k.as_str()),
            )),
            EventSpec::Read(t, k) => {
                EventAction::ClientRequest(ClientOpSpec::Read(*t, EcoString::from(k.as_str())))
            }
            EventSpec::Crash(t) => EventAction::CrashNode(*t),
            EventSpec::Recover(t) => EventAction::RecoverNode(*t),
            EventSpec::AllowTimer(t, label) => EventAction::AllowTimer(*t, label.clone()),
            EventSpec::Partition(spec) => EventAction::Partition(spec.clone()),
            EventSpec::Heal => EventAction::Heal,
            EventSpec::Deliver { function, from, to } => EventAction::Deliver(DeliverSpec {
                function: function.clone(),
                from: *from,
                to: *to,
            }),
        }
    }
}

fn default_quick_fire_multiplier() -> f64 {
    5.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlanFileConfig {
    pub num_servers: i32,
    pub num_runs: i32,
    pub max_iterations: i32,

    #[serde(default)]
    pub schedule_policy: SchedulePolicy,

    #[serde(default)]
    pub queue_policy: QueuePolicyConfig,

    #[serde(default = "default_quick_fire_multiplier")]
    pub quick_fire_multiplier: f64,

    /// When true, labeled timers only fire when explicitly allowed by an AllowTimer event.
    #[serde(default)]
    pub strict_timers: bool,

    pub events: HashMap<String, EventSpec>,
    #[serde(default)]
    pub dependencies: Vec<(String, String)>,

    #[serde(default)]
    pub purgatory: PurgatoryConfig,
}

impl PlanFileConfig {
    pub fn validate(&self) -> Result<(), PlanConfigError> {
        if self.num_servers < 1 {
            return Err(PlanConfigError::InvalidNumServers(self.num_servers));
        }
        if self.num_runs < 1 {
            return Err(PlanConfigError::InvalidNumRuns(self.num_runs));
        }

        // Validate node indices
        for (id, spec) in &self.events {
            spec.validate_nodes(self.num_servers, id)?;
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
