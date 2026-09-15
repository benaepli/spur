use crate::analysis::resolver::NameId;
use crate::compiler::cfg::Program;
use crate::simulator::core::steer_terms::{ResolvedTerms, SteerTerms};
use crate::simulator::core::state::NodeId;
use crate::simulator::deploy::path::{check_path, group_role, node_role, parse_path, resolve_group, resolve_node};
use crate::simulator::deploy::{Deployment, evaluate_deploy, select_deploy};
use ecow::EcoString;
use petgraph::graph::DiGraph;
use serde::Deserialize;
use serde_json::{Map, Value as Json, json};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

use crate::simulator::core::{
    PurgatoryConfig, QueuePolicyConfig, SchedulePolicy, WithinQueueSelector,
};
use crate::simulator::feedback::FeedbackConfig;
use crate::simulator::path::plan::{
    ClientOpSpec, DeliverSpec, EventAction, ExecutionPlan, PartitionAction, PlannedEvent,
};

#[derive(Debug, Error)]
pub enum PlanConfigError {
    #[error("unknown event id in dependencies: {0}")]
    UnknownEventId(String),
    #[error("duplicate event id: {0}")]
    DuplicateEventId(String),
    #[error("event '{event_id}', path `{path}`: {reason}")]
    PathError {
        event_id: String,
        path: String,
        reason: String,
    },
    #[error("event '{event_id}': {reason}")]
    InvalidEvent { event_id: String, reason: String },
    #[error("the deploy rejects params {0}")]
    DeployRejected(String),
    #[error("{0}")]
    Deploy(String),
    #[error("num_runs must be >= 1, got {0}")]
    InvalidNumRuns(i32),
    #[error("invalid feedback config: {0}")]
    InvalidFeedback(String),
    #[error("invalid steer terms: {0}")]
    InvalidSteerTerms(String),
}

/// Positions in `side_a` and `bridge` index the group.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartitionSpec {
    IsolateOne { node: String },
    Halves { group: String, side_a: Vec<usize> },
    MajoritiesRing { group: String },
    Bridge { group: String, bridge: usize },
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpSpec {
    #[serde(default)]
    pub dest: Option<String>,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimerSpec {
    pub node: String,
    pub label: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSpec {
    Write(OpSpec),
    Read(OpSpec),
    Rmw(OpSpec),
    Crash(String),
    Recover(String),
    AllowTimer(TimerSpec),
    Partition(PartitionSpec),
    Heal,
    Deliver {
        function: String,
        #[serde(default)]
        from: Option<String>,
        #[serde(default)]
        to: Option<String>,
    },
}

/// A plan whose paths are resolved against the deployment its tuple builds.
pub struct ResolvedPlan {
    pub deployment: Arc<Deployment>,
    pub graph: ExecutionPlan,
    /// The plan with every path replaced by global node indices.
    pub json: Json,
}

/// Resolves the paths of one plan against one deployment.
struct Resolver<'a> {
    program: &'a Program,
    deployment: &'a Deployment,
    event_id: &'a str,
}

impl Resolver<'_> {
    fn path_error(&self, path: &str, reason: String) -> PlanConfigError {
        PlanConfigError::PathError {
            event_id: self.event_id.to_string(),
            path: path.to_string(),
            reason,
        }
    }

    fn node(&self, path: &str) -> Result<NodeId, PlanConfigError> {
        let schema = &self.program.topology;
        let segments = parse_path(path).map_err(|e| self.path_error(path, e))?;
        let ty = check_path(&segments, &self.deployment.spec.root, schema)
            .map_err(|e| self.path_error(path, e))?;
        node_role(&ty, schema).map_err(|e| self.path_error(path, e))?;
        resolve_node(&segments, self.deployment).map_err(|e| self.path_error(path, e))
    }

    fn group(&self, path: &str) -> Result<Vec<NodeId>, PlanConfigError> {
        let schema = &self.program.topology;
        let segments = parse_path(path).map_err(|e| self.path_error(path, e))?;
        let ty = check_path(&segments, &self.deployment.spec.root, schema)
            .map_err(|e| self.path_error(path, e))?;
        group_role(&ty, schema).map_err(|e| self.path_error(path, e))?;
        resolve_group(&segments, self.deployment).map_err(|e| self.path_error(path, e))
    }

    fn position(&self, group: &[NodeId], position: usize) -> Result<(), PlanConfigError> {
        if position < group.len() {
            return Ok(());
        }
        Err(PlanConfigError::InvalidEvent {
            event_id: self.event_id.to_string(),
            reason: format!("position {position} is outside a group of {}", group.len()),
        })
    }

    /// The destination of a client operation, checked against the role of the
    /// operation's `dest` parameter.
    fn dest(&self, op: &str, spec: &OpSpec) -> Result<Option<NodeId>, PlanConfigError> {
        let invalid = |reason: String| PlanConfigError::InvalidEvent {
            event_id: self.event_id.to_string(),
            reason,
        };
        let client = self.deployment.client_role;
        let expected: Option<Option<NameId>> = self
            .program
            .topology
            .roles
            .get(&client)
            .and_then(|r| r.destinations.get(op).copied());
        let Some(expected) = expected else {
            return Err(invalid(format!("client {} defines no {op}", self.program.id_to_name[&client])));
        };
        match (expected, &spec.dest) {
            (None, None) => Ok(None),
            (None, Some(_)) => Err(invalid(format!("{op} takes no destination"))),
            (Some(_), None) => Err(invalid(format!("{op} takes a destination"))),
            (Some(role), Some(path)) => {
                let node = self.node(path)?;
                if node.role != role {
                    return Err(self.path_error(
                        path,
                        format!("{op} takes a destination of role {}", self.program.id_to_name[&role]),
                    ));
                }
                Ok(Some(node))
            }
        }
    }

    fn event(&self, spec: &EventSpec) -> Result<(EventAction, Json), PlanConfigError> {
        let op = |name: &str, key: &str, dest: Option<NodeId>| -> Json {
            let mut body = Map::new();
            if let Some(node) = dest {
                body.insert("dest".into(), json!(node.index));
            }
            body.insert("key".into(), json!(key));
            json!({ name: body })
        };
        Ok(match spec {
            EventSpec::Write(s) => {
                let dest = self.dest("Write", s)?;
                let json = op("write", &s.key, dest);
                (EventAction::ClientRequest(ClientOpSpec::Write(dest, EcoString::from(s.key.as_str()))), json)
            }
            EventSpec::Read(s) => {
                let dest = self.dest("Read", s)?;
                let json = op("read", &s.key, dest);
                (EventAction::ClientRequest(ClientOpSpec::Read(dest, EcoString::from(s.key.as_str()))), json)
            }
            EventSpec::Rmw(s) => {
                let dest = self.dest("RMW", s)?;
                let json = op("rmw", &s.key, dest);
                (EventAction::ClientRequest(ClientOpSpec::Rmw(dest, EcoString::from(s.key.as_str()))), json)
            }
            EventSpec::Crash(path) => {
                let node = self.node(path)?;
                (EventAction::CrashNode(node), json!({"crash": node.index}))
            }
            EventSpec::Recover(path) => {
                let node = self.node(path)?;
                (EventAction::RecoverNode(node), json!({"recover": node.index}))
            }
            EventSpec::AllowTimer(t) => {
                let node = self.node(&t.node)?;
                (
                    EventAction::AllowTimer(node, t.label.clone()),
                    json!({"allow_timer": {"node": node.index, "label": t.label}}),
                )
            }
            EventSpec::Partition(p) => {
                let indices = |group: &[NodeId]| group.iter().map(|n| n.index).collect::<Vec<_>>();
                let (action, json) = match p {
                    PartitionSpec::IsolateOne { node } => {
                        let node = self.node(node)?;
                        (PartitionAction::IsolateOne(node), json!({"type": "isolate_one", "node": node.index}))
                    }
                    PartitionSpec::Halves { group, side_a } => {
                        let group = self.group(group)?;
                        for &position in side_a {
                            self.position(&group, position)?;
                        }
                        let json = json!({"type": "halves", "group": indices(&group), "side_a": side_a});
                        (PartitionAction::Halves { group, side_a: side_a.clone() }, json)
                    }
                    PartitionSpec::MajoritiesRing { group } => {
                        let group = self.group(group)?;
                        let json = json!({"type": "majorities_ring", "group": indices(&group)});
                        (PartitionAction::MajoritiesRing { group }, json)
                    }
                    PartitionSpec::Bridge { group, bridge } => {
                        let group = self.group(group)?;
                        self.position(&group, *bridge)?;
                        let json = json!({"type": "bridge", "group": indices(&group), "bridge": bridge});
                        (PartitionAction::Bridge { group, bridge: *bridge }, json)
                    }
                };
                (EventAction::Partition(action), json!({"partition": json}))
            }
            EventSpec::Heal => (EventAction::Heal, json!("heal")),
            EventSpec::Deliver { function, from, to } => {
                let from = from.as_deref().map(|p| self.node(p)).transpose()?;
                let to = to.as_deref().map(|p| self.node(p)).transpose()?;
                let mut body = Map::new();
                body.insert("function".into(), json!(function));
                if let Some(node) = from {
                    body.insert("from".into(), json!(node.index));
                }
                if let Some(node) = to {
                    body.insert("to".into(), json!(node.index));
                }
                (
                    EventAction::Deliver(DeliverSpec { function: function.clone(), from, to }),
                    json!({"deliver": body}),
                )
            }
        })
    }
}

fn default_quick_fire_multiplier() -> f64 {
    5.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlanFileConfig {
    /// The `@deploy` function; may be omitted when the program defines
    /// exactly one.
    #[serde(default)]
    pub deploy: Option<String>,
    /// A fixed value for every field of the deploy's parameter struct.
    #[serde(default)]
    pub params: Json,
    pub num_runs: i32,
    pub max_iterations: i32,

    #[serde(default)]
    pub schedule_policy: SchedulePolicy,

    #[serde(default)]
    pub queue_policy: QueuePolicyConfig,

    #[serde(default)]
    pub within_queue_selector: WithinQueueSelector,

    #[serde(default = "default_quick_fire_multiplier")]
    pub quick_fire_multiplier: f64,

    /// The weights of the scheduling score; `recover_crashed` unset reads
    /// `quick_fire_multiplier`.
    #[serde(default)]
    pub steer_terms: SteerTerms,

    /// When true, labeled timers only fire when explicitly allowed by an AllowTimer event.
    #[serde(default)]
    pub strict_timers: bool,

    pub events: HashMap<String, EventSpec>,
    #[serde(default)]
    pub dependencies: Vec<(String, String)>,

    #[serde(default)]
    pub purgatory: PurgatoryConfig,

    /// Probability that a scheduling step withholds a pending crash whose node
    /// is not in the middle of its own fan-out. See the explorer config field of
    /// the same name; 0.0 withholds nothing.
    #[serde(default)]
    pub partial_fanout_crash_bias: f64,

    #[serde(default)]
    pub feedback: FeedbackConfig,
}

impl PlanFileConfig {
    pub fn validate(&self) -> Result<(), PlanConfigError> {
        if self.num_runs < 1 {
            return Err(PlanConfigError::InvalidNumRuns(self.num_runs));
        }
        self.feedback
            .validate()
            .map_err(PlanConfigError::InvalidFeedback)?;
        self.steer_terms
            .resolve(self.quick_fire_multiplier)
            .map_err(PlanConfigError::InvalidSteerTerms)?;

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

    /// The score weights every plan run uses; `validate` has already
    /// rejected a block that cannot resolve.
    pub fn steer_terms_resolved(&self) -> ResolvedTerms {
        self.steer_terms
            .resolve(self.quick_fire_multiplier)
            .expect("steer_terms were validated with the plan")
    }

    /// Builds the plan's deployment from its fixed tuple and resolves every
    /// path against it. Any path that does not resolve is an error.
    pub fn resolve(&self, program: &Program) -> Result<ResolvedPlan, PlanConfigError> {
        self.validate()?;
        let deploy = select_deploy(program, self.deploy.as_deref()).map_err(PlanConfigError::Deploy)?;
        let params = if self.params.is_null() { json!({}) } else { self.params.clone() };
        let deployment = evaluate_deploy(program, deploy, &params)
            .map_err(PlanConfigError::Deploy)?
            .ok_or_else(|| PlanConfigError::DeployRejected(params.to_string()))?;

        let mut graph = DiGraph::new();
        let mut id_to_node = HashMap::new();
        let mut events = Map::new();
        for (id, spec) in &self.events {
            let resolver = Resolver { program, deployment: &deployment, event_id: id };
            let (action, json) = resolver.event(spec)?;
            let node_idx = graph.add_node(PlannedEvent { action });
            id_to_node.insert(id.clone(), node_idx);
            events.insert(id.clone(), json);
        }
        for (from, to) in &self.dependencies {
            graph.add_edge(id_to_node[from], id_to_node[to], ());
        }

        let json = json!({
            "deploy": deploy.name,
            "params": params,
            "node_count": deployment.node_count(),
            "num_runs": self.num_runs,
            "max_iterations": self.max_iterations,
            "strict_timers": self.strict_timers,
            "events": events,
            "dependencies": self.dependencies,
        });
        Ok(ResolvedPlan { deployment: Arc::new(deployment), graph, json })
    }
}
