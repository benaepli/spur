pub mod allocator;
pub mod evaluate;
pub mod params;
pub mod path;
pub mod space;
pub use evaluate::{evaluate_deploy, select_deploy};
use std::any::Any;
use std::sync::Arc;

use spur_ast::types::DeployMetadata;

use super::core::state::NodeId;
use super::core::values::{Value, ValueKind};
use super::hash_utils::{HashPolicy, NoHashing};
use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{FunctionInfo, Program};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoleFunctions {
    /// Whether the role parameter occupies a node slot; when it does not, the
    /// initializers, Init and RecoverInit take it as an argument.
    pub param_in_env: bool,
    pub base_init: Option<FunctionInfo>,
    pub init: Option<FunctionInfo>,
    pub recover_init: Option<FunctionInfo>,
    pub write: Option<FunctionInfo>,
    pub read: Option<FunctionInfo>,
    pub rmw: Option<FunctionInfo>,
}

/// The functions runs call on each role and client, indexed by role id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoleTable {
    roles: Vec<RoleFunctions>,
}

impl RoleTable {
    pub fn new(program: &Program) -> Self {
        let len = program.roles.iter().map(|(r, _)| r.0 + 1).max().unwrap_or(0);
        let mut roles = vec![RoleFunctions::default(); len];
        for (id, name) in &program.roles {
            let get = |suffix| program.get_func_by_name(&format!("{name}.{suffix}")).cloned();
            roles[id.0] = RoleFunctions {
                param_in_env: program.topology.roles.get(id).is_none_or(|r| r.param_in_env),
                base_init: get("BASE_NODE_INIT"),
                init: get("Init"),
                recover_init: get("RecoverInit"),
                write: get("Write"),
                read: get("Read"),
                rmw: get("RMW"),
            };
        }
        Self { roles }
    }

    pub fn functions(&self, role: NameId) -> &RoleFunctions {
        &self.roles[role.0]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub paths: Vec<String>,
    pub role: NameId,
    pub members: Vec<NodeId>,
    pub quorum: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Deployment {
    pub spec: DeployMetadata,
    pub root: Value<NoHashing>,
    pub contexts: Vec<Value<NoHashing>>,
    pub ordinals: Vec<usize>,
    pub paths: Vec<Option<String>>,
    pub hash: u64,
    pub canonical_params: serde_json::Value,
    pub nodes: Arc<[NodeId]>,
    pub groups: Vec<Group>,
    pub fanout_width: Vec<u32>,
    pub crash_candidates: Vec<usize>,
    pub peer_indices: Vec<Vec<usize>>,
    pub client_role: NameId,
    /// Destination candidates of Write, Read and RMW, in index order; `None`
    /// for an operation that takes no destination or is not defined.
    pub destinations: [Option<Vec<NodeId>>; 3],
    pub roles: Arc<RoleTable>,
}

impl Deployment {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn functions(&self, role: NameId) -> &RoleFunctions {
        self.roles.functions(role)
    }

    /// The Porcupine model the client's operations define.
    pub fn model(&self) -> &'static str {
        if self.functions(self.client_role).rmw.is_some() { "kv_rmw" } else { "kv" }
    }

    /// The role parameter of the node at `index`.
    pub fn context<H: HashPolicy>(&self, index: usize) -> Value<H> {
        adopt(&self.contexts[index])
    }

    /// The deployment root, which is every client's parameter.
    pub fn root_value<H: HashPolicy>(&self) -> Value<H> {
        adopt(&self.root)
    }

    pub fn peers(&self, node: usize) -> &[usize] {
        &self.peer_indices[node]
    }

    pub fn can_retarget(&self, planned: usize, candidate: usize) -> bool {
        self.peer_indices[planned].contains(&candidate)
    }
}

/// Runs under `NoHashing` share the deployment's values; any other policy
/// gets a structural copy.
fn adopt<H: HashPolicy>(value: &Value<NoHashing>) -> Value<H> {
    match (value as &dyn Any).downcast_ref::<Value<H>>() {
        Some(shared) => shared.clone(),
        None => convert(value),
    }
}

fn convert<H: HashPolicy>(value: &Value<NoHashing>) -> Value<H> {
    let boxed = |v: &Arc<Value<NoHashing>>| Arc::new(convert(v));
    match &value.kind {
        ValueKind::Int(i) => Value::int(*i),
        ValueKind::Bool(b) => Value::bool(*b),
        ValueKind::String(s) => Value::string(s.clone()),
        ValueKind::Unit => Value::unit(),
        ValueKind::Node(n) => Value::node(*n),
        ValueKind::List(xs) => Value::list(xs.iter().map(convert).collect()),
        ValueKind::Tuple(xs) => Value::tuple(xs.iter().map(convert).collect()),
        ValueKind::Option(v) => Value::option(v.as_ref().map(boxed)),
        ValueKind::Variant(id, name, payload) => Value::variant(*id, name.clone(), payload.as_ref().map(boxed)),
        ValueKind::Struct(shape, fields) => Value::struct_of(shape, fields.iter().map(convert).collect()),
        ValueKind::Map(map) => Value::map(map.iter().map(|(k, v)| (convert(k), convert(v))).collect()),
        _ => unreachable!("deployable values hold no node-owned resources"),
    }
}

#[cfg(test)]
impl Deployment {
    /// One quorum group of `count` nodes of one role, every client operation
    /// addressed to it.
    pub fn test_cluster(count: usize) -> Self {
        use spur_ast::types::Type;
        let role = NameId(0);
        let nodes: Arc<[NodeId]> = (0..count).map(|index| NodeId { role, index }).collect();
        Deployment {
            spec: DeployMetadata {
                id: NameId(2),
                name: "Main".into(),
                client: NameId(1),
                parameter: None,
                fields: vec![],
                root: Type::Tuple(vec![]),
            },
            root: Value::unit(),
            contexts: vec![Value::unit(); count],
            ordinals: (0..count).collect(),
            paths: (0..count).map(|i| Some(format!("nodes[{i}]"))).collect(),
            hash: 0,
            canonical_params: serde_json::json!({}),
            groups: vec![Group { paths: vec!["nodes".into()], role, members: nodes.to_vec(), quorum: true }],
            fanout_width: vec![count.saturating_sub(1) as u32; count],
            crash_candidates: (0..count).collect(),
            peer_indices: vec![(0..count).collect(); count],
            client_role: NameId(1),
            destinations: [Some(nodes.to_vec()), Some(nodes.to_vec()), Some(nodes.to_vec())],
            nodes,
            roles: Arc::default(),
        }
    }
}

#[cfg(test)]
mod test;
