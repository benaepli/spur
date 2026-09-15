pub mod allocator;
pub mod evaluate;
pub mod params;
pub mod path;
pub use evaluate::{evaluate_deploy, select_deploy};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::core::error::RuntimeError;
use super::core::state::NodeId;
use super::core::values::Value;
use super::hash_utils::HashPolicy;
use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{FunctionInfo, Program};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoleFunctions {
    pub base_init: Option<FunctionInfo>,
    pub init: Option<FunctionInfo>,
    pub recover_init: Option<FunctionInfo>,
    pub write: Option<FunctionInfo>,
    pub read: Option<FunctionInfo>,
    pub rmw: Option<FunctionInfo>,
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
    pub spec: Option<spur_ast::types::DeployMetadata>,
    pub root: Value<super::hash_utils::NoHashing>,
    pub contexts: Vec<Value<super::hash_utils::NoHashing>>,
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
    pub roles: Arc<Vec<RoleFunctions>>,
}

impl Deployment {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn functions(&self, role: NameId) -> &RoleFunctions {
        &self.roles[role.0]
    }

    pub fn peer_list<H: HashPolicy>(&self) -> Value<H> {
        Value::list(self.nodes.iter().copied().map(Value::node).collect())
    }

    pub fn init_args<H: HashPolicy>(&self, node: NodeId, peers: Value<H>) -> [Value<H>; 2] {
        [Value::int(node.index as i64), peers]
    }

    pub fn peers(&self, node: usize) -> &[usize] {
        &self.peer_indices[node]
    }

    pub fn can_retarget(&self, planned: usize, candidate: usize) -> bool {
        self.peer_indices[planned].contains(&candidate)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeploymentCatalog {
    roles: Arc<Vec<RoleFunctions>>,
    server: Option<NameId>,
    client: Option<NameId>,
    cache: Arc<Mutex<HashMap<usize, Arc<Deployment>>>>,
}

impl PartialEq for DeploymentCatalog {
    fn eq(&self, other: &Self) -> bool {
        self.roles == other.roles && self.server == other.server && self.client == other.client
    }
}

impl DeploymentCatalog {
    pub fn new(program: &Program) -> Self {
        let mut roles = vec![
            RoleFunctions::default();
            program
                .roles
                .iter()
                .map(|(r, _)| r.0 + 1)
                .max()
                .unwrap_or(0)
        ];
        for (id, name) in &program.roles {
            let get = |suffix| {
                program
                    .get_func_by_name(&format!("{name}.{suffix}"))
                    .cloned()
            };
            roles[id.0] = RoleFunctions {
                base_init: get("BASE_NODE_INIT"),
                init: get("Init"),
                recover_init: get("RecoverInit"),
                write: get("Write"),
                read: get("Read"),
                rmw: get("RMW"),
            };
        }
        Self {
            roles: Arc::new(roles),
            server: program
                .roles
                .iter()
                .find(|(_, n)| n == "Node")
                .map(|(r, _)| *r),
            client: program
                .roles
                .iter()
                .find(|(_, n)| n == "ClientInterface")
                .map(|(r, _)| *r),
            cache: Arc::default(),
        }
    }

    pub fn functions(&self, role: NameId) -> &RoleFunctions {
        &self.roles[role.0]
    }

    pub fn get(&self, count: usize) -> Result<Arc<Deployment>, RuntimeError> {
        let server = self
            .server
            .ok_or_else(|| RuntimeError::RoleNotFound("Node".into()))?;
        let client_role = self
            .client
            .ok_or_else(|| RuntimeError::RoleNotFound("ClientInterface".into()))?;
        let mut cache = self.cache.lock().unwrap();
        Ok(cache
            .entry(count)
            .or_insert_with(|| {
                let nodes: Arc<[NodeId]> = (0..count)
                    .map(|index| NodeId {
                        role: server,
                        index,
                    })
                    .collect();
                Arc::new(Deployment {
                    spec: None, root: Value::unit(), contexts: vec![], ordinals: (0..count).collect(), paths: vec![None; count], hash: 0, canonical_params: serde_json::json!({}),
                    groups: vec![Group {
                        paths: vec!["nodes".into()],
                        role: server,
                        members: nodes.to_vec(),
                        quorum: true,
                    }],
                    fanout_width: vec![count.saturating_sub(1) as u32; count],
                    crash_candidates: (0..count).collect(),
                    peer_indices: vec![(0..count).collect(); count],
                    nodes,
                    client_role,
                    roles: self.roles.clone(),
                })
            })
            .clone())
    }
}

#[cfg(test)]
mod test;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deployments_share_instances_and_resolve_role_functions() {
        let program = crate::compiler::compile(
            include_str!("../../tests/fixtures/canchor.spur"),
            "canchor.spur",
        )
        .into_program()
        .unwrap();
        let a = program.deployments.get(3).unwrap();
        let b = program.deployments.get(3).unwrap();
        let c = program.deployments.get(5).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(
            a.nodes.iter().map(|n| n.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(a.groups[0].members.as_slice(), a.nodes.as_ref());
        assert_eq!(a.fanout_width, vec![2; 3]);
        assert!(a.functions(a.client_role).write.is_some());
        assert!(a.functions(a.nodes[0].role).base_init.is_some());
    }
}
