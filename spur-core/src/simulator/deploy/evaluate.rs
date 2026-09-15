use super::*;
use super::allocator::Allocator;
use crate::simulator::core::{PurgatoryConfig, SchedulePolicy, State, build_frame, exec_sync_on_node};
use crate::simulator::core::values::ValueKind;
use crate::simulator::feedback::NoFeedback;
use crate::simulator::hash_utils::NoHashing;
use crate::simulator::path::Logs;
use rand::{SeedableRng, rngs::SmallRng};
use serde_json::{Value as Json, json};
use spur_ast::types::{DeployMetadata, TopologyMetadata, Type};
use std::collections::HashSet;

pub fn select_deploy<'a>(program: &'a Program, name: Option<&str>) -> Result<&'a DeployMetadata, String> {
    if let Some(name) = name {
        program.topology.deploys.iter().find(|d| d.name == name).ok_or_else(|| format!("unknown deploy {name}"))
    } else {
        match program.topology.deploys.as_slice() {
            [deploy] => Ok(deploy),
            [] => Err("the program has no @deploy function".into()),
            _ => Err("select a deploy by name when the program has multiple @deploy functions".into()),
        }
    }
}

pub fn json_value(value: &Json, ty: &Type, schema: &TopologyMetadata) -> Result<Value<NoHashing>, String> {
    let error = || format!("expected {ty}, got {value}");
    match ty {
        Type::Int => value.as_i64().map(Value::int).ok_or_else(error),
        Type::Bool => value.as_bool().map(Value::bool).ok_or_else(error),
        Type::String => value.as_str().map(|s| Value::string(s.into())).ok_or_else(error),
        Type::Enum(id, _) => {
            let name = value.as_str().ok_or_else(error)?;
            if !schema.enums[id].iter().any(|(_, n, p)| n == name && p.is_none()) { return Err(error()); }
            Ok(Value::new(ValueKind::Variant(id.0 as u32, name.into(), None)))
        }
        Type::Struct(id, _) => {
            let object = value.as_object().ok_or_else(error)?;
            let fields = &schema.structs[id];
            if object.len() != fields.len() || object.keys().any(|k| !fields.iter().any(|(_, n, _)| n == k)) { return Err("parameter fields must exactly match the deploy parameter struct".into()); }
            let values = fields.iter().map(|(_, name, ty)| Ok((Value::string(name.as_str().into()), json_value(object.get(name).ok_or_else(|| format!("missing parameter {name}"))?, ty, schema)?))).collect::<Result<_, String>>()?;
            Ok(Value::map(values))
        }
        _ => Err(error()),
    }
}

pub fn field<'a>(value: &'a Value<NoHashing>, name: &str) -> Option<&'a Value<NoHashing>> {
    match &value.kind {
        ValueKind::Struct(shape, fields) => shape.names().iter().position(|n| n == name).map(|i| &fields[i]),
        ValueKind::Map(map) => map.get(&Value::string(name.into())),
        _ => None,
    }
}

pub fn canonical_value(value: &Value<NoHashing>) -> Json {
    match &value.kind {
        ValueKind::Int(i) => json!(["int", i]),
        ValueKind::Bool(b) => json!(["bool", b]),
        ValueKind::String(s) => json!(["string", s.as_str()]),
        ValueKind::Unit => json!(["unit"]),
        ValueKind::Node(n) => json!(["node", n.role.0, n.index]),
        ValueKind::List(xs) => json!(["list", xs.iter().map(canonical_value).collect::<Vec<_>>()]),
        ValueKind::Tuple(xs) => json!(["tuple", xs.iter().map(canonical_value).collect::<Vec<_>>()]),
        ValueKind::Option(v) => json!(["optional", v.as_ref().map(|v| canonical_value(v))]),
        ValueKind::Variant(id, name, payload) => json!(["enum", id, name.as_str(), payload.as_ref().map(|v| canonical_value(v))]),
        ValueKind::Map(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            json!(["map", entries.into_iter().map(|(k, v)| json!([canonical_value(k), canonical_value(v)])).collect::<Vec<_>>()])
        }
        ValueKind::Struct(shape, fields) => {
            let mut entries: Vec<_> = shape.names().iter().zip(fields.iter()).collect();
            entries.sort_by_key(|(n, _)| *n);
            json!(["map", entries.into_iter().map(|(n, v)| json!([["string", n.as_str()], canonical_value(v)])).collect::<Vec<_>>()])
        }
        _ => unreachable!("deployable values cannot contain node-owned resources"),
    }
}

pub fn evaluate_deploy(program: &Program, deploy: &DeployMetadata, params: &Json) -> Result<Option<Deployment>, String> {
    let result = evaluate_inner(program, deploy, params);
    result.map_err(|e| format!("deploy {} with params {}: {e}", deploy.name, params))
}

fn evaluate_inner(program: &Program, deploy: &DeployMetadata, params: &Json) -> Result<Option<Deployment>, String> {
    let args = match &deploy.parameter {
        Some(ty) => vec![json_value(params, ty, &program.topology)?],
        None if params.as_object().is_some_and(|p| p.is_empty()) => vec![],
        None => return Err("this deploy takes no parameters".into()),
    };
    let pseudo = NodeId { role: NameId(usize::MAX - 1), index: 0 };
    let mut state = State::<NoHashing>::new(&[(pseudo.role, 1)], program.max_node_slots as usize);
    state.allocator = Some(Box::new(Allocator { nodes: vec![] }));
    let function = &program.rpc[&deploy.id];
    let mut frame = build_frame(function, &args);
    let mut logs = Logs::default();
    let mut rng = SmallRng::seed_from_u64(0);
    let root = exec_sync_on_node::<NoHashing, _, NoFeedback>(&mut state, &mut logs, program, &mut frame, pseudo, function.entry, &(), &mut (), &SchedulePolicy::Fixed, &PurgatoryConfig::default(), &mut rng).map_err(|e| e.to_string())?;
    if !logs.entries.is_empty() { log::info!("deploy {}: {}", deploy.name, logs.text.log_content.str_from(0)); }
    let allocator = state.allocator.take().unwrap();
    allocator.finish().map_err(|e| e.to_string())?;
    let root = match root.kind { ValueKind::Option(None) => return Ok(None), ValueKind::Option(Some(v)) => v.as_ref().clone(), _ => return Err("deploy result is not optional".into()) };
    let count = allocator.nodes.len();
    let nodes: Arc<[NodeId]> = allocator.nodes.iter().map(|n| n.id).collect();
    let contexts = allocator.nodes.iter().map(|n| n.ctx.clone().unwrap()).collect();
    let ordinals = allocator.nodes.iter().map(|n| n.ordinal).collect();
    let mut deployment = Deployment {
        nodes, contexts, ordinals,
        spec: Some(deploy.clone()), root, hash: 0, canonical_params: params.clone(),
        paths: vec![None; count], groups: vec![], fanout_width: vec![0; count], crash_candidates: (0..count).collect(),
        peer_indices: vec![vec![]; count], client_role: deploy.client, roles: program.deployments.roles.clone(),
    };
    let mut reachable = HashSet::new();
    walk(&deployment.root.clone(), &deploy.root, Some("$".into()), false, &program.topology, &mut deployment, &mut reachable);
    for node in deployment.nodes.iter() {
        if !reachable.contains(&node.index) { log::warn!("UnreachableNode: {}[{}] in deploy {}", program.id_to_name[&node.role], deployment.ordinals[node.index], deploy.name); }
        let groups: Vec<_> = deployment.groups.iter().filter(|g| g.members.contains(node)).collect();
        deployment.fanout_width[node.index] = groups.iter().map(|g| g.members.len().saturating_sub(1)).max().unwrap_or(count.saturating_sub(1)) as u32;
        deployment.peer_indices[node.index] = if groups.is_empty() {
            deployment.nodes.iter().filter(|n| n.role == node.role).map(|n| n.index).collect()
        } else { deployment.nodes.iter().filter(|n| groups.iter().any(|g| g.members.contains(n))).map(|n| n.index).collect() };
    }
    let identity = json!([deploy.id.0, canonical_value(&deployment.root), deployment.nodes.iter().zip(&deployment.contexts).map(|(n, ctx)| json!([n.role.0, canonical_value(ctx)])).collect::<Vec<_>>()]);
    let mut hash = 0xcbf29ce484222325u64;
    for byte in identity.to_string().bytes() { hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3); }
    deployment.hash = hash;
    Ok(Some(deployment))
}

fn child_path(parent: &Option<String>, suffix: &str) -> Option<String> {
    parent.as_ref().map(|p| if p == "$" && suffix.starts_with('.') && suffix[1..].chars().next().is_some_and(|c| !c.is_ascii_digit()) { suffix[1..].into() } else if p == "$" && suffix.starts_with('[') { suffix.into() } else { format!("{p}{suffix}") })
}

fn walk(value: &Value<NoHashing>, ty: &Type, path: Option<String>, quorum: bool, schema: &TopologyMetadata, deployment: &mut Deployment, reachable: &mut HashSet<usize>) {
    match (ty, &value.kind) {
        (Type::Role(_, _), ValueKind::Node(node)) => {
            reachable.insert(node.index);
            if deployment.paths[node.index].is_none() { deployment.paths[node.index] = path; }
        }
        (Type::List(element), ValueKind::List(xs)) => {
            if let Type::Role(role, _) = element.as_ref() {
                let members: Vec<_> = xs.iter().map(|v| v.as_node().unwrap()).collect();
                if let Some(group) = deployment.groups.iter_mut().find(|g| g.role == *role && g.members == members) {
                    group.quorum |= quorum;
                    if let Some(p) = &path { if !group.paths.contains(p) { group.paths.push(p.clone()); } }
                } else { deployment.groups.push(Group { role: *role, members, quorum, paths: path.iter().cloned().collect() }); }
            }
            for (i, x) in xs.iter().enumerate() { walk(x, element, child_path(&path, &format!("[{i}]")), false, schema, deployment, reachable); }
        }
        (Type::Struct(id, _), _) => {
            for (fid, name, t) in &schema.structs[id] {
                let tagged = schema.tags.get(fid).is_some_and(|tags| tags.iter().any(|a| a.name == "quorum"));
                walk(field(value, name).unwrap(), t, child_path(&path, &format!(".{name}")), tagged, schema, deployment, reachable);
            }
        }
        (Type::Tuple(types), ValueKind::Tuple(xs)) => { for (i, (x, t)) in xs.iter().zip(types).enumerate() { walk(x, t, child_path(&path, &format!(".{i}")), false, schema, deployment, reachable); } }
        (Type::Map(kt, t), ValueKind::Map(map)) => {
            let mut entries: Vec<_> = map.iter().collect(); entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            for (k, v) in entries {
                walk(k, kt, None, false, schema, deployment, reachable);
                let p = match &k.kind { ValueKind::Int(i) => child_path(&path, &format!("[{i}]")), ValueKind::String(s) => child_path(&path, &format!("[{}]", json!(s.as_str()))), _ => None };
                walk(v, t, p, false, schema, deployment, reachable);
            }
        }
        (Type::Map(kt, t), ValueKind::Struct(shape, fields)) if **kt == Type::String => {
            let mut entries: Vec<_> = shape.names().iter().zip(fields.iter()).collect();
            entries.sort_by_key(|(name, _)| *name);
            for (name, v) in entries {
                walk(v, t, child_path(&path, &format!("[{}]", json!(name.as_str()))), false, schema, deployment, reachable);
            }
        }
        (Type::Optional(t), ValueKind::Option(Some(v))) => walk(v, t, path, quorum, schema, deployment, reachable),
        (Type::Enum(id, _), ValueKind::Variant(_, name, Some(v))) => {
            if let Some((_, _, Some(t))) = schema.enums[id].iter().find(|(_, n, _)| n == name.as_str()) { walk(v, t, None, false, schema, deployment, reachable); }
        }
        _ => {}
    }
}

impl Deployment {
    pub fn report(&self, program: &Program) -> Json {
        let mut roles = serde_json::Map::new();
        for n in self.nodes.iter() {
            let key = &program.id_to_name[&n.role];
            let count = roles.get(key).and_then(Json::as_u64).unwrap_or(0);
            roles.insert(key.clone(), json!(count + 1));
        }
        json!({ "deploy": self.spec.as_ref().map(|d| &d.name), "hash": self.hash, "params": self.canonical_params, "roles": roles,
            "nodes": self.nodes.iter().map(|n| json!({"index": n.index, "role": program.id_to_name[&n.role], "ordinal": self.ordinals[n.index], "path": self.paths[n.index]})).collect::<Vec<_>>(),
            "groups": self.groups.iter().map(|g| json!({"path": g.paths.first(), "aliases": g.paths.iter().skip(1).collect::<Vec<_>>(), "role": program.id_to_name[&g.role], "members": g.members.iter().map(|n| n.index).collect::<Vec<_>>(), "quorum": g.quorum})).collect::<Vec<_>>() })
    }
}
