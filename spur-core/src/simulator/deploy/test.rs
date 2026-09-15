use rand::{SeedableRng, rngs::SmallRng};
use serde_json::json;

use super::params::{DeployCache, DeployOutcome};
use super::*;
use crate::simulator::core::{PurgatoryConfig, RuntimeError, SchedulePolicy, State, build_frame, exec_sync_on_node};
use crate::simulator::feedback::NoFeedback;
use crate::simulator::hash_utils::NoHashing;
use crate::simulator::path::Logs;

const SPEC: &str = r#"
type Cluster { @quorum nodes: list<Node>; };
type Mixed { @quorum a: list<Node>; w: list<Witness>; b: list<Node>; };
type Params { @scale n: int; };

role Node(c: Cluster) {
    var me: int = index_of(c.nodes, self)!;
    fn Init() {}
}

role Witness(w: int) {
    fn Init() {}
}

client KV(sys: Cluster) {
    async fn Write(dest: Node, key: string, uid: int) {}
    async fn Read(dest: Node, key: string): list<int> { [] }
}

client MixedKV(sys: Mixed) {
    async fn Write(dest: Node, key: string, uid: int) {}
    async fn Read(dest: Witness, key: string): list<int> { [] }
}

fn cluster(n: int): Cluster {
    var nodes = spawn<Node>(n);
    var c = Cluster { nodes: nodes };
    provide_all(nodes, c);
    c
}

@deploy(client = KV)
fn Odd(p: Params): Cluster? {
    if p.n < 1 { return nil; }
    if p.n % 2 == 0 { return cluster(p.n + 1); }
    cluster(p.n)
}

@deploy(client = MixedKV)
fn Interleaved(): Mixed? {
    var a = spawn<Node>(2);
    var w = spawn<Witness>(1);
    var b = spawn<Node>(2);
    provide_all(a, Cluster { nodes: a });
    provide_all(w, 7);
    provide_all(b, Cluster { nodes: b });
    Mixed { a: a, w: w, b: b }
}

@deploy(client = KV)
fn Twice(): Cluster? {
    var nodes = spawn<Node>(1);
    provide(nodes[0], Cluster { nodes: nodes });
    provide(nodes[0], Cluster { nodes: nodes });
    Cluster { nodes: nodes }
}

@deploy(client = KV)
fn Missing(): Cluster? {
    var nodes = spawn<Node>(2);
    provide(nodes[0], Cluster { nodes: nodes });
    Cluster { nodes: nodes }
}

@deploy(client = KV)
fn Hidden(): Cluster? {
    var nodes = spawn<Node>(3);
    var c = Cluster { nodes: [nodes[0], nodes[1]] };
    provide_all(nodes, c);
    c
}

@deploy(client = KV)
fn Broken(): Cluster? {
    var nodes = spawn<Node>(1);
    provide_all(nodes, Cluster { nodes: nodes });
    var absent: int? = nil;
    var forced = absent!;
    Cluster { nodes: nodes }
}
"#;

fn program() -> Program {
    crate::compiler::compile(SPEC, "deploy_test.spur")
        .into_program()
        .expect("the deploy test spec compiles")
}

fn build(program: &Program, name: &str, params: serde_json::Value) -> Result<Option<Deployment>, String> {
    let deploy = select_deploy(program, Some(name)).unwrap();
    evaluate_deploy(program, deploy, &params)
}

fn role_names(program: &Program, deployment: &Deployment) -> Vec<String> {
    deployment
        .nodes
        .iter()
        .map(|n| program.id_to_name[&n.role].clone())
        .collect()
}

#[test]
fn spawn_calls_take_global_indices_in_call_order() {
    let program = program();
    let d = build(&program, "Interleaved", json!({})).unwrap().unwrap();
    assert_eq!(
        d.nodes.iter().map(|n| n.index).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(role_names(&program, &d), vec!["Node", "Node", "Witness", "Node", "Node"]);
    assert_eq!(d.ordinals, vec![0, 1, 0, 2, 3]);
    assert_eq!(
        d.paths,
        vec![
            Some("a[0]".to_string()),
            Some("a[1]".to_string()),
            Some("w[0]".to_string()),
            Some("b[0]".to_string()),
            Some("b[1]".to_string()),
        ]
    );
    let groups: Vec<_> = d
        .groups
        .iter()
        .map(|g| (g.paths.clone(), g.members.iter().map(|n| n.index).collect::<Vec<_>>(), g.quorum))
        .collect();
    assert_eq!(
        groups,
        vec![
            (vec!["a".to_string()], vec![0, 1], true),
            (vec!["w".to_string()], vec![2], false),
            (vec!["b".to_string()], vec![3, 4], false),
        ]
    );
    assert_eq!(d.fanout_width, vec![1, 1, 0, 1, 1]);
}

#[test]
fn a_nil_result_rejects_the_tuple() {
    let program = program();
    assert!(build(&program, "Odd", json!({"n": 0})).unwrap().is_none());
}

#[test]
fn a_handle_provided_twice_fails_at_the_second_provide() {
    let program = program();
    let error = build(&program, "Twice", json!({})).unwrap_err();
    assert!(error.contains("provided twice"), "{error}");
}

#[test]
fn a_handle_never_provided_fails_when_the_deploy_returns() {
    let program = program();
    let error = build(&program, "Missing", json!({})).unwrap_err();
    assert!(error.contains("without provide"), "{error}");
}

#[test]
fn an_unreachable_handle_still_runs_but_has_no_path() {
    let program = program();
    let d = build(&program, "Hidden", json!({})).unwrap().unwrap();
    assert_eq!(d.node_count(), 3);
    assert_eq!(d.paths[2], None);
    assert_eq!(d.crash_candidates, vec![0, 1, 2]);
    assert_eq!(d.fanout_width[2], 2);
}

#[test]
fn a_runtime_error_in_the_deploy_is_a_deploy_error() {
    let program = program();
    let error = build(&program, "Broken", json!({})).unwrap_err();
    assert!(error.starts_with("deploy Broken with params {}"), "{error}");
}

#[test]
fn equal_deployments_from_different_tuples_share_one_id() {
    let program = program();
    let deploy = select_deploy(&program, Some("Odd")).unwrap();
    let mut cache = DeployCache::default();
    let id = |outcome| match outcome {
        DeployOutcome::Built { id, .. } => Some(id),
        DeployOutcome::Rejected => None,
    };
    assert_eq!(id(cache.get(&program, deploy, &json!({"n": 5})).unwrap()), Some(0));
    assert_eq!(id(cache.get(&program, deploy, &json!({"n": 4})).unwrap()), Some(0));
    assert_eq!(id(cache.get(&program, deploy, &json!({"n": 3})).unwrap()), Some(1));
    assert_eq!(id(cache.get(&program, deploy, &json!({"n": 0})).unwrap()), None);
    assert_eq!(id(cache.get(&program, deploy, &json!({"n": 4})).unwrap()), Some(0));
    assert_eq!(cache.deployments.len(), 2);
    assert_eq!(cache.tuples_aliased, 1);
    assert_eq!(cache.rejections, 1);
    assert_eq!(cache.aliases[0], vec![json!({"n": 4})]);
}

#[test]
fn allocation_without_an_allocator_is_a_runtime_error() {
    let program = program();
    let deploy = select_deploy(&program, Some("Hidden")).unwrap();
    let function = &program.rpc[&deploy.id];
    let node = NodeId { role: NameId(usize::MAX - 1), index: 0 };
    let mut state = State::<NoHashing>::new(&[(node.role, 1)], program.max_node_slots as usize);
    let args: Vec<Value<NoHashing>> = vec![];
    let mut frame = build_frame(function, &args);
    let mut logs = Logs::default();
    let mut rng = SmallRng::seed_from_u64(0);
    let result = exec_sync_on_node::<NoHashing, _, NoFeedback>(
        &mut state,
        &mut logs,
        &program,
        &mut frame,
        node,
        function.entry,
        &(),
        &mut (),
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut rng,
    );
    assert!(matches!(result, Err(RuntimeError::AllocationOutsideDeploy)), "{result:?}");
}

fn bind(program: &Program, name: &str, params: serde_json::Value, cache: &Arc<std::sync::Mutex<DeployCache>>) -> Result<super::space::DeploySpace, String> {
    super::space::DeploySpace::bind(program, Some(name), &params, cache)
}

#[test]
fn a_bound_space_lists_each_deployment_once_smallest_first() {
    let program = program();
    let cache = Arc::new(std::sync::Mutex::new(DeployCache::default()));
    let space = bind(&program, "Odd", json!({"n": {"min": 0, "max": 5}}), &cache).unwrap();
    let grid: Vec<_> = space
        .grid()
        .iter()
        .map(|s| (s.params.values.clone(), s.deployment.node_count()))
        .collect();
    assert_eq!(grid, vec![(json!({"n": 1}), 1), (json!({"n": 2}), 3), (json!({"n": 4}), 5)]);
    let again = bind(&program, "Odd", json!({"n": {"min": 3, "max": 3}}), &cache).unwrap();
    assert_eq!(again.grid()[0].id, space.grid()[1].id, "ids are shared across binds of one session");
}

#[test]
fn draws_never_select_a_rejected_tuple() {
    let program = program();
    let cache = Arc::new(std::sync::Mutex::new(DeployCache::default()));
    let space = bind(&program, "Odd", json!({"n": {"min": 0, "max": 5}}), &cache).unwrap();
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..200 {
        let drawn = space.random(&mut rng);
        assert_ne!(drawn.params.values, json!({"n": 0}));
        let child = space.mutate(&drawn, &mut rng);
        assert_ne!(child.params.values, json!({"n": 0}));
    }
    assert_eq!(space.lower(0.0).params.values, json!({"n": 1}), "a rejected lowering falls back to the smallest");
    assert_eq!(space.lower(1.0).params.values, json!({"n": 5}));
    assert_eq!(space.lower(1.0).deployment.node_count(), 5);
}

#[test]
fn binding_rejects_oversized_spaces_and_missing_params() {
    let program = program();
    let cache = Arc::new(std::sync::Mutex::new(DeployCache::default()));
    let error = bind(&program, "Odd", json!({"n": {"min": 1, "max": 5000}}), &cache).unwrap_err();
    assert!(error.contains("more than"), "{error}");
    let error = bind(&program, "Odd", serde_json::Value::Null, &cache).unwrap_err();
    assert!(error.contains("params is required"), "{error}");
    let space = bind(&program, "Interleaved", serde_json::Value::Null, &cache).unwrap();
    assert_eq!(space.grid().len(), 1);
}

/// A value's structure with the per-policy signatures and marker types removed.
fn shape<H: crate::simulator::hash_utils::HashPolicy>(value: &Value<H>) -> String {
    let text = format!("{:?}", value.kind).replace("WithHashing", "NoHashing");
    let mut out = String::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("sig: ") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + "sig: ".len()..];
        rest = &tail[tail.find(',').expect("a signature is followed by a comma") + 1..];
    }
    out.push_str(rest);
    out
}

#[test]
fn contexts_are_shared_under_no_hashing_and_copied_under_hashing() {
    use crate::simulator::hash_utils::WithHashing;
    let program = program();
    let d = build(&program, "Interleaved", json!({})).unwrap().unwrap();
    let plain: Value<NoHashing> = d.context(0);
    let hashed: Value<WithHashing> = d.context(0);
    assert_eq!(shape(&plain), shape(&hashed));
    assert_eq!(shape(&d.root_value::<WithHashing>()), shape(&d.root));
    assert!(d.functions(d.client_role).write.is_some());
    assert!(d.functions(d.nodes[2].role).init.is_some());
    assert_eq!(d.model(), "kv");
}
