use spur_core::analysis::checker::TypeError;
use spur_core::analysis::resolver::ResolutionError;
use spur_core::compiler::{CompileResult, compile};

const TYPES: &str = "type Cluster { @quorum nodes: list<Node>; };\ntype Params { @scale n: int; };\n";

const ROLE: &str = "role Node(c: Cluster) {\n    fn Init() {}\n}\n";

const CLIENT: &str = "client KV(sys: Cluster) {
    async fn Write(dest: Node, key: string, uid: int) {}
    async fn Read(dest: Node, key: string): list<int> { [] }
}
";

const DEPLOY: &str = "@deploy(client = KV)
fn Main(p: Params): Cluster? {
    var nodes = spawn<Node>(p.n);
    var c = Cluster { nodes: nodes };
    provide_all(nodes, c);
    c
}
";

fn spec(role: &str, client: &str, extra: &str) -> String {
    format!("{TYPES}{role}{client}{DEPLOY}{extra}")
}

fn compiled(src: &str) -> CompileResult {
    let result = compile(src, "topology_checker.spur");
    assert!(result.lex_errors.is_empty(), "lex errors: {:?}", result.lex_errors);
    assert!(result.parse_errors.is_empty(), "parse errors: {:?}\n{src}", result.parse_errors);
    result
}

fn type_errors(src: &str) -> Vec<TypeError> {
    compiled(src).type_errors
}

macro_rules! assert_error {
    ($src:expr, $pattern:pat) => {{
        let src = $src;
        let errors = type_errors(&src);
        assert!(
            errors.iter().any(|e| matches!(e, $pattern)),
            "expected {} in {errors:#?}\n{src}",
            stringify!($pattern)
        );
    }};
}

#[test]
fn the_base_program_is_accepted_and_exports_its_deploy() {
    let result = compiled(&spec(ROLE, CLIENT, ""));
    assert!(result.resolution_errors.is_empty(), "{:?}", result.resolution_errors);
    assert!(result.type_errors.is_empty(), "{:#?}", result.type_errors);
    let program = result.program.expect("a program");
    assert_eq!(program.topology.deploys.len(), 1);
    let deploy = &program.topology.deploys[0];
    assert_eq!(deploy.name, "Main");
    assert_eq!(deploy.fields.len(), 1);
    assert_eq!(deploy.fields[0].name, "n");
    assert!(program.topology.warnings.is_empty(), "{:?}", program.topology.warnings);
}

#[test]
fn role_parameters_must_be_deployable() {
    assert_error!(
        spec("role Node(c: chan<int>) {}\n", CLIENT, ""),
        TypeError::RoleParamNotDeployable { .. }
    );
}

#[test]
fn role_parameters_are_read_only() {
    assert_error!(
        spec("role Node(c: Cluster) {\n    fn Init() { c = Cluster { nodes: [] }; }\n}\n", CLIENT, ""),
        TypeError::AssignToRoleParam { .. }
    );
}

#[test]
fn a_role_variable_may_not_shadow_the_parameter() {
    let result = compiled(&spec("role Node(c: Cluster) {\n    var c: int = 0;\n}\n", CLIENT, ""));
    assert!(
        result
            .resolution_errors
            .iter()
            .any(|e| matches!(e, ResolutionError::DuplicateName(name, _) if name == "c")),
        "{:?}",
        result.resolution_errors
    );
}

#[test]
fn init_takes_no_parameters() {
    assert_error!(
        spec("role Node(c: Cluster) {\n    fn Init(me: int) {}\n}\n", CLIENT, ""),
        TypeError::InitSignature { .. }
    );
    assert_error!(
        spec("role Node(c: Cluster) {\n    async fn RecoverInit(me: int) {}\n}\n", CLIENT, ""),
        TypeError::InitSignature { .. }
    );
}

#[test]
fn self_exists_only_in_roles_and_clients() {
    assert_error!(
        spec(ROLE, CLIENT, "fn f(): int {\n    var x = self;\n    0\n}\n"),
        TypeError::SelfOutsideRole { .. }
    );
}

#[test]
fn spawn_names_a_role() {
    assert_error!(
        spec(ROLE, CLIENT, "fn f(): int {\n    var x = spawn<KV>(1);\n    0\n}\n"),
        TypeError::SpawnNotRole { .. }
    );
    assert_error!(
        spec(ROLE, CLIENT, "fn f(): int {\n    var x = spawn<int>(1);\n    0\n}\n"),
        TypeError::SpawnNotRole { .. }
    );
}

#[test]
fn provide_targets_role_handles() {
    assert_error!(
        spec(ROLE, CLIENT, "fn f(): int {\n    provide(1, 2);\n    0\n}\n"),
        TypeError::ProvideTargetNotRole { .. }
    );
}

#[test]
fn provided_values_match_the_role_parameter() {
    assert_error!(
        spec(ROLE, CLIENT, "fn f(nodes: list<Node>): int {\n    provide_all(nodes, 5);\n    0\n}\n"),
        TypeError::ProvideValueType { .. }
    );
}

#[test]
fn a_deploy_may_not_reach_a_node_bound_operation() {
    assert_error!(
        spec(
            ROLE,
            CLIENT,
            "fn helper(): int {\n    persist_data(1);\n    0\n}\n@deploy(client = KV)\nfn Bound(): Cluster? {\n    var x = helper();\n    nil\n}\n"
        ),
        TypeError::DeployNodeBound { .. }
    );
}

#[test]
fn deploy_parameters_are_one_struct() {
    assert_error!(
        spec(ROLE, CLIENT, "@deploy(client = KV)\nfn Bad(n: int): Cluster? { nil }\n"),
        TypeError::DeployParamNotStruct { .. }
    );
}

#[test]
fn every_parameter_field_carries_one_tag() {
    let with_params = |fields: &str| {
        spec(
            ROLE,
            CLIENT,
            &format!("type P2 {{ {fields} }};\n@deploy(client = KV)\nfn Bad(p: P2): Cluster? {{ nil }}\n"),
        )
    };
    assert_error!(with_params("n: int;"), TypeError::ParamFieldUntagged { .. });
    assert_error!(with_params("@scale @choice n: int;"), TypeError::ParamFieldTwoTags { .. });
    assert_error!(with_params("@scale n: string;"), TypeError::ScaleType { .. });
    assert_error!(with_params("@choice n: list<int>;"), TypeError::ChoiceType { .. });
}

#[test]
fn deploy_marks_free_functions_only() {
    assert_error!(
        spec("role Node(c: Cluster) {\n    @deploy(client = KV)\n    fn Init() {}\n}\n", CLIENT, ""),
        TypeError::DeployNotFree { .. }
    );
}

#[test]
fn a_deploy_returns_an_optional_deployable_root() {
    assert_error!(
        spec(ROLE, CLIENT, "@deploy(client = KV)\nfn Bad(): Cluster { Cluster { nodes: [] } }\n"),
        TypeError::DeployReturnType { .. }
    );
}

#[test]
fn a_deploy_names_a_known_client() {
    assert_error!(
        spec(ROLE, CLIENT, "@deploy\nfn Bad(): Cluster? { nil }\n"),
        TypeError::DeployClientMissing { .. }
    );
    assert_error!(
        spec(ROLE, CLIENT, "@deploy(client = Nope)\nfn Bad(): Cluster? { nil }\n"),
        TypeError::DeployClientUnknown { .. }
    );
}

#[test]
fn the_client_parameter_equals_the_deploy_root() {
    assert_error!(
        spec(ROLE, CLIENT, "type Other { x: int; };\n@deploy(client = KV)\nfn Bad(): Other? { nil }\n"),
        TypeError::ClientParamMismatch { .. }
    );
}

#[test]
fn clients_define_async_read_and_write() {
    assert_error!(
        spec(ROLE, "client KV(sys: Cluster) {\n    async fn Write(dest: Node, key: string, uid: int) {}\n}\n", ""),
        TypeError::ClientOpMissing { .. }
    );
    assert_error!(
        spec(
            ROLE,
            "client KV(sys: Cluster) {\n    async fn Write(dest: Node, key: string, uid: int) {}\n    fn Read(dest: Node, key: string): list<int> { [] }\n}\n",
            ""
        ),
        TypeError::ClientOpSync { .. }
    );
    assert_error!(
        spec(
            ROLE,
            "client KV(sys: Cluster) {\n    async fn Write(dest: Node, key: string, uid: int) {}\n    async fn Read(dest: Node, key: int): list<int> { [] }\n}\n",
            ""
        ),
        TypeError::ClientOpSignature { .. }
    );
}

#[test]
fn tags_are_checked_for_placement_and_identity() {
    assert_error!(spec(ROLE, CLIENT, "@scale\nfn f(): int { 0 }\n"), TypeError::TagPlacement { .. });
    assert_error!(spec(ROLE, CLIENT, "type Q { @quorum n: int; };\n"), TypeError::QuorumType { .. });
    assert_error!(spec(ROLE, CLIENT, "@bogus\nfn f(): int { 0 }\n"), TypeError::UnknownTag { .. });
    assert_error!(
        spec("role Node(c: Cluster) {\n    @trace\n    @trace\n    async fn H() {}\n}\n", CLIENT, ""),
        TypeError::DuplicateTag { .. }
    );
}

#[test]
fn a_parameter_tag_outside_any_deploy_is_reported() {
    let result = compiled(&spec(ROLE, CLIENT, "type Unused { @scale n: int; };\n"));
    assert!(result.type_errors.is_empty(), "{:#?}", result.type_errors);
    let warnings = result.program.expect("a program").topology.warnings;
    assert!(warnings.iter().any(|w| w.starts_with("TagWithoutConsumer")), "{warnings:?}");
}
