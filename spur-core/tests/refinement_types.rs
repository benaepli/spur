use spur_core::compiler::compile;

fn minimal_spec_with_refinement(refinement: &str) -> String {
    format!(
        r#"
type Pos = {refinement};

role Node {{
    var x: Pos = 0;

    fn Init(me: int) {{}}

    @trace
    async fn HandleRequest(req: int) {{
        var val: Pos = req;
    }}
}}

ClientInterface {{
    async fn Write(dest: Node, key: string, value: string) {{ }}
    async fn Read(dest: Node, key: string): string? {{ nil }}
}}
"#
    )
}

#[test]
fn test_parse_refined_int() {
    let src = minimal_spec_with_refinement("int { x | x > 0 }");
    let result = compile(&src, "test");
    assert!(result.lex_errors.is_empty(), "lex errors: {:?}", result.lex_errors);
    assert!(result.parse_errors.is_empty(), "parse errors: {:?}", result.parse_errors);
    assert!(result.validation_errors.is_empty(), "validation errors: {:?}", result.validation_errors);
    assert!(result.resolution_errors.is_empty(), "resolution errors: {:?}", result.resolution_errors);
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.refinement_errors.is_empty(), "refinement errors: {:?}", result.refinement_errors);
    assert!(result.program.is_some(), "expected program output");
}

#[test]
fn test_refined_function_param() {
    let src = r#"
role Node {
    var x: int = 0;

    fn Init(me: int) {}

    @trace
    async fn HandleRequest(n: int { x | x > 0 }) {
        var val = n;
    }
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) { }
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;
    let result = compile(src, "test");
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.program.is_some());
}

#[test]
fn test_refined_return_type() {
    let src = r#"
role Node {
    var x: int = 0;

    fn Init(me: int) {}

    fn getPositive(): int { x | x > 0 } {
        return 1;
    }
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) { }
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;
    let result = compile(src, "test");
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.program.is_some());
}

#[test]
fn test_refined_struct_field() {
    let src = r#"
type Container {
    value: int { x | x > 0 };
    name: string;
}

role Node {
    var data: Container = Container { value: 1, name: "a" };

    fn Init(me: int) {}

    @trace
    async fn HandleRequest(req: int) {}
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) { }
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;
    let result = compile(src, "test");
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.program.is_some());
}

#[test]
fn test_refined_non_bool_body_rejected() {
    let src = minimal_spec_with_refinement("int { x | x }");
    let result = compile(&src, "test");
    assert!(!result.type_errors.is_empty(), "expected type error for non-Bool body");
}

#[test]
fn test_refinement_compatibility_with_inner() {
    let src = r#"
role Node {
    var x: int { v | v > 0 } = 0;

    fn Init(me: int) {}

    @trace
    async fn HandleRequest(req: int) {
        var val: int { v | v > 0 } = req;
    }
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) { }
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;
    let result = compile(src, "test");
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.program.is_some());
}

#[test]
fn test_refinement_complex_body() {
    let src = minimal_spec_with_refinement("int { x | x > 0 && x < 100 }");
    let result = compile(&src, "test");
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.program.is_some());
}

#[test]
fn test_refinement_nested_type() {
    // Refined type inside a list
    let src = r#"
role Node {
    var x: list<int { y | y > 0 }> = [];

    fn Init(me: int) {}

    @trace
    async fn HandleRequest(req: int) {}
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) { }
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;
    let result = compile(src, "test");
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.program.is_some());
}

#[test]
fn test_refinement_with_builtin_call() {
    // Body uses len() builtin — should be allowed
    let src = r#"
role Node {
    var items: list<int> = [];

    fn Init(me: int) {}

    @trace
    async fn HandleRequest(req: int) {
        var n: int { x | x > 0 } = len(items);
    }
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) { }
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;
    let result = compile(src, "test");
    assert!(result.type_errors.is_empty(), "type errors: {:?}", result.type_errors);
    assert!(result.program.is_some());
}
