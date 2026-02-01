use super::*;
use pretty_assertions::assert_eq;

fn dummy_span() -> Span {
    Span::default()
}

fn expr(kind: ExprKind) -> Expr {
    Expr {
        kind,
        span: dummy_span(),
    }
}

fn int_lit(val: i64) -> Expr {
    expr(ExprKind::IntLit(val))
}

fn str_lit(val: &str) -> Expr {
    expr(ExprKind::StringLit(val.to_string()))
}

fn bool_lit(val: bool) -> Expr {
    expr(ExprKind::BoolLit(val))
}

fn var_expr(name: &str) -> Expr {
    expr(ExprKind::Var(name.to_string()))
}

fn bin_op(op: BinOp, l: Expr, r: Expr) -> Expr {
    expr(ExprKind::BinOp(op, Box::new(l), Box::new(r)))
}

fn stmt(kind: StatementKind) -> Statement {
    Statement {
        kind,
        span: dummy_span(),
    }
}

fn var_init(name: &str, value: Expr) -> Statement {
    stmt(StatementKind::VarInit(VarInit {
        name: name.to_string(),
        type_def: None,
        value,
        span: dummy_span(),
    }))
}

// Helpers for variant testing
impl Resolver {
    // Expose declaring a type for testing purposes
    fn declare_test_type(&mut self, name: &str) -> NameId {
        self.declare_type(name, dummy_span()).unwrap()
    }
}

fn variant_lit(enum_name: &str, variant_name: &str, payload: Option<Expr>) -> Expr {
    expr(ExprKind::VariantLit(
        enum_name.to_string(),
        variant_name.to_string(),
        payload.map(Box::new),
    ))
}

fn named_dot(first: &str, second: &str, payload: Option<Expr>) -> Expr {
    expr(ExprKind::NamedDotAccess(
        first.to_string(),
        second.to_string(),
        payload.map(Box::new),
    ))
}

#[test]
fn test_resolve_literals() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    let res_int = resolver.resolve_expr(int_lit(42))?;
    match res_int.kind {
        ResolvedExprKind::IntLit(val) => assert_eq!(val, 42),
        _ => panic!("expected IntLit"),
    }

    let res_str = resolver.resolve_expr(str_lit("hello"))?;
    match res_str.kind {
        ResolvedExprKind::StringLit(val) => assert_eq!(val, "hello"),
        _ => panic!("expected StringLit"),
    }

    let res_bool = resolver.resolve_expr(bool_lit(true))?;
    match res_bool.kind {
        ResolvedExprKind::BoolLit(val) => assert_eq!(val, true),
        _ => panic!("expected BoolLit"),
    }

    Ok(())
}

#[test]
fn test_resolve_binop() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    let op = bin_op(BinOp::Add, int_lit(1), int_lit(2));
    let res = resolver.resolve_expr(op)?;

    match res.kind {
        ResolvedExprKind::BinOp(op, l, r) => {
            assert_eq!(op, BinOp::Add);
            match l.kind {
                ResolvedExprKind::IntLit(val) => assert_eq!(val, 1),
                _ => panic!("expected IntLit for left operand"),
            }
            match r.kind {
                ResolvedExprKind::IntLit(val) => assert_eq!(val, 2),
                _ => panic!("expected IntLit for right operand"),
            }
        }
        _ => panic!("expected BinOp"),
    }

    Ok(())
}

#[test]
fn test_resolve_variable_scope() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // Declare a variable
    let id = resolver.declare_var("x", dummy_span())?;

    // Resolve usage of the variable
    let res = resolver.resolve_expr(var_expr("x"))?;

    match res.kind {
        ResolvedExprKind::Var(resolved_id, name) => {
            assert_eq!(resolved_id, id);
            assert_eq!(name, "x");
        }
        _ => panic!("expected Var"),
    }

    Ok(())
}

#[test]
fn test_resolve_undefined_variable() {
    let mut resolver = Resolver::new();
    let err = resolver.resolve_expr(var_expr("undefined")).unwrap_err();

    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "undefined"));
}

#[test]
fn test_resolve_shadowing() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // Outer scope
    resolver.enter_scope();
    let id1 = resolver.declare_var("x", dummy_span())?;

    // Inner scope
    resolver.enter_scope();
    let id2 = resolver.declare_var("x", dummy_span())?;

    assert_ne!(id1, id2);

    let res = resolver.resolve_expr(var_expr("x"))?;
    match res.kind {
        ResolvedExprKind::Var(resolved_id, _) => {
            assert_eq!(resolved_id, id2, "Should resolve to inner variable");
        }
        _ => panic!("expected Var"),
    }

    resolver.exit_scope();

    // Back to outer scope
    let res = resolver.resolve_expr(var_expr("x"))?;
    match res.kind {
        ResolvedExprKind::Var(resolved_id, _) => {
            assert_eq!(resolved_id, id1, "Should resolve to outer variable");
        }
        _ => panic!("expected Var"),
    }

    resolver.exit_scope();
    Ok(())
}

#[test]
fn test_resolve_duplicate_variable_same_scope() {
    let mut resolver = Resolver::new();

    resolver.enter_scope();
    resolver.declare_var("x", dummy_span()).unwrap();
    let err = resolver.declare_var("x", dummy_span()).unwrap_err();

    assert!(matches!(err, ResolutionError::DuplicateName(name, _) if name == "x"));
}

#[test]
fn test_resolve_block_scope() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    let block = vec![
        var_init("x", int_lit(10)),
        stmt(StatementKind::Expr(var_expr("x"))),
    ];

    let resolved_block = resolver.resolve_block(block)?;
    assert_eq!(resolved_block.len(), 2);

    match &resolved_block[0].kind {
        ResolvedStatementKind::VarInit(vi) => {
            assert_eq!(vi.original_name, "x");
        }
        _ => panic!("expected VarInit"),
    }

    // "x" should not be available outside the block if resolve_block handles scoping correctly
    let err = resolver.resolve_expr(var_expr("x")).unwrap_err();
    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "x"));

    Ok(())
}

#[test]
fn test_resolve_function_call() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // We need a role context or client context to resolve user functions.
    let func_name = "my_func";
    let func_id = resolver.new_name_id(func_name);
    resolver
        .client_func_scope
        .insert(func_name.to_string(), func_id);

    let call = ExprKind::FuncCall(FuncCall {
        name: func_name.to_string(),
        args: vec![int_lit(1)],
        span: dummy_span(),
    });

    let res = resolver.resolve_expr(expr(call))?;

    match res.kind {
        ResolvedExprKind::FuncCall(ResolvedFuncCall::User(user_call)) => {
            assert_eq!(user_call.name, func_id);
            assert_eq!(user_call.original_name, func_name);
            assert_eq!(user_call.args.len(), 1);
        }
        _ => panic!("expected User FuncCall"),
    }

    Ok(())
}

#[test]
fn test_resolve_builtin_function_call() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    let call = ExprKind::FuncCall(FuncCall {
        name: "println".to_string(),
        args: vec![str_lit("hello")],
        span: dummy_span(),
    });

    let res = resolver.resolve_expr(expr(call))?;

    match res.kind {
        ResolvedExprKind::FuncCall(ResolvedFuncCall::Builtin(builtin, args, _)) => {
            assert_eq!(builtin, BuiltinFn::Println);
            assert_eq!(args.len(), 1);
        }
        _ => panic!("expected Builtin FuncCall"),
    }

    Ok(())
}

#[test]
fn test_prepopulated_types() {
    let resolver = Resolver::new();
    let pre = resolver.get_pre_populated_types();

    assert!(resolver.lookup_type("int", dummy_span()).is_ok());
    assert_eq!(resolver.lookup_type("int", dummy_span()).unwrap(), pre.int);

    assert!(resolver.lookup_type("string", dummy_span()).is_ok());
    assert_eq!(
        resolver.lookup_type("string", dummy_span()).unwrap(),
        pre.string
    );

    assert!(resolver.lookup_type("bool", dummy_span()).is_ok());
    assert_eq!(
        resolver.lookup_type("bool", dummy_span()).unwrap(),
        pre.bool
    );

    assert!(resolver.lookup_type("unit", dummy_span()).is_ok());
    assert_eq!(
        resolver.lookup_type("unit", dummy_span()).unwrap(),
        pre.unit
    );
}

#[test]
fn test_undefined_type() {
    let resolver = Resolver::new();
    let err = resolver
        .lookup_type("UndefinedType", dummy_span())
        .unwrap_err();

    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "UndefinedType"));
}

#[test]
fn test_undefined_function() {
    let mut resolver = Resolver::new();

    let call = ExprKind::FuncCall(FuncCall {
        name: "undefined_func".to_string(),
        args: vec![],
        span: dummy_span(),
    });

    let err = resolver.resolve_expr(expr(call)).unwrap_err();
    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "undefined_func"));
}

#[test]
fn test_variable_used_before_declaration_in_same_expression() {
    let mut resolver = Resolver::new();
    resolver.enter_scope();

    // Try to use variable in its own initializer
    let var_init_stmt = stmt(StatementKind::VarInit(VarInit {
        name: "x".to_string(),
        type_def: None,
        value: bin_op(BinOp::Add, var_expr("x"), int_lit(1)),
        span: dummy_span(),
    }));

    let err = resolver.resolve_statement(var_init_stmt).unwrap_err();
    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "x"));
}

#[test]
fn test_nested_scope_variable_isolation() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // Outer scope
    resolver.enter_scope();
    let outer_id = resolver.declare_var("outer", dummy_span())?;

    // Inner scope
    resolver.enter_scope();
    let inner_id = resolver.declare_var("inner", dummy_span())?;

    // Both should be visible in inner scope
    let res_inner = resolver.resolve_expr(var_expr("inner"))?;
    match res_inner.kind {
        ResolvedExprKind::Var(id, _) => assert_eq!(id, inner_id),
        _ => panic!("expected Var"),
    }

    let res_outer = resolver.resolve_expr(var_expr("outer"))?;
    match res_outer.kind {
        ResolvedExprKind::Var(id, _) => assert_eq!(id, outer_id),
        _ => panic!("expected Var"),
    }

    resolver.exit_scope();

    // Back to outer scope - inner should not be visible
    let err = resolver.resolve_expr(var_expr("inner")).unwrap_err();
    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "inner"));

    // But outer should still be visible
    let res_outer = resolver.resolve_expr(var_expr("outer"))?;
    match res_outer.kind {
        ResolvedExprKind::Var(id, _) => assert_eq!(id, outer_id),
        _ => panic!("expected Var"),
    }

    resolver.exit_scope();
    Ok(())
}

#[test]
fn test_list_literal_resolution() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    let list = expr(ExprKind::ListLit(vec![int_lit(1), int_lit(2), int_lit(3)]));

    let res = resolver.resolve_expr(list)?;
    match res.kind {
        ResolvedExprKind::ListLit(items) => {
            assert_eq!(items.len(), 3);
            for (i, item) in items.iter().enumerate() {
                match &item.kind {
                    ResolvedExprKind::IntLit(val) => assert_eq!(*val, (i + 1) as i64),
                    _ => panic!("expected IntLit"),
                }
            }
        }
        _ => panic!("expected ListLit"),
    }

    Ok(())
}

#[test]
fn test_map_literal_resolution() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    let map = expr(ExprKind::MapLit(vec![
        (str_lit("a"), int_lit(1)),
        (str_lit("b"), int_lit(2)),
    ]));

    let res = resolver.resolve_expr(map)?;
    match res.kind {
        ResolvedExprKind::MapLit(pairs) => {
            assert_eq!(pairs.len(), 2);
        }
        _ => panic!("expected MapLit"),
    }

    Ok(())
}

#[test]
fn test_resolve_variant_literal() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // Declare Enum type "E"
    let enum_id = resolver.declare_test_type("E");

    // Case 1: Unambiguous VariantLit (already parsed as VariantLit)
    // E.V1
    let expr1 = variant_lit("E", "V1", None);
    let res1 = resolver.resolve_expr(expr1)?;
    match res1.kind {
        ResolvedExprKind::VariantLit(type_id, variant_name, payload) => {
            assert_eq!(type_id, enum_id);
            assert_eq!(variant_name, "V1");
            assert!(payload.is_none());
        }
        _ => panic!("expected VariantLit"),
    }

    Ok(())
}

#[test]
fn test_resolve_ambiguous_variant_literal() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // Declare Enum type "E"
    let enum_id = resolver.declare_test_type("E");

    // Case 2: Ambiguous NamedDotAccess resolving to VariantLit
    // E.V1, where E is a Type
    let expr1 = named_dot("E", "V1", None);
    let res1 = resolver.resolve_expr(expr1)?;
    match res1.kind {
        ResolvedExprKind::VariantLit(type_id, variant_name, payload) => {
            assert_eq!(type_id, enum_id);
            assert_eq!(variant_name, "V1");
            assert!(payload.is_none());
        }
        _ => panic!("expected VariantLit"),
    }

    Ok(())
}

#[test]
fn test_resolve_ambiguous_field_access() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // Declare variable "x"
    let var_id = resolver.declare_var("x", dummy_span())?;

    // Case 3: Ambiguous NamedDotAccess resolving to FieldAccess
    // x.f, where x is a variable
    let expr1 = named_dot("x", "f", None);
    let res1 = resolver.resolve_expr(expr1)?;
    match res1.kind {
        ResolvedExprKind::FieldAccess(target, field_name) => {
            match target.kind {
                ResolvedExprKind::Var(id, _) => assert_eq!(id, var_id),
                _ => panic!("expected Var target"),
            }
            assert_eq!(field_name, "f");
        }
        _ => panic!("expected FieldAccess"),
    }

    Ok(())
}

#[test]
fn test_resolve_ambiguous_error_unknown_base() {
    let mut resolver = Resolver::new();

    // Unknown.f
    let expr1 = named_dot("Unknown", "f", None);
    let err = resolver.resolve_expr(expr1).unwrap_err();

    // Should fail as NameNotFound (neither type nor variable)
    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "Unknown"));
}

#[test]
fn test_resolve_ambiguous_error_field_access_with_payload() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();

    // Declare variable "x"
    resolver.declare_var("x", dummy_span())?;

    let expr1 = named_dot("x", "f", Some(int_lit(42)));
    let err = resolver.resolve_expr(expr1).unwrap_err();
    assert!(matches!(err, ResolutionError::NameNotFound(name, _) if name == "x"));

    Ok(())
}

#[test]
fn test_resolve_match_expression() -> Result<(), ResolutionError> {
    let mut resolver = Resolver::new();
    let enum_id = resolver.declare_test_type("E");
    let var_id = resolver.declare_var("x", dummy_span())?;

    let match_expr = ExprKind::Match(
        Box::new(var_expr("x")),
        vec![MatchArm {
            pattern: Pattern {
                kind: PatternKind::Variant("E".to_string(), "V1".to_string(), None),
                span: dummy_span(),
            },
            body: vec![stmt(StatementKind::Break)],
            span: dummy_span(),
        }],
    );

    let res = resolver.resolve_expr(expr(match_expr))?;
    match res.kind {
        ResolvedExprKind::Match(scrutinee, arms) => {
            match scrutinee.kind {
                ResolvedExprKind::Var(id, _) => assert_eq!(id, var_id),
                _ => panic!("expected Var scrutinee"),
            }
            assert_eq!(arms.len(), 1);
            match &arms[0].pattern.kind {
                ResolvedPatternKind::Variant(type_id, variant_name, payload) => {
                    assert_eq!(*type_id, enum_id);
                    assert_eq!(variant_name, "V1");
                    assert!(payload.is_none());
                }
                _ => panic!("expected Variant pattern"),
            }
        }
        _ => panic!("expected Match"),
    }

    Ok(())
}
