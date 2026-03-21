use super::*;
use crate::parser::{BinOp, Span};

fn dummy_span() -> Span {
    Span::default()
}

fn id(i: usize) -> NameId {
    NameId(i)
}

fn lowerer() -> Lowerer {
    Lowerer { next_name_id: 100 }
}

fn typed_expr(kind: TypedExprKind, ty: Type) -> TypedExpr {
    TypedExpr {
        kind,
        ty,
        span: dummy_span(),
    }
}

fn typed_int(v: i64) -> TypedExpr {
    typed_expr(TypedExprKind::IntLit(v), Type::Int)
}

fn typed_bool(v: bool) -> TypedExpr {
    typed_expr(TypedExprKind::BoolLit(v), Type::Bool)
}

#[test]
fn test_lower_int_literal() {
    let mut l = lowerer();
    let result = l.lower_expr(typed_int(42));
    assert_eq!(result.kind, LExprKind::IntLit(42));
    assert_eq!(result.ty, Type::Int);
}

#[test]
fn test_lower_short_circuit_and() {
    let mut l = lowerer();
    let expr = typed_expr(
        TypedExprKind::BinOp(
            BinOp::And,
            Box::new(typed_bool(true)),
            Box::new(typed_bool(false)),
        ),
        Type::Bool,
    );
    let result = l.lower_expr(expr);

    // a && b  →  if a { b } else { false }
    if let LExprKind::Conditional(cond) = &result.kind {
        assert_eq!(cond.if_branch.condition.kind, LExprKind::BoolLit(true));
        // then branch: the rhs value
        let then_tail = cond.if_branch.body.tail_expr.as_ref().unwrap();
        assert_eq!(then_tail.kind, LExprKind::BoolLit(false));
        // else branch: literal false
        let else_block = cond.else_branch.as_ref().unwrap();
        let else_tail = else_block.tail_expr.as_ref().unwrap();
        assert_eq!(else_tail.kind, LExprKind::BoolLit(false));
    } else {
        panic!("expected Conditional, got {:?}", result.kind);
    }
}

#[test]
fn test_lower_short_circuit_or() {
    let mut l = lowerer();
    let expr = typed_expr(
        TypedExprKind::BinOp(
            BinOp::Or,
            Box::new(typed_bool(false)),
            Box::new(typed_bool(true)),
        ),
        Type::Bool,
    );
    let result = l.lower_expr(expr);

    // a || b  →  if a { true } else { b }
    if let LExprKind::Conditional(cond) = &result.kind {
        assert_eq!(cond.if_branch.condition.kind, LExprKind::BoolLit(false));
        // then branch: literal true
        let then_tail = cond.if_branch.body.tail_expr.as_ref().unwrap();
        assert_eq!(then_tail.kind, LExprKind::BoolLit(true));
        // else branch: the rhs value
        let else_block = cond.else_branch.as_ref().unwrap();
        let else_tail = else_block.tail_expr.as_ref().unwrap();
        assert_eq!(else_tail.kind, LExprKind::BoolLit(true));
    } else {
        panic!("expected Conditional, got {:?}", result.kind);
    }
}

#[test]
fn test_lower_regular_binop() {
    let mut l = lowerer();
    let expr = typed_expr(
        TypedExprKind::BinOp(
            BinOp::Add,
            Box::new(typed_int(1)),
            Box::new(typed_int(2)),
        ),
        Type::Int,
    );
    let result = l.lower_expr(expr);

    if let LExprKind::BinOp(op, lhs, rhs) = &result.kind {
        assert_eq!(*op, BinOp::Add);
        assert_eq!(lhs.kind, LExprKind::IntLit(1));
        assert_eq!(rhs.kind, LExprKind::IntLit(2));
    } else {
        panic!("expected BinOp, got {:?}", result.kind);
    }
}

#[test]
fn test_lower_for_loop_no_init() {
    // for (; cond; ) { body }  →  loop { if !cond { break }; body }
    let mut l = lowerer();
    let for_loop = TypedForLoop {
        init: None,
        condition: Some(typed_bool(true)),
        increment: None,
        body: vec![TypedStatement {
            kind: TypedStatementKind::Expr(typed_int(1)),
            span: dummy_span(),
        }],
        span: dummy_span(),
    };
    let result = l.lower_for_loop(for_loop, dummy_span());

    // Without init, should be a Loop directly (not wrapped in a Block)
    if let LStatementKind::Loop(body) = &result.kind {
        // First statement: if !cond { break }
        assert!(body.len() >= 2);
        if let LStatementKind::Expr(cond_expr) = &body[0].kind {
            assert!(matches!(&cond_expr.kind, LExprKind::Conditional(_)));
        } else {
            panic!("expected condition check as first statement");
        }
        // Second statement: the body expression
        if let LStatementKind::Expr(body_expr) = &body[1].kind {
            assert_eq!(body_expr.kind, LExprKind::IntLit(1));
        } else {
            panic!("expected body expr as second statement");
        }
    } else {
        panic!("expected Loop, got {:?}", result.kind);
    }
}

#[test]
fn test_lower_for_loop_with_init() {
    // for (let x = 0; cond; x = x + 1) { body }
    //   →  { let x = 0; loop { if !cond { break }; body; x = x + 1 } }
    let mut l = lowerer();
    let for_loop = TypedForLoop {
        init: Some(TypedForLoopInit::VarInit(TypedVarInit {
            target: TypedVarTarget::Name(id(0), "x".to_string()),
            type_def: Type::Int,
            value: typed_int(0),
            span: dummy_span(),
        })),
        condition: Some(typed_bool(true)),
        increment: Some(TypedAssignment {
            target: typed_expr(TypedExprKind::Var(id(0), "x".to_string()), Type::Int),
            value: typed_int(1),
            span: dummy_span(),
        }),
        body: vec![],
        span: dummy_span(),
    };
    let result = l.lower_for_loop(for_loop, dummy_span());

    // With init, should be wrapped: Expr(Block { [init, Loop], None })
    if let LStatementKind::Expr(block_expr) = &result.kind {
        if let LExprKind::Block(block) = &block_expr.kind {
            assert_eq!(block.statements.len(), 2);
            assert!(matches!(&block.statements[0].kind, LStatementKind::VarInit(_)));
            if let LStatementKind::Loop(loop_body) = &block.statements[1].kind {
                // loop body: [cond-break, increment]
                // (no user body statements in this test)
                assert!(loop_body.len() >= 1);
                // Last statement should be the increment assignment
                let last = loop_body.last().unwrap();
                assert!(matches!(&last.kind, LStatementKind::Assignment(_)));
            } else {
                panic!("expected Loop as second statement in block");
            }
        } else {
            panic!("expected Block, got {:?}", block_expr.kind);
        }
    } else {
        panic!("expected Expr(Block), got {:?}", result.kind);
    }
}

#[test]
fn test_lower_match_variant() {
    // match x { A(v) => v, B => 0 }
    //   →  { let __match_scrutinee = x; if isVariant(s, "A") { ... } else { if isVariant(s, "B") { ... } } }
    let mut l = lowerer();
    let enum_id = id(10);
    let scrutinee = typed_expr(
        TypedExprKind::Var(id(1), "x".to_string()),
        Type::Enum(enum_id, "MyEnum".to_string()),
    );
    let arms = vec![
        TypedMatchArm {
            pattern: TypedPattern {
                kind: TypedPatternKind::Variant(
                    enum_id,
                    "A".to_string(),
                    Some(Box::new(TypedPattern {
                        kind: TypedPatternKind::Var(id(2), "v".to_string()),
                        ty: Type::Int,
                        span: dummy_span(),
                    })),
                ),
                ty: Type::Enum(enum_id, "MyEnum".to_string()),
                span: dummy_span(),
            },
            body: TypedBlock {
                statements: vec![],
                tail_expr: Some(Box::new(typed_expr(
                    TypedExprKind::Var(id(2), "v".to_string()),
                    Type::Int,
                ))),
                ty: Type::Int,
                span: dummy_span(),
            },
            span: dummy_span(),
        },
        TypedMatchArm {
            pattern: TypedPattern {
                kind: TypedPatternKind::Variant(enum_id, "B".to_string(), None),
                ty: Type::Enum(enum_id, "MyEnum".to_string()),
                span: dummy_span(),
            },
            body: TypedBlock {
                statements: vec![],
                tail_expr: Some(Box::new(typed_int(0))),
                ty: Type::Int,
                span: dummy_span(),
            },
            span: dummy_span(),
        },
    ];

    let result = l.lower_match(scrutinee, arms, Type::Int, dummy_span());

    // Should be Block { let __match_scrutinee = x; <conditional chain> }
    if let LExprKind::Block(block) = &result.kind {
        // First statement: scrutinee binding
        assert_eq!(block.statements.len(), 1);
        if let LStatementKind::VarInit(vi) = &block.statements[0].kind {
            if let LVarTarget::Name(_, name) = &vi.target {
                assert_eq!(name, "__match_scrutinee");
            } else {
                panic!("expected Name target for scrutinee");
            }
        } else {
            panic!("expected VarInit for scrutinee");
        }

        // Tail expr: conditional chain starting with IsVariant(_, "A")
        let tail = block.tail_expr.as_ref().unwrap();
        if let LExprKind::Conditional(cond) = &tail.kind {
            if let LExprKind::IsVariant(_, variant_name) = &cond.if_branch.condition.kind {
                assert_eq!(variant_name, "A");
            } else {
                panic!("expected IsVariant condition, got {:?}", cond.if_branch.condition.kind);
            }

            // The else branch should contain the second arm (IsVariant "B")
            let else_block = cond.else_branch.as_ref().unwrap();
            let else_tail = else_block.tail_expr.as_ref().unwrap();
            if let LExprKind::Conditional(inner_cond) = &else_tail.kind {
                if let LExprKind::IsVariant(_, variant_name) = &inner_cond.if_branch.condition.kind {
                    assert_eq!(variant_name, "B");
                } else {
                    panic!("expected IsVariant 'B', got {:?}", inner_cond.if_branch.condition.kind);
                }
            } else {
                panic!("expected nested Conditional for second arm");
            }
        } else {
            panic!("expected Conditional tail, got {:?}", tail.kind);
        }
    } else {
        panic!("expected Block, got {:?}", result.kind);
    }
}
