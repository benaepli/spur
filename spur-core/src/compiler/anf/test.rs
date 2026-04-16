use super::ast::*;
use super::lower::lower_program;
use crate::analysis::resolver::NameId;
use crate::analysis::types::Type;
use crate::compiler::lowered::*;
use crate::parser::{BinOp, Span};

fn dummy_span() -> Span {
    Span::default()
}

fn id(i: usize) -> NameId {
    NameId(i)
}

fn lexpr(kind: LExprKind, ty: Type) -> LExpr {
    LExpr {
        kind,
        ty,
        span: dummy_span(),
    }
}

fn lint(v: i64) -> LExpr {
    lexpr(LExprKind::IntLit(v), Type::Int)
}

fn lvar(i: usize, name: &str) -> LExpr {
    lexpr(LExprKind::Var(id(i), name.to_string()), Type::Int)
}

// -----------------------------------------------------------------------
// 1. Atomic passthrough — no extra bindings generated
// -----------------------------------------------------------------------

#[test]
fn test_atomic_int() {
    let program = make_program(vec![], vec![lint(42)]);
    let anf = lower_program(program);
    let body = first_func_body(&anf);

    // The tail expression should be the atomic int directly
    assert_eq!(body.tail_expr, Some(AAtomic::IntLit(42)));
    // No extra let bindings
    assert!(body.statements.is_empty());
}

#[test]
fn test_atomic_var() {
    let program = make_program(
        vec![lparam(0, "x")],
        vec![lvar(0, "x")],
    );
    let anf = lower_program(program);
    let body = first_func_body(&anf);

    assert_eq!(body.tail_expr, Some(AAtomic::Var(id(0), "x".to_string())));
    assert!(body.statements.is_empty());
}

// -----------------------------------------------------------------------
// 2. Nested BinOp flattening
// -----------------------------------------------------------------------

#[test]
fn test_nested_binop_flattening() {
    // (a + b) * c  →  let t1 = a + b; let t2 = t1 * c; tail = t2
    let expr = lexpr(
        LExprKind::BinOp(
            BinOp::Multiply,
            Box::new(lexpr(
                LExprKind::BinOp(
                    BinOp::Add,
                    Box::new(lvar(0, "a")),
                    Box::new(lvar(1, "b")),
                ),
                Type::Int,
            )),
            Box::new(lvar(2, "c")),
        ),
        Type::Int,
    );

    let program = make_program(
        vec![lparam(0, "a"), lparam(1, "b"), lparam(2, "c")],
        vec![expr],
    );
    let anf = lower_program(program);
    let body = first_func_body(&anf);

    // Should have 2 let bindings: one for (a+b), one for (t*c)
    assert_eq!(body.statements.len(), 2);

    // First binding: let __anf_N = a + b
    let s0 = expect_let_atom(&body.statements[0]);
    if let AExprKind::BinOp(BinOp::Add, ref l, ref r) = s0.value.kind {
        assert_eq!(*l, AAtomic::Var(id(0), "a".to_string()));
        assert_eq!(*r, AAtomic::Var(id(1), "b".to_string()));
    } else {
        panic!("expected BinOp(Add), got {:?}", s0.value.kind);
    }
    let t1 = AAtomic::Var(s0.name, s0.original_name.clone());

    // Second binding: let __anf_M = t1 * c
    let s1 = expect_let_atom(&body.statements[1]);
    if let AExprKind::BinOp(BinOp::Multiply, ref l, ref r) = s1.value.kind {
        assert_eq!(*l, t1);
        assert_eq!(*r, AAtomic::Var(id(2), "c".to_string()));
    } else {
        panic!("expected BinOp(Mul), got {:?}", s1.value.kind);
    }

    // Tail is the second temp
    let t2 = AAtomic::Var(s1.name, s1.original_name.clone());
    assert_eq!(body.tail_expr, Some(t2));
}

// -----------------------------------------------------------------------
// 3. Nested FuncCall flattening
// -----------------------------------------------------------------------

#[test]
fn test_nested_func_call_flattening() {
    // f(g(x))  →  let t1 = g(x); let t2 = f(t1); tail = t2
    let inner_call = LExpr {
        kind: LExprKind::FuncCall(LFuncCall::User(LUserFuncCall {
            name: id(10),
            original_name: "g".to_string(),
            args: vec![lvar(0, "x")],
            return_type: Type::Int,
            span: dummy_span(),
        })),
        ty: Type::Int,
        span: dummy_span(),
    };
    let outer_call = LExpr {
        kind: LExprKind::FuncCall(LFuncCall::User(LUserFuncCall {
            name: id(11),
            original_name: "f".to_string(),
            args: vec![inner_call],
            return_type: Type::Int,
            span: dummy_span(),
        })),
        ty: Type::Int,
        span: dummy_span(),
    };

    let program = make_program(vec![lparam(0, "x")], vec![outer_call]);
    let anf = lower_program(program);
    let body = first_func_body(&anf);

    // Should have 2 let bindings
    assert_eq!(body.statements.len(), 2);

    // First: let t1 = g(x)
    let s0 = expect_let_atom(&body.statements[0]);
    if let AExprKind::FuncCall(AFuncCall::User(ref u)) = s0.value.kind {
        assert_eq!(u.original_name, "g");
        assert_eq!(u.args, vec![AAtomic::Var(id(0), "x".to_string())]);
    } else {
        panic!("expected User FuncCall for g, got {:?}", s0.value.kind);
    }
    let t1 = AAtomic::Var(s0.name, s0.original_name.clone());

    // Second: let t2 = f(t1)
    let s1 = expect_let_atom(&body.statements[1]);
    if let AExprKind::FuncCall(AFuncCall::User(ref u)) = s1.value.kind {
        assert_eq!(u.original_name, "f");
        assert_eq!(u.args, vec![t1]);
    } else {
        panic!("expected User FuncCall for f, got {:?}", s1.value.kind);
    }
}

// -----------------------------------------------------------------------
// 4. L-value preservation
// -----------------------------------------------------------------------

#[test]
fn test_lhs_preservation() {
    // db[key].field = val;
    // The LHS should remain as ALhsExpr::FieldAccess(Index(Var(db), key), "field")
    // The RHS should be flattened to atomic

    let lhs = lexpr(
        LExprKind::FieldAccess(
            Box::new(lexpr(
                LExprKind::Index(
                    Box::new(lvar(0, "db")),
                    Box::new(lvar(1, "key")),
                ),
                Type::Int,
            )),
            "field".to_string(),
        ),
        Type::Int,
    );

    let rhs = lvar(2, "val");

    let stmt = LStatement {
        kind: LStatementKind::Assignment(LAssignment {
            target: lhs,
            value: rhs,
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let program = make_program_with_stmts(
        vec![lparam(0, "db"), lparam(1, "key"), lparam(2, "val")],
        vec![stmt],
    );
    let anf = lower_program(program);
    let body = first_func_body(&anf);

    // Should have exactly 1 statement: the Assign
    assert_eq!(body.statements.len(), 1);
    if let AStatementKind::Assign(ref assign) = body.statements[0].kind {
        // LHS should be FieldAccess(Index(Var(db), key), "field")
        if let ALhsExpr::FieldAccess(ref inner, ref fname) = assign.target {
            assert_eq!(fname, "field");
            if let ALhsExpr::Index(ref base, ref idx) = **inner {
                assert!(matches!(&**base, ALhsExpr::Var(_, name) if name == "db"));
                assert_eq!(*idx, AAtomic::Var(id(1), "key".to_string()));
            } else {
                panic!("expected Index inside FieldAccess, got {:?}", inner);
            }
        } else {
            panic!("expected FieldAccess LHS, got {:?}", assign.target);
        }
        // RHS should be atomic
        assert_eq!(assign.value, AAtomic::Var(id(2), "val".to_string()));
    } else {
        panic!("expected Assign statement, got {:?}", body.statements[0].kind);
    }
}

// -----------------------------------------------------------------------
// 5. Conditional tail atomicity
// -----------------------------------------------------------------------

#[test]
fn test_conditional_tail_atomicity() {
    // if cond { a + b } else { c }
    // The branches should flatten a+b into a let binding with atomic tail

    let cond_expr = lexpr(
        LExprKind::Conditional(Box::new(LCondExpr {
            if_branch: LIfBranch {
                condition: lexpr(LExprKind::BoolLit(true), Type::Bool),
                body: LBlock {
                    statements: vec![],
                    tail_expr: Some(Box::new(lexpr(
                        LExprKind::BinOp(
                            BinOp::Add,
                            Box::new(lvar(0, "a")),
                            Box::new(lvar(1, "b")),
                        ),
                        Type::Int,
                    ))),
                    ty: Type::Int,
                    span: dummy_span(),
                },
                span: dummy_span(),
            },
            elseif_branches: vec![],
            else_branch: Some(LBlock {
                statements: vec![],
                tail_expr: Some(Box::new(lvar(2, "c"))),
                ty: Type::Int,
                span: dummy_span(),
            }),
            span: dummy_span(),
        })),
        Type::Int,
    );

    let program = make_program(
        vec![lparam(0, "a"), lparam(1, "b"), lparam(2, "c")],
        vec![cond_expr],
    );
    let anf = lower_program(program);
    let body = first_func_body(&anf);

    // The tail should be atomic (the conditional result bound to a temp)
    assert!(body.tail_expr.is_some());

    // Find the conditional — it's bound to a LetAtom, and the cond
    // expression's if-branch body should have a let binding and atomic tail
    let s0 = expect_let_atom(&body.statements[0]);
    if let AExprKind::Conditional(ref cond) = s0.value.kind {
        let if_body = &cond.if_branch.body;
        // a + b should be flattened inside the branch
        assert_eq!(if_body.statements.len(), 1);
        let inner_let = expect_let_atom(&if_body.statements[0]);
        assert!(matches!(inner_let.value.kind, AExprKind::BinOp(BinOp::Add, _, _)));
        // Tail must be atomic
        assert!(if_body.tail_expr.is_some());

        // Else branch: tail is just c (already atomic)
        let else_body = cond.else_branch.as_ref().unwrap();
        assert!(else_body.statements.is_empty());
        assert_eq!(
            else_body.tail_expr,
            Some(AAtomic::Var(id(2), "c".to_string()))
        );
    } else {
        panic!("expected Conditional AExpr, got {:?}", s0.value.kind);
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn lparam(i: usize, name: &str) -> LFuncParam {
    LFuncParam {
        name: id(i),
        original_name: name.to_string(),
        ty: Type::Int,
        span: dummy_span(),
    }
}

/// Build a minimal LProgram with a single free function whose body is
/// a block with the given tail expressions (the last one becomes the tail).
fn make_program(params: Vec<LFuncParam>, mut tail_exprs: Vec<LExpr>) -> LProgram {
    let tail = if tail_exprs.len() == 1 {
        Some(Box::new(tail_exprs.remove(0)))
    } else {
        tail_exprs.pop().map(Box::new)
    };

    let stmts: Vec<LStatement> = tail_exprs
        .into_iter()
        .map(|e| LStatement {
            kind: LStatementKind::Expr(e),
            span: dummy_span(),
        })
        .collect();

    LProgram {
        top_level_defs: vec![LTopLevelDef::FreeFunc(LFuncDef {
            name: id(99),
            original_name: "test_fn".to_string(),
            is_sync: true,
            is_traced: false,
            params,
            return_type: Type::Int,
            body: LBlock {
                statements: stmts,
                tail_expr: tail,
                ty: Type::Int,
                span: dummy_span(),
            },
            span: dummy_span(),
        })],
        next_name_id: 100,
        id_to_name: std::collections::HashMap::new(),
        struct_defs: std::collections::HashMap::new(),
        enum_defs: std::collections::HashMap::new(),
    }
}

/// Like make_program but takes explicit statements (no tail expr).
fn make_program_with_stmts(
    params: Vec<LFuncParam>,
    stmts: Vec<LStatement>,
) -> LProgram {
    LProgram {
        top_level_defs: vec![LTopLevelDef::FreeFunc(LFuncDef {
            name: id(99),
            original_name: "test_fn".to_string(),
            is_sync: true,
            is_traced: false,
            params,
            return_type: Type::Nil,
            body: LBlock {
                statements: stmts,
                tail_expr: None,
                ty: Type::Nil,
                span: dummy_span(),
            },
            span: dummy_span(),
        })],
        next_name_id: 100,
        id_to_name: std::collections::HashMap::new(),
        struct_defs: std::collections::HashMap::new(),
        enum_defs: std::collections::HashMap::new(),
    }
}

fn first_func_body(program: &AProgram) -> &ABlock {
    match &program.top_level_defs[0] {
        ATopLevelDef::FreeFunc(f) => &f.body,
        ATopLevelDef::Role(r) => &r.func_defs[0].body,
    }
}

fn expect_let_atom(stmt: &AStatement) -> &ALetAtom {
    if let AStatementKind::LetAtom(ref la) = stmt.kind {
        la
    } else {
        panic!("expected LetAtom, got {:?}", stmt.kind);
    }
}
