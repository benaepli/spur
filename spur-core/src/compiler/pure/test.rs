use std::collections::{HashMap, HashSet};

use super::lower::lower_program;
use crate::analysis::resolver::NameId;
use crate::analysis::types::Type;
use crate::liquid::pure::ast::*;
use crate::liquid::threaded::ast::*;
use crate::parser::{BinOp, Span};

fn dummy_span() -> Span {
    Span::default()
}

fn id(i: usize) -> NameId {
    NameId(i)
}

fn make_free_func_program(func: TFuncDef) -> TProgram {
    TProgram {
        top_level_defs: vec![TTopLevelDef::FreeFunc(func)],
        next_name_id: 100,
        id_to_name: HashMap::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
    }
}

fn simple_func(
    name_id: usize,
    name: &str,
    params: Vec<TFuncParam>,
    return_type: Type,
    body: TBlock,
) -> TFuncDef {
    TFuncDef {
        name: id(name_id),
        original_name: name.to_string(),
        is_sync: true,
        is_traced: false,
        params,
        return_type,
        body,
        span: dummy_span(),
    }
}

fn tparam(i: usize, name: &str, ty: Type) -> TFuncParam {
    TFuncParam {
        name: id(i),
        original_name: name.to_string(),
        ty,
        span: dummy_span(),
    }
}

fn tlet(nid: usize, name: &str, ty: Type, value: TExpr) -> TStatement {
    TStatement {
        kind: TStatementKind::LetAtom(TLetAtom {
            name: id(nid),
            original_name: name.to_string(),
            ty,
            value,
            span: dummy_span(),
        }),
        span: dummy_span(),
    }
}

fn tassign_var(nid: usize, name: &str, value: TAtomic) -> TStatement {
    TStatement {
        kind: TStatementKind::Assign(TAssign {
            target_id: id(nid),
            target_name: name.to_string(),
            ty: Type::Int,
            value,
            span: dummy_span(),
        }),
        span: dummy_span(),
    }
}

fn tatom_expr(a: TAtomic, ty: Type) -> TExpr {
    TExpr {
        kind: TExprKind::Atomic(a),
        ty,
        span: dummy_span(),
    }
}

fn texpr_int(a: TAtomic) -> TExpr {
    tatom_expr(a, Type::Int)
}

fn tblock(stmts: Vec<TStatement>, tail: Option<TAtomic>, ty: Type) -> TBlock {
    TBlock {
        statements: stmts,
        tail_expr: tail,
        ty,
        span: dummy_span(),
    }
}

fn expect_free_func(program: &PProgram, idx: usize) -> &PFuncDef {
    match &program.top_level_defs[idx] {
        PTopLevelDef::FreeFunc(f) => f,
        other => panic!("expected FreeFunc at index {}, got {:?}", idx, other),
    }
}

fn count_loop_converted(program: &PProgram) -> usize {
    program
        .top_level_defs
        .iter()
        .filter(|d| match d {
            PTopLevelDef::FreeFunc(f) => f.kind == PFuncKind::LoopConverted,
            _ => false,
        })
        .count()
}

fn tloop(body: Vec<TStatement>) -> TStatement {
    TStatement {
        kind: TStatementKind::Loop(body),
        span: dummy_span(),
    }
}

fn tbreak() -> TStatement {
    TStatement {
        kind: TStatementKind::Break,
        span: dummy_span(),
    }
}

fn tcontinue() -> TStatement {
    TStatement {
        kind: TStatementKind::Continue,
        span: dummy_span(),
    }
}

fn treturn(a: TAtomic) -> TStatement {
    TStatement {
        kind: TStatementKind::Return(a),
        span: dummy_span(),
    }
}

fn assert_no_loop(block: &PBlock) {
    for stmt in &block.statements {
        match &stmt.kind {
            PStatementKind::LetAtom(la) => assert_no_loop_expr(&la.value),
            PStatementKind::Expr(e) => assert_no_loop_expr(e),
            _ => {}
        }
    }
}

fn assert_no_loop_expr(e: &PExpr) {
    match &e.kind {
        PExprKind::Conditional(c) => {
            assert_no_loop(&c.if_branch.body);
            for b in &c.elseif_branches {
                assert_no_loop(&b.body);
            }
            if let Some(b) = &c.else_branch {
                assert_no_loop(b);
            }
        }
        PExprKind::Block(b) => assert_no_loop(b),
        _ => {}
    }
}

fn expect_let_atom(stmt: &PStatement) -> &PLetAtom {
    match &stmt.kind {
        PStatementKind::LetAtom(la) => la,
        other => panic!("expected LetAtom, got {:?}", other),
    }
}

fn assert_no_assign(block: &PBlock) {
    // PStatementKind has no Assign variant — the invariant is structural.
    // This helper walks recursively for sanity and also asserts no
    // leftover Error statements surface in the output.
    for stmt in &block.statements {
        match &stmt.kind {
            PStatementKind::LetAtom(la) => assert_no_assign_expr(&la.value),
            PStatementKind::Expr(e) => assert_no_assign_expr(e),
            PStatementKind::Return(_) => {}
            PStatementKind::Error => panic!("unexpected Error stmt"),
        }
    }
}

fn assert_no_assign_expr(e: &PExpr) {
    match &e.kind {
        PExprKind::Conditional(cond) => {
            assert_no_assign(&cond.if_branch.body);
            for b in &cond.elseif_branches {
                assert_no_assign(&b.body);
            }
            if let Some(b) = &cond.else_branch {
                assert_no_assign(b);
            }
        }
        PExprKind::Block(b) => assert_no_assign(b),
        _ => {}
    }
}

// ===== M0 / baseline =====

#[test]
fn test_trivial_passthrough() {
    let func = simple_func(
        10,
        "id_int",
        vec![tparam(0, "a", Type::Int)],
        Type::Int,
        tblock(vec![], Some(TAtomic::Var(id(0), "a".to_string())), Type::Int),
    );
    let program = make_free_func_program(func);
    let pure = lower_program(program);
    let f = expect_free_func(&pure, 0);
    assert_eq!(f.kind, PFuncKind::Sync);
    assert_eq!(
        f.body.tail_expr,
        Some(PAtomic::Var(id(0), "a".to_string()))
    );
}

#[test]
fn test_async_func_kind() {
    let mut f = simple_func(
        11,
        "do_it",
        vec![],
        Type::Nil,
        tblock(vec![], None, Type::Nil),
    );
    f.is_sync = false;
    let program = make_free_func_program(f);
    let pure = lower_program(program);
    assert_eq!(expect_free_func(&pure, 0).kind, PFuncKind::Async);
}


// Test 1: pure-assign-only block — var x = 1; x = 2; tail x
#[test]
fn test_var_assign_eliminates_to_env_update() {
    // var x = 1;
    let stmt_let = tlet(0, "x", Type::Int, texpr_int(TAtomic::IntLit(1)));
    // x = 2;
    let stmt_assign = tassign_var(0, "x", TAtomic::IntLit(2));

    let body = tblock(
        vec![stmt_let, stmt_assign],
        Some(TAtomic::Var(id(0), "x".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);
    let f = expect_free_func(&pure, 0);

    // Only the original LetAtom remains (the assign becomes an env-only op).
    assert_eq!(f.body.statements.len(), 1);
    let la = expect_let_atom(&f.body.statements[0]);
    assert_eq!(la.original_name, "x");
    // tail_expr must resolve to the post-assign value: IntLit(2)
    assert_eq!(f.body.tail_expr, Some(PAtomic::IntLit(2)));
    assert_no_assign(&f.body);
}

// Test 2: assign in both arms of if — tuple-join with single joined var.
// var x = 0;
// if c { x = 1 } else { x = 2 };
// tail x
#[test]
fn test_cond_stmt_joins_mutated_var() {
    let stmt_let_c = tlet(1, "c", Type::Bool, texpr_int(TAtomic::BoolLit(true)));
    let stmt_let_x = tlet(0, "x", Type::Int, texpr_int(TAtomic::IntLit(0)));

    // if c { x = 1 } else { x = 2 }
    let cond = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(
                vec![tassign_var(0, "x", TAtomic::IntLit(1))],
                None,
                Type::Nil,
            ),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: Some(tblock(
            vec![tassign_var(0, "x", TAtomic::IntLit(2))],
            None,
            Type::Nil,
        )),
        span: dummy_span(),
    };
    let cond_expr = TExpr {
        kind: TExprKind::Conditional(Box::new(cond)),
        ty: Type::Nil,
        span: dummy_span(),
    };
    let cond_stmt = TStatement {
        kind: TStatementKind::Expr(cond_expr),
        span: dummy_span(),
    };

    let body = tblock(
        vec![stmt_let_c, stmt_let_x, cond_stmt],
        Some(TAtomic::Var(id(0), "x".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);
    let f = expect_free_func(&pure, 0);

    // Should find a LetAtom binding a Conditional whose type is a Tuple.
    let cond_bind = f
        .body
        .statements
        .iter()
        .find_map(|s| match &s.kind {
            PStatementKind::LetAtom(la) => match &la.value.kind {
                PExprKind::Conditional(_) => Some(la),
                _ => None,
            },
            _ => None,
        })
        .expect("expected a LetAtom binding a Conditional");

    // With joined vars, the tuple is (joined_var_0,) — arity 1 because no
    // value from the conditional is being consumed.
    match &cond_bind.ty {
        Type::Tuple(ts) => assert_eq!(ts.len(), 1),
        other => panic!("expected tuple cond type, got {:?}", other),
    }

    // The joined var is x. After projection, env[x] should resolve to a
    // fresh var, NOT to IntLit. Verify tail_expr is a Var.
    match &f.body.tail_expr {
        Some(PAtomic::Var(_, _)) => {}
        other => panic!("expected tail to be a Var, got {:?}", other),
    }

    // Sanity: both branches' tail_expr is now a Var of the tuple binding.
    let cond = match &cond_bind.value.kind {
        PExprKind::Conditional(c) => c,
        _ => unreachable!(),
    };
    assert!(matches!(cond.if_branch.body.tail_expr, Some(PAtomic::Var(_, _))));
    assert!(matches!(
        cond.else_branch.as_ref().unwrap().tail_expr,
        Some(PAtomic::Var(_, _))
    ));
    assert_no_assign(&f.body);
}

// Test 3: assign in only one arm — unchanged arm carries parent env.
// var x = 0;
// if c { x = 1 };
// tail x
#[test]
fn test_cond_one_arm_mutates_synthesizes_else() {
    let stmt_let_c = tlet(1, "c", Type::Bool, texpr_int(TAtomic::BoolLit(true)));
    let stmt_let_x = tlet(0, "x", Type::Int, texpr_int(TAtomic::IntLit(0)));

    let cond = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(
                vec![tassign_var(0, "x", TAtomic::IntLit(1))],
                None,
                Type::Nil,
            ),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: None,
        span: dummy_span(),
    };
    let cond_expr = TExpr {
        kind: TExprKind::Conditional(Box::new(cond)),
        ty: Type::Nil,
        span: dummy_span(),
    };
    let cond_stmt = TStatement {
        kind: TStatementKind::Expr(cond_expr),
        span: dummy_span(),
    };

    let body = tblock(
        vec![stmt_let_c, stmt_let_x, cond_stmt],
        Some(TAtomic::Var(id(0), "x".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);
    let f = expect_free_func(&pure, 0);

    // An else branch must have been synthesized carrying the snapshot x.
    let cond_bind = f
        .body
        .statements
        .iter()
        .find_map(|s| match &s.kind {
            PStatementKind::LetAtom(la) => match &la.value.kind {
                PExprKind::Conditional(c) => Some(c),
                _ => None,
            },
            _ => None,
        })
        .expect("expected conditional binding");
    assert!(cond_bind.else_branch.is_some());

    // Synthesized else branch is a tuple of (snapshot_x,). The env at
    // conditional entry maps x's NameId to its SSA var reference Var(0, "x")
    // — not to the inline literal the LetAtom bound — so that's what the
    // synthesized else carries through.
    let else_blk = cond_bind.else_branch.as_ref().unwrap();
    let tup_let = expect_let_atom(&else_blk.statements[0]);
    match &tup_let.value.kind {
        PExprKind::TupleLit(elems) => {
            assert_eq!(elems.len(), 1);
            assert_eq!(elems[0], PAtomic::Var(id(0), "x".to_string()));
        }
        other => panic!("expected TupleLit, got {:?}", other),
    }
    assert_no_assign(&f.body);
}

// Test 4 (modified): LetAtom RHS is a conditional that mutates a variable.
// var x = 0;
// var z = if c { x = 1; 10 } else { x = 2; 20 };
// tail z  (and x was also mutated)
#[test]
fn test_cond_value_with_join() {
    let stmt_let_c = tlet(1, "c", Type::Bool, texpr_int(TAtomic::BoolLit(true)));
    let stmt_let_x = tlet(0, "x", Type::Int, texpr_int(TAtomic::IntLit(0)));

    let cond = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(
                vec![tassign_var(0, "x", TAtomic::IntLit(1))],
                Some(TAtomic::IntLit(10)),
                Type::Int,
            ),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: Some(tblock(
            vec![tassign_var(0, "x", TAtomic::IntLit(2))],
            Some(TAtomic::IntLit(20)),
            Type::Int,
        )),
        span: dummy_span(),
    };
    let cond_expr = TExpr {
        kind: TExprKind::Conditional(Box::new(cond)),
        ty: Type::Int,
        span: dummy_span(),
    };
    let stmt_let_z = tlet(2, "z", Type::Int, cond_expr);

    let body = tblock(
        vec![stmt_let_c, stmt_let_x, stmt_let_z],
        Some(TAtomic::Var(id(2), "z".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);
    let f = expect_free_func(&pure, 0);

    // Expect the conditional result to be bound to a tuple (int, int),
    // then projected into `z` and the new `x`.
    let cond_bind = f
        .body
        .statements
        .iter()
        .find_map(|s| match &s.kind {
            PStatementKind::LetAtom(la) => match &la.value.kind {
                PExprKind::Conditional(_) => Some(la),
                _ => None,
            },
            _ => None,
        })
        .expect("expected cond binding");
    match &cond_bind.ty {
        Type::Tuple(ts) => assert_eq!(ts.len(), 2, "expected (value, x)"),
        other => panic!("expected Tuple type, got {:?}", other),
    }

    // After the cond binding, two LetAtoms: one for z, one for new x.
    let tup_idx = f
        .body
        .statements
        .iter()
        .position(|s| match &s.kind {
            PStatementKind::LetAtom(la) => {
                matches!(la.value.kind, PExprKind::Conditional(_))
            }
            _ => false,
        })
        .unwrap();
    // First projection binds la.name = cond_bind.projected_0 → z
    let z_proj = expect_let_atom(&f.body.statements[tup_idx + 1]);
    assert_eq!(z_proj.original_name, "z");
    match &z_proj.value.kind {
        PExprKind::TupleAccess(base, 0) => match base {
            PAtomic::Var(nid, _) => assert_eq!(*nid, cond_bind.name),
            _ => panic!("expected var"),
        },
        other => panic!("expected TupleAccess(_, 0), got {:?}", other),
    }
    // tail should point at z_proj
    assert_eq!(
        f.body.tail_expr,
        Some(PAtomic::Var(z_proj.name, z_proj.original_name.clone()))
    );
    assert_no_assign(&f.body);
}


// Test 8: loop with break only.
// var n = 0;
// loop { if c { break } };
// tail n
#[test]
fn test_loop_with_break_lifts_to_free_func() {
    let stmt_let_c = tlet(1, "c", Type::Bool, texpr_int(TAtomic::BoolLit(true)));
    let stmt_let_n = tlet(0, "n", Type::Int, texpr_int(TAtomic::IntLit(0)));

    // if c { break }
    let inner_if = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(vec![tbreak()], None, Type::Nil),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: None,
        span: dummy_span(),
    };
    let if_stmt = TStatement {
        kind: TStatementKind::Expr(TExpr {
            kind: TExprKind::Conditional(Box::new(inner_if)),
            ty: Type::Nil,
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let loop_stmt = tloop(vec![if_stmt]);

    let body = tblock(
        vec![stmt_let_c, stmt_let_n, loop_stmt],
        Some(TAtomic::Var(id(0), "n".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);

    // Exactly one lifted function emitted.
    assert_eq!(count_loop_converted(&pure), 1);
    let main = expect_free_func(&pure, 0);
    assert_no_loop(&main.body);

    // The lifted function has PFuncKind::LoopConverted and returns an Enum.
    let lifted = pure
        .top_level_defs
        .iter()
        .find_map(|d| match d {
            PTopLevelDef::FreeFunc(f) if f.kind == PFuncKind::LoopConverted => Some(f),
            _ => None,
        })
        .unwrap();
    match &lifted.return_type {
        Type::Enum(enum_id, _) => {
            let variants = pure.enum_defs.get(enum_id).expect("enum def registered");
            assert_eq!(variants.len(), 2);
            let names: Vec<&str> = variants.iter().map(|(n, _)| n.as_str()).collect();
            assert!(names.contains(&"Exit"));
            assert!(names.contains(&"Return"));
        }
        other => panic!("expected Enum return type, got {:?}", other),
    }

    // The lifted function's body ends with a tail-call to itself (fallthrough).
    let last = lifted.body.statements.last().unwrap();
    match &last.kind {
        PStatementKind::Return(PAtomic::Var(_, _)) => {}
        other => panic!("expected Return(var), got {:?}", other),
    }
    // Find a call-to-self LetAtom in the body.
    let self_call = lifted.body.statements.iter().any(|s| match &s.kind {
        PStatementKind::LetAtom(la) => match &la.value.kind {
            PExprKind::FuncCall(PFuncCall::User(u)) => u.name == lifted.name,
            _ => false,
        },
        _ => false,
    });
    assert!(self_call, "expected tail-call to self");

    // Main function: contains a LetAtom that calls the lifted fn, then
    // IsVariant, then the outer cond, then VariantPayload destructuring.
    let call_count = main
        .body
        .statements
        .iter()
        .filter(|s| match &s.kind {
            PStatementKind::LetAtom(la) => match &la.value.kind {
                PExprKind::FuncCall(PFuncCall::User(u)) => u.name == lifted.name,
                _ => false,
            },
            _ => false,
        })
        .count();
    assert_eq!(call_count, 1);

    let is_variant_present = main.body.statements.iter().any(|s| match &s.kind {
        PStatementKind::LetAtom(la) => matches!(la.value.kind, PExprKind::IsVariant(_, _)),
        _ => false,
    });
    assert!(is_variant_present, "expected IsVariant check");

    assert_no_assign(&main.body);
}

// Test 11: loop mutating outer var with break. Live-out flows through Exit.
// var n = 0;
// loop { n = 1; if c { break } };
// tail n
#[test]
fn test_loop_exit_payload_threads_live_out() {
    let stmt_let_c = tlet(1, "c", Type::Bool, texpr_int(TAtomic::BoolLit(true)));
    let stmt_let_n = tlet(0, "n", Type::Int, texpr_int(TAtomic::IntLit(0)));

    let inner_if = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(vec![tbreak()], None, Type::Nil),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: None,
        span: dummy_span(),
    };
    let if_stmt = TStatement {
        kind: TStatementKind::Expr(TExpr {
            kind: TExprKind::Conditional(Box::new(inner_if)),
            ty: Type::Nil,
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    // n = 1
    let assign_n = tassign_var(0, "n", TAtomic::IntLit(1));

    let loop_stmt = tloop(vec![assign_n, if_stmt]);

    let body = tblock(
        vec![stmt_let_c, stmt_let_n, loop_stmt],
        Some(TAtomic::Var(id(0), "n".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);

    assert_eq!(count_loop_converted(&pure), 1);
    let main = expect_free_func(&pure, 0);

    // After the loop, there must be a projection for `n` from the exit payload.
    // That means a LetAtom whose value is TupleAccess(..., 0). Find it.
    let has_proj = main.body.statements.iter().any(|s| match &s.kind {
        PStatementKind::LetAtom(la) => matches!(la.value.kind, PExprKind::TupleAccess(_, _)),
        _ => false,
    });
    assert!(has_proj, "expected TupleAccess projection for live-out n");

    // tail_expr should resolve to the new projected n (not the original IntLit
    // or the original var-0).
    match &main.body.tail_expr {
        Some(PAtomic::Var(nid, _)) => {
            assert_ne!(
                *nid,
                id(0),
                "tail expected to resolve to a fresh projected name"
            );
        }
        other => panic!("expected tail Var, got {:?}", other),
    }
    assert_no_assign(&main.body);
    assert_no_loop(&main.body);
}

// Test: trailing use of mutated var reads the re-projected version.
// var x = 0;
// if c { x = 1 } else { x = 2 };
// var y = x + 1;
// tail y
#[test]
fn test_post_cond_uses_joined_var() {
    let stmt_let_c = tlet(1, "c", Type::Bool, texpr_int(TAtomic::BoolLit(true)));
    let stmt_let_x = tlet(0, "x", Type::Int, texpr_int(TAtomic::IntLit(0)));

    let cond = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(
                vec![tassign_var(0, "x", TAtomic::IntLit(1))],
                None,
                Type::Nil,
            ),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: Some(tblock(
            vec![tassign_var(0, "x", TAtomic::IntLit(2))],
            None,
            Type::Nil,
        )),
        span: dummy_span(),
    };
    let cond_stmt = TStatement {
        kind: TStatementKind::Expr(TExpr {
            kind: TExprKind::Conditional(Box::new(cond)),
            ty: Type::Nil,
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    // var y = x + 1;
    let add_expr = TExpr {
        kind: TExprKind::BinOp(
            BinOp::Add,
            TAtomic::Var(id(0), "x".to_string()),
            TAtomic::IntLit(1),
        ),
        ty: Type::Int,
        span: dummy_span(),
    };
    let stmt_let_y = tlet(3, "y", Type::Int, add_expr);

    let body = tblock(
        vec![stmt_let_c, stmt_let_x, cond_stmt, stmt_let_y],
        Some(TAtomic::Var(id(3), "y".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);
    let f = expect_free_func(&pure, 0);

    // Find the LetAtom `y` and confirm its BinOp reads a *fresh* x
    // (not the original nameid 0, but a projected fresh name).
    let y_let = f
        .body
        .statements
        .iter()
        .find_map(|s| match &s.kind {
            PStatementKind::LetAtom(la) if la.original_name == "y" => Some(la),
            _ => None,
        })
        .expect("expected let y");
    match &y_let.value.kind {
        PExprKind::BinOp(BinOp::Add, a, b) => {
            match a {
                PAtomic::Var(nid, _) => assert_ne!(
                    *nid,
                    id(0),
                    "expected `x` read to resolve to a projected fresh NameId"
                ),
                other => panic!("expected Var, got {:?}", other),
            }
            assert_eq!(*b, PAtomic::IntLit(1));
        }
        other => panic!("expected BinOp, got {:?}", other),
    }
    assert_no_assign(&f.body);
}


fn find_lifted_funcs(program: &PProgram) -> Vec<&PFuncDef> {
    program
        .top_level_defs
        .iter()
        .filter_map(|d| match d {
            PTopLevelDef::FreeFunc(f) if f.kind == PFuncKind::LoopConverted => Some(f),
            _ => None,
        })
        .collect()
}

fn block_contains_variant_lit_named(block: &PBlock, target: &str) -> bool {
    block.statements.iter().any(|s| match &s.kind {
        PStatementKind::LetAtom(la) => match &la.value.kind {
            PExprKind::VariantLit(_, name, _) => name == target,
            PExprKind::Conditional(c) => {
                block_contains_variant_lit_named(&c.if_branch.body, target)
                    || c.elseif_branches
                        .iter()
                        .any(|b| block_contains_variant_lit_named(&b.body, target))
                    || c.else_branch
                        .as_ref()
                        .map_or(false, |b| block_contains_variant_lit_named(b, target))
            }
            PExprKind::Block(b) => block_contains_variant_lit_named(b, target),
            _ => false,
        },
        PStatementKind::Expr(e) => match &e.kind {
            PExprKind::Conditional(c) => {
                block_contains_variant_lit_named(&c.if_branch.body, target)
                    || c.elseif_branches
                        .iter()
                        .any(|b| block_contains_variant_lit_named(&b.body, target))
                    || c.else_branch
                        .as_ref()
                        .map_or(false, |b| block_contains_variant_lit_named(b, target))
            }
            PExprKind::Block(b) => block_contains_variant_lit_named(b, target),
            _ => false,
        },
        _ => false,
    })
}

fn block_contains_plain_return(block: &PBlock) -> bool {
    block.statements.iter().any(|s| match &s.kind {
        PStatementKind::Return(_) => true,
        PStatementKind::Expr(e) => match &e.kind {
            PExprKind::Conditional(c) => {
                block_contains_plain_return(&c.if_branch.body)
                    || c.elseif_branches
                        .iter()
                        .any(|b| block_contains_plain_return(&b.body))
                    || c.else_branch
                        .as_ref()
                        .map_or(false, |b| block_contains_plain_return(b))
            }
            PExprKind::Block(b) => block_contains_plain_return(b),
            _ => false,
        },
        _ => false,
    })
}

// Test 9: loop with continue — continue arm is a tail-call to lifted fn.
#[test]
fn test_loop_continue_tail_calls_self() {
    let loop_stmt = tloop(vec![tcontinue()]);
    let body = tblock(vec![loop_stmt], Some(TAtomic::IntLit(0)), Type::Int);
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);

    assert_eq!(count_loop_converted(&pure), 1);
    let lifted = find_lifted_funcs(&pure)[0];

    // Continue emits a self-call + Return; fallthrough at body end also emits
    // a self-call + Return. Both sit directly at the top level of the lifted
    // function body.
    let self_calls = lifted
        .body
        .statements
        .iter()
        .filter(|s| match &s.kind {
            PStatementKind::LetAtom(la) => match &la.value.kind {
                PExprKind::FuncCall(PFuncCall::User(u)) => u.name == lifted.name,
                _ => false,
            },
            _ => false,
        })
        .count();
    assert_eq!(
        self_calls, 2,
        "expected 2 self-calls (continue + fallthrough)"
    );

    let main = expect_free_func(&pure, 0);
    assert_no_loop(&main.body);
    assert_no_assign(&main.body);
}

// Test 10: loop with return — wrapped as LoopResult::Return; main extracts
// payload and emits plain return.
#[test]
fn test_loop_return_wraps_and_main_extracts() {
    let stmt_let_n = tlet(0, "n", Type::Int, texpr_int(TAtomic::IntLit(0)));
    let loop_stmt = tloop(vec![treturn(TAtomic::IntLit(7))]);
    let body = tblock(
        vec![stmt_let_n, loop_stmt],
        Some(TAtomic::Var(id(0), "n".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);

    assert_eq!(count_loop_converted(&pure), 1);
    let lifted = find_lifted_funcs(&pure)[0];

    // Lifted body must contain a VariantLit wrapping IntLit(7) as the Return
    // variant of its LoopResult enum.
    let wraps_return_7 = lifted.body.statements.iter().any(|s| match &s.kind {
        PStatementKind::LetAtom(la) => match &la.value.kind {
            PExprKind::VariantLit(_, name, Some(PAtomic::IntLit(7))) => name == "Return",
            _ => false,
        },
        _ => false,
    });
    assert!(
        wraps_return_7,
        "lifted should wrap Return(7) as LoopResult::Return"
    );

    let main = expect_free_func(&pure, 0);

    // Main body tests IsVariant for "Return".
    let has_is_return = main.body.statements.iter().any(|s| match &s.kind {
        PStatementKind::LetAtom(la) => match &la.value.kind {
            PExprKind::IsVariant(_, name) => name == "Return",
            _ => false,
        },
        _ => false,
    });
    assert!(has_is_return);

    // The conditional in main's body has a then-branch that extracts payload
    // and emits a plain Return (we're not in a loop context here).
    let cond = main
        .body
        .statements
        .iter()
        .find_map(|s| match &s.kind {
            PStatementKind::Expr(e) => match &e.kind {
                PExprKind::Conditional(c) => Some(c),
                _ => None,
            },
            _ => None,
        })
        .expect("main should contain cond-stmt for is_return dispatch");
    let then_body = &cond.if_branch.body;
    let has_payload = then_body.statements.iter().any(|s| match &s.kind {
        PStatementKind::LetAtom(la) => matches!(la.value.kind, PExprKind::VariantPayload(_)),
        _ => false,
    });
    assert!(has_payload, "then-branch should extract VariantPayload");
    let ends_plain_return = matches!(
        then_body.statements.last().map(|s| &s.kind),
        Some(PStatementKind::Return(_))
    );
    assert!(
        ends_plain_return,
        "then-branch should end with a plain Return"
    );
    assert_no_loop(&main.body);
    assert_no_assign(&main.body);
}

// Test 12: nested loops, inner break stays local — two lifted functions, each
// with its own LoopResult enum.
#[test]
fn test_nested_loops_inner_break_stays_local() {
    let inner_loop = tloop(vec![tbreak()]);
    let outer_loop = tloop(vec![inner_loop]);
    let body = tblock(vec![outer_loop], Some(TAtomic::IntLit(0)), Type::Int);
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);

    assert_eq!(count_loop_converted(&pure), 2);
    let lifted = find_lifted_funcs(&pure);
    assert_ne!(lifted[0].name, lifted[1].name);

    // Each lifted func returns its own synthesized Enum.
    let enum_ids: Vec<NameId> = lifted
        .iter()
        .map(|f| match &f.return_type {
            Type::Enum(eid, _) => *eid,
            other => panic!("expected Enum return, got {:?}", other),
        })
        .collect();
    assert_ne!(enum_ids[0], enum_ids[1]);
    assert!(pure.enum_defs.contains_key(&enum_ids[0]));
    assert!(pure.enum_defs.contains_key(&enum_ids[1]));

    let main = expect_free_func(&pure, 0);
    assert_no_loop(&main.body);
    assert_no_assign(&main.body);
}

// Test 13: nested loops, inner return — propagates through outer loop via
// Return variant wrapping at each level.
#[test]
fn test_nested_loops_inner_return_propagates() {
    let inner_loop = tloop(vec![treturn(TAtomic::IntLit(5))]);
    let outer_loop = tloop(vec![inner_loop]);
    let body = tblock(vec![outer_loop], Some(TAtomic::IntLit(0)), Type::Int);
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);

    assert_eq!(count_loop_converted(&pure), 2);
    let lifted = find_lifted_funcs(&pure);

    // Both lifted functions must contain a "Return" VariantLit somewhere:
    // - inner wraps IntLit(5)
    // - outer wraps the payload extracted from the inner call's is_return arm
    for f in &lifted {
        assert!(
            block_contains_variant_lit_named(&f.body, "Return"),
            "lifted {:?} should wrap Return via its LoopResult enum",
            f.original_name
        );
    }

    // Main's body dispatches via a plain Return inside its is_return arm.
    let main = expect_free_func(&pure, 0);
    assert!(
        block_contains_plain_return(&main.body),
        "main should ultimately emit a plain Return"
    );
    assert_no_loop(&main.body);
    assert_no_assign(&main.body);
}

// Test 14: global SSA invariant — every PLetAtom NameId is unique across the
// entire lowered program.
#[test]
fn test_global_ssa_invariant_unique_letatom_ids() {
    // Exercise the full zoo: conditional-join, loop, live-out projection.
    let stmt_let_c = tlet(1, "c", Type::Bool, texpr_int(TAtomic::BoolLit(true)));
    let stmt_let_n = tlet(0, "n", Type::Int, texpr_int(TAtomic::IntLit(0)));

    let cond = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(
                vec![tassign_var(0, "n", TAtomic::IntLit(1))],
                None,
                Type::Nil,
            ),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: None,
        span: dummy_span(),
    };
    let cond_stmt = TStatement {
        kind: TStatementKind::Expr(TExpr {
            kind: TExprKind::Conditional(Box::new(cond)),
            ty: Type::Nil,
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let inner_if = TCondExpr {
        if_branch: TIfBranch {
            condition: TAtomic::Var(id(1), "c".to_string()),
            body: tblock(vec![tbreak()], None, Type::Nil),
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: None,
        span: dummy_span(),
    };
    let if_stmt = TStatement {
        kind: TStatementKind::Expr(TExpr {
            kind: TExprKind::Conditional(Box::new(inner_if)),
            ty: Type::Nil,
            span: dummy_span(),
        }),
        span: dummy_span(),
    };
    let loop_stmt = tloop(vec![tassign_var(0, "n", TAtomic::IntLit(2)), if_stmt]);

    let body = tblock(
        vec![stmt_let_c, stmt_let_n, cond_stmt, loop_stmt],
        Some(TAtomic::Var(id(0), "n".to_string())),
        Type::Int,
    );
    let func = simple_func(10, "f", vec![], Type::Int, body);
    let program = make_free_func_program(func);
    let pure = lower_program(program);

    let mut seen: HashSet<NameId> = HashSet::new();
    for def in &pure.top_level_defs {
        if let PTopLevelDef::FreeFunc(f) = def {
            collect_let_atom_ids(&f.body, &mut seen);
        }
    }
    // Also sanity-check at least 1 loop was lifted.
    assert!(count_loop_converted(&pure) >= 1);
}

fn collect_let_atom_ids(block: &PBlock, seen: &mut HashSet<NameId>) {
    for s in &block.statements {
        match &s.kind {
            PStatementKind::LetAtom(la) => {
                assert!(
                    seen.insert(la.name),
                    "duplicate PLetAtom NameId {:?} (original_name = {:?})",
                    la.name,
                    la.original_name
                );
                collect_let_atom_ids_expr(&la.value, seen);
            }
            PStatementKind::Expr(e) => collect_let_atom_ids_expr(e, seen),
            _ => {}
        }
    }
}

fn collect_let_atom_ids_expr(e: &PExpr, seen: &mut HashSet<NameId>) {
    match &e.kind {
        PExprKind::Conditional(c) => {
            collect_let_atom_ids(&c.if_branch.body, seen);
            for b in &c.elseif_branches {
                collect_let_atom_ids(&b.body, seen);
            }
            if let Some(b) = &c.else_branch {
                collect_let_atom_ids(b, seen);
            }
        }
        PExprKind::Block(b) => collect_let_atom_ids(b, seen),
        _ => {}
    }
}
