use super::lower::lower_program;
use crate::analysis::resolver::NameId;
use crate::analysis::types::Type;
use crate::compiler::anf::*;
use crate::liquid::threaded::ast::*;
use crate::parser::Span;
use std::collections::HashMap;

fn dummy_span() -> Span {
    Span::default()
}

fn id(i: usize) -> NameId {
    NameId(i)
}

fn make_role_program(
    role_name: &str,
    role_name_id: usize,
    var_inits: Vec<AVarInit>,
    func_defs: Vec<AFuncDef>,
) -> AProgram {
    AProgram {
        top_level_defs: vec![ATopLevelDef::Role(ARoleDef {
            name: id(role_name_id),
            original_name: role_name.to_string(),
            var_inits,
            func_defs,
            span: dummy_span(),
        })],
        next_name_id: 100,
        id_to_name: HashMap::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
    }
}

fn make_free_func_program(func: AFuncDef) -> AProgram {
    AProgram {
        top_level_defs: vec![ATopLevelDef::FreeFunc(func)],
        next_name_id: 100,
        id_to_name: HashMap::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
    }
}

fn simple_func(
    name_id: usize,
    name: &str,
    params: Vec<AFuncParam>,
    return_type: Type,
    body: ABlock,
) -> AFuncDef {
    AFuncDef {
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

fn aparam(i: usize, name: &str, ty: Type) -> AFuncParam {
    AFuncParam {
        name: id(i),
        original_name: name.to_string(),
        ty,
        span: dummy_span(),
    }
}

fn expect_role(program: &TProgram, idx: usize) -> &TRoleDef {
    match &program.top_level_defs[idx] {
        TTopLevelDef::Role(r) => r,
        other => panic!("expected Role at index {}, got {:?}", idx, other),
    }
}

fn expect_free_func(program: &TProgram, idx: usize) -> &TFuncDef {
    match &program.top_level_defs[idx] {
        TTopLevelDef::FreeFunc(f) => f,
        other => panic!("expected FreeFunc at index {}, got {:?}", idx, other),
    }
}

fn expect_let_atom(stmt: &TStatement) -> &TLetAtom {
    match &stmt.kind {
        TStatementKind::LetAtom(la) => la,
        other => panic!("expected LetAtom, got {:?}", other),
    }
}

fn expect_assign(stmt: &TStatement) -> &TAssign {
    match &stmt.kind {
        TStatementKind::Assign(a) => a,
        other => panic!("expected Assign, got {:?}", other),
    }
}

#[test]
fn test_free_func_passthrough() {
    let func = simple_func(
        10,
        "add",
        vec![aparam(0, "a", Type::Int), aparam(1, "b", Type::Int)],
        Type::Int,
        ABlock {
            statements: vec![],
            tail_expr: Some(AAtomic::Var(id(0), "a".to_string())),
            ty: Type::Int,
            span: dummy_span(),
        },
    );
    let program = make_free_func_program(func);
    let threaded = lower_program(program);

    let f = expect_free_func(&threaded, 0);
    assert_eq!(f.original_name, "add");
    assert_eq!(f.params.len(), 2); // No extra s param
    assert_eq!(f.return_type, Type::Int); // Not wrapped in tuple
}

#[test]
fn test_role_state_extraction() {
    let var_inits = vec![
        AVarInit {
            name: id(0),
            original_name: "db".to_string(),
            type_def: Type::Map(Box::new(Type::String), Box::new(Type::Int)),
            stmts: vec![],
            value: AExpr {
                kind: AExprKind::MapLit(vec![]),
                ty: Type::Map(Box::new(Type::String), Box::new(Type::Int)),
                span: dummy_span(),
            },
            span: dummy_span(),
        },
        AVarInit {
            name: id(1),
            original_name: "count".to_string(),
            type_def: Type::Int,
            stmts: vec![],
            value: AExpr {
                kind: AExprKind::Atomic(AAtomic::IntLit(0)),
                ty: Type::Int,
                span: dummy_span(),
            },
            span: dummy_span(),
        },
    ];

    // Simple identity function
    let func = simple_func(
        10,
        "get_count",
        vec![],
        Type::Int,
        ABlock {
            statements: vec![],
            tail_expr: Some(AAtomic::IntLit(42)),
            ty: Type::Int,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, var_inits, vec![func]);
    let threaded = lower_program(program);

    // Init function should be first (index 0), role second (index 1)
    let init_func = expect_free_func(&threaded, 0);
    assert!(init_func.original_name.contains("Node_init"));
    assert_eq!(init_func.params.len(), 0);

    // The return type should be the state struct type
    match &init_func.return_type {
        Type::Struct(_, name) => assert!(name.contains("Node_state")),
        other => panic!("expected Struct return type, got {:?}", other),
    }

    // Struct should be registered in struct_defs
    let state_struct_id = match &init_func.return_type {
        Type::Struct(id, _) => *id,
        _ => unreachable!(),
    };
    let fields = threaded.struct_defs.get(&state_struct_id).unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].0, "db");
    assert_eq!(fields[1].0, "count");

    // Role should have no var_inits (only func_defs)
    let role = expect_role(&threaded, 1);
    assert_eq!(role.original_name, "Node");
}

#[test]
fn test_func_signature_resign() {
    let func = simple_func(
        10,
        "process",
        vec![aparam(2, "v", Type::Int)],
        Type::Int,
        ABlock {
            statements: vec![],
            tail_expr: Some(AAtomic::Var(id(2), "v".to_string())),
            ty: Type::Int,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // Should have s as first param, then v
    assert_eq!(f.params.len(), 2);
    assert_eq!(f.params[0].original_name, "s");
    assert_eq!(f.params[1].original_name, "v");

    // Return type should be (State, Int)
    match &f.return_type {
        Type::Tuple(types) => {
            assert_eq!(types.len(), 2);
            assert!(matches!(&types[0], Type::Struct(_, _)));
            assert_eq!(types[1], Type::Int);
        }
        other => panic!("expected Tuple return type, got {:?}", other),
    }
}

#[test]
fn test_role_var_assign_rewrite() {
    let func = simple_func(
        10,
        "set_x",
        vec![],
        Type::Nil,
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::Assign(AAssign {
                    target_id: id(0),
                    target_name: "x".to_string(),
                    ty: Type::Int,
                    value: AAtomic::IntLit(5),
                    span: dummy_span(),
                }),
                span: dummy_span(),
            }],
            tail_expr: None,
            ty: Type::Nil,
            span: dummy_span(),
        },
    );

    let var_inits = vec![AVarInit {
        name: id(0),
        original_name: "x".to_string(),
        type_def: Type::Int,
        stmts: vec![],
        value: AExpr {
            kind: AExprKind::Atomic(AAtomic::IntLit(0)),
            ty: Type::Int,
            span: dummy_span(),
        },
        span: dummy_span(),
    }];

    let program = make_role_program("Node", 50, var_inits, vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // Role-level writes desugar to
    //   var __upd = Store(s, "x", 5);
    //   s = __upd;
    // Find the LetAtom whose value is Store(Var(s), StringLit("x"), IntLit(5)).
    let store_let = f
        .body
        .statements
        .iter()
        .find_map(|s| match &s.kind {
            TStatementKind::LetAtom(la) => match &la.value.kind {
                TExprKind::Store(base, key, val) => {
                    if matches!(base, TAtomic::Var(_, n) if n == "s")
                        && matches!(key, TAtomic::StringLit(k) if k == "x")
                        && matches!(val, TAtomic::IntLit(5))
                    {
                        Some(la)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        })
        .expect("expected LetAtom(Store(s, \"x\", 5))");

    // The following statement should be `s = __upd;` as a flat TAssign.
    let assign_stmt = f
        .body
        .statements
        .iter()
        .find(|s| matches!(s.kind, TStatementKind::Assign(_)))
        .expect("expected an Assign statement");
    let assign = expect_assign(assign_stmt);
    assert_eq!(assign.target_name, "s");
    assert!(matches!(assign.ty, Type::Struct(_, _)));
    match &assign.value {
        TAtomic::Var(nid, name) => {
            assert_eq!(*nid, store_let.name);
            assert_eq!(name, &store_let.original_name);
        }
        other => panic!("expected Var(__upd), got {:?}", other),
    }
}

#[test]
fn test_role_method_call_threading() {
    // var res = modify(a);  where modify is a role method (!is_free)
    let call_expr = AExpr {
        kind: AExprKind::FuncCall(AFuncCall::User(AUserFuncCall {
            name: id(11),
            original_name: "modify".to_string(),
            args: vec![AAtomic::Var(id(2), "a".to_string())],
            return_type: Type::Int,
            is_free: false,
            span: dummy_span(),
        })),
        ty: Type::Int,
        span: dummy_span(),
    };

    let func = simple_func(
        10,
        "caller",
        vec![aparam(2, "a", Type::Int)],
        Type::Int,
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::LetAtom(ALetAtom {
                    name: id(3),
                    original_name: "res".to_string(),
                    ty: Type::Int,
                    value: call_expr,
                    span: dummy_span(),
                }),
                span: dummy_span(),
            }],
            tail_expr: Some(AAtomic::Var(id(3), "res".to_string())),
            ty: Type::Int,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // Should have expanded: var __tup = modify(s, a); s = __tup.0; var res = __tup.1;
    // That's 4 statements: LetAtom(__tup), LetAtom(__s_new), Assign(s), LetAtom(res)
    // Plus the tail wrapping: LetAtom(__ret_tuple)
    // So we expect at least 4 statements before the tail binding
    assert!(
        f.body.statements.len() >= 4,
        "expected >= 4 stmts, got {}:\n{:#?}",
        f.body.statements.len(),
        f.body.statements,
    );

    // First statement should be LetAtom binding the tuple from the call
    let tup_let = expect_let_atom(&f.body.statements[0]);
    match &tup_let.value.kind {
        TExprKind::FuncCall(TFuncCall::User(call)) => {
            assert_eq!(call.original_name, "modify");
            // First arg should be s
            assert!(matches!(&call.args[0], TAtomic::Var(_, name) if name == "s"));
            // Second arg should be a
            assert_eq!(call.args[1], TAtomic::Var(id(2), "a".to_string()));
        }
        other => panic!("expected User FuncCall, got {:?}", other),
    }
}

#[test]
fn test_return_tuple_wrapping() {
    let func = simple_func(
        10,
        "get_val",
        vec![],
        Type::Int,
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::Return(AAtomic::IntLit(42)),
                span: dummy_span(),
            }],
            tail_expr: None,
            ty: Type::Int,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // Should have: LetAtom(__ret_tuple = (s, 42)), Return(__ret_tuple)
    assert_eq!(f.body.statements.len(), 2);

    let ret_let = expect_let_atom(&f.body.statements[0]);
    match &ret_let.value.kind {
        TExprKind::TupleLit(items) => {
            assert_eq!(items.len(), 2);
            assert!(matches!(&items[0], TAtomic::Var(_, name) if name == "s"));
            assert_eq!(items[1], TAtomic::IntLit(42));
        }
        other => panic!("expected TupleLit, got {:?}", other),
    }

    match &f.body.statements[1].kind {
        TStatementKind::Return(atomic) => {
            assert_eq!(
                *atomic,
                TAtomic::Var(ret_let.name, ret_let.original_name.clone())
            );
        }
        other => panic!("expected Return, got {:?}", other),
    }
}

#[test]
fn test_send_state_injection() {
    // var res = Send(chan, val);  in a role method
    let send_expr = AExpr {
        kind: AExprKind::Send(
            AAtomic::Var(id(2), "chan".to_string()),
            AAtomic::Var(id(3), "val".to_string()),
        ),
        ty: Type::Nil,
        span: dummy_span(),
    };

    let func = simple_func(
        10,
        "sender",
        vec![
            aparam(2, "chan", Type::Chan(Box::new(Type::Int))),
            aparam(3, "val", Type::Int),
        ],
        Type::Nil,
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::LetAtom(ALetAtom {
                    name: id(4),
                    original_name: "res".to_string(),
                    ty: Type::Nil,
                    value: send_expr,
                    span: dummy_span(),
                }),
                span: dummy_span(),
            }],
            tail_expr: None,
            ty: Type::Nil,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // The Send LetAtom should be expanded with tuple unpacking
    // First stmt: LetAtom(__tup = Send(s, chan, val))
    let tup_let = expect_let_atom(&f.body.statements[0]);
    match &tup_let.value.kind {
        TExprKind::Send(state, chan, val) => {
            assert!(matches!(state, TAtomic::Var(_, name) if name == "s"));
            assert_eq!(*chan, TAtomic::Var(id(2), "chan".to_string()));
            assert_eq!(*val, TAtomic::Var(id(3), "val".to_string()));
        }
        other => panic!("expected Send, got {:?}", other),
    }
}

/// Explicit `return val;` must wrap the value in (State, T) using the
/// declared return type, not a hardcoded `Nil`. Previously lowered as
/// `(State, Nil)` regardless of the function signature.
#[test]
fn test_explicit_return_uses_declared_return_type() {
    let ret_ty = Type::Optional(Box::new(Type::String));
    let func = simple_func(
        10,
        "get_opt",
        vec![],
        ret_ty.clone(),
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::Return(AAtomic::NilLit),
                span: dummy_span(),
            }],
            tail_expr: None,
            ty: ret_ty.clone(),
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    let ret_let = expect_let_atom(&f.body.statements[0]);
    match &ret_let.ty {
        Type::Tuple(types) => {
            assert_eq!(types.len(), 2);
            assert!(matches!(&types[0], Type::Struct(_, _)));
            assert_eq!(types[1], ret_ty, "second tuple element must match declared return type");
        }
        other => panic!("expected Tuple type on __ret_tuple, got {:?}", other),
    }
}

/// Async user calls (return type `chan<T>`) must lower 1:1 with just `s`
/// prepended — no `__tup`, no tuple access. The state handoff happens at
/// `<-` (see `test_send_state_injection` / `emit_threaded_send_recv`).
#[test]
fn test_async_call_no_tuple_unpack() {
    let chan_ty = Type::Chan(Box::new(Type::String));
    let call_expr = AExpr {
        kind: AExprKind::FuncCall(AFuncCall::User(AUserFuncCall {
            name: id(11),
            original_name: "Read".to_string(),
            args: vec![AAtomic::Var(id(2), "k".to_string())],
            return_type: chan_ty.clone(),
            is_free: false,
            span: dummy_span(),
        })),
        ty: chan_ty.clone(),
        span: dummy_span(),
    };

    let func = simple_func(
        10,
        "caller",
        vec![aparam(2, "k", Type::String)],
        Type::Nil,
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::LetAtom(ALetAtom {
                    name: id(3),
                    original_name: "ch".to_string(),
                    ty: chan_ty.clone(),
                    value: call_expr,
                    span: dummy_span(),
                }),
                span: dummy_span(),
            }],
            tail_expr: None,
            ty: Type::Nil,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // First statement is the direct binding `var ch: chan<string> = Read(s, k);`.
    let ch_let = expect_let_atom(&f.body.statements[0]);
    assert_eq!(ch_let.original_name, "ch");
    assert_eq!(ch_let.ty, chan_ty);
    match &ch_let.value.kind {
        TExprKind::FuncCall(TFuncCall::User(tcall)) => {
            assert_eq!(tcall.original_name, "Read");
            assert_eq!(tcall.return_type, chan_ty);
            assert_eq!(tcall.args.len(), 2);
            assert!(matches!(&tcall.args[0], TAtomic::Var(_, n) if n == "s"));
            assert_eq!(tcall.args[1], TAtomic::Var(id(2), "k".to_string()));
        }
        other => panic!("expected FuncCall, got {:?}", other),
    }

    // There must not be a tuple access or an `s = __tup.0` update generated
    // by this call — state handoff must be deferred to `<-`.
    for stmt in &f.body.statements {
        if let TStatementKind::LetAtom(la) = &stmt.kind {
            if let TExprKind::TupleAccess(_, _) = &la.value.kind {
                panic!(
                    "async call produced an unexpected TupleAccess: {:?}",
                    stmt
                );
            }
        }
    }
}

/// Sync role-method calls used as expression-statements must still thread
/// state: `f(a);` → `var __tup = f(s, a); s = __tup.0;` (no `res` binding).
/// This is the bug that caused `enter_view_change(new_view);` in VR.spur to
/// lose its mutations.
#[test]
fn test_sync_statement_call_threads_state() {
    let call_expr = AExpr {
        kind: AExprKind::FuncCall(AFuncCall::User(AUserFuncCall {
            name: id(11),
            original_name: "modify".to_string(),
            args: vec![AAtomic::Var(id(2), "a".to_string())],
            return_type: Type::Int,
            is_free: false,
            span: dummy_span(),
        })),
        ty: Type::Int,
        span: dummy_span(),
    };

    let func = simple_func(
        10,
        "caller",
        vec![aparam(2, "a", Type::Int)],
        Type::Nil,
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::Expr(call_expr),
                span: dummy_span(),
            }],
            tail_expr: None,
            ty: Type::Nil,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // stmt[0]: let __tup = modify(s, a);
    let tup_let = expect_let_atom(&f.body.statements[0]);
    match &tup_let.value.kind {
        TExprKind::FuncCall(TFuncCall::User(tcall)) => {
            assert_eq!(tcall.original_name, "modify");
            assert!(matches!(&tcall.args[0], TAtomic::Var(_, n) if n == "s"));
            assert_eq!(tcall.args[1], TAtomic::Var(id(2), "a".to_string()));
        }
        other => panic!("expected FuncCall, got {:?}", other),
    }
    match &tup_let.ty {
        Type::Tuple(types) => {
            assert_eq!(types.len(), 2);
            assert_eq!(types[1], Type::Int);
        }
        other => panic!("expected Tuple type on __tup, got {:?}", other),
    }

    // stmt[1]: let __s_new = __tup.0;
    let s_new_let = expect_let_atom(&f.body.statements[1]);
    match &s_new_let.value.kind {
        TExprKind::TupleAccess(TAtomic::Var(_, n), 0) => {
            assert_eq!(n, &tup_let.original_name);
        }
        other => panic!("expected TupleAccess(tup, 0), got {:?}", other),
    }

    // stmt[2]: s = __s_new;
    let assign = expect_assign(&f.body.statements[2]);
    assert_eq!(assign.target_name, "s");

    // No subsequent LetAtom should bind `__tup.1` — the result is discarded.
    for stmt in &f.body.statements[3..] {
        if let TStatementKind::LetAtom(la) = &stmt.kind {
            if let TExprKind::TupleAccess(_, 1) = &la.value.kind {
                panic!("discard path should not bind __tup.1: {:?}", stmt);
            }
        }
    }
}

/// Async role-method calls used as expression-statements become plain
/// `f(s, args);` spawn — no state update, since the result channel is
/// never awaited.
#[test]
fn test_async_statement_call_prepends_s_only() {
    let chan_ty = Type::Chan(Box::new(Type::Nil));
    let call_expr = AExpr {
        kind: AExprKind::FuncCall(AFuncCall::User(AUserFuncCall {
            name: id(11),
            original_name: "fire".to_string(),
            args: vec![AAtomic::Var(id(2), "a".to_string())],
            return_type: chan_ty.clone(),
            is_free: false,
            span: dummy_span(),
        })),
        ty: chan_ty.clone(),
        span: dummy_span(),
    };

    let func = simple_func(
        10,
        "caller",
        vec![aparam(2, "a", Type::Int)],
        Type::Nil,
        ABlock {
            statements: vec![AStatement {
                kind: AStatementKind::Expr(call_expr),
                span: dummy_span(),
            }],
            tail_expr: None,
            ty: Type::Nil,
            span: dummy_span(),
        },
    );

    let program = make_role_program("Node", 50, vec![], vec![func]);
    let threaded = lower_program(program);

    let role = expect_role(&threaded, 1);
    let f = &role.func_defs[0];

    // Should be exactly one statement: Expr(fire(s, a))
    // plus the trailing __ret_tuple wrapping the nil tail. Locate the spawn.
    let spawn_stmt = f
        .body
        .statements
        .iter()
        .find(|s| matches!(&s.kind, TStatementKind::Expr(_)))
        .expect("expected an Expr statement for the async spawn");
    match &spawn_stmt.kind {
        TStatementKind::Expr(texpr) => match &texpr.kind {
            TExprKind::FuncCall(TFuncCall::User(tcall)) => {
                assert_eq!(tcall.original_name, "fire");
                assert_eq!(tcall.return_type, chan_ty);
                assert_eq!(tcall.args.len(), 2);
                assert!(matches!(&tcall.args[0], TAtomic::Var(_, n) if n == "s"));
                assert_eq!(tcall.args[1], TAtomic::Var(id(2), "a".to_string()));
            }
            other => panic!("expected FuncCall, got {:?}", other),
        },
        other => panic!("expected Expr, got {:?}", other),
    }

    // Spawn must NOT produce a state update (no `s = __tup.0`).
    for stmt in &f.body.statements {
        if let TStatementKind::Assign(a) = &stmt.kind {
            if a.target_name == "s" {
                panic!("async spawn must not update s: {:?}", stmt);
            }
        }
    }
}
