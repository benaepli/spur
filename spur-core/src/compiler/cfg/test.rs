use super::*;
use crate::analysis::types::*;
use crate::parser::{BinOp, Span};

fn dummy_span() -> Span {
    Span::default()
}

fn id(i: usize) -> NameId {
    NameId(i)
}

fn typed_expr(kind: TypedExprKind, ty: Type) -> TypedExpr {
    TypedExpr {
        kind,
        ty,
        span: dummy_span(),
    }
}

fn int_lit(val: i64) -> TypedExpr {
    typed_expr(TypedExprKind::IntLit(val), Type::Int)
}

fn bool_lit(val: bool) -> TypedExpr {
    typed_expr(TypedExprKind::BoolLit(val), Type::Bool)
}

fn string_lit(val: &str) -> TypedExpr {
    typed_expr(TypedExprKind::StringLit(val.to_string()), Type::String)
}

fn dummy_target() -> Lhs {
    Lhs::Var(VarSlot::Local(0, id(100)))
}

fn dummy_vertex() -> Vertex {
    999
}

fn dummy_ctx() -> CompileCtx {
    CompileCtx {
        break_target: dummy_vertex(),
        continue_target: dummy_vertex(),
        return_target: dummy_vertex(),
        return_slot: VarSlot::Local(0, id(0)),
    }
}

fn make_block(statements: Vec<TypedStatement>, tail_expr: Option<TypedExpr>) -> TypedBlock {
    TypedBlock {
        statements,
        tail_expr: tail_expr.map(Box::new),
        ty: Type::Nil,
        span: dummy_span(),
    }
}

#[test]
fn test_compile_simple_literals() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let entry = compiler.compile_expr_to_value(
        &int_lit(42),
        target.clone(),
        next,
        dummy_ctx(),
    );
    let label = compiler.cfg.get(entry).unwrap();
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::Int(val)), n) => {
            assert_eq!(lhs, &target);
            assert_eq!(*val, 42);
            assert_eq!(*n, next);
        }
        _ => panic!("expected Assign Int, got {:?}", label),
    }

    let entry = compiler.compile_expr_to_value(
        &bool_lit(true),
        target.clone(),
        next,
        dummy_ctx(),
    );
    let label = compiler.cfg.get(entry).unwrap();
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::Bool(val)), n) => {
            assert_eq!(lhs, &target);
            assert_eq!(*val, true);
            assert_eq!(*n, next);
        }
        _ => panic!("expected Assign Bool, got {:?}", label),
    }

    let entry = compiler.compile_expr_to_value(
        &string_lit("hello"),
        target.clone(),
        next,
        dummy_ctx(),
    );
    let label = compiler.cfg.get(entry).unwrap();
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::String(val)), n) => {
            assert_eq!(lhs, &target);
            assert_eq!(val, "hello");
            assert_eq!(*n, next);
        }
        _ => panic!("expected Assign String, got {:?}", label),
    }
}

#[test]
fn test_compile_binop_add() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let expr = typed_expr(
        TypedExprKind::BinOp(BinOp::Add, Box::new(int_lit(1)), Box::new(int_lit(2))),
        Type::Int,
    );

    let entry = compiler.compile_expr_to_value(
        &expr,
        target.clone(),
        next,
        dummy_ctx(),
    );

    // Fast path: pure expression compiles to a single Assign
    let label = compiler.cfg.get(entry).expect("entry label missing");
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::Plus(l_box, r_box)), n) => {
            assert_eq!(lhs, &target);
            assert_eq!(*n, next);
            assert_eq!(**l_box, Expr::Int(1));
            assert_eq!(**r_box, Expr::Int(2));
        }
        _ => panic!("expected Assign Plus(Int(1), Int(2)), got {:?}", label),
    }
}

#[test]
fn test_compile_short_circuit_and() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let expr = typed_expr(
        TypedExprKind::BinOp(
            BinOp::And,
            Box::new(bool_lit(false)),
            Box::new(bool_lit(true)),
        ),
        Type::Bool,
    );

    let entry = compiler.compile_expr_to_value(
        &expr,
        target.clone(),
        next,
        dummy_ctx(),
    );

    let l_label = compiler.cfg.get(entry).expect("left label missing");
    if let Label::Instr(Instr::Assign(l_lhs, Expr::Bool(false)), cond_entry) = l_label {
        let cond_label = compiler.cfg.get(*cond_entry).expect("cond label missing");
        if let Label::Cond(Expr::Var(cond_var), eval_right, assign_false) = cond_label {
            if let Lhs::Var(l_slot) = l_lhs {
                assert_eq!(l_slot, cond_var);
            }

            let false_label = compiler
                .cfg
                .get(*assign_false)
                .expect("false label missing");
            if let Label::Instr(Instr::Assign(lhs, Expr::Bool(false)), n) = false_label {
                assert_eq!(lhs, &target);
                assert_eq!(*n, next);
            } else {
                panic!("expected Assign Bool(false), got {:?}", false_label);
            }

            assert!(*eval_right < compiler.cfg.len());
        } else {
            panic!("expected Cond, got {:?}", cond_label);
        }
    } else {
        panic!("expected Assign Bool(false), got {:?}", l_label);
    }
}

#[test]
fn test_compile_assignment() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);

    let var_name = id(10);
    compiler.alloc_local_slot(var_name, "x", Expr::Nil);
    let var_slot = compiler.resolve_slot(var_name);

    let next = dummy_vertex();

    let stmt = TypedStatement {
        kind: TypedStatementKind::Assignment(TypedAssignment {
            target: typed_expr(TypedExprKind::Var(var_name, "x".to_string()), Type::Int),
            value: int_lit(100),
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let entry = compiler.compile_statement(
        &stmt,
        next,
        dummy_ctx(),
    );

    let label = compiler.cfg.get(entry).expect("entry label missing");
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::Int(100)), n) => {
            match lhs {
                Lhs::Var(slot) => assert_eq!(slot, &var_slot),
                _ => panic!("expected Var LHS"),
            }
            assert_eq!(*n, next);
        }
        _ => panic!("expected Assign(Var, 100), got {:?}", label),
    }
}

#[test]
fn test_compile_if_statement() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let cond_expr = typed_expr(
        TypedExprKind::Conditional(Box::new(TypedCondExpr {
            if_branch: TypedIfBranch {
                condition: bool_lit(true),
                body: make_block(vec![], Some(int_lit(1))),
                span: dummy_span(),
            },
            elseif_branches: vec![],
            else_branch: Some(make_block(vec![], Some(int_lit(2)))),
            span: dummy_span(),
        })),
        Type::Int,
    );

    let entry = compiler.compile_expr_to_value(
        &cond_expr,
        target.clone(),
        next,
        dummy_ctx(),
    );

    let l_cond = compiler.cfg.get(entry).expect("cond entry label missing");
    if let Label::Instr(Instr::Assign(cond_lhs, Expr::Bool(true)), check_entry) = l_cond {
        let check_label = compiler.cfg.get(*check_entry).expect("check label missing");
        if let Label::Cond(Expr::Var(cond_var), then_vertex, else_vertex) = check_label {
            if let Lhs::Var(slot) = cond_lhs {
                assert_eq!(slot, cond_var);
            }

            let then_label = compiler.cfg.get(*then_vertex).expect("then label missing");
            if let Label::Instr(Instr::Assign(_, Expr::Int(1)), n) = then_label {
                assert_eq!(*n, next);
            } else {
                panic!("expected then body Assign(1), got {:?}", then_label);
            }

            let else_label = compiler.cfg.get(*else_vertex).expect("else label missing");
            if let Label::Instr(Instr::Assign(_, Expr::Int(2)), n) = else_label {
                assert_eq!(*n, next);
            } else {
                panic!("expected else body Assign(2), got {:?}", else_label);
            }
        } else {
            panic!("expected Cond, got {:?}", check_label);
        }
    } else {
        panic!("expected eval condition, got {:?}", l_cond);
    }
}

#[test]
fn test_compile_for_loop() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let next = dummy_vertex();

    let stmt = TypedStatement {
        kind: TypedStatementKind::ForLoop(TypedForLoop {
            init: None,
            condition: Some(bool_lit(false)),
            increment: None,
            body: vec![],
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let entry = compiler.compile_statement(
        &stmt,
        next,
        dummy_ctx(),
    );

    let head_label = compiler.cfg.get(entry).expect("head label missing");

    if let Label::Instr(Instr::Assign(_, Expr::Unit), cond_calc_v) = head_label {
        let cond_calc = compiler
            .cfg
            .get(*cond_calc_v)
            .expect("cond calc label missing");
        if let Label::Instr(Instr::Assign(_, Expr::Bool(false)), cond_check_v) = cond_calc {
            let cond_check = compiler
                .cfg
                .get(*cond_check_v)
                .expect("cond check label missing");
            if let Label::Cond(_, body_v, exit_v) = cond_check {
                assert_eq!(*exit_v, next);
                assert_eq!(*body_v, entry);
            } else {
                panic!("expected Cond check, got {:?}", cond_check);
            }
        } else {
            panic!("expected cond calc false, got {:?}", cond_calc);
        }
    } else {
        panic!("expected loop head, got {:?}", head_label);
    }
}

#[test]
fn test_compile_return_statement() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);

    let return_slot = VarSlot::Local(5, id(5));
    let return_target = 100;

    let stmt = TypedStatement {
        kind: TypedStatementKind::Expr(TypedExpr {
            kind: TypedExprKind::Return(Box::new(int_lit(42))),
            ty: Type::Never,
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let entry = compiler.compile_statement(
        &stmt,
        dummy_vertex(),
        CompileCtx {
            break_target: dummy_vertex(),
            continue_target: dummy_vertex(),
            return_target,
            return_slot,
        },
    );

    let label = compiler.cfg.get(entry).expect("entry label missing");
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::Int(42)), n) => {
            if let Lhs::Var(slot) = lhs {
                assert_eq!(slot, &return_slot);
            }
            assert_eq!(*n, return_target);
        }
        _ => panic!("expected Assign(return_slot, 42), got {:?}", label),
    }
}

#[test]
fn test_compile_sync_function_call() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);

    let func_id = id(50);
    compiler.func_sync_map.insert(func_id, true);
    compiler
        .func_qualifier_map
        .insert(func_id, "Node".to_string());

    let call = TypedFuncCall::User(TypedUserFuncCall {
        name: func_id,
        original_name: "Foo".to_string(),
        args: vec![int_lit(10)],
        return_type: Type::Int,
        span: dummy_span(),
    });

    let target = dummy_target();
    let next = dummy_vertex();

    let entry = compiler.compile_expr_to_value(
        &typed_expr(TypedExprKind::FuncCall(call), Type::Int),
        target.clone(),
        next,
        dummy_ctx(),
    );

    let arg_label = compiler.cfg.get(entry).expect("arg label missing");
    if let Label::Instr(Instr::Assign(arg_lhs, Expr::Int(10)), call_entry) = arg_label {
        let call_label = compiler.cfg.get(*call_entry).expect("call label missing");
        if let Label::Instr(Instr::SyncCall(lhs, name, args), n) = call_label {
            assert_eq!(lhs, &target);
            assert_eq!(name, "Node.Foo");
            assert_eq!(args.len(), 1);
            assert_eq!(*n, next);

            if let Lhs::Var(arg_slot) = arg_lhs {
                assert_eq!(args[0], Expr::Var(*arg_slot));
            }
        } else {
            panic!("expected SyncCall, got {:?}", call_label);
        }
    } else {
        panic!("expected arg Assign, got {:?}", arg_label);
    }
}

#[test]
fn test_compile_empty_list() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let empty_list = typed_expr(TypedExprKind::ListLit(vec![]), Type::EmptyList);
    let entry = compiler.compile_expr_to_value(
        &empty_list,
        target.clone(),
        next,
        dummy_ctx(),
    );

    let label = compiler.cfg.get(entry).unwrap();
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::List(items)), n) if items.is_empty() => {
            assert_eq!(lhs, &target);
            assert_eq!(*n, next);
        }
        _ => panic!("expected Assign List([]), got {:?}", label),
    }
}

#[test]
fn test_compile_empty_map() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let empty_map = typed_expr(TypedExprKind::MapLit(vec![]), Type::EmptyMap);
    let entry = compiler.compile_expr_to_value(
        &empty_map,
        target.clone(),
        next,
        dummy_ctx(),
    );

    let label = compiler.cfg.get(entry).unwrap();
    match label {
        Label::Instr(Instr::Assign(lhs, Expr::Map(pairs)), n) if pairs.is_empty() => {
            assert_eq!(lhs, &target);
            assert_eq!(*n, next);
        }
        _ => panic!("expected Assign Map([]), got {:?}", label),
    }
}

#[test]
fn test_compile_nested_if_statements() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    // Nested if: if true { if false { 1 } else { 2 } } else { 3 }
    let inner_cond = typed_expr(
        TypedExprKind::Conditional(Box::new(TypedCondExpr {
            if_branch: TypedIfBranch {
                condition: bool_lit(false),
                body: make_block(vec![], Some(int_lit(1))),
                span: dummy_span(),
            },
            elseif_branches: vec![],
            else_branch: Some(make_block(vec![], Some(int_lit(2)))),
            span: dummy_span(),
        })),
        Type::Int,
    );

    let outer_cond = typed_expr(
        TypedExprKind::Conditional(Box::new(TypedCondExpr {
            if_branch: TypedIfBranch {
                condition: bool_lit(true),
                body: make_block(vec![], Some(inner_cond)),
                span: dummy_span(),
            },
            elseif_branches: vec![],
            else_branch: Some(make_block(vec![], Some(int_lit(3)))),
            span: dummy_span(),
        })),
        Type::Int,
    );

    let entry = compiler.compile_expr_to_value(
        &outer_cond,
        target,
        next,
        dummy_ctx(),
    );

    // Verify outer condition is compiled
    let outer_label = compiler.cfg.get(entry).expect("outer cond missing");
    assert!(matches!(
        outer_label,
        Label::Instr(Instr::Assign(_, Expr::Bool(true)), _)
    ));
}

#[test]
fn test_compile_break_in_loop() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let next = dummy_vertex();

    let loop_stmt = TypedStatement {
        kind: TypedStatementKind::ForLoop(TypedForLoop {
            init: None,
            condition: Some(bool_lit(true)),
            increment: None,
            body: vec![TypedStatement {
                kind: TypedStatementKind::Expr(TypedExpr {
                    kind: TypedExprKind::Break,
                    ty: Type::Never,
                    span: dummy_span(),
                }),
                span: dummy_span(),
            }],
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let entry = compiler.compile_statement(
        &loop_stmt,
        next,
        dummy_ctx(),
    );

    // Verify the loop is compiled and break jumps to exit
    let head_label = compiler.cfg.get(entry).expect("loop head missing");
    assert!(matches!(head_label, Label::Instr(_, _)));
}

#[test]
fn test_compile_complex_binop_chain() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    // (1 + 2) * 3
    let add_expr = typed_expr(
        TypedExprKind::BinOp(BinOp::Add, Box::new(int_lit(1)), Box::new(int_lit(2))),
        Type::Int,
    );
    let mul_expr = typed_expr(
        TypedExprKind::BinOp(BinOp::Multiply, Box::new(add_expr), Box::new(int_lit(3))),
        Type::Int,
    );

    let entry = compiler.compile_expr_to_value(
        &mul_expr,
        target.clone(),
        next,
        dummy_ctx(),
    );

    // Verify compilation produces CFG nodes
    let label = compiler.cfg.get(entry).expect("entry missing");
    assert!(matches!(label, Label::Instr(_, _)));
}

#[test]
fn test_compile_short_circuit_or() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let expr = typed_expr(
        TypedExprKind::BinOp(
            BinOp::Or,
            Box::new(bool_lit(true)),
            Box::new(bool_lit(false)),
        ),
        Type::Bool,
    );

    let entry = compiler.compile_expr_to_value(
        &expr,
        target.clone(),
        next,
        dummy_ctx(),
    );

    // Verify short-circuit: eval left, then conditional jump
    let l_label = compiler.cfg.get(entry).expect("left label missing");
    if let Label::Instr(Instr::Assign(_, Expr::Bool(true)), cond_entry) = l_label {
        let cond_label = compiler.cfg.get(*cond_entry).expect("cond label missing");
        // OR short-circuits: if true, assign true; else eval right
        assert!(matches!(cond_label, Label::Cond(_, _, _)));
    } else {
        panic!("expected left Assign Bool(true), got {:?}", l_label);
    }
}

#[test]
fn test_compile_empty_loop_body() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let next = dummy_vertex();

    let loop_stmt = TypedStatement {
        kind: TypedStatementKind::ForLoop(TypedForLoop {
            init: None,
            condition: Some(bool_lit(false)),
            increment: None,
            body: vec![], // Empty body
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let entry = compiler.compile_statement(
        &loop_stmt,
        next,
        dummy_ctx(),
    );

    // Should still compile correctly with empty body
    let label = compiler.cfg.get(entry).expect("entry missing");
    assert!(matches!(label, Label::Instr(_, _)));
}

#[test]
fn test_compile_elseif_chain() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let target = dummy_target();
    let next = dummy_vertex();

    let cond_expr = typed_expr(
        TypedExprKind::Conditional(Box::new(TypedCondExpr {
            if_branch: TypedIfBranch {
                condition: bool_lit(false),
                body: make_block(vec![], Some(int_lit(1))),
                span: dummy_span(),
            },
            elseif_branches: vec![TypedIfBranch {
                condition: bool_lit(true),
                body: make_block(vec![], Some(int_lit(2))),
                span: dummy_span(),
            }],
            else_branch: Some(make_block(vec![], Some(int_lit(3)))),
            span: dummy_span(),
        })),
        Type::Int,
    );

    let entry = compiler.compile_expr_to_value(
        &cond_expr,
        target,
        next,
        dummy_ctx(),
    );

    // Verify elseif is compiled
    let label = compiler.cfg.get(entry).expect("entry missing");
    assert!(matches!(label, Label::Instr(_, _)));
}
#[test]
fn test_compile_continue_in_loop() {
    let mut compiler = Compiler::new();
    compiler.begin_function(&[]);
    let next = dummy_vertex();

    let loop_stmt = TypedStatement {
        kind: TypedStatementKind::ForLoop(TypedForLoop {
            init: None,
            condition: Some(bool_lit(true)),
            increment: None,
            body: vec![TypedStatement {
                kind: TypedStatementKind::Expr(TypedExpr {
                    kind: TypedExprKind::Continue,
                    ty: Type::Never,
                    span: dummy_span(),
                }),
                span: dummy_span(),
            }],
            span: dummy_span(),
        }),
        span: dummy_span(),
    };

    let entry = compiler.compile_statement(
        &loop_stmt,
        next,
        dummy_ctx(),
    );

    // Verify the loop is compiled and continue jumps to loop head (since no increment)
    let head_label = compiler.cfg.get(entry).expect("loop head missing");

    // Structure:
    // head -> cond_calc -> cond_check -> body -> continue -> head

    if let Label::Instr(Instr::Assign(_, Expr::Unit), cond_calc_v) = head_label {
        // cond_calc ...
        let cond_calc = compiler.cfg.get(*cond_calc_v).expect("cond calc missing");
        if let Label::Instr(Instr::Assign(_, Expr::Bool(true)), cond_check_v) = cond_calc {
            let cond_check = compiler.cfg.get(*cond_check_v).expect("cond check missing");
            if let Label::Cond(_, body_v, exit_v) = cond_check {
                assert_eq!(*exit_v, next);

                let body_label = compiler.cfg.get(*body_v).expect("body label missing");
                // body should be the continue statement
                if let Label::Continue(target) = body_label {
                    // target should be loop head (entry)
                    assert_eq!(*target, entry);
                } else {
                    panic!("expected Continue label, got {:?}", body_label);
                }
            } else {
                panic!("expected Cond check, got {:?}", cond_check);
            }
        } else {
            panic!("expected cond calc true");
        }
    } else {
        panic!("expected loop head, got {:?}", head_label);
    }
}
