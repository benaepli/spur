use super::*;
use spur_ast::span::Span;

fn span() -> Span {
    Span::default()
}

fn nid(n: usize) -> NameId {
    NameId(n)
}

fn var(id: usize) -> PAtomic {
    PAtomic::Var(nid(id), format!("v{}", id))
}

fn let_atom(name: usize, ty: Type, value: PExpr) -> PStatement {
    PStatement {
        kind: PStatementKind::LetAtom(PLetAtom {
            name: nid(name),
            original_name: format!("v{}", name),
            ty,
            value,
            user_annotated: false,
            span: span(),
        }),
        span: span(),
    }
}

fn block(stmts: Vec<PStatement>, tail: Option<PAtomic>, ty: Type) -> PBlock {
    PBlock {
        statements: stmts,
        tail_expr: tail,
        ty,
        span: span(),
    }
}

fn func(
    name: usize,
    params: Vec<(usize, Type)>,
    return_type: Type,
    body: PBlock,
) -> PFuncDef {
    PFuncDef {
        name: nid(name),
        original_name: format!("f{}", name),
        kind: PFuncKind::Sync,
        is_traced: false,
        params: params
            .into_iter()
            .map(|(id, ty)| PFuncParam {
                name: nid(id),
                original_name: format!("p{}", id),
                ty,
                span: span(),
            })
            .collect(),
        return_type,
        body,
        span: span(),
    }
}

fn empty_program(next_id: usize) -> PProgram {
    PProgram {
        top_level_defs: vec![],
        next_name_id: next_id,
        id_to_name: HashMap::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
    }
}

/// Look up an extern entry by its `original_name`. Panics if not found.
/// Tests use this to avoid relying on the order externs land in
/// `extern_funcs`, which now depends on when each is first interned
/// (e.g. the `list<T>` invariant interns `array_len<T>` before any user
/// call site is lowered).
#[track_caller]
fn find_extern<'a>(prog: &'a CProgram, name: &str) -> &'a CExternFunc {
    prog.extern_funcs
        .iter()
        .find(|e| e.original_name == name)
        .unwrap_or_else(|| {
            panic!(
                "expected extern {:?}, got: {:?}",
                name,
                prog.extern_funcs
                    .iter()
                    .map(|e| &e.original_name)
                    .collect::<Vec<_>>()
            )
        })
}

#[test]
fn append_int_and_append_string_get_distinct_externs() {
    // f(list_int: list<int>, list_str: list<string>, x: int, y: string) -> () {
    //   _ = append(list_int, x);
    //   _ = append(list_str, y);
    //   ()
    // }
    let list_int_ty = Type::List(Box::new(Type::Int));
    let list_str_ty = Type::List(Box::new(Type::String));

    let body = block(
        vec![
            let_atom(
                10,
                list_int_ty.clone(),
                PExpr {
                    kind: PExprKind::Append(var(1), var(3)),
                    ty: list_int_ty.clone(),
                    span: span(),
                },
            ),
            let_atom(
                11,
                list_str_ty.clone(),
                PExpr {
                    kind: PExprKind::Append(var(2), var(4)),
                    ty: list_str_ty.clone(),
                    span: span(),
                },
            ),
        ],
        None,
        Type::Tuple(vec![]),
    );

    let f = func(
        5,
        vec![
            (1, list_int_ty),
            (2, list_str_ty),
            (3, Type::Int),
            (4, Type::String),
        ],
        Type::Tuple(vec![]),
        body,
    );

    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));

    let c = lower_program(prog).program;

    // Each `list<T>` param's invariant interns one `array_len<T>` extern,
    // and each `Append` call interns an `array_append<T>` extern, for a
    // total of 4: array_len<int>, array_len<string>, array_append<int>,
    // array_append<string>.
    assert_eq!(c.extern_funcs.len(), 4);
    let app_int = find_extern(&c, "array_append<int>");
    let app_str = find_extern(&c, "array_append<string>");
    assert_ne!(app_int.name, app_str.name);
    let app_int_param_tys: Vec<CType> =
        app_int.params.iter().map(|p| p.ty.clone()).collect();
    let app_str_param_tys: Vec<CType> =
        app_str.params.iter().map(|p| p.ty.clone()).collect();
    assert_eq!(
        app_int_param_tys,
        vec![CType::Array(Box::new(CType::Int)), CType::Int]
    );
    assert_eq!(
        app_str_param_tys,
        vec![CType::Array(Box::new(CType::String)), CType::String]
    );
    // Both `array_len<T>` externs exist (from the list<T> invariant).
    assert!(c.extern_funcs.iter().any(|e| e.original_name == "array_len<int>"));
    assert!(c.extern_funcs.iter().any(|e| e.original_name == "array_len<string>"));
}

#[test]
fn append_called_twice_with_same_type_dedups() {
    let list_int_ty = Type::List(Box::new(Type::Int));
    let body = block(
        vec![
            let_atom(
                10,
                list_int_ty.clone(),
                PExpr {
                    kind: PExprKind::Append(var(1), var(2)),
                    ty: list_int_ty.clone(),
                    span: span(),
                },
            ),
            let_atom(
                11,
                list_int_ty.clone(),
                PExpr {
                    kind: PExprKind::Append(var(1), var(2)),
                    ty: list_int_ty.clone(),
                    span: span(),
                },
            ),
        ],
        None,
        Type::Tuple(vec![]),
    );
    let f = func(
        5,
        vec![(1, list_int_ty), (2, Type::Int)],
        Type::Tuple(vec![]),
        body,
    );
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));

    let c = lower_program(prog).program;
    // Two externs: `array_len<int>` (from the list<int> invariant) and
    // `array_append<int>` (deduped across both calls).
    assert_eq!(c.extern_funcs.len(), 2);
    assert!(c.extern_funcs.iter().any(|e| e.original_name == "array_len<int>"));
    assert!(c.extern_funcs.iter().any(|e| e.original_name == "array_append<int>"));
}

#[test]
fn make_iter_returns_iter_of_elem() {
    let list_int_ty = Type::List(Box::new(Type::Int));
    let iter_int_ty = Type::Iter(Box::new(Type::Int));
    let body = block(
        vec![let_atom(
            10,
            iter_int_ty.clone(),
            PExpr {
                kind: PExprKind::MakeIter(var(1)),
                ty: iter_int_ty,
                span: span(),
            },
        )],
        None,
        Type::Tuple(vec![]),
    );
    let f = func(5, vec![(1, list_int_ty)], Type::Tuple(vec![]), body);
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));

    let c = lower_program(prog).program;
    // Two externs: `array_len<int>` (from the list<int> param invariant)
    // and `iter_make<int>` (from the body's MakeIter call).
    assert_eq!(c.extern_funcs.len(), 2);
    let ext = find_extern(&c, "iter_make<int>");
    let ext_param_tys: Vec<CType> = ext.params.iter().map(|p| p.ty.clone()).collect();
    assert_eq!(ext_param_tys, vec![CType::Array(Box::new(CType::Int))]);
    assert_eq!(ext.return_type, CType::Iter(Box::new(CType::Int)));
}

use spur_ast::binop::BinOp;
use spur_ast::types::{
    RefinementBody, RefinementHandle, TypedExpr, TypedExprKind, TypedUserFuncCall,
};

fn typed(kind: TypedExprKind, ty: Type) -> TypedExpr {
    TypedExpr {
        kind,
        ty,
        span: span(),
    }
}

/// Build a `Type::Refined(inner, { bound | body })` with a fresh handle.
fn refined(inner: Type, bound: usize, body: TypedExpr) -> Type {
    Type::Refined(
        Box::new(inner),
        RefinementHandle::new(RefinementBody {
            bound: nid(bound),
            original_bound: format!("v{}", bound),
            body,
        }),
    )
}

/// A function that takes a parameter typed with the given refined type
/// and has an empty body. Lowering it forces the refinement-body
/// machinery to run.
fn func_with_refined_param(name: usize, param_id: usize, refined_ty: Type) -> PFuncDef {
    func(
        name,
        vec![(param_id, refined_ty)],
        Type::Tuple(vec![]),
        block(vec![], None, Type::Tuple(vec![])),
    )
}

fn lower_with_refinement(refined_ty: Type) -> LowerOutput {
    let f = func_with_refined_param(1, 2, refined_ty);
    let mut prog = empty_program(50);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    lower_program(prog)
}

#[test]
fn refinement_body_simple_predicate_uses_no_externs() {
    // int { x | x > 0 }
    let body = typed(
        TypedExprKind::BinOp(
            BinOp::Greater,
            Box::new(typed(
                TypedExprKind::Var(nid(7), "x".to_string()),
                Type::Int,
            )),
            Box::new(typed(TypedExprKind::IntLit(0), Type::Int)),
        ),
        Type::Bool,
    );
    let refined_ty = refined(Type::Int, 7, body);

    let lowered = lower_with_refinement(refined_ty);
    let c = lowered.program;

    assert!(lowered.refinement_errors.is_empty());
    assert_eq!(c.extern_funcs.len(), 0);

    let CType::Refined(inner, handle) = &c.funcs[0].params[0].ty else {
        panic!("expected refined type, got {:?}", c.funcs[0].params[0].ty);
    };
    assert_eq!(**inner, CType::Int);
    match &handle.body.kind {
        RefinementExprKind::BinOp(CBinOp::Greater, l, r) => {
            assert!(matches!(l.kind, RefinementExprKind::Var(NameId(7), _)));
            assert!(matches!(r.kind, RefinementExprKind::IntLit(0)));
        }
        other => panic!("expected BinOp(Greater, ..), got {:?}", other),
    }
}

#[test]
fn refinement_len_call_shares_extern_with_function_body_call() {
    // Refined param: list<int> { xs | len(xs) > 0 }
    let list_int_ty = Type::List(Box::new(Type::Int));
    let body = typed(
        TypedExprKind::BinOp(
            BinOp::Greater,
            Box::new(typed(
                TypedExprKind::Len(Box::new(typed(
                    TypedExprKind::Var(nid(8), "xs".to_string()),
                    list_int_ty.clone(),
                ))),
                Type::Int,
            )),
            Box::new(typed(TypedExprKind::IntLit(0), Type::Int)),
        ),
        Type::Bool,
    );
    let refined_param_ty = refined(list_int_ty.clone(), 8, body);

    // Build a function whose body also calls len(items) on a list<int>
    // parameter, so we can confirm the extern is shared.
    let body_block = block(
        vec![let_atom(
            20,
            Type::Int,
            PExpr {
                kind: PExprKind::Len(var(2)),
                ty: Type::Int,
                span: span(),
            },
        )],
        None,
        Type::Tuple(vec![]),
    );
    let f = func(
        1,
        vec![(2, list_int_ty), (3, refined_param_ty)],
        Type::Tuple(vec![]),
        body_block,
    );

    let mut prog = empty_program(50);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    let lowered = lower_program(prog);
    let c = lowered.program;

    assert!(
        lowered.refinement_errors.is_empty(),
        "unexpected refinement errors: {:?}",
        lowered.refinement_errors
    );

    // Exactly one extern: array_len<int>, shared between the function-body
    // len(items) call and the refinement-body len(xs) call.
    assert_eq!(c.extern_funcs.len(), 1);
    let ext = &c.extern_funcs[0];
    assert_eq!(ext.original_name, "array_len<int>");
    let array_len_id = ext.name;

    // Verify the refinement body's ExternCall targets the same NameId.
    let CType::Refined(_, handle) = &c.funcs[0].params[1].ty else {
        panic!("expected refined type on second param");
    };
    let len_call = match &handle.body.kind {
        RefinementExprKind::BinOp(_, l, _) => l,
        other => panic!("expected BinOp at top of body, got {:?}", other),
    };
    match &len_call.kind {
        RefinementExprKind::ExternCall { target, .. } => {
            assert_eq!(*target, array_len_id);
        }
        other => panic!("expected ExternCall, got {:?}", other),
    }
}

#[test]
fn refinement_user_function_call_records_validation_error() {
    // int { x | helper(x) } -- helper is a user function, not allowed.
    let user_call = TypedExprKind::FuncCall(
        spur_ast::types::TypedFuncCall::User(TypedUserFuncCall {
            name: nid(99),
            original_name: "helper".to_string(),
            args: vec![typed(
                TypedExprKind::Var(nid(7), "x".to_string()),
                Type::Int,
            )],
            return_type: Type::Bool,
            is_free: true,
            span: span(),
        }),
    );
    let body = typed(user_call, Type::Bool);
    let refined_ty = refined(Type::Int, 7, body);

    let lowered = lower_with_refinement(refined_ty);
    let c = lowered.program;

    assert_eq!(lowered.refinement_errors.len(), 1);
    match &lowered.refinement_errors[0].kind {
        RefinementValidationErrorKind::UserFunctionCall(name) => {
            assert_eq!(name, "helper");
        }
        other => panic!("expected UserFunctionCall error, got {:?}", other),
    }

    let CType::Refined(_, handle) = &c.funcs[0].params[0].ty else {
        panic!("expected refined type");
    };
    assert!(matches!(handle.body.kind, RefinementExprKind::Error));
}

#[test]
fn refinement_handle_dedup_lowers_body_once() {
    // Construct one RefinementHandle and reuse it in two type positions
    // (the param and the function's return type). The body should only
    // be lowered once thanks to body_memo, so we get exactly one extern.
    let list_int_ty = Type::List(Box::new(Type::Int));
    let body = typed(
        TypedExprKind::BinOp(
            BinOp::Equal,
            Box::new(typed(
                TypedExprKind::Len(Box::new(typed(
                    TypedExprKind::Var(nid(8), "xs".to_string()),
                    list_int_ty.clone(),
                ))),
                Type::Int,
            )),
            Box::new(typed(TypedExprKind::IntLit(1), Type::Int)),
        ),
        Type::Bool,
    );
    let handle = RefinementHandle::new(RefinementBody {
        bound: nid(8),
        original_bound: "xs".to_string(),
        body,
    });
    let refined_ty = Type::Refined(Box::new(list_int_ty), handle);

    let f = func(
        1,
        vec![(2, refined_ty.clone())],
        refined_ty,
        block(vec![], None, Type::Tuple(vec![])),
    );
    let mut prog = empty_program(50);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    let lowered = lower_program(prog);
    let c = lowered.program;

    assert!(lowered.refinement_errors.is_empty());
    // One extern: array_len<int>. Equality on int is now a CBinOp, so no
    // eq<int> extern is interned.
    assert_eq!(c.extern_funcs.len(), 1);
    assert_eq!(c.extern_funcs[0].original_name, "array_len<int>");

    // Both the param's and return type's refinement handles must be the
    // *same* CRefinementHandle (Arc identity) thanks to body_memo.
    let CType::Refined(_, p_handle) = &c.funcs[0].params[0].ty else {
        panic!("expected refined param");
    };
    let CType::Refined(_, r_handle) = &c.funcs[0].return_type else {
        panic!("expected refined return");
    };
    assert!(std::ptr::eq(p_handle.as_ptr(), r_handle.as_ptr()));

    // The top of the refinement body is `len(xs) == 1`, lowered to a
    // BinOp(IntEq, ExternCall(array_len<int>, [xs]), IntLit(1)).
    match &p_handle.body.kind {
        RefinementExprKind::BinOp(CBinOp::IntEq, l, r) => {
            assert!(matches!(l.kind, RefinementExprKind::ExternCall { .. }));
            assert!(matches!(r.kind, RefinementExprKind::IntLit(1)));
        }
        other => panic!("expected BinOp(IntEq, ..), got {:?}", other),
    }
}

#[test]
fn refinement_non_linear_arithmetic_is_rejected_end_to_end() {
    // int { x | x * x > 0 }
    // The lowerer will faithfully produce a BinOp(Multiply, x, x) in the
    // refinement body, then the post-lowering validation pass should
    // surface a NonLinearArithmetic error.
    let body = typed(
        TypedExprKind::BinOp(
            BinOp::Greater,
            Box::new(typed(
                TypedExprKind::BinOp(
                    BinOp::Multiply,
                    Box::new(typed(
                        TypedExprKind::Var(nid(7), "x".to_string()),
                        Type::Int,
                    )),
                    Box::new(typed(
                        TypedExprKind::Var(nid(7), "x".to_string()),
                        Type::Int,
                    )),
                ),
                Type::Int,
            )),
            Box::new(typed(TypedExprKind::IntLit(0), Type::Int)),
        ),
        Type::Bool,
    );
    let refined_ty = refined(Type::Int, 7, body);

    let lowered = lower_with_refinement(refined_ty);

    assert_eq!(lowered.refinement_errors.len(), 1);
    assert!(matches!(
        lowered.refinement_errors[0].kind,
        RefinementValidationErrorKind::NonLinearArithmetic {
            op: CBinOp::Multiply
        }
    ));
}

#[test]
fn refinement_linear_multiplication_is_accepted_end_to_end() {
    // int { x | x * 2 > 0 } — at least one operand is constant.
    let body = typed(
        TypedExprKind::BinOp(
            BinOp::Greater,
            Box::new(typed(
                TypedExprKind::BinOp(
                    BinOp::Multiply,
                    Box::new(typed(
                        TypedExprKind::Var(nid(7), "x".to_string()),
                        Type::Int,
                    )),
                    Box::new(typed(TypedExprKind::IntLit(2), Type::Int)),
                ),
                Type::Int,
            )),
            Box::new(typed(TypedExprKind::IntLit(0), Type::Int)),
        ),
        Type::Bool,
    );
    let refined_ty = refined(Type::Int, 7, body);

    let lowered = lower_with_refinement(refined_ty);
    assert!(
        lowered.refinement_errors.is_empty(),
        "unexpected: {:?}",
        lowered.refinement_errors
    );
}

#[test]
fn role_functions_get_flattened_with_role_metadata() {
    let f1 = func(10, vec![], Type::Tuple(vec![]), block(vec![], None, Type::Tuple(vec![])));
    let f2 = func(11, vec![], Type::Tuple(vec![]), block(vec![], None, Type::Tuple(vec![])));
    let role = PRoleDef {
        name: nid(99),
        original_name: "Node".to_string(),
        func_defs: vec![f1, f2],
        span: span(),
    };
    let free = func(
        12,
        vec![],
        Type::Tuple(vec![]),
        block(vec![], None, Type::Tuple(vec![])),
    );

    let mut prog = empty_program(100);
    prog.top_level_defs.push(PTopLevelDef::Role(role));
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(free));

    let c = lower_program(prog).program;
    assert_eq!(c.funcs.len(), 3);
    assert_eq!(c.funcs[0].role, Some(nid(99)));
    assert_eq!(c.funcs[1].role, Some(nid(99)));
    assert_eq!(c.funcs[2].role, None);
}

/// Build a single-function program whose body is `let _: bool = a == b;`
/// (or `!=` if `not_equal`) where `a` and `b` are parameters of `ty`.
fn lower_eq_program(ty: Type, not_equal: bool) -> CProgram {
    let op = if not_equal {
        BinOp::NotEqual
    } else {
        BinOp::Equal
    };
    let body = block(
        vec![let_atom(
            10,
            Type::Bool,
            PExpr {
                kind: PExprKind::BinOp(op, var(1), var(2)),
                ty: Type::Bool,
                span: span(),
            },
        )],
        None,
        Type::Tuple(vec![]),
    );
    let f = func(
        5,
        vec![(1, ty.clone()), (2, ty)],
        Type::Tuple(vec![]),
        body,
    );
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    lower_program(prog).program
}

fn first_let_value_kind(c: &CProgram) -> &CExprKind {
    let stmt = &c.funcs[0].body.statements[0];
    match &stmt.kind {
        CStatementKind::LetAtom(la) => &la.value.kind,
        other => panic!("expected LetAtom, got {:?}", other),
    }
}

#[test]
fn eq_on_int_lowers_to_binop_no_extern() {
    let c = lower_eq_program(Type::Int, false);
    assert!(c.extern_funcs.is_empty(), "expected no externs, got {:?}", c.extern_funcs);
    match first_let_value_kind(&c) {
        CExprKind::BinOp(CBinOp::IntEq, _, _) => {}
        other => panic!("expected BinOp(IntEq, ..), got {:?}", other),
    }
}

#[test]
fn neq_on_int_lowers_to_binop_no_extern() {
    let c = lower_eq_program(Type::Int, true);
    assert!(c.extern_funcs.is_empty());
    match first_let_value_kind(&c) {
        CExprKind::BinOp(CBinOp::IntNeq, _, _) => {}
        other => panic!("expected BinOp(IntNeq, ..), got {:?}", other),
    }
}

#[test]
fn eq_on_bool_lowers_to_binop_no_extern() {
    let c = lower_eq_program(Type::Bool, false);
    assert!(c.extern_funcs.is_empty());
    match first_let_value_kind(&c) {
        CExprKind::BinOp(CBinOp::BoolEq, _, _) => {}
        other => panic!("expected BinOp(BoolEq, ..), got {:?}", other),
    }
}

#[test]
fn eq_on_string_still_lowers_to_extern() {
    let c = lower_eq_program(Type::String, false);
    assert_eq!(c.extern_funcs.len(), 1);
    let ext = &c.extern_funcs[0];
    assert_eq!(ext.original_name, "eq<string>");
    let expected_target = ext.name;
    match first_let_value_kind(&c) {
        CExprKind::FuncCall(call) => {
            assert_eq!(call.target, expected_target);
            assert_eq!(call.return_type, CType::Bool);
        }
        other => panic!("expected FuncCall, got {:?}", other),
    }
}

/// Recursively unwrap any number of outer `Refined` wrappers and return
/// the underlying type. Tests use this to peer past the baked-in
/// `len_geq_0` invariant and any per-call append-return refinement.
fn strip_refined(ty: &CType) -> &CType {
    let mut cur = ty;
    while let CType::Refined(inner, _) = cur {
        cur = inner;
    }
    cur
}

#[test]
fn empty_list_lit_lowers_to_array_empty() {
    // let xs: list<int> = [];
    let list_int_ty = Type::List(Box::new(Type::Int));
    let body = block(
        vec![let_atom(
            10,
            list_int_ty.clone(),
            PExpr {
                kind: PExprKind::ListLit(vec![]),
                ty: list_int_ty.clone(),
                span: span(),
            },
        )],
        None,
        Type::Tuple(vec![]),
    );
    let f = func(5, vec![], Type::Tuple(vec![]), body);
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    let c = lower_program(prog).program;

    // Expected externs: array_len<int> (from invariant) +
    // array_empty<int> (from the literal).
    let empty_ext = find_extern(&c, "array_empty<int>");
    assert!(empty_ext.params.is_empty());
    // Return type is the layered shape Refined(Refined(Array(Int),
    // len_geq_0), len_eq_0). Strip both refines down to the raw array.
    assert_eq!(
        strip_refined(&empty_ext.return_type),
        &CType::Array(Box::new(CType::Int))
    );
    // Both Refined layers exist.
    let outer = match &empty_ext.return_type {
        CType::Refined(inner, _) => inner,
        other => panic!("expected Refined return type, got {:?}", other),
    };
    assert!(matches!(**outer, CType::Refined(_, _)));

    // Statements: `let _t_empty = array_empty<int>();` then
    // `let xs = Atomic(Var(_t_empty));`.
    let stmts = &c.funcs[0].body.statements;
    assert_eq!(stmts.len(), 2);
    match &stmts[0].kind {
        CStatementKind::LetAtom(la) => match &la.value.kind {
            CExprKind::FuncCall(call) => {
                assert_eq!(call.target, empty_ext.name);
                assert!(call.args.is_empty());
            }
            other => panic!("stmt[0] not array_empty FuncCall, got {:?}", other),
        },
        other => panic!("stmt[0] not LetAtom, got {:?}", other),
    }
    match &stmts[1].kind {
        CStatementKind::LetAtom(la) => {
            assert_eq!(la.name, NameId(10));
            assert!(matches!(la.value.kind, CExprKind::Atomic(CAtomic::Var(_, _))));
        }
        other => panic!("stmt[1] not LetAtom, got {:?}", other),
    }
}

#[test]
fn non_empty_list_lit_lowers_to_append_chain() {
    // let xs: list<int> = [1, 2, 3];
    let list_int_ty = Type::List(Box::new(Type::Int));
    let body = block(
        vec![let_atom(
            10,
            list_int_ty.clone(),
            PExpr {
                kind: PExprKind::ListLit(vec![
                    PAtomic::IntLit(1),
                    PAtomic::IntLit(2),
                    PAtomic::IntLit(3),
                ]),
                ty: list_int_ty.clone(),
                span: span(),
            },
        )],
        None,
        Type::Tuple(vec![]),
    );
    let f = func(5, vec![], Type::Tuple(vec![]), body);
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    let c = lower_program(prog).program;

    let empty_ext = find_extern(&c, "array_empty<int>");
    let append_ext = find_extern(&c, "array_append<int>");

    // Five statements: empty, three intermediate appends, then the user's
    // outer let bound to an Atomic ref.
    let stmts = &c.funcs[0].body.statements;
    assert_eq!(stmts.len(), 5, "got: {:#?}", stmts);

    // stmts[0] = empty()
    match &stmts[0].kind {
        CStatementKind::LetAtom(la) => match &la.value.kind {
            CExprKind::FuncCall(call) => {
                assert_eq!(call.target, empty_ext.name);
                assert!(call.args.is_empty());
            }
            other => panic!("stmt[0] not FuncCall, got {:?}", other),
        },
        other => panic!("stmt[0] not LetAtom, got {:?}", other),
    }

    // stmts[1..=3] = three append calls, all targeting the same extern.
    for (i, expected_arg) in [1, 2, 3].iter().enumerate() {
        let stmt = &stmts[i + 1];
        match &stmt.kind {
            CStatementKind::LetAtom(la) => match &la.value.kind {
                CExprKind::FuncCall(call) => {
                    assert_eq!(call.target, append_ext.name);
                    assert_eq!(call.args.len(), 2);
                    assert!(matches!(call.args[0], CAtomic::Var(_, _)));
                    assert_eq!(call.args[1], CAtomic::IntLit(*expected_arg));
                }
                other => panic!("append[{}] not FuncCall, got {:?}", i, other),
            },
            other => panic!("append[{}] not LetAtom, got {:?}", i, other),
        }
    }

    // stmts[4] = the original `let xs = Atomic(Var(...))`.
    match &stmts[4].kind {
        CStatementKind::LetAtom(la) => {
            assert_eq!(la.name, NameId(10));
            assert!(matches!(la.value.kind, CExprKind::Atomic(CAtomic::Var(_, _))));
        }
        other => panic!("stmt[4] not LetAtom, got {:?}", other),
    }
}

#[test]
fn list_type_carries_len_geq_zero_invariant() {
    // Function with one `list<int>` parameter; the lowered param type
    // must be Refined(Array(Int), { _xs | array_len(_xs) >= 0 }).
    let list_int_ty = Type::List(Box::new(Type::Int));
    let f = func(
        5,
        vec![(1, list_int_ty)],
        Type::Tuple(vec![]),
        block(vec![], None, Type::Tuple(vec![])),
    );
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    let c = lower_program(prog).program;

    let CType::Refined(inner, handle) = &c.funcs[0].params[0].ty else {
        panic!("expected Refined param, got {:?}", c.funcs[0].params[0].ty);
    };
    assert_eq!(**inner, CType::Array(Box::new(CType::Int)));

    // Body shape: BinOp(GreaterEqual, ExternCall(array_len, [_xs]),
    // IntLit(0)).
    let array_len_ext = find_extern(&c, "array_len<int>");
    match &handle.body.kind {
        RefinementExprKind::BinOp(CBinOp::GreaterEqual, l, r) => {
            match &l.kind {
                RefinementExprKind::ExternCall { target, args, .. } => {
                    assert_eq!(*target, array_len_ext.name);
                    assert_eq!(args.len(), 1);
                    match &args[0].kind {
                        RefinementExprKind::Var(id, _) => {
                            assert_eq!(*id, handle.bound);
                        }
                        other => panic!("expected Var bound, got {:?}", other),
                    }
                }
                other => panic!("expected ExternCall on lhs, got {:?}", other),
            }
            assert!(matches!(r.kind, RefinementExprKind::IntLit(0)));
        }
        other => panic!("expected BinOp(GreaterEqual, ..), got {:?}", other),
    }
}

#[test]
fn list_invariant_handle_dedups_by_element_type() {
    // Two parameters of type list<int> share the same invariant handle
    // (Arc-pointer-equal).
    let list_int_ty = Type::List(Box::new(Type::Int));
    let f = func(
        5,
        vec![(1, list_int_ty.clone()), (2, list_int_ty)],
        Type::Tuple(vec![]),
        block(vec![], None, Type::Tuple(vec![])),
    );
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    let c = lower_program(prog).program;

    let CType::Refined(_, h1) = &c.funcs[0].params[0].ty else {
        panic!("expected Refined param 0");
    };
    let CType::Refined(_, h2) = &c.funcs[0].params[1].ty else {
        panic!("expected Refined param 1");
    };
    assert!(
        std::ptr::eq(h1.as_ptr(), h2.as_ptr()),
        "list<int> invariant should dedup across parameter slots"
    );
}

#[test]
fn array_append_return_refers_to_param_nameid() {
    // After an Append, the array_append<int> extern's outer-refinement
    // body should reference the extern's first parameter NameId via
    // RefinementExprKind::Var.
    let list_int_ty = Type::List(Box::new(Type::Int));
    let body = block(
        vec![let_atom(
            10,
            list_int_ty.clone(),
            PExpr {
                kind: PExprKind::Append(var(1), var(2)),
                ty: list_int_ty.clone(),
                span: span(),
            },
        )],
        None,
        Type::Tuple(vec![]),
    );
    let f = func(
        5,
        vec![(1, list_int_ty), (2, Type::Int)],
        Type::Tuple(vec![]),
        body,
    );
    let mut prog = empty_program(20);
    prog.top_level_defs.push(PTopLevelDef::FreeFunc(f));
    let c = lower_program(prog).program;

    let append_ext = find_extern(&c, "array_append<int>");
    // Return type should be Refined(Refined(Array, len_geq_0), { ys |
    // array_len(ys) == array_len(xs) + 1 }) — peel the outer to inspect
    // the append-return body.
    let outer_handle = match &append_ext.return_type {
        CType::Refined(_, h) => h,
        other => panic!("expected Refined return, got {:?}", other),
    };
    let xs_param_id = append_ext.params[0].name;

    // Body: BinOp(IntEq, ExternCall(array_len, [ys]),
    //                    BinOp(Add, ExternCall(array_len, [xs]), IntLit(1))).
    let mut found_xs_ref = false;
    fn visit(e: &RefinementExpr, target_id: NameId, found: &mut bool) {
        match &e.kind {
            RefinementExprKind::Var(id, _) => {
                if *id == target_id {
                    *found = true;
                }
            }
            RefinementExprKind::BinOp(_, l, r) => {
                visit(l, target_id, found);
                visit(r, target_id, found);
            }
            RefinementExprKind::ExternCall { args, .. } => {
                for a in args {
                    visit(a, target_id, found);
                }
            }
            _ => {}
        }
    }
    visit(&outer_handle.body, xs_param_id, &mut found_xs_ref);
    assert!(
        found_xs_ref,
        "append return refinement body must reference params[0].name (xs)"
    );
}

#[test]
fn refinement_side_list_lit_uses_externs() {
    // Refinement body shape: list<int> { v | v == [1, 2] } (well-typed
    // for the lowering pipeline; the equality op desugar matters less
    // than the ListLit on the rhs.)
    let list_int_ty = Type::List(Box::new(Type::Int));
    let lit_expr = typed(
        TypedExprKind::ListLit(vec![
            typed(TypedExprKind::IntLit(1), Type::Int),
            typed(TypedExprKind::IntLit(2), Type::Int),
        ]),
        list_int_ty.clone(),
    );
    let body = typed(
        TypedExprKind::BinOp(
            BinOp::Equal,
            Box::new(typed(
                TypedExprKind::Var(nid(7), "v".to_string()),
                list_int_ty.clone(),
            )),
            Box::new(lit_expr),
        ),
        Type::Bool,
    );
    let refined_ty = refined(list_int_ty, 7, body);

    let lowered = lower_with_refinement(refined_ty);
    let c = lowered.program;

    assert!(
        lowered.refinement_errors.is_empty(),
        "unexpected: {:?}",
        lowered.refinement_errors
    );
    let empty_ext = find_extern(&c, "array_empty<int>");
    let append_ext = find_extern(&c, "array_append<int>");

    // Find the user refinement body. It's the OUTER refinement layer on
    // the param's type (the inner is the baked-in invariant).
    let CType::Refined(_, user_handle) = &c.funcs[0].params[0].ty else {
        panic!("expected refined param");
    };
    // Top of body is BinOp(_, Var(v), <ListLit_lowered>). Walk to the rhs
    // and confirm it is a nested ExternCall(append, [append(empty, 1), 2]).
    let rhs = match &user_handle.body.kind {
        RefinementExprKind::ExternCall { args, .. } => &args[1],
        RefinementExprKind::BinOp(_, _, r) => r.as_ref(),
        other => panic!("expected BinOp/ExternCall at top, got {:?}", other),
    };
    // Outer append (with item = 2)
    let (outer_args,) = match &rhs.kind {
        RefinementExprKind::ExternCall { target, args, .. } => {
            assert_eq!(*target, append_ext.name);
            assert_eq!(args.len(), 2);
            (args,)
        }
        other => panic!("expected outer ExternCall(append), got {:?}", other),
    };
    assert!(matches!(outer_args[1].kind, RefinementExprKind::IntLit(2)));
    // Inner append (with item = 1)
    let (inner_args,) = match &outer_args[0].kind {
        RefinementExprKind::ExternCall { target, args, .. } => {
            assert_eq!(*target, append_ext.name);
            assert_eq!(args.len(), 2);
            (args,)
        }
        other => panic!("expected inner ExternCall(append), got {:?}", other),
    };
    assert!(matches!(inner_args[1].kind, RefinementExprKind::IntLit(1)));
    // Innermost: array_empty
    match &inner_args[0].kind {
        RefinementExprKind::ExternCall { target, args, .. } => {
            assert_eq!(*target, empty_ext.name);
            assert!(args.is_empty());
        }
        other => panic!("expected innermost array_empty, got {:?}", other),
    }
}

#[test]
fn refinement_side_empty_list_lit_uses_array_empty() {
    // list<int> { v | v == [] } — the surrounding type is list<int>, so
    // `[]` should pick up the element type via extract_array_elem on the
    // expression's lowered expected type.
    let list_int_ty = Type::List(Box::new(Type::Int));
    let lit_expr = typed(TypedExprKind::ListLit(vec![]), list_int_ty.clone());
    let body = typed(
        TypedExprKind::BinOp(
            BinOp::Equal,
            Box::new(typed(
                TypedExprKind::Var(nid(7), "v".to_string()),
                list_int_ty.clone(),
            )),
            Box::new(lit_expr),
        ),
        Type::Bool,
    );
    let refined_ty = refined(list_int_ty, 7, body);

    let lowered = lower_with_refinement(refined_ty);
    let c = lowered.program;

    assert!(
        lowered.refinement_errors.is_empty(),
        "unexpected: {:?}",
        lowered.refinement_errors
    );
    let empty_ext = find_extern(&c, "array_empty<int>");

    let CType::Refined(_, user_handle) = &c.funcs[0].params[0].ty else {
        panic!("expected refined param");
    };
    let rhs = match &user_handle.body.kind {
        RefinementExprKind::ExternCall { args, .. } => &args[1],
        RefinementExprKind::BinOp(_, _, r) => r.as_ref(),
        other => panic!("unexpected body shape: {:?}", other),
    };
    match &rhs.kind {
        RefinementExprKind::ExternCall { target, args, .. } => {
            assert_eq!(*target, empty_ext.name);
            assert!(args.is_empty());
        }
        other => panic!("expected array_empty ExternCall, got {:?}", other),
    }
}
