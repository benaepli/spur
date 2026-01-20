use super::*;

use crate::analysis::resolver::*;
use crate::analysis::types::Type;
use crate::parser::{BinOp, Span};
use pretty_assertions::assert_eq;

fn dummy_span() -> Span {
    Span::default()
}

fn id(i: usize) -> NameId {
    NameId(i)
}

fn setup_checker() -> TypeChecker {
    let pre = PrepopulatedTypes {
        int: id(0),
        string: id(1),
        bool: id(2),
        unit: id(3),
    };
    TypeChecker::new(pre)
}

fn expr(kind: ResolvedExprKind) -> ResolvedExpr {
    ResolvedExpr {
        kind,
        span: dummy_span(),
    }
}

fn int_lit(val: i64) -> ResolvedExpr {
    expr(ResolvedExprKind::IntLit(val))
}

fn str_lit(val: &str) -> ResolvedExpr {
    expr(ResolvedExprKind::StringLit(val.to_string()))
}

fn bool_lit(val: bool) -> ResolvedExpr {
    expr(ResolvedExprKind::BoolLit(val))
}

#[test]
fn test_infer_literals() -> Result<(), TypeError> {
    let checker = setup_checker();

    assert_eq!(checker.infer_expr(int_lit(42))?.ty, Type::Int);
    assert_eq!(checker.infer_expr(str_lit("hello"))?.ty, Type::String);
    assert_eq!(checker.infer_expr(bool_lit(true))?.ty, Type::Bool);

    Ok(())
}

#[test]
fn test_binop_arithmetic_valid() -> Result<(), TypeError> {
    let checker = setup_checker();

    let op = expr(ResolvedExprKind::BinOp(
        BinOp::Add,
        Box::new(int_lit(10)),
        Box::new(int_lit(20)),
    ));

    let typed = checker.infer_expr(op)?;
    assert_eq!(typed.ty, Type::Int);

    Ok(())
}

#[test]
fn test_binop_type_mismatch() {
    let checker = setup_checker();

    let op = expr(ResolvedExprKind::BinOp(
        BinOp::Add,
        Box::new(int_lit(10)),
        Box::new(str_lit("fail")),
    ));

    let err = checker.infer_expr(op).unwrap_err();

    assert!(matches!(
        err,
        TypeError::InvalidBinOp {
            op: BinOp::Add,
            left: Type::Int,
            right: Type::String,
            ..
        }
    ));
}

#[test]
fn test_binop_logic_valid() -> Result<(), TypeError> {
    let checker = setup_checker();

    let op = expr(ResolvedExprKind::BinOp(
        BinOp::And,
        Box::new(bool_lit(true)),
        Box::new(bool_lit(false)),
    ));

    assert_eq!(checker.infer_expr(op)?.ty, Type::Bool);
    Ok(())
}

#[test]
fn test_list_inference_homogeneous() -> Result<(), TypeError> {
    let checker = setup_checker();

    let list = expr(ResolvedExprKind::ListLit(vec![
        int_lit(1),
        int_lit(2),
        int_lit(3),
    ]));

    let typed = checker.infer_expr(list)?;

    assert!(matches!(typed.ty, Type::List(inner) if *inner == Type::Int));

    Ok(())
}

#[test]
fn test_list_inference_heterogeneous_fail() {
    let checker = setup_checker();

    let list = expr(ResolvedExprKind::ListLit(vec![int_lit(1), str_lit("two")]));

    let err = checker.infer_expr(list).unwrap_err();
    assert!(matches!(err, TypeError::Mismatch { .. }));
}

#[test]
fn test_empty_list_inference() -> Result<(), TypeError> {
    let checker = setup_checker();

    let list = expr(ResolvedExprKind::ListLit(vec![]));
    let typed = checker.infer_expr(list)?;

    assert_eq!(typed.ty, Type::EmptyList);
    Ok(())
}

#[test]
fn test_variable_scope_resolution() -> Result<(), TypeError> {
    let mut checker = setup_checker();
    let var_id = id(100);

    checker.enter_scope();
    checker.add_var(var_id, Type::Int);

    let var_expr = expr(ResolvedExprKind::Var(var_id, "x".into()));
    let typed = checker.infer_expr(var_expr)?;

    assert_eq!(typed.ty, Type::Int);

    checker.exit_scope();
    Ok(())
}

#[test]
fn test_undefined_variable() {
    let checker = setup_checker();
    let var_id = id(999);

    let var_expr = expr(ResolvedExprKind::Var(var_id, "unknown".into()));

    let err = checker.infer_expr(var_expr).unwrap_err();
    assert!(matches!(err, TypeError::UndefinedType(_)));
}

#[test]
fn test_map_inference_valid() -> Result<(), TypeError> {
    let checker = setup_checker();

    let map = expr(ResolvedExprKind::MapLit(vec![
        (str_lit("a"), int_lit(1)),
        (str_lit("b"), int_lit(2)),
    ]));

    let typed = checker.infer_expr(map)?;

    assert!(matches!(
        typed.ty,
        Type::Map(k, v) if *k == Type::String && *v == Type::Int
    ));

    Ok(())
}

#[test]
fn test_assignment_type_check() -> Result<(), TypeError> {
    let mut checker = setup_checker();
    let var_id = id(50);

    checker.enter_scope();
    checker.add_var(var_id, Type::Int);

    // Valid Assignment
    let assign_ok = ResolvedAssignment {
        target: expr(ResolvedExprKind::Var(var_id, "x".into())),
        value: int_lit(10),
        span: dummy_span(),
    };
    checker.check_assignment(assign_ok)?;

    // Invalid Assignment
    let assign_bad = ResolvedAssignment {
        target: expr(ResolvedExprKind::Var(var_id, "x".into())),
        value: str_lit("no"),
        span: dummy_span(),
    };

    let err = checker.check_assignment(assign_bad).unwrap_err();
    assert!(matches!(err, TypeError::Mismatch { .. }));

    checker.exit_scope();
    Ok(())
}

#[test]
fn test_function_return_validation() -> Result<(), TypeError> {
    let mut checker = setup_checker();

    checker.enter_scope();
    checker.current_return_type = Some(Type::Int);

    // Valid Return
    let ret_stmt = ResolvedStatement {
        kind: ResolvedStatementKind::Return(int_lit(5)),
        span: dummy_span(),
    };

    let (_, returns) = checker.check_statement(ret_stmt)?;
    assert!(returns, "Statement should indicate it returns");
    // Invalid Return Type
    let bad_ret = ResolvedStatement {
        kind: ResolvedStatementKind::Return(str_lit("bad")),
        span: dummy_span(),
    };

    let err = checker.check_statement(bad_ret).unwrap_err();
    assert!(matches!(err, TypeError::Mismatch { .. }));

    checker.exit_scope();
    Ok(())
}
