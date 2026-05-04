use crate::ir::*;

use super::context::{Counter, Env, GlobalCtx};
use super::smt::SmtSolver;
use super::subtype::subtype;
use super::synth::{refine_env_false, refine_env_true, synth_expr};
use super::CheckResult;

pub fn check_expr(
    env: &Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    expr: &CExpr,
    ctr: &mut Counter,
    expected: &CType,
) -> CheckResult {
    match &expr.kind {
        CExprKind::Block(b) => check_block(env, ctx, solver, f_ret, b, ctr, expected),
        CExprKind::Conditional(c) => check_cond(env, ctx, solver, f_ret, c, ctr, expected),
        _ => {
            let Some(synth_ty) = synth_expr(env, ctx, solver, f_ret, expr, ctr) else {
                return CheckResult::Fail(expr.span);
            };
            if subtype(env, solver, &synth_ty, expected) {
                CheckResult::Ok
            } else {
                CheckResult::Fail(expr.span)
            }
        }
    }
}

pub fn check_block(
    env: &Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    block: &CBlock,
    ctr: &mut Counter,
    expected: &CType,
) -> CheckResult {
    let mut cur_env = env.clone();

    for stmt in &block.statements {
        match &stmt.kind {
            CStatementKind::LetAtom(let_atom) => {
                if let_atom.user_annotated {
                    let result =
                        check_expr(&cur_env, ctx, solver, f_ret, &let_atom.value, ctr, &let_atom.ty);
                    if result != CheckResult::Ok {
                        return result;
                    }
                    cur_env.insert(0, (let_atom.name, let_atom.ty.clone()));
                } else if matches!(let_atom.value.kind, CExprKind::Conditional(_)) {
                    // For conditionals bound to a variable, check against
                    // the expected type so branch-specific refinements are
                    // verified inside check_cond where the env is refined.
                    let result =
                        check_expr(&cur_env, ctx, solver, f_ret, &let_atom.value, ctr, expected);
                    if result != CheckResult::Ok {
                        return result;
                    }
                    cur_env.insert(0, (let_atom.name, expected.clone()));
                } else {
                    let Some(ty) =
                        synth_expr(&cur_env, ctx, solver, f_ret, &let_atom.value, ctr)
                    else {
                        return CheckResult::Fail(let_atom.span);
                    };
                    cur_env.insert(0, (let_atom.name, ty));
                }
            }
            CStatementKind::Expr(e) => {
                // When a conditional's true branch diverges (returns),
                // subsequent statements can assume the condition is false.
                if let CExprKind::Conditional(c) = &e.kind {
                    let (result, diverges) = check_cond_and_refine(
                        &mut cur_env, ctx, solver, f_ret, c, ctr, expected,
                    );
                    if result != CheckResult::Ok {
                        return result;
                    }
                    if diverges {
                        return CheckResult::Ok;
                    }
                } else {
                    let Some(ty) = synth_expr(&cur_env, ctx, solver, f_ret, e, ctr) else {
                        return CheckResult::Fail(e.span);
                    };
                    if matches!(ty, CType::Never) {
                        return CheckResult::Ok;
                    }
                }
            }
            CStatementKind::Return(a) => {
                let e = CExpr {
                    kind: CExprKind::Atomic(a.clone()),
                    ty: CType::Nil,
                    span: stmt.span,
                };
                return check_expr(&cur_env, ctx, solver, f_ret, &e, ctr, f_ret);
            }
            CStatementKind::Error => return CheckResult::Fail(stmt.span),
        }
    }

    if let Some(tail) = &block.tail_expr {
        let e = CExpr {
            kind: CExprKind::Atomic(tail.clone()),
            ty: CType::Nil,
            span: block.span,
        };
        check_expr(&cur_env, ctx, solver, f_ret, &e, ctr, expected)
    } else {
        CheckResult::Ok
    }
}

/// Process a conditional as an expression-statement. Checks each branch,
/// and for branches that diverge (return), refines `env` with the negated
/// condition so subsequent statements in the outer block benefit.
/// Returns (result, diverges) where diverges is true if all paths return.
fn check_cond_and_refine(
    env: &mut Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    cond: &CCondExpr,
    ctr: &mut Counter,
    expected: &CType,
) -> (CheckResult, bool) {
    let mut cur_env = env.clone();
    let mut all_diverge = true;

    for branch in std::iter::once(&cond.if_branch).chain(cond.elseif_branches.iter()) {
        let env_true = refine_env_true(&cur_env, &branch.condition, ctr);
        let result = check_block(&env_true, ctx, solver, f_ret, &branch.body, ctr, expected);
        if result != CheckResult::Ok {
            return (result, false);
        }
        if !block_diverges(&branch.body) {
            all_diverge = false;
        }
        cur_env = refine_env_false(&cur_env, &branch.condition, ctr);
    }

    if let Some(else_block) = &cond.else_branch {
        let result = check_block(&cur_env, ctx, solver, f_ret, else_block, ctr, expected);
        if result != CheckResult::Ok {
            return (result, false);
        }
        if !block_diverges(else_block) {
            all_diverge = false;
        }
    } else {
        all_diverge = false;
    }

    *env = cur_env;
    (CheckResult::Ok, all_diverge)
}

fn block_diverges(block: &CBlock) -> bool {
    block.statements.iter().any(|s| matches!(s.kind, CStatementKind::Return(_)))
}

fn check_cond(
    env: &Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    cond: &CCondExpr,
    ctr: &mut Counter,
    expected: &CType,
) -> CheckResult {
    let mut cur_env = env.clone();

    for branch in std::iter::once(&cond.if_branch).chain(cond.elseif_branches.iter()) {
        let env_true = refine_env_true(&cur_env, &branch.condition, ctr);
        let result = check_block(&env_true, ctx, solver, f_ret, &branch.body, ctr, expected);
        if result != CheckResult::Ok {
            return result;
        }
        cur_env = refine_env_false(&cur_env, &branch.condition, ctr);
    }

    if let Some(else_block) = &cond.else_branch {
        check_block(&cur_env, ctx, solver, f_ret, else_block, ctr, expected)
    } else {
        CheckResult::Ok
    }
}
