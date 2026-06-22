//! Early-return desugar.
//!
//! Spec patterns like
//!
//! ```text
//! if (cond) { return; }
//! rest...
//! ```
//!
//! historically don't get path-sensitive treatment in the Formulog rules:
//! `s_expr(Conditional)` falls back to `synth_expr`, which doesn't propagate
//! `!cond` into the env that flows into `rest`. Indexing preconditions on
//! `rest` then fail to discharge even when the guard makes them safe.
//!
//! This pass rewrites every diverging guard at the block level into the
//! structured form
//!
//! ```text
//! let __r: T = if (cond) { return; } else { rest... };
//! __r
//! ```
//!
//! Annotating the let with the surrounding block's type forces the
//! conditional through `check_cond`, which already refines the env with
//! `!cond` when it descends into the (now-non-empty) else.
//!
//! When the original conditional had a non-diverging else, we leave the
//! block alone — the partial-divergence case is not worth the duplication
//! it would introduce, and the existing rules handle the joined env.

use spur_ast::name::NameId;
use std::collections::HashMap;

use crate::ir::{
    CAtomic, CBlock, CExpr, CExprKind, CLetAtom, CProgram, CStatement, CStatementKind,
};

pub fn desugar_early_returns(program: &mut CProgram) {
    let mut fresh = FreshNames {
        next: program.next_name_id,
    };
    let mut id_to_name = std::mem::take(&mut program.id_to_name);
    for func in &mut program.funcs {
        rewrite_block(&mut func.body, &mut fresh, &mut id_to_name);
    }
    program.next_name_id = fresh.next;
    program.id_to_name = id_to_name;
}

struct FreshNames {
    next: usize,
}

impl FreshNames {
    fn fresh(&mut self) -> NameId {
        let id = NameId(self.next);
        self.next += 1;
        id
    }
}

/// True when every path through `b` exits before reaching the tail —
/// that is, the value-producing tail is unreachable.
fn block_diverges(b: &CBlock) -> bool {
    if b
        .statements
        .iter()
        .any(|s| matches!(s.kind, CStatementKind::Return(_)))
    {
        return true;
    }
    if matches!(b.tail_expr, Some(CAtomic::Never)) {
        return true;
    }
    if let Some(last) = b.statements.last() {
        if let CStatementKind::Expr(e) = &last.kind {
            if let CExprKind::Conditional(c) = &e.kind {
                let if_div = block_diverges(&c.if_branch.body);
                let elseifs_div = c.elseif_branches.iter().all(|br| block_diverges(&br.body));
                let else_div = c
                    .else_branch
                    .as_ref()
                    .map_or(false, |eb| block_diverges(eb));
                if if_div && elseifs_div && else_div {
                    return true;
                }
            }
        }
    }
    false
}

fn rewrite_block(
    b: &mut CBlock,
    fresh: &mut FreshNames,
    id_to_name: &mut HashMap<NameId, String>,
) {
    for stmt in &mut b.statements {
        rewrite_stmt_children(stmt, fresh, id_to_name);
    }

    let Some(idx) = find_guard_idx(b) else {
        return;
    };

    let pre_len = idx + 1;
    let nothing_to_move = pre_len >= b.statements.len() && b.tail_expr.is_none();
    if nothing_to_move {
        return;
    }

    let rest_stmts: Vec<CStatement> = b.statements.drain(pre_len..).collect();
    let rest_tail = b.tail_expr.take();
    let block_ty = b.ty.clone();
    let block_span = b.span;

    let cond_stmt = b.statements.pop().unwrap();
    let CStatement {
        kind: cond_kind,
        span: cond_span,
    } = cond_stmt;
    let CStatementKind::Expr(mut cond_expr) = cond_kind else {
        unreachable!("find_guard_idx returned a non-Expr index");
    };

    {
        let CExprKind::Conditional(cond) = &mut cond_expr.kind else {
            unreachable!("find_guard_idx returned a non-Conditional Expr");
        };

        if let Some(else_block) = cond.else_branch.as_mut() {
            else_block.statements.extend(rest_stmts);
            else_block.tail_expr = rest_tail;
            else_block.ty = block_ty.clone();
        } else {
            cond.else_branch = Some(CBlock {
                statements: rest_stmts,
                tail_expr: rest_tail,
                ty: block_ty.clone(),
                span: block_span,
            });
        }

        if let Some(else_block) = cond.else_branch.as_mut() {
            rewrite_block(else_block, fresh, id_to_name);
        }
    }

    cond_expr.ty = block_ty.clone();

    let new_name = fresh.fresh();
    let new_name_str = format!("__guard_rest_{}", new_name.0);
    id_to_name.insert(new_name, new_name_str.clone());

    b.statements.push(CStatement {
        kind: CStatementKind::LetAtom(CLetAtom {
            name: new_name,
            original_name: new_name_str.clone(),
            ty: block_ty,
            value: cond_expr,
            user_annotated: true,
            span: cond_span,
        }),
        span: cond_span,
    });
    b.tail_expr = Some(CAtomic::Var(new_name, new_name_str));
}

fn find_guard_idx(b: &CBlock) -> Option<usize> {
    for (idx, stmt) in b.statements.iter().enumerate() {
        if let CStatementKind::Expr(e) = &stmt.kind {
            if let CExprKind::Conditional(c) = &e.kind {
                let if_div = block_diverges(&c.if_branch.body);
                let elseifs_div = c.elseif_branches.iter().all(|br| block_diverges(&br.body));
                // We want to fold `rest` into the conditional's else only
                // when there's a non-diverging path through the conditional.
                // If the else also diverges, the conditional itself is Never
                // and `rest` is unreachable — leave it alone.
                let rest_reachable_via_else = match &c.else_branch {
                    None => true,
                    Some(eb) => !block_diverges(eb),
                };
                if if_div && elseifs_div && rest_reachable_via_else {
                    return Some(idx);
                }
            }
        }
    }
    None
}

fn rewrite_stmt_children(
    stmt: &mut CStatement,
    fresh: &mut FreshNames,
    id_to_name: &mut HashMap<NameId, String>,
) {
    match &mut stmt.kind {
        CStatementKind::LetAtom(la) => rewrite_expr_children(&mut la.value, fresh, id_to_name),
        CStatementKind::Expr(e) => rewrite_expr_children(e, fresh, id_to_name),
        CStatementKind::Return(_) | CStatementKind::Error => {}
    }
}

fn rewrite_expr_children(
    e: &mut CExpr,
    fresh: &mut FreshNames,
    id_to_name: &mut HashMap<NameId, String>,
) {
    match &mut e.kind {
        CExprKind::Block(b) => rewrite_block(b, fresh, id_to_name),
        CExprKind::Conditional(c) => {
            rewrite_block(&mut c.if_branch.body, fresh, id_to_name);
            for branch in &mut c.elseif_branches {
                rewrite_block(&mut branch.body, fresh, id_to_name);
            }
            if let Some(else_block) = c.else_branch.as_mut() {
                rewrite_block(else_block, fresh, id_to_name);
            }
        }
        _ => {}
    }
}
