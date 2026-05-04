use crate::ir::{CBinOp, CType};
use crate::refinement::{RefinementExpr, RefinementExprKind};

use super::context::Env;
use super::smt::SmtSolver;
use super::subst::subst_refexpr;

pub fn canonicalize(ty: &CType) -> CType {
    match ty {
        CType::Refined(base, outer_handle) => match base.as_ref() {
            CType::Refined(inner_base, inner_handle) => {
                let renamed_inner = subst_refexpr(
                    &RefinementExpr {
                        kind: RefinementExprKind::Var(outer_handle.bound, outer_handle.original_bound.clone()),
                        ty: CType::Int, // placeholder
                        span: spur_ast::span::Span::default(),
                    },
                    inner_handle.bound,
                    &inner_handle.body,
                );
                let combined = RefinementExpr {
                    kind: RefinementExprKind::BinOp(
                        CBinOp::And,
                        Box::new(renamed_inner),
                        Box::new(outer_handle.body.clone()),
                    ),
                    ty: CType::Bool,
                    span: spur_ast::span::Span::default(),
                };
                let new_ty = CType::Refined(
                    inner_base.clone(),
                    crate::ir::CRefinementHandle::new(crate::ir::CRefinementBody {
                        bound: outer_handle.bound,
                        original_bound: outer_handle.original_bound.clone(),
                        body: combined,
                    }),
                );
                canonicalize(&new_ty)
            }
            _ => ty.clone(),
        },
        _ => ty.clone(),
    }
}

pub fn base_type(ty: &CType) -> &CType {
    match ty {
        CType::Refined(base, _) => base_type(base),
        _ => ty,
    }
}

pub fn subtype(env: &Env, solver: &mut SmtSolver, t1: &CType, t2: &CType) -> bool {
    let t1 = canonicalize(t1);
    let t2 = canonicalize(t2);
    subtype_inner(env, solver, &t1, &t2)
}

fn subtype_inner(env: &Env, solver: &mut SmtSolver, t1: &CType, t2: &CType) -> bool {
    match (t1, t2) {
        (CType::Never, _) => true,
        (CType::Int, CType::Int) => true,
        (CType::Bool, CType::Bool) => true,
        (CType::String, CType::String) => true,
        (CType::Nil, CType::Nil) => true,
        (CType::Nil, CType::Optional(_)) => true,
        (CType::Tuple(ts1), CType::Tuple(ts2)) => {
            ts1.len() == ts2.len()
                && ts1
                    .iter()
                    .zip(ts2.iter())
                    .all(|(a, b)| subtype_inner(env, solver, a, b))
        }
        (CType::Array(a), CType::Array(b)) => subtype_inner(env, solver, a, b),
        (CType::Map(k1, v1), CType::Map(k2, v2)) => {
            subtype_inner(env, solver, k1, k2) && subtype_inner(env, solver, v1, v2)
        }
        (CType::Optional(a), CType::Optional(b)) => subtype_inner(env, solver, a, b),
        (CType::Chan(a), CType::Chan(b)) => subtype_inner(env, solver, a, b),
        (CType::FifoLink(a), CType::FifoLink(b)) => subtype_inner(env, solver, a, b),
        (CType::Iter(a), CType::Iter(b)) => subtype_inner(env, solver, a, b),
        (CType::Struct(a), CType::Struct(b)) => a == b,
        (CType::Variant(a), CType::Variant(b)) => a == b,
        (CType::Role(a), CType::Role(b)) => a == b,
        (CType::Refined(base1, h1), CType::Refined(base2, h2)) => {
            if !subtype_inner(env, solver, base1, base2) {
                return false;
            }
            solver.check_implication(env, h1, h2)
        }
        (CType::Refined(base, _), _) => subtype_inner(env, solver, base, t2),
        _ => false,
    }
}
