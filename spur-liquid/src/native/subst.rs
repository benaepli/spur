use spur_ast::name::NameId;

use crate::ir::CType;
use crate::refinement::{RefinementExpr, RefinementExprKind};

pub fn subst_refexpr(new_expr: &RefinementExpr, old_id: NameId, expr: &RefinementExpr) -> RefinementExpr {
    let kind = match &expr.kind {
        RefinementExprKind::Var(id, name) => {
            if *id == old_id {
                return new_expr.clone();
            }
            RefinementExprKind::Var(*id, name.clone())
        }
        RefinementExprKind::BinOp(op, lhs, rhs) => RefinementExprKind::BinOp(
            *op,
            Box::new(subst_refexpr(new_expr, old_id, lhs)),
            Box::new(subst_refexpr(new_expr, old_id, rhs)),
        ),
        RefinementExprKind::Not(inner) => {
            RefinementExprKind::Not(Box::new(subst_refexpr(new_expr, old_id, inner)))
        }
        RefinementExprKind::Negate(inner) => {
            RefinementExprKind::Negate(Box::new(subst_refexpr(new_expr, old_id, inner)))
        }
        RefinementExprKind::ExternCall {
            target,
            args,
            return_type,
        } => RefinementExprKind::ExternCall {
            target: *target,
            args: args.iter().map(|a| subst_refexpr(new_expr, old_id, a)).collect(),
            return_type: return_type.clone(),
        },
        RefinementExprKind::IsVariant(scrutinee, eid, vid) => RefinementExprKind::IsVariant(
            Box::new(subst_refexpr(new_expr, old_id, scrutinee)),
            *eid,
            *vid,
        ),
        RefinementExprKind::TupleAccess(inner, idx) => {
            RefinementExprKind::TupleAccess(Box::new(subst_refexpr(new_expr, old_id, inner)), *idx)
        }
        RefinementExprKind::FieldAccess(inner, field) => {
            RefinementExprKind::FieldAccess(Box::new(subst_refexpr(new_expr, old_id, inner)), *field)
        }
        _ => return expr.clone(),
    };
    RefinementExpr {
        kind,
        ty: expr.ty.clone(),
        span: expr.span,
    }
}

pub fn subst_type(new_expr: &RefinementExpr, old_id: NameId, ty: &CType) -> CType {
    match ty {
        CType::Refined(base, handle) => {
            if handle.bound == old_id {
                return ty.clone();
            }
            let new_body = subst_refexpr(new_expr, old_id, &handle.body);
            let new_base = Box::new(subst_type(new_expr, old_id, base));
            CType::Refined(
                new_base,
                crate::ir::CRefinementHandle::new(crate::ir::CRefinementBody {
                    bound: handle.bound,
                    original_bound: handle.original_bound.clone(),
                    body: new_body,
                }),
            )
        }
        CType::Array(inner) => CType::Array(Box::new(subst_type(new_expr, old_id, inner))),
        CType::Tuple(ts) => {
            CType::Tuple(ts.iter().map(|t| subst_type(new_expr, old_id, t)).collect())
        }
        CType::Map(k, v) => CType::Map(
            Box::new(subst_type(new_expr, old_id, k)),
            Box::new(subst_type(new_expr, old_id, v)),
        ),
        CType::Optional(inner) => CType::Optional(Box::new(subst_type(new_expr, old_id, inner))),
        _ => ty.clone(),
    }
}
