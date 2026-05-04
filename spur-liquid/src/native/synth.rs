use spur_ast::name::NameId;
use spur_ast::span::Span;

use crate::ir::*;
use crate::refinement::{RefinementExpr, RefinementExprKind};

use super::CheckResult;
use super::check;
use super::context::{Counter, Env, GlobalCtx, env_lookup};
use super::smt::SmtSolver;
use super::subst::subst_type;
use super::subtype::base_type;

pub fn synth_expr(
    env: &Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    expr: &CExpr,
    ctr: &mut Counter,
) -> Option<CType> {
    match &expr.kind {
        CExprKind::Atomic(a) => synth_atomic(env, ctx, a, ctr),

        CExprKind::BinOp(op, a1, a2) => synth_binop(env, ctx, *op, a1, a2, ctr),

        CExprKind::Not(a) => {
            let v = ctr.fresh();
            let re = atomic_to_refexpr(a);
            let pred = iff_refexpr(
                RefinementExpr {
                    kind: RefinementExprKind::Var(v, String::new()),
                    ty: CType::Bool,
                    span: Span::default(),
                },
                RefinementExpr {
                    kind: RefinementExprKind::Not(Box::new(re)),
                    ty: CType::Bool,
                    span: Span::default(),
                },
            );
            Some(refined(v, CType::Bool, pred))
        }

        CExprKind::Negate(a) => {
            let v = ctr.fresh();
            let re = atomic_to_refexpr(a);
            let pred = RefinementExpr {
                kind: RefinementExprKind::BinOp(
                    CBinOp::IntEq,
                    Box::new(RefinementExpr {
                        kind: RefinementExprKind::Var(v, String::new()),
                        ty: CType::Int,
                        span: Span::default(),
                    }),
                    Box::new(RefinementExpr {
                        kind: RefinementExprKind::BinOp(
                            CBinOp::Subtract,
                            Box::new(RefinementExpr {
                                kind: RefinementExprKind::IntLit(0),
                                ty: CType::Int,
                                span: Span::default(),
                            }),
                            Box::new(re),
                        ),
                        ty: CType::Int,
                        span: Span::default(),
                    }),
                ),
                ty: CType::Bool,
                span: Span::default(),
            };
            Some(refined(v, CType::Int, pred))
        }

        CExprKind::TupleAccess(a, idx) => {
            if let CAtomic::Var(x, _) = a {
                let x_ty = env_lookup(*x, env)?;
                let base = base_type(x_ty);
                if let CType::Tuple(tys) = base {
                    let elem_ty = tys.get(*idx)?.clone();
                    let acc_id = ctx.tuple_accessor_id(*idx)?;
                    let v = ctr.fresh();
                    let pred = eq_refexpr(
                        v,
                        RefinementExpr {
                            kind: RefinementExprKind::ExternCall {
                                target: acc_id,
                                args: vec![RefinementExpr {
                                    kind: RefinementExprKind::Var(*x, String::new()),
                                    ty: CType::Int,
                                    span: Span::default(),
                                }],
                                return_type: elem_ty.clone(),
                            },
                            ty: elem_ty.clone(),
                            span: Span::default(),
                        },
                    );
                    return Some(refined(v, elem_ty, pred));
                }
            }
            None
        }

        CExprKind::FieldAccess(a, field_id) => {
            if let CAtomic::Var(x, _) = a {
                let x_ty = env_lookup(*x, env)?;
                let base = base_type(x_ty);
                if let CType::Struct(sid) = base {
                    let fields = ctx.lookup_struct_fields(*sid)?;
                    let field_ty = fields
                        .iter()
                        .find(|(fid, _, _)| *fid == *field_id)
                        .map(|(_, _, ty)| ty.clone())?;
                    let v = ctr.fresh();
                    let pred = eq_refexpr(
                        v,
                        RefinementExpr {
                            kind: RefinementExprKind::ExternCall {
                                target: *field_id,
                                args: vec![RefinementExpr {
                                    kind: RefinementExprKind::Var(*x, String::new()),
                                    ty: CType::Int,
                                    span: Span::default(),
                                }],
                                return_type: field_ty.clone(),
                            },
                            ty: field_ty.clone(),
                            span: Span::default(),
                        },
                    );
                    return Some(refined(v, field_ty, pred));
                }
            }
            None
        }

        CExprKind::FuncCall(call) => {
            let sig = ctx.lookup_func(call.target)?;
            synth_func_call(
                env,
                ctx,
                solver,
                f_ret,
                &sig.params.clone(),
                &call.args,
                &sig.return_type.clone(),
                ctr,
            )
        }

        CExprKind::TupleLit(elems) => {
            let mut tys = Vec::new();
            for a in elems {
                let e = CExpr {
                    kind: CExprKind::Atomic(a.clone()),
                    ty: CType::Nil,
                    span: Span::default(),
                };
                let t = synth_expr(env, ctx, solver, f_ret, &e, ctr)?;
                tys.push(t);
            }
            Some(CType::Tuple(tys))
        }

        CExprKind::StructLit(sid, assignments) => {
            let fields = ctx.lookup_struct_fields(*sid)?;
            for (field_id, _, field_ty) in fields {
                let val = assignments
                    .iter()
                    .find(|(fid, _)| *fid == *field_id)?
                    .1
                    .clone();
                let e = CExpr {
                    kind: CExprKind::Atomic(val),
                    ty: CType::Nil,
                    span: Span::default(),
                };
                let result = check::check_expr(env, ctx, solver, f_ret, &e, ctr, field_ty);
                if result != CheckResult::Ok {
                    return None;
                }
            }
            Some(CType::Struct(*sid))
        }

        CExprKind::VariantLit(eid, vid, payload) => {
            let variants = ctx.lookup_enum_variants(*eid)?;
            let variant = variants.iter().find(|(v, _, _)| *v == *vid)?;
            match (&variant.2, payload) {
                (None, None) => {}
                (Some(payload_ty), Some(val)) => {
                    let e = CExpr {
                        kind: CExprKind::Atomic(val.clone()),
                        ty: CType::Nil,
                        span: Span::default(),
                    };
                    let result = check::check_expr(env, ctx, solver, f_ret, &e, ctr, payload_ty);
                    if result != CheckResult::Ok {
                        return None;
                    }
                }
                _ => return None,
            }
            Some(CType::Variant(*eid))
        }

        CExprKind::IsVariant(a, eid, vid) => {
            if let CAtomic::Var(x, _) = a {
                let x_ty = env_lookup(*x, env)?;
                let base = base_type(x_ty);
                if let CType::Variant(e) = base {
                    if *e != *eid {
                        return None;
                    }
                }
                let variants = ctx.lookup_enum_variants(*eid)?;
                let _ = variants.iter().find(|(v, _, _)| *v == *vid)?;
                let v = ctr.fresh();
                let pred = RefinementExpr {
                    kind: RefinementExprKind::IsVariant(
                        Box::new(RefinementExpr {
                            kind: RefinementExprKind::Var(*x, String::new()),
                            ty: CType::Int,
                            span: Span::default(),
                        }),
                        *eid,
                        *vid,
                    ),
                    ty: CType::Bool,
                    span: Span::default(),
                };
                return Some(refined(v, CType::Bool, pred));
            }
            None
        }

        CExprKind::VariantPayload(a) => {
            let _ = a;
            Some(expr.ty.clone())
        }

        CExprKind::Block(b) => synth_block(env, ctx, solver, f_ret, b, ctr),
        CExprKind::Conditional(c) => synth_cond(env, ctx, solver, f_ret, c, ctr),
    }
}

pub fn synth_block(
    env: &Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    block: &CBlock,
    ctr: &mut Counter,
) -> Option<CType> {
    let mut cur_env = env.clone();
    for stmt in &block.statements {
        match &stmt.kind {
            CStatementKind::LetAtom(let_atom) => {
                if let_atom.user_annotated {
                    let result = check::check_expr(
                        &cur_env,
                        ctx,
                        solver,
                        f_ret,
                        &let_atom.value,
                        ctr,
                        &let_atom.ty,
                    );
                    if result != CheckResult::Ok {
                        return None;
                    }
                    cur_env.insert(0, (let_atom.name, let_atom.ty.clone()));
                } else {
                    let ty = synth_expr(&cur_env, ctx, solver, f_ret, &let_atom.value, ctr)?;
                    cur_env.insert(0, (let_atom.name, ty));
                }
            }
            CStatementKind::Expr(e) => {
                synth_expr(&cur_env, ctx, solver, f_ret, e, ctr)?;
            }
            CStatementKind::Return(a) => {
                let e = CExpr {
                    kind: CExprKind::Atomic(a.clone()),
                    ty: CType::Nil,
                    span: Span::default(),
                };
                let result = check::check_expr(&cur_env, ctx, solver, f_ret, &e, ctr, f_ret);
                if result != CheckResult::Ok {
                    return None;
                }
                return Some(CType::Never);
            }
            CStatementKind::Error => return None,
        }
    }

    if let Some(tail) = &block.tail_expr {
        let e = CExpr {
            kind: CExprKind::Atomic(tail.clone()),
            ty: CType::Nil,
            span: Span::default(),
        };
        synth_expr(&cur_env, ctx, solver, f_ret, &e, ctr)
    } else {
        Some(CType::Never)
    }
}

pub fn synth_cond(
    env: &Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    cond: &CCondExpr,
    ctr: &mut Counter,
) -> Option<CType> {
    let mut cur_env = env.clone();
    let mut result_ty: Option<CType> = None;

    for branch in std::iter::once(&cond.if_branch).chain(cond.elseif_branches.iter()) {
        let env_true = refine_env_true(&cur_env, &branch.condition, ctr);
        let ty = synth_block(&env_true, ctx, solver, f_ret, &branch.body, ctr)?;
        if result_ty.is_none() {
            result_ty = Some(base_type(&ty).clone());
        }
        cur_env = refine_env_false(&cur_env, &branch.condition, ctr);
    }

    if let Some(else_block) = &cond.else_branch {
        synth_block(&cur_env, ctx, solver, f_ret, else_block, ctr)?;
    }

    result_ty
}

fn synth_func_call(
    env: &Env,
    ctx: &GlobalCtx,
    solver: &mut SmtSolver,
    f_ret: &CType,
    params: &[(NameId, CType)],
    args: &[CAtomic],
    ret_ty: &CType,
    ctr: &mut Counter,
) -> Option<CType> {
    let mut remaining_params = params.to_vec();
    let mut cur_ret = ret_ty.clone();

    for (i, arg) in args.iter().enumerate() {
        if i >= remaining_params.len() {
            return None;
        }
        let (param_id, param_ty) = &remaining_params[i];
        let param_id = *param_id;

        let e = CExpr {
            kind: CExprKind::Atomic(arg.clone()),
            ty: CType::Nil,
            span: Span::default(),
        };
        let result = check::check_expr(env, ctx, solver, f_ret, &e, ctr, param_ty);
        if result != CheckResult::Ok {
            return None;
        }

        let arg_ref = atomic_to_refexpr(arg);
        for j in (i + 1)..remaining_params.len() {
            remaining_params[j].1 = subst_type(&arg_ref, param_id, &remaining_params[j].1);
        }
        cur_ret = subst_type(&arg_ref, param_id, &cur_ret);
    }

    Some(cur_ret)
}

fn synth_atomic(env: &Env, _ctx: &GlobalCtx, a: &CAtomic, ctr: &mut Counter) -> Option<CType> {
    match a {
        CAtomic::Var(x, _) => {
            let ty = env_lookup(*x, env)?;
            let base = base_type(ty).clone();
            let v = ctr.fresh();
            let pred = eq_refexpr(
                v,
                RefinementExpr {
                    kind: RefinementExprKind::Var(*x, String::new()),
                    ty: base.clone(),
                    span: Span::default(),
                },
            );
            Some(refined(v, base, pred))
        }
        CAtomic::IntLit(n) => {
            let v = ctr.fresh();
            let pred = eq_refexpr(
                v,
                RefinementExpr {
                    kind: RefinementExprKind::IntLit(*n),
                    ty: CType::Int,
                    span: Span::default(),
                },
            );
            Some(refined(v, CType::Int, pred))
        }
        CAtomic::BoolLit(true) => {
            let v = ctr.fresh();
            let pred = RefinementExpr {
                kind: RefinementExprKind::Var(v, String::new()),
                ty: CType::Bool,
                span: Span::default(),
            };
            Some(refined(v, CType::Bool, pred))
        }
        CAtomic::BoolLit(false) => {
            let v = ctr.fresh();
            let pred = RefinementExpr {
                kind: RefinementExprKind::Not(Box::new(RefinementExpr {
                    kind: RefinementExprKind::Var(v, String::new()),
                    ty: CType::Bool,
                    span: Span::default(),
                })),
                ty: CType::Bool,
                span: Span::default(),
            };
            Some(refined(v, CType::Bool, pred))
        }
        CAtomic::StringLit(_) => Some(CType::String),
        CAtomic::NilLit => Some(CType::Nil),
        CAtomic::Never => Some(CType::Never),
    }
}

fn synth_binop(
    _env: &Env,
    _ctx: &GlobalCtx,
    op: CBinOp,
    a1: &CAtomic,
    a2: &CAtomic,
    ctr: &mut Counter,
) -> Option<CType> {
    let re1 = atomic_to_refexpr(a1);
    let re2 = atomic_to_refexpr(a2);
    let v = ctr.fresh();

    if is_arithmetic_op(op) {
        let pred = eq_refexpr(
            v,
            RefinementExpr {
                kind: RefinementExprKind::BinOp(op, Box::new(re1), Box::new(re2)),
                ty: CType::Int,
                span: Span::default(),
            },
        );
        Some(refined(v, CType::Int, pred))
    } else {
        // comparison or logical
        let inner = RefinementExpr {
            kind: RefinementExprKind::BinOp(op, Box::new(re1), Box::new(re2)),
            ty: CType::Bool,
            span: Span::default(),
        };
        let pred = iff_refexpr(
            RefinementExpr {
                kind: RefinementExprKind::Var(v, String::new()),
                ty: CType::Bool,
                span: Span::default(),
            },
            inner,
        );
        Some(refined(v, CType::Bool, pred))
    }
}

pub fn refine_env_true(env: &Env, cond: &CAtomic, ctr: &mut Counter) -> Env {
    let v = ctr.fresh();
    let pred = atomic_to_refexpr(cond);
    let mut new_env = env.clone();
    new_env.insert(0, (v, refined(v, CType::Bool, pred)));
    new_env
}

pub fn refine_env_false(env: &Env, cond: &CAtomic, ctr: &mut Counter) -> Env {
    let v = ctr.fresh();
    let pred = RefinementExpr {
        kind: RefinementExprKind::Not(Box::new(atomic_to_refexpr(cond))),
        ty: CType::Bool,
        span: Span::default(),
    };
    let mut new_env = env.clone();
    new_env.insert(0, (v, refined(v, CType::Bool, pred)));
    new_env
}

fn atomic_to_refexpr(a: &CAtomic) -> RefinementExpr {
    let kind = match a {
        CAtomic::Var(id, _) => RefinementExprKind::Var(*id, String::new()),
        CAtomic::IntLit(n) => RefinementExprKind::IntLit(*n),
        CAtomic::StringLit(s) => RefinementExprKind::StringLit(s.clone()),
        CAtomic::BoolLit(b) => RefinementExprKind::BoolLit(*b),
        CAtomic::NilLit => RefinementExprKind::NilLit,
        CAtomic::Never => RefinementExprKind::Var(NameId(usize::MAX), String::new()),
    };
    RefinementExpr {
        kind,
        ty: CType::Int,
        span: Span::default(),
    }
}

fn eq_refexpr(v: NameId, rhs: RefinementExpr) -> RefinementExpr {
    RefinementExpr {
        kind: RefinementExprKind::BinOp(
            CBinOp::IntEq,
            Box::new(RefinementExpr {
                kind: RefinementExprKind::Var(v, String::new()),
                ty: CType::Int,
                span: Span::default(),
            }),
            Box::new(rhs),
        ),
        ty: CType::Bool,
        span: Span::default(),
    }
}

fn iff_refexpr(v_expr: RefinementExpr, p: RefinementExpr) -> RefinementExpr {
    // (¬v ∨ p) ∧ (v ∨ ¬p)
    RefinementExpr {
        kind: RefinementExprKind::BinOp(
            CBinOp::And,
            Box::new(RefinementExpr {
                kind: RefinementExprKind::BinOp(
                    CBinOp::Or,
                    Box::new(RefinementExpr {
                        kind: RefinementExprKind::Not(Box::new(v_expr.clone())),
                        ty: CType::Bool,
                        span: Span::default(),
                    }),
                    Box::new(p.clone()),
                ),
                ty: CType::Bool,
                span: Span::default(),
            }),
            Box::new(RefinementExpr {
                kind: RefinementExprKind::BinOp(
                    CBinOp::Or,
                    Box::new(v_expr),
                    Box::new(RefinementExpr {
                        kind: RefinementExprKind::Not(Box::new(p)),
                        ty: CType::Bool,
                        span: Span::default(),
                    }),
                ),
                ty: CType::Bool,
                span: Span::default(),
            }),
        ),
        ty: CType::Bool,
        span: Span::default(),
    }
}

fn refined(v: NameId, base: CType, pred: RefinementExpr) -> CType {
    CType::Refined(
        Box::new(base),
        CRefinementHandle::new(CRefinementBody {
            bound: v,
            original_bound: String::new(),
            body: pred,
        }),
    )
}

fn is_arithmetic_op(op: CBinOp) -> bool {
    matches!(
        op,
        CBinOp::Add | CBinOp::Subtract | CBinOp::Multiply | CBinOp::Divide | CBinOp::Modulo
    )
}
