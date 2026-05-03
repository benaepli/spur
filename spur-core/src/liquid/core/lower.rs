use std::collections::HashMap;

use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::{
    RefinementBody, Type, TypedBlock, TypedCondExpr, TypedExpr, TypedExprKind, TypedFuncCall,
};
use crate::parser::BinOp;
use crate::liquid::pure::ast::*;

use super::ast::*;
use super::builtins::BuiltinKind;
use super::refinement::{
    RefinementCond, RefinementExpr, RefinementExprKind, RefinementIfBranch,
};

fn to_cbinop(op: &BinOp) -> CBinOp {
    match op {
        BinOp::Add => CBinOp::Add,
        BinOp::Subtract => CBinOp::Subtract,
        BinOp::Multiply => CBinOp::Multiply,
        BinOp::Divide => CBinOp::Divide,
        BinOp::Modulo => CBinOp::Modulo,
        BinOp::Less => CBinOp::Less,
        BinOp::LessEqual => CBinOp::LessEqual,
        BinOp::Greater => CBinOp::Greater,
        BinOp::GreaterEqual => CBinOp::GreaterEqual,
        BinOp::And => CBinOp::And,
        BinOp::Or => CBinOp::Or,
        BinOp::Equal | BinOp::NotEqual | BinOp::Coalesce => {
            unreachable!("polymorphic/desugared ops should not reach to_cbinop")
        }
    }
}

pub struct LowerOutput {
    pub program: CProgram,
    pub refinement_errors: Vec<RefinementValidationError>,
}

pub fn lower_program(program: PProgram) -> LowerOutput {
    let mut lowerer = CoreLowerer {
        next_name_id: program.next_name_id,
        id_to_name: program.id_to_name,
        var_types: HashMap::new(),
        extern_cache: HashMap::new(),
        extern_funcs: Vec::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
        funcs: Vec::new(),
        body_memo: HashMap::new(),
        refinement_errors: Vec::new(),
        list_invariant_cache: HashMap::new(),
        list_nonempty_cache: HashMap::new(),
        pending_stmts: Vec::new(),
    };

    let struct_entries: Vec<_> = program
        .struct_defs
        .iter()
        .map(|(id, fields)| (*id, fields.clone()))
        .collect();
    for (id, fields) in struct_entries {
        let lowered: Vec<_> = fields
            .iter()
            .map(|(name, ty)| (name.clone(), lowerer.lower_type(ty)))
            .collect();
        lowerer.struct_defs.insert(id, lowered);
    }
    let enum_entries: Vec<_> = program
        .enum_defs
        .iter()
        .map(|(id, variants)| (*id, variants.clone()))
        .collect();
    for (id, variants) in enum_entries {
        let lowered: Vec<_> = variants
            .iter()
            .map(|(name, payload)| (name.clone(), payload.as_ref().map(|p| lowerer.lower_type(p))))
            .collect();
        lowerer.enum_defs.insert(id, lowered);
    }

    for top_def in program.top_level_defs {
        match top_def {
            PTopLevelDef::Role(role) => {
                let role_id = role.name;
                for func in role.func_defs {
                    let lowered = lowerer.lower_func(func, Some(role_id));
                    lowerer.funcs.push(lowered);
                }
            }
            PTopLevelDef::FreeFunc(func) => {
                let lowered = lowerer.lower_func(func, None);
                lowerer.funcs.push(lowered);
            }
        }
    }

    let program = CProgram {
        funcs: lowerer.funcs,
        extern_funcs: lowerer.extern_funcs,
        struct_defs: lowerer.struct_defs,
        enum_defs: lowerer.enum_defs,
        next_name_id: lowerer.next_name_id,
        id_to_name: lowerer.id_to_name,
    };
    let mut refinement_errors = lowerer.refinement_errors;
    refinement_errors.extend(super::validate::validate_refinements(&program));
    LowerOutput {
        program,
        refinement_errors,
    }
}

struct CoreLowerer {
    next_name_id: usize,
    id_to_name: HashMap<NameId, String>,
    /// Original P-types for every named binding, populated as we walk.
    var_types: HashMap<NameId, Type>,
    extern_cache: HashMap<(BuiltinKind, Vec<CType>), NameId>,
    extern_funcs: Vec<CExternFunc>,
    struct_defs: HashMap<NameId, Vec<(String, CType)>>,
    enum_defs: HashMap<NameId, Vec<(String, Option<CType>)>>,
    funcs: Vec<CFuncDef>,
    /// Cache keyed by `Arc::as_ptr` of source refinement bodies so identical
    /// refinements produce the same lowered CRefinementHandle.
    body_memo: HashMap<*const RefinementBody, CRefinementHandle>,
    /// Errors collected while lowering refinement bodies.
    refinement_errors: Vec<RefinementValidationError>,
    /// Cache of the baked-in `array_len(xs) >= 0` invariant per element type.
    /// Ensures every `list<T>` produced by `lower_type` reuses one handle.
    list_invariant_cache: HashMap<CType, CRefinementHandle>,
    /// Cache of the `{ _xs | array_len(_xs) >= 1 }` non-empty precondition,
    /// shared by `array_head` and `array_tail` parameter slots.
    list_nonempty_cache: HashMap<CType, CRefinementHandle>,
    /// Statements emitted as side-effects of lowering an expression (e.g. the
    /// `array_empty`/`array_append` chain for `[a, b, c]`). `lower_block`
    /// drains this buffer in front of each statement it pushes.
    pending_stmts: Vec<CStatement>,
}

impl CoreLowerer {
    fn lower_func(&mut self, func: PFuncDef, role: Option<NameId>) -> CFuncDef {
        for param in &func.params {
            self.var_types.insert(param.name, param.ty.clone());
        }
        let mut params = Vec::with_capacity(func.params.len());
        for p in &func.params {
            params.push(CFuncParam {
                name: p.name,
                original_name: p.original_name.clone(),
                ty: self.lower_type(&p.ty),
                span: p.span,
            });
        }

        let return_type = self.lower_type(&func.return_type);
        let body = self.lower_block(func.body);

        CFuncDef {
            name: func.name,
            original_name: func.original_name,
            kind: match func.kind {
                PFuncKind::Sync => CFuncKind::Sync,
                PFuncKind::Async => CFuncKind::Async,
                PFuncKind::LoopConverted => CFuncKind::LoopConverted,
            },
            is_traced: func.is_traced,
            role,
            params,
            return_type,
            body,
            span: func.span,
        }
    }

    fn lower_block(&mut self, block: PBlock) -> CBlock {
        let mut statements = Vec::with_capacity(block.statements.len());
        for stmt in block.statements {
            let lowered = self.lower_statement(stmt);
            statements.append(&mut self.pending_stmts);
            statements.push(lowered);
        }
        debug_assert!(
            self.pending_stmts.is_empty(),
            "pending_stmts must be drained at end of lower_block",
        );
        let ty = self.lower_type(&block.ty);
        CBlock {
            statements,
            tail_expr: block.tail_expr.map(lower_atomic),
            ty,
            span: block.span,
        }
    }

    fn lower_statement(&mut self, stmt: PStatement) -> CStatement {
        let kind = match stmt.kind {
            PStatementKind::LetAtom(let_atom) => {
                let value = self.lower_expr(let_atom.value);
                self.var_types.insert(let_atom.name, let_atom.ty.clone());
                let ty = self.lower_type(&let_atom.ty);
                CStatementKind::LetAtom(CLetAtom {
                    name: let_atom.name,
                    original_name: let_atom.original_name,
                    ty,
                    value,
                    user_annotated: let_atom.user_annotated,
                    span: let_atom.span,
                })
            }
            PStatementKind::Expr(expr) => CStatementKind::Expr(self.lower_expr(expr)),
            PStatementKind::Return(atom) => CStatementKind::Return(lower_atomic(atom)),
            PStatementKind::Error => CStatementKind::Error,
        };
        CStatement {
            kind,
            span: stmt.span,
        }
    }

    fn lower_expr(&mut self, expr: PExpr) -> CExpr {
        let span = expr.span;
        let result_ty = self.lower_type(&expr.ty);
        let kind = self.lower_expr_kind(expr.kind, &expr.ty);
        CExpr {
            kind,
            ty: result_ty,
            span,
        }
    }

    fn lower_expr_kind(&mut self, kind: PExprKind, result_ty: &Type) -> CExprKind {
        match kind {
            PExprKind::Atomic(a) => CExprKind::Atomic(lower_atomic(a)),
            PExprKind::BinOp(op, a, b) => match op {
                BinOp::Equal | BinOp::NotEqual => {
                    let operand_ty = self.lower_type(&self.atomic_p_type(&a));
                    let prim_op = match (&operand_ty, &op) {
                        (CType::Int, BinOp::Equal) => Some(CBinOp::IntEq),
                        (CType::Int, BinOp::NotEqual) => Some(CBinOp::IntNeq),
                        (CType::Bool, BinOp::Equal) => Some(CBinOp::BoolEq),
                        (CType::Bool, BinOp::NotEqual) => Some(CBinOp::BoolNeq),
                        _ => None,
                    };
                    if let Some(prim_op) = prim_op {
                        CExprKind::BinOp(prim_op, lower_atomic(a), lower_atomic(b))
                    } else {
                        let kind = if op == BinOp::Equal {
                            BuiltinKind::Eq
                        } else {
                            BuiltinKind::Neq
                        };
                        self.emit_extern_call(
                            kind,
                            vec![operand_ty.clone()],
                            vec![operand_ty.clone(), operand_ty],
                            CType::Bool,
                            vec![a, b],
                        )
                    }
                }
                _ => CExprKind::BinOp(to_cbinop(&op), lower_atomic(a), lower_atomic(b)),
            },
            PExprKind::Not(a) => CExprKind::Not(lower_atomic(a)),
            PExprKind::Negate(a) => CExprKind::Negate(lower_atomic(a)),

            PExprKind::FuncCall(call) => match call {
                PFuncCall::User(user) => CExprKind::FuncCall(CFuncCall {
                    target: user.name,
                    args: user.args.into_iter().map(lower_atomic).collect(),
                    return_type: self.lower_type(&user.return_type),
                }),
                PFuncCall::Builtin(b, args, ret) => {
                    let kind = match b {
                        BuiltinFn::Println => BuiltinKind::Println,
                        BuiltinFn::IntToString => BuiltinKind::IntToString,
                        BuiltinFn::BoolToString => BuiltinKind::BoolToString,
                        BuiltinFn::RoleToString => BuiltinKind::RoleToString,
                        BuiltinFn::UniqueId => BuiltinKind::UniqueId,
                    };
                    let arg_types: Vec<CType> =
                        args.iter().map(|a| self.lower_type(&self.atomic_p_type(a))).collect();
                    let ret_c = self.lower_type(&ret);
                    self.emit_extern_call(kind, vec![], arg_types, ret_c, args)
                }
            },

            PExprKind::ListLit(items) => self.desugar_list_lit_stmt(items, result_ty),
            PExprKind::TupleLit(items) => {
                CExprKind::TupleLit(items.into_iter().map(lower_atomic).collect())
            }
            PExprKind::MapLit(pairs) => CExprKind::MapLit(
                pairs
                    .into_iter()
                    .map(|(k, v)| (lower_atomic(k), lower_atomic(v)))
                    .collect(),
            ),

            PExprKind::Append(list, item) => {
                let elem = self.array_elem_of(&list);
                let raw_array = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call_with_params(
                    BuiltinKind::ArrayAppend,
                    vec![elem.clone()],
                    vec![
                        ("xs".to_string(), raw_array),
                        ("e".to_string(), elem.clone()),
                    ],
                    move |this: &mut Self, params: &[CExternParam]| {
                        let inner_invariant = this.make_list_invariant_handle(elem.clone());
                        let inner_refined = CType::Refined(
                            Box::new(CType::Array(Box::new(elem.clone()))),
                            inner_invariant,
                        );
                        let outer = this.make_append_return_handle(
                            elem,
                            params[0].name,
                            params[0].original_name.clone(),
                        );
                        CType::Refined(Box::new(inner_refined), outer)
                    },
                    vec![list, item],
                )
            }
            PExprKind::Prepend(list, item) => {
                let elem = self.array_elem_of(&list);
                let raw_array = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call_with_params(
                    BuiltinKind::ArrayPrepend,
                    vec![elem.clone()],
                    vec![
                        ("xs".to_string(), raw_array),
                        ("e".to_string(), elem.clone()),
                    ],
                    move |this: &mut Self, params: &[CExternParam]| {
                        let inner_invariant = this.make_list_invariant_handle(elem.clone());
                        let inner_refined = CType::Refined(
                            Box::new(CType::Array(Box::new(elem.clone()))),
                            inner_invariant,
                        );
                        let outer = this.make_append_return_handle(
                            elem,
                            params[0].name,
                            params[0].original_name.clone(),
                        );
                        CType::Refined(Box::new(inner_refined), outer)
                    },
                    vec![list, item],
                )
            }
            PExprKind::Min(a, b) => {
                let ty = self.lower_type(&self.atomic_p_type(&a));
                self.emit_extern_call(
                    BuiltinKind::Min,
                    vec![ty.clone()],
                    vec![ty.clone(), ty.clone()],
                    ty,
                    vec![a, b],
                )
            }
            PExprKind::Exists(map, key) => {
                let (k, v) = self.map_kv_of(&map);
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                self.emit_extern_call(
                    BuiltinKind::MapExists,
                    vec![k.clone(), v],
                    vec![map_ty, k],
                    CType::Bool,
                    vec![map, key],
                )
            }
            PExprKind::Erase(map, key) => {
                let (k, v) = self.map_kv_of(&map);
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                let k_for_param = k.clone();
                let map_for_param = map_ty.clone();
                let k_for_post = k.clone();
                let v_for_post = v.clone();
                let map_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |_this, _params| map_for_param);
                let key_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |_this, _params| k_for_param);
                let target = self.intern_extern_with_dependent_params(
                    BuiltinKind::MapErase,
                    vec![k.clone(), v],
                    vec![
                        ("m".to_string(), map_builder),
                        ("k".to_string(), key_builder),
                    ],
                    move |this, params| {
                        let post = this.make_map_erase_return_handle(
                            k_for_post,
                            v_for_post,
                            params[1].name,
                            params[1].original_name.clone(),
                        );
                        CType::Refined(Box::new(map_ty), post)
                    },
                );
                let return_type = self.extern_return_type(target);
                CExprKind::FuncCall(CFuncCall {
                    target,
                    args: vec![lower_atomic(map), lower_atomic(key)],
                    return_type,
                })
            }
            PExprKind::Store(map, key, val) => {
                let (k, v) = self.map_kv_of(&map);
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                let k_for_param = k.clone();
                let v_for_param = v.clone();
                let map_for_param = map_ty.clone();
                let k_for_post = k.clone();
                let v_for_post = v.clone();
                let map_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |_this, _params| map_for_param);
                let key_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |_this, _params| k_for_param);
                let val_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |_this, _params| v_for_param);
                let target = self.intern_extern_with_dependent_params(
                    BuiltinKind::MapStore,
                    vec![k.clone(), v.clone()],
                    vec![
                        ("m".to_string(), map_builder),
                        ("k".to_string(), key_builder),
                        ("v".to_string(), val_builder),
                    ],
                    move |this, params| {
                        let post = this.make_map_store_return_handle(
                            k_for_post,
                            v_for_post,
                            params[1].name,
                            params[1].original_name.clone(),
                        );
                        CType::Refined(Box::new(map_ty), post)
                    },
                );
                let return_type = self.extern_return_type(target);
                CExprKind::FuncCall(CFuncCall {
                    target,
                    args: vec![lower_atomic(map), lower_atomic(key), lower_atomic(val)],
                    return_type,
                })
            }
            PExprKind::Head(list) => {
                let elem = self.array_elem_of(&list);
                let raw_array = CType::Array(Box::new(elem.clone()));
                let elem_for_pre = elem.clone();
                let elem_for_ret = elem.clone();
                let xs_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |this, _params| {
                    let nonempty =
                        this.make_array_nonempty_precondition_handle(elem_for_pre);
                    CType::Refined(Box::new(raw_array), nonempty)
                });
                let target = self.intern_extern_with_dependent_params(
                    BuiltinKind::ArrayHead,
                    vec![elem],
                    vec![("xs".to_string(), xs_builder)],
                    move |_this, _params| elem_for_ret,
                );
                let return_type = self.extern_return_type(target);
                CExprKind::FuncCall(CFuncCall {
                    target,
                    args: vec![lower_atomic(list)],
                    return_type,
                })
            }
            PExprKind::Tail(list) => {
                let elem = self.array_elem_of(&list);
                let raw_array = CType::Array(Box::new(elem.clone()));
                let elem_for_pre = elem.clone();
                let elem_for_inner_inv = elem.clone();
                let elem_for_post = elem.clone();
                let raw_array_for_pre = raw_array.clone();
                let xs_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |this, _params| {
                    let nonempty =
                        this.make_array_nonempty_precondition_handle(elem_for_pre);
                    CType::Refined(Box::new(raw_array_for_pre), nonempty)
                });
                let target = self.intern_extern_with_dependent_params(
                    BuiltinKind::ArrayTail,
                    vec![elem],
                    vec![("xs".to_string(), xs_builder)],
                    move |this, params| {
                        let inner_invariant =
                            this.make_list_invariant_handle(elem_for_inner_inv.clone());
                        let inner_refined = CType::Refined(
                            Box::new(CType::Array(Box::new(elem_for_inner_inv.clone()))),
                            inner_invariant,
                        );
                        let outer = this.make_array_tail_return_handle(
                            elem_for_post,
                            params[0].name,
                            params[0].original_name.clone(),
                        );
                        CType::Refined(Box::new(inner_refined), outer)
                    },
                );
                let return_type = self.extern_return_type(target);
                let _ = raw_array;
                CExprKind::FuncCall(CFuncCall {
                    target,
                    args: vec![lower_atomic(list)],
                    return_type,
                })
            }
            PExprKind::Len(list) => {
                let elem = self.array_elem_of(&list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::ArrayLen,
                    vec![elem],
                    vec![list_ty],
                    CType::Int,
                    vec![list],
                )
            }
            PExprKind::Slice(list, lo, hi) => {
                let elem = self.array_elem_of(&list);
                let raw_array = CType::Array(Box::new(elem.clone()));
                let elem_for_hi = elem.clone();
                let elem_for_inner_inv = elem.clone();
                let elem_for_post = elem.clone();
                let xs_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |_this, _params| raw_array);
                let lo_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |this, _params| {
                    let pre = this.make_array_slice_lo_precondition_handle();
                    CType::Refined(Box::new(CType::Int), pre)
                });
                let hi_builder: Box<
                    dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                > = Box::new(move |this, params| {
                    let pre = this.make_array_slice_hi_precondition_handle(
                        elem_for_hi,
                        params[1].name,
                        params[1].original_name.clone(),
                        params[0].name,
                        params[0].original_name.clone(),
                    );
                    CType::Refined(Box::new(CType::Int), pre)
                });
                let target = self.intern_extern_with_dependent_params(
                    BuiltinKind::ArraySlice,
                    vec![elem],
                    vec![
                        ("xs".to_string(), xs_builder),
                        ("lo".to_string(), lo_builder),
                        ("hi".to_string(), hi_builder),
                    ],
                    move |this, params| {
                        let inner_invariant =
                            this.make_list_invariant_handle(elem_for_inner_inv.clone());
                        let inner_refined = CType::Refined(
                            Box::new(CType::Array(Box::new(elem_for_inner_inv.clone()))),
                            inner_invariant,
                        );
                        let outer = this.make_array_slice_return_handle(
                            elem_for_post,
                            params[1].name,
                            params[1].original_name.clone(),
                            params[2].name,
                            params[2].original_name.clone(),
                        );
                        CType::Refined(Box::new(inner_refined), outer)
                    },
                );
                let return_type = self.extern_return_type(target);
                CExprKind::FuncCall(CFuncCall {
                    target,
                    args: vec![lower_atomic(list), lower_atomic(lo), lower_atomic(hi)],
                    return_type,
                })
            }

            PExprKind::RpcCall(dest, call) => {
                let dest_ty = self.lower_type(&self.atomic_p_type(&dest));
                let call_arg_types: Vec<CType> = call
                    .args
                    .iter()
                    .map(|a| self.lower_type(&self.atomic_p_type(a)))
                    .collect();
                let mut params = vec![dest_ty];
                params.extend(call_arg_types);
                let mut args = vec![dest];
                args.extend(call.args);
                let ret_ty = self.lower_type(&call.return_type);
                self.emit_extern_call(
                    BuiltinKind::Rpc(call.name),
                    vec![],
                    params,
                    ret_ty,
                    args,
                )
            }

            PExprKind::Conditional(cond) => {
                let lowered = self.lower_cond(*cond);
                CExprKind::Conditional(Box::new(lowered))
            }
            PExprKind::Block(b) => CExprKind::Block(Box::new(self.lower_block(*b))),

            PExprKind::VariantLit(id, name, payload) => {
                CExprKind::VariantLit(id, name, payload.map(lower_atomic))
            }
            PExprKind::IsVariant(a, name) => CExprKind::IsVariant(lower_atomic(a), name),
            PExprKind::VariantPayload(a) => CExprKind::VariantPayload(lower_atomic(a)),

            PExprKind::UnwrapOptional(a) => {
                let inner = self.optional_inner_of(&a);
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                self.emit_extern_call(
                    BuiltinKind::OptionalUnwrap,
                    vec![inner.clone()],
                    vec![opt_ty],
                    inner,
                    vec![a],
                )
            }
            PExprKind::WrapInOptional(a) => {
                let inner = self.lower_type(&self.atomic_p_type(&a));
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                self.emit_extern_call(
                    BuiltinKind::OptionalWrap,
                    vec![inner.clone()],
                    vec![inner],
                    opt_ty,
                    vec![a],
                )
            }

            PExprKind::MakeIter(list) => {
                let elem = self.array_elem_of(&list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                let iter_ty = CType::Iter(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::IterMake,
                    vec![elem],
                    vec![list_ty],
                    iter_ty,
                    vec![list],
                )
            }
            PExprKind::IterIsDone(it) => {
                let elem = self.iter_inner_of(&it);
                let iter_ty = CType::Iter(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::IterIsDone,
                    vec![elem],
                    vec![iter_ty],
                    CType::Bool,
                    vec![it],
                )
            }
            PExprKind::IterNext(it) => {
                let elem = self.iter_inner_of(&it);
                let iter_ty = CType::Iter(Box::new(elem.clone()));
                let result = CType::Tuple(vec![iter_ty.clone(), elem.clone()]);
                let ret = if matches!(result_ty, Type::Error) {
                    result
                } else {
                    self.lower_type(result_ty)
                };
                self.emit_extern_call(
                    BuiltinKind::IterNext,
                    vec![elem],
                    vec![iter_ty],
                    ret,
                    vec![it],
                )
            }

            PExprKind::MakeChannel => {
                let elem = match result_ty {
                    Type::Chan(t) => self.lower_type(t),
                    _ => CType::Never,
                };
                let chan_ty = CType::Chan(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::ChanMake,
                    vec![elem],
                    vec![],
                    chan_ty,
                    vec![],
                )
            }
            PExprKind::Send(state, chan, value) => {
                let elem = lower_type_simple(&self.atomic_p_type(&value));
                let chan_ty = CType::Chan(Box::new(elem.clone()));
                let state_ty = lower_type_simple(&self.atomic_p_type(&state));
                let ret_ty = self.lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::ChanSend,
                    vec![elem.clone()],
                    vec![state_ty, chan_ty, elem],
                    ret_ty,
                    vec![state, chan, value],
                )
            }
            PExprKind::Recv(state, chan) => {
                let elem = self.chan_inner_of(&chan);
                let chan_ty = CType::Chan(Box::new(elem.clone()));
                let state_ty = lower_type_simple(&self.atomic_p_type(&state));
                let ret_ty = self.lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::ChanRecv,
                    vec![elem],
                    vec![state_ty, chan_ty],
                    ret_ty,
                    vec![state, chan],
                )
            }

            PExprKind::SetTimer(_label) => {
                let elem = match result_ty {
                    Type::Chan(t) => self.lower_type(t),
                    _ => CType::Tuple(vec![]),
                };
                let chan_ty = CType::Chan(Box::new(elem));
                self.emit_extern_call(BuiltinKind::TimerSet, vec![], vec![], chan_ty, vec![])
            }

            PExprKind::Fifo(peer) => {
                let role_ty = lower_type_simple(&self.atomic_p_type(&peer));
                let inner = match result_ty {
                    Type::FifoLink(t) => self.lower_type(t),
                    _ => role_ty.clone(),
                };
                let link_ty = CType::FifoLink(Box::new(inner));
                self.emit_extern_call(
                    BuiltinKind::FifoCreate,
                    vec![role_ty.clone()],
                    vec![role_ty],
                    link_ty,
                    vec![peer],
                )
            }

            PExprKind::Index(coll, idx) => {
                let coll_ty = self.atomic_p_type(&coll);
                match coll_ty {
                    Type::List(_) => {
                        let elem = self.array_elem_of(&coll);
                        let raw_array = CType::Array(Box::new(elem.clone()));
                        let elem_for_i = elem.clone();
                        let elem_for_ret = elem.clone();
                        let xs_builder: Box<
                            dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                        > = Box::new(move |_this, _params| raw_array);
                        let i_builder: Box<
                            dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                        > = Box::new(move |this, params| {
                            let precondition = this.make_index_precondition_handle(
                                elem_for_i,
                                params[0].name,
                                params[0].original_name.clone(),
                            );
                            CType::Refined(Box::new(CType::Int), precondition)
                        });
                        let target = self.intern_extern_with_dependent_params(
                            BuiltinKind::ArrayIndex,
                            vec![elem],
                            vec![
                                ("xs".to_string(), xs_builder),
                                ("i".to_string(), i_builder),
                            ],
                            move |_this, _params| elem_for_ret,
                        );
                        let return_type = self.extern_return_type(target);
                        CExprKind::FuncCall(CFuncCall {
                            target,
                            args: vec![lower_atomic(coll), lower_atomic(idx)],
                            return_type,
                        })
                    }
                    _ => {
                        let (k, v) = self.map_kv_of(&coll);
                        let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                        let k_for_pre = k.clone();
                        let v_for_pre = v.clone();
                        let v_for_ret = v.clone();
                        let map_builder: Box<
                            dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                        > = Box::new(move |_this, _params| map_ty);
                        let key_builder: Box<
                            dyn FnOnce(&mut Self, &[CExternParam]) -> CType,
                        > = Box::new(move |this, params| {
                            let precondition = this.make_map_index_precondition_handle(
                                k_for_pre.clone(),
                                v_for_pre,
                                params[0].name,
                                params[0].original_name.clone(),
                            );
                            CType::Refined(Box::new(k_for_pre), precondition)
                        });
                        let target = self.intern_extern_with_dependent_params(
                            BuiltinKind::MapIndex,
                            vec![k, v],
                            vec![
                                ("m".to_string(), map_builder),
                                ("k".to_string(), key_builder),
                            ],
                            move |_this, _params| v_for_ret,
                        );
                        let return_type = self.extern_return_type(target);
                        CExprKind::FuncCall(CFuncCall {
                            target,
                            args: vec![lower_atomic(coll), lower_atomic(idx)],
                            return_type,
                        })
                    }
                }
            }
            PExprKind::TupleAccess(t, i) => CExprKind::TupleAccess(lower_atomic(t), i),
            PExprKind::FieldAccess(s, f) => CExprKind::FieldAccess(lower_atomic(s), f),

            PExprKind::SafeFieldAccess(s, field) => {
                let struct_ty = lower_type_simple(&self.atomic_p_type(&s));
                let ret_c = self.lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::SafeField(field.clone()),
                    vec![struct_ty.clone(), ret_c.clone()],
                    vec![struct_ty],
                    ret_c,
                    vec![s],
                )
            }
            PExprKind::SafeIndex(coll, idx) => {
                let coll_ty = lower_type_simple(&self.atomic_p_type(&coll));
                let idx_ty = lower_type_simple(&self.atomic_p_type(&idx));
                let ret_c = self.lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::SafeIndex,
                    vec![coll_ty.clone(), idx_ty.clone(), ret_c.clone()],
                    vec![coll_ty, idx_ty],
                    ret_c,
                    vec![coll, idx],
                )
            }
            PExprKind::SafeTupleAccess(t, i) => {
                let tuple_ty = lower_type_simple(&self.atomic_p_type(&t));
                let ret_c = self.lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::SafeTupleAccess(i),
                    vec![tuple_ty.clone(), ret_c.clone()],
                    vec![tuple_ty],
                    ret_c,
                    vec![t],
                )
            }

            PExprKind::StructLit(id, fields) => CExprKind::StructLit(
                id,
                fields
                    .into_iter()
                    .map(|(n, a)| (n, lower_atomic(a)))
                    .collect(),
            ),

            PExprKind::PersistData(value) => {
                let inner = lower_type_simple(&self.atomic_p_type(&value));
                let ret_ty = self.lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::Persist,
                    vec![inner.clone()],
                    vec![inner],
                    ret_ty,
                    vec![value],
                )
            }
            PExprKind::RetrieveData(ty) => {
                let inner = self.lower_type(&ty);
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                self.emit_extern_call(
                    BuiltinKind::Retrieve,
                    vec![inner],
                    vec![],
                    opt_ty,
                    vec![],
                )
            }
            PExprKind::DiscardData => {
                let ret_ty = self.lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::Discard,
                    vec![],
                    vec![],
                    ret_ty,
                    vec![],
                )
            }
        }
    }

    fn lower_cond(&mut self, cond: PCondExpr) -> CCondExpr {
        let if_branch = self.lower_if_branch(cond.if_branch);
        let elseif_branches = cond
            .elseif_branches
            .into_iter()
            .map(|b| self.lower_if_branch(b))
            .collect();
        let else_branch = cond.else_branch.map(|b| self.lower_block(b));
        CCondExpr {
            if_branch,
            elseif_branches,
            else_branch,
            span: cond.span,
        }
    }

    fn lower_if_branch(&mut self, branch: PIfBranch) -> CIfBranch {
        CIfBranch {
            condition: lower_atomic(branch.condition),
            body: self.lower_block(branch.body),
            span: branch.span,
        }
    }

    /// Desugar `[a, b, c]` (statement-side) into a chain of `array_empty` +
    /// `array_append` calls. For each element, push a `let` of the
    /// intermediate result into `self.pending_stmts`; return an `Atomic`
    /// expression referencing the final intermediate variable. The empty
    /// case `[]` pushes a single `array_empty` let and returns its var.
    fn desugar_list_lit_stmt(&mut self, items: Vec<PAtomic>, result_ty: &Type) -> CExprKind {
        let elem = match result_ty {
            Type::List(t) => self.lower_type(t),
            Type::EmptyList => CType::Never,
            other => extract_array_elem(&self.lower_type(other)).unwrap_or(CType::Never),
        };
        let span = crate::parser::Span::default();

        // Build the empty list as the seed.
        let empty_value = self.lower_array_empty_stmt(elem.clone(), span);
        let mut current_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        let current_orig = format!("_list_lit{}", current_id.0);
        self.id_to_name.insert(current_id, current_orig.clone());
        let mut current_ty = empty_value.ty.clone();
        self.pending_stmts.push(CStatement {
            kind: CStatementKind::LetAtom(CLetAtom {
                name: current_id,
                original_name: current_orig,
                ty: current_ty.clone(),
                value: empty_value,
                user_annotated: false,
                span,
            }),
            span,
        });

        // Append each element in order.
        for item in items {
            let elem_for_builder = elem.clone();
            let raw_array = CType::Array(Box::new(elem.clone()));
            let kind = self.emit_extern_call_with_params(
                BuiltinKind::ArrayAppend,
                vec![elem.clone()],
                vec![
                    ("xs".to_string(), raw_array),
                    ("e".to_string(), elem.clone()),
                ],
                move |this, params| {
                    let inner_invariant =
                        this.make_list_invariant_handle(elem_for_builder.clone());
                    let inner_refined = CType::Refined(
                        Box::new(CType::Array(Box::new(elem_for_builder.clone()))),
                        inner_invariant,
                    );
                    let outer = this.make_append_return_handle(
                        elem_for_builder,
                        params[0].name,
                        params[0].original_name.clone(),
                    );
                    CType::Refined(Box::new(inner_refined), outer)
                },
                vec![
                    PAtomic::Var(current_id, "".to_string()),
                    item,
                ],
            );
            let return_ty = match &kind {
                CExprKind::FuncCall(call) => call.return_type.clone(),
                _ => unreachable!("emit_extern_call_with_params returns FuncCall"),
            };
            let value = CExpr {
                kind,
                ty: return_ty.clone(),
                span,
            };

            let next_id = NameId(self.next_name_id);
            self.next_name_id += 1;
            let next_orig = format!("_list_lit{}", next_id.0);
            self.id_to_name.insert(next_id, next_orig.clone());
            self.pending_stmts.push(CStatement {
                kind: CStatementKind::LetAtom(CLetAtom {
                    name: next_id,
                    original_name: next_orig.clone(),
                    ty: return_ty.clone(),
                    value,
                    user_annotated: false,
                    span,
                }),
                span,
            });
            current_id = next_id;
            current_ty = return_ty;
        }
        let _ = current_ty;
        CExprKind::Atomic(CAtomic::Var(current_id, format!("_list_lit{}", current_id.0)))
    }

    /// Resolve-or-allocate an extern entry, then build the FuncCall expression.
    /// `param_tys` is the raw parameter type list; synthetic NameIds (`_p0`,
    /// `_p1`, ...) are minted lazily on cache miss inside `intern_extern`.
    fn emit_extern_call(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        param_tys: Vec<CType>,
        return_type: CType,
        args: Vec<PAtomic>,
    ) -> CExprKind {
        let target = self.intern_extern(kind, type_args, param_tys, return_type.clone());
        CExprKind::FuncCall(CFuncCall {
            target,
            args: args.into_iter().map(lower_atomic).collect(),
            return_type,
        })
    }

    /// Same as `emit_extern_call`, but the return type is built from the
    /// freshly-minted parameter `CExternParam`s on cache miss. On cache hit
    /// the cached `CFuncCall.return_type` is reused. This is used for array
    /// constructors whose return refines `array_len(out) == array_len(xs)+1`.
    fn emit_extern_call_with_params<F>(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        param_tys: Vec<(String, CType)>,
        return_type_builder: F,
        args: Vec<PAtomic>,
    ) -> CExprKind
    where
        F: FnOnce(&mut Self, &[CExternParam]) -> CType,
    {
        let target = self.intern_extern_with_params(
            kind,
            type_args,
            param_tys,
            return_type_builder,
        );
        let return_type = self.extern_return_type(target);
        CExprKind::FuncCall(CFuncCall {
            target,
            args: args.into_iter().map(lower_atomic).collect(),
            return_type,
        })
    }

    /// Convenience wrapper for sites with a precomputed return type and no
    /// need for a parameter-name hint. Synthetic names `_p0..` are minted.
    fn intern_extern(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        param_tys: Vec<CType>,
        return_type: CType,
    ) -> NameId {
        let return_type_for_builder = return_type.clone();
        let named: Vec<(String, CType)> = param_tys
            .into_iter()
            .enumerate()
            .map(|(i, ty)| (format!("_p{}", i), ty))
            .collect();
        self.intern_extern_with_params(
            kind,
            type_args,
            named,
            move |_, _| return_type_for_builder,
        )
    }

    /// Core extern-interning routine. Cache key is `(kind, type_args)`. On
    /// cache miss, mints a fresh NameId per parameter and runs
    /// `return_type_builder` so refined returns can reference parameter
    /// NameIds. On cache hit, the closure is dropped without running.
    fn intern_extern_with_params<F>(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        param_tys: Vec<(String, CType)>,
        return_type_builder: F,
    ) -> NameId
    where
        F: FnOnce(&mut Self, &[CExternParam]) -> CType,
    {
        let key = (kind.clone(), type_args.clone());
        if let Some(id) = self.extern_cache.get(&key) {
            return *id;
        }
        let params: Vec<CExternParam> = param_tys
            .into_iter()
            .map(|(hint, ty)| self.mint_extern_param(&hint, ty))
            .collect();
        let return_type = return_type_builder(self, &params);
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        let original_name = format_extern_name(&kind, &type_args);
        self.id_to_name.insert(id, original_name.clone());
        self.extern_funcs.push(CExternFunc {
            name: id,
            original_name,
            params,
            return_type,
        });
        self.extern_cache.insert(key, id);
        id
    }

    /// Like `intern_extern_with_params`, but each parameter's type is built
    /// lazily by a closure that receives the slice of already-minted earlier
    /// parameters. Used when a later parameter's refinement needs to
    /// reference an earlier parameter's NameId — e.g. `array_index<T>`'s
    /// index parameter, whose precondition `0 <= i && i < array_len(xs)`
    /// references the `xs` parameter. Cache key is unchanged: `(kind,
    /// type_args)`.
    fn intern_extern_with_dependent_params<G>(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        param_specs: Vec<(String, Box<dyn FnOnce(&mut Self, &[CExternParam]) -> CType>)>,
        return_type_builder: G,
    ) -> NameId
    where
        G: FnOnce(&mut Self, &[CExternParam]) -> CType,
    {
        let key = (kind.clone(), type_args.clone());
        if let Some(id) = self.extern_cache.get(&key) {
            return *id;
        }
        let mut params: Vec<CExternParam> = Vec::with_capacity(param_specs.len());
        for (hint, builder) in param_specs {
            let ty = builder(self, &params);
            let p = self.mint_extern_param(&hint, ty);
            params.push(p);
        }
        let return_type = return_type_builder(self, &params);
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        let original_name = format_extern_name(&kind, &type_args);
        self.id_to_name.insert(id, original_name.clone());
        self.extern_funcs.push(CExternFunc {
            name: id,
            original_name,
            params,
            return_type,
        });
        self.extern_cache.insert(key, id);
        id
    }

    /// Allocate a fresh NameId for an extern parameter binder.
    fn mint_extern_param(&mut self, hint: &str, ty: CType) -> CExternParam {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(id, hint.to_string());
        CExternParam {
            name: id,
            original_name: hint.to_string(),
            ty,
        }
    }

    /// Look up the canonical return type of an interned extern by NameId.
    fn extern_return_type(&self, id: NameId) -> CType {
        self.extern_funcs
            .iter()
            .find(|f| f.name == id)
            .map(|f| f.return_type.clone())
            .expect("intern_extern_with_params should have just allocated this NameId")
    }

    /// Build (or fetch from cache) the baked-in `array_len(_xs) >= 0`
    /// refinement that every `list<T>` carries. The handle is created from
    /// the *raw* element type (no recursion through `lower_type`), so the
    /// `array_len` extern's parameter type stays raw `Array(elem)` and we
    /// avoid an infinite descent through the invariant. Callers that pass
    /// a value of refined list type rely on subtyping to forget the inner
    /// invariant when matching the extern's raw parameter signature.
    fn make_list_invariant_handle(&mut self, elem: CType) -> CRefinementHandle {
        if let Some(handle) = self.list_invariant_cache.get(&elem) {
            return handle.clone();
        }
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "_xs".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let xs_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "_xs".to_string()),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem.clone()],
            vec![raw_array],
            CType::Int,
        );
        let len_call = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![xs_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let zero = RefinementExpr {
            kind: RefinementExprKind::IntLit(0),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::GreaterEqual,
                Box::new(len_call),
                Box::new(zero),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        let handle = CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "_xs".to_string(),
            body,
        });
        self.list_invariant_cache.insert(elem, handle.clone());
        handle
    }

    /// Build a refined list type `Refined(Array(elem), { _xs | array_len(_xs)
    /// >= 0 })`. This is the canonical lowered shape of every `list<T>`.
    fn refined_list_ty(&mut self, elem: CType) -> CType {
        let raw = CType::Array(Box::new(elem.clone()));
        let invariant = self.make_list_invariant_handle(elem);
        CType::Refined(Box::new(raw), invariant)
    }

    /// Build the return-type refinement for `array_empty<T>`:
    /// `{ ys | array_len(ys) == 0 }`. The handle's bound is `ys`, of raw
    /// `Array(elem)` type.
    fn make_empty_return_handle(&mut self, elem: CType) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "ys".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let ys_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "ys".to_string()),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem],
            vec![raw_array],
            CType::Int,
        );
        let len_call = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![ys_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let zero = RefinementExpr {
            kind: RefinementExprKind::IntLit(0),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::IntEq,
                Box::new(len_call),
                Box::new(zero),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "ys".to_string(),
            body,
        })
    }

    /// Build the return-type refinement for `array_append<T>` /
    /// `array_prepend<T>`: `{ ys | array_len(ys) == array_len(list_param) + 1
    /// }`, where `list_param` is the input-list parameter NameId of the
    /// *enclosing* extern. The body therefore has a free Var referencing the
    /// extern's parameter binder — see "Free variables in extern return-type
    /// bodies" in the design plan.
    fn make_append_return_handle(
        &mut self,
        elem: CType,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "ys".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let ys_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "ys".to_string()),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let xs_var = RefinementExpr {
            kind: RefinementExprKind::Var(list_param, list_param_name),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem],
            vec![raw_array],
            CType::Int,
        );
        let len_ys = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![ys_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let len_xs = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![xs_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let one = RefinementExpr {
            kind: RefinementExprKind::IntLit(1),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let xs_plus_one = RefinementExpr {
            kind: RefinementExprKind::BinOp(CBinOp::Add, Box::new(len_xs), Box::new(one)),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::IntEq,
                Box::new(len_ys),
                Box::new(xs_plus_one),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "ys".to_string(),
            body,
        })
    }

    /// Build the bounds precondition for `array_index<T>`'s index parameter:
    /// `{ i | 0 <= i && i < array_len(list_param) }`, where `list_param` is
    /// the input-list parameter NameId of the *enclosing* extern. The body
    /// has a free Var referencing the extern's first parameter binder — the
    /// same trick used by `make_append_return_handle`.
    fn make_index_precondition_handle(
        &mut self,
        elem: CType,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "i".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let i_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "i".to_string()),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let xs_var = RefinementExpr {
            kind: RefinementExprKind::Var(list_param, list_param_name),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem],
            vec![raw_array],
            CType::Int,
        );
        let len_xs = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![xs_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let zero = RefinementExpr {
            kind: RefinementExprKind::IntLit(0),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let lower_bound = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::LessEqual,
                Box::new(zero),
                Box::new(i_var.clone()),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        let upper_bound = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::Less,
                Box::new(i_var),
                Box::new(len_xs),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::And,
                Box::new(lower_bound),
                Box::new(upper_bound),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "i".to_string(),
            body,
        })
    }

    /// Build the bounds precondition for `map_index<K, V>`'s key parameter:
    /// `{ k | map_exists(map_param, k) }`, where `map_param` is the input-map
    /// parameter NameId of the *enclosing* extern. The body has a free Var
    /// referencing the extern's first parameter binder.
    fn make_map_index_precondition_handle(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        map_param: NameId,
        map_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "k".to_string());

        let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
        let k_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "k".to_string()),
            ty: k_ty.clone(),
            span: crate::parser::Span::default(),
        };
        let m_var = RefinementExpr {
            kind: RefinementExprKind::Var(map_param, map_param_name),
            ty: map_ty.clone(),
            span: crate::parser::Span::default(),
        };
        let exists_id = self.intern_extern(
            BuiltinKind::MapExists,
            vec![k_ty.clone(), v_ty],
            vec![map_ty, k_ty],
            CType::Bool,
        );
        let body = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: exists_id,
                args: vec![m_var, k_var],
                return_type: CType::Bool,
            },
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "k".to_string(),
            body,
        })
    }

    /// Build the return-type refinement for `map_store<K, V>`:
    /// `{ ys | map_exists(ys, key_param) }`, where `key_param` is the key
    /// parameter NameId of the enclosing extern. The body has a free Var
    /// referencing the key binder.
    fn make_map_store_return_handle(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        key_param: NameId,
        key_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "ys".to_string());

        let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
        let ys_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "ys".to_string()),
            ty: map_ty.clone(),
            span: crate::parser::Span::default(),
        };
        let k_var = RefinementExpr {
            kind: RefinementExprKind::Var(key_param, key_param_name),
            ty: k_ty.clone(),
            span: crate::parser::Span::default(),
        };
        let exists_id = self.intern_extern(
            BuiltinKind::MapExists,
            vec![k_ty.clone(), v_ty],
            vec![map_ty, k_ty],
            CType::Bool,
        );
        let body = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: exists_id,
                args: vec![ys_var, k_var],
                return_type: CType::Bool,
            },
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "ys".to_string(),
            body,
        })
    }

    /// Build the return-type refinement for `map_erase<K, V>`:
    /// `{ ys | !map_exists(ys, key_param) }`.
    fn make_map_erase_return_handle(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        key_param: NameId,
        key_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "ys".to_string());

        let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
        let ys_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "ys".to_string()),
            ty: map_ty.clone(),
            span: crate::parser::Span::default(),
        };
        let k_var = RefinementExpr {
            kind: RefinementExprKind::Var(key_param, key_param_name),
            ty: k_ty.clone(),
            span: crate::parser::Span::default(),
        };
        let exists_id = self.intern_extern(
            BuiltinKind::MapExists,
            vec![k_ty.clone(), v_ty],
            vec![map_ty, k_ty],
            CType::Bool,
        );
        let exists_call = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: exists_id,
                args: vec![ys_var, k_var],
                return_type: CType::Bool,
            },
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::Not(Box::new(exists_call)),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "ys".to_string(),
            body,
        })
    }

    /// Build (or fetch) the `{ _xs | array_len(_xs) >= 1 }` non-empty
    /// precondition shared by `array_head` and `array_tail`.
    fn make_array_nonempty_precondition_handle(&mut self, elem: CType) -> CRefinementHandle {
        if let Some(handle) = self.list_nonempty_cache.get(&elem) {
            return handle.clone();
        }
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "_xs".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let xs_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "_xs".to_string()),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem.clone()],
            vec![raw_array],
            CType::Int,
        );
        let len_call = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![xs_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let one = RefinementExpr {
            kind: RefinementExprKind::IntLit(1),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::GreaterEqual,
                Box::new(len_call),
                Box::new(one),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        let handle = CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "_xs".to_string(),
            body,
        });
        self.list_nonempty_cache.insert(elem, handle.clone());
        handle
    }

    /// Build the return-type refinement for `array_tail<T>`:
    /// `{ ys | array_len(ys) == array_len(list_param) - 1 }`.
    fn make_array_tail_return_handle(
        &mut self,
        elem: CType,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "ys".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let ys_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "ys".to_string()),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let xs_var = RefinementExpr {
            kind: RefinementExprKind::Var(list_param, list_param_name),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem],
            vec![raw_array],
            CType::Int,
        );
        let len_ys = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![ys_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let len_xs = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![xs_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let one = RefinementExpr {
            kind: RefinementExprKind::IntLit(1),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let xs_minus_one = RefinementExpr {
            kind: RefinementExprKind::BinOp(CBinOp::Subtract, Box::new(len_xs), Box::new(one)),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::IntEq,
                Box::new(len_ys),
                Box::new(xs_minus_one),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "ys".to_string(),
            body,
        })
    }

    /// Build the precondition for `array_slice<T>`'s `lo` parameter:
    /// `{ lo | 0 <= lo }`.
    fn make_array_slice_lo_precondition_handle(&mut self) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "lo".to_string());

        let lo_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "lo".to_string()),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let zero = RefinementExpr {
            kind: RefinementExprKind::IntLit(0),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::LessEqual,
                Box::new(zero),
                Box::new(lo_var),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "lo".to_string(),
            body,
        })
    }

    /// Build the precondition for `array_slice<T>`'s `hi` parameter:
    /// `{ hi | lo_param <= hi && hi <= array_len(list_param) }`.
    fn make_array_slice_hi_precondition_handle(
        &mut self,
        elem: CType,
        lo_param: NameId,
        lo_param_name: String,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "hi".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let hi_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "hi".to_string()),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let lo_var = RefinementExpr {
            kind: RefinementExprKind::Var(lo_param, lo_param_name),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let xs_var = RefinementExpr {
            kind: RefinementExprKind::Var(list_param, list_param_name),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem],
            vec![raw_array],
            CType::Int,
        );
        let len_xs = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![xs_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let lower_bound = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::LessEqual,
                Box::new(lo_var),
                Box::new(hi_var.clone()),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        let upper_bound = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::LessEqual,
                Box::new(hi_var),
                Box::new(len_xs),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::And,
                Box::new(lower_bound),
                Box::new(upper_bound),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "hi".to_string(),
            body,
        })
    }

    /// Build the return-type refinement for `array_slice<T>`:
    /// `{ ys | array_len(ys) == hi_param - lo_param }`.
    fn make_array_slice_return_handle(
        &mut self,
        elem: CType,
        lo_param: NameId,
        lo_param_name: String,
        hi_param: NameId,
        hi_param_name: String,
    ) -> CRefinementHandle {
        let bound_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(bound_id, "ys".to_string());

        let raw_array = CType::Array(Box::new(elem.clone()));
        let ys_var = RefinementExpr {
            kind: RefinementExprKind::Var(bound_id, "ys".to_string()),
            ty: raw_array.clone(),
            span: crate::parser::Span::default(),
        };
        let lo_var = RefinementExpr {
            kind: RefinementExprKind::Var(lo_param, lo_param_name),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let hi_var = RefinementExpr {
            kind: RefinementExprKind::Var(hi_param, hi_param_name),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let array_len_id = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem],
            vec![raw_array],
            CType::Int,
        );
        let len_ys = RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target: array_len_id,
                args: vec![ys_var],
                return_type: CType::Int,
            },
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let hi_minus_lo = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::Subtract,
                Box::new(hi_var),
                Box::new(lo_var),
            ),
            ty: CType::Int,
            span: crate::parser::Span::default(),
        };
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::IntEq,
                Box::new(len_ys),
                Box::new(hi_minus_lo),
            ),
            ty: CType::Bool,
            span: crate::parser::Span::default(),
        };
        CRefinementHandle::new(CRefinementBody {
            bound: bound_id,
            original_bound: "ys".to_string(),
            body,
        })
    }

    fn atomic_p_type(&self, atom: &PAtomic) -> Type {
        match atom {
            PAtomic::Var(id, _) => self
                .var_types
                .get(id)
                .cloned()
                .unwrap_or(Type::Error),
            PAtomic::IntLit(_) => Type::Int,
            PAtomic::StringLit(_) => Type::String,
            PAtomic::BoolLit(_) => Type::Bool,
            PAtomic::NilLit => Type::Nil,
            PAtomic::Never => Type::Never,
        }
    }

    fn array_elem_of(&self, atom: &PAtomic) -> CType {
        match self.atomic_p_type(atom) {
            Type::List(t) => lower_type_simple(&t),
            _ => CType::Never,
        }
    }

    fn map_kv_of(&self, atom: &PAtomic) -> (CType, CType) {
        match self.atomic_p_type(atom) {
            Type::Map(k, v) => (lower_type_simple(&k), lower_type_simple(&v)),
            _ => (CType::Never, CType::Never),
        }
    }

    fn optional_inner_of(&self, atom: &PAtomic) -> CType {
        match self.atomic_p_type(atom) {
            Type::Optional(t) => lower_type_simple(&t),
            _ => CType::Never,
        }
    }

    fn chan_inner_of(&self, atom: &PAtomic) -> CType {
        match self.atomic_p_type(atom) {
            Type::Chan(t) => lower_type_simple(&t),
            _ => CType::Never,
        }
    }

    fn iter_inner_of(&self, atom: &PAtomic) -> CType {
        match self.atomic_p_type(atom) {
            Type::Iter(t) => lower_type_simple(&t),
            _ => CType::Never,
        }
    }
}

fn lower_atomic(a: PAtomic) -> CAtomic {
    match a {
        PAtomic::Var(id, name) => CAtomic::Var(id, name),
        PAtomic::IntLit(v) => CAtomic::IntLit(v),
        PAtomic::StringLit(v) => CAtomic::StringLit(v),
        PAtomic::BoolLit(v) => CAtomic::BoolLit(v),
        PAtomic::NilLit => CAtomic::NilLit,
        PAtomic::Never => CAtomic::Never,
    }
}

/// Strip outer `Refined` wrappers and pattern-match on `Array(elem)`. Returns
/// `Some(elem)` when `ty` is (transitively) an array, `None` otherwise. Used
/// to recover the element type of an empty list literal from its expected
/// type after the baked-in `len_geq_0` invariant has been applied.
fn extract_array_elem(ty: &CType) -> Option<CType> {
    let mut cur = ty;
    loop {
        match cur {
            CType::Array(elem) => return Some((**elem).clone()),
            CType::Refined(inner, _) => cur = inner,
            _ => return None,
        }
    }
}

/// Pure type lowering used by read-only helper methods (no body_memo needed).
fn lower_type_simple(ty: &Type) -> CType {
    match ty {
        Type::Int => CType::Int,
        Type::String => CType::String,
        Type::Bool => CType::Bool,
        Type::List(t) => CType::Array(Box::new(lower_type_simple(t))),
        Type::Map(k, v) => CType::Map(Box::new(lower_type_simple(k)), Box::new(lower_type_simple(v))),
        Type::Tuple(ts) => CType::Tuple(ts.iter().map(lower_type_simple).collect()),
        Type::Struct(id, _) => CType::Struct(*id),
        Type::Enum(id, _) => CType::Variant(*id),
        Type::Role(id, _) => CType::Role(*id),
        Type::Optional(t) => CType::Optional(Box::new(lower_type_simple(t))),
        Type::Chan(t) => CType::Chan(Box::new(lower_type_simple(t))),
        Type::FifoLink(t) => CType::FifoLink(Box::new(lower_type_simple(t))),
        Type::Iter(t) => CType::Iter(Box::new(lower_type_simple(t))),
        Type::Refined(inner, _) => lower_type_simple(inner),
        Type::EmptyList => CType::Array(Box::new(CType::Never)),
        Type::EmptyMap => CType::Map(Box::new(CType::Never), Box::new(CType::Never)),
        Type::UnknownChannel => CType::Chan(Box::new(CType::Never)),
        Type::Nil => CType::Nil,
        Type::Never => CType::Never,
        Type::Error => CType::Never,
    }
}

impl CoreLowerer {
    pub(crate) fn lower_type(&mut self, ty: &Type) -> CType {
        match ty {
            Type::Int => CType::Int,
            Type::String => CType::String,
            Type::Bool => CType::Bool,
            Type::List(t) => {
                let elem = self.lower_type(t);
                self.refined_list_ty(elem)
            }
            Type::Map(k, v) => {
                let k = self.lower_type(k);
                let v = self.lower_type(v);
                CType::Map(Box::new(k), Box::new(v))
            }
            Type::Tuple(ts) => CType::Tuple(ts.iter().map(|t| self.lower_type(t)).collect()),
            Type::Struct(id, _) => CType::Struct(*id),
            Type::Enum(id, _) => CType::Variant(*id),
            Type::Role(id, _) => CType::Role(*id),
            Type::Optional(t) => CType::Optional(Box::new(self.lower_type(t))),
            Type::Chan(t) => CType::Chan(Box::new(self.lower_type(t))),
            Type::FifoLink(t) => CType::FifoLink(Box::new(self.lower_type(t))),
            Type::Iter(t) => CType::Iter(Box::new(self.lower_type(t))),
            Type::Refined(inner, handle) => {
                let ci = self.lower_type(inner);
                let key: *const RefinementBody = handle.as_ptr();
                if let Some(cached) = self.body_memo.get(&key) {
                    return CType::Refined(Box::new(ci), cached.clone());
                }
                let body_src = handle.body.clone();
                let body = self.lower_refinement_expr(&body_src);
                let cbody = CRefinementHandle::new(CRefinementBody {
                    bound: handle.bound,
                    original_bound: handle.original_bound.clone(),
                    body,
                });
                self.body_memo.insert(key, cbody.clone());
                CType::Refined(Box::new(ci), cbody)
            }
            Type::EmptyList => self.refined_list_ty(CType::Never),
            Type::EmptyMap => CType::Map(Box::new(CType::Never), Box::new(CType::Never)),
            Type::UnknownChannel => CType::Chan(Box::new(CType::Never)),
            Type::Nil => CType::Nil,
            Type::Never => CType::Never,
            Type::Error => CType::Never,
        }
    }
}

impl CoreLowerer {
    /// Lower a type-checker `TypedExpr` (the body of a refinement) into the
    /// pure-only `RefinementExpr` IR. Builtin operations are routed through
    /// `intern_extern`, sharing the program's `extern_funcs`/`extern_cache`.
    fn lower_refinement_expr(&mut self, expr: &TypedExpr) -> RefinementExpr {
        let span = expr.span;
        let result_ty = self.lower_type(&expr.ty);

        let kind = match &expr.kind {
            TypedExprKind::Var(id, name) => RefinementExprKind::Var(*id, name.clone()),
            TypedExprKind::IntLit(v) => RefinementExprKind::IntLit(*v),
            TypedExprKind::StringLit(v) => RefinementExprKind::StringLit(v.clone()),
            TypedExprKind::BoolLit(v) => RefinementExprKind::BoolLit(*v),
            TypedExprKind::NilLit => RefinementExprKind::NilLit,

            TypedExprKind::BinOp(op, l, r) => match op {
                BinOp::Equal | BinOp::NotEqual => {
                    let operand_ty = self.lower_type(&l.ty);
                    let prim_op = match (&operand_ty, op) {
                        (CType::Int, BinOp::Equal) => Some(CBinOp::IntEq),
                        (CType::Int, BinOp::NotEqual) => Some(CBinOp::IntNeq),
                        (CType::Bool, BinOp::Equal) => Some(CBinOp::BoolEq),
                        (CType::Bool, BinOp::NotEqual) => Some(CBinOp::BoolNeq),
                        _ => None,
                    };
                    if let Some(prim_op) = prim_op {
                        let l = Box::new(self.lower_refinement_expr(l));
                        let r = Box::new(self.lower_refinement_expr(r));
                        RefinementExprKind::BinOp(prim_op, l, r)
                    } else {
                        let kind = if *op == BinOp::Equal {
                            BuiltinKind::Eq
                        } else {
                            BuiltinKind::Neq
                        };
                        let target = self.intern_extern(
                            kind,
                            vec![operand_ty.clone()],
                            vec![operand_ty.clone(), operand_ty],
                            CType::Bool,
                        );
                        let l = self.lower_refinement_expr(l);
                        let r = self.lower_refinement_expr(r);
                        RefinementExprKind::ExternCall {
                            target,
                            args: vec![l, r],
                            return_type: CType::Bool,
                        }
                    }
                }
                _ => {
                    let l = Box::new(self.lower_refinement_expr(l));
                    let r = Box::new(self.lower_refinement_expr(r));
                    RefinementExprKind::BinOp(to_cbinop(op), l, r)
                }
            },
            TypedExprKind::Not(e) => {
                let e = Box::new(self.lower_refinement_expr(e));
                RefinementExprKind::Not(e)
            }
            TypedExprKind::Negate(e) => {
                let e = Box::new(self.lower_refinement_expr(e));
                RefinementExprKind::Negate(e)
            }

            TypedExprKind::FuncCall(call) => match call {
                TypedFuncCall::User(u) => {
                    self.refinement_errors.push(RefinementValidationError {
                        kind: RefinementValidationErrorKind::UserFunctionCall(
                            u.original_name.clone(),
                        ),
                        span: u.span,
                    });
                    RefinementExprKind::Error
                }
                TypedFuncCall::Builtin(b, args, ret) => {
                    let kind = match b {
                        BuiltinFn::Println => BuiltinKind::Println,
                        BuiltinFn::IntToString => BuiltinKind::IntToString,
                        BuiltinFn::BoolToString => BuiltinKind::BoolToString,
                        BuiltinFn::RoleToString => BuiltinKind::RoleToString,
                        BuiltinFn::UniqueId => BuiltinKind::UniqueId,
                    };
                    let arg_types: Vec<CType> =
                        args.iter().map(|a| self.lower_type(&a.ty)).collect();
                    let ret_c = self.lower_type(ret);
                    let target = self.intern_extern(kind, vec![], arg_types, ret_c.clone());
                    let lowered_args =
                        args.iter().map(|a| self.lower_refinement_expr(a)).collect();
                    RefinementExprKind::ExternCall {
                        target,
                        args: lowered_args,
                        return_type: ret_c,
                    }
                }
            },

            TypedExprKind::MapLit(pairs) => {
                let lowered: Vec<(RefinementExpr, RefinementExpr)> = pairs
                    .iter()
                    .map(|(k, v)| {
                        (self.lower_refinement_expr(k), self.lower_refinement_expr(v))
                    })
                    .collect();
                RefinementExprKind::MapLit(lowered)
            }
            TypedExprKind::ListLit(es) => {
                let elem = if let Some(first) = es.first() {
                    self.lower_type(&first.ty)
                } else {
                    let lowered_outer = self.lower_type(&expr.ty);
                    extract_array_elem(&lowered_outer).unwrap_or(CType::Never)
                };
                let mut current = self.lower_array_empty_refinement(elem.clone());
                for item in es {
                    let item_l = self.lower_refinement_expr(item);
                    let elem_for_builder = elem.clone();
                    let raw_array = CType::Array(Box::new(elem.clone()));
                    let target = self.intern_extern_with_params(
                        BuiltinKind::ArrayAppend,
                        vec![elem.clone()],
                        vec![
                            ("xs".to_string(), raw_array),
                            ("e".to_string(), elem.clone()),
                        ],
                        move |this, params| {
                            let inner_invariant =
                                this.make_list_invariant_handle(elem_for_builder.clone());
                            let inner_refined = CType::Refined(
                                Box::new(CType::Array(Box::new(elem_for_builder.clone()))),
                                inner_invariant,
                            );
                            let outer = this.make_append_return_handle(
                                elem_for_builder,
                                params[0].name,
                                params[0].original_name.clone(),
                            );
                            CType::Refined(Box::new(inner_refined), outer)
                        },
                    );
                    let return_type = self.extern_return_type(target);
                    current = RefinementExpr {
                        kind: RefinementExprKind::ExternCall {
                            target,
                            args: vec![current, item_l],
                            return_type: return_type.clone(),
                        },
                        ty: return_type,
                        span,
                    };
                }
                return current;
            }
            TypedExprKind::TupleLit(es) => RefinementExprKind::TupleLit(
                es.iter().map(|e| self.lower_refinement_expr(e)).collect(),
            ),

            TypedExprKind::Append(list, item) => {
                self.lower_array_extern(BuiltinKind::ArrayAppend, list, item)
            }
            TypedExprKind::Prepend(list, item) => {
                self.lower_array_extern(BuiltinKind::ArrayPrepend, list, item)
            }
            TypedExprKind::Head(list) => {
                let elem = match &list.ty {
                    Type::List(t) => self.lower_type(t),
                    _ => CType::Never,
                };
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                let target = self.intern_extern(
                    BuiltinKind::ArrayHead,
                    vec![elem.clone()],
                    vec![list_ty],
                    elem.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![list_l],
                    return_type: elem,
                }
            }
            TypedExprKind::Tail(list) => {
                let elem = match &list.ty {
                    Type::List(t) => self.lower_type(t),
                    _ => CType::Never,
                };
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                let target = self.intern_extern(
                    BuiltinKind::ArrayTail,
                    vec![elem],
                    vec![list_ty.clone()],
                    list_ty.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![list_l],
                    return_type: list_ty,
                }
            }
            TypedExprKind::Len(list) => {
                let elem = match &list.ty {
                    Type::List(t) => self.lower_type(t),
                    _ => CType::Never,
                };
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                let target =
                    self.intern_extern(BuiltinKind::ArrayLen, vec![elem], vec![list_ty], CType::Int);
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![list_l],
                    return_type: CType::Int,
                }
            }
            TypedExprKind::Slice(list, lo, hi) => {
                let elem = match &list.ty {
                    Type::List(t) => self.lower_type(t),
                    _ => CType::Never,
                };
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                let lo_l = self.lower_refinement_expr(lo);
                let hi_l = self.lower_refinement_expr(hi);
                let target = self.intern_extern(
                    BuiltinKind::ArraySlice,
                    vec![elem],
                    vec![list_ty.clone(), CType::Int, CType::Int],
                    list_ty.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![list_l, lo_l, hi_l],
                    return_type: list_ty,
                }
            }
            TypedExprKind::Min(a, b) => {
                let ty = self.lower_type(&a.ty);
                let a_l = self.lower_refinement_expr(a);
                let b_l = self.lower_refinement_expr(b);
                let target = self.intern_extern(
                    BuiltinKind::Min,
                    vec![ty.clone()],
                    vec![ty.clone(), ty.clone()],
                    ty.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![a_l, b_l],
                    return_type: ty,
                }
            }
            TypedExprKind::Exists(map, key) => {
                let (k, v) = match &map.ty {
                    Type::Map(k, v) => (self.lower_type(k), self.lower_type(v)),
                    _ => (CType::Never, CType::Never),
                };
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                let map_l = self.lower_refinement_expr(map);
                let key_l = self.lower_refinement_expr(key);
                let target = self.intern_extern(
                    BuiltinKind::MapExists,
                    vec![k.clone(), v],
                    vec![map_ty, k],
                    CType::Bool,
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![map_l, key_l],
                    return_type: CType::Bool,
                }
            }
            TypedExprKind::Erase(map, key) => {
                let (k, v) = match &map.ty {
                    Type::Map(k, v) => (self.lower_type(k), self.lower_type(v)),
                    _ => (CType::Never, CType::Never),
                };
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                let map_l = self.lower_refinement_expr(map);
                let key_l = self.lower_refinement_expr(key);
                let target = self.intern_extern(
                    BuiltinKind::MapErase,
                    vec![k.clone(), v],
                    vec![map_ty.clone(), k],
                    map_ty.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![map_l, key_l],
                    return_type: map_ty,
                }
            }
            TypedExprKind::Store(map, key, val) => {
                let (k, v) = match &map.ty {
                    Type::Map(k, v) => (self.lower_type(k), self.lower_type(v)),
                    _ => (CType::Never, CType::Never),
                };
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                let map_l = self.lower_refinement_expr(map);
                let key_l = self.lower_refinement_expr(key);
                let val_l = self.lower_refinement_expr(val);
                let target = self.intern_extern(
                    BuiltinKind::MapStore,
                    vec![k.clone(), v.clone()],
                    vec![map_ty.clone(), k, v],
                    map_ty.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![map_l, key_l, val_l],
                    return_type: map_ty,
                }
            }

            TypedExprKind::UnwrapOptional(e) => {
                let inner = match &e.ty {
                    Type::Optional(t) => self.lower_type(t),
                    _ => CType::Never,
                };
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                let e_l = self.lower_refinement_expr(e);
                let target = self.intern_extern(
                    BuiltinKind::OptionalUnwrap,
                    vec![inner.clone()],
                    vec![opt_ty],
                    inner.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![e_l],
                    return_type: inner,
                }
            }
            TypedExprKind::WrapInOptional(e) => {
                let inner = self.lower_type(&e.ty);
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                let e_l = self.lower_refinement_expr(e);
                let target = self.intern_extern(
                    BuiltinKind::OptionalWrap,
                    vec![inner.clone()],
                    vec![inner],
                    opt_ty.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![e_l],
                    return_type: opt_ty,
                }
            }

            TypedExprKind::Index(coll, idx) => {
                let coll_l = self.lower_refinement_expr(coll);
                let idx_l = self.lower_refinement_expr(idx);
                match &coll.ty {
                    Type::List(t) => {
                        let elem = self.lower_type(t);
                        let raw_array = CType::Array(Box::new(elem.clone()));
                        let target = self.intern_extern(
                            BuiltinKind::ArrayIndex,
                            vec![elem.clone()],
                            vec![raw_array, CType::Int],
                            elem.clone(),
                        );
                        RefinementExprKind::ExternCall {
                            target,
                            args: vec![coll_l, idx_l],
                            return_type: elem,
                        }
                    }
                    Type::Map(k, v) => {
                        let k_c = self.lower_type(k);
                        let v_c = self.lower_type(v);
                        let map_ty =
                            CType::Map(Box::new(k_c.clone()), Box::new(v_c.clone()));
                        let target = self.intern_extern(
                            BuiltinKind::MapIndex,
                            vec![k_c.clone(), v_c.clone()],
                            vec![map_ty, k_c],
                            v_c.clone(),
                        );
                        RefinementExprKind::ExternCall {
                            target,
                            args: vec![coll_l, idx_l],
                            return_type: v_c,
                        }
                    }
                    _ => {
                        let _ = (coll_l, idx_l);
                        RefinementExprKind::Error
                    }
                }
            }
            TypedExprKind::TupleAccess(t, i) => {
                let t_l = Box::new(self.lower_refinement_expr(t));
                RefinementExprKind::TupleAccess(t_l, *i)
            }
            TypedExprKind::FieldAccess(s, f) => {
                let s_l = Box::new(self.lower_refinement_expr(s));
                RefinementExprKind::FieldAccess(s_l, f.clone())
            }
            TypedExprKind::SafeFieldAccess(s, field) => {
                let struct_ty = self.lower_type(&s.ty);
                let ret_c = self.lower_type(&expr.ty);
                let s_l = self.lower_refinement_expr(s);
                let target = self.intern_extern(
                    BuiltinKind::SafeField(field.clone()),
                    vec![struct_ty.clone(), ret_c.clone()],
                    vec![struct_ty],
                    ret_c.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![s_l],
                    return_type: ret_c,
                }
            }
            TypedExprKind::SafeIndex(coll, idx) => {
                let coll_ty = self.lower_type(&coll.ty);
                let idx_ty = self.lower_type(&idx.ty);
                let ret_c = self.lower_type(&expr.ty);
                let coll_l = self.lower_refinement_expr(coll);
                let idx_l = self.lower_refinement_expr(idx);
                let target = self.intern_extern(
                    BuiltinKind::SafeIndex,
                    vec![coll_ty.clone(), idx_ty.clone(), ret_c.clone()],
                    vec![coll_ty, idx_ty],
                    ret_c.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![coll_l, idx_l],
                    return_type: ret_c,
                }
            }
            TypedExprKind::SafeTupleAccess(t, i) => {
                let tuple_ty = self.lower_type(&t.ty);
                let ret_c = self.lower_type(&expr.ty);
                let t_l = self.lower_refinement_expr(t);
                let target = self.intern_extern(
                    BuiltinKind::SafeTupleAccess(*i),
                    vec![tuple_ty.clone(), ret_c.clone()],
                    vec![tuple_ty],
                    ret_c.clone(),
                );
                RefinementExprKind::ExternCall {
                    target,
                    args: vec![t_l],
                    return_type: ret_c,
                }
            }

            TypedExprKind::StructLit(id, fields) => RefinementExprKind::StructLit(
                *id,
                fields
                    .iter()
                    .map(|(n, e)| (n.clone(), self.lower_refinement_expr(e)))
                    .collect(),
            ),
            TypedExprKind::VariantLit(id, name, payload) => RefinementExprKind::VariantLit(
                *id,
                name.clone(),
                payload
                    .as_ref()
                    .map(|p| Box::new(self.lower_refinement_expr(p))),
            ),

            TypedExprKind::Conditional(cond) => {
                let lowered = self.lower_refinement_cond(cond);
                RefinementExprKind::Conditional(Box::new(lowered))
            }
            TypedExprKind::Block(block) => {
                if let Some(expr) = self.lower_refinement_block(block, span) {
                    return expr;
                } else {
                    RefinementExprKind::Error
                }
            }

            TypedExprKind::Error => RefinementExprKind::Error,

            TypedExprKind::RpcCall(_, call) => {
                self.refinement_errors.push(RefinementValidationError {
                    kind: RefinementValidationErrorKind::DisallowedExpression("rpc call"),
                    span: call.span,
                });
                RefinementExprKind::Error
            }
            TypedExprKind::Match(_, _) => {
                self.refinement_errors.push(RefinementValidationError {
                    kind: RefinementValidationErrorKind::DisallowedExpression("match expression"),
                    span,
                });
                RefinementExprKind::Error
            }
            TypedExprKind::MakeChannel => self.disallowed("make-channel", span),
            TypedExprKind::Send(_, _) => self.disallowed("channel send", span),
            TypedExprKind::Recv(_) => self.disallowed("channel recv", span),
            TypedExprKind::SetTimer(_) => self.disallowed("set-timer", span),
            TypedExprKind::Fifo(_) => self.disallowed("fifo", span),
            TypedExprKind::PersistData(_) => self.disallowed("persist-data", span),
            TypedExprKind::RetrieveData(_) => self.disallowed("retrieve-data", span),
            TypedExprKind::DiscardData => self.disallowed("discard-data", span),
            TypedExprKind::Return(_) => self.disallowed("return", span),
            TypedExprKind::Break => self.disallowed("break", span),
            TypedExprKind::Continue => self.disallowed("continue", span),
        };

        RefinementExpr {
            kind,
            ty: result_ty,
            span,
        }
    }

    fn disallowed(&mut self, what: &'static str, span: crate::parser::Span) -> RefinementExprKind {
        self.refinement_errors.push(RefinementValidationError {
            kind: RefinementValidationErrorKind::DisallowedExpression(what),
            span,
        });
        RefinementExprKind::Error
    }

    /// Helper for `array_append` / `array_prepend` in refinement bodies. The
    /// extern's return type is the layered refined list:
    /// `Refined(Refined(Array(elem), len_geq_0), { ys | array_len(ys) ==
    /// array_len(xs) + 1 })` — the outer refinement references the input
    /// list parameter's `NameId` minted on cache miss inside `intern_extern_
    /// with_params`.
    fn lower_array_extern(
        &mut self,
        kind: BuiltinKind,
        list: &TypedExpr,
        item: &TypedExpr,
    ) -> RefinementExprKind {
        let elem = match &list.ty {
            Type::List(t) => self.lower_type(t),
            _ => CType::Never,
        };
        let raw_array = CType::Array(Box::new(elem.clone()));

        let list_l = self.lower_refinement_expr(list);
        let item_l = self.lower_refinement_expr(item);

        let target = {
            let elem_for_builder = elem.clone();
            self.intern_extern_with_params(
                kind,
                vec![elem.clone()],
                vec![
                    ("xs".to_string(), raw_array.clone()),
                    ("e".to_string(), elem.clone()),
                ],
                move |this, params| {
                    let inner_invariant =
                        this.make_list_invariant_handle(elem_for_builder.clone());
                    let inner_refined = CType::Refined(
                        Box::new(CType::Array(Box::new(elem_for_builder.clone()))),
                        inner_invariant,
                    );
                    let outer = this.make_append_return_handle(
                        elem_for_builder,
                        params[0].name,
                        params[0].original_name.clone(),
                    );
                    CType::Refined(Box::new(inner_refined), outer)
                },
            )
        };
        let return_type = self.extern_return_type(target);
        RefinementExprKind::ExternCall {
            target,
            args: vec![list_l, item_l],
            return_type,
        }
    }

    /// Build a refinement-side `array_empty<T>()` ExternCall expression. The
    /// extern's return type is `Refined(Refined(Array(elem), len_geq_0),
    /// { ys | array_len(ys) == 0 })`.
    fn lower_array_empty_refinement(&mut self, elem: CType) -> RefinementExpr {
        let elem_for_builder = elem.clone();
        let target = self.intern_extern_with_params(
            BuiltinKind::ArrayEmpty,
            vec![elem.clone()],
            vec![],
            move |this, _params| {
                let inner_invariant = this.make_list_invariant_handle(elem_for_builder.clone());
                let inner_refined = CType::Refined(
                    Box::new(CType::Array(Box::new(elem_for_builder.clone()))),
                    inner_invariant,
                );
                let outer = this.make_empty_return_handle(elem_for_builder);
                CType::Refined(Box::new(inner_refined), outer)
            },
        );
        let return_type = self.extern_return_type(target);
        RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target,
                args: vec![],
                return_type: return_type.clone(),
            },
            ty: return_type,
            span: crate::parser::Span::default(),
        }
    }

    /// Build a statement-side `array_empty<T>()` FuncCall. Returns a `CExpr`
    /// suitable for use as a let-atom value. The extern's return type matches
    /// the refinement-side construction (layered refined list).
    fn lower_array_empty_stmt(&mut self, elem: CType, span: crate::parser::Span) -> CExpr {
        let elem_for_builder = elem.clone();
        let kind = self.emit_extern_call_with_params(
            BuiltinKind::ArrayEmpty,
            vec![elem.clone()],
            vec![],
            move |this, _params| {
                let inner_invariant = this.make_list_invariant_handle(elem_for_builder.clone());
                let inner_refined = CType::Refined(
                    Box::new(CType::Array(Box::new(elem_for_builder.clone()))),
                    inner_invariant,
                );
                let outer = this.make_empty_return_handle(elem_for_builder);
                CType::Refined(Box::new(inner_refined), outer)
            },
            vec![],
        );
        let ty = match &kind {
            CExprKind::FuncCall(call) => call.return_type.clone(),
            _ => unreachable!("emit_extern_call_with_params returns FuncCall"),
        };
        CExpr { kind, ty, span }
    }

    fn lower_refinement_cond(&mut self, cond: &TypedCondExpr) -> RefinementCond {
        let if_branch = RefinementIfBranch {
            condition: self.lower_refinement_expr(&cond.if_branch.condition),
            body: self
                .lower_refinement_block_or_error(&cond.if_branch.body, cond.if_branch.body.span),
            span: cond.if_branch.span,
        };
        let elseif_branches = cond
            .elseif_branches
            .iter()
            .map(|b| RefinementIfBranch {
                condition: self.lower_refinement_expr(&b.condition),
                body: self.lower_refinement_block_or_error(&b.body, b.body.span),
                span: b.span,
            })
            .collect();
        let else_branch = cond
            .else_branch
            .as_ref()
            .map(|b| self.lower_refinement_block_or_error(b, b.span));
        RefinementCond {
            if_branch,
            elseif_branches,
            else_branch,
            span: cond.span,
        }
    }

    /// Refinement bodies don't allow statements; a `TypedBlock` here is
    /// expected to be a tail-only expression (e.g. the body of an `if`
    /// branch). If the block has any statements we record an error.
    fn lower_refinement_block(
        &mut self,
        block: &TypedBlock,
        span: crate::parser::Span,
    ) -> Option<RefinementExpr> {
        if !block.statements.is_empty() {
            self.refinement_errors.push(RefinementValidationError {
                kind: RefinementValidationErrorKind::DisallowedExpression(
                    "statement inside refinement body",
                ),
                span,
            });
            return None;
        }
        block
            .tail_expr
            .as_ref()
            .map(|e| self.lower_refinement_expr(e))
    }

    fn lower_refinement_block_or_error(
        &mut self,
        block: &TypedBlock,
        span: crate::parser::Span,
    ) -> RefinementExpr {
        match self.lower_refinement_block(block, span) {
            Some(e) => e,
            None => RefinementExpr {
                kind: RefinementExprKind::Error,
                ty: self.lower_type(&block.ty),
                span,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefinementValidationError {
    pub kind: RefinementValidationErrorKind,
    pub span: crate::parser::Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RefinementValidationErrorKind {
    /// A user function was called inside a refinement body.
    UserFunctionCall(String),
    /// Some other side-effecting or non-pure construct (RPC, channel op,
    /// persistence, control flow, etc.) appeared inside a refinement body.
    DisallowedExpression(&'static str),
    /// A refinement body contains `*`, `/`, or `%` whose operands are both
    /// non-constant. 
    NonLinearArithmetic { op: CBinOp },
}

impl std::fmt::Display for RefinementValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            RefinementValidationErrorKind::UserFunctionCall(name) => {
                write!(f, "user function '{}' called in refinement body", name)
            }
            RefinementValidationErrorKind::DisallowedExpression(desc) => {
                write!(f, "{} is not allowed in a refinement body", desc)
            }
            RefinementValidationErrorKind::NonLinearArithmetic { op } => {
                let sym = match op {
                    CBinOp::Multiply => "*",
                    CBinOp::Divide => "/",
                    CBinOp::Modulo => "%",
                    _ => "?",
                };
                write!(
                    f,
                    "non-linear arithmetic ('{}' requires at least one constant operand) is not allowed in a refinement body",
                    sym
                )
            }
        }
    }
}

fn format_extern_name(kind: &BuiltinKind, type_args: &[CType]) -> String {
    if type_args.is_empty() {
        kind.base_name()
    } else {
        let args = type_args
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}<{}>", kind.base_name(), args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Span;

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

    use crate::analysis::types::{
        RefinementBody, RefinementHandle, TypedExpr, TypedExprKind, TypedUserFuncCall,
    };
    use crate::parser::BinOp;

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
            crate::analysis::types::TypedFuncCall::User(TypedUserFuncCall {
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
}
