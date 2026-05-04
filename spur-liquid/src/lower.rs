use std::collections::HashMap;

use spur_ast::binop::BinOp;
use spur_ast::name::{BuiltinFn, NameId};
use spur_ast::pure::*;
use spur_ast::types::{
    RefinementBody, Type, TypedBlock, TypedCondExpr, TypedExpr, TypedExprKind, TypedFuncCall,
};

use crate::builtins::BuiltinKind;
use crate::ir::*;
use crate::refinement::{
    RefinementCond, RefinementExpr, RefinementExprKind, RefinementIfBranch,
};

/// Build a dependent-parameter type closure that returns a fixed `CType`,
/// ignoring the lowerer and any earlier params. Used at `intern_extern_with_
/// dependent_params` call sites where some parameters carry no refinement
/// and are just plain types.
fn const_param(ty: CType) -> Box<dyn FnOnce(&mut CoreLowerer, &[CExternParam]) -> CType> {
    Box::new(move |_, _| ty)
}

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
            .map(|(field_id, name, ty)| (*field_id, name.clone(), lowerer.lower_type(ty)))
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
            .map(|(variant_id, name, payload)| (*variant_id, name.clone(), payload.as_ref().map(|p| lowerer.lower_type(p))))
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
    refinement_errors.extend(crate::validate::validate_refinements(&program));
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
    struct_defs: HashMap<NameId, Vec<(NameId, String, CType)>>,
    enum_defs: HashMap<NameId, Vec<(NameId, String, Option<CType>)>>,
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
            PExprKind::MapLit(pairs) => self.desugar_map_lit_stmt(pairs, result_ty),

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
                let k_for_post = k.clone();
                let v_for_post = v.clone();
                let target = self.intern_extern_with_dependent_params(
                    BuiltinKind::MapErase,
                    vec![k.clone(), v],
                    vec![
                        ("m".to_string(), const_param(map_ty.clone())),
                        ("k".to_string(), const_param(k.clone())),
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
            PExprKind::Store(receiver, key, val) => {
                match self.atomic_p_type(&receiver) {
                    Type::Struct(struct_id, _) => {
                        self.lower_struct_field_store(struct_id, receiver, key, val)
                    }
                    _ => {
                        let (k, v) = self.map_kv_of(&receiver);
                        let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                        let k_for_post = k.clone();
                        let v_for_post = v.clone();
                        let target = self.intern_extern_with_dependent_params(
                            BuiltinKind::MapStore,
                            vec![k.clone(), v.clone()],
                            vec![
                                ("m".to_string(), const_param(map_ty.clone())),
                                ("k".to_string(), const_param(k.clone())),
                                ("v".to_string(), const_param(v.clone())),
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
                            args: vec![lower_atomic(receiver), lower_atomic(key), lower_atomic(val)],
                            return_type,
                        })
                    }
                }
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
                        ("xs".to_string(), const_param(raw_array)),
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
                let inner_ty = self.lower_type(&call.return_type);
                let ret_ty = CType::Chan(Box::new(inner_ty));
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

            PExprKind::VariantLit(enum_id, variant_id, payload) => {
                CExprKind::VariantLit(enum_id, variant_id, payload.map(lower_atomic))
            }
            PExprKind::IsVariant(a, enum_id, variant_id) => {
                CExprKind::IsVariant(lower_atomic(a), enum_id, variant_id)
            }
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
                    vec![state_ty.clone(), elem.clone()],
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
                    vec![state_ty.clone(), elem],
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
                                ("xs".to_string(), const_param(raw_array)),
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
                                ("m".to_string(), const_param(map_ty)),
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
        let span = spur_ast::span::Span::default();

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

    /// Desugar `{k1: v1, k2: v2, ...}` (statement-side) into a chain of
    /// `map_empty` + `map_store` calls. Mirrors `desugar_list_lit_stmt`. The
    /// empty case `{}` pushes a single `map_empty` let and returns its var.
    fn desugar_map_lit_stmt(
        &mut self,
        pairs: Vec<(PAtomic, PAtomic)>,
        result_ty: &Type,
    ) -> CExprKind {
        let (k_ty, v_ty) = match result_ty {
            Type::Map(k, v) => (self.lower_type(k), self.lower_type(v)),
            Type::EmptyMap => (CType::Never, CType::Never),
            other => match self.lower_type(other) {
                CType::Map(k, v) => ((*k).clone(), (*v).clone()),
                _ => (CType::Never, CType::Never),
            },
        };
        let span = spur_ast::span::Span::default();

        let empty_value = self.lower_map_empty_stmt(k_ty.clone(), v_ty.clone(), span);
        let mut current_id = NameId(self.next_name_id);
        self.next_name_id += 1;
        let current_orig = format!("_map_lit{}", current_id.0);
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

        for (k_atom, v_atom) in pairs {
            let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
            let k_for_post = k_ty.clone();
            let v_for_post = v_ty.clone();
            let target = self.intern_extern_with_dependent_params(
                BuiltinKind::MapStore,
                vec![k_ty.clone(), v_ty.clone()],
                vec![
                    ("m".to_string(), const_param(map_ty.clone())),
                    ("k".to_string(), const_param(k_ty.clone())),
                    ("v".to_string(), const_param(v_ty.clone())),
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
            let kind = CExprKind::FuncCall(CFuncCall {
                target,
                args: vec![
                    CAtomic::Var(current_id, format!("_map_lit{}", current_id.0)),
                    lower_atomic(k_atom),
                    lower_atomic(v_atom),
                ],
                return_type: return_type.clone(),
            });
            let value = CExpr {
                kind,
                ty: return_type.clone(),
                span,
            };

            let next_id = NameId(self.next_name_id);
            self.next_name_id += 1;
            let next_orig = format!("_map_lit{}", next_id.0);
            self.id_to_name.insert(next_id, next_orig.clone());
            self.pending_stmts.push(CStatement {
                kind: CStatementKind::LetAtom(CLetAtom {
                    name: next_id,
                    original_name: next_orig,
                    ty: return_type.clone(),
                    value,
                    user_annotated: false,
                    span,
                }),
                span,
            });
            current_id = next_id;
            current_ty = return_type;
        }
        let _ = current_ty;
        CExprKind::Atomic(CAtomic::Var(current_id, format!("_map_lit{}", current_id.0)))
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

    

    fn r_var(&self, id: NameId, name: &str, ty: CType) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::Var(id, name.to_string()),
            ty,
            span: spur_ast::span::Span::default(),
        }
    }

    fn r_int(&self, n: i64) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::IntLit(n),
            ty: CType::Int,
            span: spur_ast::span::Span::default(),
        }
    }

    fn r_binop(
        &self,
        op: CBinOp,
        l: RefinementExpr,
        r: RefinementExpr,
        ty: CType,
    ) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::BinOp(op, Box::new(l), Box::new(r)),
            ty,
            span: spur_ast::span::Span::default(),
        }
    }

    fn r_extern_call(
        &self,
        target: NameId,
        args: Vec<RefinementExpr>,
        return_type: CType,
    ) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target,
                args,
                return_type: return_type.clone(),
            },
            ty: return_type,
            span: spur_ast::span::Span::default(),
        }
    }

    fn r_not(&self, e: RefinementExpr) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::Not(Box::new(e)),
            ty: CType::Bool,
            span: spur_ast::span::Span::default(),
        }
    }

    /// Mint a fresh NameId for a refinement-handle bound binder, registering
    /// its display name in `id_to_name`. Each handle has exactly one bound.
    fn mint_bound(&mut self, hint: &str) -> NameId {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(id, hint.to_string());
        id
    }

    /// `array_len<elem>(arg)` as a refinement expr. Routes through
    /// `intern_extern` so the extern is shared with all other call sites.
    fn r_array_len(&mut self, elem: CType, arg: RefinementExpr) -> RefinementExpr {
        let raw_array = CType::Array(Box::new(elem.clone()));
        let target = self.intern_extern(
            BuiltinKind::ArrayLen,
            vec![elem],
            vec![raw_array],
            CType::Int,
        );
        self.r_extern_call(target, vec![arg], CType::Int)
    }

    /// `map_exists<k, v>(m, key)` as a refinement expr.
    fn r_map_exists(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        m: RefinementExpr,
        key: RefinementExpr,
    ) -> RefinementExpr {
        let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
        let target = self.intern_extern(
            BuiltinKind::MapExists,
            vec![k_ty.clone(), v_ty],
            vec![map_ty, k_ty],
            CType::Bool,
        );
        self.r_extern_call(target, vec![m, key], CType::Bool)
    }

    /// Wrap a freshly built body into a `CRefinementHandle` with the given
    /// bound NameId and display name.
    fn wrap_handle(
        &self,
        bound: NameId,
        hint: &str,
        body: RefinementExpr,
    ) -> CRefinementHandle {
        CRefinementHandle::new(CRefinementBody {
            bound,
            original_bound: hint.to_string(),
            body,
        })
    }

    /// Build (or fetch from cache) the baked-in `array_len(_xs) >= 0`
    /// refinement that every `list<T>` carries. The handle is created from
    /// the *raw* element type (no recursion through `lower_type`), so the
    /// `array_len` extern's parameter type stays raw `Array(elem)` and we
    /// avoid an infinite descent through the invariant.
    fn make_list_invariant_handle(&mut self, elem: CType) -> CRefinementHandle {
        if let Some(handle) = self.list_invariant_cache.get(&elem) {
            return handle.clone();
        }
        let bound = self.mint_bound("_xs");
        let xs = self.r_var(bound, "_xs", CType::Array(Box::new(elem.clone())));
        let len = self.r_array_len(elem.clone(), xs);
        let zero = self.r_int(0);
        let body = self.r_binop(CBinOp::GreaterEqual, len, zero, CType::Bool);
        let handle = self.wrap_handle(bound, "_xs", body);
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

    /// `array_empty<T>` return refinement: `{ ys | array_len(ys) == 0 }`.
    fn make_empty_return_handle(&mut self, elem: CType) -> CRefinementHandle {
        let bound = self.mint_bound("ys");
        let ys = self.r_var(bound, "ys", CType::Array(Box::new(elem.clone())));
        let len = self.r_array_len(elem, ys);
        let zero = self.r_int(0);
        let body = self.r_binop(CBinOp::IntEq, len, zero, CType::Bool);
        self.wrap_handle(bound, "ys", body)
    }

    /// `array_append<T>` / `array_prepend<T>` return refinement:
    /// `{ ys | array_len(ys) == array_len(list_param) + 1 }`. The body has
    /// a free Var referencing the enclosing extern's input-list parameter.
    fn make_append_return_handle(
        &mut self,
        elem: CType,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("ys");
        let raw = CType::Array(Box::new(elem.clone()));
        let ys = self.r_var(bound, "ys", raw.clone());
        let xs = self.r_var(list_param, &list_param_name, raw);
        let len_ys = self.r_array_len(elem.clone(), ys);
        let len_xs = self.r_array_len(elem, xs);
        let one = self.r_int(1);
        let xs_plus_one = self.r_binop(CBinOp::Add, len_xs, one, CType::Int);
        let body = self.r_binop(CBinOp::IntEq, len_ys, xs_plus_one, CType::Bool);
        self.wrap_handle(bound, "ys", body)
    }

    /// `array_index<T>` index-parameter precondition:
    /// `{ i | 0 <= i && i < array_len(list_param) }`.
    fn make_index_precondition_handle(
        &mut self,
        elem: CType,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("i");
        let i = self.r_var(bound, "i", CType::Int);
        let xs = self.r_var(
            list_param,
            &list_param_name,
            CType::Array(Box::new(elem.clone())),
        );
        let len_xs = self.r_array_len(elem, xs);
        let zero = self.r_int(0);
        let lower = self.r_binop(CBinOp::LessEqual, zero, i.clone(), CType::Bool);
        let upper = self.r_binop(CBinOp::Less, i, len_xs, CType::Bool);
        let body = self.r_binop(CBinOp::And, lower, upper, CType::Bool);
        self.wrap_handle(bound, "i", body)
    }

    /// `map_index<K, V>` key-parameter precondition:
    /// `{ k | map_exists(map_param, k) }`.
    fn make_map_index_precondition_handle(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        map_param: NameId,
        map_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("k");
        let k = self.r_var(bound, "k", k_ty.clone());
        let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
        let m = self.r_var(map_param, &map_param_name, map_ty);
        let body = self.r_map_exists(k_ty, v_ty, m, k);
        self.wrap_handle(bound, "k", body)
    }

    /// `map_store<K, V>` return refinement: `{ ys | map_exists(ys, key_param) }`.
    fn make_map_store_return_handle(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        key_param: NameId,
        key_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("ys");
        let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
        let ys = self.r_var(bound, "ys", map_ty);
        let k = self.r_var(key_param, &key_param_name, k_ty.clone());
        let body = self.r_map_exists(k_ty, v_ty, ys, k);
        self.wrap_handle(bound, "ys", body)
    }

    /// Resolve a struct field by name, returning `(field_id, field_ctype)`.
    fn resolve_struct_field(&self, struct_id: NameId, field_name: &str) -> (NameId, CType) {
        let fields = self
            .struct_defs
            .get(&struct_id)
            .expect("struct must be registered in struct_defs");
        fields
            .iter()
            .find(|(_, name, _)| name == field_name)
            .map(|(fid, _, ty)| (*fid, ty.clone()))
            .unwrap_or_else(|| {
                panic!(
                    "field {:?} not found on struct {:?}",
                    field_name, struct_id
                )
            })
    }

    /// Lower a struct-field store (`record.field := val`) in statement position.
    fn lower_struct_field_store(
        &mut self,
        struct_id: NameId,
        receiver: PAtomic,
        key: PAtomic,
        val: PAtomic,
    ) -> CExprKind {
        let field_name = match &key {
            PAtomic::StringLit(s) => s.clone(),
            _ => panic!("struct field store key must be a string literal"),
        };
        let (field_id, field_ty) = self.resolve_struct_field(struct_id, &field_name);
        let struct_ty = CType::Struct(struct_id);
        let struct_ty_for_ret = struct_ty.clone();
        let field_ty_for_ret = field_ty.clone();

        let target = self.intern_extern_with_dependent_params(
            BuiltinKind::StructFieldStore { struct_id, field_id },
            vec![],
            vec![
                ("s".to_string(), const_param(struct_ty.clone())),
                ("v".to_string(), const_param(field_ty)),
            ],
            move |this, params| {
                let post = this.make_struct_field_store_return_handle(
                    struct_ty_for_ret,
                    field_id,
                    field_ty_for_ret,
                    params[1].name,
                    params[1].original_name.clone(),
                );
                CType::Refined(Box::new(CType::Struct(struct_id)), post)
            },
        );
        let return_type = self.extern_return_type(target);
        CExprKind::FuncCall(CFuncCall {
            target,
            args: vec![lower_atomic(receiver), lower_atomic(val)],
            return_type,
        })
    }

    /// `struct_store` return refinement: `{ s | s.field == val_param }`.
    /// For field types where equality isn't expressible in the refinement
    /// language (structs, enums), we skip the refinement and return the
    /// bare struct type.
    fn make_struct_field_store_return_handle(
        &mut self,
        struct_ty: CType,
        field_id: NameId,
        field_ty: CType,
        val_param: NameId,
        val_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("s");
        let s = self.r_var(bound, "s", struct_ty);
        let lhs = RefinementExpr {
            kind: RefinementExprKind::FieldAccess(Box::new(s), field_id),
            ty: field_ty.clone(),
            span: spur_ast::span::Span::default(),
        };
        let rhs = self.r_var(val_param, &val_param_name, field_ty.clone());
        let body = match &field_ty {
            CType::Int => self.r_binop(CBinOp::IntEq, lhs, rhs, CType::Bool),
            CType::Bool => self.r_binop(CBinOp::BoolEq, lhs, rhs, CType::Bool),
            _ => {
                // For types without first-class equality in the refinement
                // language, emit a trivially-true refinement (true).
                RefinementExpr {
                    kind: RefinementExprKind::BoolLit(true),
                    ty: CType::Bool,
                    span: spur_ast::span::Span::default(),
                }
            }
        };
        self.wrap_handle(bound, "s", body)
    }

    /// `map_erase<K, V>` return refinement: `{ ys | !map_exists(ys, key_param) }`.
    fn make_map_erase_return_handle(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        key_param: NameId,
        key_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("ys");
        let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
        let ys = self.r_var(bound, "ys", map_ty);
        let k = self.r_var(key_param, &key_param_name, k_ty.clone());
        let exists = self.r_map_exists(k_ty, v_ty, ys, k);
        let body = self.r_not(exists);
        self.wrap_handle(bound, "ys", body)
    }

    /// `array_head` / `array_tail` shared non-empty precondition:
    /// `{ _xs | array_len(_xs) >= 1 }`.
    fn make_array_nonempty_precondition_handle(&mut self, elem: CType) -> CRefinementHandle {
        if let Some(handle) = self.list_nonempty_cache.get(&elem) {
            return handle.clone();
        }
        let bound = self.mint_bound("_xs");
        let xs = self.r_var(bound, "_xs", CType::Array(Box::new(elem.clone())));
        let len = self.r_array_len(elem.clone(), xs);
        let one = self.r_int(1);
        let body = self.r_binop(CBinOp::GreaterEqual, len, one, CType::Bool);
        let handle = self.wrap_handle(bound, "_xs", body);
        self.list_nonempty_cache.insert(elem, handle.clone());
        handle
    }

    /// `array_tail<T>` return refinement:
    /// `{ ys | array_len(ys) == array_len(list_param) - 1 }`.
    fn make_array_tail_return_handle(
        &mut self,
        elem: CType,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("ys");
        let raw = CType::Array(Box::new(elem.clone()));
        let ys = self.r_var(bound, "ys", raw.clone());
        let xs = self.r_var(list_param, &list_param_name, raw);
        let len_ys = self.r_array_len(elem.clone(), ys);
        let len_xs = self.r_array_len(elem, xs);
        let one = self.r_int(1);
        let xs_minus_one = self.r_binop(CBinOp::Subtract, len_xs, one, CType::Int);
        let body = self.r_binop(CBinOp::IntEq, len_ys, xs_minus_one, CType::Bool);
        self.wrap_handle(bound, "ys", body)
    }

    /// `array_slice<T>` `lo`-parameter precondition: `{ lo | 0 <= lo }`.
    fn make_array_slice_lo_precondition_handle(&mut self) -> CRefinementHandle {
        let bound = self.mint_bound("lo");
        let lo = self.r_var(bound, "lo", CType::Int);
        let zero = self.r_int(0);
        let body = self.r_binop(CBinOp::LessEqual, zero, lo, CType::Bool);
        self.wrap_handle(bound, "lo", body)
    }

    /// `array_slice<T>` `hi`-parameter precondition:
    /// `{ hi | lo_param <= hi && hi <= array_len(list_param) }`.
    fn make_array_slice_hi_precondition_handle(
        &mut self,
        elem: CType,
        lo_param: NameId,
        lo_param_name: String,
        list_param: NameId,
        list_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("hi");
        let hi = self.r_var(bound, "hi", CType::Int);
        let lo = self.r_var(lo_param, &lo_param_name, CType::Int);
        let xs = self.r_var(
            list_param,
            &list_param_name,
            CType::Array(Box::new(elem.clone())),
        );
        let len_xs = self.r_array_len(elem, xs);
        let lower = self.r_binop(CBinOp::LessEqual, lo, hi.clone(), CType::Bool);
        let upper = self.r_binop(CBinOp::LessEqual, hi, len_xs, CType::Bool);
        let body = self.r_binop(CBinOp::And, lower, upper, CType::Bool);
        self.wrap_handle(bound, "hi", body)
    }

    /// `array_slice<T>` return refinement:
    /// `{ ys | array_len(ys) == hi_param - lo_param }`.
    fn make_array_slice_return_handle(
        &mut self,
        elem: CType,
        lo_param: NameId,
        lo_param_name: String,
        hi_param: NameId,
        hi_param_name: String,
    ) -> CRefinementHandle {
        let bound = self.mint_bound("ys");
        let ys = self.r_var(bound, "ys", CType::Array(Box::new(elem.clone())));
        let lo = self.r_var(lo_param, &lo_param_name, CType::Int);
        let hi = self.r_var(hi_param, &hi_param_name, CType::Int);
        let len_ys = self.r_array_len(elem, ys);
        let hi_minus_lo = self.r_binop(CBinOp::Subtract, hi, lo, CType::Int);
        let body = self.r_binop(CBinOp::IntEq, len_ys, hi_minus_lo, CType::Bool);
        self.wrap_handle(bound, "ys", body)
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
    /// `TypedExpr`-side analogue of `array_elem_of`. Returns `Never` when the
    /// expression's type isn't a list (e.g. an upstream type error).
    fn t_array_elem(&mut self, e: &TypedExpr) -> CType {
        match &e.ty {
            Type::List(t) => self.lower_type(t),
            _ => CType::Never,
        }
    }

    fn t_map_kv(&mut self, e: &TypedExpr) -> (CType, CType) {
        match &e.ty {
            Type::Map(k, v) => (self.lower_type(k), self.lower_type(v)),
            _ => (CType::Never, CType::Never),
        }
    }

    fn t_optional_inner(&mut self, e: &TypedExpr) -> CType {
        match &e.ty {
            Type::Optional(t) => self.lower_type(t),
            _ => CType::Never,
        }
    }

    /// Common refinement-side shape: intern an extern with `(type_args,
    /// param_tys, ret)` and wrap it in `RefinementExprKind::ExternCall` over
    /// already-lowered `args`. Replaces the literal `intern_extern + match
    /// ExternCall { ... }` block at every refinement-side builtin arm.
    fn r_extern_call_kind(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        param_tys: Vec<CType>,
        args: Vec<RefinementExpr>,
        ret: CType,
    ) -> RefinementExprKind {
        let target = self.intern_extern(kind, type_args, param_tys, ret.clone());
        RefinementExprKind::ExternCall {
            target,
            args,
            return_type: ret,
        }
    }

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
                let (k_ty, v_ty) = match &expr.ty {
                    Type::Map(k, v) => (self.lower_type(k), self.lower_type(v)),
                    Type::EmptyMap => (CType::Never, CType::Never),
                    _ => match self.lower_type(&expr.ty) {
                        CType::Map(k, v) => ((*k).clone(), (*v).clone()),
                        _ => (CType::Never, CType::Never),
                    },
                };
                let mut current = self.lower_map_empty_refinement(k_ty.clone(), v_ty.clone());
                for (k_expr, v_expr) in pairs {
                    let k_l = self.lower_refinement_expr(k_expr);
                    let v_l = self.lower_refinement_expr(v_expr);
                    let map_ty = CType::Map(Box::new(k_ty.clone()), Box::new(v_ty.clone()));
                    let k_for_post = k_ty.clone();
                    let v_for_post = v_ty.clone();
                    let target = self.intern_extern_with_dependent_params(
                        BuiltinKind::MapStore,
                        vec![k_ty.clone(), v_ty.clone()],
                        vec![
                            ("m".to_string(), const_param(map_ty.clone())),
                            ("k".to_string(), const_param(k_ty.clone())),
                            ("v".to_string(), const_param(v_ty.clone())),
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
                    current = RefinementExpr {
                        kind: RefinementExprKind::ExternCall {
                            target,
                            args: vec![current, k_l, v_l],
                            return_type: return_type.clone(),
                        },
                        ty: return_type,
                        span,
                    };
                }
                return current;
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
                let elem = self.t_array_elem(list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                self.r_extern_call_kind(
                    BuiltinKind::ArrayHead,
                    vec![elem.clone()],
                    vec![list_ty],
                    vec![list_l],
                    elem,
                )
            }
            TypedExprKind::Tail(list) => {
                let elem = self.t_array_elem(list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                self.r_extern_call_kind(
                    BuiltinKind::ArrayTail,
                    vec![elem],
                    vec![list_ty.clone()],
                    vec![list_l],
                    list_ty,
                )
            }
            TypedExprKind::Len(list) => {
                let elem = self.t_array_elem(list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                self.r_extern_call_kind(
                    BuiltinKind::ArrayLen,
                    vec![elem],
                    vec![list_ty],
                    vec![list_l],
                    CType::Int,
                )
            }
            TypedExprKind::Slice(list, lo, hi) => {
                let elem = self.t_array_elem(list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                let list_l = self.lower_refinement_expr(list);
                let lo_l = self.lower_refinement_expr(lo);
                let hi_l = self.lower_refinement_expr(hi);
                self.r_extern_call_kind(
                    BuiltinKind::ArraySlice,
                    vec![elem],
                    vec![list_ty.clone(), CType::Int, CType::Int],
                    vec![list_l, lo_l, hi_l],
                    list_ty,
                )
            }
            TypedExprKind::Min(a, b) => {
                let ty = self.lower_type(&a.ty);
                let a_l = self.lower_refinement_expr(a);
                let b_l = self.lower_refinement_expr(b);
                self.r_extern_call_kind(
                    BuiltinKind::Min,
                    vec![ty.clone()],
                    vec![ty.clone(), ty.clone()],
                    vec![a_l, b_l],
                    ty,
                )
            }
            TypedExprKind::Exists(map, key) => {
                let (k, v) = self.t_map_kv(map);
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                let map_l = self.lower_refinement_expr(map);
                let key_l = self.lower_refinement_expr(key);
                self.r_extern_call_kind(
                    BuiltinKind::MapExists,
                    vec![k.clone(), v],
                    vec![map_ty, k],
                    vec![map_l, key_l],
                    CType::Bool,
                )
            }
            TypedExprKind::Erase(map, key) => {
                let (k, v) = self.t_map_kv(map);
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                let map_l = self.lower_refinement_expr(map);
                let key_l = self.lower_refinement_expr(key);
                self.r_extern_call_kind(
                    BuiltinKind::MapErase,
                    vec![k.clone(), v],
                    vec![map_ty.clone(), k],
                    vec![map_l, key_l],
                    map_ty,
                )
            }
            TypedExprKind::Store(receiver, key, val) => {
                match &receiver.ty {
                    Type::Struct(struct_id, _) => {
                        let struct_id = *struct_id;
                        let field_name = match &key.kind {
                            TypedExprKind::StringLit(s) => s.clone(),
                            _ => panic!("struct field store key must be a string literal"),
                        };
                        let (field_id, field_ty) = self.resolve_struct_field(struct_id, &field_name);
                        let struct_ty = CType::Struct(struct_id);
                        let receiver_l = self.lower_refinement_expr(receiver);
                        let val_l = self.lower_refinement_expr(val);
                        self.r_extern_call_kind(
                            BuiltinKind::StructFieldStore { struct_id, field_id },
                            vec![],
                            vec![struct_ty.clone(), field_ty],
                            vec![receiver_l, val_l],
                            struct_ty,
                        )
                    }
                    _ => {
                        let (k, v) = self.t_map_kv(receiver);
                        let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                        let map_l = self.lower_refinement_expr(receiver);
                        let key_l = self.lower_refinement_expr(key);
                        let val_l = self.lower_refinement_expr(val);
                        self.r_extern_call_kind(
                            BuiltinKind::MapStore,
                            vec![k.clone(), v.clone()],
                            vec![map_ty.clone(), k, v],
                            vec![map_l, key_l, val_l],
                            map_ty,
                        )
                    }
                }
            }

            TypedExprKind::UnwrapOptional(e) => {
                let inner = self.t_optional_inner(e);
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                let e_l = self.lower_refinement_expr(e);
                self.r_extern_call_kind(
                    BuiltinKind::OptionalUnwrap,
                    vec![inner.clone()],
                    vec![opt_ty],
                    vec![e_l],
                    inner,
                )
            }
            TypedExprKind::WrapInOptional(e) => {
                let inner = self.lower_type(&e.ty);
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                let e_l = self.lower_refinement_expr(e);
                self.r_extern_call_kind(
                    BuiltinKind::OptionalWrap,
                    vec![inner.clone()],
                    vec![inner],
                    vec![e_l],
                    opt_ty,
                )
            }

            TypedExprKind::Index(coll, idx) => {
                let coll_l = self.lower_refinement_expr(coll);
                let idx_l = self.lower_refinement_expr(idx);
                match &coll.ty {
                    Type::List(t) => {
                        let elem = self.lower_type(t);
                        let raw_array = CType::Array(Box::new(elem.clone()));
                        self.r_extern_call_kind(
                            BuiltinKind::ArrayIndex,
                            vec![elem.clone()],
                            vec![raw_array, CType::Int],
                            vec![coll_l, idx_l],
                            elem,
                        )
                    }
                    Type::Map(k, v) => {
                        let k_c = self.lower_type(k);
                        let v_c = self.lower_type(v);
                        let map_ty =
                            CType::Map(Box::new(k_c.clone()), Box::new(v_c.clone()));
                        self.r_extern_call_kind(
                            BuiltinKind::MapIndex,
                            vec![k_c.clone(), v_c.clone()],
                            vec![map_ty, k_c],
                            vec![coll_l, idx_l],
                            v_c,
                        )
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
            TypedExprKind::FieldAccess(s, field_id, _name) => {
                let s_l = Box::new(self.lower_refinement_expr(s));
                RefinementExprKind::FieldAccess(s_l, *field_id)
            }
            TypedExprKind::SafeFieldAccess(s, field_id, _name) => {
                let struct_ty = self.lower_type(&s.ty);
                let ret_c = self.lower_type(&expr.ty);
                let s_l = self.lower_refinement_expr(s);
                let target = self.intern_extern(
                    BuiltinKind::SafeField(*field_id),
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
                    .map(|(field_id, _name, e)| (*field_id, self.lower_refinement_expr(e)))
                    .collect(),
            ),
            TypedExprKind::VariantLit(enum_id, variant_id, _name, payload) => {
                RefinementExprKind::VariantLit(
                    *enum_id,
                    *variant_id,
                    payload
                        .as_ref()
                        .map(|p| Box::new(self.lower_refinement_expr(p))),
                )
            }

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

    fn disallowed(&mut self, what: &'static str, span: spur_ast::span::Span) -> RefinementExprKind {
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
            span: spur_ast::span::Span::default(),
        }
    }

    /// Build a refinement-side `map_empty<K, V>()` ExternCall expression.
    /// The extern's return type is plain `Map(K, V)` — successive `map_store`
    /// calls layer the per-key `map_exists` invariants on top.
    fn lower_map_empty_refinement(&mut self, k_ty: CType, v_ty: CType) -> RefinementExpr {
        let k_for_builder = k_ty.clone();
        let v_for_builder = v_ty.clone();
        let target = self.intern_extern_with_params(
            BuiltinKind::MapEmpty,
            vec![k_ty.clone(), v_ty.clone()],
            vec![],
            move |_this, _params| CType::Map(Box::new(k_for_builder), Box::new(v_for_builder)),
        );
        let return_type = self.extern_return_type(target);
        RefinementExpr {
            kind: RefinementExprKind::ExternCall {
                target,
                args: vec![],
                return_type: return_type.clone(),
            },
            ty: return_type,
            span: spur_ast::span::Span::default(),
        }
    }

    /// Build a statement-side `map_empty<K, V>()` FuncCall. Returns a `CExpr`
    /// suitable for use as a let-atom value.
    fn lower_map_empty_stmt(
        &mut self,
        k_ty: CType,
        v_ty: CType,
        span: spur_ast::span::Span,
    ) -> CExpr {
        let k_for_builder = k_ty.clone();
        let v_for_builder = v_ty.clone();
        let kind = self.emit_extern_call_with_params(
            BuiltinKind::MapEmpty,
            vec![k_ty.clone(), v_ty.clone()],
            vec![],
            move |_this, _params| CType::Map(Box::new(k_for_builder), Box::new(v_for_builder)),
            vec![],
        );
        let ty = match &kind {
            CExprKind::FuncCall(call) => call.return_type.clone(),
            _ => unreachable!("emit_extern_call_with_params returns FuncCall"),
        };
        CExpr { kind, ty, span }
    }

    /// Build a statement-side `array_empty<T>()` FuncCall. Returns a `CExpr`
    /// suitable for use as a let-atom value. The extern's return type matches
    /// the refinement-side construction (layered refined list).
    fn lower_array_empty_stmt(&mut self, elem: CType, span: spur_ast::span::Span) -> CExpr {
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
        span: spur_ast::span::Span,
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
        span: spur_ast::span::Span,
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
    pub span: spur_ast::span::Span,
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
mod test;
