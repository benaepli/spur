use std::collections::HashMap;

use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::{
    RefinementBody, Type, TypedBlock, TypedCondExpr, TypedExpr, TypedExprKind, TypedFuncCall,
};
use crate::liquid::pure::ast::*;

use super::ast::*;
use super::builtins::BuiltinKind;
use super::refinement::{
    RefinementCond, RefinementExpr, RefinementExprKind, RefinementIfBranch,
};

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

    LowerOutput {
        program: CProgram {
            funcs: lowerer.funcs,
            extern_funcs: lowerer.extern_funcs,
            struct_defs: lowerer.struct_defs,
            enum_defs: lowerer.enum_defs,
            next_name_id: lowerer.next_name_id,
            id_to_name: lowerer.id_to_name,
        },
        refinement_errors: lowerer.refinement_errors,
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
            statements.push(self.lower_statement(stmt));
        }
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
            PExprKind::BinOp(op, a, b) => CExprKind::BinOp(op, lower_atomic(a), lower_atomic(b)),
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

            PExprKind::ListLit(items) => {
                CExprKind::ListLit(items.into_iter().map(lower_atomic).collect())
            }
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
                let list_ty = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::ArrayAppend,
                    vec![elem.clone()],
                    vec![list_ty.clone(), elem],
                    list_ty,
                    vec![list, item],
                )
            }
            PExprKind::Prepend(list, item) => {
                let elem = self.array_elem_of(&list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::ArrayPrepend,
                    vec![elem.clone()],
                    vec![list_ty.clone(), elem],
                    list_ty,
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
                self.emit_extern_call(
                    BuiltinKind::MapErase,
                    vec![k.clone(), v],
                    vec![map_ty.clone(), k],
                    map_ty,
                    vec![map, key],
                )
            }
            PExprKind::Store(map, key, val) => {
                let (k, v) = self.map_kv_of(&map);
                let map_ty = CType::Map(Box::new(k.clone()), Box::new(v.clone()));
                self.emit_extern_call(
                    BuiltinKind::MapStore,
                    vec![k.clone(), v.clone()],
                    vec![map_ty.clone(), k, v],
                    map_ty,
                    vec![map, key, val],
                )
            }
            PExprKind::Head(list) => {
                let elem = self.array_elem_of(&list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::ArrayHead,
                    vec![elem.clone()],
                    vec![list_ty],
                    elem,
                    vec![list],
                )
            }
            PExprKind::Tail(list) => {
                let elem = self.array_elem_of(&list);
                let list_ty = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::ArrayTail,
                    vec![elem.clone()],
                    vec![list_ty.clone()],
                    list_ty,
                    vec![list],
                )
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
                let list_ty = CType::Array(Box::new(elem.clone()));
                self.emit_extern_call(
                    BuiltinKind::ArraySlice,
                    vec![elem.clone()],
                    vec![list_ty.clone(), CType::Int, CType::Int],
                    list_ty,
                    vec![list, lo, hi],
                )
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

            PExprKind::Index(coll, idx) => CExprKind::Index(lower_atomic(coll), lower_atomic(idx)),
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

    /// Resolve-or-allocate an extern entry, then build the FuncCall expression.
    fn emit_extern_call(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        params: Vec<CType>,
        return_type: CType,
        args: Vec<PAtomic>,
    ) -> CExprKind {
        let target = self.intern_extern(kind, type_args, params, return_type.clone());
        CExprKind::FuncCall(CFuncCall {
            target,
            args: args.into_iter().map(lower_atomic).collect(),
            return_type,
        })
    }

    fn intern_extern(
        &mut self,
        kind: BuiltinKind,
        type_args: Vec<CType>,
        params: Vec<CType>,
        return_type: CType,
    ) -> NameId {
        let key = (kind.clone(), type_args.clone());
        if let Some(id) = self.extern_cache.get(&key) {
            return *id;
        }
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
            Type::List(t) => CType::Array(Box::new(self.lower_type(t))),
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
            Type::EmptyList => CType::Array(Box::new(CType::Never)),
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

            TypedExprKind::BinOp(op, l, r) => {
                let l = Box::new(self.lower_refinement_expr(l));
                let r = Box::new(self.lower_refinement_expr(r));
                RefinementExprKind::BinOp(op.clone(), l, r)
            }
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
            TypedExprKind::ListLit(es) => RefinementExprKind::ListLit(
                es.iter().map(|e| self.lower_refinement_expr(e)).collect(),
            ),
            TypedExprKind::TupleLit(es) => RefinementExprKind::TupleLit(
                es.iter().map(|e| self.lower_refinement_expr(e)).collect(),
            ),

            TypedExprKind::Append(list, item) => self.lower_array_extern(
                BuiltinKind::ArrayAppend,
                list,
                Some(item),
                /* return_list */ true,
                span,
            ),
            TypedExprKind::Prepend(list, item) => self.lower_array_extern(
                BuiltinKind::ArrayPrepend,
                list,
                Some(item),
                true,
                span,
            ),
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
                let coll_l = Box::new(self.lower_refinement_expr(coll));
                let idx_l = Box::new(self.lower_refinement_expr(idx));
                RefinementExprKind::Index(coll_l, idx_l)
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

    /// Helper for two-arg array operations that take `(list, item)` and
    /// return either the same list type or the element type.
    fn lower_array_extern(
        &mut self,
        kind: BuiltinKind,
        list: &TypedExpr,
        item: Option<&TypedExpr>,
        return_list: bool,
        _span: crate::parser::Span,
    ) -> RefinementExprKind {
        let elem = match &list.ty {
            Type::List(t) => self.lower_type(t),
            _ => CType::Never,
        };
        let list_ty = CType::Array(Box::new(elem.clone()));
        let return_type = if return_list { list_ty.clone() } else { elem.clone() };

        let mut params = vec![list_ty.clone()];
        let mut args_lowered = vec![self.lower_refinement_expr(list)];
        if let Some(item) = item {
            params.push(elem.clone());
            args_lowered.push(self.lower_refinement_expr(item));
        }

        let target = self.intern_extern(kind, vec![elem], params, return_type.clone());
        RefinementExprKind::ExternCall {
            target,
            args: args_lowered,
            return_type,
        }
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

        assert_eq!(c.extern_funcs.len(), 2);
        assert_ne!(c.extern_funcs[0].name, c.extern_funcs[1].name);
        assert_eq!(c.extern_funcs[0].original_name, "array_append<int>");
        assert_eq!(c.extern_funcs[1].original_name, "array_append<string>");
        assert_eq!(
            c.extern_funcs[0].params,
            vec![CType::Array(Box::new(CType::Int)), CType::Int]
        );
        assert_eq!(
            c.extern_funcs[1].params,
            vec![CType::Array(Box::new(CType::String)), CType::String]
        );
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
        assert_eq!(c.extern_funcs.len(), 1);
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
        assert_eq!(c.extern_funcs.len(), 1);
        let ext = &c.extern_funcs[0];
        assert_eq!(ext.original_name, "iter_make<int>");
        assert_eq!(ext.params, vec![CType::Array(Box::new(CType::Int))]);
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
            RefinementExprKind::BinOp(BinOp::Greater, l, r) => {
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
}
