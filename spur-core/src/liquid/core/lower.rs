use std::collections::HashMap;

use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::Type;
use crate::liquid::pure::ast::*;

use super::ast::*;
use super::builtins::BuiltinKind;

pub fn lower_program(program: PProgram) -> CProgram {
    let mut lowerer = CoreLowerer {
        next_name_id: program.next_name_id,
        id_to_name: program.id_to_name,
        var_types: HashMap::new(),
        extern_cache: HashMap::new(),
        extern_funcs: Vec::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
        funcs: Vec::new(),
    };

    for (id, fields) in &program.struct_defs {
        let lowered = fields
            .iter()
            .map(|(name, ty)| (name.clone(), lower_type(ty)))
            .collect();
        lowerer.struct_defs.insert(*id, lowered);
    }
    for (id, variants) in &program.enum_defs {
        let lowered = variants
            .iter()
            .map(|(name, payload)| (name.clone(), payload.as_ref().map(lower_type)))
            .collect();
        lowerer.enum_defs.insert(*id, lowered);
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

    CProgram {
        funcs: lowerer.funcs,
        extern_funcs: lowerer.extern_funcs,
        struct_defs: lowerer.struct_defs,
        enum_defs: lowerer.enum_defs,
        next_name_id: lowerer.next_name_id,
        id_to_name: lowerer.id_to_name,
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
}

impl CoreLowerer {
    fn lower_func(&mut self, func: PFuncDef, role: Option<NameId>) -> CFuncDef {
        for param in &func.params {
            self.var_types.insert(param.name, param.ty.clone());
        }
        let params = func
            .params
            .iter()
            .map(|p| CFuncParam {
                name: p.name,
                original_name: p.original_name.clone(),
                ty: lower_type(&p.ty),
                span: p.span,
            })
            .collect();

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
            return_type: lower_type(&func.return_type),
            body,
            span: func.span,
        }
    }

    fn lower_block(&mut self, block: PBlock) -> CBlock {
        let statements = block
            .statements
            .into_iter()
            .map(|stmt| self.lower_statement(stmt))
            .collect();
        CBlock {
            statements,
            tail_expr: block.tail_expr.map(lower_atomic),
            ty: lower_type(&block.ty),
            span: block.span,
        }
    }

    fn lower_statement(&mut self, stmt: PStatement) -> CStatement {
        let kind = match stmt.kind {
            PStatementKind::LetAtom(let_atom) => {
                let value = self.lower_expr(let_atom.value);
                self.var_types.insert(let_atom.name, let_atom.ty.clone());
                CStatementKind::LetAtom(CLetAtom {
                    name: let_atom.name,
                    original_name: let_atom.original_name,
                    ty: lower_type(&let_atom.ty),
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
        let result_ty = lower_type(&expr.ty);
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
                    return_type: lower_type(&user.return_type),
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
                        args.iter().map(|a| lower_type(&self.atomic_p_type(a))).collect();
                    let ret_c = lower_type(&ret);
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
                let ty = lower_type(&self.atomic_p_type(&a));
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
                let dest_ty = lower_type(&self.atomic_p_type(&dest));
                let call_arg_types: Vec<CType> = call
                    .args
                    .iter()
                    .map(|a| lower_type(&self.atomic_p_type(a)))
                    .collect();
                let mut params = vec![dest_ty];
                params.extend(call_arg_types);
                let mut args = vec![dest];
                args.extend(call.args);
                self.emit_extern_call(
                    BuiltinKind::Rpc(call.name),
                    vec![],
                    params,
                    lower_type(&call.return_type),
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
                let inner = lower_type(&self.atomic_p_type(&a));
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
                self.emit_extern_call(
                    BuiltinKind::IterNext,
                    vec![elem],
                    vec![iter_ty],
                    // The result we declare matches the spec's PExpr::ty if it
                    // exists; otherwise fall back to (iter, elem).
                    if matches!(result_ty, Type::Error) {
                        result
                    } else {
                        lower_type(result_ty)
                    },
                    vec![it],
                )
            }

            PExprKind::MakeChannel => {
                let elem = match result_ty {
                    Type::Chan(t) => lower_type(t),
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
                let elem = lower_type(&self.atomic_p_type(&value));
                let chan_ty = CType::Chan(Box::new(elem.clone()));
                let state_ty = lower_type(&self.atomic_p_type(&state));
                self.emit_extern_call(
                    BuiltinKind::ChanSend,
                    vec![elem.clone()],
                    vec![state_ty, chan_ty, elem],
                    lower_type(result_ty),
                    vec![state, chan, value],
                )
            }
            PExprKind::Recv(state, chan) => {
                let elem = self.chan_inner_of(&chan);
                let chan_ty = CType::Chan(Box::new(elem.clone()));
                let state_ty = lower_type(&self.atomic_p_type(&state));
                self.emit_extern_call(
                    BuiltinKind::ChanRecv,
                    vec![elem],
                    vec![state_ty, chan_ty],
                    lower_type(result_ty),
                    vec![state, chan],
                )
            }

            PExprKind::SetTimer(_label) => {
                let elem = match result_ty {
                    Type::Chan(t) => lower_type(t),
                    _ => CType::Tuple(vec![]),
                };
                let chan_ty = CType::Chan(Box::new(elem));
                self.emit_extern_call(BuiltinKind::TimerSet, vec![], vec![], chan_ty, vec![])
            }

            PExprKind::Index(coll, idx) => CExprKind::Index(lower_atomic(coll), lower_atomic(idx)),
            PExprKind::TupleAccess(t, i) => CExprKind::TupleAccess(lower_atomic(t), i),
            PExprKind::FieldAccess(s, f) => CExprKind::FieldAccess(lower_atomic(s), f),

            PExprKind::SafeFieldAccess(s, field) => {
                let struct_ty = lower_type(&self.atomic_p_type(&s));
                let ret_c = lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::SafeField(field.clone()),
                    vec![struct_ty.clone(), ret_c.clone()],
                    vec![struct_ty],
                    ret_c,
                    vec![s],
                )
            }
            PExprKind::SafeIndex(coll, idx) => {
                let coll_ty = lower_type(&self.atomic_p_type(&coll));
                let idx_ty = lower_type(&self.atomic_p_type(&idx));
                let ret_c = lower_type(result_ty);
                self.emit_extern_call(
                    BuiltinKind::SafeIndex,
                    vec![coll_ty.clone(), idx_ty.clone(), ret_c.clone()],
                    vec![coll_ty, idx_ty],
                    ret_c,
                    vec![coll, idx],
                )
            }
            PExprKind::SafeTupleAccess(t, i) => {
                let tuple_ty = lower_type(&self.atomic_p_type(&t));
                let ret_c = lower_type(result_ty);
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
                let inner = lower_type(&self.atomic_p_type(&value));
                self.emit_extern_call(
                    BuiltinKind::Persist,
                    vec![inner.clone()],
                    vec![inner],
                    lower_type(result_ty),
                    vec![value],
                )
            }
            PExprKind::RetrieveData(ty) => {
                let inner = lower_type(&ty);
                let opt_ty = CType::Optional(Box::new(inner.clone()));
                self.emit_extern_call(
                    BuiltinKind::Retrieve,
                    vec![inner],
                    vec![],
                    opt_ty,
                    vec![],
                )
            }
            PExprKind::DiscardData => self.emit_extern_call(
                BuiltinKind::Discard,
                vec![],
                vec![],
                lower_type(result_ty),
                vec![],
            ),
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
            Type::List(t) => lower_type(&t),
            _ => CType::Never,
        }
    }

    fn map_kv_of(&self, atom: &PAtomic) -> (CType, CType) {
        match self.atomic_p_type(atom) {
            Type::Map(k, v) => (lower_type(&k), lower_type(&v)),
            _ => (CType::Never, CType::Never),
        }
    }

    fn optional_inner_of(&self, atom: &PAtomic) -> CType {
        match self.atomic_p_type(atom) {
            Type::Optional(t) => lower_type(&t),
            _ => CType::Never,
        }
    }

    fn chan_inner_of(&self, atom: &PAtomic) -> CType {
        match self.atomic_p_type(atom) {
            Type::Chan(t) => lower_type(&t),
            _ => CType::Never,
        }
    }

    fn iter_inner_of(&self, atom: &PAtomic) -> CType {
        match self.atomic_p_type(atom) {
            Type::Iter(t) => lower_type(&t),
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

pub fn lower_type(ty: &Type) -> CType {
    match ty {
        Type::Int => CType::Int,
        Type::String => CType::String,
        Type::Bool => CType::Bool,
        Type::List(t) => CType::Array(Box::new(lower_type(t))),
        Type::Map(k, v) => CType::Map(Box::new(lower_type(k)), Box::new(lower_type(v))),
        Type::Tuple(ts) => CType::Tuple(ts.iter().map(lower_type).collect()),
        Type::Struct(id, _) => CType::Struct(*id),
        Type::Enum(id, _) => CType::Variant(*id),
        Type::Role(id, _) => CType::Role(*id),
        Type::Optional(t) => CType::Optional(Box::new(lower_type(t))),
        Type::Chan(t) => CType::Chan(Box::new(lower_type(t))),
        Type::Iter(t) => CType::Iter(Box::new(lower_type(t))),
        Type::EmptyList => CType::Array(Box::new(CType::Never)),
        Type::EmptyMap => CType::Map(Box::new(CType::Never), Box::new(CType::Never)),
        Type::UnknownChannel => CType::Chan(Box::new(CType::Never)),
        Type::Nil => CType::Nil,
        Type::Never => CType::Never,
        Type::Error => CType::Never,
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

        let c = lower_program(prog);

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

        let c = lower_program(prog);
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

        let c = lower_program(prog);
        assert_eq!(c.extern_funcs.len(), 1);
        let ext = &c.extern_funcs[0];
        assert_eq!(ext.original_name, "iter_make<int>");
        assert_eq!(ext.params, vec![CType::Array(Box::new(CType::Int))]);
        assert_eq!(ext.return_type, CType::Iter(Box::new(CType::Int)));
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

        let c = lower_program(prog);
        assert_eq!(c.funcs.len(), 3);
        assert_eq!(c.funcs[0].role, Some(nid(99)));
        assert_eq!(c.funcs[1].role, Some(nid(99)));
        assert_eq!(c.funcs[2].role, None);
    }
}
