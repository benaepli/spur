use std::collections::HashSet;

use crate::analysis::resolver::NameId;
use crate::analysis::types::Type;
use crate::compiler::anf::*;
use crate::liquid::threaded::ast::*;
use crate::parser::Span;

/// Context struct for the ANF → Threaded AST transformation.
/// Threads an explicit state variable `s` through role methods.
struct ThreadLowerer {
    next_name_id: usize,
    /// NameIds of role-level variables for the role currently being lowered.
    role_var_ids: HashSet<NameId>,
    /// NameId allocated for the `s` state parameter.
    state_name_id: NameId,
    /// The Type::Struct for the current role's state.
    state_type: Option<Type>,
    /// Whether we are currently inside a role method (vs free function).
    in_role_method: bool,
}

impl ThreadLowerer {
    fn fresh_name(&mut self, prefix: &str) -> (NameId, String) {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        let name = format!("__{}", prefix);
        (id, name)
    }

    fn s_atomic(&self) -> TAtomic {
        TAtomic::Var(self.state_name_id, "s".to_string())
    }

    fn s_lhs(&self) -> TLhsExpr {
        TLhsExpr::Var(self.state_name_id, "s".to_string())
    }

    fn lower_atomic(&self, a: AAtomic) -> TAtomic {
        match a {
            AAtomic::Var(id, name) => TAtomic::Var(id, name),
            AAtomic::IntLit(v) => TAtomic::IntLit(v),
            AAtomic::StringLit(v) => TAtomic::StringLit(v),
            AAtomic::BoolLit(v) => TAtomic::BoolLit(v),
            AAtomic::NilLit => TAtomic::NilLit,
            AAtomic::Never => TAtomic::Never,
        }
    }

    fn lower_lhs(&self, lhs: ALhsExpr) -> TLhsExpr {
        match lhs {
            ALhsExpr::Var(id, name) => {
                if self.in_role_method && self.role_var_ids.contains(&id) {
                    TLhsExpr::FieldAccess(Box::new(self.s_lhs()), name)
                } else {
                    TLhsExpr::Var(id, name)
                }
            }
            ALhsExpr::Index(base, idx) => {
                TLhsExpr::Index(Box::new(self.lower_lhs(*base)), self.lower_atomic(idx))
            }
            ALhsExpr::FieldAccess(base, name) => {
                TLhsExpr::FieldAccess(Box::new(self.lower_lhs(*base)), name)
            }
            ALhsExpr::TupleAccess(base, idx) => {
                TLhsExpr::TupleAccess(Box::new(self.lower_lhs(*base)), idx)
            }
            ALhsExpr::SafeIndex(base, idx) => {
                TLhsExpr::SafeIndex(Box::new(self.lower_lhs(*base)), self.lower_atomic(idx))
            }
            ALhsExpr::SafeFieldAccess(base, name) => {
                TLhsExpr::SafeFieldAccess(Box::new(self.lower_lhs(*base)), name)
            }
            ALhsExpr::SafeTupleAccess(base, idx) => {
                TLhsExpr::SafeTupleAccess(Box::new(self.lower_lhs(*base)), idx)
            }
        }
    }

    fn lower_expr(&mut self, expr: AExpr) -> TExpr {
        let span = expr.span;
        let ty = expr.ty;
        let kind = match expr.kind {
            AExprKind::Atomic(a) => TExprKind::Atomic(self.lower_atomic(a)),
            AExprKind::BinOp(op, a, b) => {
                TExprKind::BinOp(op, self.lower_atomic(a), self.lower_atomic(b))
            }
            AExprKind::Not(a) => TExprKind::Not(self.lower_atomic(a)),
            AExprKind::Negate(a) => TExprKind::Negate(self.lower_atomic(a)),
            AExprKind::FuncCall(call) => TExprKind::FuncCall(self.lower_func_call(call)),
            AExprKind::MapLit(pairs) => TExprKind::MapLit(
                pairs
                    .into_iter()
                    .map(|(k, v)| (self.lower_atomic(k), self.lower_atomic(v)))
                    .collect(),
            ),
            AExprKind::ListLit(items) => {
                TExprKind::ListLit(items.into_iter().map(|a| self.lower_atomic(a)).collect())
            }
            AExprKind::TupleLit(items) => {
                TExprKind::TupleLit(items.into_iter().map(|a| self.lower_atomic(a)).collect())
            }
            AExprKind::Append(a, b) => {
                TExprKind::Append(self.lower_atomic(a), self.lower_atomic(b))
            }
            AExprKind::Prepend(a, b) => {
                TExprKind::Prepend(self.lower_atomic(a), self.lower_atomic(b))
            }
            AExprKind::Min(a, b) => TExprKind::Min(self.lower_atomic(a), self.lower_atomic(b)),
            AExprKind::Exists(a, b) => {
                TExprKind::Exists(self.lower_atomic(a), self.lower_atomic(b))
            }
            AExprKind::Erase(a, b) => TExprKind::Erase(self.lower_atomic(a), self.lower_atomic(b)),
            AExprKind::Store(a, b, c) => TExprKind::Store(
                self.lower_atomic(a),
                self.lower_atomic(b),
                self.lower_atomic(c),
            ),
            AExprKind::Head(a) => TExprKind::Head(self.lower_atomic(a)),
            AExprKind::Tail(a) => TExprKind::Tail(self.lower_atomic(a)),
            AExprKind::Len(a) => TExprKind::Len(self.lower_atomic(a)),
            AExprKind::RpcCall(target, call) => {
                TExprKind::RpcCall(self.lower_atomic(target), self.lower_user_func_call(call))
            }
            AExprKind::Conditional(cond) => {
                TExprKind::Conditional(Box::new(self.lower_cond_expr(*cond)))
            }
            AExprKind::Block(block) => TExprKind::Block(Box::new(self.lower_block(*block))),
            AExprKind::VariantLit(id, name, payload) => {
                TExprKind::VariantLit(id, name, payload.map(|a| self.lower_atomic(a)))
            }
            AExprKind::IsVariant(a, name) => TExprKind::IsVariant(self.lower_atomic(a), name),
            AExprKind::VariantPayload(a) => TExprKind::VariantPayload(self.lower_atomic(a)),
            AExprKind::UnwrapOptional(a) => TExprKind::UnwrapOptional(self.lower_atomic(a)),
            AExprKind::MakeIter(a) => TExprKind::MakeIter(self.lower_atomic(a)),
            AExprKind::IterIsDone(a) => TExprKind::IterIsDone(self.lower_atomic(a)),
            AExprKind::IterNext(a) => TExprKind::IterNext(self.lower_atomic(a)),
            AExprKind::MakeChannel => TExprKind::MakeChannel,
            AExprKind::Send(chan, val) => {
                if self.in_role_method {
                    TExprKind::Send(
                        self.s_atomic(),
                        self.lower_atomic(chan),
                        self.lower_atomic(val),
                    )
                } else {
                    TExprKind::Send(
                        TAtomic::NilLit,
                        self.lower_atomic(chan),
                        self.lower_atomic(val),
                    )
                }
            }
            AExprKind::Recv(chan) => {
                if self.in_role_method {
                    TExprKind::Recv(self.s_atomic(), self.lower_atomic(chan))
                } else {
                    TExprKind::Recv(TAtomic::NilLit, self.lower_atomic(chan))
                }
            }
            AExprKind::SetTimer(label) => TExprKind::SetTimer(label),
            AExprKind::Index(a, b) => TExprKind::Index(self.lower_atomic(a), self.lower_atomic(b)),
            AExprKind::Slice(a, b, c) => TExprKind::Slice(
                self.lower_atomic(a),
                self.lower_atomic(b),
                self.lower_atomic(c),
            ),
            AExprKind::TupleAccess(a, idx) => TExprKind::TupleAccess(self.lower_atomic(a), idx),
            AExprKind::FieldAccess(a, name) => TExprKind::FieldAccess(self.lower_atomic(a), name),
            AExprKind::SafeFieldAccess(a, name) => {
                TExprKind::SafeFieldAccess(self.lower_atomic(a), name)
            }
            AExprKind::SafeIndex(a, b) => {
                TExprKind::SafeIndex(self.lower_atomic(a), self.lower_atomic(b))
            }
            AExprKind::SafeTupleAccess(a, idx) => {
                TExprKind::SafeTupleAccess(self.lower_atomic(a), idx)
            }
            AExprKind::StructLit(id, fields) => TExprKind::StructLit(
                id,
                fields
                    .into_iter()
                    .map(|(name, a)| (name, self.lower_atomic(a)))
                    .collect(),
            ),
            AExprKind::WrapInOptional(a) => TExprKind::WrapInOptional(self.lower_atomic(a)),
            AExprKind::PersistData(a) => TExprKind::PersistData(self.lower_atomic(a)),
            AExprKind::RetrieveData(t) => TExprKind::RetrieveData(t),
            AExprKind::DiscardData => TExprKind::DiscardData,
        };
        TExpr { kind, ty, span }
    }

    fn lower_func_call(&self, call: AFuncCall) -> TFuncCall {
        match call {
            AFuncCall::User(u) => TFuncCall::User(self.lower_user_func_call(u)),
            AFuncCall::Builtin(builtin, args, ret_ty) => {
                let targs: Vec<_> = args.into_iter().map(|a| self.lower_atomic(a)).collect();
                TFuncCall::Builtin(builtin, targs, ret_ty)
            }
        }
    }

    fn lower_user_func_call(&self, call: AUserFuncCall) -> TUserFuncCall {
        TUserFuncCall {
            name: call.name,
            original_name: call.original_name,
            args: call
                .args
                .into_iter()
                .map(|a| self.lower_atomic(a))
                .collect(),
            return_type: call.return_type,
            is_free: call.is_free,
            span: call.span,
        }
    }

    /// Lower a user func call for a role method: prepend `s` to args,
    /// update return type to (State, T).
    fn lower_user_func_call_threaded(&mut self, call: AUserFuncCall) -> (TUserFuncCall, Type) {
        let original_return = call.return_type.clone();
        let state_type = self.state_type.clone().unwrap();
        let threaded_return = Type::Tuple(vec![state_type, original_return.clone()]);
        let mut args: Vec<TAtomic> = vec![self.s_atomic()];
        args.extend(call.args.into_iter().map(|a| self.lower_atomic(a)));
        let tcall = TUserFuncCall {
            name: call.name,
            original_name: call.original_name,
            args,
            return_type: threaded_return.clone(),
            is_free: call.is_free,
            span: call.span,
        };
        (tcall, original_return)
    }

    fn lower_cond_expr(&mut self, cond: ACondExpr) -> TCondExpr {
        let if_branch = {
            let condition = self.lower_atomic(cond.if_branch.condition.clone());
            let body = self.lower_block(cond.if_branch.body.clone());
            TIfBranch {
                condition,
                body,
                span: cond.if_branch.span,
            }
        };
        let elseif_branches: Vec<_> = cond
            .elseif_branches
            .into_iter()
            .map(|b| {
                let condition = self.lower_atomic(b.condition.clone());
                let body = self.lower_block(b.body);
                TIfBranch {
                    condition,
                    body,
                    span: b.span,
                }
            })
            .collect();
        let else_branch = cond.else_branch.map(|b| self.lower_block(b));
        TCondExpr {
            if_branch,
            elseif_branches,
            else_branch,
            span: cond.span,
        }
    }

    fn lower_block(&mut self, block: ABlock) -> TBlock {
        let mut tstmts = Vec::new();
        for stmt in block.statements {
            self.lower_statement(stmt, &mut tstmts);
        }
        let tail = block.tail_expr.map(|a| self.lower_atomic(a));
        TBlock {
            statements: tstmts,
            tail_expr: tail,
            ty: block.ty,
            span: block.span,
        }
    }

    fn lower_statement(&mut self, stmt: AStatement, out: &mut Vec<TStatement>) {
        let span = stmt.span;
        match stmt.kind {
            AStatementKind::LetAtom(la) => {
                self.lower_let_atom(la, span, out);
            }
            AStatementKind::Assign(assign) => {
                let tlhs = self.lower_lhs(assign.target);
                let tval = self.lower_atomic(assign.value);
                out.push(TStatement {
                    kind: TStatementKind::Assign(TAssign {
                        target: tlhs,
                        value: tval,
                        span: assign.span,
                    }),
                    span,
                });
            }
            AStatementKind::Expr(expr) => {
                let texpr = self.lower_expr(expr);
                out.push(TStatement {
                    kind: TStatementKind::Expr(texpr),
                    span,
                });
            }
            AStatementKind::Loop(body) => {
                let mut tbody = Vec::new();
                for s in body {
                    self.lower_statement(s, &mut tbody);
                }
                out.push(TStatement {
                    kind: TStatementKind::Loop(tbody),
                    span,
                });
            }
            AStatementKind::Return(atomic) => {
                if self.in_role_method {
                    // return val  →  var __ret = (s, val); return __ret;
                    let tval = self.lower_atomic(atomic);
                    let state_type = self.state_type.clone().unwrap();
                    let tuple_ty = Type::Tuple(vec![state_type, Type::Nil]);
                    let tuple_expr = TExpr {
                        kind: TExprKind::TupleLit(vec![self.s_atomic(), tval]),
                        ty: tuple_ty.clone(),
                        span,
                    };
                    let (tid, tname) = self.fresh_name("ret_tuple");
                    out.push(TStatement {
                        kind: TStatementKind::LetAtom(TLetAtom {
                            name: tid,
                            original_name: tname.clone(),
                            ty: tuple_ty,
                            value: tuple_expr,
                            span,
                        }),
                        span,
                    });
                    out.push(TStatement {
                        kind: TStatementKind::Return(TAtomic::Var(tid, tname)),
                        span,
                    });
                } else {
                    out.push(TStatement {
                        kind: TStatementKind::Return(self.lower_atomic(atomic)),
                        span,
                    });
                }
            }
            AStatementKind::Break => {
                out.push(TStatement {
                    kind: TStatementKind::Break,
                    span,
                });
            }
            AStatementKind::Continue => {
                out.push(TStatement {
                    kind: TStatementKind::Continue,
                    span,
                });
            }
            AStatementKind::Error => {
                out.push(TStatement {
                    kind: TStatementKind::Error,
                    span,
                });
            }
        }
    }

    /// Lower a LetAtom, potentially threading state through function calls.
    fn lower_let_atom(&mut self, la: ALetAtom, span: Span, out: &mut Vec<TStatement>) {
        if self.in_role_method {
            match &la.value.kind {
                AExprKind::FuncCall(AFuncCall::User(call)) if !call.is_free => {
                    // Role method call: var res = f(args)
                    // →  var __tup = f(s, args); s = __tup.0; var res = __tup.1;
                    let call_clone = call.clone();
                    self.emit_threaded_call(
                        la.name,
                        la.original_name,
                        la.ty,
                        call_clone,
                        span,
                        out,
                    );
                    return;
                }
                AExprKind::Send(_, _) | AExprKind::Recv(_) => {
                    // Send/Recv already have state injected via lower_expr.
                    // The threaded versions return (State, OriginalReturn).
                    // We need to unpack: var __tup = op; s = __tup.0; var res = __tup.1;
                    self.emit_threaded_send_recv(la, span, out);
                    return;
                }
                _ => {}
            }
        }

        // Default: 1:1 lowering
        let texpr = self.lower_expr(la.value);
        out.push(TStatement {
            kind: TStatementKind::LetAtom(TLetAtom {
                name: la.name,
                original_name: la.original_name,
                ty: la.ty,
                value: texpr,
                span: la.span,
            }),
            span,
        });
    }

    /// Emit threaded role-method call:
    /// ```text
    /// var res = f(args)
    /// →
    /// var __tup = f(s, args);
    /// s = __tup.0;
    /// var res = __tup.1;
    /// ```
    fn emit_threaded_call(
        &mut self,
        orig_name_id: NameId,
        orig_name: String,
        orig_ty: Type,
        call: AUserFuncCall,
        span: Span,
        out: &mut Vec<TStatement>,
    ) {
        let (tcall, _original_return) = self.lower_user_func_call_threaded(call);
        let tuple_ty = tcall.return_type.clone();
        let (tup_id, tup_name) = self.fresh_name("tup");

        // var __tup = f(s, args);
        out.push(TStatement {
            kind: TStatementKind::LetAtom(TLetAtom {
                name: tup_id,
                original_name: tup_name.clone(),
                ty: tuple_ty.clone(),
                value: TExpr {
                    kind: TExprKind::FuncCall(TFuncCall::User(tcall)),
                    ty: tuple_ty.clone(),
                    span,
                },
                span,
            }),
            span,
        });

        // s = __tup.0;
        let state_type = self.state_type.clone().unwrap();
        let (s_extract_id, s_extract_name) = self.fresh_name("s_new");
        out.push(TStatement {
            kind: TStatementKind::LetAtom(TLetAtom {
                name: s_extract_id,
                original_name: s_extract_name.clone(),
                ty: state_type,
                value: TExpr {
                    kind: TExprKind::TupleAccess(TAtomic::Var(tup_id, tup_name.clone()), 0),
                    ty: self.state_type.clone().unwrap(),
                    span,
                },
                span,
            }),
            span,
        });
        out.push(TStatement {
            kind: TStatementKind::Assign(TAssign {
                target: self.s_lhs(),
                value: TAtomic::Var(s_extract_id, s_extract_name),
                span,
            }),
            span,
        });

        // var res = __tup.1;
        out.push(TStatement {
            kind: TStatementKind::LetAtom(TLetAtom {
                name: orig_name_id,
                original_name: orig_name,
                ty: orig_ty.clone(),
                value: TExpr {
                    kind: TExprKind::TupleAccess(TAtomic::Var(tup_id, tup_name), 1),
                    ty: orig_ty,
                    span,
                },
                span,
            }),
            span,
        });
    }

    /// Emit threaded Send/Recv unpacking:
    /// The threaded Send/Recv return (State, OriginalValue).
    fn emit_threaded_send_recv(&mut self, la: ALetAtom, span: Span, out: &mut Vec<TStatement>) {
        let orig_ty = la.ty.clone();
        let state_type = self.state_type.clone().unwrap();
        let tuple_ty = Type::Tuple(vec![state_type.clone(), orig_ty.clone()]);

        // Lower the expression (Send/Recv with state injected)
        let texpr = self.lower_expr(la.value);

        let (tup_id, tup_name) = self.fresh_name("tup");

        // var __tup = Send/Recv(s, ...);
        out.push(TStatement {
            kind: TStatementKind::LetAtom(TLetAtom {
                name: tup_id,
                original_name: tup_name.clone(),
                ty: tuple_ty,
                value: TExpr {
                    kind: texpr.kind,
                    ty: Type::Tuple(vec![state_type.clone(), orig_ty.clone()]),
                    span,
                },
                span,
            }),
            span,
        });

        // s = __tup.0;
        let (s_extract_id, s_extract_name) = self.fresh_name("s_new");
        out.push(TStatement {
            kind: TStatementKind::LetAtom(TLetAtom {
                name: s_extract_id,
                original_name: s_extract_name.clone(),
                ty: state_type.clone(),
                value: TExpr {
                    kind: TExprKind::TupleAccess(TAtomic::Var(tup_id, tup_name.clone()), 0),
                    ty: state_type,
                    span,
                },
                span,
            }),
            span,
        });
        out.push(TStatement {
            kind: TStatementKind::Assign(TAssign {
                target: self.s_lhs(),
                value: TAtomic::Var(s_extract_id, s_extract_name),
                span,
            }),
            span,
        });

        // var res = __tup.1;
        out.push(TStatement {
            kind: TStatementKind::LetAtom(TLetAtom {
                name: la.name,
                original_name: la.original_name,
                ty: orig_ty.clone(),
                value: TExpr {
                    kind: TExprKind::TupleAccess(TAtomic::Var(tup_id, tup_name), 1),
                    ty: orig_ty,
                    span,
                },
                span,
            }),
            span,
        });
    }

    fn lower_func_def_role(&mut self, func: AFuncDef) -> TFuncDef {
        let state_type = self.state_type.clone().unwrap();

        // Add s as first parameter
        let mut params = vec![TFuncParam {
            name: self.state_name_id,
            original_name: "s".to_string(),
            ty: state_type.clone(),
            span: func.span,
        }];
        params.extend(func.params.into_iter().map(|p| TFuncParam {
            name: p.name,
            original_name: p.original_name,
            ty: p.ty,
            span: p.span,
        }));

        // Wrap return type: T → (State, T)
        let original_return = func.return_type.clone();
        let threaded_return = Type::Tuple(vec![state_type.clone(), original_return]);

        // Lower body
        self.in_role_method = true;
        let mut body = self.lower_block(func.body);

        // Wrap tail_expr: tail → TupleLit([s, tail])
        if let Some(tail) = body.tail_expr.take() {
            let tuple_expr = TExpr {
                kind: TExprKind::TupleLit(vec![self.s_atomic(), tail]),
                ty: threaded_return.clone(),
                span: body.span,
            };
            let (tid, tname) = self.fresh_name("ret_tuple");
            body.statements.push(TStatement {
                kind: TStatementKind::LetAtom(TLetAtom {
                    name: tid,
                    original_name: tname.clone(),
                    ty: threaded_return.clone(),
                    value: tuple_expr,
                    span: body.span,
                }),
                span: body.span,
            });
            body.tail_expr = Some(TAtomic::Var(tid, tname));
        }
        body.ty = threaded_return.clone();
        self.in_role_method = false;

        TFuncDef {
            name: func.name,
            original_name: func.original_name,
            is_sync: func.is_sync,
            is_traced: func.is_traced,
            params,
            return_type: threaded_return,
            body,
            span: func.span,
        }
    }

    fn lower_func_def_free(&mut self, func: AFuncDef) -> TFuncDef {
        self.in_role_method = false;
        let body = self.lower_block(func.body);
        TFuncDef {
            name: func.name,
            original_name: func.original_name,
            is_sync: func.is_sync,
            is_traced: func.is_traced,
            params: func
                .params
                .into_iter()
                .map(|p| TFuncParam {
                    name: p.name,
                    original_name: p.original_name,
                    ty: p.ty,
                    span: p.span,
                })
                .collect(),
            return_type: func.return_type,
            body,
            span: func.span,
        }
    }
}

pub fn lower_program(program: AProgram) -> TProgram {
    let mut lowerer = ThreadLowerer {
        next_name_id: program.next_name_id,
        role_var_ids: HashSet::new(),
        state_name_id: NameId(0),
        state_type: None,
        in_role_method: false,
    };

    let mut struct_defs = program.struct_defs.clone();
    let mut top_level_defs = Vec::new();

    for def in program.top_level_defs {
        match def {
            ATopLevelDef::Role(role) => {
                let (trole, state_struct_id, state_fields, init_func) =
                    lower_role(&mut lowerer, role);
                struct_defs.insert(state_struct_id, state_fields);
                top_level_defs.push(TTopLevelDef::FreeFunc(init_func));
                top_level_defs.push(TTopLevelDef::Role(trole));
            }
            ATopLevelDef::FreeFunc(func) => {
                let tfunc = lowerer.lower_func_def_free(func);
                top_level_defs.push(TTopLevelDef::FreeFunc(tfunc));
            }
        }
    }

    TProgram {
        top_level_defs,
        next_name_id: lowerer.next_name_id,
        id_to_name: program.id_to_name,
        struct_defs,
        enum_defs: program.enum_defs,
    }
}

fn lower_role(
    lowerer: &mut ThreadLowerer,
    role: ARoleDef,
) -> (TRoleDef, NameId, Vec<(String, Type)>, TFuncDef) {
    // 1. Extract state struct fields
    let state_fields: Vec<(String, Type)> = role
        .var_inits
        .iter()
        .map(|vi| (vi.original_name.clone(), vi.type_def.clone()))
        .collect();

    let (state_struct_id, state_struct_name) =
        lowerer.fresh_name(&format!("{}_state", role.original_name));
    let state_type = Type::Struct(state_struct_id, state_struct_name);

    // 2. Configure lowerer context
    lowerer.role_var_ids = role.var_inits.iter().map(|vi| vi.name).collect();
    let (s_id, _) = lowerer.fresh_name("s");
    lowerer.state_name_id = s_id;
    lowerer.state_type = Some(state_type.clone());

    // 3. Generate init function
    let init_func = generate_init_func(lowerer, &role, state_struct_id, &state_type);

    // 4. Lower each function
    let func_defs: Vec<TFuncDef> = role
        .func_defs
        .into_iter()
        .map(|f| lowerer.lower_func_def_role(f))
        .collect();

    // 5. Clear context
    lowerer.role_var_ids.clear();
    lowerer.state_type = None;

    let trole = TRoleDef {
        name: role.name,
        original_name: role.original_name,
        func_defs,
        span: role.span,
    };

    (trole, state_struct_id, state_fields, init_func)
}

fn generate_init_func(
    lowerer: &mut ThreadLowerer,
    role: &ARoleDef,
    state_struct_id: NameId,
    state_type: &Type,
) -> TFuncDef {
    let (func_name_id, func_name) = lowerer.fresh_name(&format!("{}_init", role.original_name));

    let mut body_stmts = Vec::new();
    let mut struct_fields: Vec<(String, TAtomic)> = Vec::new();

    for vi in &role.var_inits {
        for stmt in &vi.stmts {
            lowerer.lower_statement(stmt.clone(), &mut body_stmts);
        }

        let texpr = lowerer.lower_expr(vi.value.clone());
        match &texpr.kind {
            TExprKind::Atomic(a) => {
                struct_fields.push((vi.original_name.clone(), a.clone()));
            }
            _ => {
                let (tid, tname) = lowerer.fresh_name(&format!("init_{}", vi.original_name));
                body_stmts.push(TStatement {
                    kind: TStatementKind::LetAtom(TLetAtom {
                        name: tid,
                        original_name: tname.clone(),
                        ty: vi.type_def.clone(),
                        value: texpr,
                        span: vi.span,
                    }),
                    span: vi.span,
                });
                struct_fields.push((vi.original_name.clone(), TAtomic::Var(tid, tname)));
            }
        }
    }

    let struct_lit = TExpr {
        kind: TExprKind::StructLit(state_struct_id, struct_fields),
        ty: state_type.clone(),
        span: role.span,
    };

    let (ret_id, ret_name) = lowerer.fresh_name("init_result");
    body_stmts.push(TStatement {
        kind: TStatementKind::LetAtom(TLetAtom {
            name: ret_id,
            original_name: ret_name.clone(),
            ty: state_type.clone(),
            value: struct_lit,
            span: role.span,
        }),
        span: role.span,
    });

    TFuncDef {
        name: func_name_id,
        original_name: func_name,
        is_sync: true,
        is_traced: false,
        params: vec![],
        return_type: state_type.clone(),
        body: TBlock {
            statements: body_stmts,
            tail_expr: Some(TAtomic::Var(ret_id, ret_name)),
            ty: state_type.clone(),
            span: role.span,
        },
        span: role.span,
    }
}
