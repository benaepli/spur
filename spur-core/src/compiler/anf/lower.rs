use crate::analysis::resolver::NameId;
use crate::analysis::types::Type;
use crate::compiler::lowered::*;
use crate::parser::Span;

use super::ast::*;

struct AnfLowerer {
    next_name_id: usize,
}

impl AnfLowerer {
    fn fresh_name(&mut self) -> (NameId, String) {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        let name = format!("__anf_{}", id.0);
        (id, name)
    }

    fn lower_expr_to_atomic(&mut self, expr: LExpr, stmts: &mut Vec<AStatement>) -> AAtomic {
        let span = expr.span;
        let ty = expr.ty.clone();

        // If already atomic, return directly
        match &expr.kind {
            LExprKind::Var(id, name) => return AAtomic::Var(*id, name.clone()),
            LExprKind::IntLit(v) => return AAtomic::IntLit(*v),
            LExprKind::StringLit(v) => return AAtomic::StringLit(v.clone()),
            LExprKind::BoolLit(v) => return AAtomic::BoolLit(*v),
            LExprKind::NilLit => return AAtomic::NilLit,
            _ => {}
        }

        // Otherwise, lower to AExpr and bind to a fresh variable
        let aexpr = self.lower_expr_to_aexpr(expr, stmts);
        self.bind_aexpr(aexpr, ty, span, stmts)
    }

    /// Bind an AExpr to a fresh temp and return the AAtomic::Var for it.
    fn bind_aexpr(
        &mut self,
        aexpr: AExpr,
        ty: Type,
        span: Span,
        stmts: &mut Vec<AStatement>,
    ) -> AAtomic {
        let (id, name) = self.fresh_name();
        stmts.push(AStatement {
            kind: AStatementKind::LetAtom(ALetAtom {
                name: id,
                original_name: name.clone(),
                ty: ty,
                value: aexpr,
                span,
            }),
            span,
        });
        AAtomic::Var(id, name)
    }

    fn lower_expr_to_aexpr(&mut self, expr: LExpr, stmts: &mut Vec<AStatement>) -> AExpr {
        let span = expr.span;
        let ty = expr.ty;

        match expr.kind {
            // Atomics → wrap
            LExprKind::Var(id, name) => AExpr {
                kind: AExprKind::Atomic(AAtomic::Var(id, name)),
                ty,
                span,
            },
            LExprKind::IntLit(v) => AExpr {
                kind: AExprKind::Atomic(AAtomic::IntLit(v)),
                ty,
                span,
            },
            LExprKind::StringLit(v) => AExpr {
                kind: AExprKind::Atomic(AAtomic::StringLit(v)),
                ty,
                span,
            },
            LExprKind::BoolLit(v) => AExpr {
                kind: AExprKind::Atomic(AAtomic::BoolLit(v)),
                ty,
                span,
            },
            LExprKind::NilLit => AExpr {
                kind: AExprKind::Atomic(AAtomic::NilLit),
                ty,
                span,
            },

            // Binary ops
            LExprKind::BinOp(op, lhs, rhs) => {
                let a = self.lower_expr_to_atomic(*lhs, stmts);
                let b = self.lower_expr_to_atomic(*rhs, stmts);
                AExpr {
                    kind: AExprKind::BinOp(op, a, b),
                    ty,
                    span,
                }
            }
            LExprKind::Not(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::Not(a),
                    ty,
                    span,
                }
            }
            LExprKind::Negate(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::Negate(a),
                    ty,
                    span,
                }
            }

            // Function calls
            LExprKind::FuncCall(call) => {
                let acall = self.lower_func_call(call, stmts);
                AExpr {
                    kind: AExprKind::FuncCall(acall),
                    ty,
                    span,
                }
            }

            // Collection literals
            LExprKind::MapLit(pairs) => {
                let apairs: Vec<_> = pairs
                    .into_iter()
                    .map(|(k, v)| {
                        let ak = self.lower_expr_to_atomic(k, stmts);
                        let av = self.lower_expr_to_atomic(v, stmts);
                        (ak, av)
                    })
                    .collect();
                AExpr {
                    kind: AExprKind::MapLit(apairs),
                    ty,
                    span,
                }
            }
            LExprKind::ListLit(items) => {
                let aitems: Vec<_> = items
                    .into_iter()
                    .map(|e| self.lower_expr_to_atomic(e, stmts))
                    .collect();
                AExpr {
                    kind: AExprKind::ListLit(aitems),
                    ty,
                    span,
                }
            }
            LExprKind::TupleLit(items) => {
                let aitems: Vec<_> = items
                    .into_iter()
                    .map(|e| self.lower_expr_to_atomic(e, stmts))
                    .collect();
                AExpr {
                    kind: AExprKind::TupleLit(aitems),
                    ty,
                    span,
                }
            }

            // Two-arg builtins
            LExprKind::Append(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::Append(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::Prepend(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::Prepend(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::Min(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::Min(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::Exists(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::Exists(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::Erase(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::Erase(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::Store(a, b, c) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                let ac = self.lower_expr_to_atomic(*c, stmts);
                AExpr {
                    kind: AExprKind::Store(aa, ab, ac),
                    ty,
                    span,
                }
            }

            // One-arg builtins
            LExprKind::Head(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::Head(a),
                    ty,
                    span,
                }
            }
            LExprKind::Tail(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::Tail(a),
                    ty,
                    span,
                }
            }
            LExprKind::Len(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::Len(a),
                    ty,
                    span,
                }
            }

            // RPC
            LExprKind::RpcCall(target, call) => {
                let atarget = self.lower_expr_to_atomic(*target, stmts);
                let acall = self.lower_user_func_call(call, stmts);
                AExpr {
                    kind: AExprKind::RpcCall(atarget, acall),
                    ty,
                    span,
                }
            }

            // Conditional (expression)
            LExprKind::Conditional(cond) => {
                let acond = self.lower_cond_expr_in_stmts(*cond, stmts);
                AExpr {
                    kind: AExprKind::Conditional(Box::new(acond)),
                    ty,
                    span,
                }
            }

            // Block (expression)
            LExprKind::Block(block) => {
                let ablock = self.lower_block(*block);
                AExpr {
                    kind: AExprKind::Block(Box::new(ablock)),
                    ty,
                    span,
                }
            }

            // Variant
            LExprKind::VariantLit(id, name, payload) => {
                let apayload = payload.map(|e| self.lower_expr_to_atomic(*e, stmts));
                AExpr {
                    kind: AExprKind::VariantLit(id, name, apayload),
                    ty,
                    span,
                }
            }
            LExprKind::IsVariant(e, name) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::IsVariant(a, name),
                    ty,
                    span,
                }
            }
            LExprKind::VariantPayload(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::VariantPayload(a),
                    ty,
                    span,
                }
            }

            LExprKind::UnwrapOptional(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::UnwrapOptional(a),
                    ty,
                    span,
                }
            }

            // Iterators
            LExprKind::MakeIter(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::MakeIter(a),
                    ty,
                    span,
                }
            }
            LExprKind::IterIsDone(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::IterIsDone(a),
                    ty,
                    span,
                }
            }
            LExprKind::IterNext(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::IterNext(a),
                    ty,
                    span,
                }
            }

            // Channels
            LExprKind::MakeChannel => AExpr {
                kind: AExprKind::MakeChannel,
                ty,
                span,
            },
            LExprKind::Send(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::Send(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::Recv(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::Recv(a),
                    ty,
                    span,
                }
            }

            LExprKind::SetTimer(label) => AExpr {
                kind: AExprKind::SetTimer(label),
                ty,
                span,
            },
            LExprKind::Fifo(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::Fifo(a),
                    ty,
                    span,
                }
            }

            // Indexing / access
            LExprKind::Index(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::Index(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::Slice(a, b, c) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                let ac = self.lower_expr_to_atomic(*c, stmts);
                AExpr {
                    kind: AExprKind::Slice(aa, ab, ac),
                    ty,
                    span,
                }
            }
            LExprKind::TupleAccess(e, idx) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::TupleAccess(a, idx),
                    ty,
                    span,
                }
            }
            LExprKind::FieldAccess(e, name) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::FieldAccess(a, name),
                    ty,
                    span,
                }
            }

            LExprKind::SafeFieldAccess(e, name) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::SafeFieldAccess(a, name),
                    ty,
                    span,
                }
            }
            LExprKind::SafeIndex(a, b) => {
                let aa = self.lower_expr_to_atomic(*a, stmts);
                let ab = self.lower_expr_to_atomic(*b, stmts);
                AExpr {
                    kind: AExprKind::SafeIndex(aa, ab),
                    ty,
                    span,
                }
            }
            LExprKind::SafeTupleAccess(e, idx) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::SafeTupleAccess(a, idx),
                    ty,
                    span,
                }
            }

            // Struct
            LExprKind::StructLit(id, fields) => {
                let afields: Vec<_> = fields
                    .into_iter()
                    .map(|(name, e)| {
                        let a = self.lower_expr_to_atomic(e, stmts);
                        (name, a)
                    })
                    .collect();
                AExpr {
                    kind: AExprKind::StructLit(id, afields),
                    ty,
                    span,
                }
            }

            LExprKind::WrapInOptional(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::WrapInOptional(a),
                    ty,
                    span,
                }
            }
            LExprKind::PersistData(e) => {
                let a = self.lower_expr_to_atomic(*e, stmts);
                AExpr {
                    kind: AExprKind::PersistData(a),
                    ty,
                    span,
                }
            }
            LExprKind::RetrieveData(t) => AExpr {
                kind: AExprKind::RetrieveData(t),
                ty,
                span,
            },
            LExprKind::DiscardData => AExpr {
                kind: AExprKind::DiscardData,
                ty,
                span,
            },

            // Control flow that appears as expressions in lowered AST
            LExprKind::Return(e) => {
                // Return is handled at statement level; if it appears as an
                // expression, lower the inner to atomic and wrap as a block
                // containing a Return statement.
                let inner = self.lower_expr_to_atomic(*e, stmts);
                // We emit a Return statement into stmts directly; there is no
                // meaningful "value" for the expression after a return.
                stmts.push(AStatement {
                    kind: AStatementKind::Return(inner),
                    span,
                });
                // Return a dummy Never as the "value" — it will never be used.
                AExpr {
                    kind: AExprKind::Atomic(AAtomic::Never),
                    ty,
                    span,
                }
            }
            LExprKind::Break => {
                stmts.push(AStatement {
                    kind: AStatementKind::Break,
                    span,
                });
                AExpr {
                    kind: AExprKind::Atomic(AAtomic::Never),
                    ty,
                    span,
                }
            }
            LExprKind::Continue => {
                stmts.push(AStatement {
                    kind: AStatementKind::Continue,
                    span,
                });
                AExpr {
                    kind: AExprKind::Atomic(AAtomic::Never),
                    ty,
                    span,
                }
            }

            LExprKind::Error => AExpr {
                kind: AExprKind::Atomic(AAtomic::Never),
                ty,
                span,
            },
        }
    }
    fn lower_func_call(&mut self, call: LFuncCall, stmts: &mut Vec<AStatement>) -> AFuncCall {
        match call {
            LFuncCall::User(u) => AFuncCall::User(self.lower_user_func_call(u, stmts)),
            LFuncCall::Builtin(builtin, args, ret_ty) => {
                let aargs: Vec<_> = args
                    .into_iter()
                    .map(|e| self.lower_expr_to_atomic(e, stmts))
                    .collect();
                AFuncCall::Builtin(builtin, aargs, ret_ty)
            }
        }
    }

    fn lower_user_func_call(
        &mut self,
        call: LUserFuncCall,
        stmts: &mut Vec<AStatement>,
    ) -> AUserFuncCall {
        let aargs: Vec<_> = call
            .args
            .into_iter()
            .map(|e| self.lower_expr_to_atomic(e, stmts))
            .collect();
        AUserFuncCall {
            name: call.name,
            original_name: call.original_name,
            args: aargs,
            return_type: call.return_type,
            is_free: call.is_free,
            span: call.span,
        }
    }

    fn lower_block(&mut self, block: LBlock) -> ABlock {
        let mut astmts = Vec::new();
        for stmt in block.statements {
            self.lower_statement(stmt, &mut astmts);
        }
        let tail = block
            .tail_expr
            .map(|e| self.lower_expr_to_atomic(*e, &mut astmts));
        ABlock {
            statements: astmts,
            tail_expr: tail,
            ty: block.ty,
            span: block.span,
        }
    }

    /// Lower a conditional expression. Accepts the parent's statement
    /// accumulator so that the if-branch condition bindings (which must
    /// execute eagerly) can be hoisted into the enclosing scope.
    fn lower_cond_expr_in_stmts(
        &mut self,
        cond: LCondExpr,
        parent_stmts: &mut Vec<AStatement>,
    ) -> ACondExpr {
        // The if-branch condition is always evaluated, so any bindings
        // produced by flattening it belong in the parent scope.
        let if_branch = self.lower_if_branch_eager(cond.if_branch, parent_stmts);

        // Elseif conditions only execute when prior branches fail, so
        // their flattening bindings get prepended into the branch body.
        let elseif_branches: Vec<_> = cond
            .elseif_branches
            .into_iter()
            .map(|b| self.lower_if_branch_lazy(b))
            .collect();

        let else_branch = cond.else_branch.map(|b| self.lower_block(b));
        ACondExpr {
            if_branch,
            elseif_branches,
            else_branch,
            span: cond.span,
        }
    }

    /// Flatten the branch condition into `parent_stmts` (eager evaluation).
    fn lower_if_branch_eager(
        &mut self,
        branch: LIfBranch,
        parent_stmts: &mut Vec<AStatement>,
    ) -> AIfBranch {
        let acond = self.lower_expr_to_atomic(branch.condition, parent_stmts);
        let body = self.lower_block(branch.body);
        AIfBranch {
            condition: acond,
            body,
            span: branch.span,
        }
    }

    /// Flatten the branch condition into the branch body prefix (lazy).
    /// The condition bindings only execute when this branch is reached.
    fn lower_if_branch_lazy(&mut self, branch: LIfBranch) -> AIfBranch {
        let mut cond_stmts = Vec::new();
        let acond = self.lower_expr_to_atomic(branch.condition, &mut cond_stmts);
        let mut body = self.lower_block(branch.body);
        if !cond_stmts.is_empty() {
            cond_stmts.extend(body.statements);
            body.statements = cond_stmts;
        }
        AIfBranch {
            condition: acond,
            body,
            span: branch.span,
        }
    }

    fn lower_statement(&mut self, stmt: LStatement, stmts: &mut Vec<AStatement>) {
        let span = stmt.span;
        match stmt.kind {
            LStatementKind::VarInit(init) => {
                let aexpr = self.lower_expr_to_aexpr(init.value, stmts);
                stmts.push(AStatement {
                    kind: AStatementKind::LetAtom(ALetAtom {
                        name: init.name,
                        original_name: init.original_name,
                        ty: init.type_def,
                        value: aexpr,
                        span: init.span,
                    }),
                    span,
                });
            }
            LStatementKind::Assignment(assign) => {
                let aval = self.lower_expr_to_atomic(assign.value, stmts);
                stmts.push(AStatement {
                    kind: AStatementKind::Assign(AAssign {
                        target_id: assign.target_id,
                        target_name: assign.target_name,
                        ty: assign.ty,
                        value: aval,
                        span: assign.span,
                    }),
                    span,
                });
            }
            LStatementKind::Expr(expr) => {
                // Handle control flow expressions at statement level
                match expr.kind {
                    LExprKind::Return(inner) => {
                        let a = self.lower_expr_to_atomic(*inner, stmts);
                        stmts.push(AStatement {
                            kind: AStatementKind::Return(a),
                            span,
                        });
                    }
                    LExprKind::Break => {
                        stmts.push(AStatement {
                            kind: AStatementKind::Break,
                            span,
                        });
                    }
                    LExprKind::Continue => {
                        stmts.push(AStatement {
                            kind: AStatementKind::Continue,
                            span,
                        });
                    }
                    _ => {
                        let aexpr = self.lower_expr_to_aexpr(
                            LExpr {
                                kind: expr.kind,
                                ty: expr.ty,
                                span: expr.span,
                            },
                            stmts,
                        );
                        stmts.push(AStatement {
                            kind: AStatementKind::Expr(aexpr),
                            span,
                        });
                    }
                }
            }
            LStatementKind::ForLoop(_) => {
                panic!("ForLoop should have been removed before ANF lowering");
            }
            LStatementKind::ForInLoop(_) => {
                panic!("ForInLoop should have been removed before ANF lowering");
            }
            LStatementKind::Loop(body) => {
                let mut abody = Vec::new();
                for s in body {
                    self.lower_statement(s, &mut abody);
                }
                stmts.push(AStatement {
                    kind: AStatementKind::Loop(abody),
                    span,
                });
            }
            LStatementKind::Error => {
                stmts.push(AStatement {
                    kind: AStatementKind::Error,
                    span,
                });
            }
        }
    }

    fn lower_func_def(&mut self, func: LFuncDef) -> AFuncDef {
        let params = func
            .params
            .into_iter()
            .map(|p| AFuncParam {
                name: p.name,
                original_name: p.original_name,
                ty: p.ty,
                span: p.span,
            })
            .collect();
        let body = self.lower_block(func.body);
        AFuncDef {
            name: func.name,
            original_name: func.original_name,
            is_sync: func.is_sync,
            is_traced: func.is_traced,
            params,
            return_type: func.return_type,
            body,
            span: func.span,
        }
    }

    fn lower_role_def(&mut self, role: LRoleDef) -> ARoleDef {
        let var_inits = role
            .var_inits
            .into_iter()
            .map(|vi| {
                let mut stmts = Vec::new();
                let aexpr = self.lower_expr_to_aexpr(vi.value, &mut stmts);
                AVarInit {
                    name: vi.name,
                    original_name: vi.original_name,
                    type_def: vi.type_def,
                    stmts,
                    value: aexpr,
                    span: vi.span,
                }
            })
            .collect();
        let func_defs = role
            .func_defs
            .into_iter()
            .map(|f| self.lower_func_def(f))
            .collect();
        ARoleDef {
            name: role.name,
            original_name: role.original_name,
            var_inits,
            func_defs,
            span: role.span,
        }
    }

    fn lower_top_level_def(&mut self, def: LTopLevelDef) -> ATopLevelDef {
        match def {
            LTopLevelDef::Role(r) => ATopLevelDef::Role(self.lower_role_def(r)),
            LTopLevelDef::FreeFunc(f) => ATopLevelDef::FreeFunc(self.lower_func_def(f)),
        }
    }
}

pub fn lower_program(program: LProgram) -> AProgram {
    let mut lowerer = AnfLowerer {
        next_name_id: program.next_name_id,
    };
    let top_level_defs = program
        .top_level_defs
        .into_iter()
        .map(|def| lowerer.lower_top_level_def(def))
        .collect();
    AProgram {
        top_level_defs,
        next_name_id: lowerer.next_name_id,
        id_to_name: program.id_to_name,
        struct_defs: program.struct_defs,
        enum_defs: program.enum_defs,
    }
}
