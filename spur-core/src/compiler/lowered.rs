use crate::analysis::resolver::NameId;
use crate::analysis::types::*;
use crate::parser::{BinOp, Span};

mod ast;

pub use ast::*;

pub fn lower_program(typed: TypedProgram) -> LProgram {
    let mut lowerer = Lowerer {
        next_name_id: typed.next_name_id,
    };

    let top_level_defs = typed
        .top_level_defs
        .into_iter()
        .map(|def| lowerer.lower_top_level_def(def))
        .collect();

    LProgram {
        top_level_defs,
        next_name_id: lowerer.next_name_id,
        id_to_name: typed.id_to_name,
        struct_defs: typed.struct_defs,
        enum_defs: typed.enum_defs,
    }
}

struct Lowerer {
    next_name_id: usize,
}

impl Lowerer {
    fn fresh_name_id(&mut self) -> NameId {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        id
    }

    fn lower_top_level_def(&mut self, def: TypedTopLevelDef) -> LTopLevelDef {
        match def {
            TypedTopLevelDef::Role(role) => LTopLevelDef::Role(self.lower_role_def(role)),
            TypedTopLevelDef::FreeFunc(func) => LTopLevelDef::FreeFunc(self.lower_func_def(func)),
        }
    }

    fn lower_role_def(&mut self, role: TypedRoleDef) -> LRoleDef {
        let var_inits = role
            .var_inits
            .into_iter()
            .flat_map(|vi| self.lower_var_init_to_stmts(vi))
            .filter_map(|stmt| {
                if let LStatementKind::VarInit(init) = stmt.kind {
                    Some(init)
                } else {
                    None
                }
            })
            .collect();
        let func_defs = role
            .func_defs
            .into_iter()
            .map(|f| self.lower_func_def(f))
            .collect();
        LRoleDef {
            name: role.name,
            original_name: role.original_name,
            var_inits,
            func_defs,
            span: role.span,
        }
    }

    fn lower_func_def(&mut self, func: TypedFuncDef) -> LFuncDef {
        let params = func
            .params
            .into_iter()
            .map(|p| LFuncParam {
                name: p.name,
                original_name: p.original_name,
                ty: p.ty,
                span: p.span,
            })
            .collect();
        let body = self.lower_block(func.body);
        LFuncDef {
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

    fn lower_block(&mut self, block: TypedBlock) -> LBlock {
        let statements = block
            .statements
            .into_iter()
            .flat_map(|s| self.lower_statement(s))
            .collect();
        let tail_expr = block.tail_expr.map(|e| Box::new(self.lower_expr(*e)));
        LBlock {
            statements,
            tail_expr,
            ty: block.ty,
            span: block.span,
        }
    }

    fn lower_statement(&mut self, stmt: TypedStatement) -> Vec<LStatement> {
        let span = stmt.span;
        match stmt.kind {
            TypedStatementKind::VarInit(vi) => self.lower_var_init_to_stmts(vi),
            TypedStatementKind::Assignment(a) => {
                vec![LStatement {
                    kind: LStatementKind::Assignment(self.lower_assignment(a)),
                    span,
                }]
            }
            TypedStatementKind::Expr(e) => {
                vec![LStatement {
                    kind: LStatementKind::Expr(self.lower_expr(e)),
                    span,
                }]
            }
            TypedStatementKind::ForLoop(fl) => vec![self.lower_for_loop(fl, span)],
            TypedStatementKind::ForInLoop(fil) => {
                vec![LStatement {
                    kind: LStatementKind::ForInLoop(self.lower_for_in_loop(fil)),
                    span,
                }]
            }
            TypedStatementKind::Error => vec![LStatement {
                kind: LStatementKind::Error,
                span,
            }],
        }
    }

    fn lower_statements(&mut self, stmts: Vec<TypedStatement>) -> Vec<LStatement> {
        stmts.into_iter().flat_map(|s| self.lower_statement(s)).collect()
    }

    fn lower_var_init_to_stmts(&mut self, vi: TypedVarInit) -> Vec<LStatement> {
        let pattern = var_target_to_pattern(vi.target, vi.type_def.clone(), vi.span);
        let value = self.lower_expr(vi.value);
        self.bind_pattern(&pattern, value)
    }

    fn lower_assignment(&mut self, a: TypedAssignment) -> LAssignment {
        LAssignment {
            target: self.lower_expr(a.target),
            value: self.lower_expr(a.value),
            span: a.span,
        }
    }

    fn lower_for_loop(&mut self, fl: TypedForLoop, span: Span) -> LStatement {
        let init = fl.init.map(|i| match i {
            TypedForLoopInit::VarInit(vi) => {
                let value = self.lower_expr(vi.value);
                match vi.target {
                    TypedVarTarget::Name(id, name) => {
                        LForLoopInit::VarInit(LVarInit {
                            name: id,
                            original_name: name,
                            type_def: vi.type_def,
                            value,
                            span: vi.span,
                        })
                    }
                    TypedVarTarget::Tuple(_) => {
                        panic!("Tuple destructuring in for-loop init is not supported")
                    }
                }
            }
            TypedForLoopInit::Assignment(a) => {
                LForLoopInit::Assignment(self.lower_assignment(a))
            }
        });
        let condition = fl.condition.map(|c| self.lower_expr(c));
        let increment = fl.increment.map(|i| self.lower_assignment(i));
        let body = self.lower_statements(fl.body);

        LStatement {
            kind: LStatementKind::ForLoop(LForLoop {
                init,
                condition,
                increment,
                body,
                span,
            }),
            span,
        }
    }

    fn lower_for_in_loop(&mut self, fil: TypedForInLoop) -> LForInLoop {
        let iterable = self.lower_expr(fil.iterable);
        let mut body = self.lower_statements(fil.body);

        match &fil.pattern.kind {
            TypedPatternKind::Var(id, name) => {
                LForInLoop {
                    binding_name: *id,
                    binding_original_name: name.clone(),
                    iterable,
                    body,
                    span: fil.span,
                }
            }
            TypedPatternKind::Wildcard => {
                let tmp_id = self.fresh_name_id();
                LForInLoop {
                    binding_name: tmp_id,
                    binding_original_name: "_".to_string(),
                    iterable,
                    body,
                    span: fil.span,
                }
            }
            _ => {
                // Tuple or other complex pattern: bind to a temp, prepend destructuring
                let tmp_id = self.fresh_name_id();
                let tmp_name = "__for_binding".to_string();
                let tmp_var = LExpr {
                    kind: LExprKind::Var(tmp_id, tmp_name.clone()),
                    ty: fil.pattern.ty.clone(),
                    span: fil.pattern.span,
                };
                let mut destructure_stmts = self.bind_pattern(&fil.pattern, tmp_var);
                destructure_stmts.append(&mut body);
                LForInLoop {
                    binding_name: tmp_id,
                    binding_original_name: tmp_name,
                    iterable,
                    body: destructure_stmts,
                    span: fil.span,
                }
            }
        }
    }

    fn lower_expr(&mut self, expr: TypedExpr) -> LExpr {
        let span = expr.span;
        let ty = expr.ty;
        match expr.kind {
            TypedExprKind::Var(id, name) => LExpr {
                kind: LExprKind::Var(id, name),
                ty,
                span,
            },
            TypedExprKind::IntLit(v) => LExpr {
                kind: LExprKind::IntLit(v),
                ty,
                span,
            },
            TypedExprKind::StringLit(v) => LExpr {
                kind: LExprKind::StringLit(v),
                ty,
                span,
            },
            TypedExprKind::BoolLit(v) => LExpr {
                kind: LExprKind::BoolLit(v),
                ty,
                span,
            },
            TypedExprKind::NilLit => LExpr {
                kind: LExprKind::NilLit,
                ty,
                span,
            },

            // Short-circuit operators → Conditional
            TypedExprKind::BinOp(BinOp::And, lhs, rhs) => {
                let lowered_lhs = self.lower_expr(*lhs);
                let lowered_rhs = self.lower_expr(*rhs);
                let rhs_span = lowered_rhs.span;
                LExpr {
                    kind: LExprKind::Conditional(Box::new(LCondExpr {
                        if_branch: LIfBranch {
                            condition: lowered_lhs,
                            body: LBlock {
                                statements: vec![],
                                tail_expr: Some(Box::new(lowered_rhs)),
                                ty: Type::Bool,
                                span: rhs_span,
                            },
                            span,
                        },
                        elseif_branches: vec![],
                        else_branch: Some(LBlock {
                            statements: vec![],
                            tail_expr: Some(Box::new(LExpr {
                                kind: LExprKind::BoolLit(false),
                                ty: Type::Bool,
                                span,
                            })),
                            ty: Type::Bool,
                            span,
                        }),
                        span,
                    })),
                    ty,
                    span,
                }
            }
            TypedExprKind::BinOp(BinOp::Or, lhs, rhs) => {
                let lowered_lhs = self.lower_expr(*lhs);
                let lowered_rhs = self.lower_expr(*rhs);
                let rhs_span = lowered_rhs.span;
                LExpr {
                    kind: LExprKind::Conditional(Box::new(LCondExpr {
                        if_branch: LIfBranch {
                            condition: lowered_lhs,
                            body: LBlock {
                                statements: vec![],
                                tail_expr: Some(Box::new(LExpr {
                                    kind: LExprKind::BoolLit(true),
                                    ty: Type::Bool,
                                    span,
                                })),
                                ty: Type::Bool,
                                span,
                            },
                            span,
                        },
                        elseif_branches: vec![],
                        else_branch: Some(LBlock {
                            statements: vec![],
                            tail_expr: Some(Box::new(lowered_rhs)),
                            ty: Type::Bool,
                            span: rhs_span,
                        }),
                        span,
                    })),
                    ty,
                    span,
                }
            }

            TypedExprKind::BinOp(op, lhs, rhs) => LExpr {
                kind: LExprKind::BinOp(
                    op,
                    Box::new(self.lower_expr(*lhs)),
                    Box::new(self.lower_expr(*rhs)),
                ),
                ty,
                span,
            },
            TypedExprKind::Not(e) => LExpr {
                kind: LExprKind::Not(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },
            TypedExprKind::Negate(e) => LExpr {
                kind: LExprKind::Negate(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },

            TypedExprKind::FuncCall(call) => LExpr {
                kind: LExprKind::FuncCall(self.lower_func_call(call)),
                ty,
                span,
            },
            TypedExprKind::MapLit(pairs) => LExpr {
                kind: LExprKind::MapLit(
                    pairs
                        .into_iter()
                        .map(|(k, v)| (self.lower_expr(k), self.lower_expr(v)))
                        .collect(),
                ),
                ty,
                span,
            },
            TypedExprKind::ListLit(items) => LExpr {
                kind: LExprKind::ListLit(items.into_iter().map(|e| self.lower_expr(e)).collect()),
                ty,
                span,
            },
            TypedExprKind::TupleLit(items) => LExpr {
                kind: LExprKind::TupleLit(items.into_iter().map(|e| self.lower_expr(e)).collect()),
                ty,
                span,
            },

            TypedExprKind::Append(a, b) => LExpr {
                kind: LExprKind::Append(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                ),
                ty,
                span,
            },
            TypedExprKind::Prepend(a, b) => LExpr {
                kind: LExprKind::Prepend(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                ),
                ty,
                span,
            },
            TypedExprKind::Min(a, b) => LExpr {
                kind: LExprKind::Min(Box::new(self.lower_expr(*a)), Box::new(self.lower_expr(*b))),
                ty,
                span,
            },
            TypedExprKind::Exists(a, b) => LExpr {
                kind: LExprKind::Exists(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                ),
                ty,
                span,
            },
            TypedExprKind::Erase(a, b) => LExpr {
                kind: LExprKind::Erase(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                ),
                ty,
                span,
            },
            TypedExprKind::Store(a, b, c) => LExpr {
                kind: LExprKind::Store(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                    Box::new(self.lower_expr(*c)),
                ),
                ty,
                span,
            },
            TypedExprKind::Head(e) => LExpr {
                kind: LExprKind::Head(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },
            TypedExprKind::Tail(e) => LExpr {
                kind: LExprKind::Tail(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },
            TypedExprKind::Len(e) => LExpr {
                kind: LExprKind::Len(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },

            TypedExprKind::RpcCall(target, call) => LExpr {
                kind: LExprKind::RpcCall(
                    Box::new(self.lower_expr(*target)),
                    self.lower_user_func_call(call),
                ),
                ty,
                span,
            },

            // Match → Conditional chain with IsVariant/VariantPayload
            TypedExprKind::Match(scrutinee, arms) => self.lower_match(*scrutinee, arms, ty, span),

            TypedExprKind::Conditional(cond) => LExpr {
                kind: LExprKind::Conditional(Box::new(self.lower_cond_expr(*cond))),
                ty,
                span,
            },
            TypedExprKind::Block(block) => LExpr {
                kind: LExprKind::Block(Box::new(self.lower_block(*block))),
                ty,
                span,
            },
            TypedExprKind::VariantLit(id, name, payload) => LExpr {
                kind: LExprKind::VariantLit(
                    id,
                    name,
                    payload.map(|e| Box::new(self.lower_expr(*e))),
                ),
                ty,
                span,
            },

            TypedExprKind::UnwrapOptional(e) => LExpr {
                kind: LExprKind::UnwrapOptional(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },

            TypedExprKind::MakeChannel => LExpr {
                kind: LExprKind::MakeChannel,
                ty,
                span,
            },
            TypedExprKind::Send(a, b) => LExpr {
                kind: LExprKind::Send(Box::new(self.lower_expr(*a)), Box::new(self.lower_expr(*b))),
                ty,
                span,
            },
            TypedExprKind::Recv(e) => LExpr {
                kind: LExprKind::Recv(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },
            TypedExprKind::SetTimer(label) => LExpr {
                kind: LExprKind::SetTimer(label),
                ty,
                span,
            },

            TypedExprKind::Index(a, b) => LExpr {
                kind: LExprKind::Index(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                ),
                ty,
                span,
            },
            TypedExprKind::Slice(a, b, c) => LExpr {
                kind: LExprKind::Slice(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                    Box::new(self.lower_expr(*c)),
                ),
                ty,
                span,
            },
            TypedExprKind::TupleAccess(e, idx) => LExpr {
                kind: LExprKind::TupleAccess(Box::new(self.lower_expr(*e)), idx),
                ty,
                span,
            },
            TypedExprKind::FieldAccess(e, name) => LExpr {
                kind: LExprKind::FieldAccess(Box::new(self.lower_expr(*e)), name),
                ty,
                span,
            },

            TypedExprKind::SafeFieldAccess(e, name) => LExpr {
                kind: LExprKind::SafeFieldAccess(Box::new(self.lower_expr(*e)), name),
                ty,
                span,
            },
            TypedExprKind::SafeIndex(a, b) => LExpr {
                kind: LExprKind::SafeIndex(
                    Box::new(self.lower_expr(*a)),
                    Box::new(self.lower_expr(*b)),
                ),
                ty,
                span,
            },
            TypedExprKind::SafeTupleAccess(e, idx) => LExpr {
                kind: LExprKind::SafeTupleAccess(Box::new(self.lower_expr(*e)), idx),
                ty,
                span,
            },

            TypedExprKind::StructLit(id, fields) => LExpr {
                kind: LExprKind::StructLit(
                    id,
                    fields
                        .into_iter()
                        .map(|(name, e)| (name, self.lower_expr(e)))
                        .collect(),
                ),
                ty,
                span,
            },

            TypedExprKind::WrapInOptional(e) => LExpr {
                kind: LExprKind::WrapInOptional(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },
            TypedExprKind::PersistData(e) => LExpr {
                kind: LExprKind::PersistData(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },
            TypedExprKind::RetrieveData(t) => LExpr {
                kind: LExprKind::RetrieveData(t),
                ty,
                span,
            },
            TypedExprKind::DiscardData => LExpr {
                kind: LExprKind::DiscardData,
                ty,
                span,
            },

            TypedExprKind::Return(e) => LExpr {
                kind: LExprKind::Return(Box::new(self.lower_expr(*e))),
                ty,
                span,
            },
            TypedExprKind::Break => LExpr {
                kind: LExprKind::Break,
                ty,
                span,
            },
            TypedExprKind::Continue => LExpr {
                kind: LExprKind::Continue,
                ty,
                span,
            },

            TypedExprKind::Error => LExpr {
                kind: LExprKind::Error,
                ty,
                span,
            },
        }
    }

    fn lower_match(
        &mut self,
        scrutinee: TypedExpr,
        arms: Vec<TypedMatchArm>,
        match_ty: Type,
        span: Span,
    ) -> LExpr {
        let scrutinee_span = scrutinee.span;
        let scrutinee_ty = scrutinee.ty.clone();
        let lowered_scrutinee = self.lower_expr(scrutinee);

        // Allocate a fresh variable for the scrutinee so it is evaluated once.
        let scrutinee_id = self.fresh_name_id();
        let scrutinee_name = "__match_scrutinee".to_string();
        let scrutinee_var = LExpr {
            kind: LExprKind::Var(scrutinee_id, scrutinee_name.clone()),
            ty: scrutinee_ty.clone(),
            span: scrutinee_span,
        };

        let cond_expr = self.lower_match_arms(&scrutinee_var, &arms, &match_ty, span);

        // Wrap in a block: { let __match_scrutinee = <scrutinee>; <cond_chain> }
        let init_stmt = LStatement {
            kind: LStatementKind::VarInit(LVarInit {
                name: scrutinee_id,
                original_name: scrutinee_name,
                type_def: scrutinee_ty,
                value: lowered_scrutinee,
                span: scrutinee_span,
            }),
            span: scrutinee_span,
        };

        LExpr {
            kind: LExprKind::Block(Box::new(LBlock {
                statements: vec![init_stmt],
                tail_expr: Some(Box::new(cond_expr)),
                ty: match_ty.clone(),
                span,
            })),
            ty: match_ty,
            span,
        }
    }

    fn lower_match_arms(
        &mut self,
        scrutinee_var: &LExpr,
        arms: &[TypedMatchArm],
        match_ty: &Type,
        span: Span,
    ) -> LExpr {
        if arms.is_empty() {
            return LExpr {
                kind: LExprKind::Error,
                ty: match_ty.clone(),
                span,
            };
        }

        let arm = &arms[0];
        let remaining = &arms[1..];

        match &arm.pattern.kind {
            TypedPatternKind::Variant(_enum_id, variant_name, payload_pat) => {
                let condition = LExpr {
                    kind: LExprKind::IsVariant(
                        Box::new(scrutinee_var.clone()),
                        variant_name.clone(),
                    ),
                    ty: Type::Bool,
                    span: arm.pattern.span,
                };

                let body = self.lower_match_arm_body(
                    scrutinee_var,
                    payload_pat.as_deref(),
                    &arm.body,
                    arm.span,
                );

                if remaining.is_empty() {
                    LExpr {
                        kind: LExprKind::Conditional(Box::new(LCondExpr {
                            if_branch: LIfBranch {
                                condition,
                                body,
                                span: arm.span,
                            },
                            elseif_branches: vec![],
                            else_branch: None,
                            span,
                        })),
                        ty: match_ty.clone(),
                        span,
                    }
                } else {
                    let else_expr = self.lower_match_arms(scrutinee_var, remaining, match_ty, span);
                    let else_block = LBlock {
                        statements: vec![],
                        tail_expr: Some(Box::new(else_expr)),
                        ty: match_ty.clone(),
                        span,
                    };
                    LExpr {
                        kind: LExprKind::Conditional(Box::new(LCondExpr {
                            if_branch: LIfBranch {
                                condition,
                                body,
                                span: arm.span,
                            },
                            elseif_branches: vec![],
                            else_branch: Some(else_block),
                            span,
                        })),
                        ty: match_ty.clone(),
                        span,
                    }
                }
            }

            // Wildcard or variable — catch-all, becomes the else branch.
            TypedPatternKind::Wildcard | TypedPatternKind::Var(_, _) => {
                let mut body = self.lower_block(arm.body.clone());

                // For Var patterns, bind the variable to the scrutinee.
                if let TypedPatternKind::Var(name_id, name) = &arm.pattern.kind {
                    let bind_stmt = LStatement {
                        kind: LStatementKind::VarInit(LVarInit {
                            name: *name_id,
                            original_name: name.clone(),
                            type_def: arm.pattern.ty.clone(),
                            value: scrutinee_var.clone(),
                            span: arm.pattern.span,
                        }),
                        span: arm.pattern.span,
                    };
                    body.statements.insert(0, bind_stmt);
                }

                LExpr {
                    kind: LExprKind::Block(Box::new(body)),
                    ty: match_ty.clone(),
                    span,
                }
            }
            TypedPatternKind::Tuple(_) => {
                let mut body = self.lower_block(arm.body.clone());
                let bind_stmts = self.bind_pattern(&arm.pattern, scrutinee_var.clone());
                let mut new_stmts = bind_stmts;
                new_stmts.append(&mut body.statements);
                body.statements = new_stmts;
                LExpr {
                    kind: LExprKind::Block(Box::new(body)),
                    ty: match_ty.clone(),
                    span,
                }
            }
            TypedPatternKind::Error => {
                let body = self.lower_block(arm.body.clone());
                LExpr {
                    kind: LExprKind::Block(Box::new(body)),
                    ty: match_ty.clone(),
                    span,
                }
            }
        }
    }

    fn lower_match_arm_body(
        &mut self,
        scrutinee_var: &LExpr,
        payload_pat: Option<&TypedPattern>,
        body: &TypedBlock,
        _arm_span: Span,
    ) -> LBlock {
        let mut lowered_block = self.lower_block(body.clone());

        if let Some(pat) = payload_pat {
            let payload_expr = LExpr {
                kind: LExprKind::VariantPayload(Box::new(scrutinee_var.clone())),
                ty: pat.ty.clone(),
                span: pat.span,
            };
            let bind_stmts = self.bind_pattern(pat, payload_expr);
            let mut new_stmts = bind_stmts;
            new_stmts.append(&mut lowered_block.statements);
            lowered_block.statements = new_stmts;
        }

        lowered_block
    }

    fn bind_pattern(&mut self, pat: &TypedPattern, value: LExpr) -> Vec<LStatement> {
        match &pat.kind {
            TypedPatternKind::Var(name_id, name) => {
                vec![LStatement {
                    kind: LStatementKind::VarInit(LVarInit {
                        name: *name_id,
                        original_name: name.clone(),
                        type_def: pat.ty.clone(),
                        value,
                        span: pat.span,
                    }),
                    span: pat.span,
                }]
            }
            TypedPatternKind::Wildcard => {
                vec![]
            }
            TypedPatternKind::Tuple(sub_pats) => {
                let mut stmts = Vec::new();
                for (i, sub_pat) in sub_pats.iter().enumerate() {
                    let access = LExpr {
                        kind: LExprKind::TupleAccess(Box::new(value.clone()), i),
                        ty: sub_pat.ty.clone(),
                        span: sub_pat.span,
                    };
                    stmts.extend(self.bind_pattern(sub_pat, access));
                }
                stmts
            }
            TypedPatternKind::Variant(_, _, payload_pat) => {
                if let Some(payload_pat) = payload_pat {
                    let payload_expr = LExpr {
                        kind: LExprKind::VariantPayload(Box::new(value)),
                        ty: payload_pat.ty.clone(),
                        span: payload_pat.span,
                    };
                    self.bind_pattern(payload_pat, payload_expr)
                } else {
                    vec![]
                }
            }
            TypedPatternKind::Error => vec![],
        }
    }

    fn lower_cond_expr(&mut self, cond: TypedCondExpr) -> LCondExpr {
        let if_branch = LIfBranch {
            condition: self.lower_expr(cond.if_branch.condition),
            body: self.lower_block(cond.if_branch.body),
            span: cond.if_branch.span,
        };
        let elseif_branches = cond
            .elseif_branches
            .into_iter()
            .map(|b| LIfBranch {
                condition: self.lower_expr(b.condition),
                body: self.lower_block(b.body),
                span: b.span,
            })
            .collect();
        let else_branch = cond.else_branch.map(|b| self.lower_block(b));
        LCondExpr {
            if_branch,
            elseif_branches,
            else_branch,
            span: cond.span,
        }
    }

    fn lower_func_call(&mut self, call: TypedFuncCall) -> LFuncCall {
        match call {
            TypedFuncCall::User(uc) => LFuncCall::User(self.lower_user_func_call(uc)),
            TypedFuncCall::Builtin(builtin, args, ret_ty) => LFuncCall::Builtin(
                builtin,
                args.into_iter().map(|a| self.lower_expr(a)).collect(),
                ret_ty,
            ),
        }
    }

    fn lower_user_func_call(&mut self, call: TypedUserFuncCall) -> LUserFuncCall {
        LUserFuncCall {
            name: call.name,
            original_name: call.original_name,
            args: call.args.into_iter().map(|a| self.lower_expr(a)).collect(),
            return_type: call.return_type,
            span: call.span,
        }
    }
}

fn var_target_to_pattern(target: TypedVarTarget, type_def: Type, span: Span) -> TypedPattern {
    match target {
        TypedVarTarget::Name(id, name) => TypedPattern {
            kind: TypedPatternKind::Var(id, name),
            ty: type_def,
            span,
        },
        TypedVarTarget::Tuple(elements) => {
            let sub_pats = elements
                .into_iter()
                .map(|(id, name, ty)| TypedPattern {
                    kind: TypedPatternKind::Var(id, name),
                    ty,
                    span,
                })
                .collect();
            TypedPattern {
                kind: TypedPatternKind::Tuple(sub_pats),
                ty: type_def,
                span,
            }
        }
    }
}

#[cfg(test)]
mod test;