use super::ast::*;
use crate::analysis::resolver::NameId;

pub fn remove_for_loops(program: &mut LProgram) {
    let mut remover = LoopRemover {
        next_name_id: &mut program.next_name_id,
        id_to_name: &mut program.id_to_name,
    };
    for def in &mut program.top_level_defs {
        match def {
            LTopLevelDef::Role(role) => {
                for func in &mut role.func_defs {
                    remover.remove_for_loops_in_func(func);
                }
            }
            LTopLevelDef::FreeFunc(func) => {
                remover.remove_for_loops_in_func(func);
            }
        }
    }
}

pub struct LoopRemover<'a> {
    next_name_id: &'a mut usize,
    id_to_name: &'a mut std::collections::HashMap<NameId, String>,
}

impl<'a> LoopRemover<'a> {
    fn fresh_name_id(&mut self) -> NameId {
        let id = NameId(*self.next_name_id);
        *self.next_name_id += 1;
        id
    }

    fn remove_for_loops_in_func(&mut self, func: &mut LFuncDef) {
        self.remove_for_loops_in_block(&mut func.body);
    }

    fn remove_for_loops_in_block(&mut self, block: &mut LBlock) {
        let mut new_stmts = Vec::new();
        for stmt in std::mem::take(&mut block.statements) {
            self.remove_for_loops_in_stmt(stmt, &mut new_stmts);
        }
        block.statements = new_stmts;

        if let Some(tail) = &mut block.tail_expr {
            self.remove_for_loops_in_expr(tail);
        }
    }

    fn remove_for_loops_in_stmt(&mut self, mut stmt: LStatement, out: &mut Vec<LStatement>) {
        let span = stmt.span;
        match &mut stmt.kind {
            LStatementKind::VarInit(vi) => {
                self.remove_for_loops_in_expr(&mut vi.value);
                out.push(stmt);
            }
            LStatementKind::Assignment(a) => {
                self.remove_for_loops_in_expr(&mut a.value);
                out.push(stmt);
            }
            LStatementKind::Expr(e) => {
                self.remove_for_loops_in_expr(e);
                out.push(stmt);
            }
            LStatementKind::Loop(stmts) => {
                let mut new_inner = Vec::new();
                for s in std::mem::take(stmts) {
                    self.remove_for_loops_in_stmt(s, &mut new_inner);
                }
                *stmts = new_inner;
                out.push(stmt);
            }
            LStatementKind::ForInLoop(fil) => {
                let mut iterable = fil.iterable.clone();
                self.remove_for_loops_in_expr(&mut iterable);

                let mut body = fil.body.clone();

                let element_ty = match &iterable.ty {
                    crate::analysis::types::Type::List(t) => *t.clone(),
                    crate::analysis::types::Type::Map(k, v) => crate::analysis::types::Type::Tuple(vec![*k.clone(), *v.clone()]),
                    _ => crate::analysis::types::Type::Error,
                };

                let tmp_iter = self.fresh_name_id();
                let tmp_iter_name = "__tmp_iter".to_string();
                self.id_to_name.insert(tmp_iter, tmp_iter_name.clone());

                let next_res = self.fresh_name_id();
                let next_res_name = "__next_res".to_string();
                self.id_to_name.insert(next_res, next_res_name.clone());

                let iter_ty = crate::analysis::types::Type::Iter(Box::new(element_ty.clone()));
                let tuple_ty = crate::analysis::types::Type::Tuple(vec![element_ty.clone(), iter_ty.clone()]);
                
                let init_stmt = LForLoopInit::VarInit(LVarInit {
                    name: tmp_iter,
                    original_name: tmp_iter_name.clone(),
                    type_def: iter_ty.clone(),
                    value: LExpr {
                        kind: LExprKind::MakeIter(Box::new(iterable)),
                        ty: iter_ty.clone(),
                        span: fil.span,
                    },
                    span: fil.span,
                });

                let iter_var = LExpr {
                    kind: LExprKind::Var(tmp_iter, tmp_iter_name.clone()),
                    ty: iter_ty.clone(),
                    span: fil.span,
                };

                let condition = LExpr {
                    kind: LExprKind::Not(Box::new(LExpr {
                        kind: LExprKind::IterIsDone(Box::new(iter_var.clone())),
                        ty: crate::analysis::types::Type::Bool,
                        span: fil.span,
                    })),
                    ty: crate::analysis::types::Type::Bool,
                    span: fil.span,
                };

                let next_res_var = LExpr {
                    kind: LExprKind::Var(next_res, next_res_name.clone()),
                    ty: tuple_ty.clone(),
                    span: fil.span,
                };

                let increment = vec![LStatement {
                    kind: LStatementKind::Assignment(LAssignment {
                        target_id: tmp_iter,
                        target_name: tmp_iter_name.clone(),
                        ty: iter_ty.clone(),
                        value: LExpr {
                            kind: LExprKind::TupleAccess(Box::new(next_res_var.clone()), 1),
                            ty: iter_ty.clone(),
                            span: fil.span,
                        },
                        span: fil.span,
                    }),
                    span: fil.span,
                }];

                let mut new_body = vec![
                    LStatement {
                        kind: LStatementKind::VarInit(LVarInit {
                            name: next_res,
                            original_name: next_res_name.clone(),
                            type_def: tuple_ty.clone(),
                            value: LExpr {
                                kind: LExprKind::IterNext(Box::new(iter_var.clone())),
                                ty: tuple_ty.clone(),
                                span: fil.span,
                            },
                            span: fil.span,
                        }),
                        span: fil.span,
                    },
                    LStatement {
                        kind: LStatementKind::VarInit(LVarInit {
                            name: fil.binding_name,
                            original_name: fil.binding_original_name.clone(),
                            type_def: element_ty.clone(),
                            value: LExpr {
                                kind: LExprKind::TupleAccess(Box::new(next_res_var.clone()), 0),
                                ty: element_ty.clone(),
                                span: fil.span,
                            },
                            span: fil.span,
                        }),
                        span: fil.span,
                    },
                ];

                new_body.extend(body);

                let new_for_loop = LStatement {
                    kind: LStatementKind::ForLoop(LForLoop {
                        init: Some(init_stmt),
                        condition: Some(condition),
                        increment,
                        body: new_body,
                        span: fil.span,
                    }),
                    span: fil.span,
                };

                self.remove_for_loops_in_stmt(new_for_loop, out);
            }
            LStatementKind::ForLoop(fl) => {
                let mut transformed_body = Vec::new();
                for s in std::mem::take(&mut fl.body) {
                    self.remove_for_loops_in_stmt(s, &mut transformed_body);
                }

                let increment_clone = fl.increment.clone();
                if !increment_clone.is_empty() {
                    replace_continue_in_stmts(&mut transformed_body, &increment_clone);
                }

                if let Some(mut cond) = fl.condition.take() {
                    self.remove_for_loops_in_expr(&mut cond);
                    let cond_span = cond.span;
                    let break_stmt = LStatement {
                        kind: LStatementKind::Expr(LExpr {
                            kind: LExprKind::Break,
                            ty: crate::analysis::types::Type::Never,
                            span: cond_span,
                        }),
                        span: cond_span,
                    };
                    let if_cond = LStatement {
                        kind: LStatementKind::Expr(LExpr {
                            kind: LExprKind::Conditional(Box::new(LCondExpr {
                                if_branch: LIfBranch {
                                    condition: LExpr {
                                        kind: LExprKind::Not(Box::new(cond)),
                                        ty: crate::analysis::types::Type::Bool,
                                        span: cond_span,
                                    },
                                    body: LBlock {
                                        statements: vec![break_stmt],
                                        tail_expr: None,
                                        ty: crate::analysis::types::Type::Never,
                                        span: cond_span,
                                    },
                                    span: cond_span,
                                },
                                elseif_branches: vec![],
                                else_branch: None,
                                span: cond_span,
                            })),
                            ty: crate::analysis::types::Type::Nil,
                            span: cond_span,
                        }),
                        span: cond_span,
                    };
                    transformed_body.insert(0, if_cond);
                }

                transformed_body.extend(std::mem::take(&mut fl.increment));

                let loop_stmt = LStatement {
                    kind: LStatementKind::Loop(transformed_body),
                    span,
                };

                if let Some(init) = fl.init.take() {
                    match init {
                        LForLoopInit::VarInit(mut vi) => {
                            self.remove_for_loops_in_expr(&mut vi.value);
                            out.push(LStatement {
                                kind: LStatementKind::VarInit(vi),
                                span,
                            });
                        }
                        LForLoopInit::Assignment(mut a) => {
                            self.remove_for_loops_in_expr(&mut a.value);
                            out.push(LStatement {
                                kind: LStatementKind::Assignment(a),
                                span,
                            });
                        }
                    }
                }

                out.push(loop_stmt);
            }
            LStatementKind::Error => {
                out.push(stmt);
            }
        }
    }

    fn remove_for_loops_in_expr(&mut self, expr: &mut LExpr) {
        match &mut expr.kind {
            LExprKind::Block(b) => {
                self.remove_for_loops_in_block(b);
            }
            LExprKind::Conditional(c) => {
                self.remove_for_loops_in_expr(&mut c.if_branch.condition);
                self.remove_for_loops_in_block(&mut c.if_branch.body);
                for elseif in &mut c.elseif_branches {
                    self.remove_for_loops_in_expr(&mut elseif.condition);
                    self.remove_for_loops_in_block(&mut elseif.body);
                }
                if let Some(eb) = &mut c.else_branch {
                    self.remove_for_loops_in_block(eb);
                }
            }
            LExprKind::BinOp(_, lhs, rhs) => {
                self.remove_for_loops_in_expr(lhs);
                self.remove_for_loops_in_expr(rhs);
            }
            LExprKind::Not(e)
            | LExprKind::Negate(e)
            | LExprKind::Head(e)
            | LExprKind::Tail(e)
            | LExprKind::Len(e)
            | LExprKind::VariantPayload(e)
            | LExprKind::UnwrapOptional(e)
            | LExprKind::MakeIter(e)
            | LExprKind::IterIsDone(e)
            | LExprKind::IterNext(e)
            | LExprKind::Recv(e)
            | LExprKind::WrapInOptional(e)
            | LExprKind::PersistData(e)
            | LExprKind::Return(e) => {
                self.remove_for_loops_in_expr(e);
            }
            LExprKind::Append(l, r)
            | LExprKind::Prepend(l, r)
            | LExprKind::Min(l, r)
            | LExprKind::Exists(l, r)
            | LExprKind::Erase(l, r)
            | LExprKind::Send(l, r)
            | LExprKind::Index(l, r)
            | LExprKind::SafeIndex(l, r) => {
                self.remove_for_loops_in_expr(l);
                self.remove_for_loops_in_expr(r);
            }
            LExprKind::Slice(a, b, c) | LExprKind::Store(a, b, c) => {
                self.remove_for_loops_in_expr(a);
                self.remove_for_loops_in_expr(b);
                self.remove_for_loops_in_expr(c);
            }
            LExprKind::FuncCall(fc) => match fc {
                LFuncCall::User(u) => {
                    for arg in &mut u.args {
                        self.remove_for_loops_in_expr(arg);
                    }
                }
                LFuncCall::Builtin(_, args, _) => {
                    for arg in args {
                        self.remove_for_loops_in_expr(arg);
                    }
                }
            },
            LExprKind::RpcCall(t, fc) => {
                self.remove_for_loops_in_expr(t);
                for arg in &mut fc.args {
                    self.remove_for_loops_in_expr(arg);
                }
            }
            LExprKind::MapLit(pairs) => {
                for (k, v) in pairs {
                    self.remove_for_loops_in_expr(k);
                    self.remove_for_loops_in_expr(v);
                }
            }
            LExprKind::ListLit(args) | LExprKind::TupleLit(args) => {
                for arg in args {
                    self.remove_for_loops_in_expr(arg);
                }
            }
            LExprKind::StructLit(_, fields) => {
                for (_, e) in fields {
                    self.remove_for_loops_in_expr(e);
                }
            }
            LExprKind::VariantLit(_, _, Some(e)) => {
                self.remove_for_loops_in_expr(e);
            }
            LExprKind::IsVariant(e, _)
            | LExprKind::FieldAccess(e, _)
            | LExprKind::SafeFieldAccess(e, _)
            | LExprKind::TupleAccess(e, _)
            | LExprKind::SafeTupleAccess(e, _) => {
                self.remove_for_loops_in_expr(e);
            }
            LExprKind::Var(_, _)
            | LExprKind::IntLit(_)
            | LExprKind::StringLit(_)
            | LExprKind::BoolLit(_)
            | LExprKind::NilLit
            | LExprKind::VariantLit(_, _, None)
            | LExprKind::MakeChannel
            | LExprKind::SetTimer(_)
            | LExprKind::RetrieveData(_)
            | LExprKind::DiscardData
            | LExprKind::Break
            | LExprKind::Continue
            | LExprKind::Error => {}
        }
    }
}

fn replace_continue_in_stmts(stmts: &mut Vec<LStatement>, increment: &[LStatement]) {
    for stmt in stmts {
        match &mut stmt.kind {
            LStatementKind::VarInit(vi) => replace_continue_in_expr(&mut vi.value, increment),
            LStatementKind::Assignment(a) => {
                replace_continue_in_expr(&mut a.value, increment);
            }
            LStatementKind::Expr(e) => replace_continue_in_expr(e, increment),
            LStatementKind::Loop(_) | LStatementKind::ForLoop(_) | LStatementKind::ForInLoop(_) => {
            }
            LStatementKind::Error => {}
        }
    }
}

fn replace_continue_in_expr(expr: &mut LExpr, increment: &[LStatement]) {
    match &mut expr.kind {
        LExprKind::Continue => {
            let span = expr.span;
            let mut stmts = increment.to_vec();
            stmts.push(LStatement {
                kind: LStatementKind::Expr(LExpr {
                    kind: LExprKind::Continue,
                    ty: crate::analysis::types::Type::Never,
                    span,
                }),
                span,
            });
            expr.kind = LExprKind::Block(Box::new(LBlock {
                statements: stmts,
                tail_expr: None,
                ty: crate::analysis::types::Type::Never,
                span,
            }));
        }
        LExprKind::Block(b) => replace_continue_in_block(b, increment),
        LExprKind::Conditional(c) => {
            replace_continue_in_block(&mut c.if_branch.body, increment);
            replace_continue_in_expr(&mut c.if_branch.condition, increment);
            for elseif in &mut c.elseif_branches {
                replace_continue_in_expr(&mut elseif.condition, increment);
                replace_continue_in_block(&mut elseif.body, increment);
            }
            if let Some(eb) = &mut c.else_branch {
                replace_continue_in_block(eb, increment);
            }
        }
        LExprKind::BinOp(_, lhs, rhs) => {
            replace_continue_in_expr(lhs, increment);
            replace_continue_in_expr(rhs, increment);
        }
        LExprKind::Not(e)
        | LExprKind::Negate(e)
        | LExprKind::Head(e)
        | LExprKind::Tail(e)
        | LExprKind::Len(e)
        | LExprKind::VariantPayload(e)
        | LExprKind::UnwrapOptional(e)
        | LExprKind::MakeIter(e)
        | LExprKind::IterIsDone(e)
        | LExprKind::IterNext(e)
        | LExprKind::Recv(e)
        | LExprKind::WrapInOptional(e)
        | LExprKind::PersistData(e)
        | LExprKind::Return(e) => {
            replace_continue_in_expr(e, increment);
        }
        LExprKind::Append(l, r)
        | LExprKind::Prepend(l, r)
        | LExprKind::Min(l, r)
        | LExprKind::Exists(l, r)
        | LExprKind::Erase(l, r)
        | LExprKind::Send(l, r)
        | LExprKind::Index(l, r)
        | LExprKind::SafeIndex(l, r) => {
            replace_continue_in_expr(l, increment);
            replace_continue_in_expr(r, increment);
        }
        LExprKind::Slice(a, b, c) | LExprKind::Store(a, b, c) => {
            replace_continue_in_expr(a, increment);
            replace_continue_in_expr(b, increment);
            replace_continue_in_expr(c, increment);
        }
        LExprKind::FuncCall(fc) => match fc {
            LFuncCall::User(u) => {
                for arg in &mut u.args {
                    replace_continue_in_expr(arg, increment);
                }
            }
            LFuncCall::Builtin(_, args, _) => {
                for arg in args {
                    replace_continue_in_expr(arg, increment);
                }
            }
        },
        LExprKind::RpcCall(t, fc) => {
            replace_continue_in_expr(t, increment);
            for arg in &mut fc.args {
                replace_continue_in_expr(arg, increment);
            }
        }
        LExprKind::MapLit(pairs) => {
            for (k, v) in pairs {
                replace_continue_in_expr(k, increment);
                replace_continue_in_expr(v, increment);
            }
        }
        LExprKind::ListLit(args) | LExprKind::TupleLit(args) => {
            for arg in args {
                replace_continue_in_expr(arg, increment);
            }
        }
        LExprKind::StructLit(_, fields) => {
            for (_, e) in fields {
                replace_continue_in_expr(e, increment);
            }
        }
        LExprKind::VariantLit(_, _, Some(e)) => {
            replace_continue_in_expr(e, increment);
        }
        LExprKind::IsVariant(e, _)
        | LExprKind::FieldAccess(e, _)
        | LExprKind::SafeFieldAccess(e, _)
        | LExprKind::TupleAccess(e, _)
        | LExprKind::SafeTupleAccess(e, _) => {
            replace_continue_in_expr(e, increment);
        }
        LExprKind::Var(_, _)
        | LExprKind::IntLit(_)
        | LExprKind::StringLit(_)
        | LExprKind::BoolLit(_)
        | LExprKind::NilLit
        | LExprKind::VariantLit(_, _, None)
        | LExprKind::MakeChannel
        | LExprKind::SetTimer(_)
        | LExprKind::RetrieveData(_)
        | LExprKind::DiscardData
        | LExprKind::Break
        | LExprKind::Error => {}
    }
}

fn replace_continue_in_block(block: &mut LBlock, increment: &[LStatement]) {
    replace_continue_in_stmts(&mut block.statements, increment);
    if let Some(t) = &mut block.tail_expr {
        replace_continue_in_expr(t, increment);
    }
}
