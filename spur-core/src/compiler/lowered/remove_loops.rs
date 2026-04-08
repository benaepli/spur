use super::ast::*;

pub fn remove_for_loops(program: &mut LProgram) {
    for def in &mut program.top_level_defs {
        match def {
            LTopLevelDef::Role(role) => {
                for func in &mut role.func_defs {
                    remove_for_loops_in_func(func);
                }
            }
            LTopLevelDef::FreeFunc(func) => {
                remove_for_loops_in_func(func);
            }
        }
    }
}

fn remove_for_loops_in_func(func: &mut LFuncDef) {
    remove_for_loops_in_block(&mut func.body);
}

fn remove_for_loops_in_block(block: &mut LBlock) {
    let mut new_stmts = Vec::new();
    for stmt in std::mem::take(&mut block.statements) {
        remove_for_loops_in_stmt(stmt, &mut new_stmts);
    }
    block.statements = new_stmts;

    // Tail expr doesn't contain statements by definition,
    // though it might contain blocks. We need to traverse exprs too!
    if let Some(tail) = &mut block.tail_expr {
        remove_for_loops_in_expr(tail);
    }
}

fn remove_for_loops_in_stmt(mut stmt: LStatement, out: &mut Vec<LStatement>) {
    let span = stmt.span;
    match &mut stmt.kind {
        LStatementKind::VarInit(vi) => {
            remove_for_loops_in_expr(&mut vi.value);
            out.push(stmt);
        }
        LStatementKind::Assignment(a) => {
            remove_for_loops_in_expr(&mut a.target);
            remove_for_loops_in_expr(&mut a.value);
            out.push(stmt);
        }
        LStatementKind::Expr(e) => {
            remove_for_loops_in_expr(e);
            out.push(stmt);
        }
        LStatementKind::Loop(stmts) => {
            let mut new_inner = Vec::new();
            for s in std::mem::take(stmts) {
                remove_for_loops_in_stmt(s, &mut new_inner);
            }
            *stmts = new_inner;
            out.push(stmt);
        }
        LStatementKind::ForInLoop(fil) => {
            remove_for_loops_in_expr(&mut fil.iterable);
            let mut new_inner = Vec::new();
            for s in std::mem::take(&mut fil.body) {
                remove_for_loops_in_stmt(s, &mut new_inner);
            }
            fil.body = new_inner;
            out.push(stmt);
        }
        LStatementKind::ForLoop(fl) => {
            // First, transform the body statements
            let mut transformed_body = Vec::new();
            for s in std::mem::take(&mut fl.body) {
                remove_for_loops_in_stmt(s, &mut transformed_body);
            }

            // We replace continue in the body with `{ inc; continue }`.
            let increment_clone = fl.increment.clone();
            if !increment_clone.is_empty() {
                replace_continue_in_stmts(&mut transformed_body, &increment_clone);
            }

            // condition goes to the top
            if let Some(mut cond) = fl.condition.take() {
                remove_for_loops_in_expr(&mut cond);
                let cond_span = cond.span;
                // if !cond { break; }
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

            // Finally append increment blocks
            transformed_body.extend(std::mem::take(&mut fl.increment));

            let loop_stmt = LStatement {
                kind: LStatementKind::Loop(transformed_body),
                span,
            };

            if let Some(init) = fl.init.take() {
                match init {
                    LForLoopInit::VarInit(mut vi) => {
                        remove_for_loops_in_expr(&mut vi.value);
                        out.push(LStatement {
                            kind: LStatementKind::VarInit(vi),
                            span,
                        });
                    }
                    LForLoopInit::Assignment(mut a) => {
                        remove_for_loops_in_expr(&mut a.target);
                        remove_for_loops_in_expr(&mut a.value);
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

fn replace_continue_in_stmts(stmts: &mut Vec<LStatement>, increment: &[LStatement]) {
    for stmt in stmts {
        match &mut stmt.kind {
            LStatementKind::VarInit(vi) => replace_continue_in_expr(&mut vi.value, increment),
            LStatementKind::Assignment(a) => {
                replace_continue_in_expr(&mut a.target, increment);
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
            // Replace `continue` with `{ inc; continue }`
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
        // Leaf nodes do nothing
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

fn remove_for_loops_in_expr(expr: &mut LExpr) {
    match &mut expr.kind {
        LExprKind::Block(b) => {
            remove_for_loops_in_block(b);
        }
        LExprKind::Conditional(c) => {
            remove_for_loops_in_expr(&mut c.if_branch.condition);
            remove_for_loops_in_block(&mut c.if_branch.body);
            for elseif in &mut c.elseif_branches {
                remove_for_loops_in_expr(&mut elseif.condition);
                remove_for_loops_in_block(&mut elseif.body);
            }
            if let Some(eb) = &mut c.else_branch {
                remove_for_loops_in_block(eb);
            }
        }
        LExprKind::BinOp(_, lhs, rhs) => {
            remove_for_loops_in_expr(lhs);
            remove_for_loops_in_expr(rhs);
        }
        LExprKind::Not(e)
        | LExprKind::Negate(e)
        | LExprKind::Head(e)
        | LExprKind::Tail(e)
        | LExprKind::Len(e)
        | LExprKind::VariantPayload(e)
        | LExprKind::UnwrapOptional(e)
        | LExprKind::Recv(e)
        | LExprKind::WrapInOptional(e)
        | LExprKind::PersistData(e)
        | LExprKind::Return(e) => {
            remove_for_loops_in_expr(e);
        }
        LExprKind::Append(l, r)
        | LExprKind::Prepend(l, r)
        | LExprKind::Min(l, r)
        | LExprKind::Exists(l, r)
        | LExprKind::Erase(l, r)
        | LExprKind::Send(l, r)
        | LExprKind::Index(l, r)
        | LExprKind::SafeIndex(l, r) => {
            remove_for_loops_in_expr(l);
            remove_for_loops_in_expr(r);
        }
        LExprKind::Slice(a, b, c) | LExprKind::Store(a, b, c) => {
            remove_for_loops_in_expr(a);
            remove_for_loops_in_expr(b);
            remove_for_loops_in_expr(c);
        }
        LExprKind::FuncCall(fc) => match fc {
            LFuncCall::User(u) => {
                for arg in &mut u.args {
                    remove_for_loops_in_expr(arg);
                }
            }
            LFuncCall::Builtin(_, args, _) => {
                for arg in args {
                    remove_for_loops_in_expr(arg);
                }
            }
        },
        LExprKind::RpcCall(t, fc) => {
            remove_for_loops_in_expr(t);
            for arg in &mut fc.args {
                remove_for_loops_in_expr(arg);
            }
        }
        LExprKind::MapLit(pairs) => {
            for (k, v) in pairs {
                remove_for_loops_in_expr(k);
                remove_for_loops_in_expr(v);
            }
        }
        LExprKind::ListLit(args) | LExprKind::TupleLit(args) => {
            for arg in args {
                remove_for_loops_in_expr(arg);
            }
        }
        LExprKind::StructLit(_, fields) => {
            for (_, e) in fields {
                remove_for_loops_in_expr(e);
            }
        }
        LExprKind::VariantLit(_, _, Some(e)) => {
            remove_for_loops_in_expr(e);
        }
        LExprKind::IsVariant(e, _)
        | LExprKind::FieldAccess(e, _)
        | LExprKind::SafeFieldAccess(e, _)
        | LExprKind::TupleAccess(e, _)
        | LExprKind::SafeTupleAccess(e, _) => {
            remove_for_loops_in_expr(e);
        }
        // Leaf nodes do nothing
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
