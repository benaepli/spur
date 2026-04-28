use std::collections::{HashMap, HashSet};

use crate::analysis::resolver::NameId;
use crate::analysis::types::Type;
use crate::liquid::pure::ast::*;
use crate::liquid::threaded::ast::*;
use crate::parser::Span;

/// Lowers the state-threaded IR into the pure / SSA IR used by the refinement
/// type checker. Eliminates all local mutability: no Assign, no Loop,
/// no Break, no Continue.
pub struct PureLowerer {
    next_name_id: usize,
    /// SSA substitution: threaded NameId -> current PAtomic.
    env: HashMap<NameId, PAtomic>,
    /// Threaded NameId -> declared type. Populated from params and LetAtoms.
    type_env: HashMap<NameId, Type>,
    /// Threaded NameId -> original (user-facing) name.
    orig_name: HashMap<NameId, String>,
    /// Cloned from TProgram; read-only during lowering.
    struct_defs: HashMap<NameId, Vec<(String, Type)>>,
    /// Mutable: gets extra entries for each lifted loop's LoopResult enum.
    enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
    /// Mutable: gets extra entries for synthesized enum / function names.
    id_to_name: HashMap<NameId, String>,
    /// Loop functions lifted out of inline Loop statements.
    lifted_funcs: Vec<PFuncDef>,
    /// Stack of enclosing loop contexts — used to wrap `return` and to handle
    /// `break` / `continue` correctly under nesting.
    loop_stack: Vec<LoopCtx>,
    /// Return type of the user function currently being lowered. Used to size
    /// the Return variant of the synthesized LoopResult enum.
    current_func_return_type: Option<Type>,
}

/// A loop currently being lowered. Pushed on entry to the lifted function's
/// body; popped on exit. The outermost loop has `loop_stack.len() == 1`
/// during its body lowering.
#[allow(dead_code)]
#[derive(Clone)]
struct LoopCtx {
    loop_result_enum_id: NameId,
    loop_result_enum_name: String,
    /// Lifted function — used for self-recursion (fallthrough and continue).
    func_name_id: NameId,
    func_original_name: String,
    /// Ordered list of (threaded NameId, original name, type) for parameters
    /// of the lifted function. These are also the live-out variables (we use
    /// `live_in = live_out` conservatively).
    live_vars: Vec<(NameId, String, Type)>,
    /// Type of the Exit variant payload = Tuple(live_vars' types).
    exit_payload_ty: Type,
    span: Span,
}

impl PureLowerer {
    fn new(
        next_name_id: usize,
        struct_defs: HashMap<NameId, Vec<(String, Type)>>,
        enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
        id_to_name: HashMap<NameId, String>,
    ) -> Self {
        Self {
            next_name_id,
            env: HashMap::new(),
            type_env: HashMap::new(),
            orig_name: HashMap::new(),
            struct_defs,
            enum_defs,
            id_to_name,
            lifted_funcs: Vec::new(),
            loop_stack: Vec::new(),
            current_func_return_type: None,
        }
    }

    fn loop_result_ty(&self, ctx: &LoopCtx) -> Type {
        Type::Enum(ctx.loop_result_enum_id, ctx.loop_result_enum_name.clone())
    }

    fn fresh_name(&mut self, prefix: &str) -> (NameId, String) {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        (id, format!("__{}_{}", prefix, id.0))
    }

    fn lookup(&self, nid: NameId, fallback_name: &str) -> PAtomic {
        self.env
            .get(&nid)
            .cloned()
            .unwrap_or_else(|| PAtomic::Var(nid, fallback_name.to_string()))
    }

    fn lower_atomic(&self, a: TAtomic) -> PAtomic {
        match a {
            TAtomic::Var(id, name) => self.lookup(id, &name),
            TAtomic::IntLit(v) => PAtomic::IntLit(v),
            TAtomic::StringLit(v) => PAtomic::StringLit(v),
            TAtomic::BoolLit(v) => PAtomic::BoolLit(v),
            TAtomic::NilLit => PAtomic::NilLit,
            TAtomic::Never => PAtomic::Never,
        }
    }

    fn lower_user_call(&self, call: TUserFuncCall) -> PUserFuncCall {
        PUserFuncCall {
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

    fn lower_func_call(&self, call: TFuncCall) -> PFuncCall {
        match call {
            TFuncCall::User(u) => PFuncCall::User(self.lower_user_call(u)),
            TFuncCall::Builtin(b, args, ty) => PFuncCall::Builtin(
                b,
                args.into_iter().map(|a| self.lower_atomic(a)).collect(),
                ty,
            ),
        }
    }

    fn lower_expr(&mut self, expr: TExpr) -> PExpr {
        let span = expr.span;
        let ty = expr.ty;
        let kind = match expr.kind {
            TExprKind::Atomic(a) => PExprKind::Atomic(self.lower_atomic(a)),
            TExprKind::BinOp(op, a, b) => {
                PExprKind::BinOp(op, self.lower_atomic(a), self.lower_atomic(b))
            }
            TExprKind::Not(a) => PExprKind::Not(self.lower_atomic(a)),
            TExprKind::Negate(a) => PExprKind::Negate(self.lower_atomic(a)),
            TExprKind::FuncCall(call) => PExprKind::FuncCall(self.lower_func_call(call)),
            TExprKind::MapLit(pairs) => PExprKind::MapLit(
                pairs
                    .into_iter()
                    .map(|(k, v)| (self.lower_atomic(k), self.lower_atomic(v)))
                    .collect(),
            ),
            TExprKind::ListLit(items) => {
                PExprKind::ListLit(items.into_iter().map(|a| self.lower_atomic(a)).collect())
            }
            TExprKind::TupleLit(items) => {
                PExprKind::TupleLit(items.into_iter().map(|a| self.lower_atomic(a)).collect())
            }
            TExprKind::Append(a, b) => {
                PExprKind::Append(self.lower_atomic(a), self.lower_atomic(b))
            }
            TExprKind::Prepend(a, b) => {
                PExprKind::Prepend(self.lower_atomic(a), self.lower_atomic(b))
            }
            TExprKind::Min(a, b) => PExprKind::Min(self.lower_atomic(a), self.lower_atomic(b)),
            TExprKind::Exists(a, b) => {
                PExprKind::Exists(self.lower_atomic(a), self.lower_atomic(b))
            }
            TExprKind::Erase(a, b) => PExprKind::Erase(self.lower_atomic(a), self.lower_atomic(b)),
            TExprKind::Store(a, b, c) => PExprKind::Store(
                self.lower_atomic(a),
                self.lower_atomic(b),
                self.lower_atomic(c),
            ),
            TExprKind::Head(a) => PExprKind::Head(self.lower_atomic(a)),
            TExprKind::Tail(a) => PExprKind::Tail(self.lower_atomic(a)),
            TExprKind::Len(a) => PExprKind::Len(self.lower_atomic(a)),
            TExprKind::RpcCall(target, call) => {
                PExprKind::RpcCall(self.lower_atomic(target), self.lower_user_call(call))
            }
            TExprKind::Conditional(cond) => {
                // Non-root conditional — no mutation join expected at this level
                // (ANF flattening should hoist mutating conditionals to statement
                // position).
                PExprKind::Conditional(Box::new(self.lower_cond_expr_plain(*cond)))
            }
            TExprKind::Block(b) => PExprKind::Block(Box::new(self.lower_block(*b))),
            TExprKind::VariantLit(enum_id, variant, payload) => {
                PExprKind::VariantLit(enum_id, variant, payload.map(|a| self.lower_atomic(a)))
            }
            TExprKind::IsVariant(a, name) => PExprKind::IsVariant(self.lower_atomic(a), name),
            TExprKind::VariantPayload(a) => PExprKind::VariantPayload(self.lower_atomic(a)),
            TExprKind::UnwrapOptional(a) => PExprKind::UnwrapOptional(self.lower_atomic(a)),
            TExprKind::MakeIter(a) => PExprKind::MakeIter(self.lower_atomic(a)),
            TExprKind::IterIsDone(a) => PExprKind::IterIsDone(self.lower_atomic(a)),
            TExprKind::IterNext(a) => PExprKind::IterNext(self.lower_atomic(a)),
            TExprKind::MakeChannel => PExprKind::MakeChannel,
            TExprKind::Send(s, c, v) => PExprKind::Send(
                self.lower_atomic(s),
                self.lower_atomic(c),
                self.lower_atomic(v),
            ),
            TExprKind::Recv(s, c) => PExprKind::Recv(self.lower_atomic(s), self.lower_atomic(c)),
            TExprKind::SetTimer(label) => PExprKind::SetTimer(label),
            TExprKind::Fifo(a) => PExprKind::Fifo(self.lower_atomic(a)),
            TExprKind::Index(a, b) => PExprKind::Index(self.lower_atomic(a), self.lower_atomic(b)),
            TExprKind::Slice(a, b, c) => PExprKind::Slice(
                self.lower_atomic(a),
                self.lower_atomic(b),
                self.lower_atomic(c),
            ),
            TExprKind::TupleAccess(a, i) => PExprKind::TupleAccess(self.lower_atomic(a), i),
            TExprKind::FieldAccess(a, name) => PExprKind::FieldAccess(self.lower_atomic(a), name),
            TExprKind::SafeFieldAccess(a, name) => {
                PExprKind::SafeFieldAccess(self.lower_atomic(a), name)
            }
            TExprKind::SafeIndex(a, b) => {
                PExprKind::SafeIndex(self.lower_atomic(a), self.lower_atomic(b))
            }
            TExprKind::SafeTupleAccess(a, i) => PExprKind::SafeTupleAccess(self.lower_atomic(a), i),
            TExprKind::StructLit(struct_id, fields) => PExprKind::StructLit(
                struct_id,
                fields
                    .into_iter()
                    .map(|(n, a)| (n, self.lower_atomic(a)))
                    .collect(),
            ),
            TExprKind::WrapInOptional(a) => PExprKind::WrapInOptional(self.lower_atomic(a)),
            TExprKind::PersistData(a) => PExprKind::PersistData(self.lower_atomic(a)),
            TExprKind::RetrieveData(t) => PExprKind::RetrieveData(t),
            TExprKind::DiscardData => PExprKind::DiscardData,
        };
        PExpr { kind, ty, span }
    }

    /// Plain (non-mutation-aware) conditional lowering — used when a
    /// conditional appears as a sub-expression where no join is required.
    /// Mutation-aware lowering happens in `lower_cond_stmt`.
    fn lower_cond_expr_plain(&mut self, cond: TCondExpr) -> PCondExpr {
        PCondExpr {
            if_branch: self.lower_if_branch_plain(cond.if_branch),
            elseif_branches: cond
                .elseif_branches
                .into_iter()
                .map(|b| self.lower_if_branch_plain(b))
                .collect(),
            else_branch: cond.else_branch.map(|b| self.lower_block(b)),
            span: cond.span,
        }
    }

    fn lower_if_branch_plain(&mut self, branch: TIfBranch) -> PIfBranch {
        PIfBranch {
            condition: self.lower_atomic(branch.condition),
            body: self.lower_block(branch.body),
            span: branch.span,
        }
    }

    fn lower_block(&mut self, block: TBlock) -> PBlock {
        let mut out = Vec::with_capacity(block.statements.len());
        for stmt in block.statements {
            self.lower_statement(stmt, &mut out);
        }
        PBlock {
            statements: out,
            tail_expr: block.tail_expr.map(|a| self.lower_atomic(a)),
            ty: block.ty,
            span: block.span,
        }
    }

    fn lower_statement(&mut self, stmt: TStatement, out: &mut Vec<PStatement>) {
        let span = stmt.span;
        match stmt.kind {
            TStatementKind::LetAtom(la) => {
                // Conditional RHS may need mutation-aware lowering with join.
                if let TExprKind::Conditional(cond) = la.value.kind {
                    self.lower_cond_stmt(
                        Some((la.name, la.original_name, la.ty.clone())),
                        la.ty,
                        *cond,
                        la.span,
                        out,
                    );
                } else {
                    let value = self.lower_expr(la.value);
                    out.push(PStatement {
                        kind: PStatementKind::LetAtom(PLetAtom {
                            name: la.name,
                            original_name: la.original_name.clone(),
                            ty: la.ty.clone(),
                            value,
                            span: la.span,
                        }),
                        span,
                    });
                    self.type_env.insert(la.name, la.ty);
                    self.orig_name.insert(la.name, la.original_name.clone());
                    self.env
                        .insert(la.name, PAtomic::Var(la.name, la.original_name));
                }
            }
            TStatementKind::Assign(a) => self.lower_assign(a, out),
            TStatementKind::Expr(e) => {
                if let TExprKind::Conditional(cond) = e.kind {
                    self.lower_cond_stmt(None, e.ty, *cond, e.span, out);
                } else {
                    let pe = self.lower_expr(e);
                    out.push(PStatement {
                        kind: PStatementKind::Expr(pe),
                        span,
                    });
                }
            }
            TStatementKind::Return(a) => {
                let pa = self.lower_atomic(a);
                self.emit_function_return(pa, span, out);
            }
            TStatementKind::Error => out.push(PStatement {
                kind: PStatementKind::Error,
                span,
            }),
            TStatementKind::Loop(body) => self.lower_loop(body, span, out),
            TStatementKind::Break => self.emit_loop_break(span, out),
            TStatementKind::Continue => self.emit_loop_continue(span, out),
        }
    }

    fn lower_assign(&mut self, a: TAssign, _out: &mut Vec<PStatement>) {
        let new_val = self.lower_atomic(a.value);
        self.env.insert(a.target_id, new_val);
    }

    /// Lowers a conditional at statement position (either as a LetAtom RHS or
    /// a value-less Expr stmt), producing the join over mutated live-out
    /// variables as a tuple that each branch constructs and the outer context
    /// projects.
    fn lower_cond_stmt(
        &mut self,
        binding: Option<(NameId, String, Type)>,
        cond_value_ty: Type,
        cond: TCondExpr,
        span: Span,
        out: &mut Vec<PStatement>,
    ) {
        let snapshot = self.env.clone();

        // Conditions are atomics evaluated in the pre-branch env.
        let if_cond_atom = self.lower_atomic(cond.if_branch.condition);
        let elseif_cond_atoms: Vec<(PAtomic, Span, TBlock)> = cond
            .elseif_branches
            .into_iter()
            .map(|b| (self.lower_atomic(b.condition), b.span, b.body))
            .collect();

        // Lower each branch body with a cloned snapshot env; collect the
        // child env after each branch to compute diffs.
        let if_span = cond.if_branch.span;
        self.env = snapshot.clone();
        let mut if_body = self.lower_block(cond.if_branch.body);
        let if_child_env = self.env.clone();

        let mut elseif_lowered: Vec<(PAtomic, Span, PBlock, HashMap<NameId, PAtomic>)> =
            Vec::with_capacity(elseif_cond_atoms.len());
        for (cond_atom, bspan, body) in elseif_cond_atoms {
            self.env = snapshot.clone();
            let lowered = self.lower_block(body);
            let child = self.env.clone();
            elseif_lowered.push((cond_atom, bspan, lowered, child));
        }

        let else_lowered: Option<(PBlock, HashMap<NameId, PAtomic>)> = match cond.else_branch {
            Some(b) => {
                self.env = snapshot.clone();
                let lowered = self.lower_block(b);
                let child = self.env.clone();
                Some((lowered, child))
            }
            None => None,
        };

        // Reset env so outer code resumes from snapshot; we'll install fresh
        // versions of joined vars via projection below.
        self.env = snapshot.clone();

        // Compute joined vars = variables present in snapshot whose binding
        // was changed in any branch.
        let mut joined_ids: Vec<NameId> = Vec::new();
        let mut seen: HashSet<NameId> = HashSet::new();
        let consider_branch = |child: &HashMap<NameId, PAtomic>,
                                   seen: &mut HashSet<NameId>,
                                   joined_ids: &mut Vec<NameId>| {
            for (nid, atom) in child {
                if snapshot.get(nid) != Some(atom)
                    && snapshot.contains_key(nid)
                    && seen.insert(*nid)
                {
                    joined_ids.push(*nid);
                }
            }
        };
        consider_branch(&if_child_env, &mut seen, &mut joined_ids);
        for (_, _, _, child) in &elseif_lowered {
            consider_branch(child, &mut seen, &mut joined_ids);
        }
        if let Some((_, child)) = &else_lowered {
            consider_branch(child, &mut seen, &mut joined_ids);
        }
        joined_ids.sort_by_key(|n| n.0);

        let joined_vars: Vec<(NameId, String, Type)> = joined_ids
            .iter()
            .map(|nid| {
                let name = self
                    .orig_name
                    .get(nid)
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", nid.0));
                let ty = self
                    .type_env
                    .get(nid)
                    .cloned()
                    .unwrap_or_else(|| panic!("no type for joined var {:?}", nid));
                (*nid, name, ty)
            })
            .collect();

        let has_value = binding.is_some();

        // Fast path: no joins — emit as a plain conditional.
        if joined_vars.is_empty() {
            if has_value {
                let (bind_id, bind_name, bind_ty) = binding.unwrap();
                let cond_pexpr = PExpr {
                    kind: PExprKind::Conditional(Box::new(PCondExpr {
                        if_branch: PIfBranch {
                            condition: if_cond_atom,
                            body: if_body,
                            span: if_span,
                        },
                        elseif_branches: elseif_lowered
                            .into_iter()
                            .map(|(c, s, b, _)| PIfBranch {
                                condition: c,
                                body: b,
                                span: s,
                            })
                            .collect(),
                        else_branch: else_lowered.map(|(b, _)| b),
                        span,
                    })),
                    ty: cond_value_ty.clone(),
                    span,
                };
                out.push(PStatement {
                    kind: PStatementKind::LetAtom(PLetAtom {
                        name: bind_id,
                        original_name: bind_name.clone(),
                        ty: bind_ty.clone(),
                        value: cond_pexpr,
                        span,
                    }),
                    span,
                });
                self.type_env.insert(bind_id, bind_ty);
                self.orig_name.insert(bind_id, bind_name.clone());
                self.env.insert(bind_id, PAtomic::Var(bind_id, bind_name));
            } else {
                let cond_pexpr = PExpr {
                    kind: PExprKind::Conditional(Box::new(PCondExpr {
                        if_branch: PIfBranch {
                            condition: if_cond_atom,
                            body: if_body,
                            span: if_span,
                        },
                        elseif_branches: elseif_lowered
                            .into_iter()
                            .map(|(c, s, b, _)| PIfBranch {
                                condition: c,
                                body: b,
                                span: s,
                            })
                            .collect(),
                        else_branch: else_lowered.map(|(b, _)| b),
                        span,
                    })),
                    ty: cond_value_ty,
                    span,
                };
                out.push(PStatement {
                    kind: PStatementKind::Expr(cond_pexpr),
                    span,
                });
            }
            return;
        }

        // Joined path: augment each branch's tail_expr with a tuple of
        // [original_value?, joined_var_values...].
        let tup_types: Vec<Type> = {
            let mut ts = Vec::new();
            if has_value {
                ts.push(cond_value_ty.clone());
            }
            for (_, _, t) in &joined_vars {
                ts.push(t.clone());
            }
            ts
        };
        let tup_ty = Type::Tuple(tup_types.clone());

        let augment = |this: &mut PureLowerer,
                       body: &mut PBlock,
                       child_env: &HashMap<NameId, PAtomic>,
                       snapshot: &HashMap<NameId, PAtomic>,
                       joined_vars: &[(NameId, String, Type)],
                       has_value: bool,
                       tup_ty: &Type,
                       span: Span| {
            let original_tail = body.tail_expr.take();
            let mut tuple_elems = Vec::new();
            if has_value {
                tuple_elems.push(original_tail.unwrap_or(PAtomic::Never));
            }
            for (xid, xname, _) in joined_vars {
                let val = child_env
                    .get(xid)
                    .cloned()
                    .or_else(|| snapshot.get(xid).cloned())
                    .unwrap_or_else(|| PAtomic::Var(*xid, xname.clone()));
                tuple_elems.push(val);
            }
            let (tup_id, tup_name) = this.fresh_name("cond_tup");
            body.statements.push(PStatement {
                kind: PStatementKind::LetAtom(PLetAtom {
                    name: tup_id,
                    original_name: tup_name.clone(),
                    ty: tup_ty.clone(),
                    value: PExpr {
                        kind: PExprKind::TupleLit(tuple_elems),
                        ty: tup_ty.clone(),
                        span,
                    },
                    span,
                }),
                span,
            });
            body.tail_expr = Some(PAtomic::Var(tup_id, tup_name));
            body.ty = tup_ty.clone();
        };

        augment(
            self,
            &mut if_body,
            &if_child_env,
            &snapshot,
            &joined_vars,
            has_value,
            &tup_ty,
            span,
        );
        let mut elseif_out: Vec<PIfBranch> = Vec::with_capacity(elseif_lowered.len());
        for (cond_atom, bspan, mut body, child) in elseif_lowered {
            augment(
                self,
                &mut body,
                &child,
                &snapshot,
                &joined_vars,
                has_value,
                &tup_ty,
                span,
            );
            elseif_out.push(PIfBranch {
                condition: cond_atom,
                body,
                span: bspan,
            });
        }
        let else_out: Option<PBlock> = match else_lowered {
            Some((mut body, child)) => {
                augment(
                    self,
                    &mut body,
                    &child,
                    &snapshot,
                    &joined_vars,
                    has_value,
                    &tup_ty,
                    span,
                );
                Some(body)
            }
            None => {
                // Synthesize a no-op else branch: carry snapshot values for
                // joined vars and Never for the original result.
                let mut tuple_elems = Vec::new();
                if has_value {
                    tuple_elems.push(PAtomic::Never);
                }
                for (xid, xname, _) in &joined_vars {
                    tuple_elems.push(
                        snapshot
                            .get(xid)
                            .cloned()
                            .unwrap_or_else(|| PAtomic::Var(*xid, xname.clone())),
                    );
                }
                let (tup_id, tup_name) = self.fresh_name("cond_tup");
                let stmts = vec![PStatement {
                    kind: PStatementKind::LetAtom(PLetAtom {
                        name: tup_id,
                        original_name: tup_name.clone(),
                        ty: tup_ty.clone(),
                        value: PExpr {
                            kind: PExprKind::TupleLit(tuple_elems),
                            ty: tup_ty.clone(),
                            span,
                        },
                        span,
                    }),
                    span,
                }];
                Some(PBlock {
                    statements: stmts,
                    tail_expr: Some(PAtomic::Var(tup_id, tup_name)),
                    ty: tup_ty.clone(),
                    span,
                })
            }
        };

        // Emit: let __cond_result = if ... { ... } else { ... };
        let (result_id, result_name) = self.fresh_name("cond_result");
        let cond_pexpr = PExpr {
            kind: PExprKind::Conditional(Box::new(PCondExpr {
                if_branch: PIfBranch {
                    condition: if_cond_atom,
                    body: if_body,
                    span: if_span,
                },
                elseif_branches: elseif_out,
                else_branch: else_out,
                span,
            })),
            ty: tup_ty.clone(),
            span,
        };
        out.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: result_id,
                original_name: result_name.clone(),
                ty: tup_ty.clone(),
                value: cond_pexpr,
                span,
            }),
            span,
        });
        let result_atom = PAtomic::Var(result_id, result_name);

        // Project the tuple: [value?, joined_vars...].
        let mut idx = 0usize;
        if let Some((bind_id, bind_name, bind_ty)) = binding {
            out.push(PStatement {
                kind: PStatementKind::LetAtom(PLetAtom {
                    name: bind_id,
                    original_name: bind_name.clone(),
                    ty: bind_ty.clone(),
                    value: PExpr {
                        kind: PExprKind::TupleAccess(result_atom.clone(), idx),
                        ty: bind_ty.clone(),
                        span,
                    },
                    span,
                }),
                span,
            });
            self.type_env.insert(bind_id, bind_ty);
            self.orig_name.insert(bind_id, bind_name.clone());
            self.env.insert(bind_id, PAtomic::Var(bind_id, bind_name));
            idx += 1;
        }
        for (xid, xname, xty) in joined_vars {
            let (new_xid, new_xname) = self.fresh_name(&xname);
            out.push(PStatement {
                kind: PStatementKind::LetAtom(PLetAtom {
                    name: new_xid,
                    original_name: new_xname.clone(),
                    ty: xty.clone(),
                    value: PExpr {
                        kind: PExprKind::TupleAccess(result_atom.clone(), idx),
                        ty: xty.clone(),
                        span,
                    },
                    span,
                }),
                span,
            });
            self.type_env.insert(new_xid, xty);
            self.orig_name.insert(new_xid, xname);
            self.env.insert(xid, PAtomic::Var(new_xid, new_xname));
            idx += 1;
        }
    }

    fn lower_param(&mut self, p: TFuncParam) -> PFuncParam {
        self.env
            .insert(p.name, PAtomic::Var(p.name, p.original_name.clone()));
        self.type_env.insert(p.name, p.ty.clone());
        self.orig_name.insert(p.name, p.original_name.clone());
        PFuncParam {
            name: p.name,
            original_name: p.original_name,
            ty: p.ty,
            span: p.span,
        }
    }

    fn lower_func_def(&mut self, f: TFuncDef) -> PFuncDef {
        self.env.clear();
        self.type_env.clear();
        self.orig_name.clear();
        // loop_stack should already be empty between functions — user-level
        // functions can't be nested inside loops — but clear defensively.
        self.loop_stack.clear();
        self.current_func_return_type = Some(f.return_type.clone());
        let params = f.params.into_iter().map(|p| self.lower_param(p)).collect();
        let body = self.lower_block(f.body);
        let result = PFuncDef {
            name: f.name,
            original_name: f.original_name,
            kind: if f.is_sync {
                PFuncKind::Sync
            } else {
                PFuncKind::Async
            },
            is_traced: f.is_traced,
            params,
            return_type: f.return_type,
            body,
            span: f.span,
        };
        self.current_func_return_type = None;
        result
    }

    fn lower_role(&mut self, r: TRoleDef) -> PRoleDef {
        PRoleDef {
            name: r.name,
            original_name: r.original_name,
            func_defs: r
                .func_defs
                .into_iter()
                .map(|f| self.lower_func_def(f))
                .collect(),
            span: r.span,
        }
    }

    fn lower_top_level(&mut self, d: TTopLevelDef) -> PTopLevelDef {
        match d {
            TTopLevelDef::Role(r) => PTopLevelDef::Role(self.lower_role(r)),
            TTopLevelDef::FreeFunc(f) => PTopLevelDef::FreeFunc(self.lower_func_def(f)),
        }
    }

    // ===== Loop lifting =====

    /// Emit a return from the user function currently being lowered.
    /// If we're inside a lifted loop function, wrap the value in
    /// `LoopResult::Return(...)` for the nearest enclosing loop so the outer
    /// caller can unwrap and propagate.
    fn emit_function_return(&mut self, v: PAtomic, span: Span, out: &mut Vec<PStatement>) {
        if let Some(ctx) = self.loop_stack.last().cloned() {
            let ret_ty = self.loop_result_ty(&ctx);
            let (vid, vname) = self.fresh_name("wrap_ret");
            out.push(PStatement {
                kind: PStatementKind::LetAtom(PLetAtom {
                    name: vid,
                    original_name: vname.clone(),
                    ty: ret_ty.clone(),
                    value: PExpr {
                        kind: PExprKind::VariantLit(
                            ctx.loop_result_enum_id,
                            "Return".to_string(),
                            Some(v),
                        ),
                        ty: ret_ty.clone(),
                        span,
                    },
                    span,
                }),
                span,
            });
            out.push(PStatement {
                kind: PStatementKind::Return(PAtomic::Var(vid, vname)),
                span,
            });
        } else {
            out.push(PStatement {
                kind: PStatementKind::Return(v),
                span,
            });
        }
    }

    /// `break` at source level. Inside the lifted function, this becomes
    /// `return LoopResult_N::Exit(<current live-out tuple>)`.
    fn emit_loop_break(&mut self, span: Span, out: &mut Vec<PStatement>) {
        let ctx = self
            .loop_stack
            .last()
            .cloned()
            .expect("break outside a loop");
        let payload = self.build_exit_payload(&ctx, span, out);
        let ret_ty = self.loop_result_ty(&ctx);
        let (wid, wname) = self.fresh_name("exit");
        out.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: wid,
                original_name: wname.clone(),
                ty: ret_ty.clone(),
                value: PExpr {
                    kind: PExprKind::VariantLit(
                        ctx.loop_result_enum_id,
                        "Exit".to_string(),
                        Some(payload),
                    ),
                    ty: ret_ty.clone(),
                    span,
                },
                span,
            }),
            span,
        });
        out.push(PStatement {
            kind: PStatementKind::Return(PAtomic::Var(wid, wname)),
            span,
        });
    }

    /// `continue` at source level: emit a tail-call to the lifted function
    /// with the current live-var values, then return its result. Under
    /// nesting, this remains local to the nearest enclosing loop.
    fn emit_loop_continue(&mut self, span: Span, out: &mut Vec<PStatement>) {
        let ctx = self
            .loop_stack
            .last()
            .cloned()
            .expect("continue outside a loop");
        let args = self.collect_live_args(&ctx);
        let ret_ty = self.loop_result_ty(&ctx);
        let (rid, rname) = self.fresh_name("cont_r");
        out.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: rid,
                original_name: rname.clone(),
                ty: ret_ty.clone(),
                value: PExpr {
                    kind: PExprKind::FuncCall(PFuncCall::User(PUserFuncCall {
                        name: ctx.func_name_id,
                        original_name: ctx.func_original_name.clone(),
                        args,
                        return_type: ret_ty.clone(),
                        is_free: true,
                        span,
                    })),
                    ty: ret_ty.clone(),
                    span,
                },
                span,
            }),
            span,
        });
        out.push(PStatement {
            kind: PStatementKind::Return(PAtomic::Var(rid, rname)),
            span,
        });
    }

    /// Build a tuple of current env values for each live-out variable of the
    /// given loop context. Used for `break` and for fallthrough tail-call
    /// argument construction.
    fn build_exit_payload(
        &mut self,
        ctx: &LoopCtx,
        span: Span,
        out: &mut Vec<PStatement>,
    ) -> PAtomic {
        let elems: Vec<PAtomic> = ctx
            .live_vars
            .iter()
            .map(|(nid, name, _)| {
                self.env
                    .get(nid)
                    .cloned()
                    .unwrap_or_else(|| PAtomic::Var(*nid, name.clone()))
            })
            .collect();
        let (tid, tname) = self.fresh_name("exit_tup");
        out.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: tid,
                original_name: tname.clone(),
                ty: ctx.exit_payload_ty.clone(),
                value: PExpr {
                    kind: PExprKind::TupleLit(elems),
                    ty: ctx.exit_payload_ty.clone(),
                    span,
                },
                span,
            }),
            span,
        });
        PAtomic::Var(tid, tname)
    }

    /// Compute the live-in set for a loop body: threaded NameIds that are
    /// read inside the body and were NOT (re-)bound inside the body. Filtered
    /// to those currently in scope (`env.contains_key`).
    fn loop_live_in(&self, body: &[TStatement]) -> Vec<(NameId, String, Type)> {
        let mut reads: HashSet<NameId> = HashSet::new();
        let mut defs: HashSet<NameId> = HashSet::new();
        for stmt in body {
            reads_stmt(stmt, &mut reads, &mut defs);
        }
        let mut ids: Vec<NameId> = reads
            .difference(&defs)
            .copied()
            .filter(|nid| self.env.contains_key(nid))
            .collect();
        ids.sort_by_key(|n| n.0);
        ids.into_iter()
            .map(|nid| {
                let name = self
                    .orig_name
                    .get(&nid)
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", nid.0));
                let ty = self
                    .type_env
                    .get(&nid)
                    .cloned()
                    .unwrap_or_else(|| panic!("no type for live-in var {:?}", nid));
                (nid, name, ty)
            })
            .collect()
    }

    fn lower_loop(
        &mut self,
        body: Vec<TStatement>,
        loop_span: Span,
        out: &mut Vec<PStatement>,
    ) {
        // 1. Live-in analysis. Conservative live_out = live_in.
        let live_vars = self.loop_live_in(&body);

        // 2. Synthesize __LoopResult_N enum.
        let (enum_id, enum_name) = self.fresh_name("LoopResult");
        let exit_payload_ty =
            Type::Tuple(live_vars.iter().map(|(_, _, t)| t.clone()).collect());
        let return_payload_ty = self
            .current_func_return_type
            .clone()
            .unwrap_or(Type::Nil);
        self.enum_defs.insert(
            enum_id,
            vec![
                ("Exit".to_string(), Some(exit_payload_ty.clone())),
                ("Return".to_string(), Some(return_payload_ty)),
            ],
        );
        self.id_to_name.insert(enum_id, enum_name.clone());
        let loop_result_ty = Type::Enum(enum_id, enum_name.clone());

        // 3. Allocate the lifted function name.
        let (func_id, func_name) = self.fresh_name("loop");
        self.id_to_name.insert(func_id, func_name.clone());

        // 4. Allocate fresh parameter NameIds. Inside the lifted function,
        //    reads of each live-in threaded NameId resolve to its param Var.
        let fresh_params: Vec<(NameId, String)> = live_vars
            .iter()
            .map(|(_, name, _)| self.fresh_name(name))
            .collect();
        let params: Vec<PFuncParam> = live_vars
            .iter()
            .zip(fresh_params.iter())
            .map(|((_, _, ty), (pid, pname))| PFuncParam {
                name: *pid,
                original_name: pname.clone(),
                ty: ty.clone(),
                span: loop_span,
            })
            .collect();

        // 5. Push loop context, swap env for the lifted body, lower, restore.
        let ctx = LoopCtx {
            loop_result_enum_id: enum_id,
            loop_result_enum_name: enum_name.clone(),
            func_name_id: func_id,
            func_original_name: func_name.clone(),
            live_vars: live_vars.clone(),
            exit_payload_ty: exit_payload_ty.clone(),
            span: loop_span,
        };

        let saved_env = std::mem::replace(&mut self.env, HashMap::new());
        for ((orig_nid, _, ty), (pid, pname)) in live_vars.iter().zip(fresh_params.iter()) {
            self.env
                .insert(*orig_nid, PAtomic::Var(*pid, pname.clone()));
            self.type_env.insert(*pid, ty.clone());
            self.orig_name.insert(*pid, pname.clone());
        }
        self.loop_stack.push(ctx.clone());

        let mut body_stmts: Vec<PStatement> = Vec::new();
        for stmt in body {
            self.lower_statement(stmt, &mut body_stmts);
        }

        // 6. Fallthrough = tail-call self with current live-var values.
        //    let __r = __loop_N(...); return __r;
        let tail_args = self.collect_live_args(&ctx);
        let (tail_rid, tail_rname) = self.fresh_name("r");
        body_stmts.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: tail_rid,
                original_name: tail_rname.clone(),
                ty: loop_result_ty.clone(),
                value: PExpr {
                    kind: PExprKind::FuncCall(PFuncCall::User(PUserFuncCall {
                        name: func_id,
                        original_name: func_name.clone(),
                        args: tail_args,
                        return_type: loop_result_ty.clone(),
                        is_free: true,
                        span: loop_span,
                    })),
                    ty: loop_result_ty.clone(),
                    span: loop_span,
                },
                span: loop_span,
            }),
            span: loop_span,
        });
        body_stmts.push(PStatement {
            kind: PStatementKind::Return(PAtomic::Var(tail_rid, tail_rname)),
            span: loop_span,
        });

        self.loop_stack.pop();
        self.env = saved_env;

        // 7. Register the lifted function.
        self.lifted_funcs.push(PFuncDef {
            name: func_id,
            original_name: func_name.clone(),
            kind: PFuncKind::LoopConverted,
            is_traced: false,
            params,
            return_type: loop_result_ty.clone(),
            body: PBlock {
                statements: body_stmts,
                tail_expr: None,
                ty: loop_result_ty.clone(),
                span: loop_span,
            },
            span: loop_span,
        });

        // 8. Outer caller. Call the lifted function, test IsVariant(r,
        //    "Return") to propagate returns, otherwise destructure the Exit
        //    payload and rebind live-out vars in outer env.
        let outer_args: Vec<PAtomic> = ctx
            .live_vars
            .iter()
            .map(|(nid, name, _)| {
                self.env
                    .get(nid)
                    .cloned()
                    .unwrap_or_else(|| PAtomic::Var(*nid, name.clone()))
            })
            .collect();

        let (outer_rid, outer_rname) = self.fresh_name("loop_r");
        out.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: outer_rid,
                original_name: outer_rname.clone(),
                ty: loop_result_ty.clone(),
                value: PExpr {
                    kind: PExprKind::FuncCall(PFuncCall::User(PUserFuncCall {
                        name: func_id,
                        original_name: func_name.clone(),
                        args: outer_args,
                        return_type: loop_result_ty.clone(),
                        is_free: true,
                        span: loop_span,
                    })),
                    ty: loop_result_ty.clone(),
                    span: loop_span,
                },
                span: loop_span,
            }),
            span: loop_span,
        });
        let outer_r_atom = PAtomic::Var(outer_rid, outer_rname);

        // IsVariant(outer_r, "Return")
        let (isret_id, isret_name) = self.fresh_name("is_return");
        out.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: isret_id,
                original_name: isret_name.clone(),
                ty: Type::Bool,
                value: PExpr {
                    kind: PExprKind::IsVariant(outer_r_atom.clone(), "Return".to_string()),
                    ty: Type::Bool,
                    span: loop_span,
                },
                span: loop_span,
            }),
            span: loop_span,
        });

        // if is_return { let rp = VariantPayload(r); <emit_function_return(rp)> }
        let mut then_stmts: Vec<PStatement> = Vec::new();
        let (rp_id, rp_name) = self.fresh_name("ret_payload");
        let return_payload_ty = self
            .current_func_return_type
            .clone()
            .unwrap_or(Type::Nil);
        then_stmts.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: rp_id,
                original_name: rp_name.clone(),
                ty: return_payload_ty.clone(),
                value: PExpr {
                    kind: PExprKind::VariantPayload(outer_r_atom.clone()),
                    ty: return_payload_ty.clone(),
                    span: loop_span,
                },
                span: loop_span,
            }),
            span: loop_span,
        });
        self.emit_function_return(
            PAtomic::Var(rp_id, rp_name),
            loop_span,
            &mut then_stmts,
        );
        let then_block = PBlock {
            statements: then_stmts,
            tail_expr: None,
            ty: Type::Nil,
            span: loop_span,
        };
        let if_pexpr = PExpr {
            kind: PExprKind::Conditional(Box::new(PCondExpr {
                if_branch: PIfBranch {
                    condition: PAtomic::Var(isret_id, isret_name),
                    body: then_block,
                    span: loop_span,
                },
                elseif_branches: vec![],
                else_branch: None,
                span: loop_span,
            })),
            ty: Type::Nil,
            span: loop_span,
        };
        out.push(PStatement {
            kind: PStatementKind::Expr(if_pexpr),
            span: loop_span,
        });

        // Unconditionally destructure the Exit payload.
        // let __exit_payload = VariantPayload(outer_r);
        let (ep_id, ep_name) = self.fresh_name("exit_payload");
        out.push(PStatement {
            kind: PStatementKind::LetAtom(PLetAtom {
                name: ep_id,
                original_name: ep_name.clone(),
                ty: ctx.exit_payload_ty.clone(),
                value: PExpr {
                    kind: PExprKind::VariantPayload(outer_r_atom.clone()),
                    ty: ctx.exit_payload_ty.clone(),
                    span: loop_span,
                },
                span: loop_span,
            }),
            span: loop_span,
        });
        // For each live-out var, project and rebind in outer env.
        for (i, (xid, xname, xty)) in ctx.live_vars.iter().enumerate() {
            let (new_xid, new_xname) = self.fresh_name(xname);
            out.push(PStatement {
                kind: PStatementKind::LetAtom(PLetAtom {
                    name: new_xid,
                    original_name: new_xname.clone(),
                    ty: xty.clone(),
                    value: PExpr {
                        kind: PExprKind::TupleAccess(
                            PAtomic::Var(ep_id, ep_name.clone()),
                            i,
                        ),
                        ty: xty.clone(),
                        span: loop_span,
                    },
                    span: loop_span,
                }),
                span: loop_span,
            });
            self.type_env.insert(new_xid, xty.clone());
            self.orig_name.insert(new_xid, xname.clone());
            self.env.insert(*xid, PAtomic::Var(new_xid, new_xname));
        }
    }

    fn collect_live_args(&self, ctx: &LoopCtx) -> Vec<PAtomic> {
        ctx.live_vars
            .iter()
            .map(|(nid, name, _)| {
                self.env
                    .get(nid)
                    .cloned()
                    .unwrap_or_else(|| PAtomic::Var(*nid, name.clone()))
            })
            .collect()
    }
}

// ===== Module-level read-collection helpers =====

fn reads_atomic(a: &TAtomic, reads: &mut HashSet<NameId>) {
    if let TAtomic::Var(nid, _) = a {
        reads.insert(*nid);
    }
}

fn reads_expr(e: &TExpr, reads: &mut HashSet<NameId>) {
    use TExprKind::*;
    match &e.kind {
        Atomic(a)
        | Not(a)
        | Negate(a)
        | Head(a)
        | Tail(a)
        | Len(a)
        | UnwrapOptional(a)
        | MakeIter(a)
        | IterIsDone(a)
        | IterNext(a)
        | WrapInOptional(a)
        | PersistData(a)
        | IsVariant(a, _)
        | VariantPayload(a)
        | TupleAccess(a, _)
        | FieldAccess(a, _)
        | SafeFieldAccess(a, _)
        | SafeTupleAccess(a, _)
        | Fifo(a) => reads_atomic(a, reads),
        BinOp(_, a, b)
        | Append(a, b)
        | Prepend(a, b)
        | Min(a, b)
        | Exists(a, b)
        | Erase(a, b)
        | Index(a, b)
        | SafeIndex(a, b)
        | Recv(a, b) => {
            reads_atomic(a, reads);
            reads_atomic(b, reads);
        }
        Store(a, b, c) | Slice(a, b, c) | Send(a, b, c) => {
            reads_atomic(a, reads);
            reads_atomic(b, reads);
            reads_atomic(c, reads);
        }
        FuncCall(call) => match call {
            TFuncCall::User(u) => {
                for a in &u.args {
                    reads_atomic(a, reads);
                }
            }
            TFuncCall::Builtin(_, args, _) => {
                for a in args {
                    reads_atomic(a, reads);
                }
            }
        },
        MapLit(pairs) => {
            for (k, v) in pairs {
                reads_atomic(k, reads);
                reads_atomic(v, reads);
            }
        }
        ListLit(items) | TupleLit(items) => {
            for a in items {
                reads_atomic(a, reads);
            }
        }
        RpcCall(target, u) => {
            reads_atomic(target, reads);
            for a in &u.args {
                reads_atomic(a, reads);
            }
        }
        Conditional(c) => {
            reads_atomic(&c.if_branch.condition, reads);
            let mut inner_defs = HashSet::new();
            reads_block(&c.if_branch.body, reads, &mut inner_defs);
            for b in &c.elseif_branches {
                reads_atomic(&b.condition, reads);
                let mut d = HashSet::new();
                reads_block(&b.body, reads, &mut d);
            }
            if let Some(eb) = &c.else_branch {
                let mut d = HashSet::new();
                reads_block(eb, reads, &mut d);
            }
        }
        Block(b) => {
            let mut d = HashSet::new();
            reads_block(b, reads, &mut d);
        }
        VariantLit(_, _, payload) => {
            if let Some(a) = payload {
                reads_atomic(a, reads);
            }
        }
        StructLit(_, fields) => {
            for (_, a) in fields {
                reads_atomic(a, reads);
            }
        }
        MakeChannel | SetTimer(_) | RetrieveData(_) | DiscardData => {}
    }
}

fn reads_stmt(stmt: &TStatement, reads: &mut HashSet<NameId>, defs: &mut HashSet<NameId>) {
    match &stmt.kind {
        TStatementKind::LetAtom(la) => {
            reads_expr(&la.value, reads);
            defs.insert(la.name);
        }
        TStatementKind::Assign(a) => {
            // Treat the target as a "read" for loop live-in analysis: any
            // pre-existing var rebound inside a loop body must be threaded
            // through the loop's exit payload so callers see the update.
            reads.insert(a.target_id);
            reads_atomic(&a.value, reads);
        }
        TStatementKind::Expr(e) => reads_expr(e, reads),
        TStatementKind::Return(a) => reads_atomic(a, reads),
        TStatementKind::Loop(body) => {
            for s in body {
                reads_stmt(s, reads, defs);
            }
        }
        TStatementKind::Break | TStatementKind::Continue | TStatementKind::Error => {}
    }
}

fn reads_block(block: &TBlock, reads: &mut HashSet<NameId>, defs: &mut HashSet<NameId>) {
    for stmt in &block.statements {
        reads_stmt(stmt, reads, defs);
    }
    if let Some(a) = &block.tail_expr {
        reads_atomic(a, reads);
    }
}

pub fn lower_program(program: TProgram) -> PProgram {
    let mut lowerer = PureLowerer::new(
        program.next_name_id,
        program.struct_defs,
        program.enum_defs,
        program.id_to_name,
    );
    let mut top_level_defs: Vec<PTopLevelDef> = program
        .top_level_defs
        .into_iter()
        .map(|d| lowerer.lower_top_level(d))
        .collect();
    // Append lifted loop functions as free functions at the end, preserving
    // emission order (outer loops lifted before inner).
    for lifted in lowerer.lifted_funcs.drain(..) {
        top_level_defs.push(PTopLevelDef::FreeFunc(lifted));
    }
    PProgram {
        top_level_defs,
        next_name_id: lowerer.next_name_id,
        id_to_name: lowerer.id_to_name,
        struct_defs: lowerer.struct_defs,
        enum_defs: lowerer.enum_defs,
    }
}
