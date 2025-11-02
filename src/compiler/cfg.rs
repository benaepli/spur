use crate::analysis::resolver::NameId;
use crate::analysis::types::{
    TypedCondStmts, TypedExpr, TypedExprKind, TypedForInLoop,
    TypedForLoop, TypedForLoopInit, TypedFuncCall, TypedFuncDef, TypedPattern,
    TypedPatternKind, TypedProgram, TypedStatement, TypedStatementKind,
    TypedTopLevelDef, TypedVarInit,
};
use crate::parser::BinOp;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Expr {
    EVar(String),
    EFind(Box<Expr>, Box<Expr>),
    EInt(i64),
    EBool(bool),
    ENot(Box<Expr>),
    EAnd(Box<Expr>, Box<Expr>),
    EOr(Box<Expr>, Box<Expr>),
    EEqualsEquals(Box<Expr>, Box<Expr>),
    EMap(Vec<(Expr, Expr)>),
    EList(Vec<Expr>),
    EListPrepend(Box<Expr>, Box<Expr>),
    EListAppend(Box<Expr>, Box<Expr>),
    EListSubsequence(Box<Expr>, Box<Expr>, Box<Expr>),
    EString(String),
    ELessThan(Box<Expr>, Box<Expr>),
    ELessThanEquals(Box<Expr>, Box<Expr>),
    EGreaterThan(Box<Expr>, Box<Expr>),
    EGreaterThanEquals(Box<Expr>, Box<Expr>),
    EKeyExists(Box<Expr>, Box<Expr>),
    EMapErase(Box<Expr>, Box<Expr>),
    EListLen(Box<Expr>),
    EListAccess(Box<Expr>, usize),
    EPlus(Box<Expr>, Box<Expr>),
    EMinus(Box<Expr>, Box<Expr>),
    ETimes(Box<Expr>, Box<Expr>),
    EDiv(Box<Expr>, Box<Expr>),
    EMod(Box<Expr>, Box<Expr>),
    EPollForResps(Box<Expr>, Box<Expr>),
    EPollForAnyResp(Box<Expr>),
    ENextResp(Box<Expr>),
    EMin(Box<Expr>, Box<Expr>),
    ETuple(Vec<Expr>),
    ETupleAccess(Box<Expr>, usize),
    EUnit,
    ENil,
    EUnwrap(Box<Expr>),
    ECoalesce(Box<Expr>, Box<Expr>),
    ECreatePromise,
    ECreateLock,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Lhs {
    Var(String),
    Access(Expr, Expr),
    Tuple(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Instr {
    Assign(Lhs, Expr),
    Async(Lhs, Expr, String, Vec<Expr>),
    Copy(Lhs, Expr),
    Resolve(Lhs, Expr),
}

pub type Vertex = usize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionInfo {
    pub entry: Vertex,
    pub name: String,
    pub formals: Vec<String>,
    pub locals: Vec<(String, Expr)>, // (name, default_value)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Label {
    Instr(Instr, Vertex /* next_vertex */),
    Pause(Vertex /* next_vertex */),
    Await(Lhs, Expr, Vertex /* next_vertex */),
    Return(Expr),
    Cond(
        Expr,
        Vertex, /* then_vertex */
        Vertex, /* else_vertex */
    ),
    ForLoopIn(
        Lhs,
        Expr,
        Vertex, /* body_vertex */
        Vertex, /* next_vertex */
    ),
    Print(Expr, Vertex /* next_vertex */),
    Break(Vertex /* break_target_vertex */),
    Lock(Expr, Vertex /* next_vertex */),
    Unlock(Expr, Vertex /* next_vertex */),
}

#[derive(Debug, Clone)]
pub enum Value {
    VInt(i64),
    VBool(bool),
    VString(String),
    VNode(i64),
    VUnit,
    VOption(Option<Box<Value>>),
    VTuple(Vec<Value>),
    VList(Rc<RefCell<Vec<Value>>>),
    VFuture(Rc<RefCell<Option<Value>>>),
    VMap(Rc<RefCell<HashMap<Value, Value>>>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    // The CFG is just a list of all vertices.
    // The `Vertex` indices in `Label` and `FunctionInfo`
    // are indices into this Vec.
    pub cfg: Vec<Label>,

    // Map of function_name -> FunctionInfo
    pub rpc: HashMap<String, FunctionInfo>,
    pub client_ops: HashMap<String, FunctionInfo>,
}

pub struct Compiler {
    /// The CFG being built. New labels are pushed here.
    cfg: Vec<Label>,
    /// Counter for generating fresh temporary variable names.
    temp_counter: usize,

    rpc_map: HashMap<String, FunctionInfo>,
    client_ops_map: HashMap<String, FunctionInfo>,
    compiling_for_role: bool,
}

fn resolved_name(name_id: NameId, original_name: &str) -> String {
    format!("{}_{}", name_id.0, original_name)
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            cfg: Vec::new(),
            temp_counter: 0,
            rpc_map: HashMap::new(),
            client_ops_map: HashMap::new(),
            compiling_for_role: false,
        }
    }

    /// Allocates a new, unique temporary variable name.
    fn new_temp_var(&mut self) -> String {
        let name = format!("_tmp{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    /// Adds a new label to the CFG and returns its Vertex index.
    fn add_label(&mut self, label: Label) -> Vertex {
        let vertex = self.cfg.len();
        self.cfg.push(label);
        vertex
    }

    /// The main entry point.
    pub fn compile_program(mut self, program: TypedProgram) -> Program {
        // Compile all top-level definitions
        for def in program.top_level_defs {
            match def {
                TypedTopLevelDef::Role(role) => {
                    self.compiling_for_role = true;
                    // Compile role's var_inits into a special init function
                    let init_fn = self.compile_init_func(
                        &role.var_inits,
                        format!(
                            "{}_BASE_NODE_INIT",
                            resolved_name(role.name, &role.original_name)
                        ),
                    );
                    self.rpc_map.insert(init_fn.name.clone(), init_fn);

                    // Compile all other functions in the role
                    for func in role.func_defs {
                        let func_info = self.compile_func_def(func);
                        self.rpc_map.insert(func_info.name.clone(), func_info);
                    }
                }
            }
        }

        self.compiling_for_role = false;

        // Compile client definition
        let client_init_fn = self.compile_init_func(
            &program.client_def.var_inits,
            "BASE_CLIENT_INIT".to_string(),
        );
        self.client_ops_map
            .insert(client_init_fn.name.clone(), client_init_fn);
        for func in program.client_def.func_defs {
            let func_info = self.compile_func_def(func);
            self.client_ops_map
                .insert(func_info.name.clone(), func_info);
        }

        Program {
            cfg: self.cfg,
            rpc: self.rpc_map,
            client_ops: self.client_ops_map,
        }
    }

    fn compile_init_func(&mut self, inits: &[TypedVarInit], name: String) -> FunctionInfo {
        let return_var_name = self.new_temp_var();
        let final_vertex = self.add_label(Label::Return(Expr::EVar(return_var_name.clone())));

        let mut next_vertex = final_vertex;
        // Build the init chain backwards
        for init in inits.iter().rev() {
            let var_name = resolved_name(init.name, &init.original_name);
            next_vertex = self.compile_expr_to_value(&init.value, Lhs::Var(var_name), next_vertex);
        }

        let entry = next_vertex;
        FunctionInfo {
            entry,
            name,
            formals: vec![],
            locals: vec![(return_var_name, Expr::EUnit)], // Inits are "globals"
        }
    }

    fn compile_func_def(&mut self, func: TypedFuncDef) -> FunctionInfo {
        let return_var_name = self.new_temp_var();
        let final_return_vertex =
            self.add_label(Label::Return(Expr::EVar(return_var_name.clone())));

        let entry = self.compile_block(
            &func.body,
            final_return_vertex,
            final_return_vertex, // `break_target` (breaks go to end of function)
            final_return_vertex, // `return_target`
            &return_var_name,
        );

        let formals = func
            .params
            .into_iter()
            .map(|p| resolved_name(p.name, &p.original_name))
            .collect();

        let mut locals = self.scan_body(&func.body);
        locals.push((return_var_name.clone(), Expr::EUnit)); // Default return val

        FunctionInfo {
            entry,
            name: resolved_name(func.name, &func.original_name),
            formals,
            locals,
        }
    }

    fn compile_block(
        &mut self,
        body: &[TypedStatement],
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        let mut next = next_vertex;
        // Iterate in reverse to chain statements:
        // StmtN -> next_vertex
        // StmtN-1 -> StmtN
        // ...
        // Stmt0 -> Stmt1
        for stmt in body.iter().rev() {
            next = self.compile_statement(stmt, next, break_target, return_target, return_var);
        }
        next
    }

    fn compile_statement(
        &mut self,
        stmt: &TypedStatement,
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        match &stmt.kind {
            TypedStatementKind::VarInit(init) => {
                // Compile the value, assigning to the new variable
                self.compile_expr_to_value(
                    &init.value,
                    Lhs::Var(resolved_name(init.name, &init.original_name)),
                    next_vertex,
                )
            }
            TypedStatementKind::Assignment(assign) => {
                // Compile the value, assigning to the target LHS
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, next_vertex)
            }
            TypedStatementKind::Expr(expr) => {
                // Compile the expression, but discard the result into a dummy var
                let dummy_var = self.new_temp_var();
                self.compile_expr_to_value(expr, Lhs::Var(dummy_var), next_vertex)
            }
            TypedStatementKind::Return(expr) => {
                // Compile the expression, store result in the dedicated return_var,
                // and then jump to the function's final return_target.
                self.compile_expr_to_value(expr, Lhs::Var(return_var.to_string()), return_target)
            }
            TypedStatementKind::Print(expr) => {
                let print_var = self.new_temp_var();
                let print_label = Label::Print(Expr::EVar(print_var.clone()), next_vertex);
                let print_vertex = self.add_label(print_label);
                self.compile_expr_to_value(expr, Lhs::Var(print_var), print_vertex)
            }
            TypedStatementKind::ForLoop(loop_stmt) => {
                self.compile_for_loop(loop_stmt, next_vertex, return_target, return_var)
            }
            TypedStatementKind::Conditional(cond) => {
                self.compile_conditional(cond, next_vertex, break_target, return_target, return_var)
            }
            TypedStatementKind::ForInLoop(loop_stmt) => {
                // `next_vertex` is the break target for a loop
                self.compile_for_in_loop(loop_stmt, next_vertex, return_target, return_var)
            }
            TypedStatementKind::Break => self.add_label(Label::Break(break_target)),
            TypedStatementKind::Lock(lock_expr, body) => self.compile_lock_statement(
                lock_expr,
                body,
                next_vertex,
                break_target,
                return_target,
                return_var,
            ),
        }
    }

    fn compile_for_loop(
        &mut self,
        loop_stmt: &TypedForLoop,
        next_vertex: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        // The [Condition] check is the "head" of the loop.
        let cond_vertex = self.add_label(Label::Return(Expr::EUnit)); // Dummy label

        // Compile the [Increment] block.
        //    After it runs, it loops back to the [Condition].
        let increment_vertex = match &loop_stmt.increment {
            Some(assign) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, cond_vertex)
            }
            None => cond_vertex, // No increment, just jump to condition
        };

        // Compile the [Body] block.
        // After it runs, it goes to [Increment].
        // Any `Break` inside it goes to `next_vertex`.
        let body_vertex = self.compile_block(
            &loop_stmt.body,
            increment_vertex, // After body, go to increment
            next_vertex,      // `break` goes to after the loop
            return_target,
            return_var,
        );

        let cond_expr = loop_stmt.condition.as_ref().map_or(
            Expr::EBool(true), // No condition == `true`
            |c| self.convert_simple_expr(c),
        );
        let cond_label = Label::Cond(
            cond_expr,
            body_vertex, // On true, go to body
            next_vertex, // On false, exit loop
        );

        self.cfg[cond_vertex] = cond_label;

        // Compile the [Init] block.
        // It runs once, then goes to [Condition].
        let init_vertex = match &loop_stmt.init {
            Some(TypedForLoopInit::VarInit(vi)) => self.compile_expr_to_value(
                &vi.value,
                Lhs::Var(resolved_name(vi.name, &vi.original_name)),
                cond_vertex,
            ),
            Some(TypedForLoopInit::Assignment(assign)) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, cond_vertex)
            }
            None => cond_vertex,
        };

        init_vertex
    }

    fn compile_conditional(
        &mut self,
        cond: &TypedCondStmts,
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        let else_vertex = cond.else_branch.as_ref().map_or(next_vertex, |body| {
            self.compile_block(body, next_vertex, break_target, return_target, return_var)
        });

        let mut next_cond_vertex = else_vertex;
        for branch in cond.elseif_branches.iter().rev() {
            let body_vertex = self.compile_block(
                &branch.body,
                next_vertex,
                break_target,
                return_target,
                return_var,
            );
            let cond_expr = self.convert_simple_expr(&branch.condition);
            next_cond_vertex =
                self.add_label(Label::Cond(cond_expr, body_vertex, next_cond_vertex));
        }

        let if_body_vertex = self.compile_block(
            &cond.if_branch.body,
            next_vertex,
            break_target,
            return_target,
            return_var,
        );
        let if_cond_expr = self.convert_simple_expr(&cond.if_branch.condition);

        self.add_label(Label::Cond(if_cond_expr, if_body_vertex, next_cond_vertex))
    }

    fn compile_for_in_loop(
        &mut self,
        loop_stmt: &TypedForInLoop,
        next_vertex: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        // The [ForLoopIn] label is the "head" of the loop.
        // We add a dummy label to reserve its vertex index.
        let for_vertex = self.add_label(Label::Return(Expr::EUnit)); // Dummy label

        // Compile the [Body] block.
        // After it runs, it loops back to the [ForLoopIn] check.
        // Any `Break` inside it goes to `next_vertex`.
        let body_vertex = self.compile_block(
            &loop_stmt.body,
            for_vertex,  // Loop back to the ForLoopIn label
            next_vertex, // Break target
            return_target,
            return_var,
        );

        // Now we create the real [ForLoopIn] label.
        let lhs = self.convert_pattern(&loop_stmt.pattern);
        // We iterate over a local copy to avoid modification-during-iteration issues.
        let iterable_copy_var = "_local_copy".to_string();
        let for_label = Label::ForLoopIn(
            lhs,
            Expr::EVar(iterable_copy_var.clone()),
            body_vertex, // On iteration, go to body
            next_vertex, // When done, exit loop
        );

        // Replace the dummy label with the real one.
        self.cfg[for_vertex] = for_label;

        // Compile the [Iterable] expression.
        // This `Copy` instruction runs once, then goes to the [ForLoopIn] label.
        let iterable_expr = self.convert_simple_expr(&loop_stmt.iterable);
        let copy_vertex = self.add_label(Label::Instr(
            Instr::Copy(Lhs::Var(iterable_copy_var), iterable_expr),
            for_vertex, // After copy, go to the loop head
        ));

        // The entry point to the whole statement is the `Copy` instruction.
        copy_vertex
    }

    fn compile_lock_statement(
        &mut self,
        lock_expr: &TypedExpr,
        body: &[TypedStatement],
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        let lock_var = self.new_temp_var();
        let lock_var_expr = Expr::EVar(lock_var.clone());

        let unlock_then_break_vertex =
            self.add_label(Label::Unlock(lock_var_expr.clone(), break_target));

        let unlock_then_return_vertex =
            self.add_label(Label::Unlock(lock_var_expr.clone(), return_target));

        let unlock_then_continue_vertex =
            self.add_label(Label::Unlock(lock_var_expr.clone(), next_vertex));

        let body_vertex = self.compile_block(
            body,
            unlock_then_continue_vertex, // next_vertex for the block
            unlock_then_break_vertex,    // break_target for the block
            unlock_then_return_vertex,   // return_target for the block
            return_var,
        );

        // Create the Lock label, which runs before the body
        let lock_vertex = self.add_label(Label::Lock(lock_var_expr, body_vertex));

        let entry_vertex = self.compile_expr_to_value(lock_expr, Lhs::Var(lock_var), lock_vertex);

        entry_vertex
    }

    // Helper to generate temp vars and EVar expressions for a list
    fn compile_temp_list(&mut self, count: usize) -> (Vec<String>, Vec<Expr>) {
        (0..count)
            .map(|_| {
                let tmp = self.new_temp_var();
                (tmp.clone(), Expr::EVar(tmp))
            })
            .unzip()
    }

    // Helper to generate temp vars and EVar expressions for (key, value) pairs
    fn compile_temp_pairs(
        &mut self,
        count: usize,
    ) -> (Vec<String>, Vec<String>, Vec<(Expr, Expr)>) {
        let mut key_tmps = Vec::with_capacity(count);
        let mut val_tmps = Vec::with_capacity(count);
        let mut simple_pairs = Vec::with_capacity(count);
        for _ in 0..count {
            let key_tmp = self.new_temp_var();
            let val_tmp = self.new_temp_var();
            simple_pairs.push((Expr::EVar(key_tmp.clone()), Expr::EVar(val_tmp.clone())));
            key_tmps.push(key_tmp);
            val_tmps.push(val_tmp);
        }
        (key_tmps, val_tmps, simple_pairs)
    }

    // Helper to recursively compile a list of expressions into a list of temp vars
    fn compile_expr_list_recursive<'a, I>(
        &mut self,
        exprs: I,
        tmps: Vec<String>,
        next_vertex: Vertex,
    ) -> Vertex
    where
        I: DoubleEndedIterator<Item = &'a TypedExpr>,
    {
        let mut next = next_vertex;
        for (expr, tmp) in exprs.rev().zip(tmps.iter().rev()) {
            next = self.compile_expr_to_value(expr, Lhs::Var(tmp.clone()), next);
        }
        next
    }

    fn convert_simple_binop(&mut self, op: &BinOp, left: Box<Expr>, right: Box<Expr>) -> Expr {
        match op {
            BinOp::Add => Expr::EPlus(left, right),
            BinOp::Subtract => Expr::EMinus(left, right),
            BinOp::Multiply => Expr::ETimes(left, right),
            BinOp::Divide => Expr::EDiv(left, right),
            BinOp::Modulo => Expr::EMod(left, right),
            BinOp::And => Expr::EAnd(left, right),
            BinOp::Or => Expr::EOr(left, right),
            BinOp::Equal => Expr::EEqualsEquals(left, right),
            BinOp::NotEqual => Expr::ENot(Box::new(Expr::EEqualsEquals(left, right))),
            BinOp::Less => Expr::ELessThan(left, right),
            BinOp::LessEqual => Expr::ELessThanEquals(left, right),
            BinOp::Greater => Expr::EGreaterThan(left, right),
            BinOp::GreaterEqual => Expr::EGreaterThanEquals(left, right),
            BinOp::Coalesce => Expr::ECoalesce(left, right),
        }
    }

    /// Compiles any expression into CFG labels, storing the
    /// result in `target`.
    /// Returns the entry vertex for this sequence of instructions.
    fn compile_expr_to_value(
        &mut self,
        expr: &TypedExpr,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        match &expr.kind {
            // --- Complex, side-effecting cases ---
            TypedExprKind::FuncCall(call) => self.compile_func_call(call, target, next_vertex),
            TypedExprKind::RpcCall(target_expr, call) => {
                self.compile_rpc_call(target_expr, call, target, next_vertex)
            }
            TypedExprKind::Await(e) => {
                let future_var = self.new_temp_var();
                // The Await label takes the `target` Lhs directly.
                // This is where the result of the future will be stored.
                let await_label = Label::Await(target, Expr::EVar(future_var.clone()), next_vertex);
                let await_vertex = self.add_label(await_label);
                // Compile the inner expression `e` to get the future,
                // storing it in `future_var`.
                self.compile_expr_to_value(e, Lhs::Var(future_var), await_vertex)
            }

            // --- Compound expressions that may contain complex sub-expressions ---
            TypedExprKind::BinOp(op, l, r) => {
                let l_tmp = self.new_temp_var();
                let r_tmp = self.new_temp_var();
                let l_expr = Box::new(Expr::EVar(l_tmp.clone()));
                let r_expr = Box::new(Expr::EVar(r_tmp.clone()));
                let final_expr = self.convert_simple_binop(op, l_expr, r_expr);

                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let r_vertex = self.compile_expr_to_value(r, Lhs::Var(r_tmp), assign_vertex);
                let l_vertex = self.compile_expr_to_value(l, Lhs::Var(l_tmp), r_vertex);
                l_vertex
            }

            TypedExprKind::Not(e) => {
                let e_tmp = self.new_temp_var();
                let final_expr = Expr::ENot(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::Negate(e) => {
                let e_tmp = self.new_temp_var();
                let final_expr =
                    Expr::EMinus(Box::new(Expr::EInt(0)), Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::MapLit(pairs) => {
                let (key_tmps, val_tmps, simple_pairs) = self.compile_temp_pairs(pairs.len());
                let final_expr = Expr::EMap(simple_pairs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));

                let keys_entry_vertex = self.compile_expr_list_recursive(
                    pairs.iter().map(|(k, _v)| k),
                    key_tmps,
                    assign_vertex,
                );

                self.compile_expr_list_recursive(
                    pairs.iter().map(|(_k, v)| v),
                    val_tmps,
                    keys_entry_vertex,
                )
            }

            TypedExprKind::ListLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(items.len());
                let final_expr = Expr::EList(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex)
            }

            TypedExprKind::TupleLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(items.len());
                let final_expr = Expr::ETuple(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex)
            }

            TypedExprKind::Append(l, i) => {
                let l_tmp = self.new_temp_var();
                let i_tmp = self.new_temp_var();
                let final_expr = Expr::EListAppend(
                    Box::new(Expr::EVar(l_tmp.clone())),
                    Box::new(Expr::EVar(i_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let i_vertex = self.compile_expr_to_value(i, Lhs::Var(i_tmp), assign_vertex);
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), i_vertex)
            }

            TypedExprKind::Prepend(i, l) => {
                let i_tmp = self.new_temp_var();
                let l_tmp = self.new_temp_var();
                let final_expr = Expr::EListPrepend(
                    Box::new(Expr::EVar(i_tmp.clone())),
                    Box::new(Expr::EVar(l_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let l_vertex = self.compile_expr_to_value(l, Lhs::Var(l_tmp), assign_vertex);
                self.compile_expr_to_value(i, Lhs::Var(i_tmp), l_vertex)
            }

            TypedExprKind::Len(e) => {
                let e_tmp = self.new_temp_var();
                let final_expr = Expr::EListLen(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::Min(l, r) => {
                let l_tmp = self.new_temp_var();
                let r_tmp = self.new_temp_var();
                let final_expr = Expr::EMin(
                    Box::new(Expr::EVar(l_tmp.clone())),
                    Box::new(Expr::EVar(r_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let r_vertex = self.compile_expr_to_value(r, Lhs::Var(r_tmp), assign_vertex);
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), r_vertex)
            }

            TypedExprKind::Exists(map, key) => {
                let map_tmp = self.new_temp_var();
                let key_tmp = self.new_temp_var();
                let final_expr = Expr::EKeyExists(
                    Box::new(Expr::EVar(key_tmp.clone())),
                    Box::new(Expr::EVar(map_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let key_vertex = self.compile_expr_to_value(key, Lhs::Var(key_tmp), assign_vertex);
                self.compile_expr_to_value(map, Lhs::Var(map_tmp), key_vertex)
            }

            TypedExprKind::Erase(map, key) => {
                let map_tmp = self.new_temp_var();
                let key_tmp = self.new_temp_var();
                let final_expr = Expr::EMapErase(
                    Box::new(Expr::EVar(key_tmp.clone())),
                    Box::new(Expr::EVar(map_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let key_vertex = self.compile_expr_to_value(key, Lhs::Var(key_tmp), assign_vertex);
                self.compile_expr_to_value(map, Lhs::Var(map_tmp), key_vertex)
            }

            TypedExprKind::Head(l) => {
                let l_tmp = self.new_temp_var();
                let final_expr = Expr::EListAccess(Box::new(Expr::EVar(l_tmp.clone())), 0);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), assign_vertex)
            }

            TypedExprKind::Tail(l) => {
                let l_tmp = self.new_temp_var();
                let list_expr = Expr::EVar(l_tmp.clone());
                let final_expr = Expr::EListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::EInt(1)),
                    Box::new(Expr::EListLen(Box::new(list_expr))),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), assign_vertex)
            }

            TypedExprKind::Index(target_expr, index_expr) => {
                let target_tmp = self.new_temp_var();
                let index_tmp = self.new_temp_var();
                let final_expr = Expr::EFind(
                    Box::new(Expr::EVar(target_tmp.clone())),
                    Box::new(Expr::EVar(index_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let index_vertex =
                    self.compile_expr_to_value(index_expr, Lhs::Var(index_tmp), assign_vertex);
                self.compile_expr_to_value(target_expr, Lhs::Var(target_tmp), index_vertex)
            }

            TypedExprKind::Slice(target_expr, start_expr, end_expr) => {
                let target_tmp = self.new_temp_var();
                let start_tmp = self.new_temp_var();
                let end_tmp = self.new_temp_var();
                let final_expr = Expr::EListSubsequence(
                    Box::new(Expr::EVar(target_tmp.clone())),
                    Box::new(Expr::EVar(start_tmp.clone())),
                    Box::new(Expr::EVar(end_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let end_vertex =
                    self.compile_expr_to_value(end_expr, Lhs::Var(end_tmp), assign_vertex);
                let start_vertex =
                    self.compile_expr_to_value(start_expr, Lhs::Var(start_tmp), end_vertex);
                self.compile_expr_to_value(target_expr, Lhs::Var(target_tmp), start_vertex)
            }

            TypedExprKind::TupleAccess(tuple_expr, index) => {
                let tuple_tmp = self.new_temp_var();
                let final_expr =
                    Expr::ETupleAccess(Box::new(Expr::EVar(tuple_tmp.clone())), *index);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(tuple_expr, Lhs::Var(tuple_tmp), assign_vertex)
            }

            TypedExprKind::FieldAccess(target_expr, field) => {
                let target_tmp = self.new_temp_var();
                let final_expr = Expr::EFind(
                    Box::new(Expr::EVar(target_tmp.clone())),
                    Box::new(Expr::EString(field.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(target_expr, Lhs::Var(target_tmp), assign_vertex)
            }

            TypedExprKind::UnwrapOptional(e) => {
                let e_tmp = self.new_temp_var();
                let final_expr = Expr::EUnwrap(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::StructLit(_, fields) => {
                let (field_names, val_exprs): (Vec<_>, Vec<_>) = fields.iter().cloned().unzip();
                let (tmps, simple_exprs) = self.compile_temp_list(val_exprs.len());
                let final_pairs = field_names
                    .into_iter()
                    .map(Expr::EString)
                    .zip(simple_exprs)
                    .collect();
                let final_expr = Expr::EMap(final_pairs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(val_exprs.iter(), tmps, assign_vertex)
            }

            TypedExprKind::PollForResps(e1, e2) => {
                let e1_tmp = self.new_temp_var();
                let e2_tmp = self.new_temp_var();
                let final_expr = Expr::EPollForResps(
                    Box::new(Expr::EVar(e1_tmp.clone())),
                    Box::new(Expr::EVar(e2_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let e2_vertex = self.compile_expr_to_value(e2, Lhs::Var(e2_tmp), assign_vertex);
                self.compile_expr_to_value(e1, Lhs::Var(e1_tmp), e2_vertex)
            }

            TypedExprKind::PollForAnyResp(e) => {
                let e_tmp = self.new_temp_var();
                let final_expr = Expr::EPollForAnyResp(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::NextResp(e) => {
                let e_tmp = self.new_temp_var();
                let final_expr = Expr::ENextResp(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::ResolvePromise(p, v) => {
                let p_tmp = self.new_temp_var();
                let v_tmp = self.new_temp_var();

                // Assign unit to the target (e.g., `_tmp = ()`)
                let assign_unit_label =
                    Label::Instr(Instr::Assign(target, Expr::EUnit), next_vertex);
                let assign_unit_vertex = self.add_label(assign_unit_label);

                // Resolve the promise (e.g., `resolve(_p_tmp, _v_tmp)`)
                let resolve_label = Label::Instr(
                    Instr::Resolve(Lhs::Var(p_tmp.clone()), Expr::EVar(v_tmp.clone())),
                    assign_unit_vertex,
                );
                let resolve_vertex = self.add_label(resolve_label);

                // Compile p and v
                let v_vertex = self.compile_expr_to_value(v, Lhs::Var(v_tmp), resolve_vertex);
                self.compile_expr_to_value(p, Lhs::Var(p_tmp), v_vertex)
            }

            // --- Truly simple, non-side-effecting cases ---
            TypedExprKind::CreateFuture(e) => {
                // create_future(p) is just a copy of p.
                self.compile_expr_to_value(e, target, next_vertex)
            }

            TypedExprKind::CreatePromise => {
                let label = Label::Instr(Instr::Assign(target, Expr::ECreatePromise), next_vertex);
                self.add_label(label)
            }

            TypedExprKind::CreateLock => {
                let label = Label::Instr(Instr::Assign(target, Expr::ECreateLock), next_vertex);
                self.add_label(label)
            }

            _ => {
                // Desugar the simple expression
                let sim_expr = self.convert_simple_expr(expr);
                // Create a single `Assign` instruction
                let label = Label::Instr(Instr::Assign(target, sim_expr), next_vertex);
                self.add_label(label)
            }
        }
    }

    fn compile_async_call_internal(
        &mut self,
        node_expr: Expr, // The node to call the function on
        call: &TypedFuncCall,
        target: Lhs, // The final destination for the *future*
        next_vertex: Vertex,
    ) -> Vertex {
        // Allocate all temporary variable names up front.
        // Create temp var names for each argument.
        let (arg_temp_vars, arg_exprs): (Vec<String>, Vec<Expr>) = (0..call.args.len())
            .map(|_| {
                let tmp = self.new_temp_var();
                (tmp.clone(), Expr::EVar(tmp))
            })
            .unzip();

        // Build the call chain backwards (Async -> Pause).
        let async_label = Label::Instr(
            Instr::Async(
                target,
                node_expr, // Use the provided node expression
                resolved_name(call.name, &call.original_name),
                arg_exprs,
            ),
            next_vertex,
        );
        let async_vertex = self.add_label(async_label);

        let pause_vertex = self.add_label(Label::Pause(async_vertex));

        // Build the argument compilation chain backwards.
        let mut entry_vertex = pause_vertex;
        for (arg, tmp_var) in call.args.iter().rev().zip(arg_temp_vars.iter().rev()) {
            entry_vertex = self.compile_expr_to_value(arg, Lhs::Var(tmp_var.clone()), entry_vertex);
        }

        // This is the entry point for the "call" part
        entry_vertex
    }

    fn compile_func_call(
        &mut self,
        call: &TypedFuncCall,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        // A local call is just an async call on "self"
        let node_expr = Expr::EVar("self".to_string());
        self.compile_async_call_internal(node_expr, call, target, next_vertex)
    }

    fn compile_rpc_call(
        &mut self,
        target_expr: &TypedExpr,
        call: &TypedFuncCall,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        //  Allocate a temp var to hold the target role
        let target_var = self.new_temp_var();
        let node_expr = Expr::EVar(target_var.clone());

        // Get the entry vertex for the internal call logic
        let internal_call_vertex =
            self.compile_async_call_internal(node_expr, call, target, next_vertex);

        // Compile the target_expr, which runs *before* the internal call
        // The result is stored in `target_var`, which `internal_call_vertex` will use.
        self.compile_expr_to_value(target_expr, Lhs::Var(target_var), internal_call_vertex)
    }

    /// Converts a "simple" `TypedExpr` to a `cfg::Expr`.
    /// Panics if it encounters a complex (side-effecting) expression.
    fn convert_simple_expr(&mut self, expr: &TypedExpr) -> Expr {
        match &expr.kind {
            TypedExprKind::Var(id, name) => Expr::EVar(resolved_name(*id, &name)),
            TypedExprKind::IntLit(i) => Expr::EInt(i.clone()),
            TypedExprKind::StringLit(s) => Expr::EString(s.clone()),
            TypedExprKind::BoolLit(b) => Expr::EBool(b.clone()),
            TypedExprKind::NilLit => Expr::ENil,
            TypedExprKind::BinOp(op, l, r) => {
                let left = Box::new(self.convert_simple_expr(l));
                let right = Box::new(self.convert_simple_expr(r));
                self.convert_simple_binop(op, left, right)
            }
            TypedExprKind::Not(e) => Expr::ENot(Box::new(self.convert_simple_expr(e))),
            TypedExprKind::Negate(e) => Expr::EMinus(
                Box::new(Expr::EInt(0)),
                Box::new(self.convert_simple_expr(e)),
            ),
            TypedExprKind::MapLit(pairs) => Expr::EMap(
                pairs
                    .iter()
                    .map(|(k, v)| (self.convert_simple_expr(k), self.convert_simple_expr(v)))
                    .collect(),
            ),
            TypedExprKind::ListLit(items) => {
                Expr::EList(items.iter().map(|e| self.convert_simple_expr(e)).collect())
            }
            TypedExprKind::TupleLit(items) => {
                Expr::ETuple(items.iter().map(|e| self.convert_simple_expr(e)).collect())
            }
            // Desugar: `append(l, i)` -> `EListAppend(l, i)`
            TypedExprKind::Append(l, i) => Expr::EListAppend(
                Box::new(self.convert_simple_expr(l)),
                Box::new(self.convert_simple_expr(i)),
            ),
            TypedExprKind::Prepend(i, l) => Expr::EListPrepend(
                Box::new(self.convert_simple_expr(i)),
                Box::new(self.convert_simple_expr(l)),
            ),
            // Desugar: `len(e)` -> `EListLen(e)`
            TypedExprKind::Len(e) => Expr::EListLen(Box::new(self.convert_simple_expr(e))),
            TypedExprKind::Min(l, r) => Expr::EMin(
                Box::new(self.convert_simple_expr(l)),
                Box::new(self.convert_simple_expr(r)),
            ),
            TypedExprKind::Exists(map, key) => Expr::EKeyExists(
                Box::new(self.convert_simple_expr(key)),
                Box::new(self.convert_simple_expr(map)),
            ),
            TypedExprKind::Erase(map, key) => Expr::EMapErase(
                Box::new(self.convert_simple_expr(key)),
                Box::new(self.convert_simple_expr(map)),
            ),
            // Desugar: `head(l)` -> `EListAccess(l, 0)`
            TypedExprKind::Head(l) => Expr::EListAccess(Box::new(self.convert_simple_expr(l)), 0),
            // Desugar: `tail(l)` -> `EListSubsequence(l, 1, len(l))`
            TypedExprKind::Tail(l) => {
                let list_expr = self.convert_simple_expr(l);
                Expr::EListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::EInt(1)),
                    Box::new(Expr::EListLen(Box::new(list_expr))),
                )
            }
            TypedExprKind::CreateFuture(e) => self.convert_simple_expr(e),
            // Desugar: `s[i]` -> `EFind(s, i)`
            TypedExprKind::Index(target, index) => Expr::EFind(
                Box::new(self.convert_simple_expr(target)),
                Box::new(self.convert_simple_expr(index)),
            ),
            // Desugar: `s[i:j]` -> `EListSubsequence(s, i, j)`
            TypedExprKind::Slice(target, start, end) => Expr::EListSubsequence(
                Box::new(self.convert_simple_expr(target)),
                Box::new(self.convert_simple_expr(start)),
                Box::new(self.convert_simple_expr(end)),
            ),
            TypedExprKind::TupleAccess(tuple, index) => {
                Expr::ETupleAccess(Box::new(self.convert_simple_expr(tuple)), *index)
            }
            // Desugar: `s.field` -> `EFind(s, "field")`
            TypedExprKind::FieldAccess(target, field) => Expr::EFind(
                Box::new(self.convert_simple_expr(target)),
                Box::new(Expr::EString(field.clone())),
            ),
            TypedExprKind::UnwrapOptional(e) => {
                Expr::EUnwrap(Box::new(self.convert_simple_expr(e)))
            }
            // Desugar: `MyStruct { ... }` -> `EMap { ... }`
            TypedExprKind::StructLit(_, fields) => Expr::EMap(
                fields
                    .iter()
                    .map(|(name, val)| (Expr::EString(name.clone()), self.convert_simple_expr(val)))
                    .collect(),
            ),

            TypedExprKind::CreateLock => Expr::ECreateLock,

            // --- Panic on complex expressions ---
            TypedExprKind::FuncCall(_)
            | TypedExprKind::RpcCall(_, _)
            | TypedExprKind::Await(_)
            | TypedExprKind::CreatePromise
            | TypedExprKind::ResolvePromise(_, _) => {
                panic!(
                    "Cannot have complex call inside a simple expression: {:?}",
                    expr
                )
            }
            TypedExprKind::PollForResps(e1, e2) => Expr::EPollForResps(
                Box::new(self.convert_simple_expr(e1)),
                Box::new(self.convert_simple_expr(e2)),
            ),
            TypedExprKind::PollForAnyResp(e) => {
                Expr::EPollForAnyResp(Box::new(self.convert_simple_expr(e)))
            }
            TypedExprKind::NextResp(e) => Expr::ENextResp(Box::new(self.convert_simple_expr(e))),
        }
    }

    fn convert_lhs(&mut self, expr: &TypedExpr) -> Lhs {
        match &expr.kind {
            TypedExprKind::Var(id, name) => Lhs::Var(resolved_name(*id, name)),
            // Desugar: `s[i] = ...` -> `LAccess(s, i)`
            TypedExprKind::Index(target, index) => Lhs::Access(
                self.convert_simple_expr(target),
                self.convert_simple_expr(index),
            ),
            // Desugar: `s.field = ...` -> `LAccess(s, "field")`
            TypedExprKind::FieldAccess(target, field) => Lhs::Access(
                self.convert_simple_expr(target),
                Expr::EString(field.clone()),
            ),
            _ => panic!("Invalid assignment target: {:?}", expr),
        }
    }

    fn convert_pattern(&mut self, pat: &TypedPattern) -> Lhs {
        match &pat.kind {
            TypedPatternKind::Var(id, name) => Lhs::Var(resolved_name(*id, name)),
            TypedPatternKind::Wildcard => Lhs::Var(self.new_temp_var()),
            TypedPatternKind::Tuple(pats) => {
                let names = pats
                    .iter()
                    .map(|p| {
                        match &p.kind {
                            TypedPatternKind::Var(id, name) => resolved_name(*id, name),
                            _ => self.new_temp_var(), // Use dummy var for wildcard/unit
                        }
                    })
                    .collect();
                Lhs::Tuple(names)
            }
        }
    }

    fn scan_body(&mut self, body: &[TypedStatement]) -> Vec<(String, Expr)> {
        let mut locals = Vec::new();
        for stmt in body {
            self.scan_stmt_for_locals(stmt, &mut locals);
        }
        locals
    }

    fn scan_body_for_locals(&mut self, body: &[TypedStatement], locals: &mut Vec<(String, Expr)>) {
        for stmt in body {
            self.scan_stmt_for_locals(stmt, locals);
        }
    }

    fn scan_stmt_for_locals(&mut self, stmt: &TypedStatement, locals: &mut Vec<(String, Expr)>) {
        match &stmt.kind {
            TypedStatementKind::VarInit(init) => {
                // `let x = ...` adds a local
                locals.push((
                    resolved_name(init.name, &init.original_name),
                    Expr::ENil, // Use placeholder, actual init is handled by CFG
                ));
            }
            TypedStatementKind::Conditional(cond) => {
                self.scan_body_for_locals(&cond.if_branch.body, locals);
                for branch in &cond.elseif_branches {
                    self.scan_body_for_locals(&branch.body, locals);
                }
                if let Some(body) = &cond.else_branch {
                    self.scan_body_for_locals(body, locals);
                }
            }
            TypedStatementKind::ForLoop(fl) => {
                // This is the C-style loop, which might have an init
                if let Some(TypedForLoopInit::VarInit(init)) = &fl.init {
                    locals.push((
                        resolved_name(init.name, &init.original_name),
                        Expr::ENil, // Use placeholder
                    ));
                }
                self.scan_body_for_locals(&fl.body, locals);
            }
            TypedStatementKind::ForInLoop(loop_stmt) => {
                // `for (x, y) in ...` adds locals
                self.scan_pattern_for_locals(&loop_stmt.pattern, locals);
                self.scan_body_for_locals(&loop_stmt.body, locals);
            }
            // Other statements don't declare locals
            _ => {}
        }
    }

    fn scan_pattern_for_locals(&mut self, pat: &TypedPattern, locals: &mut Vec<(String, Expr)>) {
        match &pat.kind {
            TypedPatternKind::Var(id, name) => {
                // `ENil` is just a placeholder default
                locals.push((resolved_name(*id, name), Expr::ENil));
            }
            TypedPatternKind::Tuple(pats) => {
                for p in pats {
                    self.scan_pattern_for_locals(p, locals);
                }
            }
            _ => {} // Wildcard, Unit don't add named locals
        }
    }
}
