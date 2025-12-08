use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::{
    TypedCondStmts, TypedExpr, TypedExprKind, TypedForInLoop, TypedForLoop, TypedForLoopInit,
    TypedFuncCall, TypedFuncDef, TypedPattern, TypedPatternKind, TypedProgram, TypedStatement,
    TypedStatementKind, TypedTopLevelDef, TypedUserFuncCall, TypedVarInit,
};
use crate::parser::BinOp;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
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
    EStore(Box<Expr>, Box<Expr>, Box<Expr>),
    EListLen(Box<Expr>),
    EListAccess(Box<Expr>, usize),
    EPlus(Box<Expr>, Box<Expr>),
    EMinus(Box<Expr>, Box<Expr>),
    ETimes(Box<Expr>, Box<Expr>),
    EDiv(Box<Expr>, Box<Expr>),
    EMod(Box<Expr>, Box<Expr>),
    EMin(Box<Expr>, Box<Expr>),
    ETuple(Vec<Expr>),
    ETupleAccess(Box<Expr>, usize),
    EUnit,
    ENil,
    EUnwrap(Box<Expr>),
    ECoalesce(Box<Expr>, Box<Expr>),
    ECreatePromise,
    ECreateLock,
    ESome(Box<Expr>),
    EIntToString(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Lhs {
    Var(String),
    Tuple(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Instr {
    Assign(Lhs, Expr),
    Async(Lhs, Expr, String, Vec<Expr>),
    Copy(Lhs, Expr),
    Resolve(Lhs, Expr),
    SyncCall(Lhs, String, Vec<Expr>),
}

pub type Vertex = usize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionInfo {
    pub entry: Vertex,
    pub name: String,
    pub formals: Vec<String>,
    pub locals: Vec<(String, Expr)>, // (name, default_value)
    pub is_sync: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Label {
    Instr(Instr, Vertex /* next_vertex */),
    Pause(Vertex /* next_vertex */),
    Await(Lhs, Expr, Vertex /* next_vertex */),
    SpinAwait(Expr, Vertex /* next_vertex */),
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

pub struct FutureValue {
    pub value: Option<Value>,
    pub waiters: Vec<Box<dyn FnMut(Value)>>,
}

impl fmt::Debug for FutureValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FutureValue")
            .field("value", &self.value)
            .field("waiters", &format!("<{} waiters>", self.waiters.len()))
            .finish()
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    VInt(i64),
    VBool(bool),
    VMap(Rc<RefCell<HashMap<Value, Value>>>),
    VList(Rc<RefCell<Vec<Value>>>),
    VOption(Option<Box<Value>>),
    VFuture(Rc<RefCell<FutureValue>>),
    VLock(Rc<RefCell<bool>>),
    VNode(i64),
    VString(String),
    VUnit,
    VTuple(Vec<Value>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    // The CFG is just a list of all vertices.
    // The `Vertex` indices in `Label` and `FunctionInfo`
    // are indices into this Vec.
    pub cfg: Vec<Label>,

    // Map of function_name -> FunctionInfo
    pub rpc: HashMap<String, FunctionInfo>,
}

pub struct Compiler {
    /// The CFG being built. New labels are pushed here.
    cfg: Vec<Label>,
    /// Counter for generating fresh temporary variable names.
    temp_counter: usize,

    rpc_map: HashMap<String, FunctionInfo>,

    func_sync_map: HashMap<NameId, bool>,
    func_qualifier_map: HashMap<NameId, String>,
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
            func_sync_map: HashMap::new(),
            func_qualifier_map: HashMap::new(),
        }
    }

    /// Allocates a new, unique temporary variable name.
    fn new_temp_var(&mut self, locals: &mut Vec<(String, Expr)>) -> String {
        let name = format!("_tmp{}", self.temp_counter);
        self.temp_counter += 1;
        locals.push((name.clone(), Expr::ENil));
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
        // Build func_sync_map and compile all top-level definitions
        for def in &program.top_level_defs {
            match def {
                TypedTopLevelDef::Role(role) => {
                    let qualifier = role.original_name.clone();
                    for func in &role.func_defs {
                        self.func_sync_map.insert(func.name, func.is_sync);
                        self.func_qualifier_map.insert(func.name, qualifier.clone());
                    }
                }
            }
        }

        // Compile all top-level definitions
        for def in program.top_level_defs {
            match def {
                TypedTopLevelDef::Role(role) => {
                    // Compile role's var_inits into a special init function
                    let init_fn = self.compile_init_func(
                        &role.var_inits,
                        format!("{}.{}", role.original_name, "BASE_NODE_INIT"),
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

        Program {
            cfg: self.cfg,
            rpc: self.rpc_map,
        }
    }

    fn compile_init_func(&mut self, inits: &[TypedVarInit], name: String) -> FunctionInfo {
        let mut locals = Vec::new();
        let return_var_name = self.new_temp_var(&mut locals);
        let final_vertex = self.add_label(Label::Return(Expr::EVar(return_var_name.clone())));

        let mut next_vertex = final_vertex;
        // Build the init chain backwards
        for init in inits.iter().rev() {
            let var_name = resolved_name(init.name, &init.original_name);
            next_vertex = self.compile_expr_to_value(
                &mut locals,
                &init.value,
                Lhs::Var(var_name),
                next_vertex,
            );
        }

        let entry = self.add_label(Label::Instr(
            Instr::Assign(Lhs::Var(return_var_name.clone()), Expr::EUnit),
            next_vertex,
        ));

        if let Some(local) = locals.iter_mut().find(|(n, _)| *n == return_var_name) {
            local.1 = Expr::EUnit;
        }

        FunctionInfo {
            entry,
            name,
            formals: vec![],
            locals, // Inits are "globals"
            is_sync: true,
        }
    }

    fn compile_func_def(&mut self, func: TypedFuncDef) -> FunctionInfo {
        let mut locals = self.scan_body(&func.body);
        let return_var_name = self.new_temp_var(&mut locals);
        let final_return_vertex =
            self.add_label(Label::Return(Expr::EVar(return_var_name.clone())));

        let body_entry = self.compile_block(
            &mut locals,
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

        if let Some(local) = locals.iter_mut().find(|(n, _)| *n == return_var_name) {
            local.1 = Expr::EUnit;
        }

        let entry = self.add_label(Label::Instr(
            Instr::Assign(Lhs::Var(return_var_name.clone()), Expr::EUnit),
            body_entry,
        ));

        let qualifier = self
            .func_qualifier_map
            .get(&func.name)
            .expect("Function qualifier should exist in map");

        FunctionInfo {
            entry,
            name: format!("{}.{}", qualifier, func.original_name),
            formals,
            locals,
            is_sync: func.is_sync,
        }
    }

    fn compile_block(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
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
            next =
                self.compile_statement(locals, stmt, next, break_target, return_target, return_var);
        }
        next
    }

    fn compile_statement(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
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
                    locals,
                    &init.value,
                    Lhs::Var(resolved_name(init.name, &init.original_name)),
                    next_vertex,
                )
            }
            TypedStatementKind::Assignment(assign) => {
                // Compile the value, assigning to the target LHS
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(locals, &assign.value, lhs, next_vertex)
            }
            TypedStatementKind::Expr(expr) => {
                // Compile the expression, but discard the result into a dummy var
                let dummy_var = self.new_temp_var(locals);
                self.compile_expr_to_value(locals, expr, Lhs::Var(dummy_var), next_vertex)
            }
            TypedStatementKind::Return(expr) => {
                // Compile the expression, store result in the dedicated return_var,
                // and then jump to the function's final return_target.
                self.compile_expr_to_value(
                    locals,
                    expr,
                    Lhs::Var(return_var.to_string()),
                    return_target,
                )
            }
            TypedStatementKind::ForLoop(loop_stmt) => {
                self.compile_for_loop(locals, loop_stmt, next_vertex, return_target, return_var)
            }
            TypedStatementKind::Conditional(cond) => self.compile_conditional(
                locals,
                cond,
                next_vertex,
                break_target,
                return_target,
                return_var,
            ),
            TypedStatementKind::ForInLoop(loop_stmt) => {
                // `next_vertex` is the break target for a loop
                self.compile_for_in_loop(locals, loop_stmt, next_vertex, return_target, return_var)
            }
            TypedStatementKind::Break => self.add_label(Label::Break(break_target)),
            TypedStatementKind::Lock(lock_expr, body) => self.compile_lock_statement(
                locals,
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
        locals: &mut Vec<(String, Expr)>,
        loop_stmt: &TypedForLoop,
        next_vertex: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        let loop_head_vertex = self.add_label(Label::Return(Expr::EUnit));

        let increment_vertex = match &loop_stmt.increment {
            Some(assign) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(locals, &assign.value, lhs, loop_head_vertex)
            }
            None => loop_head_vertex,
        };

        let body_vertex = self.compile_block(
            locals,
            &loop_stmt.body,
            increment_vertex, // After body, go to increment
            next_vertex,      // Break goes to exit
            return_target,
            return_var,
        );

        let cond_var = self.new_temp_var(locals);

        let cond_check_vertex = self.add_label(Label::Cond(
            Expr::EVar(cond_var.clone()),
            body_vertex, // True -> Body
            next_vertex, // False -> Exit
        ));

        let cond_expr = loop_stmt.condition.as_ref().cloned().unwrap_or_else(|| {
            // If no condition, default to true
            TypedExpr {
                kind: TypedExprKind::BoolLit(true),
                ty: crate::analysis::types::Type::Bool,
                span: loop_stmt.span,
            }
        });

        let cond_calc_vertex =
            self.compile_expr_to_value(locals, &cond_expr, Lhs::Var(cond_var), cond_check_vertex);

        self.cfg[loop_head_vertex] = Label::Instr(
            Instr::Assign(Lhs::Var(self.new_temp_var(locals)), Expr::EUnit),
            cond_calc_vertex,
        );

        match &loop_stmt.init {
            Some(TypedForLoopInit::VarInit(vi)) => self.compile_expr_to_value(
                locals,
                &vi.value,
                Lhs::Var(resolved_name(vi.name, &vi.original_name)),
                loop_head_vertex,
            ),
            Some(TypedForLoopInit::Assignment(assign)) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(locals, &assign.value, lhs, loop_head_vertex)
            }
            None => loop_head_vertex,
        }
    }

    fn compile_conditional(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
        cond: &TypedCondStmts,
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        let else_vertex = cond.else_branch.as_ref().map_or(next_vertex, |body| {
            self.compile_block(
                locals,
                body,
                next_vertex,
                break_target,
                return_target,
                return_var,
            )
        });

        let mut next_cond_vertex = else_vertex;
        for branch in cond.elseif_branches.iter().rev() {
            let body_vertex = self.compile_block(
                locals,
                &branch.body,
                next_vertex,
                break_target,
                return_target,
                return_var,
            );
            let cond_tmp = self.new_temp_var(locals);
            let check_vertex = self.add_label(Label::Cond(
                Expr::EVar(cond_tmp.clone()),
                body_vertex,
                next_cond_vertex,
            ));
            next_cond_vertex = self.compile_expr_to_value(
                locals,
                &branch.condition,
                Lhs::Var(cond_tmp),
                check_vertex,
            );
        }

        let if_body_vertex = self.compile_block(
            locals,
            &cond.if_branch.body,
            next_vertex,
            break_target,
            return_target,
            return_var,
        );
        let if_cond_tmp = self.new_temp_var(locals);

        let if_check_vertex = self.add_label(Label::Cond(
            Expr::EVar(if_cond_tmp.clone()),
            if_body_vertex,
            next_cond_vertex,
        ));

        self.compile_expr_to_value(
            locals,
            &cond.if_branch.condition,
            Lhs::Var(if_cond_tmp),
            if_check_vertex,
        )
    }

    fn compile_for_in_loop(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
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
            locals,
            &loop_stmt.body,
            for_vertex,  // Loop back to the ForLoopIn label
            next_vertex, // Break target
            return_target,
            return_var,
        );

        // Now we create the real [ForLoopIn] label.
        let lhs = self.convert_pattern(locals, &loop_stmt.pattern);
        // We iterate over a local copy to avoid modification-during-iteration issues.
        let iterable_copy_var = self.new_temp_var(locals);
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
        locals: &mut Vec<(String, Expr)>,
        lock_expr: &TypedExpr,
        body: &[TypedStatement],
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_var: &str,
    ) -> Vertex {
        let lock_var = self.new_temp_var(locals);
        let lock_var_expr = Expr::EVar(lock_var.clone());

        let unlock_then_break_vertex =
            self.add_label(Label::Unlock(lock_var_expr.clone(), break_target));

        let unlock_then_return_vertex =
            self.add_label(Label::Unlock(lock_var_expr.clone(), return_target));

        let unlock_then_continue_vertex =
            self.add_label(Label::Unlock(lock_var_expr.clone(), next_vertex));

        let body_vertex = self.compile_block(
            locals,
            body,
            unlock_then_continue_vertex, // next_vertex for the block
            unlock_then_break_vertex,    // break_target for the block
            unlock_then_return_vertex,   // return_target for the block
            return_var,
        );

        // Create the Lock label, which runs before the body
        let lock_vertex = self.add_label(Label::Lock(lock_var_expr, body_vertex));

        let entry_vertex =
            self.compile_expr_to_value(locals, lock_expr, Lhs::Var(lock_var), lock_vertex);

        entry_vertex
    }

    // Helper to generate temp vars and EVar expressions for a list
    fn compile_temp_list(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
        count: usize,
    ) -> (Vec<String>, Vec<Expr>) {
        (0..count)
            .map(|_| {
                let tmp = self.new_temp_var(locals);
                (tmp.clone(), Expr::EVar(tmp))
            })
            .unzip()
    }

    // Helper to generate temp vars and EVar expressions for (key, value) pairs
    fn compile_temp_pairs(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
        count: usize,
    ) -> (Vec<String>, Vec<String>, Vec<(Expr, Expr)>) {
        let mut key_tmps = Vec::with_capacity(count);
        let mut val_tmps = Vec::with_capacity(count);
        let mut simple_pairs = Vec::with_capacity(count);
        for _ in 0..count {
            let key_tmp = self.new_temp_var(locals);
            let val_tmp = self.new_temp_var(locals);
            simple_pairs.push((Expr::EVar(key_tmp.clone()), Expr::EVar(val_tmp.clone())));
            key_tmps.push(key_tmp);
            val_tmps.push(val_tmp);
        }
        (key_tmps, val_tmps, simple_pairs)
    }

    // Helper to recursively compile a list of expressions into a list of temp vars
    fn compile_expr_list_recursive<'a, I>(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
        exprs: I,
        tmps: Vec<String>,
        next_vertex: Vertex,
    ) -> Vertex
    where
        I: DoubleEndedIterator<Item = &'a TypedExpr>,
    {
        let mut next = next_vertex;
        for (expr, tmp) in exprs.rev().zip(tmps.iter().rev()) {
            next = self.compile_expr_to_value(locals, expr, Lhs::Var(tmp.clone()), next);
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
        locals: &mut Vec<(String, Expr)>,
        expr: &TypedExpr,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        match &expr.kind {
            // --- Complex, side-effecting cases ---
            TypedExprKind::FuncCall(call) => {
                self.compile_func_call(locals, call, target, next_vertex)
            }
            TypedExprKind::RpcCall(target_expr, call) => {
                self.compile_rpc_call(locals, target_expr, call, target, next_vertex)
            }
            TypedExprKind::Await(e) => {
                let future_var = self.new_temp_var(locals);
                // The Await label takes the `target` Lhs directly.
                // This is where the result of the future will be stored.
                let await_label = Label::Await(target, Expr::EVar(future_var.clone()), next_vertex);
                let await_vertex = self.add_label(await_label);
                // Compile the inner expression `e` to get the future,
                // storing it in `future_var`.
                self.compile_expr_to_value(locals, e, Lhs::Var(future_var), await_vertex)
            }
            TypedExprKind::SpinAwait(e) => {
                let bool_var = self.new_temp_var(locals);

                // 1. Assign unit to the target, which runs *after* the spinawait.
                let assign_unit_vertex = self.add_label(Label::Instr(
                    Instr::Assign(target, Expr::EUnit),
                    next_vertex,
                ));

                // 2. The SpinAwait label. It just blocks, then goes to assign unit.
                let spin_label = Label::SpinAwait(Expr::EVar(bool_var.clone()), assign_unit_vertex);
                let spin_vertex = self.add_label(spin_label);

                // 3. Compile the inner expression `e` to get the boolean,
                //    storing it in `bool_var`.
                self.compile_expr_to_value(locals, e, Lhs::Var(bool_var), spin_vertex)
            }

            TypedExprKind::BinOp(op, l, r) => {
                match op {
                    BinOp::And => {
                        let assign_false_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target.clone(), Expr::EBool(false)),
                            next_vertex,
                        ));

                        let eval_right_vertex =
                            self.compile_expr_to_value(locals, r, target.clone(), next_vertex);

                        // Branch based on Left result
                        let l_tmp = self.new_temp_var(locals);
                        let cond_vertex = self.add_label(Label::Cond(
                            Expr::EVar(l_tmp.clone()),
                            eval_right_vertex,   // If True: evaluate right
                            assign_false_vertex, // If False: short-circuit to false
                        ));

                        self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), cond_vertex)
                    }
                    BinOp::Or => {
                        let assign_true_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target.clone(), Expr::EBool(true)),
                            next_vertex,
                        ));

                        let eval_right_vertex =
                            self.compile_expr_to_value(locals, r, target.clone(), next_vertex);

                        // Branch based on Left result
                        let l_tmp = self.new_temp_var(locals);
                        let cond_vertex = self.add_label(Label::Cond(
                            Expr::EVar(l_tmp.clone()),
                            assign_true_vertex, // If True: short-circuit to true
                            eval_right_vertex,  // If False: evaluate right
                        ));

                        self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), cond_vertex)
                    }
                    _ => {
                        // Standard strict evaluation for non-boolean ops (+, -, *, etc.)
                        let l_tmp = self.new_temp_var(locals);
                        let r_tmp = self.new_temp_var(locals);
                        let l_expr = Box::new(Expr::EVar(l_tmp.clone()));
                        let r_expr = Box::new(Expr::EVar(r_tmp.clone()));
                        let final_expr = self.convert_simple_binop(op, l_expr, r_expr);

                        let assign_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target, final_expr),
                            next_vertex,
                        ));
                        let r_vertex =
                            self.compile_expr_to_value(locals, r, Lhs::Var(r_tmp), assign_vertex);
                        let l_vertex =
                            self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), r_vertex);
                        l_vertex
                    }
                }
            }

            TypedExprKind::Not(e) => {
                let e_tmp = self.new_temp_var(locals);
                let final_expr = Expr::ENot(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::Negate(e) => {
                let e_tmp = self.new_temp_var(locals);
                let final_expr =
                    Expr::EMinus(Box::new(Expr::EInt(0)), Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::MapLit(pairs) => {
                let (key_tmps, val_tmps, simple_pairs) =
                    self.compile_temp_pairs(locals, pairs.len());
                let final_expr = Expr::EMap(simple_pairs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));

                let keys_entry_vertex = self.compile_expr_list_recursive(
                    locals,
                    pairs.iter().map(|(k, _v)| k),
                    key_tmps,
                    assign_vertex,
                );

                self.compile_expr_list_recursive(
                    locals,
                    pairs.iter().map(|(_k, v)| v),
                    val_tmps,
                    keys_entry_vertex,
                )
            }

            TypedExprKind::ListLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(locals, items.len());
                let final_expr = Expr::EList(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(locals, items.iter(), tmps, assign_vertex)
            }

            TypedExprKind::TupleLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(locals, items.len());
                let final_expr = Expr::ETuple(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(locals, items.iter(), tmps, assign_vertex)
            }

            TypedExprKind::Append(l, i) => {
                let l_tmp = self.new_temp_var(locals);
                let i_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EListAppend(
                    Box::new(Expr::EVar(l_tmp.clone())),
                    Box::new(Expr::EVar(i_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let i_vertex =
                    self.compile_expr_to_value(locals, i, Lhs::Var(i_tmp), assign_vertex);
                self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), i_vertex)
            }

            TypedExprKind::Prepend(i, l) => {
                let i_tmp = self.new_temp_var(locals);
                let l_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EListPrepend(
                    Box::new(Expr::EVar(i_tmp.clone())),
                    Box::new(Expr::EVar(l_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let l_vertex =
                    self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), assign_vertex);
                self.compile_expr_to_value(locals, i, Lhs::Var(i_tmp), l_vertex)
            }

            TypedExprKind::Len(e) => {
                let e_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EListLen(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::Min(l, r) => {
                let l_tmp = self.new_temp_var(locals);
                let r_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EMin(
                    Box::new(Expr::EVar(l_tmp.clone())),
                    Box::new(Expr::EVar(r_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let r_vertex =
                    self.compile_expr_to_value(locals, r, Lhs::Var(r_tmp), assign_vertex);
                self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), r_vertex)
            }

            TypedExprKind::Exists(map, key) => {
                let map_tmp = self.new_temp_var(locals);
                let key_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EKeyExists(
                    Box::new(Expr::EVar(key_tmp.clone())),
                    Box::new(Expr::EVar(map_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let key_vertex =
                    self.compile_expr_to_value(locals, key, Lhs::Var(key_tmp), assign_vertex);
                self.compile_expr_to_value(locals, map, Lhs::Var(map_tmp), key_vertex)
            }

            TypedExprKind::Erase(map, key) => {
                let map_tmp = self.new_temp_var(locals);
                let key_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EMapErase(
                    Box::new(Expr::EVar(key_tmp.clone())),
                    Box::new(Expr::EVar(map_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let key_vertex =
                    self.compile_expr_to_value(locals, key, Lhs::Var(key_tmp), assign_vertex);
                self.compile_expr_to_value(locals, map, Lhs::Var(map_tmp), key_vertex)
            }

            TypedExprKind::Head(l) => {
                let l_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EListAccess(Box::new(Expr::EVar(l_tmp.clone())), 0);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), assign_vertex)
            }

            TypedExprKind::Tail(l) => {
                let l_tmp = self.new_temp_var(locals);
                let list_expr = Expr::EVar(l_tmp.clone());
                let final_expr = Expr::EListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::EInt(1)),
                    Box::new(Expr::EListLen(Box::new(list_expr))),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, l, Lhs::Var(l_tmp), assign_vertex)
            }

            TypedExprKind::Index(target_expr, index_expr) => {
                let target_tmp = self.new_temp_var(locals);
                let index_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EFind(
                    Box::new(Expr::EVar(target_tmp.clone())),
                    Box::new(Expr::EVar(index_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let index_vertex = self.compile_expr_to_value(
                    locals,
                    index_expr,
                    Lhs::Var(index_tmp),
                    assign_vertex,
                );
                self.compile_expr_to_value(locals, target_expr, Lhs::Var(target_tmp), index_vertex)
            }

            TypedExprKind::Slice(target_expr, start_expr, end_expr) => {
                let target_tmp = self.new_temp_var(locals);
                let start_tmp = self.new_temp_var(locals);
                let end_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EListSubsequence(
                    Box::new(Expr::EVar(target_tmp.clone())),
                    Box::new(Expr::EVar(start_tmp.clone())),
                    Box::new(Expr::EVar(end_tmp.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let end_vertex =
                    self.compile_expr_to_value(locals, end_expr, Lhs::Var(end_tmp), assign_vertex);
                let start_vertex =
                    self.compile_expr_to_value(locals, start_expr, Lhs::Var(start_tmp), end_vertex);
                self.compile_expr_to_value(locals, target_expr, Lhs::Var(target_tmp), start_vertex)
            }

            TypedExprKind::TupleAccess(tuple_expr, index) => {
                let tuple_tmp = self.new_temp_var(locals);
                let final_expr =
                    Expr::ETupleAccess(Box::new(Expr::EVar(tuple_tmp.clone())), *index);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, tuple_expr, Lhs::Var(tuple_tmp), assign_vertex)
            }

            TypedExprKind::FieldAccess(target_expr, field) => {
                let target_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EFind(
                    Box::new(Expr::EVar(target_tmp.clone())),
                    Box::new(Expr::EString(field.clone())),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, target_expr, Lhs::Var(target_tmp), assign_vertex)
            }

            TypedExprKind::UnwrapOptional(e) => {
                let e_tmp = self.new_temp_var(locals);
                let final_expr = Expr::EUnwrap(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::WrapInOptional(e) => {
                let e_tmp = self.new_temp_var(locals);
                let final_expr = Expr::ESome(Box::new(Expr::EVar(e_tmp.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(locals, e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::StructLit(_, fields) => {
                let (field_names, val_exprs): (Vec<_>, Vec<_>) = fields.iter().cloned().unzip();
                let (tmps, simple_exprs) = self.compile_temp_list(locals, val_exprs.len());
                let final_pairs = field_names
                    .into_iter()
                    .map(Expr::EString)
                    .zip(simple_exprs)
                    .collect();
                let final_expr = Expr::EMap(final_pairs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(locals, val_exprs.iter(), tmps, assign_vertex)
            }

            TypedExprKind::ResolvePromise(p, v) => {
                let p_tmp = self.new_temp_var(locals);
                let v_tmp = self.new_temp_var(locals);

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
                let v_vertex =
                    self.compile_expr_to_value(locals, v, Lhs::Var(v_tmp), resolve_vertex);
                self.compile_expr_to_value(locals, p, Lhs::Var(p_tmp), v_vertex)
            }

            // --- Truly simple, non-side-effecting cases ---
            TypedExprKind::CreateFuture(e) => {
                // create_future(p) is just a copy of p.
                self.compile_expr_to_value(locals, e, target, next_vertex)
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
        locals: &mut Vec<(String, Expr)>,
        node_expr: Expr, // The node to call the function on
        call: &TypedUserFuncCall,
        target: Lhs, // The final destination for the future
        next_vertex: Vertex,
    ) -> Vertex {
        // Allocate all temporary variable names up front.
        // Create temp var names for each argument.
        let (arg_temp_vars, arg_exprs): (Vec<String>, Vec<Expr>) = (0..call.args.len())
            .map(|_| {
                let tmp = self.new_temp_var(locals);
                (tmp.clone(), Expr::EVar(tmp))
            })
            .unzip();

        let qualifier = self
            .func_qualifier_map
            .get(&call.name)
            .expect("Function qualifier should exist in map");
        let func_name = format!("{}.{}", qualifier, call.original_name);

        // Build the call chain backwards (Async -> next_vertex).
        let async_label = Label::Instr(
            Instr::Async(
                target, node_expr, // Use the provided node expression
                func_name, arg_exprs,
            ),
            next_vertex,
        );
        let async_vertex = self.add_label(async_label);

        let mut entry_vertex = async_vertex;

        // Build the argument compilation chain backwards.
        for (arg, tmp_var) in call.args.iter().rev().zip(arg_temp_vars.iter().rev()) {
            entry_vertex =
                self.compile_expr_to_value(locals, arg, Lhs::Var(tmp_var.clone()), entry_vertex);
        }

        // This is the entry point for the "call" part
        entry_vertex
    }

    fn compile_func_call(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
        call: &TypedFuncCall,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        match call {
            TypedFuncCall::User(user_call) => {
                let is_sync = self
                    .func_sync_map
                    .get(&user_call.name)
                    .cloned()
                    .unwrap_or(false);

                if is_sync {
                    // Allocate temp vars for all arguments
                    let (arg_tmps, arg_exprs): (Vec<String>, Vec<Expr>) = (0..user_call.args.len())
                        .map(|_| {
                            let tmp = self.new_temp_var(locals);
                            (tmp.clone(), Expr::EVar(tmp))
                        })
                        .unzip();

                    let qualifier = self
                        .func_qualifier_map
                        .get(&user_call.name)
                        .expect("Function qualifier should exist in map");
                    let func_name = format!("{}.{}", qualifier, user_call.original_name);

                    // Create the SyncCall instruction.
                    let sync_call_label = Label::Instr(
                        Instr::SyncCall(
                            target, // The final Lhs for the return value
                            func_name, arg_exprs,
                        ),
                        next_vertex,
                    );
                    let sync_call_vertex = self.add_label(sync_call_label);

                    // Compile all argument expressions, chaining them backwards.
                    self.compile_expr_list_recursive(
                        locals,
                        user_call.args.iter(),
                        arg_tmps,
                        sync_call_vertex,
                    )
                } else {
                    let node_expr = Expr::EVar("self".to_string());
                    self.compile_async_call_internal(
                        locals,
                        node_expr,
                        user_call,
                        target,
                        next_vertex,
                    )
                }
            }
            TypedFuncCall::Builtin(builtin, args, _return_ty) => {
                // --- This is the new logic for built-ins ---
                self.compile_builtin_call(locals, *builtin, args, target, next_vertex)
            }
        }
    }

    fn compile_builtin_call(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
        builtin: BuiltinFn,
        args: &Vec<TypedExpr>,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        match builtin {
            BuiltinFn::Println => {
                let assign_vertex = self.add_label(Label::Instr(
                    Instr::Assign(target, Expr::EUnit),
                    next_vertex,
                ));

                let arg_var = self.new_temp_var(locals);
                let print_label = Label::Print(Expr::EVar(arg_var.clone()), assign_vertex);
                let print_vertex = self.add_label(print_label);

                let arg_expr = args.get(0).expect("Println should have 1 arg");
                self.compile_expr_to_value(locals, arg_expr, Lhs::Var(arg_var), print_vertex)
            }
            BuiltinFn::IntToString => {
                let arg_var = self.new_temp_var(locals);
                let final_expr = Expr::EIntToString(Box::new(Expr::EVar(arg_var.clone())));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));

                let arg_expr = args.get(0).expect("IntToString should have 1 arg");
                self.compile_expr_to_value(locals, arg_expr, Lhs::Var(arg_var), assign_vertex)
            }
        }
    }

    fn compile_rpc_call(
        &mut self,
        locals: &mut Vec<(String, Expr)>,
        target_expr: &TypedExpr,
        call: &TypedUserFuncCall,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        //  Allocate a temp var to hold the target role
        let target_var = self.new_temp_var(locals);
        let node_expr = Expr::EVar(target_var.clone());

        // Get the entry vertex for the internal call logic
        let internal_call_vertex =
            self.compile_async_call_internal(locals, node_expr, call, target, next_vertex);

        // Compile the target_expr, which runs *before* the internal call
        // The result is stored in `target_var`, which `internal_call_vertex` will use.
        self.compile_expr_to_value(
            locals,
            target_expr,
            Lhs::Var(target_var),
            internal_call_vertex,
        )
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
            TypedExprKind::Store(collection, key, value) => Expr::EStore(
                Box::new(self.convert_simple_expr(collection)),
                Box::new(self.convert_simple_expr(key)),
                Box::new(self.convert_simple_expr(value)),
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

            TypedExprKind::WrapInOptional(e) => Expr::ESome(Box::new(self.convert_simple_expr(e))),
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
            | TypedExprKind::SpinAwait(_)
            | TypedExprKind::CreatePromise
            | TypedExprKind::ResolvePromise(_, _) => {
                panic!(
                    "Cannot have complex call inside a simple expression: {:?}",
                    expr
                )
            }
        }
    }

    fn convert_lhs(&mut self, expr: &TypedExpr) -> Lhs {
        match &expr.kind {
            TypedExprKind::Var(id, name) => Lhs::Var(resolved_name(*id, name)),
            _ => panic!("Invalid assignment target: {:?}", expr),
        }
    }

    fn convert_pattern(&mut self, locals: &mut Vec<(String, Expr)>, pat: &TypedPattern) -> Lhs {
        match &pat.kind {
            TypedPatternKind::Var(id, name) => Lhs::Var(resolved_name(*id, name)),
            TypedPatternKind::Wildcard => Lhs::Var(self.new_temp_var(locals)),
            TypedPatternKind::Tuple(pats) => {
                let names = pats
                    .iter()
                    .map(|p| {
                        match &p.kind {
                            TypedPatternKind::Var(id, name) => resolved_name(*id, name),
                            _ => self.new_temp_var(locals), // Use dummy var for wildcard/unit
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
