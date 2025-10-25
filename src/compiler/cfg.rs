use crate::analysis::resolver::{
    NameId, ResolvedCondStmts, ResolvedExpr, ResolvedExprKind, ResolvedForInLoop, ResolvedForLoop,
    ResolvedForLoopInit, ResolvedFuncCall, ResolvedFuncDef, ResolvedPattern, ResolvedPatternKind,
    ResolvedProgram, ResolvedStatement, ResolvedStatementKind, ResolvedTopLevelDef,
    ResolvedVarInit,
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
}

fn resolved_name(name_id: NameId, original_name: &str) -> String {
    format!("{}_{}", name_id.0, original_name)
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            cfg: Vec::new(),
            temp_counter: 0,
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
    pub fn compile_program(mut self, program: ResolvedProgram) -> Program {
        let mut rpc_map = HashMap::new();
        let mut client_ops_map = HashMap::new();

        // Compile all top-level definitions
        for def in program.top_level_defs {
            match def {
                ResolvedTopLevelDef::Role(role) => {
                    // Compile role's var_inits into a special init function
                    let init_fn = self.compile_init_func(
                        &role.var_inits,
                        format!(
                            "{}_BASE_NODE_INIT",
                            resolved_name(role.name, &role.original_name)
                        ),
                    );
                    rpc_map.insert(init_fn.name.clone(), init_fn);

                    // Compile all other functions in the role
                    for func in role.func_defs {
                        let func_info = self.compile_func_def(func);
                        rpc_map.insert(func_info.name.clone(), func_info);
                    }
                }
                ResolvedTopLevelDef::Type(_) => {
                    // Type definitions have no CFG representation
                }
            }
        }

        // Compile client definition
        let client_init_fn = self.compile_init_func(
            &program.client_def.var_inits,
            "BASE_CLIENT_INIT".to_string(),
        );
        client_ops_map.insert(client_init_fn.name.clone(), client_init_fn);
        for func in program.client_def.func_defs {
            let func_info = self.compile_func_def(func);
            client_ops_map.insert(func_info.name.clone(), func_info);
        }

        Program {
            cfg: self.cfg,
            rpc: rpc_map,
            client_ops: client_ops_map,
        }
    }

    fn compile_init_func(&mut self, inits: &[ResolvedVarInit], name: String) -> FunctionInfo {
        // All init functions end with `Return(EUnit)`
        let final_vertex = self.add_label(Label::Return(Expr::EUnit));

        let mut next_vertex = final_vertex;
        // Build the init chain backwards
        for init in inits.iter().rev() {
            let var_name = init.original_name.clone();
            next_vertex = self.compile_expr_to_value(&init.value, Lhs::Var(var_name), next_vertex);
        }

        let entry = next_vertex;
        FunctionInfo {
            entry,
            name,
            formals: vec![],
            locals: vec![], // Inits are treated as "globals" for the node
        }
    }

    fn compile_func_def(&mut self, func: ResolvedFuncDef) -> FunctionInfo {
        // Add an implicit `Return(EUnit)` at the end of every function,
        // just in case it doesn't have an explicit return.
        let final_vertex = self.add_label(Label::Return(Expr::EUnit));

        let entry = self.compile_block(
            &func.body,
            final_vertex,
            final_vertex, // `break_target` (breaks go to end of function)
        );

        let formals = func.params.into_iter().map(|p| p.original_name).collect();

        let locals = self.scan_body(&func.body);

        FunctionInfo {
            entry,
            name: func.original_name,
            formals,
            locals,
        }
    }

    fn compile_block(
        &mut self,
        body: &[ResolvedStatement],
        next_vertex: Vertex,
        break_target: Vertex,
    ) -> Vertex {
        let mut next = next_vertex;
        // Iterate in reverse to chain statements:
        // StmtN -> next_vertex
        // StmtN-1 -> StmtN
        // ...
        // Stmt0 -> Stmt1
        for stmt in body.iter().rev() {
            next = self.compile_statement(stmt, next, break_target);
        }
        next
    }

    fn compile_statement(
        &mut self,
        stmt: &ResolvedStatement,
        next_vertex: Vertex,
        break_target: Vertex,
    ) -> Vertex {
        match &stmt.kind {
            ResolvedStatementKind::VarInit(init) => {
                // Compile the value, assigning to the new variable
                self.compile_expr_to_value(
                    &init.value,
                    Lhs::Var(init.original_name.clone()),
                    next_vertex,
                )
            }
            ResolvedStatementKind::Assignment(assign) => {
                // Compile the value, assigning to the target LHS
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, next_vertex)
            }
            ResolvedStatementKind::Expr(expr) => {
                // Compile the expression, but discard the result into a dummy var
                let dummy_var = self.new_temp_var();
                self.compile_expr_to_value(expr, Lhs::Var(dummy_var), next_vertex)
            }
            ResolvedStatementKind::Return(expr) => {
                let ret_var = self.new_temp_var();
                let return_label = Label::Return(Expr::EVar(ret_var.clone()));
                let return_vertex = self.add_label(return_label);
                self.compile_expr_to_value(expr, Lhs::Var(ret_var), return_vertex)
            }
            ResolvedStatementKind::Print(expr) => {
                let print_var = self.new_temp_var();
                let print_label = Label::Print(Expr::EVar(print_var.clone()), next_vertex);
                let print_vertex = self.add_label(print_label);
                self.compile_expr_to_value(expr, Lhs::Var(print_var), print_vertex)
            }
            ResolvedStatementKind::ForLoop(loop_stmt) => {
                self.compile_for_loop(loop_stmt, next_vertex)
            }
            ResolvedStatementKind::Conditional(cond) => {
                self.compile_conditional(cond, next_vertex, break_target)
            }
            ResolvedStatementKind::ForInLoop(loop_stmt) => {
                // `next_vertex` is the break target for a loop
                self.compile_for_in_loop(loop_stmt, next_vertex)
            }
            ResolvedStatementKind::Await(expr) => {
                let future_var = self.new_temp_var();
                let await_label = Label::Await(
                    Lhs::Var(future_var.clone()),
                    Expr::EVar(future_var.clone()),
                    next_vertex,
                );
                let await_vertex = self.add_label(await_label);
                self.compile_expr_to_value(expr, Lhs::Var(future_var), await_vertex)
            }
            ResolvedStatementKind::Break => self.add_label(Label::Break(break_target)),
        }
    }

    fn compile_for_loop(&mut self, loop_stmt: &ResolvedForLoop, next_vertex: Vertex) -> Vertex {
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
        );

        // Now we know `body_vertex`, so we can create the *real* condition label.
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
            Some(ResolvedForLoopInit::VarInit(vi)) => self.compile_expr_to_value(
                &vi.value,
                Lhs::Var(vi.original_name.clone()),
                cond_vertex, // [FIX 4] This also correctly points to cond_vertex.
            ),
            Some(ResolvedForLoopInit::Assignment(assign)) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, cond_vertex)
            }
            None => cond_vertex,
        };

        init_vertex
    }

    fn compile_conditional(
        &mut self,
        cond: &ResolvedCondStmts,
        next_vertex: Vertex,
        break_target: Vertex,
    ) -> Vertex {
        let else_vertex = cond.else_branch.as_ref().map_or(next_vertex, |body| {
            self.compile_block(body, next_vertex, break_target)
        });

        let mut next_cond_vertex = else_vertex;
        for branch in cond.elseif_branches.iter().rev() {
            let body_vertex = self.compile_block(&branch.body, next_vertex, break_target);
            let cond_expr = self.convert_simple_expr(&branch.condition);
            next_cond_vertex =
                self.add_label(Label::Cond(cond_expr, body_vertex, next_cond_vertex));
        }

        let if_body_vertex = self.compile_block(&cond.if_branch.body, next_vertex, break_target);
        let if_cond_expr = self.convert_simple_expr(&cond.if_branch.condition);

        self.add_label(Label::Cond(if_cond_expr, if_body_vertex, next_cond_vertex))
    }

    fn compile_for_in_loop(
        &mut self,
        loop_stmt: &ResolvedForInLoop,
        next_vertex: Vertex,
    ) -> Vertex {
        let for_vertex = self.cfg.len();

        // Compile the body, which loops back to `for_vertex`.
        let body_vertex = self.compile_block(
            &loop_stmt.body,
            for_vertex,  // Loop back to self
            next_vertex, // Break target
        );

        let lhs = self.convert_pattern(&loop_stmt.pattern);

        let iterable_copy_var = "_local_copy".to_string();
        let label = Label::ForLoopIn(
            lhs,
            Expr::EVar(iterable_copy_var.clone()),
            body_vertex, // Go here on iteration
            next_vertex, // Go here when done
        );
        // This assertion just ensures our vertex indices are correct.
        assert_eq!(self.add_label(label), for_vertex);

        let iterable_expr = self.convert_simple_expr(&loop_stmt.iterable);
        self.add_label(Label::Instr(
            Instr::Copy(Lhs::Var(iterable_copy_var), iterable_expr),
            for_vertex,
        ))
    }

    /// Compiles any expression into CFG labels, storing the
    /// result in `target`.
    /// Returns the entry vertex for this sequence of instructions.
    fn compile_expr_to_value(
        &mut self,
        expr: &ResolvedExpr,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        match &expr.kind {
            // --- Complex, side-effecting cases ---
            ResolvedExprKind::FuncCall(call) => self.compile_func_call(call, target, next_vertex),
            ResolvedExprKind::RpcCall(target_expr, call) => {
                self.compile_rpc_call(target_expr, call, target, next_vertex)
            }
            ResolvedExprKind::RpcAsyncCall(target_expr, call) => {
                self.compile_rpc_async_call(target_expr, call, target, next_vertex)
            }

            // --- Simple, non-side-effecting cases ---
            _ => {
                // Desugar the simple expression
                let sim_expr = self.convert_simple_expr(expr);
                // Create a single `Assign` instruction
                let label = Label::Instr(Instr::Assign(target, sim_expr), next_vertex);
                self.add_label(label)
            }
        }
    }

    fn compile_func_call(
        &mut self,
        call: &ResolvedFuncCall,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        // Allocate all temporary variable names up front.
        let async_ret_var = self.new_temp_var();
        let async_future_var = self.new_temp_var();

        // Create temp var names for each argument. `arg_exprs` will be
        // a list of `EVar`s (e.g., `_tmp2`, `_tmp3`) that refer to these.
        let (arg_temp_vars, arg_exprs): (Vec<String>, Vec<Expr>) = (0..call.args.len())
            .map(|_| {
                let tmp = self.new_temp_var();
                (tmp.clone(), Expr::EVar(tmp))
            })
            .unzip();

        // Build the call chain backwards (Assign -> Await -> Async -> Pause).
        // Step D (Last): `target = async_ret_var;`
        let assign_label = Label::Instr(
            Instr::Assign(target, Expr::EVar(async_ret_var.clone())),
            next_vertex, // After assignment, go to the original next_vertex
        );
        let assign_vertex = self.add_label(assign_label);

        // Step C: `async_ret_var = await async_future_var;`
        let await_label = Label::Await(
            Lhs::Var(async_ret_var),              // Store result in `async_ret_var`
            Expr::EVar(async_future_var.clone()), // Await the future
            assign_vertex,                        // Go to step D
        );
        let await_vertex = self.add_label(await_label);

        // Step B: `async_future_var = async self.func(arg_exprs);`
        // This performs the async call to the function on the same node.
        let async_label = Label::Instr(
            Instr::Async(
                Lhs::Var(async_future_var),     // Store the new future here
                Expr::EVar("self".to_string()), // Target node is "self"
                resolved_name(call.name, &call.original_name),
                arg_exprs,
            ),
            await_vertex, // Go to step C
        );
        let async_vertex = self.add_label(async_label);

        // Step A: `pause;`
        // This yields to the scheduler before the async call.
        let pause_vertex = self.add_label(Label::Pause(async_vertex)); // Go to step B

        // 3. Build the argument compilation chain backwards.
        let mut entry_vertex = pause_vertex;
        for (arg, tmp_var) in call.args.iter().rev().zip(arg_temp_vars.iter().rev()) {
            entry_vertex = self.compile_expr_to_value(arg, Lhs::Var(tmp_var.clone()), entry_vertex);
        }

        entry_vertex
    }

    fn compile_rpc_call(
        &mut self,
        target_expr: &ResolvedExpr,
        call: &ResolvedFuncCall,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        let async_ret_var = self.new_temp_var();
        let async_future_var = self.new_temp_var();
        let target_var = self.new_temp_var();

        // Create temp var names for each argument.
        let (arg_temp_vars, arg_exprs): (Vec<String>, Vec<Expr>) = (0..call.args.len())
            .map(|_| {
                let tmp = self.new_temp_var();
                (tmp.clone(), Expr::EVar(tmp))
            })
            .unzip();

        // 2. Build the call chain backwards (Assign -> Await -> Async -> Pause).

        let assign_label = Label::Instr(
            Instr::Assign(target, Expr::EVar(async_ret_var.clone())),
            next_vertex,
        );
        let assign_vertex = self.add_label(assign_label);

        let await_label = Label::Await(
            Lhs::Var(async_ret_var),
            Expr::EVar(async_future_var.clone()), // Await the future
            assign_vertex,
        );
        let await_vertex = self.add_label(await_label);

        let async_label = Label::Instr(
            Instr::Async(
                Lhs::Var(async_future_var), // Store the new future here
                Expr::EVar(target_var.clone()),
                resolved_name(call.name, &call.original_name),
                arg_exprs,
            ),
            await_vertex,
        );
        let async_vertex = self.add_label(async_label);

        let pause_vertex = self.add_label(Label::Pause(async_vertex));

        // 3. Build the argument compilation chain backwards.
        let mut arg_entry_vertex = pause_vertex;
        for (arg, tmp_var) in call.args.iter().rev().zip(arg_temp_vars.iter().rev()) {
            arg_entry_vertex =
                self.compile_expr_to_value(arg, Lhs::Var(tmp_var.clone()), arg_entry_vertex);
        }

        // 4. Build the target expression compilation.
        let target_entry_vertex =
            self.compile_expr_to_value(target_expr, Lhs::Var(target_var), arg_entry_vertex);

        target_entry_vertex
    }

    fn compile_rpc_async_call(
        &mut self,
        target_expr: &ResolvedExpr,
        call: &ResolvedFuncCall,
        target: Lhs, // The final destination for the *future*
        next_vertex: Vertex,
    ) -> Vertex {
        // 1. Allocate all temporary variable names up front.
        let target_var = self.new_temp_var(); // To store the result of `target_expr`

        // Create temp var names for each argument.
        let (arg_temp_vars, arg_exprs): (Vec<String>, Vec<Expr>) = (0..call.args.len())
            .map(|_| {
                let tmp = self.new_temp_var();
                (tmp.clone(), Expr::EVar(tmp))
            })
            .unzip();

        // 2. Build the call chain backwards (Async -> Pause).
        let async_label = Label::Instr(
            Instr::Async(
                target,
                Expr::EVar(target_var.clone()),
                resolved_name(call.name, &call.original_name),
                arg_exprs,
            ),
            next_vertex,
        );
        let async_vertex = self.add_label(async_label);

        let pause_vertex = self.add_label(Label::Pause(async_vertex)); // Go to step B

        // 3. Build the argument compilation chain backwards.
        let mut arg_entry_vertex = pause_vertex;
        for (arg, tmp_var) in call.args.iter().rev().zip(arg_temp_vars.iter().rev()) {
            arg_entry_vertex =
                self.compile_expr_to_value(arg, Lhs::Var(tmp_var.clone()), arg_entry_vertex);
        }

        // 4. Build the target expression compilation.
        let target_entry_vertex =
            self.compile_expr_to_value(target_expr, Lhs::Var(target_var), arg_entry_vertex);
        target_entry_vertex
    }

    /// Converts a "simple" `ResolvedExpr` to a `cfg::Expr`.
    /// Panics if it encounters a complex (side-effecting) expression.
    fn convert_simple_expr(&mut self, expr: &ResolvedExpr) -> Expr {
        match &expr.kind {
            ResolvedExprKind::Var(id, name) => Expr::EVar(resolved_name(*id, &name)),
            ResolvedExprKind::IntLit(i) => Expr::EInt(i.clone()),
            ResolvedExprKind::StringLit(s) => Expr::EString(s.clone()),
            ResolvedExprKind::BoolLit(b) => Expr::EBool(b.clone()),
            ResolvedExprKind::NilLit => Expr::ENil,
            ResolvedExprKind::BinOp(op, l, r) => {
                let left = Box::new(self.convert_simple_expr(l));
                let right = Box::new(self.convert_simple_expr(r));
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
            ResolvedExprKind::Not(e) => Expr::ENot(Box::new(self.convert_simple_expr(e))),
            ResolvedExprKind::Negate(e) => Expr::EMinus(
                Box::new(Expr::EInt(0)),
                Box::new(self.convert_simple_expr(e)),
            ),
            ResolvedExprKind::MapLit(pairs) => Expr::EMap(
                pairs
                    .iter()
                    .map(|(k, v)| (self.convert_simple_expr(k), self.convert_simple_expr(v)))
                    .collect(),
            ),
            ResolvedExprKind::ListLit(items) => {
                Expr::EList(items.iter().map(|e| self.convert_simple_expr(e)).collect())
            }
            ResolvedExprKind::TupleLit(items) => {
                Expr::ETuple(items.iter().map(|e| self.convert_simple_expr(e)).collect())
            }
            // Desugar: `append(l, i)` -> `EListAppend(l, i)`
            ResolvedExprKind::Append(l, i) => Expr::EListAppend(
                Box::new(self.convert_simple_expr(l)),
                Box::new(self.convert_simple_expr(i)),
            ),
            ResolvedExprKind::Prepend(i, l) => Expr::EListPrepend(
                Box::new(self.convert_simple_expr(i)),
                Box::new(self.convert_simple_expr(l)),
            ),
            // Desugar: `len(e)` -> `EListLen(e)`
            ResolvedExprKind::Len(e) => Expr::EListLen(Box::new(self.convert_simple_expr(e))),
            ResolvedExprKind::Min(l, r) => Expr::EMin(
                Box::new(self.convert_simple_expr(l)),
                Box::new(self.convert_simple_expr(r)),
            ),
            ResolvedExprKind::Exists(map, key) => Expr::EKeyExists(
                Box::new(self.convert_simple_expr(key)),
                Box::new(self.convert_simple_expr(map)),
            ),
            // Desugar: `head(l)` -> `EListAccess(l, 0)`
            ResolvedExprKind::Head(l) => {
                Expr::EListAccess(Box::new(self.convert_simple_expr(l)), 0)
            }
            // Desugar: `tail(l)` -> `EListSubsequence(l, 1, len(l))`
            ResolvedExprKind::Tail(l) => {
                let list_expr = self.convert_simple_expr(l);
                Expr::EListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::EInt(1)),
                    Box::new(Expr::EListLen(Box::new(list_expr))),
                )
            }
            // Desugar: `s[i]` -> `EFind(s, i)`
            ResolvedExprKind::Index(target, index) => Expr::EFind(
                Box::new(self.convert_simple_expr(target)),
                Box::new(self.convert_simple_expr(index)),
            ),
            // Desugar: `s[i:j]` -> `EListSubsequence(s, i, j)`
            ResolvedExprKind::Slice(target, start, end) => Expr::EListSubsequence(
                Box::new(self.convert_simple_expr(target)),
                Box::new(self.convert_simple_expr(start)),
                Box::new(self.convert_simple_expr(end)),
            ),
            ResolvedExprKind::TupleAccess(tuple, index) => {
                Expr::ETupleAccess(Box::new(self.convert_simple_expr(tuple)), *index)
            }
            // Desugar: `s.field` -> `EFind(s, "field")`
            ResolvedExprKind::FieldAccess(target, field) => Expr::EFind(
                Box::new(self.convert_simple_expr(target)),
                Box::new(Expr::EString(field.clone())),
            ),
            ResolvedExprKind::Unwrap(e) => Expr::EUnwrap(Box::new(self.convert_simple_expr(e))),
            // Desugar: `MyStruct { ... }` -> `EMap { ... }`
            ResolvedExprKind::StructLit(_, fields) => Expr::EMap(
                fields
                    .iter()
                    .map(|(name, val)| (Expr::EString(name.clone()), self.convert_simple_expr(val)))
                    .collect(),
            ),

            // --- Panic on complex expressions ---
            ResolvedExprKind::FuncCall(_)
            | ResolvedExprKind::RpcCall(_, _)
            | ResolvedExprKind::RpcAsyncCall(_, _) => {
                panic!(
                    "Cannot have complex call inside a simple expression: {:?}",
                    expr
                )
            }
            ResolvedExprKind::PollForResps(e1, e2) => Expr::EPollForResps(
                Box::new(self.convert_simple_expr(e1)),
                Box::new(self.convert_simple_expr(e2)),
            ),
            ResolvedExprKind::PollForAnyResp(e) => {
                Expr::EPollForAnyResp(Box::new(self.convert_simple_expr(e)))
            }
            ResolvedExprKind::NextResp(e) => Expr::ENextResp(Box::new(self.convert_simple_expr(e))),
        }
    }

    fn convert_lhs(&mut self, expr: &ResolvedExpr) -> Lhs {
        match &expr.kind {
            ResolvedExprKind::Var(id, name) => Lhs::Var(resolved_name(*id, name)),
            // Desugar: `s[i] = ...` -> `LAccess(s, i)`
            ResolvedExprKind::Index(target, index) => Lhs::Access(
                self.convert_simple_expr(target),
                self.convert_simple_expr(index),
            ),
            // Desugar: `s.field = ...` -> `LAccess(s, "field")`
            ResolvedExprKind::FieldAccess(target, field) => Lhs::Access(
                self.convert_simple_expr(target),
                Expr::EString(field.clone()),
            ),
            _ => panic!("Invalid assignment target: {:?}", expr),
        }
    }

    fn convert_pattern(&mut self, pat: &ResolvedPattern) -> Lhs {
        match &pat.kind {
            ResolvedPatternKind::Var(id, name) => Lhs::Var(resolved_name(*id, name)),
            ResolvedPatternKind::Wildcard => Lhs::Var(self.new_temp_var()),
            ResolvedPatternKind::Unit => Lhs::Var(self.new_temp_var()),
            ResolvedPatternKind::Tuple(pats) => {
                let names = pats
                    .iter()
                    .map(|p| {
                        match &p.kind {
                            ResolvedPatternKind::Var(id, name) => resolved_name(*id, name),
                            _ => self.new_temp_var(), // Use dummy var for wildcard/unit
                        }
                    })
                    .collect();
                Lhs::Tuple(names)
            }
        }
    }

    fn scan_body(&mut self, body: &[ResolvedStatement]) -> Vec<(String, Expr)> {
        let mut locals = Vec::new();
        for stmt in body {
            self.scan_stmt_for_locals(stmt, &mut locals);
        }
        locals
    }

    fn scan_body_for_locals(
        &mut self,
        body: &[ResolvedStatement],
        locals: &mut Vec<(String, Expr)>,
    ) {
        for stmt in body {
            self.scan_stmt_for_locals(stmt, locals);
        }
    }

    fn scan_stmt_for_locals(&mut self, stmt: &ResolvedStatement, locals: &mut Vec<(String, Expr)>) {
        match &stmt.kind {
            ResolvedStatementKind::VarInit(init) => {
                // `let x = ...` adds a local
                locals.push((
                    resolved_name(init.name, &init.original_name),
                    self.convert_simple_expr(&init.value),
                ));
            }
            ResolvedStatementKind::Conditional(cond) => {
                self.scan_body_for_locals(&cond.if_branch.body, locals);
                for branch in &cond.elseif_branches {
                    self.scan_body_for_locals(&branch.body, locals);
                }
                if let Some(body) = &cond.else_branch {
                    self.scan_body_for_locals(body, locals);
                }
            }
            ResolvedStatementKind::ForLoop(fl) => {
                // This is the C-style loop, which might have an init
                if let Some(ResolvedForLoopInit::VarInit(init)) = &fl.init {
                    locals.push((
                        resolved_name(init.name, &init.original_name),
                        self.convert_simple_expr(&init.value),
                    ));
                }
                self.scan_body_for_locals(&fl.body, locals);
            }
            ResolvedStatementKind::ForInLoop(loop_stmt) => {
                // `for (x, y) in ...` adds locals
                self.scan_pattern_for_locals(&loop_stmt.pattern, locals);
                self.scan_body_for_locals(&loop_stmt.body, locals);
            }
            // Other statements don't declare locals
            _ => {}
        }
    }

    fn scan_pattern_for_locals(&mut self, pat: &ResolvedPattern, locals: &mut Vec<(String, Expr)>) {
        match &pat.kind {
            ResolvedPatternKind::Var(id, name) => {
                // `ENil` is just a placeholder default
                locals.push((resolved_name(*id, name), Expr::ENil));
            }
            ResolvedPatternKind::Tuple(pats) => {
                for p in pats {
                    self.scan_pattern_for_locals(p, locals);
                }
            }
            _ => {} // Wildcard, Unit don't add named locals
        }
    }
}
