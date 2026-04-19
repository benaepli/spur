mod ir;
pub use ir::*;

use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::type_id::{TypeId, TypeIdMap};
use crate::compiler::lowered::{
    LBlock, LCondExpr, LExpr, LExprKind, LForInLoop, LForLoop, LForLoopInit,
    LFuncCall, LFuncDef, LProgram, LStatement, LStatementKind, LTopLevelDef,
    LUserFuncCall, LVarInit,
};
use crate::parser::Span;
use ecow::EcoString;
use std::collections::HashMap;

/// Bundles the control-flow targets that are threaded through every
/// compilation call (break, continue, return, return slot).
#[derive(Clone, Copy)]
struct CompileCtx {
    break_target: Vertex,
    continue_target: Vertex,
    return_target: Vertex,
    return_slot: VarSlot,
}

pub struct Compiler {
    /// The CFG being built. New labels are pushed here.
    cfg: Vec<Label>,

    rpc_map: HashMap<NameId, FunctionInfo>,

    func_sync_map: HashMap<NameId, bool>,
    func_traced_map: HashMap<NameId, bool>,
    func_qualifier_map: HashMap<NameId, String>,

    /// Mapping from qualified function name strings to NameId
    func_name_to_id: HashMap<String, NameId>,

    /// Mapping from NameId to original name (for debugging/display)
    id_to_name: HashMap<NameId, String>,

    /// Counter for generating new NameIds for temp variables
    next_name_id: usize,

    /// Tracks the span of the statement currently being compiled
    current_span: Option<Span>,
    /// Maps each vertex to its source statement span
    vertex_to_span: HashMap<Vertex, Span>,

    /// Current function's local variable slot assignments
    local_slots: HashMap<NameId, u32>,
    next_local_slot: u32,

    /// Node-level (role instance) variable slot assignments
    node_slots: HashMap<NameId, u32>,
    next_node_slot: u32,

    /// Debug names for current function's slots
    current_slot_names: Vec<String>,

    /// Default values for current function's local slots
    current_local_defaults: Vec<Expr>,

    /// Maximum number of node-level slots encountered so far
    max_node_slots: u32,

    /// Ordered list of role (NameId, original_name)
    roles: Vec<(NameId, String)>,

    /// Type ID map for looking up TypeIds during compilation
    type_ids: TypeIdMap,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            cfg: Vec::new(),
            rpc_map: HashMap::new(),
            func_sync_map: HashMap::new(),
            func_traced_map: HashMap::new(),
            func_qualifier_map: HashMap::new(),
            func_name_to_id: HashMap::new(),
            id_to_name: HashMap::new(),
            next_name_id: 0,
            current_span: None,
            vertex_to_span: HashMap::new(),
            local_slots: HashMap::new(),
            next_local_slot: 0,
            node_slots: HashMap::new(),
            next_node_slot: 1, // Slot 0 reserved for 'self'
            current_slot_names: Vec::new(),
            current_local_defaults: Vec::new(),
            max_node_slots: 1,
            roles: Vec::new(),
            type_ids: TypeIdMap::new(),
        }
    }

    /// Call when starting to compile a new role
    fn begin_role(&mut self, var_inits: &[LVarInit]) {
        self.node_slots.clear();
        self.next_node_slot = 1; // Slot 0 reserved for 'self'

        for init in var_inits {
            self.node_slots.insert(init.name, self.next_node_slot);
            self.next_node_slot += 1;
        }

        if self.next_node_slot > self.max_node_slots {
            self.max_node_slots = self.next_node_slot;
        }
    }

    /// Call when starting to compile a new function
    fn begin_function(&mut self, params: &[(NameId, String)]) {
        self.local_slots.clear();
        self.next_local_slot = 0;
        self.current_slot_names.clear();
        self.current_local_defaults.clear();

        for (name_id, name_str) in params {
            self.local_slots.insert(*name_id, self.next_local_slot);
            self.current_slot_names.push(name_str.clone());
            self.next_local_slot += 1;
            // Parameters don't need defaults as they are provided by caller
        }
    }

    /// Allocate a local slot for a variable declaration
    fn alloc_local_slot(&mut self, name_id: NameId, name_str: &str, default: Expr) -> u32 {
        let slot = self.next_local_slot;
        self.next_local_slot += 1;
        self.local_slots.insert(name_id, slot);
        self.current_slot_names.push(name_str.to_string());
        self.current_local_defaults.push(default);
        slot
    }

    /// Allocate an anonymous temp slot
    fn alloc_temp_slot(&mut self) -> VarSlot {
        let slot = self.next_local_slot;
        self.next_local_slot += 1;
        let name_str = format!("_tmp{}", slot);
        let name_id = self.alloc_name_id(name_str);
        self.current_slot_names.push(format!("_tmp{}", slot));
        self.current_local_defaults.push(Expr::Unit);
        VarSlot::Local(slot, name_id)
    }

    /// Look up the slot for a resolved variable
    fn resolve_slot(&self, name_id: NameId) -> VarSlot {
        if name_id == SELF_NAME {
            return SELF_SLOT;
        }
        if let Some(&slot) = self.local_slots.get(&name_id) {
            return VarSlot::Local(slot, name_id);
        }
        if let Some(&slot) = self.node_slots.get(&name_id) {
            return VarSlot::Node(slot, name_id);
        }
        let var_name = self
            .id_to_name
            .get(&name_id)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>");
        panic!(
            "Variable '{}' (NameId {:?}) not found in slot maps. \
             This indicates a bug in the resolver or CFG builder.",
            var_name, name_id
        );
    }

    /// Allocates a new NameId and registers it with a display name.
    fn alloc_name_id(&mut self, display_name: String) -> NameId {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        self.id_to_name.insert(id, display_name);
        id
    }

    /// Adds a new label to the CFG and returns its Vertex index.
    /// If current_span is set, associates this vertex with that span.
    fn add_label(&mut self, label: Label) -> Vertex {
        let vertex = self.cfg.len();
        self.cfg.push(label);

        // Associate vertex with current statement span if available
        if let Some(span) = self.current_span {
            self.vertex_to_span.insert(vertex, span);
        }

        vertex
    }

    /// Allocates a new NameId for a function and registers it.
    fn alloc_func_name_id(&mut self, qualified_name: String) -> NameId {
        let id = self.alloc_name_id(qualified_name.clone());
        self.func_name_to_id.insert(qualified_name, id);
        id
    }

    /// The main entry point.
    pub fn compile_program(mut self, program: LProgram, type_ids: TypeIdMap) -> Program {
        let next_name_id = program.next_name_id;
        self.next_name_id = next_name_id;
        // Initialize id_to_name from the resolver's map
        self.id_to_name = program.id_to_name;
        // Store type_ids for use during compilation
        self.type_ids = type_ids;

        // Build func_sync_map and compile all top-level definitions
        for def in &program.top_level_defs {
            match def {
                LTopLevelDef::Role(role) => {
                    let qualifier = role.original_name.clone();
                    self.roles.push((role.name, qualifier.clone()));
                    for func in &role.func_defs {
                        self.func_sync_map.insert(func.name, func.is_sync);
                        self.func_traced_map.insert(func.name, func.is_traced);
                        self.func_qualifier_map.insert(func.name, qualifier.clone());
                    }
                }
                LTopLevelDef::FreeFunc(func) => {
                    self.func_sync_map.insert(func.name, true); // free functions are always sync
                    self.func_traced_map.insert(func.name, func.is_traced);
                    self.func_qualifier_map.insert(func.name, "__free".to_string());
                }
            }
        }

        // Compile all top-level definitions
        for def in program.top_level_defs {
            match def {
                LTopLevelDef::Role(role) => {
                    // Set up node-level slot assignments for this role
                    self.begin_role(&role.var_inits);

                    // Compile role's var_inits into a special init function
                    let init_func_name = format!("{}.{}", role.original_name, "BASE_NODE_INIT");
                    let init_fn = self.compile_init_func(&role.var_inits, init_func_name);
                    self.rpc_map.insert(init_fn.name, init_fn);

                    // Compile all other functions in the role
                    for func in role.func_defs {
                        let func_info = self.compile_func_def(func);
                        self.rpc_map.insert(func_info.name, func_info);
                    }
                }
                LTopLevelDef::FreeFunc(func) => {
                    let func_info = self.compile_func_def(func);
                    self.rpc_map.insert(func_info.name, func_info);
                }
            }
        }

        Program {
            cfg: Cfg { graph: self.cfg },
            rpc: self.rpc_map,
            func_name_to_id: self.func_name_to_id,
            id_to_name: self.id_to_name,
            next_name_id,
            vertex_to_span: self.vertex_to_span,
            max_node_slots: self.max_node_slots,
            roles: self.roles,
            type_ids: self.type_ids,
        }
    }

    fn compile_init_func(
        &mut self,
        inits: &[LVarInit],
        qualified_name: String,
    ) -> FunctionInfo {
        // Begin a new function with no parameters
        self.begin_function(&[]);

        // Allocate a temp slot for the return value
        let return_slot = self.alloc_temp_slot();
        let final_vertex = self.add_label(Label::Return(Expr::Var(return_slot)));

        let ctx = CompileCtx {
            break_target: final_vertex,
            continue_target: final_vertex,
            return_target: final_vertex,
            return_slot,
        };

        let mut next_vertex = final_vertex;
        // Build the init chain backwards
        // These assign to NODE slots, not local slots
        for init in inits.iter().rev() {
            let lhs = Lhs::Var(self.resolve_slot(init.name));
            next_vertex = self.compile_expr_to_value(&init.value, lhs, next_vertex, ctx);
        }

        let entry = self.add_label(Label::Instr(
            Instr::Assign(Lhs::Var(return_slot), Expr::Unit),
            next_vertex,
        ));

        let func_name_id = self.alloc_func_name_id(qualified_name);

        FunctionInfo {
            entry,
            name: func_name_id,
            param_count: 0,
            local_slot_count: self.next_local_slot,
            local_defaults: self.current_local_defaults.clone(),
            is_sync: true,
            debug_slot_names: self.current_slot_names.clone(),
        }
    }

    fn compile_func_def(&mut self, func: LFuncDef) -> FunctionInfo {
        // Collect parameter info: (NameId, display_name)
        let params: Vec<(NameId, String)> = func
            .params
            .iter()
            .map(|p| (p.name, p.original_name.clone()))
            .collect();
        let param_count = params.len() as u32;
        let is_traced = func.is_traced;

        // Begin the function - this assigns slots to parameters
        self.begin_function(&params);

        // Scan the body for all local variable declarations and assign them slots
        self.scan_and_assign_slots(&func.body);

        // Allocate a temp slot for the return value
        let return_slot = self.alloc_temp_slot();
        let final_return_vertex = self.add_label(Label::Return(Expr::Var(return_slot)));

        // Build the qualified name early so we can use it for trace labels
        let qualifier = self.func_qualifier_map.get(&func.name).unwrap_or_else(|| {
            panic!(
                "Function qualifier not found for function '{}'. \
                     This indicates a bug in the CFG builder initialization.",
                func.original_name
            )
        });
        let qualified_name = format!("{}.{}", qualifier, func.original_name);

        // If traced, allocate a hidden slot for the trace invocation ID.
        let trace_id_slot = if is_traced {
            Some(self.alloc_temp_slot())
        } else {
            None
        };

        // If traced, insert TraceExit before the final return and use it as
        // the return_target so all return paths funnel through it.
        let effective_return_target = if is_traced {
            let tid_slot = trace_id_slot.unwrap();
            self.add_label(Label::TraceExit(
                qualified_name.clone(),
                Expr::Var(tid_slot),
                Expr::Var(return_slot),
                final_return_vertex,
            ))
        } else {
            final_return_vertex
        };

        let ctx = CompileCtx {
            break_target: effective_return_target,
            continue_target: effective_return_target,
            return_target: effective_return_target,
            return_slot,
        };

        let body_entry = self.compile_block(&func.body, Lhs::Var(return_slot), effective_return_target, ctx);

        // If traced, insert TraceEnter after the return-slot init and before the body.
        let after_init = if is_traced {
            let tid_slot = trace_id_slot.unwrap();
            let param_exprs: Vec<Expr> = (0..param_count)
                .map(|i| {
                    let name_id = params[i as usize].0;
                    Expr::Var(VarSlot::Local(i, name_id))
                })
                .collect();
            self.add_label(Label::TraceEnter(
                qualified_name.clone(),
                param_exprs,
                Lhs::Var(tid_slot),
                body_entry,
            ))
        } else {
            body_entry
        };

        let entry = self.add_label(Label::Instr(
            Instr::Assign(Lhs::Var(return_slot), Expr::Unit),
            after_init,
        ));

        // Use the existing func.name NameId from the resolver, not a new one
        self.func_name_to_id.insert(qualified_name, func.name);

        FunctionInfo {
            entry,
            name: func.name,
            param_count,
            local_slot_count: self.next_local_slot,
            local_defaults: self.current_local_defaults.clone(),
            is_sync: func.is_sync,
            debug_slot_names: self.current_slot_names.clone(),
        }
    }


    /// Scan the function body for all local variable declarations and assign them slots.
    fn scan_and_assign_slots(&mut self, body: &LBlock) {
        self.scan_block_slots(body);
    }

    fn scan_block_slots(&mut self, block: &LBlock) {
        for stmt in &block.statements {
            self.scan_stmt_slots(stmt);
        }
        if let Some(tail) = &block.tail_expr {
            self.scan_expr_slots(tail);
        }
    }

    fn scan_stmt_slots(&mut self, stmt: &LStatement) {
        match &stmt.kind {
            LStatementKind::VarInit(init) => {
                let name_str = self
                    .id_to_name
                    .get(&init.name)
                    .cloned()
                    .unwrap_or_else(|| format!("var_{}", init.name.0));
                self.alloc_local_slot(init.name, &name_str, Expr::Nil);
                self.scan_expr_slots(&init.value);
            }
            LStatementKind::Assignment(assign) => {
                self.scan_expr_slots(&assign.value);
            }
            LStatementKind::Expr(expr) => {
                self.scan_expr_slots(expr);
            }
            LStatementKind::ForLoop(fl) => {
                if let Some(init) = &fl.init {
                    match init {
                        LForLoopInit::VarInit(vi) => {
                            let name_str = self
                                .id_to_name
                                .get(&vi.name)
                                .cloned()
                                .unwrap_or_else(|| format!("var_{}", vi.name.0));
                            self.alloc_local_slot(vi.name, &name_str, Expr::Nil);
                            self.scan_expr_slots(&vi.value);
                        }
                        LForLoopInit::Assignment(a) => {
                            self.scan_expr_slots(&a.value);
                        }
                    }
                }
                if let Some(cond) = &fl.condition {
                    self.scan_expr_slots(cond);
                }
                for inc_stmt in &fl.increment {
                    self.scan_stmt_slots(inc_stmt);
                }
                for stmt in &fl.body {
                    self.scan_stmt_slots(stmt);
                }
            }
            LStatementKind::ForInLoop(fil) => {
                let name_str = self
                    .id_to_name
                    .get(&fil.binding_name)
                    .cloned()
                    .unwrap_or_else(|| format!("var_{}", fil.binding_name.0));
                self.alloc_local_slot(fil.binding_name, &name_str, Expr::Nil);
                self.scan_expr_slots(&fil.iterable);
                for stmt in &fil.body {
                    self.scan_stmt_slots(stmt);
                }
            }
            LStatementKind::Loop(body) => {
                for stmt in body {
                    self.scan_stmt_slots(stmt);
                }
            }
            LStatementKind::Error => {}
        }
    }

    fn scan_expr_slots(&mut self, expr: &LExpr) {
        use LExprKind::*;
        match &expr.kind {
            BinOp(_, l, r)
            | Append(l, r)
            | Prepend(l, r)
            | Min(l, r)
            | Exists(l, r)
            | Erase(l, r)
            | Send(l, r)
            | Index(l, r)
            | SafeIndex(l, r) => {
                self.scan_expr_slots(l);
                self.scan_expr_slots(r);
            }
            PersistData(e) => {
                self.scan_expr_slots(e);
            }
            RetrieveData(_) => {}
            DiscardData => {}
            Not(e)
            | Negate(e)
            | Head(e)
            | Tail(e)
            | Len(e)
            | UnwrapOptional(e)
            | MakeIter(e)
            | IterIsDone(e)
            | IterNext(e)
            | Recv(e)
            | TupleAccess(e, _)
            | FieldAccess(e, _)
            | SafeFieldAccess(e, _)
            | SafeTupleAccess(e, _)
            | WrapInOptional(e)
            | IsVariant(e, _)
            | VariantPayload(e) => {
                self.scan_expr_slots(e);
            }
            MakeChannel => {}
            FuncCall(call) => match call {
                LFuncCall::User(u) => {
                    for arg in &u.args {
                        self.scan_expr_slots(arg);
                    }
                }
                LFuncCall::Builtin(_, args, _) => {
                    for arg in args {
                        self.scan_expr_slots(arg);
                    }
                }
            },
            MapLit(pairs) => {
                for (k, v) in pairs {
                    self.scan_expr_slots(k);
                    self.scan_expr_slots(v);
                }
            }
            ListLit(exprs) | TupleLit(exprs) => {
                for e in exprs {
                    self.scan_expr_slots(e);
                }
            }
            Slice(a, b, c) | Store(a, b, c) => {
                self.scan_expr_slots(a);
                self.scan_expr_slots(b);
                self.scan_expr_slots(c);
            }
            RpcCall(target, call) => {
                self.scan_expr_slots(target);
                for arg in &call.args {
                    self.scan_expr_slots(arg);
                }
            }
            Conditional(cond) => {
                self.scan_expr_slots(&cond.if_branch.condition);
                self.scan_block_slots(&cond.if_branch.body);
                for branch in &cond.elseif_branches {
                    self.scan_expr_slots(&branch.condition);
                    self.scan_block_slots(&branch.body);
                }
                if let Some(body) = &cond.else_branch {
                    self.scan_block_slots(body);
                }
            }
            Block(block) => {
                self.scan_block_slots(block);
            }
            VariantLit(_, _, payload) => {
                if let Some(p) = payload {
                    self.scan_expr_slots(p);
                }
            }
            StructLit(_, fields) => {
                for (_, e) in fields {
                    self.scan_expr_slots(e);
                }
            }
            Return(inner) => {
                self.scan_expr_slots(inner);
            }
            Break | Continue => {}
            Var(_, _) | IntLit(_) | StringLit(_) | BoolLit(_) | NilLit | SetTimer(_) => {}
            Error => {}
        }
    }


    fn compile_tailless_block(
        &mut self,
        body: &[LStatement],
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let mut next = next_vertex;
        for stmt in body.iter().rev() {
            next = self.compile_statement(stmt, next, ctx);
        }
        next
    }

    fn compile_block(
        &mut self,
        block: &LBlock,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let tail_vertex = if let Some(tail) = &block.tail_expr {
            self.compile_expr_to_value(tail, target, next_vertex, ctx)
        } else {
            self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex))
        };
        self.compile_tailless_block(&block.statements, tail_vertex, ctx)
    }

    fn compile_statement(
        &mut self,
        stmt: &LStatement,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let prev_span = self.current_span;
        self.current_span = Some(stmt.span);

        let result = match &stmt.kind {
            LStatementKind::VarInit(init) => {
                let lhs = Lhs::Var(self.resolve_slot(init.name));
                self.compile_expr_to_value(&init.value, lhs, next_vertex, ctx)
            }
            LStatementKind::Assignment(assign) => {
                let lhs = Lhs::Var(self.resolve_slot(assign.target_id));
                self.compile_expr_to_value(&assign.value, lhs, next_vertex, ctx)
            }
            LStatementKind::Expr(expr) => match &expr.kind {
                LExprKind::Return(inner) => {
                    self.compile_expr_to_value(
                        inner,
                        Lhs::Var(ctx.return_slot),
                        ctx.return_target,
                        ctx,
                    )
                }
                LExprKind::Break => self.add_label(Label::Break(ctx.break_target)),
                LExprKind::Continue => self.add_label(Label::Continue(ctx.continue_target)),
                _ => {
                    let dummy_slot = self.alloc_temp_slot();
                    self.compile_expr_to_value(expr, Lhs::Var(dummy_slot), next_vertex, ctx)
                }
            }
            LStatementKind::ForLoop(for_loop) => {
                self.compile_for_loop(for_loop, next_vertex, ctx)
            }
            LStatementKind::Loop(body) => {
                self.compile_loop(body, next_vertex, ctx)
            }
            LStatementKind::ForInLoop(loop_stmt) => {
                self.compile_for_in_loop(loop_stmt, next_vertex, ctx)
            }
            LStatementKind::Error => {
                unreachable!("Error statements should not reach CFG generation")
            }
        };

        self.current_span = prev_span;
        result
    }

    fn compile_loop(
        &mut self,
        body: &[LStatement],
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        // Allocate a placeholder vertex for the loop head
        let loop_head_vertex = self.add_label(Label::Return(Expr::Unit));

        let loop_ctx = CompileCtx {
            break_target: next_vertex,
            continue_target: loop_head_vertex,
            ..ctx
        };

        let body_vertex = self.compile_tailless_block(body, loop_head_vertex, loop_ctx);

        // Replace the placeholder with a jump to the body
        let dummy_slot = self.alloc_temp_slot();
        self.cfg[loop_head_vertex] = Label::Instr(
            Instr::Assign(Lhs::Var(dummy_slot), Expr::Unit),
            body_vertex,
        );

        // The entry point is the loop head
        loop_head_vertex
    }

    fn compile_conditional(
        &mut self,
        cond: &LCondExpr,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let else_vertex = if let Some(body) = &cond.else_branch {
            self.compile_block(body, target.clone(), next_vertex, ctx)
        } else {
            self.add_label(Label::Instr(Instr::Assign(target.clone(), Expr::Unit), next_vertex))
        };

        let mut next_cond_vertex = else_vertex;
        for branch in cond.elseif_branches.iter().rev() {
            let body_vertex = self.compile_block(&branch.body, target.clone(), next_vertex, ctx);
            let cond_slot = self.alloc_temp_slot();
            let check_vertex = self.add_label(Label::Cond(
                Expr::Var(cond_slot),
                body_vertex,
                next_cond_vertex,
            ));
            next_cond_vertex = self.compile_expr_to_value(
                &branch.condition,
                Lhs::Var(cond_slot),
                check_vertex,
                ctx,
            );
        }

        let if_body_vertex = self.compile_block(&cond.if_branch.body, target.clone(), next_vertex, ctx);
        let if_cond_slot = self.alloc_temp_slot();

        let if_check_vertex = self.add_label(Label::Cond(
            Expr::Var(if_cond_slot),
            if_body_vertex,
            next_cond_vertex,
        ));

        self.compile_expr_to_value(
            &cond.if_branch.condition,
            Lhs::Var(if_cond_slot),
            if_check_vertex,
            ctx,
        )
    }

    fn compile_for_loop(
        &mut self,
        fl: &LForLoop,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        // Placeholder: the stable back-edge target for the loop
        let cond_entry = self.add_label(Label::Return(Expr::Unit));

        // 1. Increment → cond_entry
        let inc_vertex = if fl.increment.is_empty() {
            cond_entry
        } else {
            self.compile_tailless_block(&fl.increment, cond_entry, ctx)
        };

        // 2. Body: continue → inc_vertex, break → next_vertex
        let loop_ctx = CompileCtx {
            break_target: next_vertex,
            continue_target: inc_vertex,
            ..ctx
        };
        let body_vertex = self.compile_tailless_block(&fl.body, inc_vertex, loop_ctx);

        // 3. Condition → branch(body, exit), or unconditional → body
        if let Some(cond) = &fl.condition {
            let cond_slot = self.alloc_temp_slot();
            let branch = self.add_label(Label::Cond(
                Expr::Var(cond_slot),
                body_vertex,
                next_vertex,
            ));
            let eval_entry =
                self.compile_expr_to_value(cond, Lhs::Var(cond_slot), branch, ctx);
            // Trampoline → condition evaluation
            let dummy = self.alloc_temp_slot();
            self.cfg[cond_entry] = Label::Instr(
                Instr::Assign(Lhs::Var(dummy), Expr::Unit),
                eval_entry,
            );
        } else {
            // No condition: infinite loop, jump straight to body
            let dummy = self.alloc_temp_slot();
            self.cfg[cond_entry] = Label::Instr(
                Instr::Assign(Lhs::Var(dummy), Expr::Unit),
                body_vertex,
            );
        }

        // 4. Init → cond_entry (runs once before the loop)
        if let Some(init) = &fl.init {
            match init {
                LForLoopInit::VarInit(vi) => {
                    let lhs = Lhs::Var(self.resolve_slot(vi.name));
                    self.compile_expr_to_value(&vi.value, lhs, cond_entry, ctx)
                }
                LForLoopInit::Assignment(a) => {
                    let init_stmt = LStatement {
                        kind: LStatementKind::Assignment(a.clone()),
                        span: a.span,
                    };
                    self.compile_statement(&init_stmt, cond_entry, ctx)
                }
            }
        } else {
            cond_entry
        }
    }

    fn compile_for_in_loop(
        &mut self,
        loop_stmt: &LForInLoop,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let for_vertex = self.add_label(Label::Return(Expr::Unit));

        let loop_ctx = CompileCtx {
            break_target: next_vertex,
            continue_target: for_vertex,
            ..ctx
        };

        let body_vertex = self.compile_tailless_block(&loop_stmt.body, for_vertex, loop_ctx);

        let lhs = Lhs::Var(self.resolve_slot(loop_stmt.binding_name));
        let iter_state_slot = self.alloc_temp_slot();
        let iterable_slot = self.alloc_temp_slot();

        let for_label = Label::ForLoopIn(
            lhs,
            Expr::Var(iterable_slot), // The collection expression
            iter_state_slot,           // Iterator state slot
            body_vertex,               // On iteration, go to body
            next_vertex,               // When done, exit loop
        );

        // Replace the dummy label with the real one.
        self.cfg[for_vertex] = for_label;
        self.compile_expr_to_value(
            &loop_stmt.iterable,
            Lhs::Var(iterable_slot),
            for_vertex,
            ctx,
        )
    }


    // Helper to generate temp slots and Expr::Var expressions for a list
    fn compile_temp_list(&mut self, count: usize) -> (Vec<VarSlot>, Vec<Expr>) {
        (0..count)
            .map(|_| {
                let slot = self.alloc_temp_slot();
                (slot, Expr::Var(slot))
            })
            .unzip()
    }

    // Helper to generate temp slots and Expr::Var expressions for (key, value) pairs
    fn compile_temp_pairs(
        &mut self,
        count: usize,
    ) -> (Vec<VarSlot>, Vec<VarSlot>, Vec<(Expr, Expr)>) {
        let mut key_slots = Vec::with_capacity(count);
        let mut val_slots = Vec::with_capacity(count);
        let mut simple_pairs = Vec::with_capacity(count);
        for _ in 0..count {
            let key_slot = self.alloc_temp_slot();
            let val_slot = self.alloc_temp_slot();
            simple_pairs.push((Expr::Var(key_slot), Expr::Var(val_slot)));
            key_slots.push(key_slot);
            val_slots.push(val_slot);
        }
        (key_slots, val_slots, simple_pairs)
    }

    fn compile_expr_list_recursive<'a, I>(
        &mut self,
        exprs: I,
        slots: Vec<VarSlot>,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex
    where
        I: DoubleEndedIterator<Item = &'a LExpr>,
    {
        let mut next = next_vertex;
        for (expr, slot) in exprs.rev().zip(slots.iter().rev()) {
            next = self.compile_expr_to_value(expr, Lhs::Var(*slot), next, ctx);
        }
        next
    }

    /// Compile a single sub-expression, apply `make_expr`, assign to `target`.
    fn compile_unary(
        &mut self,
        e: &LExpr,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
        make_expr: impl FnOnce(Box<Expr>) -> Expr,
    ) -> Vertex {
        let tmp = self.alloc_temp_slot();
        let final_expr = make_expr(Box::new(Expr::Var(tmp)));
        let assign = self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
        self.compile_expr_to_value(e, Lhs::Var(tmp), assign, ctx)
    }

    /// Compile two sub-expressions (left then right), apply `make_expr`, assign to `target`.
    fn compile_binary(
        &mut self,
        l: &LExpr,
        r: &LExpr,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
        make_expr: impl FnOnce(Box<Expr>, Box<Expr>) -> Expr,
    ) -> Vertex {
        let l_tmp = self.alloc_temp_slot();
        let r_tmp = self.alloc_temp_slot();
        let final_expr = make_expr(Box::new(Expr::Var(l_tmp)), Box::new(Expr::Var(r_tmp)));
        let assign = self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
        let r_v = self.compile_expr_to_value(r, Lhs::Var(r_tmp), assign, ctx);
        self.compile_expr_to_value(l, Lhs::Var(l_tmp), r_v, ctx)
    }

    /// Compile three sub-expressions, apply `make_expr`, assign to `target`.
    fn compile_ternary(
        &mut self,
        a: &LExpr,
        b: &LExpr,
        c: &LExpr,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
        make_expr: impl FnOnce(Box<Expr>, Box<Expr>, Box<Expr>) -> Expr,
    ) -> Vertex {
        let a_tmp = self.alloc_temp_slot();
        let b_tmp = self.alloc_temp_slot();
        let c_tmp = self.alloc_temp_slot();
        let final_expr = make_expr(
            Box::new(Expr::Var(a_tmp)),
            Box::new(Expr::Var(b_tmp)),
            Box::new(Expr::Var(c_tmp)),
        );
        let assign = self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
        let c_v = self.compile_expr_to_value(c, Lhs::Var(c_tmp), assign, ctx);
        let b_v = self.compile_expr_to_value(b, Lhs::Var(b_tmp), c_v, ctx);
        self.compile_expr_to_value(a, Lhs::Var(a_tmp), b_v, ctx)
    }


    /// Compiles any expression into CFG labels, storing the
    /// result in `target`.
    /// Returns the entry vertex for this sequence of instructions.
    fn compile_expr_to_value(
        &mut self,
        expr: &LExpr,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        // Fast path: if the whole expression is pure, convert directly
        if let Some(simple) = self.try_to_expr(expr) {
            return self.add_label(Label::Instr(Instr::Assign(target, simple), next_vertex));
        }

        match &expr.kind {
            LExprKind::FuncCall(call) => self.compile_func_call(call, target, next_vertex, ctx),
            LExprKind::RpcCall(target_expr, call) => {
                self.compile_rpc_call(target_expr, call, target, next_vertex, ctx)
            }
            LExprKind::MakeChannel => {
                self.add_label(Label::MakeChannel(target, None, next_vertex))
            }
            LExprKind::Send(chan_expr, val_expr) => {
                let chan_tmp = self.alloc_temp_slot();
                let val_tmp = self.alloc_temp_slot();

                let assign_unit_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex));

                let send_vertex = self.add_label(Label::Send(
                    Expr::Var(chan_tmp),
                    Expr::Var(val_tmp),
                    assign_unit_vertex,
                ));

                let val_vertex =
                    self.compile_expr_to_value(val_expr, Lhs::Var(val_tmp), send_vertex, ctx);
                self.compile_expr_to_value(chan_expr, Lhs::Var(chan_tmp), val_vertex, ctx)
            }
            LExprKind::Recv(chan_expr) => {
                let chan_tmp = self.alloc_temp_slot();
                let recv_vertex =
                    self.add_label(Label::Recv(target, Expr::Var(chan_tmp), next_vertex));
                self.compile_expr_to_value(chan_expr, Lhs::Var(chan_tmp), recv_vertex, ctx)
            }

            LExprKind::BinOp(op, l, r) => {
                let op = op.clone();
                self.compile_binary(l, r, target, next_vertex, ctx,
                    |l, r| convert_binop(&op, l, r))
            }

            LExprKind::Not(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::Not)
            }
            LExprKind::Negate(e) => {
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::Minus(Box::new(Expr::Int(0)), v))
            }

            LExprKind::MapLit(pairs) => {
                let (key_tmps, val_tmps, simple_pairs) = self.compile_temp_pairs(pairs.len());
                let final_expr = Expr::Map(simple_pairs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));

                let keys_entry_vertex = self.compile_expr_list_recursive(
                    pairs.iter().map(|(k, _v)| k),
                    key_tmps,
                    assign_vertex,
                    ctx,
                );

                self.compile_expr_list_recursive(
                    pairs.iter().map(|(_k, v)| v),
                    val_tmps,
                    keys_entry_vertex,
                    ctx,
                )
            }

            LExprKind::ListLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(items.len());
                let final_expr = Expr::List(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex, ctx)
            }

            LExprKind::TupleLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(items.len());
                let final_expr = Expr::Tuple(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex, ctx)
            }

            LExprKind::Append(l, i) => {
                self.compile_binary(l, i, target, next_vertex, ctx, Expr::ListAppend)
            }
            LExprKind::Prepend(i, l) => {
                self.compile_binary(i, l, target, next_vertex, ctx, Expr::ListPrepend)
            }
            LExprKind::Len(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::ListLen)
            }
            LExprKind::Min(l, r) => {
                self.compile_binary(l, r, target, next_vertex, ctx, Expr::Min)
            }
            LExprKind::Exists(map, key) => {
                self.compile_binary(map, key, target, next_vertex, ctx,
                    |l, r| Expr::KeyExists(r, l))
            }
            LExprKind::Erase(map, key) => {
                self.compile_binary(map, key, target, next_vertex, ctx,
                    |l, r| Expr::MapErase(r, l))
            }
            LExprKind::Head(e) => {
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::ListAccess(v, 0))
            }
            LExprKind::Tail(e) => {
                self.compile_unary(e, target, next_vertex, ctx, |v| {
                    Expr::ListSubsequence(
                        v.clone(),
                        Box::new(Expr::Int(1)),
                        Box::new(Expr::ListLen(v)),
                    )
                })
            }
            LExprKind::Index(a, b) => {
                self.compile_binary(a, b, target, next_vertex, ctx, Expr::Find)
            }
            LExprKind::Slice(a, b, c) => {
                self.compile_ternary(a, b, c, target, next_vertex, ctx, Expr::ListSubsequence)
            }
            LExprKind::TupleAccess(e, i) => {
                let i = *i;
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::TupleAccess(v, i))
            }
            LExprKind::FieldAccess(e, field) => {
                let field = EcoString::from(field.as_str());
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::Find(v, Box::new(Expr::String(field))))
            }
            LExprKind::MakeIter(_) | LExprKind::IterIsDone(_) | LExprKind::IterNext(_) => todo!("Iterator primitives to CFG"),
            LExprKind::UnwrapOptional(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::Unwrap)
            }
            LExprKind::WrapInOptional(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::Some)
            }
            LExprKind::SafeFieldAccess(e, field) => {
                let field = EcoString::from(field.as_str());
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::SafeFind(v, Box::new(Expr::String(field))))
            }
            LExprKind::SafeIndex(a, b) => {
                self.compile_binary(a, b, target, next_vertex, ctx, Expr::SafeFind)
            }
            LExprKind::SafeTupleAccess(e, i) => {
                let i = *i;
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::SafeTupleAccess(v, i))
            }

            LExprKind::StructLit(_, fields) => {
                let (field_names, val_exprs): (Vec<_>, Vec<_>) = fields.iter().cloned().unzip();
                let (tmps, simple_exprs) = self.compile_temp_list(val_exprs.len());
                let final_pairs = field_names
                    .into_iter()
                    .map(|s| Expr::String(EcoString::from(s.as_str())))
                    .zip(simple_exprs)
                    .collect();
                let final_expr = Expr::Map(final_pairs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(val_exprs.iter(), tmps, assign_vertex, ctx)
            }

            LExprKind::VariantLit(enum_id, variant_name, Some(payload_expr)) => {
                let eid = enum_id.0 as u32;
                let vname = EcoString::from(variant_name.as_str());
                self.compile_unary(payload_expr, target, next_vertex, ctx,
                    |v| Expr::Variant(eid, vname, Some(v)))
            }

            LExprKind::IsVariant(e, variant_name) => {
                let vname = EcoString::from(variant_name.as_str());
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::IsVariant(v, vname))
            }

            LExprKind::VariantPayload(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::VariantPayload)
            }

            LExprKind::Conditional(cond) => {
                self.compile_conditional(cond, target, next_vertex, ctx)
            }

            LExprKind::Block(block) => {
                self.compile_block(block, target, next_vertex, ctx)
            }

            LExprKind::Store(a, b, c) => {
                self.compile_ternary(a, b, c, target, next_vertex, ctx, Expr::Store)
            }

            LExprKind::SetTimer(label) => self.add_label(Label::SetTimer(target, next_vertex, label.clone())),

            LExprKind::PersistData(e) => {
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex));

                let e_tmp = self.alloc_temp_slot();
                let type_id = self.type_ids.get(&e.ty).copied().unwrap_or(TypeId(0));
                let persist_label = Label::PersistData(type_id, Expr::Var(e_tmp), assign_vertex);
                let persist_vertex = self.add_label(persist_label);

                self.compile_expr_to_value(e, Lhs::Var(e_tmp), persist_vertex, ctx)
            }
            LExprKind::RetrieveData(inner_type) => {
                let type_id = self.type_ids.get(inner_type).copied().unwrap_or(TypeId(0));
                self.add_label(Label::RetrieveData(type_id, target, next_vertex))
            }
            LExprKind::DiscardData => {
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex));
                let discard_label = Label::DiscardData(assign_vertex);
                self.add_label(discard_label)
            }

            // Control flow expressions — handled at statement level in compile_statement,
            // but can appear nested in expression position.
            LExprKind::Return(inner) => {
                self.compile_expr_to_value(
                    inner,
                    Lhs::Var(ctx.return_slot),
                    ctx.return_target,
                    ctx,
                )
            }
            LExprKind::Break => self.add_label(Label::Break(ctx.break_target)),
            LExprKind::Continue => self.add_label(Label::Continue(ctx.continue_target)),

            _ => unreachable!(
                "All simple expressions handled by fast path, all complex expressions \
                 handled above. Expression: {:?}",
                expr.kind
            ),
        }
    }

    fn compile_async_call_internal(
        &mut self,
        node_expr: Expr,
        call: &LUserFuncCall,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        // Allocate all temporary variable names up front.
        // Create temp var names for each argument.
        let (arg_temp_vars, arg_exprs): (Vec<VarSlot>, Vec<Expr>) = (0..call.args.len())
            .map(|_| {
                let tmp = self.alloc_temp_slot();
                (tmp, Expr::Var(tmp))
            })
            .unzip();

        let qualifier = self.func_qualifier_map.get(&call.name).unwrap_or_else(|| {
            panic!(
                "Function qualifier not found for function '{}'. \
                     This indicates a bug in the CFG builder initialization.",
                call.original_name
            )
        });
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
            entry_vertex = self.compile_expr_to_value(arg, Lhs::Var(*tmp_var), entry_vertex, ctx);
        }

        // This is the entry point for the "call" part
        entry_vertex
    }

    fn compile_func_call(
        &mut self,
        call: &LFuncCall,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        match call {
            LFuncCall::User(user_call) => {
                let is_sync = self
                    .func_sync_map
                    .get(&user_call.name)
                    .copied()
                    .unwrap_or(false);

                if is_sync {
                    // Allocate temp vars for all arguments
                    let (arg_tmps, arg_exprs): (Vec<VarSlot>, Vec<Expr>) =
                        (0..user_call.args.len())
                            .map(|_| {
                                let tmp = self.alloc_temp_slot();
                                (tmp, Expr::Var(tmp))
                            })
                            .unzip();

                    let qualifier =
                        self.func_qualifier_map
                            .get(&user_call.name)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Function qualifier not found for function '{}'. \
                                 This indicates a bug in the CFG builder initialization.",
                                    user_call.original_name
                                )
                            });
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
                        user_call.args.iter(),
                        arg_tmps,
                        sync_call_vertex,
                        ctx,
                    )
                } else {
                    let node_expr = Expr::Var(SELF_SLOT);
                    self.compile_async_call_internal(node_expr, user_call, target, next_vertex, ctx)
                }
            }
            LFuncCall::Builtin(builtin, args, _return_ty) => {
                self.compile_builtin_call(*builtin, args, target, next_vertex, ctx)
            }
        }
    }

    fn compile_builtin_call(
        &mut self,
        builtin: BuiltinFn,
        args: &Vec<LExpr>,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        match builtin {
            BuiltinFn::Println => {
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex));

                let arg_var = self.alloc_temp_slot();
                let print_label = Label::Print(Expr::Var(arg_var), assign_vertex);
                let print_vertex = self.add_label(print_label);

                let arg_expr = args.first().expect("Println should have 1 arg");
                self.compile_expr_to_value(arg_expr, Lhs::Var(arg_var), print_vertex, ctx)
            }
            BuiltinFn::IntToString => {
                let arg = &args[0];
                self.compile_unary(arg, target, next_vertex, ctx, Expr::IntToString)
            }
            BuiltinFn::BoolToString => {
                let arg = &args[0];
                self.compile_unary(arg, target, next_vertex, ctx, Expr::BoolToString)
            }
            BuiltinFn::RoleToString => {
                let arg = &args[0];
                self.compile_unary(arg, target, next_vertex, ctx, Expr::NodeToString)
            }
            BuiltinFn::UniqueId => self.add_label(Label::UniqueId(target, next_vertex)),
        }
    }

    fn compile_rpc_call(
        &mut self,
        target_expr: &LExpr,
        call: &LUserFuncCall,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let (arg_tmps, arg_exprs) = self.compile_temp_list(call.args.len());

        let node_tmp = self.alloc_temp_slot();
        let node_expr = Expr::Var(node_tmp);

        let qualifier = self.func_qualifier_map.get(&call.name).unwrap_or_else(|| {
            panic!(
                "Function qualifier not found for function '{}'. \
                     This indicates a bug in the CFG builder initialization.",
                call.original_name
            )
        });
        let func_name = format!("{}.{}", qualifier, call.original_name);

        // The simulator's Async handler automatically creates a channel/future and stores it in `target`.
        let async_label = Label::Instr(
            Instr::Async(target, node_expr, func_name.clone(), arg_exprs.clone()),
            next_vertex,
        );
        let async_vertex = self.add_label(async_label);

        let effective_async_target = if self
            .func_traced_map
            .get(&call.name)
            .copied()
            .unwrap_or(false)
        {
            self.add_label(Label::TraceDispatch(func_name, arg_exprs, async_vertex))
        } else {
            async_vertex
        };

        let args_entry_vertex = self.compile_expr_list_recursive(
            call.args.iter(),
            arg_tmps,
            effective_async_target,
            ctx,
        );

        self.compile_expr_to_value(target_expr, Lhs::Var(node_tmp), args_entry_vertex, ctx)
    }


    /// Tries to convert a `LExpr` into a pure `cfg::Expr`.
    /// Returns `None` for expressions that require CFG control flow
    /// (function calls, conditionals, channels, etc.).
    fn try_to_expr(&mut self, expr: &LExpr) -> Option<Expr> {
        Some(match &expr.kind {
            LExprKind::Var(id, _name) => Expr::Var(self.resolve_slot(*id)),
            LExprKind::IntLit(i) => Expr::Int(*i),
            LExprKind::StringLit(s) => Expr::String(EcoString::from(s.as_str())),
            LExprKind::BoolLit(b) => Expr::Bool(*b),
            LExprKind::NilLit => Expr::Nil,
            LExprKind::BinOp(op, l, r) => {
                let left = Box::new(self.try_to_expr(l)?);
                let right = Box::new(self.try_to_expr(r)?);
                convert_binop(op, left, right)
            }
            LExprKind::Not(e) => Expr::Not(Box::new(self.try_to_expr(e)?)),
            LExprKind::Negate(e) => {
                Expr::Minus(Box::new(Expr::Int(0)), Box::new(self.try_to_expr(e)?))
            }
            LExprKind::MapLit(pairs) => {
                let mut out = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    out.push((self.try_to_expr(k)?, self.try_to_expr(v)?));
                }
                Expr::Map(out)
            }
            LExprKind::ListLit(items) => {
                let mut out = Vec::with_capacity(items.len());
                for e in items {
                    out.push(self.try_to_expr(e)?);
                }
                Expr::List(out)
            }
            LExprKind::TupleLit(items) => {
                let mut out = Vec::with_capacity(items.len());
                for e in items {
                    out.push(self.try_to_expr(e)?);
                }
                Expr::Tuple(out)
            }
            LExprKind::Append(l, i) => Expr::ListAppend(
                Box::new(self.try_to_expr(l)?),
                Box::new(self.try_to_expr(i)?),
            ),
            LExprKind::Prepend(i, l) => Expr::ListPrepend(
                Box::new(self.try_to_expr(i)?),
                Box::new(self.try_to_expr(l)?),
            ),
            LExprKind::Len(e) => Expr::ListLen(Box::new(self.try_to_expr(e)?)),
            LExprKind::Min(l, r) => Expr::Min(
                Box::new(self.try_to_expr(l)?),
                Box::new(self.try_to_expr(r)?),
            ),
            LExprKind::Exists(map, key) => Expr::KeyExists(
                Box::new(self.try_to_expr(key)?),
                Box::new(self.try_to_expr(map)?),
            ),
            LExprKind::Erase(map, key) => Expr::MapErase(
                Box::new(self.try_to_expr(key)?),
                Box::new(self.try_to_expr(map)?),
            ),
            LExprKind::Store(collection, key, value) => Expr::Store(
                Box::new(self.try_to_expr(collection)?),
                Box::new(self.try_to_expr(key)?),
                Box::new(self.try_to_expr(value)?),
            ),
            LExprKind::Head(l) => Expr::ListAccess(Box::new(self.try_to_expr(l)?), 0),
            LExprKind::Tail(l) => {
                let list_expr = self.try_to_expr(l)?;
                Expr::ListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::Int(1)),
                    Box::new(Expr::ListLen(Box::new(list_expr))),
                )
            }
            LExprKind::Index(target, index) => Expr::Find(
                Box::new(self.try_to_expr(target)?),
                Box::new(self.try_to_expr(index)?),
            ),
            LExprKind::Slice(target, start, end) => Expr::ListSubsequence(
                Box::new(self.try_to_expr(target)?),
                Box::new(self.try_to_expr(start)?),
                Box::new(self.try_to_expr(end)?),
            ),
            LExprKind::TupleAccess(tuple, index) => {
                Expr::TupleAccess(Box::new(self.try_to_expr(tuple)?), *index)
            }
            LExprKind::FieldAccess(target, field) => Expr::Find(
                Box::new(self.try_to_expr(target)?),
                Box::new(Expr::String(EcoString::from(field.as_str()))),
            ),
            LExprKind::MakeIter(_) | LExprKind::IterIsDone(_) | LExprKind::IterNext(_) => return None,
            LExprKind::UnwrapOptional(e) => Expr::Unwrap(Box::new(self.try_to_expr(e)?)),
            LExprKind::WrapInOptional(e) => Expr::Some(Box::new(self.try_to_expr(e)?)),
            LExprKind::SafeFieldAccess(target, field) => Expr::SafeFind(
                Box::new(self.try_to_expr(target)?),
                Box::new(Expr::String(EcoString::from(field.as_str()))),
            ),
            LExprKind::SafeIndex(target, index) => Expr::SafeFind(
                Box::new(self.try_to_expr(target)?),
                Box::new(self.try_to_expr(index)?),
            ),
            LExprKind::SafeTupleAccess(tuple, index) => {
                Expr::SafeTupleAccess(Box::new(self.try_to_expr(tuple)?), *index)
            }
            LExprKind::StructLit(_, fields) => {
                let mut out = Vec::with_capacity(fields.len());
                for (name, val) in fields {
                    out.push((
                        Expr::String(EcoString::from(name.as_str())),
                        self.try_to_expr(val)?,
                    ));
                }
                Expr::Map(out)
            }
            LExprKind::VariantLit(enum_id, variant_name, None) => Expr::Variant(
                enum_id.0 as u32,
                EcoString::from(variant_name.as_str()),
                None,
            ),
            LExprKind::IsVariant(e, variant_name) => Expr::IsVariant(
                Box::new(self.try_to_expr(e)?),
                EcoString::from(variant_name.as_str()),
            ),
            LExprKind::VariantPayload(e) => Expr::VariantPayload(Box::new(self.try_to_expr(e)?)),

            // Control flow expressions — cannot be inlined into pure expressions
            LExprKind::Return(_)
            | LExprKind::Break
            | LExprKind::Continue => return None,

            // Complex expressions that need CFG control flow
            LExprKind::FuncCall(_)
            | LExprKind::RpcCall(_, _)
            | LExprKind::MakeChannel
            | LExprKind::Send(_, _)
            | LExprKind::Recv(_)
            | LExprKind::Conditional(_)
            | LExprKind::Block(_)
            | LExprKind::PersistData(_)
            | LExprKind::RetrieveData(_)
            | LExprKind::DiscardData
            | LExprKind::SetTimer(_)
            | LExprKind::VariantLit(_, _, Some(_))
            | LExprKind::Error => return None,
        })
    }

}

#[cfg(test)]
mod test;
