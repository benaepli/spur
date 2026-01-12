use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::{
    TypedCondStmts, TypedExpr, TypedExprKind, TypedForInLoop, TypedForLoop, TypedForLoopInit,
    TypedFuncCall, TypedFuncDef, TypedPattern, TypedPatternKind, TypedProgram, TypedStatement,
    TypedStatementKind, TypedTopLevelDef, TypedUserFuncCall, TypedVarInit,
};
use crate::parser::{BinOp, Span};
use ecow::EcoString;
use serde::Serialize;
use std::collections::HashMap;

pub const SELF_NAME: NameId = NameId(usize::MAX);

/// Identifies where a variable lives at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum VarSlot {
    /// Index into the function's local environment
    Local(u32, NameId),
    /// Index into the node's persistent environment
    Node(u32, NameId),
}

/// Slot 0 in node env is reserved for 'self'
pub const SELF_SLOT: VarSlot = VarSlot::Node(0, SELF_NAME);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Expr {
    Var(VarSlot),
    Find(Box<Expr>, Box<Expr>),
    Int(i64),
    Bool(bool),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    EqualsEquals(Box<Expr>, Box<Expr>),
    Map(Vec<(Expr, Expr)>),
    List(Vec<Expr>),
    ListPrepend(Box<Expr>, Box<Expr>),
    ListAppend(Box<Expr>, Box<Expr>),
    ListSubsequence(Box<Expr>, Box<Expr>, Box<Expr>),
    String(EcoString),
    LessThan(Box<Expr>, Box<Expr>),
    LessThanEquals(Box<Expr>, Box<Expr>),
    GreaterThan(Box<Expr>, Box<Expr>),
    GreaterThanEquals(Box<Expr>, Box<Expr>),
    KeyExists(Box<Expr>, Box<Expr>),
    MapErase(Box<Expr>, Box<Expr>),
    Store(Box<Expr>, Box<Expr>, Box<Expr>),
    ListLen(Box<Expr>),
    ListAccess(Box<Expr>, usize),
    Plus(Box<Expr>, Box<Expr>),
    Minus(Box<Expr>, Box<Expr>),
    Times(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Tuple(Vec<Expr>),
    TupleAccess(Box<Expr>, usize),
    Unit,
    Nil,
    Unwrap(Box<Expr>),
    Coalesce(Box<Expr>, Box<Expr>),
    Some(Box<Expr>),
    IntToString(Box<Expr>),
    BoolToString(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Hash)]
pub enum Lhs {
    Var(VarSlot),
    Tuple(Vec<VarSlot>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Instr {
    Assign(Lhs, Expr),
    Async(Lhs, Expr, String, Vec<Expr>),
    Copy(Lhs, Expr),
    SyncCall(Lhs, String, Vec<Expr>),
}

pub type Vertex = usize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionInfo {
    pub entry: Vertex,
    pub name: NameId,

    /// Number of parameters (occupy slots 0..param_count)
    pub param_count: u32,

    /// Total local slots needed (params + locals + temps)
    pub local_slot_count: u32,

    /// Default values for locals, indexed by (slot - param_count)
    pub local_defaults: Vec<Expr>,

    pub is_sync: bool,

    /// For debugging: maps slot index -> original variable name
    #[serde(skip)]
    pub debug_slot_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Label {
    Instr(Instr, Vertex /* next_vertex */),
    Pause(Vertex /* next_vertex */),
    MakeChannel(Lhs, usize, Vertex),
    SetTimer(Lhs, Vertex),
    UniqueId(Lhs, Vertex),
    Send(Expr, Expr, Vertex),
    Recv(Lhs, Expr, Vertex),
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
        VarSlot, /* iterator_state_slot */
        Vertex,  /* body_vertex */
        Vertex,  /* next_vertex */
    ),
    Print(Expr, Vertex /* next_vertex */),
    Break(Vertex /* break_target_vertex */),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cfg {
    pub graph: Vec<Label>,
}

impl Cfg {
    pub fn get_label(&self, v: Vertex) -> &Label {
        &self.graph[v]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    // The CFG is just a list of all vertices.
    // The `Vertex` indices in `Label` and `FunctionInfo`
    // are indices into this Vec.
    pub cfg: Cfg,

    // Map of function NameId -> FunctionInfo
    pub rpc: HashMap<NameId, FunctionInfo>,

    // Mapping from qualified function name strings to NameId
    pub func_name_to_id: HashMap<String, NameId>,

    // Mapping from NameId to original name (for debugging/display)
    pub id_to_name: HashMap<NameId, String>,

    // The next available NameId for generating new names
    pub next_name_id: usize,

    /// Maps vertices to their source statement spans.
    /// Not all vertices have spans - compiler-generated vertices
    /// (function entry/exit, etc.) will not appear in this map.
    /// All vertices generated from compiling a statement share that statement's span.
    #[serde(skip)]
    pub vertex_to_span: HashMap<Vertex, Span>,

    /// Maximum number of node-level slots used by any role in this program.
    pub max_node_slots: u32,
}

impl Program {
    /// Look up a function by its qualified name (e.g., "Node.Init").
    /// Returns the FunctionInfo if found, None otherwise.
    pub fn get_func_by_name(&self, name: &str) -> Option<&FunctionInfo> {
        self.func_name_to_id
            .get(name)
            .and_then(|id| self.rpc.get(id))
    }
}

pub struct Compiler {
    /// The CFG being built. New labels are pushed here.
    cfg: Vec<Label>,

    rpc_map: HashMap<NameId, FunctionInfo>,

    func_sync_map: HashMap<NameId, bool>,
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
        }
    }

    /// Call when starting to compile a new role
    fn begin_role(&mut self, var_inits: &[TypedVarInit]) {
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
    pub fn compile_program(mut self, program: TypedProgram) -> Program {
        let next_name_id = program.next_name_id;
        self.next_name_id = next_name_id;
        // Initialize id_to_name from the resolver's map
        self.id_to_name = program.id_to_name;

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
        }
    }

    fn compile_init_func(
        &mut self,
        inits: &[TypedVarInit],
        qualified_name: String,
    ) -> FunctionInfo {
        // Begin a new function with no parameters
        self.begin_function(&[]);

        // Allocate a temp slot for the return value
        let return_slot = self.alloc_temp_slot();
        let final_vertex = self.add_label(Label::Return(Expr::Var(return_slot)));

        let mut next_vertex = final_vertex;
        // Build the init chain backwards
        // These assign to NODE slots, not local slots
        for init in inits.iter().rev() {
            let node_slot = self.resolve_slot(init.name);
            next_vertex = self.compile_expr_to_value(&init.value, Lhs::Var(node_slot), next_vertex);
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

    fn compile_func_def(&mut self, func: TypedFuncDef) -> FunctionInfo {
        // Collect parameter info: (NameId, display_name)
        let params: Vec<(NameId, String)> = func
            .params
            .iter()
            .map(|p| (p.name, p.original_name.clone()))
            .collect();
        let param_count = params.len() as u32;

        // Begin the function - this assigns slots to parameters
        self.begin_function(&params);

        // Scan the body for all local variable declarations and assign them slots
        self.scan_and_assign_slots(&func.body);

        // Allocate a temp slot for the return value
        let return_slot = self.alloc_temp_slot();
        let final_return_vertex = self.add_label(Label::Return(Expr::Var(return_slot)));

        let body_entry = self.compile_block(
            &func.body,
            final_return_vertex,
            final_return_vertex, // `break_target` (breaks go to end of function)
            final_return_vertex, // `return_target`
            return_slot,
        );

        let entry = self.add_label(Label::Instr(
            Instr::Assign(Lhs::Var(return_slot), Expr::Unit),
            body_entry,
        ));

        let qualifier = self.func_qualifier_map.get(&func.name).unwrap_or_else(|| {
            panic!(
                "Function qualifier not found for function '{}'. \
                     This indicates a bug in the CFG builder initialization.",
                func.original_name
            )
        });

        // Use the existing func.name NameId from the resolver, not a new one
        let qualified_name = format!("{}.{}", qualifier, func.original_name);
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
    fn scan_and_assign_slots(&mut self, body: &[TypedStatement]) {
        for stmt in body {
            self.scan_stmt_and_assign_slots(stmt);
        }
    }

    fn scan_stmt_and_assign_slots(&mut self, stmt: &TypedStatement) {
        match &stmt.kind {
            TypedStatementKind::VarInit(init) => {
                let name_str = self
                    .id_to_name
                    .get(&init.name)
                    .cloned()
                    .unwrap_or_else(|| format!("var_{}", init.name.0));
                self.alloc_local_slot(init.name, &name_str, Expr::Nil);
            }
            TypedStatementKind::Conditional(cond) => {
                self.scan_body_and_assign_slots(&cond.if_branch.body);
                for branch in &cond.elseif_branches {
                    self.scan_body_and_assign_slots(&branch.body);
                }
                if let Some(body) = &cond.else_branch {
                    self.scan_body_and_assign_slots(body);
                }
            }
            TypedStatementKind::ForLoop(fl) => {
                if let Some(TypedForLoopInit::VarInit(init)) = &fl.init {
                    let name_str = self
                        .id_to_name
                        .get(&init.name)
                        .cloned()
                        .unwrap_or_else(|| format!("var_{}", init.name.0));
                    self.alloc_local_slot(init.name, &name_str, Expr::Nil);
                }
                self.scan_body_and_assign_slots(&fl.body);
            }
            TypedStatementKind::ForInLoop(loop_stmt) => {
                self.scan_pattern_and_assign_slots(&loop_stmt.pattern);
                self.scan_body_and_assign_slots(&loop_stmt.body);
            }
            _ => {}
        }
    }

    fn scan_body_and_assign_slots(&mut self, body: &[TypedStatement]) {
        for stmt in body {
            self.scan_stmt_and_assign_slots(stmt);
        }
    }

    fn scan_pattern_and_assign_slots(&mut self, pat: &TypedPattern) {
        match &pat.kind {
            TypedPatternKind::Var(id, name) => {
                self.alloc_local_slot(*id, name, Expr::Nil);
            }
            TypedPatternKind::Tuple(pats) => {
                for p in pats {
                    self.scan_pattern_and_assign_slots(p);
                }
            }
            _ => {} // Wildcard doesn't add a named local
        }
    }

    fn compile_block(
        &mut self,
        body: &[TypedStatement],
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_slot: VarSlot,
    ) -> Vertex {
        let mut next = next_vertex;
        // Iterate in reverse to chain statements:
        // StmtN -> next_vertex
        // StmtN-1 -> StmtN
        // ...
        // Stmt0 -> Stmt1
        for stmt in body.iter().rev() {
            next = self.compile_statement(stmt, next, break_target, return_target, return_slot);
        }
        next
    }

    fn compile_statement(
        &mut self,
        stmt: &TypedStatement,
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_slot: VarSlot,
    ) -> Vertex {
        // Save previous span and set current span
        let prev_span = self.current_span;
        self.current_span = Some(stmt.span);

        let result = match &stmt.kind {
            TypedStatementKind::VarInit(init) => {
                // Compile the value, assigning to the new variable's slot
                let slot = self.resolve_slot(init.name);
                self.compile_expr_to_value(&init.value, Lhs::Var(slot), next_vertex)
            }
            TypedStatementKind::Assignment(assign) => {
                // Compile the value, assigning to the target LHS
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, next_vertex)
            }
            TypedStatementKind::Expr(expr) => {
                // Compile the expression, but discard the result into a dummy slot
                let dummy_slot = self.alloc_temp_slot();
                self.compile_expr_to_value(expr, Lhs::Var(dummy_slot), next_vertex)
            }
            TypedStatementKind::Return(expr) => {
                // Compile the expression, store result in the dedicated return_slot,
                // and then jump to the function's final return_target.
                self.compile_expr_to_value(expr, Lhs::Var(return_slot), return_target)
            }
            TypedStatementKind::ForLoop(loop_stmt) => {
                self.compile_for_loop(loop_stmt, next_vertex, return_target, return_slot)
            }
            TypedStatementKind::Conditional(cond) => self.compile_conditional(
                cond,
                next_vertex,
                break_target,
                return_target,
                return_slot,
            ),
            TypedStatementKind::ForInLoop(loop_stmt) => {
                // `next_vertex` is the break target for a loop
                self.compile_for_in_loop(loop_stmt, next_vertex, return_target, return_slot)
            }
            TypedStatementKind::Break => self.add_label(Label::Break(break_target)),
        };

        // Restore previous span
        self.current_span = prev_span;

        result
    }

    fn compile_for_loop(
        &mut self,
        loop_stmt: &TypedForLoop,
        next_vertex: Vertex,
        return_target: Vertex,
        return_slot: VarSlot,
    ) -> Vertex {
        let loop_head_vertex = self.add_label(Label::Return(Expr::Unit));

        let increment_vertex = match &loop_stmt.increment {
            Some(assign) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, loop_head_vertex)
            }
            None => loop_head_vertex,
        };

        let body_vertex = self.compile_block(
            &loop_stmt.body,
            increment_vertex, // After body, go to increment
            next_vertex,      // Break goes to exit
            return_target,
            return_slot,
        );

        let cond_slot = self.alloc_temp_slot();

        let cond_check_vertex = self.add_label(Label::Cond(
            Expr::Var(cond_slot),
            body_vertex, // True -> Body
            next_vertex, // False -> Exit
        ));

        let cond_expr = loop_stmt.condition.as_ref().cloned().unwrap_or({
            // If no condition, default to true
            TypedExpr {
                kind: TypedExprKind::BoolLit(true),
                ty: crate::analysis::types::Type::Bool,
                span: loop_stmt.span,
            }
        });

        let cond_calc_vertex =
            self.compile_expr_to_value(&cond_expr, Lhs::Var(cond_slot), cond_check_vertex);

        let dummy_slot = self.alloc_temp_slot();
        self.cfg[loop_head_vertex] = Label::Instr(
            Instr::Assign(Lhs::Var(dummy_slot), Expr::Unit),
            cond_calc_vertex,
        );

        match &loop_stmt.init {
            Some(TypedForLoopInit::VarInit(vi)) => {
                let slot = self.resolve_slot(vi.name);
                self.compile_expr_to_value(&vi.value, Lhs::Var(slot), loop_head_vertex)
            }
            Some(TypedForLoopInit::Assignment(assign)) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, loop_head_vertex)
            }
            None => loop_head_vertex,
        }
    }

    fn compile_conditional(
        &mut self,
        cond: &TypedCondStmts,
        next_vertex: Vertex,
        break_target: Vertex,
        return_target: Vertex,
        return_slot: VarSlot,
    ) -> Vertex {
        let else_vertex = cond.else_branch.as_ref().map_or(next_vertex, |body| {
            self.compile_block(body, next_vertex, break_target, return_target, return_slot)
        });

        let mut next_cond_vertex = else_vertex;
        for branch in cond.elseif_branches.iter().rev() {
            let body_vertex = self.compile_block(
                &branch.body,
                next_vertex,
                break_target,
                return_target,
                return_slot,
            );
            let cond_slot = self.alloc_temp_slot();
            let check_vertex = self.add_label(Label::Cond(
                Expr::Var(cond_slot),
                body_vertex,
                next_cond_vertex,
            ));
            next_cond_vertex =
                self.compile_expr_to_value(&branch.condition, Lhs::Var(cond_slot), check_vertex);
        }

        let if_body_vertex = self.compile_block(
            &cond.if_branch.body,
            next_vertex,
            break_target,
            return_target,
            return_slot,
        );
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
        )
    }

    fn compile_for_in_loop(
        &mut self,
        loop_stmt: &TypedForInLoop,
        next_vertex: Vertex,
        return_target: Vertex,
        return_slot: VarSlot,
    ) -> Vertex {
        // The [ForLoopIn] label is the "head" of the loop.
        // We add a dummy label to reserve its vertex index.
        let for_vertex = self.add_label(Label::Return(Expr::Unit)); // Dummy label

        // Compile the [Body] block.
        // After it runs, it loops back to the [ForLoopIn] check.
        // Any `Break` inside it goes to `next_vertex`.
        let body_vertex = self.compile_block(
            &loop_stmt.body,
            for_vertex,  // Loop back to the ForLoopIn label
            next_vertex, // Break target
            return_target,
            return_slot,
        );

        // Now we create the real [ForLoopIn] label.
        let lhs = self.convert_pattern(&loop_stmt.pattern);
        // Allocate a slot for the iterator state (the remaining collection)
        let iter_state_slot = self.alloc_temp_slot();
        let iterable_expr = self.convert_simple_expr(&loop_stmt.iterable);

        let for_label = Label::ForLoopIn(
            lhs,
            iterable_expr,   // The collection expression
            iter_state_slot, // Iterator state slot
            body_vertex,     // On iteration, go to body
            next_vertex,     // When done, exit loop
        );

        // Replace the dummy label with the real one.
        self.cfg[for_vertex] = for_label;

        // The entry point to the whole statement is the for_vertex.
        for_vertex
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

    // Helper to recursively compile a list of expressions into a list of temp slots
    fn compile_expr_list_recursive<'a, I>(
        &mut self,
        exprs: I,
        slots: Vec<VarSlot>,
        next_vertex: Vertex,
    ) -> Vertex
    where
        I: DoubleEndedIterator<Item = &'a TypedExpr>,
    {
        let mut next = next_vertex;
        for (expr, slot) in exprs.rev().zip(slots.iter().rev()) {
            next = self.compile_expr_to_value(expr, Lhs::Var(*slot), next);
        }
        next
    }

    fn convert_simple_binop(&mut self, op: &BinOp, left: Box<Expr>, right: Box<Expr>) -> Expr {
        match op {
            BinOp::Add => Expr::Plus(left, right),
            BinOp::Subtract => Expr::Minus(left, right),
            BinOp::Multiply => Expr::Times(left, right),
            BinOp::Divide => Expr::Div(left, right),
            BinOp::Modulo => Expr::Mod(left, right),
            BinOp::And => Expr::And(left, right),
            BinOp::Or => Expr::Or(left, right),
            BinOp::Equal => Expr::EqualsEquals(left, right),
            BinOp::NotEqual => Expr::Not(Box::new(Expr::EqualsEquals(left, right))),
            BinOp::Less => Expr::LessThan(left, right),
            BinOp::LessEqual => Expr::LessThanEquals(left, right),
            BinOp::Greater => Expr::GreaterThan(left, right),
            BinOp::GreaterEqual => Expr::GreaterThanEquals(left, right),
            BinOp::Coalesce => Expr::Coalesce(left, right),
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
            TypedExprKind::MakeChannel(cap_expr) => {
                let capacity = match &cap_expr.kind {
                    TypedExprKind::IntLit(n) => *n as usize,
                    _ => 0,
                };
                self.add_label(Label::MakeChannel(target, capacity, next_vertex))
            }
            TypedExprKind::Send(chan_expr, val_expr) => {
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
                    self.compile_expr_to_value(val_expr, Lhs::Var(val_tmp), send_vertex);
                self.compile_expr_to_value(chan_expr, Lhs::Var(chan_tmp), val_vertex)
            }
            TypedExprKind::Recv(chan_expr) => {
                let chan_tmp = self.alloc_temp_slot();
                let recv_vertex =
                    self.add_label(Label::Recv(target, Expr::Var(chan_tmp), next_vertex));
                self.compile_expr_to_value(chan_expr, Lhs::Var(chan_tmp), recv_vertex)
            }

            TypedExprKind::BinOp(op, l, r) => {
                match op {
                    BinOp::And => {
                        let assign_false_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target.clone(), Expr::Bool(false)),
                            next_vertex,
                        ));

                        let eval_right_vertex =
                            self.compile_expr_to_value(r, target.clone(), next_vertex);

                        // Branch based on Left result
                        let l_tmp = self.alloc_temp_slot();
                        let cond_vertex = self.add_label(Label::Cond(
                            Expr::Var(l_tmp),
                            eval_right_vertex,   // If True: evaluate right
                            assign_false_vertex, // If False: short-circuit to false
                        ));

                        self.compile_expr_to_value(l, Lhs::Var(l_tmp), cond_vertex)
                    }
                    BinOp::Or => {
                        let assign_true_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target.clone(), Expr::Bool(true)),
                            next_vertex,
                        ));

                        let eval_right_vertex =
                            self.compile_expr_to_value(r, target.clone(), next_vertex);

                        // Branch based on Left result
                        let l_tmp = self.alloc_temp_slot();
                        let cond_vertex = self.add_label(Label::Cond(
                            Expr::Var(l_tmp),
                            assign_true_vertex, // If True: short-circuit to true
                            eval_right_vertex,  // If False: evaluate right
                        ));

                        self.compile_expr_to_value(l, Lhs::Var(l_tmp), cond_vertex)
                    }
                    _ => {
                        // Standard strict evaluation for non-boolean ops (+, -, *, etc.)
                        let l_tmp = self.alloc_temp_slot();
                        let r_tmp = self.alloc_temp_slot();
                        let l_expr = Box::new(Expr::Var(l_tmp));
                        let r_expr = Box::new(Expr::Var(r_tmp));
                        let final_expr = self.convert_simple_binop(op, l_expr, r_expr);

                        let assign_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target, final_expr),
                            next_vertex,
                        ));
                        let r_vertex =
                            self.compile_expr_to_value(r, Lhs::Var(r_tmp), assign_vertex);

                        self.compile_expr_to_value(l, Lhs::Var(l_tmp), r_vertex)
                    }
                }
            }

            TypedExprKind::Not(e) => {
                let e_tmp = self.alloc_temp_slot();
                let final_expr = Expr::Not(Box::new(Expr::Var(e_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::Negate(e) => {
                let e_tmp = self.alloc_temp_slot();
                let final_expr = Expr::Minus(Box::new(Expr::Int(0)), Box::new(Expr::Var(e_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::MapLit(pairs) => {
                let (key_tmps, val_tmps, simple_pairs) = self.compile_temp_pairs(pairs.len());
                let final_expr = Expr::Map(simple_pairs);
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
                let final_expr = Expr::List(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex)
            }

            TypedExprKind::TupleLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(items.len());
                let final_expr = Expr::Tuple(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex)
            }

            TypedExprKind::Append(l, i) => {
                let l_tmp = self.alloc_temp_slot();
                let i_tmp = self.alloc_temp_slot();
                let final_expr =
                    Expr::ListAppend(Box::new(Expr::Var(l_tmp)), Box::new(Expr::Var(i_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let i_vertex = self.compile_expr_to_value(i, Lhs::Var(i_tmp), assign_vertex);
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), i_vertex)
            }

            TypedExprKind::Prepend(i, l) => {
                let i_tmp = self.alloc_temp_slot();
                let l_tmp = self.alloc_temp_slot();
                let final_expr =
                    Expr::ListPrepend(Box::new(Expr::Var(i_tmp)), Box::new(Expr::Var(l_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let l_vertex = self.compile_expr_to_value(l, Lhs::Var(l_tmp), assign_vertex);
                self.compile_expr_to_value(i, Lhs::Var(i_tmp), l_vertex)
            }

            TypedExprKind::Len(e) => {
                let e_tmp = self.alloc_temp_slot();
                let final_expr = Expr::ListLen(Box::new(Expr::Var(e_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::Min(l, r) => {
                let l_tmp = self.alloc_temp_slot();
                let r_tmp = self.alloc_temp_slot();
                let final_expr = Expr::Min(Box::new(Expr::Var(l_tmp)), Box::new(Expr::Var(r_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let r_vertex = self.compile_expr_to_value(r, Lhs::Var(r_tmp), assign_vertex);
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), r_vertex)
            }

            TypedExprKind::Exists(map, key) => {
                let map_tmp = self.alloc_temp_slot();
                let key_tmp = self.alloc_temp_slot();
                let final_expr =
                    Expr::KeyExists(Box::new(Expr::Var(key_tmp)), Box::new(Expr::Var(map_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let key_vertex = self.compile_expr_to_value(key, Lhs::Var(key_tmp), assign_vertex);
                self.compile_expr_to_value(map, Lhs::Var(map_tmp), key_vertex)
            }

            TypedExprKind::Erase(map, key) => {
                let map_tmp = self.alloc_temp_slot();
                let key_tmp = self.alloc_temp_slot();
                let final_expr =
                    Expr::MapErase(Box::new(Expr::Var(key_tmp)), Box::new(Expr::Var(map_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let key_vertex = self.compile_expr_to_value(key, Lhs::Var(key_tmp), assign_vertex);
                self.compile_expr_to_value(map, Lhs::Var(map_tmp), key_vertex)
            }

            TypedExprKind::Head(l) => {
                let l_tmp = self.alloc_temp_slot();
                let final_expr = Expr::ListAccess(Box::new(Expr::Var(l_tmp)), 0);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), assign_vertex)
            }

            TypedExprKind::Tail(l) => {
                let l_tmp = self.alloc_temp_slot();
                let list_expr = Expr::Var(l_tmp);
                let final_expr = Expr::ListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::Int(1)),
                    Box::new(Expr::ListLen(Box::new(list_expr))),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(l, Lhs::Var(l_tmp), assign_vertex)
            }

            TypedExprKind::Index(target_expr, index_expr) => {
                let target_tmp = self.alloc_temp_slot();
                let index_tmp = self.alloc_temp_slot();
                let final_expr = Expr::Find(
                    Box::new(Expr::Var(target_tmp)),
                    Box::new(Expr::Var(index_tmp)),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                let index_vertex =
                    self.compile_expr_to_value(index_expr, Lhs::Var(index_tmp), assign_vertex);
                self.compile_expr_to_value(target_expr, Lhs::Var(target_tmp), index_vertex)
            }

            TypedExprKind::Slice(target_expr, start_expr, end_expr) => {
                let target_tmp = self.alloc_temp_slot();
                let start_tmp = self.alloc_temp_slot();
                let end_tmp = self.alloc_temp_slot();
                let final_expr = Expr::ListSubsequence(
                    Box::new(Expr::Var(target_tmp)),
                    Box::new(Expr::Var(start_tmp)),
                    Box::new(Expr::Var(end_tmp)),
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
                let tuple_tmp = self.alloc_temp_slot();
                let final_expr = Expr::TupleAccess(Box::new(Expr::Var(tuple_tmp)), *index);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(tuple_expr, Lhs::Var(tuple_tmp), assign_vertex)
            }

            TypedExprKind::FieldAccess(target_expr, field) => {
                let target_tmp = self.alloc_temp_slot();
                let final_expr = Expr::Find(
                    Box::new(Expr::Var(target_tmp)),
                    Box::new(Expr::String(EcoString::from(field.as_str()))),
                );
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(target_expr, Lhs::Var(target_tmp), assign_vertex)
            }

            TypedExprKind::UnwrapOptional(e) => {
                let e_tmp = self.alloc_temp_slot();
                let final_expr = Expr::Unwrap(Box::new(Expr::Var(e_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::WrapInOptional(e) => {
                let e_tmp = self.alloc_temp_slot();
                let final_expr = Expr::Some(Box::new(Expr::Var(e_tmp)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_to_value(e, Lhs::Var(e_tmp), assign_vertex)
            }

            TypedExprKind::StructLit(_, fields) => {
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
                self.compile_expr_list_recursive(val_exprs.iter(), tmps, assign_vertex)
            }

            TypedExprKind::SetTimer => self.add_label(Label::SetTimer(target, next_vertex)),

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
        call: &TypedUserFuncCall,
        target: Lhs, // The final destination for the future
        next_vertex: Vertex,
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
            entry_vertex = self.compile_expr_to_value(arg, Lhs::Var(*tmp_var), entry_vertex);
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
        match call {
            TypedFuncCall::User(user_call) => {
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
                    )
                } else {
                    let node_expr = Expr::Var(SELF_SLOT);
                    self.compile_async_call_internal(node_expr, user_call, target, next_vertex)
                }
            }
            TypedFuncCall::Builtin(builtin, args, _return_ty) => {
                // --- This is the new logic for built-ins ---
                self.compile_builtin_call(*builtin, args, target, next_vertex)
            }
        }
    }

    fn compile_builtin_call(
        &mut self,
        builtin: BuiltinFn,
        args: &Vec<TypedExpr>,
        target: Lhs,
        next_vertex: Vertex,
    ) -> Vertex {
        match builtin {
            BuiltinFn::Println => {
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex));

                let arg_var = self.alloc_temp_slot();
                let print_label = Label::Print(Expr::Var(arg_var), assign_vertex);
                let print_vertex = self.add_label(print_label);

                let arg_expr = args.first().expect("Println should have 1 arg");
                self.compile_expr_to_value(arg_expr, Lhs::Var(arg_var), print_vertex)
            }
            BuiltinFn::IntToString => {
                let arg_var = self.alloc_temp_slot();
                let final_expr = Expr::IntToString(Box::new(Expr::Var(arg_var)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));

                let arg_expr = args.first().expect("IntToString should have 1 arg");
                self.compile_expr_to_value(arg_expr, Lhs::Var(arg_var), assign_vertex)
            }
            BuiltinFn::BoolToString => {
                let arg_var = self.alloc_temp_slot();
                let final_expr = Expr::BoolToString(Box::new(Expr::Var(arg_var)));
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));

                let arg_expr = args.first().expect("BoolToString should have 1 arg");
                self.compile_expr_to_value(arg_expr, Lhs::Var(arg_var), assign_vertex)
            }
            BuiltinFn::UniqueId => {
                self.add_label(Label::UniqueId(target, next_vertex))
            }
        }
    }

    fn compile_rpc_call(
        &mut self,
        target_expr: &TypedExpr,
        call: &TypedUserFuncCall,
        target: Lhs,
        next_vertex: Vertex,
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
            Instr::Async(target, node_expr, func_name, arg_exprs),
            next_vertex,
        );
        let async_vertex = self.add_label(async_label);

        let args_entry_vertex =
            self.compile_expr_list_recursive(call.args.iter(), arg_tmps, async_vertex);

        self.compile_expr_to_value(target_expr, Lhs::Var(node_tmp), args_entry_vertex)
    }

    /// Converts a "simple" `TypedExpr` to a `cfg::Expr`.
    /// Panics if it encounters a complex (side-effecting) expression.
    fn convert_simple_expr(&mut self, expr: &TypedExpr) -> Expr {
        match &expr.kind {
            TypedExprKind::Var(id, _name) => Expr::Var(self.resolve_slot(*id)),
            TypedExprKind::IntLit(i) => Expr::Int(*i),
            TypedExprKind::StringLit(s) => Expr::String(EcoString::from(s.as_str())),
            TypedExprKind::BoolLit(b) => Expr::Bool(*b),
            TypedExprKind::NilLit => Expr::Nil,
            TypedExprKind::BinOp(op, l, r) => {
                let left = Box::new(self.convert_simple_expr(l));
                let right = Box::new(self.convert_simple_expr(r));
                self.convert_simple_binop(op, left, right)
            }
            TypedExprKind::Not(e) => Expr::Not(Box::new(self.convert_simple_expr(e))),
            TypedExprKind::Negate(e) => Expr::Minus(
                Box::new(Expr::Int(0)),
                Box::new(self.convert_simple_expr(e)),
            ),
            TypedExprKind::MapLit(pairs) => Expr::Map(
                pairs
                    .iter()
                    .map(|(k, v)| (self.convert_simple_expr(k), self.convert_simple_expr(v)))
                    .collect(),
            ),
            TypedExprKind::ListLit(items) => {
                Expr::List(items.iter().map(|e| self.convert_simple_expr(e)).collect())
            }
            TypedExprKind::TupleLit(items) => {
                Expr::Tuple(items.iter().map(|e| self.convert_simple_expr(e)).collect())
            }
            // Desugar: `append(l, i)` -> `EListAppend(l, i)`
            TypedExprKind::Append(l, i) => Expr::ListAppend(
                Box::new(self.convert_simple_expr(l)),
                Box::new(self.convert_simple_expr(i)),
            ),
            TypedExprKind::Prepend(i, l) => Expr::ListPrepend(
                Box::new(self.convert_simple_expr(i)),
                Box::new(self.convert_simple_expr(l)),
            ),
            // Desugar: `len(e)` -> `EListLen(e)`
            TypedExprKind::Len(e) => Expr::ListLen(Box::new(self.convert_simple_expr(e))),
            TypedExprKind::Min(l, r) => Expr::Min(
                Box::new(self.convert_simple_expr(l)),
                Box::new(self.convert_simple_expr(r)),
            ),
            TypedExprKind::Exists(map, key) => Expr::KeyExists(
                Box::new(self.convert_simple_expr(key)),
                Box::new(self.convert_simple_expr(map)),
            ),
            TypedExprKind::Erase(map, key) => Expr::MapErase(
                Box::new(self.convert_simple_expr(key)),
                Box::new(self.convert_simple_expr(map)),
            ),
            TypedExprKind::Store(collection, key, value) => Expr::Store(
                Box::new(self.convert_simple_expr(collection)),
                Box::new(self.convert_simple_expr(key)),
                Box::new(self.convert_simple_expr(value)),
            ),
            // Desugar: `head(l)` -> `EListAccess(l, 0)`
            TypedExprKind::Head(l) => Expr::ListAccess(Box::new(self.convert_simple_expr(l)), 0),
            // Desugar: `tail(l)` -> `EListSubsequence(l, 1, len(l))`
            TypedExprKind::Tail(l) => {
                let list_expr = self.convert_simple_expr(l);
                Expr::ListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::Int(1)),
                    Box::new(Expr::ListLen(Box::new(list_expr))),
                )
            }
            // Desugar: `s[i]` -> `EFind(s, i)`
            TypedExprKind::Index(target, index) => Expr::Find(
                Box::new(self.convert_simple_expr(target)),
                Box::new(self.convert_simple_expr(index)),
            ),
            // Desugar: `s[i:j]` -> `EListSubsequence(s, i, j)`
            TypedExprKind::Slice(target, start, end) => Expr::ListSubsequence(
                Box::new(self.convert_simple_expr(target)),
                Box::new(self.convert_simple_expr(start)),
                Box::new(self.convert_simple_expr(end)),
            ),
            TypedExprKind::TupleAccess(tuple, index) => {
                Expr::TupleAccess(Box::new(self.convert_simple_expr(tuple)), *index)
            }
            // Desugar: `s.field` -> `EFind(s, "field")`
            TypedExprKind::FieldAccess(target, field) => Expr::Find(
                Box::new(self.convert_simple_expr(target)),
                Box::new(Expr::String(EcoString::from(field.as_str()))),
            ),
            TypedExprKind::UnwrapOptional(e) => Expr::Unwrap(Box::new(self.convert_simple_expr(e))),

            TypedExprKind::WrapInOptional(e) => Expr::Some(Box::new(self.convert_simple_expr(e))),
            // Desugar: `MyStruct { ... }` -> `EMap { ... }`
            TypedExprKind::StructLit(_, fields) => Expr::Map(
                fields
                    .iter()
                    .map(|(name, val)| {
                        (
                            Expr::String(EcoString::from(name.as_str())),
                            self.convert_simple_expr(val),
                        )
                    })
                    .collect(),
            ),

            TypedExprKind::SetTimer => {
                unreachable!(
                    "SetTimer should be compiled as a Label, not an Expr. \
                     This should have been caught earlier. Expression: {:?}",
                    expr
                )
            }

            // --- Panic on complex expressions ---
            TypedExprKind::FuncCall(_)
            | TypedExprKind::RpcCall(_, _)
            | TypedExprKind::MakeChannel(_)
            | TypedExprKind::Send(_, _)
            | TypedExprKind::Recv(_) => {
                unreachable!(
                    "Cannot have complex call inside a simple expression. \
                     This should have been caught by the type checker. Expression: {:?}",
                    expr
                )
            }
        }
    }

    fn convert_lhs(&mut self, expr: &TypedExpr) -> Lhs {
        match &expr.kind {
            TypedExprKind::Var(id, _name) => Lhs::Var(self.resolve_slot(*id)),
            _ => unreachable!(
                "Invalid assignment target. \
                 This should have been caught by the type checker. Expression: {:?}",
                expr
            ),
        }
    }

    fn convert_pattern(&mut self, pat: &TypedPattern) -> Lhs {
        match &pat.kind {
            TypedPatternKind::Var(id, _name) => Lhs::Var(self.resolve_slot(*id)),
            TypedPatternKind::Wildcard => Lhs::Var(self.alloc_temp_slot()),
            TypedPatternKind::Tuple(pats) => {
                let names = pats
                    .iter()
                    .map(|p| match &p.kind {
                        TypedPatternKind::Var(id, _name) => self.resolve_slot(*id),
                        _ => self.alloc_temp_slot(), // Use dummy var for wildcard/unit
                    })
                    .collect();
                Lhs::Tuple(names)
            }
        }
    }
}

#[cfg(test)]
mod test;
