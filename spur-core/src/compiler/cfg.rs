use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::type_id::{TypeId, TypeIdMap};
use crate::analysis::types::{
    TypedBlock, TypedCondExpr, TypedExpr, TypedExprKind, TypedForInLoop, TypedForLoop,
    TypedForLoopInit, TypedFuncCall, TypedFuncDef, TypedMatchArm, TypedPattern, TypedPatternKind,
    TypedProgram, TypedStatement, TypedStatementKind, TypedTopLevelDef, TypedUserFuncCall,
    TypedVarInit, TypedVarTarget,
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
    NodeToString(Box<Expr>),
    // Variant operations
    Variant(u32, EcoString, Option<Box<Expr>>), // (enum_id, variant_name, payload)
    IsVariant(Box<Expr>, EcoString),            // Check if value is a specific variant
    VariantPayload(Box<Expr>),                  // Extract payload from variant
    // Safe navigation: nil-check then access, wrapping result in Option
    SafeFind(Box<Expr>, Box<Expr>),
    SafeTupleAccess(Box<Expr>, usize),
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

/// Bundles the control-flow targets that are threaded through every
/// compilation call (break, continue, return, return slot).
#[derive(Clone, Copy)]
struct CompileCtx {
    break_target: Vertex,
    continue_target: Vertex,
    return_target: Vertex,
    return_slot: VarSlot,
}

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
    MakeChannel(Lhs, Option<usize>, Vertex),
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
    PersistData(TypeId, Expr, Vertex /* next_vertex */),
    RetrieveData(TypeId, Lhs, Vertex /* next_vertex */),
    DiscardData(Vertex /* next_vertex */),
    Break(Vertex /* break_target_vertex */),
    Continue(Vertex /* continue_target_vertex */),
    TraceDispatch(
        String,    /* func_name */
        Vec<Expr>, /* param exprs */
        Vertex,    /* next */
    ),
    TraceEnter(
        String,    /* func_name */
        Vec<Expr>, /* param exprs */
        Lhs,       /* trace_id_slot */
        Vertex,    /* next */
    ),
    TraceExit(
        String, /* func_name */
        Expr,   /* trace_id_expr */
        Expr,   /* return_val_expr */
        Vertex, /* next */
    ),
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

    /// Ordered list of role (NameId, original_name) pairs, in declaration order.
    pub roles: Vec<(NameId, String)>,

    /// Maps each structurally distinct Type to a unique TypeId.
    pub type_ids: TypeIdMap,
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

fn convert_binop(op: &BinOp, left: Box<Expr>, right: Box<Expr>) -> Expr {
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
    fn begin_role(&mut self, var_inits: &[TypedVarInit]) {
        self.node_slots.clear();
        self.next_node_slot = 1; // Slot 0 reserved for 'self'

        for init in var_inits {
            match &init.target {
                TypedVarTarget::Name(name, _) => {
                    self.node_slots.insert(*name, self.next_node_slot);
                    self.next_node_slot += 1;
                }
                TypedVarTarget::Tuple(elements) => {
                    for (name, _, _) in elements {
                        self.node_slots.insert(*name, self.next_node_slot);
                        self.next_node_slot += 1;
                    }
                }
            }
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
    pub fn compile_program(mut self, program: TypedProgram, type_ids: TypeIdMap) -> Program {
        let next_name_id = program.next_name_id;
        self.next_name_id = next_name_id;
        // Initialize id_to_name from the resolver's map
        self.id_to_name = program.id_to_name;
        // Store type_ids for use during compilation
        self.type_ids = type_ids;

        // Build func_sync_map and compile all top-level definitions
        for def in &program.top_level_defs {
            match def {
                TypedTopLevelDef::Role(role) => {
                    let qualifier = role.original_name.clone();
                    self.roles.push((role.name, qualifier.clone()));
                    for func in &role.func_defs {
                        self.func_sync_map.insert(func.name, func.is_sync);
                        self.func_traced_map.insert(func.name, func.is_traced);
                        self.func_qualifier_map.insert(func.name, qualifier.clone());
                    }
                }
                TypedTopLevelDef::FreeFunc(func) => {
                    self.func_sync_map.insert(func.name, true); // free functions are always sync
                    self.func_traced_map.insert(func.name, func.is_traced);
                    self.func_qualifier_map.insert(func.name, "__free".to_string());
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
                TypedTopLevelDef::FreeFunc(func) => {
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
        inits: &[TypedVarInit],
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
            let lhs = match &init.target {
                TypedVarTarget::Name(name, _) => Lhs::Var(self.resolve_slot(*name)),
                TypedVarTarget::Tuple(elements) => {
                    let slots = elements
                        .iter()
                        .map(|(name, _, _)| self.resolve_slot(*name))
                        .collect();
                    Lhs::Tuple(slots)
                }
            };

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

    fn compile_func_def(&mut self, func: TypedFuncDef) -> FunctionInfo {
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

        let body_entry = self.compile_typed_block(&func.body, Lhs::Var(return_slot), effective_return_target, ctx);

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
    fn scan_and_assign_slots(&mut self, body: &TypedBlock) {
        self.scan_block_and_assign_slots(body);
    }

    fn scan_block_and_assign_slots(&mut self, block: &TypedBlock) {
        for stmt in &block.statements {
            self.scan_stmt_and_assign_slots(stmt);
        }
        if let Some(tail) = &block.tail_expr {
            self.scan_expr_and_assign_slots(tail);
        }
    }

    fn scan_stmt_and_assign_slots(&mut self, stmt: &TypedStatement) {
        match &stmt.kind {
            TypedStatementKind::VarInit(init) => {
                match &init.target {
                    TypedVarTarget::Name(name, _) => {
                        let name_str = self
                            .id_to_name
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| format!("var_{}", name.0));
                        self.alloc_local_slot(*name, &name_str, Expr::Nil);
                    }
                    TypedVarTarget::Tuple(elements) => {
                        for (name, _, _) in elements {
                            let name_str = self
                                .id_to_name
                                .get(name)
                                .cloned()
                                .unwrap_or_else(|| format!("var_{}", name.0));
                            self.alloc_local_slot(*name, &name_str, Expr::Nil);
                        }
                    }
                }
                self.scan_expr_and_assign_slots(&init.value);
            }
            TypedStatementKind::Assignment(assign) => {
                self.scan_expr_and_assign_slots(&assign.target);
                self.scan_expr_and_assign_slots(&assign.value);
            }
            TypedStatementKind::Expr(expr) => {
                self.scan_expr_and_assign_slots(expr);
            }
            TypedStatementKind::ForLoop(fl) => {
                match &fl.init {
                    Some(TypedForLoopInit::VarInit(init)) => {
                        match &init.target {
                            TypedVarTarget::Name(name, _) => {
                                let name_str = self
                                    .id_to_name
                                    .get(name)
                                    .cloned()
                                    .unwrap_or_else(|| format!("var_{}", name.0));
                                self.alloc_local_slot(*name, &name_str, Expr::Nil);
                            }
                            TypedVarTarget::Tuple(elements) => {
                                for (name, _, _) in elements {
                                    let name_str = self
                                        .id_to_name
                                        .get(name)
                                        .cloned()
                                        .unwrap_or_else(|| format!("var_{}", name.0));
                                    self.alloc_local_slot(*name, &name_str, Expr::Nil);
                                }
                            }
                        }
                        self.scan_expr_and_assign_slots(&init.value);
                    }
                    Some(TypedForLoopInit::Assignment(assign)) => {
                        self.scan_expr_and_assign_slots(&assign.target);
                        self.scan_expr_and_assign_slots(&assign.value);
                    }
                    None => {}
                }
                if let Some(cond) = &fl.condition {
                    self.scan_expr_and_assign_slots(cond);
                }
                if let Some(inc) = &fl.increment {
                    self.scan_expr_and_assign_slots(&inc.target);
                    self.scan_expr_and_assign_slots(&inc.value);
                }
                self.scan_body_and_assign_slots(&fl.body);
            }
            TypedStatementKind::ForInLoop(loop_stmt) => {
                self.scan_pattern_and_assign_slots(&loop_stmt.pattern);
                self.scan_expr_and_assign_slots(&loop_stmt.iterable);
                self.scan_body_and_assign_slots(&loop_stmt.body);
            }
            TypedStatementKind::Error => {}
        }
    }

    fn scan_body_and_assign_slots(&mut self, body: &[TypedStatement]) {
        for stmt in body {
            self.scan_stmt_and_assign_slots(stmt);
        }
    }

    fn scan_expr_and_assign_slots(&mut self, expr: &TypedExpr) {
        use crate::analysis::types::TypedExprKind::*;
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
                self.scan_expr_and_assign_slots(l);
                self.scan_expr_and_assign_slots(r);
            }
            PersistData(e) => {
                self.scan_expr_and_assign_slots(e);
            }
            RetrieveData(_) => {}
            DiscardData => {}
            Not(e)
            | Negate(e)
            | Head(e)
            | Tail(e)
            | Len(e)
            | UnwrapOptional(e)
            | Recv(e)
            | TupleAccess(e, _)
            | FieldAccess(e, _)
            | SafeFieldAccess(e, _)
            | SafeTupleAccess(e, _)
            | WrapInOptional(e) => {
                self.scan_expr_and_assign_slots(e);
            }
            MakeChannel => {}
            FuncCall(call) => match call {
                TypedFuncCall::User(u) => {
                    for arg in &u.args {
                        self.scan_expr_and_assign_slots(arg);
                    }
                }
                TypedFuncCall::Builtin(_, args, _) => {
                    for arg in args {
                        self.scan_expr_and_assign_slots(arg);
                    }
                }
            },
            MapLit(pairs) => {
                for (k, v) in pairs {
                    self.scan_expr_and_assign_slots(k);
                    self.scan_expr_and_assign_slots(v);
                }
            }
            ListLit(exprs) | TupleLit(exprs) => {
                for e in exprs {
                    self.scan_expr_and_assign_slots(e);
                }
            }
            Slice(a, b, c) | Store(a, b, c) => {
                self.scan_expr_and_assign_slots(a);
                self.scan_expr_and_assign_slots(b);
                self.scan_expr_and_assign_slots(c);
            }
            RpcCall(target, call) => {
                self.scan_expr_and_assign_slots(target);
                for arg in &call.args {
                    self.scan_expr_and_assign_slots(arg);
                }
            }
            Match(scrutinee, arms) => {
                self.scan_expr_and_assign_slots(scrutinee);
                for arm in arms {
                    self.scan_pattern_and_assign_slots(&arm.pattern);
                    self.scan_block_and_assign_slots(&arm.body);
                }
            }
            Conditional(cond) => {
                self.scan_expr_and_assign_slots(&cond.if_branch.condition);
                self.scan_block_and_assign_slots(&cond.if_branch.body);
                for branch in &cond.elseif_branches {
                    self.scan_expr_and_assign_slots(&branch.condition);
                    self.scan_block_and_assign_slots(&branch.body);
                }
                if let Some(body) = &cond.else_branch {
                    self.scan_block_and_assign_slots(body);
                }
            }
            Block(block) => {
                self.scan_block_and_assign_slots(block);
            }
            VariantLit(_, _, payload) => {
                if let Some(p) = payload {
                    self.scan_expr_and_assign_slots(p);
                }
            }
            StructLit(_, fields) => {
                for (_, e) in fields {
                    self.scan_expr_and_assign_slots(e);
                }
            }
            Return(inner) => {
                self.scan_expr_and_assign_slots(inner);
            }
            Break | Continue => {}
            Var(_, _) | IntLit(_) | StringLit(_) | BoolLit(_) | NilLit | SetTimer => {}
            Error => {}
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
            TypedPatternKind::Variant(_, _, payload) => {
                if let Some(p) = payload {
                    self.scan_pattern_and_assign_slots(p);
                }
            }
            _ => {} // Wildcard doesn't add a named local
        }
    }

    fn compile_tailless_block(
        &mut self,
        body: &[TypedStatement],
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let mut next = next_vertex;
        for stmt in body.iter().rev() {
            next = self.compile_statement(stmt, next, ctx);
        }
        next
    }

    fn compile_typed_block(
        &mut self,
        block: &TypedBlock,
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
        stmt: &TypedStatement,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let prev_span = self.current_span;
        self.current_span = Some(stmt.span);

        let result = match &stmt.kind {
            TypedStatementKind::VarInit(init) => {
                let lhs = match &init.target {
                    TypedVarTarget::Name(name, _) => Lhs::Var(self.resolve_slot(*name)),
                    TypedVarTarget::Tuple(elements) => {
                        let slots = elements
                            .iter()
                            .map(|(name, _, _)| self.resolve_slot(*name))
                            .collect();
                        Lhs::Tuple(slots)
                    }
                };
                self.compile_expr_to_value(&init.value, lhs, next_vertex, ctx)
            }
            TypedStatementKind::Assignment(assign) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, next_vertex, ctx)
            }
            TypedStatementKind::Expr(expr) => match &expr.kind {
                TypedExprKind::Return(inner) => {
                    self.compile_expr_to_value(
                        inner,
                        Lhs::Var(ctx.return_slot),
                        ctx.return_target,
                        ctx,
                    )
                }
                TypedExprKind::Break => self.add_label(Label::Break(ctx.break_target)),
                TypedExprKind::Continue => self.add_label(Label::Continue(ctx.continue_target)),
                _ => {
                    let dummy_slot = self.alloc_temp_slot();
                    self.compile_expr_to_value(expr, Lhs::Var(dummy_slot), next_vertex, ctx)
                }
            }
            TypedStatementKind::ForLoop(loop_stmt) => {
                self.compile_for_loop(loop_stmt, next_vertex, ctx)
            }
            TypedStatementKind::ForInLoop(loop_stmt) => {
                self.compile_for_in_loop(loop_stmt, next_vertex, ctx)
            }
            TypedStatementKind::Error => {
                unreachable!("Error statements should not reach CFG generation")
            }
        };

        self.current_span = prev_span;
        result
    }

    fn compile_for_loop(
        &mut self,
        loop_stmt: &TypedForLoop,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let loop_head_vertex = self.add_label(Label::Return(Expr::Unit));

        // Inside the loop, break/continue targets are overridden
        let loop_ctx_base = CompileCtx {
            break_target: next_vertex,
            continue_target: next_vertex, // placeholder, updated below
            ..ctx
        };

        let increment_vertex = match &loop_stmt.increment {
            Some(assign) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, loop_head_vertex, loop_ctx_base)
            }
            None => loop_head_vertex,
        };

        let loop_ctx = CompileCtx {
            continue_target: increment_vertex,
            ..loop_ctx_base
        };

        let body_vertex = self.compile_tailless_block(&loop_stmt.body, increment_vertex, loop_ctx);

        let cond_slot = self.alloc_temp_slot();

        let cond_check_vertex =
            self.add_label(Label::Cond(Expr::Var(cond_slot), body_vertex, next_vertex));

        let cond_expr = loop_stmt.condition.as_ref().cloned().unwrap_or({
            TypedExpr {
                kind: TypedExprKind::BoolLit(true),
                ty: crate::analysis::types::Type::Bool,
                span: loop_stmt.span,
            }
        });

        let cond_calc_vertex = self.compile_expr_to_value(
            &cond_expr,
            Lhs::Var(cond_slot),
            cond_check_vertex,
            loop_ctx_base,
        );

        let dummy_slot = self.alloc_temp_slot();
        self.cfg[loop_head_vertex] = Label::Instr(
            Instr::Assign(Lhs::Var(dummy_slot), Expr::Unit),
            cond_calc_vertex,
        );

        match &loop_stmt.init {
            Some(TypedForLoopInit::VarInit(vi)) => {
                let lhs = match &vi.target {
                    TypedVarTarget::Name(name, _) => Lhs::Var(self.resolve_slot(*name)),
                    TypedVarTarget::Tuple(elements) => {
                        let slots = elements
                            .iter()
                            .map(|(name, _, _)| self.resolve_slot(*name))
                            .collect();
                        Lhs::Tuple(slots)
                    }
                };
                self.compile_expr_to_value(&vi.value, lhs, loop_head_vertex, loop_ctx_base)
            }
            Some(TypedForLoopInit::Assignment(assign)) => {
                let lhs = self.convert_lhs(&assign.target);
                self.compile_expr_to_value(&assign.value, lhs, loop_head_vertex, loop_ctx_base)
            }
            None => loop_head_vertex,
        }
    }

    fn compile_conditional(
        &mut self,
        cond: &TypedCondExpr,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let else_vertex = if let Some(body) = &cond.else_branch {
            self.compile_typed_block(body, target.clone(), next_vertex, ctx)
        } else {
            self.add_label(Label::Instr(Instr::Assign(target.clone(), Expr::Unit), next_vertex))
        };

        let mut next_cond_vertex = else_vertex;
        for branch in cond.elseif_branches.iter().rev() {
            let body_vertex = self.compile_typed_block(&branch.body, target.clone(), next_vertex, ctx);
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

        let if_body_vertex = self.compile_typed_block(&cond.if_branch.body, target.clone(), next_vertex, ctx);
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

    fn compile_for_in_loop(
        &mut self,
        loop_stmt: &TypedForInLoop,
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

        // Now we create the real [ForLoopIn] label.
        let lhs = self.convert_pattern(&loop_stmt.pattern);
        // Allocate a slot for the iterator state (the remaining collection)
        let iter_state_slot = self.alloc_temp_slot();
        let iterable_expr = self
            .try_to_expr(&loop_stmt.iterable)
            .expect("for-in iterable must be a simple expression");

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

    fn compile_expr_list_recursive<'a, I>(
        &mut self,
        exprs: I,
        slots: Vec<VarSlot>,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex
    where
        I: DoubleEndedIterator<Item = &'a TypedExpr>,
    {
        let mut next = next_vertex;
        for (expr, slot) in exprs.rev().zip(slots.iter().rev()) {
            next = self.compile_expr_to_value(expr, Lhs::Var(*slot), next, ctx);
        }
        next
    }

    fn compile_match(
        &mut self,
        scrutinee: &TypedExpr,
        arms: &[TypedMatchArm],
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        let scrutinee_slot = self.alloc_temp_slot();
        let mut current_else_vertex = next_vertex;

        for arm in arms.iter().rev() {
            let (variant_name, payload_pattern) = match &arm.pattern.kind {
                TypedPatternKind::Variant(_, name, payload_pat) => {
                    (name.clone(), payload_pat.clone())
                }
                _ => {
                    let body_vertex =
                        self.compile_match_arm_body(&arm.body, target.clone(), next_vertex, ctx);
                    current_else_vertex = body_vertex;
                    continue;
                }
            };

            let body_start = if let Some(payload_pat) = payload_pattern {
                let payload_lhs = self.convert_pattern(&payload_pat);
                let body_vertex =
                    self.compile_match_arm_body(&arm.body, target.clone(), next_vertex, ctx);
                self.add_label(Label::Instr(
                    Instr::Assign(
                        payload_lhs,
                        Expr::VariantPayload(Box::new(Expr::Var(scrutinee_slot))),
                    ),
                    body_vertex,
                ))
            } else {
                self.compile_match_arm_body(&arm.body, target.clone(), next_vertex, ctx)
            };

            let cond_vertex = self.add_label(Label::Cond(
                Expr::IsVariant(
                    Box::new(Expr::Var(scrutinee_slot)),
                    EcoString::from(variant_name.as_str()),
                ),
                body_start,
                current_else_vertex,
            ));

            current_else_vertex = cond_vertex;
        }

        self.compile_expr_to_value(
            scrutinee,
            Lhs::Var(scrutinee_slot),
            current_else_vertex,
            ctx,
        )
    }

    fn compile_match_arm_body(
        &mut self,
        body: &TypedBlock,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        self.compile_typed_block(body, target, next_vertex, ctx)
    }

    /// Compile a single sub-expression, apply `make_expr`, assign to `target`.
    fn compile_unary(
        &mut self,
        e: &TypedExpr,
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
        l: &TypedExpr,
        r: &TypedExpr,
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
        a: &TypedExpr,
        b: &TypedExpr,
        c: &TypedExpr,
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
        expr: &TypedExpr,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
    ) -> Vertex {
        // Fast path: if the whole expression is pure, convert directly
        if let Some(simple) = self.try_to_expr(expr) {
            return self.add_label(Label::Instr(Instr::Assign(target, simple), next_vertex));
        }

        match &expr.kind {
            TypedExprKind::FuncCall(call) => self.compile_func_call(call, target, next_vertex, ctx),
            TypedExprKind::RpcCall(target_expr, call) => {
                self.compile_rpc_call(target_expr, call, target, next_vertex, ctx)
            }
            TypedExprKind::MakeChannel => {
                self.add_label(Label::MakeChannel(target, None, next_vertex))
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
                    self.compile_expr_to_value(val_expr, Lhs::Var(val_tmp), send_vertex, ctx);
                self.compile_expr_to_value(chan_expr, Lhs::Var(chan_tmp), val_vertex, ctx)
            }
            TypedExprKind::Recv(chan_expr) => {
                let chan_tmp = self.alloc_temp_slot();
                let recv_vertex =
                    self.add_label(Label::Recv(target, Expr::Var(chan_tmp), next_vertex));
                self.compile_expr_to_value(chan_expr, Lhs::Var(chan_tmp), recv_vertex, ctx)
            }

            TypedExprKind::BinOp(op, l, r) => {
                match op {
                    BinOp::And => {
                        let assign_false_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target.clone(), Expr::Bool(false)),
                            next_vertex,
                        ));

                        let eval_right_vertex =
                            self.compile_expr_to_value(r, target.clone(), next_vertex, ctx);

                        // Branch based on Left result
                        let l_tmp = self.alloc_temp_slot();
                        let cond_vertex = self.add_label(Label::Cond(
                            Expr::Var(l_tmp),
                            eval_right_vertex,   // If True: evaluate right
                            assign_false_vertex, // If False: short-circuit to false
                        ));

                        self.compile_expr_to_value(l, Lhs::Var(l_tmp), cond_vertex, ctx)
                    }
                    BinOp::Or => {
                        let assign_true_vertex = self.add_label(Label::Instr(
                            Instr::Assign(target.clone(), Expr::Bool(true)),
                            next_vertex,
                        ));

                        let eval_right_vertex =
                            self.compile_expr_to_value(r, target.clone(), next_vertex, ctx);

                        // Branch based on Left result
                        let l_tmp = self.alloc_temp_slot();
                        let cond_vertex = self.add_label(Label::Cond(
                            Expr::Var(l_tmp),
                            assign_true_vertex, // If True: short-circuit to true
                            eval_right_vertex,  // If False: evaluate right
                        ));

                        self.compile_expr_to_value(l, Lhs::Var(l_tmp), cond_vertex, ctx)
                    }
                    _ => {
                        let op = op.clone();
                        self.compile_binary(l, r, target, next_vertex, ctx,
                            |l, r| convert_binop(&op, l, r))
                    }
                }
            }

            TypedExprKind::Not(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::Not)
            }
            TypedExprKind::Negate(e) => {
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::Minus(Box::new(Expr::Int(0)), v))
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
                    ctx,
                );

                self.compile_expr_list_recursive(
                    pairs.iter().map(|(_k, v)| v),
                    val_tmps,
                    keys_entry_vertex,
                    ctx,
                )
            }

            TypedExprKind::ListLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(items.len());
                let final_expr = Expr::List(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex, ctx)
            }

            TypedExprKind::TupleLit(items) => {
                let (tmps, simple_exprs) = self.compile_temp_list(items.len());
                let final_expr = Expr::Tuple(simple_exprs);
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, final_expr), next_vertex));
                self.compile_expr_list_recursive(items.iter(), tmps, assign_vertex, ctx)
            }

            TypedExprKind::Append(l, i) => {
                self.compile_binary(l, i, target, next_vertex, ctx, Expr::ListAppend)
            }
            TypedExprKind::Prepend(i, l) => {
                self.compile_binary(i, l, target, next_vertex, ctx, Expr::ListPrepend)
            }
            TypedExprKind::Len(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::ListLen)
            }
            TypedExprKind::Min(l, r) => {
                self.compile_binary(l, r, target, next_vertex, ctx, Expr::Min)
            }
            TypedExprKind::Exists(map, key) => {
                self.compile_binary(map, key, target, next_vertex, ctx,
                    |l, r| Expr::KeyExists(r, l))
            }
            TypedExprKind::Erase(map, key) => {
                self.compile_binary(map, key, target, next_vertex, ctx,
                    |l, r| Expr::MapErase(r, l))
            }
            TypedExprKind::Head(e) => {
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::ListAccess(v, 0))
            }
            TypedExprKind::Tail(e) => {
                self.compile_unary(e, target, next_vertex, ctx, |v| {
                    Expr::ListSubsequence(
                        v.clone(),
                        Box::new(Expr::Int(1)),
                        Box::new(Expr::ListLen(v)),
                    )
                })
            }
            TypedExprKind::Index(a, b) => {
                self.compile_binary(a, b, target, next_vertex, ctx, Expr::Find)
            }
            TypedExprKind::Slice(a, b, c) => {
                self.compile_ternary(a, b, c, target, next_vertex, ctx, Expr::ListSubsequence)
            }
            TypedExprKind::TupleAccess(e, i) => {
                let i = *i;
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::TupleAccess(v, i))
            }
            TypedExprKind::FieldAccess(e, field) => {
                let field = EcoString::from(field.as_str());
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::Find(v, Box::new(Expr::String(field))))
            }
            TypedExprKind::UnwrapOptional(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::Unwrap)
            }
            TypedExprKind::WrapInOptional(e) => {
                self.compile_unary(e, target, next_vertex, ctx, Expr::Some)
            }
            TypedExprKind::SafeFieldAccess(e, field) => {
                let field = EcoString::from(field.as_str());
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::SafeFind(v, Box::new(Expr::String(field))))
            }
            TypedExprKind::SafeIndex(a, b) => {
                self.compile_binary(a, b, target, next_vertex, ctx, Expr::SafeFind)
            }
            TypedExprKind::SafeTupleAccess(e, i) => {
                let i = *i;
                self.compile_unary(e, target, next_vertex, ctx,
                    |v| Expr::SafeTupleAccess(v, i))
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
                self.compile_expr_list_recursive(val_exprs.iter(), tmps, assign_vertex, ctx)
            }

            TypedExprKind::VariantLit(enum_id, variant_name, Some(payload_expr)) => {
                let eid = enum_id.0 as u32;
                let vname = EcoString::from(variant_name.as_str());
                self.compile_unary(payload_expr, target, next_vertex, ctx,
                    |v| Expr::Variant(eid, vname, Some(v)))
            }

            TypedExprKind::Match(scrutinee, arms) => {
                self.compile_match(scrutinee, arms, target, next_vertex, ctx)
            }

            TypedExprKind::Conditional(cond) => {
                self.compile_conditional(cond, target, next_vertex, ctx)
            }

            TypedExprKind::Block(block) => {
                self.compile_typed_block(block, target, next_vertex, ctx)
            }

            TypedExprKind::Store(a, b, c) => {
                self.compile_ternary(a, b, c, target, next_vertex, ctx, Expr::Store)
            }

            TypedExprKind::SetTimer => self.add_label(Label::SetTimer(target, next_vertex)),

            TypedExprKind::PersistData(e) => {
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex));

                let e_tmp = self.alloc_temp_slot();
                let type_id = self.type_ids.get(&e.ty).copied().unwrap_or(TypeId(0));
                let persist_label = Label::PersistData(type_id, Expr::Var(e_tmp), assign_vertex);
                let persist_vertex = self.add_label(persist_label);

                self.compile_expr_to_value(e, Lhs::Var(e_tmp), persist_vertex, ctx)
            }
            TypedExprKind::RetrieveData(inner_type) => {
                let type_id = self.type_ids.get(inner_type).copied().unwrap_or(TypeId(0));
                self.add_label(Label::RetrieveData(type_id, target, next_vertex))
            }
            TypedExprKind::DiscardData => {
                let assign_vertex =
                    self.add_label(Label::Instr(Instr::Assign(target, Expr::Unit), next_vertex));
                let discard_label = Label::DiscardData(assign_vertex);
                self.add_label(discard_label)
            }

            // Control flow expressions — handled at statement level in compile_statement,
            // but can appear nested in expression position.
            TypedExprKind::Return(inner) => {
                self.compile_expr_to_value(
                    inner,
                    Lhs::Var(ctx.return_slot),
                    ctx.return_target,
                    ctx,
                )
            }
            TypedExprKind::Break => self.add_label(Label::Break(ctx.break_target)),
            TypedExprKind::Continue => self.add_label(Label::Continue(ctx.continue_target)),

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
        call: &TypedUserFuncCall,
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
        call: &TypedFuncCall,
        target: Lhs,
        next_vertex: Vertex,
        ctx: CompileCtx,
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
                        ctx,
                    )
                } else {
                    let node_expr = Expr::Var(SELF_SLOT);
                    self.compile_async_call_internal(node_expr, user_call, target, next_vertex, ctx)
                }
            }
            TypedFuncCall::Builtin(builtin, args, _return_ty) => {
                // --- This is the new logic for built-ins ---
                self.compile_builtin_call(*builtin, args, target, next_vertex, ctx)
            }
        }
    }

    fn compile_builtin_call(
        &mut self,
        builtin: BuiltinFn,
        args: &Vec<TypedExpr>,
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
        target_expr: &TypedExpr,
        call: &TypedUserFuncCall,
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

    /// Tries to convert a `TypedExpr` into a pure `cfg::Expr`.
    /// Returns `None` for expressions that require CFG control flow
    /// (function calls, short-circuit logic, match, channels, etc.).
    fn try_to_expr(&mut self, expr: &TypedExpr) -> Option<Expr> {
        Some(match &expr.kind {
            TypedExprKind::Var(id, _name) => Expr::Var(self.resolve_slot(*id)),
            TypedExprKind::IntLit(i) => Expr::Int(*i),
            TypedExprKind::StringLit(s) => Expr::String(EcoString::from(s.as_str())),
            TypedExprKind::BoolLit(b) => Expr::Bool(*b),
            TypedExprKind::NilLit => Expr::Nil,
            TypedExprKind::BinOp(op, l, r) => {
                match op {
                    BinOp::And | BinOp::Or => return None,
                    _ => {}
                }
                let left = Box::new(self.try_to_expr(l)?);
                let right = Box::new(self.try_to_expr(r)?);
                convert_binop(op, left, right)
            }
            TypedExprKind::Not(e) => Expr::Not(Box::new(self.try_to_expr(e)?)),
            TypedExprKind::Negate(e) => {
                Expr::Minus(Box::new(Expr::Int(0)), Box::new(self.try_to_expr(e)?))
            }
            TypedExprKind::MapLit(pairs) => {
                let mut out = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    out.push((self.try_to_expr(k)?, self.try_to_expr(v)?));
                }
                Expr::Map(out)
            }
            TypedExprKind::ListLit(items) => {
                let mut out = Vec::with_capacity(items.len());
                for e in items {
                    out.push(self.try_to_expr(e)?);
                }
                Expr::List(out)
            }
            TypedExprKind::TupleLit(items) => {
                let mut out = Vec::with_capacity(items.len());
                for e in items {
                    out.push(self.try_to_expr(e)?);
                }
                Expr::Tuple(out)
            }
            TypedExprKind::Append(l, i) => Expr::ListAppend(
                Box::new(self.try_to_expr(l)?),
                Box::new(self.try_to_expr(i)?),
            ),
            TypedExprKind::Prepend(i, l) => Expr::ListPrepend(
                Box::new(self.try_to_expr(i)?),
                Box::new(self.try_to_expr(l)?),
            ),
            TypedExprKind::Len(e) => Expr::ListLen(Box::new(self.try_to_expr(e)?)),
            TypedExprKind::Min(l, r) => Expr::Min(
                Box::new(self.try_to_expr(l)?),
                Box::new(self.try_to_expr(r)?),
            ),
            TypedExprKind::Exists(map, key) => Expr::KeyExists(
                Box::new(self.try_to_expr(key)?),
                Box::new(self.try_to_expr(map)?),
            ),
            TypedExprKind::Erase(map, key) => Expr::MapErase(
                Box::new(self.try_to_expr(key)?),
                Box::new(self.try_to_expr(map)?),
            ),
            TypedExprKind::Store(collection, key, value) => Expr::Store(
                Box::new(self.try_to_expr(collection)?),
                Box::new(self.try_to_expr(key)?),
                Box::new(self.try_to_expr(value)?),
            ),
            TypedExprKind::Head(l) => Expr::ListAccess(Box::new(self.try_to_expr(l)?), 0),
            TypedExprKind::Tail(l) => {
                let list_expr = self.try_to_expr(l)?;
                Expr::ListSubsequence(
                    Box::new(list_expr.clone()),
                    Box::new(Expr::Int(1)),
                    Box::new(Expr::ListLen(Box::new(list_expr))),
                )
            }
            TypedExprKind::Index(target, index) => Expr::Find(
                Box::new(self.try_to_expr(target)?),
                Box::new(self.try_to_expr(index)?),
            ),
            TypedExprKind::Slice(target, start, end) => Expr::ListSubsequence(
                Box::new(self.try_to_expr(target)?),
                Box::new(self.try_to_expr(start)?),
                Box::new(self.try_to_expr(end)?),
            ),
            TypedExprKind::TupleAccess(tuple, index) => {
                Expr::TupleAccess(Box::new(self.try_to_expr(tuple)?), *index)
            }
            TypedExprKind::FieldAccess(target, field) => Expr::Find(
                Box::new(self.try_to_expr(target)?),
                Box::new(Expr::String(EcoString::from(field.as_str()))),
            ),
            TypedExprKind::UnwrapOptional(e) => Expr::Unwrap(Box::new(self.try_to_expr(e)?)),
            TypedExprKind::WrapInOptional(e) => Expr::Some(Box::new(self.try_to_expr(e)?)),
            TypedExprKind::SafeFieldAccess(target, field) => Expr::SafeFind(
                Box::new(self.try_to_expr(target)?),
                Box::new(Expr::String(EcoString::from(field.as_str()))),
            ),
            TypedExprKind::SafeIndex(target, index) => Expr::SafeFind(
                Box::new(self.try_to_expr(target)?),
                Box::new(self.try_to_expr(index)?),
            ),
            TypedExprKind::SafeTupleAccess(tuple, index) => {
                Expr::SafeTupleAccess(Box::new(self.try_to_expr(tuple)?), *index)
            }
            TypedExprKind::StructLit(_, fields) => {
                let mut out = Vec::with_capacity(fields.len());
                for (name, val) in fields {
                    out.push((
                        Expr::String(EcoString::from(name.as_str())),
                        self.try_to_expr(val)?,
                    ));
                }
                Expr::Map(out)
            }
            TypedExprKind::VariantLit(enum_id, variant_name, None) => Expr::Variant(
                enum_id.0 as u32,
                EcoString::from(variant_name.as_str()),
                None,
            ),

            // Control flow expressions — cannot be inlined into pure expressions
            TypedExprKind::Return(_)
            | TypedExprKind::Break
            | TypedExprKind::Continue => return None,

            // Complex expressions that need CFG control flow
            TypedExprKind::FuncCall(_)
            | TypedExprKind::RpcCall(_, _)
            | TypedExprKind::MakeChannel
            | TypedExprKind::Send(_, _)
            | TypedExprKind::Recv(_)
            | TypedExprKind::Match(_, _)
            | TypedExprKind::Conditional(_)
            | TypedExprKind::Block(_)
            | TypedExprKind::PersistData(_)
            | TypedExprKind::RetrieveData(_)
            | TypedExprKind::DiscardData
            | TypedExprKind::SetTimer
            | TypedExprKind::VariantLit(_, _, Some(_))
            | TypedExprKind::Error => return None,
        })
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
            TypedPatternKind::Variant(_, _, _) => {
                // TODO: Implement variant pattern matching in CFG
                // For now, we use a temp slot; match compilation needs to be added
                Lhs::Var(self.alloc_temp_slot())
            }
            TypedPatternKind::Error => unreachable!("Error pattern during CFG generation"),
        }
    }
}

#[cfg(test)]
mod test;
