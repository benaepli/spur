use crate::analysis::resolver::NameId;
use crate::analysis::type_id::{TypeId, TypeIdMap};
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
    MakeChannel(Lhs, Option<usize>, Vertex),
    SetTimer(Lhs, Vertex, Option<String>),
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

pub fn convert_binop(op: &BinOp, left: Box<Expr>, right: Box<Expr>) -> Expr {
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
