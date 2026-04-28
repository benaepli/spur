use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::Type;
use crate::parser::{BinOp, Span};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum PAtomic {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,
    Never,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PExpr {
    pub kind: PExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PExprKind {
    Atomic(PAtomic),

    BinOp(BinOp, PAtomic, PAtomic),
    Not(PAtomic),
    Negate(PAtomic),

    FuncCall(PFuncCall),

    MapLit(Vec<(PAtomic, PAtomic)>),
    ListLit(Vec<PAtomic>),
    TupleLit(Vec<PAtomic>),

    Append(PAtomic, PAtomic),
    Prepend(PAtomic, PAtomic),
    Min(PAtomic, PAtomic),
    Exists(PAtomic, PAtomic),
    Erase(PAtomic, PAtomic),
    Store(PAtomic, PAtomic, PAtomic),
    Head(PAtomic),
    Tail(PAtomic),
    Len(PAtomic),

    RpcCall(PAtomic, PUserFuncCall),

    Conditional(Box<PCondExpr>),
    Block(Box<PBlock>),

    VariantLit(NameId, String, Option<PAtomic>),
    IsVariant(PAtomic, String),
    VariantPayload(PAtomic),

    UnwrapOptional(PAtomic),

    MakeIter(PAtomic),
    IterIsDone(PAtomic),
    IterNext(PAtomic),

    MakeChannel,
    Send(PAtomic, PAtomic, PAtomic), // (state, current_chan, value)
    Recv(PAtomic, PAtomic),          // (state, current_chan)

    SetTimer(Option<String>),
    Fifo(PAtomic),

    Index(PAtomic, PAtomic),
    Slice(PAtomic, PAtomic, PAtomic),
    TupleAccess(PAtomic, usize),
    FieldAccess(PAtomic, String),

    SafeFieldAccess(PAtomic, String),
    SafeIndex(PAtomic, PAtomic),
    SafeTupleAccess(PAtomic, usize),

    StructLit(NameId, Vec<(String, PAtomic)>),

    WrapInOptional(PAtomic),
    PersistData(PAtomic),
    RetrieveData(Type),
    DiscardData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PFuncCall {
    User(PUserFuncCall),
    Builtin(BuiltinFn, Vec<PAtomic>, Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PUserFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<PAtomic>,
    pub return_type: Type,
    pub is_free: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PStatement {
    pub kind: PStatementKind,
    pub span: Span,
}

/// Pure IR statements. In SSA form.
#[derive(Debug, Clone, PartialEq)]
pub enum PStatementKind {
    LetAtom(PLetAtom),
    Expr(PExpr),
    Return(PAtomic),
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PLetAtom {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub value: PExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PBlock {
    pub statements: Vec<PStatement>,
    pub tail_expr: Option<PAtomic>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PCondExpr {
    pub if_branch: PIfBranch,
    pub elseif_branches: Vec<PIfBranch>,
    pub else_branch: Option<PBlock>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIfBranch {
    pub condition: PAtomic,
    pub body: PBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub span: Span,
}

/// User-written functions keep their Sync/Async tag from the source.
/// LoopConverted marks compiler-synthesized tail-recursive helpers produced
/// by lifting `Loop` statements. Lifted helpers are never user entry points;
/// their effective sync/async context is resolved through their single caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PFuncKind {
    Sync,
    Async,
    LoopConverted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub kind: PFuncKind,
    pub is_traced: bool,
    pub params: Vec<PFuncParam>,
    pub return_type: Type,
    pub body: PBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PRoleDef {
    pub name: NameId,
    pub original_name: String,
    pub func_defs: Vec<PFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PTopLevelDef {
    Role(PRoleDef),
    FreeFunc(PFuncDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PProgram {
    pub top_level_defs: Vec<PTopLevelDef>,
    pub next_name_id: usize,
    pub id_to_name: HashMap<NameId, String>,
    pub struct_defs: HashMap<NameId, Vec<(String, Type)>>,
    pub enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
}
