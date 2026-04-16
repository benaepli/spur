use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::Type;
use crate::parser::{BinOp, Span};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum TAtomic {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,
    Never,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TExpr {
    pub kind: TExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TExprKind {
    Atomic(TAtomic),

    BinOp(BinOp, TAtomic, TAtomic),
    Not(TAtomic),
    Negate(TAtomic),

    FuncCall(TFuncCall),

    MapLit(Vec<(TAtomic, TAtomic)>),
    ListLit(Vec<TAtomic>),
    TupleLit(Vec<TAtomic>),

    Append(TAtomic, TAtomic),
    Prepend(TAtomic, TAtomic),
    Min(TAtomic, TAtomic),
    Exists(TAtomic, TAtomic),
    Erase(TAtomic, TAtomic),
    Store(TAtomic, TAtomic, TAtomic),
    Head(TAtomic),
    Tail(TAtomic),
    Len(TAtomic),

    RpcCall(TAtomic, TUserFuncCall),

    Conditional(Box<TCondExpr>),
    Block(Box<TBlock>),

    VariantLit(NameId, String, Option<TAtomic>),
    IsVariant(TAtomic, String),
    VariantPayload(TAtomic),

    UnwrapOptional(TAtomic),

    MakeIter(TAtomic),
    IterIsDone(TAtomic),
    IterNext(TAtomic),

    MakeChannel,
    Send(TAtomic, TAtomic, TAtomic), // (state, current_chan, value)
    Recv(TAtomic, TAtomic),          // (state, current_chan)

    SetTimer(Option<String>),

    Index(TAtomic, TAtomic),
    Slice(TAtomic, TAtomic, TAtomic),
    TupleAccess(TAtomic, usize),
    FieldAccess(TAtomic, String),

    SafeFieldAccess(TAtomic, String),
    SafeIndex(TAtomic, TAtomic),
    SafeTupleAccess(TAtomic, usize),

    StructLit(NameId, Vec<(String, TAtomic)>),

    WrapInOptional(TAtomic),
    PersistData(TAtomic),
    RetrieveData(Type),
    DiscardData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TFuncCall {
    User(TUserFuncCall),
    Builtin(BuiltinFn, Vec<TAtomic>, Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TUserFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<TAtomic>,
    pub return_type: Type,
    pub is_free: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TLhsExpr {
    Var(NameId, String),
    Index(Box<TLhsExpr>, TAtomic),
    FieldAccess(Box<TLhsExpr>, String),
    TupleAccess(Box<TLhsExpr>, usize),
    SafeIndex(Box<TLhsExpr>, TAtomic),
    SafeFieldAccess(Box<TLhsExpr>, String),
    SafeTupleAccess(Box<TLhsExpr>, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TStatement {
    pub kind: TStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TStatementKind {
    LetAtom(TLetAtom),
    Assign(TAssign),
    Expr(TExpr),
    Loop(Vec<TStatement>),
    Return(TAtomic),
    Break,
    Continue,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TLetAtom {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub value: TExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TAssign {
    pub target: TLhsExpr,
    pub value: TAtomic,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TBlock {
    pub statements: Vec<TStatement>,
    pub tail_expr: Option<TAtomic>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TCondExpr {
    pub if_branch: TIfBranch,
    pub elseif_branches: Vec<TIfBranch>,
    pub else_branch: Option<TBlock>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TIfBranch {
    pub condition: TAtomic,
    pub body: TBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub is_sync: bool,
    pub is_traced: bool,
    pub params: Vec<TFuncParam>,
    pub return_type: Type,
    pub body: TBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TRoleDef {
    pub name: NameId,
    pub original_name: String,
    pub func_defs: Vec<TFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TTopLevelDef {
    Role(TRoleDef),
    FreeFunc(TFuncDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TProgram {
    pub top_level_defs: Vec<TTopLevelDef>,
    pub next_name_id: usize,
    pub id_to_name: HashMap<NameId, String>,
    pub struct_defs: HashMap<NameId, Vec<(String, Type)>>,
    pub enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
}
