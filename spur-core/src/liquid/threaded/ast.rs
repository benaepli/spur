use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::Type;
use crate::parser::{BinOp, Span};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct TExpr {
    pub kind: TExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TExprKind {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,

    BinOp(BinOp, Box<TExpr>, Box<TExpr>),
    Not(Box<TExpr>),
    Negate(Box<TExpr>),

    FuncCall(TFuncCall),
    MapLit(Vec<(TExpr, TExpr)>),
    ListLit(Vec<TExpr>),
    TupleLit(Vec<TExpr>),

    Append(Box<TExpr>, Box<TExpr>),
    Prepend(Box<TExpr>, Box<TExpr>),
    Min(Box<TExpr>, Box<TExpr>),
    Exists(Box<TExpr>, Box<TExpr>),
    Erase(Box<TExpr>, Box<TExpr>),
    Store(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    Head(Box<TExpr>),
    Tail(Box<TExpr>),
    Len(Box<TExpr>),

    RpcCall(Box<TExpr>, TUserFuncCall),

    Conditional(Box<TCondExpr>),
    Block(Box<TBlock>),
    VariantLit(NameId, String, Option<Box<TExpr>>),

    IsVariant(Box<TExpr>, String),
    VariantPayload(Box<TExpr>),

    UnwrapOptional(Box<TExpr>),

    MakeIter(Box<TExpr>),
    IterIsDone(Box<TExpr>),
    IterNext(Box<TExpr>),

    MakeChannel,
    Send(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    Recv(Box<TExpr>, Box<TExpr>),

    SetTimer(Option<String>),

    Index(Box<TExpr>, Box<TExpr>),
    Slice(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    TupleAccess(Box<TExpr>, usize),
    FieldAccess(Box<TExpr>, String),

    SafeFieldAccess(Box<TExpr>, String),
    SafeIndex(Box<TExpr>, Box<TExpr>),
    SafeTupleAccess(Box<TExpr>, usize),

    StructLit(NameId, Vec<(String, TExpr)>),

    WrapInOptional(Box<TExpr>),
    PersistData(Box<TExpr>),
    RetrieveData(Type),
    DiscardData,

    Return(Box<TExpr>),
    Break,
    Continue,

    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TFuncCall {
    User(TUserFuncCall),
    Builtin(BuiltinFn, Vec<TExpr>, Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TUserFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<TExpr>,
    pub return_type: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TStatement {
    pub kind: TStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TStatementKind {
    VarInit(TVarInit),
    Assignment(TAssignment),
    Expr(TExpr),
    Loop(Vec<TStatement>),
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TBlock {
    pub statements: Vec<TStatement>,
    pub tail_expr: Option<Box<TExpr>>,
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
    pub condition: TExpr,
    pub body: TBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TVarInit {
    pub name: NameId,
    pub original_name: String,
    pub type_def: Type,
    pub value: TExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TAssignment {
    pub target: TExpr,
    pub value: TExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TProgram {
    pub top_level_defs: Vec<TTopLevelDef>,
    pub next_name_id: usize,
    pub id_to_name: HashMap<NameId, String>,
    pub struct_defs: HashMap<NameId, Vec<(String, Type)>>,
    pub enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TTopLevelDef {
    Role(TRoleDef),
    FreeFunc(TFuncDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TRoleDef {
    pub name: NameId,
    pub original_name: String,
    pub func_defs: Vec<TFuncDef>,
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
pub struct TFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub span: Span,
}
