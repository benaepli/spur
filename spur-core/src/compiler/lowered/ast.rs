use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::Type;
use crate::parser::{BinOp, Span};
use std::collections::HashMap;


#[derive(Debug, Clone, PartialEq)]
pub struct LExpr {
    pub kind: LExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LExprKind {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,

    BinOp(BinOp, Box<LExpr>, Box<LExpr>),
    Not(Box<LExpr>),
    Negate(Box<LExpr>),

    FuncCall(LFuncCall),
    MapLit(Vec<(LExpr, LExpr)>),
    ListLit(Vec<LExpr>),
    TupleLit(Vec<LExpr>),

    Append(Box<LExpr>, Box<LExpr>),
    Prepend(Box<LExpr>, Box<LExpr>),
    Min(Box<LExpr>, Box<LExpr>),
    Exists(Box<LExpr>, Box<LExpr>),
    Erase(Box<LExpr>, Box<LExpr>),
    Store(Box<LExpr>, Box<LExpr>, Box<LExpr>),
    Head(Box<LExpr>),
    Tail(Box<LExpr>),
    Len(Box<LExpr>),

    RpcCall(Box<LExpr>, LUserFuncCall),

    Conditional(Box<LCondExpr>),
    Block(Box<LBlock>),
    VariantLit(NameId, String, Option<Box<LExpr>>),

    IsVariant(Box<LExpr>, String),
    VariantPayload(Box<LExpr>),

    UnwrapOptional(Box<LExpr>),

    MakeIter(Box<LExpr>),
    IterIsDone(Box<LExpr>),
    IterNext(Box<LExpr>),

    MakeChannel,
    Send(Box<LExpr>, Box<LExpr>),
    Recv(Box<LExpr>),

    SetTimer(Option<String>),

    Index(Box<LExpr>, Box<LExpr>),
    Slice(Box<LExpr>, Box<LExpr>, Box<LExpr>),
    TupleAccess(Box<LExpr>, usize),
    FieldAccess(Box<LExpr>, String),

    SafeFieldAccess(Box<LExpr>, String),
    SafeIndex(Box<LExpr>, Box<LExpr>),
    SafeTupleAccess(Box<LExpr>, usize),

    StructLit(NameId, Vec<(String, LExpr)>),

    WrapInOptional(Box<LExpr>),
    PersistData(Box<LExpr>),
    RetrieveData(Type),
    DiscardData,

    Return(Box<LExpr>),
    Break,
    Continue,

    Error,
}


#[derive(Debug, Clone, PartialEq)]
pub enum LFuncCall {
    User(LUserFuncCall),
    Builtin(BuiltinFn, Vec<LExpr>, Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LUserFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<LExpr>,
    pub return_type: Type,
    pub is_free: bool,
    pub span: Span,
}


#[derive(Debug, Clone, PartialEq)]
pub struct LStatement {
    pub kind: LStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LStatementKind {
    VarInit(LVarInit),
    Assignment(LAssignment),
    Expr(LExpr),
    ForLoop(LForLoop),
    ForInLoop(LForInLoop),
    Loop(Vec<LStatement>),
    Error,
}


#[derive(Debug, Clone, PartialEq)]
pub struct LBlock {
    pub statements: Vec<LStatement>,
    pub tail_expr: Option<Box<LExpr>>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LCondExpr {
    pub if_branch: LIfBranch,
    pub elseif_branches: Vec<LIfBranch>,
    pub else_branch: Option<LBlock>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LIfBranch {
    pub condition: LExpr,
    pub body: LBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LVarInit {
    pub name: NameId,
    pub original_name: String,
    pub type_def: Type,
    pub value: LExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LAssignment {
    pub target: LExpr,
    pub value: LExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LForLoopInit {
    VarInit(LVarInit),
    Assignment(LAssignment),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LForLoop {
    pub init: Option<LForLoopInit>,
    pub condition: Option<LExpr>,
    pub increment: Vec<LStatement>,
    pub body: Vec<LStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LForInLoop {
    pub binding_name: NameId,
    pub binding_original_name: String,
    pub iterable: LExpr,
    pub body: Vec<LStatement>,
    pub span: Span,
}


#[derive(Debug, Clone, PartialEq)]
pub struct LProgram {
    pub top_level_defs: Vec<LTopLevelDef>,
    pub next_name_id: usize,
    pub id_to_name: HashMap<NameId, String>,
    pub struct_defs: HashMap<NameId, Vec<(String, Type)>>,
    pub enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LTopLevelDef {
    Role(LRoleDef),
    FreeFunc(LFuncDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LRoleDef {
    pub name: NameId,
    pub original_name: String,
    pub var_inits: Vec<LVarInit>,
    pub func_defs: Vec<LFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub is_sync: bool,
    pub is_traced: bool,
    pub params: Vec<LFuncParam>,
    pub return_type: Type,
    pub body: LBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub span: Span,
}
