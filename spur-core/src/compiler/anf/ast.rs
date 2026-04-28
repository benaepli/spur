use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::Type;
use crate::parser::{BinOp, Span};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum AAtomic {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,
    /// Placeholder for unreachable values after diverging control flow
    /// (return, break, continue). Should never be evaluated at runtime.
    Never,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AExpr {
    pub kind: AExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AExprKind {
    Atomic(AAtomic),

    BinOp(BinOp, AAtomic, AAtomic),
    Not(AAtomic),
    Negate(AAtomic),

    FuncCall(AFuncCall),

    MapLit(Vec<(AAtomic, AAtomic)>),
    ListLit(Vec<AAtomic>),
    TupleLit(Vec<AAtomic>),

    Append(AAtomic, AAtomic),
    Prepend(AAtomic, AAtomic),
    Min(AAtomic, AAtomic),
    Exists(AAtomic, AAtomic),
    Erase(AAtomic, AAtomic),
    Store(AAtomic, AAtomic, AAtomic),
    Head(AAtomic),
    Tail(AAtomic),
    Len(AAtomic),

    RpcCall(AAtomic, AUserFuncCall),

    Conditional(Box<ACondExpr>),
    Block(Box<ABlock>),

    VariantLit(NameId, String, Option<AAtomic>),
    IsVariant(AAtomic, String),
    VariantPayload(AAtomic),

    UnwrapOptional(AAtomic),

    MakeIter(AAtomic),
    IterIsDone(AAtomic),
    IterNext(AAtomic),

    MakeChannel,
    Send(AAtomic, AAtomic),
    Recv(AAtomic),

    SetTimer(Option<String>),
    Fifo(AAtomic),

    Index(AAtomic, AAtomic),
    Slice(AAtomic, AAtomic, AAtomic),
    TupleAccess(AAtomic, usize),
    FieldAccess(AAtomic, String),

    SafeFieldAccess(AAtomic, String),
    SafeIndex(AAtomic, AAtomic),
    SafeTupleAccess(AAtomic, usize),

    StructLit(NameId, Vec<(String, AAtomic)>),

    WrapInOptional(AAtomic),
    PersistData(AAtomic),
    RetrieveData(Type),
    DiscardData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AFuncCall {
    User(AUserFuncCall),
    Builtin(BuiltinFn, Vec<AAtomic>, Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AUserFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<AAtomic>,
    pub return_type: Type,
    pub is_free: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AStatement {
    pub kind: AStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AStatementKind {
    /// `var name: ty = <complex expr>;` binds the result of an AExpr.
    LetAtom(ALetAtom),
    /// `<lhs_path> = <atomic>;`
    Assign(AAssign),
    /// Expression evaluated for side-effects.
    Expr(AExpr),
    Loop(Vec<AStatement>),
    /// `return <atomic>;`
    Return(AAtomic),
    Break,
    Continue,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ALetAtom {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub value: AExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AAssign {
    pub target_id: NameId,
    pub target_name: String,
    pub ty: Type,
    pub value: AAtomic,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ABlock {
    pub statements: Vec<AStatement>,
    /// Tail expression in a block must be atomic.
    pub tail_expr: Option<AAtomic>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ACondExpr {
    pub if_branch: AIfBranch,
    pub elseif_branches: Vec<AIfBranch>,
    pub else_branch: Option<ABlock>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AIfBranch {
    pub condition: AAtomic,
    pub body: ABlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AVarInit {
    pub name: NameId,
    pub original_name: String,
    pub type_def: Type,
    pub stmts: Vec<AStatement>,
    pub value: AExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub is_sync: bool,
    pub is_traced: bool,
    pub params: Vec<AFuncParam>,
    pub return_type: Type,
    pub body: ABlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ARoleDef {
    pub name: NameId,
    pub original_name: String,
    pub var_inits: Vec<AVarInit>,
    pub func_defs: Vec<AFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ATopLevelDef {
    Role(ARoleDef),
    FreeFunc(AFuncDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AProgram {
    pub top_level_defs: Vec<ATopLevelDef>,
    pub next_name_id: usize,
    pub id_to_name: HashMap<NameId, String>,
    pub struct_defs: HashMap<NameId, Vec<(String, Type)>>,
    pub enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
}
