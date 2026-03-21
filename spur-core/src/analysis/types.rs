use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::parser::{BinOp, Span};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum Type {
    Int,
    String,
    Bool,
    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Tuple(Vec<Type>),
    Struct(NameId, String),
    Enum(NameId, String),
    Role(NameId, String),
    Optional(Box<Type>),
    Chan(Box<Type>),

    // Placeholder types.
    EmptyList,
    EmptyMap,
    UnknownChannel,
    Nil,
    Never, // For diverging expressions (return, break, etc.)
    Error, // Unknown type due to earlier error — unifies with anything
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::String => write!(f, "string"),
            Type::Bool => write!(f, "bool"),
            Type::List(t) => write!(f, "list<{}>", t),
            Type::Map(k, v) => write!(f, "map<{}, {}>", k, v),
            Type::Tuple(ts) => {
                let inner = ts
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "({})", inner)
            }
            Type::Struct(_, name) => write!(f, "{}", name),
            Type::Enum(_, name) => write!(f, "{}", name),
            Type::Role(_, name) => write!(f, "{}", name),
            Type::Optional(t) => write!(f, "{}?", t),
            Type::Chan(t) => write!(f, "chan<{}>", t),
            Type::EmptyList => write!(f, "empty list"),
            Type::EmptyMap => write!(f, "empty map"),
            Type::UnknownChannel => write!(f, "unknown channel"),
            Type::Nil => write!(f, "nil"),
            Type::Never => write!(f, "!"),
            Type::Error => write!(f, "<error>"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr {
    pub kind: TypedExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedExprKind {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,

    BinOp(BinOp, Box<TypedExpr>, Box<TypedExpr>),
    Not(Box<TypedExpr>),
    Negate(Box<TypedExpr>),

    FuncCall(TypedFuncCall),
    MapLit(Vec<(TypedExpr, TypedExpr)>),
    ListLit(Vec<TypedExpr>),
    TupleLit(Vec<TypedExpr>),

    Append(Box<TypedExpr>, Box<TypedExpr>),
    Prepend(Box<TypedExpr>, Box<TypedExpr>),
    Min(Box<TypedExpr>, Box<TypedExpr>),
    Exists(Box<TypedExpr>, Box<TypedExpr>),
    Erase(Box<TypedExpr>, Box<TypedExpr>),
    Store(Box<TypedExpr>, Box<TypedExpr>, Box<TypedExpr>),
    Head(Box<TypedExpr>),
    Tail(Box<TypedExpr>),
    Len(Box<TypedExpr>),

    RpcCall(Box<TypedExpr>, TypedUserFuncCall),

    Match(Box<TypedExpr>, Vec<TypedMatchArm>),
    Conditional(Box<TypedCondExpr>),
    Block(Box<TypedBlock>),
    VariantLit(NameId, String, Option<Box<TypedExpr>>),

    UnwrapOptional(Box<TypedExpr>), // T? -> T

    MakeChannel,
    Send(Box<TypedExpr>, Box<TypedExpr>),
    Recv(Box<TypedExpr>),

    SetTimer,

    Index(Box<TypedExpr>, Box<TypedExpr>),
    Slice(Box<TypedExpr>, Box<TypedExpr>, Box<TypedExpr>),
    TupleAccess(Box<TypedExpr>, usize),
    FieldAccess(Box<TypedExpr>, String),

    SafeFieldAccess(Box<TypedExpr>, String),
    SafeIndex(Box<TypedExpr>, Box<TypedExpr>),
    SafeTupleAccess(Box<TypedExpr>, usize),

    StructLit(NameId, Vec<(String, TypedExpr)>),

    WrapInOptional(Box<TypedExpr>), // Internal representation of widening.
    PersistData(Box<TypedExpr>),
    RetrieveData(Type),
    DiscardData,

    // Control flow expressions
    Return(Box<TypedExpr>),
    Break,
    Continue,

    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedFuncCall {
    User(TypedUserFuncCall),
    Builtin(BuiltinFn, Vec<TypedExpr>, Type), // (builtin, args, return_type)
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedUserFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<TypedExpr>,
    pub return_type: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedMatchArm {
    pub pattern: TypedPattern,
    pub body: TypedBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedStatement {
    pub kind: TypedStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedStatementKind {
    VarInit(TypedVarInit),
    Assignment(TypedAssignment),
    Expr(TypedExpr),
    ForLoop(TypedForLoop),
    ForInLoop(TypedForInLoop),
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedBlock {
    pub statements: Vec<TypedStatement>,
    pub tail_expr: Option<Box<TypedExpr>>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedCondExpr {
    pub if_branch: TypedIfBranch,
    pub elseif_branches: Vec<TypedIfBranch>,
    pub else_branch: Option<TypedBlock>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedIfBranch {
    pub condition: TypedExpr,
    pub body: TypedBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedVarTarget {
    Name(NameId, String),
    Tuple(Vec<(NameId, String, Type)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedVarInit {
    pub target: TypedVarTarget,
    pub type_def: Type,
    pub value: TypedExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedAssignment {
    pub target: TypedExpr,
    pub value: TypedExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedForLoopInit {
    VarInit(TypedVarInit),
    Assignment(TypedAssignment),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedForLoop {
    pub init: Option<TypedForLoopInit>,
    pub condition: Option<TypedExpr>,
    pub increment: Option<TypedAssignment>,
    pub body: Vec<TypedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedForInLoop {
    pub pattern: TypedPattern,
    pub iterable: TypedExpr,
    pub body: Vec<TypedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedPattern {
    pub kind: TypedPatternKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedPatternKind {
    Var(NameId, String),
    Wildcard,
    Tuple(Vec<TypedPattern>),
    Variant(NameId, String, Option<Box<TypedPattern>>),
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedProgram {
    pub top_level_defs: Vec<TypedTopLevelDef>,
    pub next_name_id: usize,
    pub id_to_name: HashMap<NameId, String>,
    /// Struct definitions: NameId → list of (field_name, field_type)
    pub struct_defs: HashMap<NameId, Vec<(String, Type)>>,
    /// Enum definitions: NameId → list of (variant_name, optional payload_type)
    pub enum_defs: HashMap<NameId, Vec<(String, Option<Type>)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedTopLevelDef {
    Role(TypedRoleDef),
    FreeFunc(TypedFuncDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedRoleDef {
    pub name: NameId,
    pub original_name: String,
    pub var_inits: Vec<TypedVarInit>,
    pub func_defs: Vec<TypedFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub is_sync: bool,
    pub is_traced: bool,
    pub params: Vec<TypedFuncParam>,
    pub return_type: Type,
    pub body: TypedBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub span: Span,
}
