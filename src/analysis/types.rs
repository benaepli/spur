use crate::analysis::resolver::NameId;
use crate::parser::{BinOp, Span};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Tuple(Vec<Type>),
    Struct(NameId, String),
    Role(NameId, String),
    Optional(Box<Type>),
    Future(Box<Type>),
    Promise(Box<Type>),
    Lock,

    // Placeholder types.
    EmptyList,
    EmptyMap,
    EmptyPromise,
    Nil,
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
            Type::Role(_, name) => write!(f, "{}", name),
            Type::Optional(t) => write!(f, "{}?", t),
            Type::Future(t) => write!(f, "future<{}>", t),
            Type::Promise(t) => write!(f, "promise<{}>", t),
            Type::Lock => write!(f, "lock"),
            Type::EmptyList => write!(f, "empty list"),
            Type::EmptyMap => write!(f, "empty map"),
            Type::EmptyPromise => write!(f, "empty promise"),
            Type::Nil => write!(f, "nil"),
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
    PollForResps(Box<TypedExpr>, Box<TypedExpr>),
    PollForAnyResp(Box<TypedExpr>),
    NextResp(Box<TypedExpr>),
    Min(Box<TypedExpr>, Box<TypedExpr>),
    Exists(Box<TypedExpr>, Box<TypedExpr>),
    Erase(Box<TypedExpr>, Box<TypedExpr>),
    Head(Box<TypedExpr>),
    Tail(Box<TypedExpr>),
    Len(Box<TypedExpr>),

    RpcCall(Box<TypedExpr>, TypedFuncCall),

    UnwrapOptional(Box<TypedExpr>), // T? -> T
    Await(Box<TypedExpr>),          // future<T> -> T

    CreatePromise,
    CreateFuture(Box<TypedExpr>),
    ResolvePromise(Box<TypedExpr>, Box<TypedExpr>),
    CreateLock,

    Index(Box<TypedExpr>, Box<TypedExpr>),
    Slice(Box<TypedExpr>, Box<TypedExpr>, Box<TypedExpr>),
    TupleAccess(Box<TypedExpr>, usize),
    FieldAccess(Box<TypedExpr>, String),

    StructLit(NameId, Vec<(String, TypedExpr)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<TypedExpr>,
    pub return_type: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedStatement {
    pub kind: TypedStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedStatementKind {
    Conditional(TypedCondStmts),
    VarInit(TypedVarInit),
    Assignment(TypedAssignment),
    Expr(TypedExpr),
    Return(TypedExpr),
    ForLoop(TypedForLoop),
    ForInLoop(TypedForInLoop),
    Print(TypedExpr),
    Break,
    Lock(Box<TypedExpr>, Vec<TypedStatement>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedCondStmts {
    pub if_branch: TypedIfBranch,
    pub elseif_branches: Vec<TypedIfBranch>,
    pub else_branch: Option<Vec<TypedStatement>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedIfBranch {
    pub condition: TypedExpr,
    pub body: Vec<TypedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedVarInit {
    pub name: NameId,
    pub original_name: String,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedProgram {
    pub top_level_defs: Vec<TypedTopLevelDef>,
    pub client_def: TypedClientDef,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedTopLevelDef {
    Role(TypedRoleDef),
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
pub struct TypedClientDef {
    pub var_inits: Vec<TypedVarInit>,
    pub func_defs: Vec<TypedFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub params: Vec<TypedFuncParam>,
    pub return_type: Type,
    pub body: Vec<TypedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: Type,
    pub span: Span,
}
