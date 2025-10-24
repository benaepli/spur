pub mod format;

use crate::lexer::{Token, TokenKind};
use chumsky::input::BorrowInput;
use chumsky::prelude::*;
use chumsky::span::SimpleSpan;

pub type Span = SimpleSpan<usize>;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub top_level_defs: Vec<TopLevelDef>,
    pub client_def: ClientDef,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopLevelDef {
    Role(RoleDef),
    Type(TypeDefStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoleDef {
    pub name: String,
    pub var_inits: Vec<VarInit>,
    pub func_defs: Vec<FuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClientDef {
    pub var_inits: Vec<VarInit>,
    pub func_defs: Vec<FuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub name: String,
    pub params: Vec<FuncParam>,
    pub return_type: Option<TypeDef>,
    pub body: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncParam {
    pub name: String,
    pub type_def: TypeDef,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarInit {
    pub name: String,
    pub type_def: TypeDef,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDefStmt {
    pub name: String,
    pub def: TypeDefStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDefStmtKind {
    Struct(Vec<FieldDef>),
    Alias(TypeDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub name: String,
    pub type_def: TypeDef,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDef {
    Named(String),
    Map(Box<TypeDef>, Box<TypeDef>),
    List(Box<TypeDef>),
    Tuple(Vec<TypeDef>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span,
}

impl Statement {
    fn new(kind: StatementKind, span: Span) -> Self {
        Statement { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StatementKind {
    Conditional(CondStmts),
    VarInit(VarInit),
    Assignment(Assignment),
    Expr(Expr),
    Return(Expr),
    ForLoop(ForLoop),
    ForInLoop(ForInLoop),
    Await(Expr),
    Print(Expr),
    Break,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CondStmts {
    pub if_branch: IfBranch,
    pub elseif_branches: Vec<IfBranch>,
    pub else_branch: Option<Vec<Statement>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfBranch {
    pub condition: Expr,
    pub body: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForLoop {
    pub init: Assignment,
    pub condition: Expr,
    pub increment: Assignment,
    pub body: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForInLoop {
    pub pattern: Pattern,
    pub iterable: Expr,
    pub body: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub pattern: Pattern,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

impl Pattern {
    fn new(kind: PatternKind, span: Span) -> Self {
        Pattern { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Var(String),
    Wildcard,
    Unit,
    Tuple(Vec<Pattern>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    fn new(kind: ExprKind, span: Span) -> Self {
        Expr { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    // Variables and literals
    Var(String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),

    // Binary operations
    BinOp(BinOp, Box<Expr>, Box<Expr>),

    // Unary operations
    Not(Box<Expr>),
    Negate(Box<Expr>),

    // Function call
    FuncCall(FuncCall),

    // Collections
    MapLit(Vec<(Expr, Expr)>),
    ListLit(Vec<Expr>),
    TupleLit(Vec<Expr>),

    // Options literal
    Options(Vec<String>),

    // Built-in operations
    Append(Box<Expr>, Box<Expr>),
    Prepend(Box<Expr>, Box<Expr>),
    PollForResps(Box<Expr>, Box<Expr>),
    PollForAnyResp(Box<Expr>),
    NextResp(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Exists(Box<Expr>, Box<Expr>),
    Head(Box<Expr>),
    Tail(Box<Expr>),
    Len(Box<Expr>),

    // RPC calls
    RpcCall(Box<Expr>, FuncCall),
    RpcAsyncCall(Box<Expr>, FuncCall),

    // Postfix operations
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Box<Expr>, Box<Expr>),
    TupleAccess(Box<Expr>, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncCall {
    pub name: String,
    pub args: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    // Logical
    And,
    Or,

    // Comparison
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}
