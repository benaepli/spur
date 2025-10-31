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
pub enum TypeDefKind {
    Named(String),
    Map(Box<TypeDef>, Box<TypeDef>),
    List(Box<TypeDef>),
    Tuple(Vec<TypeDef>),
    Optional(Box<TypeDef>),
    Promise(Box<TypeDef>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub kind: TypeDefKind,
    pub span: Span,
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
pub enum ForLoopInit {
    VarInit(VarInit),
    Assignment(Assignment),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForLoop {
    pub init: Option<ForLoopInit>,
    pub condition: Option<Expr>,
    pub increment: Option<Assignment>,
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
    pub target: Expr,
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
    NilLit,

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
    StructLit(String, Vec<(String, Expr)>),

    // Built-in operations
    Append(Box<Expr>, Box<Expr>),
    Prepend(Box<Expr>, Box<Expr>),
    PollForResps(Box<Expr>, Box<Expr>),
    PollForAnyResp(Box<Expr>),
    NextResp(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Exists(Box<Expr>, Box<Expr>),
    Erase(Box<Expr>, Box<Expr>),
    Head(Box<Expr>),
    Tail(Box<Expr>),
    Len(Box<Expr>),

    // RPC calls
    RpcCall(Box<Expr>, FuncCall),
    RpcAsyncCall(Box<Expr>, FuncCall),

    Await(Box<Expr>),
    Spawn(Box<Expr>),

    CreatePromise,
    CreateFuture(Box<Expr>),
    ResolvePromise(Box<Expr>, Box<Expr>),

    // Postfix operations
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Box<Expr>, Box<Expr>),
    TupleAccess(Box<Expr>, usize),
    FieldAccess(Box<Expr>, String),
    Unwrap(Box<Expr>),
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
    Coalesce,
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::And => write!(f, "and"),
            BinOp::Or => write!(f, "or"),
            BinOp::Equal => write!(f, "=="),
            BinOp::NotEqual => write!(f, "!="),
            BinOp::Less => write!(f, "<"),
            BinOp::LessEqual => write!(f, "<="),
            BinOp::Greater => write!(f, ">"),
            BinOp::GreaterEqual => write!(f, ">="),
            BinOp::Add => write!(f, "+"),
            BinOp::Subtract => write!(f, "-"),
            BinOp::Multiply => write!(f, "*"),
            BinOp::Divide => write!(f, "/"),
            BinOp::Modulo => write!(f, "%"),
            BinOp::Coalesce => write!(f, "??"),
        }
    }
}

// Helper to build a parser for a left-associative binary operation
fn build_binary_op<'a, I, P>(
    next_parser: P,
    op_parser: impl Parser<'a, I, BinOp, extra::Err<Rich<'a, TokenKind>>> + Clone,
) -> impl Parser<'a, I, Expr, extra::Err<Rich<'a, TokenKind>>> + Clone
where
    I: BorrowInput<'a, Token = TokenKind, Span = SimpleSpan> + Clone,
    P: Parser<'a, I, Expr, extra::Err<Rich<'a, TokenKind>>> + Clone,
{
    next_parser.clone().foldl(
        op_parser.then(next_parser).repeated(),
        |left, (op, right)| {
            let span = left.span.union(right.span);
            Expr::new(ExprKind::BinOp(op, Box::new(left), Box::new(right)), span)
        },
    )
}

fn program<'a, I>() -> impl Parser<'a, I, Program, extra::Err<Rich<'a, TokenKind>>>
where
    I: BorrowInput<'a, Token = TokenKind, Span = SimpleSpan> + Clone,
{
    let mut expr = Recursive::declare();
    let mut statement = Recursive::declare();
    let mut type_def = Recursive::declare();
    let mut pattern = Recursive::declare();

    let ident = select! { TokenKind::Identifier(s) => s.clone() };

    pattern.define({
        let atom = choice((
            ident.clone().map(PatternKind::Var),
            just(TokenKind::Underscore).to(PatternKind::Wildcard),
        ))
        .map_with(|kind, e| Pattern::new(kind, e.span()));

        let items = pattern
            .clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .collect::<Vec<_>>();

        let tuple = items
            .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen))
            .map_with(|pats, e| {
                if pats.is_empty() {
                    Pattern::new(PatternKind::Unit, e.span())
                } else {
                    Pattern::new(PatternKind::Tuple(pats), e.span())
                }
            });

        choice((atom, tuple))
    });

    type_def.define({
        let named = ident.clone().map_with(|name, e| TypeDef {
            kind: TypeDefKind::Named(name),
            span: e.span(),
        });

        let map = just(TokenKind::Map)
            .ignore_then(
                type_def
                    .clone()
                    .then_ignore(just(TokenKind::Comma))
                    .then(type_def.clone())
                    .delimited_by(just(TokenKind::Less), just(TokenKind::Greater)),
            )
            .map_with(|(key_type, val_type), e| TypeDef {
                kind: TypeDefKind::Map(Box::new(key_type), Box::new(val_type)),
                span: e.span(),
            });

        let list = just(TokenKind::List)
            .ignore_then(
                type_def
                    .clone()
                    .delimited_by(just(TokenKind::Less), just(TokenKind::Greater)),
            )
            .map_with(|t, e| TypeDef {
                kind: TypeDefKind::List(Box::new(t)),
                span: e.span(),
            });

        let tuple = type_def
            .clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen))
            .map_with(|ts, e| TypeDef {
                kind: TypeDefKind::Tuple(ts),
                span: e.span(),
            });

        let promise = just(TokenKind::Promise)
            .ignore_then(
                type_def
                    .clone()
                    .delimited_by(just(TokenKind::Less), just(TokenKind::Greater)),
            )
            .map_with(|t, e| TypeDef {
                kind: TypeDefKind::Promise(Box::new(t)),
                span: e.span(),
            });

        let base_type = choice((named, map, list, tuple, promise));

        base_type
            .clone()
            .then(just(TokenKind::Question).or_not())
            .map_with(|(base, opt_q), e| {
                if opt_q.is_some() {
                    TypeDef {
                        kind: TypeDefKind::Optional(Box::new(base)),
                        span: e.span(),
                    }
                } else {
                    base
                }
            })
    });

    let primary = {
        let val = select! {
            TokenKind::Integer(i) => ExprKind::IntLit(i),
            TokenKind::String(s) => ExprKind::StringLit(s),
            TokenKind::True => ExprKind::BoolLit(true),
            TokenKind::False => ExprKind::BoolLit(false),
            TokenKind::Nil => ExprKind::NilLit,
        };

        let args = || {
            expr.clone()
                .separated_by(just(TokenKind::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen))
        };

        let func_call = ident
            .clone()
            .then(args())
            .map_with(|(name, args), e| FuncCall {
                name,
                args,
                span: e.span(),
            });

        let list_lit = expr
            .clone()
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(TokenKind::LeftBracket), just(TokenKind::RightBracket))
            .map(ExprKind::ListLit);

        let kv_pair = expr
            .clone()
            .then_ignore(just(TokenKind::Colon))
            .then(expr.clone());
        let map_lit = kv_pair
            .separated_by(just(TokenKind::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(TokenKind::LeftBrace), just(TokenKind::RightBrace))
            .map(ExprKind::MapLit);

        let tuple_lit = choice((
            just(TokenKind::LeftParen)
                .then(just(TokenKind::RightParen))
                .to(vec![]),
            expr.clone()
                .then_ignore(just(TokenKind::Comma))
                .then(
                    expr.clone()
                        .separated_by(just(TokenKind::Comma))
                        .allow_trailing()
                        .collect::<Vec<_>>(),
                )
                .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen))
                .map(|(first, mut rest)| {
                    rest.insert(0, first);
                    rest
                }),
        ))
        .map(ExprKind::TupleLit);

        let struct_lit = ident
            .clone()
            .then(
                ident
                    .clone()
                    .then_ignore(just(TokenKind::Colon))
                    .then(expr.clone())
                    .separated_by(just(TokenKind::Comma))
                    .allow_trailing()
                    .collect::<Vec<(String, Expr)>>()
                    .delimited_by(just(TokenKind::LeftBrace), just(TokenKind::RightBrace)),
            )
            .map(|(name, fields)| ExprKind::StructLit(name, fields));

        let paren_expr = expr
            .clone()
            .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen));

        let two_arg_builtin = |name, constructor: fn(Box<Expr>, Box<Expr>) -> ExprKind| {
            just(name).ignore_then(
                expr.clone()
                    .then_ignore(just(TokenKind::Comma))
                    .then(expr.clone())
                    .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen))
                    .map(move |(a, b)| constructor(Box::new(a), Box::new(b))),
            )
        };
        let one_arg_builtin = |name, constructor: fn(Box<Expr>) -> ExprKind| {
            just(name)
                .ignore_then(
                    expr.clone()
                        .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen)),
                )
                .map(move |a| constructor(Box::new(a)))
        };

        let zero_arg_builtin = |name, kind: ExprKind| {
            just(name)
                .ignore_then(just(TokenKind::LeftParen))
                .ignore_then(just(TokenKind::RightParen))
                .to(kind)
        };

        let rpc_call = choice((
            just(TokenKind::RpcCall).to(false),
            just(TokenKind::RpcAsyncCall).to(true),
        ))
        .then(
            expr.clone()
                .then_ignore(just(TokenKind::Comma))
                .then(func_call.clone())
                .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen)),
        )
        .map(|(is_async, (target, call))| {
            if is_async {
                ExprKind::RpcAsyncCall(Box::new(target), call)
            } else {
                ExprKind::RpcCall(Box::new(target), call)
            }
        });

        let atom = choice((
            val,
            list_lit,
            map_lit,
            struct_lit,
            two_arg_builtin(TokenKind::Append, ExprKind::Append),
            two_arg_builtin(TokenKind::Prepend, ExprKind::Prepend),
            two_arg_builtin(TokenKind::PollForResps, ExprKind::PollForResps),
            one_arg_builtin(TokenKind::PollForAnyResp, ExprKind::PollForAnyResp),
            one_arg_builtin(TokenKind::NextResp, ExprKind::NextResp),
            two_arg_builtin(TokenKind::Min, ExprKind::Min),
            two_arg_builtin(TokenKind::Exists, ExprKind::Exists),
            two_arg_builtin(TokenKind::Erase, ExprKind::Erase),
            one_arg_builtin(TokenKind::Head, ExprKind::Head),
            one_arg_builtin(TokenKind::Tail, ExprKind::Tail),
            one_arg_builtin(TokenKind::Len, ExprKind::Len),
            zero_arg_builtin(TokenKind::CreatePromise, ExprKind::CreatePromise),
            one_arg_builtin(TokenKind::CreateFuture, ExprKind::CreateFuture),
            two_arg_builtin(TokenKind::ResolvePromise, ExprKind::ResolvePromise),
            func_call.map(ExprKind::FuncCall),
            ident.clone().map(ExprKind::Var),
            tuple_lit,
            rpc_call,
        ))
        .map_with(|kind, e| Expr::new(kind, e.span()));

        let primary_base = choice((
            paren_expr, // Try to parse (expr) first
            atom,       // Then try all other atoms
        ));

        #[derive(Clone)]
        enum PostfixOp {
            Index(Expr),
            Slice(Expr, Expr),
            TupleAccess(usize, Span),
            FieldAccess(String, Span),
            Unwrap(Span),
        }

        let postfix_op = choice((
            // Index: [expr]
            expr.clone()
                .delimited_by(just(TokenKind::LeftBracket), just(TokenKind::RightBracket))
                .map(PostfixOp::Index),
            // Slice: [expr:expr]
            expr.clone()
                .then_ignore(just(TokenKind::Colon))
                .then(expr.clone())
                .delimited_by(just(TokenKind::LeftBracket), just(TokenKind::RightBracket))
                .map(|(start, end)| PostfixOp::Slice(start, end)),
            // Tuple Access: .INT
            just(TokenKind::Dot)
                .ignore_then(select! { TokenKind::Integer(i) => i as usize })
                .map_with(|idx, e| PostfixOp::TupleAccess(idx, e.span())),
            // Field Access: .ID
            just(TokenKind::Dot)
                .ignore_then(ident.clone())
                .map_with(|name, e| PostfixOp::FieldAccess(name, e.span())),
            just(TokenKind::Bang).map_with(|_, e| PostfixOp::Unwrap(e.span())),
        ));

        primary_base.foldl(postfix_op.repeated(), |lhs, op| match op {
            PostfixOp::Index(idx) => {
                let span = lhs.span.union(idx.span);
                Expr::new(ExprKind::Index(Box::new(lhs), Box::new(idx)), span)
            }
            PostfixOp::Slice(start, end) => {
                let span = lhs.span.union(end.span);
                Expr::new(
                    ExprKind::Slice(Box::new(lhs), Box::new(start), Box::new(end)),
                    span,
                )
            }
            PostfixOp::TupleAccess(idx, op_span) => {
                let span = lhs.span.union(op_span);
                Expr::new(ExprKind::TupleAccess(Box::new(lhs), idx), span)
            }
            PostfixOp::FieldAccess(name, op_span) => {
                let span = lhs.span.union(op_span);
                Expr::new(ExprKind::FieldAccess(Box::new(lhs), name), span)
            }
            PostfixOp::Unwrap(op_span) => {
                let span = lhs.span.union(op_span);
                Expr::new(ExprKind::Unwrap(Box::new(lhs)), span)
            }
        })
    };

    expr.define({
        let unary = recursive(|unary| {
            choice((
                just(TokenKind::Bang)
                    .then(unary.clone())
                    .map_with(|(_, val), e| Expr::new(ExprKind::Not(Box::new(val)), e.span())),
                just(TokenKind::Minus)
                    .then(unary.clone())
                    .map_with(|(_, val), e| Expr::new(ExprKind::Negate(Box::new(val)), e.span())),
                just(TokenKind::Await)
                    .then(unary.clone())
                    .map_with(|(_, val), e| Expr::new(ExprKind::Await(Box::new(val)), e.span())),
                just(TokenKind::Spawn)
                    .then(unary.clone())
                    .map_with(|(_, val), e| Expr::new(ExprKind::Spawn(Box::new(val)), e.span())),
                primary.clone(),
            ))
        });

        let multiplicative = build_binary_op(
            unary,
            choice((
                just(TokenKind::Star).to(BinOp::Multiply),
                just(TokenKind::Slash).to(BinOp::Divide),
                just(TokenKind::Percent).to(BinOp::Modulo),
            )),
        );
        let additive = build_binary_op(
            multiplicative,
            choice((
                just(TokenKind::Plus).to(BinOp::Add),
                just(TokenKind::Minus).to(BinOp::Subtract),
            )),
        );

        let comparison_op = choice((
            just(TokenKind::EqualEqual).to(BinOp::Equal),
            just(TokenKind::BangEqual).to(BinOp::NotEqual),
            just(TokenKind::Less).to(BinOp::Less),
            just(TokenKind::LessEqual).to(BinOp::LessEqual),
            just(TokenKind::Greater).to(BinOp::Greater),
            just(TokenKind::GreaterEqual).to(BinOp::GreaterEqual),
        ));
        let comparison = additive
            .clone()
            .then(comparison_op.then(additive).or_not())
            .map(|(lhs, rhs_opt)| {
                if let Some((op, rhs)) = rhs_opt {
                    let span = lhs.span.union(rhs.span);
                    Expr::new(ExprKind::BinOp(op, Box::new(lhs), Box::new(rhs)), span)
                } else {
                    lhs
                }
            });

        let and_expr = build_binary_op(comparison, just(TokenKind::And).to(BinOp::And));
        let or_expr = build_binary_op(and_expr, just(TokenKind::Or).to(BinOp::Or));

        let coalescing_expr = build_binary_op(
            or_expr,
            just(TokenKind::QuestionQuestion).to(BinOp::Coalesce),
        );

        coalescing_expr
    });

    let var_init_core = just(TokenKind::Let)
        .ignore_then(ident.clone())
        .then_ignore(just(TokenKind::Colon))
        .then(type_def.clone())
        .then_ignore(just(TokenKind::Equal))
        .then(expr.clone())
        .map_with(|((name, type_def), value), e| VarInit {
            name,
            type_def,
            value,
            span: e.span(),
        });

    statement.define({
        let block = || {
            statement
                .clone()
                .repeated()
                .collect()
                .delimited_by(just(TokenKind::LeftBrace), just(TokenKind::RightBrace))
        };

        let assignment = primary
            .clone()
            .then_ignore(just(TokenKind::Equal))
            .then(expr.clone())
            .map_with(|(target, value), e| Assignment {
                target,
                value,
                span: e.span(),
            });

        let var_init_stmt = var_init_core
            .clone()
            .then_ignore(just(TokenKind::Semicolon))
            .map_with(|var_init, e| Statement::new(StatementKind::VarInit(var_init), e.span()));

        let if_branch = |kw| {
            just(kw)
                .ignore_then(
                    expr.clone()
                        .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen)),
                )
                .then(block())
                .map_with(|(condition, body), e| IfBranch {
                    condition,
                    body,
                    span: e.span(),
                })
        };
        let cond_stmts = if_branch(TokenKind::If)
            .then(if_branch(TokenKind::ElseIf).repeated().collect())
            .then(just(TokenKind::Else).ignore_then(block()).or_not())
            .map_with(|((if_branch, elseif_branches), else_branch), e| {
                Statement::new(
                    StatementKind::Conditional(CondStmts {
                        if_branch,
                        elseif_branches,
                        else_branch,
                        span: e.span(),
                    }),
                    e.span(),
                )
            });

        let for_loop_init = choice((
            var_init_core.clone().map(ForLoopInit::VarInit), // <-- MODIFIED
            assignment.clone().map(ForLoopInit::Assignment),
        ));

        let for_loop = just(TokenKind::For)
            .ignore_then(
                for_loop_init
                    .clone()
                    .or_not()
                    .then_ignore(just(TokenKind::Semicolon))
                    .then(expr.clone().or_not())
                    .then_ignore(just(TokenKind::Semicolon))
                    .then(assignment.clone().or_not())
                    .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen)),
            )
            .then(block())
            .map_with(|(((init, condition), increment), body), e| {
                Statement::new(
                    StatementKind::ForLoop(ForLoop {
                        init,
                        condition,
                        increment,
                        body,
                        span: e.span(),
                    }),
                    e.span(),
                )
            });

        let for_in_loop = just(TokenKind::For)
            .ignore_then(
                pattern
                    .clone()
                    .then_ignore(just(TokenKind::In))
                    .then(expr.clone())
                    .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen)),
            )
            .then(block())
            .map_with(|((pattern, iterable), body), e| {
                Statement::new(
                    StatementKind::ForInLoop(ForInLoop {
                        pattern,
                        iterable,
                        body,
                        span: e.span(),
                    }),
                    e.span(),
                )
            });

        let simple_stmt = |kind, constructor: fn(Expr) -> StatementKind| {
            just(kind)
                .ignore_then(expr.clone())
                .then_ignore(just(TokenKind::Semicolon))
                .map_with(move |expr, e| Statement::new(constructor(expr), e.span()))
        };

        choice((
            cond_stmts,
            var_init_stmt,
            assignment
                .then_ignore(just(TokenKind::Semicolon))
                .map_with(|a, e| Statement::new(StatementKind::Assignment(a), e.span())),
            expr.clone()
                .then_ignore(just(TokenKind::Semicolon))
                .map_with(|e, s| Statement::new(StatementKind::Expr(e), s.span())),
            simple_stmt(TokenKind::Return, StatementKind::Return),
            for_loop,
            for_in_loop,
            just(TokenKind::Print)
                .ignore_then(
                    expr.clone()
                        .delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen)),
                )
                .then_ignore(just(TokenKind::Semicolon))
                .map_with(|e, s| Statement::new(StatementKind::Print(e), s.span())),
            just(TokenKind::Break)
                .then_ignore(just(TokenKind::Semicolon))
                .map_with(|_, e| Statement::new(StatementKind::Break, e.span())),
        ))
    });

    let var_init = just(TokenKind::Let)
        .ignore_then(ident.clone())
        .then_ignore(just(TokenKind::Colon))
        .then(type_def.clone())
        .then_ignore(just(TokenKind::Equal))
        .then(expr.clone())
        .then_ignore(just(TokenKind::Semicolon))
        .map_with(|((name, type_def), value), e| VarInit {
            name,
            type_def,
            value,
            span: e.span(),
        });

    let func_param = ident
        .clone()
        .then_ignore(just(TokenKind::Colon))
        .then(type_def.clone())
        .map_with(|(name, type_def), e| FuncParam {
            name,
            type_def,
            span: e.span(),
        });
    let func_params = func_param
        .separated_by(just(TokenKind::Comma))
        .allow_trailing()
        .collect::<Vec<_>>();

    let func_def = just(TokenKind::Func)
        .ignore_then(ident.clone())
        .then(func_params.delimited_by(just(TokenKind::LeftParen), just(TokenKind::RightParen)))
        .then(
            just(TokenKind::Arrow)
                .ignore_then(type_def.clone())
                .or_not(),
        )
        .then(
            statement
                .clone()
                .repeated()
                .collect()
                .delimited_by(just(TokenKind::LeftBrace), just(TokenKind::RightBrace)),
        )
        .map_with(|(((name, params), return_type), body), e| FuncDef {
            name,
            params,
            return_type,
            body,
            span: e.span(),
        });

    let field_def = ident
        .clone()
        .then_ignore(just(TokenKind::Colon))
        .then(type_def.clone())
        .map_with(|(name, type_def), e| FieldDef {
            name,
            type_def,
            span: e.span(),
        });
    let field_defs = field_def
        .separated_by(just(TokenKind::Semicolon))
        .allow_trailing()
        .collect::<Vec<_>>();

    let type_def_stmt = just(TokenKind::Type)
        .ignore_then(ident.clone())
        .then(choice((
            field_defs
                .delimited_by(just(TokenKind::LeftBrace), just(TokenKind::RightBrace))
                .map(TypeDefStmtKind::Struct),
            just(TokenKind::Equal)
                .ignore_then(type_def.clone())
                .map(TypeDefStmtKind::Alias),
        )))
        .then_ignore(just(TokenKind::Semicolon))
        .map_with(|(name, def), e| TypeDefStmt {
            name,
            def,
            span: e.span(),
        });

    let role_contents = || {
        var_init
            .clone()
            .repeated()
            .collect::<Vec<_>>()
            .then(func_def.clone().repeated().collect::<Vec<_>>())
            .delimited_by(just(TokenKind::LeftBrace), just(TokenKind::RightBrace))
    };

    let role_def = just(TokenKind::Role)
        .ignore_then(ident)
        .then(role_contents())
        .map_with(|(name, (var_inits, func_defs)), e| {
            TopLevelDef::Role(RoleDef {
                name,
                var_inits,
                func_defs,
                span: e.span(),
            })
        });

    let top_level_def = choice((role_def, type_def_stmt.map(TopLevelDef::Type)));

    let client_def = just(TokenKind::ClientInterface)
        .ignore_then(role_contents())
        .map_with(|(var_inits, func_defs), e| ClientDef {
            var_inits,
            func_defs,
            span: e.span(),
        });

    // --- Final Program Parser ---
    top_level_def
        .repeated()
        .collect()
        .then(client_def)
        .map_with(|(top_level_defs, client_def), e| Program {
            top_level_defs,
            client_def,
            span: e.span(),
        })
        .then_ignore(end())
}

fn make_input(
    eoi: SimpleSpan,
    tokens: &[Token],
) -> impl BorrowInput<'_, Token = TokenKind, Span = SimpleSpan> + Clone {
    tokens.map(eoi, |t| (&t.kind, &t.span))
}

pub fn parse_program(tokens: &'_ [Token]) -> ParseResult<Program, Rich<'_, TokenKind>> {
    let len = tokens.last().map(|t| t.span.end()).unwrap_or(0);
    let input = make_input((0..len).into(), &tokens);

    program().parse(input)
}
