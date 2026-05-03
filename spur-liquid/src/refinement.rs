use spur_ast::name::NameId;
use spur_ast::span::Span;

use crate::ir::{CBinOp, CType};

/// A pure, tree-form expression representing a lowered refinement-body
/// predicate. Refinements are restricted to side-effect-free expressions, so
/// this IR omits statements, control flow, channels, RPC, persistence, and
/// user function calls. Builtin operations are represented via [`ExternCall`],
/// which references entries in `CProgram.extern_funcs` (shared with the
/// program's regular call sites).
#[derive(Debug, Clone, PartialEq)]
pub struct RefinementExpr {
    pub kind: RefinementExprKind,
    pub ty: CType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RefinementExprKind {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,

    BinOp(CBinOp, Box<RefinementExpr>, Box<RefinementExpr>),
    Not(Box<RefinementExpr>),
    Negate(Box<RefinementExpr>),

    /// Call an interned extern (builtin operation) by NameId. The extern entry
    /// lives in `CProgram.extern_funcs`, deduped via the same `extern_cache`
    /// used by ordinary call sites.
    ExternCall {
        target: NameId,
        args: Vec<RefinementExpr>,
        return_type: CType,
    },

    TupleLit(Vec<RefinementExpr>),
    StructLit(NameId, Vec<(NameId, RefinementExpr)>),
    VariantLit(NameId, NameId, Option<Box<RefinementExpr>>),

    IsVariant(Box<RefinementExpr>, NameId, NameId),
    VariantPayload(Box<RefinementExpr>),

    TupleAccess(Box<RefinementExpr>, usize),
    FieldAccess(Box<RefinementExpr>, NameId),

    Conditional(Box<RefinementCond>),

    // Placeholder.
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefinementCond {
    pub if_branch: RefinementIfBranch,
    pub elseif_branches: Vec<RefinementIfBranch>,
    pub else_branch: Option<RefinementExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefinementIfBranch {
    pub condition: RefinementExpr,
    pub body: RefinementExpr,
    pub span: Span,
}

/// Returns true if `e` is an integer-valued expression whose value is
/// determined entirely at compile time, built only from `IntLit`, unary
/// `Negate`, and the integer arithmetic `BinOp`s applied recursively to
/// constant subtrees.
pub fn is_constant_int(e: &RefinementExpr) -> bool {
    match &e.kind {
        RefinementExprKind::IntLit(_) => true,
        RefinementExprKind::Negate(c) => is_constant_int(c),
        RefinementExprKind::BinOp(
            CBinOp::Add
            | CBinOp::Subtract
            | CBinOp::Multiply
            | CBinOp::Divide
            | CBinOp::Modulo,
            l,
            r,
        ) => is_constant_int(l) && is_constant_int(r),
        _ => false,
    }
}
