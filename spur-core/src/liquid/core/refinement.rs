use crate::analysis::resolver::NameId;
use crate::parser::{BinOp, Span};

use super::ast::CType;

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

    BinOp(BinOp, Box<RefinementExpr>, Box<RefinementExpr>),
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

    ListLit(Vec<RefinementExpr>),
    TupleLit(Vec<RefinementExpr>),
    MapLit(Vec<(RefinementExpr, RefinementExpr)>),
    StructLit(NameId, Vec<(String, RefinementExpr)>),
    VariantLit(NameId, String, Option<Box<RefinementExpr>>),

    IsVariant(Box<RefinementExpr>, String),
    VariantPayload(Box<RefinementExpr>),

    Index(Box<RefinementExpr>, Box<RefinementExpr>),
    TupleAccess(Box<RefinementExpr>, usize),
    FieldAccess(Box<RefinementExpr>, String),

    Conditional(Box<RefinementCond>),

    /// Placeholder produced when a refinement body contains a construct that
    /// is not legal in a predicate (e.g. a user function call). The lowerer
    /// records a [`RefinementValidationError`] alongside emitting this node so
    /// downstream passes don't have to re-walk the body to detect errors.
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
