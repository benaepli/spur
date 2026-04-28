use crate::analysis::resolver::NameId;
use crate::parser::{BinOp, Span};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::refinement::RefinementExpr;

#[derive(Debug, Clone)]
pub enum CType {
    Int,
    Bool,
    String,
    Nil,
    Never,
    Array(Box<CType>),
    Map(Box<CType>, Box<CType>),
    Tuple(Vec<CType>),
    Optional(Box<CType>),
    Chan(Box<CType>),
    FifoLink(Box<CType>),
    Iter(Box<CType>),
    Role(NameId),
    Struct(NameId),
    Variant(NameId),
    Refined(Box<CType>, CRefinementHandle),
}

#[derive(Debug, Clone)]
pub struct CRefinementBody {
    pub bound: NameId,
    pub original_bound: String,
    pub body: RefinementExpr,
}

/// Handle for a refinement body that uses identity-based equality and hashing.
#[derive(Debug, Clone)]
pub struct CRefinementHandle(pub Arc<CRefinementBody>);

impl CRefinementHandle {
    pub fn new(body: CRefinementBody) -> Self {
        CRefinementHandle(Arc::new(body))
    }

    pub fn as_ptr(&self) -> *const CRefinementBody {
        Arc::as_ptr(&self.0)
    }
}

impl std::ops::Deref for CRefinementHandle {
    type Target = CRefinementBody;
    fn deref(&self) -> &CRefinementBody {
        &self.0
    }
}

impl PartialEq for CRefinementHandle {
    fn eq(&self, other: &CRefinementHandle) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CRefinementHandle {}

impl Hash for CRefinementHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl PartialEq for CType {
    fn eq(&self, other: &CType) -> bool {
        match (self, other) {
            (CType::Int, CType::Int)
            | (CType::Bool, CType::Bool)
            | (CType::String, CType::String)
            | (CType::Nil, CType::Nil)
            | (CType::Never, CType::Never) => true,
            (CType::Array(a), CType::Array(b)) => a == b,
            (CType::Map(k1, v1), CType::Map(k2, v2)) => k1 == k2 && v1 == v2,
            (CType::Tuple(a), CType::Tuple(b)) => a == b,
            (CType::Optional(a), CType::Optional(b)) => a == b,
            (CType::Chan(a), CType::Chan(b)) => a == b,
            (CType::FifoLink(a), CType::FifoLink(b)) => a == b,
            (CType::Iter(a), CType::Iter(b)) => a == b,
            (CType::Role(a), CType::Role(b)) => a == b,
            (CType::Struct(a), CType::Struct(b)) => a == b,
            (CType::Variant(a), CType::Variant(b)) => a == b,
            (CType::Refined(ia, ha), CType::Refined(ib, hb)) => ia == ib && ha == hb,
            _ => false,
        }
    }
}

impl Eq for CType {}

impl Hash for CType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            CType::Int | CType::Bool | CType::String | CType::Nil | CType::Never => {}
            CType::Array(t)
            | CType::Optional(t)
            | CType::Chan(t)
            | CType::FifoLink(t)
            | CType::Iter(t) => {
                t.hash(state);
            }
            CType::Map(k, v) => {
                k.hash(state);
                v.hash(state);
            }
            CType::Tuple(ts) => ts.hash(state),
            CType::Role(id) | CType::Struct(id) | CType::Variant(id) => id.hash(state),
            CType::Refined(inner, body) => {
                inner.hash(state);
                body.hash(state);
            }
        }
    }
}

impl std::fmt::Display for CType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CType::Int => write!(f, "int"),
            CType::Bool => write!(f, "bool"),
            CType::String => write!(f, "string"),
            CType::Nil => write!(f, "nil"),
            CType::Never => write!(f, "!"),
            CType::Array(t) => write!(f, "array<{}>", t),
            CType::Map(k, v) => write!(f, "map<{}, {}>", k, v),
            CType::Tuple(ts) => {
                let inner = ts
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "({})", inner)
            }
            CType::Optional(t) => write!(f, "{}?", t),
            CType::Chan(t) => write!(f, "chan<{}>", t),
            CType::FifoLink(t) => write!(f, "FifoLink<{}>", t),
            CType::Iter(t) => write!(f, "iter<{}>", t),
            CType::Role(id) => write!(f, "role#{}", id.0),
            CType::Struct(id) => write!(f, "struct#{}", id.0),
            CType::Variant(id) => write!(f, "variant#{}", id.0),
            CType::Refined(inner, body) => {
                write!(f, "{} {{ {} | … }}", inner, body.original_bound)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CAtomic {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,
    Never,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CExpr {
    pub kind: CExprKind,
    pub ty: CType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CExprKind {
    Atomic(CAtomic),

    BinOp(BinOp, CAtomic, CAtomic),
    Not(CAtomic),
    Negate(CAtomic),

    FuncCall(CFuncCall),

    ListLit(Vec<CAtomic>),
    TupleLit(Vec<CAtomic>),
    MapLit(Vec<(CAtomic, CAtomic)>),
    StructLit(NameId, Vec<(String, CAtomic)>),
    VariantLit(NameId, String, Option<CAtomic>),
    IsVariant(CAtomic, String),
    VariantPayload(CAtomic),

    Index(CAtomic, CAtomic),
    TupleAccess(CAtomic, usize),
    FieldAccess(CAtomic, String),

    Conditional(Box<CCondExpr>),
    Block(Box<CBlock>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CFuncCall {
    pub target: NameId,
    pub args: Vec<CAtomic>,
    pub return_type: CType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CExternFunc {
    pub name: NameId,
    pub original_name: String,
    pub params: Vec<CType>,
    pub return_type: CType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CStatement {
    pub kind: CStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CStatementKind {
    LetAtom(CLetAtom),
    Expr(CExpr),
    Return(CAtomic),
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CLetAtom {
    pub name: NameId,
    pub original_name: String,
    pub ty: CType,
    pub value: CExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CBlock {
    pub statements: Vec<CStatement>,
    pub tail_expr: Option<CAtomic>,
    pub ty: CType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CCondExpr {
    pub if_branch: CIfBranch,
    pub elseif_branches: Vec<CIfBranch>,
    pub else_branch: Option<CBlock>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CIfBranch {
    pub condition: CAtomic,
    pub body: CBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub ty: CType,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CFuncKind {
    Sync,
    Async,
    LoopConverted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub kind: CFuncKind,
    pub is_traced: bool,
    /// The role grouping itself is erased from the program structure.
    pub role: Option<NameId>,
    pub params: Vec<CFuncParam>,
    pub return_type: CType,
    pub body: CBlock,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CProgram {
    /// All user-defined functions, flattened across roles.
    pub funcs: Vec<CFuncDef>,
    /// Header: every domain-specific operator that lowering desugared into a
    /// FuncCall has its monomorphic signature recorded here.
    pub extern_funcs: Vec<CExternFunc>,
    pub struct_defs: HashMap<NameId, Vec<(String, CType)>>,
    pub enum_defs: HashMap<NameId, Vec<(String, Option<CType>)>>,
    pub next_name_id: usize,
    pub id_to_name: HashMap<NameId, String>,
}
