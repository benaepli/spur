use crate::analysis::resolver::NameId;
use crate::parser::{BinOp, Span};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    Iter(Box<CType>),
    Role(NameId),
    Struct(NameId),
    Variant(NameId),
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
            CType::Iter(t) => write!(f, "iter<{}>", t),
            CType::Role(id) => write!(f, "role#{}", id.0),
            CType::Struct(id) => write!(f, "struct#{}", id.0),
            CType::Variant(id) => write!(f, "variant#{}", id.0),
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
