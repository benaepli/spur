use crate::analysis::resolver::{NameId, ResolvedTypeDefStmtKind};
use crate::parser::{BinOp, Span};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    Unit,
    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Tuple(Vec<Type>),
    Struct(NameId),
    Role(NameId),
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::String => write!(f, "string"),
            Type::Bool => write!(f, "bool"),
            Type::Unit => write!(f, "unit"),
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
            Type::Struct(id) => write!(f, "struct(id:{})", id.0),
            Type::Role(id) => write!(f, "role(id:{})", id.0),
        }
    }
}

#[derive(Error, Debug, PartialEq, Clone)]
pub enum TypeError {
    #[error("Type Mismatch: Expected `{expected}`, but found `{found}`")]
    Mismatch {
        expected: Type,
        found: Type,
        span: Span,
    },
    #[error("Undefined type name used (this should be caught by resolver)")]
    UndefinedType(Span),
    #[error("Operator `{op}` cannot be applied to type `{ty}`")]
    InvalidUnaryOp {
        op: &'static str,
        ty: Type,
        span: Span,
    },
    #[error("Operator `{op}` cannot be applied to types `{left}` and `{right}`")]
    InvalidBinOp {
        op: BinOp,
        left: Type,
        right: Type,
        span: Span,
    },
    #[error("Function call has wrong number of arguments. Expected {expected}, got {got}")]
    WrongNumberOfArgs {
        expected: usize,
        got: usize,
        span: Span,
    },
    #[error("Attempted to access field `{field_name}` on a non-struct type `{ty}`")]
    NotAStruct {
        ty: Type,
        field_name: String,
        span: Span,
    },
    #[error("Field `{field_name}` not found on struct")]
    FieldNotFound { field_name: String, span: Span },
    #[error("Attempted to index a non-indexable type `{ty}`")]
    NotIndexable { ty: Type, span: Span },
    #[error("A function that is not `unit` must have a return statement on all paths")]
    MissingReturn, // More advanced control flow analysis needed for full accuracy
    #[error("Return statement found outside of a function")]
    ReturnOutsideFunction(Span),
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    params: Vec<Type>,
    return_type: Type,
}

pub struct TypeChecker {
    /// A stack of scopes, mapping a variable's `NameId` to its `Type`.
    scopes: Vec<HashMap<NameId, Type>>,
    /// Maps a type's `NameId` to its definition. Populated in the first pass.
    type_defs: HashMap<NameId, ResolvedTypeDefStmtKind>,
    /// Maps a function's `NameId` to its signature. Populated in the first pass.
    func_signatures: HashMap<NameId, FunctionSignature>,
    /// The expected return type of the current function being checked.
    current_return_type: Option<Type>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()], // Global scope
            type_defs: HashMap::new(),
            func_signatures: HashMap::new(),
            current_return_type: None,
        }
    }
}
