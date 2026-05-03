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
