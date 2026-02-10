use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum RuntimeError {
    #[error("type error: expected {expected}, got {got}")]
    TypeError {
        expected: &'static str,
        got: &'static str,
    },

    #[error("function not found: {0}")]
    FunctionNotFound(String),

    #[error("index out of bounds: index {index}, length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("map key not found")]
    KeyNotFound,

    #[error("tuple length mismatch: expected {expected}, got {got}")]
    TupleLengthMismatch { expected: usize, got: usize },

    #[error("cannot assign non-tuple to tuple LHS, got {got}")]
    NonTupleAssignment { got: &'static str },

    #[error("unwrap called on None")]
    UnwrapNone,

    #[error("coalesce called on non-option, got {got}")]
    CoalesceNonOption { got: &'static str },

    #[error("sync function tried to call non-sync function: {0}")]
    SyncCallToAsyncFunction(String),

    #[error("instruction not allowed in sync function: {0}")]
    UnsupportedSyncInstruction(String),

    #[error("for-loop over map expects tuple LHS")]
    ForLoopMapExpectsTupleLhs,

    #[error("for-loop requires collection, got {got}")]
    ForLoopNotCollection { got: &'static str },

    #[error("cannot read remote channel directly")]
    RemoteChannelRead,

    #[error("networked channels are unsupported")]
    NetworkedChannelUnsupported,

    #[error("cannot index into non-collection, got {got}")]
    NotACollection { got: &'static str },

    #[error("list subsequence out of bounds: start {start}, end {end}, length {len}")]
    SubsequenceOutOfBounds {
        start: usize,
        end: usize,
        len: usize,
    },

    #[error("channel not found: {0}")]
    ChannelNotFound(usize),

    #[error("operation on empty collection")]
    EmptyCollection,

    #[error("required function not found: {0}")]
    MissingRequiredFunction(String),

    #[error("iterator state must be local")]
    InvalidIteratorState,

    #[error("variant has no payload")]
    VariantHasNoPayload,

    #[error("role not found: {0}")]
    RoleNotFound(String),
}
