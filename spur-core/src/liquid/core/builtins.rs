use crate::analysis::resolver::NameId;

/// Closed enum naming every language operator that lowering folds into the
/// `extern_funcs` header. Each `(BuiltinKind, Vec<CType>)` instantiation
/// produces exactly one extern entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltinKind {
    // Array ops
    ArrayAppend,
    ArrayPrepend,
    ArrayLen,
    ArrayHead,
    ArrayTail,
    ArraySlice,

    // Map ops
    MapStore,
    MapExists,
    MapErase,

    // Misc data
    Min,

    // Iterators
    IterMake,
    IterIsDone,
    IterNext,

    // Channels
    ChanMake,
    ChanSend,
    ChanRecv,

    // Timers
    TimerSet,

    // Optionals
    OptionalWrap,
    OptionalUnwrap,
    SafeField(String),
    SafeIndex,
    SafeTupleAccess(usize),

    // Persistence
    Persist,
    Retrieve,
    Discard,

    // RPC — keyed by the user function being invoked remotely
    Rpc(NameId),

    // Existing resolver builtins
    Println,
    IntToString,
    BoolToString,
    RoleToString,
    UniqueId,
}

impl BuiltinKind {
    /// The leading identifier used in generated extern names. Type arguments
    /// are appended by the lowerer, e.g. `array_append<int>`.
    pub fn base_name(&self) -> String {
        match self {
            BuiltinKind::ArrayAppend => "array_append".into(),
            BuiltinKind::ArrayPrepend => "array_prepend".into(),
            BuiltinKind::ArrayLen => "array_len".into(),
            BuiltinKind::ArrayHead => "array_head".into(),
            BuiltinKind::ArrayTail => "array_tail".into(),
            BuiltinKind::ArraySlice => "array_slice".into(),
            BuiltinKind::MapStore => "map_store".into(),
            BuiltinKind::MapExists => "map_exists".into(),
            BuiltinKind::MapErase => "map_erase".into(),
            BuiltinKind::Min => "min".into(),
            BuiltinKind::IterMake => "iter_make".into(),
            BuiltinKind::IterIsDone => "iter_is_done".into(),
            BuiltinKind::IterNext => "iter_next".into(),
            BuiltinKind::ChanMake => "chan_make".into(),
            BuiltinKind::ChanSend => "chan_send".into(),
            BuiltinKind::ChanRecv => "chan_recv".into(),
            BuiltinKind::TimerSet => "timer_set".into(),
            BuiltinKind::OptionalWrap => "optional_wrap".into(),
            BuiltinKind::OptionalUnwrap => "optional_unwrap".into(),
            BuiltinKind::SafeField(field) => format!("safe_field_{}", field),
            BuiltinKind::SafeIndex => "safe_index".into(),
            BuiltinKind::SafeTupleAccess(i) => format!("safe_tuple_{}", i),
            BuiltinKind::Persist => "persist".into(),
            BuiltinKind::Retrieve => "retrieve".into(),
            BuiltinKind::Discard => "discard".into(),
            BuiltinKind::Rpc(target) => format!("rpc_to_{}", target.0),
            BuiltinKind::Println => "println".into(),
            BuiltinKind::IntToString => "int_to_string".into(),
            BuiltinKind::BoolToString => "bool_to_string".into(),
            BuiltinKind::RoleToString => "role_to_string".into(),
            BuiltinKind::UniqueId => "unique_id".into(),
        }
    }
}
