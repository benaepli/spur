//! Traced function names as `&'static str`. Every label naming a function,
//! in any program the process compiles, holds the same copy, so a trace row
//! copies a pointer and touches no reference count.

use std::collections::BTreeSet;
use std::sync::Mutex;

static NAMES: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());

/// Returns the process-wide copy of `name`, leaking one the first time the
/// name is seen. The leak is bounded by the distinct traced names compiled in
/// the process. Called while building a graph, never while running one.
pub(crate) fn intern_trace_name(name: &str) -> &'static str {
    let mut names = NAMES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(&interned) = names.get(name) {
        return interned;
    }
    let interned: &'static str = Box::leak(Box::<str>::from(name));
    names.insert(interned);
    crate::simulator::util_stats::record_trace_name_interned();
    interned
}
