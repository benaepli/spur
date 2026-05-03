//! Tiny in-memory cache for refinement-check results.
//!
//! The cache is keyed by a blake3 hash of the encoded fact bundle that
//! would be shipped to the `flg` binary, plus the sorted list of
//! function ids being verified. This makes hits on equivalent
//! programs cheap (and avoids running the SMT-backed checker on every
//! debounced LSP edit).
//!
//! Currently the cache is a wrapped `Mutex<HashMap>` with no eviction
//! policy. Realistic LSP sessions rarely accumulate more than a few
//! dozen entries; if we ever want to bound it, swap in `lru` here.

use std::collections::HashMap;
use std::sync::Mutex;

use spur_ast::name::NameId;

use crate::flg::EncodedFacts;

/// 32-byte blake3 hash. Used as the cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; 32]);

impl CacheKey {
    /// Compute the cache key from the encoded fact bundle and the
    /// (sorted) list of functions being checked. The function list is
    /// sorted before hashing so callers don't have to.
    pub fn from_facts(facts: &EncodedFacts, fns_to_check: &[NameId]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(facts.tuple_accessor.as_bytes());
        hasher.update(b"\x00");
        hasher.update(facts.program_in.as_bytes());
        hasher.update(b"\x00");
        hasher.update(facts.expr_origin.as_bytes());
        hasher.update(b"\x00");
        let mut sorted: Vec<i32> = fns_to_check.iter().map(|n| n.0 as i32).collect();
        sorted.sort_unstable();
        for f in &sorted {
            hasher.update(&f.to_le_bytes());
        }
        let hash = hasher.finalize();
        CacheKey(*hash.as_bytes())
    }
}

/// Process-local cache: maps fact-bundle hashes to the list of
/// refinement diagnostics produced for that bundle. Calling
/// [`Cache::get_or_insert_with`] with the same key in a hot path
/// avoids re-running `flg`.
pub struct Cache<V: Clone> {
    inner: Mutex<HashMap<CacheKey, V>>,
}

impl<V: Clone> Default for Cache<V> {
    fn default() -> Self {
        Cache {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

impl<V: Clone> Cache<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &CacheKey) -> Option<V> {
        let g = self.inner.lock().expect("cache mutex poisoned");
        g.get(key).cloned()
    }

    pub fn insert(&self, key: CacheKey, value: V) {
        let mut g = self.inner.lock().expect("cache mutex poisoned");
        g.insert(key, value);
    }

    pub fn get_or_insert_with<F>(&self, key: CacheKey, f: F) -> V
    where
        F: FnOnce() -> V,
    {
        if let Some(v) = self.get(&key) {
            return v;
        }
        let v = f();
        self.insert(key, v.clone());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_facts_produce_equal_keys() {
        let a = EncodedFacts {
            tuple_accessor: "0\t100\n".into(),
            program_in: "gctx([], [], [])\t[]\t-1\n".into(),
            fn_to_check: "1\n".into(),
            expr_origin: String::new(),
        };
        let b = a.clone();
        assert_eq!(
            CacheKey::from_facts(&a, &[NameId(1)]),
            CacheKey::from_facts(&b, &[NameId(1)])
        );
    }

    #[test]
    fn different_facts_produce_different_keys() {
        let a = EncodedFacts {
            tuple_accessor: "0\t100\n".into(),
            program_in: "gctx([], [], [])\t[]\t-1\n".into(),
            fn_to_check: "1\n".into(),
            expr_origin: String::new(),
        };
        let mut b = a.clone();
        b.program_in = "gctx([], [], [])\t[]\t-2\n".into();
        assert_ne!(
            CacheKey::from_facts(&a, &[NameId(1)]),
            CacheKey::from_facts(&b, &[NameId(1)])
        );
    }

    #[test]
    fn fns_to_check_order_does_not_matter() {
        let facts = EncodedFacts::default();
        assert_eq!(
            CacheKey::from_facts(&facts, &[NameId(1), NameId(2)]),
            CacheKey::from_facts(&facts, &[NameId(2), NameId(1)])
        );
    }

    #[test]
    fn cache_round_trips() {
        let cache: Cache<Vec<i32>> = Cache::new();
        let key = CacheKey([0u8; 32]);
        assert!(cache.get(&key).is_none());
        cache.insert(key, vec![1, 2, 3]);
        assert_eq!(cache.get(&key).unwrap(), vec![1, 2, 3]);

        let v = cache.get_or_insert_with(key, || panic!("should not run"));
        assert_eq!(v, vec![1, 2, 3]);
    }
}
