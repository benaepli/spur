//! Hash utilities for incremental signature computation.
//!
//! Provides mixing functions for position-dependent hashing used by Value, Env, and State.

use std::hash::{DefaultHasher, Hash, Hasher};

/// Fibonacci hash constant (golden ratio based)
pub const HASH_PRIME: u64 = 0x9E3779B97F4A7C15;

/// Mix a hash value with a position index to create position-dependent hashing.
///
/// Uses bit rotation and multiplication to ensure that the same value at different
/// positions produces different contributions to a combined hash.
#[inline(always)]
pub fn mix(hash: u64, index: u32) -> u64 {
    hash.rotate_left((index % 63) + 1) ^ (index as u64).wrapping_mul(HASH_PRIME)
}

/// Compute the default hash of any hashable value as a u64.
#[inline]
pub fn compute_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
