//! Hash utilities for incremental signature computation.
//!
//! Provides mixing functions for position-dependent hashing used by Value, Env, and State.
//! Also defines the HashPolicy trait for conditional hashing based on simulation mode.

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

/// Trait for conditional hashing based on simulation mode.
///
/// Two implementations:
/// - `WithHashing`: Full signature computation (for model checker state deduplication)
/// - `NoHashing`: No-op hashing (for exec_plan path execution)
///
/// The trait methods will be implemented alongside the types they work with to avoid
/// circular dependencies. WithHashing implementations will live in values.rs and state.rs.
pub trait HashPolicy: 'static + Clone + std::fmt::Debug + Send + Sync + Hash {
    /// Mix a hash with a position index for order-dependent aggregation.
    fn mix(sig: u64, pos: u32) -> u64;

    /// Update Env signature when a slot changes.
    /// Typically XORs out old value signature and XORs in new value signature.
    fn update_env_sig(old_sig: u64, old_val_sig: u64, new_val_sig: u64, slot: u32) -> u64;
}

/// Full hashing implementation - computes actual signatures.
/// Used by model checker for state deduplication.
#[derive(Clone, Debug, Hash)]
pub struct WithHashing;

/// No-op hashing implementation - always returns 0.
/// Used by exec_plan path execution to avoid hashing overhead.
#[derive(Clone, Debug, Hash)]
pub struct NoHashing;

impl HashPolicy for WithHashing {
    #[inline]
    fn mix(sig: u64, pos: u32) -> u64 {
        mix(sig, pos)
    }

    #[inline]
    fn update_env_sig(old_sig: u64, old_val_sig: u64, new_val_sig: u64, slot: u32) -> u64 {
        old_sig ^ Self::mix(old_val_sig, slot) ^ Self::mix(new_val_sig, slot)
    }
}

impl HashPolicy for NoHashing {
    #[inline]
    fn mix(_sig: u64, _pos: u32) -> u64 {
        0
    }

    #[inline]
    fn update_env_sig(_old_sig: u64, _old_val_sig: u64, _new_val_sig: u64, _slot: u32) -> u64 {
        0
    }
}
