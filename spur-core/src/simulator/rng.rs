//! Compile-time-selectable RNG strategies for record-and-replay (RnR).
//!
//! The simulator threads a single `&mut impl Rng` through the scheduling and
//! execution stack (see `core::scheduler`, `core::exec`, `path::exec_plan`).
//! `RngSource` selects, at compile time, whether that stream is live,
//! recorded, or replayed from a tape. `LiveRng` monomorphizes to the bare
//! inner RNG, so the common (non-RnR) explore path pays nothing.
//!
//! A "tape" (`Recording`) is the sequence of `u64` draws the scheduler
//! pulled. Replaying a tape through the same scheduler reproduces a
//! schedule, and mutating a few entries (`mutate_tape`) yields a
//! similar-but-distinct, always-feasible schedule, since the scheduler still
//! ranges each draw over the live enabled set.

use rand::RngCore;
use rand::rngs::SmallRng;
use std::sync::Arc;

/// A schedule recording: the sequence of `u64` draws consumed by the scheduler.
pub type Recording = Arc<[u64]>;

/// Compile-time RNG strategy. Default `into_recording` returns `None` so that
/// `LiveRng` collapses to nothing.
pub trait RngSource: 'static + Send {
    /// Per-run tape state, threaded as `&mut`.
    type Tape: Send;
    /// Whether draws are recorded. Const-folded; no push when `false`.
    const RECORDS: bool = false;

    /// Build a fresh tape, optionally seeded from a recording to replay.
    fn new_tape(seed: Option<Recording>) -> Self::Tape;
    /// Draw the next `u64`.
    fn next_u64(tape: &mut Self::Tape, inner: &mut SmallRng) -> u64;
    /// Extract the complete recording, if this strategy produces one.
    fn into_recording(_tape: Self::Tape) -> Option<Recording> {
        None
    }
}

/// A `RngCore` newtype over an `RngSource`, so every threaded `&mut impl Rng`
/// call site is unchanged. Borrows the tape and inner RNG for one run.
pub struct RecRng<'a, S: RngSource> {
    pub tape: &'a mut S::Tape,
    pub inner: &'a mut SmallRng,
}

impl<S: RngSource> RngCore for RecRng<'_, S> {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        S::next_u64(self.tape, self.inner) as u32
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        S::next_u64(self.tape, self.inner)
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        for chunk in dst.chunks_mut(8) {
            let bytes = S::next_u64(self.tape, self.inner).to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&bytes[..n]);
        }
    }
}

/// No tape: draws come straight from the inner RNG. Monomorphizes to the bare
/// RNG, the zero-cost default for `-e standard`/`-e genetic`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiveRng;

impl RngSource for LiveRng {
    type Tape = ();
    #[inline]
    fn new_tape(_seed: Option<Recording>) -> Self::Tape {}
    #[inline]
    fn next_u64(_tape: &mut (), inner: &mut SmallRng) -> u64 {
        inner.next_u64()
    }
}

/// Live draws, logged into a tape for later replay/mutation.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecordRng;

impl RngSource for RecordRng {
    type Tape = Vec<u64>;
    const RECORDS: bool = true;
    #[inline]
    fn new_tape(_seed: Option<Recording>) -> Vec<u64> {
        Vec::new()
    }
    #[inline]
    fn next_u64(tape: &mut Vec<u64>, inner: &mut SmallRng) -> u64 {
        let v = inner.next_u64();
        tape.push(v);
        v
    }
    fn into_recording(tape: Vec<u64>) -> Option<Recording> {
        Some(tape.into())
    }
}

/// Replay-then-record: returns `src[pos++]` until the source tape is
/// exhausted, then falls back to the inner RNG. Always pushes to `out`, so
/// the produced recording is the complete effective stream (replay prefix
/// plus fallback tail) and stays self-contained even when a mutation
/// lengthens the run past the source.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayRng;

#[derive(Debug, Default)]
pub struct ReplayTape {
    src: Recording,
    pos: usize,
    out: Vec<u64>,
}

impl RngSource for ReplayRng {
    type Tape = ReplayTape;
    const RECORDS: bool = true;
    fn new_tape(seed: Option<Recording>) -> ReplayTape {
        ReplayTape {
            src: seed.unwrap_or_default(),
            pos: 0,
            out: Vec::new(),
        }
    }
    #[inline]
    fn next_u64(tape: &mut ReplayTape, inner: &mut SmallRng) -> u64 {
        let v = if tape.pos < tape.src.len() {
            let v = tape.src[tape.pos];
            tape.pos += 1;
            v
        } else {
            inner.next_u64()
        };
        tape.out.push(v);
        v
    }
    fn into_recording(tape: ReplayTape) -> Option<Recording> {
        Some(tape.out.into())
    }
}

/// Controlled deviation: clone the recording and replace `k` random
/// positions with fresh draws. Preserves length; replaying the result
/// diverges from the original only after the first mutated position.
pub fn mutate_tape(r: &Recording, k: usize, rng: &mut impl rand::Rng) -> Recording {
    if r.is_empty() {
        return r.clone();
    }
    let mut v = r.to_vec();
    let len = v.len();
    for _ in 0..k.min(len) {
        let idx = rng.random_range(0..len);
        v[idx] = rng.random();
    }
    v.into()
}

/// SplitMix64 finalizer: a cheap, well-distributed mixer for deriving seeds.
pub fn splitmix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive a per-run seed from a session seed, the run id, and a domain salt.
/// Distinct salts (e.g. workload vs schedule) yield uncorrelated streams while
/// keeping `(session_seed, run_id)` fully reproducible.
pub fn derive_seed(session_seed: u64, run_id: i64, salt: u64) -> u64 {
    splitmix(splitmix(session_seed ^ salt) ^ (run_id as u64))
}

/// Domain salts for `derive_seed`.
pub const WORKLOAD_SALT: u64 = 0x_5742_4C4F_4144_5345; // "WBLOADSE"
pub const SCHEDULE_SALT: u64 = 0x_5343_4845_4453_4C45; // "SCHEDSLE"

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn record_then_replay_reproduces_stream() {
        // Record a stream of draws.
        let mut inner = SmallRng::seed_from_u64(7);
        let mut tape = RecordRng::new_tape(None);
        let recorded: Vec<u64> = {
            let mut rec = RecRng::<RecordRng> {
                tape: &mut tape,
                inner: &mut inner,
            };
            (0..16).map(|_| rec.random::<u64>()).collect()
        };
        let recording = RecordRng::into_recording(tape).unwrap();

        // Replay it; the source seed is irrelevant while the tape covers draws.
        let mut inner2 = SmallRng::seed_from_u64(999);
        let mut rtape = ReplayRng::new_tape(Some(recording.clone()));
        let replayed: Vec<u64> = {
            let mut rep = RecRng::<ReplayRng> {
                tape: &mut rtape,
                inner: &mut inner2,
            };
            (0..16).map(|_| rep.random::<u64>()).collect()
        };
        assert_eq!(recorded, replayed);
        // The replay's own recording equals the source (no fallback used).
        assert_eq!(&*ReplayRng::into_recording(rtape).unwrap(), &recorded[..]);
    }

    #[test]
    fn replay_records_fallback_tail() {
        // A 4-entry source, but we draw 10 times: the tail comes from `inner`
        // and must be captured so the produced recording is self-contained.
        let src: Recording = vec![1, 2, 3, 4].into();
        let mut inner = SmallRng::seed_from_u64(3);
        let mut rtape = ReplayRng::new_tape(Some(src));
        let first: Vec<u64> = {
            let mut rep = RecRng::<ReplayRng> {
                tape: &mut rtape,
                inner: &mut inner,
            };
            (0..10).map(|_| rep.next_u64()).collect()
        };
        let complete = ReplayRng::into_recording(rtape).unwrap();
        assert_eq!(complete.len(), 10);
        assert_eq!(&complete[..4], &[1, 2, 3, 4]);

        // Re-replaying the complete tape reproduces the full stream with no
        // dependence on the fallback seed.
        let mut inner2 = SmallRng::seed_from_u64(123);
        let mut rtape2 = ReplayRng::new_tape(Some(complete));
        let second: Vec<u64> = {
            let mut rep = RecRng::<ReplayRng> {
                tape: &mut rtape2,
                inner: &mut inner2,
            };
            (0..10).map(|_| rep.next_u64()).collect()
        };
        assert_eq!(first, second);
    }

    #[test]
    fn mutate_tape_changes_k_positions() {
        let r: Recording = (0..20u64).collect::<Vec<_>>().into();
        let mut rng = SmallRng::seed_from_u64(1);
        let m = mutate_tape(&r, 1, &mut rng);
        assert_eq!(m.len(), r.len());
        let diffs = r.iter().zip(m.iter()).filter(|(a, b)| a != b).count();
        assert!(diffs <= 1, "k=1 mutation changed {} positions", diffs);
    }

    #[test]
    fn derive_seed_is_distinct_per_salt_and_run() {
        let w = derive_seed(0, 5, WORKLOAD_SALT);
        let s = derive_seed(0, 5, SCHEDULE_SALT);
        assert_ne!(w, s, "workload and schedule seeds must differ");
        assert_ne!(derive_seed(0, 5, SCHEDULE_SALT), derive_seed(0, 6, SCHEDULE_SALT));
    }
}
