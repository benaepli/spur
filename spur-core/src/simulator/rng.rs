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
//!
//! `StreamSet` supplies the raw values behind that tape. It can hold one
//! generator for all decisions, or one per `Stream`, in which case a decision
//! kind that consumes a different number of draws leaves the values every
//! other kind sees untouched at a fixed seed.

use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::sync::Arc;

/// A schedule recording: the sequence of `u64` draws consumed by the scheduler.
pub type Recording = Arc<[u64]>;

/// The kind of decision a scheduler draw belongs to.
///
/// When `StreamSet` is built with isolation on, each kind draws from its own
/// generator, so adding or removing a draw in one kind cannot shift the values
/// any other kind sees at a fixed seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// Which queue the next step is taken from.
    QueueChoice,
    /// Which item inside the chosen queue runs.
    WithinQueue,
    /// Base priority of a newly created message or continuation.
    MessagePriority,
    /// Base priority of a newly armed timer.
    TimerPriority,
    /// Base priority of a crash or recover event.
    FaultPriority,
    /// Base priority of a partition or heal event.
    PartitionPriority,
    /// Whether a remote send is held back, and for how long.
    SendDelay,
}

impl Stream {
    pub const COUNT: usize = 7;

    #[inline]
    fn index(self) -> usize {
        match self {
            Stream::QueueChoice => 0,
            Stream::WithinQueue => 1,
            Stream::MessagePriority => 2,
            Stream::TimerPriority => 3,
            Stream::FaultPriority => 4,
            Stream::PartitionPriority => 5,
            Stream::SendDelay => 6,
        }
    }
}

/// The raw generators behind one run's scheduling draws.
///
/// With isolation off there is a single generator and stream selection is
/// ignored, which is the shared-stream draw order. With isolation on there is
/// one generator per `Stream`, each seeded from the run seed through a distinct
/// salt, and `use_stream` decides which one the next draw comes from.
pub struct StreamSet {
    gens: Box<[SmallRng]>,
    active: usize,
    isolated: bool,
}

impl StreamSet {
    pub fn new(seed: u64, isolated: bool) -> Self {
        let gens: Box<[SmallRng]> = if isolated {
            (0..Stream::COUNT)
                .map(|i| SmallRng::seed_from_u64(derive_seed(seed, i as i64, STREAM_SALT)))
                .collect()
        } else {
            Box::new([SmallRng::seed_from_u64(seed)])
        };
        Self {
            gens,
            active: 0,
            isolated,
        }
    }

    #[inline]
    fn select(&mut self, stream: Stream) {
        if self.isolated {
            self.active = stream.index();
        }
    }

    #[inline]
    fn current(&mut self) -> &mut SmallRng {
        &mut self.gens[self.active]
    }
}

impl RngCore for StreamSet {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.current().next_u32()
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.current().next_u64()
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.current().fill_bytes(dst)
    }
}

/// A random source that can route draws to a named substream. Sources without
/// substreams accept the routing and ignore it, so the same scheduling code
/// runs against a plain generator in tests.
pub trait StreamRng: RngCore {
    #[inline]
    fn use_stream(&mut self, _stream: Stream) {}
}

impl StreamRng for SmallRng {}
impl StreamRng for rand::rngs::StdRng {}
impl StreamRng for StreamSet {
    #[inline]
    fn use_stream(&mut self, stream: Stream) {
        self.select(stream);
    }
}

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
    fn next_u64(tape: &mut Self::Tape, inner: &mut StreamSet) -> u64;
    /// Extract the complete recording, if this strategy produces one.
    fn into_recording(_tape: Self::Tape) -> Option<Recording> {
        None
    }
}

/// A `RngCore` newtype over an `RngSource`, so every threaded `&mut impl Rng`
/// call site is unchanged. Borrows the tape and inner RNG for one run.
pub struct RecRng<'a, S: RngSource> {
    pub tape: &'a mut S::Tape,
    pub inner: &'a mut StreamSet,
}

impl<S: RngSource> StreamRng for RecRng<'_, S> {
    #[inline]
    fn use_stream(&mut self, stream: Stream) {
        self.inner.select(stream);
    }
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
    fn next_u64(_tape: &mut (), inner: &mut StreamSet) -> u64 {
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
    fn next_u64(tape: &mut Vec<u64>, inner: &mut StreamSet) -> u64 {
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
    fn next_u64(tape: &mut ReplayTape, inner: &mut StreamSet) -> u64 {
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
/// Salt separating the per-decision substreams of one run's schedule seed.
pub const STREAM_SALT: u64 = 0x_5354_5245_414D_5342; // "STREAMSB"

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn record_then_replay_reproduces_stream() {
        // Record a stream of draws.
        let mut inner = StreamSet::new(7, false);
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
        let mut inner2 = StreamSet::new(999, false);
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
        let mut inner = StreamSet::new(3, false);
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
        let mut inner2 = StreamSet::new(123, false);
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

    /// Draw `queue_draws` values from `QueueChoice`, having first consumed
    /// `filler` values from `WithinQueue`.
    fn queue_values(seed: u64, isolated: bool, filler: usize, queue_draws: usize) -> Vec<u64> {
        let mut set = StreamSet::new(seed, isolated);
        set.use_stream(Stream::WithinQueue);
        for _ in 0..filler {
            set.next_u64();
        }
        set.use_stream(Stream::QueueChoice);
        (0..queue_draws).map(|_| set.next_u64()).collect()
    }

    #[test]
    fn isolated_streams_are_immune_to_draws_elsewhere() {
        // The exact-equality oracle: a change that only alters how many values
        // one decision kind consumes must leave the others bit-identical.
        assert_eq!(
            queue_values(11, true, 0, 8),
            queue_values(11, true, 5, 8),
            "isolated QueueChoice shifted when WithinQueue consumed more"
        );
    }

    #[test]
    fn shared_stream_is_perturbed_by_draws_elsewhere() {
        assert_ne!(
            queue_values(11, false, 0, 8),
            queue_values(11, false, 5, 8),
            "shared stream must stay sensitive to draw counts"
        );
    }

    #[test]
    fn isolation_off_ignores_stream_selection() {
        let mut selected = StreamSet::new(4, false);
        let plain = {
            let mut set = StreamSet::new(4, false);
            (0..8).map(|_| set.next_u64()).collect::<Vec<_>>()
        };
        let streams = [
            Stream::QueueChoice,
            Stream::WithinQueue,
            Stream::MessagePriority,
            Stream::TimerPriority,
            Stream::FaultPriority,
            Stream::PartitionPriority,
            Stream::SendDelay,
        ];
        let mixed: Vec<u64> = (0..8)
            .map(|i| {
                selected.use_stream(streams[i % streams.len()]);
                selected.next_u64()
            })
            .collect();
        assert_eq!(plain, mixed);
    }

    #[test]
    fn distinct_streams_do_not_share_values() {
        let mut set = StreamSet::new(2, true);
        set.use_stream(Stream::QueueChoice);
        let a = set.next_u64();
        set.use_stream(Stream::SendDelay);
        let b = set.next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn derive_seed_is_distinct_per_salt_and_run() {
        let w = derive_seed(0, 5, WORKLOAD_SALT);
        let s = derive_seed(0, 5, SCHEDULE_SALT);
        assert_ne!(w, s, "workload and schedule seeds must differ");
        assert_ne!(derive_seed(0, 5, SCHEDULE_SALT), derive_seed(0, 6, SCHEDULE_SALT));
    }
}
