//! A run's text columns, each laid end to end in one byte buffer, and the
//! bounded list that carries those buffers from the history writers back to
//! the simulation threads.

use crate::simulator::util_stats;
use crossbeam::queue::ArrayQueue;
use serde::Serialize;
use std::sync::LazyLock;

/// Bytes that are always valid UTF-8. Every append is a whole `&str` or a
/// complete JSON text, so every length the buffer has had is a character
/// boundary.
#[derive(Debug, Default)]
pub struct TextBuffer(Vec<u8>);

impl TextBuffer {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    pub fn push_str(&mut self, s: &str) {
        self.0.extend_from_slice(s.as_bytes());
    }

    /// Appends `s` as a JSON string literal.
    pub fn push_json_str(&mut self, s: &str) {
        self.push_json(s);
    }

    /// Appends `value` as JSON text. On failure nothing is appended and the
    /// result is false.
    pub fn push_json<T: Serialize + ?Sized>(&mut self, value: &T) -> bool {
        let start = self.0.len();
        match serde_json::to_writer(&mut self.0, value) {
            Ok(()) => true,
            Err(_) => {
                self.0.truncate(start);
                false
            }
        }
    }

    /// The text from byte `start` to the end; empty when `start` is not a
    /// character boundary within the buffer.
    pub fn str_from(&self, start: usize) -> &str {
        self.0
            .get(start..)
            .and_then(|b| std::str::from_utf8(b).ok())
            .unwrap_or("")
    }

    /// Whether byte offset `i` falls between two characters of the text or
    /// at its end.
    pub fn is_char_boundary(&self, i: usize) -> bool {
        match self.0.get(i) {
            // A UTF-8 continuation byte is 0b10xxxxxx; any other byte starts
            // a character.
            Some(&b) => (b as i8) >= -0x40,
            None => i == self.0.len(),
        }
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// Reuses the storage of `bytes`, discarding its contents.
    pub(crate) fn from_storage(mut bytes: Vec<u8>) -> Self {
        bytes.clear();
        Self(bytes)
    }
}

impl std::fmt::Write for TextBuffer {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

/// Largest number of buffer sets the free list holds.
pub const FREE_LIST_SETS: usize = 64;

/// A buffer whose capacity is above this is freed rather than recycled, so
/// one long run cannot pin a large buffer for every later run.
pub const RECYCLE_CAPACITY_LIMIT: usize = 1 << 20;

static FREE_LIST: LazyLock<ArrayQueue<TextBuffers>> =
    LazyLock::new(|| ArrayQueue::new(FREE_LIST_SETS));

/// The four text columns of one run. A set is taken on the simulation thread
/// at run start, travels with the run's rows to a history writer, and goes
/// back on the free list once the writer is done with it, so its storage is
/// allocated and freed on no particular thread only when the list is empty
/// or full.
#[derive(Debug, Default)]
pub struct TextBuffers {
    pub log_content: TextBuffer,
    pub trace_payload: TextBuffer,
    pub action: TextBuffer,
    pub op_payload: TextBuffer,
    /// Which buffers began the run without storage, one bit per buffer in
    /// field order.
    began_empty: u8,
}

/// What returning one set to the free list did with its buffers.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RecycleOutcome {
    pub recycled: u64,
    pub dropped_oversize: u64,
}

impl TextBuffers {
    fn buffers(&self) -> [&TextBuffer; 4] {
        [&self.log_content, &self.trace_payload, &self.action, &self.op_payload]
    }

    fn buffers_mut(&mut self) -> [&mut TextBuffer; 4] {
        [
            &mut self.log_content,
            &mut self.trace_payload,
            &mut self.action,
            &mut self.op_payload,
        ]
    }

    /// An empty set, reusing storage from the free list when it has any.
    pub fn take() -> Self {
        Self::take_from(&FREE_LIST)
    }

    fn take_from(list: &ArrayQueue<TextBuffers>) -> Self {
        let mut set = list.pop().unwrap_or_default();
        set.began_empty = set
            .buffers()
            .iter()
            .enumerate()
            .filter(|(_, b)| b.capacity() == 0)
            .fold(0, |mask, (i, _)| mask | (1 << i));
        set
    }

    /// Buffers that began the run without storage and now have some: storage
    /// this run allocated rather than reused.
    pub fn allocated_this_run(&self) -> u64 {
        self.buffers()
            .iter()
            .enumerate()
            .filter(|(i, b)| self.began_empty & (1 << i) != 0 && b.capacity() > 0)
            .count() as u64
    }

    /// Counts the storage this run allocated rather than reused.
    pub fn record_allocated(&self) {
        let allocated = self.allocated_this_run();
        if allocated > 0 {
            util_stats::record_text_buffers_allocated(allocated);
        }
    }

    /// Clears the set and returns it to the free list, counting what was kept
    /// and what was freed.
    pub fn recycle(self) {
        let outcome = self.recycle_into(&FREE_LIST);
        util_stats::record_text_buffers_returned(outcome.recycled, outcome.dropped_oversize);
    }

    /// A buffer above `RECYCLE_CAPACITY_LIMIT` is freed and its slot left
    /// without storage. When the list is full the whole set is freed and
    /// nothing counts as recycled.
    fn recycle_into(mut self, list: &ArrayQueue<TextBuffers>) -> RecycleOutcome {
        let mut outcome = RecycleOutcome::default();
        let mut kept = 0;
        for buffer in self.buffers_mut() {
            if buffer.capacity() > RECYCLE_CAPACITY_LIMIT {
                *buffer = TextBuffer::default();
                outcome.dropped_oversize += 1;
            } else {
                buffer.0.clear();
                if buffer.capacity() > 0 {
                    kept += 1;
                }
            }
        }
        self.began_empty = 0;
        if list.push(self).is_ok() {
            outcome.recycled = kept;
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_with_capacities(caps: [usize; 4]) -> TextBuffers {
        let mut set = TextBuffers::default();
        for (buffer, cap) in set.buffers_mut().into_iter().zip(caps) {
            buffer.reserve(cap);
            buffer.push_str("x");
        }
        set
    }

    #[test]
    fn the_free_list_stays_bounded_and_frees_oversize_buffers() {
        let list = ArrayQueue::new(FREE_LIST_SETS);
        let mut recycled = 0;
        for _ in 0..FREE_LIST_SETS + 10 {
            let outcome = set_with_capacities([16, 16, 16, 16]).recycle_into(&list);
            assert_eq!(outcome.dropped_oversize, 0);
            recycled += outcome.recycled;
        }
        assert_eq!(list.len(), FREE_LIST_SETS, "the list holds at most its capacity");
        assert_eq!(recycled, 4 * FREE_LIST_SETS as u64, "a set the full list refused is not recycled");

        let small = ArrayQueue::new(2);
        let outcome = set_with_capacities([16, RECYCLE_CAPACITY_LIMIT + 1, 16, 0]).recycle_into(&small);
        assert_eq!(outcome, RecycleOutcome { recycled: 3, dropped_oversize: 1 });
        let taken = TextBuffers::take_from(&small);
        assert!(taken.buffers().iter().all(|b| b.is_empty()), "a recycled set comes back cleared");
        assert_eq!(taken.trace_payload.capacity(), 0, "the oversize buffer was freed");
        assert!(taken.log_content.capacity() >= 16, "a small buffer kept its storage");
        assert_eq!(taken.allocated_this_run(), 0);
    }

    #[test]
    fn allocation_counts_only_buffers_that_started_without_storage() {
        let empty = ArrayQueue::new(1);
        let mut fresh = TextBuffers::take_from(&empty);
        assert_eq!(fresh.allocated_this_run(), 0, "an unwritten buffer allocated nothing");
        fresh.log_content.push_str("line");
        fresh.op_payload.push_str("[]");
        assert_eq!(fresh.allocated_this_run(), 2);

        let list = ArrayQueue::new(1);
        let _ = fresh.recycle_into(&list);
        let mut reused = TextBuffers::take_from(&list);
        reused.log_content.push_str("line");
        reused.trace_payload.push_str("[]");
        assert_eq!(reused.allocated_this_run(), 1, "only the trace buffer lacked storage");
    }

    #[test]
    fn text_stays_utf8_at_every_reported_length() {
        let mut t = TextBuffer::default();
        t.push_str("e\u{301}");
        let mid = t.len();
        t.push_json_str("q\"\u{1f600}\n");
        assert!(t.is_char_boundary(0) && t.is_char_boundary(mid) && t.is_char_boundary(t.len()));
        assert!(!t.is_char_boundary(t.len() + 1));
        assert!(!t.is_char_boundary(1 + "e".len()), "inside the combining accent");
        assert_eq!(t.str_from(mid), "\"q\\\"\u{1f600}\\n\"");
        assert_eq!(t.str_from(0), "e\u{301}\"q\\\"\u{1f600}\\n\"");
    }
}
