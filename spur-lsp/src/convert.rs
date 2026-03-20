use tower_lsp::lsp_types::{Position, Range};

/// Index mapping byte offsets to line/column positions.
pub struct LineIndex {
    /// Byte offset of the start of each line.
    line_starts: Vec<usize>,
    /// The full source text (needed for UTF-16 column calculation).
    text: String,
}

impl LineIndex {
    pub fn new(text: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            text: text.to_string(),
        }
    }

    /// Convert a byte offset to an LSP `Position` (0-indexed line, UTF-16 code unit column).
    pub fn offset_to_position(&self, offset: usize) -> Position {
        let offset = offset.min(self.text.len());
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let line_start = self.line_starts[line];
        let line_text = &self.text[line_start..offset];
        let col_utf16: u32 = line_text.encode_utf16().count() as u32;
        Position {
            line: line as u32,
            character: col_utf16,
        }
    }

    /// Convert a `SimpleSpan<usize>` (byte start..end) to an LSP `Range`.
    pub fn span_to_range(&self, start: usize, end: usize) -> Range {
        Range {
            start: self.offset_to_position(start),
            end: self.offset_to_position(end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line() {
        let idx = LineIndex::new("hello world");
        assert_eq!(idx.offset_to_position(0), Position { line: 0, character: 0 });
        assert_eq!(idx.offset_to_position(5), Position { line: 0, character: 5 });
    }

    #[test]
    fn multi_line() {
        let idx = LineIndex::new("abc\ndef\nghi");
        assert_eq!(idx.offset_to_position(0), Position { line: 0, character: 0 });
        assert_eq!(idx.offset_to_position(4), Position { line: 1, character: 0 });
        assert_eq!(idx.offset_to_position(5), Position { line: 1, character: 1 });
        assert_eq!(idx.offset_to_position(8), Position { line: 2, character: 0 });
    }

    #[test]
    fn span_range() {
        let idx = LineIndex::new("abc\ndef\nghi");
        let range = idx.span_to_range(4, 7);
        assert_eq!(range.start, Position { line: 1, character: 0 });
        assert_eq!(range.end, Position { line: 1, character: 3 });
    }

    #[test]
    fn offset_past_end() {
        let idx = LineIndex::new("abc");
        let pos = idx.offset_to_position(999);
        assert_eq!(pos, Position { line: 0, character: 3 });
    }
}
