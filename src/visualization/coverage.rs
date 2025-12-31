//! Coverage heatmap visualization.
//!
//! Generates an HTML heatmap showing source code coverage by mapping
//! CFG vertex hit counts to source byte positions.

use crate::compiler::cfg::Vertex;
use crate::parser::Span;
use std::collections::HashMap;
use std::fmt::Write;

/// Converts vertex-level coverage to per-byte hit counts.
///
/// For each vertex with a hit count, looks up its source span and sums
/// the hit count at each byte position. Handles overlapping spans by
/// accumulating counts.
pub fn vertex_coverage_to_byte_coverage(
    vertex_coverage: &HashMap<Vertex, u64>,
    vertex_to_span: &HashMap<Vertex, Span>,
    source_len: usize,
) -> Vec<u64> {
    let mut byte_hits = vec![0u64; source_len];

    for (vertex, &count) in vertex_coverage {
        if let Some(span) = vertex_to_span.get(vertex) {
            let start = span.start;
            let end = span.end.min(source_len);
            for i in start..end {
                byte_hits[i] += count;
            }
        }
    }

    byte_hits
}

/// Renders an HTML heatmap of coverage.
///
/// Maps hit counts to a red→green color gradient:
/// - Red (0 hits) means uncovered
/// - Yellow (mid-range) means partially covered  
/// - Green (max hits) means heavily covered
pub fn render_html_heatmap(source: &str, byte_hits: &[u64]) -> String {
    let max_hits = byte_hits.iter().copied().max().unwrap_or(1).max(1);

    let mut html = String::new();
    html.push_str(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Coverage Heatmap</title>
<style>
body {
    font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 20px;
    margin: 0;
}
pre {
    margin: 0;
    line-height: 1.4;
    font-size: 14px;
}
pre span[data-hits] {
    cursor: default;
}
pre span[data-hits]:hover {
    outline: 1px solid #fff;
    z-index: 1;
}
.legend {
    margin-bottom: 20px;
    padding: 10px;
    background: #2d2d2d;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.legend-bar {
    width: 200px;
    height: 20px;
    background: linear-gradient(to right, #ff4444, #ffff44, #44ff44);
    border-radius: 2px;
}
#tooltip {
    position: fixed;
    background: #333;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    display: none;
    z-index: 1000;
    white-space: nowrap;
}
</style>
</head>
<body>
<div id="tooltip"></div>
<div class="legend">
    <span>Coverage:</span>
    <span>0 hits</span>
    <div class="legend-bar"></div>
    <span>"#,
    );
    let _ = write!(html, "{} hits", max_hits);
    html.push_str(
        r#"</span>
</div>
<pre>"#,
    );

    for (i, ch) in source.char_indices() {
        let hits = byte_hits.get(i).copied().unwrap_or(0);
        let color = hits_to_color(hits, max_hits);

        if ch == '<' {
            let _ = write!(
                html,
                "<span style=\"background:{};\" data-hits=\"{}\">&lt;</span>",
                color, hits
            );
        } else if ch == '>' {
            let _ = write!(
                html,
                "<span style=\"background:{};\" data-hits=\"{}\">&gt;</span>",
                color, hits
            );
        } else if ch == '&' {
            let _ = write!(
                html,
                "<span style=\"background:{};\" data-hits=\"{}\">&amp;</span>",
                color, hits
            );
        } else if ch == '\n' {
            html.push('\n');
        } else {
            let _ = write!(
                html,
                "<span style=\"background:{};\" data-hits=\"{}\">{}</span>",
                color, hits, ch
            );
        }
    }

    html.push_str(
        r#"</pre>
<script>
const tooltip = document.getElementById('tooltip');
document.querySelector('pre').addEventListener('mouseover', (e) => {
    if (e.target.dataset.hits !== undefined) {
        tooltip.textContent = e.target.dataset.hits + ' hits';
        tooltip.style.display = 'block';
    }
});
document.querySelector('pre').addEventListener('mousemove', (e) => {
    tooltip.style.left = (e.clientX + 10) + 'px';
    tooltip.style.top = (e.clientY + 10) + 'px';
});
document.querySelector('pre').addEventListener('mouseout', (e) => {
    if (e.target.dataset.hits !== undefined) {
        tooltip.style.display = 'none';
    }
});
</script>
</body>
</html>"#,
    );

    html
}

/// Maps hit count to a color on the red→yellow→green gradient.
fn hits_to_color(hits: u64, max_hits: u64) -> String {
    if hits == 0 {
        return "#442222".to_string(); // Dark red for uncovered
    }

    // Normalize to 0.0-1.0 range
    let ratio = (hits as f64) / (max_hits as f64);

    // Red → Yellow → Green gradient
    let (r, g) = if ratio < 0.5 {
        // Red to Yellow (increase green)
        let t = ratio * 2.0;
        (255, (255.0 * t) as u8)
    } else {
        // Yellow to Green (decrease red)
        let t = (ratio - 0.5) * 2.0;
        ((255.0 * (1.0 - t)) as u8, 255)
    };

    format!("#{:02x}{:02x}22", r, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_coverage_conversion() {
        use chumsky::span::SimpleSpan;

        let mut vertex_coverage = HashMap::new();
        vertex_coverage.insert(0, 5);
        vertex_coverage.insert(1, 3);

        let mut vertex_to_span = HashMap::new();
        vertex_to_span.insert(0, SimpleSpan::from(0..3)); // bytes 0-2
        vertex_to_span.insert(1, SimpleSpan::from(2..5)); // bytes 2-4 (overlaps at byte 2)

        let byte_hits = vertex_coverage_to_byte_coverage(&vertex_coverage, &vertex_to_span, 6);

        assert_eq!(byte_hits[0], 5); // from vertex 0
        assert_eq!(byte_hits[1], 5); // from vertex 0
        assert_eq!(byte_hits[2], 8); // from vertex 0 + 1 (overlap)
        assert_eq!(byte_hits[3], 3); // from vertex 1
        assert_eq!(byte_hits[4], 3); // from vertex 1
        assert_eq!(byte_hits[5], 0); // uncovered
    }

    #[test]
    fn test_hits_to_color() {
        let no_hits = hits_to_color(0, 100);
        assert_eq!(no_hits, "#442222");

        let max_hits = hits_to_color(100, 100);
        assert!(max_hits.contains("ff22") || max_hits.contains("00ff")); // green-ish
    }
}
