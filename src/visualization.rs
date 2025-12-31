mod cfg;
mod coverage;

pub use cfg::render_svg;
pub use coverage::{render_html_heatmap, vertex_coverage_to_byte_coverage};
