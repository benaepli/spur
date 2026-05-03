//! Re-exports of the post-typecheck IR types.
//!
//! The actual definitions live in `spur-ast` so that other crates
//! (notably `spur-liquid`) can read them without depending on
//! `spur-core`.

pub use spur_ast::types::*;
