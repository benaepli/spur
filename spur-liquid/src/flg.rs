//! Serialization of `CProgram`s into Formulog input fact files.
//!
//! Formulog's compiled C++ binary reads input EDBs as TSV files where
//! each row's columns are textual term representations (the same syntax
//! you'd write in a `.flg` source file: `t_int`, `r_app(0, [r_var(1)])`,
//! `[1, 2]`, `(a, b)`, etc.) The two layers here are:
//!
//! - [`term`] — a small, hand-rolled term IR plus a renderer that
//!   produces strings the Formulog C++ parser accepts.
//! - [`encode`] — a `CProgram` walker that allocates the synthetic
//!   `NameId`s the `.flg` rules expect (one per field-of-struct,
//!   variant-of-enum, and tuple-accessor index) and emits one [`term`]
//!   per row of each input relation declared in `spur.flg`.
//!
//! See [`encode::encode_program`] for the entry point.

pub mod encode;
pub mod term;

pub use encode::{EncodedFacts, EncodeError, encode_program};
pub use term::Term;
