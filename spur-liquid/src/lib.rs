//! Refinement / liquid type checking for spur, plus the Formulog driver.
//!
//! Owns the post-lowering ("core") refinement IR ([`ir`]), the lowering
//! pass from `PProgram` to [`ir::CProgram`] ([`lower`]), structural
//! validation of refinement bodies ([`validate`]), and (gated by the
//! `formulog` feature) the runtime that ships fact files to the
//! Formulog-generated `flg` binary and reads back diagnostics.

pub mod builtins;
pub mod cache;
pub mod flg;
#[cfg(feature = "formulog")]
pub mod formulog;
pub mod ir;
pub mod lower;
pub mod refinement;
pub mod validate;

pub use lower::{
    LowerOutput, RefinementValidationError, RefinementValidationErrorKind, lower_program,
};

#[cfg(feature = "formulog")]
pub use formulog::{
    FormulogError, RefinementCheckError, RefinementCheckErrorKind, check_with_formulog,
};

/// Path to the Formulog-generated `flg` binary, populated by `build.rs`
/// when the `formulog` feature is enabled. May be overridden at runtime
/// by setting the `SPUR_FLG_BIN` environment variable (useful for
/// pointing at a hand-built binary produced by `cargo xtask
/// formulog-codegen`).
pub fn flg_binary_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("SPUR_FLG_BIN") {
        return Some(std::path::PathBuf::from(p));
    }
    option_env!("SPUR_FLG_BIN").map(std::path::PathBuf::from)
}
