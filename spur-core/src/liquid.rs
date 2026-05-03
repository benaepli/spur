//! Backwards-compatible shim for code paths that used to live under
//! `crate::liquid::*`.
//!
//! - The "core" refinement IR and lowering moved to the [`spur_liquid`] crate.
//! - The pure / SSA IR types live in `spur-ast` and are re-exported here at
//!   their old `crate::liquid::pure::ast::*` path.

pub mod core {
    //! Re-exports of the post-lowering refinement IR and validation passes,
    //! now hosted in the `spur-liquid` crate.
    pub use spur_liquid::ir as ast;
    pub use spur_liquid::lower;
    pub use spur_liquid::refinement;
    pub use spur_liquid::validate;
    pub use spur_liquid::{builtins, lower_program};
}

pub mod pure;
