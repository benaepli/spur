//! End-to-end smoke test: encode an empty `CProgram`, ship it to the
//! Formulog driver, and assert no errors come back.

#![cfg(feature = "formulog")]

use std::collections::HashMap;
use std::time::Duration;

use spur_liquid::ir::CProgram;
use spur_liquid::{check_with_formulog, flg_binary_path};

#[test]
fn empty_program_produces_no_diagnostics() {
    let bin = flg_binary_path().expect("SPUR_FLG_BIN must be set; build with --features formulog");

    let program = CProgram {
        funcs: vec![],
        extern_funcs: vec![],
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
        next_name_id: 0,
        id_to_name: HashMap::new(),
    };
    let result = check_with_formulog(&program, &[], &bin, Duration::from_secs(20));
    let errs = result.expect("flg pipeline failed");
    assert!(errs.is_empty(), "expected no errors, got {:?}", errs);
}
