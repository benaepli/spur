#![cfg(feature = "formulog")]

use std::time::Duration;

use spur_core::compiler::compile_with_refinements;
use spur_liquid::flg_binary_path;

fn flg_bin() -> std::path::PathBuf {
    flg_binary_path()
        .expect("SPUR_FLG_BIN must be set; build with --features formulog so build.rs runs")
}

/// Helper: locate the byte offsets of `needle` inside `haystack`. Used
/// to check that the diagnostic span covers exactly the offending
/// sub-expression in the source.
fn find_span(haystack: &str, needle: &str) -> std::ops::Range<usize> {
    let start = haystack
        .find(needle)
        .unwrap_or_else(|| panic!("expected to find {:?} in source", needle));
    start..start + needle.len()
}

#[test]
fn failed_refinement_pinpoints_sub_expression() {
    // The `var bad : int { v | v >= 0 } = -1;` line is the obvious
    // culprit: the annotated type rules out negative values, so the
    // subtype check on the right-hand side fails.
    let src = r#"
role Node {
    var x: int = 0;

    fn Init(me: int) {}

    fn failsRefine(): int {
        var bad: int { v | v >= 0 } = -1;
        return bad;
    }

    @trace
    async fn HandleRequest(req: int) {}
}

ClientInterface {
    async fn Write(dest: Node, key: string, value: string) { }
    async fn Read(dest: Node, key: string): string? { nil }
}
"#;

    let bin = flg_bin();
    let result = compile_with_refinements(src, "test.spur", &bin, Duration::from_secs(60))
        .expect("formulog driver returned an error");

    assert!(
        result.lex_errors.is_empty(),
        "lex errors: {:?}",
        result.lex_errors
    );
    assert!(
        result.parse_errors.is_empty(),
        "parse errors: {:?}",
        result.parse_errors
    );
    assert!(
        result.type_errors.is_empty(),
        "type errors: {:?}",
        result.type_errors
    );
    assert!(
        result.refinement_errors.is_empty(),
        "refinement validation errors: {:?}",
        result.refinement_errors
    );
    assert!(
        !result.refinement_check_errors.is_empty(),
        "expected at least one refinement check failure"
    );

    // Find the error attached to `failsRefine` (the function id changes
    // run-to-run, so look it up by name via refinement_ir).
    let cprog = result
        .refinement_ir
        .as_ref()
        .expect("refinement_ir should be populated when no earlier-stage errors fired");
    let target = cprog
        .funcs
        .iter()
        .find(|f| f.original_name == "failsRefine")
        .expect("`failsRefine` should be in refinement_ir.funcs");

    let err = result
        .refinement_check_errors
        .iter()
        .find(|e| e.function == target.name)
        .expect("expected a refinement check failure for `failsRefine`");

    let span = err
        .span
        .unwrap_or_else(|| panic!("RefinementCheckError.span should be populated, got None - the e_id pipeline did not attribute the failure to a specific expression"));

    let needle = find_span(src, "-1");
    assert!(
        span.start >= needle.start && span.end <= needle.end + 1,
        "expected span to be inside `-1` (byte range {:?}), got start={} end={}",
        needle,
        span.start,
        span.end,
    );
    // The function-level span would be much wider; ensure we're not
    // just falling back to it.
    let func_span_len = target.span.end - target.span.start;
    let err_span_len = span.end - span.start;
    assert!(
        err_span_len < func_span_len,
        "expected sub-expression span (len {}) to be tighter than function span (len {})",
        err_span_len,
        func_span_len,
    );
}
