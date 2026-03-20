use spur_core::analysis::checker::TypeError;
use spur_core::analysis::resolver::ResolutionError;
use spur_core::compiler::CompileResult;
use spur_core::lexer::LexError;
use spur_core::parser::ParseError;
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity};

use crate::convert::LineIndex;

/// Convert all errors in a `CompileResult` to LSP diagnostics.
pub fn compile_result_to_diagnostics(
    result: &CompileResult,
    line_index: &LineIndex,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();

    for e in &result.lex_errors {
        out.push(lex_error_to_diagnostic(e, line_index));
    }
    for e in &result.parse_errors {
        out.push(parse_error_to_diagnostic(e, line_index));
    }
    for e in &result.resolution_errors {
        out.push(resolution_error_to_diagnostic(e, line_index));
    }
    for e in &result.type_errors {
        out.push(type_error_to_diagnostic(e, line_index));
    }

    out
}

fn lex_error_to_diagnostic(e: &LexError, line_index: &LineIndex) -> Diagnostic {
    let span = match e {
        LexError::UnexpectedChar(s) => s,
        LexError::UnterminatedString(s) => s,
    };
    Diagnostic {
        range: line_index.span_to_range(span.start, span.end),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("spur".into()),
        message: e.to_string(),
        ..Default::default()
    }
}

fn parse_error_to_diagnostic(e: &ParseError, line_index: &LineIndex) -> Diagnostic {
    Diagnostic {
        range: line_index.span_to_range(e.span.start, e.span.end),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("spur".into()),
        message: e.message.clone(),
        ..Default::default()
    }
}

fn resolution_error_to_diagnostic(e: &ResolutionError, line_index: &LineIndex) -> Diagnostic {
    let span = match e {
        ResolutionError::NameNotFound(_, s) => s,
        ResolutionError::DuplicateName(_, s) => s,
    };
    Diagnostic {
        range: line_index.span_to_range(span.start, span.end),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("spur".into()),
        message: e.to_string(),
        ..Default::default()
    }
}

fn type_error_to_diagnostic(e: &TypeError, line_index: &LineIndex) -> Diagnostic {
    let span = type_error_span(e);
    Diagnostic {
        range: line_index.span_to_range(span.start, span.end),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("spur".into()),
        message: e.to_string(),
        ..Default::default()
    }
}

/// Extract the span from any `TypeError` variant.
fn type_error_span(e: &TypeError) -> spur_core::parser::Span {
    use TypeError::*;
    match e {
        Mismatch { span, .. }
        | UndefinedType(span)
        | InvalidUnaryOp { span, .. }
        | InvalidBinOp { span, .. }
        | WrongNumberOfArgs { span, .. }
        | NotAStruct { span, .. }
        | FieldNotFound { span, .. }
        | NotIndexable { span, .. }
        | InvalidIndexType { span, .. }
        | InvalidMapKeyType { span, .. }
        | InvalidStructKeyType { span, .. }
        | StoreOnInvalidType { span, .. }
        | InvalidAssignmentTarget(span)
        | MissingReturn(span)
        | ReturnOutsideFunction(span)
        | BreakOutsideLoop(span)
        | ContinueOutsideLoop(span)
        | NotIterable { span, .. }
        | PatternMismatch { span, .. }
        | TupleIndexOutOfBounds { span, .. }
        | NotATuple { span, .. }
        | UndefinedStructField { span, .. }
        | MissingStructField { span, .. }
        | RpcCallTargetNotRole { span, .. }
        | NotAList { span, .. }
        | UnwrapOnNonOptional { span, .. }
        | NotAChannel { span, .. }
        | RecvInSyncFunc { span, .. }
        | SendInSyncFunc { span, .. }
        | RpcCallToSyncFunc { span, .. }
        | EnumNotFound { span, .. }
        | VariantNotFound { span, .. }
        | VariantPayloadMismatch { span, .. }
        | VariantExpectsNoPayload { span, .. }
        | VariantExpectsPayload { span, .. }
        | MatchScrutineeNotEnum { span, .. }
        | MatchArmTypeMismatch { span, .. }
        | NonTriviallyCopyable { span, .. }
        | SafeNavOnNonOptional { span, .. } => *span,
    }
}
