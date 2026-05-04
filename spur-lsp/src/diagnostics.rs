use spur_core::analysis::checker::TypeError;
use spur_core::analysis::resolver::ResolutionError;
use spur_core::compiler::CompileResult;
use spur_core::lexer::LexError;
use spur_core::parser::{ParseError, ValidationError};
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
    for e in &result.validation_errors {
        out.push(validation_error_to_diagnostic(e, line_index));
    }
    for e in &result.resolution_errors {
        out.push(resolution_error_to_diagnostic(e, line_index));
    }
    for e in &result.type_errors {
        out.push(type_error_to_diagnostic(e, line_index));
    }
    for e in &result.refinement_errors {
        out.push(refinement_validation_error_to_diagnostic(e, line_index));
    }
    #[cfg(feature = "formulog")]
    for e in &result.refinement_check_errors {
        out.push(refinement_check_error_to_diagnostic(e, &result, line_index));
    }

    out
}

fn refinement_validation_error_to_diagnostic(
    e: &spur_liquid::RefinementValidationError,
    line_index: &LineIndex,
) -> Diagnostic {
    let span = e.span;
    Diagnostic {
        range: line_index.span_to_range(span.start, span.end),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("spur:refinement".into()),
        message: format!("{}", e),
        ..Default::default()
    }
}

#[cfg(feature = "formulog")]
fn refinement_check_error_to_diagnostic(
    e: &spur_liquid::RefinementCheckError,
    result: &CompileResult,
    line_index: &LineIndex,
) -> Diagnostic {
    // Prefer the precise sub-expression span the encoder resolved
    // through `id_to_span`. Fall back to the function's declaration
    // span when the failure was attributed to an atomic position (no
    // precise id) or when the encoder couldn't recover the expression
    // for some reason.
    let func_span = result
        .refinement_ir
        .as_ref()
        .and_then(|p| p.funcs.iter().find(|f| f.name == e.function).map(|f| f.span));
    let span = e.span.or(func_span).unwrap_or_default();

    let name = result
        .refinement_ir
        .as_ref()
        .and_then(|p| p.funcs.iter().find(|f| f.name == e.function).map(|f| f.original_name.clone()))
        .unwrap_or_else(|| format!("#{}", e.function.0));

    let detail = if e.span.is_some() {
        format!(
            "refinement check failed in function `{}`: this expression's type \
             does not satisfy its declared bound",
            name
        )
    } else {
        format!("function `{}` failed refinement check", name)
    };

    Diagnostic {
        range: line_index.span_to_range(span.start, span.end),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("spur:refinement".into()),
        message: detail,
        ..Default::default()
    }
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

fn validation_error_to_diagnostic(e: &ValidationError, line_index: &LineIndex) -> Diagnostic {
    let ValidationError::VarDeclInForIncrement(span) = e;
    Diagnostic {
        range: line_index.span_to_range(span.start, span.end),
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("spur".into()),
        message: e.to_string(),
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
        | FifoTargetNotRole { span, .. }
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
        | SafeNavOnNonOptional { span, .. }
        | InternalError { span, .. } => *span,
    }
}
