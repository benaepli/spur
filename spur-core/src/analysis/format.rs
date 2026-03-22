use crate::analysis::checker::TypeError;
use crate::analysis::resolver::ResolutionError;
use ariadne::{Color, Label, Report, ReportKind, Source};

pub fn report_resolution_errors(
    source: &str,
    errors: &[ResolutionError],
    filename: &str,
) -> Result<(), std::io::Error> {
    for error in errors {
        match error {
            ResolutionError::NameNotFound(name, span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("cannot find name `{}` in this scope", name))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("not found in this scope")
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: check for typos or declare `{}` before its use",
                        name
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            ResolutionError::DuplicateName(name, span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("name `{}` is defined more than once", name))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("duplicate definition")
                            .with_color(Color::Red),
                    )
                    .with_note("names must be unique within a given scope")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }
        }
    }

    Ok(())
}

pub fn report_type_errors(
    source: &str,
    errors: &[TypeError],
    filename: &str,
) -> Result<(), std::io::Error> {
    for error in errors {
        match error {
            TypeError::Mismatch {
                expected,
                found,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!(
                        "type mismatch: expected `{}`, found `{}`",
                        expected, found
                    ))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected `{}`, found `{}`", expected, found))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: consider converting the value to type `{}`",
                        expected
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::UndefinedType(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("undefined type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("type not defined")
                            .with_color(Color::Red),
                    )
                    .with_note("note: this error should have been caught by the name resolver")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::InvalidUnaryOp { op, ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("invalid unary operation")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!(
                                "operator `{}` cannot be applied to type `{}`",
                                op, ty
                            ))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: the operator `{}` is not defined for type `{}`",
                        op, ty
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::InvalidBinOp {
                op,
                left,
                right,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("invalid binary operation")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!(
                                "operator `{}` cannot be applied to types `{}` and `{}`",
                                op, left, right
                            ))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: the operator `{}` is not defined for types `{}` and `{}`",
                        op, left, right
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::WrongNumberOfArgs {
                expected,
                got,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("wrong number of function arguments")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!(
                                "expected {} argument{}, found {}",
                                expected,
                                if *expected == 1 { "" } else { "s" },
                                got
                            ))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: this function expects {} argument{}",
                        expected,
                        if *expected == 1 { "" } else { "s" }
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::NotAStruct {
                ty,
                field_name,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("field access on non-struct type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("type `{}` has no fields", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(
                        vec![
                            format!(
                                "help: only struct types have fields, but `{}` is not a struct",
                                ty
                            ),
                            format!("note: cannot access field `{}` on this type", field_name),
                        ]
                        .join("\n"),
                    )
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::FieldNotFound { field_name, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("field `{}` not found", field_name))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("unknown field")
                            .with_color(Color::Red),
                    )
                    .with_note("help: check the struct definition for available fields")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::NotIndexable { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("cannot index into this type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("type `{}` cannot be indexed", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(format!("help: only lists, maps, and strings can be indexed, but `{}` is not indexable", ty))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::InvalidIndexType { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("invalid index type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected `int`, found `{}`", ty))
                            .with_color(Color::Red),
                    )
                    .with_note("help: index values must be of type `int`")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::InvalidMapKeyType {
                expected,
                found,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("invalid map key type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected `{}`, found `{}`", expected, found))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: all keys in this map must be of type `{}`",
                        expected
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::InvalidStructKeyType { found, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("invalid struct key type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected `string`, found `{}`", found))
                            .with_color(Color::Red),
                    )
                    .with_note("help: struct fields must be accessed with string literal keys")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::StoreOnInvalidType { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("store requires a list, map, or struct")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected list, map, or struct, found `{}`", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(format!("help: store can only be used on lists, maps, or structs, but `{}` is none of these", ty))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::InvalidAssignmentTarget(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("invalid assignment target")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("cannot assign to this expression")
                            .with_color(Color::Red),
                    )
                    .with_note("help: only variables can be assigned to")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::MissingReturn(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("missing return statement")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("function must return a value on all code paths")
                            .with_color(Color::Red),
                    )
                    .with_note("help: add a return statement or change the function's return type to `unit`")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::ReturnOutsideFunction(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("return statement outside of function")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("cannot return from here")
                            .with_color(Color::Red),
                    )
                    .with_note("help: return statements can only appear inside functions")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::BreakOutsideLoop(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("break statement outside of loop")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("cannot break from here")
                            .with_color(Color::Red),
                    )
                    .with_note("help: break statements can only appear inside loops")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::ContinueOutsideLoop(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("continue statement outside of loop")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("cannot continue from here")
                            .with_color(Color::Red),
                    )
                    .with_note("help: continue statements can only appear inside loops")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::NotIterable { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("cannot iterate over this type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("type `{}` is not iterable", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: only lists and maps can be iterated over, but `{}` is not iterable",
                        ty
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::PatternMismatch {
                pattern_ty,
                iterable_ty,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("pattern does not match iterable type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!(
                                "expected pattern matching `{}`, found `{}`",
                                iterable_ty, pattern_ty
                            ))
                            .with_color(Color::Red),
                    )
                    .with_note(
                        "help: ensure the loop pattern matches the type of elements being iterated",
                    )
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::TupleIndexOutOfBounds { index, size, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("tuple index out of bounds")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!(
                                "index {} is out of bounds for tuple of size {}",
                                index, size
                            ))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: valid indices for this tuple are 0 to {}",
                        size.saturating_sub(1)
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::NotATuple { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("tuple index on non-tuple type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("type `{}` is not a tuple", ty))
                            .with_color(Color::Red),
                    )
                    .with_note("help: only tuple types support index access with `.N` syntax")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::UndefinedStructField { field_name, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("undefined struct field `{}`", field_name))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("field not defined in struct")
                            .with_color(Color::Red),
                    )
                    .with_note("help: check the struct definition for the correct field names")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::MissingStructField { field_name, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("missing struct field `{}`", field_name))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("field must be initialized")
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: add the missing field `{}` to the struct literal",
                        field_name
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::RpcCallTargetNotRole { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("invalid RPC call target")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected role type, found `{}`", ty))
                            .with_color(Color::Red),
                    )
                    .with_note("help: RPC calls can only target role types")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::NotAList { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("list operation on non-list type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected list type, found `{}`", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: this operation requires a list type, but `{}` is not a list",
                        ty
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::UnwrapOnNonOptional { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("cannot force-unwrap non-optional type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("type `{}` is not optional", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(format!("help: the force-unwrap operator `!` can only be used on optional types, but `{}` is not optional", ty))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::NotAChannel { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("channel operation on non-channel type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected channel type, found `{}`", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(format!("help: the `<-` operator can only be used on channel types, but `{}` is not a channel", ty))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::RecvInSyncFunc { span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("recv cannot be used in sync function")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("recv (`<-`) not allowed here")
                            .with_color(Color::Red),
                    )
                    .with_note(vec![
                        "help: recv can only be used in async functions, but this function is marked as `sync`",
                        "note: remove the `sync` keyword from the function declaration to use recv"
                    ].join("\n"))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::SendInSyncFunc { span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("send cannot be used in sync function")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("send (`>-`) not allowed here")
                            .with_color(Color::Red),
                    )
                    .with_note(vec![
                        "help: send can only be used in async functions, but this function is marked as `sync`",
                        "note: remove the `sync` keyword from the function declaration to use send"
                    ].join("\n"))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::RpcCallToSyncFunc { func_name, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("RPC call to sync function not allowed")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("`{}` is a sync function", func_name))
                            .with_color(Color::Red),
                    )
                    .with_note(vec![
                        format!("help: RPC calls can only target async functions, but `{}` is marked as sync", func_name),
                        "note: consider removing the `sync` keyword from the target function".to_string()
                    ].join("\n"))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::EnumNotFound { name, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("enum `{}` not found", name))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("unknown enum")
                            .with_color(Color::Red),
                    )
                    .with_note("help: check that the enum is defined and in scope")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::VariantNotFound {
                enum_name,
                variant,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!(
                        "variant `{}` not found on enum `{}`",
                        variant, enum_name
                    ))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("unknown variant `{}`", variant))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: check the definition of `{}` for available variants",
                        enum_name
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::VariantPayloadMismatch {
                variant,
                expected,
                found,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("variant `{}` payload type mismatch", variant))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected `{}`, found `{}`", expected, found))
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: variant `{}` expects a payload of type `{}`",
                        variant, expected
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::VariantExpectsNoPayload { variant, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("variant `{}` expects no payload", variant))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("unexpected payload")
                            .with_color(Color::Red),
                    )
                    .with_note(format!("help: use `{}` without parentheses", variant))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::VariantExpectsPayload { variant, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("variant `{}` expects a payload", variant))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("missing payload")
                            .with_color(Color::Red),
                    )
                    .with_note(format!(
                        "help: use `{}(value)` with the expected payload",
                        variant
                    ))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::MatchScrutineeNotEnum { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("match scrutinee must be an enum")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected enum, found `{}`", ty))
                            .with_color(Color::Red),
                    )
                    .with_note("help: match expressions can only match on enum types")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::MatchArmTypeMismatch {
                expected,
                found,
                span,
            } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("match arms have incompatible types")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("expected `{}`, found `{}`", expected, found))
                            .with_color(Color::Red),
                    )
                    .with_note("help: all match arms must evaluate to the same type")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::NonTriviallyCopyable { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message(format!("type `{}` is not trivially copyable", ty))
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("type `{}` cannot be persisted or retrieved", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(vec![
                        "help: only trivially copyable types (int, string, bool, lists/maps/structs of these) can be used with persist_data and retrieve_data",
                        "note: channels and types containing channels are not trivially copyable"
                    ].join("\n"))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::SafeNavOnNonOptional { ty, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("safe navigation on non-optional type")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(format!("type `{}` is not optional", ty))
                            .with_color(Color::Red),
                    )
                    .with_note(vec![
                        format!("help: the `?.` and `?[]` operators can only be used on optional types, but `{}` is not optional", ty),
                        "note: use `.` or `[]` for non-optional access".to_string()
                    ].join("\n"))
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }

            TypeError::InternalError { message, span } => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("Internal compiler error")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message(message)
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }
        }
    }

    Ok(())
}
