use crate::analysis::resolver::ResolutionError;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use crate::analysis::checker::TypeError;

pub fn report_resolution_errors(
    source: &str,
    errors: &[ResolutionError],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut files = SimpleFiles::new();
    let file_id = files.add(filename, source);

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = term::Config::default();
    let mut writer_lock = writer.lock();

    for error in errors {
        let diagnostic = match error {
            ResolutionError::NameNotFound(name, span) => Diagnostic::error()
                .with_message(format!("cannot find name `{}` in this scope", name))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("not found in this scope"),
                ])
                .with_notes(vec![format!(
                    "help: check for typos or declare `{}` before its use",
                    name
                )]),

            ResolutionError::DuplicateName(name, span) => Diagnostic::error()
                .with_message(format!("name `{}` is defined more than once", name))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("duplicate definition"),
                ])
                .with_notes(vec![
                    "names must be unique within a given scope".to_string(),
                ]),

            ResolutionError::InvalidCrossRoleCall(name, span) => Diagnostic::error()
                .with_message("invalid cross-role function call")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("`{}` belongs to another role", name)),
                ])
                .with_notes(vec![
                    "help: Direct calls are only allowed to functions in the same role. \
                    To call a function in another role, use an `rpc_call`."
                        .to_string(),
                ]),
        };
        term::emit_to_write_style(&mut writer_lock, &config, &files, &diagnostic)?;
    }

    Ok(())
}

pub fn report_type_errors(
    source: &str,
    errors: &[TypeError],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut files = SimpleFiles::new();
    let file_id = files.add(filename, source);

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = term::Config::default();
    let mut writer_lock = writer.lock();

    for error in errors {
        let diagnostic = match error {
            TypeError::Mismatch { expected, found, span } => Diagnostic::error()
                .with_message(format!("type mismatch: expected `{}`, found `{}`", expected, found))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected `{}`, found `{}`", expected, found)),
                ])
                .with_notes(vec![
                    format!("help: consider converting the value to type `{}`", expected),
                ]),

            TypeError::UndefinedType(span) => Diagnostic::error()
                .with_message("undefined type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("type not defined"),
                ])
                .with_notes(vec![
                    "note: this error should have been caught by the name resolver".to_string(),
                ]),

            TypeError::InvalidUnaryOp { op, ty, span } => Diagnostic::error()
                .with_message(format!("invalid unary operation"))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("operator `{}` cannot be applied to type `{}`", op, ty)),
                ])
                .with_notes(vec![
                    format!("help: the operator `{}` is not defined for type `{}`", op, ty),
                ]),

            TypeError::InvalidBinOp { op, left, right, span } => Diagnostic::error()
                .with_message(format!("invalid binary operation"))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!(
                            "operator `{}` cannot be applied to types `{}` and `{}`",
                            op, left, right
                        )),
                ])
                .with_notes(vec![
                    format!("help: the operator `{}` is not defined for types `{}` and `{}`", op, left, right),
                ]),

            TypeError::WrongNumberOfArgs { expected, got, span } => Diagnostic::error()
                .with_message("wrong number of function arguments")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected {} argument{}, found {}",
                                              expected,
                                              if *expected == 1 { "" } else { "s" },
                                              got
                        )),
                ])
                .with_notes(vec![
                    format!("help: this function expects {} argument{}", expected, if *expected == 1 { "" } else { "s" }),
                ]),

            TypeError::NotAStruct { ty, field_name, span } => Diagnostic::error()
                .with_message("field access on non-struct type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("type `{}` has no fields", ty)),
                ])
                .with_notes(vec![
                    format!("help: only struct types have fields, but `{}` is not a struct", ty),
                    format!("note: cannot access field `{}` on this type", field_name),
                ]),

            TypeError::FieldNotFound { field_name, span } => Diagnostic::error()
                .with_message(format!("field `{}` not found", field_name))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("unknown field"),
                ])
                .with_notes(vec![
                    format!("help: check the struct definition for available fields"),
                ]),

            TypeError::NotIndexable { ty, span } => Diagnostic::error()
                .with_message("cannot index into this type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("type `{}` cannot be indexed", ty)),
                ])
                .with_notes(vec![
                    format!("help: only lists, maps, and strings can be indexed, but `{}` is not indexable", ty),
                ]),

            TypeError::InvalidIndexType { ty, span } => Diagnostic::error()
                .with_message("invalid index type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected `int`, found `{}`", ty)),
                ])
                .with_notes(vec![
                    "help: index values must be of type `int`".to_string(),
                ]),

            TypeError::InvalidMapKeyType { expected, found, span } => Diagnostic::error()
                .with_message("invalid map key type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected `{}`, found `{}`", expected, found)),
                ])
                .with_notes(vec![
                    format!("help: all keys in this map must be of type `{}`", expected),
                ]),

            TypeError::InvalidMapValueType { expected, found, span } => Diagnostic::error()
                .with_message("invalid map value type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected `{}`, found `{}`", expected, found)),
                ])
                .with_notes(vec![
                    format!("help: all values in this map must be of type `{}`", expected),
                ]),

            TypeError::InvalidAssignmentTarget(span) => Diagnostic::error()
                .with_message("invalid assignment target")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("cannot assign to this expression"),
                ])
                .with_notes(vec![
                    "help: only variables, fields, and index expressions can be assigned to".to_string(),
                ]),

            TypeError::MissingReturn(span) => Diagnostic::error()
                .with_message("missing return statement")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("function must return a value on all code paths"),
                ])
                .with_notes(vec![
                    "help: add a return statement or change the function's return type to `unit`".to_string(),
                ]),

            TypeError::ReturnOutsideFunction(span) => Diagnostic::error()
                .with_message("return statement outside of function")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("cannot return from here"),
                ])
                .with_notes(vec![
                    "help: return statements can only appear inside functions".to_string(),
                ]),

            TypeError::BreakOutsideLoop(span) => Diagnostic::error()
                .with_message("break statement outside of loop")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("cannot break from here"),
                ])
                .with_notes(vec![
                    "help: break statements can only appear inside loops".to_string(),
                ]),

            TypeError::NotIterable { ty, span } => Diagnostic::error()
                .with_message("cannot iterate over this type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("type `{}` is not iterable", ty)),
                ])
                .with_notes(vec![
                    format!("help: only lists and maps can be iterated over, but `{}` is not iterable", ty),
                ]),

            TypeError::PatternMismatch { pattern_ty, iterable_ty, span } => Diagnostic::error()
                .with_message("pattern does not match iterable type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!(
                            "expected pattern matching `{}`, found `{}`",
                            iterable_ty, pattern_ty
                        )),
                ])
                .with_notes(vec![
                    "help: ensure the loop pattern matches the type of elements being iterated".to_string(),
                ]),

            TypeError::TupleIndexOutOfBounds { index, size, span } => Diagnostic::error()
                .with_message("tuple index out of bounds")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("index {} is out of bounds for tuple of size {}", index, size)),
                ])
                .with_notes(vec![
                    format!("help: valid indices for this tuple are 0 to {}", size.saturating_sub(1)),
                ]),

            TypeError::NotATuple { ty, span } => Diagnostic::error()
                .with_message("tuple index on non-tuple type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("type `{}` is not a tuple", ty)),
                ])
                .with_notes(vec![
                    format!("help: only tuple types support index access with `.N` syntax"),
                ]),

            TypeError::UndefinedStructField { field_name, span } => Diagnostic::error()
                .with_message(format!("undefined struct field `{}`", field_name))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("field not defined in struct"),
                ])
                .with_notes(vec![
                    format!("help: check the struct definition for the correct field names"),
                ]),

            TypeError::MissingStructField { field_name, span } => Diagnostic::error()
                .with_message(format!("missing struct field `{}`", field_name))
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message("field must be initialized"),
                ])
                .with_notes(vec![
                    format!("help: add the missing field `{}` to the struct literal", field_name),
                ]),

            TypeError::RpcCallTargetNotRole { ty, span } => Diagnostic::error()
                .with_message("invalid RPC call target")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected role type, found `{}`", ty)),
                ])
                .with_notes(vec![
                    "help: RPC calls can only target role types".to_string(),
                ]),

            TypeError::NotAList { ty, span } => Diagnostic::error()
                .with_message("list operation on non-list type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected list type, found `{}`", ty)),
                ])
                .with_notes(vec![
                    format!("help: this operation requires a list type, but `{}` is not a list", ty),
                ]),
            
            TypeError::InvalidRoleOperation { ty, span } => Diagnostic::error()
                .with_message("invalid operation on role type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("cannot apply this operation to role type `{}`", ty)),
                ])
                .with_notes(vec![
                    "help: role types can only be used for RPC calls".to_string(),
                ]),

            TypeError::UnwrapOnNonOptional { ty, span } => Diagnostic::error()
                .with_message("cannot force-unwrap non-optional type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("type `{}` is not optional", ty)),
                ])
                .with_notes(vec![
                    format!("help: the force-unwrap operator `!` can only be used on optional types, but `{}` is not optional", ty),
                ]),

            TypeError::AwaitOnNonFuture { ty, span } => Diagnostic::error()
                .with_message("cannot await non-future type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected future type, found `{}`", ty)),
                ])
                .with_notes(vec![
                    format!("help: await can only be used on future types, but `{}` is not a future", ty),
                ]),

            TypeError::NotAPromise { ty, span } => Diagnostic::error()
                .with_message("promise operation on non-promise type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected promise type, found `{}`", ty)),
                ])
                .with_notes(vec![
                    format!("help: this operation requires a promise type, but `{}` is not a promise", ty),
                ]),

            TypeError::NotALock { ty, span } => Diagnostic::error()
                .with_message("cannot await lock on non-lock type")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected lock type, found `{}`", ty)),
                ])
                .with_notes(vec![
                    format!("help: await lock can only be used on lock types, but `{}` is not a lock", ty),
                ]),

            TypeError::PollingOnInvalidType { ty, span } => Diagnostic::error()
                .with_message("invalid polling operation")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected collection of futures or bools, found `{}`", ty)),
                ])
                .with_notes(vec![
                    format!("help: polling operations require a list or map of futures or bools, but `{}` does not match this requirement", ty),
                ]),

            TypeError::NextRespOnInvalidType { ty, span } => Diagnostic::error()
                .with_message("invalid next_resp operation")
                .with_labels(vec![
                    Label::primary(file_id, span.start..span.end)
                        .with_message(format!("expected map of futures, found `{}`", ty)),
                ])
                .with_notes(vec![
                    format!("help: next_resp requires a map where values are futures, but `{}` does not match this requirement", ty),
                ]),
        };

        term::emit_to_write_style(&mut writer_lock, &config, &files, &diagnostic)?;
    }

    Ok(())
}
