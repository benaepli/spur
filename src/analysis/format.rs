use crate::analysis::resolver::ResolutionError;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

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
