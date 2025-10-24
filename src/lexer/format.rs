use crate::lexer::LexError;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

/// Reports lexer errors using codespan-reporting.
///
/// This function takes a source string and a list of lexer errors, and prints
/// nicely formatted error messages with source code context to stderr.
pub fn report_errors(
    source: &str,
    errors: &[LexError],
    filename: &str,
) -> Result<(), codespan_reporting::files::Error> {
    let mut files = SimpleFiles::new();
    let file_id = files.add(filename, source);

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = term::Config::default();

    for error in errors {
        let diagnostic = match error {
            LexError::UnexpectedChar(pos) => Diagnostic::error()
                .with_message("unexpected character")
                .with_labels(vec![
                    Label::primary(file_id, *pos..*pos + 1)
                        .with_message("this character is not valid here"),
                ]),
            LexError::UnterminatedString(pos) => Diagnostic::error()
                .with_message("unterminated string literal")
                .with_labels(vec![
                    Label::primary(file_id, *pos..*pos + 1)
                        .with_message("string starts here but is never closed"),
                ])
                .with_notes(vec![
                    "string literals must be terminated with a closing quote".to_string(),
                ]),
        };

        term::emit_to_write_style(&mut writer.lock(), &config, &files, &diagnostic)?;
    }

    Ok(())
}
