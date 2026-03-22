use crate::lexer::LexError;
use ariadne::{Color, Label, Report, ReportKind, Source};

/// Reports lexer errors using ariadne.
///
/// This function takes a source string and a list of lexer errors, and prints
/// nicely formatted error messages with source code context to stderr.
pub fn report_errors(
    source: &str,
    errors: &[LexError],
    filename: &str,
) -> Result<(), std::io::Error> {
    for error in errors {
        match error {
            LexError::UnexpectedChar(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("unexpected character")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("this character is not valid here")
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }
            LexError::UnterminatedString(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("unterminated string literal")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("string starts here but is never closed")
                            .with_color(Color::Red),
                    )
                    .with_note("string literals must be terminated with a closing quote")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }
        }
    }

    Ok(())
}
