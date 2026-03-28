use crate::lexer::TokenKind;
use crate::parser::ValidationError;
use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::prelude::Rich;

pub fn report_errors<'a, F>(source: &str, errors: F, filename: &str) -> Result<(), std::io::Error>
where
    F: Iterator<Item = &'a Rich<'a, TokenKind>>,
{
    for error in errors {
        let span = error.span();
        Report::build(ReportKind::Error, filename, span.start)
            .with_message(error.to_string())
            .with_label(
                Label::new((filename, span.start..span.end))
                    .with_message(error.to_string())
                    .with_color(Color::Red),
            )
            .finish()
            .eprint((filename, Source::from(source)))?;
    }

    Ok(())
}

pub fn report_validation_errors(
    source: &str,
    errors: &[ValidationError],
    filename: &str,
) -> Result<(), std::io::Error> {
    for error in errors {
        match error {
            ValidationError::VarDeclInForIncrement(span) => {
                Report::build(ReportKind::Error, filename, span.start)
                    .with_message("variable declaration in for-loop increment")
                    .with_label(
                        Label::new((filename, span.start..span.end))
                            .with_message("variable initialization not allowed here")
                            .with_color(Color::Red),
                    )
                    .with_note("use assignment (`=`) instead of declaration (`var`) in the increment clause")
                    .finish()
                    .eprint((filename, Source::from(source)))?;
            }
        }
    }

    Ok(())
}
