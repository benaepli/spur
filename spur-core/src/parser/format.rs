use crate::lexer::TokenKind;
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
