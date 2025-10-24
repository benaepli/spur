use anyhow::anyhow;
use crate::{lexer, parser};
use crate::lexer::Lexer;
use crate::parser::parse_program;

pub fn compile(input: &str, name: &str) -> Result<(), anyhow::Error> {
    let mut lexer = Lexer::new(input);
    let (lexed, errors) = lexer.collect_all();
    lexer::format::report_errors(input, &errors, name)?;
    if !errors.is_empty() {
        return Err(errors[0].clone().into());
    }
    let parsed = parse_program(&lexed);
    if parsed.has_errors() {
        parser::format::report_errors(input, parsed.errors(), name)?;
        return Err(anyhow!("parsing error"));
    }
    let _ = match parsed.into_output() {
        None => return Err(anyhow!("no output generated")),
        Some(v) => v,
    };
    Ok(())
}