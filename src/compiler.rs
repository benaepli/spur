use crate::analysis::format::report_resolution_errors;
use crate::analysis::resolver::Resolver;
use crate::lexer::Lexer;
use crate::parser::parse_program;
use crate::{lexer, parser};
use anyhow::anyhow;

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
    let parsed = match parsed.into_output() {
        None => return Err(anyhow!("no output generated")),
        Some(v) => v,
    };
    let resolver = Resolver::new();
    let _ = match resolver.resolve_program(parsed) {
        Err(e) => {
            let _ = report_resolution_errors(input, &[e], name);
            return Err(anyhow!("resolution error"));
        }
        Ok(r) => r,
    };
    Ok(())
}
