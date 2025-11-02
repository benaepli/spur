pub mod cfg;

use crate::analysis::checker::TypeChecker;
use crate::analysis::format::{report_resolution_errors, report_type_errors};
use crate::analysis::resolver::Resolver;
use crate::compiler::cfg::Compiler as CfgCompiler;
use crate::lexer::Lexer;
use crate::parser::parse_program;
use crate::{lexer, parser};
use anyhow::anyhow;

pub fn compile(input: &str, name: &str) -> Result<cfg::Program, anyhow::Error> {
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
    let prepopulated_types = resolver.get_pre_populated_types().clone();
    let resolved = match resolver.resolve_program(parsed) {
        Err(e) => {
            let _ = report_resolution_errors(input, &[e], name);
            return Err(anyhow!("resolution error"));
        }
        Ok(r) => r,
    };

    let mut type_checker = TypeChecker::new(prepopulated_types);
    let typed = match type_checker.check_program(resolved.clone()) {
        Err(e) => {
            let _ = report_type_errors(input, &[e], name);
            return Err(anyhow!("type checking error"));
        }
        Ok(r) => r,
    };

    // Compile to CFG
    let cfg_compiler = CfgCompiler::new();
    let program = cfg_compiler.compile_program(typed);

    Ok(program)
}
