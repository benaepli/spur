pub mod anf;
pub mod cfg;
pub mod lowered;
pub mod threaded;

use crate::analysis::checker::{TypeChecker, TypeError};
use crate::analysis::format::{report_resolution_errors, report_type_errors};
use crate::analysis::resolver::{ResolutionError, Resolver};
use crate::analysis::{trivially_copyable, type_id};
use crate::compiler::cfg::Compiler as CfgCompiler;
use crate::compiler::lowered::lower_program;
use crate::lexer::{LexError, Lexer};
use crate::parser::{ParseError, ValidationError, parse_program};
use crate::{lexer, parser};

/// Result of compilation that always carries diagnostics and optionally a program.
pub struct CompileResult {
    pub program: Option<cfg::Program>,
    pub lex_errors: Vec<LexError>,
    pub parse_errors: Vec<ParseError>,
    pub validation_errors: Vec<ValidationError>,
    pub resolution_errors: Vec<ResolutionError>,
    pub type_errors: Vec<TypeError>,
}

impl CompileResult {
    /// Returns true if any errors were encountered during compilation.
    pub fn has_errors(&self) -> bool {
        !self.lex_errors.is_empty()
            || !self.parse_errors.is_empty()
            || !self.validation_errors.is_empty()
            || !self.resolution_errors.is_empty()
            || !self.type_errors.is_empty()
    }

    /// Converts into a Result, returning the program if no errors, or an error otherwise.
    pub fn into_program(self) -> Result<cfg::Program, anyhow::Error> {
        if !self.lex_errors.is_empty() {
            return Err(self.lex_errors[0].clone().into());
        }
        if !self.parse_errors.is_empty() {
            return Err(anyhow::anyhow!("{}", self.parse_errors[0].message));
        }
        if !self.validation_errors.is_empty() {
            return Err(anyhow::anyhow!(
                "validation error: {}",
                self.validation_errors[0]
            ));
        }
        if !self.resolution_errors.is_empty() {
            return Err(anyhow::anyhow!(
                "resolution error: {}",
                self.resolution_errors[0]
            ));
        }
        if !self.type_errors.is_empty() {
            return Err(anyhow::anyhow!("type error: {}", self.type_errors[0]));
        }
        self.program
            .ok_or_else(|| anyhow::anyhow!("no output generated"))
    }
}

pub fn compile(input: &str, name: &str) -> CompileResult {
    let mut result = CompileResult {
        program: None,
        lex_errors: Vec::new(),
        parse_errors: Vec::new(),
        validation_errors: Vec::new(),
        resolution_errors: Vec::new(),
        type_errors: Vec::new(),
    };

    let mut lexer = Lexer::new(input);
    let (lexed, lex_errors) = lexer.collect_all();
    if !lex_errors.is_empty() {
        let _ = lexer::format::report_errors(input, &lex_errors, name);
    }
    result.lex_errors = lex_errors;

    let parsed = parse_program(&lexed);
    if parsed.has_errors() {
        let _ = parser::format::report_errors(input, parsed.errors(), name);
        result.parse_errors = parsed
            .errors()
            .map(|e| ParseError {
                message: e.to_string(),
                span: *e.span(),
            })
            .collect();
    }
    let parsed = match parsed.into_output() {
        None => return result,
        Some(v) => v,
    };

    let validation_errors = parser::validate_parsed(&parsed);
    if !validation_errors.is_empty() {
        let _ = parser::format::report_validation_errors(input, &validation_errors, name);
        result.validation_errors = validation_errors;
        return result;
    }

    let resolver = Resolver::new();
    let prepopulated_types = resolver.get_pre_populated_types().clone();
    let (resolved, resolution_errors) = resolver.resolve_program(parsed);
    if !resolution_errors.is_empty() {
        let _ = report_resolution_errors(input, &resolution_errors, name);
    }
    result.resolution_errors = resolution_errors;

    let mut type_checker = TypeChecker::new(prepopulated_types);
    let (typed, type_errors) = type_checker.check_program(resolved.clone());
    if !type_errors.is_empty() {
        let _ = report_type_errors(input, &type_errors, name);
        result.type_errors = type_errors;
        return result;
    }

    let _trivially_copyable_map =
        trivially_copyable::compute_trivially_copyable(&typed.struct_defs, &typed.enum_defs);

    let type_ids = type_id::assign_type_ids(&typed);

    let lowered = lower_program(typed);
    // lowered::remove_for_loops(&mut lowered);

    // ANF flattening (Phase 1)
    let anf = anf::lower_program(lowered.clone());

    // State threading (Phase 2) — runs but CFG backend still
    // consumes the original lowered output until full migration.
    let _threaded = threaded::lower_program(anf);

    let cfg_compiler = CfgCompiler::new();
    let program = cfg_compiler.compile_program(lowered, type_ids);
    result.program = Some(program);

    result
}

/// A version of compile tailored for the language server.
pub fn compile_lsp(input: &str) -> CompileResult {
    let mut result = CompileResult {
        program: None,
        lex_errors: Vec::new(),
        parse_errors: Vec::new(),
        validation_errors: Vec::new(),
        resolution_errors: Vec::new(),
        type_errors: Vec::new(),
    };

    let mut lexer = Lexer::new(input);
    let (lexed, lex_errors) = lexer.collect_all();
    result.lex_errors = lex_errors;

    let parsed = parse_program(&lexed);
    if parsed.has_errors() {
        result.parse_errors = parsed
            .errors()
            .map(|e| ParseError {
                message: e.to_string(),
                span: *e.span(),
            })
            .collect();
    }
    let parsed = match parsed.into_output() {
        None => return result,
        Some(v) => v,
    };

    let validation_errors = parser::validate_parsed(&parsed);
    if !validation_errors.is_empty() {
        result.validation_errors = validation_errors;
        return result;
    }

    let resolver = Resolver::new();
    let prepopulated_types = resolver.get_pre_populated_types().clone();
    let (resolved, resolution_errors) = resolver.resolve_program(parsed);
    result.resolution_errors = resolution_errors;

    let mut type_checker = TypeChecker::new(prepopulated_types);
    let (_typed, type_errors) = type_checker.check_program(resolved);
    result.type_errors = type_errors;

    result
}
