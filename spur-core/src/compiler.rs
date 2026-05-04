pub mod anf;
pub mod cfg;
pub mod lowered;
pub mod pure;
pub mod threaded;

use crate::analysis::checker::{TypeChecker, TypeError};
use crate::analysis::format::{report_resolution_errors, report_type_errors};
use crate::analysis::resolver::{ResolutionError, Resolver};
use crate::analysis::{trivially_copyable, type_id};
use crate::compiler::cfg::Compiler as CfgCompiler;
use crate::compiler::lowered::lower_program;
use crate::lexer::{LexError, Lexer};
use spur_ast::pure::PProgram;
use spur_liquid::RefinementValidationError;
use spur_liquid::lower_program as liquid_lower_program;

#[cfg(feature = "formulog")]
use spur_liquid::ir::CProgram;
#[cfg(feature = "formulog")]
use spur_liquid::{FormulogError, RefinementCheckError};
use crate::parser::{ParseError, ValidationError, parse_program};
use crate::{lexer, parser};

/// Result of compilation that always carries diagnostics and optionally a program.
pub struct CompileResult {
    pub program: Option<cfg::Program>,
    /// Pure / SSA IR. Populated by `compile` after a successful type check;
    /// `compile_lsp` leaves this `None`.
    pub pure: Option<PProgram>,
    /// Post-lowering refinement IR ([`spur_liquid::ir::CProgram`]).
    /// Populated whenever liquid lowering ran without bailing out;
    /// the caller can hand this to [`spur_liquid::check_with_formulog`]
    /// for full refinement checking.
    #[cfg(feature = "formulog")]
    pub refinement_ir: Option<CProgram>,
    pub lex_errors: Vec<LexError>,
    pub parse_errors: Vec<ParseError>,
    pub validation_errors: Vec<ValidationError>,
    pub resolution_errors: Vec<ResolutionError>,
    pub type_errors: Vec<TypeError>,
    pub refinement_errors: Vec<RefinementValidationError>,
    /// Per-function refinement check failures from the Formulog
    /// driver, populated only when `compile_with_refinements` ran the
    /// SMT-backed checker. Empty otherwise.
    #[cfg(feature = "formulog")]
    pub refinement_check_errors: Vec<RefinementCheckError>,
}

impl CompileResult {
    /// Returns true if any errors were encountered during compilation.
    pub fn has_errors(&self) -> bool {
        let staged = !self.lex_errors.is_empty()
            || !self.parse_errors.is_empty()
            || !self.validation_errors.is_empty()
            || !self.resolution_errors.is_empty()
            || !self.type_errors.is_empty()
            || !self.refinement_errors.is_empty();

        #[cfg(feature = "formulog")]
        return staged || !self.refinement_check_errors.is_empty();
        #[cfg(not(feature = "formulog"))]
        return staged;
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
        if !self.refinement_errors.is_empty() {
            return Err(anyhow::anyhow!(
                "refinement error: {}",
                self.refinement_errors[0]
            ));
        }
        self.program
            .ok_or_else(|| anyhow::anyhow!("no output generated"))
    }
}

pub fn compile(input: &str, name: &str) -> CompileResult {
    let mut result = CompileResult {
        program: None,
        pure: None,
        #[cfg(feature = "formulog")]
        refinement_ir: None,
        lex_errors: Vec::new(),
        parse_errors: Vec::new(),
        validation_errors: Vec::new(),
        resolution_errors: Vec::new(),
        type_errors: Vec::new(),
        refinement_errors: Vec::new(),
        #[cfg(feature = "formulog")]
        refinement_check_errors: Vec::new(),
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

    // Not used yet for compilation.
    let mut lowered_copy = lowered.clone();
    lowered::remove_for_loops(&mut lowered_copy);
    let anf = anf::lower_program(lowered_copy);
    let _threaded = threaded::lower_program(anf);
    let pure = pure::lower_program(_threaded);

    let liquid_out = liquid_lower_program(pure.clone());
    result.refinement_errors = liquid_out.refinement_errors;
    #[cfg(feature = "formulog")]
    {
        result.refinement_ir = Some(liquid_out.program);
    }
    let _ = liquid_out; // keep the binding alive for both cfgs
    result.pure = Some(pure);

    let cfg_compiler = CfgCompiler::new();
    let program = cfg_compiler.compile_program(lowered, type_ids);
    result.program = Some(program);

    result
}

/// A version of compile tailored for the language server.
pub fn compile_lsp(input: &str) -> CompileResult {
    let mut result = CompileResult {
        program: None,
        pure: None,
        #[cfg(feature = "formulog")]
        refinement_ir: None,
        lex_errors: Vec::new(),
        parse_errors: Vec::new(),
        validation_errors: Vec::new(),
        resolution_errors: Vec::new(),
        type_errors: Vec::new(),
        refinement_errors: Vec::new(),
        #[cfg(feature = "formulog")]
        refinement_check_errors: Vec::new(),
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

/// Run [`compile`] and, if no earlier-stage errors fired, hand the
/// lowered refinement IR to the Formulog driver. Returns the same
/// [`CompileResult`] but with `refinement_check_errors` populated.
///
/// The `flg_bin` argument should be the path to the Formulog-compiled
/// driver (typically [`spur_liquid::flg_binary_path`]); set `timeout`
/// to whatever budget you want to give the SMT-backed checker.
#[cfg(feature = "formulog")]
pub fn compile_with_refinements(
    input: &str,
    name: &str,
    flg_bin: &std::path::Path,
    timeout: std::time::Duration,
) -> Result<CompileResult, FormulogError> {
    let mut result = compile(input, name);

    if result.has_errors() {
        return Ok(result);
    }
    let Some(cprog) = &result.refinement_ir else {
        return Ok(result);
    };
    let fns: Vec<spur_ast::name::NameId> = cprog.funcs.iter().map(|f| f.name).collect();
    let errs = spur_liquid::check_with_formulog(cprog, &fns, flg_bin, timeout)?;
    result.refinement_check_errors = errs;
    Ok(result)
}

/// Run [`compile`] and, if no earlier-stage errors fired, hand the
/// lowered refinement IR to the native Rust refinement checker.
#[cfg(feature = "native-checker")]
pub fn compile_with_native_refinements(
    input: &str,
    name: &str,
) -> Result<NativeRefinementResult, spur_liquid::NativeCheckError> {
    let result = compile(input, name);
    if result.has_errors() {
        return Ok(NativeRefinementResult {
            compile_result: result,
            refinement_check_errors: Vec::new(),
        });
    }

    let liquid_out = liquid_lower_program(result.pure.as_ref().unwrap().clone());
    let fns: Vec<spur_ast::name::NameId> =
        liquid_out.program.funcs.iter().map(|f| f.name).collect();
    let errs = spur_liquid::check_native(&liquid_out.program, &fns)?;
    Ok(NativeRefinementResult {
        compile_result: result,
        refinement_check_errors: errs,
    })
}

#[cfg(feature = "native-checker")]
pub struct NativeRefinementResult {
    pub compile_result: CompileResult,
    pub refinement_check_errors: Vec<spur_liquid::native::RefinementCheckError>,
}
