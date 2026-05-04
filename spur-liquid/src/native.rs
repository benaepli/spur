mod check;
mod context;
mod smt;
mod subst;
mod subtype;
mod synth;

use spur_ast::name::NameId;
use spur_ast::span::Span;

use crate::ir::CProgram;

pub use context::{Env, GlobalCtx};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementCheckError {
    pub function: NameId,
    pub span: Option<Span>,
}

pub fn check_native(
    program: &CProgram,
    fns_to_check: &[NameId],
) -> Result<Vec<RefinementCheckError>, NativeCheckError> {
    let ctx = GlobalCtx::from_program(program);
    let mut solver = smt::SmtSolver::new()?;
    let mut errors = Vec::new();

    for &func_id in fns_to_check {
        let Some(func) = program.funcs.iter().find(|f| f.name == func_id) else {
            continue;
        };

        let env: Env = func
            .params
            .iter()
            .map(|p| (p.name, p.ty.clone()))
            .collect();
        let mut ctr = context::Counter::new(ctx.fresh_start);

        let result =
            check::check_block(&env, &ctx, &mut solver, &func.return_type, &func.body, &mut ctr, &func.return_type);

        if let CheckResult::Fail(span) = result {
            errors.push(RefinementCheckError {
                function: func_id,
                span: if span == Span::default() {
                    None
                } else {
                    Some(span)
                },
            });
        }
    }

    Ok(errors)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckResult {
    Ok,
    Fail(Span),
}

#[derive(Debug, thiserror::Error)]
pub enum NativeCheckError {
    #[error("could not spawn z3: {0}")]
    SolverSpawn(#[from] std::io::Error),
}
