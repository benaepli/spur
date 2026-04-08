use crate::compiler::lowered::ast as lowered;
use super::ast as threaded;

/// Transforms a lowered AST (`lowered::LProgram`) into a threaded AST (`threaded::TProgram`),
/// explicitly threading state as parameters to functions and making state-mutations explicit.
pub struct ThreadedTransformer {
    // Transformer state will go here
}

impl ThreadedTransformer {
    pub fn transform_program(program: lowered::LProgram) -> threaded::TProgram {
        // Pass implementation
        todo!("State threading AST transformation pass is not yet implemented")
    }
}
