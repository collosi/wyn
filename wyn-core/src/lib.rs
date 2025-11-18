// pub mod annotator;
pub mod ast;
pub mod borrow_checker;
pub mod builtin_registry;
pub mod cfg;
pub mod cfg_nemo;
pub mod codegen;
pub mod constant_folding;
pub mod defunctionalization;
pub mod error;
pub mod inference;
pub mod lexer;
pub mod lowering;
pub mod mir;
pub mod mirize;
pub mod module;
pub mod monomorphize;
pub mod nemo_facts;
pub mod parser;
pub mod scope;
pub mod type_checker;
pub mod visitor;

#[cfg(test)]
mod mirize_tests;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod defunctionalization_tests;

#[cfg(test)]
mod type_checker_tests;

use error::Result;

pub struct Compiler;

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Compiler
    }

    /// Type check source code without generating SPIR-V
    pub fn check_only(&self, source: &str) -> Result<()> {
        // Tokenize
        let tokens = lexer::tokenize(source).map_err(error::CompilerError::ParseError)?;

        // Parse
        let mut parser = parser::Parser::new(tokens);
        let program = parser.parse()?;
        let node_counter = parser.take_node_counter();

        // Constant folding pass
        let mut constant_folder = constant_folding::ConstantFolder::new_with_counter(node_counter);
        let program = constant_folder.fold_program(&program)?;
        let node_counter = constant_folder.take_node_counter();

        // Defunctionalization pass
        let type_context = polytype::Context::default();
        let mut defunc =
            defunctionalization::Defunctionalizer::new_with_counter(node_counter, type_context);
        let program = defunc.defunctionalize_program(&program)?;
        let type_context = defunc.take_type_var_gen();

        // Type check
        let mut type_checker = type_checker::TypeChecker::new_with_context(type_context);
        type_checker.load_builtins()?;
        let _type_table = type_checker.check_program(&program)?;

        // Print warnings to stderr
        for warning in type_checker.warnings() {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| type_checker.format_type(t)),
                warning.span()
            );
        }

        Ok(())
    }

    pub fn compile(&self, source: &str) -> Result<Vec<u32>> {
        // Tokenize
        let tokens = lexer::tokenize(source).map_err(error::CompilerError::ParseError)?;

        // Parse
        let mut parser = parser::Parser::new(tokens);
        let program = parser.parse()?;
        let node_counter = parser.take_node_counter();

        // Constant folding pass
        let mut constant_folder = constant_folding::ConstantFolder::new_with_counter(node_counter);
        let program = constant_folder.fold_program(&program)?;
        let node_counter = constant_folder.take_node_counter();

        // Defunctionalization pass
        let type_context = polytype::Context::default();
        let mut defunc =
            defunctionalization::Defunctionalizer::new_with_counter(node_counter, type_context);
        let program = defunc.defunctionalize_program(&program)?;
        let type_context = defunc.take_type_var_gen();

        // Type check
        let mut type_checker = type_checker::TypeChecker::new_with_context(type_context);
        type_checker.load_builtins()?;
        let type_table = type_checker.check_program(&program)?;

        // Print warnings to stderr
        for warning in type_checker.warnings() {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| type_checker.format_type(t)),
                warning.span()
            );
        }

        // Monomorphization pass
        let monomorphizer = monomorphize::Monomorphizer::new(type_table.clone());
        let program = monomorphizer.monomorphize_program(&program)?;

        // Convert AST to MIR (Mid-level Intermediate Representation)
        let mirize = mirize::Mirize::new(type_table);
        let mir_module = mirize.mirize_program(&program)?;

        // Lower MIR to SPIR-V
        let lowering = lowering::Lowering::new();
        let spirv_bytes = lowering.lower_module(&mir_module)?;

        Ok(spirv_bytes)
    }
}
