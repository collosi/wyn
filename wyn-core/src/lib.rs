// pub mod annotator;
pub mod ast;
pub mod builtin_registry;
pub mod diags;
pub mod error;
pub mod flattening;
pub mod inference;
pub mod lexer;
pub mod mir;
pub mod parser;
pub mod pattern;
pub mod scope;
pub mod type_checker;
pub mod visitor;

// Disabled for reorganization
#[cfg(any())]
pub mod borrow_checker;
#[cfg(any())]
pub mod cfg;
#[cfg(any())]
pub mod cfg_nemo;
pub mod constant_folding;
pub mod lowering;
#[cfg(any())]
pub mod module;
#[cfg(any())]
pub mod nemo_facts;

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
        let _node_counter = parser.take_node_counter();

        // Type check
        let mut type_checker = type_checker::TypeChecker::new();
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

    pub fn compile(&self, _source: &str) -> Result<Vec<u32>> {
        // Disabled during reorganization
        Err(error::CompilerError::SpirvError(
            "compile() disabled during reorganization".to_string(),
        ))
    }
}
