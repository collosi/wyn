// pub mod annotator;
pub mod ast;
pub mod builtin_registry;
pub mod diags;
pub mod error;
pub mod inference;
pub mod lexer;
pub mod parser;
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
#[cfg(any())]
pub mod codegen;
#[cfg(any())]
pub mod constant_folding;
#[cfg(any())]
pub mod flattening;
#[cfg(any())]
pub mod lir;
#[cfg(any())]
pub mod lirize;
#[cfg(any())]
pub mod lowering;
#[cfg(any())]
pub mod module;
#[cfg(any())]
pub mod monomorphize;
#[cfg(any())]
pub mod nemo_facts;

// Disabled during reorganization - many tests depend on defunctionalization
#[cfg(any())]
mod type_checker_tests;

// Disabled test modules
#[cfg(any())]
mod defunctionalization_tests;
#[cfg(any())]
mod integration_tests;
#[cfg(any())]
mod mirize_tests;

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
