pub mod annotator;
pub mod ast;
pub mod borrow_checker;
pub mod builtins;
pub mod cfg;
pub mod cfg_nemo;
pub mod codegen;
pub mod defunctionalization;
pub mod error;
pub mod inference;
pub mod lexer;
pub mod nemo_facts;
pub mod parser;
pub mod scope;
pub mod type_checker;

#[cfg(test)]
mod integration_tests;

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

    pub fn compile(&self, source: &str) -> Result<Vec<u32>> {
        // Tokenize
        let tokens = lexer::tokenize(source).map_err(error::CompilerError::ParseError)?;

        // Parse
        let mut parser = parser::Parser::new(tokens);
        let program = parser.parse()?;

        // Type check
        let mut type_checker = type_checker::TypeChecker::new();
        type_checker.check_program(&program)?;

        // Defunctionalization (convert higher-order functions to first-order)
        let mut defunctionalizer = defunctionalization::Defunctionalizer::new();
        let defunctionalized_program = defunctionalizer.defunctionalize_program(&program)?;

        // Generate SPIR-V using rspirv
        let codegen = codegen::CodeGenerator::new("wyn_module");
        let raw_spirv = codegen.generate(&defunctionalized_program)?;
        
        // Post-process SPIR-V to fix compatibility issues (now handled within rspirv backend)
        Ok(raw_spirv)
    }
}
