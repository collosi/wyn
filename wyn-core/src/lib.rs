pub mod ast;
pub mod codegen;
pub mod error;
pub mod inference;
pub mod lexer;
pub mod parser;
pub mod type_checker;

#[cfg(test)]
mod integration_tests;

use error::Result;

pub struct Compiler;

impl Compiler {
    pub fn new() -> Self {
        Compiler
    }

    pub fn compile(&self, source: &str) -> Result<Vec<u32>> {
        use inkwell::context::Context;

        // Tokenize
        let tokens = lexer::tokenize(source).map_err(|e| error::CompilerError::ParseError(e))?;

        // Parse
        let mut parser = parser::Parser::new(tokens);
        let program = parser.parse()?;

        // Type check
        let mut type_checker = type_checker::TypeChecker::new();
        type_checker.check_program(&program)?;

        // Generate SPIR-V using LLVM/Inkwell
        let context = Context::create();
        let codegen = codegen::CodeGenerator::new(&context, "wyn_module");
        codegen.generate(&program)
    }
}
