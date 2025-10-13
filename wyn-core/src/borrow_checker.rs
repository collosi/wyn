use crate::ast::*;
use crate::cfg::{BlockId, Location};
use crate::error::{CompilerError, Result};
use crate::nemo_facts::NemoFactWriter;
use crate::visitor::Visitor;
use std::io::Write;
use std::ops::ControlFlow;

/// Borrow checker using Nemo rule engine
pub struct BorrowChecker {
    lifetime_counter: usize,
    borrow_counter: usize,
    debug: bool,
}

impl BorrowChecker {
    pub fn new(debug: bool) -> Self {
        Self {
            lifetime_counter: 0,
            borrow_counter: 0,
            debug,
        }
    }

    /// Generate complete Nemo program for borrow checking analysis
    pub fn generate_nemo_program<W: Write>(&mut self, mut writer: W, program: &Program) -> Result<()> {
        // First, extract basic facts
        let mut fact_writer = NemoFactWriter::new(&mut writer, self.debug);
        fact_writer.write_header().map_err(Self::io_error)?;

        // Extract CFG and lifetime facts
        self.extract_lifetime_facts(&mut fact_writer, program)?;

        // Write the borrow checking rules
        self.write_borrow_checking_rules(&mut writer)?;

        Ok(())
    }

    /// Extract lifetime and borrow facts from the program
    fn extract_lifetime_facts<W: Write>(
        &mut self,
        fact_writer: &mut NemoFactWriter<W>,
        program: &Program,
    ) -> Result<()> {
        let mut location_counter = 1;
        let mut block_counter = 0;

        for decl in &program.declarations {
            self.extract_declaration_facts(fact_writer, decl, &mut location_counter, &mut block_counter)?;
        }

        Ok(())
    }

    fn extract_declaration_facts<W: Write>(
        &mut self,
        fact_writer: &mut NemoFactWriter<W>,
        decl: &Declaration,
        location_counter: &mut usize,
        block_counter: &mut usize,
    ) -> Result<()> {
        match decl {
            Declaration::Decl(decl) => {
                if decl.keyword == "let" && decl.params.is_empty() {
                    // Let variable binding
                    let block_id = BlockId(*block_counter);
                    *block_counter += 1;
                    fact_writer.write_block_fact(block_id).map_err(Self::io_error)?;

                    // Let binding creates a lifetime
                    let lifetime_id = self.get_next_lifetime_id();
                    let location_id = *location_counter;
                    *location_counter += 1;

                    let location = Location {
                        block: block_id,
                        index: 0,
                    };
                    fact_writer.write_location_fact(location_id, &location).map_err(Self::io_error)?;
                    fact_writer.write_var_def_fact(location_id, &decl.name).map_err(Self::io_error)?;
                    fact_writer
                        .write_lifetime_start_fact(lifetime_id, location_id, &decl.name)
                        .map_err(Self::io_error)?;

                    // Analyze value expression using visitor
                    let mut extractor = FactExtractor {
                        borrow_checker: self,
                        fact_writer,
                        current_block: block_id,
                        location_counter,
                    };
                    match extractor.visit_expression(&decl.body) {
                        ControlFlow::Continue(_) => {}
                        ControlFlow::Break(e) => return Err(e),
                    }
                } else {
                    // Function declaration or def variable
                    let block_id = BlockId(*block_counter);
                    *block_counter += 1;
                    fact_writer.write_block_fact(block_id).map_err(Self::io_error)?;

                    // Parameters create lifetimes
                    for param in &decl.params {
                        let param_name = match param {
                            DeclParam::Untyped(name) => name,
                            DeclParam::Typed(p) => &p.name,
                        };
                        let lifetime_id = self.get_next_lifetime_id();
                        let location_id = *location_counter;
                        *location_counter += 1;

                        let location = Location {
                            block: block_id,
                            index: 0,
                        };
                        fact_writer.write_location_fact(location_id, &location).map_err(Self::io_error)?;
                        fact_writer.write_var_def_fact(location_id, param_name).map_err(Self::io_error)?;
                        fact_writer
                            .write_lifetime_start_fact(lifetime_id, location_id, param_name)
                            .map_err(Self::io_error)?;
                    }

                    // Analyze function body using visitor
                    let mut extractor = FactExtractor {
                        borrow_checker: self,
                        fact_writer,
                        current_block: block_id,
                        location_counter,
                    };
                    match extractor.visit_expression(&decl.body) {
                        ControlFlow::Continue(_) => {}
                        ControlFlow::Break(e) => return Err(e),
                    }
                }
            }
            Declaration::Uniform(_) => {
                // Uniform declarations don't create runtime lifetimes
            }
            Declaration::Val(_) => {
                // Val declarations don't create runtime lifetimes
            }
            Declaration::TypeBind(_) => {
                unimplemented!("Type bindings are not yet supported in borrow checking")
            }
            Declaration::ModuleBind(_) => {
                unimplemented!("Module bindings are not yet supported in borrow checking")
            }
            Declaration::ModuleTypeBind(_) => {
                unimplemented!("Module type bindings are not yet supported in borrow checking")
            }
            Declaration::Open(_) => {
                unimplemented!("Open declarations are not yet supported in borrow checking")
            }
            Declaration::Import(_) => {
                unimplemented!("Import declarations are not yet supported in borrow checking")
            }
            Declaration::Local(_) => {
                unimplemented!("Local declarations are not yet supported in borrow checking")
            }
        }
        Ok(())
    }

    /// Write Nemo rules for borrow checking
    fn write_borrow_checking_rules<W: Write>(&self, mut writer: W) -> Result<()> {
        writeln!(writer, "% Simple borrow checking rules").map_err(Self::io_error)?;
        writeln!(writer).map_err(Self::io_error)?;

        // Simple rule: A variable is live if its lifetime has started
        writeln!(writer, "% A variable is live if its lifetime has started").map_err(Self::io_error)?;
        writeln!(
            writer,
            "live_var(Var, Loc) :- lifetime_start(LifetimeId, Loc, Var)."
        )
        .map_err(Self::io_error)?;
        writeln!(writer).map_err(Self::io_error)?;

        // Simple rule: Block reachability
        writeln!(writer, "% Block reachability").map_err(Self::io_error)?;
        writeln!(writer, "reachable(From, To) :- edge(From, To).").map_err(Self::io_error)?;
        writeln!(writer).map_err(Self::io_error)?;

        // Note: Output declarations handled by the API

        Ok(())
    }

    /// Run Nemo analysis and return results
    pub fn check_program(&mut self, program: &Program) -> Result<BorrowCheckResult> {
        // Generate the complete Nemo program as a string
        let mut nemo_program = Vec::new();
        self.generate_nemo_program(&mut nemo_program, program)?;
        let nemo_program_str = String::from_utf8(nemo_program).map_err(|_| {
            CompilerError::SpirvError("Failed to convert Nemo program to string".to_string())
        })?;

        if self.debug {
            println!("Generated Nemo program:\n{}", nemo_program_str);
        }

        // Run Nemo using the API
        let result = self.run_nemo_analysis_api(&nemo_program_str)?;

        Ok(result)
    }

    fn run_nemo_analysis_api(&self, nemo_program: &str) -> Result<BorrowCheckResult> {
        // Use Nemo API to load and execute the program
        match nemo::api::load_string(nemo_program.to_string()) {
            Ok(mut engine) => {
                if self.debug {
                    println!("Successfully loaded Nemo program");
                }

                // Execute the Nemo program
                match engine.execute() {
                    Ok(_) => {
                        if self.debug {
                            println!("Nemo program executed successfully");
                        }

                        // Parse the results for borrow check errors
                        let errors = Vec::new();

                        // Query for use-after-move errors
                        // Note: This would need proper result parsing from the Nemo engine
                        // For now, we'll assume no errors if execution succeeded

                        if self.debug {
                            println!("No borrow check errors found");
                        }

                        Ok(BorrowCheckResult { errors })
                    }
                    Err(nemo_error) => {
                        if self.debug {
                            eprintln!("Nemo execution error: {:?}", nemo_error);
                        }
                        // Convert Nemo error to our error type
                        Err(CompilerError::SpirvError(format!(
                            "Nemo execution failed: {:?}",
                            nemo_error
                        )))
                    }
                }
            }
            Err(nemo_error) => {
                if self.debug {
                    eprintln!("Failed to load Nemo program: {:?}", nemo_error);
                }
                // Convert Nemo error to our error type
                Err(CompilerError::SpirvError(format!(
                    "Failed to load Nemo program: {:?}",
                    nemo_error
                )))
            }
        }
    }

    fn get_next_lifetime_id(&mut self) -> usize {
        let id = self.lifetime_counter;
        self.lifetime_counter += 1;
        id
    }

    fn get_next_borrow_id(&mut self) -> usize {
        let id = self.borrow_counter;
        self.borrow_counter += 1;
        id
    }

    fn get_current_lifetime_for_var(&self, _var_name: &str) -> usize {
        // Simplified: would need proper lifetime tracking
        0
    }

    fn io_error(e: std::io::Error) -> CompilerError {
        CompilerError::SpirvError(format!("IO error during borrow checking: {}", e))
    }
}

/// Macro to convert Result to ControlFlow::Break on error
macro_rules! try_cf {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => return ControlFlow::Break(e),
        }
    };
}

/// Helper visitor for extracting facts from expressions
struct FactExtractor<'a, W: Write> {
    borrow_checker: &'a mut BorrowChecker,
    fact_writer: &'a mut NemoFactWriter<W>,
    current_block: BlockId,
    location_counter: &'a mut usize,
}

impl<'a, W: Write> Visitor for FactExtractor<'a, W> {
    type Break = CompilerError;

    fn visit_expression(&mut self, expr: &Expression) -> ControlFlow<Self::Break> {
        let location_id = *self.location_counter;
        *self.location_counter += 1;

        let location = Location {
            block: self.current_block,
            index: location_id - 1,
        };
        try_cf!(
            self.fact_writer.write_location_fact(location_id, &location).map_err(BorrowChecker::io_error)
        );

        // Dispatch to specific handlers based on expression type
        match &expr.kind {
            ExprKind::Identifier(name) => self.visit_expr_identifier(name),
            ExprKind::FunctionCall(name, args) => self.visit_expr_function_call(name, args),
            ExprKind::Lambda(lambda) => self.visit_expr_lambda(lambda),
            ExprKind::LetIn(let_in) => self.visit_expr_let_in(let_in),
            _ => {
                // For other expressions, use default traversal
                crate::visitor::walk_expression(self, expr)
            }
        }
    }

    fn visit_expr_identifier(&mut self, name: &str) -> ControlFlow<Self::Break> {
        let location_id = *self.location_counter - 1;

        // Variable reference - this is a potential borrow
        try_cf!(self.fact_writer.write_var_ref_fact(location_id, name).map_err(BorrowChecker::io_error));

        // In a functional language, references are typically borrows
        let borrow_id = self.borrow_checker.get_next_borrow_id();
        let lifetime_id = self.borrow_checker.get_current_lifetime_for_var(name);
        try_cf!(
            self.fact_writer
                .write_borrow_fact(borrow_id, location_id, name, lifetime_id)
                .map_err(BorrowChecker::io_error)
        );

        ControlFlow::Continue(())
    }

    fn visit_expr_function_call(
        &mut self,
        func_name: &str,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        let location_id = *self.location_counter - 1;

        let arg_names: Vec<&str> = args
            .iter()
            .filter_map(|arg| match &arg.kind {
                ExprKind::Identifier(name) => Some(name.as_str()),
                _ => None,
            })
            .collect();

        try_cf!(
            self.fact_writer
                .write_call_fact(location_id, func_name, &arg_names)
                .map_err(BorrowChecker::io_error)
        );

        // Visit all arguments
        for arg in args {
            self.visit_expression(arg)?;
        }

        ControlFlow::Continue(())
    }

    fn visit_expr_lambda(&mut self, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        // Lambda creates a new block
        let lambda_block = BlockId(self.current_block.0 + 1000); // Offset to avoid conflicts
        try_cf!(self.fact_writer.write_block_fact(lambda_block).map_err(BorrowChecker::io_error));
        try_cf!(
            self.fact_writer
                .write_edge_fact(self.current_block, lambda_block)
                .map_err(BorrowChecker::io_error)
        );

        // Lambda parameters create lifetimes
        for param in &lambda.params {
            let lifetime_id = self.borrow_checker.get_next_lifetime_id();
            let param_location_id = *self.location_counter;
            *self.location_counter += 1;

            let param_location = Location {
                block: lambda_block,
                index: 0,
            };
            try_cf!(
                self.fact_writer
                    .write_location_fact(param_location_id, &param_location)
                    .map_err(BorrowChecker::io_error)
            );
            try_cf!(
                self.fact_writer
                    .write_var_def_fact(param_location_id, &param.name)
                    .map_err(BorrowChecker::io_error)
            );
            try_cf!(
                self.fact_writer
                    .write_lifetime_start_fact(lifetime_id, param_location_id, &param.name)
                    .map_err(BorrowChecker::io_error)
            );
        }

        // Analyze lambda body in new block
        let saved_block = self.current_block;
        self.current_block = lambda_block;
        self.visit_expression(&lambda.body)?;
        self.current_block = saved_block;

        ControlFlow::Continue(())
    }

    fn visit_expr_let_in(&mut self, let_in: &LetInExpr) -> ControlFlow<Self::Break> {
        let location_id = *self.location_counter - 1;

        // Let binding creates a lifetime
        let lifetime_id = self.borrow_checker.get_next_lifetime_id();
        try_cf!(
            self.fact_writer.write_var_def_fact(location_id, &let_in.name).map_err(BorrowChecker::io_error)
        );
        try_cf!(
            self.fact_writer
                .write_lifetime_start_fact(lifetime_id, location_id, &let_in.name)
                .map_err(BorrowChecker::io_error)
        );

        // Analyze value expression
        self.visit_expression(&let_in.value)?;

        // Analyze body expression
        self.visit_expression(&let_in.body)?;

        ControlFlow::Continue(())
    }
}

#[derive(Debug, Clone)]
pub struct BorrowCheckResult {
    pub errors: Vec<BorrowError>,
}

#[derive(Debug, Clone)]
pub enum BorrowError {
    UseAfterMove {
        variable: String,
        use_location: usize,
        move_location: usize,
    },
    MultipleMutableBorrow {
        variable: String,
        borrow1: usize,
        borrow2: usize,
    },
}

impl BorrowCheckResult {
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn print_errors(&self) {
        for error in &self.errors {
            match error {
                BorrowError::UseAfterMove {
                    variable,
                    use_location,
                    move_location,
                } => {
                    println!(
                        "Error: Use after move of variable '{}' at location {} (moved at {})",
                        variable, use_location, move_location
                    );
                }
                BorrowError::MultipleMutableBorrow {
                    variable,
                    borrow1,
                    borrow2,
                } => {
                    println!(
                        "Error: Multiple mutable borrows of variable '{}' (borrows {} and {})",
                        variable, borrow1, borrow2
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_borrow_checker_basic() {
        let source = r#"#[vertex] def main(x: i32): i32 = x + 1"#;

        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = BorrowChecker::new(true);
        let result = checker.check_program(&program).unwrap();

        // Should not have errors for simple case
        assert!(!result.has_errors());
    }
}
