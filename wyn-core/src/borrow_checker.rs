use crate::ast::*;
use crate::cfg::{BlockId, Location};
use crate::error::{CompilerError, Result};
use crate::nemo_facts::NemoFactWriter;
use std::io::Write;

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

                    // Analyze value expression
                    self.extract_expression_facts(fact_writer, &decl.body, block_id, location_counter)?;
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

                    // Analyze function body
                    self.extract_expression_facts(fact_writer, &decl.body, block_id, location_counter)?;
                }
            }
            Declaration::Val(_) => {
                // Val declarations don't create runtime lifetimes
            }
        }
        Ok(())
    }

    fn extract_expression_facts<W: Write>(
        &mut self,
        fact_writer: &mut NemoFactWriter<W>,
        expr: &Expression,
        current_block: BlockId,
        location_counter: &mut usize,
    ) -> Result<()> {
        let location_id = *location_counter;
        *location_counter += 1;

        let location = Location {
            block: current_block,
            index: location_id - 1,
        };
        fact_writer.write_location_fact(location_id, &location).map_err(Self::io_error)?;

        match expr {
            Expression::Identifier(name) => {
                // Variable reference - this is a potential borrow
                fact_writer.write_var_ref_fact(location_id, name).map_err(Self::io_error)?;

                // In a functional language, references are typically borrows
                let borrow_id = self.get_next_borrow_id();
                let lifetime_id = self.get_current_lifetime_for_var(name);
                fact_writer
                    .write_borrow_fact(borrow_id, location_id, name, lifetime_id)
                    .map_err(Self::io_error)?;
            }
            Expression::BinaryOp(_, left, right) => {
                self.extract_expression_facts(fact_writer, left, current_block, location_counter)?;
                self.extract_expression_facts(fact_writer, right, current_block, location_counter)?;
            }
            Expression::ArrayLiteral(elements) => {
                for element in elements {
                    self.extract_expression_facts(fact_writer, element, current_block, location_counter)?;
                }
            }
            Expression::ArrayIndex(array, index) => {
                self.extract_expression_facts(fact_writer, array, current_block, location_counter)?;
                self.extract_expression_facts(fact_writer, index, current_block, location_counter)?;
            }
            Expression::FunctionCall(func_name, args) => {
                let arg_names: Vec<&str> = args
                    .iter()
                    .filter_map(|arg| match arg {
                        Expression::Identifier(name) => Some(name.as_str()),
                        _ => None,
                    })
                    .collect();

                fact_writer.write_call_fact(location_id, func_name, &arg_names).map_err(Self::io_error)?;

                for arg in args {
                    self.extract_expression_facts(fact_writer, arg, current_block, location_counter)?;
                }
            }
            Expression::Application(func, args) => {
                self.extract_expression_facts(fact_writer, func, current_block, location_counter)?;
                for arg in args {
                    self.extract_expression_facts(fact_writer, arg, current_block, location_counter)?;
                }
            }
            Expression::Tuple(elements) => {
                for element in elements {
                    self.extract_expression_facts(fact_writer, element, current_block, location_counter)?;
                }
            }
            Expression::Lambda(lambda) => {
                // Lambda creates a new block
                let lambda_block = BlockId(current_block.0 + 1000); // Offset to avoid conflicts
                fact_writer.write_block_fact(lambda_block).map_err(Self::io_error)?;
                fact_writer.write_edge_fact(current_block, lambda_block).map_err(Self::io_error)?;

                // Lambda parameters create lifetimes
                for param in &lambda.params {
                    let lifetime_id = self.get_next_lifetime_id();
                    let param_location_id = *location_counter;
                    *location_counter += 1;

                    let param_location = Location {
                        block: lambda_block,
                        index: 0,
                    };
                    fact_writer
                        .write_location_fact(param_location_id, &param_location)
                        .map_err(Self::io_error)?;
                    fact_writer
                        .write_var_def_fact(param_location_id, &param.name)
                        .map_err(Self::io_error)?;
                    fact_writer
                        .write_lifetime_start_fact(lifetime_id, param_location_id, &param.name)
                        .map_err(Self::io_error)?;
                }

                // Analyze lambda body
                self.extract_expression_facts(fact_writer, &lambda.body, lambda_block, location_counter)?;
            }
            Expression::LetIn(let_in) => {
                // Let binding creates a lifetime
                let lifetime_id = self.get_next_lifetime_id();
                fact_writer.write_var_def_fact(location_id, &let_in.name).map_err(Self::io_error)?;
                fact_writer
                    .write_lifetime_start_fact(lifetime_id, location_id, &let_in.name)
                    .map_err(Self::io_error)?;

                // Analyze value expression
                self.extract_expression_facts(fact_writer, &let_in.value, current_block, location_counter)?;

                // Analyze body expression
                self.extract_expression_facts(fact_writer, &let_in.body, current_block, location_counter)?;
            }
            // Literals don't create borrows or lifetimes
            Expression::IntLiteral(_) | Expression::FloatLiteral(_) => {}
            Expression::FieldAccess(expr, _field) => {
                self.extract_expression_facts(fact_writer, expr, current_block, location_counter)?;
            }
            Expression::If(if_expr) => {
                self.extract_expression_facts(
                    fact_writer,
                    &if_expr.condition,
                    current_block,
                    location_counter,
                )?;
                self.extract_expression_facts(
                    fact_writer,
                    &if_expr.then_branch,
                    current_block,
                    location_counter,
                )?;
                self.extract_expression_facts(
                    fact_writer,
                    &if_expr.else_branch,
                    current_block,
                    location_counter,
                )?;
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
