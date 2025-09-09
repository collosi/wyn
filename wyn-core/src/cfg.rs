// Control Flow Graph extraction for Nemo-based analysis
// Focuses on building accurate basic blocks for lifetime/borrow tracking

use crate::ast::*;
use crate::error::{CompilerError, Result};
use std::io::Write;

/// Basic block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

/// Location within a basic block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location {
    pub block: BlockId,
    pub index: usize,  // Position within the block
}

/// Control flow graph extractor
pub struct CfgExtractor<W: Write> {
    writer: W,
    next_block_id: usize,
    current_block: Option<BlockId>,
    current_index: usize,
    
    // Debug mode for annotations
    debug: bool,
}

impl<W: Write> CfgExtractor<W> {
    pub fn new(writer: W, debug: bool) -> Self {
        Self {
            writer,
            next_block_id: 0,
            current_block: None,
            current_index: 0,
            debug,
        }
    }
    
    pub fn extract_cfg(mut self, program: &Program) -> Result<()> {
        writeln!(self.writer, "% Control Flow Graph facts").map_err(Self::io_error)?;
        writeln!(self.writer, "% Basic blocks and control flow edges\n").map_err(Self::io_error)?;
        
        for decl in &program.declarations {
            self.extract_declaration_cfg(decl)?;
        }
        
        Ok(())
    }
    
    fn extract_declaration_cfg(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Entry(entry) => {
                self.extract_function_cfg(&entry.name, &entry.body)?;
            }
            Declaration::Def(def) => {
                self.extract_function_cfg(&def.name, &def.body)?;
            }
            Declaration::Let(let_decl) => {
                // Module-level let creates its own basic block
                let block = self.new_block();
                self.enter_block(block);
                
                self.write_fact("module_let", &[
                    &format!("\"{}\"", let_decl.name),
                    &block.0.to_string(),
                ])?;
                
                self.visit_expression(&let_decl.value)?;
                self.exit_block();
            }
            Declaration::Val(_) => {
                // Val declarations don't have control flow
            }
        }
        Ok(())
    }
    
    fn extract_function_cfg(&mut self, name: &str, body: &Expression) -> Result<()> {
        // Each function starts with an entry block
        let entry_block = self.new_block();
        
        self.write_fact("function_entry", &[
            &format!("\"{}\"", name),
            &entry_block.0.to_string(),
        ])?;
        
        self.enter_block(entry_block);
        
        // Process the function body
        self.visit_expression(body)?;
        
        // Current block is the exit block
        if let Some(current) = self.current_block {
            self.write_fact("function_exit", &[
                &format!("\"{}\"", name),
                &current.0.to_string(),
            ])?;
        }
        
        self.exit_block();
        Ok(())
    }
    
    fn visit_expression(&mut self, expr: &Expression) -> Result<()> {
        // Record this expression's location in the current block
        let loc = self.current_location();
        
        match expr {
            // Simple expressions stay in current block
            Expression::IntLiteral(_) | 
            Expression::FloatLiteral(_) | 
            Expression::Identifier(_) => {
                self.write_location_fact("expr", &loc, &format!("{:?}", expr))?;
                self.advance_index();
            }
            
            // Array and tuple literals - each element is evaluated in sequence
            Expression::ArrayLiteral(elements) | Expression::Tuple(elements) => {
                self.write_location_fact("expr_start", &loc, "array_or_tuple")?;
                for elem in elements {
                    self.visit_expression(elem)?;
                }
                self.write_location_fact("expr_end", &loc, "array_or_tuple")?;
            }
            
            // Binary operations - evaluate left, then right
            Expression::BinaryOp(op, left, right) => {
                self.write_location_fact("binop_start", &loc, &format!("{:?}", op))?;
                self.visit_expression(left)?;
                self.visit_expression(right)?;
                self.write_location_fact("binop_end", &loc, &format!("{:?}", op))?;
            }
            
            // Array indexing
            Expression::ArrayIndex(array, index) => {
                self.write_location_fact("array_index_start", &loc, "")?;
                self.visit_expression(array)?;
                self.visit_expression(index)?;
                self.write_location_fact("array_index_end", &loc, "")?;
            }
            
            // Function calls - evaluate arguments left to right
            Expression::FunctionCall(name, args) => {
                self.write_location_fact("call_start", &loc, name)?;
                for arg in args {
                    self.visit_expression(arg)?;
                }
                self.write_location_fact("call_end", &loc, name)?;
            }
            
            // Application (curried function call)
            Expression::Application(func, args) => {
                self.write_location_fact("apply_start", &loc, "")?;
                self.visit_expression(func)?;
                for arg in args {
                    self.visit_expression(arg)?;
                }
                self.write_location_fact("apply_end", &loc, "")?;
            }
            
            // Let-in expressions
            Expression::LetIn(let_in) => {
                self.write_location_fact("let", &loc, &let_in.name)?;
                self.visit_expression(&let_in.value)?;
                self.write_location_fact("in", &loc, &let_in.name)?;
                self.visit_expression(&let_in.body)?;
                self.advance_index();
            }
            
            // Lambda expressions create new basic blocks for their body
            Expression::Lambda(lambda) => {
                self.write_location_fact("lambda", &loc, "")?;
                
                // Create a new block for the lambda body
                let lambda_block = self.new_block();
                
                self.write_fact("lambda_body", &[
                    &loc.block.0.to_string(),
                    &loc.index.to_string(),
                    &lambda_block.0.to_string(),
                ])?;
                
                // Save current context
                let saved_block = self.current_block;
                let saved_index = self.current_index;
                
                // Process lambda body in new block
                self.enter_block(lambda_block);
                self.visit_expression(&lambda.body)?;
                self.exit_block();
                
                // Restore context
                self.current_block = saved_block;
                self.current_index = saved_index;
                self.advance_index();
            }
        }
        
        Ok(())
    }
    
    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        
        if self.debug {
            writeln!(self.writer, "% Created block B{}", id.0).ok();
        }
        
        id
    }
    
    fn enter_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
        self.current_index = 0;
        
        if self.debug {
            writeln!(self.writer, "% Entering block B{}", block.0).ok();
        }
    }
    
    fn exit_block(&mut self) {
        if self.debug {
            if let Some(block) = self.current_block {
                writeln!(self.writer, "% Exiting block B{}", block.0).ok();
            }
        }
        
        self.current_block = None;
        self.current_index = 0;
    }
    
    fn current_location(&self) -> Location {
        Location {
            block: self.current_block.expect("Not in a block"),
            index: self.current_index,
        }
    }
    
    fn advance_index(&mut self) {
        self.current_index += 1;
    }
    
    fn write_fact(&mut self, predicate: &str, args: &[&str]) -> Result<()> {
        write!(self.writer, "{}(", predicate).map_err(Self::io_error)?;
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                write!(self.writer, ", ").map_err(Self::io_error)?;
            }
            write!(self.writer, "{}", arg).map_err(Self::io_error)?;
        }
        writeln!(self.writer, ").").map_err(Self::io_error)?;
        Ok(())
    }
    
    fn write_location_fact(&mut self, predicate: &str, loc: &Location, info: &str) -> Result<()> {
        self.write_fact(predicate, &[
            &loc.block.0.to_string(),
            &loc.index.to_string(),
            &format!("\"{}\"", info),
        ])
    }
    
    fn io_error(e: std::io::Error) -> CompilerError {
        CompilerError::SpirvError(format!("IO error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_cfg() {
        let source = r#"
            entry main(x: i32): i32 = x + 1
        "#;
        
        // Parse the source
        let tokens = crate::lexer::tokenize(source).unwrap();
        let mut parser = crate::parser::Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        // Extract CFG to a string buffer
        let mut buffer = Vec::new();
        let extractor = CfgExtractor::new(&mut buffer, true);
        extractor.extract_cfg(&program).unwrap();
        
        let output = String::from_utf8(buffer).unwrap();
        println!("CFG output:\n{}", output);
        
        // Check that basic facts are present
        assert!(output.contains("function_entry"));
        assert!(output.contains("function_exit"));
        assert!(output.contains("binop_start"));
    }
    
    #[test]
    fn test_lambda_cfg() {
        let source = r#"
            entry main(x: i32): i32 = 
                let f = \y -> y + x in
                f(10)
        "#;
        
        let tokens = crate::lexer::tokenize(source).unwrap();
        let mut parser = crate::parser::Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let mut buffer = Vec::new();
        let extractor = CfgExtractor::new(&mut buffer, true);
        extractor.extract_cfg(&program).unwrap();
        
        let output = String::from_utf8(buffer).unwrap();
        println!("Lambda CFG output:\n{}", output);
        
        // Check for lambda body block
        assert!(output.contains("lambda_body"));
    }
}