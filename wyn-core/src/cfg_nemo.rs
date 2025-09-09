// Enhanced CFG extractor that generates both Nemo facts and maintains the existing format
use crate::ast::*;
use crate::cfg::{BlockId, Location};
use crate::error::{CompilerError, Result};
use crate::nemo_facts::{NemoFactWriter, expr_type_name};
use std::io::Write;
use std::collections::HashMap;

/// CFG extractor that can output both regular facts and Nemo facts
pub struct CfgNemoExtractor<W: Write> {
    nemo_writer: NemoFactWriter<W>,
    next_block_id: usize,
    current_block: Option<BlockId>,
    current_index: usize,
    location_counter: usize,
    location_map: HashMap<Expression, usize>, // Maps expressions to location IDs
    debug: bool,
}

impl<W: Write> CfgNemoExtractor<W> {
    pub fn new(writer: W, debug: bool) -> Self {
        Self {
            nemo_writer: NemoFactWriter::new(writer, debug),
            next_block_id: 0,
            current_block: None,
            current_index: 0,
            location_counter: 1, // Start at 1 for cleaner fact IDs
            location_map: HashMap::new(),
            debug,
        }
    }
    
    pub fn extract_cfg(mut self, program: &Program) -> Result<()> {
        self.nemo_writer.write_header().map_err(Self::io_error)?;
        
        for decl in &program.declarations {
            self.extract_declaration_cfg(decl)?;
        }
        
        Ok(())
    }
    
    fn extract_declaration_cfg(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Entry(entry_decl) => {
                self.start_new_block()?; // Entry points start new blocks
                self.extract_expression_cfg(&entry_decl.body)?;
            }
            Declaration::Let(let_decl) => {
                if self.current_block.is_none() {
                    self.start_new_block()?; // Start block if needed
                }
                self.extract_expression_cfg(&let_decl.value)?;
            }
            Declaration::Def(def_decl) => {
                self.start_new_block()?; // Function definitions start new blocks
                self.extract_expression_cfg(&def_decl.body)?;
            }
            Declaration::Val(_val_decl) => {
                // Val declarations are type signatures, no body to process
            }
        }
        Ok(())
    }
    
    fn extract_expression_cfg(&mut self, expr: &Expression) -> Result<()> {
        let location_id = self.get_next_location_id();
        
        if let Some(current_block) = self.current_block {
            let location = Location {
                block: current_block,
                index: self.current_index,
            };
            
            // Write location fact
            self.nemo_writer.write_location_fact(location_id, &location).map_err(Self::io_error)?;
            
            // Write expression type fact
            let expr_type = expr_type_name(expr);
            self.nemo_writer.write_expr_fact(location_id, expr_type).map_err(Self::io_error)?;
            
            self.current_index += 1;
        }
        
        match expr {
            Expression::Identifier(name) => {
                // Variable reference
                self.nemo_writer.write_var_ref_fact(location_id, name).map_err(Self::io_error)?;
            }
            Expression::BinaryOp(_, left, right) => {
                self.extract_expression_cfg(left)?;
                self.extract_expression_cfg(right)?;
            }
            Expression::ArrayLiteral(elements) => {
                for element in elements {
                    self.extract_expression_cfg(element)?;
                }
            }
            Expression::ArrayIndex(array, index) => {
                self.extract_expression_cfg(array)?;
                self.extract_expression_cfg(index)?;
            }
            Expression::FunctionCall(_, args) => {
                for arg in args {
                    self.extract_expression_cfg(arg)?;
                }
            }
            Expression::Application(func, args) => {
                self.extract_expression_cfg(func)?;
                for arg in args {
                    self.extract_expression_cfg(arg)?;
                }
            }
            Expression::Tuple(elements) => {
                for element in elements {
                    self.extract_expression_cfg(element)?;
                }
            }
            // Lambda expressions create new basic blocks for their body
            Expression::Lambda(lambda) => {
                let lambda_block = self.start_new_block()?;
                
                // Write edge from current block to lambda block
                if let Some(current_block) = self.current_block {
                    self.nemo_writer.write_edge_fact(current_block, lambda_block).map_err(Self::io_error)?;
                }
                
                // Process lambda parameters as variable definitions
                for param in &lambda.params {
                    let param_location_id = self.get_next_location_id();
                    let location = Location {
                        block: lambda_block,
                        index: self.current_index,
                    };
                    
                    self.nemo_writer.write_location_fact(param_location_id, &location).map_err(Self::io_error)?;
                    self.nemo_writer.write_var_def_fact(param_location_id, &param.name).map_err(Self::io_error)?;
                    self.current_index += 1;
                }
                
                // Process lambda body
                self.extract_expression_cfg(&lambda.body)?;
                
                if self.debug {
                    eprintln!("DEBUG: Lambda created block {} for body", lambda_block.0);
                }
            }
            // Literals don't require further processing
            Expression::IntLiteral(_) | Expression::FloatLiteral(_) => {}
        }
        
        Ok(())
    }
    
    fn start_new_block(&mut self) -> Result<BlockId> {
        let block_id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        self.current_block = Some(block_id);
        self.current_index = 0;
        
        // Write block fact
        self.nemo_writer.write_block_fact(block_id).map_err(Self::io_error)?;
        
        if self.debug {
            eprintln!("DEBUG: Started new block {}", block_id.0);
        }
        
        Ok(block_id)
    }
    
    fn get_next_location_id(&mut self) -> usize {
        let id = self.location_counter;
        self.location_counter += 1;
        id
    }
    
    fn io_error(e: std::io::Error) -> CompilerError {
        CompilerError::SpirvError(format!("IO error during CFG extraction: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_nemo_cfg_extraction() {
        let source = r#"entry main(x: i32): i32 = x + 1"#;
        
        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let mut output = Vec::new();
        let extractor = CfgNemoExtractor::new(&mut output, true);
        extractor.extract_cfg(&program).unwrap();
        
        let result = String::from_utf8(output).unwrap();
        println!("Nemo CFG output:\n{}", result);
        
        // Should have block, location, and expression facts
        assert!(result.contains("block(0)."));
        assert!(result.contains("location("));
        assert!(result.contains("expr_at("));
    }

    #[test] 
    fn test_lambda_cfg_extraction() {
        let source = r#"let f: i32 -> i32 = \x -> x + 1"#;
        
        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let mut output = Vec::new();
        let extractor = CfgNemoExtractor::new(&mut output, true);
        extractor.extract_cfg(&program).unwrap();
        
        let result = String::from_utf8(output).unwrap();
        println!("Lambda Nemo CFG output:\n{}", result);
        
        // Should have multiple blocks and an edge between them
        assert!(result.contains("block(0)."));
        assert!(result.contains("block(1)."));
        assert!(result.contains("edge(0, 1)."));
        assert!(result.contains("var_def("));
    }
}