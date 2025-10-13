// Enhanced CFG extractor that generates both Nemo facts and maintains the existing format
use crate::ast::*;
use crate::cfg::{BlockId, Location};
use crate::error::{CompilerError, Result};
use crate::nemo_facts::{NemoFactWriter, expr_type_name};
use crate::visitor::Visitor;
use std::io::Write;
use std::ops::ControlFlow;

/// Macro to convert Result to ControlFlow::Break on error
macro_rules! try_cf {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => return ControlFlow::Break(e),
        }
    };
}

/// CFG extractor that can output both regular facts and Nemo facts
pub struct CfgNemoExtractor<W: Write> {
    nemo_writer: NemoFactWriter<W>,
    next_block_id: usize,
    current_block: Option<BlockId>,
    current_index: usize,
    location_counter: usize,
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
            Declaration::Decl(decl) => {
                if decl.keyword == "let" && decl.params.is_empty() {
                    // Let variable binding
                    if self.current_block.is_none() {
                        self.start_new_block()?; // Start block if needed
                    }

                    // Use visitor to extract CFG
                    match self.visit_expression(&decl.body) {
                        ControlFlow::Continue(_) => {}
                        ControlFlow::Break(e) => return Err(e),
                    }
                } else {
                    // Function declaration or def variable
                    self.start_new_block()?; // Function definitions start new blocks

                    // Use visitor to extract CFG
                    match self.visit_expression(&decl.body) {
                        ControlFlow::Continue(_) => {}
                        ControlFlow::Break(e) => return Err(e),
                    }
                }
            }
            Declaration::Uniform(_) => {
                // Uniform declarations have no body to process
            }
            Declaration::Val(_val_decl) => {
                // Val declarations are type signatures, no body to process
            }
            Declaration::TypeBind(_) => {
                unimplemented!("Type bindings are not yet supported in CFG extraction")
            }
            Declaration::ModuleBind(_) => {
                unimplemented!("Module bindings are not yet supported in CFG extraction")
            }
            Declaration::ModuleTypeBind(_) => {
                unimplemented!("Module type bindings are not yet supported in CFG extraction")
            }
            Declaration::Open(_) => {
                unimplemented!("Open declarations are not yet supported in CFG extraction")
            }
            Declaration::Import(_) => {
                unimplemented!("Import declarations are not yet supported in CFG extraction")
            }
            Declaration::Local(_) => {
                unimplemented!("Local declarations are not yet supported in CFG extraction")
            }
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

impl<W: Write> Visitor for CfgNemoExtractor<W> {
    type Break = CompilerError;

    fn visit_expression(&mut self, expr: &Expression) -> ControlFlow<Self::Break> {
        let location_id = self.get_next_location_id();

        if let Some(current_block) = self.current_block {
            let location = Location {
                block: current_block,
                index: self.current_index,
            };

            // Write location fact
            try_cf!(self.nemo_writer.write_location_fact(location_id, &location).map_err(Self::io_error));

            // Write expression type fact
            let expr_type = expr_type_name(expr);
            try_cf!(self.nemo_writer.write_expr_fact(location_id, expr_type).map_err(Self::io_error));

            self.current_index += 1;
        }

        // Dispatch to specific handlers for special cases
        match &expr.kind {
            ExprKind::Identifier(name) => self.visit_expr_identifier(name),
            ExprKind::Lambda(lambda) => self.visit_expr_lambda(lambda),
            _ => {
                // Use default traversal for other expressions
                crate::visitor::walk_expression(self, expr)
            }
        }
    }

    fn visit_expr_identifier(&mut self, name: &str) -> ControlFlow<Self::Break> {
        let location_id = self.location_counter - 1;

        // Variable reference
        try_cf!(self.nemo_writer.write_var_ref_fact(location_id, name).map_err(Self::io_error));

        ControlFlow::Continue(())
    }

    fn visit_expr_lambda(&mut self, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        // Capture the current block BEFORE creating the new one
        let parent_block = self.current_block;
        let lambda_block = try_cf!(self.start_new_block());

        // Write edge from parent block to lambda block
        if let Some(parent) = parent_block {
            try_cf!(self.nemo_writer.write_edge_fact(parent, lambda_block).map_err(Self::io_error));
        }

        // Process lambda parameters as variable definitions
        for param in &lambda.params {
            let param_location_id = self.get_next_location_id();
            let location = Location {
                block: lambda_block,
                index: self.current_index,
            };

            try_cf!(
                self.nemo_writer.write_location_fact(param_location_id, &location).map_err(Self::io_error)
            );
            if let Some(param_name) = param.simple_name() {
                try_cf!(
                    self.nemo_writer
                        .write_var_def_fact(param_location_id, param_name)
                        .map_err(Self::io_error)
                );
            }
            self.current_index += 1;
        }

        // Process lambda body
        self.visit_expression(&lambda.body)?;

        if self.debug {
            eprintln!("DEBUG: Lambda created block {} for body", lambda_block.0);
        }

        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_nemo_cfg_extraction() {
        let source = r#"#[vertex] def main(x: i32): i32 = x + 1"#;

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
