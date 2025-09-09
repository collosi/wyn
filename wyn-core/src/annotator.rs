// Code formatter that annotates Wyn source with location and basic block IDs
// Produces human-readable annotated code for debugging CFG extraction

use crate::ast::*;
use crate::cfg::{BlockId, Location};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Maps AST nodes to their location information
pub type LocationMap = HashMap<*const Expression, Location>;

/// Annotated code formatter
pub struct CodeAnnotator {
    location_map: LocationMap,
    output: String,
    current_block: Option<BlockId>,
    current_index: usize,
    next_block_id: usize,
}

impl CodeAnnotator {
    pub fn new() -> Self {
        Self {
            location_map: HashMap::new(),
            output: String::new(),
            current_block: None,
            current_index: 0,
            next_block_id: 0,
        }
    }
    
    /// Generate annotated source code showing block IDs and locations
    pub fn annotate_program(&mut self, program: &Program) -> String {
        self.output.clear();
        
        writeln!(self.output, "// === ANNOTATED WYN CODE ===").unwrap();
        writeln!(self.output, "// Format: #B<block_id>.<index> <original_code>").unwrap();
        writeln!(self.output, "").unwrap();
        
        // Process all declarations
        for decl in &program.declarations {
            self.annotate_declaration(decl);
            self.output.push('\n');
        }
        
        self.output.clone()
    }
    
    fn annotate_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Let(let_decl) => {
                let block = self.new_block();
                self.enter_block(block);
                
                write!(self.output, "#B{}.0 let {}", block.0, let_decl.name).unwrap();
                if let Some(ref ty) = let_decl.ty {
                    self.output.push_str(": ");
                    self.write_type(ty);
                }
                self.output.push_str(" = ");
                self.annotate_expression(&let_decl.value);
                
                self.exit_block();
            }
            
            Declaration::Entry(entry) => {
                let block = self.new_block();
                self.enter_block(block);
                
                write!(self.output, "#B{}.0 entry {}(", block.0, entry.name).unwrap();
                
                for (i, param) in entry.params.iter().enumerate() {
                    if i > 0 { self.output.push_str(", "); }
                    write!(self.output, "{}: ", param.name).unwrap();
                    self.write_type(&param.ty);
                }
                
                self.output.push_str("): ");
                self.write_type(&entry.return_type.ty);
                self.output.push_str(" =\n    ");
                
                self.annotate_expression(&entry.body);
                
                self.exit_block();
            }
            
            Declaration::Def(def) => {
                let block = self.new_block();
                self.enter_block(block);
                
                write!(self.output, "#B{}.0 def {}(", block.0, def.name).unwrap();
                
                for (i, param) in def.params.iter().enumerate() {
                    if i > 0 { self.output.push_str(", "); }
                    write!(self.output, "{}", param).unwrap();
                }
                
                self.output.push_str(") = ");
                
                self.annotate_expression(&def.body);
                
                self.exit_block();
            }
            
            Declaration::Val(val) => {
                write!(self.output, "val {}: ", val.name).unwrap();
                self.write_type(&val.ty);
            }
        }
    }
    
    fn annotate_expression(&mut self, expr: &Expression) {
        let loc = self.current_location();
        
        match expr {
            Expression::IntLiteral(n) => {
                write!(self.output, "#B{}.{} {}", loc.block.0, loc.index, n).unwrap();
                self.advance_index();
            }
            
            Expression::FloatLiteral(f) => {
                write!(self.output, "#B{}.{} {}", loc.block.0, loc.index, f).unwrap();
                self.advance_index();
            }
            
            Expression::Identifier(name) => {
                write!(self.output, "#B{}.{} {}", loc.block.0, loc.index, name).unwrap();
                self.advance_index();
            }
            
            Expression::ArrayLiteral(elements) => {
                write!(self.output, "#B{}.{} [", loc.block.0, loc.index).unwrap();
                
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 { self.output.push_str(", "); }
                    self.annotate_expression(elem);
                }
                
                self.output.push(']');
            }
            
            Expression::ArrayIndex(array, index) => {
                write!(self.output, "#B{}.{} ", loc.block.0, loc.index).unwrap();
                self.annotate_expression(array);
                self.output.push('[');
                self.annotate_expression(index);
                self.output.push(']');
            }
            
            Expression::BinaryOp(op, left, right) => {
                write!(self.output, "#B{}.{} (", loc.block.0, loc.index).unwrap();
                self.annotate_expression(left);
                
                let op_str = match op {
                    BinaryOp::Add => " + ",
                    BinaryOp::Divide => " / ",
                };
                self.output.push_str(op_str);
                
                self.annotate_expression(right);
                self.output.push(')');
            }
            
            Expression::FunctionCall(name, args) => {
                write!(self.output, "#B{}.{} {}(", loc.block.0, loc.index, name).unwrap();
                
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { self.output.push_str(", "); }
                    self.annotate_expression(arg);
                }
                
                self.output.push(')');
            }
            
            Expression::Application(func, args) => {
                write!(self.output, "#B{}.{} ", loc.block.0, loc.index).unwrap();
                self.annotate_expression(func);
                
                for arg in args {
                    self.output.push(' ');
                    self.annotate_expression(arg);
                }
            }
            
            Expression::Tuple(elements) => {
                write!(self.output, "#B{}.{} (", loc.block.0, loc.index).unwrap();
                
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 { self.output.push_str(", "); }
                    self.annotate_expression(elem);
                }
                
                self.output.push(')');
            }
            
            Expression::Lambda(lambda) => {
                write!(self.output, "#B{}.{} \\", loc.block.0, loc.index).unwrap();
                
                // Parameters
                for (i, param) in lambda.params.iter().enumerate() {
                    if i > 0 { self.output.push(' '); }
                    self.output.push_str(&param.name);
                    if let Some(ref ty) = param.ty {
                        self.output.push(':');
                        self.write_type(ty);
                    }
                }
                
                if let Some(ref ret_ty) = lambda.return_type {
                    self.output.push_str(": ");
                    self.write_type(ret_ty);
                }
                
                self.output.push_str(" -> ");
                
                // Create new block for lambda body
                let lambda_block = self.new_block();
                let saved_block = self.current_block;
                let saved_index = self.current_index;
                
                self.enter_block(lambda_block);
                self.annotate_expression(&lambda.body);
                self.exit_block();
                
                // Restore context
                self.current_block = saved_block;
                self.current_index = saved_index;
                self.advance_index();
            }
            
            Expression::LetIn(let_in) => {
                write!(self.output, "#B{}.{} let {} = ", loc.block.0, loc.index, let_in.name).unwrap();
                self.annotate_expression(&let_in.value);
                self.output.push_str(" in ");
                self.annotate_expression(&let_in.body);
                self.advance_index();
            }
        }
    }
    
    fn write_type(&mut self, ty: &polytype::Type) {
        // Simplified type formatting
        match ty {
            polytype::Type::Constructed(name, args) => {
                self.output.push_str(name);
                if !args.is_empty() {
                    self.output.push('<');
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 { self.output.push_str(", "); }
                        self.write_type(arg);
                    }
                    self.output.push('>');
                }
            }
            polytype::Type::Variable(var) => {
                write!(self.output, "'t{}", var).unwrap();
            }
        }
    }
    
    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }
    
    fn enter_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
        self.current_index = 0;
    }
    
    fn exit_block(&mut self) {
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
}

impl Default for CodeAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;
    
    #[test]
    fn test_simple_annotation() {
        let source = r#"entry main(x: i32): i32 = x + 1"#;
        
        // Parse the source
        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        // Annotate the code
        let mut annotator = CodeAnnotator::new();
        let annotated = annotator.annotate_program(&program);
        
        println!("Original: {}", source);
        println!("Annotated:\n{}", annotated);
        
        // Check that annotations are present
        assert!(annotated.contains("#B0.0"));
        assert!(annotated.contains("entry main"));
        assert!(annotated.contains(" + "));
    }
    
    #[test]
    fn test_lambda_annotation() {
        let source = r#"entry main(x: i32): i32 = 
            let f = \y -> y + x in
            f 10"#;
        
        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let mut annotator = CodeAnnotator::new();
        let annotated = annotator.annotate_program(&program);
        
        println!("Lambda annotation:\n{}", annotated);
        
        // Should have multiple blocks for lambda
        assert!(annotated.contains("#B0."));
        assert!(annotated.contains("#B1."));
        assert!(annotated.contains("\\y ->"));
    }
    
    #[test]
    fn test_complex_expression() {
        let source = r#"entry main(arr: [4]i32): i32 = 
            arr[0] + arr[1]"#;
        
        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let mut annotator = CodeAnnotator::new();
        let annotated = annotator.annotate_program(&program);
        
        println!("Complex expression:\n{}", annotated);
        
        // Should have block annotations for array access
        assert!(annotated.contains("arr["));
        assert!(annotated.contains("#B0."));
    }
}