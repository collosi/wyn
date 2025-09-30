// Code formatter that annotates Wyn source with location and basic block IDs
// Produces human-readable annotated code for debugging CFG extraction

use crate::ast::TypeName;
use crate::ast::*;
use crate::cfg::{BlockId, Location};
use std::fmt::Write as FmtWrite;

/// Annotated code formatter
pub struct CodeAnnotator {
    output: String,
    current_block: Option<BlockId>,
    current_index: usize,
    next_block_id: usize,
}

impl CodeAnnotator {
    pub fn new() -> Self {
        Self {
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
        writeln!(self.output).unwrap();

        // Process all declarations
        for decl in &program.declarations {
            self.annotate_declaration(decl);
            self.output.push('\n');
        }

        self.output.clone()
    }

    fn annotate_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Decl(decl_node) => {
                let block = self.new_block();
                self.enter_block(block);

                write!(
                    self.output,
                    "#B{}.0 {} {}",
                    block.0, decl_node.keyword, decl_node.name
                )
                .unwrap();

                // Add parameters if this is a function
                if !decl_node.params.is_empty() {
                    self.output.push('(');
                    for (i, param) in decl_node.params.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        match param {
                            DeclParam::Untyped(name) => write!(self.output, "{}", name).unwrap(),
                            DeclParam::Typed(p) => write!(self.output, "{}", p.name).unwrap(),
                        }
                    }
                    self.output.push(')');
                }

                // Add type annotation if present
                if let Some(ref ty) = decl_node.ty {
                    self.output.push_str(": ");
                    self.write_type(ty);
                }

                self.output.push_str(" = ");
                self.annotate_expression(&decl_node.body);

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
                    if i > 0 {
                        self.output.push_str(", ");
                    }
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

                let op_str = format!(" {} ", op.op);
                self.output.push_str(&op_str);

                self.annotate_expression(right);
                self.output.push(')');
            }

            Expression::FunctionCall(name, args) => {
                write!(self.output, "#B{}.{} {}(", loc.block.0, loc.index, name).unwrap();

                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
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
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.annotate_expression(elem);
                }

                self.output.push(')');
            }

            Expression::Lambda(lambda) => {
                write!(self.output, "#B{}.{} \\", loc.block.0, loc.index).unwrap();

                // Parameters
                for (i, param) in lambda.params.iter().enumerate() {
                    if i > 0 {
                        self.output.push(' ');
                    }
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
                write!(
                    self.output,
                    "#B{}.{} let {} = ",
                    loc.block.0, loc.index, let_in.name
                )
                .unwrap();
                self.annotate_expression(&let_in.value);
                self.output.push_str(" in ");
                self.annotate_expression(&let_in.body);
                self.advance_index();
            }
            Expression::FieldAccess(expr, _field) => {
                self.annotate_expression(expr);
            }
            Expression::If(if_expr) => {
                self.output.push_str("if ");
                self.annotate_expression(&if_expr.condition);
                self.output.push_str(" then ");
                self.annotate_expression(&if_expr.then_branch);
                self.output.push_str(" else ");
                self.annotate_expression(&if_expr.else_branch);
            }
        }
    }

    fn write_type(&mut self, ty: &Type) {
        // Simplified type formatting
        match ty {
            Type::Constructed(name, args) => {
                match name {
                    TypeName::Str(s) => self.output.push_str(s),
                    TypeName::Array(s, size) => {
                        self.output.push_str(&format!("{}[{}]", s, size));
                    }
                }
                if !args.is_empty() {
                    self.output.push('<');
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
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
        let source = r#"#[vertex] def main(x: i32): i32 = x + 1"#;

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
        assert!(annotated.contains("def main"));
        assert!(annotated.contains(" + "));
    }

    #[test]
    fn test_lambda_annotation() {
        let source = r#"#[vertex] def main(x: i32): i32 = 
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
        let source = r#"#[vertex] def main(arr: [4]i32): i32 = 
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
