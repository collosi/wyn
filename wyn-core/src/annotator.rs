// Code formatter that annotates Wyn source with location and basic block IDs
// Produces human-readable annotated code for debugging CFG extraction

use crate::ast::TypeName;
use crate::ast::*;
use crate::cfg::{BlockId, Location};
use crate::visitor::{Visitor, walk_expr_lambda};
use std::fmt::Write as FmtWrite;
use std::ops::ControlFlow;

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

        // Use visitor pattern to traverse
        let _ = self.visit_program(program);

        self.output.clone()
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

    fn write_type(&mut self, ty: &Type) {
        // Simplified type formatting
        match ty {
            Type::Constructed(name, args) => {
                match name {
                    TypeName::Str(s) => self.output.push_str(s),
                    TypeName::Array => {
                        // Array(Size(n), elem) - display as Array[n]<elem>
                        self.output.push_str("Array");
                    }
                    TypeName::Size(n) => {
                        self.output.push_str(&n.to_string());
                    }
                    TypeName::SizeVar(name) => {
                        self.output.push_str(name);
                    }
                    TypeName::Unique => {
                        self.output.push('*');
                    }
                    TypeName::Record(fields) => {
                        self.output.push('{');
                        // BTreeMap maintains sorted order automatically
                        for (i, (name, ty)) in fields.iter().enumerate() {
                            if i > 0 {
                                self.output.push_str(", ");
                            }
                            self.output.push_str(name);
                            self.output.push_str(": ");
                            self.write_type(ty);
                        }
                        self.output.push('}');
                    }
                    TypeName::Sum(variants) => {
                        for (i, (name, types)) in variants.iter().enumerate() {
                            if i > 0 {
                                self.output.push_str(" | ");
                            }
                            self.output.push_str(name);
                            for ty in types {
                                self.output.push(' ');
                                self.write_type(ty);
                            }
                        }
                    }
                    TypeName::Existential(vars, ty) => {
                        self.output.push('?');
                        for var in vars {
                            self.output.push('[');
                            self.output.push_str(var);
                            self.output.push(']');
                        }
                        self.output.push('.');
                        self.write_type(ty);
                    }
                    TypeName::NamedParam(name, ty) => {
                        self.output.push('(');
                        self.output.push_str(name);
                        self.output.push(':');
                        self.write_type(ty);
                        self.output.push(')');
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
}

impl Visitor for CodeAnnotator {
    type Break = ();

    fn visit_declaration(&mut self, decl: &Declaration) -> ControlFlow<Self::Break> {
        match decl {
            Declaration::Decl(decl_node) => {
                self.visit_decl(decl_node)?;
                self.output.push('\n');
            }
            Declaration::Uniform(uniform) => {
                self.visit_uniform_decl(uniform)?;
                self.output.push('\n');
            }
            Declaration::Sig(sig) => {
                self.visit_sig_decl(sig)?;
                self.output.push('\n');
            }
            Declaration::TypeBind(_) => {
                unimplemented!("Type bindings are not yet supported in code annotation")
            }
            Declaration::ModuleBind(_) => {
                unimplemented!("Module bindings are not yet supported in code annotation")
            }
            Declaration::ModuleTypeBind(_) => {
                unimplemented!("Module type bindings are not yet supported in code annotation")
            }
            Declaration::Open(_) => {
                unimplemented!("Open declarations are not yet supported in code annotation")
            }
            Declaration::Import(_) => {
                unimplemented!("Import declarations are not yet supported in code annotation")
            }
            Declaration::Local(_) => {
                unimplemented!("Local declarations are not yet supported in code annotation")
            }
        }
        ControlFlow::Continue(())
    }

    fn visit_decl(&mut self, d: &Decl) -> ControlFlow<Self::Break> {
        let block = self.new_block();
        self.enter_block(block);

        write!(self.output, "#B{}.0 {} {}", block.0, d.keyword, d.name).unwrap();

        // Add parameters if this is a function
        if !d.params.is_empty() {
            self.output.push('(');
            for (i, param) in d.params.iter().enumerate() {
                if i > 0 {
                    self.output.push_str(", ");
                }
                if let Some(name) = param.simple_name() {
                    write!(self.output, "{}", name).unwrap();
                }
            }
            self.output.push(')');
        }

        // Add type annotation if present
        if let Some(ref ty) = d.ty {
            self.output.push_str(": ");
            self.write_type(ty);
        }

        self.output.push_str(" = ");
        self.visit_expression(&d.body)?;

        self.exit_block();
        ControlFlow::Continue(())
    }

    fn visit_uniform_decl(&mut self, u: &UniformDecl) -> ControlFlow<Self::Break> {
        write!(self.output, "#[uniform] def {}: ", u.name).unwrap();
        self.write_type(&u.ty);
        ControlFlow::Continue(())
    }

    fn visit_sig_decl(&mut self, v: &SigDecl) -> ControlFlow<Self::Break> {
        write!(self.output, "sig {}: ", v.name).unwrap();
        self.write_type(&v.ty);
        ControlFlow::Continue(())
    }

    fn visit_expr_int_literal(&mut self, n: i32) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} {}", loc.block.0, loc.index, n).unwrap();
        self.advance_index();
        ControlFlow::Continue(())
    }

    fn visit_expr_float_literal(&mut self, f: f32) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} {}", loc.block.0, loc.index, f).unwrap();
        self.advance_index();
        ControlFlow::Continue(())
    }

    fn visit_expr_identifier(&mut self, name: &str) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} {}", loc.block.0, loc.index, name).unwrap();
        self.advance_index();
        ControlFlow::Continue(())
    }

    fn visit_expr_array_literal(&mut self, elements: &[Expression]) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} [", loc.block.0, loc.index).unwrap();

        for (i, elem) in elements.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            self.visit_expression(elem)?;
        }

        self.output.push(']');
        ControlFlow::Continue(())
    }

    fn visit_expr_array_index(
        &mut self,
        array: &Expression,
        index: &Expression,
    ) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} ", loc.block.0, loc.index).unwrap();
        self.visit_expression(array)?;
        self.output.push('[');
        self.visit_expression(index)?;
        self.output.push(']');
        ControlFlow::Continue(())
    }

    fn visit_expr_binary_op(
        &mut self,
        op: &BinaryOp,
        left: &Expression,
        right: &Expression,
    ) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} (", loc.block.0, loc.index).unwrap();
        self.visit_expression(left)?;

        let op_str = format!(" {} ", op.op);
        self.output.push_str(&op_str);

        self.visit_expression(right)?;
        self.output.push(')');
        ControlFlow::Continue(())
    }

    fn visit_expr_function_call(&mut self, name: &str, args: &[Expression]) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} {}(", loc.block.0, loc.index, name).unwrap();

        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            self.visit_expression(arg)?;
        }

        self.output.push(')');
        ControlFlow::Continue(())
    }

    fn visit_expr_tuple(&mut self, elements: &[Expression]) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} (", loc.block.0, loc.index).unwrap();

        for (i, elem) in elements.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            self.visit_expression(elem)?;
        }

        self.output.push(')');
        ControlFlow::Continue(())
    }

    fn visit_expr_lambda(&mut self, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} \\", loc.block.0, loc.index).unwrap();

        // Parameters
        for (i, param) in lambda.params.iter().enumerate() {
            if i > 0 {
                self.output.push(' ');
            }
            // For now, only handle simple name patterns
            if let Some(name) = param.simple_name() {
                self.output.push_str(name);
            } else {
                self.output.push_str("_pattern_");
            }
            if let Some(ref ty) = param.pattern_type() {
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
        // Use default walk for lambda body
        walk_expr_lambda(self, lambda)?;
        self.exit_block();

        // Restore context
        self.current_block = saved_block;
        self.current_index = saved_index;
        self.advance_index();

        ControlFlow::Continue(())
    }

    fn visit_expr_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(self.output, "#B{}.{} ", loc.block.0, loc.index).unwrap();
        self.visit_expression(func)?;

        for arg in args {
            self.output.push(' ');
            self.visit_expression(arg)?;
        }
        ControlFlow::Continue(())
    }

    fn visit_expr_let_in(&mut self, let_in: &LetInExpr) -> ControlFlow<Self::Break> {
        let loc = self.current_location();
        write!(
            self.output,
            "#B{}.{} let {} = ",
            loc.block.0, loc.index, let_in.name
        )
        .unwrap();
        self.visit_expression(&let_in.value)?;
        self.output.push_str(" in ");
        self.visit_expression(&let_in.body)?;
        self.advance_index();
        ControlFlow::Continue(())
    }

    fn visit_expr_field_access(&mut self, expr: &Expression, _field: &str) -> ControlFlow<Self::Break> {
        self.visit_expression(expr)
    }

    fn visit_expr_if(&mut self, if_expr: &IfExpr) -> ControlFlow<Self::Break> {
        self.output.push_str("if ");
        self.visit_expression(&if_expr.condition)?;
        self.output.push_str(" then ");
        self.visit_expression(&if_expr.then_branch)?;
        self.output.push_str(" else ");
        self.visit_expression(&if_expr.else_branch)?;
        ControlFlow::Continue(())
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
