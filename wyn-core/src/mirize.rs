//! MIRize pass - Convert typed AST to MIR
//!
//! This pass walks the typed AST (after type checking) and converts it to
//! the MIR representation. It relies on type annotations being present on
//! all expressions.

use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::mir::{self, FunctionId, Register};
use std::collections::HashMap;

pub struct Mirize {
    builder: mir::Builder,

    /// Map from AST variable names to MIR registers in current scope
    env: HashMap<String, Register>,

    /// Type table from type checker - maps NodeId to inferred Type
    type_table: HashMap<NodeId, Type>,

    /// Entry points for SPIR-V generation
    entry_points: Vec<FunctionId>,
}

impl Mirize {
    pub fn new(type_table: HashMap<NodeId, Type>) -> Self {
        Mirize {
            builder: mir::Builder::new(),
            env: HashMap::new(),
            type_table,
            entry_points: Vec::new(),
        }
    }

    pub fn mirize_program(mut self, program: &Program) -> Result<mir::Module> {
        // First pass: identify top-level constants
        let mut constant_decls = Vec::new();
        let mut function_decls = Vec::new();

        for decl in &program.declarations {
            if let Declaration::Decl(d) = decl {
                // Check if this is a top-level constant (def with no params, no entry point attributes)
                let is_entry_point =
                    d.attributes.iter().any(|attr| matches!(attr, Attribute::Vertex | Attribute::Fragment));

                if d.keyword == "def" && d.params.is_empty() && !is_entry_point {
                    // This is a top-level constant
                    constant_decls.push(d);
                } else {
                    // This is a function
                    function_decls.push(decl);
                }
            } else {
                function_decls.push(decl);
            }
        }

        // Generate _init() compute shader if there are constants
        if !constant_decls.is_empty() {
            self.generate_init_function(&constant_decls)?;
        }

        // Second pass: process functions
        for decl in &function_decls {
            self.mirize_declaration(decl)?;
        }

        Ok(self.builder.finish(self.entry_points))
    }

    /// Generate _init() compute shader that initializes top-level constants
    fn generate_init_function(&mut self, constant_decls: &[&Decl]) -> Result<()> {
        use crate::ast::TypeName;

        // Register all constants with the builder
        for decl in constant_decls {
            let ty = if let Some(ty) = &decl.ty {
                ty.clone()
            } else {
                // Use type from type table
                let node_id = decl.body.h.id;
                self.type_table
                    .get(&node_id)
                    .ok_or_else(|| {
                        CompilerError::MirError(format!(
                            "Type not found for constant '{}' in type table",
                            decl.name
                        ))
                    })?
                    .clone()
            };

            self.builder.register_constant(decl.name.clone(), ty);
        }

        // Create _init() function with void return
        let void_type = Type::Constructed(TypeName::Str("void"), vec![]);
        let func_id = self.builder.begin_function("_init".to_string(), vec![], void_type);

        // Mark as compute shader entry point
        self.entry_points.push(func_id);

        // TODO: Add compute entry point attribute to MIR

        // Compute each constant and store to buffer
        for decl in constant_decls {
            // Evaluate the constant expression
            let value_reg = self.mirize_expression(&decl.body)?;

            // Get the offset for this constant
            let offset = self.builder.get_constant_offset(&decl.name).ok_or_else(|| {
                CompilerError::MirError(format!("Constant '{}' not found in layout", decl.name))
            })?;

            // Store to buffer at offset
            self.builder.build_buffer_store(offset, value_reg);
        }

        // Return void
        self.builder.build_return_void();
        self.builder.end_function();

        Ok(())
    }

    fn mirize_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(d) => self.mirize_decl(d),
            Declaration::Uniform(_) => {
                // Uniforms will be handled separately in SPIR-V codegen
                Ok(())
            }
            Declaration::Val(v) => self.mirize_val_decl(v),
            Declaration::TypeBind(_) => Ok(()), // Type declarations don't produce MIR
            Declaration::ModuleBind(_) => {
                unimplemented!("Module bindings not yet supported in MIR")
            }
            Declaration::ModuleTypeBind(_) => Ok(()),
            Declaration::Open(_) => {
                unimplemented!("Open declarations not yet supported in MIR")
            }
            Declaration::Import(_) => Ok(()), // Imports handled at compile time
            Declaration::Local(_) => {
                unimplemented!("Local declarations not yet supported in MIR")
            }
        }
    }

    fn mirize_decl(&mut self, decl: &Decl) -> Result<()> {
        // Extract parameter types and return type
        let param_types: Vec<(String, Type)> = decl
            .params
            .iter()
            .map(|p| match p {
                DeclParam::Typed(param) => Ok((param.name.clone(), param.ty.clone())),
                DeclParam::Untyped(name) => Err(CompilerError::MirError(format!(
                    "Function parameter '{}' missing type annotation",
                    name
                ))),
            })
            .collect::<Result<Vec<_>>>()?;

        // Get return type - either from ty or from attributed_return_type
        let return_type = if let Some(ty) = &decl.ty {
            ty.clone()
        } else if let Some(attr_types) = &decl.attributed_return_type {
            // Build a tuple type from the attributed types
            if attr_types.len() == 1 {
                attr_types[0].ty.clone()
            } else {
                use crate::ast::TypeName;
                Type::Constructed(
                    TypeName::Str("tuple"),
                    attr_types.iter().map(|at| at.ty.clone()).collect(),
                )
            }
        } else {
            return Err(CompilerError::MirError(
                "Function missing return type annotation".to_string(),
            ));
        };

        // Check if this is an entry point
        let is_entry_point =
            decl.attributes.iter().any(|attr| matches!(attr, Attribute::Vertex | Attribute::Fragment));

        // Start building the function
        let func_id = self.builder.begin_function(decl.name.clone(), param_types.clone(), return_type);

        if is_entry_point {
            self.entry_points.push(func_id);
        }

        // Extract parameter attributes
        let param_attrs: Vec<Vec<Attribute>> = decl
            .params
            .iter()
            .map(|p| match p {
                DeclParam::Typed(param) => param.attributes.clone(),
                DeclParam::Untyped(_) => Vec::new(),
            })
            .collect();

        // Set function attributes before building body
        self.builder.set_function_attributes(
            param_attrs,
            decl.return_attributes.clone(),
            decl.attributed_return_type.clone(),
        );

        // Bind parameters in environment
        for (idx, (name, _)) in param_types.iter().enumerate() {
            let param_reg = self
                .builder
                .get_param(idx)
                .ok_or_else(|| CompilerError::MirError(format!("Parameter {} not found", idx)))?;
            self.env.insert(name.clone(), param_reg);
        }

        // Convert body expression to MIR
        let result_reg = self.mirize_expression(&decl.body)?;

        // Return the result
        self.builder.build_return(result_reg);

        // Finalize function
        self.builder.end_function();

        // Clear environment for next function
        self.env.clear();

        Ok(())
    }

    fn mirize_val_decl(&mut self, _val: &ValDecl) -> Result<()> {
        unimplemented!("Val declarations not yet supported in MIR")
    }

    fn mirize_expression(&mut self, expr: &Expression) -> Result<Register> {
        // Get the type of this expression from the type table
        let expr_type = self.type_table.get(&expr.h.id).cloned().ok_or_else(|| {
            CompilerError::MirError(format!("Expression {:?} missing type annotation", expr.h.id))
        })?;

        match &expr.kind {
            ExprKind::IntLiteral(n) => Ok(self.builder.build_const_int(*n, expr_type)),

            ExprKind::FloatLiteral(f) => Ok(self.builder.build_const_float(*f, expr_type)),

            ExprKind::Identifier(name) => {
                // Check if this is a constant reference
                if let Some(offset) = self.builder.get_constant_offset(name) {
                    // Load from constants buffer
                    Ok(self.builder.build_buffer_load(offset, expr_type))
                } else {
                    // Regular variable lookup
                    self.env
                        .get(name)
                        .cloned()
                        .ok_or_else(|| CompilerError::MirError(format!("Undefined variable: {}", name)))
                }
            }

            ExprKind::ArrayLiteral(elements) => {
                let element_regs: Vec<Register> =
                    elements.iter().map(|e| self.mirize_expression(e)).collect::<Result<Vec<_>>>()?;

                Ok(self.builder.build_array(element_regs, expr_type))
            }

            ExprKind::ArrayIndex(array, index) => {
                let array_reg = self.mirize_expression(array)?;
                let index_reg = self.mirize_expression(index)?;

                Ok(self.builder.build_array_index(array_reg, index_reg, expr_type))
            }

            ExprKind::BinaryOp(op, left, right) => {
                let left_reg = self.mirize_expression(left)?;
                let right_reg = self.mirize_expression(right)?;

                match op.op.as_str() {
                    "+" => Ok(self.builder.build_add(left_reg, right_reg)),
                    "-" => Ok(self.builder.build_sub(left_reg, right_reg)),
                    "*" => Ok(self.builder.build_mul(left_reg, right_reg)),
                    "/" => Ok(self.builder.build_div(left_reg, right_reg)),
                    "==" => Ok(self.builder.build_eq(left_reg, right_reg)),
                    "!=" => Ok(self.builder.build_ne(left_reg, right_reg)),
                    "<" => Ok(self.builder.build_lt(left_reg, right_reg)),
                    "<=" => Ok(self.builder.build_le(left_reg, right_reg)),
                    ">" => Ok(self.builder.build_gt(left_reg, right_reg)),
                    ">=" => Ok(self.builder.build_ge(left_reg, right_reg)),
                    _ => Err(CompilerError::MirError(format!(
                        "Unknown binary operator: {}",
                        op.op
                    ))),
                }
            }

            ExprKind::FunctionCall(func_name, args) => {
                let arg_regs: Vec<Register> =
                    args.iter().map(|a| self.mirize_expression(a)).collect::<Result<Vec<_>>>()?;

                Ok(self.builder.build_call(func_name, arg_regs, expr_type))
            }

            ExprKind::Tuple(elements) => {
                let element_regs: Vec<Register> =
                    elements.iter().map(|e| self.mirize_expression(e)).collect::<Result<Vec<_>>>()?;

                Ok(self.builder.build_tuple(element_regs, expr_type))
            }

            ExprKind::Lambda(_) => Err(CompilerError::MirError(
                "Lambda expressions should be eliminated by defunctionalization".to_string(),
            )),

            ExprKind::LetIn(let_in_expr) => {
                // Save current environment
                let saved_env = self.env.clone();

                // Process binding
                let value_reg = self.mirize_expression(&let_in_expr.value)?;
                self.env.insert(let_in_expr.name.clone(), value_reg);

                // Process body
                let result_reg = self.mirize_expression(&let_in_expr.body)?;

                // Restore environment
                self.env = saved_env;

                Ok(result_reg)
            }

            ExprKind::Application(_, _) => Err(CompilerError::MirError(
                "Application should be eliminated by defunctionalization".to_string(),
            )),

            ExprKind::FieldAccess(record, field_name) => {
                let record_reg = self.mirize_expression(record)?;
                // TODO: Look up field index from record type
                // For now, map field names to indices (x=0, y=1, z=2, w=3)
                let field_idx = match field_name.as_str() {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    _ => return Err(CompilerError::MirError(format!("Unknown field: {}", field_name))),
                };
                Ok(self.builder.build_extract_element(record_reg, field_idx, expr_type))
            }

            ExprKind::If(if_expr) => {
                // Evaluate condition
                let cond_reg = self.mirize_expression(&if_expr.condition)?;

                // Allocate block IDs: then, else, and reserve merge block ID
                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let merge_block_id = self.builder.new_block_id(); // Reserve ID but don't create yet

                // Branch on condition with merge block annotation
                self.builder.build_branch_cond(cond_reg, then_block, else_block, merge_block_id);

                // Then block
                self.builder.select_block(then_block);
                let then_reg = self.mirize_expression(&if_expr.then_branch)?;
                let then_end_block = self.builder.current_block().unwrap();

                // Else block
                self.builder.select_block(else_block);
                let else_reg = self.mirize_expression(&if_expr.else_branch)?;
                let else_end_block = self.builder.current_block().unwrap();

                // NOW create the merge block (after nested blocks are created)
                let merge_block = self.builder.create_block_with_id(merge_block_id);

                // Add branches from both end blocks to merge
                self.builder.select_block(then_end_block);
                self.builder.build_branch(merge_block);

                self.builder.select_block(else_end_block);
                self.builder.build_branch(merge_block);

                // Merge block with phi
                self.builder.select_block(merge_block);
                let result_reg = self.builder.build_phi(
                    vec![(then_reg, then_end_block), (else_reg, else_end_block)],
                    expr_type,
                );

                Ok(result_reg)
            }

            // New expression kinds - to be implemented
            ExprKind::TypeHole => {
                todo!("TypeHole not yet implemented in MIR")
            }

            ExprKind::QualifiedName(_, _) => {
                todo!("QualifiedName not yet implemented in MIR")
            }

            ExprKind::UnaryOp(_, _) => {
                todo!("UnaryOp not yet implemented in MIR")
            }

            ExprKind::Loop(_) => {
                todo!("Loop not yet implemented in MIR")
            }

            ExprKind::Match(_) => {
                todo!("Match not yet implemented in MIR")
            }

            ExprKind::Range(_) => {
                todo!("Range not yet implemented in MIR")
            }

            ExprKind::Pipe(_, _) => {
                todo!("Pipe not yet implemented in MIR")
            }

            ExprKind::TypeAscription(_, _) => {
                todo!("TypeAscription not yet implemented in MIR")
            }

            ExprKind::TypeCoercion(_, _) => {
                todo!("TypeCoercion not yet implemented in MIR")
            }

            ExprKind::Unsafe(_) => {
                todo!("Unsafe not yet implemented in MIR")
            }

            ExprKind::Assert(_, _) => {
                todo!("Assert not yet implemented in MIR")
            }
        } // NEWCASESHERE - add new cases before this closing brace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;
    use crate::parser::Parser;
    use crate::type_checker::TypeChecker;

    #[test]
    fn test_simple_function() {
        let source = "def add(#[location(0)] x: i32, #[location(1)] y: i32): i32 = x + y";
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().unwrap();
        let type_table = type_checker.check_program(&program).unwrap();

        eprintln!("Type table has {} entries", type_table.len());
        for (node_id, ty) in &type_table {
            eprintln!("  {:?} -> {:?}", node_id, ty);
        }

        eprintln!("Program AST:");
        eprintln!("{:#?}", program);

        let mirize = Mirize::new(type_table);
        let mir_module = mirize.mirize_program(&program).unwrap();

        assert_eq!(mir_module.functions.len(), 1);
        assert_eq!(mir_module.functions[0].name, "add");
        assert_eq!(mir_module.functions[0].params.len(), 2);
    }
}
