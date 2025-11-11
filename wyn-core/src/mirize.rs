//! MIRize pass - Convert typed AST to MIR
//!
//! This pass walks the typed AST (after type checking) and converts it to
//! the MIR representation. It relies on type annotations being present on
//! all expressions.

use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::mir::{self, FunctionId, Instruction, LoopInfo, Register};
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
        // Process declarations in order
        // Functions must be defined before they are used
        for decl in &program.declarations {
            self.mirize_declaration(decl)?;
        }

        Ok(self.builder.finish(self.entry_points))
    }

    /// Convert a top-level constant to a zero-argument function
    fn mirize_constant_as_function(&mut self, decl: &Decl) -> Result<()> {
        // Get the type of the constant
        let return_type = if let Some(ty) = &decl.ty {
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

        // Create a zero-argument function
        self.builder.begin_function(decl.name.clone(), vec![], return_type);

        // Evaluate the constant expression
        let value_reg = self.mirize_expression(&decl.body)?;

        // Return the value
        self.builder.build_return(value_reg);
        self.builder.end_function();

        Ok(())
    }

    fn mirize_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(d) => self.mirize_decl(d),
            Declaration::Entry(e) => self.mirize_entry_decl(e),
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
        // Extract parameter types (skip Unit patterns)
        let param_types: Vec<(String, Type)> = decl
            .params
            .iter()
            .filter(|p| !matches!(p.kind, PatternKind::Unit))
            .map(|p| {
                let name = p.simple_name().ok_or_else(|| {
                    CompilerError::MirError("Complex patterns not supported in MIR".to_string())
                })?;
                let ty = p.pattern_type().ok_or_else(|| {
                    CompilerError::MirError(format!(
                        "Function parameter '{}' missing type annotation",
                        name
                    ))
                })?;
                Ok((name.to_string(), ty.clone()))
            })
            .collect::<Result<Vec<_>>>()?;

        // Get return type from ty
        let return_type = if let Some(ty) = &decl.ty {
            ty.clone()
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
        let param_attrs: Vec<Vec<Attribute>> =
            decl.params
                .iter()
                .map(|p| {
                    if let PatternKind::Attributed(attrs, _) = &p.kind { attrs.clone() } else { Vec::new() }
                })
                .collect();

        // Set function attributes before building body
        // Regular decl has no attributed return types (those are only in EntryDecl)
        self.builder.set_function_attributes(param_attrs, vec![], None);

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

    fn mirize_entry_decl(&mut self, entry: &EntryDecl) -> Result<()> {
        // Extract parameter types (skip Unit patterns)
        let param_types: Vec<(String, Type)> = entry
            .params
            .iter()
            .filter(|p| !matches!(p.kind, PatternKind::Unit))
            .map(|p| {
                let name = p.simple_name().ok_or_else(|| {
                    CompilerError::MirError("Complex patterns not supported in MIR".to_string())
                })?;
                let ty = p.pattern_type().ok_or_else(|| {
                    CompilerError::MirError(format!(
                        "Entry point parameter '{}' missing type annotation",
                        name
                    ))
                })?;
                Ok((name.to_string(), ty.clone()))
            })
            .collect::<Result<Vec<_>>>()?;

        // Build return type from parallel arrays
        let return_type = if entry.return_types.len() == 1 {
            entry.return_types[0].clone()
        } else {
            use crate::ast::TypeName;
            Type::Constructed(TypeName::Str("tuple"), entry.return_types.clone())
        };

        // Start building the function
        let func_id = self.builder.begin_function(entry.name.clone(), param_types.clone(), return_type);

        // Entry points are always entry points
        self.entry_points.push(func_id);

        // Extract parameter attributes (skip Unit patterns)
        let param_attrs: Vec<Vec<Attribute>> = entry
            .params
            .iter()
            .filter(|p| !matches!(p.kind, PatternKind::Unit))
            .map(|p| {
                // Check if pattern is attributed, or if it's a Typed pattern with an attributed inner pattern
                match &p.kind {
                    PatternKind::Attributed(attrs, _) => attrs.clone(),
                    PatternKind::Typed(inner, _) => {
                        if let PatternKind::Attributed(attrs, _) = &inner.kind {
                            attrs.clone()
                        } else {
                            Vec::new()
                        }
                    }
                    _ => Vec::new(),
                }
            })
            .collect();

        // Convert return_attributes to AttributedType format for MIR
        let attributed_return_types: Vec<AttributedType> = entry
            .return_types
            .iter()
            .zip(entry.return_attributes.iter())
            .map(|(ty, attr_opt)| {
                let attributes = if let Some(attr) = attr_opt { vec![attr.clone()] } else { vec![] };
                AttributedType {
                    attributes,
                    ty: ty.clone(),
                }
            })
            .collect();

        // Set function attributes before building body
        // Convert Option<Attribute> to Vec<Attribute> by filtering out Nones
        let return_attrs: Vec<Attribute> =
            entry.return_attributes.iter().filter_map(|opt| opt.clone()).collect();

        self.builder.set_function_attributes(param_attrs, return_attrs, Some(attributed_return_types));

        // Bind parameters in environment
        for (idx, (name, _)) in param_types.iter().enumerate() {
            let param_reg = self
                .builder
                .get_param(idx)
                .ok_or_else(|| CompilerError::MirError(format!("Parameter {} not found", idx)))?;
            self.env.insert(name.clone(), param_reg);
        }

        // Convert body expression to MIR
        let result_reg = self.mirize_expression(&entry.body)?;

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

            ExprKind::BoolLiteral(b) => Ok(self.builder.build_const_bool(*b, expr_type)),

            ExprKind::Identifier(name) => {
                // Check if this is a variable in scope
                if let Some(reg) = self.env.get(name) {
                    Ok(reg.clone())
                } else {
                    // Must be a constant (zero-argument function) - call it
                    Ok(self.builder.build_call(name, vec![], expr_type))
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

                // Process binding - bind all names from pattern
                let value_reg = self.mirize_expression(&let_in_expr.value)?;
                for name in let_in_expr.pattern.collect_names() {
                    self.env.insert(name, value_reg.clone());
                }

                // Process body
                let result_reg = self.mirize_expression(&let_in_expr.body)?;

                // Restore environment
                self.env = saved_env;

                Ok(result_reg)
            }

            ExprKind::Application(func, args) => Err(CompilerError::MirError(format!(
                "Application should be eliminated by defunctionalization: {:?} applied to {} args",
                func.kind,
                args.len()
            ))),

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

            ExprKind::Loop(loop_expr) => {
                // Only support while loops for now
                let cond_expr = match &loop_expr.form {
                    LoopForm::While(cond) => cond,
                    LoopForm::For(_, _) => {
                        return Err(CompilerError::MirError(
                            "Loop for form not yet supported in MIR".to_string(),
                        ));
                    }
                    LoopForm::ForIn(_, _) => {
                        return Err(CompilerError::MirError(
                            "Loop for-in form not yet supported in MIR".to_string(),
                        ));
                    }
                };

                // Only support simple name patterns
                let loop_var_name = match &loop_expr.pattern.kind {
                    PatternKind::Name(name) => name.clone(),
                    _ => {
                        return Err(CompilerError::MirError(
                            "Only simple name patterns in loops supported for now".to_string(),
                        ));
                    }
                };

                // Evaluate initial value
                let init_reg = if let Some(init) = &loop_expr.init {
                    self.mirize_expression(init)?
                } else {
                    return Err(CompilerError::MirError(
                        "Loop without explicit init not yet supported in MIR".to_string(),
                    ));
                };

                // Create blocks
                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let merge_block = self.builder.create_block();
                let pre_header_block = self.builder.current_block().unwrap();

                // Create registers
                let phi_reg = self.builder.new_register(expr_type.clone());
                let result_reg = self.builder.new_register(expr_type.clone());

                // Save environment and bind loop variable
                let saved_env = self.env.clone();
                self.env.insert(loop_var_name, phi_reg.clone());

                // Generate body expression first to get body_result_reg
                self.builder.select_block(body_block);
                let body_result_reg = self.mirize_expression(&loop_expr.body)?;
                self.builder.build_branch(header_block);

                // Now generate header block with proper Phi
                self.builder.select_block(header_block);
                let phi_inst = Instruction::Phi(
                    phi_reg.clone(),
                    vec![
                        (init_reg.clone(), pre_header_block),
                        (body_result_reg.clone(), body_block),
                    ],
                );
                self.builder.insert_instruction_at_start(phi_inst);
                let cond_reg = self.mirize_expression(cond_expr)?;
                self.builder.emit_instruction(Instruction::BranchLoop(
                    cond_reg.clone(),
                    body_block,
                    merge_block,
                    merge_block,
                    body_block,
                ));

                // Generate merge block: result Phi
                self.builder.select_block(merge_block);
                let result_phi =
                    Instruction::Phi(result_reg.clone(), vec![(phi_reg.clone(), header_block)]);
                self.builder.emit_instruction(result_phi);

                // Build LoopInfo with all the metadata
                let loop_info = LoopInfo {
                    phi_reg: phi_reg.clone(),
                    result_reg: result_reg.clone(),
                    init_reg,
                    body_result_reg,
                    cond_reg,
                    pre_header_block,
                    header_block,
                    body_block,
                    merge_block,
                };

                // Emit the Loop instruction in pre-header for metadata
                self.builder.select_block(pre_header_block);
                self.builder.emit_instruction(Instruction::Loop(loop_info));

                // Continue in merge block (where result is available)
                self.builder.select_block(merge_block);

                // Restore environment
                self.env = saved_env;

                Ok(result_reg)
            }

            ExprKind::Match(_) => {
                todo!("Match not yet implemented in MIR")
            }

            ExprKind::Range(_) => {
                todo!("Range not yet implemented in MIR")
            }

            ExprKind::Pipe(_, _) => Err(CompilerError::MirError(
                "Pipe operator should have been desugared by defunctionalization".to_string(),
            )),

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
        let source = "def add(#[location(0)] x: i32) (#[location(1)] y: i32): i32 = x + y";
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
