//! MIRize pass - Convert typed AST to MIR
//!
//! This pass walks the typed AST (after type checking) and converts it to
//! the MIR representation. It relies on type annotations being present on
//! all expressions.

use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::mir::{self, FunctionId, Instruction, Register};
use crate::scope::ScopeStack;
use std::collections::HashMap;

pub struct Mirize {
    builder: mir::Builder,

    /// Map from AST variable names to MIR registers in current scope
    env: ScopeStack<Register>,

    /// Type table from type checker - maps NodeId to inferred Type
    type_table: HashMap<NodeId, Type>,

    /// Entry points for SPIR-V generation
    entry_points: Vec<FunctionId>,
}

impl Mirize {
    pub fn new(type_table: HashMap<NodeId, Type>) -> Self {
        Mirize {
            builder: mir::Builder::new(),
            env: ScopeStack::new(),
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
            return Err(CompilerError::MirError(format!(
                "Function '{}' missing return type annotation",
                decl.name
            )));
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

        // Push scope for function body
        self.env.push_scope();

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

        // Pop function scope
        self.env.pop_scope();

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

        // Push scope for function body
        self.env.push_scope();

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

        // Pop function scope
        self.env.pop_scope();

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
            ExprKind::RecordLiteral(fields) => {
                // Evaluate all field values in order
                let mut field_regs = Vec::new();
                for (_field_name, field_expr) in fields {
                    let field_reg = self.mirize_expression(field_expr)?;
                    field_regs.push(field_reg);
                }

                // Records are represented as tuples in MIR (field order matters)
                Ok(self.builder.build_tuple(field_regs, expr_type))
            }

            ExprKind::IntLiteral(n) => Ok(self.builder.build_const_int(*n, expr_type)),

            ExprKind::FloatLiteral(f) => Ok(self.builder.build_const_float(*f, expr_type)),

            ExprKind::BoolLiteral(b) => Ok(self.builder.build_const_bool(*b, expr_type)),

            ExprKind::Identifier(name) => {
                // Check if this is a variable in scope
                if let Ok(reg) = self.env.lookup(name) {
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
                // Push scope for let bindings
                self.env.push_scope();

                // Process binding value
                let value_reg = self.mirize_expression(&let_in_expr.value)?;

                // Bind names from pattern
                // For tuple patterns, we need to extract elements; for simple names, bind directly
                match &let_in_expr.pattern.kind {
                    crate::ast::PatternKind::Name(name) => {
                        // Simple binding
                        self.env.insert(name.clone(), value_reg.clone());
                    }
                    crate::ast::PatternKind::Tuple(_patterns) => {
                        // Tuple pattern - extract each element
                        let names = let_in_expr.pattern.collect_names();

                        for (idx, name) in names.iter().enumerate() {
                            // Get element type from the value's tuple type
                            let elem_type = match &value_reg.ty {
                                Type::Constructed(TypeName::Str(n), elem_types) if *n == "tuple" => {
                                    elem_types.get(idx).cloned().unwrap_or_else(|| value_reg.ty.clone())
                                }
                                _ => value_reg.ty.clone(),
                            };

                            let elem_reg = self.builder.build_extract_element(
                                value_reg.clone(),
                                idx as u32,
                                elem_type,
                            );
                            self.env.insert(name.clone(), elem_reg);
                        }
                    }
                    _ => {
                        // Other patterns - bind all names to the whole value (TODO: implement properly)
                        for name in let_in_expr.pattern.collect_names() {
                            self.env.insert(name, value_reg.clone());
                        }
                    }
                }

                // Process body
                let result_reg = self.mirize_expression(&let_in_expr.body)?;

                // Pop scope
                self.env.pop_scope();

                Ok(result_reg)
            }

            ExprKind::Application(func, args) => Err(CompilerError::MirError(format!(
                "Application should be eliminated by defunctionalization: {:?} applied to {} args",
                func.kind,
                args.len()
            ))),

            ExprKind::FieldAccess(record, field_name) => {
                let record_reg = self.mirize_expression(record)?;

                // Look up field index from record type
                let field_idx = match &record_reg.ty {
                    Type::Constructed(TypeName::Vec, _) => {
                        // Vector field access (x, y, z, w)
                        match field_name.as_str() {
                            "x" => 0,
                            "y" => 1,
                            "z" => 2,
                            "w" => 3,
                            _ => {
                                return Err(CompilerError::MirError(format!(
                                    "Unknown vector field: {}",
                                    field_name
                                )));
                            }
                        }
                    }
                    Type::Constructed(TypeName::Record(fields), _) => {
                        // Record field access - find field index by name
                        fields
                            .keys()
                            .enumerate()
                            .find(|(_, name)| name.as_str() == field_name)
                            .map(|(idx, _)| idx as u32)
                            .ok_or_else(|| {
                                CompilerError::MirError(format!(
                                    "Unknown field '{}' in record with fields: {:?}",
                                    field_name,
                                    fields.keys().collect::<Vec<_>>()
                                ))
                            })?
                    }
                    Type::Constructed(TypeName::Str(name), _) if *name == "tuple" => {
                        // Tuple field access - parse numeric field name (0, 1, 2, ...)
                        field_name.parse::<u32>().map_err(|_| {
                            CompilerError::MirError(format!(
                                "Invalid tuple field access: '{}' (expected numeric index). Tuple type: {:?}",
                                field_name, record_reg.ty
                            ))
                        })?
                    }
                    _ => {
                        return Err(CompilerError::MirError(format!(
                            "Field access on non-record type: {:?}",
                            record_reg.ty
                        )));
                    }
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

            ExprKind::UnaryOp(op, operand) => {
                let operand_reg = self.mirize_expression(operand)?;

                match op.op.as_str() {
                    "-" => Ok(self.builder.build_neg(operand_reg)),
                    "!" => Ok(self.builder.build_not(operand_reg)),
                    _ => Err(CompilerError::MirError(format!(
                        "Unknown unary operator: {}",
                        op.op
                    ))),
                }
            }

            ExprKind::Loop(_) => {
                return Err(CompilerError::MirError(
                    "Loop should be desugared to InternalLoop before MIR generation".to_string(),
                ));
            }

            ExprKind::InternalLoop(internal_loop) => {
                // Generate MIR for InternalLoop following SPIR-V OpPhi structure

                // 1. Evaluate init expressions and create phi registers
                let (init_regs, phi_regs): (Vec<_>, Vec<_>) = internal_loop
                    .phi_vars
                    .iter()
                    .map(|phi_var| {
                        let init_reg = self.mirize_expression(&phi_var.init_expr)?;
                        let phi_type = phi_var.loop_var_type.clone().unwrap_or_else(|| init_reg.ty.clone());
                        let phi_reg = self.builder.new_register(phi_type);
                        Ok((init_reg, phi_reg))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .unzip();

                // 2. Create blocks
                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let continue_block = self.builder.create_block();
                let merge_block = self.builder.create_block();
                let pre_header_block = self.builder.current_block().unwrap();

                // 3. Push scope and bind loop_vars to phi registers
                self.env.push_scope();
                for (phi_var, phi_reg) in internal_loop.phi_vars.iter().zip(&phi_regs) {
                    self.env.insert(phi_var.loop_var_name.clone(), phi_reg.clone());
                }

                // 4. Generate body block
                self.builder.select_block(body_block);
                let body_result_reg = self.mirize_expression(&internal_loop.body)?;

                // 5. Bind body result to environment and evaluate next_exprs
                self.env.insert("__body_result".to_string(), body_result_reg.clone());

                let mut next_regs = Vec::new();
                for phi_var in &internal_loop.phi_vars {
                    let next_reg = self.mirize_expression(&phi_var.next_expr)?;
                    next_regs.push(next_reg);
                }

                self.builder.build_branch(continue_block);

                // 6. Generate continue block (back-edge)
                self.builder.select_block(continue_block);
                self.builder.build_branch(header_block);

                // 7. Generate header block with phi nodes and condition
                self.builder.select_block(header_block);

                // Insert phi instructions for each loop variable
                for (i, phi_reg) in phi_regs.iter().enumerate() {
                    let phi_inst = mir::Instruction::Phi(
                        phi_reg.clone(),
                        vec![
                            (init_regs[i].clone(), pre_header_block),
                            (next_regs[i].clone(), continue_block),
                        ],
                    );
                    self.builder.insert_instruction_at_start(phi_inst);
                }

                // Check condition and branch
                if let Some(cond_expr) = &internal_loop.condition {
                    let cond_reg = self.mirize_expression(cond_expr)?;
                    self.builder.emit_instruction(mir::Instruction::BranchLoop(
                        cond_reg,
                        body_block,
                        merge_block,
                        merge_block,
                        continue_block,
                    ));
                } else {
                    // No explicit condition - iterator style
                    // TODO: Generate implicit condition (index < length)
                    todo!("Iterator-style loop (condition=None) not yet implemented");
                }

                // 9. Generate merge block with result phi
                self.builder.select_block(merge_block);
                let result_reg = self.builder.new_register(expr_type.clone());

                // Result comes from the phi registers when loop exits
                // If single loop var, use it directly; otherwise body_result is already a tuple
                let exit_value = if phi_regs.len() == 1 {
                    phi_regs[0].clone()
                } else {
                    // For multiple vars, we need to rebuild the tuple from phi regs
                    // But actually the body produces the tuple, so we should use body_result_reg type
                    // Create a synthetic tuple from phi values
                    phi_regs[0].clone() // TODO: This needs to construct proper tuple
                };

                let result_phi =
                    mir::Instruction::Phi(result_reg.clone(), vec![(exit_value, header_block)]);
                self.builder.emit_instruction(result_phi);

                // 10. Emit Loop metadata instruction in pre-header
                self.builder.select_block(pre_header_block);
                // Build LoopInfo for metadata
                let loop_info = mir::LoopInfo {
                    phi_reg: phi_regs[0].clone(), // Primary phi (or first one)
                    result_reg: result_reg.clone(),
                    init_reg: init_regs[0].clone(),
                    body_result_reg,
                    cond_reg: result_reg.clone(), // Placeholder
                    pre_header_block,
                    header_block,
                    body_block,
                    merge_block,
                };
                self.builder.emit_instruction(mir::Instruction::Loop(loop_info));

                // 11. Continue in merge block and pop scope
                self.builder.select_block(merge_block);
                self.env.pop_scope();

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

    #[test]
    fn test_let_tuple_pattern() {
        let source = r#"
def test : i32 =
  let pair = (1, 2) in
  let (a, b) = pair in
  a + b
"#;
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().unwrap();
        let type_table = type_checker.check_program(&program).unwrap();

        let mirize = Mirize::new(type_table);
        let mir_module = mirize.mirize_program(&program).unwrap();

        assert_eq!(mir_module.functions.len(), 1);
        assert_eq!(mir_module.functions[0].name, "test");

        // Should have instructions to create tuple, extract elements, and add them
        let entry_block = &mir_module.functions[0].blocks[0];
        let has_make_tuple =
            entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::MakeTuple(..)));
        let has_extract =
            entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::ExtractElement(..)));

        assert!(has_make_tuple, "Should have MakeTuple instruction");
        assert!(has_extract, "Should have ExtractElement instruction");
    }

    #[test]
    fn test_let_tuple_pattern_with_field_access() {
        let source = r#"
def test : f32 =
  let pair = (1.0f32, vec3 2.0f32 3.0f32 4.0f32) in
  let (a, v) = pair in
  a + v.x
"#;
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().unwrap();
        let type_table = type_checker.check_program(&program).unwrap();

        let mirize = Mirize::new(type_table);
        let mir_module = mirize.mirize_program(&program).unwrap();

        assert_eq!(mir_module.functions.len(), 1);
        assert_eq!(mir_module.functions[0].name, "test");

        // Should have instructions to create tuple, extract elements, then access field
        let entry_block = &mir_module.functions[0].blocks[0];
        let has_make_tuple =
            entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::MakeTuple(..)));
        let extract_count = entry_block
            .instructions
            .iter()
            .filter(|inst| matches!(inst, Instruction::ExtractElement(..)))
            .count();

        assert!(has_make_tuple, "Should have MakeTuple instruction");
        // Should have at least 3 extracts: 2 from tuple pattern, 1 from vec3.x
        assert!(
            extract_count >= 3,
            "Should have at least 3 ExtractElement instructions, got {}",
            extract_count
        );
    }
}
