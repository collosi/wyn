use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use std::collections::{HashMap, HashSet};

/// Static values for defunctionalization, as described in the Futhark paper
#[derive(Debug, Clone, PartialEq)]
pub enum StaticValue {
    Dyn(Type),                            // Dynamic value with type
    Lam(String, Expression, Environment), // Lambda: param name, body, environment
    Rcd(HashMap<String, StaticValue>),    // Record of static values
    Arr(Box<StaticValue>),                // Array of static values
}

/// Translation environment mapping variables to static values
/// Note: This is being replaced by ScopeStack<StaticValue> throughout the codebase
pub type Environment = HashMap<String, StaticValue>;

/// Generated function for defunctionalized lambda
#[derive(Debug, Clone)]
pub struct DefunctionalizedFunction {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Type,
    pub body: Expression,
}

pub struct Defunctionalizer {
    next_function_id: usize,
    generated_functions: Vec<DefunctionalizedFunction>,
    node_counter: NodeCounter,
}

impl Default for Defunctionalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Defunctionalizer {
    pub fn new() -> Self {
        Defunctionalizer {
            next_function_id: 0,
            generated_functions: Vec::new(),
            node_counter: NodeCounter::new(),
        }
    }

    pub fn new_with_counter(node_counter: NodeCounter) -> Self {
        Defunctionalizer {
            next_function_id: 0,
            generated_functions: Vec::new(),
            node_counter,
        }
    }

    pub fn defunctionalize_program(&mut self, program: &Program) -> Result<Program> {
        let mut new_declarations = Vec::new();
        let mut scope_stack = ScopeStack::new();

        // First pass: collect all declarations and transform them
        for decl in &program.declarations {
            match decl {
                Declaration::Decl(decl_node) => {
                    if decl_node.keyword == "let" && decl_node.params.is_empty() {
                        // Let variable declaration - needs defunctionalization
                        let (transformed_decl, _sv) =
                            self.defunctionalize_decl(decl_node, &mut scope_stack)?;
                        new_declarations.push(transformed_decl);
                    } else {
                        // Def declarations or function declarations - already first-order
                        new_declarations.push(Declaration::Decl(decl_node.clone()));
                    }
                }
                Declaration::Entry(entry) => {
                    // Entry points need defunctionalization too
                    let (transformed_body, _sv) =
                        self.defunctionalize_expression(&entry.body, &mut scope_stack)?;
                    let mut transformed_entry = entry.clone();
                    transformed_entry.body = transformed_body;
                    new_declarations.push(Declaration::Entry(transformed_entry));
                }
                Declaration::Uniform(uniform_decl) => {
                    // Uniform declarations have no body
                    new_declarations.push(Declaration::Uniform(uniform_decl.clone()));
                }
                Declaration::Val(val_decl) => {
                    // Type signatures only
                    new_declarations.push(Declaration::Val(val_decl.clone()));
                }
                Declaration::TypeBind(_) => {
                    unimplemented!("Type bindings are not yet supported in defunctionalization")
                }
                Declaration::ModuleBind(_) => {
                    unimplemented!("Module bindings are not yet supported in defunctionalization")
                }
                Declaration::ModuleTypeBind(_) => {
                    unimplemented!("Module type bindings are not yet supported in defunctionalization")
                }
                Declaration::Open(_) => {
                    unimplemented!("Open declarations are not yet supported in defunctionalization")
                }
                Declaration::Import(_) => {
                    unimplemented!("Import declarations are not yet supported in defunctionalization")
                }
                Declaration::Local(_) => {
                    unimplemented!("Local declarations are not yet supported in defunctionalization")
                }
            }
        }

        // Add generated functions as def declarations
        for func in &self.generated_functions {
            new_declarations.push(Declaration::Decl(Decl {
                keyword: "def",
                attributes: vec![],
                name: func.name.clone(),
                params: func
                    .params
                    .iter()
                    .map(|p| self.node_counter.mk_node(PatternKind::Name(p.name.clone()), Span::dummy()))
                    .collect(),
                ty: None, // Function definitions don't have explicit type annotations
                body: func.body.clone(),
            }));
        }

        Ok(Program {
            declarations: new_declarations,
        })
    }

    fn defunctionalize_decl(
        &mut self,
        decl: &Decl,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Declaration, StaticValue)> {
        let (transformed_expr, sv) = self.defunctionalize_expression(&decl.body, scope_stack)?;

        // Add the binding to the current scope
        scope_stack.insert(decl.name.clone(), sv.clone());

        let transformed_decl = Declaration::Decl(Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            params: decl.params.clone(),
            ty: decl.ty.clone(),
            body: transformed_expr,
        });

        Ok((transformed_decl, sv))
    }

    fn defunctionalize_expression(
        &mut self,
        expr: &Expression,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        let span = expr.h.span;
        match &expr.kind {
            ExprKind::IntLiteral(n) => Ok((
                self.node_counter.mk_node(ExprKind::IntLiteral(*n), span),
                StaticValue::Dyn(types::i32()),
            )),
            ExprKind::FloatLiteral(f) => Ok((
                self.node_counter.mk_node(ExprKind::FloatLiteral(*f), span),
                StaticValue::Dyn(types::f32()),
            )),
            ExprKind::BoolLiteral(b) => Ok((
                self.node_counter.mk_node(ExprKind::BoolLiteral(*b), span),
                StaticValue::Dyn(types::bool_type()),
            )),
            ExprKind::Identifier(name) => {
                if let Ok(sv) = scope_stack.lookup(name) {
                    match sv {
                        StaticValue::Dyn(_) => {
                            // Regular variable reference
                            Ok((
                                self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                                sv.clone(),
                            ))
                        }
                        StaticValue::Lam(_, _, _) => {
                            // Reference to a function - this would need special handling
                            // For now, keep as identifier
                            Ok((
                                self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                                sv.clone(),
                            ))
                        }
                        StaticValue::Rcd(_) => {
                            // Reference to a closure record
                            Ok((
                                self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                                sv.clone(),
                            ))
                        }
                        StaticValue::Arr(_) => Ok((
                            self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                            sv.clone(),
                        )),
                    }
                } else {
                    // Unknown variable - assume dynamic with type variable
                    Ok((
                        self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                        StaticValue::Dyn(polytype::Type::Variable(0)),
                    ))
                }
            }
            ExprKind::Lambda(lambda) => self.defunctionalize_lambda(lambda, scope_stack),
            ExprKind::Application(func, args) => self.defunctionalize_application(func, args, scope_stack),
            ExprKind::ArrayLiteral(elements) => {
                let mut transformed_elements = Vec::new();
                let mut element_sv = None;

                for elem in elements {
                    let (transformed_elem, sv) = self.defunctionalize_expression(elem, scope_stack)?;
                    transformed_elements.push(transformed_elem);

                    // All elements should have the same static value structure
                    if element_sv.is_none() {
                        element_sv = Some(sv);
                    }
                }

                let array_sv =
                    StaticValue::Arr(Box::new(element_sv.unwrap_or(StaticValue::Dyn(types::i32()))));
                Ok((
                    self.node_counter.mk_node(ExprKind::ArrayLiteral(transformed_elements), span),
                    array_sv,
                ))
            }
            ExprKind::ArrayIndex(array, index) => {
                let (transformed_array, _array_sv) = self.defunctionalize_expression(array, scope_stack)?;
                let (transformed_index, _index_sv) = self.defunctionalize_expression(index, scope_stack)?;

                // Result type depends on array element type - for now, assume dynamic
                Ok((
                    self.node_counter.mk_node(
                        ExprKind::ArrayIndex(Box::new(transformed_array), Box::new(transformed_index)),
                        span,
                    ),
                    StaticValue::Dyn(polytype::Type::Variable(1)),
                ))
            }
            ExprKind::BinaryOp(op, left, right) => {
                let (transformed_left, left_sv) = self.defunctionalize_expression(left, scope_stack)?;
                let (transformed_right, right_sv) = self.defunctionalize_expression(right, scope_stack)?;

                // For binary arithmetic operations, the result type should be the same as the operand types
                // (assuming type checking has already ensured they match)
                let result_type = match (&left_sv, &right_sv) {
                    (StaticValue::Dyn(left_type), StaticValue::Dyn(_right_type)) => {
                        // Use the left operand type (they should be the same after type checking)
                        left_type.clone()
                    }
                    (StaticValue::Dyn(ty), _) | (_, StaticValue::Dyn(ty)) => {
                        // If one is dynamic, use that type
                        ty.clone()
                    }
                    _ => {
                        // Fallback to a generic type variable if we can't determine the type
                        polytype::Type::Variable(4)
                    }
                };

                Ok((
                    self.node_counter.mk_node(
                        ExprKind::BinaryOp(
                            op.clone(),
                            Box::new(transformed_left),
                            Box::new(transformed_right),
                        ),
                        span,
                    ),
                    StaticValue::Dyn(result_type),
                ))
            }
            ExprKind::FunctionCall(name, args) => {
                // Regular function calls (first-order) remain unchanged
                let mut transformed_args = Vec::new();
                for arg in args {
                    let (transformed_arg, _sv) = self.defunctionalize_expression(arg, scope_stack)?;
                    transformed_args.push(transformed_arg);
                }

                Ok((
                    self.node_counter.mk_node(ExprKind::FunctionCall(name.clone(), transformed_args), span),
                    StaticValue::Dyn(polytype::Type::Variable(2)),
                ))
            }
            ExprKind::Tuple(elements) => {
                let mut transformed_elements = Vec::new();
                let mut element_types = Vec::new();

                for elem in elements {
                    let (transformed_elem, sv) = self.defunctionalize_expression(elem, scope_stack)?;
                    transformed_elements.push(transformed_elem);

                    // Extract type from static value
                    let elem_type = match sv {
                        StaticValue::Dyn(ty) => ty,
                        _ => polytype::Type::Variable(3),
                    };
                    element_types.push(elem_type);
                }

                Ok((
                    self.node_counter.mk_node(ExprKind::Tuple(transformed_elements), span),
                    StaticValue::Dyn(types::tuple(element_types)),
                ))
            }
            ExprKind::LetIn(let_in) => {
                // Transform the value expression
                let (transformed_value, value_sv) =
                    self.defunctionalize_expression(&let_in.value, scope_stack)?;

                // Push new scope and add binding
                scope_stack.push_scope();
                scope_stack.insert(let_in.name.clone(), value_sv);

                // Transform the body expression
                let (transformed_body, body_sv) =
                    self.defunctionalize_expression(&let_in.body, scope_stack)?;

                // Pop scope
                scope_stack.pop_scope();

                Ok((
                    self.node_counter.mk_node(
                        ExprKind::LetIn(crate::ast::LetInExpr {
                            name: let_in.name.clone(),
                            ty: let_in.ty.clone(),
                            value: Box::new(transformed_value),
                            body: Box::new(transformed_body),
                        }),
                        span,
                    ),
                    body_sv,
                ))
            }
            ExprKind::FieldAccess(expr, field) => {
                let (transformed_expr, expr_sv) = self.defunctionalize_expression(expr, scope_stack)?;
                Ok((
                    self.node_counter.mk_node(
                        ExprKind::FieldAccess(Box::new(transformed_expr), field.clone()),
                        span,
                    ),
                    expr_sv, // Field access doesn't change the static value representation
                ))
            }
            ExprKind::If(if_expr) => {
                let (condition, _condition_sv) =
                    self.defunctionalize_expression(&if_expr.condition, scope_stack)?;
                let (then_branch, _then_sv) =
                    self.defunctionalize_expression(&if_expr.then_branch, scope_stack)?;
                let (else_branch, _else_sv) =
                    self.defunctionalize_expression(&if_expr.else_branch, scope_stack)?;
                Ok((
                    self.node_counter.mk_node(
                        ExprKind::If(IfExpr {
                            condition: Box::new(condition),
                            then_branch: Box::new(then_branch),
                            else_branch: Box::new(else_branch),
                        }),
                        span,
                    ),
                    StaticValue::Dyn(Type::Constructed(TypeName::Str("unknown"), vec![])), // If expressions are runtime values
                ))
            }

            ExprKind::TypeHole => Ok((
                self.node_counter.mk_node(ExprKind::TypeHole, span),
                StaticValue::Dyn(polytype::Type::Variable(0)), // Type to be inferred
            )),

            ExprKind::Pipe(left, right) => {
                // a |> f desugars to f(a)
                // Defunctionalize both sides
                let (left_expr, _left_sv) = self.defunctionalize_expression(left, scope_stack)?;
                let (right_expr, _right_sv) = self.defunctionalize_expression(right, scope_stack)?;

                // Desugar into application
                self.defunctionalize_application(&right_expr, &[left_expr], scope_stack)
            }

            ExprKind::QualifiedName(_, _)
            | ExprKind::UnaryOp(_, _)
            | ExprKind::Loop(_)
            | ExprKind::Match(_)
            | ExprKind::Range(_)
            | ExprKind::TypeAscription(_, _)
            | ExprKind::TypeCoercion(_, _)
            | ExprKind::Unsafe(_)
            | ExprKind::Assert(_, _) => {
                todo!("New expression kinds not yet implemented in defunctionalization")
            }
        } // NEWCASESHERE - add new cases before this closing brace
    }

    fn defunctionalize_lambda(
        &mut self,
        lambda: &LambdaExpr,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        // Find free variables in the lambda body
        let free_vars = self.find_free_variables(
            &lambda.body,
            &lambda.params.iter().filter_map(|p| p.simple_name().map(|s| s.to_string())).collect(),
        )?;

        // Create a closure record with free variables
        let mut closure_fields = HashMap::new();
        for var in &free_vars {
            if let Ok(sv) = scope_stack.lookup(var) {
                closure_fields.insert(var.clone(), sv.clone());
            }
        }

        // Generate a unique function name
        let func_name = format!("__lambda_{}", self.next_function_id);
        self.next_function_id += 1;

        // Create parameters: closure record + lambda parameters
        let mut func_params = vec![Parameter {
            attributes: vec![],
            name: "__closure".to_string(),
            ty: polytype::Type::Variable(4), // Will be refined later
        }];

        for param in &lambda.params {
            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    CompilerError::ParseError(
                        "Complex patterns in lambda parameters not yet supported".to_string(),
                    )
                })?
                .to_string();
            let param_ty = param.pattern_type().cloned().unwrap_or(polytype::Type::Variable(5));
            func_params.push(Parameter {
                attributes: vec![],
                name: param_name,
                ty: param_ty,
            });
        }

        // Transform lambda body with parameter scope
        scope_stack.push_scope();
        for param in &lambda.params {
            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    CompilerError::ParseError(
                        "Complex patterns in lambda parameters not yet supported".to_string(),
                    )
                })?
                .to_string();
            let param_ty = param.pattern_type().cloned().unwrap_or(polytype::Type::Variable(6));
            scope_stack.insert(param_name, StaticValue::Dyn(param_ty));
        }

        let (transformed_body, _body_sv) = self.defunctionalize_expression(&lambda.body, scope_stack)?;

        // Pop parameter scope
        scope_stack.pop_scope();

        // Create the generated function
        let return_type = lambda.return_type.clone().unwrap_or(polytype::Type::Variable(7));
        let generated_func = DefunctionalizedFunction {
            name: func_name.clone(),
            params: func_params,
            return_type,
            body: transformed_body,
        };

        self.generated_functions.push(generated_func);

        // Create closure constructor expression
        if free_vars.is_empty() {
            // No free variables - just return function name
            Ok((
                self.node_counter.mk_node(ExprKind::Identifier(func_name), Span::dummy()),
                StaticValue::Lam("__unused".to_string(), (*lambda.body).clone(), HashMap::new()),
            ))
        } else {
            // Create closure record
            let closure_record = self.create_closure_record(&func_name, &free_vars)?;
            Ok((closure_record, StaticValue::Rcd(closure_fields)))
        }
    }

    fn defunctionalize_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        let (transformed_func, func_sv) = self.defunctionalize_expression(func, scope_stack)?;

        let mut transformed_args = Vec::new();
        for arg in args {
            let (transformed_arg, _arg_sv) = self.defunctionalize_expression(arg, scope_stack)?;
            transformed_args.push(transformed_arg);
        }

        match func_sv {
            StaticValue::Lam(_param, _body, _closure_env) => {
                // Direct lambda application - inline if simple enough
                // For now, convert to function call
                match &transformed_func.kind {
                    ExprKind::Identifier(func_name) => {
                        // Function call without closure
                        Ok((
                            self.node_counter.mk_node(
                                ExprKind::FunctionCall(func_name.clone(), transformed_args),
                                Span::dummy(),
                            ),
                            StaticValue::Dyn(polytype::Type::Variable(2)),
                        ))
                    }
                    _ => {
                        // More complex case - would need closure unpacking
                        // For now, return error
                        Err(CompilerError::SpirvError(
                            "Complex function application not yet supported in defunctionalization"
                                .to_string(),
                        ))
                    }
                }
            }
            StaticValue::Rcd(_) => {
                // Closure application - would need to unpack closure and call function
                Err(CompilerError::SpirvError(
                    "Closure application not yet implemented in defunctionalization".to_string(),
                ))
            }
            _ => {
                // Regular function call
                match &transformed_func.kind {
                    ExprKind::Identifier(func_name) => {
                        // Special handling for higher-order builtins like map
                        if func_name == "map" && transformed_args.len() == 2 {
                            // map f xs -> loop-based implementation
                            return self.defunctionalize_map(&transformed_args[0], &transformed_args[1], scope_stack);
                        }

                        Ok((
                            self.node_counter.mk_node(
                                ExprKind::FunctionCall(func_name.clone(), transformed_args),
                                Span::dummy(),
                            ),
                            StaticValue::Dyn(polytype::Type::Variable(2)),
                        ))
                    }
                    ExprKind::FieldAccess(base, field) => {
                        // Qualified name like f32.cos - convert to dotted name for builtin lookup
                        let qual_name =
                            crate::ast::QualName::new(vec![Self::extract_base_name(base)?], field.clone());
                        let dotted_name = qual_name.to_dotted();
                        Ok((
                            self.node_counter.mk_node(
                                ExprKind::FunctionCall(dotted_name, transformed_args),
                                Span::dummy(),
                            ),
                            StaticValue::Dyn(polytype::Type::Variable(2)),
                        ))
                    }
                    ExprKind::FunctionCall(func_name, existing_args) => {
                        // This is a partial application that's already been transformed to FunctionCall
                        // Append the new args to the existing ones
                        let mut all_args = existing_args.clone();
                        all_args.extend(transformed_args);
                        Ok((
                            self.node_counter.mk_node(
                                ExprKind::FunctionCall(func_name.clone(), all_args),
                                Span::dummy(),
                            ),
                            StaticValue::Dyn(polytype::Type::Variable(2)),
                        ))
                    }
                    ExprKind::Application(nested_func, nested_args) => {
                        // Recursive application - this shouldn't happen after defunctionalization
                        // but handle it by recursively processing
                        self.defunctionalize_application(
                            &Expression {
                                h: transformed_func.h.clone(),
                                kind: ExprKind::Application(nested_func.clone(), nested_args.clone()),
                            },
                            args,
                            scope_stack,
                        )
                    }
                    _ => Err(CompilerError::SpirvError(format!(
                        "Invalid function in application: {:?}",
                        transformed_func.kind
                    ))),
                }
            }
        }
    }

    /// Extract the base identifier name from an expression
    /// E.g., for `f32` in `f32.cos`, returns "f32"
    fn extract_base_name(expr: &Expression) -> Result<String> {
        match &expr.kind {
            ExprKind::Identifier(name) => Ok(name.clone()),
            _ => Err(CompilerError::SpirvError(format!(
                "Expected identifier as base of qualified name, got {:?}",
                expr.kind
            ))),
        }
    }

    fn find_free_variables(
        &self,
        expr: &Expression,
        bound_vars: &HashSet<String>,
    ) -> Result<HashSet<String>> {
        let mut free_vars = HashSet::new();
        self.collect_free_variables(expr, bound_vars, &mut free_vars)?;
        Ok(free_vars)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_free_variables(
        &self,
        expr: &Expression,
        bound_vars: &HashSet<String>,
        free_vars: &mut HashSet<String>,
    ) -> Result<()> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                if !bound_vars.contains(name) {
                    free_vars.insert(name.clone());
                }
            }
            ExprKind::Lambda(lambda) => {
                let mut extended_bound = bound_vars.clone();
                for param in &lambda.params {
                    if let Some(name) = param.simple_name() {
                        extended_bound.insert(name.to_string());
                    }
                }
                self.collect_free_variables(&lambda.body, &extended_bound, free_vars)?;
            }
            ExprKind::Application(func, args) => {
                self.collect_free_variables(func, bound_vars, free_vars)?;
                for arg in args {
                    self.collect_free_variables(arg, bound_vars, free_vars)?;
                }
            }
            ExprKind::ArrayLiteral(elements) => {
                for elem in elements {
                    self.collect_free_variables(elem, bound_vars, free_vars)?;
                }
            }
            ExprKind::ArrayIndex(array, index) => {
                self.collect_free_variables(array, bound_vars, free_vars)?;
                self.collect_free_variables(index, bound_vars, free_vars)?;
            }
            ExprKind::BinaryOp(_, left, right) => {
                self.collect_free_variables(left, bound_vars, free_vars)?;
                self.collect_free_variables(right, bound_vars, free_vars)?;
            }
            ExprKind::FunctionCall(_, args) => {
                for arg in args {
                    self.collect_free_variables(arg, bound_vars, free_vars)?;
                }
            }
            ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.collect_free_variables(elem, bound_vars, free_vars)?;
                }
            }
            ExprKind::LetIn(let_in) => {
                // Collect free variables from value expression
                self.collect_free_variables(&let_in.value, bound_vars, free_vars)?;

                // Add let binding to bound variables and collect from body
                let mut extended_bound = bound_vars.clone();
                extended_bound.insert(let_in.name.clone());
                self.collect_free_variables(&let_in.body, &extended_bound, free_vars)?;
            }
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::TypeHole => {
                // No free variables in literals or type holes
            }
            ExprKind::FieldAccess(expr, _field) => {
                self.collect_free_variables(expr, bound_vars, free_vars)?;
            }
            ExprKind::If(if_expr) => {
                self.collect_free_variables(&if_expr.condition, bound_vars, free_vars)?;
                self.collect_free_variables(&if_expr.then_branch, bound_vars, free_vars)?;
                self.collect_free_variables(&if_expr.else_branch, bound_vars, free_vars)?;
            }
            ExprKind::Pipe(left, right) => {
                self.collect_free_variables(left, bound_vars, free_vars)?;
                self.collect_free_variables(right, bound_vars, free_vars)?;
            }

            ExprKind::QualifiedName(_, _)
            | ExprKind::UnaryOp(_, _)
            | ExprKind::Loop(_)
            | ExprKind::Match(_)
            | ExprKind::Range(_)
            | ExprKind::TypeAscription(_, _)
            | ExprKind::TypeCoercion(_, _)
            | ExprKind::Unsafe(_)
            | ExprKind::Assert(_, _) => {
                todo!("New expression kinds not yet implemented in collect_free_variables")
            }
        } // NEWCASESHERE - add new cases before this closing brace
        Ok(())
    }

    fn create_closure_record(
        &mut self,
        func_name: &str,
        free_vars: &HashSet<String>,
    ) -> Result<Expression> {
        // For now, create a simple record-like structure
        // In a full implementation, this would create a proper record expression
        // For SPIR-V compatibility, we might need to represent this as an array or struct

        // Create a tuple with function name and free variables
        let mut elements =
            vec![self.node_counter.mk_node(ExprKind::Identifier(func_name.to_string()), Span::dummy())];
        for var in free_vars {
            elements.push(self.node_counter.mk_node(ExprKind::Identifier(var.clone()), Span::dummy()));
        }

        Ok(self.node_counter.mk_node(ExprKind::Tuple(elements), Span::dummy()))
    }

    /// Transform map f xs into a loop that allocates output array and fills it
    /// map f xs  =>
    ///   let xs = <array> in
    ///   let len = length xs in
    ///   let out = replicate len 0 in  // Allocate array with default values
    ///   loop (i, out) = (0, out) while i < len do
    ///     let updated_out = __array_update(out, i, f xs[i]) in
    ///     (i + 1, updated_out)
    ///
    /// TODO: Eventually handle in-place updates to avoid copying output array each iteration
    fn defunctionalize_map(
        &mut self,
        func: &Expression,
        array: &Expression,
        _scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        let span = Span::dummy();

        // Extract function name (assuming it's a simple identifier for now)
        let func_name = match &func.kind {
            ExprKind::Identifier(name) => name.clone(),
            _ => return Err(CompilerError::SpirvError(
                "map currently only supports simple function identifiers".to_string()
            )),
        };

        // Generate unique variable names
        let i_var = format!("__map_i_{}", self.next_function_id);
        let out_var = format!("__map_out_{}", self.next_function_id);
        let xs_var = format!("__map_xs_{}", self.next_function_id);
        let len_var = format!("__map_len_{}", self.next_function_id);
        self.next_function_id += 1;

        // Build ALL leaf nodes first to avoid borrow checker issues
        let xs_ident_for_len = self.node_counter.mk_node(ExprKind::Identifier(xs_var.clone()), span);
        let len_ident_for_replicate = self.node_counter.mk_node(ExprKind::Identifier(len_var.clone()), span);
        let zero_for_replicate = self.node_counter.mk_node(ExprKind::IntLiteral(0), span);

        // length xs
        let len_call = self.node_counter.mk_node(
            ExprKind::FunctionCall(
                "length".to_string(),
                vec![xs_ident_for_len],
            ),
            span,
        );

        // Initialize output array using replicate: replicate len default_value
        // The default value doesn't matter since we'll write to every element
        let init_out = self.node_counter.mk_node(
            ExprKind::FunctionCall(
                "replicate".to_string(),
                vec![len_ident_for_replicate, zero_for_replicate],
            ),
            span,
        );

        // Build more leaf nodes for loop body
        let i_ident = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let len_ident = self.node_counter.mk_node(ExprKind::Identifier(len_var.clone()), span);
        let xs_ident = self.node_counter.mk_node(ExprKind::Identifier(xs_var.clone()), span);
        let i_ident2 = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let out_ident = self.node_counter.mk_node(ExprKind::Identifier(out_var.clone()), span);
        let i_ident3 = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let i_ident4 = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let one_lit = self.node_counter.mk_node(ExprKind::IntLiteral(1), span);
        let zero_lit = self.node_counter.mk_node(ExprKind::IntLiteral(0), span);
        let out_ident2 = self.node_counter.mk_node(ExprKind::Identifier(out_var.clone()), span);

        // i < len
        let condition = self.node_counter.mk_node(
            ExprKind::BinaryOp(
                BinaryOp { op: "<".to_string() },
                Box::new(i_ident),
                Box::new(len_ident),
            ),
            span,
        );

        // xs[i]
        let array_index = self.node_counter.mk_node(
            ExprKind::ArrayIndex(
                Box::new(xs_ident),
                Box::new(i_ident2),
            ),
            span,
        );

        // f xs[i]
        let func_app = self.node_counter.mk_node(
            ExprKind::FunctionCall(func_name, vec![array_index]),
            span,
        );

        // __array_update(out, i, f xs[i])
        let updated_out = self.node_counter.mk_node(
            ExprKind::FunctionCall(
                "__array_update".to_string(),
                vec![out_ident, i_ident3, func_app],
            ),
            span,
        );

        // i + 1
        let i_inc = self.node_counter.mk_node(
            ExprKind::BinaryOp(
                BinaryOp { op: "+".to_string() },
                Box::new(i_ident4),
                Box::new(one_lit),
            ),
            span,
        );

        // (i + 1, updated_out)
        let loop_body = self.node_counter.mk_node(
            ExprKind::Tuple(vec![i_inc, updated_out]),
            span,
        );

        // (0, out)
        let initial_value = self.node_counter.mk_node(
            ExprKind::Tuple(vec![zero_lit, out_ident2]),
            span,
        );

        // loop (i, out) = (0, out) while i < len do (i + 1, updated_out)
        let i_pattern = self.node_counter.mk_node(PatternKind::Name(i_var), span);
        let out_pattern = self.node_counter.mk_node(PatternKind::Name(out_var.clone()), span);
        let loop_pattern = self.node_counter.mk_node(
            PatternKind::Tuple(vec![i_pattern, out_pattern]),
            span,
        );
        let loop_expr = self.node_counter.mk_node(
            ExprKind::Loop(LoopExpr {
                pattern: loop_pattern,
                init: Some(Box::new(initial_value)),
                form: LoopForm::While(Box::new(condition)),
                body: Box::new(loop_body),
            }),
            span,
        );

        // let out = replicate len 0 in <loop>
        let with_out = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                name: out_var,
                ty: None,
                value: Box::new(init_out),
                body: Box::new(loop_expr),
            }),
            span,
        );

        // let len = length xs in <with_out>
        let with_len = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                name: len_var,
                ty: None,
                value: Box::new(len_call),
                body: Box::new(with_out),
            }),
            span,
        );

        // let xs = <array> in <with_len>
        let result = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                name: xs_var,
                ty: None,
                value: Box::new(array.clone()),
                body: Box::new(with_len),
            }),
            span,
        );

        Ok((result, StaticValue::Dyn(polytype::Type::Variable(0))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_defunctionalize_simple_lambda() {
        let input = r#"let f: i32 -> i32 = \x -> x"#;
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut defunc = Defunctionalizer::new();
        let result = defunc.defunctionalize_program(&program).unwrap();

        // Should have generated a new function
        assert!(defunc.generated_functions.len() > 0);

        // The let declaration should be transformed
        assert_eq!(result.declarations.len(), 2); // original let + generated function
    }

    #[test]
    fn test_defunctionalize_nested_application() {
        // Test that ((f x) y) z becomes f(x, y, z)
        let input = "def test = vec3 1.0f32 0.5f32 0.25f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut defunc = Defunctionalizer::new();
        let result = defunc.defunctionalize_program(&program).unwrap();

        // Check that the result doesn't contain any Application nodes
        let decl = &result.declarations[0];
        if let Declaration::Decl(d) = decl {
            // The body should be a FunctionCall, not an Application
            match &d.body.kind {
                ExprKind::FunctionCall(name, args) => {
                    assert_eq!(name, "vec3");
                    assert_eq!(args.len(), 3);
                }
                ExprKind::Application(_, _) => {
                    panic!(
                        "Found Application node after defunctionalization - nested applications not flattened"
                    );
                }
                other => panic!("Expected FunctionCall, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_defunctionalize_application_with_division() {
        // Test that constant-folded divisions inside function calls work
        let input = "def test = vec3 (255.0f32/255.0f32) 0.5f32 0.25f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        // Debug: print the AST before constant folding
        if let Declaration::Decl(d) = &program.declarations[0] {
            eprintln!("Before constant folding: {:?}", d.body.kind);
        }

        // Run constant folding first
        let mut folder = crate::constant_folding::ConstantFolder::new();
        let program = folder.fold_program(&program).unwrap();

        // Debug: print the AST after constant folding
        if let Declaration::Decl(d) = &program.declarations[0] {
            eprintln!("After constant folding: {:?}", d.body.kind);
        }

        // Then defunctionalize
        let mut defunc = Defunctionalizer::new();
        let result = defunc.defunctionalize_program(&program).unwrap();

        // Check that the result doesn't contain any Application nodes
        let decl = &result.declarations[0];
        if let Declaration::Decl(d) = decl {
            match &d.body.kind {
                ExprKind::FunctionCall(name, args) => {
                    assert_eq!(name, "vec3");
                    assert_eq!(args.len(), 3);
                    // First arg should be the constant-folded result (1.0)
                    match &args[0].kind {
                        ExprKind::FloatLiteral(v) => assert_eq!(*v, 1.0),
                        other => panic!("Expected FloatLiteral after constant folding, got {:?}", other),
                    }
                }
                ExprKind::Application(_, _) => {
                    panic!("Found Application node after defunctionalization");
                }
                other => panic!("Expected FunctionCall, got {:?}", other),
            }
        }
    }
}
