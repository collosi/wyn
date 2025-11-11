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

impl StaticValue {
    /// Extract a type from a StaticValue for use in closure type construction
    pub fn to_type<G: crate::type_checker::TypeVarGenerator>(&self, gen: &mut G) -> Type {
        match self {
            StaticValue::Dyn(ty) => ty.clone(),
            StaticValue::Arr(elem_sv) => {
                // Array type - create Array(size, element_type)
                // Use a fresh type variable for size since we don't track it statically
                let elem_type = elem_sv.to_type(gen);
                Type::Constructed(TypeName::Array, vec![gen.new_variable(), elem_type])
            }
            StaticValue::Rcd(fields) => {
                // Record type - extract field types
                let field_types: Vec<(String, Type)> =
                    fields.iter().map(|(name, sv)| (name.clone(), sv.to_type(gen))).collect();
                crate::ast::types::record(field_types)
            }
            StaticValue::Lam(_param, _body, _env) => {
                // Lambda - use a fresh type variable for now
                // TODO: construct proper function type
                gen.new_variable()
            }
        }
    }
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

pub struct Defunctionalizer<T: crate::type_checker::TypeVarGenerator> {
    next_function_id: usize,
    type_var_gen: T,
    generated_functions: Vec<DefunctionalizedFunction>,
    node_counter: NodeCounter,
}

impl<T: crate::type_checker::TypeVarGenerator> Defunctionalizer<T> {
    pub fn new_with_counter(node_counter: NodeCounter, type_var_gen: T) -> Self {
        Defunctionalizer {
            next_function_id: 0,
            type_var_gen,
            generated_functions: Vec::new(),
            node_counter,
        }
    }

    pub fn take_type_var_gen(self) -> T {
        self.type_var_gen
    }

    pub fn defunctionalize_program(&mut self, program: &Program) -> Result<Program> {
        let mut new_declarations = Vec::new();
        let mut scope_stack = ScopeStack::new();

        // First pass: collect all declarations and transform them
        for decl in &program.declarations {
            match decl {
                Declaration::Decl(decl_node) => {
                    if decl_node.params.is_empty() {
                        // Variable declarations (let/def with no params) - may contain Application nodes
                        let (transformed_decl, _sv) =
                            self.defunctionalize_decl(decl_node, &mut scope_stack)?;
                        new_declarations.push(transformed_decl);
                    } else {
                        // Function declarations with params - need to defunctionalize body
                        let (transformed_body, _sv) =
                            self.defunctionalize_expression(&decl_node.body, &mut scope_stack)?;
                        let mut transformed_decl = decl_node.clone();
                        transformed_decl.body = transformed_body;
                        new_declarations.push(Declaration::Decl(transformed_decl));
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

        // Add generated functions as def declarations AT THE BEGINNING
        // so they're in scope when functions that reference them are type-checked
        let mut generated_decls = Vec::new();
        for func in &self.generated_functions {
            generated_decls.push(Declaration::Decl(Decl {
                keyword: "def",
                attributes: vec![],
                name: func.name.clone(),
                size_params: vec![],
                type_params: vec![],
                params: func
                    .params
                    .iter()
                    .map(|p| {
                        // Create a pattern with type annotation if the parameter has a concrete type
                        let name_pattern =
                            self.node_counter.mk_node(PatternKind::Name(p.name.clone()), Span::dummy());
                        // Wrap in Typed pattern to preserve type information
                        self.node_counter.mk_node(
                            PatternKind::Typed(Box::new(name_pattern), p.ty.clone()),
                            Span::dummy(),
                        )
                    })
                    .collect(),
                ty: None, // Function definitions don't have explicit type annotations
                body: func.body.clone(),
            }));
        }
        // Prepend generated functions to the beginning of declarations
        generated_decls.extend(new_declarations);
        let new_declarations = generated_decls;

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
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
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
                        StaticValue::Dyn(self.type_var_gen.new_variable()),
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
                    StaticValue::Dyn(self.type_var_gen.new_variable()),
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
                        // Fallback to a fresh type variable if we can't determine the type
                        self.type_var_gen.new_variable()
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
                    StaticValue::Dyn(self.type_var_gen.new_variable()),
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
                        _ => self.type_var_gen.new_variable(),
                    };
                    element_types.push(elem_type);
                }

                Ok((
                    self.node_counter.mk_node(ExprKind::Tuple(transformed_elements), span),
                    StaticValue::Dyn(types::tuple(element_types)),
                ))
            }
            ExprKind::RecordLiteral(fields) => {
                let mut transformed_fields = Vec::new();
                let mut field_svs = HashMap::new();

                for (field_name, field_expr) in fields {
                    let (transformed_expr, sv) =
                        self.defunctionalize_expression(field_expr, scope_stack)?;
                    transformed_fields.push((field_name.clone(), transformed_expr));
                    field_svs.insert(field_name.clone(), sv);
                }

                Ok((
                    self.node_counter.mk_node(ExprKind::RecordLiteral(transformed_fields), span),
                    StaticValue::Rcd(field_svs),
                ))
            }
            ExprKind::LetIn(let_in) => {
                // Transform the value expression
                let (transformed_value, value_sv) =
                    self.defunctionalize_expression(&let_in.value, scope_stack)?;

                // Push new scope and add bindings for all names in the pattern
                scope_stack.push_scope();
                for name in let_in.pattern.collect_names() {
                    scope_stack.insert(name, value_sv.clone());
                }

                // Transform the body expression
                let (transformed_body, body_sv) =
                    self.defunctionalize_expression(&let_in.body, scope_stack)?;

                // Pop scope
                scope_stack.pop_scope();

                Ok((
                    self.node_counter.mk_node(
                        ExprKind::LetIn(crate::ast::LetInExpr {
                            pattern: let_in.pattern.clone(),
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
                    StaticValue::Dyn(self.type_var_gen.new_variable()), // If expressions are runtime values
                ))
            }

            ExprKind::TypeHole => Ok((
                self.node_counter.mk_node(ExprKind::TypeHole, span),
                StaticValue::Dyn(self.type_var_gen.new_variable()), // Type to be inferred
            )),

            ExprKind::Pipe(left, right) => {
                // a |> f desugars to f(a)
                // Defunctionalize both sides
                let (left_expr, _left_sv) = self.defunctionalize_expression(left, scope_stack)?;
                let (right_expr, _right_sv) = self.defunctionalize_expression(right, scope_stack)?;

                // Desugar into application
                self.defunctionalize_application(&right_expr, &[left_expr], scope_stack)
            }

            ExprKind::UnaryOp(op, operand) => {
                let (transformed_operand, sv) = self.defunctionalize_expression(operand, scope_stack)?;
                let transformed_expr = Expression {
                    h: expr.h.clone(),
                    kind: ExprKind::UnaryOp(op.clone(), Box::new(transformed_operand)),
                };
                Ok((transformed_expr, sv))
            }

            ExprKind::Loop(loop_expr) => {
                let transformed_init = if let Some(init) = &loop_expr.init {
                    let (transformed, _) = self.defunctionalize_expression(init, scope_stack)?;
                    Some(Box::new(transformed))
                } else {
                    None
                };
                let (transformed_body, _) =
                    self.defunctionalize_expression(&loop_expr.body, scope_stack)?;
                let transformed_loop = LoopExpr {
                    init: transformed_init,
                    body: Box::new(transformed_body),
                    ..loop_expr.clone()
                };
                let transformed_expr = Expression {
                    h: expr.h.clone(),
                    kind: ExprKind::Loop(transformed_loop),
                };
                Ok((
                    transformed_expr,
                    StaticValue::Dyn(self.type_var_gen.new_variable()),
                ))
            }

            ExprKind::TypeAscription(inner, ty) => {
                let (transformed_inner, sv) = self.defunctionalize_expression(inner, scope_stack)?;
                let transformed_expr = Expression {
                    h: expr.h.clone(),
                    kind: ExprKind::TypeAscription(Box::new(transformed_inner), ty.clone()),
                };
                Ok((transformed_expr, sv))
            }

            ExprKind::TypeCoercion(inner, ty) => {
                let (transformed_inner, sv) = self.defunctionalize_expression(inner, scope_stack)?;
                let transformed_expr = Expression {
                    h: expr.h.clone(),
                    kind: ExprKind::TypeCoercion(Box::new(transformed_inner), ty.clone()),
                };
                Ok((transformed_expr, sv))
            }

            ExprKind::QualifiedName(_, _)
            | ExprKind::Match(_)
            | ExprKind::Range(_)
            | ExprKind::Unsafe(_)
            | ExprKind::Assert(_, _) => {
                todo!(
                    "Expression kinds not yet implemented in defunctionalization: {:?}",
                    expr.kind
                )
            }
        } // NEWCASESHERE - add new cases before this closing brace
    }

    fn defunctionalize_lambda(
        &mut self,
        lambda: &LambdaExpr,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        // Find free variables in the lambda body
        // Build set of bound parameters
        let mut bound = HashSet::new();
        for p in &lambda.params {
            if let Some(n) = p.simple_name() {
                bound.insert(n.to_string());
            }
        }
        let free_vars = self.find_free_variables(&lambda.body, &bound)?;

        // Create a closure record with free variables
        let mut closure_fields = HashMap::new();
        let mut closure_type_fields = Vec::new();
        for var in &free_vars {
            if let Ok(sv) = scope_stack.lookup(var) {
                closure_fields.insert(var.clone(), sv.clone());
                // Extract type from StaticValue for the closure record type
                let var_type = sv.to_type(&mut self.type_var_gen);
                closure_type_fields.push((var.clone(), var_type));
            }
        }

        // Generate a unique function name
        let func_name = format!("__lambda_{}", self.next_function_id);
        self.next_function_id += 1;

        // Create parameters: only add __closure if there are free variables
        let mut func_params = Vec::new();

        if !free_vars.is_empty() {
            let closure_type = crate::ast::types::record(closure_type_fields.clone());
            func_params.push(Parameter {
                attributes: vec![],
                name: "__closure".to_string(),
                ty: closure_type,
            });
        }

        for param in &lambda.params {
            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    CompilerError::ParseError(
                        "Complex patterns in lambda parameters not yet supported".to_string(),
                    )
                })?
                .to_string();
            let param_ty =
                param.pattern_type().cloned().unwrap_or_else(|| self.type_var_gen.new_variable());
            func_params.push(Parameter {
                attributes: vec![],
                name: param_name,
                ty: param_ty,
            });
        }

        // Rewrite free variable references in lambda body to access them from __closure
        let rewritten_body = if !free_vars.is_empty() {
            self.rewrite_free_variables(&*lambda.body, &free_vars)
        } else {
            (*lambda.body).clone()
        };

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
            let param_ty =
                param.pattern_type().cloned().unwrap_or_else(|| self.type_var_gen.new_variable());
            scope_stack.insert(param_name, StaticValue::Dyn(param_ty));
        }

        let (transformed_body, _body_sv) = self.defunctionalize_expression(&rewritten_body, scope_stack)?;

        // Pop parameter scope
        scope_stack.pop_scope();

        // Create the generated function
        let return_type = lambda.return_type.clone().unwrap_or_else(|| self.type_var_gen.new_variable());
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
                            StaticValue::Dyn(self.type_var_gen.new_variable()),
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
            StaticValue::Rcd(_closure_fields) => {
                // Closure application - the closure is a record with __fun field and free variable fields
                // Extract the function name from __fun field and call it with (closure, args...)
                // The closure record itself becomes the first argument (__closure parameter)
                match &transformed_func.kind {
                    ExprKind::RecordLiteral(fields) => {
                        // Extract __fun field
                        let func_name = fields
                            .iter()
                            .find(|(name, _)| name == "__fun")
                            .and_then(|(_, expr)| match &expr.kind {
                                ExprKind::Identifier(name) => Some(name.clone()),
                                _ => None,
                            })
                            .ok_or_else(|| {
                                CompilerError::DefunctionalizationError(
                                    "Closure record missing __fun field".to_string(),
                                )
                            })?;

                        // Call the function with closure as first arg, then the application args
                        let mut all_args = vec![transformed_func];
                        all_args.extend(transformed_args);

                        Ok((
                            self.node_counter
                                .mk_node(ExprKind::FunctionCall(func_name, all_args), Span::dummy()),
                            StaticValue::Dyn(self.type_var_gen.new_variable()),
                        ))
                    }
                    ExprKind::Identifier(closure_var) => {
                        // The closure is a variable - need to extract __fun field at runtime
                        // Create: __fun_var = closure_var.__fun; __fun_var(closure_var, args...)
                        let fun_access = self.node_counter.mk_node(
                            ExprKind::FieldAccess(Box::new(transformed_func.clone()), "__fun".to_string()),
                            Span::dummy(),
                        );

                        // For now, we can't directly call a field access result
                        // This would require let-binding or inline evaluation
                        // Simplest: assume the __fun field contains a known function name
                        // For a more complete implementation, we'd need indirect calls or trampolines
                        Err(CompilerError::DefunctionalizationError(
                            "Closure application with variable closures not yet fully supported. \
                             Closures must be applied directly where they're created."
                                .to_string(),
                        ))
                    }
                    _ => Err(CompilerError::DefunctionalizationError(format!(
                        "Unexpected closure expression form: {:?}",
                        transformed_func.kind
                    ))),
                }
            }
            _ => {
                // Regular function call
                match &transformed_func.kind {
                    ExprKind::Identifier(func_name) => {
                        // Special handling for higher-order builtins like map
                        if func_name == "map" && transformed_args.len() == 2 {
                            // map f xs -> loop-based implementation
                            return self.defunctionalize_map(
                                &transformed_args[0],
                                &transformed_args[1],
                                scope_stack,
                            );
                        }

                        Ok((
                            self.node_counter.mk_node(
                                ExprKind::FunctionCall(func_name.clone(), transformed_args),
                                Span::dummy(),
                            ),
                            StaticValue::Dyn(self.type_var_gen.new_variable()),
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
                            StaticValue::Dyn(self.type_var_gen.new_variable()),
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
                            StaticValue::Dyn(self.type_var_gen.new_variable()),
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
            ExprKind::RecordLiteral(fields) => {
                for (_name, field_expr) in fields {
                    self.collect_free_variables(field_expr, bound_vars, free_vars)?;
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

                // Add all names from pattern to bound variables and collect from body
                let mut extended_bound = bound_vars.clone();
                for name in let_in.pattern.collect_names() {
                    extended_bound.insert(name);
                }
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
        // Create a record literal with __fun field (function name) and free variables
        // The __fun field stores the function to call
        // The other field names match the variable names so that rewrite_free_variables works
        let mut fields = Vec::new();

        // Add __fun field with the function name
        let fun_value =
            self.node_counter.mk_node(ExprKind::Identifier(func_name.to_string()), Span::dummy());
        fields.push(("__fun".to_string(), fun_value));

        // Add free variable fields
        for var in free_vars {
            let field_value = self.node_counter.mk_node(ExprKind::Identifier(var.clone()), Span::dummy());
            fields.push((var.clone(), field_value));
        }

        Ok(self.node_counter.mk_node(ExprKind::RecordLiteral(fields), Span::dummy()))
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
            _ => {
                return Err(CompilerError::SpirvError(
                    "map currently only supports simple function identifiers".to_string(),
                ));
            }
        };

        // Generate unique variable names
        let i_var = format!("__map_i_{}", self.next_function_id);
        let out_var = format!("__map_out_{}", self.next_function_id);
        let xs_var = format!("__map_xs_{}", self.next_function_id);
        let len_var = format!("__map_len_{}", self.next_function_id);
        self.next_function_id += 1;

        // Build ALL leaf nodes first to avoid borrow checker issues
        let xs_ident_for_len = self.node_counter.mk_node(ExprKind::Identifier(xs_var.clone()), span);
        let len_ident_for_replicate =
            self.node_counter.mk_node(ExprKind::Identifier(len_var.clone()), span);
        let zero_for_replicate = self.node_counter.mk_node(ExprKind::IntLiteral(0), span);

        // length xs
        let len_call = self.node_counter.mk_node(
            ExprKind::FunctionCall("length".to_string(), vec![xs_ident_for_len]),
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
        let array_index =
            self.node_counter.mk_node(ExprKind::ArrayIndex(Box::new(xs_ident), Box::new(i_ident2)), span);

        // f xs[i]
        let func_app =
            self.node_counter.mk_node(ExprKind::FunctionCall(func_name, vec![array_index]), span);

        // __array_update(out, i, f xs[i])
        let updated_out = self.node_counter.mk_node(
            ExprKind::FunctionCall("__array_update".to_string(), vec![out_ident, i_ident3, func_app]),
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
        let loop_body = self.node_counter.mk_node(ExprKind::Tuple(vec![i_inc, updated_out]), span);

        // (0, out)
        let initial_value = self.node_counter.mk_node(ExprKind::Tuple(vec![zero_lit, out_ident2]), span);

        // loop (i, out) = (0, out) while i < len do (i + 1, updated_out)
        let i_pattern = self.node_counter.mk_node(PatternKind::Name(i_var), span);
        let out_pattern = self.node_counter.mk_node(PatternKind::Name(out_var.clone()), span);
        let loop_pattern =
            self.node_counter.mk_node(PatternKind::Tuple(vec![i_pattern, out_pattern]), span);
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
        let out_pattern = self.node_counter.mk_node(crate::ast::PatternKind::Name(out_var), span);
        let with_out = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: out_pattern,
                ty: None,
                value: Box::new(init_out),
                body: Box::new(loop_expr),
            }),
            span,
        );

        // let len = length xs in <with_out>
        let len_pattern = self.node_counter.mk_node(crate::ast::PatternKind::Name(len_var), span);
        let with_len = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: len_pattern,
                ty: None,
                value: Box::new(len_call),
                body: Box::new(with_out),
            }),
            span,
        );

        // let xs = <array> in <with_len>
        let xs_pattern = self.node_counter.mk_node(crate::ast::PatternKind::Name(xs_var), span);
        let result = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: xs_pattern,
                ty: None,
                value: Box::new(array.clone()),
                body: Box::new(with_len),
            }),
            span,
        );

        Ok((result, StaticValue::Dyn(self.type_var_gen.new_variable())))
    }

    /// Rewrite free variable references in an expression to access them from a closure record
    /// Replaces Identifier(var) with FieldAccess(Identifier("__closure"), var) for each free variable
    fn rewrite_free_variables(&mut self, expr: &Expression, free_vars: &HashSet<String>) -> Expression {
        let span = expr.h.span;
        match &expr.kind {
            ExprKind::Identifier(name) if free_vars.contains(name) => {
                // Rewrite free variable to closure field access: __closure.name
                let closure_id =
                    self.node_counter.mk_node(ExprKind::Identifier("__closure".to_string()), span);
                self.node_counter.mk_node(ExprKind::FieldAccess(Box::new(closure_id), name.clone()), span)
            }
            ExprKind::Identifier(_) => expr.clone(),
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::TypeHole => expr.clone(),
            ExprKind::RecordLiteral(fields) => {
                let rewritten_fields: Vec<_> = fields
                    .iter()
                    .map(|(name, field_expr)| {
                        (name.clone(), self.rewrite_free_variables(field_expr, free_vars))
                    })
                    .collect();
                self.node_counter.mk_node(ExprKind::RecordLiteral(rewritten_fields), span)
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_rewritten = self.rewrite_free_variables(left, free_vars);
                let right_rewritten = self.rewrite_free_variables(right, free_vars);
                self.node_counter.mk_node(
                    ExprKind::BinaryOp(op.clone(), Box::new(left_rewritten), Box::new(right_rewritten)),
                    span,
                )
            }
            ExprKind::UnaryOp(op, operand) => {
                let operand_rewritten = self.rewrite_free_variables(operand, free_vars);
                self.node_counter.mk_node(ExprKind::UnaryOp(op.clone(), Box::new(operand_rewritten)), span)
            }
            ExprKind::FunctionCall(name, args) => {
                let args_rewritten: Vec<_> =
                    args.iter().map(|arg| self.rewrite_free_variables(arg, free_vars)).collect();
                self.node_counter.mk_node(ExprKind::FunctionCall(name.clone(), args_rewritten), span)
            }
            ExprKind::Application(func, args) => {
                let func_rewritten = self.rewrite_free_variables(func, free_vars);
                let args_rewritten: Vec<_> =
                    args.iter().map(|arg| self.rewrite_free_variables(arg, free_vars)).collect();
                self.node_counter.mk_node(
                    ExprKind::Application(Box::new(func_rewritten), args_rewritten),
                    span,
                )
            }
            ExprKind::LetIn(let_in) => {
                let value_rewritten = self.rewrite_free_variables(&let_in.value, free_vars);

                // Remove bound names from free_vars for the body
                let mut body_free_vars = free_vars.clone();
                for name in let_in.pattern.collect_names() {
                    body_free_vars.remove(&name);
                }
                let body_rewritten = self.rewrite_free_variables(&let_in.body, &body_free_vars);

                self.node_counter.mk_node(
                    ExprKind::LetIn(LetInExpr {
                        pattern: let_in.pattern.clone(),
                        ty: let_in.ty.clone(),
                        value: Box::new(value_rewritten),
                        body: Box::new(body_rewritten),
                    }),
                    span,
                )
            }
            ExprKind::If(if_expr) => {
                let condition = self.rewrite_free_variables(&if_expr.condition, free_vars);
                let then_branch = self.rewrite_free_variables(&if_expr.then_branch, free_vars);
                let else_branch = self.rewrite_free_variables(&if_expr.else_branch, free_vars);
                self.node_counter.mk_node(
                    ExprKind::If(IfExpr {
                        condition: Box::new(condition),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    }),
                    span,
                )
            }
            ExprKind::Tuple(elements) => {
                let elements_rewritten: Vec<_> =
                    elements.iter().map(|elem| self.rewrite_free_variables(elem, free_vars)).collect();
                self.node_counter.mk_node(ExprKind::Tuple(elements_rewritten), span)
            }
            ExprKind::ArrayLiteral(elements) => {
                let elements_rewritten: Vec<_> =
                    elements.iter().map(|elem| self.rewrite_free_variables(elem, free_vars)).collect();
                self.node_counter.mk_node(ExprKind::ArrayLiteral(elements_rewritten), span)
            }
            ExprKind::ArrayIndex(array, index) => {
                let array_rewritten = self.rewrite_free_variables(array, free_vars);
                let index_rewritten = self.rewrite_free_variables(index, free_vars);
                self.node_counter.mk_node(
                    ExprKind::ArrayIndex(Box::new(array_rewritten), Box::new(index_rewritten)),
                    span,
                )
            }
            ExprKind::FieldAccess(expr_inner, field) => {
                let expr_rewritten = self.rewrite_free_variables(expr_inner, free_vars);
                self.node_counter.mk_node(
                    ExprKind::FieldAccess(Box::new(expr_rewritten), field.clone()),
                    span,
                )
            }
            ExprKind::Loop(loop_expr) => {
                let init_rewritten = loop_expr
                    .init
                    .as_ref()
                    .map(|init| Box::new(self.rewrite_free_variables(init, free_vars)));

                // Remove loop pattern bound names from free_vars for the body
                let mut body_free_vars = free_vars.clone();
                for name in loop_expr.pattern.collect_names() {
                    body_free_vars.remove(&name);
                }

                let form_rewritten = match &loop_expr.form {
                    LoopForm::While(cond) => {
                        LoopForm::While(Box::new(self.rewrite_free_variables(cond, &body_free_vars)))
                    }
                    LoopForm::For(iter_name, iter_expr) => {
                        let iter_expr_rewritten = self.rewrite_free_variables(iter_expr, free_vars);
                        // Also remove iterator name from body free vars
                        body_free_vars.remove(iter_name);
                        LoopForm::For(iter_name.clone(), Box::new(iter_expr_rewritten))
                    }
                    LoopForm::ForIn(iter_pat, iter_expr) => {
                        let iter_expr_rewritten = self.rewrite_free_variables(iter_expr, free_vars);
                        // Also remove iterator pattern names
                        for name in iter_pat.collect_names() {
                            body_free_vars.remove(&name);
                        }
                        LoopForm::ForIn(iter_pat.clone(), Box::new(iter_expr_rewritten))
                    }
                };

                let body_rewritten = self.rewrite_free_variables(&loop_expr.body, &body_free_vars);

                self.node_counter.mk_node(
                    ExprKind::Loop(LoopExpr {
                        init: init_rewritten,
                        pattern: loop_expr.pattern.clone(),
                        form: form_rewritten,
                        body: Box::new(body_rewritten),
                    }),
                    span,
                )
            }
            ExprKind::TypeAscription(inner, ty) => {
                let inner_rewritten = self.rewrite_free_variables(inner, free_vars);
                self.node_counter.mk_node(
                    ExprKind::TypeAscription(Box::new(inner_rewritten), ty.clone()),
                    span,
                )
            }
            ExprKind::TypeCoercion(inner, ty) => {
                let inner_rewritten = self.rewrite_free_variables(inner, free_vars);
                self.node_counter.mk_node(
                    ExprKind::TypeCoercion(Box::new(inner_rewritten), ty.clone()),
                    span,
                )
            }
            // For lambda, we don't rewrite - it will be defunctionalized separately
            ExprKind::Lambda(_) => expr.clone(),
            // Other expression kinds - just clone for now
            _ => expr.clone(),
        }
    }
}
