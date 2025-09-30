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
                Declaration::Val(val_decl) => {
                    // Type signatures only
                    new_declarations.push(Declaration::Val(val_decl.clone()));
                }
            }
        }

        // Add generated functions as def declarations
        for func in &self.generated_functions {
            new_declarations.push(Declaration::Decl(Decl {
                keyword: "def",
                attributes: vec![],
                name: func.name.clone(),
                params: func.params.iter().map(|p| DeclParam::Untyped(p.name.clone())).collect(),
                ty: None, // Function definitions don't have explicit type annotations
                return_attributes: vec![],
                attributed_return_type: None,
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
            return_attributes: decl.return_attributes.clone(),
            attributed_return_type: decl.attributed_return_type.clone(),
            body: transformed_expr,
        });

        Ok((transformed_decl, sv))
    }


    fn defunctionalize_expression(
        &mut self,
        expr: &Expression,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        match expr {
            Expression::IntLiteral(n) => {
                Ok((Expression::IntLiteral(*n), StaticValue::Dyn(types::i32())))
            }
            Expression::FloatLiteral(f) => {
                Ok((Expression::FloatLiteral(*f), StaticValue::Dyn(types::f32())))
            }
            Expression::Identifier(name) => {
                if let Some(sv) = scope_stack.lookup(name) {
                    match sv {
                        StaticValue::Dyn(_) => {
                            // Regular variable reference
                            Ok((Expression::Identifier(name.clone()), sv.clone()))
                        }
                        StaticValue::Lam(_, _, _) => {
                            // Reference to a function - this would need special handling
                            // For now, keep as identifier
                            Ok((Expression::Identifier(name.clone()), sv.clone()))
                        }
                        StaticValue::Rcd(_) => {
                            // Reference to a closure record
                            Ok((Expression::Identifier(name.clone()), sv.clone()))
                        }
                        StaticValue::Arr(_) => {
                            Ok((Expression::Identifier(name.clone()), sv.clone()))
                        }
                    }
                } else {
                    // Unknown variable - assume dynamic with type variable
                    Ok((
                        Expression::Identifier(name.clone()),
                        StaticValue::Dyn(polytype::Type::Variable(0)),
                    ))
                }
            }
            Expression::Lambda(lambda) => self.defunctionalize_lambda(lambda, scope_stack),
            Expression::Application(func, args) => {
                self.defunctionalize_application(func, args, scope_stack)
            }
            Expression::ArrayLiteral(elements) => {
                let mut transformed_elements = Vec::new();
                let mut element_sv = None;

                for elem in elements {
                    let (transformed_elem, sv) =
                        self.defunctionalize_expression(elem, scope_stack)?;
                    transformed_elements.push(transformed_elem);

                    // All elements should have the same static value structure
                    if element_sv.is_none() {
                        element_sv = Some(sv);
                    }
                }

                let array_sv = StaticValue::Arr(Box::new(
                    element_sv.unwrap_or(StaticValue::Dyn(types::i32())),
                ));
                Ok((Expression::ArrayLiteral(transformed_elements), array_sv))
            }
            Expression::ArrayIndex(array, index) => {
                let (transformed_array, _array_sv) =
                    self.defunctionalize_expression(array, scope_stack)?;
                let (transformed_index, _index_sv) =
                    self.defunctionalize_expression(index, scope_stack)?;

                // Result type depends on array element type - for now, assume dynamic
                Ok((
                    Expression::ArrayIndex(
                        Box::new(transformed_array),
                        Box::new(transformed_index),
                    ),
                    StaticValue::Dyn(polytype::Type::Variable(1)),
                ))
            }
            Expression::BinaryOp(op, left, right) => {
                let (transformed_left, left_sv) =
                    self.defunctionalize_expression(left, scope_stack)?;
                let (transformed_right, right_sv) =
                    self.defunctionalize_expression(right, scope_stack)?;

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
                    Expression::BinaryOp(
                        op.clone(),
                        Box::new(transformed_left),
                        Box::new(transformed_right),
                    ),
                    StaticValue::Dyn(result_type),
                ))
            }
            Expression::FunctionCall(name, args) => {
                // Regular function calls (first-order) remain unchanged
                let mut transformed_args = Vec::new();
                for arg in args {
                    let (transformed_arg, _sv) =
                        self.defunctionalize_expression(arg, scope_stack)?;
                    transformed_args.push(transformed_arg);
                }

                Ok((
                    Expression::FunctionCall(name.clone(), transformed_args),
                    StaticValue::Dyn(polytype::Type::Variable(2)),
                ))
            }
            Expression::Tuple(elements) => {
                let mut transformed_elements = Vec::new();
                let mut element_types = Vec::new();

                for elem in elements {
                    let (transformed_elem, sv) =
                        self.defunctionalize_expression(elem, scope_stack)?;
                    transformed_elements.push(transformed_elem);

                    // Extract type from static value
                    let elem_type = match sv {
                        StaticValue::Dyn(ty) => ty,
                        _ => polytype::Type::Variable(3),
                    };
                    element_types.push(elem_type);
                }

                Ok((
                    Expression::Tuple(transformed_elements),
                    StaticValue::Dyn(types::tuple(element_types)),
                ))
            }
            Expression::LetIn(let_in) => {
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
                    Expression::LetIn(crate::ast::LetInExpr {
                        name: let_in.name.clone(),
                        ty: let_in.ty.clone(),
                        value: Box::new(transformed_value),
                        body: Box::new(transformed_body),
                    }),
                    body_sv,
                ))
            }
            Expression::FieldAccess(expr, field) => {
                let (transformed_expr, expr_sv) = self.defunctionalize_expression(expr, scope_stack)?;
                Ok((
                    Expression::FieldAccess(Box::new(transformed_expr), field.clone()),
                    expr_sv, // Field access doesn't change the static value representation
                ))
            }
            Expression::If(if_expr) => {
                let (condition, _condition_sv) = self.defunctionalize_expression(&if_expr.condition, scope_stack)?;
                let (then_branch, _then_sv) = self.defunctionalize_expression(&if_expr.then_branch, scope_stack)?;
                let (else_branch, _else_sv) = self.defunctionalize_expression(&if_expr.else_branch, scope_stack)?;
                Ok((
                    Expression::If(IfExpr {
                        condition: Box::new(condition),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    }),
                    StaticValue::Dyn(Type::Constructed(TypeName::Str("unknown"), vec![])), // If expressions are runtime values
                ))
            }
        }
    }

    fn defunctionalize_lambda(
        &mut self,
        lambda: &LambdaExpr,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        // Find free variables in the lambda body
        let free_vars = self.find_free_variables(
            &lambda.body,
            &lambda.params.iter().map(|p| p.name.clone()).collect(),
        )?;

        // Create a closure record with free variables
        let mut closure_fields = HashMap::new();
        for var in &free_vars {
            if let Some(sv) = scope_stack.lookup(var) {
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
            func_params.push(Parameter {
                attributes: vec![],
                name: param.name.clone(),
                ty: param.ty.clone().unwrap_or(polytype::Type::Variable(5)),
            });
        }

        // Transform lambda body with parameter scope
        scope_stack.push_scope();
        for param in &lambda.params {
            scope_stack.insert(
                param.name.clone(),
                StaticValue::Dyn(param.ty.clone().unwrap_or(polytype::Type::Variable(6))),
            );
        }

        let (transformed_body, _body_sv) =
            self.defunctionalize_expression(&lambda.body, scope_stack)?;

        // Pop parameter scope
        scope_stack.pop_scope();

        // Create the generated function
        let return_type = lambda
            .return_type
            .clone()
            .unwrap_or(polytype::Type::Variable(7));
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
                Expression::Identifier(func_name),
                StaticValue::Lam(
                    "__unused".to_string(),
                    (*lambda.body).clone(),
                    HashMap::new(),
                ),
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
                match transformed_func {
                    Expression::Identifier(func_name) => {
                        // Function call without closure
                        Ok((
                            Expression::FunctionCall(func_name, transformed_args),
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
                match transformed_func {
                    Expression::Identifier(func_name) => Ok((
                        Expression::FunctionCall(func_name, transformed_args),
                        StaticValue::Dyn(polytype::Type::Variable(2)),
                    )),
                    _ => Err(CompilerError::SpirvError(
                        "Invalid function in application".to_string(),
                    )),
                }
            }
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
        match expr {
            Expression::Identifier(name) => {
                if !bound_vars.contains(name) {
                    free_vars.insert(name.clone());
                }
            }
            Expression::Lambda(lambda) => {
                let mut extended_bound = bound_vars.clone();
                for param in &lambda.params {
                    extended_bound.insert(param.name.clone());
                }
                self.collect_free_variables(&lambda.body, &extended_bound, free_vars)?;
            }
            Expression::Application(func, args) => {
                self.collect_free_variables(func, bound_vars, free_vars)?;
                for arg in args {
                    self.collect_free_variables(arg, bound_vars, free_vars)?;
                }
            }
            Expression::ArrayLiteral(elements) => {
                for elem in elements {
                    self.collect_free_variables(elem, bound_vars, free_vars)?;
                }
            }
            Expression::ArrayIndex(array, index) => {
                self.collect_free_variables(array, bound_vars, free_vars)?;
                self.collect_free_variables(index, bound_vars, free_vars)?;
            }
            Expression::BinaryOp(_, left, right) => {
                self.collect_free_variables(left, bound_vars, free_vars)?;
                self.collect_free_variables(right, bound_vars, free_vars)?;
            }
            Expression::FunctionCall(_, args) => {
                for arg in args {
                    self.collect_free_variables(arg, bound_vars, free_vars)?;
                }
            }
            Expression::Tuple(elements) => {
                for elem in elements {
                    self.collect_free_variables(elem, bound_vars, free_vars)?;
                }
            }
            Expression::LetIn(let_in) => {
                // Collect free variables from value expression
                self.collect_free_variables(&let_in.value, bound_vars, free_vars)?;

                // Add let binding to bound variables and collect from body
                let mut extended_bound = bound_vars.clone();
                extended_bound.insert(let_in.name.clone());
                self.collect_free_variables(&let_in.body, &extended_bound, free_vars)?;
            }
            Expression::IntLiteral(_) | Expression::FloatLiteral(_) => {
                // No free variables in literals
            }
            Expression::FieldAccess(expr, _field) => {
                self.collect_free_variables(expr, bound_vars, free_vars)?;
            }
            Expression::If(if_expr) => {
                self.collect_free_variables(&if_expr.condition, bound_vars, free_vars)?;
                self.collect_free_variables(&if_expr.then_branch, bound_vars, free_vars)?;
                self.collect_free_variables(&if_expr.else_branch, bound_vars, free_vars)?;
            }
        }
        Ok(())
    }

    fn create_closure_record(
        &self,
        func_name: &str,
        free_vars: &HashSet<String>,
    ) -> Result<Expression> {
        // For now, create a simple record-like structure
        // In a full implementation, this would create a proper record expression
        // For SPIR-V compatibility, we might need to represent this as an array or struct

        // Create a tuple with function name and free variables
        let mut elements = vec![Expression::Identifier(func_name.to_string())];
        for var in free_vars {
            elements.push(Expression::Identifier(var.clone()));
        }

        Ok(Expression::Tuple(elements))
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
}
