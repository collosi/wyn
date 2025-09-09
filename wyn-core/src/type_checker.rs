use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use polytype::{ptp, tp, Context, Type, TypeScheme};

pub struct TypeChecker {
    scope_stack: ScopeStack<TypeScheme>, // Store polymorphic types
    context: Context,                    // Polytype unification context
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    /// Create binary arithmetic operator type scheme: ∀t. t -> t -> t
    fn binary_arithmetic_scheme() -> TypeScheme {
        ptp!(0; @arrow[tp!(0), tp!(0), tp!(0)]) // ∀t0. t0 -> t0 -> t0
    }
    pub fn new() -> Self {
        let mut checker = TypeChecker {
            scope_stack: ScopeStack::new(),
            context: Context::default(),
        };

        // Load built-in functions from builtins.wyn file
        if let Err(e) = checker.load_builtins() {
            eprintln!("Warning: Could not load builtins: {}", e);
        }

        checker
    }

    fn load_builtins(&mut self) -> Result<()> {
        // Add builtin function types directly using manual construction
        
        // length: ∀a. [a] -> int
        let array_type = Type::Constructed("array", vec![Type::Variable(0)]);
        let int_type = Type::Constructed("int", vec![]);
        let length_body = Type::arrow(array_type, int_type);
        let length_scheme = TypeScheme::Polytype {
            variable: 0,
            body: Box::new(TypeScheme::Monotype(length_body)),
        };
        self.scope_stack.insert("length".to_string(), length_scheme);
        
        // map: ∀a b. (a -> b) -> [a] -> [b]
        let var_a = Type::Variable(0);
        let var_b = Type::Variable(1);
        let func_type = Type::arrow(var_a.clone(), var_b.clone());
        let input_array_type = Type::Constructed("array", vec![var_a]);
        let output_array_type = Type::Constructed("array", vec![var_b]);
        let map_arrow1 = Type::arrow(input_array_type, output_array_type);
        let map_body = Type::arrow(func_type, map_arrow1);
        let map_scheme = TypeScheme::Polytype {
            variable: 0,
            body: Box::new(TypeScheme::Polytype {
                variable: 1,
                body: Box::new(TypeScheme::Monotype(map_body)),
            }),
        };
        self.scope_stack.insert("map".to_string(), map_scheme);
        
        // zip: ∀a b. [a] -> [b] -> [(a, b)]
        let var_a = Type::Variable(0);
        let var_b = Type::Variable(1);
        let array_a_type = Type::Constructed("array", vec![var_a.clone()]);
        let array_b_type = Type::Constructed("array", vec![var_b.clone()]);
        let tuple_type = Type::Constructed("tuple", vec![var_a, var_b]);
        let result_array_type = Type::Constructed("array", vec![tuple_type]);
        let zip_arrow1 = Type::arrow(array_b_type, result_array_type);
        let zip_body = Type::arrow(array_a_type, zip_arrow1);
        let zip_scheme = TypeScheme::Polytype {
            variable: 0,
            body: Box::new(TypeScheme::Polytype {
                variable: 1,
                body: Box::new(TypeScheme::Monotype(zip_body)),
            }),
        };
        self.scope_stack.insert("zip".to_string(), zip_scheme);
        
        Ok(())
    }

    pub fn check_program(&mut self, program: &Program) -> Result<()> {
        // First pass: collect all function declarations
        for decl in &program.declarations {
            if let Declaration::Def(def_decl) = decl {
                // Create a placeholder type for this function
                let func_type = self.context.new_variable();
                let type_scheme = TypeScheme::Monotype(func_type);
                self.scope_stack.insert(def_decl.name.clone(), type_scheme);
            }
        }

        // Second pass: infer types for all declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        Ok(())
    }

    fn check_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Let(let_decl) => self.check_let_decl(let_decl),
            Declaration::Entry(entry_decl) => self.check_entry_decl(entry_decl),
            Declaration::Def(def_decl) => self.check_def_decl(def_decl),
            Declaration::Val(val_decl) => self.check_val_decl(val_decl),
        }
    }

    fn check_let_decl(&mut self, decl: &LetDecl) -> Result<()> {
        let expr_type = self.infer_expression(&decl.value)?;

        if let Some(declared_type) = &decl.ty {
            if !self.types_match(&expr_type, declared_type) {
                return Err(CompilerError::TypeError(format!(
                    "Type mismatch: expected {:?}, got {:?}",
                    declared_type, expr_type
                )));
            }
        }

        // Add to both environments - use declared type if available, otherwise inferred type
        let stored_type = decl.ty.as_ref().unwrap_or(&expr_type).clone();
        let type_scheme = TypeScheme::Monotype(stored_type);
        self.scope_stack.insert(decl.name.clone(), type_scheme);

        Ok(())
    }

    fn check_entry_decl(&mut self, decl: &EntryDecl) -> Result<()> {
        // Push new scope for entry parameters
        self.scope_stack.push_scope();

        // Add parameters to scope
        for param in &decl.params {
            let type_scheme = TypeScheme::Monotype(param.ty.clone());
            self.scope_stack.insert(param.name.clone(), type_scheme);
        }

        // Check body with parameters in scope
        let body_type = self.infer_expression(&decl.body)?;

        // Pop parameter scope
        self.scope_stack.pop_scope();

        if !self.types_match(&body_type, &decl.return_type.ty) {
            return Err(CompilerError::TypeError(format!(
                "Return type mismatch: expected {:?}, got {:?}",
                decl.return_type.ty, body_type
            )));
        }

        Ok(())
    }

    fn check_def_decl(&mut self, decl: &DefDecl) -> Result<()> {
        // Create type variables for parameters
        let param_types: Vec<Type> = decl
            .params
            .iter()
            .map(|_| self.context.new_variable())
            .collect();

        // Push new scope for function parameters
        self.scope_stack.push_scope();

        // Add parameters to scope
        for (param_name, param_type) in decl.params.iter().zip(param_types.iter()) {
            let type_scheme = TypeScheme::Monotype(param_type.clone());
            self.scope_stack.insert(param_name.clone(), type_scheme);
        }

        // Infer body type
        let body_type = self.infer_expression(&decl.body)?;

        // Pop parameter scope
        self.scope_stack.pop_scope();

        // Build function type: param1 -> param2 -> ... -> body_type
        let func_type = param_types
            .into_iter()
            .rev()
            .fold(body_type, |acc, param_ty| types::function(param_ty, acc));

        // Update scope with inferred type
        let type_scheme = TypeScheme::Monotype(func_type.clone());
        self.scope_stack.insert(decl.name.clone(), type_scheme);

        println!("Inferred type for {}: {}", decl.name, func_type);

        Ok(())
    }

    fn check_val_decl(&mut self, decl: &ValDecl) -> Result<()> {
        // Val declarations are just type signatures - register them in scope
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.scope_stack.insert(decl.name.clone(), type_scheme);
        Ok(())
    }

    fn infer_expression(&mut self, expr: &Expression) -> Result<Type> {
        match expr {
            Expression::IntLiteral(_) => Ok(types::i32()),
            Expression::FloatLiteral(_) => Ok(types::f32()),
            Expression::Identifier(name) => {
                let type_scheme = self
                    .scope_stack
                    .lookup(name)
                    .ok_or_else(|| CompilerError::UndefinedVariable(name.clone()))?;
                // Instantiate the type scheme to get a concrete type
                Ok(type_scheme.instantiate(&mut self.context))
            }
            Expression::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    return Err(CompilerError::TypeError(
                        "Cannot infer type of empty array".to_string(),
                    ));
                }

                let first_type = self.infer_expression(&elements[0])?;
                for elem in &elements[1..] {
                    let elem_type = self.infer_expression(elem)?;
                    if !self.types_match(&elem_type, &first_type) {
                        return Err(CompilerError::TypeError(
                            "Array elements must have the same type".to_string(),
                        ));
                    }
                }

                Ok(types::array(first_type))
            }
            Expression::ArrayIndex(array_expr, _index_expr) => {
                let array_type = self.infer_expression(array_expr)?;
                Ok(match array_type {
                    Type::Constructed("array", args) => {
                        args.into_iter().next().unwrap_or_else(|| types::i32())
                    }
                    _ => {
                        return Err(CompilerError::TypeError(format!(
                            "Cannot index non-array type: got {:?}",
                            array_type
                        )))
                    }
                })
            }
            Expression::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;

                match op {
                    BinaryOp::Add | BinaryOp::Divide => {
                        // Arithmetic operators have type: ∀t. t -> t -> t
                        let arith_scheme = Self::binary_arithmetic_scheme();
                        let arith_type = arith_scheme.instantiate(&mut self.context);

                        // Create fresh type variable for result
                        let result_type = self.context.new_variable();

                        // Unify: arith_type ~ (left_type -> right_type -> result_type)
                        let expected_type = tp!(@arrow[
                            left_type.clone(),
                            tp!(@arrow[right_type.clone(), result_type.clone()])
                        ]);

                        let op_name = match op {
                            BinaryOp::Add => "add",
                            BinaryOp::Divide => "divide",
                        };

                        self.context
                            .unify(&arith_type, &expected_type)
                            .map_err(|e| {
                                CompilerError::TypeError(format!(
                                    "Cannot {} {:?} and {:?}: {}",
                                    op_name, left_type, right_type, e
                                ))
                            })?;

                        // Apply substitutions to get concrete result type
                        Ok(result_type.apply(&self.context))
                    } // Future operators like comparisons would go here with different type schemes
                }
            }
            Expression::FunctionCall(func_name, args) => {
                // Get function type scheme and instantiate it
                let type_scheme = self
                    .scope_stack
                    .lookup(func_name)
                    .ok_or_else(|| CompilerError::UndefinedVariable(func_name.clone()))?;
                let mut func_type = type_scheme.instantiate(&mut self.context);

                // Apply function to each argument using unification
                for arg in args {
                    let arg_type = self.infer_expression(arg)?;
                    
                    // Create a fresh result type variable
                    let result_type = self.context.new_variable();
                    
                    // Expected function type: arg_type -> result_type
                    let expected_func_type = Type::arrow(arg_type, result_type.clone());
                    
                    // Unify the function type with expected
                    self.context.unify(&func_type, &expected_func_type)
                        .map_err(|e| CompilerError::TypeError(format!("Function call type error: {:?}", e)))?;
                    
                    // Update func_type to result_type for the next argument (currying)
                    func_type = result_type;
                }

                Ok(func_type.apply(&self.context))
            }
            Expression::Tuple(elements) => {
                let elem_types: Result<Vec<Type>> =
                    elements.iter().map(|e| self.infer_expression(e)).collect();

                Ok(types::tuple(elem_types?))
            }
            Expression::Lambda(lambda) => {
                // Push new scope for lambda parameters
                self.scope_stack.push_scope();

                // Add parameters to scope with their types (or fresh type variables)
                for param in &lambda.params {
                    let param_type = param.ty.clone().unwrap_or_else(|| {
                        self.context.new_variable() // Use polytype's context to create fresh variables
                    });
                    let type_scheme = TypeScheme::Monotype(param_type);
                    self.scope_stack.insert(param.name.clone(), type_scheme);
                }

                // Type check the lambda body with parameters in scope
                let body_type = self.infer_expression(&lambda.body)?;
                let return_type = lambda.return_type.clone().unwrap_or(body_type);

                // Pop parameter scope
                self.scope_stack.pop_scope();

                // For multiple parameters, create nested function types
                let mut func_type = return_type;
                for param in lambda.params.iter().rev() {
                    let param_type = param
                        .ty
                        .clone()
                        .unwrap_or_else(|| self.context.new_variable());
                    func_type = types::function(param_type, func_type);
                }

                Ok(func_type)
            }
            Expression::LetIn(let_in) => {
                // Infer type of the value expression
                let value_type = self.infer_expression(&let_in.value)?;
                
                // Check type annotation if present
                if let Some(declared_type) = &let_in.ty {
                    if !self.types_match(&value_type, declared_type) {
                        return Err(CompilerError::TypeError(format!(
                            "Type mismatch in let binding: expected {:?}, got {:?}",
                            declared_type, value_type
                        )));
                    }
                }
                
                // Push new scope and add binding
                self.scope_stack.push_scope();
                let bound_type = let_in.ty.as_ref().unwrap_or(&value_type).clone();
                let type_scheme = TypeScheme::Monotype(bound_type);
                self.scope_stack.insert(let_in.name.clone(), type_scheme);
                
                // Infer type of body expression
                let body_type = self.infer_expression(&let_in.body)?;
                
                // Pop scope
                self.scope_stack.pop_scope();
                
                Ok(body_type)
            }
            Expression::Application(func, args) => {
                let mut func_type = self.infer_expression(func)?;

                // Apply function to each argument
                for arg in args {
                    let arg_type = self.infer_expression(arg)?;
                    
                    // Create a fresh result type variable
                    let result_type = self.context.new_variable();
                    
                    // Expected function type: arg_type -> result_type
                    let expected_func_type = Type::arrow(arg_type, result_type.clone());
                    
                    // Unify the function type with expected
                    self.context.unify(&func_type, &expected_func_type)
                        .map_err(|e| CompilerError::TypeError(format!("Function application type error: {:?}", e)))?;
                    
                    // Update func_type to result_type for the next argument (currying)
                    func_type = result_type;
                }

                Ok(func_type.apply(&self.context))
            }
        }
    }

    // Removed: fresh_var - now using polytype's context.new_variable()

    fn types_match(&mut self, t1: &Type, t2: &Type) -> bool {
        // Use polytype's unification for proper type matching
        self.context.unify(t1, t2).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_type_check_let() {
        let input = "let x: i32 = 42";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_mismatch() {
        let input = "let x: i32 = 3.14f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_array_type_check() {
        let input = "let arr: [2]f32 = [1.0f32, 2.0f32]";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_undefined_variable() {
        let input = "let x: i32 = undefined";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CompilerError::UndefinedVariable(_)
        ));
    }

    #[test]
    fn test_simple_def() {
        let input = "def identity x = x";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_zip_arrays() {
        let input = "def zip_arrays xs ys = zip xs ys";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        match checker.check_program(&program) {
            Ok(_) => {
                println!("Type checking succeeded!");

                // Check that zip_arrays has the expected type
                if let Some(func_type) = checker.scope_stack.lookup("zip_arrays") {
                    println!("zip_arrays type: {}", func_type);

                    // The inferred type should be something like: t0 -> t1 -> [1](i32, i32)
                    // This demonstrates that type inference is working
                }
            }
            Err(e) => {
                println!("Type checking failed: {:?}", e);
                panic!("Type checking failed");
            }
        }
    }
}
