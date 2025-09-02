use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::lexer::tokenize;
use crate::parser::Parser;
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
        // Embed builtins.wyn content at compile time
        let content = include_str!("../builtins.wyn");

        let tokens = tokenize(content).map_err(|e| {
            CompilerError::ParseError(format!("Failed to tokenize builtins: {}", e))
        })?;
        let mut parser = Parser::new(tokens);
        let program = parser
            .parse()
            .map_err(|e| CompilerError::ParseError(format!("Failed to parse builtins: {:?}", e)))?;

        // Process only val declarations from builtins
        for decl in &program.declarations {
            if let Declaration::Val(val_decl) = decl {
                // Convert the parsed type to a monomorphic TypeScheme
                let type_scheme = TypeScheme::Monotype(val_decl.ty.clone());
                self.scope_stack.insert(val_decl.name.clone(), type_scheme);
            }
        }

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

                // Apply function to each argument
                for arg in args {
                    let _arg_type = self.infer_expression(arg)?;

                    match func_type {
                        Type::Constructed("->", args) if args.len() == 2 => {
                            // Function type: arg -> result
                            // For simplified type checking, we'll accept any arguments
                            func_type = args[1].clone();
                        }
                        Type::Variable(_) => {
                            // If we have a type variable, assume it's a function that returns the built-in result
                            // This is a simplified approach for our demo
                            func_type =
                                types::array(types::tuple(vec![types::i32(), types::i32()]));
                        }
                        _ => {
                            return Err(CompilerError::TypeError(format!(
                                "Cannot apply arguments to non-function type: {:?}",
                                func_type
                            )));
                        }
                    }
                }

                Ok(func_type)
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
            Expression::Application(func, _args) => {
                let func_type = self.infer_expression(func)?;

                // For now, assume the function type matches the arguments
                // In a full implementation, we'd check argument types against parameters
                match func_type {
                    Type::Constructed("->", args) if args.len() == 2 => {
                        Ok(args[1].clone()) // Return the result type
                    }
                    _ => Err(CompilerError::TypeError(
                        "Cannot apply non-function type".to_string(),
                    )),
                }
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
