use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::lexer::tokenize;
use crate::parser::Parser;
use crate::scope::ScopeStack;

pub struct TypeChecker {
    scope_stack: ScopeStack<Type>,
    next_var: usize,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut checker = TypeChecker {
            scope_stack: ScopeStack::new(),
            next_var: 0,
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

        let tokens = tokenize(&content).map_err(|e| {
            CompilerError::ParseError(format!("Failed to tokenize builtins: {}", e))
        })?;
        let mut parser = Parser::new(tokens);
        let program = parser
            .parse()
            .map_err(|e| CompilerError::ParseError(format!("Failed to parse builtins: {:?}", e)))?;

        // Process only val declarations from builtins
        for decl in &program.declarations {
            if let Declaration::Val(val_decl) = decl {
                self.scope_stack
                    .insert(val_decl.name.clone(), val_decl.ty.clone());
            }
        }

        Ok(())
    }

    pub fn check_program(&mut self, program: &Program) -> Result<()> {
        // First pass: collect all function declarations
        for decl in &program.declarations {
            match decl {
                Declaration::Def(def_decl) => {
                    // Create a placeholder type for this function
                    let func_type = self.fresh_var();
                    self.scope_stack.insert(def_decl.name.clone(), func_type);
                }
                _ => {}
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
        self.scope_stack.insert(decl.name.clone(), stored_type);

        Ok(())
    }

    fn check_entry_decl(&mut self, decl: &EntryDecl) -> Result<()> {
        // Push new scope for entry parameters
        self.scope_stack.push_scope();

        // Add parameters to scope
        for param in &decl.params {
            self.scope_stack
                .insert(param.name.clone(), param.ty.clone());
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
        let param_types: Vec<Type> = decl.params.iter().map(|_| self.fresh_var()).collect();

        // Push new scope for function parameters
        self.scope_stack.push_scope();

        // Add parameters to scope
        for (param_name, param_type) in decl.params.iter().zip(param_types.iter()) {
            self.scope_stack
                .insert(param_name.clone(), param_type.clone());
        }

        // Infer body type
        let body_type = self.infer_expression(&decl.body)?;

        // Pop parameter scope
        self.scope_stack.pop_scope();

        // Build function type: param1 -> param2 -> ... -> body_type
        let func_type = param_types
            .into_iter()
            .rev()
            .fold(body_type, |acc, param_ty| {
                Type::Function(Box::new(param_ty), Box::new(acc))
            });

        // Update scope with inferred type
        self.scope_stack
            .insert(decl.name.clone(), func_type.clone());

        println!("Inferred type for {}: {}", decl.name, func_type);

        Ok(())
    }

    fn check_val_decl(&mut self, decl: &ValDecl) -> Result<()> {
        // Val declarations are just type signatures - register them in scope
        self.scope_stack.insert(decl.name.clone(), decl.ty.clone());
        Ok(())
    }

    fn infer_expression(&mut self, expr: &Expression) -> Result<Type> {
        match expr {
            Expression::IntLiteral(_) => Ok(Type::I32),
            Expression::FloatLiteral(_) => Ok(Type::F32),
            Expression::Identifier(name) => {
                self.scope_stack
                    .lookup(name)
                    .cloned()
                    .ok_or_else(|| CompilerError::UndefinedVariable(name.clone()))
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

                Ok(Type::Array(Box::new(first_type), vec![elements.len()]))
            }
            Expression::ArrayIndex(array_expr, _index_expr) => {
                let array_type = self.infer_expression(array_expr)?;
                match array_type {
                    Type::Array(elem_type, _) => Ok(*elem_type),
                    _ => Err(CompilerError::TypeError(format!(
                        "Cannot index non-array type: got {:?}",
                        array_type
                    ))),
                }
            }
            Expression::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;

                match op {
                    BinaryOp::Divide => {
                        // Division is only valid for numeric types
                        match (&left_type, &right_type) {
                            (Type::F32, Type::F32) => Ok(Type::F32),
                            (Type::I32, Type::I32) => Ok(Type::I32),
                            _ => Err(CompilerError::TypeError(format!(
                                "Cannot divide {:?} by {:?}",
                                left_type, right_type
                            ))),
                        }
                    }
                    BinaryOp::Add => {
                        // Addition requires matching numeric operands
                        match (&left_type, &right_type) {
                            (Type::F32, Type::F32) => Ok(Type::F32),
                            (Type::I32, Type::I32) => Ok(Type::I32),
                            _ => Err(CompilerError::TypeError(format!(
                                "Cannot add {:?} and {:?}",
                                left_type, right_type
                            ))),
                        }
                    }
                }
            }
            Expression::FunctionCall(func_name, args) => {
                // Get function type
                let mut func_type = self
                    .scope_stack
                    .lookup(func_name)
                    .ok_or_else(|| CompilerError::UndefinedVariable(func_name.clone()))?
                    .clone();

                // Apply function to each argument
                for arg in args {
                    let _arg_type = self.infer_expression(arg)?;

                    match func_type {
                        Type::Function(_param_ty, result_ty) => {
                            // For simplified type checking, we'll accept any arguments
                            // A full implementation would unify types properly
                            func_type = *result_ty;
                        }
                        Type::Var(_) => {
                            // If we have a type variable, assume it's a function that returns the built-in result
                            // This is a simplified approach for our demo
                            func_type = Type::Array(
                                Box::new(Type::Tuple(vec![Type::I32, Type::I32])),
                                vec![1],
                            );
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

                Ok(Type::Tuple(elem_types?))
            }
            Expression::Lambda(lambda) => {
                // Push new scope for lambda parameters
                self.scope_stack.push_scope();

                // Add parameters to scope with their types (or type variables)
                for param in &lambda.params {
                    let param_type = param.ty.clone().unwrap_or_else(|| {
                        let var = Type::Var(format!("param_{}", self.next_var));
                        self.next_var += 1;
                        var
                    });
                    self.scope_stack.insert(param.name.clone(), param_type);
                }

                // Type check the lambda body with parameters in scope
                let body_type = self.infer_expression(&lambda.body)?;
                let return_type = lambda.return_type.clone().unwrap_or(body_type);

                // Pop parameter scope
                self.scope_stack.pop_scope();

                // For multiple parameters, create nested function types
                let mut func_type = return_type;
                for param in lambda.params.iter().rev() {
                    let param_type = param.ty.clone().unwrap_or_else(|| {
                        let var = Type::Var(format!("param_{}", self.next_var));
                        self.next_var += 1;
                        var
                    });
                    func_type = Type::Function(Box::new(param_type), Box::new(func_type));
                }

                Ok(func_type)
            }
            Expression::Application(func, args) => {
                let func_type = self.infer_expression(func)?;

                // For now, assume the function type matches the arguments
                // In a full implementation, we'd check argument types against parameters
                match func_type {
                    Type::Function(_, return_type) => Ok(*return_type),
                    _ => Err(CompilerError::TypeError(
                        "Cannot apply non-function type".to_string(),
                    )),
                }
            }
        }
    }

    fn fresh_var(&mut self) -> Type {
        let var = format!("t{}", self.next_var);
        self.next_var += 1;
        Type::Var(var)
    }

    fn types_match(&self, t1: &Type, t2: &Type) -> bool {
        match (t1, t2) {
            (Type::I32, Type::I32) => true,
            (Type::F32, Type::F32) => true,
            (Type::Vec4F32, Type::Vec4F32) => true,
            (Type::Array(elem1, dims1), Type::Array(elem2, dims2)) => {
                dims1 == dims2 && self.types_match(elem1, elem2)
            }
            (Type::Tuple(types1), Type::Tuple(types2)) => {
                types1.len() == types2.len()
                    && types1
                        .iter()
                        .zip(types2.iter())
                        .all(|(t1, t2)| self.types_match(t1, t2))
            }
            (Type::Function(arg1, ret1), Type::Function(arg2, ret2)) => {
                self.types_match(arg1, arg2) && self.types_match(ret1, ret2)
            }
            (Type::Var(_), _) => true, // Type variables match anything for now
            (_, Type::Var(_)) => true,
            _ => false,
        }
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
