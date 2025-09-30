use crate::ast::*;
use crate::error::{CompilerError, Result};

/// Constant folder that performs compile-time evaluation of constant expressions
pub struct ConstantFolder {}

impl Default for ConstantFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantFolder {
    pub fn new() -> Self {
        ConstantFolder {}
    }

    /// Fold constants in an entire program
    pub fn fold_program(&mut self, program: &Program) -> Result<Program> {
        let mut folded_declarations = Vec::new();

        for decl in &program.declarations {
            let folded_decl = self.fold_declaration(decl)?;
            folded_declarations.push(folded_decl);
        }

        Ok(Program {
            declarations: folded_declarations,
        })
    }

    /// Fold constants in a declaration
    fn fold_declaration(&mut self, decl: &Declaration) -> Result<Declaration> {
        match decl {
            Declaration::Decl(d) => {
                let folded_body = self.fold_expression(&d.body)?;
                Ok(Declaration::Decl(Decl {
                    keyword: d.keyword,
                    name: d.name.clone(),
                    params: d.params.clone(), // Parameters don't need folding
                    ty: d.ty.clone(),
                    body: folded_body,
                    attributes: d.attributes.clone(),
                    return_attributes: d.return_attributes.clone(),
                    attributed_return_type: d.attributed_return_type.clone(),
                }))
            }
            Declaration::Val(v) => {
                // Val declarations don't have expressions to fold
                Ok(Declaration::Val(v.clone()))
            }
        }
    }

    /// Fold constants in an expression (recursive)
    fn fold_expression(&mut self, expr: &Expression) -> Result<Expression> {
        match expr {
            // Handle binary operations - this is where the magic happens
            Expression::BinaryOp(op, left, right) => {
                let folded_left = self.fold_expression(left)?;
                let folded_right = self.fold_expression(right)?;

                // Try to evaluate if both sides are literals
                match (&folded_left, &folded_right) {
                    (Expression::FloatLiteral(l), Expression::FloatLiteral(r)) => {
                        match op.op.as_str() {
                            "+" => Ok(Expression::FloatLiteral(l + r)),
                            "-" => Ok(Expression::FloatLiteral(l - r)),
                            "*" => Ok(Expression::FloatLiteral(l * r)),
                            "/" => {
                                if *r == 0.0 {
                                    Err(CompilerError::TypeError(
                                        "Division by zero in constant expression".to_string(),
                                    ))
                                } else {
                                    Ok(Expression::FloatLiteral(l / r))
                                }
                            }
                            _ => {
                                // Non-arithmetic operations, keep as binary op
                                Ok(Expression::BinaryOp(
                                    op.clone(),
                                    Box::new(folded_left),
                                    Box::new(folded_right),
                                ))
                            }
                        }
                    }
                    (Expression::IntLiteral(l), Expression::IntLiteral(r)) => {
                        match op.op.as_str() {
                            "+" => Ok(Expression::IntLiteral(l + r)),
                            "-" => Ok(Expression::IntLiteral(l - r)),
                            "*" => Ok(Expression::IntLiteral(l * r)),
                            "/" => {
                                if *r == 0 {
                                    Err(CompilerError::TypeError(
                                        "Division by zero in constant expression".to_string(),
                                    ))
                                } else {
                                    Ok(Expression::IntLiteral(l / r))
                                }
                            }
                            _ => {
                                // Non-arithmetic operations, keep as binary op
                                Ok(Expression::BinaryOp(
                                    op.clone(),
                                    Box::new(folded_left),
                                    Box::new(folded_right),
                                ))
                            }
                        }
                    }
                    _ => {
                        // Can't fold, but use folded children
                        Ok(Expression::BinaryOp(
                            op.clone(),
                            Box::new(folded_left),
                            Box::new(folded_right),
                        ))
                    }
                }
            }

            // Handle array literals - fold each element
            Expression::ArrayLiteral(elements) => {
                let mut folded_elements = Vec::new();
                for elem in elements {
                    folded_elements.push(self.fold_expression(elem)?);
                }
                Ok(Expression::ArrayLiteral(folded_elements))
            }

            // Handle tuples - fold each element
            Expression::Tuple(elements) => {
                let mut folded_elements = Vec::new();
                for elem in elements {
                    folded_elements.push(self.fold_expression(elem)?);
                }
                Ok(Expression::Tuple(folded_elements))
            }

            // Handle function calls - fold arguments
            Expression::FunctionCall(name, args) => {
                let mut folded_args = Vec::new();
                for arg in args {
                    folded_args.push(self.fold_expression(arg)?);
                }
                Ok(Expression::FunctionCall(name.clone(), folded_args))
            }

            // Handle if-then-else - fold all branches
            Expression::If(if_expr) => {
                let folded_condition = self.fold_expression(&if_expr.condition)?;
                let folded_then = self.fold_expression(&if_expr.then_branch)?;
                let folded_else = self.fold_expression(&if_expr.else_branch)?;

                Ok(Expression::If(IfExpr {
                    condition: Box::new(folded_condition),
                    then_branch: Box::new(folded_then),
                    else_branch: Box::new(folded_else),
                }))
            }

            // Handle let-in expressions
            Expression::LetIn(let_in) => {
                let folded_value = self.fold_expression(&let_in.value)?;
                let folded_body = self.fold_expression(&let_in.body)?;

                Ok(Expression::LetIn(LetInExpr {
                    name: let_in.name.clone(),
                    ty: let_in.ty.clone(),
                    value: Box::new(folded_value),
                    body: Box::new(folded_body),
                }))
            }

            // Handle lambdas - fold body
            Expression::Lambda(lambda) => {
                let folded_body = self.fold_expression(&lambda.body)?;

                Ok(Expression::Lambda(LambdaExpr {
                    params: lambda.params.clone(),
                    return_type: lambda.return_type.clone(),
                    body: Box::new(folded_body),
                }))
            }

            // Handle application - fold function and arguments
            Expression::Application(func, args) => {
                let folded_func = self.fold_expression(func)?;
                let mut folded_args = Vec::new();
                for arg in args {
                    folded_args.push(self.fold_expression(arg)?);
                }
                Ok(Expression::Application(Box::new(folded_func), folded_args))
            }

            // Handle array indexing - fold array and index
            Expression::ArrayIndex(array, index) => {
                let folded_array = self.fold_expression(array)?;
                let folded_index = self.fold_expression(index)?;
                Ok(Expression::ArrayIndex(
                    Box::new(folded_array),
                    Box::new(folded_index),
                ))
            }

            // Handle field access - fold the object
            Expression::FieldAccess(obj, field) => {
                let folded_obj = self.fold_expression(obj)?;
                Ok(Expression::FieldAccess(Box::new(folded_obj), field.clone()))
            }

            // Literals and identifiers don't need folding
            Expression::Identifier(_) | Expression::IntLiteral(_) | Expression::FloatLiteral(_) => {
                Ok(expr.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_division_folding() {
        let mut folder = ConstantFolder::new();

        // Test 135f32 / 255f32
        let expr = Expression::BinaryOp(
            BinaryOp { op: "/".to_string() },
            Box::new(Expression::FloatLiteral(135.0)),
            Box::new(Expression::FloatLiteral(255.0)),
        );

        let result = folder.fold_expression(&expr).unwrap();

        match result {
            Expression::FloatLiteral(val) => {
                // Should be approximately 0.529411765
                assert!((val - 0.529411765).abs() < 0.000001);
            }
            _ => panic!("Expected folded float literal, got {:?}", result),
        }
    }

    #[test]
    fn test_array_with_division_folding() {
        let mut folder = ConstantFolder::new();

        // Test [135f32/255f32, 206f32/255f32, 1.0f32]
        let expr = Expression::ArrayLiteral(vec![
            Expression::BinaryOp(
                BinaryOp { op: "/".to_string() },
                Box::new(Expression::FloatLiteral(135.0)),
                Box::new(Expression::FloatLiteral(255.0)),
            ),
            Expression::BinaryOp(
                BinaryOp { op: "/".to_string() },
                Box::new(Expression::FloatLiteral(206.0)),
                Box::new(Expression::FloatLiteral(255.0)),
            ),
            Expression::FloatLiteral(1.0),
        ]);

        let result = folder.fold_expression(&expr).unwrap();

        match result {
            Expression::ArrayLiteral(elements) => {
                assert_eq!(elements.len(), 3);

                // First element should be folded
                match &elements[0] {
                    Expression::FloatLiteral(val) => {
                        assert!((val - 0.529411765).abs() < 0.000001);
                    }
                    _ => panic!("Expected folded float literal"),
                }

                // Second element should be folded
                match &elements[1] {
                    Expression::FloatLiteral(val) => {
                        assert!((val - 0.807843137).abs() < 0.000001);
                    }
                    _ => panic!("Expected folded float literal"),
                }

                // Third element should remain unchanged
                match &elements[2] {
                    Expression::FloatLiteral(val) => {
                        assert_eq!(*val, 1.0);
                    }
                    _ => panic!("Expected unchanged float literal"),
                }
            }
            _ => panic!("Expected array literal, got {:?}", result),
        }
    }
}
