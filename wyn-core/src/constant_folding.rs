use crate::ast::*;
use crate::error::{CompilerError, Result};

/// Constant folder that performs compile-time evaluation of constant expressions
pub struct ConstantFolder {
    node_counter: NodeCounter,
}

impl Default for ConstantFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantFolder {
    pub fn new() -> Self {
        ConstantFolder {
            node_counter: NodeCounter::new(),
        }
    }

    pub fn new_with_counter(node_counter: NodeCounter) -> Self {
        ConstantFolder { node_counter }
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
                }))
            }
            Declaration::Entry(e) => {
                let folded_body = self.fold_expression(&e.body)?;
                Ok(Declaration::Entry(EntryDecl {
                    entry_type: e.entry_type.clone(),
                    name: e.name.clone(),
                    params: e.params.clone(),
                    return_types: e.return_types.clone(),
                    return_attributes: e.return_attributes.clone(),
                    body: folded_body,
                }))
            }
            Declaration::Uniform(u) => {
                // Uniform declarations don't have expressions to fold
                Ok(Declaration::Uniform(u.clone()))
            }
            Declaration::Val(v) => {
                // Val declarations don't have expressions to fold
                Ok(Declaration::Val(v.clone()))
            }
            Declaration::TypeBind(_) => {
                unimplemented!("Type bindings are not yet supported in constant folding")
            }
            Declaration::ModuleBind(_) => {
                unimplemented!("Module bindings are not yet supported in constant folding")
            }
            Declaration::ModuleTypeBind(_) => {
                unimplemented!("Module type bindings are not yet supported in constant folding")
            }
            Declaration::Open(_) => {
                unimplemented!("Open declarations are not yet supported in constant folding")
            }
            Declaration::Import(_) => {
                unimplemented!("Import declarations are not yet supported in constant folding")
            }
            Declaration::Local(_) => {
                unimplemented!("Local declarations are not yet supported in constant folding")
            }
        }
    }

    /// Fold constants in an expression (recursive)
    fn fold_expression(&mut self, expr: &Expression) -> Result<Expression> {
        let span = expr.h.span;
        match &expr.kind {
            // Handle binary operations - this is where the magic happens
            ExprKind::BinaryOp(op, left, right) => {
                let folded_left = self.fold_expression(left)?;
                let folded_right = self.fold_expression(right)?;

                // Try to evaluate if both sides are literals
                match (&folded_left.kind, &folded_right.kind) {
                    (ExprKind::FloatLiteral(l), ExprKind::FloatLiteral(r)) => {
                        match op.op.as_str() {
                            "+" => Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(l + r), span)),
                            "-" => Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(l - r), span)),
                            "*" => Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(l * r), span)),
                            "/" => {
                                if *r == 0.0 {
                                    Err(CompilerError::TypeError(
                                        "Division by zero in constant expression".to_string(),
                                    ))
                                } else {
                                    Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(l / r), span))
                                }
                            }
                            _ => {
                                // Non-arithmetic operations, keep as binary op
                                Ok(self.node_counter.mk_node(
                                    ExprKind::BinaryOp(
                                        op.clone(),
                                        Box::new(folded_left),
                                        Box::new(folded_right),
                                    ),
                                    span,
                                ))
                            }
                        }
                    }
                    (ExprKind::IntLiteral(l), ExprKind::IntLiteral(r)) => {
                        match op.op.as_str() {
                            "+" => Ok(self.node_counter.mk_node(ExprKind::IntLiteral(l + r), span)),
                            "-" => Ok(self.node_counter.mk_node(ExprKind::IntLiteral(l - r), span)),
                            "*" => Ok(self.node_counter.mk_node(ExprKind::IntLiteral(l * r), span)),
                            "/" => {
                                if *r == 0 {
                                    Err(CompilerError::TypeError(
                                        "Division by zero in constant expression".to_string(),
                                    ))
                                } else {
                                    Ok(self.node_counter.mk_node(ExprKind::IntLiteral(l / r), span))
                                }
                            }
                            _ => {
                                // Non-arithmetic operations, keep as binary op
                                Ok(self.node_counter.mk_node(
                                    ExprKind::BinaryOp(
                                        op.clone(),
                                        Box::new(folded_left),
                                        Box::new(folded_right),
                                    ),
                                    span,
                                ))
                            }
                        }
                    }
                    _ => {
                        // Can't fold, but use folded children
                        Ok(self.node_counter.mk_node(
                            ExprKind::BinaryOp(op.clone(), Box::new(folded_left), Box::new(folded_right)),
                            span,
                        ))
                    }
                }
            }

            // Handle array literals - fold each element
            ExprKind::ArrayLiteral(elements) => {
                let mut folded_elements = Vec::new();
                for elem in elements {
                    folded_elements.push(self.fold_expression(elem)?);
                }
                Ok(self.node_counter.mk_node(ExprKind::ArrayLiteral(folded_elements), span))
            }

            // Handle tuples - fold each element
            ExprKind::Tuple(elements) => {
                let mut folded_elements = Vec::new();
                for elem in elements {
                    folded_elements.push(self.fold_expression(elem)?);
                }
                Ok(self.node_counter.mk_node(ExprKind::Tuple(folded_elements), span))
            }

            // Handle function calls - fold arguments
            ExprKind::FunctionCall(name, args) => {
                let mut folded_args = Vec::new();
                for arg in args {
                    folded_args.push(self.fold_expression(arg)?);
                }
                Ok(self.node_counter.mk_node(ExprKind::FunctionCall(name.clone(), folded_args), span))
            }

            // Handle if-then-else - fold all branches
            ExprKind::If(if_expr) => {
                let folded_condition = self.fold_expression(&if_expr.condition)?;
                let folded_then = self.fold_expression(&if_expr.then_branch)?;
                let folded_else = self.fold_expression(&if_expr.else_branch)?;

                Ok(self.node_counter.mk_node(
                    ExprKind::If(IfExpr {
                        condition: Box::new(folded_condition),
                        then_branch: Box::new(folded_then),
                        else_branch: Box::new(folded_else),
                    }),
                    span,
                ))
            }

            // Handle let-in expressions
            ExprKind::LetIn(let_in) => {
                let folded_value = self.fold_expression(&let_in.value)?;
                let folded_body = self.fold_expression(&let_in.body)?;

                Ok(self.node_counter.mk_node(
                    ExprKind::LetIn(LetInExpr {
                        name: let_in.name.clone(),
                        ty: let_in.ty.clone(),
                        value: Box::new(folded_value),
                        body: Box::new(folded_body),
                    }),
                    span,
                ))
            }

            // Handle lambdas - fold body
            ExprKind::Lambda(lambda) => {
                let folded_body = self.fold_expression(&lambda.body)?;

                Ok(self.node_counter.mk_node(
                    ExprKind::Lambda(LambdaExpr {
                        params: lambda.params.clone(),
                        return_type: lambda.return_type.clone(),
                        body: Box::new(folded_body),
                    }),
                    span,
                ))
            }

            // Handle application - fold function and arguments
            ExprKind::Application(func, args) => {
                let folded_func = self.fold_expression(func)?;
                let mut folded_args = Vec::new();
                for arg in args {
                    folded_args.push(self.fold_expression(arg)?);
                }
                Ok(self
                    .node_counter
                    .mk_node(ExprKind::Application(Box::new(folded_func), folded_args), span))
            }

            // Handle array indexing - fold array and index
            ExprKind::ArrayIndex(array, index) => {
                let folded_array = self.fold_expression(array)?;
                let folded_index = self.fold_expression(index)?;
                Ok(self.node_counter.mk_node(
                    ExprKind::ArrayIndex(Box::new(folded_array), Box::new(folded_index)),
                    span,
                ))
            }

            // Handle field access - fold the object
            ExprKind::FieldAccess(obj, field) => {
                let folded_obj = self.fold_expression(obj)?;
                Ok(self
                    .node_counter
                    .mk_node(ExprKind::FieldAccess(Box::new(folded_obj), field.clone()), span))
            }

            // Literals and identifiers don't need folding
            ExprKind::Identifier(_)
            | ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::TypeHole => Ok(expr.clone()),

            ExprKind::Pipe(left, right) => {
                let folded_left = self.fold_expression(left)?;
                let folded_right = self.fold_expression(right)?;
                Ok(self.node_counter.mk_node(
                    ExprKind::Pipe(Box::new(folded_left), Box::new(folded_right)),
                    span,
                ))
            }

            ExprKind::Loop(loop_expr) => {
                // Fold the init expression if present
                let folded_init = if let Some(init) = &loop_expr.init {
                    Some(Box::new(self.fold_expression(init)?))
                } else {
                    None
                };

                // Fold the loop form's expression(s)
                let folded_form = match &loop_expr.form {
                    LoopForm::For(name, expr) => {
                        LoopForm::For(name.clone(), Box::new(self.fold_expression(expr)?))
                    }
                    LoopForm::ForIn(pattern, expr) => {
                        LoopForm::ForIn(pattern.clone(), Box::new(self.fold_expression(expr)?))
                    }
                    LoopForm::While(expr) => LoopForm::While(Box::new(self.fold_expression(expr)?)),
                };

                // Fold the loop body
                let folded_body = Box::new(self.fold_expression(&loop_expr.body)?);

                Ok(self.node_counter.mk_node(
                    ExprKind::Loop(LoopExpr {
                        pattern: loop_expr.pattern.clone(),
                        init: folded_init,
                        form: folded_form,
                        body: folded_body,
                    }),
                    span,
                ))
            }

            ExprKind::QualifiedName(_, _) => {
                // QualifiedName doesn't have subexpressions to fold
                Ok(expr.clone())
            }

            ExprKind::UnaryOp(op, operand) => {
                let folded_operand = self.fold_expression(operand)?;

                // Try to evaluate if operand is a literal
                match (op.op.as_str(), &folded_operand.kind) {
                    ("-", ExprKind::IntLiteral(val)) => {
                        Ok(self.node_counter.mk_node(ExprKind::IntLiteral(-val), span))
                    }
                    ("-", ExprKind::FloatLiteral(val)) => {
                        Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(-val), span))
                    }
                    ("!", ExprKind::BoolLiteral(val)) => {
                        Ok(self.node_counter.mk_node(ExprKind::BoolLiteral(!val), span))
                    }
                    _ => {
                        // Can't fold, but use folded operand
                        Ok(self
                            .node_counter
                            .mk_node(ExprKind::UnaryOp(op.clone(), Box::new(folded_operand)), span))
                    }
                }
            }

            ExprKind::Match(_)
            | ExprKind::Range(_)
            | ExprKind::TypeAscription(_, _)
            | ExprKind::TypeCoercion(_, _)
            | ExprKind::Unsafe(_)
            | ExprKind::Assert(_, _) => {
                todo!(
                    "New expression kinds not yet implemented in constant folding: {:?}",
                    expr.kind
                )
            }
        } // NEWCASESHERE - add new cases before this closing brace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_division_folding() {
        let mut folder = ConstantFolder::new();
        let mut counter = NodeCounter::new();

        // Test 135f32 / 255f32
        let left = counter.mk_node_dummy(ExprKind::FloatLiteral(135.0));
        let right = counter.mk_node_dummy(ExprKind::FloatLiteral(255.0));
        let expr = counter.mk_node_dummy(ExprKind::BinaryOp(
            BinaryOp { op: "/".to_string() },
            Box::new(left),
            Box::new(right),
        ));

        let result = folder.fold_expression(&expr).unwrap();

        match result.kind {
            ExprKind::FloatLiteral(val) => {
                // Should be approximately 0.529411765
                assert!((val - 0.529411765).abs() < 0.000001);
            }
            _ => panic!("Expected folded float literal, got {:?}", result),
        }
    }

    #[test]
    fn test_array_with_division_folding() {
        let mut folder = ConstantFolder::new();
        let mut counter = NodeCounter::new();

        // Test [135f32/255f32, 206f32/255f32, 1.0f32]
        let left1 = counter.mk_node_dummy(ExprKind::FloatLiteral(135.0));
        let right1 = counter.mk_node_dummy(ExprKind::FloatLiteral(255.0));
        let elem1 = counter.mk_node_dummy(ExprKind::BinaryOp(
            BinaryOp { op: "/".to_string() },
            Box::new(left1),
            Box::new(right1),
        ));

        let left2 = counter.mk_node_dummy(ExprKind::FloatLiteral(206.0));
        let right2 = counter.mk_node_dummy(ExprKind::FloatLiteral(255.0));
        let elem2 = counter.mk_node_dummy(ExprKind::BinaryOp(
            BinaryOp { op: "/".to_string() },
            Box::new(left2),
            Box::new(right2),
        ));
        let elem3 = counter.mk_node_dummy(ExprKind::FloatLiteral(1.0));

        let expr = counter.mk_node_dummy(ExprKind::ArrayLiteral(vec![elem1, elem2, elem3]));

        let result = folder.fold_expression(&expr).unwrap();

        match result.kind {
            ExprKind::ArrayLiteral(elements) => {
                assert_eq!(elements.len(), 3);

                // First element should be folded
                match &elements[0].kind {
                    ExprKind::FloatLiteral(val) => {
                        assert!((val - 0.529411765).abs() < 0.000001);
                    }
                    _ => panic!("Expected folded float literal"),
                }

                // Second element should be folded
                match &elements[1].kind {
                    ExprKind::FloatLiteral(val) => {
                        assert!((val - 0.807843137).abs() < 0.000001);
                    }
                    _ => panic!("Expected folded float literal"),
                }

                // Third element should remain unchanged
                match &elements[2].kind {
                    ExprKind::FloatLiteral(val) => {
                        assert_eq!(*val, 1.0);
                    }
                    _ => panic!("Expected unchanged float literal"),
                }
            }
            _ => panic!("Expected array literal, got {:?}", result),
        }
    }
}
