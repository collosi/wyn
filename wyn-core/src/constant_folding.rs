//! Constant folding pass for MIR.
//!
//! This pass evaluates constant expressions at compile time, reducing
//! operations on literals to their computed values.

use crate::error::{CompilerError, Result};
use crate::mir::{Def, Expr, ExprKind, Literal, LoopKind, Program};

/// Constant folder that performs compile-time evaluation of constant expressions.
pub struct ConstantFolder;

impl Default for ConstantFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantFolder {
    pub fn new() -> Self {
        ConstantFolder
    }

    /// Fold constants in an entire MIR program.
    pub fn fold_program(&mut self, program: &Program) -> Result<Program> {
        let mut folded_defs = Vec::new();

        for def in &program.defs {
            let folded_def = self.fold_def(def)?;
            folded_defs.push(folded_def);
        }

        Ok(Program {
            defs: folded_defs,
            lambda_registry: program.lambda_registry.clone(),
        })
    }

    /// Fold constants in a definition.
    fn fold_def(&mut self, def: &Def) -> Result<Def> {
        match def {
            Def::Function {
                name,
                params,
                ret_type,
                attributes,
                param_attributes,
                return_attributes,
                body,
                span,
            } => {
                let folded_body = self.fold_expr(body)?;
                Ok(Def::Function {
                    name: name.clone(),
                    params: params.clone(),
                    ret_type: ret_type.clone(),
                    attributes: attributes.clone(),
                    param_attributes: param_attributes.clone(),
                    return_attributes: return_attributes.clone(),
                    body: folded_body,
                    span: *span,
                })
            }
            Def::Constant {
                name,
                ty,
                attributes,
                body,
                span,
            } => {
                let folded_body = self.fold_expr(body)?;
                Ok(Def::Constant {
                    name: name.clone(),
                    ty: ty.clone(),
                    attributes: attributes.clone(),
                    body: folded_body,
                    span: *span,
                })
            }
        }
    }

    /// Fold constants in an expression (recursive).
    pub fn fold_expr(&mut self, expr: &Expr) -> Result<Expr> {
        let span = expr.span;
        let ty = expr.ty.clone();

        let kind = match &expr.kind {
            // Binary operations - try to evaluate if both sides are literals
            ExprKind::BinOp { op, lhs, rhs } => {
                let folded_lhs = self.fold_expr(lhs)?;
                let folded_rhs = self.fold_expr(rhs)?;

                // Try to fold if both are literals
                if let Some(folded) = self.try_fold_binop(op, &folded_lhs, &folded_rhs, &ty, span)? {
                    return Ok(folded);
                }

                // Can't fold, return with folded children
                ExprKind::BinOp {
                    op: op.clone(),
                    lhs: Box::new(folded_lhs),
                    rhs: Box::new(folded_rhs),
                }
            }

            // Unary operations - try to evaluate if operand is a literal
            ExprKind::UnaryOp { op, operand } => {
                let folded_operand = self.fold_expr(operand)?;

                // Try to fold if operand is a literal
                if let Some(folded) = self.try_fold_unaryop(op, &folded_operand, &ty, span)? {
                    return Ok(folded);
                }

                ExprKind::UnaryOp {
                    op: op.clone(),
                    operand: Box::new(folded_operand),
                }
            }

            // If expression - fold all branches, and simplify if condition is constant
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let folded_cond = self.fold_expr(cond)?;
                let folded_then = self.fold_expr(then_branch)?;
                let folded_else = self.fold_expr(else_branch)?;

                // If condition is a constant bool, return the appropriate branch
                if let ExprKind::Literal(Literal::Bool(b)) = &folded_cond.kind {
                    return Ok(if *b { folded_then } else { folded_else });
                }

                ExprKind::If {
                    cond: Box::new(folded_cond),
                    then_branch: Box::new(folded_then),
                    else_branch: Box::new(folded_else),
                }
            }

            // Let binding - fold value and body
            ExprKind::Let { name, value, body } => {
                let folded_value = self.fold_expr(value)?;
                let folded_body = self.fold_expr(body)?;

                ExprKind::Let {
                    name: name.clone(),
                    value: Box::new(folded_value),
                    body: Box::new(folded_body),
                }
            }

            // Loop - fold init bindings, loop condition/iter, and body
            ExprKind::Loop {
                init_bindings,
                kind,
                body,
            } => {
                let folded_bindings: Result<Vec<_>> = init_bindings
                    .iter()
                    .map(|(name, expr)| Ok((name.clone(), self.fold_expr(expr)?)))
                    .collect();
                let folded_bindings = folded_bindings?;

                let folded_kind = match kind {
                    LoopKind::For { var, iter } => LoopKind::For {
                        var: var.clone(),
                        iter: Box::new(self.fold_expr(iter)?),
                    },
                    LoopKind::ForRange { var, bound } => LoopKind::ForRange {
                        var: var.clone(),
                        bound: Box::new(self.fold_expr(bound)?),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.fold_expr(cond)?),
                    },
                };

                let folded_body = self.fold_expr(body)?;

                ExprKind::Loop {
                    init_bindings: folded_bindings,
                    kind: folded_kind,
                    body: Box::new(folded_body),
                }
            }

            // Call - fold arguments
            ExprKind::Call { func, args } => {
                let folded_args: Result<Vec<_>> = args.iter().map(|arg| self.fold_expr(arg)).collect();

                ExprKind::Call {
                    func: func.clone(),
                    args: folded_args?,
                }
            }

            // Intrinsic - fold arguments
            ExprKind::Intrinsic { name, args } => {
                let folded_args: Result<Vec<_>> = args.iter().map(|arg| self.fold_expr(arg)).collect();

                ExprKind::Intrinsic {
                    name: name.clone(),
                    args: folded_args?,
                }
            }

            // Attributed - fold inner expression
            ExprKind::Attributed { attributes, expr } => {
                let folded_expr = self.fold_expr(expr)?;

                ExprKind::Attributed {
                    attributes: attributes.clone(),
                    expr: Box::new(folded_expr),
                }
            }

            // Literals - fold nested expressions in compound literals
            ExprKind::Literal(lit) => {
                let folded_lit = self.fold_literal(lit)?;
                ExprKind::Literal(folded_lit)
            }

            // Variables - nothing to fold
            ExprKind::Var(_) => expr.kind.clone(),
        };

        Ok(Expr::new(ty, kind, span))
    }

    /// Fold constants in a literal (for compound literals like arrays, tuples, records).
    fn fold_literal(&mut self, lit: &Literal) -> Result<Literal> {
        match lit {
            Literal::Tuple(exprs) => {
                let folded: Result<Vec<_>> = exprs.iter().map(|e| self.fold_expr(e)).collect();
                Ok(Literal::Tuple(folded?))
            }
            Literal::Array(exprs) => {
                let folded: Result<Vec<_>> = exprs.iter().map(|e| self.fold_expr(e)).collect();
                Ok(Literal::Array(folded?))
            }
            Literal::Record(fields) => {
                let folded: Result<Vec<_>> =
                    fields.iter().map(|(name, expr)| Ok((name.clone(), self.fold_expr(expr)?))).collect();
                Ok(Literal::Record(folded?))
            }
            // Simple literals don't need folding
            Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_) => Ok(lit.clone()),
        }
    }

    /// Try to fold a binary operation on two literals.
    fn try_fold_binop(
        &self,
        op: &str,
        lhs: &Expr,
        rhs: &Expr,
        result_ty: &polytype::Type<crate::ast::TypeName>,
        span: crate::ast::Span,
    ) -> Result<Option<Expr>> {
        match (&lhs.kind, &rhs.kind) {
            // Float operations
            (ExprKind::Literal(Literal::Float(l)), ExprKind::Literal(Literal::Float(r))) => {
                let l: f64 = l
                    .parse()
                    .map_err(|_| CompilerError::TypeError("Invalid float literal".to_string(), span))?;
                let r: f64 = r
                    .parse()
                    .map_err(|_| CompilerError::TypeError("Invalid float literal".to_string(), span))?;

                let result = match op {
                    "+" => Some(l + r),
                    "-" => Some(l - r),
                    "*" => Some(l * r),
                    "/" => {
                        if r == 0.0 {
                            return Err(CompilerError::TypeError(
                                "Division by zero in constant expression".to_string(),
                                span,
                            ));
                        }
                        Some(l / r)
                    }
                    _ => None,
                };

                if let Some(val) = result {
                    return Ok(Some(Expr::new(
                        lhs.ty.clone(),
                        ExprKind::Literal(Literal::Float(val.to_string())),
                        span,
                    )));
                }

                // Boolean comparison operations on floats
                let bool_result = match op {
                    "==" => Some(l == r),
                    "!=" => Some(l != r),
                    "<" => Some(l < r),
                    "<=" => Some(l <= r),
                    ">" => Some(l > r),
                    ">=" => Some(l >= r),
                    _ => None,
                };

                if let Some(val) = bool_result {
                    return Ok(Some(Expr::new(
                        result_ty.clone(),
                        ExprKind::Literal(Literal::Bool(val)),
                        span,
                    )));
                }
            }

            // Integer operations
            (ExprKind::Literal(Literal::Int(l)), ExprKind::Literal(Literal::Int(r))) => {
                let l: i64 = l
                    .parse()
                    .map_err(|_| CompilerError::TypeError("Invalid integer literal".to_string(), span))?;
                let r: i64 = r
                    .parse()
                    .map_err(|_| CompilerError::TypeError("Invalid integer literal".to_string(), span))?;

                let result = match op {
                    "+" => Some(l + r),
                    "-" => Some(l - r),
                    "*" => Some(l * r),
                    "/" => {
                        if r == 0 {
                            return Err(CompilerError::TypeError(
                                "Division by zero in constant expression".to_string(),
                                span,
                            ));
                        }
                        Some(l / r)
                    }
                    "%" => {
                        if r == 0 {
                            return Err(CompilerError::TypeError(
                                "Modulo by zero in constant expression".to_string(),
                                span,
                            ));
                        }
                        Some(l % r)
                    }
                    _ => None,
                };

                if let Some(val) = result {
                    return Ok(Some(Expr::new(
                        lhs.ty.clone(),
                        ExprKind::Literal(Literal::Int(val.to_string())),
                        span,
                    )));
                }

                // Boolean comparison operations on integers
                let bool_result = match op {
                    "==" => Some(l == r),
                    "!=" => Some(l != r),
                    "<" => Some(l < r),
                    "<=" => Some(l <= r),
                    ">" => Some(l > r),
                    ">=" => Some(l >= r),
                    _ => None,
                };

                if let Some(val) = bool_result {
                    return Ok(Some(Expr::new(
                        result_ty.clone(),
                        ExprKind::Literal(Literal::Bool(val)),
                        span,
                    )));
                }
            }

            // Boolean operations
            (ExprKind::Literal(Literal::Bool(l)), ExprKind::Literal(Literal::Bool(r))) => {
                let result = match op {
                    "&&" => Some(*l && *r),
                    "||" => Some(*l || *r),
                    "==" => Some(l == r),
                    "!=" => Some(l != r),
                    _ => None,
                };

                if let Some(val) = result {
                    return Ok(Some(Expr::new(
                        result_ty.clone(),
                        ExprKind::Literal(Literal::Bool(val)),
                        span,
                    )));
                }
            }

            _ => {}
        }

        Ok(None)
    }

    /// Try to fold a unary operation on a literal.
    fn try_fold_unaryop(
        &self,
        op: &str,
        operand: &Expr,
        result_ty: &polytype::Type<crate::ast::TypeName>,
        span: crate::ast::Span,
    ) -> Result<Option<Expr>> {
        match (op, &operand.kind) {
            // Negation of float
            ("-", ExprKind::Literal(Literal::Float(val))) => {
                let v: f64 = val
                    .parse()
                    .map_err(|_| CompilerError::TypeError("Invalid float literal".to_string(), span))?;
                Ok(Some(Expr::new(
                    result_ty.clone(),
                    ExprKind::Literal(Literal::Float((-v).to_string())),
                    span,
                )))
            }

            // Negation of integer
            ("-", ExprKind::Literal(Literal::Int(val))) => {
                let v: i64 = val
                    .parse()
                    .map_err(|_| CompilerError::TypeError("Invalid integer literal".to_string(), span))?;
                Ok(Some(Expr::new(
                    result_ty.clone(),
                    ExprKind::Literal(Literal::Int((-v).to_string())),
                    span,
                )))
            }

            // Boolean not
            ("!", ExprKind::Literal(Literal::Bool(val))) => Ok(Some(Expr::new(
                result_ty.clone(),
                ExprKind::Literal(Literal::Bool(!val)),
                span,
            ))),

            _ => Ok(None),
        }
    }
}

/// Fold constants in a MIR program (convenience function).
pub fn fold_constants(program: Program) -> Result<Program> {
    let mut folder = ConstantFolder::new();
    folder.fold_program(&program)
}
