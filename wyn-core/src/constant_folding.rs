//! Constant folding pass for MIR.
//!
//! This pass evaluates constant expressions at compile time, reducing
//! operations on literals to their computed values.

use crate::error::{CompilerError, Result};
use crate::mir::folder::MirFolder;
use crate::mir::{Expr, ExprKind, Literal, Program};
use crate::{bail_type_at, err_type_at};

/// Constant folder that performs compile-time evaluation of constant expressions.
pub struct ConstantFolder;

impl Default for ConstantFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl MirFolder for ConstantFolder {
    type Error = CompilerError;
    type Ctx = ();

    fn visit_expr_bin_op(
        &mut self,
        op: String,
        lhs: Expr,
        rhs: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> std::result::Result<Expr, Self::Error> {
        // First, recursively fold the operands
        let folded_lhs = self.visit_expr(lhs, ctx)?;
        let folded_rhs = self.visit_expr(rhs, ctx)?;

        // Try to fold if both are literals
        if let Some(folded) = self.try_fold_binop(&op, &folded_lhs, &folded_rhs, &expr, expr.span)? {
            return Ok(folded);
        }

        // Can't fold, return with folded children
        Ok(Expr {
            kind: ExprKind::BinOp {
                op,
                lhs: Box::new(folded_lhs),
                rhs: Box::new(folded_rhs),
            },
            ..expr
        })
    }

    fn visit_expr_unary_op(
        &mut self,
        op: String,
        operand: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> std::result::Result<Expr, Self::Error> {
        // First, recursively fold the operand
        let folded_operand = self.visit_expr(operand, ctx)?;

        // Try to fold if operand is a literal
        if let Some(folded) = self.try_fold_unaryop(&op, &folded_operand, &expr, expr.span)? {
            return Ok(folded);
        }

        // Can't fold, return with folded child
        Ok(Expr {
            kind: ExprKind::UnaryOp {
                op,
                operand: Box::new(folded_operand),
            },
            ..expr
        })
    }

    fn visit_expr_if(
        &mut self,
        cond: Expr,
        then_branch: Expr,
        else_branch: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> std::result::Result<Expr, Self::Error> {
        // First, recursively fold all branches
        let folded_cond = self.visit_expr(cond, ctx)?;
        let folded_then = self.visit_expr(then_branch, ctx)?;
        let folded_else = self.visit_expr(else_branch, ctx)?;

        // If condition is a constant bool, return the appropriate branch
        if let ExprKind::Literal(Literal::Bool(b)) = &folded_cond.kind {
            return Ok(if *b { folded_then } else { folded_else });
        }

        // Can't fold, return with folded children
        Ok(Expr {
            kind: ExprKind::If {
                cond: Box::new(folded_cond),
                then_branch: Box::new(folded_then),
                else_branch: Box::new(folded_else),
            },
            ..expr
        })
    }
}

impl ConstantFolder {
    pub fn new() -> Self {
        ConstantFolder
    }

    /// Convenience wrapper for tests - folds an expression by cloning it
    pub fn fold_expr(&mut self, expr: &Expr) -> Result<Expr> {
        self.visit_expr(expr.clone(), &mut ())
    }

    /// Try to fold a binary operation on two literals.
    /// Uses the original expr's id for the folded result.
    fn try_fold_binop(
        &self,
        op: &str,
        lhs: &Expr,
        rhs: &Expr,
        orig_expr: &Expr,
        span: crate::ast::Span,
    ) -> Result<Option<Expr>> {
        match (&lhs.kind, &rhs.kind) {
            // Float operations
            (ExprKind::Literal(Literal::Float(l)), ExprKind::Literal(Literal::Float(r))) => {
                let l: f64 = l.parse().map_err(|_| err_type_at!(span, "Invalid float literal"))?;
                let r: f64 = r.parse().map_err(|_| err_type_at!(span, "Invalid float literal"))?;

                let result = match op {
                    "+" => Some(l + r),
                    "-" => Some(l - r),
                    "*" => Some(l * r),
                    "/" => {
                        if r == 0.0 {
                            bail_type_at!(span, "Division by zero in constant expression");
                        }
                        Some(l / r)
                    }
                    _ => None,
                };

                if let Some(val) = result {
                    return Ok(Some(Expr::new(
                        orig_expr.id,
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
                        orig_expr.id,
                        orig_expr.ty.clone(),
                        ExprKind::Literal(Literal::Bool(val)),
                        span,
                    )));
                }
            }

            // Integer operations
            (ExprKind::Literal(Literal::Int(l)), ExprKind::Literal(Literal::Int(r))) => {
                let l: i64 = l.parse().map_err(|_| err_type_at!(span, "Invalid integer literal"))?;
                let r: i64 = r.parse().map_err(|_| err_type_at!(span, "Invalid integer literal"))?;

                let result = match op {
                    "+" => Some(l + r),
                    "-" => Some(l - r),
                    "*" => Some(l * r),
                    "/" => {
                        if r == 0 {
                            bail_type_at!(span, "Division by zero in constant expression");
                        }
                        Some(l / r)
                    }
                    "%" => {
                        if r == 0 {
                            bail_type_at!(span, "Modulo by zero in constant expression");
                        }
                        Some(l % r)
                    }
                    _ => None,
                };

                if let Some(val) = result {
                    return Ok(Some(Expr::new(
                        orig_expr.id,
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
                        orig_expr.id,
                        orig_expr.ty.clone(),
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
                        orig_expr.id,
                        orig_expr.ty.clone(),
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
    /// Uses the original expr's id for the folded result.
    fn try_fold_unaryop(
        &self,
        op: &str,
        operand: &Expr,
        orig_expr: &Expr,
        span: crate::ast::Span,
    ) -> Result<Option<Expr>> {
        match (op, &operand.kind) {
            // Negation of float
            ("-", ExprKind::Literal(Literal::Float(val))) => {
                let v: f64 = val.parse().map_err(|_| err_type_at!(span, "Invalid float literal"))?;
                Ok(Some(Expr::new(
                    orig_expr.id,
                    orig_expr.ty.clone(),
                    ExprKind::Literal(Literal::Float((-v).to_string())),
                    span,
                )))
            }

            // Negation of integer
            ("-", ExprKind::Literal(Literal::Int(val))) => {
                let v: i64 = val.parse().map_err(|_| err_type_at!(span, "Invalid integer literal"))?;
                Ok(Some(Expr::new(
                    orig_expr.id,
                    orig_expr.ty.clone(),
                    ExprKind::Literal(Literal::Int((-v).to_string())),
                    span,
                )))
            }

            // Boolean not
            ("!", ExprKind::Literal(Literal::Bool(val))) => Ok(Some(Expr::new(
                orig_expr.id,
                orig_expr.ty.clone(),
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
    folder.visit_program(program, &mut ())
}
