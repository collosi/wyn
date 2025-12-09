//! Materialize hoisting pass for MIR.
//!
//! This pass finds duplicate `Materialize` expressions within the same scope
//! (particularly in both branches of an if-then-else) and hoists them to a
//! single let binding, eliminating redundant OpStore/OpLoad in SPIR-V.
//!
//! Example transformation:
//! ```text
//! if c then @index(@materialize(f x), i) else @index(@materialize(f x), i)
//! ```
//! becomes:
//! ```text
//! let __mat_0 = @materialize(f x) in
//! if c then @index(__mat_0, i) else @index(__mat_0, i)
//! ```

use crate::ast::TypeName;
use crate::mir::{Def, Expr, ExprKind, Program};
use polytype::Type;
use std::sync::atomic::{AtomicU64, Ordering};

// Global counter for unique binding IDs
static NEXT_BINDING_ID: AtomicU64 = AtomicU64::new(1_000_000);

fn fresh_binding_id() -> u64 {
    NEXT_BINDING_ID.fetch_add(1, Ordering::SeqCst)
}

/// Hoist duplicate materializations in a program.
pub fn hoist_materializations(program: Program) -> Program {
    Program {
        defs: program.defs.into_iter().map(hoist_in_def).collect(),
        lambda_registry: program.lambda_registry,
        local_tables: program.local_tables,
    }
}

fn hoist_in_def(def: Def) -> Def {
    match def {
        Def::Function {
            id,
            name,
            params,
            ret_type,
            attributes,
            body,
            span,
        } => {
            let body = hoist_in_expr(body);
            Def::Function {
                id,
                name,
                params,
                ret_type,
                attributes,
                body,
                span,
            }
        }
        Def::Constant {
            id,
            name,
            ty,
            attributes,
            body,
            span,
        } => {
            let body = hoist_in_expr(body);
            Def::Constant {
                id,
                name,
                ty,
                attributes,
                body,
                span,
            }
        }
        Def::EntryPoint {
            id,
            name,
            execution_model,
            inputs,
            outputs,
            body,
            span,
        } => {
            let body = hoist_in_expr(body);
            Def::EntryPoint {
                id,
                name,
                execution_model,
                inputs,
                outputs,
                body,
                span,
            }
        }
        other => other,
    }
}

/// Recursively process an expression, hoisting materializations from if branches.
fn hoist_in_expr(expr: Expr) -> Expr {
    let Expr { id, ty, kind, span } = expr;

    let kind = match kind {
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            // First recurse into children
            let cond = Box::new(hoist_in_expr(*cond));
            let then_branch = hoist_in_expr(*then_branch);
            let else_branch = hoist_in_expr(*else_branch);

            // Collect materializations from both branches
            let then_mats = collect_materializations(&then_branch);
            let else_mats = collect_materializations(&else_branch);

            // Find common materializations (by structural equality of inner expr)
            let common = find_common_materializations(&then_mats, &else_mats);

            if common.is_empty() {
                // No common materializations, just return the if
                ExprKind::If {
                    cond,
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                }
            } else {
                // Hoist common materializations
                let mut result_then = then_branch;
                let mut result_else = else_branch;

                // For each common materialization, create a let binding and replace occurrences
                let mut hoisted_bindings: Vec<(String, u64, Expr)> = Vec::new();

                for (inner_expr, mat_ty) in common {
                    let binding_id = fresh_binding_id();
                    let var_name = format!("_w_mat_{}", binding_id);

                    // Replace occurrences in both branches
                    result_then = replace_materialize(&result_then, &inner_expr, &var_name, &mat_ty);
                    result_else = replace_materialize(&result_else, &inner_expr, &var_name, &mat_ty);

                    // Create the materialization expression for the let binding
                    let mat_expr = Expr {
                        id,
                        ty: mat_ty.clone(),
                        kind: ExprKind::Materialize(Box::new(inner_expr)),
                        span,
                    };

                    hoisted_bindings.push((var_name, binding_id, mat_expr));
                }

                // Build the if expression
                let if_expr = Expr {
                    id,
                    ty: ty.clone(),
                    kind: ExprKind::If {
                        cond,
                        then_branch: Box::new(result_then),
                        else_branch: Box::new(result_else),
                    },
                    span,
                };

                // Wrap in let bindings (innermost first, so we build from inside out)
                let mut result = if_expr;
                for (var_name, binding_id, mat_expr) in hoisted_bindings.into_iter().rev() {
                    result = Expr {
                        id,
                        ty: result.ty.clone(),
                        kind: ExprKind::Let {
                            name: var_name,
                            binding_id,
                            value: Box::new(mat_expr),
                            body: Box::new(result),
                        },
                        span,
                    };
                }

                return result;
            }
        }

        ExprKind::Let {
            name,
            binding_id,
            value,
            body,
        } => {
            let value = Box::new(hoist_in_expr(*value));
            let body = Box::new(hoist_in_expr(*body));
            ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            }
        }

        ExprKind::BinOp { op, lhs, rhs } => {
            let lhs = Box::new(hoist_in_expr(*lhs));
            let rhs = Box::new(hoist_in_expr(*rhs));
            ExprKind::BinOp { op, lhs, rhs }
        }

        ExprKind::UnaryOp { op, operand } => {
            let operand = Box::new(hoist_in_expr(*operand));
            ExprKind::UnaryOp { op, operand }
        }

        ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let init = Box::new(hoist_in_expr(*init));
            let body = Box::new(hoist_in_expr(*body));
            let init_bindings =
                init_bindings.into_iter().map(|(name, expr)| (name, hoist_in_expr(expr))).collect();
            ExprKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
            }
        }

        ExprKind::Call { func, args } => {
            let args = args.into_iter().map(hoist_in_expr).collect();
            ExprKind::Call { func, args }
        }

        ExprKind::Intrinsic { name, args } => {
            let args = args.into_iter().map(hoist_in_expr).collect();
            ExprKind::Intrinsic { name, args }
        }

        ExprKind::Attributed { attributes, expr } => {
            let expr = Box::new(hoist_in_expr(*expr));
            ExprKind::Attributed { attributes, expr }
        }

        ExprKind::Materialize(inner) => {
            let inner = Box::new(hoist_in_expr(*inner));
            ExprKind::Materialize(inner)
        }

        // Literals and other simple expressions don't need recursion
        other => other,
    };

    Expr { id, ty, kind, span }
}

/// Collect all Materialize expressions from an expression tree.
/// Returns a list of (inner_expr, materialize_type) pairs.
fn collect_materializations(expr: &Expr) -> Vec<(Expr, Type<TypeName>)> {
    let mut result = Vec::new();
    collect_materializations_rec(expr, &mut result);
    result
}

fn collect_materializations_rec(expr: &Expr, result: &mut Vec<(Expr, Type<TypeName>)>) {
    match &expr.kind {
        ExprKind::Materialize(inner) => {
            result.push((*inner.clone(), expr.ty.clone()));
            collect_materializations_rec(inner, result);
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_materializations_rec(cond, result);
            collect_materializations_rec(then_branch, result);
            collect_materializations_rec(else_branch, result);
        }
        ExprKind::Let { value, body, .. } => {
            collect_materializations_rec(value, result);
            collect_materializations_rec(body, result);
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            collect_materializations_rec(lhs, result);
            collect_materializations_rec(rhs, result);
        }
        ExprKind::UnaryOp { operand, .. } => {
            collect_materializations_rec(operand, result);
        }
        ExprKind::Loop {
            init,
            init_bindings,
            body,
            ..
        } => {
            collect_materializations_rec(init, result);
            for (_, binding) in init_bindings {
                collect_materializations_rec(binding, result);
            }
            collect_materializations_rec(body, result);
        }
        ExprKind::Call { args, .. } | ExprKind::Intrinsic { args, .. } => {
            for arg in args {
                collect_materializations_rec(arg, result);
            }
        }
        ExprKind::Attributed { expr, .. } => {
            collect_materializations_rec(expr, result);
        }
        ExprKind::Literal(lit) => {
            use crate::mir::Literal;
            match lit {
                Literal::Tuple(exprs) | Literal::Array(exprs) | Literal::Vector(exprs) => {
                    for e in exprs {
                        collect_materializations_rec(e, result);
                    }
                }
                Literal::Matrix(rows) => {
                    for row in rows {
                        for e in row {
                            collect_materializations_rec(e, result);
                        }
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}

/// Find materializations that appear in both lists (by structural equality of inner expr).
fn find_common_materializations(
    then_mats: &[(Expr, Type<TypeName>)],
    else_mats: &[(Expr, Type<TypeName>)],
) -> Vec<(Expr, Type<TypeName>)> {
    let mut common = Vec::new();

    for (then_inner, then_ty) in then_mats {
        for (else_inner, _else_ty) in else_mats {
            if exprs_equal(then_inner, else_inner) {
                // Check we haven't already added this one
                if !common.iter().any(|(e, _)| exprs_equal(e, then_inner)) {
                    common.push((then_inner.clone(), then_ty.clone()));
                }
            }
        }
    }

    common
}

/// Check if two expressions are structurally equal (ignoring IDs and spans).
fn exprs_equal(a: &Expr, b: &Expr) -> bool {
    // Types should match
    if a.ty != b.ty {
        return false;
    }

    match (&a.kind, &b.kind) {
        (ExprKind::Literal(la), ExprKind::Literal(lb)) => literals_equal(la, lb),
        (ExprKind::Unit, ExprKind::Unit) => true,
        (ExprKind::Var(na), ExprKind::Var(nb)) => na == nb,
        (
            ExprKind::BinOp {
                op: opa,
                lhs: la,
                rhs: ra,
            },
            ExprKind::BinOp {
                op: opb,
                lhs: lb,
                rhs: rb,
            },
        ) => opa == opb && exprs_equal(la, lb) && exprs_equal(ra, rb),
        (ExprKind::UnaryOp { op: opa, operand: oa }, ExprKind::UnaryOp { op: opb, operand: ob }) => {
            opa == opb && exprs_equal(oa, ob)
        }
        (ExprKind::Call { func: fa, args: aa }, ExprKind::Call { func: fb, args: ab }) => {
            fa == fb && aa.len() == ab.len() && aa.iter().zip(ab.iter()).all(|(x, y)| exprs_equal(x, y))
        }
        (ExprKind::Intrinsic { name: na, args: aa }, ExprKind::Intrinsic { name: nb, args: ab }) => {
            na == nb && aa.len() == ab.len() && aa.iter().zip(ab.iter()).all(|(x, y)| exprs_equal(x, y))
        }
        (ExprKind::Materialize(ia), ExprKind::Materialize(ib)) => exprs_equal(ia, ib),
        // For simplicity, don't consider if/let/loop as equal even if structurally same
        // (they're complex and unlikely to be duplicated materializations anyway)
        _ => false,
    }
}

fn literals_equal(a: &crate::mir::Literal, b: &crate::mir::Literal) -> bool {
    use crate::mir::Literal;
    match (a, b) {
        (Literal::Int(a), Literal::Int(b)) => a == b,
        (Literal::Float(a), Literal::Float(b)) => a == b,
        (Literal::Bool(a), Literal::Bool(b)) => a == b,
        (Literal::String(a), Literal::String(b)) => a == b,
        (Literal::Tuple(a), Literal::Tuple(b))
        | (Literal::Array(a), Literal::Array(b))
        | (Literal::Vector(a), Literal::Vector(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| exprs_equal(x, y))
        }
        (Literal::Matrix(a), Literal::Matrix(b)) => {
            a.len() == b.len()
                && a.iter().zip(b.iter()).all(|(ra, rb)| {
                    ra.len() == rb.len() && ra.iter().zip(rb.iter()).all(|(x, y)| exprs_equal(x, y))
                })
        }
        _ => false,
    }
}

/// Replace occurrences of @materialize(inner_expr) with a variable reference.
fn replace_materialize(expr: &Expr, target_inner: &Expr, var_name: &str, mat_ty: &Type<TypeName>) -> Expr {
    let Expr { id, ty, kind, span } = expr.clone();

    let kind = match kind {
        ExprKind::Materialize(inner) => {
            if exprs_equal(&inner, target_inner) {
                // Replace with variable reference
                return Expr {
                    id,
                    ty: mat_ty.clone(),
                    kind: ExprKind::Var(var_name.to_string()),
                    span,
                };
            } else {
                // Recurse into the inner expression
                ExprKind::Materialize(Box::new(replace_materialize(
                    &inner,
                    target_inner,
                    var_name,
                    mat_ty,
                )))
            }
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => ExprKind::If {
            cond: Box::new(replace_materialize(&cond, target_inner, var_name, mat_ty)),
            then_branch: Box::new(replace_materialize(&then_branch, target_inner, var_name, mat_ty)),
            else_branch: Box::new(replace_materialize(&else_branch, target_inner, var_name, mat_ty)),
        },

        ExprKind::Let {
            name,
            binding_id,
            value,
            body,
        } => ExprKind::Let {
            name,
            binding_id,
            value: Box::new(replace_materialize(&value, target_inner, var_name, mat_ty)),
            body: Box::new(replace_materialize(&body, target_inner, var_name, mat_ty)),
        },

        ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
            op,
            lhs: Box::new(replace_materialize(&lhs, target_inner, var_name, mat_ty)),
            rhs: Box::new(replace_materialize(&rhs, target_inner, var_name, mat_ty)),
        },

        ExprKind::UnaryOp { op, operand } => ExprKind::UnaryOp {
            op,
            operand: Box::new(replace_materialize(&operand, target_inner, var_name, mat_ty)),
        },

        ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind: loop_kind,
            body,
        } => ExprKind::Loop {
            loop_var,
            init: Box::new(replace_materialize(&init, target_inner, var_name, mat_ty)),
            init_bindings: init_bindings
                .into_iter()
                .map(|(n, e)| (n, replace_materialize(&e, target_inner, var_name, mat_ty)))
                .collect(),
            kind: loop_kind,
            body: Box::new(replace_materialize(&body, target_inner, var_name, mat_ty)),
        },

        ExprKind::Call { func, args } => ExprKind::Call {
            func,
            args: args
                .into_iter()
                .map(|a| replace_materialize(&a, target_inner, var_name, mat_ty))
                .collect(),
        },

        ExprKind::Intrinsic { name, args } => ExprKind::Intrinsic {
            name,
            args: args
                .into_iter()
                .map(|a| replace_materialize(&a, target_inner, var_name, mat_ty))
                .collect(),
        },

        ExprKind::Attributed {
            attributes,
            expr: inner,
        } => ExprKind::Attributed {
            attributes,
            expr: Box::new(replace_materialize(&inner, target_inner, var_name, mat_ty)),
        },

        // Literals with nested expressions
        ExprKind::Literal(lit) => {
            use crate::mir::Literal;
            let new_lit = match lit {
                Literal::Tuple(exprs) => Literal::Tuple(
                    exprs
                        .into_iter()
                        .map(|e| replace_materialize(&e, target_inner, var_name, mat_ty))
                        .collect(),
                ),
                Literal::Array(exprs) => Literal::Array(
                    exprs
                        .into_iter()
                        .map(|e| replace_materialize(&e, target_inner, var_name, mat_ty))
                        .collect(),
                ),
                Literal::Vector(exprs) => Literal::Vector(
                    exprs
                        .into_iter()
                        .map(|e| replace_materialize(&e, target_inner, var_name, mat_ty))
                        .collect(),
                ),
                Literal::Matrix(rows) => Literal::Matrix(
                    rows.into_iter()
                        .map(|row| {
                            row.into_iter()
                                .map(|e| replace_materialize(&e, target_inner, var_name, mat_ty))
                                .collect()
                        })
                        .collect(),
                ),
                other => other,
            };
            ExprKind::Literal(new_lit)
        }

        // Simple cases that don't contain nested expressions
        other => other,
    };

    Expr { id, ty, kind, span }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{NodeId, Span};

    #[test]
    fn test_exprs_equal_var() {
        let span = Span::new(0, 0, 0, 0);
        let a = Expr::new(
            NodeId(1),
            Type::Constructed(TypeName::Int(32), vec![]),
            ExprKind::Var("x".to_string()),
            span,
        );
        let b = Expr::new(
            NodeId(2), // Different ID
            Type::Constructed(TypeName::Int(32), vec![]),
            ExprKind::Var("x".to_string()),
            span,
        );
        assert!(exprs_equal(&a, &b));
    }

    #[test]
    fn test_exprs_not_equal_different_var() {
        let span = Span::new(0, 0, 0, 0);
        let a = Expr::new(
            NodeId(1),
            Type::Constructed(TypeName::Int(32), vec![]),
            ExprKind::Var("x".to_string()),
            span,
        );
        let b = Expr::new(
            NodeId(1),
            Type::Constructed(TypeName::Int(32), vec![]),
            ExprKind::Var("y".to_string()),
            span,
        );
        assert!(!exprs_equal(&a, &b));
    }
}
