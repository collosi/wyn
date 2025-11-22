// Desugaring pass for overloaded surface syntax
//
// This pass runs after type checking and rewrites certain surface-level
// identifiers to internal variants based on the shapes of their argument types.
//
// Example: `mul a b` where a and b are matrices becomes `mul_mat_mat a b`

use crate::ast::*;
use crate::error::{CompilerError, Result};
use std::collections::HashMap;

/// Type table mapping expression IDs to their types
pub type TypeTable = HashMap<NodeId, TypeScheme>;

/// Shape classification for desugaring decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgShape {
    Matrix, // mat<n,m,a>
    Vector, // Vec<n,a>
    Other,
}

/// Classify the shape of a type for desugaring purposes
/// Ignores concrete sizes and element types, only looks at structure
fn classify_shape(ty: &Type) -> ArgShape {
    match ty {
        Type::Constructed(TypeName::Mat, _) => ArgShape::Matrix,
        Type::Constructed(TypeName::Vec, _) => ArgShape::Vector,
        _ => ArgShape::Other,
    }
}

/// Desugar multiplication operators in an expression
///
/// Rewrites surface-level `mul` to internal variants:
/// - `mul_mat_mat` for matrix × matrix
/// - `mul_mat_vec` for matrix × vector
/// - `mul_vec_mat` for vector × matrix
pub fn desugar_mul(expr: &mut Expression, type_table: &TypeTable) -> Result<()> {
    match &mut expr.kind {
        ExprKind::Application(func, args) => {
            // Check if this is a call to "mul" before recursing (to avoid borrow issues)
            let is_mul_call =
                if let ExprKind::Identifier(name) = &func.kind { name == "mul" } else { false };

            if is_mul_call {
                if args.len() != 2 {
                    return Err(CompilerError::TypeError(
                        format!("mul expects 2 arguments, got {}", args.len()),
                        expr.h.span,
                    ));
                }

                // Get the types of the arguments from the type table
                let arg1_scheme = type_table.get(&args[0].h.id).ok_or_else(|| {
                    CompilerError::TypeError(
                        format!("No type found for expression {:?}", args[0].h.id),
                        args[0].h.span,
                    )
                })?;
                let arg2_scheme = type_table.get(&args[1].h.id).ok_or_else(|| {
                    CompilerError::TypeError(
                        format!("No type found for expression {:?}", args[1].h.id),
                        args[1].h.span,
                    )
                })?;

                // Extract the monotype (these should be Monotype after type checking)
                let arg1_ty = match arg1_scheme {
                    TypeScheme::Monotype(t) => t,
                    TypeScheme::Polytype { .. } => {
                        return Err(CompilerError::TypeError(
                            "Unexpected polytype in argument position after type checking".to_string(),
                            args[0].h.span,
                        ));
                    }
                };
                let arg2_ty = match arg2_scheme {
                    TypeScheme::Monotype(t) => t,
                    TypeScheme::Polytype { .. } => {
                        return Err(CompilerError::TypeError(
                            "Unexpected polytype in argument position after type checking".to_string(),
                            args[1].h.span,
                        ));
                    }
                };

                // Classify the shapes
                let shape1 = classify_shape(arg1_ty);
                let shape2 = classify_shape(arg2_ty);

                // Determine the variant name based on shapes
                let variant_name = match (shape1, shape2) {
                    (ArgShape::Matrix, ArgShape::Matrix) => "mul_mat_mat",
                    (ArgShape::Matrix, ArgShape::Vector) => "mul_mat_vec",
                    (ArgShape::Vector, ArgShape::Matrix) => "mul_vec_mat",
                    _ => {
                        return Err(CompilerError::TypeError(
                            format!("No mul variant for argument shapes: {:?} × {:?}", shape1, shape2),
                            expr.h.span,
                        ));
                    }
                };

                // Rewrite the function name
                func.kind = ExprKind::Identifier(variant_name.to_string());
            }

            // Now recurse into function and arguments
            desugar_mul(func, type_table)?;
            for arg in args {
                desugar_mul(arg, type_table)?;
            }

            Ok(())
        }
        ExprKind::Lambda(lambda) => {
            desugar_mul(&mut lambda.body, type_table)?;
            Ok(())
        }
        ExprKind::LetIn(let_expr) => {
            desugar_mul(&mut let_expr.value, type_table)?;
            desugar_mul(&mut let_expr.body, type_table)?;
            Ok(())
        }
        ExprKind::If(if_expr) => {
            desugar_mul(&mut if_expr.condition, type_table)?;
            desugar_mul(&mut if_expr.then_branch, type_table)?;
            desugar_mul(&mut if_expr.else_branch, type_table)?;
            Ok(())
        }
        ExprKind::Loop(loop_expr) => {
            if let Some(init) = &mut loop_expr.init {
                desugar_mul(init, type_table)?;
            }
            match &mut loop_expr.form {
                LoopForm::For(_, cond) => desugar_mul(cond, type_table)?,
                LoopForm::ForIn(_, iter) => desugar_mul(iter, type_table)?,
                LoopForm::While(cond) => desugar_mul(cond, type_table)?,
            }
            desugar_mul(&mut loop_expr.body, type_table)?;
            Ok(())
        }
        ExprKind::BinaryOp(_, left, right) => {
            desugar_mul(left, type_table)?;
            desugar_mul(right, type_table)?;
            Ok(())
        }
        ExprKind::FieldAccess(inner, _) => {
            desugar_mul(inner, type_table)?;
            Ok(())
        }
        ExprKind::ArrayIndex(array, index) => {
            desugar_mul(array, type_table)?;
            desugar_mul(index, type_table)?;
            Ok(())
        }
        ExprKind::ArrayLiteral(elements) => {
            for elem in elements {
                desugar_mul(elem, type_table)?;
            }
            Ok(())
        }
        ExprKind::Tuple(elements) => {
            for elem in elements {
                desugar_mul(elem, type_table)?;
            }
            Ok(())
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, expr) in fields {
                desugar_mul(expr, type_table)?;
            }
            Ok(())
        }
        // Expression kinds we don't need to recurse into or don't support yet
        ExprKind::Identifier(_)
        | ExprKind::QualifiedName(_, _)
        | ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::TypeHole
        | ExprKind::Match(_)
        | ExprKind::Range(_)
        | ExprKind::Pipe(_, _)
        | ExprKind::TypeAscription(_, _)
        | ExprKind::TypeCoercion(_, _)
        | ExprKind::Unsafe(_)
        | ExprKind::Assert(_, _)
        | ExprKind::UnaryOp(_, _) => {
            // TODO: Recursion for these if they can contain mul calls
            Ok(())
        }
    }
}

/// Desugar a declaration
pub fn desugar_decl(decl: &mut Declaration, type_table: &TypeTable) -> Result<()> {
    match decl {
        Declaration::Decl(d) => {
            desugar_mul(&mut d.body, type_table)?;
            Ok(())
        }
        Declaration::Entry(entry) => {
            desugar_mul(&mut entry.body, type_table)?;
            Ok(())
        }
        // Other declarations don't have expressions
        _ => Ok(()),
    }
}

/// Desugar an entire program
pub fn desugar_program(program: &mut Program, type_table: &TypeTable) -> Result<()> {
    for decl in &mut program.declarations {
        desugar_decl(decl, type_table)?;
    }
    Ok(())
}
