//! Binding lifting (code motion) pass for MIR.
//!
//! This pass hoists loop-invariant bindings out of loops to reduce
//! redundant computation. A binding is loop-invariant if its value
//! depends only on variables defined outside the loop.
//!
//! ## Example Transformation
//!
//! ```text
//! // Before:
//! loop acc = 0 for i < n do
//!     let x = expensive_constant in
//!     let y = acc + x in
//!     y
//!
//! // After:
//! let x = expensive_constant in
//! loop acc = 0 for i < n do
//!     let y = acc + x in
//!     y
//! ```

use std::collections::HashSet;

use polytype::Type;

use crate::ast::{Span, TypeName};
use crate::error::Result;
use crate::mir::{Def, Expr, ExprKind, Literal, LoopKind, Program};

/// A single binding in linear form, extracted from nested Let chains.
struct LinearBinding {
    name: String,
    binding_id: u64,
    value: Expr,
    /// Set of free variables in the value expression.
    free_vars: HashSet<String>,
    /// Type of the binding (from the body's type context).
    ty: Type<TypeName>,
    span: Span,
}

/// Linearized representation of a Let chain.
struct LinearizedBody {
    /// Bindings in topological order (dependencies before uses).
    bindings: Vec<LinearBinding>,
    /// The final result expression (non-Let).
    result: Expr,
}

/// Binding lifter pass for hoisting loop-invariant bindings.
pub struct BindingLifter {}

impl BindingLifter {
    pub fn new() -> Self {
        BindingLifter {}
    }

    /// Lift bindings in all definitions in a program.
    pub fn lift_program(&mut self, program: Program) -> Result<Program> {
        let defs = program.defs.into_iter().map(|def| self.lift_def(def)).collect::<Result<Vec<_>>>()?;

        Ok(Program {
            defs,
            lambda_registry: program.lambda_registry,
        })
    }

    /// Lift bindings in a single definition.
    fn lift_def(&mut self, def: Def) -> Result<Def> {
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
                let body = self.lift_expr(body)?;
                Ok(Def::Function {
                    name,
                    params,
                    ret_type,
                    attributes,
                    param_attributes,
                    return_attributes,
                    body,
                    span,
                })
            }
            Def::Constant {
                name,
                ty,
                attributes,
                body,
                span,
            } => {
                let body = self.lift_expr(body)?;
                Ok(Def::Constant {
                    name,
                    ty,
                    attributes,
                    body,
                    span,
                })
            }
            Def::Uniform { .. } => Ok(def),
        }
    }

    /// Main recursive driver: lift bindings in an expression.
    pub fn lift_expr(&mut self, expr: Expr) -> Result<Expr> {
        let ty = expr.ty.clone();
        let span = expr.span;

        match expr.kind {
            ExprKind::Loop { .. } => self.lift_loop(expr),

            ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            } => {
                // Recursively lift in both value and body
                let value = self.lift_expr(*value)?;
                let body = self.lift_expr(*body)?;
                Ok(Expr::new(
                    ty,
                    ExprKind::Let {
                        name,
                        binding_id,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    span,
                ))
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond = self.lift_expr(*cond)?;
                let then_branch = self.lift_expr(*then_branch)?;
                let else_branch = self.lift_expr(*else_branch)?;
                Ok(Expr::new(
                    ty,
                    ExprKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    span,
                ))
            }

            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs = self.lift_expr(*lhs)?;
                let rhs = self.lift_expr(*rhs)?;
                Ok(Expr::new(
                    ty,
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    span,
                ))
            }

            ExprKind::UnaryOp { op, operand } => {
                let operand = self.lift_expr(*operand)?;
                Ok(Expr::new(
                    ty,
                    ExprKind::UnaryOp {
                        op,
                        operand: Box::new(operand),
                    },
                    span,
                ))
            }

            ExprKind::Call { func, args } => {
                let args = args.into_iter().map(|a| self.lift_expr(a)).collect::<Result<Vec<_>>>()?;
                Ok(Expr::new(ty, ExprKind::Call { func, args }, span))
            }

            ExprKind::Intrinsic { name, args } => {
                let args = args.into_iter().map(|a| self.lift_expr(a)).collect::<Result<Vec<_>>>()?;
                Ok(Expr::new(ty, ExprKind::Intrinsic { name, args }, span))
            }

            ExprKind::Attributed {
                attributes,
                expr: inner,
            } => {
                let inner = self.lift_expr(*inner)?;
                Ok(Expr::new(
                    ty,
                    ExprKind::Attributed {
                        attributes,
                        expr: Box::new(inner),
                    },
                    span,
                ))
            }

            ExprKind::Materialize(inner) => {
                let inner = self.lift_expr(*inner)?;
                Ok(Expr::new(ty, ExprKind::Materialize(Box::new(inner)), span))
            }

            ExprKind::Literal(lit) => {
                let lit = self.lift_literal(lit)?;
                Ok(Expr::new(ty, ExprKind::Literal(lit), span))
            }

            // Leaf nodes - no children to process
            ExprKind::Var(_) | ExprKind::Unit => Ok(Expr::new(ty, expr.kind, span)),
        }
    }

    /// Lift bindings in literals (tuples, arrays, etc. may contain expressions).
    fn lift_literal(&mut self, lit: Literal) -> Result<Literal> {
        match lit {
            Literal::Tuple(elems) => {
                let elems = elems.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>()?;
                Ok(Literal::Tuple(elems))
            }
            Literal::Array(elems) => {
                let elems = elems.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>()?;
                Ok(Literal::Array(elems))
            }
            Literal::Vector(elems) => {
                let elems = elems.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>()?;
                Ok(Literal::Vector(elems))
            }
            Literal::Matrix(rows) => {
                let rows = rows
                    .into_iter()
                    .map(|row| row.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()?;
                Ok(Literal::Matrix(rows))
            }
            // Scalar literals have no sub-expressions
            Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_) => Ok(lit),
        }
    }

    /// Lift loop-invariant bindings out of a loop.
    fn lift_loop(&mut self, loop_expr: Expr) -> Result<Expr> {
        let ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } = loop_expr.kind
        else {
            unreachable!("lift_loop called on non-loop expression");
        };

        let ty = loop_expr.ty;
        let span = loop_expr.span;

        // 1. Recursively lift in init expression
        let init = self.lift_expr(*init)?;

        // 2. Recursively lift in init_bindings expressions
        let init_bindings = init_bindings
            .into_iter()
            .map(|(name, expr)| Ok((name, self.lift_expr(expr)?)))
            .collect::<Result<Vec<_>>>()?;

        // 3. Recursively lift in loop kind (iter expression or condition)
        let kind = self.lift_loop_kind(kind)?;

        // 4. Recursively lift nested loops in body first
        let body = self.lift_expr(*body)?;

        // 5. Bubble up Lets from inside pure contexts (arrays, function args, etc.)
        //    so they become visible to linearize_body
        // let body = bubble_up_lets(body);  // TEMPORARILY DISABLED

        // 6. Linearize the body
        let LinearizedBody { bindings, result } = linearize_body(body);

        // If no bindings, nothing to hoist
        if bindings.is_empty() {
            return Ok(Expr::new(
                ty,
                ExprKind::Loop {
                    loop_var,
                    init: Box::new(init),
                    init_bindings,
                    kind,
                    body: Box::new(result),
                },
                span,
            ));
        }

        // 6. Compute loop-scoped variables
        let mut loop_vars: HashSet<String> = HashSet::new();
        loop_vars.insert(loop_var.clone());
        for (name, _) in &init_bindings {
            loop_vars.insert(name.clone());
        }
        match &kind {
            LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } => {
                loop_vars.insert(var.clone());
            }
            LoopKind::While { .. } => {}
        }

        // 7. Partition bindings into hoistable and remaining
        let (hoistable, remaining) = partition_bindings(bindings, &loop_vars);

        // 8. Rebuild the loop body with remaining bindings
        let new_body = rebuild_nested_lets(remaining, result);

        // 9. Create the new loop
        let new_loop = Expr::new(
            ty,
            ExprKind::Loop {
                loop_var,
                init: Box::new(init),
                init_bindings,
                kind,
                body: Box::new(new_body),
            },
            span,
        );

        // 10. Wrap hoisted bindings around the loop
        Ok(rebuild_nested_lets(hoistable, new_loop))
    }

    /// Lift bindings in loop kind expressions.
    fn lift_loop_kind(&mut self, kind: LoopKind) -> Result<LoopKind> {
        match kind {
            LoopKind::For { var, iter } => {
                let iter = self.lift_expr(*iter)?;
                Ok(LoopKind::For {
                    var,
                    iter: Box::new(iter),
                })
            }
            LoopKind::ForRange { var, bound } => {
                let bound = self.lift_expr(*bound)?;
                Ok(LoopKind::ForRange {
                    var,
                    bound: Box::new(bound),
                })
            }
            LoopKind::While { cond } => {
                let cond = self.lift_expr(*cond)?;
                Ok(LoopKind::While { cond: Box::new(cond) })
            }
        }
    }
}

/// Bubble up Let expressions from inside pure contexts (arrays, tuples, function args, etc.)
/// to the surface where they can be seen by linearize_body.
///
/// Transforms: `@[(let x = A in B), C]` => `let x = A in @[B, C]`
///
/// Does NOT extract from if branches since only one branch executes.
fn bubble_up_lets(expr: Expr) -> Expr {
    let ty = expr.ty.clone();
    let span = expr.span;

    match expr.kind {
        // Already a Let - recurse into value and body, then bubble from value
        ExprKind::Let {
            name,
            binding_id,
            value,
            body,
        } => {
            let value = bubble_up_lets(*value);
            let body = bubble_up_lets(*body);
            // If value is itself a Let, hoist it
            // let x = (let y = A in B) in C => let y = A in let x = B in C
            if let ExprKind::Let {
                name: inner_name,
                binding_id: inner_binding_id,
                value: inner_value,
                body: inner_body,
            } = value.kind
            {
                let new_inner = Expr::new(
                    ty,
                    ExprKind::Let {
                        name,
                        binding_id,
                        value: inner_body,
                        body: Box::new(body),
                    },
                    span,
                );
                // Recurse to handle chains
                bubble_up_lets(Expr::new(
                    new_inner.ty.clone(),
                    ExprKind::Let {
                        name: inner_name,
                        binding_id: inner_binding_id,
                        value: inner_value,
                        body: Box::new(new_inner),
                    },
                    span,
                ))
            } else {
                Expr::new(
                    ty,
                    ExprKind::Let {
                        name,
                        binding_id,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    span,
                )
            }
        }

        // Pure contexts - extract Lets from children
        ExprKind::Literal(lit) => {
            let (lets, lit) = bubble_up_from_literal(lit);
            wrap_lets(lets, Expr::new(ty, ExprKind::Literal(lit), span))
        }

        ExprKind::Call { func, args } => {
            let (lets, args) = bubble_up_from_exprs(args);
            wrap_lets(lets, Expr::new(ty, ExprKind::Call { func, args }, span))
        }

        ExprKind::Intrinsic { name, args } => {
            let (lets, args) = bubble_up_from_exprs(args);
            wrap_lets(lets, Expr::new(ty, ExprKind::Intrinsic { name, args }, span))
        }

        ExprKind::BinOp { op, lhs, rhs } => {
            let lhs = bubble_up_lets(*lhs);
            let rhs = bubble_up_lets(*rhs);
            let (lets, exprs) = bubble_up_from_exprs(vec![lhs, rhs]);
            let mut it = exprs.into_iter();
            let lhs = it.next().unwrap();
            let rhs = it.next().unwrap();
            wrap_lets(
                lets,
                Expr::new(
                    ty,
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    span,
                ),
            )
        }

        ExprKind::UnaryOp { op, operand } => {
            let operand = bubble_up_lets(*operand);
            if let ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            } = operand.kind
            {
                let inner = Expr::new(ty, ExprKind::UnaryOp { op, operand: body }, span);
                Expr::new(
                    inner.ty.clone(),
                    ExprKind::Let {
                        name,
                        binding_id,
                        value,
                        body: Box::new(inner),
                    },
                    span,
                )
            } else {
                Expr::new(
                    ty,
                    ExprKind::UnaryOp {
                        op,
                        operand: Box::new(operand),
                    },
                    span,
                )
            }
        }

        ExprKind::Materialize(inner) => {
            let inner = bubble_up_lets(*inner);
            if let ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            } = inner.kind
            {
                let mat = Expr::new(ty, ExprKind::Materialize(body), span);
                Expr::new(
                    mat.ty.clone(),
                    ExprKind::Let {
                        name,
                        binding_id,
                        value,
                        body: Box::new(mat),
                    },
                    span,
                )
            } else {
                Expr::new(ty, ExprKind::Materialize(Box::new(inner)), span)
            }
        }

        // If branches - recurse but do NOT extract (only one branch executes)
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let cond = bubble_up_lets(*cond);
            let then_branch = bubble_up_lets(*then_branch);
            let else_branch = bubble_up_lets(*else_branch);
            // Only extract from cond (always evaluated), not branches
            if let ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            } = cond.kind
            {
                let inner = Expr::new(
                    ty,
                    ExprKind::If {
                        cond: body,
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    span,
                );
                Expr::new(
                    inner.ty.clone(),
                    ExprKind::Let {
                        name,
                        binding_id,
                        value,
                        body: Box::new(inner),
                    },
                    span,
                )
            } else {
                Expr::new(
                    ty,
                    ExprKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    span,
                )
            }
        }

        // Loop - recurse into parts but don't extract from body
        ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let init = bubble_up_lets(*init);
            let init_bindings =
                init_bindings.into_iter().map(|(name, e)| (name, bubble_up_lets(e))).collect();
            let kind = bubble_up_from_loop_kind(kind);
            let body = bubble_up_lets(*body);
            Expr::new(
                ty,
                ExprKind::Loop {
                    loop_var,
                    init: Box::new(init),
                    init_bindings,
                    kind,
                    body: Box::new(body),
                },
                span,
            )
        }

        ExprKind::Attributed {
            attributes,
            expr: inner,
        } => {
            let inner = bubble_up_lets(*inner);
            Expr::new(
                ty,
                ExprKind::Attributed {
                    attributes,
                    expr: Box::new(inner),
                },
                span,
            )
        }

        // Leaf nodes - no children
        ExprKind::Var(_) | ExprKind::Unit => expr,
    }
}

/// Extract Lets from a list of expressions.
/// Returns (extracted_lets, transformed_exprs) where extracted_lets
/// is a list of (name, value) pairs in order.
fn bubble_up_from_exprs(exprs: Vec<Expr>) -> (Vec<(String, Expr)>, Vec<Expr>) {
    let mut lets = Vec::new();
    let mut results = Vec::new();

    for expr in exprs {
        let expr = bubble_up_lets(expr);
        // Peel off any top-level Let
        let mut current = expr;
        while let ExprKind::Let {
            name, value, body, ..
        } = current.kind
        {
            lets.push((name, *value));
            current = *body;
        }
        results.push(current);
    }

    (lets, results)
}

/// Extract Lets from a literal (array, tuple, etc.)
fn bubble_up_from_literal(lit: Literal) -> (Vec<(String, Expr)>, Literal) {
    match lit {
        Literal::Tuple(elems) => {
            let (lets, elems) = bubble_up_from_exprs(elems);
            (lets, Literal::Tuple(elems))
        }
        Literal::Array(elems) => {
            let (lets, elems) = bubble_up_from_exprs(elems);
            (lets, Literal::Array(elems))
        }
        Literal::Vector(elems) => {
            let (lets, elems) = bubble_up_from_exprs(elems);
            (lets, Literal::Vector(elems))
        }
        Literal::Matrix(rows) => {
            let mut all_lets = Vec::new();
            let mut new_rows = Vec::new();
            for row in rows {
                let (lets, row) = bubble_up_from_exprs(row);
                all_lets.extend(lets);
                new_rows.push(row);
            }
            (all_lets, Literal::Matrix(new_rows))
        }
        // Scalar literals - no sub-expressions
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_) => (vec![], lit),
    }
}

/// Extract Lets from loop kind expressions.
fn bubble_up_from_loop_kind(kind: LoopKind) -> LoopKind {
    match kind {
        LoopKind::For { var, iter } => LoopKind::For {
            var,
            iter: Box::new(bubble_up_lets(*iter)),
        },
        LoopKind::ForRange { var, bound } => LoopKind::ForRange {
            var,
            bound: Box::new(bubble_up_lets(*bound)),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: Box::new(bubble_up_lets(*cond)),
        },
    }
}

/// Wrap an expression with a series of Let bindings.
fn wrap_lets(lets: Vec<(String, Expr)>, inner: Expr) -> Expr {
    lets.into_iter().rev().fold(inner, |body, (name, value)| {
        let span = body.span;
        Expr::new(
            body.ty.clone(),
            ExprKind::Let {
                name,
                binding_id: 0, // TODO: proper binding ID assignment
                value: Box::new(value),
                body: Box::new(body),
            },
            span,
        )
    })
}

/// Linearize a nested Let chain into a flat list of bindings.
fn linearize_body(mut expr: Expr) -> LinearizedBody {
    let mut bindings = Vec::new();

    while let ExprKind::Let {
        name,
        binding_id,
        value,
        body,
    } = expr.kind
    {
        let free_vars = collect_free_vars(&value);
        bindings.push(LinearBinding {
            name,
            binding_id,
            value: *value,
            free_vars,
            ty: expr.ty.clone(),
            span: expr.span,
        });
        expr = *body;
    }

    LinearizedBody {
        bindings,
        result: expr,
    }
}

/// Partition bindings into hoistable (loop-invariant) and remaining (loop-dependent).
fn partition_bindings(
    bindings: Vec<LinearBinding>,
    loop_vars: &HashSet<String>,
) -> (Vec<LinearBinding>, Vec<LinearBinding>) {
    let mut tainted = loop_vars.clone();
    let mut hoistable = Vec::new();
    let mut remaining = Vec::new();

    for binding in bindings {
        if binding.free_vars.is_disjoint(&tainted) {
            // Can hoist - no loop dependencies
            hoistable.push(binding);
        } else {
            // Cannot hoist - mark this name as tainted for subsequent bindings
            tainted.insert(binding.name.clone());
            remaining.push(binding);
        }
    }

    (hoistable, remaining)
}

/// Rebuild a nested Let chain from linear bindings.
fn rebuild_nested_lets(bindings: Vec<LinearBinding>, result: Expr) -> Expr {
    bindings.into_iter().rev().fold(result, |body, binding| {
        Expr::new(
            body.ty.clone(),
            ExprKind::Let {
                name: binding.name,
                binding_id: binding.binding_id,
                value: Box::new(binding.value),
                body: Box::new(body),
            },
            binding.span,
        )
    })
}

/// Collect free variables in an expression.
pub fn collect_free_vars(expr: &Expr) -> HashSet<String> {
    let mut free = HashSet::new();
    collect_free_vars_inner(expr, &HashSet::new(), &mut free);
    free
}

/// Inner recursive function for collecting free variables.
fn collect_free_vars_inner(expr: &Expr, bound: &HashSet<String>, free: &mut HashSet<String>) {
    match &expr.kind {
        ExprKind::Var(name) => {
            if !bound.contains(name) {
                free.insert(name.clone());
            }
        }

        ExprKind::Let {
            name, value, body, ..
        } => {
            collect_free_vars_inner(value, bound, free);
            let mut extended = bound.clone();
            extended.insert(name.clone());
            collect_free_vars_inner(body, &extended, free);
        }

        ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } => {
            collect_free_vars_inner(init, bound, free);

            // init_bindings reference loop_var, but their expressions are evaluated
            // in the context where loop_var is bound
            let mut extended = bound.clone();
            extended.insert(loop_var.clone());

            for (name, binding_expr) in init_bindings {
                collect_free_vars_inner(binding_expr, &extended, free);
                extended.insert(name.clone());
            }

            match kind {
                LoopKind::For { var, iter } => {
                    collect_free_vars_inner(iter, bound, free);
                    extended.insert(var.clone());
                }
                LoopKind::ForRange { var, bound: upper } => {
                    collect_free_vars_inner(upper, bound, free);
                    extended.insert(var.clone());
                }
                LoopKind::While { cond } => {
                    collect_free_vars_inner(cond, &extended, free);
                }
            }

            collect_free_vars_inner(body, &extended, free);
        }

        ExprKind::BinOp { lhs, rhs, .. } => {
            collect_free_vars_inner(lhs, bound, free);
            collect_free_vars_inner(rhs, bound, free);
        }

        ExprKind::UnaryOp { operand, .. } => {
            collect_free_vars_inner(operand, bound, free);
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_free_vars_inner(cond, bound, free);
            collect_free_vars_inner(then_branch, bound, free);
            collect_free_vars_inner(else_branch, bound, free);
        }

        ExprKind::Call { args, .. } | ExprKind::Intrinsic { args, .. } => {
            for arg in args {
                collect_free_vars_inner(arg, bound, free);
            }
        }

        ExprKind::Literal(lit) => {
            collect_free_vars_in_literal(lit, bound, free);
        }

        ExprKind::Attributed { expr, .. } => {
            collect_free_vars_inner(expr, bound, free);
        }

        ExprKind::Materialize(inner) => {
            collect_free_vars_inner(inner, bound, free);
        }

        ExprKind::Unit => {}
    }
}

/// Collect free variables in literal expressions.
fn collect_free_vars_in_literal(lit: &Literal, bound: &HashSet<String>, free: &mut HashSet<String>) {
    match lit {
        Literal::Tuple(elems) | Literal::Array(elems) | Literal::Vector(elems) => {
            for elem in elems {
                collect_free_vars_inner(elem, bound, free);
            }
        }
        Literal::Matrix(rows) => {
            for row in rows {
                for elem in row {
                    collect_free_vars_inner(elem, bound, free);
                }
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_) => {}
    }
}
