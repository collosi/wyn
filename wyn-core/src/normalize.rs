//! A-Normal Form (ANF) normalization pass for MIR.
//!
//! This pass ensures that all compound expressions have atomic operands,
//! enabling code motion optimizations. After normalization:
//! - BinOp/UnaryOp operands are Var or scalar Literal
//! - Call/Intrinsic args are Var or scalar Literal
//! - Tuple/Array/Vector/Matrix elements are Var only
//! - Materialize inner is Var or scalar Literal
//! - If/Loop conditions are Var or scalar Literal

use crate::ast::NodeCounter;
use crate::mir::{Def, Expr, ExprKind, Literal, LocalId, LocalTable, LoopKind, Program};
use polytype::Type;
use crate::ast::{Span, TypeName};

/// A pending let binding (local_id, value).
type Binding = (LocalId, Expr);

/// Normalizer state for the ANF transformation.
pub struct Normalizer {
    /// Counter for generating unique temp names.
    next_temp_id: usize,
    /// Counter for generating unique node IDs.
    node_counter: NodeCounter,
    /// Current function's local table for allocating new locals.
    current_local_table: LocalTable,
}

impl Normalizer {
    /// Create a new normalizer with the given node counter.
    pub fn new(node_counter: NodeCounter) -> Self {
        Normalizer {
            next_temp_id: 0,
            node_counter,
            current_local_table: LocalTable::new(),
        }
    }

    /// Allocate a fresh local for a temp variable.
    fn alloc_temp(&mut self, ty: Type<TypeName>, span: Option<Span>) -> LocalId {
        let id = self.next_temp_id;
        self.next_temp_id += 1;
        let name = format!("_w_norm_{}", id);
        self.current_local_table.alloc(name, ty, span)
    }

    /// Reset local table for a new function.
    fn reset_local_table(&mut self) {
        self.current_local_table = LocalTable::new();
    }

    /// Normalize an entire program.
    pub fn normalize_program(&mut self, program: Program) -> Program {
        let defs = program.defs.into_iter().map(|d| self.normalize_def(d)).collect();
        Program {
            defs,
            lambda_registry: program.lambda_registry,
            local_tables: program.local_tables,
        }
    }

    /// Normalize a single definition.
    fn normalize_def(&mut self, def: Def) -> Def {
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
                let mut bindings = Vec::new();
                let body = self.normalize_expr(body, &mut bindings);
                let body = self.wrap_bindings(body, bindings);
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
            // Constants must remain compile-time literals, so don't normalize them
            Def::Constant { .. } => def,
            Def::Uniform { .. } => def,  // Uniforms have no body to normalize
            Def::Storage { .. } => def,  // Storage buffers have no body to normalize
            Def::EntryPoint {
                id,
                name,
                execution_model,
                inputs,
                outputs,
                body,
                span,
            } => {
                let mut bindings = Vec::new();
                let body = self.normalize_expr(body, &mut bindings);
                let body = self.wrap_bindings(body, bindings);
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
        }
    }

    /// Normalize an expression, collecting pending bindings.
    pub fn normalize_expr(&mut self, expr: Expr, bindings: &mut Vec<Binding>) -> Expr {
        let id = expr.id;
        let span = expr.span;
        let ty = expr.ty.clone();

        match expr.kind {
            // Already atomic - return as-is
            ExprKind::Var(_) | ExprKind::Unit => expr,

            // Scalar literals are atomic - return as-is
            ExprKind::Literal(Literal::Int(_))
            | ExprKind::Literal(Literal::Float(_))
            | ExprKind::Literal(Literal::Bool(_))
            | ExprKind::Literal(Literal::String(_)) => expr,

            // Binary operation - atomize both operands
            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs = self.normalize_expr(*lhs, bindings);
                let lhs = self.atomize(lhs, bindings);
                let rhs = self.normalize_expr(*rhs, bindings);
                let rhs = self.atomize(rhs, bindings);
                Expr::new(
                    id,
                    ty,
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    span,
                )
            }

            // Unary operation - atomize operand
            ExprKind::UnaryOp { op, operand } => {
                let operand = self.normalize_expr(*operand, bindings);
                let operand = self.atomize(operand, bindings);
                Expr::new(
                    id,
                    ty,
                    ExprKind::UnaryOp {
                        op,
                        operand: Box::new(operand),
                    },
                    span,
                )
            }

            // Function call - atomize all args
            ExprKind::Call { func, func_name, args } => {
                let args = args
                    .into_iter()
                    .map(|a| {
                        let a = self.normalize_expr(a, bindings);
                        self.atomize(a, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Call { func, func_name, args }, span)
            }

            // Intrinsic - atomize all args
            ExprKind::Intrinsic { id: intrinsic_id, args } => {
                let args = args
                    .into_iter()
                    .map(|a| {
                        let a = self.normalize_expr(a, bindings);
                        self.atomize(a, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Intrinsic { id: intrinsic_id, args }, span)
            }

            // Tuple literal - handle empty and non-empty cases
            ExprKind::Literal(Literal::Tuple(ref elems)) if elems.is_empty() => {
                // Empty tuples are atomic
                expr
            }
            ExprKind::Literal(Literal::Tuple(elems)) => {
                let elems = elems
                    .into_iter()
                    .map(|e| {
                        let e = self.normalize_expr(e, bindings);
                        self.atomize(e, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Tuple(elems)), span)
            }

            // Array literal - atomize all elements
            ExprKind::Literal(Literal::Array(elems)) => {
                let elems = elems
                    .into_iter()
                    .map(|e| {
                        let e = self.normalize_expr(e, bindings);
                        self.atomize(e, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Array(elems)), span)
            }

            // Vector literal - atomize all elements
            ExprKind::Literal(Literal::Vector(elems)) => {
                let elems = elems
                    .into_iter()
                    .map(|e| {
                        let e = self.normalize_expr(e, bindings);
                        self.atomize(e, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Vector(elems)), span)
            }

            // Matrix literal - atomize all elements in all rows
            ExprKind::Literal(Literal::Matrix(rows)) => {
                let rows = rows
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|e| {
                                let e = self.normalize_expr(e, bindings);
                                self.atomize(e, bindings)
                            })
                            .collect()
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Matrix(rows)), span)
            }

            // Materialize - atomize inner
            ExprKind::Materialize(inner) => {
                let inner = self.normalize_expr(*inner, bindings);
                let inner = self.atomize(inner, bindings);
                Expr::new(id, ty, ExprKind::Materialize(Box::new(inner)), span)
            }

            // Let binding - normalize value and body
            ExprKind::Let { local, value, body } => {
                // Value gets normalized with outer bindings (value is evaluated before local is bound)
                let value = self.normalize_expr(*value, bindings);

                // Body gets its own scope - bindings from body may reference `local`
                let mut body_bindings = Vec::new();
                let body = self.normalize_expr(*body, &mut body_bindings);
                let body = self.wrap_bindings(body, body_bindings);

                Expr::new(
                    id,
                    ty,
                    ExprKind::Let {
                        local,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    span,
                )
            }

            // If expression - atomize condition, normalize branches with fresh scopes
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond = self.normalize_expr(*cond, bindings);
                let cond = self.atomize(cond, bindings);

                // Branches get their own binding scopes
                let mut then_bindings = Vec::new();
                let then_branch = self.normalize_expr(*then_branch, &mut then_bindings);
                let then_branch = self.wrap_bindings(then_branch, then_bindings);

                let mut else_bindings = Vec::new();
                let else_branch = self.normalize_expr(*else_branch, &mut else_bindings);
                let else_branch = self.wrap_bindings(else_branch, else_bindings);

                Expr::new(
                    id,
                    ty,
                    ExprKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    span,
                )
            }

            // Loop - atomize init, normalize init_bindings and body with fresh scopes
            ExprKind::Loop {
                loop_var,
                init,
                init_bindings: loop_init_bindings,
                kind,
                body,
            } => {
                // Init gets atomized
                let init = self.normalize_expr(*init, bindings);
                let init = self.atomize(init, bindings);

                // Init bindings get normalized
                let loop_init_bindings = loop_init_bindings
                    .into_iter()
                    .map(|(name, expr)| {
                        let mut init_b = Vec::new();
                        let expr = self.normalize_expr(expr, &mut init_b);
                        let expr = self.wrap_bindings(expr, init_b);
                        (name, expr)
                    })
                    .collect();

                // Loop kind gets normalized
                let kind = match kind {
                    LoopKind::While { cond } => {
                        let mut cond_bindings = Vec::new();
                        let cond = self.normalize_expr(*cond, &mut cond_bindings);
                        let cond = self.atomize(cond, &mut cond_bindings);
                        let cond = self.wrap_bindings(cond, cond_bindings);
                        LoopKind::While { cond: Box::new(cond) }
                    }
                    LoopKind::ForRange { var, bound } => {
                        let mut bound_bindings = Vec::new();
                        let bound = self.normalize_expr(*bound, &mut bound_bindings);
                        let bound = self.atomize(bound, &mut bound_bindings);
                        let bound = self.wrap_bindings(bound, bound_bindings);
                        LoopKind::ForRange {
                            var,
                            bound: Box::new(bound),
                        }
                    }
                    LoopKind::For { var, iter } => {
                        let mut iter_bindings = Vec::new();
                        let iter = self.normalize_expr(*iter, &mut iter_bindings);
                        let iter = self.atomize(iter, &mut iter_bindings);
                        let iter = self.wrap_bindings(iter, iter_bindings);
                        LoopKind::For {
                            var,
                            iter: Box::new(iter),
                        }
                    }
                };

                // Body gets its own scope
                let mut body_bindings = Vec::new();
                let body = self.normalize_expr(*body, &mut body_bindings);
                let body = self.wrap_bindings(body, body_bindings);

                Expr::new(
                    id,
                    ty,
                    ExprKind::Loop {
                        loop_var,
                        init: Box::new(init),
                        init_bindings: loop_init_bindings,
                        kind,
                        body: Box::new(body),
                    },
                    span,
                )
            }

            // Attributed - normalize inner
            ExprKind::Attributed {
                attributes,
                expr: inner,
            } => {
                let inner = self.normalize_expr(*inner, bindings);
                Expr::new(
                    id,
                    ty,
                    ExprKind::Attributed {
                        attributes,
                        expr: Box::new(inner),
                    },
                    span,
                )
            }

            // Closure - normalize and atomize captures
            ExprKind::Closure { lambda, captures } => {
                let captures = captures
                    .into_iter()
                    .map(|c| {
                        let c = self.normalize_expr(c, bindings);
                        self.atomize(c, bindings)
                    })
                    .collect();
                Expr::new(
                    id,
                    ty,
                    ExprKind::Closure { lambda, captures },
                    span,
                )
            }
        }
    }

    /// Ensure an expression is atomic, creating a temp binding if needed.
    fn atomize(&mut self, expr: Expr, bindings: &mut Vec<Binding>) -> Expr {
        if is_atomic(&expr) {
            expr
        } else {
            let ty = expr.ty.clone();
            let span = expr.span;
            let local_id = self.alloc_temp(ty.clone(), Some(span));
            bindings.push((local_id, expr));
            let node_id = self.node_counter.next();
            Expr::new(node_id, ty, ExprKind::Var(local_id), span)
        }
    }

    /// Wrap collected bindings around an expression as nested Lets.
    fn wrap_bindings(&mut self, mut expr: Expr, bindings: Vec<Binding>) -> Expr {
        for (local, value) in bindings.into_iter().rev() {
            let node_id = self.node_counter.next();
            let ty = expr.ty.clone();
            let span = expr.span;
            expr = Expr::new(
                node_id,
                ty,
                ExprKind::Let {
                    local,
                    value: Box::new(value),
                    body: Box::new(expr),
                },
                span,
            );
        }
        expr
    }
}

/// Check if an expression is atomic (can appear as an operand).
pub fn is_atomic(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Var(_) | ExprKind::Unit => true,
        ExprKind::Closure { .. } => true, // Closures are atomic values
        ExprKind::Literal(Literal::Tuple(elems)) => elems.is_empty(),
        ExprKind::Literal(lit) => is_scalar_literal(lit),
        _ => false,
    }
}

/// Check if a literal is a scalar (not a container).
fn is_scalar_literal(lit: &Literal) -> bool {
    matches!(
        lit,
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_)
    )
}

/// Normalize a MIR program to A-normal form.
pub fn normalize_program(program: Program, node_counter: NodeCounter) -> (Program, NodeCounter) {
    let mut normalizer = Normalizer::new(node_counter);
    let program = normalizer.normalize_program(program);
    (program, normalizer.node_counter)
}
