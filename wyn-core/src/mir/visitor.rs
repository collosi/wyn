//! MIR visitor pattern for traversing and transforming the Wyn MIR.
//!
//! This module provides a centralized traversal mechanism for the MIR.
//! Each pass (optimization, lowering, analysis, etc.) can implement the
//! `MirVisitor` trait and override only the hooks they need, while the
//! `walk_*` functions handle the actual tree traversal.
//!
//! The visitor is a tree-mutating visitor: it consumes and returns values.
//! Read-only passes can just return the input unchanged.

use crate::ast::{Span, TypeName};
use crate::mir::*;
use polytype::Type;

/// Visitor trait for traversing and transforming the MIR.
///
/// All methods have default implementations that delegate to `walk_*` functions.
/// Implementors can override specific hooks to customize behavior.
///
/// The visitor consumes and returns values, making it suitable for both:
/// - Transformation passes (monomorphization, constant folding, etc.)
/// - Read-only passes (just return the input unchanged)
///
/// The `Error` associated type allows visitors to propagate errors.
pub trait MirVisitor: Sized {
    type Error;

    // --- Top-level program ---

    fn visit_program(&mut self, p: Program) -> Result<Program, Self::Error> {
        walk_program(self, p)
    }

    fn visit_def(&mut self, d: Def) -> Result<Def, Self::Error> {
        walk_def(self, d)
    }

    fn visit_function(
        &mut self,
        name: String,
        params: Vec<Param>,
        ret_type: Type<TypeName>,
        attributes: Vec<Attribute>,
        param_attributes: Vec<Vec<Attribute>>,
        return_attributes: Vec<Vec<Attribute>>,
        body: Expr,
        span: Span,
    ) -> Result<Def, Self::Error> {
        walk_function(
            self,
            name,
            params,
            ret_type,
            attributes,
            param_attributes,
            return_attributes,
            body,
            span,
        )
    }

    fn visit_constant(
        &mut self,
        name: String,
        ty: Type<TypeName>,
        attributes: Vec<Attribute>,
        body: Expr,
        span: Span,
    ) -> Result<Def, Self::Error> {
        walk_constant(self, name, ty, attributes, body, span)
    }

    fn visit_param(&mut self, p: Param) -> Result<Param, Self::Error> {
        walk_param(self, p)
    }

    fn visit_attribute(&mut self, a: Attribute) -> Result<Attribute, Self::Error> {
        walk_attribute(self, a)
    }

    fn visit_type(&mut self, ty: Type<TypeName>) -> Result<Type<TypeName>, Self::Error> {
        Ok(ty)
    }

    // --- Expressions ---

    fn visit_expr(&mut self, e: Expr) -> Result<Expr, Self::Error> {
        walk_expr(self, e)
    }

    fn visit_expr_literal(&mut self, lit: Literal, expr: Expr) -> Result<Expr, Self::Error> {
        let lit = walk_literal(self, lit)?;
        Ok(Expr {
            kind: ExprKind::Literal(lit),
            ..expr
        })
    }

    fn visit_expr_var(&mut self, _name: String, expr: Expr) -> Result<Expr, Self::Error> {
        Ok(expr)
    }

    fn visit_expr_bin_op(
        &mut self,
        op: String,
        lhs: Expr,
        rhs: Expr,
        expr: Expr,
    ) -> Result<Expr, Self::Error> {
        walk_expr_bin_op(self, op, lhs, rhs, expr)
    }

    fn visit_expr_unary_op(&mut self, op: String, operand: Expr, expr: Expr) -> Result<Expr, Self::Error> {
        walk_expr_unary_op(self, op, operand, expr)
    }

    fn visit_expr_if(
        &mut self,
        cond: Expr,
        then_branch: Expr,
        else_branch: Expr,
        expr: Expr,
    ) -> Result<Expr, Self::Error> {
        walk_expr_if(self, cond, then_branch, else_branch, expr)
    }

    fn visit_expr_let(
        &mut self,
        name: String,
        value: Expr,
        body: Expr,
        expr: Expr,
    ) -> Result<Expr, Self::Error> {
        walk_expr_let(self, name, value, body, expr)
    }

    fn visit_expr_loop(
        &mut self,
        loop_var: String,
        init: Expr,
        init_bindings: Vec<(String, Expr)>,
        kind: LoopKind,
        body: Expr,
        expr: Expr,
    ) -> Result<Expr, Self::Error> {
        walk_expr_loop(self, loop_var, init, init_bindings, kind, body, expr)
    }

    fn visit_expr_call(&mut self, func: String, args: Vec<Expr>, expr: Expr) -> Result<Expr, Self::Error> {
        walk_expr_call(self, func, args, expr)
    }

    fn visit_expr_intrinsic(
        &mut self,
        name: String,
        args: Vec<Expr>,
        expr: Expr,
    ) -> Result<Expr, Self::Error> {
        walk_expr_intrinsic(self, name, args, expr)
    }

    fn visit_expr_attributed(
        &mut self,
        attributes: Vec<Attribute>,
        inner: Expr,
        expr: Expr,
    ) -> Result<Expr, Self::Error> {
        walk_expr_attributed(self, attributes, inner, expr)
    }

    fn visit_expr_materialize(&mut self, inner: Expr, expr: Expr) -> Result<Expr, Self::Error> {
        walk_expr_materialize(self, inner, expr)
    }

    // --- Literals ---

    fn visit_literal_int(&mut self, value: String) -> Result<Literal, Self::Error> {
        Ok(Literal::Int(value))
    }

    fn visit_literal_float(&mut self, value: String) -> Result<Literal, Self::Error> {
        Ok(Literal::Float(value))
    }

    fn visit_literal_bool(&mut self, value: bool) -> Result<Literal, Self::Error> {
        Ok(Literal::Bool(value))
    }

    fn visit_literal_string(&mut self, value: String) -> Result<Literal, Self::Error> {
        Ok(Literal::String(value))
    }

    fn visit_literal_tuple(&mut self, elements: Vec<Expr>) -> Result<Literal, Self::Error> {
        walk_literal_tuple(self, elements)
    }

    fn visit_literal_array(&mut self, elements: Vec<Expr>) -> Result<Literal, Self::Error> {
        walk_literal_array(self, elements)
    }

    // --- Loops ---

    fn visit_loop_kind(&mut self, kind: LoopKind) -> Result<LoopKind, Self::Error> {
        walk_loop_kind(self, kind)
    }

    fn visit_for_loop(&mut self, var: String, iter: Expr) -> Result<LoopKind, Self::Error> {
        walk_for_loop(self, var, iter)
    }

    fn visit_for_range_loop(&mut self, var: String, bound: Expr) -> Result<LoopKind, Self::Error> {
        walk_for_range_loop(self, var, bound)
    }

    fn visit_while_loop(&mut self, cond: Expr) -> Result<LoopKind, Self::Error> {
        walk_while_loop(self, cond)
    }
}

// --- Walk functions: canonical traversal ---

pub fn walk_program<V: MirVisitor>(v: &mut V, p: Program) -> Result<Program, V::Error> {
    let Program {
        defs,
        lambda_registry,
    } = p;

    let defs = defs.into_iter().map(|d| v.visit_def(d)).collect::<Result<Vec<_>, _>>()?;

    Ok(Program {
        defs,
        lambda_registry,
    })
}

pub fn walk_def<V: MirVisitor>(v: &mut V, d: Def) -> Result<Def, V::Error> {
    match d {
        Def::Function {
            name,
            params,
            ret_type,
            attributes,
            param_attributes,
            return_attributes,
            body,
            span,
        } => v.visit_function(
            name,
            params,
            ret_type,
            attributes,
            param_attributes,
            return_attributes,
            body,
            span,
        ),
        Def::Constant {
            name,
            ty,
            attributes,
            body,
            span,
        } => v.visit_constant(name, ty, attributes, body, span),
        Def::Uniform { name, ty } => {
            // Uniforms have no body, just pass through
            Ok(Def::Uniform { name, ty })
        }
    }
}

pub fn walk_function<V: MirVisitor>(
    v: &mut V,
    name: String,
    params: Vec<Param>,
    ret_type: Type<TypeName>,
    attributes: Vec<Attribute>,
    param_attributes: Vec<Vec<Attribute>>,
    return_attributes: Vec<Vec<Attribute>>,
    body: Expr,
    span: Span,
) -> Result<Def, V::Error> {
    let params = params.into_iter().map(|p| v.visit_param(p)).collect::<Result<Vec<_>, _>>()?;

    let ret_type = v.visit_type(ret_type)?;

    let attributes = attributes.into_iter().map(|a| v.visit_attribute(a)).collect::<Result<Vec<_>, _>>()?;

    let param_attributes = param_attributes
        .into_iter()
        .map(|attrs| attrs.into_iter().map(|a| v.visit_attribute(a)).collect::<Result<Vec<_>, _>>())
        .collect::<Result<Vec<_>, _>>()?;

    let return_attributes = return_attributes
        .into_iter()
        .map(|attrs| attrs.into_iter().map(|a| v.visit_attribute(a)).collect::<Result<Vec<_>, _>>())
        .collect::<Result<Vec<_>, _>>()?;

    let body = v.visit_expr(body)?;

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

pub fn walk_constant<V: MirVisitor>(
    v: &mut V,
    name: String,
    ty: Type<TypeName>,
    attributes: Vec<Attribute>,
    body: Expr,
    span: Span,
) -> Result<Def, V::Error> {
    let ty = v.visit_type(ty)?;

    let attributes = attributes.into_iter().map(|a| v.visit_attribute(a)).collect::<Result<Vec<_>, _>>()?;

    let body = v.visit_expr(body)?;

    Ok(Def::Constant {
        name,
        ty,
        attributes,
        body,
        span,
    })
}

pub fn walk_param<V: MirVisitor>(v: &mut V, p: Param) -> Result<Param, V::Error> {
    let Param {
        name,
        ty,
        is_consumed,
    } = p;
    let ty = v.visit_type(ty)?;
    Ok(Param {
        name,
        ty,
        is_consumed,
    })
}

pub fn walk_attribute<V: MirVisitor>(_v: &mut V, a: Attribute) -> Result<Attribute, V::Error> {
    Ok(a)
}

// --- Expressions ---

pub fn walk_expr<V: MirVisitor>(v: &mut V, e: Expr) -> Result<Expr, V::Error> {
    let Expr { ty, kind, span } = e;
    let ty = v.visit_type(ty)?;

    match kind {
        ExprKind::Literal(lit) => {
            let expr = Expr {
                ty,
                kind: ExprKind::Literal(lit.clone()),
                span,
            };
            v.visit_expr_literal(lit, expr)
        }
        ExprKind::Unit => Ok(Expr {
            ty,
            kind: ExprKind::Unit,
            span,
        }),
        ExprKind::Var(ref name) => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(name.clone()),
                span,
            };
            v.visit_expr_var(name.clone(), expr)
        }
        ExprKind::BinOp { op, lhs, rhs } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()), // Dummy kind, won't be used
                span,
            };
            v.visit_expr_bin_op(op, *lhs, *rhs, expr)
        }
        ExprKind::UnaryOp { op, operand } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_unary_op(op, *operand, expr)
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_if(*cond, *then_branch, *else_branch, expr)
        }
        ExprKind::Let { name, value, body } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_let(name, *value, *body, expr)
        }
        ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_loop(loop_var, *init, init_bindings, kind, *body, expr)
        }
        ExprKind::Call { func, args } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_call(func, args, expr)
        }
        ExprKind::Intrinsic { name, args } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_intrinsic(name, args, expr)
        }
        ExprKind::Attributed {
            attributes,
            expr: inner,
        } => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_attributed(attributes, *inner, expr)
        }
        ExprKind::Materialize(inner) => {
            let expr = Expr {
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_materialize(*inner, expr)
        }
    }
}

pub fn walk_expr_bin_op<V: MirVisitor>(
    v: &mut V,
    op: String,
    lhs: Expr,
    rhs: Expr,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let lhs = v.visit_expr(lhs)?;
    let rhs = v.visit_expr(rhs)?;
    Ok(Expr {
        kind: ExprKind::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        ..expr
    })
}

pub fn walk_expr_unary_op<V: MirVisitor>(
    v: &mut V,
    op: String,
    operand: Expr,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let operand = v.visit_expr(operand)?;
    Ok(Expr {
        kind: ExprKind::UnaryOp {
            op,
            operand: Box::new(operand),
        },
        ..expr
    })
}

pub fn walk_expr_if<V: MirVisitor>(
    v: &mut V,
    cond: Expr,
    then_branch: Expr,
    else_branch: Expr,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let cond = v.visit_expr(cond)?;
    let then_branch = v.visit_expr(then_branch)?;
    let else_branch = v.visit_expr(else_branch)?;
    Ok(Expr {
        kind: ExprKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
        ..expr
    })
}

pub fn walk_expr_let<V: MirVisitor>(
    v: &mut V,
    name: String,
    value: Expr,
    body: Expr,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let value = v.visit_expr(value)?;
    let body = v.visit_expr(body)?;
    Ok(Expr {
        kind: ExprKind::Let {
            name,
            value: Box::new(value),
            body: Box::new(body),
        },
        ..expr
    })
}

pub fn walk_expr_loop<V: MirVisitor>(
    v: &mut V,
    loop_var: String,
    init: Expr,
    init_bindings: Vec<(String, Expr)>,
    kind: LoopKind,
    body: Expr,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let init = v.visit_expr(init)?;
    let init_bindings = init_bindings
        .into_iter()
        .map(|(name, e)| Ok((name, v.visit_expr(e)?)))
        .collect::<Result<Vec<_>, _>>()?;

    let kind = v.visit_loop_kind(kind)?;
    let body = v.visit_expr(body)?;

    Ok(Expr {
        kind: ExprKind::Loop {
            loop_var,
            init: Box::new(init),
            init_bindings,
            kind,
            body: Box::new(body),
        },
        ..expr
    })
}

pub fn walk_expr_call<V: MirVisitor>(
    v: &mut V,
    func: String,
    args: Vec<Expr>,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let args = args.into_iter().map(|arg| v.visit_expr(arg)).collect::<Result<Vec<_>, _>>()?;
    Ok(Expr {
        kind: ExprKind::Call { func, args },
        ..expr
    })
}

pub fn walk_expr_intrinsic<V: MirVisitor>(
    v: &mut V,
    name: String,
    args: Vec<Expr>,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let args = args.into_iter().map(|arg| v.visit_expr(arg)).collect::<Result<Vec<_>, _>>()?;
    Ok(Expr {
        kind: ExprKind::Intrinsic { name, args },
        ..expr
    })
}

pub fn walk_expr_attributed<V: MirVisitor>(
    v: &mut V,
    attributes: Vec<Attribute>,
    inner: Expr,
    expr: Expr,
) -> Result<Expr, V::Error> {
    let attributes = attributes.into_iter().map(|a| v.visit_attribute(a)).collect::<Result<Vec<_>, _>>()?;

    let inner = v.visit_expr(inner)?;

    Ok(Expr {
        kind: ExprKind::Attributed {
            attributes,
            expr: Box::new(inner),
        },
        ..expr
    })
}

pub fn walk_expr_materialize<V: MirVisitor>(v: &mut V, inner: Expr, expr: Expr) -> Result<Expr, V::Error> {
    let inner = v.visit_expr(inner)?;
    Ok(Expr {
        kind: ExprKind::Materialize(Box::new(inner)),
        ..expr
    })
}

// --- Literals ---

pub fn walk_literal<V: MirVisitor>(v: &mut V, lit: Literal) -> Result<Literal, V::Error> {
    match lit {
        Literal::Int(s) => v.visit_literal_int(s),
        Literal::Float(s) => v.visit_literal_float(s),
        Literal::Bool(b) => v.visit_literal_bool(b),
        Literal::String(s) => v.visit_literal_string(s),
        Literal::Tuple(elems) => v.visit_literal_tuple(elems),
        Literal::Array(elems) => v.visit_literal_array(elems),
        Literal::Vector(elems) => {
            let elems = elems.into_iter().map(|e| v.visit_expr(e)).collect::<Result<Vec<_>, _>>()?;
            Ok(Literal::Vector(elems))
        }
        Literal::Matrix(rows) => {
            let rows = rows
                .into_iter()
                .map(|row| row.into_iter().map(|e| v.visit_expr(e)).collect::<Result<Vec<_>, _>>())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Literal::Matrix(rows))
        }
    }
}

pub fn walk_literal_tuple<V: MirVisitor>(v: &mut V, elements: Vec<Expr>) -> Result<Literal, V::Error> {
    let elements = elements.into_iter().map(|e| v.visit_expr(e)).collect::<Result<Vec<_>, _>>()?;
    Ok(Literal::Tuple(elements))
}

pub fn walk_literal_array<V: MirVisitor>(v: &mut V, elements: Vec<Expr>) -> Result<Literal, V::Error> {
    let elements = elements.into_iter().map(|e| v.visit_expr(e)).collect::<Result<Vec<_>, _>>()?;
    Ok(Literal::Array(elements))
}

// --- Loop kinds ---

pub fn walk_loop_kind<V: MirVisitor>(v: &mut V, kind: LoopKind) -> Result<LoopKind, V::Error> {
    match kind {
        LoopKind::For { var, iter } => v.visit_for_loop(var, *iter),
        LoopKind::ForRange { var, bound } => v.visit_for_range_loop(var, *bound),
        LoopKind::While { cond } => v.visit_while_loop(*cond),
    }
}

pub fn walk_for_loop<V: MirVisitor>(v: &mut V, var: String, iter: Expr) -> Result<LoopKind, V::Error> {
    let iter = v.visit_expr(iter)?;
    Ok(LoopKind::For {
        var,
        iter: Box::new(iter),
    })
}

pub fn walk_for_range_loop<V: MirVisitor>(
    v: &mut V,
    var: String,
    bound: Expr,
) -> Result<LoopKind, V::Error> {
    let bound = v.visit_expr(bound)?;
    Ok(LoopKind::ForRange {
        var,
        bound: Box::new(bound),
    })
}

pub fn walk_while_loop<V: MirVisitor>(v: &mut V, cond: Expr) -> Result<LoopKind, V::Error> {
    let cond = v.visit_expr(cond)?;
    Ok(LoopKind::While { cond: Box::new(cond) })
}
