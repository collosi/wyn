//! MIR visitor pattern for traversing the Wyn MIR.
//!
//! This module provides a centralized traversal mechanism for the MIR.
//! Each pass (optimization, lowering, analysis, etc.) can implement the
//! `MirVisitor` trait and override only the hooks they need, while the
//! `walk_*` functions handle the actual tree traversal.

use crate::mir::*;
use crate::ast::{Span, TypeName};
use polytype::Type;
use std::ops::ControlFlow;

/// Visitor trait for traversing the MIR.
///
/// All methods have default implementations that delegate to `walk_*` functions.
/// Implementors can override specific hooks to customize behavior.
///
/// The `Break` associated type allows visitors to return errors or other data
/// when they need to short-circuit traversal.
pub trait MirVisitor: Sized {
    type Break;

    // --- Top-level program ---

    fn visit_program(&mut self, p: &Program) -> ControlFlow<Self::Break> {
        walk_program(self, p)
    }

    fn visit_def(&mut self, d: &Def) -> ControlFlow<Self::Break> {
        walk_def(self, d)
    }

    fn visit_function(
        &mut self,
        name: &str,
        params: &[Param],
        ret_type: &Type<TypeName>,
        attributes: &[Attribute],
        param_attributes: &[Vec<Attribute>],
        return_attributes: &[Vec<Attribute>],
        body: &Expr,
        span: &Span,
    ) -> ControlFlow<Self::Break> {
        // Default traversal ignores `name` and `span`.
        let _ = (name, span);
        walk_function(self, params, ret_type, attributes, param_attributes, return_attributes, body)
    }

    fn visit_constant(
        &mut self,
        name: &str,
        ty: &Type<TypeName>,
        attributes: &[Attribute],
        body: &Expr,
        span: &Span,
    ) -> ControlFlow<Self::Break> {
        // Default traversal ignores `name` and `span`.
        let _ = (name, span);
        walk_constant(self, ty, attributes, body)
    }

    fn visit_param(&mut self, p: &Param) -> ControlFlow<Self::Break> {
        walk_param(self, p)
    }

    fn visit_attribute(&mut self, a: &Attribute) -> ControlFlow<Self::Break> {
        walk_attribute(self, a)
    }

    fn visit_attribute_arg(&mut self, _arg: &str) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_type(&mut self, _ty: &Type<TypeName>) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_lambda_registry_entry(
        &mut self,
        _name: &str,
        _arity: usize,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    // --- Expressions ---

    fn visit_expr(&mut self, e: &Expr) -> ControlFlow<Self::Break> {
        walk_expr(self, e)
    }

    fn visit_expr_literal(&mut self, lit: &Literal, _expr: &Expr) -> ControlFlow<Self::Break> {
        walk_literal(self, lit)
    }

    fn visit_expr_var(&mut self, _name: &str, _expr: &Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_bin_op(
        &mut self,
        _op: &str,
        lhs: &Expr,
        rhs: &Expr,
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_bin_op(self, lhs, rhs)
    }

    fn visit_expr_unary_op(
        &mut self,
        _op: &str,
        operand: &Expr,
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_unary_op(self, operand)
    }

    fn visit_expr_if(
        &mut self,
        cond: &Expr,
        then_branch: &Expr,
        else_branch: &Expr,
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_if(self, cond, then_branch, else_branch)
    }

    fn visit_expr_let(
        &mut self,
        _name: &str,
        value: &Expr,
        body: &Expr,
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_let(self, value, body)
    }

    fn visit_expr_loop(
        &mut self,
        init_bindings: &[(String, Expr)],
        kind: &LoopKind,
        body: &Expr,
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_loop(self, init_bindings, kind, body)
    }

    fn visit_expr_call(
        &mut self,
        _func: &str,
        args: &[Expr],
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_call(self, args)
    }

    fn visit_expr_intrinsic(
        &mut self,
        _name: &str,
        args: &[Expr],
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_intrinsic(self, args)
    }

    fn visit_expr_attributed(
        &mut self,
        attributes: &[Attribute],
        inner: &Expr,
        _expr: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_expr_attributed(self, attributes, inner)
    }

    // --- Literals ---

    fn visit_literal_int(&mut self, _value: &str) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_literal_float(&mut self, _value: &str) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_literal_bool(&mut self, _value: bool) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_literal_string(&mut self, _value: &str) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_literal_tuple(&mut self, elements: &[Expr]) -> ControlFlow<Self::Break> {
        walk_literal_tuple(self, elements)
    }

    fn visit_literal_array(&mut self, elements: &[Expr]) -> ControlFlow<Self::Break> {
        walk_literal_array(self, elements)
    }

    fn visit_literal_record(
        &mut self,
        fields: &[(String, Expr)],
    ) -> ControlFlow<Self::Break> {
        walk_literal_record(self, fields)
    }

    // --- Loops ---

    fn visit_loop_kind(&mut self, kind: &LoopKind) -> ControlFlow<Self::Break> {
        walk_loop_kind(self, kind)
    }

    fn visit_for_loop(&mut self, _var: &str, iter: &Expr) -> ControlFlow<Self::Break> {
        walk_for_loop(self, iter)
    }

    fn visit_for_range_loop(
        &mut self,
        _var: &str,
        bound: &Expr,
    ) -> ControlFlow<Self::Break> {
        walk_for_range_loop(self, bound)
    }

    fn visit_while_loop(&mut self, cond: &Expr) -> ControlFlow<Self::Break> {
        walk_while_loop(self, cond)
    }
}

// --- Walk functions: canonical traversal ---

pub fn walk_program<V: MirVisitor>(v: &mut V, p: &Program) -> ControlFlow<V::Break> {
    for def in &p.defs {
        v.visit_def(def)?;
    }

    for (name, arity) in &p.lambda_registry {
        v.visit_lambda_registry_entry(name, *arity)?;
    }

    ControlFlow::Continue(())
}

pub fn walk_def<V: MirVisitor>(v: &mut V, d: &Def) -> ControlFlow<V::Break> {
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
    }
}

pub fn walk_function<V: MirVisitor>(
    v: &mut V,
    params: &[Param],
    ret_type: &Type<TypeName>,
    attributes: &[Attribute],
    param_attributes: &[Vec<Attribute>],
    return_attributes: &[Vec<Attribute>],
    body: &Expr,
) -> ControlFlow<V::Break> {
    for p in params {
        v.visit_param(p)?;
    }

    v.visit_type(ret_type)?;

    for attr in attributes {
        v.visit_attribute(attr)?;
    }

    for attrs in param_attributes {
        for attr in attrs {
            v.visit_attribute(attr)?;
        }
    }

    for attrs in return_attributes {
        for attr in attrs {
            v.visit_attribute(attr)?;
        }
    }

    v.visit_expr(body)
}

pub fn walk_constant<V: MirVisitor>(
    v: &mut V,
    ty: &Type<TypeName>,
    attributes: &[Attribute],
    body: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_type(ty)?;

    for attr in attributes {
        v.visit_attribute(attr)?;
    }

    v.visit_expr(body)
}

pub fn walk_param<V: MirVisitor>(v: &mut V, p: &Param) -> ControlFlow<V::Break> {
    v.visit_type(&p.ty)
}

pub fn walk_attribute<V: MirVisitor>(v: &mut V, a: &Attribute) -> ControlFlow<V::Break> {
    for arg in &a.args {
        v.visit_attribute_arg(arg)?;
    }
    ControlFlow::Continue(())
}

// --- Expressions ---

pub fn walk_expr<V: MirVisitor>(v: &mut V, e: &Expr) -> ControlFlow<V::Break> {
    match &e.kind {
        ExprKind::Literal(lit) => v.visit_expr_literal(lit, e),
        ExprKind::Var(name) => v.visit_expr_var(name, e),
        ExprKind::BinOp { op, lhs, rhs } => {
            v.visit_expr_bin_op(op, lhs, rhs, e)
        }
        ExprKind::UnaryOp { op, operand } => {
            v.visit_expr_unary_op(op, operand, e)
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => v.visit_expr_if(cond, then_branch, else_branch, e),
        ExprKind::Let { name, value, body } => {
            v.visit_expr_let(name, value, body, e)
        }
        ExprKind::Loop {
            init_bindings,
            kind,
            body,
        } => v.visit_expr_loop(init_bindings, kind, body, e),
        ExprKind::Call { func, args } => v.visit_expr_call(func, args, e),
        ExprKind::Intrinsic { name, args } => {
            v.visit_expr_intrinsic(name, args, e)
        }
        ExprKind::Attributed { attributes, expr } => {
            v.visit_expr_attributed(attributes, expr, e)
        }
    }
}

pub fn walk_expr_bin_op<V: MirVisitor>(
    v: &mut V,
    lhs: &Expr,
    rhs: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_expr(lhs)?;
    v.visit_expr(rhs)
}

pub fn walk_expr_unary_op<V: MirVisitor>(
    v: &mut V,
    operand: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_expr(operand)
}

pub fn walk_expr_if<V: MirVisitor>(
    v: &mut V,
    cond: &Expr,
    then_branch: &Expr,
    else_branch: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_expr(cond)?;
    v.visit_expr(then_branch)?;
    v.visit_expr(else_branch)
}

pub fn walk_expr_let<V: MirVisitor>(
    v: &mut V,
    value: &Expr,
    body: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_expr(value)?;
    v.visit_expr(body)
}

pub fn walk_expr_loop<V: MirVisitor>(
    v: &mut V,
    init_bindings: &[(String, Expr)],
    kind: &LoopKind,
    body: &Expr,
) -> ControlFlow<V::Break> {
    for (_name, init_expr) in init_bindings {
        v.visit_expr(init_expr)?;
    }

    v.visit_loop_kind(kind)?;
    v.visit_expr(body)
}

pub fn walk_expr_call<V: MirVisitor>(
    v: &mut V,
    args: &[Expr],
) -> ControlFlow<V::Break> {
    for arg in args {
        v.visit_expr(arg)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_intrinsic<V: MirVisitor>(
    v: &mut V,
    args: &[Expr],
) -> ControlFlow<V::Break> {
    for arg in args {
        v.visit_expr(arg)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_attributed<V: MirVisitor>(
    v: &mut V,
    attributes: &[Attribute],
    inner: &Expr,
) -> ControlFlow<V::Break> {
    for attr in attributes {
        v.visit_attribute(attr)?;
    }

    v.visit_expr(inner)
}

// --- Literals ---

pub fn walk_literal<V: MirVisitor>(
    v: &mut V,
    lit: &Literal,
) -> ControlFlow<V::Break> {
    match lit {
        Literal::Int(s) => v.visit_literal_int(s),
        Literal::Float(s) => v.visit_literal_float(s),
        Literal::Bool(b) => v.visit_literal_bool(*b),
        Literal::String(s) => v.visit_literal_string(s),
        Literal::Tuple(elems) => v.visit_literal_tuple(elems),
        Literal::Array(elems) => v.visit_literal_array(elems),
        Literal::Record(fields) => v.visit_literal_record(fields),
    }
}

pub fn walk_literal_tuple<V: MirVisitor>(
    v: &mut V,
    elements: &[Expr],
) -> ControlFlow<V::Break> {
    for e in elements {
        v.visit_expr(e)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_literal_array<V: MirVisitor>(
    v: &mut V,
    elements: &[Expr],
) -> ControlFlow<V::Break> {
    for e in elements {
        v.visit_expr(e)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_literal_record<V: MirVisitor>(
    v: &mut V,
    fields: &[(String, Expr)],
) -> ControlFlow<V::Break> {
    for (_name, expr) in fields {
        v.visit_expr(expr)?;
    }
    ControlFlow::Continue(())
}

// --- Loop kinds ---

pub fn walk_loop_kind<V: MirVisitor>(
    v: &mut V,
    kind: &LoopKind,
) -> ControlFlow<V::Break> {
    match kind {
        LoopKind::For { var, iter } => v.visit_for_loop(var, iter),
        LoopKind::ForRange { var, bound } => v.visit_for_range_loop(var, bound),
        LoopKind::While { cond } => v.visit_while_loop(cond),
    }
}

pub fn walk_for_loop<V: MirVisitor>(
    v: &mut V,
    iter: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_expr(iter)
}

pub fn walk_for_range_loop<V: MirVisitor>(
    v: &mut V,
    bound: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_expr(bound)
}

pub fn walk_while_loop<V: MirVisitor>(
    v: &mut V,
    cond: &Expr,
) -> ControlFlow<V::Break> {
    v.visit_expr(cond)
}
