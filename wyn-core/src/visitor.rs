//! AST visitor pattern for traversing the Wyn syntax tree
//!
//! This module provides a centralized traversal mechanism for the AST.
//! Each pass (type checking, defunctionalization, etc.) can implement the
//! Visitor trait and override only the hooks they need, while the walk_*
//! functions handle the actual tree traversal.

use crate::ast::*;
use std::ops::ControlFlow;

/// Visitor trait for traversing the AST
///
/// All methods have default implementations that delegate to walk_* functions.
/// Implementors can override specific hooks to customize behavior.
///
/// The Break associated type allows visitors to return errors or other data
/// when they need to short-circuit traversal.
pub trait Visitor: Sized {
    type Break;

    // --- Top-level program ---
    fn visit_program(&mut self, p: &Program) -> ControlFlow<Self::Break> {
        walk_program(self, p)
    }

    fn visit_declaration(&mut self, d: &Declaration) -> ControlFlow<Self::Break> {
        walk_declaration(self, d)
    }

    // --- Declarations ---
    fn visit_decl(&mut self, d: &Decl) -> ControlFlow<Self::Break> {
        walk_decl(self, d)
    }

    fn visit_uniform_decl(&mut self, u: &UniformDecl) -> ControlFlow<Self::Break> {
        walk_uniform_decl(self, u)
    }

    fn visit_val_decl(&mut self, v: &ValDecl) -> ControlFlow<Self::Break> {
        walk_val_decl(self, v)
    }

    // --- Expressions ---
    fn visit_expression(&mut self, e: &Expression) -> ControlFlow<Self::Break> {
        walk_expression(self, e)
    }

    fn visit_expr_int_literal(&mut self, _n: i32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_float_literal(&mut self, _f: f32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_identifier(&mut self, _name: &str) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_array_literal(&mut self, elements: &[Expression]) -> ControlFlow<Self::Break> {
        walk_expr_array_literal(self, elements)
    }

    fn visit_expr_array_index(
        &mut self,
        array: &Expression,
        index: &Expression,
    ) -> ControlFlow<Self::Break> {
        walk_expr_array_index(self, array, index)
    }

    fn visit_expr_binary_op(
        &mut self,
        _op: &BinaryOp,
        left: &Expression,
        right: &Expression,
    ) -> ControlFlow<Self::Break> {
        walk_expr_binary_op(self, left, right)
    }

    fn visit_expr_function_call(&mut self, _name: &str, args: &[Expression]) -> ControlFlow<Self::Break> {
        walk_expr_function_call(self, args)
    }

    fn visit_expr_tuple(&mut self, elements: &[Expression]) -> ControlFlow<Self::Break> {
        walk_expr_tuple(self, elements)
    }

    fn visit_expr_lambda(&mut self, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        walk_expr_lambda(self, lambda)
    }

    fn visit_expr_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        walk_expr_application(self, func, args)
    }

    fn visit_expr_let_in(&mut self, let_in: &LetInExpr) -> ControlFlow<Self::Break> {
        walk_expr_let_in(self, let_in)
    }

    fn visit_expr_field_access(&mut self, expr: &Expression, _field: &str) -> ControlFlow<Self::Break> {
        walk_expr_field_access(self, expr)
    }

    fn visit_expr_if(&mut self, if_expr: &IfExpr) -> ControlFlow<Self::Break> {
        walk_expr_if(self, if_expr)
    }

    // --- Types ---
    fn visit_type(&mut self, _t: &Type) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_parameter(&mut self, p: &Parameter) -> ControlFlow<Self::Break> {
        walk_parameter(self, p)
    }
}

// --- Walk functions: canonical traversal ---

pub fn walk_program<V: Visitor>(v: &mut V, p: &Program) -> ControlFlow<V::Break> {
    for decl in &p.declarations {
        v.visit_declaration(decl)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_declaration<V: Visitor>(v: &mut V, d: &Declaration) -> ControlFlow<V::Break> {
    match d {
        Declaration::Decl(decl) => v.visit_decl(decl),
        Declaration::Uniform(uniform) => v.visit_uniform_decl(uniform),
        Declaration::Val(val) => v.visit_val_decl(val),
        Declaration::TypeBind(_) => {
            unimplemented!("Type bindings are not yet supported in visitor")
        }
        Declaration::ModuleBind(_) => {
            unimplemented!("Module bindings are not yet supported in visitor")
        }
        Declaration::ModuleTypeBind(_) => {
            unimplemented!("Module type bindings are not yet supported in visitor")
        }
        Declaration::Open(_) => {
            unimplemented!("Open declarations are not yet supported in visitor")
        }
        Declaration::Import(_) => {
            unimplemented!("Import declarations are not yet supported in visitor")
        }
        Declaration::Local(_) => {
            unimplemented!("Local declarations are not yet supported in visitor")
        }
    }
}

pub fn walk_decl<V: Visitor>(v: &mut V, d: &Decl) -> ControlFlow<V::Break> {
    // Visit parameters
    for param in &d.params {
        if let DeclParam::Typed(p) = param {
            v.visit_parameter(p)?;
        }
    }

    // Visit type annotation if present
    if let Some(ty) = &d.ty {
        v.visit_type(ty)?;
    }

    // Visit attributed return type if present
    if let Some(attr_types) = &d.attributed_return_type {
        for attr_type in attr_types {
            v.visit_type(&attr_type.ty)?;
        }
    }

    // Visit body
    v.visit_expression(&d.body)
}

pub fn walk_uniform_decl<V: Visitor>(v: &mut V, u: &UniformDecl) -> ControlFlow<V::Break> {
    v.visit_type(&u.ty)
}

pub fn walk_val_decl<V: Visitor>(v: &mut V, val: &ValDecl) -> ControlFlow<V::Break> {
    v.visit_type(&val.ty)
}

pub fn walk_parameter<V: Visitor>(v: &mut V, p: &Parameter) -> ControlFlow<V::Break> {
    v.visit_type(&p.ty)
}

pub fn walk_expression<V: Visitor>(v: &mut V, e: &Expression) -> ControlFlow<V::Break> {
    match &e.kind {
        ExprKind::IntLiteral(n) => v.visit_expr_int_literal(*n),
        ExprKind::FloatLiteral(f) => v.visit_expr_float_literal(*f),
        ExprKind::Identifier(name) => v.visit_expr_identifier(name),
        ExprKind::ArrayLiteral(elements) => v.visit_expr_array_literal(elements),
        ExprKind::ArrayIndex(array, index) => v.visit_expr_array_index(array, index),
        ExprKind::BinaryOp(op, left, right) => v.visit_expr_binary_op(op, left, right),
        ExprKind::FunctionCall(name, args) => v.visit_expr_function_call(name, args),
        ExprKind::Tuple(elements) => v.visit_expr_tuple(elements),
        ExprKind::Lambda(lambda) => v.visit_expr_lambda(lambda),
        ExprKind::Application(func, args) => v.visit_expr_application(func, args),
        ExprKind::LetIn(let_in) => v.visit_expr_let_in(let_in),
        ExprKind::FieldAccess(expr, field) => v.visit_expr_field_access(expr, field),
        ExprKind::If(if_expr) => v.visit_expr_if(if_expr),

        ExprKind::TypeHole => ControlFlow::Continue(()),
        ExprKind::QualifiedName(_, _)
        | ExprKind::UnaryOp(_, _)
        | ExprKind::Loop(_)
        | ExprKind::Match(_)
        | ExprKind::Range(_)
        | ExprKind::Pipe(_, _)
        | ExprKind::TypeAscription(_, _)
        | ExprKind::TypeCoercion(_, _)
        | ExprKind::Unsafe(_)
        | ExprKind::Assert(_, _) => {
            todo!("New expression kinds not yet implemented in visitor")
        }
    } // NEWCASESHERE - add new cases before this closing brace
}

pub fn walk_expr_array_literal<V: Visitor>(v: &mut V, elements: &[Expression]) -> ControlFlow<V::Break> {
    for elem in elements {
        v.visit_expression(elem)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_array_index<V: Visitor>(
    v: &mut V,
    array: &Expression,
    index: &Expression,
) -> ControlFlow<V::Break> {
    v.visit_expression(array)?;
    v.visit_expression(index)
}

pub fn walk_expr_binary_op<V: Visitor>(
    v: &mut V,
    left: &Expression,
    right: &Expression,
) -> ControlFlow<V::Break> {
    v.visit_expression(left)?;
    v.visit_expression(right)
}

pub fn walk_expr_function_call<V: Visitor>(v: &mut V, args: &[Expression]) -> ControlFlow<V::Break> {
    for arg in args {
        v.visit_expression(arg)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_tuple<V: Visitor>(v: &mut V, elements: &[Expression]) -> ControlFlow<V::Break> {
    for elem in elements {
        v.visit_expression(elem)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_lambda<V: Visitor>(v: &mut V, lambda: &LambdaExpr) -> ControlFlow<V::Break> {
    // Visit parameter types
    for param in &lambda.params {
        if let Some(ty) = &param.ty {
            v.visit_type(ty)?;
        }
    }

    // Visit return type
    if let Some(ty) = &lambda.return_type {
        v.visit_type(ty)?;
    }

    // Visit body
    v.visit_expression(&lambda.body)
}

pub fn walk_expr_application<V: Visitor>(
    v: &mut V,
    func: &Expression,
    args: &[Expression],
) -> ControlFlow<V::Break> {
    v.visit_expression(func)?;
    for arg in args {
        v.visit_expression(arg)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_let_in<V: Visitor>(v: &mut V, let_in: &LetInExpr) -> ControlFlow<V::Break> {
    // Visit type annotation if present
    if let Some(ty) = &let_in.ty {
        v.visit_type(ty)?;
    }

    // Visit value
    v.visit_expression(&let_in.value)?;

    // Visit body
    v.visit_expression(&let_in.body)
}

pub fn walk_expr_field_access<V: Visitor>(v: &mut V, expr: &Expression) -> ControlFlow<V::Break> {
    v.visit_expression(expr)
}

pub fn walk_expr_if<V: Visitor>(v: &mut V, if_expr: &IfExpr) -> ControlFlow<V::Break> {
    v.visit_expression(&if_expr.condition)?;
    v.visit_expression(&if_expr.then_branch)?;
    v.visit_expression(&if_expr.else_branch)
}
