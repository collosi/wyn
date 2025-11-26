#![cfg(test)]

use crate::ast::Span;
use crate::constant_folding::ConstantFolder;
use crate::mir::{Expr, ExprKind, Literal};

fn i32_type() -> polytype::Type<crate::ast::TypeName> {
    polytype::Type::Constructed(crate::ast::TypeName::Int(32), vec![])
}

fn f32_type() -> polytype::Type<crate::ast::TypeName> {
    polytype::Type::Constructed(crate::ast::TypeName::Float(32), vec![])
}

fn bool_type() -> polytype::Type<crate::ast::TypeName> {
    polytype::Type::Constructed(crate::ast::TypeName::Str("bool".into()), vec![])
}

#[test]
fn test_fold_float_division() {
    let mut folder = ConstantFolder::new();

    // 135.0 / 255.0
    let expr = Expr::new(
        f32_type(),
        ExprKind::BinOp {
            op: "/".to_string(),
            lhs: Box::new(Expr::new(
                f32_type(),
                ExprKind::Literal(Literal::Float("135.0".to_string())),
                Span::dummy(),
            )),
            rhs: Box::new(Expr::new(
                f32_type(),
                ExprKind::Literal(Literal::Float("255.0".to_string())),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Float(val)) => {
            let v: f64 = val.parse().unwrap();
            assert!((v - 0.529411765).abs() < 0.000001);
        }
        _ => panic!("Expected folded float literal, got {:?}", result),
    }
}

#[test]
fn test_fold_integer_addition() {
    let mut folder = ConstantFolder::new();

    // 10 + 32
    let expr = Expr::new(
        i32_type(),
        ExprKind::BinOp {
            op: "+".to_string(),
            lhs: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("10".to_string())),
                Span::dummy(),
            )),
            rhs: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("32".to_string())),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Int(val)) => {
            assert_eq!(val, "42");
        }
        _ => panic!("Expected folded int literal, got {:?}", result),
    }
}

#[test]
fn test_fold_constant_if_true() {
    let mut folder = ConstantFolder::new();

    // if true then 1 else 2
    let expr = Expr::new(
        i32_type(),
        ExprKind::If {
            cond: Box::new(Expr::new(
                bool_type(),
                ExprKind::Literal(Literal::Bool(true)),
                Span::dummy(),
            )),
            then_branch: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("1".to_string())),
                Span::dummy(),
            )),
            else_branch: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("2".to_string())),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Int(val)) => {
            assert_eq!(val, "1");
        }
        _ => panic!("Expected folded to then branch, got {:?}", result),
    }
}

#[test]
fn test_fold_constant_if_false() {
    let mut folder = ConstantFolder::new();

    // if false then 1 else 2
    let expr = Expr::new(
        i32_type(),
        ExprKind::If {
            cond: Box::new(Expr::new(
                bool_type(),
                ExprKind::Literal(Literal::Bool(false)),
                Span::dummy(),
            )),
            then_branch: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("1".to_string())),
                Span::dummy(),
            )),
            else_branch: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("2".to_string())),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Int(val)) => {
            assert_eq!(val, "2");
        }
        _ => panic!("Expected folded to else branch, got {:?}", result),
    }
}

#[test]
fn test_fold_array_literal() {
    let mut folder = ConstantFolder::new();

    // [1 + 2, 3 * 4]
    let expr = Expr::new(
        i32_type(), // simplified, would be array type
        ExprKind::Literal(Literal::Array(vec![
            Expr::new(
                i32_type(),
                ExprKind::BinOp {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::new(
                        i32_type(),
                        ExprKind::Literal(Literal::Int("1".to_string())),
                        Span::dummy(),
                    )),
                    rhs: Box::new(Expr::new(
                        i32_type(),
                        ExprKind::Literal(Literal::Int("2".to_string())),
                        Span::dummy(),
                    )),
                },
                Span::dummy(),
            ),
            Expr::new(
                i32_type(),
                ExprKind::BinOp {
                    op: "*".to_string(),
                    lhs: Box::new(Expr::new(
                        i32_type(),
                        ExprKind::Literal(Literal::Int("3".to_string())),
                        Span::dummy(),
                    )),
                    rhs: Box::new(Expr::new(
                        i32_type(),
                        ExprKind::Literal(Literal::Int("4".to_string())),
                        Span::dummy(),
                    )),
                },
                Span::dummy(),
            ),
        ])),
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Array(elements)) => {
            assert_eq!(elements.len(), 2);
            match &elements[0].kind {
                ExprKind::Literal(Literal::Int(v)) => assert_eq!(v, "3"),
                _ => panic!("Expected int literal"),
            }
            match &elements[1].kind {
                ExprKind::Literal(Literal::Int(v)) => assert_eq!(v, "12"),
                _ => panic!("Expected int literal"),
            }
        }
        _ => panic!("Expected array literal, got {:?}", result),
    }
}

#[test]
fn test_fold_negation() {
    let mut folder = ConstantFolder::new();

    // -42
    let expr = Expr::new(
        i32_type(),
        ExprKind::UnaryOp {
            op: "-".to_string(),
            operand: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("42".to_string())),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Int(val)) => {
            assert_eq!(val, "-42");
        }
        _ => panic!("Expected folded negation, got {:?}", result),
    }
}

#[test]
fn test_fold_boolean_not() {
    let mut folder = ConstantFolder::new();

    // !true
    let expr = Expr::new(
        bool_type(),
        ExprKind::UnaryOp {
            op: "!".to_string(),
            operand: Box::new(Expr::new(
                bool_type(),
                ExprKind::Literal(Literal::Bool(true)),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Bool(val)) => {
            assert!(!*val);
        }
        _ => panic!("Expected folded boolean, got {:?}", result),
    }
}

#[test]
fn test_fold_comparison() {
    let mut folder = ConstantFolder::new();

    // 10 < 20
    let expr = Expr::new(
        bool_type(),
        ExprKind::BinOp {
            op: "<".to_string(),
            lhs: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("10".to_string())),
                Span::dummy(),
            )),
            rhs: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("20".to_string())),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    match &result.kind {
        ExprKind::Literal(Literal::Bool(val)) => {
            assert!(*val);
        }
        _ => panic!("Expected folded comparison, got {:?}", result),
    }
}

#[test]
fn test_no_fold_with_variable() {
    let mut folder = ConstantFolder::new();

    // x + 1 (should not fold since x is a variable)
    let expr = Expr::new(
        i32_type(),
        ExprKind::BinOp {
            op: "+".to_string(),
            lhs: Box::new(Expr::new(
                i32_type(),
                ExprKind::Var("x".to_string()),
                Span::dummy(),
            )),
            rhs: Box::new(Expr::new(
                i32_type(),
                ExprKind::Literal(Literal::Int("1".to_string())),
                Span::dummy(),
            )),
        },
        Span::dummy(),
    );

    let result = folder.fold_expr(&expr).unwrap();

    // Should remain a BinOp since we can't fold with a variable
    match &result.kind {
        ExprKind::BinOp { op, .. } => {
            assert_eq!(op, "+");
        }
        _ => panic!("Expected unfoldable binop, got {:?}", result),
    }
}
