use crate::ast::{Span, TypeName};
use polytype::Type;

use super::*;

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Str("i32".into()), vec![])
}

fn f32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Str("f32".into()), vec![])
}

#[test]
fn test_simple_function() {
    // Represents: def add(x, y) = x + y
    let add_fn = Def::Function {
        name: "add".to_string(),
        params: vec![
            Param {
                name: "x".to_string(),
                ty: i32_type(),
                is_consumed: false,
            },
            Param {
                name: "y".to_string(),
                ty: i32_type(),
                is_consumed: false,
            },
        ],
        ret_type: i32_type(),
        attributes: vec![],
        param_attributes: vec![],
        return_attributes: vec![],
        body: Expr::new(
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
                    ExprKind::Var("y".to_string()),
                    Span::dummy(),
                )),
            },
            Span::dummy(),
        ),
        span: Span::dummy(),
    };

    let program = Program {
        defs: vec![add_fn],
        lambda_registry: vec![],
    };

    assert_eq!(program.defs.len(), 1);
    match &program.defs[0] {
        Def::Function { name, .. } => assert_eq!(name, "add"),
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_constant() {
    // Represents: def pi = 3.14159
    let pi_const = Def::Constant {
        name: "pi".to_string(),
        ty: f32_type(),
        attributes: vec![],
        body: Expr::new(
            f32_type(),
            ExprKind::Literal(Literal::Float("3.14159".to_string())),
            Span::dummy(),
        ),
        span: Span::dummy(),
    };

    match pi_const {
        Def::Constant { name, .. } => assert_eq!(name, "pi"),
        _ => panic!("Expected Constant"),
    }
}
