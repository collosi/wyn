use crate::IdArena;
use crate::ast::{NodeCounter, Span, TypeName};
use polytype::Type;

use super::*;

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn f32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn test_span() -> Span {
    Span::new(1, 1, 1, 1)
}

#[test]
fn test_simple_function() {
    let mut nc = NodeCounter::new();
    let span = test_span();

    // Represents: def add(x, y) = x + y
    let add_fn = Def::Function {
        id: nc.next(),
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
        body: Expr::new(
            nc.next(),
            i32_type(),
            ExprKind::BinOp {
                op: "+".to_string(),
                lhs: Box::new(Expr::new(
                    nc.next(),
                    i32_type(),
                    ExprKind::Var("x".to_string()),
                    span,
                )),
                rhs: Box::new(Expr::new(
                    nc.next(),
                    i32_type(),
                    ExprKind::Var("y".to_string()),
                    span,
                )),
            },
            span,
        ),
        span,
    };

    let program = Program {
        defs: vec![add_fn],
        lambda_registry: IdArena::new(),
    };

    assert_eq!(program.defs.len(), 1);
    match &program.defs[0] {
        Def::Function { name, .. } => assert_eq!(name, "add"),
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_constant() {
    let mut nc = NodeCounter::new();
    let span = test_span();

    // Represents: def pi = 3.14159
    let pi_const = Def::Constant {
        id: nc.next(),
        name: "pi".to_string(),
        ty: f32_type(),
        attributes: vec![],
        body: Expr::new(
            nc.next(),
            f32_type(),
            ExprKind::Literal(Literal::Float("3.14159".to_string())),
            span,
        ),
        span,
    };

    match pi_const {
        Def::Constant { name, .. } => assert_eq!(name, "pi"),
        _ => panic!("Expected Constant"),
    }
}
