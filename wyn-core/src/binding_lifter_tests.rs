//! Tests for the binding lifting (code motion) optimization pass.

use crate::ast::{Span, TypeName};
use crate::binding_lifter::BindingLifter;
use crate::mir::{Expr, ExprKind, Literal, LoopKind};
use polytype::Type;

// =============================================================================
// Test Helpers - Type Constructors
// =============================================================================

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

// =============================================================================
// Test Helpers - Expression Constructors
// =============================================================================

fn var(name: &str, ty: Type<TypeName>) -> Expr {
    Expr::new(ty, ExprKind::Var(name.to_string()), Span::dummy())
}

fn int_lit(n: i32) -> Expr {
    Expr::new(
        i32_type(),
        ExprKind::Literal(Literal::Int(n.to_string())),
        Span::dummy(),
    )
}

fn binop(op: &str, lhs: Expr, rhs: Expr) -> Expr {
    let ty = lhs.ty.clone();
    Expr::new(
        ty,
        ExprKind::BinOp {
            op: op.to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        Span::dummy(),
    )
}

fn let_bind(name: &str, value: Expr, body: Expr) -> Expr {
    let ty = body.ty.clone();
    Expr::new(
        ty,
        ExprKind::Let {
            name: name.to_string(),
            binding_id: 0,
            value: Box::new(value),
            body: Box::new(body),
        },
        Span::dummy(),
    )
}

/// Create a for-range loop: `loop loop_var = init for iter_var < bound do body`
fn for_range_loop(loop_var: &str, init: Expr, iter_var: &str, bound: Expr, body: Expr) -> Expr {
    let ty = body.ty.clone();
    let init_ty = init.ty.clone();
    Expr::new(
        ty,
        ExprKind::Loop {
            loop_var: loop_var.to_string(),
            init: Box::new(init),
            init_bindings: vec![(loop_var.to_string(), var(loop_var, init_ty))],
            kind: LoopKind::ForRange {
                var: iter_var.to_string(),
                bound: Box::new(bound),
            },
            body: Box::new(body),
        },
        Span::dummy(),
    )
}

// =============================================================================
// Test Helpers - Structure Inspection
// =============================================================================

/// Get a simplified structural representation of an expression.
/// Returns a vector of tags like ["let", "loop", "let", "expr"].
fn get_structure(expr: &Expr) -> Vec<&'static str> {
    let mut result = vec![];
    collect_structure(expr, &mut result);
    result
}

fn collect_structure(expr: &Expr, result: &mut Vec<&'static str>) {
    match &expr.kind {
        ExprKind::Let { body, .. } => {
            result.push("let");
            collect_structure(body, result);
        }
        ExprKind::Loop { body, .. } => {
            result.push("loop");
            collect_structure(body, result);
        }
        _ => {
            result.push("expr");
        }
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_hoist_constant_from_loop() {
    // loop acc = 0 for i < 10 do
    //   let x = 42 in acc + x
    // =>
    // let x = 42 in loop acc = 0 for i < 10 do acc + x

    let body = let_bind(
        "x",
        int_lit(42),
        binop("+", var("acc", i32_type()), var("x", i32_type())),
    );
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    assert_eq!(structure, vec!["let", "loop", "expr"]); // x hoisted before loop
}

#[test]
fn test_no_hoist_loop_dependent() {
    // loop acc = 0 for i < 10 do
    //   let x = acc + 1 in x * 2
    // => unchanged (x depends on acc)

    let body = let_bind(
        "x",
        binop("+", var("acc", i32_type()), int_lit(1)),
        binop("*", var("x", i32_type()), int_lit(2)),
    );
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    assert_eq!(structure, vec!["loop", "let", "expr"]); // x stays in loop
}

#[test]
fn test_hoist_chain_of_invariants() {
    // loop acc = 0 for i < 10 do
    //   let x = 1 in
    //   let y = x + 2 in
    //   acc + y
    // =>
    // let x = 1 in let y = x + 2 in loop acc = 0 for i < 10 do acc + y

    let body = let_bind(
        "x",
        int_lit(1),
        let_bind(
            "y",
            binop("+", var("x", i32_type()), int_lit(2)),
            binop("+", var("acc", i32_type()), var("y", i32_type())),
        ),
    );
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    assert_eq!(structure, vec!["let", "let", "loop", "expr"]);
}

#[test]
fn test_partial_hoist_transitive_dependency() {
    // loop acc = 0 for i < 10 do
    //   let x = 42 in          -- hoistable (no deps)
    //   let y = acc + x in     -- NOT hoistable (depends on acc)
    //   let z = y * 2 in       -- NOT hoistable (depends on y which depends on acc)
    //   z

    let body = let_bind(
        "x",
        int_lit(42),
        let_bind(
            "y",
            binop("+", var("acc", i32_type()), var("x", i32_type())),
            let_bind(
                "z",
                binop("*", var("y", i32_type()), int_lit(2)),
                var("z", i32_type()),
            ),
        ),
    );
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    // x hoisted, y and z stay (transitive dependency through y)
    assert_eq!(structure, vec!["let", "loop", "let", "let", "expr"]);
}

#[test]
fn test_iteration_var_dependency() {
    // loop acc = 0 for i < 10 do
    //   let x = i * 2 in       -- NOT hoistable (depends on i)
    //   acc + x

    let body = let_bind(
        "x",
        binop("*", var("i", i32_type()), int_lit(2)),
        binop("+", var("acc", i32_type()), var("x", i32_type())),
    );
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    assert_eq!(structure, vec!["loop", "let", "expr"]); // x stays in loop
}

#[test]
fn test_no_bindings_in_loop() {
    // loop acc = 0 for i < 10 do acc + 1
    // => unchanged (nothing to hoist)

    let body = binop("+", var("acc", i32_type()), int_lit(1));
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    assert_eq!(structure, vec!["loop", "expr"]);
}

#[test]
fn test_multiple_independent_hoistable() {
    // loop acc = 0 for i < 10 do
    //   let x = 1 in
    //   let y = 2 in
    //   acc + x + y
    // =>
    // let x = 1 in let y = 2 in loop ... acc + x + y

    let body = let_bind(
        "x",
        int_lit(1),
        let_bind(
            "y",
            int_lit(2),
            binop(
                "+",
                binop("+", var("acc", i32_type()), var("x", i32_type())),
                var("y", i32_type()),
            ),
        ),
    );
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    assert_eq!(structure, vec!["let", "let", "loop", "expr"]);
}

#[test]
fn test_mixed_hoistable_and_dependent() {
    // loop acc = 0 for i < 10 do
    //   let a = 100 in         -- hoistable
    //   let b = acc in         -- NOT hoistable
    //   let c = 200 in         -- hoistable (doesn't depend on b)
    //   b + a + c

    let body = let_bind(
        "a",
        int_lit(100),
        let_bind(
            "b",
            var("acc", i32_type()),
            let_bind(
                "c",
                int_lit(200),
                binop(
                    "+",
                    binop("+", var("b", i32_type()), var("a", i32_type())),
                    var("c", i32_type()),
                ),
            ),
        ),
    );
    let input = for_range_loop("acc", int_lit(0), "i", int_lit(10), body);

    let mut lifter = BindingLifter::new();
    let result = lifter.lift_expr(input).unwrap();

    let structure = get_structure(&result);
    // a and c hoisted, b stays
    assert_eq!(structure, vec!["let", "let", "loop", "let", "expr"]);
}

// =============================================================================
// Integration Tests - Source Code to MIR
// =============================================================================

/// Compile source through flattening (before lifting)
fn compile_to_flattened(source: &str) -> crate::mir::Program {
    crate::Compiler::parse(source)
        .expect("parse failed")
        .elaborate()
        .expect("elaborate failed")
        .resolve()
        .expect("resolve failed")
        .type_check()
        .expect("type_check failed")
        .alias_check()
        .expect("alias_check failed")
        .flatten()
        .expect("flatten failed")
        .monomorphize()
        .expect("monomorphize failed")
        .filter_reachable()
        .fold_constants()
        .expect("fold_constants failed")
        .mir
}
