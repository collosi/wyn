//! Tests for the ANF normalization pass.

use crate::ast::{NodeCounter, NodeId, Span, TypeName};
use crate::error::CompilerError;
use crate::mir::{DefId, Expr, ExprKind, Literal, LocalId};
use crate::normalize::{Normalizer, is_atomic};
use polytype::Type;
use std::sync::atomic::{AtomicU32, Ordering};

/// Helper to run full pipeline through lowering with normalization
fn compile_through_lowering(input: &str) -> Result<(), CompilerError> {
    let parsed = crate::Compiler::parse(input)?;
    let module_manager = crate::cached_module_manager(parsed.node_counter.clone());
    parsed
        .elaborate(module_manager)?
        .resolve()?
        .type_check()?
        .alias_check()?
        .fold_ast_constants()
        .flatten()?
        .hoist_materializations()
        .normalize()
        .monomorphize()?
        .filter_reachable()
        .fold_constants()?
        .lift_bindings()?
        .lower()?;
    Ok(())
}

fn test_span() -> Span {
    Span::new(1, 1, 1, 1)
}

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

// Global counter for test node IDs (tests don't care about the actual values)
static TEST_NODE_ID: AtomicU32 = AtomicU32::new(0);
static TEST_LOCAL_ID: AtomicU32 = AtomicU32::new(1000); // Start high to avoid conflicts

fn next_id() -> NodeId {
    NodeId(TEST_NODE_ID.fetch_add(1, Ordering::Relaxed))
}

fn next_local_id() -> LocalId {
    LocalId(TEST_LOCAL_ID.fetch_add(1, Ordering::Relaxed))
}

fn var(local_id: LocalId) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::Var(local_id),
        test_span(),
    )
}

fn int_lit(n: i32) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::Literal(Literal::Int(n.to_string())),
        test_span(),
    )
}

fn binop(op: &str, lhs: Expr, rhs: Expr) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::BinOp {
            op: op.to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        test_span(),
    )
}

fn call(func: DefId, args: Vec<Expr>) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::Call {
            func,
            func_name: None,
            args,
        },
        test_span(),
    )
}

#[test]
fn test_atomic_var() {
    let local_id = next_local_id();
    let expr = var(local_id);
    assert!(is_atomic(&expr));
}

#[test]
fn test_atomic_int_literal() {
    let expr = int_lit(42);
    assert!(is_atomic(&expr));
}

#[test]
fn test_not_atomic_binop() {
    let a = next_local_id();
    let b = next_local_id();
    let expr = binop("+", var(a), var(b));
    assert!(!is_atomic(&expr));
}

#[test]
fn test_normalize_binop_with_atomic_operands() {
    // a + b should stay as a + b (no new bindings)
    let a = next_local_id();
    let b = next_local_id();
    let expr = binop("+", var(a), var(b));
    let mut bindings = Vec::new();
    let mut normalizer = Normalizer::new(NodeCounter::new());
    let result = normalizer.normalize_expr(expr, &mut bindings);

    assert!(bindings.is_empty());
    matches!(result.kind, ExprKind::BinOp { .. });
}

#[test]
fn test_normalize_nested_binop() {
    // (a + b) + c should become:
    // let temp = a + b in temp + c
    let a = next_local_id();
    let b = next_local_id();
    let c = next_local_id();
    let inner = binop("+", var(a), var(b));
    let expr = binop("+", inner, var(c));
    let mut bindings = Vec::new();
    let mut normalizer = Normalizer::new(NodeCounter::new());
    let result = normalizer.normalize_expr(expr, &mut bindings);

    // Should have one binding for the inner binop
    assert_eq!(bindings.len(), 1);
    let (temp_local, _) = &bindings[0];

    // Result should be temp + c
    if let ExprKind::BinOp { lhs, rhs, .. } = &result.kind {
        // LHS should be the temp variable
        assert!(matches!(lhs.kind, ExprKind::Var(ref v) if *v == *temp_local));
        // RHS should be c
        assert!(matches!(rhs.kind, ExprKind::Var(ref v) if *v == c));
    } else {
        panic!("Expected BinOp");
    }
}

#[test]
fn test_normalize_call_with_binop_arg() {
    // foo(a + b) should become:
    // let temp = a + b in foo(temp)
    let a = next_local_id();
    let b = next_local_id();
    let arg = binop("+", var(a), var(b));
    let func_id = DefId(999); // Use sentinel for test
    let expr = call(func_id, vec![arg]);
    let mut bindings = Vec::new();
    let mut normalizer = Normalizer::new(NodeCounter::new());
    let result = normalizer.normalize_expr(expr, &mut bindings);

    // Should have one binding for the binop
    assert_eq!(bindings.len(), 1);
    let (temp_local, _) = &bindings[0];

    // Result should be foo(temp)
    if let ExprKind::Call { args, .. } = &result.kind {
        assert_eq!(args.len(), 1);
        assert!(matches!(args[0].kind, ExprKind::Var(ref v) if *v == *temp_local));
    } else {
        panic!("Expected Call");
    }
}

#[test]
fn test_normalize_loop_with_tuple_state() {
    // This tests that loops with tuple state work correctly after normalization.
    // The loop produces a tuple (acc, i) on each iteration.
    let source = r#"
def sum [n] (arr:[n]f32) : f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length arr do
    (acc + arr[i], i + 1)
  in result

#[vertex]
def vertex_main (vertex_id:i32) : #[builtin(position)] vec4f32 =
  let result = sum [1.0f32, 1.0f32, 1.0f32] in
  @[result, result, 0.0f32, 1.0f32]
"#;
    compile_through_lowering(source).expect("Should compile with normalized loop");
}
