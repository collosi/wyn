#![cfg(test)]

use crate::flattening::Flattener;
use crate::lexer::tokenize;
use crate::mir;
use crate::parser::Parser;
use crate::type_checker::TypeChecker;
use std::collections::HashMap;

fn flatten_program(input: &str) -> mir::Program {
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("Parsing failed");
    let type_table = HashMap::new(); // Empty - tests don't use field access
    let mut flattener = Flattener::new(type_table);
    flattener.flatten_program(&ast).expect("Flattening failed")
}

/// Flatten with type checking (required for tests that use field access or captures)
fn flatten_with_types(input: &str) -> mir::Program {
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("Parsing failed");

    let mut type_checker = TypeChecker::new();
    type_checker.load_builtins().expect("Failed to load builtins");
    let type_table = type_checker.check_program(&ast).expect("Type checking failed");

    let mut flattener = Flattener::new(type_table);
    flattener.flatten_program(&ast).expect("Flattening failed")
}

fn flatten_to_string(input: &str) -> String {
    format!("{}", flatten_program(input))
}

#[test]
fn test_simple_constant() {
    let mir = flatten_to_string("def x = 42");
    assert!(mir.contains("def x ="));
    assert!(mir.contains("42"));
}

#[test]
fn test_simple_function() {
    let mir = flatten_to_string("def add x y = x + y");
    assert!(mir.contains("def add x y ="));
    assert!(mir.contains("(x + y)"));
}

#[test]
fn test_let_binding() {
    let mir = flatten_to_string("def f = let x = 1 in x + 2");
    assert!(mir.contains("let x = 1 in"));
}

#[test]
fn test_tuple_pattern() {
    let mir = flatten_to_string("def f = let (a, b) = (1, 2) in a + b");
    // Should generate tuple extraction
    assert!(mir.contains("tuple_access"));
}

#[test]
fn test_lambda_defunctionalization() {
    let mir = flatten_program("def f = \\x -> x + 1");
    // Should generate a lambda function
    assert!(mir.defs.len() >= 2); // Original + lambda

    // Check that closure record is created
    let mir_str = format!("{}", mir);
    assert!(mir_str.contains("__lam_f_"));
    assert!(mir_str.contains("__tag"));
}

#[test]
fn test_lambda_with_capture() {
    let mir = flatten_program("def f y = let g = \\x -> x + y in g 1");
    let mir_str = format!("{}", mir);

    // Lambda should capture y
    assert!(mir_str.contains("__closure"));
    // Should reference y from closure
    assert!(mir_str.contains("record_access") || mir_str.contains("__closure"));
}

#[test]
fn test_nested_let() {
    let mir = flatten_to_string("def f = let x = 1 in let y = 2 in x + y");
    assert!(mir.contains("let x = 1"));
    assert!(mir.contains("let y = 2"));
}

#[test]
fn test_if_expression() {
    let mir = flatten_to_string("def f x = if x then 1 else 0");
    assert!(mir.contains("if x then 1 else 0"));
}

#[test]
fn test_function_call() {
    let mir = flatten_to_string("def f x = g(x, 1)");
    // g(x, 1) in source becomes g (x, 1) - call with tuple argument
    assert!(mir.contains("g (x, 1)"));
}

#[test]
fn test_array_literal() {
    let mir = flatten_to_string("def arr = [1, 2, 3]");
    assert!(mir.contains("[1, 2, 3]"));
}

#[test]
fn test_record_literal() {
    let mir = flatten_to_string("def r = {x: 1, y: 2}");
    assert!(mir.contains("x=1"));
    assert!(mir.contains("y=2"));
}

#[test]
fn test_while_loop() {
    let mir = flatten_to_string("def f = loop x = 0 while x < 10 do x + 1");
    assert!(mir.contains("loop"));
    assert!(mir.contains("while"));
}

#[test]
fn test_for_range_loop() {
    let mir = flatten_to_string("def f = loop acc = 0 for i < 10 do acc + i");
    assert!(mir.contains("loop"));
    assert!(mir.contains("for i <"));
}

#[test]
fn test_binary_ops() {
    let mir = flatten_to_string("def f x y = x * y + x / y");
    assert!(mir.contains("*"));
    assert!(mir.contains("+"));
    assert!(mir.contains("/"));
}

#[test]
fn test_unary_op() {
    let mir = flatten_to_string("def f x = -x");
    assert!(mir.contains("(-x)"));
}

#[test]
fn test_array_index() {
    let mir = flatten_to_string("def f arr i = arr[i]");
    assert!(mir.contains("index"));
}

#[test]
fn test_multiple_lambdas() {
    let mir = flatten_program(
        r#"
def f =
    let a = \x -> x + 1 in
    let b = \y -> y * 2 in
    (a, b)
"#,
    );
    // Should have original + 2 lambdas
    assert!(mir.defs.len() >= 3);
}

#[test]
fn test_lambda_captures_typed_variable() {
    // This test reproduces an issue where a lambda captures a typed variable (like an array),
    // and the free variable rewriting creates __closure.mat, which then fails when trying
    // to resolve 'mat' as a field access on the closure.
    let mir = flatten_with_types(
        r#"
def test_capture (arr:[4]i32) : i32 =
    let result = map (\(i:i32) -> arr[i]) [0, 1, 2, 3] in
    result[0]
"#,
    );
    let mir_str = format!("{}", mir);
    // Lambda should capture arr and access it via closure
    assert!(mir_str.contains("__closure") || mir_str.contains("record_access"));
}

#[test]
fn test_qualified_name_f32_sqrt() {
    // This test reproduces an issue where f32.sqrt is treated as field access
    // instead of a qualified builtin name. The identifier 'f32' has no type in
    // the type_table because it's a type name, not a variable.
    let mir = flatten_with_types(
        r#"
def length2 (v:vec2f32) : f32 =
    f32.sqrt (v.x * v.x + v.y * v.y)
"#,
    );
    let mir_str = format!("{}", mir);
    // Should contain f32.sqrt as a qualified name/call, not a field access error
    assert!(mir_str.contains("f32.sqrt"));
}

#[test]
fn test_map_with_closure_application() {
    // This test checks that map with a lambda generates a closure record
    // and registers the lambda in the registry for dispatch.
    let mir = flatten_with_types(
        r#"
def test_map (arr:[4]i32) : [4]i32 =
    map (\(x:i32) -> x + 1) arr
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    println!("Lambda registry: {:?}", mir.lambda_registry);

    // Should have generated lambda function
    assert!(mir_str.contains("__lam_test_map_"));
    // Should have closure record with tag
    assert!(mir_str.contains("__tag=0"));
    // Lambda registry should map tag 0 to the lambda function
    assert_eq!(mir.lambda_registry.len(), 1);
    assert_eq!(mir.lambda_registry[0].0, "__lam_test_map_0");
    assert_eq!(mir.lambda_registry[0].1, 1); // arity 1
}

#[test]
fn test_direct_closure_call() {
    // This test checks that directly calling a closure generates apply1 intrinsic
    let mir = flatten_with_types(
        r#"
def test_apply (x:i32) : i32 =
    let f = \(y:i32) -> y + x in
    f 10
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    println!("Lambda registry: {:?}", mir.lambda_registry);

    // Should have apply1 intrinsic for direct closure call
    assert!(mir_str.contains("apply1"));
}
