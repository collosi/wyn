#![cfg(test)]

use crate::lexer::tokenize;
use crate::mir;
use crate::parser::Parser;
use crate::type_checker::TypeChecker;

fn flatten_program(input: &str) -> mir::Program {
    // Use the typestate API to ensure proper compilation pipeline
    crate::Compiler::parse(input)
        .expect("Parsing failed")
        .elaborate()
        .expect("Elaboration failed")
        .resolve()
        .expect("Name resolution failed")
        .type_check()
        .expect("Type checking failed")
        .flatten()
        .expect("Flattening failed")
        .mir
}

fn flatten_to_string(input: &str) -> String {
    format!("{}", flatten_program(input))
}

#[test]
fn test_simple_constant() {
    let mir = flatten_to_string("def x = 42");
    assert!(mir.contains("def x:"));
    assert!(mir.contains("42"));
}

#[test]
fn test_simple_function() {
    let mir = flatten_to_string("def add x y = x + y");
    // MIR format includes types: def add (x: type) (y: type): return_type =
    assert!(mir.contains("def add"));
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
    assert!(mir_str.contains("__lambda_name"));
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
    let mir = flatten_to_string("def g (y:i32) : i32 = y + 1\ndef f x = g x");
    assert!(mir.contains("def g"));
    assert!(mir.contains("def f"));
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
fn test_map_with_lambda() {
    // Test map with inline lambda
    let source = r#"
def test : [4]i32 =
    map (\(x:i32) -> x + 1) [0, 1, 2, 3]
"#;

    // Parse
    let parsed = crate::Compiler::parse(source).expect("Parsing failed");

    // Elaborate
    let elaborated = parsed.elaborate().expect("Elaboration failed");

    // Resolve
    let resolved = elaborated.resolve().expect("Name resolution failed");

    // Print AST to see what NodeId(6) is
    println!("AST:");
    println!("{:#?}", resolved.ast);

    // Type check
    let typed = resolved.type_check().expect("Type checking failed");
    println!("\nType table has {} entries", typed.type_table.len());
    for (id, scheme) in &typed.type_table {
        println!("  NodeId({:?}): {:?}", id, scheme);
    }

    println!("\nNodeId(6) is missing from type table!");

    // Flatten - this is where it fails
    let flattened = typed.flatten().expect("Flattening failed");
    let mir_str = format!("{}", flattened.mir);
    println!("MIR: {}", mir_str);
    assert!(mir_str.contains("def test"));
}

#[test]
fn test_lambda_captures_typed_variable() {
    // This test reproduces an issue where a lambda captures a typed variable (like an array),
    // and the free variable rewriting creates __closure.mat, which then fails when trying
    // to resolve 'mat' as a field access on the closure.
    let mir = flatten_program(
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
    let mir = flatten_program(
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
    let mir = flatten_program(
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
    // Should have closure record with __lambda_name
    assert!(mir_str.contains("__lambda_name"));
    // Lambda registry should have the lambda function
    assert_eq!(mir.lambda_registry.len(), 1);
    assert_eq!(mir.lambda_registry[0].0, "__lam_test_map_0");
    assert_eq!(mir.lambda_registry[0].1, 1); // arity 1
}

#[test]
fn test_direct_closure_call() {
    // This test checks that directly calling a closure generates a direct lambda call
    let mir = flatten_program(
        r#"
def test_apply (x:i32) : i32 =
    let f = \(y:i32) -> y + x in
    f 10
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    println!("Lambda registry: {:?}", mir.lambda_registry);

    // Should generate a direct call to __lam_test_apply_0 with the closure as first argument
    assert!(mir_str.contains("__lam_test_apply_0 f 10"));
    // Should NOT generate apply1 intrinsic
    assert!(!mir_str.contains("apply1"));
}

// Tests for function value restrictions (Futhark-style defunctionalization constraints)

#[test]
fn test_error_array_of_functions() {
    // Arrays of functions are not permitted
    let input = r#"
def test : [2](i32 -> i32) =
    [\(x:i32) -> x + 1, \(x:i32) -> x * 2]
"#;
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("Parsing failed");

    let mut type_checker = TypeChecker::new();
    type_checker.load_builtins().expect("Failed to load builtins");
    let result = type_checker.check_program(&ast);

    assert!(result.is_err(), "Should reject arrays of functions");
}

#[test]
fn test_error_function_from_if() {
    // A function cannot be returned from an if expression
    let input = r#"
def choose (b:bool) : (i32 -> i32) =
    if b then \(x:i32) -> x + 1 else \(x:i32) -> x * 2
"#;
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("Parsing failed");

    let mut type_checker = TypeChecker::new();
    type_checker.load_builtins().expect("Failed to load builtins");
    let result = type_checker.check_program(&ast);

    assert!(
        result.is_err(),
        "Should reject function returned from if expression"
    );
}

#[test]
fn test_error_loop_parameter_function() {
    // A loop parameter cannot be a function
    let input = r#"
def test : (i32 -> i32) =
    loop f = \(x:i32) -> x while false do f
"#;
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("Parsing failed");

    let mut type_checker = TypeChecker::new();
    type_checker.load_builtins().expect("Failed to load builtins");
    let result = type_checker.check_program(&ast);

    assert!(result.is_err(), "Should reject function as loop parameter");
}

#[test]
fn test_lambda_calling_builtin_constructor() {
    // This test reproduces an issue where vec4 inside a lambda is incorrectly
    // captured as a free variable and rewritten to __closure.vec4
    let mir = flatten_program(
        r#"
def test (v:vec3f32) : vec4f32 =
    let f = \(x:vec3f32) -> vec4 x.x x.y x.z 1.0f32 in
    f v
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);

    // Should call vec4 directly, not __closure.vec4
    assert!(mir_str.contains("vec4"));
    // vec4 should NOT be accessed through closure
    assert!(!mir_str.contains("__closure.vec4"));
}

#[test]
fn test_f32_sum_inline_definition() {
    // Test sum with inline definition (simpler than using prelude)
    let mir = flatten_program(
        r#"
def mysum [n] (arr:[n]f32) : f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length arr do
    (acc + arr[i], i + 1)
  in result

def test : f32 =
  let arr = [1.0f32, 2.0f32, 3.0f32] in
  mysum arr
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("mysum"));
}

#[test]
fn test_f32_sum_simple() {
    // Test that f32.sum from prelude works through full compilation
    let source = r#"
def test : f32 =
  let arr = [1.0f32, 2.0f32, 3.0f32] in
  f32.sum arr
"#;

    // This should compile successfully
    let result = crate::Compiler::parse(source)
        .and_then(|p| p.elaborate())
        .and_then(|e| e.resolve())
        .and_then(|r| r.type_check())
        .and_then(|t| t.flatten())
        .and_then(|f| f.monomorphize())
        .map(|m| m.filter_reachable())
        .and_then(|r| r.fold_constants())
        .and_then(|f| f.lower());
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());
}

#[test]
fn test_f32_conversions() {
    // Test f32 type conversion builtins
    let mir = flatten_program(
        r#"
def test_conversions (x: i32): f32 =
  let f1 = __builtin_f32_from_i32 x in
  let i1 = __builtin_f32_to_i32 f1 in
  let f2 = __builtin_f32_from_i32 i1 in
  f2
"#,
    );
    let mir_str = format!("{}", mir);
    // Should contain the builtin conversions
    assert!(mir_str.contains("__builtin_f32_from_i32"));
    assert!(mir_str.contains("__builtin_f32_to_i32"));
}

#[test]
fn test_f32_math_operations() {
    // Test f32 math operations including GLSL extended ops
    let mir = flatten_program(
        r#"
def test_math (x: f32): f32 =
  let a = f32.sin x in
  let b = f32.cos x in
  let c = f32.sqrt a in
  let d = f32.exp b in
  let e = f32.log c in
  let f = f32.pow d 2.0f32 in
  let g = f32.sinh x in
  let h = f32.asinh g in
  let i = f32.atan2 x a in
  f32.fma f e i
"#,
    );
    let mir_str = format!("{}", mir);
    // Should contain f32 math operations
    assert!(mir_str.contains("f32.sin"));
    assert!(mir_str.contains("f32.cos"));
    assert!(mir_str.contains("f32.sqrt"));
    assert!(mir_str.contains("f32.sinh"));
    assert!(mir_str.contains("f32.asinh"));
    assert!(mir_str.contains("f32.atan2"));
    assert!(mir_str.contains("f32.fma"));
}

#[test]
fn test_operator_section_direct_application() {
    // Test operator section applied directly to arguments
    let mir = flatten_program(
        r#"
def test_add (x: i32) (y: i32): i32 = (+) x y
"#,
    );
    let mir_str = format!("{}", mir);
    // Operator section (+) applied to arguments should flatten correctly
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_add"));
}

#[test]
fn test_operator_section_with_map() {
    // Test operator section passed to map (special higher-order function)
    let mir = flatten_program(
        r#"
def test_map (arr: [3]i32): [3]i32 = map (\(x:i32) -> (+) x 1) arr
"#,
    );
    let mir_str = format!("{}", mir);
    // Should successfully flatten with lambda wrapping operator section
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map"));
}
