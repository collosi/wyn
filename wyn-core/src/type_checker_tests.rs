use crate::ast::{Type, TypeName};
use crate::error::CompilerError;
use crate::lexer::tokenize;
use crate::parser::Parser;
use crate::type_checker::{TypeChecker, TypeWarning};

/// Helper to parse and type check source code, expecting success
fn typecheck_program(input: &str) {
    let result = try_typecheck_program(input);
    if let Err(e) = &result {
        eprintln!("\n=== TYPE CHECK ERROR ===");
        eprintln!("{:?}", e);
    }
    result.expect("Type checking should succeed");
}

/// Helper to parse and type check source code, returning the result
fn try_typecheck_program(input: &str) -> Result<(), CompilerError> {
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Parsing failed");

    let mut type_checker = TypeChecker::new();
    type_checker.load_builtins().expect("Loading builtins failed");
    type_checker.check_program(&program)?;
    Ok(())
}

#[test]
fn test_type_check_let() {
    typecheck_program("let x: i32 = 42");
}

#[test]
fn test_type_mismatch() {
    assert!(try_typecheck_program("let x: i32 = 3.14f32").is_err());
}

#[test]
fn test_array_type_check() {
    typecheck_program("let arr: [2]f32 = [1.0f32, 2.0f32]");
}

#[test]
fn test_undefined_variable() {
    let result = try_typecheck_program("let x: i32 = undefined");
    assert!(matches!(
        result.unwrap_err(),
        CompilerError::UndefinedVariable(_, _)
    ));
}

#[test]
fn test_simple_def() {
    typecheck_program("def identity x = x");
}

#[test]
fn test_two_length_and_replicate_calls() {
    // Simplified test: two calls to length/replicate with different array element types
    // This tests that type variables don't bleed between the two calls
    typecheck_program(r#"
def test : f32 =
    let v4s : [2]vec4f32 = [vec4 1.0f32 2.0f32 3.0f32 4.0f32, vec4 5.0f32 6.0f32 7.0f32 8.0f32] in
    let len1 = length v4s in
    let out1 = replicate len1 (__uninit()) in

    let indices : [2]i32 = [0, 1] in
    let len2 = length indices in
    let out2 = replicate len2 (__uninit()) in

    42.0f32
        "#);
}

#[test]
fn test_zip_arrays() {
    typecheck_program("def zip_arrays xs ys = zip xs ys");
}

/// Helper function to check a program with a type hole and return the inferred type
fn check_type_hole(source: &str) -> Type {
    use crate::lexer;
    use crate::parser::Parser;

    // Parse
    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // Type check
    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();
    let _type_table = checker.check_program(&program).unwrap();

    // Check warnings
    let warnings = checker.warnings();
    assert_eq!(warnings.len(), 1, "Expected exactly one type hole warning");

    match &warnings[0] {
        TypeWarning::TypeHoleFilled { inferred_type, .. } => {
            // Apply the context to normalize type variables
            inferred_type.apply(checker.context())
        }
    }
}

#[test]
fn test_type_hole_in_array() {
    let inferred = check_type_hole("def arr = [1i32, ???, 3i32]");

    // ??? should be inferred as i32 (to match array elements)
    let expected = Type::Constructed(TypeName::Str("i32"), vec![]);
    assert_eq!(inferred, expected);
}

#[test]
fn test_type_hole_in_binop() {
    let inferred = check_type_hole("def result = 5i32 + ???");

    // ??? should be inferred as i32 (to match addition operand)
    let expected = Type::Constructed(TypeName::Str("i32"), vec![]);
    assert_eq!(inferred, expected);
}

#[test]
fn test_type_hole_function_arg() {
    let inferred = check_type_hole("def apply = (\\x:i32 -> x + 1i32) ???");

    // ??? should be inferred as i32 (the function argument type)
    let expected = Type::Constructed(TypeName::Str("i32"), vec![]);
    assert_eq!(inferred, expected);
}

#[test]
fn test_lambda_param_with_annotation() {
    // Test that lambda parameter works with type annotation (Futhark-style)
    // Field projection requires the parameter type to be known
    typecheck_program("def test : [2]f32 = let arr : [2]vec3f32 = [vec3 1.0f32 2.0f32 3.0f32, vec3 4.0f32 5.0f32 6.0f32] in map (\\(v:vec3f32) -> v.x) arr");
}

#[test]
fn test_bidirectional_with_concrete_type() {
    // Test bidirectional checking with a CONCRETE expected type
    // This demonstrates where bidirectional checking actually helps
    typecheck_program(r#"
            def apply_to_vec (f : vec3f32 -> f32) : f32 =
              f (vec3 1.0f32 2.0f32 3.0f32)

            def test : f32 = apply_to_vec (\v -> v.x)
        "#);
}

#[test]
fn test_bidirectional_explicit_annotation_mismatch() {
    // Minimal test demonstrating bidirectional checking bug with explicit parameter annotations.
    // Two chained maps: vec3f32->vec4f32, then vec4f32->vec3f32
    // The second lambda's parameter annotation (q:vec4f32) is correct (v4s is [1]vec4f32),
    // but bidirectional checking incorrectly rejects it.
    typecheck_program(r#"
            def test =
              let arr : [1]vec3f32 = [vec3 1.0f32 2.0f32 3.0f32] in
              let v4s : [1]vec4f32 = map (\(v:vec3f32) -> vec4 v.x v.y v.z 1.0f32) arr in
              map (\(q:vec4f32) -> vec3 q.x q.y q.z) v4s
        "#);
}

#[test]
fn test_map_with_unannotated_lambda_and_array_index() {
    // Test that bidirectional checking infers lambda parameter type from array type
    typecheck_program(r#"
            def test : [12]i32 =
              let edges : [12][2]i32 = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]] in
              map (\e -> e[0]) edges
        "#);
}

#[test]
fn test_lambda_with_tuple_pattern() {
    // Test that lambdas with tuple patterns work
    typecheck_program(r#"
            def test : (i32, i32) -> i32 =
              \(x, y) -> x + y
        "#);
}

#[test]
fn test_lambda_with_wildcard_in_tuple() {
    // Test that lambdas with wildcard in tuple patterns work
    typecheck_program(r#"
            def test : (i32, i32) -> i32 =
              \(_, acc) -> acc
        "#);
}

// Tests for loop type checking will be added once Loop support is implemented
// in the type checker (currently todo!())

#[test]
fn test_map_with_array_size_inference() {
    typecheck_program(r#"
def test : [8]i32 =
  let arr = [1, 2, 3, 4, 5, 6, 7, 8] in
  map (\x -> x + 1) arr
"#);
}

#[test]
fn test_let_polymorphism() {
    // Test that let-bound values are properly generalized
    // Without generalization, this would fail because id would be monomorphic
    typecheck_program(r#"
            def test : bool =
                let id = \x -> x in
                let test1 : i32 = id ??? in
                let test2 : bool = id ??? in
                test2
        "#);
}

#[test]
fn test_top_level_polymorphism() {
    // Test that top-level let/def declarations are generalized
    typecheck_program(r#"
            def id = \x -> x
            def test1 : i32 = id ???
            def test2 : bool = id ???
        "#);
}

#[test]
fn test_polymorphic_id_tuple() {
    // Classic HM polymorphism test: let id = \x -> x in (id 5, id true)
    typecheck_program(r#"
            def test =
                let id = \x -> x in
                (id ???, id ???)
        "#);
}

#[test]
fn test_qualified_name_sqrt() {
    // Test that qualified names like f32.sqrt type check correctly
    typecheck_program(r#"
            def test : f32 = f32.sqrt 4.0f32
        "#);
}

#[test]
fn test_nested_array_indexing() {
    // Test that nested array indexing type inference works
    // Reproduces the de_rasterizer.wyn issue: e[0] where e : [2]i32
    typecheck_program(r#"
            def test =
                let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                let e : [2]i32 = edges[0] in
                let idx : i32 = e[0] in
                verts[idx]
        "#);
}

#[test]
fn test_nested_array_indexing_in_lambda() {
    // Test that nested array indexing works inside a lambda in map
    // This reproduces the actual de_rasterizer.wyn pattern:
    // map (\e -> verts[e[0]]) edges
    typecheck_program(r#"
            def test =
                let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                map (\e -> verts[e[0]]) edges
        "#);
}

#[test]
fn test_nested_array_indexing_with_literal() {
    // Test with array literal directly in map call, without type annotation
    // This is closer to the de_rasterizer pattern
    typecheck_program(r#"
            def test =
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                map (\e -> verts[e[0]]) [[0,1], [1,2], [2,0]]
        "#);
}

#[test]
fn test_size_parameter_binding() {
    // Test that size parameters are properly bound and substituted
    typecheck_program(r#"
def identity [n] (xs: [n]i32): [n]i32 = xs

def test : [5]i32 =
  let arr = [1, 2, 3, 4, 5] in
  identity arr
"#);
}

#[test]
fn test_f32_sum_with_map_over_nested_array() {
    // Test f32.sum with map over nested arrays
    typecheck_program(r#"
def test : f32 =
    let edges : [3][2]i32 = [[0, 1], [1, 2], [2, 0]] in
    f32.sum (map (\e -> 1.0f32) edges)
    "#);
}

#[test]
fn test_map_lambda_param_type_unification() {
    // Test that lambda parameter type variable gets unified with input array element type
    typecheck_program(r#"
def test : [3]f32 =
    let edges : [3][2]i32 = [[0, 1], [1, 2], [2, 0]] in
    map (\e -> let a = e[0] in 1.0f32) edges
    "#);
}

#[test]
fn test_map_with_capturing_closure() {
    // Test with dot product and f32.min
    typecheck_program(r#"
def line : f32 =
  let denom = dot (vec2 0.0f32 0.0f32) (vec2 9.0f32 9.0f32) in
  f32.min denom 1.0f32
"#);
}

#[test]
fn test_f32_sum_with_map_indexing_nested_array() {
    // Test f32.sum with map accessing elements of nested array
    typecheck_program(r#"
def test : f32 =
    let edges : [3][2]i32 = [[0, 1], [1, 2], [2, 0]] in
    f32.sum (map (\e -> let a = e[0] in let b = e[1] in 1.0f32) edges)
    "#);
}
