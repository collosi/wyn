use crate::ast::{Type, TypeName};
use crate::error::CompilerError;
use crate::lexer::tokenize;
use crate::parser::Parser;
use crate::type_checker::{TypeChecker, TypeWarning};

#[test]
fn test_type_check_let() {
    let input = "let x: i32 = 42";
    let tokens = tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());
}

#[test]
fn test_type_mismatch() {
    let input = "let x: i32 = 3.14f32";
    let tokens = tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_err());
}

#[test]
fn test_array_type_check() {
    let input = "let arr: [2]f32 = [1.0f32, 2.0f32]";
    let tokens = tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());
}

#[test]
fn test_undefined_variable() {
    let input = "let x: i32 = undefined";
    let tokens = tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    let result = checker.check_program(&program);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        CompilerError::UndefinedVariable(_, _)
    ));
}

#[test]
fn test_simple_def() {
    let input = "def identity x = x";
    let tokens = tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());
}

#[test]
fn test_two_length_and_replicate_calls() {
    // Simplified test: two calls to length/replicate with different array element types
    // This tests that type variables don't bleed between the two calls
    let input = r#"
def test : f32 =
    let v4s : [2]vec4f32 = [vec4 1.0f32 2.0f32 3.0f32 4.0f32, vec4 5.0f32 6.0f32 7.0f32 8.0f32] in
    let len1 = length v4s in
    let out1 = replicate len1 (__uninit()) in

    let indices : [2]i32 = [0, 1] in
    let len2 = length indices in
    let out2 = replicate len2 (__uninit()) in

    42.0f32
        "#;

    std::env::set_var("RUST_LOG", "debug");
    env_logger::try_init().ok();

    let tokens = tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();
    let result = checker.check_program(&program);

    if let Err(e) = &result {
        eprintln!("Type check failed: {:?}", e);
    }

    assert!(
        result.is_ok(),
        "Two length/replicate calls should type-check successfully"
    );
}

#[test]
fn test_zip_arrays() {
    let input = "def zip_arrays xs ys = zip xs ys";
    let tokens = tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();
    match checker.check_program(&program) {
        Ok(_) => {
            println!("Type checking succeeded!");

            // Check that zip_arrays has the expected type
            if let Some(func_type) = checker.lookup("zip_arrays") {
                println!("zip_arrays type: {}", func_type);

                // The inferred type should be something like: t0 -> t1 -> [1](i32, i32)
                // This demonstrates that type inference is working
            }
        }
        Err(e) => {
            println!("Type checking failed: {:?}", e);
            panic!("Type checking failed");
        }
    }
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
    let source = "def test : [2]f32 = let arr : [2]vec3f32 = [vec3 1.0f32 2.0f32 3.0f32, vec3 4.0f32 5.0f32 6.0f32] in map (\\(v:vec3f32) -> v.x) arr";

    use crate::lexer;
    use crate::parser::Parser;

    // Parse
    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // Type check
    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed with type annotation
        }
        Err(e) => {
            panic!("Type checking failed: {:?}", e);
        }
    }
}

#[test]
fn test_bidirectional_with_concrete_type() {
    // Test bidirectional checking with a CONCRETE expected type
    // This demonstrates where bidirectional checking actually helps
    let source = r#"
            def apply_to_vec (f : vec3f32 -> f32) : f32 =
              f (vec3 1.0f32 2.0f32 3.0f32)

            def test : f32 = apply_to_vec (\v -> v.x)
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    // Parse
    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // Type check
    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed! apply_to_vec expects vec3f32 -> f32 (concrete)
            // so bidirectional checking gives parameter v the type vec3f32
        }
        Err(e) => {
            panic!("Type checking failed: {:?}", e);
        }
    }
}

#[test]
fn test_bidirectional_explicit_annotation_mismatch() {
    // Minimal test demonstrating bidirectional checking bug with explicit parameter annotations.
    // Two chained maps: vec3f32->vec4f32, then vec4f32->vec3f32
    // The second lambda's parameter annotation (q:vec4f32) is correct (v4s is [1]vec4f32),
    // but bidirectional checking incorrectly rejects it.
    let source = r#"
            def test =
              let arr : [1]vec3f32 = [vec3 1.0f32 2.0f32 3.0f32] in
              let v4s : [1]vec4f32 = map (\(v:vec3f32) -> vec4 v.x v.y v.z 1.0f32) arr in
              map (\(q:vec4f32) -> vec3 q.x q.y q.z) v4s
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed! Both lambda parameter annotations are correct.
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_map_with_unannotated_lambda_and_array_index() {
    // Test that bidirectional checking infers lambda parameter type from array type
    let source = r#"
            def test : [12]i32 =
              let edges : [12][2]i32 = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]] in
              map (\e -> e[0]) edges
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed! Bidirectional checking should infer e : [2]i32 from edges
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_lambda_with_tuple_pattern() {
    // Test that lambdas with tuple patterns work
    let source = r#"
            def test : (i32, i32) -> i32 =
              \(x, y) -> x + y
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_lambda_with_wildcard_in_tuple() {
    // Test that lambdas with wildcard in tuple patterns work
    let source = r#"
            def test : (i32, i32) -> i32 =
              \(_, acc) -> acc
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_loop_with_tuple_pattern() {
    // Test that loops with tuple patterns work after defunctionalization
    let source = r#"
            def test : i32 =
              loop (idx, acc) = (0, 10) while idx < 5 do
                (idx + 1, acc + idx)
        "#;

    use crate::constant_folding::ConstantFolder;
    use crate::defunctionalization::Defunctionalizer;
    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // Run constant folding
    let mut folder = ConstantFolder::new();
    let folded_program = folder.fold_program(&program).unwrap();

    // Run defunctionalization to convert Loop to InternalLoop
    let node_counter = parser.take_node_counter();
    let type_context = polytype::Context::default();
    let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_context);
    let defunc_program = defunc.defunctionalize_program(&folded_program).unwrap();

    // Type check the defunctionalized program
    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&defunc_program) {
        Ok(_) => {
            panic!("Type checking should fail - loop returns tuple but assigned to i32");
        }
        Err(_) => {
            // Should fail - type mismatch (loop returns (i32, i32) but def expects i32)
        }
    }
}

#[test]
fn test_loop_with_tuple_pattern_returns_tuple() {
    // Test that loops with tuple patterns correctly return a tuple type
    let source = r#"
            def test : (i32, i32) =
              loop (idx, acc) = (0, 10) while idx < 5 do
                (idx + 1, acc + idx)
        "#;

    use crate::constant_folding::ConstantFolder;
    use crate::defunctionalization::Defunctionalizer;
    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // Run constant folding
    let mut folder = ConstantFolder::new();
    let folded_program = folder.fold_program(&program).unwrap();

    // Run defunctionalization to convert Loop to InternalLoop
    let node_counter = parser.take_node_counter();
    let type_context = polytype::Context::default();
    let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_context);
    let defunc_program = defunc.defunctionalize_program(&folded_program).unwrap();

    // Type check the defunctionalized program
    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&defunc_program) {
        Ok(_) => {
            // Should succeed - loop returns (i32, i32) tuple
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

fn contains_type_variable(ty: &Type) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(_, args) => args.iter().any(contains_type_variable),
    }
}

/// Helper to parse, fold, and defunctionalize source code
fn parse_and_defunctionalize(source: &str) -> crate::ast::Program {
    use crate::constant_folding::ConstantFolder;
    use crate::defunctionalization::Defunctionalizer;
    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // Run constant folding
    let mut folder = ConstantFolder::new();
    let folded_program = folder.fold_program(&program).unwrap();

    // Run defunctionalization to convert Loop to InternalLoop and map to loops
    let node_counter = parser.take_node_counter();
    let type_context = polytype::Context::default();
    let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_context);
    defunc.defunctionalize_program(&folded_program).unwrap()
}

#[test]
fn test_map_with_array_size_inference() {
    let source = r#"
def test : [8]i32 =
  let arr = [1, 2, 3, 4, 5, 6, 7, 8] in
  map (\x -> x + 1) arr
"#;

    use crate::constant_folding::ConstantFolder;
    use crate::defunctionalization::Defunctionalizer;
    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // Run constant folding
    let mut folder = ConstantFolder::new();
    let folded_program = folder.fold_program(&program).unwrap();

    // Run defunctionalization to convert map to InternalLoop
    let node_counter = parser.take_node_counter();
    let type_context = polytype::Context::default();
    let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_context);
    let defunc_program = defunc.defunctionalize_program(&folded_program).unwrap();

    // Print defunctionalized AST
    eprintln!("Defunctionalized AST:");
    eprintln!("{}", crate::diags::AstFormatter::format_program(&defunc_program));

    // Type check the defunctionalized program with shared context and ascription variables
    let (type_context, ascription_variables) = defunc.take();
    let mut checker = TypeChecker::new_with_context(type_context, ascription_variables);
    checker.load_builtins().unwrap();

    match checker.check_program(&defunc_program) {
        Ok(type_table) => {
            // Check that all types are fully resolved (no Variables)
            for (node_id, ty) in &type_table {
                if matches!(ty, Type::Variable(_)) || contains_type_variable(ty) {
                    panic!(
                        "Type table contains unresolved type variable at {:?}: {:?}",
                        node_id, ty
                    );
                }
            }
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_let_polymorphism() {
    // Test that let-bound values are properly generalized
    // Without generalization, this would fail because id would be monomorphic
    let source = r#"
            def test : bool =
                let id = \x -> x in
                let test1 : i32 = id ??? in
                let test2 : bool = id ??? in
                test2
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_warnings) => {
            // Should succeed - id is polymorphic ∀a. a -> a
            // Without generalization, this would fail because id would be monomorphic
            // and couldn't be used at both i32 and bool
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_top_level_polymorphism() {
    // Test that top-level let/def declarations are generalized
    let source = r#"
            def id = \x -> x
            def test1 : i32 = id ???
            def test2 : bool = id ???
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_warnings) => {
            // Should succeed - id is polymorphic ∀a. a -> a
            // Without generalization, this would fail because id would be monomorphic
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_polymorphic_id_tuple() {
    // Classic HM polymorphism test: let id = \x -> x in (id 5, id true)
    let source = r#"
            def test =
                let id = \x -> x in
                (id ???, id ???)
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_warnings) => {
            // Should succeed - id is polymorphic and can be used at multiple types
            // Without generalization, this would fail because first use would fix id's type
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_qualified_name_sqrt() {
    // Test that qualified names like f32.sqrt type check correctly
    let source = r#"
            def test : f32 = f32.sqrt 4.0f32
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed - f32.sqrt is a valid builtin
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_nested_array_indexing() {
    // Test that nested array indexing type inference works
    // Reproduces the de_rasterizer.wyn issue: e[0] where e : [2]i32
    let source = r#"
            def test =
                let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                let e : [2]i32 = edges[0] in
                let idx : i32 = e[0] in
                verts[idx]
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed - e[0] should be inferred as i32
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_nested_array_indexing_in_lambda() {
    // Test that nested array indexing works inside a lambda in map
    // This reproduces the actual de_rasterizer.wyn pattern:
    // map (\e -> verts[e[0]]) edges
    let source = r#"
            def test =
                let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                map (\e -> verts[e[0]]) edges
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed - e[0] should be inferred as i32 from e : [2]i32
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_nested_array_indexing_with_literal() {
    // Test with array literal directly in map call, without type annotation
    // This is closer to the de_rasterizer pattern
    let source = r#"
            def test =
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                map (\e -> verts[e[0]]) [[0,1], [1,2], [2,0]]
        "#;

    use crate::lexer;
    use crate::parser::Parser;

    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_nested_array_indexing_in_loop() {
    // Test nested array indexing inside a loop with map
    // Exact de_rasterizer pattern: loop with map that indexes with e[0]
    let source = r#"
            def test =
                loop (idx, acc) = (0i32, 0.0f32) while idx < 2 do
                    let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                    let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
                    let result = map (\e -> verts[e[0]]) edges in
                    (idx + 1, acc + 1.0f32)
        "#;

    let program = parse_and_defunctionalize(source);

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_function_call_in_lambda_inside_loop() {
    // Test that top-level functions are visible inside nested contexts
    // Reproduces de_rasterizer pattern: calling a helper function from within
    // a lambda that's inside a map inside a loop, with captured variables
    // and nested array indexing
    let source = r#"
            def line (weight:f32) (p:f32) (p0:f32) (p1:f32) (w:f32) : f32 =
              weight + p + p0 + p1 + w

            def main =
              let weight = 1.0f32 in
              let uv = 2.0f32 in
              let line_width = 3.0f32 in
              loop (idx, acc) = (0i32, 0.0f32) while idx < 2 do
                let edges : [2][2]i32 = [[0,1], [1,2]] in
                let verts1 : [3]f32 = [1.0f32, 2.0f32, 3.0f32] in
                let result = map (\e ->
                  let a = verts1[e[0]] in
                  let b = verts1[e[1]] in
                  line weight uv a b line_width)
                  edges in
                (idx + 1, acc + 1.0f32)
        "#;

    let program = parse_and_defunctionalize(source);

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed - helper should be visible in lambda
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

#[test]
fn test_size_parameter_binding() {
    // Test that size parameters are properly bound and substituted
    let source = r#"
def identity [n] (xs: [n]i32): [n]i32 = xs

def test : [5]i32 =
  let arr = [1, 2, 3, 4, 5] in
  identity arr
"#;

    let program = parse_and_defunctionalize(source);

    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(_) => {
            // Should succeed - identity preserves array size
        }
        Err(e) => {
            panic!("Type checking should succeed but failed with: {:?}", e);
        }
    }
}

/// Test that type variables in internal loops get properly resolved through back edges
/// This is a minimal reproduction of the issue with map's __alloc_array
#[test]
fn test_internal_loop_type_variable_resolution() {
    use crate::ast::{
        Decl, Declaration, ExprKind, InternalLoop, NodeCounter, PatternKind, PhiVar, Program, Span,
    };

    let mut nc = NodeCounter::new();

    // Build a minimal internal loop:
    // def test: i32 =
    //   let init_val = ??? in   // type variable ?0
    //   internal_loop
    //     loop_phi acc = [init: init_val] [next: __body_result]
    //     while acc < 10
    //     body: acc + 1
    //
    // The issue: init_val has type ?0, body returns i32,
    // but if we don't unify next_expr type with loop_var type, ?0 stays unresolved

    // init_val = ??? (type hole creates a fresh type variable)
    let init_val_expr = nc.mk_node(ExprKind::TypeHole, Span::dummy());

    // acc (identifier in condition and body)
    let acc_ident = nc.mk_node(ExprKind::Identifier("acc".to_string()), Span::dummy());
    let acc_ident2 = nc.mk_node(ExprKind::Identifier("acc".to_string()), Span::dummy());

    // acc < 10
    let ten = nc.mk_node(ExprKind::IntLiteral(10), Span::dummy());
    let condition = nc.mk_node(
        ExprKind::BinaryOp(
            crate::ast::BinaryOp { op: "<".to_string() },
            Box::new(acc_ident),
            Box::new(ten),
        ),
        Span::dummy(),
    );

    // acc + 1
    let one = nc.mk_node(ExprKind::IntLiteral(1), Span::dummy());
    let body = nc.mk_node(
        ExprKind::BinaryOp(
            crate::ast::BinaryOp { op: "+".to_string() },
            Box::new(acc_ident2),
            Box::new(one),
        ),
        Span::dummy(),
    );

    // __body_result (reference to loop body result for next iteration)
    let body_result = nc.mk_node(ExprKind::Identifier("__body_result".to_string()), Span::dummy());

    // The internal loop
    let internal_loop = nc.mk_node(
        ExprKind::InternalLoop(InternalLoop {
            phi_vars: vec![PhiVar {
                init_name: "__init_acc".to_string(),
                init_expr: Box::new(init_val_expr),
                loop_var_name: "acc".to_string(),
                loop_var_type: None, // Will be inferred from init
                next_expr: body_result,
            }],
            condition: Some(Box::new(condition)),
            body: Box::new(body),
        }),
        Span::dummy(),
    );

    let decl = Declaration::Decl(Decl {
        keyword: "def",
        attributes: vec![],
        name: "test".to_string(),
        size_params: vec![],
        type_params: vec![],
        params: vec![],
        ty: Some(crate::ast::types::i32()), // Return type is i32
        body: internal_loop,
    });

    let program = Program {
        declarations: vec![decl],
    };

    // Type check
    let mut checker = TypeChecker::new();
    checker.load_builtins().unwrap();

    match checker.check_program(&program) {
        Ok(type_table) => {
            // Check that all types are fully resolved
            for (node_id, ty) in &type_table {
                if matches!(ty, Type::Variable(_)) || contains_type_variable(ty) {
                    panic!(
                        "Type table contains unresolved type variable at {:?}: {:?}",
                        node_id, ty
                    );
                }
            }
        }
        Err(e) => {
            panic!("Type checking failed: {:?}", e);
        }
    }
}

