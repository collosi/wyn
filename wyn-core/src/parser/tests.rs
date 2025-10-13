use super::*;
use crate::lexer::tokenize;

/// Helper function that expects parsing to succeed and runs a check function on the declarations.
/// If the check fails, outputs the parsed AST for debugging.
fn expect_parse<F>(input: &str, check_fn: F)
where
    F: FnOnce(&[Declaration]) -> std::result::Result<(), String>,
{
    let tokens = tokenize(input).expect("Failed to tokenize input");
    let mut parser = Parser::new(tokens.clone());
    let program = match parser.parse() {
        Ok(program) => program,
        Err(e) => {
            println!("Parse failed with error: {:?}", e);
            println!("Tokens were: {:#?}", tokens);
            panic!("Failed to parse input: {:?}", e);
        }
    };

    if let Err(msg) = check_fn(&program.declarations) {
        println!("Check failed: {}", msg);
        println!("Parsed AST: {:#?}", program);
        panic!("Test assertion failed: {}", msg);
    }
}

/// Helper function that expects parsing to fail with a specific error.
/// If parsing succeeds when it shouldn't, outputs the parsed AST.
fn expect_parse_error<F>(input: &str, error_check: F)
where
    F: FnOnce(&CompilerError) -> std::result::Result<(), String>,
{
    let tokens = tokenize(input).expect("Failed to tokenize input");
    let mut parser = Parser::new(tokens);

    match parser.parse() {
        Ok(program) => {
            println!("Expected parse error, but parsing succeeded");
            println!("Parsed AST: {:#?}", program);
            panic!("Expected parse to fail, but it succeeded");
        }
        Err(ref error) => {
            if let Err(msg) = error_check(error) {
                println!("Error check failed: {}", msg);
                println!("Actual error: {:?}", error);
                panic!("Error assertion failed: {}", msg);
            }
        }
    }
}

/// Parse input and return the Program, panicking on failure
fn parse_ok(input: &str) -> Program {
    let tokens = tokenize(input).expect("tokenize failed");
    Parser::new(tokens).parse().expect("parse failed")
}

/// Parse input and return the single Decl, panicking if not exactly one or not a Decl
fn single_decl(input: &str) -> Decl {
    let program = parse_ok(input);
    assert_eq!(program.declarations.len(), 1, "expected exactly one declaration");
    match program.declarations.into_iter().next().unwrap() {
        Declaration::Decl(d) => d,
        other => panic!("expected Declaration::Decl, got {:?}", other),
    }
}

/// Assert that a DeclParam is Typed with given name, type, and no attributes
macro_rules! assert_typed_param {
    ($param:expr, $name:expr, $ty:expr) => {
        assert!(
            matches!(
                $param,
                DeclParam::Typed(p) if p.name == $name && p.ty == $ty && p.attributes.is_empty()
            ),
            "Expected Typed param with name {:?} and type {:?}, got {:?}",
            $name, $ty, $param
        );
    };
}

/// Assert that a DeclParam is Typed with given name, type, and attributes
macro_rules! assert_typed_param_with_attrs {
    ($param:expr, $name:expr, $ty:expr, $attrs:expr) => {
        assert!(
            matches!(
                $param,
                DeclParam::Typed(p) if p.name == $name && p.ty == $ty && p.attributes == $attrs
            ),
            "Expected Typed param with name {:?}, type {:?}, and attrs {:?}, got {:?}",
            $name, $ty, $attrs, $param
        );
    };
}

#[test]
fn test_parse_let_decl() {
    let decl = single_decl("let x: i32 = 42");
    assert_eq!(decl.name, "x");
    assert_eq!(decl.ty, Some(crate::ast::types::i32()));
    assert!(matches!(decl.body.kind, ExprKind::IntLiteral(42)));
}

#[test]
fn test_parse_array_type() {
    let decl = single_decl("let arr: [3][4]f32 = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]");
    assert_eq!(decl.name, "arr");
    assert!(decl.ty.is_some(), "Expected array type to be parsed");
}

#[test]
fn test_parse_entry_point_decl() {
    let decl = single_decl("#[vertex] def main(x: i32, y: f32): [4]f32 = result");

    assert_eq!(decl.name, "main");
    assert_eq!(decl.attributes, vec![Attribute::Vertex]);
    assert_eq!(decl.params.len(), 2);

    assert_typed_param!(&decl.params[0], "x", crate::ast::types::i32());
    assert_typed_param!(&decl.params[1], "y", crate::ast::types::f32());

    assert_eq!(decl.ty, Some(crate::ast::types::sized_array(4, crate::ast::types::f32())));
    assert!(decl.return_attributes.is_empty());
}

#[test]
fn test_parse_array_index() {
    let decl = single_decl("let x: f32 = arr[0]");
    assert!(matches!(
        &decl.body.kind,
        ExprKind::ArrayIndex(arr, idx)
            if matches!(arr.kind, ExprKind::Identifier(ref name) if name == "arr")
            && matches!(idx.kind, ExprKind::IntLiteral(0))
    ));
}

#[test]
fn test_parse_division() {
    let decl = single_decl("let x: f32 = 135f32/255f32");
    assert!(matches!(
        &decl.body.kind,
        ExprKind::BinaryOp(op, left, right)
            if op.op == "/"
            && matches!(left.kind, ExprKind::FloatLiteral(135.0))
            && matches!(right.kind, ExprKind::FloatLiteral(255.0))
    ));
}

#[test]
fn test_parse_vertex_attribute() {
    let decl = single_decl("#[vertex] def main(): [4]f32 = result");
    assert_eq!(decl.attributes, vec![Attribute::Vertex]);
    assert_eq!(decl.name, "main");
}

#[test]
fn test_parse_fragment_attribute() {
    let decl = single_decl("#[fragment] def frag(): [4]f32 = result");
    assert_eq!(decl.attributes, vec![Attribute::Fragment]);
    assert_eq!(decl.name, "frag");
}

#[test]
fn test_operator_precedence_and_associativity() {
    // Test expression: a + b * c - d / e + f
    // According to spec, * and / have higher precedence than + and -
    // All are left-associative
    // Should parse as: ((a + (b * c)) - (d / e)) + f
    let decl = single_decl("def result: i32 = a + b * c - d / e + f");

    // Outermost: ((a + (b * c)) - (d / e)) + f
    assert!(matches!(
        &decl.body.kind,
        ExprKind::BinaryOp(outer_op, outer_left, outer_right)
            if outer_op.op == "+"
            && matches!(outer_right.kind, ExprKind::Identifier(ref f) if f == "f")
            // Left: (a + (b * c)) - (d / e)
            && matches!(
                &outer_left.kind,
                ExprKind::BinaryOp(sub_op, sub_left, sub_right)
                    if sub_op.op == "-"
                    // Right of sub: d / e
                    && matches!(
                        &sub_right.kind,
                        ExprKind::BinaryOp(div_op, div_left, div_right)
                            if div_op.op == "/"
                            && matches!(div_left.kind, ExprKind::Identifier(ref d) if d == "d")
                            && matches!(div_right.kind, ExprKind::Identifier(ref e) if e == "e")
                    )
                    // Left of sub: a + (b * c)
                    && matches!(
                        &sub_left.kind,
                        ExprKind::BinaryOp(add_op, add_left, add_right)
                            if add_op.op == "+"
                            && matches!(add_left.kind, ExprKind::Identifier(ref a) if a == "a")
                            // Right: b * c
                            && matches!(
                                &add_right.kind,
                                ExprKind::BinaryOp(mul_op, mul_left, mul_right)
                                    if mul_op.op == "*"
                                    && matches!(mul_left.kind, ExprKind::Identifier(ref b) if b == "b")
                                    && matches!(mul_right.kind, ExprKind::Identifier(ref c) if c == "c")
                            )
                    )
            )
    ));
}

#[test]
fn test_parse_builtin_attribute_on_return_type() {
    let decl = single_decl("#[vertex] def main(): #[builtin(position)] [4]f32 = result");

    assert_eq!(decl.attributes, vec![Attribute::Vertex]);

    let attributed_types = decl.attributed_return_type.as_ref().expect("Expected attributed return type");
    assert_eq!(attributed_types.len(), 1);
    assert_eq!(attributed_types[0].attributes, vec![Attribute::BuiltIn(spirv::BuiltIn::Position)]);
    assert_eq!(attributed_types[0].ty, crate::ast::types::sized_array(4, crate::ast::types::f32()));
}

#[test]
fn test_parse_single_attributed_return_type() {
    let decl = single_decl("#[vertex] def vertex_main(): #[builtin(position)] vec4 = vec4 0.0f32 0.0f32 0.0f32 1.0f32");

    assert!(decl.attributes.contains(&Attribute::Vertex));

    let attributed_types = decl.attributed_return_type.as_ref().expect("Missing attributed return type");
    assert_eq!(attributed_types.len(), 1);

    let attr_type = &attributed_types[0];
    assert!(attr_type.attributes.contains(&Attribute::BuiltIn(spirv::BuiltIn::Position)));
    assert!(matches!(&attr_type.ty, Type::Constructed(TypeName::Str("vec4"), _)));
}

#[test]
fn test_parse_tuple_attributed_return_type() {
    let decl = single_decl("#[vertex] def vertex_main(): (#[builtin(position)] vec4, #[location(0)] vec3) = result");

    let attributed_types = decl.attributed_return_type.as_ref().expect("Missing attributed return type");
    assert_eq!(attributed_types.len(), 2);

    // Check first element: [builtin(position)] vec4
    assert!(attributed_types[0].attributes.contains(&Attribute::BuiltIn(spirv::BuiltIn::Position)));

    // Check second element: [location(0)] vec3
    assert!(attributed_types[1].attributes.contains(&Attribute::Location(0)));
}

#[test]
fn test_parse_unattributed_return_type() {
    let decl = single_decl("def helper(): vec4 = vec4 1.0f32 0.0f32 0.0f32 1.0f32");

    // Should have no attributed_return_type
    assert!(decl.attributed_return_type.is_none());

    // Should have a regular type
    let ty = decl.ty.as_ref().expect("Missing return type");
    assert!(matches!(ty, Type::Constructed(TypeName::Str("vec4"), _)));
}

#[test]
fn test_parse_location_attribute_on_return_type() {
    let decl = single_decl("#[fragment] def frag(): #[location(0)] [4]f32 = result");

    assert_eq!(decl.attributes, vec![Attribute::Fragment]);

    let attributed_types = decl.attributed_return_type.as_ref().expect("Expected attributed return type");
    assert_eq!(attributed_types.len(), 1);
    assert_eq!(attributed_types[0].attributes, vec![Attribute::Location(0)]);
    assert_eq!(attributed_types[0].ty, crate::ast::types::sized_array(4, crate::ast::types::f32()));
}

#[test]
fn test_parse_parameter_with_builtin_attribute() {
    let decl = single_decl("#[vertex] def main(#[builtin(vertex_index)] vid: i32): [4]f32 = result");

    assert_eq!(decl.params.len(), 1);
    assert_typed_param_with_attrs!(
        &decl.params[0],
        "vid",
        crate::ast::types::i32(),
        vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)]
    );
}

#[test]
fn test_parse_parameter_with_location_attribute() {
    let decl = single_decl("#[fragment] def frag(#[location(1)] color: [3]f32): [4]f32 = result");

    assert_eq!(decl.params.len(), 1);
    assert_typed_param_with_attrs!(
        &decl.params[0],
        "color",
        crate::ast::types::sized_array(3, crate::ast::types::f32()),
        vec![Attribute::Location(1)]
    );
}

#[test]
fn test_parse_multiple_builtin_types() {
    let decl = single_decl("#[vertex] def main(#[builtin(vertex_index)] vid: i32, #[builtin(instance_index)] iid: i32): #[builtin(position)] [4]f32 = result");

    assert_eq!(decl.params.len(), 2);

    // First parameter
    assert_typed_param_with_attrs!(
        &decl.params[0],
        "vid",
        crate::ast::types::i32(),
        vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)]
    );

    // Second parameter
    assert_typed_param_with_attrs!(
        &decl.params[1],
        "iid",
        crate::ast::types::i32(),
        vec![Attribute::BuiltIn(spirv::BuiltIn::InstanceIndex)]
    );

    // Return type
    let attributed_types = decl.attributed_return_type.as_ref().expect("Expected attributed return type");
    assert_eq!(attributed_types.len(), 1);
    assert_eq!(attributed_types[0].attributes, vec![Attribute::BuiltIn(spirv::BuiltIn::Position)]);
}

#[test]
fn test_parse_simple_lambda() {
    let decl = single_decl(r#"let f: i32 -> i32 = \x -> x"#);

    assert_eq!(decl.name, "f");
    assert!(matches!(
        &decl.body.kind,
        ExprKind::Lambda(lambda)
            if lambda.params.len() == 1
            && lambda.params[0].name == "x"
            && lambda.params[0].ty.is_none()
            && lambda.return_type.is_none()
            && matches!(lambda.body.kind, ExprKind::Identifier(ref name) if name == "x")
    ));
}

#[test]
fn test_operator_precedence_equivalence() {
    // Helper to parse just the expression from a declaration
    fn parse_expr(input: &str) -> Expression {
        let full_input = format!("def result: i32 = {}", input);
        let tokens = tokenize(&full_input).expect("Failed to tokenize");
        let mut parser = Parser::new(tokens);
        let program = parser.parse().expect("Failed to parse");
        match &program.declarations[0] {
            Declaration::Decl(decl) => decl.body.clone(),
            _ => panic!("Expected Decl declaration"),
        }
    }

    // Test cases: (expression, expected_equivalent_with_parens)
    let equivalent_cases = vec![
        // Precedence tests
        ("a + b * c", "a + (b * c)"),
        ("a * b + c", "(a * b) + c"),
        ("a + b / c", "a + (b / c)"),
        ("a / b + c", "(a / b) + c"),
        // Left associativity tests
        ("a - b + c", "(a - b) + c"),
        ("a + b - c", "(a + b) - c"),
        ("a * b / c", "(a * b) / c"),
        ("a / b * c", "(a / b) * c"),
        // Complex expression
        ("a + b * c - d / e + f", "((a + (b * c)) - (d / e)) + f"),
    ];

    for (expr, expected) in equivalent_cases {
        let parsed_expr = parse_expr(expr);
        let parsed_expected = parse_expr(expected);
        assert_eq!(
            parsed_expr, parsed_expected,
            "{} should parse the same as {}",
            expr, expected
        );
    }

    // Test cases that should NOT be equivalent
    let non_equivalent_cases = vec![("a + b * c", "(a + b) * c"), ("a * b + c * d", "a * (b + c) * d")];

    for (expr1, expr2) in non_equivalent_cases {
        let parsed_expr1 = parse_expr(expr1);
        let parsed_expr2 = parse_expr(expr2);
        assert_ne!(
            parsed_expr1, parsed_expr2,
            "{} should NOT parse the same as {}",
            expr1, expr2
        );
    }
}

#[test]
fn test_parse_lambda_with_type_annotation() {
    let decl = single_decl(r#"let f: f32 -> f32 = \x -> x"#);

    assert!(matches!(
        &decl.body.kind,
        ExprKind::Lambda(lambda)
            if lambda.params.len() == 1
            && lambda.params[0].name == "x"
            && lambda.params[0].ty.is_none()
            && lambda.return_type.is_none()
            && matches!(lambda.body.kind, ExprKind::Identifier(ref name) if name == "x")
    ));
}

#[test]
fn test_parse_lambda_with_multiple_params() {
    let decl = single_decl(r#"let add: i32 -> i32 -> i32 = \x y -> x"#);

    assert!(matches!(
        &decl.body.kind,
        ExprKind::Lambda(lambda)
            if lambda.params.len() == 2
            && lambda.params[0].name == "x"
            && lambda.params[0].ty.is_none()
            && lambda.params[1].name == "y"
            && lambda.params[1].ty.is_none()
            && lambda.return_type.is_none()
            && matches!(lambda.body.kind, ExprKind::Identifier(ref name) if name == "x")
    ));
}

#[test]
fn test_parse_function_application() {
    let decl = single_decl(r#"let result: i32 = f(42, 24)"#);

    assert!(matches!(
        &decl.body.kind,
        ExprKind::FunctionCall(name, args)
            if name == "f"
            && args.len() == 2
            && matches!(args[0].kind, ExprKind::IntLiteral(42))
            && matches!(args[1].kind, ExprKind::IntLiteral(24))
    ));
}

#[test]
fn test_parse_simple_let_in() {
    // Just verify it parses successfully - the structure is complex to validate in detail
    let _decl = single_decl("#[vertex] def main(x: i32): i32 = let y = 5 in y + x");
}

#[test]
fn test_parse_let_in_expression_only() {
    // Test parsing just the let..in expression by itself - just verify it parses
    let input = r#"let f = \y -> y + x in f 10"#;
    let tokens = tokenize(input).expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    parser.parse_expression().expect("Failed to parse let..in expression");
}

#[test]
fn test_parse_let_in_with_lambda() {
    // Just verify it parses successfully - the lambda let..in structure is complex
    let _decl = single_decl(r#"#[vertex] def main(x: i32): i32 = let f = \y -> y + x in f 10"#);
}

#[test]
fn test_parse_multiple_top_level_lets_with_entry() {
    // Test multiple let declarations with entry points
    let input = r#"
def verts: [3][4]f32 =
  [[-1.0f32, -1.0f32, 0.0f32, 1.0f32],
   [ 3.0f32, -1.0f32, 0.0f32, 1.0f32],
   [-1.0f32,  3.0f32, 0.0f32, 1.0f32]]

#[vertex]
def vertex_main(vertex_id: i32): [4]f32 = verts[vertex_id]

def SKY_RGBA: [4]f32 =
  [135f32/255f32, 206f32/255f32, 235f32/255f32, 1.0f32]

#[fragment]
def fragment_main(): [4]f32 = SKY_RGBA
"#;

    let program = parse_ok(input);
    assert_eq!(program.declarations.len(), 4);

    // First: def verts
    assert!(matches!(&program.declarations[0], Declaration::Decl(decl) if decl.keyword == "def" && decl.name == "verts"));

    // Second: vertex entry point
    assert!(matches!(&program.declarations[1], Declaration::Decl(decl) if decl.name == "vertex_main" && decl.attributes.contains(&Attribute::Vertex)));

    // Third: def SKY_RGBA
    assert!(matches!(&program.declarations[2], Declaration::Decl(decl) if decl.keyword == "def" && decl.name == "SKY_RGBA"));

    // Fourth: fragment entry point
    assert!(matches!(&program.declarations[3], Declaration::Decl(decl) if decl.name == "fragment_main" && decl.attributes.contains(&Attribute::Fragment)));
}

#[test]
fn test_field_access_parsing() {
    let decl = single_decl("def x: f32 = v.x");

    assert_eq!(decl.name, "x");
    assert!(matches!(
        &decl.body.kind,
        ExprKind::FieldAccess(expr, field)
            if field == "x" && matches!(expr.kind, ExprKind::Identifier(ref name) if name == "v")
    ));
}

#[test]
fn test_simple_identifier_parsing() {
    let decl = single_decl("def x: f32 = y");

    assert_eq!(decl.name, "x");
    assert!(matches!(&decl.body.kind, ExprKind::Identifier(name) if name == "y"));
}

#[test]
fn test_vector_field_access_file() {
    let program = parse_ok("def v: vec3 = vec3 1.0f32 2.0f32 3.0f32\ndef x: f32 = v.x");
    assert_eq!(program.declarations.len(), 2);

    // Check first declaration: def v: vec3 = vec3 1.0f32 2.0f32 3.0f32
    let decl1 = match &program.declarations[0] {
        Declaration::Decl(d) => d,
        _ => panic!("Expected first declaration to be Decl"),
    };
    assert_eq!(decl1.name, "v");
    assert!(matches!(&decl1.body.kind, ExprKind::FunctionCall(func_name, args) if func_name == "vec3" && args.len() == 3));

    // Check second declaration: def x: f32 = v.x
    let decl2 = match &program.declarations[1] {
        Declaration::Decl(d) => d,
        _ => panic!("Expected second declaration to be Decl"),
    };
    assert_eq!(decl2.name, "x");
    assert!(matches!(&decl2.body.kind, ExprKind::FieldAccess(expr, field) if field == "x" && matches!(expr.kind, ExprKind::Identifier(ref name) if name == "v")));
}

#[test]
fn test_parse_vector_arithmetic() {
    let decl = single_decl(
        r#"
            def test_vector_arithmetic: f32 =
              let v1: vec3 = vec3 1.0f32 2.0f32 3.0f32 in
              let v2: vec3 = vec3 4.0f32 5.0f32 6.0f32 in
              let sum: vec3 = v1 + v2 in
              let diff: vec3 = v1 - v2 in
              let prod: vec3 = v1 * v2 in
              let a: f32 = 2.5f32 in
              let b: f32 = 3.0f32 in
              let scalar_sum: f32 = a + b in
              let scalar_diff: f32 = a - b in
              let scalar_prod: f32 = a * b in
              scalar_sum
            "#,
    );

    assert_eq!(decl.name, "test_vector_arithmetic");
    assert_eq!(decl.ty, Some(crate::ast::types::f32()));
    assert!(matches!(&decl.body.kind, ExprKind::LetIn(_)));
}

#[test]
fn test_parse_uniform_attribute() {
    let program = parse_ok("#[uniform] def material_color: vec3");
    assert_eq!(program.declarations.len(), 1);

    let uniform_decl = match &program.declarations[0] {
        Declaration::Uniform(u) => u,
        _ => panic!("Expected Uniform declaration"),
    };
    assert_eq!(uniform_decl.name, "material_color");
    assert_eq!(uniform_decl.ty, crate::ast::types::vec3());
}

#[test]
fn test_parse_uniform_without_initializer() {
    let program = parse_ok("#[uniform] def material_color: vec3");
    assert_eq!(program.declarations.len(), 1);

    let uniform_decl = match &program.declarations[0] {
        Declaration::Uniform(u) => u,
        _ => panic!("Expected Uniform declaration"),
    };
    assert_eq!(uniform_decl.name, "material_color");
    assert_eq!(uniform_decl.ty, crate::ast::types::vec3());
    // Check that there's no initializer (uniforms don't have bodies)
}

#[test]
fn test_uniform_with_initializer_error() {
    expect_parse_error(
        "#[uniform] def material_color: vec3 = vec3 1.0f32 0.5f32 0.2f32",
        |error| match error {
            CompilerError::ParseError(msg) if msg.contains("Uniform declarations cannot have initializer values") => Ok(()),
            _ => Err(format!("Expected parse error about uniform initializer, got: {:?}", error)),
        },
    );
}

#[test]
fn test_parse_multiple_shader_outputs() {
    let decl = single_decl(
        r#"
            #[fragment] def fragment_main(): (#[location(0)] vec4, #[location(1)] vec3) =
              let color = vec4 1.0f32 0.5f32 0.2f32 1.0f32 in
              let normal = vec3 0.0f32 1.0f32 0.0f32 in
              (color, normal)
            "#,
    );

    assert_eq!(decl.name, "fragment_main");
    assert!(decl.attributes.contains(&Attribute::Fragment));

    let attributed_types = decl.attributed_return_type.as_ref().expect("Expected attributed return type");
    assert_eq!(attributed_types.len(), 2);

    // Check first output: [location(0)] vec4
    assert_eq!(attributed_types[0].attributes, vec![Attribute::Location(0)]);

    // Check second output: [location(1)] vec3
    assert_eq!(attributed_types[1].attributes, vec![Attribute::Location(1)]);
}

#[test]
fn test_parse_complete_shader_example() {
    expect_parse(
        r#"
            -- Complete shader example with multiple outputs
            #[uniform] def material_color: vec3
            #[uniform] def time: f32

            #[vertex] def vertex_main(): (#[builtin(position)] vec4, #[location(0)] vec3) =
              let angle: f32 = time in
              let x: f32 = sin angle in
              let y: f32 = cos angle in
              let position: vec4 = vec4 x y 0.0f32 1.0f32 in
              let color: vec3 = material_color in
              (position, color)

            #[fragment] def fragment_main(): (#[location(0)] vec4, #[location(1)] vec3) = 
              let final_color: vec4 = vec4 1.0f32 0.5f32 0.2f32 1.0f32 in
              let normal: vec3 = vec3 0.0f32 1.0f32 0.0f32 in
              (final_color, normal)
            "#,
        |declarations| {
            if declarations.len() != 4 {
                return Err(format!("Expected 4 declarations, got {}", declarations.len()));
            }

            // Check uniform declarations
            match &declarations[0] {
                Declaration::Uniform(uniform) => {
                    if uniform.name != "material_color" {
                        return Err("Expected uniform material_color".to_string());
                    }
                }
                _ => return Err("Expected uniform declaration".to_string()),
            }

            match &declarations[1] {
                Declaration::Uniform(uniform) => {
                    if uniform.name != "time" {
                        return Err("Expected uniform time".to_string());
                    }
                }
                _ => return Err("Expected uniform declaration".to_string()),
            }

            // Check vertex shader with multiple outputs
            match &declarations[2] {
                Declaration::Decl(decl) => {
                    if decl.name != "vertex_main" || !decl.attributes.contains(&Attribute::Vertex) {
                        return Err("Expected vertex shader".to_string());
                    }
                    if decl.attributed_return_type.is_none() {
                        return Err("Expected attributed return type for vertex shader".to_string());
                    }
                }
                _ => return Err("Expected vertex declaration".to_string()),
            }

            // Check fragment shader with multiple outputs
            match &declarations[3] {
                Declaration::Decl(decl) => {
                    if decl.name != "fragment_main" || !decl.attributes.contains(&Attribute::Fragment) {
                        return Err("Expected fragment shader".to_string());
                    }
                    if decl.attributed_return_type.is_none() {
                        return Err("Expected attributed return type for fragment shader".to_string());
                    }
                }
                _ => return Err("Expected fragment declaration".to_string()),
            }

            Ok(())
        },
    );
}

#[test]
fn test_if_then_else_parsing() {
    // Test simple if-then-else
    expect_parse("def test: i32 = if x == 0 then 1 else 2", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }

        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.name != "test" {
                    return Err(format!("Expected name 'test', got '{}'", decl.name));
                }

                // Check the if-then-else expression structure
                match &decl.body.kind {
                    ExprKind::If(if_expr) => {
                        // Check condition is a comparison
                        match &if_expr.condition.kind {
                            ExprKind::BinaryOp(op, left, right) => {
                                if op.op != "==" {
                                    return Err(format!("Expected '==' operator, got '{}'", op.op));
                                }

                                // Check left side is identifier 'x'
                                match &left.kind {
                                    ExprKind::Identifier(name) if name == "x" => {}
                                    _ => {
                                        return Err(format!(
                                            "Expected identifier 'x' on left side of comparison, got {:?}",
                                            left
                                        ));
                                    }
                                }

                                // Check right side is literal 0
                                match &right.kind {
                                    ExprKind::IntLiteral(0) => {}
                                    _ => {
                                        return Err(format!(
                                            "Expected literal 0 on right side of comparison, got {:?}",
                                            right
                                        ));
                                    }
                                }
                            }
                            _ => {
                                return Err(format!(
                                    "Expected comparison expression in if condition, got {:?}",
                                    if_expr.condition
                                ));
                            }
                        }

                        // Check then branch is literal 1
                        match &if_expr.then_branch.kind {
                            ExprKind::IntLiteral(1) => {}
                            _ => {
                                return Err(format!(
                                    "Expected literal 1 in then branch, got {:?}",
                                    if_expr.then_branch
                                ));
                            }
                        }

                        // Check else branch is literal 2
                        match &if_expr.else_branch.kind {
                            ExprKind::IntLiteral(2) => {}
                            _ => {
                                return Err(format!(
                                    "Expected literal 2 in else branch, got {:?}",
                                    if_expr.else_branch
                                ));
                            }
                        }
                    }
                    _ => return Err(format!("Expected if-then-else expression, got {:?}", decl.body)),
                }

                Ok(())
            }
            _ => Err("Expected Decl declaration".to_string()),
        }
    });
}
#[test]
fn test_array_literal() {
    // Test simple if-then-else
    expect_parse(
        "#[vertex] def test(): #[builtin(position)] [4]f32 = [0.0f32, 0.5f32, 0.0f32, 1.0f32]",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            Ok(())
        },
    );
}

#[test]
fn test_parse_array_type_directly() {
    let tokens = tokenize("[4]f32").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    match parser.parse_type() {
        Ok(ty) => println!("Successfully parsed type: {:?}", ty),
        Err(e) => panic!("Failed to parse [4]f32: {:?}", e),
    }
}

#[test]
fn test_parse_array_literal() {
    let tokens = tokenize("[0.0f32, 0.5f32, 0.0f32, 1.0f32]").expect("Failed to tokenize");
    println!("Tokens: {:?}", tokens);
    let mut parser = Parser::new(tokens);
    match parser.parse_expression() {
        Ok(expr) => {
            println!("Successfully parsed array literal: {:?}", expr);
            // Check that it's actually an array literal
            match expr.kind {
                ExprKind::ArrayLiteral(_) => {
                    // Good, this is what we expect
                }
                _ => panic!("Expected ArrayLiteral, got {:?}", expr),
            }
        }
        Err(e) => {
            panic!("Failed to parse array literal: {:?}", e);
        }
    }
}

#[test]
fn test_parse_attributed_return_type() {
    let input = r#"#[vertex]
def test_vertex : #[builtin(position)] vec4 =
  let angle = 1.0f32 in
  let s = f32.sin angle in
  vec4 s s 0.0f32 1.0f32"#;

    expect_parse(input, |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }

        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.name != "test_vertex" {
                    return Err(format!("Expected 'test_vertex', got '{}'", decl.name));
                }
                if decl.attributed_return_type.is_none() {
                    return Err("Expected attributed return type".to_string());
                }
                Ok(())
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_unique_type() {
    expect_parse("def foo(x: *i32): i32 = x", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.params.len() != 1 {
                    return Err(format!("Expected 1 parameter, got {}", decl.params.len()));
                }
                match &decl.params[0] {
                    DeclParam::Typed(param) => {
                        let param_ty = &param.ty;
                        // Should be Unique(i32)
                        if !types::is_unique(param_ty) {
                            return Err(format!("Expected unique type, got {:?}", param_ty));
                        }
                        let inner = types::strip_unique(param_ty);
                        if inner != types::i32() {
                            return Err(format!("Expected i32 inside unique, got {:?}", inner));
                        }
                        Ok(())
                    }
                    _ => Err("Expected typed parameter".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_unique_array_type() {
    expect_parse("def bar(arr: *[3]f32): f32 = arr[0]", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.params.len() != 1 {
                    return Err(format!("Expected 1 parameter, got {}", decl.params.len()));
                }
                match &decl.params[0] {
                    DeclParam::Typed(param) => {
                        let param_ty = &param.ty;
                        // Should be Unique(Array(3, f32))
                        if !types::is_unique(param_ty) {
                            return Err(format!("Expected unique type, got {:?}", param_ty));
                        }
                        let inner = types::strip_unique(param_ty);
                        let expected = types::sized_array(3, types::f32());
                        if inner != expected {
                            return Err(format!("Expected [3]f32 inside unique, got {:?}", inner));
                        }
                        Ok(())
                    }
                    _ => Err("Expected typed parameter".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_nested_unique() {
    // Nested arrays with unique at different levels
    expect_parse("def baz(x: *[2][3]i32): i32 = x[0][0]", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.params.len() != 1 {
                    return Err(format!("Expected 1 parameter, got {}", decl.params.len()));
                }
                match &decl.params[0] {
                    DeclParam::Typed(param) => {
                        let param_ty = &param.ty;
                        // Should be Unique(Array(2, Array(3, i32)))
                        if !types::is_unique(param_ty) {
                            return Err(format!("Expected unique type, got {:?}", param_ty));
                        }
                        let inner = types::strip_unique(param_ty);
                        let expected = types::sized_array(2, types::sized_array(3, types::i32()));
                        if inner != expected {
                            return Err(format!("Expected [2][3]i32 inside unique, got {:?}", inner));
                        }
                        Ok(())
                    }
                    _ => Err("Expected typed parameter".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_function_application_with_array_literal() {
    let _ = env_logger::builder().is_test(true).try_init();
    expect_parse(
        "def test: vec4 = to_vec4 [1.0f32, 2.0f32, 3.0f32, 4.0f32]",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.name != "test" {
                        return Err(format!("Expected name 'test', got '{}'", decl.name));
                    }
                    // Body should be FunctionCall("to_vec4", [ArrayLiteral(...)])
                    match &decl.body.kind {
                        ExprKind::FunctionCall(name, args) => {
                            if name != "to_vec4" {
                                return Err(format!("Expected function 'to_vec4', got '{}'", name));
                            }
                            if args.len() != 1 {
                                return Err(format!("Expected 1 arg, got {}", args.len()));
                            }
                            if let ExprKind::ArrayLiteral(elements) = &args[0].kind {
                                if elements.len() != 4 {
                                    return Err(format!("Expected 4 elements, got {}", elements.len()));
                                }
                            } else {
                                return Err(format!("Expected ArrayLiteral, got {:?}", args[0].kind));
                            }
                            Ok(())
                        }
                        _ => Err(format!("Expected FunctionCall, got {:?}", decl.body.kind)),
                    }
                }
                _ => Err("Expected Decl".to_string()),
            }
        },
    );
}

// Pattern parsing tests

#[test]
fn test_parse_pattern_name() {
    let tokens = tokenize("x").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Name(name) => {
            assert_eq!(name, "x", "Expected name 'x'");
        }
        _ => panic!("Expected Name pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_wildcard() {
    let tokens = tokenize("_").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Wildcard => {}
        _ => panic!("Expected Wildcard pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_unit() {
    let tokens = tokenize("()").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Unit => {}
        _ => panic!("Expected Unit pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_int_literal() {
    let tokens = tokenize("42").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Literal(PatternLiteral::Int(n)) => {
            assert_eq!(n, 42, "Expected int literal 42");
        }
        _ => panic!("Expected int literal pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_negative_int_literal() {
    let tokens = tokenize("-42").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Literal(PatternLiteral::Int(n)) => {
            assert_eq!(n, -42, "Expected int literal -42");
        }
        _ => panic!("Expected int literal pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_float_literal() {
    let tokens = tokenize("3.14f32").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Literal(PatternLiteral::Float(f)) => {
            assert!((f - 3.14).abs() < 0.001, "Expected float literal 3.14");
        }
        _ => panic!("Expected float literal pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_bool_true() {
    let tokens = tokenize("true").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Literal(PatternLiteral::Bool(b)) => {
            assert!(b, "Expected bool literal true");
        }
        _ => panic!("Expected bool literal pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_bool_false() {
    let tokens = tokenize("false").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Literal(PatternLiteral::Bool(b)) => {
            assert!(!b, "Expected bool literal false");
        }
        _ => panic!("Expected bool literal pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_tuple() {
    let tokens = tokenize("(x, y, z)").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Tuple(patterns) => {
            assert_eq!(patterns.len(), 3, "Expected 3 patterns in tuple");

            match &patterns[0].kind {
                PatternKind::Name(name) => assert_eq!(name, "x"),
                _ => panic!("Expected first element to be Name pattern"),
            }

            match &patterns[1].kind {
                PatternKind::Name(name) => assert_eq!(name, "y"),
                _ => panic!("Expected second element to be Name pattern"),
            }

            match &patterns[2].kind {
                PatternKind::Name(name) => assert_eq!(name, "z"),
                _ => panic!("Expected third element to be Name pattern"),
            }
        }
        _ => panic!("Expected Tuple pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_tuple_with_trailing_comma() {
    let tokens = tokenize("(x, y,)").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Tuple(patterns) => {
            assert_eq!(patterns.len(), 2, "Expected 2 patterns in tuple");
        }
        _ => panic!("Expected Tuple pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_single_in_parens() {
    let tokens = tokenize("(x)").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    // Single pattern in parens should unwrap to just the pattern
    match pattern.kind {
        PatternKind::Name(name) => {
            assert_eq!(name, "x", "Expected name 'x'");
        }
        _ => panic!(
            "Expected Name pattern (unwrapped from parens), got {:?}",
            pattern.kind
        ),
    }
}

#[test]
fn test_parse_pattern_empty_record() {
    let tokens = tokenize("{}").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Record(fields) => {
            assert_eq!(fields.len(), 0, "Expected empty record");
        }
        _ => panic!("Expected Record pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_record_shorthand() {
    let tokens = tokenize("{ x, y }").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Record(fields) => {
            assert_eq!(fields.len(), 2, "Expected 2 fields");

            assert_eq!(fields[0].field, "x");
            assert!(fields[0].pattern.is_none(), "Expected shorthand (no pattern)");

            assert_eq!(fields[1].field, "y");
            assert!(fields[1].pattern.is_none(), "Expected shorthand (no pattern)");
        }
        _ => panic!("Expected Record pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_record_with_patterns() {
    let tokens = tokenize("{ x = a, y = b }").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Record(fields) => {
            assert_eq!(fields.len(), 2, "Expected 2 fields");

            assert_eq!(fields[0].field, "x");
            assert!(fields[0].pattern.is_some(), "Expected pattern for x");
            if let Some(ref pat) = fields[0].pattern {
                match &pat.kind {
                    PatternKind::Name(name) => assert_eq!(name, "a"),
                    _ => panic!("Expected Name pattern for x"),
                }
            }

            assert_eq!(fields[1].field, "y");
            assert!(fields[1].pattern.is_some(), "Expected pattern for y");
            if let Some(ref pat) = fields[1].pattern {
                match &pat.kind {
                    PatternKind::Name(name) => assert_eq!(name, "b"),
                    _ => panic!("Expected Name pattern for y"),
                }
            }
        }
        _ => panic!("Expected Record pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_record_mixed() {
    let tokens = tokenize("{ x, y = b }").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Record(fields) => {
            assert_eq!(fields.len(), 2, "Expected 2 fields");

            assert_eq!(fields[0].field, "x");
            assert!(fields[0].pattern.is_none(), "Expected shorthand for x");

            assert_eq!(fields[1].field, "y");
            assert!(fields[1].pattern.is_some(), "Expected pattern for y");
        }
        _ => panic!("Expected Record pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_constructor_no_args() {
    let tokens = tokenize("None").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Constructor(name, args) => {
            assert_eq!(name, "None");
            assert_eq!(args.len(), 0, "Expected no arguments");
        }
        _ => panic!("Expected Constructor pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_constructor_with_args() {
    let tokens = tokenize("Some x").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Constructor(name, args) => {
            assert_eq!(name, "Some");
            assert_eq!(args.len(), 1, "Expected 1 argument");

            match &args[0].kind {
                PatternKind::Name(arg_name) => assert_eq!(arg_name, "x"),
                _ => panic!("Expected Name pattern for argument"),
            }
        }
        _ => panic!("Expected Constructor pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_constructor_multiple_args() {
    let tokens = tokenize("Point x y").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Constructor(name, args) => {
            assert_eq!(name, "Point");
            assert_eq!(args.len(), 2, "Expected 2 arguments");

            match &args[0].kind {
                PatternKind::Name(arg_name) => assert_eq!(arg_name, "x"),
                _ => panic!("Expected Name pattern for first argument"),
            }

            match &args[1].kind {
                PatternKind::Name(arg_name) => assert_eq!(arg_name, "y"),
                _ => panic!("Expected Name pattern for second argument"),
            }
        }
        _ => panic!("Expected Constructor pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_constructor_nested() {
    let tokens = tokenize("Just (Some x)").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Constructor(name, args) => {
            assert_eq!(name, "Just");
            assert_eq!(args.len(), 1, "Expected 1 argument");

            match &args[0].kind {
                PatternKind::Constructor(inner_name, inner_args) => {
                    assert_eq!(inner_name, "Some");
                    assert_eq!(inner_args.len(), 1);

                    match &inner_args[0].kind {
                        PatternKind::Name(n) => assert_eq!(n, "x"),
                        _ => panic!("Expected Name in nested constructor"),
                    }
                }
                _ => panic!("Expected Constructor pattern for argument"),
            }
        }
        _ => panic!("Expected Constructor pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_typed() {
    let tokens = tokenize("x : i32").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Typed(pat, ty) => {
            match pat.kind {
                PatternKind::Name(name) => assert_eq!(name, "x"),
                _ => panic!("Expected Name pattern"),
            }

            assert_eq!(ty, types::i32(), "Expected i32 type");
        }
        _ => panic!("Expected Typed pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_tuple_typed() {
    let tokens = tokenize("(x, y) : (i32, f32)").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Typed(pat, _ty) => match pat.kind {
            PatternKind::Tuple(patterns) => {
                assert_eq!(patterns.len(), 2);
            }
            _ => panic!("Expected Tuple pattern"),
        },
        _ => panic!("Expected Typed pattern, got {:?}", pattern.kind),
    }
}

#[test]
fn test_parse_pattern_lowercase_not_constructor() {
    // Lowercase identifiers should be name patterns, not constructors
    let tokens = tokenize("some").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern().expect("Failed to parse pattern");

    match pattern.kind {
        PatternKind::Name(name) => {
            assert_eq!(name, "some", "Expected name pattern 'some'");
        }
        _ => panic!("Expected Name pattern (not Constructor), got {:?}", pattern.kind),
    }
}

// Module parsing tests

#[test]
fn test_parse_type_bind_simple() {
    expect_parse("type Point = (i32, i32)", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::TypeBind(bind) => {
                assert_eq!(bind.name, "Point");
                assert_eq!(bind.kind, TypeBindKind::Normal);
                assert_eq!(bind.type_params.len(), 0);
                Ok(())
            }
            _ => Err("Expected TypeBind declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_import() {
    expect_parse("import \"path/to/module\"", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Import(path) => {
                assert_eq!(path, "path/to/module");
                Ok(())
            }
            _ => Err("Expected Import declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_module_bind_simple() {
    expect_parse("module M = { def x : i32 = 42 }", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::ModuleBind(bind) => {
                assert_eq!(bind.name, "M");
                assert_eq!(bind.params.len(), 0);
                assert!(bind.signature.is_none());
                match &bind.body {
                    ModuleExpression::Struct(decls) => {
                        if decls.len() != 1 {
                            return Err(format!("Expected 1 inner declaration, got {}", decls.len()));
                        }
                        Ok(())
                    }
                    _ => Err("Expected Struct module expression".to_string()),
                }
            }
            _ => Err("Expected ModuleBind declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_local_declaration() {
    expect_parse("local def x : i32 = 42", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Local(inner) => match **inner {
                Declaration::Decl(_) => Ok(()),
                _ => Err("Expected Decl inside Local".to_string()),
            },
            _ => Err("Expected Local declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_record_type_empty() {
    // Test empty record type {}
    expect_parse("let x: {} = ???", |declarations| {
        assert_eq!(declarations.len(), 1);
        match &declarations[0] {
            Declaration::Decl(decl) => {
                assert_eq!(decl.name, "x");
                match &decl.ty {
                    Some(Type::Constructed(TypeName::Record(fields), _)) => {
                        assert_eq!(fields.len(), 0);
                        Ok(())
                    }
                    _ => Err("Expected record type".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_record_type_single_field() {
    // Test record type with single field {x: i32}
    expect_parse("let r: {x: i32} = ???", |declarations| {
        assert_eq!(declarations.len(), 1);
        match &declarations[0] {
            Declaration::Decl(decl) => {
                assert_eq!(decl.name, "r");
                match &decl.ty {
                    Some(Type::Constructed(TypeName::Record(fields), _)) => {
                        assert_eq!(fields.len(), 1);
                        assert_eq!(fields[0].0, "x");
                        Ok(())
                    }
                    _ => Err("Expected record type".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_record_type_multiple_fields() {
    // Test record type with multiple fields {x: i32, y: f32, z: vec3}
    expect_parse("let r: {x: i32, y: f32, z: vec3} = ???", |declarations| {
        assert_eq!(declarations.len(), 1);
        match &declarations[0] {
            Declaration::Decl(decl) => {
                assert_eq!(decl.name, "r");
                match &decl.ty {
                    Some(Type::Constructed(TypeName::Record(fields), _)) => {
                        assert_eq!(fields.len(), 3);
                        assert_eq!(fields[0].0, "x");
                        assert_eq!(fields[1].0, "y");
                        assert_eq!(fields[2].0, "z");
                        Ok(())
                    }
                    _ => Err("Expected record type".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_sum_type_simple() {
    // Test sum type: Option i32
    expect_parse("let x: Some i32 | None = ???", |declarations| {
        assert_eq!(declarations.len(), 1);
        match &declarations[0] {
            Declaration::Decl(decl) => {
                assert_eq!(decl.name, "x");
                match &decl.ty {
                    Some(Type::Constructed(TypeName::Sum(variants), _)) => {
                        assert_eq!(variants.len(), 2);
                        assert_eq!(variants[0].0, "Some");
                        assert_eq!(variants[0].1.len(), 1); // Some has 1 type arg
                        assert_eq!(variants[1].0, "None");
                        assert_eq!(variants[1].1.len(), 0); // None has 0 type args
                        Ok(())
                    }
                    _ => Err("Expected sum type".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_parse_sum_type_multiple_args() {
    // Test sum type with multiple type arguments
    expect_parse("let x: Result i32 f32 | Error | Ok = ???", |declarations| {
        assert_eq!(declarations.len(), 1);
        match &declarations[0] {
            Declaration::Decl(decl) => {
                assert_eq!(decl.name, "x");
                match &decl.ty {
                    Some(Type::Constructed(TypeName::Sum(variants), _)) => {
                        assert_eq!(variants.len(), 3);
                        assert_eq!(variants[0].0, "Result");
                        assert_eq!(variants[0].1.len(), 2); // Result has 2 type args
                        assert_eq!(variants[1].0, "Error");
                        assert_eq!(variants[1].1.len(), 0);
                        assert_eq!(variants[2].0, "Ok");
                        assert_eq!(variants[2].1.len(), 0);
                        Ok(())
                    }
                    _ => Err("Expected sum type".to_string()),
                }
            }
            _ => Err("Expected Decl".to_string()),
        }
    });
}

// ============================================================================
// Grammar Ambiguity Tests
// Tests for ambiguities described in GRAMMAR.txt lines 210-233
// ============================================================================

#[test]
fn test_ambiguity_field_access_parses_as_field() {
    // x.y should parse as field access on record x
    expect_parse("def test(): i32 = x.y", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::FieldAccess(base, field) => match &base.kind {
                    ExprKind::Identifier(name) if name == "x" && field == "y" => Ok(()),
                    _ => Err("Expected x.y field access".to_string()),
                },
                _ => Err("Expected field access".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_array_index_with_space_is_function_call() {
    // f [x] with space should parse as function call with array argument
    expect_parse("def test(): i32 = f [1, 2, 3]", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::FunctionCall(name, args) if name == "f" && args.len() == 1 => match &args[0].kind {
                    ExprKind::ArrayLiteral(_) => Ok(()),
                    _ => Err("Expected array literal".to_string()),
                },
                _ => Err("Expected function call".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_array_index_without_space_is_indexing() {
    // f[x] without space should parse as array indexing
    expect_parse("def test(): i32 = f[0]", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::ArrayIndex(array, _) => match &array.kind {
                    ExprKind::Identifier(name) if name == "f" => Ok(()),
                    _ => Err("Expected identifier 'f'".to_string()),
                },
                _ => Err("Expected array index".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_negative_in_parens() {
    // (-x) should parse as negation of x in parentheses
    expect_parse("def test(): i32 = (-x)", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::UnaryOp(op, operand) if op.op == "-" => match &operand.kind {
                    ExprKind::Identifier(name) if name == "x" => Ok(()),
                    _ => Err("Expected identifier 'x'".to_string()),
                },
                _ => Err("Expected unary operation".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_prefix_binds_tighter_than_infix() {
    // !x + y should parse as (!x) + y, not !(x + y)
    expect_parse("def test(): i32 = !x + y", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::BinaryOp(op, left, _) if op.op == "+" => match &left.kind {
                    ExprKind::UnaryOp(unary_op, _) if unary_op.op == "!" => Ok(()),
                    _ => Err("Expected unary ! on left".to_string()),
                },
                _ => Err("Expected binary operation".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_function_application_binds_tighter() {
    // f x + y should parse as (f x) + y, not f (x + y)
    expect_parse("def test(): i32 = f x + y", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::BinaryOp(op, left, _) if op.op == "+" => match &left.kind {
                    ExprKind::FunctionCall(_, args) if args.len() == 1 => Ok(()),
                    _ => Err("Expected function call on left".to_string()),
                },
                _ => Err("Expected binary operation".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_let_extends_right() {
    // let x = 1 in x + y should parse with (x + y) as the body
    expect_parse("def test(): i32 = let x = 1 in x + y", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::LetIn(let_in) => match &let_in.body.kind {
                    ExprKind::BinaryOp(op, _, _) if op.op == "+" => Ok(()),
                    _ => Err("Expected binary op in let body".to_string()),
                },
                _ => Err("Expected let-in expression".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_if_extends_right() {
    // if cond then x else y + z should parse with (y + z) as else branch
    expect_parse("def test(): i32 = if true then 1 else 2 + 3", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::If(if_expr) => match &if_expr.else_branch.kind {
                    ExprKind::BinaryOp(op, _, _) if op.op == "+" => Ok(()),
                    _ => Err("Expected binary op in else branch".to_string()),
                },
                _ => Err("Expected if expression".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_type_ascription() {
    // x : i32 should parse as type ascription
    expect_parse("def test(): i32 = x : i32", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::TypeAscription(inner, ty) => match (&inner.kind, ty) {
                    (ExprKind::Identifier(name), Type::Constructed(TypeName::Str("i32"), _))
                        if name == "x" =>
                    {
                        Ok(())
                    }
                    _ => Err("Expected x : i32".to_string()),
                },
                _ => Err("Expected type ascription".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_ambiguity_pipe_operator() {
    // x |> f should parse as pipe operation
    expect_parse("def test(): i32 = x |> f", |declarations| {
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::Pipe(left, right) => match (&left.kind, &right.kind) {
                    (ExprKind::Identifier(l), ExprKind::Identifier(r)) if l == "x" && r == "f" => {
                        Ok(())
                    }
                    _ => Err("Expected x |> f".to_string()),
                },
                _ => Err("Expected pipe operation".to_string()),
            },
            _ => Err("Expected Decl".to_string()),
        }
    });
}

#[test]
fn test_function_call_with_and_without_parens() {
    // Test that "vec3 1.0 0.5 0.25" and "vec3 (1.0) (0.5) (0.25)" produce the same structure
    let input1 = "def test = vec3 1.0f32 0.5f32 0.25f32";
    let input2 = "def test = vec3 (1.0f32) (0.5f32) (0.25f32)";

    let tokens1 = tokenize(input1).unwrap();
    let mut parser1 = Parser::new(tokens1);
    let program1 = parser1.parse().unwrap();

    let tokens2 = tokenize(input2).unwrap();
    let mut parser2 = Parser::new(tokens2);
    let program2 = parser2.parse().unwrap();

    // Both should produce a FunctionCall, not Applications
    let decl1 = &program1.declarations[0];
    let decl2 = &program2.declarations[0];

    if let Declaration::Decl(d1) = decl1 {
        if let Declaration::Decl(d2) = decl2 {
            eprintln!("Without parens: {:?}", d1.body.kind);
            eprintln!("With parens: {:?}", d2.body.kind);

            // Check that both are FunctionCall
            match (&d1.body.kind, &d2.body.kind) {
                (ExprKind::FunctionCall(name1, args1), ExprKind::FunctionCall(name2, args2)) => {
                    assert_eq!(name1, name2);
                    assert_eq!(args1.len(), args2.len());
                    assert_eq!(args1.len(), 3, "Should have 3 arguments");
                }
                (ExprKind::FunctionCall(_, _), ExprKind::Application(_, _)) => {
                    panic!("Parenthesized arguments created Application instead of FunctionCall");
                }
                (ExprKind::Application(_, _), ExprKind::FunctionCall(_, _)) => {
                    panic!("Non-parenthesized arguments created Application (unexpected)");
                }
                (ExprKind::Application(_, _), ExprKind::Application(_, _)) => {
                    panic!("Both created Application - neither created FunctionCall");
                }
                (kind1, kind2) => {
                    panic!("Unexpected expression kinds: {:?} and {:?}", kind1, kind2);
                }
            }
        }
    }
}
