use super::*;
use crate::lexer::tokenize;

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
    let tokens_clone = tokens.clone();
    Parser::new(tokens).parse().unwrap_or_else(|e| {
        println!("Parse failed with error: {:?}", e);
        println!("Tokens were: {:#?}", tokens_clone);
        panic!("parse failed: {:?}", e);
    })
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

/// Parse input and return the single EntryDecl, panicking if not exactly one or not an Entry
fn single_entry(input: &str) -> EntryDecl {
    let program = parse_ok(input);
    assert_eq!(program.declarations.len(), 1, "expected exactly one declaration");
    match program.declarations.into_iter().next().unwrap() {
        Declaration::Entry(e) => e,
        other => panic!("expected Declaration::Entry, got {:?}", other),
    }
}

/// Assert that a Pattern has given name and type
macro_rules! assert_typed_param {
    ($param:expr, $name:expr, $ty:expr) => {
        assert_eq!(
            $param.simple_name(),
            Some($name),
            "Expected param with name {:?}, got {:?}",
            $name,
            $param
        );
        assert_eq!(
            $param.pattern_type(),
            Some(&$ty),
            "Expected param with type {:?}, got {:?}",
            $ty,
            $param.pattern_type()
        );
    };
}

/// Assert that a Pattern has given name, type, and attributes
macro_rules! assert_typed_param_with_attrs {
    ($param:expr, $name:expr, $ty:expr, $attrs:expr) => {
        assert_eq!(
            $param.simple_name(),
            Some($name),
            "Expected param with name {:?}, got {:?}",
            $name,
            $param
        );
        assert_eq!(
            $param.pattern_type(),
            Some(&$ty),
            "Expected param with type {:?}, got {:?}",
            $ty,
            $param.pattern_type()
        );
        // Pattern structure is Attributed(attrs, Typed(Name, type))
        // The attribute is on the outside, wrapping the typed pattern
        if let PatternKind::Attributed(attrs, inner) = &$param.kind {
            assert_eq!(
                attrs, &$attrs,
                "Expected attrs {:?}, got {:?}",
                $attrs, attrs
            );
            // Verify the inner pattern is Typed
            if !matches!(inner.kind, PatternKind::Typed(_, _)) {
                panic!(
                    "Expected typed pattern inside attributed pattern, got {:?}",
                    inner
                );
            }
        } else {
            panic!("Expected attributed pattern, got {:?}", $param);
        }
    };
}

/// Assert that a value matches a pattern, showing actual value on failure
macro_rules! assert_matches {
    ($expr:expr, $pattern:pat) => {
        match $expr {
            $pattern => {}
            ref actual => {
                panic!(
                    "assertion failed: `(left matches right)`\n  left: `{:#?}`\n right: `{}`",
                    actual,
                    stringify!($pattern)
                );
            }
        }
    };
    ($expr:expr, $pattern:pat if $cond:expr) => {
        match $expr {
            $pattern if $cond => {}
            ref actual => {
                panic!(
                    "assertion failed: `(left matches right)`\n  left: `{:#?}`\n right: `{} if {}`",
                    actual,
                    stringify!($pattern),
                    stringify!($cond)
                );
            }
        }
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
    let entry = single_entry("#[vertex] def main (x:i32) (y:f32) : [4]f32 = result");

    assert_eq!(entry.name, "main");
    assert_eq!(entry.entry_type, Attribute::Vertex);
    assert_eq!(entry.params.len(), 2);

    assert_typed_param!(&entry.params[0], "x", crate::ast::types::i32());
    assert_typed_param!(&entry.params[1], "y", crate::ast::types::f32());

    assert_eq!(entry.return_types.len(), 1);
    assert_eq!(
        entry.return_types[0],
        crate::ast::types::sized_array(4, crate::ast::types::f32())
    );
    // return_attributes removed - only on EntryDecl now
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
    let entry = single_entry("#[vertex] def main(): [4]f32 = result");
    assert_eq!(entry.entry_type, Attribute::Vertex);
    assert_eq!(entry.name, "main");
}

#[test]
fn test_parse_fragment_attribute() {
    let entry = single_entry("#[fragment] def frag(): [4]f32 = result");
    assert_eq!(entry.entry_type, Attribute::Fragment);
    assert_eq!(entry.name, "frag");
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
    let entry = single_entry("#[vertex] def main(): #[builtin(position)] [4]f32 = result");

    assert_eq!(entry.entry_type, Attribute::Vertex);
    assert_eq!(entry.return_types.len(), 1);
    assert_eq!(entry.return_attributes.len(), 1);
    assert_eq!(
        entry.return_attributes[0],
        Some(Attribute::BuiltIn(spirv::BuiltIn::Position))
    );
    assert_eq!(
        entry.return_types[0],
        crate::ast::types::sized_array(4, crate::ast::types::f32())
    );
}

#[test]
fn test_parse_single_attributed_return_type() {
    let entry = single_entry(
        "#[vertex] def vertex_main(): #[builtin(position)] vec4 = vec4 0.0f32 0.0f32 0.0f32 1.0f32",
    );

    assert_eq!(entry.entry_type, Attribute::Vertex);
    assert_eq!(entry.return_types.len(), 1);
    assert_eq!(entry.return_attributes.len(), 1);
    assert_eq!(
        entry.return_attributes[0],
        Some(Attribute::BuiltIn(spirv::BuiltIn::Position))
    );
    assert_matches!(
        &entry.return_types[0],
        Type::Constructed(TypeName::Named(name), _) if name == "vec4"
    );
}

#[test]
fn test_parse_tuple_attributed_return_type() {
    let entry = single_entry(
        "#[vertex] def vertex_main(): (#[builtin(position)] vec4, #[location(0)] vec3) = result",
    );

    assert_eq!(entry.return_types.len(), 2);
    assert_eq!(entry.return_attributes.len(), 2);

    // Check first element: [builtin(position)] vec4
    assert_eq!(
        entry.return_attributes[0],
        Some(Attribute::BuiltIn(spirv::BuiltIn::Position))
    );

    // Check second element: [location(0)] vec3
    assert_eq!(entry.return_attributes[1], Some(Attribute::Location(0)));
}

#[test]
fn test_parse_unattributed_return_type() {
    let decl = single_decl("def helper(): vec4 = vec4 1.0f32 0.0f32 0.0f32 1.0f32");

    // Regular decl no longer has attributed_return_type field

    // Should have a regular type
    let ty = decl.ty.as_ref().expect("Missing return type");
    assert_matches!(ty, Type::Constructed(TypeName::Named(name), _) if name == "vec4");
}

#[test]
fn test_parse_location_attribute_on_return_type() {
    let entry = single_entry("#[fragment] def frag(): #[location(0)] [4]f32 = result");

    assert_eq!(entry.entry_type, Attribute::Fragment);
    assert_eq!(entry.return_types.len(), 1);
    assert_eq!(entry.return_attributes.len(), 1);
    assert_eq!(entry.return_attributes[0], Some(Attribute::Location(0)));
    assert_eq!(
        entry.return_types[0],
        crate::ast::types::sized_array(4, crate::ast::types::f32())
    );
}

#[test]
fn test_parse_parameter_with_builtin_attribute() {
    let entry = single_entry("#[vertex] def main #[builtin(vertex_index)] (vid:i32) : [4]f32 = result");

    assert_eq!(entry.params.len(), 1);
    assert_typed_param_with_attrs!(
        &entry.params[0],
        "vid",
        crate::ast::types::i32(),
        vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)]
    );
}

#[test]
fn test_parse_parameter_with_location_attribute() {
    let entry = single_entry("#[fragment] def frag #[location(1)] (color:[3]f32) : [4]f32 = result");

    assert_eq!(entry.params.len(), 1);
    assert_typed_param_with_attrs!(
        &entry.params[0],
        "color",
        crate::ast::types::sized_array(3, crate::ast::types::f32()),
        vec![Attribute::Location(1)]
    );
}

#[test]
fn test_parse_multiple_builtin_types() {
    let entry = single_entry(
        "#[vertex] def main #[builtin(vertex_index)] (vid:i32) #[builtin(instance_index)] (iid:i32) : #[builtin(position)] [4]f32 = result",
    );

    assert_eq!(entry.params.len(), 2);

    // First parameter
    assert_typed_param_with_attrs!(
        &entry.params[0],
        "vid",
        crate::ast::types::i32(),
        vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)]
    );

    // Second parameter
    assert_typed_param_with_attrs!(
        &entry.params[1],
        "iid",
        crate::ast::types::i32(),
        vec![Attribute::BuiltIn(spirv::BuiltIn::InstanceIndex)]
    );

    // Return type
    assert_eq!(entry.return_types.len(), 1);
    assert_eq!(entry.return_attributes.len(), 1);
    assert_eq!(
        entry.return_attributes[0],
        Some(Attribute::BuiltIn(spirv::BuiltIn::Position))
    );
}

#[test]
fn test_parse_simple_lambda() {
    let decl = single_decl(r#"let f: i32 -> i32 = \x -> x"#);

    assert_eq!(decl.name, "f");
    assert!(matches!(
        &decl.body.kind,
        ExprKind::Lambda(lambda)
            if lambda.params.len() == 1
            && lambda.params[0].simple_name() == Some("x")
            && lambda.params[0].pattern_type().is_none()
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
            && lambda.params[0].simple_name() == Some("x")
            && lambda.params[0].pattern_type().is_none()
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
            && lambda.params[0].simple_name() == Some("x")
            && lambda.params[0].pattern_type().is_none()
            && lambda.params[1].simple_name() == Some("y")
            && lambda.params[1].pattern_type().is_none()
            && lambda.return_type.is_none()
            && matches!(lambda.body.kind, ExprKind::Identifier(ref name) if name == "x")
    ));
}

#[test]
fn test_parse_lambda_with_return_type() {
    // \x:i32 -> body means: untyped parameter x, return type i32
    let decl = single_decl(r#"let f: i32 -> i32 = \x:i32 -> x + 7i32"#);

    if let ExprKind::Lambda(lambda) = &decl.body.kind {
        // Check parameter: should be untyped
        assert_eq!(lambda.params.len(), 1);
        assert_eq!(lambda.params[0].simple_name(), Some("x"));
        assert!(
            lambda.params[0].pattern_type().is_none(),
            "Parameter should not have a type"
        );

        // Check return type: should be i32
        assert!(lambda.return_type.is_some(), "Lambda should have a return type");

        // Check body: should be x + 7i32
        assert!(matches!(lambda.body.kind, ExprKind::BinaryOp(_, _, _)));
        if let ExprKind::BinaryOp(_op, left, right) = &lambda.body.kind {
            assert!(matches!(left.kind, ExprKind::Identifier(ref name) if name == "x"));
            assert!(matches!(right.kind, ExprKind::IntLiteral(7)));
        }
    } else {
        panic!("Expected lambda, got {:?}", decl.body.kind);
    }
}

#[test]
fn test_parse_lambda_with_typed_parameter() {
    // \(x:i32) -> body means: typed parameter x:i32, no return type
    let decl = single_decl(r#"let f: i32 -> i32 = \(x:i32) -> x + 7i32"#);

    if let ExprKind::Lambda(lambda) = &decl.body.kind {
        // Check parameter: should be typed as i32
        assert_eq!(lambda.params.len(), 1);
        assert_eq!(lambda.params[0].simple_name(), Some("x"));
        assert!(
            lambda.params[0].pattern_type().is_some(),
            "Parameter should have a type"
        );

        // Check return type: should be None
        assert!(
            lambda.return_type.is_none(),
            "Lambda should not have a return type"
        );

        // Check body: should be x + 7i32
        assert!(matches!(lambda.body.kind, ExprKind::BinaryOp(_, _, _)));
        if let ExprKind::BinaryOp(_op, left, right) = &lambda.body.kind {
            assert!(matches!(left.kind, ExprKind::Identifier(ref name) if name == "x"));
            assert!(matches!(right.kind, ExprKind::IntLiteral(7)));
        }
    } else {
        panic!("Expected lambda, got {:?}", decl.body.kind);
    }
}

#[test]
fn test_parse_lambda_return_type_simple() {
    // Test just the lambda with return type parses correctly
    let decl = single_decl(r#"def f = \x:i32 -> x"#);

    if let ExprKind::Lambda(lambda) = &decl.body.kind {
        assert_eq!(lambda.params.len(), 1);
        assert_eq!(lambda.params[0].simple_name(), Some("x"));
        assert!(lambda.return_type.is_some(), "Lambda should have return type");
    } else {
        panic!("Expected lambda, got {:?}", decl.body.kind);
    }
}

#[test]
fn test_parse_lambda_in_parens() {
    // Test lambda in parentheses
    let decl = single_decl(r#"def f = (\x:i32 -> x)"#);

    if let ExprKind::Lambda(lambda) = &decl.body.kind {
        assert_eq!(lambda.params.len(), 1);
        assert!(lambda.return_type.is_some(), "Lambda should have return type");
    } else {
        panic!("Expected lambda, got {:?}", decl.body.kind);
    }
}

#[test]
fn test_parse_lambda_application_with_literal() {
    // Test function application: (lambda) arg with literal
    let decl = single_decl(r#"def apply = (\x:i32 -> x + 1i32) 5i32"#);

    // Should be a function application (eitherApplication)
    match &decl.body.kind {
        ExprKind::Application(func, args) => {
            assert!(!args.is_empty(), "Should have at least one argument");
        }
        _ => panic!("Expected function application, got {:?}", decl.body.kind),
    }
}

#[test]
fn test_parse_lambda_application_with_type_hole() {
    // Test function application: (lambda) arg
    let decl = single_decl(r#"def apply = (\x:i32 -> x + 1i32) ???"#);

    // Should be a function application (either Application)
    match &decl.body.kind {
        ExprKind::Application(_func, args) => {
            assert!(!args.is_empty(), "Should have at least one argument");
        }
        _ => panic!("Expected function application, got {:?}", decl.body.kind),
    }
}

#[test]
fn test_parse_simple_let_in() {
    // Just verify it parses successfully - the structure is complex to validate in detail
    let _entry = single_entry("#[vertex] def main (x:i32) : i32 = let y = 5 in y + x");
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
    let _entry = single_entry(r#"#[vertex] def main (x:i32) : i32 = let f = \y -> y + x in f 10"#);
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
    assert!(
        matches!(&program.declarations[0], Declaration::Decl(decl) if decl.keyword == "def" && decl.name == "verts")
    );

    // Second: vertex entry point
    assert!(
        matches!(&program.declarations[1], Declaration::Entry(entry) if entry.name == "vertex_main" && entry.entry_type == Attribute::Vertex)
    );

    // Third: def SKY_RGBA
    assert!(
        matches!(&program.declarations[2], Declaration::Decl(decl) if decl.keyword == "def" && decl.name == "SKY_RGBA")
    );

    // Fourth: fragment entry point
    assert!(
        matches!(&program.declarations[3], Declaration::Entry(entry) if entry.name == "fragment_main" && entry.entry_type == Attribute::Fragment)
    );
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
    assert!(
        matches!(&decl1.body.kind, ExprKind::Application(func, args) if matches!(&func.kind, ExprKind::Identifier(name) if name == "vec3") && args.len() == 3)
    );

    // Check second declaration: def x: f32 = v.x
    let decl2 = match &program.declarations[1] {
        Declaration::Decl(d) => d,
        _ => panic!("Expected second declaration to be Decl"),
    };
    assert_eq!(decl2.name, "x");
    assert!(
        matches!(&decl2.body.kind, ExprKind::FieldAccess(expr, field) if field == "x" && matches!(expr.kind, ExprKind::Identifier(ref name) if name == "v"))
    );
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
    assert_matches!(
        &uniform_decl.ty,
        Type::Constructed(TypeName::Named(name), args) if name == "vec3" && args.is_empty()
    );
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
    assert_matches!(
        &uniform_decl.ty,
        Type::Constructed(TypeName::Named(name), args) if name == "vec3" && args.is_empty()
    );
    // Check that there's no initializer (uniforms don't have bodies)
}

#[test]
fn test_uniform_with_initializer_error() {
    expect_parse_error(
        "#[uniform] def material_color: vec3 = vec3 1.0f32 0.5f32 0.2f32",
        |error| match error {
            CompilerError::ParseError(msg)
                if msg.contains("Uniform declarations cannot have initializer values") =>
            {
                Ok(())
            }
            _ => Err(format!(
                "Expected parse error about uniform initializer, got: {:?}",
                error
            )),
        },
    );
}

#[test]
fn test_parse_multiple_shader_outputs() {
    let entry = single_entry(
        r#"
            #[fragment] def fragment_main(): (#[location(0)] vec4, #[location(1)] vec3) =
              let color = vec4 1.0f32 0.5f32 0.2f32 1.0f32 in
              let normal = vec3 0.0f32 1.0f32 0.0f32 in
              (color, normal)
            "#,
    );

    assert_eq!(entry.name, "fragment_main");
    assert_eq!(entry.entry_type, Attribute::Fragment);

    assert_eq!(entry.return_types.len(), 2);
    assert_eq!(entry.return_attributes.len(), 2);

    // Check first output: [location(0)] vec4
    assert_eq!(entry.return_attributes[0], Some(Attribute::Location(0)));

    // Check second output: [location(1)] vec3
    assert_eq!(entry.return_attributes[1], Some(Attribute::Location(1)));
}

#[test]
fn test_parse_complete_shader_example() {
    let program = parse_ok(
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
    );

    assert_eq!(program.declarations.len(), 4);

    // Check uniform declarations
    assert!(matches!(&program.declarations[0], Declaration::Uniform(u) if u.name == "material_color"));
    assert!(matches!(&program.declarations[1], Declaration::Uniform(u) if u.name == "time"));

    // Check vertex shader with multiple outputs
    assert!(
        matches!(&program.declarations[2], Declaration::Entry(entry) if entry.name == "vertex_main" && entry.entry_type == Attribute::Vertex)
    );

    // Check fragment shader with multiple outputs
    assert!(
        matches!(&program.declarations[3], Declaration::Entry(entry) if entry.name == "fragment_main" && entry.entry_type == Attribute::Fragment)
    );
}

#[test]
fn test_if_then_else_parsing() {
    let decl = single_decl("def test: i32 = if x == 0 then 1 else 2");

    assert_eq!(decl.name, "test");

    // Check the if-then-else expression structure
    assert!(matches!(
        &decl.body.kind,
        ExprKind::If(if_expr)
            if matches!(&if_expr.condition.kind, ExprKind::BinaryOp(op, left, right)
                if op.op == "=="
                && matches!(left.kind, ExprKind::Identifier(ref name) if name == "x")
                && matches!(right.kind, ExprKind::IntLiteral(0))
            )
            && matches!(if_expr.then_branch.kind, ExprKind::IntLiteral(1))
            && matches!(if_expr.else_branch.kind, ExprKind::IntLiteral(2))
    ));
}
#[test]
fn test_parse_unit_pattern_simple() {
    // Test parsing () as a unit pattern parameter
    let decl = single_decl("def test () : i32 = 42");
    println!("params: {:?}", decl.params);
    assert_eq!(decl.params.len(), 1);
    println!("param[0] kind: {:?}", decl.params[0].kind);
    assert!(matches!(decl.params[0].kind, PatternKind::Unit));
}

#[test]
fn test_parse_attributed_return_simple() {
    let _ = env_logger::builder().is_test(true).try_init();
    // Test parsing a single attributed return type - must be an entry point
    let entry = single_entry(
        "#[vertex] def test () : #[builtin(position)] vec4 = vec4 0.0f32 0.0f32 0.0f32 1.0f32",
    );
    println!("entry: {:?}", entry);
    assert_eq!(entry.params.len(), 1);
    assert!(matches!(entry.params[0].kind, PatternKind::Unit));
    assert_eq!(entry.return_types.len(), 1);
    assert_eq!(entry.return_attributes.len(), 1);
}

#[test]
fn test_array_literal() {
    // Just verify it parses successfully
    let _entry = single_entry(
        "#[vertex] def test(): #[builtin(position)] [4]f32 = [0.0f32, 0.5f32, 0.0f32, 1.0f32]",
    );
}

#[test]
fn test_parse_array_type_directly() {
    let tokens = tokenize("[4]f32").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    parser.parse_type().expect("Failed to parse [4]f32");
}

#[test]
fn test_parse_array_literal() {
    let tokens = tokenize("[0.0f32, 0.5f32, 0.0f32, 1.0f32]").expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().expect("Failed to parse array literal");
    assert!(matches!(expr.kind, ExprKind::ArrayLiteral(_)));
}

#[test]
fn test_parse_attributed_return_type() {
    let entry = single_entry(
        r#"#[vertex]
def test_vertex : #[builtin(position)] vec4 =
  let angle = 1.0f32 in
  let s = f32.sin angle in
  vec4 s s 0.0f32 1.0f32"#,
    );

    assert_eq!(entry.name, "test_vertex");
    assert_eq!(entry.return_types.len(), 1);
    assert_eq!(entry.return_attributes.len(), 1);
}

#[test]
fn test_parse_unique_type() {
    let decl = single_decl("def foo (x:*i32) : i32 = x");

    assert_eq!(decl.params.len(), 1);
    let param_ty = decl.params[0].pattern_type().expect("Expected typed parameter");
    assert!(types::is_unique(param_ty));
    assert_eq!(types::strip_unique(param_ty), types::i32());
}

#[test]
fn test_parse_unique_array_type() {
    let decl = single_decl("def bar (arr:*[3]f32) : f32 = arr[0]");

    assert_eq!(decl.params.len(), 1);
    let param_ty = decl.params[0].pattern_type().expect("Expected typed parameter");
    assert!(types::is_unique(param_ty));
    assert_eq!(types::strip_unique(param_ty), types::sized_array(3, types::f32()));
}

#[test]
fn test_parse_nested_unique() {
    // Nested arrays with unique at different levels
    let decl = single_decl("def baz (x:*[2][3]i32) : i32 = x[0][0]");

    assert_eq!(decl.params.len(), 1);
    let param_ty = decl.params[0].pattern_type().expect("Expected typed parameter");
    assert!(types::is_unique(param_ty));
    assert_eq!(
        types::strip_unique(param_ty),
        types::sized_array(2, types::sized_array(3, types::i32()))
    );
}

#[test]
fn test_parse_function_application_with_array_literal() {
    let _ = env_logger::builder().is_test(true).try_init();
    let decl = single_decl("def test: vec4 = to_vec4 [1.0f32, 2.0f32, 3.0f32, 4.0f32]");

    assert_eq!(decl.name, "test");
    assert!(matches!(
        &decl.body.kind,
        ExprKind::Application(func, args)
            if matches!(&func.kind, ExprKind::Identifier(name) if name == "to_vec4")
            && args.len() == 1
            && matches!(&args[0].kind, ExprKind::ArrayLiteral(elements) if elements.len() == 4)
    ));
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

    assert!(matches!(&pattern.kind, PatternKind::Name(name) if name == "some"));
}

// Module parsing tests

#[test]
fn test_parse_type_bind_simple() {
    let program = parse_ok("type Point = (i32, i32)");
    assert_eq!(program.declarations.len(), 1);

    let bind = match &program.declarations[0] {
        Declaration::TypeBind(b) => b,
        _ => panic!("Expected TypeBind declaration"),
    };
    assert_eq!(bind.name, "Point");
    assert_eq!(bind.kind, TypeBindKind::Normal);
    assert_eq!(bind.type_params.len(), 0);
}

#[test]
fn test_parse_import() {
    let program = parse_ok("import \"path/to/module\"");
    assert_eq!(program.declarations.len(), 1);

    let path = match &program.declarations[0] {
        Declaration::Import(p) => p,
        _ => panic!("Expected Import declaration"),
    };
    assert_eq!(path, "path/to/module");
}

#[test]
fn test_parse_module_bind_simple() {
    let program = parse_ok("module M = { def x : i32 = 42 }");
    assert_eq!(program.declarations.len(), 1);

    let bind = match &program.declarations[0] {
        Declaration::ModuleBind(b) => b,
        _ => panic!("Expected ModuleBind declaration"),
    };
    assert_eq!(bind.name, "M");
    assert_eq!(bind.params.len(), 0);
    assert!(bind.signature.is_none());
    assert!(matches!(&bind.body, ModuleExpression::Struct(decls) if decls.len() == 1));
}

#[test]
fn test_parse_module_with_signature() {
    let program = parse_ok("module M : { val x : i32 } = { def x : i32 = 42 }");
    assert_eq!(program.declarations.len(), 1);

    let bind = match &program.declarations[0] {
        Declaration::ModuleBind(b) => b,
        _ => panic!("Expected ModuleBind declaration"),
    };
    assert_eq!(bind.name, "M");
    assert!(bind.signature.is_some());
    assert!(matches!(&bind.signature, Some(ModuleTypeExpression::Signature(specs)) if specs.len() == 1));
}

#[test]
fn test_parse_module_type_bind() {
    let program = parse_ok("module type numeric = { type t val add : t -> t -> t }");
    assert_eq!(program.declarations.len(), 1);

    match &program.declarations[0] {
        Declaration::ModuleTypeBind(mtb) => {
            assert_eq!(mtb.name, "numeric");
            assert!(matches!(&mtb.definition, ModuleTypeExpression::Signature(specs) if specs.len() == 2));
        }
        _ => panic!("Expected ModuleTypeBind declaration"),
    };
}

#[test]
fn test_parse_module_signature_with_abstract_type() {
    let program = parse_ok("module type S = { type t }");
    assert_eq!(program.declarations.len(), 1);

    match &program.declarations[0] {
        Declaration::ModuleTypeBind(mtb) => {
            assert_eq!(mtb.name, "S");
            match &mtb.definition {
                ModuleTypeExpression::Signature(specs) => {
                    assert_eq!(specs.len(), 1);
                    assert!(matches!(&specs[0], Spec::Type(_, name, _, None) if name == "t"));
                }
                _ => panic!("Expected Signature"),
            }
        }
        _ => panic!("Expected ModuleTypeBind declaration"),
    };
}

#[test]
fn test_parse_module_signature_with_concrete_type() {
    let program = parse_ok("module type S = { type t = i32 }");
    assert_eq!(program.declarations.len(), 1);

    match &program.declarations[0] {
        Declaration::ModuleTypeBind(mtb) => match &mtb.definition {
            ModuleTypeExpression::Signature(specs) => {
                assert_eq!(specs.len(), 1);
                assert!(matches!(&specs[0], Spec::Type(_, name, _, Some(_)) if name == "t"));
            }
            _ => panic!("Expected Signature"),
        },
        _ => panic!("Expected ModuleTypeBind declaration"),
    };
}

#[test]
fn test_parse_empty_module() {
    let program = parse_ok("module M = {}");
    assert_eq!(program.declarations.len(), 1);

    let bind = match &program.declarations[0] {
        Declaration::ModuleBind(b) => b,
        _ => panic!("Expected ModuleBind declaration"),
    };
    assert_eq!(bind.name, "M");
    assert!(matches!(&bind.body, ModuleExpression::Struct(decls) if decls.is_empty()));
}

#[test]
fn test_parse_module_multiple_declarations() {
    let program = parse_ok("module M = { type t = i32 def x : t = 42 def y : t = 99 }");
    assert_eq!(program.declarations.len(), 1);

    let bind = match &program.declarations[0] {
        Declaration::ModuleBind(b) => b,
        _ => panic!("Expected ModuleBind declaration"),
    };
    assert!(matches!(&bind.body, ModuleExpression::Struct(decls) if decls.len() == 3));
}

#[test]
fn test_parse_simple_field_access() {
    // First test that basic field access works
    let decl = single_decl("def x: f32 = f32.cos");

    match &decl.body.kind {
        ExprKind::FieldAccess(base, field) => {
            assert!(matches!(&base.kind, ExprKind::Identifier(name) if name == "f32"));
            assert_eq!(field, "cos");
        }
        _ => panic!("Expected FieldAccess, got {:?}", decl.body.kind),
    }
}

#[test]
fn test_parse_qualified_name() {
    // At parse time, f32.cos is parsed as field access
    // The type checker will resolve it to a module member
    let decl = single_decl("def x: f32 = f32.cos 0.5f32");

    match &decl.body.kind {
        ExprKind::Application(func, args) => {
            match &func.kind {
                ExprKind::FieldAccess(base, field) => {
                    assert!(matches!(&base.kind, ExprKind::Identifier(name) if name == "f32"));
                    assert_eq!(field, "cos");
                }
                _ => panic!("Expected FieldAccess, got {:?}", func.kind),
            }
            assert_eq!(args.len(), 1);
        }
        _ => panic!("Expected Application expression"),
    }
}

#[test]
fn test_parse_nested_qualified_name() {
    // At parse time, M.N.foo is parsed as nested field access
    let decl = single_decl("def x: i32 = M.N.foo 42");

    match &decl.body.kind {
        ExprKind::Application(func, args) => {
            match &func.kind {
                ExprKind::FieldAccess(base, field) => {
                    assert_eq!(field, "foo");
                    // base should be M.N
                    match &base.kind {
                        ExprKind::FieldAccess(inner_base, inner_field) => {
                            assert!(matches!(&inner_base.kind, ExprKind::Identifier(name) if name == "M"));
                            assert_eq!(inner_field, "N");
                        }
                        _ => panic!("Expected nested FieldAccess"),
                    }
                }
                _ => panic!("Expected FieldAccess"),
            }
            assert_eq!(args.len(), 1);
        }
        _ => panic!("Expected Application expression"),
    }
}

#[test]
fn test_parse_local_declaration() {
    let program = parse_ok("local def x : i32 = 42");
    assert_eq!(program.declarations.len(), 1);

    assert!(
        matches!(&program.declarations[0], Declaration::Local(inner) if matches!(**inner, Declaration::Decl(_)))
    );
}

#[test]
fn test_parse_record_type_empty() {
    // Test empty record type {}
    let decl = single_decl("let x: {} = ???");

    assert_eq!(decl.name, "x");
    assert!(matches!(&decl.ty, Some(Type::Constructed(TypeName::Record(fields), _)) if fields.is_empty()));
}

#[test]
fn test_parse_record_type_single_field() {
    // Test record type with single field {x: i32}
    let decl = single_decl("let r: {x: i32} = ???");

    assert_eq!(decl.name, "r");
    assert!(
        matches!(&decl.ty, Some(Type::Constructed(TypeName::Record(fields), _)) if fields.len() == 1 && fields.contains_key("x"))
    );
}

#[test]
fn test_parse_record_type_multiple_fields() {
    // Test record type with multiple fields {x: i32, y: f32, z: vec3}
    let decl = single_decl("let r: {x: i32, y: f32, z: vec3} = ???");

    assert_eq!(decl.name, "r");
    assert!(
        matches!(&decl.ty, Some(Type::Constructed(TypeName::Record(fields), _))
        if fields.len() == 3 && fields.contains_key("x") && fields.contains_key("y") && fields.contains_key("z"))
    );
}

#[test]
fn test_parse_sum_type_simple() {
    // Test sum type: Option i32
    let decl = single_decl("let x: Some i32 | None = ???");

    assert_eq!(decl.name, "x");
    assert!(
        matches!(&decl.ty, Some(Type::Constructed(TypeName::Sum(variants), _))
        if variants.len() == 2 && variants[0].0 == "Some" && variants[0].1.len() == 1
        && variants[1].0 == "None" && variants[1].1.len() == 0)
    );
}

#[test]
fn test_parse_sum_type_multiple_args() {
    // Test sum type with multiple type arguments
    let decl = single_decl("let x: Result i32 f32 | Error | Ok = ???");

    assert_eq!(decl.name, "x");
    assert!(
        matches!(&decl.ty, Some(Type::Constructed(TypeName::Sum(variants), _))
        if variants.len() == 3 && variants[0].0 == "Result" && variants[0].1.len() == 2
        && variants[1].0 == "Error" && variants[1].1.len() == 0
        && variants[2].0 == "Ok" && variants[2].1.len() == 0)
    );
}

// ============================================================================
// Grammar Ambiguity Tests
// Tests for ambiguities described in GRAMMAR.txt lines 210-233
// ============================================================================

#[test]
fn test_ambiguity_field_access_parses_as_field() {
    // x.y should parse as field access on record x
    let decl = single_decl("def test(): i32 = x.y");

    assert!(matches!(&decl.body.kind, ExprKind::FieldAccess(base, field)
        if matches!(&base.kind, ExprKind::Identifier(name) if name == "x") && field == "y"));
}

#[test]
fn test_ambiguity_array_index_with_space_is_function_call() {
    // f [x] with space should parse as function call with array argument
    let decl = single_decl("def test(): i32 = f [1, 2, 3]");

    assert!(matches!(&decl.body.kind, ExprKind::Application(func, args)
        if matches!(&func.kind, ExprKind::Identifier(name) if name == "f") && args.len() == 1 && matches!(&args[0].kind, ExprKind::ArrayLiteral(_))));
}

#[test]
fn test_ambiguity_array_index_without_space_is_indexing() {
    // f[x] without space should parse as array indexing
    let decl = single_decl("def test(): i32 = f[0]");

    assert!(matches!(&decl.body.kind, ExprKind::ArrayIndex(array, _)
        if matches!(&array.kind, ExprKind::Identifier(name) if name == "f")));
}

#[test]
fn test_ambiguity_negative_in_parens() {
    // (-x) should parse as negation of x in parentheses
    let decl = single_decl("def test(): i32 = (-x)");

    assert!(matches!(&decl.body.kind, ExprKind::UnaryOp(op, operand)
        if op.op == "-" && matches!(&operand.kind, ExprKind::Identifier(name) if name == "x")));
}

#[test]
fn test_ambiguity_prefix_binds_tighter_than_infix() {
    // !x + y should parse as (!x) + y, not !(x + y)
    let decl = single_decl("def test(): i32 = !x + y");

    assert!(matches!(&decl.body.kind, ExprKind::BinaryOp(op, left, _)
        if op.op == "+" && matches!(&left.kind, ExprKind::UnaryOp(unary_op, _) if unary_op.op == "!")));
}

#[test]
fn test_ambiguity_function_application_binds_tighter() {
    // f x + y should parse as (f x) + y, not f (x + y)
    let decl = single_decl("def test(): i32 = f x + y");

    assert!(matches!(&decl.body.kind, ExprKind::BinaryOp(op, left, _)
        if op.op == "+" && matches!(&left.kind, ExprKind::Application(_, args) if args.len() == 1)));
}

#[test]
fn test_ambiguity_let_extends_right() {
    // let x = 1 in x + y should parse with (x + y) as the body
    let decl = single_decl("def test(): i32 = let x = 1 in x + y");

    assert!(matches!(&decl.body.kind, ExprKind::LetIn(let_in)
        if matches!(&let_in.body.kind, ExprKind::BinaryOp(op, _, _) if op.op == "+")));
}

#[test]
fn test_ambiguity_if_extends_right() {
    // if cond then x else y + z should parse with (y + z) as else branch
    let decl = single_decl("def test(): i32 = if true then 1 else 2 + 3");

    assert!(matches!(&decl.body.kind, ExprKind::If(if_expr)
        if matches!(&if_expr.else_branch.kind, ExprKind::BinaryOp(op, _, _) if op.op == "+")));
}

#[test]
fn test_ambiguity_type_ascription() {
    // x : i32 should parse as type ascription
    let decl = single_decl("def test(): i32 = x : i32");

    assert!(matches!(&decl.body.kind, ExprKind::TypeAscription(inner, ty)
        if matches!(&inner.kind, ExprKind::Identifier(name) if name == "x")
        && matches!(ty, Type::Constructed(TypeName::Str("i32"), _))));
}

#[test]
fn test_ambiguity_pipe_operator() {
    // x |> f should parse as pipe operation
    let decl = single_decl("def test(): i32 = x |> f");

    assert!(matches!(&decl.body.kind, ExprKind::Pipe(left, right)
        if matches!(&left.kind, ExprKind::Identifier(l) if l == "x")
        && matches!(&right.kind, ExprKind::Identifier(r) if r == "f")));
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

    // Both should produce Applications with the same structure
    let decl1 = &program1.declarations[0];
    let decl2 = &program2.declarations[0];

    if let Declaration::Decl(d1) = decl1 {
        if let Declaration::Decl(d2) = decl2 {
            eprintln!("Without parens: {:?}", d1.body.kind);
            eprintln!("With parens: {:?}", d2.body.kind);

            // Check that both are Applications with the same structure
            match (&d1.body.kind, &d2.body.kind) {
                (ExprKind::Application(func1, args1), ExprKind::Application(func2, args2)) => {
                    // Both should call vec3 with 3 arguments
                    assert!(matches!(&func1.kind, ExprKind::Identifier(name) if name == "vec3"));
                    assert!(matches!(&func2.kind, ExprKind::Identifier(name) if name == "vec3"));
                    assert_eq!(args1.len(), 3);
                    assert_eq!(args2.len(), 3);
                }
                (kind1, kind2) => {
                    panic!("Expected both to be Application, got {:?} and {:?}", kind1, kind2);
                }
            }
        }
    }
}

#[test]
fn test_span_tracking() {
    // Test that spans are correctly tracked for a multi-line program
    let source = r#"def sum : i32 =
  let x = 10 + 20
  in x * 2

def main : i32 =
  sum"#;

    let tokens = tokenize(source).expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Failed to parse");

    assert_eq!(program.declarations.len(), 2);

    // Check first declaration (sum function) - spans line 1-3
    if let Declaration::Decl(sum_decl) = &program.declarations[0] {
        assert_eq!(sum_decl.name, "sum");
        // The declaration should span from line 1 to line 3
        assert!(
            sum_decl.body.h.span.start_line >= 1 && sum_decl.body.h.span.end_line <= 3,
            "sum body span should be lines 1-3, got {}..{}",
            sum_decl.body.h.span.start_line,
            sum_decl.body.h.span.end_line
        );

        // Check that the let-in expression has the right span
        if let ExprKind::LetIn(let_in) = &sum_decl.body.kind {
            // The let-in should start at "let" on line 2
            assert_eq!(
                let_in.value.h.span.start_line, 2,
                "let value should start on line 2"
            );

            // The binary op (a + b) should be on line 2
            if let ExprKind::BinaryOp(_, left, right) = &let_in.value.kind {
                assert_eq!(left.h.span.start_line, 2, "left operand should be on line 2");
                assert_eq!(right.h.span.start_line, 2, "right operand should be on line 2");
            }

            // The body (x * 2) should be on line 3
            assert_eq!(let_in.body.h.span.start_line, 3, "let body should be on line 3");
        } else {
            panic!("Expected LetIn expression, got {:?}", sum_decl.body.kind);
        }
    } else {
        panic!("Expected Decl, got {:?}", program.declarations[0]);
    }

    // Check second declaration (main function) - should be on line 5-6
    if let Declaration::Decl(main_decl) = &program.declarations[1] {
        assert_eq!(main_decl.name, "main");

        // The identifier should be on line 6
        if let ExprKind::Identifier(name) = &main_decl.body.kind {
            assert_eq!(name, "sum");
            assert_eq!(
                main_decl.body.h.span.start_line, 6,
                "identifier should be on line 6"
            );
        } else {
            panic!("Expected Identifier, got {:?}", main_decl.body.kind);
        }
    } else {
        panic!("Expected Decl, got {:?}", program.declarations[1]);
    }
}

#[test]
fn test_parse_pattern_x_i32() {
    let source = "x:i32";
    let tokens = tokenize(source).expect("Failed to tokenize");
    println!("Tokens for 'x:i32': {:#?}", tokens);
    let mut parser = Parser::new(tokens);
    match parser.parse_pattern() {
        Ok(pattern) => {
            println!("Successfully parsed pattern: {:#?}", pattern);
            assert_eq!(pattern.simple_name(), Some("x"));
            assert!(pattern.pattern_type().is_some());
        }
        Err(e) => panic!("Failed to parse pattern: {:?}", e),
    }
}

#[test]
fn test_parse_pattern_y_i32() {
    let source = "y:i32";
    let tokens = tokenize(source).expect("Failed to tokenize");
    println!("Tokens for 'y:i32': {:#?}", tokens);
    let mut parser = Parser::new(tokens);
    match parser.parse_pattern() {
        Ok(pattern) => {
            println!("Successfully parsed pattern: {:#?}", pattern);
            assert_eq!(pattern.simple_name(), Some("y"));
            assert!(pattern.pattern_type().is_some());
        }
        Err(e) => panic!("Failed to parse pattern: {:?}", e),
    }
}

#[test]
fn test_parse_lambda_with_tuple_pattern() {
    let source = r#"def test : i32 = let f = \(x,y) -> x in f (1, 2)"#;
    let tokens = tokenize(source).expect("Failed to tokenize");
    println!("Tokens: {:#?}", tokens);
    let mut parser = Parser::new(tokens);
    match parser.parse() {
        Ok(program) => {
            println!("Parsed: {:#?}", program);
            // Check that we have a lambda with a tuple pattern
            assert_eq!(program.declarations.len(), 1);
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
            panic!("Failed to parse: {:?}", e);
        }
    }
}

#[test]
fn test_parse_lambda_with_wildcard_in_tuple() {
    let source = r#"def test : i32 = let f = \(_,acc) -> acc in f (1, 2)"#;
    let tokens = tokenize(source).expect("Failed to tokenize");
    println!("Tokens: {:#?}", tokens);
    let mut parser = Parser::new(tokens);
    match parser.parse() {
        Ok(program) => {
            println!("Parsed: {:#?}", program);
            // Check that we have a lambda with a tuple pattern containing wildcard
            assert_eq!(program.declarations.len(), 1);
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
            panic!("Failed to parse: {:?}", e);
        }
    }
}

#[test]
fn test_parse_nested_let_in() {
    let source = r#"def test : i32 =
  let x = 1 in
  let y = 2 in
  let z = 3 in
  x + y + z"#;
    let decl = single_decl(source);
    assert_eq!(decl.name, "test");
}

#[test]
fn test_parse_loop_with_nested_lets() {
    let source = r#"def test : (i32, i32) =
  loop (idx, acc) = (0, 0) while idx < 10 do
    let x = idx * 2 in
    let y = x + 1 in
    (idx + 1, acc + y)"#;
    let decl = single_decl(source);
    assert_eq!(decl.name, "test");
}

#[test]
fn test_parse_function_with_typed_patterns() {
    let decl = single_decl("def add (x:i32) (y:i32) : i32 = x + y");
    assert_eq!(decl.name, "add");
    assert_eq!(decl.params.len(), 2);

    // Check first parameter: x:i32
    assert_eq!(decl.params[0].simple_name(), Some("x"));
    assert!(decl.params[0].pattern_type().is_some());

    // Check second parameter: y:i32
    assert_eq!(decl.params[1].simple_name(), Some("y"));
    assert!(decl.params[1].pattern_type().is_some());

    // Check return type
    assert!(decl.ty.is_some());
}

#[test]
fn test_parse_curried_function_call_with_paren_expr() {
    // Test parsing: myfunc a b (x + y)
    // This should parse as a function call with 3 arguments: a, b, and (x + y)
    let input = r#"
def test : f32 = myfunc arg1 arg2 (x + y)
"#;

    let decl = single_decl(input);

    // The body should be a function call
    match &decl.body.kind {
        ExprKind::Application(func, args) => {
            assert!(matches!(&func.kind, ExprKind::Identifier(name) if name == "myfunc"));
            assert_eq!(args.len(), 3, "Expected 3 arguments to myfunc");

            // First arg: arg1 (identifier)
            assert!(matches!(args[0].kind, ExprKind::Identifier(ref name) if name == "arg1"));

            // Second arg: arg2 (identifier)
            assert!(matches!(args[1].kind, ExprKind::Identifier(ref name) if name == "arg2"));

            // Third arg: (x + y) (binary operation)
            assert!(matches!(args[2].kind, ExprKind::BinaryOp(_, _, _)));
        }
        other => panic!("Expected Application, got {:?}", other),
    }
}

#[test]
fn test_parse_curried_vec3_call_with_paren_expr() {
    // Test the actual failing case from de_rasterizer
    let input = r#"
def mix3v (a:vec3f32) (b:vec3f32) (t:f32) : vec3f32 = a

def test (t:f32) : vec3f32 = mix3v a b (t*2.0f32 - 1.0f32)
"#;

    let program = parse_ok(input);
    assert_eq!(program.declarations.len(), 2);

    // Get the second declaration (test function)
    let test_decl = match &program.declarations[1] {
        Declaration::Decl(d) => d,
        _ => panic!("Expected Decl"),
    };

    // The body should be: mix3v a b (t*2.0f32 - 1.0f32)
    match &test_decl.body.kind {
        ExprKind::Application(func, args) => {
            assert!(matches!(&func.kind, ExprKind::Identifier(name) if name == "mix3v"));
            assert_eq!(args.len(), 3, "Expected 3 arguments to mix3v");

            // Third argument should be a binary op expression
            assert!(
                matches!(args[2].kind, ExprKind::BinaryOp(_, _, _)),
                "Third argument should be BinaryOp, got {:?}",
                args[2].kind
            );
        }
        other => panic!("Expected Application, got {:?}", other),
    }
}

#[test]
fn test_parse_map_with_lambda_and_array_index() {
    // Test parsing of: map (\e -> e[0]) edges
    let input = r#"
def test : [12]i32 =
  let edges : [12][2]i32 = [[0,1]] in
  map (\e -> e[0]) edges
"#;

    let program = parse_ok(input);
    assert_eq!(program.declarations.len(), 1);

    let decl = match &program.declarations[0] {
        Declaration::Decl(d) => d,
        _ => panic!("Expected Decl"),
    };

    // Body should be a let-in expression
    match &decl.body.kind {
        ExprKind::LetIn(let_in) => {
            // The body of the let should be: map (\e -> e[0]) edges
            match &let_in.body.kind {
                ExprKind::Application(func, args) => {
                    assert!(matches!(&func.kind, ExprKind::Identifier(name) if name == "map"));
                    assert_eq!(args.len(), 2, "map should have 2 arguments");

                    // First argument should be a lambda
                    match &args[0].kind {
                        ExprKind::Lambda(lambda) => {
                            assert_eq!(lambda.params.len(), 1, "Lambda should have 1 parameter");
                            assert_eq!(lambda.params[0].simple_name(), Some("e"));

                            // Lambda body should be array indexing: e[0]
                            match &lambda.body.kind {
                                ExprKind::ArrayIndex(arr, idx) => {
                                    // arr should be identifier 'e'
                                    match &arr.kind {
                                        ExprKind::Identifier(name) => assert_eq!(name, "e"),
                                        other => panic!("Expected Identifier, got {:?}", other),
                                    }
                                    // idx should be int literal 0
                                    match &idx.kind {
                                        ExprKind::IntLiteral(n) => assert_eq!(*n, 0),
                                        other => panic!("Expected IntLiteral, got {:?}", other),
                                    }
                                }
                                other => panic!("Expected ArrayIndex in lambda body, got {:?}", other),
                            }
                        }
                        other => panic!("Expected Lambda as first argument, got {:?}", other),
                    }

                    // Second argument should be identifier 'edges'
                    match &args[1].kind {
                        ExprKind::Identifier(name) => assert_eq!(name, "edges"),
                        other => panic!("Expected Identifier as second argument, got {:?}", other),
                    }
                }
                other => panic!("Expected Application in let body, got {:?}", other),
            }
        }
        other => panic!("Expected LetIn, got {:?}", other),
    }
}

#[test]
fn test_parse_loop_with_tuple_pattern_and_pipe() {
    // Test parsing of: loop (idx, acc) = (0, 10) while ... do ... |> f
    // Per SPECIFICATION.md line 559, loop body extends "as far to the right as possible",
    // so the pipe IS part of the loop body, not outside it. This test verifies correct parsing.
    let input = r#"
def test : i32 =
  loop (idx, acc) = (0, 10) while idx < 5 do
    (idx + 1, acc + idx)
  |> (\(_, result) -> result)
"#;

    let program = parse_ok(input);
    assert_eq!(program.declarations.len(), 1);

    let decl = match &program.declarations[0] {
        Declaration::Decl(d) => d,
        _ => panic!("Expected Decl"),
    };

    // Body should be a loop with pipe in its body (not pipe outside loop)
    // This is correct per SPECIFICATION.md line 559
    match &decl.body.kind {
        ExprKind::Loop(loop_expr) => {
            // Loop body should be a pipe expression
            match &loop_expr.body.kind {
                ExprKind::Pipe(left, right) => {
                    // Left side should be the tuple
                    match &left.kind {
                        ExprKind::Tuple(elements) => {
                            assert_eq!(elements.len(), 2);
                        }
                        other => panic!("Expected Tuple on left side of pipe, got {:?}", other),
                    }

                    // Right side should be the lambda
                    match &right.kind {
                        ExprKind::Lambda(lambda) => {
                            // Lambda should have tuple pattern parameter
                            assert_eq!(lambda.params.len(), 1);
                            match &lambda.params[0].kind {
                                PatternKind::Tuple(patterns) => {
                                    assert_eq!(patterns.len(), 2);
                                    // First should be wildcard
                                    match &patterns[0].kind {
                                        PatternKind::Wildcard => {}
                                        other => panic!("Expected Wildcard, got {:?}", other),
                                    }
                                    // Second should be name "result"
                                    match &patterns[1].kind {
                                        PatternKind::Name(name) => assert_eq!(name, "result"),
                                        other => panic!("Expected Name, got {:?}", other),
                                    }
                                }
                                other => panic!("Expected Tuple pattern, got {:?}", other),
                            }
                        }
                        other => panic!("Expected Lambda on right side of pipe, got {:?}", other),
                    }
                }
                other => panic!("Expected Pipe in loop body, got {:?}", other),
            }
        }
        other => panic!("Expected Loop, got {:?}", other),
    }
}

#[test]
fn test_let_tuple_pattern() {
    let input = "def test = let (x, y) = (1, 2) in x + y";
    let decl = single_decl(input);

    // Check that we have a LetIn expression with a tuple pattern
    match &decl.body.kind {
        ExprKind::LetIn(let_in) => {
            // Check that the pattern is a tuple pattern
            match &let_in.pattern.kind {
                PatternKind::Tuple(patterns) => {
                    assert_eq!(patterns.len(), 2);
                    match &patterns[0].kind {
                        PatternKind::Name(n) => assert_eq!(n, "x"),
                        other => panic!("Expected Name pattern, got {:?}", other),
                    }
                    match &patterns[1].kind {
                        PatternKind::Name(n) => assert_eq!(n, "y"),
                        other => panic!("Expected Name pattern, got {:?}", other),
                    }
                }
                other => panic!("Expected Tuple pattern, got {:?}", other),
            }
        }
        other => panic!("Expected LetIn, got {:?}", other),
    }
}

#[test]
fn test_parse_record_literal_empty() {
    let input = "def test = {}";
    let program = parse_ok(input);
    let decl = single_decl(input);

    match &decl.body.kind {
        ExprKind::RecordLiteral(fields) => {
            assert_eq!(fields.len(), 0);
        }
        other => panic!("Expected RecordLiteral, got {:?}", other),
    }
}

#[test]
fn test_parse_record_literal_single_field() {
    let input = "def test = {x: 42}";
    let decl = single_decl(input);

    match &decl.body.kind {
        ExprKind::RecordLiteral(fields) => {
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].0, "x");
            match &fields[0].1.kind {
                ExprKind::IntLiteral(n) => assert_eq!(*n, 42),
                other => panic!("Expected IntLiteral, got {:?}", other),
            }
        }
        other => panic!("Expected RecordLiteral, got {:?}", other),
    }
}

#[test]
fn test_parse_record_literal_multiple_fields() {
    let input = "def test = {__tag: 0, v3s: v3s}";
    let decl = single_decl(input);

    match &decl.body.kind {
        ExprKind::RecordLiteral(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "__tag");
            assert_eq!(fields[1].0, "v3s");
        }
        other => panic!("Expected RecordLiteral, got {:?}", other),
    }
}
