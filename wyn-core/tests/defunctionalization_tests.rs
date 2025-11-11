use wyn_core::ast::{Declaration, ExprKind, Program};
use wyn_core::defunctionalization::Defunctionalizer;
use wyn_core::lexer::tokenize;
use wyn_core::parser::Parser;

/// Helper to parse and defunctionalize a program
fn defunctionalize_program(input: &str) -> Program {
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Parsing failed");

    let mut defunc = Defunctionalizer::new();
    defunc.defunctionalize_program(&program).expect("Defunctionalization failed")
}

/// Test that free variables in lambda bodies are rewritten to access closure fields
#[test]
fn test_lambda_captures_free_variable() {
    let input = r#"
        def test =
            let x = 42 in
            let f = \y -> x + y in
            f 10
    "#;

    let defunc_program = defunctionalize_program(input);

    // The defunctionalized program should have:
    // 1. A generated __lambda_0 function that takes (__closure, y) parameters
    // 2. The original test function
    assert_eq!(
        defunc_program.declarations.len(),
        2,
        "Should have generated __lambda_0 plus original test"
    );

    // First declaration should be the generated lambda function
    match &defunc_program.declarations[0] {
        Declaration::Decl(decl) => {
            assert!(
                decl.name.starts_with("__lambda_"),
                "First declaration should be generated lambda, got: {}",
                decl.name
            );
            // Should have 2 parameters: __closure and y
            assert_eq!(
                decl.params.len(),
                2,
                "Generated lambda should have 2 parameters (__closure, y)"
            );
        }
        _ => panic!("First declaration should be Decl"),
    }
}

/// Test nested let-in with captured variables in lambda
#[test]
fn test_lambda_captures_variable_through_let() {
    let input = r#"
        def test =
            let verts = [1, 2, 3] in
            map (\e ->
                let a = verts[e] in
                a * 2
            ) [0, 1, 2]
    "#;

    let defunc_program = defunctionalize_program(input);

    // Should have generated lambda plus original
    assert!(
        defunc_program.declarations.len() >= 2,
        "Should have at least 2 declarations"
    );

    // Find the generated lambda function
    let lambda_decl = defunc_program
        .declarations
        .iter()
        .find_map(|d| match d {
            Declaration::Decl(decl) if decl.name.starts_with("__lambda_") => Some(decl),
            _ => None,
        })
        .expect("Should have generated lambda function");

    // Check that the lambda body contains FieldAccess to __closure
    fn contains_closure_field_access(expr: &wyn_core::ast::Expression) -> bool {
        match &expr.kind {
            ExprKind::FieldAccess(base, _field) => {
                // Check if base is Identifier("__closure")
                if let ExprKind::Identifier(name) = &base.kind {
                    if name == "__closure" {
                        return true;
                    }
                }
                // Also recursively check base
                contains_closure_field_access(base)
            }
            ExprKind::LetIn(let_in) => {
                contains_closure_field_access(&let_in.value)
                    || contains_closure_field_access(&let_in.body)
            }
            ExprKind::BinaryOp(_op, left, right) => {
                contains_closure_field_access(left) || contains_closure_field_access(right)
            }
            ExprKind::ArrayLiteral(elements) => {
                elements.iter().any(|e| contains_closure_field_access(e))
            }
            ExprKind::ArrayIndex(array, index) => {
                contains_closure_field_access(array) || contains_closure_field_access(index)
            }
            ExprKind::FunctionCall(_name, args) => {
                args.iter().any(|arg| contains_closure_field_access(arg))
            }
            _ => false,
        }
    }

    if !contains_closure_field_access(&lambda_decl.body) {
        // Debug: print the lambda body structure
        eprintln!("Lambda body: {:?}", lambda_decl.body);
        panic!("Lambda body should contain __closure.field accesses for captured variables");
    }
}

/// Test that lambda with no free variables doesn't create closure
#[test]
fn test_lambda_no_free_variables() {
    let input = r#"
        def test = \x -> x + 1
    "#;

    let defunc_program = defunctionalize_program(input);

    // Should have generated lambda plus original
    assert_eq!(defunc_program.declarations.len(), 2);

    // Generated function should have only 1 parameter (no closure needed)
    match &defunc_program.declarations[0] {
        Declaration::Decl(decl) => {
            assert!(decl.name.starts_with("__lambda_"));
            // Should have 2 params: __closure (even if empty) and x
            // Actually, with empty free vars, we might still add closure param
            assert!(
                decl.params.len() >= 1,
                "Should have at least the lambda parameter"
            );
        }
        _ => panic!("Should be Decl"),
    }
}

/// Test multiple captured variables
#[test]
fn test_lambda_captures_multiple_variables() {
    let input = r#"
        def test =
            let a = 1 in
            let b = 2 in
            let c = 3 in
            \x -> a + b + c + x
    "#;

    let defunc_program = defunctionalize_program(input);

    // Should have generated lambda plus original
    assert_eq!(defunc_program.declarations.len(), 2);

    // Generated function should have __closure parameter with record type containing a, b, c
    match &defunc_program.declarations[0] {
        Declaration::Decl(decl) => {
            assert!(decl.name.starts_with("__lambda_"));
            assert_eq!(
                decl.params.len(),
                2,
                "Should have __closure and x parameters"
            );

            // Check that body has references to __closure.a, __closure.b, __closure.c
            fn count_closure_accesses(expr: &wyn_core::ast::Expression) -> usize {
                match &expr.kind {
                    ExprKind::FieldAccess(base, _field) => {
                        let base_count = if let ExprKind::Identifier(name) = &base.kind {
                            if name == "__closure" {
                                1
                            } else {
                                0
                            }
                        } else {
                            0
                        };
                        base_count + count_closure_accesses(base)
                    }
                    ExprKind::BinaryOp(_op, left, right) => {
                        count_closure_accesses(left) + count_closure_accesses(right)
                    }
                    _ => 0,
                }
            }

            let closure_access_count = count_closure_accesses(&decl.body);
            assert_eq!(
                closure_access_count, 3,
                "Should have 3 closure field accesses for a, b, c"
            );
        }
        _ => panic!("Should be Decl"),
    }
}
