use crate::ast::{Declaration, ExprKind, Program, Type};
use crate::defunctionalization::Defunctionalizer;
use crate::lexer::tokenize;
use crate::parser::Parser;
use crate::type_checker::TypeVarGenerator;

/// Simple test implementation of TypeVarGenerator
struct TestTypeVarGen {
    next_id: usize,
}

impl TestTypeVarGen {
    fn new() -> Self {
        TestTypeVarGen { next_id: 0 }
    }
}

impl TypeVarGenerator for TestTypeVarGen {
    fn new_variable(&mut self) -> Type {
        let id = self.next_id;
        self.next_id += 1;
        Type::Variable(id)
    }
}

/// Helper to parse and defunctionalize a program
fn defunctionalize_program(input: &str) -> Program {
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Parsing failed");
    let node_counter = parser.take_node_counter();

    let type_var_gen = TestTypeVarGen::new();
    let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_var_gen);
    defunc.defunctionalize_program(&program).expect("Defunctionalization failed")
}

/// Helper to parse, defunctionalize, and type-check a program
/// Takes an assertion function that can inspect the defunctionalized program
/// If the assertion fails or type checking fails, prints the defunctionalized program
fn check_defunctionalized<F>(input: &str, assertion: F)
where
    F: FnOnce(&Program),
{
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Parsing failed");
    let node_counter = parser.take_node_counter();

    let type_context = polytype::Context::default();
    let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_context);
    let defunc_program = defunc.defunctionalize_program(&program).expect("Defunctionalization failed");

    // Run the assertion
    assertion(&defunc_program);

    // Now type-check the defunctionalized program
    let type_context = defunc.take_type_var_gen();

    // Enable debug logging
    std::env::set_var("RUST_LOG", "debug");
    env_logger::try_init().ok();

    let mut type_checker = crate::type_checker::TypeChecker::new_with_context(type_context);
    type_checker.load_builtins().expect("Loading builtins failed");

    eprintln!("\n=== STARTING TYPE CHECK ===");
    let result = type_checker.check_program(&defunc_program);

    if let Err(e) = result {
        eprintln!("\n=== TYPE CHECK ERROR ===");
        eprintln!("{:?}", e);
        eprintln!("\n=== DEFUNCTIONALIZED PROGRAM ===");
        eprintln!("{:#?}", defunc_program);
        panic!("Type checking failed: {:?}", e);
    }
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
    // 1. A generated __lam_* function that takes (__closure, y) parameters
    // 2. The original test function (no dispatcher with direct calls optimization)
    assert_eq!(
        defunc_program.declarations.len(),
        2,
        "Should have generated __lam_* and original test"
    );

    // First declaration should be the generated lambda function
    match &defunc_program.declarations[0] {
        Declaration::Decl(decl) => {
            assert!(
                decl.name.starts_with("__lam_"),
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
            Declaration::Decl(decl) if decl.name.starts_with("__lam_") => Some(decl),
            _ => None,
        })
        .expect("Should have generated lambda function");

    // Check that the lambda body contains FieldAccess to __closure
    fn contains_closure_field_access(expr: &crate::ast::Expression) -> bool {
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
                contains_closure_field_access(&let_in.value) || contains_closure_field_access(&let_in.body)
            }
            ExprKind::BinaryOp(_op, left, right) => {
                contains_closure_field_access(left) || contains_closure_field_access(right)
            }
            ExprKind::ArrayLiteral(elements) => elements.iter().any(|e| contains_closure_field_access(e)),
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

    // Should have generated lambda plus original (no dispatcher with direct calls optimization)
    assert_eq!(defunc_program.declarations.len(), 2);

    // Generated function should have __closure parameter (even if empty) and x
    match &defunc_program.declarations[0] {
        Declaration::Decl(decl) => {
            assert!(decl.name.starts_with("__lam_"));
            // Should have 2 params: __closure (even if empty) and x
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

    // Should have generated lambda plus original (no dispatcher with direct calls optimization)
    assert_eq!(defunc_program.declarations.len(), 2);

    // Generated function should have __closure parameter with record type containing a, b, c
    match &defunc_program.declarations[0] {
        Declaration::Decl(decl) => {
            assert!(decl.name.starts_with("__lam_"));
            assert_eq!(decl.params.len(), 2, "Should have __closure and x parameters");

            // Check that body has references to __closure.a, __closure.b, __closure.c
            fn count_closure_accesses(expr: &crate::ast::Expression) -> usize {
                match &expr.kind {
                    ExprKind::FieldAccess(base, _field) => {
                        let base_count = if let ExprKind::Identifier(name) = &base.kind {
                            if name == "__closure" { 1 } else { 0 }
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_defunctionalize_simple_lambda() {
        let input = r#"let f: i32 -> i32 = \x -> x"#;
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        let node_counter = parser.take_node_counter();

        let type_var_gen = TestTypeVarGen::new();
        let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_var_gen);
        let result = defunc.defunctionalize_program(&program).unwrap();

        // The let declaration should be transformed
        // Should have at least the original declaration plus any generated functions
        assert!(result.declarations.len() >= 1);
    }

    #[test]
    fn test_defunctionalize_nested_application() {
        // Test that ((f x) y) z becomes f(x, y, z)
        let input = "def test = vec3 1.0f32 0.5f32 0.25f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        let node_counter = parser.take_node_counter();

        let type_var_gen = TestTypeVarGen::new();
        let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_var_gen);
        let result = defunc.defunctionalize_program(&program).unwrap();

        // Check that the result doesn't contain any Application nodes
        let decl = &result.declarations[0];
        if let Declaration::Decl(d) = decl {
            // The body should be a FunctionCall, not an Application
            match &d.body.kind {
                ExprKind::FunctionCall(name, args) => {
                    assert_eq!(name, "vec3");
                    assert_eq!(args.len(), 3);
                }
                ExprKind::Application(_, _) => {
                    panic!(
                        "Found Application node after defunctionalization - nested applications not flattened"
                    );
                }
                other => panic!("Expected FunctionCall, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_defunctionalize_qualified_name() {
        // Test that qualified names like f32.sqrt are defunctionalized correctly
        let input = "def test = f32.sqrt 4.0f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        // Debug: print the AST before defunctionalization
        if let Declaration::Decl(d) = &program.declarations[0] {
            eprintln!("Before defunctionalization: {:?}", d.body.kind);
        }

        let node_counter = parser.take_node_counter();
        let type_var_gen = TestTypeVarGen::new();
        let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_var_gen);
        let result = defunc.defunctionalize_program(&program).unwrap();

        // Debug: print the AST after defunctionalization
        if let Declaration::Decl(d) = &result.declarations[0] {
            eprintln!("After defunctionalization: {:?}", d.body.kind);
        }

        // Check that the result is a FunctionCall with dotted name
        let decl = &result.declarations[0];
        if let Declaration::Decl(d) = decl {
            match &d.body.kind {
                ExprKind::FunctionCall(name, args) => {
                    assert_eq!(name, "f32.sqrt");
                    assert_eq!(args.len(), 1);
                }
                ExprKind::Application(_, _) => {
                    panic!("Found Application node after defunctionalization");
                }
                other => panic!("Expected FunctionCall, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_defunctionalize_application_with_division() {
        // Test that constant-folded divisions inside function calls work
        let input = "def test = vec3 (255.0f32/255.0f32) 0.5f32 0.25f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        // Debug: print the AST before constant folding
        if let Declaration::Decl(d) = &program.declarations[0] {
            eprintln!("Before constant folding: {:?}", d.body.kind);
        }

        // Run constant folding first
        let mut folder = crate::constant_folding::ConstantFolder::new();
        let program = folder.fold_program(&program).unwrap();

        // Debug: print the AST after constant folding
        if let Declaration::Decl(d) = &program.declarations[0] {
            eprintln!("After constant folding: {:?}", d.body.kind);
        }

        // Then defunctionalize
        let node_counter = folder.take_node_counter();
        let type_var_gen = TestTypeVarGen::new();
        let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_var_gen);
        let result = defunc.defunctionalize_program(&program).unwrap();

        // Check that the result doesn't contain any Application nodes
        let decl = &result.declarations[0];
        if let Declaration::Decl(d) = decl {
            match &d.body.kind {
                ExprKind::FunctionCall(name, args) => {
                    assert_eq!(name, "vec3");
                    assert_eq!(args.len(), 3);
                    // First arg should be the constant-folded result (1.0)
                    match &args[0].kind {
                        ExprKind::FloatLiteral(v) => assert_eq!(*v, 1.0),
                        other => panic!("Expected FloatLiteral after constant folding, got {:?}", other),
                    }
                }
                ExprKind::Application(_, _) => {
                    panic!("Found Application node after defunctionalization");
                }
                other => panic!("Expected FunctionCall, got {:?}", other),
            }
        }
    }
}
#[test]
fn test_direct_closure_application_typechecks() {
    let input = r#"
        def test : i32 =
            let x = 5 in
            let f = \y -> x + y in
            f 10
    "#;

    // Parse
    let tokens = crate::lexer::tokenize(input).expect("Tokenization failed");
    let mut parser = crate::parser::Parser::new(tokens);
    let program = parser.parse().expect("Parsing failed");
    let node_counter = parser.take_node_counter();

    // Print original program
    eprintln!("\n=== ORIGINAL PROGRAM ===");
    for (i, decl) in program.declarations.iter().enumerate() {
        match decl {
            crate::ast::Declaration::Decl(d) => {
                eprintln!("Decl {}: {}", i, d.name);
                if d.name == "test" {
                    eprintln!("  Original test body: {:#?}", d.body.kind);
                }
            }
            _ => {}
        }
    }

    // Defunctionalize
    let type_context = polytype::Context::default();
    let mut defunc =
        crate::defunctionalization::Defunctionalizer::new_with_counter(node_counter, type_context);
    let defunc_program = defunc.defunctionalize_program(&program).expect("Defunctionalization failed");
    let type_context = defunc.take_type_var_gen();

    // Print the defunctionalized program structure
    eprintln!("\n=== DEFUNCTIONALIZED PROGRAM ===");
    for (i, decl) in defunc_program.declarations.iter().enumerate() {
        match decl {
            crate::ast::Declaration::Decl(d) => {
                eprintln!("Decl {}: {} with {} params", i, d.name, d.params.len());
                eprintln!("  Body kind: {:?}", std::mem::discriminant(&d.body.kind));

                // For the test declaration, print the actual expression structure
                if d.name == "test" {
                    eprintln!("  Test body details: {:#?}", d.body.kind);
                }
            }
            _ => eprintln!("Decl {}: Other", i),
        }
    }

    // Type check
    let mut type_checker = crate::type_checker::TypeChecker::new_with_context(type_context);
    type_checker.load_builtins().expect("Loading builtins failed");
    let result = type_checker.check_program(&defunc_program);

    if let Err(e) = &result {
        eprintln!("\n=== TYPE CHECK ERROR ===");
        eprintln!("{:?}", e);
    }

    result.expect("Type checking should succeed");
}

#[test]
fn test_map_with_closure_typechecks() {
    let input = r#"
        def test : [3]i32 =
            let x = 5 in
            map (\y -> x + y) [1, 2, 3]
    "#;

    // Parse
    let tokens = crate::lexer::tokenize(input).expect("Tokenization failed");
    let mut parser = crate::parser::Parser::new(tokens);
    let program = parser.parse().expect("Parsing failed");
    let node_counter = parser.take_node_counter();

    // Defunctionalize
    let type_context = polytype::Context::default();
    let mut defunc =
        crate::defunctionalization::Defunctionalizer::new_with_counter(node_counter, type_context);
    let defunc_program = defunc.defunctionalize_program(&program).expect("Defunctionalization failed");
    let type_context = defunc.take_type_var_gen();

    // Print the defunctionalized program structure
    eprintln!("\n=== MAP CLOSURE TEST - DEFUNCTIONALIZED PROGRAM ===");
    for (i, decl) in defunc_program.declarations.iter().enumerate() {
        match decl {
            crate::ast::Declaration::Decl(d) => {
                eprintln!("Decl {}: {} with {} params", i, d.name, d.params.len());

                // Print details of the test function body
                if d.name == "test" {
                    eprintln!("  Test body structure:");
                    print_expr_structure(&d.body, 2);
                }
            }
            _ => eprintln!("Decl {}: Other", i),
        }
    }

    // Type check
    let mut type_checker = crate::type_checker::TypeChecker::new_with_context(type_context);
    type_checker.load_builtins().expect("Loading builtins failed");
    let result = type_checker.check_program(&defunc_program);

    if let Err(e) = &result {
        eprintln!("\n=== TYPE CHECK ERROR ===");
        eprintln!("{:?}", e);
    }

    result.expect("Type checking should succeed");
}

// Helper function to print expression structure
fn print_expr_structure(expr: &crate::ast::Expression, indent: usize) {
    let prefix = " ".repeat(indent);
    match &expr.kind {
        crate::ast::ExprKind::LetIn(let_in) => {
            if let crate::ast::PatternKind::Name(n) = &let_in.pattern.kind {
                eprintln!("{}let {} = ...", prefix, n);
            }
            eprintln!("{}  value:", prefix);
            print_expr_structure(&let_in.value, indent + 4);
            eprintln!("{}  body:", prefix);
            print_expr_structure(&let_in.body, indent + 4);
        }
        crate::ast::ExprKind::FunctionCall(name, args) => {
            eprintln!("{}FunctionCall({}, {} args)", prefix, name, args.len());
            for (i, arg) in args.iter().enumerate() {
                eprintln!("{}  arg {}:", prefix, i);
                print_expr_structure(arg, indent + 4);
            }
        }
        crate::ast::ExprKind::Loop(loop_expr) => {
            eprintln!("{}Loop", prefix);
            eprintln!("{}  body:", prefix);
            print_expr_structure(&loop_expr.body, indent + 4);
        }
        crate::ast::ExprKind::Identifier(name) => {
            eprintln!("{}Identifier({})", prefix, name);
        }
        crate::ast::ExprKind::RecordLiteral(fields) => {
            eprintln!("{}RecordLiteral with {} fields", prefix, fields.len());
            for (name, _) in fields {
                eprintln!("{}  field: {}", prefix, name);
            }
        }
        crate::ast::ExprKind::IntLiteral(n) => {
            eprintln!("{}IntLiteral({})", prefix, n);
        }
        crate::ast::ExprKind::ArrayLiteral(elements) => {
            eprintln!("{}ArrayLiteral({} elements)", prefix, elements.len());
        }
        _ => {
            eprintln!("{}Other({:?})", prefix, std::mem::discriminant(&expr.kind));
        }
    }
}

#[test]
fn test_two_maps_second_captures_first_result() {
    // Minimal failing case: two maps where second map's lambda captures result of first map
    // This tests that type variables don't bleed between the two __apply1 calls
    let input = r#"
def test : f32 =
    let v4s : [2]vec4f32 = [vec4 1.0f32 2.0f32 3.0f32 4.0f32, vec4 5.0f32 6.0f32 7.0f32 8.0f32] in
    let v3s : [2]vec3f32 = map (\(q:vec4f32) -> vec3 q.x q.y q.z) v4s in
    let indices : [2]i32 = [0, 1] in
    let results : [2]f32 = map (\i -> v3s[i].x) indices in
    results[0]
    "#;

    check_defunctionalized(input, |_program| {
        // Just checking that defunctionalization and type-checking succeed
    });
}

#[test]
fn test_desugar_loop_to_internal_structure() {
    let input = r#"
def test : i32 =
    loop (idx, acc) = (0, 100) while idx < 10 do
        (idx + 1, acc + idx)
"#;

    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Parsing failed");
    let node_counter = parser.take_node_counter();

    // Run constant folding first
    let mut folder = crate::constant_folding::ConstantFolder::new();
    let folded_program = folder.fold_program(&program).expect("Constant folding failed");

    let type_context = polytype::Context::default();
    let mut defunc = Defunctionalizer::new_with_counter(node_counter, type_context);
    let defunc_program =
        defunc.defunctionalize_program(&folded_program).expect("Defunctionalization failed");

    // Find the function body
    let decl = &defunc_program.declarations[0];
    let body = match decl {
        Declaration::Decl(func) => &func.body,
        _ => panic!("Expected function declaration"),
    };

    // The body should be a LetIn wrapping an InternalLoop
    let internal_loop = match &body.kind {
        ExprKind::LetIn(let_in) => match &let_in.body.kind {
            ExprKind::InternalLoop(il) => il,
            _ => panic!("Expected InternalLoop inside LetIn, got {:?}", let_in.body.kind),
        },
        _ => panic!("Expected LetIn wrapping InternalLoop, got {:?}", body.kind),
    };

    // Check phi_vars count - should be 2 (one for idx, one for acc)
    assert_eq!(internal_loop.phi_vars.len(), 2, "Expected 2 phi_vars");

    // Check loop var names
    assert_eq!(internal_loop.phi_vars[0].loop_var_name, "idx");
    assert_eq!(internal_loop.phi_vars[1].loop_var_name, "acc");

    // Check that condition exists
    assert!(internal_loop.condition.is_some());

    eprintln!("phi_vars:");
    for (i, phi_var) in internal_loop.phi_vars.iter().enumerate() {
        eprintln!(
            "  {}: init_name={}, loop_var_name={}",
            i, phi_var.init_name, phi_var.loop_var_name
        );
    }
}
