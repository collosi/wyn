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

#[test]
fn test_parse_let_decl() {
    expect_parse("let x: i32 = 42", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.name != "x" {
                    return Err(format!("Expected name 'x', got '{}'", decl.name));
                }
                if decl.ty != Some(crate::ast::types::i32()) {
                    return Err(format!("Expected i32 type, got {:?}", decl.ty));
                }
                if decl.body.kind != ExprKind::IntLiteral(42) {
                    return Err(format!("Expected IntLiteral(42), got {:?}", decl.body));
                }
                Ok(())
            }
            _ => Err("Expected Let declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_array_type() {
    expect_parse(
        "let arr: [3][4]f32 = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.name != "arr" {
                        return Err(format!("Expected name 'arr', got '{}'", decl.name));
                    }
                    if decl.ty.is_none() {
                        return Err("Expected array type to be parsed".to_string());
                    }
                    Ok(())
                }
                _ => Err("Expected Let declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_entry_point_decl() {
    expect_parse(
        "#[vertex] def main(x: i32, y: f32): [4]f32 = result",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.name != "main" {
                        return Err(format!("Expected name 'main', got '{}'", decl.name));
                    }
                    if decl.attributes != vec![Attribute::Vertex] {
                        return Err(format!("Expected Vertex attribute, got {:?}", decl.attributes));
                    }
                    if decl.params.len() != 2 {
                        return Err(format!("Expected 2 parameters, got {}", decl.params.len()));
                    }
                    match &decl.params[0] {
                        DeclParam::Typed(param) => {
                            if param.name != "x" {
                                return Err(format!("Expected first param name 'x', got '{}'", param.name));
                            }
                            if param.ty != crate::ast::types::i32() {
                                return Err(format!(
                                    "Expected i32 type for first param, got {:?}",
                                    param.ty
                                ));
                            }
                            if !param.attributes.is_empty() {
                                return Err(format!(
                                    "Expected no attributes for first param, got {:?}",
                                    param.attributes
                                ));
                            }
                        }
                        _ => return Err("Expected typed parameter".to_string()),
                    }
                    match &decl.params[1] {
                        DeclParam::Typed(param) => {
                            if param.name != "y" {
                                return Err(format!(
                                    "Expected second param name 'y', got '{}'",
                                    param.name
                                ));
                            }
                            if param.ty != crate::ast::types::f32() {
                                return Err(format!(
                                    "Expected f32 type for second param, got {:?}",
                                    param.ty
                                ));
                            }
                            if !param.attributes.is_empty() {
                                return Err(format!(
                                    "Expected no attributes for second param, got {:?}",
                                    param.attributes
                                ));
                            }
                        }
                        _ => return Err("Expected typed parameter".to_string()),
                    }
                    if let Some(ref ty) = decl.ty {
                        if *ty != crate::ast::types::sized_array(4, crate::ast::types::f32()) {
                            return Err(format!("Expected [4]f32 return type, got {:?}", ty));
                        }
                    } else {
                        return Err("Expected return type".to_string());
                    }
                    if !decl.return_attributes.is_empty() {
                        return Err(format!(
                            "Expected no attributes on return type, got {:?}",
                            decl.return_attributes
                        ));
                    }
                    Ok(())
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_array_index() {
    expect_parse("let x: f32 = arr[0]", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::ArrayIndex(arr, idx) => {
                    if arr.kind != ExprKind::Identifier("arr".to_string()) {
                        return Err(format!("Expected array identifier 'arr', got {:?}", **arr));
                    }
                    if idx.kind != ExprKind::IntLiteral(0) {
                        return Err(format!("Expected index 0, got {:?}", **idx));
                    }
                    Ok(())
                }
                _ => Err(format!("Expected ArrayIndex expression, got {:?}", decl.body)),
            },
            _ => Err("Expected Let declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_division() {
    expect_parse("let x: f32 = 135f32/255f32", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::BinaryOp(op, left, right) if op.op == "/" => {
                    if left.kind != ExprKind::FloatLiteral(135.0) {
                        return Err(format!("Expected left operand 135.0, got {:?}", **left));
                    }
                    if right.kind != ExprKind::FloatLiteral(255.0) {
                        return Err(format!("Expected right operand 255.0, got {:?}", **right));
                    }
                    Ok(())
                }
                _ => Err(format!("Expected BinaryOp expression, got {:?}", decl.body)),
            },
            _ => Err("Expected Let declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_vertex_attribute() {
    expect_parse("#[vertex] def main(): [4]f32 = result", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.attributes != vec![Attribute::Vertex] {
                    return Err(format!("Expected Vertex attribute, got {:?}", decl.attributes));
                }
                if decl.name != "main" {
                    return Err(format!("Expected name 'main', got '{}'", decl.name));
                }
                Ok(())
            }
            _ => Err("Expected Decl declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_fragment_attribute() {
    expect_parse("#[fragment] def frag(): [4]f32 = result", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.attributes != vec![Attribute::Fragment] {
                    return Err(format!("Expected Fragment attribute, got {:?}", decl.attributes));
                }
                if decl.name != "frag" {
                    return Err(format!("Expected name 'frag', got '{}'", decl.name));
                }
                Ok(())
            }
            _ => Err("Expected Decl declaration".to_string()),
        }
    });
}

#[test]
fn test_operator_precedence_and_associativity() {
    // Test expression: a + b * c - d / e + f
    // According to spec, * and / have higher precedence than + and -
    // All are left-associative
    // Should parse as: ((a + (b * c)) - (d / e)) + f
    expect_parse("def result: i32 = a + b * c - d / e + f", |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                // The outermost operation should be the last + (left-associative)
                match &decl.body.kind {
                    ExprKind::BinaryOp(outer_op, outer_left, outer_right) if outer_op.op == "+" => {
                        // Right side should be 'f'
                        if !matches!(outer_right.kind, ExprKind::Identifier(ref name) if name == "f") {
                            return Err(format!("Expected right side to be 'f', got {:?}", outer_right));
                        }

                        // Left side should be (a + (b * c)) - (d / e)
                        match &outer_left.kind {
                            ExprKind::BinaryOp(sub_op, sub_left, sub_right) if sub_op.op == "-" => {
                                // Right side of subtraction should be (d / e)
                                match &sub_right.kind {
                                    ExprKind::BinaryOp(div_op, div_left, div_right) if div_op.op == "/" => {
                                        if !matches!(div_left.kind, ExprKind::Identifier(ref name) if name == "d")
                                        {
                                            return Err(format!(
                                                "Expected 'd' in division left, got {:?}",
                                                div_left
                                            ));
                                        }
                                        if !matches!(div_right.kind, ExprKind::Identifier(ref name) if name == "e")
                                        {
                                            return Err(format!(
                                                "Expected 'e' in division right, got {:?}",
                                                div_right
                                            ));
                                        }
                                    }
                                    _ => {
                                        return Err(format!(
                                            "Expected division on right of subtraction, got {:?}",
                                            sub_right
                                        ));
                                    }
                                }

                                // Left side of subtraction should be a + (b * c)
                                match &sub_left.kind {
                                    ExprKind::BinaryOp(add_op, add_left, add_right) if add_op.op == "+" => {
                                        if !matches!(add_left.kind, ExprKind::Identifier(ref name) if name == "a")
                                        {
                                            return Err(format!(
                                                "Expected 'a' on left of first addition, got {:?}",
                                                add_left
                                            ));
                                        }

                                        // Right side should be (b * c)
                                        match &add_right.kind {
                                            ExprKind::BinaryOp(mul_op, mul_left, mul_right)
                                                if mul_op.op == "*" =>
                                            {
                                                if !matches!(mul_left.kind, ExprKind::Identifier(ref name) if name == "b")
                                                {
                                                    return Err(format!(
                                                        "Expected 'b' in multiplication left, got {:?}",
                                                        mul_left
                                                    ));
                                                }
                                                if !matches!(mul_right.kind, ExprKind::Identifier(ref name) if name == "c")
                                                {
                                                    return Err(format!(
                                                        "Expected 'c' in multiplication right, got {:?}",
                                                        mul_right
                                                    ));
                                                }
                                                Ok(())
                                            }
                                            _ => Err(format!(
                                                "Expected multiplication on right of first addition, got {:?}",
                                                add_right
                                            )),
                                        }
                                    }
                                    _ => Err(format!(
                                        "Expected addition on left of subtraction, got {:?}",
                                        sub_left
                                    )),
                                }
                            }
                            _ => Err(format!(
                                "Expected subtraction on left of final addition, got {:?}",
                                outer_left
                            )),
                        }
                    }
                    _ => Err(format!(
                        "Expected outermost operation to be addition, got {:?}",
                        decl.body
                    )),
                }
            }
            _ => Err("Expected Decl declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_builtin_attribute_on_return_type() {
    expect_parse(
        "#[vertex] def main(): #[builtin(position)] [4]f32 = result",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.attributes != vec![Attribute::Vertex] {
                        return Err(format!("Expected Vertex attribute, got {:?}", decl.attributes));
                    }
                    if let Some(ref attributed_types) = decl.attributed_return_type {
                        if attributed_types.len() != 1 {
                            return Err(format!(
                                "Expected 1 attributed return type, got {}",
                                attributed_types.len()
                            ));
                        }
                        if attributed_types[0].attributes
                            != vec![Attribute::BuiltIn(spirv::BuiltIn::Position)]
                        {
                            return Err(format!(
                                "Expected Position builtin on return type, got {:?}",
                                attributed_types[0].attributes
                            ));
                        }
                    } else {
                        return Err("Expected attributed return type".to_string());
                    }
                    // Check the type within the attributed return type
                    if let Some(ref attributed_types) = decl.attributed_return_type {
                        if attributed_types[0].ty
                            != crate::ast::types::sized_array(4, crate::ast::types::f32())
                        {
                            return Err(format!(
                                "Expected [4]f32 return type, got {:?}",
                                attributed_types[0].ty
                            ));
                        }
                    }
                    Ok(())
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
#[should_panic(expected = "Size variables in array types not yet implemented")]
fn test_parse_single_attributed_return_type() {
    // Test single attributed return type
    expect_parse(
        "#[vertex] def vertex_main(): [builtin(position)] vec4 = vec4 0.0f32 0.0f32 0.0f32 1.0f32",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }

            if let Declaration::Decl(decl) = &declarations[0] {
                // Check that it has vertex attribute
                if !decl.attributes.iter().any(|a| matches!(a, Attribute::Vertex)) {
                    return Err("Missing vertex attribute".to_string());
                }

                // Check the attributed return type
                if let Some(attributed_types) = &decl.attributed_return_type {
                    if attributed_types.len() != 1 {
                        return Err(format!(
                            "Expected 1 attributed type, got {}",
                            attributed_types.len()
                        ));
                    }

                    let attr_type = &attributed_types[0];
                    // Check for builtin(position) attribute
                    if !attr_type
                        .attributes
                        .iter()
                        .any(|a| matches!(a, Attribute::BuiltIn(spirv::BuiltIn::Position)))
                    {
                        return Err("Missing builtin(position) attribute".to_string());
                    }

                    // Check the type is vec4
                    match &attr_type.ty {
                        Type::Constructed(TypeName::Str("vec4"), _) => Ok(()),
                        _ => Err("Expected vec4 type".to_string()),
                    }
                } else {
                    Err("Missing attributed return type".to_string())
                }
            } else {
                Err("Expected Decl".to_string())
            }
        },
    );
}

#[test]
#[should_panic(expected = "Size variables in array types not yet implemented")]
fn test_parse_tuple_attributed_return_type() {
    // Test tuple of attributed return types
    expect_parse(
        "#[vertex] def vertex_main(): ([builtin(position)] vec4, [location(0)] vec3) = result",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }

            if let Declaration::Decl(decl) = &declarations[0] {
                if let Some(attributed_types) = &decl.attributed_return_type {
                    if attributed_types.len() != 2 {
                        return Err(format!(
                            "Expected 2 attributed types, got {}",
                            attributed_types.len()
                        ));
                    }

                    // Check first element: [builtin(position)] vec4
                    let first = &attributed_types[0];
                    if !first
                        .attributes
                        .iter()
                        .any(|a| matches!(a, Attribute::BuiltIn(spirv::BuiltIn::Position)))
                    {
                        return Err("First element missing builtin(position) attribute".to_string());
                    }

                    // Check second element: [location(0)] vec3
                    let second = &attributed_types[1];
                    if !second.attributes.iter().any(|a| matches!(a, Attribute::Location(0))) {
                        return Err("Second element missing location(0) attribute".to_string());
                    }

                    Ok(())
                } else {
                    Err("Missing attributed return type".to_string())
                }
            } else {
                Err("Expected Decl".to_string())
            }
        },
    );
}

#[test]
fn test_parse_unattributed_return_type() {
    // Test regular return type without attributes
    expect_parse(
        "def helper(): vec4 = vec4 1.0f32 0.0f32 0.0f32 1.0f32",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }

            if let Declaration::Decl(decl) = &declarations[0] {
                // Should have no attributed_return_type
                if decl.attributed_return_type.is_some() {
                    return Err("Unexpected attributed return type".to_string());
                }

                // Should have a regular type
                if let Some(ty) = &decl.ty {
                    match ty {
                        Type::Constructed(TypeName::Str("vec4"), _) => Ok(()),
                        _ => Err("Expected vec4 type".to_string()),
                    }
                } else {
                    Err("Missing return type".to_string())
                }
            } else {
                Err("Expected Decl".to_string())
            }
        },
    );
}

#[test]
fn test_parse_location_attribute_on_return_type() {
    expect_parse(
        "#[fragment] def frag(): #[location(0)] [4]f32 = result",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.attributes != vec![Attribute::Fragment] {
                        return Err(format!("Expected Fragment attribute, got {:?}", decl.attributes));
                    }
                    if let Some(ref attributed_types) = decl.attributed_return_type {
                        if attributed_types.len() != 1 {
                            return Err(format!(
                                "Expected 1 attributed return type, got {}",
                                attributed_types.len()
                            ));
                        }
                        if attributed_types[0].attributes != vec![Attribute::Location(0)] {
                            return Err(format!(
                                "Expected Location(0) attribute on return type, got {:?}",
                                attributed_types[0].attributes
                            ));
                        }
                    } else {
                        return Err("Expected attributed return type".to_string());
                    }
                    // Check the type within the attributed return type
                    if let Some(ref attributed_types) = decl.attributed_return_type {
                        if attributed_types[0].ty
                            != crate::ast::types::sized_array(4, crate::ast::types::f32())
                        {
                            return Err(format!(
                                "Expected [4]f32 return type, got {:?}",
                                attributed_types[0].ty
                            ));
                        }
                    }
                    Ok(())
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_parameter_with_builtin_attribute() {
    expect_parse(
        "#[vertex] def main(#[builtin(vertex_index)] vid: i32): [4]f32 = result",
        |declarations| {
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
                            if param.name != "vid" {
                                return Err(format!("Expected param name 'vid', got '{}'", param.name));
                            }
                            if param.ty != crate::ast::types::i32() {
                                return Err(format!("Expected i32 param type, got {:?}", param.ty));
                            }
                            if param.attributes != vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)] {
                                return Err(format!(
                                    "Expected VertexIndex attribute, got {:?}",
                                    param.attributes
                                ));
                            }
                            Ok(())
                        }
                        _ => Err("Expected typed parameter".to_string()),
                    }
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_parameter_with_location_attribute() {
    expect_parse(
        "#[fragment] def frag(#[location(1)] color: [3]f32): [4]f32 = result",
        |declarations| {
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
                            if param.name != "color" {
                                return Err(format!("Expected param name 'color', got '{}'", param.name));
                            }
                            if param.ty != crate::ast::types::sized_array(3, crate::ast::types::f32()) {
                                return Err(format!("Expected [3]f32 param type, got {:?}", param.ty));
                            }
                            if param.attributes != vec![Attribute::Location(1)] {
                                return Err(format!(
                                    "Expected Location(1) attribute, got {:?}",
                                    param.attributes
                                ));
                            }
                            Ok(())
                        }
                        _ => Err("Expected typed parameter".to_string()),
                    }
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_multiple_builtin_types() {
    expect_parse(
        "#[vertex] def main(#[builtin(vertex_index)] vid: i32, #[builtin(instance_index)] iid: i32): #[builtin(position)] [4]f32 = result",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.params.len() != 2 {
                        return Err(format!("Expected 2 parameters, got {}", decl.params.len()));
                    }

                    // First parameter
                    match &decl.params[0] {
                        DeclParam::Typed(param) => {
                            if param.name != "vid" {
                                return Err(format!(
                                    "Expected first param name 'vid', got '{}'",
                                    param.name
                                ));
                            }
                            if param.attributes != vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)] {
                                return Err(format!(
                                    "Expected VertexIndex attribute, got {:?}",
                                    param.attributes
                                ));
                            }
                        }
                        _ => return Err("Expected typed parameter".to_string()),
                    }

                    // Second parameter
                    match &decl.params[1] {
                        DeclParam::Typed(param) => {
                            if param.name != "iid" {
                                return Err(format!(
                                    "Expected second param name 'iid', got '{}'",
                                    param.name
                                ));
                            }
                            if param.attributes != vec![Attribute::BuiltIn(spirv::BuiltIn::InstanceIndex)] {
                                return Err(format!(
                                    "Expected InstanceIndex attribute, got {:?}",
                                    param.attributes
                                ));
                            }
                        }
                        _ => return Err("Expected typed parameter".to_string()),
                    }

                    // Return type
                    if let Some(ref attributed_types) = decl.attributed_return_type {
                        if attributed_types.len() != 1 {
                            return Err(format!(
                                "Expected 1 attributed return type, got {}",
                                attributed_types.len()
                            ));
                        }
                        if attributed_types[0].attributes
                            != vec![Attribute::BuiltIn(spirv::BuiltIn::Position)]
                        {
                            return Err(format!(
                                "Expected Position attribute on return type, got {:?}",
                                attributed_types[0].attributes
                            ));
                        }
                    } else {
                        return Err("Expected attributed return type".to_string());
                    }

                    Ok(())
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_simple_lambda() {
    expect_parse(r#"let f: i32 -> i32 = \x -> x"#, |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => {
                if decl.name != "f" {
                    return Err(format!("Expected name 'f', got '{}'", decl.name));
                }
                match &decl.body.kind {
                    ExprKind::Lambda(lambda) => {
                        if lambda.params.len() != 1 {
                            return Err(format!("Expected 1 lambda param, got {}", lambda.params.len()));
                        }
                        if lambda.params[0].name != "x" {
                            return Err(format!(
                                "Expected param name 'x', got '{}'",
                                lambda.params[0].name
                            ));
                        }
                        if lambda.params[0].ty.is_some() {
                            return Err(format!("Expected no param type, got {:?}", lambda.params[0].ty));
                        }
                        if lambda.return_type.is_some() {
                            return Err(format!("Expected no return type, got {:?}", lambda.return_type));
                        }
                        match lambda.body.kind {
                            ExprKind::Identifier(ref name) => {
                                if name != "x" {
                                    return Err(format!(
                                        "Expected identifier 'x' in lambda body, got '{}'",
                                        name
                                    ));
                                }
                            }
                            _ => {
                                return Err(format!(
                                    "Expected identifier in lambda body, got {:?}",
                                    lambda.body
                                ));
                            }
                        }
                        Ok(())
                    }
                    _ => Err(format!("Expected lambda expression, got {:?}", decl.body)),
                }
            }
            _ => Err("Expected Let declaration".to_string()),
        }
    });
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
    expect_parse(r#"let f: f32 -> f32 = \x -> x"#, |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::Lambda(lambda) => {
                    if lambda.params.len() != 1 {
                        return Err(format!("Expected 1 lambda param, got {}", lambda.params.len()));
                    }
                    if lambda.params[0].name != "x" {
                        return Err(format!(
                            "Expected param name 'x', got '{}'",
                            lambda.params[0].name
                        ));
                    }
                    if lambda.params[0].ty.is_some() {
                        return Err(format!(
                            "Expected no param type (parameters are untyped), got {:?}",
                            lambda.params[0].ty
                        ));
                    }
                    if lambda.return_type.is_some() {
                        return Err(format!(
                            "Expected no return type annotation in lambda, got {:?}",
                            lambda.return_type
                        ));
                    }
                    match lambda.body.kind {
                        ExprKind::Identifier(ref name) => {
                            if name != "x" {
                                return Err(format!(
                                    "Expected identifier 'x' in lambda body, got '{}'",
                                    name
                                ));
                            }
                        }
                        _ => {
                            return Err(format!(
                                "Expected identifier in lambda body, got {:?}",
                                lambda.body
                            ));
                        }
                    }
                    Ok(())
                }
                _ => Err(format!("Expected lambda expression, got {:?}", decl.body)),
            },
            _ => Err("Expected Let declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_lambda_with_multiple_params() {
    expect_parse(r#"let add: i32 -> i32 -> i32 = \x y -> x"#, |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::Lambda(lambda) => {
                    if lambda.params.len() != 2 {
                        return Err(format!("Expected 2 lambda params, got {}", lambda.params.len()));
                    }
                    if lambda.params[0].name != "x" {
                        return Err(format!(
                            "Expected first param name 'x', got '{}'",
                            lambda.params[0].name
                        ));
                    }
                    if lambda.params[0].ty.is_some() {
                        return Err(format!(
                            "Expected no type for first param, got {:?}",
                            lambda.params[0].ty
                        ));
                    }
                    if lambda.params[1].name != "y" {
                        return Err(format!(
                            "Expected second param name 'y', got '{}'",
                            lambda.params[1].name
                        ));
                    }
                    if lambda.params[1].ty.is_some() {
                        return Err(format!(
                            "Expected no type for second param, got {:?}",
                            lambda.params[1].ty
                        ));
                    }
                    if lambda.return_type.is_some() {
                        return Err(format!("Expected no return type, got {:?}", lambda.return_type));
                    }
                    Ok(())
                }
                _ => Err(format!("Expected lambda expression, got {:?}", decl.body)),
            },
            _ => Err("Expected Let declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_function_application() {
    expect_parse(r#"let result: i32 = f(42, 24)"#, |declarations| {
        if declarations.len() != 1 {
            return Err(format!("Expected 1 declaration, got {}", declarations.len()));
        }
        match &declarations[0] {
            Declaration::Decl(decl) => match &decl.body.kind {
                ExprKind::Application(func, args) => {
                    match func.kind {
                        ExprKind::Identifier(ref name) => {
                            if name != "f" {
                                return Err(format!("Expected function identifier 'f', got '{}'", name));
                            }
                        }
                        _ => return Err(format!("Expected function identifier, got {:?}", func)),
                    }
                    if args.len() != 2 {
                        return Err(format!("Expected 2 arguments, got {}", args.len()));
                    }
                    match &args[0].kind {
                        ExprKind::IntLiteral(n) => {
                            if *n != 42 {
                                return Err(format!("Expected first argument 42, got {}", n));
                            }
                        }
                        _ => {
                            return Err(format!(
                                "Expected int literal for first argument, got {:?}",
                                args[0]
                            ));
                        }
                    }
                    match &args[1].kind {
                        ExprKind::IntLiteral(n) => {
                            if *n != 24 {
                                return Err(format!("Expected second argument 24, got {}", n));
                            }
                        }
                        _ => {
                            return Err(format!(
                                "Expected int literal for second argument, got {:?}",
                                args[1]
                            ));
                        }
                    }
                    Ok(())
                }
                _ => Err(format!("Expected function application, got {:?}", decl.body)),
            },
            _ => Err("Expected Let declaration".to_string()),
        }
    });
}

#[test]
fn test_parse_simple_let_in() {
    expect_parse(
        "#[vertex] def main(x: i32): i32 = let y = 5 in y + x",
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            // Just verify it parses successfully - the structure is complex to validate in detail
            Ok(())
        },
    );
}

#[test]
fn test_parse_let_in_expression_only() {
    // Test parsing just the let..in expression by itself
    let input = r#"let f = \y -> y + x in f 10"#;
    let tokens = tokenize(input).expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let result = parser.parse_expression();

    // Just verify it parses without error - this is a complex expression
    match result {
        Ok(_expr) => {
            // Test passes if parsing succeeds
        }
        Err(e) => panic!("Failed to parse let..in expression: {:?}", e),
    }
}

#[test]
fn test_parse_let_in_with_lambda() {
    expect_parse(
        r#"#[vertex] def main(x: i32): i32 = let f = \y -> y + x in f 10"#,
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            // Just verify it parses successfully - the lambda let..in structure is complex
            Ok(())
        },
    );
}

#[test]
fn test_parse_multiple_top_level_lets_with_entry() {
    // Test the failing case from integration tests: multiple let declarations with entry points
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

    expect_parse(input, |declarations| {
        if declarations.len() != 4 {
            return Err(format!(
                "Expected 4 declarations (2 lets + 2 entries), got {}",
                declarations.len()
            ));
        }

        // First should be def verts
        match &declarations[0] {
            Declaration::Decl(decl) if decl.keyword == "def" => {
                if decl.name != "verts" {
                    return Err(format!("Expected first def to be 'verts', got '{}'", decl.name));
                }
            }
            _ => return Err("Expected first declaration to be Def".to_string()),
        }

        // Second should be vertex entry point (now a Decl with vertex attribute)
        match &declarations[1] {
            Declaration::Decl(decl) => {
                if decl.name != "vertex_main" {
                    return Err(format!(
                        "Expected vertex entry to be 'vertex_main', got '{}'",
                        decl.name
                    ));
                }
                if !decl.attributes.contains(&Attribute::Vertex) {
                    return Err(format!(
                        "Expected vertex attribute on vertex_main, got {:?}",
                        decl.attributes
                    ));
                }
            }
            _ => return Err("Expected second declaration to be Decl".to_string()),
        }

        // Third should be def SKY_RGBA
        match &declarations[2] {
            Declaration::Decl(decl) if decl.keyword == "def" => {
                if decl.name != "SKY_RGBA" {
                    return Err(format!(
                        "Expected third def to be 'SKY_RGBA', got '{}'",
                        decl.name
                    ));
                }
            }
            _ => return Err("Expected third declaration to be Def".to_string()),
        }

        // Fourth should be fragment entry point (now a Decl with fragment attribute)
        match &declarations[3] {
            Declaration::Decl(decl) => {
                if decl.name != "fragment_main" {
                    return Err(format!(
                        "Expected fragment entry to be 'fragment_main', got '{}'",
                        decl.name
                    ));
                }
                if !decl.attributes.contains(&Attribute::Fragment) {
                    return Err(format!(
                        "Expected fragment attribute on fragment_main, got {:?}",
                        decl.attributes
                    ));
                }
            }
            _ => return Err("Expected fourth declaration to be Decl".to_string()),
        }

        Ok(())
    });
}

#[test]
fn test_field_access_parsing() {
    expect_parse("def x: f32 = v.x", |declarations| {
        assert_eq!(declarations.len(), 1);
        match &declarations[0] {
            Declaration::Decl(decl) => {
                assert_eq!(decl.name, "x");
                match &decl.body.kind {
                    ExprKind::FieldAccess(expr, field) => {
                        assert_eq!(field, "x");
                        match expr.kind {
                            ExprKind::Identifier(ref name) => {
                                if name == "v" {
                                    Ok(())
                                } else {
                                    Err(format!("Expected identifier 'v', got '{}'", name))
                                }
                            }
                            _ => Err(format!("Expected identifier in field access, got: {:?}", expr)),
                        }
                    }
                    _ => Err(format!("Expected field access expression, got: {:?}", decl.body)),
                }
            }
            _ => Err("Expected Decl declaration".to_string()),
        }
    });
}

#[test]
fn test_simple_identifier_parsing() {
    expect_parse("def x: f32 = y", |declarations| {
        assert_eq!(declarations.len(), 1);
        match &declarations[0] {
            Declaration::Decl(decl) => {
                assert_eq!(decl.name, "x");
                match &decl.body.kind {
                    ExprKind::Identifier(name) => {
                        if name == "y" {
                            Ok(())
                        } else {
                            Err(format!("Expected identifier 'y', got '{}'", name))
                        }
                    }
                    _ => Err(format!("Expected identifier expression, got: {:?}", decl.body)),
                }
            }
            _ => Err("Expected Decl declaration".to_string()),
        }
    });
}

#[test]
fn test_vector_field_access_file() {
    expect_parse(
        "def v: vec3 = vec3 1.0f32 2.0f32 3.0f32\ndef x: f32 = v.x",
        |declarations| {
            assert_eq!(declarations.len(), 2);

            // Check first declaration: def v: vec3 = vec3 1.0f32 2.0f32 3.0f32
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    assert_eq!(decl.name, "v");
                    // Body should be a function call: vec3 1.0f32 2.0f32 3.0f32
                    match &decl.body.kind {
                        ExprKind::FunctionCall(func_name, args) => {
                            assert_eq!(func_name, "vec3");
                            assert_eq!(args.len(), 3);
                        }
                        _ => {
                            return Err(format!(
                                "Expected function call for vec3 constructor, got: {:?}",
                                decl.body
                            ));
                        }
                    }
                }
                _ => return Err("Expected first declaration to be Decl".to_string()),
            }

            // Check second declaration: def x: f32 = v.x
            match &declarations[1] {
                Declaration::Decl(decl) => {
                    assert_eq!(decl.name, "x");
                    match &decl.body.kind {
                        ExprKind::FieldAccess(expr, field) => {
                            assert_eq!(field, "x");
                            match expr.kind {
                                ExprKind::Identifier(ref name) => {
                                    if name == "v" {
                                        Ok(())
                                    } else {
                                        Err(format!("Expected identifier 'v', got '{}'", name))
                                    }
                                }
                                _ => Err(format!("Expected identifier in field access, got: {:?}", expr)),
                            }
                        }
                        _ => Err(format!("Expected field access expression, got: {:?}", decl.body)),
                    }
                }
                _ => Err("Expected second declaration to be Decl".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_vector_arithmetic() {
    expect_parse(
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
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.name != "test_vector_arithmetic" {
                        return Err(format!(
                            "Expected name 'test_vector_arithmetic', got '{}'",
                            decl.name
                        ));
                    }
                    if decl.ty != Some(crate::ast::types::f32()) {
                        return Err(format!("Expected f32 type, got {:?}", decl.ty));
                    }
                    // Check that the body contains nested let-in expressions with binary operations
                    match &decl.body.kind {
                        ExprKind::LetIn(_) => Ok(()),
                        _ => Err(format!("Expected LetIn expression, got {:?}", decl.body)),
                    }
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_uniform_attribute() {
    expect_parse(
        r#"
            #[uniform] def material_color: vec3
            "#,
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Uniform(uniform_decl) => {
                    if uniform_decl.name != "material_color" {
                        return Err(format!(
                            "Expected name 'material_color', got '{}'",
                            uniform_decl.name
                        ));
                    }
                    if uniform_decl.ty != crate::ast::types::vec3() {
                        return Err(format!("Expected vec3 type, got {:?}", uniform_decl.ty));
                    }
                    Ok(())
                }
                _ => Err("Expected Uniform declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_parse_uniform_without_initializer() {
    expect_parse(
        r#"
            #[uniform] def material_color: vec3
            "#,
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Uniform(uniform_decl) => {
                    if uniform_decl.name != "material_color" {
                        return Err(format!(
                            "Expected name 'material_color', got '{}'",
                            uniform_decl.name
                        ));
                    }
                    if uniform_decl.ty != crate::ast::types::vec3() {
                        return Err(format!("Expected vec3 type, got {:?}", uniform_decl.ty));
                    }
                    // Check that there's no initializer (uniforms don't have bodies)
                    Ok(())
                }
                _ => Err("Expected Uniform declaration".to_string()),
            }
        },
    );
}

#[test]
fn test_uniform_with_initializer_error() {
    expect_parse_error(
        r#"
            #[uniform] def material_color: vec3 = vec3 1.0f32 0.5f32 0.2f32
            "#,
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
#[should_panic(expected = "Size variables in array types not yet implemented")]
fn test_parse_multiple_shader_outputs() {
    expect_parse(
        r#"
            #[fragment] def fragment_main(): ([location(0)] vec4, [location(1)] vec3) = 
              let color = vec4 1.0f32 0.5f32 0.2f32 1.0f32 in
              let normal = vec3 0.0f32 1.0f32 0.0f32 in
              (color, normal)
            "#,
        |declarations| {
            if declarations.len() != 1 {
                return Err(format!("Expected 1 declaration, got {}", declarations.len()));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.name != "fragment_main" {
                        return Err(format!("Expected name 'fragment_main', got '{}'", decl.name));
                    }
                    if !decl.attributes.contains(&Attribute::Fragment) {
                        return Err(format!("Expected Fragment attribute, got {:?}", decl.attributes));
                    }
                    // Check that we have attributed return type
                    if decl.attributed_return_type.is_none() {
                        return Err("Expected attributed return type for multiple outputs".to_string());
                    }
                    let attributed_types = decl.attributed_return_type.as_ref().unwrap();
                    if attributed_types.len() != 2 {
                        return Err(format!(
                            "Expected 2 output components, got {}",
                            attributed_types.len()
                        ));
                    }

                    // Check first output: [location(0)] vec4
                    if attributed_types[0].attributes != vec![Attribute::Location(0)] {
                        return Err(format!(
                            "Expected Location(0) on first output, got {:?}",
                            attributed_types[0].attributes
                        ));
                    }

                    // Check second output: [location(1)] vec3
                    if attributed_types[1].attributes != vec![Attribute::Location(1)] {
                        return Err(format!(
                            "Expected Location(1) on second output, got {:?}",
                            attributed_types[1].attributes
                        ));
                    }

                    Ok(())
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        },
    );
}

#[test]
#[should_panic(expected = "Size variables in array types not yet implemented")]
fn test_parse_complete_shader_example() {
    expect_parse(
        r#"
            -- Complete shader example with multiple outputs
            #[uniform] def material_color: vec3
            #[uniform] def time: f32

            #[vertex] def vertex_main(): ([builtin(position)] vec4, [location(0)] vec3) =
              let angle: f32 = time in
              let x: f32 = sin angle in
              let y: f32 = cos angle in
              let position: vec4 = vec4 x y 0.0f32 1.0f32 in
              let color: vec3 = material_color in
              (position, color)

            #[fragment] def fragment_main(): ([location(0)] vec4, [location(1)] vec3) = 
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
                Declaration::Decl(decl) => {
                    if decl.name != "material_color" || !decl.attributes.contains(&Attribute::Uniform) {
                        return Err("Expected uniform material_color".to_string());
                    }
                }
                _ => return Err("Expected uniform declaration".to_string()),
            }

            match &declarations[1] {
                Declaration::Decl(decl) => {
                    if decl.name != "time" || !decl.attributes.contains(&Attribute::Uniform) {
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
