//! Tests for the module system

use super::*;
use crate::ast::{Declaration, ModuleBind, ModuleExpression, Program};
use crate::parser::Parser;

#[test]
fn test_simple_module() {
    let source = r#"
        module M = {
            type t = i32
            def x: t = 42
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program);

    match result {
        Ok(elaborated) => {
            // Check that module was elaborated
            // For now, just verify it doesn't error
            assert!(elaborated.declarations.len() >= 2); // type and def
        }
        Err(e) => panic!("Elaboration failed: {}", e),
    }
}

#[test]
fn test_empty_module() {
    let source = r#"
        module M = {}
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program);

    match result {
        Ok(elaborated) => {
            // Empty module produces no declarations
            assert_eq!(elaborated.declarations.len(), 0);
        }
        Err(e) => panic!("Elaboration failed: {}", e),
    }
}

#[test]
fn test_module_name_qualification() {
    let source = r#"
        module M = {
            type t = i32
            def x: t = 42
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Check that names are qualified with M_
    assert_eq!(result.declarations.len(), 2);

    // Check type binding is qualified
    match &result.declarations[0] {
        crate::ast::Declaration::TypeBind(tb) => {
            assert_eq!(tb.name, "M_t");
        }
        _ => panic!("Expected TypeBind declaration"),
    }

    // Check value binding is qualified
    match &result.declarations[1] {
        crate::ast::Declaration::Decl(d) => {
            assert_eq!(d.name, "M_x");
        }
        _ => panic!("Expected Decl declaration"),
    }
}

#[test]
fn test_module_with_signature() {
    let source = r#"
        module M : { val x : i32 } = {
            type t = i32
            def x: i32 = 42
            def y: i32 = 99
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Should only export x, not t or y (filtered by signature)
    assert_eq!(result.declarations.len(), 1);

    match &result.declarations[0] {
        crate::ast::Declaration::Decl(d) => {
            assert_eq!(d.name, "M_x");
        }
        _ => panic!("Expected Decl declaration for x"),
    }
}

#[test]
fn test_module_signature_filters_types() {
    let source = r#"
        module M : { val x : i32 } = {
            type t = i32
            type u = f32
            def x: i32 = 42
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Should only export x, types t and u should be filtered out
    assert_eq!(result.declarations.len(), 1);
    assert!(matches!(
        &result.declarations[0],
        crate::ast::Declaration::Decl(_)
    ));
}

#[test]
fn test_module_signature_with_type() {
    let source = r#"
        module M : { type t val x : t } = {
            type t = i32
            def x: t = 42
            def y: i32 = 99
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Should export both type t and val x
    assert_eq!(result.declarations.len(), 2);

    assert!(matches!(&result.declarations[0], crate::ast::Declaration::TypeBind(tb) if tb.name == "M_t"));
    assert!(matches!(&result.declarations[1], crate::ast::Declaration::Decl(d) if d.name == "M_x"));
}

#[test]
fn test_module_signature_missing_required_value() {
    let source = r#"
        module M : { val x : i32 val y : i32 } = {
            def x: i32 = 42
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program);

    // Should fail because y is required but not provided
    assert!(result.is_err());
    assert!(matches!(result, Err(crate::error::CompilerError::ModuleError(_))));
}

#[test]
fn test_module_signature_missing_required_type() {
    let source = r#"
        module M : { type t type u } = {
            type t = i32
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program);

    // Should fail because type u is required but not provided
    assert!(result.is_err());
    assert!(matches!(result, Err(crate::error::CompilerError::ModuleError(_))));
}

#[test]
fn test_nested_module_path_qualification() {
    let source = r#"
        module M = {
            module N = {
                def x: i32 = 42
            }
        }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Should have nested qualified name M_$_N_x
    assert_eq!(result.declarations.len(), 1);
    match &result.declarations[0] {
        crate::ast::Declaration::Decl(d) => {
            assert_eq!(d.name, "M_$_N_x");
        }
        _ => panic!("Expected Decl declaration"),
    }
}

#[test]
fn test_module_type_bind_is_erased() {
    let source = r#"
        module type numeric = { type t }
        module M = { def x: i32 = 42 }
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Module type bindings should be erased, only module M's contents remain
    assert_eq!(result.declarations.len(), 1);
    assert!(matches!(&result.declarations[0], crate::ast::Declaration::Decl(d) if d.name == "M_x"));
}

#[test]
fn test_module_instantiation_with_type_param() {
    let source = r#"
        module type numeric = {
            type t
            val add: t -> t -> t
            val sub: t -> t -> t
        }

        module i32 : (numeric with t = i32)
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Should generate Val declarations for i32_add and i32_sub
    assert_eq!(result.declarations.len(), 3); // type t, val add, val sub

    // Check that we have a type binding for i32_t
    assert!(result.declarations.iter().any(|d| matches!(d,
        crate::ast::Declaration::TypeBind(tb) if tb.name == "i32_t"
    )));

    // Check that we have val declarations for operations
    assert!(result.declarations.iter().any(|d| matches!(d,
        crate::ast::Declaration::Val(v) if v.name == "i32_add"
    )));

    assert!(result.declarations.iter().any(|d| matches!(d,
        crate::ast::Declaration::Val(v) if v.name == "i32_sub"
    )));
}

#[test]
fn test_module_instantiation_with_include() {
    let source = r#"
        module type base = {
            type t
            val zero: t
        }

        module type numeric = {
            include base
            val add: t -> t -> t
        }

        module i32 : (numeric with t = i32)
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Should expand include and generate declarations for all specs
    assert!(result.declarations.len() >= 3); // type t, val zero, val add

    // Check that include was expanded
    assert!(result.declarations.iter().any(|d| matches!(d,
        crate::ast::Declaration::Val(v) if v.name == "i32_zero"
    )));

    assert!(result.declarations.iter().any(|d| matches!(d,
        crate::ast::Declaration::Val(v) if v.name == "i32_add"
    )));
}

#[test]
fn test_module_instantiation_chained_with() {
    let source = r#"
        module type float = {
            type t
            type int_t
            val to_bits: t -> int_t
        }

        module f32 : (float with t = f32 with int_t = u32)
    "#;

    let tokens = crate::lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut elaborator = ModuleElaborator::new();
    let result = elaborator.elaborate(program).unwrap();

    // Should have type bindings for both t and int_t, plus val for to_bits
    assert!(result.declarations.len() >= 3);

    // Check type bindings
    assert!(result.declarations.iter().any(|d| matches!(d,
        crate::ast::Declaration::TypeBind(tb) if tb.name == "f32_t"
    )));

    assert!(result.declarations.iter().any(|d| matches!(d,
        crate::ast::Declaration::TypeBind(tb) if tb.name == "f32_int_t"
    )));
}
