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
