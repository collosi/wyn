//! Environment/scope management for code generation
//!
//! Provides a stack-based scoping system with special support for:
//! - Built-in functions (bottom layer, never popped)
//! - Global constants (second layer, never popped)
//! - Local scopes (pushed/popped as needed)

use super::Value;
use crate::ast::Type;
use std::collections::HashMap;

/// Binding combines a SPIR-V value with its source-level type
#[derive(Debug, Clone)]
pub struct Binding {
    pub value: Value,
    pub ty: Option<Type>, // None for values that don't have a source type (e.g., some temporaries)
}

impl Binding {
    pub fn new(value: Value, ty: Type) -> Self {
        Binding { value, ty: Some(ty) }
    }

    pub fn value_only(value: Value) -> Self {
        Binding { value, ty: None }
    }
}

/// Environment manages variable and constant bindings during code generation
#[derive(Debug)]
pub struct Environment {
    /// Built-in functions and intrinsics (never popped)
    builtins: HashMap<String, Binding>,

    /// Global constants defined at module scope (never popped)
    globals: HashMap<String, Binding>,

    /// Stack of local scopes (can be pushed/popped)
    local_scopes: Vec<HashMap<String, Binding>>,
}

impl Environment {
    /// Create a new empty environment
    pub fn new() -> Self {
        Environment {
            builtins: HashMap::new(),
            globals: HashMap::new(),
            local_scopes: vec![HashMap::new()], // Start with one local scope
        }
    }

    /// Define a builtin (only accessible before any code generation)
    pub fn define_builtin(&mut self, name: String, value: Value, ty: Type) {
        self.builtins.insert(name, Binding::new(value, ty));
    }

    /// Define a global constant (only accessible during global constant processing)
    pub fn define_global(&mut self, name: String, value: Value, ty: Type) {
        self.globals.insert(name, Binding::new(value, ty));
    }

    /// Define a local variable in the current scope
    pub fn define_local(&mut self, name: String, value: Value, ty: Type) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, Binding::new(value, ty));
        }
    }

    /// Look up a variable by name, searching from innermost to outermost scope
    /// Search order: local scopes (innermost to outermost) -> globals -> builtins
    pub fn lookup(&self, name: &str) -> Option<&Binding> {
        // Search local scopes from innermost to outermost
        for scope in self.local_scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(binding);
            }
        }

        // Search globals
        if let Some(binding) = self.globals.get(name) {
            return Some(binding);
        }

        // Search builtins
        if let Some(binding) = self.builtins.get(name) {
            return Some(binding);
        }

        None
    }

    /// Push a new local scope onto the stack
    pub fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    /// Pop the top local scope from the stack
    /// Returns the popped scope for inspection if needed
    pub fn pop_scope(&mut self) -> Option<HashMap<String, Binding>> {
        if self.local_scopes.len() > 1 {
            self.local_scopes.pop()
        } else {
            // Don't pop the last scope
            None
        }
    }

    /// Clear all local scopes (useful when starting a new function)
    pub fn clear_locals(&mut self) {
        self.local_scopes.clear();
        self.local_scopes.push(HashMap::new());
    }

    /// Get the number of local scopes currently on the stack
    pub fn scope_depth(&self) -> usize {
        self.local_scopes.len()
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_lookup_order() {
        let mut env = Environment::new();

        // Add to different layers
        env.define_builtin("x".to_string(), Value { id: 1, type_id: 10 }, crate::ast::types::i32());
        env.define_global("y".to_string(), Value { id: 2, type_id: 20 }, crate::ast::types::i32());
        env.define_local("z".to_string(), Value { id: 3, type_id: 30 }, crate::ast::types::i32());

        // Lookup should find them
        assert_eq!(env.lookup("x").unwrap().value.id, 1);
        assert_eq!(env.lookup("y").unwrap().value.id, 2);
        assert_eq!(env.lookup("z").unwrap().value.id, 3);
    }

    #[test]
    fn test_shadowing() {
        let mut env = Environment::new();

        env.define_global("x".to_string(), Value { id: 1, type_id: 10 }, crate::ast::types::i32());

        // Push a new scope so we can actually pop it later
        env.push_scope();
        env.define_local("x".to_string(), Value { id: 2, type_id: 10 }, crate::ast::types::i32());

        // Local should shadow global
        assert_eq!(env.lookup("x").unwrap().value.id, 2);

        // After popping, should see global again
        env.pop_scope();
        assert_eq!(env.lookup("x").unwrap().value.id, 1);
    }

    #[test]
    fn test_scope_push_pop() {
        let mut env = Environment::new();

        env.define_local("a".to_string(), Value { id: 1, type_id: 10 }, crate::ast::types::i32());

        env.push_scope();
        env.define_local("b".to_string(), Value { id: 2, type_id: 10 }, crate::ast::types::i32());

        assert!(env.lookup("a").is_some());
        assert!(env.lookup("b").is_some());

        env.pop_scope();

        assert!(env.lookup("a").is_some());
        assert!(env.lookup("b").is_none());
    }

    #[test]
    fn test_cannot_pop_last_scope() {
        let mut env = Environment::new();

        env.define_local("x".to_string(), Value { id: 1, type_id: 10 }, crate::ast::types::i32());

        // Try to pop the only scope
        let result = env.pop_scope();
        assert!(result.is_none());

        // Should still be able to lookup
        assert!(env.lookup("x").is_some());
    }

    #[test]
    fn test_clear_locals() {
        let mut env = Environment::new();

        env.define_global("g".to_string(), Value { id: 1, type_id: 10 }, crate::ast::types::i32());
        env.define_local("l".to_string(), Value { id: 2, type_id: 10 }, crate::ast::types::i32());

        env.clear_locals();

        // Global should still be there
        assert!(env.lookup("g").is_some());
        // Local should be gone
        assert!(env.lookup("l").is_none());
    }
}
