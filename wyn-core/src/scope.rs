use std::collections::HashMap;

/// A single scope containing variable bindings
#[derive(Debug, Clone)]
pub struct Scope<T> {
    bindings: HashMap<String, T>,
}

impl<T> Default for Scope<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Scope<T> {
    pub fn new() -> Self {
        Scope {
            bindings: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, value: T) {
        self.bindings.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<&T> {
        self.bindings.get(name)
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.bindings.keys()
    }
}

/// A stack-based scope manager that tracks nested scopes
#[derive(Debug, Clone)]
pub struct ScopeStack<T> {
    scopes: Vec<Scope<T>>,
}

impl<T> Default for ScopeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ScopeStack<T> {
    /// Create a new scope stack with a global scope
    pub fn new() -> Self {
        ScopeStack {
            scopes: vec![Scope::new()],
        }
    }

    /// Push a new scope onto the stack
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Pop the current scope from the stack
    /// Returns None if trying to pop the global scope
    pub fn pop_scope(&mut self) -> Option<Scope<T>> {
        if self.scopes.len() > 1 {
            self.scopes.pop()
        } else {
            None
        }
    }

    /// Insert a binding in the current (innermost) scope
    pub fn insert(&mut self, name: String, value: T) {
        if let Some(current_scope) = self.scopes.last_mut() {
            current_scope.insert(name, value);
        }
    }

    /// Look up a binding, searching from innermost to outermost scope
    pub fn lookup(&self, name: &str) -> Option<&T> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        None
    }

    /// Check if a name is defined in the current scope (not outer scopes)
    pub fn is_defined_in_current_scope(&self, name: &str) -> bool {
        self.scopes
            .last()
            .map(|scope| scope.contains_key(name))
            .unwrap_or(false)
    }

    /// Check if a name is defined in any scope
    pub fn is_defined(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }

    /// Get the current scope depth (0 = global scope)
    pub fn depth(&self) -> usize {
        self.scopes.len().saturating_sub(1)
    }

    /// Get all bindings in the current scope
    pub fn current_scope_bindings(&self) -> impl Iterator<Item = (&String, &T)> {
        self.scopes
            .last()
            .map(|scope| scope.bindings.iter())
            .into_iter()
            .flatten()
    }

    /// Collect all names that are defined in outer scopes but not current scope
    /// This is useful for free variable analysis
    pub fn collect_free_variables(&self, used_names: &[String]) -> Vec<String> {
        let mut free_vars = Vec::new();

        if let Some(current_scope) = self.scopes.last() {
            for name in used_names {
                // If the name is used but not defined in current scope,
                // check if it's defined in outer scopes
                if !current_scope.contains_key(name) {
                    // Search in outer scopes
                    for outer_scope in self.scopes[..self.scopes.len() - 1].iter().rev() {
                        if outer_scope.contains_key(name) {
                            free_vars.push(name.clone());
                            break;
                        }
                    }
                }
            }
        }

        free_vars
    }
}

// Manual scope management - use push_scope() and pop_scope() explicitly

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_scope_operations() {
        let mut scope_stack: ScopeStack<i32> = ScopeStack::new();

        // Insert in global scope
        scope_stack.insert("x".to_string(), 1);
        assert_eq!(scope_stack.lookup("x"), Some(&1));

        // Push new scope and shadow variable
        scope_stack.push_scope();
        scope_stack.insert("x".to_string(), 2);
        scope_stack.insert("y".to_string(), 3);

        assert_eq!(scope_stack.lookup("x"), Some(&2)); // Shadows outer x
        assert_eq!(scope_stack.lookup("y"), Some(&3));

        // Pop scope
        scope_stack.pop_scope();
        assert_eq!(scope_stack.lookup("x"), Some(&1)); // Back to outer x
        assert_eq!(scope_stack.lookup("y"), None); // y is gone
    }

    #[test]
    fn test_free_variables() {
        let mut scope_stack: ScopeStack<i32> = ScopeStack::new();

        // Global scope
        scope_stack.insert("global_var".to_string(), 1);

        // Outer function scope
        scope_stack.push_scope();
        scope_stack.insert("outer_param".to_string(), 2);

        // Inner lambda scope
        scope_stack.push_scope();
        scope_stack.insert("inner_param".to_string(), 3);

        let used_names = vec![
            "inner_param".to_string(), // Defined in current scope
            "outer_param".to_string(), // Free variable from outer scope
            "global_var".to_string(),  // Free variable from global scope
            "undefined".to_string(),   // Not defined anywhere
        ];

        let free_vars = scope_stack.collect_free_variables(&used_names);

        // Should include variables from outer scopes, not current scope or undefined
        assert!(free_vars.contains(&"outer_param".to_string()));
        assert!(free_vars.contains(&"global_var".to_string()));
        assert!(!free_vars.contains(&"inner_param".to_string()));
        assert!(!free_vars.contains(&"undefined".to_string()));
    }

    #[test]
    fn test_manual_scope_management() {
        let mut scope_stack: ScopeStack<i32> = ScopeStack::new();
        scope_stack.insert("x".to_string(), 1);

        // Manual scope push
        scope_stack.push_scope();
        scope_stack.insert("x".to_string(), 2);
        assert_eq!(scope_stack.lookup("x"), Some(&2));

        // Manual scope pop
        scope_stack.pop_scope();
        assert_eq!(scope_stack.lookup("x"), Some(&1));
    }
}
