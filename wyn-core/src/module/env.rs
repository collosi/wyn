//! Module environment for tracking module bindings and signatures

use crate::ast::{ModuleBind, ModuleTypeExpression, Type};
use std::collections::HashMap;

/// Qualified name: a path through nested modules
pub type QualifiedName = Vec<String>;

/// Module environment tracks all module bindings and their signatures
pub struct ModuleEnv {
    /// Map from module names to their signatures
    modules: HashMap<QualifiedName, ModuleSignature>,

    /// Map from module type names to their definitions
    module_types: HashMap<QualifiedName, ModuleTypeExpression>,

    /// Current module path (for generating qualified names)
    path: Vec<String>,

    /// Modules that have been opened in the current scope
    opened: Vec<QualifiedName>,
}

impl ModuleEnv {
    pub fn new() -> Self {
        ModuleEnv {
            modules: HashMap::new(),
            module_types: HashMap::new(),
            path: Vec::new(),
            opened: Vec::new(),
        }
    }

    /// Enter a module (push onto path)
    pub fn enter_module(&mut self, name: String) {
        self.path.push(name);
    }

    /// Exit current module (pop from path)
    pub fn exit_module(&mut self) {
        self.path.pop();
    }

    /// Get the current qualified path
    pub fn current_path(&self) -> QualifiedName {
        self.path.clone()
    }

    /// Qualify a name with the current path
    pub fn qualify(&self, name: String) -> QualifiedName {
        let mut qualified = self.path.clone();
        qualified.push(name);
        qualified
    }

    /// Register a module binding
    pub fn insert_module(&mut self, name: QualifiedName, sig: ModuleSignature) {
        self.modules.insert(name, sig);
    }

    /// Look up a module's signature
    pub fn lookup_module(&self, name: &QualifiedName) -> Option<&ModuleSignature> {
        self.modules.get(name)
    }

    /// Register a module type binding
    pub fn insert_module_type(&mut self, name: QualifiedName, mt: ModuleTypeExpression) {
        self.module_types.insert(name, mt);
    }

    /// Look up a module type
    pub fn lookup_module_type(&self, name: &QualifiedName) -> Option<&ModuleTypeExpression> {
        self.module_types.get(name)
    }

    /// Open a module (bring its names into scope)
    pub fn open_module(&mut self, name: QualifiedName) {
        self.opened.push(name);
    }

    /// Close the most recently opened module
    pub fn close_opened(&mut self) {
        self.opened.pop();
    }
}

impl Default for ModuleEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Signature of a module - what types, values, and submodules it contains
#[derive(Debug, Clone)]
pub struct ModuleSignature {
    /// Types defined in this module
    pub types: HashMap<String, TypeInfo>,

    /// Values defined in this module
    pub values: HashMap<String, Type>,

    /// Nested modules
    pub modules: HashMap<String, ModuleSignature>,
}

impl ModuleSignature {
    pub fn new() -> Self {
        ModuleSignature {
            types: HashMap::new(),
            values: HashMap::new(),
            modules: HashMap::new(),
        }
    }

    /// Add a type to this signature
    pub fn add_type(&mut self, name: String, info: TypeInfo) {
        self.types.insert(name, info);
    }

    /// Add a value to this signature
    pub fn add_value(&mut self, name: String, ty: Type) {
        self.values.insert(name, ty);
    }

    /// Add a nested module to this signature
    pub fn add_module(&mut self, name: String, sig: ModuleSignature) {
        self.modules.insert(name, sig);
    }
}

impl Default for ModuleSignature {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a type in a module signature
#[derive(Debug, Clone)]
pub enum TypeInfo {
    /// Abstract type with a unique generated name
    Abstract(String),

    /// Type alias
    Alias(Type),
    // Future: Datatype definitions
}
