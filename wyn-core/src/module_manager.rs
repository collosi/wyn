//! Module manager for lazy loading and caching module definitions

use crate::ast::{Declaration, NodeCounter, Program};
use crate::error::{CompilerError, Result};
use crate::lexer;
use crate::parser::Parser;
use std::collections::{HashMap, HashSet};

/// Manages lazy loading of module files
pub struct ModuleManager {
    /// Cached parsed modules: module_name -> Program
    cached_modules: HashMap<String, Program>,
    /// Set of known module names (for name resolution)
    known_modules: HashSet<String>,
    /// Shared node counter for unique NodeIds across all modules
    node_counter: NodeCounter,
}

impl ModuleManager {
    /// Create a new module manager with a fresh NodeCounter
    pub fn new() -> Self {
        let known_modules = [
            "f32", "f64", "f16", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "bool",
            "graphics32", "graphics64",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        ModuleManager {
            cached_modules: HashMap::new(),
            known_modules,
            node_counter: NodeCounter::new(),
        }
    }

    /// Create a new module manager with a shared NodeCounter
    /// This ensures NodeIds don't collide with user code that was already parsed
    pub fn new_with_counter(node_counter: NodeCounter) -> Self {
        let known_modules = [
            "f32", "f64", "f16", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "bool",
            "graphics32", "graphics64",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        ModuleManager {
            cached_modules: HashMap::new(),
            known_modules,
            node_counter,
        }
    }

    /// Check if a name is a known module
    pub fn is_known_module(&self, name: &str) -> bool {
        self.known_modules.contains(name)
    }

    /// Get the source code for a prelude module
    fn get_prelude_source(module_name: &str) -> Option<&'static str> {
        match module_name {
            "math" => Some(include_str!("../../prelude/math.wyn")),
            "graphics32" => Some(include_str!("../../prelude/graphics32.wyn")),
            "graphics64" => Some(include_str!("../../prelude/graphics64.wyn")),
            _ => None,
        }
    }

    /// Load a module by name (e.g., "f32" loads the embedded f32 prelude)
    /// Returns the parsed program with definitions prefixed by module name
    pub fn load_module(&mut self, module_name: &str) -> Result<&Program> {
        // Check cache first
        if self.cached_modules.contains_key(module_name) {
            return Ok(self.cached_modules.get(module_name).unwrap());
        }

        // Get embedded source
        let source = Self::get_prelude_source(module_name)
            .ok_or_else(|| CompilerError::ModuleError(format!("Unknown module '{}'", module_name)))?;

        // Parse module using the shared node counter to avoid NodeId collisions
        let tokens = lexer::tokenize(source).map_err(CompilerError::ParseError)?;
        // Take ownership of node_counter, parse, then put it back
        let counter = std::mem::take(&mut self.node_counter);
        let mut parser = Parser::new_with_counter(tokens, counter);
        let mut program = parser.parse()?;
        self.node_counter = parser.take_node_counter();

        // Prefix all declaration names with module name
        for decl in &mut program.declarations {
            match decl {
                Declaration::Decl(d) => {
                    let prefixed_name = format!("{}.{}", module_name, d.name);
                    d.name = prefixed_name;
                }
                Declaration::Entry(entry_def) => {
                    let prefixed_name = format!("{}.{}", module_name, entry_def.name);
                    entry_def.name = prefixed_name;
                }
                _ => {} // Other declarations don't need prefixing
            }
        }

        // Cache and return
        self.cached_modules.insert(module_name.to_string(), program);
        Ok(self.cached_modules.get(module_name).unwrap())
    }

    /// Check if a name is a qualified module reference (e.g., "f32.sum")
    pub fn is_qualified_name(name: &str) -> bool {
        name.contains('.')
    }

    /// Split a qualified name into (module, function) parts
    /// E.g., "f32.sum" -> Some(("f32", "sum"))
    pub fn split_qualified_name(name: &str) -> Option<(&str, &str)> {
        let parts: Vec<&str> = name.splitn(2, '.').collect();
        if parts.len() == 2 { Some((parts[0], parts[1])) } else { None }
    }

    /// Get all loaded module declarations (for inlining into the main program)
    pub fn get_all_loaded_declarations(&self) -> Vec<Declaration> {
        let mut all_decls = Vec::new();
        for program in self.cached_modules.values() {
            all_decls.extend(program.declarations.clone());
        }
        all_decls
    }
}

impl Default for ModuleManager {
    fn default() -> Self {
        Self::new()
    }
}
