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

    /// Load a prelude file (e.g., "math.wyn") which may contain multiple module definitions
    /// Returns the parsed program
    pub fn load_file(&mut self, file_path: &str) -> Result<()> {
        // For now, we only support embedded prelude files
        let source = match file_path {
            "math.wyn" => include_str!("../../../prelude/math.wyn"),
            "graphics32.wyn" => include_str!("../../../prelude/graphics32.wyn"),
            "graphics64.wyn" => include_str!("../../../prelude/graphics64.wyn"),
            _ => return Err(CompilerError::ModuleError(format!("Unknown prelude file: {}", file_path))),
        };

        // Parse the file
        let tokens = lexer::tokenize(source).map_err(CompilerError::ParseError)?;
        let counter = std::mem::take(&mut self.node_counter);
        let mut parser = Parser::new_with_counter(tokens, counter);
        let program = parser.parse()?;
        self.node_counter = parser.take_node_counter();

        // TODO: Extract individual modules from the parsed program
        // For now, just cache the whole file under the file_path key
        self.cached_modules.insert(file_path.to_string(), program);
        Ok(())
    }

    /// Query the type of a function in a specific module
    /// e.g., get_module_function_type("f32", "sin") -> Type
    pub fn get_module_function_type(&self, module_name: &str, function_name: &str) -> Result<crate::ast::Type> {
        // TODO: Implement proper module extraction and lookup
        // For now, search through all cached programs
        for program in self.cached_modules.values() {
            for decl in &program.declarations {
                match decl {
                    Declaration::ModuleBind(mb) if mb.name == module_name => {
                        // Found the module, now look for the function in its body
                        // TODO: Navigate the module structure to find the function
                        return Err(CompilerError::ModuleError(
                            "Module extraction not yet implemented".to_string()
                        ));
                    }
                    _ => {}
                }
            }
        }
        Err(CompilerError::ModuleError(format!(
            "Module '{}' or function '{}' not found",
            module_name, function_name
        )))
    }

    /// Load a module by name (e.g., "f32" loads the f32 module from math.wyn)
    /// Returns the parsed program with definitions prefixed by module name
    pub fn load_module(&mut self, _module_name: &str) -> Result<&Program> {
        todo!("Redesign module system to support multiple modules per file")
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

#[cfg(test)]
mod tests;
