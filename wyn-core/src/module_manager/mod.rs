//! Module manager for lazy loading and caching module definitions

use crate::ast::{Declaration, ModuleTypeExpression, NodeCounter, Program, Spec, Type, TypeParam};
use crate::error::{CompilerError, Result};
use crate::lexer;
use crate::parser::Parser;
use std::collections::{HashMap, HashSet};

/// Represents a fully inflated module with all includes expanded and type substitutions applied
#[derive(Debug, Clone)]
pub struct InflatedModule {
    pub name: String,
    /// Fully expanded, flattened list of specs with type substitutions applied
    pub specs: Vec<Spec>,
}

/// Manages lazy loading of module files
pub struct ModuleManager {
    /// Cached parsed modules: file_path -> Program
    cached_modules: HashMap<String, Program>,
    /// Module type registry: type name -> ModuleTypeExpression
    module_type_registry: HashMap<String, ModuleTypeExpression>,
    /// Inflated modules: module_name -> InflatedModule
    inflated_modules: HashMap<String, InflatedModule>,
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
            module_type_registry: HashMap::new(),
            inflated_modules: HashMap::new(),
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
            module_type_registry: HashMap::new(),
            inflated_modules: HashMap::new(),
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

        // Register module types first
        self.register_module_types(&program)?;

        // Inflate all modules from the program
        self.inflate_all_modules(&program)?;

        // Cache the parsed program
        self.cached_modules.insert(file_path.to_string(), program);
        Ok(())
    }

    /// Inflate all module bindings from a parsed program
    fn inflate_all_modules(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            if let Declaration::ModuleBind(mb) = decl {
                // Inflate the module signature if it exists
                if let Some(signature) = &mb.signature {
                    let specs = self.inflate_module_type(signature, &HashMap::new())?;
                    let inflated = InflatedModule {
                        name: mb.name.clone(),
                        specs,
                    };

                    if self.inflated_modules.contains_key(&mb.name) {
                        return Err(CompilerError::ModuleError(format!(
                            "Module '{}' is already defined",
                            mb.name
                        )));
                    }

                    self.inflated_modules.insert(mb.name.clone(), inflated);
                }
            }
        }
        Ok(())
    }

    /// Register all module type definitions from a parsed program
    fn register_module_types(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            if let Declaration::ModuleTypeBind(mtb) = decl {
                if self.module_type_registry.contains_key(&mtb.name) {
                    return Err(CompilerError::ModuleError(format!(
                        "Module type '{}' is already defined",
                        mtb.name
                    )));
                }
                self.module_type_registry
                    .insert(mtb.name.clone(), mtb.definition.clone());
            }
        }
        Ok(())
    }

    /// Inflate a module type expression into a flat list of specs
    /// Recursively expands includes and applies type substitutions
    fn inflate_module_type(
        &self,
        mte: &ModuleTypeExpression,
        substitutions: &HashMap<String, Type>,
    ) -> Result<Vec<Spec>> {
        match mte {
            ModuleTypeExpression::Name(name) => {
                // Look up the module type in the registry
                let definition = self.module_type_registry.get(name).ok_or_else(|| {
                    CompilerError::ModuleError(format!("Module type '{}' not found", name))
                })?;
                // Recurse on the definition
                self.inflate_module_type(definition, substitutions)
            }

            ModuleTypeExpression::Signature(specs) => {
                // Process each spec, expanding includes and applying substitutions
                let mut result = Vec::new();
                for spec in specs {
                    match spec {
                        Spec::Include(inner_mte) => {
                            // Recursively inflate the included module type
                            let included_specs = self.inflate_module_type(inner_mte, substitutions)?;
                            result.extend(included_specs);
                        }
                        _ => {
                            // Apply type substitutions to the spec and add it
                            let substituted_spec = self.substitute_in_spec(spec, substitutions);
                            result.push(substituted_spec);
                        }
                    }
                }
                Ok(result)
            }

            ModuleTypeExpression::With(inner, type_name, _type_params, type_value) => {
                // Add the type substitution and recurse on the inner expression
                let mut new_substitutions = substitutions.clone();
                new_substitutions.insert(type_name.clone(), type_value.clone());
                self.inflate_module_type(inner, &new_substitutions)
            }

            ModuleTypeExpression::Arrow(_, _, _) | ModuleTypeExpression::FunctorType(_, _) => {
                // Functor types not yet supported
                Err(CompilerError::ModuleError(
                    "Functor types are not yet supported".to_string(),
                ))
            }
        }
    }

    /// Apply type substitutions to a spec
    fn substitute_in_spec(&self, spec: &Spec, substitutions: &HashMap<String, Type>) -> Spec {
        match spec {
            Spec::Sig(name, type_params, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::Sig(name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::SigOp(op, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::SigOp(op.clone(), substituted_ty)
            }
            Spec::Type(kind, name, type_params, maybe_ty) => {
                let substituted_ty = maybe_ty
                    .as_ref()
                    .map(|ty| self.substitute_in_type(ty, substitutions));
                Spec::Type(kind.clone(), name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::Module(name, mte) => {
                // Don't substitute in nested module signatures for now
                Spec::Module(name.clone(), mte.clone())
            }
            Spec::Include(_) => {
                // Includes should have been expanded by now
                spec.clone()
            }
        }
    }

    /// Apply type substitutions to a type
    fn substitute_in_type(&self, ty: &Type, substitutions: &HashMap<String, Type>) -> Type {
        use crate::ast::TypeName;

        match ty {
            Type::Constructed(name, args) => {
                // Check if this is a named type that should be substituted
                if let TypeName::Named(type_name) = name {
                    if args.is_empty() {
                        if let Some(replacement) = substitutions.get(type_name) {
                            return replacement.clone();
                        }
                    }
                }

                // Recursively substitute in type arguments
                let new_args: Vec<Type> = args
                    .iter()
                    .map(|arg| self.substitute_in_type(arg, substitutions))
                    .collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Query the type of a function in a specific module
    /// e.g., get_module_function_type("f32", "sin") -> Type
    pub fn get_module_function_type(&self, module_name: &str, function_name: &str) -> Result<Type> {
        // Look up the inflated module
        let inflated = self.inflated_modules.get(module_name).ok_or_else(|| {
            CompilerError::ModuleError(format!("Module '{}' not found", module_name))
        })?;

        // Search for the function in the inflated specs
        for spec in &inflated.specs {
            match spec {
                Spec::Sig(name, _type_params, ty) if name == function_name => {
                    return Ok(ty.clone());
                }
                Spec::SigOp(op, ty) if op == function_name => {
                    return Ok(ty.clone());
                }
                _ => {}
            }
        }

        Err(CompilerError::ModuleError(format!(
            "Function '{}' not found in module '{}'",
            function_name, module_name
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
