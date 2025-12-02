//! Module system elaboration
//!
//! This module implements the ML-style module system for Wyn.
//! It handles:
//! - Module definitions and nesting
//! - Module type (signature) checking
//! - Parametric modules (functors)
//! - Name resolution and flattening
//!
//! The elaboration happens after parsing but before type checking.
//! The output is a flattened AST with all modules resolved to qualified names.

mod env;

#[cfg(test)]
mod tests;

use crate::ast::{Declaration, ModuleBind, ModuleExpression, ModuleTypeExpression, Program, Spec};
use crate::error::{CompilerError, Result};
use env::ModuleEnv;

/// Main entry point for module elaboration
pub struct ModuleElaborator {
    env: ModuleEnv,
}

impl ModuleElaborator {
    pub fn new() -> Self {
        let mut elaborator = ModuleElaborator {
            env: ModuleEnv::new(),
        };
        elaborator.load_prelude().ok(); // Ignore errors for now
        elaborator
    }

    /// Load the math prelude to define builtin module types and modules
    fn load_prelude(&mut self) -> Result<()> {
        // Parse the prelude file
        let prelude_source = include_str!("../../../prelude/math.wyn");
        let tokens =
            crate::lexer::tokenize(prelude_source).map_err(crate::error::CompilerError::ParseError)?;
        let mut parser = crate::parser::Parser::new(tokens);
        let prelude_program = parser.parse()?;

        // Process declarations to populate module environment
        for decl in prelude_program.declarations {
            match decl {
                Declaration::ModuleTypeBind(mtb) => {
                    // Register module type in environment
                    self.env.register_module_type(mtb.name.clone(), mtb.definition);
                }
                Declaration::ModuleBind(mb) => {
                    // Register module in environment
                    // The module signature tells us what operations are available
                    self.env.register_builtin_module(mb.name.clone(), mb.signature);
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Elaborate a program, resolving all modules and producing a flattened AST
    pub fn elaborate(&mut self, program: Program) -> Result<Program> {
        let mut flat_decls = Vec::new();

        // First, add builtin module members (e.g., f32_cos, f32_sin, etc.)
        // These come from the builtin registry and are automatically available
        flat_decls.extend(self.create_builtin_module_declarations());

        // Then elaborate user-defined declarations
        for decl in program.declarations {
            flat_decls.extend(self.elaborate_declaration(decl)?);
        }

        Ok(Program {
            declarations: flat_decls,
            library_modules: std::collections::HashMap::new(),
        })
    }

    /// Create synthetic declarations for builtin module members
    /// E.g., f32.cos becomes a Val declaration with mangled name f32_cos
    fn create_builtin_module_declarations(&self) -> Vec<Declaration> {
        // For now, return empty - builtins are handled in the type checker
        // In the future, we could generate proper Val declarations here
        // that would allow builtin modules to be treated as first-class modules
        vec![]
    }

    /// Elaborate a single declaration
    fn elaborate_declaration(&mut self, decl: Declaration) -> Result<Vec<Declaration>> {
        match decl {
            Declaration::ModuleBind(mb) => self.elaborate_module_bind(mb),
            Declaration::ModuleTypeBind(mtb) => {
                // Register the module type in the environment
                self.env.register_module_type(mtb.name.clone(), mtb.definition);
                // Module types are erased after elaboration
                Ok(vec![])
            }
            Declaration::Open(me) => {
                // Open a module, bringing its contents into scope without qualification
                match me {
                    ModuleExpression::Name(name) => {
                        // Look up the module's declarations
                        if let Some(decls) = self.env.get_module_contents(&name) {
                            // Clone and strip the module prefix from each declaration
                            let prefix = format!("{}_", name);
                            let opened_decls: Vec<Declaration> =
                                decls.iter().filter_map(|d| self.strip_module_prefix(d, &prefix)).collect();
                            Ok(opened_decls)
                        } else {
                            Err(CompilerError::ModuleError(format!(
                                "Cannot open module '{}': module not found",
                                name
                            )))
                        }
                    }
                    _ => Err(CompilerError::ModuleError(
                        "Only simple module names can be opened".to_string(),
                    )),
                }
            }
            // All other declarations pass through unchanged for now
            _ => Ok(vec![decl]),
        }
    }

    /// Elaborate a module binding
    fn elaborate_module_bind(&mut self, mb: ModuleBind) -> Result<Vec<Declaration>> {
        // For now, only handle simple (non-parametric) modules
        if !mb.params.is_empty() {
            return Err(CompilerError::ModuleError(
                "Parametric modules not yet implemented".to_string(),
            ));
        }

        // If there's a signature with type parameter substitution (e.g., module i32 : (integral with t = i32))
        // we need to elaborate the signature into actual declarations
        if let Some(sig) = &mb.signature {
            // Check if this is a signature-only module (empty body)
            // This is the case for module instantiations like: module i32 : (integral with t = i32)
            if matches!(mb.body, ModuleExpression::Struct(ref decls) if decls.is_empty()) {
                // This is a module instantiation from a signature
                return self.elaborate_signature_instantiation(&mb.name, sig);
            }
        }

        // Enter the module's namespace
        self.env.enter_module(mb.name.clone());

        // Elaborate the module body
        let body_decls = self.elaborate_module_expr(&mb.body)?;

        // Exit the module's namespace
        self.env.exit_module();

        // If there's a signature, check it and filter
        let decls = if let Some(sig) = &mb.signature {
            self.check_and_filter_signature(body_decls, sig)?
        } else {
            body_decls
        };

        // Store the module contents for later use by `open`
        self.env.store_module_contents(mb.name.clone(), decls.clone());

        Ok(decls)
    }

    /// Elaborate a module instantiation from a signature (e.g., module i32 : (integral with t = i32))
    fn elaborate_signature_instantiation(
        &mut self,
        module_name: &str,
        sig: &ModuleTypeExpression,
    ) -> Result<Vec<Declaration>> {
        // Resolve the signature with any type parameter substitutions
        let resolved_sig = self.resolve_module_type_expr(sig)?;

        // Generate Val declarations from the signature
        let decls = self.generate_declarations_from_signature(module_name, &resolved_sig)?;

        // Store the module contents for later use by `open`
        self.env.store_module_contents(module_name.to_string(), decls.clone());

        Ok(decls)
    }

    /// Resolve a module type expression, handling with clauses and includes
    fn resolve_module_type_expr(&self, expr: &ModuleTypeExpression) -> Result<Vec<Spec>> {
        match expr {
            ModuleTypeExpression::Name(name) => {
                // Look up the module type definition
                let qualname = vec![name.clone()];
                if let Some(mt_def) = self.env.lookup_module_type(&qualname) {
                    self.resolve_module_type_expr(mt_def)
                } else {
                    Err(CompilerError::ModuleError(format!(
                        "Module type '{}' not found",
                        name
                    )))
                }
            }
            ModuleTypeExpression::Signature(specs) => {
                // Expand includes and return all specs
                self.expand_specs(specs)
            }
            ModuleTypeExpression::With(base, type_name, type_params, ty) => {
                // Resolve the base signature
                let mut specs = self.resolve_module_type_expr(base)?;

                // Substitute the type parameter throughout the specs
                self.substitute_type_in_specs(&mut specs, type_name, type_params, ty);

                Ok(specs)
            }
            ModuleTypeExpression::Arrow(..) => Err(CompilerError::ModuleError(
                "Parameterized module types (T -> ...) not yet implemented in type resolution".to_string(),
            )),
            ModuleTypeExpression::FunctorType(..) => Err(CompilerError::ModuleError(
                "Functor types (mod_type -> mod_type) not yet implemented in type resolution".to_string(),
            )),
        }
    }

    /// Expand specs, resolving include directives
    fn expand_specs(&self, specs: &[Spec]) -> Result<Vec<Spec>> {
        let mut result = Vec::new();

        for spec in specs {
            match spec {
                Spec::Include(mt_expr) => {
                    // Recursively expand the included module type
                    let included_specs = self.resolve_module_type_expr(mt_expr)?;
                    result.extend(included_specs);
                }
                _ => {
                    result.push(spec.clone());
                }
            }
        }

        Ok(result)
    }

    /// Substitute a type parameter in all specs
    fn substitute_type_in_specs(
        &self,
        specs: &mut [Spec],
        type_name: &str,
        _type_params: &[crate::ast::TypeParam],
        replacement: &crate::ast::Type,
    ) {
        use crate::ast::Spec;

        for spec in specs.iter_mut() {
            match spec {
                Spec::Sig(_, _, ty) => {
                    *ty = self.substitute_type_in_type(ty, type_name, replacement);
                }
                Spec::SigOp(_, ty) => {
                    *ty = self.substitute_type_in_type(ty, type_name, replacement);
                }
                Spec::Type(_, name, _, def) => {
                    // Don't substitute in type definitions themselves
                    // But do substitute in their definitions
                    if let Some(ty) = def {
                        *ty = self.substitute_type_in_type(ty, type_name, replacement);
                    }
                    // If this is the type being substituted, mark it as concrete
                    if name == type_name {
                        if let Some(ty) = def {
                            *ty = replacement.clone();
                        } else {
                            *def = Some(replacement.clone());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Substitute a type parameter in a type
    fn substitute_type_in_type(
        &self,
        ty: &crate::ast::Type,
        type_name: &str,
        replacement: &crate::ast::Type,
    ) -> crate::ast::Type {
        use crate::ast::{Type, TypeName};

        match ty {
            Type::Constructed(TypeName::Named(name), _args) if name == type_name => {
                // This is a reference to the type parameter, replace it
                replacement.clone()
            }
            Type::Constructed(name, args) => {
                // Recursively substitute in type arguments
                let new_args: Vec<Type> = args
                    .iter()
                    .map(|arg| self.substitute_type_in_type(arg, type_name, replacement))
                    .collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Generate Val declarations from a resolved signature
    fn generate_declarations_from_signature(
        &self,
        module_name: &str,
        specs: &[Spec],
    ) -> Result<Vec<Declaration>> {
        let mut decls = Vec::new();

        for spec in specs {
            match spec {
                Spec::Sig(name, type_params, ty) => {
                    // Generate a Sig declaration with mangled name
                    let mangled_name = format!("{}_{}", module_name, name);
                    decls.push(Declaration::Sig(crate::ast::SigDecl {
                        attributes: vec![],
                        name: mangled_name,
                        size_params: type_params
                            .iter()
                            .filter_map(|tp| match tp {
                                crate::ast::TypeParam::Size(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect(),
                        type_params: type_params
                            .iter()
                            .filter_map(|tp| match tp {
                                crate::ast::TypeParam::Type(t) => Some(t.clone()),
                                _ => None,
                            })
                            .collect(),
                        ty: ty.clone(),
                    }));
                }
                Spec::SigOp(op, ty) => {
                    // Generate a Sig declaration for operator with mangled name
                    let mangled_name = format!("{}_({})", module_name, op);
                    decls.push(Declaration::Sig(crate::ast::SigDecl {
                        attributes: vec![],
                        name: mangled_name,
                        size_params: vec![],
                        type_params: vec![],
                        ty: ty.clone(),
                    }));
                }
                Spec::Type(_, name, type_params, def) => {
                    // Generate a type binding if there's a definition
                    if let Some(ty_def) = def {
                        let mangled_name = format!("{}_{}", module_name, name);
                        decls.push(Declaration::TypeBind(crate::ast::TypeBind {
                            kind: crate::ast::TypeBindKind::Normal,
                            name: mangled_name,
                            type_params: type_params.clone(),
                            definition: ty_def.clone(),
                        }));
                    }
                }
                _ => {
                    // Ignore other specs for now
                }
            }
        }

        Ok(decls)
    }

    /// Elaborate a module expression
    fn elaborate_module_expr(&mut self, expr: &ModuleExpression) -> Result<Vec<Declaration>> {
        match expr {
            ModuleExpression::Struct(decls) => {
                let mut result = Vec::new();
                for decl in decls {
                    let elaborated = self.elaborate_declaration(decl.clone())?;
                    // Qualify the names in each declaration
                    // Note: nested modules are already qualified by their own elaborate_module_bind
                    for d in elaborated {
                        // Don't double-qualify: ModuleBind declarations are already qualified
                        // during their own elaboration
                        if matches!(decl, Declaration::ModuleBind(_)) {
                            result.push(d);
                        } else {
                            result.push(self.qualify_declaration(d));
                        }
                    }
                }
                Ok(result)
            }
            ModuleExpression::Name(_n) => {
                // TODO: look up module and return its declarations
                Err(CompilerError::ModuleError(
                    "Module references not yet implemented".to_string(),
                ))
            }
            ModuleExpression::Application(_, _) => Err(CompilerError::ModuleError(
                "Functor application not yet implemented".to_string(),
            )),
            ModuleExpression::Ascription(mod_expr, sig) => {
                // First elaborate the module expression
                let decls = self.elaborate_module_expr(mod_expr)?;
                // Then filter/check against the signature
                self.check_and_filter_signature(decls, sig)
            }
            ModuleExpression::Lambda(_, _, _) => Err(CompilerError::ModuleError(
                "Module lambdas not yet implemented".to_string(),
            )),
            ModuleExpression::Import(_) => Err(CompilerError::ModuleError(
                "Module imports not yet implemented".to_string(),
            )),
        }
    }

    /// Qualify a declaration's name with the current module path
    fn qualify_declaration(&self, decl: Declaration) -> Declaration {
        let path = self.env.current_path();
        if path.is_empty() {
            // Top-level, no qualification needed
            return decl;
        }

        // Create the qualified name prefix (e.g., M_$_N for module M.N)
        let prefix = path.join("_$_");

        match decl {
            Declaration::Decl(mut d) => {
                d.name = format!("{}_{}", prefix, d.name);
                Declaration::Decl(d)
            }
            Declaration::TypeBind(mut tb) => {
                tb.name = format!("{}_{}", prefix, tb.name);
                Declaration::TypeBind(tb)
            }
            Declaration::Sig(mut v) => {
                v.name = format!("{}_{}", prefix, v.name);
                Declaration::Sig(v)
            }
            Declaration::Entry(mut e) => {
                e.name = format!("{}_{}", prefix, e.name);
                Declaration::Entry(e)
            }
            Declaration::Uniform(mut u) => {
                u.name = format!("{}_{}", prefix, u.name);
                Declaration::Uniform(u)
            }
            // Other declarations don't have names to qualify or are module-level
            _ => decl,
        }
    }

    /// Check that declarations satisfy a signature and filter to only include what's in the signature
    fn check_and_filter_signature(
        &self,
        decls: Vec<Declaration>,
        sig: &ModuleTypeExpression,
    ) -> Result<Vec<Declaration>> {
        // First resolve the module type expression to get expanded specs
        // This handles With constraints, includes, etc.
        let specs = self.resolve_module_type_expr(sig)?;

        {
            let specs = &specs;
            let mut result = Vec::new();

            // For each spec in the signature, find the matching declaration
            for spec in specs {
                match spec {
                    Spec::Sig(name, _type_params, _ty) => {
                        // Find the value declaration with this name
                        let matching_decl = decls.iter().find(|d| match d {
                            Declaration::Decl(decl) => decl.name.ends_with(name),
                            Declaration::Sig(sig) => sig.name.ends_with(name),
                            _ => false,
                        });

                        if let Some(decl) = matching_decl {
                            result.push(decl.clone());
                        } else {
                            return Err(CompilerError::ModuleError(format!(
                                "Module does not provide value '{}' required by signature",
                                name
                            )));
                        }
                    }
                    Spec::Type(_kind, name, _type_params, _def) => {
                        // Find the type declaration with this name
                        let matching_decl = decls.iter().find(|d| match d {
                            Declaration::TypeBind(tb) => tb.name.ends_with(name),
                            _ => false,
                        });

                        if let Some(decl) = matching_decl {
                            result.push(decl.clone());
                        } else {
                            return Err(CompilerError::ModuleError(format!(
                                "Module does not provide type '{}' required by signature",
                                name
                            )));
                        }
                    }
                    Spec::Module(_name, _sig) => {
                        // TODO: Handle nested modules
                        return Err(CompilerError::ModuleError(
                            "Nested modules in signatures not yet implemented".to_string(),
                        ));
                    }
                    Spec::Include(_) => {
                        // TODO: Handle includes
                        return Err(CompilerError::ModuleError(
                            "Include in signatures not yet implemented".to_string(),
                        ));
                    }
                    Spec::SigOp(op, _ty) => {
                        // Operator specs like sig (+): t -> t -> t
                        // The definition will have name "(+)" for operator "+"
                        let op_name = format!("({})", op);
                        let matching_decl = decls.iter().find(|d| match d {
                            Declaration::Decl(decl) => decl.name.ends_with(&op_name),
                            _ => false,
                        });

                        if let Some(decl) = matching_decl {
                            result.push(decl.clone());
                        } else {
                            return Err(CompilerError::ModuleError(format!(
                                "Module does not provide operator '{}' required by signature",
                                op
                            )));
                        }
                    }
                }
            }

            Ok(result)
        }
    }

    /// Strip the module prefix from a declaration's name
    /// Returns None if the declaration doesn't have the expected prefix
    fn strip_module_prefix(&self, decl: &Declaration, prefix: &str) -> Option<Declaration> {
        match decl {
            Declaration::Decl(d) => {
                if d.name.starts_with(prefix) {
                    let mut new_decl = d.clone();
                    new_decl.name = d.name[prefix.len()..].to_string();
                    Some(Declaration::Decl(new_decl))
                } else {
                    None
                }
            }
            Declaration::Sig(v) => {
                if v.name.starts_with(prefix) {
                    let mut new_sig = v.clone();
                    new_sig.name = v.name[prefix.len()..].to_string();
                    Some(Declaration::Sig(new_sig))
                } else {
                    None
                }
            }
            Declaration::TypeBind(tb) => {
                if tb.name.starts_with(prefix) {
                    let mut new_tb = tb.clone();
                    new_tb.name = tb.name[prefix.len()..].to_string();
                    Some(Declaration::TypeBind(new_tb))
                } else {
                    None
                }
            }
            // Other declarations don't get opened
            _ => None,
        }
    }
}

impl Default for ModuleElaborator {
    fn default() -> Self {
        Self::new()
    }
}
