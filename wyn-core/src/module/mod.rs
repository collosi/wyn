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

use crate::ast::{
    Declaration, ExprKind, Expression, ModuleBind, ModuleExpression, ModuleTypeExpression, Program, Span,
    Spec,
};
use crate::builtin_registry::BuiltinRegistry;
use crate::error::{CompilerError, Result};
use env::{ModuleEnv, ModuleSignature};

/// Main entry point for module elaboration
pub struct ModuleElaborator {
    env: ModuleEnv,
    builtin_registry: BuiltinRegistry,
}

impl ModuleElaborator {
    pub fn new() -> Self {
        let mut elaborator = ModuleElaborator {
            env: ModuleEnv::new(),
            builtin_registry: BuiltinRegistry::new(
                &mut polytype::Context::<crate::ast::TypeName>::default(),
            ),
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
            Declaration::ModuleTypeBind(_mtb) => {
                // Module types are erased after elaboration
                Ok(vec![])
            }
            Declaration::Open(_me) => {
                // TODO: implement open
                Ok(vec![])
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

        // Enter the module's namespace
        self.env.enter_module(mb.name.clone());

        // Elaborate the module body
        let body_decls = self.elaborate_module_expr(&mb.body)?;

        // Exit the module's namespace
        self.env.exit_module();

        // If there's a signature, check it and filter
        if let Some(sig) = &mb.signature {
            self.check_and_filter_signature(body_decls, sig)
        } else {
            Ok(body_decls)
        }
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
            Declaration::Val(mut v) => {
                v.name = format!("{}_{}", prefix, v.name);
                Declaration::Val(v)
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
        match sig {
            ModuleTypeExpression::Signature(specs) => {
                let mut result = Vec::new();

                // For each spec in the signature, find the matching declaration
                for spec in specs {
                    match spec {
                        Spec::Val(name, _type_params, _ty) => {
                            // Find the value declaration with this name
                            let matching_decl = decls.iter().find(|d| match d {
                                Declaration::Decl(decl) => decl.name.ends_with(name),
                                Declaration::Val(val) => val.name.ends_with(name),
                                _ => false,
                            });

                            if let Some(decl) = matching_decl {
                                result.push(decl.clone());
                                // TODO: Check that the type matches
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
                                // TODO: Check constraints and make abstract if needed
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
                        Spec::ValOp(_op, _ty) => {
                            // TODO: Handle operator specs
                            return Err(CompilerError::ModuleError(
                                "Operator specs not yet implemented".to_string(),
                            ));
                        }
                    }
                }

                Ok(result)
            }
            ModuleTypeExpression::Name(_) => {
                // TODO: Look up named module type and use it
                Err(CompilerError::ModuleError(
                    "Named module types not yet implemented".to_string(),
                ))
            }
            _ => Err(CompilerError::ModuleError(
                "Complex module type expressions not yet implemented".to_string(),
            )),
        }
    }
}

impl Default for ModuleElaborator {
    fn default() -> Self {
        Self::new()
    }
}
