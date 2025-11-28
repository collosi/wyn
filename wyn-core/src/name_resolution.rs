//! Name resolution pass
//!
//! Resolves module-qualified names by:
//! 1. Rewriting FieldAccess(Identifier(module), field) -> QualifiedName([module], field)
//!    when `module` is a known module name
//! 2. Loading referenced modules
//! 3. Merging module declarations into the program

use crate::ast::{Declaration, ExprKind, Expression, NodeCounter, Program};
use crate::error::Result;
use crate::module_manager::ModuleManager;
use std::collections::HashSet;

pub struct NameResolver {
    module_manager: ModuleManager,
    builtin_registry: crate::builtin_registry::BuiltinRegistry,
    referenced_modules: HashSet<String>,
}

impl NameResolver {
    pub fn new() -> Self {
        NameResolver {
            module_manager: ModuleManager::new(),
            builtin_registry: crate::builtin_registry::BuiltinRegistry::default(),
            referenced_modules: HashSet::new(),
        }
    }

    /// Create a new NameResolver with a shared NodeCounter
    /// This ensures modules are parsed with NodeIds that don't collide with user code
    pub fn new_with_counter(node_counter: NodeCounter) -> Self {
        NameResolver {
            module_manager: ModuleManager::new_with_counter(node_counter),
            builtin_registry: crate::builtin_registry::BuiltinRegistry::default(),
            referenced_modules: HashSet::new(),
        }
    }

    /// Resolve names in a program and merge in referenced modules
    pub fn resolve_program(&mut self, program: &mut Program) -> Result<()> {
        // First pass: rewrite FieldAccess -> QualifiedName and collect referenced modules
        for decl in &mut program.declarations {
            self.resolve_declaration(decl)?;
        }

        // Second pass: load all referenced modules
        for module_name in &self.referenced_modules {
            let module_prog = self.module_manager.load_module(module_name)?;
            program.library_declarations.extend(module_prog.declarations.clone());
        }

        Ok(())
    }

    fn resolve_declaration(&mut self, decl: &mut Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(d) => {
                self.resolve_expr(&mut d.body)?;
            }
            Declaration::Entry(entry) => {
                self.resolve_expr(&mut entry.body)?;
            }
            Declaration::Val(_) => {
                // ValDecl has no body, only a type signature
            }
            _ => {}
        }
        Ok(())
    }

    fn resolve_expr(&mut self, expr: &mut Expression) -> Result<()> {
        match &mut expr.kind {
            ExprKind::FieldAccess(obj, field) => {
                // Check if this is module.name pattern
                if let ExprKind::Identifier(name) = &obj.kind {
                    if self.module_manager.is_known_module(name) {
                        // Build the qualified name
                        let module = name.clone();
                        let func_name = field.clone();
                        let full_name = format!("{}.{}", module, func_name);

                        // Rewrite to QualifiedName
                        expr.kind = ExprKind::QualifiedName(vec![module.clone()], func_name);

                        // Only mark module as referenced if this is NOT a builtin
                        // (builtins like f32.sqrt shouldn't trigger module loading)
                        if !self.builtin_registry.is_builtin(&full_name) {
                            self.referenced_modules.insert(module);
                        }
                        return Ok(());
                    }
                }
                // Otherwise, it's a real field access - recurse into object
                self.resolve_expr(obj)?;
            }
            ExprKind::Application(func, args) => {
                self.resolve_expr(func)?;
                for arg in args {
                    self.resolve_expr(arg)?;
                }
            }
            ExprKind::Lambda(lambda) => {
                self.resolve_expr(&mut lambda.body)?;
            }
            ExprKind::LetIn(let_in) => {
                self.resolve_expr(&mut let_in.value)?;
                self.resolve_expr(&mut let_in.body)?;
            }
            ExprKind::If(if_expr) => {
                self.resolve_expr(&mut if_expr.condition)?;
                self.resolve_expr(&mut if_expr.then_branch)?;
                self.resolve_expr(&mut if_expr.else_branch)?;
            }
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.resolve_expr(lhs)?;
                self.resolve_expr(rhs)?;
            }
            ExprKind::UnaryOp(_, operand) => {
                self.resolve_expr(operand)?;
            }
            ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) => {
                for e in exprs {
                    self.resolve_expr(e)?;
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.resolve_expr(arr)?;
                self.resolve_expr(idx)?;
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, e) in fields {
                    self.resolve_expr(e)?;
                }
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(ref mut init) = loop_expr.init {
                    self.resolve_expr(init)?;
                }
                match &mut loop_expr.form {
                    crate::ast::LoopForm::While(cond) => {
                        self.resolve_expr(cond)?;
                    }
                    crate::ast::LoopForm::For(_, bound) => {
                        self.resolve_expr(bound)?;
                    }
                    crate::ast::LoopForm::ForIn(_, iter) => {
                        self.resolve_expr(iter)?;
                    }
                }
                self.resolve_expr(&mut loop_expr.body)?;
            }
            ExprKind::Match(match_expr) => {
                self.resolve_expr(&mut match_expr.scrutinee)?;
                for case in &mut match_expr.cases {
                    self.resolve_expr(&mut case.body)?;
                }
            }
            ExprKind::Pipe(lhs, rhs) => {
                self.resolve_expr(lhs)?;
                self.resolve_expr(rhs)?;
            }
            ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
                self.resolve_expr(e)?;
            }
            ExprKind::Assert(cond, body) => {
                self.resolve_expr(cond)?;
                self.resolve_expr(body)?;
            }
            ExprKind::Range(range) => {
                self.resolve_expr(&mut range.start)?;
                self.resolve_expr(&mut range.end)?;
                if let Some(ref mut step) = range.step {
                    self.resolve_expr(step)?;
                }
            }
            // Base cases - no sub-expressions to resolve
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::Identifier(_)
            | ExprKind::OperatorSection(_)
            | ExprKind::QualifiedName(_, _)
            | ExprKind::TypeHole => {}
        }
        Ok(())
    }
}

impl Default for NameResolver {
    fn default() -> Self {
        Self::new()
    }
}
