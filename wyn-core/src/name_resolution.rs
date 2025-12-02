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
use std::collections::{HashMap, HashSet};

pub struct NameResolver {
    module_manager: ModuleManager,
    impl_source: crate::impl_source::ImplSource,
    referenced_modules: HashSet<String>,
}

impl NameResolver {
    pub fn new() -> Self {
        NameResolver {
            module_manager: ModuleManager::new(),
            impl_source: crate::impl_source::ImplSource::default(),
            referenced_modules: HashSet::new(),
        }
    }

    /// Create a new NameResolver with a shared NodeCounter
    /// This ensures modules are parsed with NodeIds that don't collide with user code
    pub fn new_with_counter(node_counter: NodeCounter) -> Self {
        NameResolver {
            module_manager: ModuleManager::new_with_counter(node_counter),
            impl_source: crate::impl_source::ImplSource::default(),
            referenced_modules: HashSet::new(),
        }
    }

    /// Resolve names in a program and merge in referenced modules
    /// Uses two-phase loading:
    /// 1. Collect all referenced modules recursively (modules can reference other modules)
    /// 2. Resolve names in user code and all loaded modules
    /// 3. Merge resolved modules into program
    pub fn resolve_program(&mut self, program: &mut Program) -> Result<()> {
        // Phase 1: Collect all referenced modules recursively
        let mut modules_to_load = HashSet::new();
        let mut loaded_modules = HashSet::new();

        // Find modules referenced by user code
        self.collect_module_references_from_program(program, &mut modules_to_load)?;

        // Load modules and find transitive references
        while !modules_to_load.is_empty() {
            let module_names: Vec<String> = modules_to_load.drain().collect();

            for module_name in module_names {
                if loaded_modules.contains(&module_name) {
                    continue; // Already loaded
                }

                // Load the module
                let module_prog = self.module_manager.load_module(&module_name)?;
                loaded_modules.insert(module_name.clone());

                // Find modules referenced by this module (clone to avoid borrow issues)
                let module_prog_clone = module_prog.clone();
                self.collect_module_references_from_program(&module_prog_clone, &mut modules_to_load)?;
            }
        }

        // Phase 2: Resolve names in user code
        for decl in &mut program.declarations {
            self.resolve_declaration(decl)?;
        }

        // Resolve names in all loaded modules and store locally
        let mut resolved_modules: HashMap<String, Program> = HashMap::new();
        for module_name in &loaded_modules {
            let module_prog = self.module_manager.load_module(module_name)?;
            let mut mutable_prog = module_prog.clone();
            for decl in &mut mutable_prog.declarations {
                self.resolve_declaration(decl)?;
            }
            resolved_modules.insert(module_name.clone(), mutable_prog);
        }

        // Phase 3: Store all resolved modules in program, preserving module origins
        for (module_name, resolved_prog) in resolved_modules {
            program.library_modules.insert(module_name, resolved_prog.declarations);
        }

        Ok(())
    }

    /// Collect module references from a program without resolving names
    /// Just finds FieldAccess(Identifier(module), _) where module is a known module
    fn collect_module_references_from_program(&self, program: &Program, modules: &mut HashSet<String>) -> Result<()> {
        for decl in &program.declarations {
            self.collect_module_references_from_decl(decl, modules)?;
        }
        Ok(())
    }

    fn collect_module_references_from_decl(&self, decl: &Declaration, modules: &mut HashSet<String>) -> Result<()> {
        match decl {
            Declaration::Decl(d) => {
                self.collect_module_references_from_expr(&d.body, modules)?;
            }
            Declaration::Entry(entry) => {
                self.collect_module_references_from_expr(&entry.body, modules)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn collect_module_references_from_expr(&self, expr: &Expression, modules: &mut HashSet<String>) -> Result<()> {
        match &expr.kind {
            ExprKind::FieldAccess(obj, field) => {
                if let ExprKind::Identifier(name) = &obj.kind {
                    if self.module_manager.is_known_module(name) {
                        let full_name = format!("{}.{}", name, field);
                        // Only add if NOT a builtin
                        if !self.impl_source.is_builtin(&full_name) {
                            modules.insert(name.clone());
                        }
                    }
                }
                self.collect_module_references_from_expr(obj, modules)?;
            }
            ExprKind::Application(func, args) => {
                self.collect_module_references_from_expr(func, modules)?;
                for arg in args {
                    self.collect_module_references_from_expr(arg, modules)?;
                }
            }
            ExprKind::Lambda(lambda) => {
                self.collect_module_references_from_expr(&lambda.body, modules)?;
            }
            ExprKind::LetIn(let_in) => {
                self.collect_module_references_from_expr(&let_in.value, modules)?;
                self.collect_module_references_from_expr(&let_in.body, modules)?;
            }
            ExprKind::If(if_expr) => {
                self.collect_module_references_from_expr(&if_expr.condition, modules)?;
                self.collect_module_references_from_expr(&if_expr.then_branch, modules)?;
                self.collect_module_references_from_expr(&if_expr.else_branch, modules)?;
            }
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.collect_module_references_from_expr(lhs, modules)?;
                self.collect_module_references_from_expr(rhs, modules)?;
            }
            ExprKind::UnaryOp(_, operand) => {
                self.collect_module_references_from_expr(operand, modules)?;
            }
            ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
                for e in exprs {
                    self.collect_module_references_from_expr(e, modules)?;
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.collect_module_references_from_expr(arr, modules)?;
                self.collect_module_references_from_expr(idx, modules)?;
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, e) in fields {
                    self.collect_module_references_from_expr(e, modules)?;
                }
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(ref init) = loop_expr.init {
                    self.collect_module_references_from_expr(init, modules)?;
                }
                match &loop_expr.form {
                    crate::ast::LoopForm::While(cond) => {
                        self.collect_module_references_from_expr(cond, modules)?;
                    }
                    crate::ast::LoopForm::For(_, bound) => {
                        self.collect_module_references_from_expr(bound, modules)?;
                    }
                    crate::ast::LoopForm::ForIn(_, iter) => {
                        self.collect_module_references_from_expr(iter, modules)?;
                    }
                }
                self.collect_module_references_from_expr(&loop_expr.body, modules)?;
            }
            ExprKind::Match(match_expr) => {
                self.collect_module_references_from_expr(&match_expr.scrutinee, modules)?;
                for case in &match_expr.cases {
                    self.collect_module_references_from_expr(&case.body, modules)?;
                }
            }
            ExprKind::Pipe(lhs, rhs) => {
                self.collect_module_references_from_expr(lhs, modules)?;
                self.collect_module_references_from_expr(rhs, modules)?;
            }
            ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
                self.collect_module_references_from_expr(e, modules)?;
            }
            ExprKind::Assert(cond, body) => {
                self.collect_module_references_from_expr(cond, modules)?;
                self.collect_module_references_from_expr(body, modules)?;
            }
            ExprKind::Range(range) => {
                self.collect_module_references_from_expr(&range.start, modules)?;
                self.collect_module_references_from_expr(&range.end, modules)?;
                if let Some(ref step) = range.step {
                    self.collect_module_references_from_expr(step, modules)?;
                }
            }
            ExprKind::QualifiedName(module_path, _) => {
                // Handle already-qualified names (e.g., from parser)
                if !module_path.is_empty() {
                    let module_name = &module_path[0];
                    if self.module_manager.is_known_module(module_name) {
                        modules.insert(module_name.clone());
                    }
                }
            }
            _ => {}
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
            Declaration::Sig(_) => {
                // SigDecl has no body, only a type signature
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
                        if !self.impl_source.is_builtin(&full_name) {
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
            ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
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
