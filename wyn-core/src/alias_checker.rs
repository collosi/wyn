//! Alias checker for tracking backing stores and detecting use-after-move errors.
//!
//! This module implements a visitor-based approach that tracks "backing stores" -
//! the underlying memory that variables reference. When a variable is consumed
//! (passed to a function with a `*T` parameter), all variables referencing the
//! same backing store become invalid.

use crate::ast::*;
use crate::error::Result;
use crate::visitor::{self, Visitor};
use crate::{NodeId, TypeTable};
use polytype::TypeScheme;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

/// Unique identifier for a backing store (the actual memory/array)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackingStoreId(u32);

/// State of a backing store
#[derive(Debug, Clone, PartialEq)]
pub enum StoreState {
    Live,
    Consumed {
        at: NodeId,
        var_name: String,
    },
}

/// What an expression evaluates to in terms of aliasing
#[derive(Debug, Clone, Default)]
pub struct AliasInfo {
    /// The backing stores this expression references
    pub stores: HashSet<BackingStoreId>,
}

impl AliasInfo {
    pub fn copy() -> Self {
        Self {
            stores: HashSet::new(),
        }
    }

    pub fn fresh(id: BackingStoreId) -> Self {
        let mut stores = HashSet::new();
        stores.insert(id);
        Self { stores }
    }

    pub fn references(stores: HashSet<BackingStoreId>) -> Self {
        Self { stores }
    }

    pub fn is_copy(&self) -> bool {
        self.stores.is_empty()
    }

    /// Merge two AliasInfos (for if/else branches, function returns)
    pub fn union(&self, other: &AliasInfo) -> AliasInfo {
        let mut stores = self.stores.clone();
        stores.extend(other.stores.iter().cloned());
        AliasInfo { stores }
    }
}

/// An alias-related error
#[derive(Debug, Clone)]
pub struct AliasError {
    pub kind: AliasErrorKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum AliasErrorKind {
    UseAfterMove {
        variable: String,
        consumed_var: String,
        consumed_at: NodeId,
    },
}

/// The alias checker that walks the AST using the visitor pattern
pub struct AliasChecker<'a> {
    type_table: &'a TypeTable,
    /// All backing stores and their states
    stores: HashMap<BackingStoreId, StoreState>,
    /// Stack of scopes, each mapping variable names to their backing stores
    scopes: Vec<HashMap<String, HashSet<BackingStoreId>>>,
    /// Counter for generating unique store IDs
    next_store_id: u32,
    /// Computed AliasInfo for each expression node
    results: HashMap<NodeId, AliasInfo>,
    /// Collected errors
    errors: Vec<AliasError>,
}

impl<'a> AliasChecker<'a> {
    pub fn new(type_table: &'a TypeTable) -> Self {
        Self {
            type_table,
            stores: HashMap::new(),
            scopes: vec![HashMap::new()],
            next_store_id: 0,
            results: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Create a new backing store and return its ID
    fn new_store(&mut self) -> BackingStoreId {
        let id = BackingStoreId(self.next_store_id);
        self.next_store_id += 1;
        self.stores.insert(id, StoreState::Live);
        id
    }

    /// Push a new scope
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the current scope
    fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Bind a variable to backing stores in the current scope
    fn bind_variable(&mut self, name: &str, info: &AliasInfo) {
        if !info.stores.is_empty() {
            if let Some(scope) = self.scopes.last_mut() {
                scope.insert(name.to_string(), info.stores.clone());
            }
        }
    }

    /// Look up a variable's backing stores
    fn lookup_variable(&self, name: &str) -> Option<HashSet<BackingStoreId>> {
        for scope in self.scopes.iter().rev() {
            if let Some(stores) = scope.get(name) {
                return Some(stores.clone());
            }
        }
        None
    }

    /// Check if any of the given stores have been consumed
    fn check_stores_live(&self, stores: &HashSet<BackingStoreId>) -> Option<(&str, NodeId)> {
        for store_id in stores {
            if let Some(StoreState::Consumed { at, var_name }) = self.stores.get(store_id) {
                return Some((var_name.as_str(), *at));
            }
        }
        None
    }

    /// Consume all given backing stores
    fn consume_stores(&mut self, stores: &HashSet<BackingStoreId>, at: NodeId, var_name: &str) {
        for store_id in stores {
            self.stores.insert(
                *store_id,
                StoreState::Consumed {
                    at,
                    var_name: var_name.to_string(),
                },
            );
        }
    }

    /// Store the result for a node
    fn set_result(&mut self, id: NodeId, info: AliasInfo) {
        self.results.insert(id, info);
    }

    /// Get the result for a node (defaults to Copy if not found)
    fn get_result(&self, id: NodeId) -> AliasInfo {
        self.results.get(&id).cloned().unwrap_or_default()
    }

    /// Check a program for alias errors
    pub fn check_program(mut self, program: &Program) -> Result<AliasCheckResult> {
        for decl in &program.declarations {
            match decl {
                Declaration::Decl(d) => self.check_decl(d),
                Declaration::Entry(e) => self.check_entry(e),
                _ => {} // Skip other declarations
            }
        }

        Ok(AliasCheckResult { errors: self.errors })
    }

    fn check_decl(&mut self, decl: &Decl) {
        self.push_scope();

        // Bind parameters - each gets a fresh backing store if non-copy
        for param in &decl.params {
            self.bind_pattern_params(param);
        }

        // Check the body using visitor
        let _ = self.visit_expression(&decl.body);

        self.pop_scope();
    }

    fn check_entry(&mut self, entry: &EntryDecl) {
        self.push_scope();

        for param in &entry.params {
            self.bind_pattern_params(param);
        }

        let _ = self.visit_expression(&entry.body);

        self.pop_scope();
    }

    /// Bind pattern parameters, creating fresh backing stores for non-copy types
    fn bind_pattern_params(&mut self, pattern: &Pattern) {
        let names = pattern.collect_names();
        for name in names {
            if !self.node_is_copy_type(pattern.h.id) {
                let store_id = self.new_store();
                self.bind_variable(&name, &AliasInfo::fresh(store_id));
            }
        }
    }

    /// Check if a node's type is Copy
    fn node_is_copy_type(&self, node_id: NodeId) -> bool {
        if let Some(scheme) = self.type_table.get(&node_id) {
            is_copy_type_scheme(scheme)
        } else {
            true // Conservative: treat unknown as copy
        }
    }

    /// Check if the i-th parameter of a function is consuming
    fn is_param_consuming(&self, func_id: NodeId, param_index: usize) -> bool {
        if let Some(scheme) = self.type_table.get(&func_id) {
            let ty = unwrap_scheme(scheme);
            is_param_consuming_in_type(ty, param_index)
        } else {
            false
        }
    }

    /// Check if a function's return type is alias-free
    fn return_is_fresh(&self, func_id: NodeId) -> bool {
        if let Some(scheme) = self.type_table.get(&func_id) {
            let ty = unwrap_scheme(scheme);
            get_return_type_is_fresh(ty)
        } else {
            false
        }
    }
}

impl<'a> Visitor for AliasChecker<'a> {
    type Break = ();

    fn visit_expr_int_literal(&mut self, id: NodeId, _n: i32) -> ControlFlow<Self::Break> {
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_float_literal(&mut self, id: NodeId, _f: f32) -> ControlFlow<Self::Break> {
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_bool_literal(&mut self, id: NodeId, _b: bool) -> ControlFlow<Self::Break> {
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_identifier(&mut self, id: NodeId, name: &str) -> ControlFlow<Self::Break> {
        if self.node_is_copy_type(id) {
            self.set_result(id, AliasInfo::copy());
        } else if let Some(stores) = self.lookup_variable(name) {
            // Check if any backing store has been consumed
            if let Some((consumed_var, consumed_at)) = self.check_stores_live(&stores) {
                // TODO: Pass AST or SpanTable to get actual span from NodeId
                self.errors.push(AliasError {
                    kind: AliasErrorKind::UseAfterMove {
                        variable: name.to_string(),
                        consumed_var: consumed_var.to_string(),
                        consumed_at,
                    },
                    span: Span::new(0, 0, 0, 0),
                });
            }
            self.set_result(id, AliasInfo::references(stores));
        } else {
            self.set_result(id, AliasInfo::copy());
        }
        ControlFlow::Continue(())
    }

    fn visit_expr_array_literal(
        &mut self,
        id: NodeId,
        elements: &[Expression],
    ) -> ControlFlow<Self::Break> {
        // Visit all elements first
        for elem in elements {
            self.visit_expression(elem)?;
        }
        // Array literal creates a fresh backing store
        let store_id = self.new_store();
        self.set_result(id, AliasInfo::fresh(store_id));
        ControlFlow::Continue(())
    }

    fn visit_expr_array_index(
        &mut self,
        id: NodeId,
        array: &Expression,
        index: &Expression,
    ) -> ControlFlow<Self::Break> {
        self.visit_expression(array)?;
        self.visit_expression(index)?;
        // Element type determines if copy or not
        if self.node_is_copy_type(id) {
            self.set_result(id, AliasInfo::copy());
        } else {
            // Non-copy element could alias the array
            // For simplicity, treat as copy for now
            self.set_result(id, AliasInfo::copy());
        }
        ControlFlow::Continue(())
    }

    fn visit_expr_binary_op(
        &mut self,
        id: NodeId,
        _op: &BinaryOp,
        left: &Expression,
        right: &Expression,
    ) -> ControlFlow<Self::Break> {
        self.visit_expression(left)?;
        self.visit_expression(right)?;
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_tuple(&mut self, id: NodeId, elements: &[Expression]) -> ControlFlow<Self::Break> {
        let mut all_stores = HashSet::new();
        for elem in elements {
            self.visit_expression(elem)?;
            all_stores.extend(self.get_result(elem.h.id).stores);
        }
        self.set_result(id, AliasInfo::references(all_stores));
        ControlFlow::Continue(())
    }

    fn visit_expr_let_in(&mut self, id: NodeId, let_in: &LetInExpr) -> ControlFlow<Self::Break> {
        // Visit the value expression
        self.visit_expression(&let_in.value)?;
        let value_info = self.get_result(let_in.value.h.id);

        // Push scope for the body
        self.push_scope();

        // Bind the pattern to the value's alias info
        let names = let_in.pattern.collect_names();
        for name in names {
            self.bind_variable(&name, &value_info);
        }

        // Visit the body
        self.visit_expression(&let_in.body)?;
        let body_info = self.get_result(let_in.body.h.id);

        self.pop_scope();

        self.set_result(id, body_info);
        ControlFlow::Continue(())
    }

    fn visit_expr_if(&mut self, id: NodeId, if_expr: &IfExpr) -> ControlFlow<Self::Break> {
        self.visit_expression(&if_expr.condition)?;
        self.visit_expression(&if_expr.then_branch)?;
        self.visit_expression(&if_expr.else_branch)?;

        let then_info = self.get_result(if_expr.then_branch.h.id);
        let else_info = self.get_result(if_expr.else_branch.h.id);

        // Result aliases union of both branches
        self.set_result(id, then_info.union(&else_info));
        ControlFlow::Continue(())
    }

    fn visit_expr_lambda(&mut self, id: NodeId, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        self.push_scope();

        for param in &lambda.params {
            self.bind_pattern_params(param);
        }

        self.visit_expression(&lambda.body)?;

        self.pop_scope();

        // Lambdas are copy types
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_application(
        &mut self,
        id: NodeId,
        func: &Expression,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        // Visit function
        self.visit_expression(func)?;

        // Start with any alias info the function itself carries
        // (important for curried calls where partial applications carry aliasing)
        let func_info = self.get_result(func.h.id);
        let mut observing_stores: HashSet<BackingStoreId> = func_info.stores.clone();

        // Visit and process each argument
        for (i, arg) in args.iter().enumerate() {
            self.visit_expression(arg)?;
            let arg_info = self.get_result(arg.h.id);

            // Check if this parameter is consuming (*T)
            let param_is_consuming = self.is_param_consuming(func.h.id, i);

            if param_is_consuming {
                // Consume the argument's backing stores
                if !arg_info.stores.is_empty() {
                    let var_name = if let ExprKind::Identifier(name) = &arg.kind {
                        name.clone()
                    } else {
                        format!("<expr>")
                    };
                    self.consume_stores(&arg_info.stores, arg.h.id, &var_name);
                }
            } else {
                // Non-consuming parameter - result might alias this arg
                observing_stores.extend(arg_info.stores);
            }
        }

        // Check if return type is alias-free (*T)
        if self.return_is_fresh(func.h.id) {
            let store_id = self.new_store();
            self.set_result(id, AliasInfo::fresh(store_id));
        } else if observing_stores.is_empty() {
            self.set_result(id, AliasInfo::copy());
        } else {
            // Result aliases all observing arguments (including those from partial applications)
            self.set_result(id, AliasInfo::references(observing_stores));
        }

        ControlFlow::Continue(())
    }

    fn visit_expr_field_access(
        &mut self,
        id: NodeId,
        expr: &Expression,
        _field: &str,
    ) -> ControlFlow<Self::Break> {
        self.visit_expression(expr)?;
        let base_info = self.get_result(expr.h.id);

        if self.node_is_copy_type(id) {
            self.set_result(id, AliasInfo::copy());
        } else {
            // Field access aliases the base
            self.set_result(id, base_info);
        }
        ControlFlow::Continue(())
    }

    // Default implementation handles other cases
    fn visit_expression(&mut self, e: &Expression) -> ControlFlow<Self::Break> {
        // Use the walk function which dispatches to specific handlers
        visitor::walk_expression(self, e)?;

        // If no specific handler set a result, default to copy
        if !self.results.contains_key(&e.h.id) {
            self.set_result(e.h.id, AliasInfo::copy());
        }

        ControlFlow::Continue(())
    }
}

// --- Helper functions for type checking ---

fn is_copy_type(ty: &polytype::Type<TypeName>) -> bool {
    match ty {
        polytype::Type::Constructed(name, args) => match name {
            TypeName::Int(_) => true,
            TypeName::UInt(_) => true,
            TypeName::Float(_) => true,
            TypeName::Str("bool") => true,
            TypeName::Str("unit") => true,
            TypeName::Array => false,
            TypeName::Vec => false,
            TypeName::Mat => false,
            TypeName::Unique => false,
            TypeName::Tuple(_) => args.iter().all(is_copy_type),
            TypeName::Arrow => true, // Functions are copy
            _ => true,               // Conservative: treat unknown as copy
        },
        polytype::Type::Variable(_) => true,
    }
}

fn is_copy_type_scheme(scheme: &TypeScheme<TypeName>) -> bool {
    is_copy_type(unwrap_scheme(scheme))
}

fn is_unique_type(ty: &polytype::Type<TypeName>) -> bool {
    matches!(ty, polytype::Type::Constructed(TypeName::Unique, _))
}

fn unwrap_scheme(scheme: &TypeScheme<TypeName>) -> &polytype::Type<TypeName> {
    match scheme {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { body, .. } => unwrap_scheme(body),
    }
}

fn is_param_consuming_in_type(ty: &polytype::Type<TypeName>, param_index: usize) -> bool {
    match ty {
        polytype::Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
            if param_index == 0 {
                is_unique_type(&args[0])
            } else {
                is_param_consuming_in_type(&args[1], param_index - 1)
            }
        }
        _ => false,
    }
}

fn get_return_type_is_fresh(ty: &polytype::Type<TypeName>) -> bool {
    match ty {
        polytype::Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
            get_return_type_is_fresh(&args[1])
        }
        polytype::Type::Constructed(TypeName::Unique, _) => true,
        _ => false,
    }
}

/// Result of alias checking
#[derive(Debug)]
pub struct AliasCheckResult {
    pub errors: Vec<AliasError>,
}

impl AliasCheckResult {
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn print_errors(&self) {
        for error in &self.errors {
            match &error.kind {
                AliasErrorKind::UseAfterMove {
                    variable,
                    consumed_var,
                    ..
                } => {
                    eprintln!("error: use of moved value `{}`", variable);
                    eprintln!("  --> {:?}", error.span);
                    eprintln!("  = note: value was moved when `{}` was consumed", consumed_var);
                }
            }
        }
    }
}
