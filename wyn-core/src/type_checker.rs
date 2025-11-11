use crate::ast::TypeName;
use crate::ast::*;
use crate::builtin_registry::{BuiltinDescriptor, TypeVarGenerator};
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use log::debug;
use polytype::{Context, TypeScheme};
use std::collections::{BTreeSet, HashMap};

// Implement TypeVarGenerator for Context
impl TypeVarGenerator for Context<TypeName> {
    fn new_variable(&mut self) -> Type {
        Context::new_variable(self)
    }
}

/// A warning produced during type checking
#[derive(Debug, Clone)]
pub enum TypeWarning {
    /// A type hole was filled with an inferred type
    TypeHoleFilled {
        inferred_type: Type,
        span: Span,
    },
}

impl TypeWarning {
    /// Get the span for this warning
    pub fn span(&self) -> &Span {
        match self {
            TypeWarning::TypeHoleFilled { span, .. } => span,
        }
    }

    /// Format the warning as a display message
    pub fn message(&self, formatter: &dyn Fn(&Type) -> String) -> String {
        match self {
            TypeWarning::TypeHoleFilled { inferred_type, .. } => {
                format!("Hole of type {}", formatter(inferred_type))
            }
        }
    }
}

pub struct TypeChecker {
    scope_stack: ScopeStack<TypeScheme<TypeName>>, // Store polymorphic types
    context: Context<TypeName>,                    // Polytype unification context
    record_field_map: HashMap<(String, String), Type>, // Map (type_name, field_name) -> field_type
    builtin_registry: crate::builtin_registry::BuiltinRegistry, // Centralized builtin registry
    type_table: HashMap<crate::ast::NodeId, Type>, // Maps expression NodeId to inferred type
    warnings: Vec<TypeWarning>,                    // Collected warnings
    type_holes: Vec<(NodeId, Span)>,               // Track type hole locations for warning emission
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute free type variables in a Type
fn fv_type(ty: &Type) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn go(t: &Type, acc: &mut BTreeSet<usize>) {
        match t {
            Type::Variable(n) => {
                acc.insert(*n);
            }
            Type::Constructed(_, args) => {
                for a in args {
                    go(a, acc);
                }
            }
        }
    }
    go(ty, &mut out);
    out
}

/// Compute free type variables in a TypeScheme
fn fv_scheme(s: &TypeScheme<TypeName>) -> BTreeSet<usize> {
    match s {
        TypeScheme::Monotype(t) => fv_type(t),
        TypeScheme::Polytype { variable, body } => {
            let mut set = fv_scheme(body);
            set.remove(variable);
            set
        }
    }
}

/// Wrap a TypeScheme in nested Polytype quantifiers for the given variables
fn quantify(mut body: TypeScheme<TypeName>, vars: &BTreeSet<usize>) -> TypeScheme<TypeName> {
    // Quantify in descending order so the smallest variable ends up outermost
    for v in vars.iter().rev() {
        body = TypeScheme::Polytype {
            variable: *v,
            body: Box::new(body),
        };
    }
    body
}

impl TypeChecker {
    fn types_equal(&self, left: &Type, right: &Type) -> bool {
        match (left, right) {
            (Type::Constructed(l_name, l_args), Type::Constructed(r_name, r_args)) => {
                l_name == r_name
                    && l_args.len() == r_args.len()
                    && l_args.iter().zip(r_args.iter()).all(|(l, r)| self.types_equal(l, r))
            }
            (Type::Variable(l_id), Type::Variable(r_id)) => l_id == r_id,
            _ => false,
        }
    }

    /// Format a type for error messages by applying current substitution
    pub fn format_type(&self, ty: &Type) -> String {
        let applied = ty.apply(&self.context);
        match applied {
            Type::Constructed(name, args) if args.is_empty() => format!("{}", name),
            Type::Constructed(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.format_type(a)).collect();
                format!("{}[{}]", name, arg_strs.join(", "))
            }
            Type::Variable(id) => format!("?{}", id),
        }
    }

    /// Compute all free type variables in the current environment (scope stack)
    fn env_free_type_vars(&self) -> BTreeSet<usize> {
        let mut acc = BTreeSet::new();
        self.scope_stack.for_each_binding(|_name, sch| {
            acc.extend(fv_scheme(sch));
        });
        acc
    }

    /// HM-style generalization at let: ∀(fv(ty) \ fv(env)). ty
    /// Quantifies over type variables that are free in ty but not free in the environment
    fn generalize(&self, ty: &Type) -> TypeScheme<TypeName> {
        // Always generalize the *solved* view
        let applied = ty.apply(&self.context);

        // Free vars in type
        let mut fv_ty = fv_type(&applied);

        // Free vars in environment
        let fv_env = self.env_free_type_vars();

        // vars to quantify = fv(ty) \ fv(env)
        for v in fv_env {
            fv_ty.remove(&v);
        }

        // Wrap in nested Polytype quantifiers
        quantify(TypeScheme::Monotype(applied), &fv_ty)
    }

    /// Try to extract a qualified name from a FieldAccess expression chain
    /// Returns Some(QualName) if the expression is a chain of Identifier + FieldAccess
    /// E.g., M.N.x -> QualName { qualifiers: ["M", "N"], name: "x" }
    ///       f32.cos -> QualName { qualifiers: ["f32"], name: "cos" }
    fn try_extract_qual_name(expr: &Expression, final_field: &str) -> Option<crate::ast::QualName> {
        let mut qualifiers = Vec::new();
        let mut current = expr;

        // Walk up the FieldAccess chain collecting qualifiers
        loop {
            match &current.kind {
                ExprKind::Identifier(name) => {
                    // Base case: found the root identifier
                    qualifiers.push(name.clone());
                    qualifiers.reverse();
                    return Some(crate::ast::QualName::new(qualifiers, final_field.to_string()));
                }
                ExprKind::FieldAccess(base, field) => {
                    // Intermediate field access - this is a qualifier
                    qualifiers.push(field.clone());
                    current = base;
                }
                _ => {
                    // Not a simple qualified name chain (e.g., function call, literal, etc.)
                    return None;
                }
            }
        }
    }

    pub fn new() -> Self {
        let mut context = Context::default();
        let builtin_registry = crate::builtin_registry::BuiltinRegistry::new(&mut context);

        TypeChecker {
            scope_stack: ScopeStack::new(),
            context,
            record_field_map: HashMap::new(),
            builtin_registry,
            type_table: HashMap::new(),
            warnings: Vec::new(),
            type_holes: Vec::new(),
        }
    }

    /// Get all warnings collected during type checking
    pub fn warnings(&self) -> &[TypeWarning] {
        &self.warnings
    }

    /// Create a fresh type for a pattern based on its structure
    /// For tuple patterns, creates a tuple of fresh type variables
    /// For simple patterns, creates a single fresh type variable
    fn fresh_type_for_pattern(&mut self, pattern: &Pattern) -> Type {
        match &pattern.kind {
            PatternKind::Tuple(patterns) => {
                // Create a tuple type with fresh type variable for each element
                let elem_types: Vec<Type> =
                    patterns.iter().map(|p| self.fresh_type_for_pattern(p)).collect();
                types::tuple(elem_types)
            }
            PatternKind::Typed(_, annotated_type) => {
                // Pattern has explicit type, use it
                annotated_type.clone()
            }
            PatternKind::Attributed(_, inner_pattern) => {
                // Ignore attributes, recurse on inner pattern
                self.fresh_type_for_pattern(inner_pattern)
            }
            _ => {
                // For simple patterns (Name, Wildcard, etc.), create a fresh type variable
                self.context.new_variable()
            }
        }
    }

    /// Bind a pattern with a given type, adding bindings to the current scope
    /// Returns the actual type that the pattern matches (for type checking)
    /// If generalize is true, generalizes types for polymorphism (used in let bindings)
    fn bind_pattern(&mut self, pattern: &Pattern, expected_type: &Type, generalize: bool) -> Result<Type> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                // Simple name binding
                let type_scheme = if generalize {
                    self.generalize(expected_type)
                } else {
                    TypeScheme::Monotype(expected_type.clone())
                };
                self.scope_stack.insert(name.clone(), type_scheme);
                Ok(expected_type.clone())
            }
            PatternKind::Wildcard => {
                // Wildcard doesn't bind anything
                Ok(expected_type.clone())
            }
            PatternKind::Tuple(patterns) => {
                // Expected type should be a tuple with matching arity
                let expected_applied = expected_type.apply(&self.context);

                match expected_applied {
                    Type::Constructed(TypeName::Str("tuple"), ref elem_types) => {
                        if elem_types.len() != patterns.len() {
                            return Err(CompilerError::TypeError(
                                format!(
                                    "Tuple pattern has {} elements but type has {}",
                                    patterns.len(),
                                    elem_types.len()
                                ),
                                pattern.h.span,
                            ));
                        }

                        // Bind each sub-pattern with its corresponding element type
                        for (sub_pattern, elem_type) in patterns.iter().zip(elem_types.iter()) {
                            self.bind_pattern(sub_pattern, elem_type, generalize)?;
                        }

                        Ok(expected_type.clone())
                    }
                    _ => Err(CompilerError::TypeError(
                        format!(
                            "Expected tuple type for tuple pattern, got {}",
                            self.format_type(&expected_applied)
                        ),
                        pattern.h.span,
                    )),
                }
            }
            PatternKind::Typed(inner_pattern, annotated_type) => {
                // Pattern has a type annotation - unify with expected type
                self.context.unify(annotated_type, expected_type).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "Pattern type annotation {} doesn't match expected type {}",
                            self.format_type(annotated_type),
                            self.format_type(expected_type)
                        ),
                        pattern.h.span,
                    )
                })?;
                // Bind the inner pattern with the annotated type
                self.bind_pattern(inner_pattern, annotated_type, generalize)
            }
            PatternKind::Attributed(_, inner_pattern) => {
                // Ignore attributes, bind the inner pattern
                self.bind_pattern(inner_pattern, expected_type, generalize)
            }
            PatternKind::Unit => {
                // Unit pattern should match unit type
                let unit_type = types::tuple(vec![]);
                self.context.unify(&unit_type, expected_type).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "Unit pattern doesn't match expected type {}",
                            self.format_type(expected_type)
                        ),
                        pattern.h.span,
                    )
                })?;
                Ok(unit_type)
            }
            _ => {
                // Other patterns not yet supported in lambda parameters
                Err(CompilerError::TypeError(
                    format!(
                        "Pattern {:?} not yet supported in lambda parameters",
                        pattern.kind
                    ),
                    pattern.h.span,
                ))
            }
        }
    }

    /// Check an expression against an expected type (bidirectional checking mode)
    /// Returns the actual type (which should unify with expected_type)
    fn check_expression(&mut self, expr: &Expression, expected_type: &Type) -> Result<Type> {
        match &expr.kind {
            ExprKind::Lambda(lambda) => {
                // Special handling for lambdas in check mode
                // Extract parameter types from the expected function type
                let original_expected_type = expected_type.clone();
                let mut expected_type = expected_type.clone();
                let mut expected_param_types = Vec::new();

                // Unwrap nested function types to get parameter types
                for _ in 0..lambda.params.len() {
                    let applied = expected_type.apply(&self.context);
                    if let Some((param_type, result_type)) = types::as_arrow(&applied) {
                        expected_param_types.push(param_type.clone());
                        expected_type = result_type.clone();
                    } else {
                        // Expected type doesn't match lambda structure, fall back to inference
                        return self.infer_expression(expr);
                    }
                }

                // Now check the lambda with known parameter types
                self.scope_stack.push_scope();

                let mut param_types = Vec::new();
                for (param, expected_param_type) in lambda.params.iter().zip(expected_param_types.iter()) {
                    // If parameter has a type annotation, trust it
                    // Otherwise use the expected type from bidirectional checking
                    let param_type = if let Some(annotated_type) = param.pattern_type() {
                        annotated_type.clone()
                    } else {
                        // Use the expected type for the parameter
                        expected_param_type.clone()
                    };

                    param_types.push(param_type.clone());

                    // Bind the pattern (handles tuples, wildcards, etc.)
                    // Lambda parameters are not generalized
                    self.bind_pattern(param, &param_type, false)?;
                }

                // Check the body
                let body_type = self.infer_expression(&lambda.body)?;

                // If return type annotation exists, unify it with the body type
                let return_type = if let Some(annotated_return_type) = &lambda.return_type {
                    self.context.unify(&body_type, annotated_return_type).map_err(|_| {
                        CompilerError::TypeError(
                            format!(
                                "Lambda body type {} does not match return type annotation {}",
                                self.format_type(&body_type),
                                self.format_type(annotated_return_type)
                            ),
                            lambda.body.h.span,
                        )
                    })?;
                    annotated_return_type.clone()
                } else {
                    body_type
                };

                self.scope_stack.pop_scope();

                // Build the function type
                let mut func_type = return_type;
                for param_type in param_types.iter().rev() {
                    func_type = types::function(param_type.clone(), func_type);
                }

                // Unify the built function type with the original expected type
                self.context.unify(&func_type, &original_expected_type).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "Lambda type {} doesn't match expected type {}",
                            self.format_type(&func_type),
                            self.format_type(&original_expected_type)
                        ),
                        expr.h.span,
                    )
                })?;

                Ok(func_type)
            }
            _ => {
                // For non-lambdas, infer and unify with expected
                let actual_type = self.infer_expression(expr)?;
                self.context.unify(&actual_type, expected_type).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "Type mismatch: expected {}, got {}",
                            self.format_type(expected_type),
                            self.format_type(&actual_type)
                        ),
                        expr.h.span,
                    )
                })?;
                Ok(actual_type)
            }
        }
    }

    /// Substitute UserVars with bound type variables (recursive helper)
    fn substitute_type_params_static(ty: &Type, bindings: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Constructed(TypeName::UserVar(name), _) => {
                // Replace UserVar with the bound type variable
                bindings.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Constructed(name, args) => {
                let new_args: Vec<Type> =
                    args.iter().map(|arg| Self::substitute_type_params_static(arg, bindings)).collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Instantiate a builtin function type with fresh type variables
    fn instantiate_builtin_type(&mut self, desc: &BuiltinDescriptor) -> Type {
        use std::collections::HashMap;

        // Collect all type variables used in the builtin signature
        let mut var_map: HashMap<usize, Type> = HashMap::new();

        fn instantiate_with_map(
            ty: &Type,
            var_map: &mut HashMap<usize, Type>,
            context: &mut Context<TypeName>,
        ) -> Type {
            match ty {
                Type::Variable(n) => {
                    // Get or create a fresh variable for this type variable
                    var_map.entry(*n).or_insert_with(|| context.new_variable()).clone()
                }
                Type::Constructed(name, args) => {
                    let new_args =
                        args.iter().map(|arg| instantiate_with_map(arg, var_map, context)).collect();
                    Type::Constructed(name.clone(), new_args)
                }
            }
        }

        // Instantiate the return type and all parameter types
        let return_type = instantiate_with_map(&desc.return_type, &mut var_map, &mut self.context);
        let param_types: Vec<Type> = desc
            .param_types
            .iter()
            .map(|pt| instantiate_with_map(pt, &mut var_map, &mut self.context))
            .collect();

        // Build the function type: param1 -> param2 -> ... -> return
        let func_type =
            param_types.iter().rev().fold(return_type, |acc, param_ty| Type::arrow(param_ty.clone(), acc));

        // Apply the context to resolve any constraints
        func_type.apply(&self.context)
    }

    // TODO: Polymorphic builtins (map, zip, length) need special handling
    // They should either be added to BuiltinRegistry with TypeScheme support,
    // or kept separate with manual registration here
    pub fn load_builtins(&mut self) -> Result<()> {
        // Add builtin function types directly using manual construction

        // length: ∀a n. [n]a -> i32
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();
        let array_type = Type::Constructed(TypeName::Array, vec![var_n, var_a]);
        let length_body = Type::arrow(array_type, types::i32());
        let length_scheme = TypeScheme::Monotype(length_body);
        self.scope_stack.insert("length".to_string(), length_scheme);

        // map: ∀a b n. (a -> b) -> *Array(n, a) -> Array(n, b)
        // The input array is consumed (unique), output is fresh
        // Build the type using Type::Variable(0,1,2) for proper polymorphism
        let var_a = Type::Variable(0);
        let var_b = Type::Variable(1);
        let var_n = Type::Variable(2);
        let func_type = Type::arrow(var_a.clone(), var_b.clone());
        let input_array_type =
            types::unique(Type::Constructed(TypeName::Array, vec![var_n.clone(), var_a]));
        let output_array_type = Type::Constructed(TypeName::Array, vec![var_n, var_b]);
        let map_arrow1 = Type::arrow(input_array_type, output_array_type);
        let map_body = Type::arrow(func_type, map_arrow1);
        // Create nested Polytype for ∀a b n
        let map_scheme = TypeScheme::Polytype {
            variable: 0,
            body: Box::new(TypeScheme::Polytype {
                variable: 1,
                body: Box::new(TypeScheme::Polytype {
                    variable: 2,
                    body: Box::new(TypeScheme::Monotype(map_body)),
                }),
            }),
        };
        self.scope_stack.insert("map".to_string(), map_scheme);

        // zip: ∀a b n. [n]a -> [n]b -> [n](a, b)
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let array_a_type = Type::Constructed(TypeName::Array, vec![var_n.clone(), var_a.clone()]);
        let array_b_type = Type::Constructed(TypeName::Array, vec![var_n.clone(), var_b.clone()]);
        let tuple_type = types::tuple(vec![var_a, var_b]);
        let result_array_type = Type::Constructed(TypeName::Array, vec![var_n, tuple_type]);
        let zip_arrow1 = Type::arrow(array_b_type, result_array_type);
        let zip_body = Type::arrow(array_a_type, zip_arrow1);
        let zip_scheme = TypeScheme::Monotype(zip_body);
        self.scope_stack.insert("zip".to_string(), zip_scheme);

        // Array to vector conversion: to_vec
        // to_vec: ∀n a. [n]a -> Vec(n, a)
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();
        let array_input = Type::Constructed(TypeName::Array, vec![var_n.clone(), var_a.clone()]);
        let vec_output = Type::Constructed(TypeName::Vec, vec![var_n, var_a]);
        let to_vec_body = Type::arrow(array_input, vec_output);
        self.scope_stack.insert("to_vec".to_string(), TypeScheme::Monotype(to_vec_body));

        // replicate: ∀a. i32 -> a -> [?]a
        // Creates an array of length n filled with the given value
        // Note: The size is determined by type inference from context
        let var_a = self.context.new_variable();
        let var_size = self.context.new_variable(); // Size will be inferred
        let output_array = Type::Constructed(TypeName::Array, vec![var_size.clone(), var_a.clone()]);
        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);
        let replicate_body = Type::arrow(i32_type, Type::arrow(var_a, output_array));
        self.scope_stack.insert("replicate".to_string(), TypeScheme::Monotype(replicate_body));

        // Vector operations
        // dot: ∀a b. a -> a -> b
        // Polymorphic: takes two values of same type, returns a value (likely scalar)
        // The SPIR-V validator will ensure the types are actually compatible
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let dot_body = Type::arrow(var_a.clone(), Type::arrow(var_a, var_b));
        self.scope_stack.insert("dot".to_string(), TypeScheme::Monotype(dot_body));

        // TODO: Add vector magnitude function (GLSL length)
        // Should be: vec[n]f32 -> f32 or more generally vec[n]t -> t
        // For now, removed to avoid conflict with array length function

        // Trigonometric functions: f32 -> f32
        let trig_type = Type::arrow(types::f32(), types::f32());
        self.scope_stack.insert("sin".to_string(), TypeScheme::Monotype(trig_type.clone()));
        self.scope_stack.insert("cos".to_string(), TypeScheme::Monotype(trig_type.clone()));
        self.scope_stack.insert("tan".to_string(), TypeScheme::Monotype(trig_type));

        // Register vector field mappings
        self.register_vector_fields();

        Ok(())
    }

    fn register_vector_fields(&mut self) {
        // Vector field access is now handled directly in the FieldAccess case
        // Vec(size, element_type) fields (x, y, z, w) return element_type
    }

    /// Register a record type with its field mappings
    pub fn register_record_type(&mut self, type_name: &str, fields: Vec<(String, Type)>) {
        for (field_name, field_type) in fields {
            self.record_field_map.insert((type_name.to_string(), field_name), field_type);
        }
    }

    pub fn check_program(&mut self, program: &Program) -> Result<HashMap<crate::ast::NodeId, Type>> {
        // Process declarations in order - each can only refer to preceding declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        // Emit warnings for all type holes now that types are fully inferred
        self.emit_hole_warnings();

        Ok(self.type_table.clone())
    }

    /// Emit warnings for all type holes showing their inferred types
    fn emit_hole_warnings(&mut self) {
        // Clone the holes list to avoid borrow checker issues
        let holes = self.type_holes.clone();
        for (node_id, span) in holes {
            if let Some(hole_type) = self.type_table.get(&node_id) {
                let resolved_type = hole_type.apply(&self.context);
                self.warnings.push(TypeWarning::TypeHoleFilled {
                    inferred_type: resolved_type,
                    span,
                });
            }
        }
    }

    /// Helper to type check a function body with parameters in scope
    /// Returns (param_types, body_type)
    /// If type_param_bindings is provided, UserVars in parameter types will be substituted
    fn check_function_with_params(
        &mut self,
        params: &[Pattern],
        body: &Expression,
        type_param_bindings: &HashMap<String, Type>,
    ) -> Result<(Vec<Type>, Type)> {
        // Create type variables or use explicit types for parameters
        let param_types: Vec<Type> = params
            .iter()
            .map(|p| {
                let ty = p.pattern_type().cloned().unwrap_or_else(|| self.context.new_variable());
                // Substitute UserVars with bound type variables
                Self::substitute_type_params_static(&ty, type_param_bindings)
            })
            .collect();

        // Push new scope for function parameters
        self.scope_stack.push_scope();

        // Add parameters to scope
        for (param, param_type) in params.iter().zip(param_types.iter()) {
            // Skip unit patterns (no parameters)
            if matches!(param.kind, PatternKind::Unit) {
                continue;
            }

            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    CompilerError::TypeError(
                        "Complex patterns in function parameters not yet supported".to_string(),
                        param.h.span,
                    )
                })?
                .to_string();
            let type_scheme = TypeScheme::Monotype(param_type.clone());
            debug!(
                "Adding parameter '{}' to scope with type: {:?}",
                param_name, param_type
            );
            self.scope_stack.insert(param_name, type_scheme);
        }

        // Infer body type
        let body_type = self.infer_expression(body)?;

        // Pop parameter scope
        self.scope_stack.pop_scope();

        Ok((param_types, body_type))
    }

    fn check_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(decl_node) => {
                debug!("Checking {} declaration: {}", decl_node.keyword, decl_node.name);
                self.check_decl(decl_node)
            }
            Declaration::Entry(entry) => {
                debug!("Checking entry point: {}", entry.name);
                let (_param_types, body_type) =
                    self.check_function_with_params(&entry.params, &entry.body, &HashMap::new())?;
                debug!("Entry point '{}' body type: {:?}", entry.name, body_type);
                // TODO: Validate body_type against entry.return_types and entry.return_attributes
                Ok(())
            }
            Declaration::Uniform(uniform_decl) => {
                debug!("Checking Uniform declaration: {}", uniform_decl.name);
                self.check_uniform_decl(uniform_decl)
            }
            Declaration::Val(val_decl) => {
                debug!("Checking Val declaration: {}", val_decl.name);
                self.check_val_decl(val_decl)
            }
            Declaration::TypeBind(_) => {
                unimplemented!("Type bindings are not yet supported in type checking")
            }
            Declaration::ModuleBind(_) => {
                unimplemented!("Module bindings are not yet supported in type checking")
            }
            Declaration::ModuleTypeBind(_) => {
                unimplemented!("Module type bindings are not yet supported in type checking")
            }
            Declaration::Open(_) => {
                unimplemented!("Open declarations are not yet supported in type checking")
            }
            Declaration::Import(_) => {
                unimplemented!("Import declarations are not yet supported in type checking")
            }
            Declaration::Local(_) => {
                unimplemented!("Local declarations are not yet supported in type checking")
            }
        }
    }

    fn check_uniform_decl(&mut self, decl: &UniformDecl) -> Result<()> {
        // Add the uniform to scope with its declared type
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.scope_stack.insert(decl.name.clone(), type_scheme);
        debug!("Inserting uniform variable '{}' into scope", decl.name);
        Ok(())
    }

    fn check_decl(&mut self, decl: &Decl) -> Result<()> {
        // Bind type parameters to fresh type variables
        // This ensures all occurrences of 'a in the function signature refer to the same variable
        let mut type_param_bindings: HashMap<String, Type> = HashMap::new();
        for type_param in &decl.type_params {
            let fresh_var = self.context.new_variable();
            type_param_bindings.insert(type_param.clone(), fresh_var);
        }

        // Note: substitution function defined as static method below

        if decl.params.is_empty() {
            // Variable or entry point declaration: let/def name: type = value or let/def name = value
            let expr_type = if let Some(declared_type) = &decl.ty {
                // Use bidirectional checking when type annotation is present
                self.check_expression(&decl.body, declared_type)?
            } else {
                // No type annotation, infer the type
                self.infer_expression(&decl.body)?
            };

            if let Some(declared_type) = &decl.ty {
                if !self.types_match(&expr_type, declared_type) {
                    return Err(CompilerError::TypeError(
                        format!(
                            "Type mismatch: expected {}, got {}",
                            self.format_type(declared_type),
                            self.format_type(&expr_type)
                        ),
                        decl.body.h.span,
                    ));
                }
            }

            // Add to scope - use declared type if available, otherwise inferred type
            let stored_type = decl.ty.as_ref().unwrap_or(&expr_type).clone();
            // Generalize the type to enable polymorphism
            let type_scheme = self.generalize(&stored_type);
            debug!("Inserting variable '{}' into scope", decl.name);
            self.scope_stack.insert(decl.name.clone(), type_scheme);
            debug!("Inferred type for {}: {}", decl.name, stored_type);
        } else {
            // Function declaration: let/def name param1 param2 = body

            let (param_types, body_type) =
                self.check_function_with_params(&decl.params, &decl.body, &type_param_bindings)?;
            debug!(
                "Successfully inferred body type for '{}': {:?}",
                decl.name, body_type
            );

            // Build function type: param1 -> param2 -> ... -> body_type
            let func_type = param_types
                .into_iter()
                .rev()
                .fold(body_type.clone(), |acc, param_ty| types::function(param_ty, acc));

            // Check against declared type if provided
            if let Some(declared_type) = &decl.ty {
                // Substitute UserVars in the declared return type
                let substituted_return_type =
                    Self::substitute_type_params_static(declared_type, &type_param_bindings);

                // When a function has parameters, decl.ty is just the return type annotation
                // Unify the body type with the declared return type
                if !decl.params.is_empty() {
                    self.context.unify(&body_type, &substituted_return_type).map_err(|e| {
                        CompilerError::TypeError(
                            format!("Function return type mismatch for '{}': {}", decl.name, e),
                            decl.body.h.span,
                        )
                    })?;
                } else {
                    // For functions without parameters, ty should be the full type
                    // But currently we're storing just the value type
                    // Since func_type for parameterless functions is just the body type,
                    // we can just check body_type against substituted declared_type
                    self.context.unify(&body_type, &substituted_return_type).map_err(|_| {
                        CompilerError::TypeError(
                            format!(
                                "Type mismatch for '{}': declared {}, inferred {}",
                                decl.name,
                                self.format_type(declared_type),
                                self.format_type(&body_type)
                            ),
                            decl.body.h.span,
                        )
                    })?;
                }
            }

            // Entry points are now handled separately via Declaration::Entry
            // Regular Decl no longer has attributed return types

            // Update scope with inferred type using generalization
            let type_scheme = self.generalize(&func_type);
            self.scope_stack.insert(decl.name.clone(), type_scheme);

            debug!("Inferred type for {}: {}", decl.name, func_type);
        }

        Ok(())
    }

    fn check_val_decl(&mut self, decl: &ValDecl) -> Result<()> {
        // Val declarations are just type signatures - register them in scope
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.scope_stack.insert(decl.name.clone(), type_scheme);
        Ok(())
    }

    fn infer_expression(&mut self, expr: &Expression) -> Result<Type> {
        let ty = match &expr.kind {
            ExprKind::TypeHole => {
                // Record this hole for warning emission after type inference completes
                self.type_holes.push((expr.h.id, expr.h.span.clone()));
                Ok(self.context.new_variable())
            }
            ExprKind::IntLiteral(_) => Ok(types::i32()),
            ExprKind::FloatLiteral(_) => Ok(types::f32()),
            ExprKind::BoolLiteral(_) => Ok(types::bool_type()),
            ExprKind::Identifier(name) => {
                debug!("Looking up identifier '{}'", name);
                debug!("Current scope depth: {}", self.scope_stack.depth());

                // First check scope stack for variables
                if let Ok(type_scheme) = self.scope_stack.lookup(name) {
                    debug!("Found '{}' in scope stack with type: {:?}", name, type_scheme);
                    // Instantiate the type scheme to get a concrete type
                    Ok(type_scheme.instantiate(&mut self.context))
                } else if self.builtin_registry.is_builtin(name) {
                    // Then check builtin registry for builtin functions/constructors
                    debug!("'{}' is a builtin", name);
                    if let Some(desc) = self.builtin_registry.get(name) {
                        // Instantiate with fresh type variables
                        let func_type = self.instantiate_builtin_type(&desc.clone());
                        debug!("Built function type for builtin '{}': {:?}", name, func_type);
                        Ok(func_type)
                    } else {
                        Err(CompilerError::UndefinedVariable(name.clone(), expr.h.span))
                    }
                } else {
                    // Not found anywhere
                    debug!("Variable lookup failed for '{}' - not in scope or builtins", name);
                    debug!("Scope stack contents: {:?}", self.scope_stack);
                    Err(CompilerError::UndefinedVariable(name.clone(), expr.h.span))
                }
            }
            ExprKind::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    Err(CompilerError::TypeError(
                        "Cannot infer type of empty array".to_string(),
                        expr.h.span,
                    ))
                } else {
                    let first_type = self.infer_expression(&elements[0])?;
                    for elem in &elements[1..] {
                        let elem_type = self.infer_expression(elem)?;
                        self.context.unify(&elem_type, &first_type).map_err(|_| {
                            CompilerError::TypeError(
                                format!(
                                    "Array elements must have the same type, expected {}, got {}",
                                    self.format_type(&first_type),
                                    self.format_type(&elem_type)
                                ),
                                elem.h.span
                            )
                        })?;
                    }

                    // Array literals have concrete sizes: [1, 2, 3] has type [3]i32
                    // Variable sizes require explicit type parameters: def f[n]: [n]i32 = ...
                    Ok(types::sized_array(elements.len(), first_type))
                }
            }
            ExprKind::ArrayIndex(array_expr, index_expr) => {
                let array_type = self.infer_expression(array_expr)?;
                let index_type = self.infer_expression(index_expr)?;

                // Per spec: array index may be "any unsigned integer type"
                // We also accept signed integers for compatibility
                // Apply context first to resolve any type variables
                let index_type_resolved = index_type.apply(&self.context);
                if !types::is_integer_type(&index_type_resolved) {
                    return Err(CompilerError::TypeError(
                        format!(
                            "Array index must be an integer type, got {}",
                            self.format_type(&index_type_resolved)
                        ),
                        index_expr.h.span
                    ));
                }

                // Use HM-style unification instead of pattern matching
                // This allows indexing arrays whose type is currently a meta-var
                let size_var = self.context.new_variable();
                let elem_var = self.context.new_variable();
                let want_array = Type::Constructed(TypeName::Array, vec![size_var, elem_var.clone()]);

                self.context.unify(&array_type, &want_array).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "Cannot index non-array type: got {}",
                            self.format_type(&array_type.apply(&self.context))
                        ),
                        array_expr.h.span
                    )
                })?;

                // Return the element type, resolved through the context
                Ok(elem_var.apply(&self.context))
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;

                // Check that both operands have compatible types
                self.context.unify(&left_type, &right_type).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "Binary operator '{}' requires operands of the same type, got {} and {}",
                            op.op, left_type, right_type
                        ),
                        expr.h.span
                    )
                })?;

                // Determine return type based on operator
                match op.op.as_str() {
                    "==" | "!=" | "<" | ">" | "<=" | ">=" => {
                        // Comparison operators return boolean
                        Ok(Type::Constructed(TypeName::Str("bool"), vec![]))
                    }
                    "+" | "-" | "*" | "/" => {
                        // Arithmetic operators return the same type as operands
                        Ok(left_type.apply(&self.context))
                    }
                    _ => Err(CompilerError::TypeError(
                        format!(
                            "Unknown binary operator: {}",
                            op.op
                        ),
                        expr.h.span
                    )),
                }
            }
            ExprKind::FunctionCall(func_name, args) => {
                // Get function type - check scope stack first, then builtin registry
                let func_type_result: Result<Type> =
                    if let Ok(type_scheme) = self.scope_stack.lookup(func_name) {
                        Ok(type_scheme.instantiate(&mut self.context))
                    } else if self.builtin_registry.is_builtin(func_name) {
                        // Get type from builtin registry
                        if let Some(desc) = self.builtin_registry.get(func_name) {
                            Ok(self.instantiate_builtin_type(&desc.clone()))
                        } else {
                            Err(CompilerError::UndefinedVariable(func_name.clone(), expr.h.span))
                        }
                    } else {
                        Err(CompilerError::UndefinedVariable(func_name.clone(), expr.h.span))
                    };

                let func_type = func_type_result?;

                // Use two-pass application for better lambda inference
                self.apply_two_pass(func_type, args)
            }
            ExprKind::Tuple(elements) => {
                let elem_types: Result<Vec<Type>> =
                    elements.iter().map(|e| self.infer_expression(e)).collect();

                Ok(types::tuple(elem_types?))
            }
            ExprKind::Lambda(lambda) => {
                // Push new scope for lambda parameters
                self.scope_stack.push_scope();

                // Add parameters to scope with their types (or fresh type variables)
                // Save the parameter types so we can reuse them when building the function type
                let mut param_types = Vec::new();
                for param in &lambda.params {
                    let param_type = param.pattern_type().cloned().unwrap_or_else(|| {
                        // No explicit type annotation - infer from pattern shape
                        self.fresh_type_for_pattern(param)
                    });
                    param_types.push(param_type.clone());

                    // Bind the pattern (handles tuples, wildcards, etc.)
                    // Lambda parameters are not generalized
                    self.bind_pattern(param, &param_type, false)?;
                }

                // Type check the lambda body with parameters in scope
                let body_type = self.infer_expression(&lambda.body)?;

                // If return type annotation exists, unify it with the body type
                let return_type = if let Some(annotated_return_type) = &lambda.return_type {
                    self.context.unify(&body_type, annotated_return_type).map_err(|_| {
                        CompilerError::TypeError(
                            format!(
                                "Lambda body type {} does not match return type annotation {}",
                                self.format_type(&body_type),
                                self.format_type(annotated_return_type)
                            ),
                            lambda.body.h.span
                        )
                    })?;
                    annotated_return_type.clone()
                } else {
                    body_type
                };

                // Pop parameter scope
                self.scope_stack.pop_scope();

                // For multiple parameters, create nested function types using the SAME type variables
                // we used when adding parameters to scope
                let mut func_type = return_type;
                for param_type in param_types.iter().rev() {
                    func_type = types::function(param_type.clone(), func_type);
                }

                Ok(func_type)
            }
            ExprKind::LetIn(let_in) => {
                // Infer type of the value expression
                let value_type = self.infer_expression(&let_in.value)?;

                // Check type annotation if present
                if let Some(declared_type) = &let_in.ty {
                    self.context.unify(&value_type, declared_type).map_err(|_| {
                        CompilerError::TypeError(
                            format!(
                                "Type mismatch in let binding: expected {}, got {}",
                                declared_type, value_type
                            ),
                            let_in.value.h.span
                        )
                    })?;
                }

                // Push new scope and bind pattern
                self.scope_stack.push_scope();
                let bound_type = let_in.ty.as_ref().unwrap_or(&value_type).clone();

                // Bind all names in the pattern
                // Let bindings should be generalized for polymorphism
                self.bind_pattern(&let_in.pattern, &bound_type, true)?;

                // Infer type of body expression
                let body_type = self.infer_expression(&let_in.body)?;

                // Pop scope
                self.scope_stack.pop_scope();

                Ok(body_type)
            }
            ExprKind::Application(func, args) => {
                let func_type = self.infer_expression(func)?;

                // Use two-pass application for better lambda inference
                // This enables proper inference for expressions like (map (\x -> ...) arr)
                // or (|>) operators with lambdas
                self.apply_two_pass(func_type, args)
            }
            ExprKind::FieldAccess(expr, field) => {
                // Try to extract a qualified name (e.g., f32.cos, M.N.x)
                if let Some(qual_name) = Self::try_extract_qual_name(expr, field) {
                    let dotted = qual_name.to_dotted();
                    let mangled = qual_name.mangle();

                    // Check if this is a module-qualified name (mangled name exists in scope)
                    if let Ok(scheme) = self.scope_stack.lookup(&mangled) {
                        // Instantiate the type scheme
                        let ty = scheme.instantiate(&mut self.context);
                        return Ok(ty);
                    }

                    // Check if this is a builtin function (e.g., f32.sin)
                    if self.builtin_registry.is_builtin(&dotted) {
                        if let Some(desc) = self.builtin_registry.get(&dotted) {
                            // Instantiate with fresh type variables
                            return Ok(self.instantiate_builtin_type(&desc.clone()));
                        }
                    }

                    // Qualified name not found as module or builtin - fall through to field access
                }

                // Not a qualified name (or wasn't found), treat as normal field access
                {
                    // Not a qualified name, proceed with normal field access
                    let expr_type = self.infer_expression(expr)?;

                    // Apply context to resolve any type variables that have been unified
                    let expr_type = expr_type.apply(&self.context);

                    // Extract the type name from the expression type
                    match expr_type {
                        Type::Constructed(type_name, ref args) => {
                            // Handle Vec type specially for field access
                            if matches!(type_name, TypeName::Vec) {
                                // Vec(size, element_type) - must have exactly 2 args
                                if args.len() != 2 {
                                    return Err(CompilerError::TypeError(
                                        format!(
                                            "Malformed Vec type - expected 2 arguments (size, element), got {}",
                                            args.len()
                                        ),
                                        expr.h.span,
                                    ));
                                }

                                // Fields x, y, z, w return the element type (args[1])
                                let element_type = &args[1];

                                // Check if field is valid (x, y, z, w)
                                if matches!(field.as_str(), "x" | "y" | "z" | "w") {
                                    Ok(element_type.clone())
                                } else {
                                    Err(CompilerError::TypeError(
                                        format!(
                                            "Vector type has no field '{}'",
                                            field
                                        ),
                                        expr.h.span
                                    ))
                                }
                            } else {
                                // Get the type name as a string for other types
                                let type_name_str = match &type_name {
                                    TypeName::Str(s) => s.to_string(),
                                    TypeName::Array => "array".to_string(),
                                    TypeName::Vec => "vec".to_string(),
                                    TypeName::Size(n) => n.to_string(),
                                    TypeName::SizeVar(name) => name.clone(),
                                    TypeName::UserVar(name) => format!("'{}", name),
                                    TypeName::Named(name) => name.clone(),
                                    TypeName::Unique => "unique".to_string(),
                                    TypeName::Record(_) => "record".to_string(),
                                    TypeName::Sum(_) => "sum".to_string(),
                                    TypeName::Existential(_, _) => "existential".to_string(),
                                    TypeName::NamedParam(_, _) => "named_param".to_string(),
                                };

                                // Look up field in builtin registry (for vector types)
                                if let Some(field_type) =
                                    self.builtin_registry.get_field_type(&type_name_str, field)
                                {
                                    Ok(field_type)
                                } else if let Some(field_type) =
                                    self.record_field_map.get(&(type_name_str.clone(), field.clone()))
                                {
                                    Ok(field_type.clone())
                                } else {
                                    Err(CompilerError::TypeError(
                                        format!(
                                            "Type '{}' has no field '{}'",
                                            type_name_str, field
                                        ),
                                        expr.h.span
                                    ))
                                }
                            }
                        }
                        _ => Err(CompilerError::TypeError(
                            format!(
                                "Field access '{}' not supported on type {}",
                                field, expr_type
                            ),
                            expr.h.span
                        )),
                    }
                }
            }
            ExprKind::If(if_expr) => {
                // Infer condition type - should be bool
                let condition_ty = self.infer_expression(&if_expr.condition)?;
                let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

                // Unify condition with bool type
                self.context.unify(&condition_ty, &bool_ty).map_err(|_| {
                    CompilerError::TypeError(
                        format!("If condition must be boolean, got: {}", self.format_type(&condition_ty)),
                        if_expr.condition.h.span
                    )
                })?;

                // Infer then and else branch types - they must be the same
                let then_ty = self.infer_expression(&if_expr.then_branch)?;
                let else_ty = self.infer_expression(&if_expr.else_branch)?;

                // Unify then and else types
                self.context.unify(&then_ty, &else_ty).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "If branches have incompatible types: then={}, else={}",
                            then_ty, else_ty
                        ),
                        if_expr.else_branch.h.span
                    )
                })?;

                Ok(then_ty)
            }

            // New expression kinds - to be implemented
            ExprKind::QualifiedName(_, _) => {
                todo!("QualifiedName not yet implemented in type checker")
            }

            ExprKind::UnaryOp(op, operand) => {
                let operand_type = self.infer_expression(operand)?;
                let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
                match op.op.as_str() {
                    "-" => {
                        // Numeric negation - operand must be numeric, returns same type
                        Ok(operand_type)
                    }
                    "!" => {
                        // Logical not - operand must be bool, returns bool
                        self.context.unify(&operand_type, &bool_ty).map_err(|_| {
                            CompilerError::TypeError(
                                format!(
                                    "Logical not (!) requires bool operand, got {:?}",
                                    operand_type
                                ),
                                operand.h.span
                            )
                        })?;
                        Ok(bool_ty)
                    }
                    _ => Err(CompilerError::TypeError(
                        format!(
                            "Unknown unary operator: {}",
                            op.op
                        ),
                        expr.h.span
                    )),
                }
            }

            ExprKind::Loop(loop_expr) => {
                // Type check the initial value
                let init_type = if let Some(init) = &loop_expr.init {
                    self.infer_expression(init)?
                } else {
                    // If no init provided, infer from pattern variables in scope
                    // For now, just create a fresh type variable
                    self.context.new_variable()
                };

                // Extract pattern type if annotated, otherwise use init type
                let pattern_type = loop_expr.pattern.pattern_type().cloned().unwrap_or(init_type.clone());

                // Unify init with pattern type
                self.context.unify(&init_type, &pattern_type).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Loop initial value type mismatch: {:?}", e),
                        loop_expr.init.as_ref().map(|e| e.h.span).unwrap_or(expr.h.span)
                    )
                })?;

                // Push new scope for loop variables
                self.scope_stack.push_scope();

                // Bind the loop pattern with its type
                // Loop variables are not generalized
                self.bind_pattern(&loop_expr.pattern, &pattern_type, false)?;

                // Type check loop form
                match &loop_expr.form {
                    LoopForm::While(cond) => {
                        let cond_type = self.infer_expression(cond)?;
                        // Condition must be boolean
                        self.context.unify(&cond_type, &types::bool_type()).map_err(|e| {
                            CompilerError::TypeError(
                                format!("Loop while condition must be bool: {:?}", e),
                                cond.h.span
                            )
                        })?;
                    }
                    LoopForm::For(name, bound) => {
                        // For x < n form: name is the loop variable, bound is n
                        // Bind name as i32
                        let type_scheme = TypeScheme::Monotype(types::i32());
                        self.scope_stack.insert(name.clone(), type_scheme);

                        // Check that bound is i32
                        let bound_type = self.infer_expression(bound)?;
                        self.context.unify(&bound_type, &types::i32()).map_err(|e| {
                            CompilerError::TypeError(
                                format!("Loop for bound must be i32: {:?}", e),
                                bound.h.span
                            )
                        })?;
                    }
                    LoopForm::ForIn(iter_pat, iter_expr) => {
                        // Type check the iterator expression
                        let iter_type = self.infer_expression(iter_expr)?;

                        // Iterator must be an array type: Array(size, elem_type)
                        if let Type::Constructed(TypeName::Array, args) = &iter_type {
                            if args.len() != 2 {
                                return Err(CompilerError::TypeError(
                                    format!(
                                        "Malformed Array type - expected 2 arguments (size, element), got {}",
                                        args.len()
                                    ),
                                    iter_expr.h.span
                                ));
                            }

                            // Array is Array(size, elem_type), so element is at index 1
                            let elem_type = &args[1];
                            // Bind iterator pattern with element type
                            // For-in loop variables are not generalized
                            self.bind_pattern(iter_pat, elem_type, false)?;
                        } else {
                            return Err(CompilerError::TypeError(
                                "Loop for-in expression must be an array".to_string(),
                                iter_expr.h.span
                            ));
                        }
                    }
                }

                // Type check loop body
                let body_type = self.infer_expression(&loop_expr.body)?;

                // Body type must match pattern type (loop accumulator)
                self.context.unify(&body_type, &pattern_type).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Loop body type must match pattern type: {:?}", e),
                        loop_expr.body.h.span
                    )
                })?;

                // Pop loop scope
                self.scope_stack.pop_scope();

                // Loop result type is the pattern/body type
                Ok(pattern_type)
            }

            ExprKind::Match(_) => {
                todo!("Match not yet implemented in type checker")
            }

            ExprKind::Range(_) => {
                todo!("Range not yet implemented in type checker")
            }

            ExprKind::Pipe(left, right) => {
                // a |> f desugars to f(a)
                // Type check left to get argument type
                let arg_type = self.infer_expression(left)?;

                // Type check right (the function)
                let func_type = self.infer_expression(right)?;

                // Create a fresh result type variable
                let result_type = self.context.new_variable();

                // Unify func_type with (arg_type -> result_type)
                let expected_func_type = types::function(arg_type, result_type.clone());
                self.context.unify(&func_type, &expected_func_type).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Pipe operator type error: {:?}", e),
                        expr.h.span
                    )
                })?;

                Ok(result_type)
            }

            ExprKind::TypeAscription(_, _) => {
                todo!("TypeAscription not yet implemented in type checker")
            }

            ExprKind::TypeCoercion(_, _) => {
                todo!("TypeCoercion not yet implemented in type checker")
            }

            ExprKind::Unsafe(_) => {
                todo!("Unsafe not yet implemented in type checker")
            }

            ExprKind::Assert(_, _) => {
                todo!("Assert not yet implemented in type checker")
            }
        } // NEWCASESHERE - add new cases before this closing brace
        ?;

        // Store the inferred type in the type table
        self.type_table.insert(expr.h.id, ty.clone());
        Ok(ty)
    }

    // Removed: fresh_var - now using polytype's context.new_variable()

    /// Two-pass function application for better lambda inference
    ///
    /// Pass 1: Process non-lambda arguments to constrain type variables
    /// Pass 2: Process lambda arguments with bidirectionally checked expected types
    ///
    /// This allows map (\x -> ...) arr to infer properly regardless of argument order
    fn apply_two_pass(&mut self, mut func_type: Type, args: &[Expression]) -> Result<Type> {
        // Collect argument types and expected types for lambdas
        let mut arg_types: Vec<Option<Type>> = vec![None; args.len()];
        let mut lambda_expected_types: Vec<Option<Type>> = vec![None; args.len()];

        // First pass: process arguments to constrain type variables
        for (i, arg) in args.iter().enumerate() {
            if matches!(&arg.kind, ExprKind::Lambda(_)) {
                // For lambdas: peel the head with a fresh arrow (α -> β) and unify
                let param_type_var = self.context.new_variable();
                let result_type = self.context.new_variable();
                let expected_func_type = Type::arrow(param_type_var.clone(), result_type.clone());

                self.context.unify(&func_type, &expected_func_type).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Function application type error: {:?}", e),
                        arg.h.span,
                    )
                })?;

                // Extract the parameter type by applying context
                let param_type = param_type_var.apply(&self.context);
                lambda_expected_types[i] = Some(param_type);

                func_type = result_type;
            } else {
                // For non-lambda argument: infer type and unify
                let arg_type = self.infer_expression(arg)?;
                arg_types[i] = Some(arg_type.clone());

                // Peel the head with a fresh arrow
                let param_type_var = self.context.new_variable();
                let result_type = self.context.new_variable();
                let expected_func_type = Type::arrow(param_type_var.clone(), result_type.clone());

                self.context.unify(&func_type, &expected_func_type).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Function application type error: {:?}", e),
                        arg.h.span,
                    )
                })?;

                // Extract the expected parameter type
                let expected_param_type = param_type_var.apply(&self.context);

                // Check for unique ownership
                let expects_unique = types::is_unique(&expected_param_type);

                // Strip uniqueness for unification
                let arg_type_for_unify = types::strip_unique(&arg_type);
                let expected_param_for_unify = types::strip_unique(&expected_param_type);

                self.context.unify(&arg_type_for_unify, &expected_param_for_unify).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Function argument type mismatch: {:?}", e),
                        arg.h.span,
                    )
                })?;

                // Handle uniqueness/consumption:
                // TODO: Implement proper alias tracking and explicit consumption
                //
                // Current limitation: We only track consumption for direct identifiers.
                // This is unsound because we allow expressions that may alias variables
                // to be passed to consuming parameters without explicit 'consume' annotations.
                //
                // Proper semantics (Futhark-like):
                // 1. Track alias sets during inference: infer_expression -> (Type, AliasSet)
                // 2. At consuming positions, check if arg aliases any variables
                // 3. If aliases non-empty, require explicit 'consume x' expression
                // 4. If aliases empty (fresh value), allow without consumption
                //
                // Examples that should require explicit consume:
                // - xs[0] where indexing returns a view/reference (not scalar copy)
                // - slice xs i j (aliases xs)
                // - f xs where f returns an alias to its parameter
                //
                // For now, we conservatively only consume direct identifiers:
                if expects_unique {
                    if let ExprKind::Identifier(var_name) = &arg.kind {
                        self.scope_stack.mark_consumed(var_name).map_err(|e| {
                            CompilerError::TypeError(
                                format!("Cannot consume variable '{}': {}", var_name, e),
                                arg.h.span,
                            )
                        })?;
                    }
                    // WARNING: Non-identifiers are currently allowed without checking aliases.
                    // This may allow unsound programs where aliased values are consumed.
                }

                func_type = result_type;
            }
        }

        // Second pass: process lambda arguments with bidirectional checking
        for (i, arg) in args.iter().enumerate() {
            if !matches!(&arg.kind, ExprKind::Lambda(_)) {
                continue;
            }

            // Get the expected type from first pass
            let expected_param_type = lambda_expected_types[i].as_ref().map(|t| t.apply(&self.context));

            // Use bidirectional checking for lambdas
            if let Some(ref expected) = expected_param_type {
                self.check_expression(arg, expected)?;
            } else {
                self.infer_expression(arg)?;
            }
        }

        Ok(func_type.apply(&self.context))
    }

    /// Check if two types match, treating tuple and attributed_tuple as compatible.
    ///
    /// This allows attributed_tuple (used in entry point return types) to match
    /// plain tuple types. The attributes are metadata for code generation and don't
    /// affect type compatibility.
    fn types_match(&self, t1: &Type, t2: &Type) -> bool {
        // Apply current substitution without mutating context
        let a = t1.apply(&self.context);
        let b = t2.apply(&self.context);

        // Handle attributed_tuple vs tuple matching (symmetric)
        match (&a, &b) {
            // tuple matches attributed_tuple if component types match
            (
                Type::Constructed(TypeName::Str("tuple"), types1),
                Type::Constructed(TypeName::Str("attributed_tuple"), types2),
            )
            | (
                Type::Constructed(TypeName::Str("attributed_tuple"), types1),
                Type::Constructed(TypeName::Str("tuple"), types2),
            ) => {
                types1.len() == types2.len()
                    && types1.iter().zip(types2.iter()).all(|(t1, t2)| self.types_equal(t1, t2))
            }
            // Regular case - use structural equality after applying substitution
            _ => self.types_equal(&a, &b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_type_check_let() {
        let input = "let x: i32 = 42";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_mismatch() {
        let input = "let x: i32 = 3.14f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_array_type_check() {
        let input = "let arr: [2]f32 = [1.0f32, 2.0f32]";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_undefined_variable() {
        let input = "let x: i32 = undefined";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CompilerError::UndefinedVariable(_, _)
        ));
    }

    #[test]
    fn test_simple_def() {
        let input = "def identity x = x";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_zip_arrays() {
        let input = "def zip_arrays xs ys = zip xs ys";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();
        match checker.check_program(&program) {
            Ok(_) => {
                println!("Type checking succeeded!");

                // Check that zip_arrays has the expected type
                if let Ok(func_type) = checker.scope_stack.lookup("zip_arrays") {
                    println!("zip_arrays type: {}", func_type);

                    // The inferred type should be something like: t0 -> t1 -> [1](i32, i32)
                    // This demonstrates that type inference is working
                }
            }
            Err(e) => {
                println!("Type checking failed: {:?}", e);
                panic!("Type checking failed");
            }
        }
    }

    /// Helper function to check a program with a type hole and return the inferred type
    fn check_type_hole(source: &str) -> Type {
        use crate::lexer;
        use crate::parser::Parser;

        // Parse
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        // Type check
        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();
        let _type_table = checker.check_program(&program).unwrap();

        // Check warnings
        let warnings = checker.warnings();
        assert_eq!(warnings.len(), 1, "Expected exactly one type hole warning");

        match &warnings[0] {
            TypeWarning::TypeHoleFilled { inferred_type, .. } => {
                // Apply the context to normalize type variables
                inferred_type.apply(&checker.context)
            }
        }
    }

    #[test]
    fn test_type_hole_in_array() {
        let inferred = check_type_hole("def arr = [1i32, ???, 3i32]");

        // ??? should be inferred as i32 (to match array elements)
        let expected = Type::Constructed(TypeName::Str("i32"), vec![]);
        assert_eq!(inferred, expected);
    }

    #[test]
    fn test_type_hole_in_binop() {
        let inferred = check_type_hole("def result = 5i32 + ???");

        // ??? should be inferred as i32 (to match addition operand)
        let expected = Type::Constructed(TypeName::Str("i32"), vec![]);
        assert_eq!(inferred, expected);
    }

    #[test]
    fn test_type_hole_function_arg() {
        let inferred = check_type_hole("def apply = (\\x:i32 -> x + 1i32) ???");

        // ??? should be inferred as i32 (the function argument type)
        let expected = Type::Constructed(TypeName::Str("i32"), vec![]);
        assert_eq!(inferred, expected);
    }

    #[test]
    fn test_lambda_param_with_annotation() {
        // Test that lambda parameter works with type annotation (Futhark-style)
        // Field projection requires the parameter type to be known
        let source = "def test : [2]f32 = let arr : [2]vec3f32 = [vec3 1.0f32 2.0f32 3.0f32, vec3 4.0f32 5.0f32 6.0f32] in map (\\(v:vec3f32) -> v.x) arr";

        use crate::lexer;
        use crate::parser::Parser;

        // Parse
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        // Type check
        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed with type annotation
            }
            Err(e) => {
                panic!("Type checking failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_bidirectional_with_concrete_type() {
        // Test bidirectional checking with a CONCRETE expected type
        // This demonstrates where bidirectional checking actually helps
        let source = r#"
            def apply_to_vec (f : vec3f32 -> f32) : f32 =
              f (vec3 1.0f32 2.0f32 3.0f32)

            def test : f32 = apply_to_vec (\v -> v.x)
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        // Parse
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        // Type check
        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed! apply_to_vec expects vec3f32 -> f32 (concrete)
                // so bidirectional checking gives parameter v the type vec3f32
            }
            Err(e) => {
                panic!("Type checking failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_bidirectional_explicit_annotation_mismatch() {
        // Minimal test demonstrating bidirectional checking bug with explicit parameter annotations.
        // Two chained maps: vec3f32->vec4f32, then vec4f32->vec3f32
        // The second lambda's parameter annotation (q:vec4f32) is correct (v4s is [1]vec4f32),
        // but bidirectional checking incorrectly rejects it.
        let source = r#"
            def test =
              let arr : [1]vec3f32 = [vec3 1.0f32 2.0f32 3.0f32] in
              let v4s : [1]vec4f32 = map (\(v:vec3f32) -> vec4 v.x v.y v.z 1.0f32) arr in
              map (\(q:vec4f32) -> vec3 q.x q.y q.z) v4s
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed! Both lambda parameter annotations are correct.
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_map_with_unannotated_lambda_and_array_index() {
        // Test that bidirectional checking infers lambda parameter type from array type
        let source = r#"
            def test : [12]i32 =
              let edges : [12][2]i32 = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]] in
              map (\e -> e[0]) edges
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed! Bidirectional checking should infer e : [2]i32 from edges
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_lambda_with_tuple_pattern() {
        // Test that lambdas with tuple patterns work
        let source = r#"
            def test : (i32, i32) -> i32 =
              \(x, y) -> x + y
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_lambda_with_wildcard_in_tuple() {
        // Test that lambdas with wildcard in tuple patterns work
        let source = r#"
            def test : (i32, i32) -> i32 =
              \(_, acc) -> acc
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_loop_with_tuple_pattern() {
        // Test that loops with tuple patterns work
        let source = r#"
            def test : i32 =
              loop (idx, acc) = (0, 10) while idx < 5 do
                (idx + 1, acc + idx)
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                panic!("Type checking should fail - loop returns tuple but assigned to i32");
            }
            Err(_) => {
                // Should fail - type mismatch
            }
        }
    }

    #[test]
    fn test_loop_with_tuple_pattern_and_pipe() {
        // Test that loops with tuple patterns can be piped to extract result
        // Per SPECIFICATION.md line 559, loop bodies extend as far right as possible,
        // so the pipe must be outside the loop by wrapping in parentheses
        let source = r#"
            def test : i32 =
              (loop (idx, acc) = (0, 10) while idx < 5 do
                (idx + 1, acc + idx))
              |> (\(_, result) -> result)
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed - loop returns (i32, i32), pipe extracts i32
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_let_polymorphism() {
        // Test that let-bound values are properly generalized
        // Without generalization, this would fail because id would be monomorphic
        let source = r#"
            def test : bool =
                let id = \x -> x in
                let test1 : i32 = id ??? in
                let test2 : bool = id ??? in
                test2
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_warnings) => {
                // Should succeed - id is polymorphic ∀a. a -> a
                // Without generalization, this would fail because id would be monomorphic
                // and couldn't be used at both i32 and bool
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_top_level_polymorphism() {
        // Test that top-level let/def declarations are generalized
        let source = r#"
            def id = \x -> x
            def test1 : i32 = id ???
            def test2 : bool = id ???
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_warnings) => {
                // Should succeed - id is polymorphic ∀a. a -> a
                // Without generalization, this would fail because id would be monomorphic
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_polymorphic_id_tuple() {
        // Classic HM polymorphism test: let id = \x -> x in (id 5, id true)
        let source = r#"
            def test =
                let id = \x -> x in
                (id ???, id ???)
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_warnings) => {
                // Should succeed - id is polymorphic and can be used at multiple types
                // Without generalization, this would fail because first use would fix id's type
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }

    #[test]
    fn test_qualified_name_sqrt() {
        // Test that qualified names like f32.sqrt type check correctly
        let source = r#"
            def test : f32 = f32.sqrt 4.0f32
        "#;

        use crate::lexer;
        use crate::parser::Parser;

        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        checker.load_builtins().unwrap();

        match checker.check_program(&program) {
            Ok(_) => {
                // Should succeed - f32.sqrt is a valid builtin
            }
            Err(e) => {
                panic!("Type checking should succeed but failed with: {:?}", e);
            }
        }
    }
}
