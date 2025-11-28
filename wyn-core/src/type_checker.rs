use crate::ast::TypeName;
use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use log::debug;
use polytype::{Context, TypeScheme};
use std::collections::{BTreeSet, HashMap};

/// Trait for generating fresh type variables
pub trait TypeVarGenerator {
    fn new_variable(&mut self) -> Type;
}

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
    module_manager: crate::module_manager::ModuleManager, // Lazy module loading
    type_table: HashMap<crate::ast::NodeId, TypeScheme<TypeName>>, // Maps NodeId to type scheme
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
        match &applied {
            Type::Constructed(TypeName::Str(s), args) if *s == "->" && args.len() == 2 => {
                // Special case for arrow types
                format!("{} -> {}", self.format_type(&args[0]), self.format_type(&args[1]))
            }
            Type::Constructed(TypeName::Tuple(_), args) => {
                // Special case for tuple types
                let arg_strs: Vec<String> = args.iter().map(|a| self.format_type(a)).collect();
                format!("({})", arg_strs.join(", "))
            }
            Type::Constructed(TypeName::Str(s), args) if *s == "Array" && args.len() == 2 => {
                // Special case for array types [size]elem
                format!("[{}]{}", self.format_type(&args[0]), self.format_type(&args[1]))
            }
            Type::Constructed(name, args) if args.is_empty() => format!("{}", name),
            Type::Constructed(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.format_type(a)).collect();
                format!("{}[{}]", name, arg_strs.join(", "))
            }
            Type::Variable(id) => format!("?{}", id),
        }
    }

    /// Format a type scheme for error messages
    pub fn format_scheme(&self, scheme: &TypeScheme<TypeName>) -> String {
        match scheme {
            TypeScheme::Monotype(ty) => self.format_type(ty),
            TypeScheme::Polytype { variable, body } => {
                // For display, we can show quantified vars or just the body
                // For now, just show the body
                format!("∀{}. {}", variable, self.format_scheme(body))
            }
        }
    }

    /// Look up a variable in the scope stack (for testing)
    pub fn lookup(&self, name: &str) -> Option<TypeScheme<TypeName>> {
        self.scope_stack.lookup(name).ok().cloned()
    }

    /// Get a reference to the context (for testing)
    pub fn context(&self) -> &Context<TypeName> {
        &self.context
    }

    /// Compute all free type variables in the current environment (scope stack)
    fn env_free_type_vars(&self) -> BTreeSet<usize> {
        let mut acc = BTreeSet::new();
        self.scope_stack.for_each_binding(|_name, sch| {
            acc.extend(fv_scheme(sch));
        });
        acc
    }

    /// HM-style generalization at let: ∀(fv(ty) \ fv(env) \ ascription_vars). ty
    /// Quantifies over type variables that are free in ty but not free in the environment
    /// and not in the set of ascription variables (which must remain monomorphic)
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
            module_manager: crate::module_manager::ModuleManager::new(),
            type_table: HashMap::new(),
            warnings: Vec::new(),
            type_holes: Vec::new(),
        }
    }

    /// Get all warnings collected during type checking
    pub fn warnings(&self) -> &[TypeWarning] {
        &self.warnings
    }

    /// Get all loaded module declarations (for inlining during flattening)
    pub fn get_loaded_module_declarations(&self) -> Vec<Declaration> {
        self.module_manager.get_all_loaded_declarations()
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
                // Store resolved type in type_table for mirize
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(expected_type.clone())
            }
            PatternKind::Wildcard => {
                // Wildcard doesn't bind anything
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(expected_type.clone())
            }
            PatternKind::Tuple(patterns) => {
                // Expected type should be a tuple with matching arity
                let expected_applied = expected_type.apply(&self.context);

                match expected_applied {
                    Type::Constructed(TypeName::Tuple(_), ref elem_types) => {
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

                        self.type_table.insert(
                            pattern.h.id,
                            TypeScheme::Monotype(expected_type.apply(&self.context)),
                        );
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
                let result = self.bind_pattern(inner_pattern, annotated_type, generalize)?;
                // Also store type for the outer Typed pattern
                let resolved = annotated_type.apply(&self.context);
                self.type_table.insert(pattern.h.id, TypeScheme::Monotype(resolved));
                Ok(result)
            }
            PatternKind::Attributed(_, inner_pattern) => {
                // Ignore attributes, bind the inner pattern
                let result = self.bind_pattern(inner_pattern, expected_type, generalize)?;
                // Also store type for the outer Attributed pattern
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(result)
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
                self.type_table.insert(pattern.h.id, TypeScheme::Monotype(unit_type.apply(&self.context)));
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

    /// Try to unify an overload's function type with the given argument types
    /// Returns the return type if successful, None if unification fails
    fn try_unify_overload(
        func_type: &Type,
        arg_types: &[Type],
        ctx: &mut Context<TypeName>,
    ) -> Option<Type> {
        let mut current_type = func_type.clone();

        for arg_type in arg_types {
            // Decompose the function type: should be param_ty -> rest
            let param_ty = ctx.new_variable();
            let rest_ty = ctx.new_variable();
            let expected_arrow = Type::arrow(param_ty.clone(), rest_ty.clone());

            // Unify current function type with the expected arrow type
            if ctx.unify(&current_type, &expected_arrow).is_err() {
                return None;
            }

            // Unify the parameter type with the argument type
            let param_ty = param_ty.apply(ctx);
            if ctx.unify(&param_ty, arg_type).is_err() {
                return None;
            }

            // Continue with the rest of the function type
            current_type = rest_ty.apply(ctx);
        }

        // After processing all arguments, current_type should be the return type
        Some(current_type)
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

                // Store the checked type in the type table
                self.type_table.insert(expr.h.id, TypeScheme::Monotype(func_type.clone()));
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

    /// Substitute UserVars and SizeVars with bound type variables (recursive helper)
    fn substitute_type_params_static(ty: &Type, bindings: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Constructed(TypeName::UserVar(name), _) => {
                // Replace UserVar with the bound type variable
                bindings.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Constructed(TypeName::SizeVar(name), _) => {
                // Replace SizeVar with the bound size variable
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

    // TODO: Polymorphic builtins (map, zip, length) need special handling
    // They should either be added to BuiltinRegistry with TypeScheme support,
    // or kept separate with manual registration here
    pub fn load_builtins(&mut self) -> Result<()> {
        // Add builtin function types directly using manual construction

        // length: ∀a n. [n]a -> i32
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };

        let array_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let length_body = Type::arrow(array_type, types::i32());

        // Create Polytype ∀n a. [n]a -> i32
        let length_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Monotype(length_body)),
            }),
        };
        self.scope_stack.insert("length".to_string(), length_scheme);

        // map: ∀a b n. (a -> b) -> *Array(n, a) -> Array(n, b)
        // The input array is consumed (unique), output is fresh
        // Build the type using fresh type variables for proper polymorphism
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let var_n = self.context.new_variable();

        // Extract the variable IDs for the TypeScheme
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };
        let var_b_id = if let Type::Variable(id) = var_b { id } else { panic!("Expected Type::Variable") };
        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };

        let func_type = Type::arrow(Type::Variable(var_a_id), Type::Variable(var_b_id));
        let input_array_type = types::unique(Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        ));
        let output_array_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_b_id)],
        );
        let map_arrow1 = Type::arrow(input_array_type, output_array_type);
        let map_body = Type::arrow(func_type, map_arrow1);
        // Create nested Polytype for ∀a b n
        let map_scheme = TypeScheme::Polytype {
            variable: var_a_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_b_id,
                body: Box::new(TypeScheme::Polytype {
                    variable: var_n_id,
                    body: Box::new(TypeScheme::Monotype(map_body)),
                }),
            }),
        };
        self.scope_stack.insert("map".to_string(), map_scheme);

        // zip: ∀a b n. [n]a -> [n]b -> [n](a, b)
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };
        let var_b_id = if let Type::Variable(id) = var_b { id } else { panic!("Expected Type::Variable") };

        let array_a_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let array_b_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_b_id)],
        );
        let tuple_type = types::tuple(vec![Type::Variable(var_a_id), Type::Variable(var_b_id)]);
        let result_array_type =
            Type::Constructed(TypeName::Array, vec![Type::Variable(var_n_id), tuple_type]);
        let zip_arrow1 = Type::arrow(array_b_type, result_array_type);
        let zip_body = Type::arrow(array_a_type, zip_arrow1);

        let zip_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Polytype {
                    variable: var_b_id,
                    body: Box::new(TypeScheme::Monotype(zip_body)),
                }),
            }),
        };
        self.scope_stack.insert("zip".to_string(), zip_scheme);

        // Array to vector conversion: to_vec
        // to_vec: ∀n a. [n]a -> Vec(n, a)
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };

        let array_input = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let vec_output = Type::Constructed(
            TypeName::Vec,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let to_vec_body = Type::arrow(array_input, vec_output);

        let to_vec_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Monotype(to_vec_body)),
            }),
        };
        self.scope_stack.insert("to_vec".to_string(), to_vec_scheme);

        // replicate: ∀size a. i32 -> a -> [size]a
        // Creates an array of length n filled with the given value
        // Note: The size is determined by type inference from context
        let var_a = self.context.new_variable();
        let var_size = self.context.new_variable(); // Size will be inferred

        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };
        let var_size_id =
            if let Type::Variable(id) = var_size { id } else { panic!("Expected Type::Variable") };

        let output_array = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_size_id), Type::Variable(var_a_id)],
        );
        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
        let replicate_body = Type::arrow(i32_type, Type::arrow(Type::Variable(var_a_id), output_array));

        let replicate_scheme = TypeScheme::Polytype {
            variable: var_size_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Monotype(replicate_body)),
            }),
        };
        self.scope_stack.insert("replicate".to_string(), replicate_scheme);

        // __alloc_array: ∀n t. i32 -> [n]t
        // Allocates an uninitialized array of the given size
        // Used by map desugaring; size n and element type t are inferred from usage
        let var_n = self.context.new_variable();
        let var_t = self.context.new_variable();
        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_t_id = if let Type::Variable(id) = var_t { id } else { panic!("Expected Type::Variable") };

        let array_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_t_id)],
        );
        let alloc_array_body = Type::arrow(types::i32(), array_type);

        let alloc_array_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_t_id,
                body: Box::new(TypeScheme::Monotype(alloc_array_body)),
            }),
        };
        self.scope_stack.insert("__alloc_array".to_string(), alloc_array_scheme);

        // Vector operations
        // dot: ∀n t. Vec(n, t) -> Vec(n, t) -> t
        // Takes two vectors of the same size and element type, returns the element type
        let var_n = self.context.new_variable();
        let var_t = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_t_id = if let Type::Variable(id) = var_t { id } else { panic!("Expected Type::Variable") };

        let vec_type = Type::Constructed(
            TypeName::Vec,
            vec![Type::Variable(var_n_id), Type::Variable(var_t_id)],
        );
        let dot_body = Type::arrow(vec_type.clone(), Type::arrow(vec_type, Type::Variable(var_t_id)));

        let dot_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_t_id,
                body: Box::new(TypeScheme::Monotype(dot_body)),
            }),
        };
        self.scope_stack.insert("dot".to_string(), dot_scheme);

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

    pub fn check_program(
        &mut self,
        program: &Program,
    ) -> Result<HashMap<crate::ast::NodeId, TypeScheme<TypeName>>> {
        // Process library declarations first (so they're available to user code)
        for decl in &program.library_declarations {
            self.check_declaration(decl)?;
        }

        // Process user declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        // Emit warnings for all type holes now that types are fully inferred
        self.emit_hole_warnings();

        // Apply the context to all types in the type table to resolve type variables
        let resolved_table: HashMap<crate::ast::NodeId, TypeScheme<TypeName>> = self
            .type_table
            .iter()
            .map(|(node_id, scheme)| {
                let resolved = match scheme {
                    TypeScheme::Monotype(ty) => {
                        let resolved_ty = ty.apply(&self.context);
                        TypeScheme::Monotype(resolved_ty)
                    }
                    TypeScheme::Polytype { variable, body } => {
                        // For polytypes, apply context to the body but preserve quantified variables
                        TypeScheme::Polytype {
                            variable: *variable,
                            body: Box::new(match body.as_ref() {
                                TypeScheme::Monotype(ty) => TypeScheme::Monotype(ty.apply(&self.context)),
                                other => other.clone(), // Nested polytypes stay as-is for now
                            }),
                        }
                    }
                };
                (*node_id, resolved)
            })
            .collect();

        Ok(resolved_table)
    }

    /// Emit warnings for all type holes showing their inferred types
    fn emit_hole_warnings(&mut self) {
        // Clone the holes list to avoid borrow checker issues
        let holes = self.type_holes.clone();
        for (node_id, span) in holes {
            if let Some(hole_scheme) = self.type_table.get(&node_id) {
                let resolved_type = match hole_scheme {
                    TypeScheme::Monotype(ty) => ty.apply(&self.context),
                    TypeScheme::Polytype { body, .. } => {
                        // For polytypes, just show the body type
                        match body.as_ref() {
                            TypeScheme::Monotype(ty) => ty.apply(&self.context),
                            _ => continue, // Skip nested polytypes for now
                        }
                    }
                };
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

            // Store resolved type in type_table for mirize
            // Need to insert for the outer pattern node ID
            let resolved_param_type = param_type.apply(&self.context);
            self.type_table.insert(param.h.id, TypeScheme::Monotype(resolved_param_type));

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
            Declaration::TypeBind(type_bind) => {
                debug!("Processing TypeBind: {}", type_bind.name);
                // Type bindings are registered in the environment during elaboration
                // For now, just skip them in type checking
                Ok(())
            }
            Declaration::ModuleBind(_) => {
                // Module bindings should be elaborated away before type checking
                // If we encounter one here, it means elaboration wasn't run or failed
                Err(CompilerError::ModuleError(
                    "Module bindings should be elaborated before type checking".to_string(),
                ))
            }
            Declaration::ModuleTypeBind(_) => {
                // Module type bindings are erased during elaboration
                // If we see one, elaboration wasn't run
                Ok(())
            }
            Declaration::Open(_) => {
                // Open declarations should be elaborated away
                Ok(())
            }
            Declaration::Import(_) => {
                // Import declarations should be resolved during elaboration
                Ok(())
            }
            Declaration::Local(inner) => {
                // Local just marks a declaration as local scope
                // Check the inner declaration
                self.check_declaration(inner)
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

        // Bind size parameters to fresh type variables
        // Size parameters like [n] in "def f [n] (xs: [n]i32): i32" are treated as type variables
        // that can unify with concrete sizes (Size(8)) or other size variables
        for size_param in &decl.size_params {
            let fresh_var = self.context.new_variable();
            type_param_bindings.insert(size_param.clone(), fresh_var);
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

            // Special handling for __applyN dispatchers (generated by defunctionalization)
            // These functions route to different lambda functions based on closure tags
            // and can't be properly type-checked because different branches have different closure types
            if decl.name.starts_with("__apply") {
                // Create a polymorphic type: closure -> arg1 -> ... -> argN -> result
                // All types are fresh variables to allow maximum flexibility
                let mut param_types = Vec::new();
                for _ in &decl.params {
                    param_types.push(self.context.new_variable());
                }
                let result_type = self.context.new_variable();

                let func_type = param_types
                    .into_iter()
                    .rev()
                    .fold(result_type, |acc, param_ty| types::function(param_ty, acc));

                // Register the dispatcher with its polymorphic type
                let type_scheme = self.generalize(&func_type);
                self.scope_stack.insert(decl.name.clone(), type_scheme);
                debug!(
                    "Registered __apply dispatcher '{}' with polymorphic type",
                    decl.name
                );
                return Ok(());
            }

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
            ExprKind::RecordLiteral(fields) => {
                let mut field_types = Vec::new();
                for (field_name, field_expr) in fields {
                    let field_ty = self.infer_expression(field_expr)?;
                    field_types.push((field_name.clone(), field_ty));
                }
                Ok(types::record(field_types))
            }
            ExprKind::TypeHole => {
                // Record this hole for warning emission after type inference completes
                self.type_holes.push((expr.h.id, expr.h.span.clone()));
                Ok(self.context.new_variable())
            }
            ExprKind::IntLiteral(_) => Ok(types::i32()),
            ExprKind::FloatLiteral(_) => Ok(types::f32()),
            ExprKind::BoolLiteral(_) => Ok(types::bool_type()),
            ExprKind::OperatorSection(_op) => {
                // Operator sections like (+), (-), etc. are functions
                // Their specific type depends on context and will be resolved via unification
                // For now, return a polymorphic function type: 'a -> 'a -> 'a
                let a = self.context.new_variable();
                let func_type = Type::arrow(a.clone(), Type::arrow(a.clone(), a));
                Ok(func_type)
            }
            ExprKind::Identifier(name) => {
                debug!("Looking up identifier '{}'", name);
                debug!("Current scope depth: {}", self.scope_stack.depth());

                // First check scope stack for variables
                if let Ok(type_scheme) = self.scope_stack.lookup(name) {
                    debug!("Found '{}' in scope stack with type: {:?}", name, type_scheme);
                    // Instantiate the type scheme to get a concrete type
                    Ok(type_scheme.instantiate(&mut self.context))
                } else if let Some(lookup) = self.builtin_registry.get(name) {
                    // Check builtin registry for builtin functions/constructors
                    use crate::builtin_registry::BuiltinLookup;
                    debug!("'{}' is a builtin", name);
                    let func_type = match lookup {
                        BuiltinLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                        BuiltinLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                    };
                    debug!("Built function type for builtin '{}': {:?}", name, func_type);
                    Ok(func_type)
                } else if crate::module_manager::ModuleManager::is_qualified_name(name) {
                    // Check for qualified module reference (e.g., "f32.sum")
                    if let Some((module_name, _func_name)) = crate::module_manager::ModuleManager::split_qualified_name(name) {
                        debug!("'{}' is a qualified name: module='{}', func='{}'", name, module_name, _func_name);
                        // Load the module and type-check its declarations
                        let module_program = self.module_manager.load_module(module_name)?;
                        let declarations = module_program.declarations.clone();
                        for decl in &declarations {
                            self.check_declaration(decl)?;
                        }

                        // Now look up the fully qualified name in scope
                        if let Ok(type_scheme) = self.scope_stack.lookup(name) {
                            Ok(type_scheme.instantiate(&mut self.context))
                        } else {
                            Err(CompilerError::UndefinedVariable(name.clone(), expr.h.span))
                        }
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

                // Unify index type with i32
                // Per spec: array index may be "any unsigned integer type"
                // We use i32 for now for compatibility
                self.context.unify(&index_type, &types::i32()).map_err(|_| {
                    CompilerError::TypeError(
                        format!(
                            "Array index must be an integer type, got {}",
                            self.format_type(&index_type.apply(&self.context))
                        ),
                        index_expr.h.span
                    )
                })?;

                // Constrain array type to be Array(n, a) even if it's currently unknown
                // This allows indexing arrays whose type is a meta-variable
                // Strip uniqueness marker - indexing a *[n]T should work like indexing [n]T
                let array_type_stripped = types::strip_unique(&array_type);
                let size_var = self.context.new_variable();
                let elem_var = self.context.new_variable();
                let want_array = Type::Constructed(TypeName::Array, vec![size_var, elem_var.clone()]);

                self.context.unify(&array_type_stripped, &want_array).map_err(|_| {
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
                // Check if the function is an overloaded builtin identifier
                // If so, perform overload resolution based on argument types
                if let ExprKind::Identifier(name) = &func.kind {
                    use crate::builtin_registry::BuiltinLookup;
                    // Clone entries to release the borrow on self.builtin_registry
                    let overload_entries = match self.builtin_registry.get(name) {
                        Some(BuiltinLookup::Overloaded(overload_set)) => {
                            Some(overload_set.entries().to_vec())
                        }
                        _ => None,
                    };

                    if let Some(entries) = overload_entries {
                        // Infer argument types first
                        let mut arg_types = Vec::new();
                        for arg in args {
                            arg_types.push(self.infer_expression(arg)?);
                        }

                        // Try each overload with backtracking
                        for entry in &entries {
                            let saved_context = self.context.clone();
                            let func_type = entry.scheme.instantiate(&mut self.context);

                            if let Some(return_type) = Self::try_unify_overload(&func_type, &arg_types, &mut self.context) {
                                // Store the types in the type table
                                // Each identifier node is unique, so storing the resolved type is correct
                                let resolved_func_type = func_type.apply(&self.context);
                                self.type_table.insert(func.h.id, TypeScheme::Monotype(resolved_func_type));
                                self.type_table.insert(expr.h.id, TypeScheme::Monotype(return_type.clone()));
                                return Ok(return_type);
                            }

                            self.context = saved_context;
                        }

                        return Err(CompilerError::TypeError(
                            format!(
                                "No matching overload for '{}' with argument types: {}",
                                name,
                                arg_types.iter().map(|t| self.format_type(t)).collect::<Vec<_>>().join(", ")
                            ),
                            expr.h.span,
                        ));
                    }
                }

                // Not an overloaded builtin, use standard application
                let func_type = self.infer_expression(func)?;

                // Use two-pass application for better lambda inference
                // This enables proper inference for expressions like (map (\x -> ...) arr)
                // or (|>) operators with lambdas
                self.apply_two_pass(func_type, args)
            }
            ExprKind::FieldAccess(inner_expr, field) => {
                // Try to extract a qualified name (e.g., f32.cos, M.N.x)
                if let Some(qual_name) = Self::try_extract_qual_name(inner_expr, field) {
                    let dotted = qual_name.to_dotted();
                    let mangled = qual_name.mangle();

                    // Check if this is a module-qualified name (mangled name exists in scope)
                    if let Ok(scheme) = self.scope_stack.lookup(&mangled) {
                        // Instantiate the type scheme
                        let ty = scheme.instantiate(&mut self.context);
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Check if this is a builtin function (e.g., f32.sin)
                    if let Some(lookup) = self.builtin_registry.get(&dotted) {
                        use crate::builtin_registry::BuiltinLookup;
                        let ty = match lookup {
                            BuiltinLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                            BuiltinLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                        };
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Qualified name not found as builtin - fall through to field access
                }

                // Not a qualified name (or wasn't found), treat as normal field access
                {
                    // Not a qualified name, proceed with normal field access
                    let expr_type = self.infer_expression(inner_expr)?;

                    // Check if this is a __lambda_name field access (closure lambda name for direct dispatch)
                    // Allow it on any type variable and return string type
                    if field == "__lambda_name" {
                        // The type checker can't verify this is actually a closure record,
                        // but the defunctionalizer guarantees it. Just return string type.
                        let ty = Type::Constructed(TypeName::Str("string".into()), vec![]);
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Apply context to resolve any type variables that have been unified
                    let expr_type = expr_type.apply(&self.context);

                    // Extract the type name from the expression type
                    // First check if it's a record with the requested field
                    if let Type::Constructed(TypeName::Record(fields), field_types) = &expr_type {
                        if let Some(field_index) = fields.get_index(field) {
                            if field_index < field_types.len() {
                                let field_type = &field_types[field_index];
                                self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                                return Ok(field_type.clone());
                            }
                        }
                    }

                    // Check if this is a tuple numeric field access (0, 1, 2, etc.)
                    if let Ok(index) = field.parse::<usize>() {
                        // Tuple field access: t.0, t.1, etc.
                        // The expr_type should be a tuple
                        if let Type::Constructed(TypeName::Tuple(_), elem_types) = &expr_type {
                            if index < elem_types.len() {
                                let field_type = elem_types[index].clone();
                                self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                                return Ok(field_type);
                            } else {
                                return Err(CompilerError::TypeError(
                                    format!("Tuple index {} out of bounds (tuple has {} elements)", index, elem_types.len()),
                                    expr.h.span
                                ));
                            }
                        } else {
                            return Err(CompilerError::TypeError(
                                format!("Numeric field access '.{}' requires a tuple type, got {}",
                                    index,
                                    self.format_type(&expr_type)),
                                expr.h.span
                            ));
                        }
                    }

                    // Check if this is a vector field access (x, y, z, w)
                    // If so, constrain the type to be a Vec even if it's currently unknown
                    if matches!(field.as_str(), "x" | "y" | "z" | "w") {
                        // Create a Vec type with unknown size and element type
                        let size_var = self.context.new_variable();
                        let elem_var = self.context.new_variable();
                        let want_vec = Type::Constructed(TypeName::Vec, vec![size_var, elem_var.clone()]);

                        // Unify to constrain expr_type to be a Vec
                        self.context.unify(&expr_type, &want_vec).map_err(|_| {
                            CompilerError::TypeError(
                                format!(
                                    "Field access '{}' requires a vector type, got {}",
                                    field,
                                    self.format_type(&expr_type.apply(&self.context))
                                ),
                                expr.h.span
                            )
                        })?;

                        // Return the element type
                        let result_ty = elem_var.apply(&self.context);
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(result_ty.clone()));
                        return Ok(result_ty);
                    }

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
                            } else if let TypeName::Record(fields) = &type_name {
                                // Handle Record type specially - look up field in the record's field map
                                if let Some(field_index) = fields.get_index(field) {
                                    if field_index < args.len() {
                                        let field_type = &args[field_index];
                                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                                        return Ok(field_type.clone());
                                    }
                                }
                                // Field not found in record
                                return Err(CompilerError::TypeError(
                                    format!(
                                        "Record type has no field '{}'. Available fields: {}",
                                        field,
                                        fields.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                                    ),
                                    expr.h.span
                                ));
                            } else {
                                // Get the type name as a string for other types
                                let type_name_str = match &type_name {
                                    TypeName::Str(s) => s.to_string(),
                                    TypeName::Float(bits) => format!("f{}", bits),
                                    TypeName::UInt(bits) => format!("u{}", bits),
                                    TypeName::Int(bits) => format!("i{}", bits),
                                    TypeName::Array => "array".to_string(),
                                    TypeName::Unsized => "unsized".to_string(),
                                    TypeName::Vec => "vec".to_string(),
                                    TypeName::Mat => "mat".to_string(),
                                    TypeName::Size(n) => n.to_string(),
                                    TypeName::SizeVar(name) => name.clone(),
                                    TypeName::UserVar(name) => format!("'{}", name),
                                    TypeName::Named(name) => name.clone(),
                                    TypeName::Unique => "unique".to_string(),
                                    TypeName::Record(_) => "record".to_string(),
                                    TypeName::Unit => "unit".to_string(),
                                    TypeName::Tuple(_) => "tuple".to_string(),
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

            ExprKind::QualifiedName(quals, name) => {
                // Qualified name like f32.sum (module function) or f32.sqrt (builtin)
                let full_name = if quals.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", quals.join("."), name)
                };

                debug!("Looking up qualified name '{}'", full_name);

                // Check if it's a builtin first (e.g., f32.sqrt, f32.sin)
                if let Some(lookup) = self.builtin_registry.get(&full_name) {
                    use crate::builtin_registry::BuiltinLookup;
                    let ty = match lookup {
                        BuiltinLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                        BuiltinLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                    };
                    self.type_table.insert(expr.h.id, polytype::TypeScheme::Monotype(ty.clone()));
                    return Ok(ty);
                }

                // Look up in scope stack (for module functions like f32.sum)
                if let Ok(type_scheme) = self.scope_stack.lookup(&full_name) {
                    debug!("Found '{}' in scope stack with type: {:?}", full_name, type_scheme);
                    Ok(type_scheme.instantiate(&mut self.context))
                } else {
                    debug!("Qualified name '{}' not found in scope or builtins", full_name);
                    Err(CompilerError::UndefinedVariable(full_name, expr.h.span))
                }
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
                // Push a new scope for loop variables
                self.scope_stack.push_scope();

                // Get or infer the type of the loop variable from init
                let loop_var_type = if let Some(init) = &loop_expr.init {
                    self.infer_expression(init)?
                } else {
                    // No init - create a fresh type variable
                    self.context.new_variable()
                };

                // Bind pattern to the loop variable type
                self.bind_pattern(&loop_expr.pattern, &loop_var_type, false)?;

                // Type check the loop form
                match &loop_expr.form {
                    LoopForm::While(cond) => {
                        // Condition must be bool
                        let cond_type = self.infer_expression(cond)?;
                        self.context.unify(&cond_type, &types::bool_type()).map_err(|e| {
                            CompilerError::TypeError(
                                format!("While condition must be bool: {:?}", e),
                                cond.h.span,
                            )
                        })?;
                    }
                    LoopForm::For(var_name, bound) => {
                        // Iteration variable is i32
                        self.scope_stack
                            .insert(var_name.clone(), TypeScheme::Monotype(types::i32()));

                        // Bound must be integer
                        let bound_type = self.infer_expression(bound)?;
                        self.context.unify(&bound_type, &types::i32()).map_err(|e| {
                            CompilerError::TypeError(
                                format!("Loop bound must be i32: {:?}", e),
                                bound.h.span,
                            )
                        })?;
                    }
                    LoopForm::ForIn(pat, arr) => {
                        // Array must be an array type
                        let arr_type = self.infer_expression(arr)?;
                        let elem_type = self.context.new_variable();
                        let size_type = self.context.new_variable();
                        let expected_arr = Type::Constructed(TypeName::Array, vec![size_type, elem_type.clone()]);

                        self.context.unify(&arr_type, &expected_arr).map_err(|e| {
                            CompilerError::TypeError(
                                format!("for-in requires an array: {:?}", e),
                                arr.h.span,
                            )
                        })?;

                        // Bind pattern to element type
                        self.bind_pattern(pat, &elem_type, false)?;
                    }
                }

                // Type check the body - its type must match the loop variable type
                let body_type = self.infer_expression(&loop_expr.body)?;
                self.context.unify(&body_type, &loop_var_type).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Loop body type must match loop variable type: {:?}", e),
                        loop_expr.body.h.span,
                    )
                })?;

                // Pop the scope
                self.scope_stack.pop_scope();

                // The loop returns the loop variable type
                Ok(loop_var_type)
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

            ExprKind::TypeAscription(expr, ascribed_ty) => {
                // Type ascription: check the inner expression and unify with ascribed type
                let expr_ty = self.infer_expression(expr)?;
                self.context.unify(&expr_ty, ascribed_ty).map_err(|e| {
                    CompilerError::TypeError(
                        format!("Type ascription failed: {:?}", e),
                        expr.h.span,
                    )
                })?;
                Ok(ascribed_ty.clone())
            }

            ExprKind::TypeCoercion(_, _) => {
                todo!("TypeCoercion not yet implemented in type checker")
            }

            ExprKind::Assert(_, _) => {
                todo!("Assert not yet implemented in type checker")
            }
        } // NEWCASESHERE - add new cases before this closing brace
        ?;

        // Store the inferred type in the type table
        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
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

                // Strip uniqueness for unification
                let arg_type_for_unify = types::strip_unique(&arg_type);
                let expected_param_for_unify = types::strip_unique(&expected_param_type);

                self.context.unify(&arg_type_for_unify, &expected_param_for_unify).map_err(|e| {
                    let error_msg = if arg.h.span.is_generated() {
                        format!(
                            "Function argument type mismatch at argument {}: {:?}\n\
                             Expected param type: {}\n\
                             Actual arg type: {}\n\
                             Generated expression: {:#?}",
                            i + 1,
                            e,
                            self.format_type(&expected_param_for_unify),
                            self.format_type(&arg_type_for_unify),
                            arg
                        )
                    } else {
                        format!("Function argument type mismatch: {:?}", e)
                    };
                    CompilerError::TypeError(error_msg, arg.h.span)
                })?;

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
                Type::Constructed(TypeName::Tuple(_), types1),
                Type::Constructed(TypeName::Str("attributed_tuple"), types2),
            )
            | (
                Type::Constructed(TypeName::Str("attributed_tuple"), types1),
                Type::Constructed(TypeName::Tuple(_), types2),
            ) => {
                types1.len() == types2.len()
                    && types1.iter().zip(types2.iter()).all(|(t1, t2)| self.types_equal(t1, t2))
            }
            // Regular case - use structural equality after applying substitution
            _ => self.types_equal(&a, &b),
        }
    }
}
