use crate::ast::TypeName;
use crate::ast::*;
use crate::builtin_registry::{BuiltinDescriptor, TypeVarGenerator};
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use log::debug;
use polytype::{Context, TypeScheme};
use std::collections::HashMap;

// Implement TypeVarGenerator for Context
impl TypeVarGenerator for Context<TypeName> {
    fn new_variable(&mut self) -> Type {
        Context::new_variable(self)
    }
}

pub struct TypeChecker {
    scope_stack: ScopeStack<TypeScheme<TypeName>>, // Store polymorphic types
    context: Context<TypeName>,                    // Polytype unification context
    record_field_map: HashMap<(String, String), Type>, // Map (type_name, field_name) -> field_type
    builtin_registry: crate::builtin_registry::BuiltinRegistry, // Centralized builtin registry
    type_table: HashMap<crate::ast::NodeId, Type>, // Maps expression NodeId to inferred type
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
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
    fn format_type(&self, ty: &Type) -> String {
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
                _ => ty.clone(),
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

        // length: ∀a. [a] -> int
        let var_a = self.context.new_variable();
        let array_type = Type::Constructed(TypeName::Str("array"), vec![var_a]);
        let int_type = Type::Constructed(TypeName::Str("int"), vec![]);
        let length_body = Type::arrow(array_type, int_type);
        let length_scheme = TypeScheme::Monotype(length_body);
        self.scope_stack.insert("length".to_string(), length_scheme);

        // map: ∀a b n. (a -> b) -> *Array(n, a) -> Array(n, b)
        // The input array is consumed (unique), output is fresh
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let var_n = self.context.new_variable(); // Array size variable
        let func_type = Type::arrow(var_a.clone(), var_b.clone());
        let input_array_type =
            types::unique(Type::Constructed(TypeName::Array, vec![var_n.clone(), var_a]));
        let output_array_type = Type::Constructed(TypeName::Array, vec![var_n, var_b]);
        let map_arrow1 = Type::arrow(input_array_type, output_array_type);
        let map_body = Type::arrow(func_type, map_arrow1);
        let map_scheme = TypeScheme::Monotype(map_body);
        self.scope_stack.insert("map".to_string(), map_scheme);

        // zip: ∀a b. [a] -> [b] -> [(a, b)]
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let array_a_type = Type::Constructed(TypeName::Str("array"), vec![var_a.clone()]);
        let array_b_type = Type::Constructed(TypeName::Str("array"), vec![var_b.clone()]);
        let tuple_type = Type::Constructed(TypeName::Str("tuple"), vec![var_a, var_b]);
        let result_array_type = Type::Constructed(TypeName::Str("array"), vec![tuple_type]);
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

        // Vector operations
        // dot: ∀a b. a -> a -> b
        // Polymorphic: takes two values of same type, returns a value (likely scalar)
        // The SPIR-V validator will ensure the types are actually compatible
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let dot_body = Type::arrow(var_a.clone(), Type::arrow(var_a, var_b));
        self.scope_stack.insert("dot".to_string(), TypeScheme::Monotype(dot_body));

        // length: ∀a b. a -> b
        // Polymorphic: takes a vector, returns a scalar
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let length_body = Type::arrow(var_a, var_b);
        self.scope_stack.insert("length".to_string(), TypeScheme::Monotype(length_body));

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

        Ok(self.type_table.clone())
    }

    /// Helper to type check a function body with parameters in scope
    /// Returns (param_types, body_type)
    fn check_function_with_params(
        &mut self,
        params: &[Pattern],
        body: &Expression,
    ) -> Result<(Vec<Type>, Type)> {
        // Create type variables or use explicit types for parameters
        let param_types: Vec<Type> = params
            .iter()
            .map(|p| p.pattern_type().cloned().unwrap_or_else(|| self.context.new_variable()))
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
                    CompilerError::TypeError(format!(
                        "Complex patterns in function parameters not yet supported"
                    ))
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
                    self.check_function_with_params(&entry.params, &entry.body)?;
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
        if decl.params.is_empty() {
            // Variable or entry point declaration: let/def name: type = value or let/def name = value
            let expr_type = self.infer_expression(&decl.body)?;

            if let Some(declared_type) = &decl.ty {
                if !self.types_match(&expr_type, declared_type) {
                    return Err(CompilerError::TypeError(format!(
                        "Type mismatch: expected {}, got {}",
                        self.format_type(declared_type),
                        self.format_type(&expr_type)
                    )));
                }
            }

            // Add to scope - use declared type if available, otherwise inferred type
            let stored_type = decl.ty.as_ref().unwrap_or(&expr_type).clone();
            let type_scheme = TypeScheme::Monotype(stored_type.clone());
            debug!("Inserting variable '{}' into scope", decl.name);
            self.scope_stack.insert(decl.name.clone(), type_scheme);
            debug!("Inferred type for {}: {}", decl.name, stored_type);
        } else {
            // Function declaration: let/def name param1 param2 = body
            let (param_types, body_type) = self.check_function_with_params(&decl.params, &decl.body)?;
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
                // When a function has parameters, decl.ty is just the return type annotation
                // Check the body type matches the declared return type
                if !decl.params.is_empty() {
                    if !self.types_match(&body_type, declared_type) {
                        return Err(CompilerError::TypeError(format!(
                            "Function return type mismatch for '{}': expected {}, got {}",
                            decl.name,
                            self.format_type(declared_type),
                            self.format_type(&body_type)
                        )));
                    }
                } else {
                    // For functions without parameters, ty should be the full type
                    // But currently we're storing just the value type
                    // Since func_type for parameterless functions is just the body type,
                    // we can just check body_type against declared_type
                    self.context.unify(&body_type, declared_type).map_err(|_| {
                        CompilerError::TypeError(format!(
                            "Type mismatch for '{}': declared {}, inferred {}",
                            decl.name,
                            self.format_type(declared_type),
                            self.format_type(&body_type)
                        ))
                    })?;
                }
            }

            // Entry points are now handled separately via Declaration::Entry
            // Regular Decl no longer has attributed return types

            // Update scope with inferred type
            let type_scheme = TypeScheme::Monotype(func_type.clone());
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
                        Err(CompilerError::UndefinedVariable(name.clone()))
                    }
                } else {
                    // Not found anywhere
                    debug!("Variable lookup failed for '{}' - not in scope or builtins", name);
                    debug!("Scope stack contents: {:?}", self.scope_stack);
                    Err(CompilerError::UndefinedVariable(name.clone()))
                }
            }
            ExprKind::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    Err(CompilerError::TypeError(
                        "Cannot infer type of empty array".to_string(),
                    ))
                } else {
                    let first_type = self.infer_expression(&elements[0])?;
                    for elem in &elements[1..] {
                        let elem_type = self.infer_expression(elem)?;
                        self.context.unify(&elem_type, &first_type).map_err(|_| {
                            CompilerError::TypeError(format!(
                                "Array elements must have the same type, expected {}, got {}",
                                first_type, elem_type
                            ))
                        })?;
                    }

                    Ok(types::sized_array(elements.len(), first_type))
                }
            }
            ExprKind::ArrayIndex(array_expr, index_expr) => {
                let array_type = self.infer_expression(array_expr)?;
                let index_type = self.infer_expression(index_expr)?;

                // Check index type is i32
                self.context.unify(&index_type, &types::i32()).map_err(|_| {
                    CompilerError::TypeError(format!("Array index must be i32, got {}", index_type))
                })?;

                match &array_type {
                    Type::Constructed(TypeName::Array, args) => {
                        // Array type is: Array(Size(n), elem_type)
                        // So element type is at index 1
                        args.get(1).cloned().ok_or_else(|| {
                            CompilerError::TypeError(format!(
                                "Array type has no element type: {}",
                                array_type
                            ))
                        })
                    }
                    _ => Err(CompilerError::TypeError(format!(
                        "Cannot index non-array type: got {}",
                        array_type
                    ))),
                }
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;

                // Check that both operands have compatible types
                self.context.unify(&left_type, &right_type).map_err(|_| {
                    CompilerError::TypeError(format!(
                        "Binary operator '{}' requires operands of the same type, got {} and {}",
                        op.op, left_type, right_type
                    ))
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
                    _ => Err(CompilerError::TypeError(format!(
                        "Unknown binary operator: {}",
                        op.op
                    ))),
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
                            Err(CompilerError::UndefinedVariable(func_name.clone()))
                        }
                    } else {
                        Err(CompilerError::UndefinedVariable(func_name.clone()))
                    };

                let mut func_type = func_type_result?;

                // Apply function to each argument using unification
                for arg in args {
                    let arg_type = self.infer_expression(arg)?;

                    // Extract parameter type from function type (apply arrow destructor)
                    // The function type should be: param_type -> rest_type
                    let func_type_applied = func_type.apply(&self.context);

                    // Check if the expected parameter is unique (for consumption tracking)
                    // We need to peek into the arrow type structure
                    let expects_unique =
                        if let Type::Constructed(TypeName::Str("arrow"), arrow_args) = &func_type_applied {
                            if let Some(param_type) = arrow_args.first() {
                                types::is_unique(param_type)
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                    // Create a fresh result type variable
                    let result_type = self.context.new_variable();

                    // Strip uniqueness from arg_type for unification
                    // This allows non-unique values to be passed to unique parameters
                    let arg_type_for_unify = types::strip_unique(&arg_type);

                    // Also strip uniqueness from the function type before unifying
                    // This ensures unique parameters can accept non-unique arguments
                    let func_type_for_unify = types::strip_unique(&func_type);

                    // Expected function type: arg_type_for_unify -> result_type
                    let expected_func_type = Type::arrow(arg_type_for_unify, result_type.clone());

                    // Unify the function type with expected (with uniqueness stripped)
                    self.context.unify(&func_type_for_unify, &expected_func_type).map_err(|e| {
                        CompilerError::TypeError(format!("Function call type error: {:?}", e))
                    })?;

                    // If the parameter expects unique ownership, mark the variable as consumed
                    if expects_unique {
                        if let ExprKind::Identifier(var_name) = &arg.kind {
                            self.scope_stack.mark_consumed(var_name).map_err(|e| {
                                CompilerError::TypeError(format!(
                                    "Cannot consume variable '{}': {}",
                                    var_name, e
                                ))
                            })?;
                        }
                        // Note: literals and other expressions can be consumed without tracking
                    }

                    // Update func_type to result_type for the next argument (currying)
                    func_type = result_type;
                }

                Ok(func_type.apply(&self.context))
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
                for param in &lambda.params {
                    let param_type = param.pattern_type().cloned().unwrap_or_else(|| {
                        self.context.new_variable() // Use polytype's context to create fresh variables
                    });
                    let type_scheme = TypeScheme::Monotype(param_type.clone());

                    // For now, only support simple name patterns in lambda parameters
                    let param_name = param.simple_name().ok_or_else(|| {
                        CompilerError::TypeError("Complex patterns in lambda parameters not yet supported".to_string())
                    })?;
                    self.scope_stack.insert(param_name.to_string(), type_scheme);
                }

                // Type check the lambda body with parameters in scope
                let body_type = self.infer_expression(&lambda.body)?;
                let return_type = lambda.return_type.clone().unwrap_or(body_type);

                // Pop parameter scope
                self.scope_stack.pop_scope();

                // For multiple parameters, create nested function types
                let mut func_type = return_type;
                for param in lambda.params.iter().rev() {
                    let param_type = param.pattern_type().cloned().unwrap_or_else(|| self.context.new_variable());
                    func_type = types::function(param_type, func_type);
                }

                Ok(func_type)
            }
            ExprKind::LetIn(let_in) => {
                // Infer type of the value expression
                let value_type = self.infer_expression(&let_in.value)?;

                // Check type annotation if present
                if let Some(declared_type) = &let_in.ty {
                    self.context.unify(&value_type, declared_type).map_err(|_| {
                        CompilerError::TypeError(format!(
                            "Type mismatch in let binding: expected {}, got {}",
                            declared_type, value_type
                        ))
                    })?;
                }

                // Push new scope and add binding
                self.scope_stack.push_scope();
                let bound_type = let_in.ty.as_ref().unwrap_or(&value_type).clone();
                let type_scheme = TypeScheme::Monotype(bound_type);
                self.scope_stack.insert(let_in.name.clone(), type_scheme);

                // Infer type of body expression
                let body_type = self.infer_expression(&let_in.body)?;

                // Pop scope
                self.scope_stack.pop_scope();

                Ok(body_type)
            }
            ExprKind::Application(func, args) => {
                let mut func_type = self.infer_expression(func)?;

                // Apply function to each argument
                for arg in args {
                    let arg_type = self.infer_expression(arg)?;

                    // Create a fresh result type variable
                    let result_type = self.context.new_variable();

                    // Expected function type: arg_type -> result_type
                    let expected_func_type = Type::arrow(arg_type, result_type.clone());

                    // Unify the function type with expected
                    self.context.unify(&func_type, &expected_func_type).map_err(|e| {
                        CompilerError::TypeError(format!("Function application type error: {:?}", e))
                    })?;

                    // Update func_type to result_type for the next argument (currying)
                    func_type = result_type;
                }

                Ok(func_type.apply(&self.context))
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

                    // Extract the type name from the expression type
                    match expr_type {
                        Type::Constructed(type_name, ref args) => {
                            // Handle Vec type specially for field access
                            if matches!(type_name, TypeName::Vec) {
                                // Vec(size, element_type)
                                // Fields x, y, z, w return the element type
                                if let Some(element_type) = args.get(1) {
                                    // Check if field is valid (x, y, z, w)
                                    if matches!(field.as_str(), "x" | "y" | "z" | "w") {
                                        Ok(element_type.clone())
                                    } else {
                                        Err(CompilerError::TypeError(format!(
                                            "Vector type has no field '{}'",
                                            field
                                        )))
                                    }
                                } else {
                                    Err(CompilerError::TypeError(
                                        "Malformed Vec type - missing element type".to_string(),
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
                                    Err(CompilerError::TypeError(format!(
                                        "Type '{}' has no field '{}'",
                                        type_name_str, field
                                    )))
                                }
                            }
                        }
                        _ => Err(CompilerError::TypeError(format!(
                            "Field access '{}' not supported on type {}",
                            field, expr_type
                        ))),
                    }
                }
            }
            ExprKind::If(if_expr) => {
                // Infer condition type - should be bool
                let condition_ty = self.infer_expression(&if_expr.condition)?;
                let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

                // Unify condition with bool type
                self.context.unify(&condition_ty, &bool_ty).map_err(|_| {
                    CompilerError::TypeError(format!("If condition must be boolean, got: {}", condition_ty))
                })?;

                // Infer then and else branch types - they must be the same
                let then_ty = self.infer_expression(&if_expr.then_branch)?;
                let else_ty = self.infer_expression(&if_expr.else_branch)?;

                // Unify then and else types
                self.context.unify(&then_ty, &else_ty).map_err(|_| {
                    CompilerError::TypeError(format!(
                        "If branches have incompatible types: then={}, else={}",
                        then_ty, else_ty
                    ))
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
                            CompilerError::TypeError(format!(
                                "Logical not (!) requires bool operand, got {:?}",
                                operand_type
                            ))
                        })?;
                        Ok(bool_ty)
                    }
                    _ => Err(CompilerError::TypeError(format!(
                        "Unknown unary operator: {}",
                        op.op
                    ))),
                }
            }

            ExprKind::Loop(_) => {
                todo!("Loop not yet implemented in type checker")
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
                    CompilerError::TypeError(format!("Pipe operator type error: {:?}", e))
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

    fn types_match(&self, t1: &Type, t2: &Type) -> bool {
        // Apply current substitution without mutating context
        let a = t1.apply(&self.context);
        let b = t2.apply(&self.context);

        // Handle attributed_tuple vs tuple matching
        match (&a, &b) {
            // Allow regular tuple to match attributed_tuple if component types match
            (
                Type::Constructed(TypeName::Str("tuple"), actual_types),
                Type::Constructed(TypeName::Str("attributed_tuple"), expected_types),
            ) => {
                expected_types.len() == actual_types.len()
                    && expected_types.iter().zip(actual_types.iter()).all(|(e, a)| self.types_equal(a, e))
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
        assert!(matches!(result.unwrap_err(), CompilerError::UndefinedVariable(_)));
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
}
