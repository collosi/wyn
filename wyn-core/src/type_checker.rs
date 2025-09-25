use crate::ast::TypeName;
use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use polytype::{Context, TypeScheme};
use std::collections::HashMap;

pub struct TypeChecker {
    scope_stack: ScopeStack<TypeScheme<TypeName>>, // Store polymorphic types
    context: Context<TypeName>,                    // Polytype unification context
    record_field_map: HashMap<(String, String), Type>, // Map (type_name, field_name) -> field_type
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    fn resolve_operator_function(&self, op: &BinaryOp, left_type: &Type, right_type: &Type) -> Result<(String, Type)> {
        // For now, require both operands to be the same type
        if !self.types_equal(left_type, right_type) {
            return Err(CompilerError::TypeError(format!(
                "Binary operator requires matching operand types, got {:?} and {:?}",
                left_type, right_type
            )));
        }

        let op_prefix = match op.op.as_str() {
            "+" => "op_add",
            "-" => "op_subtract", 
            "*" => "op_multiply",
            "/" => "op_divide",
            "==" => "op_equal",
            "!=" => "op_not_equal", 
            "<" => "op_less_than",
            ">" => "op_greater_than",
            "<=" => "op_less_than_equal",
            ">=" => "op_greater_than_equal",
            _ => return Err(CompilerError::TypeError(format!(
                "Unknown binary operator: {}", op.op
            ))),
        };

        let type_suffix = self.get_type_suffix(left_type)?;
        let op_name = format!("{}_{}", op_prefix, type_suffix);
        
        // Determine the return type based on operator
        let return_type = match op.op.as_str() {
            "==" | "!=" | "<" | ">" | "<=" | ">=" => {
                // Comparison operators return boolean
                Type::Constructed(TypeName::Str("bool"), vec![])
            }
            _ => {
                // Arithmetic operators return the same type as operands
                left_type.clone()
            }
        };
        
        Ok((op_name, return_type))
    }

    fn get_type_suffix(&self, ty: &Type) -> Result<String> {
        match ty {
            Type::Constructed(TypeName::Str(name), _) => {
                match *name {
                    "int" => Ok("i32".to_string()),
                    "float" => Ok("f32".to_string()),
                    "vec2" => Ok("vec2".to_string()),
                    "vec3" => Ok("vec3".to_string()),
                    "vec4" => Ok("vec4".to_string()),
                    "ivec2" => Ok("ivec2".to_string()),
                    "ivec3" => Ok("ivec3".to_string()),
                    "ivec4" => Ok("ivec4".to_string()),
                    _ => Err(CompilerError::TypeError(format!(
                        "Arithmetic operations not supported for type: {}", name
                    )))
                }
            }
            _ => Err(CompilerError::TypeError(format!(
                "Arithmetic operations not supported for type: {:?}", ty
            )))
        }
    }

    fn types_equal(&self, left: &Type, right: &Type) -> bool {
        match (left, right) {
            (Type::Constructed(l_name, l_args), Type::Constructed(r_name, r_args)) => {
                l_name == r_name && l_args.len() == r_args.len() && 
                l_args.iter().zip(r_args.iter()).all(|(l, r)| self.types_equal(l, r))
            }
            (Type::Variable(l_id), Type::Variable(r_id)) => l_id == r_id,
            _ => false,
        }
    }

    /// Create binary arithmetic operator type scheme: ∀t. t -> t -> t
    fn binary_arithmetic_scheme() -> TypeScheme<TypeName> {
        // Manually construct ∀t0. t0 -> t0 -> t0
        let mut ctx = Context::<TypeName>::default();
        let var_0 = ctx.new_variable();
        let arrow_type = Type::arrow(var_0.clone(), Type::arrow(var_0.clone(), var_0));

        // For now, create a monomorphic type - we'll fix polymorphic types later
        TypeScheme::Monotype(arrow_type)
    }
    pub fn new() -> Self {
        let mut checker = TypeChecker {
            scope_stack: ScopeStack::new(),
            context: Context::default(),
            record_field_map: HashMap::new(),
        };

        // Load built-in functions from builtins.wyn file
        if let Err(e) = checker.load_builtins() {
            eprintln!("Warning: Could not load builtins: {}", e);
        }

        checker
    }

    fn load_builtins(&mut self) -> Result<()> {
        // Add builtin function types directly using manual construction

        // length: ∀a. [a] -> int
        let var_a = self.context.new_variable();
        let array_type = Type::Constructed(TypeName::Str("array"), vec![var_a]);
        let int_type = Type::Constructed(TypeName::Str("int"), vec![]);
        let length_body = Type::arrow(array_type, int_type);
        let length_scheme = TypeScheme::Monotype(length_body);
        self.scope_stack.insert("length".to_string(), length_scheme);

        // map: ∀a b. (a -> b) -> [a] -> [b]
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let func_type = Type::arrow(var_a.clone(), var_b.clone());
        let input_array_type = Type::Constructed(TypeName::Str("array"), vec![var_a]);
        let output_array_type = Type::Constructed(TypeName::Str("array"), vec![var_b]);
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

        // Vector constructors
        // vec2: f32 -> f32 -> vec2
        let vec2_type = types::vec2();
        let vec2_body = Type::arrow(types::f32(), Type::arrow(types::f32(), vec2_type));
        self.scope_stack.insert("vec2".to_string(), TypeScheme::Monotype(vec2_body));
        
        // vec3: f32 -> f32 -> f32 -> vec3
        let vec3_type = types::vec3();
        let vec3_body = Type::arrow(types::f32(), 
            Type::arrow(types::f32(), 
                Type::arrow(types::f32(), vec3_type)));
        self.scope_stack.insert("vec3".to_string(), TypeScheme::Monotype(vec3_body));
        
        // vec4: f32 -> f32 -> f32 -> f32 -> vec4
        let vec4_type = types::vec4();
        let vec4_body = Type::arrow(types::f32(),
            Type::arrow(types::f32(),
                Type::arrow(types::f32(),
                    Type::arrow(types::f32(), vec4_type))));
        self.scope_stack.insert("vec4".to_string(), TypeScheme::Monotype(vec4_body));

        // Similarly for ivec2, ivec3, ivec4
        let ivec2_type = types::ivec2();
        let ivec2_body = Type::arrow(types::i32(), Type::arrow(types::i32(), ivec2_type));
        self.scope_stack.insert("ivec2".to_string(), TypeScheme::Monotype(ivec2_body));
        
        let ivec3_type = types::ivec3();
        let ivec3_body = Type::arrow(types::i32(),
            Type::arrow(types::i32(),
                Type::arrow(types::i32(), ivec3_type)));
        self.scope_stack.insert("ivec3".to_string(), TypeScheme::Monotype(ivec3_body));
        
        let ivec4_type = types::ivec4();
        let ivec4_body = Type::arrow(types::i32(),
            Type::arrow(types::i32(),
                Type::arrow(types::i32(),
                    Type::arrow(types::i32(), ivec4_type))));
        self.scope_stack.insert("ivec4".to_string(), TypeScheme::Monotype(ivec4_body));

        // Operator functions for arithmetic operations
        // f32 operations: op_add_f32, op_subtract_f32, op_multiply_f32, op_divide_f32
        let f32_binary_op = Type::arrow(types::f32(), Type::arrow(types::f32(), types::f32()));
        self.scope_stack.insert("op_add_f32".to_string(), TypeScheme::Monotype(f32_binary_op.clone()));
        self.scope_stack.insert("op_subtract_f32".to_string(), TypeScheme::Monotype(f32_binary_op.clone()));
        self.scope_stack.insert("op_multiply_f32".to_string(), TypeScheme::Monotype(f32_binary_op.clone()));
        self.scope_stack.insert("op_divide_f32".to_string(), TypeScheme::Monotype(f32_binary_op));

        // i32 operations: op_add_i32, op_subtract_i32, op_multiply_i32, op_divide_i32
        let i32_binary_op = Type::arrow(types::i32(), Type::arrow(types::i32(), types::i32()));
        self.scope_stack.insert("op_add_i32".to_string(), TypeScheme::Monotype(i32_binary_op.clone()));
        self.scope_stack.insert("op_subtract_i32".to_string(), TypeScheme::Monotype(i32_binary_op.clone()));
        self.scope_stack.insert("op_multiply_i32".to_string(), TypeScheme::Monotype(i32_binary_op.clone()));
        self.scope_stack.insert("op_divide_i32".to_string(), TypeScheme::Monotype(i32_binary_op));

        // vec2 operations
        let vec2_binary_op = Type::arrow(types::vec2(), Type::arrow(types::vec2(), types::vec2()));
        self.scope_stack.insert("op_add_vec2".to_string(), TypeScheme::Monotype(vec2_binary_op.clone()));
        self.scope_stack.insert("op_subtract_vec2".to_string(), TypeScheme::Monotype(vec2_binary_op.clone()));
        self.scope_stack.insert("op_multiply_vec2".to_string(), TypeScheme::Monotype(vec2_binary_op.clone()));
        self.scope_stack.insert("op_divide_vec2".to_string(), TypeScheme::Monotype(vec2_binary_op));

        // vec3 operations
        let vec3_binary_op = Type::arrow(types::vec3(), Type::arrow(types::vec3(), types::vec3()));
        self.scope_stack.insert("op_add_vec3".to_string(), TypeScheme::Monotype(vec3_binary_op.clone()));
        self.scope_stack.insert("op_subtract_vec3".to_string(), TypeScheme::Monotype(vec3_binary_op.clone()));
        self.scope_stack.insert("op_multiply_vec3".to_string(), TypeScheme::Monotype(vec3_binary_op.clone()));
        self.scope_stack.insert("op_divide_vec3".to_string(), TypeScheme::Monotype(vec3_binary_op));

        // vec4 operations
        let vec4_binary_op = Type::arrow(types::vec4(), Type::arrow(types::vec4(), types::vec4()));
        self.scope_stack.insert("op_add_vec4".to_string(), TypeScheme::Monotype(vec4_binary_op.clone()));
        self.scope_stack.insert("op_subtract_vec4".to_string(), TypeScheme::Monotype(vec4_binary_op.clone()));
        self.scope_stack.insert("op_multiply_vec4".to_string(), TypeScheme::Monotype(vec4_binary_op.clone()));
        self.scope_stack.insert("op_divide_vec4".to_string(), TypeScheme::Monotype(vec4_binary_op));

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
        // f32 vector fields
        // vec2 fields
        self.record_field_map.insert(("vec2".to_string(), "x".to_string()), types::f32());
        self.record_field_map.insert(("vec2".to_string(), "y".to_string()), types::f32());
        
        // vec3 fields  
        self.record_field_map.insert(("vec3".to_string(), "x".to_string()), types::f32());
        self.record_field_map.insert(("vec3".to_string(), "y".to_string()), types::f32());
        self.record_field_map.insert(("vec3".to_string(), "z".to_string()), types::f32());
        
        // vec4 fields
        self.record_field_map.insert(("vec4".to_string(), "x".to_string()), types::f32());
        self.record_field_map.insert(("vec4".to_string(), "y".to_string()), types::f32());
        self.record_field_map.insert(("vec4".to_string(), "z".to_string()), types::f32());
        self.record_field_map.insert(("vec4".to_string(), "w".to_string()), types::f32());

        // i32 vector fields
        // ivec2 fields
        self.record_field_map.insert(("ivec2".to_string(), "x".to_string()), types::i32());
        self.record_field_map.insert(("ivec2".to_string(), "y".to_string()), types::i32());
        
        // ivec3 fields
        self.record_field_map.insert(("ivec3".to_string(), "x".to_string()), types::i32());
        self.record_field_map.insert(("ivec3".to_string(), "y".to_string()), types::i32());
        self.record_field_map.insert(("ivec3".to_string(), "z".to_string()), types::i32());
        
        // ivec4 fields
        self.record_field_map.insert(("ivec4".to_string(), "x".to_string()), types::i32());
        self.record_field_map.insert(("ivec4".to_string(), "y".to_string()), types::i32());
        self.record_field_map.insert(("ivec4".to_string(), "z".to_string()), types::i32());
        self.record_field_map.insert(("ivec4".to_string(), "w".to_string()), types::i32());

        // TODO: Add other vector types (uvec, bvec, dvec, f16vec) when we have proper types for them
    }

    /// Register a record type with its field mappings
    pub fn register_record_type(&mut self, type_name: &str, fields: Vec<(String, Type)>) {
        for (field_name, field_type) in fields {
            self.record_field_map.insert((type_name.to_string(), field_name), field_type);
        }
    }

    pub fn check_program(&mut self, program: &Program) -> Result<()> {
        // First pass: collect all function declarations
        for decl in &program.declarations {
            if let Declaration::Decl(decl_node) = decl {
                if decl_node.keyword == "def" && !decl_node.params.is_empty() {
                    // This is a function definition
                    // Create a placeholder type for this function
                    let func_type = self.context.new_variable();
                    let type_scheme = TypeScheme::Monotype(func_type);
                    self.scope_stack.insert(decl_node.name.clone(), type_scheme);
                }
            }
        }

        // Second pass: infer types for all declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        Ok(())
    }

    fn check_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(decl_node) => {
                println!("DEBUG: Checking {} declaration: {}", decl_node.keyword, decl_node.name);
                self.check_decl(decl_node)
            }
            Declaration::Val(val_decl) => {
                println!("DEBUG: Checking Val declaration: {}", val_decl.name);
                self.check_val_decl(val_decl)
            }
        }
    }

    fn check_decl(&mut self, decl: &Decl) -> Result<()> {
        // Handle uniform declarations specially
        if decl.attributes.contains(&Attribute::Uniform) {
            // Uniforms must have a type annotation and no real initializer
            if decl.ty.is_none() {
                return Err(CompilerError::TypeError(format!(
                    "Uniform declaration '{}' must have a type annotation",
                    decl.name
                )));
            }
            
            // Check that the body is the placeholder (indicating no initializer was provided)
            if !matches!(decl.body, Expression::Identifier(ref id) if id == "__uniform_placeholder") {
                return Err(CompilerError::TypeError(format!(
                    "Uniform declaration '{}' cannot have an initializer value. Uniforms must be provided by the host application.",
                    decl.name
                )));
            }
            
            // Add the uniform to scope with its declared type
            let uniform_type = decl.ty.as_ref().unwrap().clone();
            let type_scheme = TypeScheme::Monotype(uniform_type);
            self.scope_stack.insert(decl.name.clone(), type_scheme);
            println!("DEBUG: Inserting uniform variable '{}' into scope", decl.name);
            
            return Ok(());
        }

        if decl.params.is_empty() {
            // Variable declaration: let/def name: type = value or let/def name = value
            let expr_type = self.infer_expression(&decl.body)?;

            if let Some(declared_type) = &decl.ty {
                if !self.types_match(&expr_type, declared_type) {
                    return Err(CompilerError::TypeError(format!(
                        "Type mismatch: expected {:?}, got {:?}",
                        declared_type, expr_type
                    )));
                }
            }

            // Add to scope - use declared type if available, otherwise inferred type
            let stored_type = decl.ty.as_ref().unwrap_or(&expr_type).clone();
            let type_scheme = TypeScheme::Monotype(stored_type.clone());
            println!("DEBUG: Inserting variable '{}' into scope", decl.name);
            self.scope_stack.insert(decl.name.clone(), type_scheme);
            println!("Inferred type for {}: {}", decl.name, stored_type);
        } else {
            // Function declaration: let/def name param1 param2 = body
            // Create type variables or use explicit types for parameters
            let param_types: Vec<Type> = decl.params
                .iter()
                .map(|p| match p {
                    DeclParam::Untyped(_) => self.context.new_variable(),
                    DeclParam::Typed(param) => param.ty.clone(),
                })
                .collect();

            // Push new scope for function parameters
            self.scope_stack.push_scope();

            // Add parameters to scope
            for (param, param_type) in decl.params.iter().zip(param_types.iter()) {
                let param_name = match param {
                    DeclParam::Untyped(name) => name.clone(),
                    DeclParam::Typed(p) => p.name.clone(),
                };
                let type_scheme = TypeScheme::Monotype(param_type.clone());
                self.scope_stack.insert(param_name, type_scheme);
            }

            // Infer body type
            let body_type = self.infer_expression(&decl.body)?;

            // Pop parameter scope
            self.scope_stack.pop_scope();

            // Build function type: param1 -> param2 -> ... -> body_type
            let func_type = param_types
                .into_iter()
                .rev()
                .fold(body_type, |acc, param_ty| types::function(param_ty, acc));

            // Update scope with inferred type
            let type_scheme = TypeScheme::Monotype(func_type.clone());
            self.scope_stack.insert(decl.name.clone(), type_scheme);

            println!("Inferred type for {}: {}", decl.name, func_type);
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
        match expr {
            Expression::IntLiteral(_) => Ok(types::i32()),
            Expression::FloatLiteral(_) => Ok(types::f32()),
            Expression::Identifier(name) => {
                let type_scheme = self
                    .scope_stack
                    .lookup(name)
                    .ok_or_else(|| {
                        println!("DEBUG: Variable lookup failed for '{}'", name);
                        CompilerError::UndefinedVariable(name.clone())
                    })?;
                // Instantiate the type scheme to get a concrete type
                Ok(type_scheme.instantiate(&mut self.context))
            }
            Expression::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    return Err(CompilerError::TypeError(
                        "Cannot infer type of empty array".to_string(),
                    ));
                }

                let first_type = self.infer_expression(&elements[0])?;
                for elem in &elements[1..] {
                    let elem_type = self.infer_expression(elem)?;
                    if !self.types_match(&elem_type, &first_type) {
                        return Err(CompilerError::TypeError(
                            "Array elements must have the same type".to_string(),
                        ));
                    }
                }

                Ok(types::sized_array(elements.len(), first_type))
            }
            Expression::ArrayIndex(array_expr, _index_expr) => {
                let array_type = self.infer_expression(array_expr)?;
                Ok(match array_type {
                    Type::Constructed(name, args)
                        if matches!(name, TypeName::Str("array") | TypeName::Array("array", _)) =>
                    {
                        args.into_iter().next().unwrap_or_else(|| types::i32())
                    }
                    _ => {
                        return Err(CompilerError::TypeError(format!(
                            "Cannot index non-array type: got {:?}",
                            array_type
                        )))
                    }
                })
            }
            Expression::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;

                // Map operator to appropriate op_* function based on operand types
                let (op_name, _expected_arg_type) = self.resolve_operator_function(op, &left_type, &right_type)?;
                
                // Look up the operator function
                let type_scheme = self
                    .scope_stack
                    .lookup(&op_name)
                    .ok_or_else(|| CompilerError::UndefinedVariable(op_name.clone()))?;
                let func_type = type_scheme.instantiate(&mut self.context);

                // Type check as if this were a function call: op_func left_type right_type
                let mut result_type = func_type.clone();
                for arg_type in [left_type, right_type] {
                    match result_type {
                        Type::Constructed(TypeName::Str("->"), args) if args.len() == 2 => {
                            let param_type = &args[0];
                            let return_type = args[1].clone();
                            self.context.unify(param_type, &arg_type).map_err(|e| {
                                CompilerError::TypeError(format!(
                                    "Type mismatch in {} operator: expected {:?}, got {:?}: {}",
                                    op_name, param_type, arg_type, e
                                ))
                            })?;
                            result_type = return_type;
                        }
                        _ => {
                            return Err(CompilerError::TypeError(format!(
                                "Operator function {} is not a function type",
                                op_name
                            )));
                        }
                    }
                }

                Ok(result_type.apply(&self.context))
            }
            Expression::FunctionCall(func_name, args) => {
                // Get function type scheme and instantiate it
                let type_scheme = self
                    .scope_stack
                    .lookup(func_name)
                    .ok_or_else(|| CompilerError::UndefinedVariable(func_name.clone()))?;
                let mut func_type = type_scheme.instantiate(&mut self.context);

                // Apply function to each argument using unification
                for arg in args {
                    let arg_type = self.infer_expression(arg)?;

                    // Create a fresh result type variable
                    let result_type = self.context.new_variable();

                    // Expected function type: arg_type -> result_type
                    let expected_func_type = Type::arrow(arg_type, result_type.clone());

                    // Unify the function type with expected
                    self.context
                        .unify(&func_type, &expected_func_type)
                        .map_err(|e| {
                            CompilerError::TypeError(format!("Function call type error: {:?}", e))
                        })?;

                    // Update func_type to result_type for the next argument (currying)
                    func_type = result_type;
                }

                Ok(func_type.apply(&self.context))
            }
            Expression::Tuple(elements) => {
                let elem_types: Result<Vec<Type>> =
                    elements.iter().map(|e| self.infer_expression(e)).collect();

                Ok(types::tuple(elem_types?))
            }
            Expression::Lambda(lambda) => {
                // Push new scope for lambda parameters
                self.scope_stack.push_scope();

                // Add parameters to scope with their types (or fresh type variables)
                for param in &lambda.params {
                    let param_type = param.ty.clone().unwrap_or_else(|| {
                        self.context.new_variable() // Use polytype's context to create fresh variables
                    });
                    let type_scheme = TypeScheme::Monotype(param_type);
                    self.scope_stack.insert(param.name.clone(), type_scheme);
                }

                // Type check the lambda body with parameters in scope
                let body_type = self.infer_expression(&lambda.body)?;
                let return_type = lambda.return_type.clone().unwrap_or(body_type);

                // Pop parameter scope
                self.scope_stack.pop_scope();

                // For multiple parameters, create nested function types
                let mut func_type = return_type;
                for param in lambda.params.iter().rev() {
                    let param_type = param
                        .ty
                        .clone()
                        .unwrap_or_else(|| self.context.new_variable());
                    func_type = types::function(param_type, func_type);
                }

                Ok(func_type)
            }
            Expression::LetIn(let_in) => {
                // Infer type of the value expression
                let value_type = self.infer_expression(&let_in.value)?;

                // Check type annotation if present
                if let Some(declared_type) = &let_in.ty {
                    if !self.types_match(&value_type, declared_type) {
                        return Err(CompilerError::TypeError(format!(
                            "Type mismatch in let binding: expected {:?}, got {:?}",
                            declared_type, value_type
                        )));
                    }
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
            Expression::Application(func, args) => {
                let mut func_type = self.infer_expression(func)?;

                // Apply function to each argument
                for arg in args {
                    let arg_type = self.infer_expression(arg)?;

                    // Create a fresh result type variable
                    let result_type = self.context.new_variable();

                    // Expected function type: arg_type -> result_type
                    let expected_func_type = Type::arrow(arg_type, result_type.clone());

                    // Unify the function type with expected
                    self.context
                        .unify(&func_type, &expected_func_type)
                        .map_err(|e| {
                            CompilerError::TypeError(format!(
                                "Function application type error: {:?}",
                                e
                            ))
                        })?;

                    // Update func_type to result_type for the next argument (currying)
                    func_type = result_type;
                }

                Ok(func_type.apply(&self.context))
            }
            Expression::FieldAccess(expr, field) => {
                let expr_type = self.infer_expression(expr)?;
                
                // Extract the type name from the expression type
                match expr_type {
                    Type::Constructed(TypeName::Str(type_name), _) => {
                        // Look up the field in our record field mapping
                        if let Some(field_type) = self.record_field_map.get(&(type_name.to_string(), field.clone())) {
                            Ok(field_type.clone())
                        } else {
                            Err(CompilerError::TypeError(format!(
                                "Type '{}' has no field '{}'",
                                type_name, field
                            )))
                        }
                    }
                    _ => {
                        Err(CompilerError::TypeError(format!(
                            "Field access '{}' not supported on type {:?}",
                            field, expr_type
                        )))
                    }
                }
            }
            Expression::If(if_expr) => {
                // Infer condition type - should be bool
                let condition_ty = self.infer_expression(&if_expr.condition)?;
                let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
                
                // Unify condition with bool type
                self.context.unify(&condition_ty, &bool_ty)
                    .map_err(|_| CompilerError::TypeError(format!(
                        "If condition must be boolean, got: {}", condition_ty
                    )))?;
                
                // Infer then and else branch types - they must be the same
                let then_ty = self.infer_expression(&if_expr.then_branch)?;
                let else_ty = self.infer_expression(&if_expr.else_branch)?;
                
                // Unify then and else types
                self.context.unify(&then_ty, &else_ty)
                    .map_err(|_| CompilerError::TypeError(format!(
                        "If branches have incompatible types: then={}, else={}", then_ty, else_ty
                    )))?;
                
                Ok(then_ty)
            }
        }
    }

    // Removed: fresh_var - now using polytype's context.new_variable()

    fn types_match(&mut self, t1: &Type, t2: &Type) -> bool {
        // Handle attributed_tuple vs tuple matching
        match (t1, t2) {
            // Allow regular tuple to match attributed_tuple if component types match
            (Type::Constructed(TypeName::Str("tuple"), actual_types),
             Type::Constructed(TypeName::Str("attributed_tuple"), expected_types)) => {
                expected_types.len() == actual_types.len() &&
                expected_types.iter().zip(actual_types.iter())
                    .all(|(e, a)| self.context.unify(a, e).is_ok())
            }
            // Regular case - use polytype's unification for proper type matching
            _ => self.context.unify(t1, t2).is_ok()
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
            CompilerError::UndefinedVariable(_)
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
        match checker.check_program(&program) {
            Ok(_) => {
                println!("Type checking succeeded!");

                // Check that zip_arrays has the expected type
                if let Some(func_type) = checker.scope_stack.lookup("zip_arrays") {
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
