use crate::ast::TypeName;
use crate::ast::*;
use crate::builtins::BuiltinManager;
use crate::error::{CompilerError, Result};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{InitializationConfig, Target, TargetTriple, TargetMachine, RelocMode, CodeModel, FileType};
use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum};
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue, GlobalValue};
use inkwell::AddressSpace;
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    builtin_manager: BuiltinManager<'ctx>,
    variable_cache: HashMap<String, PointerValue<'ctx>>,
    variable_types: HashMap<String, Type>,
    function_cache: HashMap<String, FunctionValue<'ctx>>,
    type_cache: HashMap<Type, BasicTypeEnum<'ctx>>,
    current_function: Option<FunctionValue<'ctx>>,
    entry_points: Vec<(String, spirv::ExecutionModel)>,
    uniform_globals: HashMap<String, GlobalValue<'ctx>>,
    output_globals: HashMap<String, (GlobalValue<'ctx>, u32)>, // (global, location)
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        // Set up SPIR-V target triple
        module.set_triple(&TargetTriple::create("spirv64-unknown-unknown"));

        let builtin_manager = BuiltinManager::new(context);

        CodeGenerator {
            context,
            module,
            builder,
            builtin_manager,
            variable_cache: HashMap::new(),
            variable_types: HashMap::new(),
            function_cache: HashMap::new(),
            type_cache: HashMap::new(),
            current_function: None,
            entry_points: Vec::new(),
            uniform_globals: HashMap::new(),
            output_globals: HashMap::new(),
        }
    }

    pub fn generate(mut self, program: &Program) -> Result<Vec<u32>> {
        // Load builtin functions into the module
        self.builtin_manager
            .load_builtins_into_module(&self.module)?;

        // Process all declarations
        for decl in &program.declarations {
            self.generate_declaration(decl)?;
        }

        // Clone entry points to avoid borrow issues
        let entry_points = self.entry_points.clone();

        // Add SPIR-V specific metadata for entry points
        for (name, exec_model) in &entry_points {
            self.add_spirv_entry_point_metadata(name, exec_model)?;
        }

        // Add SPIR-V decorations for uniforms and outputs
        self.add_spirv_decorations()?;

        // Verify the module
        if let Err(e) = self.module.verify() {
            return Err(CompilerError::SpirvError(format!(
                "LLVM module verification failed: {}",
                e
            )));
        }

        // Convert LLVM IR to SPIR-V
        self.emit_spirv()
    }

    fn generate_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(decl_node) => {
                // Check if this is an entry point (has Vertex or Fragment attribute)
                let is_entry_point = decl_node.attributes.iter().any(|attr| 
                    matches!(attr, Attribute::Vertex | Attribute::Fragment)
                );
                
                if is_entry_point {
                    self.generate_entry_point(decl_node)
                } else if decl_node.params.is_empty() {
                    // Check if this is a uniform declaration
                    let is_uniform = decl_node.attributes.iter().any(|attr| 
                        matches!(attr, Attribute::Uniform)
                    );
                    
                    if is_uniform {
                        self.generate_uniform_declaration(decl_node)
                    } else {
                        // Regular variable declaration: let/def name: type = value or let/def name = value
                        let ty = decl_node.ty.as_ref().ok_or_else(|| {
                            CompilerError::SpirvError(format!("{} declaration must have explicit type", decl_node.keyword))
                        })?;
                        self.generate_declaration_helper(&decl_node.name, ty, &decl_node.body)
                    }
                } else {
                    // Function declaration: let/def name param1 param2 = body (skip for now)
                    Ok(())
                }
            }
            Declaration::Val(_val_decl) => {
                // Type signatures only
                Ok(())
            }
        }
    }

    fn generate_uniform_declaration(&mut self, decl: &Decl) -> Result<()> {
        let ty = decl.ty.as_ref().ok_or_else(|| {
            CompilerError::SpirvError("Uniform declaration must have explicit type".to_string())
        })?;

        // Create a global variable for the uniform (no initializer needed)
        let llvm_type = self.get_or_create_type(ty)?;
        let global = self.module.add_global(llvm_type, Some(AddressSpace::default()), &decl.name);
        
        // Uniforms are externally provided, so no initializer
        // The global will be linked at runtime by the graphics API
        
        // Store in variable cache so it can be referenced by other code
        self.variable_cache.insert(decl.name.clone(), global.as_pointer_value());
        self.variable_types.insert(decl.name.clone(), ty.clone());
        
        // Store for SPIR-V decoration generation
        self.uniform_globals.insert(decl.name.clone(), global);
        
        println!("DEBUG: Generated uniform variable '{}' with type {:?}", decl.name, ty);
        
        Ok(())
    }

    fn generate_declaration_helper(&mut self, name: &str, ty: &Type, value_expr: &Expression) -> Result<()> {
        // For global variables, allow literal constants including arrays
        match value_expr {
            Expression::IntLiteral(_) | Expression::FloatLiteral(_) | Expression::ArrayLiteral(_) => {
                // Generate the constant expression value
                let value = self.generate_expression(value_expr)?;
                
                // Create a global variable with the correct type
                let llvm_type = self.get_or_create_type(ty)?;
                let global = self.module.add_global(llvm_type, Some(AddressSpace::default()), name);
                global.set_initializer(&value);
                
                // Store in variable cache
                self.variable_cache.insert(name.to_string(), global.as_pointer_value());
                self.variable_types.insert(name.to_string(), ty.clone());

                Ok(())
            }
            _ => {
                // For complex expressions, skip global variable generation
                // In shader programming, complex calculations should be in functions
                println!("Warning: Skipping global variable '{}' with complex initializer - use literals only for globals", name);
                Ok(())
            }
        }
    }

    fn generate_entry_point(&mut self, decl: &Decl) -> Result<()> {
        // Determine execution model
        let exec_model = if decl
            .attributes
            .iter()
            .any(|attr| matches!(attr, Attribute::Vertex))
        {
            spirv::ExecutionModel::Vertex
        } else if decl
            .attributes
            .iter()
            .any(|attr| matches!(attr, Attribute::Fragment))
        {
            spirv::ExecutionModel::Fragment
        } else {
            return Err(CompilerError::SpirvError(
                "Entry point must have either #[vertex] or #[fragment] attribute".to_string(),
            ));
        };

        self.entry_points.push((decl.name.clone(), exec_model));

        // Create function signature and handle output variables
        let return_type_ast = decl.ty.as_ref().ok_or_else(|| {
            CompilerError::SpirvError("Entry point must have return type".to_string())
        })?;
        
        // Create output globals for attributed tuple return types
        self.create_output_globals(return_type_ast)?;
        
        let return_type = self.get_or_create_type(return_type_ast)?;

        // Build parameter types
        let param_types: Vec<BasicMetadataTypeEnum> = decl
            .params
            .iter()
            .map(|param| {
                match param {
                    DeclParam::Typed(p) => self.get_or_create_type(&p.ty),
                    DeclParam::Untyped(_) => Err(CompilerError::SpirvError(
                        "Entry point parameters must have explicit types".to_string()
                    )),
                }
                .map(BasicMetadataTypeEnum::from)
            })
            .collect::<Result<Vec<_>>>()?;

        let fn_type = match return_type {
            BasicTypeEnum::ArrayType(arr_ty) => arr_ty.fn_type(&param_types, false),
            BasicTypeEnum::FloatType(float_ty) => float_ty.fn_type(&param_types, false),
            BasicTypeEnum::IntType(int_ty) => int_ty.fn_type(&param_types, false),
            BasicTypeEnum::VectorType(vec_ty) => vec_ty.fn_type(&param_types, false),
            BasicTypeEnum::StructType(struct_ty) => struct_ty.fn_type(&param_types, false),
            _ => {
                return Err(CompilerError::SpirvError(
                    "Unsupported return type".to_string(),
                ))
            }
        };

        // Create the function
        let function = self.module.add_function(&decl.name, fn_type, None);
        self.function_cache.insert(decl.name.clone(), function);
        self.current_function = Some(function);

        // Create entry block
        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

        // Map parameters to names
        for (i, param) in decl.params.iter().enumerate() {
            let param_value = function
                .get_nth_param(i as u32)
                .ok_or_else(|| CompilerError::SpirvError("Missing parameter".to_string()))?;
            
            let (param_name, param_type) = match param {
                DeclParam::Typed(p) => (&p.name, &p.ty),
                DeclParam::Untyped(_) => {
                    return Err(CompilerError::SpirvError(
                        "Entry point parameters must have explicit types".to_string()
                    ))
                }
            };

            // Get the parameter type from LLVM
            let llvm_param_type = self.get_or_create_type(param_type)?;

            // Create alloca for the parameter
            let alloca = self
                .builder
                .build_alloca(llvm_param_type, param_name)
                .map_err(|e| {
                    CompilerError::SpirvError(format!("Failed to create alloca: {}", e))
                })?;

            self.builder.build_store(alloca, param_value).map_err(|e| {
                CompilerError::SpirvError(format!("Failed to store parameter: {}", e))
            })?;

            self.variable_cache.insert(param_name.clone(), alloca);
            self.variable_types
                .insert(param_name.clone(), param_type.clone());
        }

        // Generate function body
        let result = self.generate_expression(&decl.body)?;

        // Return the result
        self.builder
            .build_return(Some(&result))
            .map_err(|e| CompilerError::SpirvError(format!("Failed to build return: {}", e)))?;

        self.current_function = None;

        Ok(())
    }

    fn generate_expression(&mut self, expr: &Expression) -> Result<BasicValueEnum<'ctx>> {
        match expr {
            Expression::IntLiteral(n) => {
                let i32_type = self.context.i32_type();
                Ok(i32_type.const_int(*n as u64, true).into())
            }
            Expression::FloatLiteral(f) => {
                let f32_type = self.context.f32_type();
                Ok(f32_type.const_float(*f as f64).into())
            }
            Expression::Identifier(name) => {
                if let Some(ptr) = self.variable_cache.get(name) {
                    let ptr_copy = *ptr;
                    // Get the variable's type for the load operation
                    let var_type = self
                        .variable_types
                        .get(name)
                        .ok_or_else(|| {
                            CompilerError::SpirvError(format!(
                                "Unknown variable type for: {}",
                                name
                            ))
                        })?
                        .clone();

                    let value_type = self.get_or_create_type(&var_type)?;

                    // Load the value from the pointer
                    let loaded = self
                        .builder
                        .build_load(value_type, ptr_copy, name)
                        .map_err(|e| {
                            CompilerError::SpirvError(format!("Failed to load variable: {}", e))
                        })?;
                    Ok(loaded)
                } else {
                    Err(CompilerError::UndefinedVariable(name.clone()))
                }
            }
            Expression::ArrayIndex(array_expr, index_expr) => {
                let array_name = match array_expr.as_ref() {
                    Expression::Identifier(name) => name,
                    _ => {
                        return Err(CompilerError::SpirvError(
                            "Only variable array indexing supported".to_string(),
                        ))
                    }
                };

                let array_ptr = self
                    .variable_cache
                    .get(array_name)
                    .copied()
                    .ok_or_else(|| CompilerError::UndefinedVariable(array_name.clone()))?;

                let index = self.generate_expression(index_expr)?;
                let index_value = index.into_int_value();

                // Get the array type
                let array_type = self
                    .variable_types
                    .get(array_name)
                    .ok_or_else(|| {
                        CompilerError::SpirvError(format!("Unknown array type for: {}", array_name))
                    })?
                    .clone();

                let element_type = match &array_type {
                    Type::Constructed(name, args)
                        if matches!(name, TypeName::Str("array") | TypeName::Array("array", _)) =>
                    {
                        args.first().cloned().unwrap_or_else(|| types::i32())
                    }
                    _ => {
                        return Err(CompilerError::SpirvError(
                            "Cannot index non-array type".to_string(),
                        ))
                    }
                };

                let element_type_id = self.get_or_create_type(&element_type)?;

                // Build GEP to get element pointer
                let zero = self.context.i32_type().const_zero();
                let indices = [zero, index_value];

                let array_llvm_type = self.get_or_create_type(&array_type)?;

                let element_ptr = unsafe {
                    self.builder
                        .build_gep(array_llvm_type, array_ptr, &indices, "array_element")
                        .map_err(|e| {
                            CompilerError::SpirvError(format!("Failed to build GEP: {}", e))
                        })?
                };

                // Load the element
                let element = self
                    .builder
                    .build_load(element_type_id, element_ptr, "element")
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to load element: {}", e))
                    })?;

                Ok(element)
            }
            Expression::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    return Err(CompilerError::SpirvError("Empty array literals not supported".to_string()));
                }

                // Generate all element values
                let mut const_values = Vec::new();
                for elem in elements {
                    const_values.push(self.generate_expression(elem)?);
                }

                // All elements should have the same type - use the first element's type
                match const_values[0] {
                    BasicValueEnum::FloatValue(_) => {
                        let float_values: Vec<_> = const_values.iter().map(|v| v.into_float_value()).collect();
                        let f32_type = self.context.f32_type();
                        Ok(f32_type.const_array(&float_values).into())
                    }
                    BasicValueEnum::IntValue(_) => {
                        let int_values: Vec<_> = const_values.iter().map(|v| v.into_int_value()).collect();
                        let i32_type = self.context.i32_type();
                        Ok(i32_type.const_array(&int_values).into())
                    }
                    BasicValueEnum::ArrayValue(_) => {
                        let array_values: Vec<_> = const_values.iter().map(|v| v.into_array_value()).collect();
                        let first_elem_type = array_values[0].get_type();
                        Ok(first_elem_type.const_array(&array_values).into())
                    }
                    _ => Err(CompilerError::SpirvError("Unsupported array element type".to_string())),
                }
            },
            Expression::BinaryOp(op, left, right) => {
                let left_val = self.generate_expression(left)?;
                let right_val = self.generate_expression(right)?;

                match op {
                    BinaryOp::Divide => {
                        // Assuming float division
                        let result = self
                            .builder
                            .build_float_div(
                                left_val.into_float_value(),
                                right_val.into_float_value(),
                                "div",
                            )
                            .map_err(|e| {
                                CompilerError::SpirvError(format!(
                                    "Failed to build division: {}",
                                    e
                                ))
                            })?;
                        Ok(result.into())
                    }
                    BinaryOp::Add => {
                        // Check the type of the operands to determine whether to use int or float addition
                        match (left_val, right_val) {
                            (BasicValueEnum::IntValue(left_int), BasicValueEnum::IntValue(right_int)) => {
                                let result = self
                                    .builder
                                    .build_int_add(left_int, right_int, "add")
                                    .map_err(|e| {
                                        CompilerError::SpirvError(format!("Failed to build int addition: {}", e))
                                    })?;
                                Ok(result.into())
                            }
                            (BasicValueEnum::FloatValue(left_float), BasicValueEnum::FloatValue(right_float)) => {
                                let result = self
                                    .builder
                                    .build_float_add(left_float, right_float, "add")
                                    .map_err(|e| {
                                        CompilerError::SpirvError(format!("Failed to build float addition: {}", e))
                                    })?;
                                Ok(result.into())
                            }
                            _ => {
                                Err(CompilerError::SpirvError(
                                    "Type mismatch in addition: operands must be both int or both float".to_string()
                                ))
                            }
                        }
                    }
                    BinaryOp::Subtract => {
                        // Check the type of the operands to determine whether to use int or float subtraction
                        match (left_val, right_val) {
                            (BasicValueEnum::IntValue(left_int), BasicValueEnum::IntValue(right_int)) => {
                                let result = self
                                    .builder
                                    .build_int_sub(left_int, right_int, "sub")
                                    .map_err(|e| {
                                        CompilerError::SpirvError(format!("Failed to build int subtraction: {}", e))
                                    })?;
                                Ok(result.into())
                            }
                            (BasicValueEnum::FloatValue(left_float), BasicValueEnum::FloatValue(right_float)) => {
                                let result = self
                                    .builder
                                    .build_float_sub(left_float, right_float, "sub")
                                    .map_err(|e| {
                                        CompilerError::SpirvError(format!("Failed to build float subtraction: {}", e))
                                    })?;
                                Ok(result.into())
                            }
                            _ => {
                                Err(CompilerError::SpirvError(
                                    "Type mismatch in subtraction: operands must be both int or both float".to_string()
                                ))
                            }
                        }
                    }
                    BinaryOp::Multiply => {
                        // Check the type of the operands to determine whether to use int or float multiplication
                        match (left_val, right_val) {
                            (BasicValueEnum::IntValue(left_int), BasicValueEnum::IntValue(right_int)) => {
                                let result = self
                                    .builder
                                    .build_int_mul(left_int, right_int, "mul")
                                    .map_err(|e| {
                                        CompilerError::SpirvError(format!("Failed to build int multiplication: {}", e))
                                    })?;
                                Ok(result.into())
                            }
                            (BasicValueEnum::FloatValue(left_float), BasicValueEnum::FloatValue(right_float)) => {
                                let result = self
                                    .builder
                                    .build_float_mul(left_float, right_float, "mul")
                                    .map_err(|e| {
                                        CompilerError::SpirvError(format!("Failed to build float multiplication: {}", e))
                                    })?;
                                Ok(result.into())
                            }
                            _ => {
                                Err(CompilerError::SpirvError(
                                    "Type mismatch in multiplication: operands must be both int or both float".to_string()
                                ))
                            }
                        }
                    }
                }
            }
            Expression::FunctionCall(func_name, args) => {
                // Check if it's a builtin function
                if self.builtin_manager.is_builtin(func_name) {
                    // Generate arguments first
                    let mut arg_values = Vec::new();
                    for arg in args {
                        arg_values.push(self.generate_expression(arg)?);
                    }

                    // Call the builtin through the manager
                    self.builtin_manager.generate_builtin_call(
                        &self.module,
                        &self.builder,
                        func_name,
                        &arg_values,
                    )
                } else {
                    match func_name.as_str() {
                        "to_vec4_f32" => {
                            if args.len() != 1 {
                                return Err(CompilerError::SpirvError(
                                    "to_vec4_f32 requires exactly one argument".to_string(),
                                ));
                            }

                            let array_val = self.generate_expression(&args[0])?;
                            self.convert_array_to_vec4(array_val)
                        }
                        // Vector constructors
                        "vec2" => self.generate_vector_constructor(2, &args, self.context.f32_type().into()),
                        "vec3" => self.generate_vector_constructor(3, &args, self.context.f32_type().into()),
                        "vec4" => self.generate_vector_constructor(4, &args, self.context.f32_type().into()),
                        "ivec2" => self.generate_vector_constructor(2, &args, self.context.i32_type().into()),
                        "ivec3" => self.generate_vector_constructor(3, &args, self.context.i32_type().into()),
                        "ivec4" => self.generate_vector_constructor(4, &args, self.context.i32_type().into()),
                        _ => Err(CompilerError::SpirvError(format!(
                            "Function call '{}' not supported",
                            func_name
                        ))),
                    }
                }
            }
            Expression::Tuple(elements) => {
                // Generate values for all tuple elements
                let mut element_values = Vec::new();
                for element in elements {
                    element_values.push(self.generate_expression(element)?);
                }

                // Create a struct with the element values
                if element_values.is_empty() {
                    // Empty tuple - use unit type (empty struct)
                    let unit_type = self.context.struct_type(&[], false);
                    Ok(unit_type.const_zero().into())
                } else {
                    // Create struct with element values
                    let element_types: Vec<BasicTypeEnum> = element_values
                        .iter()
                        .map(|val| val.get_type())
                        .collect();
                    let struct_type = self.context.struct_type(&element_types, false);
                    
                    // Build the struct value
                    let mut struct_val = struct_type.get_undef().into();
                    for (i, value) in element_values.into_iter().enumerate() {
                        struct_val = self.builder
                            .build_insert_value(struct_val, value, i as u32, "tuple_elem")
                            .map_err(|e| CompilerError::SpirvError(format!("Failed to build tuple: {}", e)))?;
                    }
                    
                    // Convert back to BasicValueEnum - structs should be StructValue
                    match struct_val {
                        inkwell::values::AggregateValueEnum::StructValue(sv) => Ok(sv.into()),
                        _ => Err(CompilerError::SpirvError("Expected struct value".to_string()))
                    }
                }
            }
            Expression::Lambda(_) => Err(CompilerError::SpirvError(
                "Lambda expressions require defunctionalization before LLVM generation".to_string(),
            )),
            Expression::Application(_, _) => Err(CompilerError::SpirvError(
                "Function applications require defunctionalization before LLVM generation"
                    .to_string(),
            )),
            Expression::LetIn(let_in) => {
                // Generate code for the value expression
                let value = self.generate_expression(&let_in.value)?;

                // Store the value in a local variable
                let var_type =
                    self.get_or_create_type(&let_in.ty.clone().unwrap_or_else(|| {
                        // For now, use i32 as default type if not specified
                        crate::ast::types::i32()
                    }))?;

                let ptr = self
                    .builder
                    .build_alloca(var_type, &let_in.name)
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to build alloca: {:?}", e))
                    })?;

                self.builder.build_store(ptr, value).map_err(|e| {
                    CompilerError::SpirvError(format!("Failed to build store: {:?}", e))
                })?;

                // Store in variable cache
                let ast_type = let_in
                    .ty
                    .clone()
                    .unwrap_or_else(|| crate::ast::types::i32());
                self.variable_cache.insert(let_in.name.clone(), ptr);
                self.variable_types.insert(let_in.name.clone(), ast_type);

                // Generate code for the body expression
                let result = self.generate_expression(&let_in.body)?;

                // Clean up variable from cache
                self.variable_cache.remove(&let_in.name);
                self.variable_types.remove(&let_in.name);

                Ok(result)
            }
            Expression::FieldAccess(expr, field) => {
                let record_value = self.generate_expression(expr)?;
                
                // For now, assume this is a vector type and extract the component
                // TODO: Need to determine record type and field layout for general records
                
                // Map field name to vector component index (for built-in vector types)
                let component_index = match field.as_str() {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    _ => return Err(CompilerError::SpirvError(format!(
                        "Field access for '{}' not yet implemented for non-vector types", field
                    ))),
                };
                
                // Use LLVM extract_element for vectors
                if let BasicValueEnum::VectorValue(vec_val) = record_value {
                    let index_val = self.context.i32_type().const_int(component_index, false);
                    let extracted = self.builder
                        .build_extract_element(vec_val, index_val, "field_extract")
                        .unwrap();
                    Ok(extracted)
                } else {
                    // For future general record types, we would use extract_value with field indices
                    Err(CompilerError::SpirvError(format!(
                        "Field access on non-vector record types not yet implemented. Got: {:?}", 
                        record_value.get_type()
                    )))
                }
            }
        }
    }


    fn generate_vector_constructor(
        &mut self,
        size: u32,
        args: &[Expression],
        elem_type: BasicTypeEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        // Check argument count
        if args.len() != size as usize {
            return Err(CompilerError::SpirvError(format!(
                "Vector constructor expects {} arguments, got {}",
                size,
                args.len()
            )));
        }

        // Generate values for all arguments
        let mut elements = Vec::new();
        for arg in args {
            let val = self.generate_expression(arg)?;
            elements.push(val);
        }

        // Create the vector type
        let vec_type = match elem_type {
            BasicTypeEnum::FloatType(f) => f.vec_type(size),
            BasicTypeEnum::IntType(i) => i.vec_type(size),
            _ => {
                return Err(CompilerError::SpirvError(
                    "Invalid element type for vector".to_string(),
                ))
            }
        };

        // Build the vector
        let mut vector = vec_type.get_undef();
        for (i, elem) in elements.iter().enumerate() {
            vector = self
                .builder
                .build_insert_element(
                    vector,
                    *elem,
                    self.context.i32_type().const_int(i as u64, false),
                    &format!("vec_elem_{}", i),
                )
                .map_err(|e| {
                    CompilerError::SpirvError(format!("Failed to insert element: {}", e))
                })?;
        }

        Ok(vector.into())
    }

    fn convert_array_to_vec4(
        &mut self,
        array_val: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>> {
        // For LLVM, we need to extract array elements and build a vector
        let f32_type = self.context.f32_type();
        let vec4_type = f32_type.vec_type(4);

        // Convert to array value
        let array = array_val.into_array_value();

        // Extract elements from the array
        let mut elements = Vec::new();
        for i in 0..4 {
            let element = self
                .builder
                .build_extract_value(array, i, &format!("elem_{}", i))
                .map_err(|e| {
                    CompilerError::SpirvError(format!("Failed to extract element: {}", e))
                })?;
            elements.push(element.into_float_value());
        }

        // Build the vector
        let mut vector = vec4_type.get_undef();
        for (i, elem) in elements.iter().enumerate() {
            vector = self
                .builder
                .build_insert_element(
                    vector,
                    *elem,
                    self.context.i32_type().const_int(i as u64, false),
                    &format!("vec_elem_{}", i),
                )
                .map_err(|e| {
                    CompilerError::SpirvError(format!("Failed to insert element: {}", e))
                })?;
        }

        Ok(vector.into())
    }

    fn get_or_create_type(&mut self, ty: &Type) -> Result<BasicTypeEnum<'ctx>> {
        if let Some(cached) = self.type_cache.get(ty) {
            return Ok(*cached);
        }

        let llvm_type = match ty {
            Type::Constructed(name, args) => {
                match name {
                    TypeName::Str("int") => BasicTypeEnum::IntType(self.context.i32_type()),
                    TypeName::Str("float") => BasicTypeEnum::FloatType(self.context.f32_type()),
                    
                    // f32 vector types
                    TypeName::Str("vec2") => {
                        let f32_type = self.context.f32_type();
                        BasicTypeEnum::VectorType(f32_type.vec_type(2))
                    }
                    TypeName::Str("vec3") => {
                        let f32_type = self.context.f32_type();
                        BasicTypeEnum::VectorType(f32_type.vec_type(3))
                    }
                    TypeName::Str("vec4") => {
                        let f32_type = self.context.f32_type();
                        BasicTypeEnum::VectorType(f32_type.vec_type(4))
                    }
                    
                    // i32 vector types
                    TypeName::Str("ivec2") => {
                        let i32_type = self.context.i32_type();
                        BasicTypeEnum::VectorType(i32_type.vec_type(2))
                    }
                    TypeName::Str("ivec3") => {
                        let i32_type = self.context.i32_type();
                        BasicTypeEnum::VectorType(i32_type.vec_type(3))
                    }
                    TypeName::Str("ivec4") => {
                        let i32_type = self.context.i32_type();
                        BasicTypeEnum::VectorType(i32_type.vec_type(4))
                    }
                    TypeName::Str("array") => {
                        let elem_ty = args.first().ok_or_else(|| {
                            CompilerError::SpirvError("Array type missing element type".to_string())
                        })?;
                        let elem_type = self.get_or_create_type(elem_ty)?;

                        // For unsized arrays, use default size
                        match elem_type {
                            BasicTypeEnum::ArrayType(arr) => arr.array_type(1).into(),
                            BasicTypeEnum::FloatType(float) => float.array_type(1).into(),
                            BasicTypeEnum::IntType(int) => int.array_type(1).into(),
                            BasicTypeEnum::VectorType(vec) => vec.array_type(1).into(),
                            _ => {
                                return Err(CompilerError::SpirvError(
                                    "Unsupported array element type".to_string(),
                                ))
                            }
                        }
                    }
                    TypeName::Array("array", size) => {
                        let elem_ty = args.first().ok_or_else(|| {
                            CompilerError::SpirvError("Array type missing element type".to_string())
                        })?;
                        let elem_type = self.get_or_create_type(elem_ty)?;

                        // Use the actual size from the type
                        match elem_type {
                            BasicTypeEnum::ArrayType(arr) => arr.array_type(*size as u32).into(),
                            BasicTypeEnum::FloatType(float) => {
                                float.array_type(*size as u32).into()
                            }
                            BasicTypeEnum::IntType(int) => int.array_type(*size as u32).into(),
                            BasicTypeEnum::VectorType(vec) => vec.array_type(*size as u32).into(),
                            _ => {
                                return Err(CompilerError::SpirvError(
                                    "Unsupported array element type".to_string(),
                                ))
                            }
                        }
                    }
                    TypeName::Str("tuple") | TypeName::Str("attributed_tuple") => {
                        // Create a struct type with the component types
                        let mut component_types = Vec::new();
                        for arg in args {
                            component_types.push(self.get_or_create_type(arg)?);
                        }
                        let struct_type = self.context.struct_type(&component_types, false);
                        BasicTypeEnum::StructType(struct_type)
                    }
                    _ => {
                        return Err(CompilerError::SpirvError(format!(
                            "Unknown type constructor: {:?}",
                            name
                        )));
                    }
                }
            }
            _ => {
                return Err(CompilerError::SpirvError(format!(
                    "Type {:?} not supported in LLVM generation",
                    ty
                )));
            }
        };

        self.type_cache.insert(ty.clone(), llvm_type);
        Ok(llvm_type)
    }

    fn add_spirv_entry_point_metadata(
        &mut self,
        name: &str,
        exec_model: &spirv::ExecutionModel,
    ) -> Result<()> {
        // Add SPIR-V specific attributes to mark entry points
        let _func = self
            .function_cache
            .get(name)
            .ok_or_else(|| CompilerError::SpirvError(format!("Entry point {} not found", name)))?;

        // For SPIR-V, we need to add specific attributes that the LLVM SPIR-V backend recognizes
        // This is a simplified version - real implementation would need proper SPIR-V metadata

        // Add execution model as a function attribute
        let _exec_model_str = match exec_model {
            spirv::ExecutionModel::Vertex => "spir_vertex",
            spirv::ExecutionModel::Fragment => "spir_fragment",
            _ => {
                return Err(CompilerError::SpirvError(
                    "Unsupported execution model".to_string(),
                ))
            }
        };

        // Note: In a real implementation, we'd add proper SPIR-V metadata here
        // For now, we'll rely on the LLVM SPIR-V translator to handle the conversion

        Ok(())
    }

    fn create_output_globals(&mut self, return_type: &Type) -> Result<()> {
        // Check if this is an attributed tuple with location attributes
        if let Type::Constructed(TypeName::Str("attributed_tuple"), args) = return_type {
            // For each component in the attributed tuple, create an output global
            for (index, component_type) in args.iter().enumerate() {
                // Extract location from the component type if it has attributes
                // For now, use sequential locations starting from 0
                let location = index as u32;
                
                // Create a global output variable for this component
                let llvm_type = self.get_or_create_type(component_type)?;
                let global_name = format!("output_{}", index);
                let global = self.module.add_global(llvm_type, Some(AddressSpace::default()), &global_name);
                
                // Store for decoration later
                self.output_globals.insert(global_name.clone(), (global, location));
                
                println!("DEBUG: Created output global '{}' at location {}", global_name, location);
            }
        }
        Ok(())
    }

    fn add_spirv_decorations(&mut self) -> Result<()> {
        // Collect the keys first to avoid borrowing issues
        let uniform_names: Vec<String> = self.uniform_globals.keys().cloned().collect();
        let output_names: Vec<String> = self.output_globals.keys().cloned().collect();
        
        // Add SPIR-V decorations for uniform variables
        for name in uniform_names {
            if let Some(global) = self.uniform_globals.get(&name) {
                self.add_uniform_decoration(&name, *global)?;
            }
        }

        // Add SPIR-V decorations for output variables
        for name in output_names {
            if let Some((global, location)) = self.output_globals.get(&name) {
                self.add_output_decoration(&name, *global, *location)?;
            }
        }

        Ok(())
    }

    fn add_uniform_decoration(&mut self, name: &str, global: GlobalValue<'ctx>) -> Result<()> {
        // For SPIR-V uniforms, we need to add proper storage class and binding decorations
        // This is a simplified approach - in practice we'd use proper SPIR-V decoration APIs
        
        // Add a comment to the global to indicate it's a uniform
        let uniform_name = format!("{}_uniform", name);
        global.set_name(&uniform_name);
        
        // Set storage class to Uniform (via address space or other mechanism)
        // In SPIR-V, uniforms typically use StorageClass::Uniform
        
        println!("DEBUG: Added uniform decoration for '{}'", name);
        Ok(())
    }

    fn add_output_decoration(&mut self, name: &str, global: GlobalValue<'ctx>, location: u32) -> Result<()> {
        // For SPIR-V outputs, we need to add location decorations
        // This is a simplified approach - in practice we'd use proper SPIR-V decoration APIs
        
        let output_name = format!("{}_output_loc{}", name, location);
        global.set_name(&output_name);
        
        println!("DEBUG: Added output decoration for '{}' at location {}", name, location);
        Ok(())
    }

    fn emit_spirv(&self) -> Result<Vec<u32>> {
        // Initialize LLVM targets
        Target::initialize_all(&InitializationConfig::default());

        // Print LLVM IR for debugging
        self.module.print_to_stderr();

        // Find the SPIR-V target
        let target_triple = TargetTriple::create("spirv64-unknown-unknown");
        let target = Target::from_triple(&target_triple)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to create SPIR-V target: {}", e)))?;

        // Create target machine for SPIR-V
        let target_machine = target
            .create_target_machine(
                &target_triple,
                "",
                "",
                inkwell::OptimizationLevel::Default,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| CompilerError::SpirvError("Failed to create target machine".to_string()))?;

        // Generate SPIR-V object code
        let object_data = target_machine
            .write_to_memory_buffer(&self.module, FileType::Object)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to generate SPIR-V: {}", e)))?;

        // Convert the memory buffer to Vec<u32>
        let bytes = object_data.as_slice();
        
        // SPIR-V is stored as 32-bit words, so convert bytes to u32
        if bytes.len() % 4 != 0 {
            return Err(CompilerError::SpirvError("SPIR-V output not aligned to 32-bit words".to_string()));
        }
        
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        println!("DEBUG: Generated {} bytes of SPIR-V ({} words)", bytes.len(), words.len());
        
        Ok(words)
    }
}
