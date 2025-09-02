use crate::ast::*;
use crate::error::{CompilerError, Result};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{InitializationConfig, Target, TargetTriple};
use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum};
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use inkwell::AddressSpace;
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    variable_cache: HashMap<String, PointerValue<'ctx>>,
    variable_types: HashMap<String, Type>,
    function_cache: HashMap<String, FunctionValue<'ctx>>,
    type_cache: HashMap<Type, BasicTypeEnum<'ctx>>,
    current_function: Option<FunctionValue<'ctx>>,
    entry_points: Vec<(String, spirv::ExecutionModel)>,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        // Set up SPIR-V target triple
        module.set_triple(&TargetTriple::create("spirv64-unknown-unknown"));

        CodeGenerator {
            context,
            module,
            builder,
            variable_cache: HashMap::new(),
            variable_types: HashMap::new(),
            function_cache: HashMap::new(),
            type_cache: HashMap::new(),
            current_function: None,
            entry_points: Vec::new(),
        }
    }

    pub fn generate(mut self, program: &Program) -> Result<Vec<u32>> {
        // Process all declarations
        for decl in &program.declarations {
            self.generate_declaration(decl)?;
        }

        // Clone entry points to avoid borrow issues
        let entry_points = self.entry_points.clone();

        // Add SPIR-V specific metadata for entry points
        for (name, exec_model) in &entry_points {
            self.add_spirv_entry_point_metadata(&name, exec_model)?;
        }

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
            Declaration::Let(let_decl) => self.generate_let_decl(let_decl),
            Declaration::Entry(entry_decl) => self.generate_entry_decl(entry_decl),
            Declaration::Def(_def_decl) => {
                // User-defined functions - skip for now
                Ok(())
            }
            Declaration::Val(_val_decl) => {
                // Type signatures only
                Ok(())
            }
        }
    }

    fn generate_let_decl(&mut self, decl: &LetDecl) -> Result<()> {
        let ty = decl.ty.as_ref().ok_or_else(|| {
            CompilerError::SpirvError("Let declaration must have explicit type".to_string())
        })?;

        let llvm_type = self.get_or_create_type(ty)?;

        match &decl.value {
            Expression::ArrayLiteral(elements) => {
                // Create a global variable for the array
                let global =
                    self.module
                        .add_global(llvm_type, Some(AddressSpace::default()), &decl.name);

                // Generate the constant array value
                let const_array = self.generate_constant_array(ty, elements)?;
                global.set_initializer(&const_array);

                // Store in variable cache
                self.variable_cache
                    .insert(decl.name.clone(), global.as_pointer_value());
                self.variable_types.insert(decl.name.clone(), ty.clone());
            }
            _ => {
                return Err(CompilerError::SpirvError(
                    "Only array literals supported for let declarations".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn generate_entry_decl(&mut self, decl: &EntryDecl) -> Result<()> {
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

        // Create function signature
        let return_type = self.get_or_create_type(&decl.return_type.ty)?;

        // Build parameter types
        let param_types: Vec<BasicMetadataTypeEnum> = decl
            .params
            .iter()
            .map(|param| {
                self.get_or_create_type(&param.ty)
                    .map(|t| BasicMetadataTypeEnum::from(t))
            })
            .collect::<Result<Vec<_>>>()?;

        let fn_type = match return_type {
            BasicTypeEnum::ArrayType(arr_ty) => arr_ty.fn_type(&param_types, false),
            BasicTypeEnum::FloatType(float_ty) => float_ty.fn_type(&param_types, false),
            BasicTypeEnum::IntType(int_ty) => int_ty.fn_type(&param_types, false),
            BasicTypeEnum::VectorType(vec_ty) => vec_ty.fn_type(&param_types, false),
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

            // Get the parameter type
            let param_type = self.get_or_create_type(&param.ty)?;

            // Create alloca for the parameter
            let alloca = self
                .builder
                .build_alloca(param_type, &param.name)
                .map_err(|e| {
                    CompilerError::SpirvError(format!("Failed to create alloca: {}", e))
                })?;

            self.builder.build_store(alloca, param_value).map_err(|e| {
                CompilerError::SpirvError(format!("Failed to store parameter: {}", e))
            })?;

            self.variable_cache.insert(param.name.clone(), alloca);
            self.variable_types
                .insert(param.name.clone(), param.ty.clone());
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
                    Type::Array(elem_ty, _dims) => elem_ty.as_ref().clone(),
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
            Expression::ArrayLiteral(_) => Err(CompilerError::SpirvError(
                "Array literals only supported in let declarations".to_string(),
            )),
            Expression::BinaryOp(BinaryOp::Divide, left, right) => {
                let left_val = self.generate_expression(left)?;
                let right_val = self.generate_expression(right)?;

                // Assuming float division
                let result = self
                    .builder
                    .build_float_div(
                        left_val.into_float_value(),
                        right_val.into_float_value(),
                        "div",
                    )
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to build division: {}", e))
                    })?;

                Ok(result.into())
            }
            Expression::FunctionCall(func_name, args) => match func_name.as_str() {
                "to_vec4_f32" => {
                    if args.len() != 1 {
                        return Err(CompilerError::SpirvError(
                            "to_vec4_f32 requires exactly one argument".to_string(),
                        ));
                    }

                    let array_val = self.generate_expression(&args[0])?;
                    self.convert_array_to_vec4(array_val)
                }
                _ => Err(CompilerError::SpirvError(format!(
                    "Function call '{}' not supported",
                    func_name
                ))),
            },
            Expression::Tuple(_) => Err(CompilerError::SpirvError(
                "Tuples not supported in LLVM generation".to_string(),
            )),
        }
    }

    fn generate_constant_array(
        &mut self,
        ty: &Type,
        elements: &[Expression],
    ) -> Result<BasicValueEnum<'ctx>> {
        match ty {
            Type::Array(elem_ty, _dims) => {
                let mut const_values = Vec::new();

                for elem in elements {
                    match elem {
                        Expression::ArrayLiteral(inner) => {
                            let inner_const = self.generate_constant_array(elem_ty, inner)?;
                            const_values.push(inner_const);
                        }
                        Expression::FloatLiteral(f) => {
                            let f32_type = self.context.f32_type();
                            const_values.push(f32_type.const_float(*f as f64).into());
                        }
                        Expression::IntLiteral(n) => {
                            let i32_type = self.context.i32_type();
                            const_values.push(i32_type.const_int(*n as u64, true).into());
                        }
                        Expression::BinaryOp(BinaryOp::Divide, left, right) => {
                            if let (Expression::FloatLiteral(l), Expression::FloatLiteral(r)) =
                                (left.as_ref(), right.as_ref())
                            {
                                let result = l / r;
                                let f32_type = self.context.f32_type();
                                const_values.push(f32_type.const_float(result as f64).into());
                            } else {
                                return Err(CompilerError::SpirvError(
                                    "Only constant float division supported in array literals"
                                        .to_string(),
                                ));
                            }
                        }
                        _ => {
                            return Err(CompilerError::SpirvError(
                                "Only literals supported in constant arrays".to_string(),
                            ));
                        }
                    }
                }

                // Create the array constant as a const_array directly from values
                let array_type = self.get_or_create_type(ty)?;
                match array_type {
                    BasicTypeEnum::ArrayType(_arr_ty) => {
                        // Handle both simple and nested arrays
                        if const_values.is_empty() {
                            return Err(CompilerError::SpirvError(
                                "Empty array not supported".to_string(),
                            ));
                        }

                        // Create array from the const values
                        match const_values[0] {
                            BasicValueEnum::FloatValue(_f) => {
                                let float_values: Vec<_> =
                                    const_values.iter().map(|v| v.into_float_value()).collect();
                                let f32_type = self.context.f32_type();
                                Ok(f32_type.const_array(&float_values).into())
                            }
                            BasicValueEnum::IntValue(_i) => {
                                let int_values: Vec<_> =
                                    const_values.iter().map(|v| v.into_int_value()).collect();
                                let i32_type = self.context.i32_type();
                                Ok(i32_type.const_array(&int_values).into())
                            }
                            BasicValueEnum::ArrayValue(_) => {
                                // Handle nested arrays
                                let array_values: Vec<_> =
                                    const_values.iter().map(|v| v.into_array_value()).collect();
                                // Get the element type of the first inner array
                                if let Some(first_elem) = array_values.first() {
                                    let elem_type = first_elem.get_type();
                                    Ok(elem_type.const_array(&array_values).into())
                                } else {
                                    Err(CompilerError::SpirvError("Empty nested array".to_string()))
                                }
                            }
                            _ => Err(CompilerError::SpirvError(
                                "Unsupported array element type in constant".to_string(),
                            )),
                        }
                    }
                    _ => Err(CompilerError::SpirvError("Expected array type".to_string())),
                }
            }
            _ => Err(CompilerError::SpirvError(
                "Expected array type for array literal".to_string(),
            )),
        }
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
            Type::I32 => BasicTypeEnum::IntType(self.context.i32_type()),
            Type::F32 => BasicTypeEnum::FloatType(self.context.f32_type()),
            Type::Vec4F32 => {
                let f32_type = self.context.f32_type();
                BasicTypeEnum::VectorType(f32_type.vec_type(4))
            }
            Type::Array(elem_ty, dims) => {
                let elem_type = self.get_or_create_type(elem_ty)?;

                // Build multi-dimensional arrays from innermost to outermost
                let mut current_type = elem_type;
                for dim in dims.iter().rev() {
                    current_type = match current_type {
                        BasicTypeEnum::ArrayType(arr) => arr.array_type(*dim as u32).into(),
                        BasicTypeEnum::FloatType(float) => float.array_type(*dim as u32).into(),
                        BasicTypeEnum::IntType(int) => int.array_type(*dim as u32).into(),
                        BasicTypeEnum::VectorType(vec) => vec.array_type(*dim as u32).into(),
                        _ => {
                            return Err(CompilerError::SpirvError(
                                "Unsupported array element type".to_string(),
                            ))
                        }
                    };
                }
                current_type
            }
            Type::Tuple(_) => {
                return Err(CompilerError::SpirvError(
                    "Tuple types not supported in LLVM generation".to_string(),
                ));
            }
            Type::Var(_) | Type::Function(_, _) | Type::SizeVar(_) => {
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

    fn emit_spirv(&self) -> Result<Vec<u32>> {
        // For now, return a placeholder SPIR-V
        // In a real implementation, we would:
        // 1. Use LLVM's SPIR-V backend to generate SPIR-V
        // 2. Or use an external tool like llvm-spirv to translate LLVM IR to SPIR-V

        // Initialize LLVM targets
        Target::initialize_all(&InitializationConfig::default());

        // For now, let's output LLVM IR and return a minimal SPIR-V header
        // This is a placeholder - real implementation would generate actual SPIR-V

        // Print LLVM IR for debugging
        self.module.print_to_stderr();

        // Return minimal SPIR-V header as placeholder
        Ok(vec![
            spirv::MAGIC_NUMBER,
            0x00010500, // Version 1.5
            0,          // Generator ID
            0,          // Bound
            0,          // Schema
        ])
    }
}
