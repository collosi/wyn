use crate::ast::TypeName;
use crate::ast::*;
use crate::builtins::BuiltinManager;
use crate::error::{CompilerError, Result};
use rspirv::dr::{Builder, Module as SpirvModule, Operand};
use rspirv::spirv::{self, StorageClass, Decoration, Capability, AddressingModel, MemoryModel};
use rspirv::binary::Assemble;
use std::collections::HashMap;

/// Value represents a SPIR-V value with its type and ID
#[derive(Debug, Clone, Copy)]
pub struct Value {
    pub id: spirv::Word,
    pub type_id: spirv::Word,
}

/// Type information for SPIR-V code generation
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub id: spirv::Word,
    pub size_bytes: u32,
}

pub struct CodeGenerator {
    builder: Builder,
    module: SpirvModule,
    builtin_manager: BuiltinManager,
    variable_cache: HashMap<String, Value>,
    variable_types: HashMap<String, Type>,
    function_cache: HashMap<String, spirv::Word>,
    type_cache: HashMap<Type, TypeInfo>,
    current_function: Option<spirv::Word>,
    entry_points: Vec<(String, spirv::ExecutionModel)>,
    uniform_globals: HashMap<String, Value>,
    output_globals: HashMap<String, (Value, u32, spirv::Word)>, // (value, location, component_type)
    
    // Type IDs for common types
    void_type: spirv::Word,
    bool_type: spirv::Word,
    i32_type: spirv::Word,
    f32_type: spirv::Word,
    ptr_input_f32_type: spirv::Word,
    ptr_output_f32_type: spirv::Word,
    ptr_uniform_f32_type: spirv::Word,
    ptr_function_f32_type: spirv::Word,
    ptr_function_i32_type: spirv::Word,
}

impl CodeGenerator {
    pub fn new(_module_name: &str) -> Self {
        let mut builder = Builder::new();
        
        // Set up SPIR-V header
        builder.set_version(1, 0);
        
        // Add capabilities
        builder.capability(Capability::Shader);
        
        // Add memory model
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        
        let mut generator = CodeGenerator {
            module: SpirvModule::new(),
            builder,
            builtin_manager: BuiltinManager::new(),
            variable_cache: HashMap::new(),
            variable_types: HashMap::new(),
            function_cache: HashMap::new(),
            type_cache: HashMap::new(),
            current_function: None,
            entry_points: Vec::new(),
            uniform_globals: HashMap::new(),
            output_globals: HashMap::new(),
            
            // Will be initialized in setup_common_types
            void_type: 0,
            bool_type: 0,
            i32_type: 0,
            f32_type: 0,
            ptr_input_f32_type: 0,
            ptr_output_f32_type: 0,
            ptr_uniform_f32_type: 0,
            ptr_function_f32_type: 0,
            ptr_function_i32_type: 0,
        };
        
        generator.setup_common_types();
        generator
    }

    fn setup_common_types(&mut self) {
        // Create common SPIR-V types
        self.void_type = self.builder.type_void();
        self.bool_type = self.builder.type_bool();
        self.i32_type = self.builder.type_int(32, 1);
        self.f32_type = self.builder.type_float(32);
        
        // Create pointer types for different storage classes
        self.ptr_input_f32_type = self.builder.type_pointer(None, StorageClass::Input, self.f32_type);
        self.ptr_output_f32_type = self.builder.type_pointer(None, StorageClass::Output, self.f32_type);
        self.ptr_uniform_f32_type = self.builder.type_pointer(None, StorageClass::Uniform, self.f32_type);
        self.ptr_function_f32_type = self.builder.type_pointer(None, StorageClass::Function, self.f32_type);
        self.ptr_function_i32_type = self.builder.type_pointer(None, StorageClass::Function, self.i32_type);
    }

    pub fn generate(mut self, program: &Program) -> Result<Vec<u32>> {
        // Load builtin functions into the module
        self.builtin_manager.load_builtins_into_module(&mut self.builder)?;

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

        // Build the module and return SPIR-V words
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

        // Create a global variable for the uniform
        let type_info = self.get_or_create_type(ty)?;
        let ptr_type = self.builder.type_pointer(None, StorageClass::Uniform, type_info.id);
        let global_id = self.builder.variable(ptr_type, None, StorageClass::Uniform, None);
        
        let value = Value {
            id: global_id,
            type_id: ptr_type,
        };
        
        // Store in variable cache so it can be referenced by other code
        self.variable_cache.insert(decl.name.clone(), value);
        self.variable_types.insert(decl.name.clone(), ty.clone());
        
        // Store for SPIR-V decoration generation
        self.uniform_globals.insert(decl.name.clone(), value);
        
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
                let type_info = self.get_or_create_type(ty)?;
                let ptr_type = self.builder.type_pointer(None, StorageClass::Private, type_info.id);
                let global_id = self.builder.variable(ptr_type, None, StorageClass::Private, Some(value.id));
                
                let global_value = Value {
                    id: global_id,
                    type_id: ptr_type,
                };
                
                // Store in variable cache
                self.variable_cache.insert(name.to_string(), global_value);
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
        if matches!(return_type_ast, Type::Constructed(TypeName::Str("attributed_tuple"), _)) {
            self.create_output_globals(return_type_ast)?;
        }
        
        let return_type_info = self.get_or_create_type(return_type_ast)?;

        // Build parameter types
        let mut param_type_ids = Vec::new();
        for param in &decl.params {
            match param {
                DeclParam::Typed(p) => {
                    let param_type_info = self.get_or_create_type(&p.ty)?;
                    param_type_ids.push(param_type_info.id);
                }
                DeclParam::Untyped(_) => {
                    return Err(CompilerError::SpirvError(
                        "Entry point parameters must have explicit types".to_string()
                    ));
                }
            }
        }

        // Create function type - entry points return void and use output variables
        let actual_return_type = if matches!(return_type_ast, Type::Constructed(TypeName::Str("attributed_tuple"), _)) {
            self.void_type
        } else {
            return_type_info.id
        };
        let fn_type = self.builder.type_function(actual_return_type, param_type_ids);
        
        // Create the function
        let function_id = self.builder.begin_function(
            actual_return_type,
            None,
            spirv::FunctionControl::NONE,
            fn_type,
        )?;
        
        self.function_cache.insert(decl.name.clone(), function_id);
        self.current_function = Some(function_id);

        // Create function parameters - need to do this after begin_block
        let mut params_to_store = Vec::new();
        for (_i, param) in decl.params.iter().enumerate() {
            let (param_name, param_type) = match param {
                DeclParam::Typed(p) => (&p.name, &p.ty),
                DeclParam::Untyped(_) => {
                    return Err(CompilerError::SpirvError(
                        "Entry point parameters must have explicit types".to_string()
                    ));
                }
            };

            let param_type_info = self.get_or_create_type(param_type)?;
            let param_id = self.builder.function_parameter(param_type_info.id)?;
            
            params_to_store.push((param_name.clone(), param_type.clone(), param_id, param_type_info));
        }

        // Create entry block
        let _entry_label = self.builder.begin_block(None)?;
        
        // Now store the parameters in local variables (must be done after begin_block)
        for (param_name, param_type, param_id, param_type_info) in params_to_store {
            // Create local variable for the parameter
            let ptr_type = self.builder.type_pointer(None, StorageClass::Function, param_type_info.id);
            let local_var = self.builder.variable(ptr_type, None, StorageClass::Function, None);
            self.builder.store(local_var, param_id, None, vec![])?;

            let param_value = Value {
                id: local_var,
                type_id: ptr_type,
            };

            self.variable_cache.insert(param_name.clone(), param_value);
            self.variable_types.insert(param_name, param_type);
        }

        // Generate function body
        let result = self.generate_expression(&decl.body)?;

        // For entry points, we need to store the result in output variables instead of returning
        if !self.output_globals.is_empty() {
            // This is an entry point with outputs - extract tuple components and store them
            match return_type_ast {
                Type::Constructed(TypeName::Str("attributed_tuple"), args) => {
                    // Extract each component from the result tuple and store in corresponding output
                    for (index, _component_type) in args.iter().enumerate() {
                        let output_name = format!("output_{}", index);
                        if let Some((output_global, _, component_type)) = self.output_globals.get(&output_name) {
                            // Extract the component from the result tuple
                            let component_id = self.builder.composite_extract(
                                *component_type, // Use the stored component type
                                None,
                                result.id,
                                vec![index as u32]
                            )?;
                            
                            // Store in the output global
                            self.builder.store(output_global.id, component_id, None, vec![])?;
                        }
                    }
                }
                _ => {
                    // Single return value - just return it normally
                    self.builder.ret_value(result.id)?;
                }
            }
            
            // Entry points return void
            self.builder.ret()?;
        } else {
            // Regular function - return the result
            self.builder.ret_value(result.id)?;
        }
        
        self.builder.end_function()?;
        self.current_function = None;

        Ok(())
    }

    fn generate_expression(&mut self, expr: &Expression) -> Result<Value> {
        match expr {
            Expression::IntLiteral(n) => {
                let const_id = self.builder.constant_bit32(self.i32_type, *n as u32);
                Ok(Value {
                    id: const_id,
                    type_id: self.i32_type,
                })
            }
            Expression::FloatLiteral(f) => {
                let const_id = self.builder.constant_bit32(self.f32_type, f.to_bits());
                Ok(Value {
                    id: const_id,
                    type_id: self.f32_type,
                })
            }
            Expression::Identifier(name) => {
                if let Some(&value) = self.variable_cache.get(name) {
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

                    let type_info = self.get_or_create_type(&var_type)?;

                    // Load the value from the pointer
                    let loaded_id = self.builder.load(type_info.id, None, value.id, None, vec![])?;
                    Ok(Value {
                        id: loaded_id,
                        type_id: type_info.id,
                    })
                } else {
                    Err(CompilerError::UndefinedVariable(name.clone()))
                }
            }
            Expression::BinaryOp(op, left, right) => {
                let left_val = self.generate_expression(left)?;
                let right_val = self.generate_expression(right)?;

                // Use unified operator string instead of enum variants
                match op.op.as_str() {
                    "/" => {
                        // Assuming float division
                        let result_id = self.builder.f_div(self.f32_type, None, left_val.id, right_val.id)?;
                        Ok(Value {
                            id: result_id,
                            type_id: self.f32_type,
                        })
                    }
                    "+" => {
                        // Check the type of the operands to determine whether to use int or float addition
                        if left_val.type_id == self.i32_type && right_val.type_id == self.i32_type {
                            let result_id = self.builder.i_add(self.i32_type, None, left_val.id, right_val.id)?;
                            Ok(Value {
                                id: result_id,
                                type_id: self.i32_type,
                            })
                        } else if left_val.type_id == self.f32_type && right_val.type_id == self.f32_type {
                            let result_id = self.builder.f_add(self.f32_type, None, left_val.id, right_val.id)?;
                            Ok(Value {
                                id: result_id,
                                type_id: self.f32_type,
                            })
                        } else {
                            Err(CompilerError::SpirvError(
                                "Type mismatch in addition: operands must be both int or both float".to_string()
                            ))
                        }
                    }
                    "-" => {
                        // Check the type of the operands to determine whether to use int or float subtraction
                        if left_val.type_id == self.i32_type && right_val.type_id == self.i32_type {
                            let result_id = self.builder.i_sub(self.i32_type, None, left_val.id, right_val.id)?;
                            Ok(Value {
                                id: result_id,
                                type_id: self.i32_type,
                            })
                        } else if left_val.type_id == self.f32_type && right_val.type_id == self.f32_type {
                            let result_id = self.builder.f_sub(self.f32_type, None, left_val.id, right_val.id)?;
                            Ok(Value {
                                id: result_id,
                                type_id: self.f32_type,
                            })
                        } else {
                            Err(CompilerError::SpirvError(
                                "Type mismatch in subtraction: operands must be both int or both float".to_string()
                            ))
                        }
                    }
                    "*" => {
                        // Check the type of the operands to determine whether to use int or float multiplication
                        if left_val.type_id == self.i32_type && right_val.type_id == self.i32_type {
                            let result_id = self.builder.i_mul(self.i32_type, None, left_val.id, right_val.id)?;
                            Ok(Value {
                                id: result_id,
                                type_id: self.i32_type,
                            })
                        } else if left_val.type_id == self.f32_type && right_val.type_id == self.f32_type {
                            let result_id = self.builder.f_mul(self.f32_type, None, left_val.id, right_val.id)?;
                            Ok(Value {
                                id: result_id,
                                type_id: self.f32_type,
                            })
                        } else {
                            Err(CompilerError::SpirvError(
                                "Type mismatch in multiplication: operands must be both int or both float".to_string()
                            ))
                        }
                    }
                    _ => {
                        Err(CompilerError::SpirvError(format!(
                            "Unknown binary operator: {}", op.op
                        )))
                    }
                }
            }
            Expression::FunctionCall(func_name, args) => {
                // Handle length as a compiler intrinsic
                if func_name == "length" {
                    if args.len() != 1 {
                        return Err(CompilerError::SpirvError(
                            "length requires exactly one argument".to_string(),
                        ));
                    }
                    
                    // For arrays, return the compile-time known size
                    let array_expr = &args[0];
                    
                    // If it's an array identifier, get its type and return the size
                    if let Expression::Identifier(array_name) = array_expr {
                        let array_type = self.variable_types.get(array_name)
                            .ok_or_else(|| CompilerError::UndefinedVariable(array_name.clone()))?;
                        
                        match array_type {
                            Type::Constructed(TypeName::Array("array", size), _) => {
                                let const_id = self.builder.constant_bit32(self.i32_type, *size as u32);
                                Ok(Value {
                                    id: const_id,
                                    type_id: self.i32_type,
                                })
                            }
                            Type::Constructed(TypeName::Str("array"), _) => {
                                // Default size for unsized arrays
                                let const_id = self.builder.constant_bit32(self.i32_type, 1);
                                Ok(Value {
                                    id: const_id,
                                    type_id: self.i32_type,
                                })
                            }
                            _ => Err(CompilerError::SpirvError(
                                "length can only be applied to arrays".to_string()
                            ))
                        }
                    } else {
                        // For array literals, count the elements
                        if let Expression::ArrayLiteral(elements) = array_expr {
                            let const_id = self.builder.constant_bit32(self.i32_type, elements.len() as u32);
                            Ok(Value {
                                id: const_id,
                                type_id: self.i32_type,
                            })
                        } else {
                            Err(CompilerError::SpirvError(
                                "length can only be applied to array variables or literals".to_string()
                            ))
                        }
                    }
                }
                // Check if it's a builtin function
                else if self.builtin_manager.is_builtin(func_name) {
                    // Generate arguments first
                    let mut arg_values = Vec::new();
                    for arg in args {
                        arg_values.push(self.generate_expression(arg)?);
                    }

                    // Call the builtin through the manager
                    self.builtin_manager.generate_builtin_call(
                        &mut self.builder,
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
                        "vec2" => self.generate_vector_constructor(2, &args),
                        "vec3" => self.generate_vector_constructor(3, &args),
                        "vec4" => self.generate_vector_constructor(4, &args),
                        "ivec2" => self.generate_vector_constructor(2, &args),
                        "ivec3" => self.generate_vector_constructor(3, &args),
                        "ivec4" => self.generate_vector_constructor(4, &args),
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
                let mut element_type_ids = Vec::new();
                for element in elements {
                    let val = self.generate_expression(element)?;
                    element_values.push(val.id);
                    element_type_ids.push(val.type_id);
                }

                // Create a struct with the element values
                if element_values.is_empty() {
                    // Empty tuple - use unit type (empty struct)
                    let unit_type = self.builder.type_struct(vec![]);
                    let const_id = self.builder.constant_null(unit_type);
                    Ok(Value {
                        id: const_id,
                        type_id: unit_type,
                    })
                } else {
                    // Create struct with element values
                    let struct_type = self.builder.type_struct(element_type_ids);
                    let struct_id = self.builder.composite_construct(struct_type, None, element_values)?;
                    
                    Ok(Value {
                        id: struct_id,
                        type_id: struct_type,
                    })
                }
            }
            Expression::LetIn(let_in) => {
                // Generate code for the value expression
                let value = self.generate_expression(&let_in.value)?;

                // Create a local variable for the let binding
                let var_type = let_in.ty.clone().unwrap_or_else(|| {
                    // For now, use i32 as default type if not specified
                    crate::ast::types::i32()
                });

                let type_info = self.get_or_create_type(&var_type)?;
                let ptr_type = self.builder.type_pointer(None, StorageClass::Function, type_info.id);
                let local_var = self.builder.variable(ptr_type, None, StorageClass::Function, None);
                self.builder.store(local_var, value.id, None, vec![])?;

                // Store in variable cache
                let local_value = Value {
                    id: local_var,
                    type_id: ptr_type,
                };
                self.variable_cache.insert(let_in.name.clone(), local_value);
                self.variable_types.insert(let_in.name.clone(), var_type);

                // Generate code for the body expression
                let result = self.generate_expression(&let_in.body)?;

                // Clean up variable from cache
                self.variable_cache.remove(&let_in.name);
                self.variable_types.remove(&let_in.name);

                Ok(result)
            }
            Expression::FieldAccess(expr, field) => {
                let record_value = self.generate_expression(expr)?;
                
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
                
                // Use SPIR-V CompositeExtract for vectors
                // For vec3/vec4, the component type is f32
                let extracted_id = self.builder.composite_extract(
                    self.f32_type, 
                    None, 
                    record_value.id, 
                    vec![component_index]
                )?;
                
                Ok(Value {
                    id: extracted_id,
                    type_id: self.f32_type,
                })
            }
            Expression::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    return Err(CompilerError::SpirvError("Empty array literals not supported".to_string()));
                }

                // Generate all element values
                let mut element_values = Vec::new();
                let mut element_type_id = None;
                for elem in elements {
                    let val = self.generate_expression(elem)?;
                    element_values.push(val.id);
                    if element_type_id.is_none() {
                        element_type_id = Some(val.type_id);
                    }
                }

                let elem_type = element_type_id.unwrap();
                
                // Create array type
                let length_id = self.builder.constant_bit32(self.i32_type, elements.len() as u32);
                let array_type = self.builder.type_array(elem_type, length_id);

                // Build the array using composite construct
                let array_id = self.builder.composite_construct(array_type, None, element_values)?;

                Ok(Value {
                    id: array_id,
                    type_id: array_type,
                })
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

                let array_value = if let Some(&value) = self.variable_cache.get(array_name) {
                    value
                } else {
                    return Err(CompilerError::UndefinedVariable(array_name.clone()));
                };

                let index = self.generate_expression(index_expr)?;

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

                let element_type_info = self.get_or_create_type(&element_type)?;

                // Create pointer type for array element access
                let element_ptr_type = self.builder.type_pointer(None, StorageClass::Function, element_type_info.id);
                
                // Use AccessChain to get element pointer
                let element_ptr = self.builder.access_chain(
                    element_ptr_type,
                    None,
                    array_value.id,
                    vec![index.id]
                )?;

                // Load the element
                let element_id = self.builder.load(element_type_info.id, None, element_ptr, None, vec![])?;

                Ok(Value {
                    id: element_id,
                    type_id: element_type_info.id,
                })
            }
            _ => Err(CompilerError::SpirvError(
                "Expression type not yet implemented for rspirv backend".to_string(),
            )),
        }
    }

    fn generate_vector_constructor(
        &mut self,
        size: u32,
        args: &[Expression],
    ) -> Result<Value> {
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
            elements.push(val.id);
        }

        // Create the vector type - assume f32 elements for now
        let vec_type = self.builder.type_vector(self.f32_type, size);

        // Build the vector using composite construct
        let vector_id = self.builder.composite_construct(vec_type, None, elements)?;

        Ok(Value {
            id: vector_id,
            type_id: vec_type,
        })
    }

    fn convert_array_to_vec4(&mut self, array_val: Value) -> Result<Value> {
        // For rspirv, we need to extract array elements and build a vector
        let vec4_type = self.builder.type_vector(self.f32_type, 4);

        // Extract elements from the array and build vector
        let mut elements = Vec::new();
        for i in 0..4 {
            let element_id = self.builder.composite_extract(self.f32_type, None, array_val.id, vec![i])?;
            elements.push(element_id);
        }

        // Build the vector
        let vector_id = self.builder.composite_construct(vec4_type, None, elements)?;

        Ok(Value {
            id: vector_id,
            type_id: vec4_type,
        })
    }

    fn get_or_create_type(&mut self, ty: &Type) -> Result<TypeInfo> {
        if let Some(cached) = self.type_cache.get(ty) {
            return Ok(cached.clone());
        }

        let (type_id, size_bytes) = match ty {
            Type::Constructed(name, args) => {
                match name {
                    TypeName::Str("int") => (self.i32_type, 4),
                    TypeName::Str("float") => (self.f32_type, 4),
                    
                    // f32 vector types
                    TypeName::Str("vec2") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 2);
                        (vec_type, 8)
                    }
                    TypeName::Str("vec3") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 3);
                        (vec_type, 12)
                    }
                    TypeName::Str("vec4") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 4);
                        (vec_type, 16)
                    }
                    
                    // i32 vector types
                    TypeName::Str("ivec2") => {
                        let vec_type = self.builder.type_vector(self.i32_type, 2);
                        (vec_type, 8)
                    }
                    TypeName::Str("ivec3") => {
                        let vec_type = self.builder.type_vector(self.i32_type, 3);
                        (vec_type, 12)
                    }
                    TypeName::Str("ivec4") => {
                        let vec_type = self.builder.type_vector(self.i32_type, 4);
                        (vec_type, 16)
                    }
                    TypeName::Str("array") => {
                        let elem_ty = args.first().ok_or_else(|| {
                            CompilerError::SpirvError("Array type missing element type".to_string())
                        })?;
                        let elem_type_info = self.get_or_create_type(elem_ty)?;

                        // For unsized arrays, use default size of 1
                        let length_id = self.builder.constant_bit32(self.i32_type, 1);
                        let array_type = self.builder.type_array(elem_type_info.id, length_id);
                        (array_type, elem_type_info.size_bytes)
                    }
                    TypeName::Array("array", size) => {
                        let elem_ty = args.first().ok_or_else(|| {
                            CompilerError::SpirvError("Array type missing element type".to_string())
                        })?;
                        let elem_type_info = self.get_or_create_type(elem_ty)?;

                        // Use the actual size from the type
                        let length_id = self.builder.constant_bit32(self.i32_type, *size as u32);
                        let array_type = self.builder.type_array(elem_type_info.id, length_id);
                        (array_type, elem_type_info.size_bytes * (*size as u32))
                    }
                    TypeName::Str("tuple") | TypeName::Str("attributed_tuple") => {
                        // Create a struct type with the component types
                        let mut component_type_ids = Vec::new();
                        let mut total_size = 0;
                        for arg in args {
                            let component_info = self.get_or_create_type(arg)?;
                            component_type_ids.push(component_info.id);
                            total_size += component_info.size_bytes;
                        }
                        let struct_type = self.builder.type_struct(component_type_ids);
                        (struct_type, total_size)
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
                    "Type {:?} not supported in SPIR-V generation",
                    ty
                )));
            }
        };

        let type_info = TypeInfo {
            id: type_id,
            size_bytes,
        };
        
        self.type_cache.insert(ty.clone(), type_info.clone());
        Ok(type_info)
    }

    fn add_spirv_entry_point_metadata(
        &mut self,
        name: &str,
        exec_model: &spirv::ExecutionModel,
    ) -> Result<()> {
        // Add SPIR-V entry point
        let func_id = self
            .function_cache
            .get(name)
            .copied()
            .ok_or_else(|| CompilerError::SpirvError(format!("Entry point {} not found", name)))?;

        // Get interface variables (uniforms and outputs)
        let mut interface_vars = Vec::new();
        for (_, value) in &self.uniform_globals {
            interface_vars.push(value.id);
        }
        for (_, (value, _, _)) in &self.output_globals {
            interface_vars.push(value.id);
        }

        self.builder.entry_point(*exec_model, func_id, name, interface_vars);
        
        println!("DEBUG: Added SPIR-V entry point metadata for '{}' with execution model {:?}", name, exec_model);

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
                let type_info = self.get_or_create_type(component_type)?;
                let ptr_type = self.builder.type_pointer(None, StorageClass::Output, type_info.id);
                let global_id = self.builder.variable(ptr_type, None, StorageClass::Output, None);
                
                let global_value = Value {
                    id: global_id,
                    type_id: ptr_type,
                };
                
                // Store for decoration later
                let global_name = format!("output_{}", index);
                self.output_globals.insert(global_name.clone(), (global_value, location, type_info.id));
                
                println!("DEBUG: Created output global '{}' at location {}", global_name, location);
            }
        }
        Ok(())
    }

    fn add_spirv_decorations(&mut self) -> Result<()> {
        // Add SPIR-V decorations for uniform variables
        for (name, value) in &self.uniform_globals.clone() {
            self.add_uniform_decoration(name, *value)?;
        }

        // Add SPIR-V decorations for output variables
        for (name, (value, location, _)) in &self.output_globals.clone() {
            self.add_output_decoration(name, *value, *location)?;
        }

        Ok(())
    }

    fn add_uniform_decoration(&mut self, name: &str, value: Value) -> Result<()> {
        // For SPIR-V uniforms, add proper decorations
        // Add descriptor set and binding decorations
        self.builder.decorate(value.id, Decoration::DescriptorSet, vec![Operand::LiteralBit32(0)]);
        self.builder.decorate(value.id, Decoration::Binding, vec![Operand::LiteralBit32(0)]);
        
        println!("DEBUG: Added uniform decoration for '{}'", name);
        Ok(())
    }

    fn add_output_decoration(&mut self, name: &str, value: Value, location: u32) -> Result<()> {
        // For SPIR-V outputs, add location decorations
        self.builder.decorate(value.id, Decoration::Location, vec![Operand::LiteralBit32(location)]);
        
        println!("DEBUG: Added output decoration for '{}' at location {}", name, location);
        Ok(())
    }

    fn emit_spirv(mut self) -> Result<Vec<u32>> {
        // Build the module (takes ownership of builder)
        self.module = self.builder.module();
        
        // Convert module to SPIR-V words
        let words = self.module.assemble();
        
        println!("DEBUG: Generated {} words of SPIR-V", words.len());
        
        Ok(words)
    }
}