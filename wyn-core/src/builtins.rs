use crate::ast::{Type, TypeName};
use crate::error::{CompilerError, Result};
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::{Module, Linkage};
use std::collections::HashMap;

/// Types of builtin implementations
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinType {
    /// LLVM IR template that gets loaded into module
    LlvmIrTemplate(String),
    /// Binary operation intrinsic with base name (e.g., "fadd", "mul")
    /// Type info is used to generate full intrinsic name
    BinOpIntrinsic(String),
}

/// Manages builtin function implementations using different strategies
pub struct BuiltinManager<'ctx> {
    context: &'ctx Context,
    specialized_functions: HashMap<String, String>, // Maps function signature to specialized IR
    builtin_registry: HashMap<String, BuiltinType>, // Maps builtin name to implementation type
}

impl<'ctx> BuiltinManager<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let mut registry = HashMap::new();
        
        // Register builtin functions with their implementation types
        
        // LLVM IR template builtins
        registry.insert("length".to_string(), BuiltinType::LlvmIrTemplate(Self::generate_length_builtin()));
        registry.insert("sin".to_string(), BuiltinType::LlvmIrTemplate(Self::generate_sin_builtin()));
        registry.insert("cos".to_string(), BuiltinType::LlvmIrTemplate(Self::generate_cos_builtin()));
        registry.insert("tan".to_string(), BuiltinType::LlvmIrTemplate(Self::generate_tan_builtin()));
        
        // Binary operation intrinsics
        registry.insert("+".to_string(), BuiltinType::BinOpIntrinsic("fadd".to_string()));
        registry.insert("-".to_string(), BuiltinType::BinOpIntrinsic("fsub".to_string()));
        registry.insert("*".to_string(), BuiltinType::BinOpIntrinsic("fmul".to_string()));
        registry.insert("/".to_string(), BuiltinType::BinOpIntrinsic("fdiv".to_string()));
        
        Self {
            context,
            specialized_functions: HashMap::new(),
            builtin_registry: registry,
        }
    }

    /// Load builtin functions into a module
    pub fn load_builtins_into_module(&mut self, module: &Module<'ctx>) -> Result<()> {
        // Load all LLVM IR template builtins
        for (name, builtin_type) in &self.builtin_registry.clone() {
            if let BuiltinType::LlvmIrTemplate(ir) = builtin_type {
                self.add_ir_to_module(module, ir)?;
                
                // Set to private linkage to avoid SPIR-V export
                if let Some(func) = module.get_function(name) {
                    func.set_linkage(Linkage::Private);
                    println!("DEBUG: Set {} function to private linkage", name);
                }
            }
            // BinOpIntrinsic functions are generated on-demand during codegen
        }

        Ok(())
    }

    /// Generate a specialized map function for a given element type
    pub fn generate_map_specialization(
        &mut self,
        module: &Module<'ctx>,
        element_type: &Type,
    ) -> Result<String> {
        let type_name = self.type_to_llvm_name(element_type)?;
        let function_name = format!("map_{}", type_name);

        // Check if already generated
        if module.get_function(&function_name).is_some() {
            return Ok(function_name);
        }

        // Generate specialized IR
        let map_ir = self.generate_map_builtin_for_type(&type_name)?;
        self.add_ir_to_module(module, &map_ir)?;

        self.specialized_functions
            .insert(function_name.clone(), map_ir);
        Ok(function_name)
    }

    /// Generate LLVM IR for length builtin - works with array metadata
    fn generate_length_builtin() -> String {
        r#"
; Get the length of an array
; Assumes arrays are represented as { i32, [0 x element_type]* }
; where the first i32 is the length
define i32 @length(i8* %array_ptr) {
entry:
  ; Cast to { i32, i8* }* to access the length field
  %array_struct = bitcast i8* %array_ptr to { i32, i8* }*
  
  ; Get pointer to length field (first element)
  %length_ptr = getelementptr { i32, i8* }, { i32, i8* }* %array_struct, i32 0, i32 0
  
  ; Load the length
  %length = load i32, i32* %length_ptr
  
  ret i32 %length
}
        "#
        .to_string()
    }

    /// Generate LLVM IR for sin builtin function
    fn generate_sin_builtin() -> String {
        r#"
; Sine function for f32
; Uses LLVM's sin intrinsic
declare float @llvm.sin.f32(float)

define float @sin(float %x) {
entry:
  %result = call float @llvm.sin.f32(float %x)
  ret float %result
}
        "#
        .to_string()
    }

    /// Generate LLVM IR for cos builtin function
    fn generate_cos_builtin() -> String {
        r#"
; Cosine function for f32
; Uses LLVM's cos intrinsic
declare float @llvm.cos.f32(float)

define float @cos(float %x) {
entry:
  %result = call float @llvm.cos.f32(float %x)
  ret float %result
}
        "#
        .to_string()
    }

    /// Generate LLVM IR for tan builtin function
    fn generate_tan_builtin() -> String {
        r#"
; Tangent function for f32
; Uses LLVM's tan intrinsic
declare float @llvm.tan.f32(float)

define float @tan(float %x) {
entry:
  %result = call float @llvm.tan.f32(float %x)
  ret float %result
}
        "#
        .to_string()
    }

    /// Generate LLVM IR for map builtin specialized for a specific type
    fn generate_map_builtin_for_type(&self, type_name: &str) -> Result<String> {
        let template = r#"
; Map function for {{TYPE}} arrays
; Takes: function pointer, input array, output array (pre-allocated)
define void @map_{{TYPE}}({{TYPE}} ({{TYPE}})* %func, i8* %input_ptr, i8* %output_ptr) {
entry:
  ; Get input array length
  %input_struct = bitcast i8* %input_ptr to { i32, {{TYPE}}* }*
  %input_len_ptr = getelementptr { i32, {{TYPE}}* }, { i32, {{TYPE}}* }* %input_struct, i32 0, i32 0
  %len = load i32, i32* %input_len_ptr
  
  ; Get input data pointer
  %input_data_ptr_ptr = getelementptr { i32, {{TYPE}}* }, { i32, {{TYPE}}* }* %input_struct, i32 0, i32 1
  %input_data_ptr = load {{TYPE}}*, {{TYPE}}** %input_data_ptr_ptr
  
  ; Get output array structure and data pointer
  %output_struct = bitcast i8* %output_ptr to { i32, {{TYPE}}* }*
  %output_len_ptr = getelementptr { i32, {{TYPE}}* }, { i32, {{TYPE}}* }* %output_struct, i32 0, i32 0
  store i32 %len, i32* %output_len_ptr
  
  %output_data_ptr_ptr = getelementptr { i32, {{TYPE}}* }, { i32, {{TYPE}}* }* %output_struct, i32 0, i32 1
  %output_data_ptr = load {{TYPE}}*, {{TYPE}}** %output_data_ptr_ptr
  
  ; Initialize loop counter
  %i_ptr = alloca i32
  store i32 0, i32* %i_ptr
  br label %loop_header

loop_header:
  %i = load i32, i32* %i_ptr
  %cond = icmp ult i32 %i, %len
  br i1 %cond, label %loop_body, label %loop_exit

loop_body:
  ; Load element from input array
  %input_elem_ptr = getelementptr {{TYPE}}, {{TYPE}}* %input_data_ptr, i32 %i
  %input_elem = load {{TYPE}}, {{TYPE}}* %input_elem_ptr
  
  ; Apply function
  %result = call {{TYPE}} %func({{TYPE}} %input_elem)
  
  ; Store result in output array
  %output_elem_ptr = getelementptr {{TYPE}}, {{TYPE}}* %output_data_ptr, i32 %i
  store {{TYPE}} %result, {{TYPE}}* %output_elem_ptr
  
  ; Increment counter
  %next_i = add i32 %i, 1
  store i32 %next_i, i32* %i_ptr
  br label %loop_header

loop_exit:
  ret void
}
        "#;

        Ok(template.replace("{{TYPE}}", type_name))
    }

    /// Add LLVM IR to a module by parsing it
    fn add_ir_to_module(&self, module: &Module<'ctx>, ir: &str) -> Result<()> {
        // Parse the IR and link it into the module
        let memory_buffer = MemoryBuffer::create_from_memory_range(ir.as_bytes(), "builtin_ir");
        let temp_module = self
            .context
            .create_module_from_ir(memory_buffer)
            .map_err(|e| {
                CompilerError::SpirvError(format!("Failed to parse builtin IR: {:?}", e))
            })?;

        module.link_in_module(temp_module).map_err(|e| {
            CompilerError::SpirvError(format!("Failed to link builtin module: {:?}", e))
        })?;

        Ok(())
    }

    /// Convert Wyn type to LLVM type name
    fn type_to_llvm_name(&self, ty: &Type) -> Result<String> {
        match ty {
            Type::Constructed(TypeName::Str("int"), _) => Ok("i32".to_string()),
            Type::Constructed(TypeName::Str("float"), _) => Ok("float".to_string()),
            Type::Constructed(TypeName::Str("bool"), _) => Ok("i1".to_string()),
            Type::Variable(_) => Ok("i32".to_string()), // Default to i32 for type variables
            Type::Constructed(name, _) => Err(CompilerError::TypeError(format!(
                "Unsupported type for LLVM builtin: {:?}",
                name
            ))),
        }
    }

    /// Get the function name for a builtin
    pub fn get_builtin_function_name(
        &self,
        builtin: &str,
        element_type: Option<&Type>,
    ) -> Result<String> {
        match builtin {
            "length" => Ok("length".to_string()),
            "map" => {
                let element_type = element_type.ok_or_else(|| {
                    CompilerError::TypeError("map builtin requires element type".to_string())
                })?;
                let type_name = self.type_to_llvm_name(element_type)?;
                Ok(format!("map_{}", type_name))
            }
            _ => Err(CompilerError::TypeError(format!(
                "Unknown builtin: {}",
                builtin
            ))),
        }
    }

    /// Check if a function name represents a builtin
    pub fn is_builtin(&self, name: &str) -> bool {
        self.builtin_registry.contains_key(name) || matches!(name, "map") || name.starts_with("map_")
    }
    
    /// Get the builtin type for a given name
    pub fn get_builtin_type(&self, name: &str) -> Option<&BuiltinType> {
        self.builtin_registry.get(name)
    }
    
    /// Generate LLVM intrinsic name with type information
    pub fn generate_intrinsic_name(&self, base_name: &str, operand_type: &Type) -> Result<String> {
        let type_suffix = match operand_type {
            Type::Constructed(TypeName::Str("int"), _) => "i32",
            Type::Constructed(TypeName::Str("float"), _) => "f32",
            Type::Constructed(TypeName::Str("vec2"), _) => "v2f32", 
            Type::Constructed(TypeName::Str("vec3"), _) => "v3f32",
            Type::Constructed(TypeName::Str("vec4"), _) => "v4f32",
            _ => return Err(CompilerError::TypeError(format!(
                "Unsupported type for intrinsic {}: {:?}", base_name, operand_type
            ))),
        };
        
        Ok(format!("llvm.{}.{}", base_name, type_suffix))
    }

    /// Generate a call to a builtin function
    pub fn generate_builtin_call(
        &mut self,
        module: &Module<'ctx>,
        builder: &inkwell::builder::Builder<'ctx>,
        func_name: &str,
        args: &[inkwell::values::BasicValueEnum<'ctx>],
    ) -> Result<inkwell::values::BasicValueEnum<'ctx>> {
        use inkwell::AddressSpace;

        match func_name {
            "length" => {
                if args.len() != 1 {
                    return Err(CompilerError::SpirvError(
                        "length builtin requires exactly one argument".to_string(),
                    ));
                }

                // Get the length function
                let length_fn = module.get_function("length").ok_or_else(|| {
                    CompilerError::SpirvError("length builtin function not found".to_string())
                })?;

                // Convert array to i8* for the builtin call
                let array_val = args[0];
                let array_ptr = array_val.into_pointer_value();
                let i8_ptr_type = self.context.ptr_type(AddressSpace::default());
                let array_as_i8_ptr = builder
                    .build_bit_cast(array_ptr, i8_ptr_type, "array_as_i8_ptr")
                    .map_err(|e| {
                        CompilerError::SpirvError(format!(
                            "Failed to cast array for length call: {}",
                            e
                        ))
                    })?;

                // Call the length function
                let call_result = builder
                    .build_call(length_fn, &[array_as_i8_ptr.into()], "length_result")
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to call length builtin: {}", e))
                    })?;

                Ok(call_result.try_as_basic_value().left().ok_or_else(|| {
                    CompilerError::SpirvError("length call returned void".to_string())
                })?)
            }
            "map" => {
                if args.len() != 2 {
                    return Err(CompilerError::SpirvError(
                        "map builtin requires exactly two arguments (function, array)".to_string(),
                    ));
                }

                let func_val = args[0];
                let array_val = args[1];

                // Determine element type (for now, assume i32)
                let element_type = Type::Constructed(TypeName::Str("int"), vec![]);

                // Generate or get the specialized map function
                let map_fn_name = self.generate_map_specialization(module, &element_type)?;

                let map_fn = module.get_function(&map_fn_name).ok_or_else(|| {
                    CompilerError::SpirvError(format!(
                        "map builtin function {} not found",
                        map_fn_name
                    ))
                })?;

                // Convert arrays to i8* for the builtin call
                let array_ptr = array_val.into_pointer_value();
                let i8_ptr_type = self.context.ptr_type(AddressSpace::default());
                let input_as_i8_ptr = builder
                    .build_bit_cast(array_ptr, i8_ptr_type, "input_as_i8_ptr")
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to cast input array: {}", e))
                    })?;

                // Allocate output array (same type as input for now)
                let output_array = builder
                    .build_alloca(array_val.get_type(), "output_array")
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to allocate output array: {}", e))
                    })?;
                let output_as_i8_ptr = builder
                    .build_bit_cast(output_array, i8_ptr_type, "output_as_i8_ptr")
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to cast output array: {}", e))
                    })?;

                // Call the map function
                let _call_result = builder
                    .build_call(
                        map_fn,
                        &[
                            func_val.into(),
                            input_as_i8_ptr.into(),
                            output_as_i8_ptr.into(),
                        ],
                        "map_result",
                    )
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to call map builtin: {}", e))
                    })?;

                // Return the output array
                Ok(output_array.into())
            }
            "sin" | "cos" | "tan" => {
                if args.len() != 1 {
                    return Err(CompilerError::SpirvError(format!(
                        "{} builtin requires exactly one argument",
                        func_name
                    )));
                }

                // Get the trigonometric function
                let trig_fn = module.get_function(func_name).ok_or_else(|| {
                    CompilerError::SpirvError(format!("{} builtin function not found", func_name))
                })?;

                // Call the function with the float argument
                let result = builder
                    .build_call(trig_fn, &[args[0].into()], &format!("{}_result", func_name))
                    .map_err(|e| {
                        CompilerError::SpirvError(format!("Failed to call {} builtin: {}", func_name, e))
                    })?;

                Ok(result.try_as_basic_value().left().unwrap())
            }
            _ => Err(CompilerError::SpirvError(format!(
                "Unknown builtin function: {}",
                func_name
            ))),
        }
    }

    /// Get type signature for builtin functions
    pub fn get_builtin_type_signature(
        &self,
        name: &str,
        element_type: Option<&Type>,
    ) -> Result<(Vec<Type>, Type)> {
        match name {
            "length" => {
                // length: [a] -> int
                let array_type = Type::Constructed(TypeName::Str("array"), vec![Type::Variable(0)]);
                let int_type = Type::Constructed(TypeName::Str("int"), vec![]);
                Ok((vec![array_type], int_type))
            }
            "map" => {
                // map: (a -> b) -> [a] -> [b]
                let element_type = element_type.ok_or_else(|| {
                    CompilerError::TypeError("map requires element type context".to_string())
                })?;

                let func_type = Type::arrow(element_type.clone(), element_type.clone());
                let input_array_type =
                    Type::Constructed(TypeName::Str("array"), vec![element_type.clone()]);
                let output_array_type =
                    Type::Constructed(TypeName::Str("array"), vec![element_type.clone()]);

                Ok((vec![func_type, input_array_type], output_array_type))
            }
            _ => Err(CompilerError::TypeError(format!(
                "Unknown builtin: {}",
                name
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn test_builtin_manager_creation() {
        let context = Context::create();
        let manager = BuiltinManager::new(&context);

        assert!(manager.is_builtin("length"));
        assert!(manager.is_builtin("map"));
        assert!(manager.is_builtin("sin"));
        assert!(manager.is_builtin("cos"));
        assert!(manager.is_builtin("tan"));
        assert!(!manager.is_builtin("unknown"));
    }

    #[test]
    fn test_type_conversion() {
        let context = Context::create();
        let manager = BuiltinManager::new(&context);

        let int_type = Type::Constructed(TypeName::Str("int"), vec![]);
        assert_eq!(manager.type_to_llvm_name(&int_type).unwrap(), "i32");

        let float_type = Type::Constructed(TypeName::Str("float"), vec![]);
        assert_eq!(manager.type_to_llvm_name(&float_type).unwrap(), "float");
    }

    #[test]
    fn test_builtin_function_names() {
        let context = Context::create();
        let manager = BuiltinManager::new(&context);

        assert_eq!(
            manager.get_builtin_function_name("length", None).unwrap(),
            "length"
        );

        let int_type = Type::Constructed(TypeName::Str("int"), vec![]);
        assert_eq!(
            manager
                .get_builtin_function_name("map", Some(&int_type))
                .unwrap(),
            "map_i32"
        );
    }
}
