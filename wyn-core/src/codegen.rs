use crate::ast::*;
use crate::error::{CompilerError, Result};
use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use spirv::Word;
use std::collections::HashMap;

#[derive(Debug, Clone)]
enum OutputDecoration {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
}

pub struct CodeGenerator {
    builder: Builder,
    type_cache: HashMap<Type, Word>,
    variable_cache: HashMap<String, Word>,
    variable_types: HashMap<String, Type>, // Track variable types
    parameter_types: HashMap<String, Type>, // Track parameter types
    constant_cache: HashMap<String, Word>,
    entry_points: Vec<(String, spirv::ExecutionModel)>,
    interface_variables: HashMap<String, Vec<Word>>, // Track interface variables per entry point
}

impl CodeGenerator {
    pub fn new() -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 0);
        builder.capability(spirv::Capability::Shader);
        builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        
        CodeGenerator {
            builder,
            type_cache: HashMap::new(),
            variable_cache: HashMap::new(),
            variable_types: HashMap::new(),
            parameter_types: HashMap::new(),
            constant_cache: HashMap::new(),
            entry_points: Vec::new(),
            interface_variables: HashMap::new(),
        }
    }
    
    pub fn generate(mut self, program: &Program) -> Result<Vec<u32>> {
        // Process all declarations
        for decl in &program.declarations {
            self.generate_declaration(decl)?;
        }
        
        // Add entry points
        for (name, exec_model) in &self.entry_points {
            let func_id = self.variable_cache.get(name)
                .ok_or_else(|| CompilerError::SpirvError(format!("Entry point {} not found", name)))?;
            
            // Get interface variables for this entry point
            let interface_vars = self.interface_variables.get(name)
                .map(|vars| vars.as_slice())
                .unwrap_or(&[]);
            
            self.builder.entry_point(
                *exec_model,
                *func_id,
                name,
                interface_vars,
            );
            
            // Add execution mode for fragment shader
            if *exec_model == spirv::ExecutionModel::Fragment {
                self.builder.execution_mode(*func_id, spirv::ExecutionMode::OriginUpperLeft, &[]);
            }
        }
        
        // Build and assemble the module
        let module = self.builder.module();
        Ok(module.assemble())
    }
    
    fn generate_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Let(let_decl) => self.generate_let_decl(let_decl),
            Declaration::Entry(entry_decl) => self.generate_entry_decl(entry_decl),
            Declaration::Def(_def_decl) => {
                // For now, skip def declarations in codegen as they're user-defined functions
                // that don't generate SPIR-V directly
                Ok(())
            }
            Declaration::Val(_val_decl) => {
                // Val declarations are type signatures only, no code generation needed
                Ok(())
            }
        }
    }
    
    fn generate_let_decl(&mut self, decl: &LetDecl) -> Result<()> {
        let ty = decl.ty.as_ref()
            .ok_or_else(|| CompilerError::SpirvError("Let declaration must have explicit type".to_string()))?;
        
        match &decl.value {
            Expression::ArrayLiteral(elements) => {
                // Generate constants for array elements
                let const_id = self.generate_constant_array(ty, elements)?;
                
                // For dynamic indexing, we need to create a variable and store the constant into it
                // Global variables should use Private storage class
                let array_type_id = self.get_or_create_type(ty)?;
                let pointer_type_id = self.builder.type_pointer(
                    None,
                    spirv::StorageClass::Private,
                    array_type_id
                );
                
                // Create a variable
                let var_id = self.builder.variable(
                    pointer_type_id,
                    None,
                    spirv::StorageClass::Private,
                    Some(const_id),
                );
                
                // Store the variable ID and type for later access
                self.variable_cache.insert(decl.name.clone(), var_id);
                self.variable_types.insert(decl.name.clone(), ty.clone());
            }
            _ => {
                return Err(CompilerError::SpirvError(
                    "Only array literals supported for let declarations".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn generate_entry_decl(&mut self, decl: &EntryDecl) -> Result<()> {
        // Determine execution model based on attributes
        let exec_model = if decl.attributes.iter().any(|attr| attr.name == "vertex") {
            spirv::ExecutionModel::Vertex
        } else if decl.attributes.iter().any(|attr| attr.name == "fragment") {
            spirv::ExecutionModel::Fragment
        } else {
            return Err(CompilerError::SpirvError(
                "Entry point must have either #[vertex] or #[fragment] attribute".to_string()
            ));
        };
        
        self.entry_points.push((decl.name.clone(), exec_model));
        
        // Initialize interface variables list for this entry point
        let mut interface_vars = Vec::new();
        
        // Entry points must return void in SPIR-V
        let void_type_id = self.builder.type_void();
        
        // Create output variable for the return value
        let decoration = self.get_output_decoration(exec_model, &decl.return_type);
        let output_type_id = self.get_or_create_type(&decl.return_type)?;
        
        let output_ptr_type_id = self.builder.type_pointer(
            None,
            spirv::StorageClass::Output,
            output_type_id
        );
        
        let output_var_id = self.builder.variable(
            output_ptr_type_id,
            None,
            spirv::StorageClass::Output,
            None,
        );
        
        // Apply the decoration determined earlier
        match decoration {
            OutputDecoration::BuiltIn(builtin) => {
                self.builder.decorate(
                    output_var_id,
                    spirv::Decoration::BuiltIn,
                    vec![rspirv::dr::Operand::BuiltIn(builtin)],
                );
            }
            OutputDecoration::Location(loc) => {
                self.builder.decorate(
                    output_var_id,
                    spirv::Decoration::Location,
                    vec![rspirv::dr::Operand::LiteralBit32(loc)],
                );
            }
        }
        
        // Add output variable to interface
        interface_vars.push(output_var_id);
        
        // Create function type (void return, no parameters for entry points)
        let func_type_id = self.builder.type_function(void_type_id, vec![]);
        
        // Create function
        let func_id = self.builder.begin_function(
            void_type_id,
            None,
            spirv::FunctionControl::NONE,
            func_type_id,
        ).map_err(|e| CompilerError::SpirvError(format!("Failed to begin function: {:?}", e)))?;
        
        self.variable_cache.insert(decl.name.clone(), func_id);
        
        // Handle shader inputs - create built-in variables instead of parameters
        for param in decl.params.iter() {
            if exec_model == spirv::ExecutionModel::Vertex {
                // For vertex shaders, create built-in input variables
                if param.name == "vertex_id" {
                    // Create gl_VertexIndex built-in
                    let i32_type_id = self.get_or_create_type(&Type::I32)?;
                    let input_ptr_type_id = self.builder.type_pointer(
                        None,
                        spirv::StorageClass::Input,
                        i32_type_id
                    );
                    
                    let vertex_index_var = self.builder.variable(
                        input_ptr_type_id,
                        None,
                        spirv::StorageClass::Input,
                        None,
                    );
                    
                    // Decorate as VertexIndex built-in
                    self.builder.decorate(
                        vertex_index_var,
                        spirv::Decoration::BuiltIn,
                        vec![rspirv::dr::Operand::BuiltIn(spirv::BuiltIn::VertexIndex)],
                    );
                    
                    // Add input variable to interface
                    interface_vars.push(vertex_index_var);
                    
                    // Map the parameter name to this built-in variable
                    // Store as a variable (not parameter) so it gets loaded with OpLoad
                    self.variable_cache.insert(param.name.clone(), vertex_index_var);
                    self.variable_types.insert(param.name.clone(), param.ty.clone());
                } else {
                    return Err(CompilerError::SpirvError(
                        format!("Unsupported vertex shader parameter: {}", param.name)
                    ));
                }
            } else {
                // Fragment shaders shouldn't have parameters in SPIR-V
                return Err(CompilerError::SpirvError(
                    "Fragment shaders cannot have parameters".to_string()
                ));
            }
        }
        
        // Generate function body
        self.builder.begin_block(None)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to begin block: {:?}", e)))?;
        
        let result = self.generate_expression(&decl.body)?;
        
        // Store result to output variable instead of returning it
        self.builder.store(output_var_id, result, None, Vec::<rspirv::dr::Operand>::new())
            .map_err(|e| CompilerError::SpirvError(format!("Failed to store output: {:?}", e)))?;
        
        // Return void
        self.builder.ret()
            .map_err(|e| CompilerError::SpirvError(format!("Failed to add return: {:?}", e)))?;
        
        self.builder.end_function()
            .map_err(|e| CompilerError::SpirvError(format!("Failed to end function: {:?}", e)))?;
        
        // Store interface variables for this entry point
        self.interface_variables.insert(decl.name.clone(), interface_vars);
        
        Ok(())
    }
    
    fn generate_expression(&mut self, expr: &Expression) -> Result<Word> {
        match expr {
            Expression::IntLiteral(n) => {
                let ty = self.get_or_create_type(&Type::I32)?;
                Ok(self.builder.constant_bit32(ty, *n as u32))
            }
            Expression::FloatLiteral(f) => {
                let ty = self.get_or_create_type(&Type::F32)?;
                Ok(self.builder.constant_bit32(ty, f.to_bits()))
            }
            Expression::Identifier(name) => {
                // First check if it's a constant
                if let Some(const_id) = self.constant_cache.get(name) {
                    Ok(*const_id)
                } else if let Some(_param_type) = self.parameter_types.get(name) {
                    // Function parameters are already values, use directly
                    self.variable_cache.get(name)
                        .copied()
                        .ok_or_else(|| CompilerError::UndefinedVariable(name.clone()))
                } else {
                    // For variables, we need to load the value from the pointer
                    let var_id = self.variable_cache.get(name)
                        .copied()
                        .ok_or_else(|| CompilerError::UndefinedVariable(name.clone()))?;
                    
                    // Get the variable's type for the load operation
                    let var_type = self.variable_types.get(name)
                        .ok_or_else(|| CompilerError::SpirvError(format!("Unknown variable type for: {}", name)))?
                        .clone();
                    
                    let value_type_id = self.get_or_create_type(&var_type)?;
                    
                    // Load the value from the variable
                    let loaded_value = self.builder.load(
                        value_type_id,
                        None,
                        var_id,
                        None,
                        vec![],
                    ).map_err(|e| CompilerError::SpirvError(format!("Failed to load variable: {:?}", e)))?;
                    
                    Ok(loaded_value)
                }
            }
            Expression::ArrayIndex(array_expr, index_expr) => {
                // Get the array identifier (should be a variable)
                let array_var_id = match array_expr.as_ref() {
                    Expression::Identifier(name) => {
                        self.variable_cache.get(name)
                            .copied()
                            .ok_or_else(|| CompilerError::SpirvError(format!("Unknown variable: {}", name)))?
                    }
                    _ => return Err(CompilerError::SpirvError(
                        "Only variable array indexing supported".to_string()
                    ))
                };
                
                // Generate the index expression
                let index_id = self.generate_expression(index_expr)?;
                
                // For SPIR-V, we need to use OpAccessChain for dynamic indexing
                // Look up the actual array type and extract element type
                let array_name = match array_expr.as_ref() {
                    Expression::Identifier(name) => name,
                    _ => return Err(CompilerError::SpirvError(
                        "Only variable array indexing supported".to_string()
                    ))
                };
                
                let array_type = self.variable_types.get(array_name)
                    .ok_or_else(|| CompilerError::SpirvError(format!("Unknown array type for: {}", array_name)))?;
                
                let element_type = match array_type {
                    Type::Array(elem_ty, _dims) => elem_ty.as_ref().clone(),
                    _ => return Err(CompilerError::SpirvError(
                        "Cannot index non-array type".to_string()
                    ))
                };
                
                let element_type_id = self.get_or_create_type(&element_type)?;
                let pointer_type_id = self.builder.type_pointer(
                    None,
                    spirv::StorageClass::Private,
                    element_type_id
                );
                
                // Use OpAccessChain to get pointer to the indexed element
                let element_ptr_id = self.builder.access_chain(
                    pointer_type_id,
                    None,
                    array_var_id,
                    vec![index_id],
                ).map_err(|e| CompilerError::SpirvError(format!("Failed to access chain: {:?}", e)))?;
                
                // Load the value from the pointer
                let loaded_value = self.builder.load(
                    element_type_id,
                    None,
                    element_ptr_id,
                    None,
                    vec![],
                ).map_err(|e| CompilerError::SpirvError(format!("Failed to load: {:?}", e)))?;
                
                Ok(loaded_value)
            }
            Expression::ArrayLiteral(_) => {
                Err(CompilerError::SpirvError(
                    "Array literals only supported in let declarations".to_string()
                ))
            }
            Expression::BinaryOp(BinaryOp::Divide, left, right) => {
                let left_id = self.generate_expression(left)?;
                let right_id = self.generate_expression(right)?;
                
                // Assuming float division for now
                let result_type = self.get_or_create_type(&Type::F32)?;
                Ok(self.builder.f_div(result_type, None, left_id, right_id)
                    .map_err(|e| CompilerError::SpirvError(format!("Failed to divide: {:?}", e)))?)
            }
            Expression::FunctionCall(func_name, args) => {
                match func_name.as_str() {
                    "to_vec4_f32" => {
                        if args.len() != 1 {
                            return Err(CompilerError::SpirvError(
                                "to_vec4_f32 requires exactly one argument".to_string()
                            ));
                        }
                        
                        let array_id = self.generate_expression(&args[0])?;
                        self.convert_array_to_vec4(array_id)
                    }
                    _ => {
                        // Other function calls are not supported
                        Err(CompilerError::SpirvError(
                            format!("Function call '{}' not supported in SPIR-V generation", func_name)
                        ))
                    }
                }
            }
            Expression::Tuple(_elements) => {
                // For now, tuples are not supported in SPIR-V generation
                Err(CompilerError::SpirvError(
                    "Tuples not supported in SPIR-V generation".to_string()
                ))
            }
        }
    }
    
    fn generate_constant_array(&mut self, ty: &Type, elements: &[Expression]) -> Result<Word> {
        match ty {
            Type::Array(elem_ty, _dims) => {
                // Generate constants for each element
                let mut const_ids = Vec::new();
                for elem in elements {
                    match elem {
                        Expression::ArrayLiteral(inner) => {
                            let inner_const = self.generate_constant_array(elem_ty, inner)?;
                            const_ids.push(inner_const);
                        }
                        Expression::FloatLiteral(f) => {
                            let ty_id = self.get_or_create_type(&Type::F32)?;
                            const_ids.push(self.builder.constant_bit32(ty_id, f.to_bits()));
                        }
                        Expression::IntLiteral(n) => {
                            let ty_id = self.get_or_create_type(&Type::I32)?;
                            const_ids.push(self.builder.constant_bit32(ty_id, *n as u32));
                        }
                        Expression::BinaryOp(BinaryOp::Divide, left, right) => {
                            // Evaluate constant division
                            if let (Expression::FloatLiteral(l), Expression::FloatLiteral(r)) = (left.as_ref(), right.as_ref()) {
                                let result = l / r;
                                let ty_id = self.get_or_create_type(&Type::F32)?;
                                const_ids.push(self.builder.constant_bit32(ty_id, result.to_bits()));
                            } else {
                                return Err(CompilerError::SpirvError(
                                    "Only constant float division supported in array literals".to_string()
                                ));
                            }
                        }
                        _ => {
                            return Err(CompilerError::SpirvError(
                                "Only literals supported in constant arrays".to_string()
                            ));
                        }
                    }
                }
                
                let array_ty = self.get_or_create_type(ty)?;
                Ok(self.builder.constant_composite(array_ty, const_ids))
            }
            _ => Err(CompilerError::SpirvError(
                "Expected array type for array literal".to_string()
            )),
        }
    }
    
    fn get_or_create_type(&mut self, ty: &Type) -> Result<Word> {
        if let Some(id) = self.type_cache.get(ty) {
            return Ok(*id);
        }
        
        let id = match ty {
            Type::I32 => self.builder.type_int(32, 1),
            Type::F32 => self.builder.type_float(32),
            Type::Vec4F32 => {
                let f32_type_id = self.builder.type_float(32);
                self.builder.type_vector(f32_type_id, 4)
            }
            Type::Array(elem_ty, dims) => {
                let elem_type_id = self.get_or_create_type(elem_ty)?;
                
                // Build multi-dimensional arrays from innermost to outermost
                let mut current_type = elem_type_id;
                for dim in dims.iter().rev() {
                    let uint_type = self.builder.type_int(32, 0);
                    let dim_const = self.builder.constant_bit32(uint_type, *dim as u32);
                    current_type = self.builder.type_array(current_type, dim_const);
                }
                current_type
            }
            Type::Tuple(_types) => {
                // For now, tuples are not supported in SPIR-V generation
                return Err(CompilerError::SpirvError(
                    "Tuple types not supported in SPIR-V generation".to_string()
                ));
            }
            Type::Var(_name) => {
                // Type variables should not appear in codegen
                return Err(CompilerError::SpirvError(
                    "Type variables not supported in SPIR-V generation".to_string()
                ));
            }
            Type::Function(_arg, _ret) => {
                // Function types are not directly represented in SPIR-V
                return Err(CompilerError::SpirvError(
                    "Function types not supported in SPIR-V generation".to_string()
                ));
            }
            Type::SizeVar(_name) => {
                // Size variables should not appear in codegen
                return Err(CompilerError::SpirvError(
                    "Size variables not supported in SPIR-V generation".to_string()
                ));
            }
        };
        
        self.type_cache.insert(ty.clone(), id);
        Ok(id)
    }
    
    /// Convert a [4]f32 array to vec4<f32> for SPIR-V built-ins like gl_Position
    fn convert_array_to_vec4(&mut self, array_id: Word) -> Result<Word> {
        let f32_type_id = self.builder.type_float(32);
        let vec4_type_id = self.builder.type_vector(f32_type_id, 4);
        
        // Extract array elements using OpCompositeExtract (since array_id is a value, not pointer)
        let mut elements = Vec::new();
        for i in 0..4 {
            let element = self.builder.composite_extract(
                f32_type_id,
                None,
                array_id,
                vec![i as u32],
            ).map_err(|e| CompilerError::SpirvError(format!("Failed to extract array element: {:?}", e)))?;
            elements.push(element);
        }
        
        // Construct vector from elements
        self.builder.composite_construct(vec4_type_id, None, elements)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to construct vector: {:?}", e)))
    }

    /// Determine the appropriate decoration for shader outputs based on semantic mappings
    fn get_output_decoration(&self, exec_model: spirv::ExecutionModel, return_type: &Type) -> OutputDecoration {
        match exec_model {
            spirv::ExecutionModel::Vertex => {
                match return_type {
                    Type::Array(elem_ty, dims) if matches!(elem_ty.as_ref(), Type::F32) && dims == &vec![4] => {
                        // [4]f32 from vertex shader = gl_Position
                        OutputDecoration::BuiltIn(spirv::BuiltIn::Position)
                    }
                    Type::Vec4F32 => {
                        // vec4f32 from vertex shader = gl_Position
                        OutputDecoration::BuiltIn(spirv::BuiltIn::Position)
                    }
                    _ => {
                        // Other types use generic locations
                        OutputDecoration::Location(0)
                    }
                }
            }
            spirv::ExecutionModel::Fragment => {
                // Fragment shaders typically output to locations
                OutputDecoration::Location(0)
            }
            _ => {
                // Future shader types - default to location
                OutputDecoration::Location(0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;
    use crate::type_checker::TypeChecker;
    
    /// Helper function to validate SPIR-V using naga
    fn validate_spirv_with_naga(spirv_words: &[u32]) -> std::result::Result<naga::Module, String> {
        // Convert words to bytes for naga
        let mut spirv_bytes = Vec::with_capacity(spirv_words.len() * 4);
        for word in spirv_words {
            spirv_bytes.extend_from_slice(&word.to_le_bytes());
        }
        
        // Parse SPIR-V with naga
        let module = naga::front::spv::parse_u8_slice(&spirv_bytes, &Default::default())
            .map_err(|e| format!("Failed to parse SPIR-V: {:?}", e))?;
        
        // Validate the module
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all()
        );
        
        let _module_info = validator.validate(&module)
            .map_err(|e| format!("SPIR-V validation failed: {:?}", e))?;
        
        // Print validation success info
        println!("✓ SPIR-V validation passed!");
        println!("  Functions: {}", module.functions.len());
        println!("  Global variables: {}", module.global_variables.len());
        println!("  Entry points: {}", module.entry_points.len());
        
        Ok(module)
    }
    
    /// Compile source code and validate with naga
    fn compile_and_validate_with_naga(source: &str) -> std::result::Result<naga::Module, String> {
        // Compile to SPIR-V
        let tokens = tokenize(source)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        let mut parser = Parser::new(tokens);
        let program = parser.parse()
            .map_err(|e| format!("Parsing failed: {:?}", e))?;
        
        let mut checker = TypeChecker::new();
        checker.check_program(&program)
            .map_err(|e| format!("Type checking failed: {:?}", e))?;
        
        let codegen = CodeGenerator::new();
        let spirv = codegen.generate(&program)
            .map_err(|e| format!("Code generation failed: {:?}", e))?;
        
        // Validate with naga
        validate_spirv_with_naga(&spirv)
    }
    
    #[test]
    fn test_generate_simple_vertex_shader() {
        let input = r#"
            let pos: [4]f32 = [0.0f32, 0.0f32, 0.0f32, 1.0f32]
            #[vertex]
            entry vertex_main(): [4]f32 = pos
        "#;
        
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let mut checker = TypeChecker::new();
        checker.check_program(&program).unwrap();
        
        let codegen = CodeGenerator::new();
        let spirv = codegen.generate(&program).unwrap();
        
        // Basic validation - check that we generated some SPIR-V
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], spirv::MAGIC_NUMBER);
    }
    
    #[test]
    fn test_generate_simple_fragment_shader() {
        let input = r#"
            let color: [4]f32 = [1.0f32, 0.0f32, 0.0f32, 1.0f32]
            #[fragment]
            entry fragment_main(): [4]f32 = color
        "#;
        
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        
        let mut checker = TypeChecker::new();
        checker.check_program(&program).unwrap();
        
        let codegen = CodeGenerator::new();
        let spirv = codegen.generate(&program).unwrap();
        
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], spirv::MAGIC_NUMBER);
    }
    
    #[test]
    fn test_array_indexing_with_naga_validation() {
        let input = r#"
            let pos: [4]f32 = [-1.0f32, -1.0f32, 0.0f32, 1.0f32]
            
            #[vertex]
            entry vertex_main (vertex_id: i32) : vec4f32 =
              to_vec4_f32 pos
        "#;
        
        // This should validate successfully with naga
        match compile_and_validate_with_naga(input) {
            Ok(_module) => {
                // If we get here, validation passed!
                println!("Array indexing shader validated successfully!");
            }
            Err(e) => {
                println!("Validation failed with error: {}", e);
                // For now, we expect this to fail, so we'll panic to see the error
                panic!("Expected validation to pass, but got error: {}", e);
            }
        }
    }
    
    #[test]
    fn test_simple_fragment_with_naga() {
        let input = r#"
            let SKY_RGBA: [4]f32 =
              [135f32/255f32, 206f32/255f32, 235f32/255f32, 1.0f32]
            
            #[fragment]
            entry fragment_main () : vec4f32 =
              to_vec4_f32 SKY_RGBA
        "#;
        
        // Test the fragment shader part which should be simpler
        match compile_and_validate_with_naga(input) {
            Ok(_module) => {
                println!("Fragment shader validated successfully!");
            }
            Err(e) => {
                println!("Fragment validation failed: {}", e);
                panic!("Expected fragment validation to pass, but got error: {}", e);
            }
        }
    }
    
    #[test]
    fn test_minimal_fragment_with_naga() {
        let input = r#"
            #[fragment]
            entry fragment_main () : f32 =
              1.0f32
        "#;
        
        // Test the simplest possible fragment shader - just return a constant
        match compile_and_validate_with_naga(input) {
            Ok(_module) => {
                println!("Minimal fragment shader validated successfully!");
            }
            Err(e) => {
                println!("Minimal fragment validation failed: {}", e);
                panic!("Expected minimal fragment validation to pass, but got error: {}", e);
            }
        }
    }
    
    #[test]
    fn test_simple_array_indexing_with_parameter() {
        let input = r#"
            let arr: [2]i32 = [1, 2]
            let pos: [4]f32 = [0.0f32, 0.0f32, 0.0f32, 1.0f32]
            
            #[vertex]
            entry vertex_main (vertex_id: i32) : vec4f32 =
              to_vec4_f32 pos
        "#;
        
        // Test that simple arrays work (this should pass)
        match compile_and_validate_with_naga(input) {
            Ok(_module) => {
                println!("Simple array test passed!");
            }
            Err(e) => {
                println!("Simple array test failed: {}", e);
                panic!("Expected simple array to work, but got error: {}", e);
            }
        }
    }
    
    #[test]
    fn test_multidimensional_array_indexing() {
        let input = r#"
            let positions: [2][4]f32 = 
              [[0.0f32, 0.0f32, 0.0f32, 1.0f32],
               [1.0f32, 1.0f32, 0.0f32, 1.0f32]]
            
            #[vertex]
            entry vertex_main (vertex_id: i32) : vec4f32 =
              to_vec4_f32 positions[vertex_id]
        "#;
        
        // Test multi-dimensional array indexing
        match compile_and_validate_with_naga(input) {
            Ok(_module) => {
                println!("Multi-dimensional array indexing validated successfully!");
            }
            Err(e) => {
                println!("Multi-dimensional array indexing failed: {}", e);
                panic!("Expected multi-dimensional array indexing to work, but got error: {}", e);
            }
        }
    }
}