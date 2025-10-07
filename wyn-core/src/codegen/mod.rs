mod global;
mod scope;

use self::global::GlobalBuilder;
use self::scope::Environment;
use crate::ast::AttrExt;
use crate::ast::TypeName;
use crate::ast::*;
use crate::builtin_registry::{BuiltinImpl, BuiltinRegistry, CustomImpl, SpirvOp};
use crate::error::{CompilerError, Result};
use log::debug;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Module as SpirvModule};
use rspirv::spirv::{self, AddressingModel, Capability, ExecutionModel, MemoryModel, StorageClass};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Pipeline {
    VertexFragment {
        vertex_shader: Decl,
        fragment_shader: Decl,
    },
}

impl Pipeline {
    pub fn from_program(program: &Program) -> Result<Option<Pipeline>> {
        let mut vertex_shader = None;
        let mut fragment_shader = None;

        for decl in &program.declarations {
            if let Declaration::Decl(decl_node) = decl {
                if decl_node.attributes.has(Attribute::is_vertex) {
                    vertex_shader = Some(decl_node.clone());
                }
                if decl_node.attributes.has(Attribute::is_fragment) {
                    fragment_shader = Some(decl_node.clone());
                }
            }
        }

        match (vertex_shader, fragment_shader) {
            (Some(v), Some(f)) => Ok(Some(Pipeline::VertexFragment {
                vertex_shader: v,
                fragment_shader: f,
            })),
            _ => Ok(None), // No complete pipeline found
        }
    }

    pub fn get_vertex_location_outputs(&self) -> Vec<(u32, Type)> {
        match self {
            Pipeline::VertexFragment { vertex_shader, .. } => {
                if let Some(attributed_types) = &vertex_shader.attributed_return_type {
                    attributed_types
                        .iter()
                        .filter_map(|attr_type| {
                            attr_type.attributes.first_location().map(|loc| (loc, attr_type.ty.clone()))
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            }
        }
    }
}

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

/// Key for caching pointer types by storage class and pointee type
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct PtrKey {
    sc: StorageClass,
    ty: spirv::Word,
}

pub struct CodeGenerator {
    builder: Builder,
    module: SpirvModule,
    builtin_registry: BuiltinRegistry,
    glsl_ext_inst_id: Option<spirv::Word>, // ID for GLSL.std.450 extension
    global_builder: GlobalBuilder,
    pipeline: Option<Pipeline>,
    environment: Environment, // Replaces variable_cache and variable_types with proper scoping
    function_cache: HashMap<String, spirv::Word>,
    type_cache: HashMap<Type, TypeInfo>,
    ptr_cache: HashMap<PtrKey, spirv::Word>,
    current_function: Option<spirv::Word>,
    current_block: Option<spirv::Word>,
    function_entry_block_index: Option<usize>,
    entry_points: Vec<(String, spirv::ExecutionModel)>,
    uniform_globals: HashMap<String, Value>,
    builtin_inputs: HashMap<String, Value>,
    pending_variables: Vec<(spirv::Word, spirv::Word)>, // (variable_id, type_id) for OpVariable instructions

    // Type IDs for common types
    void_type: spirv::Word,
    bool_type: spirv::Word,
    i32_type: spirv::Word,
    f32_type: spirv::Word,
}

impl CodeGenerator {
    /// Check if a type is an attributed tuple
    fn is_attributed_tuple(ty: &Type) -> bool {
        matches!(ty, Type::Constructed(TypeName::Str("attributed_tuple"), _))
    }

    /// Validate that all return types for an entry point are properly attributed
    /// Each element must have @location or @builtin attribute
    fn validate_entry_point_return_types(attributed_types: &[AttributedType]) -> Result<()> {
        // Check if any type is missing attributes
        if attributed_types.iter().any(|at| at.attributes.is_empty()) {
            Err(CompilerError::SpirvError(
                "Entry point return values must all be attributed with @location or @builtin".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn create_fragment_inputs(&mut self) -> Result<()> {
        // Get vertex location outputs from the pipeline (if we have one)
        if let Some(pipeline) = &self.pipeline {
            let vertex_outputs = pipeline.get_vertex_location_outputs();

            for (location, var_type) in vertex_outputs {
                let type_info = self.get_or_create_type(&var_type)?;

                // Create fragment input variable at the same location
                let _input_var_id = self.global_builder.create_or_lookup_location(
                    &mut self.builder,
                    location,
                    StorageClass::Input,
                    type_info.id,
                )?;

                debug!(
                    "Created fragment input at location {} with type {:?}",
                    location, var_type
                );
            }
        }

        Ok(())
    }

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
            builtin_registry: BuiltinRegistry::new(),
            glsl_ext_inst_id: None,
            global_builder: GlobalBuilder::new(),
            pipeline: None,
            environment: Environment::new(),
            function_cache: HashMap::new(),
            type_cache: HashMap::new(),
            ptr_cache: HashMap::new(),
            current_function: None,
            current_block: None,
            function_entry_block_index: None,
            entry_points: Vec::new(),
            uniform_globals: HashMap::new(),
            builtin_inputs: HashMap::new(),
            pending_variables: Vec::new(),

            // Will be initialized in setup_common_types
            void_type: 0,
            bool_type: 0,
            i32_type: 0,
            f32_type: 0,
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
    }

    /// Generate a call to a builtin function
    fn generate_builtin_call(&mut self, name: &str, args: &[Value]) -> Result<Value> {
        let implementation = self
            .builtin_registry
            .get(name)
            .ok_or_else(|| CompilerError::SpirvError(format!("Unknown builtin function: {}", name)))?
            .implementation
            .clone();

        match &implementation {
            BuiltinImpl::GlslExt(inst_num) => {
                let glsl_ext_id = self.glsl_ext_inst_id.ok_or_else(|| {
                    CompilerError::SpirvError("GLSL.std.450 extension not initialized".to_string())
                })?;

                if args.is_empty() {
                    return Err(CompilerError::SpirvError(format!(
                        "{} builtin requires at least one argument",
                        name
                    )));
                }

                // Result type is the same as the first argument for most GLSL functions
                let result_type = args[0].type_id;
                let arg_operands: Vec<rspirv::dr::Operand> =
                    args.iter().map(|v| rspirv::dr::Operand::IdRef(v.id)).collect();

                let result_id =
                    self.builder.ext_inst(result_type, None, glsl_ext_id, *inst_num, arg_operands)?;

                Ok(Value {
                    id: result_id,
                    type_id: result_type,
                })
            }
            BuiltinImpl::SpirvOp(op) => self.generate_spirv_op(op, args, name),
            BuiltinImpl::Custom(custom_impl) => self.generate_custom_builtin(custom_impl, args, name),
        }
    }

    /// Generate code for custom builtin implementations
    fn generate_custom_builtin(
        &mut self,
        custom_impl: &CustomImpl,
        args: &[Value],
        name: &str,
    ) -> Result<Value> {
        match custom_impl {
            CustomImpl::Placeholder => {
                // Vector constructors (vec2, vec3, vec4, ivec2, ivec3, ivec4)
                // Use OpCompositeConstruct to build the vector from scalar components

                // Determine result type from builtin name
                let result_type_ast = match name {
                    "vec2" => Type::Constructed(TypeName::Str("vec2"), vec![]),
                    "vec3" => Type::Constructed(TypeName::Str("vec3"), vec![]),
                    "vec4" => Type::Constructed(TypeName::Str("vec4"), vec![]),
                    "ivec2" => Type::Constructed(TypeName::Str("ivec2"), vec![]),
                    "ivec3" => Type::Constructed(TypeName::Str("ivec3"), vec![]),
                    "ivec4" => Type::Constructed(TypeName::Str("ivec4"), vec![]),
                    _ => {
                        return Err(CompilerError::SpirvError(format!(
                            "Unknown custom builtin: {}",
                            name
                        )));
                    }
                };

                let result_type_info = self.get_or_create_type(&result_type_ast)?;

                // Build composite from constituents
                let constituent_ids: Vec<spirv::Word> = args.iter().map(|v| v.id).collect();
                let result_id =
                    self.builder.composite_construct(result_type_info.id, None, constituent_ids)?;

                Ok(Value {
                    id: result_id,
                    type_id: result_type_info.id,
                })
            }
        }
    }

    /// Generate code for a SPIR-V core operation
    fn generate_spirv_op(&mut self, op: &SpirvOp, args: &[Value], name: &str) -> Result<Value> {
        match op {
            SpirvOp::Dot => {
                if args.len() != 2 {
                    return Err(CompilerError::SpirvError(
                        "dot requires exactly 2 vector arguments".to_string(),
                    ));
                }
                // Result type is the component type (f32)
                let result_type = self.f32_type;
                let result_id = self.builder.dot(result_type, None, args[0].id, args[1].id)?;
                Ok(Value {
                    id: result_id,
                    type_id: result_type,
                })
            }
            SpirvOp::FAdd => self.generate_binary_op(args, |b, rt, a, c| b.f_add(rt, None, a, c)),
            SpirvOp::FSub => self.generate_binary_op(args, |b, rt, a, c| b.f_sub(rt, None, a, c)),
            SpirvOp::FMul => self.generate_binary_op(args, |b, rt, a, c| b.f_mul(rt, None, a, c)),
            SpirvOp::FDiv => self.generate_binary_op(args, |b, rt, a, c| b.f_div(rt, None, a, c)),
            SpirvOp::IAdd => self.generate_binary_op(args, |b, rt, a, c| b.i_add(rt, None, a, c)),
            SpirvOp::ISub => self.generate_binary_op(args, |b, rt, a, c| b.i_sub(rt, None, a, c)),
            SpirvOp::IMul => self.generate_binary_op(args, |b, rt, a, c| b.i_mul(rt, None, a, c)),
            _ => Err(CompilerError::SpirvError(format!(
                "SPIR-V operation {:?} not yet implemented for builtin: {}",
                op, name
            ))),
        }
    }

    /// Helper for generating binary operations
    fn generate_binary_op<F>(&mut self, args: &[Value], op: F) -> Result<Value>
    where
        F: FnOnce(
            &mut Builder,
            spirv::Word,
            spirv::Word,
            spirv::Word,
        ) -> std::result::Result<spirv::Word, rspirv::dr::Error>,
    {
        if args.len() != 2 {
            return Err(CompilerError::SpirvError(
                "Binary operation requires exactly 2 arguments".to_string(),
            ));
        }
        let result_type = args[0].type_id;
        let result_id = op(&mut self.builder, result_type, args[0].id, args[1].id)?;
        Ok(Value {
            id: result_id,
            type_id: result_type,
        })
    }

    /// Get or create a pointer type for the given storage class and pointee type
    fn ptr_of(&mut self, sc: StorageClass, ty: spirv::Word) -> spirv::Word {
        let key = PtrKey { sc, ty };
        *self.ptr_cache.entry(key).or_insert_with(|| self.builder.type_pointer(None, sc, ty))
    }

    /// Create a composite (struct, array, or vector) using the appropriate instruction
    /// based on context (runtime composite_construct vs compile-time constant_composite)
    fn make_composite(&mut self, ty: spirv::Word, elems: Vec<spirv::Word>) -> Result<Value> {
        let id = if self.current_block.is_some() {
            self.builder.composite_construct(ty, None, elems)?
        } else {
            self.builder.constant_composite(ty, elems)
        };
        Ok(Value { id, type_id: ty })
    }

    /// Check if a function name is a composite constructor that can be used in constant expressions
    fn is_const_composite_constructor(name: &str) -> bool {
        name.starts_with("vec") ||   // vec2, vec3, vec4
        name.starts_with("mat") ||   // mat2, mat3, mat4
        name == "array"
    }

    /// Evaluate a composite constructor (vec, mat, array) as a constant
    fn evaluate_const_composite_constructor(
        &mut self,
        func_name: &str,
        args: &[Expression],
    ) -> Result<Value> {
        if !Self::is_const_composite_constructor(func_name) {
            return Err(CompilerError::SpirvError(format!(
                "Unsupported constant expression: function call to {}",
                func_name
            )));
        }

        // Evaluate all arguments as constants
        let mut elem_vals = Vec::new();
        for arg in args {
            let elem_val = self.evaluate_constant_expression(arg)?;
            elem_vals.push(elem_val);
        }

        // Determine the composite type based on constructor name
        if func_name.starts_with("vec") {
            let component_count = func_name[3..].parse::<usize>().map_err(|_| {
                CompilerError::SpirvError(format!("Invalid vector constructor: {}", func_name))
            })?;

            if elem_vals.len() != component_count {
                return Err(CompilerError::SpirvError(format!(
                    "Vector constructor {} expects {} arguments, got {}",
                    func_name,
                    component_count,
                    elem_vals.len()
                )));
            }

            let vec_type = self.builder.type_vector(self.f32_type, component_count as u32);
            let elem_ids: Vec<spirv::Word> = elem_vals.iter().map(|v| v.id).collect();
            let const_id = self.builder.constant_composite(vec_type, elem_ids);

            Ok(Value {
                id: const_id,
                type_id: vec_type,
            })
        } else if func_name.starts_with("mat") {
            // Matrix constructors (mat2, mat3, mat4)
            let dim = func_name[3..].parse::<usize>().map_err(|_| {
                CompilerError::SpirvError(format!("Invalid matrix constructor: {}", func_name))
            })?;

            // Matrices are column vectors
            if elem_vals.len() != dim {
                return Err(CompilerError::SpirvError(format!(
                    "Matrix constructor {} expects {} column vectors, got {}",
                    func_name,
                    dim,
                    elem_vals.len()
                )));
            }

            // Create column vector type
            let col_type = self.builder.type_vector(self.f32_type, dim as u32);
            // Create matrix type
            let mat_type = self.builder.type_matrix(col_type, dim as u32);
            let elem_ids: Vec<spirv::Word> = elem_vals.iter().map(|v| v.id).collect();
            let const_id = self.builder.constant_composite(mat_type, elem_ids);

            Ok(Value {
                id: const_id,
                type_id: mat_type,
            })
        } else {
            // array constructor
            if elem_vals.is_empty() {
                return Err(CompilerError::SpirvError(
                    "Array constructor requires at least one element".to_string(),
                ));
            }

            let elem_type = elem_vals[0].type_id;
            let array_type = self.builder.type_array(elem_type, elem_vals.len() as u32);
            let elem_ids: Vec<spirv::Word> = elem_vals.iter().map(|v| v.id).collect();
            let const_id = self.builder.constant_composite(array_type, elem_ids);

            Ok(Value {
                id: const_id,
                type_id: array_type,
            })
        }
    }

    /// Evaluate a constant expression at compile time (outside function context)
    /// Returns the SPIR-V constant value
    fn evaluate_constant_expression(&mut self, expr: &Expression) -> Result<Value> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                let const_id = self.builder.constant_bit32(self.i32_type, *n as u32);
                Ok(Value {
                    id: const_id,
                    type_id: self.i32_type,
                })
            }
            ExprKind::FloatLiteral(f) => {
                let const_id = self.builder.constant_bit32(self.f32_type, f.to_bits());
                Ok(Value {
                    id: const_id,
                    type_id: self.f32_type,
                })
            }
            ExprKind::FunctionCall(func_name, args) => {
                // Handle composite constructors (vec, mat, array)
                self.evaluate_const_composite_constructor(func_name, args)
            }
            ExprKind::Application(func, args) => {
                // Check if this is a constructor application
                if let ExprKind::Identifier(func_name) = &func.kind {
                    self.evaluate_const_composite_constructor(func_name, args)
                } else {
                    Err(CompilerError::SpirvError(
                        "Constant expressions only support direct constructor calls".to_string(),
                    ))
                }
            }
            ExprKind::ArrayLiteral(elems) => {
                if elems.is_empty() {
                    return Err(CompilerError::SpirvError(
                        "Empty array literals not supported for constants".to_string(),
                    ));
                }

                // Evaluate all elements as constants
                let mut elem_vals = Vec::new();
                for elem_expr in elems {
                    let elem_val = self.evaluate_constant_expression(elem_expr)?;
                    elem_vals.push(elem_val);
                }

                // All elements must have the same type
                let elem_type = elem_vals[0].type_id;
                let elem_ids: Vec<spirv::Word> = elem_vals.iter().map(|v| v.id).collect();

                // Create array type
                let array_type = self.builder.type_array(elem_type, elem_ids.len() as u32);

                // Use constant_composite for array
                let const_id = self.builder.constant_composite(array_type, elem_ids);
                Ok(Value {
                    id: const_id,
                    type_id: array_type,
                })
            }
            _ => Err(CompilerError::SpirvError(format!(
                "Unsupported constant expression: {:?}",
                expr
            ))),
        }
    }

    /// Generic binary operation handler that dispatches to int or float operations
    fn binop(
        &mut self,
        left: Value,
        right: Value,
        int_op: impl FnOnce(
            &mut Builder,
            spirv::Word,
            Option<spirv::Word>,
            spirv::Word,
            spirv::Word,
        ) -> std::result::Result<spirv::Word, rspirv::dr::Error>,
        float_op: impl FnOnce(
            &mut Builder,
            spirv::Word,
            Option<spirv::Word>,
            spirv::Word,
            spirv::Word,
        ) -> std::result::Result<spirv::Word, rspirv::dr::Error>,
        out_ty_int: spirv::Word,
        out_ty_float: spirv::Word,
        op_name: &str,
    ) -> Result<Value> {
        if left.type_id == self.i32_type && right.type_id == self.i32_type {
            let id = int_op(&mut self.builder, out_ty_int, None, left.id, right.id)?;
            Ok(Value {
                id,
                type_id: out_ty_int,
            })
        } else if left.type_id == self.f32_type && right.type_id == self.f32_type {
            let id = float_op(&mut self.builder, out_ty_float, None, left.id, right.id)?;
            Ok(Value {
                id,
                type_id: out_ty_float,
            })
        } else {
            Err(CompilerError::SpirvError(format!(
                "Type mismatch in {}: operands must be both int or both float",
                op_name
            )))
        }
    }

    /// Resolve where an output value should be stored (builtin vs location)
    fn resolve_output_var(
        &mut self,
        attrs: &[Attribute],
        ty_id: spirv::Word,
        exec_model: ExecutionModel,
    ) -> Result<spirv::Word> {
        if let Some(builtin) = attrs.first_builtin() {
            self.global_builder
                .create_or_lookup_builtin(&mut self.builder, builtin, StorageClass::Output, exec_model)?
                .ok_or_else(|| {
                    CompilerError::SpirvError(format!(
                        "Builtin {:?} invalid for {:?} output",
                        builtin, exec_model
                    ))
                })
        } else {
            let location = attrs.first_location().unwrap_or(0);
            self.global_builder.create_or_lookup_location(
                &mut self.builder,
                location,
                StorageClass::Output,
                ty_id,
            )
        }
    }

    /// Extract a component from a composite (tuple, struct, array, or vector)
    fn extract_component(&mut self, composite: Value, comp_ty: spirv::Word, idx: u32) -> Result<Value> {
        let id = self.builder.composite_extract(comp_ty, None, composite.id, vec![idx])?;
        Ok(Value { id, type_id: comp_ty })
    }

    /// Get element type from a vector type ID
    /// For vectors, returns f32 for vec* types and i32 for ivec* types
    fn element_type_of_vector(&self, _vec_type_id: spirv::Word) -> spirv::Word {
        // For now, assume f32 since all our current vectors are vec*
        // A complete implementation would inspect the SPIR-V type structure
        self.f32_type
    }

    /// Prepare entry point parameters, separating builtins from regular parameters
    fn prepare_params(
        &mut self,
        exec_model: ExecutionModel,
        params: &[DeclParam],
    ) -> Result<Vec<spirv::Word>> {
        let mut fn_param_type_ids = Vec::new();

        for param in params {
            let DeclParam::Typed(p) = param else {
                return Err(CompilerError::SpirvError(
                    "Entry point parameters must have explicit types".to_string(),
                ));
            };

            let ty_info = self.get_or_create_type(&p.ty)?;
            if let Some(builtin) = p.attributes.first_builtin() {
                // Builtin parameter - create global input variable
                let input_var = self
                    .global_builder
                    .create_or_lookup_builtin(&mut self.builder, builtin, StorageClass::Input, exec_model)?
                    .ok_or_else(|| {
                        CompilerError::SpirvError(format!(
                            "Builtin {:?} invalid for {:?} input",
                            builtin, exec_model
                        ))
                    })?;

                let val = Value {
                    id: input_var,
                    type_id: ty_info.id,
                };
                self.environment.define_local(p.name.clone(), val, p.ty.clone());
                self.builtin_inputs.insert(p.name.clone(), val);
            } else {
                // Regular parameter - add to function signature
                fn_param_type_ids.push(ty_info.id);
            }
        }

        Ok(fn_param_type_ids)
    }

    /// Create function parameter IDs (must be called before begin_block)
    fn create_fn_params(&mut self, fn_param_type_ids: &[spirv::Word]) -> Result<Vec<spirv::Word>> {
        let mut param_ids = Vec::new();
        for &type_id in fn_param_type_ids {
            let param_id = self.builder.function_parameter(type_id)?;
            param_ids.push(param_id);
        }
        Ok(param_ids)
    }

    /// Store function parameters in local variables (must be called after begin_block)
    fn store_fn_params(&mut self, params: &[DeclParam], param_ids: &[spirv::Word]) -> Result<()> {
        let mut param_index = 0;
        for param in params {
            let DeclParam::Typed(p) = param else { continue };

            // Handle builtin parameters specially - they reference global builtin inputs
            if p.attributes.first_builtin().is_some() {
                // The builtin global was already created in prepare_params
                // Add the parameter name to environment pointing to the builtin input
                if let Some(builtin_value) = self.builtin_inputs.get(&p.name) {
                    self.environment.define_local(p.name.clone(), *builtin_value, p.ty.clone());
                }
                continue;
            }

            // Regular parameters: create local variable and store parameter value
            let ty_info = self.get_or_create_type(&p.ty)?;
            let param_id = param_ids[param_index];
            param_index += 1;

            // Create local variable and store parameter
            let ptr_type = self.ptr_of(StorageClass::Function, ty_info.id);
            let local_var = self.builder.variable(ptr_type, None, StorageClass::Function, None);
            self.builder.store(local_var, param_id, None, vec![])?;

            let value_ptr = Value {
                id: local_var,
                type_id: ptr_type,
            };
            self.environment.define_local(p.name.clone(), value_ptr, p.ty.clone());
        }

        Ok(())
    }

    pub fn generate(mut self, program: &Program) -> Result<Vec<u32>> {
        // Try to create a pipeline from the program
        self.pipeline = Pipeline::from_program(program)?;

        // Import GLSL.std.450 extension instruction set
        self.glsl_ext_inst_id = Some(self.builder.ext_inst_import("GLSL.std.450"));

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

        // Build the module and return SPIR-V words
        self.emit_spirv()
    }

    fn generate_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(decl_node) => {
                // Check if this is an entry point (has Vertex or Fragment attribute)
                let is_entry_point = decl_node.attributes.has(Attribute::is_vertex)
                    || decl_node.attributes.has(Attribute::is_fragment);

                if is_entry_point {
                    self.generate_entry_point(decl_node)
                } else if decl_node.params.is_empty() {
                    // Regular variable declaration: let/def name: type = value or let/def name = value
                    let ty = decl_node.ty.as_ref().ok_or_else(|| {
                        CompilerError::SpirvError(format!(
                            "{} declaration must have explicit type",
                            decl_node.keyword
                        ))
                    })?;
                    self.generate_declaration_helper(&decl_node.name, ty, &decl_node.body)
                } else {
                    // Function declaration: let/def name param1 param2 = body (skip for now)
                    Ok(())
                }
            }
            Declaration::Uniform(uniform_decl) => self.generate_uniform_declaration(uniform_decl),
            Declaration::Val(_val_decl) => {
                // Type signatures only
                Ok(())
            }
        }
    }

    fn generate_uniform_declaration(&mut self, decl: &UniformDecl) -> Result<()> {
        let ty = &decl.ty;

        // Create a global variable for the uniform
        let type_info = self.get_or_create_type(ty)?;
        let ptr_type = self.ptr_of(StorageClass::Uniform, type_info.id);
        let global_id = self.builder.variable(ptr_type, None, StorageClass::Uniform, None);

        let value = Value {
            id: global_id,
            type_id: ptr_type,
        };

        // Store in environment so it can be referenced by other code
        self.environment.define_local(decl.name.clone(), value, ty.clone());

        // Store for SPIR-V decoration generation
        self.uniform_globals.insert(decl.name.clone(), value);

        debug!("Generated uniform variable '{}' with type {:?}", decl.name, ty);

        Ok(())
    }

    fn generate_declaration_helper(
        &mut self,
        name: &str,
        ty: &Type,
        value_expr: &Expression,
    ) -> Result<()> {
        // Try to evaluate as a constant expression
        match self.evaluate_constant_expression(value_expr) {
            Ok(const_value) => {
                // Successfully evaluated as a constant
                // Store directly in the global scope (no pointer needed for constants)
                self.environment.define_global(name.to_string(), const_value, ty.clone());
                Ok(())
            }
            Err(_) => {
                // Not a constant expression - skip silently
                // (only constant expressions are supported for global declarations)
                Ok(())
            }
        }
    }

    fn generate_entry_point(&mut self, decl: &Decl) -> Result<()> {
        // Determine execution model
        let exec_model = if decl.attributes.has(Attribute::is_vertex) {
            spirv::ExecutionModel::Vertex
        } else if decl.attributes.has(Attribute::is_fragment) {
            spirv::ExecutionModel::Fragment
        } else {
            return Err(CompilerError::SpirvError(
                "Entry point must have either #[vertex] or #[fragment] attribute".to_string(),
            ));
        };

        self.entry_points.push((decl.name.clone(), exec_model));

        // Validate that return type is properly attributed
        // Entry points must have attributed return types for outputs
        if let Some(attributed_returns) = &decl.attributed_return_type {
            Self::validate_entry_point_return_types(attributed_returns)?;
        } else {
            return Err(CompilerError::SpirvError(
                "Entry point must have attributed return type (e.g., (vec4 @location(0)))".to_string(),
            ));
        }

        // Handle shader-specific variable creation
        match exec_model {
            ExecutionModel::Fragment => {
                // Fragment shaders need inputs that match vertex outputs (if we have a pipeline)
                self.create_fragment_inputs()?;

                // Fragment shaders also need their own outputs
                if let Some(attributed_return_type) = &decl.attributed_return_type {
                    self.create_outputs(attributed_return_type, exec_model)?;
                }
            }
            _ => {
                // Vertex and other shaders just create outputs
                if let Some(attributed_return_type) = &decl.attributed_return_type {
                    self.create_outputs(attributed_return_type, exec_model)?;
                }
            }
        }

        // Prepare parameters (handles both builtins and regular params)
        let param_type_ids = self.prepare_params(exec_model, &decl.params)?;

        // Create function type - entry points always return void and use output variables
        let actual_return_type = self.void_type;
        let fn_type = self.builder.type_function(actual_return_type, param_type_ids.clone());

        // Create the function
        let function_id =
            self.builder.begin_function(actual_return_type, None, spirv::FunctionControl::NONE, fn_type)?;

        self.function_cache.insert(decl.name.clone(), function_id);
        self.current_function = Some(function_id);

        // Clear local scopes for the new function
        self.environment.clear_locals();

        // Create function parameters BEFORE begin_block
        let param_ids = self.create_fn_params(&param_type_ids)?;

        // Create entry block
        let entry_label = self.builder.begin_block(None)?;
        self.current_block = Some(entry_label);

        // Store parameters in local variables AFTER begin_block
        self.store_fn_params(&decl.params, &param_ids)?;

        // Save the entry block index for later
        self.function_entry_block_index = self.builder.selected_block();

        // Clear pending variables at the start of each function
        self.pending_variables.clear();

        // Generate function body (this will queue variables)
        let result = self.generate_expression(&decl.body)?;

        // Now emit all pending variables at the beginning of the entry block
        self.emit_pending_variables_at_entry_block()?;

        // Entry point coda: write return values to builtin/location outputs
        if decl.attributed_return_type.is_some() {
            self.generate_entry_point_coda(decl, &result, exec_model)?;
            // Entry points return void
            self.builder.ret()?;
        } else {
            // Regular function - return the result
            self.builder.ret_value(result.id)?;
        }

        self.builder.end_function()?;
        self.current_function = None;
        self.current_block = None;
        self.function_entry_block_index = None;

        Ok(())
    }

    /// Emit all pending OpVariable instructions at the beginning of the entry block
    fn emit_pending_variables_at_entry_block(&mut self) -> Result<()> {
        use rspirv::dr::{InsertPoint, Instruction, Operand};

        // Save current block position
        let current_block_index = self.builder.selected_block();

        // Get entry block index
        let entry_block_index = self
            .function_entry_block_index
            .ok_or_else(|| CompilerError::SpirvError("No entry block index saved".to_string()))?;

        // Switch to entry block
        self.builder.select_block(Some(entry_block_index))?;

        // Insert all pending variables at the beginning of the entry block
        // We need to reverse the order since we're inserting at the beginning
        for (var_id, type_id) in self.pending_variables.drain(..).rev() {
            // Create the OpVariable instruction manually
            let inst = Instruction::new(
                spirv::Op::Variable,
                Some(type_id),
                Some(var_id),
                vec![Operand::StorageClass(StorageClass::Function)],
            );

            // Insert at the beginning of the block (right after the label)
            self.builder.insert_into_block(InsertPoint::Begin, inst)?;
        }

        // Restore original block position
        self.builder.select_block(current_block_index)?;

        Ok(())
    }

    /// Generate the entry point coda that writes return values to builtin/location outputs
    fn generate_entry_point_coda(
        &mut self,
        decl: &Decl,
        result: &Value,
        exec_model: ExecutionModel,
    ) -> Result<()> {
        if let Some(attributed_return_type) = &decl.attributed_return_type {
            if attributed_return_type.len() == 1 {
                // Single attributed return value
                let attributed_type = &attributed_return_type[0];
                self.store_attributed_value(result, attributed_type, exec_model)?;
            } else {
                // Multiple attributed return values - extract from tuple
                for (index, attributed_type) in attributed_return_type.iter().enumerate() {
                    let component_type_info = self.get_or_create_type(&attributed_type.ty)?;
                    let component_value =
                        self.extract_component(*result, component_type_info.id, index as u32)?;
                    self.store_attributed_value(&component_value, attributed_type, exec_model)?;
                }
            }
        }
        Ok(())
    }

    /// Store a single attributed value to the appropriate builtin or location output
    fn store_attributed_value(
        &mut self,
        value: &Value,
        attributed_type: &AttributedType,
        exec_model: ExecutionModel,
    ) -> Result<()> {
        let output_var = self.resolve_output_var(&attributed_type.attributes, value.type_id, exec_model)?;
        self.builder.store(output_var, value.id, None, vec![])?;
        Ok(())
    }

    /// Flatten nested Application expressions to get the base function and all arguments
    /// For example: Application(Application(f, [x]), [y]) becomes (f, [x, y])
    fn flatten_application<'a>(
        &self,
        func_expr: &'a Expression,
        args: &'a [Expression],
    ) -> (&'a Expression, Vec<&'a Expression>) {
        let mut all_args = Vec::new();
        let mut current_func = func_expr;

        // Collect arguments from nested Applications
        loop {
            if let ExprKind::Application(inner_func, inner_args) = &current_func.kind {
                // Prepend inner arguments (they come first in the application chain)
                for arg in inner_args.iter().rev() {
                    all_args.insert(0, arg);
                }
                current_func = inner_func.as_ref();
            } else {
                break;
            }
        }

        // Add the final arguments
        all_args.extend(args.iter());

        (current_func, all_args)
    }

    fn generate_expression(&mut self, expr: &Expression) -> Result<Value> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                let const_id = self.builder.constant_bit32(self.i32_type, *n as u32);
                Ok(Value {
                    id: const_id,
                    type_id: self.i32_type,
                })
            }
            ExprKind::FloatLiteral(f) => {
                let const_id = self.builder.constant_bit32(self.f32_type, f.to_bits());
                Ok(Value {
                    id: const_id,
                    type_id: self.f32_type,
                })
            }
            ExprKind::Identifier(name) => {
                if let Some(binding) = self.environment.lookup(name) {
                    let value = binding.value;

                    // Check if this is a builtin input variable
                    if let Some(builtin_value) = self.builtin_inputs.get(name) {
                        // For builtin inputs, we need to load from the global variable
                        let loaded_id = self.builder.load(
                            builtin_value.type_id,
                            None,
                            builtin_value.id,
                            None,
                            vec![],
                        )?;
                        Ok(Value {
                            id: loaded_id,
                            type_id: builtin_value.type_id,
                        })
                    } else if self.current_block.is_none() {
                        // We're at module level - this must be a constant, return it directly
                        Ok(value)
                    } else {
                        // We're inside a function - check if this is a constant or a variable
                        // Global constants have their Value stored directly, local variables are pointers

                        // Check if this value came from the global scope by seeing if it's in the local scopes
                        // For now, we'll use a heuristic: if the type_id matches what we expect for the type,
                        // it's a constant; otherwise it's a pointer that needs loading

                        let var_type = binding
                            .ty
                            .as_ref()
                            .ok_or_else(|| {
                                CompilerError::SpirvError(format!("No type info for variable: {}", name))
                            })?
                            .clone();

                        let type_info = self.get_or_create_type(&var_type)?;

                        // If type_id matches the expected type, it's a constant value
                        if value.type_id == type_info.id {
                            // It's a constant - return it directly
                            Ok(value)
                        } else {
                            // It's a pointer - load from it
                            let loaded_id =
                                self.builder.load(type_info.id, None, value.id, None, vec![])?;
                            Ok(Value {
                                id: loaded_id,
                                type_id: type_info.id,
                            })
                        }
                    }
                } else {
                    Err(CompilerError::UndefinedVariable(name.clone()))
                }
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_val = self.generate_expression(left)?;
                let right_val = self.generate_expression(right)?;

                // Use unified operator string instead of enum variants
                match op.op.as_str() {
                    "+" => self.binop(
                        left_val,
                        right_val,
                        Builder::i_add,
                        Builder::f_add,
                        self.i32_type,
                        self.f32_type,
                        "addition",
                    ),
                    "-" => self.binop(
                        left_val,
                        right_val,
                        Builder::i_sub,
                        Builder::f_sub,
                        self.i32_type,
                        self.f32_type,
                        "subtraction",
                    ),
                    "*" => self.binop(
                        left_val,
                        right_val,
                        Builder::i_mul,
                        Builder::f_mul,
                        self.i32_type,
                        self.f32_type,
                        "multiplication",
                    ),
                    "/" => self.binop(
                        left_val,
                        right_val,
                        Builder::s_div,
                        Builder::f_div,
                        self.i32_type,
                        self.f32_type,
                        "division",
                    ),
                    "==" => self.binop(
                        left_val,
                        right_val,
                        Builder::i_equal,
                        Builder::f_ord_equal,
                        self.bool_type,
                        self.bool_type,
                        "equality",
                    ),
                    "!=" => self.binop(
                        left_val,
                        right_val,
                        Builder::i_not_equal,
                        Builder::f_ord_not_equal,
                        self.bool_type,
                        self.bool_type,
                        "inequality",
                    ),
                    "<" => self.binop(
                        left_val,
                        right_val,
                        Builder::s_less_than,
                        Builder::f_ord_less_than,
                        self.bool_type,
                        self.bool_type,
                        "less than",
                    ),
                    ">" => self.binop(
                        left_val,
                        right_val,
                        Builder::s_greater_than,
                        Builder::f_ord_greater_than,
                        self.bool_type,
                        self.bool_type,
                        "greater than",
                    ),
                    "<=" => self.binop(
                        left_val,
                        right_val,
                        Builder::s_less_than_equal,
                        Builder::f_ord_less_than_equal,
                        self.bool_type,
                        self.bool_type,
                        "less than or equal",
                    ),
                    ">=" => self.binop(
                        left_val,
                        right_val,
                        Builder::s_greater_than_equal,
                        Builder::f_ord_greater_than_equal,
                        self.bool_type,
                        self.bool_type,
                        "greater than or equal",
                    ),
                    _ => Err(CompilerError::SpirvError(format!(
                        "Unknown binary operator: {}",
                        op.op
                    ))),
                }
            }
            ExprKind::FunctionCall(func_name, args) => {
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
                    if let ExprKind::Identifier(array_name) = &array_expr.kind {
                        let binding = self
                            .environment
                            .lookup(array_name)
                            .ok_or_else(|| CompilerError::UndefinedVariable(array_name.clone()))?;
                        let array_type = binding.ty.as_ref().ok_or_else(|| {
                            CompilerError::SpirvError(format!("No type info for: {}", array_name))
                        })?;

                        match array_type {
                            Type::Constructed(TypeName::Array, args) => {
                                // Extract size from Array(Size(n), elem_type)
                                if let Some(Type::Constructed(TypeName::Size(size), _)) = args.get(0) {
                                    let const_id = self.builder.constant_bit32(self.i32_type, *size as u32);
                                    Ok(Value {
                                        id: const_id,
                                        type_id: self.i32_type,
                                    })
                                } else {
                                    Err(CompilerError::SpirvError("Array type missing size".to_string()))
                                }
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
                                "length can only be applied to arrays".to_string(),
                            )),
                        }
                    } else {
                        // For array literals, count the elements
                        if let ExprKind::ArrayLiteral(elements) = &array_expr.kind {
                            let const_id =
                                self.builder.constant_bit32(self.i32_type, elements.len() as u32);
                            Ok(Value {
                                id: const_id,
                                type_id: self.i32_type,
                            })
                        } else {
                            Err(CompilerError::SpirvError(
                                "length can only be applied to array variables or literals".to_string(),
                            ))
                        }
                    }
                }
                // Check if it's a builtin function
                else if self.builtin_registry.is_builtin(func_name) {
                    // Generate arguments first
                    let mut arg_values = Vec::new();
                    for arg in args {
                        arg_values.push(self.generate_expression(arg)?);
                    }

                    // Call the builtin
                    self.generate_builtin_call(func_name, &arg_values)
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
                        "vec2" => self.generate_vec_ctor(2, self.f32_type, args),
                        "vec3" => self.generate_vec_ctor(3, self.f32_type, args),
                        "vec4" => self.generate_vec_ctor(4, self.f32_type, args),
                        "ivec2" => self.generate_vec_ctor(2, self.i32_type, args),
                        "ivec3" => self.generate_vec_ctor(3, self.i32_type, args),
                        "ivec4" => self.generate_vec_ctor(4, self.i32_type, args),
                        // Matrix constructors (column-major)
                        "mat2" => self.generate_mat_ctor(2, 2, args),
                        "mat3" => self.generate_mat_ctor(3, 3, args),
                        "mat4" => self.generate_mat_ctor(4, 4, args),
                        "mat2x3" => self.generate_mat_ctor(2, 3, args),
                        "mat2x4" => self.generate_mat_ctor(2, 4, args),
                        "mat3x2" => self.generate_mat_ctor(3, 2, args),
                        "mat3x4" => self.generate_mat_ctor(3, 4, args),
                        "mat4x2" => self.generate_mat_ctor(4, 2, args),
                        "mat4x3" => self.generate_mat_ctor(4, 3, args),
                        _ => Err(CompilerError::SpirvError(format!(
                            "Function call '{}' not supported",
                            func_name
                        ))),
                    }
                }
            }
            ExprKind::Tuple(elements) => {
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
                    self.make_composite(struct_type, element_values)
                }
            }
            ExprKind::LetIn(let_in) => {
                // Generate code for the value expression
                let value = self.generate_expression(&let_in.value)?;

                // Create a local variable for the let binding
                let var_type = let_in.ty.clone().unwrap_or_else(|| {
                    // TODO: Type checker should fill in inferred types in the AST
                    // Currently the AST is immutable during type checking, so let-in expressions
                    // without explicit type annotations have ty=None even after type checking.
                    // This causes incorrect codegen (defaults to i32 instead of actual type).
                    // Solution: Either make AST mutable or return annotated AST from type checker.
                    panic!(
                        "BUG: let-in expression '{}' missing type annotation. \
                            Type checker should have filled this in but AST is immutable. \
                            Workaround: add explicit type annotation like 'let {} : type = ...'",
                        let_in.name, let_in.name
                    );
                });

                let type_info = self.get_or_create_type(&var_type)?;
                let ptr_type = self.ptr_of(StorageClass::Function, type_info.id);

                // Create the variable ID but queue the OpVariable instruction
                let local_var = self.builder.id();
                self.pending_variables.push((local_var, ptr_type));

                self.builder.store(local_var, value.id, None, vec![])?;

                // Store in variable cache
                let local_value = Value {
                    id: local_var,
                    type_id: ptr_type,
                };

                // Push a new scope for the let binding
                self.environment.push_scope();
                self.environment.define_local(let_in.name.clone(), local_value, var_type);

                // Generate code for the body expression
                let result = self.generate_expression(&let_in.body)?;

                // Pop the scope to clean up the let binding
                self.environment.pop_scope();

                Ok(result)
            }
            ExprKind::FieldAccess(expr, field) => {
                let record_value = self.generate_expression(expr)?;

                // Map field name to vector component index (for built-in vector types)
                let component_index = match field.as_str() {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    _ => {
                        return Err(CompilerError::SpirvError(format!(
                            "Field access for '{}' not yet implemented for non-vector types",
                            field
                        )));
                    }
                };

                // Use SPIR-V CompositeExtract for vectors
                let elem_ty = self.element_type_of_vector(record_value.type_id);
                self.extract_component(record_value, elem_ty, component_index)
            }
            ExprKind::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    return Err(CompilerError::SpirvError(
                        "Empty array literals not supported".to_string(),
                    ));
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

                // Build the array
                self.make_composite(array_type, element_values)
            }
            ExprKind::If(if_expr) => {
                self.generate_if_then_else(&if_expr.condition, &if_expr.then_branch, &if_expr.else_branch)
            }
            ExprKind::Application(func_expr, args) => {
                // Flatten nested applications (for curried calls like f x y z)
                let (base_func, all_args) = self.flatten_application(func_expr, args);

                // Check if this is a qualified builtin call like f32.sin
                if let ExprKind::FieldAccess(base_expr, method_name) = &base_func.kind {
                    if let ExprKind::Identifier(namespace) = &base_expr.kind {
                        let qualified_name = format!("{}.{}", namespace, method_name);

                        // Check if it's a builtin
                        if self.builtin_registry.is_builtin(&qualified_name) {
                            // Generate arguments
                            let mut arg_values = Vec::new();
                            for arg in &all_args {
                                arg_values.push(self.generate_expression(arg)?);
                            }

                            // Call the builtin
                            return self.generate_builtin_call(&qualified_name, &arg_values);
                        }
                    }
                }

                // Check if it's a simple identifier - treat like FunctionCall
                if let ExprKind::Identifier(func_name) = &base_func.kind {
                    let owned_args: Vec<Expression> = all_args.iter().map(|&arg| arg.clone()).collect();
                    let temp_expr = Expression {
                        h: Header { id: NodeId(0) }, // Temporary node ID for codegen
                        kind: ExprKind::FunctionCall(func_name.clone(), owned_args),
                    };
                    return self.generate_expression(&temp_expr);
                }

                // General higher-order function application not yet implemented
                Err(CompilerError::SpirvError(format!(
                    "General higher-order function application not yet implemented: base_func = {:?}",
                    base_func
                )))
            }
            ExprKind::ArrayIndex(array_expr, index_expr) => {
                let array_name = match &array_expr.kind {
                    ExprKind::Identifier(name) => name,
                    _ => {
                        return Err(CompilerError::SpirvError(
                            "Only variable array indexing supported".to_string(),
                        ));
                    }
                };

                let binding = self
                    .environment
                    .lookup(array_name)
                    .ok_or_else(|| CompilerError::UndefinedVariable(array_name.clone()))?;

                let array_value = binding.value;

                // Get the array type from the binding and clone it before generate_expression
                let array_type = binding
                    .ty
                    .as_ref()
                    .ok_or_else(|| {
                        CompilerError::SpirvError(format!("No type info for array: {}", array_name))
                    })?
                    .clone();

                let index = self.generate_expression(index_expr)?;

                let element_type = match &array_type {
                    Type::Constructed(TypeName::Array, args) => {
                        // Array(Size(n), elem_type) - element type is at index 1
                        args.get(1).cloned().unwrap_or_else(types::i32)
                    }
                    _ => {
                        return Err(CompilerError::SpirvError(
                            "Cannot index non-array type".to_string(),
                        ));
                    }
                };

                let element_type_info = self.get_or_create_type(&element_type)?;

                // Create pointer type for array element access
                // TODO: Derive storage class from base pointer. Currently assumes Function,
                // which is correct for local arrays but wrong for Uniform/Private globals.
                // Should extract SC from array_value.type_id or track it alongside Value.
                let element_ptr_type = self.ptr_of(StorageClass::Function, element_type_info.id);

                // Use AccessChain to get element pointer
                let element_ptr =
                    self.builder.access_chain(element_ptr_type, None, array_value.id, vec![index.id])?;

                // Load the element
                let element_id =
                    self.builder.load(element_type_info.id, None, element_ptr, None, vec![])?;

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

    fn generate_if_then_else(
        &mut self,
        condition: &Expression,
        then_expr: &Expression,
        else_expr: &Expression,
    ) -> Result<Value> {
        // Get current block - must exist when generating expressions
        let _current = self.current_block.ok_or_else(|| {
            CompilerError::SpirvError("No current block available for if-then-else".to_string())
        })?;

        // Generate condition
        let condition_val = self.generate_expression(condition)?;

        // Create blocks for then, else, and merge
        let then_block = self.builder.id();
        let else_block = self.builder.id();
        let merge_block = self.builder.id();

        // Create selection merge and branch
        self.builder.selection_merge(merge_block, spirv::SelectionControl::NONE)?;
        self.builder.branch_conditional(condition_val.id, then_block, else_block, vec![])?;

        // Generate then block
        self.builder.begin_block(Some(then_block))?;
        self.current_block = Some(then_block);
        let then_val = self.generate_expression(then_expr)?;
        let then_end_block = self.current_block.ok_or_else(|| {
            CompilerError::SpirvError("Lost current block during then expression".to_string())
        })?;
        self.builder.branch(merge_block)?;

        // Generate else block
        self.builder.begin_block(Some(else_block))?;
        self.current_block = Some(else_block);
        let else_val = self.generate_expression(else_expr)?;
        let else_end_block = self.current_block.ok_or_else(|| {
            CompilerError::SpirvError("Lost current block during else expression".to_string())
        })?;
        self.builder.branch(merge_block)?;

        // Create merge block with phi
        self.builder.begin_block(Some(merge_block))?;
        self.current_block = Some(merge_block);

        // Create phi instruction - both branches should have same type
        let result_type = then_val.type_id;
        let phi_id = self.builder.phi(
            result_type,
            None,
            vec![(then_val.id, then_end_block), (else_val.id, else_end_block)],
        )?;

        Ok(Value {
            id: phi_id,
            type_id: result_type,
        })
    }

    /// Generic vector constructor for vec* and ivec* types
    fn generate_vec_ctor(
        &mut self,
        lanes: u32,
        elem_ty: spirv::Word,
        args: &[Expression],
    ) -> Result<Value> {
        if args.len() != lanes as usize {
            return Err(CompilerError::SpirvError(format!(
                "Vector constructor expects {} args, got {}",
                lanes,
                args.len()
            )));
        }
        let mut elems = Vec::with_capacity(args.len());
        for a in args {
            elems.push(self.generate_expression(a)?.id);
        }
        let vty = self.builder.type_vector(elem_ty, lanes);
        self.make_composite(vty, elems)
    }

    fn generate_mat_ctor(&mut self, cols: u32, rows: u32, args: &[Expression]) -> Result<Value> {
        // Matrix constructors can take either:
        // 1. cols*rows scalars (column-major order)
        // 2. cols column vectors

        let expected_scalars = (cols * rows) as usize;
        let expected_vectors = cols as usize;

        if args.len() == expected_scalars {
            // Scalar initialization - build column vectors first
            let vec_type = self.builder.type_vector(self.f32_type, rows);
            let mut column_ids = Vec::new();

            for col in 0..cols {
                let mut column_elems = Vec::new();
                for row in 0..rows {
                    let idx = (col * rows + row) as usize;
                    let elem = self.generate_expression(&args[idx])?;
                    column_elems.push(elem.id);
                }
                let col_id = self.make_composite(vec_type, column_elems)?.id;
                column_ids.push(col_id);
            }

            let mat_type = self.builder.type_matrix(vec_type, cols);
            self.make_composite(mat_type, column_ids)
        } else if args.len() == expected_vectors {
            // Column vector initialization
            let mut column_ids = Vec::new();
            for arg in args {
                let col_val = self.generate_expression(arg)?;
                column_ids.push(col_val.id);
            }

            let vec_type = self.builder.type_vector(self.f32_type, rows);
            let mat_type = self.builder.type_matrix(vec_type, cols);
            self.make_composite(mat_type, column_ids)
        } else {
            Err(CompilerError::SpirvError(format!(
                "Matrix{}x{} constructor expects {} scalars or {} column vectors, got {} args",
                cols,
                rows,
                expected_scalars,
                expected_vectors,
                args.len()
            )))
        }
    }

    fn convert_array_to_vec4(&mut self, array_val: Value) -> Result<Value> {
        // For rspirv, we need to extract array elements and build a vector
        let vec4_type = self.builder.type_vector(self.f32_type, 4);

        // Extract elements from the array and build vector
        let mut elements = Vec::new();
        for i in 0..4 {
            let element = self.extract_component(array_val, self.f32_type, i)?;
            elements.push(element.id);
        }

        // Build the vector
        self.make_composite(vec4_type, elements)
    }

    fn get_or_create_type(&mut self, ty: &Type) -> Result<TypeInfo> {
        if let Some(cached) = self.type_cache.get(ty) {
            return Ok(cached.clone());
        }

        let (type_id, size_bytes) = match ty {
            Type::Constructed(name, args) => {
                match name {
                    TypeName::Str("i32") => (self.i32_type, 4),
                    TypeName::Str("f32") => (self.f32_type, 4),

                    // f32 vector types
                    TypeName::Str(s) if s.starts_with("vec") && s.len() == 4 => {
                        let n: u32 = s[3..]
                            .parse()
                            .map_err(|_| CompilerError::SpirvError("Invalid vec arity".to_string()))?;
                        let vec_type = self.builder.type_vector(self.f32_type, n);
                        (vec_type, 4 * n)
                    }

                    // i32 vector types
                    TypeName::Str(s) if s.starts_with("ivec") && s.len() == 5 => {
                        let n: u32 = s[4..]
                            .parse()
                            .map_err(|_| CompilerError::SpirvError("Invalid ivec arity".to_string()))?;
                        let vec_type = self.builder.type_vector(self.i32_type, n);
                        (vec_type, 4 * n)
                    }

                    // Matrix types (f32, column-major)
                    TypeName::Str("mat2") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 2);
                        let mat_type = self.builder.type_matrix(vec_type, 2);
                        (mat_type, 16) // 2 cols * 2 rows * 4 bytes
                    }
                    TypeName::Str("mat3") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 3);
                        let mat_type = self.builder.type_matrix(vec_type, 3);
                        (mat_type, 36) // 3 cols * 3 rows * 4 bytes
                    }
                    TypeName::Str("mat4") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 4);
                        let mat_type = self.builder.type_matrix(vec_type, 4);
                        (mat_type, 64) // 4 cols * 4 rows * 4 bytes
                    }
                    TypeName::Str("mat2x3") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 3);
                        let mat_type = self.builder.type_matrix(vec_type, 2);
                        (mat_type, 24) // 2 cols * 3 rows * 4 bytes
                    }
                    TypeName::Str("mat2x4") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 4);
                        let mat_type = self.builder.type_matrix(vec_type, 2);
                        (mat_type, 32) // 2 cols * 4 rows * 4 bytes
                    }
                    TypeName::Str("mat3x2") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 2);
                        let mat_type = self.builder.type_matrix(vec_type, 3);
                        (mat_type, 24) // 3 cols * 2 rows * 4 bytes
                    }
                    TypeName::Str("mat3x4") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 4);
                        let mat_type = self.builder.type_matrix(vec_type, 3);
                        (mat_type, 48) // 3 cols * 4 rows * 4 bytes
                    }
                    TypeName::Str("mat4x2") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 2);
                        let mat_type = self.builder.type_matrix(vec_type, 4);
                        (mat_type, 32) // 4 cols * 2 rows * 4 bytes
                    }
                    TypeName::Str("mat4x3") => {
                        let vec_type = self.builder.type_vector(self.f32_type, 3);
                        let mat_type = self.builder.type_matrix(vec_type, 4);
                        (mat_type, 48) // 4 cols * 3 rows * 4 bytes
                    }

                    TypeName::Array => {
                        // Array(Size(n), elem_type)
                        let size = if let Some(Type::Constructed(TypeName::Size(n), _)) = args.get(0) {
                            *n
                        } else {
                            return Err(CompilerError::SpirvError("Array type missing size".to_string()));
                        };

                        let elem_ty = args.get(1).ok_or_else(|| {
                            CompilerError::SpirvError("Array type missing element type".to_string())
                        })?;
                        let elem_type_info = self.get_or_create_type(elem_ty)?;

                        // Use the actual size from the type
                        let length_id = self.builder.constant_bit32(self.i32_type, size as u32);
                        let array_type = self.builder.type_array(elem_type_info.id, length_id);
                        (array_type, elem_type_info.size_bytes * (size as u32))
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

        // Get interface variables using GlobalBuilder
        let mut interface_vars = Vec::new();

        // Add uniform variables
        for value in self.uniform_globals.values() {
            interface_vars.push(value.id);
        }

        // Add builtin variables for this execution model
        let builtin_vars = self.global_builder.get_builtin_interface_variables(*exec_model);
        interface_vars.extend(builtin_vars);

        // Add location-based variables for outputs
        let output_location_vars =
            self.global_builder.get_location_interface_variables(StorageClass::Output);
        interface_vars.extend(output_location_vars);

        self.builder.entry_point(*exec_model, func_id, name, interface_vars);

        // Add execution modes for specific shader types
        match exec_model {
            spirv::ExecutionModel::Fragment => {
                // Fragment shaders need coordinate origin mode
                self.builder.execution_mode(func_id, spirv::ExecutionMode::OriginUpperLeft, vec![]);
            }
            _ => {} // Other execution models don't need special modes for now
        }

        debug!(
            "Added SPIR-V entry point metadata for '{}' with execution model {:?}",
            name, exec_model
        );

        Ok(())
    }

    /// Create output variables for attributed return types
    fn create_outputs(
        &mut self,
        attributed_types: &[AttributedType],
        exec_model: ExecutionModel,
    ) -> Result<()> {
        // Simply resolve each output - GlobalBuilder handles idempotency
        for attributed_type in attributed_types {
            let type_info = self.get_or_create_type(&attributed_type.ty)?;
            let _output_var =
                self.resolve_output_var(&attributed_type.attributes, type_info.id, exec_model)?;
        }
        Ok(())
    }

    fn emit_spirv(mut self) -> Result<Vec<u32>> {
        // Build the module (takes ownership of builder)
        self.module = self.builder.module();

        // Convert module to SPIR-V words
        let words = self.module.assemble();

        debug!("Generated {} words of SPIR-V", words.len());

        Ok(words)
    }
}
