//! SPIR-V Lowering
//!
//! This module converts MIR (Mid-level Intermediate Representation) to SPIR-V.

use crate::ast::TypeName;
use crate::builtin_registry::{BuiltinImpl, BuiltinRegistry, SpirvOp};
use crate::error::{CompilerError, Result};
use crate::mir::{self, BlockId, FunctionId, Instruction, Module as MirModule, Register};
use polytype::Type;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, AddressingModel, Capability, ExecutionModel, MemoryModel, StorageClass};
use std::collections::HashMap;

/// Maps MIR registers to SPIR-V value IDs
type RegisterMap = HashMap<u32, spirv::Word>;

/// Maps MIR function IDs to SPIR-V function IDs
type FunctionMap = HashMap<FunctionId, spirv::Word>;

/// Lowers MIR to SPIR-V
pub struct Lowering {
    builder: Builder,
    builtin_registry: BuiltinRegistry,

    // Type caching
    type_cache: HashMap<Type<TypeName>, spirv::Word>,
    ptr_cache: HashMap<(StorageClass, spirv::Word), spirv::Word>,

    // Constant caching
    int_const_cache: HashMap<(i32, spirv::Word), spirv::Word>,
    float_const_cache: HashMap<(u32, spirv::Word), spirv::Word>, // bits as u32
    bool_const_cache: HashMap<(bool, spirv::Word), spirv::Word>,

    // Common types
    void_type: spirv::Word,
    bool_type: spirv::Word,
    i32_type: spirv::Word,
    f32_type: spirv::Word,

    // GLSL extended instruction set
    glsl_ext_inst_id: spirv::Word,

    // Function mapping
    function_map: FunctionMap,

    // Current function context
    current_register_map: RegisterMap,
    current_block_map: HashMap<BlockId, spirv::Word>,
    current_is_entry_point: bool,
    current_output_vars: Vec<spirv::Word>,
    current_input_vars: Vec<(spirv::Word, Register)>, // (var_id, register) to load in first block
    current_function_blocks: Vec<mir::Block>,         // For control flow analysis

    // Entry point interface variables - maps function ID to (inputs, outputs)
    entry_point_interfaces: HashMap<FunctionId, (Vec<spirv::Word>, Vec<spirv::Word>)>,
}

impl Default for Lowering {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowering {
    pub fn new() -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 0);

        // Create a temporary context just for builtin registry initialization
        let mut temp_ctx = polytype::Context::<crate::ast::TypeName>::default();
        let builtin_registry = BuiltinRegistry::new(&mut temp_ctx);

        let mut lowering = Lowering {
            builder,
            builtin_registry,
            type_cache: HashMap::new(),
            ptr_cache: HashMap::new(),
            int_const_cache: HashMap::new(),
            float_const_cache: HashMap::new(),
            bool_const_cache: HashMap::new(),
            void_type: 0,
            bool_type: 0,
            i32_type: 0,
            f32_type: 0,
            glsl_ext_inst_id: 0,
            function_map: HashMap::new(),
            current_register_map: HashMap::new(),
            current_block_map: HashMap::new(),
            current_is_entry_point: false,
            current_output_vars: Vec::new(),
            current_input_vars: Vec::new(),
            current_function_blocks: Vec::new(),
            entry_point_interfaces: HashMap::new(),
        };

        // Initialize common types
        lowering.void_type = lowering.builder.type_void();
        lowering.bool_type = lowering.builder.type_bool();
        lowering.i32_type = lowering.builder.type_int(32, 1);
        lowering.f32_type = lowering.builder.type_float(32);

        // Import GLSL.std.450 extended instruction set
        lowering.glsl_ext_inst_id = lowering.builder.ext_inst_import("GLSL.std.450");

        lowering
    }

    /// Lower a complete MIR module to SPIR-V
    pub fn lower_module(mut self, mir: &MirModule) -> Result<Vec<u32>> {
        // Set up SPIR-V capabilities and addressing model
        self.builder.capability(Capability::Shader);
        self.builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        // Lower all functions
        for function in &mir.functions {
            let is_entry_point = mir.entry_points.contains(&function.id);
            self.lower_function(function, is_entry_point)?;
        }

        // Add entry points
        for &func_id in &mir.entry_points {
            if let Some(&spirv_func_id) = self.function_map.get(&func_id) {
                let func = mir.functions.iter().find(|f| f.id == func_id).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Entry point function {} not found", func_id))
                })?;

                // For now, assume vertex shaders based on function name
                let execution_model = if func.name.contains("vertex") {
                    ExecutionModel::Vertex
                } else if func.name.contains("fragment") {
                    ExecutionModel::Fragment
                } else {
                    ExecutionModel::GLCompute
                };

                // Get interface variables for this entry point
                let interface_vars =
                    if let Some((inputs, outputs)) = self.entry_point_interfaces.get(&func_id) {
                        inputs.iter().chain(outputs.iter()).copied().collect::<Vec<_>>()
                    } else {
                        Vec::new()
                    };

                // Register entry point with interface variables
                self.builder.entry_point(execution_model, spirv_func_id, &func.name, &interface_vars);

                // Add required execution modes
                if execution_model == ExecutionModel::Fragment {
                    self.builder.execution_mode(spirv_func_id, spirv::ExecutionMode::OriginUpperLeft, []);
                }
            }
        }

        // Assemble the module
        Ok(self.builder.module().assemble())
    }

    /// Get or create a SPIR-V type for a MIR type
    fn get_or_create_type(&mut self, ty: &Type<TypeName>) -> Result<spirv::Word> {
        if let Some(&cached) = self.type_cache.get(ty) {
            return Ok(cached);
        }

        let type_id = match ty {
            Type::Constructed(TypeName::Array, args) if args.len() == 2 => {
                // Array[Size(n), element_type]
                if let Type::Constructed(TypeName::Size(n), _) = &args[0] {
                    let elem_type_id = self.get_or_create_type(&args[1])?;
                    let length_id = self.get_or_create_int_const(*n as i32, self.i32_type);
                    self.builder.type_array(elem_type_id, length_id)
                } else {
                    return Err(CompilerError::SpirvError(format!(
                        "Array size must be a Size literal, got: {:?}",
                        args[0]
                    )));
                }
            }
            Type::Constructed(TypeName::Vec, args) if args.len() == 2 => {
                // Vec[Size(n), element_type]
                if let Type::Constructed(TypeName::Size(n), _) = &args[0] {
                    let elem_type_id = self.get_or_create_type(&args[1])?;
                    self.builder.type_vector(elem_type_id, *n as u32)
                } else {
                    return Err(CompilerError::SpirvError(format!(
                        "Vector size must be a Size literal, got: {:?}",
                        args[0]
                    )));
                }
            }
            Type::Constructed(type_name, args) => {
                let name_str = match type_name {
                    TypeName::Str(s) => *s,
                    TypeName::Named(s) => s.as_str(),
                    _ => {
                        return Err(CompilerError::SpirvError(format!(
                            "Unsupported type name: {:?}",
                            type_name
                        )));
                    }
                };
                match name_str {
                    "i32" => self.i32_type,
                    "f32" => self.f32_type,
                    "bool" => self.bool_type,
                    "void" => self.void_type,
                    "vec2" | "vec2f32" => {
                        let f32_id = self.f32_type;
                        self.builder.type_vector(f32_id, 2)
                    }
                    "vec3" | "vec3f32" => {
                        let f32_id = self.f32_type;
                        self.builder.type_vector(f32_id, 3)
                    }
                    "vec4" | "vec4f32" => {
                        let f32_id = self.f32_type;
                        self.builder.type_vector(f32_id, 4)
                    }
                    "vec2i32" => {
                        let i32_id = self.i32_type;
                        self.builder.type_vector(i32_id, 2)
                    }
                    "vec3i32" => {
                        let i32_id = self.i32_type;
                        self.builder.type_vector(i32_id, 3)
                    }
                    "vec4i32" => {
                        let i32_id = self.i32_type;
                        self.builder.type_vector(i32_id, 4)
                    }
                    "tuple" => {
                        // Get component types
                        let mut component_type_ids = Vec::new();
                        for arg in args {
                            component_type_ids.push(self.get_or_create_type(arg)?);
                        }
                        self.builder.type_struct(component_type_ids)
                    }
                    "pointer" if args.len() == 1 => {
                        // Pointer to some type (default to Function storage class)
                        let pointee_id = self.get_or_create_type(&args[0])?;
                        self.get_or_create_ptr_type(StorageClass::Function, pointee_id)
                    }
                    _ => return Err(CompilerError::SpirvError(format!("Unsupported type: {:?}", ty))),
                }
            }
            _ => return Err(CompilerError::SpirvError(format!("Unsupported type: {:?}", ty))),
        };

        self.type_cache.insert(ty.clone(), type_id);
        Ok(type_id)
    }

    /// Get or create a pointer type
    fn get_or_create_ptr_type(
        &mut self,
        storage_class: StorageClass,
        pointee_id: spirv::Word,
    ) -> spirv::Word {
        let key = (storage_class, pointee_id);
        if let Some(&cached) = self.ptr_cache.get(&key) {
            return cached;
        }

        let ptr_id = self.builder.type_pointer(None, storage_class, pointee_id);
        self.ptr_cache.insert(key, ptr_id);
        ptr_id
    }

    /// Get or create an integer constant
    fn get_or_create_int_const(&mut self, value: i32, type_id: spirv::Word) -> spirv::Word {
        let key = (value, type_id);
        if let Some(&cached) = self.int_const_cache.get(&key) {
            return cached;
        }

        let const_id = self.builder.constant_bit32(type_id, value as u32);
        self.int_const_cache.insert(key, const_id);
        const_id
    }

    /// Get or create a float constant
    fn get_or_create_float_const(&mut self, value: f32, type_id: spirv::Word) -> spirv::Word {
        let bits = value.to_bits();
        let key = (bits, type_id);
        if let Some(&cached) = self.float_const_cache.get(&key) {
            return cached;
        }

        let const_id = self.builder.constant_bit32(type_id, value.to_bits());
        self.float_const_cache.insert(key, const_id);
        const_id
    }

    /// Get or create a boolean constant
    fn get_or_create_bool_const(&mut self, value: bool, type_id: spirv::Word) -> spirv::Word {
        let key = (value, type_id);
        if let Some(&cached) = self.bool_const_cache.get(&key) {
            return cached;
        }

        let const_id =
            if value { self.builder.constant_true(type_id) } else { self.builder.constant_false(type_id) };
        self.bool_const_cache.insert(key, const_id);
        const_id
    }

    /// Lower a MIR function to SPIR-V
    fn lower_function(&mut self, func: &mir::Function, is_entry_point: bool) -> Result<()> {
        // For entry points, create void(void) signature and use global variables
        let (return_type_id, param_type_ids) = if is_entry_point {
            (self.void_type, vec![])
        } else {
            let return_type_id = self.get_or_create_type(&func.return_type)?;
            let mut param_type_ids = Vec::new();
            for param in &func.params {
                param_type_ids.push(self.get_or_create_type(&param.ty)?);
            }
            (return_type_id, param_type_ids)
        };

        let func_type_id = self.builder.type_function(return_type_id, param_type_ids.clone());
        let func_id = self.builder.begin_function(
            return_type_id,
            None,
            spirv::FunctionControl::NONE,
            func_type_id,
        )?;

        // Store function mapping
        self.function_map.insert(func.id, func_id);

        // Reset register and block maps for this function
        self.current_register_map.clear();
        self.current_block_map.clear();
        self.current_is_entry_point = is_entry_point;
        self.current_output_vars.clear();
        self.current_input_vars.clear();
        self.current_function_blocks = func.blocks.clone();

        // For entry points, create global input/output variables
        if is_entry_point {
            self.create_entry_point_interface(func)?;
        } else {
            // Create regular parameters
            for (i, param) in func.params.iter().enumerate() {
                let param_id = self.builder.function_parameter(param_type_ids[i])?;
                self.current_register_map.insert(param.id, param_id);
            }
        }

        // Create all blocks first (for forward references)
        for block in &func.blocks {
            let block_id = self.builder.id();
            self.current_block_map.insert(block.id, block_id);
        }

        // Lower all blocks
        let first_block_id = func.entry_block;
        for block in &func.blocks {
            let is_first = block.id == first_block_id;
            self.lower_block(block, is_first)?;
        }

        self.builder.end_function()?;
        Ok(())
    }

    /// Lower a MIR basic block to SPIR-V
    fn lower_block(&mut self, block: &mir::Block, is_first_block: bool) -> Result<()> {
        eprintln!("DEBUG: Lowering block {} with {} instructions", block.id, block.instructions.len());
        let spirv_block_id = *self
            .current_block_map
            .get(&block.id)
            .ok_or_else(|| CompilerError::SpirvError(format!("Block {} not found in map", block.id)))?;

        self.builder.begin_block(Some(spirv_block_id))?;

        // For entry points, load input variables in the first block
        if is_first_block && self.current_is_entry_point && !self.current_input_vars.is_empty() {
            let input_vars = self.current_input_vars.clone();
            for (var_id, param) in input_vars {
                let param_type_id = self.get_or_create_type(&param.ty)?;
                let loaded_value = self.builder.load(param_type_id, None, var_id, None, [])?;
                self.current_register_map.insert(param.id, loaded_value);
            }
            self.current_input_vars.clear();
        }

        for (idx, instruction) in block.instructions.iter().enumerate() {
            eprintln!("DEBUG:   Instruction {}: {:?}", idx, instruction);
            self.lower_instruction(instruction)?;
        }

        Ok(())
    }

    /// Lower a MIR instruction to SPIR-V
    fn lower_instruction(&mut self, inst: &Instruction) -> Result<()> {
        match inst {
            Instruction::ConstInt(dest, value) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let const_id = self.get_or_create_int_const(*value, type_id);
                self.current_register_map.insert(dest.id, const_id);
            }

            Instruction::ConstFloat(dest, value) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let const_id = self.get_or_create_float_const(*value, type_id);
                self.current_register_map.insert(dest.id, const_id);
            }

            Instruction::ConstBool(dest, value) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let const_id = self.get_or_create_bool_const(*value, type_id);
                self.current_register_map.insert(dest.id, const_id);
            }

            Instruction::Add(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&dest.ty) {
                    self.builder.f_add(type_id, None, left_id, right_id)?
                } else {
                    self.builder.i_add(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Sub(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&dest.ty) {
                    self.builder.f_sub(type_id, None, left_id, right_id)?
                } else {
                    self.builder.i_sub(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Mul(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&dest.ty) {
                    self.builder.f_mul(type_id, None, left_id, right_id)?
                } else {
                    self.builder.i_mul(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Div(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&dest.ty) {
                    self.builder.f_div(type_id, None, left_id, right_id)?
                } else {
                    self.builder.s_div(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Eq(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&left.ty) {
                    self.builder.f_ord_equal(type_id, None, left_id, right_id)?
                } else {
                    self.builder.i_equal(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Ne(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&left.ty) {
                    self.builder.f_ord_not_equal(type_id, None, left_id, right_id)?
                } else {
                    self.builder.i_not_equal(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Lt(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&left.ty) {
                    self.builder.f_ord_less_than(type_id, None, left_id, right_id)?
                } else {
                    self.builder.s_less_than(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Le(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&left.ty) {
                    self.builder.f_ord_less_than_equal(type_id, None, left_id, right_id)?
                } else {
                    self.builder.s_less_than_equal(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Gt(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&left.ty) {
                    self.builder.f_ord_greater_than(type_id, None, left_id, right_id)?
                } else {
                    self.builder.s_greater_than(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Ge(dest, left, right) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let left_id = self.get_register(left)?;
                let right_id = self.get_register(right)?;

                let result_id = if self.is_float_type(&left.ty) {
                    self.builder.f_ord_greater_than_equal(type_id, None, left_id, right_id)?
                } else {
                    self.builder.s_greater_than_equal(type_id, None, left_id, right_id)?
                };

                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::Branch(target_block) => {
                let target_id = *self.current_block_map.get(target_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Block {} not found in map", target_block))
                })?;
                self.builder.branch(target_id)?;
            }

            Instruction::BranchCond(cond, true_block, false_block, merge_block) => {
                let cond_id = self.get_register(cond)?;
                let true_id = *self.current_block_map.get(true_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Block {} not found in map", true_block))
                })?;
                let false_id = *self.current_block_map.get(false_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Block {} not found in map", false_block))
                })?;
                let merge_spirv_id = *self.current_block_map.get(merge_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Merge block {} not found in map", merge_block))
                })?;

                // Add OpSelectionMerge before the conditional branch
                self.builder.selection_merge(merge_spirv_id, spirv::SelectionControl::NONE)?;
                self.builder.branch_conditional(cond_id, true_id, false_id, [])?;
            }

            Instruction::BranchLoop(cond, body_block, exit_block, merge_block, continue_block) => {
                let cond_id = self.get_register(cond)?;
                let body_id = *self.current_block_map.get(body_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Body block {} not found in map", body_block))
                })?;
                let exit_id = *self.current_block_map.get(exit_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Exit block {} not found in map", exit_block))
                })?;
                let merge_spirv_id = *self.current_block_map.get(merge_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Merge block {} not found in map", merge_block))
                })?;
                let continue_id = *self.current_block_map.get(continue_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Continue block {} not found in map", continue_block))
                })?;

                // Add OpLoopMerge before the conditional branch
                self.builder.loop_merge(merge_spirv_id, continue_id, spirv::LoopControl::NONE, [])?;
                self.builder.branch_conditional(cond_id, body_id, exit_id, [])?;
            }

            Instruction::Loop(loop_info) => {
                eprintln!("DEBUG: Loop metadata - pre-allocating SPIR-V IDs for forward references");
                // Pre-allocate SPIR-V IDs for registers with forward references (Phi operands)
                let phi_spirv_id = self.builder.id();
                let result_spirv_id = self.builder.id();
                let body_result_spirv_id = self.builder.id();

                self.current_register_map.insert(loop_info.phi_reg.id, phi_spirv_id);
                self.current_register_map.insert(loop_info.result_reg.id, result_spirv_id);
                self.current_register_map.insert(loop_info.body_result_reg.id, body_result_spirv_id);

                // Store loop info for later use when we see special loop instructions
                // For now, just branch to header
                let header_id = *self.current_block_map.get(&loop_info.header_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Header block {} not found", loop_info.header_block))
                })?;
                self.builder.branch(header_id)?;

                // The header, body, and merge blocks will be lowered separately
                // when we process them in the MIR block list
            }

            Instruction::Phi(dest, incoming) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let mut operands = Vec::new();
                for (value_reg, block_id) in incoming {
                    let value_id = self.get_register(value_reg)?;
                    let block_spirv_id = *self.current_block_map.get(block_id).ok_or_else(|| {
                        CompilerError::SpirvError(format!("Block {} not found in map", block_id))
                    })?;
                    operands.push((value_id, block_spirv_id));
                }
                let result_id = self.builder.phi(type_id, None, operands)?;
                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::CallBuiltin(dest, name, args) => {
                // Handle builtin functions
                match name.as_str() {
                    // Vector constructors
                    "vec2" | "vec3" | "vec4" | "vec2i32" | "vec3i32" | "vec4i32" => {
                        let type_id = self.get_or_create_type(&dest.ty)?;
                        let mut arg_ids = Vec::new();
                        for arg in args {
                            arg_ids.push(self.get_register(arg)?);
                        }
                        let result_id = self.builder.composite_construct(type_id, None, arg_ids)?;
                        self.current_register_map.insert(dest.id, result_id);
                    }
                    // Array builtins
                    "replicate" => {
                        // replicate length value creates an array filled with value
                        let type_id = self.get_or_create_type(&dest.ty)?;
                        let value_id = self.get_register(&args[1])?;

                        // Get array length from type
                        let length = if let Type::Constructed(TypeName::Array, type_args) = &dest.ty {
                            if let Type::Constructed(TypeName::Size(n), _) = &type_args[0] {
                                *n
                            } else {
                                return Err(CompilerError::SpirvError(
                                    "replicate: array size must be known at compile time".to_string(),
                                ));
                            }
                        } else {
                            return Err(CompilerError::SpirvError(
                                "replicate: result must be an array type".to_string(),
                            ));
                        };

                        // Create array by replicating value
                        let element_ids = vec![value_id; length];
                        let result_id = self.builder.composite_construct(type_id, None, element_ids)?;
                        self.current_register_map.insert(dest.id, result_id);
                    }
                    "__array_update" => {
                        // __array_update array index new_value creates a new array with element at index replaced
                        let type_id = self.get_or_create_type(&dest.ty)?;
                        let array_id = self.get_register(&args[0])?;
                        let index_id = self.get_register(&args[1])?;
                        let new_value_id = self.get_register(&args[2])?;

                        // Use OpCompositeInsert to create new composite with updated element
                        let result_id = self.builder.composite_insert(
                            type_id,
                            None,
                            new_value_id,
                            array_id,
                            [index_id],
                        )?;
                        self.current_register_map.insert(dest.id, result_id);
                    }
                    // Look up in builtin registry
                    _ => {
                        if let Some(builtin_desc) = self.builtin_registry.get(name) {
                            let impl_clone = builtin_desc.implementation.clone();
                            let result_id = self.lower_builtin_call(&impl_clone, dest, args)?;
                            self.current_register_map.insert(dest.id, result_id);
                        } else {
                            return Err(CompilerError::SpirvError(format!(
                                "Builtin function not found in registry: {}",
                                name
                            )));
                        }
                    }
                }
            }

            Instruction::MakeTuple(dest, elements) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let mut element_ids = Vec::new();
                for elem in elements {
                    element_ids.push(self.get_register(elem)?);
                }
                let result_id = self.builder.composite_construct(type_id, None, element_ids)?;
                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::ExtractElement(dest, composite, index) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let composite_id = self.get_register(composite)?;
                // Index is a u32 constant, use OpCompositeExtract
                let result_id = self.builder.composite_extract(type_id, None, composite_id, [*index])?;
                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::MakeArray(dest, elements) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let mut element_ids = Vec::new();
                for elem in elements {
                    element_ids.push(self.get_register(elem)?);
                }
                let result_id = self.builder.composite_construct(type_id, None, element_ids)?;
                self.current_register_map.insert(dest.id, result_id);
            }

            Instruction::ArrayIndex(dest, array, index) => {
                let type_id = self.get_or_create_type(&dest.ty)?;
                let array_id = self.get_register(array)?;
                let index_id = self.get_register(index)?;

                // Check if index is a constant integer
                // Look up the index register to see if it was created by ConstInt
                let is_const_index = self.int_const_cache.values().any(|&id| id == index_id);

                if is_const_index {
                    // Use OpCompositeExtract for constant indices
                    // Need to get the actual constant value
                    let index_value = self
                        .int_const_cache
                        .iter()
                        .find(|(_, &id)| id == index_id)
                        .map(|((val, _), _)| *val as u32)
                        .unwrap_or(0);

                    let result_id =
                        self.builder.composite_extract(type_id, None, array_id, [index_value])?;
                    self.current_register_map.insert(dest.id, result_id);
                } else {
                    // Dynamic index - use OpAccessChain (requires array to be in memory)
                    // For now, this is not supported without proper variable allocation
                    return Err(CompilerError::SpirvError(
                        "Dynamic array indexing not yet supported (requires proper variable allocation)"
                            .to_string(),
                    ));
                }
            }

            Instruction::ReturnVoid => {
                self.builder.ret()?;
            }

            Instruction::Return(value) => {
                if self.current_is_entry_point {
                    // For entry points, store to output variables instead of returning
                    let value_id = self.get_register(value)?;

                    // Check if the value is a tuple (multiple outputs)
                    if let Type::Constructed(TypeName::Str(name), component_types) = &value.ty {
                        if *name == "tuple" {
                            // Extract each component and store to corresponding output variable
                            for (i, output_var) in self.current_output_vars.clone().iter().enumerate() {
                                let component_type_id = self.get_or_create_type(&component_types[i])?;
                                let component_id = self.builder.composite_extract(
                                    component_type_id,
                                    None,
                                    value_id,
                                    [i as u32],
                                )?;
                                self.builder.store(*output_var, component_id, None, [])?;
                            }
                        } else {
                            // Single value
                            if let Some(&output_var) = self.current_output_vars.first() {
                                self.builder.store(output_var, value_id, None, [])?;
                            }
                        }
                    } else {
                        // Single value
                        if let Some(&output_var) = self.current_output_vars.first() {
                            self.builder.store(output_var, value_id, None, [])?;
                        }
                    }

                    // Return void
                    self.builder.ret()?;
                } else {
                    // Regular function return
                    let value_id = self.get_register(value)?;
                    self.builder.ret_value(value_id)?;
                }
            }

            Instruction::Call(dest, func_id, args) => {
                let result_type_id = self.get_or_create_type(&dest.ty)?;

                // Get SPIR-V function ID
                let spirv_func_id = self.function_map.get(func_id).copied().ok_or_else(|| {
                    CompilerError::SpirvError(format!("Function {} not found in lowering", func_id))
                })?;

                // Get argument IDs
                let mut arg_ids = Vec::new();
                for arg in args {
                    arg_ids.push(self.get_register(arg)?);
                }

                // Generate function call
                let result_id = self.builder.function_call(result_type_id, None, spirv_func_id, arg_ids)?;
                self.current_register_map.insert(dest.id, result_id);
            }

            // TODO: Implement remaining instructions
            _ => {
                return Err(CompilerError::SpirvError(format!(
                    "Instruction not yet implemented in lowering: {:?}",
                    inst
                )));
            }
        }

        Ok(())
    }

    /// Get the SPIR-V value ID for a MIR register
    fn get_register(&self, reg: &Register) -> Result<spirv::Word> {
        self.current_register_map
            .get(&reg.id)
            .copied()
            .ok_or_else(|| {
                eprintln!("DEBUG: Register {} not found. Current map: {:?}", reg.id, self.current_register_map);
                CompilerError::SpirvError(format!("Register {} not found", reg.id))
            })
    }

    /// Check if a type is a floating point type
    fn is_float_type(&self, ty: &Type<TypeName>) -> bool {
        matches!(ty, Type::Constructed(TypeName::Str(name), _) if *name == "f32" || name.starts_with("vec"))
    }

    /// Create global Input/Output variables for entry point interface
    fn create_entry_point_interface(&mut self, func: &mir::Function) -> Result<()> {
        use crate::ast::Attribute;

        let mut input_vars = Vec::new();
        let mut output_vars = Vec::new();

        // Create Input variables for parameters
        for (i, param) in func.params.iter().enumerate() {
            let param_type_id = self.get_or_create_type(&param.ty)?;
            let ptr_type_id = self.get_or_create_ptr_type(StorageClass::Input, param_type_id);

            // Create the global variable
            let var_id = self.builder.variable(ptr_type_id, None, StorageClass::Input, None);

            // Add decorations based on attributes
            if i < func.param_attributes.len() {
                for attr in &func.param_attributes[i] {
                    match attr {
                        Attribute::Location(loc) => {
                            self.builder.decorate(
                                var_id,
                                spirv::Decoration::Location,
                                [rspirv::dr::Operand::LiteralBit32(*loc)],
                            );
                        }
                        Attribute::BuiltIn(builtin) => {
                            self.builder.decorate(
                                var_id,
                                spirv::Decoration::BuiltIn,
                                [rspirv::dr::Operand::BuiltIn(*builtin)],
                            );
                        }
                        _ => {} // Ignore other attributes for now
                    }
                }
            }

            input_vars.push(var_id);

            // Store for loading later (must be done inside a block)
            self.current_input_vars.push((var_id, param.clone()));
        }

        // Create Output variables for return value
        // Handle both single return and tuple return (for multiple outputs)
        if let Some(attributed_types) = &func.attributed_return_types {
            // Multiple attributed outputs - create a global for each
            for attr_ty in attributed_types.iter() {
                let ret_type_id = self.get_or_create_type(&attr_ty.ty)?;
                let ptr_type_id = self.get_or_create_ptr_type(StorageClass::Output, ret_type_id);

                let var_id = self.builder.variable(ptr_type_id, None, StorageClass::Output, None);

                // Add decorations
                for attr in &attr_ty.attributes {
                    match attr {
                        Attribute::Location(loc) => {
                            self.builder.decorate(
                                var_id,
                                spirv::Decoration::Location,
                                [rspirv::dr::Operand::LiteralBit32(*loc)],
                            );
                        }
                        Attribute::BuiltIn(builtin) => {
                            self.builder.decorate(
                                var_id,
                                spirv::Decoration::BuiltIn,
                                [rspirv::dr::Operand::BuiltIn(*builtin)],
                            );
                        }
                        _ => {}
                    }
                }

                output_vars.push(var_id);
                self.current_output_vars.push(var_id);
            }
        } else {
            // Single return value - create one output variable (unless void)
            use crate::ast::TypeName;
            let is_void =
                matches!(&func.return_type, Type::Constructed(TypeName::Str(name), _) if *name == "void");

            if !is_void {
                let ret_type_id = self.get_or_create_type(&func.return_type)?;
                let ptr_type_id = self.get_or_create_ptr_type(StorageClass::Output, ret_type_id);

                let var_id = self.builder.variable(ptr_type_id, None, StorageClass::Output, None);

                // Add decorations from return_attributes
                for attr in &func.return_attributes {
                    match attr {
                        Attribute::Location(loc) => {
                            self.builder.decorate(
                                var_id,
                                spirv::Decoration::Location,
                                [rspirv::dr::Operand::LiteralBit32(*loc)],
                            );
                        }
                        Attribute::BuiltIn(builtin) => {
                            self.builder.decorate(
                                var_id,
                                spirv::Decoration::BuiltIn,
                                [rspirv::dr::Operand::BuiltIn(*builtin)],
                            );
                        }
                        _ => {}
                    }
                }

                output_vars.push(var_id);
                self.current_output_vars.push(var_id);
            }
        }

        // Store the interface variables for this entry point
        self.entry_point_interfaces.insert(func.id, (input_vars, output_vars));

        Ok(())
    }

    /// Lower a builtin function call to SPIR-V
    fn lower_builtin_call(
        &mut self,
        impl_: &BuiltinImpl,
        dest: &Register,
        args: &[Register],
    ) -> Result<spirv::Word> {
        let result_type_id = self.get_or_create_type(&dest.ty)?;
        let mut arg_ids = Vec::new();
        for arg in args {
            arg_ids.push(self.get_register(arg)?);
        }

        match impl_ {
            BuiltinImpl::GlslExt(inst_num) => {
                // GLSL.std.450 extended instruction
                let operands: Vec<_> = arg_ids.into_iter().map(Operand::IdRef).collect();
                let result_id = self.builder.ext_inst(
                    result_type_id,
                    None,
                    self.glsl_ext_inst_id,
                    *inst_num,
                    operands,
                )?;
                Ok(result_id)
            }
            BuiltinImpl::SpirvOp(op) => {
                // Core SPIR-V operation
                self.lower_spirv_op(op, result_type_id, &arg_ids)
            }
            BuiltinImpl::Custom(_) => Err(CompilerError::SpirvError(
                "Custom builtin implementations not yet supported".to_string(),
            )),
        }
    }

    /// Lower a SPIR-V core operation
    fn lower_spirv_op(
        &mut self,
        op: &SpirvOp,
        result_type_id: spirv::Word,
        arg_ids: &[spirv::Word],
    ) -> Result<spirv::Word> {
        use spirv::Op;

        let result_id = match op {
            // Arithmetic
            SpirvOp::FAdd => self.builder.f_add(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::FSub => self.builder.f_sub(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::FMul => self.builder.f_mul(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::FDiv => self.builder.f_div(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::FRem => self.builder.f_rem(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::FMod => self.builder.f_mod(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::IAdd => self.builder.i_add(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::ISub => self.builder.i_sub(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::IMul => self.builder.i_mul(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::SDiv => self.builder.s_div(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::UDiv => self.builder.u_div(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::SRem => self.builder.s_rem(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::SMod => self.builder.s_mod(result_type_id, None, arg_ids[0], arg_ids[1])?,

            // Comparisons
            SpirvOp::FOrdEqual => self.builder.f_ord_equal(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::FOrdNotEqual => {
                self.builder.f_ord_not_equal(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::FOrdLessThan => {
                self.builder.f_ord_less_than(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::FOrdGreaterThan => {
                self.builder.f_ord_greater_than(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::FOrdLessThanEqual => {
                self.builder.f_ord_less_than_equal(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::FOrdGreaterThanEqual => {
                self.builder.f_ord_greater_than_equal(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::IEqual => self.builder.i_equal(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::INotEqual => self.builder.i_not_equal(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::SLessThan => self.builder.s_less_than(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::ULessThan => self.builder.u_less_than(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::SGreaterThan => {
                self.builder.s_greater_than(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::UGreaterThan => {
                self.builder.u_greater_than(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::SLessThanEqual => {
                self.builder.s_less_than_equal(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::ULessThanEqual => {
                self.builder.u_less_than_equal(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::SGreaterThanEqual => {
                self.builder.s_greater_than_equal(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::UGreaterThanEqual => {
                self.builder.u_greater_than_equal(result_type_id, None, arg_ids[0], arg_ids[1])?
            }

            // Bitwise
            SpirvOp::BitwiseAnd => {
                self.builder.bitwise_and(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::BitwiseOr => self.builder.bitwise_or(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::BitwiseXor => {
                self.builder.bitwise_xor(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::Not => self.builder.not(result_type_id, None, arg_ids[0])?,
            SpirvOp::ShiftLeftLogical => {
                self.builder.shift_left_logical(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::ShiftRightArithmetic => {
                self.builder.shift_right_arithmetic(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::ShiftRightLogical => {
                self.builder.shift_right_logical(result_type_id, None, arg_ids[0], arg_ids[1])?
            }

            // Vector/Matrix operations
            SpirvOp::Dot => self.builder.dot(result_type_id, None, arg_ids[0], arg_ids[1])?,
            SpirvOp::OuterProduct => {
                self.builder.outer_product(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::MatrixTimesMatrix => {
                self.builder.matrix_times_matrix(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::MatrixTimesVector => {
                self.builder.matrix_times_vector(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::VectorTimesMatrix => {
                self.builder.vector_times_matrix(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::VectorTimesScalar => {
                self.builder.vector_times_scalar(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
            SpirvOp::MatrixTimesScalar => {
                self.builder.matrix_times_scalar(result_type_id, None, arg_ids[0], arg_ids[1])?
            }
        };

        Ok(result_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::Builder as MirBuilder;

    #[test]
    fn test_lower_simple_function() {
        // Create a simple MIR function: def add(x: i32, y: i32): i32 = x + y
        let mut mir_builder = MirBuilder::new();

        let i32_ty = Type::Constructed(TypeName::Str("i32"), vec![]);

        let func_id = mir_builder.begin_function(
            "add".to_string(),
            vec![
                ("x".to_string(), i32_ty.clone()),
                ("y".to_string(), i32_ty.clone()),
            ],
            i32_ty.clone(),
        );

        // Get parameter registers (they're id 0 and 1)
        let x_reg = Register {
            id: 0,
            ty: i32_ty.clone(),
        };
        let y_reg = Register {
            id: 1,
            ty: i32_ty.clone(),
        };

        let sum_reg = mir_builder.build_add(x_reg, y_reg);
        mir_builder.build_return(sum_reg);
        mir_builder.end_function();

        let module = mir_builder.finish(vec![func_id]);

        // Lower to SPIR-V
        let lowering = Lowering::new();
        let spirv = lowering.lower_module(&module).unwrap();

        // Verify we got SPIR-V output
        assert!(!spirv.is_empty());
    }
}
