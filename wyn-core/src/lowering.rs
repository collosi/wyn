//! SPIR-V Lowering
//!
//! This module converts MIR (Mid-level Intermediate Representation) to SPIR-V.

use crate::ast::TypeName;
use crate::error::{CompilerError, Result};
use crate::mir::{self, BlockId, FunctionId, Instruction, Module as MirModule, Register};
use polytype::Type;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Module as SpirvModule};
use rspirv::spirv::{self, AddressingModel, Capability, ExecutionModel, MemoryModel, StorageClass};
use std::collections::HashMap;

/// Maps MIR registers to SPIR-V value IDs
type RegisterMap = HashMap<u32, spirv::Word>;

/// Maps MIR function IDs to SPIR-V function IDs
type FunctionMap = HashMap<FunctionId, spirv::Word>;

/// Lowers MIR to SPIR-V
pub struct Lowering {
    builder: Builder,
    module: SpirvModule,

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

    // Function mapping
    function_map: FunctionMap,

    // Current function context
    current_register_map: RegisterMap,
    current_block_map: HashMap<BlockId, spirv::Word>,
}

impl Lowering {
    pub fn new() -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 0);

        let mut lowering = Lowering {
            builder,
            module: SpirvModule::new(),
            type_cache: HashMap::new(),
            ptr_cache: HashMap::new(),
            int_const_cache: HashMap::new(),
            float_const_cache: HashMap::new(),
            bool_const_cache: HashMap::new(),
            void_type: 0,
            bool_type: 0,
            i32_type: 0,
            f32_type: 0,
            function_map: HashMap::new(),
            current_register_map: HashMap::new(),
            current_block_map: HashMap::new(),
        };

        // Initialize common types
        lowering.void_type = lowering.builder.type_void();
        lowering.bool_type = lowering.builder.type_bool();
        lowering.i32_type = lowering.builder.type_int(32, 1);
        lowering.f32_type = lowering.builder.type_float(32);

        lowering
    }

    /// Lower a complete MIR module to SPIR-V
    pub fn lower_module(mut self, mir: &MirModule) -> Result<Vec<u32>> {
        // Set up SPIR-V capabilities and addressing model
        self.builder.capability(Capability::Shader);
        self.builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        // Lower all functions
        for function in &mir.functions {
            self.lower_function(function)?;
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

                self.builder.entry_point(execution_model, spirv_func_id, &func.name, &[]);
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
            Type::Constructed(TypeName::Str(name), args) => {
                match *name {
                    "i32" => self.i32_type,
                    "f32" => self.f32_type,
                    "bool" => self.bool_type,
                    "void" => self.void_type,
                    "vec2" => {
                        let f32_id = self.f32_type;
                        self.builder.type_vector(f32_id, 2)
                    }
                    "vec3" => {
                        let f32_id = self.f32_type;
                        self.builder.type_vector(f32_id, 3)
                    }
                    "vec4" => {
                        let f32_id = self.f32_type;
                        self.builder.type_vector(f32_id, 4)
                    }
                    "tuple" => {
                        // Get component types
                        let mut component_type_ids = Vec::new();
                        for arg in args {
                            component_type_ids.push(self.get_or_create_type(arg)?);
                        }
                        self.builder.type_struct(component_type_ids)
                    }
                    "Array" if args.len() == 2 => {
                        // Array(Size(n), element_type)
                        if let Type::Constructed(TypeName::Size(n), _) = &args[0] {
                            let elem_type_id = self.get_or_create_type(&args[1])?;
                            let len_const = self.get_or_create_int_const(*n as i32, self.i32_type);
                            self.builder.type_array(elem_type_id, len_const)
                        } else {
                            return Err(CompilerError::SpirvError(format!(
                                "Invalid array size type: {:?}",
                                args[0]
                            )));
                        }
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
    fn lower_function(&mut self, func: &mir::Function) -> Result<()> {
        // Create function type
        let return_type_id = self.get_or_create_type(&func.return_type)?;
        let mut param_type_ids = Vec::new();
        for param in &func.params {
            param_type_ids.push(self.get_or_create_type(&param.ty)?);
        }

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

        // Create parameters
        for (i, param) in func.params.iter().enumerate() {
            let param_id = self.builder.function_parameter(param_type_ids[i])?;
            self.current_register_map.insert(param.id, param_id);
        }

        // Create all blocks first (for forward references)
        for block in &func.blocks {
            let block_id = self.builder.id();
            self.current_block_map.insert(block.id, block_id);
        }

        // Lower all blocks
        for block in &func.blocks {
            self.lower_block(block)?;
        }

        self.builder.end_function()?;
        Ok(())
    }

    /// Lower a MIR basic block to SPIR-V
    fn lower_block(&mut self, block: &mir::Block) -> Result<()> {
        let spirv_block_id = *self
            .current_block_map
            .get(&block.id)
            .ok_or_else(|| CompilerError::SpirvError(format!("Block {} not found in map", block.id)))?;

        self.builder.begin_block(Some(spirv_block_id))?;

        for instruction in &block.instructions {
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

            Instruction::BranchCond(cond, true_block, false_block) => {
                let cond_id = self.get_register(cond)?;
                let true_id = *self.current_block_map.get(true_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Block {} not found in map", true_block))
                })?;
                let false_id = *self.current_block_map.get(false_block).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Block {} not found in map", false_block))
                })?;
                self.builder.branch_conditional(cond_id, true_id, false_id, [])?;
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
                // Handle builtin functions like vec4, vec3, etc.
                match name.as_str() {
                    "vec2" | "vec3" | "vec4" => {
                        let type_id = self.get_or_create_type(&dest.ty)?;
                        let mut arg_ids = Vec::new();
                        for arg in args {
                            arg_ids.push(self.get_register(arg)?);
                        }
                        let result_id = self.builder.composite_construct(type_id, None, arg_ids)?;
                        self.current_register_map.insert(dest.id, result_id);
                    }
                    _ => {
                        return Err(CompilerError::SpirvError(format!(
                            "Builtin function not yet implemented: {}",
                            name
                        )));
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

            Instruction::Return(value) => {
                let value_id = self.get_register(value)?;
                self.builder.ret_value(value_id)?;
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
            .ok_or_else(|| CompilerError::SpirvError(format!("Register {} not found", reg.id)))
    }

    /// Check if a type is a floating point type
    fn is_float_type(&self, ty: &Type<TypeName>) -> bool {
        matches!(ty, Type::Constructed(TypeName::Str(name), _) if *name == "f32" || name.starts_with("vec"))
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
