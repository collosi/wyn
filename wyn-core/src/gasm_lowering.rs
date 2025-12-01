/// GASM to SPIR-V Lowering
///
/// This module provides lowering from GASM (Graphics Assembly) IR to SPIR-V.
/// GASM is a low-level SSA-based IR that closely mirrors SPIR-V operations.
use gasm::{
    AddressSpace, BasicBlock, Constant, Function, Global, Module, Operation, ReturnType, Terminator, Type,
    Value,
};
use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use spirv::Word;
use std::collections::HashMap;

pub struct GasmLowering {
    builder: Builder,
    u32_type: Word,

    /// Map from GASM register names to SPIR-V IDs
    registers: HashMap<String, Word>,

    /// Map from GASM global names to SPIR-V IDs
    globals: HashMap<String, Word>,

    /// Map from GASM function names to SPIR-V function IDs
    functions: HashMap<String, Word>,

    /// Map from GASM basic block labels to SPIR-V IDs
    labels: HashMap<String, Word>,

    /// Map from GASM types to SPIR-V type IDs
    type_cache: HashMap<Type, Word>,
}

impl GasmLowering {
    pub fn new() -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 3);

        // Initialize with basic types
        let u32_type = builder.type_int(32, 0);

        let mut type_cache = HashMap::new();
        type_cache.insert(Type::U32, u32_type);

        Self {
            builder,
            u32_type,
            registers: HashMap::new(),
            globals: HashMap::new(),
            functions: HashMap::new(),
            labels: HashMap::new(),
            type_cache,
        }
    }

    /// Lower a GASM module to SPIR-V
    pub fn lower_module(mut self, module: &Module) -> Result<Vec<u32>, String> {
        // Set up SPIR-V header
        self.builder.capability(spirv::Capability::Shader);
        self.builder.capability(spirv::Capability::VulkanMemoryModel);
        self.builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::Vulkan);

        // Lower globals
        for global in &module.globals {
            self.lower_global(global)?;
        }

        // Declare functions
        for function in &module.functions {
            self.declare_function(function)?;
        }

        // Lower function bodies
        for function in &module.functions {
            self.lower_function(function)?;
        }

        Ok(self.builder.module().assemble())
    }

    /// Get or create a SPIR-V type for a GASM type
    fn get_or_create_type(&mut self, ty: &Type) -> Word {
        if let Some(&type_id) = self.type_cache.get(ty) {
            return type_id;
        }

        let type_id = match ty {
            Type::I8 => self.builder.type_int(8, 1),
            Type::I16 => self.builder.type_int(16, 1),
            Type::I32 => self.builder.type_int(32, 1),
            Type::I64 => self.builder.type_int(64, 1),
            Type::U8 => self.builder.type_int(8, 0),
            Type::U16 => self.builder.type_int(16, 0),
            Type::U32 => self.builder.type_int(32, 0),
            Type::U64 => self.builder.type_int(64, 0),
            Type::F16 => self.builder.type_float(16),
            Type::F32 => self.builder.type_float(32),
            Type::F64 => self.builder.type_float(64),
            Type::Pointer(ptr_ty) => {
                let pointee_type = self.get_or_create_type(&ptr_ty.pointee);
                let storage_class = self.address_space_to_storage_class(&ptr_ty.address_space);
                self.builder.type_pointer(None, storage_class, pointee_type)
            }
            Type::Array(element_ty, len) => {
                // Fixed-size array [N; T]
                let element_type = self.get_or_create_type(element_ty);
                let len_const = self.builder.constant_bit32(self.u32_type, *len);
                self.builder.type_array(element_type, len_const)
            }
            Type::RuntimeArray(element_ty) => {
                // Unsized runtime array, only for StorageBuffer
                let element_type = self.get_or_create_type(element_ty);
                self.builder.type_runtime_array(element_type)
            }
        };

        self.type_cache.insert(ty.clone(), type_id);
        type_id
    }

    /// Convert GASM address space to SPIR-V storage class
    fn address_space_to_storage_class(&self, addr_space: &AddressSpace) -> spirv::StorageClass {
        match addr_space {
            AddressSpace::Generic => spirv::StorageClass::Generic,
            AddressSpace::Global => spirv::StorageClass::StorageBuffer,
            AddressSpace::Shared => spirv::StorageClass::Workgroup,
            AddressSpace::Local => spirv::StorageClass::Function,
            AddressSpace::Private => spirv::StorageClass::Private,
            AddressSpace::Const => spirv::StorageClass::UniformConstant,
        }
    }

    /// Lower a global variable
    fn lower_global(&mut self, global: &Global) -> Result<(), String> {
        let pointee_type = self.get_or_create_type(&global.ty.pointee);
        let storage_class = self.address_space_to_storage_class(&global.ty.address_space);
        let ptr_type = self.builder.type_pointer(None, storage_class, pointee_type);

        // TODO: Handle initializers
        let var_id = self.builder.variable(ptr_type, None, storage_class, None);

        self.globals.insert(global.name.clone(), var_id);
        Ok(())
    }

    /// Declare a function (create its type and ID)
    fn declare_function(&mut self, function: &Function) -> Result<(), String> {
        // Build parameter types
        let param_types: Vec<Word> =
            function.params.iter().map(|p| self.get_or_create_type(&p.ty)).collect();

        // Build return type
        let return_type_id = match &function.return_type {
            ReturnType::Void => self.builder.type_void(),
            ReturnType::Type(ty) => self.get_or_create_type(ty),
        };

        // Create function type
        let fn_type = self.builder.type_function(return_type_id, param_types.clone());

        // Create function
        let fn_id = self
            .builder
            .begin_function(return_type_id, None, spirv::FunctionControl::NONE, fn_type)
            .unwrap();

        self.functions.insert(function.name.clone(), fn_id);

        // End function declaration (we'll fill in the body later)
        self.builder.end_function().unwrap();

        Ok(())
    }

    /// Lower a function body
    fn lower_function(&mut self, function: &Function) -> Result<(), String> {
        // Clear register and label maps for this function
        self.registers.clear();
        self.labels.clear();

        // Get function ID
        let fn_id = *self
            .functions
            .get(&function.name)
            .ok_or_else(|| format!("Function {} not declared", function.name))?;

        // Build parameter types and return type again
        let param_types: Vec<Word> =
            function.params.iter().map(|p| self.get_or_create_type(&p.ty)).collect();

        let return_type_id = match &function.return_type {
            ReturnType::Void => self.builder.type_void(),
            ReturnType::Type(ty) => self.get_or_create_type(ty),
        };

        let fn_type = self.builder.type_function(return_type_id, param_types.clone());

        // Begin function
        self.builder
            .begin_function(return_type_id, Some(fn_id), spirv::FunctionControl::NONE, fn_type)
            .unwrap();

        // Create parameters
        for param in &function.params {
            let param_type = self.get_or_create_type(&param.ty);
            let param_id = self.builder.function_parameter(param_type).unwrap();
            self.registers.insert(param.name.clone(), param_id);
        }

        // Pre-allocate labels for all basic blocks
        for block in &function.blocks {
            let label_id = self.builder.id();
            self.labels.insert(block.label.clone(), label_id);
        }

        // Pre-allocate IDs for all SSA register definitions.
        // This is needed because phi nodes can reference values defined later.
        // Now that we removed mov/uconst/iconst/fconst, every SSA def produces
        // a real SPIR-V result-id, so this is straightforward.
        for block in &function.blocks {
            for inst in &block.instructions {
                if let Some(result_name) = &inst.result {
                    if !self.registers.contains_key(result_name) {
                        let reg_id = self.builder.id();
                        self.registers.insert(result_name.clone(), reg_id);
                    }
                }
            }
        }

        // Lower basic blocks
        for block in &function.blocks {
            self.lower_basic_block(block)?;
        }

        // End function
        self.builder.end_function().unwrap();

        Ok(())
    }

    /// Lower a basic block
    fn lower_basic_block(&mut self, block: &BasicBlock) -> Result<(), String> {
        // Get the label ID
        let label_id =
            *self.labels.get(&block.label).ok_or_else(|| format!("Label {} not found", block.label))?;

        // Begin basic block
        self.builder.begin_block(Some(label_id)).unwrap();

        // Lower instructions
        for inst in &block.instructions {
            self.lower_instruction(inst)?;
        }

        // If this is a loop header, emit OpLoopMerge right before the terminator
        // (OpLoopMerge must be the second-to-last instruction in the block)
        if let Some((merge_label, continue_label)) = &block.loop_header {
            let merge_id = *self
                .labels
                .get(merge_label)
                .ok_or_else(|| format!("Loop merge label {} not found", merge_label))?;
            let continue_id = *self
                .labels
                .get(continue_label)
                .ok_or_else(|| format!("Loop continue label {} not found", continue_label))?;
            self.builder
                .loop_merge(merge_id, continue_id, spirv::LoopControl::NONE, [])
                .map_err(|e| format!("Failed to emit OpLoopMerge: {:?}", e))?;
        }

        // Lower terminator
        self.lower_terminator(&block.terminator)?;

        Ok(())
    }

    /// Get the pre-allocated result ID for an instruction
    fn get_result_id(&self, inst: &gasm::Instruction) -> Option<spirv::Word> {
        inst.result.as_ref().and_then(|name| self.registers.get(name).copied())
    }

    /// Lower an instruction
    fn lower_instruction(&mut self, inst: &gasm::Instruction) -> Result<(), String> {
        // Get the pre-allocated result ID (if this instruction produces a result)
        let result_id = self.get_result_id(inst);

        // Lower the operation
        match &inst.op {
            // Bitwise operations
            Operation::And(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .bitwise_and(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Or(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .bitwise_or(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Xor(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .bitwise_xor(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::IXor(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let i32_type = self.builder.type_int(32, 1);
                self.builder
                    .bitwise_xor(i32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Shl(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .shift_left_logical(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Shr(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .shift_right_logical(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            // Arithmetic operations
            Operation::Add(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .i_add(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Sub(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .i_sub(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Rem(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .u_mod(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Div(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                self.builder
                    .u_div(self.u32_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            // Comparisons
            Operation::UCmpLt(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .u_less_than(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::UCmpLe(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .u_less_than_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::UCmpGt(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .u_greater_than(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::UCmpGe(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .u_greater_than_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::UCmpEq(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .i_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::UCmpNe(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .i_not_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            // Signed integer comparisons
            Operation::ICmpLt(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .s_less_than(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::ICmpLe(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .s_less_than_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::ICmpGt(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .s_greater_than(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::ICmpGe(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .s_greater_than_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::ICmpEq(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .i_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::ICmpNe(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                self.builder
                    .i_not_equal(bool_type, result_id, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            // Select (u32)
            Operation::Select(cond, true_val, false_val) => {
                let cond_id = self.get_value(cond)?;
                let true_id = self.get_value(true_val)?;
                let false_id = self.get_value(false_val)?;
                self.builder
                    .select(self.u32_type, result_id, cond_id, true_id, false_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            // ISelect (i32)
            Operation::ISelect(cond, true_val, false_val) => {
                let cond_id = self.get_value(cond)?;
                let true_id = self.get_value(true_val)?;
                let false_id = self.get_value(false_val)?;
                let i32_type = self.builder.type_int(32, 1);
                self.builder
                    .select(i32_type, result_id, cond_id, true_id, false_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            // Type conversions
            Operation::Bitcast(val) => {
                let val_id = self.get_value(val)?;
                // TODO: Get proper result type
                self.builder
                    .bitcast(self.u32_type, result_id, val_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            // Memory operations
            Operation::Load(ptr) => {
                let ptr_id = self.get_value(ptr)?;
                // TODO: Get proper result type
                self.builder
                    .load(self.u32_type, result_id, ptr_id, None, [])
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Store(ptr, value) => {
                let ptr_id = self.get_value(ptr)?;
                let value_id = self.get_value(value)?;
                self.builder
                    .store(ptr_id, value_id, None, [])
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Gep {
                result_type,
                base,
                index,
                stride: _,
            } => {
                let base_id = self.get_value(base)?;
                let index_id = self.get_value(index)?;

                // Get the SPIR-V pointer type from the explicit result_type
                let ptr_type_id = self.get_or_create_type(&Type::Pointer(Box::new(result_type.clone())));

                // SPIR-V AccessChain indexing depends on the base type:
                // - For globals (storage buffers): base is ptr<Struct<RuntimeArray>>, need [0, index]
                //   to first enter the struct member, then index into the array
                // - For other pointers (ptr<Array> or ptr<RuntimeArray>): just need [index]
                let is_global = matches!(base, Value::Global(_));
                if is_global {
                    let const_0 = self.builder.constant_bit32(self.u32_type, 0);
                    self.builder
                        .access_chain(ptr_type_id, result_id, base_id, [const_0, index_id])
                        .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                } else {
                    self.builder
                        .access_chain(ptr_type_id, result_id, base_id, [index_id])
                        .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                }
                return Ok(());
            }

            // Atomic operations
            Operation::AtomicLoad { ptr, ordering, scope } => {
                let ptr_id = self.get_value(ptr)?;
                let scope_id = self.lower_memory_scope(scope);
                let ordering_id = self.lower_memory_ordering(ordering);

                self.builder
                    .atomic_load(self.u32_type, result_id, ptr_id, scope_id, ordering_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::AtomicRmw {
                op,
                ptr,
                value,
                ordering,
                scope,
            } => {
                use gasm::AtomicOp;
                let ptr_id = self.get_value(ptr)?;
                let value_id = self.get_value(value)?;
                let scope_id = self.lower_memory_scope(scope);
                let ordering_id = self.lower_memory_ordering(ordering);

                match op {
                    AtomicOp::Add => self
                        .builder
                        .atomic_i_add(self.u32_type, result_id, ptr_id, scope_id, ordering_id, value_id)
                        .map_err(|e| format!("SPIR-V error: {:?}", e))?,
                    _ => return Err(format!("Atomic operation {:?} not yet implemented", op)),
                };
                return Ok(());
            }

            Operation::Phi { ty, incoming } => {
                let result_type = self.get_or_create_type(ty);
                let pairs: Result<Vec<_>, String> = incoming
                    .iter()
                    .map(|(val, label)| {
                        let val_id = self.get_value(val)?;
                        let label_id = self
                            .labels
                            .get(label)
                            .copied()
                            .ok_or_else(|| format!("Label {} not found in phi", label))?;
                        Ok((val_id, label_id))
                    })
                    .collect();
                let pairs = pairs?;
                self.builder
                    .phi(result_type, result_id, pairs)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Call { func, args } => {
                // Look up the function ID
                let func_id = self
                    .functions
                    .get(func)
                    .copied()
                    .ok_or_else(|| format!("Function {} not found", func))?;

                // Lower arguments
                let arg_ids: Result<Vec<Word>, String> =
                    args.iter().map(|arg| self.get_value(arg)).collect();
                let arg_ids = arg_ids?;

                // For void functions, we don't need a result
                // SPIR-V OpFunctionCall requires a result type
                let void_type = self.builder.type_void();
                self.builder
                    .function_call(void_type, None, func_id, arg_ids)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            _ => return Err(format!("Instruction {:?} not yet implemented", inst.op)),
        }
    }

    /// Lower a constant to a SPIR-V constant
    fn lower_constant(&mut self, constant: &Constant) -> Result<Word, String> {
        let const_id = match constant {
            Constant::I8(v) => {
                let ty = self.builder.type_int(8, 1);
                self.builder.constant_bit32(ty, *v as u32)
            }
            Constant::I16(v) => {
                let ty = self.builder.type_int(16, 1);
                self.builder.constant_bit32(ty, *v as u32)
            }
            Constant::I32(v) => {
                let ty = self.builder.type_int(32, 1);
                self.builder.constant_bit32(ty, *v as u32)
            }
            Constant::I64(v) => {
                let ty = self.builder.type_int(64, 1);
                self.builder.constant_bit64(ty, *v as u64)
            }
            Constant::U8(v) => {
                let ty = self.builder.type_int(8, 0);
                self.builder.constant_bit32(ty, *v as u32)
            }
            Constant::U16(v) => {
                let ty = self.builder.type_int(16, 0);
                self.builder.constant_bit32(ty, *v as u32)
            }
            Constant::U32(v) => {
                let ty = self.builder.type_int(32, 0);
                self.builder.constant_bit32(ty, *v)
            }
            Constant::U64(v) => {
                let ty = self.builder.type_int(64, 0);
                self.builder.constant_bit64(ty, *v)
            }
            Constant::F16(bits) => {
                let ty = self.builder.type_float(16);
                self.builder.constant_bit32(ty, *bits as u32)
            }
            Constant::F32(v) => {
                let ty = self.builder.type_float(32);
                self.builder.constant_bit32(ty, v.to_bits())
            }
            Constant::F64(v) => {
                let ty = self.builder.type_float(64);
                self.builder.constant_bit64(ty, v.to_bits())
            }
        };

        Ok(const_id)
    }

    /// Get the SPIR-V ID for a GASM value
    fn get_value(&mut self, value: &Value) -> Result<Word, String> {
        match value {
            Value::Register(name) => {
                self.registers.get(name).copied().ok_or_else(|| format!("Register {} not found", name))
            }
            Value::Global(name) => {
                self.globals.get(name).copied().ok_or_else(|| format!("Global {} not found", name))
            }
            Value::Constant(c) => self.lower_constant(c),
        }
    }

    /// Convert GASM memory ordering to SPIR-V constant
    fn lower_memory_ordering(&mut self, ordering: &gasm::MemoryOrdering) -> Word {
        use gasm::MemoryOrdering;
        let semantics = match ordering {
            MemoryOrdering::Relaxed => spirv::MemorySemantics::NONE,
            MemoryOrdering::Acquire => spirv::MemorySemantics::ACQUIRE,
            MemoryOrdering::Release => spirv::MemorySemantics::RELEASE,
            MemoryOrdering::AcqRel => spirv::MemorySemantics::ACQUIRE_RELEASE,
            MemoryOrdering::SeqCst => spirv::MemorySemantics::SEQUENTIALLY_CONSISTENT,
        };
        self.builder.constant_bit32(self.u32_type, semantics.bits())
    }

    /// Convert GASM memory scope to SPIR-V constant
    fn lower_memory_scope(&mut self, scope: &gasm::MemoryScope) -> Word {
        use gasm::MemoryScope;
        let scope_val = match scope {
            MemoryScope::Invocation => spirv::Scope::Invocation,
            MemoryScope::Subgroup => spirv::Scope::Subgroup,
            MemoryScope::Workgroup => spirv::Scope::Workgroup,
            MemoryScope::Device => spirv::Scope::Device,
            MemoryScope::System => spirv::Scope::CrossDevice,
        };
        self.builder.constant_bit32(self.u32_type, scope_val as u32)
    }

    /// Lower a terminator instruction
    fn lower_terminator(&mut self, terminator: &Terminator) -> Result<(), String> {
        match terminator {
            Terminator::Ret(val) => {
                if let Some(v) = val {
                    let val_id = self.get_value(v)?;
                    self.builder.ret_value(val_id).unwrap();
                } else {
                    self.builder.ret().unwrap();
                }
            }

            Terminator::Br(label) => {
                let label_id =
                    *self.labels.get(label).ok_or_else(|| format!("Label {} not found", label))?;
                self.builder.branch(label_id).unwrap();
            }

            Terminator::BrIf {
                cond,
                true_label,
                false_label,
                merge_label,
            } => {
                let cond_id = self.get_value(cond)?;
                let true_id = *self
                    .labels
                    .get(true_label)
                    .ok_or_else(|| format!("Label {} not found", true_label))?;
                let false_id = *self
                    .labels
                    .get(false_label)
                    .ok_or_else(|| format!("Label {} not found", false_label))?;
                // Emit OpSelectionMerge before OpBranchConditional only if merge_label is provided
                // (not needed when inside a loop header block, which has its own OpLoopMerge)
                if let Some(merge_label) = merge_label {
                    let merge_id = *self
                        .labels
                        .get(merge_label)
                        .ok_or_else(|| format!("Merge label {} not found", merge_label))?;
                    self.builder.selection_merge(merge_id, spirv::SelectionControl::NONE).unwrap();
                }
                self.builder.branch_conditional(cond_id, true_id, false_id, []).unwrap();
            }
        }

        Ok(())
    }
}

/// Lower a GASM module to SPIR-V bytecode
pub fn lower_gasm_module(module: &Module) -> Result<Vec<u32>, String> {
    let lowering = GasmLowering::new();
    lowering.lower_module(module)
}

/// Lower a single GASM function into an existing SPIR-V Builder
/// Returns the SPIR-V function ID
///
/// globals: Map from GASM global names (like "gdp_buffer", without @ prefix) to SPIR-V variable IDs
/// type_cache: Map from GASM types to SPIR-V type IDs (shared across multiple function lowerings)
/// functions: Map from GASM function names to SPIR-V function IDs (for calling other GASM functions)
pub fn lower_function_into_builder(
    builder: &mut Builder,
    function: &Function,
    globals: HashMap<String, Word>,
    type_cache: &mut HashMap<Type, Word>,
    functions: HashMap<String, Word>,
) -> Result<Word, String> {
    // Create a minimal GasmLowering context for this function
    let u32_type = builder.type_int(32, 0);
    // Initialize type cache with basic types if not present
    type_cache.entry(Type::U32).or_insert(u32_type);

    let mut lowering = GasmLowering {
        builder: std::mem::replace(builder, Builder::new()), // Temporarily take ownership
        u32_type,
        registers: HashMap::new(),
        globals,   // Use provided globals map
        functions, // Use provided functions map
        labels: HashMap::new(),
        type_cache: std::mem::take(type_cache), // Take ownership temporarily
    };

    // First, declare the function to get its ID
    let fn_id = lowering.builder.id();
    lowering.functions.insert(function.name.clone(), fn_id);

    // Now lower the function (this will reference the ID we just created)
    lowering.lower_function(function)?;

    // Return the builder and updated type cache
    *builder = lowering.builder;
    *type_cache = lowering.type_cache;

    Ok(fn_id)
}
