/// GASM to SPIR-V Lowering
///
/// This module provides lowering from GASM (Graphics Assembly) IR to SPIR-V.
/// GASM is a low-level SSA-based IR that closely mirrors SPIR-V operations.

use gasm::{
    AddressSpace, BasicBlock, Constant, Function, Global, Module, Operation, ReturnType,
    Terminator, Type, Value,
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
        self.builder.memory_model(
            spirv::AddressingModel::Logical,
            spirv::MemoryModel::Vulkan,
        );

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
        let param_types: Vec<Word> = function
            .params
            .iter()
            .map(|p| self.get_or_create_type(&p.ty))
            .collect();

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
            .begin_function(
                return_type_id,
                None,
                spirv::FunctionControl::NONE,
                fn_type,
            )
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
        let fn_id = *self.functions.get(&function.name)
            .ok_or_else(|| format!("Function {} not declared", function.name))?;

        // Build parameter types and return type again
        let param_types: Vec<Word> = function
            .params
            .iter()
            .map(|p| self.get_or_create_type(&p.ty))
            .collect();

        let return_type_id = match &function.return_type {
            ReturnType::Void => self.builder.type_void(),
            ReturnType::Type(ty) => self.get_or_create_type(ty),
        };

        let fn_type = self.builder.type_function(return_type_id, param_types.clone());

        // Begin function
        self.builder.begin_function(
            return_type_id,
            Some(fn_id),
            spirv::FunctionControl::NONE,
            fn_type,
        ).unwrap();

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
        let label_id = *self.labels.get(&block.label)
            .ok_or_else(|| format!("Label {} not found", block.label))?;

        // Begin basic block
        self.builder.begin_block(Some(label_id)).unwrap();

        // Lower instructions
        for inst in &block.instructions {
            self.lower_instruction(inst)?;
        }

        // Lower terminator
        self.lower_terminator(&block.terminator)?;

        Ok(())
    }

    /// Lower an instruction
    fn lower_instruction(&mut self, inst: &gasm::Instruction) -> Result<(), String> {
        let _result_id = if let Some(_result_name) = &inst.result {
            Some(self.builder.id())
        } else {
            None
        };

        // Lower the operation
        match &inst.op {
            Operation::Mov(val) => {
                let val_id = self.get_value(val)?;
                // Mov is just a copy - we can use OpCopyObject
                // But we need the type
                // TODO: Track types for values
                self.registers.insert(inst.result.as_ref().unwrap().clone(), val_id);
                return Ok(());
            }

            Operation::UConst(constant) => {
                let const_id = self.lower_constant(constant)?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), const_id);
                return Ok(());
            }

            Operation::IConst(constant) => {
                let const_id = self.lower_constant(constant)?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), const_id);
                return Ok(());
            }

            Operation::FConst(constant) => {
                let const_id = self.lower_constant(constant)?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), const_id);
                return Ok(());
            }

            // Bitwise operations
            Operation::And(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                // TODO: Get result type properly
                let result_id = self.builder.bitwise_and(self.u32_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            Operation::Or(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let result_id = self.builder.bitwise_or(self.u32_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            Operation::Shl(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let result_id = self.builder.shift_left_logical(self.u32_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            Operation::Shr(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let result_id = self.builder.shift_right_logical(self.u32_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            // Arithmetic operations
            Operation::Add(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let result_id = self.builder.i_add(self.u32_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            Operation::Sub(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let result_id = self.builder.i_sub(self.u32_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            Operation::Rem(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let result_id = self.builder.u_mod(self.u32_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            // Comparisons
            Operation::UCmpLt(lhs, rhs) => {
                let lhs_id = self.get_value(lhs)?;
                let rhs_id = self.get_value(rhs)?;
                let bool_type = self.builder.type_bool();
                let result_id = self.builder.u_less_than(bool_type, None, lhs_id, rhs_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            // Select
            Operation::Select(cond, true_val, false_val) => {
                let cond_id = self.get_value(cond)?;
                let true_id = self.get_value(true_val)?;
                let false_id = self.get_value(false_val)?;
                let result_id = self.builder.select(self.u32_type, None, cond_id, true_id, false_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            // Type conversions
            Operation::Bitcast(val) => {
                let val_id = self.get_value(val)?;
                // TODO: Get proper result type
                let result_id = self.builder.bitcast(self.u32_type, None, val_id)
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            // Memory operations
            Operation::Load(ptr) => {
                let ptr_id = self.get_value(ptr)?;
                // TODO: Get proper result type
                let result_id = self.builder.load(self.u32_type, None, ptr_id, None, [])
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            Operation::Store(ptr, value) => {
                let ptr_id = self.get_value(ptr)?;
                let value_id = self.get_value(value)?;
                self.builder.store(ptr_id, value_id, None, [])
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                return Ok(());
            }

            Operation::Gep { base, index, stride } => {
                let base_id = self.get_value(base)?;
                let index_id = self.get_value(index)?;
                // SPIR-V AccessChain: access an element of a composite
                // For now, simple pointer arithmetic
                // TODO: Implement proper GEP
                let result_id = self.builder.access_chain(self.u32_type, None, base_id, [index_id])
                    .map_err(|e| format!("SPIR-V error: {:?}", e))?;
                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
                return Ok(());
            }

            // Atomic operations
            Operation::AtomicRmw { op, ptr, value, ordering, scope } => {
                use gasm::AtomicOp;
                let ptr_id = self.get_value(ptr)?;
                let value_id = self.get_value(value)?;
                let scope_id = self.lower_memory_scope(scope);
                let ordering_id = self.lower_memory_ordering(ordering);

                let result_id = match op {
                    AtomicOp::Add => {
                        self.builder.atomic_i_add(
                            self.u32_type,
                            None,
                            ptr_id,
                            scope_id,
                            ordering_id,
                            value_id,
                        ).map_err(|e| format!("SPIR-V error: {:?}", e))?
                    }
                    _ => return Err(format!("Atomic operation {:?} not yet implemented", op)),
                };

                self.registers.insert(inst.result.as_ref().unwrap().clone(), result_id);
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
    fn get_value(&self, value: &Value) -> Result<Word, String> {
        match value {
            Value::Register(name) => self.registers
                .get(name)
                .copied()
                .ok_or_else(|| format!("Register {} not found", name)),
            Value::Global(name) => self.globals
                .get(name)
                .copied()
                .ok_or_else(|| format!("Global {} not found", name)),
            Value::Constant(_) => {
                Err("Constants should be lowered separately".to_string())
            }
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
                let label_id = *self.labels.get(label)
                    .ok_or_else(|| format!("Label {} not found", label))?;
                self.builder.branch(label_id).unwrap();
            }

            Terminator::BrIf { cond, true_label, false_label } => {
                let cond_id = self.get_value(cond)?;
                let true_id = *self.labels.get(true_label)
                    .ok_or_else(|| format!("Label {} not found", true_label))?;
                let false_id = *self.labels.get(false_label)
                    .ok_or_else(|| format!("Label {} not found", false_label))?;
                self.builder.branch_conditional(cond_id, true_id, false_id, []).unwrap();
            }
        }

        Ok(())
    }
}

/// Lower a GASM module to SPIR-V bytecode
pub fn lower_gasm_module(module: &Module) -> Result<Vec<u32>, String> {
    let mut lowering = GasmLowering::new();
    lowering.lower_module(module)
}

/// Lower a single GASM function into an existing SPIR-V Builder
/// Returns the SPIR-V function ID
///
/// globals: Map from GASM global names (like "gdp_buffer", without @ prefix) to SPIR-V variable IDs
pub fn lower_function_into_builder(
    builder: &mut Builder,
    function: &Function,
    globals: HashMap<String, Word>,
) -> Result<Word, String> {
    // Create a minimal GasmLowering context for this function
    let u32_type = builder.type_int(32, 0);

    let mut lowering = GasmLowering {
        builder: std::mem::replace(builder, Builder::new()), // Temporarily take ownership
        u32_type,
        registers: HashMap::new(),
        globals, // Use provided globals map
        functions: HashMap::new(),
        labels: HashMap::new(),
        type_cache: HashMap::new(),
    };

    // First, declare the function to get its ID
    let fn_id = lowering.builder.id();
    lowering.functions.insert(function.name.clone(), fn_id);

    // Now lower the function (this will reference the ID we just created)
    lowering.lower_function(function)?;

    // Return the builder
    *builder = lowering.builder;

    Ok(fn_id)
}
