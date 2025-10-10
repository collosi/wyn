//! MIR (Mid-level Intermediate Representation)
//!
//! This module provides a typed, SSA-based intermediate representation that sits
//! between the Wyn AST and SPIR-V. It handles complexities like:
//! - Variable declarations in entry blocks
//! - Type and constant deduplication
//! - Lowering complex types to basic types and pointers
//! - Polymorphic builtin resolution

use crate::ast::TypeName;
use polytype::Type;
use std::collections::HashMap;

/// SSA register with type information
#[derive(Debug, Clone, PartialEq)]
pub struct Register {
    pub id: u32,
    pub ty: Type<TypeName>,
}

/// Function ID
pub type FunctionId = u32;

/// Block ID for control flow
pub type BlockId = u32;

/// MIR instruction - SSA-based operations with full type information
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // Constants
    ConstInt(Register, i32),
    ConstFloat(Register, f32),
    ConstBool(Register, bool),

    // Binary operations
    Add(Register, Register, Register), // dest = left + right
    Sub(Register, Register, Register),
    Mul(Register, Register, Register),
    Div(Register, Register, Register),

    // Comparisons
    Eq(Register, Register, Register),
    Ne(Register, Register, Register),
    Lt(Register, Register, Register),
    Le(Register, Register, Register),
    Gt(Register, Register, Register),
    Ge(Register, Register, Register),

    // Memory operations (for mutable variables)
    // Alloca creates a pointer in the entry block
    Alloca(Register),          // dest_ptr (type includes pointee info)
    Load(Register, Register),  // dest, src_ptr
    Store(Register, Register), // dest_ptr, src_value

    // Storage buffer operations
    // Access to a global storage buffer at a specific offset
    BufferStore(u32, Register), // offset_in_bytes, value
    BufferLoad(Register, u32),  // dest, offset_in_bytes

    // Function calls
    Call(Register, FunctionId, Vec<Register>), // dest, func_id, args
    CallBuiltin(Register, String, Vec<Register>), // dest, builtin_name, args

    // Tuple operations
    MakeTuple(Register, Vec<Register>),      // dest, elements
    ExtractElement(Register, Register, u32), // dest, tuple, index

    // Array operations
    MakeArray(Register, Vec<Register>),       // dest, elements
    ArrayIndex(Register, Register, Register), // dest, array, index

    // Control flow
    Branch(BlockId),                                 // unconditional jump
    BranchCond(Register, BlockId, BlockId, BlockId), // cond, true_block, false_block, merge_block
    Phi(Register, Vec<(Register, BlockId)>),         // dest, vec of (value, predecessor_block)
    Return(Register),                                // return value
    ReturnVoid,                                      // return from void function
}

/// Basic block in control flow graph
#[derive(Debug, Clone)]
pub struct Block {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
}

/// Parameter with attributes (for entry points)
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub register: Register,
    pub attributes: Vec<crate::ast::Attribute>,
}

/// MIR function
#[derive(Debug, Clone)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<Register>,
    pub param_attributes: Vec<Vec<crate::ast::Attribute>>, // Attributes for each parameter
    pub return_type: Type<TypeName>,
    pub return_attributes: Vec<crate::ast::Attribute>, // Attributes on the return type
    pub attributed_return_types: Option<Vec<crate::ast::AttributedType>>, // For multiple outputs
    pub blocks: Vec<Block>,
    pub entry_block: BlockId,
}

/// MIR module - the complete program representation
#[derive(Debug, Clone)]
pub struct Module {
    pub functions: Vec<Function>,
    pub entry_points: Vec<FunctionId>,
    /// Size of the constants storage buffer in bytes (0 if no constants)
    pub constants_buffer_size: u32,
}

/// Builder for constructing MIR while visiting the AST
pub struct Builder {
    /// Current function being built
    current_function: Option<FunctionId>,

    /// All functions in the module
    functions: Vec<Function>,

    /// Current block being built
    current_block: Option<BlockId>,

    /// Next register ID to allocate
    next_register: u32,

    /// Next function ID to allocate
    next_function_id: FunctionId,

    /// Next block ID to allocate
    next_block_id: BlockId,

    /// Map from function names to function IDs
    function_map: HashMap<String, FunctionId>,

    /// Integer constant deduplication (value, type) -> Register
    int_const_cache: HashMap<(i32, Type<TypeName>), Register>,

    /// Float constant deduplication (bits, type) -> Register
    float_const_cache: HashMap<(u32, Type<TypeName>), Register>,

    /// Pending allocas to be inserted in entry block
    pending_allocas: Vec<Instruction>,

    /// Constants buffer management
    /// Maps constant name to (type, offset)
    constants: HashMap<String, (Type<TypeName>, u32)>,
    next_constant_offset: u32,
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            current_function: None,
            functions: Vec::new(),
            current_block: None,
            next_register: 0,
            next_function_id: 0,
            next_block_id: 0,
            function_map: HashMap::new(),
            int_const_cache: HashMap::new(),
            float_const_cache: HashMap::new(),
            pending_allocas: Vec::new(),
            constants: HashMap::new(),
            next_constant_offset: 0,
        }
    }

    /// Allocate a new SSA register with given type
    pub fn new_register(&mut self, ty: Type<TypeName>) -> Register {
        let id = self.next_register;
        self.next_register += 1;
        Register { id, ty }
    }

    /// Allocate a new block ID (without creating the block yet)
    pub fn new_block_id(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;
        id
    }

    /// Create a block with a previously allocated ID
    pub fn create_block_with_id(&mut self, id: BlockId) -> BlockId {
        let block = Block {
            id,
            instructions: Vec::new(),
        };
        if let Some(func_id) = self.current_function {
            self.functions[func_id as usize].blocks.push(block);
        }
        id
    }

    /// Start building a new function
    pub fn begin_function(
        &mut self,
        name: String,
        params: Vec<(String, Type<TypeName>)>,
        return_type: Type<TypeName>,
    ) -> FunctionId {
        let func_id = self.next_function_id;
        self.next_function_id += 1;

        // Clear constant caches for new function (constants don't cross function boundaries)
        self.int_const_cache.clear();
        self.float_const_cache.clear();

        // Create parameter registers
        let param_regs: Vec<Register> = params.into_iter().map(|(_, ty)| self.new_register(ty)).collect();

        // Create entry block
        let entry_block = self.new_block_id();

        let function = Function {
            id: func_id,
            name: name.clone(),
            params: param_regs,
            param_attributes: Vec::new(), // Will be filled in by mirize
            return_type,
            return_attributes: Vec::new(), // Will be filled in by mirize
            attributed_return_types: None, // Will be filled in by mirize
            blocks: vec![Block {
                id: entry_block,
                instructions: Vec::new(),
            }],
            entry_block,
        };

        self.functions.push(function);
        self.function_map.insert(name, func_id);
        self.current_function = Some(func_id);
        self.current_block = Some(entry_block);
        self.pending_allocas.clear();

        func_id
    }

    /// Finish building the current function and insert pending allocas
    /// Set attributes for the current function (must be called before end_function)
    pub fn set_function_attributes(
        &mut self,
        param_attributes: Vec<Vec<crate::ast::Attribute>>,
        return_attributes: Vec<crate::ast::Attribute>,
        attributed_return_types: Option<Vec<crate::ast::AttributedType>>,
    ) {
        if let Some(func_id) = self.current_function {
            let func = &mut self.functions[func_id as usize];
            func.param_attributes = param_attributes;
            func.return_attributes = return_attributes;
            func.attributed_return_types = attributed_return_types;
        }
    }

    pub fn end_function(&mut self) {
        if let Some(func_id) = self.current_function {
            let func = &mut self.functions[func_id as usize];

            // Insert pending allocas at the beginning of entry block
            if !self.pending_allocas.is_empty() {
                let entry_block = &mut func.blocks[0];
                let mut new_instructions = std::mem::take(&mut self.pending_allocas);
                new_instructions.append(&mut entry_block.instructions);
                entry_block.instructions = new_instructions;
            }
        }

        self.current_function = None;
        self.current_block = None;
    }

    /// Get parameter register by index
    pub fn get_param(&self, index: usize) -> Option<Register> {
        self.current_function
            .and_then(|func_id| self.functions[func_id as usize].params.get(index).cloned())
    }

    /// Emit an instruction to the current block
    fn emit(&mut self, inst: Instruction) {
        if let (Some(func_id), Some(block_id)) = (self.current_function, self.current_block) {
            let func = &mut self.functions[func_id as usize];
            let block = func.blocks.iter_mut().find(|b| b.id == block_id).expect("Current block not found");
            block.instructions.push(inst);
        }
    }

    /// Create a new block in the current function
    pub fn create_block(&mut self) -> BlockId {
        let block_id = self.new_block_id();

        if let Some(func_id) = self.current_function {
            let func = &mut self.functions[func_id as usize];
            func.blocks.push(Block {
                id: block_id,
                instructions: Vec::new(),
            });
        }

        block_id
    }

    /// Switch to building in a different block
    pub fn select_block(&mut self, block_id: BlockId) {
        self.current_block = Some(block_id);
    }

    /// Get the currently selected block ID
    pub fn current_block(&self) -> Option<BlockId> {
        self.current_block
    }

    // === Constant builders with deduplication ===

    pub fn build_const_int(&mut self, value: i32, ty: Type<TypeName>) -> Register {
        let key = (value, ty.clone());
        if let Some(reg) = self.int_const_cache.get(&key) {
            return reg.clone();
        }

        let reg = self.new_register(ty.clone());
        self.emit(Instruction::ConstInt(reg.clone(), value));
        self.int_const_cache.insert(key, reg.clone());
        reg
    }

    pub fn build_const_float(&mut self, value: f32, ty: Type<TypeName>) -> Register {
        // Note: Using bits for HashMap key since f32 doesn't implement Eq/Hash
        let bits = value.to_bits();
        let key = (bits, ty.clone());
        if let Some(reg) = self.float_const_cache.get(&key) {
            return reg.clone();
        }

        let reg = self.new_register(ty.clone());
        self.emit(Instruction::ConstFloat(reg.clone(), value));
        self.float_const_cache.insert(key, reg.clone());
        reg
    }

    pub fn build_const_bool(&mut self, value: bool, ty: Type<TypeName>) -> Register {
        let reg = self.new_register(ty);
        self.emit(Instruction::ConstBool(reg.clone(), value));
        reg
    }

    // === Binary operations ===

    pub fn build_add(&mut self, left: Register, right: Register) -> Register {
        let dest = self.new_register(left.ty.clone());
        self.emit(Instruction::Add(dest.clone(), left, right));
        dest
    }

    pub fn build_sub(&mut self, left: Register, right: Register) -> Register {
        let dest = self.new_register(left.ty.clone());
        self.emit(Instruction::Sub(dest.clone(), left, right));
        dest
    }

    pub fn build_mul(&mut self, left: Register, right: Register) -> Register {
        let dest = self.new_register(left.ty.clone());
        self.emit(Instruction::Mul(dest.clone(), left, right));
        dest
    }

    pub fn build_div(&mut self, left: Register, right: Register) -> Register {
        let dest = self.new_register(left.ty.clone());
        self.emit(Instruction::Div(dest.clone(), left, right));
        dest
    }

    // === Comparison operations ===

    pub fn build_eq(&mut self, left: Register, right: Register) -> Register {
        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
        let dest = self.new_register(bool_type);
        self.emit(Instruction::Eq(dest.clone(), left, right));
        dest
    }

    pub fn build_ne(&mut self, left: Register, right: Register) -> Register {
        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
        let dest = self.new_register(bool_type);
        self.emit(Instruction::Ne(dest.clone(), left, right));
        dest
    }

    pub fn build_lt(&mut self, left: Register, right: Register) -> Register {
        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
        let dest = self.new_register(bool_type);
        self.emit(Instruction::Lt(dest.clone(), left, right));
        dest
    }

    pub fn build_le(&mut self, left: Register, right: Register) -> Register {
        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
        let dest = self.new_register(bool_type);
        self.emit(Instruction::Le(dest.clone(), left, right));
        dest
    }

    pub fn build_gt(&mut self, left: Register, right: Register) -> Register {
        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
        let dest = self.new_register(bool_type);
        self.emit(Instruction::Gt(dest.clone(), left, right));
        dest
    }

    pub fn build_ge(&mut self, left: Register, right: Register) -> Register {
        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
        let dest = self.new_register(bool_type);
        self.emit(Instruction::Ge(dest.clone(), left, right));
        dest
    }

    // === Memory operations ===

    /// Allocate space for a mutable variable (will be placed in entry block)
    /// ptr_type should be a pointer type (the Register's type includes pointee info)
    pub fn build_alloca(&mut self, ptr_type: Type<TypeName>) -> Register {
        let ptr_reg = self.new_register(ptr_type);
        self.pending_allocas.push(Instruction::Alloca(ptr_reg.clone()));
        ptr_reg
    }

    pub fn build_load(&mut self, ptr: Register, pointee_type: Type<TypeName>) -> Register {
        let dest = self.new_register(pointee_type);
        self.emit(Instruction::Load(dest.clone(), ptr));
        dest
    }

    pub fn build_store(&mut self, ptr: Register, value: Register) {
        self.emit(Instruction::Store(ptr, value));
    }

    // === Function calls ===

    pub fn build_call(
        &mut self,
        func_name: &str,
        args: Vec<Register>,
        result_type: Type<TypeName>,
    ) -> Register {
        let dest = self.new_register(result_type);

        if let Some(&func_id) = self.function_map.get(func_name) {
            self.emit(Instruction::Call(dest.clone(), func_id, args));
        } else {
            // Assume it's a builtin
            self.emit(Instruction::CallBuiltin(
                dest.clone(),
                func_name.to_string(),
                args,
            ));
        }

        dest
    }

    pub fn build_call_builtin(
        &mut self,
        builtin_name: &str,
        args: Vec<Register>,
        result_type: Type<TypeName>,
    ) -> Register {
        let dest = self.new_register(result_type);
        self.emit(Instruction::CallBuiltin(
            dest.clone(),
            builtin_name.to_string(),
            args,
        ));
        dest
    }

    // === Tuple operations ===

    pub fn build_tuple(&mut self, elements: Vec<Register>, tuple_type: Type<TypeName>) -> Register {
        let dest = self.new_register(tuple_type);
        self.emit(Instruction::MakeTuple(dest.clone(), elements));
        dest
    }

    pub fn build_extract_element(
        &mut self,
        tuple: Register,
        index: u32,
        element_type: Type<TypeName>,
    ) -> Register {
        let dest = self.new_register(element_type);
        self.emit(Instruction::ExtractElement(dest.clone(), tuple, index));
        dest
    }

    // === Array operations ===

    pub fn build_array(&mut self, elements: Vec<Register>, array_type: Type<TypeName>) -> Register {
        let dest = self.new_register(array_type);
        self.emit(Instruction::MakeArray(dest.clone(), elements));
        dest
    }

    pub fn build_array_index(
        &mut self,
        array: Register,
        index: Register,
        element_type: Type<TypeName>,
    ) -> Register {
        let dest = self.new_register(element_type);
        self.emit(Instruction::ArrayIndex(dest.clone(), array, index));
        dest
    }

    // === Control flow ===

    pub fn build_branch(&mut self, target: BlockId) {
        self.emit(Instruction::Branch(target));
    }

    pub fn build_branch_cond(
        &mut self,
        cond: Register,
        true_block: BlockId,
        false_block: BlockId,
        merge_block: BlockId,
    ) {
        self.emit(Instruction::BranchCond(
            cond,
            true_block,
            false_block,
            merge_block,
        ));
    }

    pub fn build_phi(
        &mut self,
        incoming: Vec<(Register, BlockId)>,
        result_type: Type<TypeName>,
    ) -> Register {
        let dest = self.new_register(result_type);
        self.emit(Instruction::Phi(dest.clone(), incoming));
        dest
    }

    pub fn build_return(&mut self, value: Register) {
        self.emit(Instruction::Return(value));
    }

    pub fn build_return_void(&mut self) {
        self.emit(Instruction::ReturnVoid);
    }

    // === Storage buffer operations ===

    /// Register a top-level constant, allocating space in the constants buffer
    /// Returns the offset where this constant will be stored
    pub fn register_constant(&mut self, name: String, ty: Type<TypeName>) -> u32 {
        // Calculate size for this type
        let size = Self::size_of_type(&ty);
        let offset = self.next_constant_offset;

        self.constants.insert(name, (ty, offset));
        self.next_constant_offset += size;

        offset
    }

    /// Get the offset of a previously registered constant
    pub fn get_constant_offset(&self, name: &str) -> Option<u32> {
        self.constants.get(name).map(|(_, offset)| *offset)
    }

    /// Store a value to the constants buffer at a specific offset
    pub fn build_buffer_store(&mut self, offset: u32, value: Register) {
        self.emit(Instruction::BufferStore(offset, value));
    }

    /// Load a value from the constants buffer at a specific offset
    pub fn build_buffer_load(&mut self, offset: u32, value_type: Type<TypeName>) -> Register {
        let dest = self.new_register(value_type);
        self.emit(Instruction::BufferLoad(dest.clone(), offset));
        dest
    }

    /// Calculate size in bytes for a type (for buffer layout)
    fn size_of_type(ty: &Type<TypeName>) -> u32 {
        use crate::ast::TypeName;
        match ty {
            Type::Constructed(TypeName::Str(name), args) => match *name {
                "i32" | "f32" | "bool" => 4,
                "vec2" => 8,
                "vec3" => 12, // Note: vec3 has padding, often treated as 16 bytes in std140
                "vec4" => 16,
                "tuple" => {
                    // Sum of component sizes (simplified, doesn't handle alignment)
                    args.iter().map(Self::size_of_type).sum()
                }
                _ => 4, // Default to 4 bytes
            },
            _ => 4,
        }
    }

    // === Module finalization ===

    pub fn finish(self, entry_points: Vec<FunctionId>) -> Module {
        Module {
            functions: self.functions,
            entry_points,
            constants_buffer_size: self.next_constant_offset,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeName;

    #[test]
    fn test_basic_function() {
        let mut builder = Builder::new();

        // Create function: def add(x: i32, y: i32) -> i32
        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);
        let func_id = builder.begin_function(
            "add".to_string(),
            vec![
                ("x".to_string(), i32_type.clone()),
                ("y".to_string(), i32_type.clone()),
            ],
            i32_type,
        );

        // Get parameters
        let x = builder.get_param(0).unwrap();
        let y = builder.get_param(1).unwrap();

        // result = x + y
        let result = builder.build_add(x, y);

        // return result
        builder.build_return(result);
        builder.end_function();

        assert_eq!(builder.functions.len(), 1);
        assert_eq!(builder.functions[0].id, func_id);
        assert_eq!(builder.functions[0].name, "add");
        assert_eq!(builder.functions[0].params.len(), 2);
    }

    #[test]
    fn test_constant_deduplication() {
        let mut builder = Builder::new();

        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);
        builder.begin_function("test".to_string(), vec![], i32_type.clone());

        // Create the same constant twice
        let c1 = builder.build_const_int(42, i32_type.clone());
        let c2 = builder.build_const_int(42, i32_type.clone());

        // Should return the same register
        assert_eq!(c1, c2);

        // Create a different constant
        let c3 = builder.build_const_int(100, i32_type);
        assert_ne!(c1, c3);

        builder.end_function();
    }

    #[test]
    fn test_alloca_in_entry_block() {
        let mut builder = Builder::new();

        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);
        let ptr_type = Type::Constructed(TypeName::Str("ptr"), vec![i32_type.clone()]);

        builder.begin_function("test".to_string(), vec![], i32_type.clone());

        // Create some allocas
        let ptr1 = builder.build_alloca(ptr_type.clone());
        let ptr2 = builder.build_alloca(ptr_type.clone());

        // Store and load
        let value = builder.build_const_int(42, i32_type.clone());
        builder.build_store(ptr1.clone(), value);
        let loaded = builder.build_load(ptr1.clone(), i32_type);
        builder.build_return(loaded);

        builder.end_function();

        // Check that allocas are at the beginning of entry block
        let func = &builder.functions[0];
        let entry_block = &func.blocks[0];
        assert!(matches!(entry_block.instructions[0], Instruction::Alloca(..)));
        assert!(matches!(entry_block.instructions[1], Instruction::Alloca(..)));

        // Allocas should have different registers
        assert_ne!(ptr1, ptr2);
    }

    #[test]
    fn test_control_flow() {
        let mut builder = Builder::new();

        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);

        builder.begin_function(
            "test_if".to_string(),
            vec![("cond".to_string(), bool_type)],
            i32_type.clone(),
        );

        let cond = builder.get_param(0).unwrap();

        // Create blocks
        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let merge_block = builder.create_block();

        // Entry: branch based on condition
        builder.build_branch_cond(cond, then_block, else_block, merge_block);

        // Then block: return 1
        builder.select_block(then_block);
        let then_value = builder.build_const_int(1, i32_type.clone());
        builder.build_branch(merge_block);

        // Else block: return 2
        builder.select_block(else_block);
        let else_value = builder.build_const_int(2, i32_type.clone());
        builder.build_branch(merge_block);

        // Merge block: phi node
        builder.select_block(merge_block);
        let result = builder.build_phi(vec![(then_value, then_block), (else_value, else_block)], i32_type);
        builder.build_return(result);

        builder.end_function();

        let func = &builder.functions[0];
        assert_eq!(func.blocks.len(), 4); // entry + then + else + merge
    }

    #[test]
    fn test_function_calls() {
        let mut builder = Builder::new();

        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);

        // Define helper function
        let helper_id = builder.begin_function(
            "helper".to_string(),
            vec![("x".to_string(), i32_type.clone())],
            i32_type.clone(),
        );
        let x = builder.get_param(0).unwrap();
        builder.build_return(x);
        builder.end_function();

        // Define main function that calls helper
        builder.begin_function("main".to_string(), vec![], i32_type.clone());
        let arg = builder.build_const_int(42, i32_type.clone());
        let result = builder.build_call("helper", vec![arg], i32_type);
        builder.build_return(result);
        builder.end_function();

        assert_eq!(builder.functions.len(), 2);

        // Check that call references the right function
        let main_func = &builder.functions[1];
        let entry_block = &main_func.blocks[0];
        assert!(
            entry_block
                .instructions
                .iter()
                .any(|inst| matches!(inst, Instruction::Call(_, fid, _) if *fid == helper_id))
        );
    }

    #[test]
    fn test_builtin_calls() {
        let mut builder = Builder::new();

        let f32_type = Type::Constructed(TypeName::Str("f32"), vec![]);
        let vec2_type = Type::Constructed(TypeName::Str("vec2"), vec![]);

        builder.begin_function("test".to_string(), vec![], f32_type.clone());

        let x = builder.build_const_float(1.0, f32_type.clone());
        let y = builder.build_const_float(2.0, f32_type.clone());
        let v = builder.build_call_builtin("vec2", vec![x, y], vec2_type);
        let len = builder.build_call_builtin("length", vec![v], f32_type);
        builder.build_return(len);

        builder.end_function();

        let func = &builder.functions[0];
        let entry_block = &func.blocks[0];

        // Check for builtin calls
        let builtin_calls: Vec<_> = entry_block
            .instructions
            .iter()
            .filter_map(|inst| {
                if let Instruction::CallBuiltin(_, name, _) = inst { Some(name.as_str()) } else { None }
            })
            .collect();

        assert_eq!(builtin_calls, vec!["vec2", "length"]);
    }

    #[test]
    fn test_tuple_operations() {
        let mut builder = Builder::new();

        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);
        let tuple_type =
            Type::Constructed(TypeName::Str("tuple"), vec![i32_type.clone(), i32_type.clone()]);

        builder.begin_function("test".to_string(), vec![], i32_type.clone());

        let a = builder.build_const_int(1, i32_type.clone());
        let b = builder.build_const_int(2, i32_type.clone());
        let tuple = builder.build_tuple(vec![a, b], tuple_type);
        let first = builder.build_extract_element(tuple, 0, i32_type.clone());
        builder.build_return(first);

        builder.end_function();

        let func = &builder.functions[0];
        let entry_block = &func.blocks[0];
        assert!(entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::MakeTuple(..))));
        assert!(
            entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::ExtractElement(..)))
        );
    }

    #[test]
    fn test_array_operations() {
        let mut builder = Builder::new();

        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);
        let array_type = Type::Constructed(TypeName::Str("array"), vec![i32_type.clone()]);

        builder.begin_function("test".to_string(), vec![], i32_type.clone());

        let e1 = builder.build_const_int(1, i32_type.clone());
        let e2 = builder.build_const_int(2, i32_type.clone());
        let e3 = builder.build_const_int(3, i32_type.clone());
        let array = builder.build_array(vec![e1, e2, e3], array_type);
        let idx = builder.build_const_int(1, i32_type.clone());
        let element = builder.build_array_index(array, idx, i32_type.clone());
        builder.build_return(element);

        builder.end_function();

        let func = &builder.functions[0];
        let entry_block = &func.blocks[0];
        assert!(entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::MakeArray(..))));
        assert!(entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::ArrayIndex(..))));
    }
}
