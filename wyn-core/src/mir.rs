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

/// SSA register ID
pub type RegisterId = u32;

/// Function ID
pub type FunctionId = u32;

/// Block ID for control flow
pub type BlockId = u32;

/// MIR instruction - SSA-based operations
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // Constants
    ConstInt(RegisterId, i32),
    ConstFloat(RegisterId, f32),
    ConstBool(RegisterId, bool),

    // Binary operations
    Add(RegisterId, RegisterId, RegisterId), // dest = left + right
    Sub(RegisterId, RegisterId, RegisterId),
    Mul(RegisterId, RegisterId, RegisterId),
    Div(RegisterId, RegisterId, RegisterId),

    // Comparisons
    Eq(RegisterId, RegisterId, RegisterId),
    Ne(RegisterId, RegisterId, RegisterId),
    Lt(RegisterId, RegisterId, RegisterId),
    Le(RegisterId, RegisterId, RegisterId),
    Gt(RegisterId, RegisterId, RegisterId),
    Ge(RegisterId, RegisterId, RegisterId),

    // Memory operations (for mutable variables)
    // Alloca creates a pointer in the entry block
    Alloca(RegisterId, Type<TypeName>), // dest_ptr, pointee_type
    Load(RegisterId, RegisterId),       // dest, src_ptr
    Store(RegisterId, RegisterId),      // dest_ptr, src_value

    // Function calls
    Call(RegisterId, FunctionId, Vec<RegisterId>), // dest, func_id, args
    CallBuiltin(RegisterId, String, Vec<RegisterId>), // dest, builtin_name, args

    // Tuple operations
    MakeTuple(RegisterId, Vec<RegisterId>),      // dest, elements
    ExtractElement(RegisterId, RegisterId, u32), // dest, tuple, index

    // Array operations
    MakeArray(RegisterId, Vec<RegisterId>),         // dest, elements
    ArrayIndex(RegisterId, RegisterId, RegisterId), // dest, array, index

    // Control flow
    Branch(BlockId),                             // unconditional jump
    BranchCond(RegisterId, BlockId, BlockId),    // cond, true_block, false_block
    Phi(RegisterId, Vec<(RegisterId, BlockId)>), // dest, vec of (value, predecessor_block)
    Return(RegisterId),                          // return value
}

/// Basic block in control flow graph
#[derive(Debug, Clone)]
pub struct Block {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
}

/// MIR function
#[derive(Debug, Clone)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<(RegisterId, Type<TypeName>)>,
    pub return_type: Type<TypeName>,
    pub blocks: Vec<Block>,
    pub entry_block: BlockId,
}

/// MIR module - the complete program representation
#[derive(Debug, Clone)]
pub struct Module {
    pub functions: Vec<Function>,
    pub entry_points: Vec<FunctionId>,
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
    next_register: RegisterId,

    /// Next function ID to allocate
    next_function_id: FunctionId,

    /// Next block ID to allocate
    next_block_id: BlockId,

    /// Map from function names to function IDs
    function_map: HashMap<String, FunctionId>,

    /// Type deduplication - maps types to their canonical register holding that type
    type_cache: HashMap<Type<TypeName>, RegisterId>,

    /// Integer constant deduplication
    int_const_cache: HashMap<i32, RegisterId>,

    /// Float constant deduplication (using bits as key since f32 doesn't impl Eq/Hash)
    float_const_cache: HashMap<u32, RegisterId>,

    /// Pending allocas to be inserted in entry block
    pending_allocas: Vec<Instruction>,
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
            type_cache: HashMap::new(),
            int_const_cache: HashMap::new(),
            float_const_cache: HashMap::new(),
            pending_allocas: Vec::new(),
        }
    }

    /// Allocate a new SSA register
    pub fn new_register(&mut self) -> RegisterId {
        let id = self.next_register;
        self.next_register += 1;
        id
    }

    /// Allocate a new block ID
    pub fn new_block_id(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;
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

        // Create parameter registers
        let param_regs: Vec<(RegisterId, Type<TypeName>)> =
            params.into_iter().map(|(_, ty)| (self.new_register(), ty)).collect();

        // Create entry block
        let entry_block = self.new_block_id();

        let function = Function {
            id: func_id,
            name: name.clone(),
            params: param_regs,
            return_type,
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
    pub fn get_param(&self, index: usize) -> Option<RegisterId> {
        self.current_function
            .and_then(|func_id| self.functions[func_id as usize].params.get(index).map(|(reg, _)| *reg))
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

    // === Constant builders with deduplication ===

    pub fn build_const_int(&mut self, value: i32) -> RegisterId {
        if let Some(&reg) = self.int_const_cache.get(&value) {
            return reg;
        }

        let reg = self.new_register();
        self.emit(Instruction::ConstInt(reg, value));
        self.int_const_cache.insert(value, reg);
        reg
    }

    pub fn build_const_float(&mut self, value: f32) -> RegisterId {
        // Note: Using bits for HashMap key since f32 doesn't implement Eq/Hash
        let bits = value.to_bits();
        if let Some(&reg) = self.float_const_cache.get(&bits) {
            return reg;
        }

        let reg = self.new_register();
        self.emit(Instruction::ConstFloat(reg, value));
        self.float_const_cache.insert(bits, reg);
        reg
    }

    pub fn build_const_bool(&mut self, value: bool) -> RegisterId {
        let reg = self.new_register();
        self.emit(Instruction::ConstBool(reg, value));
        reg
    }

    // === Binary operations ===

    pub fn build_add(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Add(dest, left, right));
        dest
    }

    pub fn build_sub(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Sub(dest, left, right));
        dest
    }

    pub fn build_mul(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Mul(dest, left, right));
        dest
    }

    pub fn build_div(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Div(dest, left, right));
        dest
    }

    // === Comparison operations ===

    pub fn build_eq(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Eq(dest, left, right));
        dest
    }

    pub fn build_ne(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Ne(dest, left, right));
        dest
    }

    pub fn build_lt(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Lt(dest, left, right));
        dest
    }

    pub fn build_le(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Le(dest, left, right));
        dest
    }

    pub fn build_gt(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Gt(dest, left, right));
        dest
    }

    pub fn build_ge(&mut self, left: RegisterId, right: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Ge(dest, left, right));
        dest
    }

    // === Memory operations ===

    /// Allocate space for a mutable variable (will be placed in entry block)
    pub fn build_alloca(&mut self, pointee_type: Type<TypeName>) -> RegisterId {
        let ptr_reg = self.new_register();
        self.pending_allocas.push(Instruction::Alloca(ptr_reg, pointee_type));
        ptr_reg
    }

    pub fn build_load(&mut self, ptr: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Load(dest, ptr));
        dest
    }

    pub fn build_store(&mut self, ptr: RegisterId, value: RegisterId) {
        self.emit(Instruction::Store(ptr, value));
    }

    // === Function calls ===

    pub fn build_call(&mut self, func_name: &str, args: Vec<RegisterId>) -> RegisterId {
        let dest = self.new_register();

        if let Some(&func_id) = self.function_map.get(func_name) {
            self.emit(Instruction::Call(dest, func_id, args));
        } else {
            // Assume it's a builtin
            self.emit(Instruction::CallBuiltin(dest, func_name.to_string(), args));
        }

        dest
    }

    pub fn build_call_builtin(&mut self, builtin_name: &str, args: Vec<RegisterId>) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::CallBuiltin(dest, builtin_name.to_string(), args));
        dest
    }

    // === Tuple operations ===

    pub fn build_tuple(&mut self, elements: Vec<RegisterId>) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::MakeTuple(dest, elements));
        dest
    }

    pub fn build_extract_element(&mut self, tuple: RegisterId, index: u32) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::ExtractElement(dest, tuple, index));
        dest
    }

    // === Array operations ===

    pub fn build_array(&mut self, elements: Vec<RegisterId>) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::MakeArray(dest, elements));
        dest
    }

    pub fn build_array_index(&mut self, array: RegisterId, index: RegisterId) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::ArrayIndex(dest, array, index));
        dest
    }

    // === Control flow ===

    pub fn build_branch(&mut self, target: BlockId) {
        self.emit(Instruction::Branch(target));
    }

    pub fn build_branch_cond(&mut self, cond: RegisterId, true_block: BlockId, false_block: BlockId) {
        self.emit(Instruction::BranchCond(cond, true_block, false_block));
    }

    pub fn build_phi(&mut self, incoming: Vec<(RegisterId, BlockId)>) -> RegisterId {
        let dest = self.new_register();
        self.emit(Instruction::Phi(dest, incoming));
        dest
    }

    pub fn build_return(&mut self, value: RegisterId) {
        self.emit(Instruction::Return(value));
    }

    // === Module finalization ===

    pub fn finish(self, entry_points: Vec<FunctionId>) -> Module {
        Module {
            functions: self.functions,
            entry_points,
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
        builder.begin_function("test".to_string(), vec![], i32_type);

        // Create the same constant twice
        let c1 = builder.build_const_int(42);
        let c2 = builder.build_const_int(42);

        // Should return the same register
        assert_eq!(c1, c2);

        // Create a different constant
        let c3 = builder.build_const_int(100);
        assert_ne!(c1, c3);

        builder.end_function();
    }

    #[test]
    fn test_alloca_in_entry_block() {
        let mut builder = Builder::new();

        let i32_type = Type::Constructed(TypeName::Str("i32"), vec![]);
        builder.begin_function("test".to_string(), vec![], i32_type.clone());

        // Create some allocas
        let ptr1 = builder.build_alloca(i32_type.clone());
        let ptr2 = builder.build_alloca(i32_type.clone());

        // Store and load
        let value = builder.build_const_int(42);
        builder.build_store(ptr1, value);
        let loaded = builder.build_load(ptr1);
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
            i32_type,
        );

        let cond = builder.get_param(0).unwrap();

        // Create blocks
        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let merge_block = builder.create_block();

        // Entry: branch based on condition
        builder.build_branch_cond(cond, then_block, else_block);

        // Then block: return 1
        builder.select_block(then_block);
        let then_value = builder.build_const_int(1);
        builder.build_branch(merge_block);

        // Else block: return 2
        builder.select_block(else_block);
        let else_value = builder.build_const_int(2);
        builder.build_branch(merge_block);

        // Merge block: phi node
        builder.select_block(merge_block);
        let result = builder.build_phi(vec![(then_value, then_block), (else_value, else_block)]);
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
        let arg = builder.build_const_int(42);
        let result = builder.build_call("helper", vec![arg]);
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

        builder.begin_function("test".to_string(), vec![], f32_type);

        let x = builder.build_const_float(1.0);
        let y = builder.build_const_float(2.0);
        let v = builder.build_call_builtin("vec2", vec![x, y]);
        let len = builder.build_call_builtin("length", vec![v]);
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
        builder.begin_function("test".to_string(), vec![], i32_type.clone());

        let a = builder.build_const_int(1);
        let b = builder.build_const_int(2);
        let tuple = builder.build_tuple(vec![a, b]);
        let first = builder.build_extract_element(tuple, 0);
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
        builder.begin_function("test".to_string(), vec![], i32_type.clone());

        let e1 = builder.build_const_int(1);
        let e2 = builder.build_const_int(2);
        let e3 = builder.build_const_int(3);
        let array = builder.build_array(vec![e1, e2, e3]);
        let idx = builder.build_const_int(1);
        let element = builder.build_array_index(array, idx);
        builder.build_return(element);

        builder.end_function();

        let func = &builder.functions[0];
        let entry_block = &func.blocks[0];
        assert!(entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::MakeArray(..))));
        assert!(entry_block.instructions.iter().any(|inst| matches!(inst, Instruction::ArrayIndex(..))));
    }
}
