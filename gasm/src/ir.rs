/// GIR (GPU IR) - Intermediate Representation structures

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub globals: Vec<Global>,
    pub functions: Vec<Function>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub name: String,
    pub ty: PointerType,
    pub initializer: Option<Initializer>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Initializer {
    Addr(u64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub attributes: Vec<FunctionAttr>,
    pub params: Vec<Param>,
    pub return_type: ReturnType,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionAttr {
    Kernel,
    Inline,
    NoInline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReturnType {
    Void,
    Type(Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub label: String,
    /// If Some, this block is a loop header with (merge_label, continue_label)
    pub loop_header: Option<(String, String)>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

/// Types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    Pointer(Box<PointerType>),
    Array(Box<Type>, u32),   // Fixed-size array [N; T], for Private/Local storage
    RuntimeArray(Box<Type>), // Unsized runtime array [*T], only for StorageBuffer (global)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub address_space: AddressSpace,
    pub pointee: Box<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    Generic,
    Global,  // StorageBuffer
    Shared,  // Workgroup
    Local,   // Function
    Private, // Private
    Const,   // UniformConstant
}

/// Values (SSA)
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Register(String),
    Global(String),
    Constant(Constant),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(u16), // Store as bits
    F32(f32),
    F64(f64),
}

/// Instructions
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub result: Option<String>,
    pub op: Operation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    // Arithmetic
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),
    Rem(Value, Value),
    Neg(Value),

    // Bitwise
    And(Value, Value),
    Or(Value, Value),
    Xor(Value, Value),
    Not(Value),
    Shl(Value, Value),
    Shr(Value, Value),

    // Comparisons
    ICmpEq(Value, Value),
    ICmpNe(Value, Value),
    ICmpLt(Value, Value),
    ICmpLe(Value, Value),
    ICmpGt(Value, Value),
    ICmpGe(Value, Value),

    UCmpEq(Value, Value),
    UCmpNe(Value, Value),
    UCmpLt(Value, Value),
    UCmpLe(Value, Value),
    UCmpGt(Value, Value),
    UCmpGe(Value, Value),

    FCmpOEq(Value, Value),
    FCmpONe(Value, Value),
    FCmpOLt(Value, Value),
    FCmpOLe(Value, Value),
    FCmpOGt(Value, Value),
    FCmpOGe(Value, Value),

    FCmpUEq(Value, Value),
    FCmpUNe(Value, Value),
    FCmpULt(Value, Value),
    FCmpULe(Value, Value),
    FCmpUGt(Value, Value),
    FCmpUGe(Value, Value),

    // Select
    Select(Value, Value, Value),

    // Memory
    Load(Value),
    Store(Value, Value),

    // Address arithmetic
    Gep {
        result_type: PointerType,
        base: Value,
        index: Value,
        stride: u32,
    },

    // Type conversions
    Bitcast(Value),
    Trunc(Value),
    Zext(Value),
    Sext(Value),
    FpToSi(Value),
    FpToUi(Value),
    SiToFp(Value),
    UiToFp(Value),
    FpExt(Value),
    FpTrunc(Value),

    // Function calls
    Call {
        func: String,
        args: Vec<Value>,
    },

    // Atomics
    AtomicLoad {
        ptr: Value,
        ordering: MemoryOrdering,
        scope: MemoryScope,
    },
    AtomicStore {
        ptr: Value,
        value: Value,
        ordering: MemoryOrdering,
        scope: MemoryScope,
    },
    AtomicRmw {
        op: AtomicOp,
        ptr: Value,
        value: Value,
        ordering: MemoryOrdering,
        scope: MemoryScope,
    },
    AtomicCmpXchg {
        ptr: Value,
        expected: Value,
        desired: Value,
        ordering_succ: MemoryOrdering,
        ordering_fail: MemoryOrdering,
        scope: MemoryScope,
    },

    // Phi
    Phi {
        ty: Type,
        incoming: Vec<(Value, String)>, // (value, label)
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Sub,
    MinS,
    MinU,
    MaxS,
    MaxU,
    And,
    Or,
    Xor,
    Exchange,
    IncWrap,
    DecWrap,
    FAdd,
    FMin,
    FMax,
    FlagTestAndSet,
    FlagClear,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryScope {
    Invocation,
    Subgroup,
    Workgroup,
    Device,
    System,
}

/// Terminators
#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    Br(String),
    BrIf {
        cond: Value,
        true_label: String,
        false_label: String,
        /// Merge label for structured control flow.
        /// None when inside a loop header (OpLoopMerge handles the merge).
        /// Some when standalone selection (OpSelectionMerge needed).
        merge_label: Option<String>,
    },
    Ret(Option<Value>),
}
