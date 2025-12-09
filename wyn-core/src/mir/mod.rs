//! MIR (Mid-level Intermediate Representation) for the Wyn compiler.
//!
//! This representation assumes:
//! - Type checking has already occurred; concrete types are stored with expressions
//! - Patterns have been flattened to simple let bindings
//! - Lambdas have been lifted to top-level functions
//! - Imports and namespacing have been resolved
//! - Range expressions have been desugared

use crate::ast::{NodeId, Span, TypeName};
use polytype::Type;
use std::fmt;

// =============================================================================
// Symbol IDs - typed identifiers for MIR entities
// =============================================================================

/// ID for a local variable or parameter within a function.
/// These are assigned during flattening and remain stable through transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

impl fmt::Display for LocalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

/// ID for a top-level definition (function, constant, uniform, etc.).
/// This is the index into Program.defs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub u32);

impl fmt::Display for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "def{}", self.0)
    }
}

/// Enumeration of MIR-level intrinsic operations.
/// These are structural operations that don't correspond to user-defined functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicId {
    /// Access tuple element by index: tuple_access(tuple, index)
    TupleAccess,
    /// Array indexing: index(array, index)
    Index,
    /// Array length: length(array)
    Length,
    /// Assertion: assert(condition, message)
    Assert,
    /// Record field access: record_access(record, field_name)
    RecordAccess,
    /// Record field update: record_update(record, field_name, value)
    RecordUpdate,
}

impl IntrinsicId {
    /// Parse an intrinsic name string into an IntrinsicId.
    /// Returns None if the name is not a recognized intrinsic.
    pub fn from_str(name: &str) -> Option<Self> {
        match name {
            "tuple_access" => Some(IntrinsicId::TupleAccess),
            "index" => Some(IntrinsicId::Index),
            "length" => Some(IntrinsicId::Length),
            "assert" => Some(IntrinsicId::Assert),
            "record_access" => Some(IntrinsicId::RecordAccess),
            "record_update" => Some(IntrinsicId::RecordUpdate),
            _ => None,
        }
    }

    /// Convert an IntrinsicId back to its canonical string name.
    pub fn as_str(&self) -> &'static str {
        match self {
            IntrinsicId::TupleAccess => "tuple_access",
            IntrinsicId::Index => "index",
            IntrinsicId::Length => "length",
            IntrinsicId::Assert => "assert",
            IntrinsicId::RecordAccess => "record_access",
            IntrinsicId::RecordUpdate => "record_update",
        }
    }
}

impl fmt::Display for IntrinsicId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Symbol Tables - metadata for locals and definitions
// =============================================================================

/// Information about a local variable within a function.
#[derive(Debug, Clone)]
pub struct LocalInfo {
    /// Debug name for error messages and pretty-printing.
    pub debug_name: String,
    /// Type of this local.
    pub ty: Type<TypeName>,
    /// Source span where this local was defined.
    pub span: Option<Span>,
}

/// Symbol table for locals within a single function.
/// LocalId.0 is used as an index into this table.
#[derive(Debug, Clone, Default)]
pub struct LocalTable {
    /// Information for each local, indexed by LocalId.0.
    pub locals: Vec<LocalInfo>,
}

impl LocalTable {
    /// Create a new empty local table.
    pub fn new() -> Self {
        LocalTable { locals: Vec::new() }
    }

    /// Allocate a new local and return its ID.
    pub fn alloc(&mut self, debug_name: String, ty: Type<TypeName>, span: Option<Span>) -> LocalId {
        let id = LocalId(self.locals.len() as u32);
        self.locals.push(LocalInfo { debug_name, ty, span });
        id
    }

    /// Look up info for a local by ID.
    pub fn get(&self, id: LocalId) -> Option<&LocalInfo> {
        self.locals.get(id.0 as usize)
    }

    /// Get the debug name for a local.
    pub fn name(&self, id: LocalId) -> &str {
        self.locals.get(id.0 as usize)
            .map(|info| info.debug_name.as_str())
            .unwrap_or("<unknown>")
    }
}

#[cfg(test)]
mod tests;

pub mod folder;

/// A complete MIR program.
#[derive(Debug, Clone)]
pub struct Program {
    /// All top-level definitions in the program.
    pub defs: Vec<Def>,
    /// Lambda registry: maps tag -> (function_name, arity).
    /// Used for closure dispatch in higher-order builtins like map.
    /// Tags are assigned in order during flattening.
    pub lambda_registry: Vec<(String, usize)>,
    /// Local symbol tables for each function, keyed by DefId.
    /// Only populated for Function and EntryPoint defs.
    pub local_tables: std::collections::HashMap<DefId, LocalTable>,
}

impl Program {
    /// Look up a definition by name.
    pub fn def_by_name(&self, name: &str) -> Option<(DefId, &Def)> {
        self.defs.iter().enumerate().find_map(|(i, def)| {
            if def.name() == name {
                Some((DefId(i as u32), def))
            } else {
                None
            }
        })
    }

    /// Get the name of a definition by DefId.
    pub fn def_name(&self, id: DefId) -> Option<&str> {
        self.defs.get(id.0 as usize).map(|d| d.name())
    }

    /// Get the local table for a function.
    pub fn local_table(&self, id: DefId) -> Option<&LocalTable> {
        self.local_tables.get(&id)
    }
}

/// A top-level definition (function or constant).
#[derive(Debug, Clone)]
pub enum Def {
    /// A function definition with parameters.
    Function {
        /// Unique node identifier.
        id: NodeId,
        /// Function name.
        name: String,
        /// Function parameters with optional uniqueness markers.
        params: Vec<Param>,
        /// Return type.
        ret_type: Type<TypeName>,
        /// Attributes attached to this function (e.g., "inline", "noinline").
        attributes: Vec<Attribute>,
        /// The function body expression.
        body: Expr,
        /// Source location.
        span: Span,
    },
    /// A constant definition (no parameters).
    Constant {
        /// Unique node identifier.
        id: NodeId,
        /// Constant name.
        name: String,
        /// The type of this constant.
        ty: Type<TypeName>,
        /// Attributes attached to this constant.
        attributes: Vec<Attribute>,
        /// The constant value expression.
        body: Expr,
        /// Source location.
        span: Span,
    },
    /// A uniform declaration (external input from host).
    Uniform {
        /// Unique node identifier.
        id: NodeId,
        /// Uniform name.
        name: String,
        /// The type of this uniform.
        ty: Type<TypeName>,
        /// Descriptor set number.
        set: u32,
        /// Explicit binding number.
        binding: u32,
    },
    /// A storage buffer declaration (read-write GPU memory).
    Storage {
        /// Unique node identifier.
        id: NodeId,
        /// Storage buffer name.
        name: String,
        /// The type of this storage buffer (usually runtime-sized array).
        ty: Type<TypeName>,
        /// Descriptor set number.
        set: u32,
        /// Binding number within the set.
        binding: u32,
        /// Memory layout (std430, std140).
        layout: crate::ast::StorageLayout,
        /// Access mode (read, write, readwrite).
        access: crate::ast::StorageAccess,
    },
    /// A shader entry point (vertex, fragment, or compute shader).
    EntryPoint {
        /// Unique node identifier.
        id: NodeId,
        /// Entry point name.
        name: String,
        /// Execution model (vertex, fragment, compute).
        execution_model: ExecutionModel,
        /// Input parameters with their I/O decorations.
        inputs: Vec<EntryInput>,
        /// Output values with their I/O decorations.
        /// For single return: one element. For tuple return: one per tuple element.
        outputs: Vec<EntryOutput>,
        /// The entry point body expression.
        body: Expr,
        /// Source location.
        span: Span,
    },
}

impl Def {
    /// Get the name of this definition.
    pub fn name(&self) -> &str {
        match self {
            Def::Function { name, .. } => name,
            Def::Constant { name, .. } => name,
            Def::Uniform { name, .. } => name,
            Def::Storage { name, .. } => name,
            Def::EntryPoint { name, .. } => name,
        }
    }

    /// Get the NodeId of this definition.
    pub fn node_id(&self) -> NodeId {
        match self {
            Def::Function { id, .. } => *id,
            Def::Constant { id, .. } => *id,
            Def::Uniform { id, .. } => *id,
            Def::Storage { id, .. } => *id,
            Def::EntryPoint { id, .. } => *id,
        }
    }
}

/// A function parameter.
#[derive(Debug, Clone)]
pub struct Param {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub ty: Type<TypeName>,
    /// Whether this parameter is consumed (unique/in-place update).
    pub is_consumed: bool,
}

/// An attribute that can be attached to functions or expressions.
/// NOTE: Vertex/Fragment/Compute are deprecated for functions - use Def::EntryPoint instead.
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
    Vertex,
    Fragment,
    Compute {
        local_size: (u32, u32, u32),
    },
    Uniform,
    Storage,
}

/// Execution model for a shader entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionModel {
    Vertex,
    Fragment,
    Compute { local_size: (u32, u32, u32) },
}

/// Decoration for shader I/O (entry point parameters and return values).
#[derive(Debug, Clone, PartialEq)]
pub enum IoDecoration {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
}

/// An input parameter to a shader entry point.
#[derive(Debug, Clone)]
pub struct EntryInput {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub ty: Type<TypeName>,
    /// I/O decoration (location or builtin).
    pub decoration: Option<IoDecoration>,
}

/// An output from a shader entry point.
#[derive(Debug, Clone)]
pub struct EntryOutput {
    /// Output type.
    pub ty: Type<TypeName>,
    /// I/O decoration (location or builtin).
    pub decoration: Option<IoDecoration>,
}

/// The main expression type with source location and type.
#[derive(Debug, Clone)]
pub struct Expr {
    /// Unique node identifier.
    pub id: NodeId,
    pub ty: Type<TypeName>,
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(id: NodeId, ty: Type<TypeName>, kind: ExprKind, span: Span) -> Self {
        Expr { id, ty, kind, span }
    }
}

/// Expression kinds in MIR.
#[derive(Debug, Clone)]
pub enum ExprKind {
    /// A literal value.
    Literal(Literal),

    /// Unit value ().
    Unit,

    /// A variable reference by name.
    Var(String),

    /// A binary operation.
    BinOp {
        /// The operator (e.g., "+", "-", "*", "/", "&&", "||", "==", "<").
        op: String,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },

    /// A unary operation.
    UnaryOp {
        /// The operator (e.g., "-", "!").
        op: String,
        /// Operand.
        operand: Box<Expr>,
    },

    /// Conditional expression.
    If {
        /// Condition.
        cond: Box<Expr>,
        /// Then branch.
        then_branch: Box<Expr>,
        /// Else branch.
        else_branch: Box<Expr>,
    },

    /// Let binding: `let name = value in body`.
    Let {
        /// Bound variable name.
        name: String,
        /// Unique binding ID for tracking backing stores.
        binding_id: u64,
        /// Value to bind.
        value: Box<Expr>,
        /// Body expression where the binding is in scope.
        body: Box<Expr>,
    },

    /// Unified loop construct.
    Loop {
        /// Name for the value returned by the body each iteration.
        loop_var: String,
        /// Initial value for loop_var.
        init: Box<Expr>,
        /// Bindings that extract from loop_var for use in condition/body.
        /// For single var: `[(x, Var(loop_var))]`
        /// For tuple: `[(a, tuple_access(loop_var, 0)), (b, tuple_access(loop_var, 1))]`
        init_bindings: Vec<(String, Expr)>,
        /// The kind of loop (for, for-range, or while).
        kind: LoopKind,
        /// Loop body expression.
        body: Box<Expr>,
    },

    /// Regular function call.
    Call {
        /// Function name.
        func: String,
        /// Arguments.
        args: Vec<Expr>,
    },

    /// Compiler intrinsic call.
    Intrinsic {
        /// Intrinsic name (e.g., "index", "slice", "length", "assert",
        /// "record_access", "record_update", "tuple_access").
        name: String,
        /// Arguments.
        args: Vec<Expr>,
    },

    /// An expression with attributes attached.
    Attributed {
        /// Attributes on this expression.
        attributes: Vec<Attribute>,
        /// The inner expression.
        expr: Box<Expr>,
    },

    /// Materialize a value into a variable, returning a reference to that variable.
    /// This is used when a value needs to be stored before being indexed/accessed.
    /// In SPIR-V, this becomes: declare OpVariable, OpStore, return pointer.
    Materialize(Box<Expr>),

    /// A closure value (defunctionalized lambda).
    /// Represents a lambda that has been lifted to a top-level function with captured values.
    Closure {
        /// Name of the generated lambda function to call.
        lambda_name: String,
        /// Captured variables (may be empty for lambdas with no free variables).
        captures: Vec<Expr>,
    },
}

/// Literal values, categorized by type class.
/// The exact type is stored in out-of-band type information.
#[derive(Debug, Clone)]
pub enum Literal {
    /// Integer literal (i8, i16, i32, i64, u8, u16, u32, u64).
    /// Stored as string to preserve exact representation and support arbitrary precision.
    Int(String),
    /// Floating-point literal (f16, f32, f64).
    /// Stored as string to preserve exact representation.
    Float(String),
    /// Boolean literal.
    Bool(bool),
    /// String literal (represented as UTF-8 bytes in Futhark).
    String(String),
    /// Tuple literal.
    Tuple(Vec<Expr>),
    /// Array literal.
    Array(Vec<Expr>),
    /// Vector literal (@[1.0, 2.0, 3.0]).
    Vector(Vec<Expr>),
    /// Matrix literal (@[[1,2], [3,4]]) - outer vec is rows, inner is columns.
    Matrix(Vec<Vec<Expr>>),
}

/// The kind of loop construct.
#[derive(Debug, Clone)]
pub enum LoopKind {
    /// For loop over an array: `for x in arr`.
    For {
        /// Loop variable name.
        var: String,
        /// Array to iterate over.
        iter: Box<Expr>,
    },
    /// For loop with range bound: `for i < n`.
    ForRange {
        /// Loop variable name.
        var: String,
        /// Upper bound.
        bound: Box<Expr>,
    },
    /// While loop: `while cond`.
    While {
        /// Loop condition.
        cond: Box<Expr>,
    },
}

