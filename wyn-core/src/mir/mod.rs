//! MIR (Mid-level Intermediate Representation) for the Wyn compiler.
//!
//! This representation assumes:
//! - Type checking has already occurred; concrete types are stored with expressions
//! - Patterns have been flattened to simple let bindings
//! - Lambdas have been lifted to top-level functions
//! - Imports and namespacing have been resolved
//! - Range expressions have been desugared

use crate::ast::{Span, TypeName};
use polytype::Type;

#[cfg(test)]
mod tests;

pub mod visitor;

/// A complete MIR program.
#[derive(Debug, Clone)]
pub struct Program {
    /// All top-level definitions in the program.
    pub defs: Vec<Def>,
    /// Lambda registry: maps tag -> (function_name, arity).
    /// Used for closure dispatch in higher-order builtins like map.
    /// Tags are assigned in order during flattening.
    pub lambda_registry: Vec<(String, usize)>,
}

/// A top-level definition (function or constant).
#[derive(Debug, Clone)]
pub enum Def {
    /// A function definition with parameters.
    Function {
        /// Function name.
        name: String,
        /// Function parameters with optional uniqueness markers.
        params: Vec<Param>,
        /// Return type.
        ret_type: Type<TypeName>,
        /// Attributes attached to this function (e.g., "entry", "inline", "noinline").
        attributes: Vec<Attribute>,
        /// Attributes for each parameter (for shader I/O decorations).
        param_attributes: Vec<Vec<Attribute>>,
        /// Return value attributes (for shader I/O decorations).
        /// For multiple outputs, each element corresponds to one output.
        return_attributes: Vec<Vec<Attribute>>,
        /// The function body expression.
        body: Expr,
        /// Source location.
        span: Span,
    },
    /// A constant definition (no parameters).
    Constant {
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
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
    Vertex,
    Fragment,
    Uniform,
}

/// The main expression type with source location and type.
#[derive(Debug, Clone)]
pub struct Expr {
    pub ty: Type<TypeName>,
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(ty: Type<TypeName>, kind: ExprKind, span: Span) -> Self {
        Expr { ty, kind, span }
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

/// Extract lambda function name from a closure tuple if present.
/// Closure tuples have the lambda name at the last index as a string literal
/// that identifies which lambda function the closure calls.
pub fn extract_lambda_name(expr: &Expr) -> Option<&str> {
    match &expr.kind {
        ExprKind::Literal(Literal::Tuple(elems)) if !elems.is_empty() => {
            if let ExprKind::Literal(Literal::String(lambda_name)) = &elems.last()?.kind {
                return Some(lambda_name.as_str());
            }
            None
        }
        _ => None,
    }
}
