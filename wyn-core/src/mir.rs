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

/// A complete MIR program.
#[derive(Debug, Clone)]
pub struct Program {
    /// All top-level definitions in the program.
    pub defs: Vec<Def>,
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
#[derive(Debug, Clone)]
pub struct Attribute {
    /// Attribute name (e.g., "entry", "trace", "unsafe", "unroll").
    pub name: String,
    /// Optional attribute arguments.
    pub args: Vec<String>,
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
        /// Initial bindings: `loop (x, y) = (init_x, init_y)`.
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
    /// Record literal.
    Record(Vec<(String, Expr)>),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn i32_type() -> Type<TypeName> {
        Type::Constructed(TypeName::Str("i32".into()), vec![])
    }

    fn f32_type() -> Type<TypeName> {
        Type::Constructed(TypeName::Str("f32".into()), vec![])
    }

    #[test]
    fn test_simple_function() {
        // Represents: def add(x, y) = x + y
        let add_fn = Def::Function {
            name: "add".to_string(),
            params: vec![
                Param {
                    name: "x".to_string(),
                    ty: i32_type(),
                    is_consumed: false,
                },
                Param {
                    name: "y".to_string(),
                    ty: i32_type(),
                    is_consumed: false,
                },
            ],
            ret_type: i32_type(),
            attributes: vec![],
            body: Expr::new(
                i32_type(),
                ExprKind::BinOp {
                    op: "+".to_string(),
                    lhs: Box::new(Expr::new(
                        i32_type(),
                        ExprKind::Var("x".to_string()),
                        Span::dummy(),
                    )),
                    rhs: Box::new(Expr::new(
                        i32_type(),
                        ExprKind::Var("y".to_string()),
                        Span::dummy(),
                    )),
                },
                Span::dummy(),
            ),
            span: Span::dummy(),
        };

        let program = Program { defs: vec![add_fn] };

        assert_eq!(program.defs.len(), 1);
        match &program.defs[0] {
            Def::Function { name, .. } => assert_eq!(name, "add"),
            _ => panic!("Expected Function"),
        }
    }

    #[test]
    fn test_constant() {
        // Represents: def pi = 3.14159
        let pi_const = Def::Constant {
            name: "pi".to_string(),
            ty: f32_type(),
            attributes: vec![],
            body: Expr::new(
                f32_type(),
                ExprKind::Literal(Literal::Float("3.14159".to_string())),
                Span::dummy(),
            ),
            span: Span::dummy(),
        };

        match pi_const {
            Def::Constant { name, .. } => assert_eq!(name, "pi"),
            _ => panic!("Expected Constant"),
        }
    }
}
