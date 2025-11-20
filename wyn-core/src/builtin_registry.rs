// Centralized builtin function registry
// Provides type signatures and code generation implementations for all builtin functions

use crate::ast::{Type, TypeName, TypeScheme};
use crate::type_checker::TypeVarGenerator;
use std::collections::HashMap;

/// Implementation strategy for a builtin function
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinImpl {
    /// GLSL.std.450 extended instruction
    GlslExt(u32),

    /// Core SPIR-V instruction (e.g., OpDot, OpOuterProduct)
    SpirvOp(SpirvOp),

    /// Custom assembly/implementation
    Custom(CustomImpl),
}

/// Core SPIR-V operations
#[derive(Debug, Clone, PartialEq)]
pub enum SpirvOp {
    Dot,
    OuterProduct,
    MatrixTimesMatrix,
    MatrixTimesVector,
    VectorTimesMatrix,
    VectorTimesScalar,
    MatrixTimesScalar,
    // Arithmetic ops
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    FMod,
    IAdd,
    ISub,
    IMul,
    SDiv,
    UDiv,
    SRem,
    SMod,
    // Comparison ops
    FOrdEqual,
    FOrdNotEqual,
    FOrdLessThan,
    FOrdGreaterThan,
    FOrdLessThanEqual,
    FOrdGreaterThanEqual,
    IEqual,
    INotEqual,
    SLessThan,
    ULessThan,
    SGreaterThan,
    UGreaterThan,
    SLessThanEqual,
    ULessThanEqual,
    SGreaterThanEqual,
    UGreaterThanEqual,
    // Bitwise ops
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Not,
    ShiftLeftLogical,
    ShiftRightArithmetic,
    ShiftRightLogical,
}

/// Custom implementation (for complex builtins that need special handling)
#[derive(Debug, Clone, PartialEq)]
pub enum CustomImpl {
    /// Placeholder for future custom implementations
    Placeholder,
    /// Uninitialized/poison value for allocation bootstrapping
    /// SAFETY: Must be fully overwritten before being read
    Uninit,
    /// Array replication: creates array filled with a value
    Replicate,
    /// Functional array update: immutable copy-with-update
    ArrayUpdate,
}

/// Polymorphic implementation dispatcher
/// Dispatches to concrete implementations based on resolved types
#[derive(Debug, Clone, PartialEq)]
pub enum PolyImpl {
    /// Dispatch based on whether type is float or integer
    NumericDispatch {
        float_impl: BuiltinImpl,
        int_impl: BuiltinImpl,
    },

    /// Dispatch based on whether integer is signed or unsigned
    IntegralDispatch {
        signed_impl: BuiltinImpl,
        unsigned_impl: BuiltinImpl,
    },

    /// Real number operations (floats only)
    RealOp(BuiltinImpl),
}

impl PolyImpl {
    fn type_name(ty: &Type) -> Option<&str> {
        match ty {
            Type::Constructed(TypeName::Str(name), _) => Some(name),
            _ => None,
        }
    }

    fn is_real(ty: &Type) -> bool {
        matches!(Self::type_name(ty), Some("f16" | "f32" | "f64"))
    }

    fn is_integral(ty: &Type) -> bool {
        matches!(
            Self::type_name(ty),
            Some("i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64")
        )
    }
}

/// Builtin function descriptor (monomorphic)
#[derive(Debug, Clone)]
pub struct BuiltinDescriptor {
    /// Fully qualified name (e.g., "f32.sin", "i32.+")
    pub name: String,

    /// Parameter types
    pub param_types: Vec<Type>,

    /// Return type
    pub return_type: Type,

    /// Code generation implementation
    pub implementation: BuiltinImpl,
}

/// Polymorphic builtin descriptor
/// Represents a family of related functions across multiple types
#[derive(Debug, Clone)]
pub struct PolyBuiltinDescriptor {
    /// Base name without module qualifier (e.g., "abs", "cos", "+")
    pub name: String,

    /// Type scheme (e.g., "forall a. a -> a")
    pub type_scheme: TypeScheme,

    /// Implementation dispatcher
    pub implementation: PolyImpl,
}

impl PolyImpl {
    /// Resolve to concrete implementation for a given type
    pub fn resolve(&self, ty: &Type) -> Option<BuiltinImpl> {
        match self {
            PolyImpl::NumericDispatch { float_impl, int_impl } => {
                if Self::is_real(ty) {
                    Some(float_impl.clone())
                } else if Self::is_integral(ty) {
                    Some(int_impl.clone())
                } else {
                    None
                }
            }
            PolyImpl::IntegralDispatch {
                signed_impl,
                unsigned_impl,
            } => {
                let is_signed = matches!(Self::type_name(ty), Some("i8" | "i16" | "i32" | "i64"));
                let is_unsigned = matches!(Self::type_name(ty), Some("u8" | "u16" | "u32" | "u64"));

                if is_signed {
                    Some(signed_impl.clone())
                } else if is_unsigned {
                    Some(unsigned_impl.clone())
                } else {
                    None
                }
            }
            PolyImpl::RealOp(impl_) => {
                if Self::is_real(ty) {
                    Some(impl_.clone())
                } else {
                    None
                }
            }
        }
    }
}

/// Central registry for all builtin functions
pub struct BuiltinRegistry {
    /// Monomorphic builtins (fully qualified names like "f32.sin")
    builtins: HashMap<String, BuiltinDescriptor>,

    /// Polymorphic builtins (base names like "abs", indexed by module.name)
    poly_builtins: HashMap<String, PolyBuiltinDescriptor>,
}

impl BuiltinRegistry {
    pub fn new(ctx: &mut impl TypeVarGenerator) -> Self {
        let mut registry = BuiltinRegistry {
            builtins: HashMap::new(),
            poly_builtins: HashMap::new(),
        };

        registry.register_from_prim_module();
        registry.register_numeric_modules();
        registry.register_integral_modules();
        registry.register_real_modules();
        registry.register_float_modules();
        registry.register_vector_operations(ctx);
        registry.register_matrix_operations();
        registry.register_vector_constructors();
        registry.register_higher_order_functions(ctx);

        registry
    }

    /// Check if a name is a registered builtin (monomorphic or polymorphic)
    pub fn is_builtin(&self, name: &str) -> bool {
        self.builtins.contains_key(name) || self.poly_builtins.contains_key(name)
    }

    /// Get all builtin names as a HashSet (for use in flattening to exclude from capture)
    pub fn all_names(&self) -> std::collections::HashSet<String> {
        let mut names = std::collections::HashSet::new();
        for name in self.builtins.keys() {
            names.insert(name.clone());
        }
        for name in self.poly_builtins.keys() {
            names.insert(name.clone());
        }
        names
    }

    /// Get monomorphic builtin descriptor
    pub fn get(&self, name: &str) -> Option<&BuiltinDescriptor> {
        self.builtins.get(name)
    }

    /// Get polymorphic builtin descriptor
    pub fn get_poly(&self, name: &str) -> Option<&PolyBuiltinDescriptor> {
        self.poly_builtins.get(name)
    }

    /// Register a polymorphic builtin
    fn register_poly(&mut self, desc: PolyBuiltinDescriptor) {
        self.poly_builtins.insert(desc.name.clone(), desc);
    }

    /// Get the type of a field on a given type (e.g., vec3.x returns f32)
    pub fn get_field_type(&self, type_name: &str, field_name: &str) -> Option<Type> {
        match (type_name, field_name) {
            // f32 vectors
            ("vec2", "x" | "y") => Some(Self::ty("f32")),
            ("vec3", "x" | "y" | "z") => Some(Self::ty("f32")),
            ("vec4", "x" | "y" | "z" | "w") => Some(Self::ty("f32")),

            // i32 vectors
            ("ivec2", "x" | "y") => Some(Self::ty("i32")),
            ("ivec3", "x" | "y" | "z") => Some(Self::ty("i32")),
            ("ivec4", "x" | "y" | "z" | "w") => Some(Self::ty("i32")),

            _ => None,
        }
    }

    /// Register a builtin function
    fn register(&mut self, desc: BuiltinDescriptor) {
        self.builtins.insert(desc.name.clone(), desc);
    }

    /// Helper to create a type from a static string
    fn ty(name: &'static str) -> Type {
        Type::Constructed(TypeName::Str(name), vec![])
    }

    /// Register from_prim module functions (type conversions)
    fn register_from_prim_module(&mut self) {
        // TODO: Implement from_prim conversions
        // These are type conversion functions like i32.i8, i32.f32, etc.
    }

    /// Register numeric module functions (common to all numeric types)
    fn register_numeric_modules(&mut self) {
        self.register_numeric_ops("i8");
        self.register_numeric_ops("i16");
        self.register_numeric_ops("i32");
        self.register_numeric_ops("i64");
        self.register_numeric_ops("u8");
        self.register_numeric_ops("u16");
        self.register_numeric_ops("u32");
        self.register_numeric_ops("u64");
        self.register_numeric_ops("f16");
        self.register_numeric_ops("f32");
        self.register_numeric_ops("f64");
    }

    /// Helper to register a binary operation
    fn register_binop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        self.register(BuiltinDescriptor {
            name: format!("{}.{}", ty_name, op),
            param_types: vec![t.clone(), t.clone()],
            return_type: t,
            implementation: impl_,
        });
    }

    /// Helper to register a comparison operation
    fn register_cmp(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        self.register(BuiltinDescriptor {
            name: format!("{}.{}", ty_name, op),
            param_types: vec![t.clone(), t.clone()],
            return_type: Self::ty("bool"),
            implementation: impl_,
        });
    }

    /// Helper to register a unary operation
    fn register_unop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        self.register(BuiltinDescriptor {
            name: format!("{}.{}", ty_name, op),
            param_types: vec![t.clone()],
            return_type: t,
            implementation: impl_,
        });
    }

    fn register_numeric_ops(&mut self, ty_name: &'static str) {
        let is_float = ty_name.starts_with('f');
        let is_signed = ty_name.starts_with('i') || is_float;

        // Arithmetic operators
        self.register_binop(
            ty_name,
            "+",
            if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FAdd)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::IAdd)
            },
        );
        self.register_binop(
            ty_name,
            "-",
            if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FSub)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::ISub)
            },
        );
        self.register_binop(
            ty_name,
            "*",
            if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FMul)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::IMul)
            },
        );
        self.register_binop(
            ty_name,
            "/",
            if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FDiv)
            } else if is_signed {
                BuiltinImpl::SpirvOp(SpirvOp::SDiv)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::UDiv)
            },
        );

        // Comparison operators
        self.register_cmp(
            ty_name,
            "<",
            if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FOrdLessThan)
            } else if is_signed {
                BuiltinImpl::SpirvOp(SpirvOp::SLessThan)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::ULessThan)
            },
        );
        self.register_cmp(
            ty_name,
            "==",
            if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FOrdEqual)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::IEqual)
            },
        );

        // min, max, abs functions
        if is_float {
            self.register_binop(ty_name, "min", BuiltinImpl::GlslExt(37)); // FMin
            self.register_binop(ty_name, "max", BuiltinImpl::GlslExt(40)); // FMax
            self.register_unop(ty_name, "abs", BuiltinImpl::GlslExt(4)); // FAbs
        } else {
            self.register_binop(
                ty_name,
                "min",
                if is_signed { BuiltinImpl::GlslExt(39) } else { BuiltinImpl::GlslExt(38) },
            );
            self.register_binop(
                ty_name,
                "max",
                if is_signed { BuiltinImpl::GlslExt(42) } else { BuiltinImpl::GlslExt(41) },
            );
            self.register_unop(ty_name, "abs", BuiltinImpl::GlslExt(5)); // SAbs
        }
    }

    /// Register integral module functions (bitwise ops, etc.)
    fn register_integral_modules(&mut self) {
        self.register_integral_ops("i8");
        self.register_integral_ops("i16");
        self.register_integral_ops("i32");
        self.register_integral_ops("i64");
        self.register_integral_ops("u8");
        self.register_integral_ops("u16");
        self.register_integral_ops("u32");
        self.register_integral_ops("u64");
    }

    fn register_integral_ops(&mut self, ty_name: &'static str) {
        self.register_binop(ty_name, "&", BuiltinImpl::SpirvOp(SpirvOp::BitwiseAnd));
        self.register_binop(ty_name, "|", BuiltinImpl::SpirvOp(SpirvOp::BitwiseOr));
        self.register_binop(ty_name, "^", BuiltinImpl::SpirvOp(SpirvOp::BitwiseXor));
        self.register_binop(ty_name, "<<", BuiltinImpl::SpirvOp(SpirvOp::ShiftLeftLogical));
        self.register_binop(
            ty_name,
            ">>",
            if ty_name.starts_with('i') {
                BuiltinImpl::SpirvOp(SpirvOp::ShiftRightArithmetic)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::ShiftRightLogical)
            },
        );
    }

    /// Register real (floating-point) module functions
    fn register_real_modules(&mut self) {
        self.register_real_ops("f16");
        self.register_real_ops("f32");
        self.register_real_ops("f64");
    }

    /// Helper to register a ternary operation
    fn register_ternop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        self.register(BuiltinDescriptor {
            name: format!("{}.{}", ty_name, op),
            param_types: vec![t.clone(), t.clone(), t.clone()],
            return_type: t,
            implementation: impl_,
        });
    }

    /// Helper to register a unary operation with bool return
    fn register_bool_unop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        self.register(BuiltinDescriptor {
            name: format!("{}.{}", ty_name, op),
            param_types: vec![t],
            return_type: Self::ty("bool"),
            implementation: impl_,
        });
    }

    fn register_real_ops(&mut self, ty_name: &'static str) {
        // TODO: These should come from a prelude module (e.g., f32.sqrt from the f32 module)
        // rather than being hardcoded in the builtin registry. For now, we register them
        // here so they're available for type checking and code generation.

        // Transcendental functions
        self.register_unop(ty_name, "sin", BuiltinImpl::GlslExt(13));
        self.register_unop(ty_name, "cos", BuiltinImpl::GlslExt(14));
        self.register_unop(ty_name, "tan", BuiltinImpl::GlslExt(15));
        self.register_unop(ty_name, "asin", BuiltinImpl::GlslExt(16));
        self.register_unop(ty_name, "acos", BuiltinImpl::GlslExt(17));
        self.register_unop(ty_name, "atan", BuiltinImpl::GlslExt(18));
        self.register_unop(ty_name, "sqrt", BuiltinImpl::GlslExt(31));
        self.register_unop(ty_name, "exp", BuiltinImpl::GlslExt(27));
        self.register_unop(ty_name, "log", BuiltinImpl::GlslExt(28));
        self.register_binop(ty_name, "pow", BuiltinImpl::GlslExt(26));
        self.register_unop(ty_name, "floor", BuiltinImpl::GlslExt(8));
        self.register_unop(ty_name, "ceil", BuiltinImpl::GlslExt(9));
        self.register_unop(ty_name, "round", BuiltinImpl::GlslExt(1));
        self.register_unop(ty_name, "trunc", BuiltinImpl::GlslExt(3));
        self.register_ternop(ty_name, "clamp", BuiltinImpl::GlslExt(43));

        // isnan, isinf
        self.register_bool_unop(ty_name, "isnan", BuiltinImpl::GlslExt(66));
        self.register_bool_unop(ty_name, "isinf", BuiltinImpl::GlslExt(67));
    }

    /// Register float-specific module functions
    fn register_float_modules(&mut self) {
        // Float-specific operations (bit manipulation, etc.)
        // TODO: Implement f16, f32, f64 specific ops like from_bits, to_bits
    }

    /// Register vector operations (length, normalize, dot, cross, etc.)
    fn register_vector_operations(&mut self, ctx: &mut impl TypeVarGenerator) {
        // Use type variables for polymorphism: Vec[?n, ?T]
        let size_var = ctx.new_variable(); // ?n - the size
        let elem_var = ctx.new_variable(); // ?T - the element type
        let vec_t = Type::Constructed(TypeName::Vec, vec![size_var.clone(), elem_var.clone()]);

        // Skip vector length (magnitude) for now - define in user code as it needs special handling
        // self.register(BuiltinDescriptor {
        //     name: "length".to_string(),
        //     param_types: vec![vec_t.clone()],
        //     return_type: elem_var.clone(),  // returns T
        //     implementation: BuiltinImpl::GlslExt(66),
        // });

        // Array length: returns the size as i32
        let arr_size_var = ctx.new_variable();
        let arr_elem_var = ctx.new_variable();
        let arr_t = Type::Constructed(TypeName::Array, vec![arr_size_var, arr_elem_var]);
        self.register(BuiltinDescriptor {
            name: "length".to_string(),
            param_types: vec![arr_t],
            return_type: Self::ty("i32"), // TODO: Should this return the size type instead?
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        // __uninit: ∀t. t
        // Returns uninitialized/poison value (must be overwritten before reading)
        let uninit_type_var = ctx.new_variable();
        self.register(BuiltinDescriptor {
            name: "__uninit".to_string(),
            param_types: vec![],
            return_type: uninit_type_var,
            implementation: BuiltinImpl::Custom(CustomImpl::Uninit),
        });

        // replicate: ∀t. i32 -> t -> []t
        // Creates array of unknown length filled with a value (length-agnostic)
        let replicate_elem_var = ctx.new_variable();
        let replicate_result_type = Type::Constructed(
            TypeName::Array,
            vec![ctx.new_variable(), replicate_elem_var.clone()],
        );
        self.register(BuiltinDescriptor {
            name: "replicate".to_string(),
            param_types: vec![Self::ty("i32"), replicate_elem_var],
            return_type: replicate_result_type,
            implementation: BuiltinImpl::Custom(CustomImpl::Replicate),
        });

        // __array_update: ∀n t. [n]t -> i32 -> t -> [n]t
        // Functional array update (immutable copy-with-update)
        let update_size_var = ctx.new_variable();
        let update_elem_var = ctx.new_variable();
        let update_arr_type =
            Type::Constructed(TypeName::Array, vec![update_size_var, update_elem_var.clone()]);
        self.register(BuiltinDescriptor {
            name: "__array_update".to_string(),
            param_types: vec![update_arr_type.clone(), Self::ty("i32"), update_elem_var],
            return_type: update_arr_type,
            implementation: BuiltinImpl::Custom(CustomImpl::ArrayUpdate),
        });

        self.register(BuiltinDescriptor {
            name: "normalize".to_string(),
            param_types: vec![vec_t.clone()],
            return_type: vec_t.clone(), // returns Vec[n, T]
            implementation: BuiltinImpl::GlslExt(69),
        });

        self.register(BuiltinDescriptor {
            name: "dot".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone()],
            return_type: elem_var.clone(), // returns T
            implementation: BuiltinImpl::SpirvOp(SpirvOp::Dot),
        });

        self.register(BuiltinDescriptor {
            name: "cross".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone()],
            return_type: vec_t.clone(), // returns Vec[n, T] (only works for Vec[3, T])
            implementation: BuiltinImpl::GlslExt(68),
        });

        self.register(BuiltinDescriptor {
            name: "distance".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone()],
            return_type: elem_var.clone(), // returns T
            implementation: BuiltinImpl::GlslExt(67),
        });

        self.register(BuiltinDescriptor {
            name: "reflect".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone()],
            return_type: vec_t.clone(),
            implementation: BuiltinImpl::GlslExt(71),
        });

        self.register(BuiltinDescriptor {
            name: "refract".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone(), elem_var.clone()],
            return_type: vec_t.clone(),
            implementation: BuiltinImpl::GlslExt(72),
        });
    }

    /// Register matrix operations
    fn register_matrix_operations(&mut self) {
        let mat_t = Self::ty("mat"); // Placeholder
        let vec_t = Self::ty("vec");

        self.register(BuiltinDescriptor {
            name: "determinant".to_string(),
            param_types: vec![mat_t.clone()],
            return_type: Self::ty("f32"),
            implementation: BuiltinImpl::GlslExt(33),
        });

        self.register(BuiltinDescriptor {
            name: "inverse".to_string(),
            param_types: vec![mat_t.clone()],
            return_type: mat_t.clone(),
            implementation: BuiltinImpl::GlslExt(34),
        });

        self.register(BuiltinDescriptor {
            name: "outer".to_string(),
            param_types: vec![vec_t.clone(), vec_t],
            return_type: mat_t,
            implementation: BuiltinImpl::SpirvOp(SpirvOp::OuterProduct),
        });

        // Matrix operations for [4]vec4f32 representation
        let vec4f32 = Type::Constructed(
            TypeName::Vec,
            vec![Type::Constructed(TypeName::Size(4), vec![]), Self::ty("f32")],
        );
        let mat4 = Type::Constructed(
            TypeName::Array,
            vec![Type::Constructed(TypeName::Size(4), vec![]), vec4f32.clone()],
        );

        // mul_mat4: [4]vec4f32 -> [4]vec4f32 -> [4]vec4f32
        self.register(BuiltinDescriptor {
            name: "mul_mat4".to_string(),
            param_types: vec![mat4.clone(), mat4.clone()],
            return_type: mat4.clone(),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        // mul_rowvec_mat4: vec4f32 -> [4]vec4f32 -> vec4f32
        self.register(BuiltinDescriptor {
            name: "mul_rowvec_mat4".to_string(),
            param_types: vec![vec4f32.clone(), mat4],
            return_type: vec4f32,
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });
    }

    /// Helper to register a vector constructor
    fn register_vec_constructor(&mut self, vec_name: &'static str, elem_type: &'static str, arity: usize) {
        let elem_t = Self::ty(elem_type);
        // Return type is Vec[arity, elem_type], not just the name "vec4"
        let return_type = Type::Constructed(
            TypeName::Vec,
            vec![Type::Constructed(TypeName::Size(arity), vec![]), elem_t.clone()],
        );
        self.register(BuiltinDescriptor {
            name: vec_name.to_string(),
            param_types: vec![elem_t; arity],
            return_type,
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });
    }

    /// Register vector constructor functions
    fn register_vector_constructors(&mut self) {
        // f32 vectors
        self.register_vec_constructor("vec2", "f32", 2);
        self.register_vec_constructor("vec3", "f32", 3);
        self.register_vec_constructor("vec4", "f32", 4);

        // i32 vectors
        self.register_vec_constructor("ivec2", "i32", 2);
        self.register_vec_constructor("ivec3", "i32", 3);
        self.register_vec_constructor("ivec4", "i32", 4);
    }

    /// Register higher-order functions and array operations
    fn register_higher_order_functions(&mut self, ctx: &mut impl TypeVarGenerator) {
        // f32.sum: [n]f32 -> f32 (sum of array elements)
        // For now, we just support fixed f32 arrays, not polymorphic
        let size_var = ctx.new_variable();
        let arr_t = Type::Constructed(TypeName::Array, vec![size_var, Self::ty("f32")]);
        self.register(BuiltinDescriptor {
            name: "f32.sum".to_string(),
            param_types: vec![arr_t],
            return_type: Self::ty("f32"),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        // TODO: Implement polymorphic function registration
        // These functions need special type handling with type variables:
        // - length: ∀a. [a] -> int
        // - map: ∀a b. (a -> b) -> [a] -> [b]
        // - zip: ∀a b. [a] -> [b] -> [(a, b)]
        //
        // Current type system uses polytype for polymorphism but BuiltinDescriptor
        // uses monomorphic Type. Need to either:
        // 1. Add TypeScheme support to BuiltinDescriptor, or
        // 2. Keep polymorphic builtins separate in type checker
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        let mut ctx = polytype::Context::<crate::ast::TypeName>::default();
        Self::new(&mut ctx)
    }
}
