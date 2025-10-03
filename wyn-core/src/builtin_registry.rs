// Centralized builtin function registry
// Provides type signatures and code generation implementations for all builtin functions

use crate::ast::{Type, TypeName};
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
}

/// Builtin function descriptor
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

/// Central registry for all builtin functions
pub struct BuiltinRegistry {
    builtins: HashMap<String, BuiltinDescriptor>,
}

impl BuiltinRegistry {
    pub fn new() -> Self {
        let mut registry = BuiltinRegistry {
            builtins: HashMap::new(),
        };

        registry.register_from_prim_module();
        registry.register_numeric_modules();
        registry.register_integral_modules();
        registry.register_real_modules();
        registry.register_float_modules();
        registry.register_vector_operations();
        registry.register_matrix_operations();
        registry.register_vector_constructors();
        registry.register_higher_order_functions();

        registry
    }

    /// Check if a name is a registered builtin
    pub fn is_builtin(&self, name: &str) -> bool {
        self.builtins.contains_key(name)
    }

    /// Get builtin descriptor
    pub fn get(&self, name: &str) -> Option<&BuiltinDescriptor> {
        self.builtins.get(name)
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

    fn register_numeric_ops(&mut self, ty_name: &'static str) {
        let t = Self::ty(ty_name);
        let bool_t = Self::ty("bool");

        let is_float = ty_name.starts_with('f');
        let is_signed = ty_name.starts_with('i') || is_float;

        // Arithmetic operators
        self.register(BuiltinDescriptor {
            name: format!("{}.+", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: if is_float { BuiltinImpl::SpirvOp(SpirvOp::FAdd) } else { BuiltinImpl::SpirvOp(SpirvOp::IAdd) },
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.-", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: if is_float { BuiltinImpl::SpirvOp(SpirvOp::FSub) } else { BuiltinImpl::SpirvOp(SpirvOp::ISub) },
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.*", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: if is_float { BuiltinImpl::SpirvOp(SpirvOp::FMul) } else { BuiltinImpl::SpirvOp(SpirvOp::IMul) },
        });

        self.register(BuiltinDescriptor {
            name: format!("{}./", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FDiv)
            } else if is_signed {
                BuiltinImpl::SpirvOp(SpirvOp::SDiv)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::UDiv)
            },
        });

        // Comparison operators
        self.register(BuiltinDescriptor {
            name: format!("{}.<", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: bool_t.clone(),
            implementation: if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FOrdLessThan)
            } else if is_signed {
                BuiltinImpl::SpirvOp(SpirvOp::SLessThan)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::ULessThan)
            },
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.==", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: bool_t.clone(),
            implementation: if is_float {
                BuiltinImpl::SpirvOp(SpirvOp::FOrdEqual)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::IEqual)
            },
        });

        // min, max, abs functions
        if is_float {
            self.register(BuiltinDescriptor {
                name: format!("{}.min", ty_name),
                param_types: vec![t.clone(), t.clone()],
                return_type: t.clone(),
                implementation: BuiltinImpl::GlslExt(37), // FMin
            });

            self.register(BuiltinDescriptor {
                name: format!("{}.max", ty_name),
                param_types: vec![t.clone(), t.clone()],
                return_type: t.clone(),
                implementation: BuiltinImpl::GlslExt(40), // FMax
            });

            self.register(BuiltinDescriptor {
                name: format!("{}.abs", ty_name),
                param_types: vec![t.clone()],
                return_type: t.clone(),
                implementation: BuiltinImpl::GlslExt(4), // FAbs
            });
        } else {
            self.register(BuiltinDescriptor {
                name: format!("{}.min", ty_name),
                param_types: vec![t.clone(), t.clone()],
                return_type: t.clone(),
                implementation: if is_signed {
                    BuiltinImpl::GlslExt(39) // SMin
                } else {
                    BuiltinImpl::GlslExt(38) // UMin
                },
            });

            self.register(BuiltinDescriptor {
                name: format!("{}.max", ty_name),
                param_types: vec![t.clone(), t.clone()],
                return_type: t.clone(),
                implementation: if is_signed {
                    BuiltinImpl::GlslExt(42) // SMax
                } else {
                    BuiltinImpl::GlslExt(41) // UMax
                },
            });

            self.register(BuiltinDescriptor {
                name: format!("{}.abs", ty_name),
                param_types: vec![t.clone()],
                return_type: t.clone(),
                implementation: BuiltinImpl::GlslExt(5), // SAbs
            });
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
        let t = Self::ty(ty_name);

        // Bitwise operators
        self.register(BuiltinDescriptor {
            name: format!("{}.&", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::SpirvOp(SpirvOp::BitwiseAnd),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.|", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::SpirvOp(SpirvOp::BitwiseOr),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.^", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::SpirvOp(SpirvOp::BitwiseXor),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.<<", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::SpirvOp(SpirvOp::ShiftLeftLogical),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.>>", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: if ty_name.starts_with('i') {
                BuiltinImpl::SpirvOp(SpirvOp::ShiftRightArithmetic)
            } else {
                BuiltinImpl::SpirvOp(SpirvOp::ShiftRightLogical)
            },
        });
    }

    /// Register real (floating-point) module functions
    fn register_real_modules(&mut self) {
        self.register_real_ops("f16");
        self.register_real_ops("f32");
        self.register_real_ops("f64");
    }

    fn register_real_ops(&mut self, ty_name: &'static str) {
        let t = Self::ty(ty_name);
        let bool_t = Self::ty("bool");

        // Transcendental functions
        self.register(BuiltinDescriptor {
            name: format!("{}.sin", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(13),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.cos", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(14),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.tan", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(15),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.asin", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(16),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.acos", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(17),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.atan", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(18),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.sqrt", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(31),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.exp", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(27),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.log", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(28),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.pow", ty_name),
            param_types: vec![t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(26),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.floor", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(8),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.ceil", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(9),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.round", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(1),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.trunc", ty_name),
            param_types: vec![t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(3),
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.clamp", ty_name),
            param_types: vec![t.clone(), t.clone(), t.clone()],
            return_type: t.clone(),
            implementation: BuiltinImpl::GlslExt(43),
        });

        // isnan, isinf
        self.register(BuiltinDescriptor {
            name: format!("{}.isnan", ty_name),
            param_types: vec![t.clone()],
            return_type: bool_t.clone(),
            implementation: BuiltinImpl::GlslExt(66), // IsNan
        });

        self.register(BuiltinDescriptor {
            name: format!("{}.isinf", ty_name),
            param_types: vec![t.clone()],
            return_type: bool_t,
            implementation: BuiltinImpl::GlslExt(67), // IsInf
        });
    }

    /// Register float-specific module functions
    fn register_float_modules(&mut self) {
        // Float-specific operations (bit manipulation, etc.)
        // TODO: Implement f16, f32, f64 specific ops like from_bits, to_bits
    }

    /// Register vector operations (length, normalize, dot, cross, etc.)
    fn register_vector_operations(&mut self) {
        let vec_t = Self::ty("vec"); // Placeholder - should be polymorphic
        let float_t = Self::ty("f32");

        self.register(BuiltinDescriptor {
            name: "length".to_string(),
            param_types: vec![vec_t.clone()],
            return_type: float_t.clone(),
            implementation: BuiltinImpl::GlslExt(66),
        });

        self.register(BuiltinDescriptor {
            name: "normalize".to_string(),
            param_types: vec![vec_t.clone()],
            return_type: vec_t.clone(),
            implementation: BuiltinImpl::GlslExt(69),
        });

        self.register(BuiltinDescriptor {
            name: "dot".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone()],
            return_type: float_t,
            implementation: BuiltinImpl::SpirvOp(SpirvOp::Dot),
        });

        self.register(BuiltinDescriptor {
            name: "cross".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone()],
            return_type: vec_t.clone(),
            implementation: BuiltinImpl::GlslExt(68),
        });

        self.register(BuiltinDescriptor {
            name: "distance".to_string(),
            param_types: vec![vec_t.clone(), vec_t.clone()],
            return_type: Self::ty("f32"),
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
            param_types: vec![vec_t.clone(), vec_t.clone(), Self::ty("f32")],
            return_type: vec_t,
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
    }

    /// Register vector constructor functions
    fn register_vector_constructors(&mut self) {
        let f32_t = Self::ty("f32");
        let i32_t = Self::ty("i32");

        // vec2, vec3, vec4 constructors
        self.register(BuiltinDescriptor {
            name: "vec2".to_string(),
            param_types: vec![f32_t.clone(), f32_t.clone()],
            return_type: Self::ty("vec2"),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        self.register(BuiltinDescriptor {
            name: "vec3".to_string(),
            param_types: vec![f32_t.clone(), f32_t.clone(), f32_t.clone()],
            return_type: Self::ty("vec3"),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        self.register(BuiltinDescriptor {
            name: "vec4".to_string(),
            param_types: vec![f32_t.clone(), f32_t.clone(), f32_t.clone(), f32_t],
            return_type: Self::ty("vec4"),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        // ivec2, ivec3, ivec4 constructors
        self.register(BuiltinDescriptor {
            name: "ivec2".to_string(),
            param_types: vec![i32_t.clone(), i32_t.clone()],
            return_type: Self::ty("ivec2"),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        self.register(BuiltinDescriptor {
            name: "ivec3".to_string(),
            param_types: vec![i32_t.clone(), i32_t.clone(), i32_t.clone()],
            return_type: Self::ty("ivec3"),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });

        self.register(BuiltinDescriptor {
            name: "ivec4".to_string(),
            param_types: vec![i32_t.clone(), i32_t.clone(), i32_t.clone(), i32_t],
            return_type: Self::ty("ivec4"),
            implementation: BuiltinImpl::Custom(CustomImpl::Placeholder),
        });
    }

    /// Register higher-order functions and array operations
    fn register_higher_order_functions(&mut self) {
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
        Self::new()
    }
}
