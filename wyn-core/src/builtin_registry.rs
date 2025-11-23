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
    /// Matrix construction from array of column vectors
    MatrixFromVectors,
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

/// Unified builtin entry with TypeScheme
/// Used for both monomorphic and polymorphic builtins
#[derive(Debug, Clone)]
pub struct BuiltinEntry {
    /// Type scheme (e.g., "forall a. a -> a")
    /// For monomorphic builtins, this is TypeScheme::Monotype
    pub scheme: TypeScheme,

    /// Code generation implementation
    pub implementation: BuiltinImpl,
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
    /// All builtins: maps name to list of overloads
    /// Each name can have multiple entries with different type schemes (for overloading)
    builtins: HashMap<String, Vec<BuiltinEntry>>,
}

impl BuiltinRegistry {
    pub fn new(ctx: &mut impl TypeVarGenerator) -> Self {
        let mut registry = BuiltinRegistry {
            builtins: HashMap::new(),
        };

        registry.register_from_prim_module();
        registry.register_numeric_modules();
        registry.register_integral_modules();
        registry.register_real_modules();
        registry.register_float_modules();
        registry.register_vector_operations(ctx);
        registry.register_matrix_operations(ctx);
        registry.register_vector_constructors();
        registry.register_higher_order_functions(ctx);
        registry.register_matav_variants();

        registry
    }

    /// Check if a name is a registered builtin
    pub fn is_builtin(&self, name: &str) -> bool {
        self.builtins.contains_key(name)
    }

    /// Get all builtin names as a HashSet (for use in flattening to exclude from capture)
    pub fn all_names(&self) -> std::collections::HashSet<String> {
        self.builtins.keys().cloned().collect()
    }

    /// Get all overloads for a builtin name
    pub fn get_overloads(&self, name: &str) -> Option<&[BuiltinEntry]> {
        self.builtins.get(name).map(|v| v.as_slice())
    }

    /// Add an overload for a builtin
    fn add_overload(&mut self, name: String, entry: BuiltinEntry) {
        self.builtins.entry(name).or_insert_with(Vec::new).push(entry);
    }

    /// Get the type of a field on a given type (e.g., vec3f32.x returns f32)
    pub fn get_field_type(&self, type_name: &str, field_name: &str) -> Option<Type> {
        // Parse vector types like vec2f32, vec3i32, vec4bool
        if type_name.starts_with("vec") {
            // Extract size: vec2f32 -> 2, vec3i32 -> 3, vec4bool -> 4
            let size = type_name.chars().nth(3)?.to_digit(10)? as usize;

            // Check if field is valid for this vector size
            let valid_field = match (size, field_name) {
                (2, "x" | "y") => true,
                (3, "x" | "y" | "z") => true,
                (4, "x" | "y" | "z" | "w") => true,
                _ => false,
            };

            if valid_field {
                // Extract element type: vec2f32 -> f32, vec3i32 -> i32
                let elem_type = type_name[4..].to_string();
                return Some(Type::Constructed(TypeName::Named(elem_type), vec![]));
            }
        }

        None
    }

    /// Register a builtin function (legacy method - converts to TypeScheme::Monotype)
    /// This builds a function type from param_types -> return_type and wraps in Monotype
    fn register(
        &mut self,
        name: &str,
        param_types: Vec<Type>,
        return_type: Type,
        implementation: BuiltinImpl,
    ) {
        // Build function type: param1 -> param2 -> ... -> return_type
        let mut func_type = return_type;
        for param_type in param_types.iter().rev() {
            func_type = Type::arrow(param_type.clone(), func_type);
        }

        let entry = BuiltinEntry {
            scheme: TypeScheme::Monotype(func_type),
            implementation,
        };

        self.add_overload(name.to_string(), entry);
    }

    /// Register a polymorphic builtin with type variables
    /// Automatically wraps type variables in forall quantifiers
    fn register_poly(
        &mut self,
        name: &str,
        param_types: Vec<Type>,
        return_type: Type,
        implementation: BuiltinImpl,
    ) {
        // Build function type: param1 -> param2 -> ... -> return_type
        let mut func_type = return_type;
        for param_type in param_types.iter().rev() {
            func_type = Type::arrow(param_type.clone(), func_type);
        }

        // Collect all type variables in the function type
        let type_vars = func_type.vars();

        // Wrap in forall quantifiers
        let mut scheme = TypeScheme::Monotype(func_type);
        for var in type_vars.into_iter().rev() {
            scheme = TypeScheme::Polytype {
                variable: var,
                body: Box::new(scheme),
            };
        }

        let entry = BuiltinEntry {
            scheme,
            implementation,
        };

        self.add_overload(name.to_string(), entry);
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
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, vec![t.clone(), t.clone()], t, impl_);
    }

    /// Helper to register a comparison operation
    fn register_cmp(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, vec![t.clone(), t.clone()], Self::ty("bool"), impl_);
    }

    /// Helper to register a unary operation
    fn register_unop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, vec![t.clone()], t, impl_);
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
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, vec![t.clone(), t.clone(), t.clone()], t, impl_);
    }

    /// Helper to register a unary operation with bool return
    fn register_bool_unop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let t = Self::ty(ty_name);
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, vec![t], Self::ty("bool"), impl_);
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
        self.register_poly(
            "length",
            vec![arr_t],
            Self::ty("i32"),
            BuiltinImpl::Custom(CustomImpl::Placeholder),
        );

        // __uninit: ∀t. t
        // Returns uninitialized/poison value (must be overwritten before reading)
        let uninit_type_var = ctx.new_variable();
        self.register_poly(
            "__uninit",
            vec![],
            uninit_type_var,
            BuiltinImpl::Custom(CustomImpl::Uninit),
        );

        // replicate: ∀t. i32 -> t -> []t
        // Creates array of unknown length filled with a value (length-agnostic)
        let replicate_elem_var = ctx.new_variable();
        let replicate_result_type = Type::Constructed(
            TypeName::Array,
            vec![ctx.new_variable(), replicate_elem_var.clone()],
        );
        self.register_poly(
            "replicate",
            vec![Self::ty("i32"), replicate_elem_var],
            replicate_result_type,
            BuiltinImpl::Custom(CustomImpl::Replicate),
        );

        // __array_update: ∀n t. [n]t -> i32 -> t -> [n]t
        // Functional array update (immutable copy-with-update)
        let update_size_var = ctx.new_variable();
        let update_elem_var = ctx.new_variable();
        let update_arr_type =
            Type::Constructed(TypeName::Array, vec![update_size_var, update_elem_var.clone()]);
        self.register_poly(
            "__array_update",
            vec![update_arr_type.clone(), Self::ty("i32"), update_elem_var],
            update_arr_type,
            BuiltinImpl::Custom(CustomImpl::ArrayUpdate),
        );

        self.register_poly(
            "normalize",
            vec![vec_t.clone()],
            vec_t.clone(),
            BuiltinImpl::GlslExt(69),
        );

        self.register_poly(
            "dot",
            vec![vec_t.clone(), vec_t.clone()],
            elem_var.clone(),
            BuiltinImpl::SpirvOp(SpirvOp::Dot),
        );

        self.register_poly(
            "cross",
            vec![vec_t.clone(), vec_t.clone()],
            vec_t.clone(),
            BuiltinImpl::GlslExt(68),
        );

        self.register_poly(
            "distance",
            vec![vec_t.clone(), vec_t.clone()],
            elem_var.clone(),
            BuiltinImpl::GlslExt(67),
        );

        self.register_poly(
            "reflect",
            vec![vec_t.clone(), vec_t.clone()],
            vec_t.clone(),
            BuiltinImpl::GlslExt(71),
        );

        self.register_poly(
            "refract",
            vec![vec_t.clone(), vec_t.clone(), elem_var.clone()],
            vec_t.clone(),
            BuiltinImpl::GlslExt(72),
        );
    }

    /// Register matrix operations
    fn register_matrix_operations(&mut self, ctx: &mut impl TypeVarGenerator) {
        let mat_t = Self::ty("mat"); // Placeholder
        let vec_t = Self::ty("vec");

        self.register(
            "determinant",
            vec![mat_t.clone()],
            Self::ty("f32"),
            BuiltinImpl::GlslExt(33),
        );

        self.register(
            "inverse",
            vec![mat_t.clone()],
            mat_t.clone(),
            BuiltinImpl::GlslExt(34),
        );

        self.register(
            "outer",
            vec![vec_t.clone(), vec_t],
            mat_t,
            BuiltinImpl::SpirvOp(SpirvOp::OuterProduct),
        );

        // Internal multiplication variants (desugared from surface "mul")
        // Surface code uses "mul", which is desugared to these based on argument shapes

        // mul_mat_mat : ∀n m a. mat<n,m,a> -> mat<m,p,a> -> mat<n,p,a>
        // For square matrices: mat<n,n,a> -> mat<n,n,a> -> mat<n,n,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let mat_n_m_a = Type::Constructed(TypeName::Mat, vec![n.clone(), m.clone(), a.clone()]);
        self.register_poly(
            "mul_mat_mat",
            vec![mat_n_m_a.clone(), mat_n_m_a.clone()],
            mat_n_m_a,
            BuiltinImpl::SpirvOp(SpirvOp::MatrixTimesMatrix),
        );

        // mul_mat_vec : ∀n m a. mat<n,m,a> -> vec<m,a> -> vec<n,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n.clone(), a.clone()]);
        let mat_n_m = Type::Constructed(TypeName::Mat, vec![n, m, a.clone()]);
        self.register_poly(
            "mul_mat_vec",
            vec![mat_n_m, vec_m_a],
            vec_n_a,
            BuiltinImpl::SpirvOp(SpirvOp::MatrixTimesVector),
        );

        // mul_vec_mat : ∀n m a. vec<n,a> -> mat<n,m,a> -> vec<m,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n.clone(), a.clone()]);
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let mat_n_m = Type::Constructed(TypeName::Mat, vec![n, m, a]);
        self.register_poly(
            "mul_vec_mat",
            vec![vec_n_a, mat_n_m],
            vec_m_a,
            BuiltinImpl::SpirvOp(SpirvOp::VectorTimesMatrix),
        );

        // Surface "mul" overloads (will be desugared to the above variants)
        // These are registered for type checking; desugaring rewrites them before lowering

        // mul : ∀n m a. mat<n,m,a> -> mat<n,m,a> -> mat<n,m,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let mat_n_m_a = Type::Constructed(TypeName::Mat, vec![n.clone(), m.clone(), a.clone()]);
        self.register_poly(
            "mul",
            vec![mat_n_m_a.clone(), mat_n_m_a.clone()],
            mat_n_m_a,
            BuiltinImpl::Custom(CustomImpl::Placeholder), // Will be desugared
        );

        // mul : ∀n m a. mat<n,m,a> -> vec<m,a> -> vec<n,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n.clone(), a.clone()]);
        let mat_n_m = Type::Constructed(TypeName::Mat, vec![n, m, a.clone()]);
        self.register_poly(
            "mul",
            vec![mat_n_m, vec_m_a],
            vec_n_a,
            BuiltinImpl::Custom(CustomImpl::Placeholder), // Will be desugared
        );

        // mul : ∀n m a. vec<n,a> -> mat<n,m,a> -> vec<m,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n.clone(), a.clone()]);
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let mat_n_m = Type::Constructed(TypeName::Mat, vec![n, m, a]);
        self.register_poly(
            "mul",
            vec![vec_n_a, mat_n_m],
            vec_m_a,
            BuiltinImpl::Custom(CustomImpl::Placeholder), // Will be desugared
        );

        // Matrix construction from array of vectors
        // Surface matav: ∀n m a. [n]vec<m,a> -> mat<n,m,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let array_n_vec = Type::Constructed(TypeName::Array, vec![n.clone(), vec_m_a]);
        let mat_n_m_a = Type::Constructed(TypeName::Mat, vec![n, m, a]);
        self.register_poly(
            "matav",
            vec![array_n_vec],
            mat_n_m_a.clone(),
            BuiltinImpl::Custom(CustomImpl::Placeholder), // Will be desugared
        );

        // Internal matav variants (still polymorphic in element type, but concrete in dimensions)
        // These are what the surface matav desugars to
        // We'll generate these on-demand during desugaring, similar to how we handle mul
    }

    /// Helper to register a vector constructor
    fn register_vec_constructor(&mut self, vec_name: &'static str, elem_type: &'static str, arity: usize) {
        let elem_t = Self::ty(elem_type);
        // Return type is Vec[arity, elem_type], not just the name "vec4"
        let return_type = Type::Constructed(
            TypeName::Vec,
            vec![Type::Constructed(TypeName::Size(arity), vec![]), elem_t.clone()],
        );
        self.register(
            vec_name,
            vec![elem_t; arity],
            return_type,
            BuiltinImpl::Custom(CustomImpl::Placeholder),
        );
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
        let size_var = ctx.new_variable();
        let arr_t = Type::Constructed(TypeName::Array, vec![size_var, Self::ty("f32")]);
        self.register_poly(
            "f32.sum",
            vec![arr_t],
            Self::ty("f32"),
            BuiltinImpl::Custom(CustomImpl::Placeholder),
        );

        // TODO: map and zip are registered manually in TypeChecker::load_builtins
        // because they involve function types which need more careful handling
    }

    /// Register concrete matav variants (matrix from array of vectors)
    /// matav_n_m_elem : [n]vec<m,elem> -> mat<n,m,elem>
    fn register_matav_variants(&mut self) {
        let sizes = [2, 3, 4];
        let elem_types = ["f32", "i32"];

        for &n in &sizes {
            for &m in &sizes {
                for &elem_ty in &elem_types {
                    let vec_type = Type::Constructed(
                        TypeName::Vec,
                        vec![Type::Constructed(TypeName::Size(m), vec![]), Self::ty(elem_ty)],
                    );
                    let array_type = Type::Constructed(
                        TypeName::Array,
                        vec![Type::Constructed(TypeName::Size(n), vec![]), vec_type],
                    );
                    let mat_type = Type::Constructed(
                        TypeName::Mat,
                        vec![
                            Type::Constructed(TypeName::Size(n), vec![]),
                            Type::Constructed(TypeName::Size(m), vec![]),
                            Self::ty(elem_ty),
                        ],
                    );
                    self.register(
                        &format!("matav_{}_{}_{}", n, m, elem_ty),
                        vec![array_type],
                        mat_type,
                        BuiltinImpl::Custom(CustomImpl::MatrixFromVectors),
                    );
                }
            }
        }
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        let mut ctx = polytype::Context::<crate::ast::TypeName>::default();
        Self::new(&mut ctx)
    }
}
