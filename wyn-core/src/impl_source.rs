// Implementation source for builtin functions and intrinsics
// Provides code generation implementations (SPIR-V opcodes, intrinsics, GASM functions)
// Types for functions are provided by modules or PolymorphicBuiltins

use crate::ast::{Type, TypeName};
use std::collections::HashMap;

/// Implementation strategy for a builtin function
///
/// Builtins are organized into four semantic categories:
/// 1. Library-level (CoreFn): Can be written in the language itself
/// 2. Core primitives (PrimOp): Map fairly directly to backend operations
/// 3. Genuine intrinsics (Intrinsic): Require backend-specific lowering
/// 4. GASM functions (GasmFn): GPU assembly implementations from builtins/*.gasm
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinImpl {
    /// Library-level builtin: implemented as normal function in prelude/core IR
    /// These can be written in the language itself (or lowered to core IR once)
    /// Examples: f32.sum, replicate, filter
    CoreFn(String), // Function name in the core IR

    /// Core primitive operation: maps fairly directly to SPIR-V/backend ops
    /// Examples: f32.add, i32.mul, dot, matrix multiply
    PrimOp(PrimOp),

    /// Genuine intrinsic: needs backend-specific lowering, can't be written in language
    /// Examples: atomics, barriers, subgroup ops, uninit/poison
    Intrinsic(Intrinsic),

    /// GASM function: GPU assembly implementation loaded from builtins/*.gasm
    /// These are lowered to SPIR-V once and called via OpFunctionCall
    GasmFn(gasm::Function),
}

/// Core primitive operations that map fairly directly to SPIR-V/backend ops
#[derive(Debug, Clone, PartialEq)]
pub enum PrimOp {
    // GLSL.std.450 extended instructions
    GlslExt(u32),

    // Core SPIR-V operations
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
    // Type conversions
    // Float to signed int
    FPToSI,
    // Float to unsigned int
    FPToUI,
    // Signed int to float
    SIToFP,
    // Unsigned int to float
    UIToFP,
    // Float precision conversion
    FPConvert,
    // Signed extension
    SConvert,
    // Unsigned/zero extension
    UConvert,
    // Bitcast (reinterpret bits)
    Bitcast,
}

/// Genuine intrinsics that need backend-specific lowering
/// These cannot be written in the language itself
#[derive(Debug, Clone, PartialEq)]
pub enum Intrinsic {
    /// Placeholder for future implementations (will be desugared or moved)
    Placeholder,
    /// Uninitialized/poison value for allocation bootstrapping
    /// SAFETY: Must be fully overwritten before being read
    Uninit,
    /// Array replication: creates array filled with a value
    /// TODO: Move to prelude as normal function once we have reduce/fold
    Replicate,
    /// Functional array update: immutable copy-with-update
    /// TODO: Move to prelude as normal function
    ArrayUpdate,
    /// Debug output: write i32 to debug ring buffer
    DebugI32,
    /// Debug output: write f32 to debug ring buffer (6 decimal places)
    DebugF32,
    /// Debug output: write string literal to debug ring buffer
    DebugStr,
    // GDP (GPU Debug Protocol) intrinsics for ring buffer access
    /// Atomic add on GDP buffer at index, returns old value
    /// __gdp_atomic_add : u32 -> u32 -> u32 (index, delta -> old_value)
    GdpAtomicAdd,
    /// Load u32 from GDP buffer at index
    /// __gdp_load : u32 -> u32 (index -> value)
    GdpLoad,
    /// Store u32 to GDP buffer at index
    /// __gdp_store : u32 -> u32 -> () (index, value -> ())
    GdpStore,
    /// Bitcast i32 to u32 (reinterpret bits)
    /// __bitcast_i32_to_u32 : i32 -> u32
    BitcastI32ToU32,
}

/// Implementation source for all builtin functions and intrinsics
pub struct ImplSource {
    /// Maps function name to implementation
    impls: HashMap<String, BuiltinImpl>,
    /// GASM functions that can be called directly from SPIR-V lowering
    /// This includes functions with pointer parameters that can't be registered as Wyn builtins
    gasm_functions: HashMap<String, gasm::Function>,
}

impl ImplSource {
    pub fn new() -> Self {
        let mut source = ImplSource {
            impls: HashMap::new(),
            gasm_functions: HashMap::new(),
        };

        source.register_from_prim_module();
        source.register_numeric_modules();
        source.register_integral_modules();
        source.register_real_modules();
        source.register_float_modules();
        source.register_vector_operations();
        source.register_matrix_operations();
        source.register_higher_order_functions();
        source.register_debug_intrinsics();
        source.load_gasm_builtins();

        source
    }

    /// Load all GASM builtin functions from the builtins directory
    fn load_gasm_builtins(&mut self) {
        // Find the builtins directory relative to the crate root
        let builtins_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("builtins");

        if !builtins_dir.exists() {
            eprintln!("Warning: builtins directory not found at {:?}", builtins_dir);
            return;
        }

        // Read all .gasm files in the directory
        let gasm_files = match std::fs::read_dir(&builtins_dir) {
            Ok(entries) => entries,
            Err(e) => {
                eprintln!("Warning: failed to read builtins directory: {}", e);
                return;
            }
        };

        for entry in gasm_files.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gasm") {
                if let Err(e) = self.load_gasm_file(&path) {
                    eprintln!("Warning: failed to load GASM builtin {:?}: {}", path, e);
                }
            }
        }
    }

    /// Load a single GASM file and register its functions as builtins
    fn load_gasm_file(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let content = std::fs::read_to_string(path)?;

        let module = match gasm::parse_module(&content) {
            Ok(m) => m,
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("GASM parse error: {:?}", e),
                ));
            }
        };

        // Check if we got any functions - if the file looks like it should have functions but doesn't, panic
        if module.functions.is_empty() {
            // Check if the file content looks like it should have functions
            if content.contains("func @") {
                panic!(
                    "GASM parser failed silently! File {:?} contains 'func @' but parsed 0 functions. File has {} globals. This is a parser bug.",
                    path.file_name().unwrap_or_default(),
                    module.globals.len()
                );
            }
        }

        for function in module.functions {
            self.register_gasm_function(function);
        }

        Ok(())
    }

    /// Register a GASM function as a builtin
    fn register_gasm_function(&mut self, function: gasm::Function) {
        // Strip "@" prefix from GASM function name
        let name = function.name.strip_prefix('@').unwrap_or(&function.name).to_string();

        // Store the function in gasm_functions map for direct SPIR-V access
        self.gasm_functions.insert(name, function);

        // GASM functions registered for direct access via gasm_functions map
        // Types come from modules or will be handled separately
    }

    /// Check if a name is a registered implementation
    pub fn is_builtin(&self, name: &str) -> bool {
        self.impls.contains_key(name)
    }

    /// Get all implementation names as a HashSet (for use in flattening to exclude from capture)
    pub fn all_names(&self) -> std::collections::HashSet<String> {
        self.impls.keys().cloned().collect()
    }

    /// Get a builtin by name, returning either a single entry or an overload set
    /// Get the implementation for a function by name
    pub fn get(&self, name: &str) -> Option<&BuiltinImpl> {
        self.impls.get(name)
    }

    /// Get a GASM function directly by name (without going through Wyn type system)
    /// This is used for functions with pointer parameters that can't be registered as Wyn builtins
    pub fn get_gasm_function(&self, name: &str) -> Option<&gasm::Function> {
        self.gasm_functions.get(name)
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
    /// Register an implementation for a function
    fn register(&mut self, name: &str, implementation: BuiltinImpl) {
        self.impls.insert(name.to_string(), implementation);
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
    /// Helper to register a binary operation
    fn register_binop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, impl_.clone());

        // Also register as __intrinsic_OP_TYPE for use in prelude modules
        let intrinsic_name = format!("__intrinsic_{}_{}", op, ty_name);
        self.register(&intrinsic_name, impl_);
    }

    /// Helper to register a comparison operation
    fn register_cmp(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, impl_.clone());

        // Also register as __intrinsic_OP_TYPE for use in prelude modules
        let intrinsic_name = format!("__intrinsic_{}_{}", op, ty_name);
        self.register(&intrinsic_name, impl_);
    }

    /// Helper to register a unary operation
    fn register_unop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, impl_.clone());

        // Also register as __intrinsic_OP_TYPE for use in prelude modules
        let intrinsic_name = format!("__intrinsic_{}_{}", op, ty_name);
        self.register(&intrinsic_name, impl_);
    }

    fn register_numeric_ops(&mut self, ty_name: &'static str) {
        let is_float = ty_name.starts_with('f');
        let is_signed = ty_name.starts_with('i') || is_float;

        // Arithmetic operators
        self.register_binop(
            ty_name,
            "+",
            if is_float { BuiltinImpl::PrimOp(PrimOp::FAdd) } else { BuiltinImpl::PrimOp(PrimOp::IAdd) },
        );
        self.register_binop(
            ty_name,
            "-",
            if is_float { BuiltinImpl::PrimOp(PrimOp::FSub) } else { BuiltinImpl::PrimOp(PrimOp::ISub) },
        );
        self.register_binop(
            ty_name,
            "*",
            if is_float { BuiltinImpl::PrimOp(PrimOp::FMul) } else { BuiltinImpl::PrimOp(PrimOp::IMul) },
        );
        self.register_binop(
            ty_name,
            "/",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FDiv)
            } else if is_signed {
                BuiltinImpl::PrimOp(PrimOp::SDiv)
            } else {
                BuiltinImpl::PrimOp(PrimOp::UDiv)
            },
        );
        self.register_binop(
            ty_name,
            "%",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FRem)
            } else if is_signed {
                BuiltinImpl::PrimOp(PrimOp::SRem)
            } else {
                BuiltinImpl::PrimOp(PrimOp::UDiv) // Unsigned uses UMod, but we don't have it, use same as div for now
            },
        );

        // Power operator (only for floats - uses GLSL pow)
        if is_float {
            self.register_binop(ty_name, "**", BuiltinImpl::PrimOp(PrimOp::GlslExt(26))); // Pow
        }

        // Comparison operators
        self.register_cmp(
            ty_name,
            "<",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FOrdLessThan)
            } else if is_signed {
                BuiltinImpl::PrimOp(PrimOp::SLessThan)
            } else {
                BuiltinImpl::PrimOp(PrimOp::ULessThan)
            },
        );
        self.register_cmp(
            ty_name,
            "==",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FOrdEqual)
            } else {
                BuiltinImpl::PrimOp(PrimOp::IEqual)
            },
        );
        self.register_cmp(
            ty_name,
            "!=",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FOrdNotEqual)
            } else {
                BuiltinImpl::PrimOp(PrimOp::INotEqual)
            },
        );
        self.register_cmp(
            ty_name,
            ">",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FOrdGreaterThan)
            } else if is_signed {
                BuiltinImpl::PrimOp(PrimOp::SGreaterThan)
            } else {
                BuiltinImpl::PrimOp(PrimOp::UGreaterThan)
            },
        );
        self.register_cmp(
            ty_name,
            "<=",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FOrdLessThanEqual)
            } else if is_signed {
                BuiltinImpl::PrimOp(PrimOp::SLessThanEqual)
            } else {
                BuiltinImpl::PrimOp(PrimOp::ULessThanEqual)
            },
        );
        self.register_cmp(
            ty_name,
            ">=",
            if is_float {
                BuiltinImpl::PrimOp(PrimOp::FOrdGreaterThanEqual)
            } else if is_signed {
                BuiltinImpl::PrimOp(PrimOp::SGreaterThanEqual)
            } else {
                BuiltinImpl::PrimOp(PrimOp::UGreaterThanEqual)
            },
        );

        // min, max, abs, sign, clamp functions
        if is_float {
            self.register_binop(ty_name, "min", BuiltinImpl::PrimOp(PrimOp::GlslExt(37))); // FMin
            self.register_binop(ty_name, "max", BuiltinImpl::PrimOp(PrimOp::GlslExt(40))); // FMax
            self.register_unop(ty_name, "abs", BuiltinImpl::PrimOp(PrimOp::GlslExt(4))); // FAbs
            self.register_unop(ty_name, "sign", BuiltinImpl::PrimOp(PrimOp::GlslExt(6))); // FSign
            self.register_ternop(ty_name, "clamp", BuiltinImpl::PrimOp(PrimOp::GlslExt(43))); // FClamp
        } else {
            self.register_binop(
                ty_name,
                "min",
                if is_signed {
                    BuiltinImpl::PrimOp(PrimOp::GlslExt(39)) // SMin
                } else {
                    BuiltinImpl::PrimOp(PrimOp::GlslExt(38)) // UMin
                },
            );
            self.register_binop(
                ty_name,
                "max",
                if is_signed {
                    BuiltinImpl::PrimOp(PrimOp::GlslExt(42)) // SMax
                } else {
                    BuiltinImpl::PrimOp(PrimOp::GlslExt(41)) // UMax
                },
            );
            self.register_ternop(
                ty_name,
                "clamp",
                if is_signed {
                    BuiltinImpl::PrimOp(PrimOp::GlslExt(45)) // SClamp
                } else {
                    BuiltinImpl::PrimOp(PrimOp::GlslExt(44)) // UClamp
                },
            );
            if is_signed {
                self.register_unop(ty_name, "abs", BuiltinImpl::PrimOp(PrimOp::GlslExt(5))); // SAbs
                self.register_unop(ty_name, "sign", BuiltinImpl::PrimOp(PrimOp::GlslExt(7))); // SSign
            }
            // Note: abs and sign don't make sense for unsigned types
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
        self.register_binop(ty_name, "&", BuiltinImpl::PrimOp(PrimOp::BitwiseAnd));
        self.register_binop(ty_name, "|", BuiltinImpl::PrimOp(PrimOp::BitwiseOr));
        self.register_binop(ty_name, "^", BuiltinImpl::PrimOp(PrimOp::BitwiseXor));
        self.register_binop(ty_name, "<<", BuiltinImpl::PrimOp(PrimOp::ShiftLeftLogical));
        self.register_binop(
            ty_name,
            ">>",
            if ty_name.starts_with('i') {
                BuiltinImpl::PrimOp(PrimOp::ShiftRightArithmetic)
            } else {
                BuiltinImpl::PrimOp(PrimOp::ShiftRightLogical)
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
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, impl_.clone());

        // Also register as __intrinsic_OP_TYPE for use in prelude modules
        let intrinsic_name = format!("__intrinsic_{}_{}", op, ty_name);
        self.register(&intrinsic_name, impl_);
    }

    /// Helper to register a unary operation with bool return
    fn register_bool_unop(&mut self, ty_name: &'static str, op: &str, impl_: BuiltinImpl) {
        let name = format!("{}.{}", ty_name, op);
        self.register(&name, impl_.clone());

        // Also register as __intrinsic_OP_TYPE for use in prelude modules
        let intrinsic_name = format!("__intrinsic_{}_{}", op, ty_name);
        self.register(&intrinsic_name, impl_);
    }

    fn register_real_ops(&mut self, ty_name: &'static str) {
        // TODO: These should come from a prelude module (e.g., f32.sqrt from the f32 module)
        // rather than being hardcoded in the builtin registry. For now, we register them
        // here so they're available for type checking and code generation.

        // Transcendental functions
        self.register_unop(ty_name, "sin", BuiltinImpl::PrimOp(PrimOp::GlslExt(13)));
        self.register_unop(ty_name, "cos", BuiltinImpl::PrimOp(PrimOp::GlslExt(14)));
        self.register_unop(ty_name, "tan", BuiltinImpl::PrimOp(PrimOp::GlslExt(15)));
        self.register_unop(ty_name, "asin", BuiltinImpl::PrimOp(PrimOp::GlslExt(16)));
        self.register_unop(ty_name, "acos", BuiltinImpl::PrimOp(PrimOp::GlslExt(17)));
        self.register_unop(ty_name, "atan", BuiltinImpl::PrimOp(PrimOp::GlslExt(18)));

        // Hyperbolic functions
        self.register_unop(ty_name, "sinh", BuiltinImpl::PrimOp(PrimOp::GlslExt(19)));
        self.register_unop(ty_name, "cosh", BuiltinImpl::PrimOp(PrimOp::GlslExt(20)));
        self.register_unop(ty_name, "tanh", BuiltinImpl::PrimOp(PrimOp::GlslExt(21)));
        self.register_unop(ty_name, "asinh", BuiltinImpl::PrimOp(PrimOp::GlslExt(22)));
        self.register_unop(ty_name, "acosh", BuiltinImpl::PrimOp(PrimOp::GlslExt(23)));
        self.register_unop(ty_name, "atanh", BuiltinImpl::PrimOp(PrimOp::GlslExt(24)));
        self.register_binop(ty_name, "atan2", BuiltinImpl::PrimOp(PrimOp::GlslExt(25)));

        // Exponential and logarithmic
        self.register_unop(ty_name, "sqrt", BuiltinImpl::PrimOp(PrimOp::GlslExt(31)));
        self.register_unop(ty_name, "rsqrt", BuiltinImpl::PrimOp(PrimOp::GlslExt(32))); // InverseSqrt
        self.register_unop(ty_name, "exp", BuiltinImpl::PrimOp(PrimOp::GlslExt(27)));
        self.register_unop(ty_name, "log", BuiltinImpl::PrimOp(PrimOp::GlslExt(28)));
        self.register_unop(ty_name, "log2", BuiltinImpl::PrimOp(PrimOp::GlslExt(30)));
        self.register_binop(ty_name, "pow", BuiltinImpl::PrimOp(PrimOp::GlslExt(26)));

        // Rounding functions
        self.register_unop(ty_name, "floor", BuiltinImpl::PrimOp(PrimOp::GlslExt(8)));
        self.register_unop(ty_name, "ceil", BuiltinImpl::PrimOp(PrimOp::GlslExt(9)));
        self.register_unop(ty_name, "round", BuiltinImpl::PrimOp(PrimOp::GlslExt(1)));
        self.register_unop(ty_name, "trunc", BuiltinImpl::PrimOp(PrimOp::GlslExt(3)));

        // Misc operations (clamp is registered in register_numeric_ops)
        self.register_ternop(ty_name, "lerp", BuiltinImpl::PrimOp(PrimOp::GlslExt(46))); // FMix
        self.register_ternop(ty_name, "fma", BuiltinImpl::PrimOp(PrimOp::GlslExt(50))); // Fused multiply-add

        // isnan, isinf
        self.register_bool_unop(ty_name, "isnan", BuiltinImpl::PrimOp(PrimOp::GlslExt(66)));
        self.register_bool_unop(ty_name, "isinf", BuiltinImpl::PrimOp(PrimOp::GlslExt(67)));
    }

    /// Register float-specific module functions
    fn register_float_modules(&mut self) {
        self.register_f32_conversions();
        // TODO: Implement f16, f64 specific conversions
    }

    /// Register f32 type conversion builtins (implementations only)
    fn register_f32_conversions(&mut self) {
        // User-facing conversions: f32.i32, f32.u32, etc.
        self.register("f32.i8", BuiltinImpl::PrimOp(PrimOp::SIToFP));
        self.register("f32.i16", BuiltinImpl::PrimOp(PrimOp::SIToFP));
        self.register("f32.i32", BuiltinImpl::PrimOp(PrimOp::SIToFP));
        self.register("f32.i64", BuiltinImpl::PrimOp(PrimOp::SIToFP));
        self.register("f32.u8", BuiltinImpl::PrimOp(PrimOp::UIToFP));
        self.register("f32.u16", BuiltinImpl::PrimOp(PrimOp::UIToFP));
        self.register("f32.u32", BuiltinImpl::PrimOp(PrimOp::UIToFP));
        self.register("f32.u64", BuiltinImpl::PrimOp(PrimOp::UIToFP));

        // Conversions from signed integers to f32 (internal builtins)
        self.register("__builtin_f32_from_i8", BuiltinImpl::PrimOp(PrimOp::SIToFP));
        self.register("__builtin_f32_from_i16", BuiltinImpl::PrimOp(PrimOp::SIToFP));
        self.register("__builtin_f32_from_i32", BuiltinImpl::PrimOp(PrimOp::SIToFP));
        self.register("__builtin_f32_from_i64", BuiltinImpl::PrimOp(PrimOp::SIToFP));

        // Conversions from unsigned integers to f32
        self.register("__builtin_f32_from_u8", BuiltinImpl::PrimOp(PrimOp::UIToFP));
        self.register("__builtin_f32_from_u16", BuiltinImpl::PrimOp(PrimOp::UIToFP));
        self.register("__builtin_f32_from_u32", BuiltinImpl::PrimOp(PrimOp::UIToFP));
        self.register("__builtin_f32_from_u64", BuiltinImpl::PrimOp(PrimOp::UIToFP));

        // Conversions from other floats to f32
        self.register("__builtin_f32_from_f16", BuiltinImpl::PrimOp(PrimOp::FPConvert));
        self.register("__builtin_f32_from_f64", BuiltinImpl::PrimOp(PrimOp::FPConvert));

        // Conversions from f32 to integers
        self.register("__builtin_f32_to_i8", BuiltinImpl::PrimOp(PrimOp::FPToSI));
        self.register("__builtin_f32_to_i16", BuiltinImpl::PrimOp(PrimOp::FPToSI));
        self.register("__builtin_f32_to_i32", BuiltinImpl::PrimOp(PrimOp::FPToSI));
        self.register("__builtin_f32_to_i64", BuiltinImpl::PrimOp(PrimOp::FPToSI));
        self.register("__builtin_f32_to_u8", BuiltinImpl::PrimOp(PrimOp::FPToUI));
        self.register("__builtin_f32_to_u16", BuiltinImpl::PrimOp(PrimOp::FPToUI));
        self.register("__builtin_f32_to_u32", BuiltinImpl::PrimOp(PrimOp::FPToUI));
        self.register("__builtin_f32_to_u64", BuiltinImpl::PrimOp(PrimOp::FPToUI));

        // Bit manipulation: reinterpret bits without conversion
        self.register("__builtin_f32_from_bits", BuiltinImpl::PrimOp(PrimOp::Bitcast));
        self.register("__builtin_f32_to_bits", BuiltinImpl::PrimOp(PrimOp::Bitcast));
    }

    /// Register vector operations (length, normalize, dot, cross, etc.)
    /// Register vector operations (implementations only, types in PolymorphicBuiltins)
    fn register_vector_operations(&mut self) {
        self.register("magnitude", BuiltinImpl::PrimOp(PrimOp::GlslExt(66)));
        self.register("length", BuiltinImpl::Intrinsic(Intrinsic::Placeholder));
        self.register("__uninit", BuiltinImpl::Intrinsic(Intrinsic::Uninit));
        self.register("replicate", BuiltinImpl::Intrinsic(Intrinsic::Replicate));

        self.register("__array_update", BuiltinImpl::Intrinsic(Intrinsic::ArrayUpdate));
        self.register("normalize", BuiltinImpl::PrimOp(PrimOp::GlslExt(69)));
        self.register("dot", BuiltinImpl::PrimOp(PrimOp::Dot));
        self.register("cross", BuiltinImpl::PrimOp(PrimOp::GlslExt(68)));
        self.register("distance", BuiltinImpl::PrimOp(PrimOp::GlslExt(67)));
        self.register("reflect", BuiltinImpl::PrimOp(PrimOp::GlslExt(71)));
        self.register("refract", BuiltinImpl::PrimOp(PrimOp::GlslExt(72)));

        // Float-only operations (no integer variants needed)
        self.register("floor", BuiltinImpl::PrimOp(PrimOp::GlslExt(8)));
        self.register("ceil", BuiltinImpl::PrimOp(PrimOp::GlslExt(9)));
        self.register("fract", BuiltinImpl::PrimOp(PrimOp::GlslExt(10)));
        self.register("mix", BuiltinImpl::PrimOp(PrimOp::GlslExt(46))); // FMix
        self.register("smoothstep", BuiltinImpl::PrimOp(PrimOp::GlslExt(49)));
    }

    /// Register matrix operations (implementations only, types in PolymorphicBuiltins)
    fn register_matrix_operations(&mut self) {
        self.register("determinant", BuiltinImpl::PrimOp(PrimOp::GlslExt(33)));
        self.register("inverse", BuiltinImpl::PrimOp(PrimOp::GlslExt(34)));
        self.register("outer", BuiltinImpl::PrimOp(PrimOp::OuterProduct));

        // Internal multiplication variants (desugared from surface "mul")
        self.register("mul_mat_mat", BuiltinImpl::PrimOp(PrimOp::MatrixTimesMatrix));
        self.register("mul_mat_vec", BuiltinImpl::PrimOp(PrimOp::MatrixTimesVector));
        self.register("mul_vec_mat", BuiltinImpl::PrimOp(PrimOp::VectorTimesMatrix));

        // Surface "mul" overloads (will be desugared to the above variants)
        self.register("mul", BuiltinImpl::Intrinsic(Intrinsic::Placeholder));
    }

    /// Register higher-order functions and array operations
    fn register_higher_order_functions(&mut self) {
        // f32.sum is now implemented in the prelude (prelude/f32.wyn)
        // No need to register it here

        // TODO: map and zip are registered manually in TypeChecker::load_builtins
        // because they involve function types which need more careful handling
    }

    /// Register debug output intrinsics (implementations only)
    /// These write to a ring buffer for shader debugging
    fn register_debug_intrinsics(&mut self) {
        self.register("debug_i32", BuiltinImpl::Intrinsic(Intrinsic::DebugI32));
        self.register("debug_f32", BuiltinImpl::Intrinsic(Intrinsic::DebugF32));
        self.register("debug_str", BuiltinImpl::Intrinsic(Intrinsic::DebugStr));

        // GDP (GPU Debug Protocol) intrinsics for ring buffer access
        // These are low-level primitives used by the GDP Wyn module
        self.register("__gdp_atomic_add", BuiltinImpl::Intrinsic(Intrinsic::GdpAtomicAdd));
        self.register("__gdp_load", BuiltinImpl::Intrinsic(Intrinsic::GdpLoad));
        self.register("__gdp_store", BuiltinImpl::Intrinsic(Intrinsic::GdpStore));
        self.register("__bitcast_i32_to_u32", BuiltinImpl::Intrinsic(Intrinsic::BitcastI32ToU32));
    }
}

impl Default for ImplSource {
    fn default() -> Self {
        Self::new()
    }
}
