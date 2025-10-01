use crate::ast::{Type, TypeName};
use crate::codegen::Value;
use crate::error::{CompilerError, Result};
use log::debug;
use rspirv::dr::Builder;
use rspirv::spirv;
use std::collections::HashMap;

/// Types of builtin implementations
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinType {
    /// SPIR-V intrinsic instruction
    SpirvIntrinsic(String),
    /// Extended instruction set operation (GLSL.std.450)
    ExtInstruction(u32), // Instruction number in GLSL.std.450
}

/// Manages builtin function implementations using rspirv
pub struct BuiltinManager {
    builtin_registry: HashMap<String, BuiltinType>, // Maps builtin name to implementation type
    glsl_ext_inst_id: Option<spirv::Word>,          // ID for GLSL.std.450 extension
}

impl Default for BuiltinManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BuiltinManager {
    pub fn new() -> Self {
        let mut registry = HashMap::new();

        // Register builtin functions with their implementation types

        // GLSL extended instructions (GLSL.std.450) - Basic math
        registry.insert("sin".to_string(), BuiltinType::ExtInstruction(13)); // GLSL Sin
        registry.insert("cos".to_string(), BuiltinType::ExtInstruction(14)); // GLSL Cos
        registry.insert("tan".to_string(), BuiltinType::ExtInstruction(15)); // GLSL Tan
        registry.insert("sqrt".to_string(), BuiltinType::ExtInstruction(31)); // GLSL Sqrt
        registry.insert("abs".to_string(), BuiltinType::ExtInstruction(4)); // GLSL FAbs
        registry.insert("floor".to_string(), BuiltinType::ExtInstruction(8)); // GLSL Floor
        registry.insert("ceil".to_string(), BuiltinType::ExtInstruction(9)); // GLSL Ceil

        // GLSL extended instructions - Geometric operations
        registry.insert("length".to_string(), BuiltinType::ExtInstruction(66)); // GLSL Length
        registry.insert("distance".to_string(), BuiltinType::ExtInstruction(67)); // GLSL Distance
        registry.insert("cross".to_string(), BuiltinType::ExtInstruction(68)); // GLSL Cross
        registry.insert("normalize".to_string(), BuiltinType::ExtInstruction(69)); // GLSL Normalize
        registry.insert("faceforward".to_string(), BuiltinType::ExtInstruction(70)); // GLSL FaceForward
        registry.insert("reflect".to_string(), BuiltinType::ExtInstruction(71)); // GLSL Reflect
        registry.insert("refract".to_string(), BuiltinType::ExtInstruction(72)); // GLSL Refract

        // GLSL extended instructions - Matrix operations
        registry.insert("determinant".to_string(), BuiltinType::ExtInstruction(33)); // GLSL Determinant
        registry.insert("inverse".to_string(), BuiltinType::ExtInstruction(34)); // GLSL MatrixInverse

        // Core SPIR-V instructions
        registry.insert("dot".to_string(), BuiltinType::SpirvIntrinsic("OpDot".to_string()));
        registry.insert("outer".to_string(), BuiltinType::SpirvIntrinsic("OpOuterProduct".to_string()));

        Self {
            builtin_registry: registry,
            glsl_ext_inst_id: None,
        }
    }

    /// Load builtin functions into a module
    pub fn load_builtins_into_module(&mut self, builder: &mut Builder) -> Result<()> {
        // Import GLSL.std.450 extension instruction set
        self.glsl_ext_inst_id = Some(builder.ext_inst_import("GLSL.std.450"));

        debug!("Imported GLSL.std.450 extension instruction set");
        Ok(())
    }

    /// Check if a function name represents a builtin
    pub fn is_builtin(&self, name: &str) -> bool {
        self.builtin_registry.contains_key(name)
    }

    /// Get the builtin type for a given name
    pub fn get_builtin_type(&self, name: &str) -> Option<&BuiltinType> {
        self.builtin_registry.get(name)
    }

    /// Generate a call to a builtin function
    pub fn generate_builtin_call(
        &mut self,
        builder: &mut Builder,
        func_name: &str,
        args: &[Value],
    ) -> Result<Value> {
        match self.builtin_registry.get(func_name) {
            Some(BuiltinType::ExtInstruction(inst_num)) => {
                if args.is_empty() {
                    return Err(CompilerError::SpirvError(format!(
                        "{} builtin requires at least one argument",
                        func_name
                    )));
                }

                let glsl_ext_id = self.glsl_ext_inst_id.ok_or_else(|| {
                    CompilerError::SpirvError("GLSL.std.450 extension not imported".to_string())
                })?;

                // For most GLSL functions, the result type is the same as the first argument
                let result_type = args[0].type_id;
                let arg_operands: Vec<rspirv::dr::Operand> =
                    args.iter().map(|v| rspirv::dr::Operand::IdRef(v.id)).collect();

                let result_id =
                    builder.ext_inst(result_type, None, glsl_ext_id, *inst_num, arg_operands)?;

                Ok(Value {
                    id: result_id,
                    type_id: result_type,
                })
            }
            Some(BuiltinType::SpirvIntrinsic(op_name)) => {
                match op_name.as_str() {
                    "OpDot" => {
                        // dot(vec, vec) -> scalar
                        if args.len() != 2 {
                            return Err(CompilerError::SpirvError(
                                "dot requires exactly 2 vector arguments".to_string(),
                            ));
                        }
                        // Result type is the component type of the vector
                        // For now, assume f32 (will need proper type introspection later)
                        let result_type = builder.type_float(32);
                        let result_id = builder.dot(result_type, None, args[0].id, args[1].id)?;
                        Ok(Value {
                            id: result_id,
                            type_id: result_type,
                        })
                    }
                    "OpOuterProduct" => {
                        // outer(vec, vec) -> matrix
                        if args.len() != 2 {
                            return Err(CompilerError::SpirvError(
                                "outer requires exactly 2 vector arguments".to_string(),
                            ));
                        }
                        // Result type is a matrix - for now we'll need type inference
                        // This is tricky without full type information
                        return Err(CompilerError::SpirvError(
                            "outer product not yet fully implemented - needs matrix type support".to_string(),
                        ));
                    }
                    _ => Err(CompilerError::SpirvError(format!(
                        "SPIR-V intrinsic {} not yet implemented",
                        op_name
                    ))),
                }
            }
            None => Err(CompilerError::SpirvError(format!(
                "Unknown builtin function: {}",
                func_name
            ))),
        }
    }

    /// Get type signature for builtin functions
    /// Note: This is a simplified type checker. Actual types are polymorphic and work
    /// with vec2, vec3, vec4, etc. Proper type checking should be done during compilation.
    pub fn get_builtin_type_signature(
        &self,
        name: &str,
        _element_type: Option<&Type>,
    ) -> Result<(Vec<Type>, Type)> {
        let float_type = Type::Constructed(TypeName::Str("float"), vec![]);

        // For now, use a generic "vec" type as a placeholder
        // In practice, the actual vector size will be inferred from usage
        let vec_type = Type::Constructed(TypeName::Str("vec"), vec![]);

        match name {
            // Basic math: float -> float
            "sin" | "cos" | "tan" | "sqrt" | "abs" | "floor" | "ceil" => {
                Ok((vec![float_type.clone()], float_type))
            }
            // Vector operations that return scalar
            "length" => {
                // length(vecN) -> float (works for vec2, vec3, vec4)
                Ok((vec![vec_type.clone()], float_type))
            }
            "distance" | "dot" => {
                // distance(vecN, vecN) -> float
                // dot(vecN, vecN) -> float
                // Works for vec2, vec3, vec4
                Ok((vec![vec_type.clone(), vec_type.clone()], float_type))
            }
            // Vector operations that return vector
            "normalize" => {
                // normalize(vecN) -> vecN (preserves type)
                Ok((vec![vec_type.clone()], vec_type))
            }
            "cross" => {
                // cross(vec3, vec3) -> vec3 (only for vec3)
                // TODO: Should validate vec3 specifically
                Ok((vec![vec_type.clone(), vec_type.clone()], vec_type))
            }
            "reflect" => {
                // reflect(vecN, vecN) -> vecN
                Ok((vec![vec_type.clone(), vec_type.clone()], vec_type))
            }
            "refract" => {
                // refract(vecN, vecN, float) -> vecN
                Ok((vec![vec_type.clone(), vec_type.clone(), float_type], vec_type))
            }
            "faceforward" => {
                // faceforward(vecN, vecN, vecN) -> vecN
                Ok((vec![vec_type.clone(), vec_type.clone(), vec_type.clone()], vec_type))
            }
            // Matrix operations
            "determinant" => {
                // determinant(matNxN) -> float
                // TODO: Need matrix type support
                Err(CompilerError::TypeError("determinant needs matrix type support".to_string()))
            }
            "inverse" => {
                // inverse(matNxN) -> matNxN
                // TODO: Need matrix type support
                Err(CompilerError::TypeError("inverse needs matrix type support".to_string()))
            }
            "outer" => {
                // outer(vecN, vecM) -> matNxM
                // TODO: Need matrix type support
                Err(CompilerError::TypeError("outer needs matrix type support".to_string()))
            }
            _ => Err(CompilerError::TypeError(format!("Unknown builtin: {}", name))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_manager_creation() {
        let manager = BuiltinManager::new();

        assert!(manager.is_builtin("sin"));
        assert!(manager.is_builtin("cos"));
        assert!(manager.is_builtin("tan"));
        assert!(manager.is_builtin("sqrt"));
        assert!(!manager.is_builtin("unknown"));
    }

    #[test]
    fn test_builtin_type_signatures() {
        let manager = BuiltinManager::new();

        let (args, ret) = manager.get_builtin_type_signature("sin", None).unwrap();
        assert_eq!(args.len(), 1);
        match &args[0] {
            Type::Constructed(TypeName::Str("float"), _) => (),
            _ => panic!("Expected float argument type"),
        }
        match &ret {
            Type::Constructed(TypeName::Str("float"), _) => (),
            _ => panic!("Expected float return type"),
        }
    }
}
