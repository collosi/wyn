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

        // GLSL extended instructions (GLSL.std.450)
        registry.insert("sin".to_string(), BuiltinType::ExtInstruction(13)); // GLSL Sin
        registry.insert("cos".to_string(), BuiltinType::ExtInstruction(14)); // GLSL Cos
        registry.insert("tan".to_string(), BuiltinType::ExtInstruction(15)); // GLSL Tan
        registry.insert("sqrt".to_string(), BuiltinType::ExtInstruction(31)); // GLSL Sqrt
        registry.insert("abs".to_string(), BuiltinType::ExtInstruction(4)); // GLSL FAbs
        registry.insert("floor".to_string(), BuiltinType::ExtInstruction(8)); // GLSL Floor
        registry.insert("ceil".to_string(), BuiltinType::ExtInstruction(9)); // GLSL Ceil

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
            Some(BuiltinType::SpirvIntrinsic(_)) => {
                // Handle other SPIR-V intrinsics if needed
                Err(CompilerError::SpirvError(format!(
                    "SPIR-V intrinsic {} not yet implemented",
                    func_name
                )))
            }
            None => Err(CompilerError::SpirvError(format!(
                "Unknown builtin function: {}",
                func_name
            ))),
        }
    }

    /// Get type signature for builtin functions
    pub fn get_builtin_type_signature(
        &self,
        name: &str,
        _element_type: Option<&Type>,
    ) -> Result<(Vec<Type>, Type)> {
        match name {
            "sin" | "cos" | "tan" | "sqrt" | "abs" | "floor" | "ceil" => {
                // These functions take float and return float
                let float_type = Type::Constructed(TypeName::Str("float"), vec![]);
                Ok((vec![float_type.clone()], float_type))
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
