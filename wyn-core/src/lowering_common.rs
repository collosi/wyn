//! Common utilities shared between lowering backends (SPIR-V, GLSL, etc.)

use crate::ast::TypeName;
use crate::mir::{self, Attribute, Def};
use polytype::Type as PolyType;

/// Shader stage for entry points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

/// An entry point extracted from MIR
#[derive(Debug, Clone)]
pub struct EntryPoint {
    pub name: String,
    pub stage: ShaderStage,
}

/// Extract all entry points from a MIR program
pub fn find_entry_points(program: &mir::Program) -> Vec<EntryPoint> {
    let mut entry_points = Vec::new();

    for def in &program.defs {
        if let Def::Function { name, attributes, .. } = def {
            for attr in attributes {
                match attr {
                    Attribute::Vertex => {
                        entry_points.push(EntryPoint {
                            name: name.clone(),
                            stage: ShaderStage::Vertex,
                        });
                    }
                    Attribute::Fragment => {
                        entry_points.push(EntryPoint {
                            name: name.clone(),
                            stage: ShaderStage::Fragment,
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    entry_points
}

/// Check if a function has entry point attributes
pub fn is_entry_point(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|a| matches!(a, Attribute::Vertex | Attribute::Fragment))
}

/// Check if a type represents an empty closure (no captured variables)
pub fn is_empty_closure_type(ty: &PolyType<TypeName>) -> bool {
    match ty {
        PolyType::Constructed(TypeName::Tuple(_), args) => args.is_empty(),
        PolyType::Constructed(TypeName::Record(fields), _) => {
            // Empty if only field is __lambda_name
            fields.iter().all(|name| name == "__lambda_name")
        }
        _ => false,
    }
}
