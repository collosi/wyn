use crate::error::{CompilerError, Result};
use rspirv::dr::{Module, Operand, Loader};
use rspirv::binary::{Assemble, parse_bytes};

/// SPIR-V post-processor using rspirv for high-level manipulation
/// 
/// This module provides a clean way to modify SPIR-V after LLVM generation
/// to fix compatibility issues with various consumers like wgpu.
pub struct SpirvPostProcessor;

impl SpirvPostProcessor {
    /// Post-process SPIR-V to fix compatibility issues
    /// 
    /// Takes raw SPIR-V words from LLVM and applies transformations:
    /// - Remove Linkage capability and LinkageAttributes decorations
    /// - Fix any other compatibility issues
    /// 
    /// Returns the modified SPIR-V as words
    pub fn process(spirv_words: Vec<u32>) -> Result<Vec<u32>> {
        // Convert words to bytes for parsing
        let bytes: Vec<u8> = spirv_words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        
        // Parse SPIR-V into rspirv Module using Loader
        let mut loader = Loader::new();
        parse_bytes(&bytes, &mut loader)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to parse SPIR-V: {:?}", e)))?;
        let mut module = loader.module();
        
        // Apply transformations
        Self::remove_linkage_support(&mut module)?;
        
        // Convert back to binary
        let binary = module.assemble();
        Ok(binary)
    }
    
    /// Remove Linkage capability and related decorations
    /// 
    /// This fixes compatibility with wgpu which doesn't support the Linkage capability
    fn remove_linkage_support(module: &mut Module) -> Result<()> {
        // Remove Linkage capability
        module.capabilities.retain(|cap| {
            if let Some(Operand::Capability(capability)) = cap.operands.get(0) {
                *capability != spirv::Capability::Linkage
            } else {
                true
            }
        });
        
        // Remove LinkageAttributes decorations
        module.annotations.retain(|annotation| {
            if annotation.class.opcode == spirv::Op::Decorate {
                if let Some(Operand::Decoration(decoration)) = annotation.operands.get(1) {
                    return *decoration != spirv::Decoration::LinkageAttributes;
                }
            }
            true
        });
        
        Ok(())
    }
}