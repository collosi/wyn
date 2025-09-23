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
        Self::fix_load_alignment(&mut module)?;
        
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
    
    /// Fix Load instructions with invalid alignment operands
    /// 
    /// SPIR-V Load instructions don't support alignment operands like LLVM IR does
    fn fix_load_alignment(module: &mut Module) -> Result<()> {
        // Fix Load instructions in all functions
        for function in &mut module.functions {
            for block in &mut function.blocks {
                for instruction in &mut block.instructions {
                    if instruction.class.opcode == spirv::Op::Load {
                        // Remove ALIGNED memory access operands from OpLoad instructions
                        // SPIR-V OpLoad with ALIGNED memory access causes InvalidOperandCount errors
                        if instruction.operands.len() > 1 {
                            // Check if second operand is ALIGNED memory access
                            if let Some(Operand::MemoryAccess(access)) = instruction.operands.get(1) {
                                if access.contains(spirv::MemoryAccess::ALIGNED) {
                                    // Keep only the pointer (first operand)
                                    instruction.operands.truncate(1);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Note: Skipping global variables for now since the field structure differs
        
        Ok(())
    }
    
    /// Helper to fix Load instructions in individual instructions
    fn fix_instruction_load(instruction: &mut rspirv::dr::Instruction) {
        if instruction.class.opcode == spirv::Op::Load && instruction.operands.len() > 3 {
            instruction.operands.truncate(3);
        }
    }
}