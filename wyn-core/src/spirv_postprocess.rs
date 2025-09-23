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
    /// Takes raw SPIR-V words from LLVM and applies transformations.
    /// Each transformation phase is independent and can be disabled by commenting out
    /// the respective function call.
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
        
        // Apply transformation phases (each can be independently disabled)
        
        // Phase 1: Remove Linkage capability and LinkageAttributes decorations
        Self::remove_linkage_support(&mut module)?;
        
        // Phase 2: Fix Load instruction alignment operands
        Self::fix_load_alignment(&mut module)?;
        
        // Phase 3: Add missing OpEntryPoint declarations
        Self::add_entry_points(&mut module)?;
        
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
    
    /// Add missing OpEntryPoint declarations
    /// 
    /// LLVM SPIR-V backend may not generate proper entry points, so we add them manually
    fn add_entry_points(module: &mut Module) -> Result<()> {
        use rspirv::dr::Instruction;
        
        // Look for functions named vertex_main and fragment_main
        let mut vertex_id = None;
        let mut fragment_id = None;
        
        // Find entry point function IDs by looking at debug names
        for debug_name in &module.debug_names {
            if debug_name.class.opcode == spirv::Op::Name {
                if let (Some(Operand::IdRef(func_id)), Some(Operand::LiteralString(name))) = 
                    (debug_name.operands.get(0), debug_name.operands.get(1)) {
                    match name.as_str() {
                        "vertex_main" => vertex_id = Some(*func_id),
                        "fragment_main" => fragment_id = Some(*func_id),
                        _ => {}
                    }
                }
            }
        }
        
        // Add OpEntryPoint instructions
        if let Some(vertex_id) = vertex_id {
            let entry_point = Instruction::new(
                spirv::Op::EntryPoint,
                None,
                None,
                vec![
                    Operand::ExecutionModel(spirv::ExecutionModel::Vertex),
                    Operand::IdRef(vertex_id),
                    Operand::LiteralString("vertex_main".to_string()),
                ]
            );
            module.entry_points.push(entry_point);
            println!("DEBUG: Added vertex entry point for function ID {}", vertex_id);
        }
        
        if let Some(fragment_id) = fragment_id {
            let entry_point = Instruction::new(
                spirv::Op::EntryPoint,
                None,
                None,
                vec![
                    Operand::ExecutionModel(spirv::ExecutionModel::Fragment),
                    Operand::IdRef(fragment_id),
                    Operand::LiteralString("fragment_main".to_string()),
                ]
            );
            module.entry_points.push(entry_point);
            println!("DEBUG: Added fragment entry point for function ID {}", fragment_id);
        }
        
        Ok(())
    }
    
    /// Helper to fix Load instructions in individual instructions
    fn fix_instruction_load(instruction: &mut rspirv::dr::Instruction) {
        if instruction.class.opcode == spirv::Op::Load && instruction.operands.len() > 3 {
            instruction.operands.truncate(3);
        }
    }
}