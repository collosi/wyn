// Control Flow Graph basic types used by the Nemo-based analysis
// The actual CFG extraction is handled by CfgNemoExtractor in cfg_nemo.rs

/// Basic block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

/// Location within a basic block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location {
    pub block: BlockId,
    pub index: usize,  // Position within the block
}