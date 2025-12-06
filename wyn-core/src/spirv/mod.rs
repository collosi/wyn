//! SPIR-V code generation backend.
//!
//! This module contains the lowering pass from MIR to SPIR-V.

pub mod lowering;

pub use lowering::lower;
