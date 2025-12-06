//! GLSL code generation backend.
//!
//! This module contains the lowering pass from MIR to GLSL.

pub mod lowering;

pub use lowering::{GlslOutput, lower};
