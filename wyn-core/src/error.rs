use crate::ast::Span;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Type error at {1}: {0}")]
    TypeError(String, Span),

    #[error("Undefined variable '{0}' at {1}")]
    UndefinedVariable(String, Span),

    #[error("Invalid array index")]
    InvalidArrayIndex,

    #[error("SPIR-V generation error: {0}")]
    SpirvError(String),

    #[error("MIR generation error: {0}")]
    MirError(String),

    #[error("Module system error: {0}")]
    ModuleError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SPIR-V builder error: {0}")]
    SpirvBuilderError(#[from] rspirv::dr::Error),
}

pub type Result<T> = std::result::Result<T, CompilerError>;
