use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Type error: {0}")]
    TypeError(String),
    
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),
    
    #[error("Invalid array index")]
    InvalidArrayIndex,
    
    #[error("SPIR-V generation error: {0}")]
    SpirvError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CompilerError>;