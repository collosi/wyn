use crate::ast::Span;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Parse error: {0}")]
    ParseError(String, Option<Span>),

    #[error("Type error: {0}")]
    TypeError(String, Option<Span>),

    #[error("Undefined variable '{0}'")]
    UndefinedVariable(String, Option<Span>),

    #[error("SPIR-V generation error: {0}")]
    SpirvError(String, Option<Span>),

    #[error("Module system error: {0}")]
    ModuleError(String, Option<Span>),

    #[error("Flattening error: {0}")]
    FlatteningError(String, Option<Span>),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SPIR-V builder error: {0}")]
    SpirvBuilderError(#[from] rspirv::dr::Error),
}

impl CompilerError {
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::ParseError(_, span) => *span,
            Self::TypeError(_, span) => *span,
            Self::UndefinedVariable(_, span) => *span,
            Self::SpirvError(_, span) => *span,
            Self::ModuleError(_, span) => *span,
            Self::FlatteningError(_, span) => *span,
            Self::IoError(_) | Self::SpirvBuilderError(_) => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, CompilerError>;

// Bail macros without span

#[macro_export]
macro_rules! bail_parse {
    ($($arg:tt)*) => {
        return Err($crate::error::CompilerError::ParseError(format!($($arg)*), None))
    };
}

#[macro_export]
macro_rules! bail_type {
    ($($arg:tt)*) => {
        return Err($crate::error::CompilerError::TypeError(format!($($arg)*), None))
    };
}

#[macro_export]
macro_rules! bail_spirv {
    ($($arg:tt)*) => {
        return Err($crate::error::CompilerError::SpirvError(format!($($arg)*), None))
    };
}

#[macro_export]
macro_rules! bail_module {
    ($($arg:tt)*) => {
        return Err($crate::error::CompilerError::ModuleError(format!($($arg)*), None))
    };
}

#[macro_export]
macro_rules! bail_flatten {
    ($($arg:tt)*) => {
        return Err($crate::error::CompilerError::FlatteningError(format!($($arg)*), None))
    };
}

// Bail macros with span

#[macro_export]
macro_rules! bail_parse_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::error::CompilerError::ParseError(format!($($arg)*), Some($span)))
    };
}

#[macro_export]
macro_rules! bail_type_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::error::CompilerError::TypeError(format!($($arg)*), Some($span)))
    };
}

#[macro_export]
macro_rules! bail_undef_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::error::CompilerError::UndefinedVariable(format!($($arg)*), Some($span)))
    };
}

#[macro_export]
macro_rules! bail_spirv_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::error::CompilerError::SpirvError(format!($($arg)*), Some($span)))
    };
}

#[macro_export]
macro_rules! bail_module_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::error::CompilerError::ModuleError(format!($($arg)*), Some($span)))
    };
}

#[macro_export]
macro_rules! bail_flatten_at {
    ($span:expr, $($arg:tt)*) => {
        return Err($crate::error::CompilerError::FlatteningError(format!($($arg)*), Some($span)))
    };
}
