use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParseError {
    #[error("Unexpected end of input")]
    UnexpectedEof,

    #[error("Expected {expected}, found {found}")]
    ExpectedToken {
        expected: String,
        found: String,
    },

    #[error("Invalid type: {0}")]
    InvalidType(String),

    #[error("Invalid address space: {0}")]
    InvalidAddressSpace(String),

    #[error("Invalid instruction: {0}")]
    InvalidInstruction(String),

    #[error("Invalid terminator: {0}")]
    InvalidTerminator(String),

    #[error("Invalid literal: {0}")]
    InvalidLiteral(String),

    #[error("Invalid identifier: {0}")]
    InvalidIdentifier(String),

    #[error("Invalid memory ordering: {0}")]
    InvalidMemoryOrdering(String),

    #[error("Invalid memory scope: {0}")]
    InvalidMemoryScope(String),

    #[error("Invalid atomic operation: {0}")]
    InvalidAtomicOp(String),

    #[error("Invalid function attribute: {0}")]
    InvalidFunctionAttr(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Duplicate label: {0}")]
    DuplicateLabel(String),

    #[error("Parse error: {0}")]
    NomError(String),
}

pub type Result<T> = std::result::Result<T, ParseError>;
