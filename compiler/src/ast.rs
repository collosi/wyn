#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Let(LetDecl),
    Entry(EntryDecl),
    Def(DefDecl),
    Val(ValDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetDecl {
    pub name: String,
    pub ty: Option<Type>,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Type,
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DefDecl {
    pub name: String,
    pub params: Vec<String>, // Parameter names without explicit types (for inference)
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValDecl {
    pub name: String,
    pub size_params: Vec<String>,    // Size parameters: [n], [m]
    pub type_params: Vec<String>,    // Type parameters: 'a, 'b
    pub ty: Type,                    // The function type signature
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    I32,
    F32,
    Vec4F32, // 4-component vector for SPIR-V gl_Position
    Array(Box<Type>, Vec<usize>),
    Tuple(Vec<Type>),
    Var(String), // Type variable for inference
    Function(Box<Type>, Box<Type>), // Function type: arg -> result
    SizeVar(String), // Size variable for array dimensions
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    IntLiteral(i32),
    FloatLiteral(f32),
    Identifier(String),
    ArrayLiteral(Vec<Expression>),
    ArrayIndex(Box<Expression>, Box<Expression>),
    BinaryOp(BinaryOp, Box<Expression>, Box<Expression>),
    FunctionCall(String, Vec<Expression>),
    Tuple(Vec<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Divide,
}

impl Type {
    pub fn element_type(&self) -> Option<Type> {
        match self {
            Type::Array(elem_ty, _) => Some(elem_ty.as_ref().clone()),
            _ => None,
        }
    }
    
    pub fn dimensions(&self) -> Vec<usize> {
        match self {
            Type::Array(_, dims) => dims.clone(),
            _ => vec![],
        }
    }
}