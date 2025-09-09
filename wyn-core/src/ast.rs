pub use polytype::{Type, TypeScheme};
pub use spirv;

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
pub enum Attribute {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
    Vertex,
    Fragment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttributedType {
    pub attributes: Vec<Attribute>,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetDecl {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub ty: Option<Type>,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: AttributedType,
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DefDecl {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub params: Vec<String>, // Parameter names without explicit types (for inference)
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValDecl {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub ty: Type,                 // The function type signature
}

// We now use polytype::Type instead of our own Type enum

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
    Lambda(LambdaExpr),
    Application(Box<Expression>, Vec<Expression>), // Function application
    LetIn(LetInExpr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambdaParam {
    pub name: String,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambdaExpr {
    pub params: Vec<LambdaParam>,
    pub return_type: Option<Type>,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetInExpr {
    pub name: String,
    pub ty: Option<Type>,
    pub value: Box<Expression>,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Divide,
    Add,
}

// Helper module for creating common polytype Types
pub mod types {
    use polytype::Type;

    pub fn i32() -> Type {
        Type::Constructed("int", vec![])
    }

    pub fn f32() -> Type {
        Type::Constructed("float", vec![])
    }


    pub fn array(elem_type: Type) -> Type {
        Type::Constructed("array", vec![elem_type])
    }

    pub fn tuple(types: Vec<Type>) -> Type {
        Type::Constructed("tuple", types)
    }

    pub fn function(arg: Type, ret: Type) -> Type {
        Type::arrow(arg, ret)
    }
}
