pub use polytype::TypeScheme;
pub use spirv;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeName {
    Str(&'static str),          // "int", "float", "tuple"
    Array(&'static str, usize), // "array" with size
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TypeName::Str(s) => write!(f, "{}", s),
            TypeName::Array(s, size) => write!(f, "{}@{}", s, size),
        }
    }
}

impl std::str::FromStr for TypeName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(at_pos) = s.find('@') {
            let (name, size_str) = s.split_at(at_pos);
            let size_str = &size_str[1..]; // Skip the '@'
            let size = size_str
                .parse::<usize>()
                .map_err(|_| format!("Invalid array size: {}", size_str))?;
            // We need to leak the string to get &'static str for now
            let leaked_name = Box::leak(name.to_string().into_boxed_str());
            Ok(TypeName::Array(leaked_name, size))
        } else {
            // We need to leak the string to get &'static str for now
            let leaked_str = Box::leak(s.to_string().into_boxed_str());
            Ok(TypeName::Str(leaked_str))
        }
    }
}

impl polytype::Name for TypeName {
    fn arrow() -> Self {
        TypeName::Str("->")
    }
}

impl From<&'static str> for TypeName {
    fn from(s: &'static str) -> Self {
        TypeName::Str(s)
    }
}

pub type Type = polytype::Type<TypeName>;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Decl(Decl),       // Unified let/def declarations
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
pub struct Decl {
    pub keyword: &'static str,          // Either "let" or "def"
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub params: Vec<DeclParam>,         // Parameters - can be typed or untyped
    pub ty: Option<Type>,                // Return type for functions or type annotation for variables
    pub return_attributes: Vec<Attribute>, // Attributes on the return type (for entry points)
    pub body: Expression,                // The value/expression for let/def declarations
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeclParam {
    Untyped(String),                    // Just a name for regular functions
    Typed(Parameter),                   // Full parameter with type and attributes for entry points
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub ty: Type,
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
    use super::{Type, TypeName};

    pub fn i32() -> Type {
        Type::Constructed(TypeName::Str("int"), vec![])
    }

    pub fn f32() -> Type {
        Type::Constructed(TypeName::Str("float"), vec![])
    }

    // f32 vector types
    pub fn vec2() -> Type {
        Type::Constructed(TypeName::Str("vec2"), vec![])
    }
    
    pub fn vec3() -> Type {
        Type::Constructed(TypeName::Str("vec3"), vec![])
    }
    
    pub fn vec4() -> Type {
        Type::Constructed(TypeName::Str("vec4"), vec![])
    }

    // i32 vector types
    pub fn ivec2() -> Type {
        Type::Constructed(TypeName::Str("ivec2"), vec![])
    }
    
    pub fn ivec3() -> Type {
        Type::Constructed(TypeName::Str("ivec3"), vec![])
    }
    
    pub fn ivec4() -> Type {
        Type::Constructed(TypeName::Str("ivec4"), vec![])
    }

    // u32 vector types
    pub fn uvec2() -> Type {
        Type::Constructed(TypeName::Str("uvec2"), vec![])
    }
    
    pub fn uvec3() -> Type {
        Type::Constructed(TypeName::Str("uvec3"), vec![])
    }
    
    pub fn uvec4() -> Type {
        Type::Constructed(TypeName::Str("uvec4"), vec![])
    }

    // bool vector types
    pub fn bvec2() -> Type {
        Type::Constructed(TypeName::Str("bvec2"), vec![])
    }
    
    pub fn bvec3() -> Type {
        Type::Constructed(TypeName::Str("bvec3"), vec![])
    }
    
    pub fn bvec4() -> Type {
        Type::Constructed(TypeName::Str("bvec4"), vec![])
    }

    // f64 vector types  
    pub fn dvec2() -> Type {
        Type::Constructed(TypeName::Str("dvec2"), vec![])
    }
    
    pub fn dvec3() -> Type {
        Type::Constructed(TypeName::Str("dvec3"), vec![])
    }
    
    pub fn dvec4() -> Type {
        Type::Constructed(TypeName::Str("dvec4"), vec![])
    }

    // f16 vector types
    pub fn f16vec2() -> Type {
        Type::Constructed(TypeName::Str("f16vec2"), vec![])
    }
    
    pub fn f16vec3() -> Type {
        Type::Constructed(TypeName::Str("f16vec3"), vec![])
    }
    
    pub fn f16vec4() -> Type {
        Type::Constructed(TypeName::Str("f16vec4"), vec![])
    }

    pub fn sized_array(size: usize, elem_type: Type) -> Type {
        Type::Constructed(TypeName::Array("array", size), vec![elem_type])
    }

    pub fn tuple(types: Vec<Type>) -> Type {
        Type::Constructed(TypeName::Str("tuple"), types)
    }

    pub fn function(arg: Type, ret: Type) -> Type {
        Type::arrow(arg, ret)
    }
}
