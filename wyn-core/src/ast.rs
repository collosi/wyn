pub use polytype::TypeScheme;
pub use spirv;

/// Unique identifier for AST nodes (expressions)
/// Used to look up inferred types in the type table
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn new(id: u32) -> Self {
        NodeId(id)
    }
}

impl From<u32> for NodeId {
    fn from(value: u32) -> Self {
        NodeId(value)
    }
}

/// Counter for generating unique node IDs across compilation phases
#[derive(Debug, Clone)]
pub struct NodeCounter {
    next_id: u32,
}

impl NodeCounter {
    pub fn new() -> Self {
        NodeCounter { next_id: 0 }
    }

    pub fn next(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        NodeId(id)
    }

    pub fn mk_node<T>(&mut self, kind: T) -> Node<T> {
        Node {
            h: Header { id: self.next() },
            kind,
        }
    }
}

impl Default for NodeCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub id: NodeId,
    //pub span: Span,          // or Option<Span>
    // hygiene, source file id, etc.
}

#[derive(Clone, Debug)]
pub struct Node<T> {
    pub h: Header,
    pub kind: T,
}

impl<T> PartialEq for Node<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
pub type Expression = Node<ExprKind>;

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
            let size =
                size_str.parse::<usize>().map_err(|_| format!("Invalid array size: {}", size_str))?;
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
    Decl(Decl), // Unified let/def declarations
    Val(ValDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
    Vertex,
    Fragment,
    Uniform,
}

impl Attribute {
    pub fn is_vertex(&self) -> bool {
        matches!(self, Attribute::Vertex)
    }
    pub fn is_fragment(&self) -> bool {
        matches!(self, Attribute::Fragment)
    }
}

pub trait AttrExt {
    fn has<F: Fn(&Attribute) -> bool>(&self, pred: F) -> bool;
    fn first_builtin(&self) -> Option<spirv::BuiltIn>;
    fn first_location(&self) -> Option<u32>;
}

impl AttrExt for [Attribute] {
    fn has<F: Fn(&Attribute) -> bool>(&self, pred: F) -> bool {
        self.iter().any(pred)
    }
    fn first_builtin(&self) -> Option<spirv::BuiltIn> {
        self.iter().find_map(|a| if let Attribute::BuiltIn(b) = a { Some(*b) } else { None })
    }
    fn first_location(&self) -> Option<u32> {
        self.iter().find_map(|a| if let Attribute::Location(l) = a { Some(*l) } else { None })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttributedType {
    pub attributes: Vec<Attribute>,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decl {
    pub keyword: &'static str, // Either "let" or "def"
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub params: Vec<DeclParam>,            // Parameters - can be typed or untyped
    pub ty: Option<Type>,                  // Return type for functions or type annotation for variables
    pub return_attributes: Vec<Attribute>, // Attributes on the return type (for entry points)
    pub attributed_return_type: Option<Vec<AttributedType>>, // For multiple outputs with per-component attributes
    pub body: Expression, // The value/expression for let/def declarations
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeclParam {
    Untyped(String),  // Just a name for regular functions
    Typed(Parameter), // Full parameter with type and attributes for entry points
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
pub enum ExprKind {
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
    FieldAccess(Box<Expression>, String), // e.g. v.x, v.y
    If(IfExpr),                           // if-then-else expression
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

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOp {
    pub op: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr {
    pub condition: Box<Expression>,
    pub then_branch: Box<Expression>,
    pub else_branch: Box<Expression>,
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

    // Matrix types (f32) - column-major, CxR naming
    pub fn mat2() -> Type {
        Type::Constructed(TypeName::Str("mat2"), vec![])
    }

    pub fn mat3() -> Type {
        Type::Constructed(TypeName::Str("mat3"), vec![])
    }

    pub fn mat4() -> Type {
        Type::Constructed(TypeName::Str("mat4"), vec![])
    }

    pub fn mat2x3() -> Type {
        Type::Constructed(TypeName::Str("mat2x3"), vec![])
    }

    pub fn mat2x4() -> Type {
        Type::Constructed(TypeName::Str("mat2x4"), vec![])
    }

    pub fn mat3x2() -> Type {
        Type::Constructed(TypeName::Str("mat3x2"), vec![])
    }

    pub fn mat3x4() -> Type {
        Type::Constructed(TypeName::Str("mat3x4"), vec![])
    }

    pub fn mat4x2() -> Type {
        Type::Constructed(TypeName::Str("mat4x2"), vec![])
    }

    pub fn mat4x3() -> Type {
        Type::Constructed(TypeName::Str("mat4x3"), vec![])
    }

    pub fn sized_array(size: usize, elem_type: Type) -> Type {
        Type::Constructed(TypeName::Array("array", size), vec![elem_type])
    }

    pub fn tuple(types: Vec<Type>) -> Type {
        Type::Constructed(TypeName::Str("tuple"), types)
    }

    pub fn attributed_tuple(attributed_types: Vec<crate::ast::AttributedType>) -> Type {
        // For now, extract just the types for the type system
        // The attributes will be handled separately during codegen
        let types = attributed_types.into_iter().map(|at| at.ty).collect();
        Type::Constructed(TypeName::Str("attributed_tuple"), types)
    }

    pub fn function(arg: Type, ret: Type) -> Type {
        Type::arrow(arg, ret)
    }
}
