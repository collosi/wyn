pub use spirv;

// Type aliases for polytype types specialized to our TypeName
pub type Type = polytype::Type<TypeName>;
pub type TypeScheme = polytype::TypeScheme<TypeName>;

/// Qualified name representing a path through modules to a name
/// E.g., M.N.x is represented as QualName { qualifiers: ["M", "N"], name: "x" }
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualName {
    pub qualifiers: Vec<String>,
    pub name: String,
}

impl QualName {
    /// Create a new qualified name
    pub fn new(qualifiers: Vec<String>, name: String) -> Self {
        QualName { qualifiers, name }
    }

    /// Create an unqualified name (no qualifiers)
    pub fn unqualified(name: String) -> Self {
        QualName {
            qualifiers: vec![],
            name,
        }
    }

    /// Mangle into the format used by module elaboration
    /// E.g., M.N.x -> "M_$_N_x", f32.cos -> "f32_cos"
    pub fn mangle(&self) -> String {
        if self.qualifiers.is_empty() {
            self.name.clone()
        } else {
            format!("{}_{}", self.qualifiers.join("_$_"), self.name)
        }
    }

    /// Get the dotted notation (for display/debugging)
    /// E.g., "M.N.x"
    pub fn to_dotted(&self) -> String {
        if self.qualifiers.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.qualifiers.join("."), self.name)
        }
    }

    /// Check if this is an unqualified name
    pub fn is_unqualified(&self) -> bool {
        self.qualifiers.is_empty()
    }
}

/// Source location span tracking (line, column) start and end positions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl Span {
    pub fn new(start_line: usize, start_col: usize, end_line: usize, end_col: usize) -> Self {
        Span {
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    /// Check if this is a generated/dummy span (all zeros)
    pub fn is_generated(&self) -> bool {
        self.start_line == 0 && self.start_col == 0 && self.end_line == 0 && self.end_col == 0
    }

    /// Merge two spans to create a span covering both
    pub fn merge(&self, other: &Span) -> Span {
        let (start_line, start_col) = if self.start_line < other.start_line
            || (self.start_line == other.start_line && self.start_col <= other.start_col)
        {
            (self.start_line, self.start_col)
        } else {
            (other.start_line, other.start_col)
        };

        let (end_line, end_col) = if self.end_line > other.end_line
            || (self.end_line == other.end_line && self.end_col >= other.end_col)
        {
            (self.end_line, self.end_col)
        } else {
            (other.end_line, other.end_col)
        };

        Span {
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.start_line == self.end_line {
            write!(f, "{}:{}..{}", self.start_line, self.start_col, self.end_col)
        } else {
            write!(
                f,
                "{}:{}..{}:{}",
                self.start_line, self.start_col, self.end_line, self.end_col
            )
        }
    }
}

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

    pub fn mk_node<T>(&mut self, kind: T, span: Span) -> Node<T> {
        Node {
            h: Header {
                id: self.next(),
                span,
            },
            kind,
        }
    }
}

#[cfg(test)]
impl NodeCounter {
    /// Create a node with a dummy span (for testing only)
    pub fn mk_node_dummy<T>(&mut self, kind: T) -> Node<T> {
        self.mk_node(kind, Span::dummy())
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
    pub span: Span,
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

/// Type name constructors for the Wyn type system.
///
/// Note on type name variants:
/// - `Float/Int/SInt`: Numeric primitive types with bit widths (e.g., Float(32), SInt(32))
/// - `Str`: Other primitive type names hardcoded in the compiler (e.g., "->", "bool")
///          Uses static strings for efficiency
/// - `Tuple`: Tuple type constructor with arity (number of fields)
/// - `Named`: Type names parsed from user source code (e.g., "vec3", "MyType")
///            Could refer to built-in types, type aliases, or user-defined types
///            Uses owned String since the name comes from parsed input

/// Record field names that preserve source order but have order-independent equality.
/// The actual field types are stored in Type::Constructed's type argument vector.
#[derive(Debug, Clone, Hash)]
pub struct RecordFields(pub Vec<String>);

impl RecordFields {
    pub fn new(fields: Vec<String>) -> Self {
        RecordFields(fields)
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_vec(self) -> Vec<String> {
        self.0
    }

    pub fn contains(&self, key: &str) -> bool {
        self.0.iter().any(|name| name == key)
    }

    pub fn get_index(&self, key: &str) -> Option<usize> {
        self.0.iter().position(|name| name == key)
    }
}

impl From<Vec<String>> for RecordFields {
    fn from(fields: Vec<String>) -> Self {
        RecordFields(fields)
    }
}

impl FromIterator<String> for RecordFields {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        RecordFields(iter.into_iter().collect())
    }
}

impl PartialEq for RecordFields {
    fn eq(&self, other: &Self) -> bool {
        // Order-independent equality: check same field names exist
        if self.0.len() != other.0.len() {
            return false;
        }
        for name in &self.0 {
            if !other.0.contains(name) {
                return false;
            }
        }
        true
    }
}

impl Eq for RecordFields {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeName {
    /// Primitive type names hardcoded in compiler: "->", "bool", etc.
    /// Numeric types use dedicated Float/Int/SInt variants instead.
    /// Tuples use the dedicated Tuple(usize) variant.
    Str(&'static str),
    /// Floating point types: f16, f32, f64, etc.
    Float(usize),
    /// Unsigned integer types: u8, u16, u32, u64
    UInt(usize),
    /// Signed integer types: i8, i16, i32, i64
    Int(usize),
    /// Array type constructor (takes size and element type)
    Array,
    /// Unsized/anonymous array size placeholder (for []t syntax where size is inferred)
    Unsized,
    /// Function arrow type constructor (T1 -> T2)
    Arrow,
    /// Vector type constructor (takes size and element type)
    Vec,
    /// Matrix type constructor (takes rows, columns, and element type)
    /// Corresponds to SPIR-V matrix types
    Mat,
    /// Array size literal
    Size(usize),
    /// Size variable: [n]
    SizeVar(String),
    /// Type variable from user code: 'a, 'b (not yet bound to TypeVar)
    UserVar(String),
    /// Type names parsed from source code (user-defined types, type aliases)
    Named(String),
    /// Uniqueness/consuming type marker (corresponds to "*" prefix)
    Unique,
    /// Record type: {field1: type1, field2: type2}
    /// Preserves source order of fields, but equality is order-independent
    Record(RecordFields),
    /// Unit type: () - the empty tuple, used for side-effect-only functions
    Unit,
    /// Tuple type with arity (size). Field types stored in Type::Constructed args.
    Tuple(usize),
    /// Sum type: Constructor1 type* | Constructor2 type*
    Sum(Vec<(String, Vec<Type>)>),
    /// Existential size: ?[n][m]. type
    Existential(Vec<String>, Box<Type>),
    /// Named parameter: (name: type)
    NamedParam(String, Box<Type>),
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TypeName::Str(s) => write!(f, "{}", s),
            TypeName::Float(bits) => write!(f, "f{}", bits),
            TypeName::UInt(bits) => write!(f, "u{}", bits),
            TypeName::Int(bits) => write!(f, "i{}", bits),
            TypeName::Array => write!(f, "Array"),
            TypeName::Unsized => write!(f, ""),
            TypeName::Arrow => write!(f, "->"),
            TypeName::Vec => write!(f, "Vec"),
            TypeName::Mat => write!(f, "Mat"),
            TypeName::Size(n) => write!(f, "{}", n),
            TypeName::SizeVar(name) => write!(f, "{}", name),
            TypeName::UserVar(name) => write!(f, "'{}", name),
            TypeName::Named(name) => write!(f, "{}", name),
            TypeName::Unique => write!(f, "*"),
            TypeName::Record(fields) => {
                write!(f, "{{")?;
                for (i, name) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", name)?;
                }
                write!(f, "}}")
            }
            TypeName::Unit => write!(f, "()"),
            TypeName::Tuple(n) => write!(f, "Tuple({})", n),
            TypeName::Sum(variants) => {
                for (i, (name, types)) in variants.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", name)?;
                    for ty in types {
                        write!(f, " {}", ty)?;
                    }
                }
                Ok(())
            }
            TypeName::Existential(vars, ty) => {
                write!(f, "?")?;
                for var in vars {
                    write!(f, "[{}]", var)?;
                }
                write!(f, ".{}", ty)
            }
            TypeName::NamedParam(name, ty) => {
                write!(f, "({}:{})", name, ty)
            }
        }
    }
}

impl polytype::Name for TypeName {
    fn arrow() -> Self {
        TypeName::Arrow
    }

    fn show(&self) -> String {
        match self {
            TypeName::Str(s) => s.to_string(),
            TypeName::Float(bits) => format!("f{}", bits),
            TypeName::UInt(bits) => format!("u{}", bits),
            TypeName::Int(bits) => format!("i{}", bits),
            TypeName::Array => "Array".to_string(),
            TypeName::Unsized => "".to_string(),
            TypeName::Arrow => "->".to_string(),
            TypeName::Vec => "Vec".to_string(),
            TypeName::Mat => "Mat".to_string(),
            TypeName::Size(n) => format!("Size({})", n),
            TypeName::SizeVar(v) => format!("[{}]", v),
            TypeName::UserVar(v) => format!("'{}", v),
            TypeName::Named(name) => name.clone(),
            TypeName::Unique => "*".to_string(),
            TypeName::Record(fields) => {
                let field_strs: Vec<String> = fields.iter().map(|name| name.clone()).collect();
                format!("{{{}}}", field_strs.join(", "))
            }
            TypeName::Unit => "()".to_string(),
            TypeName::Tuple(n) => format!("Tuple({})", n),
            TypeName::Sum(variants) => {
                let variant_strs: Vec<String> = variants
                    .iter()
                    .map(|(name, types)| {
                        if types.is_empty() {
                            name.clone()
                        } else {
                            format!(
                                "{} {}",
                                name,
                                types.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(" ")
                            )
                        }
                    })
                    .collect();
                variant_strs.join(" | ")
            }
            TypeName::Existential(vars, ty) => format!("?{}. {}", vars.join(""), ty),
            TypeName::NamedParam(name, ty) => format!("{}: {}", name, ty),
        }
    }
}

impl From<&'static str> for TypeName {
    fn from(s: &'static str) -> Self {
        TypeName::Str(s)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
    /// Declarations from loaded modules, organized by module name
    /// Key is module name (e.g., "f32"), value is list of declarations from that module
    /// These should only be lowered if referenced by user code
    pub library_modules: std::collections::HashMap<String, Vec<Declaration>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Decl(Decl),           // Unified let/def declarations
    Entry(EntryDecl),     // Entry point declarations (vertex/fragment shaders)
    Uniform(UniformDecl), // Uniform declarations (no initializer)
    Sig(SigDecl),
    TypeBind(TypeBind),             // Type declarations
    ModuleBind(ModuleBind),         // Module declarations
    ModuleTypeBind(ModuleTypeBind), // Module type declarations
    Open(ModuleExpression),         // open mod_exp
    Import(String),                 // import "path"
    Local(Box<Declaration>),        // local dec
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
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub params: Vec<Pattern>,     // Parameters as patterns (name, name:type, tuples, etc.)
    pub ty: Option<Type>,         // Return type for functions or type annotation for variables
    pub body: Expression,         // The value/expression for let/def declarations
}

/// Entry point declaration (vertex/fragment shader)
#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl {
    pub entry_type: Attribute, // Attribute::Vertex or Attribute::Fragment
    pub name: String,
    pub params: Vec<Pattern>,                      // Input parameters as patterns
    pub return_types: Vec<Type>,                   // Return tuple field types (parallel array)
    pub return_attributes: Vec<Option<Attribute>>, // Attributes per return field (parallel array)
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SigDecl {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub size_params: Vec<String>, // Size parameters: [n], [m]
    pub type_params: Vec<String>, // Type parameters: 'a, 'b
    pub ty: Type,                 // The function type signature
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniformDecl {
    pub name: String,
    pub ty: Type, // Uniforms always have an explicit type
}

// Module system types
#[derive(Debug, Clone, PartialEq)]
pub struct TypeBind {
    pub kind: TypeBindKind, // type, type^, or type~
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub definition: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeBindKind {
    Normal, // type
    Lifted, // type^
    Size,   // type~
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeParam {
    Size(String),       // [n]
    Type(String),       // 'a
    SizeType(String),   // '~a
    LiftedType(String), // '^a
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleBind {
    pub name: String,
    pub params: Vec<ModuleParam>,
    pub signature: Option<ModuleTypeExpression>,
    pub body: ModuleExpression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleParam {
    pub name: String,
    pub signature: ModuleTypeExpression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleTypeBind {
    pub name: String,
    pub definition: ModuleTypeExpression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleExpression {
    Name(String),                                            // qualname
    Ascription(Box<ModuleExpression>, ModuleTypeExpression), // mod_exp : mod_type_exp
    Lambda(
        Vec<ModuleParam>,
        Option<ModuleTypeExpression>,
        Box<ModuleExpression>,
    ), // \ (params) [: sig] -> body
    Application(Box<ModuleExpression>, Box<ModuleExpression>), // mod_exp mod_exp
    Struct(Vec<Declaration>),                                // { dec* }
    Import(String),                                          // import "path"
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModuleTypeExpression {
    Name(String),                                                        // qualname
    Signature(Vec<Spec>),                                                // { spec* }
    With(Box<ModuleTypeExpression>, String, Vec<TypeParam>, Type), // mod_type with qualname type_params = type
    Arrow(String, Box<ModuleTypeExpression>, Box<ModuleTypeExpression>), // (name : mod_type) -> mod_type
    FunctorType(Box<ModuleTypeExpression>, Box<ModuleTypeExpression>), // mod_type -> mod_type
}

#[derive(Debug, Clone, PartialEq)]
pub enum Spec {
    Sig(String, Vec<TypeParam>, Type), // sig name type_params : type
    SigOp(String, Type),               // sig (symbol) : type or sig symbol : type
    Type(TypeBindKind, String, Vec<TypeParam>, Option<Type>), // type declarations with optional definition
    Module(String, ModuleTypeExpression), // module name : mod_type_exp
    Include(ModuleTypeExpression),     // include mod_type_exp
}

// We now use polytype::Type instead of our own Type enum

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    IntLiteral(i32),
    FloatLiteral(f32),
    BoolLiteral(bool),
    StringLiteral(String),
    Unit,
    Identifier(String),
    QualifiedName(Vec<String>, String), // quals, name - e.g., f32.sin is (["f32"], "sin")
    OperatorSection(String),            // e.g., (+), (-), (*) - operator as a value
    ArrayLiteral(Vec<Expression>),
    VecMatLiteral(Vec<Expression>), // @[...] - vector or matrix literal (type inferred from context)
    ArrayIndex(Box<Expression>, Box<Expression>),
    BinaryOp(BinaryOp, Box<Expression>, Box<Expression>),
    UnaryOp(UnaryOp, Box<Expression>), // Unary operations: -, !
    Tuple(Vec<Expression>),
    RecordLiteral(Vec<(String, Expression)>), // e.g. {x: 1, y: 2}
    Lambda(LambdaExpr),
    Application(Box<Expression>, Vec<Expression>), // Function application
    LetIn(LetInExpr),
    FieldAccess(Box<Expression>, String),     // e.g. v.x, v.y
    If(IfExpr),                               // if-then-else expression
    Loop(LoopExpr),                           // loop expression
    Match(MatchExpr),                         // match expression
    Range(RangeExpr),                         // range expressions: a..b, a..<b, a..>b, a...b
    Pipe(Box<Expression>, Box<Expression>),   // |> pipe operator
    TypeAscription(Box<Expression>, Type),    // exp : type
    TypeCoercion(Box<Expression>, Type),      // exp :> type
    Assert(Box<Expression>, Box<Expression>), // assert cond exp
    TypeHole,                                 // ??? - placeholder for any expression
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambdaExpr {
    pub params: Vec<Pattern>,
    pub return_type: Option<Type>,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetInExpr {
    pub pattern: Pattern, // Can be Name, Tuple, etc.
    pub ty: Option<Type>, // Optional type annotation
    pub value: Box<Expression>,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOp {
    pub op: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryOp {
    pub op: String, // "-" or "!"
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr {
    pub condition: Box<Expression>,
    pub then_branch: Box<Expression>,
    pub else_branch: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopExpr {
    pub pattern: Pattern,              // loop variable pattern
    pub init: Option<Box<Expression>>, // initial value (optional)
    pub form: LoopForm,                // for/while condition
    pub body: Box<Expression>,         // loop body
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopForm {
    For(String, Box<Expression>),    // for name < exp
    ForIn(Pattern, Box<Expression>), // for pat in exp
    While(Box<Expression>),          // while exp
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpr {
    pub scrutinee: Box<Expression>, // expression being matched
    pub cases: Vec<MatchCase>,      // case branches
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeExpr {
    pub start: Box<Expression>,
    pub step: Option<Box<Expression>>, // Optional middle expression in start..step..end
    pub end: Box<Expression>,
    pub kind: RangeKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RangeKind {
    Inclusive,   // ... (three dots)
    Exclusive,   // .. (two dots)
    ExclusiveLt, // ..<
    ExclusiveGt, // ..>
}

// Pattern types for match expressions and let bindings
#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Name(String),                             // Simple name binding
    Wildcard,                                 // _ wildcard
    Literal(PatternLiteral),                  // Literal patterns
    Unit,                                     // () unit pattern
    Tuple(Vec<Pattern>),                      // (pat1, pat2, ...)
    Record(Vec<RecordPatternField>),          // { field1, field2 = pat, ... }
    Constructor(String, Vec<Pattern>),        // Constructor application
    Typed(Box<Pattern>, Type),                // pat : type
    Attributed(Vec<Attribute>, Box<Pattern>), // #[attr] pat
}

pub type Pattern = Node<PatternKind>;

impl Pattern {
    /// Extract the simple name from a pattern if possible
    /// For Name("x") returns Some("x")
    /// For Typed(Name("x"), _) returns Some("x")
    /// For Attributed(_, Name("x")) returns Some("x")
    /// Returns None for complex patterns like tuples, records, etc.
    pub fn simple_name(&self) -> Option<&str> {
        match &self.kind {
            PatternKind::Name(name) => Some(name),
            PatternKind::Typed(inner, _) => inner.simple_name(),
            PatternKind::Attributed(_, inner) => inner.simple_name(),
            _ => None,
        }
    }

    /// Extract the type from a typed pattern
    pub fn pattern_type(&self) -> Option<&Type> {
        match &self.kind {
            PatternKind::Typed(_, ty) => Some(ty),
            PatternKind::Attributed(_, inner) => inner.pattern_type(),
            _ => None,
        }
    }

    /// Collect all names bound by this pattern
    /// For Name("x") returns vec!["x"]
    /// For Tuple([Name("x"), Name("y")]) returns vec!["x", "y"]
    /// For nested patterns, recursively collects all names
    pub fn collect_names(&self) -> Vec<String> {
        match &self.kind {
            PatternKind::Name(name) => vec![name.clone()],
            PatternKind::Tuple(patterns) => patterns.iter().flat_map(|p| p.collect_names()).collect(),
            PatternKind::Typed(inner, _) => inner.collect_names(),
            PatternKind::Attributed(_, inner) => inner.collect_names(),
            PatternKind::Record(fields) => fields
                .iter()
                .flat_map(|f| {
                    if let Some(ref pat) = f.pattern { pat.collect_names() } else { vec![f.field.clone()] }
                })
                .collect(),
            PatternKind::Constructor(_, patterns) => {
                patterns.iter().flat_map(|p| p.collect_names()).collect()
            }
            _ => vec![], // Wildcard, Literal, Unit bind no names
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternLiteral {
    Int(i32),
    Float(f32),
    Char(char),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordPatternField {
    pub field: String,
    pub pattern: Option<Pattern>, // None means shorthand (just field name)
}

// Helper module for creating common polytype Types
pub mod types {
    use super::{RecordFields, Type, TypeName};

    pub fn i32() -> Type {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    pub fn f32() -> Type {
        Type::Constructed(TypeName::Float(32), vec![])
    }

    pub fn bool_type() -> Type {
        Type::Constructed(TypeName::Str("bool"), vec![])
    }

    pub fn string() -> Type {
        Type::Constructed(TypeName::Str("string"), vec![])
    }

    pub fn unit() -> Type {
        Type::Constructed(TypeName::Unit, vec![])
    }

    /// All valid SPIR-V scalar element types for vectors and matrices
    fn spirv_element_types() -> Vec<(&'static str, Type)> {
        vec![
            ("i8", Type::Constructed(TypeName::Int(8), vec![])),
            ("i16", Type::Constructed(TypeName::Int(16), vec![])),
            ("i32", Type::Constructed(TypeName::Int(32), vec![])),
            ("i64", Type::Constructed(TypeName::Int(64), vec![])),
            ("u8", Type::Constructed(TypeName::UInt(8), vec![])),
            ("u16", Type::Constructed(TypeName::UInt(16), vec![])),
            ("u32", Type::Constructed(TypeName::UInt(32), vec![])),
            ("u64", Type::Constructed(TypeName::UInt(64), vec![])),
            ("f16", Type::Constructed(TypeName::Float(16), vec![])),
            ("f32", Type::Constructed(TypeName::Float(32), vec![])),
            ("f64", Type::Constructed(TypeName::Float(64), vec![])),
            ("bool", Type::Constructed(TypeName::Str("bool"), vec![])),
        ]
    }

    // Vector types
    /// Create a vector type: Vec(size, element_type)
    /// Example: vec(2, f32()) creates Vec(Size(2), f32) for vec2
    pub fn vec(size: usize, element_type: Type) -> Type {
        Type::Constructed(
            TypeName::Vec,
            vec![Type::Constructed(TypeName::Size(size), vec![]), element_type],
        )
    }

    /// Generate all vector type constructors using cartesian product of sizes and element types
    /// Returns a HashMap mapping names like "vec2f32", "vec3i32", "vec4bool" to their Type representations
    pub fn vector_type_constructors() -> std::collections::HashMap<String, Type> {
        use itertools::Itertools;

        let sizes = [2, 3, 4];
        let elem_types = spirv_element_types();

        sizes
            .iter()
            .cartesian_product(elem_types.iter())
            .map(|(size, (elem_name, elem_type))| {
                let name = format!("vec{}{}", size, elem_name);
                let vec_type = Type::Constructed(
                    TypeName::Vec,
                    vec![
                        Type::Constructed(TypeName::Size(*size), vec![]),
                        elem_type.clone(),
                    ],
                );
                (name, vec_type)
            })
            .collect()
    }

    // Matrix types - column-major, CxR naming
    /// Create a matrix type: mat<rows, cols, elem_type>
    pub fn mat(rows: usize, cols: usize, elem_type: Type) -> Type {
        Type::Constructed(
            TypeName::Mat,
            vec![
                Type::Constructed(TypeName::Size(rows), vec![]),
                Type::Constructed(TypeName::Size(cols), vec![]),
                elem_type,
            ],
        )
    }

    /// Generate all matrix type constructors using cartesian product of dimensions and element types
    /// Returns a HashMap mapping names like "mat2f32", "mat3x4i32" to their Type representations
    pub fn matrix_type_constructors() -> std::collections::HashMap<String, Type> {
        use itertools::Itertools;

        let dims = [2, 3, 4];
        let elem_types = spirv_element_types();

        dims.iter()
            .cartesian_product(dims.iter())
            .cartesian_product(elem_types.iter())
            .flat_map(|((rows, cols), (elem_name, elem_type))| {
                let matrix_type = Type::Constructed(
                    TypeName::Mat,
                    vec![
                        Type::Constructed(TypeName::Size(*rows), vec![]),
                        Type::Constructed(TypeName::Size(*cols), vec![]),
                        elem_type.clone(),
                    ],
                );

                if rows == cols {
                    // Square matrices: add both matNf32 and matNxNf32 as aliases
                    vec![
                        (format!("mat{}{}", rows, elem_name), matrix_type.clone()),
                        (format!("mat{}x{}{}", rows, cols, elem_name), matrix_type),
                    ]
                } else {
                    // Non-square matrices: only matRxCf32
                    vec![(format!("mat{}x{}{}", rows, cols, elem_name), matrix_type)]
                }
            })
            .collect()
    }

    pub fn sized_array(size: usize, elem_type: Type) -> Type {
        Type::Constructed(
            TypeName::Array,
            vec![Type::Constructed(TypeName::Size(size), vec![]), elem_type],
        )
    }

    pub fn tuple(types: Vec<Type>) -> Type {
        let arity = types.len();
        Type::Constructed(TypeName::Tuple(arity), types)
    }

    pub fn function(arg: Type, ret: Type) -> Type {
        Type::arrow(arg, ret)
    }

    /// Destructure an arrow type into (param_type, result_type)
    /// Returns None if the type is not an arrow type
    pub fn as_arrow(ty: &Type) -> Option<(&Type, &Type)> {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => Some((&args[0], &args[1])),
            _ => None,
        }
    }

    /// Check if a type is an integer type (signed or unsigned)
    /// Per spec: array indices may be "any unsigned integer type",
    /// but we also accept signed integers for compatibility
    pub fn is_integer_type(ty: &Type) -> bool {
        match ty {
            Type::Constructed(TypeName::Str(name), args) if args.is_empty() => {
                matches!(
                    name.as_ref(),
                    "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
                )
            }
            _ => false,
        }
    }

    /// Create a record type: {field1: type1, field2: type2}
    pub fn record(fields: Vec<(String, Type)>) -> Type {
        let (field_names, field_types): (Vec<String>, Vec<Type>) = fields.into_iter().unzip();
        Type::Constructed(TypeName::Record(RecordFields::new(field_names)), field_types)
    }

    /// Create a sum type: Constructor1 type* | Constructor2 type*
    pub fn sum(variants: Vec<(String, Vec<Type>)>) -> Type {
        Type::Constructed(TypeName::Sum(variants), vec![])
    }

    /// Create an existential size type: ?[n][m]. type
    pub fn existential(size_vars: Vec<String>, inner: Type) -> Type {
        Type::Constructed(TypeName::Existential(size_vars, Box::new(inner)), vec![])
    }

    /// Create a named parameter type: (name: type)
    pub fn named_param(name: String, ty: Type) -> Type {
        Type::Constructed(TypeName::NamedParam(name, Box::new(ty)), vec![])
    }

    /// Create a size variable in array types: [n]
    pub fn size_var(name: String) -> Type {
        Type::Constructed(TypeName::SizeVar(name), vec![])
    }

    /// Create a unique (consuming/alias-free) type: *t
    pub fn unique(inner: Type) -> Type {
        Type::Constructed(TypeName::Unique, vec![inner])
    }

    /// Check if a type is marked as unique/consuming
    pub fn is_unique(ty: &Type) -> bool {
        matches!(ty, Type::Constructed(TypeName::Unique, _))
    }

    /// Strip uniqueness marker from a type, returning the inner type
    pub fn strip_unique(ty: &Type) -> Type {
        match ty {
            Type::Constructed(TypeName::Unique, args) => {
                // Recursively strip from the inner type
                let inner = args.first().cloned().unwrap_or_else(|| ty.clone());
                strip_unique(&inner)
            }
            Type::Constructed(name, args) => {
                // Recursively strip from constructor arguments (e.g., arrow types)
                let stripped_args: Vec<Type> = args.iter().map(strip_unique).collect();
                Type::Constructed(name.clone(), stripped_args)
            }
            _ => ty.clone(),
        }
    }
}
