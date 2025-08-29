use crate::ast::Type;
use polytype::{Name, TypeScheme};
use std::fmt;

// We need to implement Name for our Type enum so it can be used with polytype
impl Name for Type {
    fn arrow() -> Self {
        Type::Function(Box::new(Type::Var("_".to_string())), Box::new(Type::Var("_".to_string())))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::I32 => write!(f, "i32"),
            Type::F32 => write!(f, "f32"),
            Type::Array(elem_ty, dims) => {
                let dim_str = dims.iter()
                    .map(|d| format!("[{}]", d))
                    .collect::<String>();
                write!(f, "{}{}", dim_str, elem_ty)
            }
            Type::Tuple(types) => {
                let type_strs: Vec<String> = types.iter()
                    .map(|t| format!("{}", t))
                    .collect();
                write!(f, "({})", type_strs.join(", "))
            }
            Type::Var(name) => write!(f, "{}", name),
            Type::Function(arg, ret) => {
                write!(f, "{} -> {}", arg, ret)
            }
            Type::SizeVar(name) => write!(f, "{}", name),
        }
    }
}

// Helper function to create type schemes
pub fn monotype(ty: Type) -> TypeScheme<Type> {
    TypeScheme::Monotype(polytype::Type::Constructed(ty, vec![]))
}

// Built-in function types
pub fn builtin_zip_type() -> TypeScheme<Type> {
    use polytype::Type as PT;
    
    // zip : [d]a -> [d]b -> [d](a, b)
    // For simplicity, we'll make this concrete for now
    // A full implementation would use proper polymorphic types
    
    // [n]i32 -> [n]i32 -> [n](i32, i32) as a simplified version
    let arr_a = PT::Constructed(
        Type::Array(Box::new(Type::I32), vec![1]), // simplified to concrete size
        vec![]
    );
    
    let arr_b = PT::Constructed(
        Type::Array(Box::new(Type::I32), vec![1]),
        vec![]
    );
    
    let _tuple_result = PT::Constructed(
        Type::Tuple(vec![Type::I32, Type::I32]),
        vec![]
    );
    
    let arr_result = PT::Constructed(
        Type::Array(Box::new(Type::Tuple(vec![Type::I32, Type::I32])), vec![1]),
        vec![]
    );
    
    // [n]a -> ([n]b -> [n](a, b))
    let func_type = PT::arrow(arr_a, PT::arrow(arr_b, arr_result));
    
    TypeScheme::Monotype(func_type)
}