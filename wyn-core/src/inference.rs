use crate::ast::{types, Type, TypeName};
use polytype::TypeScheme;

// Helper function to create monomorphic type schemes
pub fn monotype(ty: Type) -> TypeScheme<TypeName> {
    TypeScheme::Monotype(ty)
}

// Built-in function types using our type constructors
pub fn builtin_zip_type() -> TypeScheme<TypeName> {
    // zip : [n]i32 -> [n]i32 -> [n](i32, i32) (simplified version)
    let arr_i32 = types::sized_array(1, types::i32()); // placeholder size for polymorphic arrays
    let tuple_i32_i32 = types::tuple(vec![types::i32(), types::i32()]);
    let arr_tuple = types::sized_array(1, tuple_i32_i32);

    // arr_i32 -> arr_i32 -> arr_tuple
    let func_type = types::function(arr_i32.clone(), types::function(arr_i32, arr_tuple));

    TypeScheme::Monotype(func_type)
}
