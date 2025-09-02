use crate::ast::{Type, TypeScheme, types};

// Helper function to create monomorphic type schemes
pub fn monotype(ty: Type) -> TypeScheme {
    TypeScheme::Monotype(ty)
}

// Built-in function types using our type constructors
pub fn builtin_zip_type() -> TypeScheme {
    // zip : [n]i32 -> [n]i32 -> [n](i32, i32) (simplified version)
    let arr_i32 = types::array(types::i32());
    let tuple_i32_i32 = types::tuple(vec![types::i32(), types::i32()]);
    let arr_tuple = types::array(tuple_i32_i32);
    
    // arr_i32 -> arr_i32 -> arr_tuple
    let func_type = types::function(
        arr_i32.clone(),
        types::function(arr_i32, arr_tuple)
    );

    TypeScheme::Monotype(func_type)
}
