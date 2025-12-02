use super::ModuleManager;

#[test]
fn test_query_f32_sin_from_math_prelude() {
    let manager = ModuleManager::new();

    // Prelude files are automatically loaded on creation

    // Query for the f32 module's sin function type
    let sin_type = manager.get_module_function_type("f32", "sin").expect("Failed to find f32.sin");

    // Should be f32 -> f32 (or t -> t where t = f32)
    println!("Found f32.sin with type: {:?}", sin_type);

    // Also test that f32.sum is found (from module body)
    let sum_type = manager.get_module_function_type("f32", "sum").expect("Failed to find f32.sum");
    println!("Found f32.sum with type: {:?}", sum_type);

    // TODO: Once we can properly extract the type, assert it's f32 -> f32
    // assert!(matches expected type structure);
}
