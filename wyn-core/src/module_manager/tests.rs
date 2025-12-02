use super::ModuleManager;

#[test]
fn test_query_f32_sin_from_math_prelude() {
    let mut manager = ModuleManager::new();

    // Load math.wyn which contains multiple modules (f32, i32, etc.)
    manager.load_file("math.wyn").expect("Failed to load math.wyn");

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

#[test]
fn test_dump_all_elaborated_modules() {
    let mut manager = ModuleManager::new();

    // Load math.wyn which contains multiple modules (f32, i32, etc.)
    manager.load_file("math.wyn").expect("Failed to load math.wyn");

    println!("\n=== ELABORATED MODULES ===\n");

    // Access the elaborated_modules field
    for (module_name, elaborated) in &manager.elaborated_modules {
        println!("MODULE: {}", module_name);
        println!("  Items ({} total):", elaborated.items.len());
        for (i, item) in elaborated.items.iter().enumerate() {
            match item {
                super::ElaboratedItem::Spec(spec) => {
                    println!("    [{}] Spec: {:?}", i, spec);
                }
                super::ElaboratedItem::Decl(decl) => {
                    println!("    [{}] Decl: {} : {:?}", i, decl.name, decl.ty);
                }
            }
        }
        println!();
    }
    assert!(false);
}
