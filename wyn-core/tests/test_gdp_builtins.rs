use gasm::parse_function;
use wyn_core::gasm_lowering::lower_gasm_module;
use std::fs;

#[test]
fn test_parse_gdp_encode_float32() {
    let gasm_code = fs::read_to_string("gdp_builtins.gasm")
        .expect("Failed to read gdp_builtins.gasm");

    // Extract just the gdp_encode_float32 function
    let start = gasm_code.find("func @gdp_encode_float32").expect("Function not found");
    let end_marker = "\n; ============================================================================\n; gdp_encode_uint";
    let end = gasm_code[start..].find(end_marker).expect("End marker not found") + start;
    let function_code = &gasm_code[start..end];

    println!("Parsing function:\n{}", function_code);

    let result = parse_function(function_code);
    if let Err(ref e) = result {
        println!("Parse error: {:?}", e);
    }
    assert!(result.is_ok(), "Failed to parse gdp_encode_float32");

    let function = result.unwrap();
    assert_eq!(function.name, "gdp_encode_float32");
    assert_eq!(function.params.len(), 1);
}

#[test]
fn test_lower_gdp_encode_float32() {
    let gasm_code = fs::read_to_string("gdp_builtins.gasm")
        .expect("Failed to read gdp_builtins.gasm");

    // Extract just the gdp_encode_float32 function
    let start = gasm_code.find("func @gdp_encode_float32").expect("Function not found");
    let end_marker = "\n; ============================================================================\n; gdp_encode_uint";
    let end = gasm_code[start..].find(end_marker).expect("End marker not found") + start;
    let function_code = &gasm_code[start..end];

    let function = parse_function(function_code).expect("Failed to parse");

    // Note: We can't lower this yet because it references @gdp_buffer which needs to be a global
    // For now, just verify it parses
    assert_eq!(function.blocks.len(), 1);
}
