use gasm::parse_function;
use std::fs;

#[test]
fn test_parse_gdp_encode_float32() {
    let gasm_code = fs::read_to_string("gdp_builtins.gasm").expect("Failed to read gdp_builtins.gasm");

    // Extract just the gdp_encode_float32 function
    let start = gasm_code.find("func @gdp_encode_float32").expect("Function not found");
    let end_marker = "\n; ============================================================================\n; gdp_encode_uint";
    let end = gasm_code[start..].find(end_marker).expect("End marker not found") + start;
    let function_code = &gasm_code[start..end];

    let result = parse_function(function_code);
    if let Err(ref e) = result {
        println!("Parse error: {:?}", e);
    }
    assert!(result.is_ok(), "Failed to parse gdp_encode_float32");

    let function = result.unwrap();
    assert_eq!(function.name, "gdp_encode_float32");
    assert_eq!(function.params.len(), 1);
    assert_eq!(function.blocks.len(), 1);
}

#[test]
fn test_parse_gdp_encode_uint() {
    let gasm_code = fs::read_to_string("gdp_builtins.gasm").expect("Failed to read gdp_builtins.gasm");

    // Extract gdp_encode_uint function
    let start = gasm_code.find("func @gdp_encode_uint").expect("Function not found");
    let end_marker = "\n; ============================================================================\n; gdp_encode_string";
    let end = gasm_code[start..].find(end_marker).expect("End marker not found") + start;
    let function_code = &gasm_code[start..end];

    let result = parse_function(function_code);
    if let Err(ref e) = result {
        println!("Parse error: {:?}", e);
    }
    assert!(result.is_ok(), "Failed to parse gdp_encode_uint");

    let function = result.unwrap();
    assert_eq!(function.name, "gdp_encode_uint");
    assert_eq!(function.params.len(), 1);
    // Should have entry, encode_inline, and encode_full blocks
    assert_eq!(function.blocks.len(), 3);
}

#[test]
fn test_parse_gdp_encode_string() {
    let gasm_code = fs::read_to_string("gdp_builtins.gasm").expect("Failed to read gdp_builtins.gasm");

    // Extract gdp_encode_string function (goes to end of file)
    let start = gasm_code.find("func @gdp_encode_string").expect("Function not found");
    let function_code = &gasm_code[start..];

    let result = parse_function(function_code);
    if let Err(ref e) = result {
        println!("Parse error: {:?}", e);
    }
    assert!(result.is_ok(), "Failed to parse gdp_encode_string");

    let function = result.unwrap();
    assert_eq!(function.name, "gdp_encode_string");
    assert_eq!(function.params.len(), 2);
    assert_eq!(function.blocks.len(), 1);
}
