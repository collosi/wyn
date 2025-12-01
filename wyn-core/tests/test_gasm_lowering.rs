use gasm::parse_function;
use wyn_core::gasm_lowering::lower_gasm_module;

#[test]
fn test_simple_gasm_lowering() {
    let gasm_code = r#"
func @test(%x: u32) -> u32 {
entry:
  ret 1u
}
"#;

    let function = parse_function(gasm_code).expect("Failed to parse GASM");

    let module = gasm::Module {
        globals: vec![],
        functions: vec![function],
    };

    let spirv = lower_gasm_module(&module).expect("Failed to lower GASM to SPIR-V");

    assert!(!spirv.is_empty(), "SPIR-V output should not be empty");

    // First word should be SPIR-V magic number
    assert_eq!(spirv[0], 0x07230203, "First word should be SPIR-V magic number");
}

#[test]
fn test_bitwise_operations() {
    let gasm_code = r#"
func @byte_swap(%x: u32) -> u32 {
entry:
  %byte0 = and %x, 0xFFu
  %shr = shr %x, 8u
  %byte1 = and %shr, 0xFFu
  %shifted = shl %byte0, 8u
  %result = or %shifted, %byte1

  ret %result
}
"#;

    let function = parse_function(gasm_code).expect("Failed to parse GASM");

    let module = gasm::Module {
        globals: vec![],
        functions: vec![function],
    };

    let spirv = lower_gasm_module(&module).expect("Failed to lower GASM to SPIR-V");

    assert!(!spirv.is_empty(), "SPIR-V output should not be empty");
    assert_eq!(spirv[0], 0x07230203, "First word should be SPIR-V magic number");
}
