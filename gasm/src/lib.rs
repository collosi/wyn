pub mod error;
pub mod ir;
pub mod parser;

pub use error::{ParseError, Result};
pub use ir::*;
pub use parser::{parse_function, parse_module};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let input = r#"
func @max_u32(%a: u32, %b: u32) -> u32 {
entry:
  %cmp = ucmp.ge %a, %b
  br_if %cmp, a_ge_b, otherwise, a_ge_b

a_ge_b:
  ret %a

otherwise:
  ret %b
}
"#;
        let result = parse_function(input);
        assert!(result.is_ok());
        let func = result.unwrap();
        assert_eq!(func.name, "max_u32");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.blocks.len(), 3);
    }

    #[test]
    fn test_parse_atomic_increment() {
        let input = r#"
func kernel @bump(%ptr: ptr[global]<u32>) -> void {
entry:
  %old = atomic.rmw add %ptr, 1u ordering=acq_rel scope=device
  ret
}
"#;
        let result = parse_function(input);
        assert!(result.is_ok());
        let func = result.unwrap();
        assert_eq!(func.name, "bump");
        assert_eq!(func.attributes.len(), 1);
        assert!(matches!(func.attributes[0], FunctionAttr::Kernel));
    }

    /// Test byte reversal with immediate constants (modern GASM syntax)
    #[test]
    fn test_parse_byte_reversal() {
        let input = r#"
func @swap_bytes_u32(%x: u32) -> u32 {
entry:
  %byte0    = and %x, 0xFFu
  %shr_8    = shr %x, 8u
  %byte1    = and %shr_8, 0xFFu
  %shr_16   = shr %x, 16u
  %byte2    = and %shr_16, 0xFFu
  %byte3    = shr %x, 24u

  %b1_shift = shl %byte2, 8u
  %b2_shift = shl %byte1, 16u
  %b3_shift = shl %byte0, 24u

  %temp1 = or %byte3, %b1_shift
  %temp2 = or %temp1, %b2_shift
  %result = or %temp2, %b3_shift

  ret %result
}
"#;

        let result = parse_function(input);
        assert!(
            result.is_ok(),
            "Failed to parse byte_reversal: {:?}",
            result.err()
        );
        let func = result.unwrap();
        assert_eq!(func.name, "swap_bytes_u32");
        assert_eq!(func.blocks[0].instructions.len(), 12);
    }

    #[test]
    fn test_parse_conditional_select() {
        let input = r#"
func @clamp_u32(%x: u32, %min: u32, %max: u32) -> u32 {
entry:
  %lt_min = ucmp.lt %x, %min
  %clamped_min = select %lt_min, %min, %x
  %gt_max = ucmp.gt %clamped_min, %max
  %clamped = select %gt_max, %max, %clamped_min
  ret %clamped
}
"#;

        let result = parse_function(input);
        assert!(result.is_ok());
        let func = result.unwrap();
        assert_eq!(func.blocks[0].instructions.len(), 4);
    }

    #[test]
    fn test_parse_spinlock() {
        let input = r#"
func @lock_acquire(%lock: ptr[global]<u32>) -> void {
entry:
  br try_lock

try_lock:
  %old = atomic.cmpxchg %lock, 0u, 1u ordering_succ=acq_rel ordering_fail=acquire scope=device
  %ok  = ucmp.eq %old, 0u
  br_if %ok, done, try_lock, done

done:
  ret
}
"#;

        let result = parse_function(input);
        assert!(result.is_ok());
        let func = result.unwrap();
        assert_eq!(func.blocks.len(), 3);
        assert_eq!(func.blocks[1].label, "try_lock");
    }

    #[test]
    fn test_parse_module() {
        let input = r#"
global @counter : ptr[global]<u32> = addr(0x1000)
global @lock : ptr[global]<u32>

func kernel @increment() -> void {
entry:
  %old = atomic.rmw add @counter, 1u ordering=acq_rel scope=device
  ret
}
"#;

        let result = parse_module(input);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert_eq!(module.globals.len(), 2);
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.globals[0].name, "counter");
        assert!(module.globals[0].initializer.is_some());
        assert_eq!(module.functions[0].name, "increment");
    }

    #[test]
    fn test_parse_phi_node() {
        let input = r#"
func @abs_i32(%x: i32) -> i32 {
entry:
  %is_neg = icmp.lt %x, 0i
  br_if %is_neg, neg, done, done

neg:
  %neg_x = sub 0i, %x
  br done

done:
  %res = phi i32 [ %x, entry ], [ %neg_x, neg ]
  ret %res
}
"#;

        let result = parse_function(input);
        assert!(result.is_ok());
        let func = result.unwrap();
        assert_eq!(func.blocks.len(), 3);

        // Check phi instruction in done block
        let done_block = &func.blocks[2];
        assert_eq!(done_block.label, "done");
        assert_eq!(done_block.instructions.len(), 1);
        assert!(matches!(done_block.instructions[0].op, Operation::Phi { .. }));
    }

    #[test]
    fn test_type_conversions() {
        let input = r#"
func @conversions(%i: i32, %f: f32) -> f64 {
entry:
  %bits = bitcast %f
  %extended = sext %i
  %float = sitofp %extended
  %double = fpext %float
  ret %double
}
"#;

        let result = parse_function(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_call() {
        let input = r#"
func @test_call(%x: u32, %y: u32) -> u32 {
entry:
  %sum = call @add(%x, %y)
  %doubled = call @double(%sum)
  ret %doubled
}
"#;

        let result = parse_function(input);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        let func = result.unwrap();
        assert_eq!(func.name, "test_call");
        assert_eq!(func.blocks[0].instructions.len(), 2);

        // Check first call
        assert!(matches!(
            &func.blocks[0].instructions[0].op,
            Operation::Call { func, args } if func == "add" && args.len() == 2
        ));

        // Check second call
        assert!(matches!(
            &func.blocks[0].instructions[1].op,
            Operation::Call { func, args } if func == "double" && args.len() == 1
        ));
    }

    #[test]
    fn test_parse_error_reporting() {
        // This should fail - invalid syntax to test error reporting
        let input = r#"
func @broken_function(%x: u32) -> u32 {
entry:
  %bad = invalid_opcode %x
  ret %bad
}
"#;

        let result = parse_function(input);
        assert!(result.is_err(), "Should fail to parse invalid opcode");
        eprintln!("Error (expected): {:?}", result.err());
    }

    #[test]
    fn test_parse_minimal_multiblock() {
        let input = r#"
func @test(%x: u32) -> void {
entry:
  %y = add %x, %x
  br next_block

next_block:
  ret
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Should parse minimal multi-block function");

        let func = result.unwrap();
        assert_eq!(func.name, "test");
        assert_eq!(func.blocks.len(), 2);
        assert_eq!(func.blocks[0].label, "entry");
        assert_eq!(func.blocks[1].label, "next_block");
    }

    #[test]
    fn test_parse_multiblock_with_brif() {
        let input = r#"
func @test(%x: u32) -> void {
entry:
  %is_small = ucmp.lt %x, 64u
  br_if %is_small, small_case, big_case, small_case

small_case:
  ret

big_case:
  ret
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Should parse multi-block function with br_if");

        let func = result.unwrap();
        assert_eq!(func.name, "test");
        assert_eq!(func.blocks.len(), 3);
    }

    #[test]
    fn test_parse_gdp_first_block_only() {
        // Just the entry block from gdp_encode_float32
        let input = r#"
func @gdp_encode_float32(%f: f32) -> void {
entry:
  %bits = bitcast %f
  %byte0 = and %bits, 0xFFu
  ret
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Should parse entry block from GDP function");
    }

    #[test]
    fn test_parse_gdp_two_blocks() {
        // Entry block + one more block
        let input = r#"
func @gdp_encode_float32(%f: f32) -> void {
entry:
  %bits = bitcast %f
  %is_full = ucmp.ge %bits, 8u
  br_if %is_full, full_exit, check_wrap, check_wrap

check_wrap:
  ret

full_exit:
  ret
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Should parse two blocks from GDP function");
    }

    #[test]
    fn test_parse_gdp_last_5_instructions() {
        // Test atomic operations with the GDP buffer
        let input = r#"
global @gdp_buffer : ptr[global]<u32>

func @test() -> void {
entry:
  ; Get pointers
  %write_head_ptr = gep[ptr[global]<u32>] @gdp_buffer, 0u, stride=4
  %max_loops_ptr = gep[ptr[global]<u32>] @gdp_buffer, 2u, stride=4

  ; Load max_loops
  %max_loops = load %max_loops_ptr

  ; Atomic load current write_head to check if over limit
  %current_write_head = atomic.load %write_head_ptr ordering=acq_rel scope=device

  ; Calculate loop count: current_write_head / ring_size
  %current_loop = udiv %current_write_head, 4093u

  ; Check if we're over the limit
  %is_full = ucmp.ge %current_loop, %max_loops
  br_if %is_full, full_exit, check_wrap, check_wrap

check_wrap:
  ret

full_exit:
  ret
}
"#;

        let result = parse_module(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(
            result.is_ok(),
            "Should parse last 5 instructions from gdp_encode_float32"
        );
    }

    #[test]
    fn test_parse_atomic_load_instruction() {
        let input = r#"
func @test(%ptr: ptr[global]<u32>) -> u32 {
entry:
  %value = atomic.load %ptr ordering=acq_rel scope=device
  ret %value
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Should parse atomic.load instruction");

        let func = result.unwrap();
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }

    #[test]
    fn test_parse_module_with_global_and_function() {
        let input = r#"
global @gdp_buffer : ptr[global]<u32>

; Comment line 1
; Comment line 2

; ========================================
; Function description
; ========================================

func @test_func(%x: u32) -> void {
entry:
  ret
}
"#;

        let result = parse_module(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Should parse module with global and function");

        let module = result.unwrap();
        assert_eq!(module.globals.len(), 1, "Should have 1 global");
        assert_eq!(module.functions.len(), 1, "Should have 1 function");
    }

    #[test]
    fn test_parse_gdp_builtins_file_directly() {
        // Try to parse the actual file
        if let Ok(content) = std::fs::read_to_string("../wyn-core/builtins/gdp_builtins.gasm") {
            // Extract from start to end of first function (global + first function)
            if let Some(func_start) = content.find("func @gdp_encode_float32") {
                if let Some(next_func_pos) = content[func_start..].find("\nfunc @") {
                    // Find the last closing brace before the next function
                    let before_next = &content[func_start..func_start + next_func_pos];
                    if let Some(closing_brace_offset) = before_next.rfind('}') {
                        // Extract from the beginning (including global) to end of first function
                        let module_content = &content[..func_start + closing_brace_offset + 1];
                        eprintln!("Extracted module with {} bytes", module_content.len());
                        eprintln!(
                            "Starts with: {:?}",
                            &module_content[..100.min(module_content.len())]
                        );
                        eprintln!(
                            "Ends with: {:?}",
                            &module_content[module_content.len().saturating_sub(100)..]
                        );

                        // Try parsing with parse_module
                        let result = parse_module(module_content);
                        if let Err(e) = &result {
                            eprintln!("Parse error with parse_module: {:?}", e);
                            panic!("Failed to parse module: {:?}", e);
                        } else {
                            eprintln!(
                                "parse_module succeeded! Got {} globals and {} functions",
                                result.as_ref().unwrap().globals.len(),
                                result.as_ref().unwrap().functions.len()
                            );
                            assert_eq!(result.as_ref().unwrap().globals.len(), 1, "Should have 1 global");
                            assert_eq!(
                                result.as_ref().unwrap().functions.len(),
                                1,
                                "Should have 1 function"
                            );
                        }
                    } else {
                        panic!("Could not find closing brace for first function");
                    }
                } else {
                    panic!("Could not find second function");
                }
            } else {
                panic!("Could not find gdp_encode_float32");
            }
        }
    }
}
