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
  br_if %cmp, a_ge_b, otherwise

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

    /// Test case translating the SPIR-V emit_gdp_encode_float32 function to GIR assembly
    /// This function converts a float32 to byte-reversed gob-encoded format
    #[test]
    fn test_gdp_encode_float32() {
        let input = r#"
func @gdp_encode_float32(%f: f32) -> u32 {
entry:
  ; 1. Bitcast f32 to u32
  %bits = bitcast %f

  ; 2. Byte-reverse: swap bytes in SPIR-V
  ; Extract bytes from the u32 value
  %const_8 = uconst 8u
  %const_16 = uconst 16u
  %const_24 = uconst 24u
  %const_0xFF = uconst 0xFFu

  %byte0 = and %bits, %const_0xFF
  %shr_8 = shr %bits, %const_8
  %byte1 = and %shr_8, %const_0xFF
  %shr_16 = shr %bits, %const_16
  %byte2 = and %shr_16, %const_0xFF
  %shr_24 = shr %bits, %const_24
  %byte3 = mov %shr_24

  ; Rebuild in reversed order
  %b1_shift = shl %byte2, %const_8
  %b2_shift = shl %byte1, %const_16
  %b3_shift = shl %byte0, %const_24

  %temp1 = or %byte3, %b1_shift
  %temp2 = or %temp1, %b2_shift
  %reversed = or %temp2, %b3_shift

  ; 3. Determine byte count for gob encoding
  %const_0x100 = uconst 0x100u
  %const_0x10000 = uconst 0x10000u
  %const_0x1000000 = uconst 0x1000000u

  %lt_100 = ucmp.lt %reversed, %const_0x100
  %lt_10000 = ucmp.lt %reversed, %const_0x10000
  %lt_1000000 = ucmp.lt %reversed, %const_0x1000000

  %const_1 = uconst 1u
  %const_2 = uconst 2u
  %const_3 = uconst 3u
  %const_4 = uconst 4u

  ; Select byte count using nested selects
  %count_if_3_or_4 = select %lt_1000000, %const_3, %const_4
  %count_if_2_or_more = select %lt_10000, %const_2, %count_if_3_or_4
  %byte_count = select %lt_100, %const_1, %count_if_2_or_more

  ; Negated count for gob encoding
  %const_0 = uconst 0u
  %negated_count = sub %const_0, %byte_count
  %negated_count_byte = and %negated_count, %const_0xFF

  ; Extract bytes from reversed value for big-endian gob encoding
  %rev_byte0 = and %reversed, %const_0xFF
  %rev_shr_8 = shr %reversed, %const_8
  %rev_byte1 = and %rev_shr_8, %const_0xFF
  %rev_shr_16 = shr %reversed, %const_16
  %rev_byte2 = and %rev_shr_16, %const_0xFF
  %rev_shr_24 = shr %reversed, %const_24
  %rev_byte3 = mov %rev_shr_24

  ; Pack into first word: [0x03, negated_count, rev_byte3, rev_byte2]
  %const_0x03 = uconst 0x03u
  %word0_b0 = mov %const_0x03
  %word0_b1 = shl %negated_count_byte, %const_8
  %word0_b2 = shl %rev_byte3, %const_16
  %word0_b3 = shl %rev_byte2, %const_24

  %word0_temp1 = or %word0_b0, %word0_b1
  %word0_temp2 = or %word0_temp1, %word0_b2
  %word0 = or %word0_temp2, %word0_b3

  ; Pack into second word: [rev_byte1, rev_byte0, 0x00, 0x00]
  %word1_b0 = mov %rev_byte1
  %word1_b1 = shl %rev_byte0, %const_8
  %word1 = or %word1_b0, %word1_b1

  ; Return first word (for demonstration)
  ret %word0
}
"#;

        let result = parse_function(input);
        assert!(result.is_ok(), "Failed to parse gdp_encode_float32: {:?}", result.err());
        let func = result.unwrap();

        // Validate function structure
        assert_eq!(func.name, "gdp_encode_float32");
        assert_eq!(func.params.len(), 1);
        assert_eq!(func.params[0].name, "f");
        assert!(matches!(func.params[0].ty, Type::F32));

        // Validate return type
        assert!(matches!(
            func.return_type,
            ReturnType::Type(Type::U32)
        ));

        // Validate basic block exists
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].label, "entry");

        // Validate we have the expected instructions
        let instructions = &func.blocks[0].instructions;
        assert!(instructions.len() > 40); // Should have many instructions

        // Validate first instruction is bitcast
        assert!(matches!(
            instructions[0].op,
            Operation::Bitcast(_)
        ));

        // Validate we have select operations for byte count determination
        let has_select = instructions
            .iter()
            .any(|inst| matches!(inst.op, Operation::Select(_, _, _)));
        assert!(has_select);

        // Validate terminator is ret
        assert!(matches!(func.blocks[0].terminator, Terminator::Ret(_)));
    }

    #[test]
    fn test_parse_byte_reversal() {
        let input = r#"
func @swap_bytes_u32(%x: u32) -> u32 {
entry:
  %const_8  = uconst 8u
  %const_16 = uconst 16u
  %const_24 = uconst 24u
  %const_ff = uconst 0xFFu

  %byte0    = and %x, %const_ff
  %shr_8    = shr %x, %const_8
  %byte1    = and %shr_8, %const_ff
  %shr_16   = shr %x, %const_16
  %byte2    = and %shr_16, %const_ff
  %byte3    = shr %x, %const_24

  %b1_shift = shl %byte2, %const_8
  %b2_shift = shl %byte1, %const_16
  %b3_shift = shl %byte0, %const_24

  %temp1 = or %byte3, %b1_shift
  %temp2 = or %temp1, %b2_shift
  %result = or %temp2, %b3_shift

  ret %result
}
"#;

        let result = parse_function(input);
        assert!(result.is_ok(), "Failed to parse byte_reversal: {:?}", result.err());
        let func = result.unwrap();
        assert_eq!(func.name, "swap_bytes_u32");
        assert_eq!(func.blocks[0].instructions.len(), 16);
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
  br_if %ok, done, try_lock

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
  %p = mov @counter
  %old = atomic.rmw add %p, 1u ordering=acq_rel scope=device
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
  br_if %is_neg, neg, done

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
        assert!(matches!(
            done_block.instructions[0].op,
            Operation::Phi { .. }
        ));
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
  %const_64 = uconst 64u
  %is_small = ucmp.lt %x, %const_64
  br_if %is_small, small_case, big_case

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
  %const_8 = uconst 8u
  %const_0xFF = uconst 0xFFu
  %byte0 = and %bits, %const_0xFF
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
  %const_8 = uconst 8u
  %is_full = ucmp.ge %bits, %const_8
  br_if %is_full, full_exit, check_wrap

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
        // Last 5 instructions from gdp_encode_float32 entry block (around line 119-133)
        let input = r#"
global @gdp_buffer : ptr[global]<u32>

func @test() -> void {
entry:
  %const_write_head_idx = uconst 0u
  %const_max_loops_idx = uconst 2u
  %ring_size = uconst 4093u

  ; Get pointers
  %write_head_ptr = gep @gdp_buffer, %const_write_head_idx, stride=4
  %max_loops_ptr = gep @gdp_buffer, %const_max_loops_idx, stride=4

  ; Load max_loops
  %max_loops = load %max_loops_ptr

  ; Atomic load current write_head to check if over limit
  %current_write_head = atomic.load %write_head_ptr ordering=acq_rel scope=device

  ; Calculate loop count: current_write_head / ring_size
  %current_loop = udiv %current_write_head, %ring_size

  ; Check if we're over the limit
  %is_full = ucmp.ge %current_loop, %max_loops
  br_if %is_full, full_exit, check_wrap

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
        assert!(result.is_ok(), "Should parse last 5 instructions from gdp_encode_float32");
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
                    let before_next = &content[func_start..func_start+next_func_pos];
                    if let Some(closing_brace_offset) = before_next.rfind('}') {
                        // Extract from the beginning (including global) to end of first function
                        let module_content = &content[..func_start+closing_brace_offset+1];
                        eprintln!("Extracted module with {} bytes", module_content.len());
                        eprintln!("Starts with: {:?}", &module_content[..100.min(module_content.len())]);
                        eprintln!("Ends with: {:?}", &module_content[module_content.len().saturating_sub(100)..]);

                        // Try parsing with parse_module
                        let result = parse_module(module_content);
                        if let Err(e) = &result {
                            eprintln!("Parse error with parse_module: {:?}", e);
                            panic!("Failed to parse module: {:?}", e);
                        } else {
                            eprintln!("parse_module succeeded! Got {} globals and {} functions",
                                result.as_ref().unwrap().globals.len(),
                                result.as_ref().unwrap().functions.len());
                            assert_eq!(result.as_ref().unwrap().globals.len(), 1, "Should have 1 global");
                            assert_eq!(result.as_ref().unwrap().functions.len(), 1, "Should have 1 function");
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

    #[test]
    fn test_parse_gdp_encode_float32_full() {
        // Full gdp_encode_float32 from wyn-core/builtins/gdp_builtins.gasm
        let input = r#"
func @gdp_encode_float32(%f: f32) -> void {
entry:
  ; 1. Bitcast f32 to u32
  %bits = bitcast %f

  ; 2. Byte-reverse: swap bytes in SPIR-V
  ; Extract bytes from the u32 value
  %const_8 = uconst 8u
  %const_16 = uconst 16u
  %const_24 = uconst 24u
  %const_0xFF = uconst 0xFFu

  %byte0 = and %bits, %const_0xFF
  %shr_8 = shr %bits, %const_8
  %byte1 = and %shr_8, %const_0xFF
  %shr_16 = shr %bits, %const_16
  %byte2 = and %shr_16, %const_0xFF
  %shr_24 = shr %bits, %const_24
  %byte3 = mov %shr_24

  ; Rebuild in reversed order
  %b1_shift = shl %byte2, %const_8
  %b2_shift = shl %byte1, %const_16
  %b3_shift = shl %byte0, %const_24

  %temp1 = or %byte3, %b1_shift
  %temp2 = or %temp1, %b2_shift
  %reversed = or %temp2, %b3_shift

  ; 3. Determine byte count for gob encoding
  %const_0x100 = uconst 0x100u
  %const_0x10000 = uconst 0x10000u
  %const_0x1000000 = uconst 0x1000000u

  %lt_100 = ucmp.lt %reversed, %const_0x100
  %lt_10000 = ucmp.lt %reversed, %const_0x10000
  %lt_1000000 = ucmp.lt %reversed, %const_0x1000000

  %const_1 = uconst 1u
  %const_2 = uconst 2u
  %const_3 = uconst 3u
  %const_4 = uconst 4u

  ; Select byte count using nested selects
  %count_if_3_or_4 = select %lt_1000000, %const_3, %const_4
  %count_if_2_or_more = select %lt_10000, %const_2, %count_if_3_or_4
  %byte_count = select %lt_100, %const_1, %count_if_2_or_more

  ; Negated count for gob encoding
  %const_0 = uconst 0u
  %negated_count = sub %const_0, %byte_count
  %negated_count_byte = and %negated_count, %const_0xFF

  ; Extract bytes from reversed value for big-endian gob encoding
  %rev_byte0 = and %reversed, %const_0xFF
  %rev_shr_8 = shr %reversed, %const_8
  %rev_byte1 = and %rev_shr_8, %const_0xFF
  %rev_shr_16 = shr %reversed, %const_16
  %rev_byte2 = and %rev_shr_16, %const_0xFF
  %rev_shr_24 = shr %reversed, %const_24
  %rev_byte3 = mov %rev_shr_24

  ; Pack into first word: [0x03, negated_count, rev_byte3, rev_byte2]
  %const_0x03 = uconst 0x03u
  %word0_b0 = mov %const_0x03
  %word0_b1 = shl %negated_count_byte, %const_8
  %word0_b2 = shl %rev_byte3, %const_16
  %word0_b3 = shl %rev_byte2, %const_24

  %word0_temp1 = or %word0_b0, %word0_b1
  %word0_temp2 = or %word0_temp1, %word0_b2
  %word0 = or %word0_temp2, %word0_b3

  ; Pack into second word: [rev_byte1, rev_byte0, 0x00, 0x00]
  %word1_b0 = mov %rev_byte1
  %word1_b1 = shl %rev_byte0, %const_8
  %word1 = or %word1_b0, %word1_b1

  ; === Reserve 2 words with max_loops check and wrap-to-zero logic ===
  %ring_size = uconst 4093u
  %const_write_head_idx = uconst 0u
  %const_max_loops_idx = uconst 2u
  %const_header_offset = uconst 3u
  %count_to_reserve = uconst 2u

  ; Get pointers
  %write_head_ptr = gep @gdp_buffer, %const_write_head_idx, stride=4
  %max_loops_ptr = gep @gdp_buffer, %const_max_loops_idx, stride=4

  ; Load max_loops
  %max_loops = load %max_loops_ptr

  ; Atomic load current write_head to check if over limit
  %current_write_head = atomic.load %write_head_ptr ordering=acq_rel scope=device

  ; Calculate loop count: current_write_head / ring_size
  %current_loop = udiv %current_write_head, %ring_size

  ; Check if we're over the limit
  %is_full = ucmp.ge %current_loop, %max_loops
  br_if %is_full, full_exit, check_wrap

check_wrap:
  ; Calculate ring position
  %ring_pos = urem %current_write_head, %ring_size

  ; Check if allocation would cross boundary: ring_pos + count > ring_size
  %pos_plus_count = add %ring_pos, %count_to_reserve
  %would_cross = ucmp.gt %pos_plus_count, %ring_size

  br_if %would_cross, wrap_to_zero, normal_reserve

wrap_to_zero:
  ; Skip to next ring: increment by (ring_size - ring_pos) + count
  %skip_amount = sub %ring_size, %ring_pos
  %total_skip = add %skip_amount, %count_to_reserve
  %skipped_pos = atomic.rmw add %write_head_ptr, %total_skip ordering=acq_rel scope=device
  ; After skip, we're at position 0 in the ring
  %wrapped_index = uconst 0u
  br write_data

normal_reserve:
  ; Normal allocation: atomically increment and get position
  %global_pos = atomic.rmw add %write_head_ptr, %count_to_reserve ordering=acq_rel scope=device
  %normal_index = urem %global_pos, %ring_size
  br write_data

write_data:
  ; Phi to get the starting index
  %start_index = phi u32 [%wrapped_index, wrap_to_zero], [%normal_index, normal_reserve]

  ; Calculate actual buffer indices (add 3 for header)
  %data_index0 = add %start_index, %const_header_offset
  %const_1_offset = uconst 1u
  %index_plus_1 = add %start_index, %const_1_offset
  %index_plus_1_mod = urem %index_plus_1, %ring_size
  %data_index1 = add %index_plus_1_mod, %const_header_offset

  ; Get element pointers and store
  %ptr0 = gep @gdp_buffer, %data_index0, stride=4
  %ptr1 = gep @gdp_buffer, %data_index1, stride=4
  store %ptr0, %word0
  store %ptr1, %word1
  ret

full_exit:
  ; Buffer full, skip write
  ret
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Failed to parse gdp_encode_float32");

        let func = result.unwrap();
        assert_eq!(func.name, "gdp_encode_float32");
        assert_eq!(func.params.len(), 1);
        assert_eq!(func.blocks.len(), 6); // entry, check_wrap, wrap_to_zero, normal_reserve, write_data, full_exit
    }

    #[test]
    fn test_parse_gdp_encode_uint_full() {
        // Full gdp_encode_uint from wyn-core/builtins/gdp_builtins.gasm
        let input = r#"
func @gdp_encode_uint(%value: u32) -> void {
entry:
  ; Check if value fits in inline encoding (0-63)
  %const_64 = uconst 64u
  %is_small = ucmp.lt %value, %const_64

  br_if %is_small, encode_inline, encode_full

encode_inline:
  ; Inline encoding: shift value left by 2
  %const_2 = uconst 2u
  %inline_word = shl %value, %const_2

  ; Reserve 1 word with max_loops check (no wrap-to-zero needed for single word)
  %ring_size = uconst 4093u
  %const_write_head_idx = uconst 0u
  %const_max_loops_idx = uconst 2u
  %const_header_offset = uconst 3u
  %const_1 = uconst 1u

  %write_head_ptr = gep @gdp_buffer, %const_write_head_idx, stride=4
  %max_loops_ptr = gep @gdp_buffer, %const_max_loops_idx, stride=4
  %max_loops = load %max_loops_ptr
  %current_write_head = atomic.load %write_head_ptr ordering=acq_rel scope=device
  %current_loop = udiv %current_write_head, %ring_size
  %is_full = ucmp.ge %current_loop, %max_loops
  br_if %is_full, exit_inline, reserve_inline

reserve_inline:
  %start_index_inline = atomic.rmw add %write_head_ptr, %const_1 ordering=acq_rel scope=device
  %index_raw = urem %start_index_inline, %ring_size
  %index = add %index_raw, %const_header_offset
  %ptr = gep @gdp_buffer, %index, stride=4
  store %ptr, %inline_word
  ret

exit_inline:
  ret

encode_full:
  ; Full encoding: type byte 0x00 + gob uint
  ; Determine byte count (1-4 bytes for u32)
  %const_0x100 = uconst 0x100u
  %const_0x10000 = uconst 0x10000u
  %const_0x1000000 = uconst 0x1000000u

  %lt_100 = ucmp.lt %value, %const_0x100
  %lt_10000 = ucmp.lt %value, %const_0x10000
  %lt_1000000 = ucmp.lt %value, %const_0x1000000

  %const_1 = uconst 1u
  %const_2_bytes = uconst 2u
  %const_3 = uconst 3u
  %const_4 = uconst 4u

  %count_if_3_or_4 = select %lt_1000000, %const_3, %const_4
  %count_if_2_or_more = select %lt_10000, %const_2_bytes, %count_if_3_or_4
  %byte_count = select %lt_100, %const_1, %count_if_2_or_more

  ; Negated count for gob encoding
  %const_0 = uconst 0u
  %negated_count = sub %const_0, %byte_count
  %const_0xFF = uconst 0xFFu
  %negated_count_byte = and %negated_count, %const_0xFF

  ; Extract bytes from value (big-endian)
  %byte0 = and %value, %const_0xFF
  %const_8 = uconst 8u
  %shr_8 = shr %value, %const_8
  %byte1 = and %shr_8, %const_0xFF
  %const_16 = uconst 16u
  %shr_16 = shr %value, %const_16
  %byte2 = and %shr_16, %const_0xFF
  %const_24 = uconst 24u
  %shr_24 = shr %value, %const_24
  %byte3 = mov %shr_24

  ; Pack into first word: [0x00, negated_count, byte3, byte2]
  %const_0x00 = uconst 0x00u
  %word0_b0 = mov %const_0x00
  %word0_b1 = shl %negated_count_byte, %const_8
  %word0_b2 = shl %byte3, %const_16
  %word0_b3 = shl %byte2, %const_24

  %word0_temp1 = or %word0_b0, %word0_b1
  %word0_temp2 = or %word0_temp1, %word0_b2
  %word0 = or %word0_temp2, %word0_b3

  ; Pack into second word: [byte1, byte0, 0x00, 0x00]
  %word1_b0 = mov %byte1
  %word1_b1 = shl %byte0, %const_8
  %word1 = or %word1_b0, %word1_b1

  ; Reserve 2 words with wrap-to-zero logic (same as float32)
  %ring_size2 = uconst 4093u
  %const_write_head_idx2 = uconst 0u
  %const_max_loops_idx2 = uconst 2u
  %const_header_offset2 = uconst 3u
  %count_to_reserve2 = uconst 2u

  %write_head_ptr2 = gep @gdp_buffer, %const_write_head_idx2, stride=4
  %max_loops_ptr2 = gep @gdp_buffer, %const_max_loops_idx2, stride=4
  %max_loops2 = load %max_loops_ptr2
  %current_write_head2 = atomic.load %write_head_ptr2 ordering=acq_rel scope=device
  %current_loop2 = udiv %current_write_head2, %ring_size2
  %is_full2 = ucmp.ge %current_loop2, %max_loops2
  br_if %is_full2, full_exit2, check_wrap2

check_wrap2:
  %ring_pos2 = urem %current_write_head2, %ring_size2
  %pos_plus_count2 = add %ring_pos2, %count_to_reserve2
  %would_cross2 = ucmp.gt %pos_plus_count2, %ring_size2
  br_if %would_cross2, wrap_to_zero2, normal_reserve2

wrap_to_zero2:
  %skip_amount2 = sub %ring_size2, %ring_pos2
  %total_skip2 = add %skip_amount2, %count_to_reserve2
  %skipped_pos2 = atomic.rmw add %write_head_ptr2, %total_skip2 ordering=acq_rel scope=device
  %wrapped_index2 = uconst 0u
  br write_data2

normal_reserve2:
  %global_pos2 = atomic.rmw add %write_head_ptr2, %count_to_reserve2 ordering=acq_rel scope=device
  %normal_index2 = urem %global_pos2, %ring_size2
  br write_data2

write_data2:
  %start_index2 = phi u32 [%wrapped_index2, wrap_to_zero2], [%normal_index2, normal_reserve2]
  %data_index0_2 = add %start_index2, %const_header_offset2
  %const_1_offset2 = uconst 1u
  %index_plus_1_2 = add %start_index2, %const_1_offset2
  %index_plus_1_mod2 = urem %index_plus_1_2, %ring_size2
  %data_index1_2 = add %index_plus_1_mod2, %const_header_offset2
  %ptr0_2 = gep @gdp_buffer, %data_index0_2, stride=4
  %ptr1_2 = gep @gdp_buffer, %data_index1_2, stride=4
  store %ptr0_2, %word0
  store %ptr1_2, %word1
  ret

full_exit2:
  ret
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Failed to parse gdp_encode_uint");

        let func = result.unwrap();
        assert_eq!(func.name, "gdp_encode_uint");
        assert_eq!(func.params.len(), 1);
        // Should have: entry, encode_inline, exit_inline, reserve_inline, encode_full, check_wrap2, wrap_to_zero2, normal_reserve2, write_data2, full_exit2
        assert!(func.blocks.len() >= 8, "Expected at least 8 blocks, got {}", func.blocks.len());
    }

    #[test]
    fn test_parse_gdp_encode_string_full() {
        // Full gdp_encode_string from wyn-core/builtins/gdp_builtins.gasm
        let input = r#"
func @gdp_encode_string(%str_ptr: ptr[global]<u8>, %str_len: u32) -> void {
entry:
  ; For strings < 64 bytes: type byte is (length << 2) | 0x02
  ; For simplicity, assume length < 64 (TODO: handle longer strings)
  %const_64 = uconst 64u
  %is_short = ucmp.lt %str_len, %const_64
  br_if %is_short, encode_short, encode_exit

encode_short:
  ; Type byte: (str_len << 2) | 0x02
  %const_2 = uconst 2u
  %len_shifted = shl %str_len, %const_2
  %const_0x02 = uconst 0x02u
  %type_byte = or %len_shifted, %const_0x02

  ; Calculate total bytes: 1 (type) + str_len
  %const_1 = uconst 1u
  %total_bytes = add %str_len, %const_1

  ; Calculate words needed: (total_bytes + 3) / 4
  %const_3 = uconst 3u
  %const_4 = uconst 4u
  %bytes_rounded = add %total_bytes, %const_3
  %total_words = udiv %bytes_rounded, %const_4

  ; Reserve space (simplified - no max_loops check for now)
  ; TODO: Add full reservation logic with wrap-to-zero
  %const_write_head_idx = uconst 0u
  %write_head_ptr = gep @gdp_buffer, %const_write_head_idx, stride=4
  %start_pos = atomic.rmw add %write_head_ptr, %total_words ordering=acq_rel scope=device

  ; Calculate ring position
  %ring_size = uconst 4093u
  %ring_pos = urem %start_pos, %ring_size

  ; Pack and write words
  ; First byte is type_byte, then string bytes, then padding
  ; We'll pack 4 bytes per word: [b0, b1, b2, b3]

  ; Initialize loop: byte_idx = 0, word_idx = 0, current_word = type_byte
  %const_0 = uconst 0u
  %init_word = mov %type_byte
  br pack_loop

pack_loop:
  ; Phi: byte_idx, word_idx, current_word, shift
  %byte_idx = phi u32 [%const_0, encode_short], [%next_byte_idx, pack_continue], [%next_byte_idx2, write_word]
  %word_idx = phi u32 [%const_0, encode_short], [%word_idx, pack_continue], [%next_word_idx, write_word]
  %current_word = phi u32 [%init_word, encode_short], [%updated_word, pack_continue], [%const_0, write_word]
  %shift = phi u32 [%const_8, encode_short], [%next_shift, pack_continue], [%const_0, write_word]

  ; Check if we're done with all bytes
  %done = ucmp.ge %byte_idx, %str_len
  br_if %done, finalize, pack_byte

pack_byte:
  ; Load byte from string
  %byte_ptr = gep %str_ptr, %byte_idx, stride=1
  %byte_val = load %byte_ptr

  ; Shift byte and OR into current_word
  %byte_shifted = shl %byte_val, %shift
  %updated_word = or %current_word, %byte_shifted

  ; Increment byte index
  %next_byte_idx = add %byte_idx, %const_1

  ; Calculate next shift (shift + 8)
  %const_8 = uconst 8u
  %next_shift = add %shift, %const_8

  ; Check if word is full (shift == 32)
  %const_32 = uconst 32u
  %word_full = ucmp.eq %next_shift, %const_32
  br_if %word_full, write_word, pack_continue

pack_continue:
  br pack_loop

write_word:
  ; Write current_word to buffer
  %const_header_offset = uconst 3u
  %buffer_idx_base = add %ring_pos, %word_idx
  %buffer_idx_wrapped = urem %buffer_idx_base, %ring_size
  %buffer_idx = add %buffer_idx_wrapped, %const_header_offset
  %word_ptr = gep @gdp_buffer, %buffer_idx, stride=4
  store %word_ptr, %updated_word

  ; Increment word index
  %next_word_idx = add %word_idx, %const_1

  ; Reset for next word (next_byte_idx already incremented)
  %next_byte_idx2 = mov %next_byte_idx
  br pack_loop

finalize:
  ; If there's a partial word, write it
  %has_partial = ucmp.ne %shift, %const_0
  br_if %has_partial, write_final, encode_exit

write_final:
  %const_header_offset2 = uconst 3u
  %final_idx_base = add %ring_pos, %word_idx
  %final_idx_wrapped = urem %final_idx_base, %ring_size
  %final_idx = add %final_idx_wrapped, %const_header_offset2
  %final_ptr = gep @gdp_buffer, %final_idx, stride=4
  store %final_ptr, %current_word
  ret

encode_exit:
  ret
}
"#;

        let result = parse_function(input);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        assert!(result.is_ok(), "Failed to parse gdp_encode_string");

        let func = result.unwrap();
        assert_eq!(func.name, "gdp_encode_string");
        assert_eq!(func.params.len(), 2);
        // Should have: entry, encode_short, pack_loop, pack_byte, pack_continue, write_word, finalize, write_final, encode_exit
        assert!(func.blocks.len() >= 8, "Expected at least 8 blocks, got {}", func.blocks.len());
    }
}

