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
}
