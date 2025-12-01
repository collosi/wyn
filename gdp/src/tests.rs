use super::*;

/// Helper to construct u32 buffer from byte sequence
/// Handles endianness correctly for testing
fn bytes_to_u32s(bytes: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    for chunk in bytes.chunks(4) {
        let mut word = 0u32;
        for (i, &byte) in chunk.iter().enumerate() {
            word |= (byte as u32) << (i * 8);
        }
        result.push(word);
    }
    result
}

#[test]
fn test_decode_small_uint() {
    // Small values (< 128) are encoded as a single byte
    let buffer = [7u32, 0, 0, 0];
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_uint().unwrap(), 7);
    assert_eq!(decoder.position(), 4); // Aligned to word boundary
}

#[test]
fn test_decode_zero() {
    let buffer = [0u32];
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_uint().unwrap(), 0);
}

#[test]
fn test_decode_127() {
    // Largest single-byte value
    let buffer = [127u32];
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_uint().unwrap(), 127);
}

#[test]
fn test_decode_large_uint_256() {
    // 256 encoded as: FE (negated byte count -2) 01 00
    let buffer = bytes_to_u32s(&[0xFE, 0x01, 0x00]);
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_uint().unwrap(), 256);
}

#[test]
fn test_decode_large_uint_65535() {
    // 65535 = 0xFFFF, needs 2 bytes
    // Encoded as: FE FF FF
    let buffer = bytes_to_u32s(&[0xFE, 0xFF, 0xFF]);
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_uint().unwrap(), 65535);
}

#[test]
fn test_decode_positive_int() {
    // Positive integers: value << 1
    // 64 << 1 = 128, but 128 is not < 128, so it needs multi-byte encoding
    // 128 = 0x80, needs 1 byte
    // Encoded as: FF (negated count = -1) 80
    let buffer = bytes_to_u32s(&[0xFF, 0x80]);
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_int().unwrap(), 64);
}

#[test]
fn test_decode_negative_int() {
    // -1: (~0 << 1) | 1 = (0xFFFFFFFFFFFFFFFF << 1) | 1 = 0xFFFFFFFFFFFFFFFF
    // Which is 1 as unsigned, encodes as single byte 0x01
    let buffer = [1u32];
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_int().unwrap(), -1);
}

#[test]
fn test_decode_negative_129() {
    // -129: !(-129i64) = 128
    // (128 << 1) | 1 = 257
    // 257 encodes as: FE 01 01
    let buffer = bytes_to_u32s(&[0xFE, 0x01, 0x01]);
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_int().unwrap(), -129);
}

#[test]
fn test_decode_string_short() {
    // String "hi" = length 2, then 'h' (0x68), 'i' (0x69)
    // Bytes: 02 68 69 [padding to word]
    let buffer = bytes_to_u32s(&[0x02, 0x68, 0x69, 0x00]);
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_string().unwrap(), "hi");
}

#[test]
fn test_decode_string_empty() {
    // Empty string: length 0
    let buffer = [0u32];
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_string().unwrap(), "");
}

#[test]
fn test_decode_string_with_alignment() {
    // String "test" = length 4, then 't' 'e' 's' 't'
    // Bytes: 04 74 65 73 74 [padding to word boundary]
    // After 5 bytes, need 3 bytes padding to reach 8
    let buffer = bytes_to_u32s(&[0x04, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00, 0x00]);
    let mut decoder = GdpDecoder::new(&buffer);
    assert_eq!(decoder.decode_string().unwrap(), "test");
    assert_eq!(decoder.position(), 8); // Aligned to next word
}

#[test]
fn test_word_alignment_between_values() {
    // Two small uints: 5, then 7
    // Each aligns to word boundary
    // Bytes: 05 [pad pad pad] 07 [pad pad pad]
    let buffer = [5u32, 7u32];
    let mut decoder = GdpDecoder::new(&buffer);

    assert_eq!(decoder.decode_uint().unwrap(), 5);
    assert_eq!(decoder.position(), 4); // Advanced to word boundary

    assert_eq!(decoder.decode_uint().unwrap(), 7);
    assert_eq!(decoder.position(), 8);
}

#[test]
fn test_multiple_values_mixed() {
    // Encode: uint(42), int(-5), string("ok")
    // 42: 0x2A
    // -5: !(-5i64) = 4, (4 << 1) | 1 = 9: 0x09
    // "ok": len=2, 'o'=0x6F, 'k'=0x6B: 02 6F 6B

    // Bytes with word alignment:
    // 0x2A [pad pad pad] = 4 bytes
    // 0x09 [pad pad pad] = 4 bytes
    // 0x02 0x6F 0x6B [pad] = 4 bytes

    let buffer = bytes_to_u32s(&[
        0x2A, 0x00, 0x00, 0x00, // uint(42)
        0x09, 0x00, 0x00, 0x00, // int(-5)
        0x02, 0x6F, 0x6B, 0x00, // string("ok")
    ]);
    let mut decoder = GdpDecoder::new(&buffer);

    assert_eq!(decoder.decode_uint().unwrap(), 42);
    assert_eq!(decoder.decode_int().unwrap(), -5);
    assert_eq!(decoder.decode_string().unwrap(), "ok");
}

#[test]
fn test_is_empty() {
    let buffer = [5u32];
    let mut decoder = GdpDecoder::new(&buffer);

    assert!(!decoder.is_empty());
    decoder.decode_uint().unwrap();
    assert!(decoder.is_empty());
}

#[test]
fn test_remaining() {
    let buffer = [5u32, 7u32];
    let mut decoder = GdpDecoder::new(&buffer);

    assert_eq!(decoder.remaining(), 8);
    decoder.decode_uint().unwrap();
    assert_eq!(decoder.remaining(), 4);
    decoder.decode_uint().unwrap();
    assert_eq!(decoder.remaining(), 0);
}

#[test]
fn test_error_unexpected_end() {
    // FC means need 4 bytes, but bytes_to_u32s pads to exactly 4 bytes total
    // [FC 00 00 00] = 1 indicator byte + 3 padding bytes, but we need 4 data bytes
    // So we're 1 byte short
    let buffer = bytes_to_u32s(&[0xFC]);
    let mut decoder = GdpDecoder::new(&buffer);

    assert!(decoder.decode_uint().is_err());
}

#[test]
fn test_error_string_too_long() {
    // String claims to be 100 bytes long but buffer is only 8 bytes
    let buffer = [100u32, 0u32];
    let mut decoder = GdpDecoder::new(&buffer);

    assert!(decoder.decode_string().is_err());
}

#[test]
fn test_decode_raw_debug_output() {
    // Raw hex from actual debug output (starting after header at word 3):
    // 61726616 00003a67 c000fd03 000043c7 c000fd03 00004395 6c6f6312 0000003a 0000fe03 00007fc0
    //
    // Expected decoding:
    // - "frag:" string
    // - float (should be ~400.0 = pos.x)
    // - float (should be ~300.0 = pos.y)
    // - "col:" string
    // - float (color.x)
    // etc.

    let buffer: [u32; 10] = [
        0x61726616, // "frag:" - type byte 0x16 = (5 << 2) | 0x02 = 5-char string
        0x00003a67, // "g:" + padding
        0xc000fd03, // float: type 0x03, neg_count 0xfd (-3)
        0x000043c7, 0xc000fd03, // second float
        0x00004395, 0x6c6f6312, // "col:" - type byte 0x12 = (4 << 2) | 0x02 = 4-char string
        0x0000003a, 0x0000fe03, // float: type 0x03, neg_count 0xfe (-2)
        0x00007fc0,
    ];

    let mut decoder = GdpDecoder::new(&buffer);

    // Decode and print each value
    let val1 = decoder.decode_value().unwrap();
    println!("val1: {:?}", val1);

    let val2 = decoder.decode_value().unwrap();
    println!("val2: {:?}", val2);

    let val3 = decoder.decode_value().unwrap();
    println!("val3: {:?}", val3);

    let val4 = decoder.decode_value().unwrap();
    println!("val4: {:?}", val4);

    let val5 = decoder.decode_value().unwrap();
    println!("val5: {:?}", val5);

    // Check string decoding
    assert!(matches!(val1, GdpValue::String(s) if s == "frag:"));
    assert!(matches!(val4, GdpValue::String(s) if s == "col:"));

    // Print float values to see what we're getting
    if let GdpValue::Float32(f) = val2 {
        println!("First float raw bits: 0x{:08X}, value: {}", f.to_bits(), f);
    }
    if let GdpValue::Float32(f) = val3 {
        println!("Second float raw bits: 0x{:08X}, value: {}", f.to_bits(), f);
    }
}
