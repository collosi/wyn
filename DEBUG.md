# GPU Debug Protocol (GDP) - Wire Format Specification

## Overview

GDP is a simplified gob-like binary protocol designed for GPU debug output. It addresses the constraint that GPU storage buffers use u32 arrays while maintaining compatibility with the gob encoding specification for integers, booleans, and strings.

## Ring Buffer Structure

The debug buffer is organized as a single ring buffer with the following layout:

```
Offset | Size  | Description
-------|-------|-------------
0      | 4B    | write_head: atomic u32 cursor (counts total words written)
4      | 4B    | read_head: u32 (start of valid data, updated by host)
8      | 4*N B | data: ring buffer of N u32 words (N=4094 for 16KB total buffer)
```

**Total size**: 4096 u32s = 16KB (configurable)

**Access pattern:**
- GPU shaders use atomic add on `write_head` to reserve space
- Each write is placed at index `(write_head % 4094) + 2` to wrap around
- Host reads `write_head` and `read_head` to determine valid data range
- When buffer wraps, host updates `read_head` to track oldest valid entry
- Host decodes GDP-encoded data from the valid range

## Design Rationale

Many GPU platforms (particularly Metal, WebGPU) do not support 8-bit integer types in storage buffers. To work around this while maintaining an efficient byte-level protocol:

1. **Physical storage**: Data is stored in u32 arrays (required by GPU)
2. **Logical encoding**: Data is treated as a big-endian byte stream
3. **Word alignment**: Values are padded to 4-byte boundaries to prevent partial reads across GPU word boundaries
4. **Ring buffer**: Wraps around to allow continuous operation even if buffer fills

## Supported Types

GDP supports a minimal set of types sufficient for debug output:

- **Unsigned integers** (`uint`) - arbitrary precision
- **Signed integers** (`int`) - arbitrary precision
- **Booleans** (`bool`)
- **Strings** (`string`) - UTF-8 encoded

## Physical Layout

### Storage Format
```
GPU Buffer: [u32, u32, u32, ...]
            ↓ transmute
Byte View:  [u8, u8, u8, u8, u8, u8, ...]
```

The decoder immediately transmutes the u32 array to a byte slice and operates on bytes.

### Word Alignment Rule

After encoding each complete value, if the current position is not at a 4-byte boundary (position % 4 ≠ 0), advance to the next word boundary by adding padding bytes.

**Example:**
```
Value 1: [0x2A] → [0x2A 0x00 0x00 0x00]  (padded to 4 bytes)
Value 2: [0x09] → [0x09 0x00 0x00 0x00]  (padded to 4 bytes)
```

## Type Tags

GDP uses type-tagged encoding where each value begins with a type byte that encodes both the type and (optionally) a small value/length:

```
Type Byte Format: VVVVVVTT

Bits 0-1 (TT): Type tag
  00 = unsigned integer (uint)
  01 = signed integer (int)
  10 = string
  11 = boolean

Bits 2-7 (VVVVVV): Inline value/length (6 bits, range 0-63)
  0 = Full value/length follows (encoded as GDP uint)
  1-63 = Inline value/length (optimization for small values)
```

**Examples:**
- `uint 5`: `[0x14]` → `(5 << 2) | 0b00` = 0x14 (1 byte total)
- `uint 200`: `[0x00, 0xC8]` → type=00, inline=0 (value follows), then 200 (2 bytes total)
- `string "hi"`: `[0x0A, 'h', 'i', 0x00]` → `(2 << 2) | 0b10`, then 2 chars, then padding

## Encoding Rules

### Unsigned Integers

Type tag: `0b00`

With type-tagged format:

**Small values (< 128):**
- Encoded as a single byte with that value

**Large values (≥ 128):**
- Byte 0: Negated byte count (count of following bytes)
- Bytes 1-N: Value in big-endian format

**Negated byte count calculation:**
```
negated_count = ~count + 1 = -count (in two's complement)
```

**Examples:**
```
0       → [0x00]
7       → [0x07]
127     → [0x7F]
128     → [0xFF 0x80]           (FF = -1, need 1 byte)
256     → [0xFE 0x01 0x00]      (FE = -2, need 2 bytes)
65535   → [0xFE 0xFF 0xFF]      (FE = -2, need 2 bytes)
```

**With word alignment:**
```
7       → [0x07 0x00 0x00 0x00]
256     → [0xFE 0x01 0x00 0x00] (only 3 bytes used, 1 byte padding)
```

### Signed Integers

Following the gob specification for signed integer encoding:

**Encoding algorithm:**
```rust
let u: u64 = if i < 0 {
    (!i << 1) | 1    // complement value, set bit 0
} else {
    i << 1           // don't complement, clear bit 0
};
encode_unsigned(u)
```

**Decoding algorithm:**
```rust
let u = decode_unsigned();
let value = (u >> 1) as i64;
let result = if (u & 1) != 0 {
    !value           // bit 0 set: complement
} else {
    value            // bit 0 clear: use as-is
};
```

**Bit layout:**
- Bit 0: Complement flag (1 = negative, 0 = positive)
- Bits 1+: Absolute value (or its complement)

**Examples:**
```
0       → 0 << 1 = 0               → [0x00 0x00 0x00 0x00]
1       → 1 << 1 = 2               → [0x02 0x00 0x00 0x00]
-1      → (!0 << 1) | 1 = 1        → [0x01 0x00 0x00 0x00]
64      → 64 << 1 = 128            → [0xFF 0x80 0x00 0x00]
-64     → (!63 << 1) | 1 = 127     → [0x7F 0x00 0x00 0x00]
-129    → (!128 << 1) | 1 = 257    → [0xFE 0x01 0x01 0x00]
```

### Booleans

Encoded as unsigned integers:
- `false` → 0
- `true` → 1

**Examples:**
```
false   → [0x00 0x00 0x00 0x00]
true    → [0x01 0x00 0x00 0x00]
```

### Strings

Following the gob specification:

1. Encode length as unsigned integer (WITHOUT word alignment)
2. Encode UTF-8 bytes
3. Apply word alignment to the entire string value

**Important:** The length is part of the string value, not a separate value. Word alignment happens after both length and data.

**Examples:**
```
""      → [0x00 0x00 0x00 0x00]
        (length=0, then padding to word)

"hi"    → [0x02 0x68 0x69 0x00]
        (length=2, 'h'=0x68, 'i'=0x69, then 1 byte padding)

"test"  → [0x04 0x74 0x65 0x73] [0x74 0x00 0x00 0x00]
        (length=4, then 4 chars, then 3 bytes padding)
```

## Complete Examples

### Example 1: Simple Values
```
Encode: uint(42), bool(true)

Bytes:
[0x2A 0x00 0x00 0x00]  // uint(42)
[0x01 0x00 0x00 0x00]  // bool(true)

As u32 buffer:
[0x0000002A, 0x00000001]
```

### Example 2: Mixed Types
```
Encode: uint(256), int(-5), string("ok")

Bytes:
[0xFE 0x01 0x00 0x00]  // uint(256)
[0x09 0x00 0x00 0x00]  // int(-5): (!4 << 1) | 1 = 9
[0x02 0x6F 0x6B 0x00]  // string("ok"): len=2, 'o', 'k', pad

As u32 buffer:
[0x000001FE, 0x00000009, 0x006B6F02]
```

### Example 3: Debug Output Pattern
```
Encode: string("val:"), int(42), string("done")

Bytes:
[0x04 0x76 0x61 0x6C] [0x3A 0x00 0x00 0x00]  // "val:" (len=4)
[0x54 0x00 0x00 0x00]                         // int(42): 42 << 1 = 84
[0x04 0x64 0x6F 0x6E] [0x65 0x00 0x00 0x00]  // "done" (len=4)

As u32 buffer:
[0x6C616704, 0x0000003A, 0x00000054, 0x6E6F6404, 0x00000065]
```

## Implementation Notes

### Decoder

The decoder (`gdp` crate) provides:

```rust
pub struct GdpDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GdpDecoder<'a> {
    pub fn new(buffer: &'a [u32]) -> Self;
    pub fn decode_uint(&mut self) -> Result<u64, &'static str>;
    pub fn decode_int(&mut self) -> Result<i64, &'static str>;
    pub fn decode_bool(&mut self) -> Result<bool, &'static str>;
    pub fn decode_string(&mut self) -> Result<String, &'static str>;
}
```

### Encoder (To Be Implemented)

The encoder will be implemented in SPIR-V/shader code to write debug data from GPU:

```glsl
// Conceptual interface (actual implementation in Wyn compiler)
void debug_uint(uint value);
void debug_int(int value);
void debug_bool(bool value);
void debug_str(string value);
```

## Differences from Gob

GDP differs from full gob in the following ways:

1. **Reduced type set**: Only uint, int, bool, string (no structs, arrays, maps, etc.)
2. **No type definitions**: All values are self-describing at the primitive level
3. **Word alignment**: Required for GPU compatibility, not present in gob
4. **No streaming overhead**: No message byte counts or type IDs
5. **Fixed encoding**: No optimization for type reuse across messages

## Future Extensions

Potential additions while maintaining compatibility:

1. **Float encoding**: IEEE 754 following gob's float64 encoding
2. **Array encoding**: Length prefix + elements (with alignment)
3. **Structured records**: Field count + (field_id, value) pairs

All extensions must maintain the word alignment invariant.

## References

- [Go gob package documentation](https://pkg.go.dev/encoding/gob)
- [Gobs of data (design article)](https://blog.golang.org/gobs-of-data)
- WebGPU Storage Buffer alignment requirements
- SPIR-V specification for buffer access
