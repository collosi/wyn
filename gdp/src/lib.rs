//! GPU Debug Protocol (GDP)
//!
//! A simplified gob-like protocol designed for GPU debug output.
//! Data is stored in u32 arrays (required for GPU storage buffers) but
//! logically treated as a big-endian byte stream.
//!
//! Supported types: unsigned integers, signed integers, booleans, strings.
//!
//! Key feature: word alignment - after each value, if not at a 4-byte boundary,
//! skip to the next word boundary to prevent partial reads crossing GPU word boundaries.

#[cfg(test)]
mod tests;

/// GPU Debug Protocol decoder
///
/// Decodes data from a u32 buffer following a simplified gob-like protocol.
pub struct GdpDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GdpDecoder<'a> {
    /// Create a new decoder from a u32 buffer
    ///
    /// The u32 buffer is immediately transmuted to a byte slice for processing.
    pub fn new(buffer: &'a [u32]) -> Self {
        // Transmute u32 slice to bytes (big-endian layout assumed)
        let data = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 4) };

        GdpDecoder { data, pos: 0 }
    }

    /// Check if we've consumed all data
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Get current position in bytes
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Skip to next word (4-byte) boundary
    ///
    /// This ensures we don't have partial values crossing word boundaries,
    /// which is important for GPU memory access patterns.
    fn align_to_word(&mut self) {
        let remainder = self.pos % 4;
        if remainder != 0 {
            self.pos += 4 - remainder;
        }
    }

    /// Read a single byte
    fn read_byte(&mut self) -> Result<u8, &'static str> {
        if self.pos >= self.data.len() {
            return Err("unexpected end of data");
        }
        let byte = self.data[self.pos];
        self.pos += 1;
        Ok(byte)
    }

    /// Internal: decode unsigned integer without alignment
    fn decode_uint_internal(&mut self) -> Result<u64, &'static str> {
        let first = self.read_byte()?;

        if first < 128 {
            // Small value encoded directly
            return Ok(first as u64);
        }

        // Negated byte count: ~first + 1 gives us the actual count
        let byte_count = (!first).wrapping_add(1) as usize;

        if byte_count > 8 {
            return Err("unsigned integer too large");
        }

        // Read big-endian bytes
        let mut value: u64 = 0;
        for _ in 0..byte_count {
            let byte = self.read_byte()?;
            value = (value << 8) | (byte as u64);
        }

        Ok(value)
    }

    /// Decode unsigned integer following gob format
    ///
    /// - If value < 128: encoded as single byte with that value
    /// - Otherwise: encoded as negated byte count followed by big-endian bytes
    ///
    /// Examples:
    /// - 0 → 0x00
    /// - 7 → 0x07
    /// - 256 → 0xFE 0x01 0x00 (FE = -2, followed by 2 bytes)
    pub fn decode_uint(&mut self) -> Result<u64, &'static str> {
        let value = self.decode_uint_internal()?;
        self.align_to_word();
        Ok(value)
    }

    /// Decode signed integer following gob format
    ///
    /// Encoded within unsigned integer where:
    /// - Bits 1+ contain the value
    /// - Bit 0 indicates whether to complement on receipt
    ///
    /// Encoding algorithm:
    /// ```ignore
    /// if i < 0:
    ///     u = ((~i) << 1) | 1  // complement, bit 0 is 1
    /// else:
    ///     u = i << 1           // no complement, bit 0 is 0
    /// ```
    ///
    /// Examples:
    /// - -129 → encodes as (~128 << 1) | 1 = 257 → 0xFE 0x01 0x01
    pub fn decode_int(&mut self) -> Result<i64, &'static str> {
        let u = self.decode_uint()?;

        // Decode: bit 0 is complement flag, bits 1+ are value
        let value = (u >> 1) as i64;

        if (u & 1) != 0 {
            // Complement bit is set
            Ok(!value)
        } else {
            Ok(value)
        }
    }

    /// Decode boolean
    ///
    /// Encoded as unsigned integer: 0 for false, 1 for true.
    pub fn decode_bool(&mut self) -> Result<bool, &'static str> {
        let value = self.decode_uint()?;
        Ok(value != 0)
    }

    /// Decode string
    ///
    /// Encoded as unsigned count of bytes followed by that many UTF-8 bytes.
    /// After the string data, automatically aligns to next word boundary.
    pub fn decode_string(&mut self) -> Result<String, &'static str> {
        // Use internal version - don't align after length, only after whole string
        let len = self.decode_uint_internal()? as usize;

        if self.pos + len > self.data.len() {
            return Err("string length exceeds available data");
        }

        let bytes = &self.data[self.pos..self.pos + len];
        self.pos += len;

        self.align_to_word();

        String::from_utf8(bytes.to_vec()).map_err(|_| "invalid UTF-8 in string")
    }

    /// Peek at next byte without consuming
    pub fn peek_byte(&self) -> Result<u8, &'static str> {
        if self.pos >= self.data.len() {
            return Err("unexpected end of data");
        }
        Ok(self.data[self.pos])
    }

    /// Get remaining bytes available
    pub fn remaining(&self) -> usize {
        if self.pos >= self.data.len() { 0 } else { self.data.len() - self.pos }
    }

    /// Decode a type-tagged value
    ///
    /// Returns a GdpValue containing the typed value.
    /// Type byte format: VVVVVVTT where TT=type (bits 0-1), VVVVVV=value/length (bits 2-7)
    pub fn decode_value(&mut self) -> Result<GdpValue, &'static str> {
        let type_byte = self.read_byte()?;
        let type_tag = type_byte & 0x03;
        let inline_value = (type_byte >> 2) as u64;

        match type_tag {
            0x00 => {
                // Unsigned integer
                let value = if inline_value == 0 { self.decode_uint_internal()? } else { inline_value };
                self.align_to_word();
                Ok(GdpValue::UInt(value))
            }
            0x01 => {
                // Signed integer
                let result = if inline_value == 0 {
                    // Full value follows, decode as gob int
                    let u = self.decode_uint_internal()?;
                    let value = (u >> 1) as i64;
                    if (u & 1) != 0 { !value } else { value }
                } else {
                    // Inline value is stored directly (not gob-encoded for small values)
                    // Treat as signed 6-bit value
                    if inline_value < 32 {
                        inline_value as i64
                    } else {
                        // Negative: treat as 6-bit two's complement
                        (inline_value as i64) - 64
                    }
                };
                self.align_to_word();
                Ok(GdpValue::Int(result))
            }
            0x02 => {
                // String
                let len = if inline_value == 0 {
                    self.decode_uint_internal()? as usize
                } else {
                    inline_value as usize
                };

                if self.pos + len > self.data.len() {
                    return Err("string length exceeds available data");
                }

                let bytes = &self.data[self.pos..self.pos + len];
                self.pos += len;
                self.align_to_word();

                let string = String::from_utf8(bytes.to_vec()).map_err(|_| "invalid UTF-8 in string")?;
                Ok(GdpValue::String(string))
            }
            0x03 => {
                // Type tag 0b11: Float32
                // inline_value should always be 0 for float32
                if inline_value != 0 {
                    return Err("float32 type byte must have inline value 0");
                }
                // Decode gob uint, byte-reverse, bitcast to f32
                let reversed = self.decode_uint_internal()? as u32;
                // Byte-reverse: swap_bytes
                let bits = reversed.swap_bytes();
                let value = f32::from_bits(bits);
                self.align_to_word();
                Ok(GdpValue::Float32(value))
            }
            _ => unreachable!(),
        }
    }
}

/// A decoded GDP value with type information
#[derive(Debug, Clone, PartialEq)]
pub enum GdpValue {
    UInt(u64),
    Int(i64),
    String(String),
    Bool(bool), // Booleans are decoded as UInt, but kept for backwards compatibility
    Float32(f32),
}
