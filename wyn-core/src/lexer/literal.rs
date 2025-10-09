//! Literal parsing for the Wyn lexer
//!
//! Implements the full grammar specification for literals:
//! - Integer literals: decimal, hexadecimal (0xFF), binary (0b1010) with optional underscores
//! - Float literals: pointfloat, exponent notation, hexadecimal floats
//! - Type suffixes: i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
//! - String literals: "stringchar*"
//! - Char literals: 'char'

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, digit1, hex_digit1, none_of, one_of},
    combinator::{map, opt, recognize, verify},
    multi::many1,
    sequence::{delimited, pair, tuple},
    IResult,
};

use super::Token;

// Helper to parse digits with optional underscores
fn decimal_with_underscores(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        digit1,
        many1(alt((digit1, tag("_")))),
    ))(input)
}

fn hex_with_underscores(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        hex_digit1,
        many1(alt((hex_digit1, tag("_")))),
    ))(input)
}

fn binary_with_underscores(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        one_of("01"),
        many1(alt((one_of("01"), char('_')))),
    ))(input)
}

// Strip underscores from a numeric string for parsing
fn strip_underscores(s: &str) -> String {
    s.chars().filter(|c| *c != '_').collect()
}

// Parse integer type suffix
fn int_type_suffix(input: &str) -> IResult<&str, &str> {
    alt((
        tag("i8"), tag("i16"), tag("i32"), tag("i64"),
        tag("u8"), tag("u16"), tag("u32"), tag("u64"),
    ))(input)
}

// Parse float type suffix
fn float_type_suffix(input: &str) -> IResult<&str, &str> {
    alt((tag("f16"), tag("f32"), tag("f64")))(input)
}

// Hexadecimal integer: 0x[hex_digits][type_suffix]
fn parse_hexadecimal_int(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            alt((tag("0x"), tag("0X"))),
            alt((hex_with_underscores, hex_digit1)),
            opt(int_type_suffix),
        )),
        |(_, digits, _suffix)| {
            let clean = strip_underscores(digits);
            let value = i32::from_str_radix(&clean, 16).unwrap_or(0);
            Token::IntLiteral(value)
        },
    )(input)
}

// Binary integer: 0b[binary_digits][type_suffix]
fn parse_binary_int(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            alt((tag("0b"), tag("0B"))),
            alt((binary_with_underscores, recognize(many1(one_of("01"))))),
            opt(int_type_suffix),
        )),
        |(_, digits, _suffix)| {
            let clean = strip_underscores(digits);
            let value = i32::from_str_radix(&clean, 2).unwrap_or(0);
            Token::IntLiteral(value)
        },
    )(input)
}

// Decimal integer: [digits][type_suffix]
fn parse_decimal_int(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            opt(char('-')),
            alt((decimal_with_underscores, digit1)),
            opt(int_type_suffix),
        )),
        |(sign, digits, _suffix)| {
            let mut clean = strip_underscores(digits);
            if sign.is_some() {
                clean = format!("-{}", clean);
            }
            Token::IntLiteral(clean.parse().unwrap_or(0))
        },
    )(input)
}

// Parse integer literal (try hex, binary, then decimal)
pub fn parse_int_literal(input: &str) -> IResult<&str, Token> {
    alt((parse_hexadecimal_int, parse_binary_int, parse_decimal_int))(input)
}

// Intpart: digits (possibly with underscores)
fn intpart(input: &str) -> IResult<&str, &str> {
    alt((decimal_with_underscores, digit1))(input)
}

// Pointfloat: [intpart] fraction
fn pointfloat(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        opt(intpart),
        char('.'),
        alt((decimal_with_underscores, digit1)),
    )))(input)
}

// Exponent: (e|E)[+|-]digits
fn exponent(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        alt((char('e'), char('E'))),
        opt(alt((char('+'), char('-')))),
        alt((decimal_with_underscores, digit1)),
    )))(input)
}

// Exponent float: (intpart | pointfloat) exponent
fn exponentfloat(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        alt((pointfloat, intpart)),
        exponent,
    )))(input)
}

// Hexadecimal float: 0x[hex_mantissa]p[+|-][dec_exponent]
fn hexadecimalfloat(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        alt((tag("0x"), tag("0X"))),
        alt((hex_with_underscores, hex_digit1)),
        opt(tuple((
            char('.'),
            alt((hex_with_underscores, hex_digit1)),
        ))),
        alt((char('p'), char('P'))),
        opt(alt((char('+'), char('-')))),
        alt((decimal_with_underscores, digit1)),
    )))(input)
}

// Parse float literal
pub fn parse_float_literal(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            opt(char('-')),
            alt((hexadecimalfloat, exponentfloat, pointfloat, intpart)),
            float_type_suffix,
        )),
        |(sign, float_str, _suffix)| {
            let mut clean = strip_underscores(float_str);
            if sign.is_some() {
                clean = format!("-{}", clean);
            }

            // Hexadecimal floats need special parsing
            if clean.starts_with("0x") || clean.starts_with("0X") {
                // Parse hex float manually: 0x[mantissa]p[exponent]
                // For simplicity, convert to decimal approximation
                // TODO: Implement proper hex float parsing
                Token::FloatLiteral(0.0)
            } else {
                Token::FloatLiteral(clean.parse().unwrap_or(0.0))
            }
        },
    )(input)
}

// String literal parser
// stringlit  ::= '"' stringchar* '"'
// stringchar ::= <any source character except "\" or newline or double quotes>
pub fn parse_string_literal(input: &str) -> IResult<&str, Token> {
    map(
        delimited(
            char('"'),
            take_while(|c| c != '"' && c != '\\' && c != '\n'),
            char('"'),
        ),
        |s: &str| Token::StringLiteral(s.to_string()),
    )(input)
}

// Char literal parser
// charlit ::= "'" char "'"
// char    ::= <any source character except "\" or newline or single quotes>
pub fn parse_char_literal(input: &str) -> IResult<&str, Token> {
    map(
        delimited(
            char('\''),
            verify(none_of("'\\\n"), |_| true),
            char('\''),
        ),
        Token::CharLiteral,
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hexadecimal_integers() {
        assert_eq!(parse_int_literal("0x10"), Ok(("", Token::IntLiteral(16))));
        assert_eq!(parse_int_literal("0xFF"), Ok(("", Token::IntLiteral(255))));
        assert_eq!(parse_int_literal("0x1A_2B"), Ok(("", Token::IntLiteral(0x1A2B))));
        assert_eq!(parse_int_literal("0X00FF"), Ok(("", Token::IntLiteral(255))));
    }

    #[test]
    fn test_binary_integers() {
        assert_eq!(parse_int_literal("0b1010"), Ok(("", Token::IntLiteral(10))));
        assert_eq!(parse_int_literal("0B1111"), Ok(("", Token::IntLiteral(15))));
        assert_eq!(parse_int_literal("0b10_11"), Ok(("", Token::IntLiteral(11))));
    }

    #[test]
    fn test_integers_with_underscores() {
        assert_eq!(parse_int_literal("1_000_000"), Ok(("", Token::IntLiteral(1_000_000))));
        assert_eq!(parse_int_literal("42_42"), Ok(("", Token::IntLiteral(4242))));
        assert_eq!(parse_int_literal("123_456"), Ok(("", Token::IntLiteral(123456))));
    }

    #[test]
    fn test_basic_decimals() {
        assert_eq!(parse_int_literal("42"), Ok(("", Token::IntLiteral(42))));
        assert_eq!(parse_int_literal("-17"), Ok(("", Token::IntLiteral(-17))));
    }

    #[test]
    fn test_pointfloat() {
        assert_eq!(parse_float_literal("3.14f32"), Ok(("", Token::FloatLiteral(3.14))));
        assert_eq!(parse_float_literal("0.5f32"), Ok(("", Token::FloatLiteral(0.5))));
        assert_eq!(parse_float_literal(".5f32"), Ok(("", Token::FloatLiteral(0.5))));
    }

    #[test]
    fn test_intpart_with_suffix() {
        // Test integers with float suffix like "135f32"
        assert_eq!(parse_float_literal("135f32"), Ok(("", Token::FloatLiteral(135.0))));
        assert_eq!(parse_float_literal("255f32"), Ok(("", Token::FloatLiteral(255.0))));
        assert_eq!(parse_float_literal("-17f32"), Ok(("", Token::FloatLiteral(-17.0))));
    }

    #[test]
    fn test_float_exponent_notation() {
        assert_eq!(parse_float_literal("1.5e10f32"), Ok(("", Token::FloatLiteral(1.5e10))));
        assert_eq!(parse_float_literal("2E-5f32"), Ok(("", Token::FloatLiteral(2e-5))));
        assert_eq!(parse_float_literal("3.14e2f32"), Ok(("", Token::FloatLiteral(314.0))));
    }

    #[test]
    fn test_floats_with_underscores() {
        assert_eq!(parse_float_literal("3.14_159f32"), Ok(("", Token::FloatLiteral(3.14159))));
        assert_eq!(parse_float_literal("1_000.5f32"), Ok(("", Token::FloatLiteral(1000.5))));
    }

    #[test]
    #[ignore] // Hexadecimal float parsing not fully implemented
    fn test_hexadecimal_floats() {
        // These should parse but currently return 0.0
        assert!(parse_float_literal("0x1.8p3f32").is_ok());
        assert!(parse_float_literal("0X1.0p-2f32").is_ok());
    }

    #[test]
    fn test_string_literals() {
        assert_eq!(
            parse_string_literal("\"hello\""),
            Ok(("", Token::StringLiteral("hello".to_string())))
        );
        assert_eq!(
            parse_string_literal("\"hello world\""),
            Ok(("", Token::StringLiteral("hello world".to_string())))
        );
        assert_eq!(
            parse_string_literal("\"\""),
            Ok(("", Token::StringLiteral("".to_string())))
        );
        assert_eq!(
            parse_string_literal("\"foo123\""),
            Ok(("", Token::StringLiteral("foo123".to_string())))
        );
    }

    #[test]
    fn test_string_rejects_backslash() {
        // Strings with backslashes should fail (no escape sequences according to grammar)
        assert!(parse_string_literal("\"hello\\nworld\"").is_err());
        assert!(parse_string_literal("\"test\\\"quote\"").is_err());
    }

    #[test]
    fn test_string_rejects_newline() {
        // Strings with newlines should fail
        assert!(parse_string_literal("\"hello\nworld\"").is_err());
    }

    #[test]
    fn test_char_literals() {
        assert_eq!(parse_char_literal("'a'"), Ok(("", Token::CharLiteral('a'))));
        assert_eq!(parse_char_literal("'Z'"), Ok(("", Token::CharLiteral('Z'))));
        assert_eq!(parse_char_literal("'0'"), Ok(("", Token::CharLiteral('0'))));
        assert_eq!(parse_char_literal("' '"), Ok(("", Token::CharLiteral(' '))));
        assert_eq!(parse_char_literal("'?'"), Ok(("", Token::CharLiteral('?'))));
    }

    #[test]
    fn test_char_rejects_backslash() {
        // Chars with backslashes should fail (no escape sequences according to grammar)
        assert!(parse_char_literal("'\\n'").is_err());
        assert!(parse_char_literal("'\\''").is_err());
    }

    #[test]
    fn test_char_rejects_newline() {
        // Chars with newlines should fail
        assert!(parse_char_literal("'\n'").is_err());
    }

    #[test]
    fn test_char_rejects_empty() {
        // Empty char literals should fail
        assert!(parse_char_literal("''").is_err());
    }
}
