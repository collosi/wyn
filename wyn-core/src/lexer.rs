use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, multispace1, one_of},
    combinator::{eof, map, opt, peek, recognize, value},
    multi::many0,
    sequence::{pair, preceded, terminated, tuple},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Let,
    Def,
    Val,
    In,
    If,
    Then,
    Else,

    // Identifiers and literals
    Identifier(String),
    IntLiteral(i32),
    FloatLiteral(f32),

    // Operators
    Assign,
    BinOp(String), // Binary operators: +, -, *, /
    Arrow,
    Backslash, // \ for lambda expressions
    Dot,       // . for field access
    Star,      // * for uniqueness types (prefix)

    // Delimiters
    LeftParen,
    RightParen,
    LeftBracket,       // [ without preceding whitespace (for indexing: arr[0])
    LeftBracketSpaced, // [ with preceding whitespace (for array literals: f [1,2,3])
    RightBracket,
    Colon,
    Comma,

    // Attributes
    AttributeStart, // #[

    // Comments (to be skipped)
    Comment(String),
}

fn parse_comment(input: &str) -> IResult<&str, Token> {
    map(preceded(tag("--"), take_until("\n")), |s: &str| {
        Token::Comment(s.to_string())
    })(input)
}

fn parse_keyword(input: &str) -> IResult<&str, Token> {
    // Helper function to match a keyword with word boundaries
    let keyword = |kw: &'static str, token: Token| {
        map(
            terminated(
                tag(kw),
                peek(alt((eof, recognize(one_of(" \t\n\r()[]{}=:,+-/*#<>"))))),
            ),
            move |_| token.clone(),
        )
    };

    alt((
        keyword("let", Token::Let),
        keyword("def", Token::Def),
        keyword("val", Token::Val),
        keyword("in", Token::In),
        keyword("if", Token::If),
        keyword("then", Token::Then),
        keyword("else", Token::Else),
    ))(input)
}

// Removed parse_type - i32 and f32 are now treated as regular identifiers
// They are only special in type position (handled by parser) and as suffixes on literals

fn parse_type_variable(input: &str) -> IResult<&str, Token> {
    map(
        recognize(tuple((
            tag("'"),
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))),
        |s: &str| Token::Identifier(s.to_string()),
    )(input)
}

fn parse_identifier(input: &str) -> IResult<&str, Token> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        )),
        |s: &str| Token::Identifier(s.to_string()),
    )(input)
}

fn parse_float_literal(input: &str) -> IResult<&str, Token> {
    map(
        recognize(tuple((
            opt(char('-')),
            digit1,
            opt(tuple((char('.'), digit1))),
            tag("f32"),
        ))),
        |s: &str| {
            let num_str = &s[..s.len() - 3]; // Remove "f32" suffix
            Token::FloatLiteral(num_str.parse().unwrap_or(0.0))
        },
    )(input)
}

fn parse_int_literal(input: &str) -> IResult<&str, Token> {
    map(recognize(pair(opt(char('-')), digit1)), |s: &str| {
        Token::IntLiteral(s.parse().unwrap_or(0))
    })(input)
}

fn parse_operator(input: &str) -> IResult<&str, Token> {
    alt((
        value(Token::Arrow, tag("->")),
        // Comparison operators (must come before single =, <, >)
        map(tag("=="), |s: &str| Token::BinOp(s.to_string())),
        map(tag("!="), |s: &str| Token::BinOp(s.to_string())),
        map(tag("<="), |s: &str| Token::BinOp(s.to_string())),
        map(tag(">="), |s: &str| Token::BinOp(s.to_string())),
        map(tag("<"), |s: &str| Token::BinOp(s.to_string())),
        map(tag(">"), |s: &str| Token::BinOp(s.to_string())),
        // Assignment (must come after ==)
        value(Token::Assign, tag("=")),
        // Arithmetic operators
        map(tag("/"), |s: &str| Token::BinOp(s.to_string())),
        map(char('+'), |c| Token::BinOp(c.to_string())),
        map(char('-'), |c| Token::BinOp(c.to_string())),
        map(char('*'), |c| Token::BinOp(c.to_string())),
        value(Token::Backslash, char('\\')),
        value(Token::Dot, char('.')),
    ))(input)
}

fn parse_delimiter(input: &str) -> IResult<&str, Token> {
    alt((
        value(Token::AttributeStart, tag("#[")),
        value(Token::LeftParen, char('(')),
        value(Token::RightParen, char(')')),
        value(Token::LeftBracket, char('[')),
        value(Token::RightBracket, char(']')),
        value(Token::Colon, char(':')),
        value(Token::Comma, char(',')),
    ))(input)
}

fn parse_token(input: &str) -> IResult<&str, Token> {
    preceded(
        multispace0,
        alt((
            parse_comment,
            parse_keyword,
            parse_float_literal,
            parse_type_variable,
            parse_identifier,
            parse_int_literal,
            parse_operator,
            parse_delimiter,
        )),
    )(input)
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut remaining = input;
    let mut tokens = Vec::new();
    let mut had_whitespace = true; // Start of input counts as having whitespace

    while !remaining.is_empty() {
        // Check for and skip leading whitespace
        if let Ok((rest, _)) = multispace1::<&str, nom::error::Error<&str>>(remaining) {
            remaining = rest;
            had_whitespace = true;
            continue;
        }

        match parse_token(remaining) {
            Ok((rest, mut token)) => {
                // Skip comments
                if matches!(token, Token::Comment(_)) {
                    remaining = rest;
                    had_whitespace = true; // Comments act like whitespace
                    continue;
                }

                // Convert LeftBracket based on whitespace
                if matches!(token, Token::LeftBracket) {
                    token = if had_whitespace { Token::LeftBracketSpaced } else { Token::LeftBracket };
                }

                tokens.push(token);
                remaining = rest;
                had_whitespace = false; // Next token won't have whitespace unless we skip some
            }
            Err(_) if remaining.trim().is_empty() => break,
            Err(e) => return Err(format!("Tokenization error: {:?}", e)),
        }
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_keywords() {
        let input = "let def";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens, vec![Token::Let, Token::Def]);
    }

    #[test]
    fn test_tokenize_types() {
        let input = "i32 f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("i32".to_string()),
                Token::Identifier("f32".to_string())
            ]
        );
    }

    #[test]
    fn test_tokenize_identifiers() {
        let input = "vertex_main SKY_RGBA verts";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("vertex_main".to_string()),
                Token::Identifier("SKY_RGBA".to_string()),
                Token::Identifier("verts".to_string()),
            ]
        );
    }

    #[test]
    fn test_tokenize_literals() {
        let input = "-1.0f32 42 3.14f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::FloatLiteral(-1.0),
                Token::IntLiteral(42),
                Token::FloatLiteral(3.14),
            ]
        );
    }

    #[test]
    fn test_tokenize_with_comments() {
        let input = "-- This is a comment\nlet x = 42";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Identifier("x".to_string()),
                Token::Assign,
                Token::IntLiteral(42),
            ]
        );
    }

    #[test]
    fn test_tokenize_array_syntax() {
        let input = "[3][4]f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LeftBracketSpaced, // At start of input, counts as having whitespace
                Token::IntLiteral(3),
                Token::RightBracket,
                Token::LeftBracket, // No space before this one
                Token::IntLiteral(4),
                Token::RightBracket,
                Token::Identifier("f32".to_string()),
            ]
        );
    }

    #[test]
    fn test_tokenize_division() {
        let input = "135f32/255f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::FloatLiteral(135.0),
                Token::BinOp("/".to_string()),
                Token::FloatLiteral(255.0),
            ]
        );
    }

    #[test]
    fn test_tokenize_binary_operators() {
        let input = "a + b - c * d / e";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::BinOp("+".to_string()),
                Token::Identifier("b".to_string()),
                Token::BinOp("-".to_string()),
                Token::Identifier("c".to_string()),
                Token::BinOp("*".to_string()),
                Token::Identifier("d".to_string()),
                Token::BinOp("/".to_string()),
                Token::Identifier("e".to_string()),
            ]
        );
    }

    #[test]
    fn test_tokenize_attributes() {
        let input = "#[vertex]";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::AttributeStart,
                Token::Identifier("vertex".to_string()),
                Token::RightBracket,
            ]
        );
    }
}
