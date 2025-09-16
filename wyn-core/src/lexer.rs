use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, multispace1, one_of},
    combinator::{eof, map, opt, peek, recognize, value},
    multi::many0,
    sequence::{pair, preceded, terminated, tuple},
    IResult,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Let,
    Def,
    Val,
    In,

    // Identifiers and literals
    Identifier(String),
    IntLiteral(i32),
    FloatLiteral(f32),

    // Types
    I32Type,
    F32Type,

    // Operators
    Assign,
    Divide,
    Add,
    Arrow,
    Backslash, // \ for lambda expressions
    Dot,       // . for field access

    // Delimiters
    LeftParen,
    RightParen,
    LeftBracket,
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
    ))(input)
}

fn parse_type(input: &str) -> IResult<&str, Token> {
    alt((
        // Base types
        value(Token::I32Type, tag("i32")),
        value(Token::F32Type, tag("f32")),
    ))(input)
}

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
        value(Token::Assign, tag("=")),
        value(Token::Divide, tag("/")),
        value(Token::Add, char('+')),
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
            parse_type,
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

    while !remaining.is_empty() {
        // Skip leading whitespace
        if let Ok((rest, _)) = multispace1::<&str, nom::error::Error<&str>>(remaining) {
            remaining = rest;
            continue;
        }

        match parse_token(remaining) {
            Ok((rest, token)) => {
                // Skip comments
                if !matches!(token, Token::Comment(_)) {
                    tokens.push(token);
                }
                remaining = rest;
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
        assert_eq!(tokens, vec![Token::I32Type, Token::F32Type]);
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
                Token::LeftBracket,
                Token::IntLiteral(3),
                Token::RightBracket,
                Token::LeftBracket,
                Token::IntLiteral(4),
                Token::RightBracket,
                Token::F32Type,
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
                Token::Divide,
                Token::FloatLiteral(255.0),
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
