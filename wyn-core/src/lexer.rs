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
    Loop,
    For,
    While,
    Do,
    Match,
    Case,
    Module,
    Open,
    Import,
    Local,
    Type,
    Include,
    With,
    Unsafe,
    Assert,

    // Identifiers and literals
    Identifier(String),
    IntLiteral(i32),
    FloatLiteral(f32),
    CharLiteral(char),
    StringLiteral(String),

    // Boolean literals
    True,
    False,

    // Operators
    Assign,
    BinOp(String), // Binary operators: +, -, *, /
    Arrow,
    Backslash,  // \ for lambda expressions
    Dot,        // . for field access
    DotDot,     // .. for ranges
    DotDotLt,   // ..< for ranges
    DotDotGt,   // ..> for ranges
    Ellipsis,   // ... for ranges
    Pipe,       // | for pattern matching
    PipeOp,     // |> for pipe operator
    Star,       // * for uniqueness types (prefix)
    Minus,      // - (can be unary or binary)
    Underscore, // _ for wildcard patterns
    Bang,       // ! for negation
    QuestionMark, // ? for existential types
    At,         // @ for as-patterns (future)

    // Delimiters
    LeftParen,
    RightParen,
    LeftBracket,       // [ without preceding whitespace (for indexing: arr[0])
    LeftBracketSpaced, // [ with preceding whitespace (for array literals: f [1,2,3])
    RightBracket,
    LeftBrace,
    RightBrace,
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
        alt((
            keyword("module", Token::Module),
            keyword("import", Token::Import),
            keyword("include", Token::Include),
            keyword("unsafe", Token::Unsafe),
            keyword("assert", Token::Assert),
            keyword("match", Token::Match),
            keyword("case", Token::Case),
            keyword("local", Token::Local),
        )),
        alt((
            keyword("let", Token::Let),
            keyword("loop", Token::Loop),
            keyword("while", Token::While),
            keyword("def", Token::Def),
            keyword("val", Token::Val),
            keyword("type", Token::Type),
            keyword("with", Token::With),
            keyword("open", Token::Open),
        )),
        alt((
            keyword("then", Token::Then),
            keyword("else", Token::Else),
            keyword("true", Token::True),
            keyword("false", Token::False),
            keyword("for", Token::For),
            keyword("in", Token::In),
            keyword("if", Token::If),
            keyword("do", Token::Do),
        )),
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
            many0(alt((alphanumeric1, tag("_"), tag("'")))),
        )),
        |s: &str| {
            // Handle underscore specially - it's a wildcard pattern, not an identifier
            if s == "_" { Token::Underscore } else { Token::Identifier(s.to_string()) }
        },
    )(input)
}

fn parse_string_literal(input: &str) -> IResult<&str, Token> {
    use nom::bytes::complete::take_while;

    let (rest, _) = char('"')(input)?;
    let (rest, content) = take_while(|c| c != '"')(rest)?;
    let (rest, _) = char('"')(rest)?;

    Ok((rest, Token::StringLiteral(content.to_string())))
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
        alt((
            value(Token::Arrow, tag("->")),
            // Range operators (must come before ..)
            value(Token::Ellipsis, tag("...")),
            value(Token::DotDotLt, tag("..<")),
            value(Token::DotDotGt, tag("..>")),
            value(Token::DotDot, tag("..")),
            // Pipe operator (must come before |)
            value(Token::PipeOp, tag("|>")),
        )),
        alt((
            // Comparison operators (must come before single =, <, >)
            map(tag("=="), |s: &str| Token::BinOp(s.to_string())),
            map(tag("!="), |s: &str| Token::BinOp(s.to_string())),
            map(tag("<="), |s: &str| Token::BinOp(s.to_string())),
            map(tag(">="), |s: &str| Token::BinOp(s.to_string())),
            map(tag("<"), |s: &str| Token::BinOp(s.to_string())),
            map(tag(">"), |s: &str| Token::BinOp(s.to_string())),
        )),
        alt((
            // Assignment (must come after ==)
            value(Token::Assign, tag("=")),
            // Arithmetic operators
            map(tag("/"), |s: &str| Token::BinOp(s.to_string())),
            map(char('+'), |c| Token::BinOp(c.to_string())),
            map(char('-'), |c| Token::BinOp(c.to_string())),
            map(char('*'), |c| Token::BinOp(c.to_string())),
            value(Token::Backslash, char('\\')),
        )),
        alt((
            value(Token::Dot, char('.')),
            value(Token::Pipe, char('|')),
            value(Token::Bang, char('!')),
            value(Token::QuestionMark, char('?')),
            value(Token::At, char('@')),
        )),
    ))(input)
}

fn parse_delimiter(input: &str) -> IResult<&str, Token> {
    alt((
        value(Token::AttributeStart, tag("#[")),
        value(Token::LeftParen, char('(')),
        value(Token::RightParen, char(')')),
        value(Token::LeftBracket, char('[')),
        value(Token::RightBracket, char(']')),
        value(Token::LeftBrace, char('{')),
        value(Token::RightBrace, char('}')),
        value(Token::Colon, char(':')),
        value(Token::Comma, char(',')),
    ))(input)
}

fn parse_token(input: &str) -> IResult<&str, Token> {
    preceded(
        multispace0,
        alt((
            parse_comment,
            parse_string_literal,
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
    fn test_all_literal_types() {
        // literal ::= intnumber | floatnumber | "true" | "false"
        let input = "true false 123 45.67f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::True,
                Token::False,
                Token::IntLiteral(123),
                Token::FloatLiteral(45.67),
            ]
        );
    }

    #[test]
    fn test_integer_literal_formats() {
        // Test decimal integers (basic support)
        let input = "42 -10 0";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::IntLiteral(42),
                Token::IntLiteral(-10),
                Token::IntLiteral(0),
            ]
        );
    }

    #[test]
    #[ignore] // TODO: Implement hexadecimal integer support
    fn test_hexadecimal_integers() {
        // hexadecimal ::= 0 ("x" | "X") hexdigit (hexdigit |"_")*
        let input = "0x10 0xFF 0x1A_2B";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::IntLiteral(16),
                Token::IntLiteral(255),
                Token::IntLiteral(0x1A2B),
            ]
        );
    }

    #[test]
    #[ignore] // TODO: Implement binary integer support
    fn test_binary_integers() {
        // binary ::= 0 ("b" | "B") bindigit (bindigit | "_")*
        let input = "0b1010 0B1111 0b10_11";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::IntLiteral(0b1010),
                Token::IntLiteral(0b1111),
                Token::IntLiteral(0b1011),
            ]
        );
    }

    #[test]
    #[ignore] // TODO: Implement underscore support in decimals
    fn test_integers_with_underscores() {
        // decimal ::= decdigit (decdigit |"_")*
        let input = "1_000_000 42_42";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::IntLiteral(1_000_000),
                Token::IntLiteral(4242),
            ]
        );
    }

    #[test]
    fn test_float_literal_formats() {
        // Test basic pointfloat format (currently supported)
        let input = "1.0f32 -2.5f32 0.0f32 3.14159f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::FloatLiteral(1.0),
                Token::FloatLiteral(-2.5),
                Token::FloatLiteral(0.0),
                Token::FloatLiteral(3.14159),
            ]
        );
    }

    #[test]
    #[ignore] // TODO: Implement exponent notation for floats
    fn test_float_exponent_notation() {
        // exponentfloat ::= (intpart | pointfloat) exponent
        // exponent ::= ("e" | "E") ["+" | "-"] decdigit+
        let input = "1.5e10f32 2E-5f32 3.14e+2f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::FloatLiteral(1.5e10),
                Token::FloatLiteral(2E-5),
                Token::FloatLiteral(3.14e2),
            ]
        );
    }

    #[test]
    #[ignore] // TODO: Implement underscore support in floats
    fn test_floats_with_underscores() {
        // fraction ::= "." decdigit (decdigit |"_")*
        let input = "3.14_159f32 1_000.5f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::FloatLiteral(3.14159),
                Token::FloatLiteral(1000.5),
            ]
        );
    }

    #[test]
    #[ignore] // TODO: Implement hexadecimal float support
    fn test_hexadecimal_floats() {
        // hexadecimalfloat ::= 0 ("x" | "X") hexintpart hexfraction ("p"|"P") ["+" | "-"] decdigit+
        let input = "0x1.8p3f32 0X1.0p-2f32";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::FloatLiteral(12.0),  // 1.5 * 2^3 = 12
                Token::FloatLiteral(0.25),  // 1.0 * 2^-2 = 0.25
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

    // Tests for grammar rules: name with constituent characters
    #[test]
    fn test_name_with_prime() {
        // constituent ::= letter | digit | "_" | "'"
        let input = "acc' uv' x'' my_var'";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("acc'".to_string()),
                Token::Identifier("uv'".to_string()),
                Token::Identifier("x''".to_string()),
                Token::Identifier("my_var'".to_string()),
            ]
        );
    }

    #[test]
    fn test_name_starting_with_underscore() {
        // name ::= lowercase constituent* | "_" constituent*
        let input = "_foo _bar123 _x'";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("_foo".to_string()),
                Token::Identifier("_bar123".to_string()),
                Token::Identifier("_x'".to_string()),
            ]
        );
    }

    #[test]
    fn test_constructor_names() {
        // constructor ::= uppercase constituent*
        let input = "Some None True' MyType123";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("Some".to_string()),
                Token::Identifier("None".to_string()),
                Token::Identifier("True'".to_string()),
                Token::Identifier("MyType123".to_string()),
            ]
        );
    }

    #[test]
    fn test_new_operators() {
        // Test |>, .., ..., ..<, ..>, |, !, ?, @
        let input = "|> .. ... ..< ..>";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::PipeOp,
                Token::DotDot,
                Token::Ellipsis,
                Token::DotDotLt,
                Token::DotDotGt,
            ]
        );
    }

    #[test]
    fn test_bang_and_pipe() {
        let input = "! | !x |> y";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Bang,
                Token::Pipe,
                Token::Bang,
                Token::Identifier("x".to_string()),
                Token::PipeOp,
                Token::Identifier("y".to_string()),
            ]
        );
    }

    #[test]
    fn test_question_and_at() {
        let input = "? @ x ?[n]";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::QuestionMark,
                Token::At,
                Token::Identifier("x".to_string()),
                Token::QuestionMark,
                Token::LeftBracket,
                Token::Identifier("n".to_string()),
                Token::RightBracket,
            ]
        );
    }

    #[test]
    fn test_new_keywords() {
        let input = "loop for while do match case unsafe assert";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Loop,
                Token::For,
                Token::While,
                Token::Do,
                Token::Match,
                Token::Case,
                Token::Unsafe,
                Token::Assert,
            ]
        );
    }

    #[test]
    fn test_qualified_names_tokenization() {
        // quals ::= (name ".")+
        // qualname ::= name | quals name
        // The parser will handle this, but lexer should produce: Identifier, Dot, Identifier
        let input = "f32.sin i32.max std.math.sqrt";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("f32".to_string()),
                Token::Dot,
                Token::Identifier("sin".to_string()),
                Token::Identifier("i32".to_string()),
                Token::Dot,
                Token::Identifier("max".to_string()),
                Token::Identifier("std".to_string()),
                Token::Dot,
                Token::Identifier("math".to_string()),
                Token::Dot,
                Token::Identifier("sqrt".to_string()),
            ]
        );
    }

    #[test]
    fn test_range_operators_in_context() {
        // Test that range operators work in realistic context
        let input = "a..b a...b a..<b a..>b";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::DotDot,
                Token::Identifier("b".to_string()),
                Token::Identifier("a".to_string()),
                Token::Ellipsis,
                Token::Identifier("b".to_string()),
                Token::Identifier("a".to_string()),
                Token::DotDotLt,
                Token::Identifier("b".to_string()),
                Token::Identifier("a".to_string()),
                Token::DotDotGt,
                Token::Identifier("b".to_string()),
            ]
        );
    }

    #[test]
    fn test_comparison_operators() {
        let input = "== != <= >= < >";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::BinOp("==".to_string()),
                Token::BinOp("!=".to_string()),
                Token::BinOp("<=".to_string()),
                Token::BinOp(">=".to_string()),
                Token::BinOp("<".to_string()),
                Token::BinOp(">".to_string()),
            ]
        );
    }

    #[test]
    fn test_mixed_identifiers_and_operators() {
        // Test realistic expression with new features
        let input = "acc' |> map (\\x -> -x) arr[0]";
        let tokens = tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("acc'".to_string()),
                Token::PipeOp,
                Token::Identifier("map".to_string()),
                Token::LeftParen,
                Token::Backslash,
                Token::Identifier("x".to_string()),
                Token::Arrow,
                Token::BinOp("-".to_string()),
                Token::Identifier("x".to_string()),
                Token::RightParen,
                Token::Identifier("arr".to_string()),
                Token::LeftBracket,
                Token::IntLiteral(0),
                Token::RightBracket,
            ]
        );
    }
}
