use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::lexer::Token;
use crate::parser::Parser;
use log::trace;

impl Parser {
    /// Parse a pattern according to the grammar:
    /// ```text
    /// pat ::= name
    ///       | pat_literal
    ///       | "_"
    ///       | "(" ")"
    ///       | "(" pat ")"
    ///       | "(" pat ("," pat)+ [","] ")"
    ///       | "{" "}"
    ///       | "{" fieldid ["=" pat] ("," fieldid ["=" pat])* [","] "}"
    ///       | constructor pat*
    ///       | pat ":" type
    ///       | "#[" attr "]" pat
    /// ```
    pub fn parse_pattern(&mut self) -> Result<Pattern> {
        trace!("parse_pattern: next token = {:?}", self.peek());

        // Parse optional attributes
        let attributes = self.parse_attributes()?;

        let pattern = if !attributes.is_empty() {
            // #[attr] pat
            let inner = self.parse_pattern_without_attributes()?;
            self.node_counter.mk_node_dummy(PatternKind::Attributed(attributes, Box::new(inner)))
        } else {
            self.parse_pattern_without_attributes()?
        };

        // Check for type annotation (pat : type)
        if self.check(&Token::Colon) {
            self.advance();
            let ty = self.parse_type()?;
            return Ok(self.node_counter.mk_node_dummy(PatternKind::Typed(Box::new(pattern), ty)));
        }

        Ok(pattern)
    }

    fn parse_pattern_without_attributes(&mut self) -> Result<Pattern> {
        match self.peek() {
            Some(Token::Underscore) => {
                self.advance();
                Ok(self.node_counter.mk_node_dummy(PatternKind::Wildcard))
            }

            Some(Token::LeftParen) => self.parse_paren_pattern(),

            Some(Token::LeftBrace) => self.parse_record_pattern(),

            Some(Token::Identifier(name)) => {
                // Check if it's a constructor (starts with uppercase)
                if name.chars().next().is_some_and(|c| c.is_uppercase()) {
                    self.parse_constructor_pattern()
                } else {
                    // Simple name binding
                    let name = self.expect_identifier()?;
                    Ok(self.node_counter.mk_node_dummy(PatternKind::Name(name)))
                }
            }

            Some(Token::IntLiteral(_))
            | Some(Token::FloatLiteral(_))
            | Some(Token::True)
            | Some(Token::False)
            | Some(Token::CharLiteral(_)) => self.parse_pattern_literal(),

            Some(Token::Minus) => {
                // Negative literal
                self.parse_pattern_literal()
            }

            _ => Err(CompilerError::ParseError(format!(
                "Expected pattern, got {:?}",
                self.peek()
            ))),
        }
    }

    fn parse_paren_pattern(&mut self) -> Result<Pattern> {
        self.expect(Token::LeftParen)?;

        if self.check(&Token::RightParen) {
            // () unit pattern
            self.advance();
            return Ok(self.node_counter.mk_node_dummy(PatternKind::Unit));
        }

        let first = self.parse_pattern()?;

        if self.check(&Token::Comma) {
            // Tuple pattern: (pat, pat, ...)
            let mut patterns = vec![first];

            while self.check(&Token::Comma) {
                self.advance();
                // Allow trailing comma
                if self.check(&Token::RightParen) {
                    break;
                }
                patterns.push(self.parse_pattern()?);
            }

            self.expect(Token::RightParen)?;
            Ok(self.node_counter.mk_node_dummy(PatternKind::Tuple(patterns)))
        } else {
            // Single pattern in parens: (pat)
            self.expect(Token::RightParen)?;
            Ok(first)
        }
    }

    fn parse_record_pattern(&mut self) -> Result<Pattern> {
        self.expect(Token::LeftBrace)?;

        if self.check(&Token::RightBrace) {
            // {} empty record
            self.advance();
            return Ok(self.node_counter.mk_node_dummy(PatternKind::Record(vec![])));
        }

        let mut fields = Vec::new();

        loop {
            let field_name = self.expect_identifier()?;

            let pattern = if self.check(&Token::Assign) {
                // field = pat
                self.advance();
                Some(self.parse_pattern()?)
            } else {
                // Shorthand: just field name
                None
            };

            fields.push(RecordPatternField {
                field: field_name,
                pattern,
            });

            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();

            // Allow trailing comma
            if self.check(&Token::RightBrace) {
                break;
            }
        }

        self.expect(Token::RightBrace)?;
        Ok(self.node_counter.mk_node_dummy(PatternKind::Record(fields)))
    }

    fn parse_constructor_pattern(&mut self) -> Result<Pattern> {
        let constructor = self.expect_identifier()?;

        // Parse constructor arguments (zero or more patterns)
        let mut args = Vec::new();

        // Keep parsing patterns as long as the next token can start a pattern
        // but stop if we see tokens that indicate the end of the constructor pattern
        while self.can_start_pattern() && !self.is_pattern_terminator() {
            args.push(self.parse_pattern_without_attributes()?);
        }

        Ok(self.node_counter.mk_node_dummy(PatternKind::Constructor(constructor, args)))
    }

    fn can_start_pattern(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Identifier(_))
                | Some(Token::Underscore)
                | Some(Token::LeftParen)
                | Some(Token::LeftBrace)
                | Some(Token::IntLiteral(_))
                | Some(Token::FloatLiteral(_))
                | Some(Token::CharLiteral(_))
                | Some(Token::True)
                | Some(Token::False)
                | Some(Token::Minus)
        )
    }

    fn is_pattern_terminator(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Assign)
                | Some(Token::In)
                | Some(Token::Arrow)
                | Some(Token::RightParen)
                | Some(Token::RightBrace)
                | Some(Token::Comma)
                | Some(Token::Colon)
        )
    }

    /// Parse a pattern literal:
    /// ```text
    /// pat_literal ::= [ "-" ] intnumber
    ///               | [ "-" ] floatnumber
    ///               | charlit
    ///               | "true"
    ///               | "false"
    /// ```
    fn parse_pattern_literal(&mut self) -> Result<Pattern> {
        trace!("parse_pattern_literal: next token = {:?}", self.peek());

        // Check for negative sign
        let is_negative = if self.check(&Token::Minus) {
            self.advance();
            true
        } else {
            false
        };

        let literal = match self.peek() {
            Some(Token::IntLiteral(n)) => {
                let value = *n;
                self.advance();
                PatternLiteral::Int(if is_negative { -value } else { value })
            }

            Some(Token::FloatLiteral(f)) => {
                let value = *f;
                self.advance();
                PatternLiteral::Float(if is_negative { -value } else { value })
            }

            Some(Token::CharLiteral(c)) => {
                if is_negative {
                    return Err(CompilerError::ParseError(
                        "Character literals cannot be negative".to_string(),
                    ));
                }
                let ch = *c;
                self.advance();
                PatternLiteral::Char(ch)
            }

            Some(Token::True) => {
                if is_negative {
                    return Err(CompilerError::ParseError(
                        "Boolean literals cannot be negative".to_string(),
                    ));
                }
                self.advance();
                PatternLiteral::Bool(true)
            }

            Some(Token::False) => {
                if is_negative {
                    return Err(CompilerError::ParseError(
                        "Boolean literals cannot be negative".to_string(),
                    ));
                }
                self.advance();
                PatternLiteral::Bool(false)
            }

            _ => {
                return Err(CompilerError::ParseError(format!(
                    "Expected literal in pattern, got {:?}",
                    self.peek()
                )));
            }
        };

        Ok(self.node_counter.mk_node_dummy(PatternKind::Literal(literal)))
    }
}
