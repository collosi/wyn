use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::lexer::Token;
use crate::parser::Parser;
use log::trace;

impl Parser {
    /// Parse a type binding declaration:
    /// ```text
    /// type_bind ::= ("type" | "type^" | "type~") name type_param* "=" type
    /// ```
    pub fn parse_type_bind(&mut self) -> Result<TypeBind> {
        trace!("parse_type_bind: next token = {:?}", self.peek());

        // Parse the type keyword variant
        let kind = match self.peek() {
            Some(Token::Type) => {
                self.advance();
                // Check for ^ or ~ suffix on the identifier that follows
                // For simplicity, we'll just use TypeBindKind::Normal for now
                TypeBindKind::Normal
            }
            _ => return Err(CompilerError::ParseError("Expected 'type' keyword".to_string())),
        };

        let name = self.expect_identifier()?;

        // Parse type parameters
        let mut type_params = Vec::new();
        while self.can_start_type_param() {
            type_params.push(self.parse_type_param()?);
        }

        self.expect(Token::Assign)?;
        let definition = self.parse_type()?;

        Ok(TypeBind {
            kind,
            name,
            type_params,
            definition,
        })
    }

    fn can_start_type_param(&self) -> bool {
        match self.peek() {
            Some(Token::LeftBracket) => true,
            Some(Token::Identifier(s)) if s.starts_with('\'') => true,
            _ => false,
        }
    }

    /// Parse a type parameter:
    /// ```text
    /// type_param ::= "[" name "]" | "'" name | "'~" name | "'^" name
    /// ```
    fn parse_type_param(&mut self) -> Result<TypeParam> {
        match self.peek() {
            Some(Token::LeftBracket) => {
                self.advance();
                let name = self.expect_identifier()?;
                self.expect(Token::RightBracket)?;
                Ok(TypeParam::Size(name))
            }
            Some(Token::Identifier(id)) if id.starts_with('\'') => {
                let id = id.clone();
                self.advance();

                if id.starts_with("'~") {
                    Ok(TypeParam::SizeType(id[2..].to_string()))
                } else if id.starts_with("'^") {
                    Ok(TypeParam::LiftedType(id[2..].to_string()))
                } else if id.starts_with('\'') {
                    Ok(TypeParam::Type(id[1..].to_string()))
                } else {
                    Err(CompilerError::ParseError(format!(
                        "Invalid type parameter: {}",
                        id
                    )))
                }
            }
            _ => Err(CompilerError::ParseError("Expected type parameter".to_string())),
        }
    }

    /// Parse a module binding:
    /// ```text
    /// mod_bind ::= "module" name mod_param* [":" mod_type_exp] ["=" mod_exp]
    /// ```
    /// Note: If no body is provided but a signature is present, this is a signature-only
    /// module instantiation (e.g., `module i32 : (integral with t = i32)`)
    pub fn parse_module_bind(&mut self) -> Result<ModuleBind> {
        trace!("parse_module_bind: next token = {:?}", self.peek());

        self.expect(Token::Module)?;
        let name = self.expect_identifier()?;

        // Parse module parameters
        let mut params = Vec::new();
        while self.check(&Token::LeftParen) {
            params.push(self.parse_module_param()?);
        }

        // Parse optional signature
        let signature = if self.check(&Token::Colon) {
            self.advance();
            Some(self.parse_module_type_expression()?)
        } else {
            None
        };

        // Parse optional body
        // If there's a signature but no body, this is a signature-only instantiation
        let body = if self.check(&Token::Assign) {
            self.advance();
            self.parse_module_expression()?
        } else if signature.is_some() {
            // Signature-only module: create a synthetic empty body
            // The elaborator will recognize this and generate declarations from the signature
            ModuleExpression::Struct(vec![])
        } else {
            return Err(CompilerError::ParseError(
                "Module binding must have either a body (= mod_exp) or a signature (: mod_type)"
                    .to_string(),
            ));
        };

        Ok(ModuleBind {
            name,
            params,
            signature,
            body,
        })
    }

    /// Parse a module parameter:
    /// ```text
    /// mod_param ::= "(" name ":" mod_type_exp ")"
    /// ```
    fn parse_module_param(&mut self) -> Result<ModuleParam> {
        self.expect(Token::LeftParen)?;
        let name = self.expect_identifier()?;
        self.expect(Token::Colon)?;
        let signature = self.parse_module_type_expression()?;
        self.expect(Token::RightParen)?;

        Ok(ModuleParam { name, signature })
    }

    /// Parse a module type binding:
    /// ```text
    /// mod_type_bind ::= "module" "type" name "=" mod_type_exp
    /// ```
    pub fn parse_module_type_bind(&mut self) -> Result<ModuleTypeBind> {
        trace!("parse_module_type_bind: next token = {:?}", self.peek());

        self.expect(Token::Module)?;
        self.expect(Token::Type)?;
        let name = self.expect_identifier()?;
        self.expect(Token::Assign)?;
        let definition = self.parse_module_type_expression()?;

        Ok(ModuleTypeBind { name, definition })
    }

    /// Parse a module expression:
    /// ```text
    /// mod_exp ::= qualname
    ///           | mod_exp ":" mod_type_exp
    ///           | "\" "(" mod_param* ")" [":" mod_type_exp] "->" mod_exp
    ///           | mod_exp mod_exp
    ///           | "(" mod_exp ")"
    ///           | "{" dec* "}"
    ///           | "import" stringlit
    /// ```
    pub fn parse_module_expression(&mut self) -> Result<ModuleExpression> {
        trace!("parse_module_expression: next token = {:?}", self.peek());

        let mut expr = match self.peek() {
            Some(Token::Backslash) => {
                // Lambda: \ (params) [: sig] -> body
                self.advance();
                self.expect(Token::LeftParen)?;

                let mut params = Vec::new();
                while !self.check(&Token::RightParen) {
                    params.push(self.parse_module_param()?);
                    if !self.check(&Token::Comma) {
                        break;
                    }
                    self.advance();
                }
                self.expect(Token::RightParen)?;

                let signature = if self.check(&Token::Colon) {
                    self.advance();
                    Some(self.parse_module_type_expression()?)
                } else {
                    None
                };

                self.expect(Token::Arrow)?;
                let body = self.parse_module_expression()?;

                ModuleExpression::Lambda(params, signature, Box::new(body))
            }

            Some(Token::LeftParen) => {
                // Parenthesized module expression
                self.advance();
                let expr = self.parse_module_expression()?;
                self.expect(Token::RightParen)?;
                expr
            }

            Some(Token::LeftBrace) => {
                // Struct: { dec* }
                self.advance();
                let mut declarations = Vec::new();

                while !self.check(&Token::RightBrace) {
                    declarations.push(self.parse_declaration()?);
                }

                self.expect(Token::RightBrace)?;
                ModuleExpression::Struct(declarations)
            }

            Some(Token::Import) => {
                // import "path"
                self.advance();
                let path = self.expect_string_literal()?;
                ModuleExpression::Import(path)
            }

            Some(Token::Identifier(_)) => {
                // qualname
                let name = self.expect_identifier()?;
                ModuleExpression::Name(name)
            }

            _ => {
                return Err(CompilerError::ParseError(format!(
                    "Expected module expression, got {:?}",
                    self.peek()
                )));
            }
        };

        // Handle postfix operations
        loop {
            match self.peek() {
                Some(Token::Colon) => {
                    // Ascription: mod_exp : mod_type_exp
                    self.advance();
                    let sig = self.parse_module_type_expression()?;
                    expr = ModuleExpression::Ascription(Box::new(expr), sig);
                }

                Some(Token::Identifier(_)) | Some(Token::LeftParen) | Some(Token::LeftBrace) => {
                    // Application: mod_exp mod_exp
                    let arg = self.parse_module_expression_atom()?;
                    expr = ModuleExpression::Application(Box::new(expr), Box::new(arg));
                }

                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_module_expression_atom(&mut self) -> Result<ModuleExpression> {
        match self.peek() {
            Some(Token::LeftParen) => {
                self.advance();
                let expr = self.parse_module_expression()?;
                self.expect(Token::RightParen)?;
                Ok(expr)
            }

            Some(Token::LeftBrace) => {
                self.advance();
                let mut declarations = Vec::new();
                while !self.check(&Token::RightBrace) {
                    declarations.push(self.parse_declaration()?);
                }
                self.expect(Token::RightBrace)?;
                Ok(ModuleExpression::Struct(declarations))
            }

            Some(Token::Identifier(_)) => {
                let name = self.expect_identifier()?;
                Ok(ModuleExpression::Name(name))
            }

            _ => Err(CompilerError::ParseError(format!(
                "Expected module expression atom, got {:?}",
                self.peek()
            ))),
        }
    }

    /// Parse a module type expression:
    /// ```text
    /// mod_type_exp ::= qualname
    ///                | "{" spec* "}"
    ///                | mod_type_exp "with" qualname type_param* "=" type
    ///                | "(" mod_type_exp ")"
    ///                | "(" name ":" mod_type_exp ")" "->" mod_type_exp
    ///                | mod_type_exp "->" mod_type_exp
    /// ```
    pub fn parse_module_type_expression(&mut self) -> Result<ModuleTypeExpression> {
        trace!("parse_module_type_expression: next token = {:?}", self.peek());

        let mut expr = match self.peek() {
            Some(Token::LeftParen) => {
                self.advance();

                // Could be: (mod_type) or (name : mod_type) -> mod_type
                // We need to check for arrow type carefully:
                // Save position to backtrack if needed
                let checkpoint = self.current;

                // Try to parse as arrow type: (name : mod_type) -> mod_type
                if let Some(Token::Identifier(_)) = self.peek() {
                    let name = self.expect_identifier()?;

                    if self.check(&Token::Colon) {
                        // Arrow type: (name : mod_type) -> mod_type
                        self.advance();
                        let param_sig = self.parse_module_type_expression()?;
                        self.expect(Token::RightParen)?;
                        self.expect(Token::Arrow)?;
                        let result = self.parse_module_type_expression()?;
                        return Ok(ModuleTypeExpression::Arrow(
                            name,
                            Box::new(param_sig),
                            Box::new(result),
                        ));
                    } else {
                        // Not an arrow type, backtrack and parse as regular expression
                        self.current = checkpoint;
                    }
                }

                // Parenthesized module type expression
                let expr = self.parse_module_type_expression()?;
                self.expect(Token::RightParen)?;
                expr
            }

            Some(Token::LeftBrace) => {
                // Signature: { spec* }
                self.advance();
                let mut specs = Vec::new();

                while !self.check(&Token::RightBrace) {
                    specs.push(self.parse_spec()?);
                }

                self.expect(Token::RightBrace)?;
                ModuleTypeExpression::Signature(specs)
            }

            Some(Token::Identifier(_)) => {
                let name = self.expect_identifier()?;
                ModuleTypeExpression::Name(name)
            }

            _ => {
                return Err(CompilerError::ParseError(format!(
                    "Expected module type expression, got {:?}",
                    self.peek()
                )));
            }
        };

        // Handle postfix operations
        loop {
            match self.peek() {
                Some(Token::With) => {
                    // with clause: mod_type with qualname type_params = type
                    self.advance();
                    let qualname = self.expect_identifier()?;

                    let mut type_params = Vec::new();
                    while self.can_start_type_param() {
                        type_params.push(self.parse_type_param()?);
                    }

                    self.expect(Token::Assign)?;
                    let ty = self.parse_type()?;

                    expr = ModuleTypeExpression::With(Box::new(expr), qualname, type_params, ty);
                }

                Some(Token::Arrow) => {
                    // Functor type: mod_type -> mod_type
                    self.advance();
                    let result = self.parse_module_type_expression()?;
                    expr = ModuleTypeExpression::FunctorType(Box::new(expr), Box::new(result));
                }

                _ => break,
            }
        }

        Ok(expr)
    }

    /// Parse a specification:
    /// ```text
    /// spec ::= "sig" name type_param* ":" type
    ///        | "sig" "(" symbol ")" ":" type
    ///        | "sig" symbol type_param* ":" type
    ///        | ("type" | "type^" | "type~") name type_param* "=" type
    ///        | ("type" | "type^" | "type~") name type_param*
    ///        | "module" name ":" mod_type_exp
    ///        | "include" mod_type_exp
    ///        | "#[" attr "]" spec
    /// ```
    fn parse_spec(&mut self) -> Result<Spec> {
        trace!("parse_spec: next token = {:?}", self.peek());

        // Parse optional attributes (for attributed specs)
        let _attributes = self.parse_attributes()?;

        match self.peek() {
            Some(Token::Sig) => {
                self.advance();

                // Check for operator in parens: sig (op) : type
                if self.check(&Token::LeftParen) {
                    self.advance();
                    // Handle both identifiers and binary operators
                    let op = match self.peek() {
                        Some(Token::Identifier(id)) => {
                            let id = id.clone();
                            self.advance();
                            id
                        }
                        Some(Token::BinOp(op)) => {
                            let op = op.clone();
                            self.advance();
                            op
                        }
                        _ => {
                            return Err(CompilerError::ParseError(format!(
                                "Expected identifier or operator in sig spec at {}",
                                self.current_span()
                            )));
                        }
                    };
                    self.expect(Token::RightParen)?;
                    self.expect(Token::Colon)?;
                    let ty = self.parse_type()?;
                    return Ok(Spec::SigOp(op, ty));
                }

                let name = self.expect_identifier()?;

                // Parse type parameters
                let mut type_params = Vec::new();
                while self.can_start_type_param() {
                    type_params.push(self.parse_type_param()?);
                }

                self.expect(Token::Colon)?;
                let ty = self.parse_type()?;

                Ok(Spec::Sig(name, type_params, ty))
            }

            Some(Token::Type) => {
                self.advance();
                let kind = TypeBindKind::Normal; // Simplified
                let name = self.expect_identifier()?;

                let mut type_params = Vec::new();
                while self.can_start_type_param() {
                    type_params.push(self.parse_type_param()?);
                }

                let definition = if self.check(&Token::Assign) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };

                Ok(Spec::Type(kind, name, type_params, definition))
            }

            Some(Token::Module) => {
                self.advance();
                let name = self.expect_identifier()?;
                self.expect(Token::Colon)?;
                let sig = self.parse_module_type_expression()?;

                Ok(Spec::Module(name, sig))
            }

            Some(Token::Include) => {
                self.advance();
                let sig = self.parse_module_type_expression()?;
                Ok(Spec::Include(sig))
            }

            _ => Err(CompilerError::ParseError(format!(
                "Expected spec, got {:?}",
                self.peek()
            ))),
        }
    }
}
