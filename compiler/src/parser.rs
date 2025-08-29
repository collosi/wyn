use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::lexer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut declarations = Vec::new();

        while !self.is_at_end() {
            declarations.push(self.parse_declaration()?);
        }

        Ok(Program { declarations })
    }

    fn parse_declaration(&mut self) -> Result<Declaration> {
        match self.peek() {
            Some(Token::Let) => Ok(Declaration::Let(self.parse_let_decl()?)),
            Some(Token::Entry) => Ok(Declaration::Entry(self.parse_entry_decl()?)),
            Some(Token::Def) => Ok(Declaration::Def(self.parse_def_decl()?)),
            Some(Token::Val) => Ok(Declaration::Val(self.parse_val_decl()?)),
            _ => Err(CompilerError::ParseError(
                "Expected 'let', 'entry', 'def', or 'val' declaration".to_string(),
            )),
        }
    }

    fn parse_let_decl(&mut self) -> Result<LetDecl> {
        self.expect(Token::Let)?;
        let name = self.expect_identifier()?;
        self.expect(Token::Colon)?;
        let ty = Some(self.parse_type()?);
        self.expect(Token::Assign)?;
        let value = self.parse_expression()?;

        Ok(LetDecl { name, ty, value })
    }

    fn parse_entry_decl(&mut self) -> Result<EntryDecl> {
        self.expect(Token::Entry)?;
        let name = self.expect_identifier()?;
        self.expect(Token::LeftParen)?;

        let mut params = Vec::new();
        if !self.check(&Token::RightParen) {
            loop {
                let param_name = self.expect_identifier()?;
                self.expect(Token::Colon)?;
                let param_ty = self.parse_type()?;
                params.push(Parameter {
                    name: param_name,
                    ty: param_ty,
                });

                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance();
            }
        }

        self.expect(Token::RightParen)?;
        self.expect(Token::Colon)?;
        let return_type = self.parse_type()?;
        self.expect(Token::Assign)?;
        let body = self.parse_expression()?;

        Ok(EntryDecl {
            name,
            params,
            return_type,
            body,
        })
    }

    fn parse_def_decl(&mut self) -> Result<DefDecl> {
        self.expect(Token::Def)?;
        let name = self.expect_identifier()?;

        // Parse parameter names (without types for inference)
        let mut params = Vec::new();
        while !self.check(&Token::Assign) && !self.is_at_end() {
            params.push(self.expect_identifier()?);
        }

        self.expect(Token::Assign)?;
        let body = self.parse_expression()?;

        Ok(DefDecl { name, params, body })
    }

    fn parse_val_decl(&mut self) -> Result<ValDecl> {
        self.expect(Token::Val)?;
        let name = self.expect_identifier()?;

        // Parse size parameters: [n] [m] ...
        let mut size_params = Vec::new();
        while self.check(&Token::LeftBracket) {
            self.advance(); // consume '['
            let size_param = self.expect_identifier()?;
            size_params.push(size_param);
            self.expect(Token::RightBracket)?;
        }

        // Parse type parameters: 'a 'b ...
        let mut type_params = Vec::new();
        while self.check_type_variable() {
            // Expect apostrophe followed by identifier
            let type_param = self.parse_type_variable()?;
            type_params.push(type_param);
        }

        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;

        Ok(ValDecl {
            name,
            size_params,
            type_params,
            ty,
        })
    }

    fn check_type_variable(&self) -> bool {
        // Check if current token is an apostrophe (we'll need to add this to lexer)
        // For now, we'll use a simplified approach
        match self.peek() {
            Some(Token::Identifier(name)) if name.starts_with('\'') => true,
            _ => false,
        }
    }

    fn parse_type_variable(&mut self) -> Result<String> {
        match self.advance() {
            Some(Token::Identifier(name)) if name.starts_with('\'') => {
                Ok(name[1..].to_string()) // Remove the apostrophe
            }
            _ => Err(CompilerError::ParseError(
                "Expected type variable (e.g., 'a)".to_string(),
            )),
        }
    }

    fn parse_type(&mut self) -> Result<Type> {
        self.parse_function_type()
    }

    fn parse_function_type(&mut self) -> Result<Type> {
        let mut left = self.parse_array_or_base_type()?;

        // Handle function arrows: T1 -> T2 -> T3
        while self.check(&Token::Arrow) {
            // We'll need to add Arrow token
            self.advance();
            let right = self.parse_array_or_base_type()?;
            left = Type::Function(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_array_or_base_type(&mut self) -> Result<Type> {
        // Check for array type [dim]baseType (Futhark style)
        if self.check(&Token::LeftBracket) {
            self.advance(); // consume '['

            // Parse dimension - could be integer literal or identifier (size variable)
            if let Some(Token::IntLiteral(n)) = self.peek() {
                let n = *n as usize;
                self.advance();
                self.expect(Token::RightBracket)?;
                let elem_type = self.parse_array_or_base_type()?; // Allow nested arrays
                Ok(Type::Array(Box::new(elem_type), vec![n]))
            } else {
                // Size variable like 'n'
                let _size_var = self.expect_identifier()?;
                self.expect(Token::RightBracket)?;
                let elem_type = self.parse_array_or_base_type()?; // Allow nested arrays
                                                                  // For now, we'll represent size variables as arrays with size 0 as a placeholder
                                                                  // In a proper implementation, we'd need to track size variables differently
                Ok(Type::Array(Box::new(elem_type), vec![0])) // placeholder for size variables
            }
        } else {
            self.parse_base_type()
        }
    }

    fn parse_base_type(&mut self) -> Result<Type> {
        match self.peek() {
            Some(Token::I32Type) => {
                self.advance();
                Ok(Type::I32)
            }
            Some(Token::F32Type) => {
                self.advance();
                Ok(Type::F32)
            }
            Some(Token::Vec4F32Type) => {
                self.advance();
                Ok(Type::Vec4F32)
            }
            Some(Token::Identifier(name)) if name.starts_with('\'') => {
                // Type variable like 't1, 't2
                let type_var = self.parse_type_variable()?;
                Ok(Type::Var(type_var))
            }
            Some(Token::LeftParen) => {
                // Tuple type (T1, T2, T3)
                self.advance(); // consume '('
                let mut types = Vec::new();

                if !self.check(&Token::RightParen) {
                    loop {
                        types.push(self.parse_type()?);
                        if !self.check(&Token::Comma) {
                            break;
                        }
                        self.advance(); // consume ','
                    }
                }

                self.expect(Token::RightParen)?;
                Ok(Type::Tuple(types))
            }
            _ => Err(CompilerError::ParseError("Expected type".to_string())),
        }
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_binary_expression()
    }

    fn parse_binary_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_postfix_expression()?;

        while let Some(Token::Divide) = self.peek() {
            self.advance();
            let right = self.parse_postfix_expression()?;
            left = Expression::BinaryOp(BinaryOp::Divide, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_postfix_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary_expression()?;

        while let Some(Token::LeftBracket) = self.peek() {
            self.advance();
            let index = self.parse_expression()?;
            self.expect(Token::RightBracket)?;
            expr = Expression::ArrayIndex(Box::new(expr), Box::new(index));
        }

        Ok(expr)
    }

    fn parse_primary_expression(&mut self) -> Result<Expression> {
        match self.peek() {
            Some(Token::IntLiteral(n)) => {
                let n = *n;
                self.advance();
                Ok(Expression::IntLiteral(n))
            }
            Some(Token::FloatLiteral(f)) => {
                let f = *f;
                self.advance();
                Ok(Expression::FloatLiteral(f))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.advance();

                // Check if this is a function call (identifier followed by arguments)
                // For simplicity, we'll parse function calls without parentheses like "zip xs ys"
                let mut args = Vec::new();

                // Collect arguments until we hit an operator or end of expression
                while self.peek().is_some()
                    && !matches!(
                        self.peek(),
                        Some(Token::Divide)
                            | Some(Token::Comma)
                            | Some(Token::RightBracket)
                            | Some(Token::RightParen)
                    )
                {
                    // If we see an identifier, parse it as an argument (with potential postfix operations)
                    if let Some(Token::Identifier(_)) = self.peek() {
                        let mut arg = self.parse_primary_expression()?;

                        // Handle array indexing on arguments
                        while let Some(Token::LeftBracket) = self.peek() {
                            self.advance();
                            let index = self.parse_expression()?;
                            self.expect(Token::RightBracket)?;
                            arg = Expression::ArrayIndex(Box::new(arg), Box::new(index));
                        }

                        args.push(arg);
                    } else {
                        break;
                    }
                }

                if !args.is_empty() {
                    Ok(Expression::FunctionCall(name, args))
                } else {
                    Ok(Expression::Identifier(name))
                }
            }
            Some(Token::LeftBracket) => self.parse_array_literal(),
            _ => Err(CompilerError::ParseError("Expected expression".to_string())),
        }
    }

    fn parse_array_literal(&mut self) -> Result<Expression> {
        self.expect(Token::LeftBracket)?;

        let mut elements = Vec::new();
        if !self.check(&Token::RightBracket) {
            loop {
                elements.push(self.parse_expression()?);
                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance();
            }
        }

        self.expect(Token::RightBracket)?;
        Ok(Expression::ArrayLiteral(elements))
    }

    // Helper methods
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current)
    }

    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.current += 1;
            self.tokens.get(self.current - 1)
        } else {
            None
        }
    }

    fn check(&self, token: &Token) -> bool {
        if let Some(t) = self.peek() {
            std::mem::discriminant(t) == std::mem::discriminant(token)
        } else {
            false
        }
    }

    fn expect(&mut self, token: Token) -> Result<()> {
        if self.check(&token) {
            self.advance();
            Ok(())
        } else {
            Err(CompilerError::ParseError(format!(
                "Expected {:?}, got {:?}",
                token,
                self.peek()
            )))
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name.clone()),
            _ => Err(CompilerError::ParseError("Expected identifier".to_string())),
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    #[test]
    fn test_parse_let_decl() {
        let input = "let x: i32 = 42";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        assert_eq!(program.declarations.len(), 1);
        match &program.declarations[0] {
            Declaration::Let(decl) => {
                assert_eq!(decl.name, "x");
                assert_eq!(decl.ty, Some(Type::I32));
                assert_eq!(decl.value, Expression::IntLiteral(42));
            }
            _ => panic!("Expected Let declaration"),
        }
    }

    #[test]
    fn test_parse_array_type() {
        let input = "let arr: [3][4]f32 = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        match &program.declarations[0] {
            Declaration::Let(decl) => {
                assert_eq!(decl.name, "arr");
                // The type should be Array(Array(F32, [4]), [3])
                match &decl.ty {
                    Some(Type::Array(inner, dims)) => {
                        assert_eq!(dims, &vec![3]);
                        match inner.as_ref() {
                            Type::Array(base, inner_dims) => {
                                assert_eq!(inner_dims, &vec![4]);
                                assert_eq!(**base, Type::F32);
                            }
                            _ => panic!("Expected nested array type"),
                        }
                    }
                    _ => panic!("Expected array type"),
                }
            }
            _ => panic!("Expected Let declaration"),
        }
    }

    #[test]
    fn test_parse_entry_decl() {
        let input = "entry main(x: i32, y: f32): [4]f32 = result";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        match &program.declarations[0] {
            Declaration::Entry(decl) => {
                assert_eq!(decl.name, "main");
                assert_eq!(decl.params.len(), 2);
                assert_eq!(decl.params[0].name, "x");
                assert_eq!(decl.params[0].ty, Type::I32);
                assert_eq!(decl.params[1].name, "y");
                assert_eq!(decl.params[1].ty, Type::F32);
                assert_eq!(decl.return_type, Type::Array(Box::new(Type::F32), vec![4]));
            }
            _ => panic!("Expected Entry declaration"),
        }
    }

    #[test]
    fn test_parse_array_index() {
        let input = "let x: f32 = arr[0]";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        match &program.declarations[0] {
            Declaration::Let(decl) => match &decl.value {
                Expression::ArrayIndex(arr, idx) => {
                    assert_eq!(**arr, Expression::Identifier("arr".to_string()));
                    assert_eq!(**idx, Expression::IntLiteral(0));
                }
                _ => panic!("Expected ArrayIndex expression"),
            },
            _ => panic!("Expected Let declaration"),
        }
    }

    #[test]
    fn test_parse_division() {
        let input = "let x: f32 = 135f32/255f32";
        let tokens = tokenize(input).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        match &program.declarations[0] {
            Declaration::Let(decl) => match &decl.value {
                Expression::BinaryOp(BinaryOp::Divide, left, right) => {
                    assert_eq!(**left, Expression::FloatLiteral(135.0));
                    assert_eq!(**right, Expression::FloatLiteral(255.0));
                }
                _ => panic!("Expected BinaryOp expression"),
            },
            _ => panic!("Expected Let declaration"),
        }
    }
}
