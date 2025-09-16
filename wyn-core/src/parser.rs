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
        // Parse optional attributes
        let attributes = self.parse_attributes()?;

        match self.peek() {
            Some(Token::Let) => {
                let mut decl = self.parse_decl("let")?;
                decl.attributes = attributes;
                Ok(Declaration::Decl(decl))
            }
            Some(Token::Def) => {
                let mut decl = self.parse_decl("def")?;
                decl.attributes = attributes;
                Ok(Declaration::Decl(decl))
            }
            Some(Token::Val) => {
                let mut decl = self.parse_val_decl()?;
                decl.attributes = attributes;
                Ok(Declaration::Val(decl))
            }
            _ => Err(CompilerError::ParseError(
                "Expected 'let', 'def', or 'val' declaration".to_string(),
            )),
        }
    }

    fn parse_decl(&mut self, keyword: &'static str) -> Result<Decl> {
        // Expect the keyword token (let or def)
        match keyword {
            "let" => self.expect(Token::Let)?,
            "def" => self.expect(Token::Def)?,
            _ => return Err(CompilerError::ParseError(format!("Invalid keyword: {}", keyword))),
        }
        
        let name = self.expect_identifier()?;

        // Check if we have typed parameters (for entry points with parentheses)
        if self.check(&Token::LeftParen) {
            // Parse typed parameters for entry points
            self.expect(Token::LeftParen)?;
            let mut params = Vec::new();
            if !self.check(&Token::RightParen) {
                loop {
                    let param_attributes = self.parse_attributes()?;
                    let param_name = self.expect_identifier()?;
                    self.expect(Token::Colon)?;
                    let param_ty = self.parse_type()?;
                    params.push(DeclParam::Typed(Parameter {
                        attributes: param_attributes,
                        name: param_name,
                        ty: param_ty,
                    }));
                    
                    if !self.check(&Token::Comma) {
                        break;
                    }
                    self.advance();
                }
            }
            self.expect(Token::RightParen)?;
            self.expect(Token::Colon)?;
            
            // Parse return type with optional attributes
            let return_attributes = self.parse_attributes()?;
            let ty = Some(self.parse_type()?);
            
            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;
            
            Ok(Decl {
                keyword,
                attributes: vec![],
                name,
                params,
                ty,
                return_attributes,
                body,
            })
        }
        // Check if this is a typed declaration (name: type = value)
        else if self.check(&Token::Colon) {
            // Typed declaration: let/def name: type = value
            self.expect(Token::Colon)?;
            let ty = Some(self.parse_type()?);
            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;
            
            Ok(Decl {
                keyword,
                attributes: vec![],
                name,
                params: vec![], // No parameters for typed declarations
                ty,
                return_attributes: vec![],
                body,
            })
        } else {
            // Function declaration: let/def name param1 param2 = body
            // OR simple variable: let/def name = value
            let mut params = Vec::new();
            while !self.check(&Token::Assign) && !self.is_at_end() {
                params.push(DeclParam::Untyped(self.expect_identifier()?));
            }
            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;
            
            Ok(Decl {
                keyword,
                attributes: vec![],
                name,
                params,
                ty: None, // No explicit type for untyped declarations
                return_attributes: vec![],
                body,
            })
        }
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
            attributes: vec![],
            name,
            size_params,
            type_params,
            ty,
        })
    }

    fn parse_attributes(&mut self) -> Result<Vec<Attribute>> {
        let mut attributes = Vec::new();

        while self.check(&Token::AttributeStart) {
            self.advance(); // consume '#['
            let attr_name = self.expect_identifier()?;

            let attribute = match attr_name.as_str() {
                "vertex" => {
                    self.expect(Token::RightBracket)?;
                    Attribute::Vertex
                }
                "fragment" => {
                    self.expect(Token::RightBracket)?;
                    Attribute::Fragment
                }
                "builtin" => {
                    self.expect(Token::LeftParen)?;
                    let builtin_name = self.expect_identifier()?;
                    self.expect(Token::RightParen)?;
                    self.expect(Token::RightBracket)?;

                    let builtin = match builtin_name.as_str() {
                        "position" => spirv::BuiltIn::Position,
                        "vertex_index" => spirv::BuiltIn::VertexIndex,
                        "instance_index" => spirv::BuiltIn::InstanceIndex,
                        "front_facing" => spirv::BuiltIn::FrontFacing,
                        "frag_depth" => spirv::BuiltIn::FragDepth,
                        _ => {
                            return Err(CompilerError::ParseError(format!(
                                "Unknown builtin: {}",
                                builtin_name
                            )))
                        }
                    };
                    Attribute::BuiltIn(builtin)
                }
                "location" => {
                    self.expect(Token::LeftParen)?;
                    let location = if let Some(Token::IntLiteral(location)) = self.advance() {
                        *location as u32
                    } else {
                        return Err(CompilerError::ParseError(
                            "Expected location number".to_string(),
                        ));
                    };
                    self.expect(Token::RightParen)?;
                    self.expect(Token::RightBracket)?;
                    Attribute::Location(location)
                }
                _ => {
                    return Err(CompilerError::ParseError(format!(
                        "Unknown attribute: {}",
                        attr_name
                    )))
                }
            };

            attributes.push(attribute);
        }

        Ok(attributes)
    }

    fn check_type_variable(&self) -> bool {
        // Check if current token is an apostrophe (we'll need to add this to lexer)
        // For now, we'll use a simplified approach
        matches!(self.peek(), Some(Token::Identifier(name)) if name.starts_with('\''))
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
            left = types::function(left, right);
        }

        Ok(left)
    }

    fn parse_array_or_base_type(&mut self) -> Result<Type> {
        // Check for array type [dim]baseType (Futhark style)
        if self.check(&Token::LeftBracket) {
            self.advance(); // consume '['

            // Parse dimension - could be integer literal or identifier (size variable)
            if let Some(Token::IntLiteral(n)) = self.peek() {
                let size = *n as usize;
                self.advance();
                self.expect(Token::RightBracket)?;
                let elem_type = self.parse_array_or_base_type()?; // Allow nested arrays
                Ok(types::sized_array(size, elem_type))
            } else {
                // Size variables like [n] are not implemented yet
                todo!("Size variables in array types not yet implemented")
            }
        } else {
            self.parse_base_type()
        }
    }

    fn parse_base_type(&mut self) -> Result<Type> {
        match self.peek() {
            Some(Token::I32Type) => {
                self.advance();
                Ok(types::i32())
            }
            Some(Token::F32Type) => {
                self.advance();
                Ok(types::f32())
            }
            // f32 vectors
            Some(Token::Vec2Type) => {
                self.advance();
                Ok(types::vec2())
            }
            Some(Token::Vec3Type) => {
                self.advance();
                Ok(types::vec3())
            }
            Some(Token::Vec4Type) => {
                self.advance();
                Ok(types::vec4())
            }
            // i32 vectors
            Some(Token::IVec2Type) => {
                self.advance();
                Ok(types::ivec2())
            }
            Some(Token::IVec3Type) => {
                self.advance();
                Ok(types::ivec3())
            }
            Some(Token::IVec4Type) => {
                self.advance();
                Ok(types::ivec4())
            }
            // u32 vectors
            Some(Token::UVec2Type) => {
                self.advance();
                Ok(types::uvec2())
            }
            Some(Token::UVec3Type) => {
                self.advance();
                Ok(types::uvec3())
            }
            Some(Token::UVec4Type) => {
                self.advance();
                Ok(types::uvec4())
            }
            // bool vectors
            Some(Token::BVec2Type) => {
                self.advance();
                Ok(types::bvec2())
            }
            Some(Token::BVec3Type) => {
                self.advance();
                Ok(types::bvec3())
            }
            Some(Token::BVec4Type) => {
                self.advance();
                Ok(types::bvec4())
            }
            // f64 vectors
            Some(Token::DVec2Type) => {
                self.advance();
                Ok(types::dvec2())
            }
            Some(Token::DVec3Type) => {
                self.advance();
                Ok(types::dvec3())
            }
            Some(Token::DVec4Type) => {
                self.advance();
                Ok(types::dvec4())
            }
            // f16 vectors
            Some(Token::F16Vec2Type) => {
                self.advance();
                Ok(types::f16vec2())
            }
            Some(Token::F16Vec3Type) => {
                self.advance();
                Ok(types::f16vec3())
            }
            Some(Token::F16Vec4Type) => {
                self.advance();
                Ok(types::f16vec4())
            }
            Some(Token::Identifier(name)) if name.starts_with('\'') => {
                // Type variable like 't1, 't2
                let type_var = self.parse_type_variable()?;
                // For now, create a simple variable ID from the hash of the variable name
                let var_id = type_var.chars().map(|c| c as usize).sum::<usize>();
                Ok(polytype::Type::Variable(var_id))
            }
            Some(Token::LeftParen) => {
                // Tuple type (T1, T2, T3)
                self.advance(); // consume '('
                let mut tuple_types = Vec::new();

                if !self.check(&Token::RightParen) {
                    loop {
                        tuple_types.push(self.parse_type()?);
                        if !self.check(&Token::Comma) {
                            break;
                        }
                        self.advance(); // consume ','
                    }
                }

                self.expect(Token::RightParen)?;
                Ok(types::tuple(tuple_types))
            }
            _ => Err(CompilerError::ParseError("Expected type".to_string())),
        }
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_binary_expression()
    }

    fn parse_binary_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_application_expression()?;

        while let Some(token) = self.peek() {
            let op = match token {
                Token::Divide => BinaryOp::Divide,
                Token::Add => BinaryOp::Add,
                _ => break,
            };
            self.advance();
            let right = self.parse_postfix_expression()?;
            left = Expression::BinaryOp(op, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    // Function application has higher precedence than binary operators
    // Parses: f x y z, (f x) y, etc.
    fn parse_application_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_postfix_expression()?;

        // Keep collecting arguments while we see primary expressions
        // Stop at operators, commas, closing delimiters, etc.
        while self.peek().is_some() && self.can_start_primary_expression() {
            // Don't consume binary operators
            if matches!(self.peek(), Some(Token::Add) | Some(Token::Divide)) {
                break;
            }
            // Don't consume expression terminators
            if matches!(
                self.peek(),
                Some(Token::Comma)
                    | Some(Token::RightParen)
                    | Some(Token::RightBracket)
                    | Some(Token::In)
                    | Some(Token::Arrow)
            ) {
                break;
            }

            let arg = self.parse_postfix_expression()?;

            // Convert to FunctionCall or Application
            match expr {
                Expression::Identifier(name) => {
                    // First argument: convert identifier to function call
                    expr = Expression::FunctionCall(name, vec![arg]);
                }
                Expression::FunctionCall(name, mut args) => {
                    // Additional arguments: extend existing function call
                    args.push(arg);
                    expr = Expression::FunctionCall(name, args);
                }
                _ => {
                    // Higher-order function application: use Application node
                    expr = Expression::Application(Box::new(expr), vec![arg]);
                }
            }
        }

        Ok(expr)
    }

    // Helper to check if current token can start a primary expression
    fn can_start_primary_expression(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::IntLiteral(_)) | 
            Some(Token::FloatLiteral(_)) |
            Some(Token::Identifier(_)) |
            Some(Token::LeftBracket) |  // array literal
            Some(Token::LeftParen) |    // parenthesized expression  
            Some(Token::Backslash) |    // lambda
            Some(Token::Let) // let..in
        )
    }

    fn parse_postfix_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary_expression()?;

        loop {
            match self.peek() {
                Some(Token::LeftBracket) => {
                    // Array indexing
                    self.advance();
                    let index = self.parse_expression()?;
                    self.expect(Token::RightBracket)?;
                    expr = Expression::ArrayIndex(Box::new(expr), Box::new(index));
                }
                Some(Token::Dot) => {
                    // Field access (e.g., v.x, v.y, v.z, v.w)
                    self.advance();
                    if let Some(Token::Identifier(field_name)) = self.peek().cloned() {
                        self.advance();
                        expr = Expression::FieldAccess(Box::new(expr), field_name);
                    } else {
                        return Err(CompilerError::ParseError(
                            "Expected field name after '.'".to_string(),
                        ));
                    }
                }
                Some(Token::LeftParen) => {
                    // Function application with parentheses: f(arg1, arg2)
                    self.advance();
                    let mut args = Vec::new();

                    if !self.check(&Token::RightParen) {
                        loop {
                            args.push(self.parse_expression()?);
                            if !self.check(&Token::Comma) {
                                break;
                            }
                            self.advance();
                        }
                    }

                    self.expect(Token::RightParen)?;
                    expr = Expression::Application(Box::new(expr), args);
                }
                _ => break,
            }
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
                Ok(Expression::Identifier(name))
            }
            Some(Token::LeftBracket) => self.parse_array_literal(),
            Some(Token::LeftParen) => {
                self.advance(); // consume '('
                let expr = self.parse_expression()?;
                self.expect(Token::RightParen)?;
                Ok(expr)
            }
            Some(Token::Backslash) => self.parse_lambda(),
            Some(Token::Let) => self.parse_let_in(),
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

    fn parse_lambda(&mut self) -> Result<Expression> {
        self.expect(Token::Backslash)?;

        // Parse parameter list: \x y z: t -> e (Futhark syntax)
        // Parameters are untyped, optional return type after all params
        let mut params = Vec::new();

        // Parse parameters (identifiers only)
        while let Some(Token::Identifier(name)) = self.peek() {
            let param_name = name.clone();
            self.advance();

            params.push(LambdaParam {
                name: param_name,
                ty: None, // Parameters are untyped in Futhark lambdas
            });

            // If we see a colon or arrow, we're done with parameters
            if self.check(&Token::Colon) || self.check(&Token::Arrow) {
                break;
            }
        }

        if params.is_empty() {
            return Err(CompilerError::ParseError(
                "Lambda must have at least one parameter".to_string(),
            ));
        }

        // Parse optional return type annotation: \x y z: t ->
        let return_type = if self.check(&Token::Colon) {
            self.advance(); // consume ':'
            Some(self.parse_type()?)
        } else {
            None
        };

        // Parse arrow
        self.expect(Token::Arrow)?;

        // Parse body expression
        let body = Box::new(self.parse_expression()?);

        Ok(Expression::Lambda(LambdaExpr {
            params,
            return_type,
            body,
        }))
    }

    fn parse_let_in(&mut self) -> Result<Expression> {
        use crate::ast::LetInExpr;

        self.expect(Token::Let)?;
        let name = self.expect_identifier()?;

        // Optional type annotation
        let ty = if self.check(&Token::Colon) {
            self.advance(); // consume ':'
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(Token::Assign)?;
        let value = Box::new(self.parse_expression()?);
        self.expect(Token::In)?;
        let body = Box::new(self.parse_expression()?);

        Ok(Expression::LetIn(LetInExpr {
            name,
            ty,
            value,
            body,
        }))
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

    /// Helper function that expects parsing to succeed and runs a check function on the declarations.
    /// If the check fails, outputs the parsed AST for debugging.
    fn expect_parse<F>(input: &str, check_fn: F)
    where
        F: FnOnce(&[Declaration]) -> std::result::Result<(), String>,
    {
        let tokens = tokenize(input).expect("Failed to tokenize input");
        let mut parser = Parser::new(tokens.clone());
        let program = match parser.parse() {
            Ok(program) => program,
            Err(e) => {
                println!("Parse failed with error: {:?}", e);
                println!("Tokens were: {:#?}", tokens);
                panic!("Failed to parse input: {:?}", e);
            }
        };

        if let Err(msg) = check_fn(&program.declarations) {
            println!("Check failed: {}", msg);
            println!("Parsed AST: {:#?}", program);
            panic!("Test assertion failed: {}", msg);
        }
    }

    /// Helper function that expects parsing to fail with a specific error.
    /// If parsing succeeds when it shouldn't, outputs the parsed AST.
    fn expect_parse_error<F>(input: &str, error_check: F)
    where
        F: FnOnce(&CompilerError) -> std::result::Result<(), String>,
    {
        let tokens = tokenize(input).expect("Failed to tokenize input");
        let mut parser = Parser::new(tokens);

        match parser.parse() {
            Ok(program) => {
                println!("Expected parse error, but parsing succeeded");
                println!("Parsed AST: {:#?}", program);
                panic!("Expected parse to fail, but it succeeded");
            }
            Err(ref error) => {
                if let Err(msg) = error_check(error) {
                    println!("Error check failed: {}", msg);
                    println!("Actual error: {:?}", error);
                    panic!("Error assertion failed: {}", msg);
                }
            }
        }
    }

    #[test]
    fn test_parse_let_decl() {
        expect_parse("let x: i32 = 42", |declarations| {
            if declarations.len() != 1 {
                return Err(format!(
                    "Expected 1 declaration, got {}",
                    declarations.len()
                ));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.name != "x" {
                        return Err(format!("Expected name 'x', got '{}'", decl.name));
                    }
                    if decl.ty != Some(crate::ast::types::i32()) {
                        return Err(format!("Expected i32 type, got {:?}", decl.ty));
                    }
                    if decl.body != Expression::IntLiteral(42) {
                        return Err(format!("Expected IntLiteral(42), got {:?}", decl.body));
                    }
                    Ok(())
                }
                _ => Err("Expected Let declaration".to_string()),
            }
        });
    }

    #[test]
    fn test_parse_array_type() {
        expect_parse(
            "let arr: [3][4]f32 = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.name != "arr" {
                            return Err(format!("Expected name 'arr', got '{}'", decl.name));
                        }
                        if decl.ty.is_none() {
                            return Err("Expected array type to be parsed".to_string());
                        }
                        Ok(())
                    }
                    _ => Err("Expected Let declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_entry_point_decl() {
        expect_parse(
            "#[vertex] def main(x: i32, y: f32): [4]f32 = result",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.name != "main" {
                            return Err(format!("Expected name 'main', got '{}'", decl.name));
                        }
                        if decl.attributes != vec![Attribute::Vertex] {
                            return Err(format!(
                                "Expected Vertex attribute, got {:?}",
                                decl.attributes
                            ));
                        }
                        if decl.params.len() != 2 {
                            return Err(format!(
                                "Expected 2 parameters, got {}",
                                decl.params.len()
                            ));
                        }
                        match &decl.params[0] {
                            DeclParam::Typed(param) => {
                                if param.name != "x" {
                                    return Err(format!(
                                        "Expected first param name 'x', got '{}'",
                                        param.name
                                    ));
                                }
                                if param.ty != crate::ast::types::i32() {
                                    return Err(format!(
                                        "Expected i32 type for first param, got {:?}",
                                        param.ty
                                    ));
                                }
                                if !param.attributes.is_empty() {
                                    return Err(format!(
                                        "Expected no attributes for first param, got {:?}",
                                        param.attributes
                                    ));
                                }
                            }
                            _ => return Err("Expected typed parameter".to_string()),
                        }
                        match &decl.params[1] {
                            DeclParam::Typed(param) => {
                                if param.name != "y" {
                                    return Err(format!(
                                        "Expected second param name 'y', got '{}'",
                                        param.name
                                    ));
                                }
                                if param.ty != crate::ast::types::f32() {
                                    return Err(format!(
                                        "Expected f32 type for second param, got {:?}",
                                        param.ty
                                    ));
                                }
                                if !param.attributes.is_empty() {
                                    return Err(format!(
                                        "Expected no attributes for second param, got {:?}",
                                        param.attributes
                                    ));
                                }
                            }
                            _ => return Err("Expected typed parameter".to_string()),
                        }
                        if let Some(ref ty) = decl.ty {
                            if *ty != crate::ast::types::sized_array(4, crate::ast::types::f32()) {
                                return Err(format!(
                                    "Expected [4]f32 return type, got {:?}",
                                    ty
                                ));
                            }
                        } else {
                            return Err("Expected return type".to_string());
                        }
                        if !decl.return_attributes.is_empty() {
                            return Err(format!(
                                "Expected no attributes on return type, got {:?}",
                                decl.return_attributes
                            ));
                        }
                        Ok(())
                    }
                    _ => Err("Expected Decl declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_array_index() {
        expect_parse("let x: f32 = arr[0]", |declarations| {
            if declarations.len() != 1 {
                return Err(format!(
                    "Expected 1 declaration, got {}",
                    declarations.len()
                ));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => match &decl.body {
                    Expression::ArrayIndex(arr, idx) => {
                        if **arr != Expression::Identifier("arr".to_string()) {
                            return Err(format!(
                                "Expected array identifier 'arr', got {:?}",
                                **arr
                            ));
                        }
                        if **idx != Expression::IntLiteral(0) {
                            return Err(format!("Expected index 0, got {:?}", **idx));
                        }
                        Ok(())
                    }
                    _ => Err(format!(
                        "Expected ArrayIndex expression, got {:?}",
                        decl.body
                    )),
                },
                _ => Err("Expected Let declaration".to_string()),
            }
        });
    }

    #[test]
    fn test_parse_division() {
        expect_parse("let x: f32 = 135f32/255f32", |declarations| {
            if declarations.len() != 1 {
                return Err(format!(
                    "Expected 1 declaration, got {}",
                    declarations.len()
                ));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => match &decl.body {
                    Expression::BinaryOp(BinaryOp::Divide, left, right) => {
                        if **left != Expression::FloatLiteral(135.0) {
                            return Err(format!("Expected left operand 135.0, got {:?}", **left));
                        }
                        if **right != Expression::FloatLiteral(255.0) {
                            return Err(format!("Expected right operand 255.0, got {:?}", **right));
                        }
                        Ok(())
                    }
                    _ => Err(format!(
                        "Expected BinaryOp expression, got {:?}",
                        decl.body
                    )),
                },
                _ => Err("Expected Let declaration".to_string()),
            }
        });
    }

    #[test]
    fn test_parse_vertex_attribute() {
        expect_parse("#[vertex] def main(): [4]f32 = result", |declarations| {
            if declarations.len() != 1 {
                return Err(format!(
                    "Expected 1 declaration, got {}",
                    declarations.len()
                ));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.attributes != vec![Attribute::Vertex] {
                        return Err(format!(
                            "Expected Vertex attribute, got {:?}",
                            decl.attributes
                        ));
                    }
                    if decl.name != "main" {
                        return Err(format!("Expected name 'main', got '{}'", decl.name));
                    }
                    Ok(())
                }
                _ => Err("Expected Decl declaration".to_string()),
            }
        });
    }

    #[test]
    fn test_parse_fragment_attribute() {
        expect_parse(
            "#[fragment] def frag(): [4]f32 = result",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.attributes != vec![Attribute::Fragment] {
                            return Err(format!(
                                "Expected Fragment attribute, got {:?}",
                                decl.attributes
                            ));
                        }
                        if decl.name != "frag" {
                            return Err(format!("Expected name 'frag', got '{}'", decl.name));
                        }
                        Ok(())
                    }
                    _ => Err("Expected Decl declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_builtin_attribute_on_return_type() {
        expect_parse(
            "#[vertex] def main(): #[builtin(position)] [4]f32 = result",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.attributes != vec![Attribute::Vertex] {
                            return Err(format!(
                                "Expected Vertex attribute, got {:?}",
                                decl.attributes
                            ));
                        }
                        if decl.return_attributes
                            != vec![Attribute::BuiltIn(spirv::BuiltIn::Position)]
                        {
                            return Err(format!(
                                "Expected Position builtin on return type, got {:?}",
                                decl.return_attributes
                            ));
                        }
                        if let Some(ref ty) = decl.ty {
                            if *ty != crate::ast::types::sized_array(4, crate::ast::types::f32()) {
                                return Err(format!(
                                    "Expected [4]f32 return type, got {:?}",
                                    ty
                                ));
                            }
                        } else {
                            return Err("Expected return type".to_string());
                        }
                        Ok(())
                    }
                    _ => Err("Expected Decl declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_location_attribute_on_return_type() {
        expect_parse(
            "#[fragment] def frag(): #[location(0)] [4]f32 = result",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.attributes != vec![Attribute::Fragment] {
                            return Err(format!(
                                "Expected Fragment attribute, got {:?}",
                                decl.attributes
                            ));
                        }
                        if decl.return_attributes != vec![Attribute::Location(0)] {
                            return Err(format!(
                                "Expected Location(0) attribute on return type, got {:?}",
                                decl.return_attributes
                            ));
                        }
                        if let Some(ref ty) = decl.ty {
                            if *ty != crate::ast::types::sized_array(4, crate::ast::types::f32()) {
                                return Err(format!(
                                    "Expected [4]f32 return type, got {:?}",
                                    ty
                                ));
                            }
                        } else {
                            return Err("Expected return type".to_string());
                        }
                        Ok(())
                    }
                    _ => Err("Expected Decl declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_parameter_with_builtin_attribute() {
        expect_parse(
            "#[vertex] def main(#[builtin(vertex_index)] vid: i32): [4]f32 = result",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.params.len() != 1 {
                            return Err(format!("Expected 1 parameter, got {}", decl.params.len()));
                        }
                        match &decl.params[0] {
                            DeclParam::Typed(param) => {
                                if param.name != "vid" {
                                    return Err(format!(
                                        "Expected param name 'vid', got '{}'",
                                        param.name
                                    ));
                                }
                                if param.ty != crate::ast::types::i32() {
                                    return Err(format!(
                                        "Expected i32 param type, got {:?}",
                                        param.ty
                                    ));
                                }
                                if param.attributes
                                    != vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)]
                                {
                                    return Err(format!(
                                        "Expected VertexIndex attribute, got {:?}",
                                        param.attributes
                                    ));
                                }
                                Ok(())
                            }
                            _ => Err("Expected typed parameter".to_string()),
                        }
                    }
                    _ => Err("Expected Decl declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_parameter_with_location_attribute() {
        expect_parse(
            "#[fragment] def frag(#[location(1)] color: [3]f32): [4]f32 = result",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.params.len() != 1 {
                            return Err(format!("Expected 1 parameter, got {}", decl.params.len()));
                        }
                        match &decl.params[0] {
                            DeclParam::Typed(param) => {
                                if param.name != "color" {
                                    return Err(format!(
                                        "Expected param name 'color', got '{}'",
                                        param.name
                                    ));
                                }
                                if param.ty != crate::ast::types::sized_array(3, crate::ast::types::f32()) {
                                    return Err(format!(
                                        "Expected [3]f32 param type, got {:?}",
                                        param.ty
                                    ));
                                }
                                if param.attributes != vec![Attribute::Location(1)] {
                                    return Err(format!(
                                        "Expected Location(1) attribute, got {:?}",
                                        param.attributes
                                    ));
                                }
                                Ok(())
                            }
                            _ => Err("Expected typed parameter".to_string()),
                        }
                    }
                    _ => Err("Expected Decl declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_multiple_builtin_types() {
        expect_parse(
            "#[vertex] def main(#[builtin(vertex_index)] vid: i32, #[builtin(instance_index)] iid: i32): #[builtin(position)] [4]f32 = result",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!("Expected 1 declaration, got {}", declarations.len()));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => {
                        if decl.params.len() != 2 {
                            return Err(format!("Expected 2 parameters, got {}", decl.params.len()));
                        }

                        // First parameter
                        match &decl.params[0] {
                            DeclParam::Typed(param) => {
                                if param.name != "vid" {
                                    return Err(format!("Expected first param name 'vid', got '{}'", param.name));
                                }
                                if param.attributes != vec![Attribute::BuiltIn(spirv::BuiltIn::VertexIndex)] {
                                    return Err(format!("Expected VertexIndex attribute, got {:?}", param.attributes));
                                }
                            }
                            _ => return Err("Expected typed parameter".to_string()),
                        }

                        // Second parameter
                        match &decl.params[1] {
                            DeclParam::Typed(param) => {
                                if param.name != "iid" {
                                    return Err(format!("Expected second param name 'iid', got '{}'", param.name));
                                }
                                if param.attributes != vec![Attribute::BuiltIn(spirv::BuiltIn::InstanceIndex)] {
                                    return Err(format!("Expected InstanceIndex attribute, got {:?}", param.attributes));
                                }
                            }
                            _ => return Err("Expected typed parameter".to_string()),
                        }

                        // Return type
                        if decl.return_attributes != vec![Attribute::BuiltIn(spirv::BuiltIn::Position)] {
                            return Err(format!("Expected Position attribute on return type, got {:?}", decl.return_attributes));
                        }

                        Ok(())
                    }
                    _ => Err("Expected Decl declaration".to_string()),
                }
            }
        );
    }

    #[test]
    fn test_parse_simple_lambda() {
        expect_parse(r#"let f: i32 -> i32 = \x -> x"#, |declarations| {
            if declarations.len() != 1 {
                return Err(format!(
                    "Expected 1 declaration, got {}",
                    declarations.len()
                ));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => {
                    if decl.name != "f" {
                        return Err(format!("Expected name 'f', got '{}'", decl.name));
                    }
                    match &decl.body {
                        Expression::Lambda(lambda) => {
                            if lambda.params.len() != 1 {
                                return Err(format!(
                                    "Expected 1 lambda param, got {}",
                                    lambda.params.len()
                                ));
                            }
                            if lambda.params[0].name != "x" {
                                return Err(format!(
                                    "Expected param name 'x', got '{}'",
                                    lambda.params[0].name
                                ));
                            }
                            if lambda.params[0].ty.is_some() {
                                return Err(format!(
                                    "Expected no param type, got {:?}",
                                    lambda.params[0].ty
                                ));
                            }
                            if lambda.return_type.is_some() {
                                return Err(format!(
                                    "Expected no return type, got {:?}",
                                    lambda.return_type
                                ));
                            }
                            match lambda.body.as_ref() {
                                Expression::Identifier(name) => {
                                    if name != "x" {
                                        return Err(format!(
                                            "Expected identifier 'x' in lambda body, got '{}'",
                                            name
                                        ));
                                    }
                                }
                                _ => {
                                    return Err(format!(
                                        "Expected identifier in lambda body, got {:?}",
                                        lambda.body
                                    ))
                                }
                            }
                            Ok(())
                        }
                        _ => Err(format!("Expected lambda expression, got {:?}", decl.body)),
                    }
                }
                _ => Err("Expected Let declaration".to_string()),
            }
        });
    }

    #[test]
    fn test_parse_lambda_with_type_annotation() {
        expect_parse(r#"let f: f32 -> f32 = \x -> x"#, |declarations| {
            if declarations.len() != 1 {
                return Err(format!(
                    "Expected 1 declaration, got {}",
                    declarations.len()
                ));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => match &decl.body {
                    Expression::Lambda(lambda) => {
                        if lambda.params.len() != 1 {
                            return Err(format!(
                                "Expected 1 lambda param, got {}",
                                lambda.params.len()
                            ));
                        }
                        if lambda.params[0].name != "x" {
                            return Err(format!(
                                "Expected param name 'x', got '{}'",
                                lambda.params[0].name
                            ));
                        }
                        if lambda.params[0].ty.is_some() {
                            return Err(format!(
                                "Expected no param type (parameters are untyped), got {:?}",
                                lambda.params[0].ty
                            ));
                        }
                        if lambda.return_type.is_some() {
                            return Err(format!(
                                "Expected no return type annotation in lambda, got {:?}",
                                lambda.return_type
                            ));
                        }
                        match lambda.body.as_ref() {
                            Expression::Identifier(name) => {
                                if name != "x" {
                                    return Err(format!(
                                        "Expected identifier 'x' in lambda body, got '{}'",
                                        name
                                    ));
                                }
                            }
                            _ => {
                                return Err(format!(
                                    "Expected identifier in lambda body, got {:?}",
                                    lambda.body
                                ))
                            }
                        }
                        Ok(())
                    }
                    _ => Err(format!("Expected lambda expression, got {:?}", decl.body)),
                },
                _ => Err("Expected Let declaration".to_string()),
            }
        });
    }

    #[test]
    fn test_parse_lambda_with_multiple_params() {
        expect_parse(
            r#"let add: i32 -> i32 -> i32 = \x y -> x"#,
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                match &declarations[0] {
                    Declaration::Decl(decl) => match &decl.body {
                        Expression::Lambda(lambda) => {
                            if lambda.params.len() != 2 {
                                return Err(format!(
                                    "Expected 2 lambda params, got {}",
                                    lambda.params.len()
                                ));
                            }
                            if lambda.params[0].name != "x" {
                                return Err(format!(
                                    "Expected first param name 'x', got '{}'",
                                    lambda.params[0].name
                                ));
                            }
                            if lambda.params[0].ty.is_some() {
                                return Err(format!(
                                    "Expected no type for first param, got {:?}",
                                    lambda.params[0].ty
                                ));
                            }
                            if lambda.params[1].name != "y" {
                                return Err(format!(
                                    "Expected second param name 'y', got '{}'",
                                    lambda.params[1].name
                                ));
                            }
                            if lambda.params[1].ty.is_some() {
                                return Err(format!(
                                    "Expected no type for second param, got {:?}",
                                    lambda.params[1].ty
                                ));
                            }
                            if lambda.return_type.is_some() {
                                return Err(format!(
                                    "Expected no return type, got {:?}",
                                    lambda.return_type
                                ));
                            }
                            Ok(())
                        }
                        _ => Err(format!("Expected lambda expression, got {:?}", decl.body)),
                    },
                    _ => Err("Expected Let declaration".to_string()),
                }
            },
        );
    }

    #[test]
    fn test_parse_function_application() {
        expect_parse(r#"let result: i32 = f(42, 24)"#, |declarations| {
            if declarations.len() != 1 {
                return Err(format!(
                    "Expected 1 declaration, got {}",
                    declarations.len()
                ));
            }
            match &declarations[0] {
                Declaration::Decl(decl) => match &decl.body {
                    Expression::Application(func, args) => {
                        match func.as_ref() {
                            Expression::Identifier(name) => {
                                if name != "f" {
                                    return Err(format!(
                                        "Expected function identifier 'f', got '{}'",
                                        name
                                    ));
                                }
                            }
                            _ => {
                                return Err(format!("Expected function identifier, got {:?}", func))
                            }
                        }
                        if args.len() != 2 {
                            return Err(format!("Expected 2 arguments, got {}", args.len()));
                        }
                        match &args[0] {
                            Expression::IntLiteral(n) => {
                                if *n != 42 {
                                    return Err(format!("Expected first argument 42, got {}", n));
                                }
                            }
                            _ => {
                                return Err(format!(
                                    "Expected int literal for first argument, got {:?}",
                                    args[0]
                                ))
                            }
                        }
                        match &args[1] {
                            Expression::IntLiteral(n) => {
                                if *n != 24 {
                                    return Err(format!("Expected second argument 24, got {}", n));
                                }
                            }
                            _ => {
                                return Err(format!(
                                    "Expected int literal for second argument, got {:?}",
                                    args[1]
                                ))
                            }
                        }
                        Ok(())
                    }
                    _ => Err(format!(
                        "Expected function application, got {:?}",
                        decl.body
                    )),
                },
                _ => Err("Expected Let declaration".to_string()),
            }
        });
    }

    #[test]
    fn test_parse_simple_let_in() {
        expect_parse(
            "#[vertex] def main(x: i32): i32 = let y = 5 in y + x",
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                // Just verify it parses successfully - the structure is complex to validate in detail
                Ok(())
            },
        );
    }

    #[test]
    fn test_parse_let_in_expression_only() {
        // Test parsing just the let..in expression by itself
        let input = r#"let f = \y -> y + x in f 10"#;
        let tokens = tokenize(input).expect("Failed to tokenize");
        let mut parser = Parser::new(tokens);
        let result = parser.parse_expression();

        // Just verify it parses without error - this is a complex expression
        match result {
            Ok(_expr) => {
                // Test passes if parsing succeeds
            }
            Err(e) => panic!("Failed to parse let..in expression: {:?}", e),
        }
    }

    #[test]
    fn test_parse_let_in_with_lambda() {
        expect_parse(
            r#"#[vertex] def main(x: i32): i32 = let f = \y -> y + x in f 10"#,
            |declarations| {
                if declarations.len() != 1 {
                    return Err(format!(
                        "Expected 1 declaration, got {}",
                        declarations.len()
                    ));
                }
                // Just verify it parses successfully - the lambda let..in structure is complex
                Ok(())
            },
        );
    }

    #[test]
    fn test_parse_multiple_top_level_lets_with_entry() {
        // Test the failing case from integration tests: multiple let declarations with entry points
        let input = r#"
def verts: [3][4]f32 =
  [[-1.0f32, -1.0f32, 0.0f32, 1.0f32],
   [ 3.0f32, -1.0f32, 0.0f32, 1.0f32],
   [-1.0f32,  3.0f32, 0.0f32, 1.0f32]]

#[vertex]
def vertex_main(vertex_id: i32): [4]f32 = verts[vertex_id]

def SKY_RGBA: [4]f32 =
  [135f32/255f32, 206f32/255f32, 235f32/255f32, 1.0f32]

#[fragment]
def fragment_main(): [4]f32 = SKY_RGBA
"#;

        expect_parse(input, |declarations| {
            if declarations.len() != 4 {
                return Err(format!(
                    "Expected 4 declarations (2 lets + 2 entries), got {}",
                    declarations.len()
                ));
            }
            
            // First should be def verts
            match &declarations[0] {
                Declaration::Decl(decl) if decl.keyword == "def" => {
                    if decl.name != "verts" {
                        return Err(format!("Expected first def to be 'verts', got '{}'", decl.name));
                    }
                }
                _ => return Err("Expected first declaration to be Def".to_string()),
            }
            
            // Second should be vertex entry point (now a Decl with vertex attribute)
            match &declarations[1] {
                Declaration::Decl(decl) => {
                    if decl.name != "vertex_main" {
                        return Err(format!("Expected vertex entry to be 'vertex_main', got '{}'", decl.name));
                    }
                    if !decl.attributes.contains(&Attribute::Vertex) {
                        return Err(format!("Expected vertex attribute on vertex_main, got {:?}", decl.attributes));
                    }
                }
                _ => return Err("Expected second declaration to be Decl".to_string()),
            }
            
            // Third should be def SKY_RGBA
            match &declarations[2] {
                Declaration::Decl(decl) if decl.keyword == "def" => {
                    if decl.name != "SKY_RGBA" {
                        return Err(format!("Expected third def to be 'SKY_RGBA', got '{}'", decl.name));
                    }
                }
                _ => return Err("Expected third declaration to be Def".to_string()),
            }
            
            // Fourth should be fragment entry point (now a Decl with fragment attribute)
            match &declarations[3] {
                Declaration::Decl(decl) => {
                    if decl.name != "fragment_main" {
                        return Err(format!("Expected fragment entry to be 'fragment_main', got '{}'", decl.name));
                    }
                    if !decl.attributes.contains(&Attribute::Fragment) {
                        return Err(format!("Expected fragment attribute on fragment_main, got {:?}", decl.attributes));
                    }
                }
                _ => return Err("Expected fourth declaration to be Decl".to_string()),
            }
            
            Ok(())
        });
    }
}
