use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::lexer::Token;
use log::trace;

mod module;
mod pattern;
#[cfg(test)]
mod tests;

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    node_counter: NodeCounter,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            current: 0,
            node_counter: NodeCounter::new(),
        }
    }

    pub fn new_with_counter(tokens: Vec<Token>, node_counter: NodeCounter) -> Self {
        Parser {
            tokens,
            current: 0,
            node_counter,
        }
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut declarations = Vec::new();

        while !self.is_at_end() {
            declarations.push(self.parse_declaration()?);
        }

        Ok(Program { declarations })
    }

    fn parse_declaration(&mut self) -> Result<Declaration> {
        trace!("parse_declaration: next token = {:?}", self.peek());
        // Parse optional attributes
        let attributes = self.parse_attributes()?;

        match self.peek() {
            Some(Token::Let) => self.parse_decl("let", attributes),
            Some(Token::Def) => self.parse_decl("def", attributes),
            Some(Token::Val) => {
                let mut decl = self.parse_val_decl()?;
                decl.attributes = attributes;
                Ok(Declaration::Val(decl))
            }
            Some(Token::Type) => {
                let type_bind = self.parse_type_bind()?;
                Ok(Declaration::TypeBind(type_bind))
            }
            Some(Token::Module) => {
                // Check if it's "module type" or just "module"
                let saved_pos = self.current;
                self.advance();
                if self.check(&Token::Type) {
                    // module type declaration
                    self.current = saved_pos;
                    let mod_type_bind = self.parse_module_type_bind()?;
                    Ok(Declaration::ModuleTypeBind(mod_type_bind))
                } else {
                    // module declaration
                    self.current = saved_pos;
                    let mod_bind = self.parse_module_bind()?;
                    Ok(Declaration::ModuleBind(mod_bind))
                }
            }
            Some(Token::Open) => {
                self.advance();
                let mod_exp = self.parse_module_expression()?;
                Ok(Declaration::Open(mod_exp))
            }
            Some(Token::Import) => {
                self.advance();
                let path = self.expect_string_literal()?;
                Ok(Declaration::Import(path))
            }
            Some(Token::Local) => {
                self.advance();
                let inner = self.parse_declaration()?;
                Ok(Declaration::Local(Box::new(inner)))
            }
            _ => Err(CompilerError::ParseError(format!(
                "Expected declaration, got {:?}",
                self.peek()
            ))),
        }
    }

    fn expect_string_literal(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::StringLiteral(s)) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(CompilerError::ParseError("Expected string literal".to_string())),
        }
    }

    fn parse_decl(&mut self, keyword: &'static str, attributes: Vec<Attribute>) -> Result<Declaration> {
        trace!("parse_decl({}): next token = {:?}", keyword, self.peek());
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
            let (ty, attributed_return_type) = self.parse_return_type()?;

            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;

            Ok(Declaration::Decl(Decl {
                keyword,
                attributes,
                name,
                params,
                // For functions with parameters, don't store attributed_tuple in ty
                // The type checker will build the full function type from params and body
                ty: if attributed_return_type.is_some() { None } else { ty },
                return_attributes: vec![], // No separate return attributes for now
                attributed_return_type,
                body,
            }))
        }
        // Check if this is a typed declaration (name: type = value or name: type for uniforms)
        else if self.check(&Token::Colon) {
            // Typed declaration: let/def name: type = value or let/def name: type (uniform)
            self.expect(Token::Colon)?;

            // Check if this is an attributed return type
            let (ty, attributed_return_type) = if self.check(&Token::AttributeStart) {
                self.parse_return_type()?
            } else {
                (Some(self.parse_type()?), None)
            };

            // Check if this is a uniform declaration (no initializer allowed)
            let has_uniform_attr = attributes.iter().any(|attr| matches!(attr, Attribute::Uniform));

            if has_uniform_attr {
                // Uniforms don't have initializers
                if self.check(&Token::Assign) {
                    return Err(CompilerError::ParseError(
                        "Uniform declarations cannot have initializer values".to_string(),
                    ));
                }
                // Return UniformDecl
                Ok(Declaration::Uniform(UniformDecl {
                    name,
                    ty: ty.ok_or_else(|| {
                        CompilerError::ParseError("Uniform must have explicit type annotation".to_string())
                    })?,
                }))
            } else {
                // Regular typed declaration requires an initializer
                self.expect(Token::Assign)?;
                let body = self.parse_expression()?;

                Ok(Declaration::Decl(Decl {
                    keyword,
                    attributes,
                    name,
                    params: vec![], // No parameters for typed declarations
                    ty,
                    return_attributes: vec![],
                    attributed_return_type,
                    body,
                }))
            }
        } else {
            // Function declaration: let/def name param1 param2 = body
            // OR simple variable: let/def name = value
            let mut params = Vec::new();
            while !self.check(&Token::Assign) && !self.is_at_end() {
                params.push(DeclParam::Untyped(self.expect_identifier()?));
            }
            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;

            Ok(Declaration::Decl(Decl {
                keyword,
                attributes,
                name,
                params,
                ty: None, // No explicit type for untyped declarations
                return_attributes: vec![],
                attributed_return_type: None,
                body,
            }))
        }
    }

    fn parse_val_decl(&mut self) -> Result<ValDecl> {
        trace!("parse_val_decl: next token = {:?}", self.peek());
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

    fn parse_attributed_type(&mut self) -> Result<AttributedType> {
        trace!("parse_attributed_type: next token = {:?}", self.peek());
        // Parse optional #[attribute] syntax
        let attributes = if self.check(&Token::AttributeStart) {
            self.advance(); // consume '#['
            let attribute = self.parse_attribute()?;
            vec![attribute]
        } else {
            vec![]
        };

        // Parse the type
        let ty = self.parse_type()?;

        Ok(AttributedType { attributes, ty })
    }

    fn parse_return_type(&mut self) -> Result<(Option<Type>, Option<Vec<AttributedType>>)> {
        trace!("parse_return_type: next token = {:?}", self.peek());
        // Check if it's an attributed tuple: ([attr1] type1, [attr2] type2, ...)
        if self.check(&Token::LeftParen) {
            self.advance(); // consume '('
            let mut attributed_types = Vec::new();

            if !self.check(&Token::RightParen) {
                loop {
                    let attributed_type = self.parse_attributed_type()?;
                    attributed_types.push(attributed_type);

                    if !self.check(&Token::Comma) {
                        break;
                    }
                    self.advance(); // consume ','
                }
            }

            self.expect(Token::RightParen)?;

            // Create a tuple type from the attributed types for the type system
            let tuple_type = types::attributed_tuple(attributed_types.clone());
            Ok((Some(tuple_type), Some(attributed_types)))
        } else if self.check(&Token::AttributeStart) {
            // Single attributed type: #[attribute] type
            let attributed_type = self.parse_attributed_type()?;

            // Return as a single-element attributed tuple
            let attributed_types = vec![attributed_type];
            let tuple_type = types::attributed_tuple(attributed_types.clone());
            Ok((Some(tuple_type), Some(attributed_types)))
        } else {
            // Regular single return type without attributes
            let ty = self.parse_type()?;
            Ok((Some(ty), None))
        }
    }

    fn parse_attribute(&mut self) -> Result<Attribute> {
        trace!("parse_attribute: next token = {:?}", self.peek());
        let attr_name = self.expect_identifier()?;

        match attr_name.as_str() {
            "vertex" => {
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Vertex)
            }
            "fragment" => {
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Fragment)
            }
            "uniform" => {
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Uniform)
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
                        )));
                    }
                };
                Ok(Attribute::BuiltIn(builtin))
            }
            "location" => {
                self.expect(Token::LeftParen)?;
                let location = if let Some(Token::IntLiteral(location)) = self.advance() {
                    *location as u32
                } else {
                    return Err(CompilerError::ParseError("Expected location number".to_string()));
                };
                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Location(location))
            }
            _ => Err(CompilerError::ParseError(format!(
                "Unknown attribute: {}",
                attr_name
            ))),
        }
    }

    fn parse_attributes(&mut self) -> Result<Vec<Attribute>> {
        trace!("parse_attributes: next token = {:?}", self.peek());
        let mut attributes = Vec::new();

        while self.check(&Token::AttributeStart) {
            self.advance(); // consume '#['
            let attribute = self.parse_attribute()?;
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
        trace!("parse_type_variable: next token = {:?}", self.peek());
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
        trace!("parse_type: next token = {:?}", self.peek());
        self.parse_function_type()
    }

    fn parse_function_type(&mut self) -> Result<Type> {
        trace!("parse_function_type: next token = {:?}", self.peek());
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
        trace!("parse_array_or_base_type: next token = {:?}", self.peek());
        // Check for uniqueness prefix *
        if matches!(self.peek(), Some(Token::BinOp(op)) if op == "*") {
            self.advance(); // consume '*'
            let inner_type = self.parse_array_or_base_type()?;
            return Ok(types::unique(inner_type));
        }

        // Check for array type [dim]baseType (Futhark style)
        // Accept both LeftBracket and LeftBracketSpaced in type position
        if self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
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
        trace!("parse_base_type: next token = {:?}", self.peek());
        match self.peek() {
            Some(Token::Identifier(name)) if name == "i32" => {
                self.advance();
                Ok(types::i32())
            }
            Some(Token::Identifier(name)) if name == "f32" => {
                self.advance();
                Ok(types::f32())
            }
            // Vector types
            Some(Token::Identifier(name)) if name == "vec2" => {
                self.advance();
                Ok(types::vec2())
            }
            Some(Token::Identifier(name)) if name == "vec3" => {
                self.advance();
                Ok(types::vec3())
            }
            Some(Token::Identifier(name)) if name == "vec4" => {
                self.advance();
                Ok(types::vec4())
            }
            Some(Token::Identifier(name)) if name == "ivec2" => {
                self.advance();
                Ok(types::ivec2())
            }
            Some(Token::Identifier(name)) if name == "ivec3" => {
                self.advance();
                Ok(types::ivec3())
            }
            Some(Token::Identifier(name)) if name == "ivec4" => {
                self.advance();
                Ok(types::ivec4())
            }
            // Matrix types
            Some(Token::Identifier(name)) if name == "mat2" => {
                self.advance();
                Ok(types::mat2())
            }
            Some(Token::Identifier(name)) if name == "mat3" => {
                self.advance();
                Ok(types::mat3())
            }
            Some(Token::Identifier(name)) if name == "mat4" => {
                self.advance();
                Ok(types::mat4())
            }
            Some(Token::Identifier(name)) if name == "mat2x3" => {
                self.advance();
                Ok(types::mat2x3())
            }
            Some(Token::Identifier(name)) if name == "mat2x4" => {
                self.advance();
                Ok(types::mat2x4())
            }
            Some(Token::Identifier(name)) if name == "mat3x2" => {
                self.advance();
                Ok(types::mat3x2())
            }
            Some(Token::Identifier(name)) if name == "mat3x4" => {
                self.advance();
                Ok(types::mat3x4())
            }
            Some(Token::Identifier(name)) if name == "mat4x2" => {
                self.advance();
                Ok(types::mat4x2())
            }
            Some(Token::Identifier(name)) if name == "mat4x3" => {
                self.advance();
                Ok(types::mat4x3())
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
        trace!("parse_expression: next token = {:?}", self.peek());
        self.parse_binary_expression()
    }

    fn parse_binary_expression(&mut self) -> Result<Expression> {
        trace!("parse_binary_expression: next token = {:?}", self.peek());
        self.parse_binary_expression_with_precedence(0)
    }

    fn get_operator_precedence(op: &str) -> Option<(i32, bool)> {
        // Returns (precedence, is_left_associative)
        // Higher precedence binds tighter
        match op {
            "*" | "/" => Some((3, true)), // Multiplication and division
            "+" | "-" => Some((2, true)), // Addition and subtraction
            "==" | "!=" | "<" | ">" | "<=" | ">=" => Some((1, true)), // Comparison operators
            _ => None,
        }
    }

    fn parse_binary_expression_with_precedence(&mut self, min_precedence: i32) -> Result<Expression> {
        trace!(
            "parse_binary_expression_with_precedence({}): next token = {:?}",
            min_precedence,
            self.peek()
        );
        let mut left = self.parse_application_expression()?;

        loop {
            // Check if we have a binary operator
            let op_string = match self.peek() {
                Some(Token::BinOp(op)) => op.clone(),
                _ => break,
            };

            // Get operator precedence
            let (precedence, is_left_assoc) = match Self::get_operator_precedence(&op_string) {
                Some(p) => p,
                None => break,
            };

            // Check if this operator has high enough precedence to be parsed here
            if precedence < min_precedence {
                break;
            }

            // Consume the operator
            self.advance();

            // Parse right side with appropriate precedence
            let next_min_precedence = if is_left_assoc {
                precedence + 1 // For left-associative, parse with higher precedence
            } else {
                precedence // For right-associative, parse with same precedence
            };

            let right = self.parse_binary_expression_with_precedence(next_min_precedence)?;

            // Build the binary operation
            left = self.node_counter.mk_node(ExprKind::BinaryOp(
                BinaryOp { op: op_string },
                Box::new(left),
                Box::new(right),
            ));
        }

        Ok(left)
    }

    // Function application has higher precedence than binary operators
    // Parses: f x y z, (f x) y, etc.
    fn parse_application_expression(&mut self) -> Result<Expression> {
        trace!("parse_application_expression: next token = {:?}", self.peek());
        let mut expr = self.parse_postfix_expression()?;

        // Keep collecting arguments while we see primary expressions
        // Stop at operators, commas, closing delimiters, etc.
        while self.peek().is_some() && self.can_start_primary_expression() {
            // Don't consume binary operators
            if matches!(self.peek(), Some(Token::BinOp(_))) {
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
            match expr.kind {
                ExprKind::Identifier(name) => {
                    // First argument: convert identifier to function call
                    expr = self.node_counter.mk_node(ExprKind::FunctionCall(name, vec![arg]));
                }
                ExprKind::FunctionCall(name, mut args) => {
                    // Additional arguments: extend existing function call
                    args.push(arg);
                    expr = self.node_counter.mk_node(ExprKind::FunctionCall(name, args));
                }
                _ => {
                    // Higher-order function application: use Application node
                    expr = self.node_counter.mk_node(ExprKind::Application(Box::new(expr), vec![arg]));
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
            Some(Token::LeftBracket) |  // array literal (no space) or array indexing context
            Some(Token::LeftBracketSpaced) |  // array literal (with space)
            Some(Token::LeftParen) |    // parenthesized expression
            Some(Token::Backslash) |    // lambda
            Some(Token::Let) // let..in
        )
    }

    fn parse_postfix_expression(&mut self) -> Result<Expression> {
        trace!("parse_postfix_expression: next token = {:?}", self.peek());
        let mut expr = self.parse_unary_expression()?;

        loop {
            match self.peek() {
                Some(Token::LeftBracket) => {
                    // Array indexing (no space before [): arr[0]
                    self.advance();
                    let index = self.parse_expression()?;
                    self.expect(Token::RightBracket)?;
                    expr = self.node_counter.mk_node(ExprKind::ArrayIndex(Box::new(expr), Box::new(index)));
                }
                Some(Token::LeftBracketSpaced) => {
                    // Space before [ means it's not array indexing, it's a new expression
                    // Stop postfix parsing and let the caller handle it
                    break;
                }
                Some(Token::Dot) => {
                    // Field access (e.g., v.x, v.y, v.z, v.w)
                    self.advance();
                    if let Some(Token::Identifier(field_name)) = self.peek().cloned() {
                        self.advance();
                        expr = self.node_counter.mk_node(ExprKind::FieldAccess(Box::new(expr), field_name));
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
                    expr = self.node_counter.mk_node(ExprKind::Application(Box::new(expr), args));
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        trace!("parse_unary_expression: next token = {:?}", self.peek());
        // Check for unary operators: - and !
        match self.peek() {
            Some(Token::BinOp(op)) if op == "-" => {
                self.advance();
                let operand = self.parse_unary_expression()?; // Right-associative
                Ok(self.node_counter.mk_node(ExprKind::UnaryOp(
                    UnaryOp { op: "-".to_string() },
                    Box::new(operand),
                )))
            }
            Some(Token::Bang) => {
                self.advance();
                let operand = self.parse_unary_expression()?; // Right-associative
                Ok(self.node_counter.mk_node(ExprKind::UnaryOp(
                    UnaryOp { op: "!".to_string() },
                    Box::new(operand),
                )))
            }
            _ => self.parse_primary_expression(),
        }
    }

    fn parse_primary_expression(&mut self) -> Result<Expression> {
        trace!("parse_primary_expression: next token = {:?}", self.peek());
        match self.peek() {
            Some(Token::IntLiteral(n)) => {
                let n = *n;
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::IntLiteral(n)))
            }
            Some(Token::FloatLiteral(f)) => {
                let f = *f;
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(f)))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::Identifier(name)))
            }
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => self.parse_array_literal(),
            Some(Token::LeftParen) => {
                self.advance(); // consume '('

                // Check for empty tuple
                if self.check(&Token::RightParen) {
                    self.advance();
                    return Ok(self.node_counter.mk_node(ExprKind::Tuple(vec![])));
                }

                // Parse first expression
                let first_expr = self.parse_expression()?;

                // Check if it's a tuple or just a parenthesized expression
                if self.check(&Token::Comma) {
                    // It's a tuple
                    let mut elements = vec![first_expr];
                    while self.check(&Token::Comma) {
                        self.advance(); // consume ','
                        if self.check(&Token::RightParen) {
                            break; // Allow trailing comma
                        }
                        elements.push(self.parse_expression()?);
                    }
                    self.expect(Token::RightParen)?;
                    Ok(self.node_counter.mk_node(ExprKind::Tuple(elements)))
                } else {
                    // Just a parenthesized expression
                    self.expect(Token::RightParen)?;
                    Ok(first_expr)
                }
            }
            Some(Token::Backslash) => self.parse_lambda(),
            Some(Token::Let) => self.parse_let_in(),
            Some(Token::If) => self.parse_if_then_else(),
            _ => Err(CompilerError::ParseError("Expected expression".to_string())),
        }
    }

    fn parse_array_literal(&mut self) -> Result<Expression> {
        trace!("parse_array_literal: next token = {:?}", self.peek());
        // Accept either LeftBracket or LeftBracketSpaced
        match self.peek() {
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => {
                self.advance();
            }
            _ => return Err(CompilerError::ParseError("Expected '['".to_string())),
        }

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
        Ok(self.node_counter.mk_node(ExprKind::ArrayLiteral(elements)))
    }

    fn parse_lambda(&mut self) -> Result<Expression> {
        trace!("parse_lambda: next token = {:?}", self.peek());
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

        Ok(self.node_counter.mk_node(ExprKind::Lambda(LambdaExpr {
            params,
            return_type,
            body,
        })))
    }

    fn parse_let_in(&mut self) -> Result<Expression> {
        trace!("parse_let_in: next token = {:?}", self.peek());
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

        Ok(self.node_counter.mk_node(ExprKind::LetIn(LetInExpr {
            name,
            ty,
            value,
            body,
        })))
    }

    fn parse_if_then_else(&mut self) -> Result<Expression> {
        trace!("parse_if_then_else: next token = {:?}", self.peek());
        use crate::ast::IfExpr;

        self.expect(Token::If)?;
        let condition = Box::new(self.parse_expression()?);
        self.expect(Token::Then)?;
        let then_branch = Box::new(self.parse_expression()?);
        self.expect(Token::Else)?;
        let else_branch = Box::new(self.parse_expression()?);

        Ok(self.node_counter.mk_node(ExprKind::If(IfExpr {
            condition,
            then_branch,
            else_branch,
        })))
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
