use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::lexer::{LocatedToken, Token};
use log::trace;

mod module;
mod pattern;
#[cfg(test)]
mod tests;

pub struct Parser {
    tokens: Vec<LocatedToken>,
    current: usize,
    node_counter: NodeCounter,
}

impl Parser {
    pub fn new(tokens: Vec<LocatedToken>) -> Self {
        Parser {
            tokens,
            current: 0,
            node_counter: NodeCounter::new(),
        }
    }

    pub fn new_with_counter(tokens: Vec<LocatedToken>, node_counter: NodeCounter) -> Self {
        Parser {
            tokens,
            current: 0,
            node_counter,
        }
    }

    /// Get the span of the current token
    fn current_span(&self) -> Span {
        self.tokens.get(self.current).map(|t| t.span).unwrap_or_else(Span::dummy)
    }

    /// Get the span of the previous token
    fn previous_span(&self) -> Span {
        if self.current > 0 {
            self.tokens.get(self.current - 1).map(|t| t.span).unwrap_or_else(Span::dummy)
        } else {
            Span::dummy()
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

        // Check for special attributes that require specific declaration types
        let entry_type =
            attributes.iter().find(|attr| matches!(attr, Attribute::Vertex | Attribute::Fragment));
        let is_uniform = attributes.iter().any(|attr| matches!(attr, Attribute::Uniform));

        if is_uniform {
            // Uniform declaration - delegate to helper
            if keyword != "def" {
                return Err(CompilerError::ParseError(
                    "Uniform declarations must use 'def', not 'let'".to_string(),
                ));
            }
            return self.parse_uniform_decl();
        } else if let Some(entry_attr) = entry_type {
            // Entry point: must be 'def', not 'let'
            if keyword != "def" {
                return Err(CompilerError::ParseError(
                    "Entry point declarations must use 'def', not 'let'".to_string(),
                ));
            }
            self.expect(Token::Def)?;

            let name = self.expect_identifier()?;

            // Parse parameters
            let mut params = Vec::new();
            while !self.check(&Token::Colon) && !self.check(&Token::Assign) && !self.is_at_end() {
                params.push(self.parse_function_parameter()?);
            }

            // Entry points must have an explicit return type
            if !self.check(&Token::Colon) {
                return Err(CompilerError::ParseError(
                    "Entry point declarations must have an explicit return type".to_string(),
                ));
            }

            self.advance(); // consume ':'

            // Parse return type (which may have optional attributes)
            let (return_types, return_attributes) = if self.check(&Token::AttributeStart) || self.check(&Token::LeftParen) {
                // Attributed return type(s)
                self.parse_return_type()?
            } else {
                // Simple unattributed return type
                let ty = self.parse_type()?;
                (vec![ty], vec![None])
            };

            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;

            Ok(Declaration::Entry(EntryDecl {
                entry_type: entry_attr.clone(),
                name,
                params,
                return_types,
                return_attributes,
                body,
            }))
        } else {
            // Regular declaration (let or def)
            match keyword {
                "let" => self.expect(Token::Let)?,
                "def" => self.expect(Token::Def)?,
                _ => return Err(CompilerError::ParseError(format!("Invalid keyword: {}", keyword))),
            }

            let name = self.expect_identifier()?;

            // Parse patterns until we hit : (return type) or = (body)
            let mut params = Vec::new();
            while !self.check(&Token::Colon) && !self.check(&Token::Assign) && !self.is_at_end() {
                params.push(self.parse_function_parameter()?);
            }

            // Determine function return type
            let (params, ty) = if self.check(&Token::Colon) {
                // Explicit function return type: def f x : type = ...
                self.advance();
                let return_ty = Some(self.parse_type()?);
                (params, return_ty)
            } else if let Some(last_param) = params.last() {
                // Check if last parameter is typed: def f x:type = ...
                // If so, extract the type as function return type
                if let PatternKind::Typed(inner_pattern, return_ty) = &last_param.kind {
                    let inner_pattern_clone = (**inner_pattern).clone();
                    let return_ty_clone = return_ty.clone();
                    let mut actual_params = params;
                    actual_params.pop();
                    actual_params.push(inner_pattern_clone);
                    (actual_params, Some(return_ty_clone))
                } else {
                    // No return type specified
                    (params, None)
                }
            } else {
                // No parameters and no return type
                (params, None)
            };

            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;

            Ok(Declaration::Decl(Decl {
                keyword,
                attributes,
                name,
                params,
                ty,
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

    /// Parse return type with optional attributes, returning parallel arrays
    /// Returns (return_types, return_attributes)
    fn parse_return_type(&mut self) -> Result<(Vec<Type>, Vec<Option<Attribute>>)> {
        trace!("parse_return_type: next token = {:?}", self.peek());

        // Check if it's a tuple: ([attr1] type1, [attr2] type2, ...)
        if self.check(&Token::LeftParen) {
            self.advance(); // consume '('
            let mut types = Vec::new();
            let mut attributes = Vec::new();

            if !self.check(&Token::RightParen) {
                loop {
                    // Parse optional #[attribute]
                    let attr = if self.check(&Token::AttributeStart) {
                        self.advance(); // consume '#['
                        let attribute = self.parse_attribute()?;
                        Some(attribute)
                    } else {
                        None
                    };

                    // Parse the type
                    let ty = self.parse_type()?;

                    types.push(ty);
                    attributes.push(attr);

                    if !self.check(&Token::Comma) {
                        break;
                    }
                    self.advance(); // consume ','
                }
            }

            self.expect(Token::RightParen)?;
            Ok((types, attributes))
        } else if self.check(&Token::AttributeStart) {
            // Single attributed type: #[attribute] type
            self.advance(); // consume '#['
            let attribute = self.parse_attribute()?;
            let ty = self.parse_type()?;

            Ok((vec![ty], vec![Some(attribute)]))
        } else {
            // Regular single return type without attributes
            let ty = self.parse_type()?;
            Ok((vec![ty], vec![None]))
        }
    }

    fn parse_uniform_decl(&mut self) -> Result<Declaration> {
        // Consume 'def' keyword
        self.expect(Token::Def)?;

        let name = self.expect_identifier()?;

        // Uniforms must have an explicit type annotation
        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;

        // Uniforms must NOT have initializers
        if self.check(&Token::Assign) {
            return Err(CompilerError::ParseError(
                "Uniform declarations cannot have initializer values".to_string(),
            ));
        }

        Ok(Declaration::Uniform(UniformDecl { name, ty }))
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
        // Check for existential size: ?[n][m]. type
        if self.check(&Token::QuestionMark) {
            return self.parse_existential_type();
        }
        self.parse_function_type()
    }

    fn parse_existential_type(&mut self) -> Result<Type> {
        self.expect(Token::QuestionMark)?;
        let mut size_vars = Vec::new();

        // Parse one or more [name] size variables
        while self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
            self.advance(); // consume '['
            let var_name = self.expect_identifier()?;
            self.expect(Token::RightBracket)?;
            size_vars.push(var_name);
        }

        if size_vars.is_empty() {
            return Err(CompilerError::ParseError(
                "Existential type must have at least one size variable".to_string(),
            ));
        }

        self.expect(Token::Dot)?;
        let inner_type = self.parse_function_type()?;

        Ok(types::existential(size_vars, inner_type))
    }

    fn parse_function_type(&mut self) -> Result<Type> {
        trace!("parse_function_type: next token = {:?}", self.peek());

        // Check for named parameter: (name: type) -> type
        if self.check(&Token::LeftParen) {
            let saved_pos = self.current;
            self.advance(); // consume '('

            // Try to parse as named parameter
            if let Some(Token::Identifier(name)) = self.peek() {
                let param_name = name.clone();
                self.advance();

                if self.check(&Token::Colon) {
                    // It's a named parameter
                    self.advance(); // consume ':'
                    let param_type = self.parse_type()?;
                    self.expect(Token::RightParen)?;

                    // Must be followed by ->
                    if self.check(&Token::Arrow) {
                        self.advance();
                        let return_type = self.parse_type()?;
                        let named_param = types::named_param(param_name, param_type);
                        return Ok(types::function(named_param, return_type));
                    } else {
                        return Err(CompilerError::ParseError(
                            "Named parameter must be followed by ->".to_string(),
                        ));
                    }
                }
            }

            // Not a named parameter, restore position and parse normally
            self.current = saved_pos;
        }

        // Regular function type or type application
        let mut left = self.parse_type_application()?;

        // Handle function arrows: T1 -> T2 -> T3
        while self.check(&Token::Arrow) {
            self.advance();
            let right = self.parse_type_application()?;
            left = types::function(left, right);
        }

        Ok(left)
    }

    fn parse_type_application(&mut self) -> Result<Type> {
        trace!("parse_type_application: next token = {:?}", self.peek());

        let mut base = self.parse_array_or_base_type()?;

        // Type application loop: keep applying type arguments
        // Grammar: type_application ::= type type_arg | "*" type
        //          type_arg         ::= "[" [dim] "]" | type
        loop {
            if self.is_at_type_boundary() {
                break;
            }

            match self.peek() {
                // Array dimension application: [n] or []
                Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => {
                    self.advance();

                    if self.check(&Token::RightBracket) {
                        // Empty brackets []
                        self.advance();
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![Type::Constructed(TypeName::Str("unsized"), vec![]), base],
                        );
                    } else if let Some(Token::Identifier(name)) = self.peek() {
                        // Size variable [n]
                        let size_var = name.clone();
                        self.advance();
                        self.expect(Token::RightBracket)?;
                        base = Type::Constructed(TypeName::Array, vec![types::size_var(size_var), base]);
                    } else if let Some(Token::IntLiteral(n)) = self.peek() {
                        // Size literal [3]
                        let size = *n as usize;
                        self.advance();
                        self.expect(Token::RightBracket)?;
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![Type::Constructed(TypeName::Size(size), vec![]), base],
                        );
                    } else {
                        return Err(CompilerError::ParseError(
                            "Expected size in array type application".to_string(),
                        ));
                    }
                }
                // Regular type argument application
                Some(Token::Identifier(_)) | Some(Token::LeftParen) | Some(Token::LeftBrace) => {
                    let arg_type = self.parse_array_or_base_type()?;
                    base = Type::Constructed(TypeName::Str("app"), vec![base, arg_type]);
                }
                _ => break,
            }
        }

        Ok(base)
    }

    // Helper to check if current token can start a type
    fn can_start_type(&self) -> bool {
        match self.peek() {
            Some(Token::LeftParen) => true, // Tuple type
            Some(Token::LeftBrace) => true, // Record type
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => true, // Array type
            Some(Token::BinOp(op)) if op == "*" => true, // Unique type
            Some(Token::Identifier(name)) => {
                // Uppercase = constructor/sum type, lowercase = base type like i32/f32
                name.chars().next().map_or(false, |c| {
                    c.is_uppercase()
                        || name == "i32"
                        || name == "f32"
                        || name.starts_with("vec")
                        || name.starts_with("ivec")
                        || name.starts_with("mat")
                        || name.starts_with('\'')
                })
            }
            _ => false,
        }
    }

    // Helper to check if we're at a type boundary (don't continue type application)
    fn is_at_type_boundary(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Arrow)
                | Some(Token::RightParen)
                | Some(Token::RightBrace)
                | Some(Token::Comma)
                | Some(Token::Assign)
                | Some(Token::Pipe)
                | Some(Token::Colon)
                | None
        ) || !self.can_start_type()
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

            // Check for empty brackets []
            if self.check(&Token::RightBracket) {
                self.advance();
                let elem_type = self.parse_array_or_base_type()?;
                return Ok(Type::Constructed(
                    TypeName::Array,
                    vec![Type::Constructed(TypeName::Str("unsized"), vec![]), elem_type],
                ));
            }

            // Parse dimension - could be integer literal or identifier (size variable)
            if let Some(Token::IntLiteral(n)) = self.peek() {
                let size = *n as usize;
                self.advance();
                self.expect(Token::RightBracket)?;
                let elem_type = self.parse_array_or_base_type()?; // Allow nested arrays
                Ok(types::sized_array(size, elem_type))
            } else if let Some(Token::Identifier(name)) = self.peek() {
                // Size variable [n]
                let size_var = name.clone();
                self.advance();
                self.expect(Token::RightBracket)?;
                let elem_type = self.parse_array_or_base_type()?;
                Ok(Type::Constructed(
                    TypeName::Array,
                    vec![types::size_var(size_var), elem_type],
                ))
            } else {
                Err(CompilerError::ParseError(
                    "Expected size literal or variable in array type".to_string(),
                ))
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
                // Tuple type (T1, T2, T3) or empty tuple ()
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
            Some(Token::LeftBrace) => {
                // Record type {field1: type1, field2: type2} or empty record {}
                self.parse_record_type()
            }
            Some(Token::Identifier(name)) if name.chars().next().unwrap().is_uppercase() => {
                // Sum type: Constructor type* | Constructor type*
                self.parse_sum_type()
            }
            _ => {
                let span = self.current_span();
                Err(CompilerError::ParseError(format!("Expected type at {}", span)))
            }
        }
    }

    fn parse_record_type(&mut self) -> Result<Type> {
        self.expect(Token::LeftBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RightBrace) {
            loop {
                // Parse field identifier (can be a number or name)
                let field_name = match self.peek() {
                    Some(Token::Identifier(name)) => {
                        let n = name.clone();
                        self.advance();
                        n
                    }
                    Some(Token::IntLiteral(n)) => {
                        let num = n.to_string();
                        self.advance();
                        num
                    }
                    _ => {
                        return Err(CompilerError::ParseError(
                            "Expected field name or number".to_string(),
                        ));
                    }
                };

                self.expect(Token::Colon)?;
                let field_type = self.parse_type()?;
                fields.push((field_name, field_type));

                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance(); // consume ','

                // Allow trailing comma
                if self.check(&Token::RightBrace) {
                    break;
                }
            }
        }

        self.expect(Token::RightBrace)?;
        Ok(types::record(fields))
    }

    fn parse_sum_type(&mut self) -> Result<Type> {
        let mut variants = Vec::new();

        loop {
            // Parse constructor name (uppercase identifier)
            let constructor_name = match self.peek() {
                Some(Token::Identifier(name)) if name.chars().next().unwrap().is_uppercase() => {
                    let n = name.clone();
                    self.advance();
                    n
                }
                _ => return Err(CompilerError::ParseError("Expected constructor name".to_string())),
            };

            // Parse zero or more type arguments for this constructor
            let mut arg_types = Vec::new();
            while !self.check(&Token::Pipe)
                && !self.check(&Token::RightParen)
                && !self.check(&Token::RightBracket)
                && !self.check(&Token::RightBrace)
                && !self.check(&Token::Comma)
                && !self.check(&Token::Arrow)
                && self.current < self.tokens.len()
            {
                // Try to parse a type argument
                // This is tricky - we need to avoid consuming tokens that aren't part of the sum type
                // For now, we'll be conservative and only parse simple types
                match self.peek() {
                    Some(Token::Identifier(_))
                    | Some(Token::LeftParen)
                    | Some(Token::LeftBrace)
                    | Some(Token::LeftBracket)
                    | Some(Token::LeftBracketSpaced) => {
                        arg_types.push(self.parse_array_or_base_type()?);
                    }
                    Some(Token::BinOp(op)) if op == "*" => {
                        arg_types.push(self.parse_array_or_base_type()?);
                    }
                    _ => break,
                }
            }

            variants.push((constructor_name, arg_types));

            // Check for more variants
            if self.check(&Token::Pipe) {
                self.advance(); // consume '|'
            } else {
                break;
            }
        }

        Ok(types::sum(variants))
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        trace!("parse_expression: next token = {:?}", self.peek());
        self.parse_type_ascription()
    }

    // Parse type ascription and coercion (lowest precedence)
    fn parse_type_ascription(&mut self) -> Result<Expression> {
        let mut expr = self.parse_range_expression()?;

        // Check for type ascription (:) or type coercion (:>)
        match self.peek() {
            Some(Token::Colon) => {
                let start_span = expr.h.span;
                self.advance();
                let ty = self.parse_type()?;
                let end_span = self.previous_span();
                let span = start_span.merge(&end_span);
                expr = self.node_counter.mk_node(ExprKind::TypeAscription(Box::new(expr), ty), span);
            }
            Some(Token::TypeCoercion) => {
                let start_span = expr.h.span;
                self.advance();
                let ty = self.parse_type()?;
                let end_span = self.previous_span();
                let span = start_span.merge(&end_span);
                expr = self.node_counter.mk_node(ExprKind::TypeCoercion(Box::new(expr), ty), span);
            }
            _ => {}
        }

        Ok(expr)
    }

    // Parse range expressions: a..b, a..<b, a..>b, a...b, a..step..end
    fn parse_range_expression(&mut self) -> Result<Expression> {
        let mut start = self.parse_binary_expression()?;

        // Check if we have a range operator
        match self.peek() {
            Some(Token::DotDot) | Some(Token::DotDotLt) | Some(Token::DotDotGt) | Some(Token::Ellipsis) => {
                let start_span = start.h.span;
                self.advance();
                let first_op = self.tokens[self.current - 1].token.clone();

                // Check if there's a step value (for a..step..end)
                let (step, end_op) = if matches!(first_op, Token::DotDot) {
                    // Parse potential step
                    let step_expr = self.parse_binary_expression()?;

                    // Check if there's another range operator
                    match self.peek() {
                        Some(Token::DotDotLt) | Some(Token::DotDotGt) | Some(Token::Ellipsis) => {
                            self.advance();
                            let second_op = self.tokens[self.current - 1].token.clone();
                            (Some(Box::new(step_expr)), second_op)
                        }
                        _ => {
                            // No second operator, step_expr is actually the end
                            let end_span = step_expr.h.span;
                            let span = start_span.merge(&end_span);
                            return Ok(self.node_counter.mk_node(
                                ExprKind::Range(RangeExpr {
                                    start: Box::new(start),
                                    step: None,
                                    end: Box::new(step_expr),
                                    kind: RangeKind::Exclusive,
                                }),
                                span,
                            ));
                        }
                    }
                } else {
                    (None, first_op)
                };

                // Parse the end expression
                let end = self.parse_binary_expression()?;
                let end_span = end.h.span;

                // Determine range kind
                let kind = match end_op {
                    Token::Ellipsis => RangeKind::Inclusive,
                    Token::DotDotLt => RangeKind::ExclusiveLt,
                    Token::DotDotGt => RangeKind::ExclusiveGt,
                    Token::DotDot => RangeKind::Exclusive,
                    _ => unreachable!(),
                };

                let span = start_span.merge(&end_span);
                start = self.node_counter.mk_node(
                    ExprKind::Range(RangeExpr {
                        start: Box::new(start),
                        step,
                        end: Box::new(end),
                        kind,
                    }),
                    span,
                );
            }
            _ => {}
        }

        Ok(start)
    }

    fn parse_binary_expression(&mut self) -> Result<Expression> {
        trace!("parse_binary_expression: next token = {:?}", self.peek());
        self.parse_binary_expression_with_precedence(0)
    }

    fn get_operator_precedence(op: &str) -> Option<(i32, bool)> {
        // Returns (precedence, is_left_associative)
        // Higher precedence binds tighter
        match op {
            "|>" => Some((8, true)),                                  // Pipe operator
            "*" | "/" => Some((3, true)),                             // Multiplication and division
            "+" | "-" => Some((2, true)),                             // Addition and subtraction
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
            // Check if we have a binary operator or pipe operator
            let op_string = match self.peek() {
                Some(Token::BinOp(op)) => op.clone(),
                Some(Token::PipeOp) => "|>".to_string(),
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

            // Build the appropriate operation with span from left to right
            let span = left.h.span.merge(&right.h.span);
            left = if op_string == "|>" {
                // Pipe operator creates a Pipe node
                self.node_counter.mk_node(ExprKind::Pipe(Box::new(left), Box::new(right)), span)
            } else {
                // Regular binary operation
                self.node_counter.mk_node(
                    ExprKind::BinaryOp(BinaryOp { op: op_string }, Box::new(left), Box::new(right)),
                    span,
                )
            };
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

            // Convert to FunctionCall or Application with span
            let span = expr.h.span.merge(&arg.h.span);
            match expr.kind {
                ExprKind::Identifier(name) => {
                    // First argument: convert identifier to function call
                    expr = self.node_counter.mk_node(ExprKind::FunctionCall(name, vec![arg]), span);
                }
                ExprKind::FunctionCall(name, mut args) => {
                    // Additional arguments: extend existing function call
                    args.push(arg);
                    expr = self.node_counter.mk_node(ExprKind::FunctionCall(name, args), span);
                }
                _ => {
                    // Higher-order function application: use Application node
                    expr =
                        self.node_counter.mk_node(ExprKind::Application(Box::new(expr), vec![arg]), span);
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
                    let start_span = expr.h.span;
                    self.advance();
                    let index = self.parse_expression()?;
                    self.expect(Token::RightBracket)?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    expr = self
                        .node_counter
                        .mk_node(ExprKind::ArrayIndex(Box::new(expr), Box::new(index)), span);
                }
                Some(Token::LeftBracketSpaced) => {
                    // Space before [ means it's not array indexing, it's a new expression
                    // Stop postfix parsing and let the caller handle it
                    break;
                }
                Some(Token::Dot) => {
                    // Field access (e.g., v.x, v.y, v.z, v.w)
                    let start_span = expr.h.span;
                    self.advance();
                    if let Some(Token::Identifier(field_name)) = self.peek().cloned() {
                        self.advance();
                        let end_span = self.previous_span();
                        let span = start_span.merge(&end_span);
                        expr = self
                            .node_counter
                            .mk_node(ExprKind::FieldAccess(Box::new(expr), field_name), span);
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
                    let end_span = self.previous_span();
                    let span = expr.h.span.merge(&end_span);

                    // Create FunctionCall for identifiers, Application for higher-order
                    expr = match &expr.kind {
                        ExprKind::Identifier(name) => {
                            self.node_counter.mk_node(ExprKind::FunctionCall(name.clone(), args), span)
                        }
                        ExprKind::FunctionCall(name, existing_args) => {
                            let mut all_args = existing_args.clone();
                            all_args.extend(args);
                            self.node_counter.mk_node(ExprKind::FunctionCall(name.clone(), all_args), span)
                        }
                        _ => self.node_counter.mk_node(ExprKind::Application(Box::new(expr), args), span),
                    };
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
                let start_span = self.current_span();
                self.advance();
                let operand = self.parse_unary_expression()?; // Right-associative
                let span = start_span.merge(&operand.h.span);
                Ok(self.node_counter.mk_node(
                    ExprKind::UnaryOp(UnaryOp { op: "-".to_string() }, Box::new(operand)),
                    span,
                ))
            }
            Some(Token::Bang) => {
                let start_span = self.current_span();
                self.advance();
                let operand = self.parse_unary_expression()?; // Right-associative
                let span = start_span.merge(&operand.h.span);
                Ok(self.node_counter.mk_node(
                    ExprKind::UnaryOp(UnaryOp { op: "!".to_string() }, Box::new(operand)),
                    span,
                ))
            }
            _ => self.parse_primary_expression(),
        }
    }

    fn parse_primary_expression(&mut self) -> Result<Expression> {
        trace!("parse_primary_expression: next token = {:?}", self.peek());
        match self.peek() {
            Some(Token::TypeHole) => {
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::TypeHole, span))
            }
            Some(Token::IntLiteral(n)) => {
                let n = *n;
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::IntLiteral(n), span))
            }
            Some(Token::FloatLiteral(f)) => {
                let f = *f;
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(f), span))
            }
            Some(Token::True) => {
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::BoolLiteral(true), span))
            }
            Some(Token::False) => {
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::BoolLiteral(false), span))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::Identifier(name), span))
            }
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => self.parse_array_literal(),
            Some(Token::LeftParen) => {
                let start_span = self.current_span();
                self.advance(); // consume '('

                // Check for empty tuple
                if self.check(&Token::RightParen) {
                    self.advance();
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    return Ok(self.node_counter.mk_node(ExprKind::Tuple(vec![]), span));
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
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    Ok(self.node_counter.mk_node(ExprKind::Tuple(elements), span))
                } else {
                    // Just a parenthesized expression
                    self.expect(Token::RightParen)?;
                    Ok(first_expr)
                }
            }
            Some(Token::Backslash) => self.parse_lambda(),
            Some(Token::Let) => self.parse_let_in(),
            Some(Token::If) => self.parse_if_then_else(),
            Some(Token::Loop) => self.parse_loop(),
            Some(Token::Match) => self.parse_match(),
            Some(Token::Unsafe) => self.parse_unsafe(),
            Some(Token::Assert) => self.parse_assert(),
            _ => {
                let span = self.current_span();
                Err(CompilerError::ParseError(format!(
                    "Expected expression, got {:?} at {}",
                    self.peek(),
                    span
                )))
            }
        }
    }

    fn parse_array_literal(&mut self) -> Result<Expression> {
        trace!("parse_array_literal: next token = {:?}", self.peek());
        // Accept either LeftBracket or LeftBracketSpaced
        let start_span = self.current_span();
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
        let end_span = self.previous_span();
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(ExprKind::ArrayLiteral(elements), span))
    }

    fn parse_lambda(&mut self) -> Result<Expression> {
        trace!("parse_lambda: next token = {:?}", self.peek());
        let start_span = self.current_span();
        self.expect(Token::Backslash)?;

        // Parse parameter patterns: \pat1 pat2 ... [: type] -> exp
        let mut params = Vec::new();

        // Parse patterns until we hit : or ->
        while !self.check(&Token::Colon) && !self.check(&Token::Arrow) && !self.is_at_end() {
            params.push(self.parse_pattern()?);
        }

        if params.is_empty() {
            let span = self.current_span();
            return Err(CompilerError::ParseError(format!(
                "Lambda must have at least one parameter at {}",
                span
            )));
        }

        // Parse optional return type annotation: \pat1 pat2: t ->
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
        let span = start_span.merge(&body.h.span);

        Ok(self.node_counter.mk_node(
            ExprKind::Lambda(LambdaExpr {
                params,
                return_type,
                body,
            }),
            span,
        ))
    }

    fn parse_let_in(&mut self) -> Result<Expression> {
        trace!("parse_let_in: next token = {:?}", self.peek());
        use crate::ast::LetInExpr;

        let start_span = self.current_span();
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
        let span = start_span.merge(&body.h.span);

        Ok(self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                name,
                ty,
                value,
                body,
            }),
            span,
        ))
    }

    fn parse_if_then_else(&mut self) -> Result<Expression> {
        trace!("parse_if_then_else: next token = {:?}", self.peek());
        use crate::ast::IfExpr;

        let start_span = self.current_span();
        self.expect(Token::If)?;
        let condition = Box::new(self.parse_expression()?);
        self.expect(Token::Then)?;
        let then_branch = Box::new(self.parse_expression()?);
        self.expect(Token::Else)?;
        let else_branch = Box::new(self.parse_expression()?);
        let span = start_span.merge(&else_branch.h.span);

        Ok(self.node_counter.mk_node(
            ExprKind::If(IfExpr {
                condition,
                then_branch,
                else_branch,
            }),
            span,
        ))
    }

    fn parse_loop(&mut self) -> Result<Expression> {
        trace!("parse_loop: next token = {:?}", self.peek());
        use crate::ast::{LoopExpr, LoopForm};

        let start_span = self.current_span();
        self.expect(Token::Loop)?;
        let pattern = self.parse_pattern()?;

        // Check for optional initialization: = exp
        let init = if self.check(&Token::Assign) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        // Parse loop form
        let form = if self.check(&Token::For) {
            self.advance();
            // Check if it's "for name < exp" or "for pat in exp"
            let saved_pos = self.current;

            // Try to parse as pattern first
            if let Ok(pat) = self.parse_pattern() {
                if self.check(&Token::In) {
                    // It's "for pat in exp"
                    self.advance();
                    let iter_expr = Box::new(self.parse_expression()?);
                    LoopForm::ForIn(pat, iter_expr)
                } else {
                    // Backtrack and try as "for name < exp"
                    self.current = saved_pos;
                    let name = self.expect_identifier()?;
                    self.expect(Token::BinOp("<".to_string()))?;
                    let bound = Box::new(self.parse_expression()?);
                    LoopForm::For(name, bound)
                }
            } else {
                return Err(CompilerError::ParseError(
                    "Expected pattern in for loop".to_string(),
                ));
            }
        } else if self.check(&Token::While) {
            self.advance();
            let condition = Box::new(self.parse_expression()?);
            LoopForm::While(condition)
        } else {
            return Err(CompilerError::ParseError(
                "Expected 'for' or 'while' in loop".to_string(),
            ));
        };

        self.expect(Token::Do)?;
        let body = Box::new(self.parse_expression()?);
        let span = start_span.merge(&body.h.span);

        Ok(self.node_counter.mk_node(
            ExprKind::Loop(LoopExpr {
                pattern,
                init,
                form,
                body,
            }),
            span,
        ))
    }

    fn parse_match(&mut self) -> Result<Expression> {
        trace!("parse_match: next token = {:?}", self.peek());
        use crate::ast::{MatchCase, MatchExpr};

        let start_span = self.current_span();
        self.expect(Token::Match)?;
        let scrutinee = Box::new(self.parse_expression()?);

        // Parse one or more case branches
        let mut cases = Vec::new();
        let mut last_span = scrutinee.h.span;
        while self.check(&Token::Case) {
            self.advance();
            let pattern = self.parse_pattern()?;
            self.expect(Token::Arrow)?;
            let body = Box::new(self.parse_expression()?);
            last_span = body.h.span;
            cases.push(MatchCase { pattern, body });
        }

        if cases.is_empty() {
            return Err(CompilerError::ParseError(
                "Match expression must have at least one case".to_string(),
            ));
        }

        let span = start_span.merge(&last_span);
        Ok(self.node_counter.mk_node(ExprKind::Match(MatchExpr { scrutinee, cases }), span))
    }

    fn parse_unsafe(&mut self) -> Result<Expression> {
        trace!("parse_unsafe: next token = {:?}", self.peek());
        let start_span = self.current_span();
        self.expect(Token::Unsafe)?;
        let expr = Box::new(self.parse_expression()?);
        let span = start_span.merge(&expr.h.span);
        Ok(self.node_counter.mk_node(ExprKind::Unsafe(expr), span))
    }

    fn parse_assert(&mut self) -> Result<Expression> {
        trace!("parse_assert: next token = {:?}", self.peek());
        let start_span = self.current_span();
        self.expect(Token::Assert)?;
        // According to grammar: "assert" atom exp
        // atom is a primary expression (condition)
        let condition = Box::new(self.parse_primary_expression()?);
        let body = Box::new(self.parse_expression()?);
        let span = start_span.merge(&body.h.span);
        Ok(self.node_counter.mk_node(ExprKind::Assert(condition, body), span))
    }

    // Helper methods
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current).map(|lt| &lt.token)
    }

    fn peek_ahead(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.current + n).map(|lt| &lt.token)
    }

    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.current += 1;
            self.tokens.get(self.current - 1).map(|lt| &lt.token)
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
            let span = self.current_span();
            Err(CompilerError::ParseError(format!(
                "Expected {:?}, got {:?} at {}",
                token,
                self.peek(),
                span
            )))
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        let span = self.current_span();
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name.clone()),
            _ => Err(CompilerError::ParseError(format!(
                "Expected identifier at {}",
                span
            ))),
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }
}
