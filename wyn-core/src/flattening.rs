//! Flattening pass: AST -> MIR
//!
//! This pass performs:
//! - Defunctionalization: lambdas become top-level functions with closure records
//! - Pattern flattening: complex patterns become simple let bindings
//! - Lambda lifting: all functions become top-level Def entries

use crate::ast::{self, ExprKind, Expression, NodeId, PatternKind, Span, Type, TypeName};
use crate::error::{CompilerError, Result};
use crate::mir::{self, Expr};
use crate::pattern;
use polytype::TypeScheme;
use std::collections::{HashMap, HashSet};

/// Flattener converts AST to MIR with defunctionalization.
pub struct Flattener {
    /// Counter for generating unique names
    next_id: usize,
    /// Generated lambda functions (collected during flattening)
    generated_functions: Vec<mir::Def>,
    /// Stack of enclosing declaration names for lambda naming
    enclosing_decl_stack: Vec<String>,
    /// Registry of lambdas: tag -> (function_name, arity)
    lambda_registry: HashMap<i32, (String, usize)>,
    /// Next tag to assign
    next_tag: i32,
    /// Type table from type checking - maps NodeId to TypeScheme
    type_table: HashMap<NodeId, TypeScheme<TypeName>>,
}

impl Flattener {
    pub fn new(type_table: HashMap<NodeId, TypeScheme<TypeName>>) -> Self {
        Flattener {
            next_id: 0,
            generated_functions: Vec::new(),
            enclosing_decl_stack: Vec::new(),
            lambda_registry: HashMap::new(),
            next_tag: 0,
            type_table,
        }
    }

    /// Extract the monotype from a TypeScheme
    fn get_monotype<'a>(&self, scheme: &'a TypeScheme<TypeName>) -> Option<&'a Type> {
        match scheme {
            TypeScheme::Monotype(ty) => Some(ty),
            TypeScheme::Polytype { body, .. } => self.get_monotype(body),
        }
    }

    /// Get the type of an AST expression from the type table
    fn get_expr_type(&self, expr: &Expression) -> Type {
        self.type_table
            .get(&expr.h.id)
            .and_then(|scheme| self.get_monotype(scheme))
            .cloned()
            .unwrap_or_else(|| {
                // Fallback to a placeholder type if not found
                Type::Constructed(TypeName::Str("unknown".into()), vec![])
            })
    }

    /// Get the type of an AST pattern from the type table
    fn get_pattern_type(&self, pat: &ast::Pattern) -> Type {
        self.type_table
            .get(&pat.h.id)
            .and_then(|scheme| self.get_monotype(scheme))
            .cloned()
            .unwrap_or_else(|| Type::Constructed(TypeName::Str("unknown".into()), vec![]))
    }

    /// Generate a unique ID
    fn fresh_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Generate a unique variable name
    fn fresh_name(&mut self, prefix: &str) -> String {
        format!("__{}{}", prefix, self.fresh_id())
    }

    /// Resolve a field name to a numeric index using type information
    fn resolve_field_index(&self, obj: &Expression, field: &str) -> Result<usize> {
        // First try numeric index (for tuple access like .0, .1)
        if let Ok(idx) = field.parse::<usize>() {
            return Ok(idx);
        }

        // Look up the type of the object
        let scheme = self.type_table.get(&obj.h.id).ok_or_else(|| {
            CompilerError::FlatteningError(format!("No type information for field access target"))
        })?;

        let obj_type = self.get_monotype(scheme).ok_or_else(|| {
            CompilerError::FlatteningError(format!("Could not extract monotype from scheme"))
        })?;

        // Resolve based on type
        match obj_type {
            // Vector types: x=0, y=1, z=2, w=3
            Type::Constructed(TypeName::Vec, _) => match field {
                "x" => Ok(0),
                "y" => Ok(1),
                "z" => Ok(2),
                "w" => Ok(3),
                _ => Err(CompilerError::FlatteningError(format!(
                    "Unknown vector field: {}",
                    field
                ))),
            },
            // Record types: look up field by name
            Type::Constructed(TypeName::Record(fields), _) => fields
                .keys()
                .enumerate()
                .find(|(_, name)| name.as_str() == field)
                .map(|(idx, _)| idx)
                .ok_or_else(|| CompilerError::FlatteningError(format!("Unknown record field: {}", field))),
            // Tuple types: should use numeric access
            Type::Constructed(TypeName::Str(s), _) if *s == "tuple" => Err(CompilerError::FlatteningError(
                format!("Tuple access must use numeric index, not '{}'", field),
            )),
            _ => Err(CompilerError::FlatteningError(format!(
                "Cannot access field '{}' on type {:?}",
                field, obj_type
            ))),
        }
    }

    /// Flatten an entire program
    pub fn flatten_program(&mut self, program: &ast::Program) -> Result<mir::Program> {
        let mut defs = Vec::new();

        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    self.enclosing_decl_stack.push(d.name.clone());

                    let def = if d.params.is_empty() {
                        // Constant
                        let body = self.flatten_expr(&d.body)?;
                        let ty = self.get_expr_type(&d.body);
                        mir::Def::Constant {
                            name: d.name.clone(),
                            ty,
                            attributes: self.convert_attributes(&d.attributes),
                            body,
                            span: d.body.h.span,
                        }
                    } else {
                        // Function
                        let params = self.flatten_params(&d.params)?;
                        let param_attrs = self.extract_param_attributes(&d.params);
                        let body = self.flatten_expr(&d.body)?;
                        let ret_type = self.get_expr_type(&d.body);
                        mir::Def::Function {
                            name: d.name.clone(),
                            params,
                            ret_type,
                            attributes: self.convert_attributes(&d.attributes),
                            param_attributes: param_attrs,
                            return_attributes: vec![],
                            body,
                            span: d.body.h.span,
                        }
                    };

                    // Collect generated lambdas before the definition
                    defs.append(&mut self.generated_functions);
                    defs.push(def);

                    self.enclosing_decl_stack.pop();
                    self.lambda_registry.clear();
                    self.next_tag = 0;
                }
                ast::Declaration::Entry(e) => {
                    self.enclosing_decl_stack.push(e.name.clone());

                    let params = self.flatten_params(&e.params)?;
                    let param_attrs = self.extract_param_attributes(&e.params);
                    let body = self.flatten_expr(&e.body)?;
                    let ret_type = self.get_expr_type(&e.body);
                    let entry_kind = if e.entry_type.is_vertex() { "vertex" } else { "fragment" };
                    let attrs = vec![mir::Attribute {
                        name: entry_kind.to_string(),
                        args: vec![],
                    }];

                    // Extract return attributes from EntryDecl
                    // Each return value gets its own Vec of attributes
                    let return_attrs: Vec<Vec<mir::Attribute>> =
                        e.return_attributes
                            .iter()
                            .map(|opt| {
                                if let Some(attr) = opt {
                                    vec![self.convert_attribute(attr)]
                                } else {
                                    vec![]
                                }
                            })
                            .collect();

                    let def = mir::Def::Function {
                        name: e.name.clone(),
                        params,
                        ret_type,
                        attributes: attrs,
                        param_attributes: param_attrs,
                        return_attributes: return_attrs,
                        body,
                        span: e.body.h.span,
                    };

                    defs.append(&mut self.generated_functions);
                    defs.push(def);

                    self.enclosing_decl_stack.pop();
                    self.lambda_registry.clear();
                    self.next_tag = 0;
                }
                ast::Declaration::Uniform(_)
                | ast::Declaration::Val(_)
                | ast::Declaration::TypeBind(_)
                | ast::Declaration::ModuleBind(_)
                | ast::Declaration::ModuleTypeBind(_)
                | ast::Declaration::Open(_)
                | ast::Declaration::Import(_)
                | ast::Declaration::Local(_) => {
                    // Skip declarations that don't produce MIR defs
                }
            }
        }

        Ok(mir::Program { defs })
    }

    /// Convert AST attributes to MIR attributes
    fn convert_attributes(&self, attrs: &[ast::Attribute]) -> Vec<mir::Attribute> {
        attrs.iter().map(|a| self.convert_attribute(a)).collect()
    }

    /// Convert a single AST attribute to MIR attribute
    fn convert_attribute(&self, attr: &ast::Attribute) -> mir::Attribute {
        match attr {
            ast::Attribute::BuiltIn(builtin) => mir::Attribute {
                name: "builtin".to_string(),
                args: vec![format!("{:?}", builtin)],
            },
            ast::Attribute::Location(loc) => mir::Attribute {
                name: "location".to_string(),
                args: vec![loc.to_string()],
            },
            ast::Attribute::Vertex => mir::Attribute {
                name: "vertex".to_string(),
                args: vec![],
            },
            ast::Attribute::Fragment => mir::Attribute {
                name: "fragment".to_string(),
                args: vec![],
            },
            ast::Attribute::Uniform => mir::Attribute {
                name: "uniform".to_string(),
                args: vec![],
            },
        }
    }

    /// Extract attributes from each parameter pattern
    fn extract_param_attributes(&self, params: &[ast::Pattern]) -> Vec<Vec<mir::Attribute>> {
        params.iter().map(|p| self.extract_pattern_attributes(p)).collect()
    }

    /// Extract attributes from a pattern (handling Attributed and Typed wrappers)
    fn extract_pattern_attributes(&self, pattern: &ast::Pattern) -> Vec<mir::Attribute> {
        match &pattern.kind {
            PatternKind::Attributed(attrs, inner) => {
                let mut result: Vec<mir::Attribute> =
                    attrs.iter().map(|a| self.convert_attribute(a)).collect();
                result.extend(self.extract_pattern_attributes(inner));
                result
            }
            PatternKind::Typed(inner, _) => self.extract_pattern_attributes(inner),
            _ => vec![],
        }
    }

    /// Flatten function parameters
    fn flatten_params(&self, params: &[ast::Pattern]) -> Result<Vec<mir::Param>> {
        let mut result = Vec::new();
        for param in params {
            let name = self.extract_param_name(param)?;
            let ty = self.get_pattern_type(param);
            result.push(mir::Param {
                name,
                ty,
                is_consumed: false, // TODO: track uniqueness
            });
        }
        Ok(result)
    }

    /// Extract parameter name from pattern
    fn extract_param_name(&self, pattern: &ast::Pattern) -> Result<String> {
        match &pattern.kind {
            PatternKind::Name(name) => Ok(name.clone()),
            PatternKind::Typed(inner, _) => self.extract_param_name(inner),
            PatternKind::Attributed(_, inner) => self.extract_param_name(inner),
            _ => Err(CompilerError::FlatteningError(
                "Complex parameter patterns not yet supported".to_string(),
            )),
        }
    }

    /// Flatten an expression
    fn flatten_expr(&mut self, expr: &Expression) -> Result<Expr> {
        let span = expr.h.span;
        let ty = self.get_expr_type(expr);
        let kind = match &expr.kind {
            ExprKind::IntLiteral(n) => mir::ExprKind::Literal(mir::Literal::Int(n.to_string())),
            ExprKind::FloatLiteral(f) => mir::ExprKind::Literal(mir::Literal::Float(f.to_string())),
            ExprKind::BoolLiteral(b) => mir::ExprKind::Literal(mir::Literal::Bool(*b)),
            ExprKind::Identifier(name) => mir::ExprKind::Var(name.clone()),
            ExprKind::QualifiedName(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                mir::ExprKind::Var(full_name)
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let lhs = self.flatten_expr(lhs)?;
                let rhs = self.flatten_expr(rhs)?;
                mir::ExprKind::BinOp {
                    op: op.op.clone(),
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
            ExprKind::UnaryOp(op, operand) => {
                let operand = self.flatten_expr(operand)?;
                mir::ExprKind::UnaryOp {
                    op: op.op.clone(),
                    operand: Box::new(operand),
                }
            }
            ExprKind::If(if_expr) => {
                let cond = self.flatten_expr(&if_expr.condition)?;
                let then_branch = self.flatten_expr(&if_expr.then_branch)?;
                let else_branch = self.flatten_expr(&if_expr.else_branch)?;
                mir::ExprKind::If {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                }
            }
            ExprKind::LetIn(let_in) => self.flatten_let_in(let_in, span)?,
            ExprKind::Lambda(lambda) => self.flatten_lambda(lambda, span)?,
            ExprKind::Application(func, args) => self.flatten_application(func, args, span)?,
            ExprKind::FunctionCall(name, args) => {
                let args: Result<Vec<_>> = args.iter().map(|a| self.flatten_expr(a)).collect();
                mir::ExprKind::Call {
                    func: name.clone(),
                    args: args?,
                }
            }
            ExprKind::Tuple(elems) => {
                let elems: Result<Vec<_>> = elems.iter().map(|e| self.flatten_expr(e)).collect();
                mir::ExprKind::Literal(mir::Literal::Tuple(elems?))
            }
            ExprKind::ArrayLiteral(elems) => {
                let elems: Result<Vec<_>> = elems.iter().map(|e| self.flatten_expr(e)).collect();
                mir::ExprKind::Literal(mir::Literal::Array(elems?))
            }
            ExprKind::RecordLiteral(fields) => {
                let fields: Result<Vec<_>> = fields
                    .iter()
                    .map(|(name, expr)| Ok((name.clone(), self.flatten_expr(expr)?)))
                    .collect();
                mir::ExprKind::Literal(mir::Literal::Record(fields?))
            }
            ExprKind::ArrayIndex(arr, idx) => {
                let arr = self.flatten_expr(arr)?;
                let idx = self.flatten_expr(idx)?;
                mir::ExprKind::Intrinsic {
                    name: "index".to_string(),
                    args: vec![arr, idx],
                }
            }
            ExprKind::FieldAccess(obj_expr, field) => {
                let obj = self.flatten_expr(obj_expr)?;

                // Resolve field name to index using type information
                let idx = self.resolve_field_index(obj_expr, field)?;

                // Create i32 type for the index literal
                let i32_type = Type::Constructed(TypeName::Str("i32".into()), vec![]);

                mir::ExprKind::Intrinsic {
                    name: "tuple_access".to_string(),
                    args: vec![
                        obj,
                        Expr::new(
                            i32_type,
                            mir::ExprKind::Literal(mir::Literal::Int(idx.to_string())),
                            span,
                        ),
                    ],
                }
            }
            ExprKind::Loop(loop_expr) => self.flatten_loop(loop_expr, span)?,
            ExprKind::Pipe(lhs, rhs) => {
                // a |> f  =>  f(a)
                let lhs_flat = self.flatten_expr(lhs)?;
                // Treat rhs as a function to apply to lhs
                match &rhs.kind {
                    ExprKind::Identifier(name) => mir::ExprKind::Call {
                        func: name.clone(),
                        args: vec![lhs_flat],
                    },
                    _ => {
                        // General case: application
                        let rhs_flat = self.flatten_expr(rhs)?;
                        // This would need closure calling, simplified for now
                        return Err(CompilerError::FlatteningError(
                            "Complex pipe expressions not yet supported".to_string(),
                        ));
                    }
                }
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                // Type annotations don't affect runtime, just flatten inner
                return self.flatten_expr(inner);
            }
            ExprKind::Unsafe(inner) => {
                let inner = self.flatten_expr(inner)?;
                mir::ExprKind::Attributed {
                    attributes: vec![mir::Attribute {
                        name: "unsafe".to_string(),
                        args: vec![],
                    }],
                    expr: Box::new(inner),
                }
            }
            ExprKind::Assert(cond, body) => {
                let cond = self.flatten_expr(cond)?;
                let body = self.flatten_expr(body)?;
                mir::ExprKind::Intrinsic {
                    name: "assert".to_string(),
                    args: vec![cond, body],
                }
            }
            ExprKind::TypeHole => {
                return Err(CompilerError::FlatteningError(
                    "Type holes should be resolved before flattening".to_string(),
                ));
            }
            ExprKind::Match(_) => {
                return Err(CompilerError::FlatteningError(
                    "Match expressions not yet supported".to_string(),
                ));
            }
            ExprKind::Range(_) => {
                return Err(CompilerError::FlatteningError(
                    "Range expressions should be desugared before flattening".to_string(),
                ));
            }
        };

        Ok(Expr::new(ty, kind, span))
    }

    /// Flatten a let-in expression, handling pattern destructuring
    fn flatten_let_in(&mut self, let_in: &ast::LetInExpr, span: Span) -> Result<mir::ExprKind> {
        let value = self.flatten_expr(&let_in.value)?;

        // Check if pattern is simple (just a name)
        match &let_in.pattern.kind {
            PatternKind::Name(name) => {
                let body = self.flatten_expr(&let_in.body)?;
                Ok(mir::ExprKind::Let {
                    name: name.clone(),
                    value: Box::new(value),
                    body: Box::new(body),
                })
            }
            PatternKind::Typed(inner, _) => {
                // Recursively handle typed pattern
                let inner_let = ast::LetInExpr {
                    pattern: (**inner).clone(),
                    ty: let_in.ty.clone(),
                    value: let_in.value.clone(),
                    body: let_in.body.clone(),
                };
                self.flatten_let_in(&inner_let, span)
            }
            PatternKind::Wildcard => {
                // Bind to ignored variable, just for side effects
                let body = self.flatten_expr(&let_in.body)?;
                Ok(mir::ExprKind::Let {
                    name: self.fresh_name("ignored"),
                    value: Box::new(value),
                    body: Box::new(body),
                })
            }
            PatternKind::Tuple(patterns) => {
                // Generate a temp name for the tuple value
                let tmp = self.fresh_name("tup");

                // Get the tuple type and element types
                let tuple_ty = self.get_pattern_type(&let_in.pattern);
                let elem_types: Vec<Type> = match &tuple_ty {
                    Type::Constructed(TypeName::Str(s), args) if *s == "tuple" => args.clone(),
                    _ => {
                        // Fallback: use unknown types
                        patterns.iter().map(|p| self.get_pattern_type(p)).collect()
                    }
                };

                // Build nested lets from inside out
                let mut body = self.flatten_expr(&let_in.body)?;

                // Extract each element
                for (i, pat) in patterns.iter().enumerate().rev() {
                    let name = match &pat.kind {
                        PatternKind::Name(n) => n.clone(),
                        PatternKind::Typed(inner, _) => match &inner.kind {
                            PatternKind::Name(n) => n.clone(),
                            _ => {
                                return Err(CompilerError::FlatteningError(
                                    "Nested complex patterns not supported".to_string(),
                                ));
                            }
                        },
                        PatternKind::Wildcard => continue, // Skip wildcards
                        _ => {
                            return Err(CompilerError::FlatteningError(
                                "Complex nested patterns not supported".to_string(),
                            ));
                        }
                    };

                    let elem_ty = elem_types
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| Type::Constructed(TypeName::Str("unknown".into()), vec![]));
                    let i32_type = Type::Constructed(TypeName::Str("i32".into()), vec![]);

                    let extract = Expr::new(
                        elem_ty.clone(),
                        mir::ExprKind::Intrinsic {
                            name: "tuple_access".to_string(),
                            args: vec![
                                Expr::new(tuple_ty.clone(), mir::ExprKind::Var(tmp.clone()), span),
                                Expr::new(
                                    i32_type,
                                    mir::ExprKind::Literal(mir::Literal::Int(i.to_string())),
                                    span,
                                ),
                            ],
                        },
                        span,
                    );

                    body = Expr::new(
                        body.ty.clone(),
                        mir::ExprKind::Let {
                            name,
                            value: Box::new(extract),
                            body: Box::new(body),
                        },
                        span,
                    );
                }

                // Wrap with the tuple binding
                Ok(mir::ExprKind::Let {
                    name: tmp,
                    value: Box::new(value),
                    body: Box::new(body),
                })
            }
            _ => Err(CompilerError::FlatteningError(format!(
                "Pattern kind {:?} not yet supported in let",
                let_in.pattern.kind
            ))),
        }
    }

    /// Flatten a lambda expression (defunctionalization)
    fn flatten_lambda(&mut self, lambda: &ast::LambdaExpr, span: Span) -> Result<mir::ExprKind> {
        // Find free variables
        let mut bound = HashSet::new();
        for param in &lambda.params {
            if let Some(name) = param.simple_name() {
                bound.insert(name.to_string());
            }
        }
        let free_vars = self.find_free_variables(&lambda.body, &bound);

        // Generate function name
        let id = self.fresh_id();
        let enclosing = self.enclosing_decl_stack.last().map(|s| s.as_str()).unwrap_or("anon");
        let func_name = format!("__lam_{}_{}", enclosing, id);

        // Assign tag
        let tag = self.next_tag;
        self.next_tag += 1;
        let arity = lambda.params.len();
        self.lambda_registry.insert(tag, (func_name.clone(), arity));

        // Build parameters: closure first, then lambda params
        // Use a placeholder type for closure (it's a record, but we don't track its structure here)
        let closure_type = Type::Constructed(TypeName::Str("closure".into()), vec![]);
        let mut params = vec![mir::Param {
            name: "__closure".to_string(),
            ty: closure_type,
            is_consumed: false,
        }];

        for param in &lambda.params {
            let name = param
                .simple_name()
                .ok_or_else(|| {
                    CompilerError::FlatteningError(
                        "Complex lambda parameter patterns not supported".to_string(),
                    )
                })?
                .to_string();
            let ty = self.get_pattern_type(param);
            params.push(mir::Param {
                name,
                ty,
                is_consumed: false,
            });
        }

        // Rewrite body to access free variables from closure
        let rewritten_body = self.rewrite_free_vars(&lambda.body, &free_vars)?;
        let body = self.flatten_expr(&rewritten_body)?;
        let ret_type = body.ty.clone();

        // Create the generated function
        let func = mir::Def::Function {
            name: func_name.clone(),
            params,
            ret_type,
            attributes: vec![],
            param_attributes: vec![],
            return_attributes: vec![],
            body,
            span,
        };
        self.generated_functions.push(func);

        // Create closure record: { __tag: tag, var1: var1, var2: var2, ... }
        let i32_type = Type::Constructed(TypeName::Str("i32".into()), vec![]);
        let mut fields = vec![(
            "__tag".to_string(),
            Expr::new(
                i32_type,
                mir::ExprKind::Literal(mir::Literal::Int(tag.to_string())),
                span,
            ),
        )];

        let mut sorted_vars: Vec<_> = free_vars.iter().collect();
        sorted_vars.sort();
        for var in &sorted_vars {
            // TODO: We don't have easy access to the type of free variables here
            // For now, use unknown type
            let var_type = Type::Constructed(TypeName::Str("unknown".into()), vec![]);
            fields.push((
                (*var).clone(),
                Expr::new(var_type, mir::ExprKind::Var((*var).clone()), span),
            ));
        }

        Ok(mir::ExprKind::Literal(mir::Literal::Record(fields)))
    }

    /// Flatten an application expression
    fn flatten_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
        span: Span,
    ) -> Result<mir::ExprKind> {
        let func_flat = self.flatten_expr(func)?;
        let args_flat: Result<Vec<_>> = args.iter().map(|a| self.flatten_expr(a)).collect();
        let args_flat = args_flat?;

        // Check if this is applying a known function name
        match &func.kind {
            ExprKind::Identifier(name) => {
                // Direct function call
                Ok(mir::ExprKind::Call {
                    func: name.clone(),
                    args: args_flat,
                })
            }
            _ => {
                // Closure call: need to call __applyN or direct lambda call
                // For simplicity, use intrinsic for now
                let arity = args_flat.len();
                let mut all_args = vec![func_flat];
                all_args.extend(args_flat);

                Ok(mir::ExprKind::Intrinsic {
                    name: format!("apply{}", arity),
                    args: all_args,
                })
            }
        }
    }

    /// Flatten a loop expression
    fn flatten_loop(&mut self, loop_expr: &ast::LoopExpr, span: Span) -> Result<mir::ExprKind> {
        // Extract init bindings from pattern
        let init_bindings =
            self.extract_loop_bindings(&loop_expr.pattern, loop_expr.init.as_deref(), span)?;

        // Flatten loop kind
        let kind = match &loop_expr.form {
            ast::LoopForm::While(cond) => {
                let cond = self.flatten_expr(cond)?;
                mir::LoopKind::While { cond: Box::new(cond) }
            }
            ast::LoopForm::For(var, bound) => {
                let bound = self.flatten_expr(bound)?;
                mir::LoopKind::ForRange {
                    var: var.clone(),
                    bound: Box::new(bound),
                }
            }
            ast::LoopForm::ForIn(pat, iter) => {
                let var = match &pat.kind {
                    PatternKind::Name(n) => n.clone(),
                    _ => {
                        return Err(CompilerError::FlatteningError(
                            "Complex for-in patterns not supported".to_string(),
                        ));
                    }
                };
                let iter = self.flatten_expr(iter)?;
                mir::LoopKind::For {
                    var,
                    iter: Box::new(iter),
                }
            }
        };

        let body = self.flatten_expr(&loop_expr.body)?;

        Ok(mir::ExprKind::Loop {
            init_bindings,
            kind,
            body: Box::new(body),
        })
    }

    /// Extract loop init bindings from pattern and init expression
    fn extract_loop_bindings(
        &mut self,
        pattern: &ast::Pattern,
        init: Option<&Expression>,
        span: Span,
    ) -> Result<Vec<(String, Expr)>> {
        let init_expr = init
            .ok_or_else(|| CompilerError::FlatteningError("Loop must have init expression".to_string()))?;

        match &pattern.kind {
            PatternKind::Name(name) => {
                let init_flat = self.flatten_expr(init_expr)?;
                Ok(vec![(name.clone(), init_flat)])
            }
            PatternKind::Typed(inner, _) => self.extract_loop_bindings(inner, init, span),
            PatternKind::Tuple(patterns) => {
                // Init should also be a tuple
                let init_flat = self.flatten_expr(init_expr)?;
                let tuple_ty = init_flat.ty.clone();
                let tmp = self.fresh_name("init");

                // Get element types from tuple type
                let elem_types: Vec<Type> = match &tuple_ty {
                    Type::Constructed(TypeName::Str(s), args) if *s == "tuple" => args.clone(),
                    _ => patterns.iter().map(|p| self.get_pattern_type(p)).collect(),
                };

                let mut bindings = Vec::new();
                for (i, pat) in patterns.iter().enumerate() {
                    let name = match &pat.kind {
                        PatternKind::Name(n) => n.clone(),
                        PatternKind::Typed(inner, _) => match &inner.kind {
                            PatternKind::Name(n) => n.clone(),
                            _ => {
                                return Err(CompilerError::FlatteningError(
                                    "Complex loop patterns not supported".to_string(),
                                ));
                            }
                        },
                        _ => {
                            return Err(CompilerError::FlatteningError(
                                "Complex loop patterns not supported".to_string(),
                            ));
                        }
                    };

                    let elem_ty = elem_types
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| Type::Constructed(TypeName::Str("unknown".into()), vec![]));
                    let i32_type = Type::Constructed(TypeName::Str("i32".into()), vec![]);

                    let extract = Expr::new(
                        elem_ty,
                        mir::ExprKind::Intrinsic {
                            name: "tuple_access".to_string(),
                            args: vec![
                                Expr::new(tuple_ty.clone(), mir::ExprKind::Var(tmp.clone()), span),
                                Expr::new(
                                    i32_type,
                                    mir::ExprKind::Literal(mir::Literal::Int(i.to_string())),
                                    span,
                                ),
                            ],
                        },
                        span,
                    );

                    bindings.push((name, extract));
                }

                // Prepend the tuple binding
                bindings.insert(0, (tmp, init_flat));
                Ok(bindings)
            }
            _ => Err(CompilerError::FlatteningError(format!(
                "Loop pattern {:?} not supported",
                pattern.kind
            ))),
        }
    }

    /// Find free variables in an expression
    fn find_free_variables(&self, expr: &Expression, bound: &HashSet<String>) -> HashSet<String> {
        let mut free = HashSet::new();
        self.collect_free_vars(expr, bound, &mut free);
        free
    }

    fn collect_free_vars(&self, expr: &Expression, bound: &HashSet<String>, free: &mut HashSet<String>) {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                if !bound.contains(name) {
                    free.insert(name.clone());
                }
            }
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::TypeHole => {}
            ExprKind::QualifiedName(_, _) => {}
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.collect_free_vars(lhs, bound, free);
                self.collect_free_vars(rhs, bound, free);
            }
            ExprKind::UnaryOp(_, operand) => {
                self.collect_free_vars(operand, bound, free);
            }
            ExprKind::If(if_expr) => {
                self.collect_free_vars(&if_expr.condition, bound, free);
                self.collect_free_vars(&if_expr.then_branch, bound, free);
                self.collect_free_vars(&if_expr.else_branch, bound, free);
            }
            ExprKind::LetIn(let_in) => {
                self.collect_free_vars(&let_in.value, bound, free);
                let mut extended = bound.clone();
                for name in pattern::bound_names(&let_in.pattern) {
                    extended.insert(name);
                }
                self.collect_free_vars(&let_in.body, &extended, free);
            }
            ExprKind::Lambda(lambda) => {
                let mut extended = bound.clone();
                for param in &lambda.params {
                    if let Some(name) = param.simple_name() {
                        extended.insert(name.to_string());
                    }
                }
                self.collect_free_vars(&lambda.body, &extended, free);
            }
            ExprKind::Application(func, args) => {
                self.collect_free_vars(func, bound, free);
                for arg in args {
                    self.collect_free_vars(arg, bound, free);
                }
            }
            ExprKind::FunctionCall(_, args) => {
                for arg in args {
                    self.collect_free_vars(arg, bound, free);
                }
            }
            ExprKind::Tuple(elems) | ExprKind::ArrayLiteral(elems) => {
                for elem in elems {
                    self.collect_free_vars(elem, bound, free);
                }
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, expr) in fields {
                    self.collect_free_vars(expr, bound, free);
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.collect_free_vars(arr, bound, free);
                self.collect_free_vars(idx, bound, free);
            }
            ExprKind::FieldAccess(obj, _) => {
                self.collect_free_vars(obj, bound, free);
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &loop_expr.init {
                    self.collect_free_vars(init, bound, free);
                }
                let mut extended = bound.clone();
                for name in pattern::bound_names(&loop_expr.pattern) {
                    extended.insert(name);
                }
                match &loop_expr.form {
                    ast::LoopForm::While(cond) => {
                        self.collect_free_vars(cond, &extended, free);
                    }
                    ast::LoopForm::For(var, bound_expr) => {
                        extended.insert(var.clone());
                        self.collect_free_vars(bound_expr, &extended, free);
                    }
                    ast::LoopForm::ForIn(pat, iter) => {
                        self.collect_free_vars(iter, bound, free);
                        for name in pattern::bound_names(pat) {
                            extended.insert(name);
                        }
                    }
                }
                self.collect_free_vars(&loop_expr.body, &extended, free);
            }
            ExprKind::Pipe(lhs, rhs) => {
                self.collect_free_vars(lhs, bound, free);
                self.collect_free_vars(rhs, bound, free);
            }
            ExprKind::TypeAscription(inner, _)
            | ExprKind::TypeCoercion(inner, _)
            | ExprKind::Unsafe(inner) => {
                self.collect_free_vars(inner, bound, free);
            }
            ExprKind::Assert(cond, body) => {
                self.collect_free_vars(cond, bound, free);
                self.collect_free_vars(body, bound, free);
            }
            ExprKind::Match(match_expr) => {
                self.collect_free_vars(&match_expr.scrutinee, bound, free);
                for case in &match_expr.cases {
                    let mut extended = bound.clone();
                    for name in pattern::bound_names(&case.pattern) {
                        extended.insert(name);
                    }
                    self.collect_free_vars(&case.body, &extended, free);
                }
            }
            ExprKind::Range(range) => {
                self.collect_free_vars(&range.start, bound, free);
                self.collect_free_vars(&range.end, bound, free);
                if let Some(step) = &range.step {
                    self.collect_free_vars(step, bound, free);
                }
            }
        }
    }

    /// Rewrite free variable references to access from closure
    fn rewrite_free_vars(&self, expr: &Expression, free_vars: &HashSet<String>) -> Result<Expression> {
        let span = expr.h.span;
        let kind = match &expr.kind {
            ExprKind::Identifier(name) if free_vars.contains(name) => {
                // Rewrite to __closure.name
                ExprKind::FieldAccess(
                    Box::new(Expression {
                        h: expr.h.clone(),
                        kind: ExprKind::Identifier("__closure".to_string()),
                    }),
                    name.clone(),
                )
            }
            ExprKind::Identifier(_)
            | ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::TypeHole
            | ExprKind::QualifiedName(_, _) => {
                return Ok(expr.clone());
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let lhs = self.rewrite_free_vars(lhs, free_vars)?;
                let rhs = self.rewrite_free_vars(rhs, free_vars)?;
                ExprKind::BinaryOp(op.clone(), Box::new(lhs), Box::new(rhs))
            }
            ExprKind::UnaryOp(op, operand) => {
                let operand = self.rewrite_free_vars(operand, free_vars)?;
                ExprKind::UnaryOp(op.clone(), Box::new(operand))
            }
            ExprKind::If(if_expr) => {
                let condition = self.rewrite_free_vars(&if_expr.condition, free_vars)?;
                let then_branch = self.rewrite_free_vars(&if_expr.then_branch, free_vars)?;
                let else_branch = self.rewrite_free_vars(&if_expr.else_branch, free_vars)?;
                ExprKind::If(ast::IfExpr {
                    condition: Box::new(condition),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                })
            }
            ExprKind::LetIn(let_in) => {
                let value = self.rewrite_free_vars(&let_in.value, free_vars)?;
                // Remove bound names from free_vars for body
                let mut body_free = free_vars.clone();
                for name in pattern::bound_names(&let_in.pattern) {
                    body_free.remove(&name);
                }
                let body = self.rewrite_free_vars(&let_in.body, &body_free)?;
                ExprKind::LetIn(ast::LetInExpr {
                    pattern: let_in.pattern.clone(),
                    ty: let_in.ty.clone(),
                    value: Box::new(value),
                    body: Box::new(body),
                })
            }
            ExprKind::FunctionCall(name, args) => {
                let args: Result<Vec<_>> =
                    args.iter().map(|a| self.rewrite_free_vars(a, free_vars)).collect();
                ExprKind::FunctionCall(name.clone(), args?)
            }
            ExprKind::Application(func, args) => {
                let func = self.rewrite_free_vars(func, free_vars)?;
                let args: Result<Vec<_>> =
                    args.iter().map(|a| self.rewrite_free_vars(a, free_vars)).collect();
                ExprKind::Application(Box::new(func), args?)
            }
            ExprKind::Tuple(elems) => {
                let elems: Result<Vec<_>> =
                    elems.iter().map(|e| self.rewrite_free_vars(e, free_vars)).collect();
                ExprKind::Tuple(elems?)
            }
            ExprKind::ArrayLiteral(elems) => {
                let elems: Result<Vec<_>> =
                    elems.iter().map(|e| self.rewrite_free_vars(e, free_vars)).collect();
                ExprKind::ArrayLiteral(elems?)
            }
            ExprKind::RecordLiteral(fields) => {
                let fields: Result<Vec<_>> = fields
                    .iter()
                    .map(|(n, e)| Ok((n.clone(), self.rewrite_free_vars(e, free_vars)?)))
                    .collect();
                ExprKind::RecordLiteral(fields?)
            }
            ExprKind::ArrayIndex(arr, idx) => {
                let arr = self.rewrite_free_vars(arr, free_vars)?;
                let idx = self.rewrite_free_vars(idx, free_vars)?;
                ExprKind::ArrayIndex(Box::new(arr), Box::new(idx))
            }
            ExprKind::FieldAccess(obj, field) => {
                let obj = self.rewrite_free_vars(obj, free_vars)?;
                ExprKind::FieldAccess(Box::new(obj), field.clone())
            }
            // For other cases, just clone (they'll be handled if needed)
            _ => return Ok(expr.clone()),
        };

        Ok(Expression {
            h: expr.h.clone(),
            kind,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    fn flatten_program(input: &str) -> mir::Program {
        let tokens = tokenize(input).expect("Tokenization failed");
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().expect("Parsing failed");
        let type_table = HashMap::new(); // Empty - tests don't use field access
        let mut flattener = Flattener::new(type_table);
        flattener.flatten_program(&ast).expect("Flattening failed")
    }

    fn flatten_to_string(input: &str) -> String {
        format!("{}", flatten_program(input))
    }

    #[test]
    fn test_simple_constant() {
        let mir = flatten_to_string("def x = 42");
        assert!(mir.contains("def x ="));
        assert!(mir.contains("42"));
    }

    #[test]
    fn test_simple_function() {
        let mir = flatten_to_string("def add x y = x + y");
        assert!(mir.contains("def add x y ="));
        assert!(mir.contains("(x + y)"));
    }

    #[test]
    fn test_let_binding() {
        let mir = flatten_to_string("def f = let x = 1 in x + 2");
        assert!(mir.contains("let x = 1 in"));
    }

    #[test]
    fn test_tuple_pattern() {
        let mir = flatten_to_string("def f = let (a, b) = (1, 2) in a + b");
        // Should generate tuple extraction
        assert!(mir.contains("tuple_access"));
    }

    #[test]
    fn test_lambda_defunctionalization() {
        let mir = flatten_program("def f = \\x -> x + 1");
        // Should generate a lambda function
        assert!(mir.defs.len() >= 2); // Original + lambda

        // Check that closure record is created
        let mir_str = format!("{}", mir);
        assert!(mir_str.contains("__lam_f_"));
        assert!(mir_str.contains("__tag"));
    }

    #[test]
    fn test_lambda_with_capture() {
        let mir = flatten_program("def f y = let g = \\x -> x + y in g 1");
        let mir_str = format!("{}", mir);

        // Lambda should capture y
        assert!(mir_str.contains("__closure"));
        // Should reference y from closure
        assert!(mir_str.contains("record_access") || mir_str.contains("__closure"));
    }

    #[test]
    fn test_nested_let() {
        let mir = flatten_to_string("def f = let x = 1 in let y = 2 in x + y");
        assert!(mir.contains("let x = 1"));
        assert!(mir.contains("let y = 2"));
    }

    #[test]
    fn test_if_expression() {
        let mir = flatten_to_string("def f x = if x then 1 else 0");
        assert!(mir.contains("if x then 1 else 0"));
    }

    #[test]
    fn test_function_call() {
        let mir = flatten_to_string("def f x = g(x, 1)");
        // g(x, 1) in source becomes g (x, 1) - call with tuple argument
        assert!(mir.contains("g (x, 1)"));
    }

    #[test]
    fn test_array_literal() {
        let mir = flatten_to_string("def arr = [1, 2, 3]");
        assert!(mir.contains("[1, 2, 3]"));
    }

    #[test]
    fn test_record_literal() {
        let mir = flatten_to_string("def r = {x: 1, y: 2}");
        assert!(mir.contains("x=1"));
        assert!(mir.contains("y=2"));
    }

    #[test]
    fn test_while_loop() {
        let mir = flatten_to_string("def f = loop x = 0 while x < 10 do x + 1");
        assert!(mir.contains("loop"));
        assert!(mir.contains("while"));
    }

    #[test]
    fn test_for_range_loop() {
        let mir = flatten_to_string("def f = loop acc = 0 for i < 10 do acc + i");
        assert!(mir.contains("loop"));
        assert!(mir.contains("for i <"));
    }

    #[test]
    fn test_binary_ops() {
        let mir = flatten_to_string("def f x y = x * y + x / y");
        assert!(mir.contains("*"));
        assert!(mir.contains("+"));
        assert!(mir.contains("/"));
    }

    #[test]
    fn test_unary_op() {
        let mir = flatten_to_string("def f x = -x");
        assert!(mir.contains("(-x)"));
    }

    #[test]
    fn test_array_index() {
        let mir = flatten_to_string("def f arr i = arr[i]");
        assert!(mir.contains("index"));
    }

    #[test]
    fn test_multiple_lambdas() {
        let mir = flatten_program(
            r#"
def f =
    let a = \x -> x + 1 in
    let b = \y -> y * 2 in
    (a, b)
"#,
        );
        // Should have original + 2 lambdas
        assert!(mir.defs.len() >= 3);
    }
}
