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
use crate::scope::ScopeStack;
use polytype::TypeScheme;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Static values for defunctionalization (Futhark TFP'18 approach).
/// Tracks what each expression evaluates to at compile time.
#[derive(Debug, Clone)]
enum StaticValue {
    /// Dynamic runtime value
    Dyn,
    /// Defunctionalized closure with known call target
    Closure {
        /// Runtime tag (index in lambda_registry)
        tag: i32,
        /// Name of the generated lambda function
        lam_name: String,
        /// Number of parameters (excluding closure)
        arity: usize,
    },
}

/// Flattener converts AST to MIR with defunctionalization.
pub struct Flattener {
    /// Counter for generating unique names
    next_id: usize,
    /// Generated lambda functions (collected during flattening)
    generated_functions: Vec<mir::Def>,
    /// Stack of enclosing declaration names for lambda naming
    enclosing_decl_stack: Vec<String>,
    /// Registry of lambdas: tag index -> (function_name, arity)
    /// Tags are assigned sequentially starting from 0.
    lambda_registry: Vec<(String, usize)>,
    /// Type table from type checking - maps NodeId to TypeScheme
    type_table: HashMap<NodeId, TypeScheme<TypeName>>,
    /// Scope stack for tracking static values of variables
    static_values: ScopeStack<StaticValue>,
    /// Set of builtin names to exclude from free variable capture
    builtins: HashSet<String>,
    /// Stack of closure types (for nested lambdas)
    closure_type_stack: Vec<Type>,
}

impl Flattener {
    pub fn new(type_table: HashMap<NodeId, TypeScheme<TypeName>>, builtins: HashSet<String>) -> Self {
        Flattener {
            next_id: 0,
            generated_functions: Vec::new(),
            enclosing_decl_stack: Vec::new(),
            lambda_registry: Vec::new(),
            type_table,
            static_values: ScopeStack::new(),
            builtins,
            closure_type_stack: Vec::new(),
        }
    }

    /// Register a lambda function and return its tag.
    fn add_lambda(&mut self, func_name: String, arity: usize) -> i32 {
        let tag = self.lambda_registry.len() as i32;
        self.lambda_registry.push((func_name, arity));
        tag
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
                    // Add this function name to builtins so it won't be captured in lambdas
                    self.builtins.insert(d.name.clone());
                    self.enclosing_decl_stack.push(d.name.clone());

                    let def = if d.params.is_empty() {
                        // Constant
                        let (body, _) = self.flatten_expr(&d.body)?;
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
                        let (body, _) = self.flatten_expr(&d.body)?;
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
                }
                ast::Declaration::Entry(e) => {
                    self.enclosing_decl_stack.push(e.name.clone());

                    let params = self.flatten_params(&e.params)?;
                    let param_attrs = self.extract_param_attributes(&e.params);
                    let (body, _) = self.flatten_expr(&e.body)?;
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

        Ok(mir::Program {
            defs,
            lambda_registry: self.lambda_registry.clone(),
        })
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

    /// Flatten an expression, returning the MIR expression and its static value
    fn flatten_expr(&mut self, expr: &Expression) -> Result<(Expr, StaticValue)> {
        let span = expr.h.span;
        let ty = self.get_expr_type(expr);
        let (kind, sv) = match &expr.kind {
            ExprKind::IntLiteral(n) => (
                mir::ExprKind::Literal(mir::Literal::Int(n.to_string())),
                StaticValue::Dyn,
            ),
            ExprKind::FloatLiteral(f) => (
                mir::ExprKind::Literal(mir::Literal::Float(f.to_string())),
                StaticValue::Dyn,
            ),
            ExprKind::BoolLiteral(b) => (mir::ExprKind::Literal(mir::Literal::Bool(*b)), StaticValue::Dyn),
            ExprKind::Identifier(name) => {
                // Look up static value for this variable
                let sv = self.static_values.lookup(name).ok().cloned().unwrap_or(StaticValue::Dyn);
                (mir::ExprKind::Var(name.clone()), sv)
            }
            ExprKind::QualifiedName(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                (mir::ExprKind::Var(full_name), StaticValue::Dyn)
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let (lhs, _) = self.flatten_expr(lhs)?;
                let (rhs, _) = self.flatten_expr(rhs)?;
                (
                    mir::ExprKind::BinOp {
                        op: op.op.clone(),
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::UnaryOp(op, operand) => {
                let (operand, _) = self.flatten_expr(operand)?;
                (
                    mir::ExprKind::UnaryOp {
                        op: op.op.clone(),
                        operand: Box::new(operand),
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::If(if_expr) => {
                let (cond, _) = self.flatten_expr(&if_expr.condition)?;
                let (then_branch, _) = self.flatten_expr(&if_expr.then_branch)?;
                let (else_branch, _) = self.flatten_expr(&if_expr.else_branch)?;
                (
                    mir::ExprKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::LetIn(let_in) => self.flatten_let_in(let_in, span)?,
            ExprKind::Lambda(lambda) => self.flatten_lambda(lambda, span)?,
            ExprKind::Application(func, args) => self.flatten_application(func, args, span)?,
            ExprKind::Tuple(elems) => {
                let elems: Result<Vec<_>> =
                    elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Tuple(elems?)),
                    StaticValue::Dyn,
                )
            }
            ExprKind::ArrayLiteral(elems) => {
                let elems: Result<Vec<_>> =
                    elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Array(elems?)),
                    StaticValue::Dyn,
                )
            }
            ExprKind::RecordLiteral(fields) => {
                let fields: Result<Vec<_>> = fields
                    .iter()
                    .map(|(name, expr)| Ok((name.clone(), self.flatten_expr(expr)?.0)))
                    .collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Record(fields?)),
                    StaticValue::Dyn,
                )
            }
            ExprKind::ArrayIndex(arr, idx) => {
                let (arr, _) = self.flatten_expr(arr)?;
                let (idx, _) = self.flatten_expr(idx)?;
                (
                    mir::ExprKind::Intrinsic {
                        name: "index".to_string(),
                        args: vec![arr, idx],
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::FieldAccess(obj_expr, field) => {
                // Check for special cases when obj is an identifier
                if let ExprKind::Identifier(name) = &obj_expr.kind {
                    // Special case: __closure field access (from lambda free var rewriting)
                    if name == "__closure" {
                        // Use the current closure type from the stack (most recent lambda)
                        let closure_type = self.closure_type_stack.last().cloned().ok_or_else(|| {
                            CompilerError::FlatteningError(
                                "Internal error: __closure accessed outside of lambda body".to_string(),
                            )
                        })?;
                        let obj =
                            Expr::new(closure_type, mir::ExprKind::Var("__closure".to_string()), span);
                        return Ok((
                            Expr::new(
                                ty,
                                mir::ExprKind::Intrinsic {
                                    name: "record_access".to_string(),
                                    args: vec![
                                        obj,
                                        Expr::new(
                                            Type::Constructed(TypeName::Str("string".into()), vec![]),
                                            mir::ExprKind::Literal(mir::Literal::String(field.clone())),
                                            span,
                                        ),
                                    ],
                                },
                                span,
                            ),
                            StaticValue::Dyn,
                        ));
                    }

                    // Check if identifier has a type (i.e., it's a variable)
                    // If not, it's a qualified name like f32.sqrt
                    if self.type_table.get(&obj_expr.h.id).is_none() {
                        let full_name = format!("{}.{}", name, field);
                        return Ok((
                            Expr::new(ty, mir::ExprKind::Var(full_name), span),
                            StaticValue::Dyn,
                        ));
                    }
                }

                let (obj, _) = self.flatten_expr(obj_expr)?;

                // Resolve field name to index using type information
                let idx = self.resolve_field_index(obj_expr, field)?;

                // Create i32 type for the index literal
                let i32_type = Type::Constructed(TypeName::Str("i32".into()), vec![]);

                (
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
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::Loop(loop_expr) => self.flatten_loop(loop_expr, span)?,
            ExprKind::Pipe(lhs, rhs) => {
                // a |> f  =>  f(a)
                let (lhs_flat, _) = self.flatten_expr(lhs)?;
                // Treat rhs as a function to apply to lhs
                match &rhs.kind {
                    ExprKind::Identifier(name) => (
                        mir::ExprKind::Call {
                            func: name.clone(),
                            args: vec![lhs_flat],
                        },
                        StaticValue::Dyn,
                    ),
                    _ => {
                        // General case: application
                        let (_rhs_flat, _) = self.flatten_expr(rhs)?;
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
                let (inner, _) = self.flatten_expr(inner)?;
                (
                    mir::ExprKind::Attributed {
                        attributes: vec![mir::Attribute {
                            name: "unsafe".to_string(),
                            args: vec![],
                        }],
                        expr: Box::new(inner),
                    },
                    StaticValue::Dyn,
                )
            }
            ExprKind::Assert(cond, body) => {
                let (cond, _) = self.flatten_expr(cond)?;
                let (body, _) = self.flatten_expr(body)?;
                (
                    mir::ExprKind::Intrinsic {
                        name: "assert".to_string(),
                        args: vec![cond, body],
                    },
                    StaticValue::Dyn,
                )
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

        Ok((Expr::new(ty, kind, span), sv))
    }

    /// Flatten a let-in expression, handling pattern destructuring
    fn flatten_let_in(
        &mut self,
        let_in: &ast::LetInExpr,
        span: Span,
    ) -> Result<(mir::ExprKind, StaticValue)> {
        let (value, value_sv) = self.flatten_expr(&let_in.value)?;

        // Check if pattern is simple (just a name)
        match &let_in.pattern.kind {
            PatternKind::Name(name) => {
                // Track the static value of this binding
                self.static_values.push_scope();
                self.static_values.insert(name.clone(), value_sv);

                let (body, body_sv) = self.flatten_expr(&let_in.body)?;

                self.static_values.pop_scope();

                Ok((
                    mir::ExprKind::Let {
                        name: name.clone(),
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    body_sv,
                ))
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
                let (body, body_sv) = self.flatten_expr(&let_in.body)?;
                Ok((
                    mir::ExprKind::Let {
                        name: self.fresh_name("ignored"),
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    body_sv,
                ))
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
                let (mut body, body_sv) = self.flatten_expr(&let_in.body)?;

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
                Ok((
                    mir::ExprKind::Let {
                        name: tmp,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    body_sv,
                ))
            }
            _ => Err(CompilerError::FlatteningError(format!(
                "Pattern kind {:?} not yet supported in let",
                let_in.pattern.kind
            ))),
        }
    }

    /// Flatten a lambda expression (defunctionalization)
    fn flatten_lambda(
        &mut self,
        lambda: &ast::LambdaExpr,
        span: Span,
    ) -> Result<(mir::ExprKind, StaticValue)> {
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

        // Register lambda and get tag
        let arity = lambda.params.len();
        let tag = self.add_lambda(func_name.clone(), arity);

        // Build the closure record fields first so we can construct the type
        let i32_type = Type::Constructed(TypeName::Str("i32".into()), vec![]);
        let mut record_fields = vec![(
            "__tag".to_string(),
            Expr::new(
                i32_type.clone(),
                mir::ExprKind::Literal(mir::Literal::Int(tag.to_string())),
                span,
            ),
        )];

        let mut sorted_vars: Vec<_> = free_vars.iter().collect();
        sorted_vars.sort();
        for var in &sorted_vars {
            // Look up the type of the free variable from the lambda body
            // We need to find an Identifier node with this name and get its type
            let var_type = self
                .find_var_type_in_expr(&lambda.body, var)
                .unwrap_or_else(|| Type::Constructed(TypeName::Str("unknown".into()), vec![]));
            record_fields.push((
                (*var).clone(),
                Expr::new(var_type, mir::ExprKind::Var((*var).clone()), span),
            ));
        }

        // Build the record type from the fields
        let mut type_fields = BTreeMap::new();
        for (name, expr) in &record_fields {
            type_fields.insert(name.clone(), expr.ty.clone());
        }
        let closure_type = Type::Constructed(TypeName::Record(type_fields), vec![]);

        // Build parameters: closure first, then lambda params
        let mut params = vec![mir::Param {
            name: "__closure".to_string(),
            ty: closure_type.clone(),
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

        // Push closure type onto stack before flattening body (for nested lambdas)
        self.closure_type_stack.push(closure_type.clone());
        let (body, _) = self.flatten_expr(&rewritten_body)?;
        self.closure_type_stack.pop();

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

        // Return the closure record along with the static value indicating it's a known closure
        let sv = StaticValue::Closure {
            tag,
            lam_name: func_name,
            arity,
        };

        Ok((mir::ExprKind::Literal(mir::Literal::Record(record_fields)), sv))
    }

    /// Find the type of a variable by searching for its use in an expression
    fn find_var_type_in_expr(&self, expr: &Expression, var_name: &str) -> Option<Type> {
        match &expr.kind {
            ExprKind::Identifier(name) if name == var_name => Some(self.get_expr_type(expr)),
            ExprKind::BinaryOp(_, lhs, rhs) => self
                .find_var_type_in_expr(lhs, var_name)
                .or_else(|| self.find_var_type_in_expr(rhs, var_name)),
            ExprKind::UnaryOp(_, operand) => self.find_var_type_in_expr(operand, var_name),
            ExprKind::If(if_expr) => self
                .find_var_type_in_expr(&if_expr.condition, var_name)
                .or_else(|| self.find_var_type_in_expr(&if_expr.then_branch, var_name))
                .or_else(|| self.find_var_type_in_expr(&if_expr.else_branch, var_name)),
            ExprKind::LetIn(let_in) => self
                .find_var_type_in_expr(&let_in.value, var_name)
                .or_else(|| self.find_var_type_in_expr(&let_in.body, var_name)),
            ExprKind::Application(func, args) => self
                .find_var_type_in_expr(func, var_name)
                .or_else(|| args.iter().find_map(|a| self.find_var_type_in_expr(a, var_name))),
            ExprKind::Tuple(elems) | ExprKind::ArrayLiteral(elems) => {
                elems.iter().find_map(|e| self.find_var_type_in_expr(e, var_name))
            }
            ExprKind::ArrayIndex(arr, idx) => self
                .find_var_type_in_expr(arr, var_name)
                .or_else(|| self.find_var_type_in_expr(idx, var_name)),
            ExprKind::FieldAccess(obj, _) => self.find_var_type_in_expr(obj, var_name),
            ExprKind::RecordLiteral(fields) => {
                fields.iter().find_map(|(_, e)| self.find_var_type_in_expr(e, var_name))
            }
            ExprKind::Lambda(lambda) => self.find_var_type_in_expr(&lambda.body, var_name),
            _ => None,
        }
    }

    /// Flatten an application expression
    fn flatten_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
        _span: Span,
    ) -> Result<(mir::ExprKind, StaticValue)> {
        let (func_flat, func_sv) = self.flatten_expr(func)?;
        let args_flat: Result<Vec<_>> = args.iter().map(|a| self.flatten_expr(a).map(|(e, _)| e)).collect();
        let args_flat = args_flat?;

        // Check if this is applying a known function name
        match &func.kind {
            ExprKind::Identifier(name) => {
                // Check if the identifier is bound to a known closure
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    // Direct call to the lambda function with closure as first argument
                    let mut all_args = vec![func_flat];
                    all_args.extend(args_flat);
                    Ok((
                        mir::ExprKind::Call {
                            func: lam_name,
                            args: all_args,
                        },
                        StaticValue::Dyn,
                    ))
                } else {
                    // Direct function call (not a closure)
                    Ok((
                        mir::ExprKind::Call {
                            func: name.clone(),
                            args: args_flat,
                        },
                        StaticValue::Dyn,
                    ))
                }
            }
            // Handle qualified names like f32.sqrt
            ExprKind::FieldAccess(obj, field) => {
                if let ExprKind::Identifier(module) = &obj.kind {
                    // Check if this is a qualified name (no type in type_table means it's a type/module name)
                    if self.type_table.get(&obj.h.id).is_none() {
                        let full_name = format!("{}.{}", module, field);
                        return Ok((
                            mir::ExprKind::Call {
                                func: full_name,
                                args: args_flat,
                            },
                            StaticValue::Dyn,
                        ));
                    }
                }
                // Not a qualified name, fall through to closure handling
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    let mut all_args = vec![func_flat];
                    all_args.extend(args_flat);
                    Ok((
                        mir::ExprKind::Call {
                            func: lam_name,
                            args: all_args,
                        },
                        StaticValue::Dyn,
                    ))
                } else {
                    Err(CompilerError::FlatteningError(format!(
                        "Cannot call closure with unknown static value (field access). \
                         Function expression: {:?}",
                        func.kind
                    )))
                }
            }
            _ => {
                // Closure call: check if we know the static value
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    // Direct call to the lambda function with closure as first argument
                    let mut all_args = vec![func_flat];
                    all_args.extend(args_flat);
                    Ok((
                        mir::ExprKind::Call {
                            func: lam_name,
                            args: all_args,
                        },
                        StaticValue::Dyn,
                    ))
                } else {
                    // Unknown closure - this should not happen with proper function value restrictions
                    Err(CompilerError::FlatteningError(format!(
                        "Cannot call closure with unknown static value. \
                         Function expression: {:?}",
                        func.kind
                    )))
                }
            }
        }
    }

    /// Flatten a loop expression
    fn flatten_loop(
        &mut self,
        loop_expr: &ast::LoopExpr,
        span: Span,
    ) -> Result<(mir::ExprKind, StaticValue)> {
        // Extract init bindings from pattern
        let init_bindings =
            self.extract_loop_bindings(&loop_expr.pattern, loop_expr.init.as_deref(), span)?;

        // Flatten loop kind
        let kind = match &loop_expr.form {
            ast::LoopForm::While(cond) => {
                let (cond, _) = self.flatten_expr(cond)?;
                mir::LoopKind::While { cond: Box::new(cond) }
            }
            ast::LoopForm::For(var, bound) => {
                let (bound, _) = self.flatten_expr(bound)?;
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
                let (iter, _) = self.flatten_expr(iter)?;
                mir::LoopKind::For {
                    var,
                    iter: Box::new(iter),
                }
            }
        };

        let (body, _) = self.flatten_expr(&loop_expr.body)?;

        Ok((
            mir::ExprKind::Loop {
                init_bindings,
                kind,
                body: Box::new(body),
            },
            StaticValue::Dyn,
        ))
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
                let (init_flat, _) = self.flatten_expr(init_expr)?;
                Ok(vec![(name.clone(), init_flat)])
            }
            PatternKind::Typed(inner, _) => self.extract_loop_bindings(inner, init, span),
            PatternKind::Tuple(patterns) => {
                // Init should also be a tuple
                let (init_flat, _) = self.flatten_expr(init_expr)?;
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
                if !bound.contains(name) && !self.builtins.contains(name) {
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
