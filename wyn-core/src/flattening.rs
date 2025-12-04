//! Flattening pass: AST -> MIR
//!
//! This pass performs:
//! - Defunctionalization: lambdas become top-level functions with closure records
//! - Pattern flattening: complex patterns become simple let bindings
//! - Lambda lifting: all functions become top-level Def entries

use crate::ast::{self, ExprKind, Expression, NodeId, PatternKind, Span, Type, TypeName, types};
use crate::error::{CompilerError, Result};
use crate::mir::{self, Expr};
use crate::pattern;
use crate::scope::ScopeStack;
use polytype::TypeScheme;
use std::collections::{HashMap, HashSet};

/// Shape classification for desugaring decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgShape {
    Matrix, // mat<n,m,a>
    Vector, // Vec<n,a>
    Other,
}

/// Static values for defunctionalization (Futhark TFP'18 approach).
/// Tracks what each expression evaluates to at compile time.
/// Each variant includes a binding_id for tracking backing stores.
#[derive(Debug, Clone)]
enum StaticValue {
    /// Dynamic runtime value with its binding ID
    Dyn {
        binding_id: u64,
    },
    /// Defunctionalized closure with known call target
    Closure {
        /// Name of the generated lambda function
        lam_name: String,
        /// Binding ID for this closure
        binding_id: u64,
    },
}

/// Flattener converts AST to MIR with defunctionalization.
pub struct Flattener {
    /// Counter for generating unique names
    next_id: usize,
    /// Counter for generating unique binding IDs
    next_binding_id: u64,
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
    /// Set of binding IDs that need backing stores (materialization)
    needs_backing_store: HashSet<u64>,
}

impl Flattener {
    pub fn new(type_table: HashMap<NodeId, TypeScheme<TypeName>>, builtins: HashSet<String>) -> Self {
        Flattener {
            next_id: 0,
            next_binding_id: 0,
            generated_functions: Vec::new(),
            enclosing_decl_stack: Vec::new(),
            lambda_registry: Vec::new(),
            type_table,
            static_values: ScopeStack::new(),
            builtins,
            closure_type_stack: Vec::new(),
            needs_backing_store: HashSet::new(),
        }
    }

    /// Generate a fresh binding ID
    fn fresh_binding_id(&mut self) -> u64 {
        let id = self.next_binding_id;
        self.next_binding_id += 1;
        id
    }

    /// Get the backing store variable name for a binding ID
    fn backing_store_name(binding_id: u64) -> String {
        format!("__ptr_{}", binding_id)
    }

    /// Register a lambda function.
    fn add_lambda(&mut self, func_name: String, arity: usize) {
        self.lambda_registry.push((func_name, arity));
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
                eprintln!("BUG: Expression (id={:?}) has no type in type table", expr.h.id);
                eprintln!("Expression kind: {:?}", expr.kind);
                eprintln!("Expression span: {:?}", expr.h.span);
                panic!("BUG: Expression (id={:?}) has no type in type table during flattening. Type checking should ensure all expressions have types.", expr.h.id)
            })
    }

    /// Desugar overloaded function names based on argument types
    /// - mul -> mul_mat_mat, mul_mat_vec, mul_vec_mat
    /// - matav -> matav_n_m
    fn desugar_function_name(&self, name: &str, args: &[Expression]) -> Result<String> {
        match name {
            "mul" => self.desugar_mul(args),
            "matav" => self.desugar_matav(args),
            _ => Ok(name.to_string()),
        }
    }

    /// Desugar mul based on argument shapes
    fn desugar_mul(&self, args: &[Expression]) -> Result<String> {
        if args.len() != 2 {
            return Ok("mul".to_string()); // Let type checker handle the error
        }

        let arg1_ty = self.get_expr_type(&args[0]);
        let arg2_ty = self.get_expr_type(&args[1]);

        let shape1 = Self::classify_shape(&arg1_ty);
        let shape2 = Self::classify_shape(&arg2_ty);

        let variant = match (shape1, shape2) {
            (ArgShape::Matrix, ArgShape::Matrix) => "mul_mat_mat",
            (ArgShape::Matrix, ArgShape::Vector) => "mul_mat_vec",
            (ArgShape::Vector, ArgShape::Matrix) => "mul_vec_mat",
            _ => "mul", // Fall back to original name
        };

        Ok(variant.to_string())
    }

    /// Desugar matav based on array and vector dimensions
    fn desugar_matav(&self, args: &[Expression]) -> Result<String> {
        if args.len() != 1 {
            return Ok("matav".to_string()); // Let type checker handle the error
        }

        let arg_ty = self.get_expr_type(&args[0]);

        // Extract array size n, vector size m, and element type a from [n]vec<m,a>
        if let Type::Constructed(TypeName::Array, array_args) = &arg_ty {
            if array_args.len() >= 2 {
                if let Type::Constructed(TypeName::Vec, vec_args) = &array_args[1] {
                    if vec_args.len() >= 2 {
                        if let (Some(n), Some(m)) = (
                            Self::extract_size(&array_args[0]),
                            Self::extract_size(&vec_args[0]),
                        ) {
                            // Extract element type
                            let elem_type_str = Self::primitive_type_to_string(&vec_args[1])?;
                            return Ok(format!("matav_{}_{}_{}", n, m, elem_type_str));
                        }
                    }
                }
            }
        }

        Ok("matav".to_string()) // Fall back to original name
    }

    /// Classify argument shape for desugaring
    fn classify_shape(ty: &Type) -> ArgShape {
        match ty {
            Type::Constructed(TypeName::Mat, _) => ArgShape::Matrix,
            Type::Constructed(TypeName::Vec, _) => ArgShape::Vector,
            _ => ArgShape::Other,
        }
    }

    /// Extract concrete size from a type
    fn extract_size(ty: &Type) -> Option<usize> {
        match ty {
            Type::Constructed(TypeName::Size(n), _) => Some(*n),
            _ => None,
        }
    }

    /// Convert a primitive numeric type to a string for name mangling.
    fn primitive_type_to_string(ty: &Type) -> Result<String> {
        match ty {
            Type::Constructed(TypeName::Float(bits), _) => Ok(format!("f{}", bits)),
            Type::Constructed(TypeName::Int(bits), _) => Ok(format!("i{}", bits)),
            Type::Constructed(TypeName::UInt(bits), _) => Ok(format!("u{}", bits)),
            Type::Constructed(TypeName::Str(s), _) if *s == "bool" => Ok("bool".to_string()),
            _ => Err(CompilerError::TypeError(
                format!(
                    "Invalid element type for matrix/vector: {:?}. \
                    Only f16/f32/f64, i8/i16/i32/i64, u8/u16/u32/u64, and bool are supported.",
                    ty
                ),
                Span::dummy(),
            )),
        }
    }

    /// Get the type of an AST pattern from the type table
    fn get_pattern_type(&self, pat: &ast::Pattern) -> Type {
        self.type_table
            .get(&pat.h.id)
            .and_then(|scheme| self.get_monotype(scheme))
            .cloned()
            .unwrap_or_else(|| {
                panic!("BUG: Pattern (id={:?}) has no type in type table during flattening. Type checking should ensure all patterns have types.", pat.h.id)
            })
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

    /// Hoist inner Let expressions out of a Let's value.
    /// Transforms: let x = (let y = A in B) in C  =>  let y = A in let x = B in C
    /// This ensures materialized pointers are at the same scope level as their referents.
    fn hoist_inner_lets(expr_kind: mir::ExprKind, span: Span) -> mir::ExprKind {
        if let mir::ExprKind::Let {
            name,
            binding_id,
            value,
            body,
        } = expr_kind
        {
            if let mir::ExprKind::Let {
                name: inner_name,
                binding_id: inner_binding_id,
                value: inner_value,
                body: inner_body,
            } = value.kind
            {
                // Hoist: let x = (let y = A in B) in C => let y = A in let x = B in C
                // Capture body's type before moving it
                let body_ty = body.ty.clone();
                let new_inner = mir::ExprKind::Let {
                    name,
                    binding_id,
                    value: inner_body,
                    body,
                };
                let new_inner_expr = Expr::new(body_ty.clone(), new_inner, span);
                // Recursively hoist in case there are more nested Lets
                let hoisted_inner = Self::hoist_inner_lets(new_inner_expr.kind, span);
                mir::ExprKind::Let {
                    name: inner_name,
                    binding_id: inner_binding_id,
                    value: inner_value,
                    body: Box::new(Expr::new(body_ty, hoisted_inner, span)),
                }
            } else {
                mir::ExprKind::Let {
                    name,
                    binding_id,
                    value,
                    body,
                }
            }
        } else {
            expr_kind
        }
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
                .iter()
                .enumerate()
                .find(|(_, name)| name.as_str() == field)
                .map(|(idx, _)| idx)
                .ok_or_else(|| CompilerError::FlatteningError(format!("Unknown record field: {}", field))),
            // Tuple types: should use numeric access
            Type::Constructed(TypeName::Tuple(_), _) => Err(CompilerError::FlatteningError(format!(
                "Tuple access must use numeric index, not '{}'",
                field
            ))),
            _ => Err(CompilerError::FlatteningError(format!(
                "Cannot access field '{}' on type {:?}",
                field, obj_type
            ))),
        }
    }

    /// Helper to flatten a single Decl
    fn flatten_single_decl(&mut self, d: &ast::Decl, defs: &mut Vec<mir::Def>) -> Result<()> {
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
            let span = d.body.h.span;

            // Register params with binding IDs before flattening body
            let param_bindings = self.register_param_bindings(&d.params)?;

            let (body, _) = self.flatten_expr(&d.body)?;

            // Wrap body with backing stores for params that need them
            let body = self.wrap_param_backing_stores(body, param_bindings, span);

            let ret_type = self.get_expr_type(&d.body);
            mir::Def::Function {
                name: d.name.clone(),
                params,
                ret_type,
                attributes: self.convert_attributes(&d.attributes),
                param_attributes: param_attrs,
                return_attributes: vec![],
                body,
                span,
            }
        };

        // Collect generated lambdas before the definition
        defs.append(&mut self.generated_functions);
        defs.push(def);

        self.enclosing_decl_stack.pop();
        Ok(())
    }

    /// Flatten an entire program
    pub fn flatten_program(&mut self, program: &ast::Program) -> Result<mir::Program> {
        let mut defs = Vec::new();

        // Flatten library module declarations first
        for (_module_name, declarations) in &program.library_modules {
            for decl in declarations {
                if let ast::Declaration::Decl(d) = decl {
                    self.builtins.insert(d.name.clone());
                    self.flatten_single_decl(d, &mut defs)?;
                }
            }
        }

        // Flatten user declarations
        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    self.builtins.insert(d.name.clone());
                    self.flatten_single_decl(d, &mut defs)?;
                }
                ast::Declaration::Entry(e) => {
                    self.enclosing_decl_stack.push(e.name.clone());

                    let params = self.flatten_params(&e.params)?;
                    let param_attrs = self.extract_param_attributes(&e.params);
                    let span = e.body.h.span;

                    // Register params with binding IDs before flattening body
                    let param_bindings = self.register_param_bindings(&e.params)?;

                    let (body, _) = self.flatten_expr(&e.body)?;

                    // Wrap body with backing stores for params that need them
                    let body = self.wrap_param_backing_stores(body, param_bindings, span);

                    let ret_type = self.get_expr_type(&e.body);
                    let attrs = vec![if e.entry_type.is_vertex() {
                        mir::Attribute::Vertex
                    } else {
                        mir::Attribute::Fragment
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
                        span,
                    };

                    defs.append(&mut self.generated_functions);
                    defs.push(def);

                    self.enclosing_decl_stack.pop();
                }
                ast::Declaration::Uniform(uniform_decl) => {
                    // Uniforms use the declared type directly (already a Type<TypeName>)
                    defs.push(mir::Def::Uniform {
                        name: uniform_decl.name.clone(),
                        ty: uniform_decl.ty.clone(),
                    });
                }
                ast::Declaration::Sig(_)
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
            ast::Attribute::BuiltIn(builtin) => mir::Attribute::BuiltIn(*builtin),
            ast::Attribute::Location(loc) => mir::Attribute::Location(*loc),
            ast::Attribute::Vertex => mir::Attribute::Vertex,
            ast::Attribute::Fragment => mir::Attribute::Fragment,
            ast::Attribute::Uniform => mir::Attribute::Uniform,
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

    /// Register function parameters with binding IDs for backing store tracking.
    /// Returns a Vec of (param_name, param_type, binding_id) tuples.
    fn register_param_bindings(&mut self, params: &[ast::Pattern]) -> Result<Vec<(String, Type, u64)>> {
        self.static_values.push_scope();
        let mut param_bindings = Vec::new();
        for param in params {
            let name = self.extract_param_name(param)?;
            let ty = self.get_pattern_type(param);
            let binding_id = self.fresh_binding_id();
            self.static_values.insert(name.clone(), StaticValue::Dyn { binding_id });
            param_bindings.push((name, ty, binding_id));
        }
        Ok(param_bindings)
    }

    /// Wrap a function body with backing store materializations for parameters that need them.
    fn wrap_param_backing_stores(
        &mut self,
        body: Expr,
        param_bindings: Vec<(String, Type, u64)>,
        span: Span,
    ) -> Expr {
        self.static_values.pop_scope();

        // Collect params that need backing stores (in reverse order for proper nesting)
        let params_needing_stores: Vec<_> = param_bindings
            .into_iter()
            .filter(|(_, _, binding_id)| self.needs_backing_store.contains(binding_id))
            .collect();

        // Wrap body with backing stores for each param that needs one
        let mut result = body;
        for (param_name, param_ty, binding_id) in params_needing_stores.into_iter().rev() {
            let ptr_name = Self::backing_store_name(binding_id);
            let ptr_binding_id = self.fresh_binding_id();
            result = Expr::new(
                result.ty.clone(),
                mir::ExprKind::Let {
                    name: ptr_name,
                    binding_id: ptr_binding_id,
                    value: Box::new(Expr::new(
                        param_ty.clone(),
                        mir::ExprKind::Materialize(Box::new(Expr::new(
                            param_ty,
                            mir::ExprKind::Var(param_name),
                            span,
                        ))),
                        span,
                    )),
                    body: Box::new(result),
                },
                span,
            );
        }
        result
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
                StaticValue::Dyn { binding_id: 0 },
            ),
            ExprKind::FloatLiteral(f) => (
                mir::ExprKind::Literal(mir::Literal::Float(f.to_string())),
                StaticValue::Dyn { binding_id: 0 },
            ),
            ExprKind::BoolLiteral(b) => (
                mir::ExprKind::Literal(mir::Literal::Bool(*b)),
                StaticValue::Dyn { binding_id: 0 },
            ),
            ExprKind::StringLiteral(s) => (
                mir::ExprKind::Literal(mir::Literal::String(s.clone())),
                StaticValue::Dyn { binding_id: 0 },
            ),
            ExprKind::Unit => (mir::ExprKind::Unit, StaticValue::Dyn { binding_id: 0 }),
            ExprKind::Identifier(name) => {
                // Look up static value for this variable
                let sv = self
                    .static_values
                    .lookup(name)
                    .ok()
                    .cloned()
                    .unwrap_or(StaticValue::Dyn { binding_id: 0 });
                (mir::ExprKind::Var(name.clone()), sv)
            }
            ExprKind::QualifiedName(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                (mir::ExprKind::Var(full_name), StaticValue::Dyn { binding_id: 0 })
            }
            ExprKind::OperatorSection(op) => {
                // Convert operator section to a lambda: (+) becomes \x y -> x + y
                // Generate function name
                let id = self.fresh_id();
                let enclosing = self.enclosing_decl_stack.last().map(|s| s.as_str()).unwrap_or("anon");
                let func_name = format!("__op_{}_{}", enclosing, id);

                // Register lambda with arity 2 (binary operators)
                let arity = 2;
                self.add_lambda(func_name.clone(), arity);

                // Get the type of the operator section to determine parameter types
                let op_type = self.get_expr_type(expr);

                // Extract parameter types from the function type 'a -> 'a -> 'b
                let (param_type, ret_type) = if let Some((param1, rest)) = ast::types::as_arrow(&op_type) {
                    if let Some((_param2, ret)) = ast::types::as_arrow(rest) {
                        // Binary operator: a -> a -> b
                        (param1.clone(), ret.clone())
                    } else {
                        return Err(CompilerError::TypeError(
                            format!(
                                "Operator section has unexpected type structure: expected a -> a -> b, got {:?}",
                                op_type
                            ),
                            span,
                        ));
                    }
                } else {
                    return Err(CompilerError::TypeError(
                        format!("Operator section must have function type, got {:?}", op_type),
                        span,
                    ));
                };

                // Build the closure tuple with __lambda_name at the end (no free variables)
                let string_type = Type::Constructed(TypeName::Str("string".into()), vec![]);
                let tuple_elems = vec![Expr::new(
                    string_type.clone(),
                    mir::ExprKind::Literal(mir::Literal::String(func_name.clone())),
                    span,
                )];

                // Build the closure type (still a record type for field name lookup during lowering)
                // __lambda_name goes last so capture indices match SPIR-V struct indices
                let type_fields = vec![("__lambda_name".to_string(), string_type)];
                let closure_type = types::record(type_fields);

                // Build parameters: closure, x, y
                let params = vec![
                    mir::Param {
                        name: "__closure".to_string(),
                        ty: closure_type.clone(),
                        is_consumed: false,
                    },
                    mir::Param {
                        name: "x".to_string(),
                        ty: param_type.clone(),
                        is_consumed: false,
                    },
                    mir::Param {
                        name: "y".to_string(),
                        ty: param_type.clone(),
                        is_consumed: false,
                    },
                ];

                // Build the body: x <op> y
                let x_var = Expr::new(param_type.clone(), mir::ExprKind::Var("x".to_string()), span);
                let y_var = Expr::new(param_type.clone(), mir::ExprKind::Var("y".to_string()), span);
                let body = Expr::new(
                    ret_type.clone(),
                    mir::ExprKind::BinOp {
                        op: op.clone(),
                        lhs: Box::new(x_var),
                        rhs: Box::new(y_var),
                    },
                    span,
                );

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

                // Return the closure tuple with static value indicating it's a known closure
                let sv = StaticValue::Closure {
                    lam_name: func_name,
                    binding_id: 0,
                };

                (mir::ExprKind::Literal(mir::Literal::Tuple(tuple_elems)), sv)
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
                    StaticValue::Dyn { binding_id: 0 },
                )
            }
            ExprKind::UnaryOp(op, operand) => {
                let (operand, _) = self.flatten_expr(operand)?;
                (
                    mir::ExprKind::UnaryOp {
                        op: op.op.clone(),
                        operand: Box::new(operand),
                    },
                    StaticValue::Dyn { binding_id: 0 },
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
                    StaticValue::Dyn { binding_id: 0 },
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
                    StaticValue::Dyn { binding_id: 0 },
                )
            }
            ExprKind::ArrayLiteral(elems) => {
                let elems: Result<Vec<_>> =
                    elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Array(elems?)),
                    StaticValue::Dyn { binding_id: 0 },
                )
            }
            ExprKind::VecMatLiteral(elems) => {
                // Check if first element is an array literal (matrix) or scalar (vector)
                if elems.is_empty() {
                    return Err(CompilerError::FlatteningError(
                        "Empty vector/matrix literal".to_string(),
                    ));
                }

                let is_matrix = matches!(&elems[0].kind, ExprKind::ArrayLiteral(_));

                if is_matrix {
                    // Matrix: extract rows
                    let mut rows = Vec::new();
                    for elem in elems {
                        if let ExprKind::ArrayLiteral(row_elems) = &elem.kind {
                            let row: Result<Vec<_>> =
                                row_elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                            rows.push(row?);
                        } else {
                            return Err(CompilerError::FlatteningError(
                                "Matrix rows must be array literals".to_string(),
                            ));
                        }
                    }
                    (
                        mir::ExprKind::Literal(mir::Literal::Matrix(rows)),
                        StaticValue::Dyn { binding_id: 0 },
                    )
                } else {
                    // Vector
                    let elems: Result<Vec<_>> =
                        elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                    (
                        mir::ExprKind::Literal(mir::Literal::Vector(elems?)),
                        StaticValue::Dyn { binding_id: 0 },
                    )
                }
            }
            ExprKind::RecordLiteral(fields) => {
                // Records become tuples with fields in source order
                let elems: Result<Vec<_>> =
                    fields.iter().map(|(_, expr)| Ok(self.flatten_expr(expr)?.0)).collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Tuple(elems?)),
                    StaticValue::Dyn { binding_id: 0 },
                )
            }
            ExprKind::ArrayIndex(arr_expr, idx_expr) => {
                let (arr, _) = self.flatten_expr(arr_expr)?;
                let (idx, _) = self.flatten_expr(idx_expr)?;

                // Check if arr is a simple Var - if so, use backing store system
                if let mir::ExprKind::Var(ref var_name) = arr.kind {
                    // Look up binding_id from static_values
                    if let Ok(sv) = self.static_values.lookup(var_name) {
                        let binding_id = match sv {
                            StaticValue::Dyn { binding_id } => *binding_id,
                            StaticValue::Closure { binding_id, .. } => *binding_id,
                        };
                        // Mark this binding as needing a backing store
                        self.needs_backing_store.insert(binding_id);
                        // Use the backing store variable name
                        let ptr_name = Self::backing_store_name(binding_id);
                        let ptr_var = Expr::new(arr.ty.clone(), mir::ExprKind::Var(ptr_name), span);
                        let kind = mir::ExprKind::Intrinsic {
                            name: "index".to_string(),
                            args: vec![ptr_var, idx],
                        };
                        return Ok((Expr::new(ty, kind, span), StaticValue::Dyn { binding_id: 0 }));
                    }
                }

                // Fallback: wrap the array in Materialize inline
                let materialized_arr =
                    Expr::new(arr.ty.clone(), mir::ExprKind::Materialize(Box::new(arr)), span);
                (
                    mir::ExprKind::Intrinsic {
                        name: "index".to_string(),
                        args: vec![materialized_arr, idx],
                    },
                    StaticValue::Dyn { binding_id: 0 },
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

                        // Resolve field name to index from closure type
                        let idx = match &closure_type {
                            Type::Constructed(TypeName::Record(fields), _) => {
                                fields.get_index(field).ok_or_else(|| {
                                    CompilerError::FlatteningError(format!(
                                        "Unknown closure field: {}",
                                        field
                                    ))
                                })?
                            }
                            _ => {
                                return Err(CompilerError::FlatteningError(
                                    "Closure type is not a record".to_string(),
                                ));
                            }
                        };

                        let obj = Expr::new(
                            closure_type.clone(),
                            mir::ExprKind::Var("__closure".to_string()),
                            span,
                        );
                        // Wrap in Materialize for pointer access
                        let materialized_obj =
                            Expr::new(closure_type, mir::ExprKind::Materialize(Box::new(obj)), span);
                        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
                        return Ok((
                            Expr::new(
                                ty,
                                mir::ExprKind::Intrinsic {
                                    name: "tuple_access".to_string(),
                                    args: vec![
                                        materialized_obj,
                                        Expr::new(
                                            i32_type,
                                            mir::ExprKind::Literal(mir::Literal::Int(idx.to_string())),
                                            span,
                                        ),
                                    ],
                                },
                                span,
                            ),
                            StaticValue::Dyn { binding_id: 0 },
                        ));
                    }

                    // FieldAccess on identifier - this is a real record field access
                }

                let (obj, obj_sv) = self.flatten_expr(obj_expr)?;

                // Resolve field name to index using type information
                let idx = self.resolve_field_index(obj_expr, field)?;

                // Create i32 type for the index literal
                let i32_type = Type::Constructed(TypeName::Int(32), vec![]);

                // Check if obj is a simple Var - if so, use backing store system
                if let mir::ExprKind::Var(ref var_name) = obj.kind {
                    // Look up binding_id from static_values
                    if let Ok(sv) = self.static_values.lookup(var_name) {
                        let binding_id = match sv {
                            StaticValue::Dyn { binding_id } => *binding_id,
                            StaticValue::Closure { binding_id, .. } => *binding_id,
                        };
                        // Mark this binding as needing a backing store
                        self.needs_backing_store.insert(binding_id);
                        // Use the backing store variable name
                        let ptr_name = Self::backing_store_name(binding_id);
                        let ptr_var = Expr::new(obj.ty.clone(), mir::ExprKind::Var(ptr_name), span);
                        let kind = mir::ExprKind::Intrinsic {
                            name: "tuple_access".to_string(),
                            args: vec![
                                ptr_var,
                                Expr::new(
                                    i32_type,
                                    mir::ExprKind::Literal(mir::Literal::Int(idx.to_string())),
                                    span,
                                ),
                            ],
                        };
                        return Ok((Expr::new(ty, kind, span), StaticValue::Dyn { binding_id: 0 }));
                    }
                }

                // Fallback for complex expressions: inline Materialize+Let
                let tmp_name = self.fresh_name("ptr");
                let tmp_binding_id = self.fresh_binding_id();
                let obj_ty = obj.ty.clone();
                let materialized_obj =
                    Expr::new(obj_ty.clone(), mir::ExprKind::Materialize(Box::new(obj)), span);

                // Reference the temp in the tuple_access
                let tmp_var = Expr::new(obj_ty.clone(), mir::ExprKind::Var(tmp_name.clone()), span);
                let access_expr = Expr::new(
                    ty.clone(),
                    mir::ExprKind::Intrinsic {
                        name: "tuple_access".to_string(),
                        args: vec![
                            tmp_var,
                            Expr::new(
                                i32_type,
                                mir::ExprKind::Literal(mir::Literal::Int(idx.to_string())),
                                span,
                            ),
                        ],
                    },
                    span,
                );

                (
                    mir::ExprKind::Let {
                        name: tmp_name,
                        binding_id: tmp_binding_id,
                        value: Box::new(materialized_obj),
                        body: Box::new(access_expr),
                    },
                    StaticValue::Dyn { binding_id: 0 },
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
                        StaticValue::Dyn { binding_id: 0 },
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
            ExprKind::Assert(cond, body) => {
                let (cond, _) = self.flatten_expr(cond)?;
                let (body, _) = self.flatten_expr(body)?;
                (
                    mir::ExprKind::Intrinsic {
                        name: "assert".to_string(),
                        args: vec![cond, body],
                    },
                    StaticValue::Dyn { binding_id: 0 },
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
                // Assign a unique binding ID for this binding
                let binding_id = self.fresh_binding_id();

                // Create static value with the binding ID
                let sv_with_id = match value_sv {
                    StaticValue::Dyn { .. } => StaticValue::Dyn { binding_id },
                    StaticValue::Closure { lam_name, .. } => StaticValue::Closure { lam_name, binding_id },
                };

                // Track the static value of this binding
                self.static_values.push_scope();
                self.static_values.insert(name.clone(), sv_with_id);

                let (body, body_sv) = self.flatten_expr(&let_in.body)?;

                self.static_values.pop_scope();

                // Check if this binding needs a backing store
                let body = if self.needs_backing_store.contains(&binding_id) {
                    // Wrap body with backing store materialization:
                    // let __ptr_{id} = materialize(name) in body
                    let ptr_name = Self::backing_store_name(binding_id);
                    let ptr_binding_id = self.fresh_binding_id();
                    Expr::new(
                        body.ty.clone(),
                        mir::ExprKind::Let {
                            name: ptr_name,
                            binding_id: ptr_binding_id,
                            value: Box::new(Expr::new(
                                value.ty.clone(), // Materialize returns pointer to same type
                                mir::ExprKind::Materialize(Box::new(Expr::new(
                                    value.ty.clone(),
                                    mir::ExprKind::Var(name.clone()),
                                    span,
                                ))),
                                span,
                            )),
                            body: Box::new(body),
                        },
                        span,
                    )
                } else {
                    body
                };

                // If the value is a Let, hoist it out:
                // let x = (let y = A in B) in C  =>  let y = A in let x = B in C
                let result = mir::ExprKind::Let {
                    name: name.clone(),
                    binding_id,
                    value: Box::new(value),
                    body: Box::new(body),
                };
                let result = Self::hoist_inner_lets(result, span);

                Ok((result, body_sv))
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
                let binding_id = self.fresh_binding_id();
                let (body, body_sv) = self.flatten_expr(&let_in.body)?;
                Ok((
                    mir::ExprKind::Let {
                        name: self.fresh_name("ignored"),
                        binding_id,
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
                    Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
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
                        .unwrap_or_else(|| {
                            panic!("BUG: Tuple pattern element {} has no type. Type checking should ensure all tuple elements have types.", i)
                        });
                    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);

                    // Wrap the tuple var in Materialize for pointer access
                    let tuple_var = Expr::new(tuple_ty.clone(), mir::ExprKind::Var(tmp.clone()), span);
                    let materialized_tuple = Expr::new(
                        tuple_ty.clone(),
                        mir::ExprKind::Materialize(Box::new(tuple_var)),
                        span,
                    );

                    let extract = Expr::new(
                        elem_ty.clone(),
                        mir::ExprKind::Intrinsic {
                            name: "tuple_access".to_string(),
                            args: vec![
                                materialized_tuple,
                                Expr::new(
                                    i32_type,
                                    mir::ExprKind::Literal(mir::Literal::Int(i.to_string())),
                                    span,
                                ),
                            ],
                        },
                        span,
                    );

                    let elem_binding_id = self.fresh_binding_id();
                    body = Expr::new(
                        body.ty.clone(),
                        mir::ExprKind::Let {
                            name,
                            binding_id: elem_binding_id,
                            value: Box::new(extract),
                            body: Box::new(body),
                        },
                        span,
                    );
                }

                // Wrap with the tuple binding
                let tuple_binding_id = self.fresh_binding_id();
                Ok((
                    mir::ExprKind::Let {
                        name: tmp,
                        binding_id: tuple_binding_id,
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

        // Register lambda
        let arity = lambda.params.len();
        self.add_lambda(func_name.clone(), arity);

        // Build the closure tuple elements (and record field info for type)
        // Layout: [free_var_1, free_var_2, ..., __lambda_name] (free vars sorted alphabetically)
        // __lambda_name goes LAST so that capture indices (0, 1, 2...) match SPIR-V struct indices
        let string_type = Type::Constructed(TypeName::Str("string"), vec![]);

        let mut tuple_elems = vec![];
        let mut type_fields = vec![];

        let mut sorted_vars: Vec<_> = free_vars.iter().collect();
        sorted_vars.sort();
        for var in &sorted_vars {
            // Look up the type of the free variable from the lambda body
            let var_type = self
                .find_var_type_in_expr(&lambda.body, var)
                .unwrap_or_else(|| {
                    panic!("BUG: Free variable '{}' in lambda has no type. Type checking should ensure all variables have types.", var)
                });
            tuple_elems.push(Expr::new(
                var_type.clone(),
                mir::ExprKind::Var((*var).clone()),
                span,
            ));
            type_fields.push(((*var).clone(), var_type));
        }

        // Last element: lambda name (phantom field, not lowered to SPIR-V)
        let lambda_name_expr = Expr::new(
            string_type.clone(),
            mir::ExprKind::Literal(mir::Literal::String(func_name.clone())),
            span,
        );
        tuple_elems.push(lambda_name_expr);
        type_fields.push(("__lambda_name".to_string(), string_type));

        // Build the record type (keeps field names for index lookup during __closure access)
        let closure_type = types::record(type_fields);

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

        // Return the closure tuple along with the static value indicating it's a known closure
        let sv = StaticValue::Closure {
            lam_name: func_name,
            binding_id: 0,
        };

        Ok((mir::ExprKind::Literal(mir::Literal::Tuple(tuple_elems)), sv))
    }

    /// Find the type of a variable by searching for its use in an expression
    fn find_var_type_in_expr(&self, expr: &Expression, var_name: &str) -> Option<Type> {
        match &expr.kind {
            ExprKind::Identifier(name) if name == var_name => Some(self.get_expr_type(expr)),
            ExprKind::OperatorSection(_) => None,
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

        // Flatten arguments while keeping static values for closure detection
        let args_with_sv: Result<Vec<_>> = args.iter().map(|a| self.flatten_expr(a)).collect();
        let args_with_sv = args_with_sv?;
        let args_flat: Vec<_> = args_with_sv.iter().map(|(e, _)| e.clone()).collect();

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
                        StaticValue::Dyn { binding_id: 0 },
                    ))
                } else {
                    // Desugar overloaded functions based on argument types
                    let desugared_name = self.desugar_function_name(name, args)?;

                    // Direct function call (not a closure)
                    Ok((
                        mir::ExprKind::Call {
                            func: desugared_name,
                            args: args_flat,
                        },
                        StaticValue::Dyn { binding_id: 0 },
                    ))
                }
            }
            // Handle qualified names like f32.sum (these come from name resolution)
            ExprKind::QualifiedName(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                Ok((
                    mir::ExprKind::Call {
                        func: full_name,
                        args: args_flat,
                    },
                    StaticValue::Dyn { binding_id: 0 },
                ))
            }
            // FieldAccess in application position - must be a closure
            ExprKind::FieldAccess(_, _) => {
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    let mut all_args = vec![func_flat];
                    all_args.extend(args_flat);
                    Ok((
                        mir::ExprKind::Call {
                            func: lam_name,
                            args: all_args,
                        },
                        StaticValue::Dyn { binding_id: 0 },
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
                        StaticValue::Dyn { binding_id: 0 },
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
        // Extract loop_var, init value, and bindings from pattern
        let (loop_var, init, init_bindings) =
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
                loop_var,
                init: Box::new(init),
                init_bindings,
                kind,
                body: Box::new(body),
            },
            StaticValue::Dyn { binding_id: 0 },
        ))
    }

    /// Extract loop_var name, init expr, and bindings from pattern and init expression.
    /// Returns (loop_var_name, init_expr, bindings) where bindings extract from loop_var.
    fn extract_loop_bindings(
        &mut self,
        pattern: &ast::Pattern,
        init: Option<&Expression>,
        span: Span,
    ) -> Result<(String, Expr, Vec<(String, Expr)>)> {
        let init_expr = init
            .ok_or_else(|| CompilerError::FlatteningError("Loop must have init expression".to_string()))?;

        let (init_flat, _) = self.flatten_expr(init_expr)?;
        let init_ty = init_flat.ty.clone();
        let loop_var = self.fresh_name("__loop_var");

        let bindings = match &pattern.kind {
            PatternKind::Name(name) => {
                // Single variable: binding is just identity (Var(loop_var))
                let binding = Expr::new(init_ty, mir::ExprKind::Var(loop_var.clone()), span);
                vec![(name.clone(), binding)]
            }
            PatternKind::Typed(inner, _) => {
                // Unwrap type annotation and recurse
                self.extract_bindings_from_pattern(inner, &loop_var, &init_ty, span)?
            }
            PatternKind::Tuple(patterns) => {
                self.extract_tuple_bindings(patterns, &loop_var, &init_ty, span)?
            }
            _ => {
                return Err(CompilerError::FlatteningError(format!(
                    "Loop pattern {:?} not supported",
                    pattern.kind
                )));
            }
        };

        Ok((loop_var, init_flat, bindings))
    }

    /// Helper to extract bindings from pattern given loop_var and init_ty
    fn extract_bindings_from_pattern(
        &self,
        pattern: &ast::Pattern,
        loop_var: &str,
        init_ty: &Type,
        span: Span,
    ) -> Result<Vec<(String, Expr)>> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                let binding = Expr::new(init_ty.clone(), mir::ExprKind::Var(loop_var.to_string()), span);
                Ok(vec![(name.clone(), binding)])
            }
            PatternKind::Typed(inner, _) => {
                self.extract_bindings_from_pattern(inner, loop_var, init_ty, span)
            }
            PatternKind::Tuple(patterns) => self.extract_tuple_bindings(patterns, loop_var, init_ty, span),
            _ => Err(CompilerError::FlatteningError(format!(
                "Loop pattern {:?} not supported",
                pattern.kind
            ))),
        }
    }

    /// Extract bindings for tuple pattern
    fn extract_tuple_bindings(
        &self,
        patterns: &[ast::Pattern],
        loop_var: &str,
        tuple_ty: &Type,
        span: Span,
    ) -> Result<Vec<(String, Expr)>> {
        // Get element types from tuple type
        let elem_types: Vec<Type> = match tuple_ty {
            Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
            _ => {
                return Err(CompilerError::FlatteningError(format!(
                    "Expected tuple type for tuple pattern, got {:?}",
                    tuple_ty
                )));
            }
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

            let elem_ty = elem_types.get(i).cloned().ok_or_else(|| {
                CompilerError::FlatteningError(format!(
                    "Tuple pattern element {} has no corresponding type",
                    i
                ))
            })?;
            let i32_type = Type::Constructed(TypeName::Int(32), vec![]);

            // Wrap the loop var in Materialize for pointer access
            let loop_var_expr = Expr::new(tuple_ty.clone(), mir::ExprKind::Var(loop_var.to_string()), span);
            let materialized_loop_var = Expr::new(
                tuple_ty.clone(),
                mir::ExprKind::Materialize(Box::new(loop_var_expr)),
                span,
            );

            let extract = Expr::new(
                elem_ty,
                mir::ExprKind::Intrinsic {
                    name: "tuple_access".to_string(),
                    args: vec![
                        materialized_loop_var,
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

        Ok(bindings)
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
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::OperatorSection(_)
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
            ExprKind::Tuple(elems) | ExprKind::ArrayLiteral(elems) | ExprKind::VecMatLiteral(elems) => {
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
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
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
        let _span = expr.h.span;
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
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::OperatorSection(_)
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
