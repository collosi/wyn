//! Flattening pass: AST -> MIR
//!
//! This pass performs:
//! - Defunctionalization: lambdas become top-level functions with closure records
//! - Pattern flattening: complex patterns become simple let bindings
//! - Lambda lifting: all functions become top-level Def entries

use crate::ast::{self, ExprKind, Expression, NodeCounter, NodeId, PatternKind, Span, Type, TypeName};
use crate::error::Result;
use crate::mir::{self, DefId, Expr, IntrinsicId, LocalId, LocalTable};
use crate::pattern;
use crate::scope::ScopeStack;
use crate::types;
use crate::{bail_flatten, err_flatten, err_type};
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
/// Each variant includes a LocalId for tracking backing stores.
#[derive(Debug, Clone)]
enum StaticValue {
    /// Dynamic runtime value with its LocalId
    Dyn {
        local: LocalId,
    },
    /// Defunctionalized closure with known call target
    Closure {
        /// Name of the generated lambda function
        lam_name: String,
        /// LocalId for this closure
        local: LocalId,
    },
}

/// Flattener converts AST to MIR with defunctionalization.
pub struct Flattener {
    /// Counter for generating unique names
    next_id: usize,
    /// Counter for generating unique MIR node IDs
    node_counter: NodeCounter,
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
    /// Set of LocalIds that need backing stores (materialization)
    needs_backing_store: HashSet<LocalId>,
    /// Current function's local table (reset for each function)
    current_local_table: LocalTable,
    /// Collected local tables for all functions, keyed by def name
    /// (will be converted to DefId keys after flattening)
    local_tables_by_name: HashMap<String, LocalTable>,
    /// Map from function name to DefId (built as we add defs)
    name_to_def_id: HashMap<String, DefId>,
}

impl Flattener {
    pub fn new(type_table: HashMap<NodeId, TypeScheme<TypeName>>, builtins: HashSet<String>) -> Self {
        Flattener {
            next_id: 0,
            node_counter: NodeCounter::new(),
            generated_functions: Vec::new(),
            enclosing_decl_stack: Vec::new(),
            lambda_registry: Vec::new(),
            type_table,
            static_values: ScopeStack::new(),
            builtins,
            needs_backing_store: HashSet::new(),
            current_local_table: LocalTable::new(),
            local_tables_by_name: HashMap::new(),
            name_to_def_id: HashMap::new(),
        }
    }

    /// Get the NodeCounter for use after flattening
    pub fn into_node_counter(self) -> NodeCounter {
        self.node_counter
    }

    /// Create a new MIR expression with a fresh NodeId
    fn mk_expr(&mut self, ty: Type, kind: mir::ExprKind, span: Span) -> Expr {
        Expr::new(self.node_counter.next(), ty, kind, span)
    }

    /// Get a fresh NodeId
    fn next_node_id(&mut self) -> NodeId {
        self.node_counter.next()
    }

    /// Allocate a new local variable in the current function's local table
    fn alloc_local(&mut self, debug_name: String, ty: Type, span: Option<Span>) -> LocalId {
        self.current_local_table.alloc(debug_name, ty, span)
    }

    /// Get the backing store local for a given LocalId
    fn backing_store_local(&mut self, original: LocalId, ty: Type, span: Option<Span>) -> LocalId {
        let debug_name = format!("_w_ptr_{}", original.0);
        self.alloc_local(debug_name, ty, span)
    }

    /// Register a lambda function.
    fn add_lambda(&mut self, func_name: String, arity: usize) {
        self.lambda_registry.push((func_name, arity));
    }

    /// Look up DefId for a function name. Returns None for builtins/externals.
    fn lookup_def_id(&self, name: &str) -> Option<DefId> {
        self.name_to_def_id.get(name).copied()
    }

    /// Register a def with the next available DefId
    #[allow(dead_code)]
    fn register_def(&mut self, name: &str, current_def_count: usize) -> DefId {
        let def_id = DefId(current_def_count as u32);
        self.name_to_def_id.insert(name.to_string(), def_id);
        def_id
    }

    /// Pre-register a function name with a specific DefId
    /// This allows lookups to work during flattening even for forward references.
    pub fn pre_register_def(&mut self, name: &str, def_id: DefId) {
        self.name_to_def_id.insert(name.to_string(), def_id);
    }

    /// Reset local table for a new function
    fn reset_local_table(&mut self) {
        self.current_local_table = LocalTable::new();
        self.needs_backing_store.clear();
    }

    /// Save current local table for a function
    fn save_local_table(&mut self, func_name: &str) {
        self.local_tables_by_name.insert(
            func_name.to_string(),
            std::mem::take(&mut self.current_local_table),
        );
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
    /// - abs/sign -> abs_f32, abs_i32, abs_u32, sign_f32, sign_i32
    /// - min/max -> min_f32, min_i32, min_u32, max_f32, max_i32, max_u32
    /// - clamp -> clamp_f32, clamp_i32, clamp_u32
    fn desugar_function_name(&self, name: &str, args: &[Expression]) -> Result<String> {
        match name {
            "mul" => self.desugar_mul(args),
            "matav" => self.desugar_matav(args),
            // Type-dispatched math functions (different GLSL opcodes for float/signed/unsigned)
            "abs" | "sign" | "min" | "max" | "clamp" => self.desugar_numeric_op(name, args),
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

    /// Desugar numeric operations (abs, sign, min, max, clamp) based on element type
    /// Transforms: abs x → f32.abs x (or i32.abs, etc. based on type)
    fn desugar_numeric_op(&self, name: &str, args: &[Expression]) -> Result<String> {
        if args.is_empty() {
            return Ok(name.to_string());
        }

        // Get the type of the first argument
        let arg_ty = self.get_expr_type(&args[0]);

        // Extract the element type (scalar or vector element)
        let elem_ty = Self::extract_element_type(&arg_ty);

        // Get the type prefix (f32, i32, u32, etc.)
        let type_prefix = Self::primitive_type_to_string(&elem_ty)?;

        // Produce qualified name: f32.abs, i32.min, etc.
        Ok(format!("{}.{}", type_prefix, name))
    }

    /// Extract the element type from a scalar or vector type
    fn extract_element_type(ty: &Type) -> Type {
        match ty {
            Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
                // vec<n, elem> -> elem
                args[1].clone()
            }
            _ => ty.clone(), // Scalar type, return as-is
        }
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
            _ => Err(err_type!(
                "Invalid element type for matrix/vector: {:?}. \
                Only f16/f32/f64, i8/i16/i32/i64, u8/u16/u32/u64, and bool are supported.",
                ty
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
        format!("_w_{}{}", prefix, self.fresh_id())
    }

    /// Hoist inner Let expressions out of a Let's value.
    /// Transforms: let x = (let y = A in B) in C  =>  let y = A in let x = B in C
    /// This ensures materialized pointers are at the same scope level as their referents.
    fn hoist_inner_lets(&mut self, expr_kind: mir::ExprKind, span: Span) -> mir::ExprKind {
        if let mir::ExprKind::Let {
            local,
            value,
            body,
        } = expr_kind
        {
            if let mir::ExprKind::Let {
                local: inner_local,
                value: inner_value,
                body: inner_body,
            } = value.kind
            {
                // Hoist: let x = (let y = A in B) in C => let y = A in let x = B in C
                // Capture body's type before moving it
                let body_ty = body.ty.clone();
                let new_inner = mir::ExprKind::Let {
                    local,
                    value: inner_body,
                    body,
                };
                let new_inner_expr = self.mk_expr(body_ty.clone(), new_inner, span);
                // Recursively hoist in case there are more nested Lets
                let hoisted_inner = self.hoist_inner_lets(new_inner_expr.kind, span);
                mir::ExprKind::Let {
                    local: inner_local,
                    value: inner_value,
                    body: Box::new(self.mk_expr(body_ty, hoisted_inner, span)),
                }
            } else {
                mir::ExprKind::Let {
                    local,
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
        let scheme = self
            .type_table
            .get(&obj.h.id)
            .ok_or_else(|| err_flatten!("No type information for field access target"))?;

        let obj_type = self
            .get_monotype(scheme)
            .ok_or_else(|| err_flatten!("Could not extract monotype from scheme"))?;

        // Resolve based on type
        match obj_type {
            // Vector types: x=0, y=1, z=2, w=3
            Type::Constructed(TypeName::Vec, _) => match field {
                "x" => Ok(0),
                "y" => Ok(1),
                "z" => Ok(2),
                "w" => Ok(3),
                _ => Err(err_flatten!("Unknown vector field: {}", field)),
            },
            // Record types: look up field by name
            Type::Constructed(TypeName::Record(fields), _) => fields
                .iter()
                .enumerate()
                .find(|(_, name)| name.as_str() == field)
                .map(|(idx, _)| idx)
                .ok_or_else(|| err_flatten!("Unknown record field: {}", field)),
            // Tuple types: should use numeric access
            Type::Constructed(TypeName::Tuple(_), _) => Err(err_flatten!(
                "Tuple access must use numeric index, not '{}'",
                field
            )),
            _ => Err(err_flatten!(
                "Cannot access field '{}' on type {:?}",
                field,
                obj_type
            )),
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
                id: self.next_node_id(),
                name: d.name.clone(),
                ty,
                attributes: self.convert_attributes(&d.attributes),
                body,
                span: d.body.h.span,
            }
        } else {
            // Function
            let params = self.flatten_params(&d.params)?;
            let span = d.body.h.span;

            // Register params with binding IDs before flattening body
            let param_bindings = self.register_param_bindings(&d.params)?;

            let (body, _) = self.flatten_expr(&d.body)?;

            // Wrap body with backing stores for params that need them
            let body = self.wrap_param_backing_stores(body, param_bindings, span);

            let ret_type = self.get_expr_type(&d.body);
            mir::Def::Function {
                id: self.next_node_id(),
                name: d.name.clone(),
                params,
                ret_type,
                attributes: self.convert_attributes(&d.attributes),
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

    /// Flatten a module declaration with a qualified name (e.g., "rand.init")
    pub fn flatten_module_decl(
        &mut self,
        d: &ast::Decl,
        qualified_name: &str,
    ) -> Result<Vec<mir::Def>> {
        self.enclosing_decl_stack.push(qualified_name.to_string());

        let def = if d.params.is_empty() {
            // Constant
            let (body, _) = self.flatten_expr(&d.body)?;
            let ty = self.get_expr_type(&d.body);
            mir::Def::Constant {
                id: self.next_node_id(),
                name: qualified_name.to_string(),
                ty,
                attributes: self.convert_attributes(&d.attributes),
                body,
                span: d.body.h.span,
            }
        } else {
            // Function
            let params = self.flatten_params(&d.params)?;
            let span = d.body.h.span;

            // Register params with binding IDs before flattening body
            let param_bindings = self.register_param_bindings(&d.params)?;

            let (body, _) = self.flatten_expr(&d.body)?;

            // Wrap body with backing stores for params that need them
            let body = self.wrap_param_backing_stores(body, param_bindings, span);

            let ret_type = self.get_expr_type(&d.body);
            mir::Def::Function {
                id: self.next_node_id(),
                name: qualified_name.to_string(),
                params,
                ret_type,
                attributes: self.convert_attributes(&d.attributes),
                body,
                span,
            }
        };

        // Collect generated lambdas before the definition
        let mut defs = Vec::new();
        defs.append(&mut self.generated_functions);
        defs.push(def);

        self.enclosing_decl_stack.pop();
        Ok(defs)
    }

    /// Flatten an entire program
    pub fn flatten_program(&mut self, program: &ast::Program) -> Result<mir::Program> {
        let mut defs = Vec::new();

        // Pre-register all function/constant names with DefIds so lookups work during flattening
        let mut def_idx = 0;
        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    self.name_to_def_id.insert(d.name.clone(), DefId(def_idx));
                    def_idx += 1;
                }
                ast::Declaration::Entry(e) => {
                    self.name_to_def_id.insert(e.name.clone(), DefId(def_idx));
                    def_idx += 1;
                }
                ast::Declaration::Uniform(u) => {
                    self.name_to_def_id.insert(u.name.clone(), DefId(def_idx));
                    def_idx += 1;
                }
                ast::Declaration::Storage(s) => {
                    self.name_to_def_id.insert(s.name.clone(), DefId(def_idx));
                    def_idx += 1;
                }
                // Other declarations don't produce defs
                _ => {}
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

                    let span = e.body.h.span;

                    // Register params with binding IDs before flattening body
                    let param_bindings = self.register_param_bindings(&e.params)?;

                    let (body, _) = self.flatten_expr(&e.body)?;

                    // Convert entry type to ExecutionModel
                    let execution_model = match &e.entry_type {
                        ast::Attribute::Vertex => mir::ExecutionModel::Vertex,
                        ast::Attribute::Fragment => mir::ExecutionModel::Fragment,
                        ast::Attribute::Compute { local_size } => mir::ExecutionModel::Compute {
                            local_size: *local_size,
                        },
                        _ => panic!("Invalid entry type attribute: {:?}", e.entry_type),
                    };

                    // Convert params to EntryInput with IoDecoration and LocalId
                    let inputs: Vec<mir::EntryInput> = e
                        .params
                        .iter()
                        .zip(param_bindings.iter())
                        .map(|(p, (_, _, local_id))| {
                            let name = self.extract_param_name(p).unwrap_or_default();
                            let ty = self.get_pattern_type(p);
                            let decoration = self.extract_io_decoration(p);
                            mir::EntryInput {
                                name,
                                local_id: *local_id,
                                ty,
                                decoration,
                            }
                        })
                        .collect();

                    // Wrap body with backing stores for params that need them
                    let body = self.wrap_param_backing_stores(body, param_bindings, span);

                    // Convert AST EntryOutput to MIR EntryOutput with IoDecoration
                    let ret_type = self.get_expr_type(&e.body);
                    let outputs: Vec<mir::EntryOutput> = if e.outputs.iter().all(|o| o.attribute.is_none()) && e.outputs.len() == 1 {
                        // Single output without explicit decoration
                        if !matches!(ret_type, polytype::Type::Constructed(ast::TypeName::Unit, _)) {
                            vec![mir::EntryOutput {
                                ty: ret_type,
                                decoration: None,
                            }]
                        } else {
                            vec![]
                        }
                    } else {
                        // Multiple outputs with decorations (tuple return)
                        if let polytype::Type::Constructed(ast::TypeName::Tuple(_), component_types) =
                            &ret_type
                        {
                            e.outputs
                                .iter()
                                .zip(component_types.iter())
                                .map(|(output, ty)| mir::EntryOutput {
                                    ty: ty.clone(),
                                    decoration: output
                                        .attribute
                                        .as_ref()
                                        .and_then(|a| self.convert_to_io_decoration(a)),
                                })
                                .collect()
                        } else {
                            // Single output with decoration
                            vec![mir::EntryOutput {
                                ty: ret_type,
                                decoration: e
                                    .outputs
                                    .first()
                                    .and_then(|o| o.attribute.as_ref())
                                    .and_then(|a| self.convert_to_io_decoration(a)),
                            }]
                        }
                    };

                    let def = mir::Def::EntryPoint {
                        id: self.next_node_id(),
                        name: e.name.clone(),
                        execution_model,
                        inputs,
                        outputs,
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
                        id: self.next_node_id(),
                        name: uniform_decl.name.clone(),
                        ty: uniform_decl.ty.clone(),
                        set: uniform_decl.set,
                        binding: uniform_decl.binding,
                    });
                }
                ast::Declaration::Storage(storage_decl) => {
                    // Storage buffers use the declared type directly
                    defs.push(mir::Def::Storage {
                        id: self.next_node_id(),
                        name: storage_decl.name.clone(),
                        ty: storage_decl.ty.clone(),
                        set: storage_decl.set,
                        binding: storage_decl.binding,
                        layout: storage_decl.layout,
                        access: storage_decl.access,
                    });
                }
                ast::Declaration::Sig(_)
                | ast::Declaration::TypeBind(_)
                | ast::Declaration::ModuleBind(_)
                | ast::Declaration::ModuleTypeBind(_)
                | ast::Declaration::Open(_)
                | ast::Declaration::Import(_) => {
                    // Skip declarations that don't produce MIR defs
                }
            }
        }

        Ok(mir::Program {
            defs,
            lambda_registry: self.lambda_registry.clone(),
            local_tables: std::collections::HashMap::new(),
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
            ast::Attribute::Compute { local_size } => mir::Attribute::Compute {
                local_size: *local_size,
            },
            // The binding is stored in Def::Uniform, not the Attribute
            ast::Attribute::Uniform { .. } => mir::Attribute::Uniform,
            // The binding is stored in Def::Storage, not the Attribute
            ast::Attribute::Storage { .. } => mir::Attribute::Storage,
        }
    }

    /// Convert an AST attribute to IoDecoration (only Location and BuiltIn are valid)
    fn convert_to_io_decoration(&self, attr: &ast::Attribute) -> Option<mir::IoDecoration> {
        match attr {
            ast::Attribute::BuiltIn(builtin) => Some(mir::IoDecoration::BuiltIn(*builtin)),
            ast::Attribute::Location(loc) => Some(mir::IoDecoration::Location(*loc)),
            _ => None,
        }
    }

    /// Extract IoDecoration from a pattern (for entry point parameters)
    fn extract_io_decoration(&self, pattern: &ast::Pattern) -> Option<mir::IoDecoration> {
        match &pattern.kind {
            PatternKind::Attributed(attrs, inner) => {
                // Look for Location or BuiltIn in attributes
                for attr in attrs {
                    if let Some(dec) = self.convert_to_io_decoration(attr) {
                        return Some(dec);
                    }
                }
                // Recurse into inner pattern
                self.extract_io_decoration(inner)
            }
            PatternKind::Typed(inner, _) => self.extract_io_decoration(inner),
            _ => None,
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

    /// Register function parameters with LocalIds for backing store tracking.
    /// Returns a Vec of (param_name, param_type, local_id) tuples.
    fn register_param_bindings(&mut self, params: &[ast::Pattern]) -> Result<Vec<(String, Type, LocalId)>> {
        self.static_values.push_scope();
        let mut param_bindings = Vec::new();
        for param in params {
            let name = self.extract_param_name(param)?;
            let ty = self.get_pattern_type(param);
            let local_id = self.alloc_local(name.clone(), ty.clone(), None);
            self.static_values.insert(name.clone(), StaticValue::Dyn { local: local_id });
            param_bindings.push((name, ty, local_id));
        }
        Ok(param_bindings)
    }

    /// Wrap a function body with backing store materializations for parameters that need them.
    fn wrap_param_backing_stores(
        &mut self,
        body: Expr,
        param_bindings: Vec<(String, Type, LocalId)>,
        span: Span,
    ) -> Expr {
        self.static_values.pop_scope();

        // Collect params that need backing stores (in reverse order for proper nesting)
        let params_needing_stores: Vec<_> = param_bindings
            .into_iter()
            .filter(|(_, _, local_id)| self.needs_backing_store.contains(local_id))
            .collect();

        // Wrap body with backing stores for each param that needs one
        let mut result = body;
        for (_, param_ty, local_id) in params_needing_stores.into_iter().rev() {
            let ptr_local = self.backing_store_local(local_id, types::pointer(param_ty.clone()), Some(span));
            // Build inner expressions first to avoid nested mutable borrows
            let var_expr = self.mk_expr(param_ty.clone(), mir::ExprKind::Var(local_id), span);
            let materialize_expr = self.mk_expr(
                types::pointer(param_ty),
                mir::ExprKind::Materialize(Box::new(var_expr)),
                span,
            );
            let result_ty = result.ty.clone();
            result = self.mk_expr(
                result_ty,
                mir::ExprKind::Let {
                    local: ptr_local,
                    value: Box::new(materialize_expr),
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
            _ => Err(err_flatten!("Complex parameter patterns not yet supported")),
        }
    }

    /// Flatten an expression, returning the MIR expression and its static value
    fn flatten_expr(&mut self, expr: &Expression) -> Result<(Expr, StaticValue)> {
        let span = expr.h.span;
        let ty = self.get_expr_type(expr);
        let (kind, sv) = match &expr.kind {
            ExprKind::IntLiteral(n) => (
                mir::ExprKind::Literal(mir::Literal::Int(n.to_string())),
                StaticValue::Dyn { local: LocalId(0) },
            ),
            ExprKind::FloatLiteral(f) => (
                mir::ExprKind::Literal(mir::Literal::Float(f.to_string())),
                StaticValue::Dyn { local: LocalId(0) },
            ),
            ExprKind::BoolLiteral(b) => (
                mir::ExprKind::Literal(mir::Literal::Bool(*b)),
                StaticValue::Dyn { local: LocalId(0) },
            ),
            ExprKind::StringLiteral(s) => (
                mir::ExprKind::Literal(mir::Literal::String(s.clone())),
                StaticValue::Dyn { local: LocalId(0) },
            ),
            ExprKind::Unit => (mir::ExprKind::Unit, StaticValue::Dyn { local: LocalId(0) }),
            ExprKind::Identifier(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                // Look up static value for this variable
                let sv =
                    self.static_values.lookup(&full_name).cloned().unwrap_or(StaticValue::Dyn { local: LocalId(0) });
                // Extract LocalId from static value
                let local_id = match &sv {
                    StaticValue::Dyn { local } => *local,
                    StaticValue::Closure { local, .. } => *local,
                };
                (mir::ExprKind::Var(local_id), sv)
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
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::UnaryOp(op, operand) => {
                let (operand, _) = self.flatten_expr(operand)?;
                (
                    mir::ExprKind::UnaryOp {
                        op: op.op.clone(),
                        operand: Box::new(operand),
                    },
                    StaticValue::Dyn { local: LocalId(0) },
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
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::LetIn(let_in) => self.flatten_let_in(let_in, span)?,
            ExprKind::Lambda(lambda) => self.flatten_lambda(lambda, span)?,
            ExprKind::Application(func, args) => self.flatten_application(func, args, &ty, span)?,
            ExprKind::Tuple(elems) => {
                let elems: Result<Vec<_>> =
                    elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Tuple(elems?)),
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::ArrayLiteral(elems) => {
                let elems: Result<Vec<_>> =
                    elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Array(elems?)),
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::VecMatLiteral(elems) => {
                // Check if first element is an array literal (matrix) or scalar (vector)
                if elems.is_empty() {
                    bail_flatten!("Empty vector/matrix literal");
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
                            bail_flatten!("Matrix rows must be array literals");
                        }
                    }
                    (
                        mir::ExprKind::Literal(mir::Literal::Matrix(rows)),
                        StaticValue::Dyn { local: LocalId(0) },
                    )
                } else {
                    // Vector
                    let elems: Result<Vec<_>> =
                        elems.iter().map(|e| self.flatten_expr(e).map(|(e, _)| e)).collect();
                    (
                        mir::ExprKind::Literal(mir::Literal::Vector(elems?)),
                        StaticValue::Dyn { local: LocalId(0) },
                    )
                }
            }
            ExprKind::RecordLiteral(fields) => {
                // Records become tuples with fields in source order
                let elems: Result<Vec<_>> =
                    fields.iter().map(|(_, expr)| Ok(self.flatten_expr(expr)?.0)).collect();
                (
                    mir::ExprKind::Literal(mir::Literal::Tuple(elems?)),
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::ArrayIndex(arr_expr, idx_expr) => {
                let (arr, _) = self.flatten_expr(arr_expr)?;
                let (idx, _) = self.flatten_expr(idx_expr)?;

                // Check if index is a constant - if so, use tuple_access (OpCompositeExtract)
                // which doesn't need a backing store
                if let mir::ExprKind::Literal(mir::Literal::Int(_)) = &idx.kind {
                    // Constant index: use tuple_access directly on value
                    let kind = mir::ExprKind::Intrinsic {
                        id: IntrinsicId::TupleAccess,
                        args: vec![arr, idx],
                    };
                    return Ok((self.mk_expr(ty, kind, span), StaticValue::Dyn { local: LocalId(0) }));
                }

                // Dynamic index: need backing store for OpAccessChain
                // Check if the original expression is an identifier we can look up
                if let ExprKind::Identifier(quals, name) = &arr_expr.kind {
                    let full_name = if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                    if let Some(sv) = self.static_values.lookup(&full_name) {
                        let local_id = match sv {
                            StaticValue::Dyn { local } => *local,
                            StaticValue::Closure { local, .. } => *local,
                        };
                        // Mark this binding as needing a backing store
                        self.needs_backing_store.insert(local_id);
                        // Allocate a backing store local and reference it
                        let ptr_local = self.backing_store_local(local_id, types::pointer(arr.ty.clone()), Some(span));
                        let ptr_var = self.mk_expr(
                            types::pointer(arr.ty.clone()),
                            mir::ExprKind::Var(ptr_local),
                            span,
                        );
                        let kind = mir::ExprKind::Intrinsic {
                            id: IntrinsicId::Index,
                            args: vec![ptr_var, idx],
                        };
                        return Ok((self.mk_expr(ty, kind, span), StaticValue::Dyn { local: LocalId(0) }));
                    }
                }

                // Fallback for dynamic index on complex expression: wrap in Materialize
                let materialized_arr = self.mk_expr(
                    types::pointer(arr.ty.clone()),
                    mir::ExprKind::Materialize(Box::new(arr)),
                    span,
                );
                (
                    mir::ExprKind::Intrinsic {
                        id: IntrinsicId::Index,
                        args: vec![materialized_arr, idx],
                    },
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::ArrayWith { array, index, value } => {
                // Flatten array with syntax to a call to _w_array_with intrinsic
                // _w_array_with : [n]a -> i32 -> a -> [n]a
                let (arr, _) = self.flatten_expr(array)?;
                let (idx, _) = self.flatten_expr(index)?;
                let (val, _) = self.flatten_expr(value)?;

                // Generate a call to _w_array_with(arr, idx, val)
                // Use sentinel DefId for builtin - will be resolved during lowering
                let func_id = self.lookup_def_id("_w_array_with").unwrap_or(DefId(u32::MAX));
                let func_name = if func_id == DefId(u32::MAX) { Some("_w_array_with".to_string()) } else { None };
                (
                    mir::ExprKind::Call {
                        func: func_id,
                        func_name,
                        args: vec![arr, idx, val],
                    },
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::FieldAccess(obj_expr, field) => {
                let (obj, _obj_sv) = self.flatten_expr(obj_expr)?;

                // Resolve field name to index using type information
                let idx = self.resolve_field_index(obj_expr, field)?;

                // Create i32 type for the index literal
                let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
                let idx_expr = self.mk_expr(
                    i32_type,
                    mir::ExprKind::Literal(mir::Literal::Int(idx.to_string())),
                    span,
                );

                // Pass value directly to tuple_access - no Materialize/backing store needed
                // Lowering handles both pointer and value inputs correctly
                (
                    mir::ExprKind::Intrinsic {
                        id: IntrinsicId::TupleAccess,
                        args: vec![obj, idx_expr],
                    },
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::Loop(loop_expr) => self.flatten_loop(loop_expr, span)?,
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                // Type annotations don't affect runtime, just flatten inner
                return self.flatten_expr(inner);
            }
            ExprKind::Assert(cond, body) => {
                let (cond, _) = self.flatten_expr(cond)?;
                let (body, _) = self.flatten_expr(body)?;
                (
                    mir::ExprKind::Intrinsic {
                        id: IntrinsicId::Assert,
                        args: vec![cond, body],
                    },
                    StaticValue::Dyn { local: LocalId(0) },
                )
            }
            ExprKind::TypeHole => {
                bail_flatten!("Type holes should be resolved before flattening");
            }
            ExprKind::Match(_) => {
                bail_flatten!("Match expressions not yet supported");
            }
            ExprKind::Range(_) => {
                bail_flatten!("Range expressions should be desugared before flattening");
            }
        };

        Ok((self.mk_expr(ty, kind, span), sv))
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
                // Allocate a LocalId for this binding
                let local_id = self.alloc_local(name.clone(), value.ty.clone(), Some(span));

                // Create static value with the LocalId
                let sv_with_id = match value_sv {
                    StaticValue::Dyn { .. } => StaticValue::Dyn { local: local_id },
                    StaticValue::Closure { lam_name, .. } => StaticValue::Closure { lam_name, local: local_id },
                };

                // Track the static value of this binding
                self.static_values.push_scope();
                self.static_values.insert(name.clone(), sv_with_id);

                let (body, body_sv) = self.flatten_expr(&let_in.body)?;

                self.static_values.pop_scope();

                // Check if this binding needs a backing store
                let body = if self.needs_backing_store.contains(&local_id) {
                    // Wrap body with backing store materialization:
                    // let _w_ptr_{id} = materialize(name) in body
                    let ptr_local = self.backing_store_local(local_id, types::pointer(value.ty.clone()), Some(span));
                    // Build inner expressions first to avoid nested mutable borrow
                    let var_expr = self.mk_expr(value.ty.clone(), mir::ExprKind::Var(local_id), span);
                    let materialize_expr = self.mk_expr(
                        types::pointer(value.ty.clone()),
                        mir::ExprKind::Materialize(Box::new(var_expr)),
                        span,
                    );
                    let body_ty = body.ty.clone();
                    self.mk_expr(
                        body_ty,
                        mir::ExprKind::Let {
                            local: ptr_local,
                            value: Box::new(materialize_expr),
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
                    local: local_id,
                    value: Box::new(value),
                    body: Box::new(body),
                };
                let result = self.hoist_inner_lets(result, span);

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
                let ignored_name = self.fresh_name("ignored");
                let local_id = self.alloc_local(ignored_name, value.ty.clone(), Some(span));
                let (body, body_sv) = self.flatten_expr(&let_in.body)?;
                Ok((
                    mir::ExprKind::Let {
                        local: local_id,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    body_sv,
                ))
            }
            PatternKind::Tuple(patterns) => {
                // Allocate a local for the tuple value
                let tuple_ty = self.get_pattern_type(&let_in.pattern);
                let tup_name = self.fresh_name("tup");
                let tuple_local = self.alloc_local(tup_name, tuple_ty.clone(), Some(span));

                // Get the element types
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
                                bail_flatten!("Nested complex patterns not supported");
                            }
                        },
                        PatternKind::Wildcard => continue, // Skip wildcards
                        _ => {
                            bail_flatten!("Complex nested patterns not supported");
                        }
                    };

                    let elem_ty = elem_types
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| {
                            panic!("BUG: Tuple pattern element {} has no type. Type checking should ensure all tuple elements have types.", i)
                        });
                    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);

                    // Pass value directly to tuple_access - no Materialize needed
                    let tuple_var = self.mk_expr(tuple_ty.clone(), mir::ExprKind::Var(tuple_local), span);
                    let idx_expr = self.mk_expr(
                        i32_type,
                        mir::ExprKind::Literal(mir::Literal::Int(i.to_string())),
                        span,
                    );

                    let extract = self.mk_expr(
                        elem_ty.clone(),
                        mir::ExprKind::Intrinsic {
                            id: IntrinsicId::TupleAccess,
                            args: vec![tuple_var, idx_expr],
                        },
                        span,
                    );

                    let elem_local = self.alloc_local(name, elem_ty, Some(span));
                    body = self.mk_expr(
                        body.ty.clone(),
                        mir::ExprKind::Let {
                            local: elem_local,
                            value: Box::new(extract),
                            body: Box::new(body),
                        },
                        span,
                    );
                }

                // Wrap with the tuple binding
                Ok((
                    mir::ExprKind::Let {
                        local: tuple_local,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    body_sv,
                ))
            }
            _ => Err(err_flatten!(
                "Pattern kind {:?} not yet supported in let",
                let_in.pattern.kind
            )),
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
        let func_name = format!("_w_lam_{}_{}", enclosing, id);

        // Register lambda
        let arity = lambda.params.len();
        self.add_lambda(func_name.clone(), arity);

        // Build closure captures
        let mut capture_elems = vec![];
        let mut capture_fields = vec![];

        let mut sorted_vars: Vec<_> = free_vars.iter().collect();
        sorted_vars.sort();
        for var in &sorted_vars {
            // Look up the type and LocalId of the free variable
            let var_type = self
                .find_var_type_in_expr(&lambda.body, var)
                .unwrap_or_else(|| {
                    panic!("BUG: Free variable '{}' in lambda has no type. Type checking should ensure all variables have types.", var)
                });
            // Look up the LocalId from static_values
            let local_id = self.static_values.lookup(*var)
                .map(|sv| match sv {
                    StaticValue::Dyn { local } => *local,
                    StaticValue::Closure { local, .. } => *local,
                })
                .unwrap_or(LocalId(0)); // Fallback for globals/builtins
            capture_elems.push(self.mk_expr(var_type.clone(), mir::ExprKind::Var(local_id), span));
            capture_fields.push(((*var).clone(), var_type));
        }

        // The closure type is the captures tuple (what the generated function receives)
        let closure_type = types::tuple(capture_fields.iter().map(|(_, ty)| ty.clone()).collect());

        // Build parameters: closure first, then lambda params
        let mut params = vec![mir::Param {
            name: "_w_closure".to_string(),
            ty: closure_type.clone(),
            is_consumed: false,
        }];

        for param in &lambda.params {
            let name = param
                .simple_name()
                .ok_or_else(|| err_flatten!("Complex lambda parameter patterns not supported"))?
                .to_string();
            let ty = self.get_pattern_type(param);
            params.push(mir::Param {
                name,
                ty,
                is_consumed: false,
            });
        }

        // Flatten the body, then wrap with let bindings to extract free vars from closure
        let (flattened_body, _) = self.flatten_expr(&lambda.body)?;
        let body =
            self.wrap_body_with_closure_bindings(flattened_body, &capture_fields, &closure_type, span);

        let ret_type = body.ty.clone();

        // Create the generated function
        let func = mir::Def::Function {
            id: self.next_node_id(),
            name: func_name.clone(),
            params,
            ret_type,
            attributes: vec![],
            body,
            span,
        };
        self.generated_functions.push(func);

        // Use sentinel DefId for now - will be resolved after all defs are collected
        let lambda_def_id = DefId(u32::MAX);

        // Return the closure along with the static value indicating it's a known closure
        let sv = StaticValue::Closure {
            lam_name: func_name,
            local: LocalId(0),
        };

        Ok((
            mir::ExprKind::Closure {
                lambda: lambda_def_id,
                captures: capture_elems,
            },
            sv,
        ))
    }

    /// Find the type of a variable by searching for its use in an expression
    fn find_var_type_in_expr(&self, expr: &Expression, var_name: &str) -> Option<Type> {
        match &expr.kind {
            ExprKind::Identifier(quals, name) if quals.is_empty() && name == var_name => Some(self.get_expr_type(expr)),
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
            ExprKind::ArrayWith { array, index, value } => self
                .find_var_type_in_expr(array, var_name)
                .or_else(|| self.find_var_type_in_expr(index, var_name))
                .or_else(|| self.find_var_type_in_expr(value, var_name)),
            ExprKind::FieldAccess(obj, _) => self.find_var_type_in_expr(obj, var_name),
            ExprKind::RecordLiteral(fields) => {
                fields.iter().find_map(|(_, e)| self.find_var_type_in_expr(e, var_name))
            }
            ExprKind::Lambda(lambda) => self.find_var_type_in_expr(&lambda.body, var_name),
            _ => None,
        }
    }

    /// Check if a type is an Arrow type and return the (param, result) if so
    fn as_arrow_type(ty: &Type) -> Option<(&Type, &Type)> {
        types::as_arrow(ty)
    }

    /// Wrap a MIR body with let bindings to extract free vars from captures tuple
    /// The _w_closure parameter IS the captures tuple directly (element 1 of the closure)
    /// Produces: let x = @tuple_access(_w_closure, 0) in ... body ...
    fn wrap_body_with_closure_bindings(
        &mut self,
        body: Expr,
        capture_fields: &[(String, Type)],
        captures_type: &Type,
        span: Span,
    ) -> Expr {
        let mut wrapped = body;
        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);

        // The _w_closure parameter is the first parameter of the generated function
        // Allocate a LocalId for it (parameter 0)
        let closure_local = self.alloc_local("_w_closure".to_string(), captures_type.clone(), Some(span));

        // Iterate in reverse so innermost let is first free var
        for (idx, (var_name, var_type)) in capture_fields.iter().enumerate().rev() {
            let closure_var = self.mk_expr(
                captures_type.clone(),
                mir::ExprKind::Var(closure_local),
                span,
            );
            let idx_expr = self.mk_expr(
                i32_type.clone(),
                mir::ExprKind::Literal(mir::Literal::Int(idx.to_string())),
                span,
            );
            let tuple_access = self.mk_expr(
                var_type.clone(),
                mir::ExprKind::Intrinsic {
                    id: IntrinsicId::TupleAccess,
                    args: vec![closure_var, idx_expr],
                },
                span,
            );

            let local = self.alloc_local(var_name.clone(), var_type.clone(), Some(span));
            wrapped = self.mk_expr(
                wrapped.ty.clone(),
                mir::ExprKind::Let {
                    local,
                    value: Box::new(tuple_access),
                    body: Box::new(wrapped),
                },
                span,
            );
        }
        wrapped
    }

    /// Synthesize a lambda for partial application of a function.
    /// Given `f x y` where f expects 4 args, creates `\a b -> f x y a b`
    fn synthesize_partial_application(
        &mut self,
        func_name: String,
        applied_args: Vec<Expr>,
        result_type: &Type,
        span: Span,
    ) -> Result<(mir::ExprKind, StaticValue)> {
        // Extract remaining parameter types from the Arrow type
        let mut remaining_param_types = vec![];
        let mut current_type = result_type.clone();
        while let Some((param_ty, ret_ty)) = Self::as_arrow_type(&current_type) {
            remaining_param_types.push(param_ty.clone());
            current_type = ret_ty.clone();
        }
        let final_return_type = current_type;

        // Generate unique lambda name
        let id = self.fresh_id();
        let enclosing = self.enclosing_decl_stack.last().map(|s| s.as_str()).unwrap_or("anon");
        let lam_name = format!("_w_partial_{}_{}", enclosing, id);

        // Register lambda
        let arity = remaining_param_types.len();
        self.add_lambda(lam_name.clone(), arity);

        // Build closure captures
        let mut capture_elems = vec![];
        let mut capture_fields = vec![];

        // Capture each applied arg
        for (i, arg) in applied_args.iter().enumerate() {
            let field_name = format!("_w_cap_{}", i);
            capture_elems.push(arg.clone());
            capture_fields.push((field_name, arg.ty.clone()));
        }

        // The closure type is the captures tuple (what the generated function receives)
        let closure_type = types::tuple(capture_fields.iter().map(|(_, ty)| ty.clone()).collect());

        // Build parameters: closure first, then remaining params
        let mut params = vec![mir::Param {
            name: "_w_closure".to_string(),
            ty: closure_type.clone(),
            is_consumed: false,
        }];

        for (i, param_ty) in remaining_param_types.iter().enumerate() {
            let param_name = format!("_w_arg_{}", i);
            params.push(mir::Param {
                name: param_name,
                ty: param_ty.clone(),
                is_consumed: false,
            });
        }

        // Allocate LocalIds for the lambda's parameters
        let closure_local = self.alloc_local("_w_closure".to_string(), closure_type.clone(), Some(span));
        let remaining_param_locals: Vec<LocalId> = remaining_param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| self.alloc_local(format!("_w_arg_{}", i), ty.clone(), Some(span)))
            .collect();

        // Build the body: call original function with captured args + remaining args
        // First, extract captured args from closure using tuple_access intrinsic
        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
        let mut call_args = vec![];
        for (i, (_, field_ty)) in capture_fields.iter().enumerate() {
            let closure_var = self.mk_expr(
                closure_type.clone(),
                mir::ExprKind::Var(closure_local),
                span,
            );
            let idx_expr = self.mk_expr(
                i32_type.clone(),
                mir::ExprKind::Literal(mir::Literal::Int(i.to_string())),
                span,
            );
            let field_access = self.mk_expr(
                field_ty.clone(),
                mir::ExprKind::Intrinsic {
                    id: IntrinsicId::TupleAccess,
                    args: vec![closure_var, idx_expr],
                },
                span,
            );
            call_args.push(field_access);
        }

        // Then add remaining parameter references
        for (i, param_ty) in remaining_param_types.iter().enumerate() {
            call_args.push(self.mk_expr(param_ty.clone(), mir::ExprKind::Var(remaining_param_locals[i]), span));
        }

        // Look up DefId for the function being called (use sentinel for builtins)
        let func_def_id = self.lookup_def_id(&func_name).unwrap_or(DefId(u32::MAX));
        let call_func_name = if func_def_id == DefId(u32::MAX) { Some(func_name.clone()) } else { None };

        // Create the call expression
        let body = self.mk_expr(
            final_return_type.clone(),
            mir::ExprKind::Call {
                func: func_def_id,
                func_name: call_func_name,
                args: call_args,
            },
            span,
        );

        // Create the generated function
        let func = mir::Def::Function {
            id: self.next_node_id(),
            name: lam_name.clone(),
            params,
            ret_type: final_return_type,
            attributes: vec![],
            body,
            span,
        };
        self.generated_functions.push(func);

        // Use sentinel DefId for now - will be resolved after all defs are collected
        let lambda_def_id = DefId(u32::MAX);

        // Return the closure
        let sv = StaticValue::Closure {
            lam_name: lam_name.clone(),
            local: LocalId(0),
        };

        Ok((
            mir::ExprKind::Closure {
                lambda: lambda_def_id,
                captures: capture_elems,
            },
            sv,
        ))
    }

    /// Flatten an application expression
    fn flatten_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
        result_type: &Type,
        span: Span,
    ) -> Result<(mir::ExprKind, StaticValue)> {
        let (func_flat, func_sv) = self.flatten_expr(func)?;

        // Flatten arguments while keeping static values for closure detection
        let args_with_sv: Result<Vec<_>> = args.iter().map(|a| self.flatten_expr(a)).collect();
        let args_with_sv = args_with_sv?;
        let args_flat: Vec<_> = args_with_sv.iter().map(|(e, _)| e.clone()).collect();

        // Check if this is applying a known function name
        match &func.kind {
            ExprKind::Identifier(quals, name) => {
                // Check if the identifier is bound to a known closure
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    // Direct call to the lambda function with closure as first argument
                    let mut all_args = vec![func_flat];
                    all_args.extend(args_flat);
                    let func_id = self.lookup_def_id(&lam_name).unwrap_or(DefId(u32::MAX));
                    let call_func_name = if func_id == DefId(u32::MAX) { Some(lam_name.clone()) } else { None };
                    Ok((
                        mir::ExprKind::Call {
                            func: func_id,
                            func_name: call_func_name,
                            args: all_args,
                        },
                        StaticValue::Dyn { local: LocalId(0) },
                    ))
                } else {
                    let full_name =
                        if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };

                    // Desugar overloaded functions based on argument types
                    let desugared_name = self.desugar_function_name(&full_name, args)?;

                    // Check if this is a partial application (result type is Arrow)
                    if Self::as_arrow_type(result_type).is_some() {
                        // Partial application: synthesize a lambda
                        self.synthesize_partial_application(desugared_name, args_flat, result_type, span)
                    } else {
                        // Direct function call (not a closure)
                        let func_id = self.lookup_def_id(&desugared_name).unwrap_or(DefId(u32::MAX));
                        let call_func_name = if func_id == DefId(u32::MAX) { Some(desugared_name.clone()) } else { None };
                        Ok((
                            mir::ExprKind::Call {
                                func: func_id,
                                func_name: call_func_name,
                                args: args_flat,
                            },
                            StaticValue::Dyn { local: LocalId(0) },
                        ))
                    }
                }
            }
            // FieldAccess in application position - must be a closure
            ExprKind::FieldAccess(_, _) => {
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    let mut all_args = vec![func_flat];
                    all_args.extend(args_flat);
                    let func_id = self.lookup_def_id(&lam_name).unwrap_or(DefId(u32::MAX));
                    let call_func_name = if func_id == DefId(u32::MAX) { Some(lam_name.clone()) } else { None };
                    Ok((
                        mir::ExprKind::Call {
                            func: func_id,
                            func_name: call_func_name,
                            args: all_args,
                        },
                        StaticValue::Dyn { local: LocalId(0) },
                    ))
                } else {
                    Err(err_flatten!(
                        "Cannot call closure with unknown static value (field access). \
                         Function expression: {:?}",
                        func.kind
                    ))
                }
            }
            _ => {
                // Closure call: check if we know the static value
                if let StaticValue::Closure { lam_name, .. } = func_sv {
                    // Direct call to the lambda function with closure as first argument
                    let mut all_args = vec![func_flat];
                    all_args.extend(args_flat);
                    let func_id = self.lookup_def_id(&lam_name).unwrap_or(DefId(u32::MAX));
                    let call_func_name = if func_id == DefId(u32::MAX) { Some(lam_name.clone()) } else { None };
                    Ok((
                        mir::ExprKind::Call {
                            func: func_id,
                            func_name: call_func_name,
                            args: all_args,
                        },
                        StaticValue::Dyn { local: LocalId(0) },
                    ))
                } else {
                    // Unknown closure - this should not happen with proper function value restrictions
                    Err(err_flatten!(
                        "Cannot call closure with unknown static value. \
                         Function expression: {:?}",
                        func.kind
                    ))
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
                // Allocate a LocalId for the loop counter variable
                let var_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let var_local = self.alloc_local(var.clone(), var_ty, Some(span));
                mir::LoopKind::ForRange {
                    var: var_local,
                    bound: Box::new(bound),
                }
            }
            ast::LoopForm::ForIn(pat, iter) => {
                let var_name = match &pat.kind {
                    PatternKind::Name(n) => n.clone(),
                    _ => {
                        bail_flatten!("Complex for-in patterns not supported");
                    }
                };
                let (iter, _) = self.flatten_expr(iter)?;
                // Allocate a LocalId for the loop element variable
                // Infer type from iterator element type
                let elem_ty = self.get_pattern_type(pat);
                let var_local = self.alloc_local(var_name, elem_ty, Some(span));
                mir::LoopKind::For {
                    var: var_local,
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
            StaticValue::Dyn { local: LocalId(0) },
        ))
    }

    /// Extract loop_var LocalId, init expr, and bindings from pattern and init expression.
    /// Returns (loop_var_local, init_expr, bindings) where bindings extract from loop_var.
    fn extract_loop_bindings(
        &mut self,
        pattern: &ast::Pattern,
        init: Option<&Expression>,
        span: Span,
    ) -> Result<(LocalId, Expr, Vec<(LocalId, Expr)>)> {
        let init_expr = init.ok_or_else(|| err_flatten!("Loop must have init expression"))?;

        let (init_flat, _) = self.flatten_expr(init_expr)?;
        let init_ty = init_flat.ty.clone();
        let loop_var_name = self.fresh_name("loop_var");
        let loop_var_local = self.alloc_local(loop_var_name, init_ty.clone(), Some(span));

        let bindings = match &pattern.kind {
            PatternKind::Name(name) => {
                // Single variable: binding is just identity (Var(loop_var))
                let binding = self.mk_expr(init_ty.clone(), mir::ExprKind::Var(loop_var_local), span);
                let name_local = self.alloc_local(name.clone(), init_ty, Some(span));
                vec![(name_local, binding)]
            }
            PatternKind::Typed(inner, _) => {
                // Unwrap type annotation and recurse
                self.extract_bindings_from_pattern(inner, loop_var_local, &init_ty, span)?
            }
            PatternKind::Tuple(patterns) => {
                self.extract_tuple_bindings(patterns, loop_var_local, &init_ty, span)?
            }
            _ => {
                bail_flatten!("Loop pattern {:?} not supported", pattern.kind);
            }
        };

        Ok((loop_var_local, init_flat, bindings))
    }

    /// Helper to extract bindings from pattern given loop_var LocalId and init_ty
    fn extract_bindings_from_pattern(
        &mut self,
        pattern: &ast::Pattern,
        loop_var: LocalId,
        init_ty: &Type,
        span: Span,
    ) -> Result<Vec<(LocalId, Expr)>> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                let binding = self.mk_expr(init_ty.clone(), mir::ExprKind::Var(loop_var), span);
                let name_local = self.alloc_local(name.clone(), init_ty.clone(), Some(span));
                Ok(vec![(name_local, binding)])
            }
            PatternKind::Typed(inner, _) => {
                self.extract_bindings_from_pattern(inner, loop_var, init_ty, span)
            }
            PatternKind::Tuple(patterns) => self.extract_tuple_bindings(patterns, loop_var, init_ty, span),
            _ => Err(err_flatten!("Loop pattern {:?} not supported", pattern.kind)),
        }
    }

    /// Extract bindings for tuple pattern
    fn extract_tuple_bindings(
        &mut self,
        patterns: &[ast::Pattern],
        loop_var: LocalId,
        tuple_ty: &Type,
        span: Span,
    ) -> Result<Vec<(LocalId, Expr)>> {
        // Get element types from tuple type
        let elem_types: Vec<Type> = match tuple_ty {
            Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
            _ => {
                bail_flatten!("Expected tuple type for tuple pattern, got {:?}", tuple_ty);
            }
        };

        let mut bindings = Vec::new();
        for (i, pat) in patterns.iter().enumerate() {
            let name = match &pat.kind {
                PatternKind::Name(n) => n.clone(),
                PatternKind::Typed(inner, _) => match &inner.kind {
                    PatternKind::Name(n) => n.clone(),
                    _ => {
                        bail_flatten!("Complex loop patterns not supported");
                    }
                },
                _ => {
                    bail_flatten!("Complex loop patterns not supported");
                }
            };

            let elem_ty = elem_types
                .get(i)
                .cloned()
                .ok_or_else(|| err_flatten!("Tuple pattern element {} has no corresponding type", i))?;
            let i32_type = Type::Constructed(TypeName::Int(32), vec![]);

            // Pass value directly to tuple_access - no Materialize needed
            let loop_var_expr =
                self.mk_expr(tuple_ty.clone(), mir::ExprKind::Var(loop_var), span);
            let idx_expr = self.mk_expr(
                i32_type,
                mir::ExprKind::Literal(mir::Literal::Int(i.to_string())),
                span,
            );

            let extract = self.mk_expr(
                elem_ty.clone(),
                mir::ExprKind::Intrinsic {
                    id: IntrinsicId::TupleAccess,
                    args: vec![loop_var_expr, idx_expr],
                },
                span,
            );

            let name_local = self.alloc_local(name, elem_ty, Some(span));
            bindings.push((name_local, extract));
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
            ExprKind::Identifier(quals, name) => {
                // Only unqualified names can be free variables
                if quals.is_empty() && !bound.contains(name) && !self.builtins.contains(name) {
                    free.insert(name.clone());
                }
            }
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {}
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
            ExprKind::ArrayWith { array, index, value } => {
                self.collect_free_vars(array, bound, free);
                self.collect_free_vars(index, bound, free);
                self.collect_free_vars(value, bound, free);
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
}
