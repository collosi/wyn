use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use std::collections::{HashMap, HashSet};

/// Static values for defunctionalization, as described in the Futhark paper
#[derive(Debug, Clone, PartialEq)]
pub enum StaticValue {
    Dyn(Type),                         // Dynamic value with type
    Lam(String, Expression),           // Lambda: param name, body (defunctionalized immediately)
    Rcd(HashMap<String, StaticValue>), // Record of static values
    Arr(Box<StaticValue>),             // Array of static values

    // Defunctionalized closure with precise call-target info
    Closure {
        tag: i32,                          // runtime tag baked into the record
        lam_name: String,                  // __lam_* symbol to call directly
        arity: usize,                      // number of (non-closure) params
        env: HashMap<String, StaticValue>, // captured fields (not including __tag)
    },
}

impl StaticValue {
    /// Extract a type from a StaticValue for use in closure type construction
    pub fn to_type<G: crate::type_checker::TypeVarGenerator>(&self, gen: &mut G) -> Type {
        match self {
            StaticValue::Dyn(ty) => ty.clone(),
            StaticValue::Arr(elem_sv) => {
                // Array type - create Array(size, element_type)
                // Use a fresh type variable for size since we don't track it statically
                let elem_type = elem_sv.to_type(gen);
                Type::Constructed(TypeName::Array, vec![gen.new_variable(), elem_type])
            }
            StaticValue::Rcd(fields) => {
                // Record type - extract field types
                // Field order doesn't matter due to custom PartialEq for TypeName::Record
                let field_types: Vec<(String, Type)> =
                    fields.iter().map(|(name, sv)| (name.clone(), sv.to_type(gen))).collect();
                crate::ast::types::record(field_types)
            }
            StaticValue::Lam(_param, _body) => {
                // Lambda - treat as opaque with fresh type variable
                // We could construct a proper function type (param_types -> ret_type),
                // but since lambdas get defunctionalized into named functions,
                // the type checker will infer the correct type from the generated function.
                gen.new_variable()
            }
            StaticValue::Closure { env, .. } => {
                // Closure value is a record { __tag: i32, <env fields...> }
                let mut fields: Vec<(String, Type)> = Vec::with_capacity(env.len() + 1);
                fields.push(("__tag".to_string(), crate::ast::types::i32()));
                for (k, v) in env {
                    fields.push((k.clone(), v.to_type(gen)));
                }
                crate::ast::types::record(fields)
            }
        }
    }
}

/// Generated function for defunctionalized lambda
#[derive(Debug, Clone)]
pub struct DefunctionalizedFunction {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Type,
    pub body: Expression,
}

/// Metadata for a lambda used to generate __applyN dispatchers
#[derive(Debug, Clone)]
struct LambdaMeta {
    tag: i32,
    name: String,
    arity: usize,
}

pub struct Defunctionalizer<T: crate::type_checker::TypeVarGenerator> {
    /// Counter for generating fresh IDs (used for function names, temporary variables, etc.)
    next_id: usize,
    type_var_gen: T,
    /// Per-declaration bucket for generated lambda functions
    current_bucket: Vec<DefunctionalizedFunction>,
    /// Stack of enclosing declaration names for better lambda naming
    enclosing_decl_stack: Vec<String>,
    /// Registry of lambdas in current bucket for __applyN generation
    lambda_registry: Vec<LambdaMeta>,
    node_counter: NodeCounter,
}

impl<T: crate::type_checker::TypeVarGenerator> Defunctionalizer<T> {
    pub fn new_with_counter(node_counter: NodeCounter, type_var_gen: T) -> Self {
        Defunctionalizer {
            next_id: 0,
            type_var_gen,
            current_bucket: Vec::new(),
            enclosing_decl_stack: Vec::new(),
            lambda_registry: Vec::new(),
            node_counter,
        }
    }

    /// Get the next syntax ID for naming lambdas, temporary variables, etc.
    fn next_syntax_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn take_type_var_gen(self) -> T {
        self.type_var_gen
    }

    /// Helper: Create a StaticValue::Dyn with a fresh type variable
    fn dyn_unknown(&mut self) -> StaticValue {
        StaticValue::Dyn(self.type_var_gen.new_variable())
    }

    /// Helper: Create an expression node with dummy span
    fn mk(&mut self, kind: ExprKind) -> Expression {
        self.node_counter.mk_node(kind, Span::dummy())
    }

    /// Extract the __tag value from a closure record literal, if present
    fn extract_closure_tag(&self, e: &Expression) -> Option<i32> {
        if let ExprKind::RecordLiteral(fields) = &e.kind {
            for (k, v) in fields {
                if k == "__tag" {
                    if let ExprKind::IntLiteral(t) = v.kind {
                        return Some(t);
                    }
                }
            }
        }
        None
    }

    /// Look up the lambda name for a given tag in the current bucket
    fn lambda_name_for_tag(&self, tag: i32) -> Option<&str> {
        self.lambda_registry.iter().find(|m| m.tag == tag).map(|m| m.name.as_str())
    }

    /// Extract type from a StaticValue for closure environment fields.
    /// With annotated types in scope, this should return concrete types from Dyn variants.
    fn env_field_type_from_sv(&mut self, sv: &StaticValue) -> Type {
        // Simply use to_type - if annotations are present, SVs will be Dyn(concrete_type)
        sv.to_type(&mut self.type_var_gen)
    }

    /// Helper: Create an expression with Dyn static value (for unknown types)
    fn ret_dyn(&mut self, kind: ExprKind) -> (Expression, StaticValue) {
        let expr = self.mk(kind);
        (expr, self.dyn_unknown())
    }

    /// Called at the start of transforming a top-level declaration
    fn begin_decl(&mut self, name: &str) {
        self.current_bucket.clear();
        self.lambda_registry.clear();
        self.enclosing_decl_stack.push(name.to_string());
    }

    /// Called at the end of transforming a top-level declaration
    fn end_decl(&mut self) {
        self.enclosing_decl_stack.pop();
        self.current_bucket.clear();
        self.lambda_registry.clear();
    }

    /// Generate __applyN dispatcher for a given arity
    fn generate_apply_dispatcher(&mut self, arity: usize, lambdas: &[LambdaMeta]) -> Declaration {
        // Parameters: __closure, a0, a1, ..., a_{N-1}
        let mut params = Vec::with_capacity(arity + 1);

        // Closure parameter
        let closure_pat =
            self.node_counter.mk_node(PatternKind::Name("__closure".to_string()), Span::dummy());
        let closure_ty = self.type_var_gen.new_variable();
        let closure_param = self.node_counter.mk_node(
            PatternKind::Typed(Box::new(closure_pat), closure_ty),
            Span::dummy(),
        );
        params.push(closure_param);

        // Application argument parameters: a0, a1, ..., a_{arity-1}
        for i in 0..arity {
            let arg_pat = self.node_counter.mk_node(PatternKind::Name(format!("a{}", i)), Span::dummy());
            let arg_ty = self.type_var_gen.new_variable();
            let arg_param =
                self.node_counter.mk_node(PatternKind::Typed(Box::new(arg_pat), arg_ty), Span::dummy());
            params.push(arg_param);
        }

        // Build match expression on __closure.__tag
        // For now, use a chained if-else structure since we don't have pattern matching on integers
        let closure_ident = self.mk(ExprKind::Identifier("__closure".to_string()));
        let tag_access = self.mk(ExprKind::FieldAccess(
            Box::new(closure_ident),
            "__tag".to_string(),
        ));

        let mut body_expr = None;

        // Build from last case to first (to construct nested if-else)
        for lambda in lambdas.iter().rev() {
            let tag_lit = self.mk(ExprKind::IntLiteral(lambda.tag));
            let tag_check = self.mk(ExprKind::BinaryOp(
                BinaryOp { op: "==".to_string() },
                Box::new(tag_access.clone()),
                Box::new(tag_lit),
            ));

            // Build function call: __lam_name(__closure, a0, a1, ..., a_{arity-1})
            let mut call_args = vec![self.mk(ExprKind::Identifier("__closure".to_string()))];
            for i in 0..arity {
                call_args.push(self.mk(ExprKind::Identifier(format!("a{}", i))));
            }
            let then_branch = self.mk(ExprKind::FunctionCall(lambda.name.clone(), call_args));

            if let Some(else_branch) = body_expr {
                body_expr = Some(self.mk(ExprKind::If(IfExpr {
                    condition: Box::new(tag_check),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                })));
            } else {
                // Last case (no else branch - this should never be reached if tags are correct)
                body_expr = Some(then_branch);
            }
        }

        let body = body_expr.unwrap_or_else(|| {
            // Empty dispatcher - should never happen, but provide a dummy error
            self.mk(ExprKind::IntLiteral(-1))
        });

        Declaration::Decl(Decl {
            keyword: "def",
            attributes: vec![],
            name: format!("__apply{}", arity),
            size_params: vec![],
            type_params: vec![],
            params,
            ty: None,
            body,
        })
    }

    /// Drain the current bucket into a vector of Declarations
    /// Note: No longer generates global __applyN dispatchers
    /// Dispatchers are now generated on-demand at call sites when needed
    fn drain_bucket_as_decls(&mut self) -> Vec<Declaration> {
        let mut out = Vec::with_capacity(self.current_bucket.len());

        // Add the lambda functions
        for func in self.current_bucket.drain(..) {
            out.push(Declaration::Decl(Decl {
                keyword: "def",
                attributes: vec![],
                name: func.name,
                size_params: vec![],
                type_params: vec![],
                params: func
                    .params
                    .iter()
                    .map(|p| {
                        let pat =
                            self.node_counter.mk_node(PatternKind::Name(p.name.clone()), Span::dummy());
                        self.node_counter
                            .mk_node(PatternKind::Typed(Box::new(pat), p.ty.clone()), Span::dummy())
                    })
                    .collect(),
                ty: Some(func.return_type),
                body: func.body,
            }));
        }

        // No longer generating global __applyN dispatchers here
        // They are generated on-demand at call sites when direct calls aren't possible

        out
    }

    /// Extract function name and existing args from a callable expression
    /// Returns (name, existing_args) where existing_args is empty for simple calls
    fn as_callable(expr: &Expression) -> Result<(String, Vec<Expression>)> {
        match &expr.kind {
            ExprKind::Identifier(name) => Ok((name.clone(), vec![])),
            ExprKind::FieldAccess(base, field) => {
                // Qualified name like f32.cos - convert to dotted name
                let qual_name =
                    crate::ast::QualName::new(vec![Self::extract_base_name(base)?], field.clone());
                Ok((qual_name.to_dotted(), vec![]))
            }
            ExprKind::FunctionCall(name, existing_args) => {
                // Partial application - return existing args to be extended
                Ok((name.clone(), existing_args.clone()))
            }
            _ => Err(CompilerError::DefunctionalizationError(format!(
                "Invalid function in application: {:?}",
                expr.kind
            ))),
        }
    }

    pub fn defunctionalize_program(&mut self, program: &Program) -> Result<Program> {
        let mut new_declarations = Vec::with_capacity(program.declarations.len() * 2);
        let mut scope_stack = ScopeStack::new();

        // Process each declaration, flushing generated lambdas before the declaration itself
        for decl in &program.declarations {
            match decl {
                Declaration::Decl(decl_node) => {
                    self.begin_decl(&decl_node.name);

                    if decl_node.params.is_empty() {
                        // Variable declarations (let/def with no params) - may contain Application nodes
                        let (transformed_decl, _sv) =
                            self.defunctionalize_decl(decl_node, &mut scope_stack)?;
                        // Flush generated lambdas before this declaration
                        new_declarations.extend(self.drain_bucket_as_decls());
                        new_declarations.push(transformed_decl);
                    } else {
                        // Function declarations with params - need to defunctionalize body
                        let (transformed_body, _sv) =
                            self.defunctionalize_expression(&decl_node.body, &mut scope_stack)?;
                        // Flush generated lambdas before this declaration
                        new_declarations.extend(self.drain_bucket_as_decls());
                        let mut transformed_decl = decl_node.clone();
                        transformed_decl.body = transformed_body;
                        new_declarations.push(Declaration::Decl(transformed_decl));
                    }

                    self.end_decl();
                }
                Declaration::Entry(entry) => {
                    self.begin_decl(&entry.name);

                    // Entry points need defunctionalization too
                    let (transformed_body, _sv) =
                        self.defunctionalize_expression(&entry.body, &mut scope_stack)?;
                    // Flush generated lambdas before this entry
                    new_declarations.extend(self.drain_bucket_as_decls());
                    let mut transformed_entry = entry.clone();
                    transformed_entry.body = transformed_body;
                    new_declarations.push(Declaration::Entry(transformed_entry));

                    self.end_decl();
                }
                Declaration::Uniform(uniform_decl) => {
                    // Uniform declarations have no body
                    new_declarations.push(Declaration::Uniform(uniform_decl.clone()));
                }
                Declaration::Val(val_decl) => {
                    // Type signatures only
                    new_declarations.push(Declaration::Val(val_decl.clone()));
                }
                Declaration::TypeBind(_) => {
                    return Err(CompilerError::DefunctionalizationError(
                        "Type bindings are not yet supported in defunctionalization".to_string(),
                    ));
                }
                Declaration::ModuleBind(_) => {
                    return Err(CompilerError::DefunctionalizationError(
                        "Module bindings are not yet supported in defunctionalization".to_string(),
                    ));
                }
                Declaration::ModuleTypeBind(_) => {
                    return Err(CompilerError::DefunctionalizationError(
                        "Module type bindings are not yet supported in defunctionalization".to_string(),
                    ));
                }
                Declaration::Open(_) => {
                    return Err(CompilerError::DefunctionalizationError(
                        "Open declarations are not yet supported in defunctionalization".to_string(),
                    ));
                }
                Declaration::Import(_) => {
                    return Err(CompilerError::DefunctionalizationError(
                        "Import declarations are not yet supported in defunctionalization".to_string(),
                    ));
                }
                Declaration::Local(_) => {
                    return Err(CompilerError::DefunctionalizationError(
                        "Local declarations are not yet supported in defunctionalization".to_string(),
                    ));
                }
            }
        }

        // All generated lambdas have been flushed into new_declarations
        // right before their enclosing declaration, so we're done
        Ok(Program {
            declarations: new_declarations,
        })
    }

    fn defunctionalize_decl(
        &mut self,
        decl: &Decl,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Declaration, StaticValue)> {
        let (transformed_expr, sv) = self.defunctionalize_expression(&decl.body, scope_stack)?;

        // Add the binding to the current scope
        scope_stack.insert(decl.name.clone(), sv.clone());

        let transformed_decl = Declaration::Decl(Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
            params: decl.params.clone(),
            ty: decl.ty.clone(),
            body: transformed_expr,
        });

        Ok((transformed_decl, sv))
    }

    fn defunctionalize_expression(
        &mut self,
        expr: &Expression,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        let span = expr.h.span;
        match &expr.kind {
            ExprKind::IntLiteral(n) => Ok((
                self.node_counter.mk_node(ExprKind::IntLiteral(*n), span),
                StaticValue::Dyn(types::i32()),
            )),
            ExprKind::FloatLiteral(f) => Ok((
                self.node_counter.mk_node(ExprKind::FloatLiteral(*f), span),
                StaticValue::Dyn(types::f32()),
            )),
            ExprKind::BoolLiteral(b) => Ok((
                self.node_counter.mk_node(ExprKind::BoolLiteral(*b), span),
                StaticValue::Dyn(types::bool_type()),
            )),
            ExprKind::Identifier(name) => {
                if let Ok(sv) = scope_stack.lookup(name) {
                    match sv {
                        StaticValue::Dyn(_) => {
                            // Regular variable reference
                            Ok((
                                self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                                sv.clone(),
                            ))
                        }
                        StaticValue::Lam(_, _) => {
                            // Reference to a function - this would need special handling
                            // For now, keep as identifier
                            Ok((
                                self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                                sv.clone(),
                            ))
                        }
                        StaticValue::Rcd(_) => {
                            // Reference to a closure record
                            Ok((
                                self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                                sv.clone(),
                            ))
                        }
                        StaticValue::Closure { .. } => {
                            // Reference to a defunctionalized closure
                            Ok((
                                self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                                sv.clone(),
                            ))
                        }
                        StaticValue::Arr(_) => Ok((
                            self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                            sv.clone(),
                        )),
                    }
                } else {
                    // Unknown variable - assume dynamic with type variable
                    Ok((
                        self.node_counter.mk_node(ExprKind::Identifier(name.clone()), span),
                        StaticValue::Dyn(self.type_var_gen.new_variable()),
                    ))
                }
            }
            ExprKind::Lambda(lambda) => self.defunctionalize_lambda(lambda, scope_stack),
            ExprKind::Application(func, args) => self.defunctionalize_application(func, args, scope_stack),
            ExprKind::ArrayLiteral(elements) => {
                let mut transformed_elements = Vec::new();
                let mut element_sv = None;

                for elem in elements {
                    let (transformed_elem, sv) = self.defunctionalize_expression(elem, scope_stack)?;
                    transformed_elements.push(transformed_elem);

                    // All elements should have the same static value structure
                    if element_sv.is_none() {
                        element_sv = Some(sv);
                    }
                }

                // Array static value wraps the element's static value
                // Use fresh type variable if array is empty
                let array_sv = StaticValue::Arr(Box::new(element_sv.unwrap_or_else(|| self.dyn_unknown())));
                Ok((
                    self.node_counter.mk_node(ExprKind::ArrayLiteral(transformed_elements), span),
                    array_sv,
                ))
            }
            ExprKind::ArrayIndex(array, index) => {
                let (transformed_array, array_sv) = self.defunctionalize_expression(array, scope_stack)?;
                let (transformed_index, _index_sv) = self.defunctionalize_expression(index, scope_stack)?;

                // Extract element type from array if available
                let elem_sv = match array_sv {
                    StaticValue::Arr(elem) => *elem,
                    _ => self.dyn_unknown(),
                };

                Ok((
                    self.node_counter.mk_node(
                        ExprKind::ArrayIndex(Box::new(transformed_array), Box::new(transformed_index)),
                        span,
                    ),
                    elem_sv,
                ))
            }
            ExprKind::BinaryOp(op, left, right) => {
                let (transformed_left, left_sv) = self.defunctionalize_expression(left, scope_stack)?;
                let (transformed_right, right_sv) = self.defunctionalize_expression(right, scope_stack)?;

                // For binary arithmetic operations, the result type should be the same as the operand types
                // (assuming type checking has already ensured they match)
                let result_type = match (&left_sv, &right_sv) {
                    (StaticValue::Dyn(left_type), StaticValue::Dyn(_right_type)) => {
                        // Use the left operand type (they should be the same after type checking)
                        left_type.clone()
                    }
                    (StaticValue::Dyn(ty), _) | (_, StaticValue::Dyn(ty)) => {
                        // If one is dynamic, use that type
                        ty.clone()
                    }
                    _ => {
                        // Fallback to a fresh type variable if we can't determine the type
                        self.type_var_gen.new_variable()
                    }
                };

                Ok((
                    self.node_counter.mk_node(
                        ExprKind::BinaryOp(
                            op.clone(),
                            Box::new(transformed_left),
                            Box::new(transformed_right),
                        ),
                        span,
                    ),
                    StaticValue::Dyn(result_type),
                ))
            }
            ExprKind::FunctionCall(name, args) => {
                // Check if the function name refers to a closure in scope
                if let Ok(sv) = scope_stack.lookup(name) {
                    match sv {
                        StaticValue::Closure { lam_name, .. } => {
                            // Clone lam_name before mutable borrow
                            let lam_name = lam_name.clone();

                            // Direct call to known lambda
                            let mut transformed_args = Vec::new();
                            for arg in args {
                                let (transformed_arg, _sv) =
                                    self.defunctionalize_expression(arg, scope_stack)?;
                                transformed_args.push(transformed_arg);
                            }

                            // Call __lam_* with closure identifier and args
                            let closure_ident = self.mk(ExprKind::Identifier(name.clone()));
                            let mut all_args = vec![closure_ident];
                            all_args.extend(transformed_args);

                            return Ok(self.ret_dyn(ExprKind::FunctionCall(lam_name, all_args)));
                        }
                        StaticValue::Rcd(_) => {
                            // Legacy path: closure as Rcd - convert to __applyN
                            let arity = args.len();
                            let apply_name = format!("__apply{}", arity);

                            // Transform arguments
                            let mut transformed_args = Vec::new();
                            for arg in args {
                                let (transformed_arg, _sv) =
                                    self.defunctionalize_expression(arg, scope_stack)?;
                                transformed_args.push(transformed_arg);
                            }

                            // Call __applyN with closure identifier and args
                            let closure_ident = self.mk(ExprKind::Identifier(name.clone()));
                            let mut all_args = vec![closure_ident];
                            all_args.extend(transformed_args);

                            return Ok(self.ret_dyn(ExprKind::FunctionCall(apply_name, all_args)));
                        }
                        _ => {}
                    }
                }

                // Transform arguments first
                let mut transformed_args = Vec::new();
                for arg in args {
                    let (transformed_arg, _sv) = self.defunctionalize_expression(arg, scope_stack)?;
                    transformed_args.push(transformed_arg);
                }

                // Special handling for higher-order builtins like map
                if name == "map" && transformed_args.len() == 2 {
                    // map f xs -> loop-based implementation
                    return self.defunctionalize_map(
                        &transformed_args[0],
                        &transformed_args[1],
                        scope_stack,
                    );
                }

                // Regular function calls (first-order) remain unchanged
                Ok((
                    self.node_counter.mk_node(ExprKind::FunctionCall(name.clone(), transformed_args), span),
                    StaticValue::Dyn(self.type_var_gen.new_variable()),
                ))
            }
            ExprKind::Tuple(elements) => {
                let mut transformed_elements = Vec::new();
                let mut element_types = Vec::new();

                for elem in elements {
                    let (transformed_elem, sv) = self.defunctionalize_expression(elem, scope_stack)?;
                    transformed_elements.push(transformed_elem);

                    // Extract type from static value
                    let elem_type = match sv {
                        StaticValue::Dyn(ty) => ty,
                        _ => self.type_var_gen.new_variable(),
                    };
                    element_types.push(elem_type);
                }

                Ok((
                    self.node_counter.mk_node(ExprKind::Tuple(transformed_elements), span),
                    StaticValue::Dyn(types::tuple(element_types)),
                ))
            }
            ExprKind::RecordLiteral(fields) => {
                let mut transformed_fields = Vec::new();
                let mut field_svs = HashMap::new();

                for (field_name, field_expr) in fields {
                    let (transformed_expr, sv) =
                        self.defunctionalize_expression(field_expr, scope_stack)?;
                    transformed_fields.push((field_name.clone(), transformed_expr));
                    field_svs.insert(field_name.clone(), sv);
                }

                Ok((
                    self.node_counter.mk_node(ExprKind::RecordLiteral(transformed_fields), span),
                    StaticValue::Rcd(field_svs),
                ))
            }
            ExprKind::LetIn(let_in) => {
                // Transform the value expression
                let (transformed_value, value_sv) =
                    self.defunctionalize_expression(&let_in.value, scope_stack)?;

                // Extract type annotation from either let_in.ty or the pattern (if it's Typed)
                let annotated_ty = let_in.ty.as_ref().or_else(|| {
                    if let PatternKind::Typed(_, ty) = &let_in.pattern.kind { Some(ty) } else { None }
                });

                // Prefer the annotated type if present for closure environment typing
                let annotated_sv = annotated_ty.map(|ty| StaticValue::Dyn(ty.clone()));
                let sv_for_binding = annotated_sv.as_ref().unwrap_or(&value_sv);

                // Push new scope and add bindings based on pattern structure
                scope_stack.push_scope();
                self.bind_pattern(&let_in.pattern, sv_for_binding, scope_stack)?;

                // Transform the body expression
                let (transformed_body, body_sv) =
                    self.defunctionalize_expression(&let_in.body, scope_stack)?;

                // Pop scope
                scope_stack.pop_scope();

                Ok((
                    self.node_counter.mk_node(
                        ExprKind::LetIn(crate::ast::LetInExpr {
                            pattern: let_in.pattern.clone(),
                            ty: let_in.ty.clone(),
                            value: Box::new(transformed_value),
                            body: Box::new(transformed_body),
                        }),
                        span,
                    ),
                    body_sv,
                ))
            }
            ExprKind::FieldAccess(expr, field) => {
                let (transformed_expr, expr_sv) = self.defunctionalize_expression(expr, scope_stack)?;

                // Extract the field's static value from the base if it's a record
                let field_sv = match &expr_sv {
                    StaticValue::Rcd(fields) => {
                        // Look up the field in the record
                        fields.get(field).cloned().unwrap_or_else(|| self.dyn_unknown())
                    }
                    _ => {
                        // Not a record, can't extract field info - use type variable
                        self.dyn_unknown()
                    }
                };

                Ok((
                    self.node_counter.mk_node(
                        ExprKind::FieldAccess(Box::new(transformed_expr), field.clone()),
                        span,
                    ),
                    field_sv,
                ))
            }
            ExprKind::If(if_expr) => {
                let (condition, _condition_sv) =
                    self.defunctionalize_expression(&if_expr.condition, scope_stack)?;
                let (then_branch, _then_sv) =
                    self.defunctionalize_expression(&if_expr.then_branch, scope_stack)?;
                let (else_branch, _else_sv) =
                    self.defunctionalize_expression(&if_expr.else_branch, scope_stack)?;
                Ok(self.ret_dyn(ExprKind::If(IfExpr {
                    condition: Box::new(condition),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                })))
            }

            ExprKind::TypeHole => Ok(self.ret_dyn(ExprKind::TypeHole)),

            ExprKind::Pipe(left, right) => {
                // a |> f desugars to f(a)
                // Defunctionalize both sides
                let (left_expr, _left_sv) = self.defunctionalize_expression(left, scope_stack)?;
                let (right_expr, _right_sv) = self.defunctionalize_expression(right, scope_stack)?;

                // Desugar into application
                self.defunctionalize_application(&right_expr, &[left_expr], scope_stack)
            }

            ExprKind::UnaryOp(op, operand) => {
                let (transformed_operand, sv) = self.defunctionalize_expression(operand, scope_stack)?;
                Ok((
                    self.node_counter
                        .mk_node(ExprKind::UnaryOp(op.clone(), Box::new(transformed_operand)), span),
                    sv,
                ))
            }

            ExprKind::Loop(loop_expr) => {
                // Convert LoopExpr to InternalLoop by desugaring patterns
                self.desugar_loop_to_internal(loop_expr, scope_stack, span)
            }

            ExprKind::InternalLoop(_) => Err(CompilerError::DefunctionalizationError(
                "InternalLoop should not appear as input to defunctionalization".to_string(),
            )),

            ExprKind::TypeAscription(inner, ty) => {
                let (transformed_inner, sv) = self.defunctionalize_expression(inner, scope_stack)?;
                Ok((
                    self.node_counter.mk_node(
                        ExprKind::TypeAscription(Box::new(transformed_inner), ty.clone()),
                        span,
                    ),
                    sv,
                ))
            }

            ExprKind::TypeCoercion(inner, ty) => {
                let (transformed_inner, sv) = self.defunctionalize_expression(inner, scope_stack)?;
                Ok((
                    self.node_counter.mk_node(
                        ExprKind::TypeCoercion(Box::new(transformed_inner), ty.clone()),
                        span,
                    ),
                    sv,
                ))
            }

            ExprKind::QualifiedName(_, _) => {
                // Qualified names are already resolved - pass through
                Ok((expr.clone(), self.dyn_unknown()))
            }

            ExprKind::Match(_) => {
                // Match expressions not yet fully supported
                Err(CompilerError::DefunctionalizationError(
                    "Match expressions not yet supported in defunctionalization".to_string(),
                ))
            }

            ExprKind::Range(range_expr) => {
                let (start_transformed, _) =
                    self.defunctionalize_expression(&range_expr.start, scope_stack)?;
                let (end_transformed, _) = self.defunctionalize_expression(&range_expr.end, scope_stack)?;
                let step_transformed = if let Some(step) = &range_expr.step {
                    let (transformed, _) = self.defunctionalize_expression(step, scope_stack)?;
                    Some(Box::new(transformed))
                } else {
                    None
                };
                Ok((
                    self.node_counter.mk_node(
                        ExprKind::Range(RangeExpr {
                            start: Box::new(start_transformed),
                            step: step_transformed,
                            end: Box::new(end_transformed),
                            kind: range_expr.kind.clone(),
                        }),
                        span,
                    ),
                    self.dyn_unknown(),
                ))
            }

            ExprKind::Unsafe(inner) => {
                let (transformed_inner, sv) = self.defunctionalize_expression(inner, scope_stack)?;
                Ok((
                    self.node_counter.mk_node(ExprKind::Unsafe(Box::new(transformed_inner)), span),
                    sv,
                ))
            }

            ExprKind::Assert(cond, inner) => {
                let (transformed_cond, _) = self.defunctionalize_expression(cond, scope_stack)?;
                let (transformed_inner, sv) = self.defunctionalize_expression(inner, scope_stack)?;
                Ok((
                    self.node_counter.mk_node(
                        ExprKind::Assert(Box::new(transformed_cond), Box::new(transformed_inner)),
                        span,
                    ),
                    sv,
                ))
            }
        } // NEWCASESHERE - add new cases before this closing brace
    }

    fn defunctionalize_lambda(
        &mut self,
        lambda: &LambdaExpr,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        // Find free variables in the lambda body
        // Build set of bound parameters
        let mut bound = HashSet::new();
        for p in &lambda.params {
            if let Some(n) = p.simple_name() {
                bound.insert(n.to_string());
            }
        }
        let free_vars = self.find_free_variables(&lambda.body, &bound)?;

        // Create a closure record with free variables
        let mut closure_fields = HashMap::new();
        for var in &free_vars {
            if let Ok(sv) = scope_stack.lookup(var) {
                closure_fields.insert(var.clone(), sv.clone());
            }
        }

        // Generate a unique function name with enclosing declaration context
        let id = self.next_syntax_id();
        let top = self.enclosing_decl_stack.last().map(String::as_str).unwrap_or("top");
        let func_name = format!("__lam_{}_{}", top, id);

        // Assign a unique tag for this lambda (per-bucket tag space)
        let tag = self.lambda_registry.len() as i32;
        let arity = lambda.params.len();

        // Build ONE canonical closure record type by extracting concrete types from scope
        // This ensures the lambda parameter type matches the actual closure record type
        let mut sorted_free_vars: Vec<_> = free_vars.iter().collect();
        sorted_free_vars.sort();

        let mut closure_type_fields = Vec::with_capacity(sorted_free_vars.len() + 1);
        closure_type_fields.push(("__tag".to_string(), crate::ast::types::i32()));

        for var in sorted_free_vars {
            let sv = scope_stack.lookup(var).map_err(|_| {
                CompilerError::DefunctionalizationError(format!(
                    "Free variable '{}' not in scope when building closure type",
                    var
                ))
            })?;
            let ty = self.env_field_type_from_sv(sv);
            closure_type_fields.push((var.clone(), ty));
        }

        let closure_type = crate::ast::types::record(closure_type_fields);

        // Create parameters: ALWAYS add __closure parameter with canonical type
        let mut func_params = Vec::new();

        func_params.push(Parameter {
            attributes: vec![],
            name: "__closure".to_string(),
            ty: closure_type.clone(),
        });

        for param in &lambda.params {
            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    CompilerError::DefunctionalizationError(
                        "Complex patterns in lambda parameters not yet supported".to_string(),
                    )
                })?
                .to_string();
            let param_ty =
                param.pattern_type().cloned().unwrap_or_else(|| self.type_var_gen.new_variable());
            func_params.push(Parameter {
                attributes: vec![],
                name: param_name,
                ty: param_ty,
            });
        }

        // Rewrite free variable references in lambda body to access them from __closure
        let rewritten_body = if !free_vars.is_empty() {
            self.rewrite_free_variables(&*lambda.body, &free_vars)
        } else {
            (*lambda.body).clone()
        };

        // Transform lambda body with parameter scope
        scope_stack.push_scope();
        for param in &lambda.params {
            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    CompilerError::DefunctionalizationError(
                        "Complex patterns in lambda parameters not yet supported".to_string(),
                    )
                })?
                .to_string();
            let param_ty =
                param.pattern_type().cloned().unwrap_or_else(|| self.type_var_gen.new_variable());
            scope_stack.insert(param_name, StaticValue::Dyn(param_ty));
        }

        let (transformed_body, _body_sv) = self.defunctionalize_expression(&rewritten_body, scope_stack)?;

        // Pop parameter scope
        scope_stack.pop_scope();

        // Create the generated function and push to current bucket
        let return_type = lambda.return_type.clone().unwrap_or_else(|| self.type_var_gen.new_variable());
        let generated_func = DefunctionalizedFunction {
            name: func_name.clone(),
            params: func_params,
            return_type,
            body: transformed_body,
        };

        self.current_bucket.push(generated_func);

        // Register lambda in the registry for __applyN generation
        self.lambda_registry.push(LambdaMeta {
            tag,
            name: func_name.clone(),
            arity,
        });

        // Create closure constructor expression - ALWAYS a record with __tag
        let closure_record = self.create_closure_record(tag, &free_vars)?;

        // Return StaticValue::Closure with precise call-target info
        let sv = StaticValue::Closure {
            tag,
            lam_name: func_name.clone(),
            arity,
            env: closure_fields, // note: without __tag
        };

        Ok((closure_record, sv))
    }

    fn defunctionalize_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        let (transformed_func, func_sv) = self.defunctionalize_expression(func, scope_stack)?;

        let mut transformed_args = Vec::new();
        for arg in args {
            let (transformed_arg, _arg_sv) = self.defunctionalize_expression(arg, scope_stack)?;
            transformed_args.push(transformed_arg);
        }

        match &func_sv {
            StaticValue::Closure { lam_name, .. } => {
                // Direct call to known lambda: __lam_foo(closure, args...)
                let mut all_args = vec![transformed_func];
                all_args.extend(transformed_args);
                Ok(self.ret_dyn(ExprKind::FunctionCall(lam_name.clone(), all_args)))
            }
            StaticValue::Rcd(_closure_fields) => {
                // Legacy path for record literals without Closure SV
                // Try to extract tag from the expression (for inline record literals)
                if let Some(tag) = self.extract_closure_tag(&transformed_func) {
                    if let Some(lam_name) = self.lambda_name_for_tag(tag) {
                        // Direct call to known lambda: __lam_foo(closure, args...)
                        let mut all_args = vec![transformed_func];
                        all_args.extend(transformed_args);
                        return Ok(self.ret_dyn(ExprKind::FunctionCall(lam_name.to_string(), all_args)));
                    }
                }

                // Fallback: use __applyN dispatcher
                let arity = transformed_args.len();
                let apply_name = format!("__apply{}", arity);

                // Call __applyN with closure and args
                let mut all_args = vec![transformed_func];
                all_args.extend(transformed_args);

                Ok(self.ret_dyn(ExprKind::FunctionCall(apply_name, all_args)))
            }
            _ => {
                // Regular function call - extract name and existing args, then append new args
                if let ExprKind::Application(nested_func, nested_args) = &transformed_func.kind {
                    // Recursive application - shouldn't happen after defunctionalization
                    // Handle by recursively processing
                    return self.defunctionalize_application(
                        &Expression {
                            h: transformed_func.h.clone(),
                            kind: ExprKind::Application(nested_func.clone(), nested_args.clone()),
                        },
                        args,
                        scope_stack,
                    );
                }

                let (func_name, existing_args) = Self::as_callable(&transformed_func)?;

                // Special handling for higher-order builtins like map
                if func_name == "map" && existing_args.is_empty() && transformed_args.len() == 2 {
                    // map f xs -> loop-based implementation
                    return self.defunctionalize_map(
                        &transformed_args[0],
                        &transformed_args[1],
                        scope_stack,
                    );
                }

                // Append new args to existing args (for partial application)
                let mut all_args = existing_args;
                all_args.extend(transformed_args);

                Ok(self.ret_dyn(ExprKind::FunctionCall(func_name, all_args)))
            }
        }
    }

    /// Extract the base identifier name from an expression
    /// E.g., for `f32` in `f32.cos`, returns "f32"
    fn extract_base_name(expr: &Expression) -> Result<String> {
        match &expr.kind {
            ExprKind::Identifier(name) => Ok(name.clone()),
            _ => Err(CompilerError::DefunctionalizationError(format!(
                "Expected identifier as base of qualified name, got {:?}",
                expr.kind
            ))),
        }
    }

    fn find_free_variables(
        &self,
        expr: &Expression,
        bound_vars: &HashSet<String>,
    ) -> Result<HashSet<String>> {
        let mut free_vars = HashSet::new();
        self.collect_free_variables(expr, bound_vars, &mut free_vars)?;
        Ok(free_vars)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_free_variables(
        &self,
        expr: &Expression,
        bound_vars: &HashSet<String>,
        free_vars: &mut HashSet<String>,
    ) -> Result<()> {
        match &expr.kind {
            ExprKind::Identifier(name) => {
                if !bound_vars.contains(name) {
                    free_vars.insert(name.clone());
                }
            }
            ExprKind::RecordLiteral(fields) => {
                for (_name, field_expr) in fields {
                    self.collect_free_variables(field_expr, bound_vars, free_vars)?;
                }
            }
            ExprKind::Lambda(lambda) => {
                let mut extended_bound = bound_vars.clone();
                for param in &lambda.params {
                    if let Some(name) = param.simple_name() {
                        extended_bound.insert(name.to_string());
                    }
                }
                self.collect_free_variables(&lambda.body, &extended_bound, free_vars)?;
            }
            ExprKind::Application(func, args) => {
                self.collect_free_variables(func, bound_vars, free_vars)?;
                for arg in args {
                    self.collect_free_variables(arg, bound_vars, free_vars)?;
                }
            }
            ExprKind::ArrayLiteral(elements) => {
                for elem in elements {
                    self.collect_free_variables(elem, bound_vars, free_vars)?;
                }
            }
            ExprKind::ArrayIndex(array, index) => {
                self.collect_free_variables(array, bound_vars, free_vars)?;
                self.collect_free_variables(index, bound_vars, free_vars)?;
            }
            ExprKind::BinaryOp(_, left, right) => {
                self.collect_free_variables(left, bound_vars, free_vars)?;
                self.collect_free_variables(right, bound_vars, free_vars)?;
            }
            ExprKind::FunctionCall(_, args) => {
                for arg in args {
                    self.collect_free_variables(arg, bound_vars, free_vars)?;
                }
            }
            ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.collect_free_variables(elem, bound_vars, free_vars)?;
                }
            }
            ExprKind::LetIn(let_in) => {
                // Collect free variables from value expression
                self.collect_free_variables(&let_in.value, bound_vars, free_vars)?;

                // Add all names from pattern to bound variables and collect from body
                let mut extended_bound = bound_vars.clone();
                for name in let_in.pattern.collect_names() {
                    extended_bound.insert(name);
                }
                self.collect_free_variables(&let_in.body, &extended_bound, free_vars)?;
            }
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::TypeHole => {
                // No free variables in literals or type holes
            }
            ExprKind::FieldAccess(expr, _field) => {
                self.collect_free_variables(expr, bound_vars, free_vars)?;
            }
            ExprKind::If(if_expr) => {
                self.collect_free_variables(&if_expr.condition, bound_vars, free_vars)?;
                self.collect_free_variables(&if_expr.then_branch, bound_vars, free_vars)?;
                self.collect_free_variables(&if_expr.else_branch, bound_vars, free_vars)?;
            }
            ExprKind::Pipe(left, right) => {
                self.collect_free_variables(left, bound_vars, free_vars)?;
                self.collect_free_variables(right, bound_vars, free_vars)?;
            }

            ExprKind::QualifiedName(_, _) => {
                // Qualified names don't contain free variables
            }
            ExprKind::UnaryOp(_, operand) => {
                self.collect_free_variables(operand, bound_vars, free_vars)?;
            }
            ExprKind::Loop(loop_expr) => {
                // Collect from init expression
                if let Some(init) = &loop_expr.init {
                    self.collect_free_variables(init, bound_vars, free_vars)?;
                }

                // Add loop pattern names to bound vars for body
                let mut loop_bound = bound_vars.clone();
                for name in loop_expr.pattern.collect_names() {
                    loop_bound.insert(name);
                }

                // Collect from loop form (condition/range)
                match &loop_expr.form {
                    LoopForm::For(_name, expr) => {
                        self.collect_free_variables(expr, &loop_bound, free_vars)?;
                    }
                    LoopForm::ForIn(_pattern, expr) => {
                        self.collect_free_variables(expr, bound_vars, free_vars)?;
                    }
                    LoopForm::While(cond) => {
                        self.collect_free_variables(cond, &loop_bound, free_vars)?;
                    }
                }

                // Collect from body
                self.collect_free_variables(&loop_expr.body, &loop_bound, free_vars)?;
            }
            ExprKind::InternalLoop(_) => {
                return Err(CompilerError::DefunctionalizationError(
                    "InternalLoop should not appear as input to defunctionalization".to_string(),
                ));
            }
            ExprKind::Match(_) => {
                // Match expressions not yet fully supported
                return Err(CompilerError::DefunctionalizationError(
                    "Match expressions not yet supported in defunctionalization".to_string(),
                ));
            }
            ExprKind::Range(range_expr) => {
                self.collect_free_variables(&range_expr.start, bound_vars, free_vars)?;
                self.collect_free_variables(&range_expr.end, bound_vars, free_vars)?;
            }
            ExprKind::TypeAscription(expr, _ty) | ExprKind::TypeCoercion(expr, _ty) => {
                self.collect_free_variables(expr, bound_vars, free_vars)?;
            }
            ExprKind::Unsafe(expr) => {
                self.collect_free_variables(expr, bound_vars, free_vars)?;
            }
            ExprKind::Assert(cond, expr) => {
                self.collect_free_variables(cond, bound_vars, free_vars)?;
                self.collect_free_variables(expr, bound_vars, free_vars)?;
            }
        } // NEWCASESHERE - add new cases before this closing brace
        Ok(())
    }

    fn create_closure_record(&mut self, tag: i32, free_vars: &HashSet<String>) -> Result<Expression> {
        // Create a record literal with __tag field and free variables
        // The __tag field is used by __applyN dispatchers to route to the correct function
        // The other field names match the variable names so that rewrite_free_variables works
        let mut fields = Vec::new();

        // Add __tag field with the integer tag
        let tag_value = self.mk(ExprKind::IntLiteral(tag));
        fields.push(("__tag".to_string(), tag_value));

        // Add free variable fields (sorted for determinism)
        let mut sorted_vars: Vec<_> = free_vars.iter().collect();
        sorted_vars.sort();
        for var in sorted_vars {
            let field_value = self.mk(ExprKind::Identifier(var.clone()));
            fields.push((var.clone(), field_value));
        }

        Ok(self.mk(ExprKind::RecordLiteral(fields)))
    }

    /// Transform map f xs into a loop that allocates output array and fills it
    /// map f xs  =>
    ///   let xs = <array> in
    ///   let len = length xs in
    ///   let out = replicate len 0 in  // Allocate array with default values
    ///   loop (i, out) = (0, out) while i < len do
    ///     let updated_out = __array_update(out, i, f xs[i]) in
    ///     (i + 1, updated_out)
    ///
    /// TODO: Eventually handle in-place updates to avoid copying output array each iteration
    fn defunctionalize_map(
        &mut self,
        func: &Expression,
        array: &Expression,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<(Expression, StaticValue)> {
        let span = Span::dummy();

        // func can be any expression (identifier or closure record)
        // We'll call __apply1 with it in the loop body

        // Generate unique variable names
        let map_id = self.next_syntax_id();
        let i_var = format!("__map_i_{}", map_id);
        let out_var = format!("__map_out_{}", map_id);
        let xs_var = format!("__map_xs_{}", map_id);
        let len_var = format!("__map_len_{}", map_id);
        let func_var = format!("__map_f_{}", map_id);

        // Build ALL leaf nodes first to avoid borrow checker issues
        let xs_ident_for_len = self.node_counter.mk_node(ExprKind::Identifier(xs_var.clone()), span);
        let xs_ident_for_alloc = self.node_counter.mk_node(ExprKind::Identifier(xs_var.clone()), span);
        let len_ident_for_loop = self.node_counter.mk_node(ExprKind::Identifier(len_var.clone()), span);

        // length xs
        let len_call = self.node_counter.mk_node(
            ExprKind::FunctionCall("length".to_string(), vec![xs_ident_for_len]),
            span,
        );

        // Initialize output array using __alloc_array: __alloc_array xs
        // This preserves the size of xs in the type system, so map [n]t -> [n]t'
        // __alloc_array has type ∀n t. [n]t -> [n]t, so it returns an array of the same size
        // SAFETY: Every element will be overwritten in the loop before the array is returned
        let init_out = self.node_counter.mk_node(
            ExprKind::FunctionCall("__alloc_array".to_string(), vec![xs_ident_for_alloc]),
            span,
        );

        // Build more leaf nodes for loop body
        let i_ident = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let len_ident = self.node_counter.mk_node(ExprKind::Identifier(len_var.clone()), span);
        let xs_ident = self.node_counter.mk_node(ExprKind::Identifier(xs_var.clone()), span);
        let i_ident2 = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let out_ident = self.node_counter.mk_node(ExprKind::Identifier(out_var.clone()), span);
        let i_ident3 = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let i_ident4 = self.node_counter.mk_node(ExprKind::Identifier(i_var.clone()), span);
        let one_lit = self.node_counter.mk_node(ExprKind::IntLiteral(1), span);
        let zero_lit = self.node_counter.mk_node(ExprKind::IntLiteral(0), span);
        let out_init_ident =
            self.node_counter.mk_node(ExprKind::Identifier(format!("{}_init", out_var)), span);

        // i < len
        let condition = self.node_counter.mk_node(
            ExprKind::BinaryOp(
                BinaryOp { op: "<".to_string() },
                Box::new(i_ident),
                Box::new(len_ident),
            ),
            span,
        );

        // xs[i]
        let array_index =
            self.node_counter.mk_node(ExprKind::ArrayIndex(Box::new(xs_ident), Box::new(i_ident2)), span);

        // Try to specialize the callee: if func is a closure record with a known __tag,
        // call the lambda directly instead of using __apply1 dispatcher
        let mut call_callee_is_direct = None;
        if let Some(tag) = self.extract_closure_tag(func) {
            if let Some(lam_name) = self.lambda_name_for_tag(tag) {
                call_callee_is_direct = Some(lam_name.to_string());
            }
        }

        // Build func identifier for call
        let func_ident = self.node_counter.mk_node(ExprKind::Identifier(func_var.clone()), span);

        // EITHER direct call to the known __lam_* ... OR generic dispatcher if callee not statically known
        let func_app = if let Some(lam_name) = call_callee_is_direct {
            // Direct call: __lam_foo(closure, xs[i])
            self.node_counter.mk_node(
                ExprKind::FunctionCall(lam_name, vec![func_ident.clone(), array_index.clone()]),
                span,
            )
        } else {
            // Generic dispatcher: __apply1(closure, xs[i])
            self.node_counter.mk_node(
                ExprKind::FunctionCall("__apply1".to_string(), vec![func_ident, array_index]),
                span,
            )
        };

        // __array_update(out, i, f xs[i])
        let updated_out = self.node_counter.mk_node(
            ExprKind::FunctionCall("__array_update".to_string(), vec![out_ident, i_ident3, func_app]),
            span,
        );

        // i + 1
        let i_inc = self.node_counter.mk_node(
            ExprKind::BinaryOp(
                BinaryOp { op: "+".to_string() },
                Box::new(i_ident4),
                Box::new(one_lit),
            ),
            span,
        );

        // (i + 1, updated_out)
        let loop_body = self.node_counter.mk_node(ExprKind::Tuple(vec![i_inc, updated_out]), span);

        // (0, out_init)
        let initial_value =
            self.node_counter.mk_node(ExprKind::Tuple(vec![zero_lit, out_init_ident]), span);

        // Create a standard Loop AST node: loop (i, out) = (0, out_init) while i < len do body
        let i_pattern = self.node_counter.mk_node(PatternKind::Name(i_var.clone()), span);
        let out_pattern = self.node_counter.mk_node(PatternKind::Name(out_var.clone()), span);
        let loop_pattern =
            self.node_counter.mk_node(PatternKind::Tuple(vec![i_pattern, out_pattern]), span);

        let loop_ast = LoopExpr {
            pattern: loop_pattern,
            init: Some(Box::new(initial_value)),
            form: LoopForm::While(Box::new(condition)),
            body: Box::new(loop_body),
        };

        // Now desugar this Loop to InternalLoop using the existing helper
        let (loop_expr, _) = self.desugar_loop_to_internal(&loop_ast, scope_stack, span)?;

        // let out = replicate len 0 in <loop>
        let out_init_pattern =
            self.node_counter.mk_node(crate::ast::PatternKind::Name(format!("{}_init", out_var)), span);
        let with_out_init = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: out_init_pattern,
                ty: None,
                value: Box::new(init_out),
                body: Box::new(loop_expr),
            }),
            span,
        );

        // let (_, result_out) = <loop> in result_out
        let wildcard_pattern = self.node_counter.mk_node(crate::ast::PatternKind::Wildcard, span);
        let result_out_pattern =
            self.node_counter.mk_node(crate::ast::PatternKind::Name(format!("{}_result", out_var)), span);
        let tuple_pattern = self.node_counter.mk_node(
            crate::ast::PatternKind::Tuple(vec![wildcard_pattern, result_out_pattern]),
            span,
        );
        let result_out_ident =
            self.node_counter.mk_node(ExprKind::Identifier(format!("{}_result", out_var)), span);
        let with_out = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: tuple_pattern,
                ty: None,
                value: Box::new(with_out_init),
                body: Box::new(result_out_ident),
            }),
            span,
        );

        // let len = length xs in <with_out>
        let len_pattern = self.node_counter.mk_node(crate::ast::PatternKind::Name(len_var), span);
        let with_len = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: len_pattern,
                ty: None,
                value: Box::new(len_call),
                body: Box::new(with_out),
            }),
            span,
        );

        // let xs = <array> in <with_len>
        let xs_pattern = self.node_counter.mk_node(crate::ast::PatternKind::Name(xs_var), span);
        let with_xs = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: xs_pattern,
                ty: None,
                value: Box::new(array.clone()),
                body: Box::new(with_len),
            }),
            span,
        );

        // let f = <func> in <with_xs>
        let func_pattern = self.node_counter.mk_node(crate::ast::PatternKind::Name(func_var), span);
        let result = self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern: func_pattern,
                ty: None,
                value: Box::new(func.clone()),
                body: Box::new(with_xs),
            }),
            span,
        );

        Ok((result, self.dyn_unknown()))
    }

    /// Rewrite free variable references in an expression to access them from a closure record
    /// Replaces Identifier(var) with FieldAccess(Identifier("__closure"), var) for each free variable
    fn rewrite_free_variables(&mut self, expr: &Expression, free_vars: &HashSet<String>) -> Expression {
        let span = expr.h.span;
        match &expr.kind {
            ExprKind::Identifier(name) if free_vars.contains(name) => {
                // Rewrite free variable to closure field access: __closure.name
                let closure_id =
                    self.node_counter.mk_node(ExprKind::Identifier("__closure".to_string()), span);
                self.node_counter.mk_node(ExprKind::FieldAccess(Box::new(closure_id), name.clone()), span)
            }
            ExprKind::Identifier(_) => expr.clone(),
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::TypeHole => expr.clone(),
            ExprKind::RecordLiteral(fields) => {
                let rewritten_fields: Vec<_> = fields
                    .iter()
                    .map(|(name, field_expr)| {
                        (name.clone(), self.rewrite_free_variables(field_expr, free_vars))
                    })
                    .collect();
                self.node_counter.mk_node(ExprKind::RecordLiteral(rewritten_fields), span)
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_rewritten = self.rewrite_free_variables(left, free_vars);
                let right_rewritten = self.rewrite_free_variables(right, free_vars);
                self.node_counter.mk_node(
                    ExprKind::BinaryOp(op.clone(), Box::new(left_rewritten), Box::new(right_rewritten)),
                    span,
                )
            }
            ExprKind::UnaryOp(op, operand) => {
                let operand_rewritten = self.rewrite_free_variables(operand, free_vars);
                self.node_counter.mk_node(ExprKind::UnaryOp(op.clone(), Box::new(operand_rewritten)), span)
            }
            ExprKind::FunctionCall(name, args) => {
                let args_rewritten: Vec<_> =
                    args.iter().map(|arg| self.rewrite_free_variables(arg, free_vars)).collect();
                self.node_counter.mk_node(ExprKind::FunctionCall(name.clone(), args_rewritten), span)
            }
            ExprKind::Application(func, args) => {
                let func_rewritten = self.rewrite_free_variables(func, free_vars);
                let args_rewritten: Vec<_> =
                    args.iter().map(|arg| self.rewrite_free_variables(arg, free_vars)).collect();
                self.node_counter.mk_node(
                    ExprKind::Application(Box::new(func_rewritten), args_rewritten),
                    span,
                )
            }
            ExprKind::LetIn(let_in) => {
                let value_rewritten = self.rewrite_free_variables(&let_in.value, free_vars);

                // Remove bound names from free_vars for the body
                let mut body_free_vars = free_vars.clone();
                for name in let_in.pattern.collect_names() {
                    body_free_vars.remove(&name);
                }
                let body_rewritten = self.rewrite_free_variables(&let_in.body, &body_free_vars);

                self.node_counter.mk_node(
                    ExprKind::LetIn(LetInExpr {
                        pattern: let_in.pattern.clone(),
                        ty: let_in.ty.clone(),
                        value: Box::new(value_rewritten),
                        body: Box::new(body_rewritten),
                    }),
                    span,
                )
            }
            ExprKind::If(if_expr) => {
                let condition = self.rewrite_free_variables(&if_expr.condition, free_vars);
                let then_branch = self.rewrite_free_variables(&if_expr.then_branch, free_vars);
                let else_branch = self.rewrite_free_variables(&if_expr.else_branch, free_vars);
                self.node_counter.mk_node(
                    ExprKind::If(IfExpr {
                        condition: Box::new(condition),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    }),
                    span,
                )
            }
            ExprKind::Tuple(elements) => {
                let elements_rewritten: Vec<_> =
                    elements.iter().map(|elem| self.rewrite_free_variables(elem, free_vars)).collect();
                self.node_counter.mk_node(ExprKind::Tuple(elements_rewritten), span)
            }
            ExprKind::ArrayLiteral(elements) => {
                let elements_rewritten: Vec<_> =
                    elements.iter().map(|elem| self.rewrite_free_variables(elem, free_vars)).collect();
                self.node_counter.mk_node(ExprKind::ArrayLiteral(elements_rewritten), span)
            }
            ExprKind::ArrayIndex(array, index) => {
                let array_rewritten = self.rewrite_free_variables(array, free_vars);
                let index_rewritten = self.rewrite_free_variables(index, free_vars);
                self.node_counter.mk_node(
                    ExprKind::ArrayIndex(Box::new(array_rewritten), Box::new(index_rewritten)),
                    span,
                )
            }
            ExprKind::FieldAccess(expr_inner, field) => {
                let expr_rewritten = self.rewrite_free_variables(expr_inner, free_vars);
                self.node_counter.mk_node(
                    ExprKind::FieldAccess(Box::new(expr_rewritten), field.clone()),
                    span,
                )
            }
            ExprKind::Loop(loop_expr) => {
                let init_rewritten = loop_expr
                    .init
                    .as_ref()
                    .map(|init| Box::new(self.rewrite_free_variables(init, free_vars)));

                // Remove loop pattern bound names from free_vars for the body
                let mut body_free_vars = free_vars.clone();
                for name in loop_expr.pattern.collect_names() {
                    body_free_vars.remove(&name);
                }

                let form_rewritten = match &loop_expr.form {
                    LoopForm::While(cond) => {
                        LoopForm::While(Box::new(self.rewrite_free_variables(cond, &body_free_vars)))
                    }
                    LoopForm::For(iter_name, iter_expr) => {
                        let iter_expr_rewritten = self.rewrite_free_variables(iter_expr, free_vars);
                        // Also remove iterator name from body free vars
                        body_free_vars.remove(iter_name);
                        LoopForm::For(iter_name.clone(), Box::new(iter_expr_rewritten))
                    }
                    LoopForm::ForIn(iter_pat, iter_expr) => {
                        let iter_expr_rewritten = self.rewrite_free_variables(iter_expr, free_vars);
                        // Also remove iterator pattern names
                        for name in iter_pat.collect_names() {
                            body_free_vars.remove(&name);
                        }
                        LoopForm::ForIn(iter_pat.clone(), Box::new(iter_expr_rewritten))
                    }
                };

                let body_rewritten = self.rewrite_free_variables(&loop_expr.body, &body_free_vars);

                self.node_counter.mk_node(
                    ExprKind::Loop(LoopExpr {
                        init: init_rewritten,
                        pattern: loop_expr.pattern.clone(),
                        form: form_rewritten,
                        body: Box::new(body_rewritten),
                    }),
                    span,
                )
            }
            ExprKind::InternalLoop(_) => {
                // InternalLoop should not appear as input to defunctionalization
                panic!("InternalLoop should not appear as input to defunctionalization");
            }
            ExprKind::TypeAscription(inner, ty) => {
                let inner_rewritten = self.rewrite_free_variables(inner, free_vars);
                self.node_counter.mk_node(
                    ExprKind::TypeAscription(Box::new(inner_rewritten), ty.clone()),
                    span,
                )
            }
            ExprKind::TypeCoercion(inner, ty) => {
                let inner_rewritten = self.rewrite_free_variables(inner, free_vars);
                self.node_counter.mk_node(
                    ExprKind::TypeCoercion(Box::new(inner_rewritten), ty.clone()),
                    span,
                )
            }
            // For lambda, we don't rewrite - it will be defunctionalized separately
            ExprKind::Lambda(_) => expr.clone(),
            // Other expression kinds - just clone for now
            _ => expr.clone(),
        }
    }

    /// Bind pattern names to appropriate StaticValues based on pattern structure
    fn bind_pattern(
        &mut self,
        pattern: &Pattern,
        value_sv: &StaticValue,
        scope_stack: &mut ScopeStack<StaticValue>,
    ) -> Result<()> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                // Simple name binding - bind to the whole value
                scope_stack.insert(name.clone(), value_sv.clone());
                Ok(())
            }
            PatternKind::Tuple(elements) => {
                // Tuple pattern - need to decompose the value
                // For now, if we don't have static tuple info, bind all to Dyn
                for (_i, elem_pattern) in elements.iter().enumerate() {
                    // Try to extract element type if available
                    // For now, just bind to a fresh type variable
                    let elem_sv = self.dyn_unknown();
                    self.bind_pattern(elem_pattern, &elem_sv, scope_stack)?;
                }
                Ok(())
            }
            PatternKind::Wildcard => {
                // Wildcard - no binding needed
                Ok(())
            }
            PatternKind::Typed(inner, _ty) => {
                // Typed pattern - recurse on inner pattern
                self.bind_pattern(inner, value_sv, scope_stack)
            }
            PatternKind::Attributed(_attrs, inner) => {
                // Attributed pattern - recurse on inner pattern
                self.bind_pattern(inner, value_sv, scope_stack)
            }
            _ => {
                // Other patterns not yet handled - bind nothing for now
                Ok(())
            }
        }
    }

    /// Desugar a pattern into a simple name pattern plus extraction bindings.
    /// Returns (simple_pattern, bindings) where bindings is a list of (name, extraction_expr).
    /// The extraction expressions reference the simple pattern's variable name.
    /// Preserves type annotations on the simple pattern.
    fn desugar_pattern(&mut self, pattern: &Pattern) -> Result<(Pattern, Vec<(String, Expression)>)> {
        match &pattern.kind {
            PatternKind::Name(_) => {
                // Already simple - no desugaring needed
                Ok((pattern.clone(), vec![]))
            }
            PatternKind::Typed(inner, ty) => {
                // Recursively desugar the inner pattern, then re-wrap with type annotation
                let (simple_inner, bindings) = self.desugar_pattern(inner)?;

                // Wrap the simple pattern with the type annotation
                let typed_pattern = self.node_counter.mk_node(
                    PatternKind::Typed(Box::new(simple_inner), ty.clone()),
                    pattern.h.span,
                );

                Ok((typed_pattern, bindings))
            }
            PatternKind::Tuple(patterns) => {
                // Generate a fresh variable name for the tuple
                let tmp_var = format!("__tmp_{}", self.next_syntax_id());

                let simple_pattern =
                    self.node_counter.mk_node(PatternKind::Name(tmp_var.clone()), pattern.h.span);

                // Create bindings for each tuple element
                let mut bindings = Vec::new();
                for (idx, elem_pat) in patterns.iter().enumerate() {
                    // Extract name from pattern (only support simple names in tuple elements for now)
                    let name = match &elem_pat.kind {
                        PatternKind::Name(n) => n.clone(),
                        PatternKind::Typed(inner, _) => match &inner.kind {
                            PatternKind::Name(n) => n.clone(),
                            _ => {
                                return Err(CompilerError::DefunctionalizationError(format!(
                                    "Complex patterns in tuple elements not supported: {:?}",
                                    inner.kind
                                )));
                            }
                        },
                        _ => {
                            return Err(CompilerError::DefunctionalizationError(format!(
                                "Complex patterns in tuple elements not supported: {:?}",
                                elem_pat.kind
                            )));
                        }
                    };

                    // Create extraction expression: tmp_var.N (field access with numeric index)
                    let tmp_var_expr =
                        self.node_counter.mk_node(ExprKind::Identifier(tmp_var.clone()), Span::dummy());
                    let extraction = self.node_counter.mk_node(
                        ExprKind::FieldAccess(Box::new(tmp_var_expr), idx.to_string()),
                        Span::dummy(),
                    );

                    bindings.push((name, extraction));
                }

                Ok((simple_pattern, bindings))
            }
            PatternKind::Record(fields) => {
                // Generate a fresh variable name for the record
                let tmp_var = format!("__tmp_{}", self.next_syntax_id());

                let simple_pattern =
                    self.node_counter.mk_node(PatternKind::Name(tmp_var.clone()), pattern.h.span);

                // Create bindings for each record field
                let mut bindings = Vec::new();
                for field in fields {
                    let field_name = field.field.clone();

                    // For full patterns, extract the binding name
                    let binding_name = if let Some(pat) = &field.pattern {
                        match &pat.kind {
                            PatternKind::Name(n) => n.clone(),
                            _ => {
                                return Err(CompilerError::DefunctionalizationError(format!(
                                    "Complex patterns in record fields not supported: {:?}",
                                    pat.kind
                                )));
                            }
                        }
                    } else {
                        // Shorthand: field name is the binding name
                        field_name.clone()
                    };

                    // Create extraction expression: tmp_var.field_name
                    let tmp_var_expr =
                        self.node_counter.mk_node(ExprKind::Identifier(tmp_var.clone()), Span::dummy());
                    let extraction = self.node_counter.mk_node(
                        ExprKind::FieldAccess(Box::new(tmp_var_expr), field_name),
                        Span::dummy(),
                    );

                    bindings.push((binding_name, extraction));
                }

                Ok((simple_pattern, bindings))
            }
            _ => Err(CompilerError::DefunctionalizationError(format!(
                "Unsupported pattern kind for desugaring: {:?}",
                pattern.kind
            ))),
        }
    }

    /// Convert a LoopExpr to InternalLoop by desugaring patterns
    fn desugar_loop_to_internal(
        &mut self,
        loop_expr: &LoopExpr,
        scope_stack: &mut ScopeStack<StaticValue>,
        span: Span,
    ) -> Result<(Expression, StaticValue)> {
        use crate::ast::{InternalLoop, LoopForm};

        match &loop_expr.form {
            LoopForm::While(condition_expr) => {
                // Desugar while-style loop
                // loop (idx, acc) = (0, base) while idx < 18 do body

                // 1. Desugar the init pattern to get simple names and extraction bindings
                let (simple_pattern, init_bindings) = self.desugar_pattern(&loop_expr.pattern)?;

                // 2. Transform and evaluate the init expression
                let init = loop_expr.init.as_ref().ok_or_else(|| {
                    CompilerError::DefunctionalizationError(
                        "While loop must have init expression".to_string(),
                    )
                })?;
                let (transformed_init, _) = self.defunctionalize_expression(init, scope_stack)?;

                // 3. Create tuple binding and init_vars for extractions
                let tuple_name = format!("__loop_init_{}", self.next_syntax_id());
                let tuple_expr = transformed_init;

                // Create init_vars that extract components from the tuple
                let mut init_vars = Vec::new();
                for (idx, (_name, _extraction_expr)) in init_bindings.iter().enumerate() {
                    let init_var_name = format!("__init_{}", self.next_syntax_id());

                    // Create expression: tuple_name.idx (field access)
                    let tuple_ident =
                        self.node_counter.mk_node(ExprKind::Identifier(tuple_name.clone()), Span::dummy());
                    let extraction = self.node_counter.mk_node(
                        ExprKind::FieldAccess(Box::new(tuple_ident), idx.to_string()),
                        Span::dummy(),
                    );

                    init_vars.push((init_var_name, Box::new(extraction)));
                }

                // 4. Extract loop_vars from the init_bindings (the actual variable names)
                let loop_vars: Vec<(String, Option<Type>)> = init_bindings
                    .iter()
                    .map(|(name, _)| (name.clone(), None)) // TODO: extract types if available
                    .collect();

                // 5. Transform condition, substituting pattern variable references
                let (transformed_condition, _) =
                    self.defunctionalize_expression(condition_expr, scope_stack)?;

                // 6. Transform body with pattern bindings
                // Note: The body should just use the loop variables directly (idx, acc)
                // which will be bound to the phi registers in MIR. No let-in wrapping needed!
                let (transformed_body, _) =
                    self.defunctionalize_expression(&loop_expr.body, scope_stack)?;
                let final_body = transformed_body;

                // 7. Create body_destructuring expressions
                // These extract values from __body_result (which MIR will bind to the body result register)
                let body_destructuring: Vec<Expression> = init_bindings
                    .iter()
                    .enumerate()
                    .map(|(idx, (_var_name, _))| {
                        // Create expression: __body_result.idx
                        let body_result_ident = self
                            .node_counter
                            .mk_node(ExprKind::Identifier("__body_result".to_string()), Span::dummy());
                        self.node_counter.mk_node(
                            ExprKind::FieldAccess(Box::new(body_result_ident), idx.to_string()),
                            Span::dummy(),
                        )
                    })
                    .collect();

                // 8. Combine into phi_vars using itertools
                use crate::ast::PhiVar;
                use itertools::izip;

                let phi_vars: Vec<PhiVar> = izip!(
                    init_vars.into_iter(),
                    loop_vars.into_iter(),
                    body_destructuring.into_iter()
                )
                .map(
                    |((init_name, init_expr), (loop_var_name, loop_var_type), next_expr)| PhiVar {
                        init_name,
                        init_expr,
                        loop_var_name,
                        loop_var_type,
                        next_expr,
                    },
                )
                .collect();

                let internal_loop = InternalLoop {
                    phi_vars,
                    condition: Some(Box::new(transformed_condition)),
                    body: Box::new(final_body),
                };

                let loop_node = self.node_counter.mk_node(ExprKind::InternalLoop(internal_loop), span);

                // Wrap in LetIn to bind the tuple before the loop
                let tuple_pattern =
                    self.node_counter.mk_node(crate::ast::PatternKind::Name(tuple_name), Span::dummy());
                let result = self.node_counter.mk_node(
                    ExprKind::LetIn(crate::ast::LetInExpr {
                        pattern: tuple_pattern,
                        ty: None,
                        value: Box::new(tuple_expr),
                        body: Box::new(loop_node),
                    }),
                    span,
                );

                Ok((result, self.dyn_unknown()))
            }
            _ => {
                // TODO: Handle ForIn and For forms
                Err(CompilerError::DefunctionalizationError(format!(
                    "Loop form not yet supported: {:?}",
                    loop_expr.form
                )))
            }
        }
    }

    /// Extract loop variable names and types from a simple pattern
    fn extract_loop_vars_from_pattern(&self, pattern: &Pattern) -> Result<Vec<(String, Option<Type>)>> {
        match &pattern.kind {
            PatternKind::Name(name) => Ok(vec![(name.clone(), None)]),
            PatternKind::Typed(inner, ty) => {
                let mut vars = self.extract_loop_vars_from_pattern(inner)?;
                // Apply type to the variable
                if vars.len() == 1 {
                    vars[0].1 = Some(ty.clone());
                }
                Ok(vars)
            }
            PatternKind::Tuple(patterns) => {
                let mut all_vars = Vec::new();
                for pat in patterns {
                    let vars = self.extract_loop_vars_from_pattern(pat)?;
                    all_vars.extend(vars);
                }
                Ok(all_vars)
            }
            _ => Err(CompilerError::DefunctionalizationError(format!(
                "Pattern should be simple after desugaring: {:?}",
                pattern.kind
            ))),
        }
    }

    /// Substitute an identifier in an expression (helper for rewriting extraction expressions)
    fn substitute_identifier_in_expr(
        &self,
        expr: &Expression,
        _old_name: &Expression,
        new_name: &str,
    ) -> Expression {
        // For now, just return a clone - this needs proper implementation
        // TODO: Walk the expression tree and replace identifiers
        expr.clone()
    }
}
