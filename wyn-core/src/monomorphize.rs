/// Monomorphization pass
///
/// Replaces polymorphic functions with concrete, type-specific versions.
/// This must run after type checking (so we have concrete types) but before
/// MIR generation (which needs concrete types for SPIR-V).
///
/// Example:
///   def identity 'a (x:'a) : 'a = x
///   def test : i32 = identity 42
///
/// After monomorphization:
///   def identity__i32 (x:i32) : i32 = x
///   def test : i32 = identity__i32 42
use crate::ast::*;
use crate::error::Result;
use std::collections::HashMap;

pub struct Monomorphizer {
    /// Maps (function_name, concrete_types_key) -> monomorphized_function_name
    instantiations: HashMap<(String, String), String>,

    /// Polymorphic function declarations
    polymorphic_functions: HashMap<String, Decl>,

    /// New monomorphized function declarations to add
    generated_functions: Vec<Decl>,

    /// Type table from type checker (maps NodeId -> inferred Type)
    type_table: HashMap<NodeId, Type>,
}

impl Monomorphizer {
    pub fn new(type_table: HashMap<NodeId, Type>) -> Self {
        Monomorphizer {
            instantiations: HashMap::new(),
            polymorphic_functions: HashMap::new(),
            generated_functions: Vec::new(),
            type_table,
        }
    }

    pub fn monomorphize_program(mut self, program: &Program) -> Result<Program> {
        // First pass: collect all polymorphic function declarations
        for decl in &program.declarations {
            if let Declaration::Decl(d) = decl {
                if !d.type_params.is_empty() {
                    self.polymorphic_functions.insert(d.name.clone(), d.clone());
                }
            }
        }

        // Second pass: transform the program, collecting instantiations
        let mut new_declarations = Vec::new();

        for decl in &program.declarations {
            match decl {
                Declaration::Decl(d) => {
                    // Skip polymorphic function declarations - they'll be instantiated on demand
                    if !d.type_params.is_empty() {
                        continue;
                    }

                    // Transform non-polymorphic declarations
                    let transformed = self.transform_decl(d)?;
                    new_declarations.push(Declaration::Decl(transformed));
                }
                _ => {
                    // Keep other declaration types as-is
                    new_declarations.push(decl.clone());
                }
            }
        }

        // Third pass: add all generated monomorphized functions AT THE BEGINNING
        // This ensures they're defined before any functions that call them
        let mut final_declarations = Vec::new();
        for generated in self.generated_functions {
            final_declarations.push(Declaration::Decl(generated));
        }
        final_declarations.extend(new_declarations);

        Ok(Program {
            declarations: final_declarations,
        })
    }

    fn transform_decl(&mut self, decl: &Decl) -> Result<Decl> {
        // Transform the body expression
        let new_body = self.transform_expr(&decl.body)?;

        Ok(Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
            params: decl.params.clone(),
            ty: decl.ty.clone(),
            body: new_body,
        })
    }

    fn transform_expr(&mut self, expr: &Expression) -> Result<Expression> {
        let new_kind = match &expr.kind {
            ExprKind::FunctionCall(func_name, args) => {
                // Check if this is a call to a polymorphic function
                let poly_decl_opt = self.polymorphic_functions.get(func_name).cloned();

                if let Some(poly_decl) = poly_decl_opt {
                    // Transform arguments first
                    let new_args: Result<Vec<Expression>> =
                        args.iter().map(|arg| self.transform_expr(arg)).collect();
                    let new_args = new_args?;

                    // Get the concrete types for this call
                    // For now, we'll extract types from the arguments
                    let arg_types: Vec<Type> = new_args
                        .iter()
                        .map(|arg| {
                            self.type_table.get(&arg.h.id).cloned().unwrap_or_else(|| {
                                // Fallback: create a placeholder type
                                Type::Constructed(TypeName::Str("unknown"), vec![])
                            })
                        })
                        .collect();

                    // Generate a specialized version of the function
                    let specialized_name =
                        self.get_or_create_specialization(func_name, &poly_decl, &arg_types)?;

                    ExprKind::FunctionCall(specialized_name, new_args)
                } else {
                    // Not a polymorphic function, but still transform arguments
                    let new_args: Result<Vec<Expression>> =
                        args.iter().map(|arg| self.transform_expr(arg)).collect();
                    ExprKind::FunctionCall(func_name.clone(), new_args?)
                }
            }
            // Handle other expression kinds recursively
            _ => {
                // For now, just clone the expression
                // TODO: recursively transform all sub-expressions
                expr.kind.clone()
            }
        };

        Ok(Expression {
            kind: new_kind,
            h: expr.h.clone(),
        })
    }

    fn get_or_create_specialization(
        &mut self,
        func_name: &str,
        poly_decl: &Decl,
        concrete_types: &[Type],
    ) -> Result<String> {
        // Create a key for this specialization
        let type_key = self.make_type_key(concrete_types);
        let inst_key = (func_name.to_string(), type_key.clone());

        // Check if we've already created this specialization
        if let Some(specialized_name) = self.instantiations.get(&inst_key) {
            return Ok(specialized_name.clone());
        }

        // Create a new specialization
        let specialized_name = format!("{}_{}", func_name, type_key);

        // Build a mapping from type parameters to concrete types
        let mut type_bindings: HashMap<String, Type> = HashMap::new();
        for (i, type_param) in poly_decl.type_params.iter().enumerate() {
            if let Some(concrete_type) = concrete_types.get(i) {
                type_bindings.insert(type_param.clone(), concrete_type.clone());
            }
        }

        // Create the specialized function by substituting types
        let specialized_decl = self.specialize_decl(poly_decl, &specialized_name, &type_bindings)?;

        // Register the specialization
        self.instantiations.insert(inst_key, specialized_name.clone());
        self.generated_functions.push(specialized_decl);

        Ok(specialized_name)
    }

    fn make_type_key(&self, types: &[Type]) -> String {
        types.iter().map(|t| self.type_to_string(t)).collect::<Vec<_>>().join("__")
    }

    fn type_to_string(&self, ty: &Type) -> String {
        match ty {
            Type::Constructed(name, args) => {
                let base = match name {
                    TypeName::Str(s) => s.to_string(),
                    TypeName::Named(s) => s.clone(),
                    TypeName::Array => "array".to_string(),
                    TypeName::Vec => "vec".to_string(),
                    TypeName::Size(n) => n.to_string(),
                    _ => "unknown".to_string(),
                };
                if args.is_empty() { base } else { format!("{}_{}", base, self.make_type_key(args)) }
            }
            Type::Variable(_) => "var".to_string(),
        }
    }

    fn specialize_decl(
        &mut self,
        decl: &Decl,
        new_name: &str,
        type_bindings: &HashMap<String, Type>,
    ) -> Result<Decl> {
        // Substitute type parameters in parameter types
        let new_params: Vec<Pattern> =
            decl.params.iter().map(|p| self.substitute_pattern_type(p, type_bindings)).collect();

        // Substitute type parameters in return type
        let new_ty = decl.ty.as_ref().map(|t| Self::substitute_type(t, type_bindings));

        // Transform the body
        let new_body = self.transform_expr(&decl.body)?;

        Ok(Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: new_name.to_string(),
            size_params: vec![], // No more size parameters in specialized version
            type_params: vec![], // No more type parameters in specialized version
            params: new_params,
            ty: new_ty,
            body: new_body,
        })
    }

    fn substitute_pattern_type(&self, pattern: &Pattern, type_bindings: &HashMap<String, Type>) -> Pattern {
        let new_kind = match &pattern.kind {
            PatternKind::Typed(inner_pattern, ty) => {
                let new_ty = Self::substitute_type(ty, type_bindings);
                let new_inner = self.substitute_pattern_type(inner_pattern, type_bindings);
                PatternKind::Typed(Box::new(new_inner), new_ty)
            }
            _ => pattern.kind.clone(),
        };

        Pattern {
            kind: new_kind,
            h: pattern.h.clone(),
        }
    }

    fn substitute_type(ty: &Type, type_bindings: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Constructed(TypeName::UserVar(name), _) => {
                // Replace UserVar with bound concrete type
                type_bindings.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Constructed(name, args) => {
                let new_args: Vec<Type> =
                    args.iter().map(|arg| Self::substitute_type(arg, type_bindings)).collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }
}
