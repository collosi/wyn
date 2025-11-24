//! Monomorphization pass for MIR
//!
//! This pass takes polymorphic functions (with size/type variables) and creates
//! specialized monomorphic copies for each concrete instantiation that's actually called.
//!
//! Example:
//!   def sum [n] (arr:[n]f32) : f32 = ...
//!
//! When called with [4]f32, creates:
//!   def sum$n4 (arr:[4]f32) : f32 = ...
//!
//! This happens after type checking and flattening, before lowering.

use crate::ast::{Type, TypeName};
use crate::error::Result;
use crate::mir::{Def, Expr, ExprKind, Param, Program};
use polytype::Type as PolyType;
use std::collections::{HashMap, HashSet, VecDeque};

/// A substitution mapping type variables to concrete types
type Substitution = HashMap<usize, Type>;

/// Monomorphize a MIR program
pub fn monomorphize(program: Program) -> Result<Program> {
    let mono = Monomorphizer::new(program);
    mono.run()
}

struct Monomorphizer {
    /// Original polymorphic functions by name
    poly_functions: HashMap<String, Def>,
    /// Generated monomorphic functions
    mono_functions: Vec<Def>,
    /// Map from (function_name, substitution) to specialized name
    specializations: HashMap<(String, SubstKey), String>,
    /// Worklist of functions to process
    worklist: VecDeque<WorkItem>,
    /// Functions already processed
    processed: HashSet<String>,
    /// Lambda registry from original program
    lambda_registry: Vec<(String, usize)>,
}

struct WorkItem {
    /// Name of the function to process
    name: String,
    /// The function definition
    def: Def,
}

/// A key for looking up specializations
/// We use a sorted Vec instead of HashMap for deterministic ordering
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SubstKey(Vec<(usize, TypeKey)>);

/// A simplified representation of types for use as hash keys
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum TypeKey {
    Var(usize),
    Size(usize),
    Named(String),
    Constructed(String, Vec<TypeKey>),
}

impl SubstKey {
    fn from_subst(subst: &Substitution) -> Self {
        let mut items: Vec<_> = subst.iter().map(|(k, v)| (*k, TypeKey::from_type(v))).collect();
        items.sort();
        SubstKey(items)
    }
}

impl TypeKey {
    fn from_type(ty: &Type) -> Self {
        match ty {
            PolyType::Variable(id) => TypeKey::Var(*id),
            PolyType::Constructed(name, args) => {
                let name_str = match name {
                    TypeName::Str(s) => s.to_string(),
                    TypeName::Size(n) => return TypeKey::Size(*n),
                    TypeName::Array => "array".to_string(),
                    TypeName::Vec => "vec".to_string(),
                    TypeName::Mat => "mat".to_string(),
                    TypeName::Unsized => "unsized".to_string(),
                    TypeName::SizeVar(s) => format!("sizevar_{}", s),
                    TypeName::UserVar(s) => format!("uservar_{}", s),
                    TypeName::Named(s) => s.clone(),
                    TypeName::Record(fields) => format!("record_{}", fields.len()),
                    TypeName::Unique => "unique".to_string(),
                    TypeName::Sum(variants) => format!("sum_{}", variants.len()),
                    TypeName::Existential(vars, _) => format!("exists_{}", vars.len()),
                    TypeName::NamedParam(name, _) => format!("named_{}", name),
                };
                TypeKey::Constructed(name_str, args.iter().map(TypeKey::from_type).collect())
            }
        }
    }
}

impl Monomorphizer {
    fn new(program: Program) -> Self {
        // Build map of polymorphic functions
        let mut poly_functions = HashMap::new();
        let mut entry_points = Vec::new();

        for def in program.defs {
            let name = match &def {
                Def::Function { name, .. } => name.clone(),
                Def::Constant { name, .. } => name.clone(),
            };

            // Check if this is an entry point
            if let Def::Function { attributes, .. } = &def {
                for attr in attributes {
                    if attr.name == "vertex" || attr.name == "fragment" {
                        entry_points.push(WorkItem {
                            name: name.clone(),
                            def: def.clone(),
                        });
                    }
                }
            }

            poly_functions.insert(name, def);
        }

        let mut worklist = VecDeque::new();
        worklist.extend(entry_points);

        Monomorphizer {
            poly_functions,
            mono_functions: Vec::new(),
            specializations: HashMap::new(),
            worklist,
            processed: HashSet::new(),
            lambda_registry: program.lambda_registry,
        }
    }

    fn run(mut self) -> Result<Program> {
        while let Some(work_item) = self.worklist.pop_front() {
            if self.processed.contains(&work_item.name) {
                continue;
            }
            self.processed.insert(work_item.name.clone());

            // Process this function: look for calls and specialize callees
            let def = self.process_function(work_item.def)?;
            self.mono_functions.push(def);
        }

        Ok(Program {
            defs: self.mono_functions,
            lambda_registry: self.lambda_registry,
        })
    }

    fn process_function(&mut self, def: Def) -> Result<Def> {
        match def {
            Def::Function {
                name,
                params,
                ret_type,
                attributes,
                param_attributes,
                return_attributes,
                body,
                span,
            } => {
                let body = self.process_expr(body)?;
                Ok(Def::Function {
                    name,
                    params,
                    ret_type,
                    attributes,
                    param_attributes,
                    return_attributes,
                    body,
                    span,
                })
            }
            Def::Constant {
                name,
                ty,
                attributes,
                body,
                span,
            } => {
                let body = self.process_expr(body)?;
                Ok(Def::Constant {
                    name,
                    ty,
                    attributes,
                    body,
                    span,
                })
            }
        }
    }

    fn process_expr(&mut self, expr: Expr) -> Result<Expr> {
        let kind = match expr.kind {
            ExprKind::Call { func, args } => {
                // Process arguments first
                let args: Result<Vec<_>> = args.into_iter().map(|arg| self.process_expr(arg)).collect();
                let args = args?;

                // Check if this is a call to a polymorphic function
                if let Some(poly_def) = self.poly_functions.get(&func).cloned() {
                    let subst = self.infer_substitution(&poly_def, &args)?;

                    if !subst.is_empty() {
                        // This is a polymorphic call - specialize it
                        let specialized_name =
                            self.get_or_create_specialization(&func, &subst, &poly_def)?;
                        ExprKind::Call {
                            func: specialized_name,
                            args,
                        }
                    } else {
                        // Monomorphic call
                        ExprKind::Call { func, args }
                    }
                } else {
                    ExprKind::Call { func, args }
                }
            }
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: Box::new(self.process_expr(*lhs)?),
                rhs: Box::new(self.process_expr(*rhs)?),
            },
            ExprKind::UnaryOp { op, operand } => ExprKind::UnaryOp {
                op,
                operand: Box::new(self.process_expr(*operand)?),
            },
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => ExprKind::If {
                cond: Box::new(self.process_expr(*cond)?),
                then_branch: Box::new(self.process_expr(*then_branch)?),
                else_branch: Box::new(self.process_expr(*else_branch)?),
            },
            ExprKind::Let { name, value, body } => ExprKind::Let {
                name,
                value: Box::new(self.process_expr(*value)?),
                body: Box::new(self.process_expr(*body)?),
            },
            ExprKind::Loop {
                init_bindings,
                kind,
                body,
            } => {
                let init_bindings: Result<Vec<_>> = init_bindings
                    .into_iter()
                    .map(|(name, expr)| Ok((name, self.process_expr(expr)?)))
                    .collect();
                ExprKind::Loop {
                    init_bindings: init_bindings?,
                    kind,
                    body: Box::new(self.process_expr(*body)?),
                }
            }
            ExprKind::Intrinsic { name, args } => {
                let args: Result<Vec<_>> = args.into_iter().map(|arg| self.process_expr(arg)).collect();
                ExprKind::Intrinsic { name, args: args? }
            }
            ExprKind::Attributed { attributes, expr } => ExprKind::Attributed {
                attributes,
                expr: Box::new(self.process_expr(*expr)?),
            },
            // Leaf nodes - no recursion needed
            ExprKind::Var(_) | ExprKind::Literal(_) => expr.kind,
        };

        Ok(Expr {
            ty: expr.ty,
            kind,
            span: expr.span,
        })
    }

    /// Infer the substitution needed for a polymorphic function call
    fn infer_substitution(&self, poly_def: &Def, args: &[Expr]) -> Result<Substitution> {
        let mut subst = Substitution::new();

        // Get parameter types from the definition
        let param_types = match poly_def {
            Def::Function { params, .. } => params.iter().map(|p| &p.ty).collect::<Vec<_>>(),
            Def::Constant { .. } => return Ok(subst), // No parameters
        };

        // Match argument types against parameter types
        for (param_ty, arg) in param_types.iter().zip(args.iter()) {
            self.unify_for_subst(param_ty, &arg.ty, &mut subst)?;
        }

        Ok(subst)
    }

    /// Unify two types to build a substitution
    fn unify_for_subst(&self, expected: &Type, actual: &Type, subst: &mut Substitution) -> Result<()> {
        match (expected, actual) {
            (PolyType::Variable(id), concrete) => {
                // This is a type variable in the polymorphic function
                // Map it to the concrete type from the call site
                if !contains_variables(concrete) {
                    subst.insert(*id, concrete.clone());
                }
            }
            (PolyType::Constructed(name1, args1), PolyType::Constructed(name2, args2)) => {
                // Recurse into constructed types
                if std::mem::discriminant(name1) == std::mem::discriminant(name2) {
                    for (a1, a2) in args1.iter().zip(args2.iter()) {
                        self.unify_for_subst(a1, a2, subst)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get or create a specialized version of a function
    fn get_or_create_specialization(
        &mut self,
        func_name: &str,
        subst: &Substitution,
        poly_def: &Def,
    ) -> Result<String> {
        let key = (func_name.to_string(), SubstKey::from_subst(subst));

        if let Some(specialized_name) = self.specializations.get(&key) {
            return Ok(specialized_name.clone());
        }

        // Create new specialized name
        let specialized_name = format!("{}${}", func_name, format_subst(subst));

        // Clone and specialize the function
        let specialized_def = self.specialize_def(poly_def.clone(), subst, &specialized_name)?;

        // Add to worklist to process its body
        self.worklist.push_back(WorkItem {
            name: specialized_name.clone(),
            def: specialized_def,
        });

        self.specializations.insert(key, specialized_name.clone());
        Ok(specialized_name)
    }

    /// Create a specialized version of a function by applying substitution
    fn specialize_def(&self, def: Def, subst: &Substitution, new_name: &str) -> Result<Def> {
        match def {
            Def::Function {
                params,
                ret_type,
                attributes,
                param_attributes,
                return_attributes,
                body,
                span,
                ..
            } => {
                let params = params
                    .into_iter()
                    .map(|p| Param {
                        name: p.name,
                        ty: apply_subst(&p.ty, subst),
                        is_consumed: p.is_consumed,
                    })
                    .collect();
                let ret_type = apply_subst(&ret_type, subst);
                let body = apply_subst_expr(body, subst);

                Ok(Def::Function {
                    name: new_name.to_string(),
                    params,
                    ret_type,
                    attributes,
                    param_attributes,
                    return_attributes,
                    body,
                    span,
                })
            }
            Def::Constant {
                ty,
                attributes,
                body,
                span,
                ..
            } => Ok(Def::Constant {
                name: new_name.to_string(),
                ty: apply_subst(&ty, subst),
                attributes,
                body: apply_subst_expr(body, subst),
                span,
            }),
        }
    }
}

/// Check if a type contains type variables
fn contains_variables(ty: &Type) -> bool {
    match ty {
        PolyType::Variable(_) => true,
        PolyType::Constructed(_, args) => args.iter().any(contains_variables),
        _ => false,
    }
}

/// Apply a substitution to a type
fn apply_subst(ty: &Type, subst: &Substitution) -> Type {
    match ty {
        PolyType::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        PolyType::Constructed(name, args) => {
            let new_args = args.iter().map(|arg| apply_subst(arg, subst)).collect();
            PolyType::Constructed(name.clone(), new_args)
        }
        _ => ty.clone(),
    }
}

/// Apply a substitution to an expression
fn apply_subst_expr(expr: Expr, subst: &Substitution) -> Expr {
    let kind = match expr.kind {
        ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
            op,
            lhs: Box::new(apply_subst_expr(*lhs, subst)),
            rhs: Box::new(apply_subst_expr(*rhs, subst)),
        },
        ExprKind::UnaryOp { op, operand } => ExprKind::UnaryOp {
            op,
            operand: Box::new(apply_subst_expr(*operand, subst)),
        },
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => ExprKind::If {
            cond: Box::new(apply_subst_expr(*cond, subst)),
            then_branch: Box::new(apply_subst_expr(*then_branch, subst)),
            else_branch: Box::new(apply_subst_expr(*else_branch, subst)),
        },
        ExprKind::Let { name, value, body } => ExprKind::Let {
            name,
            value: Box::new(apply_subst_expr(*value, subst)),
            body: Box::new(apply_subst_expr(*body, subst)),
        },
        ExprKind::Loop {
            init_bindings,
            kind,
            body,
        } => {
            let init_bindings =
                init_bindings.into_iter().map(|(name, e)| (name, apply_subst_expr(e, subst))).collect();
            ExprKind::Loop {
                init_bindings,
                kind,
                body: Box::new(apply_subst_expr(*body, subst)),
            }
        }
        ExprKind::Call { func, args } => {
            let args = args.into_iter().map(|arg| apply_subst_expr(arg, subst)).collect();
            ExprKind::Call { func, args }
        }
        ExprKind::Intrinsic { name, args } => {
            let args = args.into_iter().map(|arg| apply_subst_expr(arg, subst)).collect();
            ExprKind::Intrinsic { name, args }
        }
        ExprKind::Attributed { attributes, expr } => ExprKind::Attributed {
            attributes,
            expr: Box::new(apply_subst_expr(*expr, subst)),
        },
        other => other,
    };

    Expr {
        ty: apply_subst(&expr.ty, subst),
        kind,
        span: expr.span,
    }
}

/// Format a substitution for use in specialized function names
fn format_subst(subst: &Substitution) -> String {
    let mut items: Vec<_> = subst.iter().collect();
    items.sort_by_key(|(k, _)| *k);

    items.iter().map(|(_, ty)| format_type_compact(ty)).collect::<Vec<_>>().join("_")
}

fn format_type_compact(ty: &Type) -> String {
    match ty {
        PolyType::Variable(id) => format!("v{}", id),
        PolyType::Constructed(TypeName::Size(n), _) => format!("n{}", n),
        PolyType::Constructed(TypeName::Str(s), args) if args.is_empty() => s.to_string(),
        PolyType::Constructed(TypeName::Array, args) if args.len() == 2 => {
            format!(
                "arr{}{}",
                format_type_compact(&args[0]),
                format_type_compact(&args[1])
            )
        }
        _ => "ty".to_string(),
    }
}
