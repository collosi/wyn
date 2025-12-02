//! Reachability analysis for MIR
//!
//! This module performs reachability analysis on MIR to determine
//! which functions are actually called, starting from entry points.
//! It returns functions in topological order (callees before callers)
//! so the lowerer can process them without forward references.

use crate::mir::visitor::MirVisitor;
use crate::mir::{Attribute, Def, Expr, ExprKind, Literal, Program};
use std::collections::HashSet;

/// Find all functions reachable from entry points, in topological order.
/// Returns a Vec with callees before callers (post-order DFS).
pub fn reachable_functions_ordered(program: &Program) -> Vec<String> {
    // Find entry points (functions with #[vertex] or #[fragment] attributes)
    let mut entry_points = Vec::new();
    for def in &program.defs {
        if let Def::Function { name, attributes, .. } = def {
            for attr in attributes {
                if matches!(attr, Attribute::Vertex | Attribute::Fragment) {
                    entry_points.push(name.clone());
                    break;
                }
            }
        }
    }

    // Build a map of all functions for quick lookup
    let mut functions = std::collections::HashMap::new();
    for def in &program.defs {
        match def {
            Def::Function { name, body, .. } => {
                functions.insert(name.clone(), body);
            }
            Def::Constant { name, body, .. } => {
                // Constants might call functions too
                functions.insert(name.clone(), body);
            }
            Def::Uniform { .. } => {
                // Uniforms have no body
            }
        }
    }

    // Post-order DFS from entry points
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();
    let mut order = Vec::new();

    for entry in entry_points {
        dfs_postorder(&entry, &functions, &mut visited, &mut in_stack, &mut order);
    }

    order
}

fn dfs_postorder(
    name: &str,
    functions: &std::collections::HashMap<String, &Expr>,
    visited: &mut HashSet<String>,
    in_stack: &mut HashSet<String>,
    order: &mut Vec<String>,
) {
    if visited.contains(name) || in_stack.contains(name) {
        return;
    }

    in_stack.insert(name.to_string());

    // Visit all callees first (if they exist in the program)
    if let Some(body) = functions.get(name) {
        let callees = collect_callees(body);
        for callee in callees {
            if functions.contains_key(&callee) {
                dfs_postorder(&callee, functions, visited, in_stack, order);
            }
        }
    }

    in_stack.remove(name);
    visited.insert(name.to_string());
    order.push(name.to_string());
}

/// Find all functions reachable from entry points (unordered set).
pub fn reachable_functions(program: &Program) -> HashSet<String> {
    reachable_functions_ordered(program).into_iter().collect()
}

/// Visitor that collects function names called in expressions
struct CalleeCollector {
    callees: HashSet<String>,
}

impl MirVisitor for CalleeCollector {
    type Error = std::convert::Infallible;

    fn visit_expr_call(&mut self, func: String, args: Vec<Expr>, expr: Expr) -> Result<Expr, Self::Error> {
        // Collect the function name
        self.callees.insert(func.clone());

        // Continue traversal of arguments
        for arg in &args {
            self.visit_expr(arg.clone()).ok();
        }

        Ok(Expr {
            kind: ExprKind::Call { func, args },
            ..expr
        })
    }

    fn visit_expr_var(&mut self, name: String, expr: Expr) -> Result<Expr, Self::Error> {
        // Variable references might refer to top-level constants
        self.callees.insert(name.clone());
        Ok(Expr {
            kind: ExprKind::Var(name),
            ..expr
        })
    }

    fn visit_expr_literal(&mut self, lit: Literal, expr: Expr) -> Result<Expr, Self::Error> {
        // Check for closure records with __lambda_name field
        if let Some(lambda_name) = crate::mir::extract_lambda_name(&expr) {
            self.callees.insert(lambda_name.to_string());
        }

        // Continue traversal into tuple subexpressions (closures are tuples now)
        if let Literal::Tuple(ref elems) = lit {
            for elem in elems {
                self.visit_expr(elem.clone()).ok();
            }
        }

        Ok(Expr {
            kind: ExprKind::Literal(lit),
            ..expr
        })
    }
}

/// Collect all function names called in an expression
fn collect_callees(expr: &Expr) -> HashSet<String> {
    let mut collector = CalleeCollector {
        callees: HashSet::new(),
    };
    collector.visit_expr(expr.clone()).ok();
    collector.callees
}

/// Filter a program to only include reachable definitions, in topological order.
/// Callees come before callers, so the lowerer can process them without forward references.
pub fn filter_reachable(program: Program) -> Program {
    let ordered = reachable_functions_ordered(&program);

    // Build a map from name to def for reordering
    let mut def_map: std::collections::HashMap<String, Def> = program
        .defs
        .into_iter()
        .map(|def| {
            let name = match &def {
                Def::Function { name, .. } => name.clone(),
                Def::Constant { name, .. } => name.clone(),
                Def::Uniform { name, .. } => name.clone(),
            };
            (name, def)
        })
        .collect();

    // Collect defs in topological order
    let mut filtered_defs: Vec<Def> = ordered.into_iter().filter_map(|name| def_map.remove(&name)).collect();

    // Add all remaining uniforms (they're referenced but have no body to traverse)
    // Collect them first to avoid iterator invalidation
    let uniforms: Vec<_> = def_map.iter()
        .filter(|(_, def)| matches!(def, Def::Uniform { .. }))
        .map(|(name, _)| name.clone())
        .collect();

    for name in uniforms {
        if let Some(def) = def_map.remove(&name) {
            // Insert uniforms at the beginning so they're declared before use
            filtered_defs.insert(0, def);
        }
    }

    Program {
        defs: filtered_defs,
        lambda_registry: program.lambda_registry,
    }
}
