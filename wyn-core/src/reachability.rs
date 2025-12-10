//! Reachability analysis for MIR
//!
//! This module performs reachability analysis on MIR to determine
//! which functions are actually called, starting from entry points.
//! It returns functions in topological order (callees before callers)
//! so the lowerer can process them without forward references.

use crate::mir::folder::MirFolder;
use crate::mir::{Def, DefId, Expr, ExprKind, LocalId, Literal, Program};
use std::collections::HashSet;

/// Find all functions reachable from entry points, in topological order.
/// Returns a Vec with callees before callers (post-order DFS).
pub fn reachable_functions_ordered(program: &Program) -> Vec<String> {
    // Find entry points (Def::EntryPoint variants)
    let mut entry_points = Vec::new();
    for def in &program.defs {
        if let Def::EntryPoint { name, .. } = def {
            entry_points.push(name.clone());
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
            Def::EntryPoint { name, body, .. } => {
                functions.insert(name.clone(), body);
            }
            Def::Uniform { .. } => {
                // Uniforms have no body
            }
            Def::Storage { .. } => {
                // Storage buffers have no body
            }
        }
    }

    // Post-order DFS from entry points
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();
    let mut order = Vec::new();

    for entry in entry_points {
        dfs_postorder(&entry, &functions, program, &mut visited, &mut in_stack, &mut order);
    }

    order
}

fn dfs_postorder(
    name: &str,
    functions: &std::collections::HashMap<String, &Expr>,
    program: &Program,
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
        let callees = collect_callees(body, program);
        for callee in callees {
            if functions.contains_key(&callee) {
                dfs_postorder(&callee, functions, program, visited, in_stack, order);
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
struct CalleeCollector<'a> {
    program: &'a Program,
    callees: HashSet<String>,
}

impl<'a> MirFolder for CalleeCollector<'a> {
    type Error = std::convert::Infallible;
    type Ctx = ();

    fn visit_expr_call(
        &mut self,
        func: DefId,
        func_name: Option<String>,
        args: Vec<Expr>,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        // Look up the function name from DefId or use provided func_name
        if let Some(name) = self.program.def_name(func).or(func_name.as_deref()) {
            self.callees.insert(name.to_string());
        }

        // Continue traversal of arguments
        for arg in &args {
            self.visit_expr(arg.clone(), ctx).ok();
        }

        Ok(Expr {
            kind: ExprKind::Call { func, func_name, args },
            ..expr
        })
    }

    fn visit_expr_var(
        &mut self,
        local: LocalId,
        expr: Expr,
        _ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        // LocalId is a local variable, not a top-level reference
        // No need to collect it as a callee
        Ok(Expr {
            kind: ExprKind::Var(local),
            ..expr
        })
    }

    fn visit_expr_literal(
        &mut self,
        lit: Literal,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        // Continue traversal into tuple subexpressions
        if let Literal::Tuple(ref elems) = lit {
            for elem in elems {
                self.visit_expr(elem.clone(), ctx).ok();
            }
        }

        Ok(Expr {
            kind: ExprKind::Literal(lit),
            ..expr
        })
    }

    fn visit_expr_closure(
        &mut self,
        lambda: DefId,
        captures: Vec<Expr>,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        // Look up the lambda function name from DefId
        if let Some(name) = self.program.def_name(lambda) {
            self.callees.insert(name.to_string());
        }

        // Continue traversal into captures
        for cap in &captures {
            self.visit_expr(cap.clone(), ctx).ok();
        }

        Ok(Expr {
            kind: ExprKind::Closure {
                lambda,
                captures,
            },
            ..expr
        })
    }
}

/// Collect all function names called in an expression
fn collect_callees(expr: &Expr, program: &Program) -> HashSet<String> {
    let mut collector = CalleeCollector {
        program,
        callees: HashSet::new(),
    };
    collector.visit_expr(expr.clone(), &mut ()).ok();
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
                Def::Storage { name, .. } => name.clone(),
                Def::EntryPoint { name, .. } => name.clone(),
            };
            (name, def)
        })
        .collect();

    // Collect defs in topological order
    let mut filtered_defs: Vec<Def> =
        ordered.into_iter().filter_map(|name| def_map.remove(&name)).collect();

    // Add all remaining uniforms and storage buffers (they're referenced but have no body to traverse)
    // Collect them first to avoid iterator invalidation
    let uniforms_and_storage: Vec<_> = def_map
        .iter()
        .filter(|(_, def)| matches!(def, Def::Uniform { .. } | Def::Storage { .. }))
        .map(|(name, _)| name.clone())
        .collect();

    for name in uniforms_and_storage {
        if let Some(def) = def_map.remove(&name) {
            // Insert uniforms/storage at the beginning so they're declared before use
            filtered_defs.insert(0, def);
        }
    }

    // TODO: Remap DefIds in local_tables when defs are filtered
    Program {
        defs: filtered_defs,
        lambda_registry: program.lambda_registry,
        local_tables: program.local_tables,
    }
}
