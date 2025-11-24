//! Reachability analysis for MIR
//!
//! This module performs a simple reachability analysis on MIR to determine
//! which functions are actually called, starting from entry points.
//! This allows the lowerer to skip unused library functions.

use crate::mir::{Def, Expr, ExprKind, Program};
use std::collections::{HashSet, VecDeque};

/// Find all functions reachable from entry points
pub fn reachable_functions(program: &Program) -> HashSet<String> {
    // Find entry points (functions with #[vertex] or #[fragment] attributes)
    let mut entry_points = Vec::new();
    for def in &program.defs {
        if let Def::Function { name, attributes, .. } = def {
            for attr in attributes {
                if attr.name == "vertex" || attr.name == "fragment" {
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
        }
    }

    // BFS from entry points
    let mut seen = HashSet::new();
    let mut worklist = VecDeque::new();

    for name in entry_points {
        if functions.contains_key(&name) {
            seen.insert(name.clone());
            worklist.push_back(name);
        }
    }

    while let Some(fname) = worklist.pop_front() {
        if let Some(body) = functions.get(&fname) {
            let callees = collect_callees(body);
            for callee in callees {
                if seen.insert(callee.clone()) && functions.contains_key(&callee) {
                    worklist.push_back(callee);
                }
            }
        }
    }

    seen
}

/// Collect all function names called in an expression
fn collect_callees(expr: &Expr) -> HashSet<String> {
    let mut callees = HashSet::new();
    collect_callees_rec(expr, &mut callees);
    callees
}

fn collect_callees_rec(expr: &Expr, callees: &mut HashSet<String>) {
    match &expr.kind {
        ExprKind::Call { func, args } => {
            callees.insert(func.clone());
            for arg in args {
                collect_callees_rec(arg, callees);
            }
        }
        ExprKind::Var(_) | ExprKind::Literal(_) => {}
        ExprKind::BinOp { lhs, rhs, .. } => {
            collect_callees_rec(lhs, callees);
            collect_callees_rec(rhs, callees);
        }
        ExprKind::UnaryOp { operand, .. } => {
            collect_callees_rec(operand, callees);
        }
        ExprKind::If { cond, then_branch, else_branch } => {
            collect_callees_rec(cond, callees);
            collect_callees_rec(then_branch, callees);
            collect_callees_rec(else_branch, callees);
        }
        ExprKind::Let { value, body, .. } => {
            collect_callees_rec(value, callees);
            collect_callees_rec(body, callees);
        }
        ExprKind::Intrinsic { args, .. } => {
            for arg in args {
                collect_callees_rec(arg, callees);
            }
        }
        ExprKind::Loop { init_bindings, body, .. } => {
            for (_, init) in init_bindings {
                collect_callees_rec(init, callees);
            }
            collect_callees_rec(body, callees);
        }
        ExprKind::Attributed { expr, .. } => {
            collect_callees_rec(expr, callees);
        }
    }
}

/// Filter a program to only include reachable definitions
pub fn filter_reachable(program: Program) -> Program {
    let reachable = reachable_functions(&program);

    let filtered_defs = program
        .defs
        .into_iter()
        .filter(|def| {
            let name = match def {
                Def::Function { name, .. } => name,
                Def::Constant { name, .. } => name,
            };
            reachable.contains(name)
        })
        .collect();

    Program {
        defs: filtered_defs,
        lambda_registry: program.lambda_registry,
    }
}
