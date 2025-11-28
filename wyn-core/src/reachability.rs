//! Reachability analysis for MIR
//!
//! This module performs a simple reachability analysis on MIR to determine
//! which functions are actually called, starting from entry points.
//! This allows the lowerer to skip unused library functions.

use crate::mir::visitor::MirVisitor;
use crate::mir::{Attribute, Def, Expr, ExprKind, Literal, Program};
use std::collections::{HashSet, VecDeque};

/// Find all functions reachable from entry points
pub fn reachable_functions(program: &Program) -> HashSet<String> {
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
