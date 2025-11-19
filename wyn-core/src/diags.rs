//! Diagnostic utilities for AST formatting and display.
//!
//! Provides a less verbose formatter for AST nodes that outputs
//! something close to Wyn syntax.

use crate::ast::*;
use std::fmt::Write;

/// Formatter for AST nodes that produces readable output with line numbers.
pub struct AstFormatter {
    output: String,
    indent: usize,
    show_node_ids: bool,
}

impl AstFormatter {
    pub fn new() -> Self {
        AstFormatter {
            output: String::new(),
            indent: 0,
            show_node_ids: false,
        }
    }

    pub fn with_node_ids() -> Self {
        AstFormatter {
            output: String::new(),
            indent: 0,
            show_node_ids: true,
        }
    }

    /// Format an expression and return the formatted string.
    pub fn format_expression(expr: &Expression) -> String {
        let mut formatter = AstFormatter::new();
        formatter.write_expression(expr);
        formatter.output
    }

    /// Format a program and return the formatted string.
    pub fn format_program(program: &Program) -> String {
        let mut formatter = AstFormatter::new();
        for decl in &program.declarations {
            formatter.write_declaration(decl);
            formatter.newline();
        }
        formatter.output
    }

    /// Format a program with node IDs and return the formatted string.
    pub fn format_program_with_ids(program: &Program) -> String {
        let mut formatter = AstFormatter::with_node_ids();
        for decl in &program.declarations {
            formatter.write_declaration(decl);
            formatter.newline();
        }
        formatter.output
    }

    fn write_line(&mut self, content: &str) {
        let indent = "  ".repeat(self.indent);
        let _ = writeln!(self.output, "{}{}", indent, content);
    }

    fn write_line_no_newline(&mut self, content: &str) {
        let indent = "  ".repeat(self.indent);
        let _ = write!(self.output, "{}{}", indent, content);
    }

    fn newline(&mut self) {
        let _ = writeln!(self.output);
    }

    fn write_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Decl(d) => self.write_decl(d),
            Declaration::Entry(e) => self.write_entry(e),
            Declaration::Uniform(u) => {
                self.write_line(&format!("uniform {}: {}", u.name, u.ty));
            }
            Declaration::Val(v) => {
                self.write_line(&format!("val {}: {}", v.name, v.ty));
            }
            Declaration::TypeBind(tb) => {
                self.write_line(&format!("type {} = {}", tb.name, tb.definition));
            }
            Declaration::ModuleBind(mb) => {
                self.write_line(&format!("module {} = ...", mb.name));
            }
            Declaration::ModuleTypeBind(mtb) => {
                self.write_line(&format!("module type {} = ...", mtb.name));
            }
            Declaration::Open(_) => {
                self.write_line("open ...");
            }
            Declaration::Import(path) => {
                self.write_line(&format!("import \"{}\"", path));
            }
            Declaration::Local(inner) => {
                self.write_line("local");
                self.indent += 1;
                self.write_declaration(inner);
                self.indent -= 1;
            }
        }
    }

    fn write_decl(&mut self, decl: &Decl) {
        let mut header = format!("{} {}", decl.keyword, decl.name);

        // Size params
        for s in &decl.size_params {
            header.push_str(&format!(" [{}]", s));
        }

        // Type params
        for t in &decl.type_params {
            header.push_str(&format!(" '{}", t));
        }

        // Params
        for param in &decl.params {
            header.push_str(&format!(" {}", self.format_param(param)));
        }

        // Return type
        if let Some(ty) = &decl.ty {
            header.push_str(&format!(": {}", ty));
        }

        header.push_str(" =");
        self.write_line(&header);

        self.indent += 1;
        self.write_expression(&decl.body);
        self.indent -= 1;
    }

    /// Format a function parameter - bare name or (pattern: type)
    fn format_param(&self, pattern: &Pattern) -> String {
        match &pattern.kind {
            PatternKind::Name(name) => name.clone(),
            PatternKind::Typed(inner, ty) => {
                format!("({}: {})", self.format_pattern(inner), ty)
            }
            _ => format!("({})", self.format_pattern(pattern)),
        }
    }

    fn write_entry(&mut self, entry: &EntryDecl) {
        let entry_kind = if entry.entry_type.is_vertex() { "vertex" } else { "fragment" };
        let mut header = format!("{} {}", entry_kind, entry.name);

        for param in &entry.params {
            header.push_str(&format!(" {}", self.format_param(param)));
        }

        header.push_str(" =");
        self.write_line(&header);

        self.indent += 1;
        self.write_expression(&entry.body);
        self.indent -= 1;
    }

    fn write_expression(&mut self, expr: &Expression) {
        if self.show_node_ids {
            let _ = write!(self.output, "/* #{} */ ", expr.h.id.0);
        }
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                self.write_line(&n.to_string());
            }
            ExprKind::FloatLiteral(f) => {
                self.write_line(&format!("{}", f));
            }
            ExprKind::BoolLiteral(b) => {
                self.write_line(&b.to_string());
            }
            ExprKind::Identifier(name) => {
                self.write_line(name);
            }
            ExprKind::QualifiedName(quals, name) => {
                let qn =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                self.write_line(&qn);
            }
            ExprKind::ArrayLiteral(elems) => {
                if elems.is_empty() {
                    self.write_line("[]");
                } else if elems.len() <= 4 && elems.iter().all(|e| self.is_simple_expr(e)) {
                    let items: Vec<String> = elems.iter().map(|e| self.format_simple_expr(e)).collect();
                    self.write_line(&format!("[{}]", items.join(", ")));
                } else {
                    self.write_line("[");
                    self.indent += 1;
                    for elem in elems {
                        self.write_expression(elem);
                    }
                    self.indent -= 1;
                    self.write_line("]");
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                let arr_str = self.format_simple_expr(arr);
                let idx_str = self.format_simple_expr(idx);
                self.write_line(&format!("{}[{}]", arr_str, idx_str));
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let lhs_str = self.format_simple_expr(lhs);
                let rhs_str = self.format_simple_expr(rhs);
                self.write_line(&format!("{} {} {}", lhs_str, op.op, rhs_str));
            }
            ExprKind::UnaryOp(op, operand) => {
                let operand_str = self.format_simple_expr(operand);
                self.write_line(&format!("{}{}", op.op, operand_str));
            }
            ExprKind::FunctionCall(name, args) => {
                // Check if any args are complex (not simple expressions)
                let has_complex_args = args.iter().any(|a| !self.is_simple_expr(a));
                if has_complex_args && args.len() == 1 {
                    // Single complex argument - show it expanded
                    self.write_line(&format!("{}(", name));
                    self.indent += 1;
                    self.write_expression(&args[0]);
                    self.indent -= 1;
                    self.write_line(")");
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.format_simple_expr(a)).collect();
                    self.write_line(&format!("{}({})", name, args_str.join(", ")));
                }
            }
            ExprKind::Tuple(elems) => {
                let items: Vec<String> = elems.iter().map(|e| self.format_simple_expr(e)).collect();
                self.write_line(&format!("({})", items.join(", ")));
            }
            ExprKind::RecordLiteral(fields) => {
                let items: Vec<String> = fields
                    .iter()
                    .map(|(name, val)| format!("{}: {}", name, self.format_simple_expr(val)))
                    .collect();
                self.write_line(&format!("{{{}}}", items.join(", ")));
            }
            ExprKind::Lambda(lambda) => {
                let params: Vec<String> = lambda.params.iter().map(|p| self.format_pattern(p)).collect();
                let ret = lambda.return_type.as_ref().map(|t| format!(": {}", t)).unwrap_or_default();
                self.write_line(&format!("\\{}{} ->", params.join(" "), ret));
                self.indent += 1;
                self.write_expression(&lambda.body);
                self.indent -= 1;
            }
            ExprKind::Application(func, args) => {
                let func_str = self.format_simple_expr(func);
                let args_str: Vec<String> = args.iter().map(|a| self.format_simple_expr(a)).collect();
                self.write_line(&format!("{} {}", func_str, args_str.join(" ")));
            }
            ExprKind::LetIn(let_in) => {
                let pat = self.format_pattern(&let_in.pattern);
                let ty = let_in.ty.as_ref().map(|t| format!(": {}", t)).unwrap_or_default();

                if self.is_simple_expr(&let_in.value) {
                    let val = self.format_simple_expr(&let_in.value);
                    self.write_line(&format!("let {}{} = {} in", pat, ty, val));
                } else {
                    self.write_line(&format!("let {}{} =", pat, ty));
                    self.indent += 1;
                    self.write_expression(&let_in.value);
                    self.indent -= 1;
                    self.write_line("in");
                }
                self.write_expression(&let_in.body);
            }
            ExprKind::FieldAccess(obj, field) => {
                let obj_str = self.format_simple_expr(obj);
                self.write_line(&format!("{}.{}", obj_str, field));
            }
            ExprKind::If(if_expr) => {
                let cond = self.format_simple_expr(&if_expr.condition);
                self.write_line(&format!("if {} then", cond));
                self.indent += 1;
                self.write_expression(&if_expr.then_branch);
                self.indent -= 1;
                self.write_line("else");
                self.indent += 1;
                self.write_expression(&if_expr.else_branch);
                self.indent -= 1;
            }
            ExprKind::Loop(loop_expr) => {
                let pat = self.format_pattern(&loop_expr.pattern);
                let init = loop_expr
                    .init
                    .as_ref()
                    .map(|e| format!(" = {}", self.format_simple_expr(e)))
                    .unwrap_or_default();
                let form = match &loop_expr.form {
                    LoopForm::For(var, bound) => {
                        format!("for {} < {}", var, self.format_simple_expr(bound))
                    }
                    LoopForm::ForIn(pat, iter) => format!(
                        "for {} in {}",
                        self.format_pattern(pat),
                        self.format_simple_expr(iter)
                    ),
                    LoopForm::While(cond) => format!("while {}", self.format_simple_expr(cond)),
                };
                self.write_line(&format!("loop {}{} {} do", pat, init, form));
                self.indent += 1;
                self.write_expression(&loop_expr.body);
                self.indent -= 1;
            }
            ExprKind::InternalLoop(il) => {
                self.write_line("internal_loop");
                self.indent += 1;

                // Phi vars
                for phi in &il.phi_vars {
                    let ty = phi.loop_var_type.as_ref().map(|t| format!(": {}", t)).unwrap_or_default();
                    let init = self.format_simple_expr(&phi.init_expr);
                    let next = self.format_simple_expr(&phi.next_expr);
                    self.write_line(&format!(
                        "loop_phi {}{} = [init: {}] [next: {}]",
                        phi.loop_var_name, ty, init, next
                    ));
                }

                // Condition
                if let Some(cond) = &il.condition {
                    self.write_line("while");
                    self.indent += 1;
                    self.write_expression(cond);
                    self.indent -= 1;
                }

                // Body
                self.write_line("body:");
                self.indent += 1;
                self.write_expression(&il.body);
                self.indent -= 1;

                self.indent -= 1;
            }
            ExprKind::Match(match_expr) => {
                let scrut = self.format_simple_expr(&match_expr.scrutinee);
                self.write_line(&format!("match {}", scrut));
                self.indent += 1;
                for case in &match_expr.cases {
                    let pat = self.format_pattern(&case.pattern);
                    self.write_line(&format!("| {} ->", pat));
                    self.indent += 1;
                    self.write_expression(&case.body);
                    self.indent -= 1;
                }
                self.indent -= 1;
            }
            ExprKind::Range(range) => {
                let start = self.format_simple_expr(&range.start);
                let end = self.format_simple_expr(&range.end);
                let op = match range.kind {
                    RangeKind::Inclusive => "...",
                    RangeKind::Exclusive => "..",
                    RangeKind::ExclusiveLt => "..<",
                    RangeKind::ExclusiveGt => "..>",
                };
                if let Some(step) = &range.step {
                    let step_str = self.format_simple_expr(step);
                    self.write_line(&format!("{}..{}{}{}", start, step_str, op, end));
                } else {
                    self.write_line(&format!("{}{}{}", start, op, end));
                }
            }
            ExprKind::Pipe(lhs, rhs) => {
                self.write_expression(lhs);
                self.write_line_no_newline("|> ");
                self.write_expression(rhs);
            }
            ExprKind::TypeAscription(inner, ty) => {
                let inner_str = self.format_simple_expr(inner);
                self.write_line(&format!("{} : {}", inner_str, ty));
            }
            ExprKind::TypeCoercion(inner, ty) => {
                let inner_str = self.format_simple_expr(inner);
                self.write_line(&format!("{} :> {}", inner_str, ty));
            }
            ExprKind::Unsafe(inner) => {
                self.write_line("unsafe");
                self.indent += 1;
                self.write_expression(inner);
                self.indent -= 1;
            }
            ExprKind::Assert(cond, body) => {
                let cond_str = self.format_simple_expr(cond);
                self.write_line(&format!("assert {}", cond_str));
                self.indent += 1;
                self.write_expression(body);
                self.indent -= 1;
            }
            ExprKind::TypeHole => {
                self.write_line("???");
            }
        }
    }

    fn is_simple_expr(&self, expr: &Expression) -> bool {
        matches!(
            &expr.kind,
            ExprKind::IntLiteral(_)
                | ExprKind::FloatLiteral(_)
                | ExprKind::BoolLiteral(_)
                | ExprKind::Identifier(_)
                | ExprKind::QualifiedName(_, _)
                | ExprKind::TypeHole
        )
    }

    fn format_simple_expr(&self, expr: &Expression) -> String {
        match &expr.kind {
            ExprKind::IntLiteral(n) => n.to_string(),
            ExprKind::FloatLiteral(f) => format!("{}", f),
            ExprKind::BoolLiteral(b) => b.to_string(),
            ExprKind::Identifier(name) => name.clone(),
            ExprKind::QualifiedName(quals, name) => {
                if quals.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", quals.join("."), name)
                }
            }
            ExprKind::TypeHole => "???".to_string(),
            ExprKind::Tuple(elems) => {
                let items: Vec<String> = elems.iter().map(|e| self.format_simple_expr(e)).collect();
                format!("({})", items.join(", "))
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                format!(
                    "({} {} {})",
                    self.format_simple_expr(lhs),
                    op.op,
                    self.format_simple_expr(rhs)
                )
            }
            ExprKind::UnaryOp(op, operand) => {
                format!("({}{})", op.op, self.format_simple_expr(operand))
            }
            ExprKind::FunctionCall(name, args) => {
                let args_str: Vec<String> = args.iter().map(|a| self.format_simple_expr(a)).collect();
                format!("{}({})", name, args_str.join(", "))
            }
            ExprKind::Application(func, args) => {
                let func_str = self.format_simple_expr(func);
                let args_str: Vec<String> = args.iter().map(|a| self.format_simple_expr(a)).collect();
                format!("({} {})", func_str, args_str.join(" "))
            }
            ExprKind::ArrayIndex(arr, idx) => {
                format!(
                    "{}[{}]",
                    self.format_simple_expr(arr),
                    self.format_simple_expr(idx)
                )
            }
            ExprKind::FieldAccess(obj, field) => {
                format!("{}.{}", self.format_simple_expr(obj), field)
            }
            ExprKind::TypeAscription(inner, ty) => {
                format!("({}: {})", self.format_simple_expr(inner), ty)
            }
            _ => "<complex>".to_string(),
        }
    }

    fn format_pattern(&self, pattern: &Pattern) -> String {
        match &pattern.kind {
            PatternKind::Name(name) => name.clone(),
            PatternKind::Wildcard => "_".to_string(),
            PatternKind::Unit => "()".to_string(),
            PatternKind::Literal(lit) => match lit {
                PatternLiteral::Int(n) => n.to_string(),
                PatternLiteral::Float(f) => format!("{}", f),
                PatternLiteral::Char(c) => format!("'{}'", c),
                PatternLiteral::Bool(b) => b.to_string(),
            },
            PatternKind::Tuple(patterns) => {
                let items: Vec<String> = patterns.iter().map(|p| self.format_pattern(p)).collect();
                format!("({})", items.join(", "))
            }
            PatternKind::Record(fields) => {
                let items: Vec<String> = fields
                    .iter()
                    .map(|f| {
                        if let Some(pat) = &f.pattern {
                            format!("{} = {}", f.field, self.format_pattern(pat))
                        } else {
                            f.field.clone()
                        }
                    })
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            PatternKind::Constructor(name, patterns) => {
                if patterns.is_empty() {
                    name.clone()
                } else {
                    let items: Vec<String> = patterns.iter().map(|p| self.format_pattern(p)).collect();
                    format!("{} {}", name, items.join(" "))
                }
            }
            PatternKind::Typed(inner, ty) => {
                format!("{}: {}", self.format_pattern(inner), ty)
            }
            PatternKind::Attributed(attrs, inner) => {
                let attr_str = attrs.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>().join(" ");
                format!("#[{}] {}", attr_str, self.format_pattern(inner))
            }
        }
    }
}

impl Default for AstFormatter {
    fn default() -> Self {
        Self::new()
    }
}
