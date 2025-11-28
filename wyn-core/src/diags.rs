//! Diagnostic utilities for AST and MIR formatting and display.
//!
//! Provides less verbose formatters for AST and MIR nodes that output
//! something close to Wyn syntax.

use crate::ast::*;
use polytype::Type as PolyType;
use std::fmt::Write;

/// Pretty-print a polytype Type to a human-readable string.
///
/// Converts `Constructed(Str("f32"), [])` to `"f32"`,
/// `Constructed(Array, [Size(3), Str("f32")])` to `"[3]f32"`, etc.
pub fn format_type(ty: &PolyType<TypeName>) -> String {
    match ty {
        PolyType::Variable(id) => format!("?{}", id),
        PolyType::Constructed(name, args) => format_constructed_type(name, args),
    }
}

fn format_constructed_type(name: &TypeName, args: &[PolyType<TypeName>]) -> String {
    match name {
        TypeName::Str(s) => {
            match *s {
                "->" => {
                    // T1 -> T2
                    if args.len() == 2 {
                        let param = format_type(&args[0]);
                        let ret = format_type(&args[1]);
                        format!("{} -> {}", param, ret)
                    } else if args.is_empty() {
                        "() -> ?".to_string()
                    } else {
                        let params: Vec<_> = args[..args.len() - 1].iter().map(format_type).collect();
                        let ret = format_type(&args[args.len() - 1]);
                        format!("({}) -> {}", params.join(", "), ret)
                    }
                }
                _ => {
                    if args.is_empty() {
                        s.to_string()
                    } else {
                        // Generic type application: T<A, B>
                        let args_str: Vec<_> = args.iter().map(format_type).collect();
                        format!("{}<{}>", s, args_str.join(", "))
                    }
                }
            }
        }
        TypeName::Float(bits) => format!("f{}", bits),
        TypeName::UInt(bits) => format!("u{}", bits),
        TypeName::Int(bits) => format!("i{}", bits),
        TypeName::Size(n) => format!("{}", n),
        TypeName::SizeVar(s) => s.clone(),
        TypeName::Unsized => "?".to_string(),
        TypeName::Array => {
            // [size]elem_type
            if args.len() == 2 {
                let size = format_type(&args[0]);
                let elem = format_type(&args[1]);
                format!("[{}]{}", size, elem)
            } else {
                "[?]?".to_string()
            }
        }
        TypeName::Vec => {
            // vec<size>elem_type
            if args.len() == 2 {
                let size = format_type(&args[0]);
                let elem = format_type(&args[1]);
                format!("vec{}{}", size, elem)
            } else {
                "vec?".to_string()
            }
        }
        TypeName::Mat => {
            // mat<rows x cols>elem
            // Args are typically [rows, cols, elem_type]
            if args.len() == 3 {
                let rows = format_type(&args[0]);
                let cols = format_type(&args[1]);
                let elem = format_type(&args[2]);
                format!("mat{}x{}{}", rows, cols, elem)
            } else {
                "mat?".to_string()
            }
        }
        TypeName::Record(fields) => {
            // {field1: T1, field2: T2}
            // Field names are in RecordFields, field types are in args
            let items: Vec<_> = fields
                .iter()
                .zip(args.iter())
                .map(|(name, ty)| format!("{}: {}", name, format_type(ty)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        TypeName::Tuple(_n) => {
            // (T1, T2, ...)
            // Tuple arity is in n, field types are in args
            let items: Vec<_> = args.iter().map(format_type).collect();
            format!("({})", items.join(", "))
        }
        TypeName::Sum(variants) => {
            // Variant1 T1 | Variant2 T2
            let items: Vec<_> = variants
                .iter()
                .map(|(name, variant_args)| {
                    if variant_args.is_empty() {
                        name.clone()
                    } else {
                        let args_str: Vec<_> = variant_args.iter().map(format_type).collect();
                        format!("{} {}", name, args_str.join(" "))
                    }
                })
                .collect();
            items.join(" | ")
        }
        TypeName::Unique => {
            // *T
            if args.len() == 1 { format!("*{}", format_type(&args[0])) } else { "*?".to_string() }
        }
        TypeName::UserVar(s) => format!("'{}", s),
        TypeName::Named(s) => {
            if args.is_empty() {
                s.clone()
            } else {
                let args_str: Vec<_> = args.iter().map(format_type).collect();
                format!("{}<{}>", s, args_str.join(", "))
            }
        }
        TypeName::Existential(vars, inner) => {
            format!("?[{}]. {}", vars.join(", "), format_type(inner))
        }
        TypeName::NamedParam(name, ty) => {
            format!("({}: {})", name, format_type(ty))
        }
    }
}

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
            ExprKind::OperatorSection(op) => {
                self.write_line(&format!("({})", op));
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
            ExprKind::OperatorSection(op) => format!("({})", op),
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

// MIR Display implementations
use crate::mir;
use std::fmt::{self, Display, Formatter};

impl Display for mir::Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, def) in self.defs.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", def)?;
        }
        Ok(())
    }
}

impl Display for mir::Def {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::Def::Function {
                name,
                params,
                ret_type,
                attributes,
                body,
                ..
            } => {
                // Write attributes
                for attr in attributes {
                    writeln!(f, "{}", attr)?;
                }
                // Write function signature with types
                write!(f, "def {}", name)?;
                for param in params.iter() {
                    write!(f, " ({}: {})", param.name, format_type(&param.ty))?;
                }
                write!(f, ": {}", format_type(ret_type))?;
                writeln!(f, " =")?;
                // Write body with indentation
                write!(f, "  {}", body)
            }
            mir::Def::Constant {
                name,
                ty,
                attributes,
                body,
                ..
            } => {
                // Write attributes
                for attr in attributes {
                    writeln!(f, "{}", attr)?;
                }
                // Write constant with type
                writeln!(f, "def {}: {} =", name, format_type(ty))?;
                write!(f, "  {}", body)
            }
        }
    }
}

impl Display for mir::Param {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let prefix = if self.is_consumed { "*" } else { "" };
        write!(f, "({}{}: {})", prefix, self.name, format_type(&self.ty))
    }
}

impl Display for mir::Attribute {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::Attribute::BuiltIn(builtin) => write!(f, "#[builtin({:?})]", builtin),
            mir::Attribute::Location(loc) => write!(f, "#[location({})]", loc),
            mir::Attribute::Vertex => write!(f, "#[vertex]"),
            mir::Attribute::Fragment => write!(f, "#[fragment]"),
            mir::Attribute::Uniform => write!(f, "#[uniform]"),
        }
    }
}

impl Display for mir::Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl Display for mir::ExprKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::ExprKind::Literal(lit) => write!(f, "{}", lit),
            mir::ExprKind::Var(name) => write!(f, "{}", name),
            mir::ExprKind::BinOp { op, lhs, rhs } => {
                write!(f, "({} {} {})", lhs, op, rhs)
            }
            mir::ExprKind::UnaryOp { op, operand } => {
                write!(f, "({}{})", op, operand)
            }
            mir::ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                write!(f, "if {} then {} else {}", cond, then_branch, else_branch)
            }
            mir::ExprKind::Let { name, value, body } => {
                write!(f, "let {} = {} in {}", name, value, body)
            }
            mir::ExprKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
            } => {
                write!(f, "loop ({}, ", loop_var)?;
                if init_bindings.len() == 1 {
                    let (name, binding) = &init_bindings[0];
                    write!(f, "{}) = ({}, {})", name, init, binding)?;
                } else {
                    for (i, (name, _)) in init_bindings.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", name)?;
                    }
                    write!(f, ") = ({}, ", init)?;
                    for (i, (_, binding)) in init_bindings.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", binding)?;
                    }
                    write!(f, ")")?;
                }
                write!(f, " {} do {}", kind, body)
            }
            mir::ExprKind::Call { func, args } => {
                write!(f, "{}", func)?;
                for arg in args.iter() {
                    write!(f, " {}", arg)?;
                }
                Ok(())
            }
            mir::ExprKind::Intrinsic { name, args } => {
                write!(f, "@{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            mir::ExprKind::Attributed { attributes, expr } => {
                for attr in attributes {
                    write!(f, "{} ", attr)?;
                }
                write!(f, "{}", expr)
            }
        }
    }
}

impl Display for mir::Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::Literal::Int(s) => write!(f, "{}", s),
            mir::Literal::Float(s) => write!(f, "{}", s),
            mir::Literal::Bool(b) => write!(f, "{}", b),
            mir::Literal::String(s) => write!(f, "\"{}\"", s.escape_default()),
            mir::Literal::Tuple(exprs) => {
                write!(f, "(")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
            mir::Literal::Array(exprs) => {
                write!(f, "[")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl Display for mir::LoopKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::LoopKind::For { var, iter } => {
                write!(f, "for {} in {}", var, iter)
            }
            mir::LoopKind::ForRange { var, bound } => {
                write!(f, "for {} < {}", var, bound)
            }
            mir::LoopKind::While { cond } => {
                write!(f, "while {}", cond)
            }
        }
    }
}
