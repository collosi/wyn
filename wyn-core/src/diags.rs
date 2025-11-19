//! Diagnostic utilities for AST and MIR formatting and display.
//!
//! Provides less verbose formatters for AST and MIR nodes that output
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

// MIR formatter disabled during reorganization
#[cfg(any())]
mod mir_formatter {
    use crate::ast::*;
    use crate::lir::{Function, Instruction, Module, Register};

    /// Formatter for MIR diagnostic output
    pub struct MirFormatter {
        output: String,
        indent: usize,
    }

    impl MirFormatter {
        pub fn new() -> Self {
            MirFormatter {
                output: String::new(),
                indent: 0,
            }
        }

        pub fn format_module(module: &Module) -> String {
            let mut formatter = MirFormatter::new();
            for func in &module.functions {
                formatter.write_function(func);
                formatter.newline();
            }
            formatter.output
        }

        fn write(&mut self, s: &str) {
            self.output.push_str(s);
        }

        fn newline(&mut self) {
            self.output.push('\n');
            for _ in 0..self.indent {
                self.output.push_str("  ");
            }
        }

        fn format_type(ty: &Type) -> String {
            match ty {
                Type::Variable(id) => format!("?{}", id),
                Type::Constructed(name, args) if args.is_empty() => format!("{}", name),
                Type::Constructed(TypeName::Str(s), args) if *s == "fn" && args.len() == 2 => {
                    format!(
                        "{} -> {}",
                        Self::format_type(&args[0]),
                        Self::format_type(&args[1])
                    )
                }
                Type::Constructed(name, args) => {
                    let arg_strs: Vec<String> = args.iter().map(Self::format_type).collect();
                    format!("{}[{}]", name, arg_strs.join(", "))
                }
            }
        }

        fn format_register(reg: &Register) -> String {
            format!("r{}:{}", reg.id, Self::format_type(&reg.ty))
        }

        fn write_function(&mut self, func: &Function) {
            self.write(&format!("fn {} (", func.name));
            for (i, param) in func.params.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.write(&Self::format_register(param));
            }
            self.write(&format!(") -> {}", Self::format_type(&func.return_type)));
            self.write(" {");
            self.indent += 1;

            for block in &func.blocks {
                self.newline();
                self.write(&format!("block {}:", block.id));
                self.indent += 1;
                for inst in &block.instructions {
                    self.newline();
                    self.write_instruction(inst);
                }
                self.indent -= 1;
            }

            self.indent -= 1;
            self.newline();
            self.write("}");
            self.newline();
        }

        fn write_instruction(&mut self, inst: &Instruction) {
            match inst {
                Instruction::ConstInt(dest, val) => {
                    self.write(&format!("{} = const {}", Self::format_register(dest), val));
                }
                Instruction::ConstFloat(dest, val) => {
                    self.write(&format!("{} = const {}f32", Self::format_register(dest), val));
                }
                Instruction::ConstBool(dest, val) => {
                    self.write(&format!("{} = const {}", Self::format_register(dest), val));
                }
                Instruction::Neg(dest, src) => {
                    self.write(&format!(
                        "{} = neg {}",
                        Self::format_register(dest),
                        Self::format_register(src)
                    ));
                }
                Instruction::Not(dest, src) => {
                    self.write(&format!(
                        "{} = not {}",
                        Self::format_register(dest),
                        Self::format_register(src)
                    ));
                }
                Instruction::Add(dest, left, right) => {
                    self.write(&format!(
                        "{} = add {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Sub(dest, left, right) => {
                    self.write(&format!(
                        "{} = sub {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Mul(dest, left, right) => {
                    self.write(&format!(
                        "{} = mul {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Div(dest, left, right) => {
                    self.write(&format!(
                        "{} = div {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Eq(dest, left, right) => {
                    self.write(&format!(
                        "{} = eq {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Ne(dest, left, right) => {
                    self.write(&format!(
                        "{} = ne {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Lt(dest, left, right) => {
                    self.write(&format!(
                        "{} = lt {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Le(dest, left, right) => {
                    self.write(&format!(
                        "{} = le {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Gt(dest, left, right) => {
                    self.write(&format!(
                        "{} = gt {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Ge(dest, left, right) => {
                    self.write(&format!(
                        "{} = ge {}, {}",
                        Self::format_register(dest),
                        Self::format_register(left),
                        Self::format_register(right)
                    ));
                }
                Instruction::Alloca(dest) => {
                    self.write(&format!("{} = alloca", Self::format_register(dest)));
                }
                Instruction::Load(dest, src) => {
                    self.write(&format!(
                        "{} = load {}",
                        Self::format_register(dest),
                        Self::format_register(src)
                    ));
                }
                Instruction::Store(dest, src) => {
                    self.write(&format!(
                        "store {}, {}",
                        Self::format_register(dest),
                        Self::format_register(src)
                    ));
                }
                Instruction::Call(dest, func_id, args) => {
                    let arg_strs: Vec<String> = args.iter().map(Self::format_register).collect();
                    self.write(&format!(
                        "{} = call {}({})",
                        Self::format_register(dest),
                        func_id,
                        arg_strs.join(", ")
                    ));
                }
                Instruction::CallBuiltin(dest, name, args) => {
                    let arg_strs: Vec<String> = args.iter().map(Self::format_register).collect();
                    self.write(&format!(
                        "{} = call @{}({})",
                        Self::format_register(dest),
                        name,
                        arg_strs.join(", ")
                    ));
                }
                Instruction::MakeTuple(dest, elements) => {
                    let elem_strs: Vec<String> = elements.iter().map(Self::format_register).collect();
                    self.write(&format!(
                        "{} = tuple ({})",
                        Self::format_register(dest),
                        elem_strs.join(", ")
                    ));
                }
                Instruction::ExtractElement(dest, tuple, index) => {
                    self.write(&format!(
                        "{} = extract {}.{}",
                        Self::format_register(dest),
                        Self::format_register(tuple),
                        index
                    ));
                }
                Instruction::MakeArray(dest, elements) => {
                    let elem_strs: Vec<String> = elements.iter().map(Self::format_register).collect();
                    self.write(&format!(
                        "{} = array [{}]",
                        Self::format_register(dest),
                        elem_strs.join(", ")
                    ));
                }
                Instruction::ArrayIndex(dest, array, index) => {
                    self.write(&format!(
                        "{} = index {}[{}]",
                        Self::format_register(dest),
                        Self::format_register(array),
                        Self::format_register(index)
                    ));
                }
                Instruction::Branch(target) => {
                    self.write(&format!("br block {}", target));
                }
                Instruction::BranchCond(cond, true_block, false_block, merge_block) => {
                    self.write(&format!(
                        "br {} ? block {} : block {} -> block {}",
                        Self::format_register(cond),
                        true_block,
                        false_block,
                        merge_block
                    ));
                }
                Instruction::BranchLoop(cond, body, exit, merge, cont) => {
                    self.write(&format!(
                        "loop {} body {} exit {} merge {} continue {}",
                        Self::format_register(cond),
                        body,
                        exit,
                        merge,
                        cont
                    ));
                }
                Instruction::Loop(info) => {
                    self.write(&format!(
                        "loop phi={} init={} body_result={} cond={}",
                        Self::format_register(&info.phi_reg),
                        Self::format_register(&info.init_reg),
                        Self::format_register(&info.body_result_reg),
                        Self::format_register(&info.cond_reg)
                    ));
                }
                Instruction::Phi(dest, sources) => {
                    let source_strs: Vec<String> = sources
                        .iter()
                        .map(|(reg, block)| format!("[{}, block {}]", Self::format_register(reg), block))
                        .collect();
                    self.write(&format!(
                        "{} = phi {}",
                        Self::format_register(dest),
                        source_strs.join(", ")
                    ));
                }
                Instruction::Return(reg) => {
                    self.write(&format!("ret {}", Self::format_register(reg)));
                }
                Instruction::ReturnVoid => {
                    self.write("ret void");
                }
            }
        }
    }

    impl Default for MirFormatter {
        fn default() -> Self {
            Self::new()
        }
    }
} // end mod mir_formatter
