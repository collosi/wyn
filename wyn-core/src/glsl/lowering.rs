//! GLSL Lowering
//!
//! This module converts MIR to GLSL shader source code.
//! It generates separate strings for vertex and fragment shaders.

use crate::ast::TypeName;
use crate::bail_glsl;
use crate::error::Result;
use crate::impl_source::{BuiltinImpl, ImplSource, PrimOp};
use crate::lowering_common::{ShaderStage, is_entry_point};
use crate::mir::{Attribute, Def, Expr, ExprKind, Literal, LoopKind, Program};
use polytype::Type as PolyType;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Output from GLSL lowering - separate shader strings
#[derive(Debug, Clone)]
pub struct GlslOutput {
    /// Vertex shader source (None if no vertex entry point)
    pub vertex: Option<String>,
    /// Fragment shader source (None if no fragment entry point)
    pub fragment: Option<String>,
}

/// Lower a MIR program to GLSL
pub fn lower(program: &Program) -> Result<GlslOutput> {
    let mut ctx = LowerCtx::new(program);
    ctx.lower_program()
}

/// Context for lowering MIR to GLSL
struct LowerCtx<'a> {
    program: &'a Program,
    /// Map from definition name to its index
    def_index: HashMap<String, usize>,
    /// Functions that have been lowered
    lowered: HashSet<String>,
    /// Builtin implementations
    impl_source: ImplSource,
    /// Current indentation level
    indent: usize,
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        let mut def_index = HashMap::new();
        for (i, def) in program.defs.iter().enumerate() {
            let name = match def {
                Def::Function { name, .. } => name,
                Def::Constant { name, .. } => name,
                Def::Uniform { name, .. } => name,
            };
            def_index.insert(name.clone(), i);
        }

        LowerCtx {
            program,
            def_index,
            lowered: HashSet::new(),
            impl_source: ImplSource::default(),
            indent: 0,
        }
    }

    fn lower_program(&mut self) -> Result<GlslOutput> {
        let mut vertex_shader = None;
        let mut fragment_shader = None;

        // Find entry points and generate shaders
        for def in &self.program.defs {
            if let Def::Function { name, attributes, .. } = def {
                for attr in attributes {
                    match attr {
                        Attribute::Vertex => {
                            vertex_shader = Some(self.lower_shader(name, ShaderStage::Vertex)?);
                        }
                        Attribute::Fragment => {
                            fragment_shader = Some(self.lower_shader(name, ShaderStage::Fragment)?);
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(GlslOutput {
            vertex: vertex_shader,
            fragment: fragment_shader,
        })
    }

    fn lower_shader(&mut self, entry_name: &str, stage: ShaderStage) -> Result<String> {
        let mut output = String::new();

        // GLSL version
        writeln!(output, "#version 450").unwrap();
        writeln!(output).unwrap();

        // Collect dependencies and emit them
        self.lowered.clear();
        let deps = self.collect_dependencies(entry_name)?;

        // Emit uniforms
        for def in &self.program.defs {
            if let Def::Uniform {
                name, ty, binding, ..
            } = def
            {
                writeln!(
                    output,
                    "layout(binding = {}) uniform {} {};",
                    binding,
                    self.type_to_glsl(ty),
                    name
                )
                .unwrap();
            }
        }
        writeln!(output).unwrap();

        // Emit helper functions (non-entry points first)
        for dep_name in &deps {
            if dep_name != entry_name {
                if let Some(&idx) = self.def_index.get(dep_name) {
                    self.lower_def(&self.program.defs[idx].clone(), &mut output)?;
                }
            }
        }

        // Emit entry point
        if let Some(&idx) = self.def_index.get(entry_name) {
            self.lower_entry_point(&self.program.defs[idx].clone(), stage, &mut output)?;
        }

        Ok(output)
    }

    fn collect_dependencies(&self, name: &str) -> Result<Vec<String>> {
        let mut deps = Vec::new();
        let mut visited = HashSet::new();
        self.collect_deps_recursive(name, &mut deps, &mut visited)?;
        Ok(deps)
    }

    fn collect_deps_recursive(
        &self,
        name: &str,
        deps: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        if visited.contains(name) {
            return Ok(());
        }
        visited.insert(name.to_string());

        if let Some(&idx) = self.def_index.get(name) {
            let def = &self.program.defs[idx];
            if let Def::Function { body, .. } = def {
                self.collect_expr_deps(body, deps, visited)?;
            }
        }

        deps.push(name.to_string());
        Ok(())
    }

    fn collect_expr_deps(
        &self,
        expr: &Expr,
        deps: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        match &expr.kind {
            ExprKind::Call { func, args } => {
                // If it's a user function (not a builtin), collect it
                if self.def_index.contains_key(func) && self.impl_source.get(func).is_none() {
                    self.collect_deps_recursive(func, deps, visited)?;
                }
                for arg in args {
                    self.collect_expr_deps(arg, deps, visited)?;
                }
            }
            ExprKind::Let { value, body, .. } => {
                self.collect_expr_deps(value, deps, visited)?;
                self.collect_expr_deps(body, deps, visited)?;
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.collect_expr_deps(cond, deps, visited)?;
                self.collect_expr_deps(then_branch, deps, visited)?;
                self.collect_expr_deps(else_branch, deps, visited)?;
            }
            ExprKind::BinOp { lhs, rhs, .. } => {
                self.collect_expr_deps(lhs, deps, visited)?;
                self.collect_expr_deps(rhs, deps, visited)?;
            }
            ExprKind::UnaryOp { operand, .. } => {
                self.collect_expr_deps(operand, deps, visited)?;
            }
            ExprKind::Loop { init, body, .. } => {
                self.collect_expr_deps(init, deps, visited)?;
                self.collect_expr_deps(body, deps, visited)?;
            }
            ExprKind::Intrinsic { args, .. } => {
                for arg in args {
                    self.collect_expr_deps(arg, deps, visited)?;
                }
            }
            ExprKind::Attributed { expr, .. } => {
                self.collect_expr_deps(expr, deps, visited)?;
            }
            ExprKind::Materialize(expr) => {
                self.collect_expr_deps(expr, deps, visited)?;
            }
            ExprKind::Literal(lit) => {
                self.collect_literal_deps(lit, deps, visited)?;
            }
            ExprKind::Var(_) | ExprKind::Unit => {}
        }
        Ok(())
    }

    fn collect_literal_deps(
        &self,
        lit: &Literal,
        deps: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        match lit {
            Literal::Tuple(elems) | Literal::Array(elems) | Literal::Vector(elems) => {
                for elem in elems {
                    self.collect_expr_deps(elem, deps, visited)?;
                }
            }
            Literal::Matrix(rows) => {
                for row in rows {
                    for elem in row {
                        self.collect_expr_deps(elem, deps, visited)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn lower_def(&mut self, def: &Def, output: &mut String) -> Result<()> {
        match def {
            Def::Function {
                name,
                params,
                ret_type,
                body,
                attributes,
                ..
            } => {
                if is_entry_point(attributes) {
                    return Ok(()); // Entry points handled separately
                }

                // Function signature
                write!(output, "{} {}(", self.type_to_glsl(ret_type), name).unwrap();
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }
                    write!(output, "{} {}", self.type_to_glsl(&param.ty), param.name).unwrap();
                }
                writeln!(output, ") {{").unwrap();

                self.indent += 1;
                let result = self.lower_expr(body, output)?;
                writeln!(output, "{}return {};", self.indent_str(), result).unwrap();
                self.indent -= 1;

                writeln!(output, "}}").unwrap();
                writeln!(output).unwrap();
            }
            Def::Constant { name, ty, body, .. } => {
                write!(output, "const {} {} = ", self.type_to_glsl(ty), name).unwrap();
                let val = self.lower_expr(body, output)?;
                writeln!(output, "{};", val).unwrap();
            }
            Def::Uniform { .. } => {
                // Already emitted at top of shader
            }
        }
        Ok(())
    }

    fn lower_entry_point(&mut self, def: &Def, stage: ShaderStage, output: &mut String) -> Result<()> {
        if let Def::Function {
            params,
            ret_type,
            body,
            param_attributes,
            return_attributes,
            ..
        } = def
        {
            // Emit input declarations
            for (i, param) in params.iter().enumerate() {
                let attrs = param_attributes.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
                for attr in attrs {
                    match attr {
                        Attribute::Location(loc) => {
                            writeln!(
                                output,
                                "layout(location = {}) in {} {};",
                                loc,
                                self.type_to_glsl(&param.ty),
                                param.name
                            )
                            .unwrap();
                        }
                        Attribute::BuiltIn(builtin) => {
                            // Built-ins are accessed via gl_* variables
                        }
                        _ => {}
                    }
                }
            }

            // Emit output declarations
            // For now, handle single output
            if !return_attributes.is_empty() {
                for (i, attrs) in return_attributes.iter().enumerate() {
                    for attr in attrs {
                        match attr {
                            Attribute::Location(loc) => {
                                writeln!(
                                    output,
                                    "layout(location = {}) out {} _out{};",
                                    loc,
                                    self.type_to_glsl(ret_type), // TODO: handle tuple returns
                                    i
                                )
                                .unwrap();
                            }
                            _ => {}
                        }
                    }
                }
            }

            writeln!(output).unwrap();
            writeln!(output, "void main() {{").unwrap();
            self.indent += 1;

            let result = self.lower_expr(body, output)?;

            // Assign to outputs
            if !return_attributes.is_empty() {
                writeln!(output, "{}_out0 = {};", self.indent_str(), result).unwrap();
            }

            // Handle gl_Position for vertex shaders
            if stage == ShaderStage::Vertex {
                // Check if we need to write gl_Position
                for attrs in return_attributes {
                    for attr in attrs {
                        if let Attribute::BuiltIn(spirv::BuiltIn::Position) = attr {
                            writeln!(output, "{}gl_Position = {};", self.indent_str(), result).unwrap();
                        }
                    }
                }
            }

            self.indent -= 1;
            writeln!(output, "}}").unwrap();
        }
        Ok(())
    }

    fn lower_expr(&mut self, expr: &Expr, output: &mut String) -> Result<String> {
        match &expr.kind {
            ExprKind::Literal(lit) => self.lower_literal(lit, &expr.ty, output),

            ExprKind::Unit => Ok("".to_string()),

            ExprKind::Var(name) => Ok(name.clone()),

            ExprKind::BinOp { op, lhs, rhs } => {
                let l = self.lower_expr(lhs, output)?;
                let r = self.lower_expr(rhs, output)?;
                Ok(format!("({} {} {})", l, op, r))
            }

            ExprKind::UnaryOp { op, operand } => {
                let inner = self.lower_expr(operand, output)?;
                Ok(format!("({}{})", op, inner))
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let c = self.lower_expr(cond, output)?;
                let t = self.lower_expr(then_branch, output)?;
                let e = self.lower_expr(else_branch, output)?;
                Ok(format!("({} ? {} : {})", c, t, e))
            }

            ExprKind::Let {
                name, value, body, ..
            } => {
                let v = self.lower_expr(value, output)?;
                writeln!(
                    output,
                    "{}{} {} = {};",
                    self.indent_str(),
                    self.type_to_glsl(&value.ty),
                    name,
                    v
                )
                .unwrap();
                self.lower_expr(body, output)
            }

            ExprKind::Call { func, args } => {
                let mut arg_strs = Vec::new();
                for arg in args {
                    arg_strs.push(self.lower_expr(arg, output)?);
                }

                // Check if it's a builtin
                if let Some(impl_) = self.impl_source.get(func) {
                    self.lower_builtin_call(impl_, &arg_strs, &expr.ty)
                } else {
                    Ok(format!("{}({})", func, arg_strs.join(", ")))
                }
            }

            ExprKind::Intrinsic { name, args } => {
                let mut arg_strs = Vec::new();
                for arg in args {
                    arg_strs.push(self.lower_expr(arg, output)?);
                }
                self.lower_intrinsic(name, &arg_strs, args, &expr.ty)
            }

            ExprKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
            } => self.lower_loop(loop_var, init, init_bindings, kind, body, &expr.ty, output),

            ExprKind::Attributed { expr, .. } => self.lower_expr(expr, output),

            ExprKind::Materialize(inner) => self.lower_expr(inner, output),
        }
    }

    fn lower_literal(
        &mut self,
        lit: &Literal,
        ty: &PolyType<TypeName>,
        output: &mut String,
    ) -> Result<String> {
        match lit {
            Literal::Int(s) => Ok(s.clone()),
            Literal::Float(s) => {
                // Ensure float has decimal point
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(s.clone())
                } else {
                    Ok(format!("{}.0", s))
                }
            }
            Literal::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
            Literal::String(s) => Ok(format!("\"{}\"", s)),
            Literal::Vector(elems) => {
                let mut parts = Vec::new();
                for e in elems {
                    parts.push(self.lower_expr(e, output)?);
                }
                Ok(format!("{}({})", self.type_to_glsl(ty), parts.join(", ")))
            }
            Literal::Matrix(rows) => {
                let mut parts = Vec::new();
                // GLSL matrices are column-major, so we need to transpose
                for row in rows {
                    for elem in row {
                        parts.push(self.lower_expr(elem, output)?);
                    }
                }
                Ok(format!("{}({})", self.type_to_glsl(ty), parts.join(", ")))
            }
            Literal::Array(elems) => {
                let mut parts = Vec::new();
                for e in elems {
                    parts.push(self.lower_expr(e, output)?);
                }
                Ok(format!("{}[]({})", self.type_to_glsl(ty), parts.join(", ")))
            }
            Literal::Tuple(elems) => {
                // Tuples don't exist in GLSL - this shouldn't happen for valid shaders
                bail_glsl!("Tuples are not supported in GLSL")
            }
        }
    }

    fn lower_builtin_call(
        &self,
        impl_: &BuiltinImpl,
        args: &[String],
        ret_ty: &PolyType<TypeName>,
    ) -> Result<String> {
        match impl_ {
            BuiltinImpl::PrimOp(op) => self.lower_primop(op, args, ret_ty),
            BuiltinImpl::CoreFn(name) => Ok(format!("{}({})", name, args.join(", "))),
            BuiltinImpl::Intrinsic(intr) => {
                bail_glsl!("Intrinsic {:?} not supported in GLSL", intr)
            }
        }
    }

    fn lower_primop(&self, op: &PrimOp, args: &[String], _ret_ty: &PolyType<TypeName>) -> Result<String> {
        use PrimOp::*;
        match op {
            // GLSL.std.450 extended instructions map to GLSL functions
            GlslExt(id) => {
                let func = glsl_ext_to_name(*id);
                Ok(format!("{}({})", func, args.join(", ")))
            }

            // Math operations
            Dot => Ok(format!("dot({}, {})", args[0], args[1])),
            OuterProduct => Ok(format!("outerProduct({}, {})", args[0], args[1])),
            MatrixTimesMatrix | MatrixTimesVector | VectorTimesMatrix => {
                Ok(format!("({} * {})", args[0], args[1]))
            }
            VectorTimesScalar | MatrixTimesScalar => Ok(format!("({} * {})", args[0], args[1])),

            // Arithmetic
            FAdd | IAdd => Ok(format!("({} + {})", args[0], args[1])),
            FSub | ISub => Ok(format!("({} - {})", args[0], args[1])),
            FMul | IMul => Ok(format!("({} * {})", args[0], args[1])),
            FDiv | SDiv | UDiv => Ok(format!("({} / {})", args[0], args[1])),
            FRem | FMod | SRem | SMod => Ok(format!("mod({}, {})", args[0], args[1])),

            // Comparisons
            FOrdEqual | IEqual => Ok(format!("({} == {})", args[0], args[1])),
            FOrdNotEqual | INotEqual => Ok(format!("({} != {})", args[0], args[1])),
            FOrdLessThan | SLessThan | ULessThan => Ok(format!("({} < {})", args[0], args[1])),
            FOrdGreaterThan | SGreaterThan | UGreaterThan => Ok(format!("({} > {})", args[0], args[1])),
            FOrdLessThanEqual | SLessThanEqual | ULessThanEqual => {
                Ok(format!("({} <= {})", args[0], args[1]))
            }
            FOrdGreaterThanEqual | SGreaterThanEqual | UGreaterThanEqual => {
                Ok(format!("({} >= {})", args[0], args[1]))
            }

            // Bitwise
            BitwiseAnd => Ok(format!("({} & {})", args[0], args[1])),
            BitwiseOr => Ok(format!("({} | {})", args[0], args[1])),
            BitwiseXor => Ok(format!("({} ^ {})", args[0], args[1])),
            Not => Ok(format!("(~{})", args[0])),
            ShiftLeftLogical => Ok(format!("({} << {})", args[0], args[1])),
            ShiftRightArithmetic | ShiftRightLogical => Ok(format!("({} >> {})", args[0], args[1])),

            // Conversions
            FPToSI => Ok(format!("int({})", args[0])),
            FPToUI => Ok(format!("uint({})", args[0])),
            SIToFP | UIToFP => Ok(format!("float({})", args[0])),
            FPConvert => Ok(format!("float({})", args[0])),
            SConvert | UConvert => Ok(format!("int({})", args[0])),
            Bitcast => Ok(format!("floatBitsToInt({})", args[0])), // TODO: proper bitcast
        }
    }

    fn lower_intrinsic(
        &self,
        name: &str,
        args: &[String],
        arg_exprs: &[Expr],
        _ret_ty: &PolyType<TypeName>,
    ) -> Result<String> {
        match name {
            "tuple_access" => {
                // args[0] is the tuple, args[1] is the index
                Ok(format!("{}[{}]", args[0], args[1]))
            }
            "record_access" => {
                // args[0] is the record, args[1] is the field name (as string literal)
                if let ExprKind::Literal(Literal::String(field)) = &arg_exprs[1].kind {
                    Ok(format!("{}.{}", args[0], field))
                } else {
                    Ok(format!("{}.{}", args[0], args[1]))
                }
            }
            "index" => Ok(format!("{}[{}]", args[0], args[1])),
            "length" => Ok(format!("{}.length()", args[0])),
            _ => bail_glsl!("Unknown intrinsic: {}", name),
        }
    }

    fn lower_loop(
        &mut self,
        loop_var: &str,
        init: &Expr,
        init_bindings: &[(String, Expr)],
        kind: &LoopKind,
        body: &Expr,
        ret_ty: &PolyType<TypeName>,
        output: &mut String,
    ) -> Result<String> {
        // Emit init
        let init_val = self.lower_expr(init, output)?;
        writeln!(
            output,
            "{}{} {} = {};",
            self.indent_str(),
            self.type_to_glsl(&init.ty),
            loop_var,
            init_val
        )
        .unwrap();

        // Emit init bindings
        for (name, expr) in init_bindings {
            let val = self.lower_expr(expr, output)?;
            writeln!(
                output,
                "{}{} {} = {};",
                self.indent_str(),
                self.type_to_glsl(&expr.ty),
                name,
                val
            )
            .unwrap();
        }

        match kind {
            LoopKind::While { cond } => {
                let cond_str = self.lower_expr(cond, output)?;
                writeln!(output, "{}while ({}) {{", self.indent_str(), cond_str).unwrap();
            }
            LoopKind::ForRange { var, bound } => {
                let bound_str = self.lower_expr(bound, output)?;
                writeln!(
                    output,
                    "{}for (int {} = 0; {} < {}; {}++) {{",
                    self.indent_str(),
                    var,
                    var,
                    bound_str,
                    var
                )
                .unwrap();
            }
            LoopKind::For { var, iter } => {
                let iter_str = self.lower_expr(iter, output)?;
                writeln!(
                    output,
                    "{}for (int _i = 0; _i < {}.length(); _i++) {{",
                    self.indent_str(),
                    iter_str
                )
                .unwrap();
                self.indent += 1;
                writeln!(
                    output,
                    "{}{} {} = {}[_i];",
                    self.indent_str(),
                    self.type_to_glsl(&body.ty), // TODO: get element type
                    var,
                    iter_str
                )
                .unwrap();
                self.indent -= 1;
            }
        }

        self.indent += 1;
        let body_result = self.lower_expr(body, output)?;
        writeln!(output, "{}{} = {};", self.indent_str(), loop_var, body_result).unwrap();
        self.indent -= 1;

        writeln!(output, "{}}}", self.indent_str()).unwrap();

        Ok(loop_var.to_string())
    }

    fn type_to_glsl(&self, ty: &PolyType<TypeName>) -> String {
        match ty {
            PolyType::Constructed(name, args) => match name {
                TypeName::Float(32) => "float".to_string(),
                TypeName::Float(64) => "double".to_string(),
                TypeName::Int(32) => "int".to_string(),
                TypeName::Int(64) => "int64_t".to_string(),
                TypeName::UInt(32) => "uint".to_string(),
                TypeName::UInt(64) => "uint64_t".to_string(),
                TypeName::Str(s) if *s == "bool" => "bool".to_string(),
                TypeName::Unit => "void".to_string(),
                // Vec: args[0] is Size(n), args[1] is element type
                TypeName::Vec if args.len() >= 2 => {
                    let n = match &args[0] {
                        PolyType::Constructed(TypeName::Size(n), _) => *n,
                        _ => 4,
                    };
                    let elem = self.type_to_glsl(&args[1]);
                    match elem.as_str() {
                        "float" => format!("vec{}", n),
                        "double" => format!("dvec{}", n),
                        "int" => format!("ivec{}", n),
                        "uint" => format!("uvec{}", n),
                        "bool" => format!("bvec{}", n),
                        _ => format!("vec{}", n),
                    }
                }
                // Mat: args[0] is Size(cols), args[1] is Size(rows), args[2] is element type
                TypeName::Mat if args.len() >= 3 => {
                    let cols = match &args[0] {
                        PolyType::Constructed(TypeName::Size(n), _) => *n,
                        _ => 4,
                    };
                    let rows = match &args[1] {
                        PolyType::Constructed(TypeName::Size(n), _) => *n,
                        _ => 4,
                    };
                    let elem = self.type_to_glsl(&args[2]);
                    match elem.as_str() {
                        "float" => {
                            if rows == cols {
                                format!("mat{}", rows)
                            } else {
                                format!("mat{}x{}", cols, rows)
                            }
                        }
                        "double" => {
                            if rows == cols {
                                format!("dmat{}", rows)
                            } else {
                                format!("dmat{}x{}", cols, rows)
                            }
                        }
                        _ => format!("mat{}", rows),
                    }
                }
                // Array: args[0] is Size(n), args[1] is element type
                TypeName::Array if args.len() >= 2 => {
                    format!("{}[]", self.type_to_glsl(&args[1]))
                }
                _ => "/* unknown */".to_string(),
            },
            _ => "/* unknown */".to_string(),
        }
    }

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }
}

/// Map GLSL.std.450 extended instruction IDs to GLSL function names
fn glsl_ext_to_name(id: u32) -> &'static str {
    match id {
        1 => "round",
        2 => "roundEven",
        3 => "trunc",
        4 => "abs",
        5 => "sign",
        6 => "floor",
        7 => "ceil",
        8 => "fract",
        9 => "radians",
        10 => "degrees",
        11 => "sin",
        12 => "cos",
        13 => "tan",
        14 => "asin",
        15 => "acos",
        16 => "atan",
        17 => "sinh",
        18 => "cosh",
        19 => "tanh",
        20 => "asinh",
        21 => "acosh",
        22 => "atanh",
        23 => "atan", // atan2
        24 => "pow",
        25 => "exp",
        26 => "log",
        27 => "exp2",
        28 => "log2",
        29 => "sqrt",
        30 => "inversesqrt",
        31 => "determinant",
        32 => "inverse",
        37 => "min",
        38 => "max", // UMin
        39 => "max", // SMin
        40 => "max", // FMin
        41 => "max", // UMax
        42 => "max", // SMax
        43 => "max", // FMax
        44 => "clamp",
        45 => "clamp", // FClamp
        46 => "mix",
        66 => "length",
        67 => "distance",
        68 => "cross",
        69 => "normalize",
        70 => "faceforward",
        71 => "reflect",
        72 => "refract",
        _ => "/* unknown_glsl_ext */",
    }
}
