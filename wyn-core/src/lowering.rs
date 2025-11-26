//! SPIR-V Lowering
//!
//! This module converts MIR (from flattening) directly to SPIR-V.
//! It uses a Constructor wrapper that handles variable hoisting automatically.
//! Dependencies are lowered on-demand using ensure_lowered pattern.

/// Early return with a SPIR-V error
macro_rules! bail_spirv {
    ($($arg:tt)*) => {
        return Err(CompilerError::SpirvError(format!($($arg)*)))
    };
}

use crate::ast::TypeName;
use crate::builtin_registry::{BuiltinImpl, BuiltinRegistry, PrimOp};
use crate::error::{CompilerError, Result};
use crate::mir::{self, Def, Expr, ExprKind, Literal, LoopKind, Program};
use polytype::Type as PolyType;
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::dr::{Builder, InsertPoint};
use rspirv::spirv::{self, AddressingModel, Capability, MemoryModel, StorageClass};
use std::collections::HashMap;

/// Tracks the lowering state of each definition
#[derive(Clone, Copy, PartialEq, Eq)]
enum LowerState {
    NotStarted,
    InProgress,
    Done,
}

/// Context for on-demand lowering of MIR to SPIR-V
struct LowerCtx<'a> {
    /// The MIR program being lowered
    program: &'a Program,
    /// Map from definition name to its index in program.defs
    def_index: HashMap<String, usize>,
    /// Lowering state of each definition
    state: HashMap<String, LowerState>,
    /// The SPIR-V builder
    constructor: Constructor,
    /// Entry points to emit (name, execution model)
    entry_points: Vec<(String, spirv::ExecutionModel)>,
}

/// Constructor wraps rspirv::Builder with an ergonomic API that handles:
/// - Automatic variable hoisting to function entry block
/// - Block management with implicit branch from variables block to code
/// - Value and type caching
struct Constructor {
    builder: Builder,

    // Type caching
    void_type: spirv::Word,
    bool_type: spirv::Word,
    i32_type: spirv::Word,
    f32_type: spirv::Word,

    // Constant caching
    int_const_cache: HashMap<i32, spirv::Word>,
    float_const_cache: HashMap<u32, spirv::Word>, // bits as u32
    bool_const_cache: HashMap<bool, spirv::Word>,

    // Current function state
    current_block: Option<spirv::Word>,

    // Variables to hoist (collected during codegen, emitted at function end)
    pending_variables: Vec<(spirv::Word, spirv::Word)>, // (var_id, ptr_type_id)

    // Environment: name -> value ID
    env: HashMap<String, spirv::Word>,

    // Function map: name -> function ID
    functions: HashMap<String, spirv::Word>,

    // GLSL extended instruction set
    glsl_ext_inst_id: spirv::Word,

    // Value types: value ID -> SPIR-V type ID
    value_types: HashMap<spirv::Word, spirv::Word>,

    // Type cache: avoid recreating same types
    vec_type_cache: HashMap<(spirv::Word, u32), spirv::Word>,
    struct_type_cache: HashMap<Vec<spirv::Word>, spirv::Word>,
    ptr_type_cache: HashMap<(spirv::StorageClass, spirv::Word), spirv::Word>,

    // Entry point interface tracking
    entry_point_interfaces: HashMap<String, Vec<spirv::Word>>,
    current_is_entry_point: bool,
    current_output_vars: Vec<spirv::Word>,
    current_input_vars: Vec<(spirv::Word, String, spirv::Word)>, // (var_id, param_name, type_id)

    // Global constants: name -> (var_id, type_id)
    global_constants: HashMap<String, (spirv::Word, spirv::Word)>,
    // Pending constant initializations: (var_id, body_expr)
    pending_constant_inits: Vec<(spirv::Word, Expr)>,

    // Lambda registry: tag index -> (function_name, arity)
    lambda_registry: Vec<(String, usize)>,

    // Builtin function registry
    builtin_registry: BuiltinRegistry,
}

impl Constructor {
    fn new() -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 0);
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void_type = builder.type_void();
        let bool_type = builder.type_bool();
        let i32_type = builder.type_int(32, 1);
        let f32_type = builder.type_float(32);
        let glsl_ext_inst_id = builder.ext_inst_import("GLSL.std.450");

        Constructor {
            builder,
            void_type,
            bool_type,
            i32_type,
            f32_type,
            int_const_cache: HashMap::new(),
            float_const_cache: HashMap::new(),
            bool_const_cache: HashMap::new(),
            current_block: None,
            pending_variables: Vec::new(),
            env: HashMap::new(),
            functions: HashMap::new(),
            glsl_ext_inst_id,
            value_types: HashMap::new(),
            vec_type_cache: HashMap::new(),
            struct_type_cache: HashMap::new(),
            ptr_type_cache: HashMap::new(),
            entry_point_interfaces: HashMap::new(),
            current_is_entry_point: false,
            current_output_vars: Vec::new(),
            current_input_vars: Vec::new(),
            global_constants: HashMap::new(),
            pending_constant_inits: Vec::new(),
            lambda_registry: Vec::new(),
            builtin_registry: BuiltinRegistry::default(),
        }
    }

    /// Get or create a pointer type
    fn get_or_create_ptr_type(
        &mut self,
        storage_class: spirv::StorageClass,
        pointee_id: spirv::Word,
    ) -> spirv::Word {
        let key = (storage_class, pointee_id);
        if let Some(&ty) = self.ptr_type_cache.get(&key) {
            return ty;
        }
        let ty = self.builder.type_pointer(None, storage_class, pointee_id);
        self.ptr_type_cache.insert(key, ty);
        ty
    }

    /// Convert a polytype Type to a SPIR-V type ID
    fn ast_type_to_spirv(&mut self, ty: &PolyType<TypeName>) -> spirv::Word {
        match ty {
            PolyType::Variable(id) => {
                panic!("BUG: Unresolved type variable Variable({}) reached lowering.", id);
            }
            PolyType::Constructed(name, args) => {
                // Assert that no UserVar or SizeVar reaches lowering
                match name {
                    TypeName::UserVar(v) => {
                        panic!("BUG: UserVar('{}') reached lowering.", v);
                    }
                    TypeName::SizeVar(v) => {
                        panic!("BUG: SizeVar('{}') reached lowering.", v);
                    }
                    _ => {}
                }

                match name {
                    TypeName::Int(32) => self.i32_type,
                    TypeName::Float(32) => self.f32_type,
                    TypeName::Int(bits) => self.builder.type_int(*bits as u32, 1),
                    TypeName::UInt(bits) => self.builder.type_int(*bits as u32, 0),
                    TypeName::Float(bits) => self.builder.type_float(*bits as u32),
                    TypeName::Str(s) if *s == "bool" => self.bool_type,
                    TypeName::Str(s) if *s == "tuple" => {
                        // Tuple becomes struct
                        let field_types: Vec<spirv::Word> =
                            args.iter().map(|a| self.ast_type_to_spirv(a)).collect();
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Array => {
                        // Array type: args[0] is size, args[1] is element type
                        if args.len() < 2 {
                            panic!(
                                "BUG: Array type requires 2 arguments (size, element_type), got {}.",
                                args.len()
                            );
                        }
                        // Extract size from args[0]
                        let size = match &args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => {
                                panic!(
                                    "BUG: Array type has invalid size argument: {:?}. This should have been resolved during type checking. \
                                     This typically happens when array size inference fails to constrain a size variable to a concrete value.",
                                    args[0]
                                );
                            }
                        };
                        // Get element type from args[1]
                        let elem_type = self.ast_type_to_spirv(&args[1]);
                        let size_const = self.const_i32(size as i32);
                        self.builder.type_array(elem_type, size_const)
                    }
                    TypeName::Vec => {
                        // Vec type with args: args[0] is size, args[1] is element type
                        if args.len() < 2 {
                            panic!(
                                "BUG: Vec type requires 2 arguments (size, element_type), got {}.",
                                args.len()
                            );
                        }
                        let size = match &args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => {
                                panic!("BUG: Vec type has invalid size argument: {:?}.", args[0]);
                            }
                        };
                        let elem_type = self.ast_type_to_spirv(&args[1]);
                        self.get_or_create_vec_type(elem_type, size)
                    }
                    TypeName::Mat => {
                        // Mat type with args: args[0] is cols, args[1] is rows, args[2] is element type
                        if args.len() < 3 {
                            panic!(
                                "BUG: Mat type requires 3 arguments (cols, rows, element_type), got {}.",
                                args.len()
                            );
                        }
                        let cols = match &args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => {
                                panic!("BUG: Mat type has invalid cols argument: {:?}.", args[0]);
                            }
                        };
                        let rows = match &args[1] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => {
                                panic!("BUG: Mat type has invalid rows argument: {:?}.", args[1]);
                            }
                        };
                        let elem_type = self.ast_type_to_spirv(&args[2]);
                        let col_vec_type = self.get_or_create_vec_type(elem_type, rows);
                        self.builder.type_matrix(col_vec_type, cols)
                    }
                    TypeName::Record(fields) => {
                        // Record becomes a struct, filtering out phantom fields like __lambda_name
                        let real_fields: Vec<_> =
                            fields.iter().filter(|(name, _)| name.as_str() != "__lambda_name").collect();
                        let field_types: Vec<spirv::Word> =
                            real_fields.iter().map(|(_, ty)| self.ast_type_to_spirv(ty)).collect();
                        self.get_or_create_struct_type(field_types)
                    }
                    _ => {
                        panic!(
                            "BUG: Unknown type reached lowering: {:?}. This should have been caught during type checking.",
                            name
                        )
                    }
                }
            }
        }
    }

    /// Get or create a vector type
    fn get_or_create_vec_type(&mut self, elem_type: spirv::Word, size: u32) -> spirv::Word {
        let key = (elem_type, size);
        if let Some(&ty) = self.vec_type_cache.get(&key) {
            return ty;
        }
        let ty = self.builder.type_vector(elem_type, size);
        self.vec_type_cache.insert(key, ty);
        ty
    }

    /// Get or create a struct type
    fn get_or_create_struct_type(&mut self, field_types: Vec<spirv::Word>) -> spirv::Word {
        if let Some(&ty) = self.struct_type_cache.get(&field_types) {
            return ty;
        }
        let ty = self.builder.type_struct(field_types.clone());
        self.struct_type_cache.insert(field_types, ty);
        ty
    }

    /// Record the type of a value
    fn set_value_type(&mut self, value: spirv::Word, ty: spirv::Word) {
        self.value_types.insert(value, ty);
    }

    /// Get the type of a value
    fn get_value_type(&self, value: spirv::Word) -> spirv::Word {
        *self.value_types.get(&value).unwrap_or_else(|| {
            panic!("BUG: Attempted to get type of SPIR-V value %{} but it has no registered type. All values should have their types tracked.", value)
        })
    }

    /// Begin a new function
    fn begin_function(
        &mut self,
        name: &str,
        param_names: &[&str],
        param_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> Result<spirv::Word> {
        let func_type = self.builder.type_function(return_type, param_types.to_vec());
        let func_id =
            self.builder.begin_function(return_type, None, spirv::FunctionControl::NONE, func_type)?;

        self.functions.insert(name.to_string(), func_id);

        // Create function parameters
        for (i, &param_name) in param_names.iter().enumerate() {
            let param_id = self.builder.function_parameter(param_types[i])?;
            self.env.insert(param_name.to_string(), param_id);
        }

        // Create the entry block
        let entry_block_id = self.builder.id();
        self.builder.begin_block(Some(entry_block_id))?;
        self.current_block = Some(entry_block_id);

        // Clear pending variables for this function
        self.pending_variables.clear();

        Ok(func_id)
    }

    /// End the current function
    fn end_function(&mut self) -> Result<()> {
        self.builder.end_function()?;

        // Clear function state
        self.current_block = None;
        self.env.clear();

        Ok(())
    }

    /// Declare a variable (will be hoisted to function entry)
    fn declare_variable(&mut self, _name: &str, value_type: spirv::Word) -> spirv::Word {
        let ptr_type = self.builder.type_pointer(None, StorageClass::Function, value_type);
        let var_id = self.builder.variable(ptr_type, None, StorageClass::Function, None);
        self.pending_variables.push((var_id, ptr_type));
        var_id
    }

    /// Get or create an i32 constant
    fn const_i32(&mut self, value: i32) -> spirv::Word {
        if let Some(&id) = self.int_const_cache.get(&value) {
            return id;
        }
        let id = self.builder.constant_bit32(self.i32_type, value as u32);
        self.int_const_cache.insert(value, id);
        id
    }

    /// Get or create an f32 constant
    fn const_f32(&mut self, value: f32) -> spirv::Word {
        let bits = value.to_bits();
        if let Some(&id) = self.float_const_cache.get(&bits) {
            return id;
        }
        let id = self.builder.constant_bit32(self.f32_type, bits);
        self.float_const_cache.insert(bits, id);
        id
    }

    /// Get or create a bool constant
    fn const_bool(&mut self, value: bool) -> spirv::Word {
        if let Some(&id) = self.bool_const_cache.get(&value) {
            return id;
        }
        let id = if value {
            self.builder.constant_true(self.bool_type)
        } else {
            self.builder.constant_false(self.bool_type)
        };
        self.bool_const_cache.insert(value, id);
        id
    }

    /// Begin a block (must be called before emitting instructions into it)
    fn begin_block(&mut self, block_id: spirv::Word) -> Result<()> {
        self.builder.begin_block(Some(block_id))?;
        self.current_block = Some(block_id);
        Ok(())
    }

    /// Emit a conditional branch with selection merge
    fn branch_conditional(
        &mut self,
        cond: spirv::Word,
        true_block: spirv::Word,
        false_block: spirv::Word,
        merge_block: spirv::Word,
    ) -> Result<()> {
        self.builder.selection_merge(merge_block, spirv::SelectionControl::NONE)?;
        self.builder.branch_conditional(cond, true_block, false_block, [])?;
        Ok(())
    }

    /// Get array type
    fn type_array(&mut self, elem_type: spirv::Word, length: u32) -> spirv::Word {
        let length_id = self.const_i32(length as i32);
        self.builder.type_array(elem_type, length_id)
    }
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        let mut constructor = Constructor::new();
        constructor.lambda_registry = program.lambda_registry.clone();

        // Build index from name to def position
        let mut def_index = HashMap::new();
        let mut entry_points = Vec::new();

        for (i, def) in program.defs.iter().enumerate() {
            let name = match def {
                Def::Function { name, attributes, .. } => {
                    // Collect entry points
                    for attr in attributes {
                        match attr.name.as_str() {
                            "vertex" => entry_points.push((name.clone(), spirv::ExecutionModel::Vertex)),
                            "fragment" => {
                                entry_points.push((name.clone(), spirv::ExecutionModel::Fragment))
                            }
                            _ => {}
                        }
                    }
                    name.clone()
                }
                Def::Constant { name, .. } => name.clone(),
            };
            def_index.insert(name, i);
        }

        LowerCtx {
            program,
            def_index,
            state: HashMap::new(),
            constructor,
            entry_points,
        }
    }

    /// Ensure a definition is lowered, recursively lowering dependencies first
    fn ensure_lowered(&mut self, name: &str) -> Result<()> {
        match self.state.get(name).copied().unwrap_or(LowerState::NotStarted) {
            LowerState::Done => return Ok(()),
            LowerState::InProgress => {
                bail_spirv!("Recursive definition detected: {}", name);
            }
            LowerState::NotStarted => { /* proceed */ }
        }

        // Look up the definition
        let def_idx = match self.def_index.get(name) {
            Some(&idx) => idx,
            None => return Ok(()), // Not a user def (might be a builtin)
        };

        self.state.insert(name.to_string(), LowerState::InProgress);

        let def = &self.program.defs[def_idx];
        self.lower_def(def)?;

        self.state.insert(name.to_string(), LowerState::Done);
        Ok(())
    }

    /// Lower a single definition
    fn lower_def(&mut self, def: &Def) -> Result<()> {
        match def {
            Def::Function {
                name,
                params,
                ret_type,
                attributes,
                param_attributes,
                return_attributes,
                body,
                ..
            } => {
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                // Check if this is an entry point
                let is_entry = attributes.iter().any(|a| a.name == "vertex" || a.name == "fragment");

                if is_entry {
                    lower_entry_point(
                        &mut self.constructor,
                        name,
                        params,
                        ret_type,
                        param_attributes,
                        return_attributes,
                        body,
                    )?;
                } else {
                    lower_regular_function(&mut self.constructor, name, params, ret_type, body)?;
                }
            }
            Def::Constant { name, ty, body, .. } => {
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                // Create global variable for constant (Private storage class)
                let value_type = self.constructor.ast_type_to_spirv(ty);
                let ptr_type = self.constructor.get_or_create_ptr_type(StorageClass::Private, value_type);
                let var_id = self.constructor.builder.variable(ptr_type, None, StorageClass::Private, None);

                // Store for later initialization and lookup
                self.constructor.global_constants.insert(name.clone(), (var_id, value_type));
                self.constructor.pending_constant_inits.push((var_id, body.clone()));
            }
        }
        Ok(())
    }

    /// Walk an expression and ensure all referenced definitions are lowered
    fn ensure_deps_lowered(&mut self, expr: &Expr) -> Result<()> {
        match &expr.kind {
            ExprKind::Var(name) => {
                self.ensure_lowered(name)?;
            }
            ExprKind::Call { func, args } => {
                self.ensure_lowered(func)?;
                for arg in args {
                    self.ensure_deps_lowered(arg)?;
                }
            }
            ExprKind::BinOp { lhs, rhs, .. } => {
                self.ensure_deps_lowered(lhs)?;
                self.ensure_deps_lowered(rhs)?;
            }
            ExprKind::UnaryOp { operand, .. } => {
                self.ensure_deps_lowered(operand)?;
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.ensure_deps_lowered(cond)?;
                self.ensure_deps_lowered(then_branch)?;
                self.ensure_deps_lowered(else_branch)?;
            }
            ExprKind::Let { value, body, .. } => {
                self.ensure_deps_lowered(value)?;
                self.ensure_deps_lowered(body)?;
            }
            ExprKind::Loop {
                init_bindings, body, ..
            } => {
                for (_, init) in init_bindings {
                    self.ensure_deps_lowered(init)?;
                }
                self.ensure_deps_lowered(body)?;
            }
            ExprKind::Intrinsic { args, .. } => {
                for arg in args {
                    self.ensure_deps_lowered(arg)?;
                }
            }
            ExprKind::Attributed { expr, .. } => {
                self.ensure_deps_lowered(expr)?;
            }
            ExprKind::Literal(lit) => {
                // Check for closure records with __lambda_name field
                if let Some(lambda_name) = crate::mir::extract_lambda_name(expr) {
                    self.ensure_lowered(lambda_name)?;
                }
                // Recurse into record field values
                if let crate::mir::Literal::Record(fields) = lit {
                    for (_, field_expr) in fields {
                        self.ensure_deps_lowered(field_expr)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Run the lowering, starting from entry points
    fn run(mut self) -> Result<Vec<u32>> {
        // Lower all entry points (and their dependencies)
        let entry_names: Vec<String> = self.entry_points.iter().map(|(n, _)| n.clone()).collect();
        for name in entry_names {
            self.ensure_lowered(&name)?;
        }

        // Generate _init function to initialize global constants
        if !self.constructor.pending_constant_inits.is_empty() {
            generate_init_function(&mut self.constructor)?;
        }

        // Emit entry points with interface variables
        for (name, model) in &self.entry_points {
            if let Some(&func_id) = self.constructor.functions.get(name) {
                let interfaces =
                    self.constructor.entry_point_interfaces.get(name).cloned().unwrap_or_default();
                self.constructor.builder.entry_point(*model, func_id, name, interfaces);

                // Add execution mode for fragment shaders
                if *model == spirv::ExecutionModel::Fragment {
                    self.constructor.builder.execution_mode(
                        func_id,
                        spirv::ExecutionMode::OriginUpperLeft,
                        [],
                    );
                }
            }
        }

        Ok(self.constructor.builder.module().assemble())
    }
}

/// Lower a MIR program to SPIR-V
pub fn lower(program: &mir::Program) -> Result<Vec<u32>> {
    let ctx = LowerCtx::new(program);
    ctx.run()
}

/// Generate _init function that initializes all global constants
fn generate_init_function(constructor: &mut Constructor) -> Result<()> {
    // Take pending inits to avoid borrow issues
    let inits = std::mem::take(&mut constructor.pending_constant_inits);

    constructor.begin_function("_init", &[], &[], constructor.void_type)?;

    for (var_id, body) in inits {
        let value = lower_expr(constructor, &body)?;
        constructor.builder.store(var_id, value, None, [])?;
    }

    constructor.builder.ret()?;
    constructor.end_function()?;

    Ok(())
}

fn lower_regular_function(
    constructor: &mut Constructor,
    name: &str,
    params: &[mir::Param],
    ret_type: &PolyType<TypeName>,
    body: &Expr,
) -> Result<()> {
    let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
    let param_types: Vec<spirv::Word> =
        params.iter().map(|p| constructor.ast_type_to_spirv(&p.ty)).collect();
    let return_type = constructor.ast_type_to_spirv(ret_type);
    constructor.begin_function(name, &param_names, &param_types, return_type)?;

    let result = lower_expr(constructor, body)?;
    constructor.builder.ret_value(result)?;

    constructor.end_function()?;
    Ok(())
}

fn lower_entry_point(
    constructor: &mut Constructor,
    name: &str,
    params: &[mir::Param],
    ret_type: &PolyType<TypeName>,
    param_attributes: &[Vec<mir::Attribute>],
    return_attributes: &[Vec<mir::Attribute>],
    body: &Expr,
) -> Result<()> {
    constructor.current_is_entry_point = true;
    constructor.current_output_vars.clear();
    constructor.current_input_vars.clear();

    let mut interface_vars = Vec::new();

    // Create Input variables for parameters
    for (i, param) in params.iter().enumerate() {
        let param_type_id = constructor.ast_type_to_spirv(&param.ty);
        let ptr_type_id = constructor.get_or_create_ptr_type(StorageClass::Input, param_type_id);
        let var_id = constructor.builder.variable(ptr_type_id, None, StorageClass::Input, None);

        // Add decorations from attributes
        if i < param_attributes.len() {
            for attr in &param_attributes[i] {
                match attr.name.as_str() {
                    "location" => {
                        if let Some(loc_str) = attr.args.first() {
                            if let Ok(loc) = loc_str.parse::<u32>() {
                                constructor.builder.decorate(
                                    var_id,
                                    spirv::Decoration::Location,
                                    [rspirv::dr::Operand::LiteralBit32(loc)],
                                );
                            }
                        }
                    }
                    "builtin" => {
                        if let Some(builtin_str) = attr.args.first() {
                            if let Some(builtin) = parse_builtin(builtin_str) {
                                constructor.builder.decorate(
                                    var_id,
                                    spirv::Decoration::BuiltIn,
                                    [rspirv::dr::Operand::BuiltIn(builtin)],
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        interface_vars.push(var_id);
        constructor.current_input_vars.push((var_id, param.name.clone(), param_type_id));
    }

    // Create Output variables for return values
    // Get return type components (if tuple, each element gets its own output)
    let ret_type_id = constructor.ast_type_to_spirv(ret_type);
    let is_void = matches!(ret_type, PolyType::Constructed(TypeName::Str(s), _) if *s == "void");

    if !is_void {
        // Check if return is a tuple (multiple outputs)
        if let PolyType::Constructed(TypeName::Str(s), component_types) = ret_type {
            if *s == "tuple" && !return_attributes.is_empty() {
                // Multiple outputs - one variable per component
                for (i, comp_ty) in component_types.iter().enumerate() {
                    let comp_type_id = constructor.ast_type_to_spirv(comp_ty);
                    let ptr_type_id =
                        constructor.get_or_create_ptr_type(StorageClass::Output, comp_type_id);
                    let var_id =
                        constructor.builder.variable(ptr_type_id, None, StorageClass::Output, None);

                    // Add decorations
                    if i < return_attributes.len() {
                        for attr in &return_attributes[i] {
                            match attr.name.as_str() {
                                "location" => {
                                    if let Some(loc_str) = attr.args.first() {
                                        if let Ok(loc) = loc_str.parse::<u32>() {
                                            constructor.builder.decorate(
                                                var_id,
                                                spirv::Decoration::Location,
                                                [rspirv::dr::Operand::LiteralBit32(loc)],
                                            );
                                        }
                                    }
                                }
                                "builtin" => {
                                    if let Some(builtin_str) = attr.args.first() {
                                        if let Some(builtin) = parse_builtin(builtin_str) {
                                            constructor.builder.decorate(
                                                var_id,
                                                spirv::Decoration::BuiltIn,
                                                [rspirv::dr::Operand::BuiltIn(builtin)],
                                            );
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    interface_vars.push(var_id);
                    constructor.current_output_vars.push(var_id);
                }
            } else {
                // Single output
                let ptr_type_id = constructor.get_or_create_ptr_type(StorageClass::Output, ret_type_id);
                let var_id = constructor.builder.variable(ptr_type_id, None, StorageClass::Output, None);

                // Add decorations from first return attribute set
                if let Some(attrs) = return_attributes.first() {
                    for attr in attrs {
                        match attr.name.as_str() {
                            "location" => {
                                if let Some(loc_str) = attr.args.first() {
                                    if let Ok(loc) = loc_str.parse::<u32>() {
                                        constructor.builder.decorate(
                                            var_id,
                                            spirv::Decoration::Location,
                                            [rspirv::dr::Operand::LiteralBit32(loc)],
                                        );
                                    }
                                }
                            }
                            "builtin" => {
                                if let Some(builtin_str) = attr.args.first() {
                                    if let Some(builtin) = parse_builtin(builtin_str) {
                                        constructor.builder.decorate(
                                            var_id,
                                            spirv::Decoration::BuiltIn,
                                            [rspirv::dr::Operand::BuiltIn(builtin)],
                                        );
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }

                interface_vars.push(var_id);
                constructor.current_output_vars.push(var_id);
            }
        } else {
            // Single non-tuple output
            let ptr_type_id = constructor.get_or_create_ptr_type(StorageClass::Output, ret_type_id);
            let var_id = constructor.builder.variable(ptr_type_id, None, StorageClass::Output, None);

            if let Some(attrs) = return_attributes.first() {
                for attr in attrs {
                    match attr.name.as_str() {
                        "location" => {
                            if let Some(loc_str) = attr.args.first() {
                                if let Ok(loc) = loc_str.parse::<u32>() {
                                    constructor.builder.decorate(
                                        var_id,
                                        spirv::Decoration::Location,
                                        [rspirv::dr::Operand::LiteralBit32(loc)],
                                    );
                                }
                            }
                        }
                        "builtin" => {
                            if let Some(builtin_str) = attr.args.first() {
                                if let Some(builtin) = parse_builtin(builtin_str) {
                                    constructor.builder.decorate(
                                        var_id,
                                        spirv::Decoration::BuiltIn,
                                        [rspirv::dr::Operand::BuiltIn(builtin)],
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            interface_vars.push(var_id);
            constructor.current_output_vars.push(var_id);
        }
    }

    // Store interface variables for entry point declaration
    constructor.entry_point_interfaces.insert(name.to_string(), interface_vars);

    // Create void(void) function for entry point
    let func_type = constructor.builder.type_function(constructor.void_type, vec![]);
    let func_id = constructor.builder.begin_function(
        constructor.void_type,
        None,
        spirv::FunctionControl::NONE,
        func_type,
    )?;
    constructor.functions.insert(name.to_string(), func_id);

    // Create entry block
    let block_id = constructor.builder.id();
    constructor.builder.begin_block(Some(block_id))?;
    constructor.current_block = Some(block_id);

    // Call _init to initialize global constants
    if let Some(&init_func_id) = constructor.functions.get("_init") {
        constructor.builder.function_call(constructor.void_type, None, init_func_id, [])?;
    }

    // Load input variables into environment
    for (var_id, param_name, type_id) in constructor.current_input_vars.clone() {
        let loaded = constructor.builder.load(type_id, None, var_id, None, [])?;
        constructor.env.insert(param_name, loaded);
    }

    // Lower the body
    let result = lower_expr(constructor, body)?;

    // Store result to output variables
    if !constructor.current_output_vars.is_empty() {
        // Check if result is a tuple that needs to be decomposed
        if let PolyType::Constructed(TypeName::Str(s), component_types) = ret_type {
            if *s == "tuple" && constructor.current_output_vars.len() > 1 {
                // Extract each component and store
                for (i, &output_var) in constructor.current_output_vars.clone().iter().enumerate() {
                    let comp_type_id = constructor.ast_type_to_spirv(&component_types[i]);
                    let component =
                        constructor.builder.composite_extract(comp_type_id, None, result, [i as u32])?;
                    constructor.builder.store(output_var, component, None, [])?;
                }
            } else if let Some(&output_var) = constructor.current_output_vars.first() {
                constructor.builder.store(output_var, result, None, [])?;
            }
        } else if let Some(&output_var) = constructor.current_output_vars.first() {
            constructor.builder.store(output_var, result, None, [])?;
        }
    }

    // Return void
    constructor.builder.ret()?;
    constructor.builder.end_function()?;

    // Clean up
    constructor.current_is_entry_point = false;
    constructor.env.clear();

    Ok(())
}

/// Parse a builtin string like "Position" into a SPIR-V BuiltIn
fn parse_builtin(s: &str) -> Option<spirv::BuiltIn> {
    match s {
        "Position" => Some(spirv::BuiltIn::Position),
        "VertexIndex" => Some(spirv::BuiltIn::VertexIndex),
        "InstanceIndex" => Some(spirv::BuiltIn::InstanceIndex),
        "FragCoord" => Some(spirv::BuiltIn::FragCoord),
        "PointSize" => Some(spirv::BuiltIn::PointSize),
        "ClipDistance" => Some(spirv::BuiltIn::ClipDistance),
        "CullDistance" => Some(spirv::BuiltIn::CullDistance),
        "FragDepth" => Some(spirv::BuiltIn::FragDepth),
        _ => None,
    }
}

fn lower_expr(constructor: &mut Constructor, expr: &Expr) -> Result<spirv::Word> {
    match &expr.kind {
        ExprKind::Literal(lit) => lower_literal(constructor, lit),

        ExprKind::Var(name) => {
            // First check local environment
            if let Some(&id) = constructor.env.get(name) {
                return Ok(id);
            }
            // Then check global constants (load from global variable)
            if let Some(&(var_id, type_id)) = constructor.global_constants.get(name) {
                return Ok(constructor.builder.load(type_id, None, var_id, None, [])?);
            }
            Err(CompilerError::SpirvError(format!("Undefined variable: {}", name)))
        }

        ExprKind::BinOp { op, lhs, rhs } => {
            let lhs_id = lower_expr(constructor, lhs)?;
            let rhs_id = lower_expr(constructor, rhs)?;
            let same_out_type = constructor.ast_type_to_spirv(&lhs.ty);
            let bool_type = constructor.bool_type;

            use PolyType::*;
            use TypeName::*;
            match (op.as_str(), &lhs.ty) {
                // Float operations
                ("+", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_add(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("-", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_sub(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("*", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_mul(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("/", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_div(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("%", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_rem(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("==", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                ("!=", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_not_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                ("<", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_less_than(bool_type, None, lhs_id, rhs_id)?)
                }
                ("<=", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_less_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                (">", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_greater_than(bool_type, None, lhs_id, rhs_id)?)
                }
                (">=", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_greater_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }

                // Unsigned integer operations
                ("/", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_div(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("%", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_mod(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("<", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_less_than(bool_type, None, lhs_id, rhs_id)?)
                }
                ("<=", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_less_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                (">", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_greater_than(bool_type, None, lhs_id, rhs_id)?)
                }
                (">=", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_greater_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }

                // Signed integer operations (and fallback for +, -, *, ==, != which are the same for signed/unsigned)
                ("+", _) => Ok(constructor.builder.i_add(same_out_type, None, lhs_id, rhs_id)?),
                ("-", _) => Ok(constructor.builder.i_sub(same_out_type, None, lhs_id, rhs_id)?),
                ("*", _) => Ok(constructor.builder.i_mul(same_out_type, None, lhs_id, rhs_id)?),
                ("/", _) => Ok(constructor.builder.s_div(same_out_type, None, lhs_id, rhs_id)?),
                ("%", _) => Ok(constructor.builder.s_mod(same_out_type, None, lhs_id, rhs_id)?),
                ("==", _) => Ok(constructor.builder.i_equal(bool_type, None, lhs_id, rhs_id)?),
                ("!=", _) => Ok(constructor.builder.i_not_equal(bool_type, None, lhs_id, rhs_id)?),
                ("<", _) => Ok(constructor.builder.s_less_than(bool_type, None, lhs_id, rhs_id)?),
                ("<=", _) => Ok(constructor.builder.s_less_than_equal(bool_type, None, lhs_id, rhs_id)?),
                (">", _) => Ok(constructor.builder.s_greater_than(bool_type, None, lhs_id, rhs_id)?),
                (">=", _) => {
                    Ok(constructor.builder.s_greater_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }

                _ => Err(CompilerError::SpirvError(format!("Unknown binary op: {}", op))),
            }
        }

        ExprKind::UnaryOp { op, operand } => {
            let operand_id = lower_expr(constructor, operand)?;
            let same_type = constructor.ast_type_to_spirv(&operand.ty);

            use PolyType::*;
            use TypeName::*;
            match (op.as_str(), &operand.ty) {
                ("-", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_negate(same_type, None, operand_id)?)
                }
                ("-", Constructed(UInt(bits), _)) => Err(CompilerError::SpirvError(format!(
                    "Cannot negate unsigned integer type u{}",
                    bits
                ))),
                ("-", _) => Ok(constructor.builder.s_negate(same_type, None, operand_id)?),
                ("!", _) => Ok(constructor.builder.logical_not(constructor.bool_type, None, operand_id)?),
                _ => Err(CompilerError::SpirvError(format!("Unknown unary op: {}", op))),
            }
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let cond_id = lower_expr(constructor, cond)?;

            // Get the result type from the expression
            let result_type = constructor.ast_type_to_spirv(&expr.ty);

            // Create blocks
            let then_block_id = constructor.builder.id();
            let else_block_id = constructor.builder.id();
            let merge_block_id = constructor.builder.id();

            // Branch based on condition
            constructor.branch_conditional(cond_id, then_block_id, else_block_id, merge_block_id)?;

            // Then block
            constructor.begin_block(then_block_id)?;
            let then_result = lower_expr(constructor, then_branch)?;
            let then_exit_block = constructor.current_block.unwrap();

            constructor.builder.branch(merge_block_id)?;

            // Else block
            constructor.begin_block(else_block_id)?;
            let else_result = lower_expr(constructor, else_branch)?;
            let else_exit_block = constructor.current_block.unwrap();
            constructor.builder.branch(merge_block_id)?;

            // Merge block with phi
            constructor.begin_block(merge_block_id)?;
            let incoming = vec![(then_result, then_exit_block), (else_result, else_exit_block)];
            let result = constructor.builder.phi(result_type, None, incoming)?;
            Ok(result)
        }

        ExprKind::Let { name, value, body } => {
            let value_id = lower_expr(constructor, value)?;
            constructor.env.insert(name.clone(), value_id);
            let result = lower_expr(constructor, body)?;
            constructor.env.remove(name);
            Ok(result)
        }

        ExprKind::Loop {
            init_bindings,
            kind,
            body,
        } => {
            // Create blocks for loop structure
            let header_block_id = constructor.builder.id();
            let body_block_id = constructor.builder.id();
            let continue_block_id = constructor.builder.id();
            let merge_block_id = constructor.builder.id();

            // Evaluate init expressions in order, collecting values and types
            let mut init_values = Vec::new();
            for (name, init_expr) in init_bindings {
                let init_val = lower_expr(constructor, init_expr)?;
                let init_type = constructor.ast_type_to_spirv(&init_expr.ty);
                constructor.env.insert(name.clone(), init_val);
                init_values.push((name.clone(), init_val, init_type));
            }
            let pre_header_block = constructor.current_block.unwrap();

            // Branch to header
            constructor.builder.branch(header_block_id)?;

            // Header block - we'll add phi nodes later
            constructor.begin_block(header_block_id)?;
            let header_block_idx = constructor.builder.selected_block().expect("No block selected");

            // Allocate phi IDs and add to environment so condition/body can reference them
            let mut phi_info = Vec::new();
            for (name, init_val, var_type) in &init_values {
                let phi_id = constructor.builder.id();
                phi_info.push((name.clone(), phi_id, *init_val, *var_type));
                constructor.env.insert(name.clone(), phi_id);
            }

            // Generate condition based on loop kind
            let cond_id = match kind {
                LoopKind::While { cond } => lower_expr(constructor, cond)?,
                LoopKind::ForRange { var, bound } => {
                    let bound_id = lower_expr(constructor, bound)?;
                    let var_id = *constructor.env.get(var).ok_or_else(|| {
                        CompilerError::SpirvError(format!("Loop variable {} not found", var))
                    })?;
                    constructor.builder.s_less_than(constructor.bool_type, None, var_id, bound_id)?
                }
                LoopKind::For { .. } => {
                    bail_spirv!("For-in loops not yet implemented");
                }
            };

            // Loop merge and conditional branch
            constructor.builder.loop_merge(
                merge_block_id,
                continue_block_id,
                spirv::LoopControl::NONE,
                [],
            )?;
            constructor.builder.branch_conditional(cond_id, body_block_id, merge_block_id, [])?;

            // Body block
            constructor.begin_block(body_block_id)?;
            let body_result = lower_expr(constructor, body)?;
            constructor.builder.branch(continue_block_id)?;

            // Continue block - extract updated values from body result
            constructor.begin_block(continue_block_id)?;

            // Extract continue values based on number of loop variables
            let continue_values: Vec<spirv::Word> = if phi_info.len() == 1 {
                // Single loop variable - body result is the new value
                vec![body_result]
            } else {
                // Multiple loop variables - body returns tuple, extract each component
                let mut values = Vec::new();
                for i in 0..phi_info.len() {
                    let comp_type = phi_info[i].3; // var_type
                    let extracted =
                        constructor.builder.composite_extract(comp_type, None, body_result, [i as u32])?;
                    values.push(extracted);
                }
                values
            };

            constructor.builder.branch(header_block_id)?;
            let continue_block_id = continue_block_id;

            // Now go back and insert phi nodes at the beginning of header block
            // We need to save current function context and temporarily work on header
            let saved_current_block = constructor.current_block;
            constructor.builder.select_block(Some(header_block_idx))?;

            for (i, (name, phi_id, init_val, var_type)) in phi_info.iter().enumerate() {
                let incoming = vec![
                    (*init_val, pre_header_block),
                    (continue_values[i], continue_block_id),
                ];
                constructor.builder.insert_phi(InsertPoint::Begin, *var_type, Some(*phi_id), incoming)?;
            }

            // Deselect block before continuing
            constructor.builder.select_block(None)?;

            // Continue to merge block
            constructor.begin_block(merge_block_id)?;

            // Clean up environment
            for (name, _, _, _) in &phi_info {
                constructor.env.remove(name);
            }

            // Return the first phi value as loop result
            if let Some((_, phi_id, _, _)) = phi_info.first() {
                Ok(*phi_id)
            } else {
                Ok(constructor.const_i32(0)) // Empty loop
            }
        }

        ExprKind::Call { func, args } => {
            // Get the result type from the expression
            let result_type = constructor.ast_type_to_spirv(&expr.ty);

            // Special case for map - extract lambda name from closure before lowering
            if func == "map" {
                // map closure array -> array
                // args[0] is closure record {__lambda_name: "...", captures...}
                // args[1] is input array
                if args.len() != 2 {
                    bail_spirv!("map requires 2 args (closure, array), got {}", args.len());
                }

                // Extract lambda name from closure record's __lambda_name field
                let lambda_name = match &args[0].kind {
                    ExprKind::Literal(mir::Literal::Record(fields)) => {
                        // Find __lambda_name field
                        fields
                            .iter()
                            .find(|(name, _)| name == "__lambda_name")
                            .and_then(|(_, expr)| match &expr.kind {
                                ExprKind::Literal(mir::Literal::String(s)) => Some(s.clone()),
                                _ => None,
                            })
                            .ok_or_else(|| {
                                CompilerError::SpirvError(
                                    "Closure record missing __lambda_name field".to_string(),
                                )
                            })?
                    }
                    _ => {
                        bail_spirv!("map closure argument must be a record literal");
                    }
                };

                // Now lower both args normally
                let closure_val = lower_expr(constructor, &args[0])?;
                let array_val = lower_expr(constructor, &args[1])?;

                // Get array info from MIR types
                let (array_size, elem_mir_type) = match &expr.ty {
                    PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 2 => {
                        let size = match &type_args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => bail_spirv!("Invalid array size type"),
                        };
                        (size, &type_args[1])
                    }
                    _ => bail_spirv!("map result must be array type"),
                };

                let elem_type = constructor.ast_type_to_spirv(elem_mir_type);

                // Look up the lambda function by name
                let lambda_func_id = *constructor.functions.get(&lambda_name).ok_or_else(|| {
                    CompilerError::SpirvError(format!("Lambda function not found: {}", lambda_name))
                })?;

                // Build result array by calling lambda for each element
                let mut result_elements = Vec::new();
                for i in 0..array_size {
                    // Extract element from input array
                    let input_elem =
                        constructor.builder.composite_extract(elem_type, None, array_val, [i])?;

                    // Call lambda with closure and element
                    let args = vec![closure_val, input_elem];
                    let result_elem =
                        constructor.builder.function_call(elem_type, None, lambda_func_id, args)?;
                    result_elements.push(result_elem);
                }

                // Construct result array
                return Ok(constructor.builder.composite_construct(result_type, None, result_elements)?);
            }

            // For all other calls, lower arguments normally
            let arg_ids: Vec<spirv::Word> =
                args.iter().map(|a| lower_expr(constructor, a)).collect::<Result<Vec<_>>>()?;

            // Check for builtin vector constructors
            match func.as_str() {
                "vec2" | "vec3" | "vec4" => {
                    // Use the result type which should be the proper vector type
                    Ok(constructor.builder.composite_construct(result_type, None, arg_ids)?)
                }
                _ => {
                    // Check if it's a builtin function
                    if let Some(overloads) = constructor.builtin_registry.get_overloads(func) {
                        // All overloads share the same implementation, only types differ
                        let builtin = &overloads[0];
                        match &builtin.implementation {
                            BuiltinImpl::PrimOp(spirv_op) => {
                                // Handle core SPIR-V operations
                                match spirv_op {
                                    PrimOp::GlslExt(ext_op) => {
                                        // Call GLSL extended instruction
                                        let glsl_id = constructor.glsl_ext_inst_id;
                                        let operands: Vec<Operand> =
                                            arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();
                                        Ok(constructor.builder.ext_inst(
                                            result_type,
                                            None,
                                            glsl_id,
                                            *ext_op,
                                            operands,
                                        )?)
                                    }
                                    PrimOp::Dot => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("dot requires 2 args");
                                        }
                                        Ok(constructor.builder.dot(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::MatrixTimesMatrix => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("matrix × matrix requires 2 args");
                                        }
                                        Ok(constructor.builder.matrix_times_matrix(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::MatrixTimesVector => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("matrix × vector requires 2 args");
                                        }
                                        Ok(constructor.builder.matrix_times_vector(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::VectorTimesMatrix => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("vector × matrix requires 2 args");
                                        }
                                        Ok(constructor.builder.vector_times_matrix(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    // Type conversions
                                    PrimOp::FPToSI => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("FPToSI requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_f_to_s(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::FPToUI => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("FPToUI requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_f_to_u(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::SIToFP => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("SIToFP requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_s_to_f(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::UIToFP => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("UIToFP requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_u_to_f(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::FPConvert => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("FPConvert requires 1 arg");
                                        }
                                        Ok(constructor.builder.f_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::SConvert => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("SConvert requires 1 arg");
                                        }
                                        Ok(constructor.builder.s_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::UConvert => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("UConvert requires 1 arg");
                                        }
                                        Ok(constructor.builder.u_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::Bitcast => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("Bitcast requires 1 arg");
                                        }
                                        Ok(constructor.builder.bitcast(result_type, None, arg_ids[0])?)
                                    }
                                    _ => {
                                        bail_spirv!("Unsupported PrimOp for: {}", func)
                                    }
                                }
                            }
                            BuiltinImpl::Intrinsic(custom_impl) => {
                                use crate::builtin_registry::Intrinsic;
                                match custom_impl {
                                    Intrinsic::MatrixFromVectors => {
                                        // Matrix from array of vectors: In SPIR-V, matrices ARE arrays of column vectors
                                        // So this is essentially a no-op/identity at the SPIR-V level
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("matav expects exactly 1 argument");
                                        }
                                        Ok(arg_ids[0])
                                    }
                                    Intrinsic::Uninit => {
                                        // Return an undefined value of the result type
                                        Ok(constructor.builder.undef(result_type, None))
                                    }
                                    Intrinsic::Replicate => {
                                        // replicate n val: create array of n copies of val
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("replicate expects exactly 2 arguments");
                                        }
                                        // Extract array size from result type
                                        if let PolyType::Constructed(TypeName::Array, type_args) = &expr.ty
                                        {
                                            if let Some(PolyType::Constructed(TypeName::Size(n), _)) =
                                                type_args.get(0)
                                            {
                                                // Build array by repeating the value
                                                let val_id = arg_ids[1]; // second arg is the value
                                                let elem_ids: Vec<_> = (0..*n).map(|_| val_id).collect();
                                                Ok(constructor.builder.composite_construct(
                                                    result_type,
                                                    None,
                                                    elem_ids,
                                                )?)
                                            } else {
                                                bail_spirv!(
                                                    "replicate: cannot determine array size at compile time"
                                                )
                                            }
                                        } else {
                                            bail_spirv!("replicate: result type is not an array")
                                        }
                                    }
                                    Intrinsic::ArrayUpdate => {
                                        // array_update arr idx val: functional update, returns new array
                                        if arg_ids.len() != 3 {
                                            bail_spirv!("array_update expects exactly 3 arguments");
                                        }
                                        let arr_id = arg_ids[0];
                                        let idx_id = arg_ids[1];
                                        let val_id = arg_ids[2];

                                        // Store array in a variable, update element, load back
                                        let arr_type = result_type;
                                        let ptr_type = constructor.builder.type_pointer(
                                            None,
                                            StorageClass::Function,
                                            arr_type,
                                        );
                                        let arr_var = constructor.builder.variable(
                                            ptr_type,
                                            None,
                                            StorageClass::Function,
                                            None,
                                        );
                                        constructor.builder.store(arr_var, arr_id, None, [])?;

                                        // Get pointer to element and store new value
                                        let elem_type = constructor.ast_type_to_spirv(&args[2].ty);
                                        let elem_ptr_type = constructor.builder.type_pointer(
                                            None,
                                            StorageClass::Function,
                                            elem_type,
                                        );
                                        let elem_ptr = constructor.builder.access_chain(
                                            elem_ptr_type,
                                            None,
                                            arr_var,
                                            [idx_id],
                                        )?;
                                        constructor.builder.store(elem_ptr, val_id, None, [])?;

                                        // Load and return the updated array
                                        Ok(constructor.builder.load(arr_type, None, arr_var, None, [])?)
                                    }
                                    Intrinsic::Placeholder if func == "length" => {
                                        // Array length: extract size from array type
                                        if args.len() != 1 {
                                            bail_spirv!("length expects exactly 1 argument");
                                        }
                                        if let PolyType::Constructed(TypeName::Array, type_args) =
                                            &args[0].ty
                                        {
                                            match type_args.get(0) {
                                                Some(PolyType::Constructed(TypeName::Size(n), _)) => {
                                                    Ok(constructor.const_i32(*n as i32))
                                                }
                                                _ => bail_spirv!(
                                                    "Cannot determine compile-time array size for length: {:?}",
                                                    type_args.get(0)
                                                ),
                                            }
                                        } else {
                                            bail_spirv!("length called on non-array type: {:?}", args[0].ty)
                                        }
                                    }
                                    Intrinsic::Placeholder => {
                                        // Other placeholder intrinsics should have been desugared
                                        bail_spirv!(
                                            "Placeholder intrinsic '{}' should have been desugared before lowering",
                                            func
                                        )
                                    }
                                }
                            }
                            BuiltinImpl::CoreFn(core_fn_name) => {
                                // Library-level builtins implemented as normal functions in prelude
                                // Look up the function and call it
                                let func_id =
                                    *constructor.functions.get(core_fn_name).ok_or_else(|| {
                                        CompilerError::SpirvError(format!(
                                            "CoreFn not found: {}",
                                            core_fn_name
                                        ))
                                    })?;

                                Ok(constructor.builder.function_call(
                                    result_type,
                                    None,
                                    func_id,
                                    arg_ids,
                                )?)
                            }
                        }
                    } else {
                        // Look up user-defined function
                        let func_id = *constructor.functions.get(func).ok_or_else(|| {
                            CompilerError::SpirvError(format!("Unknown function: {}", func))
                        })?;
                        Ok(constructor.builder.function_call(result_type, None, func_id, arg_ids)?)
                    }
                }
            }
        }

        ExprKind::Intrinsic { name, args } => {
            // Get the result type from the expression
            let result_type = constructor.ast_type_to_spirv(&expr.ty);

            match name.as_str() {
                "tuple_access" => {
                    if args.len() != 2 {
                        bail_spirv!("tuple_access requires 2 args");
                    }
                    let composite_id = lower_expr(constructor, &args[0])?;
                    // Second arg should be a constant index - extract it from the literal
                    let index = match &args[1].kind {
                        ExprKind::Literal(Literal::Int(s)) => {
                            s.parse::<u32>().unwrap_or_else(|e| {
                                panic!("BUG: tuple_access index '{}' failed to parse as u32: {}. Type checking should ensure valid indices.", s, e)
                            })
                        }
                        _ => {
                            panic!("BUG: tuple_access requires a constant integer literal as second argument, got {:?}. Type checking should ensure this.", args[1].kind)
                        }
                    };

                    Ok(constructor.builder.composite_extract(result_type, None, composite_id, [index])?)
                }
                "index" => {
                    if args.len() != 2 {
                        bail_spirv!("index requires 2 args");
                    }
                    // Array indexing with OpAccessChain + OpLoad
                    let array_val = lower_expr(constructor, &args[0])?;
                    let index_val = lower_expr(constructor, &args[1])?;

                    // Store array in a variable to get a pointer
                    let array_type = constructor.ast_type_to_spirv(&args[0].ty);
                    let ptr_type =
                        constructor.builder.type_pointer(None, StorageClass::Function, array_type);
                    let array_var =
                        constructor.builder.variable(ptr_type, None, StorageClass::Function, None);
                    constructor.builder.store(array_var, array_val, None, [])?;

                    // Use OpAccessChain to get pointer to element
                    let elem_ptr_type =
                        constructor.builder.type_pointer(None, StorageClass::Function, result_type);
                    let elem_ptr =
                        constructor.builder.access_chain(elem_ptr_type, None, array_var, [index_val])?;

                    // Load the element
                    Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
                }
                "record_access" => {
                    // Record field access by name
                    // args[0] is the record, args[1] is a string literal with field name
                    if args.len() != 2 {
                        bail_spirv!("record_access requires 2 args");
                    }
                    let composite_id = lower_expr(constructor, &args[0])?;

                    // Get field name from string literal
                    let field_name = match &args[1].kind {
                        ExprKind::Literal(Literal::String(s)) => s.clone(),
                        _ => bail_spirv!("record_access field must be string literal"),
                    };

                    // Look up the field index from the record type, skipping phantom fields
                    let record_type = &args[0].ty;
                    let index = match record_type {
                        PolyType::Constructed(TypeName::Record(fields), _) => {
                            // Filter out phantom fields and find the index
                            let real_fields: Vec<_> =
                                fields.keys().filter(|name| name.as_str() != "__lambda_name").collect();
                            real_fields
                                .iter()
                                .enumerate()
                                .find(|(_, name)| name.as_str() == field_name)
                                .map(|(idx, _)| idx as u32)
                                .ok_or_else(|| {
                                    CompilerError::SpirvError(format!(
                                        "Unknown record field: {}",
                                        field_name
                                    ))
                                })?
                        }
                        _ => bail_spirv!("record_access on non-record type: {:?}", record_type),
                    };

                    Ok(constructor.builder.composite_extract(result_type, None, composite_id, [index])?)
                }
                "assert" => {
                    // Assertions are no-ops in release, return body
                    if args.len() >= 2 {
                        lower_expr(constructor, &args[1])
                    } else {
                        Ok(constructor.const_i32(0))
                    }
                }
                _ => Err(CompilerError::SpirvError(format!("Unknown intrinsic: {}", name))),
            }
        }

        ExprKind::Attributed { expr, .. } => {
            // Attributes are metadata, just lower the inner expression
            lower_expr(constructor, expr)
        }
    }
}

fn lower_literal(constructor: &mut Constructor, lit: &Literal) -> Result<spirv::Word> {
    match lit {
        Literal::Int(s) => {
            let value: i32 = s
                .parse()
                .map_err(|_| CompilerError::SpirvError(format!("Invalid integer literal: {}", s)))?;
            Ok(constructor.const_i32(value))
        }
        Literal::Float(s) => {
            let value: f32 = s
                .parse()
                .map_err(|_| CompilerError::SpirvError(format!("Invalid float literal: {}", s)))?;
            Ok(constructor.const_f32(value))
        }
        Literal::Bool(b) => Ok(constructor.const_bool(*b)),
        Literal::String(_) => Err(CompilerError::SpirvError(
            "String literals not supported in SPIR-V".to_string(),
        )),
        Literal::Tuple(elems) => {
            // Lower all elements
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|e| lower_expr(constructor, e)).collect::<Result<Vec<_>>>()?;

            // Create struct type for tuple from element types
            let elem_types: Vec<spirv::Word> =
                elems.iter().map(|e| constructor.ast_type_to_spirv(&e.ty)).collect();
            let tuple_type = constructor.builder.type_struct(elem_types);

            // Construct the composite
            Ok(constructor.builder.composite_construct(tuple_type, None, elem_ids)?)
        }
        Literal::Array(elems) => {
            // Lower all elements
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|e| lower_expr(constructor, e)).collect::<Result<Vec<_>>>()?;

            // Get element type from first element
            let elem_type = elems.first()
                .map(|e| constructor.ast_type_to_spirv(&e.ty))
                .unwrap_or_else(|| {
                    panic!("BUG: Empty array literal reached lowering. Type checking should require explicit type annotation for empty arrays or reject them entirely.")
                });

            // Create array type
            let array_type = constructor.type_array(elem_type, elem_ids.len() as u32);

            // Construct the composite

            Ok(constructor.builder.composite_construct(array_type, None, elem_ids)?)
        }
        Literal::Record(fields) => {
            // Filter out phantom fields that only exist for compile-time dispatch
            let real_fields: Vec<_> = fields.iter().filter(|(name, _)| name != "__lambda_name").collect();

            // Records are represented as structs with fields in order
            let field_ids: Vec<spirv::Word> =
                real_fields.iter().map(|(_, e)| lower_expr(constructor, e)).collect::<Result<Vec<_>>>()?;

            // Create struct type for record from field types
            let field_types: Vec<spirv::Word> =
                real_fields.iter().map(|(_, e)| constructor.ast_type_to_spirv(&e.ty)).collect();
            let record_type = constructor.builder.type_struct(field_types);

            // Construct the composite
            Ok(constructor.builder.composite_construct(record_type, None, field_ids)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flattening::Flattener;
    use crate::lexer::tokenize;
    use crate::parser::Parser;
    use crate::type_checker::TypeChecker;

    fn compile_to_spirv(source: &str) -> Result<Vec<u32>> {
        // Use the typestate API to ensure proper compilation pipeline
        let mir = crate::Compiler::parse(source)
            .expect("Parsing failed")
            .elaborate()
            .expect("Elaboration failed")
            .resolve()
            .expect("Name resolution failed")
            .type_check()
            .expect("Type checking failed")
            .flatten()
            .expect("Flattening failed")
            .mir;

        lower(&mir)
    }

    #[test]
    fn test_simple_constant() {
        let spirv = compile_to_spirv("def x = 42").unwrap();
        assert!(!spirv.is_empty());
        // SPIR-V magic number
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_simple_function() {
        let spirv = compile_to_spirv("def add x y = x + y").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_let_binding() {
        let spirv = compile_to_spirv("def f = let x = 1 in x + 2").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_arithmetic() {
        let spirv = compile_to_spirv("def f x y = x * y + x / y - 1").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_nested_let() {
        let spirv = compile_to_spirv("def f = let a = 1 in let b = 2 in a + b").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_if_expression() {
        let spirv = compile_to_spirv("def f x = if x == 0 then 1 else 2").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_comparisons() {
        let spirv = compile_to_spirv("def f x y = if x < y then 1 else if x > y then 2 else 0").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_tuple_literal() {
        let spirv = compile_to_spirv("def f = (1, 2, 3)").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_array_literal() {
        let spirv = compile_to_spirv("def f = [1, 2, 3]").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_unary_negation() {
        let spirv = compile_to_spirv("def f x = -x").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_record_field_access() {
        let spirv = compile_to_spirv(
            r#"
def get_x (r:{x:i32, y:i32}) : i32 = r.x
"#,
        )
        .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_closure_capture_access() {
        // This test requires record_access intrinsic for closure field access
        let spirv = compile_to_spirv(
            r#"
def test (x:i32) : i32 =
    let f = \(y:i32) -> x + y in
    f 10
"#,
        )
        .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }
}
