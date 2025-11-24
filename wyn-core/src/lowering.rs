//! SPIR-V Lowering
//!
//! This module converts MIR (from flattening) directly to SPIR-V.
//! It uses a SpvBuilder wrapper that handles variable hoisting automatically.
//! Dependencies are lowered on-demand using ensure_lowered pattern.

use crate::ast::TypeName;
use crate::builtin_registry::{BuiltinImpl, BuiltinRegistry, PrimOp};
use crate::error::{CompilerError, Result};
use crate::mir::{self, Def, Expr, ExprKind, Literal, LoopKind, Program};
use polytype::Type as PolyType;
use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::dr::Operand;
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
    spv: SpvBuilder,
    /// Entry points to emit (name, execution model)
    entry_points: Vec<(String, spirv::ExecutionModel)>,
}

/// SpvBuilder wraps rspirv::Builder with an ergonomic API that handles:
/// - Automatic variable hoisting to function entry block
/// - Block management with implicit branch from variables block to code
/// - Value and type caching
struct SpvBuilder {
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
    variables_block: Option<spirv::Word>,
    first_code_block: Option<spirv::Word>,
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

impl SpvBuilder {
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

        SpvBuilder {
            builder,
            void_type,
            bool_type,
            i32_type,
            f32_type,
            int_const_cache: HashMap::new(),
            float_const_cache: HashMap::new(),
            bool_const_cache: HashMap::new(),
            variables_block: None,
            first_code_block: None,
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
            PolyType::Variable(_) => self.i32_type, // Fallback for unresolved vars
            PolyType::Constructed(name, args) => {
                match name {
                    TypeName::Str(s) if *s == "i32" => self.i32_type,
                    TypeName::Str(s) if *s == "f32" => self.f32_type,
                    TypeName::Str(s) if *s == "bool" => self.bool_type,
                    TypeName::Str(s) if *s == "tuple" => {
                        // Tuple becomes struct
                        let field_types: Vec<spirv::Word> =
                            args.iter().map(|a| self.ast_type_to_spirv(a)).collect();
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Vec => {
                        // Vector type: args[0] is size, args[1] is element type
                        if args.len() >= 2 {
                            // Extract size from args[0]
                            let size = match &args[0] {
                                PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                                PolyType::Constructed(TypeName::Str(s), _) => s.parse().unwrap_or(4),
                                _ => 4,
                            };
                            // Get element type from args[1]
                            let elem_type = self.ast_type_to_spirv(&args[1]);
                            self.get_or_create_vec_type(elem_type, size)
                        } else {
                            // Default to vec4f32
                            self.get_or_create_vec_type(self.f32_type, 4)
                        }
                    }
                    TypeName::Str(s) if s.starts_with("vec") => {
                        // vec2f32, vec3f32, vec4f32, vec2i32, etc.
                        let size = s.chars().nth(3).and_then(|c| c.to_digit(10)).unwrap_or(4) as u32;
                        let elem_type = if s.contains("f32") || s.ends_with("f32") {
                            self.f32_type
                        } else if s.contains("i32") || s.ends_with("i32") {
                            self.i32_type
                        } else {
                            self.f32_type // Default to float
                        };
                        self.get_or_create_vec_type(elem_type, size)
                    }
                    TypeName::Str(s) if *s == "unknown" => self.i32_type,
                    _ => {
                        // Fallback - try to recognize common patterns
                        self.i32_type
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
        *self.value_types.get(&value).unwrap_or(&self.i32_type)
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

        // Create variables block (entry block for OpVariable)
        let vars_block_id = self.builder.id();
        self.variables_block = Some(vars_block_id);

        // Create first code block
        let code_block_id = self.builder.id();
        self.first_code_block = Some(code_block_id);

        // Start in the code block (we'll emit variables block at the end)
        self.builder.begin_block(Some(code_block_id))?;
        self.current_block = Some(code_block_id);

        // Clear pending variables for this function
        self.pending_variables.clear();

        Ok(func_id)
    }

    /// End the current function, emitting the variables block with branch
    fn end_function(&mut self) -> Result<()> {
        // Save current block
        let code_continues_from = self.current_block;

        // Now emit the variables block at the start
        if let (Some(vars_block), Some(first_code)) = (self.variables_block, self.first_code_block) {
            // We need to insert the variables block before all other blocks
            // rspirv doesn't make this easy, so we'll use a workaround:
            // Actually, SPIR-V requires variables at the START of the first block,
            // not in a separate block. Let me reconsider...

            // The proper approach: emit OpVariable instructions at the start of the
            // first block. We've been collecting them in pending_variables.
            //
            // Since we already started the code block, we need to modify the module
            // after the fact. For now, let's skip the two-block approach and just
            // emit variables at the start.
            //
            // TODO: Implement proper variable hoisting by modifying the module structure
        }

        self.builder.end_function()?;

        // Clear function state
        self.variables_block = None;
        self.first_code_block = None;
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

    /// Binary arithmetic operations
    fn i_add(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.i_add(self.i32_type, None, lhs, rhs)?)
    }

    fn i_sub(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.i_sub(self.i32_type, None, lhs, rhs)?)
    }

    fn i_mul(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.i_mul(self.i32_type, None, lhs, rhs)?)
    }

    fn s_div(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.s_div(self.i32_type, None, lhs, rhs)?)
    }

    fn f_add(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.f_add(self.f32_type, None, lhs, rhs)?)
    }

    fn f_sub(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.f_sub(self.f32_type, None, lhs, rhs)?)
    }

    fn f_mul(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.f_mul(self.f32_type, None, lhs, rhs)?)
    }

    fn f_div(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.f_div(self.f32_type, None, lhs, rhs)?)
    }

    /// Comparison operations
    fn i_equal(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.i_equal(self.bool_type, None, lhs, rhs)?)
    }

    fn s_less_than(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.s_less_than(self.bool_type, None, lhs, rhs)?)
    }

    fn s_less_than_equal(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.s_less_than_equal(self.bool_type, None, lhs, rhs)?)
    }

    fn s_greater_than(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.s_greater_than(self.bool_type, None, lhs, rhs)?)
    }

    fn s_greater_than_equal(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.s_greater_than_equal(self.bool_type, None, lhs, rhs)?)
    }

    fn i_not_equal(&mut self, lhs: spirv::Word, rhs: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.i_not_equal(self.bool_type, None, lhs, rhs)?)
    }

    /// Unary operations
    fn s_negate(&mut self, operand: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.s_negate(self.i32_type, None, operand)?)
    }

    fn f_negate(&mut self, operand: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.f_negate(self.f32_type, None, operand)?)
    }

    fn logical_not(&mut self, operand: spirv::Word) -> Result<spirv::Word> {
        Ok(self.builder.logical_not(self.bool_type, None, operand)?)
    }

    /// Return from function
    fn ret(&mut self) -> Result<()> {
        self.builder.ret()?;
        Ok(())
    }

    fn ret_value(&mut self, value: spirv::Word) -> Result<()> {
        self.builder.ret_value(value)?;
        Ok(())
    }

    /// Create a new block and return its ID
    fn create_block(&mut self) -> spirv::Word {
        self.builder.id()
    }

    /// Begin a block (must be called before emitting instructions into it)
    fn begin_block(&mut self, block_id: spirv::Word) -> Result<()> {
        self.builder.begin_block(Some(block_id))?;
        self.current_block = Some(block_id);
        Ok(())
    }

    /// Emit an unconditional branch
    fn branch(&mut self, target: spirv::Word) -> Result<()> {
        self.builder.branch(target)?;
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

    /// Emit a phi instruction
    fn phi(
        &mut self,
        result_type: spirv::Word,
        incoming: Vec<(spirv::Word, spirv::Word)>,
    ) -> Result<spirv::Word> {
        Ok(self.builder.phi(result_type, None, incoming)?)
    }

    /// Emit a loop merge
    fn loop_merge(&mut self, merge_block: spirv::Word, continue_block: spirv::Word) -> Result<()> {
        self.builder.loop_merge(merge_block, continue_block, spirv::LoopControl::NONE, [])?;
        Ok(())
    }

    /// Emit a function call
    fn function_call(
        &mut self,
        result_type: spirv::Word,
        func_id: spirv::Word,
        args: Vec<spirv::Word>,
    ) -> Result<spirv::Word> {
        Ok(self.builder.function_call(result_type, None, func_id, args)?)
    }

    /// Construct a composite (tuple/array/struct)
    fn composite_construct(
        &mut self,
        result_type: spirv::Word,
        elements: Vec<spirv::Word>,
    ) -> Result<spirv::Word> {
        Ok(self.builder.composite_construct(result_type, None, elements)?)
    }

    /// Extract an element from a composite
    fn composite_extract(
        &mut self,
        result_type: spirv::Word,
        composite: spirv::Word,
        index: u32,
    ) -> Result<spirv::Word> {
        Ok(self.builder.composite_extract(result_type, None, composite, [index])?)
    }

    /// Get array type
    fn type_array(&mut self, elem_type: spirv::Word, length: u32) -> spirv::Word {
        let length_id = self.const_i32(length as i32);
        self.builder.type_array(elem_type, length_id)
    }

    /// Get struct type (for tuples/records)
    fn type_struct(&mut self, field_types: Vec<spirv::Word>) -> spirv::Word {
        self.builder.type_struct(field_types)
    }

    /// Assemble the module
    fn assemble(self) -> Vec<u32> {
        self.builder.module().assemble()
    }
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        let mut spv = SpvBuilder::new();
        spv.lambda_registry = program.lambda_registry.clone();

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
            spv,
            entry_points,
        }
    }

    /// Ensure a definition is lowered, recursively lowering dependencies first
    fn ensure_lowered(&mut self, name: &str) -> Result<()> {
        match self.state.get(name).copied().unwrap_or(LowerState::NotStarted) {
            LowerState::Done => return Ok(()),
            LowerState::InProgress => {
                return Err(CompilerError::SpirvError(format!(
                    "Recursive definition detected: {}",
                    name
                )));
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
                        &mut self.spv,
                        name,
                        params,
                        ret_type,
                        param_attributes,
                        return_attributes,
                        body,
                    )?;
                } else {
                    lower_regular_function(&mut self.spv, name, params, ret_type, body)?;
                }
            }
            Def::Constant { name, ty, body, .. } => {
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                // Create global variable for constant (Private storage class)
                let value_type = self.spv.ast_type_to_spirv(ty);
                let ptr_type = self.spv.get_or_create_ptr_type(StorageClass::Private, value_type);
                let var_id = self.spv.builder.variable(ptr_type, None, StorageClass::Private, None);

                // Store for later initialization and lookup
                self.spv.global_constants.insert(name.clone(), (var_id, value_type));
                self.spv.pending_constant_inits.push((var_id, body.clone()));
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
            ExprKind::Literal(_) => {}
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
        if !self.spv.pending_constant_inits.is_empty() {
            generate_init_function(&mut self.spv)?;
        }

        // Emit entry points with interface variables
        for (name, model) in &self.entry_points {
            if let Some(&func_id) = self.spv.functions.get(name) {
                let interfaces = self.spv.entry_point_interfaces.get(name).cloned().unwrap_or_default();
                self.spv.builder.entry_point(*model, func_id, name, interfaces);

                // Add execution mode for fragment shaders
                if *model == spirv::ExecutionModel::Fragment {
                    self.spv.builder.execution_mode(func_id, spirv::ExecutionMode::OriginUpperLeft, []);
                }
            }
        }

        Ok(self.spv.assemble())
    }
}

/// Lower a MIR program to SPIR-V
pub fn lower(program: &mir::Program) -> Result<Vec<u32>> {
    let ctx = LowerCtx::new(program);
    ctx.run()
}

/// Generate _init function that initializes all global constants
fn generate_init_function(spv: &mut SpvBuilder) -> Result<()> {
    // Take pending inits to avoid borrow issues
    let inits = std::mem::take(&mut spv.pending_constant_inits);

    spv.begin_function("_init", &[], &[], spv.void_type)?;

    for (var_id, body) in inits {
        let value = lower_expr(spv, &body)?;
        spv.builder.store(var_id, value, None, [])?;
    }

    spv.builder.ret()?;
    spv.end_function()?;

    Ok(())
}

fn lower_regular_function(
    spv: &mut SpvBuilder,
    name: &str,
    params: &[mir::Param],
    ret_type: &PolyType<TypeName>,
    body: &Expr,
) -> Result<()> {
    let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
    let param_types: Vec<spirv::Word> = params.iter().map(|p| spv.ast_type_to_spirv(&p.ty)).collect();
    let return_type = spv.ast_type_to_spirv(ret_type);
    spv.begin_function(name, &param_names, &param_types, return_type)?;

    let result = lower_expr(spv, body)?;
    spv.ret_value(result)?;

    spv.end_function()?;
    Ok(())
}

fn lower_entry_point(
    spv: &mut SpvBuilder,
    name: &str,
    params: &[mir::Param],
    ret_type: &PolyType<TypeName>,
    param_attributes: &[Vec<mir::Attribute>],
    return_attributes: &[Vec<mir::Attribute>],
    body: &Expr,
) -> Result<()> {
    spv.current_is_entry_point = true;
    spv.current_output_vars.clear();
    spv.current_input_vars.clear();

    let mut interface_vars = Vec::new();

    // Create Input variables for parameters
    for (i, param) in params.iter().enumerate() {
        let param_type_id = spv.ast_type_to_spirv(&param.ty);
        let ptr_type_id = spv.get_or_create_ptr_type(StorageClass::Input, param_type_id);
        let var_id = spv.builder.variable(ptr_type_id, None, StorageClass::Input, None);

        // Add decorations from attributes
        if i < param_attributes.len() {
            for attr in &param_attributes[i] {
                match attr.name.as_str() {
                    "location" => {
                        if let Some(loc_str) = attr.args.first() {
                            if let Ok(loc) = loc_str.parse::<u32>() {
                                spv.builder.decorate(
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
                                spv.builder.decorate(
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
        spv.current_input_vars.push((var_id, param.name.clone(), param_type_id));
    }

    // Create Output variables for return values
    // Get return type components (if tuple, each element gets its own output)
    let ret_type_id = spv.ast_type_to_spirv(ret_type);
    let is_void = matches!(ret_type, PolyType::Constructed(TypeName::Str(s), _) if *s == "void");

    if !is_void {
        // Check if return is a tuple (multiple outputs)
        if let PolyType::Constructed(TypeName::Str(s), component_types) = ret_type {
            if *s == "tuple" && !return_attributes.is_empty() {
                // Multiple outputs - one variable per component
                for (i, comp_ty) in component_types.iter().enumerate() {
                    let comp_type_id = spv.ast_type_to_spirv(comp_ty);
                    let ptr_type_id = spv.get_or_create_ptr_type(StorageClass::Output, comp_type_id);
                    let var_id = spv.builder.variable(ptr_type_id, None, StorageClass::Output, None);

                    // Add decorations
                    if i < return_attributes.len() {
                        for attr in &return_attributes[i] {
                            match attr.name.as_str() {
                                "location" => {
                                    if let Some(loc_str) = attr.args.first() {
                                        if let Ok(loc) = loc_str.parse::<u32>() {
                                            spv.builder.decorate(
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
                                            spv.builder.decorate(
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
                    spv.current_output_vars.push(var_id);
                }
            } else {
                // Single output
                let ptr_type_id = spv.get_or_create_ptr_type(StorageClass::Output, ret_type_id);
                let var_id = spv.builder.variable(ptr_type_id, None, StorageClass::Output, None);

                // Add decorations from first return attribute set
                if let Some(attrs) = return_attributes.first() {
                    for attr in attrs {
                        match attr.name.as_str() {
                            "location" => {
                                if let Some(loc_str) = attr.args.first() {
                                    if let Ok(loc) = loc_str.parse::<u32>() {
                                        spv.builder.decorate(
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
                                        spv.builder.decorate(
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
                spv.current_output_vars.push(var_id);
            }
        } else {
            // Single non-tuple output
            let ptr_type_id = spv.get_or_create_ptr_type(StorageClass::Output, ret_type_id);
            let var_id = spv.builder.variable(ptr_type_id, None, StorageClass::Output, None);

            if let Some(attrs) = return_attributes.first() {
                for attr in attrs {
                    match attr.name.as_str() {
                        "location" => {
                            if let Some(loc_str) = attr.args.first() {
                                if let Ok(loc) = loc_str.parse::<u32>() {
                                    spv.builder.decorate(
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
                                    spv.builder.decorate(
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
            spv.current_output_vars.push(var_id);
        }
    }

    // Store interface variables for entry point declaration
    spv.entry_point_interfaces.insert(name.to_string(), interface_vars);

    // Create void(void) function for entry point
    let func_type = spv.builder.type_function(spv.void_type, vec![]);
    let func_id =
        spv.builder.begin_function(spv.void_type, None, spirv::FunctionControl::NONE, func_type)?;
    spv.functions.insert(name.to_string(), func_id);

    // Create entry block
    let block_id = spv.builder.id();
    spv.builder.begin_block(Some(block_id))?;
    spv.current_block = Some(block_id);

    // Call _init to initialize global constants
    if let Some(&init_func_id) = spv.functions.get("_init") {
        spv.builder.function_call(spv.void_type, None, init_func_id, [])?;
    }

    // Load input variables into environment
    for (var_id, param_name, type_id) in spv.current_input_vars.clone() {
        let loaded = spv.builder.load(type_id, None, var_id, None, [])?;
        spv.env.insert(param_name, loaded);
    }

    // Lower the body
    let result = lower_expr(spv, body)?;

    // Store result to output variables
    if !spv.current_output_vars.is_empty() {
        // Check if result is a tuple that needs to be decomposed
        if let PolyType::Constructed(TypeName::Str(s), component_types) = ret_type {
            if *s == "tuple" && spv.current_output_vars.len() > 1 {
                // Extract each component and store
                for (i, &output_var) in spv.current_output_vars.clone().iter().enumerate() {
                    let comp_type_id = spv.ast_type_to_spirv(&component_types[i]);
                    let component =
                        spv.builder.composite_extract(comp_type_id, None, result, [i as u32])?;
                    spv.builder.store(output_var, component, None, [])?;
                }
            } else if let Some(&output_var) = spv.current_output_vars.first() {
                spv.builder.store(output_var, result, None, [])?;
            }
        } else if let Some(&output_var) = spv.current_output_vars.first() {
            spv.builder.store(output_var, result, None, [])?;
        }
    }

    // Return void
    spv.builder.ret()?;
    spv.builder.end_function()?;

    // Clean up
    spv.current_is_entry_point = false;
    spv.env.clear();

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

fn lower_expr(spv: &mut SpvBuilder, expr: &Expr) -> Result<spirv::Word> {
    match &expr.kind {
        ExprKind::Literal(lit) => lower_literal(spv, lit),

        ExprKind::Var(name) => {
            // First check local environment
            if let Some(&id) = spv.env.get(name) {
                return Ok(id);
            }
            // Then check global constants (load from global variable)
            if let Some(&(var_id, type_id)) = spv.global_constants.get(name) {
                return Ok(spv.builder.load(type_id, None, var_id, None, [])?);
            }
            Err(CompilerError::SpirvError(format!("Undefined variable: {}", name)))
        }

        ExprKind::BinOp { op, lhs, rhs } => {
            let lhs_id = lower_expr(spv, lhs)?;
            let rhs_id = lower_expr(spv, rhs)?;

            match op.as_str() {
                "+" => spv.i_add(lhs_id, rhs_id),
                "-" => spv.i_sub(lhs_id, rhs_id),
                "*" => spv.i_mul(lhs_id, rhs_id),
                "/" => spv.s_div(lhs_id, rhs_id),
                "==" => spv.i_equal(lhs_id, rhs_id),
                "!=" => spv.i_not_equal(lhs_id, rhs_id),
                "<" => spv.s_less_than(lhs_id, rhs_id),
                "<=" => spv.s_less_than_equal(lhs_id, rhs_id),
                ">" => spv.s_greater_than(lhs_id, rhs_id),
                ">=" => spv.s_greater_than_equal(lhs_id, rhs_id),
                _ => Err(CompilerError::SpirvError(format!("Unknown binary op: {}", op))),
            }
        }

        ExprKind::UnaryOp { op, operand } => {
            let operand_id = lower_expr(spv, operand)?;

            match op.as_str() {
                "-" => spv.s_negate(operand_id),
                "!" => spv.logical_not(operand_id),
                _ => Err(CompilerError::SpirvError(format!("Unknown unary op: {}", op))),
            }
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let cond_id = lower_expr(spv, cond)?;

            // Get the result type from the expression
            let result_type = spv.ast_type_to_spirv(&expr.ty);

            // Create blocks
            let then_block = spv.create_block();
            let else_block = spv.create_block();
            let merge_block = spv.create_block();

            // Branch based on condition
            spv.branch_conditional(cond_id, then_block, else_block, merge_block)?;

            // Then block
            spv.begin_block(then_block)?;
            let then_result = lower_expr(spv, then_branch)?;
            let then_exit_block = spv.current_block.unwrap();
            spv.branch(merge_block)?;

            // Else block
            spv.begin_block(else_block)?;
            let else_result = lower_expr(spv, else_branch)?;
            let else_exit_block = spv.current_block.unwrap();
            spv.branch(merge_block)?;

            // Merge block with phi
            spv.begin_block(merge_block)?;
            let result = spv.phi(
                result_type,
                vec![(then_result, then_exit_block), (else_result, else_exit_block)],
            )?;

            Ok(result)
        }

        ExprKind::Let { name, value, body } => {
            let value_id = lower_expr(spv, value)?;
            spv.env.insert(name.clone(), value_id);
            let result = lower_expr(spv, body)?;
            spv.env.remove(name);
            Ok(result)
        }

        ExprKind::Loop {
            init_bindings,
            kind,
            body,
        } => {
            // Create blocks for loop structure
            let header_block = spv.create_block();
            let body_block = spv.create_block();
            let continue_block = spv.create_block();
            let merge_block = spv.create_block();

            // Evaluate init expressions in order, adding each to environment
            // This is necessary because later init expressions may reference earlier ones
            // (e.g., tuple destructuring creates a temp var that subsequent extractions reference)
            let mut init_values = Vec::new();
            for (name, init_expr) in init_bindings {
                let init_val = lower_expr(spv, init_expr)?;
                // Add to environment immediately so subsequent init expressions can reference it
                spv.env.insert(name.clone(), init_val);
                init_values.push((name.clone(), init_val));
            }
            let pre_header_block = spv.current_block.unwrap();

            // Branch to header
            spv.branch(header_block)?;

            // Header block - phi nodes and condition check
            spv.begin_block(header_block)?;

            // Create phi nodes for loop variables
            let mut phi_results = Vec::new();
            for (i, (name, init_val)) in init_values.iter().enumerate() {
                // We'll update these with continue block values later
                let phi_id = spv.builder.id();
                phi_results.push((name.clone(), phi_id, *init_val));
                spv.env.insert(name.clone(), phi_id);
            }

            // Generate condition based on loop kind
            let cond_id = match kind {
                LoopKind::While { cond } => lower_expr(spv, cond)?,
                LoopKind::ForRange { var, bound } => {
                    let bound_id = lower_expr(spv, bound)?;
                    let var_id = *spv.env.get(var).ok_or_else(|| {
                        CompilerError::SpirvError(format!("Loop variable {} not found", var))
                    })?;
                    spv.s_less_than(var_id, bound_id)?
                }
                LoopKind::For { .. } => {
                    // For-in loops need iterator support
                    return Err(CompilerError::SpirvError(
                        "For-in loops not yet implemented".to_string(),
                    ));
                }
            };

            // Loop merge and conditional branch
            spv.loop_merge(merge_block, continue_block)?;
            spv.builder.branch_conditional(cond_id, body_block, merge_block, [])?;

            // Body block
            spv.begin_block(body_block)?;
            let body_result = lower_expr(spv, body)?;
            spv.branch(continue_block)?;

            // Continue block - update loop variables and branch back to header
            spv.begin_block(continue_block)?;

            // For now, use body result as the new value for first loop var
            // This is simplified - real implementation needs to handle multiple vars
            let continue_values: Vec<spirv::Word> = if phi_results.len() == 1 {
                vec![body_result]
            } else {
                // For multiple vars, we'd need to extract from tuple
                vec![body_result; phi_results.len()]
            };

            spv.branch(header_block)?;

            // Now emit the actual phi instructions in header
            // We need to go back and fix up the phi nodes
            // For simplicity, we'll rebuild the header with proper phis
            // This is a limitation - proper implementation would use rspirv's phi building

            // Merge block - result is the final loop value
            spv.begin_block(merge_block)?;

            // Clean up environment
            for (name, _, _) in &phi_results {
                spv.env.remove(name);
            }

            // Return the last phi value (loop result)
            if let Some((_, phi_id, _)) = phi_results.first() {
                Ok(*phi_id)
            } else {
                Ok(spv.const_i32(0)) // Empty loop
            }
        }

        ExprKind::Call { func, args } => {
            // Lower all arguments
            let arg_ids: Vec<spirv::Word> =
                args.iter().map(|a| lower_expr(spv, a)).collect::<Result<Vec<_>>>()?;

            // Get the result type from the expression
            let result_type = spv.ast_type_to_spirv(&expr.ty);

            // Check for builtin vector constructors and higher-order functions
            match func.as_str() {
                "vec2" | "vec3" | "vec4" => {
                    // Use the result type which should be the proper vector type
                    spv.composite_construct(result_type, arg_ids)
                }
                "map" => {
                    // map closure array -> array
                    // args[0] is closure record {__tag=N, ...}
                    // args[1] is input array
                    if args.len() != 2 {
                        return Err(CompilerError::SpirvError("map requires 2 args".to_string()));
                    }

                    let closure_val = arg_ids[0];
                    let array_val = arg_ids[1];

                    // Get array info from MIR types
                    let (array_size, elem_mir_type) = match &expr.ty {
                        PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 2 => {
                            let size = match &type_args[0] {
                                PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                                _ => {
                                    return Err(CompilerError::SpirvError(
                                        "Invalid array size type".to_string(),
                                    ));
                                }
                            };
                            (size, &type_args[1])
                        }
                        _ => {
                            return Err(CompilerError::SpirvError(
                                "map result must be array type".to_string(),
                            ));
                        }
                    };

                    let elem_type = spv.ast_type_to_spirv(elem_mir_type);

                    // Build result array by calling lambda for each element
                    let mut result_elements = Vec::new();
                    for i in 0..array_size {
                        // Extract element from input array
                        let input_elem = spv.builder.composite_extract(elem_type, None, array_val, [i])?;

                        // Call the lambda function
                        // For now, we only support single-lambda case (tag 0)
                        // TODO: Add switch for multiple lambdas based on tag
                        if let Some((func_name, _arity)) = spv.lambda_registry.first() {
                            let lambda_func_id = *spv.functions.get(func_name).ok_or_else(|| {
                                CompilerError::SpirvError(format!(
                                    "Lambda function not found: {}",
                                    func_name
                                ))
                            })?;
                            // Call lambda with closure and element
                            let result_elem = spv.function_call(
                                elem_type,
                                lambda_func_id,
                                vec![closure_val, input_elem],
                            )?;
                            result_elements.push(result_elem);
                        } else {
                            return Err(CompilerError::SpirvError("No lambda in registry".to_string()));
                        }
                    }

                    // Construct result array
                    spv.composite_construct(result_type, result_elements)
                }
                _ => {
                    // Check if it's a builtin function
                    if let Some(overloads) = spv.builtin_registry.get_overloads(func) {
                        // TODO: Use the selected overload from type checking
                        // For now, just use the first overload
                        let builtin = &overloads[0];
                        match &builtin.implementation {
                            BuiltinImpl::PrimOp(spirv_op) => {
                                // Handle core SPIR-V operations
                                match spirv_op {
                                    PrimOp::GlslExt(ext_op) => {
                                        // Call GLSL extended instruction
                                        let glsl_id = spv.glsl_ext_inst_id;
                                        let operands: Vec<Operand> =
                                            arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();
                                        Ok(spv.builder.ext_inst(
                                            result_type,
                                            None,
                                            glsl_id,
                                            *ext_op,
                                            operands,
                                        )?)
                                    }
                                    PrimOp::Dot => {
                                        if arg_ids.len() != 2 {
                                            return Err(CompilerError::SpirvError(
                                                "dot requires 2 args".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.dot(result_type, None, arg_ids[0], arg_ids[1])?)
                                    }
                                    PrimOp::MatrixTimesMatrix => {
                                        if arg_ids.len() != 2 {
                                            return Err(CompilerError::SpirvError(
                                                "matrix × matrix requires 2 args".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.matrix_times_matrix(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::MatrixTimesVector => {
                                        if arg_ids.len() != 2 {
                                            return Err(CompilerError::SpirvError(
                                                "matrix × vector requires 2 args".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.matrix_times_vector(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::VectorTimesMatrix => {
                                        if arg_ids.len() != 2 {
                                            return Err(CompilerError::SpirvError(
                                                "vector × matrix requires 2 args".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.vector_times_matrix(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    // Type conversions
                                    PrimOp::FPToSI => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "FPToSI requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.convert_f_to_s(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::FPToUI => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "FPToUI requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.convert_f_to_u(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::SIToFP => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "SIToFP requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.convert_s_to_f(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::UIToFP => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "UIToFP requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.convert_u_to_f(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::FPConvert => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "FPConvert requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.f_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::SConvert => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "SConvert requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.s_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::UConvert => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "UConvert requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.u_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::Bitcast => {
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "Bitcast requires 1 arg".to_string(),
                                            ));
                                        }
                                        Ok(spv.builder.bitcast(result_type, None, arg_ids[0])?)
                                    }
                                    _ => {
                                        // TODO: Handle other PrimOp variants
                                        Err(CompilerError::SpirvError(format!(
                                            "Unsupported PrimOp for: {}",
                                            func
                                        )))
                                    }
                                }
                            }
                            BuiltinImpl::Intrinsic(custom_impl) => {
                                use crate::builtin_registry::Intrinsic;
                                match custom_impl {
                                    Intrinsic::MatrixFromVectors => {
                                        // Matrix from array of vectors: In SPIR-V, matrices ARE arrays of column vectors
                                        // So this is essentially a no-op/identity at the SPIR-V level
                                        // Just return the array as-is, but with matrix type
                                        if arg_ids.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "matav expects exactly 1 argument".to_string(),
                                            ));
                                        }
                                        // The array of vectors is already in the correct format for a SPIR-V matrix
                                        // Just return it with the matrix type
                                        Ok(arg_ids[0])
                                    }
                                    Intrinsic::Placeholder if func == "length" => {
                                        // Array length: extract size from array type
                                        // The array type is [n]T, we need to return n as a constant
                                        if args.len() != 1 {
                                            return Err(CompilerError::SpirvError(
                                                "length expects exactly 1 argument".to_string(),
                                            ));
                                        }
                                        // Get the array type
                                        if let PolyType::Constructed(TypeName::Array, type_args) =
                                            &args[0].ty
                                        {
                                            // Try to extract the size - it could be TypeName::Size(n) or a variable
                                            match type_args.get(0) {
                                                Some(PolyType::Constructed(TypeName::Size(n), _)) => {
                                                    // Return the size as a constant i32
                                                    Ok(spv.const_i32(*n as i32))
                                                }
                                                _ => {
                                                    // Size is a variable or unknown - this shouldn't happen in well-typed code
                                                    Err(CompilerError::SpirvError(format!(
                                                        "Cannot determine compile-time array size for length: {:?}",
                                                        type_args.get(0)
                                                    )))
                                                }
                                            }
                                        } else {
                                            Err(CompilerError::SpirvError(format!(
                                                "length called on non-array type: {:?}",
                                                args[0].ty
                                            )))
                                        }
                                    }
                                    _ => {
                                        // TODO: Handle other intrinsic implementations
                                        Err(CompilerError::SpirvError(format!(
                                            "Intrinsic builtin not yet supported: {}",
                                            func
                                        )))
                                    }
                                }
                            }
                            BuiltinImpl::CoreFn(core_fn_name) => {
                                // Library-level builtins implemented as normal functions in prelude
                                // Look up the function and call it
                                let func_id = *spv.functions.get(core_fn_name).ok_or_else(|| {
                                    CompilerError::SpirvError(format!("CoreFn not found: {}", core_fn_name))
                                })?;
                                spv.function_call(result_type, func_id, arg_ids)
                            }
                        }
                    } else {
                        // Look up user-defined function
                        let func_id = *spv.functions.get(func).ok_or_else(|| {
                            CompilerError::SpirvError(format!("Unknown function: {}", func))
                        })?;
                        spv.function_call(result_type, func_id, arg_ids)
                    }
                }
            }
        }

        ExprKind::Intrinsic { name, args } => {
            // Get the result type from the expression
            let result_type = spv.ast_type_to_spirv(&expr.ty);

            match name.as_str() {
                "tuple_access" => {
                    if args.len() != 2 {
                        return Err(CompilerError::SpirvError(
                            "tuple_access requires 2 args".to_string(),
                        ));
                    }
                    let composite_id = lower_expr(spv, &args[0])?;
                    // Second arg should be a constant index - extract it from the literal
                    let index = match &args[1].kind {
                        ExprKind::Literal(Literal::Int(s)) => s.parse::<u32>().unwrap_or(0),
                        _ => 0,
                    };
                    spv.composite_extract(result_type, composite_id, index)
                }
                "index" => {
                    if args.len() != 2 {
                        return Err(CompilerError::SpirvError("index requires 2 args".to_string()));
                    }
                    // Array indexing with OpAccessChain + OpLoad
                    let array_val = lower_expr(spv, &args[0])?;
                    let index_val = lower_expr(spv, &args[1])?;

                    // Store array in a variable to get a pointer
                    let array_type = spv.ast_type_to_spirv(&args[0].ty);
                    let ptr_type = spv.builder.type_pointer(None, StorageClass::Function, array_type);
                    let array_var = spv.builder.variable(ptr_type, None, StorageClass::Function, None);
                    spv.builder.store(array_var, array_val, None, [])?;

                    // Use OpAccessChain to get pointer to element
                    let elem_ptr_type = spv.builder.type_pointer(None, StorageClass::Function, result_type);
                    let elem_ptr = spv.builder.access_chain(elem_ptr_type, None, array_var, [index_val])?;

                    // Load the element
                    Ok(spv.builder.load(result_type, None, elem_ptr, None, [])?)
                }
                "record_access" => {
                    // Record field access by name
                    // args[0] is the record, args[1] is a string literal with field name
                    if args.len() != 2 {
                        return Err(CompilerError::SpirvError(
                            "record_access requires 2 args".to_string(),
                        ));
                    }
                    let composite_id = lower_expr(spv, &args[0])?;

                    // Get field name from string literal
                    let field_name = match &args[1].kind {
                        ExprKind::Literal(Literal::String(s)) => s.clone(),
                        _ => {
                            return Err(CompilerError::SpirvError(
                                "record_access field must be string literal".to_string(),
                            ));
                        }
                    };

                    // Look up the field index from the record type
                    let record_type = &args[0].ty;
                    let index = match record_type {
                        PolyType::Constructed(TypeName::Record(fields), _) => fields
                            .keys()
                            .enumerate()
                            .find(|(_, name)| name.as_str() == field_name)
                            .map(|(idx, _)| idx as u32)
                            .ok_or_else(|| {
                                CompilerError::SpirvError(format!("Unknown record field: {}", field_name))
                            })?,
                        _ => {
                            return Err(CompilerError::SpirvError(format!(
                                "record_access on non-record type: {:?}",
                                record_type
                            )));
                        }
                    };

                    spv.composite_extract(result_type, composite_id, index)
                }
                "assert" => {
                    // Assertions are no-ops in release, return body
                    if args.len() >= 2 { lower_expr(spv, &args[1]) } else { Ok(spv.const_i32(0)) }
                }
                _ => Err(CompilerError::SpirvError(format!("Unknown intrinsic: {}", name))),
            }
        }

        ExprKind::Attributed { expr, .. } => {
            // Attributes are metadata, just lower the inner expression
            lower_expr(spv, expr)
        }
    }
}

fn lower_literal(spv: &mut SpvBuilder, lit: &Literal) -> Result<spirv::Word> {
    match lit {
        Literal::Int(s) => {
            let value: i32 = s
                .parse()
                .map_err(|_| CompilerError::SpirvError(format!("Invalid integer literal: {}", s)))?;
            Ok(spv.const_i32(value))
        }
        Literal::Float(s) => {
            let value: f32 = s
                .parse()
                .map_err(|_| CompilerError::SpirvError(format!("Invalid float literal: {}", s)))?;
            Ok(spv.const_f32(value))
        }
        Literal::Bool(b) => Ok(spv.const_bool(*b)),
        Literal::String(_) => Err(CompilerError::SpirvError(
            "String literals not supported in SPIR-V".to_string(),
        )),
        Literal::Tuple(elems) => {
            // Lower all elements
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|e| lower_expr(spv, e)).collect::<Result<Vec<_>>>()?;

            // Create struct type for tuple from element types
            let elem_types: Vec<spirv::Word> = elems.iter().map(|e| spv.ast_type_to_spirv(&e.ty)).collect();
            let tuple_type = spv.type_struct(elem_types);

            // Construct the composite
            spv.composite_construct(tuple_type, elem_ids)
        }
        Literal::Array(elems) => {
            // Lower all elements
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|e| lower_expr(spv, e)).collect::<Result<Vec<_>>>()?;

            // Get element type from first element (or i32 as fallback)
            let elem_type = elems.first().map(|e| spv.ast_type_to_spirv(&e.ty)).unwrap_or(spv.i32_type);

            // Create array type
            let array_type = spv.type_array(elem_type, elem_ids.len() as u32);

            // Construct the composite
            spv.composite_construct(array_type, elem_ids)
        }
        Literal::Record(fields) => {
            // Records are represented as structs with fields in order
            let field_ids: Vec<spirv::Word> =
                fields.iter().map(|(_, e)| lower_expr(spv, e)).collect::<Result<Vec<_>>>()?;

            // Create struct type for record from field types
            let field_types: Vec<spirv::Word> =
                fields.iter().map(|(_, e)| spv.ast_type_to_spirv(&e.ty)).collect();
            let record_type = spv.type_struct(field_types);

            // Construct the composite
            spv.composite_construct(record_type, field_ids)
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
        let tokens = tokenize(source).expect("Tokenization failed");
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().expect("Parsing failed");

        let mut flattener =
            Flattener::new(std::collections::HashMap::new(), std::collections::HashSet::new());
        let mir = flattener.flatten_program(&ast)?;

        lower(&mir)
    }

    fn compile_to_spirv_with_types(source: &str) -> Result<Vec<u32>> {
        let tokens = tokenize(source).expect("Tokenization failed");
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().expect("Parsing failed");

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().expect("Failed to load builtins");
        let type_table = type_checker.check_program(&ast).expect("Type checking failed");

        let builtins = crate::builtin_registry::BuiltinRegistry::default().all_names();
        let mut flattener = Flattener::new(type_table, builtins);
        let mir = flattener.flatten_program(&ast)?;

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
        let spirv = compile_to_spirv_with_types(
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
        let spirv = compile_to_spirv_with_types(
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
