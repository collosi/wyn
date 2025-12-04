//! Module manager for lazy loading and caching module definitions

use crate::ast::{
    Decl, Declaration, ModuleExpression, ModuleTypeExpression, Node, NodeCounter, Pattern, PatternKind,
    Program, Spec, Type, TypeName, TypeParam,
};
use crate::error::Result;
use crate::lexer;
use crate::parser::Parser;
use crate::scope::ScopeStack;
use crate::{bail_module, err_module, err_parse};
use polytype::{Context, TypeScheme};
use std::collections::{HashMap, HashSet};

/// Name resolver for tracking opened modules and resolving unqualified names
/// TODO: Integrate with elaboration to handle `open` declarations
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct NameResolver {
    /// Modules currently opened (via `open` declarations)
    opened_modules: Vec<String>,
    /// Local definitions in scope (for shadowing)
    local_scope: ScopeStack<()>,
}

#[allow(dead_code)]
impl NameResolver {
    fn new() -> Self {
        NameResolver {
            opened_modules: Vec::new(),
            local_scope: ScopeStack::new(),
        }
    }

    /// Resolve an unqualified name by checking opened modules
    /// Returns None if the name can't be resolved, or Some(qualified_name) if found
    fn resolve_name(&self, name: &str, module_manager: &ModuleManager) -> Option<String> {
        // 1. Check if it's locally defined (shadows everything)
        if self.local_scope.is_defined(name) {
            return None; // Keep unqualified, it's a local binding
        }

        // 2. Try each opened module in reverse order (most recent first)
        for module_name in self.opened_modules.iter().rev() {
            // Check if this module has this function
            if let Some(elaborated) = module_manager.elaborated_modules.get(module_name) {
                for item in &elaborated.items {
                    let item_name = match item {
                        ElaboratedItem::Spec(Spec::Sig(n, _, _)) => Some(n.as_str()),
                        ElaboratedItem::Spec(Spec::SigOp(op, _)) => Some(op.as_str()),
                        ElaboratedItem::Decl(decl) => Some(decl.name.as_str()),
                        _ => None,
                    };

                    if item_name == Some(name) {
                        return Some(format!("{}.{}", module_name, name));
                    }
                }
            }
        }

        // 3. Not found in any opened module
        None
    }

    /// Open a module (bring its names into scope)
    fn open_module(&mut self, module_name: String) {
        self.opened_modules.push(module_name);
    }

    /// Close the most recently opened module
    fn close_module(&mut self) {
        self.opened_modules.pop();
    }

    /// Push a new scope for local definitions
    fn push_scope(&mut self) {
        self.local_scope.push_scope();
    }

    /// Pop the current scope
    fn pop_scope(&mut self) {
        self.local_scope.pop_scope();
    }

    /// Add a local definition (for shadowing)
    fn add_local(&mut self, name: String) {
        self.local_scope.insert(name, ());
    }
}

/// Represents a single item in an elaborated module
#[derive(Debug, Clone)]
pub enum ElaboratedItem {
    /// A signature spec (from module type) with substitutions applied
    Spec(Spec),
    /// A declaration (def/let) from module body with substitutions and resolved names
    Decl(Decl),
}

/// Represents a fully elaborated module with all includes expanded, type substitutions applied,
/// and names resolved. Contains both signature specs and body declarations in source order.
#[derive(Debug, Clone)]
pub struct ElaboratedModule {
    pub name: String,
    /// Items in source order (specs first, then body declarations)
    pub items: Vec<ElaboratedItem>,
}

/// Pre-elaborated prelude data that can be shared across compilations (for test performance)
#[derive(Clone)]
pub struct PreElaboratedPrelude {
    /// Module type registry: type name -> ModuleTypeExpression
    pub module_type_registry: HashMap<String, ModuleTypeExpression>,
    /// Elaborated modules: module_name -> ElaboratedModule
    pub elaborated_modules: HashMap<String, ElaboratedModule>,
    /// Set of known module names (for name resolution)
    pub known_modules: HashSet<String>,
}

/// Manages lazy loading of module files
pub struct ModuleManager {
    /// Module type registry: type name -> ModuleTypeExpression
    module_type_registry: HashMap<String, ModuleTypeExpression>,
    /// Elaborated modules: module_name -> ElaboratedModule
    pub(crate) elaborated_modules: HashMap<String, ElaboratedModule>,
    /// Set of known module names (for name resolution)
    known_modules: HashSet<String>,
    /// Shared node counter for unique NodeIds across all modules
    node_counter: NodeCounter,
}

impl ModuleManager {
    /// Create a new module manager with a fresh NodeCounter
    pub fn new() -> Self {
        let mut manager = Self::new_empty(NodeCounter::new());
        if let Err(e) = manager.load_prelude_files() {
            eprintln!("ERROR loading prelude files: {:?}", e);
        }
        manager
    }

    /// Create a new module manager with a shared NodeCounter
    /// This ensures NodeIds don't collide with user code that was already parsed
    pub fn new_with_counter(node_counter: NodeCounter) -> Self {
        let mut manager = Self::new_empty(node_counter);
        manager.load_prelude_files().ok(); // Ignore errors during initialization
        manager
    }

    /// Create an empty module manager without loading prelude (internal helper)
    fn new_empty(node_counter: NodeCounter) -> Self {
        let known_modules = [
            "f32",
            "f64",
            "f16",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "bool",
            "graphics32",
            "graphics64",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        ModuleManager {
            module_type_registry: HashMap::new(),
            elaborated_modules: HashMap::new(),
            known_modules,
            node_counter,
        }
    }

    /// Load all prelude files automatically
    fn load_prelude_files(&mut self) -> Result<()> {
        // Load all prelude files using include_str!
        self.load_str(include_str!("../../../prelude/math.wyn"))?;
        self.load_str(include_str!("../../../prelude/graphics.wyn"))?;
        Ok(())
    }

    /// Create a pre-elaborated prelude by loading all prelude files
    /// This can be cached and reused across compilations
    pub fn create_prelude() -> Result<PreElaboratedPrelude> {
        let mut manager = Self::new_empty(NodeCounter::new());
        manager.load_prelude_files()?;
        Ok(PreElaboratedPrelude {
            module_type_registry: manager.module_type_registry,
            elaborated_modules: manager.elaborated_modules,
            known_modules: manager.known_modules,
        })
    }

    /// Create a ModuleManager using a pre-elaborated prelude (avoids re-parsing)
    pub fn from_prelude(prelude: &PreElaboratedPrelude, node_counter: NodeCounter) -> Self {
        ModuleManager {
            module_type_registry: prelude.module_type_registry.clone(),
            elaborated_modules: prelude.elaborated_modules.clone(),
            known_modules: prelude.known_modules.clone(),
            node_counter,
        }
    }

    /// Check if a name is a known module
    pub fn is_known_module(&self, name: &str) -> bool {
        self.known_modules.contains(name)
    }

    /// Load and elaborate modules from a source string
    pub fn load_str(&mut self, source: &str) -> Result<()> {
        // Parse the source
        let tokens = lexer::tokenize(source).map_err(|e| err_parse!("{}", e))?;
        let counter = std::mem::take(&mut self.node_counter);
        let mut parser = Parser::new_with_counter(tokens, counter);
        let program = parser.parse()?;
        self.node_counter = parser.take_node_counter();

        // Register module types first
        self.register_module_types(&program)?;

        // Elaborate all modules from the program
        self.elaborate_all_modules(&program)?;

        Ok(())
    }

    /// Elaborate all module bindings from a parsed program
    fn elaborate_all_modules(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            if let Declaration::ModuleBind(mb) = decl {
                if self.elaborated_modules.contains_key(&mb.name) {
                    bail_module!("Module '{}' is already defined", mb.name);
                }

                // Extract type substitutions from the signature
                let substitutions = if let Some(signature) = &mb.signature {
                    self.extract_substitutions(signature)?
                } else {
                    HashMap::new()
                };

                let mut items = Vec::new();

                // Elaborate the module signature if it exists
                if let Some(signature) = &mb.signature {
                    let specs = self.elaborate_module_type(signature, &HashMap::new())?;
                    // Wrap specs in ElaboratedItem::Spec
                    items.extend(specs.into_iter().map(ElaboratedItem::Spec));
                }

                // Elaborate the module body
                let body_items = self.elaborate_module_body(&mb.body, &substitutions)?;
                items.extend(body_items);

                let elaborated = ElaboratedModule {
                    name: mb.name.clone(),
                    items,
                };

                self.elaborated_modules.insert(mb.name.clone(), elaborated);
            }
        }
        Ok(())
    }

    /// Extract type substitutions from a module signature
    /// e.g., (float with t = f32 with int_t = u32) -> {t: f32, int_t: u32}
    fn extract_substitutions(&self, mte: &ModuleTypeExpression) -> Result<HashMap<String, Type>> {
        let mut substitutions = HashMap::new();
        let mut current = mte;

        // Walk through nested With expressions
        loop {
            match current {
                ModuleTypeExpression::With(inner, type_name, _type_params, type_value) => {
                    substitutions.insert(type_name.clone(), type_value.clone());
                    current = inner;
                }
                _ => break,
            }
        }

        Ok(substitutions)
    }

    /// Register all module type definitions from a parsed program
    fn register_module_types(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            if let Declaration::ModuleTypeBind(mtb) = decl {
                if self.module_type_registry.contains_key(&mtb.name) {
                    bail_module!("Module type '{}' is already defined", mtb.name);
                }
                self.module_type_registry.insert(mtb.name.clone(), mtb.definition.clone());
            }
        }
        Ok(())
    }

    /// Elaborate a module type expression into a flat list of specs
    /// Recursively expands includes and applies type substitutions
    fn elaborate_module_type(
        &self,
        mte: &ModuleTypeExpression,
        substitutions: &HashMap<String, Type>,
    ) -> Result<Vec<Spec>> {
        match mte {
            ModuleTypeExpression::Name(name) => {
                // Look up the module type in the registry
                let definition = self
                    .module_type_registry
                    .get(name)
                    .ok_or_else(|| err_module!("Module type '{}' not found", name))?;
                // Recurse on the definition
                self.elaborate_module_type(definition, substitutions)
            }

            ModuleTypeExpression::Signature(specs) => {
                // Process each spec, expanding includes and applying substitutions
                let mut result = Vec::new();
                for spec in specs {
                    match spec {
                        Spec::Include(inner_mte) => {
                            // Recursively elaborate the included module type
                            let included_specs = self.elaborate_module_type(inner_mte, substitutions)?;
                            result.extend(included_specs);
                        }
                        _ => {
                            // Apply type substitutions to the spec and add it
                            let substituted_spec = self.substitute_in_spec(spec, substitutions);
                            result.push(substituted_spec);
                        }
                    }
                }
                Ok(result)
            }

            ModuleTypeExpression::With(inner, type_name, _type_params, type_value) => {
                // Add the type substitution and recurse on the inner expression
                let mut new_substitutions = substitutions.clone();
                new_substitutions.insert(type_name.clone(), type_value.clone());
                self.elaborate_module_type(inner, &new_substitutions)
            }

            ModuleTypeExpression::Arrow(_, _, _) | ModuleTypeExpression::FunctorType(_, _) => {
                // Functor types not yet supported
                Err(err_module!("Functor types are not yet supported"))
            }
        }
    }

    /// Apply type substitutions to a spec
    fn substitute_in_spec(&self, spec: &Spec, substitutions: &HashMap<String, Type>) -> Spec {
        match spec {
            Spec::Sig(name, type_params, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::Sig(name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::SigOp(op, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::SigOp(op.clone(), substituted_ty)
            }
            Spec::Type(kind, name, type_params, maybe_ty) => {
                let substituted_ty = maybe_ty.as_ref().map(|ty| self.substitute_in_type(ty, substitutions));
                Spec::Type(kind.clone(), name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::Module(name, mte) => {
                // Don't substitute in nested module signatures for now
                Spec::Module(name.clone(), mte.clone())
            }
            Spec::Include(_) => {
                // Includes should have been expanded by now
                spec.clone()
            }
        }
    }

    /// Apply type substitutions to a type
    fn substitute_in_type(&self, ty: &Type, substitutions: &HashMap<String, Type>) -> Type {
        use crate::ast::TypeName;

        match ty {
            Type::Constructed(name, args) => {
                // Check if this is a named type that should be substituted
                if let TypeName::Named(type_name) = name {
                    if args.is_empty() {
                        if let Some(replacement) = substitutions.get(type_name) {
                            return replacement.clone();
                        }
                    }
                }

                // Recursively substitute in type arguments
                let new_args: Vec<Type> =
                    args.iter().map(|arg| self.substitute_in_type(arg, substitutions)).collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Convert a type with type parameters to a polymorphic TypeScheme
    /// Converts SizeVar("n") and UserVar("t") to fresh Type::Variables
    /// and wraps the result in nested TypeScheme::Polytype layers
    fn convert_to_polytype(
        &self,
        ty: &Type,
        type_params: &[TypeParam],
        context: &mut Context<TypeName>,
    ) -> TypeScheme<TypeName> {
        if type_params.is_empty() {
            return TypeScheme::Monotype(ty.clone());
        }

        // Create fresh variables for each parameter and build substitution map
        let mut substitutions: HashMap<String, polytype::Variable> = HashMap::new();
        let mut var_ids = Vec::new();

        for param in type_params {
            let var = context.new_variable();
            if let Type::Variable(id) = var {
                var_ids.push(id);
                match param {
                    TypeParam::Size(name) => {
                        substitutions.insert(name.clone(), id);
                    }
                    TypeParam::Type(name) => {
                        substitutions.insert(name.clone(), id);
                    }
                    _ => {} // Ignore other param types for now
                }
            }
        }

        // Substitute SizeVar/UserVar with Variable in the type
        let substituted_ty = self.substitute_params(&ty, &substitutions);

        // Wrap in nested Polytype layers
        let mut result = TypeScheme::Monotype(substituted_ty);
        for &var_id in var_ids.iter().rev() {
            result = TypeScheme::Polytype {
                variable: var_id,
                body: Box::new(result),
            };
        }

        result
    }

    /// Recursively substitute SizeVar and UserVar with Variable
    fn substitute_params(&self, ty: &Type, substitutions: &HashMap<String, polytype::Variable>) -> Type {
        match ty {
            Type::Constructed(TypeName::SizeVar(name), args) => {
                if let Some(&var_id) = substitutions.get(name) {
                    Type::Variable(var_id)
                } else {
                    // Not in our substitution map, keep as-is
                    Type::Constructed(
                        TypeName::SizeVar(name.clone()),
                        args.iter().map(|a| self.substitute_params(a, substitutions)).collect(),
                    )
                }
            }
            Type::Constructed(TypeName::UserVar(name), args) => {
                if let Some(&var_id) = substitutions.get(name) {
                    Type::Variable(var_id)
                } else {
                    Type::Constructed(
                        TypeName::UserVar(name.clone()),
                        args.iter().map(|a| self.substitute_params(a, substitutions)).collect(),
                    )
                }
            }
            Type::Constructed(name, args) => Type::Constructed(
                name.clone(),
                args.iter().map(|a| self.substitute_params(a, substitutions)).collect(),
            ),
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Query the type of a function in a specific module
    /// Returns a TypeScheme for polymorphic functions (with type/size params)
    /// e.g., get_module_function_type("f32", "sum") -> TypeScheme::Polytype for [n] param
    pub fn get_module_function_type(
        &self,
        module_name: &str,
        function_name: &str,
        context: &mut Context<TypeName>,
    ) -> Result<TypeScheme<TypeName>> {
        // Look up the elaborated module
        let elaborated = self
            .elaborated_modules
            .get(module_name)
            .ok_or_else(|| err_module!("Module '{}' not found", module_name))?;

        // Search for the function in the elaborated items
        for item in &elaborated.items {
            match item {
                ElaboratedItem::Spec(spec) => match spec {
                    Spec::Sig(name, type_params, ty) if name == function_name => {
                        // Convert to TypeScheme if there are type/size parameters
                        return Ok(self.convert_to_polytype(ty, type_params, context));
                    }
                    Spec::SigOp(op, ty) if op == function_name => {
                        // Operators currently don't have type parameters, return as Monotype
                        return Ok(TypeScheme::Monotype(ty.clone()));
                    }
                    _ => {}
                },
                ElaboratedItem::Decl(decl) if decl.name == function_name => {
                    // Build the full function type from parameters and return type
                    // For def min (x: f32) (y: f32): f32, we need to construct f32 -> f32 -> f32
                    return self.build_function_type_from_decl(decl, context);
                }
                _ => {}
            }
        }

        Err(err_module!(
            "Function '{}' not found in module '{}'",
            function_name,
            module_name
        ))
    }

    /// Build the full function type from a declaration's parameters and return type
    fn build_function_type_from_decl(
        &self,
        decl: &Decl,
        context: &mut Context<TypeName>,
    ) -> Result<TypeScheme<TypeName>> {
        // Extract parameter types
        let mut param_types = Vec::new();
        for param in &decl.params {
            if let Some(param_ty) = self.extract_type_from_pattern(param) {
                param_types.push(param_ty);
            } else {
                bail_module!("Function parameter in '{}' lacks type annotation", decl.name);
            }
        }

        // Get return type (default to unit if not specified)
        let return_type = decl.ty.clone().unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));

        // Build function type by folding right-to-left
        // f32 -> f32 -> f32 is represented as f32 -> (f32 -> f32)
        let mut result_type = return_type;
        for param_ty in param_types.into_iter().rev() {
            result_type = Type::Constructed(TypeName::Arrow, vec![param_ty, result_type]);
        }

        // Convert to TypeScheme if there are type/size parameters
        // For declarations, check if the type contains UserVar or SizeVar
        let type_params = self.extract_type_params_from_type(&result_type);
        Ok(self.convert_to_polytype(&result_type, &type_params, context))
    }

    /// Extract type parameters from a type by finding all UserVar and SizeVar
    fn extract_type_params_from_type(&self, ty: &Type) -> Vec<TypeParam> {
        let mut params = HashSet::new();
        self.collect_type_params(ty, &mut params);

        params.into_iter().collect()
    }

    /// Recursively collect type parameters from a type
    fn collect_type_params(&self, ty: &Type, params: &mut HashSet<TypeParam>) {
        match ty {
            Type::Constructed(TypeName::UserVar(name), args) => {
                params.insert(TypeParam::Type(name.clone()));
                for arg in args {
                    self.collect_type_params(arg, params);
                }
            }
            Type::Constructed(TypeName::SizeVar(name), args) => {
                params.insert(TypeParam::Size(name.clone()));
                for arg in args {
                    self.collect_type_params(arg, params);
                }
            }
            Type::Constructed(_, args) => {
                for arg in args {
                    self.collect_type_params(arg, params);
                }
            }
            Type::Variable(_) => {}
        }
    }

    /// Extract type annotation from a pattern
    fn extract_type_from_pattern(&self, pattern: &Pattern) -> Option<Type> {
        match &pattern.kind {
            PatternKind::Typed(_, ty) => Some(ty.clone()),
            PatternKind::Tuple(pats) => {
                // For tuple patterns, extract types from each element
                let elem_types: Option<Vec<Type>> =
                    pats.iter().map(|p| self.extract_type_from_pattern(p)).collect();
                elem_types.map(|types| Type::Constructed(TypeName::Tuple(types.len()), types))
            }
            _ => None,
        }
    }

    /// Check if a name is a qualified module reference (e.g., "f32.sum")
    pub fn is_qualified_name(name: &str) -> bool {
        name.contains('.')
    }

    /// Split a qualified name into (module, function) parts
    /// E.g., "f32.sum" -> Some(("f32", "sum"))
    pub fn split_qualified_name(name: &str) -> Option<(&str, &str)> {
        let parts: Vec<&str> = name.splitn(2, '.').collect();
        if parts.len() == 2 { Some((parts[0], parts[1])) } else { None }
    }

    /// Elaborate a module body expression into a list of elaborated items
    /// Applies type substitutions to declaration signatures
    fn elaborate_module_body(
        &self,
        module_expr: &ModuleExpression,
        substitutions: &HashMap<String, Type>,
    ) -> Result<Vec<ElaboratedItem>> {
        match module_expr {
            ModuleExpression::Struct(declarations) => {
                let mut items = Vec::new();

                for decl in declarations {
                    match decl {
                        Declaration::Decl(d) => {
                            // Apply type substitutions to the declaration signature only
                            let elaborated_decl = self.elaborate_decl_signature(d, substitutions);
                            items.push(ElaboratedItem::Decl(elaborated_decl));
                        }
                        Declaration::Sig(sig_decl) => {
                            // Apply type substitutions to sig declarations
                            let substituted_ty = self.substitute_in_type(&sig_decl.ty, substitutions);

                            // Convert size_params and type_params to Vec<TypeParam>
                            use crate::ast::TypeParam;
                            let mut type_params_vec = Vec::new();
                            for size_param in &sig_decl.size_params {
                                type_params_vec.push(TypeParam::Size(size_param.clone()));
                            }
                            for type_param in &sig_decl.type_params {
                                type_params_vec.push(TypeParam::Type(type_param.clone()));
                            }

                            let spec = Spec::Sig(sig_decl.name.clone(), type_params_vec, substituted_ty);
                            items.push(ElaboratedItem::Spec(spec));
                        }
                        Declaration::Open(_) => {
                            // TODO: Handle open declarations for name resolution
                        }
                        _ => {
                            // Skip other declaration types (ModuleTypeBind, etc.)
                        }
                    }
                }

                Ok(items)
            }
            _ => {
                // For now, only handle struct module expressions
                Err(err_module!("Only struct module expressions are supported"))
            }
        }
    }

    /// Elaborate a declaration's signature (params and return type) with type substitutions
    fn elaborate_decl_signature(&self, decl: &Decl, substitutions: &HashMap<String, Type>) -> Decl {
        // Apply type substitutions to params
        let new_params: Vec<Pattern> =
            decl.params.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();

        // Apply type substitutions to return type
        let new_ty = decl.ty.as_ref().map(|ty| self.substitute_in_type(ty, substitutions));

        Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
            params: new_params,
            ty: new_ty,
            body: decl.body.clone(), // TODO: Apply type substitution and name resolution in body
        }
    }

    /// Apply type substitutions to a pattern
    fn substitute_in_pattern(&self, pattern: &Pattern, substitutions: &HashMap<String, Type>) -> Pattern {
        let new_kind = match &pattern.kind {
            PatternKind::Typed(inner, ty) => {
                let new_ty = self.substitute_in_type(ty, substitutions);
                PatternKind::Typed(inner.clone(), new_ty)
            }
            PatternKind::Tuple(pats) => {
                let new_pats: Vec<Pattern> =
                    pats.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();
                PatternKind::Tuple(new_pats)
            }
            PatternKind::Record(fields) => {
                let new_fields = fields
                    .iter()
                    .map(|field| crate::ast::RecordPatternField {
                        field: field.field.clone(),
                        pattern: field
                            .pattern
                            .as_ref()
                            .map(|p| self.substitute_in_pattern(p, substitutions)),
                    })
                    .collect();
                PatternKind::Record(new_fields)
            }
            PatternKind::Constructor(name, pats) => {
                let new_pats: Vec<Pattern> =
                    pats.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();
                PatternKind::Constructor(name.clone(), new_pats)
            }
            PatternKind::Attributed(attrs, inner) => {
                let new_inner = self.substitute_in_pattern(inner, substitutions);
                PatternKind::Attributed(attrs.clone(), Box::new(new_inner))
            }
            // Name, Wildcard, Literal, Unit don't contain types
            _ => pattern.kind.clone(),
        };

        Node {
            h: pattern.h.clone(),
            kind: new_kind,
        }
    }
}

impl Default for ModuleManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
