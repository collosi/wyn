pub mod ast;
pub mod diags;
pub mod error;
pub mod flattening;
pub mod impl_source;
pub mod lexer;
pub mod mir;
pub mod module_manager;
pub mod name_resolution;
pub mod parser;
pub mod pattern;
pub mod poly_builtins;
pub mod reachability;
pub mod scope;
pub mod types;
pub mod visitor;

// Re-export type_checker from its new location for backwards compatibility
pub use types::checker as type_checker;

pub mod alias_checker;
pub mod ast_const_fold;
pub mod binding_lifter;
pub mod constant_folding;
pub mod glsl;
pub mod lowering_common;
pub mod materialize_hoisting;
pub mod monomorphization;
pub mod normalize;
pub mod spirv;

#[cfg(test)]
mod alias_checker_tests;
#[cfg(test)]
mod binding_lifter_tests;
#[cfg(test)]
mod constant_folding_tests;
#[cfg(test)]
mod flattening_tests;
#[cfg(test)]
mod monomorphization_tests;
#[cfg(test)]
mod normalize_tests;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use indexmap::IndexMap;

use ast::{NodeCounter, NodeId};
use error::Result;
use polytype::TypeScheme;

// =============================================================================
// Generic ID allocation
// =============================================================================

/// Generic counter for generating unique IDs.
///
/// The ID type must implement `From<u32>` to convert the raw counter value.
#[derive(Debug, Clone)]
pub struct IdSource<Id> {
    next_id: u32,
    _phantom: PhantomData<Id>,
}

impl<Id: From<u32>> IdSource<Id> {
    pub fn new() -> Self {
        IdSource {
            next_id: 0,
            _phantom: PhantomData,
        }
    }

    pub fn next(&mut self) -> Id {
        let id = Id::from(self.next_id);
        self.next_id += 1;
        id
    }
}

impl<Id: From<u32>> Default for IdSource<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// Arena that allocates IDs and stores associated items.
///
/// Combines ID generation with storage, ensuring each item gets a unique ID.
/// Uses IndexMap for deterministic iteration order (insertion order).
#[derive(Debug, Clone)]
pub struct IdArena<Id, T> {
    source: IdSource<Id>,
    items: IndexMap<Id, T>,
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IdArena<Id, T> {
    pub fn new() -> Self {
        IdArena {
            source: IdSource::new(),
            items: IndexMap::new(),
        }
    }

    /// Allocate a new ID and store the item.
    pub fn alloc(&mut self, item: T) -> Id {
        let id = self.source.next();
        self.items.insert(id, item);
        id
    }

    /// Allocate a new ID without storing anything yet.
    /// Use `insert` later to store the item.
    pub fn alloc_id(&mut self) -> Id {
        self.source.next()
    }

    /// Insert an item with a pre-allocated ID.
    /// Panics if the ID is already in use.
    pub fn insert(&mut self, id: Id, item: T) {
        let old = self.items.insert(id, item);
        assert!(old.is_none(), "IdArena::insert called with duplicate ID");
    }

    /// Get an item by ID.
    pub fn get(&self, id: Id) -> Option<&T> {
        self.items.get(&id)
    }

    /// Get a mutable reference to an item by ID.
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.items.get_mut(&id)
    }

    /// Iterate over all (id, item) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Id, &T)> {
        self.items.iter()
    }

    /// Iterate over all items (without IDs).
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.items.values()
    }

    /// Number of items in the arena.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> Default for IdArena<Id, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for IdArena<Id, T> {
    type Item = (Id, T);
    type IntoIter = indexmap::map::IntoIter<Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a IdArena<Id, T> {
    type Item = (&'a Id, &'a T);
    type IntoIter = indexmap::map::Iter<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a mut IdArena<Id, T> {
    type Item = (&'a Id, &'a mut T);
    type IntoIter = indexmap::map::IterMut<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

// Re-export key types for the public API
pub use ast::TypeName;
pub type TypeTable = HashMap<NodeId, TypeScheme<TypeName>>;

// =============================================================================
// Typestate Compiler Pipeline
// =============================================================================
//
// Each struct represents a stage in the compilation pipeline. Methods consume
// `self` and return the next stage, enforcing valid ordering at compile time.
//
// Pipeline:
//   Compiler::parse(source)
//     -> Parsed
//       -> .elaborate()  -> Elaborated
//       -> .resolve()    -> Resolved
//       -> .type_check() -> TypeChecked
//       -> .alias_check() -> AliasChecked
//       -> .fold_ast_constants() -> AstConstFolded (integer constant folding/inlining)
//       -> .flatten()    -> Flattened
//       -> .hoist_materializations() -> MaterializationsHoisted
//       -> .normalize()  -> Normalized (ANF transformation)
//       -> .monomorphize() -> Monomorphized
//       -> .filter_reachable() -> Reachable
//       -> .fold_constants() -> Folded (MIR-level constant folding)
//       -> .lower()      -> Lowered (contains MIR + SPIR-V)

/// Entry point for the compiler. Use `Compiler::parse()` to start the pipeline.
pub struct Compiler;

impl Compiler {
    /// Parse source code into an AST. This is the entry point for the pipeline.
    pub fn parse(source: &str) -> Result<Parsed> {
        let tokens = lexer::tokenize(source).map_err(|e| err_parse!("{}", e))?;
        let mut parser = parser::Parser::new(tokens);
        let ast = parser.parse()?;
        let node_counter = parser.take_node_counter();
        Ok(Parsed { ast, node_counter })
    }
}

/// Source has been parsed into an AST
pub struct Parsed {
    pub ast: ast::Program,
    pub node_counter: ast::NodeCounter,
}

impl Parsed {
    /// Elaborate modules using the provided ModuleManager
    pub fn elaborate(self, module_manager: module_manager::ModuleManager) -> Result<Elaborated> {
        // TODO: In the future, this could transform module declarations into flat declarations
        // For now, modules are handled via module_manager registry during type checking
        Ok(Elaborated {
            ast: self.ast,
            module_manager,
        })
    }
}

/// Modules have been elaborated
pub struct Elaborated {
    pub ast: ast::Program,
    module_manager: module_manager::ModuleManager,
}

impl Elaborated {
    /// Resolve names: rewrite FieldAccess -> QualifiedName and load modules
    pub fn resolve(mut self) -> Result<Resolved> {
        let mut resolver = name_resolution::NameResolver::with_module_manager(self.module_manager);
        resolver.resolve_program(&mut self.ast)?;
        Ok(Resolved {
            ast: self.ast,
            module_manager: resolver.into_module_manager(),
        })
    }
}

/// Names have been resolved
pub struct Resolved {
    pub ast: ast::Program,
    module_manager: module_manager::ModuleManager,
}

impl Resolved {
    /// Type check the program
    pub fn type_check(self) -> Result<TypeChecked> {
        let mut checker = type_checker::TypeChecker::with_module_manager(self.module_manager);
        checker.load_builtins()?;
        let type_table = checker.check_program(&self.ast)?;

        // Collect warnings
        let warnings: Vec<_> = checker.warnings().to_vec();

        // Get the module manager back from the checker
        let module_manager = checker.into_module_manager();

        Ok(TypeChecked {
            ast: self.ast,
            type_table,
            warnings,
            module_manager,
        })
    }
}

/// Program has been type checked
pub struct TypeChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub module_manager: module_manager::ModuleManager,
}

impl TypeChecked {
    /// Print warnings to stderr (convenience method)
    pub fn print_warnings(&self) {
        // We need a type checker instance to format types
        let checker = type_checker::TypeChecker::new();
        for warning in &self.warnings {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| checker.format_type(t)),
                warning.span()
            );
        }
    }

    /// Run alias checking analysis on the program
    pub fn alias_check(self) -> Result<AliasChecked> {
        let checker = alias_checker::AliasChecker::new(&self.type_table);
        let alias_result = checker.check_program(&self.ast)?;

        Ok(AliasChecked {
            ast: self.ast,
            type_table: self.type_table,
            warnings: self.warnings,
            alias_result,
            module_manager: self.module_manager,
        })
    }
}

/// Program has been alias checked
pub struct AliasChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub alias_result: alias_checker::AliasCheckResult,
    pub module_manager: module_manager::ModuleManager,
}

impl AliasChecked {
    /// Print warnings to stderr (convenience method)
    pub fn print_warnings(&self) {
        // We need a type checker instance to format types
        let checker = type_checker::TypeChecker::new();
        for warning in &self.warnings {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| checker.format_type(t)),
                warning.span()
            );
        }
    }

    /// Check if alias checking found any errors
    pub fn has_alias_errors(&self) -> bool {
        self.alias_result.has_errors()
    }

    /// Print alias errors to stderr
    pub fn print_alias_errors(&self) {
        self.alias_result.print_errors();
    }

    /// Fold AST-level integer constants
    pub fn fold_ast_constants(mut self) -> AstConstFolded {
        ast_const_fold::fold_ast_constants(&mut self.ast);
        AstConstFolded {
            ast: self.ast,
            type_table: self.type_table,
            warnings: self.warnings,
            alias_result: self.alias_result,
            module_manager: self.module_manager,
        }
    }

    /// Flatten AST to MIR (with defunctionalization and desugaring)
    /// Note: Consider using fold_ast_constants() first for better optimization
    pub fn flatten(self) -> Result<Flattened> {
        let builtins = impl_source::ImplSource::default().all_names();
        let mut flattener = flattening::Flattener::new(self.type_table, builtins);
        let mut mir = flattener.flatten_program(&self.ast)?;

        // Flatten module function declarations so they're available in SPIR-V
        for (module_name, decl) in self.module_manager.get_module_function_declarations() {
            let qualified_name = format!("{}.{}", module_name, decl.name);
            let defs = flattener.flatten_module_decl(decl, &qualified_name)?;
            mir.defs.extend(defs);
        }

        let node_counter = flattener.into_node_counter();
        Ok(Flattened {
            mir,
            node_counter,
            module_manager: self.module_manager,
        })
    }
}

/// AST integer constants have been folded and inlined
pub struct AstConstFolded {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub alias_result: alias_checker::AliasCheckResult,
    pub module_manager: module_manager::ModuleManager,
}

impl AstConstFolded {
    /// Flatten AST to MIR (with defunctionalization and desugaring)
    pub fn flatten(self) -> Result<Flattened> {
        let builtins = impl_source::ImplSource::default().all_names();
        let mut flattener = flattening::Flattener::new(self.type_table, builtins);
        let mut mir = flattener.flatten_program(&self.ast)?;

        // Flatten module function declarations so they're available in SPIR-V
        for (module_name, decl) in self.module_manager.get_module_function_declarations() {
            let qualified_name = format!("{}.{}", module_name, decl.name);
            let defs = flattener.flatten_module_decl(decl, &qualified_name)?;
            mir.defs.extend(defs);
        }

        let node_counter = flattener.into_node_counter();
        Ok(Flattened {
            mir,
            node_counter,
            module_manager: self.module_manager,
        })
    }
}

/// AST has been flattened to MIR
pub struct Flattened {
    pub mir: mir::Program,
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
}

impl Flattened {
    /// Hoist duplicate materializations to let bindings
    pub fn hoist_materializations(self) -> MaterializationsHoisted {
        let mir = materialize_hoisting::hoist_materializations(self.mir);
        MaterializationsHoisted {
            mir,
            node_counter: self.node_counter,
            module_manager: self.module_manager,
        }
    }
}

/// Duplicate materializations have been hoisted
pub struct MaterializationsHoisted {
    pub mir: mir::Program,
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
}

impl MaterializationsHoisted {
    /// Normalize MIR to A-normal form
    pub fn normalize(self) -> Normalized {
        let (mir, node_counter) = normalize::normalize_program(self.mir, self.node_counter);
        Normalized {
            mir,
            node_counter,
            module_manager: self.module_manager,
        }
    }
}

/// MIR has been normalized to A-normal form
pub struct Normalized {
    pub mir: mir::Program,
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
}

impl Normalized {
    /// Monomorphize: specialize polymorphic functions
    pub fn monomorphize(self) -> Result<Monomorphized> {
        let mir = monomorphization::monomorphize(self.mir)?;
        Ok(Monomorphized {
            mir,
            node_counter: self.node_counter,
            module_manager: self.module_manager,
        })
    }
}

/// Program has been monomorphized
pub struct Monomorphized {
    pub mir: mir::Program,
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
}

impl Monomorphized {
    /// Filter to only reachable functions
    pub fn filter_reachable(self) -> Reachable {
        let mir = reachability::filter_reachable(self.mir);
        Reachable {
            mir,
            node_counter: self.node_counter,
            module_manager: self.module_manager,
        }
    }
}

/// Unreachable code has been filtered out
pub struct Reachable {
    pub mir: mir::Program,
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
}

impl Reachable {
    /// Fold constants: evaluate constant expressions at compile time
    pub fn fold_constants(self) -> Result<Folded> {
        let mir = constant_folding::fold_constants(self.mir)?;
        Ok(Folded {
            mir,
            node_counter: self.node_counter,
            module_manager: self.module_manager,
        })
    }
}

/// Constants have been folded
pub struct Folded {
    pub mir: mir::Program,
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
}

impl Folded {
    /// Lift loop-invariant bindings out of loops
    pub fn lift_bindings(self) -> Result<Lifted> {
        let mut lifter = binding_lifter::BindingLifter::new();
        let mir = lifter.lift_program(self.mir)?;
        Ok(Lifted {
            mir,
            node_counter: self.node_counter,
            module_manager: self.module_manager,
        })
    }
}

/// Bindings have been lifted (loop-invariant code motion)
pub struct Lifted {
    pub mir: mir::Program,
    #[allow(dead_code)]
    pub node_counter: NodeCounter,
    pub module_manager: module_manager::ModuleManager,
}

impl Lifted {
    /// Lower MIR to SPIR-V
    pub fn lower(self) -> Result<Lowered> {
        self.lower_with_options(false)
    }

    /// Lower MIR to SPIR-V with debug mode option
    pub fn lower_with_options(self, debug_enabled: bool) -> Result<Lowered> {
        // Analyze for in-place map optimization opportunities
        let inplace_info = alias_checker::analyze_map_inplace(&self.mir);
        let spirv = spirv::lower(&self.mir, debug_enabled, &inplace_info)?;
        Ok(Lowered { mir: self.mir, spirv })
    }

    /// Lower MIR to GLSL
    pub fn lower_glsl(self) -> Result<LoweredGlsl> {
        let glsl = glsl::lower(&self.mir)?;
        Ok(LoweredGlsl { mir: self.mir, glsl })
    }

    /// Lower MIR to Shadertoy-compatible GLSL (fragment shader only)
    pub fn lower_shadertoy(self) -> Result<String> {
        glsl::lower_shadertoy(&self.mir)
    }
}

/// Final stage - contains MIR and SPIR-V bytecode
pub struct Lowered {
    pub mir: mir::Program,
    pub spirv: Vec<u32>,
}

/// Final stage for GLSL - contains MIR and GLSL source strings
pub struct LoweredGlsl {
    pub mir: mir::Program,
    pub glsl: glsl::GlslOutput,
}

// =============================================================================
// Test utilities - cached prelude for faster test execution
// =============================================================================

#[cfg(test)]
use std::sync::OnceLock;

#[cfg(test)]
static PRELUDE_CACHE: OnceLock<module_manager::PreElaboratedPrelude> = OnceLock::new();

/// Get a reference to the cached pre-elaborated prelude (test-only)
/// This avoids re-parsing prelude files for each test, providing ~10x speedup
#[cfg(test)]
fn get_prelude_cache() -> &'static module_manager::PreElaboratedPrelude {
    PRELUDE_CACHE.get_or_init(|| {
        module_manager::ModuleManager::create_prelude().expect("Failed to create prelude cache")
    })
}

/// Create a ModuleManager using the cached prelude (test-only)
#[cfg(test)]
pub fn cached_module_manager(node_counter: ast::NodeCounter) -> module_manager::ModuleManager {
    module_manager::ModuleManager::from_prelude(get_prelude_cache(), node_counter)
}
