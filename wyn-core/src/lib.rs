// pub mod annotator;
pub mod ast;
pub mod diags;
pub mod error;
pub mod flattening;
pub mod impl_source;
pub mod inference;
pub mod lexer;
pub mod mir;
pub mod module_manager;
pub mod name_resolution;
pub mod parser;
pub mod pattern;
pub mod poly_builtins;
pub mod reachability;
pub mod scope;
pub mod type_checker;
pub mod visitor;

pub mod alias_checker;
pub mod constant_folding;
pub mod gasm_lowering;
pub mod lowering;
pub mod monomorphization;

#[cfg(test)]
mod alias_checker_tests;
#[cfg(test)]
mod constant_folding_tests;
#[cfg(test)]
mod flattening_tests;
#[cfg(test)]
mod monomorphization_tests;
#[cfg(test)]
mod type_checker_tests;

use std::collections::HashMap;

use ast::NodeId;
use error::Result;
use polytype::TypeScheme;

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
//       -> .flatten()    -> Flattened
//       -> .monomorphize() -> Monomorphized
//       -> .filter_reachable() -> Reachable
//       -> .fold_constants() -> Folded
//       -> .lower()      -> Lowered (contains MIR + SPIR-V)

/// Entry point for the compiler. Use `Compiler::parse()` to start the pipeline.
pub struct Compiler;

impl Compiler {
    /// Parse source code into an AST. This is the entry point for the pipeline.
    pub fn parse(source: &str) -> Result<Parsed> {
        let tokens = lexer::tokenize(source).map_err(error::CompilerError::ParseError)?;
        let mut parser = parser::Parser::new(tokens);
        let ast = parser.parse()?;
        let node_counter = parser.take_node_counter();
        Ok(Parsed { ast, node_counter })
    }
}

/// Stage 1: Source has been parsed into an AST
pub struct Parsed {
    pub ast: ast::Program,
    pub node_counter: ast::NodeCounter,
}

impl Parsed {
    /// Elaborate modules: currently a no-op, module elaboration happens in module_manager
    pub fn elaborate(self) -> Result<Elaborated> {
        // TODO: In the future, this could transform module declarations into flat declarations
        // For now, modules are handled via module_manager registry during type checking
        Ok(Elaborated {
            ast: self.ast,
            node_counter: self.node_counter,
        })
    }
}

/// Stage 2: Modules have been elaborated
pub struct Elaborated {
    pub ast: ast::Program,
    pub node_counter: ast::NodeCounter,
}

impl Elaborated {
    /// Resolve names: rewrite FieldAccess -> QualifiedName and load modules
    pub fn resolve(mut self) -> Result<Resolved> {
        let mut resolver = name_resolution::NameResolver::new_with_counter(self.node_counter);
        resolver.resolve_program(&mut self.ast)?;
        Ok(Resolved { ast: self.ast })
    }
}

/// Stage 3: Names have been resolved
pub struct Resolved {
    pub ast: ast::Program,
}

impl Resolved {
    /// Type check the program
    pub fn type_check(self) -> Result<TypeChecked> {
        let mut checker = type_checker::TypeChecker::new();
        checker.load_builtins()?;
        let type_table = checker.check_program(&self.ast)?;

        // Collect warnings
        let warnings: Vec<_> = checker.warnings().to_vec();

        Ok(TypeChecked {
            ast: self.ast,
            type_table,
            warnings,
        })
    }
}

/// Stage 4: Program has been type checked
pub struct TypeChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub warnings: Vec<type_checker::TypeWarning>,
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
        })
    }
}

/// Stage 5: Program has been alias checked
pub struct AliasChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub alias_result: alias_checker::AliasCheckResult,
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

    /// Flatten AST to MIR (with defunctionalization and desugaring)
    pub fn flatten(self) -> Result<Flattened> {
        let builtins = impl_source::ImplSource::default().all_names();
        let mut flattener = flattening::Flattener::new(self.type_table, builtins);
        let mir = flattener.flatten_program(&self.ast)?;
        Ok(Flattened { mir })
    }
}

/// Stage 6: AST has been flattened to MIR
pub struct Flattened {
    pub mir: mir::Program,
}

impl Flattened {
    /// Monomorphize: specialize polymorphic functions
    pub fn monomorphize(self) -> Result<Monomorphized> {
        let mir = monomorphization::monomorphize(self.mir)?;
        Ok(Monomorphized { mir })
    }
}

/// Stage 7: Program has been monomorphized
pub struct Monomorphized {
    pub mir: mir::Program,
}

impl Monomorphized {
    /// Filter to only reachable functions
    pub fn filter_reachable(self) -> Reachable {
        let mir = reachability::filter_reachable(self.mir);
        Reachable { mir }
    }
}

/// Stage 8: Unreachable code has been filtered out
pub struct Reachable {
    pub mir: mir::Program,
}

impl Reachable {
    /// Fold constants: evaluate constant expressions at compile time
    pub fn fold_constants(self) -> Result<Folded> {
        let mir = constant_folding::fold_constants(self.mir)?;
        Ok(Folded { mir })
    }
}

/// Stage 9: Constants have been folded
pub struct Folded {
    pub mir: mir::Program,
}

impl Folded {
    /// Lower MIR to SPIR-V
    pub fn lower(self) -> Result<Lowered> {
        self.lower_with_options(false)
    }

    /// Lower MIR to SPIR-V with debug mode option
    pub fn lower_with_options(self, debug_enabled: bool) -> Result<Lowered> {
        let spirv = lowering::lower(&self.mir, debug_enabled)?;
        Ok(Lowered { mir: self.mir, spirv })
    }
}

/// Stage 10: Final stage - contains MIR and SPIR-V bytecode
pub struct Lowered {
    pub mir: mir::Program,
    pub spirv: Vec<u32>,
}
