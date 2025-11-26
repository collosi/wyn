use crate::NodeId;
use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::nemo_facts::NemoFactWriter;
use crate::visitor::{self, Visitor};
use nemo::execution::{DefaultExecutionEngine, ExecutionEngine};
use nemo::io::import_manager::ImportManager;
use nemo::io::resource_providers::ResourceProviders;
use nemo::rule_file::RuleFile;
use nemo::rule_model::components::tag::Tag;
use nemo::rule_model::programs::handle::ProgramHandle;
use std::io::Write;
use std::ops::ControlFlow;

/// Borrow checker using Nemo rule engine
///
/// Walks the AST and emits facts based on NodeId, then runs Nemo
/// to detect use-after-move and other borrow errors.
pub struct BorrowChecker {
    debug: bool,
}

impl BorrowChecker {
    pub fn new(debug: bool) -> Self {
        Self { debug }
    }

    /// Generate Nemo program (facts + rules) for borrow checking
    pub fn generate_nemo_program<W: Write>(&self, writer: &mut W, program: &Program) -> Result<()> {
        let mut fact_writer = NemoFactWriter::new(writer, self.debug);
        fact_writer.write_header().map_err(Self::io_error)?;

        // Extract facts from the AST
        let mut extractor = FactExtractor {
            fact_writer: &mut fact_writer,
            parent_stack: Vec::new(),
        };

        for decl in &program.declarations {
            extractor.extract_declaration(decl)?;
        }

        // Write the analysis rules
        extractor.fact_writer.write_rules().map_err(Self::io_error)?;

        Ok(())
    }

    /// Run borrow checking analysis on the program
    pub fn check_program(&self, program: &Program) -> Result<BorrowCheckResult> {
        // Generate Nemo program
        let mut nemo_program = Vec::new();
        self.generate_nemo_program(&mut nemo_program, program)?;
        let nemo_program_str = String::from_utf8(nemo_program).map_err(|_| {
            CompilerError::SpirvError("Failed to convert Nemo program to string".to_string())
        })?;

        if self.debug {
            println!("Generated Nemo program:\n{}", nemo_program_str);
        }

        // Run Nemo analysis
        self.run_nemo_analysis(&nemo_program_str)
    }

    fn run_nemo_analysis(&self, nemo_program: &str) -> Result<BorrowCheckResult> {
        // Create a RuleFile from the program string
        let rule_file = RuleFile::new(nemo_program.to_string(), "borrow_check.nemo".to_string());

        // Parse and validate the program
        let program_handle = ProgramHandle::from_file(&rule_file)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to parse Nemo program: {:?}", e)))?;

        if self.debug {
            println!("Successfully parsed Nemo program");
        }

        // Materialize the program
        let program = program_handle.into_object().materialize();

        // Create import manager (empty since we have no external data)
        let import_manager = ImportManager::new(ResourceProviders::empty());

        // Initialize and run the execution engine
        let mut engine = ExecutionEngine::initialize(program, import_manager)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to initialize Nemo engine: {:?}", e)))?;

        engine
            .execute()
            .map_err(|e| CompilerError::SpirvError(format!("Nemo execution failed: {:?}", e)))?;

        if self.debug {
            println!("Nemo program executed successfully");
        }

        // Query for use_after_move errors
        let errors = self.extract_errors(&mut engine)?;
        Ok(BorrowCheckResult { errors })
    }

    fn extract_errors(&self, engine: &mut DefaultExecutionEngine) -> Result<Vec<BorrowError>> {
        let mut errors = Vec::new();

        // Try to get results from the use_after_move predicate
        let tag = Tag::new("use_after_move".to_string());
        if let Some(iter) = engine
            .predicate_rows(&tag)
            .map_err(|e| CompilerError::SpirvError(format!("Failed to query results: {:?}", e)))?
        {
            for row in iter {
                // Each row should have (NodeId, VarName)
                if row.len() >= 2 {
                    let node_id = row[0].to_string();
                    let var_name = row[1].to_string().trim_matches('"').to_string();
                    errors.push(BorrowError::UseAfterMove {
                        variable: var_name,
                        use_node: node_id,
                    });
                }
            }
        }

        Ok(errors)
    }

    fn io_error(e: std::io::Error) -> CompilerError {
        CompilerError::SpirvError(format!("IO error during borrow checking: {}", e))
    }
}

/// Macro to convert Result to ControlFlow::Break on error
macro_rules! try_cf {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => return ControlFlow::Break(e),
        }
    };
}

/// AST visitor that extracts facts for borrow checking
struct FactExtractor<'a, W: Write> {
    fact_writer: &'a mut NemoFactWriter<W>,
    /// Stack of parent NodeIds for tracking AST structure
    parent_stack: Vec<NodeId>,
}

impl<'a, W: Write> FactExtractor<'a, W> {
    fn extract_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(d) => self.extract_decl(d),
            Declaration::Entry(e) => self.extract_entry(e),
            Declaration::Uniform(_) | Declaration::Val(_) => Ok(()), // No runtime borrow checking needed
            _ => Ok(()), // Skip unsupported declarations for now
        }
    }

    fn extract_decl(&mut self, decl: &Decl) -> Result<()> {
        // Function parameters introduce scopes
        for param in &decl.params {
            self.extract_pattern_bindings(param)?;
        }

        // Visit the body
        match self.visit_expression(&decl.body) {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(e) => Err(e),
        }
    }

    fn extract_entry(&mut self, entry: &EntryDecl) -> Result<()> {
        // Entry parameters introduce scopes
        for param in &entry.params {
            self.extract_pattern_bindings(param)?;
        }

        // Visit the body
        match self.visit_expression(&entry.body) {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(e) => Err(e),
        }
    }

    fn extract_pattern_bindings(&mut self, pattern: &Pattern) -> Result<()> {
        let names = pattern.collect_names();
        for name in names {
            self.fact_writer.write_var_def(pattern.h.id, &name).map_err(BorrowChecker::io_error)?;
            self.fact_writer.write_scope_intro(pattern.h.id, &name).map_err(BorrowChecker::io_error)?;
        }
        Ok(())
    }

    fn write_parent_edge(&mut self, child_id: NodeId) -> std::io::Result<()> {
        if let Some(&parent_id) = self.parent_stack.last() {
            self.fact_writer.write_parent(parent_id, child_id)?;
        }
        Ok(())
    }
}

impl<'a, W: Write> Visitor for FactExtractor<'a, W> {
    type Break = CompilerError;

    fn visit_expression(&mut self, expr: &Expression) -> ControlFlow<Self::Break> {
        let node_id = expr.h.id;

        // Write parent edge
        try_cf!(self.write_parent_edge(node_id).map_err(BorrowChecker::io_error));

        // Push this node as potential parent
        self.parent_stack.push(node_id);

        // Use standard walk which dispatches to our overridden methods
        let result = visitor::walk_expression(self, expr);

        // Pop parent
        self.parent_stack.pop();

        result
    }

    fn visit_expr_identifier(&mut self, id: NodeId, name: &str) -> ControlFlow<Self::Break> {
        // Variable use
        try_cf!(self.fact_writer.write_var_use(id, name).map_err(BorrowChecker::io_error));
        ControlFlow::Continue(())
    }

    fn visit_expr_let_in(&mut self, id: NodeId, let_in: &LetInExpr) -> ControlFlow<Self::Break> {
        // Let binding introduces scope for bound variables
        let names = let_in.pattern.collect_names();
        for name in &names {
            try_cf!(self.fact_writer.write_var_def(id, name).map_err(BorrowChecker::io_error));
            try_cf!(self.fact_writer.write_scope_intro(id, name).map_err(BorrowChecker::io_error));
        }

        // Visit the pattern (for nested patterns)
        self.visit_pattern(&let_in.pattern)?;

        // Visit value expression
        self.visit_expression(&let_in.value)?;

        // Visit body expression
        self.visit_expression(&let_in.body)?;

        ControlFlow::Continue(())
    }

    fn visit_expr_lambda(&mut self, id: NodeId, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        // Lambda parameters introduce scopes
        for param in &lambda.params {
            let names = param.collect_names();
            for name in &names {
                try_cf!(self.fact_writer.write_var_def(id, name).map_err(BorrowChecker::io_error));
                try_cf!(self.fact_writer.write_scope_intro(id, name).map_err(BorrowChecker::io_error));
            }
            self.visit_pattern(param)?;
        }

        // Visit body
        self.visit_expression(&lambda.body)?;

        ControlFlow::Continue(())
    }

    fn visit_expr_application(
        &mut self,
        _id: NodeId,
        func: &Expression,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        // Visit function
        self.visit_expression(func)?;

        // Visit arguments - these might be move points
        for arg in args {
            self.visit_expression(arg)?;
        }

        ControlFlow::Continue(())
    }
}

#[derive(Debug, Clone)]
pub struct BorrowCheckResult {
    pub errors: Vec<BorrowError>,
}

#[derive(Debug, Clone)]
pub enum BorrowError {
    UseAfterMove {
        variable: String,
        use_node: String,
    },
}

impl BorrowCheckResult {
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn print_errors(&self) {
        for error in &self.errors {
            match error {
                BorrowError::UseAfterMove { variable, use_node } => {
                    println!(
                        "Error: Use after move of variable '{}' at node {}",
                        variable, use_node
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    #[test]
    fn test_borrow_checker_basic() {
        let source = r#"def main (x: i32): i32 = x + 1"#;

        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let checker = BorrowChecker::new(true);
        let result = checker.check_program(&program).unwrap();

        // Should not have errors for simple case
        assert!(!result.has_errors());
    }

    #[test]
    fn test_fact_extraction() {
        let source = r#"def test (x: i32): i32 = let y = x in y + x"#;

        let tokens = tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let checker = BorrowChecker::new(false);
        let mut output = Vec::new();
        checker.generate_nemo_program(&mut output, &program).unwrap();

        let nemo_str = String::from_utf8(output).unwrap();

        // Should contain var_use facts for x and y
        assert!(nemo_str.contains("var_use"));
        // Should contain scope_intro for let binding
        assert!(nemo_str.contains("scope_intro"));
        // Should contain parent edges
        assert!(nemo_str.contains("parent"));
    }
}
