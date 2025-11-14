use crate::ast::*;
use crate::cfg::{BlockId, Location};
use std::io::Write;

/// Generates Nemo/Datalog facts for basic block analysis
pub struct NemoFactWriter<W: Write> {
    writer: W,
    debug: bool,
}

impl<W: Write> NemoFactWriter<W> {
    pub fn new(writer: W, debug: bool) -> Self {
        Self { writer, debug }
    }

    /// Write a basic block fact: block(BlockId).
    pub fn write_block_fact(&mut self, block_id: BlockId) -> Result<(), std::io::Error> {
        writeln!(self.writer, "block({}).", block_id.0)?;
        if self.debug {
            eprintln!("DEBUG: Added block fact: block({})", block_id.0);
        }
        Ok(())
    }

    /// Write a location fact: location(LocationId, BlockId, Index).
    pub fn write_location_fact(
        &mut self,
        location_id: usize,
        location: &Location,
    ) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "location({}, {}, {}).",
            location_id, location.block.0, location.index
        )?;
        if self.debug {
            eprintln!(
                "DEBUG: Added location fact: location({}, {}, {})",
                location_id, location.block.0, location.index
            );
        }
        Ok(())
    }

    /// Write a control flow edge fact: edge(FromBlock, ToBlock).
    pub fn write_edge_fact(
        &mut self,
        from_block: BlockId,
        to_block: BlockId,
    ) -> Result<(), std::io::Error> {
        writeln!(self.writer, "edge({}, {}).", from_block.0, to_block.0)?;
        if self.debug {
            eprintln!("DEBUG: Added edge fact: edge({}, {})", from_block.0, to_block.0);
        }
        Ok(())
    }

    /// Write an expression fact: expr_at(LocationId, ExprType).
    pub fn write_expr_fact(&mut self, location_id: usize, expr_type: &str) -> Result<(), std::io::Error> {
        writeln!(self.writer, "expr_at({}, \"{}\").", location_id, expr_type)?;
        if self.debug {
            eprintln!(
                "DEBUG: Added expr fact: expr_at({}, \"{}\")",
                location_id, expr_type
            );
        }
        Ok(())
    }

    /// Write variable reference fact: var_ref(LocationId, VarName).
    pub fn write_var_ref_fact(&mut self, location_id: usize, var_name: &str) -> Result<(), std::io::Error> {
        writeln!(self.writer, "var_ref({}, \"{}\").", location_id, var_name)?;
        if self.debug {
            eprintln!(
                "DEBUG: Added var_ref fact: var_ref({}, \"{}\")",
                location_id, var_name
            );
        }
        Ok(())
    }

    /// Write variable definition fact: var_def(LocationId, VarName).
    pub fn write_var_def_fact(&mut self, location_id: usize, var_name: &str) -> Result<(), std::io::Error> {
        writeln!(self.writer, "var_def({}, \"{}\").", location_id, var_name)?;
        if self.debug {
            eprintln!(
                "DEBUG: Added var_def fact: var_def({}, \"{}\")",
                location_id, var_name
            );
        }
        Ok(())
    }

    /// Write lifetime start fact: lifetime_start(LifetimeId, LocationId, VarName).
    pub fn write_lifetime_start_fact(
        &mut self,
        lifetime_id: usize,
        location_id: usize,
        var_name: &str,
    ) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "lifetime_start({}, {}, \"{}\").",
            lifetime_id, location_id, var_name
        )?;
        if self.debug {
            eprintln!(
                "DEBUG: Added lifetime_start fact: lifetime_start({}, {}, \"{}\")",
                lifetime_id, location_id, var_name
            );
        }
        Ok(())
    }

    /// Write borrow fact: borrow(BorrowId, LocationId, VarName, LifetimeId).
    pub fn write_borrow_fact(
        &mut self,
        borrow_id: usize,
        location_id: usize,
        var_name: &str,
        lifetime_id: usize,
    ) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "borrow({}, {}, \"{}\", {}).",
            borrow_id, location_id, var_name, lifetime_id
        )?;
        if self.debug {
            eprintln!(
                "DEBUG: Added borrow fact: borrow({}, {}, \"{}\", {})",
                borrow_id, location_id, var_name, lifetime_id
            );
        }
        Ok(())
    }

    /// Write function call fact: call(LocationId, FuncName, Args).
    pub fn write_call_fact(
        &mut self,
        location_id: usize,
        func_name: &str,
        args: &[&str],
    ) -> Result<(), std::io::Error> {
        let args_str = args.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ");
        writeln!(
            self.writer,
            "call({}, \"{}\", [{}]).",
            location_id, func_name, args_str
        )?;
        if self.debug {
            eprintln!(
                "DEBUG: Added call fact: call({}, \"{}\", [{}])",
                location_id, func_name, args_str
            );
        }
        Ok(())
    }

    /// Write a comment header for the facts file
    pub fn write_header(&mut self) -> Result<(), std::io::Error> {
        writeln!(self.writer, "% Nemo/Datalog facts for Wyn basic block analysis")?;
        writeln!(self.writer, "% Generated by Wyn compiler")?;
        writeln!(self.writer)?;
        writeln!(self.writer, "% Facts:")?;
        writeln!(self.writer, "% block(BlockId) - A basic block exists")?;
        writeln!(
            self.writer,
            "% location(LocationId, BlockId, Index) - A location within a block"
        )?;
        writeln!(
            self.writer,
            "% edge(FromBlock, ToBlock) - Control flow edge between blocks"
        )?;
        writeln!(
            self.writer,
            "% expr_at(LocationId, ExprType) - Expression type at location"
        )?;
        writeln!(
            self.writer,
            "% var_ref(LocationId, VarName) - Variable reference at location"
        )?;
        writeln!(
            self.writer,
            "% var_def(LocationId, VarName) - Variable definition at location"
        )?;
        writeln!(self.writer)?;
        Ok(())
    }
}

/// Extract expression type name for fact generation
pub fn expr_type_name(expr: &Expression) -> &'static str {
    match &expr.kind {
        ExprKind::RecordLiteral(_) => "record_literal",
        ExprKind::IntLiteral(_) => "int_literal",
        ExprKind::FloatLiteral(_) => "float_literal",
        ExprKind::BoolLiteral(_) => "bool_literal",
        ExprKind::Identifier(_) => "identifier",
        ExprKind::ArrayLiteral(_) => "array_literal",
        ExprKind::ArrayIndex(..) => "array_index",
        ExprKind::BinaryOp(..) => "binary_op",
        ExprKind::FunctionCall(..) => "function_call",
        ExprKind::Application(..) => "application",
        ExprKind::Tuple(_) => "tuple",
        ExprKind::Lambda(_) => "lambda",
        ExprKind::LetIn(_) => "let_in",
        ExprKind::FieldAccess(..) => "field_access",
        ExprKind::If(..) => "if_expr",

        ExprKind::TypeHole => "type_hole",
        ExprKind::QualifiedName(_, _) => "qualified_name",
        ExprKind::UnaryOp(_, _) => "unary_op",
        ExprKind::Loop(_) => "loop",
        ExprKind::InternalLoop(_) => "internal_loop",
        ExprKind::Match(_) => "match",
        ExprKind::Range(_) => "range",
        ExprKind::Pipe(_, _) => "pipe",
        ExprKind::TypeAscription(_, _) => "type_ascription",
        ExprKind::TypeCoercion(_, _) => "type_coercion",
        ExprKind::Unsafe(_) => "unsafe",
        ExprKind::Assert(_, _) => "assert",
    } // NEWCASESHERE - add new cases before this closing brace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::BlockId;

    #[test]
    fn test_fact_generation() {
        let mut output = Vec::new();
        let mut writer = NemoFactWriter::new(&mut output, false);

        writer.write_header().unwrap();
        writer.write_block_fact(BlockId(0)).unwrap();
        writer.write_block_fact(BlockId(1)).unwrap();
        writer.write_edge_fact(BlockId(0), BlockId(1)).unwrap();

        let location = Location {
            block: BlockId(0),
            index: 0,
        };
        writer.write_location_fact(1, &location).unwrap();
        writer.write_expr_fact(1, "binary_op").unwrap();
        writer.write_var_ref_fact(1, "x").unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("block(0)."));
        assert!(result.contains("block(1)."));
        assert!(result.contains("edge(0, 1)."));
        assert!(result.contains("location(1, 0, 0)."));
        assert!(result.contains("expr_at(1, \"binary_op\")."));
        assert!(result.contains("var_ref(1, \"x\")."));
    }
}
