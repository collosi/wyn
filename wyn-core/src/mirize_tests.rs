#[cfg(test)]
mod tests {
    use crate::lexer;
    use crate::mirize::Mirize;
    use crate::parser::Parser;
    use crate::type_checker::TypeChecker;

    #[test]
    fn test_simple_function() {
        let source = "def add (x: i32) (y: i32): i32 = x + y";
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().unwrap();
        let type_table = type_checker.check_program(&program).unwrap();

        let mirize = Mirize::new(type_table);
        let mir = mirize.mirize_program(&program).unwrap();

        assert_eq!(mir.functions.len(), 1);
        let func = &mir.functions[0];
        assert_eq!(func.name, "add");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.blocks.len(), 1);

        // Should have: add instruction, return instruction
        let block = &func.blocks[0];
        assert_eq!(block.instructions.len(), 2);
    }

    #[test]
    fn test_simple_if() {
        let source = r#"
def test(x: i32): i32 =
  if x == 0 then 1 else 2
"#;
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().unwrap();
        let type_table = type_checker.check_program(&program).unwrap();

        let mirize = Mirize::new(type_table);
        let mir = mirize.mirize_program(&program).unwrap();

        assert_eq!(mir.functions.len(), 1);
        let func = &mir.functions[0];

        eprintln!("\n=== Simple If MIR ===");
        eprintln!("Function: {} ({} blocks)", func.name, func.blocks.len());
        for block in &func.blocks {
            eprintln!("  Block {}:", block.id);
            for inst in &block.instructions {
                eprintln!("    {:?}", inst);
            }
        }

        // Should have multiple blocks for if-then-else
        assert!(
            func.blocks.len() >= 3,
            "Expected at least 3 blocks for if-then-else, got {}",
            func.blocks.len()
        );
    }

    #[test]
    fn test_nested_if() {
        let source = r#"
def test(x: i32): i32 =
  if x == 0 then
    1
  else if x == 1 then
    2
  else
    3
"#;
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().unwrap();
        let type_table = type_checker.check_program(&program).unwrap();

        let mirize = Mirize::new(type_table);
        let mir = mirize.mirize_program(&program).unwrap();

        assert_eq!(mir.functions.len(), 1);
        let func = &mir.functions[0];

        eprintln!("\n=== Nested If MIR ===");
        eprintln!("Function: {} ({} blocks)", func.name, func.blocks.len());
        for block in &func.blocks {
            eprintln!("  Block {}:", block.id);
            for inst in &block.instructions {
                eprintln!("    {:?}", inst);
            }
        }

        // Verify Phi nodes have correct predecessor blocks
        for block in &func.blocks {
            use crate::mir::Instruction;
            for inst in &block.instructions {
                if let Instruction::Phi(dest, incoming) = inst {
                    eprintln!("\nValidating Phi in block {}: {:?}", block.id, dest);
                    for (value_reg, pred_block_id) in incoming {
                        eprintln!(
                            "  Incoming from block {}: register {}",
                            pred_block_id, value_reg.id
                        );

                        // Find the predecessor block
                        let pred_block = func.blocks.iter().find(|b| b.id == *pred_block_id);
                        assert!(
                            pred_block.is_some(),
                            "Phi in block {} references non-existent predecessor block {}",
                            block.id,
                            pred_block_id
                        );

                        let pred_block = pred_block.unwrap();

                        // Check that the predecessor block has a terminator that branches to this block
                        let last_inst = pred_block.instructions.last();
                        assert!(
                            last_inst.is_some(),
                            "Predecessor block {} has no instructions",
                            pred_block_id
                        );

                        let branches_here = match last_inst.unwrap() {
                            Instruction::Branch(target) => *target == block.id,
                            Instruction::BranchCond(_, true_target, false_target, _) => {
                                *true_target == block.id || *false_target == block.id
                            }
                            _ => false,
                        };

                        assert!(
                            branches_here,
                            "Predecessor block {} doesn't branch to block {} (last inst: {:?})",
                            pred_block_id,
                            block.id,
                            last_inst.unwrap()
                        );

                        // Check that the register is defined in the specific predecessor block
                        let pred_block = func
                            .blocks
                            .iter()
                            .find(|b| b.id == *pred_block_id)
                            .expect("Predecessor block should exist");
                        let reg_defined = pred_block.instructions.iter().any(|i| match i {
                            Instruction::ConstInt(r, _) => r.id == value_reg.id,
                            Instruction::ConstFloat(r, _) => r.id == value_reg.id,
                            Instruction::ConstBool(r, _) => r.id == value_reg.id,
                            Instruction::Add(r, _, _) => r.id == value_reg.id,
                            Instruction::Sub(r, _, _) => r.id == value_reg.id,
                            Instruction::Mul(r, _, _) => r.id == value_reg.id,
                            Instruction::Div(r, _, _) => r.id == value_reg.id,
                            Instruction::Eq(r, _, _) => r.id == value_reg.id,
                            Instruction::CallBuiltin(r, _, _) => r.id == value_reg.id,
                            Instruction::Phi(r, _) => r.id == value_reg.id,
                            _ => false,
                        });

                        assert!(
                            reg_defined,
                            "Register {} used in Phi is not defined in predecessor block {}",
                            value_reg.id, pred_block_id
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_let_binding() {
        let source = r#"
def test(): i32 =
  let x: i32 = 5 in
  let y: i32 = 10 in
  x + y
"#;
        let tokens = lexer::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut type_checker = TypeChecker::new();
        type_checker.load_builtins().unwrap();
        let type_table = type_checker.check_program(&program).unwrap();

        let mirize = Mirize::new(type_table);
        let mir = mirize.mirize_program(&program).unwrap();

        assert_eq!(mir.functions.len(), 1);
        let func = &mir.functions[0];

        eprintln!("\n=== Let Binding MIR ===");
        eprintln!("Function: {} ({} blocks)", func.name, func.blocks.len());
        for block in &func.blocks {
            eprintln!("  Block {}:", block.id);
            for inst in &block.instructions {
                eprintln!("    {:?}", inst);
            }
        }
    }
}
