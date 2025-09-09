# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wyn is a minimal compiler for a Futhark-like language that generates SPIR-V code. It's structured as a Rust workspace with two crates: the compiler library and a CLI driver.

## Commands

Build the project:
```bash
LLVM_SYS_180_PREFIX=/usr/local/Cellar/llvm@18/18.1.8 cargo build
```

Run tests:
```bash
LLVM_SYS_180_PREFIX=/usr/local/Cellar/llvm@18/18.1.8 cargo test
```

Compile a Wyn source file:
```bash
LLVM_SYS_180_PREFIX=/usr/local/Cellar/llvm@18/18.1.8 cargo run --bin wyn -- compile test.wyn -o test.spv
```

Check a source file without generating output:
```bash
LLVM_SYS_180_PREFIX=/usr/local/Cellar/llvm@18/18.1.8 cargo run --bin wyn -- check test.wyn
```

## Architecture

### Compilation Pipeline
1. **Lexer** (`compiler/src/lexer.rs`): Tokenizes input using nom combinators
2. **Parser** (`compiler/src/parser.rs`): Builds AST from tokens
3. **Type Checker** (`compiler/src/type_checker.rs`): Validates types and maintains symbol table
4. **Code Generator** (`compiler/src/codegen.rs`): Generates SPIR-V using rspirv

### Key Design Decisions
- **Error Handling**: Uses thiserror for structured error types
- **Parsing**: nom for combinator-based parsing (primarily for lexer)
- **Type System**: Simple type checking without inference (polytype removed for simplicity)
- **SPIR-V Generation**: rspirv provides safe SPIR-V building APIs

### Adding New Features
- Language features start in `lexer.rs` (tokens) and `ast.rs` (AST nodes)
- Parser rules go in `parser.rs` 
- Type checking logic in `type_checker.rs`
- SPIR-V generation in `codegen.rs`
- All new syntax elements should have unit tests