use clap::{Parser, Subcommand};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use thiserror::Error;
use wyn_core::{
    Compiler, annotator::CodeAnnotator, borrow_checker::BorrowChecker, cfg_nemo::CfgNemoExtractor, lexer,
    parser::Parser as WynParser,
};

#[derive(Parser)]
#[command(name = "wyn")]
#[command(about = "A minimal Futhark-like language compiler targeting SPIR-V", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a source file to SPIR-V
    Compile {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output SPIR-V file (defaults to input name with .spv extension)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Output annotated source code with block IDs and locations
        #[arg(long, value_name = "FILE")]
        output_annotated: Option<PathBuf>,

        /// Output Nemo/Datalog facts for basic block analysis
        #[arg(long, value_name = "FILE")]
        output_nemo: Option<PathBuf>,

        /// Run borrow checking with Nemo rule engine
        #[arg(long)]
        borrow_check: bool,

        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Validate a source file without generating output
    Check {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output annotated source code with block IDs and locations
        #[arg(long, value_name = "FILE")]
        output_annotated: Option<PathBuf>,

        /// Output Nemo/Datalog facts for basic block analysis
        #[arg(long, value_name = "FILE")]
        output_nemo: Option<PathBuf>,

        /// Run borrow checking with Nemo rule engine
        #[arg(long)]
        borrow_check: bool,

        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Debug, Error)]
enum DriverError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Compilation error: {0}")]
    CompilationError(#[from] wyn_core::error::CompilerError),
}

fn main() -> Result<(), DriverError> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            output_annotated,
            output_nemo,
            borrow_check,
            verbose,
        } => {
            compile_file(
                input,
                output,
                output_annotated,
                output_nemo,
                borrow_check,
                verbose,
            )?;
        }
        Commands::Check {
            input,
            output_annotated,
            output_nemo,
            borrow_check,
            verbose,
        } => {
            check_file(input, output_annotated, output_nemo, borrow_check, verbose)?;
        }
    }

    Ok(())
}

fn compile_file(
    input: PathBuf,
    output: Option<PathBuf>,
    output_annotated: Option<PathBuf>,
    output_nemo: Option<PathBuf>,
    borrow_check: bool,
    verbose: bool,
) -> Result<(), DriverError> {
    if verbose {
        println!("Compiling {}...", input.display());
    }

    // Read source file
    let source = fs::read_to_string(&input)?;

    // Generate annotated source if requested
    if let Some(ref annotated_path) = output_annotated {
        generate_annotated_source(&source, annotated_path, verbose)?;
    }

    // Generate Nemo facts if requested
    if let Some(ref nemo_path) = output_nemo {
        generate_nemo_facts(&source, nemo_path, verbose)?;
    }

    // Run borrow checking if requested
    if borrow_check {
        run_borrow_checking(&source, verbose)?;
    }

    // Compile to SPIR-V
    let compiler = Compiler::new();
    let spirv = compiler.compile(&source)?;

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let mut path = input.clone();
        path.set_extension("spv");
        path
    });

    // Write SPIR-V binary
    let mut file = fs::File::create(&output_path)?;
    let spirv_len = spirv.len();
    for word in spirv {
        file.write_all(&word.to_le_bytes())?;
    }

    if verbose {
        println!("Successfully compiled to {}", output_path.display());
        println!("Generated {} words of SPIR-V", spirv_len);
    }

    Ok(())
}

fn check_file(
    input: PathBuf,
    output_annotated: Option<PathBuf>,
    output_nemo: Option<PathBuf>,
    borrow_check: bool,
    verbose: bool,
) -> Result<(), DriverError> {
    if verbose {
        println!("Checking {}...", input.display());
    }

    // Read source file
    let source = fs::read_to_string(&input)?;

    // Generate annotated source if requested
    if let Some(ref annotated_path) = output_annotated {
        generate_annotated_source(&source, annotated_path, verbose)?;
    }

    // Generate Nemo facts if requested
    if let Some(ref nemo_path) = output_nemo {
        generate_nemo_facts(&source, nemo_path, verbose)?;
    }

    // Run borrow checking if requested
    if borrow_check {
        run_borrow_checking(&source, verbose)?;
    }

    // Compile but don't write output
    let compiler = Compiler::new();
    let _spirv = compiler.compile(&source)?;

    if verbose {
        println!("✓ {} is valid", input.display());
    }

    Ok(())
}

fn generate_annotated_source(
    source: &str,
    output_path: &PathBuf,
    verbose: bool,
) -> Result<(), DriverError> {
    // Parse the source to get the AST
    let tokens = lexer::tokenize(source).map_err(wyn_core::error::CompilerError::ParseError)?;
    let mut parser = WynParser::new(tokens);
    let program = parser.parse()?;

    // Generate annotated code
    let mut annotator = CodeAnnotator::new();
    let annotated = annotator.annotate_program(&program);

    // Write annotated source
    fs::write(output_path, annotated)?;

    if verbose {
        println!("Generated annotated source: {}", output_path.display());
    }

    Ok(())
}

fn generate_nemo_facts(source: &str, output_path: &PathBuf, verbose: bool) -> Result<(), DriverError> {
    // Parse the source to get the AST
    let tokens = lexer::tokenize(source).map_err(wyn_core::error::CompilerError::ParseError)?;
    let mut parser = WynParser::new(tokens);
    let program = parser.parse()?;

    // Generate Nemo facts
    let mut file = fs::File::create(output_path)?;
    let extractor = CfgNemoExtractor::new(&mut file, verbose);
    extractor.extract_cfg(&program)?;

    if verbose {
        println!("Generated Nemo facts: {}", output_path.display());
    }

    Ok(())
}

fn run_borrow_checking(source: &str, verbose: bool) -> Result<(), DriverError> {
    // Parse the source to get the AST
    let tokens = lexer::tokenize(source).map_err(wyn_core::error::CompilerError::ParseError)?;
    let mut parser = WynParser::new(tokens);
    let program = parser.parse()?;

    // Run borrow checking
    let mut checker = BorrowChecker::new(verbose);
    let result = checker.check_program(&program)?;

    if result.has_errors() {
        println!("Borrow check errors found:");
        result.print_errors();
        return Err(DriverError::CompilationError(
            wyn_core::error::CompilerError::SpirvError("Borrow check failed".to_string()),
        ));
    } else if verbose {
        println!("✓ Borrow checking passed");
    }

    Ok(())
}
