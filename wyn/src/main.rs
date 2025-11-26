use clap::{Parser, Subcommand};
use log::info;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use thiserror::Error;
use wyn_core::{Compiler, Flattened, lexer, parser::Parser as WynParser};

/// Times the execution of a closure and prints the elapsed time if verbose.
fn time<T, F: FnOnce() -> T>(name: &str, verbose: bool, f: F) -> T {
    let start = Instant::now();
    let result = f();
    if verbose {
        let elapsed = start.elapsed().as_millis();
        eprintln!("{}: {}ms", name, elapsed);
    }
    result
}

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

        /// Output MIR (Mid-level IR) to a file
        #[arg(long, value_name = "FILE")]
        output_mir: Option<PathBuf>,

        /// Output annotated source code with block IDs and locations
        #[arg(long, value_name = "FILE")]
        output_annotated: Option<PathBuf>,

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
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            output_mir,
            output_annotated,
            verbose,
        } => {
            compile_file(input, output, output_mir, output_annotated, verbose)?;
        }
        Commands::Check {
            input,
            output_annotated,
            verbose,
        } => {
            check_file(input, output_annotated, verbose)?;
        }
    }

    Ok(())
}

fn compile_file(
    input: PathBuf,
    output: Option<PathBuf>,
    output_mir: Option<PathBuf>,
    output_annotated: Option<PathBuf>,
    verbose: bool,
) -> Result<(), DriverError> {
    if verbose {
        info!("Compiling {}...", input.display());
    }

    // Read source file
    let source = fs::read_to_string(&input)?;

    // Generate annotated source if requested
    if let Some(ref annotated_path) = output_annotated {
        generate_annotated_source(&source, annotated_path, verbose)?;
    }

    // Compile through the pipeline
    let parsed = time("parse", verbose, || Compiler::parse(&source))?;
    let elaborated = time("elaborate", verbose, || parsed.elaborate())?;
    let resolved = time("resolve", verbose, || elaborated.resolve())?;
    let type_checked = time("type_check", verbose, || resolved.type_check())?;

    type_checked.print_warnings();

    let alias_checked = time("alias_check", verbose, || type_checked.alias_check())?;
    if alias_checked.has_alias_errors() {
        alias_checked.print_alias_errors();
    }

    let flattened = time("flatten", verbose, || alias_checked.flatten())?;

    // Write MIR if requested (before further passes that might fail)
    write_mir_if_requested(&flattened, &output_mir, verbose)?;

    let monomorphized = time("monomorphize", verbose, || flattened.monomorphize())?;
    let reachable = time("filter_reachable", verbose, || monomorphized.filter_reachable());
    let folded = time("fold_constants", verbose, || reachable.fold_constants())?;
    let lowered = time("lower", verbose, || folded.lower())?;

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let mut path = input.clone();
        path.set_extension("spv");
        path
    });

    // Write SPIR-V binary
    let mut file = fs::File::create(&output_path)?;
    let spirv_len = lowered.spirv.len();
    for word in lowered.spirv {
        file.write_all(&word.to_le_bytes())?;
    }

    if verbose {
        info!("Successfully compiled to {}", output_path.display());
        info!("Generated {} words of SPIR-V", spirv_len);
    }

    Ok(())
}

fn check_file(input: PathBuf, output_annotated: Option<PathBuf>, verbose: bool) -> Result<(), DriverError> {
    if verbose {
        info!("Checking {}...", input.display());
    }

    // Read source file
    let source = fs::read_to_string(&input)?;

    // Generate annotated source if requested
    if let Some(ref annotated_path) = output_annotated {
        generate_annotated_source(&source, annotated_path, verbose)?;
    }

    // Type check and alias check, don't generate code
    let type_checked = Compiler::parse(&source)?.elaborate()?.resolve()?.type_check()?;

    type_checked.print_warnings();

    let alias_checked = type_checked.alias_check()?;
    if alias_checked.has_alias_errors() {
        alias_checked.print_alias_errors();
    }

    if verbose {
        info!("✓ {} is valid", input.display());
    }

    Ok(())
}

fn write_mir_if_requested(
    flattened: &Flattened,
    output_mir: &Option<PathBuf>,
    verbose: bool,
) -> Result<(), DriverError> {
    if let Some(ref mir_path) = output_mir {
        fs::write(mir_path, format!("{}", flattened.mir))?;
        if verbose {
            info!("Wrote MIR to {}", mir_path.display());
        }
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
    let _program = parser.parse()?;

    // Generate annotated code - temporarily disabled
    // let mut annotator = CodeAnnotator::new();
    // let annotated = annotator.annotate_program(&program);

    // Write annotated source
    // fs::write(output_path, annotated)?;

    // Temporary placeholder
    fs::write(output_path, "// Annotated code generation temporarily disabled\n")?;

    if verbose {
        info!("Generated annotated source: {}", output_path.display());
    }

    Ok(())
}
