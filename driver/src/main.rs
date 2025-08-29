use clap::{Parser, Subcommand};
use runic_compiler::Compiler;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Parser)]
#[command(name = "runic")]
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
        
        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Validate a source file without generating output
    Check {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,
        
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
    CompilationError(#[from] runic_compiler::error::CompilerError),
}

fn main() -> Result<(), DriverError> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Compile { input, output, verbose } => {
            compile_file(input, output, verbose)?;
        }
        Commands::Check { input, verbose } => {
            check_file(input, verbose)?;
        }
    }
    
    Ok(())
}

fn compile_file(input: PathBuf, output: Option<PathBuf>, verbose: bool) -> Result<(), DriverError> {
    if verbose {
        println!("Compiling {}...", input.display());
    }
    
    // Read source file
    let source = fs::read_to_string(&input)?;
    
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

fn check_file(input: PathBuf, verbose: bool) -> Result<(), DriverError> {
    if verbose {
        println!("Checking {}...", input.display());
    }
    
    // Read source file
    let source = fs::read_to_string(&input)?;
    
    // Compile but don't write output
    let compiler = Compiler::new();
    let _spirv = compiler.compile(&source)?;
    
    if verbose {
        println!("✓ {} is valid", input.display());
    }
    
    Ok(())
}