use clap::{Parser, Subcommand};
use spur::compiler;
use spur::compiler::cfg::Program;
use spur::simulator::explorer::run_explorer;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "spur-frontend")]
#[command(about = "A compiler for spur.", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a specification file to JSON
    Compile {
        /// Input specification file
        spec: String,
        /// Compiled output file
        output: String,
    },
    /// Compile and run the execution explorer
    Explore {
        /// Input specification file
        spec: String,
        /// Explorer configuration JSON file
        config: String,
        /// SQLite output file for results
        output: String,
    },
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    match args.command {
        Commands::Compile { spec, output } => {
            compile_spec(&spec, &output);
        }
        Commands::Explore {
            spec,
            config,
            output,
        } => {
            let program = match compile_spec_to_program(&spec) {
                Some(p) => p,
                None => std::process::exit(1),
            };
            let start = Instant::now();
            if let Err(e) = run_explorer(&program, &config, &output) {
                eprintln!("Explorer failed: {}", e);
                std::process::exit(1);
            }
            let elapsed = start.elapsed();
            println!(
                "Explorer finished in {:.2?}. Results saved to {}",
                elapsed, output
            );
        }
    }
}

fn read_spec_file(spec: &str) -> Option<String> {
    let path = Path::new(spec);

    if !path.exists() {
        eprintln!("Error: Input file '{}' does not exist", spec);
        return None;
    }

    match fs::read_to_string(path) {
        Ok(content) => Some(content),
        Err(e) => {
            eprintln!("Error: Failed to read input file '{}': {}", spec, e);
            None
        }
    }
}

fn compile_spec_to_program(spec: &str) -> Option<Program> {
    let content = read_spec_file(spec)?;

    match compiler::compile(&content, spec) {
        Ok(program) => {
            println!("Successfully compiled {}", spec);
            Some(program)
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            None
        }
    }
}

fn compile_spec(spec: &str, output: &str) {
    println!("Input spec: {}, Output location: {}", spec, output);

    let program = match compile_spec_to_program(spec) {
        Some(p) => p,
        None => std::process::exit(1),
    };

    let json = serde_json::to_string_pretty(&program).expect("Failed to serialize program");

    if let Err(e) = fs::write(output, json) {
        eprintln!("Error: Failed to write output file '{}': {}", output, e);
        eprintln!("Possible causes:");
        eprintln!("  - Directory does not exist");
        eprintln!("  - Insufficient permissions");
        eprintln!("  - Disk full or read-only filesystem");
        std::process::exit(1);
    }

    println!("Successfully compiled to {}", output);
}
