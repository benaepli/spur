use clap::{Parser, Subcommand};
use spur::compiler;
use spur::compiler::cfg::Program;
use spur::simulator::explorer::run_explorer_genetic;
use spur::visualization::{render_html_heatmap, render_svg, vertex_coverage_to_byte_coverage};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
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
        /// Output directory for results
        output_dir: String,
        /// Skip confirmation prompt for directory deletion
        #[arg(short = 'y', long)]
        yes: bool,
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
            output_dir,
            yes,
        } => {
            let source_code = match read_spec_file(&spec) {
                Some(s) => s,
                None => std::process::exit(1),
            };
            let program = match compile_spec_to_program_with_source(&source_code, &spec) {
                Some(p) => p,
                None => std::process::exit(1),
            };

            // Handle output directory
            let output_path = Path::new(&output_dir);
            if output_path.exists() {
                if !yes {
                    print!(
                        "Output directory '{}' already exists and will be deleted. Continue? [y/N] ",
                        output_dir
                    );
                    io::stdout().flush().unwrap();

                    let mut input = String::new();
                    io::stdin().read_line(&mut input).unwrap();
                    if !input.trim().eq_ignore_ascii_case("y") {
                        println!("Aborted.");
                        std::process::exit(0);
                    }
                }
                if let Err(e) = fs::remove_dir_all(output_path) {
                    eprintln!("Failed to remove directory '{}': {}", output_dir, e);
                    std::process::exit(1);
                }
            }

            if let Err(e) = fs::create_dir_all(output_path) {
                eprintln!("Failed to create directory '{}': {}", output_dir, e);
                std::process::exit(1);
            }

            let db_path = output_path.join("results.db");
            let start = Instant::now();

            let global_state =
                match run_explorer_genetic(&program, &config, db_path.to_str().unwrap()) {
                    Ok(state) => state,
                    Err(e) => {
                        eprintln!("Explorer failed: {}", e);
                        std::process::exit(1);
                    }
                };

            let elapsed = start.elapsed();

            // Generate coverage heatmap
            let vertex_coverage: HashMap<usize, u64> = global_state
                .coverage
                .vertices()
                .iter()
                .map(|entry| (*entry.key(), *entry.value()))
                .collect();

            let byte_hits = vertex_coverage_to_byte_coverage(
                &vertex_coverage,
                &program.vertex_to_span,
                source_code.len(),
            );

            let html = render_html_heatmap(&source_code, &byte_hits);
            let heatmap_path = output_path.join("coverage.html");

            if let Err(e) = fs::write(&heatmap_path, html) {
                eprintln!("Failed to write coverage heatmap: {}", e);
            }

            // Generate CFG SVG
            let cfg_path = output_path.join("cfg.svg");
            match render_svg(&program) {
                Ok(svg) => {
                    if let Err(e) = fs::write(&cfg_path, svg) {
                        eprintln!("Failed to write CFG SVG: {}", e);
                    }
                }
                Err(e) => eprintln!("Failed to write CFG SVG: {}", e),
            };

            println!(
                "Explorer finished in {:.2?}. Results saved to {}",
                elapsed, output_dir
            );
            println!("  - Database: {}", db_path.display());
            println!("  - Coverage heatmap: {}", heatmap_path.display());
            println!("  - CFG: {}", cfg_path.display());
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

fn compile_spec_to_program_with_source(content: &str, spec: &str) -> Option<Program> {
    match compiler::compile(content, spec) {
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

    let content = match read_spec_file(spec) {
        Some(c) => c,
        None => std::process::exit(1),
    };

    let program = match compile_spec_to_program_with_source(&content, spec) {
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
