use clap::Parser;
use spur::compiler;
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "spur-frontend")]
#[command(about = "A compiler for spur.", long_about = None)]
struct Args {
    /// Input specification file
    spec: String,
    /// Compiled output file
    output: String,
}

fn main() {
    let args = Args::parse();
    println!(
        "Input spec: {}, Output location: {}",
        args.spec, args.output
    );

    let path = Path::new(&args.spec);

    if !path.exists() {
        eprintln!("Error: Input file '{}' does not exist", args.spec);
        std::process::exit(1);
    }

    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error: Failed to read input file '{}': {}", args.spec, e);
            std::process::exit(1);
        }
    };

    match compiler::compile(&content, &args.spec) {
        Ok(program) => {
            let json = serde_json::to_string_pretty(&program).expect("Failed to serialize program");

            if let Err(e) = fs::write(&args.output, json) {
                eprintln!(
                    "Error: Failed to write output file '{}': {}",
                    args.output, e
                );
                eprintln!("Possible causes:");
                eprintln!("  - Directory does not exist");
                eprintln!("  - Insufficient permissions");
                eprintln!("  - Disk full or read-only filesystem");
                std::process::exit(1);
            }

            println!("Successfully compiled to {}", args.output);
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            std::process::exit(1);
        }
    }
}
