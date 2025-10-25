use std::fs;
use std::path::Path;
use clap::Parser;
use spur::compiler;

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
    let content = fs::read_to_string(path).expect("Unable to read file");

    match compiler::compile(&content, &args.spec) {
        Ok(program) => {
            let json = serde_json::to_string_pretty(&program)
                .expect("Failed to serialize program");
            fs::write(&args.output, json).expect("Failed to write output file");
            println!("Successfully compiled to {}", args.output);
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            std::process::exit(1);
        }
    }
}
