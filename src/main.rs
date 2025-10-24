use std::fs;
use std::path::Path;
use clap::Parser;
use turnpike::compiler;

#[derive(Parser)]
#[command(name = "turnpike-frontend")]
#[command(about = "A compiler for Turnpike.", long_about = None)]
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
    let _ = compiler::compile(&content, &args.spec);
}
