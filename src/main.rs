use std::fs;
use std::path::Path;
use clap::Parser;
use turnpike::compiler;

#[derive(Parser)]
#[command(name = "program")]
#[command(about = "Program description", long_about = None)]
struct Args {
    /// Input specification file
    spec: String,

    /// Intermediate output file
    intermediate_output: String,

    /// Scheduler configuration JSON file
    scheduler_config_json: String,
}

fn main() {
    let args = Args::parse();
    println!(
        "Input spec: {}, intermediate output: {}, scheduler_config_json: {}",
        args.spec, args.intermediate_output, args.scheduler_config_json
    );
    
    let path = Path::new(&args.spec);
    let content = fs::read_to_string(path).expect("Unable to read file");
    let _ = compiler::compile(&content, &args.spec);
}
