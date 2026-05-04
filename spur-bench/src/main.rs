use clap::Parser;
use spur_bench::{BenchConfig, generate_source};

#[derive(Parser)]
#[command(about = "Generate .spur files for refinement type checker benchmarks")]
struct Cli {
    /// Number of functions to generate.
    #[arg(long, default_value_t = 50)]
    num_functions: usize,

    /// Complexity level per function (1-5). 0 = round-robin mixed.
    #[arg(long, default_value_t = 0)]
    complexity: u8,
}

fn main() {
    let cli = Cli::parse();
    let source = generate_source(&BenchConfig {
        num_functions: cli.num_functions,
        complexity: cli.complexity,
    });
    print!("{source}");
}
