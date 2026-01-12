use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use spur::compiler;
use spur::debug::SimulatorDebugger;
use spur::simulator::explorer::run_explorer_genetic;
use spur::visualization::{render_html_heatmap, render_svg, vertex_coverage_to_byte_coverage};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "spur")]
#[command(about = "A compiler and simulator for the spur language.", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a specification file to JSON
    Compile {
        /// Input specification file (.spur)
        spec: PathBuf,
        /// Compiled output file (.json)
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Check a specification file for errors
    Check {
        /// Input specification file (.spur)
        spec: PathBuf,
    },
    /// Generate a CFG visualization for a specification
    Graph {
        /// Input specification file (.spur)
        spec: PathBuf,
        /// Output path for the SVG graph
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compile and run the execution explorer
    Explore {
        /// Input specification file (.spur)
        spec: PathBuf,
        /// Explorer configuration JSON file
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory for results
        #[arg(short, long)]
        output_dir: PathBuf,
        /// Skip confirmation prompt for directory deletion
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Debug simulation results
    Debug(DebugArgs),
}

#[derive(Parser)]
pub struct DebugArgs {
    #[command(subcommand)]
    pub command: DebugSubcommands,
}

#[derive(Subcommand)]
pub enum DebugSubcommands {
    /// Show logs for a specific node in a run
    Logs {
        /// Path to the results database
        #[arg(short, long)]
        db: PathBuf,
        /// Run ID
        #[arg(long)]
        run_id: i64,
        /// Node ID
        #[arg(long)]
        node_id: Option<i64>,
    },
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let result = match args.command {
        Commands::Compile { spec, output } => run_compile(spec, output),
        Commands::Check { spec } => run_check(spec),
        Commands::Graph { spec, output } => run_graph(spec, output),
        Commands::Explore {
            spec,
            config,
            output_dir,
            yes,
        } => run_explore(spec, config, output_dir, yes),
        Commands::Debug(args) => match args.command {
            DebugSubcommands::Logs {
                db,
                run_id,
                node_id,
            } => run_debug_logs(db, run_id, node_id),
        },
    };
    if result.is_err() {
        println!("Result: {}", result.err().unwrap().to_string());
    }
}

fn run_compile(spec_path: PathBuf, output_path: PathBuf) -> Result<()> {
    let source_code = fs::read_to_string(&spec_path)
        .with_context(|| format!("Failed to read spec file: {}", spec_path.display()))?;

    let program = compiler::compile(&source_code, spec_path.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("Compilation failed: {}", e))?;

    let json =
        serde_json::to_string_pretty(&program).context("Failed to serialize program to JSON")?;

    fs::write(&output_path, json)
        .with_context(|| format!("Failed to write output to {}", output_path.display()))?;

    println!("Successfully compiled to {}", output_path.display());
    Ok(())
}

fn run_check(spec_path: PathBuf) -> Result<()> {
    let source_code = fs::read_to_string(&spec_path)
        .with_context(|| format!("Failed to read spec file: {}", spec_path.display()))?;

    compiler::compile(&source_code, spec_path.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("Check failed: {}", e))?;

    println!("Successfully checked {}", spec_path.display());
    Ok(())
}

fn run_graph(spec_path: PathBuf, output_path: PathBuf) -> Result<()> {
    let source_code = fs::read_to_string(&spec_path)
        .with_context(|| format!("Failed to read spec file: {}", spec_path.display()))?;

    let program = compiler::compile(&source_code, spec_path.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("Compilation failed: {}", e))?;

    let svg = render_svg(&program).map_err(|e| anyhow::anyhow!("Failed to render SVG: {}", e))?;

    fs::write(&output_path, svg)
        .with_context(|| format!("Failed to write SVG to {}", output_path.display()))?;

    println!("Successfully generated graph at {}", output_path.display());
    Ok(())
}

fn run_explore(
    spec_path: PathBuf,
    config_path: PathBuf,
    output_dir: PathBuf,
    yes: bool,
) -> Result<()> {
    let source_code = fs::read_to_string(&spec_path)
        .with_context(|| format!("Failed to read spec file: {}", spec_path.display()))?;

    let program = compiler::compile(&source_code, spec_path.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("Compilation failed: {}", e))?;

    // Handle output directory
    if output_dir.exists() {
        if !yes {
            print!(
                "Output directory '{}' already exists and will be deleted. Continue? [y/N] ",
                output_dir.display()
            );
            io::stdout().flush().context("Failed to flush stdout")?;

            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .context("Failed to read from stdin")?;
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Aborted.");
                return Ok(());
            }
        }
        fs::remove_dir_all(&output_dir)
            .with_context(|| format!("Failed to remove directory '{}'", output_dir.display()))?;
    }

    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create directory '{}'", output_dir.display()))?;

    let db_path = output_dir.join("results.db");
    let start = Instant::now();

    let db_path_str = db_path
        .to_str()
        .context("Database path contains invalid UTF-8")?;

    let config_path_str = config_path
        .to_str()
        .context("Config path contains invalid UTF-8")?;

    let global_state = run_explorer_genetic(&program, config_path_str, db_path_str)
        .map_err(|e| anyhow::anyhow!("Explorer failed: {}", e))?;

    let elapsed = start.elapsed();

    // Generate coverage heatmap
    let vertex_coverage: HashMap<usize, u64> = global_state
        .coverage
        .vertices_snapshot()
        .into_iter()
        .collect();

    let byte_hits = vertex_coverage_to_byte_coverage(
        &vertex_coverage,
        &program.vertex_to_span,
        source_code.len(),
    );

    let html = render_html_heatmap(&source_code, &byte_hits);
    let heatmap_path = output_dir.join("coverage.html");

    fs::write(&heatmap_path, html).context("Failed to write coverage heatmap")?;

    // Generate CFG SVG
    let cfg_path = output_dir.join("cfg.svg");
    let svg = render_svg(&program).map_err(|e| anyhow::anyhow!("Failed to render SVG: {}", e))?;
    fs::write(&cfg_path, svg).context("Failed to write CFG SVG")?;

    println!(
        "Explorer finished in {:.2?}. Results saved to {}",
        elapsed,
        output_dir.display()
    );
    println!("  - Database: {}", db_path.display());
    println!("  - Coverage heatmap: {}", heatmap_path.display());
    println!("  - CFG: {}", cfg_path.display());

    Ok(())
}

fn run_debug_logs(db_path: PathBuf, run_id: i64, node_id: Option<i64>) -> Result<()> {
    let debugger = SimulatorDebugger::new(&db_path)
        .map_err(|e| anyhow::anyhow!("Failed to open database: {}", e))?;

    if let Some(node_id) = node_id {
        let logs = debugger
            .get_node_timeline(run_id, node_id)
            .map_err(|e| anyhow::anyhow!("Failed to fetch logs: {}", e))?;

        if logs.is_empty() {
            println!("No logs found for run {} and node {}.", run_id, node_id);
            return Ok(());
        }

        println!("Logs for Run {}, Node {}:", run_id, node_id);
        println!("{:-<40}", "");
        for (step, content) in logs {
            println!("[Step {:4}] {}", step, content);
        }
        println!("{:-<40}", "");
    } else {
        let logs = debugger
            .get_all_logs(run_id)
            .map_err(|e| anyhow::anyhow!("Failed to fetch logs: {}", e))?;

        if logs.is_empty() {
            println!("No logs found for run {}.", run_id);
            return Ok(());
        }

        println!("All logs for Run {}:", run_id);
        println!("{:-<60}", "");
        for (step, node_id, content) in logs {
            let node_str = node_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "SYS".to_string());
            println!("[Step {:4}] [Node {:>3}] {}", step, node_str, content);
        }
        println!("{:-<60}", "");
    }

    Ok(())
}
