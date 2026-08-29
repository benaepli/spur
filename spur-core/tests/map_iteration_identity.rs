//! Two sessions of one config at one seed must write the same schedule. The
//! fixture walks a map on every timer firing, so the runtime's map iteration
//! order is part of the schedule; a hasher seeded per process would make the
//! two sessions differ.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::run_explorer;
use spur_core::simulator::history::LogBackend;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/mapiter.spur");

const CONFIG: &str = r#"{
  "num_servers": {"min": 2, "max": 2, "step": 1},
  "num_write_ops": {"min": 1, "max": 2, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 1, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 6,
  "max_iterations": 400,
  "session_seed": 9002,
  "rng_stream_isolation": true,
  "strict_config_keys": true
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("spur_map_iteration_identity").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

/// Every row of one table as a sorted list of its cells rendered as text,
/// with the columns that time a run left out.
fn table_rows(dir: &Path, table: &str, skip: &[&str]) -> Vec<String> {
    use arrow::util::display::{ArrayFormatter, FormatOptions};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let mut rows = Vec::new();
    for entry in fs::read_dir(dir.join(table)).expect("table dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        let reader = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let schema = batch.schema();
            let options = FormatOptions::default();
            let columns: Vec<(usize, ArrayFormatter)> = (0..batch.num_columns())
                .filter(|&i| !skip.contains(&schema.field(i).name().as_str()))
                .map(|i| (i, ArrayFormatter::try_new(batch.column(i), &options).unwrap()))
                .collect();
            for row in 0..batch.num_rows() {
                let cells: Vec<String> = columns.iter().map(|(_, f)| f.value(row).to_string()).collect();
                rows.push(cells.join("|"));
            }
        }
    }
    rows.sort();
    rows
}

fn session(name: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let program = compiler::compile(SPEC, "mapiter.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join(format!("spur_map_iteration_identity_{name}.json"));
    fs::write(&config_path, CONFIG).expect("writes config");
    let out = scratch(name);
    let _scope = config_override::exclusive_session();
    let cancelled = Arc::new(AtomicBool::new(false));
    run_explorer(
        &program,
        config_path.to_str().expect("utf-8 path"),
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .expect("session runs");
    let _ = fs::remove_file(&config_path);
    let executions = table_rows(&out, "executions", &[]);
    let traces = table_rows(&out, "traces", &[]);
    let runs = table_rows(&out, "runs", &["wall_us", "session_offset_ms"]);
    let _ = fs::remove_dir_all(&out);
    (executions, traces, runs)
}

fn first_difference(a: &[String], b: &[String]) -> String {
    match a.iter().zip(b.iter()).find(|(x, y)| x != y) {
        Some((x, y)) => format!("first differing rows:\n  {x}\n  {y}"),
        None => format!("row counts {} and {}", a.len(), b.len()),
    }
}

fn check() {
    // One worker makes a session a pure function of its seed, which is what
    // a reproducibility check compares.
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let first = session("first");
    assert!(!first.0.is_empty() && !first.2.is_empty(), "the session wrote nothing");
    assert!(
        first.1.iter().any(|row| row.contains("Node.Beat")),
        "the fixture never walked its map"
    );
    let second = session("second");
    assert!(
        first.0 == second.0,
        "two sessions of one config wrote different executions: {}",
        first_difference(&first.0, &second.0)
    );
    assert!(
        first.1 == second.1,
        "two sessions of one config wrote different traces: {}",
        first_difference(&first.1, &second.1)
    );
    assert!(
        first.2 == second.2,
        "two sessions of one config wrote different runs rows: {}",
        first_difference(&first.2, &second.2)
    );
}

#[test]
fn map_iteration_is_a_function_of_the_map() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}
