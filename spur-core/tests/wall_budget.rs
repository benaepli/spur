//! A wall budget has to end the session on the explorer's own clock and be
//! reported as the session's exposure.
//!
//! The consumer that normalises rung counts by time reads `session.wall_ms`,
//! so a session whose budget is ignored, or whose reported wall is the
//! deadline rather than the time the runs had, would turn a throughput
//! difference into a rate difference.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{SessionSummary, run_explorer};
use spur_core::simulator::history::LogBackend;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/kv.spur");

/// One configuration, so the grid alone is a known number of runs.
const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 1, "max": 1, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": RUNS,
  "max_iterations": 2000,
  "session_seed": 424242,
  "strict_config_keys": true,
  "wall_budget_sec": BUDGET
}"#;

/// Six configurations, so a cut has a grid to leave unbalanced.
const GRID_CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 1, "max": 3, "step": 1},
  "num_read_ops": {"min": 1, "max": 2, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 100000000,
  "max_iterations": 2000,
  "session_seed": 424243,
  "strict_config_keys": true,
  "wall_budget_sec": 2.0
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn config_index_counts(dir: &Path) -> HashMap<i32, u64> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::Int32Type;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let mut counts = HashMap::new();
    for entry in fs::read_dir(dir.join("runs")).expect("runs dir") {
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
            let idx = batch.column_by_name("config_index").unwrap().as_primitive::<Int32Type>();
            for i in 0..idx.len() {
                *counts.entry(idx.value(i)).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("spur_wall_budget").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

fn session(name: &str, runs: i32, budget: f64) -> SessionSummary {
    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join(format!("spur_wall_budget_{name}.json"));
    fs::write(
        &config_path,
        CONFIG
            .replace("RUNS", &runs.to_string())
            .replace("BUDGET", &budget.to_string()),
    )
    .expect("writes config");
    let out = scratch(name);
    let _scope = config_override::exclusive_session();
    let cancelled = Arc::new(AtomicBool::new(false));
    let summary = run_explorer(
        &program,
        config_path.to_str().expect("utf-8 path"),
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .expect("session runs");
    let _ = fs::remove_file(&config_path);
    summary.session.expect("the standard explorer reports its session")
}

#[test]
fn a_wall_budget_ends_the_session_and_is_the_reported_exposure() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_wall_budget)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_wall_budget() {
    let cut = session("budgeted", 100_000_000, 2.0);
    assert!(
        cut.budget_hit,
        "a grid far larger than the budget must be cut: {} runs in {} ms",
        cut.runs_completed, cut.wall_ms
    );
    assert!(cut.runs_completed > 0, "the budget left no run finished");
    assert!(
        cut.runs_completed < 100_000_000,
        "the budget did not cut the grid: {} runs completed",
        cut.runs_completed
    );
    assert!(
        cut.wall_ms >= 2_000 && cut.wall_ms < 5_000,
        "the reported wall must cover the budget and little more: {} ms",
        cut.wall_ms
    );

    let whole = session("unbounded", 3, 0.0);
    assert!(!whole.budget_hit);
    assert_eq!(whole.runs_completed, 3);
    assert_eq!(whole.runs_skipped, 0);

    // A cut has to leave every configuration within one run of every other,
    // or the corpus's composition would follow the throughput.
    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join("spur_wall_budget_grid.json");
    fs::write(&config_path, GRID_CONFIG).expect("writes config");
    let out = scratch("grid");
    {
        let _scope = config_override::exclusive_session();
        let cancelled = Arc::new(AtomicBool::new(false));
        let summary = run_explorer(
            &program,
            config_path.to_str().expect("utf-8 path"),
            out.to_str().expect("utf-8 path"),
            LogBackend::Parquet,
            &cancelled,
        )
        .expect("session runs");
        assert!(summary.session.expect("session").budget_hit);
    }
    let _ = fs::remove_file(&config_path);
    let counts = config_index_counts(&out);
    assert_eq!(counts.len(), 6, "every configuration of the grid was visited: {counts:?}");
    let min = counts.values().copied().min().unwrap_or(0);
    let max = counts.values().copied().max().unwrap_or(0);
    assert!(max - min <= 1, "a cut left the grid unbalanced: {counts:?}");

    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_wall_budget"));
}
