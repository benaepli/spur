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
use std::fs;
use std::path::PathBuf;
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

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

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

    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_wall_budget"));
}
