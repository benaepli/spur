//! The allocation has to act on the reward it reads, and every run has to
//! be attributed to the arm that issued it.
//!
//! Two grid arms differ only in their step budget: one finishes its plans,
//! the other is cut off after four steps and never does. Under a
//! completion reward the halving drops the starved arm at the first round;
//! under a run-count reward it keeps it, because short runs are many. The
//! runs table names the arm on every row.

use spur_core::compiler;
use spur_core::simulator::campaign::run_explorer_campaign;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::ExploreSummary;
use spur_core::simulator::history::LogBackend;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/kv.spur");

const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 1, "max": 2, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 4,
  "max_iterations": 2000,
  "session_seed": 424242,
  "stats": true,
  "strict_config_keys": true,
  "campaign": {
    "wall_budget_sec": 60,
    "deterministic_slice_runs": 16,
    "deterministic_rounds": 3,
    "batch_size": 8,
    "allocation": ALLOCATION,
    "reward": {"kind": "REWARD"},
    "arms": [
      {"id": "grid", "mode": "grid"},
      {"id": "starved", "mode": "grid", "overlay": {"max_iterations": 4}}
    ]
  }
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("spur_campaign_allocation")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

fn session(name: &str, allocation: &str, reward: &str) -> (ExploreSummary, PathBuf) {
    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join(format!("spur_campaign_{name}.json"));
    fs::write(
        &config_path,
        CONFIG
            .replace("ALLOCATION", allocation)
            .replace("REWARD", reward),
    )
    .expect("writes config");
    let out = scratch(name);
    let _scope = config_override::exclusive_session();
    let cancelled = Arc::new(AtomicBool::new(false));
    let summary = run_explorer_campaign(
        &program,
        config_path.to_str().expect("utf-8 path"),
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .expect("campaign runs");
    let _ = fs::remove_file(&config_path);
    (summary, out)
}

/// A session on the clock rather than in runs: the deterministic keys are
/// dropped and the budget is `wall_sec`.
fn session_timed(
    name: &str,
    allocation: &str,
    reward: &str,
    wall_sec: f64,
) -> (ExploreSummary, PathBuf) {
    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join(format!("spur_campaign_{name}.json"));
    fs::write(
        &config_path,
        CONFIG
            .replace("ALLOCATION", allocation)
            .replace("REWARD", reward)
            .replace("\"deterministic_slice_runs\": 16,", "")
            .replace("\"deterministic_rounds\": 3,", "")
            .replace(
                "\"wall_budget_sec\": 60,",
                &format!("\"wall_budget_sec\": {wall_sec},"),
            ),
    )
    .expect("writes config");
    let out = scratch(name);
    let _scope = config_override::exclusive_session();
    let cancelled = Arc::new(AtomicBool::new(false));
    let summary = run_explorer_campaign(
        &program,
        config_path.to_str().expect("utf-8 path"),
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .expect("campaign runs");
    let _ = fs::remove_file(&config_path);
    (summary, out)
}

fn arm_counts_in_runs_table(dir: &Path) -> HashMap<String, u64> {
    use arrow::array::{Array, AsArray};
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
            let arms = batch.column_by_name("arm").unwrap().as_string::<i32>();
            for i in 0..arms.len() {
                *counts.entry(arms.value(i).to_string()).or_insert(0) += 1;
            }
        }
    }
    counts
}

#[test]
fn the_reward_decides_which_arm_halving_keeps() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_allocations)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_allocations() {
    let halving = r#"{"kind": "halving", "eta": 2.0, "keep_top": 1}"#;

    let (s, out) = session("halving_completed", halving, "termination_completed");
    let report = s.campaign.expect("a campaign reports");
    assert_eq!(report.arms.len(), 2);
    assert_eq!(
        report.arms[1].dropped_at_round,
        Some(0),
        "the arm that never finishes a plan must be dropped at round 0: {:?}",
        report.arms
    );
    assert_eq!(report.arms[0].dropped_at_round, None);
    assert!(report.arms[0].runs > report.arms[1].runs);
    assert!(
        report
            .history
            .iter()
            .filter(|h| h.round > 0)
            .all(|h| h.arm == 0),
        "every slice after round 0 belongs to the survivor"
    );
    let counts = arm_counts_in_runs_table(&out);
    assert_eq!(
        counts.get("grid").copied(),
        Some(report.arms[0].runs),
        "{counts:?}"
    );
    assert_eq!(
        counts.get("starved").copied(),
        Some(report.arms[1].runs),
        "{counts:?}"
    );
    let sess = s.session.expect("a campaign reports its session");
    assert_eq!(sess.runs_completed, report.runs_total);

    // On the clock, the starved arm finishes many more runs per second, so a
    // run-count reward keeps it and drops the other.
    let (s, _) = session_timed(
        "halving_runs",
        r#"{"kind": "halving", "eta": 2.0, "keep_top": 1, "min_slice_sec": 1}"#,
        "runs",
        6.0,
    );
    let report = s.campaign.expect("a campaign reports");
    assert_eq!(
        report.arms[0].dropped_at_round,
        Some(0),
        "under a run-count reward on the clock the short-run arm wins: {:?}",
        report.arms
    );
    assert_eq!(report.arms[1].dropped_at_round, None);

    let (s, _) = session("round_robin", r#"{"kind": "round_robin"}"#, "runs");
    let report = s.campaign.expect("a campaign reports");
    assert_eq!(report.arms[0].slices, report.arms[1].slices);
    assert_eq!(report.arms[0].runs, report.arms[1].runs);
    assert_eq!(report.arms[0].runs, 3 * 16);

    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_campaign_allocation"));
}

#[test]
fn a_cancelled_campaign_stops_promptly() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_cancel)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_cancel() {
    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join("spur_campaign_cancel.json");
    fs::write(
        &config_path,
        CONFIG
            .replace(
                "ALLOCATION",
                r#"{"kind": "round_robin", "min_slice_sec": 1}"#,
            )
            .replace("REWARD", "runs")
            .replace("\"deterministic_slice_runs\": 16,", "")
            .replace("\"deterministic_rounds\": 3,", ""),
    )
    .expect("writes config");
    let out = scratch("cancel");
    let _scope = config_override::exclusive_session();
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = cancelled.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let started = std::time::Instant::now();
    let summary = run_explorer_campaign(
        &program,
        config_path.to_str().expect("utf-8 path"),
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .expect("campaign runs");
    assert!(
        started.elapsed().as_secs_f64() < 10.0,
        "cancellation must end the session well inside its budget"
    );
    assert!(summary.campaign.expect("reports").cancelled);
    let _ = fs::remove_file(&config_path);
    let _ = fs::remove_dir_all(&out);
}
