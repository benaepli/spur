//! Naming the score's terms must not move a single scheduling decision at
//! the default weights: a session with no `steer_terms` block, one with the
//! defaults spelled out, one that sets the recover multiplier through the
//! legacy key and one that sets it through the block must write the same
//! executions and the same runs table on a fixed seed.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::run_explorer;
use spur_core::simulator::history::LogBackend;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/relay.spur");

/// Two peers, faults on, a fixed seed with isolated draw streams, strict
/// keys. `{{TERMS}}` is where a session puts its terms and legacy keys.
const CONFIG: &str = r#"{
  "num_servers": {"min": 2, "max": 2, "step": 1},
  "num_write_ops": {"min": 1, "max": 2, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 1, "max": 2, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 6,
  "max_iterations": 400,
  "session_seed": 9001,
  "rng_stream_isolation": true,
  "strict_config_keys": true,
  "stats": true{{TERMS}}
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("spur_steer_terms_identity").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

/// Every row of one table as a sorted list of its cells rendered as text,
/// with the columns that time a run left out.
fn table_rows(dir: &Path, table: &str, skip: &[&str]) -> Vec<String> {
    use arrow::util::display::ArrayFormatter;
    use arrow::util::display::FormatOptions;
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
            let opts = FormatOptions::default();
            let columns: Vec<(String, ArrayFormatter)> = batch
                .schema()
                .fields()
                .iter()
                .zip(batch.columns().iter())
                .filter(|(f, _)| !skip.contains(&f.name().as_str()))
                .map(|(f, c)| (f.name().clone(), ArrayFormatter::try_new(c.as_ref(), &opts).unwrap()))
                .collect();
            for i in 0..batch.num_rows() {
                let mut cells: Vec<String> = columns
                    .iter()
                    .map(|(n, f)| format!("{n}={}", f.value(i)))
                    .collect();
                cells.sort();
                rows.push(cells.join("|"));
            }
        }
    }
    rows.sort();
    rows
}

fn session(name: &str, terms: &str) -> (Vec<String>, Vec<String>) {
    let program = compiler::compile(SPEC, "relay.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join(format!("spur_steer_terms_identity_{name}.json"));
    fs::write(&config_path, CONFIG.replace("{{TERMS}}", terms)).expect("writes config");
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
    let runs = table_rows(&out, "runs", &["wall_us", "session_offset_ms"]);
    let _ = fs::remove_dir_all(&out);
    (executions, runs)
}

#[test]
fn default_terms_leave_every_run_unchanged() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn first_difference(a: &[String], b: &[String]) -> String {
    match a.iter().zip(b.iter()).find(|(x, y)| x != y) {
        Some((x, y)) => format!("first differing rows:\n  {x}\n  {y}"),
        None => format!("row counts {} and {}", a.len(), b.len()),
    }
}

fn check() {
    // Runs share a feedback store and a dedup set, so their order across
    // worker threads changes what later runs see; one worker makes a session
    // a pure function of its seed, which is what an identity check compares.
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let plain = session("plain", "");
    assert!(!plain.0.is_empty() && !plain.1.is_empty(), "the session wrote nothing");
    let again = session("again", "");
    assert!(
        plain.0 == again.0 && plain.1 == again.1,
        "two sessions of the same config differ, so no identity can be read: {}",
        first_difference(&plain.0, &again.0)
    );
    let explicit = session(
        "explicit",
        r#",
  "steer_terms": {"novelty": 0.25, "priority": 0.75, "crash_after_timer_sends": 0,
                  "crash_after_delivery_sends": 0, "stale_late": 0, "request_before_stale": 0}"#,
    );
    assert!(plain.0 == explicit.0, "executions differ with the default block spelled out: {}", first_difference(&plain.0, &explicit.0));
    assert_eq!(plain.1, explicit.1, "runs differ with the default block spelled out");

    let legacy = session("legacy", r#",
  "quick_fire_multiplier": 3.0"#);
    let named = session("named", r#",
  "steer_terms": {"recover_crashed": 3.0}"#);
    assert!(legacy.0 == named.0, "executions differ between the legacy key and the block: {}", first_difference(&legacy.0, &named.0));
    assert_eq!(legacy.1, named.1, "runs differ between the legacy key and the block");
    assert_ne!(
        plain.1.len() + plain.0.len(),
        0,
        "the sessions must have produced rows to compare"
    );
    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_steer_terms_identity"));
}
