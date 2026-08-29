//! Every predicate has to be seen on the relay fixture: at zero weight the
//! counters say each one was present without the choice ever changing or a
//! routing draw being made, and with weights on, the choices flip, the
//! router routes, and the schedule differs from the zero-weight session.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::run_explorer;
use spur_core::simulator::history::LogBackend;
use spur_core::simulator::util_stats::{self, SteerTermStats};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/relay.spur");

const CONFIG: &str = r#"{
  "num_servers": {"min": 2, "max": 2, "step": 1},
  "num_write_ops": {"min": 2, "max": 2, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 1, "max": 3, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 30,
  "max_iterations": 800,
  "session_seed": 4242,
  "queue_policy": {"type": "Probabilistic", "p_local": 0.7, "p_timer": 0.2},
  "rng_stream_isolation": true,
  "strict_config_keys": true,
  "stats": true{{TERMS}}
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("spur_steer_terms_fire").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

/// The `(run_id, seq_num, action)` rows of the executions table, sorted.
fn executions(dir: &Path) -> Vec<(i64, i64, String)> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::Int64Type;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let mut rows = Vec::new();
    for entry in fs::read_dir(dir.join("executions")).expect("table dir") {
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
            let ids = batch.column_by_name("run_id").unwrap().as_primitive::<Int64Type>();
            let seqs = batch.column_by_name("seq_num").unwrap().as_primitive::<Int64Type>();
            let actions = batch.column_by_name("action").unwrap().as_string::<i32>();
            for i in 0..actions.len() {
                rows.push((ids.value(i), seqs.value(i), actions.value(i).to_string()));
            }
        }
    }
    rows.sort();
    rows
}

fn session(name: &str, terms: &str) -> (SteerTermStats, Vec<(i64, i64, String)>) {
    let program = compiler::compile(SPEC, "relay.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join(format!("spur_steer_terms_fire_{name}.json"));
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
    let stats = util_stats::snapshot().steer_terms;
    let _ = fs::remove_file(&config_path);
    let rows = executions(&out);
    let _ = fs::remove_dir_all(&out);
    (stats, rows)
}

#[test]
fn every_predicate_is_seen_and_weights_change_the_schedule() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check() {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let (zero, zero_rows) = session("zero", "");
    assert!(zero.decisions > 0, "no selection was counted");
    let mut missing = Vec::new();
    for (name, c) in [
        ("crash_after_timer_sends", &zero.crash_after_timer_sends),
        ("crash_after_delivery_sends", &zero.crash_after_delivery_sends),
        ("stale_late", &zero.stale_late),
        ("request_before_stale", &zero.request_before_stale),
    ] {
        if c.present == 0 || c.evaluated == 0 {
            missing.push(format!("{name}: {c:?}"));
        }
        assert_eq!(c.flipped, 0, "{name} flipped a choice at zero weight");
    }
    assert!(missing.is_empty(), "predicates never seen at zero weight: {missing:?}; all: {zero:?}");
    assert_eq!(zero.authority_draws, 0, "the router drew at zero weight");
    assert_eq!(zero.authority_routed, 0);

    let (on, on_rows) = session(
        "weighted",
        r#",
  "steer_terms": {"stale_late": 2.33, "crash_after_timer_sends": 2.33}"#,
    );
    assert!(on.stale_late.won > 0, "stale_late never won: {:?}", on.stale_late);
    assert!(
        on.stale_late.flipped + on.crash_after_timer_sends.flipped > 0,
        "the weights never changed a within-queue choice: {:?} {:?}",
        on.stale_late,
        on.crash_after_timer_sends
    );
    assert!(on.authority_draws > 0, "the router never drew");
    assert!(on.authority_routed > 0, "the router never routed");
    assert!(
        on.stale_late.measured > 0,
        "a chosen stale delivery was never measured for its effect"
    );
    assert_ne!(zero_rows, on_rows, "weights left every execution row unchanged");
    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_steer_terms_fire"));
}
