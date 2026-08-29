//! Every timer firing that wakes a waiting record is accounted on the run
//! that fired it, split by whether the woken segment changed the node's
//! state and whether a delivery to the node was pending.
//!
//! The consumer joins these columns against depth and the end reason per
//! run, so a firing counted twice, or a firing that woke a record and went
//! uncounted, would bias that join.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::run_explorer;
use spur_core::simulator::history::LogBackend;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/timer.spur");

const CONFIG: &str = r#"{
  "num_servers": {"min": 2, "max": 2, "step": 1},
  "num_write_ops": {"min": 1, "max": 1, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 1, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 12,
  "max_iterations": 400,
  "session_seed": 777,
  "strict_config_keys": true,
  "stats": true,
  "emit_acted_fraction": true
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join("spur_timer_effects");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

fn int_columns(dir: &Path, table: &str, columns: &[&str]) -> Vec<HashMap<String, i64>> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{DataType, Int32Type, Int64Type};
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
            for i in 0..batch.num_rows() {
                let mut row = HashMap::new();
                for &c in columns {
                    let col = batch.column_by_name(c).unwrap_or_else(|| panic!("column {c}"));
                    let v = match col.data_type() {
                        DataType::Int32 => col.as_primitive::<Int32Type>().value(i) as i64,
                        DataType::Int64 => col.as_primitive::<Int64Type>().value(i),
                        other => panic!("column {c} is {other:?}"),
                    };
                    row.insert(c.to_string(), v);
                }
                rows.push(row);
            }
        }
    }
    rows
}

fn string_column(dir: &Path, table: &str, id: &str, column: &str) -> Vec<(i64, String)> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::Int64Type;
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
            let ids = batch.column_by_name(id).unwrap().as_primitive::<Int64Type>();
            let vals = batch.column_by_name(column).unwrap().as_string::<i32>();
            for i in 0..vals.len() {
                rows.push((ids.value(i), vals.value(i).to_string()));
            }
        }
    }
    rows
}

#[test]
fn timer_firings_are_counted_once_per_run_and_split_by_effect() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check() {
    let program = compiler::compile(SPEC, "timer.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join("spur_timer_effects.json");
    fs::write(&config_path, CONFIG).expect("writes config");
    let out = scratch();
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

    let runs = int_columns(
        &out,
        "runs",
        &[
            "run_id",
            "timers_fired",
            "timers_acted",
            "timers_inflight_fired",
            "timers_inflight_acted",
            "timers_idle_fired",
            "timers_idle_acted",
            "max_inert_streak",
        ],
    );
    assert_eq!(runs.len(), 24, "one runs row per run");

    let mut fired_rows: HashMap<i64, i64> = HashMap::new();
    for (run_id, action) in string_column(&out, "executions", "run_id", "action") {
        if action.starts_with("System.TimerFired/") {
            *fired_rows.entry(run_id).or_insert(0) += 1;
        }
    }
    for (client_id, kind) in string_column(&out, "executions", "client_id", "kind") {
        if kind == "TimerFired" {
            assert!(client_id >= 0, "a timer row names its node in client_id");
        }
    }

    // A firing is recorded as an execution row when it happens and counted
    // on the run when the record it woke runs; a wake still queued when the
    // run ends is a row without a count, and there is at most one per timer
    // loop per node.
    const PENDING_WAKES_PER_RUN: i64 = 4;
    let mut total_fired = 0;
    let mut total_rows = 0;
    let mut total_acted = 0;
    for r in &runs {
        let id = r["run_id"];
        let fired = r["timers_fired"];
        let rows = fired_rows.get(&id).copied().unwrap_or(0);
        assert!(
            fired <= rows && fired + PENDING_WAKES_PER_RUN >= rows,
            "run {id}: {fired} firings counted against {rows} TimerFired rows"
        );
        total_rows += rows;
        assert_eq!(
            r["timers_inflight_fired"] + r["timers_idle_fired"],
            fired,
            "run {id}: in-flight and idle firings partition the total"
        );
        assert_eq!(r["timers_inflight_acted"] + r["timers_idle_acted"], r["timers_acted"]);
        assert!(r["timers_acted"] <= fired && r["timers_inflight_acted"] <= r["timers_inflight_fired"]);
        assert!(r["max_inert_streak"] <= fired);
        total_fired += fired;
        total_acted += r["timers_acted"];
    }
    assert!(total_fired > 0, "the fixture's timer loops fired nothing");
    assert!(total_fired * 2 > total_rows, "most firings must reach their woken record ({total_fired} of {total_rows})");
    assert!(
        total_acted > 0 && total_acted < total_fired,
        "one loop writes state and one does not, so acted must be strictly between 0 and fired ({total_acted}/{total_fired})"
    );
    let _ = fs::remove_dir_all(&out);
}
