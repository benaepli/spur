//! A run stopped by the learned step cap must leave the same artifacts an
//! exhausted run leaves: a runs-table row naming the cap exit and the steps
//! it granted, and its history rows flushed for the linearizability checker.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, RunOutcome, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::run_cap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/kv.spur");

/// A workload with far more client operations than the seeded cap allows
/// steps, so the run cannot complete before the cap ends it.
const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 10, "max": 10, "step": 1},
  "num_read_ops": {"min": 10, "max": 10, "step": 1},
  "num_keys": {"min": 2, "max": 2, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 1,
  "max_iterations": 256,
  "session_seed": 4242,
  "strict_config_keys": true,
  "stats": true
}"#;

const BACKUP: i32 = 256;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join("spur_run_cap_exit");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

fn int_column(dir: &Path, table: &str, column: &str) -> Vec<i64> {
    use arrow::array::AsArray;
    use arrow::datatypes::{DataType, Int32Type, Int64Type};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let mut out = Vec::new();
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
            let col = batch
                .column_by_name(column)
                .unwrap_or_else(|| panic!("column {column}"));
            for i in 0..batch.num_rows() {
                out.push(match col.data_type() {
                    DataType::Int32 => col.as_primitive::<Int32Type>().value(i) as i64,
                    DataType::Int64 => col.as_primitive::<Int64Type>().value(i),
                    other => panic!("column {column} is {other:?}"),
                });
            }
        }
    }
    out
}

fn string_column(dir: &Path, table: &str, column: &str) -> Vec<String> {
    use arrow::array::{Array, AsArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let mut out = Vec::new();
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
            let vals = batch.column_by_name(column).unwrap().as_string::<i32>();
            for i in 0..vals.len() {
                out.push(vals.value(i).to_string());
            }
        }
    }
    out
}

#[test]
fn a_capped_run_exits_through_the_learned_cap_and_flushes_its_rows() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check() {
    let _serial = config_override::exclusive_session();
    run_cap::reset();
    // 200 completed probes at 10 steps clear the sample floor; with a bucket
    // width of 1 the quantile's upper edge is 10, so the cap is 15.
    for _ in 0..200 {
        run_cap::merge_probe(BACKUP, run_cap::Outcome::Completed, 10);
    }
    let cap = run_cap::effective_cap(BACKUP);
    assert_eq!(cap, 15, "the seeded probes must engage a tiny cap");

    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let config: ExplorerConfig = serde_json::from_str(CONFIG).expect("config parses");
    let run_config = config
        .expand_grid()
        .into_iter()
        .next()
        .expect("the grid holds one config");
    assert_eq!(run_config.max_iterations, BACKUP);

    let out = scratch();
    let writer: Arc<dyn HistoryWriter> = Arc::from(
        create_writer(LogBackend::Parquet, out.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    );
    let global_state = GlobalState::<NoFeedback>::new();
    // Any run id off the probe period gets the learned cap.
    let run_id: i64 = 1;
    assert!(!run_cap::is_probe(run_id));
    let result = run_single_simulation::<NoFeedback, LiveRng>(
        &program,
        &writer,
        &global_state,
        run_id,
        &run_config,
        &Default::default(),
        7,
        11,
        None,
        &RunAttribution::mode("test"),
    )
    .expect("the run executes");
    writer.shutdown();
    run_cap::reset();

    match result.outcome {
        RunOutcome::LearnedCapReached { cap: got, .. } => {
            assert_eq!(got, cap, "the exit carries the cap that stopped the run")
        }
        other => panic!("expected a learned-cap exit, got {other:?}"),
    }

    let run_ids = int_column(&out, "runs", "run_id");
    assert_eq!(run_ids, vec![run_id], "one runs row for the capped run");
    assert_eq!(
        string_column(&out, "runs", "end_reason"),
        vec!["learned_cap_reached".to_string()]
    );
    assert_eq!(
        int_column(&out, "runs", "steps_used"),
        vec![cap as i64],
        "steps_used records what the cap granted"
    );
    let history_rows = int_column(&out, "executions", "run_id");
    assert!(
        history_rows.iter().all(|&id| id == run_id) && !history_rows.is_empty(),
        "the capped run's history rows must flush"
    );
    let _ = fs::remove_dir_all(&out);
}
