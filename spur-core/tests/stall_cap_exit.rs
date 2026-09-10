//! A run the stall cap ends must stop on the first step whose quiet gap
//! exceeds the cap, leave a runs-table row naming the exit and the steps it
//! ran, and flush its history rows. An untreated run with the same gap runs
//! to its budget and reports the gap against the standing cap; a probe of
//! either stream runs to its budget and feeds nothing.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, RunOutcome, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::run_cap;
use spur_core::simulator::stall_cap;
use spur_core::simulator::timer_context;
use spur_core::simulator::util_stats::{self, StallCapCell};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/stall.spur");

/// One write that never succeeds, so every step after the request is issued
/// is quiet and the run can only end by a cap or its budget.
const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 1, "max": 1, "step": 1},
  "num_read_ops": {"min": 0, "max": 0, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 1,
  "max_iterations": 256,
  "session_seed": 4343,
  "strict_config_keys": true,
  "stats": true
}"#;

const BACKUP: i32 = 256;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join("spur_stall_cap_exit");
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

/// The first run id at or above 1 in the given cell, with the run-cap probe
/// role as asked.
fn first_id(cell: StallCapCell, run_cap_probe: bool) -> i64 {
    (1..10_000i64)
        .find(|&id| stall_cap::cell(id) == cell && run_cap::is_probe(id) == run_cap_probe)
        .expect("the cell is reachable")
}

#[test]
fn a_treated_run_is_cut_when_its_gap_exceeds_the_cap_and_the_others_are_not() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check() {
    let _serial = config_override::exclusive_session();
    util_stats::set_enabled(true);
    run_cap::reset();
    stall_cap::reset();
    // 200 completed probes with a longest gap of 10 clear the sample floor;
    // with a bucket width of 1 the quantile's upper edge is 10, so the cap
    // is 15.
    for _ in 0..200 {
        stall_cap::merge_probe(BACKUP, 10);
    }
    let cap = stall_cap::effective_cap(BACKUP).expect("the seeded probes engage the cap");
    assert_eq!(cap, 15);
    assert_eq!(run_cap::effective_cap(BACKUP), BACKUP, "the step cap stays identity");

    let program = compiler::compile(SPEC, "stall.spur")
        .into_program()
        .expect("spec compiles");
    let config: ExplorerConfig = serde_json::from_str(CONFIG).expect("config parses");
    let run_config = config
        .expand_grid()
        .into_iter()
        .next()
        .expect("the grid holds one config");
    assert_eq!(run_config.max_iterations, BACKUP);

    let treated = first_id(StallCapCell::Treated, false);
    let untreated = first_id(StallCapCell::Untreated, false);
    let cap_probe = first_id(StallCapCell::Probe, true);
    let timer_probe = (1..10_000i64)
        .find(|&id| {
            timer_context::run_mode(id) == timer_context::RunMode::Probe
                && !run_cap::is_probe(id)
        })
        .expect("a timer-context probe exists");
    assert_eq!(stall_cap::cell(timer_probe), StallCapCell::Probe);

    let out = scratch();
    let writer: Arc<dyn HistoryWriter> = Arc::from(
        create_writer(LogBackend::Parquet, out.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    );
    let global_state = GlobalState::<NoFeedback>::new();
    let mut outcomes = Vec::new();
    let before = util_stats::snapshot().stall_cap;
    for run_id in [treated, untreated, cap_probe, timer_probe] {
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
        outcomes.push((run_id, result.outcome));
    }
    writer.shutdown();
    let after = util_stats::snapshot().stall_cap;
    let rows = stall_cap::render_run_rows().expect("the untreated run left a row");
    run_cap::reset();
    stall_cap::reset();
    util_stats::set_enabled(false);

    // The request is issued at step 0, the last mark of the run; the gap
    // after step k is k, so the first step whose gap exceeds the cap is
    // step cap + 1 and the run stops having run cap + 2 steps.
    let stop_step = cap + 2;
    match &outcomes[0].1 {
        RunOutcome::StallCapReached {
            cap: got,
            step,
            outstanding_events,
        } => {
            assert_eq!(*got, cap, "the exit carries the cap that stopped the run");
            assert_eq!(*step, stop_step, "the run stops on the first step past the cap");
            assert_eq!(*outstanding_events, 1, "the write is still outstanding");
        }
        other => panic!("expected a stall-cap exit for the treated run, got {other:?}"),
    }
    for (what, (_, outcome)) in ["untreated", "run-cap probe", "timer-context probe"]
        .iter()
        .zip(&outcomes[1..])
    {
        assert!(
            matches!(outcome, RunOutcome::IterationsExhausted { .. }),
            "the {what} run must run to its budget, got {outcome:?}"
        );
    }

    assert_eq!(after.stops, before.stops + 1);
    assert_eq!(after.treated_runs, before.treated_runs + 1);
    assert_eq!(after.untreated_runs, before.untreated_runs + 1);
    assert_eq!(after.untreated_runs_capped, before.untreated_runs_capped + 1);
    assert_eq!(
        after.untreated_over_cap_runs,
        before.untreated_over_cap_runs + 1,
        "the untreated run's gap exceeded the standing cap"
    );
    assert_eq!(
        after.steps_saved_sum,
        before.steps_saved_sum + (BACKUP - stop_step) as u64
    );
    assert_eq!(after.probes_keyed, before.probes_keyed, "exhausted probes feed nothing");
    assert!(after.marks.releases >= before.marks.releases + 4, "each run released its request");
    assert!(after.marks.rows >= before.marks.rows + 4, "each run recorded its invocation");
    assert_eq!(
        rows,
        format!(
            "run_id,longest_quiet_gap,stall_cap_standing\n{untreated},{},{cap}\n",
            BACKUP - 1
        ),
        "the untreated run's row carries its gap and the standing cap"
    );

    let run_ids = int_column(&out, "runs", "run_id");
    let reasons = string_column(&out, "runs", "end_reason");
    let steps = int_column(&out, "runs", "steps_used");
    let by_id = |id: i64| {
        let i = run_ids.iter().position(|&r| r == id).expect("a runs row per run");
        (reasons[i].as_str(), steps[i])
    };
    assert_eq!(by_id(treated), ("stall_cap_reached", stop_step as i64));
    assert_eq!(by_id(untreated), ("iterations_exhausted", BACKUP as i64));
    assert_eq!(by_id(cap_probe), ("iterations_exhausted", BACKUP as i64));
    assert_eq!(by_id(timer_probe), ("iterations_exhausted", BACKUP as i64));
    let history_rows = int_column(&out, "executions", "run_id");
    assert!(
        history_rows.contains(&treated),
        "the stopped run's history rows must flush"
    );
    let _ = fs::remove_dir_all(&out);
}
