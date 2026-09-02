//! The fan-out anchor has to fire on a real workload, not only on a
//! hand-built ledger: over a session of anchored placed runs on a fixture
//! whose handlers send to every peer, crashes must draw both waiting arms,
//! more of them must reach their phase than run out of window, and the two
//! census instants must both carry counts. The runs themselves must still be
//! ordinary runs - each ending for a reason the writer recognises, and every
//! client operation that was invoked and whose run completed its plan
//! answered.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::{crash_phase, fault_timing, run_cap, util_stats};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/fanout.spur");

const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 4, "max": 4, "step": 1},
  "num_read_ops": {"min": 2, "max": 2, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 2, "max": 2, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 1,
  "max_iterations": 600,
  "session_seed": 4242,
  "queue_policy": {"type": "Probabilistic", "p_local": 0.7, "p_timer": 0.2},
  "rng_stream_isolation": true,
  "strict_config_keys": true,
  "stats": true
}"#;

const BACKUP: i32 = 600;
/// Completed-length samples the span learner is fed, well short of a run so
/// the drawn hold expires inside one and the second stage is reached.
const SEEDED_LENGTH: i32 = 30;
const RUNS: usize = 80;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

const END_REASONS: [&str; 4] = [
    "plan_complete",
    "deadlock",
    "iterations_exhausted",
    "learned_cap_reached",
];

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join("spur_crash_phase_anchor");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

fn columns(dir: &Path, table: &str, names: &[&str]) -> Vec<Vec<String>> {
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
                let mut row = Vec::new();
                for name in names {
                    let col = batch
                        .column_by_name(name)
                        .unwrap_or_else(|| panic!("column {name} of {table}"));
                    row.push(match col.data_type() {
                        DataType::Int32 => col.as_primitive::<Int32Type>().value(i).to_string(),
                        DataType::Int64 => col.as_primitive::<Int64Type>().value(i).to_string(),
                        DataType::Utf8 => col.as_string::<i32>().value(i).to_string(),
                        other => panic!("column {name} is {other:?}"),
                    });
                }
                rows.push(row);
            }
        }
    }
    rows
}

#[test]
fn a_session_of_anchored_runs_arms_both_waiting_arms_and_censuses_them() {
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
    fault_timing::reset();
    let feeder = (0..1_000_000i64)
        .find(|&id| run_cap::is_probe(id))
        .expect("some run id is a probe");
    for _ in 0..200 {
        fault_timing::merge_stock_probe(
            feeder,
            BACKUP,
            run_cap::Outcome::Completed,
            SEEDED_LENGTH,
        );
    }
    assert!(
        fault_timing::median(BACKUP).is_some_and(|m| m < 64),
        "the seeded span must be short enough to expire inside a run"
    );

    let program = compiler::compile(SPEC, "fanout.spur")
        .into_program()
        .expect("spec compiles");
    let config: ExplorerConfig = serde_json::from_str(CONFIG).expect("config parses");
    let run_config = config
        .expand_grid()
        .into_iter()
        .next()
        .expect("the grid holds one config");

    let anchored: Vec<i64> = (0..1_000_000i64)
        .filter(|&id| crash_phase::is_anchored(id))
        .take(RUNS)
        .collect();
    assert_eq!(anchored.len(), RUNS, "not enough anchored run ids");

    let out = scratch();
    let writer: Arc<dyn HistoryWriter> = Arc::from(
        create_writer(LogBackend::Parquet, out.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    );
    let global_state = GlobalState::<NoFeedback>::new();
    util_stats::set_enabled(true);
    util_stats::set_crash_census_enabled(true);
    for &run_id in &anchored {
        run_single_simulation::<NoFeedback, LiveRng>(
            &program,
            &writer,
            &global_state,
            run_id,
            &run_config,
            &Default::default(),
            0x_C0FF_EE00 ^ run_id as u64,
            0x_5EED_1234 ^ run_id as u64,
            None,
            &RunAttribution::mode("test"),
        )
        .expect("the run executes");
    }
    writer.shutdown();
    let phase = util_stats::snapshot().crash_phase;
    util_stats::set_crash_census_enabled(false);
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    assert!(phase.armed > 0, "no crash was ever made to wait: {phase:?}");
    for (name, arm) in [("early", &phase.early), ("mid", &phase.mid)] {
        assert!(arm.armed > 0, "the {name} arm was never drawn");
        assert!(
            arm.release_decisions > 0,
            "the {name} arm recorded no release"
        );
        assert!(
            arm.expired < arm.armed,
            "every {name} wait ran out of window: {} of {}",
            arm.expired,
            arm.armed
        );
    }
    assert!(
        phase.early.released_on_condition > 0,
        "no wait ever reached its phase"
    );
    assert!(
        phase.stock_releases > 0,
        "the control arm was never drawn, so the arms are not equal-mass"
    );
    let released = phase.early.release_decisions + phase.mid.release_decisions;
    let with_inflight =
        phase.early.release_victim_had_inflight + phase.mid.release_victim_had_inflight;
    assert!(
        with_inflight > 0 && with_inflight <= released,
        "the release-time census is empty: {with_inflight} of {released}"
    );
    // A wait that reached its phase counted at least one undelivered send of
    // the victim's own, so every one of those releases is a victim holding
    // something; anything less means the two instants disagree.
    for (name, arm) in [("early", &phase.early), ("mid", &phase.mid)] {
        assert_eq!(
            arm.release_victim_had_inflight - arm.expired_victim_had_inflight,
            arm.released_on_condition,
            "a {name} wait reached its phase with nothing of the victim's in flight"
        );
    }
    let applied = phase.early.crashes_applied + phase.mid.crashes_applied;
    assert!(applied > 0, "no crash carrying a wait was ever taken");
    let apply_bucketed: u64 = [&phase.early, &phase.mid, &phase.stock]
        .iter()
        .map(|a| {
            a.inflight_bucket_0
                + a.inflight_bucket_1
                + a.inflight_bucket_2
                + a.inflight_bucket_3plus
        })
        .sum();
    let apply_decisions =
        phase.early.apply_decisions + phase.mid.apply_decisions + phase.stock.apply_decisions;
    assert_eq!(
        apply_bucketed, apply_decisions,
        "the apply-time histogram and its denominator disagree"
    );
    assert!(apply_decisions > 0, "the apply-time census is empty");

    let runs = columns(&out, "runs", &["run_id", "end_reason"]);
    assert_eq!(runs.len(), RUNS, "one runs row per run");
    for row in &runs {
        assert!(
            END_REASONS.contains(&row[1].as_str()),
            "run {} ended for an unrecognised reason {}",
            row[0],
            row[1]
        );
    }
    let completed: HashSet<String> = runs
        .iter()
        .filter(|r| r[1] == "plan_complete")
        .map(|r| r[0].clone())
        .collect();
    assert!(!completed.is_empty(), "no run completed its plan");

    let mut invoked: HashMap<(String, String), i32> = HashMap::new();
    for row in columns(&out, "executions", &["run_id", "unique_id", "kind"]) {
        let key = (row[0].clone(), row[1].clone());
        match row[2].as_str() {
            "Invocation" => *invoked.entry(key).or_default() += 1,
            "Response" => *invoked.entry(key).or_default() -= 1,
            _ => {}
        }
    }
    let unanswered: Vec<&(String, String)> = invoked
        .iter()
        .filter(|((run, _), n)| **n != 0 && completed.contains(run))
        .map(|(k, _)| k)
        .collect();
    assert!(
        unanswered.is_empty(),
        "invocations without a response in a completed run: {unanswered:?}"
    );
    let _ = fs::remove_dir_all(&out);
}
