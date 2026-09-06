//! A prefix child has to walk in its parent's steps. A fresh run recorded on
//! a fixture whose handlers send to every peer fires the fault-crossing
//! signal at some step; a child that replays the parent's draws up to that
//! point, on the parent's workload and configuration, must take the same
//! events up to that step and fire the signal at the same step, however its
//! own seed continues the run. Recording the draws must not change them: the
//! same run ids give the same event sequence whether the draws are logged or
//! taken live.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, RunResult, SingleRunConfig,
    run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::{LiveRng, RecordRng, Recording, ReplayRng, RngSource};
use spur_core::simulator::{fault_timing, run_cap, run_variant, util_stats};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/ghost.spur");

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

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;
const PARENT_CANDIDATES: usize = 200;
const IDENTITY_RUNS: usize = 12;
const BACKUP: i32 = 600;
/// Completed-length samples the span learner is fed, well short of a run, so
/// a placed crash waits inside a run and a delivery can reach its node first.
const SEEDED_LENGTH: i32 = 30;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("spur_replay_prefix_{name}"));
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

fn compile() -> (spur_core::compiler::cfg::Program, SingleRunConfig) {
    let program = compiler::compile(SPEC, "ghost.spur")
        .into_program()
        .expect("spec compiles");
    let config: ExplorerConfig = serde_json::from_str(CONFIG).expect("config parses");
    let run_config = config
        .expand_grid()
        .into_iter()
        .next()
        .expect("the grid holds one config");
    (program, run_config)
}

fn writer(dir: &Path) -> Arc<dyn HistoryWriter> {
    Arc::from(
        create_writer(LogBackend::Parquet, dir.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    )
}

fn seed_span() {
    let feeder = (0..1_000_000i64)
        .find(|&id| run_cap::is_probe(id))
        .expect("some run id is a probe");
    for _ in 0..200 {
        fault_timing::merge_stock_probe(feeder, BACKUP, run_cap::Outcome::Completed, SEEDED_LENGTH);
    }
    assert!(
        fault_timing::median(BACKUP).is_some_and(|m| m < 64),
        "the seeded span must be short enough to expire inside a run"
    );
}

fn workload_seed(run_id: i64) -> u64 {
    0x_C0FF_EE00 ^ run_id as u64
}

fn schedule_seed(run_id: i64) -> u64 {
    0x_5EED_1234 ^ run_id as u64
}

fn run<S: RngSource>(
    program: &spur_core::compiler::cfg::Program,
    run_config: &SingleRunConfig,
    writer: &Arc<dyn HistoryWriter>,
    run_id: i64,
    workload: u64,
    schedule: u64,
    tape: Option<Recording>,
) -> RunResult {
    // The arm selector steers a run only from a cell past its warmup;
    // clearing it before every run keeps each run on its coins.
    spur_core::simulator::arm_selector::reset();
    run_single_simulation::<NoFeedback, S>(
        program,
        writer,
        &GlobalState::<NoFeedback>::new(),
        run_id,
        run_config,
        &Default::default(),
        workload,
        schedule,
        tape,
        &RunAttribution::mode("test"),
    )
    .expect("the run executes")
}

/// Events by step, without the run id: what a run did and when.
fn events_through(dir: &Path, last_step: i32) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = columns(
        dir,
        "executions",
        &["seq_num", "kind", "action", "payload", "step"],
    )
    .into_iter()
    .filter(|r| r[4].parse::<i32>().expect("step") <= last_step)
    .collect();
    rows.sort_by_key(|r| r[0].parse::<i64>().expect("seq_num"));
    rows
}

#[test]
fn a_prefix_child_walks_in_its_parents_steps_up_to_the_cut() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_prefix)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_prefix() {
    let _serial = config_override::exclusive_session();
    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let (program, run_config) = compile();
    // Parent and child ids carry the same mechanism bits, so the only thing
    // that differs between the two runs is the schedule seed the child falls
    // back on once the replayed prefix runs out. Placed runs hold a queued
    // crash back, which is what lets a delivery reach its node first.
    let eligible = |id: i64| {
        let v = run_variant::from_run_id(id);
        v & run_variant::CRASH_PLACED != 0
            && v & (run_variant::RUN_CAP_PROBE | run_variant::TIMER_STEER_OFF) == 0
    };
    let candidates: Vec<i64> = (0..1_000_000i64)
        .filter(|&id| eligible(id))
        .take(3 * PARENT_CANDIDATES)
        .collect();
    assert_eq!(candidates.len(), 3 * PARENT_CANDIDATES, "not enough placed run ids");
    let (parents, children) = candidates.split_at(PARENT_CANDIDATES);

    let mut checked = 0;
    for (i, &parent_id) in parents.iter().enumerate() {
        let parent_dir = scratch(&format!("parent_{i}"));
        let w = writer(&parent_dir);
        let parent = run::<RecordRng>(
            &program,
            &run_config,
            &w,
            parent_id,
            workload_seed(parent_id),
            schedule_seed(parent_id),
            None,
        );
        w.shutdown();
        let Some(cut) = parent.cut else {
            let _ = fs::remove_dir_all(&parent_dir);
            continue;
        };
        let pos = cut.tape_pos.expect("a recording run knows its draw count");
        let tape = parent.recording.expect("a recording run produces a tape");
        assert!(pos <= tape.len(), "the cut lies past the end of the tape");
        assert!(pos > 0, "the signal fired before any draw");
        let prefix: Recording = tape[..pos].into();

        let child_id = *children
            .iter()
            .find(|&&id| run_variant::from_run_id(id) == run_variant::from_run_id(parent_id))
            .expect("a child id with the parent's mechanism bits");
        let child_dir = scratch(&format!("child_{i}"));
        let w = writer(&child_dir);
        let child = run::<ReplayRng>(
            &program,
            &run_config,
            &w,
            child_id,
            workload_seed(parent_id),
            schedule_seed(child_id),
            Some(prefix),
        );
        w.shutdown();

        assert_eq!(
            child.cut.map(|c| c.step),
            Some(cut.step),
            "parent {parent_id}: the child did not fire the signal at the parent's step"
        );
        let parent_events = events_through(&parent_dir, cut.step);
        let child_events = events_through(&child_dir, cut.step);
        assert!(!parent_events.is_empty(), "the parent wrote no events before the cut");
        assert_eq!(
            parent_events, child_events,
            "parent {parent_id}: the child's events up to step {} differ",
            cut.step
        );
        let _ = fs::remove_dir_all(&parent_dir);
        let _ = fs::remove_dir_all(&child_dir);
        checked += 1;
        if checked == 3 {
            break;
        }
    }
    assert!(checked > 0, "no candidate parent fired the signal");
    run_cap::reset();
    fault_timing::reset();
}

#[test]
fn recording_the_draws_does_not_change_them() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_identity)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_identity() {
    let _serial = config_override::exclusive_session();
    run_cap::reset();
    fault_timing::reset();
    let (program, run_config) = compile();
    let ids: Vec<i64> = (0..IDENTITY_RUNS as i64).collect();

    util_stats::set_enabled(true);
    let live_dir = scratch("live");
    let w = writer(&live_dir);
    for &id in &ids {
        let r = run::<LiveRng>(
            &program,
            &run_config,
            &w,
            id,
            workload_seed(id),
            schedule_seed(id),
            None,
        );
        assert!(r.recording.is_none(), "a live run produced a tape");
    }
    w.shutdown();

    let recorded_dir = scratch("recorded");
    let w = writer(&recorded_dir);
    let mut words = 0usize;
    for &id in &ids {
        let r = run::<RecordRng>(
            &program,
            &run_config,
            &w,
            id,
            workload_seed(id),
            schedule_seed(id),
            None,
        );
        words += r.recording.expect("a recording run produces a tape").len();
    }
    w.shutdown();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();
    assert!(words > 0, "the recorded runs drew nothing");

    let rows = |dir: &Path| -> BTreeSet<Vec<String>> {
        columns(
            dir,
            "executions",
            &["run_id", "seq_num", "kind", "action", "payload", "step"],
        )
        .into_iter()
        .collect()
    };
    let (live, recorded) = (rows(&live_dir), rows(&recorded_dir));
    assert!(!live.is_empty(), "the live session wrote nothing");
    assert_eq!(live, recorded, "recording the draws changed the event sequence");
    let ends = |dir: &Path| -> BTreeMap<String, (String, String)> {
        columns(dir, "runs", &["run_id", "end_reason", "steps_used"])
            .into_iter()
            .map(|r| (r[0].clone(), (r[1].clone(), r[2].clone())))
            .collect()
    };
    assert_eq!(ends(&live_dir), ends(&recorded_dir));
    let _ = fs::remove_dir_all(&live_dir);
    let _ = fs::remove_dir_all(&recorded_dir);
}
