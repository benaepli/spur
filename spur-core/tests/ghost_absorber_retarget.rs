//! The crash retarget has to fire on a real workload and leave the plan
//! whole: over a session of treated runs on a fixture whose handlers send to
//! every peer, planned crashes must move to the node that absorbed a
//! fault-crossing delivery, every run in which one moved must still complete
//! its plan, and every recover must land on a node that is down at that
//! moment - which after a retarget is the node the crash moved to. On the
//! untreated half nothing may change: the same run id gives the same event
//! sequence twice, and the retarget branch is never entered.

use serde_json::Value;
use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, SingleRunConfig, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::{fault_timing, ghost_absorber, run_cap, run_variant, util_stats};
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

const BACKUP: i32 = 600;
/// Completed-length samples the span learner is fed, well short of a run, so
/// a placed crash lands after the handlers have had messages in the air.
const SEEDED_LENGTH: i32 = 30;
const TREATED_RUNS: usize = 60;
const UNTREATED_RUNS: usize = 12;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("spur_ghost_absorber_{name}"));
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

/// The node index a crash or recover row names.
fn node_of(payload: &str) -> usize {
    let v: Value = serde_json::from_str(payload).expect("payload is JSON");
    v[0]["value"]["index"]
        .as_u64()
        .unwrap_or_else(|| panic!("no node index in {payload}")) as usize
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

/// Runs `ids` into `out`, one at a time, and returns for each run the change
/// in `victim_swap.applied` it produced.
fn session(
    program: &spur_core::compiler::cfg::Program,
    run_config: &SingleRunConfig,
    ids: &[i64],
    out: &Path,
) -> BTreeMap<i64, u64> {
    let writer: Arc<dyn HistoryWriter> = Arc::from(
        create_writer(LogBackend::Parquet, out.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    );
    let global_state = GlobalState::<NoFeedback>::new();
    let mut applied_by_run = BTreeMap::new();
    for &run_id in ids {
        let before = util_stats::snapshot().victim_swap.applied;
        run_single_simulation::<NoFeedback, LiveRng>(
            program,
            &writer,
            &global_state,
            run_id,
            run_config,
            &Default::default(),
            0x_C0FF_EE00 ^ run_id as u64,
            0x_5EED_1234 ^ run_id as u64,
            None,
            &RunAttribution::mode("test"),
        )
        .expect("the run executes");
        let after = util_stats::snapshot().victim_swap.applied;
        applied_by_run.insert(run_id, after - before);
    }
    writer.shutdown();
    applied_by_run
}

/// Per run, the crash and recover rows in sequence order.
fn faults(dir: &Path) -> HashMap<i64, Vec<(i64, String, usize)>> {
    let mut by_run: HashMap<i64, Vec<(i64, String, usize)>> = HashMap::new();
    for row in columns(dir, "executions", &["run_id", "seq_num", "kind", "payload"]) {
        if row[2] != "Crash" && row[2] != "Recover" {
            continue;
        }
        by_run.entry(row[0].parse().unwrap()).or_default().push((
            row[1].parse().unwrap(),
            row[2].clone(),
            node_of(&row[3]),
        ));
    }
    for events in by_run.values_mut() {
        events.sort();
    }
    by_run
}

fn runs(dir: &Path) -> HashMap<i64, (String, i32)> {
    columns(dir, "runs", &["run_id", "end_reason", "variant"])
        .into_iter()
        .map(|r| (r[0].parse().unwrap(), (r[1].clone(), r[2].parse().unwrap())))
        .collect()
}

#[test]
fn treated_runs_retarget_crashes_and_keep_every_pair_whole() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_treated)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

/// Feeds the span learner so placed runs draw a crash hold inside a run;
/// a crash taken at the first step lands before anything is in flight.
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

fn check_treated() {
    let _serial = config_override::exclusive_session();
    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let (program, run_config) = compile();
    let treated: Vec<i64> = (0..1_000_000i64)
        .filter(|&id| ghost_absorber::is_treated(id) && fault_timing::is_placed(id))
        .take(TREATED_RUNS)
        .collect();
    assert_eq!(treated.len(), TREATED_RUNS, "not enough treated run ids");

    let out = scratch("treated");
    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    let applied_by_run = session(&program, &run_config, &treated, &out);
    let after = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    let swap = &after.victim_swap;
    let applied = swap.applied - before.victim_swap.applied;
    assert!(applied > 0, "no planned crash ever moved: {swap:?}");
    assert!(
        after.ghost_signal.fired_runs > before.ghost_signal.fired_runs,
        "no run saw a fault-crossing delivery enter a node with a crash queued"
    );
    let treated_crashes = swap.census.treated.crashes - before.victim_swap.census.treated.crashes;
    assert!(treated_crashes > 0, "the treated census is empty");
    assert_eq!(
        swap.census.control.crashes, before.victim_swap.census.control.crashes,
        "a treated session must not count in the control half"
    );
    assert!(
        swap.census.treated.victim_had_absorbed
            > before.victim_swap.census.treated.victim_had_absorbed,
        "no crash landed on a node that had absorbed anything"
    );

    let runs = runs(&out);
    assert_eq!(runs.len(), TREATED_RUNS, "one runs row per run");
    let retargeted: Vec<i64> = applied_by_run
        .iter()
        .filter(|(_, n)| **n > 0)
        .map(|(id, _)| *id)
        .collect();
    assert!(
        !retargeted.is_empty(),
        "the counter moved but no run owns it"
    );
    for id in &retargeted {
        let (end, variant) = &runs[id];
        assert_eq!(
            end, "plan_complete",
            "run {id} moved a crash and then did not complete its plan"
        );
        assert_ne!(
            variant & run_variant::GHOST_ABSORBER_RETARGET,
            0,
            "run {id} is not tagged as treated"
        );
    }

    let faults = faults(&out);
    for id in &retargeted {
        let events = &faults[id];
        let mut down: BTreeSet<usize> = BTreeSet::new();
        let mut crashes = 0;
        let mut recovers = 0;
        for (seq, kind, node) in events {
            match kind.as_str() {
                "Crash" => {
                    assert!(
                        down.insert(*node),
                        "run {id} seq {seq}: node {node} crashed while already down"
                    );
                    crashes += 1;
                }
                _ => {
                    assert!(
                        down.remove(node),
                        "run {id} seq {seq}: node {node} recovered without being down"
                    );
                    recovers += 1;
                }
            }
        }
        assert_eq!(crashes, 2, "run {id} did not take both planned crashes");
        assert_eq!(recovers, 2, "run {id} did not recover both crashed nodes");
        assert!(down.is_empty(), "run {id} ended with {down:?} still down");
    }
    let _ = fs::remove_dir_all(&out);
}

#[test]
fn untreated_runs_are_unchanged_and_never_reach_the_retarget() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_untreated)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_untreated() {
    let _serial = config_override::exclusive_session();
    run_cap::reset();
    fault_timing::reset();
    let (program, run_config) = compile();
    let untreated: Vec<i64> = (0..1_000_000i64)
        .filter(|&id| !ghost_absorber::is_treated(id) && fault_timing::is_placed(id))
        .take(UNTREATED_RUNS)
        .collect();

    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    seed_span();
    let first = scratch("untreated_first");
    let applied_first = session(&program, &run_config, &untreated, &first);
    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let second = scratch("untreated_second");
    let applied_second = session(&program, &run_config, &untreated, &second);
    let after = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    for by_run in [&applied_first, &applied_second] {
        assert!(
            by_run.values().all(|n| *n == 0),
            "an untreated run moved a crash"
        );
    }
    let (b, a) = (&before.victim_swap, &after.victim_swap);
    assert_eq!(a.applied, b.applied);
    assert_eq!(a.same_victim, b.same_victim, "the ranking was consulted");
    assert_eq!(a.no_absorber, b.no_absorber, "the ranking was consulted");
    assert_eq!(a.skipped_pending_pair, b.skipped_pending_pair);
    assert_eq!(
        a.victim_crashed_holds, b.victim_crashed_holds,
        "a crash was held"
    );
    assert_eq!(a.census.treated.crashes, b.census.treated.crashes);
    assert!(
        a.census.control.crashes > b.census.control.crashes,
        "the control census saw no crash"
    );

    for id in &untreated {
        let (_, variant) = runs(&first)[id];
        assert_eq!(
            variant & run_variant::GHOST_ABSORBER_RETARGET,
            0,
            "run {id} is tagged treated"
        );
    }

    let rows = |dir: &Path| -> BTreeSet<Vec<String>> {
        columns(
            dir,
            "executions",
            &["run_id", "seq_num", "kind", "action", "payload", "step"],
        )
        .into_iter()
        .collect()
    };
    let (r1, r2) = (rows(&first), rows(&second));
    assert!(!r1.is_empty(), "the first session wrote nothing");
    assert_eq!(
        r1, r2,
        "the same untreated run ids gave different event sequences"
    );
    let ends = |dir: &Path| -> BTreeMap<i64, String> {
        runs(dir)
            .into_iter()
            .map(|(id, (end, _))| (id, end))
            .collect()
    };
    assert_eq!(ends(&first), ends(&second));
    let _ = fs::remove_dir_all(&first);
    let _ = fs::remove_dir_all(&second);
}
