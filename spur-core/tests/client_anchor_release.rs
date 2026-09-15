//! The client hold has to fire on a real workload and leave the plan whole:
//! over a session of treated runs on a fixture whose peers answer a
//! fault-crossing delivery with a fan-out, client requests that become ready
//! after the first crash must be held, every one that leaves the hold must
//! do so at expiry or when the run has nothing else to move, windows must
//! still be counted, none may still be held when a run ends, every run must
//! issue every planned request exactly once, and the share of runs that
//! complete their plan must stay close to a control session's. On the
//! untreated half nothing may change: the same run id gives the same event
//! sequence twice, no request is held, and the population is still counted.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, SingleRunConfig, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::client_anchor::{self, EXPIRY_STEPS};
use spur_core::simulator::util_stats::ClientAnchorStats;
use spur_core::simulator::{fault_timing, run_cap, run_variant, util_stats};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/canchor.spur");

const CONFIG: &str = r#"{
  "params": {"n": {"min": 3, "max": 3, "step": 1}},
  "num_write_ops": {"min": 4, "max": 4, "step": 1},
  "num_read_ops": {"min": 2, "max": 2, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 2, "max": 2, "step": 1},
  "dependency_density": [0.0],
  "post_fault_client_ops": 1,
  "num_runs_per_config": 1,
  "max_iterations": 600,
  "session_seed": 4242,
  "queue_policy": {"type": "Probabilistic", "p_local": 0.7, "p_timer": 0.2},
  "rng_stream_isolation": true,
  "strict_config_keys": true,
  "stats": true
}"#;

/// Planned client requests per run: the write and read counts above.
const PLANNED_REQUESTS: usize = 6;
const BACKUP: i32 = 600;
/// Completed-length samples the span learner is fed, well short of a run, so
/// a placed crash lands after the handlers have had messages in the air.
const SEEDED_LENGTH: i32 = 30;
const TREATED_RUNS: usize = 60;
const CONTROL_RUNS: usize = 60;
const UNTREATED_RUNS: usize = 12;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("spur_client_anchor_{name}"));
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
    let program = compiler::compile(SPEC, "canchor.spur")
        .into_program()
        .expect("spec compiles");
    let mut config: ExplorerConfig = serde_json::from_str(CONFIG).expect("config parses");
    config
        .bind(
            &program,
            &std::sync::Arc::new(std::sync::Mutex::new(spur_core::simulator::deploy::params::DeployCache::default())),
        )
        .expect("config binds");
    let run_config = config
        .expand_grid()
        .into_iter()
        .next()
        .expect("the grid holds one config");
    (program, run_config)
}

fn session(
    program: &spur_core::compiler::cfg::Program,
    run_config: &SingleRunConfig,
    ids: &[i64],
    out: &Path,
) {
    let writer: Arc<dyn HistoryWriter> = Arc::from(
        create_writer(LogBackend::Parquet, out.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    );
    let global_state = GlobalState::<NoFeedback>::new();
    for &run_id in ids {
        // The arm selector steers a run only from a cell past its warmup;
        // clearing it before every run keeps each run on its coins.
        spur_core::simulator::arm_selector::reset();
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
    }
    writer.shutdown();
}

fn runs(dir: &Path) -> HashMap<i64, (String, i32)> {
    columns(dir, "runs", &["run_id", "end_reason", "variant"])
        .into_iter()
        .map(|r| (r[0].parse().unwrap(), (r[1].clone(), r[2].parse().unwrap())))
        .collect()
}

/// Invocation rows per run.
fn invocations(dir: &Path) -> HashMap<i64, usize> {
    let mut by_run: HashMap<i64, usize> = HashMap::new();
    for row in columns(dir, "executions", &["run_id", "kind"]) {
        if row[1] == "Invocation" {
            *by_run.entry(row[0].parse().unwrap()).or_default() += 1;
        }
    }
    by_run
}

fn completed_share(dir: &Path) -> f64 {
    let all = runs(dir);
    let done = all.values().filter(|(end, _)| end == "plan_complete").count();
    done as f64 / all.len() as f64
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

fn placed_ids(treated: bool, n: usize) -> Vec<i64> {
    let ids: Vec<i64> = (0..1_000_000i64)
        .filter(|&id| {
            client_anchor::is_treated(id) == treated
                && fault_timing::is_placed(id)
        })
        .take(n)
        .collect();
    assert_eq!(ids.len(), n, "not enough run ids on that half");
    ids
}

fn fresh_session() {
    run_cap::reset();
    fault_timing::reset();
    seed_span();
}

#[test]
fn treated_runs_hold_post_crash_requests_and_issue_them_at_windows() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_treated)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_treated() {
    let _serial = config_override::exclusive_session();
    let (program, run_config) = compile();
    let treated = placed_ids(true, TREATED_RUNS);
    let control = placed_ids(false, CONTROL_RUNS);

    util_stats::set_enabled(true);
    fresh_session();
    let before = util_stats::snapshot();
    let out = scratch("treated");
    session(&program, &run_config, &treated, &out);
    let after = util_stats::snapshot();
    fresh_session();
    let control_out = scratch("control");
    session(&program, &run_config, &control, &control_out);
    let after_control = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    let (b, a): (&ClientAnchorStats, &ClientAnchorStats) =
        (&before.client_anchor, &after.client_anchor);
    let held = a.held - b.held;
    let expiry = a.released.expiry - b.released.expiry;
    let dry = a.released.dry_queue - b.released.dry_queue;
    let held_at_exit = a.held_at_exit - b.held_at_exit;
    assert!(held > 0, "no post-crash request was held: {a:?}");
    assert!(expiry > 0, "no held request expired: {a:?}");
    assert_eq!(held_at_exit, 0, "a run ended with a request still held: {a:?}");
    assert_eq!(a.runs_with_held_at_exit, b.runs_with_held_at_exit);
    assert_eq!(
        expiry + dry,
        held - held_at_exit,
        "every held request that left did so at expiry or a dry queue"
    );
    assert!(
        a.hold_steps_sum - b.hold_steps_sum >= expiry * (EXPIRY_STEPS as u64 + 1),
        "an expired request waited fewer than {} steps",
        EXPIRY_STEPS + 1
    );
    let t = &a.census.treated;
    assert_eq!(t.runs - b.census.treated.runs, TREATED_RUNS as u64);
    assert_eq!(
        t.population - b.census.treated.population,
        held,
        "the treated population is exactly what was held"
    );
    assert!(
        t.fanout_windows > b.census.treated.fanout_windows,
        "no window opened"
    );
    assert_eq!(
        t.post_fault_invocations - b.census.treated.post_fault_invocations,
        held,
        "every held request was issued"
    );
    let first_windows = |s: &ClientAnchorStats| {
        s.held_at_first_firing.zero
            + s.held_at_first_firing.one
            + s.held_at_first_firing.two
            + s.held_at_first_firing.three_plus
    };
    assert!(
        first_windows(a) > first_windows(b),
        "no treated run recorded what it held at its first window"
    );
    assert_eq!(
        a.census.control.runs, b.census.control.runs,
        "a treated session must not count in the control half"
    );
    let c = &after_control.client_anchor.census.control;
    assert_eq!(c.runs - a.census.control.runs, CONTROL_RUNS as u64);
    assert!(
        c.population > a.census.control.population,
        "the control half counted no population"
    );
    assert_eq!(
        after_control.client_anchor.held, a.held,
        "a control session held a request"
    );

    let treated_share = completed_share(&out);
    let control_share = completed_share(&control_out);
    assert!(
        treated_share >= control_share - 0.2,
        "treated runs complete their plan at {treated_share} against {control_share} on control"
    );

    let rows = runs(&out);
    assert_eq!(rows.len(), TREATED_RUNS, "one runs row per run");
    let issued = invocations(&out);
    for id in &treated {
        let (end, variant) = &rows[id];
        assert_ne!(
            variant & run_variant::CLIENT_FANOUT_RELEASE,
            0,
            "run {id} is not tagged as treated"
        );
        assert_ne!(end, "deadlock", "run {id} deadlocked");
        let n = issued.get(id).copied().unwrap_or(0);
        if end == "plan_complete" {
            assert_eq!(
                n, PLANNED_REQUESTS,
                "run {id} completed its plan with {n} invocations"
            );
        } else {
            assert!(
                n <= PLANNED_REQUESTS,
                "run {id} issued {n} invocations for {PLANNED_REQUESTS} requests"
            );
        }
    }
    let _ = fs::remove_dir_all(&out);
    let _ = fs::remove_dir_all(&control_out);
}

#[test]
fn untreated_runs_are_unchanged_and_hold_nothing() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_untreated)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_untreated() {
    let _serial = config_override::exclusive_session();
    let (program, run_config) = compile();
    let untreated = placed_ids(false, UNTREATED_RUNS);

    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    fresh_session();
    let first = scratch("untreated_first");
    session(&program, &run_config, &untreated, &first);
    fresh_session();
    let second = scratch("untreated_second");
    session(&program, &run_config, &untreated, &second);
    let after = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    let (b, a) = (&before.client_anchor, &after.client_anchor);
    assert_eq!(a.held, b.held, "an untreated run held a request");
    assert_eq!(a.released.expiry, b.released.expiry);
    assert_eq!(a.released.dry_queue, b.released.dry_queue);
    assert_eq!(a.held_at_exit, b.held_at_exit);
    assert_eq!(a.hold_steps_sum, b.hold_steps_sum);
    assert_eq!(a.census.treated.runs, b.census.treated.runs);
    assert_eq!(
        a.census.control.runs - b.census.control.runs,
        2 * UNTREATED_RUNS as u64
    );
    assert!(
        a.census.control.population > b.census.control.population,
        "the control census counted no post-crash request"
    );
    assert_eq!(
        a.census.control.post_fault_invocations - b.census.control.post_fault_invocations,
        a.census.control.population - b.census.control.population,
        "on the control half every post-crash request is issued when ready"
    );

    for id in &untreated {
        let (_, variant) = runs(&first)[id];
        assert_eq!(
            variant & run_variant::CLIENT_FANOUT_RELEASE,
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
