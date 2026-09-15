//! The fresh-first dispatch preference has to fire on a real workload and
//! change only what it claims to: over a session of treated runs on a
//! fixture whose handlers send to every peer and whose nodes crash with
//! those sends in the air, a network step whose draw fell on a ghost must
//! take an eligible fresh record from the same sender to the same
//! destination instead; every such contest at a live destination must end
//! with the fresh record taken; and ghosts must reach destinations that have
//! already heard from the restarted sender more often than on the control
//! half. On the untreated half nothing may change: the same run id gives the
//! same event sequence twice and the swap is never made, while the census
//! still counts.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, SingleRunConfig, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::util_stats::{FreshFirstHalfStats, FreshFirstStats, UtilizationSnapshot};
use spur_core::simulator::{fault_timing, fresh_first, run_cap, run_variant, util_stats};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/ghost.spur");

const CONFIG: &str = r#"{
  "params": {"n": {"min": 3, "max": 3, "step": 1}},
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
const HALF_RUNS: usize = 60;
const UNTREATED_RUNS: usize = 12;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("spur_fresh_first_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

fn columns(dir: &Path, table: &str, names: &[&str]) -> Vec<Vec<String>> {
    use arrow::array::AsArray;
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

/// Runs `ids` into `out`, one at a time, and returns for each run the change
/// in `fresh_first.swaps` it produced.
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
    let mut swaps_by_run = BTreeMap::new();
    for &run_id in ids {
        let before = util_stats::snapshot().fresh_first.swaps;
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
        let after = util_stats::snapshot().fresh_first.swaps;
        swaps_by_run.insert(run_id, after - before);
    }
    writer.shutdown();
    swaps_by_run
}

fn runs(dir: &Path) -> HashMap<i64, (String, i32)> {
    columns(dir, "runs", &["run_id", "end_reason", "variant"])
        .into_iter()
        .map(|r| (r[0].parse().unwrap(), (r[1].clone(), r[2].parse().unwrap())))
        .collect()
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

fn ids(treated: bool, n: usize) -> Vec<i64> {
    let ids: Vec<i64> = (0..1_000_000i64)
        .filter(|&id| {
            fresh_first::is_treated(id) == treated
                && fault_timing::is_placed(id)
        })
        .take(n)
        .collect();
    assert_eq!(ids.len(), n, "not enough run ids on the half");
    ids
}

fn half_delta(a: &FreshFirstHalfStats, b: &FreshFirstHalfStats) -> FreshFirstHalfStats {
    FreshFirstHalfStats {
        contested_dispatches: a.contested_dispatches - b.contested_dispatches,
        stale_drawn: a.stale_drawn - b.stale_drawn,
        contested_down: a.contested_down - b.contested_down,
        ghost_entries_from_restarted_origin: a.ghost_entries_from_restarted_origin
            - b.ghost_entries_from_restarted_origin,
        overtaken: a.overtaken - b.overtaken,
        ghost_entries_to_restarted_dest: a.ghost_entries_to_restarted_dest
            - b.ghost_entries_to_restarted_dest,
        overtaken_at_restarted_dest: a.overtaken_at_restarted_dest
            - b.overtaken_at_restarted_dest,
    }
}

fn block_delta(after: &UtilizationSnapshot, before: &UtilizationSnapshot) -> FreshFirstStats {
    let (a, b) = (&after.fresh_first, &before.fresh_first);
    FreshFirstStats {
        swaps: a.swaps - b.swaps,
        repeat_swaps: a.repeat_swaps - b.repeat_swaps,
        skipped_never_restarted_dest: a.skipped_never_restarted_dest - b.skipped_never_restarted_dest,
        swap_count_hist_1: a.swap_count_hist_1 - b.swap_count_hist_1,
        swap_count_hist_2: a.swap_count_hist_2 - b.swap_count_hist_2,
        swap_count_hist_3: a.swap_count_hist_3 - b.swap_count_hist_3,
        swap_count_hist_4plus: a.swap_count_hist_4plus - b.swap_count_hist_4plus,
        census: spur_core::simulator::util_stats::FreshFirstCensusStats {
            treated: half_delta(&a.census.treated, &b.census.treated),
            control: half_delta(&a.census.control, &b.census.control),
        },
    }
}

fn share(num: u64, den: u64) -> f64 {
    if den == 0 { 0.0 } else { num as f64 / den as f64 }
}

#[test]
fn treated_runs_take_the_fresh_record_at_every_live_contest_and_ghosts_arrive_late() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_treated)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_treated() {
    let _serial = config_override::exclusive_session();
    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let (program, run_config) = compile();
    let treated = ids(true, HALF_RUNS);

    let out = scratch("treated");
    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    let swaps_by_run = session(&program, &run_config, &treated, &out);
    let after_treated = util_stats::snapshot();
    let d = block_delta(&after_treated, &before);

    assert!(d.swaps > 0, "no contested step ever swapped: {d:?}");
    assert!(
        d.repeat_swaps < d.swaps,
        "every swap displaced a ghost that had already been displaced: {d:?}"
    );
    let t = &d.census.treated;
    assert!(t.contested_dispatches > 0, "the treated census is empty: {d:?}");
    assert!(
        t.stale_drawn > 0 && t.stale_drawn < t.contested_dispatches,
        "the draw never fell on both classes: {d:?}"
    );
    assert_eq!(
        t.stale_drawn - t.contested_down - d.swaps - d.skipped_never_restarted_dest,
        0,
        "a treated contest at a live destination took the ghost while a fresh record was eligible: {d:?}"
    );
    // Every swap displaces one ghost, and a ghost enters the histogram once,
    // when it is finally taken, so the histogram cannot exceed the number of
    // distinct ghosts displaced.
    assert!(
        d.swap_count_hist_1 + d.swap_count_hist_2 + d.swap_count_hist_3 + d.swap_count_hist_4plus
            <= d.swaps - d.repeat_swaps,
        "more ghosts were finally taken than were ever displaced: {d:?}"
    );
    assert!(
        d.swap_count_hist_1 > 0,
        "no displaced ghost was ever taken once its rival was gone: {d:?}"
    );
    assert_eq!(
        d.census.control.contested_dispatches, 0,
        "a treated session counted in the control half: {d:?}"
    );
    assert_eq!(
        d.census.control.ghost_entries_from_restarted_origin, 0,
        "a treated session counted in the control half: {d:?}"
    );
    assert!(
        t.ghost_entries_from_restarted_origin > 0,
        "no ghost ever entered a handler: {d:?}"
    );

    let runs = runs(&out);
    assert_eq!(runs.len(), HALF_RUNS, "one runs row per run");
    let swapped: Vec<i64> = swaps_by_run
        .iter()
        .filter(|(_, n)| **n > 0)
        .map(|(id, _)| *id)
        .collect();
    assert!(!swapped.is_empty(), "the counter moved but no run owns it");
    for id in &treated {
        let (end, variant) = &runs[id];
        assert_ne!(
            variant & run_variant::FRESH_FIRST_PAIR,
            0,
            "run {id} is not tagged as treated"
        );
        if swaps_by_run[id] > 0 {
            assert_eq!(end, "plan_complete", "run {id} swapped and then did not complete its plan");
        }
    }

    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let control = ids(false, HALF_RUNS);
    let control_out = scratch("control");
    let control_swaps = session(&program, &run_config, &control, &control_out);
    let after_control = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();
    let c = block_delta(&after_control, &after_treated);

    assert!(
        control_swaps.values().all(|n| *n == 0) && c.swaps == 0,
        "a control run swapped: {c:?}"
    );
    assert_eq!(c.census.treated.contested_dispatches, 0, "a control session counted in the treated half");
    let cc = &c.census.control;
    assert!(cc.contested_dispatches > 0, "the control census is empty: {c:?}");
    assert!(cc.stale_drawn > 0, "the control draw never fell on a ghost: {c:?}");
    assert!(cc.ghost_entries_from_restarted_origin > 0, "no ghost entered a handler on the control half: {c:?}");

    let treated_share = share(t.overtaken, t.ghost_entries_from_restarted_origin);
    let control_share = share(cc.overtaken, cc.ghost_entries_from_restarted_origin);
    eprintln!("fresh_first treated session: {d:?}");
    eprintln!("fresh_first control session: {c:?}");
    eprintln!("overtaken share treated {treated_share:.3} control {control_share:.3}");
    assert!(
        treated_share > control_share,
        "ghosts did not arrive late more often on the treated half: treated {treated_share:.3} ({t:?}), control {control_share:.3} ({cc:?})"
    );
    let _ = fs::remove_dir_all(&out);
    let _ = fs::remove_dir_all(&control_out);
}

#[test]
fn untreated_runs_are_unchanged_and_never_swap() {
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
    let untreated = ids(false, UNTREATED_RUNS);

    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    seed_span();
    let first = scratch("untreated_first");
    let swaps_first = session(&program, &run_config, &untreated, &first);
    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let second = scratch("untreated_second");
    let swaps_second = session(&program, &run_config, &untreated, &second);
    let after = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    for by_run in [&swaps_first, &swaps_second] {
        assert!(by_run.values().all(|n| *n == 0), "an untreated run swapped");
    }
    let d = block_delta(&after, &before);
    assert_eq!(d.swaps, 0);
    assert_eq!(d.repeat_swaps, 0);
    assert_eq!(
        d.swap_count_hist_1 + d.swap_count_hist_2 + d.swap_count_hist_3 + d.swap_count_hist_4plus,
        0,
        "a ghost was counted as displaced"
    );
    assert_eq!(d.census.treated.contested_dispatches, 0);
    assert_eq!(d.census.treated.ghost_entries_from_restarted_origin, 0);
    assert!(
        d.census.control.contested_dispatches > 0,
        "the control census saw no contest: {d:?}"
    );

    for id in &untreated {
        let (_, variant) = runs(&first)[id];
        assert_eq!(
            variant & run_variant::FRESH_FIRST_PAIR,
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
