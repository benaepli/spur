//! The pair-order dispatch preference has to fire on a real workload and
//! change only what it claims to: over a session of treated runs on a
//! fixture whose handlers send two messages in a row to every peer and whose
//! nodes crash with those sends in the air, a network step whose pick is a
//! later send of an incarnation that is no longer running must take the
//! eligible earlier send of the same class instead; every such contest must
//! end with the earliest send taken; and no message entry of that class may
//! ever be taken ahead of an earlier queued send of its class, while on the
//! control half such inversions happen. A pick of the sender's current
//! incarnation keeps the order the draw gave it, so entries of that class
//! still invert on the treated half, and the replacements the preference
//! declines to make are counted. On the untreated half nothing may change:
//! the same run id gives the same event sequence twice and no pick is
//! replaced, while the census still counts. The census is read on a salted
//! sixteenth of the runs, so the sessions here are drawn from census runs;
//! an untreated run outside the sample leaves every census counter alone.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, SingleRunConfig, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::util_stats::{
    PairOrderClassCounts, PairOrderHalfStats, PairOrderStats, UtilizationSnapshot,
};
use spur_core::simulator::{fault_timing, pair_order, run_cap, run_variant, util_stats};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/pairorder.spur");

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
  "queue_policy": {"type": "Probabilistic", "p_local": 0.6, "p_timer": 0.1},
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
const UNSAMPLED_RUNS: usize = 12;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("spur_pair_order_{name}"));
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
    let program = compiler::compile(SPEC, "pairorder.spur")
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
/// in `pair_order.corrected` it produced.
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
    let mut corrected_by_run = BTreeMap::new();
    for &run_id in ids {
        let before = util_stats::snapshot().pair_order.corrected;
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
        let after = util_stats::snapshot().pair_order.corrected;
        corrected_by_run.insert(run_id, after - before);
    }
    writer.shutdown();
    corrected_by_run
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

fn ids(treated: bool, census: bool, n: usize) -> Vec<i64> {
    let ids: Vec<i64> = (0..1_000_000i64)
        .filter(|&id| {
            pair_order::is_treated(id) == treated
                && pair_order::is_census_run(id) == census
                && fault_timing::is_placed(id)
        })
        .take(n)
        .collect();
    assert_eq!(ids.len(), n, "not enough run ids on the half");
    ids
}

fn half_delta(a: &PairOrderHalfStats, b: &PairOrderHalfStats) -> PairOrderHalfStats {
    PairOrderHalfStats {
        contests: a.contests - b.contests,
        inorder_draws: a.inorder_draws - b.inorder_draws,
        pair_entries: a.pair_entries - b.pair_entries,
        pair_entries_ghost: a.pair_entries_ghost - b.pair_entries_ghost,
        inversions: a.inversions - b.inversions,
        inversions_ghost: a.inversions_ghost - b.inversions_ghost,
        sampled_runs: a.sampled_runs - b.sampled_runs,
    }
}

fn block_delta(after: &UtilizationSnapshot, before: &UtilizationSnapshot) -> PairOrderStats {
    let (a, b) = (&after.pair_order, &before.pair_order);
    PairOrderStats {
        corrected: a.corrected - b.corrected,
        contests_by_class: PairOrderClassCounts {
            ghost: a.contests_by_class.ghost - b.contests_by_class.ghost,
            fresh: a.contests_by_class.fresh - b.contests_by_class.fresh,
        },
        corrections_ghost: a.corrections_ghost - b.corrections_ghost,
        corrections_fresh: a.corrections_fresh - b.corrections_fresh,
        corrections_fresh_suppressed: a.corrections_fresh_suppressed
            - b.corrections_fresh_suppressed,
        census: spur_core::simulator::util_stats::PairOrderCensusStats {
            treated: half_delta(&a.census.treated, &b.census.treated),
            control: half_delta(&a.census.control, &b.census.control),
        },
    }
}

fn share(num: u64, den: u64) -> f64 {
    if den == 0 { 0.0 } else { num as f64 / den as f64 }
}

#[test]
fn treated_runs_take_the_earliest_send_at_every_contest_and_never_invert_a_pair() {
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
    let treated = ids(true, true, HALF_RUNS);

    let out = scratch("treated");
    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    let corrected_by_run = session(&program, &run_config, &treated, &out);
    let after_treated = util_stats::snapshot();
    let d = block_delta(&after_treated, &before);

    assert!(d.corrected > 0, "no contested step ever replaced its pick: {d:?}");
    let t = &d.census.treated;
    assert!(t.contests > 0, "the treated census is empty: {d:?}");
    assert!(
        t.inorder_draws > 0 && t.inorder_draws < t.contests,
        "the draw never fell on both the earlier and the later send: {d:?}"
    );
    assert_eq!(
        d.contests_by_class.ghost + d.contests_by_class.fresh,
        t.contests,
        "a treated contest was counted without a class: {d:?}"
    );
    assert!(
        d.contests_by_class.fresh > 0,
        "no contest fell on the incarnation running now, so the class test is untested: {d:?}"
    );
    assert_eq!(
        d.corrections_ghost, d.corrected,
        "a replacement landed outside the dead class: {d:?}"
    );
    assert_eq!(
        d.corrections_fresh, 0,
        "a pick of the incarnation running now was replaced: {d:?}"
    );
    assert!(
        d.corrections_fresh_suppressed > 0,
        "no replacement was declined, so the class test never bound: {d:?}"
    );
    assert_eq!(
        t.contests - t.inorder_draws,
        d.corrected + d.corrections_fresh_suppressed,
        "a treated contest with an earlier eligible send was neither replaced nor declined: {d:?}"
    );
    assert_eq!(t.sampled_runs as usize, HALF_RUNS, "every treated run here is a census run: {d:?}");
    assert!(t.pair_entries > 0, "no entry from a crashed sender had a sibling: {d:?}");
    assert!(t.pair_entries_ghost > 0, "no entry of the dead class had a sibling: {d:?}");
    assert_eq!(
        t.inversions_ghost, 0,
        "a treated run took a later send of a dead incarnation ahead of an earlier queued one: {d:?}"
    );
    assert!(
        t.inversions > 0,
        "the incarnation running now never inverted, so the class test changed nothing: {d:?}"
    );
    assert_eq!(
        d.census.control.contests, 0,
        "a treated session counted in the control half: {d:?}"
    );
    assert_eq!(
        d.census.control.pair_entries, 0,
        "a treated session counted in the control half: {d:?}"
    );

    let treated_runs = runs(&out);
    assert_eq!(treated_runs.len(), HALF_RUNS, "one runs row per run");
    let mut ends: BTreeMap<(String, bool), usize> = BTreeMap::new();
    for id in &treated {
        *ends.entry((treated_runs[id].0.clone(), corrected_by_run[id] > 0)).or_insert(0) += 1;
    }
    eprintln!("treated end reasons by (reason, corrected): {ends:?}");
    let corrected: Vec<i64> = corrected_by_run
        .iter()
        .filter(|(_, n)| **n > 0)
        .map(|(id, _)| *id)
        .collect();
    assert!(!corrected.is_empty(), "the counter moved but no run owns it");
    for id in &treated {
        let (end, variant) = &treated_runs[id];
        assert_ne!(
            variant & run_variant::PAIR_SEND_ORDER,
            0,
            "run {id} is not tagged as treated"
        );
        if corrected_by_run[id] > 0 {
            assert_eq!(end, "plan_complete", "run {id} replaced a pick and then did not complete its plan");
        }
    }

    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let control = ids(false, true, HALF_RUNS);
    let control_out = scratch("control");
    let control_corrected = session(&program, &run_config, &control, &control_out);
    let after_control = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();
    let c = block_delta(&after_control, &after_treated);
    let control_runs = runs(&control_out);
    let mut control_ends: BTreeMap<String, usize> = BTreeMap::new();
    for id in &control {
        *control_ends.entry(control_runs[id].0.clone()).or_insert(0) += 1;
    }
    eprintln!("control end reasons: {control_ends:?}");

    assert!(
        control_corrected.values().all(|n| *n == 0) && c.corrected == 0,
        "a control run replaced its pick: {c:?}"
    );
    assert_eq!(c.census.treated.contests, 0, "a control session counted in the treated half");
    let cc = &c.census.control;
    assert_eq!(cc.sampled_runs as usize, HALF_RUNS, "every control run here is a census run: {c:?}");
    assert!(cc.contests > 0, "the control census is empty: {c:?}");
    assert!(cc.inorder_draws < cc.contests, "the control draw never fell on a later send: {c:?}");
    assert!(cc.pair_entries > 0, "no entry from a crashed sender had a sibling on the control half: {c:?}");
    assert!(cc.inversions > 0, "the control half never took a later send first: {c:?}");
    assert!(
        cc.inversions_ghost > 0,
        "the control half never took a later send of a dead incarnation first: {c:?}"
    );
    assert_eq!(
        c.contests_by_class.ghost + c.contests_by_class.fresh,
        0,
        "a control session counted a class: {c:?}"
    );

    eprintln!("pair_order treated session: {d:?}");
    eprintln!("pair_order control session: {c:?}");
    eprintln!(
        "inorder share treated {:.3} control {:.3}; inversion share treated {:.3} control {:.3}",
        share(t.inorder_draws, t.contests),
        share(cc.inorder_draws, cc.contests),
        share(t.inversions, t.pair_entries),
        share(cc.inversions, cc.pair_entries)
    );
    eprintln!(
        "ghost-class inversion share treated {:.3} control {:.3}; corrections {} ghost, {} declined on the incarnation running now",
        share(t.inversions_ghost, t.pair_entries_ghost),
        share(cc.inversions_ghost, cc.pair_entries_ghost),
        d.corrections_ghost,
        d.corrections_fresh_suppressed
    );
    let _ = fs::remove_dir_all(&out);
    let _ = fs::remove_dir_all(&control_out);
}

#[test]
fn untreated_runs_are_unchanged_and_never_replace_a_pick() {
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
    let untreated = ids(false, true, UNTREATED_RUNS);

    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    seed_span();
    let first = scratch("untreated_first");
    let corrected_first = session(&program, &run_config, &untreated, &first);
    run_cap::reset();
    fault_timing::reset();
    seed_span();
    let second = scratch("untreated_second");
    let corrected_second = session(&program, &run_config, &untreated, &second);
    let after = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    for by_run in [&corrected_first, &corrected_second] {
        assert!(by_run.values().all(|n| *n == 0), "an untreated run replaced its pick");
    }
    let d = block_delta(&after, &before);
    assert_eq!(d.corrected, 0);
    assert_eq!(d.census.treated.contests, 0);
    assert_eq!(d.census.treated.inorder_draws, 0);
    assert_eq!(d.census.treated.pair_entries, 0);
    assert_eq!(d.census.treated.inversions, 0);
    assert!(
        d.census.control.contests > 0,
        "the control census saw no contest: {d:?}"
    );
    assert!(
        d.census.control.pair_entries > 0,
        "the control census saw no pair entry: {d:?}"
    );

    for id in &untreated {
        let (_, variant) = runs(&first)[id];
        assert_eq!(
            variant & run_variant::PAIR_SEND_ORDER,
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

#[test]
fn untreated_runs_outside_the_census_sample_count_nothing() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_unsampled)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_unsampled() {
    let _serial = config_override::exclusive_session();
    run_cap::reset();
    fault_timing::reset();
    let (program, run_config) = compile();
    let unsampled = ids(false, false, UNSAMPLED_RUNS);

    util_stats::set_enabled(true);
    let before = util_stats::snapshot();
    seed_span();
    let out = scratch("unsampled");
    let corrected = session(&program, &run_config, &unsampled, &out);
    let after = util_stats::snapshot();
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();

    assert!(corrected.values().all(|n| *n == 0), "an untreated run replaced its pick");
    let d = block_delta(&after, &before);
    assert_eq!(d.corrected, 0);
    for half in [&d.census.treated, &d.census.control] {
        assert_eq!(half.sampled_runs, 0, "a run outside the sample was counted as a census run: {d:?}");
        assert_eq!(half.contests, 0, "a run outside the sample counted a contest: {d:?}");
        assert_eq!(half.pair_entries, 0, "a run outside the sample counted a pair entry: {d:?}");
    }
    assert_eq!(runs(&out).len(), UNSAMPLED_RUNS, "one runs row per run");
    let _ = fs::remove_dir_all(&out);
}
