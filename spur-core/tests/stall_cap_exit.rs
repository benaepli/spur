//! A run the stall cap ends must stop on the first step whose quiet gap
//! exceeds the cap, leave a runs-table row naming the exit and the steps it
//! ran, and flush its history rows. An untreated run with the same gap runs
//! to its budget and reports the gap against the standing cap; a probe of
//! either stream runs to its budget and feeds nothing.
//!
//! On the release half of the treated runs the first stall settles the
//! in-progress client operations for the plan instead: the events behind
//! them issue, the run ends by plan completion or at its next stall, a real
//! response that arrives after the settlement is recorded once, and the cut
//! half is unchanged.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, RunOutcome, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::run_cap;
use spur_core::simulator::run_variant;
use spur_core::simulator::stall_cap;
use spur_core::simulator::stall_release;
use spur_core::simulator::timer_context;
use spur_core::simulator::util_stats::{self, StallCapCell, StallReleaseCell};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPEC: &str = include_str!("fixtures/stall.spur");
const LATE_SPEC: &str = include_str!("fixtures/stall_release.spur");

/// One write that never succeeds, so every step after the request is issued
/// is quiet and the run can only end by a cap or its budget.
const CONFIG: &str = r#"{
  "params": {"n": {"min": 3, "max": 3, "step": 1}},
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

/// Two writes in a chain, so the second is planned behind the first and
/// can only issue once the first is completed or settled.
const CHAIN_CONFIG: &str = r#"{
  "params": {"n": {"min": 1, "max": 1, "step": 1}},
  "num_write_ops": {"min": 2, "max": 2, "step": 1},
  "num_read_ops": {"min": 0, "max": 0, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "max_concurrent_writes": {"min": 1, "max": 1, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 1,
  "max_iterations": 256,
  "session_seed": 4343,
  "strict_config_keys": true,
  "stats": true
}"#;

/// Three writes in a chain, so a write is planned behind the settled one
/// and another behind that.
const LATE_CONFIG: &str = r#"{
  "params": {"n": {"min": 1, "max": 1, "step": 1}},
  "num_write_ops": {"min": 3, "max": 3, "step": 1},
  "num_read_ops": {"min": 0, "max": 0, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "max_concurrent_writes": {"min": 1, "max": 1, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 1,
  "max_iterations": 256,
  "session_seed": 4343,
  "strict_config_keys": true,
  "stats": true
}"#;

const BACKUP: i32 = 256;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(name);
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

/// The first run id at or above 1 in the given stall-cap cell, with the
/// run-cap probe role as asked; a treated id is taken from the cut half of
/// the stall release so it behaves as any treated run does without it.
fn first_id(cell: StallCapCell, run_cap_probe: bool) -> i64 {
    (1..10_000i64)
        .find(|&id| {
            stall_cap::cell(id) == cell
                && run_cap::is_probe(id) == run_cap_probe
                && stall_release::cell(id) != StallReleaseCell::Release
        })
        .expect("the cell is reachable")
}

/// The first run id at or above 1 in the given stall-release cell.
fn first_release_id(cell: StallReleaseCell) -> i64 {
    (1..10_000i64)
        .find(|&id| stall_release::cell(id) == cell)
        .expect("the cell is reachable")
}

/// Seed the stall-cap learner so the cap for `BACKUP` is 15: 200 completed
/// probes with a longest gap of 10 clear the sample floor; with a bucket
/// width of 1 the quantile's upper edge is 10.
fn seed_cap() -> i32 {
    run_cap::reset();
    stall_cap::reset();
    for _ in 0..200 {
        stall_cap::merge_probe(BACKUP, 10);
    }
    let cap = stall_cap::effective_cap(BACKUP).expect("the seeded probes engage the cap");
    assert_eq!(cap, 15);
    assert_eq!(run_cap::effective_cap(BACKUP), BACKUP, "the step cap stays identity");
    cap
}

/// The history rows of one run: (unique_id, kind) in sequence order.
fn history_rows(out: &Path, run_id: i64) -> Vec<(i64, String)> {
    let run_ids = int_column(out, "executions", "run_id");
    let uids = int_column(out, "executions", "unique_id");
    let kinds = string_column(out, "executions", "kind");
    let seqs = int_column(out, "executions", "seq_num");
    let mut rows: Vec<(i64, i64, String)> = (0..run_ids.len())
        .filter(|&i| run_ids[i] == run_id)
        .map(|i| (seqs[i], uids[i], kinds[i].clone()))
        .collect();
    rows.sort();
    rows.into_iter().map(|(_, u, k)| (u, k)).collect()
}

fn runs_row(out: &Path, run_id: i64) -> (String, i64) {
    let run_ids = int_column(out, "runs", "run_id");
    let reasons = string_column(out, "runs", "end_reason");
    let steps = int_column(out, "runs", "steps_used");
    let i = run_ids
        .iter()
        .position(|&r| r == run_id)
        .expect("a runs row per run");
    (reasons[i].clone(), steps[i])
}

fn variant_of(out: &Path, run_id: i64) -> i32 {
    let run_ids = int_column(out, "runs", "run_id");
    let variants = int_column(out, "runs", "variant");
    let i = run_ids
        .iter()
        .position(|&r| r == run_id)
        .expect("a runs row per run");
    variants[i] as i32
}

fn run_session(
    spec: &str,
    config: &str,
    out: &Path,
    run_ids: &[i64],
) -> Vec<(i64, RunOutcome)> {
    let program = compiler::compile(spec, "stall.spur")
        .into_program()
        .expect("spec compiles");
    let mut config: ExplorerConfig = serde_json::from_str(config).expect("config parses");
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
    assert_eq!(run_config.max_iterations, BACKUP);
    let writer: Arc<dyn HistoryWriter> = Arc::from(
        create_writer(LogBackend::Parquet, out.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    );
    let global_state = GlobalState::<NoFeedback>::new();
    let mut outcomes = Vec::new();
    for &run_id in run_ids {
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
    outcomes
}

fn on_session_thread(f: fn()) {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(f)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

#[test]
fn a_treated_run_is_cut_when_its_gap_exceeds_the_cap_and_the_others_are_not() {
    on_session_thread(check);
}

fn check() {
    let _serial = config_override::exclusive_session();
    util_stats::set_enabled(true);
    let cap = seed_cap();

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
    let released = first_release_id(StallReleaseCell::Release);
    assert_eq!(stall_release::cell(treated), StallReleaseCell::Cut);

    let out = scratch("spur_stall_cap_exit");
    let before = util_stats::snapshot();
    let outcomes = run_session(
        SPEC,
        CONFIG,
        &out,
        &[treated, untreated, cap_probe, timer_probe, released],
    );
    let after = util_stats::snapshot();
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
        .zip(&outcomes[1..4])
    {
        assert!(
            matches!(outcome, RunOutcome::IterationsExhausted { .. }),
            "the {what} run must run to its budget, got {outcome:?}"
        );
    }
    // The released run settles its only operation at the same first stall,
    // which completes its plan, so it ends on the next step by plan
    // completion with the invocation still pending in the history.
    assert_eq!(
        outcomes[4].1,
        RunOutcome::Completed { steps: stop_step },
        "the released run's plan completes once its stuck operation is settled"
    );

    let (before_sc, after_sc) = (&before.stall_cap, &after.stall_cap);
    assert_eq!(after_sc.stops, before_sc.stops + 1, "the release is not a stop");
    assert_eq!(after_sc.treated_runs, before_sc.treated_runs + 2);
    assert_eq!(after_sc.untreated_runs, before_sc.untreated_runs + 1);
    assert_eq!(after_sc.untreated_runs_capped, before_sc.untreated_runs_capped + 1);
    assert_eq!(
        after_sc.untreated_over_cap_runs,
        before_sc.untreated_over_cap_runs + 1,
        "the untreated run's gap exceeded the standing cap"
    );
    assert_eq!(
        after_sc.steps_saved_sum,
        before_sc.steps_saved_sum + (BACKUP - stop_step) as u64
    );
    assert_eq!(after_sc.probes_keyed, before_sc.probes_keyed, "exhausted probes feed nothing");
    assert!(after_sc.marks.releases >= before_sc.marks.releases + 6, "each run released its request and the settlement is a release mark");
    assert!(after_sc.marks.rows >= before_sc.marks.rows + 5, "each run recorded its invocation");
    assert_eq!(
        after.termination.all.stall_cap_reached,
        before.termination.all.stall_cap_reached + 1
    );
    assert_eq!(
        rows,
        format!(
            "run_id,longest_quiet_gap,stall_cap_standing\n{untreated},{},{cap}\n",
            BACKUP - 1
        ),
        "the untreated run's row carries its gap and the standing cap"
    );

    let (before_sr, after_sr) = (&before.stall_release, &after.stall_release);
    assert_eq!(after_sr.releases, before_sr.releases + 1);
    assert_eq!(after_sr.ops_settled, before_sr.ops_settled + 1);
    assert_eq!(after_sr.dependents_released, before_sr.dependents_released, "nothing was planned behind the write");
    assert_eq!(after_sr.late_responses, before_sr.late_responses);
    assert_eq!(after_sr.plan_completed_after_release, before_sr.plan_completed_after_release + 1);
    assert_eq!(after_sr.second_stall_stops, before_sr.second_stall_stops);
    assert_eq!(after_sr.steps_after_release_sum, before_sr.steps_after_release_sum);
    assert_eq!(after_sr.stalls_without_ops, before_sr.stalls_without_ops);
    assert_eq!(after_sr.release_cell.runs, before_sr.release_cell.runs + 1);
    assert_eq!(after_sr.release_cell.invocations, before_sr.release_cell.invocations + 1);
    assert_eq!(after_sr.release_cell.plan_complete, before_sr.release_cell.plan_complete + 1);
    assert_eq!(after_sr.cut_cell.runs, before_sr.cut_cell.runs + 1);
    assert_eq!(after_sr.cut_cell.invocations, before_sr.cut_cell.invocations + 1);
    assert_eq!(after_sr.cut_cell.plan_complete, before_sr.cut_cell.plan_complete);

    assert_eq!(runs_row(&out, treated), ("stall_cap_reached".to_string(), stop_step as i64));
    assert_eq!(runs_row(&out, untreated), ("iterations_exhausted".to_string(), BACKUP as i64));
    assert_eq!(runs_row(&out, cap_probe), ("iterations_exhausted".to_string(), BACKUP as i64));
    assert_eq!(runs_row(&out, timer_probe), ("iterations_exhausted".to_string(), BACKUP as i64));
    assert_eq!(runs_row(&out, released), ("plan_complete".to_string(), stop_step as i64));
    for id in [treated, released] {
        assert_eq!(
            history_rows(&out, id),
            vec![(1, "Invocation".to_string())],
            "run {id}: the stuck write stays a pending invocation"
        );
    }
    let mask = run_variant::STALL_CAP | run_variant::STALL_RELEASE;
    assert_eq!(variant_of(&out, treated) & mask, run_variant::STALL_CAP);
    assert_eq!(variant_of(&out, released) & mask, mask, "the release bit rides on the cap bit");
    for id in [untreated, cap_probe, timer_probe] {
        assert_eq!(variant_of(&out, id) & mask, 0, "run {id} carries neither bit");
    }
    let _ = fs::remove_dir_all(&out);
}

#[test]
fn a_released_run_issues_the_events_behind_its_stuck_operation_and_stops_at_its_second_stall() {
    on_session_thread(check_chain);
}

fn check_chain() {
    let _serial = config_override::exclusive_session();
    util_stats::set_enabled(true);
    let cap = seed_cap();
    let released = first_release_id(StallReleaseCell::Release);
    let cut = first_release_id(StallReleaseCell::Cut);

    let out = scratch("spur_stall_release_chain");
    let before = util_stats::snapshot();
    let outcomes = run_session(SPEC, CHAIN_CONFIG, &out, &[released, cut]);
    let after = util_stats::snapshot();
    run_cap::reset();
    stall_cap::reset();
    util_stats::set_enabled(false);

    // The cut run stops at its first stall with both writes outstanding.
    // The released run settles the first write there instead, the second
    // write issues on the next step and is the run's last mark, and the
    // run stops when the gap after it exceeds the cap again.
    let first_stop = cap + 2;
    let second_stop = 2 * cap + 4;
    assert_eq!(
        outcomes[1].1,
        RunOutcome::StallCapReached {
            cap,
            step: first_stop,
            outstanding_events: 2,
        },
        "the cut run is what any treated run is"
    );
    assert_eq!(
        outcomes[0].1,
        RunOutcome::StallCapReached {
            cap,
            step: second_stop,
            outstanding_events: 1,
        },
        "the released run ends at its second stall with the second write outstanding"
    );

    let (before_sr, after_sr) = (&before.stall_release, &after.stall_release);
    assert_eq!(after_sr.releases, before_sr.releases + 1, "released exactly once");
    assert_eq!(after_sr.ops_settled, before_sr.ops_settled + 1);
    assert_eq!(after_sr.dependents_released.client, before_sr.dependents_released.client + 1);
    assert_eq!(after_sr.dependents_released.fault, before_sr.dependents_released.fault);
    assert_eq!(after_sr.dependents_released.other, before_sr.dependents_released.other);
    assert_eq!(after_sr.late_responses, before_sr.late_responses);
    assert_eq!(after_sr.plan_completed_after_release, before_sr.plan_completed_after_release);
    assert_eq!(after_sr.second_stall_stops, before_sr.second_stall_stops + 1);
    assert_eq!(
        after_sr.steps_after_release_sum,
        before_sr.steps_after_release_sum + (second_stop - first_stop) as u64
    );
    assert_eq!(after_sr.stalls_without_ops, before_sr.stalls_without_ops);
    assert_eq!(after_sr.release_cell.runs, before_sr.release_cell.runs + 1);
    assert_eq!(after_sr.release_cell.invocations, before_sr.release_cell.invocations + 2);
    assert_eq!(after_sr.release_cell.plan_complete, before_sr.release_cell.plan_complete);
    assert_eq!(after_sr.cut_cell.runs, before_sr.cut_cell.runs + 1);
    assert_eq!(after_sr.cut_cell.invocations, before_sr.cut_cell.invocations + 1);
    assert_eq!(after_sr.cut_cell.plan_complete, before_sr.cut_cell.plan_complete);

    let (before_sc, after_sc) = (&before.stall_cap, &after.stall_cap);
    assert_eq!(after_sc.stops, before_sc.stops + 2, "one stop per run, at the second stall for the released one");
    assert_eq!(after_sc.treated_runs, before_sc.treated_runs + 2);
    assert_eq!(
        after_sc.steps_saved_sum,
        before_sc.steps_saved_sum + (BACKUP - first_stop) as u64 + (BACKUP - second_stop) as u64
    );
    assert_eq!(
        after.termination.all.stall_cap_reached,
        before.termination.all.stall_cap_reached + 2
    );

    assert_eq!(runs_row(&out, cut), ("stall_cap_reached".to_string(), first_stop as i64));
    assert_eq!(runs_row(&out, released), ("stall_cap_reached".to_string(), second_stop as i64));
    assert_eq!(
        history_rows(&out, cut),
        vec![(1, "Invocation".to_string())],
        "the cut run never issues the second write"
    );
    assert_eq!(
        history_rows(&out, released),
        vec![(1, "Invocation".to_string()), (2, "Invocation".to_string())],
        "the released run issues the second write and neither write responds"
    );
    let _ = fs::remove_dir_all(&out);
}

#[test]
fn a_late_response_after_settlement_is_recorded_once_and_the_plan_completes() {
    on_session_thread(check_late);
}

fn check_late() {
    let _serial = config_override::exclusive_session();
    util_stats::set_enabled(true);
    let cap = seed_cap();
    let released = first_release_id(StallReleaseCell::Release);
    let cut = first_release_id(StallReleaseCell::Cut);

    let out = scratch("spur_stall_release_late");
    let before = util_stats::snapshot();
    let outcomes = run_session(LATE_SPEC, LATE_CONFIG, &out, &[released, cut]);
    let after = util_stats::snapshot();
    run_cap::reset();
    stall_cap::reset();
    util_stats::set_enabled(false);

    // The first write is stuck until the second reaches the node, and the
    // third succeeds only once the first has been answered. The cut run
    // stops at its first stall. The released run issues the second write,
    // which succeeds and unblocks the first; the first's late response then
    // lets the third succeed and the plan completes.
    match &outcomes[1].1 {
        RunOutcome::StallCapReached {
            cap: got,
            step,
            outstanding_events,
        } => {
            assert_eq!(*got, cap);
            assert!(*step > cap, "the cut run ran past the cap before stopping");
            assert_eq!(*outstanding_events, 3);
        }
        other => panic!("expected a stall-cap exit for the cut run, got {other:?}"),
    }
    let completed_at = match &outcomes[0].1 {
        RunOutcome::Completed { steps } => {
            assert!(*steps > cap + 2, "the plan completes after the release");
            *steps
        }
        other => panic!("expected plan completion for the released run, got {other:?}"),
    };

    let (before_sr, after_sr) = (&before.stall_release, &after.stall_release);
    assert_eq!(after_sr.releases, before_sr.releases + 1);
    assert_eq!(after_sr.ops_settled, before_sr.ops_settled + 1);
    assert_eq!(after_sr.dependents_released.client, before_sr.dependents_released.client + 1);
    assert_eq!(after_sr.late_responses, before_sr.late_responses + 1, "the first write's real response arrived after its settlement");
    assert_eq!(after_sr.plan_completed_after_release, before_sr.plan_completed_after_release + 1);
    assert_eq!(after_sr.second_stall_stops, before_sr.second_stall_stops);
    assert!(
        after_sr.steps_after_release_sum > before_sr.steps_after_release_sum
            && after_sr.steps_after_release_sum
                <= before_sr.steps_after_release_sum + completed_at as u64,
        "the steps after the release are those up to completion"
    );
    assert_eq!(after_sr.release_cell.runs, before_sr.release_cell.runs + 1);
    assert_eq!(after_sr.release_cell.invocations, before_sr.release_cell.invocations + 3);
    assert_eq!(after_sr.release_cell.plan_complete, before_sr.release_cell.plan_complete + 1);
    assert_eq!(after_sr.cut_cell.runs, before_sr.cut_cell.runs + 1);
    assert_eq!(after_sr.cut_cell.invocations, before_sr.cut_cell.invocations + 1);
    assert_eq!(after_sr.cut_cell.plan_complete, before_sr.cut_cell.plan_complete);
    assert_eq!(after.stall_cap.stops, before.stall_cap.stops + 1, "only the cut run stopped");
    assert_eq!(
        after.termination.all.plan_complete,
        before.termination.all.plan_complete + 1
    );

    assert_eq!(runs_row(&out, released), ("plan_complete".to_string(), completed_at as i64));
    let rows = history_rows(&out, released);
    let mut responses: Vec<i64> = rows
        .iter()
        .filter(|(_, k)| k == "Response")
        .map(|(u, _)| *u)
        .collect();
    responses.sort_unstable();
    let invocations: Vec<i64> = rows
        .iter()
        .filter(|(_, k)| k == "Invocation")
        .map(|(u, _)| *u)
        .collect();
    assert_eq!(invocations, vec![1, 2, 3]);
    assert_eq!(
        responses,
        vec![1, 2, 3],
        "the settled first write's response is recorded once, beside the others"
    );
    assert_eq!(rows.len(), 6, "no other history row was added by the release");
    let _ = fs::remove_dir_all(&out);
}
