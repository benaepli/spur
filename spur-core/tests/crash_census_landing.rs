//! Every census of what a crash lands on has to describe the node the crash
//! was applied to. When a planned crash moves to another node, the crash
//! anchor, the crash census and the per-half victim-swap census all read
//! that node's ledger, so over any session the whole-population counters
//! equal the sum of the treated and control halves exactly.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::{
    ExplorerConfig, GlobalState, NoFeedback, RunAttribution, SingleRunConfig, run_single_simulation,
};
use spur_core::simulator::history::{HistoryWriter, LogBackend, create_writer};
use spur_core::simulator::rng::LiveRng;
use spur_core::simulator::{fault_timing, ghost_absorber, run_cap, util_stats};
use std::fs;
use std::path::PathBuf;
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
const RUNS_PER_HALF: usize = 40;
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join("spur_crash_census_landing");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
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

#[test]
fn the_crash_censuses_agree_on_the_node_the_crash_landed_on() {
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
    seed_span();
    let (program, run_config) = compile();
    let placed = |treated: bool| -> Vec<i64> {
        (0..1_000_000i64)
            .filter(|&id| ghost_absorber::is_treated(id) == treated && fault_timing::is_placed(id))
            .take(RUNS_PER_HALF)
            .collect()
    };
    let mut ids = placed(true);
    ids.extend(placed(false));

    let out = scratch();
    let writer: Arc<dyn HistoryWriter> = Arc::from(
        create_writer(LogBackend::Parquet, out.to_str().expect("utf-8 path"))
            .expect("creates writer"),
    );
    let global_state = GlobalState::<NoFeedback>::new();
    util_stats::set_enabled(true);
    util_stats::set_crash_census_enabled(true);
    let before = util_stats::snapshot();
    for &run_id in &ids {
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
    let after = util_stats::snapshot();
    util_stats::set_crash_census_enabled(false);
    util_stats::set_enabled(false);
    run_cap::reset();
    fault_timing::reset();
    let _ = fs::remove_dir_all(&out);

    let moved = after.victim_swap.applied - before.victim_swap.applied;
    assert!(moved > 0, "no planned crash moved, so the landing node never differed");

    let census = |s: &util_stats::UtilizationSnapshot| {
        let c = &s.delivery_effects.crash_census;
        (c.decisions, c.victim_had_inflight_sends)
    };
    let (decisions, inflight) = {
        let (d1, i1) = census(&before);
        let (d2, i2) = census(&after);
        (d2 - d1, i2 - i1)
    };
    let crashes_taken = after.crash_anchor.crashes_taken - before.crash_anchor.crashes_taken;
    let applied = after.crash_anchor.applied - before.crash_anchor.applied;
    let halves = |s: &util_stats::UtilizationSnapshot| {
        let c = &s.victim_swap.census;
        (
            c.treated.crashes + c.control.crashes,
            c.treated.victim_had_inflight_sends + c.control.victim_had_inflight_sends,
        )
    };
    let (half_crashes, half_inflight) = {
        let (c1, i1) = halves(&before);
        let (c2, i2) = halves(&after);
        (c2 - c1, i2 - i1)
    };
    assert!(decisions > 0, "no crash was applied");
    assert!(inflight > 0, "no crash landed on a node with a send in flight");
    assert_eq!(decisions, crashes_taken, "the census and the anchor count different crashes");
    assert_eq!(decisions, half_crashes, "the halves do not add up to the crashes applied");
    assert_eq!(
        inflight, half_inflight,
        "the census and the halves read different nodes' ledgers"
    );
    assert_eq!(
        applied, half_inflight,
        "the anchor and the halves read different nodes' ledgers"
    );
}
