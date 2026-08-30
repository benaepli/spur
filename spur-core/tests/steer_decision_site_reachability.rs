//! Whether the scoring function is ever in a position to name a runnable that
//! priority alone would not have named, under the settings the general
//! exploration config runs with.
//!
//! The aggregate counters cannot answer that: the count of expressed
//! preferences is gated by the same condition it would have to report on, so a
//! zero there reads equally as "the ranking never disagreed" and "the ranking
//! never ran". This drives short sessions directly and reads the census that
//! records, once per step, how far that step travelled - which turns the zero
//! into the name of the condition that stopped it.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::run_explorer;
use spur_core::simulator::history::LogBackend;
use spur_core::simulator::util_stats::{self, SteerReachStats};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("../../../bin/spur/VR.spur");

/// The exploration settings of the general config's unoverlaid arm, on a grid
/// small enough to run in a test. Only the weight on one score predicate and
/// the session seed vary.
const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 2, "max": 2, "step": 1},
  "num_read_ops": {"min": 4, "max": 4, "step": 2},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 1, "max": 2, "step": 1},
  "max_concurrent_writes": {"min": 2, "max": 2, "step": 1},
  "dependency_density": [0.0],
  "post_fault_client_ops": 1,
  "purgatory": {
    "delay_probability": 0.15,
    "delay_duration_range": [5, 300],
    "hold_down_receivers": false
  },
  "feedback": {
    "mode": "timeline",
    "steer": true,
    "timeline_key_granularity": "fine",
    "steer_audit": true,
    "novelty_enabled": false
  },
  "steer_terms": {
    "crash_after_timer_sends": {{WEIGHT}},
    "crash_after_delivery_sends": 0,
    "stale_late": 0,
    "request_before_stale": 0
  },
  "stats": true,
  "num_runs_per_config": 3,
  "max_iterations": 400,
  "session_seed": {{SEED}},
  "strict_config_keys": true,
  "rng_stream_isolation": true
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

const SEEDS: [u64; 3] = [11, 2029, 90210];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("spur_steer_decision_site_reachability")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

/// Runs one session and returns the per-step census of how far each step got.
fn session(name: &str, seed: u64, weight: f64) -> SteerReachStats {
    let program = compiler::compile(SPEC, "spec.spur")
        .into_program()
        .expect("spec compiles");
    let config = CONFIG
        .replace("{{WEIGHT}}", &format!("{weight}"))
        .replace("{{SEED}}", &seed.to_string());
    let config_path = std::env::temp_dir().join(format!("spur_steer_reach_{name}.json"));
    fs::write(&config_path, config).expect("writes config");
    let out = scratch(name);
    let _scope = config_override::exclusive_session();
    let cancelled = Arc::new(AtomicBool::new(false));
    run_explorer(
        &program,
        config_path.to_str().expect("utf-8 path"),
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .expect("session runs");
    let census = util_stats::snapshot().steer_reach;
    let _ = fs::remove_file(&config_path);
    let _ = fs::remove_dir_all(&out);
    census
}

fn total(c: &SteerReachStats) -> u64 {
    c.no_schedule_attempt + c.audit_disabled + c.no_weighted_predicate + c.reached_decision()
}

#[test]
fn the_ranking_runs_only_when_a_predicate_carries_weight() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check() {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    for seed in SEEDS {
        let c = session(&format!("unweighted_{seed}"), seed, 0.0);
        println!("seed {seed}, no weight: {c:?}");
        assert!(total(&c) > 0, "seed {seed} took no steps: {c:?}");
        assert!(
            c.no_weighted_predicate > 0,
            "seed {seed} reached no scheduling point at all: {c:?}"
        );
        assert_eq!(
            c.reached_decision(),
            0,
            "seed {seed} ranked candidates with nothing carrying weight: {c:?}"
        );
    }

    for seed in SEEDS {
        let c = session(&format!("weighted_{seed}"), seed, 1.0);
        println!("seed {seed}, one predicate weighted: {c:?}");
        assert_eq!(
            c.no_weighted_predicate, 0,
            "seed {seed} skipped the ranking with a weight set: {c:?}"
        );
        assert!(
            c.reached_decision() > 0,
            "seed {seed} never ranked candidates: {c:?}"
        );
        assert!(
            c.reached_decision() > c.single_candidate,
            "seed {seed} was never offered more than one candidate, so the \
             ranking had nothing to disagree about: {c:?}"
        );
    }

    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_steer_decision_site_reachability"));
}
