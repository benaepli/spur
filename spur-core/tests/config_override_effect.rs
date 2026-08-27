//! An override has to reach the simulator, not just the config text.
//!
//! The counter pinned here is `crash_recovery.crashes`, raised deep in plan
//! execution and gated by the fault-injection range the override moves. An
//! assignment that stops at the loaded JSON makes the two sessions below
//! identical, which is what a study varying that knob would read as a flat
//! response to the knob rather than as broken plumbing.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::run_explorer;
use spur_core::simulator::history::LogBackend;
use spur_core::simulator::util_stats;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/kv.spur");

/// No faults, one fixed session seed, strict keys on. Every difference between
/// the two sessions below therefore comes from the override alone.
const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 1, "max": 1, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 0, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 8,
  "max_iterations": 2000,
  "session_seed": 424242,
  "stats": true,
  "strict_config_keys": true
}"#;

/// Compilation and interpretation both recurse through the spec, and an
/// unoptimized build spends far more stack per frame than the default a test
/// thread is given, so the work runs on a thread sized for the deepest nesting
/// rather than only passing when the crate is built with optimizations.
const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("spur_config_override_effect")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

/// Runs one session with `overrides` applied and returns the runs it executed
/// and the crashes those runs applied. `set_enabled` inside the explorer resets
/// the counters, so the guard has to be held across the snapshot as well as the
/// run for the numbers to belong to this session alone.
fn session(name: &str, config_path: &str, overrides: &[&str]) -> Result<(u64, u64), String> {
    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let out = scratch(name);
    let _scope =
        config_override::scoped_overrides(overrides.iter().map(|s| s.to_string()).collect());
    let cancelled = Arc::new(AtomicBool::new(false));
    run_explorer(
        &program,
        config_path,
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .map(|_| {
        let s = util_stats::snapshot().crash_recovery;
        (s.runs, s.crashes)
    })
    .map_err(|e| e.to_string())
}

#[test]
fn an_override_changes_what_the_simulator_does() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check_an_override_changes_what_the_simulator_does)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check_an_override_changes_what_the_simulator_does() {
    let config_path = std::env::temp_dir().join("spur_config_override_effect_config.json");
    fs::write(&config_path, CONFIG).expect("writes config");
    let config_path = config_path.to_str().expect("utf-8 path").to_string();

    let (runs, crashes) = session("baseline", &config_path, &[]).expect("baseline session runs");
    assert!(runs > 0, "the baseline session executed no runs at all");
    assert_eq!(
        crashes, 0,
        "the config injects no faults, so the session it describes must crash nothing"
    );

    let (runs, crashes) = session("overridden", &config_path, &["num_crashes.max=2"])
        .expect("overridden session runs");
    assert!(runs > 0, "the overridden session executed no runs at all");
    assert!(
        crashes > 0,
        "raising the fault-injection range left the simulator unchanged: \
         the override never reached the run"
    );

    // A path no config field claims is a hard error rather than a session that
    // measures the unchanged value.
    let nested = session("nested_typo", &config_path, &["num_crashes.maxx=2"])
        .expect_err("a misspelled nested path fails the session");
    assert!(nested.contains("num_crashes.maxx"), "{}", nested);

    let top = session("top_level_typo", &config_path, &["num_crashez=2"])
        .expect_err("a misspelled top-level path fails the session");
    assert!(top.contains("num_crashez"), "{}", top);

    let _ = fs::remove_file(&config_path);
    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_config_override_effect"));
}
