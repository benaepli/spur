//! A counter that exists in the utilization struct has to reach the JSON a
//! reader parses. Both export paths are checked against the struct itself:
//! the whole-snapshot dump the CLI writes, and the per-arm counter objects a
//! campaign builds by differencing and accumulating snapshots. Anything that
//! projects a subset of the fields on the way out fails here with the names
//! it dropped.

use serde_json::{Map, Value};
use spur_core::compiler;
use spur_core::simulator::campaign::run_explorer_campaign;
use spur_core::simulator::config_override;
use spur_core::simulator::history::LogBackend;
use spur_core::simulator::util_stats;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/kv.spur");

const CONFIG: &str = r#"{
  "num_servers": {"min": 3, "max": 3, "step": 1},
  "num_write_ops": {"min": 1, "max": 2, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 0, "max": 1, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 4,
  "max_iterations": 1500,
  "session_seed": 71717,
  "stats": true,
  "strict_config_keys": true,
  "feedback": {"mode": "timeline", "steer": true, "steer_audit": true, "steer_audit_always": true},
  "campaign": {
    "wall_budget_sec": 60,
    "deterministic_slice_runs": 32,
    "deterministic_rounds": 3,
    "batch_size": 8,
    "allocation": {"kind": "round_robin"},
    "reward": {"kind": "runs"},
    "arms": [
      {"id": "plain", "mode": "grid"},
      {"id": "shallow", "mode": "grid", "overlay": {"max_iterations": 400}}
    ]
  }
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

/// The names the evaluation record has to be able to read for a steer
/// hypothesis to be interpretable at all: without the consultation
/// denominator a zero elsewhere in the block cannot be told from a decision
/// site that never ran.
const REQUIRED: &[&str] = &[
    "steps",
    "audited",
    "preference_expressed",
    "preference_honored",
    "preference_consulted",
    "preference_source_absent",
];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("spur_stats_export_parity")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

fn keys(v: &Value, what: &str) -> BTreeSet<String> {
    let Value::Object(map) = v else {
        panic!("{what} is not a JSON object: {v}");
    };
    map.keys().cloned().collect()
}

/// Fails naming the difference in both directions, so a dropped field and a
/// stray one are distinguishable from the failure text alone.
fn assert_same(expected: &BTreeSet<String>, actual: &BTreeSet<String>, what: &str) {
    let missing: Vec<&String> = expected.difference(actual).collect();
    let extra: Vec<&String> = actual.difference(expected).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{what} does not carry the struct's fields: missing {missing:?}, extra {extra:?}"
    );
}

#[test]
fn every_steer_authority_field_reaches_the_emitted_json() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check() {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let program = compiler::compile(SPEC, "kv.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join("spur_stats_export_parity.json");
    fs::write(&config_path, CONFIG).expect("writes config");
    let out = scratch("campaign");
    let _scope = config_override::exclusive_session();
    let cancelled = Arc::new(AtomicBool::new(false));
    let summary = run_explorer_campaign(
        &program,
        config_path.to_str().expect("utf-8 path"),
        out.to_str().expect("utf-8 path"),
        LogBackend::Parquet,
        &cancelled,
    )
    .expect("campaign runs");
    let _ = fs::remove_file(&config_path);
    let _ = fs::remove_dir_all(&out);

    let snapshot = util_stats::snapshot();
    assert!(
        snapshot.steer_authority.steps > 0,
        "the campaign consumed no scheduling steps, so the export is untested"
    );

    let expected = keys(
        &serde_json::to_value(&snapshot.steer_authority).expect("the struct serializes"),
        "the steerAuthority struct",
    );
    for name in REQUIRED {
        assert!(
            expected.contains(*name),
            "the steerAuthority struct no longer has a `{name}` field"
        );
    }

    let dumped: Value = serde_json::from_str(
        &serde_json::to_string_pretty(&snapshot).expect("the snapshot serializes"),
    )
    .expect("the dump parses");
    assert_same(
        &keys(
            &serde_json::to_value(&snapshot).expect("the snapshot serializes"),
            "the utilization struct",
        ),
        &keys(&dumped, "the utilization dump"),
        "the utilization dump's blocks",
    );
    assert_same(
        &expected,
        &keys(&dumped["steer_authority"], "the utilization dump's steerAuthority"),
        "the utilization dump's steerAuthority",
    );

    let report = summary.campaign.expect("the campaign reports its arms");
    assert!(!report.arms.is_empty(), "the campaign ran no arms");
    assert!(
        report.runs_total > 0,
        "the campaign completed no runs, so the per-arm counters are untested"
    );
    for arm in &report.arms {
        if arm.slices == 0 {
            continue;
        }
        let counters = arm.counters.get("steer_authority").unwrap_or_else(|| {
            panic!("arm {} carries no steerAuthority counters at all", arm.id)
        });
        assert_same(
            &expected,
            &keys(counters, "an arm's steerAuthority counters"),
            &format!("arm {}'s counters", arm.id),
        );
        assert!(
            counters_are_integers(counters),
            "arm {}'s steerAuthority counters are not all integers: {counters}",
            arm.id
        );
    }

    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_stats_export_parity"));
}

fn counters_are_integers(v: &Value) -> bool {
    let Value::Object(map) = v else { return false };
    let map: &Map<String, Value> = map;
    map.values()
        .all(|x| x.as_i64().is_some() || x.as_u64().is_some())
}
