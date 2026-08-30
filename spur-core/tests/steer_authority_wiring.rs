//! The steer-authority counters have to reach the scheduler. A session that
//! consumed scheduling steps must report scheduling steps, whatever the scoring
//! weights are, and the outcome buckets must partition the steps that were
//! actually resolved rather than all of them.

use spur_core::compiler;
use spur_core::simulator::config_override;
use spur_core::simulator::explorer::run_explorer;
use spur_core::simulator::history::LogBackend;
use spur_core::simulator::util_stats::{
    self, EmptySliceStats, PrefixExtensionStats, SteerAuthorityStats,
};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SPEC: &str = include_str!("fixtures/relay.spur");

const CONFIG: &str = r#"{
  "num_servers": {"min": 2, "max": 2, "step": 1},
  "num_write_ops": {"min": 2, "max": 2, "step": 1},
  "num_read_ops": {"min": 1, "max": 1, "step": 1},
  "num_keys": {"min": 1, "max": 1, "step": 1},
  "num_crashes": {"min": 1, "max": 2, "step": 1},
  "dependency_density": [0.0],
  "num_runs_per_config": 20,
  "max_iterations": 600,
  "session_seed": 909,
  "rng_stream_isolation": true,
  "strict_config_keys": true,
  "stats": true,
  "emit_prefix_extension": true,
  "feedback": {"mode": "timeline", "steer": true, "steer_audit": true{{ALWAYS}}}
}"#;

const SESSION_STACK_BYTES: usize = 64 * 1024 * 1024;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("spur_steer_authority_wiring").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("creates scratch directory");
    dir
}

/// Runs one session with no predicate carrying weight and returns the
/// authority counters, the skip counters, the steps the runs consumed, and the
/// per-step census that says how many of those steps had nothing queued.
fn session(
    name: &str,
    always: bool,
) -> (SteerAuthorityStats, EmptySliceStats, u64, PrefixExtensionStats) {
    let program = compiler::compile(SPEC, "relay.spur")
        .into_program()
        .expect("spec compiles");
    let config_path = std::env::temp_dir().join(format!("spur_steer_authority_{name}.json"));
    let always = if always {
        r#", "steer_audit_always": true"#
    } else {
        ""
    };
    fs::write(&config_path, CONFIG.replace("{{ALWAYS}}", always)).expect("writes config");
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
    let snapshot = util_stats::snapshot();
    let _ = fs::remove_file(&config_path);
    let _ = fs::remove_dir_all(&out);
    (
        snapshot.steer_authority,
        snapshot.steer_empty_slice,
        snapshot.termination.all.steps_used_sum,
        snapshot.prefix_extension,
    )
}

#[test]
fn a_session_that_used_steps_reports_them() {
    std::thread::Builder::new()
        .stack_size(SESSION_STACK_BYTES)
        .spawn(check)
        .expect("spawns the session thread")
        .join()
        .expect("the session thread runs to completion");
}

fn check() {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let (skipped, skips, steps_used, skipped_census) = session("skipped", false);
    assert!(steps_used > 0, "the session consumed no scheduling steps");
    assert!(
        skipped.steps > 0,
        "steps used but none counted: {steps_used} used, counters {:?} {:?}",
        (skipped.steps, skipped.audited),
        skips
    );
    check_step_provenance(&skipped, &skipped_census);
    assert_eq!(
        (skipped.audited, skipped.honored),
        (0, 0),
        "an unweighted session resolved a preference without being asked to"
    );
    assert!(
        skips.queue_audit_skipped > 0,
        "the skipped audit was not counted: {skips:?}"
    );
    check_consultation(&skipped);
    assert_eq!(
        skipped.preference_source_absent, skipped.preference_consulted,
        "a session with no weighted predicate read a preference source anyway"
    );

    let (always, always_skips, always_steps_used, always_census) = session("always", true);
    assert!(always_steps_used > 0);
    assert!(always.steps > 0);
    check_step_provenance(&always, &always_census);
    assert!(
        always.audited > 0,
        "the audit was asked for and never ran: {}",
        always.steps
    );
    assert_eq!(
        always_skips.queue_audit_skipped, 0,
        "the audit was asked for and still skipped: {always_skips:?}"
    );
    assert!(
        always.audited <= always.steps,
        "more steps were audited than were reached"
    );
    assert_eq!(
        always.honored
            + always.no_eligible_candidates
            + always.blocked_by_order
            + always.blocked_by_timer_gate
            + always.other_queue
            + always.sampler_chose_other,
        always.audited,
        "the outcome buckets do not partition the audited steps"
    );
    assert!(
        always.honored > 0,
        "no step ever ran the top-ranked runnable"
    );
    check_consultation(&always);

    let _ = fs::remove_dir_all(std::env::temp_dir().join("spur_steer_authority_wiring"));
}

/// `steps` counts the budget steps that reached the point where the run's
/// preference is read; `steps_total` counts every budget step. The two are
/// separate counts, so the first can never exceed the second, and a session
/// with steps that had nothing queued must show a strict gap - equality there
/// would mean `steps` is following the budget rather than the decision site.
fn check_step_provenance(s: &SteerAuthorityStats, census: &PrefixExtensionStats) {
    assert!(
        s.steps <= s.steps_total,
        "more steps reached the decision site than were taken: {} of {}",
        s.steps,
        s.steps_total
    );
    let idle = census.all.steps_idle_sum;
    assert!(
        idle == 0 || s.steps < s.steps_total,
        "{idle} steps had nothing to schedule and yet every one of the {} steps \
         taken reached the decision site",
        s.steps_total
    );
    assert!(
        s.audited <= s.steps,
        "more steps were audited than reached the decision site: {} of {}",
        s.audited,
        s.steps
    );
}

/// A session that reached scheduling points must report that the decision
/// sites read a preference source, so a zero elsewhere in the block can be
/// read as "nothing to prefer" rather than "the site never ran".
fn check_consultation(s: &SteerAuthorityStats) {
    assert!(
        s.steps == 0 || s.preference_consulted > 0,
        "steps were reached and no preference source was ever consulted: {} steps",
        s.steps
    );
    assert!(
        s.preference_source_absent <= s.preference_consulted,
        "more consultations found no source than were counted at all"
    );
    assert!(
        s.preference_expressed <= s.preference_consulted,
        "a preference was expressed more often than one was consulted"
    );
}
