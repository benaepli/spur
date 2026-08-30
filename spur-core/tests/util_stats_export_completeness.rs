//! Export totality: every field of an exported counter block has to reach the
//! JSON a reader parses, carrying its own value.
//!
//! A block is filled with a distinct value per field, rendered through the same
//! function that writes the run artifact, parsed back, and compared leaf by
//! leaf. Distinct values make a field emitted under a neighbour's name fail the
//! same way a dropped field does.
//!
//! The expected leaves are read off the struct by destructuring it without a
//! rest pattern, so a field added to any block stops this file from compiling
//! until it is named here. That is the guard: an unexported counter is a build
//! failure rather than a zero nobody can interpret.

use serde_json::{Map, Value};
use spur_core::simulator::util_stats::{
    self, AcceptanceDistanceBucket, AcceptanceDistanceStats, CrashCensusStats, DeliveryEffect,
    DeliveryEffectStats, SteerAuthorityStats, TerminationStats, TerminationTally,
    UtilizationSnapshot,
};
use std::collections::BTreeMap;

/// A source of distinct nonzero values. Fractions are exactly representable so
/// the value that comes back out of the JSON compares equal to the one that
/// went in.
struct Marks(u64);

impl Marks {
    fn int(&mut self) -> u64 {
        self.0 += 1;
        self.0
    }

    fn frac(&mut self) -> f64 {
        self.0 += 1;
        self.0 as f64 + 0.5
    }
}

/// Distinct labels, so a bucket's label reaching the wrong bucket is caught
/// like any other misplaced field.
const LABELS: [&str; 3] = ["near", "mid", "far"];

fn leaf<T: Into<Value>>(prefix: &str, name: &str, v: T) -> (String, Value) {
    let path = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    };
    (path, v.into())
}

fn steer_authority(m: &mut Marks) -> SteerAuthorityStats {
    SteerAuthorityStats {
        steps_total: m.int(),
        steps: m.int(),
        audited: m.int(),
        preference_expressed: m.int(),
        preference_honored: m.int(),
        preference_consulted: m.int(),
        preference_source_absent: m.int(),
        honored: m.int(),
        no_eligible_candidates: m.int(),
        blocked_by_order: m.int(),
        blocked_by_timer_gate: m.int(),
        other_queue: m.int(),
        sampler_chose_other: m.int(),
    }
}

fn steer_authority_leaves(prefix: &str, b: &SteerAuthorityStats) -> Vec<(String, Value)> {
    let SteerAuthorityStats {
        steps_total,
        steps,
        audited,
        preference_expressed,
        preference_honored,
        preference_consulted,
        preference_source_absent,
        honored,
        no_eligible_candidates,
        blocked_by_order,
        blocked_by_timer_gate,
        other_queue,
        sampler_chose_other,
    } = b;
    vec![
        leaf(prefix, "steps_total", *steps_total),
        leaf(prefix, "steps", *steps),
        leaf(prefix, "audited", *audited),
        leaf(prefix, "preference_expressed", *preference_expressed),
        leaf(prefix, "preference_honored", *preference_honored),
        leaf(prefix, "preference_consulted", *preference_consulted),
        leaf(prefix, "preference_source_absent", *preference_source_absent),
        leaf(prefix, "honored", *honored),
        leaf(prefix, "no_eligible_candidates", *no_eligible_candidates),
        leaf(prefix, "blocked_by_order", *blocked_by_order),
        leaf(prefix, "blocked_by_timer_gate", *blocked_by_timer_gate),
        leaf(prefix, "other_queue", *other_queue),
        leaf(prefix, "sampler_chose_other", *sampler_chose_other),
    ]
}

fn termination_tally(m: &mut Marks) -> TerminationTally {
    TerminationTally {
        runs: m.int(),
        plan_complete: m.int(),
        plan_complete_with_pending_work: m.int(),
        iterations_exhausted: m.int(),
        deadlock: m.int(),
        steps_used_sum: m.int(),
        step_budget_sum: m.int(),
        pending_work_at_exit_sum: m.int(),
        planned_events_outstanding_sum: m.int(),
    }
}

fn termination_tally_leaves(prefix: &str, t: &TerminationTally) -> Vec<(String, Value)> {
    let TerminationTally {
        runs,
        plan_complete,
        plan_complete_with_pending_work,
        iterations_exhausted,
        deadlock,
        steps_used_sum,
        step_budget_sum,
        pending_work_at_exit_sum,
        planned_events_outstanding_sum,
    } = t;
    vec![
        leaf(prefix, "runs", *runs),
        leaf(prefix, "plan_complete", *plan_complete),
        leaf(
            prefix,
            "plan_complete_with_pending_work",
            *plan_complete_with_pending_work,
        ),
        leaf(prefix, "iterations_exhausted", *iterations_exhausted),
        leaf(prefix, "deadlock", *deadlock),
        leaf(prefix, "steps_used_sum", *steps_used_sum),
        leaf(prefix, "step_budget_sum", *step_budget_sum),
        leaf(prefix, "pending_work_at_exit_sum", *pending_work_at_exit_sum),
        leaf(
            prefix,
            "planned_events_outstanding_sum",
            *planned_events_outstanding_sum,
        ),
    ]
}

fn termination(m: &mut Marks) -> TerminationStats {
    TerminationStats {
        all: termination_tally(m),
        by_recovered_nodes: [
            termination_tally(m),
            termination_tally(m),
            termination_tally(m),
        ],
    }
}

fn termination_leaves(prefix: &str, t: &TerminationStats) -> Vec<(String, Value)> {
    let TerminationStats {
        all,
        by_recovered_nodes,
    } = t;
    let mut out = termination_tally_leaves(&format!("{prefix}.all"), all);
    for (i, tally) in by_recovered_nodes.iter().enumerate() {
        out.extend(termination_tally_leaves(
            &format!("{prefix}.by_recovered_nodes.{i}"),
            tally,
        ));
    }
    out
}

fn delivery_effect(m: &mut Marks) -> DeliveryEffect {
    DeliveryEffect {
        deliveries: m.int(),
        acted: m.int(),
        acted_fraction: m.frac(),
    }
}

fn delivery_effect_leaves(prefix: &str, d: &DeliveryEffect) -> Vec<(String, Value)> {
    let DeliveryEffect {
        deliveries,
        acted,
        acted_fraction,
    } = d;
    vec![
        leaf(prefix, "deliveries", *deliveries),
        leaf(prefix, "acted", *acted),
        leaf(prefix, "acted_fraction", *acted_fraction),
    ]
}

fn acceptance_bucket(m: &mut Marks, label: &'static str) -> AcceptanceDistanceBucket {
    AcceptanceDistanceBucket {
        distance: label,
        deliveries: m.int(),
        acted: m.int(),
        acted_fraction: m.frac(),
    }
}

fn acceptance_bucket_leaves(prefix: &str, b: &AcceptanceDistanceBucket) -> Vec<(String, Value)> {
    let AcceptanceDistanceBucket {
        distance,
        deliveries,
        acted,
        acted_fraction,
    } = b;
    vec![
        leaf(prefix, "distance", *distance),
        leaf(prefix, "deliveries", *deliveries),
        leaf(prefix, "acted", *acted),
        leaf(prefix, "acted_fraction", *acted_fraction),
    ]
}

fn acceptance_row(m: &mut Marks) -> Vec<AcceptanceDistanceBucket> {
    let mut row = Vec::new();
    for label in LABELS {
        row.push(acceptance_bucket(m, label));
    }
    row
}

fn acceptance_distance(m: &mut Marks) -> AcceptanceDistanceStats {
    AcceptanceDistanceStats {
        all: acceptance_row(m),
        sender_restarted: acceptance_row(m),
        receiver_restarted: acceptance_row(m),
    }
}

fn acceptance_distance_leaves(prefix: &str, a: &AcceptanceDistanceStats) -> Vec<(String, Value)> {
    let AcceptanceDistanceStats {
        all,
        sender_restarted,
        receiver_restarted,
    } = a;
    let mut out = Vec::new();
    for (name, row) in [
        ("all", all),
        ("sender_restarted", sender_restarted),
        ("receiver_restarted", receiver_restarted),
    ] {
        for (i, bucket) in row.iter().enumerate() {
            out.extend(acceptance_bucket_leaves(
                &format!("{prefix}.{name}.{i}"),
                bucket,
            ));
        }
    }
    out
}

fn delivery_effects(m: &mut Marks) -> DeliveryEffectStats {
    DeliveryEffectStats {
        all: delivery_effect(m),
        biased: delivery_effect(m),
        delayed: delivery_effect(m),
        sender_restarted: delivery_effect(m),
        receiver_restarted: delivery_effect(m),
        acceptance_distance: acceptance_distance(m),
        crashes_total: m.int(),
        crashes_with_own_sends_inflight: m.int(),
        recoveries_total: m.int(),
        recoveries_with_own_prior_sends_inflight: m.int(),
        stale_sender_deliveries_after_recovery: m.int(),
        crash_census: crash_census(m),
    }
}

fn crash_census(m: &mut Marks) -> CrashCensusStats {
    CrashCensusStats {
        decisions: m.int(),
        victim_had_inflight_sends: m.int(),
        any_candidate_had_inflight_sends: m.int(),
        inflight_bucket_0: m.int(),
        inflight_bucket_1: m.int(),
        inflight_bucket_2: m.int(),
        inflight_bucket_3plus: m.int(),
    }
}

fn crash_census_leaves(prefix: &str, c: &CrashCensusStats) -> Vec<(String, Value)> {
    let CrashCensusStats {
        decisions,
        victim_had_inflight_sends,
        any_candidate_had_inflight_sends,
        inflight_bucket_0,
        inflight_bucket_1,
        inflight_bucket_2,
        inflight_bucket_3plus,
    } = c;
    vec![
        leaf(prefix, "decisions", *decisions),
        leaf(
            prefix,
            "victim_had_inflight_sends",
            *victim_had_inflight_sends,
        ),
        leaf(
            prefix,
            "any_candidate_had_inflight_sends",
            *any_candidate_had_inflight_sends,
        ),
        leaf(prefix, "inflight_bucket_0", *inflight_bucket_0),
        leaf(prefix, "inflight_bucket_1", *inflight_bucket_1),
        leaf(prefix, "inflight_bucket_2", *inflight_bucket_2),
        leaf(prefix, "inflight_bucket_3plus", *inflight_bucket_3plus),
    ]
}

fn delivery_effects_leaves(prefix: &str, d: &DeliveryEffectStats) -> Vec<(String, Value)> {
    let DeliveryEffectStats {
        all,
        biased,
        delayed,
        sender_restarted,
        receiver_restarted,
        acceptance_distance,
        crashes_total,
        crashes_with_own_sends_inflight,
        recoveries_total,
        recoveries_with_own_prior_sends_inflight,
        stale_sender_deliveries_after_recovery,
        crash_census,
    } = d;
    let mut out = Vec::new();
    for (name, effect) in [
        ("all", all),
        ("biased", biased),
        ("delayed", delayed),
        ("sender_restarted", sender_restarted),
        ("receiver_restarted", receiver_restarted),
    ] {
        out.extend(delivery_effect_leaves(&format!("{prefix}.{name}"), effect));
    }
    out.extend(acceptance_distance_leaves(
        &format!("{prefix}.acceptance_distance"),
        acceptance_distance,
    ));
    out.extend([
        leaf(prefix, "crashes_total", *crashes_total),
        leaf(
            prefix,
            "crashes_with_own_sends_inflight",
            *crashes_with_own_sends_inflight,
        ),
        leaf(prefix, "recoveries_total", *recoveries_total),
        leaf(
            prefix,
            "recoveries_with_own_prior_sends_inflight",
            *recoveries_with_own_prior_sends_inflight,
        ),
        leaf(
            prefix,
            "stale_sender_deliveries_after_recovery",
            *stale_sender_deliveries_after_recovery,
        ),
    ]);
    out.extend(crash_census_leaves(
        &format!("{prefix}.crash_census"),
        crash_census,
    ));
    out
}

/// The blocks the snapshot is made of. Destructured without a rest pattern, so
/// a block added to the snapshot does not compile until it is named, which is
/// what keeps a whole block from going unexported.
fn block_names(s: &UtilizationSnapshot) -> Vec<&'static str> {
    let UtilizationSnapshot {
        rng_streams: _,
        steer: _,
        steer_empty_slice: _,
        steer_authority: _,
        steer_reach: _,
        multiplier_authority: _,
        purgatory: _,
        aos: _,
        dedup: _,
        feedback: _,
        curriculum: _,
        crash_recovery: _,
        recovery_window: _,
        ordered_h3: _,
        post_fault_ops: _,
        delivery_effects: _,
        timer_effects: _,
        timer_steer: _,
        crash_anchor: _,
        termination: _,
        prefix_extension: _,
        quiet_stretch: _,
        timeline_keys: _,
        steer_terms: _,
    } = s;
    vec![
        "rng_streams",
        "steer",
        "steer_empty_slice",
        "steer_authority",
        "steer_reach",
        "multiplier_authority",
        "purgatory",
        "aos",
        "dedup",
        "feedback",
        "curriculum",
        "crash_recovery",
        "recovery_window",
        "ordered_h3",
        "post_fault_ops",
        "delivery_effects",
        "timer_effects",
        "timer_steer",
        "crash_anchor",
        "termination",
        "prefix_extension",
        "quiet_stretch",
        "timeline_keys",
        "steer_terms",
    ]
}

/// Every scalar of `v` under its dotted path, array elements indexed by
/// position.
fn leaves(v: &Value, prefix: &str, out: &mut BTreeMap<String, Value>) {
    match v {
        Value::Object(m) => {
            for (k, sub) in m {
                leaves(sub, &join(prefix, k), out);
            }
        }
        Value::Array(a) => {
            for (i, sub) in a.iter().enumerate() {
                leaves(sub, &join(prefix, &i.to_string()), out);
            }
        }
        _ => {
            out.insert(prefix.to_string(), v.clone());
        }
    }
}

fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Fails naming what was dropped, what was invented, and what came back
/// holding another field's value, so the failure text alone says which.
fn assert_leaves(expected: &[(String, Value)], actual: &BTreeMap<String, Value>, what: &str) {
    let mut missing = Vec::new();
    let mut wrong = Vec::new();
    for (path, want) in expected {
        match actual.get(path) {
            None => missing.push(path.clone()),
            Some(got) if got != want => wrong.push(format!("{path}: wanted {want}, got {got}")),
            Some(_) => {}
        }
    }
    let known: std::collections::BTreeSet<&String> = expected.iter().map(|(p, _)| p).collect();
    let extra: Vec<&String> = actual.keys().filter(|p| !known.contains(p)).collect();
    assert!(
        missing.is_empty() && wrong.is_empty() && extra.is_empty(),
        "{what} is not total: missing {missing:?}, extra {extra:?}, mismatched {wrong:?}"
    );
}

/// A snapshot whose three most-read blocks carry a distinct value in every
/// field. The rest of the snapshot is whatever the process counters hold,
/// which for a test binary that has recorded nothing is zero.
fn marked_snapshot() -> (UtilizationSnapshot, Vec<(String, Value)>) {
    let mut m = Marks(0);
    let mut s = util_stats::snapshot();
    s.steer_authority = steer_authority(&mut m);
    s.termination = termination(&mut m);
    s.delivery_effects = delivery_effects(&mut m);
    let mut expected = steer_authority_leaves("steer_authority", &s.steer_authority);
    expected.extend(termination_leaves("termination", &s.termination));
    expected.extend(delivery_effects_leaves(
        "delivery_effects",
        &s.delivery_effects,
    ));
    (s, expected)
}

#[test]
fn every_counter_field_reaches_the_written_json() {
    let (snapshot, expected) = marked_snapshot();
    let rendered = util_stats::render_snapshot(&snapshot).expect("the snapshot serializes");
    let parsed: Value = serde_json::from_str(&rendered).expect("the written JSON parses");

    let blocks: Vec<String> = parsed
        .as_object()
        .expect("the snapshot is a JSON object")
        .keys()
        .cloned()
        .collect();
    let declared: Vec<String> = block_names(&snapshot)
        .into_iter()
        .map(str::to_string)
        .collect();
    let mut sorted_blocks = blocks.clone();
    sorted_blocks.sort();
    let mut sorted_declared = declared.clone();
    sorted_declared.sort();
    assert_eq!(
        sorted_declared, sorted_blocks,
        "the written JSON does not carry the snapshot's blocks"
    );

    for block in ["steer_authority", "termination", "delivery_effects"] {
        let mut actual = BTreeMap::new();
        leaves(&parsed[block], block, &mut actual);
        let want: Vec<(String, Value)> = expected
            .iter()
            .filter(|(p, _)| p.starts_with(&format!("{block}.")))
            .cloned()
            .collect();
        assert!(!want.is_empty(), "no field was marked for {block}");
        assert_leaves(&want, &actual, &format!("the written {block} block"));
    }
}

/// The second export: a campaign attributes counters to an arm by differencing
/// two snapshots and accumulating the result. That path keeps integer leaves
/// only - floats are ratios a reader recomputes, and arrays are curves - so it
/// is checked against the integer fields that are not inside an array.
#[test]
fn every_integer_counter_survives_the_difference_and_accumulate_path() {
    let zero = util_stats::snapshot_value();
    let (snapshot, expected) = marked_snapshot();
    let marked = serde_json::to_value(&snapshot).expect("the snapshot serializes");

    let mut acc = Value::Object(Map::new());
    util_stats::add(&mut acc, &util_stats::delta(&zero, &marked));

    let mut actual = BTreeMap::new();
    leaves(&acc, "", &mut actual);

    let want: Vec<(String, Value)> = expected
        .iter()
        .filter(|(path, value)| {
            value.is_u64() && !path.split('.').any(|s| s.parse::<usize>().is_ok())
        })
        .cloned()
        .collect();
    assert!(!want.is_empty(), "no integer field was marked");
    let mut missing = Vec::new();
    let mut wrong = Vec::new();
    for (path, value) in &want {
        match actual.get(path) {
            None => missing.push(path.clone()),
            Some(got) if got.as_u64() != value.as_u64() => {
                wrong.push(format!("{path}: wanted {value}, got {got}"))
            }
            Some(_) => {}
        }
    }
    assert!(
        missing.is_empty() && wrong.is_empty(),
        "the accumulated arm counters are not total: missing {missing:?}, mismatched {wrong:?}"
    );
}
