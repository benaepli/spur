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
    self, AcceptanceDistanceBucket, AcceptanceDistanceStats, ArmSelectorAxisStats, ArmSelectorExploreStats,
    ArmSelectorLearnerStats, ArmSelectorPooledStats, ArmSelectorRewardStats, ClientAnchorArmRuns,
    ClientAnchorAxisStats, ClientAnchorCensusStats, ClientAnchorDistance,
    ClientAnchorFirstDelivery, ClientAnchorFirstEntry, ClientAnchorHalfStats,
    ClientAnchorHeldHist,
    ClientAnchorReleaseStats, ClientAnchorRushStats, ClientAnchorStats,
    CrashCensusStats, CrashPhaseArmStats, CrashPhaseLandingStats, CrashPhaseMovedStats,
    CrashPhaseStats, CrashPlaceStats, DeliveryEffect, DeliveryEffectStats, FreshFirstCensusStats,
    CallTargetStats, ChannelTableStats, GridPoolStats, CompiledExprStats, CompiledOpsStats, EvalBorrowStats, FrameLayoutStats, FrameStats, HistoryFormatStats, PrintContentStats, ProgramTextStats, RunBufferStats, ValueSigStats, ValueStructStats, HistoryWriterStats, RunSetupStats, StatsLocalStats, TextBufferStats, TimelineStats, TraceFormatStats, FreshFirstHalfStats, FreshFirstStats, GhostSignalStats, PairOrderCensusStats, PairOrderClassCounts, PairOrderHalfStats, PairOrderStats, PlanDeliverStats, PlanDepsDensitySplit, PlanDepsStats, PlanDepsTally, PlanReadyStats, ReplayStats, SchedStats, GhostReleaseCellStats, GhostReleaseCellsStats, GhostReleaseSingleCellStats, GhostReleaseSingleCellsStats, GhostReleaseSingleStats, GhostReleaseStats, RunCapStats, StallCapMarks, StallCapStats, StallReleaseCellStats, StallReleaseDependents, StallReleaseStats, SteerAuthorityStats, TerminationStats, TerminationTally,
    TimerContextStats, UtilizationSnapshot, VictimSwapCensusStats, VictimSwapHalfStats,
    VictimSwapStats,
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
        learned_cap_reached: m.int(),
        stall_cap_reached: m.int(),
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
        learned_cap_reached,
        stall_cap_reached,
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
        leaf(prefix, "learned_cap_reached", *learned_cap_reached),
        leaf(prefix, "stall_cap_reached", *stall_cap_reached),
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

fn plan_deps_tally(m: &mut Marks) -> PlanDepsTally {
    PlanDepsTally {
        runs: m.int(),
        crashes: m.int(),
        unrecovered_crashes: m.int(),
        zero_recovery_runs: m.int(),
        plan_complete: m.int(),
        steps_used_sum: m.int(),
    }
}

fn plan_deps_tally_leaves(prefix: &str, t: &PlanDepsTally) -> Vec<(String, Value)> {
    let PlanDepsTally {
        runs,
        crashes,
        unrecovered_crashes,
        zero_recovery_runs,
        plan_complete,
        steps_used_sum,
    } = t;
    vec![
        leaf(prefix, "runs", *runs),
        leaf(prefix, "crashes", *crashes),
        leaf(prefix, "unrecovered_crashes", *unrecovered_crashes),
        leaf(prefix, "zero_recovery_runs", *zero_recovery_runs),
        leaf(prefix, "plan_complete", *plan_complete),
        leaf(prefix, "steps_used_sum", *steps_used_sum),
    ]
}

fn plan_deps_split(m: &mut Marks) -> PlanDepsDensitySplit {
    PlanDepsDensitySplit {
        density_zero: plan_deps_tally(m),
        density_positive: plan_deps_tally(m),
    }
}

fn plan_deps_split_leaves(prefix: &str, s: &PlanDepsDensitySplit) -> Vec<(String, Value)> {
    let PlanDepsDensitySplit {
        density_zero,
        density_positive,
    } = s;
    let mut out = plan_deps_tally_leaves(&format!("{prefix}.density_zero"), density_zero);
    out.extend(plan_deps_tally_leaves(
        &format!("{prefix}.density_positive"),
        density_positive,
    ));
    out
}

fn plan_deps(m: &mut Marks) -> PlanDepsStats {
    PlanDepsStats {
        recover_edges_dropped: m.int(),
        stock: plan_deps_split(m),
        exempt: plan_deps_split(m),
    }
}

fn plan_deps_leaves(prefix: &str, p: &PlanDepsStats) -> Vec<(String, Value)> {
    let PlanDepsStats {
        recover_edges_dropped,
        stock,
        exempt,
    } = p;
    let mut out = vec![leaf(prefix, "recover_edges_dropped", *recover_edges_dropped)];
    out.extend(plan_deps_split_leaves(&format!("{prefix}.stock"), stock));
    out.extend(plan_deps_split_leaves(&format!("{prefix}.exempt"), exempt));
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

fn run_cap(m: &mut Marks) -> RunCapStats {
    RunCapStats {
        probes: m.int(),
        probe_completions: m.int(),
        over_cap_completions: m.int(),
        cap_recomputes: m.int(),
        scopes_learned: m.int(),
        current_cap_max_scope: m.int(),
    }
}

fn run_cap_leaves(prefix: &str, r: &RunCapStats) -> Vec<(String, Value)> {
    let RunCapStats {
        probes,
        probe_completions,
        over_cap_completions,
        cap_recomputes,
        scopes_learned,
        current_cap_max_scope,
    } = r;
    vec![
        leaf(prefix, "probes", *probes),
        leaf(prefix, "probe_completions", *probe_completions),
        leaf(prefix, "over_cap_completions", *over_cap_completions),
        leaf(prefix, "cap_recomputes", *cap_recomputes),
        leaf(prefix, "scopes_learned", *scopes_learned),
        leaf(prefix, "current_cap_max_scope", *current_cap_max_scope),
    ]
}

fn stall_cap(m: &mut Marks) -> StallCapStats {
    StallCapStats {
        stops: m.int(),
        treated_runs: m.int(),
        untreated_runs: m.int(),
        steps_saved_sum: m.int(),
        suspended_steps_sum: m.int(),
        marks: StallCapMarks {
            rows: m.int(),
            acted_deliveries: m.int(),
            acted_timers: m.int(),
            releases: m.int(),
        },
        probes_keyed: m.int(),
        probe_over_cap_completions: m.int(),
        scopes_learned: m.int(),
        cap_max_scope: m.int(),
        untreated_runs_capped: m.int(),
        untreated_over_cap_runs: m.int(),
        untreated_rows_dropped: m.int(),
        untreated_gap_hist: vec![m.int(), m.int(), m.int()],
    }
}

fn stall_cap_leaves(prefix: &str, s: &StallCapStats) -> Vec<(String, Value)> {
    let StallCapStats {
        stops,
        treated_runs,
        untreated_runs,
        steps_saved_sum,
        suspended_steps_sum,
        marks,
        probes_keyed,
        probe_over_cap_completions,
        scopes_learned,
        cap_max_scope,
        untreated_runs_capped,
        untreated_over_cap_runs,
        untreated_rows_dropped,
        untreated_gap_hist,
    } = s;
    let StallCapMarks {
        rows,
        acted_deliveries,
        acted_timers,
        releases,
    } = marks;
    let mut out = vec![
        leaf(prefix, "stops", *stops),
        leaf(prefix, "treated_runs", *treated_runs),
        leaf(prefix, "untreated_runs", *untreated_runs),
        leaf(prefix, "steps_saved_sum", *steps_saved_sum),
        leaf(prefix, "suspended_steps_sum", *suspended_steps_sum),
        leaf(prefix, "marks.rows", *rows),
        leaf(prefix, "marks.acted_deliveries", *acted_deliveries),
        leaf(prefix, "marks.acted_timers", *acted_timers),
        leaf(prefix, "marks.releases", *releases),
        leaf(prefix, "probes_keyed", *probes_keyed),
        leaf(prefix, "probe_over_cap_completions", *probe_over_cap_completions),
        leaf(prefix, "scopes_learned", *scopes_learned),
        leaf(prefix, "cap_max_scope", *cap_max_scope),
        leaf(prefix, "untreated_runs_capped", *untreated_runs_capped),
        leaf(prefix, "untreated_over_cap_runs", *untreated_over_cap_runs),
        leaf(prefix, "untreated_rows_dropped", *untreated_rows_dropped),
    ];
    for (i, v) in untreated_gap_hist.iter().enumerate() {
        out.push(leaf(prefix, &format!("untreated_gap_hist.{i}"), *v));
    }
    out
}

fn stall_release_cell(m: &mut Marks) -> StallReleaseCellStats {
    StallReleaseCellStats {
        runs: m.int(),
        invocations: m.int(),
        plan_complete: m.int(),
    }
}

fn stall_release_cell_leaves(prefix: &str, c: &StallReleaseCellStats) -> Vec<(String, Value)> {
    let StallReleaseCellStats {
        runs,
        invocations,
        plan_complete,
    } = c;
    vec![
        leaf(prefix, "runs", *runs),
        leaf(prefix, "invocations", *invocations),
        leaf(prefix, "plan_complete", *plan_complete),
    ]
}

fn stall_release(m: &mut Marks) -> StallReleaseStats {
    StallReleaseStats {
        releases: m.int(),
        ops_settled: m.int(),
        dependents_released: StallReleaseDependents {
            client: m.int(),
            fault: m.int(),
            other: m.int(),
        },
        late_responses: m.int(),
        plan_completed_after_release: m.int(),
        second_stall_stops: m.int(),
        steps_after_release_sum: m.int(),
        stalls_without_ops: m.int(),
        release_cell: stall_release_cell(m),
        cut_cell: stall_release_cell(m),
    }
}

fn stall_release_leaves(prefix: &str, s: &StallReleaseStats) -> Vec<(String, Value)> {
    let StallReleaseStats {
        releases,
        ops_settled,
        dependents_released,
        late_responses,
        plan_completed_after_release,
        second_stall_stops,
        steps_after_release_sum,
        stalls_without_ops,
        release_cell,
        cut_cell,
    } = s;
    let StallReleaseDependents {
        client,
        fault,
        other,
    } = dependents_released;
    let mut out = vec![
        leaf(prefix, "releases", *releases),
        leaf(prefix, "ops_settled", *ops_settled),
        leaf(prefix, "dependents_released.client", *client),
        leaf(prefix, "dependents_released.fault", *fault),
        leaf(prefix, "dependents_released.other", *other),
        leaf(prefix, "late_responses", *late_responses),
        leaf(prefix, "plan_completed_after_release", *plan_completed_after_release),
        leaf(prefix, "second_stall_stops", *second_stall_stops),
        leaf(prefix, "steps_after_release_sum", *steps_after_release_sum),
        leaf(prefix, "stalls_without_ops", *stalls_without_ops),
    ];
    out.extend(stall_release_cell_leaves(&format!("{prefix}.release_cell"), release_cell));
    out.extend(stall_release_cell_leaves(&format!("{prefix}.cut_cell"), cut_cell));
    out
}

fn ghost_release_cell(m: &mut Marks) -> GhostReleaseCellStats {
    GhostReleaseCellStats {
        runs: m.int(),
        steps_used_sum: m.int(),
        crashes_applied: m.int(),
        later_crashes_applied: m.int(),
        applied_within_3_of_acted_ghost: m.int(),
        fired_crashes_applied: m.int(),
        fired_crashes_applied_within_3: m.int(),
        fired_crashes_on_ghost_node: m.int(),
        fired_crashes_applied_anchored: m.int(),
        fired_crashes_applied_within_3_anchored: m.int(),
        fired_crashes_applied_unanchored: m.int(),
        fired_crashes_applied_within_3_unanchored: m.int(),
        fired_crashes_applied_retarget: m.int(),
        fired_crashes_on_ghost_node_retarget: m.int(),
        fired_crashes_applied_stock: m.int(),
        fired_crashes_on_ghost_node_stock: m.int(),
        fired_inflight_bucket_0: m.int(),
        fired_inflight_bucket_1: m.int(),
        fired_inflight_bucket_2: m.int(),
        fired_inflight_bucket_3plus: m.int(),
        double_crash_after_release: m.int(),
    }
}

fn ghost_release_cell_leaves(prefix: &str, c: &GhostReleaseCellStats) -> Vec<(String, Value)> {
    let GhostReleaseCellStats {
        runs,
        steps_used_sum,
        crashes_applied,
        later_crashes_applied,
        applied_within_3_of_acted_ghost,
        fired_crashes_applied,
        fired_crashes_applied_within_3,
        fired_crashes_on_ghost_node,
        fired_crashes_applied_anchored,
        fired_crashes_applied_within_3_anchored,
        fired_crashes_applied_unanchored,
        fired_crashes_applied_within_3_unanchored,
        fired_crashes_applied_retarget,
        fired_crashes_on_ghost_node_retarget,
        fired_crashes_applied_stock,
        fired_crashes_on_ghost_node_stock,
        fired_inflight_bucket_0,
        fired_inflight_bucket_1,
        fired_inflight_bucket_2,
        fired_inflight_bucket_3plus,
        double_crash_after_release,
    } = c;
    vec![
        leaf(prefix, "runs", *runs),
        leaf(prefix, "steps_used_sum", *steps_used_sum),
        leaf(prefix, "crashes_applied", *crashes_applied),
        leaf(prefix, "later_crashes_applied", *later_crashes_applied),
        leaf(prefix, "applied_within_3_of_acted_ghost", *applied_within_3_of_acted_ghost),
        leaf(prefix, "fired_crashes_applied", *fired_crashes_applied),
        leaf(prefix, "fired_crashes_applied_within_3", *fired_crashes_applied_within_3),
        leaf(prefix, "fired_crashes_on_ghost_node", *fired_crashes_on_ghost_node),
        leaf(prefix, "fired_crashes_applied_anchored", *fired_crashes_applied_anchored),
        leaf(
            prefix,
            "fired_crashes_applied_within_3_anchored",
            *fired_crashes_applied_within_3_anchored,
        ),
        leaf(prefix, "fired_crashes_applied_unanchored", *fired_crashes_applied_unanchored),
        leaf(
            prefix,
            "fired_crashes_applied_within_3_unanchored",
            *fired_crashes_applied_within_3_unanchored,
        ),
        leaf(prefix, "fired_crashes_applied_retarget", *fired_crashes_applied_retarget),
        leaf(
            prefix,
            "fired_crashes_on_ghost_node_retarget",
            *fired_crashes_on_ghost_node_retarget,
        ),
        leaf(prefix, "fired_crashes_applied_stock", *fired_crashes_applied_stock),
        leaf(prefix, "fired_crashes_on_ghost_node_stock", *fired_crashes_on_ghost_node_stock),
        leaf(prefix, "fired_inflight_bucket_0", *fired_inflight_bucket_0),
        leaf(prefix, "fired_inflight_bucket_1", *fired_inflight_bucket_1),
        leaf(prefix, "fired_inflight_bucket_2", *fired_inflight_bucket_2),
        leaf(prefix, "fired_inflight_bucket_3plus", *fired_inflight_bucket_3plus),
        leaf(prefix, "double_crash_after_release", *double_crash_after_release),
    ]
}

fn ghost_release_single_cell(m: &mut Marks) -> GhostReleaseSingleCellStats {
    GhostReleaseSingleCellStats {
        runs: m.int(),
        steps_used_sum: m.int(),
        crashes_applied: m.int(),
        fired_crashes_applied: m.int(),
        fired_crashes_on_ghost_node: m.int(),
        double_crash_after_release: m.int(),
    }
}

fn ghost_release_single_cell_leaves(
    prefix: &str,
    c: &GhostReleaseSingleCellStats,
) -> Vec<(String, Value)> {
    let GhostReleaseSingleCellStats {
        runs,
        steps_used_sum,
        crashes_applied,
        fired_crashes_applied,
        fired_crashes_on_ghost_node,
        double_crash_after_release,
    } = c;
    vec![
        leaf(prefix, "runs", *runs),
        leaf(prefix, "steps_used_sum", *steps_used_sum),
        leaf(prefix, "crashes_applied", *crashes_applied),
        leaf(prefix, "fired_crashes_applied", *fired_crashes_applied),
        leaf(prefix, "fired_crashes_on_ghost_node", *fired_crashes_on_ghost_node),
        leaf(prefix, "double_crash_after_release", *double_crash_after_release),
    ]
}

fn ghost_release(m: &mut Marks) -> GhostReleaseStats {
    GhostReleaseStats {
        armed: m.int(),
        fired: m.int(),
        fired_nothing_held: m.int(),
        expired: m.int(),
        superseded: m.int(),
        steps_from_restart_sum: m.int(),
        released_crashes: m.int(),
        restarts_with_held_crash: m.int(),
        lag_samples: m.int(),
        lag_p50: m.int(),
        lag_p75: m.int(),
        lag_p90: m.int(),
        scopes_engaged: m.int(),
        cells: GhostReleaseCellsStats {
            untreated: ghost_release_cell(m),
            release_all: ghost_release_cell(m),
            single: ghost_release_cell(m),
            treated: ghost_release_cell(m),
        },
        single: GhostReleaseSingleStats {
            releases: m.int(),
            released_own_crash: m.int(),
            released_via_ranking: m.int(),
            released_forced: m.int(),
            no_release_case: m.int(),
            cells: GhostReleaseSingleCellsStats {
                release_all: ghost_release_single_cell(m),
                single: ghost_release_single_cell(m),
            },
        },
    }
}

fn ghost_release_leaves(prefix: &str, r: &GhostReleaseStats) -> Vec<(String, Value)> {
    let GhostReleaseStats {
        armed,
        fired,
        fired_nothing_held,
        expired,
        superseded,
        steps_from_restart_sum,
        released_crashes,
        restarts_with_held_crash,
        lag_samples,
        lag_p50,
        lag_p75,
        lag_p90,
        scopes_engaged,
        cells,
        single,
    } = r;
    let GhostReleaseCellsStats {
        untreated,
        release_all,
        single: single_cell,
        treated,
    } = cells;
    let GhostReleaseSingleStats {
        releases,
        released_own_crash,
        released_via_ranking,
        released_forced,
        no_release_case,
        cells: single_cells,
    } = single;
    let GhostReleaseSingleCellsStats {
        release_all: single_release_all,
        single: single_single,
    } = single_cells;
    let mut out = vec![
        leaf(prefix, "armed", *armed),
        leaf(prefix, "fired", *fired),
        leaf(prefix, "fired_nothing_held", *fired_nothing_held),
        leaf(prefix, "expired", *expired),
        leaf(prefix, "superseded", *superseded),
        leaf(prefix, "steps_from_restart_sum", *steps_from_restart_sum),
        leaf(prefix, "released_crashes", *released_crashes),
        leaf(prefix, "restarts_with_held_crash", *restarts_with_held_crash),
        leaf(prefix, "lag_samples", *lag_samples),
        leaf(prefix, "lag_p50", *lag_p50),
        leaf(prefix, "lag_p75", *lag_p75),
        leaf(prefix, "lag_p90", *lag_p90),
        leaf(prefix, "scopes_engaged", *scopes_engaged),
    ];
    for (name, cell) in [
        ("untreated", untreated),
        ("release_all", release_all),
        ("single", single_cell),
        ("treated", treated),
    ] {
        out.extend(ghost_release_cell_leaves(&format!("{prefix}.cells.{name}"), cell));
    }
    let s = format!("{prefix}.single");
    out.extend([
        leaf(&s, "releases", *releases),
        leaf(&s, "released_own_crash", *released_own_crash),
        leaf(&s, "released_via_ranking", *released_via_ranking),
        leaf(&s, "released_forced", *released_forced),
        leaf(&s, "no_release_case", *no_release_case),
    ]);
    for (name, cell) in [
        ("release_all", single_release_all),
        ("single", single_single),
    ] {
        out.extend(ghost_release_single_cell_leaves(&format!("{s}.cells.{name}"), cell));
    }
    out
}

fn crash_place(m: &mut Marks) -> CrashPlaceStats {
    CrashPlaceStats {
        draws: m.int(),
        capped_draws: m.int(),
        holds: m.int(),
        held_steps_sum: m.int(),
        ghost_release: ghost_release(m),
    }
}

fn crash_place_leaves(prefix: &str, c: &CrashPlaceStats) -> Vec<(String, Value)> {
    let CrashPlaceStats {
        draws,
        capped_draws,
        holds,
        held_steps_sum,
        ghost_release,
    } = c;
    let mut out = vec![
        leaf(prefix, "draws", *draws),
        leaf(prefix, "capped_draws", *capped_draws),
        leaf(prefix, "holds", *holds),
        leaf(prefix, "held_steps_sum", *held_steps_sum),
    ];
    out.extend(ghost_release_leaves(&format!("{prefix}.ghost_release"), ghost_release));
    out
}

fn crash_phase_moved(m: &mut Marks) -> CrashPhaseMovedStats {
    CrashPhaseMovedStats {
        crashes_applied: m.int(),
        apply_decisions: m.int(),
        apply_victim_had_inflight: m.int(),
        inflight_bucket_0: m.int(),
        inflight_bucket_1: m.int(),
        inflight_bucket_2: m.int(),
        inflight_bucket_3plus: m.int(),
    }
}

fn crash_phase_moved_leaves(prefix: &str, a: &CrashPhaseMovedStats) -> Vec<(String, Value)> {
    let CrashPhaseMovedStats {
        crashes_applied,
        apply_decisions,
        apply_victim_had_inflight,
        inflight_bucket_0,
        inflight_bucket_1,
        inflight_bucket_2,
        inflight_bucket_3plus,
    } = a;
    vec![
        leaf(prefix, "crashes_applied", *crashes_applied),
        leaf(prefix, "apply_decisions", *apply_decisions),
        leaf(
            prefix,
            "apply_victim_had_inflight",
            *apply_victim_had_inflight,
        ),
        leaf(prefix, "inflight_bucket_0", *inflight_bucket_0),
        leaf(prefix, "inflight_bucket_1", *inflight_bucket_1),
        leaf(prefix, "inflight_bucket_2", *inflight_bucket_2),
        leaf(prefix, "inflight_bucket_3plus", *inflight_bucket_3plus),
    ]
}

fn crash_phase_arm(m: &mut Marks) -> CrashPhaseArmStats {
    CrashPhaseArmStats {
        runs: m.int(),
        armed: m.int(),
        released_on_condition: m.int(),
        expired: m.int(),
        wait_steps_sum: m.int(),
        release_decisions: m.int(),
        release_victim_had_inflight: m.int(),
        expired_victim_had_inflight: m.int(),
        apply_decisions: m.int(),
        apply_victim_had_inflight: m.int(),
        crashes_applied: m.int(),
        inflight_bucket_0: m.int(),
        inflight_bucket_1: m.int(),
        inflight_bucket_2: m.int(),
        inflight_bucket_3plus: m.int(),
        moved_read_on_landing: crash_phase_moved(m),
    }
}

fn crash_phase_arm_leaves(prefix: &str, a: &CrashPhaseArmStats) -> Vec<(String, Value)> {
    let CrashPhaseArmStats {
        runs,
        armed,
        released_on_condition,
        expired,
        wait_steps_sum,
        release_decisions,
        release_victim_had_inflight,
        expired_victim_had_inflight,
        apply_decisions,
        apply_victim_had_inflight,
        crashes_applied,
        inflight_bucket_0,
        inflight_bucket_1,
        inflight_bucket_2,
        inflight_bucket_3plus,
        moved_read_on_landing,
    } = a;
    let mut out = vec![
        leaf(prefix, "runs", *runs),
        leaf(prefix, "armed", *armed),
        leaf(prefix, "released_on_condition", *released_on_condition),
        leaf(prefix, "expired", *expired),
        leaf(prefix, "wait_steps_sum", *wait_steps_sum),
        leaf(prefix, "release_decisions", *release_decisions),
        leaf(
            prefix,
            "release_victim_had_inflight",
            *release_victim_had_inflight,
        ),
        leaf(
            prefix,
            "expired_victim_had_inflight",
            *expired_victim_had_inflight,
        ),
        leaf(prefix, "apply_decisions", *apply_decisions),
        leaf(
            prefix,
            "apply_victim_had_inflight",
            *apply_victim_had_inflight,
        ),
        leaf(prefix, "crashes_applied", *crashes_applied),
        leaf(prefix, "inflight_bucket_0", *inflight_bucket_0),
        leaf(prefix, "inflight_bucket_1", *inflight_bucket_1),
        leaf(prefix, "inflight_bucket_2", *inflight_bucket_2),
        leaf(prefix, "inflight_bucket_3plus", *inflight_bucket_3plus),
    ];
    out.extend(crash_phase_moved_leaves(
        &format!("{prefix}.moved_read_on_landing"),
        moved_read_on_landing,
    ));
    out
}

fn crash_phase(m: &mut Marks) -> CrashPhaseStats {
    CrashPhaseStats {
        armed: m.int(),
        stock_releases: m.int(),
        early: crash_phase_arm(m),
        mid: crash_phase_arm(m),
        stock: crash_phase_arm(m),
        landing: CrashPhaseLandingStats {
            evaluated_on_other_node: m.int(),
            condition_on_other_node: m.int(),
            expired_on_other_node: m.int(),
            mismatch_at_apply: m.int(),
        },
    }
}

fn crash_phase_leaves(prefix: &str, c: &CrashPhaseStats) -> Vec<(String, Value)> {
    let CrashPhaseStats {
        armed,
        stock_releases,
        early,
        mid,
        stock,
        landing,
    } = c;
    let CrashPhaseLandingStats {
        evaluated_on_other_node,
        condition_on_other_node,
        expired_on_other_node,
        mismatch_at_apply,
    } = landing;
    let mut out = vec![
        leaf(prefix, "armed", *armed),
        leaf(prefix, "stock_releases", *stock_releases),
    ];
    for (name, arm) in [("early", early), ("mid", mid), ("stock", stock)] {
        out.extend(crash_phase_arm_leaves(&format!("{prefix}.{name}"), arm));
    }
    let landing_prefix = format!("{prefix}.landing");
    out.extend([
        leaf(
            &landing_prefix,
            "evaluated_on_other_node",
            *evaluated_on_other_node,
        ),
        leaf(
            &landing_prefix,
            "condition_on_other_node",
            *condition_on_other_node,
        ),
        leaf(
            &landing_prefix,
            "expired_on_other_node",
            *expired_on_other_node,
        ),
        leaf(&landing_prefix, "mismatch_at_apply", *mismatch_at_apply),
    ]);
    out
}

fn victim_swap_half(m: &mut Marks) -> VictimSwapHalfStats {
    VictimSwapHalfStats {
        crashes: m.int(),
        victim_had_absorbed: m.int(),
        victim_had_inflight_sends: m.int(),
    }
}

fn victim_swap_half_leaves(prefix: &str, h: &VictimSwapHalfStats) -> Vec<(String, Value)> {
    let VictimSwapHalfStats {
        crashes,
        victim_had_absorbed,
        victim_had_inflight_sends,
    } = h;
    vec![
        leaf(prefix, "crashes", *crashes),
        leaf(prefix, "victim_had_absorbed", *victim_had_absorbed),
        leaf(prefix, "victim_had_inflight_sends", *victim_had_inflight_sends),
    ]
}

fn victim_swap(m: &mut Marks) -> VictimSwapStats {
    VictimSwapStats {
        applied: m.int(),
        acted_absorber: m.int(),
        same_victim: m.int(),
        no_absorber: m.int(),
        skipped_pending_pair: m.int(),
        victim_crashed_holds: m.int(),
        forced_onto_absorber: m.int(),
        census: VictimSwapCensusStats {
            treated: victim_swap_half(m),
            control: victim_swap_half(m),
        },
    }
}

fn victim_swap_leaves(prefix: &str, v: &VictimSwapStats) -> Vec<(String, Value)> {
    let VictimSwapStats {
        applied,
        acted_absorber,
        same_victim,
        no_absorber,
        skipped_pending_pair,
        victim_crashed_holds,
        forced_onto_absorber,
        census,
    } = v;
    let VictimSwapCensusStats { treated, control } = census;
    let mut out = vec![
        leaf(prefix, "applied", *applied),
        leaf(prefix, "acted_absorber", *acted_absorber),
        leaf(prefix, "same_victim", *same_victim),
        leaf(prefix, "no_absorber", *no_absorber),
        leaf(prefix, "skipped_pending_pair", *skipped_pending_pair),
        leaf(prefix, "victim_crashed_holds", *victim_crashed_holds),
        leaf(prefix, "forced_onto_absorber", *forced_onto_absorber),
    ];
    out.extend(victim_swap_half_leaves(
        &format!("{prefix}.census.treated"),
        treated,
    ));
    out.extend(victim_swap_half_leaves(
        &format!("{prefix}.census.control"),
        control,
    ));
    out
}

fn fresh_first_half(m: &mut Marks) -> FreshFirstHalfStats {
    FreshFirstHalfStats {
        contested_dispatches: m.int(),
        stale_drawn: m.int(),
        contested_down: m.int(),
        ghost_entries_from_restarted_origin: m.int(),
        overtaken: m.int(),
        ghost_entries_to_restarted_dest: m.int(),
        overtaken_at_restarted_dest: m.int(),
    }
}

fn fresh_first_half_leaves(prefix: &str, h: &FreshFirstHalfStats) -> Vec<(String, Value)> {
    let FreshFirstHalfStats {
        contested_dispatches,
        stale_drawn,
        contested_down,
        ghost_entries_from_restarted_origin,
        overtaken,
        ghost_entries_to_restarted_dest,
        overtaken_at_restarted_dest,
    } = h;
    vec![
        leaf(prefix, "contested_dispatches", *contested_dispatches),
        leaf(prefix, "stale_drawn", *stale_drawn),
        leaf(prefix, "contested_down", *contested_down),
        leaf(
            prefix,
            "ghost_entries_from_restarted_origin",
            *ghost_entries_from_restarted_origin,
        ),
        leaf(prefix, "overtaken", *overtaken),
        leaf(
            prefix,
            "ghost_entries_to_restarted_dest",
            *ghost_entries_to_restarted_dest,
        ),
        leaf(
            prefix,
            "overtaken_at_restarted_dest",
            *overtaken_at_restarted_dest,
        ),
    ]
}

fn fresh_first(m: &mut Marks) -> FreshFirstStats {
    FreshFirstStats {
        swaps: m.int(),
        repeat_swaps: m.int(),
        skipped_never_restarted_dest: m.int(),
        swap_count_hist_1: m.int(),
        swap_count_hist_2: m.int(),
        swap_count_hist_3: m.int(),
        swap_count_hist_4plus: m.int(),
        census: FreshFirstCensusStats {
            treated: fresh_first_half(m),
            control: fresh_first_half(m),
        },
    }
}

fn fresh_first_leaves(prefix: &str, f: &FreshFirstStats) -> Vec<(String, Value)> {
    let FreshFirstStats {
        swaps,
        repeat_swaps,
        skipped_never_restarted_dest,
        swap_count_hist_1,
        swap_count_hist_2,
        swap_count_hist_3,
        swap_count_hist_4plus,
        census,
    } = f;
    let FreshFirstCensusStats { treated, control } = census;
    let mut out = vec![
        leaf(prefix, "swaps", *swaps),
        leaf(prefix, "repeat_swaps", *repeat_swaps),
        leaf(prefix, "skipped_never_restarted_dest", *skipped_never_restarted_dest),
        leaf(prefix, "swap_count_hist_1", *swap_count_hist_1),
        leaf(prefix, "swap_count_hist_2", *swap_count_hist_2),
        leaf(prefix, "swap_count_hist_3", *swap_count_hist_3),
        leaf(prefix, "swap_count_hist_4plus", *swap_count_hist_4plus),
    ];
    out.extend(fresh_first_half_leaves(
        &format!("{prefix}.census.treated"),
        treated,
    ));
    out.extend(fresh_first_half_leaves(
        &format!("{prefix}.census.control"),
        control,
    ));
    out
}

fn ghost_signal(m: &mut Marks) -> GhostSignalStats {
    GhostSignalStats {
        fired_runs: m.int(),
    }
}

fn ghost_signal_leaves(prefix: &str, g: &GhostSignalStats) -> Vec<(String, Value)> {
    let GhostSignalStats { fired_runs } = g;
    vec![leaf(prefix, "fired_runs", *fired_runs)]
}

fn sched(m: &mut Marks) -> SchedStats {
    SchedStats {
        eligible_known: m.int(),
        eligible_built: m.int(),
        eligible_built_long: m.int(),
        eligibility_counted_steps: m.int(),
        eligibility_walked_steps: m.int(),
        eligibility_walked_elements: m.int(),
        eligibility_general_steps: m.int(),
        crash_scans_skipped: m.int(),
    }
}

fn sched_leaves(prefix: &str, s: &SchedStats) -> Vec<(String, Value)> {
    let SchedStats {
        eligible_known,
        eligible_built,
        eligible_built_long,
        eligibility_counted_steps,
        eligibility_walked_steps,
        eligibility_walked_elements,
        eligibility_general_steps,
        crash_scans_skipped,
    } = s;
    vec![
        leaf(prefix, "eligible_known", *eligible_known),
        leaf(prefix, "eligible_built", *eligible_built),
        leaf(prefix, "eligible_built_long", *eligible_built_long),
        leaf(prefix, "eligibility_counted_steps", *eligibility_counted_steps),
        leaf(prefix, "eligibility_walked_steps", *eligibility_walked_steps),
        leaf(prefix, "eligibility_walked_elements", *eligibility_walked_elements),
        leaf(prefix, "eligibility_general_steps", *eligibility_general_steps),
        leaf(prefix, "crash_scans_skipped", *crash_scans_skipped),
    ]
}

fn frame(m: &mut Marks) -> FrameStats {
    FrameStats {
        calls: m.int(),
        slots_built: m.int(),
        entry_frame_copies: m.int(),
        default_slots_filled: m.int(),
        params_filled: m.int(),
    }
}

fn frame_leaves(prefix: &str, f: &FrameStats) -> Vec<(String, Value)> {
    let FrameStats {
        calls,
        slots_built,
        entry_frame_copies,
        default_slots_filled,
        params_filled,
    } = f;
    vec![
        leaf(prefix, "calls", *calls),
        leaf(prefix, "slots_built", *slots_built),
        leaf(prefix, "entry_frame_copies", *entry_frame_copies),
        leaf(prefix, "default_slots_filled", *default_slots_filled),
        leaf(prefix, "params_filled", *params_filled),
    ]
}

fn frame_layout(m: &mut Marks) -> FrameLayoutStats {
    FrameLayoutStats {
        program_slots_before: m.int(),
        program_slots_after: m.int(),
    }
}

fn frame_layout_leaves(prefix: &str, f: &FrameLayoutStats) -> Vec<(String, Value)> {
    let FrameLayoutStats {
        program_slots_before,
        program_slots_after,
    } = f;
    vec![
        leaf(prefix, "program_slots_before", *program_slots_before),
        leaf(prefix, "program_slots_after", *program_slots_after),
    ]
}

fn value_sig(m: &mut Marks) -> ValueSigStats {
    ValueSigStats {
        leaf_hashes_deferred: m.int(),
    }
}

fn value_sig_leaves(prefix: &str, v: &ValueSigStats) -> Vec<(String, Value)> {
    let ValueSigStats {
        leaf_hashes_deferred,
    } = v;
    vec![leaf(prefix, "leaf_hashes_deferred", *leaf_hashes_deferred)]
}

fn value_struct(m: &mut Marks) -> ValueStructStats {
    ValueStructStats {
        literals: m.int(),
        literals_in_order: m.int(),
        literals_permuted: m.int(),
        key_hashes_avoided: m.int(),
        literal_entries: m.int(),
        field_reads: m.int(),
        other_lookups: m.int(),
        fallbacks: m.int(),
        fallback_key_hashes: m.int(),
        shapes_kept_as_maps: m.int(),
    }
}

fn value_struct_leaves(prefix: &str, v: &ValueStructStats) -> Vec<(String, Value)> {
    let ValueStructStats {
        literals,
        literals_in_order,
        literals_permuted,
        key_hashes_avoided,
        literal_entries,
        field_reads,
        other_lookups,
        fallbacks,
        fallback_key_hashes,
        shapes_kept_as_maps,
    } = v;
    vec![
        leaf(prefix, "literals", *literals),
        leaf(prefix, "literals_in_order", *literals_in_order),
        leaf(prefix, "literals_permuted", *literals_permuted),
        leaf(prefix, "key_hashes_avoided", *key_hashes_avoided),
        leaf(prefix, "literal_entries", *literal_entries),
        leaf(prefix, "field_reads", *field_reads),
        leaf(prefix, "other_lookups", *other_lookups),
        leaf(prefix, "fallbacks", *fallbacks),
        leaf(prefix, "fallback_key_hashes", *fallback_key_hashes),
        leaf(prefix, "shapes_kept_as_maps", *shapes_kept_as_maps),
    ]
}

fn eval_borrow(m: &mut Marks) -> EvalBorrowStats {
    EvalBorrowStats {
        handles_not_cloned: m.int(),
        scalars_not_cloned: m.int(),
    }
}

fn eval_borrow_leaves(prefix: &str, e: &EvalBorrowStats) -> Vec<(String, Value)> {
    let EvalBorrowStats {
        handles_not_cloned,
        scalars_not_cloned,
    } = e;
    vec![
        leaf(prefix, "handles_not_cloned", *handles_not_cloned),
        leaf(prefix, "scalars_not_cloned", *scalars_not_cloned),
    ]
}

fn run_buffers(m: &mut Marks) -> RunBufferStats {
    RunBufferStats {
        channel_table_grows: m.int(),
        channels_created: m.int(),
        log_vec_grows: m.int(),
        trace_vec_grows: m.int(),
    }
}

fn run_buffers_leaves(prefix: &str, r: &RunBufferStats) -> Vec<(String, Value)> {
    let RunBufferStats {
        channel_table_grows,
        channels_created,
        log_vec_grows,
        trace_vec_grows,
    } = r;
    vec![
        leaf(prefix, "channel_table_grows", *channel_table_grows),
        leaf(prefix, "channels_created", *channels_created),
        leaf(prefix, "log_vec_grows", *log_vec_grows),
        leaf(prefix, "trace_vec_grows", *trace_vec_grows),
    ]
}

fn compiled_ops(m: &mut Marks) -> CompiledOpsStats {
    CompiledOpsStats {
        label_execs: m.int(),
        legacy_labels: m.int(),
        stores_skipped: m.int(),
        stores_folded: m.int(),
        prints_fused: m.int(),
        print_trees_folded: m.int(),
    }
}

fn compiled_ops_leaves(prefix: &str, c: &CompiledOpsStats) -> Vec<(String, Value)> {
    let CompiledOpsStats {
        label_execs,
        legacy_labels,
        stores_skipped,
        stores_folded,
        prints_fused,
        print_trees_folded,
    } = c;
    vec![
        leaf(prefix, "label_execs", *label_execs),
        leaf(prefix, "legacy_labels", *legacy_labels),
        leaf(prefix, "stores_skipped", *stores_skipped),
        leaf(prefix, "stores_folded", *stores_folded),
        leaf(prefix, "prints_fused", *prints_fused),
        leaf(prefix, "print_trees_folded", *print_trees_folded),
    ]
}

fn compiled_expr(m: &mut Marks) -> CompiledExprStats {
    CompiledExprStats {
        leaf_operands_inline: m.int(),
        tree_evals: m.int(),
        legacy_evals: m.int(),
    }
}

fn compiled_expr_leaves(prefix: &str, c: &CompiledExprStats) -> Vec<(String, Value)> {
    let CompiledExprStats {
        leaf_operands_inline,
        tree_evals,
        legacy_evals,
    } = c;
    vec![
        leaf(prefix, "leaf_operands_inline", *leaf_operands_inline),
        leaf(prefix, "tree_evals", *tree_evals),
        leaf(prefix, "legacy_evals", *legacy_evals),
    ]
}

fn call_targets(m: &mut Marks) -> CallTargetStats {
    CallTargetStats {
        indexed: m.int(),
        fallback: m.int(),
    }
}

fn call_targets_leaves(prefix: &str, c: &CallTargetStats) -> Vec<(String, Value)> {
    let CallTargetStats { indexed, fallback } = c;
    vec![
        leaf(prefix, "indexed", *indexed),
        leaf(prefix, "fallback", *fallback),
    ]
}

fn grid_pool(m: &mut Marks) -> GridPoolStats {
    GridPoolStats {
        worker_idle_ns: m.int(),
        run_wall_ns: m.int(),
        pool_wall_ns: m.int(),
        pools: m.int(),
        runs: m.int(),
        batches: m.int(),
        capacity_gated_batches: m.int(),
        unfilled_in_ungated_batches: m.int(),
        fresh_ahead_launched: m.int(),
        batched_worker_idle_ns: m.int(),
        batched_run_wall_ns: m.int(),
        batched_pool_wall_ns: m.int(),
        batched_batches: m.int(),
        batched_runs: m.int(),
        writer_blocked_ns: m.int(),
    }
}

fn grid_pool_leaves(prefix: &str, g: &GridPoolStats) -> Vec<(String, Value)> {
    let GridPoolStats {
        worker_idle_ns,
        run_wall_ns,
        pool_wall_ns,
        pools,
        runs,
        batches,
        capacity_gated_batches,
        unfilled_in_ungated_batches,
        fresh_ahead_launched,
        batched_worker_idle_ns,
        batched_run_wall_ns,
        batched_pool_wall_ns,
        batched_batches,
        batched_runs,
        writer_blocked_ns,
    } = g;
    vec![
        leaf(prefix, "worker_idle_ns", *worker_idle_ns),
        leaf(prefix, "run_wall_ns", *run_wall_ns),
        leaf(prefix, "pool_wall_ns", *pool_wall_ns),
        leaf(prefix, "pools", *pools),
        leaf(prefix, "runs", *runs),
        leaf(prefix, "batches", *batches),
        leaf(prefix, "capacity_gated_batches", *capacity_gated_batches),
        leaf(prefix, "unfilled_in_ungated_batches", *unfilled_in_ungated_batches),
        leaf(prefix, "fresh_ahead_launched", *fresh_ahead_launched),
        leaf(prefix, "batched_worker_idle_ns", *batched_worker_idle_ns),
        leaf(prefix, "batched_run_wall_ns", *batched_run_wall_ns),
        leaf(prefix, "batched_pool_wall_ns", *batched_pool_wall_ns),
        leaf(prefix, "batched_batches", *batched_batches),
        leaf(prefix, "batched_runs", *batched_runs),
        leaf(prefix, "writer_blocked_ns", *writer_blocked_ns),
    ]
}

fn channel_table(m: &mut Marks) -> ChannelTableStats {
    ChannelTableStats {
        lookups: m.int(),
        lookup_misses: m.int(),
        dense_inserts: m.int(),
    }
}

fn channel_table_leaves(prefix: &str, c: &ChannelTableStats) -> Vec<(String, Value)> {
    let ChannelTableStats {
        lookups,
        lookup_misses,
        dense_inserts,
    } = c;
    vec![
        leaf(prefix, "lookups", *lookups),
        leaf(prefix, "lookup_misses", *lookup_misses),
        leaf(prefix, "dense_inserts", *dense_inserts),
    ]
}

fn print_content(m: &mut Marks) -> PrintContentStats {
    PrintContentStats { presized: m.int() }
}

fn print_content_leaves(prefix: &str, p: &PrintContentStats) -> Vec<(String, Value)> {
    let PrintContentStats { presized } = p;
    vec![leaf(prefix, "presized", *presized)]
}

fn history_writer(m: &mut Marks) -> HistoryWriterStats {
    HistoryWriterStats {
        busy_ns: m.int(),
        queue_full_sends: m.int(),
        blocked_ns: m.int(),
        commands: m.int(),
        text_buffers_allocated: m.int(),
        text_buffers_recycled: m.int(),
        text_buffers_dropped_oversize: m.int(),
        int_dict_memo: m.int(),
        int_dict_direct: m.int(),
        int_dict_hashed: m.int(),
        str_stats_compared: m.int(),
        str_stats_cells: m.int(),
        gather_calls: m.int(),
        gather_copies_skipped: m.int(),
    }
}

fn history_writer_leaves(prefix: &str, h: &HistoryWriterStats) -> Vec<(String, Value)> {
    let HistoryWriterStats {
        busy_ns,
        queue_full_sends,
        blocked_ns,
        commands,
        text_buffers_allocated,
        text_buffers_recycled,
        text_buffers_dropped_oversize,
        int_dict_memo,
        int_dict_direct,
        int_dict_hashed,
        str_stats_compared,
        str_stats_cells,
        gather_calls,
        gather_copies_skipped,
    } = h;
    vec![
        leaf(prefix, "busy_ns", *busy_ns),
        leaf(prefix, "queue_full_sends", *queue_full_sends),
        leaf(prefix, "blocked_ns", *blocked_ns),
        leaf(prefix, "commands", *commands),
        leaf(prefix, "text_buffers_allocated", *text_buffers_allocated),
        leaf(prefix, "text_buffers_recycled", *text_buffers_recycled),
        leaf(prefix, "text_buffers_dropped_oversize", *text_buffers_dropped_oversize),
        leaf(prefix, "int_dict_memo", *int_dict_memo),
        leaf(prefix, "int_dict_direct", *int_dict_direct),
        leaf(prefix, "int_dict_hashed", *int_dict_hashed),
        leaf(prefix, "str_stats_compared", *str_stats_compared),
        leaf(prefix, "str_stats_cells", *str_stats_cells),
        leaf(prefix, "gather_calls", *gather_calls),
        leaf(prefix, "gather_copies_skipped", *gather_copies_skipped),
    ]
}

fn stats_local(m: &mut Marks) -> StatsLocalStats {
    StatsLocalStats {
        folds: m.int(),
        folded_increments: m.int(),
    }
}

fn stats_local_leaves(prefix: &str, s: &StatsLocalStats) -> Vec<(String, Value)> {
    let StatsLocalStats {
        folds,
        folded_increments,
    } = s;
    vec![
        leaf(prefix, "folds", *folds),
        leaf(prefix, "folded_increments", *folded_increments),
    ]
}

fn timeline(m: &mut Marks) -> TimelineStats {
    TimelineStats {
        constant_short_circuits: m.int(),
        constant_inserts: m.int(),
    }
}

fn timeline_leaves(prefix: &str, t: &TimelineStats) -> Vec<(String, Value)> {
    let TimelineStats {
        constant_short_circuits,
        constant_inserts,
    } = t;
    vec![
        leaf(prefix, "constant_short_circuits", *constant_short_circuits),
        leaf(prefix, "constant_inserts", *constant_inserts),
    ]
}

fn run_setup(m: &mut Marks) -> RunSetupStats {
    RunSetupStats {
        program_clones_avoided: m.int(),
    }
}

fn run_setup_leaves(prefix: &str, r: &RunSetupStats) -> Vec<(String, Value)> {
    let RunSetupStats {
        program_clones_avoided,
    } = r;
    vec![leaf(prefix, "program_clones_avoided", *program_clones_avoided)]
}

fn trace_format(m: &mut Marks) -> TraceFormatStats {
    TraceFormatStats {
        enter_payload_reused: m.int(),
        enter_payload_formatted: m.int(),
    }
}

fn trace_format_leaves(prefix: &str, t: &TraceFormatStats) -> Vec<(String, Value)> {
    let TraceFormatStats {
        enter_payload_reused,
        enter_payload_formatted,
    } = t;
    vec![
        leaf(prefix, "enter_payload_reused", *enter_payload_reused),
        leaf(prefix, "enter_payload_formatted", *enter_payload_formatted),
    ]
}

fn history_format(m: &mut Marks) -> HistoryFormatStats {
    HistoryFormatStats {
        ops_streamed: m.int(),
    }
}

fn history_format_leaves(prefix: &str, h: &HistoryFormatStats) -> Vec<(String, Value)> {
    let HistoryFormatStats { ops_streamed } = h;
    vec![leaf(prefix, "ops_streamed", *ops_streamed)]
}

fn program_text(m: &mut Marks) -> ProgramTextStats {
    ProgramTextStats {
        names_interned: m.int(),
    }
}

fn program_text_leaves(prefix: &str, p: &ProgramTextStats) -> Vec<(String, Value)> {
    let ProgramTextStats { names_interned } = p;
    vec![leaf(prefix, "names_interned", *names_interned)]
}

fn plan_ready(m: &mut Marks) -> PlanReadyStats {
    PlanReadyStats {
        scans: m.int(),
        scans_skipped: m.int(),
        scans_empty: m.int(),
        events_released: m.int(),
    }
}

fn plan_ready_leaves(prefix: &str, p: &PlanReadyStats) -> Vec<(String, Value)> {
    let PlanReadyStats {
        scans,
        scans_skipped,
        scans_empty,
        events_released,
    } = p;
    vec![
        leaf(prefix, "scans", *scans),
        leaf(prefix, "scans_skipped", *scans_skipped),
        leaf(prefix, "scans_empty", *scans_empty),
        leaf(prefix, "events_released", *events_released),
    ]
}

fn text_buffer(m: &mut Marks) -> TextBufferStats {
    TextBufferStats {
        str_from_off_boundary: m.int(),
    }
}

fn text_buffer_leaves(prefix: &str, t: &TextBufferStats) -> Vec<(String, Value)> {
    let TextBufferStats {
        str_from_off_boundary,
    } = t;
    vec![leaf(prefix, "str_from_off_boundary", *str_from_off_boundary)]
}

fn plan_deliver(m: &mut Marks) -> PlanDeliverStats {
    PlanDeliverStats {
        lookups: m.int(),
        lookups_skipped: m.int(),
    }
}

fn plan_deliver_leaves(prefix: &str, p: &PlanDeliverStats) -> Vec<(String, Value)> {
    let PlanDeliverStats {
        lookups,
        lookups_skipped,
    } = p;
    vec![
        leaf(prefix, "lookups", *lookups),
        leaf(prefix, "lookups_skipped", *lookups_skipped),
    ]
}

fn replay(m: &mut Marks) -> ReplayStats {
    ReplayStats {
        parents_admitted: m.int(),
        children: m.int(),
        children_prefix: m.int(),
        children_plan_only: m.int(),
        slots_unfilled: m.int(),
        prefix_faithful: m.int(),
        tape_words_sum: m.int(),
        children_signal_fired: m.int(),
    }
}

fn marks_vec(m: &mut Marks, n: usize) -> Vec<u64> {
    (0..n).map(|_| m.int()).collect()
}

fn arm_selector_reward(m: &mut Marks) -> ArmSelectorRewardStats {
    ArmSelectorRewardStats {
        reward_runs_control: m.int(),
        reward_positive_control: m.int(),
        control_runs_by_direction: marks_vec(m, 12),
        control_reward_positive_by_direction: marks_vec(m, 12),
        control_runs_by_arm_direction: marks_vec(m, 96),
        control_reward_positive_by_arm_direction: marks_vec(m, 96),
        control_runs_by_combination: marks_vec(m, 72),
        control_reward_positive_by_combination: marks_vec(m, 72),
    }
}

fn arm_selector_learner(m: &mut Marks) -> ArmSelectorLearnerStats {
    ArmSelectorLearnerStats {
        chosen_runs: m.int(),
        departures: m.int(),
        axis_leader_agreements: m.int(),
        axis_draws: m.int(),
        chosen_placed_runs: m.int(),
        cells: m.int(),
        reward_runs_treated: m.int(),
        reward_positive_treated: m.int(),
        reward_runs_control: m.int(),
        reward_positive_control: m.int(),
        chosen_by_direction: marks_vec(m, 12),
        leader_margin_micro: marks_vec(m, 5),
        control_runs_by_direction: marks_vec(m, 12),
        control_reward_positive_by_direction: marks_vec(m, 12),
        control_runs_by_arm_direction: marks_vec(m, 96),
        control_reward_positive_by_arm_direction: marks_vec(m, 96),
        chosen_by_combination: marks_vec(m, 72),
        control_runs_by_combination: marks_vec(m, 72),
        control_reward_positive_by_combination: marks_vec(m, 72),
    }
}

fn arm_selector_explore(m: &mut Marks) -> ArmSelectorExploreStats {
    ArmSelectorExploreStats {
        draws: m.int(),
        coin_runs: m.int(),
        warmup_coin_runs: m.int(),
        share_micro: m.int(),
        margin_micro: m.int(),
        draws_by_learner: marks_vec(m, 3),
        coin_runs_by_learner: marks_vec(m, 3),
        share_micro_by_learner: marks_vec(m, 3),
        draws_by_arm: marks_vec(m, 8),
        coin_runs_by_arm: marks_vec(m, 8),
        margin_micro_by_arm: marks_vec(m, 8),
        share_hist: marks_vec(m, 10),
    }
}

fn arm_selector_pooled(m: &mut Marks) -> ArmSelectorPooledStats {
    ArmSelectorPooledStats {
        draws: m.int(),
        cells: m.int(),
        cells_by_learner: marks_vec(m, 3),
        observations_by_learner: marks_vec(m, 3),
    }
}

fn arm_selector_axis(m: &mut Marks) -> ArmSelectorAxisStats {
    ArmSelectorAxisStats {
        chosen_runs: m.int(),
        departures: m.int(),
        axis_leader_agreements: m.int(),
        axis_draws: m.int(),
        observations: m.int(),
        reward_runs_treated: m.int(),
        reward_positive_treated: m.int(),
        reward_runs_control: m.int(),
        reward_positive_control: m.int(),
        cells: m.int(),
        chosen_placed_runs: m.int(),
        reward_runs_by_arm: marks_vec(m, 8),
        reward_positive_by_arm: marks_vec(m, 8),
        chosen_by_direction: marks_vec(m, 12),
        leader_margin_micro: marks_vec(m, 5),
        control_runs_by_direction: marks_vec(m, 12),
        control_reward_positive_by_direction: marks_vec(m, 12),
        control_runs_by_arm_direction: marks_vec(m, 96),
        control_reward_positive_by_arm_direction: marks_vec(m, 96),
        chosen_by_combination: marks_vec(m, 72),
        control_runs_by_combination: marks_vec(m, 72),
        control_reward_positive_by_combination: marks_vec(m, 72),
        explore: arm_selector_explore(m),
        pooled: arm_selector_pooled(m),
        overtaken_ghost: arm_selector_learner(m),
        absorber_cycle: arm_selector_learner(m),
        cycle_before_request: arm_selector_learner(m),
        ghost_signal: arm_selector_reward(m),
        either_shape: arm_selector_reward(m),
        mutual_absorber_cycle: arm_selector_reward(m),
        exchange_before_request: arm_selector_reward(m),
    }
}

fn vec_leaves(prefix: &str, name: &str, v: &[u64]) -> Vec<(String, Value)> {
    let p = format!("{prefix}.{name}");
    v.iter()
        .enumerate()
        .map(|(i, x)| leaf(&p, &i.to_string(), *x))
        .collect()
}

fn arm_selector_reward_leaves(prefix: &str, r: &ArmSelectorRewardStats) -> Vec<(String, Value)> {
    let ArmSelectorRewardStats {
        reward_runs_control,
        reward_positive_control,
        control_runs_by_direction,
        control_reward_positive_by_direction,
        control_runs_by_arm_direction,
        control_reward_positive_by_arm_direction,
        control_runs_by_combination,
        control_reward_positive_by_combination,
    } = r;
    let mut out = vec![
        leaf(prefix, "reward_runs_control", *reward_runs_control),
        leaf(prefix, "reward_positive_control", *reward_positive_control),
    ];
    out.extend(vec_leaves(prefix, "control_runs_by_direction", control_runs_by_direction));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_direction",
        control_reward_positive_by_direction,
    ));
    out.extend(vec_leaves(prefix, "control_runs_by_arm_direction", control_runs_by_arm_direction));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_arm_direction",
        control_reward_positive_by_arm_direction,
    ));
    out.extend(vec_leaves(prefix, "control_runs_by_combination", control_runs_by_combination));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_combination",
        control_reward_positive_by_combination,
    ));
    out
}

fn arm_selector_learner_leaves(prefix: &str, l: &ArmSelectorLearnerStats) -> Vec<(String, Value)> {
    let ArmSelectorLearnerStats {
        chosen_runs,
        departures,
        axis_leader_agreements,
        axis_draws,
        chosen_placed_runs,
        cells,
        reward_runs_treated,
        reward_positive_treated,
        reward_runs_control,
        reward_positive_control,
        chosen_by_direction,
        leader_margin_micro,
        control_runs_by_direction,
        control_reward_positive_by_direction,
        control_runs_by_arm_direction,
        control_reward_positive_by_arm_direction,
        chosen_by_combination,
        control_runs_by_combination,
        control_reward_positive_by_combination,
    } = l;
    let mut out = vec![
        leaf(prefix, "chosen_runs", *chosen_runs),
        leaf(prefix, "departures", *departures),
        leaf(prefix, "axis_leader_agreements", *axis_leader_agreements),
        leaf(prefix, "axis_draws", *axis_draws),
        leaf(prefix, "chosen_placed_runs", *chosen_placed_runs),
        leaf(prefix, "cells", *cells),
        leaf(prefix, "reward_runs_treated", *reward_runs_treated),
        leaf(prefix, "reward_positive_treated", *reward_positive_treated),
        leaf(prefix, "reward_runs_control", *reward_runs_control),
        leaf(prefix, "reward_positive_control", *reward_positive_control),
    ];
    out.extend(vec_leaves(prefix, "chosen_by_direction", chosen_by_direction));
    out.extend(vec_leaves(prefix, "leader_margin_micro", leader_margin_micro));
    out.extend(vec_leaves(prefix, "control_runs_by_direction", control_runs_by_direction));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_direction",
        control_reward_positive_by_direction,
    ));
    out.extend(vec_leaves(prefix, "control_runs_by_arm_direction", control_runs_by_arm_direction));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_arm_direction",
        control_reward_positive_by_arm_direction,
    ));
    out.extend(vec_leaves(prefix, "chosen_by_combination", chosen_by_combination));
    out.extend(vec_leaves(prefix, "control_runs_by_combination", control_runs_by_combination));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_combination",
        control_reward_positive_by_combination,
    ));
    out
}

fn arm_selector_explore_leaves(prefix: &str, e: &ArmSelectorExploreStats) -> Vec<(String, Value)> {
    let ArmSelectorExploreStats {
        draws,
        coin_runs,
        warmup_coin_runs,
        share_micro,
        margin_micro,
        draws_by_learner,
        coin_runs_by_learner,
        share_micro_by_learner,
        draws_by_arm,
        coin_runs_by_arm,
        margin_micro_by_arm,
        share_hist,
    } = e;
    let mut out = vec![
        leaf(prefix, "draws", *draws),
        leaf(prefix, "coin_runs", *coin_runs),
        leaf(prefix, "warmup_coin_runs", *warmup_coin_runs),
        leaf(prefix, "share_micro", *share_micro),
        leaf(prefix, "margin_micro", *margin_micro),
    ];
    out.extend(vec_leaves(prefix, "draws_by_learner", draws_by_learner));
    out.extend(vec_leaves(prefix, "coin_runs_by_learner", coin_runs_by_learner));
    out.extend(vec_leaves(prefix, "share_micro_by_learner", share_micro_by_learner));
    out.extend(vec_leaves(prefix, "draws_by_arm", draws_by_arm));
    out.extend(vec_leaves(prefix, "coin_runs_by_arm", coin_runs_by_arm));
    out.extend(vec_leaves(prefix, "margin_micro_by_arm", margin_micro_by_arm));
    out.extend(vec_leaves(prefix, "share_hist", share_hist));
    out
}

fn arm_selector_pooled_leaves(prefix: &str, p: &ArmSelectorPooledStats) -> Vec<(String, Value)> {
    let ArmSelectorPooledStats {
        draws,
        cells,
        cells_by_learner,
        observations_by_learner,
    } = p;
    let mut out = vec![leaf(prefix, "draws", *draws), leaf(prefix, "cells", *cells)];
    out.extend(vec_leaves(prefix, "cells_by_learner", cells_by_learner));
    out.extend(vec_leaves(prefix, "observations_by_learner", observations_by_learner));
    out
}

fn arm_selector_axis_leaves(prefix: &str, a: &ArmSelectorAxisStats) -> Vec<(String, Value)> {
    let ArmSelectorAxisStats {
        chosen_runs,
        departures,
        axis_leader_agreements,
        axis_draws,
        observations,
        reward_runs_treated,
        reward_positive_treated,
        reward_runs_control,
        reward_positive_control,
        cells,
        chosen_placed_runs,
        reward_runs_by_arm,
        reward_positive_by_arm,
        chosen_by_direction,
        leader_margin_micro,
        control_runs_by_direction,
        control_reward_positive_by_direction,
        control_runs_by_arm_direction,
        control_reward_positive_by_arm_direction,
        chosen_by_combination,
        control_runs_by_combination,
        control_reward_positive_by_combination,
        explore,
        pooled,
        overtaken_ghost,
        absorber_cycle,
        mutual_absorber_cycle,
        ghost_signal,
        either_shape,
        cycle_before_request,
        exchange_before_request,
    } = a;
    let mut out = vec![
        leaf(prefix, "chosen_runs", *chosen_runs),
        leaf(prefix, "departures", *departures),
        leaf(prefix, "axis_leader_agreements", *axis_leader_agreements),
        leaf(prefix, "axis_draws", *axis_draws),
        leaf(prefix, "observations", *observations),
        leaf(prefix, "reward_runs_treated", *reward_runs_treated),
        leaf(prefix, "reward_positive_treated", *reward_positive_treated),
        leaf(prefix, "reward_runs_control", *reward_runs_control),
        leaf(prefix, "reward_positive_control", *reward_positive_control),
        leaf(prefix, "cells", *cells),
        leaf(prefix, "chosen_placed_runs", *chosen_placed_runs),
    ];
    out.extend(vec_leaves(prefix, "reward_runs_by_arm", reward_runs_by_arm));
    out.extend(vec_leaves(prefix, "reward_positive_by_arm", reward_positive_by_arm));
    out.extend(vec_leaves(prefix, "chosen_by_direction", chosen_by_direction));
    out.extend(vec_leaves(prefix, "leader_margin_micro", leader_margin_micro));
    out.extend(vec_leaves(prefix, "control_runs_by_direction", control_runs_by_direction));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_direction",
        control_reward_positive_by_direction,
    ));
    out.extend(vec_leaves(prefix, "control_runs_by_arm_direction", control_runs_by_arm_direction));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_arm_direction",
        control_reward_positive_by_arm_direction,
    ));
    out.extend(vec_leaves(prefix, "chosen_by_combination", chosen_by_combination));
    out.extend(vec_leaves(prefix, "control_runs_by_combination", control_runs_by_combination));
    out.extend(vec_leaves(
        prefix,
        "control_reward_positive_by_combination",
        control_reward_positive_by_combination,
    ));
    out.extend(arm_selector_explore_leaves(&format!("{prefix}.explore"), explore));
    out.extend(arm_selector_pooled_leaves(&format!("{prefix}.pooled"), pooled));
    out.extend(arm_selector_learner_leaves(&format!("{prefix}.overtaken_ghost"), overtaken_ghost));
    out.extend(arm_selector_learner_leaves(&format!("{prefix}.absorber_cycle"), absorber_cycle));
    out.extend(arm_selector_learner_leaves(
        &format!("{prefix}.cycle_before_request"),
        cycle_before_request,
    ));
    out.extend(arm_selector_reward_leaves(&format!("{prefix}.ghost_signal"), ghost_signal));
    out.extend(arm_selector_reward_leaves(&format!("{prefix}.either_shape"), either_shape));
    out.extend(arm_selector_reward_leaves(
        &format!("{prefix}.mutual_absorber_cycle"),
        mutual_absorber_cycle,
    ));
    out.extend(arm_selector_reward_leaves(
        &format!("{prefix}.exchange_before_request"),
        exchange_before_request,
    ));
    out
}

fn replay_leaves(prefix: &str, r: &ReplayStats) -> Vec<(String, Value)> {
    let ReplayStats {
        parents_admitted,
        children,
        children_prefix,
        children_plan_only,
        slots_unfilled,
        prefix_faithful,
        tape_words_sum,
        children_signal_fired,
    } = r;
    vec![
        leaf(prefix, "parents_admitted", *parents_admitted),
        leaf(prefix, "children", *children),
        leaf(prefix, "children_prefix", *children_prefix),
        leaf(prefix, "children_plan_only", *children_plan_only),
        leaf(prefix, "slots_unfilled", *slots_unfilled),
        leaf(prefix, "prefix_faithful", *prefix_faithful),
        leaf(prefix, "tape_words_sum", *tape_words_sum),
        leaf(prefix, "children_signal_fired", *children_signal_fired),
    ]
}

fn timer_context(m: &mut Marks) -> TimerContextStats {
    TimerContextStats {
        probe_firings: m.int(),
        probe_acted: m.int(),
        biased_steps: m.int(),
        biased_steps_promoted: m.int(),
        biased_steps_suppressed: m.int(),
        steps_excluded_selector: m.int(),
        cells_engaged: m.int(),
    }
}

fn timer_context_leaves(prefix: &str, t: &TimerContextStats) -> Vec<(String, Value)> {
    let TimerContextStats {
        probe_firings,
        probe_acted,
        biased_steps,
        biased_steps_promoted,
        biased_steps_suppressed,
        steps_excluded_selector,
        cells_engaged,
    } = t;
    vec![
        leaf(prefix, "probe_firings", *probe_firings),
        leaf(prefix, "probe_acted", *probe_acted),
        leaf(prefix, "biased_steps", *biased_steps),
        leaf(prefix, "biased_steps_promoted", *biased_steps_promoted),
        leaf(prefix, "biased_steps_suppressed", *biased_steps_suppressed),
        leaf(prefix, "steps_excluded_selector", *steps_excluded_selector),
        leaf(prefix, "cells_engaged", *cells_engaged),
    ]
}

/// The blocks the snapshot is made of. Destructured without a rest pattern, so
/// a block added to the snapshot does not compile until it is named, which is
/// what keeps a whole block from going unexported.
fn pair_order_half(m: &mut Marks) -> PairOrderHalfStats {
    PairOrderHalfStats {
        contests: m.int(),
        inorder_draws: m.int(),
        pair_entries: m.int(),
        pair_entries_ghost: m.int(),
        inversions: m.int(),
        inversions_ghost: m.int(),
        sampled_runs: m.int(),
    }
}

fn pair_order_half_leaves(prefix: &str, h: &PairOrderHalfStats) -> Vec<(String, Value)> {
    let PairOrderHalfStats {
        contests,
        inorder_draws,
        pair_entries,
        pair_entries_ghost,
        inversions,
        inversions_ghost,
        sampled_runs,
    } = h;
    vec![
        leaf(prefix, "contests", *contests),
        leaf(prefix, "inorder_draws", *inorder_draws),
        leaf(prefix, "pair_entries", *pair_entries),
        leaf(prefix, "pair_entries_ghost", *pair_entries_ghost),
        leaf(prefix, "inversions", *inversions),
        leaf(prefix, "inversions_ghost", *inversions_ghost),
        leaf(prefix, "sampled_runs", *sampled_runs),
    ]
}

fn pair_order(m: &mut Marks) -> PairOrderStats {
    PairOrderStats {
        corrected: m.int(),
        contests_by_class: PairOrderClassCounts {
            ghost: m.int(),
            fresh: m.int(),
        },
        corrections_ghost: m.int(),
        corrections_fresh: m.int(),
        corrections_fresh_suppressed: m.int(),
        census: PairOrderCensusStats {
            treated: pair_order_half(m),
            control: pair_order_half(m),
        },
    }
}

fn pair_order_leaves(prefix: &str, p: &PairOrderStats) -> Vec<(String, Value)> {
    let PairOrderStats {
        corrected,
        contests_by_class,
        corrections_ghost,
        corrections_fresh,
        corrections_fresh_suppressed,
        census,
    } = p;
    let PairOrderClassCounts { ghost, fresh } = contests_by_class;
    let PairOrderCensusStats { treated, control } = census;
    let mut out = vec![
        leaf(prefix, "corrected", *corrected),
        leaf(&format!("{prefix}.contests_by_class"), "ghost", *ghost),
        leaf(&format!("{prefix}.contests_by_class"), "fresh", *fresh),
        leaf(prefix, "corrections_ghost", *corrections_ghost),
        leaf(prefix, "corrections_fresh", *corrections_fresh),
        leaf(
            prefix,
            "corrections_fresh_suppressed",
            *corrections_fresh_suppressed,
        ),
    ];
    out.extend(pair_order_half_leaves(&format!("{prefix}.census.treated"), treated));
    out.extend(pair_order_half_leaves(&format!("{prefix}.census.control"), control));
    out
}

fn client_anchor_half(m: &mut Marks) -> ClientAnchorHalfStats {
    ClientAnchorHalfStats {
        runs: m.int(),
        completed_runs: m.int(),
        population: m.int(),
        fanout_windows: m.int(),
        post_fault_invocations: m.int(),
        in_window_invocations: m.int(),
    }
}

fn client_anchor_half_leaves(prefix: &str, h: &ClientAnchorHalfStats) -> Vec<(String, Value)> {
    let ClientAnchorHalfStats {
        runs,
        completed_runs,
        population,
        fanout_windows,
        post_fault_invocations,
        in_window_invocations,
    } = h;
    vec![
        leaf(prefix, "runs", *runs),
        leaf(prefix, "completed_runs", *completed_runs),
        leaf(prefix, "population", *population),
        leaf(prefix, "fanout_windows", *fanout_windows),
        leaf(prefix, "post_fault_invocations", *post_fault_invocations),
        leaf(prefix, "in_window_invocations", *in_window_invocations),
    ]
}

fn client_anchor(m: &mut Marks) -> ClientAnchorStats {
    ClientAnchorStats {
        held: m.int(),
        released: ClientAnchorReleaseStats {
            expiry: m.int(),
            dry_queue: m.int(),
        },
        held_at_first_firing: ClientAnchorHeldHist {
            zero: m.int(),
            one: m.int(),
            two: m.int(),
            three_plus: m.int(),
        },
        held_at_exit: m.int(),
        runs_with_held_at_exit: m.int(),
        hold_steps_sum: m.int(),
        census: ClientAnchorCensusStats {
            treated: client_anchor_half(m),
            control: client_anchor_half(m),
        },
        axis: ClientAnchorAxisStats {
            arm_runs: ClientAnchorArmRuns {
                hold: m.int(),
                rush: m.int(),
                stock: m.int(),
            },
            rush: ClientAnchorRushStats {
                ops: m.int(),
                records_prioritized: m.int(),
                was_pick: m.int(),
                displaced: m.int(),
                first_delivery_distance: ClientAnchorFirstDelivery {
                    hold: client_anchor_distance(m),
                    rush: client_anchor_distance(m),
                    stock: client_anchor_distance(m),
                },
            },
        },
        first_post_fault_entry: ClientAnchorFirstEntry {
            runs: marks_vec(m, 12),
            steps_sum: marks_vec(m, 12),
        },
    }
}

fn client_anchor_distance(m: &mut Marks) -> ClientAnchorDistance {
    ClientAnchorDistance {
        sum: m.int(),
        count: m.int(),
    }
}

fn client_anchor_distance_leaves(prefix: &str, d: &ClientAnchorDistance) -> Vec<(String, Value)> {
    let ClientAnchorDistance { sum, count } = d;
    vec![leaf(prefix, "sum", *sum), leaf(prefix, "count", *count)]
}

fn client_anchor_leaves(prefix: &str, c: &ClientAnchorStats) -> Vec<(String, Value)> {
    let ClientAnchorStats {
        held,
        released,
        held_at_first_firing,
        held_at_exit,
        runs_with_held_at_exit,
        hold_steps_sum,
        census,
        axis,
        first_post_fault_entry,
    } = c;
    let ClientAnchorReleaseStats { expiry, dry_queue } = released;
    let ClientAnchorHeldHist {
        zero,
        one,
        two,
        three_plus,
    } = held_at_first_firing;
    let ClientAnchorCensusStats { treated, control } = census;
    let r = format!("{prefix}.released");
    let h = format!("{prefix}.held_at_first_firing");
    let mut out = vec![
        leaf(prefix, "held", *held),
        leaf(&r, "expiry", *expiry),
        leaf(&r, "dry_queue", *dry_queue),
        leaf(&h, "zero", *zero),
        leaf(&h, "one", *one),
        leaf(&h, "two", *two),
        leaf(&h, "three_plus", *three_plus),
        leaf(prefix, "held_at_exit", *held_at_exit),
        leaf(prefix, "runs_with_held_at_exit", *runs_with_held_at_exit),
        leaf(prefix, "hold_steps_sum", *hold_steps_sum),
    ];
    out.extend(client_anchor_half_leaves(&format!("{prefix}.census.treated"), treated));
    out.extend(client_anchor_half_leaves(&format!("{prefix}.census.control"), control));
    let ClientAnchorAxisStats { arm_runs, rush } = axis;
    let ClientAnchorArmRuns { hold, rush: rush_runs, stock } = arm_runs;
    let a = format!("{prefix}.axis.arm_runs");
    out.push(leaf(&a, "hold", *hold));
    out.push(leaf(&a, "rush", *rush_runs));
    out.push(leaf(&a, "stock", *stock));
    let ClientAnchorRushStats {
        ops,
        records_prioritized,
        was_pick,
        displaced,
        first_delivery_distance,
    } = rush;
    let u = format!("{prefix}.axis.rush");
    out.push(leaf(&u, "ops", *ops));
    out.push(leaf(&u, "records_prioritized", *records_prioritized));
    out.push(leaf(&u, "was_pick", *was_pick));
    out.push(leaf(&u, "displaced", *displaced));
    let ClientAnchorFirstDelivery {
        hold: hold_distance,
        rush: rush_distance,
        stock: stock_distance,
    } = first_delivery_distance;
    let d = format!("{u}.first_delivery_distance");
    out.extend(client_anchor_distance_leaves(&format!("{d}.hold"), hold_distance));
    out.extend(client_anchor_distance_leaves(&format!("{d}.rush"), rush_distance));
    out.extend(client_anchor_distance_leaves(&format!("{d}.stock"), stock_distance));
    let ClientAnchorFirstEntry { runs, steps_sum } = first_post_fault_entry;
    let e = format!("{prefix}.first_post_fault_entry");
    out.extend(vec_leaves(&e, "runs", runs));
    out.extend(vec_leaves(&e, "steps_sum", steps_sum));
    out
}

fn block_names(s: &UtilizationSnapshot) -> Vec<&'static str> {
    let UtilizationSnapshot {
        rng_streams: _,
        steer: _,
        steer_empty_slice: _,
        steer_authority: _,
        steer_reach: _,
        multiplier_authority: _,
        recovery_weight_placebo: _,
        sched: _,
        purgatory: _,
        aos: _,
        dedup: _,
        feedback: _,
        curriculum: _,
        crash_recovery: _,
        recovery_window: _,
        ordered_h3: _,
        post_fault_ops: _,
        plan_deps: _,
        delivery_effects: _,
        timer_effects: _,
        timer_steer: _,
        crash_anchor: _,
        termination: _,
        prefix_extension: _,
        quiet_stretch: _,
        run_cap: _,
        stall_cap: _,
        stall_release: _,
        crash_place: _,
        crash_phase: _,
        victim_swap: _,
        ghost_signal: _,
        frame: _,
        frame_layout: _,
        value_sig: _,
        value_struct: _,
        eval_borrow: _,
        run_buffers: _,
        print_content: _,
        compiled_ops: _,
        compiled_expr: _,
        call_targets: _,
        channel_table: _,
        stats_local: _,
        fresh_first: _,
        pair_order: _,
        client_anchor: _,
        arm_selector_axis: _,
        replay: _,
        timer_context: _,
        timeline_keys: _,
        steer_terms: _,
        history_writer: _,
        grid_pool: _,
        timeline: _,
        run_setup: _,
        trace_format: _,
        history_format: _,
        program_text: _,
        plan_ready: _,
        plan_deliver: _,
        text_buffer: _,
    } = s;
    vec![
        "rng_streams",
        "steer",
        "steer_empty_slice",
        "steer_authority",
        "steer_reach",
        "multiplier_authority",
        "recovery_weight_placebo",
        "sched",
        "purgatory",
        "aos",
        "dedup",
        "feedback",
        "curriculum",
        "crash_recovery",
        "recovery_window",
        "ordered_h3",
        "post_fault_ops",
        "plan_deps",
        "delivery_effects",
        "timer_effects",
        "timer_steer",
        "crash_anchor",
        "termination",
        "prefix_extension",
        "quiet_stretch",
        "run_cap",
        "stall_cap",
        "stall_release",
        "crash_place",
        "crash_phase",
        "victim_swap",
        "ghost_signal",
        "frame",
        "frame_layout",
        "value_sig",
        "value_struct",
        "eval_borrow",
        "run_buffers",
        "print_content",
        "compiled_ops",
        "compiled_expr",
        "call_targets",
        "channel_table",
        "stats_local",
        "fresh_first",
        "pair_order",
        "client_anchor",
        "arm_selector_axis",
        "replay",
        "timer_context",
        "timeline_keys",
        "steer_terms",
        "history_writer",
        "grid_pool",
        "timeline",
        "run_setup",
        "trace_format",
        "history_format",
        "program_text",
        "plan_ready",
        "plan_deliver",
        "text_buffer",
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
    s.run_cap = run_cap(&mut m);
    s.stall_cap = stall_cap(&mut m);
    s.stall_release = stall_release(&mut m);
    s.crash_place = crash_place(&mut m);
    s.crash_phase = crash_phase(&mut m);
    s.victim_swap = victim_swap(&mut m);
    s.ghost_signal = ghost_signal(&mut m);
    s.sched = sched(&mut m);
    s.frame = frame(&mut m);
    s.frame_layout = frame_layout(&mut m);
    s.value_sig = value_sig(&mut m);
    s.value_struct = value_struct(&mut m);
    s.eval_borrow = eval_borrow(&mut m);
    s.run_buffers = run_buffers(&mut m);
    s.print_content = print_content(&mut m);
    s.compiled_ops = compiled_ops(&mut m);
    s.compiled_expr = compiled_expr(&mut m);
    s.call_targets = call_targets(&mut m);
    s.channel_table = channel_table(&mut m);
    s.stats_local = stats_local(&mut m);
    s.fresh_first = fresh_first(&mut m);
    s.pair_order = pair_order(&mut m);
    s.client_anchor = client_anchor(&mut m);
    s.arm_selector_axis = arm_selector_axis(&mut m);
    s.replay = replay(&mut m);
    s.timer_context = timer_context(&mut m);
    s.plan_deps = plan_deps(&mut m);
    s.history_writer = history_writer(&mut m);
    s.grid_pool = grid_pool(&mut m);
    s.timeline = timeline(&mut m);
    s.run_setup = run_setup(&mut m);
    s.trace_format = trace_format(&mut m);
    s.history_format = history_format(&mut m);
    s.program_text = program_text(&mut m);
    s.plan_ready = plan_ready(&mut m);
    s.plan_deliver = plan_deliver(&mut m);
    s.text_buffer = text_buffer(&mut m);
    let mut expected = steer_authority_leaves("steer_authority", &s.steer_authority);
    expected.extend(termination_leaves("termination", &s.termination));
    expected.extend(plan_deps_leaves("plan_deps", &s.plan_deps));
    expected.extend(delivery_effects_leaves(
        "delivery_effects",
        &s.delivery_effects,
    ));
    expected.extend(run_cap_leaves("run_cap", &s.run_cap));
    expected.extend(stall_cap_leaves("stall_cap", &s.stall_cap));
    expected.extend(stall_release_leaves("stall_release", &s.stall_release));
    expected.extend(crash_place_leaves("crash_place", &s.crash_place));
    expected.extend(crash_phase_leaves("crash_phase", &s.crash_phase));
    expected.extend(victim_swap_leaves("victim_swap", &s.victim_swap));
    expected.extend(ghost_signal_leaves("ghost_signal", &s.ghost_signal));
    expected.extend(sched_leaves("sched", &s.sched));
    expected.extend(frame_leaves("frame", &s.frame));
    expected.extend(frame_layout_leaves("frame_layout", &s.frame_layout));
    expected.extend(value_sig_leaves("value_sig", &s.value_sig));
    expected.extend(value_struct_leaves("value_struct", &s.value_struct));
    expected.extend(eval_borrow_leaves("eval_borrow", &s.eval_borrow));
    expected.extend(run_buffers_leaves("run_buffers", &s.run_buffers));
    expected.extend(print_content_leaves("print_content", &s.print_content));
    expected.extend(compiled_ops_leaves("compiled_ops", &s.compiled_ops));
    expected.extend(compiled_expr_leaves("compiled_expr", &s.compiled_expr));
    expected.extend(call_targets_leaves("call_targets", &s.call_targets));
    expected.extend(channel_table_leaves("channel_table", &s.channel_table));
    expected.extend(stats_local_leaves("stats_local", &s.stats_local));
    expected.extend(fresh_first_leaves("fresh_first", &s.fresh_first));
    expected.extend(pair_order_leaves("pair_order", &s.pair_order));
    expected.extend(client_anchor_leaves("client_anchor", &s.client_anchor));
    expected.extend(arm_selector_axis_leaves("arm_selector_axis", &s.arm_selector_axis));
    expected.extend(replay_leaves("replay", &s.replay));
    expected.extend(timer_context_leaves("timer_context", &s.timer_context));
    expected.extend(history_writer_leaves("history_writer", &s.history_writer));
    expected.extend(grid_pool_leaves("grid_pool", &s.grid_pool));
    expected.extend(timeline_leaves("timeline", &s.timeline));
    expected.extend(run_setup_leaves("run_setup", &s.run_setup));
    expected.extend(trace_format_leaves("trace_format", &s.trace_format));
    expected.extend(history_format_leaves("history_format", &s.history_format));
    expected.extend(program_text_leaves("program_text", &s.program_text));
    expected.extend(plan_ready_leaves("plan_ready", &s.plan_ready));
    expected.extend(plan_deliver_leaves("plan_deliver", &s.plan_deliver));
    expected.extend(text_buffer_leaves("text_buffer", &s.text_buffer));
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

    for block in [
        "steer_authority",
        "termination",
        "plan_deps",
        "delivery_effects",
        "run_cap",
        "stall_cap",
        "stall_release",
        "crash_place",
        "crash_phase",
        "victim_swap",
        "ghost_signal",
        "sched",
        "frame",
        "frame_layout",
        "value_sig",
        "value_struct",
        "eval_borrow",
        "run_buffers",
        "print_content",
        "compiled_ops",
        "compiled_expr",
        "call_targets",
        "channel_table",
        "stats_local",
        "fresh_first",
        "pair_order",
        "client_anchor",
        "arm_selector_axis",
        "replay",
        "timer_context",
        "history_writer",
        "grid_pool",
        "timeline",
        "run_setup",
        "trace_format",
        "history_format",
        "program_text",
        "plan_ready",
        "plan_deliver",
        "text_buffer",
    ] {
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
