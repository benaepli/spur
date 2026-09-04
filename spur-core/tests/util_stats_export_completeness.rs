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
    self, AcceptanceDistanceBucket, AcceptanceDistanceStats, ClientAnchorCensusStats,
    ClientAnchorHalfStats, ClientAnchorHeldHist, ClientAnchorReleaseStats, ClientAnchorStats,
    CrashCensusStats, CrashPhaseArmStats, CrashPhaseLandingStats, CrashPhaseMovedStats,
    CrashPhaseStats, CrashPlaceStats, DeliveryEffect, DeliveryEffectStats, FreshFirstCensusStats,
    FreshFirstHalfStats, FreshFirstStats, GhostSignalStats, PairOrderCensusStats, PairOrderHalfStats, PairOrderStats, ReplayStats, RunCapStats, SteerAuthorityStats, TerminationStats, TerminationTally,
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

fn crash_place(m: &mut Marks) -> CrashPlaceStats {
    CrashPlaceStats {
        draws: m.int(),
        capped_draws: m.int(),
        holds: m.int(),
        held_steps_sum: m.int(),
    }
}

fn crash_place_leaves(prefix: &str, c: &CrashPlaceStats) -> Vec<(String, Value)> {
    let CrashPlaceStats {
        draws,
        capped_draws,
        holds,
        held_steps_sum,
    } = c;
    vec![
        leaf(prefix, "draws", *draws),
        leaf(prefix, "capped_draws", *capped_draws),
        leaf(prefix, "holds", *holds),
        leaf(prefix, "held_steps_sum", *held_steps_sum),
    ]
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
    }
}

fn fresh_first_half_leaves(prefix: &str, h: &FreshFirstHalfStats) -> Vec<(String, Value)> {
    let FreshFirstHalfStats {
        contested_dispatches,
        stale_drawn,
        contested_down,
        ghost_entries_from_restarted_origin,
        overtaken,
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
    ]
}

fn fresh_first(m: &mut Marks) -> FreshFirstStats {
    FreshFirstStats {
        swaps: m.int(),
        repeat_swaps: m.int(),
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
        inversions: m.int(),
        sampled_runs: m.int(),
    }
}

fn pair_order_half_leaves(prefix: &str, h: &PairOrderHalfStats) -> Vec<(String, Value)> {
    let PairOrderHalfStats {
        contests,
        inorder_draws,
        pair_entries,
        inversions,
        sampled_runs,
    } = h;
    vec![
        leaf(prefix, "contests", *contests),
        leaf(prefix, "inorder_draws", *inorder_draws),
        leaf(prefix, "pair_entries", *pair_entries),
        leaf(prefix, "inversions", *inversions),
        leaf(prefix, "sampled_runs", *sampled_runs),
    ]
}

fn pair_order(m: &mut Marks) -> PairOrderStats {
    PairOrderStats {
        corrected: m.int(),
        census: PairOrderCensusStats {
            treated: pair_order_half(m),
            control: pair_order_half(m),
        },
    }
}

fn pair_order_leaves(prefix: &str, p: &PairOrderStats) -> Vec<(String, Value)> {
    let PairOrderStats { corrected, census } = p;
    let PairOrderCensusStats { treated, control } = census;
    let mut out = vec![leaf(prefix, "corrected", *corrected)];
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
    }
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
        run_cap: _,
        crash_place: _,
        crash_phase: _,
        victim_swap: _,
        ghost_signal: _,
        fresh_first: _,
        pair_order: _,
        client_anchor: _,
        replay: _,
        timer_context: _,
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
        "recovery_weight_placebo",
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
        "run_cap",
        "crash_place",
        "crash_phase",
        "victim_swap",
        "ghost_signal",
        "fresh_first",
        "pair_order",
        "client_anchor",
        "replay",
        "timer_context",
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
    s.run_cap = run_cap(&mut m);
    s.crash_place = crash_place(&mut m);
    s.crash_phase = crash_phase(&mut m);
    s.victim_swap = victim_swap(&mut m);
    s.ghost_signal = ghost_signal(&mut m);
    s.fresh_first = fresh_first(&mut m);
    s.pair_order = pair_order(&mut m);
    s.client_anchor = client_anchor(&mut m);
    s.replay = replay(&mut m);
    s.timer_context = timer_context(&mut m);
    let mut expected = steer_authority_leaves("steer_authority", &s.steer_authority);
    expected.extend(termination_leaves("termination", &s.termination));
    expected.extend(delivery_effects_leaves(
        "delivery_effects",
        &s.delivery_effects,
    ));
    expected.extend(run_cap_leaves("run_cap", &s.run_cap));
    expected.extend(crash_place_leaves("crash_place", &s.crash_place));
    expected.extend(crash_phase_leaves("crash_phase", &s.crash_phase));
    expected.extend(victim_swap_leaves("victim_swap", &s.victim_swap));
    expected.extend(ghost_signal_leaves("ghost_signal", &s.ghost_signal));
    expected.extend(fresh_first_leaves("fresh_first", &s.fresh_first));
    expected.extend(pair_order_leaves("pair_order", &s.pair_order));
    expected.extend(client_anchor_leaves("client_anchor", &s.client_anchor));
    expected.extend(replay_leaves("replay", &s.replay));
    expected.extend(timer_context_leaves("timer_context", &s.timer_context));
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
        "delivery_effects",
        "run_cap",
        "crash_place",
        "crash_phase",
        "victim_swap",
        "ghost_signal",
        "fresh_first",
        "pair_order",
        "client_anchor",
        "replay",
        "timer_context",
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
