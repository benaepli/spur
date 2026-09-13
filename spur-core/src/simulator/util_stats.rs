//! Opt-in, process-wide utilization counters for explorer mechanisms.
//!
//! Enabled per explorer session via `ExplorerConfig::stats` and dumped by the
//! CLI to `<output_dir>/utilization.json`. Counters are observation-only: they
//! never affect scheduling, scoring, or RNG consumption. When disabled, every
//! probe is a single relaxed atomic load.

use crate::simulator::core::steer_terms::{Term, TERMS};
use crate::simulator::client_anchor;
use crate::simulator::recover_deps::RecoverDeps;
use crate::simulator::run_variant;
use crate::simulator::fresh_first;
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static ACTED_ENABLED: AtomicBool = AtomicBool::new(true);
static CRASH_CENSUS_ENABLED: AtomicBool = AtomicBool::new(false);
static STEER_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
static STEER_AUDIT_ALWAYS: AtomicBool = AtomicBool::new(false);
static MULTIPLIER_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
static QUIET_STRETCH_ENABLED: AtomicBool = AtomicBool::new(false);

static RNG_ISOLATED_RUNS: AtomicU64 = AtomicU64::new(0);
static RNG_SHARED_RUNS: AtomicU64 = AtomicU64::new(0);
static STEER_EVALUATIONS: AtomicU64 = AtomicU64::new(0);
static STEER_DIVERGENT_PICKS: AtomicU64 = AtomicU64::new(0);
static ES_CANDIDATE_MASK: AtomicU64 = AtomicU64::new(0);
static ES_RANKING_PASS: AtomicU64 = AtomicU64::new(0);
static ES_QUEUE_AUDIT: AtomicU64 = AtomicU64::new(0);
static SA_STEPS: AtomicU64 = AtomicU64::new(0);
static SA_STEPS_TOTAL: AtomicU64 = AtomicU64::new(0);
static SA_AUDITED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_EXPRESSED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_HONORED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_CONSULTED: AtomicU64 = AtomicU64::new(0);
static SA_PREFERENCE_SOURCE_ABSENT: AtomicU64 = AtomicU64::new(0);
static SA_HONORED: AtomicU64 = AtomicU64::new(0);
static SA_NO_ELIGIBLE: AtomicU64 = AtomicU64::new(0);
static SA_BLOCKED_BY_ORDER: AtomicU64 = AtomicU64::new(0);
static SA_BLOCKED_BY_TIMER_GATE: AtomicU64 = AtomicU64::new(0);
static SA_OTHER_QUEUE: AtomicU64 = AtomicU64::new(0);
static SA_SAMPLER_CHOSE_OTHER: AtomicU64 = AtomicU64::new(0);
static SR_NO_SCHEDULE_ATTEMPT: AtomicU64 = AtomicU64::new(0);
static SR_AUDIT_DISABLED: AtomicU64 = AtomicU64::new(0);
static SR_NO_WEIGHTED_PREDICATE: AtomicU64 = AtomicU64::new(0);
static SR_SINGLE_CANDIDATE: AtomicU64 = AtomicU64::new(0);
static SR_RANKING_AGREED: AtomicU64 = AtomicU64::new(0);
static SR_PREFERENCE_EXPRESSED: AtomicU64 = AtomicU64::new(0);
static PURGATORY_DELAYED_SENDS: AtomicU64 = AtomicU64::new(0);
static PURGATORY_HOLDS_DOWN_RECEIVER: AtomicU64 = AtomicU64::new(0);
static PURGATORY_HOLDS_UP_RECEIVER: AtomicU64 = AtomicU64::new(0);
static PURGATORY_PASSTHROUGH_DOWN_RECEIVER: AtomicU64 = AtomicU64::new(0);
static AOS_TAPE_WINS: AtomicU64 = AtomicU64::new(0);
static AOS_CONFIG_WINS: AtomicU64 = AtomicU64::new(0);
static DEDUP_CHECKS: AtomicU64 = AtomicU64::new(0);
static DEDUP_HITS: AtomicU64 = AtomicU64::new(0);
/// Reserved for a size-capped dedup bypass; no such path exists today, so this
/// stays 0 but is kept in the output shape for downstream consumers.
static DEDUP_SKIPPED_LARGE: AtomicU64 = AtomicU64::new(0);
// f64 sums stored as bit patterns (0u64 == 0.0f64), updated via CAS.
static FEEDBACK_TIMELINE_SCORE_SUM: AtomicU64 = AtomicU64::new(0);
static FEEDBACK_CFG_SCORE_SUM: AtomicU64 = AtomicU64::new(0);
static FEEDBACK_SCORED_RUNS: AtomicU64 = AtomicU64::new(0);
static CURRICULUM_LOWERED_RUNS: AtomicU64 = AtomicU64::new(0);
static CURRICULUM_CRASHES_SUM: AtomicU64 = AtomicU64::new(0);
static CURRICULUM_SERVERS_SUM: AtomicU64 = AtomicU64::new(0);
static CR_RUNS: AtomicU64 = AtomicU64::new(0);
static CR_CRASHES: AtomicU64 = AtomicU64::new(0);
static CR_RECOVERS: AtomicU64 = AtomicU64::new(0);
static CR_RECOVERS_WITH_INFLIGHT: AtomicU64 = AtomicU64::new(0);
static CR_HELD_AT_CRASH: AtomicU64 = AtomicU64::new(0);
static CR_DROPPED_AT_CRASH: AtomicU64 = AtomicU64::new(0);
static CR_CROSSING_DELIVERIES: AtomicU64 = AtomicU64::new(0);
static CR_RUNS_WITH_CROSSING: AtomicU64 = AtomicU64::new(0);
static CA_STEPS_WITH_CRASH_ELIGIBLE: AtomicU64 = AtomicU64::new(0);
static CA_OFFERED: AtomicU64 = AtomicU64::new(0);
static CA_CRASHES_TAKEN: AtomicU64 = AtomicU64::new(0);
static CA_APPLIED: AtomicU64 = AtomicU64::new(0);
static CA_TIMING_BIAS_EXAMINED: AtomicU64 = AtomicU64::new(0);
static CA_TIMING_BIAS_WITHHELD: AtomicU64 = AtomicU64::new(0);
static CC_DECISIONS: AtomicU64 = AtomicU64::new(0);
static CC_VICTIM_INFLIGHT: AtomicU64 = AtomicU64::new(0);
static CC_ANY_CANDIDATE_INFLIGHT: AtomicU64 = AtomicU64::new(0);
static VS_APPLIED: AtomicU64 = AtomicU64::new(0);
static VS_ACTED_ABSORBER: AtomicU64 = AtomicU64::new(0);
static VS_SAME_VICTIM: AtomicU64 = AtomicU64::new(0);
static VS_NO_ABSORBER: AtomicU64 = AtomicU64::new(0);
static VS_SKIPPED_PENDING_PAIR: AtomicU64 = AtomicU64::new(0);
static VS_VICTIM_CRASHED_HOLDS: AtomicU64 = AtomicU64::new(0);
static VS_FORCED_ONTO_ABSORBER: AtomicU64 = AtomicU64::new(0);
static GS_FIRED_RUNS: AtomicU64 = AtomicU64::new(0);
static FRAME_CALLS: AtomicU64 = AtomicU64::new(0);
static FRAME_SLOTS_BUILT: AtomicU64 = AtomicU64::new(0);
static FRAME_ENTRY_COPIES: AtomicU64 = AtomicU64::new(0);

/// Advanced by every session reset. A per-thread counter block remembers the
/// value it was activated under and is dropped rather than folded when the
/// two differ.
static STATS_GENERATION: AtomicU64 = AtomicU64::new(0);
static STATS_LOCAL_FOLDS: AtomicU64 = AtomicU64::new(0);
static STATS_LOCAL_FOLDED_INCREMENTS: AtomicU64 = AtomicU64::new(0);
static FF_SWAPS: AtomicU64 = AtomicU64::new(0);
static FF_REPEAT_SWAPS: AtomicU64 = AtomicU64::new(0);
/// How many times a ghost was displaced before it was taken: once, twice,
/// three times, four or more.
const FF_HIST_SLOTS: usize = 4;
static FF_SWAP_COUNT_HIST: [AtomicU64; FF_HIST_SLOTS] = [const { AtomicU64::new(0) }; FF_HIST_SLOTS];
/// The contested-dispatch census split by the fresh-first half: index 0 is
/// the control half, index 1 the treated half.
const FF_HALVES: usize = 2;
static FF_CONTESTED: [AtomicU64; FF_HALVES] = [const { AtomicU64::new(0) }; FF_HALVES];
static FF_STALE_DRAWN: [AtomicU64; FF_HALVES] = [const { AtomicU64::new(0) }; FF_HALVES];
static FF_CONTESTED_DOWN: [AtomicU64; FF_HALVES] = [const { AtomicU64::new(0) }; FF_HALVES];
static FF_GHOST_ENTRIES: [AtomicU64; FF_HALVES] = [const { AtomicU64::new(0) }; FF_HALVES];
static FF_OVERTAKEN: [AtomicU64; FF_HALVES] = [const { AtomicU64::new(0) }; FF_HALVES];
/// The ghost-entry census of each half split again by whether the
/// destination has itself come back from a crash in the run.
static FF_GHOST_ENTRIES_RESTARTED_DEST: [AtomicU64; FF_HALVES] =
    [const { AtomicU64::new(0) }; FF_HALVES];
static FF_OVERTAKEN_RESTARTED_DEST: [AtomicU64; FF_HALVES] =
    [const { AtomicU64::new(0) }; FF_HALVES];
static FF_SKIPPED_NEVER_RESTARTED_DEST: AtomicU64 = AtomicU64::new(0);
/// The pair-order census split by its half: index 0 is the control half,
/// index 1 the treated half.
const PO_HALVES: usize = 2;
static PO_CONTESTS: [AtomicU64; PO_HALVES] = [const { AtomicU64::new(0) }; PO_HALVES];
static PO_INORDER_DRAWS: [AtomicU64; PO_HALVES] = [const { AtomicU64::new(0) }; PO_HALVES];
static PO_PAIR_ENTRIES: [AtomicU64; PO_HALVES] = [const { AtomicU64::new(0) }; PO_HALVES];
static PO_INVERSIONS: [AtomicU64; PO_HALVES] = [const { AtomicU64::new(0) }; PO_HALVES];
static PO_SAMPLED_RUNS: [AtomicU64; PO_HALVES] = [const { AtomicU64::new(0) }; PO_HALVES];
static PO_PAIR_ENTRIES_GHOST: [AtomicU64; PO_HALVES] = [const { AtomicU64::new(0) }; PO_HALVES];
static PO_INVERSIONS_GHOST: [AtomicU64; PO_HALVES] = [const { AtomicU64::new(0) }; PO_HALVES];
static PO_CORRECTED: AtomicU64 = AtomicU64::new(0);
/// The firing of the preference on the treated half by the incarnation class
/// of the pick: index 1 is a pick whose sending incarnation is not the one
/// running now, index 0 a pick of the sender's current incarnation.
const PO_CLASSES: usize = 2;
static PO_CONTESTS_BY_CLASS: [AtomicU64; PO_CLASSES] = [const { AtomicU64::new(0) }; PO_CLASSES];
static PO_CORRECTIONS_BY_CLASS: [AtomicU64; PO_CLASSES] =
    [const { AtomicU64::new(0) }; PO_CLASSES];
static PO_FRESH_SUPPRESSED: AtomicU64 = AtomicU64::new(0);
/// The client-anchor census split by its half: index 0 is the control half,
/// index 1 the treated half.
const CAN_HALVES: usize = 2;
static CAN_RUNS: [AtomicU64; CAN_HALVES] = [const { AtomicU64::new(0) }; CAN_HALVES];
static CAN_COMPLETED_RUNS: [AtomicU64; CAN_HALVES] = [const { AtomicU64::new(0) }; CAN_HALVES];
static CAN_POPULATION: [AtomicU64; CAN_HALVES] = [const { AtomicU64::new(0) }; CAN_HALVES];
static CAN_FANOUT_WINDOWS: [AtomicU64; CAN_HALVES] = [const { AtomicU64::new(0) }; CAN_HALVES];
static CAN_POST_FAULT_INVOCATIONS: [AtomicU64; CAN_HALVES] =
    [const { AtomicU64::new(0) }; CAN_HALVES];
static CAN_IN_WINDOW_INVOCATIONS: [AtomicU64; CAN_HALVES] =
    [const { AtomicU64::new(0) }; CAN_HALVES];
static CAN_HELD: AtomicU64 = AtomicU64::new(0);
static CAN_RELEASED_EXPIRY: AtomicU64 = AtomicU64::new(0);
static CAN_RELEASED_DRY_QUEUE: AtomicU64 = AtomicU64::new(0);
/// Requests held when a treated run's first window opened: none, one, two,
/// three or more.
static CAN_HELD_AT_FIRST_FIRING: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static CAN_HELD_AT_EXIT: AtomicU64 = AtomicU64::new(0);
static CAN_RUNS_WITH_HELD_AT_EXIT: AtomicU64 = AtomicU64::new(0);
static CAN_HOLD_STEPS_SUM: AtomicU64 = AtomicU64::new(0);
/// One slot per direction of the post-crash request-timing axis, in the
/// order `client_anchor::Arm::index` gives.
const CAN_ARMS: usize = 3;
static CAN_ARM_RUNS: [AtomicU64; CAN_ARMS] = [const { AtomicU64::new(0) }; CAN_ARMS];
static CAN_RUSH_OPS: AtomicU64 = AtomicU64::new(0);
static CAN_RUSH_RECORDS_PRIORITIZED: AtomicU64 = AtomicU64::new(0);
static CAN_RUSH_WAS_PICK: AtomicU64 = AtomicU64::new(0);
static CAN_RUSH_DISPLACED: AtomicU64 = AtomicU64::new(0);
static CAN_FIRST_DELIVERY_SUM: [AtomicU64; CAN_ARMS] = [const { AtomicU64::new(0) }; CAN_ARMS];
static CAN_FIRST_DELIVERY_COUNT: [AtomicU64; CAN_ARMS] = [const { AtomicU64::new(0) }; CAN_ARMS];
/// The step of the first message entry caused by a post-fault client
/// operation, per direction the run carried, over the arm selector's
/// coin-drawn runs: runs that had one and the sum of their steps.
static CAN_FIRST_ENTRY_RUNS: [AtomicU64; run_variant::DIRECTIONS] =
    [const { AtomicU64::new(0) }; run_variant::DIRECTIONS];
static CAN_FIRST_ENTRY_STEPS_SUM: [AtomicU64; run_variant::DIRECTIONS] =
    [const { AtomicU64::new(0) }; run_variant::DIRECTIONS];
/// The per-run rewards the arm selector reads. `OvertakenGhost`,
/// `AbsorberCycle` and `CycleBeforeRequest` each train one learner; the
/// rest are read on the coin-drawn runs only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reward {
    OvertakenGhost,
    AbsorberCycle,
    MutualAbsorberCycle,
    GhostSignal,
    EitherShape,
    CycleBeforeRequest,
    ExchangeBeforeRequest,
}

impl Reward {
    pub const ALL: [Reward; 7] = [
        Reward::OvertakenGhost,
        Reward::AbsorberCycle,
        Reward::MutualAbsorberCycle,
        Reward::GhostSignal,
        Reward::EitherShape,
        Reward::CycleBeforeRequest,
        Reward::ExchangeBeforeRequest,
    ];

    fn index(self) -> usize {
        match self {
            Reward::OvertakenGhost => 0,
            Reward::AbsorberCycle => 1,
            Reward::MutualAbsorberCycle => 2,
            Reward::GhostSignal => 3,
            Reward::EitherShape => 4,
            Reward::CycleBeforeRequest => 5,
            Reward::ExchangeBeforeRequest => 6,
        }
    }
}

/// The arm selector's counters for one reward: the coin-drawn runs and
/// rewards per direction and per combination, the reward on the coin-drawn
/// runs and on the learner runs, and, for a reward a learner is trained
/// on, that learner's runs.
struct AxCounters {
    chosen_runs: AtomicU64,
    departures: AtomicU64,
    axis_leader_agreements: AtomicU64,
    axis_draws: AtomicU64,
    chosen_placed_runs: AtomicU64,
    cells: AtomicU64,
    /// Indexed by whether the run was a learner run.
    reward_runs: [AtomicU64; 2],
    reward_positive: [AtomicU64; 2],
    chosen_by_direction: [AtomicU64; run_variant::DIRECTIONS],
    /// Per axis, the sum in millionths of the pairwise probability that
    /// the highest-mean direction leads the next, over the learner runs.
    leader_margin_micro: [AtomicU64; run_variant::AXES],
    control_runs_by_direction: [AtomicU64; run_variant::DIRECTIONS],
    control_reward_positive_by_direction: [AtomicU64; run_variant::DIRECTIONS],
    /// The coin-drawn runs and rewards per direction split by campaign
    /// arm, row `arm_index + 1` of `AX_ARM_SLOTS`, flattened row-major.
    control_runs_by_arm_direction: [AtomicU64; AX_ARM_DIRECTION_CELLS],
    control_reward_positive_by_arm_direction: [AtomicU64; AX_ARM_DIRECTION_CELLS],
    chosen_by_combination: [AtomicU64; run_variant::COMBINATIONS],
    control_runs_by_combination: [AtomicU64; run_variant::COMBINATIONS],
    control_reward_positive_by_combination: [AtomicU64; run_variant::COMBINATIONS],
    /// Gauge: the cells of this reward's learner that are keyed by a
    /// configuration and shared across campaign arms.
    pooled_cells: AtomicU64,
}

impl AxCounters {
    const fn new() -> Self {
        Self {
            chosen_runs: AtomicU64::new(0),
            departures: AtomicU64::new(0),
            axis_leader_agreements: AtomicU64::new(0),
            axis_draws: AtomicU64::new(0),
            chosen_placed_runs: AtomicU64::new(0),
            cells: AtomicU64::new(0),
            pooled_cells: AtomicU64::new(0),
            reward_runs: [const { AtomicU64::new(0) }; 2],
            reward_positive: [const { AtomicU64::new(0) }; 2],
            chosen_by_direction: [const { AtomicU64::new(0) }; run_variant::DIRECTIONS],
            leader_margin_micro: [const { AtomicU64::new(0) }; run_variant::AXES],
            control_runs_by_direction: [const { AtomicU64::new(0) }; run_variant::DIRECTIONS],
            control_reward_positive_by_direction: [const { AtomicU64::new(0) };
                run_variant::DIRECTIONS],
            control_runs_by_arm_direction: [const { AtomicU64::new(0) }; AX_ARM_DIRECTION_CELLS],
            control_reward_positive_by_arm_direction: [const { AtomicU64::new(0) };
                AX_ARM_DIRECTION_CELLS],
            chosen_by_combination: [const { AtomicU64::new(0) }; run_variant::COMBINATIONS],
            control_runs_by_combination: [const { AtomicU64::new(0) }; run_variant::COMBINATIONS],
            control_reward_positive_by_combination: [const { AtomicU64::new(0) };
                run_variant::COMBINATIONS],
        }
    }
}

static AX: [AxCounters; 7] = [const { AxCounters::new() }; 7];
static AX_OBSERVATIONS: AtomicU64 = AtomicU64::new(0);
/// The first learner's reward per campaign arm over every observed run,
/// slot `arm_index + 1` so an unattributed run lands in slot zero; arms
/// past the width share the last slot.
const AX_ARM_SLOTS: usize = 8;
/// Size of a per-arm, per-direction table, row `arm_index + 1`.
const AX_ARM_DIRECTION_CELLS: usize = AX_ARM_SLOTS * run_variant::DIRECTIONS;
static AX_REWARD_RUNS_BY_ARM: [AtomicU64; AX_ARM_SLOTS] = [const { AtomicU64::new(0) }; AX_ARM_SLOTS];
static AX_REWARD_POSITIVE_BY_ARM: [AtomicU64; AX_ARM_SLOTS] =
    [const { AtomicU64::new(0) }; AX_ARM_SLOTS];

/// Number of learners the arm selector runs.
const AX_LEARNERS: usize = 3;
/// Number of equal bins the exploration share is histogrammed into.
const AX_SHARE_BINS: usize = 10;

/// The arm selector's exploration draws: every run assigned to a learner,
/// whether it came out coin-drawn or as a learner run, with the share it
/// was drawn against, by learner, by campaign arm (slot `arm_index + 1`,
/// as `AX_REWARD_RUNS_BY_ARM`) and by the share in tenths.
struct AxExploreCounters {
    draws: AtomicU64,
    coin_runs: AtomicU64,
    warmup_coin_runs: AtomicU64,
    share_micro: AtomicU64,
    margin_micro: AtomicU64,
    draws_by_learner: [AtomicU64; AX_LEARNERS],
    coin_runs_by_learner: [AtomicU64; AX_LEARNERS],
    share_micro_by_learner: [AtomicU64; AX_LEARNERS],
    draws_by_arm: [AtomicU64; AX_ARM_SLOTS],
    coin_runs_by_arm: [AtomicU64; AX_ARM_SLOTS],
    margin_micro_by_arm: [AtomicU64; AX_ARM_SLOTS],
    share_hist: [AtomicU64; AX_SHARE_BINS],
}

impl AxExploreCounters {
    const fn new() -> Self {
        Self {
            draws: AtomicU64::new(0),
            coin_runs: AtomicU64::new(0),
            warmup_coin_runs: AtomicU64::new(0),
            share_micro: AtomicU64::new(0),
            margin_micro: AtomicU64::new(0),
            draws_by_learner: [const { AtomicU64::new(0) }; AX_LEARNERS],
            coin_runs_by_learner: [const { AtomicU64::new(0) }; AX_LEARNERS],
            share_micro_by_learner: [const { AtomicU64::new(0) }; AX_LEARNERS],
            draws_by_arm: [const { AtomicU64::new(0) }; AX_ARM_SLOTS],
            coin_runs_by_arm: [const { AtomicU64::new(0) }; AX_ARM_SLOTS],
            margin_micro_by_arm: [const { AtomicU64::new(0) }; AX_ARM_SLOTS],
            share_hist: [const { AtomicU64::new(0) }; AX_SHARE_BINS],
        }
    }
}

static AX_EXPLORE: AxExploreCounters = AxExploreCounters::new();

/// The arm selector's traffic on the cells keyed by a configuration and
/// shared across campaign arms: the draws served from such a cell and,
/// per learner, the runs credited into one.
struct AxPooledCounters {
    draws: AtomicU64,
    observations_by_learner: [AtomicU64; AX_LEARNERS],
}

static AX_POOLED: AxPooledCounters = AxPooledCounters {
    draws: AtomicU64::new(0),
    observations_by_learner: [const { AtomicU64::new(0) }; AX_LEARNERS],
};
static RP_PARENTS_ADMITTED: AtomicU64 = AtomicU64::new(0);
static RP_CHILDREN: AtomicU64 = AtomicU64::new(0);
static RP_CHILDREN_PREFIX: AtomicU64 = AtomicU64::new(0);
static RP_CHILDREN_PLAN_ONLY: AtomicU64 = AtomicU64::new(0);
static RP_SLOTS_UNFILLED: AtomicU64 = AtomicU64::new(0);
static RP_PREFIX_FAITHFUL: AtomicU64 = AtomicU64::new(0);
static RP_TAPE_WORDS_SUM: AtomicU64 = AtomicU64::new(0);
static RP_CHILDREN_SIGNAL_FIRED: AtomicU64 = AtomicU64::new(0);

/// The crash census split by retarget half: index 0 is the control half,
/// index 1 the treated half.
const VS_HALVES: usize = 2;
static VS_CENSUS_CRASHES: [AtomicU64; VS_HALVES] = [const { AtomicU64::new(0) }; VS_HALVES];
static VS_CENSUS_ABSORBED: [AtomicU64; VS_HALVES] = [const { AtomicU64::new(0) }; VS_HALVES];
static VS_CENSUS_INFLIGHT: [AtomicU64; VS_HALVES] = [const { AtomicU64::new(0) }; VS_HALVES];
static RW_CLOSED: AtomicU64 = AtomicU64::new(0);
static RW_WIDTH_SUM: AtomicU64 = AtomicU64::new(0);
static RW_MAX: AtomicU64 = AtomicU64::new(0);
static RW_UNCLOSED: AtomicU64 = AtomicU64::new(0);
static OH3_RUNS: AtomicU64 = AtomicU64::new(0);
static OH3_WITH_H3: AtomicU64 = AtomicU64::new(0);
static OH3_WITH_OVERLAP: AtomicU64 = AtomicU64::new(0);
static PFO_PAIRS_SEEN: AtomicU64 = AtomicU64::new(0);
static PFO_EDGES_ADDED: AtomicU64 = AtomicU64::new(0);
static PFO_OPS_AFTER_LAST_RECOVER: AtomicU64 = AtomicU64::new(0);
static PD_RECOVER_EDGES_DROPPED: AtomicU64 = AtomicU64::new(0);
static NOVELTY_ABLATED_RUNS: AtomicU64 = AtomicU64::new(0);
static MA_DECISIONS: AtomicU64 = AtomicU64::new(0);
static MA_CONTESTED_DECISIONS: AtomicU64 = AtomicU64::new(0);
static MA_QUICK_FIRE_OFFERS: AtomicU64 = AtomicU64::new(0);
static MA_QUICK_FIRE_DECISIONS: AtomicU64 = AtomicU64::new(0);
static MA_FLIPPED_CONFIGURED: AtomicU64 = AtomicU64::new(0);
static MA_CONFIGURED_SUM: AtomicU64 = AtomicU64::new(0);
static RWP_ENABLED: AtomicBool = AtomicBool::new(false);
static RWP_DECISIONS: AtomicU64 = AtomicU64::new(0);
static RWP_EVALUATED: AtomicU64 = AtomicU64::new(0);
static RWP_PRESENT: AtomicU64 = AtomicU64::new(0);
static RWP_CONTESTED: AtomicU64 = AtomicU64::new(0);
static RWP_WON: AtomicU64 = AtomicU64::new(0);
static RWP_FLIPPED: AtomicU64 = AtomicU64::new(0);

/// Multiplier magnitudes the authority probe ranks candidates under. Index 0 is
/// the identity weighting, which every other entry is compared against, so its
/// own flip count is zero by construction and reads as a self-check.
pub const MULTIPLIER_SWEEP: [f64; 5] = [1.0, 3.0, 10.0, 100.0, 1000.0];

static MA_FLIPPED: [AtomicU64; MULTIPLIER_SWEEP.len()] =
    [const { AtomicU64::new(0) }; MULTIPLIER_SWEEP.len()];

/// Recovery-window widths are tallied into a histogram so percentiles can be
/// read without keeping every sample. Widths at or above the cap fold into the
/// last slot, which the percentile reader reports as the cap itself.
const RW_WIDTH_CAP: usize = 1024;
static RW_WIDTHS: Mutex<[u64; RW_WIDTH_CAP + 1]> = Mutex::new([0; RW_WIDTH_CAP + 1]);

/// Undelivered outbound messages the node a crash lands on was holding, as a
/// histogram index. The last slot folds in every larger count, so the shape is
/// "none, one, two, more" rather than a full distribution.
const CC_INFLIGHT_SLOTS: usize = 4;
static CC_INFLIGHT: [AtomicU64; CC_INFLIGHT_SLOTS] =
    [const { AtomicU64::new(0) }; CC_INFLIGHT_SLOTS];

/// Crash and recover events applied in one run, clamped, as the bucket index of
/// the cross-tab. This is the only ordinal measure of how much fault activity a
/// run saw that the simulator itself observes.
const FAULT_EVENT_BUCKETS: usize = 9;
static OH3_BY_FAULT_EVENTS: Mutex<[[u64; 3]; FAULT_EVENT_BUCKETS]> =
    Mutex::new([[0; 3]; FAULT_EVENT_BUCKETS]);

/// Delivery-effect counters, laid out as (total, acted) pairs. Index 0 is every
/// delivery, index 1 every delivery that carried at least one bias, and the
/// remaining indices split that by which bias the message carried (a message
/// can carry more than one, so those three do not sum to index 1).
const DELIVERY_ALL: usize = 0;
const DELIVERY_BIASED: usize = 1;
const DELIVERY_DELAYED: usize = 2;
const DELIVERY_SENDER_RESTARTED: usize = 3;
const DELIVERY_RECEIVER_RESTARTED: usize = 4;
const DELIVERY_BUCKETS: usize = 5;

static DELIVERIES: [AtomicU64; DELIVERY_BUCKETS] =
    [const { AtomicU64::new(0) }; DELIVERY_BUCKETS];
static DELIVERIES_ACTED: [AtomicU64; DELIVERY_BUCKETS] =
    [const { AtomicU64::new(0) }; DELIVERY_BUCKETS];

/// The same (total, acted) pairs again, split by how far the receiving node
/// had progressed past its own most recent restart when the message reached
/// it, counted in handler entries. A receiver that has not restarted counts
/// from the start of the run, so its distance says where in the run the
/// delivery landed rather than where in a recovery it landed.
const ACCEPT_DIST_BUCKETS: usize = 7;
const ACCEPT_DIST_LABELS: [&str; ACCEPT_DIST_BUCKETS] =
    ["0", "1", "2", "3-4", "5-8", "9-16", "17+"];

/// Delivery populations the distance census is kept for. A message that
/// carries both restarts is counted in both rows.
const ACCEPT_PATH_ALL: usize = 0;
const ACCEPT_PATH_SENDER_RESTARTED: usize = 1;
const ACCEPT_PATH_RECEIVER_RESTARTED: usize = 2;
const ACCEPT_PATHS: usize = 3;

static ACCEPT_DIST: [[AtomicU64; ACCEPT_DIST_BUCKETS]; ACCEPT_PATHS] =
    [const { [const { AtomicU64::new(0) }; ACCEPT_DIST_BUCKETS] }; ACCEPT_PATHS];
static ACCEPT_DIST_ACTED: [[AtomicU64; ACCEPT_DIST_BUCKETS]; ACCEPT_PATHS] =
    [const { [const { AtomicU64::new(0) }; ACCEPT_DIST_BUCKETS] }; ACCEPT_PATHS];
static ACCEPT_DIST_ENABLED: AtomicBool = AtomicBool::new(false);

static TERMINATION: Mutex<TerminationStats> = Mutex::new(TerminationStats::new());

static PLAN_DEPS_CELLS: Mutex<[PlanDepsDensitySplit; 2]> =
    Mutex::new([PlanDepsDensitySplit::new(); 2]);

static PREFIX_EXTENSION: Mutex<PrefixExtensionStats> = Mutex::new(PrefixExtensionStats::new());
static PREFIX_EXTENSION_ENABLED: AtomicBool = AtomicBool::new(false);

static QUIET_STRETCH: Mutex<QuietStretchState> = Mutex::new(QuietStretchState::new());

/// Per-run quiet-stretch rows a session keeps before it stops adding them. A
/// row is one finished run, so the cap bounds the size of the output rather
/// than the number of runs a session may do.
const QUIET_PER_RUN_CAP: usize = 500_000;

/// Per-term counters, laid out as `TERM_*` rows by `Term::index`: how many
/// within-queue candidates were scored with the term true (`evaluated`),
/// how many selections had such a candidate at all (`present`) and among
/// more than one eligible candidate (`contested`), how many selections
/// chose one (`won`), how many of those choices differ from what the score
/// without predicate weights would have chosen (`flipped`), and how many
/// chosen candidates were measured for their effect (`measured`) and had
/// one (`acted`).
const TERM_EVALUATED: usize = 0;
const TERM_PRESENT: usize = 1;
const TERM_CONTESTED: usize = 2;
const TERM_WON: usize = 3;
const TERM_FLIPPED: usize = 4;
const TERM_MEASURED: usize = 5;
const TERM_ACTED: usize = 6;
const TERM_COUNTERS: usize = 7;
static TERM: [[AtomicU64; TERM_COUNTERS]; TERMS] =
    [const { [const { AtomicU64::new(0) }; TERM_COUNTERS] }; TERMS];
static TERM_DECISIONS: AtomicU64 = AtomicU64::new(0);
static TERM_AUTHORITY_DRAWS: AtomicU64 = AtomicU64::new(0);
static TERM_AUTHORITY_ROUTED: AtomicU64 = AtomicU64::new(0);

/// Log2 buckets of eligible-candidate and audited-candidate counts per
/// selection: 0, 1, 2, 3-4, 5-8, ... up to 2^14 and above.
const HIST_BUCKETS: usize = 16;
static ELIGIBLE_HIST: [AtomicU64; HIST_BUCKETS] = [const { AtomicU64::new(0) }; HIST_BUCKETS];
static CANDIDATES_HIST: [AtomicU64; HIST_BUCKETS] = [const { AtomicU64::new(0) }; HIST_BUCKETS];

fn hist_bucket(n: usize) -> usize {
    match n {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => ((usize::BITS - (n - 1).leading_zeros()) as usize + 1).min(HIST_BUCKETS - 1),
    }
}

/// Context of one timer firing: the vertex the woken record resumes at (so a
/// spec's timer handlers are told apart without naming them), whether a
/// delivery to the node was pending, the node's incarnation and how many
/// firings at that vertex on that node had changed nothing before this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct TimerKey {
    pub vertex: usize,
    pub inflight: bool,
    /// 0, 1, or 2 for two or more recoveries.
    pub incarnation: u8,
    /// 0 for none, 1 for 1-2, 2 for 3-7, 3 for 8 or more.
    pub inert_streak: u8,
}

impl TimerKey {
    pub fn new(vertex: usize, inflight: bool, incarnation: u32, inert_streak: u32) -> Self {
        Self {
            vertex,
            inflight,
            incarnation: incarnation.min(2) as u8,
            inert_streak: match inert_streak {
                0 => 0,
                1..=2 => 1,
                3..=7 => 2,
                _ => 3,
            },
        }
    }
}

/// Keys are a small product of vertices and buckets; the cap only guards
/// against a spec with an unexpected number of timer resume points.
const TIMER_KEY_CAP: usize = 4096;
static TIMER_EFFECTS: LazyLock<Mutex<HashMap<TimerKey, (u64, u64)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static TIMERS_FIRED: AtomicU64 = AtomicU64::new(0);
static TIMERS_ACTED: AtomicU64 = AtomicU64::new(0);
static TIMERS_INFLIGHT_FIRED: AtomicU64 = AtomicU64::new(0);
static TIMERS_INFLIGHT_ACTED: AtomicU64 = AtomicU64::new(0);

/// One slot per `TimerKey::inert_streak` bucket. The per-key table is a list
/// and readers that difference the snapshot keep only integer leaves, so the
/// same split is also kept as named counters.
const STREAK_BUCKETS: usize = 4;
static TIMER_STREAK_FIRED: [AtomicU64; STREAK_BUCKETS] =
    [const { AtomicU64::new(0) }; STREAK_BUCKETS];
static TIMER_STREAK_ACTED: [AtomicU64; STREAK_BUCKETS] =
    [const { AtomicU64::new(0) }; STREAK_BUCKETS];

static TIMER_STEER_EVALUATED: AtomicU64 = AtomicU64::new(0);
static TIMER_STEER_RAISED: AtomicU64 = AtomicU64::new(0);
static TIMER_STEER_LOWERED: AtomicU64 = AtomicU64::new(0);

static RUN_CAP_PROBES: AtomicU64 = AtomicU64::new(0);
static RUN_CAP_PROBE_COMPLETIONS: AtomicU64 = AtomicU64::new(0);
static RUN_CAP_OVER_CAP_COMPLETIONS: AtomicU64 = AtomicU64::new(0);
static RUN_CAP_CAP_RECOMPUTES: AtomicU64 = AtomicU64::new(0);
static RUN_CAP_SCOPES_LEARNED: AtomicU64 = AtomicU64::new(0);
static RUN_CAP_CURRENT_CAP_MAX_SCOPE: AtomicU64 = AtomicU64::new(0);

static SC_STOPS: AtomicU64 = AtomicU64::new(0);
static SC_TREATED_RUNS: AtomicU64 = AtomicU64::new(0);
static SC_UNTREATED_RUNS: AtomicU64 = AtomicU64::new(0);
static SC_STEPS_SAVED_SUM: AtomicU64 = AtomicU64::new(0);
static SC_SUSPENDED_STEPS_SUM: AtomicU64 = AtomicU64::new(0);
static SC_MARK_ROWS: AtomicU64 = AtomicU64::new(0);
static SC_MARK_ACTED_DELIVERIES: AtomicU64 = AtomicU64::new(0);
static SC_MARK_ACTED_TIMERS: AtomicU64 = AtomicU64::new(0);
static SC_MARK_RELEASES: AtomicU64 = AtomicU64::new(0);
static SC_PROBES_KEYED: AtomicU64 = AtomicU64::new(0);
static SC_PROBE_OVER_CAP_COMPLETIONS: AtomicU64 = AtomicU64::new(0);
static SC_SCOPES_LEARNED: AtomicU64 = AtomicU64::new(0);
static SC_CAP_MAX_SCOPE: AtomicU64 = AtomicU64::new(0);
static SC_UNTREATED_RUNS_CAPPED: AtomicU64 = AtomicU64::new(0);
static SC_UNTREATED_OVER_CAP_RUNS: AtomicU64 = AtomicU64::new(0);
static SC_UNTREATED_ROWS_DROPPED: AtomicU64 = AtomicU64::new(0);
static SC_UNTREATED_GAP_HIST: [AtomicU64; HIST_BUCKETS] =
    [const { AtomicU64::new(0) }; HIST_BUCKETS];

static SR_RELEASES: AtomicU64 = AtomicU64::new(0);
static SR_OPS_SETTLED: AtomicU64 = AtomicU64::new(0);
static SR_DEPENDENTS_CLIENT: AtomicU64 = AtomicU64::new(0);
static SR_DEPENDENTS_FAULT: AtomicU64 = AtomicU64::new(0);
static SR_DEPENDENTS_OTHER: AtomicU64 = AtomicU64::new(0);
static SR_LATE_RESPONSES: AtomicU64 = AtomicU64::new(0);
static SR_PLAN_COMPLETED_AFTER_RELEASE: AtomicU64 = AtomicU64::new(0);
static SR_SECOND_STALL_STOPS: AtomicU64 = AtomicU64::new(0);
static SR_STEPS_AFTER_RELEASE_SUM: AtomicU64 = AtomicU64::new(0);
static SR_STALLS_WITHOUT_OPS: AtomicU64 = AtomicU64::new(0);
static SR_RELEASE_RUNS: AtomicU64 = AtomicU64::new(0);
static SR_RELEASE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static SR_RELEASE_PLAN_COMPLETE: AtomicU64 = AtomicU64::new(0);
static SR_CUT_RUNS: AtomicU64 = AtomicU64::new(0);
static SR_CUT_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static SR_CUT_PLAN_COMPLETE: AtomicU64 = AtomicU64::new(0);

static CRASH_PLACE_DRAWS: AtomicU64 = AtomicU64::new(0);
static CRASH_PLACE_CAPPED_DRAWS: AtomicU64 = AtomicU64::new(0);
static CRASH_PLACE_HOLDS: AtomicU64 = AtomicU64::new(0);
static CRASH_PLACE_HELD_STEPS_SUM: AtomicU64 = AtomicU64::new(0);

static GR_ARMED: AtomicU64 = AtomicU64::new(0);
static GR_FIRED: AtomicU64 = AtomicU64::new(0);
static GR_FIRED_NOTHING_HELD: AtomicU64 = AtomicU64::new(0);
static GR_EXPIRED: AtomicU64 = AtomicU64::new(0);
static GR_SUPERSEDED: AtomicU64 = AtomicU64::new(0);
static GR_STEPS_FROM_RESTART_SUM: AtomicU64 = AtomicU64::new(0);
static GR_RELEASED_CRASHES: AtomicU64 = AtomicU64::new(0);
static GR_RESTARTS_WITH_HELD_CRASH: AtomicU64 = AtomicU64::new(0);
static GR_LAG_SAMPLES: AtomicU64 = AtomicU64::new(0);
static GR_LAG_P50: AtomicU64 = AtomicU64::new(0);
static GR_LAG_P75: AtomicU64 = AtomicU64::new(0);
static GR_LAG_P90: AtomicU64 = AtomicU64::new(0);
static GR_SCOPES_ENGAGED: AtomicU64 = AtomicU64::new(0);
static GR_SINGLE_OWN_CRASH: AtomicU64 = AtomicU64::new(0);
static GR_SINGLE_VIA_RANKING: AtomicU64 = AtomicU64::new(0);
static GR_SINGLE_FORCED: AtomicU64 = AtomicU64::new(0);
static GR_SINGLE_NO_CASE: AtomicU64 = AtomicU64::new(0);
/// One column per placed cell of the ghost release, indexed by
/// `GhostReleaseCell::index`.
const GR_CELLS: usize = 3;
static GR_CELL_RUNS: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_STEPS_USED_SUM: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_CRASHES_APPLIED: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_LATER_CRASHES: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_WITHIN_3: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_FIRED_APPLIED: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_FIRED_WITHIN_3: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_FIRED_ON_GHOST_NODE: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
static GR_CELL_DOUBLE_CRASH: [AtomicU64; GR_CELLS] = [const { AtomicU64::new(0) }; GR_CELLS];
/// Indexed by cell, then by whether the run's crashes wait for a fan-out
/// phase.
static GR_CELL_FIRED_APPLIED_PHASE: [[AtomicU64; 2]; GR_CELLS] =
    [const { [const { AtomicU64::new(0) }; 2] }; GR_CELLS];
static GR_CELL_FIRED_WITHIN_3_PHASE: [[AtomicU64; 2]; GR_CELLS] =
    [const { [const { AtomicU64::new(0) }; 2] }; GR_CELLS];
/// Indexed by cell, then by whether the run retargets its crashes.
static GR_CELL_FIRED_APPLIED_RETARGET: [[AtomicU64; 2]; GR_CELLS] =
    [const { [const { AtomicU64::new(0) }; 2] }; GR_CELLS];
static GR_CELL_FIRED_ON_GHOST_NODE_RETARGET: [[AtomicU64; 2]; GR_CELLS] =
    [const { [const { AtomicU64::new(0) }; 2] }; GR_CELLS];
static GR_CELL_FIRED_INFLIGHT: [[AtomicU64; CC_INFLIGHT_SLOTS]; GR_CELLS] =
    [const { [const { AtomicU64::new(0) }; CC_INFLIGHT_SLOTS] }; GR_CELLS];

/// One column per arm of the fan-out anchor, indexed by `CrashPhaseArm`.
const CP_ARMS: usize = 3;
static CP_RUNS: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_ARMED: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_ON_CONDITION: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_EXPIRED: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_WAIT_STEPS_SUM: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_RELEASE_DECISIONS: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_RELEASE_INFLIGHT: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_EXPIRED_INFLIGHT: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_APPLY_DECISIONS: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_APPLY_INFLIGHT: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_CRASHES_APPLIED: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_INFLIGHT: [[AtomicU64; CC_INFLIGHT_SLOTS]; CP_ARMS] =
    [const { [const { AtomicU64::new(0) }; CC_INFLIGHT_SLOTS] }; CP_ARMS];
/// Crashes whose landing node differs from the planned victim, per arm.
static CP_MOVED_APPLIED: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_MOVED_APPLY_DECISIONS: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_MOVED_APPLY_INFLIGHT: [AtomicU64; CP_ARMS] = [const { AtomicU64::new(0) }; CP_ARMS];
static CP_MOVED_INFLIGHT: [[AtomicU64; CC_INFLIGHT_SLOTS]; CP_ARMS] =
    [const { [const { AtomicU64::new(0) }; CC_INFLIGHT_SLOTS] }; CP_ARMS];
/// Once-per-crash events of phase reads on the node a crash would land on,
/// indexed by `CrashPhaseLanding`.
const CP_LANDING_EVENTS: usize = 4;
static CP_LANDING: [AtomicU64; CP_LANDING_EVENTS] =
    [const { AtomicU64::new(0) }; CP_LANDING_EVENTS];

static TIMER_CONTEXT_PROBE_FIRINGS: AtomicU64 = AtomicU64::new(0);
static TIMER_CONTEXT_PROBE_ACTED: AtomicU64 = AtomicU64::new(0);
static TIMER_CONTEXT_BIASED_STEPS: AtomicU64 = AtomicU64::new(0);
static TIMER_CONTEXT_BIASED_STEPS_PROMOTED: AtomicU64 = AtomicU64::new(0);
static TIMER_CONTEXT_BIASED_STEPS_SUPPRESSED: AtomicU64 = AtomicU64::new(0);
static TIMER_CONTEXT_STEPS_EXCLUDED_SELECTOR: AtomicU64 = AtomicU64::new(0);
static TIMER_CONTEXT_CELLS_ENGAGED: AtomicU64 = AtomicU64::new(0);
static TIMELINE_KEYS: Mutex<TimelineKeyGrowth> = Mutex::new(TimelineKeyGrowth::new());

/// Runs per point on the timeline-key growth curve, and the most points a
/// session will report. Once the cap is reached every later run folds into the
/// final point, so the curve keeps its shape at the head where growth is fast
/// and stays bounded for a long session.
const TIMELINE_BUCKET_RUNS: u64 = 100;
const TIMELINE_MAX_BUCKETS: usize = 512;

/// Enable or disable recording for this explorer session. Enabling resets all
/// counters so repeated sessions in one process don't bleed into each other.
pub fn set_enabled(on: bool) {
    if on {
        STATS_GENERATION.fetch_add(1, Ordering::Relaxed);
        fold_run_counters();
        for c in [
            &STEER_EVALUATIONS,
            &STEER_DIVERGENT_PICKS,
            &SA_STEPS,
            &SA_STEPS_TOTAL,
            &SA_AUDITED,
            &SA_PREFERENCE_EXPRESSED,
            &SA_PREFERENCE_HONORED,
            &SA_PREFERENCE_CONSULTED,
            &SA_PREFERENCE_SOURCE_ABSENT,
            &SA_HONORED,
            &SA_NO_ELIGIBLE,
            &SA_BLOCKED_BY_ORDER,
            &SA_BLOCKED_BY_TIMER_GATE,
            &SA_OTHER_QUEUE,
            &SA_SAMPLER_CHOSE_OTHER,
            &SR_NO_SCHEDULE_ATTEMPT,
            &SR_AUDIT_DISABLED,
            &SR_NO_WEIGHTED_PREDICATE,
            &SR_SINGLE_CANDIDATE,
            &SR_RANKING_AGREED,
            &SR_PREFERENCE_EXPRESSED,
            &PURGATORY_DELAYED_SENDS,
            &PURGATORY_HOLDS_DOWN_RECEIVER,
            &PURGATORY_HOLDS_UP_RECEIVER,
            &PURGATORY_PASSTHROUGH_DOWN_RECEIVER,
            &AOS_TAPE_WINS,
            &AOS_CONFIG_WINS,
            &DEDUP_CHECKS,
            &DEDUP_HITS,
            &DEDUP_SKIPPED_LARGE,
            &FEEDBACK_TIMELINE_SCORE_SUM,
            &FEEDBACK_CFG_SCORE_SUM,
            &FEEDBACK_SCORED_RUNS,
            &CURRICULUM_LOWERED_RUNS,
            &CURRICULUM_CRASHES_SUM,
            &CURRICULUM_SERVERS_SUM,
            &CR_RUNS,
            &CR_CRASHES,
            &CR_RECOVERS,
            &CR_RECOVERS_WITH_INFLIGHT,
            &CR_HELD_AT_CRASH,
            &CR_DROPPED_AT_CRASH,
            &CR_CROSSING_DELIVERIES,
            &CR_RUNS_WITH_CROSSING,
            &CA_STEPS_WITH_CRASH_ELIGIBLE,
            &CA_OFFERED,
            &CA_CRASHES_TAKEN,
            &CA_APPLIED,
            &CA_TIMING_BIAS_EXAMINED,
            &CA_TIMING_BIAS_WITHHELD,
            &CC_DECISIONS,
            &CC_VICTIM_INFLIGHT,
            &CC_ANY_CANDIDATE_INFLIGHT,
            &VS_APPLIED,
            &VS_ACTED_ABSORBER,
            &VS_SAME_VICTIM,
            &VS_NO_ABSORBER,
            &VS_SKIPPED_PENDING_PAIR,
            &VS_VICTIM_CRASHED_HOLDS,
            &VS_FORCED_ONTO_ABSORBER,
            &GS_FIRED_RUNS,
            &FF_SWAPS,
            &FF_REPEAT_SWAPS,
            &FF_SKIPPED_NEVER_RESTARTED_DEST,
            &PO_CORRECTED,
            &PO_FRESH_SUPPRESSED,
            &CAN_HELD,
            &CAN_RELEASED_EXPIRY,
            &CAN_RELEASED_DRY_QUEUE,
            &CAN_HELD_AT_EXIT,
            &CAN_RUNS_WITH_HELD_AT_EXIT,
            &CAN_HOLD_STEPS_SUM,
            &CAN_RUSH_OPS,
            &CAN_RUSH_RECORDS_PRIORITIZED,
            &CAN_RUSH_WAS_PICK,
            &CAN_RUSH_DISPLACED,
            &RW_CLOSED,
            &RW_WIDTH_SUM,
            &RW_MAX,
            &RW_UNCLOSED,
            &OH3_RUNS,
            &OH3_WITH_H3,
            &OH3_WITH_OVERLAP,
            &PFO_PAIRS_SEEN,
            &PFO_EDGES_ADDED,
            &PFO_OPS_AFTER_LAST_RECOVER,
            &PD_RECOVER_EDGES_DROPPED,
            &NOVELTY_ABLATED_RUNS,
            &MA_DECISIONS,
            &MA_CONTESTED_DECISIONS,
            &MA_QUICK_FIRE_OFFERS,
            &MA_QUICK_FIRE_DECISIONS,
            &MA_FLIPPED_CONFIGURED,
            &MA_CONFIGURED_SUM,
            &RWP_DECISIONS,
            &RWP_EVALUATED,
            &RWP_PRESENT,
            &RWP_CONTESTED,
            &RWP_WON,
            &RWP_FLIPPED,
            &STATS_LOCAL_FOLDS,
            &STATS_LOCAL_FOLDED_INCREMENTS,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for c in DELIVERIES
            .iter()
            .chain(DELIVERIES_ACTED.iter())
            .chain(MA_FLIPPED.iter())
            .chain(CC_INFLIGHT.iter())
            .chain(VS_CENSUS_CRASHES.iter())
            .chain(VS_CENSUS_ABSORBED.iter())
            .chain(VS_CENSUS_INFLIGHT.iter())
            .chain(FF_SWAP_COUNT_HIST.iter())
            .chain(FF_CONTESTED.iter())
            .chain(FF_STALE_DRAWN.iter())
            .chain(FF_CONTESTED_DOWN.iter())
            .chain(FF_GHOST_ENTRIES.iter())
            .chain(FF_OVERTAKEN.iter())
            .chain(FF_GHOST_ENTRIES_RESTARTED_DEST.iter())
            .chain(FF_OVERTAKEN_RESTARTED_DEST.iter())
            .chain(PO_CONTESTS.iter())
            .chain(PO_INORDER_DRAWS.iter())
            .chain(PO_PAIR_ENTRIES.iter())
            .chain(PO_INVERSIONS.iter())
            .chain(PO_SAMPLED_RUNS.iter())
            .chain(PO_PAIR_ENTRIES_GHOST.iter())
            .chain(PO_INVERSIONS_GHOST.iter())
            .chain(PO_CONTESTS_BY_CLASS.iter())
            .chain(PO_CORRECTIONS_BY_CLASS.iter())
            .chain(CAN_RUNS.iter())
            .chain(CAN_COMPLETED_RUNS.iter())
            .chain(CAN_POPULATION.iter())
            .chain(CAN_FANOUT_WINDOWS.iter())
            .chain(CAN_POST_FAULT_INVOCATIONS.iter())
            .chain(CAN_IN_WINDOW_INVOCATIONS.iter())
            .chain(CAN_HELD_AT_FIRST_FIRING.iter())
            .chain(CAN_ARM_RUNS.iter())
            .chain(CAN_FIRST_DELIVERY_SUM.iter())
            .chain(CAN_FIRST_DELIVERY_COUNT.iter())
            .chain(ACCEPT_DIST.iter().flatten())
            .chain(ACCEPT_DIST_ACTED.iter().flatten())
        {
            c.store(0, Ordering::Relaxed);
        }
        if let Ok(mut w) = RW_WIDTHS.lock() {
            *w = [0; RW_WIDTH_CAP + 1];
        }
        if let Ok(mut x) = OH3_BY_FAULT_EVENTS.lock() {
            *x = [[0; 3]; FAULT_EVENT_BUCKETS];
        }
        if let Ok(mut t) = TERMINATION.lock() {
            *t = TerminationStats::new();
        }
        if let Ok(mut c) = PLAN_DEPS_CELLS.lock() {
            *c = [PlanDepsDensitySplit::new(); 2];
        }
        PLAN_DEPS_RUN.with(|r| r.set(None));
        if let Ok(mut p) = PREFIX_EXTENSION.lock() {
            *p = PrefixExtensionStats::new();
        }
        if let Ok(mut q) = QUIET_STRETCH.lock() {
            *q = QuietStretchState::new();
        }
        if let Ok(mut g) = TIMELINE_KEYS.lock() {
            *g = TimelineKeyGrowth::new();
        }
        for c in [
            &TIMERS_FIRED,
            &TIMERS_ACTED,
            &TIMERS_INFLIGHT_FIRED,
            &TIMERS_INFLIGHT_ACTED,
            &TIMER_STEER_EVALUATED,
            &TIMER_STEER_RAISED,
            &TIMER_STEER_LOWERED,
            &RUN_CAP_PROBES,
            &RUN_CAP_PROBE_COMPLETIONS,
            &RUN_CAP_OVER_CAP_COMPLETIONS,
            &RUN_CAP_CAP_RECOMPUTES,
            &RUN_CAP_SCOPES_LEARNED,
            &RUN_CAP_CURRENT_CAP_MAX_SCOPE,
            &SC_STOPS,
            &SC_TREATED_RUNS,
            &SC_UNTREATED_RUNS,
            &SC_STEPS_SAVED_SUM,
            &SC_SUSPENDED_STEPS_SUM,
            &SC_MARK_ROWS,
            &SC_MARK_ACTED_DELIVERIES,
            &SC_MARK_ACTED_TIMERS,
            &SC_MARK_RELEASES,
            &SC_PROBES_KEYED,
            &SC_PROBE_OVER_CAP_COMPLETIONS,
            &SC_SCOPES_LEARNED,
            &SC_CAP_MAX_SCOPE,
            &SC_UNTREATED_RUNS_CAPPED,
            &SC_UNTREATED_OVER_CAP_RUNS,
            &SC_UNTREATED_ROWS_DROPPED,
            &SR_RELEASES,
            &SR_OPS_SETTLED,
            &SR_DEPENDENTS_CLIENT,
            &SR_DEPENDENTS_FAULT,
            &SR_DEPENDENTS_OTHER,
            &SR_LATE_RESPONSES,
            &SR_PLAN_COMPLETED_AFTER_RELEASE,
            &SR_SECOND_STALL_STOPS,
            &SR_STEPS_AFTER_RELEASE_SUM,
            &SR_STALLS_WITHOUT_OPS,
            &SR_RELEASE_RUNS,
            &SR_RELEASE_INVOCATIONS,
            &SR_RELEASE_PLAN_COMPLETE,
            &SR_CUT_RUNS,
            &SR_CUT_INVOCATIONS,
            &SR_CUT_PLAN_COMPLETE,
            &CRASH_PLACE_DRAWS,
            &CRASH_PLACE_CAPPED_DRAWS,
            &CRASH_PLACE_HOLDS,
            &CRASH_PLACE_HELD_STEPS_SUM,
            &GR_ARMED,
            &GR_FIRED,
            &GR_FIRED_NOTHING_HELD,
            &GR_EXPIRED,
            &GR_SUPERSEDED,
            &GR_STEPS_FROM_RESTART_SUM,
            &GR_RELEASED_CRASHES,
            &GR_RESTARTS_WITH_HELD_CRASH,
            &GR_LAG_SAMPLES,
            &GR_LAG_P50,
            &GR_LAG_P75,
            &GR_LAG_P90,
            &GR_SCOPES_ENGAGED,
            &GR_SINGLE_OWN_CRASH,
            &GR_SINGLE_VIA_RANKING,
            &GR_SINGLE_FORCED,
            &GR_SINGLE_NO_CASE,
            &TIMER_CONTEXT_PROBE_FIRINGS,
            &TIMER_CONTEXT_PROBE_ACTED,
            &TIMER_CONTEXT_BIASED_STEPS,
            &TIMER_CONTEXT_BIASED_STEPS_PROMOTED,
            &TIMER_CONTEXT_BIASED_STEPS_SUPPRESSED,
            &TIMER_CONTEXT_STEPS_EXCLUDED_SELECTOR,
            &TIMER_CONTEXT_CELLS_ENGAGED,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for c in TIMER_STREAK_FIRED
            .iter()
            .chain(TIMER_STREAK_ACTED.iter())
            .chain(SC_UNTREATED_GAP_HIST.iter())
        {
            c.store(0, Ordering::Relaxed);
        }
        for c in CP_RUNS
            .iter()
            .chain(CP_ARMED.iter())
            .chain(CP_ON_CONDITION.iter())
            .chain(CP_EXPIRED.iter())
            .chain(CP_WAIT_STEPS_SUM.iter())
            .chain(CP_RELEASE_DECISIONS.iter())
            .chain(CP_RELEASE_INFLIGHT.iter())
            .chain(CP_EXPIRED_INFLIGHT.iter())
            .chain(CP_APPLY_DECISIONS.iter())
            .chain(CP_APPLY_INFLIGHT.iter())
            .chain(CP_CRASHES_APPLIED.iter())
            .chain(CP_INFLIGHT.iter().flatten())
            .chain(GR_CELL_RUNS.iter())
            .chain(GR_CELL_STEPS_USED_SUM.iter())
            .chain(GR_CELL_CRASHES_APPLIED.iter())
            .chain(GR_CELL_LATER_CRASHES.iter())
            .chain(GR_CELL_WITHIN_3.iter())
            .chain(GR_CELL_FIRED_APPLIED.iter())
            .chain(GR_CELL_FIRED_WITHIN_3.iter())
            .chain(GR_CELL_FIRED_ON_GHOST_NODE.iter())
            .chain(GR_CELL_DOUBLE_CRASH.iter())
            .chain(GR_CELL_FIRED_APPLIED_PHASE.iter().flatten())
            .chain(GR_CELL_FIRED_WITHIN_3_PHASE.iter().flatten())
            .chain(GR_CELL_FIRED_APPLIED_RETARGET.iter().flatten())
            .chain(GR_CELL_FIRED_ON_GHOST_NODE_RETARGET.iter().flatten())
            .chain(GR_CELL_FIRED_INFLIGHT.iter().flatten())
        {
            c.store(0, Ordering::Relaxed);
        }
        if let Ok(mut t) = TIMER_EFFECTS.lock() {
            t.clear();
        }
        for row in TERM.iter() {
            for c in row {
                c.store(0, Ordering::Relaxed);
            }
        }
        for c in [&TERM_DECISIONS, &TERM_AUTHORITY_DRAWS, &TERM_AUTHORITY_ROUTED] {
            c.store(0, Ordering::Relaxed);
        }
        for c in [&ES_CANDIDATE_MASK, &ES_RANKING_PASS, &ES_QUEUE_AUDIT] {
            c.store(0, Ordering::Relaxed);
        }
        for c in ELIGIBLE_HIST.iter().chain(CANDIDATES_HIST.iter()) {
            c.store(0, Ordering::Relaxed);
        }
    }
    ENABLED.store(on, Ordering::Relaxed);
}

/// One within-queue selection was scored with terms. `present` and
/// `chosen` are term masks over the eligible candidates and the chosen one;
/// `evaluated` counts, per term, the sampled candidates the term was true
/// of; `flipped` says whether the choice differs from the one the score
/// without predicate weights would have made.
#[inline]
pub fn record_term_decision(
    eligible: usize,
    present: u8,
    evaluated: &[u64; TERMS],
    chosen: u8,
    flipped: bool,
) {
    if !enabled() {
        return;
    }
    TERM_DECISIONS.fetch_add(1, Ordering::Relaxed);
    ELIGIBLE_HIST[hist_bucket(eligible)].fetch_add(1, Ordering::Relaxed);
    for t in Term::ALL {
        let i = t.index();
        let bit = 1u8 << i;
        let row = &TERM[i];
        row[TERM_EVALUATED].fetch_add(evaluated[i], Ordering::Relaxed);
        if present & bit != 0 {
            row[TERM_PRESENT].fetch_add(1, Ordering::Relaxed);
            if eligible > 1 {
                row[TERM_CONTESTED].fetch_add(1, Ordering::Relaxed);
            }
        }
        if chosen & bit != 0 {
            row[TERM_WON].fetch_add(1, Ordering::Relaxed);
            if flipped {
                row[TERM_FLIPPED].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The candidate chosen with terms `mask` true was measured for its effect.
#[inline]
pub fn record_term_acted(mask: u8, acted: bool) {
    if !enabled() || mask == 0 {
        return;
    }
    for t in Term::ALL {
        if mask & (1u8 << t.index()) != 0 {
            let row = &TERM[t.index()];
            row[TERM_MEASURED].fetch_add(1, Ordering::Relaxed);
            if acted {
                row[TERM_ACTED].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The queue router drew for a predicated candidate; `routed` says whether
/// the draw sent the step to that candidate's queue.
#[inline]
pub fn record_term_authority(routed: bool) {
    if !enabled() {
        return;
    }
    TERM_AUTHORITY_DRAWS.fetch_add(1, Ordering::Relaxed);
    if routed {
        TERM_AUTHORITY_ROUTED.fetch_add(1, Ordering::Relaxed);
    }
}

/// The steer audit ranked `candidates` runnables at one scheduling point.
#[inline]
pub fn record_audit_candidates(candidates: usize) {
    if !enabled() {
        return;
    }
    CANDIDATES_HIST[hist_bucket(candidates)].fetch_add(1, Ordering::Relaxed);
}

/// The counters of one term.
#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermCounters {
    pub evaluated: u64,
    pub present: u64,
    pub contested: u64,
    pub won: u64,
    pub flipped: u64,
    pub measured: u64,
    pub acted: u64,
}

impl TermCounters {
    fn read(i: usize) -> Self {
        let row = &TERM[i];
        let get = |k: usize| row[k].load(Ordering::Relaxed);
        Self {
            evaluated: get(TERM_EVALUATED),
            present: get(TERM_PRESENT),
            contested: get(TERM_CONTESTED),
            won: get(TERM_WON),
            flipped: get(TERM_FLIPPED),
            measured: get(TERM_MEASURED),
            acted: get(TERM_ACTED),
        }
    }
}

/// The score terms' counters for one session.
#[derive(Serialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct SteerTermStats {
    pub decisions: u64,
    pub authority_draws: u64,
    pub authority_routed: u64,
    pub crash_after_timer_sends: TermCounters,
    pub crash_after_delivery_sends: TermCounters,
    pub stale_late: TermCounters,
    pub request_before_stale: TermCounters,
    pub eligible_hist: Vec<u64>,
    pub candidates_hist: Vec<u64>,
}

impl SteerTermStats {
    fn read() -> Self {
        Self {
            decisions: TERM_DECISIONS.load(Ordering::Relaxed),
            authority_draws: TERM_AUTHORITY_DRAWS.load(Ordering::Relaxed),
            authority_routed: TERM_AUTHORITY_ROUTED.load(Ordering::Relaxed),
            crash_after_timer_sends: TermCounters::read(Term::CrashAfterTimerSends.index()),
            crash_after_delivery_sends: TermCounters::read(Term::CrashAfterDeliverySends.index()),
            stale_late: TermCounters::read(Term::StaleLate.index()),
            request_before_stale: TermCounters::read(Term::RequestBeforeStale.index()),
            eligible_hist: ELIGIBLE_HIST.iter().map(|c| c.load(Ordering::Relaxed)).collect(),
            candidates_hist: CANDIDATES_HIST.iter().map(|c| c.load(Ordering::Relaxed)).collect(),
        }
    }
}

/// Whether recording is enabled. Callers with a non-trivial probe (e.g. the
/// steer argmax comparison) should guard the computation behind this.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the delivery-effect counters for this session. They also
/// require `set_enabled(true)`; this switch exists so the per-delivery probe
/// can be turned off on its own.
pub fn set_acted_fraction_enabled(on: bool) {
    ACTED_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a delivery's effect should be measured. Callers must check this
/// before doing the before/after comparison the measurement needs.
#[inline]
pub fn acted_fraction_enabled() -> bool {
    enabled() && ACTED_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the acceptance-distance census. It rides the
/// delivery-effect probe, so it also requires `set_acted_fraction_enabled(true)`.
pub fn set_acceptance_distance_enabled(on: bool) {
    ACCEPT_DIST_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a delivery's position relative to its receiver's restart should be
/// recorded.
#[inline]
pub fn acceptance_distance_enabled() -> bool {
    acted_fraction_enabled() && ACCEPT_DIST_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the per-run record of how runs stop extending their
/// schedule. It also requires `set_enabled(true)`.
pub fn set_prefix_extension_enabled(on: bool) {
    PREFIX_EXTENSION_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a finished run should be classified by how it stopped extending.
#[inline]
pub fn prefix_extension_enabled() -> bool {
    enabled() && PREFIX_EXTENSION_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the per-run record of how long a run went without a
/// delivery changing anything. It rides the delivery-effect probe, so it also
/// requires `set_acted_fraction_enabled(true)`.
pub fn set_quiet_stretch_enabled(on: bool) {
    QUIET_STRETCH_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a finished run should contribute a quiet-stretch row.
#[inline]
pub fn quiet_stretch_enabled() -> bool {
    acted_fraction_enabled() && QUIET_STRETCH_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the steer-authority audit for this session. It also
/// requires `set_enabled(true)`; the audit walks every queue at every
/// scheduling point, so it is a separate switch from the cheap counters.
pub fn set_steer_audit_enabled(on: bool) {
    STEER_AUDIT_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a scheduling point should be audited. Callers must check this
/// before scoring the queues, which is the expensive part.
#[inline]
pub fn steer_audit_enabled() -> bool {
    enabled() && STEER_AUDIT_ENABLED.load(Ordering::Relaxed)
}

/// Extend the steer-authority audit to sessions where no predicate carries
/// weight, whose ranking is novelty and priority alone. Off leaves those
/// sessions counting only the skip.
pub fn set_steer_audit_always(on: bool) {
    STEER_AUDIT_ALWAYS.store(on, Ordering::Relaxed);
}

/// Whether a scheduling point should be audited even with an unweighted
/// ranking.
#[inline]
pub fn steer_audit_always() -> bool {
    STEER_AUDIT_ALWAYS.load(Ordering::Relaxed)
}

/// Enable or disable the multiplier-authority probe for this session. It also
/// requires `set_enabled(true)`; the probe re-ranks the eligible candidates
/// once per swept magnitude, so it is a separate switch from the cheap counters.
pub fn set_multiplier_audit_enabled(on: bool) {
    MULTIPLIER_AUDIT_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a within-queue selection should be re-ranked under each swept
/// magnitude. Callers must check this before doing the extra ranking.
#[inline]
pub fn multiplier_audit_enabled() -> bool {
    enabled() && MULTIPLIER_AUDIT_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable the identity-weighted recovery term for this session.
/// Unlike the counters, the term's candidate walk is paid whether or not
/// `set_enabled` is on, so the throughput it costs can be read on its own.
pub fn set_recovery_weight_placebo(on: bool) {
    RWP_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a within-queue selection should be walked by the identity-weighted
/// recovery term.
#[inline]
pub fn recovery_weight_placebo_enabled() -> bool {
    RWP_ENABLED.load(Ordering::Relaxed)
}

/// The outcome of one selection walked by the identity-weighted recovery term.
/// `evaluated` is how many eligible candidates the term's predicate was true
/// of, `present` whether that was any of them, `contested` whether it was among
/// more than one eligible candidate, `won` whether the top-ranked candidate was
/// one of them, and `flipped` whether the top-ranked candidate differs from the
/// one the same score without the term ranks first. The term's multiplier is
/// the identity, so `flipped` is zero in a correct build and reads as the
/// check that the term is inert.
#[inline]
pub fn record_recovery_placebo(
    evaluated: u64,
    present: bool,
    contested: bool,
    won: bool,
    flipped: bool,
) {
    if !recovery_weight_placebo_enabled() {
        return;
    }
    bump(|b| &b.rwp_decisions, &RWP_DECISIONS, 1);
    bump(|b| &b.rwp_evaluated, &RWP_EVALUATED, evaluated);
    if present {
        bump(|b| &b.rwp_present, &RWP_PRESENT, 1);
    }
    if contested {
        bump(|b| &b.rwp_contested, &RWP_CONTESTED, 1);
    }
    if won {
        bump(|b| &b.rwp_won, &RWP_WON, 1);
    }
    if flipped {
        bump(|b| &b.rwp_flipped, &RWP_FLIPPED, 1);
    }
}

/// One within-queue selection was seen by the multiplier-authority probe.
/// `contested` means more than one candidate was eligible, `quick_fire_present`
/// that at least one of them is a candidate the multiplier applies to. Only a
/// selection that is both can have its ranking changed by any magnitude, so
/// splitting the two says whether a zero flip rate means the weighting lost or
/// means it was never handed a competitor.
#[inline]
pub fn record_multiplier_decision(contested: bool, quick_fire_present: bool) {
    if !multiplier_audit_enabled() {
        return;
    }
    bump(|b| &b.ma_decisions, &MA_DECISIONS, 1);
    if contested {
        bump(|b| &b.ma_contested_decisions, &MA_CONTESTED_DECISIONS, 1);
    }
    if quick_fire_present {
        bump(|b| &b.ma_quick_fire_offers, &MA_QUICK_FIRE_OFFERS, 1);
        if contested {
            bump(|b| &b.ma_quick_fire_decisions, &MA_QUICK_FIRE_DECISIONS, 1);
        }
    }
}

/// The outcome of re-ranking one selection. `flipped[i]` means the top-ranked
/// candidate under `MULTIPLIER_SWEEP[i]` differs from the one the identity
/// weighting ranks first; `configured_flipped` is the same question asked of
/// the magnitude the session is actually running with.
#[inline]
pub fn record_multiplier_flips(
    configured_multiplier: f64,
    flipped: &[bool; MULTIPLIER_SWEEP.len()],
    configured_flipped: bool,
) {
    if !multiplier_audit_enabled() {
        return;
    }
    RUN_COUNTERS.with(|b| {
        if b.active.get() {
            b.ma_configured_sum
                .set(b.ma_configured_sum.get() + configured_multiplier);
            b.writes.set(b.writes.get() + 1);
        } else {
            add_f64(&MA_CONFIGURED_SUM, configured_multiplier);
        }
    });
    if configured_flipped {
        bump(|b| &b.ma_flipped_configured, &MA_FLIPPED_CONFIGURED, 1);
    }
    for (i, &f) in flipped.iter().enumerate() {
        if f {
            bump(|b| &b.ma_flipped[i], &MA_FLIPPED[i], 1);
        }
    }
}

/// What stood between the highest-scoring runnable and the one a scheduling
/// point actually ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerOutcome {
    /// The step ran the highest-scoring runnable.
    Honored,
    /// Every runnable was filtered out, or the queue the step routed to held
    /// nothing it was allowed to run, so no event was scheduled at all.
    NoEligibleCandidates,
    /// The highest-scoring runnable was withheld by a deliver reservation or
    /// by the per-link delivery order.
    BlockedByOrder,
    /// The highest-scoring runnable was a timer whose label was not currently
    /// permitted to fire.
    BlockedByTimerGate,
    /// The highest-scoring runnable was eligible but sat in a queue the
    /// queue-level router did not choose.
    OtherQueue,
    /// The highest-scoring runnable was eligible and in the chosen queue, but
    /// the randomized within-queue sampler took a different one.
    SamplerChoseOther,
}

/// One step of the run's budget was taken. Counted once per budget step
/// whether or not that step went on to select anything, so it is the
/// denominator `steps` is a subset of, and a session that ran steps can never
/// report zero.
#[inline]
pub fn record_steer_step_total() {
    if !steer_audit_enabled() {
        return;
    }
    bump(|b| &b.sa_steps_total, &SA_STEPS_TOTAL, 1);
}

/// One budget step reached the point where the run's preference is read and
/// the ranking can change what runs. Counted whatever the scoring weights are,
/// so `steps` is the denominator the audited steps are a subset of. Short of
/// `steps_total` by the steps that stop before that point, which is what
/// separates an independent count from an alias of the budget.
#[inline]
pub fn record_steer_step() {
    if !steer_audit_enabled() {
        return;
    }
    bump(|b| &b.sa_steps, &SA_STEPS, 1);
}

/// A scheduling decision asked what the run prefers. Recorded at every site
/// that reads a preference source, before the site decides whether the answer
/// is worth using, so a session that scheduled anything can never report zero
/// consultations. `source_present` is false when no preference source is
/// configured, i.e. no predicate carries weight, which is the reading that
/// separates "the site never ran" from "the site ran and had nothing to say".
///
/// A single scheduling point reaches several such sites - queue routing, the
/// within-queue ranking, and the audit - so this is not a count of steps.
#[inline]
pub fn record_preference_consultation(source_present: bool) {
    if !steer_audit_enabled() {
        return;
    }
    bump(|b| &b.sa_preference_consulted, &SA_PREFERENCE_CONSULTED, 1);
    if !source_present {
        bump(
            |b| &b.sa_preference_source_absent,
            &SA_PREFERENCE_SOURCE_ABSENT,
            1,
        );
    }
}

/// One scheduling point had its preference resolved. `expressed` means the
/// score ranking put a different runnable on top than priority alone would
/// have, i.e. the steering term changed what "preferred" means at this point;
/// `outcome` says what happened to that preferred runnable.
#[inline]
pub fn record_steer_authority(expressed: bool, outcome: SteerOutcome) {
    if !steer_audit_enabled() {
        return;
    }
    SA_AUDITED.fetch_add(1, Ordering::Relaxed);
    if expressed {
        SA_PREFERENCE_EXPRESSED.fetch_add(1, Ordering::Relaxed);
        if outcome == SteerOutcome::Honored {
            SA_PREFERENCE_HONORED.fetch_add(1, Ordering::Relaxed);
        }
    }
    let counter = match outcome {
        SteerOutcome::Honored => &SA_HONORED,
        SteerOutcome::NoEligibleCandidates => &SA_NO_ELIGIBLE,
        SteerOutcome::BlockedByOrder => &SA_BLOCKED_BY_ORDER,
        SteerOutcome::BlockedByTimerGate => &SA_BLOCKED_BY_TIMER_GATE,
        SteerOutcome::OtherQueue => &SA_OTHER_QUEUE,
        SteerOutcome::SamplerChoseOther => &SA_SAMPLER_CHOSE_OTHER,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

/// How far one step got along the path that ends in the scoring function
/// naming a runnable that priority alone would not have named. The variants
/// are ordered by how far the step travelled, and exactly one is recorded per
/// step, so a zero at the end can be read against where the steps stopped
/// instead of standing on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerReach {
    /// Nothing was queued, so the step never reached the scheduling point.
    NoScheduleAttempt,
    /// The scheduling point was reached with the queue-ranking switch off.
    AuditDisabled,
    /// The switch was on and no predicate carried weight, so the ranking was
    /// not resolved.
    NoWeightedPredicate,
    /// The ranking ran and was offered a single candidate, which leaves no
    /// ordering for the score to disagree with.
    SingleCandidate,
    /// The ranking ran over competing candidates and put the same one on top
    /// as priority alone.
    RankingAgreedWithPriority,
    /// The ranking ran over competing candidates and put a different one on
    /// top than priority alone.
    PreferenceExpressed,
}

/// Record where one step stopped on the way to expressing a preference.
#[inline]
pub fn record_steer_reach(reach: SteerReach) {
    if !enabled() {
        return;
    }
    match reach {
        SteerReach::NoScheduleAttempt => {
            bump(|b| &b.sr_no_schedule_attempt, &SR_NO_SCHEDULE_ATTEMPT, 1)
        }
        SteerReach::AuditDisabled => bump(|b| &b.sr_audit_disabled, &SR_AUDIT_DISABLED, 1),
        SteerReach::NoWeightedPredicate => {
            bump(|b| &b.sr_no_weighted_predicate, &SR_NO_WEIGHTED_PREDICATE, 1)
        }
        SteerReach::SingleCandidate => {
            bump(|b| &b.sr_single_candidate, &SR_SINGLE_CANDIDATE, 1)
        }
        SteerReach::RankingAgreedWithPriority => {
            bump(|b| &b.sr_ranking_agreed, &SR_RANKING_AGREED, 1)
        }
        SteerReach::PreferenceExpressed => {
            bump(|b| &b.sr_preference_expressed, &SR_PREFERENCE_EXPRESSED, 1)
        }
    }
}

/// Which scheduler perturbations a message was carrying when it was delivered.
/// A message can carry several at once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeliveryBias(u8);

impl DeliveryBias {
    /// Delivered as sent: no delay, and neither end restarted in between.
    pub const NONE: Self = Self(0);
    /// Held back by the message-delay mechanism before being made schedulable.
    pub const DELAYED: Self = Self(1);
    /// The sending node crashed and recovered between the send and this
    /// delivery, so the message comes from an incarnation that no longer exists.
    pub const SENDER_RESTARTED: Self = Self(2);
    /// The message was kept across the receiver's own crash and handed to a
    /// later incarnation of it.
    pub const RECEIVER_RESTARTED: Self = Self(4);

    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Which distance bucket `distance` handler entries past a restart falls in.
#[inline]
fn acceptance_distance_bucket(distance: u32) -> usize {
    match distance {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        _ => 6,
    }
}

/// A message reached a handler entry on another node. `acted` means the
/// handler changed that node's persistent state rather than falling through a
/// guard, which is the difference between a hazard the protocol saw and one it
/// ignored. Only the handler's first execution segment is observed: a write
/// that happens after the handler blocks on a channel is attributed to nothing.
/// `receiver_distance` is how many handler entries the receiving node had
/// taken since its own most recent restart, not counting this one.
#[inline]
pub fn record_delivery(bias: DeliveryBias, acted: bool, receiver_distance: u32) {
    if !acted_fraction_enabled() {
        return;
    }
    let mut buckets = [DELIVERY_ALL; DELIVERY_BUCKETS];
    let mut len = 1;
    if !bias.is_empty() {
        buckets[len] = DELIVERY_BIASED;
        len += 1;
    }
    for (bit, bucket) in [
        (DeliveryBias::DELAYED, DELIVERY_DELAYED),
        (DeliveryBias::SENDER_RESTARTED, DELIVERY_SENDER_RESTARTED),
        (
            DeliveryBias::RECEIVER_RESTARTED,
            DELIVERY_RECEIVER_RESTARTED,
        ),
    ] {
        if bias.contains(bit) {
            buckets[len] = bucket;
            len += 1;
        }
    }
    for &i in &buckets[..len] {
        bump(|b| &b.deliveries[i], &DELIVERIES[i], 1);
        if acted {
            bump(|b| &b.deliveries_acted[i], &DELIVERIES_ACTED[i], 1);
        }
    }
    if QUIET_STRETCH_ENABLED.load(Ordering::Relaxed) {
        RUN_CROSSING.with(|c| {
            let mut c = c.borrow_mut();
            c.deliveries = c.deliveries.saturating_add(1);
            if acted {
                c.quiet_current = 0;
            } else {
                c.quiet_current = c.quiet_current.saturating_add(1);
                c.quiet_longest = c.quiet_longest.max(c.quiet_current);
            }
        });
    }
    if !ACCEPT_DIST_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let slot = acceptance_distance_bucket(receiver_distance);
    let mut paths = [ACCEPT_PATH_ALL; ACCEPT_PATHS];
    let mut plen = 1;
    for (bit, path) in [
        (DeliveryBias::SENDER_RESTARTED, ACCEPT_PATH_SENDER_RESTARTED),
        (
            DeliveryBias::RECEIVER_RESTARTED,
            ACCEPT_PATH_RECEIVER_RESTARTED,
        ),
    ] {
        if bias.contains(bit) {
            paths[plen] = path;
            plen += 1;
        }
    }
    for &p in &paths[..plen] {
        bump(|b| &b.accept_dist[p][slot], &ACCEPT_DIST[p][slot], 1);
        if acted {
            bump(
                |b| &b.accept_dist_acted[p][slot],
                &ACCEPT_DIST_ACTED[p][slot],
                1,
            );
        }
    }
}

fn add_f64(cell: &AtomicU64, v: f64) {
    let mut cur = cell.load(Ordering::Relaxed);
    loop {
        let next = (f64::from_bits(cur) + v).to_bits();
        match cell.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// One within-queue selection over >1 eligible items was evaluated;
/// `divergent` means the blended-score argmax differed from the
/// priority-only argmax (i.e. novelty/steer changed the greedy pick).
#[inline]
pub fn record_steer_evaluation(divergent: bool) {
    if !enabled() {
        return;
    }
    bump(|b| &b.steer_evaluations, &STEER_EVALUATIONS, 1);
    if divergent {
        bump(|b| &b.steer_divergent_picks, &STEER_DIVERGENT_PICKS, 1);
    }
}

/// A stage of the scoring path that reads the run's state, or ranks a set of
/// runnables, only so the term counters have something to report. With no
/// predicate carrying weight there is nothing for those counters to separate,
/// so each stage is skipped and the skip is counted here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptySliceStage {
    /// The predicate reads over the eligible set of one within-queue selection.
    CandidateMask,
    /// The second ranking of the eligible set that compares the blended argmax
    /// with the priority-only one.
    RankingPass,
    /// The ranking of every runnable in every queue at one scheduling point.
    QueueAudit,
}

/// One occurrence of a stage being skipped. `candidate_mask_skipped` and
/// `ranking_pass_skipped` count within-queue selections, `queue_audit_skipped`
/// counts scheduling points, so a session in which the mechanism applied
/// everywhere has all three at the same magnitude as the corresponding
/// unskipped counters (`steer_terms.decisions`, `steer.evaluations`,
/// `steer_authority.audited`) reach when a predicate does carry weight.
#[inline]
pub fn record_empty_slice_skip(stage: EmptySliceStage) {
    if !enabled() {
        return;
    }
    match stage {
        EmptySliceStage::CandidateMask => {
            bump(|b| &b.es_candidate_mask, &ES_CANDIDATE_MASK, 1)
        }
        EmptySliceStage::RankingPass => bump(|b| &b.es_ranking_pass, &ES_RANKING_PASS, 1),
        EmptySliceStage::QueueAudit => bump(|b| &b.es_queue_audit, &ES_QUEUE_AUDIT, 1),
    }
}

/// A run built its scheduling random source; `isolated` means the run drew
/// each decision kind from its own generator rather than from one shared
/// stream. `isolated_runs` equal to the number of runs is what "the mechanism
/// was on for the whole session" looks like.
#[inline]
pub fn record_rng_isolation(isolated: bool) {
    if !enabled() {
        return;
    }
    if isolated {
        RNG_ISOLATED_RUNS.fetch_add(1, Ordering::Relaxed);
    } else {
        RNG_SHARED_RUNS.fetch_add(1, Ordering::Relaxed);
    }
}

/// A record/ChannelSend was moved into purgatory instead of being enqueued.
/// `receiver_down` splits the held population by whether the destination node
/// was crashed at the moment of the send.
#[inline]
pub fn record_purgatory_delay(receiver_down: bool) {
    if !enabled() {
        return;
    }
    PURGATORY_DELAYED_SENDS.fetch_add(1, Ordering::Relaxed);
    if receiver_down {
        PURGATORY_HOLDS_DOWN_RECEIVER.fetch_add(1, Ordering::Relaxed);
    } else {
        PURGATORY_HOLDS_UP_RECEIVER.fetch_add(1, Ordering::Relaxed);
    }
}

/// A send that purgatory selected for a hold was enqueued undelayed because its
/// destination node was crashed and holds into crashed receivers are disabled.
#[inline]
pub fn record_purgatory_passthrough_down_receiver() {
    if !enabled() {
        return;
    }
    PURGATORY_PASSTHROUGH_DOWN_RECEIVER.fetch_add(1, Ordering::Relaxed);
}

/// The AOS bandit chose an arm (`tape` = TapeMutate, else ConfigMutate).
#[inline]
pub fn record_aos_pick(tape: bool) {
    if !enabled() {
        return;
    }
    if tape {
        AOS_TAPE_WINS.fetch_add(1, Ordering::Relaxed);
    } else {
        AOS_CONFIG_WINS.fetch_add(1, Ordering::Relaxed);
    }
}

/// A candidate was checked against the dedup set; `hit` means it was rejected
/// as a duplicate.
#[inline]
pub fn record_dedup_check(hit: bool) {
    if !enabled() {
        return;
    }
    DEDUP_CHECKS.fetch_add(1, Ordering::Relaxed);
    if hit {
        DEDUP_HITS.fetch_add(1, Ordering::Relaxed);
    }
}

/// One run was scored by a feedback's `plan_score`; each component that the
/// feedback mode computes contributes to its sum (None = not tracked).
#[inline]
pub fn record_feedback_scores(timeline: Option<f64>, cfg: Option<f64>) {
    if !enabled() {
        return;
    }
    FEEDBACK_SCORED_RUNS.fetch_add(1, Ordering::Relaxed);
    if let Some(t) = timeline {
        add_f64(&FEEDBACK_TIMELINE_SCORE_SUM, t);
    }
    if let Some(c) = cfg {
        add_f64(&FEEDBACK_CFG_SCORE_SUM, c);
    }
}

/// Per-run bookkeeping for crash/recover crossings. Runs execute one at a time
/// per thread, so this lives in thread-local storage and needs no locking.
struct RunCrossingState {
    /// Node index -> records addressed to that node before it crashed that are
    /// being held for redelivery when it comes back.
    held: HashMap<usize, u64>,
    counted_in_runs_with_crossing: bool,
    /// Node index -> the scheduler step at which the node restarted, for nodes
    /// whose new incarnation has not yet been handed a message from anyone
    /// else. The recovery window is the span in which the protocol's restart
    /// work and the traffic aimed at it can still interleave.
    recovery_open: HashMap<usize, i32>,
    /// Nodes that completed a crash-and-recover cycle in this run.
    recovered: HashSet<usize>,
    /// Crash and recover events applied in this run.
    fault_events: u32,
    /// The crash events among `fault_events`, and the recover events.
    crashes: u32,
    recovers: u32,
    /// Some node crashed while another node's recovery window was open.
    crash_inside_recovery_window: bool,
    /// Client operations invoked since the most recent recover, or 0 while no
    /// node has recovered yet. Whatever is left here when the run ends is the
    /// client work that outlived every fault.
    client_ops_since_last_recover: u64,
    any_recover: bool,
    /// Deliveries measured for their effect in this run, and the current and
    /// longest span of consecutive ones that changed nothing. A long span is
    /// a stretch of the run in which messages kept arriving and no node's
    /// state moved.
    deliveries: u32,
    quiet_current: u32,
    quiet_longest: u32,
    finalized: bool,
}

impl Default for RunCrossingState {
    fn default() -> Self {
        Self {
            held: HashMap::new(),
            counted_in_runs_with_crossing: false,
            recovery_open: HashMap::new(),
            recovered: HashSet::new(),
            fault_events: 0,
            crashes: 0,
            recovers: 0,
            crash_inside_recovery_window: false,
            client_ops_since_last_recover: 0,
            any_recover: false,
            deliveries: 0,
            quiet_current: 0,
            quiet_longest: 0,
            // Nothing has been recorded yet, so there is no run to flush.
            finalized: true,
        }
    }
}

thread_local! {
    static RUN_CROSSING: RefCell<RunCrossingState> = RefCell::new(RunCrossingState::default());
    /// The cell the run about to execute on this thread generated its plan
    /// under, and whether its dependency density is positive. Set before the
    /// plan runs and consumed when the run terminates, so a run that never
    /// registered contributes to no cell.
    static PLAN_DEPS_RUN: std::cell::Cell<Option<(RecoverDeps, bool)>> = const { std::cell::Cell::new(None) };
    /// Frame counts for the run executing on this thread, folded into the
    /// session totals once the run ends. A thread runs one run at a time, so
    /// no atomic is taken on the interpreter's path.
    static FRAME_RUN: std::cell::Cell<FrameTally> = const { std::cell::Cell::new(FrameTally::new()) };
    /// Counters written at every scheduling step, delivery or timer firing
    /// of the run executing on this thread, folded into the session totals
    /// once the run ends. Each counter is its own cell so a write touches
    /// only that counter's memory.
    static RUN_COUNTERS: RunCounters = const { RunCounters::new() };
    /// Timer effects of the run executing on this thread, merged into the
    /// session table when its counter block folds.
    static RUN_TIMER_EFFECTS: RefCell<HashMap<TimerKey, (u64, u64)>> = RefCell::new(HashMap::new());
}

/// Fold this thread's finished run into the per-run tallies. Idempotent, so it
/// can be called both from the normal end-of-run hook and from the start of the
/// next run for runs that ended without reaching it.
fn finish_run() {
    let Some((unclosed, h3, ordered, bucket, post_recover_ops)) = RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        if c.finalized {
            return None;
        }
        c.finalized = true;
        let unclosed = c.recovery_open.len() as u64;
        c.recovery_open.clear();
        let h3 = c.recovered.len() >= 2;
        Some((
            unclosed,
            h3,
            h3 && c.crash_inside_recovery_window,
            (c.fault_events as usize).min(FAULT_EVENT_BUCKETS - 1),
            c.client_ops_since_last_recover,
        ))
    }) else {
        return;
    };
    PFO_OPS_AFTER_LAST_RECOVER.fetch_add(post_recover_ops, Ordering::Relaxed);
    RW_UNCLOSED.fetch_add(unclosed, Ordering::Relaxed);
    OH3_RUNS.fetch_add(1, Ordering::Relaxed);
    if h3 {
        OH3_WITH_H3.fetch_add(1, Ordering::Relaxed);
    }
    if ordered {
        OH3_WITH_OVERLAP.fetch_add(1, Ordering::Relaxed);
    }
    if let Ok(mut x) = OH3_BY_FAULT_EVENTS.lock() {
        x[bucket][0] += 1;
        if h3 {
            x[bucket][1] += 1;
        }
        if ordered {
            x[bucket][2] += 1;
        }
    }
}

/// One plan execution is starting on this thread. Held-message bookkeeping is
/// per-run, so anything left over from a run that ended while a node was still
/// down is discarded here rather than leaking into the next run.
pub fn begin_run() {
    if !enabled() {
        return;
    }
    finish_run();
    flush_frame_stats();
    fold_run_counters();
    RUN_COUNTERS.with(|b| {
        b.generation.set(STATS_GENERATION.load(Ordering::Relaxed));
        b.active.set(true);
    });
    CR_RUNS.fetch_add(1, Ordering::Relaxed);
    RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        c.held.clear();
        c.counted_in_runs_with_crossing = false;
        c.recovery_open.clear();
        c.recovered.clear();
        c.fault_events = 0;
        c.crashes = 0;
        c.recovers = 0;
        c.crash_inside_recovery_window = false;
        c.client_ops_since_last_recover = 0;
        c.any_recover = false;
        c.deliveries = 0;
        c.quiet_current = 0;
        c.quiet_longest = 0;
        c.finalized = false;
    });
}

/// The plan generator examined one crash-and-recover pair for post-fault client
/// work and added `edges_added` mandatory recover-before-request edges for it.
#[inline]
pub fn record_post_fault_ops(pairs_seen: u64, edges_added: u64) {
    if !enabled() {
        return;
    }
    PFO_PAIRS_SEEN.fetch_add(pairs_seen, Ordering::Relaxed);
    PFO_EDGES_ADDED.fetch_add(edges_added, Ordering::Relaxed);
}

/// The plan generator finished one plan and left `dropped` probabilistic
/// edges into a recover out of it under the exempt cell.
#[inline]
pub fn record_plan_deps_edges(dropped: u64) {
    if !enabled() {
        return;
    }
    PD_RECOVER_EDGES_DROPPED.fetch_add(dropped, Ordering::Relaxed);
}

/// The run about to execute on this thread generated its plan under `cell`,
/// with a positive dependency density or not. Its termination is then folded
/// into that cell's tally.
pub fn record_plan_deps_run(cell: RecoverDeps, density_positive: bool) {
    if !enabled() {
        return;
    }
    PLAN_DEPS_RUN.with(|r| r.set(Some((cell, density_positive))));
}

/// Termination and fault counts over the runs of one plan-dependency cell.
/// A crash is unrecovered when its recover never applied before the run
/// ended, so `unrecovered_crashes` over `crashes` is the same share as
/// `crash_recovery.crashes` minus `crash_recovery.recovers` over the
/// session's crashes; `zero_recovery_runs` counts the runs in which no node
/// completed a crash-and-recover cycle, the same runs
/// `termination.by_recovered_nodes[0]` holds.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlanDepsTally {
    pub runs: u64,
    pub crashes: u64,
    pub unrecovered_crashes: u64,
    pub zero_recovery_runs: u64,
    pub plan_complete: u64,
    pub steps_used_sum: u64,
}

impl PlanDepsTally {
    const fn new() -> Self {
        Self {
            runs: 0,
            crashes: 0,
            unrecovered_crashes: 0,
            zero_recovery_runs: 0,
            plan_complete: 0,
            steps_used_sum: 0,
        }
    }

    fn add(&mut self, s: &RunTermination, crashes: u64, recovers: u64) {
        self.runs += 1;
        self.crashes += crashes;
        self.unrecovered_crashes += crashes.saturating_sub(recovers);
        if s.recovered_nodes == 0 {
            self.zero_recovery_runs += 1;
        }
        if matches!(s.end, RunEnd::PlanComplete) {
            self.plan_complete += 1;
        }
        self.steps_used_sum += s.steps_used;
    }
}

/// One cell's tallies, split by whether the run's dependency density was
/// positive: the exempt rule can only act where the probabilistic pass adds
/// edges, so the two halves are read apart.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlanDepsDensitySplit {
    pub density_zero: PlanDepsTally,
    pub density_positive: PlanDepsTally,
}

impl PlanDepsDensitySplit {
    const fn new() -> Self {
        Self {
            density_zero: PlanDepsTally::new(),
            density_positive: PlanDepsTally::new(),
        }
    }
}

/// How the plan generator ordered node restarts against client work, and
/// how the runs of each cell ended.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlanDepsStats {
    /// Probabilistic edges into a recover left out of exempt-cell plans.
    pub recover_edges_dropped: u64,
    pub stock: PlanDepsDensitySplit,
    pub exempt: PlanDepsDensitySplit,
}

fn fold_plan_deps_run(s: &RunTermination) {
    let Some((cell, density_positive)) = PLAN_DEPS_RUN.with(|r| r.take()) else {
        return;
    };
    let (crashes, recovers) = RUN_CROSSING.with(|c| {
        let c = c.borrow();
        (c.crashes as u64, c.recovers as u64)
    });
    if let Ok(mut cells) = PLAN_DEPS_CELLS.lock() {
        let split = &mut cells[cell.index()];
        let tally = if density_positive {
            &mut split.density_positive
        } else {
            &mut split.density_zero
        };
        tally.add(s, crashes, recovers);
    }
}

/// A planned client operation was handed to a client node. Only invocations
/// that follow every recover in the run are reported, so this is the execution
/// side of the same question the generator edges are meant to force.
#[inline]
pub fn record_client_op_invoked() {
    if !enabled() {
        return;
    }
    RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        if c.any_recover {
            c.client_ops_since_last_recover += 1;
        }
    });
}

/// A node crashed: `held` messages addressed to it were kept for redelivery
/// after it recovers, `dropped` were discarded.
pub fn record_crash(node_index: usize, held: u64, dropped: u64) {
    if !enabled() {
        return;
    }
    CR_CRASHES.fetch_add(1, Ordering::Relaxed);
    CR_HELD_AT_CRASH.fetch_add(held, Ordering::Relaxed);
    CR_DROPPED_AT_CRASH.fetch_add(dropped, Ordering::Relaxed);
    let closed_unclosed = RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        if held > 0 {
            *c.held.entry(node_index).or_insert(0) += held;
        }
        c.fault_events += 1;
        c.crashes += 1;
        // A node's own crash ends its recovery window without a width: no
        // message ever reached the incarnation that came back.
        let own = c.recovery_open.remove(&node_index).is_some();
        if c.recovery_open.keys().any(|&n| n != node_index) {
            c.crash_inside_recovery_window = true;
        }
        own
    });
    if closed_unclosed {
        RW_UNCLOSED.fetch_add(1, Ordering::Relaxed);
    }
}

/// A message from another node entered a handler on `node_index` at scheduler
/// step `step`. This closes the node's recovery window if one is open.
pub fn record_message_entry(node_index: usize, step: i32) {
    if !enabled() {
        return;
    }
    let opened_at = RUN_CROSSING.with(|c| c.borrow_mut().recovery_open.remove(&node_index));
    let Some(opened_at) = opened_at else {
        return;
    };
    let width = step.saturating_sub(opened_at).max(0) as u64;
    RW_CLOSED.fetch_add(1, Ordering::Relaxed);
    RW_WIDTH_SUM.fetch_add(width, Ordering::Relaxed);
    RW_MAX.fetch_max(width, Ordering::Relaxed);
    if let Ok(mut w) = RW_WIDTHS.lock() {
        w[(width as usize).min(RW_WIDTH_CAP)] += 1;
    }
}

/// A node recovered. Every message held from before its crash is requeued to
/// it at this point, so each one is a delivery that crosses the node's own
/// crash/recover boundary. Messages that arrived while the node was down are
/// requeued too but are not counted: they were sent to an already-dead node.
///
/// `own_sends_inflight` means the node still had messages of its own
/// undelivered when it came back, so those messages are now stale-incarnation
/// deliveries waiting to happen.
pub fn record_recover(node_index: usize, step: i32, own_sends_inflight: bool) {
    if !enabled() {
        return;
    }
    CR_RECOVERS.fetch_add(1, Ordering::Relaxed);
    if own_sends_inflight {
        CR_RECOVERS_WITH_INFLIGHT.fetch_add(1, Ordering::Relaxed);
    }
    let first_crossing_of_run = RUN_CROSSING.with(|c| {
        let mut c = c.borrow_mut();
        c.fault_events += 1;
        c.recovers += 1;
        c.recovered.insert(node_index);
        c.recovery_open.insert(node_index, step);
        c.any_recover = true;
        c.client_ops_since_last_recover = 0;
        let crossings = c.held.remove(&node_index).unwrap_or(0);
        if crossings == 0 {
            return None;
        }
        let first = !c.counted_in_runs_with_crossing;
        c.counted_in_runs_with_crossing = true;
        Some((crossings, first))
    });
    let Some((crossings, first)) = first_crossing_of_run else {
        return;
    };
    CR_CROSSING_DELIVERIES.fetch_add(crossings, Ordering::Relaxed);
    if first {
        CR_RUNS_WITH_CROSSING.fetch_add(1, Ordering::Relaxed);
    }
}

/// One scheduling step was examined for the crash-after-send condition:
/// `crash_eligible` means some node had a schedulable crash waiting, and
/// `anchored` means at least one such node also had a message it sent still
/// undelivered, so crashing it there would leave that message orphaned.
#[inline]
pub fn record_crash_anchor_offer(crash_eligible: bool, anchored: bool) {
    if !enabled() {
        return;
    }
    if crash_eligible {
        bump(
            |b| &b.ca_steps_with_crash_eligible,
            &CA_STEPS_WITH_CRASH_ELIGIBLE,
            1,
        );
    }
    if anchored {
        bump(|b| &b.ca_offered, &CA_OFFERED, 1);
    }
}

/// A crash was executed; `anchored` means the crashing node had a message it
/// sent still undelivered at that moment.
#[inline]
pub fn record_crash_anchor_apply(anchored: bool) {
    if !enabled() {
        return;
    }
    CA_CRASHES_TAKEN.fetch_add(1, Ordering::Relaxed);
    if anchored {
        CA_APPLIED.fetch_add(1, Ordering::Relaxed);
    }
}

/// The crash-timing bias looked at one pending crash whose node was not in the
/// middle of its own fan-out, and either withheld it from this step or let it
/// stand. Both counts stay zero while the bias is off, which is what separates
/// "the mechanism did nothing" from "the mechanism was never on".
#[inline]
pub fn record_crash_timing_bias(withheld: bool) {
    if !enabled() {
        return;
    }
    CA_TIMING_BIAS_EXAMINED.fetch_add(1, Ordering::Relaxed);
    if withheld {
        CA_TIMING_BIAS_WITHHELD.fetch_add(1, Ordering::Relaxed);
    }
}

/// Enable or disable the census of what a crash lands on. It also requires
/// `set_enabled(true)`.
pub fn set_crash_census_enabled(on: bool) {
    CRASH_CENSUS_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether a crash should be censused. Callers must check this before reading
/// the candidate set, which walks every node's send bookkeeping.
#[inline]
pub fn crash_census_enabled() -> bool {
    enabled() && CRASH_CENSUS_ENABLED.load(Ordering::Relaxed)
}

/// A crash was applied to a node holding `victim_inflight` undelivered messages
/// of its own. `any_candidate_inflight` means some node whose crash was
/// schedulable at that same point held one, which is the same question asked of
/// the choice rather than of the node it fell on.
#[inline]
pub fn record_crash_census(victim_inflight: u32, any_candidate_inflight: bool) {
    if !crash_census_enabled() {
        return;
    }
    CC_DECISIONS.fetch_add(1, Ordering::Relaxed);
    if victim_inflight > 0 {
        CC_VICTIM_INFLIGHT.fetch_add(1, Ordering::Relaxed);
    }
    if any_candidate_inflight {
        CC_ANY_CANDIDATE_INFLIGHT.fetch_add(1, Ordering::Relaxed);
    }
    let slot = (victim_inflight as usize).min(CC_INFLIGHT_SLOTS - 1);
    CC_INFLIGHT[slot].fetch_add(1, Ordering::Relaxed);
}

/// Where a released crash landed once the absorber ranking was consulted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VictimSwap {
    /// The crash moved to another node; `acted` is that node's mark.
    Applied { acted: bool },
    /// The crash moved to the node a ghost-release firing named, whose
    /// mark had written state. Counted as `Applied` with `acted` as well.
    ForcedOntoAbsorber,
    /// The best absorber was the planned victim.
    SameVictim,
    /// No live node carried a usable mark.
    NoAbsorber,
}

/// A treated run released a planned crash and consulted the absorber
/// ranking; `skipped_pending_pair` means a better-ranked node was passed
/// over because the plan still had a crash or recover outstanding on it.
#[inline]
pub fn record_victim_swap(outcome: VictimSwap, skipped_pending_pair: bool) {
    if !enabled() {
        return;
    }
    match outcome {
        VictimSwap::Applied { acted } => {
            VS_APPLIED.fetch_add(1, Ordering::Relaxed);
            if acted {
                VS_ACTED_ABSORBER.fetch_add(1, Ordering::Relaxed);
            }
        }
        VictimSwap::ForcedOntoAbsorber => {
            VS_APPLIED.fetch_add(1, Ordering::Relaxed);
            VS_ACTED_ABSORBER.fetch_add(1, Ordering::Relaxed);
            VS_FORCED_ONTO_ABSORBER.fetch_add(1, Ordering::Relaxed);
        }
        VictimSwap::SameVictim => {
            VS_SAME_VICTIM.fetch_add(1, Ordering::Relaxed);
        }
        VictimSwap::NoAbsorber => {
            VS_NO_ABSORBER.fetch_add(1, Ordering::Relaxed);
        }
    }
    if skipped_pending_pair {
        VS_SKIPPED_PENDING_PAIR.fetch_add(1, Ordering::Relaxed);
    }
}

/// One eligibility test withheld a planned crash because its victim is
/// already down, which only a prior retarget can bring about.
#[inline]
pub fn record_victim_crashed_hold() {
    if !enabled() {
        return;
    }
    VS_VICTIM_CRASHED_HOLDS.fetch_add(1, Ordering::Relaxed);
}

/// A crash was applied on the treated or the control half to a node that
/// had (`absorbed`) or had not taken a fault-crossing delivery since its
/// last restart, and was (`inflight`) or was not holding an undelivered
/// send of its own.
#[inline]
pub fn record_victim_swap_census(treated: bool, absorbed: bool, inflight: bool) {
    if !enabled() {
        return;
    }
    let i = treated as usize;
    VS_CENSUS_CRASHES[i].fetch_add(1, Ordering::Relaxed);
    if absorbed {
        VS_CENSUS_ABSORBED[i].fetch_add(1, Ordering::Relaxed);
    }
    if inflight {
        VS_CENSUS_INFLIGHT[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// The first time in a run a fault-crossing delivery entered a node whose
/// own crash was queued. Called once per run at most.
#[inline]
pub fn record_ghost_signal_run() {
    if !enabled() {
        return;
    }
    GS_FIRED_RUNS.fetch_add(1, Ordering::Relaxed);
}

/// A network step took a remote record while an eligible remote record of
/// the opposite incarnation class - a ghost against a record of the sender's
/// current incarnation - from the same sender to the same destination was
/// present. `stale_drawn` says the draw fell on the ghost; `down` that the
/// destination was crashed at the time, which the preference leaves alone.
/// Counted on both halves, and on the treated half before any swap.
#[inline]
pub fn record_fresh_first_contest(treated: bool, stale_drawn: bool, down: bool) {
    if !enabled() {
        return;
    }
    let i = treated as usize;
    FF_CONTESTED[i].fetch_add(1, Ordering::Relaxed);
    if stale_drawn {
        FF_STALE_DRAWN[i].fetch_add(1, Ordering::Relaxed);
        if down {
            FF_CONTESTED_DOWN[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A treated run took a fresh rival in place of the drawn ghost; `repeat`
/// says the same ghost had been displaced before.
#[inline]
pub fn record_fresh_first_swap(repeat: bool) {
    if !enabled() {
        return;
    }
    FF_SWAPS.fetch_add(1, Ordering::Relaxed);
    if repeat {
        FF_REPEAT_SWAPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// A ghost that had been displaced was finally taken, on a treated run.
#[inline]
pub fn record_fresh_first_taken(count: fresh_first::DisplacedCount) {
    if !enabled() {
        return;
    }
    let slot = match count {
        fresh_first::DisplacedCount::Once => 0,
        fresh_first::DisplacedCount::Twice => 1,
        fresh_first::DisplacedCount::Thrice => 2,
        fresh_first::DisplacedCount::FourOrMore => 3,
    };
    FF_SWAP_COUNT_HIST[slot].fetch_add(1, Ordering::Relaxed);
}

/// A message entry from a sender's dead incarnation; `overtaken` says its
/// destination had already taken an entry from the sender's current
/// incarnation and `restarted_dest` that the destination has itself come
/// back from a crash in this run. Counted on both halves.
#[inline]
pub fn record_fresh_first_ghost_entry(treated: bool, overtaken: bool, restarted_dest: bool) {
    if !enabled() {
        return;
    }
    let i = treated as usize;
    FF_GHOST_ENTRIES[i].fetch_add(1, Ordering::Relaxed);
    if overtaken {
        FF_OVERTAKEN[i].fetch_add(1, Ordering::Relaxed);
    }
    if restarted_dest {
        FF_GHOST_ENTRIES_RESTARTED_DEST[i].fetch_add(1, Ordering::Relaxed);
        if overtaken {
            FF_OVERTAKEN_RESTARTED_DEST[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A contested step at a destination that has never restarted in the run
/// kept the drawn ghost instead of taking the fresh rival.
#[inline]
pub fn record_fresh_first_skipped_never_restarted_dest() {
    if !enabled() {
        return;
    }
    FF_SKIPPED_NEVER_RESTARTED_DEST.fetch_add(1, Ordering::Relaxed);
}

/// A network step's pick was a remote record from a sender that has crashed
/// at least once in the run while an eligible record of the same class -
/// same sender, same destination, same sending incarnation - was present.
/// `inorder` says the pick already carried the lowest send ordinal of its
/// class among the eligible records. Counted on every treated run before
/// any replacement, and on the census runs of the control half.
#[inline]
pub fn record_pair_order_contest(treated: bool, inorder: bool) {
    if !enabled() {
        return;
    }
    let i = treated as usize;
    PO_CONTESTS[i].fetch_add(1, Ordering::Relaxed);
    if inorder {
        PO_INORDER_DRAWS[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// A treated step's pick was contested. `ghost` says the pick's sending
/// incarnation is not the one running now. Counted before any replacement,
/// so each class's share of the contests the preference sees is readable.
#[inline]
pub fn record_pair_order_contest_class(ghost: bool) {
    if !enabled() {
        return;
    }
    PO_CONTESTS_BY_CLASS[ghost as usize].fetch_add(1, Ordering::Relaxed);
}

/// A treated run replaced the pick with the eligible record of its class
/// carrying the lowest send ordinal. `ghost` says the pick's sending
/// incarnation is not the one running now; only that class is replaced, so
/// the count of the other class is an audit that must read zero.
#[inline]
pub fn record_pair_order_correction(ghost: bool) {
    if !enabled() {
        return;
    }
    PO_CORRECTED.fetch_add(1, Ordering::Relaxed);
    PO_CORRECTIONS_BY_CLASS[ghost as usize].fetch_add(1, Ordering::Relaxed);
}

/// A treated run left a pick of the sender's current incarnation where the
/// draw put it, where a rule covering every class would have replaced it
/// with an earlier send.
#[inline]
pub fn record_pair_order_fresh_suppressed() {
    if !enabled() {
        return;
    }
    PO_FRESH_SUPPRESSED.fetch_add(1, Ordering::Relaxed);
}

/// A message entry from a sender that has crashed at least once in the run
/// whose class - same sender, same destination, same sending incarnation -
/// has a sibling already entered or still in the network queue; `inverted`
/// says a queued sibling carries a lower send ordinal, so the destination
/// acts on the later send first, and `ghost` that the entry's sending
/// incarnation is not the one running now. Counted on the census runs of
/// both halves.
#[inline]
pub fn record_pair_order_entry(treated: bool, ghost: bool, inverted: bool) {
    if !enabled() {
        return;
    }
    let i = treated as usize;
    PO_PAIR_ENTRIES[i].fetch_add(1, Ordering::Relaxed);
    if ghost {
        PO_PAIR_ENTRIES_GHOST[i].fetch_add(1, Ordering::Relaxed);
    }
    if inverted {
        PO_INVERSIONS[i].fetch_add(1, Ordering::Relaxed);
        if ghost {
            PO_INVERSIONS_GHOST[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A run of the given half is a census run: it maintains the pair-entry
/// table and reads the network queue at each message entry from a sender
/// that has crashed.
#[inline]
pub fn record_pair_order_census_run(treated: bool) {
    if !enabled() {
        return;
    }
    PO_SAMPLED_RUNS[treated as usize].fetch_add(1, Ordering::Relaxed);
}

/// A planned client request became ready after the run's first crash.
/// Counted on both halves; on the treated half the request is held.
#[inline]
pub fn record_client_anchor_post_fault_request(treated: bool) {
    if !enabled() {
        return;
    }
    CAN_POPULATION[treated as usize].fetch_add(1, Ordering::Relaxed);
    if treated {
        CAN_HELD.fetch_add(1, Ordering::Relaxed);
    }
}

/// A step's dispatch left a server that took an acted fault-crossing
/// delivery with a full fan-out in the air. Counted on both halves.
#[inline]
pub fn record_client_anchor_window(treated: bool) {
    if !enabled() {
        return;
    }
    CAN_FANOUT_WINDOWS[treated as usize].fetch_add(1, Ordering::Relaxed);
}

/// A client request that became ready after the run's first crash was
/// issued; `in_window` says the step before was one at which a window
/// opened. Counted on both halves.
#[inline]
pub fn record_client_anchor_post_fault_invocation(treated: bool, in_window: bool) {
    if !enabled() {
        return;
    }
    let i = treated as usize;
    CAN_POST_FAULT_INVOCATIONS[i].fetch_add(1, Ordering::Relaxed);
    if in_window {
        CAN_IN_WINDOW_INVOCATIONS[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// A treated run issued a held request: `release` says why it left the
/// queue, `hold_steps` how many steps it waited past its ready step.
#[inline]
pub fn record_client_anchor_release(release: client_anchor::Release, hold_steps: u64) {
    if !enabled() {
        return;
    }
    match release {
        client_anchor::Release::Expiry => {
            CAN_RELEASED_EXPIRY.fetch_add(1, Ordering::Relaxed);
        }
        client_anchor::Release::DryQueue => {
            CAN_RELEASED_DRY_QUEUE.fetch_add(1, Ordering::Relaxed);
        }
    }
    CAN_HOLD_STEPS_SUM.fetch_add(hold_steps, Ordering::Relaxed);
}

/// A treated run's first window opened with `held` requests waiting.
#[inline]
pub fn record_client_anchor_first_window(held: usize) {
    if !enabled() {
        return;
    }
    CAN_HELD_AT_FIRST_FIRING[held.min(3)].fetch_add(1, Ordering::Relaxed);
}

/// A run ended; `completed` says its plan completed and `held_at_exit` how
/// many requests were still held, which only a run that ran out of steps can
/// leave behind.
#[inline]
pub fn record_client_anchor_run_end(treated: bool, completed: bool, held_at_exit: usize) {
    if !enabled() {
        return;
    }
    let i = treated as usize;
    CAN_RUNS[i].fetch_add(1, Ordering::Relaxed);
    if completed {
        CAN_COMPLETED_RUNS[i].fetch_add(1, Ordering::Relaxed);
    }
    if held_at_exit > 0 {
        CAN_HELD_AT_EXIT.fetch_add(held_at_exit as u64, Ordering::Relaxed);
        CAN_RUNS_WITH_HELD_AT_EXIT.fetch_add(1, Ordering::Relaxed);
    }
}

/// A run took `arm` on the post-crash request-timing axis. One call per run.
#[inline]
pub fn record_client_anchor_arm_run(arm: client_anchor::Arm) {
    if !enabled() {
        return;
    }
    CAN_ARM_RUNS[arm.index()].fetch_add(1, Ordering::Relaxed);
}

fn micro(x: f64) -> u64 {
    (x.clamp(0.0, 1.0) * 1_000_000.0).round() as u64
}

/// A run assigned to the arm selector's learner at `learner` (its position
/// among the three) was drawn against exploration share `share` on
/// campaign arm `arm_index`; `pooled` says the cell it was read from is
/// keyed by a configuration and shared across arms, and `coin` that it
/// came out coin-drawn. `leader_margin` is the highest per-axis pairwise
/// probability that the axis's highest-mean direction leads the next,
/// which the share was read from, or None below the cell's warmup, where
/// the share is one.
pub fn record_arm_selector_draw(
    learner: usize,
    arm_index: i32,
    pooled: bool,
    share: f64,
    leader_margin: Option<f64>,
    coin: bool,
) {
    if !enabled() {
        return;
    }
    let e = &AX_EXPLORE;
    let slot = ((arm_index + 1).max(0) as usize).min(AX_ARM_SLOTS - 1);
    let learner = learner.min(AX_LEARNERS - 1);
    let share_micro = micro(share);
    e.draws.fetch_add(1, Ordering::Relaxed);
    if pooled {
        AX_POOLED.draws.fetch_add(1, Ordering::Relaxed);
    }
    e.share_micro.fetch_add(share_micro, Ordering::Relaxed);
    e.draws_by_learner[learner].fetch_add(1, Ordering::Relaxed);
    e.share_micro_by_learner[learner].fetch_add(share_micro, Ordering::Relaxed);
    e.draws_by_arm[slot].fetch_add(1, Ordering::Relaxed);
    let bin = ((share.clamp(0.0, 1.0) * AX_SHARE_BINS as f64) as usize).min(AX_SHARE_BINS - 1);
    e.share_hist[bin].fetch_add(1, Ordering::Relaxed);
    match leader_margin {
        Some(m) => {
            e.margin_micro.fetch_add(micro(m), Ordering::Relaxed);
            e.margin_micro_by_arm[slot].fetch_add(micro(m), Ordering::Relaxed);
        }
        None => {
            e.warmup_coin_runs.fetch_add(1, Ordering::Relaxed);
        }
    }
    if coin {
        e.coin_runs.fetch_add(1, Ordering::Relaxed);
        e.coin_runs_by_learner[learner].fetch_add(1, Ordering::Relaxed);
        e.coin_runs_by_arm[slot].fetch_add(1, Ordering::Relaxed);
    }
}

/// A learner run drew its arms: `reward` names the learner, `arms` the set
/// it drew against the run id's `coins`. `leader_agreements` counts the
/// axes on which the drawn direction was the axis's highest posterior
/// mean, and `leader_margins` carries the per-axis pairwise probability
/// that the highest-mean direction leads the next.
pub fn record_arm_selector_learner_run(
    reward: Reward,
    arms: &run_variant::ArmSet,
    coins: &run_variant::ArmSet,
    leader_agreements: u64,
    leader_margins: &[f64; run_variant::AXES],
) {
    if !enabled() {
        return;
    }
    let ax = &AX[reward.index()];
    ax.chosen_runs.fetch_add(1, Ordering::Relaxed);
    for (a, m) in leader_margins.iter().enumerate() {
        ax.leader_margin_micro[a].fetch_add(micro(*m), Ordering::Relaxed);
    }
    ax.axis_draws.fetch_add(run_variant::AXES as u64, Ordering::Relaxed);
    ax.axis_leader_agreements.fetch_add(leader_agreements, Ordering::Relaxed);
    if arms != coins {
        ax.departures.fetch_add(1, Ordering::Relaxed);
    }
    if arms.placed() {
        ax.chosen_placed_runs.fetch_add(1, Ordering::Relaxed);
    }
    for d in arms.directions() {
        ax.chosen_by_direction[d].fetch_add(1, Ordering::Relaxed);
    }
    ax.chosen_by_combination[arms.index()].fetch_add(1, Ordering::Relaxed);
}

/// A run the arm selector observed, once per run whichever rewards were
/// read on it: `arm_index` is the campaign arm the run's cell belongs to
/// and `overtaken_ghost` the first learner's reward.
pub fn record_arm_selector_run_observed(arm_index: i32, overtaken_ghost: bool) {
    if !enabled() {
        return;
    }
    AX_OBSERVATIONS.fetch_add(1, Ordering::Relaxed);
    let slot = ((arm_index + 1).max(0) as usize).min(AX_ARM_SLOTS - 1);
    AX_REWARD_RUNS_BY_ARM[slot].fetch_add(1, Ordering::Relaxed);
    if overtaken_ghost {
        AX_REWARD_POSITIVE_BY_ARM[slot].fetch_add(1, Ordering::Relaxed);
    }
}

/// One reward was read on a run: `treated` says whether the run was a
/// learner run of the reward's learner rather than coin-drawn, `arm_index`
/// the campaign arm the run ran under, `arms` the set the run carried, and
/// `positive` the reward's value.
pub fn record_arm_selector_observation(
    reward: Reward,
    treated: bool,
    arm_index: i32,
    arms: &run_variant::ArmSet,
    positive: bool,
) {
    if !enabled() {
        return;
    }
    let ax = &AX[reward.index()];
    let half = treated as usize;
    ax.reward_runs[half].fetch_add(1, Ordering::Relaxed);
    if positive {
        ax.reward_positive[half].fetch_add(1, Ordering::Relaxed);
    }
    if treated {
        return;
    }
    let row = ((arm_index + 1).max(0) as usize).min(AX_ARM_SLOTS - 1) * run_variant::DIRECTIONS;
    for d in arms.directions() {
        ax.control_runs_by_direction[d].fetch_add(1, Ordering::Relaxed);
        ax.control_runs_by_arm_direction[row + d].fetch_add(1, Ordering::Relaxed);
        if positive {
            ax.control_reward_positive_by_direction[d].fetch_add(1, Ordering::Relaxed);
            ax.control_reward_positive_by_arm_direction[row + d].fetch_add(1, Ordering::Relaxed);
        }
    }
    let c = arms.index();
    ax.control_runs_by_combination[c].fetch_add(1, Ordering::Relaxed);
    if positive {
        ax.control_reward_positive_by_combination[c].fetch_add(1, Ordering::Relaxed);
    }
}

/// A gauge, not a counter, so it reads with stats off: a cell was created
/// in the learner trained on `reward`; `pooled` says the cell is keyed by
/// a configuration and shared across campaign arms.
pub fn record_arm_selector_cell_created(reward: Reward, pooled: bool) {
    let ax = &AX[reward.index()];
    ax.cells.fetch_add(1, Ordering::Relaxed);
    if pooled {
        ax.pooled_cells.fetch_add(1, Ordering::Relaxed);
    }
}

/// A run was credited into a cell shared across campaign arms by the
/// learner at `learner` (its position among the three).
pub fn record_arm_selector_pooled_credit(learner: usize) {
    if !enabled() {
        return;
    }
    AX_POOLED.observations_by_learner[learner.min(AX_LEARNERS - 1)].fetch_add(1, Ordering::Relaxed);
}

/// The learners were cleared, so their gauges read empty.
pub fn reset_arm_selector_gauges() {
    for ax in &AX {
        ax.cells.store(0, Ordering::Relaxed);
        ax.pooled_cells.store(0, Ordering::Relaxed);
    }
}

/// A post-crash client request was issued under the rush direction.
#[inline]
pub fn record_client_anchor_rush_op() {
    if !enabled() {
        return;
    }
    CAN_RUSH_OPS.fetch_add(1, Ordering::Relaxed);
}

/// A record took the top of the priority range because the operation that
/// caused it is rushed.
#[inline]
pub fn record_client_anchor_rush_record() {
    if !enabled() {
        return;
    }
    CAN_RUSH_RECORDS_PRIORITIZED.fetch_add(1, Ordering::Relaxed);
}

/// A network step had a rushed record among its eligible candidates;
/// `was_pick` says the step dispatched one, rather than a record the
/// eligibility and preference layers took ahead of it.
#[inline]
pub fn record_client_anchor_rush_dispatch(was_pick: bool) {
    if !enabled() {
        return;
    }
    if was_pick {
        CAN_RUSH_WAS_PICK.fetch_add(1, Ordering::Relaxed);
    } else {
        CAN_RUSH_DISPLACED.fetch_add(1, Ordering::Relaxed);
    }
}

/// A post-crash client request was delivered for the first time, `steps`
/// after it was issued. Counted on every direction of the axis.
#[inline]
pub fn record_client_anchor_first_delivery(arm: client_anchor::Arm, steps: u64) {
    if !enabled() {
        return;
    }
    let i = arm.index();
    CAN_FIRST_DELIVERY_SUM[i].fetch_add(steps, Ordering::Relaxed);
    CAN_FIRST_DELIVERY_COUNT[i].fetch_add(1, Ordering::Relaxed);
}

/// A coin-drawn run's first message entry caused by a post-fault client
/// operation landed at `step`, credited to the directions the run carried.
#[inline]
pub fn record_client_anchor_first_post_fault_entry(arms: &run_variant::ArmSet, step: i32) {
    if !enabled() {
        return;
    }
    let step = step.max(0) as u64;
    for d in arms.directions() {
        CAN_FIRST_ENTRY_RUNS[d].fetch_add(1, Ordering::Relaxed);
        CAN_FIRST_ENTRY_STEPS_SUM[d].fetch_add(step, Ordering::Relaxed);
    }
}

/// A fresh grid-arm run that fired the signal entered its arm's replay corpus.
#[inline]
pub fn record_replay_parent_admitted() {
    if !enabled() {
        return;
    }
    RP_PARENTS_ADMITTED.fetch_add(1, Ordering::Relaxed);
}

/// A replay slot found its arm's corpus empty and ran fresh.
#[inline]
pub fn record_replay_slot_unfilled() {
    if !enabled() {
        return;
    }
    RP_SLOTS_UNFILLED.fetch_add(1, Ordering::Relaxed);
}

/// A fresh grid-arm run recorded `words` scheduling draws.
#[inline]
pub fn record_replay_tape_words(words: u64) {
    if !enabled() {
        return;
    }
    RP_TAPE_WORDS_SUM.fetch_add(words, Ordering::Relaxed);
}

/// A replay slot ran a child of a corpus parent. `prefix` says the child
/// replayed the parent's draws rather than only its plan; `signal_fired`
/// that the child itself reached the signal; `faithful` that a prefix child
/// reached it at the parent's own step.
#[inline]
pub fn record_replay_child(prefix: bool, signal_fired: bool, faithful: bool) {
    if !enabled() {
        return;
    }
    RP_CHILDREN.fetch_add(1, Ordering::Relaxed);
    if prefix {
        RP_CHILDREN_PREFIX.fetch_add(1, Ordering::Relaxed);
    } else {
        RP_CHILDREN_PLAN_ONLY.fetch_add(1, Ordering::Relaxed);
    }
    if signal_fired {
        RP_CHILDREN_SIGNAL_FIRED.fetch_add(1, Ordering::Relaxed);
    }
    if faithful {
        RP_PREFIX_FAITHFUL.fetch_add(1, Ordering::Relaxed);
    }
}

/// Why a single plan execution stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunEnd {
    /// Every planned event reached Completed.
    PlanComplete,
    /// The per-run step budget ran out while planned events were outstanding.
    IterationsExhausted,
    /// No runnable work and no planned event able to become ready.
    Deadlock,
    /// The learned step cap, short of the configured budget, ended the run
    /// while planned events were outstanding.
    LearnedCapReached,
    /// The learned stall cap ended the run: its clock ran past the cap
    /// without a progress mark while planned events were outstanding.
    StallCapReached,
}

/// Termination counts and running sums over one bucket of runs. `steps_used`
/// and the queue depths are summed rather than averaged so buckets can be
/// merged; divide by `runs` to read a mean.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TerminationTally {
    pub runs: u64,
    pub plan_complete: u64,
    /// Plan finished while messages, timers or delayed sends were still
    /// queued, so the remaining protocol traffic never ran.
    pub plan_complete_with_pending_work: u64,
    pub iterations_exhausted: u64,
    pub deadlock: u64,
    pub learned_cap_reached: u64,
    pub stall_cap_reached: u64,
    pub steps_used_sum: u64,
    pub step_budget_sum: u64,
    pub pending_work_at_exit_sum: u64,
    pub planned_events_outstanding_sum: u64,
}

impl TerminationTally {
    const fn new() -> Self {
        Self {
            runs: 0,
            plan_complete: 0,
            plan_complete_with_pending_work: 0,
            iterations_exhausted: 0,
            deadlock: 0,
            learned_cap_reached: 0,
            stall_cap_reached: 0,
            steps_used_sum: 0,
            step_budget_sum: 0,
            pending_work_at_exit_sum: 0,
            planned_events_outstanding_sum: 0,
        }
    }

    fn add(&mut self, end: RunEnd, s: &RunTermination) {
        self.runs += 1;
        match end {
            RunEnd::PlanComplete => {
                self.plan_complete += 1;
                if s.pending_work_at_exit > 0 {
                    self.plan_complete_with_pending_work += 1;
                }
            }
            RunEnd::IterationsExhausted => self.iterations_exhausted += 1,
            RunEnd::Deadlock => self.deadlock += 1,
            RunEnd::LearnedCapReached => self.learned_cap_reached += 1,
            RunEnd::StallCapReached => self.stall_cap_reached += 1,
        }
        self.steps_used_sum += s.steps_used;
        self.step_budget_sum += s.step_budget;
        self.pending_work_at_exit_sum += s.pending_work_at_exit;
        self.planned_events_outstanding_sum += s.planned_events_outstanding;
    }
}

/// One run's termination facts.
pub struct RunTermination {
    pub end: RunEnd,
    pub steps_used: u64,
    pub step_budget: u64,
    /// Runnables still queued (including delayed sends) when the run stopped.
    pub pending_work_at_exit: u64,
    pub planned_events_outstanding: u64,
    /// Distinct nodes that both crashed and recovered during the run. Deep
    /// fault interleavings need at least two; the bucketed tallies show
    /// whether those runs stop for a different reason than shallow ones.
    pub recovered_nodes: usize,
}

/// Termination tallies over all runs and split by how many distinct nodes
/// completed a crash-and-recover cycle (index 0, 1, and 2-or-more).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TerminationStats {
    pub all: TerminationTally,
    pub by_recovered_nodes: [TerminationTally; 3],
}

impl TerminationStats {
    const fn new() -> Self {
        Self {
            all: TerminationTally::new(),
            by_recovered_nodes: [TerminationTally::new(); 3],
        }
    }
}

/// One plan execution finished. Called once per run, off the scheduling hot
/// path.
pub fn record_run_termination(s: &RunTermination) {
    if !enabled() {
        return;
    }
    fold_run_counters();
    debug_assert!(
        !(steer_audit_enabled() && s.steps_used > 0 && SA_STEPS_TOTAL.load(Ordering::Relaxed) == 0),
        "a run took {} scheduling steps and none was counted; the steer-authority \
         counters are not reaching the scheduler",
        s.steps_used
    );
    fold_plan_deps_run(s);
    finish_run();
    let bucket = s.recovered_nodes.min(2);
    if let Ok(mut t) = TERMINATION.lock() {
        t.all.add(s.end, s);
        t.by_recovered_nodes[bucket].add(s.end, s);
    }
}

/// A budget-ended run whose last steps released nothing was not short of
/// budget, it was short of releasable work. The threshold is long enough that
/// waiting out one delayed message does not read as stopped.
const STALLED_TAIL_STEPS: u64 = 100;

/// How one run stopped extending its schedule. A run extends for as long as the
/// scheduler keeps releasing queued work; it stops when the plan has no event
/// left to complete, when the step budget ends it, or when nothing that is
/// queued can be released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixStop {
    /// The plan finished and nothing was left queued.
    PlanCompleteQuiescent,
    /// The plan finished while protocol work was still queued, so the plan,
    /// not the protocol, is what ended the run.
    PlanCompletePending,
    /// The budget ended a run that was still releasing work.
    BudgetReleasing,
    /// The budget ended a run whose queued work had stopped being releasable.
    BudgetBlocked,
    /// The budget ended a run with nothing queued to release.
    BudgetIdle,
    Deadlock,
}

/// One run's extension facts. `steps_blocked` counts the steps where queued
/// work existed and the scheduler released none of it, which is an extension
/// offered at the frontier and refused; `steps_idle` counts the steps that had
/// nothing queued to offer.
pub struct RunExtension {
    pub end: RunEnd,
    pub steps: u64,
    pub steps_released: u64,
    pub steps_blocked: u64,
    pub steps_idle: u64,
    /// Steps at the end of the run, up to termination, that released nothing.
    pub tail_without_release: u64,
    /// Runnables still queued (including delayed sends) when the run stopped.
    pub pending_at_exit: u64,
    pub recovered_nodes: usize,
}

impl RunExtension {
    fn stop(&self) -> PrefixStop {
        match self.end {
            RunEnd::Deadlock => PrefixStop::Deadlock,
            RunEnd::PlanComplete if self.pending_at_exit == 0 => PrefixStop::PlanCompleteQuiescent,
            RunEnd::PlanComplete => PrefixStop::PlanCompletePending,
            RunEnd::IterationsExhausted | RunEnd::LearnedCapReached | RunEnd::StallCapReached => {
                if self.pending_at_exit == 0 {
                    PrefixStop::BudgetIdle
                } else if self.tail_without_release >= STALLED_TAIL_STEPS {
                    PrefixStop::BudgetBlocked
                } else {
                    PrefixStop::BudgetReleasing
                }
            }
        }
    }
}

/// Stop counts and running sums over one bucket of runs. The sums are summed
/// rather than averaged so buckets can be merged; divide by `runs` to read a
/// mean.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PrefixExtensionTally {
    pub runs: u64,
    pub plan_complete_quiescent: u64,
    pub plan_complete_pending: u64,
    pub budget_releasing: u64,
    pub budget_blocked: u64,
    pub budget_idle: u64,
    pub deadlock: u64,
    pub steps_sum: u64,
    pub steps_released_sum: u64,
    pub steps_blocked_sum: u64,
    pub steps_idle_sum: u64,
    pub tail_without_release_sum: u64,
    pub pending_at_exit_sum: u64,
}

impl PrefixExtensionTally {
    const fn new() -> Self {
        Self {
            runs: 0,
            plan_complete_quiescent: 0,
            plan_complete_pending: 0,
            budget_releasing: 0,
            budget_blocked: 0,
            budget_idle: 0,
            deadlock: 0,
            steps_sum: 0,
            steps_released_sum: 0,
            steps_blocked_sum: 0,
            steps_idle_sum: 0,
            tail_without_release_sum: 0,
            pending_at_exit_sum: 0,
        }
    }

    fn add(&mut self, x: &RunExtension) {
        self.runs += 1;
        match x.stop() {
            PrefixStop::PlanCompleteQuiescent => self.plan_complete_quiescent += 1,
            PrefixStop::PlanCompletePending => self.plan_complete_pending += 1,
            PrefixStop::BudgetReleasing => self.budget_releasing += 1,
            PrefixStop::BudgetBlocked => self.budget_blocked += 1,
            PrefixStop::BudgetIdle => self.budget_idle += 1,
            PrefixStop::Deadlock => self.deadlock += 1,
        }
        self.steps_sum += x.steps;
        self.steps_released_sum += x.steps_released;
        self.steps_blocked_sum += x.steps_blocked;
        self.steps_idle_sum += x.steps_idle;
        self.tail_without_release_sum += x.tail_without_release;
        self.pending_at_exit_sum += x.pending_at_exit;
    }
}

/// Extension tallies over all runs and split by how many distinct nodes
/// completed a crash-and-recover cycle (index 0, 1, and 2-or-more), so the
/// runs that carry the deepest fault interleavings can be read apart from the
/// shallow ones that dominate the total.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PrefixExtensionStats {
    pub all: PrefixExtensionTally,
    pub by_recovered_nodes: [PrefixExtensionTally; 3],
}

impl PrefixExtensionStats {
    const fn new() -> Self {
        Self {
            all: PrefixExtensionTally::new(),
            by_recovered_nodes: [PrefixExtensionTally::new(); 3],
        }
    }
}

/// One plan execution finished. Called once per run, off the scheduling hot
/// path.
pub fn record_run_extension(x: &RunExtension) {
    if !prefix_extension_enabled() {
        return;
    }
    let bucket = x.recovered_nodes.min(2);
    if let Ok(mut p) = PREFIX_EXTENSION.lock() {
        p.all.add(x);
        p.by_recovered_nodes[bucket].add(x);
    }
}

/// One finished run's quiet stretch. `longest_quiet_stretch` counts the
/// consecutive deliveries whose handler left the receiving node's state
/// unchanged, at the longest such span of the run; `deliveries` is how many
/// deliveries the span was drawn from, so a short run is not read as a quiet
/// one. `run_id` is the key a per-run measurement taken outside the simulator,
/// such as a grader's prefix depth, is joined on.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct QuietStretchRun {
    pub run_id: i64,
    pub longest_quiet_stretch: u32,
    pub deliveries: u32,
}

struct QuietStretchState {
    /// Longest-stretch histograms indexed by how the run ended, in the order
    /// `RunEnd` is declared.
    by_end: [[u64; HIST_BUCKETS]; 5],
    per_run: Vec<QuietStretchRun>,
    dropped: u64,
}

impl QuietStretchState {
    const fn new() -> Self {
        Self {
            by_end: [[0; HIST_BUCKETS]; 5],
            per_run: Vec::new(),
            dropped: 0,
        }
    }
}

/// How long runs go without a delivery having an effect, as log2 buckets of
/// the longest such stretch (0, 1, 2, 3-4, 5-8, ...) split by how the run
/// ended, and as one row per run. The simulator has no view of a run's prefix
/// depth, so the rows carry the run id that depth is keyed by and the two
/// axes are crossed by whoever holds both. `per_run_dropped` counts the runs
/// that finished after the row cap was reached.
#[derive(Serialize)]
pub struct QuietStretchStats {
    pub runs: u64,
    pub plan_complete: Vec<u64>,
    pub iterations_exhausted: Vec<u64>,
    pub deadlock: Vec<u64>,
    pub learned_cap_reached: Vec<u64>,
    pub stall_cap_reached: Vec<u64>,
    pub per_run: Vec<QuietStretchRun>,
    pub per_run_dropped: u64,
}

impl QuietStretchStats {
    fn read() -> Self {
        let q = QUIET_STRETCH
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self {
            runs: q.by_end.iter().flatten().sum(),
            plan_complete: q.by_end[0].to_vec(),
            iterations_exhausted: q.by_end[1].to_vec(),
            deadlock: q.by_end[2].to_vec(),
            learned_cap_reached: q.by_end[3].to_vec(),
            stall_cap_reached: q.by_end[4].to_vec(),
            per_run: q.per_run.clone(),
            per_run_dropped: q.dropped,
        }
    }
}

/// One plan execution finished. Called once per run, off the scheduling hot
/// path.
pub fn record_quiet_stretch(run_id: i64, end: RunEnd) {
    if !quiet_stretch_enabled() {
        return;
    }
    let (longest, deliveries) = RUN_CROSSING.with(|c| {
        let c = c.borrow();
        (c.quiet_longest, c.deliveries)
    });
    let row = match end {
        RunEnd::PlanComplete => 0,
        RunEnd::IterationsExhausted => 1,
        RunEnd::Deadlock => 2,
        RunEnd::LearnedCapReached => 3,
        RunEnd::StallCapReached => 4,
    };
    if let Ok(mut q) = QUIET_STRETCH.lock() {
        q.by_end[row][hist_bucket(longest as usize)] += 1;
        if q.per_run.len() < QUIET_PER_RUN_CAP {
            q.per_run.push(QuietStretchRun {
                run_id,
                longest_quiet_stretch: longest,
                deliveries,
            });
        } else {
            q.dropped += 1;
        }
    }
}

/// One probe run finished, of any outcome; `completed` marks the probes
/// whose length fed the learned-cap distribution.
pub fn record_run_cap_probe(completed: bool) {
    if !enabled() {
        return;
    }
    RUN_CAP_PROBES.fetch_add(1, Ordering::Relaxed);
    if completed {
        RUN_CAP_PROBE_COMPLETIONS.fetch_add(1, Ordering::Relaxed);
    }
}

/// One completed probe ran longer than the learned cap in effect when it
/// merged, so a capped run in its place would not have completed.
pub fn record_run_cap_over_cap_completion() {
    if !enabled() {
        return;
    }
    RUN_CAP_OVER_CAP_COMPLETIONS.fetch_add(1, Ordering::Relaxed);
}

/// The learned cap for one scope was recomputed at a completed-count
/// checkpoint and now governs that scope's runs.
pub fn record_run_cap_recompute() {
    if !enabled() {
        return;
    }
    RUN_CAP_CAP_RECOMPUTES.fetch_add(1, Ordering::Relaxed);
}

/// Gauges, not counters: overwritten with the learner's current view so the
/// snapshot reads its state, ungated so it is visible with stats off.
pub fn set_run_cap_learned(scopes: u64, cap_max_scope: u64) {
    RUN_CAP_SCOPES_LEARNED.store(scopes, Ordering::Relaxed);
    RUN_CAP_CURRENT_CAP_MAX_SCOPE.store(cap_max_scope, Ordering::Relaxed);
}

/// The cell a run occupies under the stall cap: cut once its clock exceeds
/// the cap, measured but never cut, or exempt because it feeds a learner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StallCapCell {
    Treated,
    Untreated,
    Probe,
}

/// Progress marks of one run or of a session, by kind. A step carrying
/// several kinds counts under each.
#[derive(Clone, Debug, Default, Serialize)]
pub struct StallCapMarks {
    pub rows: u64,
    pub acted_deliveries: u64,
    pub acted_timers: u64,
    pub releases: u64,
}

/// One run's stall-cap facts, reported once when the run ends.
pub struct StallCapRun {
    pub cell: StallCapCell,
    pub marks: StallCapMarks,
    pub suspended_steps: u64,
    /// Steps the stop saved against the run's frozen step cap, on a run the
    /// stall cap ended.
    pub steps_saved: Option<u64>,
    /// The run's longest quiet gap, counting its final segment.
    pub longest_gap: u32,
    /// The cap standing for the run's scope when it ended, if any.
    pub standing_cap: Option<u32>,
    /// The untreated row was not kept because the row cap was reached.
    pub row_dropped: bool,
}

/// One run finished under the stall cap, of any cell. Called once per run,
/// off the scheduling hot path.
pub fn record_stall_cap_run(r: &StallCapRun) {
    if !enabled() {
        return;
    }
    match r.cell {
        StallCapCell::Treated => {
            SC_TREATED_RUNS.fetch_add(1, Ordering::Relaxed);
        }
        StallCapCell::Untreated => {
            SC_UNTREATED_RUNS.fetch_add(1, Ordering::Relaxed);
            SC_UNTREATED_GAP_HIST[hist_bucket(r.longest_gap as usize)]
                .fetch_add(1, Ordering::Relaxed);
            if let Some(cap) = r.standing_cap {
                SC_UNTREATED_RUNS_CAPPED.fetch_add(1, Ordering::Relaxed);
                if r.longest_gap > cap {
                    SC_UNTREATED_OVER_CAP_RUNS.fetch_add(1, Ordering::Relaxed);
                }
            }
            if r.row_dropped {
                SC_UNTREATED_ROWS_DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
        StallCapCell::Probe => {}
    }
    if let Some(saved) = r.steps_saved {
        SC_STOPS.fetch_add(1, Ordering::Relaxed);
        SC_STEPS_SAVED_SUM.fetch_add(saved, Ordering::Relaxed);
    }
    SC_SUSPENDED_STEPS_SUM.fetch_add(r.suspended_steps, Ordering::Relaxed);
    SC_MARK_ROWS.fetch_add(r.marks.rows, Ordering::Relaxed);
    SC_MARK_ACTED_DELIVERIES.fetch_add(r.marks.acted_deliveries, Ordering::Relaxed);
    SC_MARK_ACTED_TIMERS.fetch_add(r.marks.acted_timers, Ordering::Relaxed);
    SC_MARK_RELEASES.fetch_add(r.marks.releases, Ordering::Relaxed);
}

/// One completed run-cap probe's longest gap was folded into the stall-cap
/// learner; `over_cap` marks the probes whose gap exceeded the cap in effect
/// when they merged, so a treated run in their place would have been cut.
pub fn record_stall_cap_probe(over_cap: bool) {
    if !enabled() {
        return;
    }
    SC_PROBES_KEYED.fetch_add(1, Ordering::Relaxed);
    if over_cap {
        SC_PROBE_OVER_CAP_COMPLETIONS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Gauges of the stall-cap learner's current view, ungated like the step
/// cap's.
pub fn set_stall_cap_learned(scopes: u64, cap_max_scope: u64) {
    SC_SCOPES_LEARNED.store(scopes, Ordering::Relaxed);
    SC_CAP_MAX_SCOPE.store(cap_max_scope, Ordering::Relaxed);
}

/// The cell a run occupies under the stall release: a stall-cap-treated run
/// whose first stall settles its in-progress client operations for the plan
/// and goes on, a stall-cap-treated run its first stall ends, or a run the
/// stall cap does not treat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StallReleaseCell {
    Release,
    Cut,
    Exempt,
}

/// One run's first stall settled its in-progress client operations; the
/// dependents are the planned events that became ready as a result, by
/// kind.
pub struct StallReleaseSettlement {
    pub ops_settled: u64,
    pub dependents_client: u64,
    pub dependents_fault: u64,
    pub dependents_other: u64,
}

/// One run ended, as far as the stall release is concerned.
pub struct StallReleaseRun {
    pub cell: StallReleaseCell,
    /// Steps the run had run when its stall released it, on a released run.
    pub release_step: Option<u64>,
    /// Steps the run ran in all.
    pub steps: u64,
    /// Every planned event completed.
    pub completed: bool,
    /// The stall cap ended the run.
    pub stalled: bool,
    /// ClientInterface invocations the run issued.
    pub invocations: u64,
}

/// One run's first stall released it. Called once per released run.
pub fn record_stall_release(s: &StallReleaseSettlement) {
    if !enabled() {
        return;
    }
    SR_RELEASES.fetch_add(1, Ordering::Relaxed);
    SR_OPS_SETTLED.fetch_add(s.ops_settled, Ordering::Relaxed);
    SR_DEPENDENTS_CLIENT.fetch_add(s.dependents_client, Ordering::Relaxed);
    SR_DEPENDENTS_FAULT.fetch_add(s.dependents_fault, Ordering::Relaxed);
    SR_DEPENDENTS_OTHER.fetch_add(s.dependents_other, Ordering::Relaxed);
}

/// A release-cell run stalled for the first time with no client operation
/// in progress, so there was nothing to settle and the stall ended it.
pub fn record_stall_release_without_ops() {
    if !enabled() {
        return;
    }
    SR_STALLS_WITHOUT_OPS.fetch_add(1, Ordering::Relaxed);
}

/// A settled client operation's real response arrived after the release.
#[inline]
pub fn record_stall_release_late_response() {
    if !enabled() {
        return;
    }
    SR_LATE_RESPONSES.fetch_add(1, Ordering::Relaxed);
}

/// One run finished, of any cell and outcome. Called once per run, off the
/// scheduling hot path.
pub fn record_stall_release_run(r: &StallReleaseRun) {
    if !enabled() {
        return;
    }
    let (runs, invocations, plan_complete) = match r.cell {
        StallReleaseCell::Release => {
            (&SR_RELEASE_RUNS, &SR_RELEASE_INVOCATIONS, &SR_RELEASE_PLAN_COMPLETE)
        }
        StallReleaseCell::Cut => (&SR_CUT_RUNS, &SR_CUT_INVOCATIONS, &SR_CUT_PLAN_COMPLETE),
        StallReleaseCell::Exempt => return,
    };
    runs.fetch_add(1, Ordering::Relaxed);
    invocations.fetch_add(r.invocations, Ordering::Relaxed);
    plan_complete.fetch_add(r.completed as u64, Ordering::Relaxed);
    if let Some(at) = r.release_step {
        SR_STEPS_AFTER_RELEASE_SUM.fetch_add(r.steps.saturating_sub(at), Ordering::Relaxed);
        SR_PLAN_COMPLETED_AFTER_RELEASE.fetch_add(r.completed as u64, Ordering::Relaxed);
        SR_SECOND_STALL_STOPS.fetch_add(r.stalled as u64, Ordering::Relaxed);
    }
}

/// One placed-posture run drew a crash hold `held_steps` past the crash's
/// readiness; `capped` marks the draws whose span bound came from the step
/// cap's recovery reserve rather than the learned median.
#[inline]
pub fn record_crash_place_draw(capped: bool, held_steps: u64) {
    if !enabled() {
        return;
    }
    CRASH_PLACE_DRAWS.fetch_add(1, Ordering::Relaxed);
    if capped {
        CRASH_PLACE_CAPPED_DRAWS.fetch_add(1, Ordering::Relaxed);
    }
    CRASH_PLACE_HELD_STEPS_SUM.fetch_add(held_steps, Ordering::Relaxed);
}

/// One step offered a schedulable crash that an active crash-placement hold
/// excluded, counted per withheld node per step.
#[inline]
pub fn record_crash_place_hold() {
    if !enabled() {
        return;
    }
    CRASH_PLACE_HOLDS.fetch_add(1, Ordering::Relaxed);
}

/// The cell a run occupies under the ghost release: the placed runs whose
/// restarts arm the trigger, split into the half whose firing releases every
/// other held crash and the half that releases one; the placed runs that
/// never release; and the runs outside the placed posture, which no cell
/// counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GhostReleaseCell {
    #[default]
    Unplaced,
    Untreated,
    ReleaseAll,
    Single,
}

impl GhostReleaseCell {
    /// The column a placed cell counts under, or None outside the posture.
    #[inline]
    fn index(self) -> Option<usize> {
        match self {
            GhostReleaseCell::Unplaced => None,
            GhostReleaseCell::Untreated => Some(0),
            GhostReleaseCell::ReleaseAll => Some(1),
            GhostReleaseCell::Single => Some(2),
        }
    }
}

/// The lag learner's quantiles for one scope, in steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LagGauges {
    pub p50: u64,
    pub p75: u64,
    pub p90: u64,
}

/// One restart on a releasing run found some other node's planned crash
/// still held.
#[inline]
pub fn record_ghost_release_restart_with_held_crash() {
    if !enabled() {
        return;
    }
    GR_RESTARTS_WITH_HELD_CRASH.fetch_add(1, Ordering::Relaxed);
}

/// One ghost lag from a run that feeds the learner was folded into its
/// scope.
#[inline]
pub fn record_ghost_release_lag_sample() {
    if !enabled() {
        return;
    }
    GR_LAG_SAMPLES.fetch_add(1, Ordering::Relaxed);
}

/// Gauges, not counters: overwritten with the lag learner's current view,
/// ungated so it is visible with stats off.
pub fn set_ghost_release_learned(scopes_engaged: u64, lag: LagGauges) {
    GR_SCOPES_ENGAGED.store(scopes_engaged, Ordering::Relaxed);
    GR_LAG_P50.store(lag.p50, Ordering::Relaxed);
    GR_LAG_P75.store(lag.p75, Ordering::Relaxed);
    GR_LAG_P90.store(lag.p90, Ordering::Relaxed);
}

/// What happened to the trigger a restart arms on a releasing run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GhostReleaseTrigger {
    /// A restart found a held crash and armed the trigger.
    Armed,
    /// An entry from the restarted node's dead incarnation changed a live
    /// peer's state `steps_from_restart` steps after the restart and moved
    /// `released` holds to the next step.
    Fired { steps_from_restart: u64, released: u64 },
    /// Such an entry arrived but no hold was still ahead of the next step.
    FiredNothingHeld,
    /// The bound ended with no such entry.
    Expired,
    /// Another restart re-armed the trigger while it was still armed.
    Superseded,
}

#[inline]
pub fn record_ghost_release_trigger(event: GhostReleaseTrigger) {
    if !enabled() {
        return;
    }
    match event {
        GhostReleaseTrigger::Armed => {
            GR_ARMED.fetch_add(1, Ordering::Relaxed);
        }
        GhostReleaseTrigger::Fired {
            steps_from_restart,
            released,
        } => {
            GR_FIRED.fetch_add(1, Ordering::Relaxed);
            GR_STEPS_FROM_RESTART_SUM.fetch_add(steps_from_restart, Ordering::Relaxed);
            GR_RELEASED_CRASHES.fetch_add(released, Ordering::Relaxed);
        }
        GhostReleaseTrigger::FiredNothingHeld => {
            GR_FIRED_NOTHING_HELD.fetch_add(1, Ordering::Relaxed);
        }
        GhostReleaseTrigger::Expired => {
            GR_EXPIRED.fetch_add(1, Ordering::Relaxed);
        }
        GhostReleaseTrigger::Superseded => {
            GR_SUPERSEDED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Which crash a firing on the single half released, or that no case
/// applied and nothing was released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SingleRelease {
    /// The held crash of the node whose entry fired the trigger.
    OwnCrash,
    /// The first held crash the absorber ranking would move onto that node.
    ViaRanking,
    /// The first held crash by index, to be applied to that node.
    Forced,
    NoCase,
}

#[inline]
pub fn record_ghost_release_single(case: SingleRelease) {
    if !enabled() {
        return;
    }
    let c = match case {
        SingleRelease::OwnCrash => &GR_SINGLE_OWN_CRASH,
        SingleRelease::ViaRanking => &GR_SINGLE_VIA_RANKING,
        SingleRelease::Forced => &GR_SINGLE_FORCED,
        SingleRelease::NoCase => &GR_SINGLE_NO_CASE,
    };
    c.fetch_add(1, Ordering::Relaxed);
}

/// Where one applied crash landed. `later` marks a crash that is not the
/// run's first; `within_3_of_acted_ghost` one applied at most three steps
/// after the run's most recent fault-crossing entry that wrote state, read
/// on later crashes only; `fired` a crash a firing released, for which
/// `on_ghost_node` says it landed on the node whose entry fired the
/// trigger and `in_flight` counts the victim's undelivered sends;
/// `double_after_release` a crash applied within eight steps of a released
/// one whose victim is still down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GhostReleaseApply {
    pub later: bool,
    pub within_3_of_acted_ghost: bool,
    pub fired: bool,
    pub anchored: bool,
    pub retarget: bool,
    pub on_ghost_node: bool,
    pub in_flight: u32,
    pub double_after_release: bool,
}

/// One crash was applied on a run in `cell`.
#[inline]
pub fn record_ghost_release_apply(cell: GhostReleaseCell, a: GhostReleaseApply) {
    if !enabled() {
        return;
    }
    let Some(i) = cell.index() else {
        return;
    };
    GR_CELL_CRASHES_APPLIED[i].fetch_add(1, Ordering::Relaxed);
    if a.later {
        GR_CELL_LATER_CRASHES[i].fetch_add(1, Ordering::Relaxed);
        if a.within_3_of_acted_ghost {
            GR_CELL_WITHIN_3[i].fetch_add(1, Ordering::Relaxed);
        }
    }
    if a.double_after_release {
        GR_CELL_DOUBLE_CRASH[i].fetch_add(1, Ordering::Relaxed);
    }
    if !a.fired {
        return;
    }
    let phase = a.anchored as usize;
    let retarget = a.retarget as usize;
    GR_CELL_FIRED_APPLIED[i].fetch_add(1, Ordering::Relaxed);
    GR_CELL_FIRED_APPLIED_PHASE[i][phase].fetch_add(1, Ordering::Relaxed);
    GR_CELL_FIRED_APPLIED_RETARGET[i][retarget].fetch_add(1, Ordering::Relaxed);
    if a.within_3_of_acted_ghost {
        GR_CELL_FIRED_WITHIN_3[i].fetch_add(1, Ordering::Relaxed);
        GR_CELL_FIRED_WITHIN_3_PHASE[i][phase].fetch_add(1, Ordering::Relaxed);
    }
    if a.on_ghost_node {
        GR_CELL_FIRED_ON_GHOST_NODE[i].fetch_add(1, Ordering::Relaxed);
        GR_CELL_FIRED_ON_GHOST_NODE_RETARGET[i][retarget].fetch_add(1, Ordering::Relaxed);
    }
    let slot = (a.in_flight as usize).min(CC_INFLIGHT_SLOTS - 1);
    GR_CELL_FIRED_INFLIGHT[i][slot].fetch_add(1, Ordering::Relaxed);
}

/// One run in `cell` ended after `steps_used` steps.
#[inline]
pub fn record_ghost_release_run(cell: GhostReleaseCell, steps_used: u64) {
    if !enabled() {
        return;
    }
    let Some(i) = cell.index() else {
        return;
    };
    GR_CELL_RUNS[i].fetch_add(1, Ordering::Relaxed);
    GR_CELL_STEPS_USED_SUM[i].fetch_add(steps_used, Ordering::Relaxed);
}

/// Which phase of its victim's fan-out a placed crash's release waits for.
/// `Stock` waits for nothing and releases where it would have without the
/// anchor, so it is the control the other two are read against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashPhaseArm {
    /// Every send the victim's current handler segment issued is still
    /// undelivered.
    Early,
    /// The segment issued at least two sends, some delivered and some not.
    Mid,
    /// No wait.
    Stock,
}

impl CrashPhaseArm {
    /// Equal-mass arms, in the order a draw indexes them.
    pub const ALL: [CrashPhaseArm; 3] =
        [CrashPhaseArm::Early, CrashPhaseArm::Mid, CrashPhaseArm::Stock];

    #[inline]
    pub fn index(self) -> usize {
        match self {
            CrashPhaseArm::Early => 0,
            CrashPhaseArm::Mid => 1,
            CrashPhaseArm::Stock => 2,
        }
    }
}

/// One counter column per arm, or an arm would share another's column.
const _: () = assert!(CrashPhaseArm::ALL.len() == CP_ARMS);

/// How an anchored crash stopped being withheld.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashPhaseRelease {
    /// The arm asks for no wait, so the release is the draw.
    Immediate,
    /// The victim's fan-out reached the arm's phase.
    Condition,
    /// The window ran out with the phase unmet.
    Expired,
}

/// One crash drew `arm`; `first_in_run` marks the first draw of that arm in
/// its run, which is what the per-arm run count counts.
#[inline]
pub fn record_crash_phase_arm(arm: CrashPhaseArm, first_in_run: bool) {
    if !enabled() {
        return;
    }
    CP_ARMED[arm.index()].fetch_add(1, Ordering::Relaxed);
    if first_in_run {
        CP_RUNS[arm.index()].fetch_add(1, Ordering::Relaxed);
    }
}

/// One anchored crash's mask came off after `wait_steps` steps, with the
/// victim holding `victim_inflight` undelivered messages of its own at that
/// step. An immediate release waited for nothing, so it counts only as a
/// release decision.
#[inline]
pub fn record_crash_phase_release(
    arm: CrashPhaseArm,
    kind: CrashPhaseRelease,
    wait_steps: u64,
    victim_inflight: u32,
) {
    if !enabled() {
        return;
    }
    let i = arm.index();
    CP_RELEASE_DECISIONS[i].fetch_add(1, Ordering::Relaxed);
    if victim_inflight > 0 {
        CP_RELEASE_INFLIGHT[i].fetch_add(1, Ordering::Relaxed);
    }
    match kind {
        CrashPhaseRelease::Immediate => return,
        CrashPhaseRelease::Condition => {
            CP_ON_CONDITION[i].fetch_add(1, Ordering::Relaxed);
        }
        CrashPhaseRelease::Expired => {
            CP_EXPIRED[i].fetch_add(1, Ordering::Relaxed);
            if victim_inflight > 0 {
                CP_EXPIRED_INFLIGHT[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    CP_WAIT_STEPS_SUM[i].fetch_add(wait_steps, Ordering::Relaxed);
}

/// One crash whose release carried `arm` was applied to a node holding
/// `victim_inflight` undelivered messages of its own; `moved` means that
/// node is not the planned victim, and the crash is then counted again
/// under the arm's moved census. The census halves need the crash census
/// switched on, the same gate the unsplit census reads.
#[inline]
pub fn record_crash_phase_apply(arm: CrashPhaseArm, victim_inflight: u32, moved: bool) {
    if !enabled() {
        return;
    }
    let i = arm.index();
    CP_CRASHES_APPLIED[i].fetch_add(1, Ordering::Relaxed);
    if moved {
        CP_MOVED_APPLIED[i].fetch_add(1, Ordering::Relaxed);
    }
    if !crash_census_enabled() {
        return;
    }
    CP_APPLY_DECISIONS[i].fetch_add(1, Ordering::Relaxed);
    if victim_inflight > 0 {
        CP_APPLY_INFLIGHT[i].fetch_add(1, Ordering::Relaxed);
    }
    let slot = (victim_inflight as usize).min(CC_INFLIGHT_SLOTS - 1);
    CP_INFLIGHT[i][slot].fetch_add(1, Ordering::Relaxed);
    if moved {
        CP_MOVED_APPLY_DECISIONS[i].fetch_add(1, Ordering::Relaxed);
        if victim_inflight > 0 {
            CP_MOVED_APPLY_INFLIGHT[i].fetch_add(1, Ordering::Relaxed);
        }
        CP_MOVED_INFLIGHT[i][slot].fetch_add(1, Ordering::Relaxed);
    }
}

/// A once-per-crash event of the phase read on the node the retarget would
/// move a waiting crash to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashPhaseLanding {
    /// The phase was read on a node other than the planned victim at least
    /// once while the crash waited.
    EvaluatedOnOtherNode,
    /// The wait reached its phase with the read on another node.
    ConditionOnOtherNode,
    /// The window or the reserve ended the wait with the read on another
    /// node.
    ExpiredOnOtherNode,
    /// A crash released by its phase landed on a node other than the one
    /// the phase was last read on.
    MismatchAtApply,
}

impl CrashPhaseLanding {
    #[inline]
    fn index(self) -> usize {
        match self {
            CrashPhaseLanding::EvaluatedOnOtherNode => 0,
            CrashPhaseLanding::ConditionOnOtherNode => 1,
            CrashPhaseLanding::ExpiredOnOtherNode => 2,
            CrashPhaseLanding::MismatchAtApply => 3,
        }
    }
}

#[inline]
pub fn record_crash_phase_landing(event: CrashPhaseLanding) {
    if !enabled() {
        return;
    }
    CP_LANDING[event.index()].fetch_add(1, Ordering::Relaxed);
}

/// One steer-off probe-run timer firing was folded into the timer-context
/// learner; `acted` marks the subset that changed the node's state.
#[inline]
pub fn record_timer_context_probe(acted: bool) {
    if !enabled() {
        return;
    }
    TIMER_CONTEXT_PROBE_FIRINGS.fetch_add(1, Ordering::Relaxed);
    if acted {
        TIMER_CONTEXT_PROBE_ACTED.fetch_add(1, Ordering::Relaxed);
    }
}

/// One steered-run queue-selection roll applied an engaged context cell's
/// `multiplier` to the timer share.
#[inline]
pub fn record_timer_context_bias(multiplier: f64) {
    if !enabled() {
        return;
    }
    TIMER_CONTEXT_BIASED_STEPS.fetch_add(1, Ordering::Relaxed);
    if multiplier > 1.0 {
        TIMER_CONTEXT_BIASED_STEPS_PROMOTED.fetch_add(1, Ordering::Relaxed);
    } else if multiplier < 1.0 {
        TIMER_CONTEXT_BIASED_STEPS_SUPPRESSED.fetch_add(1, Ordering::Relaxed);
    }
}

/// An engaged context cell produced a multiplier but the step's queue
/// selector does not support the bias, so the stock roll ran instead.
#[inline]
pub fn record_timer_context_excluded() {
    if !enabled() {
        return;
    }
    TIMER_CONTEXT_STEPS_EXCLUDED_SELECTOR.fetch_add(1, Ordering::Relaxed);
}

/// Gauge, not a counter: overwritten with the timer-context learner's count
/// of cells at or over its sample floor, ungated so it is visible with
/// stats off.
pub fn set_timer_context_cells_engaged(cells: u64) {
    TIMER_CONTEXT_CELLS_ENGAGED.store(cells, Ordering::Relaxed);
}

/// One point on the timeline-key growth curve, covering `runs` consecutive
/// runs starting at `first_run`. `cumulative_distinct` is the running total of
/// keys ever inserted as of the last run in the point.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TimelineGrowthBucket {
    pub first_run: u64,
    pub runs: u64,
    pub keys_in_run_sum: u64,
    pub new_keys: u64,
    pub cumulative_distinct: u64,
}

/// Running totals over the coverage keys a session's runs produce.
struct TimelineKeyGrowth {
    runs: u64,
    keys_in_run_sum: u64,
    new_keys_sum: u64,
    distinct_keys_live: u64,
    buckets: Vec<TimelineGrowthBucket>,
}

impl TimelineKeyGrowth {
    const fn new() -> Self {
        Self {
            runs: 0,
            keys_in_run_sum: 0,
            new_keys_sum: 0,
            distinct_keys_live: 0,
            buckets: Vec::new(),
        }
    }
}

/// A run built its coverage keys with novelty turned off, so every ordering it
/// saw collapsed onto one key. Recorded once per run.
#[inline]
pub fn record_novelty_ablated_run() {
    if !enabled() {
        return;
    }
    NOVELTY_ABLATED_RUNS.fetch_add(1, Ordering::Relaxed);
}

/// One run's coverage keys were folded into the shared store: `keys_in_run` is
/// how many keys the run produced, `new_keys` how many of those the store had
/// never seen, and `distinct_keys_live` the store's size afterwards (which is
/// smaller than the cumulative total whenever the store has been decayed).
pub fn record_timeline_keys(keys_in_run: u64, new_keys: u64, distinct_keys_live: u64) {
    if !enabled() {
        return;
    }
    let Ok(mut g) = TIMELINE_KEYS.lock() else {
        return;
    };
    g.runs += 1;
    g.keys_in_run_sum += keys_in_run;
    g.new_keys_sum += new_keys;
    g.distinct_keys_live = distinct_keys_live;
    let first_run = g.runs;
    let cumulative = g.new_keys_sum;
    let start_new = match g.buckets.last() {
        Some(b) => b.runs >= TIMELINE_BUCKET_RUNS,
        None => true,
    };
    if start_new && g.buckets.len() < TIMELINE_MAX_BUCKETS {
        g.buckets.push(TimelineGrowthBucket {
            first_run,
            runs: 0,
            keys_in_run_sum: 0,
            new_keys: 0,
            cumulative_distinct: 0,
        });
    }
    if let Some(b) = g.buckets.last_mut() {
        b.runs += 1;
        b.keys_in_run_sum += keys_in_run;
        b.new_keys += new_keys;
        b.cumulative_distinct = cumulative;
    }
}

/// How fast the coverage-key space is still growing. `saturation_run_index` is
/// the run at which growth first fell below one new key per run over a whole
/// curve point, or 0 if it never did.
#[derive(Serialize)]
pub struct TimelineKeyStats {
    pub runs: u64,
    pub keys_in_run_sum: u64,
    pub cumulative_distinct_keys: u64,
    pub distinct_keys_live: u64,
    pub saturation_run_index: u64,
    pub bucket_runs: u64,
    /// Runs whose keys were built with novelty off. Nonzero means the coverage
    /// channel was ablated, and `distinct_keys_live` should then be 1.
    pub novelty_ablated_runs: u64,
    pub growth_curve: Vec<TimelineGrowthBucket>,
}

impl TimelineKeyStats {
    fn read() -> Self {
        let g = TIMELINE_KEYS
            .lock()
            .map(|g| {
                (
                    g.runs,
                    g.keys_in_run_sum,
                    g.new_keys_sum,
                    g.distinct_keys_live,
                    g.buckets.clone(),
                )
            })
            .unwrap_or_else(|p| {
                let g = p.into_inner();
                (
                    g.runs,
                    g.keys_in_run_sum,
                    g.new_keys_sum,
                    g.distinct_keys_live,
                    g.buckets.clone(),
                )
            });
        let (runs, keys_in_run_sum, new_keys_sum, distinct_keys_live, buckets) = g;
        let saturation_run_index = buckets
            .iter()
            .find(|b| b.runs >= TIMELINE_BUCKET_RUNS && b.new_keys < b.runs)
            .map(|b| b.first_run + b.runs - 1)
            .unwrap_or(0);
        Self {
            runs,
            keys_in_run_sum,
            cumulative_distinct_keys: new_keys_sum,
            distinct_keys_live,
            saturation_run_index,
            bucket_runs: TIMELINE_BUCKET_RUNS,
            novelty_ablated_runs: NOVELTY_ABLATED_RUNS.load(Ordering::Relaxed),
            growth_curve: buckets,
        }
    }
}

/// The curriculum lowered its knobs into one concrete run config. Zero here
/// means the curriculum was not on the path that produced these runs.
#[inline]
pub fn record_curriculum_lowering(num_crashes: i32, num_servers: i32) {
    if !enabled() {
        return;
    }
    CURRICULUM_LOWERED_RUNS.fetch_add(1, Ordering::Relaxed);
    CURRICULUM_CRASHES_SUM.fetch_add(num_crashes.max(0) as u64, Ordering::Relaxed);
    CURRICULUM_SERVERS_SUM.fetch_add(num_servers.max(0) as u64, Ordering::Relaxed);
}

#[derive(Serialize)]
pub struct CurriculumStats {
    pub lowered_runs: u64,
    pub crashes_sum: u64,
    pub servers_sum: u64,
}

/// How many runs drew their schedule from per-decision generators, and how
/// many drew from a single shared one.
#[derive(Serialize)]
pub struct RngStreamStats {
    pub isolated_runs: u64,
    pub shared_runs: u64,
}

#[derive(Serialize)]
pub struct SteerStats {
    pub evaluations: u64,
    pub divergent_picks: u64,
}

/// How often each stage of the scoring path was skipped because no predicate
/// carried weight. All zero means every stage ran, which is what a session with
/// a nonzero weight looks like.
#[derive(Serialize, Debug)]
pub struct EmptySliceStats {
    pub candidate_mask_skipped: u64,
    pub ranking_pass_skipped: u64,
    pub queue_audit_skipped: u64,
}

/// How often the runnable the scoring function ranked first is the one the
/// scheduling point ran, and what took precedence when it was not. The buckets
/// after `honored` partition the audited steps by the single constraint that
/// stood in the way, so all six sum to `audited`.
#[derive(Serialize)]
pub struct SteerAuthorityStats {
    /// Every budget step the session took, whether or not it went on to
    /// select anything. Zero here with steps used in the session means the
    /// counters are not wired up.
    pub steps_total: u64,
    /// The subset of `steps_total` that reached the point where the run's
    /// preference is read, audited or not. Equal to `steps_total` means no
    /// step stops before that point, so the decision site is on every step.
    pub steps: u64,
    /// The subset of `steps` where the ranking was resolved against what the
    /// step ran. Short of `steps` by the points the audit skipped, which
    /// `steer_empty_slice.queue_audit_skipped` counts.
    pub audited: u64,
    /// Steps where the steering term put a different runnable on top than
    /// priority alone would have. The denominator for `preference_honored`:
    /// on the other steps the audit cannot tell steer's choice from the
    /// choice the scheduler would have made without it.
    pub preference_expressed: u64,
    pub preference_honored: u64,
    /// Every read of a preference source, counted before the reader decides
    /// what to do with the answer. Several per step, so it is not comparable
    /// with `steps`; it is the denominator that says whether the decision
    /// sites execute at all, which `preference_expressed` alone cannot.
    pub preference_consulted: u64,
    /// The subset of `preference_consulted` where nothing was configured to
    /// have a preference. Equal to `preference_consulted` means the sites all
    /// ran and every one of them had no source to read.
    pub preference_source_absent: u64,
    pub honored: u64,
    pub no_eligible_candidates: u64,
    pub blocked_by_order: u64,
    pub blocked_by_timer_gate: u64,
    pub other_queue: u64,
    pub sampler_chose_other: u64,
}

/// Where the steps of a session stopped on the way to the point where the
/// scoring function can name a runnable priority alone would not have named.
/// The six buckets partition the steps that got as far as asking whether
/// anything was queued, so the last three summed are the steps that reached
/// that point and `preference_expressed` alone is the steps that used it.
#[derive(Serialize, Debug, Default, PartialEq, Eq)]
pub struct SteerReachStats {
    pub no_schedule_attempt: u64,
    pub audit_disabled: u64,
    pub no_weighted_predicate: u64,
    pub single_candidate: u64,
    pub ranking_agreed_with_priority: u64,
    pub preference_expressed: u64,
}

impl SteerReachStats {
    /// The steps that got far enough for the score and priority rankings to be
    /// compared, whatever the comparison then found.
    pub fn reached_decision(&self) -> u64 {
        self.single_candidate + self.ranking_agreed_with_priority + self.preference_expressed
    }
}

#[derive(Serialize)]
pub struct PurgatoryStats {
    pub delayed_sends: u64,
    /// Held sends whose destination node was crashed when the send was made.
    pub holds_down_receiver: u64,
    /// Held sends whose destination node was running when the send was made.
    pub holds_up_receiver: u64,
    /// Sends selected for a hold into a crashed destination and let through.
    pub passthrough_down_receiver: u64,
}

#[derive(Serialize)]
pub struct AosStats {
    pub tape_wins: u64,
    pub config_wins: u64,
}

#[derive(Serialize)]
pub struct DedupStats {
    pub checks: u64,
    pub hits: u64,
    pub skipped_large: u64,
}

#[derive(Serialize)]
pub struct FeedbackStats {
    pub timeline_score_sum: f64,
    pub cfg_score_sum: f64,
    pub scored_runs: u64,
}

/// How dense crash/recover activity is, and how often a message actually
/// survives a receiver's crash to be delivered after it comes back.
#[derive(Serialize)]
pub struct CrashRecoveryStats {
    /// Plan executions observed, i.e. the denominator for `runs_with_crossing`.
    pub runs: u64,
    pub crashes: u64,
    pub recovers: u64,
    /// Messages from another node that were queued to a node when it crashed
    /// and were kept for redelivery.
    pub messages_held_at_crash: u64,
    /// Work queued to a node at crash time that was thrown away instead:
    /// the node's own in-progress continuations and channel sends to it.
    pub messages_dropped_at_crash: u64,
    /// Held messages that were requeued to their target when it recovered,
    /// so they were sent to a live node, survived its downtime, and are
    /// handled by a different incarnation than the one they were sent to.
    pub crossing_deliveries: u64,
    pub runs_with_crossing: u64,
}

/// Record one timer firing that woke a waiting record and whether the woken
/// segment changed the node's state. Same switch as the delivery probe.
#[inline]
pub fn record_timer(key: TimerKey, acted: bool) {
    if !acted_fraction_enabled() {
        return;
    }
    bump(|b| &b.timers_fired, &TIMERS_FIRED, 1);
    if acted {
        bump(|b| &b.timers_acted, &TIMERS_ACTED, 1);
    }
    if key.inflight {
        bump(|b| &b.timers_inflight_fired, &TIMERS_INFLIGHT_FIRED, 1);
        if acted {
            bump(|b| &b.timers_inflight_acted, &TIMERS_INFLIGHT_ACTED, 1);
        }
    }
    let bucket = usize::from(key.inert_streak).min(STREAK_BUCKETS - 1);
    bump(
        |b| &b.timer_streak_fired[bucket],
        &TIMER_STREAK_FIRED[bucket],
        1,
    );
    if acted {
        bump(
            |b| &b.timer_streak_acted[bucket],
            &TIMER_STREAK_ACTED[bucket],
            1,
        );
    }
    if RUN_COUNTERS.with(|b| b.active.get()) {
        RUN_TIMER_EFFECTS.with(|m| {
            add_timer_effect(&mut m.borrow_mut(), key, 1, u64::from(acted));
        });
    } else if let Ok(mut t) = TIMER_EFFECTS.lock() {
        add_timer_effect(&mut t, key, 1, u64::from(acted));
    }
}

/// Adds firings to one key of a timer effect table. A key not yet present is
/// admitted only while the table is below its cap.
fn add_timer_effect(
    table: &mut HashMap<TimerKey, (u64, u64)>,
    key: TimerKey,
    fired: u64,
    acted: u64,
) {
    if let Some(e) = table.get_mut(&key) {
        e.0 += fired;
        e.1 += acted;
    } else if table.len() < TIMER_KEY_CAP {
        table.insert(key, (fired, acted));
    }
}

/// Timer firings and the share that changed the node's state, for one slice.
#[derive(Serialize)]
pub struct TimerEffect {
    pub fired: u64,
    pub acted: u64,
    pub acted_fraction: f64,
}

impl TimerEffect {
    fn of(fired: u64, acted: u64) -> Self {
        Self {
            fired,
            acted,
            acted_fraction: if fired == 0 { 0.0 } else { acted as f64 / fired as f64 },
        }
    }
}

/// One context key of the timer effect table with its counts.
#[derive(Serialize)]
pub struct TimerKeyEffect {
    pub vertex: usize,
    pub inflight: bool,
    pub incarnation: u8,
    pub inert_streak: u8,
    pub fired: u64,
    pub acted: u64,
}

/// Timer firings in the session: overall, with a delivery to the node in
/// flight, on an idle node, and per context key. Only firings that woke a
/// waiting record are counted; a firing whose channel had no reader is
/// consumed later by a receive and is not attributed.
#[derive(Serialize)]
pub struct TimerEffectStats {
    pub all: TimerEffect,
    pub with_inflight: TimerEffect,
    pub idle: TimerEffect,
    pub inert_streak: InertStreakHistogram,
    pub by_key: Vec<TimerKeyEffect>,
}

/// Timer firings grouped by how many firings at the same resume point on the
/// same node had changed nothing before this one: none, one or two, three to
/// seven, eight or more.
#[derive(Serialize)]
pub struct InertStreakHistogram {
    pub none: TimerEffect,
    pub short: TimerEffect,
    pub medium: TimerEffect,
    pub long: TimerEffect,
}

impl InertStreakHistogram {
    fn read() -> Self {
        let b = |i: usize| {
            TimerEffect::of(
                TIMER_STREAK_FIRED[i].load(Ordering::Relaxed),
                TIMER_STREAK_ACTED[i].load(Ordering::Relaxed),
            )
        };
        Self {
            none: b(0),
            short: b(1),
            medium: b(2),
            long: b(3),
        }
    }
}

/// Steps where admitting a timer was an actual choice, i.e. a timer and a
/// message delivery were both schedulable, and which of the two the step ran.
/// A mechanism that reweights timers against deliveries moves `raised` and
/// `lowered` against this denominator; with none configured the split is
/// whatever the queue selector draws.
#[derive(Serialize)]
pub struct TimerSteerStats {
    pub evaluated: u64,
    pub raised: u64,
    pub lowered: u64,
}

impl TimerSteerStats {
    fn read() -> Self {
        Self {
            evaluated: TIMER_STEER_EVALUATED.load(Ordering::Relaxed),
            raised: TIMER_STEER_RAISED.load(Ordering::Relaxed),
            lowered: TIMER_STEER_LOWERED.load(Ordering::Relaxed),
        }
    }
}

/// Record one step at which a timer and a delivery were both schedulable.
/// `chose_timer` says which one the step ran.
#[inline]
pub fn record_timer_admission(chose_timer: bool) {
    if !enabled() {
        return;
    }
    bump(|b| &b.timer_steer_evaluated, &TIMER_STEER_EVALUATED, 1);
    if chose_timer {
        bump(|b| &b.timer_steer_raised, &TIMER_STEER_RAISED, 1);
    } else {
        bump(|b| &b.timer_steer_lowered, &TIMER_STEER_LOWERED, 1);
    }
}

impl TimerEffectStats {
    fn read() -> Self {
        let fired = TIMERS_FIRED.load(Ordering::Relaxed);
        let acted = TIMERS_ACTED.load(Ordering::Relaxed);
        let inflight = TIMERS_INFLIGHT_FIRED.load(Ordering::Relaxed);
        let inflight_acted = TIMERS_INFLIGHT_ACTED.load(Ordering::Relaxed);
        let mut by_key: Vec<TimerKeyEffect> = TIMER_EFFECTS
            .lock()
            .map(|t| {
                t.iter()
                    .map(|(k, (f, a))| TimerKeyEffect {
                        vertex: k.vertex,
                        inflight: k.inflight,
                        incarnation: k.incarnation,
                        inert_streak: k.inert_streak,
                        fired: *f,
                        acted: *a,
                    })
                    .collect()
            })
            .unwrap_or_default();
        by_key.sort_by_key(|e| (e.vertex, e.inflight, e.incarnation, e.inert_streak));
        Self {
            all: TimerEffect::of(fired, acted),
            with_inflight: TimerEffect::of(inflight, inflight_acted),
            idle: TimerEffect::of(fired - inflight, acted - inflight_acted),
            inert_streak: InertStreakHistogram::read(),
            by_key,
        }
    }
}

/// Deliveries and the share of them that changed the receiving node's state,
/// for one bias bucket.
#[derive(Serialize)]
pub struct DeliveryEffect {
    pub deliveries: u64,
    pub acted: u64,
    pub acted_fraction: f64,
}

impl DeliveryEffect {
    fn read(bucket: usize) -> Self {
        let deliveries = DELIVERIES[bucket].load(Ordering::Relaxed);
        let acted = DELIVERIES_ACTED[bucket].load(Ordering::Relaxed);
        Self {
            deliveries,
            acted,
            acted_fraction: if deliveries == 0 {
                0.0
            } else {
                acted as f64 / deliveries as f64
            },
        }
    }
}

/// Deliveries and the share of them that changed the receiving node's state,
/// for one distance-from-restart bucket.
#[derive(Serialize)]
pub struct AcceptanceDistanceBucket {
    pub distance: &'static str,
    pub deliveries: u64,
    pub acted: u64,
    pub acted_fraction: f64,
}

/// The acted fraction as a function of how far the receiver had moved past its
/// own restart. A curve that falls off with distance means arrival position
/// decides whether a stale message is taken or absorbed; a flat curve means it
/// does not.
#[derive(Serialize)]
pub struct AcceptanceDistanceStats {
    pub all: Vec<AcceptanceDistanceBucket>,
    pub sender_restarted: Vec<AcceptanceDistanceBucket>,
    pub receiver_restarted: Vec<AcceptanceDistanceBucket>,
}

impl AcceptanceDistanceStats {
    fn read_path(path: usize) -> Vec<AcceptanceDistanceBucket> {
        (0..ACCEPT_DIST_BUCKETS)
            .map(|b| {
                let deliveries = ACCEPT_DIST[path][b].load(Ordering::Relaxed);
                let acted = ACCEPT_DIST_ACTED[path][b].load(Ordering::Relaxed);
                AcceptanceDistanceBucket {
                    distance: ACCEPT_DIST_LABELS[b],
                    deliveries,
                    acted,
                    acted_fraction: if deliveries == 0 {
                        0.0
                    } else {
                        acted as f64 / deliveries as f64
                    },
                }
            })
            .collect()
    }

    fn read() -> Self {
        Self {
            all: Self::read_path(ACCEPT_PATH_ALL),
            sender_restarted: Self::read_path(ACCEPT_PATH_SENDER_RESTARTED),
            receiver_restarted: Self::read_path(ACCEPT_PATH_RECEIVER_RESTARTED),
        }
    }
}

/// How often a message that reached a handler actually changed the receiver's
/// state, split by which perturbation the message was carrying. A bias whose
/// `acted_fraction` is near zero is being delivered but ignored, which is a
/// different failure than one whose `deliveries` is near zero.
///
/// The five flat fault tallies are the base rates of the crash-and-recover
/// predicates the delivery buckets are read against: how often a crash lands on
/// a node that still has a message of its own undelivered, how often a node
/// comes back with such a message still outstanding, and how many deliveries
/// then arrive from an incarnation that no longer exists. They are carried here
/// rather than beside the other fault counters so that a reader holding only
/// this block can compute them.
#[derive(Serialize)]
pub struct DeliveryEffectStats {
    pub all: DeliveryEffect,
    pub biased: DeliveryEffect,
    pub delayed: DeliveryEffect,
    pub sender_restarted: DeliveryEffect,
    pub receiver_restarted: DeliveryEffect,
    pub acceptance_distance: AcceptanceDistanceStats,
    pub crashes_total: u64,
    pub crashes_with_own_sends_inflight: u64,
    pub recoveries_total: u64,
    pub recoveries_with_own_prior_sends_inflight: u64,
    pub stale_sender_deliveries_after_recovery: u64,
    pub crash_census: CrashCensusStats,
}

/// What a crash lands on, counted over the crashes actually applied.
/// `decisions` is the denominator. `victim_had_inflight_sends` over it is how
/// often the node being crashed was already holding a message of its own that
/// nobody had received yet; `any_candidate_had_inflight_sends` asks the same of
/// the whole set of nodes whose crash was schedulable at that point, so the two
/// together bound what choosing the victim differently could buy. The histogram
/// gives the shape behind the first of those rates, since one undelivered
/// message and several are different situations.
#[derive(Serialize)]
pub struct CrashCensusStats {
    pub decisions: u64,
    pub victim_had_inflight_sends: u64,
    pub any_candidate_had_inflight_sends: u64,
    pub inflight_bucket_0: u64,
    pub inflight_bucket_1: u64,
    pub inflight_bucket_2: u64,
    pub inflight_bucket_3plus: u64,
}

impl CrashCensusStats {
    fn read() -> Self {
        Self {
            decisions: CC_DECISIONS.load(Ordering::Relaxed),
            victim_had_inflight_sends: CC_VICTIM_INFLIGHT.load(Ordering::Relaxed),
            any_candidate_had_inflight_sends: CC_ANY_CANDIDATE_INFLIGHT.load(Ordering::Relaxed),
            inflight_bucket_0: CC_INFLIGHT[0].load(Ordering::Relaxed),
            inflight_bucket_1: CC_INFLIGHT[1].load(Ordering::Relaxed),
            inflight_bucket_2: CC_INFLIGHT[2].load(Ordering::Relaxed),
            inflight_bucket_3plus: CC_INFLIGHT[3].load(Ordering::Relaxed),
        }
    }
}

/// How often a crash could be, and actually was, scheduled at the moment its
/// node had an undelivered message in the network. `offered` over
/// `steps_with_crash_eligible` says how often the situation arises at all;
/// `applied` over `crashes_taken` says how often the scheduler lands on it
/// without being pushed. `timing_bias_withheld` over `timing_bias_examined` is
/// the rate at which the crash-timing bias held a crash back from a step,
/// against the number of pending crashes it looked at.
#[derive(Serialize)]
pub struct CrashAnchorStats {
    pub steps_with_crash_eligible: u64,
    pub offered: u64,
    pub crashes_taken: u64,
    pub applied: u64,
    pub timing_bias_examined: u64,
    pub timing_bias_withheld: u64,
}

/// How wide the interval between a node's restart and the first message handed
/// to the incarnation that came back is, in scheduler steps. A window that is
/// only a step or two wide is one nothing else can be scheduled into.
#[derive(Serialize)]
pub struct RecoveryWindowStats {
    /// Windows that closed because a message reached the restarted node.
    pub count: u64,
    /// Restarts whose window never closed: the node crashed again, or the run
    /// ended, before anything was handed to it.
    pub unclosed: u64,
    pub mean_events_open: f64,
    pub p50: u64,
    pub p90: u64,
    pub max: u64,
    /// Widths at or above this are reported as this value.
    pub width_cap: u64,
}

impl RecoveryWindowStats {
    fn read() -> Self {
        let count = RW_CLOSED.load(Ordering::Relaxed);
        let hist = RW_WIDTHS
            .lock()
            .map(|w| *w)
            .unwrap_or_else(|p| *p.into_inner());
        Self {
            count,
            unclosed: RW_UNCLOSED.load(Ordering::Relaxed),
            mean_events_open: if count == 0 {
                0.0
            } else {
                RW_WIDTH_SUM.load(Ordering::Relaxed) as f64 / count as f64
            },
            p50: histogram_quantile(&hist, count, 0.5),
            p90: histogram_quantile(&hist, count, 0.9),
            max: RW_MAX.load(Ordering::Relaxed),
            width_cap: RW_WIDTH_CAP as u64,
        }
    }
}

fn histogram_quantile(hist: &[u64], total: u64, q: f64) -> u64 {
    if total == 0 {
        return 0;
    }
    let target = ((total as f64) * q).ceil().max(1.0) as u64;
    let mut seen = 0u64;
    for (width, &n) in hist.iter().enumerate() {
        seen += n;
        if seen >= target {
            return width as u64;
        }
    }
    (hist.len() - 1) as u64
}

/// Runs holding two distinct nodes' crash-and-recover cycles, split by whether
/// those cycles were merely both present or actually interleaved.
#[derive(Serialize)]
pub struct OrderedH3Stats {
    /// Runs folded into these tallies, the denominator for both rates.
    pub runs: u64,
    /// Runs in which two or more distinct nodes each crashed and came back.
    pub runs_with_h3: u64,
    /// The subset of those in which a crash landed while some other node's
    /// recovery window was still open, so the two cycles overlapped rather than
    /// running one after the other.
    pub runs_with_overlap: u64,
    /// The same three counts split by how many crash and recover events the run
    /// applied, index 0..8 with 8 meaning eight or more. The simulator has no
    /// view of the grader's prefix depth, so this is the severity axis it can
    /// report on its own.
    pub by_fault_events: Vec<OrderedH3Bucket>,
}

#[derive(Serialize)]
pub struct OrderedH3Bucket {
    pub runs: u64,
    pub runs_with_h3: u64,
    pub runs_with_overlap: u64,
}

impl OrderedH3Stats {
    fn read() -> Self {
        let table = OH3_BY_FAULT_EVENTS
            .lock()
            .map(|x| *x)
            .unwrap_or_else(|p| *p.into_inner());
        Self {
            runs: OH3_RUNS.load(Ordering::Relaxed),
            runs_with_h3: OH3_WITH_H3.load(Ordering::Relaxed),
            runs_with_overlap: OH3_WITH_OVERLAP.load(Ordering::Relaxed),
            by_fault_events: table
                .iter()
                .map(|b| OrderedH3Bucket {
                    runs: b[0],
                    runs_with_h3: b[1],
                    runs_with_overlap: b[2],
                })
                .collect(),
        }
    }
}

/// Whether the generator managed to reserve client work for after a fault, and
/// whether the reservation survived into execution. `edges_added` well below
/// `pairs_seen` times the configured count means the graph had no client
/// request left that could be ordered after a recover without closing a cycle;
/// `ops_invoked_after_last_recover` at zero means no run ever got there.
#[derive(Serialize)]
pub struct PostFaultOpsStats {
    pub pairs_seen: u64,
    pub edges_added: u64,
    pub ops_invoked_after_last_recover: u64,
}

/// Whether raising the weight of the priority term can outvote the random draw
/// priority is sampled from. `decisions` is every within-queue selection the
/// probe saw, `contested_decisions` those with more than one eligible
/// candidate, `quick_fire_offers` those holding a candidate the multiplier
/// applies to, and `quick_fire_decisions` the intersection, which is the only
/// place any magnitude can change the ranking and the denominator for every
/// flip count. `flipped_configured` is the probe's own firing count: the
/// selections the session's own magnitude reordered.
#[derive(Serialize)]
pub struct MultiplierAuthorityStats {
    pub decisions: u64,
    pub contested_decisions: u64,
    pub quick_fire_offers: u64,
    pub quick_fire_decisions: u64,
    pub mean_configured_multiplier: f64,
    pub flipped_configured: u64,
    pub sweep: Vec<MultiplierFlip>,
}

/// Selections whose top-ranked candidate under `multiplier` differs from the
/// one the identity weighting ranks first, out of `quick_fire_decisions`.
#[derive(Serialize)]
pub struct MultiplierFlip {
    pub multiplier: f64,
    pub flipped: u64,
}

impl MultiplierAuthorityStats {
    fn read() -> Self {
        let quick_fire_decisions = MA_QUICK_FIRE_DECISIONS.load(Ordering::Relaxed);
        Self {
            decisions: MA_DECISIONS.load(Ordering::Relaxed),
            contested_decisions: MA_CONTESTED_DECISIONS.load(Ordering::Relaxed),
            quick_fire_offers: MA_QUICK_FIRE_OFFERS.load(Ordering::Relaxed),
            quick_fire_decisions,
            mean_configured_multiplier: if quick_fire_decisions == 0 {
                0.0
            } else {
                f64::from_bits(MA_CONFIGURED_SUM.load(Ordering::Relaxed))
                    / quick_fire_decisions as f64
            },
            flipped_configured: MA_FLIPPED_CONFIGURED.load(Ordering::Relaxed),
            sweep: MULTIPLIER_SWEEP
                .iter()
                .zip(MA_FLIPPED.iter())
                .map(|(&multiplier, flipped)| MultiplierFlip {
                    multiplier,
                    flipped: flipped.load(Ordering::Relaxed),
                })
                .collect(),
        }
    }
}

/// What a recovery-reweighting term whose multiplier is the identity would
/// have preferred, counted the same way a weighted term is. `decisions` is
/// every selection the term walked; the rest are the term's own counters.
/// `flipped` cannot exceed zero while the multiplier is the identity, so a
/// nonzero value means the walk is not the inert copy it claims to be.
#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecoveryPlaceboStats {
    pub decisions: u64,
    pub evaluated: u64,
    pub present: u64,
    pub contested: u64,
    pub won: u64,
    pub flipped: u64,
}

impl RecoveryPlaceboStats {
    fn read() -> Self {
        Self {
            decisions: RWP_DECISIONS.load(Ordering::Relaxed),
            evaluated: RWP_EVALUATED.load(Ordering::Relaxed),
            present: RWP_PRESENT.load(Ordering::Relaxed),
            contested: RWP_CONTESTED.load(Ordering::Relaxed),
            won: RWP_WON.load(Ordering::Relaxed),
            flipped: RWP_FLIPPED.load(Ordering::Relaxed),
        }
    }
}

/// The learned-run-cap block: probe traffic, completions the cap would have
/// cut off, and gauges of what the learner currently holds.
#[derive(Serialize)]
pub struct RunCapStats {
    pub probes: u64,
    pub probe_completions: u64,
    pub over_cap_completions: u64,
    pub cap_recomputes: u64,
    pub scopes_learned: u64,
    pub current_cap_max_scope: u64,
}

impl RunCapStats {
    fn read() -> Self {
        Self {
            probes: RUN_CAP_PROBES.load(Ordering::Relaxed),
            probe_completions: RUN_CAP_PROBE_COMPLETIONS.load(Ordering::Relaxed),
            over_cap_completions: RUN_CAP_OVER_CAP_COMPLETIONS.load(Ordering::Relaxed),
            cap_recomputes: RUN_CAP_CAP_RECOMPUTES.load(Ordering::Relaxed),
            scopes_learned: RUN_CAP_SCOPES_LEARNED.load(Ordering::Relaxed),
            current_cap_max_scope: RUN_CAP_CURRENT_CAP_MAX_SCOPE.load(Ordering::Relaxed),
        }
    }
}

/// The stall-cap block. `stops` is the mechanism's firing count. The
/// `untreated_*` fields read the measured-but-never-cut quarter: how many
/// of its runs ended under a standing cap, how many of those had a longest
/// gap above it, and the log2 histogram of the longest gaps; the per-run
/// rows behind them are written beside the counters as a CSV.
#[derive(Serialize)]
pub struct StallCapStats {
    pub stops: u64,
    pub treated_runs: u64,
    pub untreated_runs: u64,
    pub steps_saved_sum: u64,
    pub suspended_steps_sum: u64,
    pub marks: StallCapMarks,
    pub probes_keyed: u64,
    pub probe_over_cap_completions: u64,
    pub scopes_learned: u64,
    pub cap_max_scope: u64,
    pub untreated_runs_capped: u64,
    pub untreated_over_cap_runs: u64,
    pub untreated_rows_dropped: u64,
    pub untreated_gap_hist: Vec<u64>,
}

impl StallCapStats {
    fn read() -> Self {
        Self {
            stops: SC_STOPS.load(Ordering::Relaxed),
            treated_runs: SC_TREATED_RUNS.load(Ordering::Relaxed),
            untreated_runs: SC_UNTREATED_RUNS.load(Ordering::Relaxed),
            steps_saved_sum: SC_STEPS_SAVED_SUM.load(Ordering::Relaxed),
            suspended_steps_sum: SC_SUSPENDED_STEPS_SUM.load(Ordering::Relaxed),
            marks: StallCapMarks {
                rows: SC_MARK_ROWS.load(Ordering::Relaxed),
                acted_deliveries: SC_MARK_ACTED_DELIVERIES.load(Ordering::Relaxed),
                acted_timers: SC_MARK_ACTED_TIMERS.load(Ordering::Relaxed),
                releases: SC_MARK_RELEASES.load(Ordering::Relaxed),
            },
            probes_keyed: SC_PROBES_KEYED.load(Ordering::Relaxed),
            probe_over_cap_completions: SC_PROBE_OVER_CAP_COMPLETIONS.load(Ordering::Relaxed),
            scopes_learned: SC_SCOPES_LEARNED.load(Ordering::Relaxed),
            cap_max_scope: SC_CAP_MAX_SCOPE.load(Ordering::Relaxed),
            untreated_runs_capped: SC_UNTREATED_RUNS_CAPPED.load(Ordering::Relaxed),
            untreated_over_cap_runs: SC_UNTREATED_OVER_CAP_RUNS.load(Ordering::Relaxed),
            untreated_rows_dropped: SC_UNTREATED_ROWS_DROPPED.load(Ordering::Relaxed),
            untreated_gap_hist: SC_UNTREATED_GAP_HIST
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect(),
        }
    }
}

/// Planned events a release made ready, by kind: client requests, faults
/// (crash, recover, partition, heal), and the rest.
#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StallReleaseDependents {
    pub client: u64,
    pub fault: u64,
    pub other: u64,
}

/// Runs of one stall-release cell, the ClientInterface invocations they
/// issued, and how many completed their plan.
#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StallReleaseCellStats {
    pub runs: u64,
    pub invocations: u64,
    pub plan_complete: u64,
}

/// The stall-release block. `releases` is the mechanism's firing count: runs
/// whose first stall settled their in-progress client operations instead of
/// ending them. `stalls_without_ops` counts release-cell runs whose first
/// stall found nothing to settle and ended them as a cut run's would. The
/// two cells are the stall cap's treated runs with and without the release.
#[derive(Serialize)]
pub struct StallReleaseStats {
    pub releases: u64,
    pub ops_settled: u64,
    pub dependents_released: StallReleaseDependents,
    pub late_responses: u64,
    pub plan_completed_after_release: u64,
    pub second_stall_stops: u64,
    pub steps_after_release_sum: u64,
    pub stalls_without_ops: u64,
    pub release_cell: StallReleaseCellStats,
    pub cut_cell: StallReleaseCellStats,
}

impl StallReleaseStats {
    fn read() -> Self {
        Self {
            releases: SR_RELEASES.load(Ordering::Relaxed),
            ops_settled: SR_OPS_SETTLED.load(Ordering::Relaxed),
            dependents_released: StallReleaseDependents {
                client: SR_DEPENDENTS_CLIENT.load(Ordering::Relaxed),
                fault: SR_DEPENDENTS_FAULT.load(Ordering::Relaxed),
                other: SR_DEPENDENTS_OTHER.load(Ordering::Relaxed),
            },
            late_responses: SR_LATE_RESPONSES.load(Ordering::Relaxed),
            plan_completed_after_release: SR_PLAN_COMPLETED_AFTER_RELEASE.load(Ordering::Relaxed),
            second_stall_stops: SR_SECOND_STALL_STOPS.load(Ordering::Relaxed),
            steps_after_release_sum: SR_STEPS_AFTER_RELEASE_SUM.load(Ordering::Relaxed),
            stalls_without_ops: SR_STALLS_WITHOUT_OPS.load(Ordering::Relaxed),
            release_cell: StallReleaseCellStats {
                runs: SR_RELEASE_RUNS.load(Ordering::Relaxed),
                invocations: SR_RELEASE_INVOCATIONS.load(Ordering::Relaxed),
                plan_complete: SR_RELEASE_PLAN_COMPLETE.load(Ordering::Relaxed),
            },
            cut_cell: StallReleaseCellStats {
                runs: SR_CUT_RUNS.load(Ordering::Relaxed),
                invocations: SR_CUT_INVOCATIONS.load(Ordering::Relaxed),
                plan_complete: SR_CUT_PLAN_COMPLETE.load(Ordering::Relaxed),
            },
        }
    }
}

/// The crash-placement block: holds drawn by placed-posture runs, the
/// subset whose span bound came from the step cap's recovery reserve, the
/// per-step offers an active hold excluded, and the summed displacement of
/// the drawn holds.
#[derive(Serialize)]
pub struct CrashPlaceStats {
    pub draws: u64,
    pub capped_draws: u64,
    pub holds: u64,
    pub held_steps_sum: u64,
    pub ghost_release: GhostReleaseStats,
}

impl CrashPlaceStats {
    fn read() -> Self {
        Self {
            draws: CRASH_PLACE_DRAWS.load(Ordering::Relaxed),
            capped_draws: CRASH_PLACE_CAPPED_DRAWS.load(Ordering::Relaxed),
            holds: CRASH_PLACE_HOLDS.load(Ordering::Relaxed),
            held_steps_sum: CRASH_PLACE_HELD_STEPS_SUM.load(Ordering::Relaxed),
            ghost_release: GhostReleaseStats::read(),
        }
    }
}

/// One cell of the ghost release. `runs` and `steps_used_sum` describe the
/// runs that ended in the cell; `crashes_applied` its crashes,
/// `later_crashes_applied` those after each run's first, and
/// `applied_within_3_of_acted_ghost` the later crashes applied at most
/// three steps after the run's most recent fault-crossing entry that wrote
/// state. The `fired_crashes_*` fields count the crashes a firing released
/// as they were applied: those within three steps of such an entry, those
/// landing on the node whose entry fired the trigger, each split by the
/// run's fan-out phase arm (anchored or not) or its crash-retarget arm, and
/// a histogram of the victim's undelivered sends at the apply.
/// `double_crash_after_release` counts crashes applied within eight steps
/// of a released one whose victim was still down.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostReleaseCellStats {
    pub runs: u64,
    pub steps_used_sum: u64,
    pub crashes_applied: u64,
    pub later_crashes_applied: u64,
    pub applied_within_3_of_acted_ghost: u64,
    pub fired_crashes_applied: u64,
    pub fired_crashes_applied_within_3: u64,
    pub fired_crashes_on_ghost_node: u64,
    pub fired_crashes_applied_anchored: u64,
    pub fired_crashes_applied_within_3_anchored: u64,
    pub fired_crashes_applied_unanchored: u64,
    pub fired_crashes_applied_within_3_unanchored: u64,
    pub fired_crashes_applied_retarget: u64,
    pub fired_crashes_on_ghost_node_retarget: u64,
    pub fired_crashes_applied_stock: u64,
    pub fired_crashes_on_ghost_node_stock: u64,
    pub fired_inflight_bucket_0: u64,
    pub fired_inflight_bucket_1: u64,
    pub fired_inflight_bucket_2: u64,
    pub fired_inflight_bucket_3plus: u64,
    pub double_crash_after_release: u64,
}

impl GhostReleaseCellStats {
    /// The columns in `cols` summed.
    fn read(cols: &[usize]) -> Self {
        let sum = |a: &[AtomicU64; GR_CELLS]| -> u64 {
            cols.iter().map(|&i| a[i].load(Ordering::Relaxed)).sum()
        };
        let sum2 = |a: &[[AtomicU64; 2]; GR_CELLS], j: usize| -> u64 {
            cols.iter().map(|&i| a[i][j].load(Ordering::Relaxed)).sum()
        };
        let sum_slot = |j: usize| -> u64 {
            cols.iter()
                .map(|&i| GR_CELL_FIRED_INFLIGHT[i][j].load(Ordering::Relaxed))
                .sum()
        };
        Self {
            runs: sum(&GR_CELL_RUNS),
            steps_used_sum: sum(&GR_CELL_STEPS_USED_SUM),
            crashes_applied: sum(&GR_CELL_CRASHES_APPLIED),
            later_crashes_applied: sum(&GR_CELL_LATER_CRASHES),
            applied_within_3_of_acted_ghost: sum(&GR_CELL_WITHIN_3),
            fired_crashes_applied: sum(&GR_CELL_FIRED_APPLIED),
            fired_crashes_applied_within_3: sum(&GR_CELL_FIRED_WITHIN_3),
            fired_crashes_on_ghost_node: sum(&GR_CELL_FIRED_ON_GHOST_NODE),
            fired_crashes_applied_anchored: sum2(&GR_CELL_FIRED_APPLIED_PHASE, 1),
            fired_crashes_applied_within_3_anchored: sum2(&GR_CELL_FIRED_WITHIN_3_PHASE, 1),
            fired_crashes_applied_unanchored: sum2(&GR_CELL_FIRED_APPLIED_PHASE, 0),
            fired_crashes_applied_within_3_unanchored: sum2(&GR_CELL_FIRED_WITHIN_3_PHASE, 0),
            fired_crashes_applied_retarget: sum2(&GR_CELL_FIRED_APPLIED_RETARGET, 1),
            fired_crashes_on_ghost_node_retarget: sum2(&GR_CELL_FIRED_ON_GHOST_NODE_RETARGET, 1),
            fired_crashes_applied_stock: sum2(&GR_CELL_FIRED_APPLIED_RETARGET, 0),
            fired_crashes_on_ghost_node_stock: sum2(&GR_CELL_FIRED_ON_GHOST_NODE_RETARGET, 0),
            fired_inflight_bucket_0: sum_slot(0),
            fired_inflight_bucket_1: sum_slot(1),
            fired_inflight_bucket_2: sum_slot(2),
            fired_inflight_bucket_3plus: sum_slot(3),
            double_crash_after_release: sum(&GR_CELL_DOUBLE_CRASH),
        }
    }
}

/// The ghost release's cells. `treated` is `release_all` and `single`
/// together, the half the release is read against `untreated`.
#[derive(Serialize, Debug)]
pub struct GhostReleaseCellsStats {
    pub untreated: GhostReleaseCellStats,
    pub release_all: GhostReleaseCellStats,
    pub single: GhostReleaseCellStats,
    pub treated: GhostReleaseCellStats,
}

/// The fields of a releasing cell the single half is read on.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostReleaseSingleCellStats {
    pub runs: u64,
    pub steps_used_sum: u64,
    pub crashes_applied: u64,
    pub fired_crashes_applied: u64,
    pub fired_crashes_on_ghost_node: u64,
    pub double_crash_after_release: u64,
}

impl GhostReleaseSingleCellStats {
    fn of(c: &GhostReleaseCellStats) -> Self {
        Self {
            runs: c.runs,
            steps_used_sum: c.steps_used_sum,
            crashes_applied: c.crashes_applied,
            fired_crashes_applied: c.fired_crashes_applied,
            fired_crashes_on_ghost_node: c.fired_crashes_on_ghost_node,
            double_crash_after_release: c.double_crash_after_release,
        }
    }
}

/// The two releasing cells as the single half is read: against the half
/// that releases every held crash.
#[derive(Serialize, Debug)]
pub struct GhostReleaseSingleCellsStats {
    pub release_all: GhostReleaseSingleCellStats,
    pub single: GhostReleaseSingleCellStats,
}

/// The single half's block. `releases` is the number that half is read as
/// having fired: firings that released exactly one crash, split by which
/// case named it; `no_release_case` counts the firings on which no case
/// applied and the trigger stayed armed.
#[derive(Serialize, Debug)]
pub struct GhostReleaseSingleStats {
    pub releases: u64,
    pub released_own_crash: u64,
    pub released_via_ranking: u64,
    pub released_forced: u64,
    pub no_release_case: u64,
    pub cells: GhostReleaseSingleCellsStats,
}

/// The ghost-release block. `fired` is the number the mechanism is read as
/// having fired: firings that moved at least one hold; `fired_nothing_held`
/// the firings that moved none; `armed`, `expired` and `superseded` how
/// the other armings ended; `steps_from_restart_sum` over `fired` how long
/// a firing took; `released_crashes` the holds the firings moved.
/// `restarts_with_held_crash` counts the restarts on releasing runs that
/// found another node's crash still held. `lag_samples` counts the ghost
/// lags fed to the learner; the `lag_p*` gauges are the widest engaged
/// scope's quantiles and `scopes_engaged` how many scopes are past their
/// floor.
#[derive(Serialize, Debug)]
pub struct GhostReleaseStats {
    pub armed: u64,
    pub fired: u64,
    pub fired_nothing_held: u64,
    pub expired: u64,
    pub superseded: u64,
    pub steps_from_restart_sum: u64,
    pub released_crashes: u64,
    pub restarts_with_held_crash: u64,
    pub lag_samples: u64,
    pub lag_p50: u64,
    pub lag_p75: u64,
    pub lag_p90: u64,
    pub scopes_engaged: u64,
    pub cells: GhostReleaseCellsStats,
    pub single: GhostReleaseSingleStats,
}

impl GhostReleaseStats {
    fn read() -> Self {
        let release_all = GhostReleaseCellStats::read(&[1]);
        let single = GhostReleaseCellStats::read(&[2]);
        let own = GR_SINGLE_OWN_CRASH.load(Ordering::Relaxed);
        let ranking = GR_SINGLE_VIA_RANKING.load(Ordering::Relaxed);
        let forced = GR_SINGLE_FORCED.load(Ordering::Relaxed);
        Self {
            armed: GR_ARMED.load(Ordering::Relaxed),
            fired: GR_FIRED.load(Ordering::Relaxed),
            fired_nothing_held: GR_FIRED_NOTHING_HELD.load(Ordering::Relaxed),
            expired: GR_EXPIRED.load(Ordering::Relaxed),
            superseded: GR_SUPERSEDED.load(Ordering::Relaxed),
            steps_from_restart_sum: GR_STEPS_FROM_RESTART_SUM.load(Ordering::Relaxed),
            released_crashes: GR_RELEASED_CRASHES.load(Ordering::Relaxed),
            restarts_with_held_crash: GR_RESTARTS_WITH_HELD_CRASH.load(Ordering::Relaxed),
            lag_samples: GR_LAG_SAMPLES.load(Ordering::Relaxed),
            lag_p50: GR_LAG_P50.load(Ordering::Relaxed),
            lag_p75: GR_LAG_P75.load(Ordering::Relaxed),
            lag_p90: GR_LAG_P90.load(Ordering::Relaxed),
            scopes_engaged: GR_SCOPES_ENGAGED.load(Ordering::Relaxed),
            cells: GhostReleaseCellsStats {
                untreated: GhostReleaseCellStats::read(&[0]),
                release_all,
                single,
                treated: GhostReleaseCellStats::read(&[1, 2]),
            },
            single: GhostReleaseSingleStats {
                releases: own + ranking + forced,
                released_own_crash: own,
                released_via_ranking: ranking,
                released_forced: forced,
                no_release_case: GR_SINGLE_NO_CASE.load(Ordering::Relaxed),
                cells: GhostReleaseSingleCellsStats {
                    release_all: GhostReleaseSingleCellStats::of(&release_all),
                    single: GhostReleaseSingleCellStats::of(&single),
                },
            },
        }
    }
}

/// One arm of the fan-out anchor. `armed` is the crashes the arm was drawn
/// for and `runs` the runs at least one of those draws fell in.
/// `released_on_condition` and `expired` split how the wait ended, and
/// `wait_steps_sum` over their sum is how long a wait lasted; all three stay
/// zero for an arm that waits for nothing. The release pair counts what the
/// victim held at the step its mask came off, the apply pair and the
/// histogram what it held at the step the crash was taken, which are
/// different questions whenever anything runs in between.
/// `expired_victim_had_inflight` is the part of the release pair that came
/// from a wait that ran out rather than one that reached its phase, so the
/// two kinds of release can be read apart: a wait that reached its phase
/// always ends with the victim holding something, since the segment sends it
/// counts are a subset of the sends still in flight.
#[derive(Serialize, Debug)]
pub struct CrashPhaseArmStats {
    pub runs: u64,
    pub armed: u64,
    pub released_on_condition: u64,
    pub expired: u64,
    pub wait_steps_sum: u64,
    pub release_decisions: u64,
    pub release_victim_had_inflight: u64,
    pub expired_victim_had_inflight: u64,
    pub apply_decisions: u64,
    pub apply_victim_had_inflight: u64,
    pub crashes_applied: u64,
    pub inflight_bucket_0: u64,
    pub inflight_bucket_1: u64,
    pub inflight_bucket_2: u64,
    pub inflight_bucket_3plus: u64,
    /// The crashes above that landed on another node than the planned
    /// victim.
    pub moved_read_on_landing: CrashPhaseMovedStats,
}

/// The apply-time census of one arm's crashes that landed on another node
/// than the planned victim.
#[derive(Serialize, Debug)]
pub struct CrashPhaseMovedStats {
    pub crashes_applied: u64,
    pub apply_decisions: u64,
    pub apply_victim_had_inflight: u64,
    pub inflight_bucket_0: u64,
    pub inflight_bucket_1: u64,
    pub inflight_bucket_2: u64,
    pub inflight_bucket_3plus: u64,
}

impl CrashPhaseMovedStats {
    fn read(arm: CrashPhaseArm) -> Self {
        let i = arm.index();
        Self {
            crashes_applied: CP_MOVED_APPLIED[i].load(Ordering::Relaxed),
            apply_decisions: CP_MOVED_APPLY_DECISIONS[i].load(Ordering::Relaxed),
            apply_victim_had_inflight: CP_MOVED_APPLY_INFLIGHT[i].load(Ordering::Relaxed),
            inflight_bucket_0: CP_MOVED_INFLIGHT[i][0].load(Ordering::Relaxed),
            inflight_bucket_1: CP_MOVED_INFLIGHT[i][1].load(Ordering::Relaxed),
            inflight_bucket_2: CP_MOVED_INFLIGHT[i][2].load(Ordering::Relaxed),
            inflight_bucket_3plus: CP_MOVED_INFLIGHT[i][3].load(Ordering::Relaxed),
        }
    }
}

/// Once-per-crash counts of phase reads on the node the retarget would
/// move a waiting crash to. `evaluated_on_other_node` is the number the
/// read is read as having fired: waiting crashes whose phase was read on
/// another node at least once. `condition_on_other_node` and
/// `expired_on_other_node` split those by how the wait ended, and
/// `mismatch_at_apply` counts the crashes released by their phase that
/// landed on a node other than the one the phase was last read on.
#[derive(Serialize, Debug)]
pub struct CrashPhaseLandingStats {
    pub evaluated_on_other_node: u64,
    pub condition_on_other_node: u64,
    pub expired_on_other_node: u64,
    pub mismatch_at_apply: u64,
}

impl CrashPhaseLandingStats {
    fn read() -> Self {
        let at = |e: CrashPhaseLanding| CP_LANDING[e.index()].load(Ordering::Relaxed);
        Self {
            evaluated_on_other_node: at(CrashPhaseLanding::EvaluatedOnOtherNode),
            condition_on_other_node: at(CrashPhaseLanding::ConditionOnOtherNode),
            expired_on_other_node: at(CrashPhaseLanding::ExpiredOnOtherNode),
            mismatch_at_apply: at(CrashPhaseLanding::MismatchAtApply),
        }
    }
}

impl CrashPhaseArmStats {
    fn read(arm: CrashPhaseArm) -> Self {
        let i = arm.index();
        Self {
            runs: CP_RUNS[i].load(Ordering::Relaxed),
            armed: CP_ARMED[i].load(Ordering::Relaxed),
            released_on_condition: CP_ON_CONDITION[i].load(Ordering::Relaxed),
            expired: CP_EXPIRED[i].load(Ordering::Relaxed),
            wait_steps_sum: CP_WAIT_STEPS_SUM[i].load(Ordering::Relaxed),
            release_decisions: CP_RELEASE_DECISIONS[i].load(Ordering::Relaxed),
            release_victim_had_inflight: CP_RELEASE_INFLIGHT[i].load(Ordering::Relaxed),
            expired_victim_had_inflight: CP_EXPIRED_INFLIGHT[i].load(Ordering::Relaxed),
            apply_decisions: CP_APPLY_DECISIONS[i].load(Ordering::Relaxed),
            apply_victim_had_inflight: CP_APPLY_INFLIGHT[i].load(Ordering::Relaxed),
            crashes_applied: CP_CRASHES_APPLIED[i].load(Ordering::Relaxed),
            inflight_bucket_0: CP_INFLIGHT[i][0].load(Ordering::Relaxed),
            inflight_bucket_1: CP_INFLIGHT[i][1].load(Ordering::Relaxed),
            inflight_bucket_2: CP_INFLIGHT[i][2].load(Ordering::Relaxed),
            inflight_bucket_3plus: CP_INFLIGHT[i][3].load(Ordering::Relaxed),
            moved_read_on_landing: CrashPhaseMovedStats::read(arm),
        }
    }
}

/// The fan-out anchor block: how many placed crashes were made to wait for a
/// phase of their victim's fan-out, and what they landed on. `armed` is the
/// two waiting arms together, the number this mechanism is read as having
/// fired; `stock_releases` is the third arm, drawn on the same runs and
/// released where it would have been anyway. `landing` counts the reads
/// made on the node a retargeting run would move the crash to.
#[derive(Serialize, Debug)]
pub struct CrashPhaseStats {
    pub armed: u64,
    pub stock_releases: u64,
    pub early: CrashPhaseArmStats,
    pub mid: CrashPhaseArmStats,
    pub stock: CrashPhaseArmStats,
    pub landing: CrashPhaseLandingStats,
}

impl CrashPhaseStats {
    fn read() -> Self {
        let early = CrashPhaseArmStats::read(CrashPhaseArm::Early);
        let mid = CrashPhaseArmStats::read(CrashPhaseArm::Mid);
        let stock = CrashPhaseArmStats::read(CrashPhaseArm::Stock);
        Self {
            armed: early.armed + mid.armed,
            stock_releases: stock.armed,
            early,
            mid,
            stock,
            landing: CrashPhaseLandingStats::read(),
        }
    }
}

/// One half of the crash census under the retarget split. `crashes` is the
/// denominator; `victim_had_absorbed` counts the crashes whose actual victim
/// had taken a fault-crossing delivery since its last restart, and
/// `victim_had_inflight_sends` those whose victim was holding an undelivered
/// send of its own.
#[derive(Serialize, Debug)]
pub struct VictimSwapHalfStats {
    pub crashes: u64,
    pub victim_had_absorbed: u64,
    pub victim_had_inflight_sends: u64,
}

impl VictimSwapHalfStats {
    fn read(treated: bool) -> Self {
        let i = treated as usize;
        Self {
            crashes: VS_CENSUS_CRASHES[i].load(Ordering::Relaxed),
            victim_had_absorbed: VS_CENSUS_ABSORBED[i].load(Ordering::Relaxed),
            victim_had_inflight_sends: VS_CENSUS_INFLIGHT[i].load(Ordering::Relaxed),
        }
    }
}

/// The crash census on each half of the retarget split.
#[derive(Serialize, Debug)]
pub struct VictimSwapCensusStats {
    pub treated: VictimSwapHalfStats,
    pub control: VictimSwapHalfStats,
}

/// The crash retarget block. `applied` is the number the mechanism is read
/// as having fired: planned crashes that landed on another node than the
/// plan's victim. `acted_absorber` is the subset whose new victim's mark had
/// written state. `same_victim` and `no_absorber` are the releases that kept
/// the victim, `skipped_pending_pair` the releases where a better-ranked
/// node was passed over for an outstanding pair, and
/// `victim_crashed_holds` the eligibility tests that withheld a planned
/// crash whose victim was already down. `forced_onto_absorber` is the
/// subset of `applied` landed by a ghost-release firing rather than the
/// ranking.
#[derive(Serialize, Debug)]
pub struct VictimSwapStats {
    pub applied: u64,
    pub acted_absorber: u64,
    pub same_victim: u64,
    pub no_absorber: u64,
    pub skipped_pending_pair: u64,
    pub victim_crashed_holds: u64,
    pub forced_onto_absorber: u64,
    pub census: VictimSwapCensusStats,
}

impl VictimSwapStats {
    fn read() -> Self {
        Self {
            applied: VS_APPLIED.load(Ordering::Relaxed),
            acted_absorber: VS_ACTED_ABSORBER.load(Ordering::Relaxed),
            same_victim: VS_SAME_VICTIM.load(Ordering::Relaxed),
            no_absorber: VS_NO_ABSORBER.load(Ordering::Relaxed),
            skipped_pending_pair: VS_SKIPPED_PENDING_PAIR.load(Ordering::Relaxed),
            victim_crashed_holds: VS_VICTIM_CRASHED_HOLDS.load(Ordering::Relaxed),
            forced_onto_absorber: VS_FORCED_ONTO_ABSORBER.load(Ordering::Relaxed),
            census: VictimSwapCensusStats {
                treated: VictimSwapHalfStats::read(true),
                control: VictimSwapHalfStats::read(false),
            },
        }
    }
}

/// Local call frames built by the interpreter. `calls` counts the frames,
/// `slots_built` sums their slot counts, and `entry_frame_copies` counts the
/// writes to a local frame that found it shared and therefore copied it.
#[derive(Serialize, Debug)]
pub struct FrameStats {
    pub calls: u64,
    pub slots_built: u64,
    pub entry_frame_copies: u64,
}

impl FrameStats {
    fn read() -> Self {
        Self {
            calls: FRAME_CALLS.load(Ordering::Relaxed),
            slots_built: FRAME_SLOTS_BUILT.load(Ordering::Relaxed),
            entry_frame_copies: FRAME_ENTRY_COPIES.load(Ordering::Relaxed),
        }
    }
}

/// Counter blocks folded into the session totals at run end, and the writes
/// they carried.
#[derive(Serialize, Debug)]
pub struct StatsLocalStats {
    pub folds: u64,
    pub folded_increments: u64,
}

impl StatsLocalStats {
    fn read() -> Self {
        Self {
            folds: STATS_LOCAL_FOLDS.load(Ordering::Relaxed),
            folded_increments: STATS_LOCAL_FOLDED_INCREMENTS.load(Ordering::Relaxed),
        }
    }
}

/// One run's frame counts, held on the running thread.
#[derive(Clone, Copy)]
struct FrameTally {
    calls: u64,
    slots_built: u64,
    entry_frame_copies: u64,
}

impl FrameTally {
    const fn new() -> Self {
        Self {
            calls: 0,
            slots_built: 0,
            entry_frame_copies: 0,
        }
    }
}

/// One local call frame of `slots` slots was built.
#[inline]
pub fn record_frame_build(slots: u64) {
    if !enabled() {
        return;
    }
    FRAME_RUN.with(|f| {
        let mut t = f.get();
        t.calls += 1;
        t.slots_built += slots;
        f.set(t);
    });
}

/// A write to a local call frame found it shared, so the write copied it.
#[inline]
pub fn record_entry_frame_copy() {
    if !enabled() {
        return;
    }
    FRAME_RUN.with(|f| {
        let mut t = f.get();
        t.entry_frame_copies += 1;
        f.set(t);
    });
}

/// Fold the running thread's frame counts into the session totals. Called
/// where a run ends, so the totals are exact once every run has ended.
pub fn flush_frame_stats() {
    if !enabled() {
        return;
    }
    let t = FRAME_RUN.with(|f| f.replace(FrameTally::new()));
    if t.calls != 0 {
        FRAME_CALLS.fetch_add(t.calls, Ordering::Relaxed);
        FRAME_SLOTS_BUILT.fetch_add(t.slots_built, Ordering::Relaxed);
    }
    if t.entry_frame_copies != 0 {
        FRAME_ENTRY_COPIES.fetch_add(t.entry_frame_copies, Ordering::Relaxed);
    }
}

/// One run's counters, held on the running thread. While `active` is false
/// every write goes straight to the session counter, so a thread that is not
/// inside a run never holds counts a snapshot cannot see.
struct RunCounters {
    active: Cell<bool>,
    /// The `STATS_GENERATION` the block was activated under.
    generation: Cell<u64>,
    /// Writes made to the block since it was activated.
    writes: Cell<u64>,
    steer_evaluations: Cell<u64>,
    steer_divergent_picks: Cell<u64>,
    sa_steps: Cell<u64>,
    sa_steps_total: Cell<u64>,
    sa_preference_consulted: Cell<u64>,
    sa_preference_source_absent: Cell<u64>,
    es_candidate_mask: Cell<u64>,
    es_ranking_pass: Cell<u64>,
    es_queue_audit: Cell<u64>,
    sr_no_schedule_attempt: Cell<u64>,
    sr_audit_disabled: Cell<u64>,
    sr_no_weighted_predicate: Cell<u64>,
    sr_single_candidate: Cell<u64>,
    sr_ranking_agreed: Cell<u64>,
    sr_preference_expressed: Cell<u64>,
    ma_decisions: Cell<u64>,
    ma_contested_decisions: Cell<u64>,
    ma_quick_fire_offers: Cell<u64>,
    ma_quick_fire_decisions: Cell<u64>,
    ma_flipped_configured: Cell<u64>,
    ma_flipped: [Cell<u64>; MULTIPLIER_SWEEP.len()],
    ma_configured_sum: Cell<f64>,
    rwp_decisions: Cell<u64>,
    rwp_evaluated: Cell<u64>,
    rwp_present: Cell<u64>,
    rwp_contested: Cell<u64>,
    rwp_won: Cell<u64>,
    rwp_flipped: Cell<u64>,
    ca_steps_with_crash_eligible: Cell<u64>,
    ca_offered: Cell<u64>,
    timer_steer_evaluated: Cell<u64>,
    timer_steer_raised: Cell<u64>,
    timer_steer_lowered: Cell<u64>,
    deliveries: [Cell<u64>; DELIVERY_BUCKETS],
    deliveries_acted: [Cell<u64>; DELIVERY_BUCKETS],
    accept_dist: [[Cell<u64>; ACCEPT_DIST_BUCKETS]; ACCEPT_PATHS],
    accept_dist_acted: [[Cell<u64>; ACCEPT_DIST_BUCKETS]; ACCEPT_PATHS],
    timers_fired: Cell<u64>,
    timers_acted: Cell<u64>,
    timers_inflight_fired: Cell<u64>,
    timers_inflight_acted: Cell<u64>,
    timer_streak_fired: [Cell<u64>; STREAK_BUCKETS],
    timer_streak_acted: [Cell<u64>; STREAK_BUCKETS],
}

impl RunCounters {
    const fn new() -> Self {
        Self {
            active: Cell::new(false),
            generation: Cell::new(0),
            writes: Cell::new(0),
            steer_evaluations: Cell::new(0),
            steer_divergent_picks: Cell::new(0),
            sa_steps: Cell::new(0),
            sa_steps_total: Cell::new(0),
            sa_preference_consulted: Cell::new(0),
            sa_preference_source_absent: Cell::new(0),
            es_candidate_mask: Cell::new(0),
            es_ranking_pass: Cell::new(0),
            es_queue_audit: Cell::new(0),
            sr_no_schedule_attempt: Cell::new(0),
            sr_audit_disabled: Cell::new(0),
            sr_no_weighted_predicate: Cell::new(0),
            sr_single_candidate: Cell::new(0),
            sr_ranking_agreed: Cell::new(0),
            sr_preference_expressed: Cell::new(0),
            ma_decisions: Cell::new(0),
            ma_contested_decisions: Cell::new(0),
            ma_quick_fire_offers: Cell::new(0),
            ma_quick_fire_decisions: Cell::new(0),
            ma_flipped_configured: Cell::new(0),
            ma_flipped: [const { Cell::new(0) }; MULTIPLIER_SWEEP.len()],
            ma_configured_sum: Cell::new(0.0),
            rwp_decisions: Cell::new(0),
            rwp_evaluated: Cell::new(0),
            rwp_present: Cell::new(0),
            rwp_contested: Cell::new(0),
            rwp_won: Cell::new(0),
            rwp_flipped: Cell::new(0),
            ca_steps_with_crash_eligible: Cell::new(0),
            ca_offered: Cell::new(0),
            timer_steer_evaluated: Cell::new(0),
            timer_steer_raised: Cell::new(0),
            timer_steer_lowered: Cell::new(0),
            deliveries: [const { Cell::new(0) }; DELIVERY_BUCKETS],
            deliveries_acted: [const { Cell::new(0) }; DELIVERY_BUCKETS],
            accept_dist: [const { [const { Cell::new(0) }; ACCEPT_DIST_BUCKETS] }; ACCEPT_PATHS],
            accept_dist_acted: [const { [const { Cell::new(0) }; ACCEPT_DIST_BUCKETS] };
                ACCEPT_PATHS],
            timers_fired: Cell::new(0),
            timers_acted: Cell::new(0),
            timers_inflight_fired: Cell::new(0),
            timers_inflight_acted: Cell::new(0),
            timer_streak_fired: [const { Cell::new(0) }; STREAK_BUCKETS],
            timer_streak_acted: [const { Cell::new(0) }; STREAK_BUCKETS],
        }
    }

    /// Every integer counter of the block paired with the session counter it
    /// folds into. A counter missing here would be zeroed without being
    /// folded.
    fn for_each_slot(&self, mut f: impl FnMut(&Cell<u64>, &AtomicU64)) {
        for (local, global) in [
            (&self.steer_evaluations, &STEER_EVALUATIONS),
            (&self.steer_divergent_picks, &STEER_DIVERGENT_PICKS),
            (&self.sa_steps, &SA_STEPS),
            (&self.sa_steps_total, &SA_STEPS_TOTAL),
            (&self.sa_preference_consulted, &SA_PREFERENCE_CONSULTED),
            (&self.sa_preference_source_absent, &SA_PREFERENCE_SOURCE_ABSENT),
            (&self.es_candidate_mask, &ES_CANDIDATE_MASK),
            (&self.es_ranking_pass, &ES_RANKING_PASS),
            (&self.es_queue_audit, &ES_QUEUE_AUDIT),
            (&self.sr_no_schedule_attempt, &SR_NO_SCHEDULE_ATTEMPT),
            (&self.sr_audit_disabled, &SR_AUDIT_DISABLED),
            (&self.sr_no_weighted_predicate, &SR_NO_WEIGHTED_PREDICATE),
            (&self.sr_single_candidate, &SR_SINGLE_CANDIDATE),
            (&self.sr_ranking_agreed, &SR_RANKING_AGREED),
            (&self.sr_preference_expressed, &SR_PREFERENCE_EXPRESSED),
            (&self.ma_decisions, &MA_DECISIONS),
            (&self.ma_contested_decisions, &MA_CONTESTED_DECISIONS),
            (&self.ma_quick_fire_offers, &MA_QUICK_FIRE_OFFERS),
            (&self.ma_quick_fire_decisions, &MA_QUICK_FIRE_DECISIONS),
            (&self.ma_flipped_configured, &MA_FLIPPED_CONFIGURED),
            (&self.rwp_decisions, &RWP_DECISIONS),
            (&self.rwp_evaluated, &RWP_EVALUATED),
            (&self.rwp_present, &RWP_PRESENT),
            (&self.rwp_contested, &RWP_CONTESTED),
            (&self.rwp_won, &RWP_WON),
            (&self.rwp_flipped, &RWP_FLIPPED),
            (&self.ca_steps_with_crash_eligible, &CA_STEPS_WITH_CRASH_ELIGIBLE),
            (&self.ca_offered, &CA_OFFERED),
            (&self.timer_steer_evaluated, &TIMER_STEER_EVALUATED),
            (&self.timer_steer_raised, &TIMER_STEER_RAISED),
            (&self.timer_steer_lowered, &TIMER_STEER_LOWERED),
            (&self.timers_fired, &TIMERS_FIRED),
            (&self.timers_acted, &TIMERS_ACTED),
            (&self.timers_inflight_fired, &TIMERS_INFLIGHT_FIRED),
            (&self.timers_inflight_acted, &TIMERS_INFLIGHT_ACTED),
        ] {
            f(local, global);
        }
        for (local, global) in self
            .ma_flipped
            .iter()
            .zip(MA_FLIPPED.iter())
            .chain(self.deliveries.iter().zip(DELIVERIES.iter()))
            .chain(self.deliveries_acted.iter().zip(DELIVERIES_ACTED.iter()))
            .chain(
                self.accept_dist
                    .iter()
                    .flatten()
                    .zip(ACCEPT_DIST.iter().flatten()),
            )
            .chain(
                self.accept_dist_acted
                    .iter()
                    .flatten()
                    .zip(ACCEPT_DIST_ACTED.iter().flatten()),
            )
            .chain(self.timer_streak_fired.iter().zip(TIMER_STREAK_FIRED.iter()))
            .chain(self.timer_streak_acted.iter().zip(TIMER_STREAK_ACTED.iter()))
        {
            f(local, global);
        }
    }
}

/// Adds `n` to one counter: to the running thread's block while a run is
/// active on it, otherwise to the session counter.
#[inline]
fn bump(local: impl FnOnce(&RunCounters) -> &Cell<u64>, global: &AtomicU64, n: u64) {
    RUN_COUNTERS.with(|b| {
        if b.active.get() {
            let c = local(b);
            c.set(c.get() + n);
            b.writes.set(b.writes.get() + 1);
        } else {
            global.fetch_add(n, Ordering::Relaxed);
        }
    });
}

/// Fold the running thread's counter block into the session totals and
/// deactivate it. Idempotent, so it can be called where a run ends, at the
/// start of the next run for runs that ended without reaching that point, and
/// before a snapshot. A block activated under an earlier session is emptied
/// without being added.
fn fold_run_counters() {
    let Some(current) = RUN_COUNTERS.with(|b| {
        if !b.active.get() {
            return None;
        }
        b.active.set(false);
        let writes = b.writes.replace(0);
        let current = b.generation.get() == STATS_GENERATION.load(Ordering::Relaxed);
        b.for_each_slot(|local, global| {
            let v = local.replace(0);
            if current && v != 0 {
                global.fetch_add(v, Ordering::Relaxed);
            }
        });
        let configured_sum = b.ma_configured_sum.replace(0.0);
        if current {
            if configured_sum != 0.0 {
                add_f64(&MA_CONFIGURED_SUM, configured_sum);
            }
            if writes > 0 {
                STATS_LOCAL_FOLDS.fetch_add(1, Ordering::Relaxed);
                STATS_LOCAL_FOLDED_INCREMENTS.fetch_add(writes, Ordering::Relaxed);
            }
        }
        Some(current)
    }) else {
        return;
    };
    RUN_TIMER_EFFECTS.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_empty() {
            return;
        }
        if current {
            if let Ok(mut t) = TIMER_EFFECTS.lock() {
                for (key, (fired, acted)) in m.drain() {
                    add_timer_effect(&mut t, key, fired, acted);
                }
            }
        }
        m.clear();
    });
}

/// Runs in which a fault-crossing delivery entered a node whose own crash
/// was queued: the situation the retarget exists to act on, counted on both
/// halves.
#[derive(Serialize, Debug)]
pub struct GhostSignalStats {
    pub fired_runs: u64,
}

impl GhostSignalStats {
    fn read() -> Self {
        Self {
            fired_runs: GS_FIRED_RUNS.load(Ordering::Relaxed),
        }
    }
}

/// One half of the contested-dispatch census. `contested_dispatches` counts
/// network steps that took a remote record while an eligible remote record
/// of the opposite incarnation class from the same sender to the same
/// destination was present; `stale_drawn` the subset whose draw fell on the
/// ghost, read before any swap, so its share of `contested_dispatches` is
/// the coin the draw flips between the two classes; `contested_down` the
/// stale draws whose destination was crashed, which no swap touches.
/// `ghost_entries_from_restarted_origin` counts message entries from a
/// sender's dead incarnation and `overtaken` those whose destination had
/// already taken an entry from the sender's current incarnation.
/// `ghost_entries_to_restarted_dest` and `overtaken_at_restarted_dest` are
/// the subsets whose destination has itself come back from a crash in the
/// run.
#[derive(Serialize, Debug)]
pub struct FreshFirstHalfStats {
    pub contested_dispatches: u64,
    pub stale_drawn: u64,
    pub contested_down: u64,
    pub ghost_entries_from_restarted_origin: u64,
    pub overtaken: u64,
    pub ghost_entries_to_restarted_dest: u64,
    pub overtaken_at_restarted_dest: u64,
}

impl FreshFirstHalfStats {
    fn read(treated: bool) -> Self {
        let i = treated as usize;
        Self {
            contested_dispatches: FF_CONTESTED[i].load(Ordering::Relaxed),
            stale_drawn: FF_STALE_DRAWN[i].load(Ordering::Relaxed),
            contested_down: FF_CONTESTED_DOWN[i].load(Ordering::Relaxed),
            ghost_entries_from_restarted_origin: FF_GHOST_ENTRIES[i].load(Ordering::Relaxed),
            overtaken: FF_OVERTAKEN[i].load(Ordering::Relaxed),
            ghost_entries_to_restarted_dest: FF_GHOST_ENTRIES_RESTARTED_DEST[i]
                .load(Ordering::Relaxed),
            overtaken_at_restarted_dest: FF_OVERTAKEN_RESTARTED_DEST[i].load(Ordering::Relaxed),
        }
    }
}

/// The census on each half of the fresh-first split.
#[derive(Serialize, Debug)]
pub struct FreshFirstCensusStats {
    pub treated: FreshFirstHalfStats,
    pub control: FreshFirstHalfStats,
}

/// The fresh-first dispatch block. `swaps` is the number the mechanism is
/// read as having fired: treated network steps whose draw fell on a ghost
/// with a live destination and an eligible fresh rival from the same sender
/// to the same destination, and that took the rival instead. It equals the
/// treated half's `stale_drawn` less its `contested_down`. `repeat_swaps` is
/// the subset whose ghost had been displaced before, and
/// `swap_count_hist_*` the distribution, over ghosts finally taken, of how
/// many times each had been displaced. `skipped_never_restarted_dest`
/// counts the swaps not made at a destination that has never restarted in
/// the run.
#[derive(Serialize, Debug)]
pub struct FreshFirstStats {
    pub swaps: u64,
    pub repeat_swaps: u64,
    pub skipped_never_restarted_dest: u64,
    pub swap_count_hist_1: u64,
    pub swap_count_hist_2: u64,
    pub swap_count_hist_3: u64,
    pub swap_count_hist_4plus: u64,
    pub census: FreshFirstCensusStats,
}

impl FreshFirstStats {
    fn read() -> Self {
        Self {
            swaps: FF_SWAPS.load(Ordering::Relaxed),
            repeat_swaps: FF_REPEAT_SWAPS.load(Ordering::Relaxed),
            skipped_never_restarted_dest: FF_SKIPPED_NEVER_RESTARTED_DEST.load(Ordering::Relaxed),
            swap_count_hist_1: FF_SWAP_COUNT_HIST[0].load(Ordering::Relaxed),
            swap_count_hist_2: FF_SWAP_COUNT_HIST[1].load(Ordering::Relaxed),
            swap_count_hist_3: FF_SWAP_COUNT_HIST[2].load(Ordering::Relaxed),
            swap_count_hist_4plus: FF_SWAP_COUNT_HIST[3].load(Ordering::Relaxed),
            census: FreshFirstCensusStats {
                treated: FreshFirstHalfStats::read(true),
                control: FreshFirstHalfStats::read(false),
            },
        }
    }
}

/// One half of the pair-order census. `contests` counts network steps whose
/// pick was a remote record from a sender that has crashed at least once in
/// the run while an eligible record of the same class - same sender, same
/// destination, same sending incarnation - was present; `inorder_draws` the
/// subset whose pick already carried the lowest send ordinal of its class
/// among the eligible records, read before any replacement, so its share of
/// `contests` is the coin the draw flips between send order and its
/// inversion. `pair_entries` counts message entries from such a sender whose
/// class has a sibling already entered or still in the network queue;
/// `inversions` the subset with a queued sibling of a lower send ordinal,
/// an entry the destination takes ahead of an earlier send.
///
/// `pair_entries_ghost` and `inversions_ghost` are the subsets of those two
/// whose entry was sent by an incarnation other than the one running now,
/// which covers every entry from a sender that is down; the rest of each
/// belongs to the sender's current incarnation. The preference acts on the
/// first subset alone, so `inversions_ghost` on the treated half is what it
/// removes.
///
/// `sampled_runs` counts the half's census runs. `pair_entries` and
/// `inversions` come from those runs alone on both halves, as do `contests`
/// and `inorder_draws` on the control half; the treated half counts its
/// contests on every treated run, so the counts of the two halves compare
/// as shares, not as totals.
#[derive(Serialize, Debug)]
pub struct PairOrderHalfStats {
    pub contests: u64,
    pub inorder_draws: u64,
    pub pair_entries: u64,
    pub pair_entries_ghost: u64,
    pub inversions: u64,
    pub inversions_ghost: u64,
    pub sampled_runs: u64,
}

impl PairOrderHalfStats {
    fn read(treated: bool) -> Self {
        let i = treated as usize;
        Self {
            contests: PO_CONTESTS[i].load(Ordering::Relaxed),
            inorder_draws: PO_INORDER_DRAWS[i].load(Ordering::Relaxed),
            pair_entries: PO_PAIR_ENTRIES[i].load(Ordering::Relaxed),
            pair_entries_ghost: PO_PAIR_ENTRIES_GHOST[i].load(Ordering::Relaxed),
            inversions: PO_INVERSIONS[i].load(Ordering::Relaxed),
            inversions_ghost: PO_INVERSIONS_GHOST[i].load(Ordering::Relaxed),
            sampled_runs: PO_SAMPLED_RUNS[i].load(Ordering::Relaxed),
        }
    }
}

/// The census on each half of the pair-order split.
#[derive(Serialize, Debug)]
pub struct PairOrderCensusStats {
    pub treated: PairOrderHalfStats,
    pub control: PairOrderHalfStats,
}

/// Contested treated steps by the incarnation class of the pick: `ghost` is
/// a pick whose sending incarnation is not the one running now, which covers
/// every record of a sender that is down, and `fresh` a pick of the sender's
/// current incarnation. Counted before any replacement, so the two sum to
/// the treated half's `contests`.
#[derive(Serialize, Debug)]
pub struct PairOrderClassCounts {
    pub ghost: u64,
    pub fresh: u64,
}

/// The pair-order dispatch block. `corrected` is the number the mechanism is
/// read as having fired: treated contests of the dead class whose pick did
/// not carry the lowest send ordinal of its class and that took the
/// lowest-ordinal record instead. `corrections_ghost` is that same count
/// split out, and `corrections_fresh` the audit that must read zero, since
/// only the dead class is replaced. `corrections_fresh_suppressed` counts
/// the replacements a rule covering every class would have made on the
/// sender's current incarnation, so its ratio to `corrected` is how much
/// firing the class test returns to the draw.
#[derive(Serialize, Debug)]
pub struct PairOrderStats {
    pub corrected: u64,
    pub contests_by_class: PairOrderClassCounts,
    pub corrections_ghost: u64,
    pub corrections_fresh: u64,
    pub corrections_fresh_suppressed: u64,
    pub census: PairOrderCensusStats,
}

impl PairOrderStats {
    fn read() -> Self {
        Self {
            corrected: PO_CORRECTED.load(Ordering::Relaxed),
            contests_by_class: PairOrderClassCounts {
                ghost: PO_CONTESTS_BY_CLASS[1].load(Ordering::Relaxed),
                fresh: PO_CONTESTS_BY_CLASS[0].load(Ordering::Relaxed),
            },
            corrections_ghost: PO_CORRECTIONS_BY_CLASS[1].load(Ordering::Relaxed),
            corrections_fresh: PO_CORRECTIONS_BY_CLASS[0].load(Ordering::Relaxed),
            corrections_fresh_suppressed: PO_FRESH_SUPPRESSED.load(Ordering::Relaxed),
            census: PairOrderCensusStats {
                treated: PairOrderHalfStats::read(true),
                control: PairOrderHalfStats::read(false),
            },
        }
    }
}

/// The client-anchor census on one half. `population` counts client requests
/// that became ready after the run's first crash; `fanout_windows` the steps
/// whose dispatch left a server that took an acted fault-crossing delivery
/// with a full fan-out in the air; `post_fault_invocations` the issues of
/// population requests and `in_window_invocations` those issued at the step
/// after a window opened. `runs` and `completed_runs` count the half's runs
/// and the ones whose plan completed.
#[derive(Serialize, Debug)]
pub struct ClientAnchorHalfStats {
    pub runs: u64,
    pub completed_runs: u64,
    pub population: u64,
    pub fanout_windows: u64,
    pub post_fault_invocations: u64,
    pub in_window_invocations: u64,
}

impl ClientAnchorHalfStats {
    fn read(treated: bool) -> Self {
        let i = treated as usize;
        Self {
            runs: CAN_RUNS[i].load(Ordering::Relaxed),
            completed_runs: CAN_COMPLETED_RUNS[i].load(Ordering::Relaxed),
            population: CAN_POPULATION[i].load(Ordering::Relaxed),
            fanout_windows: CAN_FANOUT_WINDOWS[i].load(Ordering::Relaxed),
            post_fault_invocations: CAN_POST_FAULT_INVOCATIONS[i].load(Ordering::Relaxed),
            in_window_invocations: CAN_IN_WINDOW_INVOCATIONS[i].load(Ordering::Relaxed),
        }
    }
}

/// The census on each half of the client-anchor split.
#[derive(Serialize, Debug)]
pub struct ClientAnchorCensusStats {
    pub treated: ClientAnchorHalfStats,
    pub control: ClientAnchorHalfStats,
}

/// Why held requests were issued on the treated half: `expiry` once the
/// fixed wait ran out, `dry_queue` when nothing else in the run could move.
#[derive(Serialize, Debug)]
pub struct ClientAnchorReleaseStats {
    pub expiry: u64,
    pub dry_queue: u64,
}

/// Requests held when a treated run's first window opened.
#[derive(Serialize, Debug)]
pub struct ClientAnchorHeldHist {
    pub zero: u64,
    pub one: u64,
    pub two: u64,
    pub three_plus: u64,
}

/// Runs on each direction of the post-crash request-timing axis.
#[derive(Serialize, Debug)]
pub struct ClientAnchorArmRuns {
    pub hold: u64,
    pub rush: u64,
    pub stock: u64,
}

/// The steps between a post-crash request's invocation and its first
/// delivery, summed over `count` requests.
#[derive(Serialize, Debug)]
pub struct ClientAnchorDistance {
    pub sum: u64,
    pub count: u64,
}

impl ClientAnchorDistance {
    fn read(arm: client_anchor::Arm) -> Self {
        let i = arm.index();
        Self {
            sum: CAN_FIRST_DELIVERY_SUM[i].load(Ordering::Relaxed),
            count: CAN_FIRST_DELIVERY_COUNT[i].load(Ordering::Relaxed),
        }
    }
}

/// The first-delivery distance on each direction, so the directions are
/// read against one another.
#[derive(Serialize, Debug)]
pub struct ClientAnchorFirstDelivery {
    pub hold: ClientAnchorDistance,
    pub rush: ClientAnchorDistance,
    pub stock: ClientAnchorDistance,
}

/// What the rush direction did. `ops` counts the post-crash requests it
/// issued and `records_prioritized` the records that took the top of the
/// priority range. `was_pick` and `displaced` split the network steps that
/// had a rushed record eligible by whether the step dispatched one, which
/// says how much of the top priority the eligibility and preference layers
/// left standing.
#[derive(Serialize, Debug)]
pub struct ClientAnchorRushStats {
    pub ops: u64,
    pub records_prioritized: u64,
    pub was_pick: u64,
    pub displaced: u64,
    pub first_delivery_distance: ClientAnchorFirstDelivery,
}

/// The post-crash request-timing axis: how many runs took each direction
/// and what the rush direction did.
#[derive(Serialize, Debug)]
pub struct ClientAnchorAxisStats {
    pub arm_runs: ClientAnchorArmRuns,
    pub rush: ClientAnchorRushStats,
}

impl ClientAnchorAxisStats {
    fn read() -> Self {
        let runs = |a: client_anchor::Arm| CAN_ARM_RUNS[a.index()].load(Ordering::Relaxed);
        Self {
            arm_runs: ClientAnchorArmRuns {
                hold: runs(client_anchor::Arm::Hold),
                rush: runs(client_anchor::Arm::Rush),
                stock: runs(client_anchor::Arm::Stock),
            },
            rush: ClientAnchorRushStats {
                ops: CAN_RUSH_OPS.load(Ordering::Relaxed),
                records_prioritized: CAN_RUSH_RECORDS_PRIORITIZED.load(Ordering::Relaxed),
                was_pick: CAN_RUSH_WAS_PICK.load(Ordering::Relaxed),
                displaced: CAN_RUSH_DISPLACED.load(Ordering::Relaxed),
                first_delivery_distance: ClientAnchorFirstDelivery {
                    hold: ClientAnchorDistance::read(client_anchor::Arm::Hold),
                    rush: ClientAnchorDistance::read(client_anchor::Arm::Rush),
                    stock: ClientAnchorDistance::read(client_anchor::Arm::Stock),
                },
            },
        }
    }
}

/// The step of the first message entry at a server caused by a post-fault
/// client operation, on the arm selector's coin-drawn runs, by the direction
/// the run carried in the order `run_variant::AXIS_START` gives: `runs`
/// counts the runs that had such an entry and `steps_sum` their steps, so
/// each entry of the two is one direction's mean.
#[derive(Serialize, Debug)]
pub struct ClientAnchorFirstEntry {
    pub runs: Vec<u64>,
    pub steps_sum: Vec<u64>,
}

/// The client-anchor block. `held` is the treated half's population and
/// `released` says why each held request was issued. `held_at_exit` sums
/// the requests still held when a run ended, over `runs_with_held_at_exit`
/// runs, and `hold_steps_sum` the steps every released request waited past
/// its ready step.
#[derive(Serialize, Debug)]
pub struct ClientAnchorStats {
    pub held: u64,
    pub released: ClientAnchorReleaseStats,
    pub held_at_first_firing: ClientAnchorHeldHist,
    pub held_at_exit: u64,
    pub runs_with_held_at_exit: u64,
    pub hold_steps_sum: u64,
    pub census: ClientAnchorCensusStats,
    pub axis: ClientAnchorAxisStats,
    pub first_post_fault_entry: ClientAnchorFirstEntry,
}

impl ClientAnchorStats {
    fn read() -> Self {
        let hist = |i: usize| CAN_HELD_AT_FIRST_FIRING[i].load(Ordering::Relaxed);
        Self {
            held: CAN_HELD.load(Ordering::Relaxed),
            released: ClientAnchorReleaseStats {
                expiry: CAN_RELEASED_EXPIRY.load(Ordering::Relaxed),
                dry_queue: CAN_RELEASED_DRY_QUEUE.load(Ordering::Relaxed),
            },
            held_at_first_firing: ClientAnchorHeldHist {
                zero: hist(0),
                one: hist(1),
                two: hist(2),
                three_plus: hist(3),
            },
            held_at_exit: CAN_HELD_AT_EXIT.load(Ordering::Relaxed),
            runs_with_held_at_exit: CAN_RUNS_WITH_HELD_AT_EXIT.load(Ordering::Relaxed),
            hold_steps_sum: CAN_HOLD_STEPS_SUM.load(Ordering::Relaxed),
            census: ClientAnchorCensusStats {
                treated: ClientAnchorHalfStats::read(true),
                control: ClientAnchorHalfStats::read(false),
            },
            axis: ClientAnchorAxisStats::read(),
            first_post_fault_entry: ClientAnchorFirstEntry {
                runs: CAN_FIRST_ENTRY_RUNS
                    .iter()
                    .map(|n| n.load(Ordering::Relaxed))
                    .collect(),
                steps_sum: CAN_FIRST_ENTRY_STEPS_SUM
                    .iter()
                    .map(|n| n.load(Ordering::Relaxed))
                    .collect(),
            },
        }
    }
}

/// One reward read on the arm selector's coin-drawn runs.
/// `reward_runs_control` and `reward_positive_control` are the coin-drawn
/// runs and rewards; the per-direction and per-combination arrays follow
/// `run_variant::AXIS_START` and `ArmSet::index` and say what each coin
/// direction earned. The `by_arm_direction` arrays split the per-direction
/// pair by campaign arm: row `arm_index + 1` of eight, twelve directions
/// per row, flattened row-major, so entry `(arm_index + 1) * 12 + d` is
/// direction `d` on that arm.
#[derive(Serialize, Debug, Clone)]
pub struct ArmSelectorRewardStats {
    pub reward_runs_control: u64,
    pub reward_positive_control: u64,
    pub control_runs_by_direction: Vec<u64>,
    pub control_reward_positive_by_direction: Vec<u64>,
    pub control_runs_by_arm_direction: Vec<u64>,
    pub control_reward_positive_by_arm_direction: Vec<u64>,
    pub control_runs_by_combination: Vec<u64>,
    pub control_reward_positive_by_combination: Vec<u64>,
}

impl ArmSelectorRewardStats {
    fn read(reward: Reward) -> Self {
        let ax = &AX[reward.index()];
        let load = |a: &[AtomicU64]| a.iter().map(|n| n.load(Ordering::Relaxed)).collect();
        Self {
            reward_runs_control: ax.reward_runs[0].load(Ordering::Relaxed),
            reward_positive_control: ax.reward_positive[0].load(Ordering::Relaxed),
            control_runs_by_direction: load(&ax.control_runs_by_direction),
            control_reward_positive_by_direction: load(&ax.control_reward_positive_by_direction),
            control_runs_by_arm_direction: load(&ax.control_runs_by_arm_direction),
            control_reward_positive_by_arm_direction: load(
                &ax.control_reward_positive_by_arm_direction,
            ),
            control_runs_by_combination: load(&ax.control_runs_by_combination),
            control_reward_positive_by_combination: load(
                &ax.control_reward_positive_by_combination,
            ),
        }
    }
}

/// One reward a learner is trained on, with that learner's runs.
/// `chosen_runs` counts the learner runs, every one drawn from a cell past
/// its warmup; `axis_leader_agreements / axis_draws` sits well above the
/// share independent draws would give when the learner ran as intended.
/// `departures` counts learner runs whose set differs from the id's coins;
/// `cells` is a gauge of the learner's table. `reward_runs_treated` and
/// `reward_positive_treated` are the learner runs; the control fields are
/// the coin-drawn runs, and `chosen_by_*` what the learner runs took.
/// `leader_margin_micro` is, per axis, the sum in millionths over the
/// learner runs of the pairwise probability that the axis's highest-mean
/// direction leads the next; divided by `chosen_runs` it is the mean
/// margin the pick saw. The `by_arm_direction` arrays are laid out as in
/// `ArmSelectorRewardStats`.
#[derive(Serialize, Debug, Clone)]
pub struct ArmSelectorLearnerStats {
    pub chosen_runs: u64,
    pub departures: u64,
    pub axis_leader_agreements: u64,
    pub axis_draws: u64,
    pub chosen_placed_runs: u64,
    pub cells: u64,
    pub reward_runs_treated: u64,
    pub reward_positive_treated: u64,
    pub reward_runs_control: u64,
    pub reward_positive_control: u64,
    pub chosen_by_direction: Vec<u64>,
    pub leader_margin_micro: Vec<u64>,
    pub control_runs_by_direction: Vec<u64>,
    pub control_reward_positive_by_direction: Vec<u64>,
    pub control_runs_by_arm_direction: Vec<u64>,
    pub control_reward_positive_by_arm_direction: Vec<u64>,
    pub chosen_by_combination: Vec<u64>,
    pub control_runs_by_combination: Vec<u64>,
    pub control_reward_positive_by_combination: Vec<u64>,
}

impl ArmSelectorLearnerStats {
    fn read(reward: Reward) -> Self {
        let ax = &AX[reward.index()];
        let load = |a: &[AtomicU64]| a.iter().map(|n| n.load(Ordering::Relaxed)).collect();
        Self {
            chosen_runs: ax.chosen_runs.load(Ordering::Relaxed),
            departures: ax.departures.load(Ordering::Relaxed),
            axis_leader_agreements: ax.axis_leader_agreements.load(Ordering::Relaxed),
            axis_draws: ax.axis_draws.load(Ordering::Relaxed),
            chosen_placed_runs: ax.chosen_placed_runs.load(Ordering::Relaxed),
            cells: ax.cells.load(Ordering::Relaxed),
            reward_runs_treated: ax.reward_runs[1].load(Ordering::Relaxed),
            reward_positive_treated: ax.reward_positive[1].load(Ordering::Relaxed),
            reward_runs_control: ax.reward_runs[0].load(Ordering::Relaxed),
            reward_positive_control: ax.reward_positive[0].load(Ordering::Relaxed),
            chosen_by_direction: load(&ax.chosen_by_direction),
            leader_margin_micro: load(&ax.leader_margin_micro),
            control_runs_by_direction: load(&ax.control_runs_by_direction),
            control_reward_positive_by_direction: load(&ax.control_reward_positive_by_direction),
            control_runs_by_arm_direction: load(&ax.control_runs_by_arm_direction),
            control_reward_positive_by_arm_direction: load(
                &ax.control_reward_positive_by_arm_direction,
            ),
            chosen_by_combination: load(&ax.chosen_by_combination),
            control_runs_by_combination: load(&ax.control_runs_by_combination),
            control_reward_positive_by_combination: load(
                &ax.control_reward_positive_by_combination,
            ),
        }
    }
}

/// The arm selector's exploration draws. `draws` counts every run assigned
/// to a learner, `coin_runs` those that came out coin-drawn, and
/// `warmup_coin_runs` the coin-drawn runs of cells below warmup, where the
/// share is one. `share_micro` sums the exploration share in millionths
/// over the draws, so `share_micro / draws` is the mean coin share;
/// `margin_micro` sums, over the draws past warmup, the highest per-axis
/// leader margin the share was read from. The `by_learner` arrays follow
/// `arm_selector::Learner::ALL`, so `share_micro_by_learner[l] /
/// draws_by_learner[l]` is learner `l`'s own mean share; the `by_arm`
/// arrays use slot `arm_index + 1`, so `coin_runs_by_arm / draws_by_arm` is
/// the coin share per campaign arm and `margin_micro_by_arm / (draws_by_arm
/// - the arm's warmup draws)` its mean top margin past warmup;
/// `share_hist` bins the share at draw time into ten equal tenths, a share
/// of one in the last.
#[derive(Serialize, Debug, Clone)]
pub struct ArmSelectorExploreStats {
    pub draws: u64,
    pub coin_runs: u64,
    pub warmup_coin_runs: u64,
    pub share_micro: u64,
    pub margin_micro: u64,
    pub draws_by_learner: Vec<u64>,
    pub coin_runs_by_learner: Vec<u64>,
    pub share_micro_by_learner: Vec<u64>,
    pub draws_by_arm: Vec<u64>,
    pub coin_runs_by_arm: Vec<u64>,
    pub margin_micro_by_arm: Vec<u64>,
    pub share_hist: Vec<u64>,
}

impl ArmSelectorExploreStats {
    fn read() -> Self {
        let e = &AX_EXPLORE;
        let load = |a: &[AtomicU64]| a.iter().map(|n| n.load(Ordering::Relaxed)).collect();
        Self {
            draws: e.draws.load(Ordering::Relaxed),
            coin_runs: e.coin_runs.load(Ordering::Relaxed),
            warmup_coin_runs: e.warmup_coin_runs.load(Ordering::Relaxed),
            share_micro: e.share_micro.load(Ordering::Relaxed),
            margin_micro: e.margin_micro.load(Ordering::Relaxed),
            draws_by_learner: load(&e.draws_by_learner),
            coin_runs_by_learner: load(&e.coin_runs_by_learner),
            share_micro_by_learner: load(&e.share_micro_by_learner),
            draws_by_arm: load(&e.draws_by_arm),
            coin_runs_by_arm: load(&e.coin_runs_by_arm),
            margin_micro_by_arm: load(&e.margin_micro_by_arm),
            share_hist: load(&e.share_hist),
        }
    }
}

/// The arm selector's cells keyed by a configuration and shared across
/// campaign arms. `draws` counts the draws served from such a cell;
/// `cells` is a gauge of the first learner's shared cells and
/// `cells_by_learner` the same for each learner in
/// `arm_selector::Learner::ALL` order; `observations_by_learner` counts
/// the runs each learner credited into a shared cell, so divided by that
/// learner's `cells_by_learner` it is the mean observations per shared
/// cell.
#[derive(Serialize, Debug, Clone)]
pub struct ArmSelectorPooledStats {
    pub draws: u64,
    pub cells: u64,
    pub cells_by_learner: Vec<u64>,
    pub observations_by_learner: Vec<u64>,
}

impl ArmSelectorPooledStats {
    fn read() -> Self {
        let cells_by_learner: Vec<u64> =
            [Reward::OvertakenGhost, Reward::AbsorberCycle, Reward::CycleBeforeRequest]
                .iter()
                .map(|r| AX[r.index()].pooled_cells.load(Ordering::Relaxed))
                .collect();
        Self {
            draws: AX_POOLED.draws.load(Ordering::Relaxed),
            cells: cells_by_learner[0],
            cells_by_learner,
            observations_by_learner: AX_POOLED
                .observations_by_learner
                .iter()
                .map(|n| n.load(Ordering::Relaxed))
                .collect(),
        }
    }
}

/// The arm selector block. The top-level fields are the first learner's,
/// the one trained on the overtaken-ghost reward, and repeat under
/// `overtaken_ghost`; the second learner's are under `absorber_cycle` and
/// the third's under `cycle_before_request`. `ghost_signal`,
/// `either_shape`, `mutual_absorber_cycle` and `exchange_before_request`
/// are read on the coin-drawn runs and train nothing. `explore` is the
/// exploration draw that splits the assigned runs into coin-drawn runs and
/// learner runs. `observations` counts every run the selector observed,
/// coin-drawn or learner run; `reward_runs_by_arm` and
/// `reward_positive_by_arm` split the first learner's reward over those
/// runs by campaign arm, slot `arm_index + 1`, so a young arm's rate can be
/// read against a mature one. `pooled` is the traffic on the cells shared
/// across campaign arms; `cells` counts every cell, shared or not.
#[derive(Serialize, Debug, Clone)]
pub struct ArmSelectorAxisStats {
    pub chosen_runs: u64,
    pub departures: u64,
    pub axis_leader_agreements: u64,
    pub axis_draws: u64,
    pub observations: u64,
    pub reward_runs_treated: u64,
    pub reward_positive_treated: u64,
    pub reward_runs_control: u64,
    pub reward_positive_control: u64,
    pub cells: u64,
    pub chosen_placed_runs: u64,
    pub reward_runs_by_arm: Vec<u64>,
    pub reward_positive_by_arm: Vec<u64>,
    pub chosen_by_direction: Vec<u64>,
    pub leader_margin_micro: Vec<u64>,
    pub control_runs_by_direction: Vec<u64>,
    pub control_reward_positive_by_direction: Vec<u64>,
    pub control_runs_by_arm_direction: Vec<u64>,
    pub control_reward_positive_by_arm_direction: Vec<u64>,
    pub chosen_by_combination: Vec<u64>,
    pub control_runs_by_combination: Vec<u64>,
    pub control_reward_positive_by_combination: Vec<u64>,
    pub explore: ArmSelectorExploreStats,
    pub pooled: ArmSelectorPooledStats,
    pub overtaken_ghost: ArmSelectorLearnerStats,
    pub absorber_cycle: ArmSelectorLearnerStats,
    pub cycle_before_request: ArmSelectorLearnerStats,
    pub ghost_signal: ArmSelectorRewardStats,
    pub either_shape: ArmSelectorRewardStats,
    pub mutual_absorber_cycle: ArmSelectorRewardStats,
    pub exchange_before_request: ArmSelectorRewardStats,
}

impl ArmSelectorAxisStats {
    fn read() -> Self {
        let load = |a: &[AtomicU64]| a.iter().map(|n| n.load(Ordering::Relaxed)).collect();
        let a = ArmSelectorLearnerStats::read(Reward::OvertakenGhost);
        Self {
            chosen_runs: a.chosen_runs,
            departures: a.departures,
            axis_leader_agreements: a.axis_leader_agreements,
            axis_draws: a.axis_draws,
            observations: AX_OBSERVATIONS.load(Ordering::Relaxed),
            reward_runs_treated: a.reward_runs_treated,
            reward_positive_treated: a.reward_positive_treated,
            reward_runs_control: a.reward_runs_control,
            reward_positive_control: a.reward_positive_control,
            cells: a.cells,
            chosen_placed_runs: a.chosen_placed_runs,
            reward_runs_by_arm: load(&AX_REWARD_RUNS_BY_ARM),
            reward_positive_by_arm: load(&AX_REWARD_POSITIVE_BY_ARM),
            chosen_by_direction: a.chosen_by_direction.clone(),
            leader_margin_micro: a.leader_margin_micro.clone(),
            control_runs_by_direction: a.control_runs_by_direction.clone(),
            control_reward_positive_by_direction: a.control_reward_positive_by_direction.clone(),
            control_runs_by_arm_direction: a.control_runs_by_arm_direction.clone(),
            control_reward_positive_by_arm_direction: a
                .control_reward_positive_by_arm_direction
                .clone(),
            chosen_by_combination: a.chosen_by_combination.clone(),
            control_runs_by_combination: a.control_runs_by_combination.clone(),
            control_reward_positive_by_combination: a
                .control_reward_positive_by_combination
                .clone(),
            explore: ArmSelectorExploreStats::read(),
            pooled: ArmSelectorPooledStats::read(),
            overtaken_ghost: a,
            absorber_cycle: ArmSelectorLearnerStats::read(Reward::AbsorberCycle),
            cycle_before_request: ArmSelectorLearnerStats::read(Reward::CycleBeforeRequest),
            ghost_signal: ArmSelectorRewardStats::read(Reward::GhostSignal),
            either_shape: ArmSelectorRewardStats::read(Reward::EitherShape),
            mutual_absorber_cycle: ArmSelectorRewardStats::read(Reward::MutualAbsorberCycle),
            exchange_before_request: ArmSelectorRewardStats::read(Reward::ExchangeBeforeRequest),
        }
    }
}

/// The replay corpus of the grid arms. `parents_admitted` counts fresh runs
/// whose prefix entered a corpus; `children` the slots that ran a child,
/// split into `children_prefix` and `children_plan_only`; `slots_unfilled`
/// the slots that found their corpus empty and ran fresh. The mechanism ran
/// as intended when `children_prefix` is positive and `prefix_faithful`, the
/// prefix children that reached the signal at their parent's step, is close
/// to it; `children_signal_fired` counts every child that reached the signal
/// at all. `tape_words_sum` is the recording cost every fresh grid run pays.
#[derive(Serialize, Debug)]
pub struct ReplayStats {
    pub parents_admitted: u64,
    pub children: u64,
    pub children_prefix: u64,
    pub children_plan_only: u64,
    pub slots_unfilled: u64,
    pub prefix_faithful: u64,
    pub tape_words_sum: u64,
    pub children_signal_fired: u64,
}

impl ReplayStats {
    fn read() -> Self {
        Self {
            parents_admitted: RP_PARENTS_ADMITTED.load(Ordering::Relaxed),
            children: RP_CHILDREN.load(Ordering::Relaxed),
            children_prefix: RP_CHILDREN_PREFIX.load(Ordering::Relaxed),
            children_plan_only: RP_CHILDREN_PLAN_ONLY.load(Ordering::Relaxed),
            slots_unfilled: RP_SLOTS_UNFILLED.load(Ordering::Relaxed),
            prefix_faithful: RP_PREFIX_FAITHFUL.load(Ordering::Relaxed),
            tape_words_sum: RP_TAPE_WORDS_SUM.load(Ordering::Relaxed),
            children_signal_fired: RP_CHILDREN_SIGNAL_FIRED.load(Ordering::Relaxed),
        }
    }
}

/// The timer-context block: the learner's probe traffic, the steered rolls
/// that applied a learned multiplier, the rolls an unsupported selector
/// excluded, and a gauge of the cells currently engaged. `cells_engaged` is
/// a gauge and must be read from a raw snapshot, not a difference of two.
#[derive(Serialize)]
pub struct TimerContextStats {
    pub probe_firings: u64,
    pub probe_acted: u64,
    pub biased_steps: u64,
    pub biased_steps_promoted: u64,
    pub biased_steps_suppressed: u64,
    pub steps_excluded_selector: u64,
    pub cells_engaged: u64,
}

impl TimerContextStats {
    fn read() -> Self {
        Self {
            probe_firings: TIMER_CONTEXT_PROBE_FIRINGS.load(Ordering::Relaxed),
            probe_acted: TIMER_CONTEXT_PROBE_ACTED.load(Ordering::Relaxed),
            biased_steps: TIMER_CONTEXT_BIASED_STEPS.load(Ordering::Relaxed),
            biased_steps_promoted: TIMER_CONTEXT_BIASED_STEPS_PROMOTED.load(Ordering::Relaxed),
            biased_steps_suppressed: TIMER_CONTEXT_BIASED_STEPS_SUPPRESSED
                .load(Ordering::Relaxed),
            steps_excluded_selector: TIMER_CONTEXT_STEPS_EXCLUDED_SELECTOR
                .load(Ordering::Relaxed),
            cells_engaged: TIMER_CONTEXT_CELLS_ENGAGED.load(Ordering::Relaxed),
        }
    }
}

/// A point-in-time copy of all counters, serializable to `utilization.json`.
#[derive(Serialize)]
pub struct UtilizationSnapshot {
    pub rng_streams: RngStreamStats,
    pub steer: SteerStats,
    pub steer_empty_slice: EmptySliceStats,
    pub steer_authority: SteerAuthorityStats,
    pub steer_reach: SteerReachStats,
    pub multiplier_authority: MultiplierAuthorityStats,
    pub recovery_weight_placebo: RecoveryPlaceboStats,
    pub purgatory: PurgatoryStats,
    pub aos: AosStats,
    pub dedup: DedupStats,
    pub feedback: FeedbackStats,
    pub curriculum: CurriculumStats,
    pub crash_recovery: CrashRecoveryStats,
    pub recovery_window: RecoveryWindowStats,
    pub ordered_h3: OrderedH3Stats,
    pub post_fault_ops: PostFaultOpsStats,
    pub plan_deps: PlanDepsStats,
    pub delivery_effects: DeliveryEffectStats,
    pub timer_effects: TimerEffectStats,
    pub timer_steer: TimerSteerStats,
    pub crash_anchor: CrashAnchorStats,
    pub termination: TerminationStats,
    pub prefix_extension: PrefixExtensionStats,
    pub quiet_stretch: QuietStretchStats,
    pub run_cap: RunCapStats,
    pub stall_cap: StallCapStats,
    pub stall_release: StallReleaseStats,
    pub crash_place: CrashPlaceStats,
    pub crash_phase: CrashPhaseStats,
    pub victim_swap: VictimSwapStats,
    pub ghost_signal: GhostSignalStats,
    pub frame: FrameStats,
    pub stats_local: StatsLocalStats,
    pub fresh_first: FreshFirstStats,
    pub pair_order: PairOrderStats,
    pub client_anchor: ClientAnchorStats,
    pub arm_selector_axis: ArmSelectorAxisStats,
    pub replay: ReplayStats,
    pub timer_context: TimerContextStats,
    pub timeline_keys: TimelineKeyStats,
    pub steer_terms: SteerTermStats,
}

/// The snapshot as JSON, for readers that difference or accumulate it.
pub fn snapshot_value() -> serde_json::Value {
    serde_json::to_value(snapshot()).unwrap_or(serde_json::Value::Null)
}

/// The one rendering of a snapshot that reaches a file. Every writer goes
/// through this, so a test can hold the bytes a reader will parse rather than
/// a second serialization that could differ from it.
pub fn render_snapshot(s: &UtilizationSnapshot) -> serde_json::Result<String> {
    serde_json::to_string_pretty(s)
}

/// `after - before` on every integer leaf, keeping the object structure.
/// Floats, arrays and strings are ratios, curves and labels, not counts, so
/// they are dropped; a reader recomputes ratios from the integer leaves.
pub fn delta(before: &serde_json::Value, after: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match (before, after) {
        (Value::Object(b), Value::Object(a)) => {
            let mut out = serde_json::Map::new();
            for (k, av) in a {
                let bv = b.get(k).unwrap_or(&Value::Null);
                match delta(bv, av) {
                    Value::Null => {}
                    v => {
                        out.insert(k.clone(), v);
                    }
                }
            }
            Value::Object(out)
        }
        (_, Value::Number(a)) if a.is_u64() || a.is_i64() => {
            let av = a.as_i64().unwrap_or_else(|| a.as_u64().unwrap_or(0) as i64);
            let bv = match before {
                Value::Number(b) => b.as_i64().unwrap_or_else(|| b.as_u64().unwrap_or(0) as i64),
                _ => 0,
            };
            Value::from(av - bv)
        }
        _ => Value::Null,
    }
}

/// Adds every integer leaf of `delta` into `acc`, creating what is missing.
pub fn add(acc: &mut serde_json::Value, delta: &serde_json::Value) {
    use serde_json::Value;
    if !acc.is_object() {
        *acc = Value::Object(serde_json::Map::new());
    }
    let Value::Object(d) = delta else { return };
    let acc_map = acc.as_object_mut().expect("made an object above");
    for (k, dv) in d {
        match dv {
            Value::Object(_) => {
                let slot = acc_map.entry(k.clone()).or_insert_with(|| Value::Object(serde_json::Map::new()));
                add(slot, dv);
            }
            Value::Number(n) => {
                let cur = acc_map.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
                acc_map.insert(k.clone(), Value::from(cur + n.as_i64().unwrap_or(0)));
            }
            _ => {}
        }
    }
}

pub fn snapshot() -> UtilizationSnapshot {
    fold_run_counters();
    UtilizationSnapshot {
        rng_streams: RngStreamStats {
            isolated_runs: RNG_ISOLATED_RUNS.load(Ordering::Relaxed),
            shared_runs: RNG_SHARED_RUNS.load(Ordering::Relaxed),
        },
        steer: SteerStats {
            evaluations: STEER_EVALUATIONS.load(Ordering::Relaxed),
            divergent_picks: STEER_DIVERGENT_PICKS.load(Ordering::Relaxed),
        },
        steer_empty_slice: EmptySliceStats {
            candidate_mask_skipped: ES_CANDIDATE_MASK.load(Ordering::Relaxed),
            ranking_pass_skipped: ES_RANKING_PASS.load(Ordering::Relaxed),
            queue_audit_skipped: ES_QUEUE_AUDIT.load(Ordering::Relaxed),
        },
        steer_authority: SteerAuthorityStats {
            steps_total: SA_STEPS_TOTAL.load(Ordering::Relaxed),
            steps: SA_STEPS.load(Ordering::Relaxed),
            audited: SA_AUDITED.load(Ordering::Relaxed),
            preference_expressed: SA_PREFERENCE_EXPRESSED.load(Ordering::Relaxed),
            preference_honored: SA_PREFERENCE_HONORED.load(Ordering::Relaxed),
            preference_consulted: SA_PREFERENCE_CONSULTED.load(Ordering::Relaxed),
            preference_source_absent: SA_PREFERENCE_SOURCE_ABSENT.load(Ordering::Relaxed),
            honored: SA_HONORED.load(Ordering::Relaxed),
            no_eligible_candidates: SA_NO_ELIGIBLE.load(Ordering::Relaxed),
            blocked_by_order: SA_BLOCKED_BY_ORDER.load(Ordering::Relaxed),
            blocked_by_timer_gate: SA_BLOCKED_BY_TIMER_GATE.load(Ordering::Relaxed),
            other_queue: SA_OTHER_QUEUE.load(Ordering::Relaxed),
            sampler_chose_other: SA_SAMPLER_CHOSE_OTHER.load(Ordering::Relaxed),
        },
        steer_reach: SteerReachStats {
            no_schedule_attempt: SR_NO_SCHEDULE_ATTEMPT.load(Ordering::Relaxed),
            audit_disabled: SR_AUDIT_DISABLED.load(Ordering::Relaxed),
            no_weighted_predicate: SR_NO_WEIGHTED_PREDICATE.load(Ordering::Relaxed),
            single_candidate: SR_SINGLE_CANDIDATE.load(Ordering::Relaxed),
            ranking_agreed_with_priority: SR_RANKING_AGREED.load(Ordering::Relaxed),
            preference_expressed: SR_PREFERENCE_EXPRESSED.load(Ordering::Relaxed),
        },
        multiplier_authority: MultiplierAuthorityStats::read(),
        recovery_weight_placebo: RecoveryPlaceboStats::read(),
        purgatory: PurgatoryStats {
            delayed_sends: PURGATORY_DELAYED_SENDS.load(Ordering::Relaxed),
            holds_down_receiver: PURGATORY_HOLDS_DOWN_RECEIVER.load(Ordering::Relaxed),
            holds_up_receiver: PURGATORY_HOLDS_UP_RECEIVER.load(Ordering::Relaxed),
            passthrough_down_receiver: PURGATORY_PASSTHROUGH_DOWN_RECEIVER
                .load(Ordering::Relaxed),
        },
        aos: AosStats {
            tape_wins: AOS_TAPE_WINS.load(Ordering::Relaxed),
            config_wins: AOS_CONFIG_WINS.load(Ordering::Relaxed),
        },
        dedup: DedupStats {
            checks: DEDUP_CHECKS.load(Ordering::Relaxed),
            hits: DEDUP_HITS.load(Ordering::Relaxed),
            skipped_large: DEDUP_SKIPPED_LARGE.load(Ordering::Relaxed),
        },
        feedback: FeedbackStats {
            timeline_score_sum: f64::from_bits(FEEDBACK_TIMELINE_SCORE_SUM.load(Ordering::Relaxed)),
            cfg_score_sum: f64::from_bits(FEEDBACK_CFG_SCORE_SUM.load(Ordering::Relaxed)),
            scored_runs: FEEDBACK_SCORED_RUNS.load(Ordering::Relaxed),
        },
        curriculum: CurriculumStats {
            lowered_runs: CURRICULUM_LOWERED_RUNS.load(Ordering::Relaxed),
            crashes_sum: CURRICULUM_CRASHES_SUM.load(Ordering::Relaxed),
            servers_sum: CURRICULUM_SERVERS_SUM.load(Ordering::Relaxed),
        },
        crash_recovery: CrashRecoveryStats {
            runs: CR_RUNS.load(Ordering::Relaxed),
            crashes: CR_CRASHES.load(Ordering::Relaxed),
            recovers: CR_RECOVERS.load(Ordering::Relaxed),
            messages_held_at_crash: CR_HELD_AT_CRASH.load(Ordering::Relaxed),
            messages_dropped_at_crash: CR_DROPPED_AT_CRASH.load(Ordering::Relaxed),
            crossing_deliveries: CR_CROSSING_DELIVERIES.load(Ordering::Relaxed),
            runs_with_crossing: CR_RUNS_WITH_CROSSING.load(Ordering::Relaxed),
        },
        recovery_window: RecoveryWindowStats::read(),
        ordered_h3: OrderedH3Stats::read(),
        post_fault_ops: PostFaultOpsStats {
            pairs_seen: PFO_PAIRS_SEEN.load(Ordering::Relaxed),
            edges_added: PFO_EDGES_ADDED.load(Ordering::Relaxed),
            ops_invoked_after_last_recover: PFO_OPS_AFTER_LAST_RECOVER.load(Ordering::Relaxed),
        },
        plan_deps: {
            let cells = PLAN_DEPS_CELLS
                .lock()
                .map(|c| *c)
                .unwrap_or_else(|p| *p.into_inner());
            PlanDepsStats {
                recover_edges_dropped: PD_RECOVER_EDGES_DROPPED.load(Ordering::Relaxed),
                stock: cells[RecoverDeps::Stock.index()],
                exempt: cells[RecoverDeps::Exempt.index()],
            }
        },
        delivery_effects: DeliveryEffectStats {
            all: DeliveryEffect::read(DELIVERY_ALL),
            biased: DeliveryEffect::read(DELIVERY_BIASED),
            delayed: DeliveryEffect::read(DELIVERY_DELAYED),
            sender_restarted: DeliveryEffect::read(DELIVERY_SENDER_RESTARTED),
            receiver_restarted: DeliveryEffect::read(DELIVERY_RECEIVER_RESTARTED),
            acceptance_distance: AcceptanceDistanceStats::read(),
            crashes_total: CA_CRASHES_TAKEN.load(Ordering::Relaxed),
            crashes_with_own_sends_inflight: CA_APPLIED.load(Ordering::Relaxed),
            recoveries_total: CR_RECOVERS.load(Ordering::Relaxed),
            recoveries_with_own_prior_sends_inflight: CR_RECOVERS_WITH_INFLIGHT
                .load(Ordering::Relaxed),
            stale_sender_deliveries_after_recovery: DELIVERIES[DELIVERY_SENDER_RESTARTED]
                .load(Ordering::Relaxed),
            crash_census: CrashCensusStats::read(),
        },
        timer_effects: TimerEffectStats::read(),
        timer_steer: TimerSteerStats::read(),
        crash_anchor: CrashAnchorStats {
            steps_with_crash_eligible: CA_STEPS_WITH_CRASH_ELIGIBLE.load(Ordering::Relaxed),
            offered: CA_OFFERED.load(Ordering::Relaxed),
            crashes_taken: CA_CRASHES_TAKEN.load(Ordering::Relaxed),
            applied: CA_APPLIED.load(Ordering::Relaxed),
            timing_bias_examined: CA_TIMING_BIAS_EXAMINED.load(Ordering::Relaxed),
            timing_bias_withheld: CA_TIMING_BIAS_WITHHELD.load(Ordering::Relaxed),
        },
        termination: TERMINATION
            .lock()
            .map(|t| *t)
            .unwrap_or_else(|p| *p.into_inner()),
        prefix_extension: PREFIX_EXTENSION
            .lock()
            .map(|p| *p)
            .unwrap_or_else(|p| *p.into_inner()),
        quiet_stretch: QuietStretchStats::read(),
        run_cap: RunCapStats::read(),
        stall_cap: StallCapStats::read(),
        stall_release: StallReleaseStats::read(),
        crash_place: CrashPlaceStats::read(),
        crash_phase: CrashPhaseStats::read(),
        victim_swap: VictimSwapStats::read(),
        ghost_signal: GhostSignalStats::read(),
        frame: FrameStats::read(),
        stats_local: StatsLocalStats::read(),
        fresh_first: FreshFirstStats::read(),
        pair_order: PairOrderStats::read(),
        client_anchor: ClientAnchorStats::read(),
        arm_selector_axis: ArmSelectorAxisStats::read(),
        replay: ReplayStats::read(),
        timer_context: TimerContextStats::read(),
        timeline_keys: TimelineKeyStats::read(),
        steer_terms: SteerTermStats::read(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    #[test]
    fn pair_order_counters_land_on_their_half() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        let before = snapshot().pair_order;
        record_pair_order_contest(true, false);
        record_pair_order_contest(true, true);
        record_pair_order_contest(false, false);
        record_pair_order_contest_class(true);
        record_pair_order_contest_class(false);
        record_pair_order_correction(true);
        record_pair_order_fresh_suppressed();
        record_pair_order_fresh_suppressed();
        record_pair_order_entry(true, true, false);
        record_pair_order_entry(false, true, true);
        record_pair_order_entry(false, false, false);
        record_pair_order_census_run(true);
        record_pair_order_census_run(false);
        record_pair_order_census_run(false);
        let after = snapshot().pair_order;
        set_enabled(false);
        record_pair_order_correction(true);
        record_pair_order_fresh_suppressed();
        assert_eq!(snapshot().pair_order.corrected, after.corrected, "a disabled session counted");

        let (t, c) = (&after.census.treated, &after.census.control);
        let (bt, bc) = (&before.census.treated, &before.census.control);
        assert_eq!(t.contests - bt.contests, 2);
        assert_eq!(t.inorder_draws - bt.inorder_draws, 1);
        assert_eq!(c.contests - bc.contests, 1);
        assert_eq!(c.inorder_draws - bc.inorder_draws, 0);
        assert_eq!(after.corrected - before.corrected, 1);
        assert_eq!(
            after.contests_by_class.ghost - before.contests_by_class.ghost,
            1
        );
        assert_eq!(
            after.contests_by_class.fresh - before.contests_by_class.fresh,
            1
        );
        assert_eq!(after.corrections_ghost - before.corrections_ghost, 1);
        assert_eq!(after.corrections_fresh - before.corrections_fresh, 0);
        assert_eq!(
            after.corrections_fresh_suppressed - before.corrections_fresh_suppressed,
            2
        );
        assert_eq!(t.pair_entries - bt.pair_entries, 1);
        assert_eq!(t.pair_entries_ghost - bt.pair_entries_ghost, 1);
        assert_eq!(t.inversions - bt.inversions, 0);
        assert_eq!(t.inversions_ghost - bt.inversions_ghost, 0);
        assert_eq!(c.pair_entries - bc.pair_entries, 2);
        assert_eq!(c.pair_entries_ghost - bc.pair_entries_ghost, 1);
        assert_eq!(c.inversions - bc.inversions, 1);
        assert_eq!(c.inversions_ghost - bc.inversions_ghost, 1);
        assert_eq!(t.sampled_runs - bt.sampled_runs, 1);
        assert_eq!(c.sampled_runs - bc.sampled_runs, 2);
    }

    #[test]
    fn fresh_first_counters_land_on_their_half_and_in_their_bucket() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        let before = snapshot().fresh_first;
        record_fresh_first_contest(true, true, false);
        record_fresh_first_contest(true, true, true);
        record_fresh_first_contest(true, false, true);
        record_fresh_first_contest(false, true, false);
        record_fresh_first_swap(false);
        record_fresh_first_swap(true);
        record_fresh_first_taken(fresh_first::DisplacedCount::Once);
        record_fresh_first_taken(fresh_first::DisplacedCount::Thrice);
        record_fresh_first_taken(fresh_first::DisplacedCount::FourOrMore);
        record_fresh_first_ghost_entry(true, true, true);
        record_fresh_first_ghost_entry(false, false, true);
        record_fresh_first_ghost_entry(false, true, false);
        record_fresh_first_skipped_never_restarted_dest();
        let after = snapshot().fresh_first;
        set_enabled(false);
        record_fresh_first_swap(false);
        assert_eq!(snapshot().fresh_first.swaps, after.swaps, "a disabled session counted");

        let (t, c) = (&after.census.treated, &after.census.control);
        let (bt, bc) = (&before.census.treated, &before.census.control);
        assert_eq!(t.contested_dispatches - bt.contested_dispatches, 3);
        assert_eq!(t.stale_drawn - bt.stale_drawn, 2);
        assert_eq!(t.contested_down - bt.contested_down, 1, "a fresh draw at a down destination is not a held-back swap");
        assert_eq!(c.contested_dispatches - bc.contested_dispatches, 1);
        assert_eq!(c.stale_drawn - bc.stale_drawn, 1);
        assert_eq!(c.contested_down - bc.contested_down, 0);
        assert_eq!(after.swaps - before.swaps, 2);
        assert_eq!(after.repeat_swaps - before.repeat_swaps, 1);
        assert_eq!(after.swap_count_hist_1 - before.swap_count_hist_1, 1);
        assert_eq!(after.swap_count_hist_2 - before.swap_count_hist_2, 0);
        assert_eq!(after.swap_count_hist_3 - before.swap_count_hist_3, 1);
        assert_eq!(after.swap_count_hist_4plus - before.swap_count_hist_4plus, 1);
        assert_eq!(t.ghost_entries_from_restarted_origin - bt.ghost_entries_from_restarted_origin, 1);
        assert_eq!(t.overtaken - bt.overtaken, 1);
        assert_eq!(c.ghost_entries_from_restarted_origin - bc.ghost_entries_from_restarted_origin, 2);
        assert_eq!(c.overtaken - bc.overtaken, 1);
        assert_eq!(t.ghost_entries_to_restarted_dest - bt.ghost_entries_to_restarted_dest, 1);
        assert_eq!(t.overtaken_at_restarted_dest - bt.overtaken_at_restarted_dest, 1);
        assert_eq!(c.ghost_entries_to_restarted_dest - bc.ghost_entries_to_restarted_dest, 1);
        assert_eq!(
            c.overtaken_at_restarted_dest - bc.overtaken_at_restarted_dest,
            0,
            "an overtaken entry at a destination that never restarted is not in the subset"
        );
        assert_eq!(after.skipped_never_restarted_dest - before.skipped_never_restarted_dest, 1);
    }

    #[test]
    fn term_counters_reset_and_snapshot() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        let stale = 1u8 << Term::StaleLate.index();
        let crash = 1u8 << Term::CrashAfterTimerSends.index();
        record_term_decision(3, stale | crash, &[0, 0, 2, 0], stale, true);
        record_term_decision(1, crash, &[1, 0, 0, 0], 0, false);
        record_term_acted(stale, true);
        record_term_acted(stale, false);
        record_term_authority(true);
        record_term_authority(false);
        record_audit_candidates(7);
        let s = snapshot().steer_terms;
        assert_eq!(s.decisions, 2);
        assert_eq!(s.stale_late.evaluated, 2);
        assert_eq!(s.stale_late.present, 1);
        assert_eq!(s.stale_late.contested, 1);
        assert_eq!(s.stale_late.won, 1);
        assert_eq!(s.stale_late.flipped, 1);
        assert_eq!(s.stale_late.measured, 2);
        assert_eq!(s.stale_late.acted, 1);
        assert_eq!(s.crash_after_timer_sends.present, 2);
        assert_eq!(s.crash_after_timer_sends.contested, 1);
        assert_eq!(s.crash_after_timer_sends.won, 0);
        assert_eq!(s.authority_draws, 2);
        assert_eq!(s.authority_routed, 1);
        assert_eq!(s.eligible_hist[hist_bucket(3)], 1);
        assert_eq!(s.candidates_hist[hist_bucket(7)], 1);
        set_enabled(true);
        let z = snapshot().steer_terms;
        assert_eq!(z, SteerTermStats { eligible_hist: vec![0; HIST_BUCKETS], candidates_hist: vec![0; HIST_BUCKETS], ..SteerTermStats::default() });
        set_enabled(false);
    }

    #[test]
    fn hist_buckets_are_log2_with_small_counts_exact() {
        assert_eq!(hist_bucket(0), 0);
        assert_eq!(hist_bucket(1), 1);
        assert_eq!(hist_bucket(2), 2);
        assert_eq!(hist_bucket(3), 3);
        assert_eq!(hist_bucket(4), 3);
        assert_eq!(hist_bucket(5), 4);
        assert_eq!(hist_bucket(8), 4);
        assert_eq!(hist_bucket(9), 5);
        assert_eq!(hist_bucket(1 << 20), HIST_BUCKETS - 1);
    }

    #[test]
    fn delivery_effects_split_by_bias() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);

        record_delivery(DeliveryBias::NONE, true, 0);
        record_delivery(DeliveryBias::DELAYED, false, 0);
        let mut both = DeliveryBias::DELAYED;
        both.insert(DeliveryBias::SENDER_RESTARTED);
        record_delivery(both, true, 0);

        let s = snapshot().delivery_effects;
        set_enabled(false);

        assert_eq!(s.all.deliveries, 3);
        assert_eq!(s.all.acted, 2);
        assert_eq!(s.biased.deliveries, 2);
        assert_eq!(s.biased.acted, 1);
        assert_eq!(s.delayed.deliveries, 2);
        assert_eq!(s.delayed.acted, 1);
        assert_eq!(s.sender_restarted.deliveries, 1);
        assert_eq!(s.receiver_restarted.deliveries, 0);
        assert_eq!(s.receiver_restarted.acted_fraction, 0.0);
        assert!((s.all.acted_fraction - 2.0 / 3.0).abs() < 1e-9);
        assert!(
            s.acceptance_distance.all.iter().all(|b| b.deliveries == 0),
            "census stays empty while its own switch is off"
        );
    }

    #[test]
    fn crash_census_needs_its_own_switch() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        record_crash_census(2, true);
        let off = snapshot().delivery_effects.crash_census;

        set_crash_census_enabled(true);
        record_crash_census(0, true);
        record_crash_census(1, true);
        record_crash_census(7, false);
        let s = snapshot().delivery_effects.crash_census;

        set_crash_census_enabled(false);
        set_enabled(false);

        assert_eq!(off.decisions, 0);
        assert_eq!(s.decisions, 3);
        assert_eq!(s.victim_had_inflight_sends, 2);
        assert_eq!(s.any_candidate_had_inflight_sends, 2);
        assert_eq!(s.inflight_bucket_0, 1);
        assert_eq!(s.inflight_bucket_1, 1);
        assert_eq!(s.inflight_bucket_2, 0);
        assert_eq!(s.inflight_bucket_3plus, 1);
    }

    #[test]
    fn fault_tallies_ride_along_with_the_delivery_effects() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);

        begin_run();
        record_crash_anchor_apply(true);
        record_crash(0, 0, 0);
        record_recover(0, 4, true);
        record_crash_anchor_apply(false);
        record_crash(1, 0, 0);
        record_recover(1, 9, false);
        record_delivery(DeliveryBias::SENDER_RESTARTED, true, 1);
        record_delivery(DeliveryBias::NONE, true, 1);

        let s = snapshot().delivery_effects;
        set_enabled(false);

        assert_eq!(s.crashes_total, 2);
        assert_eq!(s.crashes_with_own_sends_inflight, 1);
        assert_eq!(s.recoveries_total, 2);
        assert_eq!(s.recoveries_with_own_prior_sends_inflight, 1);
        assert_eq!(s.stale_sender_deliveries_after_recovery, 1);
        assert_eq!(
            s.stale_sender_deliveries_after_recovery,
            s.sender_restarted.deliveries
        );
    }

    #[test]
    fn acceptance_distance_splits_by_receiver_progress() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);
        set_acceptance_distance_enabled(true);

        record_delivery(DeliveryBias::SENDER_RESTARTED, true, 0);
        record_delivery(DeliveryBias::SENDER_RESTARTED, false, 20);
        record_delivery(DeliveryBias::RECEIVER_RESTARTED, false, 3);
        record_delivery(DeliveryBias::NONE, true, 6);

        let s = snapshot().delivery_effects.acceptance_distance;
        set_acceptance_distance_enabled(false);
        set_enabled(false);

        assert_eq!(s.sender_restarted[0].distance, "0");
        assert_eq!(s.sender_restarted[0].deliveries, 1);
        assert_eq!(s.sender_restarted[0].acted, 1);
        assert_eq!(s.sender_restarted[6].distance, "17+");
        assert_eq!(s.sender_restarted[6].deliveries, 1);
        assert_eq!(s.sender_restarted[6].acted, 0);
        assert_eq!(s.receiver_restarted[3].distance, "3-4");
        assert_eq!(s.receiver_restarted[3].deliveries, 1);
        assert_eq!(s.all[4].distance, "5-8");
        assert_eq!(s.all[4].deliveries, 1);
        assert_eq!(s.all.iter().map(|b| b.deliveries).sum::<u64>(), 4);
    }

    #[test]
    fn timer_effects_split_by_context_and_reset_the_streak_when_acted() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);

        record_timer(TimerKey::new(7, false, 0, 0), false);
        record_timer(TimerKey::new(7, false, 0, 1), false);
        record_timer(TimerKey::new(7, false, 0, 2), true);
        record_timer(TimerKey::new(7, true, 1, 0), false);
        record_timer(TimerKey::new(9, true, 3, 12), true);

        let s = snapshot().timer_effects;
        set_enabled(false);

        assert_eq!(s.all.fired, 5);
        assert_eq!(s.all.acted, 2);
        assert_eq!(s.with_inflight.fired, 2);
        assert_eq!(s.with_inflight.acted, 1);
        assert_eq!(s.idle.fired, 3);
        assert_eq!(s.idle.acted, 1);
        assert_eq!(s.by_key.len(), 4, "streaks 1 and 2 share a bucket");
        let short = s.by_key.iter().find(|e| e.vertex == 7 && e.inert_streak == 1).expect("streak bucket 1 keyed");
        assert_eq!((short.fired, short.acted), (2, 1));
        let deep = s.by_key.iter().find(|e| e.vertex == 9).expect("vertex 9 keyed");
        assert_eq!((deep.incarnation, deep.inert_streak), (2, 3));
        assert!(s.by_key.windows(2).all(|w| (w[0].vertex, w[0].inflight) <= (w[1].vertex, w[1].inflight)));

        let h = &s.inert_streak;
        assert_eq!((h.none.fired, h.none.acted), (2, 0));
        assert_eq!((h.short.fired, h.short.acted), (2, 1));
        assert_eq!((h.medium.fired, h.medium.acted), (0, 0));
        assert_eq!((h.long.fired, h.long.acted), (1, 1));
        assert_eq!(
            h.none.fired + h.short.fired + h.medium.fired + h.long.fired,
            s.all.fired
        );

        set_enabled(true);
        assert_eq!(snapshot().timer_effects.all.fired, 0, "enabling resets the table");
        assert_eq!(snapshot().timer_effects.inert_streak.short.fired, 0);
        set_enabled(false);
    }

    #[test]
    fn timer_admission_splits_contested_steps() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        record_timer_admission(true);
        record_timer_admission(false);
        record_timer_admission(false);

        let s = snapshot().timer_steer;
        set_enabled(false);

        assert_eq!(s.evaluated, 3);
        assert_eq!(s.raised, 1);
        assert_eq!(s.lowered, 2);
        assert_eq!(s.raised + s.lowered, s.evaluated);

        record_timer_admission(true);
        assert_eq!(snapshot().timer_steer.evaluated, 3, "records nothing when off");
    }

    #[test]
    fn a_timer_streak_counts_inert_firings_and_resets_on_an_effect() {
        use crate::analysis::resolver::NameId;
        use crate::simulator::core::state::State;
        use crate::simulator::hash_utils::NoHashing;
        let mut st = State::<NoHashing>::new(&[(NameId(0), 1)], 4);
        st.note_timer_effect(0, 5, false, false);
        st.note_timer_effect(0, 5, false, false);
        assert_eq!(st.timer_inert_streak(0, 5), 2);
        st.note_timer_effect(0, 5, true, true);
        assert_eq!(st.timer_inert_streak(0, 5), 0);
        assert_eq!(st.timer_stats.max_inert_streak, 2);
        assert_eq!((st.timer_stats.fired, st.timer_stats.acted), (3, 1));
        assert_eq!((st.timer_stats.idle_fired, st.timer_stats.inflight_fired), (2, 1));
    }

    #[test]
    fn steer_authority_outcomes_partition_the_audited_steps() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(true);

        for _ in 0..12 {
            record_steer_step_total();
        }
        for _ in 0..9 {
            record_steer_step();
        }
        record_steer_authority(true, SteerOutcome::Honored);
        record_steer_authority(false, SteerOutcome::Honored);
        record_steer_authority(true, SteerOutcome::SamplerChoseOther);
        record_steer_authority(true, SteerOutcome::OtherQueue);
        record_steer_authority(false, SteerOutcome::BlockedByTimerGate);
        record_steer_authority(false, SteerOutcome::BlockedByOrder);
        record_steer_authority(false, SteerOutcome::NoEligibleCandidates);

        let s = snapshot().steer_authority;
        set_steer_audit_enabled(false);
        set_enabled(false);

        assert_eq!((s.steps_total, s.steps), (12, 9));
        assert_eq!(s.audited, 7);
        assert_eq!(s.preference_expressed, 3);
        assert_eq!(s.preference_honored, 1);
        assert_eq!(
            s.honored
                + s.no_eligible_candidates
                + s.blocked_by_order
                + s.blocked_by_timer_gate
                + s.other_queue
                + s.sampler_chose_other,
            s.audited
        );
    }

    #[test]
    fn steer_authority_records_nothing_when_the_audit_is_off() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(false);

        record_steer_step_total();
        record_steer_step();
        record_steer_authority(true, SteerOutcome::Honored);
        record_preference_consultation(true);

        let s = snapshot().steer_authority;
        set_enabled(false);

        assert_eq!(
            (s.steps_total, s.steps, s.audited, s.preference_consulted),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn preference_consultations_split_into_sourced_and_sourceless_reads() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_steer_audit_enabled(true);

        record_preference_consultation(false);
        record_preference_consultation(false);
        record_preference_consultation(true);

        let s = snapshot().steer_authority;
        set_steer_audit_enabled(false);
        set_enabled(false);

        assert_eq!(s.preference_consulted, 3);
        assert_eq!(s.preference_source_absent, 2);
    }

    #[test]
    fn quiet_stretch_measures_the_longest_run_of_deliveries_without_an_effect() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);
        set_quiet_stretch_enabled(true);

        begin_run();
        record_delivery(DeliveryBias::NONE, false, 0);
        record_delivery(DeliveryBias::NONE, true, 0);
        record_delivery(DeliveryBias::NONE, false, 0);
        record_delivery(DeliveryBias::NONE, false, 0);
        record_delivery(DeliveryBias::NONE, false, 0);
        record_quiet_stretch(7, RunEnd::IterationsExhausted);

        begin_run();
        record_delivery(DeliveryBias::NONE, true, 0);
        record_quiet_stretch(8, RunEnd::PlanComplete);

        let s = snapshot().quiet_stretch;
        set_quiet_stretch_enabled(false);
        // The switch alone gates the block: recording stays off with the
        // delivery-effect probe on.
        begin_run();
        record_delivery(DeliveryBias::NONE, false, 0);
        record_quiet_stretch(9, RunEnd::Deadlock);
        let after_off = snapshot().quiet_stretch;
        set_enabled(false);

        assert_eq!(s.runs, 2);
        assert_eq!(s.per_run.len(), 2);
        assert_eq!(s.per_run[0].run_id, 7);
        assert_eq!(s.per_run[0].longest_quiet_stretch, 3);
        assert_eq!(s.per_run[0].deliveries, 5);
        assert_eq!(s.per_run[1].longest_quiet_stretch, 0);
        assert_eq!(s.iterations_exhausted[hist_bucket(3)], 1);
        assert_eq!(s.plan_complete[hist_bucket(0)], 1);
        assert_eq!(after_off.runs, 2);
        assert_eq!(after_off.deadlock.iter().sum::<u64>(), 0);
    }

    #[test]
    fn runs_are_split_by_what_stopped_them_from_scheduling_further_work() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_prefix_extension_enabled(true);

        let run = |end, tail_without_release, pending_at_exit, recovered_nodes| RunExtension {
            end,
            steps: 1_000,
            steps_released: 900,
            steps_blocked: 60,
            steps_idle: 40,
            tail_without_release,
            pending_at_exit,
            recovered_nodes,
        };
        record_run_extension(&run(RunEnd::IterationsExhausted, 0, 7, 0));
        record_run_extension(&run(RunEnd::IterationsExhausted, STALLED_TAIL_STEPS, 7, 2));
        record_run_extension(&run(RunEnd::IterationsExhausted, 500, 0, 1));
        record_run_extension(&run(RunEnd::PlanComplete, 0, 3, 2));
        record_run_extension(&run(RunEnd::PlanComplete, 0, 0, 0));
        record_run_extension(&run(RunEnd::StallCapReached, 0, 7, 0));

        let s = snapshot().prefix_extension;
        set_prefix_extension_enabled(false);
        // The switch alone gates the block: recording stays off with stats on.
        record_run_extension(&run(RunEnd::Deadlock, 0, 0, 0));
        let after_off = snapshot().prefix_extension;
        set_enabled(false);

        assert_eq!(s.all.runs, 6);
        assert_eq!(s.all.budget_releasing, 2, "a stall-cap exit is a budget stop");
        assert_eq!(s.all.budget_blocked, 1);
        assert_eq!(s.all.budget_idle, 1);
        assert_eq!(s.all.plan_complete_pending, 1);
        assert_eq!(s.all.plan_complete_quiescent, 1);
        assert_eq!(s.all.steps_sum, 6_000);
        assert_eq!(s.all.steps_blocked_sum, 360);
        assert_eq!(s.by_recovered_nodes[2].runs, 2);
        assert_eq!(s.by_recovered_nodes[2].budget_blocked, 1);
        assert_eq!(s.by_recovered_nodes[2].plan_complete_pending, 1);
        assert_eq!(after_off.all.runs, 6);
        assert_eq!(after_off.all.deadlock, 0);
    }

    #[test]
    fn a_learned_cap_exit_counts_only_its_own_tally_field() {
        let mut t = TerminationTally::new();
        t.add(
            RunEnd::LearnedCapReached,
            &RunTermination {
                end: RunEnd::LearnedCapReached,
                steps_used: 10,
                step_budget: 100,
                pending_work_at_exit: 1,
                planned_events_outstanding: 2,
                recovered_nodes: 0,
            },
        );
        assert_eq!(t.runs, 1);
        assert_eq!(t.learned_cap_reached, 1);
        assert_eq!(t.stall_cap_reached, 0);
        assert_eq!(t.plan_complete, 0);
        assert_eq!(t.iterations_exhausted, 0);
        assert_eq!(t.deadlock, 0);
    }

    #[test]
    fn a_stall_cap_exit_counts_only_its_own_tally_field() {
        let mut t = TerminationTally::new();
        t.add(
            RunEnd::StallCapReached,
            &RunTermination {
                end: RunEnd::StallCapReached,
                steps_used: 10,
                step_budget: 100,
                pending_work_at_exit: 1,
                planned_events_outstanding: 2,
                recovered_nodes: 0,
            },
        );
        assert_eq!(t.runs, 1);
        assert_eq!(t.stall_cap_reached, 1);
        assert_eq!(t.learned_cap_reached, 0);
        assert_eq!(t.plan_complete, 0);
        assert_eq!(t.iterations_exhausted, 0);
        assert_eq!(t.deadlock, 0);
        assert_eq!(t.steps_used_sum, 10);
    }

    #[test]
    fn a_stall_cap_exit_classifies_as_a_budget_stop() {
        let run = |tail_without_release, pending_at_exit| RunExtension {
            end: RunEnd::StallCapReached,
            steps: 100,
            steps_released: 90,
            steps_blocked: 6,
            steps_idle: 4,
            tail_without_release,
            pending_at_exit,
            recovered_nodes: 0,
        };
        assert_eq!(run(0, 0).stop(), PrefixStop::BudgetIdle);
        assert_eq!(run(STALLED_TAIL_STEPS, 7).stop(), PrefixStop::BudgetBlocked);
        assert_eq!(run(0, 7).stop(), PrefixStop::BudgetReleasing);
    }

    #[test]
    fn stall_cap_counters_reach_the_snapshot_and_reset_with_enable() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        let run = |cell, steps_saved, longest_gap, standing_cap| StallCapRun {
            cell,
            marks: StallCapMarks {
                rows: 2,
                acted_deliveries: 3,
                acted_timers: 5,
                releases: 7,
            },
            suspended_steps: 11,
            steps_saved,
            longest_gap,
            standing_cap,
            row_dropped: false,
        };
        record_stall_cap_run(&run(StallCapCell::Treated, Some(100), 30, Some(20)));
        record_stall_cap_run(&run(StallCapCell::Treated, None, 5, Some(20)));
        record_stall_cap_run(&run(StallCapCell::Untreated, None, 30, Some(20)));
        record_stall_cap_run(&run(StallCapCell::Untreated, None, 9, Some(20)));
        record_stall_cap_run(&run(StallCapCell::Untreated, None, 40, None));
        record_stall_cap_run(&run(StallCapCell::Probe, None, 40, None));
        record_stall_cap_probe(false);
        record_stall_cap_probe(true);
        set_stall_cap_learned(2, 611);
        let s = snapshot().stall_cap;
        assert_eq!(s.stops, 1);
        assert_eq!(s.treated_runs, 2);
        assert_eq!(s.untreated_runs, 3);
        assert_eq!(s.steps_saved_sum, 100);
        assert_eq!(s.suspended_steps_sum, 66);
        assert_eq!(s.marks.rows, 12);
        assert_eq!(s.marks.acted_deliveries, 18);
        assert_eq!(s.marks.acted_timers, 30);
        assert_eq!(s.marks.releases, 42);
        assert_eq!(s.probes_keyed, 2);
        assert_eq!(s.probe_over_cap_completions, 1);
        assert_eq!((s.scopes_learned, s.cap_max_scope), (2, 611));
        assert_eq!(s.untreated_runs_capped, 2, "a run under no standing cap is not capped");
        assert_eq!(s.untreated_over_cap_runs, 1);
        assert_eq!(s.untreated_gap_hist[hist_bucket(30)], 1);
        assert_eq!(s.untreated_gap_hist[hist_bucket(9)], 1);
        assert_eq!(s.untreated_gap_hist[hist_bucket(40)], 1, "the probe's gap is not in the histogram");
        assert_eq!(s.untreated_gap_hist.iter().sum::<u64>(), 3);
        set_enabled(false);
        record_stall_cap_run(&run(StallCapCell::Treated, Some(100), 30, Some(20)));
        assert_eq!(snapshot().stall_cap.stops, 1, "recording is gated on stats");
        set_enabled(true);
        let z = snapshot().stall_cap;
        set_enabled(false);
        assert_eq!(z.stops, 0);
        assert_eq!(z.treated_runs, 0);
        assert_eq!(z.untreated_runs, 0);
        assert_eq!(z.marks.releases, 0);
        assert_eq!(z.probes_keyed, 0);
        assert_eq!((z.scopes_learned, z.cap_max_scope), (0, 0));
        assert_eq!(z.untreated_gap_hist.iter().sum::<u64>(), 0);
    }

    #[test]
    fn plan_deps_folds_each_run_into_its_cell_by_density_and_resets() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        let termination = |end, steps_used, recovered_nodes| RunTermination {
            end,
            steps_used,
            step_budget: 100,
            pending_work_at_exit: 0,
            planned_events_outstanding: 0,
            recovered_nodes,
        };
        record_plan_deps_edges(3);
        // An exempt-cell run at positive density: two crashes, one recover.
        record_plan_deps_run(RecoverDeps::Exempt, true);
        begin_run();
        record_crash(0, 0, 0);
        record_crash(1, 0, 0);
        record_recover(0, 5, false);
        record_run_termination(&termination(RunEnd::LearnedCapReached, 40, 1));
        // A stock-cell run at zero density that completed without a fault.
        record_plan_deps_run(RecoverDeps::Stock, false);
        begin_run();
        record_run_termination(&termination(RunEnd::PlanComplete, 7, 0));
        // A run that registered no cell reaches the session counters only.
        begin_run();
        record_crash(2, 0, 0);
        record_run_termination(&termination(RunEnd::IterationsExhausted, 9, 0));

        let s = snapshot();
        assert_eq!(s.plan_deps.recover_edges_dropped, 3);
        let e = s.plan_deps.exempt.density_positive;
        assert_eq!(e.runs, 1);
        assert_eq!(e.crashes, 2);
        assert_eq!(e.unrecovered_crashes, 1);
        assert_eq!(e.zero_recovery_runs, 0);
        assert_eq!(e.plan_complete, 0);
        assert_eq!(e.steps_used_sum, 40);
        assert_eq!(s.plan_deps.exempt.density_zero.runs, 0);
        let f = s.plan_deps.stock.density_zero;
        assert_eq!(f.runs, 1);
        assert_eq!(f.crashes, 0);
        assert_eq!(f.unrecovered_crashes, 0);
        assert_eq!(f.zero_recovery_runs, 1);
        assert_eq!(f.plan_complete, 1);
        assert_eq!(f.steps_used_sum, 7);
        assert_eq!(s.plan_deps.stock.density_positive.runs, 0);
        // The cells and the session counters name the same crashes.
        let cells = [s.plan_deps.stock, s.plan_deps.exempt];
        let by_cell: u64 = cells
            .iter()
            .map(|c| c.density_zero.unrecovered_crashes + c.density_positive.unrecovered_crashes)
            .sum();
        assert_eq!(s.crash_recovery.crashes - s.crash_recovery.recovers, by_cell + 1);
        let zero_by_cell: u64 = cells
            .iter()
            .map(|c| c.density_zero.zero_recovery_runs + c.density_positive.zero_recovery_runs)
            .sum();
        assert_eq!(s.termination.by_recovered_nodes[0].runs, zero_by_cell + 1);

        record_plan_deps_run(RecoverDeps::Exempt, false);
        set_enabled(true);
        let s = snapshot();
        assert_eq!(s.plan_deps.recover_edges_dropped, 0);
        for c in [s.plan_deps.stock, s.plan_deps.exempt] {
            assert_eq!(c.density_zero.runs, 0);
            assert_eq!(c.density_positive.runs, 0);
            assert_eq!(c.density_positive.crashes, 0);
        }
        // The registration does not outlive the reset either.
        begin_run();
        record_run_termination(&termination(RunEnd::PlanComplete, 1, 0));
        let s = snapshot();
        assert_eq!(s.plan_deps.exempt.density_zero.runs, 0);
        assert_eq!(s.plan_deps.exempt.density_positive.runs, 0);
        set_enabled(false);
    }

    #[test]
    fn a_learned_cap_exit_classifies_as_a_budget_stop() {
        let run = |tail_without_release, pending_at_exit| RunExtension {
            end: RunEnd::LearnedCapReached,
            steps: 100,
            steps_released: 90,
            steps_blocked: 6,
            steps_idle: 4,
            tail_without_release,
            pending_at_exit,
            recovered_nodes: 0,
        };
        assert_eq!(run(0, 0).stop(), PrefixStop::BudgetIdle);
        assert_eq!(run(STALLED_TAIL_STEPS, 7).stop(), PrefixStop::BudgetBlocked);
        assert_eq!(run(0, 7).stop(), PrefixStop::BudgetReleasing);
    }

    #[test]
    fn a_learned_cap_exit_lands_in_its_own_quiet_stretch_row() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);
        set_acted_fraction_enabled(true);
        set_quiet_stretch_enabled(true);

        begin_run();
        record_delivery(DeliveryBias::NONE, false, 0);
        record_delivery(DeliveryBias::NONE, false, 0);
        record_quiet_stretch(3, RunEnd::LearnedCapReached);
        begin_run();
        record_delivery(DeliveryBias::NONE, false, 0);
        record_delivery(DeliveryBias::NONE, false, 0);
        record_delivery(DeliveryBias::NONE, false, 0);
        record_quiet_stretch(4, RunEnd::StallCapReached);

        let s = snapshot().quiet_stretch;
        set_quiet_stretch_enabled(false);
        set_enabled(false);

        assert_eq!(s.runs, 2);
        assert_eq!(s.learned_cap_reached[hist_bucket(2)], 1);
        assert_eq!(s.learned_cap_reached.iter().sum::<u64>(), 1);
        assert_eq!(s.stall_cap_reached[hist_bucket(3)], 1);
        assert_eq!(s.stall_cap_reached.iter().sum::<u64>(), 1);
        assert_eq!(s.plan_complete.iter().sum::<u64>(), 0);
        assert_eq!(s.iterations_exhausted.iter().sum::<u64>(), 0);
        assert_eq!(s.deadlock.iter().sum::<u64>(), 0);
    }

    #[test]
    fn recovery_windows_close_on_the_first_message_to_the_restarted_node() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        begin_run();
        record_crash(0, 0, 0);
        record_recover(0, 10, true);
        // A message to a node with no open window leaves the tallies alone.
        record_message_entry(1, 12);
        record_message_entry(0, 14);
        record_crash(1, 0, 0);
        record_recover(1, 20, false);

        begin_run();

        let s = snapshot();
        set_enabled(false);

        assert_eq!(s.recovery_window.count, 1);
        assert_eq!(s.recovery_window.max, 4);
        assert_eq!(s.recovery_window.p50, 4);
        assert_eq!(s.recovery_window.mean_events_open, 4.0);
        // Node 1 recovered and the run ended with nothing delivered to it.
        assert_eq!(s.recovery_window.unclosed, 1);
        assert_eq!(s.ordered_h3.runs, 1);
        assert_eq!(s.ordered_h3.runs_with_h3, 1);
        // Node 1 crashed after node 0's window had already closed.
        assert_eq!(s.ordered_h3.runs_with_overlap, 0);
        assert_eq!(s.ordered_h3.by_fault_events[4].runs, 1);
    }

    #[test]
    fn a_crash_inside_an_open_recovery_window_is_an_overlap() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        begin_run();
        record_crash(0, 0, 0);
        record_recover(0, 5, false);
        record_crash(1, 0, 0);
        record_recover(1, 9, false);
        record_message_entry(0, 11);
        begin_run();

        let s = snapshot();
        set_enabled(false);

        assert_eq!(s.ordered_h3.runs_with_h3, 1);
        assert_eq!(s.ordered_h3.runs_with_overlap, 1);
        assert_eq!(s.recovery_window.count, 1);
        assert_eq!(s.recovery_window.max, 6);
    }

    #[test]
    fn timeline_key_growth_accumulates_into_one_curve_point() {
        let _serial = config_override::exclusive_session();
        set_enabled(true);

        record_timeline_keys(10, 10, 10);
        record_timeline_keys(12, 3, 13);

        let s = snapshot().timeline_keys;
        set_enabled(false);

        assert_eq!(s.runs, 2);
        assert_eq!(s.keys_in_run_sum, 22);
        assert_eq!(s.cumulative_distinct_keys, 13);
        assert_eq!(s.distinct_keys_live, 13);
        assert_eq!(s.growth_curve.len(), 1);
        assert_eq!(s.growth_curve[0].first_run, 1);
        assert_eq!(s.growth_curve[0].runs, 2);
        assert_eq!(s.growth_curve[0].cumulative_distinct, 13);
        // Saturation needs a full point to be judged, so a short session
        // reports none rather than declaring saturation early.
        assert_eq!(s.saturation_run_index, 0);
    }
}
