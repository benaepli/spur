//! A campaign runs several exploration strategies (arms) inside one session
//! under one active-time budget, giving each arm its own feedback store that
//! persists across the slices it is given, and attributes every run to its
//! arm in the runs table.
//!
//! Violations and prefix depth are not known while the session runs; they
//! are graded afterwards, per arm, by joining the checker's and grader's
//! output to the runs table. What the campaign can read in-process is the
//! utilization counters, taken as a delta over each slice, and that is all a
//! reward is: a proxy an allocation may rank arms by. The default allocation
//! needs no proxy at all.
//!
//! A budget in seconds ends the session on active time and is not
//! reproducible run for run; `deterministic_slice_runs` sizes slices in runs
//! for a reproducible campaign.

use crate::compiler::cfg::Program;
use crate::simulator::config_override;
use crate::simulator::coverage::GlobalState;
use crate::simulator::explorer::{
    AosExplorer, CurriculumExplorer, CurriculumRnrExplorer, EXPLORER_CONFIG_KEYS, ExploreSummary,
    ExplorerConfig, RunAttribution, SessionSummary, SingleRunConfig, StepCtx, StepReport, Strategy,
    check_top_level_keys, dispatch_feedback, run_single_simulation,
};
use crate::simulator::feedback::{
    CfgFeedback, CoverageConfig, Feedback, FeedbackMode, FullFeedback, NoFeedback, TimelineFeedback,
};
use crate::simulator::history::{HistoryWriter, LogBackend, create_writer};
use crate::simulator::rng::{LiveRng, SCHEDULE_SALT, WORKLOAD_SALT, derive_seed};
use crate::simulator::util_stats;
use log::{error, info};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Instant;

/// Top-level keys the campaign adds to the explorer envelope.
pub const CAMPAIGN_CONFIG_KEYS: &[&str] = &["campaign"];

/// Envelope fields an arm may not override: they describe the session, and
/// a per-arm value would leave the session's counters and budget unreadable.
const SESSION_LEVEL_KEYS: &[&str] = &[
    "session_seed",
    "stats",
    "strict_config_keys",
    "emit_acted_fraction",
    "quiet_stretch_telemetry",
    "emit_prefix_extension",
    "emit_multiplier_authority",
    "wall_budget_sec",
    "campaign",
];

const ARM_SALT: u64 = 0x4152_4d53_4545_4453;

#[derive(Clone, Debug, Deserialize)]
pub struct CampaignConfig {
    #[serde(flatten)]
    pub envelope: ExplorerConfig,
    pub campaign: CampaignBlock,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CampaignBlock {
    /// Active-time budget for the whole session, in seconds.
    pub wall_budget_sec: f64,
    #[serde(default)]
    pub allocation: Allocation,
    #[serde(default)]
    pub reward: Reward,
    /// Runs a grid arm issues per step.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// When set, a slice is this many runs and the session ends after
    /// `deterministic_rounds` slices per arm, independent of the clock.
    #[serde(default)]
    pub deterministic_slice_runs: Option<u64>,
    #[serde(default = "default_deterministic_rounds")]
    pub deterministic_rounds: u64,
    /// Horizon and decay handed to curriculum arms.
    #[serde(default = "default_horizon")]
    pub curriculum_horizon_runs: u64,
    #[serde(default = "default_half_life")]
    pub decay_half_life_runs: u64,
    pub arms: Vec<ArmSpec>,
}

fn default_batch_size() -> usize {
    (2 * rayon::current_num_threads()).max(32)
}
fn default_deterministic_rounds() -> u64 {
    4
}
fn default_horizon() -> u64 {
    100_000
}
fn default_half_life() -> u64 {
    2000
}
fn default_min_slice() -> f64 {
    20.0
}
fn default_eta() -> f64 {
    2.0
}
fn default_keep_top() -> usize {
    1
}
fn default_ucb_c() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArmSpec {
    pub id: String,
    pub mode: ArmMode,
    /// Dotted envelope paths to values, applied on top of the envelope.
    #[serde(default)]
    pub overlay: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArmMode {
    /// The exhaustive grid, walked in rounds, wrapping when exhausted.
    Grid,
    Curriculum,
    CurriculumRnr,
    Aos,
}

/// How the budget is split across arms. Every kind gives each arm at least
/// two slices; only `halving` and `bandit` read the reward.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Allocation {
    RoundRobin {
        #[serde(default = "default_min_slice")]
        min_slice_sec: f64,
    },
    Luby {
        #[serde(default = "default_min_slice")]
        min_slice_sec: f64,
    },
    Halving {
        #[serde(default = "default_eta")]
        eta: f64,
        #[serde(default = "default_min_slice")]
        min_slice_sec: f64,
        #[serde(default = "default_keep_top")]
        keep_top: usize,
    },
    Bandit {
        #[serde(default = "default_min_slice")]
        min_slice_sec: f64,
        #[serde(default = "default_ucb_c")]
        ucb_c: f64,
    },
}

impl Default for Allocation {
    fn default() -> Self {
        Allocation::RoundRobin {
            min_slice_sec: default_min_slice(),
        }
    }
}

impl Allocation {
    fn min_slice_sec(&self) -> f64 {
        match self {
            Allocation::RoundRobin { min_slice_sec }
            | Allocation::Luby { min_slice_sec }
            | Allocation::Halving { min_slice_sec, .. }
            | Allocation::Bandit { min_slice_sec, .. } => *min_slice_sec,
        }
    }

    fn reads_reward(&self) -> bool {
        matches!(self, Allocation::Halving { .. } | Allocation::Bandit { .. })
    }
}

/// An integer counter of the utilization snapshot, read as a delta over a
/// slice. `termination_completed` is the default: runs that finished their
/// plan rather than exhausting their step budget.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Reward {
    #[default]
    TerminationCompleted,
    HazardCrossing,
    AbsorptionActed,
    TimelineNovelty,
    StepsUsed,
    Runs,
}

impl Reward {
    fn path(self) -> &'static str {
        match self {
            Reward::TerminationCompleted => "termination.all.plan_complete",
            Reward::HazardCrossing => "crash_recovery.crossing_deliveries",
            Reward::AbsorptionActed => "delivery_effects.all.acted",
            Reward::TimelineNovelty => "timeline_keys.cumulative_distinct_keys",
            Reward::StepsUsed => "termination.all.steps_used_sum",
            Reward::Runs => "termination.all.runs",
        }
    }

    fn read(self, delta: &Value) -> f64 {
        let mut cursor = delta;
        for segment in self.path().split('.') {
            match cursor.get(segment) {
                Some(child) => cursor = child,
                None => return 0.0,
            }
        }
        cursor.as_f64().unwrap_or(0.0)
    }
}

impl CampaignBlock {
    pub fn validate(&self) -> Result<(), String> {
        if self.deterministic_slice_runs.is_none()
            && (!self.wall_budget_sec.is_finite() || self.wall_budget_sec <= 0.0)
        {
            return Err(
                "campaign: wall_budget_sec must be > 0 unless deterministic_slice_runs is set"
                    .into(),
            );
        }
        if self.deterministic_slice_runs == Some(0) || self.deterministic_rounds == 0 {
            return Err(
                "campaign: deterministic_slice_runs and deterministic_rounds must be >= 1".into(),
            );
        }
        if self.arms.is_empty() {
            return Err("campaign: arms must be non-empty".into());
        }
        if self.batch_size == 0 {
            return Err("campaign: batch_size must be >= 1".into());
        }
        let mut seen = std::collections::HashSet::new();
        for arm in &self.arms {
            let ok_id = !arm.id.is_empty()
                && arm
                    .id
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                && arm
                    .id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
            if !ok_id {
                return Err(format!(
                    "campaign: arm id `{}` must match [a-z0-9][a-z0-9_-]*",
                    arm.id
                ));
            }
            if !seen.insert(arm.id.as_str()) {
                return Err(format!("campaign: duplicate arm id `{}`", arm.id));
            }
            for key in arm.overlay.keys() {
                let head = key.split('.').next().unwrap_or("");
                if SESSION_LEVEL_KEYS.contains(&head) {
                    return Err(format!(
                        "campaign: arm `{}` may not override session-level key `{}`",
                        arm.id, key
                    ));
                }
            }
        }
        match &self.allocation {
            Allocation::Halving { eta, keep_top, .. } => {
                if *eta <= 1.0 {
                    return Err("campaign: halving eta must be > 1".into());
                }
                if *keep_top == 0 {
                    return Err("campaign: halving keep_top must be >= 1".into());
                }
            }
            Allocation::Bandit { ucb_c, .. } => {
                if *ucb_c < 0.0 {
                    return Err("campaign: bandit ucb_c must be >= 0".into());
                }
            }
            _ => {}
        }
        if self.allocation.min_slice_sec() <= 0.0 {
            return Err("campaign: min_slice_sec must be > 0".into());
        }
        Ok(())
    }
}

/// The envelope with one arm's overlay applied, checked the way a `--set`
/// override is checked: under strict keys a path that reaches no field is
/// an error, not a silently unchanged arm.
pub fn arm_config(
    envelope: &Value,
    spec: &ArmSpec,
    strict: bool,
) -> Result<ExplorerConfig, String> {
    let mut root = envelope.clone();
    if let Some(obj) = root.as_object_mut() {
        obj.remove("campaign");
    }
    let mut assignments = Vec::with_capacity(spec.overlay.len());
    for (path, value) in &spec.overlay {
        config_override::set_dotted(&mut root, path, value.clone())
            .map_err(|e| format!("campaign: arm `{}` overlay `{}`: {}", spec.id, path, e))?;
        assignments.push(format!("{}={}", path, value));
    }
    let text =
        serde_json::to_string(&root).map_err(|e| format!("campaign: arm `{}`: {}", spec.id, e))?;
    if strict {
        check_top_level_keys(&text, &[EXPLORER_CONFIG_KEYS])
            .map_err(|e| format!("campaign: arm `{}`: {}", spec.id, e))?;
    }
    let config: ExplorerConfig =
        serde_json::from_str(&text).map_err(|e| format!("campaign: arm `{}`: {}", spec.id, e))?;
    if strict {
        config_override::check_override_paths(&config, &assignments)
            .map_err(|e| format!("campaign: arm `{}`: {}", spec.id, e))?;
    }
    config
        .validate()
        .map_err(|e| format!("campaign: arm `{}`: {}", spec.id, e))?;
    if matches!(spec.mode, ArmMode::Aos | ArmMode::CurriculumRnr)
        && !matches!(
            config.feedback.mode,
            FeedbackMode::Timeline | FeedbackMode::Both
        )
    {
        return Err(format!(
            "campaign: arm `{}` ({:?}) requires feedback.mode = \"timeline\" or \"both\" on its own config",
            spec.id, spec.mode
        ));
    }
    Ok(config)
}

/// One strategy with persistent state; `step` runs one internally parallel
/// batch of at most `max_runs` runs where the strategy can honour a cap.
pub(crate) trait Arm {
    fn step(&mut self, ctx: &StepCtx, max_runs: usize) -> StepReport;
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>>;
    fn epochs(&self) -> u64 {
        0
    }
}

struct StrategyArm<F: Feedback>(Box<dyn Strategy<F>>);

impl<F: Feedback> Arm for StrategyArm<F> {
    fn step(&mut self, ctx: &StepCtx, _max_runs: usize) -> StepReport {
        self.0.step(ctx)
    }
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>> {
        self.0.vertex_coverage()
    }
}

/// The exhaustive grid as an arm: configurations are visited in rounds and
/// the walk wraps when the grid is exhausted, so the arm never starves.
pub(crate) struct GridArm<F: Feedback> {
    config: ExplorerConfig,
    configs: Vec<SingleRunConfig>,
    cursor: u64,
    global_state: GlobalState<F>,
    batch_size: usize,
    arm_seed: u64,
}

impl<F: Feedback> GridArm<F> {
    fn new(config: ExplorerConfig, batch_size: usize, arm_seed: u64) -> Self {
        let configs = config.expand_grid();
        Self {
            config,
            configs,
            cursor: 0,
            global_state: GlobalState::new(),
            batch_size,
            arm_seed,
        }
    }
}

impl<F: Feedback> Arm for GridArm<F> {
    fn step(&mut self, ctx: &StepCtx, max_runs: usize) -> StepReport {
        let n = self.configs.len() as u64;
        if n == 0 {
            return StepReport {
                runs: 0,
                failed: 0,
                best_score: 0.0,
            };
        }
        let count = self.batch_size.min(max_runs.max(1));
        let batch: Vec<(i64, usize)> = (0..count)
            .map(|_| {
                let run_id = ctx.run_counter.fetch_add(1, Ordering::Relaxed);
                let config_index = (self.cursor % n) as usize;
                self.cursor += 1;
                (run_id, config_index)
            })
            .collect();
        let global_state = &self.global_state;
        let config = &self.config;
        let configs = &self.configs;
        let arm_seed = self.arm_seed;
        let scores: Vec<Option<f64>> = batch
            .par_iter()
            .map(|&(run_id, config_index)| {
                match run_single_simulation::<F, LiveRng>(
                    ctx.program,
                    ctx.writer,
                    global_state,
                    run_id,
                    &configs[config_index],
                    &config.feedback.weights,
                    derive_seed(arm_seed, run_id, WORKLOAD_SALT),
                    derive_seed(arm_seed, run_id, SCHEDULE_SALT),
                    None,
                    &ctx.attribution.with_config(config_index),
                ) {
                    Ok(r) => Some(r.score),
                    Err(e) => {
                        error!("Campaign run {} failed: {}", run_id, e);
                        None
                    }
                }
            })
            .collect();
        StepReport {
            runs: scores.len() as u64,
            failed: scores.iter().filter(|s| s.is_none()).count() as u64,
            best_score: scores.iter().flatten().cloned().fold(0.0, f64::max),
        }
    }

    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>> {
        F::vertex_coverage(&self.global_state.feedback)
    }

    fn epochs(&self) -> u64 {
        let per_epoch = self.configs.len() as u64 * self.config.num_runs_per_config.max(1) as u64;
        if per_epoch == 0 {
            0
        } else {
            self.cursor / per_epoch
        }
    }
}

struct BuiltArm {
    spec: ArmSpec,
    weights: CoverageConfig,
    attribution: RunAttribution,
    arm: Box<dyn Arm>,
}

fn build_arm(
    spec: &ArmSpec,
    cfg: ExplorerConfig,
    block: &CampaignBlock,
    index: usize,
    session_seed: u64,
) -> BuiltArm {
    let arm_seed = derive_seed(session_seed, index as i64, ARM_SALT);
    let weights = cfg.feedback.weights;
    let feedback = cfg.feedback.clone();
    let arm: Box<dyn Arm> = dispatch_feedback!(feedback, F => match spec.mode {
        ArmMode::Grid => Box::new(GridArm::<F>::new(cfg, block.batch_size, arm_seed)) as Box<dyn Arm>,
        ArmMode::Curriculum => Box::new(StrategyArm::<F>(Box::new(CurriculumExplorer::<F>::new(
            cfg, block.batch_size, block.curriculum_horizon_runs, block.decay_half_life_runs, arm_seed,
        )))) as Box<dyn Arm>,
        ArmMode::CurriculumRnr => Box::new(StrategyArm::<F>(Box::new(CurriculumRnrExplorer::<F>::new(
            cfg, block.batch_size, weights, block.curriculum_horizon_runs, arm_seed,
        )))) as Box<dyn Arm>,
        ArmMode::Aos => Box::new(StrategyArm::<F>(Box::new(AosExplorer::<F>::new(
            cfg, block.batch_size, weights, arm_seed,
        )))) as Box<dyn Arm>,
    });
    BuiltArm {
        spec: spec.clone(),
        weights,
        attribution: RunAttribution {
            arm: Arc::from(spec.id.as_str()),
            arm_index: index as i32,
            config_index: -1,
        },
        arm,
    }
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceBudget {
    Seconds(f64),
    Runs(u64),
}

#[derive(Clone, Copy, Debug)]
pub struct Slice {
    pub arm: usize,
    pub round: usize,
    pub budget: SliceBudget,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ArmLedger {
    pub slices: u64,
    pub runs: u64,
    pub wall_ms: u64,
    pub reward: f64,
    pub dropped_at_round: Option<usize>,
}

impl ArmLedger {
    /// Reward per unit of what the slices were measured in.
    fn rate(&self, deterministic: bool) -> f64 {
        let exposure = if deterministic {
            self.runs as f64
        } else {
            self.wall_ms as f64 / 1000.0
        };
        if exposure <= 0.0 {
            0.0
        } else {
            self.reward / exposure
        }
    }
}

/// The i-th term of the Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1,
/// 2, 4, 8, ...
pub fn luby(i: u64) -> u64 {
    let mut k = 1u64;
    while (1u64 << k) - 1 < i {
        k += 1;
    }
    if (1u64 << k) - 1 == i {
        1u64 << (k - 1)
    } else {
        luby(i - (1u64 << (k - 1)) + 1)
    }
}

/// Rounds a successive-halving schedule needs to get from `arms` survivors
/// down to `keep_top`.
pub fn halving_rounds(arms: usize, eta: f64, keep_top: usize) -> usize {
    if arms <= keep_top {
        return 0;
    }
    ((arms as f64 / keep_top as f64).ln() / eta.ln())
        .ceil()
        .max(1.0) as usize
}

/// The next allocation: a pure function of the configuration, the slices
/// issued so far and the ledger, so a deterministic campaign replays.
pub struct Planner {
    allocation: Allocation,
    arms: usize,
    wall_sec: f64,
    unit_sec: f64,
    deterministic: Option<u64>,
    deterministic_rounds: u64,
    issued: u64,
    survivors: Vec<usize>,
    round: usize,
    pos: usize,
    total_rounds: usize,
    slice0_mult: f64,
}

impl Planner {
    pub fn new(block: &CampaignBlock, arms: usize) -> Self {
        let k = arms.max(1) as f64;
        let unit_sec = block
            .allocation
            .min_slice_sec()
            .min(block.wall_budget_sec / (2.0 * k))
            .max(1e-3);
        let (total_rounds, slice0_mult) = match &block.allocation {
            Allocation::Halving { eta, keep_top, .. } => {
                let rounds = halving_rounds(arms, *eta, *keep_top);
                let slice0 =
                    (0.5 * block.wall_budget_sec / (k * (rounds as f64 + 1.0))).max(unit_sec);
                (rounds, slice0 / unit_sec)
            }
            _ => (0, 1.0),
        };
        Self {
            allocation: block.allocation.clone(),
            arms,
            wall_sec: block.wall_budget_sec,
            unit_sec,
            deterministic: block.deterministic_slice_runs,
            deterministic_rounds: block.deterministic_rounds,
            issued: 0,
            survivors: (0..arms).collect(),
            round: 0,
            pos: 0,
            total_rounds,
            slice0_mult,
        }
    }

    pub fn unit_sec(&self) -> f64 {
        self.unit_sec
    }

    fn spent(&self, elapsed_sec: f64) -> bool {
        match self.deterministic {
            Some(_) => self.issued >= self.deterministic_rounds * self.arms as u64,
            None => elapsed_sec >= self.wall_sec,
        }
    }

    fn budget(&self, mult: f64) -> SliceBudget {
        match self.deterministic {
            Some(n) => SliceBudget::Runs(((n as f64) * mult).round().max(1.0) as u64),
            None => SliceBudget::Seconds(self.unit_sec * mult),
        }
    }

    pub fn next(&mut self, ledger: &mut [ArmLedger], elapsed_sec: f64) -> Option<Slice> {
        if self.arms == 0 || self.spent(elapsed_sec) {
            return None;
        }
        let deterministic = self.deterministic.is_some();
        let t = self.issued;
        self.issued += 1;
        let k = self.arms as u64;
        let slice = match &self.allocation {
            Allocation::RoundRobin { .. } => {
                let mult = if deterministic {
                    1.0
                } else {
                    (self.wall_sec / (k as f64 * self.unit_sec * 4.0))
                        .floor()
                        .max(1.0)
                };
                Slice {
                    arm: (t % k) as usize,
                    round: (t / k) as usize,
                    budget: self.budget(mult),
                }
            }
            Allocation::Luby { .. } => Slice {
                arm: (t % k) as usize,
                round: (t / k) as usize,
                budget: self.budget(luby(t / k + 1) as f64),
            },
            Allocation::Halving { eta, keep_top, .. } => {
                if self.pos >= self.survivors.len() && self.round < self.total_rounds {
                    // The round is complete: keep the best fraction, drop the rest.
                    let keep = ((self.survivors.len() as f64 / eta).ceil() as usize).max(*keep_top);
                    let mut ranked = self.survivors.clone();
                    ranked.sort_by(|&a, &b| {
                        ledger[b]
                            .rate(deterministic)
                            .partial_cmp(&ledger[a].rate(deterministic))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.cmp(&b))
                    });
                    for &dropped in &ranked[keep.min(ranked.len())..] {
                        ledger[dropped].dropped_at_round = Some(self.round);
                    }
                    ranked.truncate(keep);
                    ranked.sort_unstable();
                    self.survivors = ranked;
                    self.round += 1;
                    self.pos = 0;
                }
                if self.round < self.total_rounds && self.survivors.len() > *keep_top {
                    let arm = self.survivors[self.pos];
                    self.pos += 1;
                    Slice {
                        arm,
                        round: self.round,
                        budget: self.budget(self.slice0_mult * eta.powi(self.round as i32)),
                    }
                } else {
                    let i = self.pos % self.survivors.len();
                    self.pos += 1;
                    Slice {
                        arm: self.survivors[i],
                        round: self.round,
                        budget: self.budget(1.0),
                    }
                }
            }
            Allocation::Bandit { ucb_c, .. } => {
                let arm = if t < k {
                    t as usize
                } else {
                    let max_rate = ledger
                        .iter()
                        .map(|l| l.rate(deterministic))
                        .fold(0.0, f64::max);
                    let total = ledger.iter().map(|l| l.slices).sum::<u64>().max(1) as f64;
                    let mut best = 0usize;
                    let mut best_score = f64::NEG_INFINITY;
                    for (i, l) in ledger.iter().enumerate() {
                        let exploit = if max_rate > 0.0 {
                            l.rate(deterministic) / max_rate
                        } else {
                            0.0
                        };
                        let explore = ucb_c * (total.ln() / (l.slices.max(1) as f64)).sqrt();
                        let score = exploit + explore;
                        if score > best_score {
                            best_score = score;
                            best = i;
                        }
                    }
                    best
                };
                Slice {
                    arm,
                    round: (t / k) as usize,
                    budget: self.budget(1.0),
                }
            }
        };
        Some(slice)
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct ArmReport {
    pub index: usize,
    pub id: String,
    pub mode: ArmMode,
    pub overlay: Map<String, Value>,
    pub slices: u64,
    pub runs: u64,
    pub wall_ms: u64,
    pub reward: f64,
    pub reward_rate: f64,
    pub epochs: u64,
    pub dropped_at_round: Option<usize>,
    /// Integer utilization counters attributed to this arm, as deltas.
    pub counters: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct HistoryEntry {
    pub slice: u64,
    pub arm: usize,
    pub round: usize,
    pub budget: SliceBudget,
    pub started_ms: u64,
    pub wall_ms: u64,
    pub runs: u64,
    pub reward: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CampaignReport {
    pub wall_budget_sec: f64,
    pub elapsed_sec: f64,
    pub session_seed: u64,
    pub allocation: Allocation,
    pub reward: Reward,
    pub batch_size: usize,
    pub deterministic_slice_runs: Option<u64>,
    pub slice_unit_sec: f64,
    pub runs_total: u64,
    pub cancelled: bool,
    pub arms: Vec<ArmReport>,
    pub history: Vec<HistoryEntry>,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Runs a campaign from a config carrying a `campaign` block.
pub fn run_explorer_campaign(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    info!("Starting Campaign Explorer...");
    info!("Config: {}", config_json_path);
    let config_json = config_override::load_config_text(config_json_path)?;
    let config: CampaignConfig = serde_json::from_str(&config_json)?;
    let strict = config.envelope.strict_config_keys;
    if strict {
        check_top_level_keys(&config_json, &[EXPLORER_CONFIG_KEYS, CAMPAIGN_CONFIG_KEYS])?;
        config_override::check_override_paths(
            &config_check_view(&config_json)?,
            &config_override::active_overrides(),
        )?;
    }
    config
        .envelope
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;
    config
        .campaign
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;
    if config.campaign.allocation.reads_reward() && !config.envelope.stats {
        return Err("campaign: halving and bandit allocations read the utilization counters, so `stats` must be true".into());
    }
    let envelope_value: Value = serde_json::from_str(&config_json)?;
    let mut arm_configs = Vec::with_capacity(config.campaign.arms.len());
    for spec in &config.campaign.arms {
        arm_configs.push(arm_config(&envelope_value, spec, strict)?);
    }

    info!("campaign session_seed = {}", config.envelope.session_seed);
    util_stats::set_enabled(config.envelope.stats);
    util_stats::set_acted_fraction_enabled(config.envelope.emit_acted_fraction);
    util_stats::set_acceptance_distance_enabled(config.envelope.emit_acceptance_distance);
    util_stats::set_crash_census_enabled(config.envelope.emit_crash_census);
    util_stats::set_quiet_stretch_enabled(config.envelope.quiet_stretch_telemetry);
    util_stats::set_prefix_extension_enabled(config.envelope.emit_prefix_extension);
    util_stats::set_steer_audit_enabled(config.envelope.feedback.steer_audit);
    util_stats::set_steer_audit_always(config.envelope.feedback.steer_audit_always);
    util_stats::set_multiplier_audit_enabled(config.envelope.emit_multiplier_authority);
    run_campaign_impl(
        program,
        config,
        arm_configs,
        output_path,
        backend,
        cancelled,
    )
}

/// The parsed config as the override checker sees it: every envelope field
/// plus the campaign block, so a `--set campaign.x=...` with a misspelled
/// path is caught like any other.
fn config_check_view(config_json: &str) -> Result<Value, Box<dyn Error>> {
    let parsed: CampaignConfig = serde_json::from_str(config_json)?;
    let mut root = serde_json::to_value(&parsed.envelope)?;
    root["campaign"] = serde_json::to_value(&parsed.campaign)?;
    Ok(root)
}

fn run_campaign_impl(
    program: &Program,
    config: CampaignConfig,
    arm_configs: Vec<ExplorerConfig>,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    let block = &config.campaign;
    let session_seed = config.envelope.session_seed;
    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);
    let run_counter = AtomicI64::new(0);

    let mut arms: Vec<BuiltArm> = block
        .arms
        .iter()
        .zip(arm_configs)
        .enumerate()
        .map(|(i, (spec, cfg))| build_arm(spec, cfg, block, i, session_seed))
        .collect();
    let mut ledger: Vec<ArmLedger> = vec![ArmLedger::default(); arms.len()];
    let mut counters: Vec<Value> = vec![Value::Object(Map::new()); arms.len()];
    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut planner = Planner::new(block, arms.len());
    info!(
        "Campaign: {} arm(s), budget {:.1}s, slice unit {:.1}s, allocation {:?}, reward {:?}",
        arms.len(),
        block.wall_budget_sec,
        planner.unit_sec(),
        block.allocation,
        block.reward
    );

    let session_start = Instant::now();
    crate::simulator::explorer::session_clock_start();
    let mut runs_total: u64 = 0;
    let mut runs_failed: u64 = 0;
    let mut was_cancelled = false;
    let mut slice_no: u64 = 0;
    while let Some(slice) = planner.next(&mut ledger, session_start.elapsed().as_secs_f64()) {
        if cancelled.load(Ordering::Relaxed) {
            was_cancelled = true;
            break;
        }
        let arm = &mut arms[slice.arm];
        let ctx = StepCtx {
            program,
            writer: &writer,
            run_counter: &run_counter,
            weights: &arm.weights,
            session_seed,
            attribution: &arm.attribution,
        };
        let before = util_stats::snapshot_value();
        let started = session_start.elapsed();
        let slice_start = Instant::now();
        let mut slice_runs: u64 = 0;
        loop {
            if cancelled.load(Ordering::Relaxed) {
                was_cancelled = true;
                break;
            }
            let remaining = match slice.budget {
                SliceBudget::Runs(n) => n.saturating_sub(slice_runs) as usize,
                SliceBudget::Seconds(_) => usize::MAX,
            };
            if remaining == 0 {
                break;
            }
            let report = arm.arm.step(&ctx, remaining);
            slice_runs += report.runs;
            runs_failed += report.failed;
            if report.runs == 0 {
                break;
            }
            let spent = match slice.budget {
                SliceBudget::Runs(n) => slice_runs >= n,
                SliceBudget::Seconds(s) => slice_start.elapsed().as_secs_f64() >= s,
            };
            if spent {
                break;
            }
        }
        let wall_ms = slice_start.elapsed().as_millis() as u64;
        let delta = util_stats::delta(&before, &util_stats::snapshot_value());
        let reward = block.reward.read(&delta);
        util_stats::add(&mut counters[slice.arm], &delta);
        let l = &mut ledger[slice.arm];
        l.slices += 1;
        l.runs += slice_runs;
        l.wall_ms += wall_ms;
        l.reward += reward;
        runs_total += slice_runs;
        history.push(HistoryEntry {
            slice: slice_no,
            arm: slice.arm,
            round: slice.round,
            budget: slice.budget,
            started_ms: started.as_millis() as u64,
            wall_ms,
            runs: slice_runs,
            reward,
        });
        info!(
            "[{}] slice {} round {}: {} runs in {} ms, reward {:.0}",
            arm.spec.id, slice_no, slice.round, slice_runs, wall_ms, reward
        );
        slice_no += 1;
        if was_cancelled {
            break;
        }
    }

    let elapsed = session_start.elapsed();
    let flush_start = Instant::now();
    writer.shutdown();
    let writer_flush_ms = flush_start.elapsed().as_millis() as u64;

    let mut vertex_coverage: Option<HashMap<usize, u64>> = None;
    for arm in &arms {
        if let Some(cov) = arm.arm.vertex_coverage() {
            let merged = vertex_coverage.get_or_insert_with(HashMap::new);
            for (v, c) in cov {
                *merged.entry(v).or_insert(0) += c;
            }
        }
    }
    let deterministic = block.deterministic_slice_runs.is_some();
    let arm_reports: Vec<ArmReport> = arms
        .iter()
        .enumerate()
        .map(|(i, a)| ArmReport {
            index: i,
            id: a.spec.id.clone(),
            mode: a.spec.mode,
            overlay: a.spec.overlay.clone(),
            slices: ledger[i].slices,
            runs: ledger[i].runs,
            wall_ms: ledger[i].wall_ms,
            reward: ledger[i].reward,
            reward_rate: ledger[i].rate(deterministic),
            epochs: a.arm.epochs(),
            dropped_at_round: ledger[i].dropped_at_round,
            counters: counters[i].clone(),
        })
        .collect();
    info!(
        "Campaign finished: {} runs over {} slices in {:.1}s{}",
        runs_total,
        slice_no,
        elapsed.as_secs_f64(),
        if was_cancelled { " (cancelled)" } else { "" }
    );
    Ok(ExploreSummary {
        vertex_coverage,
        session: Some(SessionSummary {
            wall_ms: elapsed.as_millis() as u64,
            runs_completed: runs_total - runs_failed,
            runs_failed,
            runs_skipped: 0,
            wall_budget_sec: block.wall_budget_sec,
            budget_hit: !was_cancelled && block.deterministic_slice_runs.is_none(),
            writer_flush_ms,
        }),
        campaign: Some(CampaignReport {
            wall_budget_sec: block.wall_budget_sec,
            elapsed_sec: elapsed.as_secs_f64(),
            session_seed,
            allocation: block.allocation.clone(),
            reward: block.reward,
            batch_size: block.batch_size,
            deterministic_slice_runs: block.deterministic_slice_runs,
            slice_unit_sec: planner.unit_sec(),
            runs_total,
            cancelled: was_cancelled,
            arms: arm_reports,
            history,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "num_servers": {"min": 3, "max": 3},
        "num_write_ops": {"min": 2, "max": 2},
        "num_read_ops": {"min": 2, "max": 2},
        "num_crashes": {"min": 0, "max": 0},
        "dependency_density": [0.0],
        "num_runs_per_config": 1,
        "max_iterations": 100,
        "strict_config_keys": true,
        "campaign": {
            "wall_budget_sec": 900,
            "arms": [
                {"id": "grid", "mode": "grid"},
                {"id": "short", "mode": "grid", "overlay": {"max_iterations": 10}}
            ]
        }
    }"#;

    fn block(k: usize, allocation: &str) -> CampaignBlock {
        let arms: Vec<String> = (0..k)
            .map(|i| format!("{{\"id\": \"a{i}\", \"mode\": \"grid\"}}"))
            .collect();
        serde_json::from_str(&format!(
            "{{\"wall_budget_sec\": 900, \"allocation\": {allocation}, \"arms\": [{}]}}",
            arms.join(",")
        ))
        .expect("block parses")
    }

    #[test]
    fn luby_sequence() {
        let got: Vec<u64> = (1..=15).map(luby).collect();
        assert_eq!(got, vec![1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]);
    }

    #[test]
    fn halving_rounds_and_survivors() {
        assert_eq!(halving_rounds(6, 2.0, 1), 3);
        assert_eq!(halving_rounds(7, 2.0, 1), 3);
        assert_eq!(halving_rounds(1, 2.0, 1), 0);
        let b = block(6, r#"{"kind": "halving", "eta": 2.0, "keep_top": 1}"#);
        let mut planner = Planner::new(&b, 6);
        let mut ledger = vec![ArmLedger::default(); 6];
        let mut survivors_per_round: Vec<usize> = Vec::new();
        let mut last_round = usize::MAX;
        for t in 0..60 {
            let s = planner
                .next(&mut ledger, 0.0)
                .expect("budget is not spent by the clock at 0 s");
            if s.round != last_round {
                survivors_per_round.push(planner.survivors.len());
                last_round = s.round;
            }
            // Arm i earns reward i per slice, so the lowest-indexed arms drop first.
            ledger[s.arm].slices += 1;
            ledger[s.arm].wall_ms += 1000;
            ledger[s.arm].reward += s.arm as f64;
            let _ = t;
        }
        assert_eq!(&survivors_per_round[..4], &[6, 3, 2, 1]);
        assert_eq!(ledger[0].dropped_at_round, Some(0));
        assert_eq!(ledger[5].dropped_at_round, None);
        // Slice 0 of halving takes half the budget over the rounds.
        assert!(
            (planner.slice0_mult * planner.unit_sec - 20.0).abs() < 1e-9,
            "slice 0 is the larger of the unit and half the budget over the rounds"
        );
    }

    #[test]
    fn slice_unit_shrinks_for_short_campaigns() {
        let mut b = block(7, r#"{"kind": "round_robin", "min_slice_sec": 20}"#);
        b.wall_budget_sec = 10.0;
        let p = Planner::new(&b, 7);
        assert!((p.unit_sec() - 10.0 / 14.0).abs() < 1e-9);
    }

    #[test]
    fn deterministic_plan_is_a_pure_function_of_config() {
        let mut b = block(3, r#"{"kind": "luby"}"#);
        b.deterministic_slice_runs = Some(16);
        b.deterministic_rounds = 4;
        let plan = |b: &CampaignBlock| -> Vec<(usize, u64)> {
            let mut p = Planner::new(b, 3);
            let mut l = vec![ArmLedger::default(); 3];
            let mut out = Vec::new();
            while let Some(s) = p.next(&mut l, 0.0) {
                l[s.arm].slices += 1;
                l[s.arm].runs += 16;
                let runs = match s.budget {
                    SliceBudget::Runs(n) => n,
                    SliceBudget::Seconds(_) => panic!("deterministic plan issues runs"),
                };
                out.push((s.arm, runs));
            }
            out
        };
        let a = plan(&b);
        assert_eq!(a, plan(&b));
        assert_eq!(a.len(), 12);
        assert_eq!(a[0], (0, 16));
        assert_eq!(a[6], (0, 32));
    }

    #[test]
    fn campaign_key_is_rejected_outside_campaign_mode() {
        assert!(check_top_level_keys(MINIMAL, &[EXPLORER_CONFIG_KEYS]).is_err());
        assert!(
            check_top_level_keys(MINIMAL, &[EXPLORER_CONFIG_KEYS, CAMPAIGN_CONFIG_KEYS]).is_ok()
        );
    }

    #[test]
    fn overlays_are_checked_like_overrides() {
        let config: CampaignConfig = serde_json::from_str(MINIMAL).expect("parses");
        let envelope: Value = serde_json::from_str(MINIMAL).expect("parses");
        let short = arm_config(&envelope, &config.campaign.arms[1], true).expect("overlay applies");
        assert_eq!(short.max_iterations, 10);
        let grid = arm_config(&envelope, &config.campaign.arms[0], true).expect("no overlay");
        assert_eq!(grid.max_iterations, 100);

        let typo = ArmSpec {
            id: "typo".into(),
            mode: ArmMode::Grid,
            overlay: serde_json::from_str(r#"{"purgatory.delayed_probability": 0.5}"#).unwrap(),
        };
        let err = arm_config(&envelope, &typo, true).expect_err("a misspelled path fails");
        assert!(err.contains("purgatory.delayed_probability"), "{err}");

        let aos = ArmSpec {
            id: "aos".into(),
            mode: ArmMode::Aos,
            overlay: serde_json::from_str(r#"{"feedback.mode": "none"}"#).unwrap(),
        };
        let err = arm_config(&envelope, &aos, true).expect_err("aos needs timeline feedback");
        assert!(err.contains("timeline"), "{err}");
    }

    #[test]
    fn block_validation_rejects_bad_shapes() {
        let mut b = block(2, r#"{"kind": "round_robin"}"#);
        assert!(b.validate().is_ok());
        b.arms[1].id = b.arms[0].id.clone();
        assert!(b.validate().unwrap_err().contains("duplicate"));
        let mut b = block(2, r#"{"kind": "round_robin"}"#);
        b.arms[0]
            .overlay
            .insert("session_seed".into(), Value::from(7));
        assert!(b.validate().unwrap_err().contains("session-level"));
        let mut b = block(2, r#"{"kind": "halving", "eta": 1.0}"#);
        assert!(b.validate().unwrap_err().contains("eta"));
        b = block(0, r#"{"kind": "round_robin"}"#);
        assert!(b.validate().unwrap_err().contains("non-empty"));
    }

    #[test]
    fn reward_reads_integer_leaves_of_a_delta() {
        let delta: Value =
            serde_json::from_str(r#"{"termination": {"all": {"plan_complete": 7, "runs": 9}}}"#)
                .unwrap();
        assert_eq!(Reward::TerminationCompleted.read(&delta), 7.0);
        assert_eq!(Reward::Runs.read(&delta), 9.0);
        assert_eq!(Reward::HazardCrossing.read(&delta), 0.0);
    }
}
