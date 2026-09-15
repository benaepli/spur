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
use crate::simulator::deploy::params::DeployCache;

use crate::compiler::cfg::Program;
use crate::simulator::config_override;
use crate::simulator::coverage::GlobalState;
use crate::simulator::explorer::{
    AosExplorer, CurriculumExplorer, CurriculumRnrExplorer, EXPLORER_CONFIG_KEYS, ExploreSummary,
    ExplorerConfig, RunAttribution, SessionSummary, SingleRunConfig, StepCtx, Strategy,
    check_top_level_keys, dispatch_feedback, run_single_simulation,
};
use crate::simulator::feedback::{
    CfgFeedback, CoverageConfig, Feedback, FeedbackMode, FullFeedback, NoFeedback, TimelineFeedback,
};
use crate::simulator::history::{HistoryWriter, LogBackend, create_writer};
use crate::simulator::replay_corpus::{self, Corpus, Seed};
use crate::simulator::rng::{LiveRng, RecordRng, ReplayRng, SCHEDULE_SALT, WORKLOAD_SALT, derive_seed};
use crate::simulator::run_variant;
use crate::simulator::util_stats;
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::sync::Mutex;
use std::sync::{Arc, mpsc};
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

/// Where one slice stops issuing runs: its budget, its start, and the
/// session's cancellation flag.
pub(crate) struct SliceLimit<'a> {
    budget: SliceBudget,
    started: Instant,
    cancelled: &'a AtomicBool,
}

impl SliceLimit<'_> {
    /// The most runs the next batch may issue once `issued` runs in `batches`
    /// batches have been issued, or `None` when the slice is over. The first
    /// batch is always issued unless the session was cancelled, which also
    /// sets `cancelled`.
    fn next_batch(&self, issued: u64, batches: u64, cancelled: &mut bool) -> Option<usize> {
        if batches > 0 {
            let spent = match self.budget {
                SliceBudget::Runs(n) => issued >= n,
                SliceBudget::Seconds(s) => self.started.elapsed().as_secs_f64() >= s,
            };
            if spent {
                return None;
            }
        }
        if self.cancelled.load(Ordering::Relaxed) {
            *cancelled = true;
            return None;
        }
        match self.budget {
            SliceBudget::Runs(n) => match n.saturating_sub(issued) as usize {
                0 => None,
                remaining => Some(remaining),
            },
            SliceBudget::Seconds(_) => Some(usize::MAX),
        }
    }
}

/// The runs a slice issued, those among them that failed, and whether the
/// slice ended on cancellation.
#[derive(Default)]
pub(crate) struct SliceOutcome {
    runs: u64,
    failed: u64,
    cancelled: bool,
}

/// One strategy with persistent state. `run_slice` issues runs until the
/// limit ends the slice and returns only once every run it issued has
/// finished, so no run of one slice is in flight during another.
pub(crate) trait Arm {
    fn run_slice(&mut self, ctx: &StepCtx, limit: &SliceLimit) -> SliceOutcome;
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>>;
    fn epochs(&self) -> u64 {
        0
    }
}

struct StrategyArm<F: Feedback>(Box<dyn Strategy<F>>);

impl<F: Feedback> Arm for StrategyArm<F> {
    fn run_slice(&mut self, ctx: &StepCtx, limit: &SliceLimit) -> SliceOutcome {
        let mut out = SliceOutcome::default();
        let mut batches = 0u64;
        while limit
            .next_batch(out.runs, batches, &mut out.cancelled)
            .is_some()
        {
            let report = self.0.step(ctx);
            out.runs += report.runs;
            out.failed += report.failed;
            batches += 1;
            if report.runs == 0 {
                break;
            }
        }
        out
    }
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>> {
        self.0.vertex_coverage()
    }
}

/// The exhaustive grid as an arm: configurations are visited in rounds and
/// the walk wraps when the grid is exhausted, so the arm never starves.
///
/// Every fresh run records its scheduling draws. A fresh run that fires the
/// fault-crossing signal enters the arm's replay corpus with its draws cut
/// at the signal, and a run whose id is a replay slot becomes a child of the
/// next parent in turn when the corpus holds one. Only fresh runs advance
/// the grid cursor, so children do not thin the grid's coverage.
///
/// Runs are issued in batches whose assignment and corpus admissions are
/// exactly those of running each batch to completion before assigning the
/// next; `run_ordered_release` keeps workers busy across a batch's slowest
/// runs without changing either.
pub(crate) struct GridArm<F: Feedback> {
    config: ExplorerConfig,
    configs: Vec<SingleRunConfig>,
    global_state: GlobalState<F>,
    batch_size: usize,
    arm_seed: u64,
    assigner: Assigner<SingleRunConfig>,
}

/// What one run of a grid-arm batch is.
enum GridRun<C> {
    Fresh { config_index: usize },
    Child { seed: Seed<C>, prefix: bool },
}

/// What a grid-arm run hands back: its score, and the corpus entry it earned
/// when it was a fresh run that fired the signal.
struct GridOutcome<C> {
    score: Option<f64>,
    admit: Option<Seed<C>>,
}

/// The grid cursor and the replay corpus: the state a run's assignment reads.
struct Assigner<C> {
    grid_len: u64,
    cursor: u64,
    corpus: Corpus<C>,
    #[cfg(test)]
    admitted: Vec<u64>,
}

impl<C: Clone> Assigner<C> {
    fn new(grid_len: usize) -> Self {
        Self {
            grid_len: grid_len as u64,
            cursor: 0,
            corpus: Corpus::new(),
            #[cfg(test)]
            admitted: Vec::new(),
        }
    }

    fn fresh(&mut self) -> GridRun<C> {
        let config_index = (self.cursor % self.grid_len) as usize;
        self.cursor += 1;
        GridRun::Fresh { config_index }
    }

    /// The run `run_id` is, and whether it is a slot that found the corpus
    /// empty and runs fresh instead.
    fn assign(&mut self, run_id: i64) -> (GridRun<C>, bool) {
        if !replay_corpus::is_slot(run_id) {
            return (self.fresh(), false);
        }
        match self.corpus.next_child() {
            Some(seed) => (
                GridRun::Child {
                    seed,
                    prefix: replay_corpus::is_prefix(run_id),
                },
                false,
            ),
            None => (self.fresh(), true),
        }
    }

    fn admit(&mut self, seed: Seed<C>) {
        #[cfg(test)]
        self.admitted.push(seed.workload_seed);
        self.corpus.admit(seed);
    }
}

/// The borrowed state a grid run reads, shared by every worker.
struct GridRunner<'a, F: Feedback> {
    configs: &'a [SingleRunConfig],
    global_state: &'a GlobalState<F>,
    weights: &'a CoverageConfig,
    arm_seed: u64,
}

impl<F: Feedback> GridRunner<'_, F> {
    fn run(
        &self,
        ctx: &StepCtx,
        run_id: i64,
        run: &GridRun<SingleRunConfig>,
    ) -> GridOutcome<SingleRunConfig> {
        let bits = run_variant::grid_arm_bits(run_id);
        let schedule_seed = derive_seed(self.arm_seed, run_id, SCHEDULE_SALT);
        let weights = self.weights;
        match run {
            GridRun::Fresh { config_index } => {
                let config_index = *config_index;
                let workload_seed = derive_seed(self.arm_seed, run_id, WORKLOAD_SALT);
                let result = run_single_simulation::<F, RecordRng>(
                    ctx.program,
                    ctx.writer,
                    self.global_state,
                    run_id,
                    &self.configs[config_index],
                    weights,
                    workload_seed,
                    schedule_seed,
                    None,
                    &ctx.attribution
                        .with_config(config_index)
                        .with_variant_bits(bits),
                );
                match result {
                    Ok(r) => {
                        let tape = r.recording.unwrap_or_default();
                        util_stats::record_replay_tape_words(tape.len() as u64);
                        let admit = r.cut.and_then(|cut| {
                            cut.tape_pos.map(|pos| Seed {
                                tape: tape[..pos.min(tape.len())].into(),
                                workload_seed,
                                cfg: self.configs[config_index].clone(),
                                config_index,
                                cut_step: cut.step,
                            })
                        });
                        GridOutcome {
                            score: Some(r.score),
                            admit,
                        }
                    }
                    Err(e) => {
                        error!("Campaign run {} failed: {}", run_id, e);
                        GridOutcome {
                            score: None,
                            admit: None,
                        }
                    }
                }
            }
            GridRun::Child { seed, prefix } => {
                let attribution = ctx
                    .attribution
                    .with_config(seed.config_index)
                    .with_variant_bits(bits);
                let result = if *prefix {
                    run_single_simulation::<F, ReplayRng>(
                        ctx.program,
                        ctx.writer,
                        self.global_state,
                        run_id,
                        &seed.cfg,
                        weights,
                        seed.workload_seed,
                        schedule_seed,
                        Some(seed.tape.clone()),
                        &attribution,
                    )
                } else {
                    run_single_simulation::<F, LiveRng>(
                        ctx.program,
                        ctx.writer,
                        self.global_state,
                        run_id,
                        &seed.cfg,
                        weights,
                        seed.workload_seed,
                        schedule_seed,
                        None,
                        &attribution,
                    )
                };
                match result {
                    Ok(r) => {
                        let faithful = *prefix && r.cut.is_some_and(|c| c.step == seed.cut_step);
                        util_stats::record_replay_child(*prefix, r.cut.is_some(), faithful);
                        GridOutcome {
                            score: Some(r.score),
                            admit: None,
                        }
                    }
                    Err(e) => {
                        error!("Campaign run {} failed: {}", run_id, e);
                        GridOutcome {
                            score: None,
                            admit: None,
                        }
                    }
                }
            }
        }
    }
}

impl<F: Feedback> GridArm<F> {
    fn new(config: ExplorerConfig, batch_size: usize, arm_seed: u64) -> Self {
        let configs = config.expand_grid();
        let grid_len = configs.len();
        Self {
            config,
            configs,
            global_state: GlobalState::new(),
            batch_size,
            arm_seed,
            assigner: Assigner::new(grid_len),
        }
    }
}

impl<F: Feedback> Arm for GridArm<F> {
    fn run_slice(&mut self, ctx: &StepCtx, limit: &SliceLimit) -> SliceOutcome {
        let mut cancelled = false;
        if self.configs.is_empty() {
            limit.next_batch(0, 0, &mut cancelled);
            return SliceOutcome {
                runs: 0,
                failed: 0,
                cancelled,
            };
        }
        let runner = GridRunner {
            configs: &self.configs,
            global_state: &self.global_state,
            weights: &self.config.feedback.weights,
            arm_seed: self.arm_seed,
        };
        let blocked_before = util_stats::history_writer_blocked_ns();
        let mut pool = run_ordered_release(
            &mut self.assigner,
            self.batch_size,
            rayon::current_num_threads(),
            ctx.run_counter,
            |issued, batches| limit.next_batch(issued, batches, &mut cancelled),
            &|run_id, run| runner.run(ctx, run_id, run),
        );
        pool.session.writer_blocked_ns =
            util_stats::history_writer_blocked_ns().saturating_sub(blocked_before);
        util_stats::record_grid_pool(&pool.session);
        SliceOutcome {
            runs: pool.session.runs,
            failed: pool.failed,
            cancelled,
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
            self.assigner.cursor / per_epoch
        }
    }
}

/// Batches, beyond the oldest batch whose slots are not yet drawn, that may
/// have their fresh runs assigned and started.
const LOOKAHEAD_BATCHES: usize = 4;

/// One batch of the ordered-release pool.
///
/// A batch is `covered` when, at its issue, the corpus's remaining children
/// less the slots reserved by earlier undrawn batches were at least its own
/// slot count. Children only fall by the draws reserved, and admissions
/// never lower them, so no slot of a covered batch finds the corpus empty
/// and its fresh runs take the grid cursor's next values in position order.
/// Only a covered batch has its fresh runs assigned before its slots are
/// drawn. Slots are drawn once every earlier batch's admissions are applied,
/// and a batch's admissions are applied in position order once its slots are
/// drawn and its fresh runs have finished: the corpus sees the same draws and
/// admissions, in the same order, as batches run one at a time.
struct Batch<C> {
    first_id: i64,
    slots: u64,
    covered: bool,
    released: bool,
    admitted: bool,
    runs: Vec<Option<GridRun<C>>>,
    assigned: Vec<bool>,
    fresh_index: Vec<Option<usize>>,
    launched: Vec<bool>,
    admits: Vec<Option<Seed<C>>>,
    ready: usize,
    fresh_unfinished: usize,
    unfinished: usize,
}

impl<C: Clone> Batch<C> {
    fn new(first_id: i64, count: usize, slots: u64, covered: bool) -> Self {
        Self {
            first_id,
            slots,
            covered,
            released: false,
            admitted: false,
            runs: (0..count).map(|_| None).collect(),
            assigned: vec![false; count],
            fresh_index: vec![None; count],
            launched: vec![false; count],
            admits: (0..count).map(|_| None).collect(),
            ready: 0,
            fresh_unfinished: 0,
            unfinished: count,
        }
    }

    fn len(&self) -> usize {
        self.runs.len()
    }

    fn set(&mut self, pos: usize, run: GridRun<C>) {
        if let GridRun::Fresh { config_index } = run {
            self.fresh_index[pos] = Some(config_index);
            self.fresh_unfinished += 1;
        }
        self.runs[pos] = Some(run);
        self.assigned[pos] = true;
        self.ready += 1;
    }
}

/// A finished job: its batch, its position, its outcome (`None` when the
/// run panicked) and its wall time on the worker.
struct Completion<C> {
    batch: u64,
    pos: usize,
    outcome: Option<GridOutcome<C>>,
    wall_ns: u64,
}

/// Sends a job's completion exactly once, including when the run unwinds,
/// so the coordinator never waits for a job that is gone.
struct CompletionSender<C> {
    tx: mpsc::Sender<Completion<C>>,
    batch: u64,
    pos: usize,
    started: Instant,
    sent: bool,
}

impl<C> CompletionSender<C> {
    fn send(mut self, outcome: GridOutcome<C>) {
        self.deliver(Some(outcome));
    }

    fn deliver(&mut self, outcome: Option<GridOutcome<C>>) {
        self.sent = true;
        let _ = self.tx.send(Completion {
            batch: self.batch,
            pos: self.pos,
            outcome,
            wall_ns: self.started.elapsed().as_nanos() as u64,
        });
    }
}

impl<C> Drop for CompletionSender<C> {
    fn drop(&mut self) {
        if !self.sent {
            self.deliver(None);
        }
    }
}

struct PoolResult {
    session: util_stats::GridPoolSession,
    failed: u64,
}

/// Assigns every not yet assigned position of a batch in position order,
/// drawing its slots. Every earlier batch's admissions must have been
/// applied.
fn release_batch<C: Clone>(
    batch: &mut Batch<C>,
    assigner: &mut Assigner<C>,
    session: &mut util_stats::GridPoolSession,
) {
    for pos in 0..batch.len() {
        if batch.assigned[pos] {
            continue;
        }
        let (run, unfilled) = assigner.assign(batch.first_id + pos as i64);
        if unfilled {
            util_stats::record_replay_slot_unfilled();
            if batch.covered {
                session.unfilled_in_ungated_batches += 1;
            }
        }
        batch.set(pos, run);
    }
    batch.released = true;
}

/// Runs batches of at most `batch_size` runs on `workers` workers until
/// `next_batch` ends the slice, then drains: when this returns, every run it
/// issued has finished and every admission has been applied.
///
/// `next_batch(issued, batches)` gives the most runs the next batch may
/// issue, or `None` to stop issuing. Run ids are taken from `run_counter` in
/// order, a batch at a time, and nothing else may take ids while this runs.
/// Every run id's assignment and the corpus's sequence of draws and
/// admissions are those of assigning a batch whole, running it to
/// completion and admitting its fresh runs' entries in position order
/// before assigning the next. With one worker, runs also start in id order.
fn run_ordered_release<C, R>(
    assigner: &mut Assigner<C>,
    batch_size: usize,
    workers: usize,
    run_counter: &AtomicI64,
    mut next_batch: impl FnMut(u64, u64) -> Option<usize>,
    run: &R,
) -> PoolResult
where
    C: Clone + Send,
    R: Fn(i64, &GridRun<C>) -> GridOutcome<C> + Sync,
{
    let workers = workers.max(1);
    let batch_size = batch_size.max(1);
    let mut session = util_stats::GridPoolSession {
        workers: workers as u64,
        ..Default::default()
    };
    let mut failed = 0u64;
    let mut window: VecDeque<Batch<C>> = VecDeque::new();
    let mut front_seq = 0u64;
    let mut in_flight = 0usize;
    let mut ready = 0usize;
    let mut stopped = false;
    let pool_started = Instant::now();
    let (tx, rx) = mpsc::channel::<Completion<C>>();

    rayon::in_place_scope(|scope| {
        loop {
            // Draw slots and apply admissions, both in batch order.
            let mut earlier_admitted = true;
            for batch in window.iter_mut() {
                if batch.admitted {
                    continue;
                }
                if !earlier_admitted {
                    break;
                }
                if !batch.released {
                    let before = batch.ready;
                    release_batch(batch, assigner, &mut session);
                    ready += batch.ready - before;
                }
                if batch.fresh_unfinished > 0 {
                    earlier_admitted = false;
                    continue;
                }
                for pos in 0..batch.len() {
                    if let Some(seed) = batch.admits[pos].take() {
                        assigner.admit(seed);
                        util_stats::record_replay_parent_admitted();
                    }
                }
                batch.admitted = true;
            }
            while window
                .front()
                .is_some_and(|b| b.admitted && b.unfinished == 0)
            {
                window.pop_front();
                front_seq += 1;
            }

            // Issue batches only while workers would otherwise go without work.
            while !stopped && in_flight + ready < workers {
                let all_admitted = window.iter().all(|b| b.admitted);
                let undrawn = window.iter().filter(|b| !b.released).count();
                if !all_admitted && undrawn > LOOKAHEAD_BATCHES {
                    break;
                }
                let Some(cap) = next_batch(session.runs, session.batches) else {
                    stopped = true;
                    break;
                };
                let count = batch_size.min(cap);
                let first_id = run_counter.load(Ordering::Relaxed);
                let slots = (0..count)
                    .filter(|&i| replay_corpus::is_slot(first_id + i as i64))
                    .count() as u64;
                let reserved: u64 = window
                    .iter()
                    .filter(|b| !b.released)
                    .map(|b| b.slots)
                    .sum();
                let covered = assigner.corpus.remaining_children() >= reserved + slots;
                if !all_admitted && !covered {
                    break;
                }
                for _ in 0..count {
                    run_counter.fetch_add(1, Ordering::Relaxed);
                }
                session.runs += count as u64;
                session.batches += 1;
                if !covered {
                    session.capacity_gated_batches += 1;
                }
                let mut batch = Batch::new(first_id, count, slots, covered);
                if all_admitted {
                    release_batch(&mut batch, assigner, &mut session);
                } else {
                    for pos in 0..count {
                        if !replay_corpus::is_slot(first_id + pos as i64) {
                            batch.set(pos, assigner.fresh());
                        }
                    }
                }
                ready += batch.ready;
                window.push_back(batch);
            }

            // Start runs oldest batch first, in position order.
            for (i, batch) in window.iter_mut().enumerate() {
                if in_flight >= workers {
                    break;
                }
                if batch.ready == 0 {
                    continue;
                }
                for pos in 0..batch.len() {
                    if in_flight >= workers || batch.ready == 0 {
                        break;
                    }
                    if batch.launched[pos] {
                        continue;
                    }
                    let Some(grid_run) = batch.runs[pos].take() else {
                        continue;
                    };
                    batch.launched[pos] = true;
                    batch.ready -= 1;
                    ready -= 1;
                    in_flight += 1;
                    if !batch.released {
                        session.fresh_ahead_launched += 1;
                    }
                    let run_id = batch.first_id + pos as i64;
                    let tx = tx.clone();
                    let seq = front_seq + i as u64;
                    scope.spawn(move |_| {
                        let sender = CompletionSender {
                            tx,
                            batch: seq,
                            pos,
                            started: Instant::now(),
                            sent: false,
                        };
                        let outcome = run(run_id, &grid_run);
                        drop(grid_run);
                        sender.send(outcome);
                    });
                }
            }

            if in_flight == 0 {
                debug_assert!(stopped && ready == 0);
                debug_assert!(window.iter().all(|b| b.admitted && b.unfinished == 0));
                break;
            }

            let mut next = rx.recv().ok();
            while let Some(done) = next {
                in_flight -= 1;
                session.job_wall_ns += done.wall_ns;
                let batch = &mut window[(done.batch - front_seq) as usize];
                batch.unfinished -= 1;
                let fresh = batch.fresh_index[done.pos].is_some();
                if fresh {
                    batch.fresh_unfinished -= 1;
                }
                match done.outcome {
                    Some(outcome) => {
                        if outcome.score.is_none() {
                            failed += 1;
                        }
                        if fresh {
                            batch.admits[done.pos] = outcome.admit;
                        }
                    }
                    None => failed += 1,
                }
                next = rx.try_recv().ok();
            }
        }
    });
    session.pool_wall_ns = pool_started.elapsed().as_nanos() as u64;
    PoolResult { session, failed }
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
            variant_bits: 0,
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
    let deploy_cache = Arc::new(Mutex::new(DeployCache::default()));
    let mut arm_configs = Vec::with_capacity(config.campaign.arms.len());
    for spec in &config.campaign.arms {
        let mut arm = arm_config(&envelope_value, spec, strict)?;
        arm.bind(program, &deploy_cache)
            .map_err(|e| format!("campaign: arm `{}`: {}", spec.id, e))?;
        arm_configs.push(arm);
    }

    info!("campaign session_seed = {}", config.envelope.session_seed);
    util_stats::set_enabled(config.envelope.stats);
    crate::simulator::run_cap::reset();
    crate::simulator::stall_cap::reset();
    crate::simulator::arm_selector::reset();
    crate::simulator::fault_timing::reset();
    crate::simulator::ghost_release::reset();
    crate::simulator::timer_context::reset();
    crate::simulator::fault_timing::set_fraction(
        config.envelope.faults.crash_placement_fraction,
    );
    util_stats::set_acted_fraction_enabled(config.envelope.emit_acted_fraction);
    util_stats::set_acceptance_distance_enabled(config.envelope.emit_acceptance_distance);
    util_stats::set_crash_census_enabled(config.envelope.emit_crash_census);
    util_stats::set_quiet_stretch_enabled(config.envelope.quiet_stretch_telemetry);
    util_stats::set_prefix_extension_enabled(config.envelope.emit_prefix_extension);
    util_stats::set_steer_audit_enabled(config.envelope.feedback.steer_audit);
    util_stats::set_steer_audit_always(config.envelope.feedback.steer_audit_always);
    util_stats::set_multiplier_audit_enabled(config.envelope.emit_multiplier_authority);
    util_stats::set_recovery_weight_placebo(config.envelope.faults.recovery_weight_placebo);
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

    let deploy_space = arm_configs.first().and_then(|c| c.deploy_space.clone());
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
        let limit = SliceLimit {
            budget: slice.budget,
            started: slice_start,
            cancelled: cancelled.as_ref(),
        };
        let outcome = arm.arm.run_slice(&ctx, &limit);
        let slice_runs = outcome.runs;
        runs_failed += outcome.failed;
        if outcome.cancelled {
            was_cancelled = true;
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
    let space = deploy_space.as_deref().expect("a campaign has at least one bound arm");
    crate::simulator::explorer::write_deployment_tables(
        output_path,
        program,
        &crate::simulator::explorer::cached_deployments(space),
    );
    let (deploy, deployments_built, deploy_rejections, tuples_aliased) =
        crate::simulator::explorer::deploy_summary(space);

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
            deploy,
            deployments_built,
            deploy_rejections,
            tuples_aliased,
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
        "params": {"n": {"min": 3, "max": 3}},
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

    /// How a synthetic run was assigned, comparable across executions.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Assigned {
        Fresh(usize),
        Child {
            parent: u64,
            prefix: bool,
            config_index: usize,
        },
    }

    fn mix(id: i64) -> u64 {
        let mut x = (id as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    /// A synthetic grid run: a fresh run earns a corpus entry when its id
    /// hashes into one of `admit_one_in` buckets, children never do, and the
    /// run sleeps a hash of its id so completions arrive out of order.
    struct Synthetic {
        admit_one_in: u64,
        sleep: bool,
        log: std::sync::Mutex<Vec<(i64, Assigned)>>,
        finished: std::sync::atomic::AtomicU64,
    }

    impl Synthetic {
        fn new(admit_one_in: u64, sleep: bool) -> Self {
            Self {
                admit_one_in,
                sleep,
                log: std::sync::Mutex::new(Vec::new()),
                finished: std::sync::atomic::AtomicU64::new(0),
            }
        }

        fn run(&self, run_id: i64, run: &GridRun<u64>) -> GridOutcome<u64> {
            if self.sleep {
                std::thread::sleep(std::time::Duration::from_micros(mix(run_id) % 400));
            }
            let (assigned, admit) = match run {
                GridRun::Fresh { config_index } => {
                    let admit = (mix(run_id ^ 0x55) % self.admit_one_in == 0).then(|| Seed {
                        tape: vec![run_id as u64, 7].into(),
                        workload_seed: run_id as u64,
                        cfg: run_id as u64,
                        config_index: *config_index,
                        cut_step: run_id as i32,
                    });
                    (Assigned::Fresh(*config_index), admit)
                }
                GridRun::Child { seed, prefix } => (
                    Assigned::Child {
                        parent: seed.workload_seed,
                        prefix: *prefix,
                        config_index: seed.config_index,
                    },
                    None,
                ),
            };
            self.log.lock().unwrap().push((run_id, assigned));
            self.finished.fetch_add(1, Ordering::Relaxed);
            GridOutcome {
                score: Some(0.0),
                admit,
            }
        }

        fn assignments(&self) -> Vec<(i64, Assigned)> {
            let mut log = self.log.lock().unwrap().clone();
            log.sort_by_key(|(id, _)| *id);
            log
        }
    }

    /// Batches assigned whole, run to completion in position order and
    /// admitted in position order, one at a time.
    fn sequential_reference(
        assigner: &mut Assigner<u64>,
        batch_size: usize,
        run_counter: &AtomicI64,
        mut next_batch: impl FnMut(u64, u64) -> Option<usize>,
        synthetic: &Synthetic,
    ) {
        let (mut issued, mut batches) = (0u64, 0u64);
        while let Some(cap) = next_batch(issued, batches) {
            let count = batch_size.min(cap);
            let batch: Vec<(i64, GridRun<u64>)> = (0..count)
                .map(|_| {
                    let run_id = run_counter.fetch_add(1, Ordering::Relaxed);
                    (run_id, assigner.assign(run_id).0)
                })
                .collect();
            let outcomes: Vec<GridOutcome<u64>> = batch
                .iter()
                .map(|(run_id, run)| synthetic.run(*run_id, run))
                .collect();
            for outcome in outcomes {
                if let Some(seed) = outcome.admit {
                    assigner.admit(seed);
                }
            }
            issued += count as u64;
            batches += 1;
        }
    }

    fn runs_budget(total: u64) -> impl FnMut(u64, u64) -> Option<usize> {
        move |issued, batches| {
            if batches > 0 && issued >= total {
                None
            } else {
                Some(total.saturating_sub(issued) as usize).filter(|&r| r > 0)
            }
        }
    }

    /// Runs `slices` slices of `slice_runs` runs through the pool and through
    /// the sequential reference, and requires identical assignments,
    /// admission sequences and cursors after every slice.
    fn pool_matches_reference(
        admit_one_in: u64,
        batch_size: usize,
        workers: usize,
        slices: usize,
        slice_runs: u64,
    ) -> util_stats::GridPoolSession {
        let grid_len = 7;
        let reference = Synthetic::new(admit_one_in, false);
        let mut ref_assigner = Assigner::<u64>::new(grid_len);
        let ref_counter = AtomicI64::new(0);
        let pooled = Synthetic::new(admit_one_in, true);
        let mut assigner = Assigner::<u64>::new(grid_len);
        let counter = AtomicI64::new(0);
        let mut total = util_stats::GridPoolSession::default();
        for _ in 0..slices {
            sequential_reference(
                &mut ref_assigner,
                batch_size,
                &ref_counter,
                runs_budget(slice_runs),
                &reference,
            );
            let result = run_ordered_release(
                &mut assigner,
                batch_size,
                workers,
                &counter,
                runs_budget(slice_runs),
                &|run_id, run| pooled.run(run_id, run),
            );
            assert_eq!(
                pooled.finished.load(Ordering::Relaxed),
                counter.load(Ordering::Relaxed) as u64,
                "a run was still in flight when the slice returned"
            );
            assert_eq!(result.session.runs, slice_runs);
            assert_eq!(result.failed, 0);
            assert_eq!(assigner.admitted, ref_assigner.admitted, "admission sequence");
            assert_eq!(assigner.cursor, ref_assigner.cursor, "grid cursor");
            assert_eq!(
                assigner.corpus.remaining_children(),
                ref_assigner.corpus.remaining_children()
            );
            assert_eq!(
                counter.load(Ordering::Relaxed),
                ref_counter.load(Ordering::Relaxed)
            );
            total.runs += result.session.runs;
            total.batches += result.session.batches;
            total.capacity_gated_batches += result.session.capacity_gated_batches;
            total.unfilled_in_ungated_batches += result.session.unfilled_in_ungated_batches;
            total.fresh_ahead_launched += result.session.fresh_ahead_launched;
        }
        assert_eq!(
            pooled.assignments(),
            reference.assignments(),
            "assignment by run id"
        );
        assert!(
            !ref_assigner.admitted.is_empty(),
            "the synthetic arm admitted nothing, so the check is vacuous"
        );
        assert_eq!(total.unfilled_in_ungated_batches, 0);
        total
    }

    #[test]
    fn ordered_release_matches_sequential_batches_with_a_full_corpus() {
        let total = pool_matches_reference(2, 16, 6, 3, 437);
        assert!(
            total.fresh_ahead_launched > 0,
            "no fresh run started ahead: {total:?}"
        );
    }

    #[test]
    fn ordered_release_matches_sequential_batches_when_the_corpus_runs_empty() {
        let total = pool_matches_reference(23, 16, 6, 3, 437);
        assert!(
            total.capacity_gated_batches > 0,
            "no batch was gated on an empty corpus: {total:?}"
        );
    }

    #[test]
    fn ordered_release_at_one_worker_starts_runs_in_id_order() {
        let synthetic = Synthetic::new(3, false);
        let mut assigner = Assigner::<u64>::new(5);
        let counter = AtomicI64::new(0);
        let starts = std::sync::Mutex::new(Vec::new());
        run_ordered_release(
            &mut assigner,
            8,
            1,
            &counter,
            runs_budget(203),
            &|run_id, run| {
                starts.lock().unwrap().push(run_id);
                synthetic.run(run_id, run)
            },
        );
        let starts = starts.into_inner().unwrap();
        assert_eq!(starts, (0..203).collect::<Vec<i64>>());
    }

    #[test]
    fn ordered_release_drains_every_run_at_slice_end() {
        let nine_batches = |_: u64, batches: u64| (batches < 9).then_some(usize::MAX);
        let reference = Synthetic::new(2, false);
        let mut ref_assigner = Assigner::<u64>::new(5);
        let ref_counter = AtomicI64::new(0);
        sequential_reference(&mut ref_assigner, 10, &ref_counter, nine_batches, &reference);
        let synthetic = Synthetic::new(2, true);
        let mut assigner = Assigner::<u64>::new(5);
        let counter = AtomicI64::new(0);
        let result = run_ordered_release(
            &mut assigner,
            10,
            8,
            &counter,
            nine_batches,
            &|run_id, run| synthetic.run(run_id, run),
        );
        assert_eq!(result.session.runs, 90);
        assert_eq!(result.session.batches, 9);
        assert_eq!(
            synthetic.finished.load(Ordering::Relaxed),
            90,
            "runs were left in flight"
        );
        assert_eq!(counter.load(Ordering::Relaxed), 90);
        assert_eq!(
            synthetic.assignments(),
            reference.assignments(),
            "assignment by run id"
        );
        assert_eq!(assigner.admitted, ref_assigner.admitted, "admission sequence");
        assert!(
            result.session.job_wall_ns <= result.session.workers * result.session.pool_wall_ns,
            "job wall exceeds worker capacity: {:?}",
            result.session
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(synthetic.finished.load(Ordering::Relaxed), 90);
    }

    #[test]
    fn slice_limit_issues_a_first_batch_and_stops_on_budget_or_cancel() {
        let flag = AtomicBool::new(false);
        let limit = SliceLimit {
            budget: SliceBudget::Runs(100),
            started: Instant::now(),
            cancelled: &flag,
        };
        let mut cancelled = false;
        assert_eq!(limit.next_batch(0, 0, &mut cancelled), Some(100));
        assert_eq!(limit.next_batch(60, 1, &mut cancelled), Some(40));
        assert_eq!(limit.next_batch(100, 2, &mut cancelled), None);
        assert!(!cancelled);
        let timed = SliceLimit {
            budget: SliceBudget::Seconds(0.0),
            started: Instant::now(),
            cancelled: &flag,
        };
        assert_eq!(timed.next_batch(0, 0, &mut cancelled), Some(usize::MAX));
        assert_eq!(timed.next_batch(60, 1, &mut cancelled), None);
        flag.store(true, Ordering::Relaxed);
        assert_eq!(limit.next_batch(0, 0, &mut cancelled), None);
        assert!(cancelled);
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
