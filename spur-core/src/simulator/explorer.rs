use crate::compiler::cfg::Program;
use crate::simulator::core::{
    Env, Logger, NodeId, PurgatoryConfig, QueuePolicyConfig, RuntimeError, SchedulePolicy, State,
    Value, WithinQueueSelector, exec_sync_on_node, make_local_env,
};
use crate::simulator::coverage::GlobalState;
use crate::simulator::curriculum::{Curriculum, lower};
use crate::simulator::feedback::{
    CfgFeedback, CoverageConfig, Feedback, FeedbackConfig, FeedbackMode, FullFeedback, NoFeedback,
    TimelineFeedback, TimelineKeyGranularity, TimelineTuple,
};
use crate::simulator::hash_utils::compute_hash;
use crate::simulator::history::{
    HistoryWriter, LogBackend, create_writer, serialize_history, serialize_logs, serialize_traces,
};
use crate::simulator::path::generator::{GeneratorConfig, generate_plan};
use crate::simulator::path::plan::ExecutionPlan;
use crate::simulator::path::{PathState, RunOutcome, Topology, TopologyInfo, exec_plan};
use crate::simulator::rng::{
    LiveRng, RecRng, RecordRng, Recording, ReplayRng, RngSource, SCHEDULE_SALT, WORKLOAD_SALT,
    derive_seed, mutate_tape,
};
use crate::simulator::util_stats;
use crossbeam::channel;
use log::{debug, error, info, warn};
use rand::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelBridge;
use serde::Deserialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::{fs, thread};

/// Non-generic result of an exploration session. The feedback type parameter is
/// fully contained inside the dispatch, so callers (the CLI) never see it.
pub struct ExploreSummary {
    /// Per-vertex CFG hit counts for the heatmap, present only when the chosen
    /// feedback strategy tracks CFG coverage.
    pub vertex_coverage: Option<HashMap<usize, u64>>,
}

/// Resolves a runtime `FeedbackConfig` to a monomorphized call. Each arm binds
/// the chosen `Feedback` strategy to the caller's type identifier and evaluates
/// `$body`, so the generic type never escapes this match.
macro_rules! dispatch_feedback {
    ($cfg:expr, $f:ident => $body:expr) => {{
        match ($cfg.mode, $cfg.steer) {
            (FeedbackMode::None, _) => {
                type $f = NoFeedback;
                $body
            }
            (FeedbackMode::Cfg, _) => {
                type $f = CfgFeedback;
                $body
            }
            (FeedbackMode::Timeline, false) => {
                type $f = TimelineFeedback<false>;
                $body
            }
            (FeedbackMode::Timeline, true) => {
                type $f = TimelineFeedback<true>;
                $body
            }
            (FeedbackMode::Both, false) => {
                type $f = FullFeedback<false>;
                $body
            }
            (FeedbackMode::Both, true) => {
                type $f = FullFeedback<true>;
                $body
            }
        }
    }};
}

#[derive(Clone, Debug, Deserialize)]
pub struct Range {
    pub min: i32,
    pub max: i32,
    #[serde(default = "default_step")]
    pub step: i32,
}

fn default_step() -> i32 {
    1
}

impl Range {
    pub fn validate(&self) -> Result<(), String> {
        if self.step <= 0 {
            return Err(format!("Invalid range step: {} (must be > 0)", self.step));
        }
        Ok(())
    }

    pub fn expand(&self) -> Vec<i32> {
        assert!(
            self.step > 0,
            "Range must be validated before calling expand()"
        );
        (self.min..=self.max).step_by(self.step as usize).collect()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExplorerConfig {
    #[serde(rename = "num_servers")]
    pub num_servers_range: Range,

    #[serde(rename = "num_write_ops")]
    pub num_write_ops_range: Range,

    #[serde(rename = "num_read_ops")]
    pub num_read_ops_range: Range,

    #[serde(rename = "num_rmw_ops", default = "default_zero_range")]
    pub num_rmw_ops_range: Range,

    #[serde(rename = "num_keys", default = "default_num_keys_range")]
    pub num_keys_range: Range,

    #[serde(rename = "num_crashes")]
    pub num_crashes_range: Range,

    #[serde(rename = "num_partitions", default = "default_partitions_range")]
    pub num_partitions_range: Range,

    #[serde(rename = "max_concurrent_writes", default)]
    pub max_concurrent_writes_range: Option<Range>,

    #[serde(rename = "dependency_density")]
    pub dependency_density_values: Vec<f64>,

    #[serde(default = "default_use_coverage_scheduling")]
    pub use_coverage_scheduling: bool,
    pub num_runs_per_config: i32,
    pub max_iterations: i32,

    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_num_generations")]
    pub num_generations: usize,

    /// Session-wide seed for reproducibility. Per-run workload/schedule seeds
    /// are derived from `(session_seed, run_id)`. Defaults to a fresh random
    /// value per session (logged), so omitting it keeps runs varied.
    #[serde(default = "default_session_seed")]
    pub session_seed: u64,

    #[serde(default)]
    pub schedule_policy: SchedulePolicy,

    #[serde(default)]
    pub queue_policy: QueuePolicyConfig,

    #[serde(default)]
    pub within_queue_selector: WithinQueueSelector,

    #[serde(default = "default_quick_fire_multiplier")]
    pub quick_fire_multiplier: f64,

    #[serde(default)]
    pub purgatory: PurgatoryConfig,

    #[serde(default)]
    pub feedback: FeedbackConfig,

    /// Opt-in utilization counters (see `util_stats`), dumped to
    /// `<output_dir>/utilization.json` at the end of the session.
    /// Observation-only; off by default.
    #[serde(default)]
    pub stats: bool,

    /// Opt-in strict config parsing: when true, a top-level key that no
    /// explorer config field claims is a hard error instead of being silently
    /// ignored. Off by default (today's serde behaviour), so existing configs
    /// keep loading unchanged; harness-generated configs turn it on so a
    /// misspelled or not-yet-implemented knob fails the session immediately
    /// instead of producing a full run that quietly measures the old code.
    #[serde(default)]
    pub strict_config_keys: bool,
}

/// Top-level JSON keys claimed by `ExplorerConfig` (serde names, i.e. after
/// `#[serde(rename)]`). Kept next to the struct: adding a field without
/// listing it here makes strict mode reject configs that use it, which the
/// `strict_config_keys_*` tests below catch.
pub const EXPLORER_CONFIG_KEYS: &[&str] = &[
    "num_servers",
    "num_write_ops",
    "num_read_ops",
    "num_rmw_ops",
    "num_keys",
    "num_crashes",
    "num_partitions",
    "max_concurrent_writes",
    "dependency_density",
    "use_coverage_scheduling",
    "num_runs_per_config",
    "max_iterations",
    "population_size",
    "num_generations",
    "session_seed",
    "schedule_policy",
    "queue_policy",
    "within_queue_selector",
    "quick_fire_multiplier",
    "purgatory",
    "feedback",
    "stats",
    "strict_config_keys",
];

/// Top-level keys added by `ContinuousConfig` on top of the envelope.
pub const CONTINUOUS_CONFIG_KEYS: &[&str] = &[
    "batch_size",
    "decay_half_life_runs",
    "rotation",
    "total_runs",
];

/// Reject top-level keys that no config field claims.
///
/// Only called when `strict_config_keys` is set. A silently ignored key is the
/// worst failure mode a config knob has: the session runs to completion and
/// reports metrics for a mechanism that was never enabled, so the result reads
/// as "mechanism does not help" rather than "mechanism was never on".
pub fn check_top_level_keys(config_json: &str, allowed: &[&[&str]]) -> Result<(), String> {
    let parsed: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| format!("config is not valid JSON: {}", e))?;
    let obj = match parsed.as_object() {
        Some(o) => o,
        None => return Err("config must be a JSON object".to_string()),
    };
    let mut unknown: Vec<&str> = obj
        .keys()
        .map(|k| k.as_str())
        .filter(|k| !allowed.iter().any(|set| set.contains(k)))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    let mut known: Vec<&str> = allowed.iter().flat_map(|set| set.iter().copied()).collect();
    known.sort_unstable();
    Err(format!(
        "unknown top-level config key(s): {} (strict_config_keys is on; known keys: {})",
        unknown.join(", "),
        known.join(", ")
    ))
}

impl ExplorerConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.feedback
            .validate()
            .map_err(|e| format!("feedback config error: {}", e))?;
        self.num_servers_range
            .validate()
            .map_err(|e| format!("num_servers range error: {}", e))?;
        self.num_write_ops_range
            .validate()
            .map_err(|e| format!("num_write_ops range error: {}", e))?;
        self.num_read_ops_range
            .validate()
            .map_err(|e| format!("num_read_ops range error: {}", e))?;
        self.num_rmw_ops_range
            .validate()
            .map_err(|e| format!("num_rmw_ops range error: {}", e))?;
        self.num_keys_range
            .validate()
            .map_err(|e| format!("num_keys range error: {}", e))?;
        if self.num_keys_range.min < 1 {
            return Err(format!(
                "num_keys range error: min must be >= 1 (got {})",
                self.num_keys_range.min
            ));
        }
        self.num_crashes_range
            .validate()
            .map_err(|e| format!("num_crashes range error: {}", e))?;
        self.num_partitions_range
            .validate()
            .map_err(|e| format!("num_partitions range error: {}", e))?;
        if let Some(r) = &self.max_concurrent_writes_range {
            r.validate()
                .map_err(|e| format!("max_concurrent_writes range error: {}", e))?;
            if r.min < 1 {
                return Err(format!(
                    "max_concurrent_writes range error: min must be >= 1 (got {})",
                    r.min
                ));
            }
        }
        Ok(())
    }
}

fn default_partitions_range() -> Range {
    Range {
        min: 0,
        max: 0,
        step: 1,
    }
}

fn default_zero_range() -> Range {
    Range {
        min: 0,
        max: 0,
        step: 1,
    }
}

fn default_num_keys_range() -> Range {
    Range {
        min: 1,
        max: 1,
        step: 1,
    }
}

fn default_population_size() -> usize {
    50
}

fn default_session_seed() -> u64 {
    rand::random()
}
fn default_num_generations() -> usize {
    100
}

fn default_use_coverage_scheduling() -> bool {
    true
}

fn default_quick_fire_multiplier() -> f64 {
    5.0
}

#[derive(Debug, Clone)]
pub struct SingleRunConfig {
    pub num_servers: i32,
    pub num_write_ops: i32,
    pub num_read_ops: i32,
    pub num_rmw_ops: i32,
    pub num_keys: i32,
    pub num_crashes: i32,
    pub num_partitions: i32,
    pub max_concurrent_writes: Option<i32>,
    pub dependency_density: f64,
    pub use_coverage_scheduling: bool,
    pub max_iterations: i32,
    pub schedule_policy: SchedulePolicy,
    pub queue_policy: QueuePolicyConfig,
    pub within_queue_selector: WithinQueueSelector,
    pub quick_fire_multiplier: f64,
    pub purgatory: PurgatoryConfig,
    pub timeline_key_granularity: TimelineKeyGranularity,
}

impl SingleRunConfig {
    pub fn random(constraints: &ExplorerConfig, rng: &mut impl Rng) -> Self {
        let queue_policy = if rng.random::<f64>() < 0.2 {
            let p_timer = if rng.random::<f64>() < 0.8 {
                rng.random_range(0.005..=0.03)
            } else {
                rng.random_range(0.05..=0.3)
            };
            let preempt_interval: i32 = rng.random_range(10..=200);
            QueuePolicyConfig::Preemptive {
                p_timer,
                preempt_interval,
            }
        } else {
            let p_local: f64 = rng.random_range(0.6..=0.95);
            let p_timer = if rng.random::<f64>() < 0.8 {
                rng.random_range(0.005..=0.03)
            } else {
                rng.random_range(0.05..=0.2)
            };
            QueuePolicyConfig::Probabilistic { p_local, p_timer }
        };
        SingleRunConfig {
            num_servers: rng.random_range(
                constraints.num_servers_range.min..=constraints.num_servers_range.max,
            ),
            num_write_ops: rng.random_range(
                constraints.num_write_ops_range.min..=constraints.num_write_ops_range.max,
            ),
            num_read_ops: rng.random_range(
                constraints.num_read_ops_range.min..=constraints.num_read_ops_range.max,
            ),
            num_rmw_ops: rng.random_range(
                constraints.num_rmw_ops_range.min..=constraints.num_rmw_ops_range.max,
            ),
            num_keys: rng
                .random_range(constraints.num_keys_range.min..=constraints.num_keys_range.max),
            num_crashes: rng.random_range(
                constraints.num_crashes_range.min..=constraints.num_crashes_range.max,
            ),
            num_partitions: rng.random_range(
                constraints.num_partitions_range.min..=constraints.num_partitions_range.max,
            ),
            max_concurrent_writes: constraints
                .max_concurrent_writes_range
                .as_ref()
                .map(|r| rng.random_range(r.min..=r.max)),
            dependency_density: *constraints
                .dependency_density_values
                .choose(rng)
                .unwrap_or(&0.5),
            use_coverage_scheduling: constraints.use_coverage_scheduling,
            max_iterations: constraints.max_iterations,
            schedule_policy: constraints.schedule_policy.clone(),
            queue_policy,
            within_queue_selector: constraints.within_queue_selector.clone(),
            quick_fire_multiplier: constraints.quick_fire_multiplier,
            purgatory: constraints.purgatory.clone(),
            timeline_key_granularity: constraints.feedback.timeline_key_granularity,
        }
    }

    pub fn mutate(&self, constraints: &ExplorerConfig, rng: &mut impl Rng) -> Self {
        let mut new_config = self.clone();

        fn mutate_int(rng: &mut impl Rng, val: i32, range: &Range) -> i32 {
            if rng.random_bool(0.3) {
                let delta = if rng.random_bool(0.5) { 1 } else { -1 };
                (val + delta).clamp(range.min, range.max)
            } else {
                val
            }
        }

        new_config.num_servers = mutate_int(rng, self.num_servers, &constraints.num_servers_range);
        new_config.num_write_ops = mutate_int(rng, self.num_write_ops, &constraints.num_write_ops_range);
        new_config.num_read_ops = mutate_int(rng, self.num_read_ops, &constraints.num_read_ops_range);
        new_config.num_rmw_ops = mutate_int(rng, self.num_rmw_ops, &constraints.num_rmw_ops_range);
        new_config.num_keys = mutate_int(rng, self.num_keys, &constraints.num_keys_range);
        new_config.num_crashes = mutate_int(rng, self.num_crashes, &constraints.num_crashes_range);
        new_config.num_partitions =
            mutate_int(rng, self.num_partitions, &constraints.num_partitions_range);

        if let (Some(current), Some(range)) = (
            self.max_concurrent_writes,
            constraints.max_concurrent_writes_range.as_ref(),
        ) {
            new_config.max_concurrent_writes = Some(mutate_int(rng, current, range));
        }

        if rng.random_bool(0.3) && !constraints.dependency_density_values.is_empty() {
            new_config.dependency_density = *constraints
                .dependency_density_values
                .choose(rng)
                .unwrap();
        }

        if rng.random_bool(0.3) {
            match new_config.queue_policy {
                QueuePolicyConfig::Probabilistic {
                    ref mut p_local,
                    ref mut p_timer,
                } => {
                    let delta: f64 = rng.random_range(-0.1..=0.1);
                    *p_local = (*p_local + delta).clamp(0.6, 0.95);
                    let delta: f64 = rng.random_range(-0.1..=0.1);
                    *p_timer = (*p_timer + delta).clamp(0.005, 0.3);
                }
                QueuePolicyConfig::Preemptive {
                    ref mut p_timer,
                    ref mut preempt_interval,
                } => {
                    let delta: f64 = rng.random_range(-0.05..=0.05);
                    *p_timer = (*p_timer + delta).clamp(0.005, 0.3);
                    let delta: i32 = rng.random_range(-20..=20);
                    *preempt_interval = (*preempt_interval + delta).clamp(5, 300);
                }
            }
        }

        new_config
    }
}

fn initialize_state<H: crate::simulator::hash_utils::HashPolicy, L: Logger, F: Feedback>(
    program: &Program,
    logger: &mut L,
    num_servers: usize,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl Rng,
) -> Result<State<H>, RuntimeError> {
    // Look up role NameIds from the program
    let server_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "Node")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("Node".to_string()))?;

    let role_node_counts = vec![(server_role, num_servers)];
    let mut state = State::<H>::new(&role_node_counts, program.max_node_slots as usize);

    if let Some(init_fn) = program.get_func_by_name("Node.BASE_NODE_INIT") {
        for i in 0..num_servers {
            let node_id = NodeId {
                role: server_role,
                index: i,
            };
            let node_env = &state.nodes[i];
            let mut env = make_local_env(
                init_fn,
                vec![],
                &Env::<H>::default(),
                node_env,
                &program.id_to_name,
            );
            exec_sync_on_node::<H, _, F>(
                &mut state,
                logger,
                program,
                &mut env,
                node_id,
                init_fn.entry,
                snapshot,
                feedback,
                &SchedulePolicy::Fixed,
                purgatory_config,
                rng,
            )?;
        }
    }

    Ok(state)
}

fn init_topology<H: crate::simulator::hash_utils::HashPolicy, L: Logger, F: Feedback>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    num_servers: usize,
    snapshot: &F::Snapshot,
    feedback: &mut F::Local,
    purgatory_config: &PurgatoryConfig,
    rng: &mut impl Rng,
) -> Result<(), RuntimeError> {
    let init_fn_name = "Node.Init";
    let Some(init_fn) = program.get_func_by_name(init_fn_name) else {
        warn!("{} not found", init_fn_name);
        return Ok(());
    };

    let server_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "Node")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("Node".to_string()))?;

    let peer_list = Value::<H>::list(
        (0..num_servers)
            .map(|i| {
                Value::<H>::node(NodeId {
                    role: server_role,
                    index: i,
                })
            })
            .collect(),
    );

    for i in 0..num_servers {
        let node_id = NodeId {
            role: server_role,
            index: i,
        };
        let actuals = vec![Value::<H>::int(i as i64), peer_list.clone()];
        let node_env = &state.nodes[i];
        let mut env = make_local_env(
            init_fn,
            actuals,
            &Env::<H>::default(),
            node_env,
            &program.id_to_name,
        );

        exec_sync_on_node::<H, _, F>(
            state,
            logger,
            program,
            &mut env,
            node_id,
            init_fn.entry,
            snapshot,
            feedback,
            &SchedulePolicy::Fixed,
            purgatory_config,
            rng,
        )?;
    }
    Ok(())
}

/// Outcome of a single run: the genetic fitness, the (optional) self-contained
/// schedule recording, and the run's timeline-tuple set (for AOS credit).
pub struct RunResult {
    pub score: f64,
    pub recording: Option<Recording>,
    pub tuples: HashSet<TimelineTuple>,
}

/// Runs a single simulation configuration. `S` selects the RNG strategy
/// (`LiveRng` for plain explore; `RecordRng`/`ReplayRng` for RnR). Workload and
/// schedule seeds are supplied by the caller (derived from `(session_seed,
/// run_id)`); `seed_tape` is the recording to replay, if any.
pub fn run_single_simulation<F: Feedback, S: RngSource>(
    program: &Program,
    writer: &Arc<dyn HistoryWriter>,
    global_state: &GlobalState<F>,
    run_id: i64,
    config: &SingleRunConfig,
    weights: &CoverageConfig,
    workload_seed: u64,
    schedule_seed: u64,
    seed_tape: Option<Recording>,
) -> Result<RunResult, Box<dyn Error>> {
    let snapshot = F::snapshot(&global_state.feedback);
    let gen_config = GeneratorConfig {
        num_servers: config.num_servers,
        num_write_ops: config.num_write_ops,
        num_read_ops: config.num_read_ops,
        num_rmw_ops: config.num_rmw_ops,
        num_keys: config.num_keys,
        num_crashes: config.num_crashes,
        num_partitions: config.num_partitions,
        dependency_density: config.dependency_density,
        max_concurrent_writes: config.max_concurrent_writes,
    };
    // Workload RNG: seeded separately from the scheduling RNG so the plan is
    // reproducible from its own seed and uncorrelated with schedule draws.
    let mut workload_rng = SmallRng::seed_from_u64(workload_seed);
    let plan = generate_plan(gen_config, &mut workload_rng);

    // Use NoHashing for exec_plan mode (no state deduplication needed)
    let num_servers = config.num_servers as usize;

    let server_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "Node")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("Node".to_string()))?;
    let client_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "ClientInterface")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("ClientInterface".to_string()))?;

    let role_node_counts = vec![(server_role, num_servers)];
    let mut path_state = PathState::<crate::simulator::hash_utils::NoHashing, F>::new(
        &role_node_counts,
        program.max_node_slots as usize,
        client_role,
    );
    F::set_key_granularity(&mut path_state.feedback, config.timeline_key_granularity);

    // One scheduling RNG per run, threaded through init + execution so all
    // schedule-shaping draws land on a single stream. `RecRng<S>` records or
    // replays that stream per the chosen strategy (zero-cost for `LiveRng`).
    let mut inner = SmallRng::seed_from_u64(schedule_seed);
    let mut tape = S::new_tape(seed_tape);
    let mut rec = RecRng::<S> {
        tape: &mut tape,
        inner: &mut inner,
    };

    // Initialize state
    path_state.state = initialize_state::<crate::simulator::hash_utils::NoHashing, _, F>(
        program,
        &mut path_state.logs,
        num_servers,
        &snapshot,
        &mut path_state.feedback,
        &config.purgatory,
        &mut rec,
    )?;

    let topology_info = TopologyInfo {
        topology: Topology::Full,
        num_servers: config.num_servers,
    };

    init_topology::<crate::simulator::hash_utils::NoHashing, _, F>(
        &mut path_state.state,
        &mut path_state.logs,
        program,
        num_servers,
        &snapshot,
        &mut path_state.feedback,
        &config.purgatory,
        &mut rec,
    )?;

    let outcome = exec_plan::<crate::simulator::hash_utils::NoHashing, F>(
        &mut path_state,
        program.clone(),
        plan,
        config.max_iterations,
        topology_info,
        global_state,
        &snapshot,
        run_id,
        &config.schedule_policy,
        false,
        &config.queue_policy,
        &config.within_queue_selector,
        config.quick_fire_multiplier,
        &config.purgatory,
        &mut rec,
    )?;

    if let RunOutcome::Deadlock { step, pending_ops } = &outcome {
        warn!(
            "Run {} deadlocked at step {} ({} pending ops)",
            run_id, step, pending_ops
        );
    }

    let plan_score = F::plan_score(&path_state.feedback, &snapshot, weights);
    util_stats::record_plan_score(plan_score);
    let tuples = F::timeline_tuples(&path_state.feedback)
        .cloned()
        .unwrap_or_default();

    let recording = S::into_recording(tape);

    F::merge(&global_state.feedback, &path_state.feedback);

    let serialized = serialize_history(&path_state.history);
    let serialized_logs = serialize_logs(&path_state.logs.entries);
    let serialized_traces = serialize_traces(&path_state.logs.traces);
    writer.write(run_id, serialized, serialized_logs, serialized_traces);

    Ok(RunResult {
        score: plan_score,
        recording,
        tuples,
    })
}

/// Runs the standard exhaustive explorer.
pub fn run_explorer(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    info!("Starting Execution Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;
    if config.strict_config_keys {
        check_top_level_keys(&config_json, &[EXPLORER_CONFIG_KEYS])?;
    }

    // Validate configuration before proceeding
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    info!("session_seed = {}", config.session_seed);
    util_stats::set_enabled(config.stats);
    util_stats::set_audit_enabled(config.feedback.audit_stats);
    dispatch_feedback!(config.feedback, F => run_explorer_impl::<F>(program, config, output_path, backend, cancelled))
}

fn run_explorer_impl<F: Feedback>(
    program: &Program,
    config: ExplorerConfig,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    let weights = config.feedback.weights;
    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);

    let (sender, receiver) = channel::bounded::<(i64, SingleRunConfig)>(100);

    let config_producer = config.clone();
    let cancelled_producer = cancelled.clone();

    thread::spawn(move || {
        let config = config_producer;

        let all_servers = config.num_servers_range.expand();
        let all_writes = config.num_write_ops_range.expand();
        let all_reads = config.num_read_ops_range.expand();
        let all_rmws = config.num_rmw_ops_range.expand();
        let all_keys = config.num_keys_range.expand();
        let all_crashes = config.num_crashes_range.expand();
        let all_partitions = config.num_partitions_range.expand();
        let all_max_concurrent: Vec<Option<i32>> = match &config.max_concurrent_writes_range {
            Some(r) => r.expand().into_iter().map(Some).collect(),
            None => vec![None],
        };
        let all_densities = &config.dependency_density_values;

        let mut config_counter = 0;
        let mut run_counter = 0;
        let total_configs = all_servers.len()
            * all_writes.len()
            * all_reads.len()
            * all_rmws.len()
            * all_keys.len()
            * all_crashes.len()
            * all_partitions.len()
            * all_max_concurrent.len()
            * all_densities.len();

        info!("Total unique configurations: {}", total_configs);
        info!("Runs per config: {}", config.num_runs_per_config);

        'outer: for &num_servers in &all_servers {
            for &num_writes in &all_writes {
                for &num_reads in &all_reads {
                    for &num_rmws in &all_rmws {
                        for &num_keys in &all_keys {
                            for &num_crashes in &all_crashes {
                                for &num_partitions in &all_partitions {
                                    for &max_concurrent in &all_max_concurrent {
                                        for &density in all_densities {
                                            if cancelled_producer.load(Ordering::Relaxed) {
                                                break 'outer;
                                            }
                                            config_counter += 1;

                                            let run_config = SingleRunConfig {
                                                num_servers,
                                                num_write_ops: num_writes,
                                                num_read_ops: num_reads,
                                                num_rmw_ops: num_rmws,
                                                num_keys,
                                                num_crashes,
                                                num_partitions,
                                                max_concurrent_writes: max_concurrent,
                                                dependency_density: density,
                                                use_coverage_scheduling: config
                                                    .use_coverage_scheduling,
                                                max_iterations: config.max_iterations,
                                                schedule_policy: config.schedule_policy.clone(),
                                                queue_policy: config.queue_policy.clone(),
                                                within_queue_selector: config
                                                    .within_queue_selector
                                                    .clone(),
                                                quick_fire_multiplier: config.quick_fire_multiplier,
                                                purgatory: config.purgatory.clone(),
                                                timeline_key_granularity: config
                                                    .feedback
                                                    .timeline_key_granularity,
                                            };

                                            info!("{}", "=".repeat(70));
                                            info!(
                                                "Queuing Config {}/{}: s{}_w{}_r{}_rmw{}_k{}_crash{}_part{}_mcw{}_d{:.2}",
                                                config_counter,
                                                total_configs,
                                                num_servers,
                                                num_writes,
                                                num_reads,
                                                num_rmws,
                                                num_keys,
                                                num_crashes,
                                                num_partitions,
                                                max_concurrent
                                                    .map(|k| k.to_string())
                                                    .unwrap_or_else(|| "-".to_string()),
                                                density
                                            );
                                            info!("{}", "=".repeat(70));

                                            for _ in 1..=config.num_runs_per_config {
                                                run_counter += 1;
                                                if sender
                                                    .send((run_counter, run_config.clone()))
                                                    .is_err()
                                                {
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    info!("Starting parallel simulation...");

    let global_state = Arc::new(GlobalState::<F>::new());

    receiver
        .into_iter()
        .par_bridge()
        .for_each(|(run_id, run_config)| {
            let start = std::time::Instant::now();
            match run_single_simulation::<F, LiveRng>(
                program,
                &writer,
                &global_state,
                run_id,
                &run_config,
                &weights,
                derive_seed(config.session_seed, run_id, WORKLOAD_SALT),
                derive_seed(config.session_seed, run_id, SCHEDULE_SALT),
                None,
            ) {
                Ok(_) => {
                    debug!(
                        "Run {} Success ({:.4}s)",
                        run_id,
                        start.elapsed().as_secs_f64()
                    );
                }
                Err(e) => error!("Run {} failed: {}", run_id, e),
            }
        });

    // Shutdown the writer, waiting for all pending writes to complete
    writer.shutdown();

    info!("Execution explorer finished.");
    Ok(ExploreSummary {
        vertex_coverage: F::vertex_coverage(&global_state.feedback),
    })
}

/// Runs a single simulation with a pre-built execution plan.
#[allow(clippy::too_many_arguments)]
fn run_single_plan<F: Feedback>(
    program: &Program,
    writer: &Arc<dyn HistoryWriter>,
    global_state: &GlobalState<F>,
    run_id: i64,
    plan: &ExecutionPlan,
    num_servers: i32,
    max_iterations: i32,
    policy: &SchedulePolicy,
    strict_timers: bool,
    queue_policy: &QueuePolicyConfig,
    within_queue: &WithinQueueSelector,
    quick_fire_multiplier: f64,
    purgatory_config: &PurgatoryConfig,
    weights: &CoverageConfig,
    key_granularity: TimelineKeyGranularity,
) -> Result<f64, Box<dyn Error>> {
    let snapshot = F::snapshot(&global_state.feedback);
    let num_servers_usize = num_servers as usize;

    let server_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "Node")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("Node".to_string()))?;
    let client_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "ClientInterface")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("ClientInterface".to_string()))?;

    let role_node_counts = vec![(server_role, num_servers_usize)];
    let mut path_state = PathState::<crate::simulator::hash_utils::NoHashing, F>::new(
        &role_node_counts,
        program.max_node_slots as usize,
        client_role,
    );
    F::set_key_granularity(&mut path_state.feedback, key_granularity);

    let mut rng = SmallRng::seed_from_u64(run_id as u64);

    path_state.state = initialize_state::<crate::simulator::hash_utils::NoHashing, _, F>(
        program,
        &mut path_state.logs,
        num_servers_usize,
        &snapshot,
        &mut path_state.feedback,
        purgatory_config,
        &mut rng,
    )?;

    let topology_info = TopologyInfo {
        topology: Topology::Full,
        num_servers,
    };

    init_topology::<crate::simulator::hash_utils::NoHashing, _, F>(
        &mut path_state.state,
        &mut path_state.logs,
        program,
        num_servers_usize,
        &snapshot,
        &mut path_state.feedback,
        purgatory_config,
        &mut rng,
    )?;

    let outcome = exec_plan::<crate::simulator::hash_utils::NoHashing, F>(
        &mut path_state,
        program.clone(),
        plan.clone(),
        max_iterations,
        topology_info,
        global_state,
        &snapshot,
        run_id,
        policy,
        strict_timers,
        queue_policy,
        within_queue,
        quick_fire_multiplier,
        purgatory_config,
        &mut rng,
    )?;

    if let RunOutcome::Deadlock { step, pending_ops } = &outcome {
        warn!(
            "Run {} deadlocked at step {} ({} pending ops)",
            run_id, step, pending_ops
        );
    }

    let plan_score = F::plan_score(&path_state.feedback, &snapshot, weights);
    util_stats::record_plan_score(plan_score);
    F::merge(&global_state.feedback, &path_state.feedback);

    let serialized = serialize_history(&path_state.history);
    let serialized_logs = serialize_logs(&path_state.logs.entries);
    let serialized_traces = serialize_traces(&path_state.logs.traces);
    writer.write(run_id, serialized, serialized_logs, serialized_traces);

    Ok(plan_score)
}

/// Runs a user-specified execution plan `num_runs` times.
pub fn run_plan(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    use crate::simulator::plan_config::PlanFileConfig;

    info!("Starting Plan Runner...");
    info!("Plan config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: PlanFileConfig = serde_json::from_str(&config_json)?;
    config
        .validate()
        .map_err(|e| format!("Plan validation failed: {}", e))?;

    dispatch_feedback!(config.feedback, F => run_plan_impl::<F>(program, config, output_path, backend, cancelled))
}

fn run_plan_impl<F: Feedback>(
    program: &Program,
    config: crate::simulator::plan_config::PlanFileConfig,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    let plan = config
        .to_execution_plan()
        .map_err(|e| format!("Failed to build execution plan: {}", e))?;

    info!(
        "Plan has {} events, {} dependencies",
        plan.node_count(),
        plan.edge_count()
    );
    info!(
        "Running {} times with {} servers",
        config.num_runs, config.num_servers
    );

    let weights = config.feedback.weights;
    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);
    let global_state = Arc::new(GlobalState::<F>::new());

    let runs: Vec<i64> = (1..=config.num_runs as i64).collect();

    runs.par_iter().for_each(|&run_id| {
        if cancelled.load(Ordering::Relaxed) {
            return;
        }
        let start = std::time::Instant::now();
        match run_single_plan::<F>(
            program,
            &writer,
            &global_state,
            run_id,
            &plan,
            config.num_servers,
            config.max_iterations,
            &config.schedule_policy,
            config.strict_timers,
            &config.queue_policy,
            &config.within_queue_selector,
            config.quick_fire_multiplier,
            &config.purgatory,
            &weights,
            config.feedback.timeline_key_granularity,
        ) {
            Ok(_) => {
                debug!(
                    "Run {} Success ({:.4}s)",
                    run_id,
                    start.elapsed().as_secs_f64()
                );
            }
            Err(e) => error!("Run {} failed: {}", run_id, e),
        }
    });

    writer.shutdown();
    info!("Plan runner finished.");
    Ok(ExploreSummary {
        vertex_coverage: F::vertex_coverage(&global_state.feedback),
    })
}

/// Runs the genetic algorithm-based explorer.
/// Returns the coverage summary for the heatmap.
pub fn run_explorer_genetic(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    info!("Starting Genetic Execution Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;
    if config.strict_config_keys {
        check_top_level_keys(&config_json, &[EXPLORER_CONFIG_KEYS])?;
    }

    // Validate configuration before proceeding
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    info!("session_seed = {}", config.session_seed);
    util_stats::set_enabled(config.stats);
    util_stats::set_audit_enabled(config.feedback.audit_stats);
    dispatch_feedback!(config.feedback, F => run_explorer_genetic_impl::<F>(program, config, output_path, backend, cancelled))
}

fn run_explorer_genetic_impl<F: Feedback>(
    program: &Program,
    config: ExplorerConfig,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    let weights = config.feedback.weights;
    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);
    let global_state = Arc::new(GlobalState::<F>::new());
    let run_counter = Arc::new(AtomicI64::new(0));
    let mut ctrl_rng = SmallRng::seed_from_u64(derive_seed(config.session_seed, 0, GENETIC_SALT));

    let mut population: Vec<SingleRunConfig> = (0..config.population_size)
        .map(|_| SingleRunConfig::random(&config, &mut ctrl_rng))
        .collect();

    for generation in 0..config.num_generations {
        if cancelled.load(Ordering::Relaxed) {
            info!(
                "Cancelled by user, stopping after generation {}",
                generation
            );
            break;
        }
        info!(
            "=== Generation {}/{} ===",
            generation + 1,
            config.num_generations
        );

        let base_id = run_counter.fetch_add(population.len() as i64, Ordering::Relaxed);
        let scored: Vec<(SingleRunConfig, f64)> = population
            .par_iter()
            .enumerate()
            .map(|(i, run_config)| {
                let run_id = base_id + i as i64;
                let result = run_single_simulation::<F, LiveRng>(
                    program,
                    &writer,
                    &global_state,
                    run_id,
                    run_config,
                    &weights,
                    derive_seed(config.session_seed, run_id, WORKLOAD_SALT),
                    derive_seed(config.session_seed, run_id, SCHEDULE_SALT),
                    None,
                );
                match result {
                    Ok(r) => (run_config.clone(), r.score),
                    Err(e) => {
                        error!("Genetic run {} failed: {}", run_id, e);
                        (run_config.clone(), 0.0)
                    }
                }
            })
            .collect();

        let mut scored = scored;
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let pop_size = config.population_size;
        let elite_count = pop_size / 10;
        let mutate_count = (pop_size * 6) / 10;
        let random_count = pop_size - elite_count - mutate_count;

        let mut next_gen = Vec::with_capacity(pop_size);

        next_gen.extend(scored.iter().take(elite_count).map(|(c, _)| c.clone()));

        for _ in 0..mutate_count {
            let parent = &scored[ctrl_rng.random_range(0..elite_count.max(1))].0;
            next_gen.push(parent.mutate(&config, &mut ctrl_rng));
        }

        for _ in 0..random_count {
            next_gen.push(SingleRunConfig::random(&config, &mut ctrl_rng));
        }

        population = next_gen;

        info!(
            "Generation {} complete. Best score: {:.4}",
            generation + 1,
            scored.first().map(|(_, s)| *s).unwrap_or(0.0)
        );
    }

    writer.shutdown();

    info!("Genetic explorer finished.");
    Ok(ExploreSummary {
        vertex_coverage: F::vertex_coverage(&global_state.feedback),
    })
}

/// Caps the credit a brand-new scenario can earn just for being new. Tuning knob.
const C_EXPLORE: usize = 25;
/// Minimum selection probability per operator (prevents starvation/domination).
const AOS_P_MIN: f64 = 0.1;
/// Number of tape positions perturbed per TapeMutate.
const AOS_MUTATE_K: usize = 5;
/// Salt for per-child mutation RNG streams (tape and workload jitter).
const MUTATE_SALT: u64 = 0x_4D55_5441_5445_5253; // "MUTATERS"
/// Salt for per-run random-config RNG streams.
const CONFIG_SALT: u64 = 0x_434F_4E46_4947_5253; // "CONFIGRS"
/// Salt for the genetic controller's RNG.
const GENETIC_SALT: u64 = 0x_4745_4E45_5449_4353; // "GENETICS"

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operator {
    TapeMutate = 0,
    ConfigMutate = 1,
}

/// Two-armed adaptive-pursuit bandit. `q` tracks windowed reward per
/// operator; `p` is the selection distribution, pursued toward the
/// current-best arm with a `p_min` floor on both.
struct Bandit {
    q: [f64; 2],
    p: [f64; 2],
    alpha: f64,
    beta: f64,
    p_min: f64,
}

impl Bandit {
    fn new(p_min: f64) -> Self {
        Self {
            q: [0.0; 2],
            p: [0.5; 2],
            alpha: 0.3,
            beta: 0.3,
            p_min,
        }
    }
    fn pick(&self, rng: &mut impl Rng) -> Operator {
        let op = if rng.random::<f64>() < self.p[0] {
            Operator::TapeMutate
        } else {
            Operator::ConfigMutate
        };
        util_stats::record_aos_pick(matches!(op, Operator::TapeMutate));
        op
    }
    fn credit(&mut self, op: Operator, reward: f64) {
        let i = op as usize;
        self.q[i] += self.alpha * (reward - self.q[i]);
    }
    fn recompute(&mut self) {
        let best = if self.q[0] >= self.q[1] { 0 } else { 1 };
        let p_max = 1.0 - self.p_min; // two arms
        for i in 0..2 {
            let target = if i == best { p_max } else { self.p_min };
            self.p[i] += self.beta * (target - self.p[i]);
        }
    }
}

/// A population member: a workload (`cfg` + `workload_seed`) plus a concrete,
/// self-contained schedule recording. `workload_seed` is the scenario key.
#[derive(Clone)]
struct Individual {
    cfg: SingleRunConfig,
    workload_seed: u64,
    tape: Recording,
    score: f64,
    timeline_hash: u64,
}

/// Order-independent hash of a timeline-tuple set (XOR-fold of per-tuple hashes).
fn timeline_set_hash(tuples: &HashSet<TimelineTuple>) -> u64 {
    tuples.iter().fold(0u64, |acc, t| acc ^ compute_hash(t))
}

/// The population doubles as the corpus: deduped on `(workload_seed,
/// timeline_hash)` and capped per scenario so one lucky scenario can't fill it.
struct Population {
    individuals: Vec<Individual>,
    per_scenario: HashMap<u64, usize>,
    seen: HashSet<(u64, u64)>,
    cap_per_scenario: usize,
    max_size: usize,
}

impl Population {
    fn new(cap_per_scenario: usize, max_size: usize) -> Self {
        Self {
            individuals: Vec::new(),
            per_scenario: HashMap::new(),
            seen: HashSet::new(),
            cap_per_scenario,
            max_size,
        }
    }

    /// Insert a member, evicting the weakest when a cap forces it. Returns the
    /// workload_seed of a scenario whose last member was evicted, if any.
    fn insert(&mut self, ind: Individual) -> Option<u64> {
        let key = (ind.workload_seed, ind.timeline_hash);
        let duplicate = self.seen.contains(&key);
        util_stats::record_dedup_check(duplicate);
        if duplicate {
            return None; // redundant timeline within this scenario
        }
        let count = self
            .per_scenario
            .get(&ind.workload_seed)
            .copied()
            .unwrap_or(0);
        if count >= self.cap_per_scenario {
            // Scenario is full: replace its weakest member if this one beats it.
            if let Some((idx, worst_score)) = self
                .individuals
                .iter()
                .enumerate()
                .filter(|(_, x)| x.workload_seed == ind.workload_seed)
                .map(|(i, x)| (i, x.score))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if ind.score > worst_score {
                    let old = &self.individuals[idx];
                    self.seen.remove(&(old.workload_seed, old.timeline_hash));
                    self.seen.insert(key);
                    self.individuals[idx] = ind;
                }
            }
            return None;
        }
        // Global cap: evict the globally weakest if full.
        let mut drained = None;
        if self.individuals.len() >= self.max_size {
            if let Some((idx, worst_score)) = self
                .individuals
                .iter()
                .enumerate()
                .map(|(i, x)| (i, x.score))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if ind.score <= worst_score {
                    return None;
                }
                let old = self.individuals.swap_remove(idx);
                self.seen.remove(&(old.workload_seed, old.timeline_hash));
                if let Some(c) = self.per_scenario.get_mut(&old.workload_seed) {
                    *c -= 1;
                    if *c == 0 {
                        self.per_scenario.remove(&old.workload_seed);
                        drained = Some(old.workload_seed);
                    }
                }
            }
        }
        self.seen.insert(key);
        *self.per_scenario.entry(ind.workload_seed).or_insert(0) += 1;
        self.individuals.push(ind);
        if drained == Some(key.0) {
            drained = None;
        }
        drained
    }

    /// Whether any member of this scenario is still in the population.
    fn contains_scenario(&self, workload_seed: u64) -> bool {
        self.per_scenario.contains_key(&workload_seed)
    }

    /// Score-weighted parent pick.
    fn select_parent(&self, rng: &mut impl Rng) -> Option<&Individual> {
        self.individuals
            .choose_weighted(rng, |x| x.score + 1e-6)
            .ok()
    }
}

/// Outcome of one controller run, before bandit credit is assigned.
struct AosChild {
    op: Operator,
    workload_seed: u64,
    individual: Option<Individual>,
    tuples: HashSet<TimelineTuple>,
}

/// Package a finished run as an `AosChild` for crediting.
fn package_child(
    op: Operator,
    cfg: SingleRunConfig,
    workload_seed: u64,
    run_id: i64,
    result: Result<RunResult, Box<dyn Error>>,
) -> AosChild {
    match result {
        Ok(r) => {
            let individual = r.recording.map(|tape| Individual {
                cfg,
                workload_seed,
                tape,
                score: r.score,
                timeline_hash: timeline_set_hash(&r.tuples),
            });
            AosChild {
                op,
                workload_seed,
                individual,
                tuples: r.tuples,
            }
        }
        Err(e) => {
            error!("Run {} failed: {}", run_id, e);
            AosChild {
                op,
                workload_seed,
                individual: None,
                tuples: HashSet::new(),
            }
        }
    }
}

/// Record one fresh scenario from `cfg`, with seeds derived from `run_id`.
fn run_recorded<F: Feedback>(
    program: &Program,
    writer: &Arc<dyn HistoryWriter>,
    global_state: &GlobalState<F>,
    weights: &CoverageConfig,
    cfg: SingleRunConfig,
    session_seed: u64,
    run_id: i64,
) -> AosChild {
    let workload_seed = derive_seed(session_seed, run_id, WORKLOAD_SALT);
    let r = run_single_simulation::<F, RecordRng>(
        program,
        writer,
        global_state,
        run_id,
        &cfg,
        weights,
        workload_seed,
        derive_seed(session_seed, run_id, SCHEDULE_SALT),
        None,
    );
    package_child(Operator::ConfigMutate, cfg, workload_seed, run_id, r)
}

/// Execute one AOS run for the chosen operator and parent.
#[allow(clippy::too_many_arguments)]
fn run_aos_child<F: Feedback>(
    program: &Program,
    writer: &Arc<dyn HistoryWriter>,
    global_state: &GlobalState<F>,
    weights: &CoverageConfig,
    config: &ExplorerConfig,
    run_id: i64,
    op: Operator,
    parent: &Individual,
) -> AosChild {
    let session = config.session_seed;
    match op {
        Operator::TapeMutate => {
            // Same scenario: perturb the parent's tape and replay it.
            let mut mrng = SmallRng::seed_from_u64(derive_seed(session, run_id, MUTATE_SALT));
            let mutated = mutate_tape(&parent.tape, AOS_MUTATE_K, &mut mrng);
            let r = run_single_simulation::<F, ReplayRng>(
                program,
                writer,
                global_state,
                run_id,
                &parent.cfg,
                weights,
                parent.workload_seed,
                derive_seed(session, run_id, SCHEDULE_SALT),
                Some(mutated),
            );
            package_child(op, parent.cfg.clone(), parent.workload_seed, run_id, r)
        }
        Operator::ConfigMutate => {
            // New scenario: jitter the workload and record fresh.
            let mut mrng = SmallRng::seed_from_u64(derive_seed(session, run_id, MUTATE_SALT));
            let cfg = parent.cfg.mutate(config, &mut mrng);
            run_recorded::<F>(program, writer, global_state, weights, cfg, session, run_id)
        }
    }
}

/// Runs the adaptive-operator-selection (AOS) record-and-replay controller.
pub fn run_explorer_aos(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    info!("Starting AOS (record-and-replay) controller...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;
    if config.strict_config_keys {
        check_top_level_keys(&config_json, &[EXPLORER_CONFIG_KEYS])?;
    }
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    if !matches!(
        config.feedback.mode,
        FeedbackMode::Timeline | FeedbackMode::Both
    ) {
        return Err(
            "AOS controller requires feedback.mode = \"timeline\" or \"both\" \
                    (the credit signal is timeline novelty)"
                .into(),
        );
    }

    info!("AOS session_seed = {}", config.session_seed);
    util_stats::set_enabled(config.stats);
    util_stats::set_audit_enabled(config.feedback.audit_stats);
    dispatch_feedback!(config.feedback, F => run_explorer_aos_impl::<F>(program, config, output_path, backend, cancelled))
}

/// Credit a finished AOS child against its scenario's accumulating tuple set,
/// then insert any produced individual into the population.
fn aos_credit_and_insert(
    child: AosChild,
    scenario_tuples: &mut HashMap<u64, HashSet<TimelineTuple>>,
    bandit: &mut Bandit,
    population: &mut Population,
) {
    let entry = scenario_tuples.entry(child.workload_seed).or_default();
    let is_new_scenario = entry.is_empty();
    let within_novel = child.tuples.iter().filter(|t| !entry.contains(t)).count();
    let credit = if is_new_scenario {
        within_novel.min(C_EXPLORE)
    } else {
        within_novel
    } as f64;
    entry.extend(child.tuples.iter().copied());
    bandit.credit(child.op, credit);
    if let Some(ind) = child.individual {
        if let Some(drained) = population.insert(ind) {
            scenario_tuples.remove(&drained);
        }
    }
    // Tuples are only useful while the scenario can still be picked as a parent.
    if !population.contains_scenario(child.workload_seed) {
        scenario_tuples.remove(&child.workload_seed);
    }
}

fn run_explorer_aos_impl<F: Feedback>(
    program: &Program,
    config: ExplorerConfig,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    let weights = config.feedback.weights;
    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);
    let run_counter = AtomicI64::new(0);
    let session_seed = config.session_seed;

    let batch_size = config.population_size.max(1);
    let num_batches = config.num_generations.max(1);
    let mut aos = AosExplorer::<F>::new(config, batch_size, weights, session_seed);

    let ctx = StepCtx {
        program,
        writer: &writer,
        run_counter: &run_counter,
        weights: &weights,
        session_seed,
    };

    for batch in 0..num_batches {
        if cancelled.load(Ordering::Relaxed) {
            info!("Cancelled by user, stopping after batch {}", batch);
            break;
        }
        let report = aos.step(&ctx);
        info!(
            "AOS batch {}/{}: p[tape/config]={:.2}/{:.2} q={:.1}/{:.1} | pop={} scenarios={} best={:.4}",
            batch + 1,
            num_batches,
            aos.bandit.p[0],
            aos.bandit.p[1],
            aos.bandit.q[0],
            aos.bandit.q[1],
            aos.population.individuals.len(),
            aos.scenario_tuples.len(),
            report.best_score,
        );
    }

    writer.shutdown();
    info!("AOS controller finished.");
    Ok(ExploreSummary {
        vertex_coverage: F::vertex_coverage(&aos.global_state.feedback),
    })
}

/// Salt for per-mode curriculum RNG streams.
const CURRICULUM_SALT: u64 = 0x_4355_5252_4943_554C; // "CURRICUL"
/// Salt for Mode B's curriculum/control RNG streams (distinct from Mode A).
const CURRICULUM_RNR_SALT: u64 = 0x_4355_5252_524E_5200; // "CURRRNR\0"

/// Consecutive non-improving batches that saturate the stagnation signal to 1.0.
const STAGNATION_PATIENCE: f64 = 5.0;

fn default_batch_size() -> usize {
    32
}
fn default_decay_half_life() -> u64 {
    2000
}
fn default_total_runs() -> u64 {
    100_000
}
fn default_rotation() -> Vec<RotationSlice> {
    vec![RotationSlice {
        mode: ModeId::Curriculum,
        budget: Budget::Runs { runs: 256 },
    }]
}

/// Which exploration strategy a rotation slice runs.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModeId {
    /// Curriculum over fresh samples (Mode A).
    Curriculum,
    /// Curriculum-seeded record-and-replay refinement (Mode B).
    CurriculumRnr,
    /// Adaptive operator selection bandit (Mode C).
    Aos,
}

/// A budget for one rotation slice. `Seconds` sacrifices reproducibility.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Budget {
    Runs { runs: u64 },
    Seconds { seconds: f64 },
}

/// One entry in the manual rotation: a mode and how long to dwell on it.
#[derive(Clone, Debug, Deserialize)]
pub struct RotationSlice {
    pub mode: ModeId,
    pub budget: Budget,
}

/// Continuous-explorer config: the `ExplorerConfig` range envelope plus the
/// continuous-session fields. Reproducible under `Runs` budgets from
/// `session_seed`.
#[derive(Clone, Debug, Deserialize)]
pub struct ContinuousConfig {
    #[serde(flatten)]
    pub envelope: ExplorerConfig,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_decay_half_life")]
    pub decay_half_life_runs: u64,
    #[serde(default = "default_rotation")]
    pub rotation: Vec<RotationSlice>,
    /// Session-wide run cap (also stoppable via Ctrl+C).
    #[serde(default = "default_total_runs")]
    pub total_runs: u64,
}

impl ContinuousConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.envelope.validate()?;
        if self.batch_size == 0 {
            return Err("continuous: batch_size must be >= 1".into());
        }
        if self.rotation.is_empty() {
            return Err("continuous: rotation must be non-empty".into());
        }
        let needs_timeline = self
            .rotation
            .iter()
            .any(|s| matches!(s.mode, ModeId::Aos | ModeId::CurriculumRnr));
        if needs_timeline
            && !matches!(
                self.envelope.feedback.mode,
                FeedbackMode::Timeline | FeedbackMode::Both
            )
        {
            return Err("continuous: aos / curriculum_rnr modes require \
                        feedback.mode = \"timeline\" or \"both\""
                .into());
        }
        Ok(())
    }
}

/// Outcome of one strategy step (one internally-parallel batch).
pub struct StepReport {
    pub runs: u64,
    pub best_score: f64,
}

/// Shared, mode-independent context threaded into every `Strategy::step`.
pub struct StepCtx<'a> {
    pub program: &'a Program,
    pub writer: &'a Arc<dyn HistoryWriter>,
    pub run_counter: &'a AtomicI64,
    pub weights: &'a CoverageConfig,
    pub session_seed: u64,
}

/// A resumable exploration strategy: each `step` runs one internally-parallel
/// batch and reports how many runs it did plus the best score, so the conductor
/// can rotate across modes while each keeps its own persistent state.
trait Strategy<F: Feedback> {
    fn step(&mut self, ctx: &StepCtx) -> StepReport;

    /// Per-vertex CFG hit counts from this mode's feedback store, if tracked.
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>>;
}

/// Mode A: curriculum over fresh `LiveRng` samples. Owns an isolated feedback
/// store and a curriculum clock.
struct CurriculumExplorer<F: Feedback> {
    envelope: ExplorerConfig,
    global_state: GlobalState<F>,
    curriculum: Curriculum,
    batch_size: usize,
    lower_rng: SmallRng,
    /// Per-batch decay factor for the isolated store (from the half-life).
    decay_factor: f64,
    /// Best score seen so far, and consecutive non-improving batches; together
    /// they drive the (optional) stagnation kick.
    best_ever: f64,
    stale_batches: u32,
}

impl<F: Feedback> CurriculumExplorer<F> {
    fn new(
        envelope: ExplorerConfig,
        batch_size: usize,
        total_runs: u64,
        half_life_runs: u64,
        seed: u64,
    ) -> Self {
        // Per-batch factor f such that f^(half_life/batch) = 1/2.
        let decay_factor = if half_life_runs == 0 {
            1.0
        } else {
            0.5_f64.powf(batch_size as f64 / half_life_runs as f64)
        };
        Self {
            curriculum: Curriculum::new(derive_seed(seed, 0, CURRICULUM_SALT), total_runs.max(1)),
            global_state: GlobalState::new(),
            envelope,
            batch_size,
            lower_rng: SmallRng::seed_from_u64(derive_seed(seed, 1, CURRICULUM_SALT)),
            decay_factor,
            best_ever: 0.0,
            stale_batches: 0,
        }
    }
}

impl<F: Feedback> Strategy<F> for CurriculumExplorer<F> {
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>> {
        F::vertex_coverage(&self.global_state.feedback)
    }

    fn step(&mut self, ctx: &StepCtx) -> StepReport {
        // Sequentially mint a batch of configs from the current curriculum
        // state (mutates the curriculum + lowering RNG)...
        let batch: Vec<(i64, SingleRunConfig)> = (0..self.batch_size)
            .map(|_| {
                let knobs = self.curriculum.sample();
                let cfg = lower(&knobs, &self.envelope, &mut self.lower_rng);
                let run_id = ctx.run_counter.fetch_add(1, Ordering::Relaxed);
                (run_id, cfg)
            })
            .collect();

        // ...then execute the batch in parallel against the isolated store.
        let global_state = &self.global_state;
        let scores: Vec<f64> = batch
            .par_iter()
            .map(|(run_id, cfg)| {
                match run_single_simulation::<F, LiveRng>(
                    ctx.program,
                    ctx.writer,
                    global_state,
                    *run_id,
                    cfg,
                    ctx.weights,
                    derive_seed(ctx.session_seed, *run_id, WORKLOAD_SALT),
                    derive_seed(ctx.session_seed, *run_id, SCHEDULE_SALT),
                    None,
                ) {
                    Ok(r) => r.score,
                    Err(e) => {
                        error!("Curriculum run {} failed: {}", run_id, e);
                        0.0
                    }
                }
            })
            .collect();

        self.curriculum.advance(self.batch_size as u64);
        let best = scores.iter().copied().fold(0.0_f64, f64::max);

        // Consecutive batches without a new best feed the curriculum's
        // stagnation kick.
        if best > self.best_ever + 1e-9 {
            self.best_ever = best;
            self.stale_batches = 0;
        } else {
            self.stale_batches = self.stale_batches.saturating_add(1);
        }
        self.curriculum
            .set_stagnation((self.stale_batches as f64 / STAGNATION_PATIENCE).min(1.0));

        // Decay the isolated store between batches: no run is in flight here,
        // so this cannot race the per-run merges.
        if self.decay_factor < 1.0 {
            F::decay(&self.global_state.feedback, self.decay_factor);
        }

        StepReport {
            runs: self.batch_size as u64,
            best_score: best,
        }
    }
}

/// The adaptive-operator-selection bandit (standalone `-e aos`, and Mode C of
/// the continuous explorer). Requires timeline feedback (enforced by both
/// entry points).
struct AosExplorer<F: Feedback> {
    envelope: ExplorerConfig,
    global_state: GlobalState<F>,
    weights: CoverageConfig,
    batch_size: usize,
    ctrl_rng: SmallRng,
    bandit: Bandit,
    scenario_tuples: HashMap<u64, HashSet<TimelineTuple>>,
    population: Population,
    seeded: bool,
}

impl<F: Feedback> AosExplorer<F> {
    fn new(
        envelope: ExplorerConfig,
        batch_size: usize,
        weights: CoverageConfig,
        seed: u64,
    ) -> Self {
        Self {
            ctrl_rng: SmallRng::seed_from_u64(seed),
            bandit: Bandit::new(AOS_P_MIN),
            scenario_tuples: HashMap::new(),
            population: Population::new(8, (batch_size * 8).max(64)),
            seeded: false,
            envelope,
            batch_size,
            weights,
            global_state: GlobalState::new(),
        }
    }
}

impl<F: Feedback> Strategy<F> for AosExplorer<F> {
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>> {
        F::vertex_coverage(&self.global_state.feedback)
    }

    fn step(&mut self, ctx: &StepCtx) -> StepReport {
        // Seed (or re-seed if the corpus drained) with fresh random scenarios.
        if !self.seeded || self.population.individuals.is_empty() {
            let ids: Vec<i64> = (0..self.batch_size)
                .map(|_| ctx.run_counter.fetch_add(1, Ordering::Relaxed))
                .collect();
            let seeds: Vec<AosChild> = ids
                .par_iter()
                .map(|&run_id| {
                    let mut cfg_rng = SmallRng::seed_from_u64(derive_seed(
                        ctx.session_seed,
                        run_id,
                        CONFIG_SALT,
                    ));
                    let cfg = SingleRunConfig::random(&self.envelope, &mut cfg_rng);
                    run_recorded::<F>(
                        ctx.program,
                        ctx.writer,
                        &self.global_state,
                        &self.weights,
                        cfg,
                        ctx.session_seed,
                        run_id,
                    )
                })
                .collect();
            for child in seeds {
                aos_credit_and_insert(
                    child,
                    &mut self.scenario_tuples,
                    &mut self.bandit,
                    &mut self.population,
                );
            }
            self.bandit.recompute();
            self.seeded = true;
        } else {
            // Sequentially pick (run_id, operator, parent) so run ids map to
            // work deterministically, then execute the batch in parallel.
            let picks: Vec<(i64, Operator, Individual)> = (0..self.batch_size)
                .map(|_| {
                    let run_id = ctx.run_counter.fetch_add(1, Ordering::Relaxed);
                    let op = self.bandit.pick(&mut self.ctrl_rng);
                    let parent = self
                        .population
                        .select_parent(&mut self.ctrl_rng)
                        .unwrap()
                        .clone();
                    (run_id, op, parent)
                })
                .collect();
            let children: Vec<AosChild> = picks
                .par_iter()
                .map(|(run_id, op, parent)| {
                    run_aos_child::<F>(
                        ctx.program,
                        ctx.writer,
                        &self.global_state,
                        &self.weights,
                        &self.envelope,
                        *run_id,
                        *op,
                        parent,
                    )
                })
                .collect();
            for child in children {
                aos_credit_and_insert(
                    child,
                    &mut self.scenario_tuples,
                    &mut self.bandit,
                    &mut self.population,
                );
            }
            self.bandit.recompute();
        }

        let best = self
            .population
            .individuals
            .iter()
            .map(|x| x.score)
            .fold(0.0_f64, f64::max);
        StepReport {
            runs: self.batch_size as u64,
            best_score: best,
        }
    }
}

/// One unit of Mode B work: a fresh curriculum-lowered scenario to record, or a
/// corpus parent to refine by tape mutation.
enum SeedOrRefine {
    Seed(SingleRunConfig),
    Refine(Individual),
}

/// Mode B: curriculum-seeded record-and-replay. The curriculum picks the regime
/// only when minting new seed scenarios; each refinement reuses a corpus
/// member's frozen `cfg` + `workload_seed` and only mutates its schedule tape.
/// No bandit: seeds and refinements run on a fixed ratio.
struct CurriculumRnrExplorer<F: Feedback> {
    envelope: ExplorerConfig,
    global_state: GlobalState<F>,
    weights: CoverageConfig,
    curriculum: Curriculum,
    population: Population,
    batch_size: usize,
    lower_rng: SmallRng,
    ctrl_rng: SmallRng,
}

impl<F: Feedback> CurriculumRnrExplorer<F> {
    fn new(
        envelope: ExplorerConfig,
        batch_size: usize,
        weights: CoverageConfig,
        total_runs: u64,
        seed: u64,
    ) -> Self {
        Self {
            curriculum: Curriculum::new(
                derive_seed(seed, 0, CURRICULUM_RNR_SALT),
                total_runs.max(1),
            ),
            population: Population::new(8, (batch_size * 8).max(64)),
            lower_rng: SmallRng::seed_from_u64(derive_seed(seed, 1, CURRICULUM_RNR_SALT)),
            ctrl_rng: SmallRng::seed_from_u64(derive_seed(seed, 2, CURRICULUM_RNR_SALT)),
            global_state: GlobalState::new(),
            envelope,
            batch_size,
            weights,
        }
    }
}

impl<F: Feedback> Strategy<F> for CurriculumRnrExplorer<F> {
    fn vertex_coverage(&self) -> Option<HashMap<usize, u64>> {
        F::vertex_coverage(&self.global_state.feedback)
    }

    fn step(&mut self, ctx: &StepCtx) -> StepReport {
        // Mostly refine, periodically seed. Seed the whole batch while the
        // corpus is empty so there is always something to refine next time.
        let seeds = if self.population.individuals.is_empty() {
            self.batch_size
        } else {
            (self.batch_size / 4).max(1)
        };

        // Sequentially mint the batch: curriculum-lowered seeds + corpus parents.
        let mut work: Vec<(i64, SeedOrRefine)> = Vec::with_capacity(self.batch_size);
        for _ in 0..seeds {
            let knobs = self.curriculum.sample();
            let cfg = lower(&knobs, &self.envelope, &mut self.lower_rng);
            let run_id = ctx.run_counter.fetch_add(1, Ordering::Relaxed);
            work.push((run_id, SeedOrRefine::Seed(cfg)));
        }
        for _ in seeds..self.batch_size {
            if let Some(parent) = self.population.select_parent(&mut self.ctrl_rng) {
                let parent = parent.clone();
                let run_id = ctx.run_counter.fetch_add(1, Ordering::Relaxed);
                work.push((run_id, SeedOrRefine::Refine(parent)));
            }
        }

        // Execute in parallel against the isolated store.
        let produced: Vec<Option<Individual>> = work
            .par_iter()
            .map(|(run_id, w)| match w {
                SeedOrRefine::Seed(cfg) => run_recorded::<F>(
                    ctx.program,
                    ctx.writer,
                    &self.global_state,
                    &self.weights,
                    cfg.clone(),
                    ctx.session_seed,
                    *run_id,
                )
                .individual,
                SeedOrRefine::Refine(parent) => {
                    run_aos_child::<F>(
                        ctx.program,
                        ctx.writer,
                        &self.global_state,
                        &self.weights,
                        &self.envelope,
                        *run_id,
                        Operator::TapeMutate,
                        parent,
                    )
                    .individual
                }
            })
            .collect();

        let runs = work.len() as u64;
        for ind in produced.into_iter().flatten() {
            let _ = self.population.insert(ind);
        }
        // The curriculum clock only advances on the seed (new-scenario) runs.
        self.curriculum.advance(seeds as u64);

        let best = self
            .population
            .individuals
            .iter()
            .map(|x| x.score)
            .fold(0.0_f64, f64::max);
        StepReport {
            runs,
            best_score: best,
        }
    }
}

/// Construct a mode instance for the rotation. State persists across slices.
fn build_mode<F: Feedback>(mode: ModeId, config: &ContinuousConfig) -> Box<dyn Strategy<F>> {
    match mode {
        ModeId::Curriculum => Box::new(CurriculumExplorer::<F>::new(
            config.envelope.clone(),
            config.batch_size,
            config.total_runs,
            config.decay_half_life_runs,
            config.envelope.session_seed,
        )),
        ModeId::CurriculumRnr => Box::new(CurriculumRnrExplorer::<F>::new(
            config.envelope.clone(),
            config.batch_size,
            config.envelope.feedback.weights,
            config.total_runs,
            config.envelope.session_seed,
        )),
        ModeId::Aos => Box::new(AosExplorer::<F>::new(
            config.envelope.clone(),
            config.batch_size,
            config.envelope.feedback.weights,
            config.envelope.session_seed,
        )),
    }
}

/// Runs the continuous adaptive explorer.
pub fn run_explorer_continuous(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    info!("Starting Continuous Adaptive Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ContinuousConfig = serde_json::from_str(&config_json)?;
    if config.envelope.strict_config_keys {
        check_top_level_keys(&config_json, &[EXPLORER_CONFIG_KEYS, CONTINUOUS_CONFIG_KEYS])?;
    }
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    info!("Continuous session_seed = {}", config.envelope.session_seed);
    util_stats::set_enabled(config.envelope.stats);
    util_stats::set_audit_enabled(config.envelope.feedback.audit_stats);
    dispatch_feedback!(config.envelope.feedback, F => run_explorer_continuous_impl::<F>(program, config, output_path, backend, cancelled))
}

fn run_explorer_continuous_impl<F: Feedback>(
    program: &Program,
    config: ContinuousConfig,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<ExploreSummary, Box<dyn Error>> {
    let weights = config.envelope.feedback.weights;
    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);
    let run_counter = AtomicI64::new(0);
    let session_seed = config.envelope.session_seed;

    // Construct each referenced mode once; its state persists across slices.
    let mut modes: HashMap<ModeId, Box<dyn Strategy<F>>> = HashMap::new();
    for slice in &config.rotation {
        if !modes.contains_key(&slice.mode) {
            modes.insert(slice.mode, build_mode::<F>(slice.mode, &config));
        }
    }

    let ctx = StepCtx {
        program,
        writer: &writer,
        run_counter: &run_counter,
        weights: &weights,
        session_seed,
    };

    info!(
        "Continuous explorer: {} mode(s), total_runs cap {}",
        modes.len(),
        config.total_runs
    );

    let mut total_runs: u64 = 0;
    'session: loop {
        for slice in &config.rotation {
            if cancelled.load(Ordering::Relaxed) {
                info!("Cancelled by user.");
                break 'session;
            }
            let mode = modes.get_mut(&slice.mode).expect("mode built above");
            let mut slice_runs: u64 = 0;
            let slice_start = std::time::Instant::now();
            loop {
                if cancelled.load(Ordering::Relaxed) {
                    break 'session;
                }
                if total_runs >= config.total_runs {
                    info!("Reached total_runs cap ({}).", config.total_runs);
                    break 'session;
                }
                let report = mode.step(&ctx);
                slice_runs += report.runs;
                total_runs += report.runs;
                debug!(
                    "[{:?}] +{} runs (slice {}, total {}) best={:.4}",
                    slice.mode, report.runs, slice_runs, total_runs, report.best_score
                );
                let spent = match slice.budget {
                    Budget::Runs { runs } => slice_runs >= runs,
                    Budget::Seconds { seconds } => slice_start.elapsed().as_secs_f64() >= seconds,
                };
                if spent {
                    break;
                }
            }
            info!(
                "[{:?}] slice complete: {} runs (total {})",
                slice.mode, slice_runs, total_runs
            );
        }
    }

    // Merge per-mode CFG coverage for the heatmap.
    let mut vertex_coverage: Option<HashMap<usize, u64>> = None;
    for mode in modes.values() {
        if let Some(cov) = mode.vertex_coverage() {
            let merged = vertex_coverage.get_or_insert_with(HashMap::new);
            for (v, c) in cov {
                *merged.entry(v).or_insert(0) += c;
            }
        }
    }

    writer.shutdown();
    info!("Continuous explorer finished after {} runs.", total_runs);
    Ok(ExploreSummary { vertex_coverage })
}

#[cfg(test)]
mod strict_config_keys_tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "num_servers": {"min": 3, "max": 3},
        "num_write_ops": {"min": 2, "max": 2},
        "num_read_ops": {"min": 2, "max": 2},
        "num_crashes": {"min": 0, "max": 0},
        "dependency_density": [0.0],
        "num_runs_per_config": 1,
        "max_iterations": 100
    }"#;

    /// Every key an existing config may legitimately carry is accepted.
    #[test]
    fn strict_config_keys_accepts_known_keys() {
        assert!(check_top_level_keys(MINIMAL, &[EXPLORER_CONFIG_KEYS]).is_ok());
        let all_known = format!(
            "{{{}}}",
            EXPLORER_CONFIG_KEYS
                .iter()
                .chain(CONTINUOUS_CONFIG_KEYS.iter())
                .map(|k| format!("\"{}\": null", k))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(
            check_top_level_keys(&all_known, &[EXPLORER_CONFIG_KEYS, CONTINUOUS_CONFIG_KEYS])
                .is_ok()
        );
    }

    /// A key no field claims — a typo, or a knob a hypothesis forgot to
    /// implement — is reported by name instead of silently ignored.
    #[test]
    fn strict_config_keys_rejects_unknown_key() {
        let cfg = MINIMAL.replace(
            "\"max_iterations\": 100",
            "\"max_iterations\": 100, \"randomly_delay_msgs\": true",
        );
        let err = check_top_level_keys(&cfg, &[EXPLORER_CONFIG_KEYS]).unwrap_err();
        assert!(err.contains("randomly_delay_msgs"), "{}", err);
        // Continuous-only keys are unknown to a non-continuous session.
        let cont = MINIMAL.replace(
            "\"max_iterations\": 100",
            "\"max_iterations\": 100, \"total_runs\": 10",
        );
        assert!(check_top_level_keys(&cont, &[EXPLORER_CONFIG_KEYS]).is_err());
        assert!(
            check_top_level_keys(&cont, &[EXPLORER_CONFIG_KEYS, CONTINUOUS_CONFIG_KEYS]).is_ok()
        );
    }

    /// Default is today's behaviour: unknown keys are ignored unless the
    /// config opts in.
    #[test]
    fn strict_config_keys_defaults_off() {
        let cfg = MINIMAL.replace(
            "\"max_iterations\": 100",
            "\"max_iterations\": 100, \"nonsense_knob\": 3",
        );
        let parsed: ExplorerConfig = serde_json::from_str(&cfg).expect("parses");
        assert!(!parsed.strict_config_keys);
        let strict: ExplorerConfig = serde_json::from_str(
            &cfg.replace("\"nonsense_knob\": 3", "\"strict_config_keys\": true"),
        )
        .expect("parses");
        assert!(strict.strict_config_keys);
    }
}
