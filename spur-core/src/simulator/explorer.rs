use crate::compiler::cfg::Program;
use crate::simulator::core::{
    Env, Logger, NodeId, RuntimeError, SchedulePolicy, State, Value, exec_sync_on_node,
    make_local_env,
};
use crate::simulator::coverage::{GlobalState, LocalCoverage, VertexMap};
use crate::simulator::history::{
    HistoryWriter, LogBackend, create_writer, serialize_history, serialize_logs, serialize_traces,
};
use crate::simulator::path::generator::{GeneratorConfig, generate_plan};
use crate::simulator::path::plan::ExecutionPlan;
use crate::simulator::path::{PathState, Topology, TopologyInfo, exec_plan};
use crossbeam::channel;
use log::{debug, error, info, warn};
use nauty_pet::graph::CanonGraph;
use rand::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelBridge;
use serde::Deserialize;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::{fs, thread};

const CANON_LIMIT: usize = 8;

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

    #[serde(rename = "num_crashes")]
    pub num_crashes_range: Range,

    #[serde(rename = "num_partitions", default = "default_partitions_range")]
    pub num_partitions_range: Range,

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

    #[serde(default)]
    pub schedule_policy: SchedulePolicy,
}

impl ExplorerConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.num_servers_range
            .validate()
            .map_err(|e| format!("num_servers range error: {}", e))?;
        self.num_write_ops_range
            .validate()
            .map_err(|e| format!("num_write_ops range error: {}", e))?;
        self.num_read_ops_range
            .validate()
            .map_err(|e| format!("num_read_ops range error: {}", e))?;
        self.num_crashes_range
            .validate()
            .map_err(|e| format!("num_crashes range error: {}", e))?;
        self.num_partitions_range
            .validate()
            .map_err(|e| format!("num_partitions range error: {}", e))?;
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

fn default_population_size() -> usize {
    50
}

fn default_num_generations() -> usize {
    100
}

fn default_use_coverage_scheduling() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct SingleRunConfig {
    pub num_servers: i32,
    pub num_write_ops: i32,
    pub num_read_ops: i32,
    pub num_crashes: i32,
    pub num_partitions: i32,
    pub dependency_density: f64,
    pub use_coverage_scheduling: bool,
    pub max_iterations: i32,
    pub schedule_policy: SchedulePolicy,
}

impl SingleRunConfig {
    pub fn random(constraints: &ExplorerConfig) -> Self {
        let mut rng = rand::rng();
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
            num_crashes: rng.random_range(
                constraints.num_crashes_range.min..=constraints.num_crashes_range.max,
            ),
            num_partitions: rng.random_range(
                constraints.num_partitions_range.min..=constraints.num_partitions_range.max,
            ),
            dependency_density: *constraints
                .dependency_density_values
                .choose(&mut rng)
                .unwrap_or(&0.5),
            use_coverage_scheduling: constraints.use_coverage_scheduling,
            max_iterations: constraints.max_iterations,
            schedule_policy: constraints.schedule_policy.clone(),
        }
    }

    pub fn mutate(&self, constraints: &ExplorerConfig) -> Self {
        let mut new_config = self.clone();
        let mut rng = rand::rng();

        let mutate_int = |val: i32, range: &Range| -> i32 {
            let mut rng = rand::rng();
            if rng.random_bool(0.3) {
                let delta = if rng.random_bool(0.5) { 1 } else { -1 };
                (val + delta).clamp(range.min, range.max)
            } else {
                val
            }
        };

        new_config.num_servers = mutate_int(self.num_servers, &constraints.num_servers_range);
        new_config.num_write_ops = mutate_int(self.num_write_ops, &constraints.num_write_ops_range);
        new_config.num_read_ops = mutate_int(self.num_read_ops, &constraints.num_read_ops_range);
        new_config.num_crashes = mutate_int(self.num_crashes, &constraints.num_crashes_range);
        new_config.num_partitions =
            mutate_int(self.num_partitions, &constraints.num_partitions_range);

        if rng.random_bool(0.3) && !constraints.dependency_density_values.is_empty() {
            new_config.dependency_density = *constraints
                .dependency_density_values
                .choose(&mut rng)
                .unwrap();
        }

        new_config
    }
}

fn initialize_state<H: crate::simulator::hash_utils::HashPolicy, L: Logger>(
    program: &Program,
    logger: &mut L,
    num_servers: usize,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
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
            exec_sync_on_node(
                &mut state,
                logger,
                program,
                &mut env,
                node_id,
                init_fn.entry,
                global_snapshot,
                local_coverage,
                &SchedulePolicy::Fixed,
            )?;
        }
    }

    Ok(state)
}

fn init_topology<H: crate::simulator::hash_utils::HashPolicy, L: Logger>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    num_servers: usize,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
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

        exec_sync_on_node(
            state,
            logger,
            program,
            &mut env,
            node_id,
            init_fn.entry,
            global_snapshot,
            local_coverage,
            &SchedulePolicy::Fixed,
        )?;
    }
    Ok(())
}

/// Runs a single simulation configuration and returns the plan score.
pub fn run_single_simulation(
    program: &Program,
    writer: &Arc<dyn HistoryWriter>,
    global_state: &GlobalState,
    run_id: i64,
    config: &SingleRunConfig,
) -> Result<f64, Box<dyn Error>> {
    let global_snapshot = global_state.coverage.snapshot();
    let gen_config = GeneratorConfig {
        num_servers: config.num_servers,
        num_write_ops: config.num_write_ops,
        num_read_ops: config.num_read_ops,
        num_crashes: config.num_crashes,
        num_partitions: config.num_partitions,
        dependency_density: config.dependency_density,
    };
    let plan = generate_plan(gen_config);
    let canonical = if plan.node_count() < CANON_LIMIT {
        let canonical = CanonGraph::from(plan.clone());
        if global_state.contains(&canonical) {
            return Ok(0.0);
        }
        Some(canonical)
    } else {
        None
    };

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
    let mut path_state = PathState::<crate::simulator::hash_utils::NoHashing>::new(
        &role_node_counts,
        program.max_node_slots as usize,
        client_role,
    );

    // Initialize state
    path_state.state = initialize_state::<crate::simulator::hash_utils::NoHashing, _>(
        program,
        &mut path_state.logs,
        num_servers,
        Some(&global_snapshot),
        &mut path_state.coverage,
    )?;

    let topology_info = TopologyInfo {
        topology: Topology::Full,
        num_servers: config.num_servers,
    };

    init_topology::<crate::simulator::hash_utils::NoHashing, _>(
        &mut path_state.state,
        &mut path_state.logs,
        program,
        num_servers,
        Some(&global_snapshot),
        &mut path_state.coverage,
    )?;

    exec_plan(
        &mut path_state,
        program.clone(),
        plan,
        config.max_iterations,
        topology_info,
        global_state,
        Some(&global_snapshot),
        run_id,
        &config.schedule_policy,
        false,
    )?;

    let plan_score = path_state.coverage.plan_score();

    global_state.coverage.merge(&path_state.coverage);
    if let Some(x) = canonical.as_ref() {
        global_state.insert(x)
    }

    let serialized = serialize_history(&path_state.history);
    let serialized_logs = serialize_logs(&path_state.logs.entries);
    let serialized_traces = serialize_traces(&path_state.logs.traces);
    writer.write(run_id, serialized, serialized_logs, serialized_traces);

    Ok(plan_score)
}

/// Runs the standard exhaustive explorer.
pub fn run_explorer(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<Arc<GlobalState>, Box<dyn Error>> {
    info!("Starting Execution Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;

    // Validate configuration before proceeding
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);

    let (sender, receiver) = channel::bounded::<(i64, SingleRunConfig)>(100);

    let config_producer = config.clone();
    let cancelled_producer = cancelled.clone();

    thread::spawn(move || {
        let config = config_producer;

        let all_servers = config.num_servers_range.expand();
        let all_writes = config.num_write_ops_range.expand();
        let all_reads = config.num_read_ops_range.expand();
        let all_crashes = config.num_crashes_range.expand();
        let all_partitions = config.num_partitions_range.expand();
        let all_densities = &config.dependency_density_values;

        let mut config_counter = 0;
        let mut run_counter = 0;
        let total_configs = all_servers.len()
            * all_writes.len()
            * all_reads.len()
            * all_crashes.len()
            * all_partitions.len()
            * all_densities.len();

        info!("Total unique configurations: {}", total_configs);
        info!("Runs per config: {}", config.num_runs_per_config);

        'outer: for &num_servers in &all_servers {
            for &num_writes in &all_writes {
                for &num_reads in &all_reads {
                    for &num_crashes in &all_crashes {
                        for &num_partitions in &all_partitions {
                            for &density in all_densities {
                                if cancelled_producer.load(Ordering::Relaxed) {
                                    break 'outer;
                                }
                                config_counter += 1;

                                let run_config = SingleRunConfig {
                                    num_servers,
                                    num_write_ops: num_writes,
                                    num_read_ops: num_reads,
                                    num_crashes,
                                    num_partitions,
                                    dependency_density: density,
                                    use_coverage_scheduling: config.use_coverage_scheduling,
                                    max_iterations: config.max_iterations,
                                    schedule_policy: config.schedule_policy.clone(),
                                };

                                info!("{}", "=".repeat(70));
                                info!(
                                    "Queuing Config {}/{}: s{}_w{}_r{}_crash{}_part{}_d{:.2}",
                                    config_counter,
                                    total_configs,
                                    num_servers,
                                    num_writes,
                                    num_reads,
                                    num_crashes,
                                    num_partitions,
                                    density
                                );
                                info!("{}", "=".repeat(70));

                                for _ in 1..=config.num_runs_per_config {
                                    run_counter += 1;
                                    if sender.send((run_counter, run_config.clone())).is_err() {
                                        return;
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

    let global_state = Arc::new(GlobalState::new(1_000_000));

    receiver
        .into_iter()
        .par_bridge()
        .for_each(|(run_id, run_config)| {
            let start = std::time::Instant::now();
            match run_single_simulation(program, &writer, &global_state, run_id, &run_config) {
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
    Ok(global_state)
}

/// Runs a single simulation with a pre-built execution plan.
fn run_single_plan(
    program: &Program,
    writer: &Arc<dyn HistoryWriter>,
    global_state: &GlobalState,
    run_id: i64,
    plan: &ExecutionPlan,
    num_servers: i32,
    max_iterations: i32,
    policy: &SchedulePolicy,
    strict_timers: bool,
) -> Result<f64, Box<dyn Error>> {
    let global_snapshot = global_state.coverage.snapshot();
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
    let mut path_state = PathState::<crate::simulator::hash_utils::NoHashing>::new(
        &role_node_counts,
        program.max_node_slots as usize,
        client_role,
    );

    path_state.state = initialize_state::<crate::simulator::hash_utils::NoHashing, _>(
        program,
        &mut path_state.logs,
        num_servers_usize,
        Some(&global_snapshot),
        &mut path_state.coverage,
    )?;

    let topology_info = TopologyInfo {
        topology: Topology::Full,
        num_servers,
    };

    init_topology::<crate::simulator::hash_utils::NoHashing, _>(
        &mut path_state.state,
        &mut path_state.logs,
        program,
        num_servers_usize,
        Some(&global_snapshot),
        &mut path_state.coverage,
    )?;

    exec_plan(
        &mut path_state,
        program.clone(),
        plan.clone(),
        max_iterations,
        topology_info,
        global_state,
        Some(&global_snapshot),
        run_id,
        policy,
        strict_timers,
    )?;

    let plan_score = path_state.coverage.plan_score();
    global_state.coverage.merge(&path_state.coverage);

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
) -> Result<Arc<GlobalState>, Box<dyn Error>> {
    use crate::simulator::plan_config::PlanFileConfig;

    info!("Starting Plan Runner...");
    info!("Plan config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: PlanFileConfig = serde_json::from_str(&config_json)?;
    config
        .validate()
        .map_err(|e| format!("Plan validation failed: {}", e))?;

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

    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);
    let global_state = Arc::new(GlobalState::new(1_000_000));

    let runs: Vec<i64> = (1..=config.num_runs as i64).collect();

    runs.par_iter().for_each(|&run_id| {
        if cancelled.load(Ordering::Relaxed) {
            return;
        }
        let start = std::time::Instant::now();
        match run_single_plan(
            program,
            &writer,
            &global_state,
            run_id,
            &plan,
            config.num_servers,
            config.max_iterations,
            &config.schedule_policy,
            config.strict_timers,
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
    Ok(global_state)
}

/// Runs the genetic algorithm-based explorer.
/// Returns the GlobalState containing accumulated coverage data.
pub fn run_explorer_genetic(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
    backend: LogBackend,
    cancelled: &Arc<AtomicBool>,
) -> Result<Arc<GlobalState>, Box<dyn Error>> {
    info!("Starting Genetic Execution Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;

    // Validate configuration before proceeding
    config
        .validate()
        .map_err(|e| format!("Configuration validation failed: {}", e))?;

    let writer: Arc<dyn HistoryWriter> = Arc::from(create_writer(backend, output_path)?);
    let global_state = Arc::new(GlobalState::new(1_000_000));
    let run_counter = Arc::new(AtomicI64::new(0));

    let mut population: Vec<SingleRunConfig> = (0..config.population_size)
        .map(|_| SingleRunConfig::random(&config))
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

        let scored: Vec<(SingleRunConfig, f64)> = population
            .par_iter()
            .map(|run_config| {
                let run_id = run_counter.fetch_add(1, Ordering::Relaxed);
                let result =
                    run_single_simulation(program, &writer, &global_state, run_id, run_config);
                match result {
                    Ok(score) => (run_config.clone(), score),
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

        let mut rng = rand::rng();
        for _ in 0..mutate_count {
            let parent = &scored[rng.random_range(0..elite_count.max(1))].0;
            next_gen.push(parent.mutate(&config));
        }

        for _ in 0..random_count {
            next_gen.push(SingleRunConfig::random(&config));
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
    Ok(global_state)
}
