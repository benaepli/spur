use crate::compiler::cfg::Program;
use crate::simulator::core::{
    Env, Logger, RuntimeError, State, Value, exec_sync_on_node, make_local_env,
};
use crate::simulator::coverage::{GlobalState, LocalCoverage};
use crate::simulator::history::{HistoryWriter, serialize_history, serialize_logs};
use crate::simulator::path::generator::{GeneratorConfig, generate_plan};
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
use std::sync::atomic::{AtomicI64, Ordering};
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
    pub fn expand(&self) -> Vec<i32> {
        if self.step <= 0 {
            panic!("Invalid range step: must be > 0");
        }
        (self.min..=self.max).step_by(self.step as usize).collect()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExplorerConfig {
    #[serde(rename = "num_servers")]
    pub num_servers_range: Range,

    #[serde(rename = "num_clients")]
    pub num_clients_range: Range,

    #[serde(rename = "num_write_ops")]
    pub num_write_ops_range: Range,

    #[serde(rename = "num_read_ops")]
    pub num_read_ops_range: Range,

    #[serde(rename = "num_timeouts")]
    pub num_timeouts_range: Range,

    #[serde(rename = "num_crashes")]
    pub num_crashes_range: Range,

    #[serde(rename = "dependency_density")]
    pub dependency_density_values: Vec<f64>,

    #[serde(default)]
    pub randomly_delay_msgs: bool,
    #[serde(default = "default_use_coverage_scheduling")]
    pub use_coverage_scheduling: bool,
    pub num_runs_per_config: i32,
    pub max_iterations: i32,

    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_num_generations")]
    pub num_generations: usize,
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
    pub num_clients: i32,
    pub num_write_ops: i32,
    pub num_read_ops: i32,
    pub num_timeouts: i32,
    pub num_crashes: i32,
    pub dependency_density: f64,
    pub randomly_delay_msgs: bool,
    pub use_coverage_scheduling: bool,
    pub max_iterations: i32,
}

impl SingleRunConfig {
    pub fn random(constraints: &ExplorerConfig) -> Self {
        let mut rng = rand::rng();
        SingleRunConfig {
            num_servers: rng.random_range(
                constraints.num_servers_range.min..=constraints.num_servers_range.max,
            ),
            num_clients: rng.random_range(
                constraints.num_clients_range.min..=constraints.num_clients_range.max,
            ),
            num_write_ops: rng.random_range(
                constraints.num_write_ops_range.min..=constraints.num_write_ops_range.max,
            ),
            num_read_ops: rng.random_range(
                constraints.num_read_ops_range.min..=constraints.num_read_ops_range.max,
            ),
            num_timeouts: rng.random_range(
                constraints.num_timeouts_range.min..=constraints.num_timeouts_range.max,
            ),
            num_crashes: rng.random_range(
                constraints.num_crashes_range.min..=constraints.num_crashes_range.max,
            ),
            dependency_density: *constraints
                .dependency_density_values
                .choose(&mut rng)
                .unwrap_or(&0.5),
            randomly_delay_msgs: constraints.randomly_delay_msgs,
            use_coverage_scheduling: constraints.use_coverage_scheduling,
            max_iterations: constraints.max_iterations,
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
        new_config.num_clients = mutate_int(self.num_clients, &constraints.num_clients_range);
        new_config.num_write_ops = mutate_int(self.num_write_ops, &constraints.num_write_ops_range);
        new_config.num_read_ops = mutate_int(self.num_read_ops, &constraints.num_read_ops_range);
        new_config.num_timeouts = mutate_int(self.num_timeouts, &constraints.num_timeouts_range);
        new_config.num_crashes = mutate_int(self.num_crashes, &constraints.num_crashes_range);

        if rng.random_bool(0.3) && !constraints.dependency_density_values.is_empty() {
            new_config.dependency_density = *constraints
                .dependency_density_values
                .choose(&mut rng)
                .unwrap();
        }

        new_config
    }
}

fn initialize_state<L: Logger>(
    program: &Program,
    logger: &mut L,
    num_servers: usize,
    num_clients: usize,
    local_coverage: &mut LocalCoverage,
) -> Result<State, RuntimeError> {
    let mut state = State::new(num_servers + num_clients, program.max_node_slots as usize);

    if let Some(init_fn) = program.get_func_by_name("ClientInterface.BASE_NODE_INIT") {
        for i in 0..num_clients {
            let client_id = num_servers + i;
            let node_env = &state.nodes[client_id];
            let mut env = make_local_env(init_fn, vec![], &Env::default(), node_env);
            exec_sync_on_node(
                &mut state,
                logger,
                program,
                &mut env,
                client_id,
                init_fn.entry,
                local_coverage,
            )?;
        }
    }

    if let Some(init_fn) = program.get_func_by_name("Node.BASE_NODE_INIT") {
        for node_id in 0..num_servers {
            let node_env = &state.nodes[node_id];
            let mut env = make_local_env(init_fn, vec![], &Env::default(), node_env);
            exec_sync_on_node(
                &mut state,
                logger,
                program,
                &mut env,
                node_id,
                init_fn.entry,
                local_coverage,
            )?;
        }
    }

    Ok(state)
}

fn init_topology<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    num_servers: usize,
    local_coverage: &mut LocalCoverage,
) -> Result<(), RuntimeError> {
    let init_fn_name = "Node.Init";
    let Some(init_fn) = program.get_func_by_name(init_fn_name) else {
        warn!("{} not found", init_fn_name);
        return Ok(());
    };

    let peer_list = Value::List((0..num_servers).map(|j| Value::Node(j)).collect());

    for node_id in 0..num_servers {
        let actuals = vec![Value::Int(node_id as i64), peer_list.clone()];
        let node_env = &state.nodes[node_id];
        let mut env = make_local_env(init_fn, actuals, &Env::default(), node_env);

        exec_sync_on_node(
            state,
            logger,
            program,
            &mut env,
            node_id,
            init_fn.entry,
            local_coverage,
        )?;
    }
    Ok(())
}

/// Runs a single simulation configuration and returns the plan score.
pub fn run_single_simulation(
    program: &Program,
    writer: &HistoryWriter,
    global_state: &GlobalState,
    run_id: i64,
    config: &SingleRunConfig,
) -> Result<f64, Box<dyn Error>> {
    let gen_config = GeneratorConfig {
        num_servers: config.num_servers,
        num_clients: config.num_clients,
        num_write_ops: config.num_write_ops,
        num_read_ops: config.num_read_ops,
        num_timeouts: config.num_timeouts,
        num_crashes: config.num_crashes,
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

    // Create PathState with free_clients initialized
    let num_servers = config.num_servers as usize;
    let num_clients = config.num_clients as usize;
    let mut path_state = PathState::new(num_servers + num_clients, program.max_node_slots as usize);
    path_state.free_clients = (0..num_clients).map(|i| (num_servers + i) as i32).collect();

    // Initialize state
    path_state.state = initialize_state(
        program,
        &mut path_state.logs,
        num_servers,
        num_clients,
        &mut path_state.coverage,
    )?;

    let topology_info = TopologyInfo {
        topology: Topology::Full,
        num_servers: config.num_servers,
    };

    init_topology(
        &mut path_state.state,
        &mut path_state.logs,
        program,
        num_servers,
        &mut path_state.coverage,
    )?;

    exec_plan(
        &mut path_state,
        program.clone(),
        plan,
        config.max_iterations,
        topology_info,
        config.randomly_delay_msgs,
        global_state,
    )?;

    let plan_score = path_state.coverage.plan_score();

    global_state.coverage.merge(&path_state.coverage);
    canonical.as_ref().map(|x| global_state.insert(x));

    let serialized = serialize_history(&path_state.history);
    let serialized_logs = serialize_logs(&path_state.logs.0);
    writer.write(run_id, serialized, serialized_logs);

    Ok(plan_score)
}

/// Runs a single simulation configuration.
pub fn run_explorer(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    info!("Starting Execution Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;

    if std::path::Path::new(output_path).exists() {
        fs::remove_file(output_path)?;
    }

    let writer = Arc::new(HistoryWriter::new(output_path)?);

    let (sender, receiver) = channel::bounded::<(i64, SingleRunConfig)>(100);

    let config_producer = config.clone();

    thread::spawn(move || {
        let config = config_producer;

        let all_servers = config.num_servers_range.expand();
        let all_clients = config.num_clients_range.expand();
        let all_writes = config.num_write_ops_range.expand();
        let all_reads = config.num_read_ops_range.expand();
        let all_timeouts = config.num_timeouts_range.expand();
        let all_crashes = config.num_crashes_range.expand();
        let all_densities = &config.dependency_density_values;

        let mut config_counter = 0;
        let mut run_counter = 0;
        let total_configs = all_servers.len()
            * all_clients.len()
            * all_writes.len()
            * all_reads.len()
            * all_timeouts.len()
            * all_crashes.len()
            * all_densities.len();

        info!("Total unique configurations: {}", total_configs);
        info!("Runs per config: {}", config.num_runs_per_config);

        for &num_servers in &all_servers {
            for &num_clients in &all_clients {
                for &num_writes in &all_writes {
                    for &num_reads in &all_reads {
                        for &num_timeouts in &all_timeouts {
                            for &num_crashes in &all_crashes {
                                for &density in all_densities {
                                    config_counter += 1;

                                    let run_config = SingleRunConfig {
                                        num_servers,
                                        num_clients,
                                        num_write_ops: num_writes,
                                        num_read_ops: num_reads,
                                        num_timeouts,
                                        num_crashes,
                                        dependency_density: density,
                                        randomly_delay_msgs: config.randomly_delay_msgs,
                                        use_coverage_scheduling: config.use_coverage_scheduling,
                                        max_iterations: config.max_iterations,
                                    };

                                    info!("{}", "=".repeat(70));
                                    info!(
                                        "Queuing Config {}/{}: s{}_c{}_w{}_r{}_t{}_crash{}_d{:.2}",
                                        config_counter,
                                        total_configs,
                                        num_servers,
                                        num_clients,
                                        num_writes,
                                        num_reads,
                                        num_timeouts,
                                        num_crashes,
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
    if let Ok(w) = Arc::try_unwrap(writer) {
        w.shutdown();
    }

    info!("Execution explorer finished.");
    Ok(())
}

/// Runs the genetic algorithm-based explorer.
/// Returns the GlobalState containing accumulated coverage data.
pub fn run_explorer_genetic(
    program: &Program,
    config_json_path: &str,
    output_path: &str,
) -> Result<Arc<GlobalState>, Box<dyn Error>> {
    info!("Starting Genetic Execution Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;

    if std::path::Path::new(output_path).exists() {
        fs::remove_file(output_path)?;
    }

    let writer = Arc::new(HistoryWriter::new(output_path)?);
    let global_state = Arc::new(GlobalState::new(1_000_000));
    let run_counter = Arc::new(AtomicI64::new(0));

    let mut population: Vec<SingleRunConfig> = (0..config.population_size)
        .map(|_| SingleRunConfig::random(&config))
        .collect();

    for generation in 0..config.num_generations {
        info!(
            "=== Generation {}/{} ===",
            generation + 1,
            config.num_generations
        );

        let scored: Vec<(SingleRunConfig, f64)> = population
            .par_iter()
            .map(|run_config| {
                let run_id = run_counter.fetch_add(1, Ordering::Relaxed);
                let score =
                    run_single_simulation(program, &writer, &global_state, run_id, run_config)
                        .unwrap_or(0.0);
                (run_config.clone(), score)
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

    if let Ok(w) = Arc::try_unwrap(writer) {
        w.shutdown();
    }

    info!("Genetic explorer finished.");
    Ok(global_state)
}
