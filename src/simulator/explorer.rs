use crate::compiler::cfg::Program;
use crate::simulator::core::{
    Env, RuntimeError, SELF_NAME_ID, State, Value, eval, exec_sync_on_node,
};
use crate::simulator::execution::{Topology, TopologyInfo, exec_plan};
use crate::simulator::generator::{GeneratorConfig, generate_plan};
use crate::simulator::history::{init_sqlite, save_history_sqlite};
use crossbeam::channel;
use log::{debug, error, info, warn};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelBridge;
use serde::Deserialize;
use std::error::Error;
use std::{fs, thread};

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
    pub num_runs_per_config: i32,
    pub max_iterations: i32,
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
    pub max_iterations: i32,
}

fn initialize_state(
    program: &Program,
    num_servers: usize,
    num_clients: usize,
) -> Result<State, RuntimeError> {
    let mut state = State::new(num_servers + num_clients);
    state.free_clients = (0..num_clients).map(|i| (num_servers + i) as i32).collect();

    if let Some(init_fn) = program.get_func_by_name("ClientInterface.BASE_NODE_INIT") {
        for i in 0..num_clients {
            let client_id = num_servers + i;
            let mut env = Env::default();
            env.insert(SELF_NAME_ID, Value::Node(client_id));
            let node_env = state.nodes[client_id].borrow();
            for (name, expr) in &init_fn.locals {
                env.insert(*name, eval(&env, &node_env, expr)?);
            }
            drop(node_env);
            exec_sync_on_node(&mut state, program, &mut env, client_id, init_fn.entry)?;
        }
    }

    if let Some(init_fn) = program.get_func_by_name("Node.BASE_NODE_INIT") {
        for node_id in 0..num_servers {
            let mut env = Env::default();
            env.insert(SELF_NAME_ID, Value::Node(node_id));
            let node_env = state.nodes[node_id].borrow();
            for (name, expr) in &init_fn.locals {
                env.insert(*name, eval(&env, &node_env, expr)?);
            }
            drop(node_env);
            exec_sync_on_node(&mut state, program, &mut env, node_id, init_fn.entry)?;
        }
    }

    Ok(state)
}

fn init_topology(
    state: &mut State,
    program: &Program,
    num_servers: usize,
) -> Result<(), RuntimeError> {
    let init_fn_name = "Node.Init";
    let Some(init_fn) = program.get_func_by_name(init_fn_name) else {
        warn!("{} not found", init_fn_name);
        return Ok(());
    };

    let peer_list = Value::List((0..num_servers).map(|j| Value::Node(j)).collect());

    for node_id in 0..num_servers {
        let actuals = vec![Value::Int(node_id as i64), peer_list.clone()];
        let mut env = Env::default();

        for (i, formal) in init_fn.formals.iter().enumerate() {
            env.insert(*formal, actuals[i].clone());
        }

        exec_sync_on_node(state, program, &mut env, node_id, init_fn.entry)?;
    }
    Ok(())
}

/// Runs a single simulation configuration.
pub fn run_single_simulation(
    program: &Program,
    pool: &Pool<SqliteConnectionManager>,
    run_id: i64,
    config: &SingleRunConfig,
) -> Result<(), Box<dyn Error>> {
    let mut state = initialize_state(
        program,
        config.num_servers as usize,
        config.num_clients as usize,
    )?;

    let topology_info = TopologyInfo {
        topology: Topology::Full,
        num_servers: config.num_servers,
    };
    init_topology(&mut state, program, config.num_servers as usize)?;

    let gen_config = GeneratorConfig {
        num_servers: config.num_servers,
        num_clients: config.num_clients,
        num_write_ops: config.num_write_ops,
        num_read_ops: config.num_read_ops,
        num_timeouts: config.num_timeouts,
        num_crashes: config.num_crashes,
        dependency_density: config.dependency_density,
    };

    let plan = generate_plan(gen_config)?;

    exec_plan(
        &mut state,
        program.clone(),
        plan,
        config.max_iterations,
        topology_info,
        config.randomly_delay_msgs,
    )?;

    let mut conn = pool.get()?;
    save_history_sqlite(&mut conn, run_id, &state.history)?;

    Ok(())
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

    let manager = SqliteConnectionManager::file(output_path).with_init(|c| init_sqlite(c)); //
    let pool = Pool::new(manager)?;

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

    receiver
        .into_iter()
        .par_bridge()
        .for_each(|(run_id, run_config)| {
            let start = std::time::Instant::now();
            match run_single_simulation(program, &pool, run_id, &run_config) {
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

    info!("Execution explorer finished.");
    Ok(())
}
