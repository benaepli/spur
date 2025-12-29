use crate::compiler::cfg::Program;
use crate::simulator::core::{exec_sync_on_node, RuntimeError, State, Value};
use crate::simulator::execution::{exec_plan, Topology, TopologyInfo};
use crate::simulator::generator::{generate_plan, GeneratorConfig};
use crate::simulator::history::{init_sqlite, save_history_sqlite};
use log::{debug, error, info, warn};
use rusqlite::Connection;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct ExplorerConfig {
    pub num_servers: Range,
    pub num_clients: Range,
    pub num_write_ops: Range,
    pub num_read_ops: Range,
    pub num_timeouts: Range,
    pub num_crashes: Range,
    pub dependency_density: Vec<f64>,
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
    state.free_clients = (0..num_clients)
        .map(|i| (num_servers + i) as i32)
        .collect();

    if let Some(init_fn) = program.rpc.get("ClientInterface.BASE_NODE_INIT") {
        for i in 0..num_clients {
            let client_id = num_servers + i;
            let mut env = HashMap::new();
            env.insert("self".to_string(), Value::Node(client_id));
            exec_sync_on_node(&mut state, program, &mut env, client_id, init_fn.entry)?;
        }
    }

    if let Some(init_fn) = program.rpc.get("Node.BASE_NODE_INIT") {
        for node_id in 0..num_servers {
            let mut env = HashMap::new();
            env.insert("self".to_string(), Value::Node(node_id));
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
    let Some(init_fn) = program.rpc.get(init_fn_name) else {
        warn!("{} not found", init_fn_name);
        return Ok(());
    };

    let peer_list = Value::List((0..num_servers).map(|j| Value::Node(j)).collect());

    for node_id in 0..num_servers {
        let actuals = vec![Value::Node(node_id), peer_list.clone()];
        let mut env = HashMap::new();

        for (i, formal) in init_fn.formals.iter().enumerate() {
            env.insert(formal.clone(), actuals[i].clone());
        }

        exec_sync_on_node(state, program, &mut env, node_id, init_fn.entry)?;
    }
    Ok(())
}

/// Runs a single simulation configuration.
pub fn run_single_simulation(
    program: &Program,
    conn: &mut Connection,
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

    save_history_sqlite(conn, run_id, &state.history)?;

    Ok(())
}

pub fn run_explorer(
    program: &mut Program,
    config_json_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    info!("Starting Execution Explorer...");
    info!("Config: {}", config_json_path);

    let config_json = fs::read_to_string(config_json_path)?;
    let config: ExplorerConfig = serde_json::from_str(&config_json)?;

    let mut conn = Connection::open(output_path)?;
    init_sqlite(&conn)?;

    let all_servers = config.num_servers.expand();
    let all_clients = config.num_clients.expand();
    let all_writes = config.num_write_ops.expand();
    let all_reads = config.num_read_ops.expand();
    let all_timeouts = config.num_timeouts.expand();
    let all_crashes = config.num_crashes.expand();
    let all_densities = &config.dependency_density;

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
                                    "Running Config {}/{}: s{}_c{}_w{}_r{}_t{}_crash{}_d{:.2}",
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

                                for i in 1..=config.num_runs_per_config {
                                    run_counter += 1;
                                    debug!("Run {}/{} (Total #{}) ... ", i, config.num_runs_per_config, run_counter);

                                    let start = std::time::Instant::now();

                                    match run_single_simulation(
                                        &program,
                                        &mut conn,
                                        run_counter,
                                        &run_config,
                                    ) {
                                        Ok(_) => debug!("Success ({:.4}s)", start.elapsed().as_secs_f64()),
                                        Err(e) => error!("Run failed: {}", e),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    info!("Execution explorer finished.");
    Ok(())
}