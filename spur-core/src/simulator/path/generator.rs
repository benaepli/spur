use petgraph::algo::has_path_connecting;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;

use crate::simulator::path::plan::{ClientOpSpec, EventAction, ExecutionPlan, PlannedEvent};
use crate::simulator::plan_config::PartitionSpec;

#[derive(Debug, Clone)]
enum ActionStub {
    Single(EventAction),
    // e.g., Crash followed by Recover
    Paired(EventAction, EventAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairPos {
    First,
    Second,
}

/// Configuration for the plan generator.
pub struct GeneratorConfig {
    pub num_servers: i32,
    // Client operations
    pub num_write_ops: i32,
    pub num_read_ops: i32,
    /// Number of distinct keys (`key1`..`keyN`) used by generated Write/Read
    /// invocations. Must be >= 1. Defaults to 1 in the explorer config; a
    /// single key concentrates per-key interleavings and surfaces most
    /// linearizability bugs faster.
    pub num_keys: i32,
    // Fault specs
    pub num_crashes: i32,     // Number of crash/recover pairs
    pub num_partitions: i32,  // Number of partition/heal pairs
    // Dependency specs
    pub dependency_density: f64, // Probability (0.0 to 1.0)
    /// Cap on concurrent in-flight Write operations. When set to K >= 1,
    /// each write[i] depends on write[i - K] (declaration order, global across
    /// keys), so at most K writes can be ready simultaneously. `None` disables
    /// the cap. The simulator rejects `Some(0)` during config validation.
    pub max_concurrent_writes: Option<i32>,
}

/// Generates a bag of action stubs based on the config.
fn generate_base_actions(config: &GeneratorConfig) -> Vec<ActionStub> {
    let mut actions = Vec::new();

    let rand_server = || rng().random_range(0..config.num_servers);
    let num_keys = config.num_keys.max(1);
    let rand_key = || format!("key{}", rng().random_range(1..=num_keys));

    for _ in 0..config.num_write_ops {
        let action = ClientOpSpec::Write(rand_server(), ecow::EcoString::from(rand_key()));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_read_ops {
        let action = ClientOpSpec::Read(rand_server(), ecow::EcoString::from(rand_key()));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_crashes {
        let s = rand_server();
        actions.push(ActionStub::Paired(
            EventAction::CrashNode(s),
            EventAction::RecoverNode(s),
        ));
    }

    for _ in 0..config.num_partitions {
        let spec = random_partition_spec(config.num_servers);
        actions.push(ActionStub::Paired(
            EventAction::Partition(spec),
            EventAction::Heal,
        ));
    }

    actions
}

/// Generate a random PartitionSpec given the number of servers.
fn random_partition_spec(num_servers: i32) -> PartitionSpec {
    let mut rng = rng();
    match rng.random_range(0..4) {
        0 => PartitionSpec::IsolateOne {
            node: rng.random_range(0..num_servers),
        },
        1 => {
            // Random non-empty proper subset for side_a
            let mut side_a: Vec<i32> = (0..num_servers)
                .filter(|_| rng.random_bool(0.5))
                .collect();
            if side_a.is_empty() {
                side_a.push(rng.random_range(0..num_servers));
            } else if side_a.len() == num_servers as usize {
                side_a.remove(rng.random_range(0..side_a.len()));
            }
            PartitionSpec::Halves { side_a }
        }
        2 => PartitionSpec::MajoritiesRing,
        _ => PartitionSpec::Bridge {
            bridge: rng.random_range(0..num_servers),
        },
    }
}

/// Main entry point: Generates a single, randomized execution plan as a DiGraph.
pub fn generate_plan(config: GeneratorConfig) -> ExecutionPlan {
    let mut graph: DiGraph<PlannedEvent, ()> = DiGraph::new();
    let mut rng = rng();

    // Track crash/recover pairs and serialization
    let mut last_recovery: HashMap<i32, NodeIndex> = HashMap::new(); // server_id -> last recover node
    // Track partition/heal serialization (only one partition active at a time)
    let mut last_heal: Option<NodeIndex> = None;

    let stubs = generate_base_actions(&config);

    // First pass: add all nodes and mandatory edges
    let mut nodes: Vec<(NodeIndex, Option<(i32, PairPos)>)> = Vec::new();
    let mut pair_group_counter = 0;

    for stub in &stubs {
        match stub {
            ActionStub::Single(action) => {
                let idx = graph.add_node(PlannedEvent {
                    action: action.clone(),
                });
                nodes.push((idx, None));
            }
            ActionStub::Paired(action1, action2) => {
                pair_group_counter += 1;
                let idx1 = graph.add_node(PlannedEvent {
                    action: action1.clone(),
                });
                let idx2 = graph.add_node(PlannedEvent {
                    action: action2.clone(),
                });

                // Crash -> Recover edge (mandatory)
                graph.add_edge(idx1, idx2, ());

                // Serialization: this crash depends on previous recovery of same server
                if let EventAction::CrashNode(s) = action1
                    && let Some(&prev_recover) = last_recovery.get(s)
                {
                    graph.add_edge(prev_recover, idx1, ());
                }
                if let EventAction::RecoverNode(s) = action2 {
                    last_recovery.insert(*s, idx2);
                }

                // Serialization: partitions are globally serialized
                if matches!(action1, EventAction::Partition(_)) {
                    if let Some(prev_heal) = last_heal {
                        graph.add_edge(prev_heal, idx1, ());
                    }
                }
                if matches!(action2, EventAction::Heal) {
                    last_heal = Some(idx2);
                }

                nodes.push((idx1, Some((pair_group_counter, PairPos::First))));
                nodes.push((idx2, Some((pair_group_counter, PairPos::Second))));
            }
        }
    }

    // Write-chain pass: enforce max_concurrent_writes by adding a mandatory
    // edge writes[i - K] -> writes[i]. Declaration order; keys are not tracked
    // separately, so K is a global cap (strict upper bound on per-key blowup).
    if let Some(k) = config.max_concurrent_writes
        && k >= 1
    {
        let write_indices: Vec<NodeIndex> = nodes
            .iter()
            .filter(|(idx, _)| {
                matches!(
                    graph[*idx].action,
                    EventAction::ClientRequest(ClientOpSpec::Write(..))
                )
            })
            .map(|(idx, _)| *idx)
            .collect();
        let k = k as usize;
        for i in k..write_indices.len() {
            graph.add_edge(write_indices[i - k], write_indices[i], ());
        }
    }

    // Shuffle node order for dependency generation
    nodes.shuffle(&mut rng);

    // Second pass: add probabilistic dependencies. Skip any candidate edge
    // whose target already has a path back to the source — this guards
    // against cycles with every mandatory edge (write-chain, crash/recover
    // serialization, partition/heal serialization).
    let mut seen: Vec<(NodeIndex, Option<(i32, PairPos)>)> = Vec::new();
    for (current_idx, current_pair) in &nodes {
        for (prev_idx, _prev_pair) in &seen {
            if rng.random::<f64>() >= config.dependency_density {
                continue;
            }
            if has_path_connecting(&graph, *current_idx, *prev_idx, None) {
                continue;
            }
            graph.add_edge(*prev_idx, *current_idx, ());
        }
        seen.push((*current_idx, *current_pair));
    }

    graph
}
