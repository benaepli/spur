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
    // Fault specs
    pub num_crashes: i32,     // Number of crash/recover pairs
    pub num_partitions: i32,  // Number of partition/heal pairs
    // Dependency specs
    pub dependency_density: f64, // Probability (0.0 to 1.0)
}

/// Generates a bag of action stubs based on the config.
fn generate_base_actions(config: &GeneratorConfig) -> Vec<ActionStub> {
    let mut actions = Vec::new();

    let rand_server = || rng().random_range(0..config.num_servers);
    let rand_key = || format!("key{}", rng().random_range(1..=3));

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

    // Shuffle node order for dependency generation
    nodes.shuffle(&mut rng);

    // Second pass: add probabilistic dependencies
    let mut seen: Vec<(NodeIndex, Option<(i32, PairPos)>)> = Vec::new();
    for (current_idx, current_pair) in &nodes {
        for (prev_idx, prev_pair) in &seen {
            // Skip cycle-forming edges within same pair
            let is_cycle = match (current_pair, prev_pair) {
                (Some((g1, PairPos::First)), Some((g2, PairPos::Second))) => g1 == g2,
                (Some((g1, PairPos::Second)), Some((g2, PairPos::First))) => g1 == g2,
                _ => false,
            };

            if !is_cycle && rng.random::<f64>() < config.dependency_density {
                graph.add_edge(*prev_idx, *current_idx, ());
            }
        }
        seen.push((*current_idx, *current_pair));
    }

    graph
}
