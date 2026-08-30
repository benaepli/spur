use petgraph::algo::has_path_connecting;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::prelude::*;
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
    pub num_rmw_ops: i32,
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
    /// Cap on concurrent in-flight write-like operations (Write and RMW).
    /// When set to K >= 1, each write-like[i] depends on write-like[i - K]
    /// (declaration order, global across keys), so at most K can be ready
    /// simultaneously. `None` disables the cap. The simulator rejects
    /// `Some(0)` during config validation.
    pub max_concurrent_writes: Option<i32>,
    /// Client requests reserved for after each node restart. When set to K >= 1,
    /// every crash/recover pair gets mandatory edges from its recover to K
    /// client requests that were not already ordered after it, so client work
    /// survives the fault instead of all of it becoming eligible up front.
    /// 0 reserves nothing.
    pub post_fault_client_ops: i32,
    /// Probability that a given crash/recover pair reserves client work at all.
    /// 1.0 reserves for every pair; a smaller value interpolates between a
    /// reservation count of 0 and `post_fault_client_ops`, which the integer
    /// knob alone cannot express. Values outside [0, 1] are clamped.
    pub post_fault_client_ops_prob: f64,
}

/// Generates a bag of action stubs based on the config.
fn generate_base_actions(config: &GeneratorConfig, rng: &mut impl Rng) -> Vec<ActionStub> {
    let mut actions = Vec::new();

    let num_keys = config.num_keys.max(1);

    for _ in 0..config.num_write_ops {
        let server = rng.random_range(0..config.num_servers);
        let key = format!("key{}", rng.random_range(1..=num_keys));
        let action = ClientOpSpec::Write(server, ecow::EcoString::from(key));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_read_ops {
        let server = rng.random_range(0..config.num_servers);
        let key = format!("key{}", rng.random_range(1..=num_keys));
        let action = ClientOpSpec::Read(server, ecow::EcoString::from(key));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_rmw_ops {
        let server = rng.random_range(0..config.num_servers);
        let key = format!("key{}", rng.random_range(1..=num_keys));
        let action = ClientOpSpec::Rmw(server, ecow::EcoString::from(key));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_crashes {
        let s = rng.random_range(0..config.num_servers);
        actions.push(ActionStub::Paired(
            EventAction::CrashNode(s),
            EventAction::RecoverNode(s),
        ));
    }

    for _ in 0..config.num_partitions {
        let spec = random_partition_spec(config.num_servers, rng);
        actions.push(ActionStub::Paired(
            EventAction::Partition(spec),
            EventAction::Heal,
        ));
    }

    actions
}

/// Generate a random PartitionSpec given the number of servers.
fn random_partition_spec(num_servers: i32, rng: &mut impl Rng) -> PartitionSpec {
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
pub fn generate_plan(config: GeneratorConfig, rng: &mut impl Rng) -> ExecutionPlan {
    let mut graph: DiGraph<PlannedEvent, ()> = DiGraph::new();

    // Track crash/recover pairs and serialization
    let mut last_recovery: HashMap<i32, NodeIndex> = HashMap::new(); // server_id -> last recover node
    // Track partition/heal serialization (only one partition active at a time)
    let mut last_heal: Option<NodeIndex> = None;

    let stubs = generate_base_actions(&config, rng);

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
                if matches!(action1, EventAction::Partition(_))
                    && let Some(prev_heal) = last_heal {
                        graph.add_edge(prev_heal, idx1, ());
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
    // Both Write and Rmw participate (both mutate state).
    if let Some(k) = config.max_concurrent_writes
        && k >= 1
    {
        let write_indices: Vec<NodeIndex> = nodes
            .iter()
            .filter(|(idx, _)| {
                matches!(
                    graph[*idx].action,
                    EventAction::ClientRequest(ClientOpSpec::Write(..))
                        | EventAction::ClientRequest(ClientOpSpec::Rmw(..))
                )
            })
            .map(|(idx, _)| *idx)
            .collect();
        let k = k as usize;
        for i in k..write_indices.len() {
            graph.add_edge(write_indices[i - k], write_indices[i], ());
        }
    }

    // Post-fault pass: order a few client requests after each recover, so a run
    // still has client work to issue once the faults have happened.
    if config.post_fault_client_ops >= 1 {
        let client_indices: Vec<NodeIndex> = nodes
            .iter()
            .filter(|(idx, _)| {
                matches!(graph[*idx].action, EventAction::ClientRequest(_))
            })
            .map(|(idx, _)| *idx)
            .collect();
        let recover_indices: Vec<NodeIndex> = nodes
            .iter()
            .filter(|(idx, _)| matches!(graph[*idx].action, EventAction::RecoverNode(_)))
            .map(|(idx, _)| *idx)
            .collect();
        let wanted = config.post_fault_client_ops as usize;
        let probability = config.post_fault_client_ops_prob.clamp(0.0, 1.0);
        let mut edges_added = 0u64;
        let mut pairs_skipped = 0u64;
        for recover in &recover_indices {
            if probability < 1.0 && !rng.random_bool(probability) {
                pairs_skipped += 1;
                continue;
            }
            let mut candidates = client_indices.clone();
            candidates.shuffle(rng);
            let mut added = 0;
            for client in candidates {
                if added >= wanted {
                    break;
                }
                if has_path_connecting(&graph, *recover, client, None) {
                    continue;
                }
                if has_path_connecting(&graph, client, *recover, None) {
                    continue;
                }
                graph.add_edge(*recover, client, ());
                added += 1;
            }
            edges_added += added as u64;
        }
        crate::simulator::util_stats::record_post_fault_ops(
            recover_indices.len() as u64,
            edges_added,
            pairs_skipped,
        );
    }

    // Shuffle node order for dependency generation
    nodes.shuffle(rng);

    // Second pass: add probabilistic dependencies. Skip any candidate edge
    // whose target already has a path back to the source. This guards
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

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Direction;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn config(post_fault_client_ops: i32) -> GeneratorConfig {
        config_with_probability(post_fault_client_ops, 1.0)
    }

    fn config_with_probability(
        post_fault_client_ops: i32,
        post_fault_client_ops_prob: f64,
    ) -> GeneratorConfig {
        GeneratorConfig {
            num_servers: 3,
            num_write_ops: 3,
            num_read_ops: 4,
            num_rmw_ops: 0,
            num_keys: 1,
            num_crashes: 2,
            num_partitions: 0,
            dependency_density: 0.0,
            max_concurrent_writes: Some(2),
            post_fault_client_ops,
            post_fault_client_ops_prob,
        }
    }

    fn recovers_with_a_client_successor(plan: &ExecutionPlan) -> (usize, usize) {
        let mut recovers = 0;
        let mut with_client = 0;
        for idx in plan.node_indices() {
            if !matches!(plan[idx].action, EventAction::RecoverNode(_)) {
                continue;
            }
            recovers += 1;
            if plan
                .neighbors_directed(idx, Direction::Outgoing)
                .any(|n| matches!(plan[n].action, EventAction::ClientRequest(_)))
            {
                with_client += 1;
            }
        }
        (recovers, with_client)
    }

    #[test]
    fn zero_reserves_no_client_work_after_a_restart() {
        for seed in 0..32u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let plan = generate_plan(config(0), &mut rng);
            let (recovers, with_client) = recovers_with_a_client_successor(&plan);
            assert_eq!(recovers, 2);
            assert_eq!(with_client, 0, "seed {}", seed);
        }
    }

    #[test]
    fn probability_zero_matches_reserving_nothing() {
        for seed in 0..32u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let plan = generate_plan(config_with_probability(1, 0.0), &mut rng);
            let (recovers, with_client) = recovers_with_a_client_successor(&plan);
            assert_eq!(recovers, 2);
            assert_eq!(with_client, 0, "seed {}", seed);
        }
    }

    #[test]
    fn a_fractional_probability_reserves_for_some_restarts_and_not_others() {
        let mut reserved = 0;
        let mut total = 0;
        for seed in 0..64u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let plan = generate_plan(config_with_probability(1, 0.5), &mut rng);
            let (recovers, with_client) = recovers_with_a_client_successor(&plan);
            reserved += with_client;
            total += recovers;
            assert!(
                !petgraph::algo::is_cyclic_directed(&plan),
                "seed {} produced a cyclic plan",
                seed
            );
        }
        assert!(reserved > 0 && reserved < total, "{}/{}", reserved, total);
    }

    #[test]
    fn every_restart_gets_a_client_request_ordered_after_it() {
        for seed in 0..32u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let plan = generate_plan(config(1), &mut rng);
            let (recovers, with_client) = recovers_with_a_client_successor(&plan);
            assert_eq!(recovers, with_client, "seed {}", seed);
            assert!(
                !petgraph::algo::is_cyclic_directed(&plan),
                "seed {} produced a cyclic plan",
                seed
            );
        }
    }
}
