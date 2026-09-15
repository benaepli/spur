use petgraph::algo::has_path_connecting;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

use std::sync::Arc;

use crate::simulator::core::state::NodeId;
use crate::simulator::deploy::Deployment;
use crate::simulator::path::plan::{ClientOpSpec, EventAction, ExecutionPlan, PartitionAction, PlannedEvent};
use crate::simulator::recover_deps::RecoverDeps;

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
    pub deployment: Arc<Deployment>,
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
    /// Whether a restart may take a probabilistic edge from an earlier event.
    /// The mandatory edges - crash before its own recover, one recover
    /// before the same node's next crash, and the post-fault recover before
    /// client edges - are the same under both values.
    pub recover_deps: RecoverDeps,
}

/// Generates a bag of action stubs based on the config.
fn generate_base_actions(config: &GeneratorConfig, rng: &mut impl Rng) -> Vec<ActionStub> {
    let mut actions = Vec::new();

    let num_keys = config.num_keys.max(1);

    for _ in 0..config.num_write_ops {
        let dest = draw_destination(&config.deployment.destinations[0], rng);
        let key = format!("key{}", rng.random_range(1..=num_keys));
        let action = ClientOpSpec::Write(dest, ecow::EcoString::from(key));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_read_ops {
        let dest = draw_destination(&config.deployment.destinations[1], rng);
        let key = format!("key{}", rng.random_range(1..=num_keys));
        let action = ClientOpSpec::Read(dest, ecow::EcoString::from(key));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    for _ in 0..config.num_rmw_ops {
        let dest = draw_destination(&config.deployment.destinations[2], rng);
        let key = format!("key{}", rng.random_range(1..=num_keys));
        let action = ClientOpSpec::Rmw(dest, ecow::EcoString::from(key));
        actions.push(ActionStub::Single(EventAction::ClientRequest(action)));
    }

    let candidates = &config.deployment.crash_candidates;
    for _ in 0..config.num_crashes {
        if candidates.is_empty() {
            break;
        }
        let s = config.deployment.nodes[candidates[draw_position(candidates.len(), rng)]];
        actions.push(ActionStub::Paired(
            EventAction::CrashNode(s),
            EventAction::RecoverNode(s),
        ));
    }

    for _ in 0..config.num_partitions {
        let Some(partition) = random_partition(&config.deployment, rng) else {
            break;
        };
        actions.push(ActionStub::Paired(
            EventAction::Partition(partition),
            EventAction::Heal,
        ));
    }

    actions
}

/// A position below `len`, drawn as an `i32` range so the draw consumes the
/// workload stream exactly as a draw over a server count did.
fn draw_position(len: usize, rng: &mut impl Rng) -> usize {
    rng.random_range(0..len as i32) as usize
}

fn draw_destination(candidates: &Option<Vec<NodeId>>, rng: &mut impl Rng) -> Option<NodeId> {
    candidates
        .as_ref()
        .map(|nodes| nodes[draw_position(nodes.len(), rng)])
}

/// Whether `members` can form a ring: at least four, none repeated.
pub(crate) fn ring_eligible(members: &[NodeId]) -> bool {
    members.len() >= 4 && members.iter().collect::<HashSet<_>>().len() == members.len()
}

/// The members of the group a group-shaped partition applies to, among the
/// groups `eligible` accepts. A shape that prefers quorum groups uses the
/// eligible ones when any exist. A single candidate takes no draw.
fn choose_group(
    deployment: &Deployment,
    prefer_quorum: bool,
    eligible: fn(&[NodeId]) -> bool,
    rng: &mut impl Rng,
) -> Option<Vec<NodeId>> {
    let accepted = || deployment.groups.iter().filter(|g| eligible(&g.members));
    let quorum: Vec<_> = accepted().filter(|g| g.quorum).collect();
    let candidates: Vec<_> = if prefer_quorum && !quorum.is_empty() {
        quorum
    } else {
        accepted().collect()
    };
    match candidates.len() {
        0 => None,
        1 => Some(candidates[0].members.clone()),
        n => Some(candidates[rng.random_range(0..n)].members.clone()),
    }
}

/// A random partition, redrawing the shape until one has an eligible target.
/// A deployment with no nodes has none.
fn random_partition(deployment: &Deployment, rng: &mut impl Rng) -> Option<PartitionAction> {
    if deployment.nodes.is_empty() {
        return None;
    }
    loop {
        match rng.random_range(0..4) {
            0 => {
                let node = deployment.nodes[draw_position(deployment.nodes.len(), rng)];
                return Some(PartitionAction::IsolateOne(node));
            }
            1 => {
                let Some(group) = choose_group(deployment, false, |m| !m.is_empty(), rng) else { continue };
                let n = group.len();
                let mut side_a: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.5)).collect();
                if side_a.is_empty() {
                    side_a.push(draw_position(n, rng));
                } else if side_a.len() == n {
                    side_a.remove(rng.random_range(0..side_a.len()));
                }
                return Some(PartitionAction::Halves { group, side_a });
            }
            2 => {
                let Some(group) = choose_group(deployment, true, ring_eligible, rng) else { continue };
                return Some(PartitionAction::MajoritiesRing { group });
            }
            _ => {
                let Some(group) = choose_group(deployment, true, |m| !m.is_empty(), rng) else { continue };
                let bridge = draw_position(group.len(), rng);
                return Some(PartitionAction::Bridge { group, bridge });
            }
        }
    }
}

/// Main entry point: Generates a single, randomized execution plan as a DiGraph.
pub fn generate_plan(config: GeneratorConfig, rng: &mut impl Rng) -> ExecutionPlan {
    let mut graph: DiGraph<PlannedEvent, ()> = DiGraph::new();

    // Track crash/recover pairs and serialization
    let mut last_recovery: HashMap<NodeId, NodeIndex> = HashMap::new(); // node -> its last recover
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
        let mut edges_added = 0u64;
        for recover in &recover_indices {
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
        );
    }

    // Shuffle node order for dependency generation
    nodes.shuffle(rng);

    // Second pass: add probabilistic dependencies. Skip any candidate edge
    // whose target already has a path back to the source. This guards
    // against cycles with every mandatory edge (write-chain, crash/recover
    // serialization, partition/heal serialization).
    //
    // Under the exempt cell an edge into a recover is still added here and
    // removed once the pass is over, so the cycle guard sees exactly the
    // graph the stock cell sees: the exempt plan is the stock plan minus
    // those edges and nothing else.
    let mut seen: Vec<(NodeIndex, Option<(i32, PairPos)>)> = Vec::new();
    let mut exempt_edges: Vec<EdgeIndex> = Vec::new();
    for (current_idx, current_pair) in &nodes {
        let current_is_recover = matches!(graph[*current_idx].action, EventAction::RecoverNode(_));
        for (prev_idx, _prev_pair) in &seen {
            if rng.random::<f64>() >= config.dependency_density {
                continue;
            }
            if has_path_connecting(&graph, *current_idx, *prev_idx, None) {
                continue;
            }
            let edge = graph.add_edge(*prev_idx, *current_idx, ());
            if current_is_recover && config.recover_deps == RecoverDeps::Exempt {
                exempt_edges.push(edge);
            }
        }
        seen.push((*current_idx, *current_pair));
    }

    crate::simulator::util_stats::record_plan_deps_edges(exempt_edges.len() as u64);
    if !exempt_edges.is_empty() {
        graph = without_edges(&graph, &exempt_edges);
    }

    graph
}

/// The same nodes, in the same order, with every edge but `dropped` in the
/// order the source graph holds them. Node indices carry over unchanged.
fn without_edges(graph: &ExecutionPlan, dropped: &[EdgeIndex]) -> ExecutionPlan {
    let dropped: HashSet<EdgeIndex> = dropped.iter().copied().collect();
    let mut out: ExecutionPlan =
        DiGraph::with_capacity(graph.node_count(), graph.edge_count() - dropped.len());
    for idx in graph.node_indices() {
        out.add_node(graph[idx].clone());
    }
    for edge in graph.edge_references() {
        if !dropped.contains(&edge.id()) {
            out.add_edge(edge.source(), edge.target(), ());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Direction;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn config(post_fault_client_ops: i32) -> GeneratorConfig {
        GeneratorConfig {
            deployment: Arc::new(Deployment::test_cluster(3)),
            num_write_ops: 3,
            num_read_ops: 4,
            num_rmw_ops: 0,
            num_keys: 1,
            num_crashes: 2,
            num_partitions: 0,
            dependency_density: 0.0,
            max_concurrent_writes: Some(2),
            post_fault_client_ops,
            recover_deps: RecoverDeps::Stock,
        }
    }

    fn cell_config(
        dependency_density: f64,
        num_crashes: i32,
        max_concurrent_writes: Option<i32>,
        recover_deps: RecoverDeps,
    ) -> GeneratorConfig {
        GeneratorConfig {
            num_crashes,
            dependency_density,
            max_concurrent_writes,
            recover_deps,
            ..config(1)
        }
    }

    fn plan(cfg: GeneratorConfig, seed: u64) -> ExecutionPlan {
        let mut rng = SmallRng::seed_from_u64(seed);
        generate_plan(cfg, &mut rng)
    }

    /// An action printed with node indices where it names nodes, the form the
    /// baseline digests were taken over.
    fn index_form(action: &EventAction) -> String {
        match action {
            EventAction::ClientRequest(op) => {
                let (name, dest, key) = match op {
                    ClientOpSpec::Write(d, k) => ("Write", d, k),
                    ClientOpSpec::Read(d, k) => ("Read", d, k),
                    ClientOpSpec::Rmw(d, k) => ("Rmw", d, k),
                };
                format!("ClientRequest({name}({}, {key:?}))", dest.expect("a destination").index)
            }
            EventAction::CrashNode(n) => format!("CrashNode({})", n.index),
            EventAction::RecoverNode(n) => format!("RecoverNode({})", n.index),
            other => format!("{other:?}"),
        }
    }

    /// A digest of the node list and the edge list in storage order, so two
    /// plans agree only when they are the same graph laid out the same way.
    fn fingerprint(plan: &ExecutionPlan) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |s: &str| {
            for b in s.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        for idx in plan.node_indices() {
            eat(&format!("{};", index_form(&plan[idx].action)));
        }
        for e in plan.raw_edges() {
            eat(&format!("{}->{};", e.source().index(), e.target().index()));
        }
        h
    }

    fn edges(plan: &ExecutionPlan) -> std::collections::BTreeSet<(usize, usize)> {
        plan.raw_edges()
            .iter()
            .map(|e| (e.source().index(), e.target().index()))
            .collect()
    }

    fn is_recover(plan: &ExecutionPlan, i: usize) -> bool {
        matches!(plan[NodeIndex::new(i)].action, EventAction::RecoverNode(_))
    }

    /// The generator's output before the cells existed, digested for eight
    /// seeds of each of three configurations. A stock plan has to stay this.
    const BASELINE_GENERAL: [u64; 8] = [
        0xb6203e06a1dc1c7b,
        0x5b8763bdd798e884,
        0x6e245f972e045827,
        0x9235bef08bc1bdd0,
        0xbdb8de1717bdbdc1,
        0x99a6384aec761aaa,
        0x439255093333f427,
        0x467c90ef8194cb25,
    ];
    const BASELINE_ZERO_DENSITY: [u64; 8] = [
        0xef77e86a21f12354,
        0xe49c0e0b617b7cdd,
        0x463ff1437f93cbdb,
        0xbef4b11f7f61c03d,
        0xfe22b8f8e7663681,
        0x8b58fe655944b767,
        0xc6239220afa9f0fe,
        0xe9724ba1d07816b5,
    ];
    const BASELINE_SINGLE_CRASH: [u64; 8] = [
        0x490f43b45fe6eb3e,
        0x89d5a7abeadf4131,
        0x9d354b7b4a1a92db,
        0x8e093d3215f24d6b,
        0xe42832d525bf04e1,
        0x4de8cb26d656bd57,
        0xb8217c51eb24017e,
        0x5b46babe3f2d1e2b,
    ];

    #[test]
    fn a_stock_plan_is_the_baseline_plan_for_the_same_seed() {
        for seed in 0..8u64 {
            let general = plan(cell_config(0.3, 2, Some(2), RecoverDeps::Stock), seed);
            assert_eq!(fingerprint(&general), BASELINE_GENERAL[seed as usize], "seed {seed}");
            let zero = plan(cell_config(0.0, 2, Some(2), RecoverDeps::Stock), seed);
            assert_eq!(fingerprint(&zero), BASELINE_ZERO_DENSITY[seed as usize], "seed {seed}");
            let single = plan(cell_config(0.5, 1, None, RecoverDeps::Stock), seed);
            assert_eq!(fingerprint(&single), BASELINE_SINGLE_CRASH[seed as usize], "seed {seed}");
        }
    }

    #[test]
    fn exempt_drops_exactly_the_probabilistic_edges_into_a_recover() {
        let mut dropped_somewhere = 0;
        for seed in 0..64u64 {
            let stock = plan(cell_config(0.3, 2, Some(2), RecoverDeps::Stock), seed);
            let exempt = plan(cell_config(0.3, 2, Some(2), RecoverDeps::Exempt), seed);
            let mandatory = plan(cell_config(0.0, 2, Some(2), RecoverDeps::Stock), seed);
            for i in stock.node_indices() {
                assert_eq!(stock[i], exempt[i], "seed {seed}: the events differ");
                assert_eq!(stock[i], mandatory[i], "seed {seed}: the events differ");
            }
            let s = edges(&stock);
            let e = edges(&exempt);
            let m = edges(&mandatory);
            assert!(m.is_subset(&s), "seed {seed}: stock lost a mandatory edge");
            assert!(m.is_subset(&e), "seed {seed}: exempt lost a mandatory edge");
            assert!(e.is_subset(&s), "seed {seed}: exempt has an edge stock lacks");
            let removed: std::collections::BTreeSet<(usize, usize)> =
                s.difference(&e).copied().collect();
            let want: std::collections::BTreeSet<(usize, usize)> = s
                .iter()
                .filter(|(_, t)| is_recover(&stock, *t))
                .filter(|edge| !m.contains(edge))
                .copied()
                .collect();
            assert_eq!(removed, want, "seed {seed}: the removed set is not the recover-target set");
            dropped_somewhere += removed.len();
            for (src, dst) in &e {
                if is_recover(&exempt, *dst) {
                    assert!(
                        matches!(exempt[NodeIndex::new(*src)].action, EventAction::CrashNode(_)),
                        "seed {seed}: a recover still waits on something other than its crash"
                    );
                }
            }
            assert!(!petgraph::algo::is_cyclic_directed(&exempt), "seed {seed}");
        }
        assert!(dropped_somewhere > 0, "no seed had an edge into a recover to drop");
    }

    #[test]
    fn generated_rings_use_only_groups_of_four_distinct_members() {
        let mut deployment = Deployment::test_cluster(8);
        let nodes = deployment.nodes.to_vec();
        deployment.groups = vec![
            crate::simulator::deploy::Group { paths: vec!["small".into()], role: nodes[0].role, members: nodes[..3].to_vec(), quorum: true },
            crate::simulator::deploy::Group { paths: vec!["large".into()], role: nodes[0].role, members: nodes[3..].to_vec(), quorum: false },
        ];
        let small = Deployment::test_cluster(3);
        let mut rings = 0;
        for seed in 0..400u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            if let Some(PartitionAction::MajoritiesRing { group }) = random_partition(&deployment, &mut rng) {
                assert_eq!(group, nodes[3..].to_vec(), "seed {seed}: the only eligible group");
                rings += 1;
            }
            let mut rng = SmallRng::seed_from_u64(seed);
            assert!(
                !matches!(random_partition(&small, &mut rng), Some(PartitionAction::MajoritiesRing { .. })),
                "seed {seed}: a three-node group is never a ring"
            );
        }
        assert!(rings > 0, "some seed drew a ring");
        assert!(ring_eligible(&nodes[..4]) && !ring_eligible(&nodes[..3]));
        assert!(!ring_eligible(&[nodes[0], nodes[1], nodes[2], nodes[0]]));
    }

    #[test]
    fn exempt_changes_nothing_at_zero_density() {
        for seed in 0..64u64 {
            let stock = plan(cell_config(0.0, 2, Some(2), RecoverDeps::Stock), seed);
            let exempt = plan(cell_config(0.0, 2, Some(2), RecoverDeps::Exempt), seed);
            assert_eq!(fingerprint(&stock), fingerprint(&exempt), "seed {seed}");
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
