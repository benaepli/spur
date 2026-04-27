use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::make_local_env;
use crate::simulator::core::exec::{exec, exec_sync_on_node};
use crate::simulator::core::partition::{activate_partition, heal_partition};
use crate::simulator::core::queue_selector::{
    QueueInfo, QueueSelection, QueueSelector, WithinQueueSelector,
};
use crate::simulator::core::state::{
    Continuation, Logger, NodeId, PurgatoryConfig, Record, Runnable, RunnableCategory,
    SchedulePolicy, ScheduleResult, State,
};
use crate::simulator::core::values::{Env, Value};
use crate::simulator::coverage::{GlobalState, LocalCoverage, VertexMap};
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::Topology;
use crate::simulator::path::TopologyInfo;
use imbl::{OrdSet, Vector};
use log::warn;
use rand::Rng;

/// A resolved deliver reservation. Runnables matching this are excluded from scheduling
/// until the deliver's DAG dependencies are met.
#[derive(Debug, Clone)]
pub struct Reservation {
    pub entry_pc: Vertex,
    pub from: Option<usize>,
    pub to: Option<usize>,
}

impl Reservation {
    pub fn matches<H: HashPolicy>(&self, runnable: &Runnable<H>) -> bool {
        match runnable {
            Runnable::Record(r) => {
                r.entry_pc == self.entry_pc
                    && self.to.map_or(true, |t| r.node.index == t)
                    && self.from.map_or(true, |f| r.origin_node.index == f)
            }
            // ChannelSend runnables are not matchable by delivers.
            // All VR inter-node messages are RPCs (Record runnables).
            _ => false,
        }
    }
}

/// Score a runnable in [0, 1] by combining novelty and priority. For Recover
/// events targeting a currently-crashed node, `quick_fire_multiplier` increases
/// the weight of priority relative to novelty while keeping the result in [0, 1].
fn score_runnable<H: HashPolicy>(
    r: &Runnable<H>,
    global_snapshot: Option<&VertexMap>,
    currently_crashed: &OrdSet<NodeId>,
    quick_fire_multiplier: f64,
) -> f64 {
    let novelty = global_snapshot.map_or(1.0, |s| s.novelty_score(r.pc()));
    let priority = r.priority();
    let is_quick_fire =
        matches!(r, Runnable::Recover { node_id, .. } if currently_crashed.contains(node_id));
    if is_quick_fire {
        let w = 0.75 * quick_fire_multiplier;
        (0.25 * novelty + w * priority) / (0.25 + w)
    } else {
        0.25 * novelty + 0.75 * priority
    }
}

/// Select an eligible item from a single queue.
///
/// `Tournament` samples `k` indices uniformly and takes the highest-scoring
/// (near-greedy for typical k). `Proportional` uses Efraimidis-Spirakis weighted
/// reservoir sampling with weight `score^exponent`, giving exact proportional
/// selection in a single O(eligible) pass.
fn select_within_queue<H: HashPolicy>(
    queue: &Vector<Runnable<H>>,
    eligible: &[usize],
    global_snapshot: Option<&VertexMap>,
    currently_crashed: &OrdSet<NodeId>,
    quick_fire_multiplier: f64,
    selector: &WithinQueueSelector,
    rng: &mut impl Rng,
) -> usize {
    if eligible.len() <= 1 {
        return eligible[0];
    }

    match selector {
        WithinQueueSelector::Tournament { k } => {
            let k = (*k).max(1);
            let mut best_idx = eligible[rng.random_range(0..eligible.len())];
            let mut best_score = score_runnable(
                &queue[best_idx],
                global_snapshot,
                currently_crashed,
                quick_fire_multiplier,
            );
            for _ in 1..k.min(eligible.len()) {
                let i = eligible[rng.random_range(0..eligible.len())];
                let s = score_runnable(
                    &queue[i],
                    global_snapshot,
                    currently_crashed,
                    quick_fire_multiplier,
                );
                if s > best_score {
                    best_idx = i;
                    best_score = s;
                }
            }
            best_idx
        }
        WithinQueueSelector::Proportional { exponent } => {
            // Efraimidis-Spirakis: argmax of (ln(u_i) / w_i) is exact weighted
            // sampling proportional to w_i. Both ln(u) (u in (0,1)) and w are
            // negative/positive respectively, so the largest key wins.
            //
            // Floor weight to keep zero-score items reachable; without this,
            // a score of exactly 0 would have 0 selection probability and a
            // score of 0 with exponent 0 would produce 0/0.
            let mut best_idx = eligible[0];
            let mut best_key = f64::NEG_INFINITY;
            for &i in eligible {
                let s = score_runnable(
                    &queue[i],
                    global_snapshot,
                    currently_crashed,
                    quick_fire_multiplier,
                );
                let weight = s.powf(*exponent).max(1e-9);
                let u: f64 = rng.random();
                // u is in (0, 1); ln(u) is negative; key = ln(u) / weight is negative.
                // Higher weight → key closer to 0 (larger), so argmax is correct.
                let key = u.ln() / weight;
                if key > best_key {
                    best_key = key;
                    best_idx = i;
                }
            }
            best_idx
        }
    }
}

pub fn schedule_runnable<H: HashPolicy, L: Logger, Q: QueueSelector>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    randomly_drop_msgs: bool,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
    topology: &TopologyInfo,
    global_state: &GlobalState,
    policy: &SchedulePolicy,
    strict_timers: bool,
    selector: &mut Q,
    within_queue: &WithinQueueSelector,
    quick_fire_multiplier: f64,
    purgatory_config: &PurgatoryConfig,
    reservations: &[Reservation],
) -> Result<ScheduleResult<H>, RuntimeError> {
    if state.all_queues_empty() {
        return Ok(ScheduleResult::None);
    }

    let mut rng = rand::rng();

    // Helper: check if a runnable is reserved
    let is_reserved = |r: &Runnable<H>| reservations.iter().any(|res| res.matches(r));

    // Build QueueInfo, accounting for strict_timers eligibility AND reservations.
    // Subtract reserved items so the QueueSelector doesn't route to queues
    // where all items are reserved (wastes iterations in fully-constrained plans).
    let timer_queue_size = if strict_timers {
        state
            .timer_queue
            .iter()
            .filter(|r| {
                if is_reserved(r) {
                    return false;
                }
                if let Runnable::Timer(t) = r {
                    t.label.as_ref().map_or(true, |l| {
                        state.allowed_timers.contains(&(t.node.index, l.clone()))
                    })
                } else {
                    true
                }
            })
            .count()
    } else {
        state.timer_queue.iter().filter(|r| !is_reserved(r)).count()
    };

    let info = QueueInfo {
        local_queue_sizes: state
            .local_queues
            .iter()
            .map(|q| q.iter().filter(|r| !is_reserved(r)).count())
            .collect(),
        network_queue_size: state.network_queue.iter().filter(|r| !is_reserved(r)).count(),
        timer_queue_size,
        step: state.crash_info.current_step,
    };

    let selection = match selector.select(&info, &mut rng) {
        Some(s) => s,
        None => return Ok(ScheduleResult::None),
    };

    let runnable = match selection {
        QueueSelection::Local(node_idx) => {
            let queue = &state.local_queues[node_idx];
            let eligible: Vec<usize> = (0..queue.len())
                .filter(|&i| !is_reserved(&queue[i]))
                .collect();
            if eligible.is_empty() {
                return Ok(ScheduleResult::None);
            }
            let idx = select_within_queue(
                queue,
                &eligible,
                global_snapshot,
                &state.crash_info.currently_crashed,
                quick_fire_multiplier,
                within_queue,
                &mut rng,
            );
            state.local_queues[node_idx].remove(idx)
        }
        QueueSelection::Network => {
            let queue = &state.network_queue;
            let eligible: Vec<usize> = (0..queue.len())
                .filter(|&i| !is_reserved(&queue[i]))
                .collect();
            if eligible.is_empty() {
                return Ok(ScheduleResult::None);
            }
            let idx = select_within_queue(
                queue,
                &eligible,
                global_snapshot,
                &state.crash_info.currently_crashed,
                quick_fire_multiplier,
                within_queue,
                &mut rng,
            );
            state.network_queue.remove(idx)
        }
        QueueSelection::Timer => {
            let queue = &state.timer_queue;
            let eligible: Vec<usize> = if strict_timers {
                (0..queue.len())
                    .filter(|&i| {
                        if is_reserved(&queue[i]) {
                            return false;
                        }
                        if let Runnable::Timer(t) = &queue[i] {
                            t.label.as_ref().map_or(true, |l| {
                                state.allowed_timers.contains(&(t.node.index, l.clone()))
                            })
                        } else {
                            true
                        }
                    })
                    .collect()
            } else {
                (0..queue.len())
                    .filter(|&i| !is_reserved(&queue[i]))
                    .collect()
            };
            if eligible.is_empty() {
                return Ok(ScheduleResult::None);
            }
            let idx = select_within_queue(
                queue,
                &eligible,
                global_snapshot,
                &state.crash_info.currently_crashed,
                quick_fire_multiplier,
                within_queue,
                &mut rng,
            );
            state.timer_queue.remove(idx)
        }
    };

    match runnable {
        Runnable::Crash { node_id, .. } => {
            crash_node(state, node_id);
            Ok(ScheduleResult::Crash { node_id })
        }
        Runnable::Recover { node_id, .. } => {
            recover_crashed_node(
                state,
                logger,
                program,
                topology,
                node_id,
                global_state,
                global_snapshot,
                local_coverage,
                policy,
                purgatory_config,
            )?;
            Ok(ScheduleResult::Recover { node_id })
        }
        Runnable::Partition { partition_type, .. } => {
            activate_partition(state, partition_type.clone());
            Ok(ScheduleResult::Partition { partition_type })
        }
        Runnable::Heal { .. } => {
            heal_partition(state);
            Ok(ScheduleResult::Heal)
        }
        Runnable::Timer(timer) => {
            if state.crash_info.currently_crashed.contains(&timer.node) {
                return Ok(ScheduleResult::None);
            }

            if let Some(mut chan) = state.channels.get(&timer.channel).cloned() {
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    let mut r_node_env = state.nodes[reader.node.index].clone();
                    if let Err(e) = crate::simulator::core::eval::store(
                        &lhs,
                        Value::<H>::unit(),
                        &mut reader.env,
                        &mut r_node_env,
                    ) {
                        log::warn!("Store failed in timer completion: {}", e);
                    }
                    let node_index = reader.node.index;
                    state.nodes[node_index] = r_node_env;
                    state.push_to_local(node_index, Runnable::Record(reader));
                } else {
                    chan.buffer.push_back(Value::<H>::unit());
                }
                state.channels.insert(timer.channel, chan);
            }
            if let Some(label) = timer.label {
                state
                    .allowed_timers
                    .remove(&(timer.node.index, label.clone()));
                Ok(ScheduleResult::TimerFired {
                    node_id: timer.node,
                    label,
                })
            } else {
                Ok(ScheduleResult::None)
            }
        }
        other => {
            let (src_node, dest_node) = match &other {
                Runnable::Record(r) => (r.origin_node, r.node),
                Runnable::ChannelSend {
                    origin_node,
                    target,
                    ..
                } => (*origin_node, *target),
                _ => unreachable!(),
            };

            if state.crash_info.currently_crashed.contains(&dest_node) {
                if let Runnable::Record(r) = other {
                    if src_node != dest_node {
                        let mut r = r;
                        r.reset();
                        state.crash_info.queued_messages.push_back((dest_node, r));
                    }
                }
                return Ok(ScheduleResult::None);
            }

            if state.partition_info.is_blocked(src_node, dest_node) {
                match other {
                    Runnable::Record(r) => {
                        let mut r = r;
                        r.reset();
                        state.partition_info.buffer_record(dest_node, r);
                    }
                    Runnable::ChannelSend {
                        channel,
                        message,
                        origin_node,
                        pc,
                        priority,
                        ..
                    } => {
                        state.partition_info.buffer_channel_send(
                            dest_node,
                            channel,
                            message,
                            origin_node,
                            pc,
                            priority,
                        );
                    }
                    _ => unreachable!(),
                }
                return Ok(ScheduleResult::None);
            }

            let is_remote = src_node != dest_node;
            if is_remote && randomly_drop_msgs && rng.random::<f64>() < 0.3 {
                return Ok(ScheduleResult::None);
            }

            match other {
                Runnable::Record(r) => {
                    let record_entry_pc = r.entry_pc;
                    let record_origin = r.origin_node;
                    let record_dest = r.node;
                    let result = exec(
                        state,
                        logger,
                        program,
                        r,
                        global_snapshot,
                        local_coverage,
                        policy,
                        purgatory_config,
                    )?;
                    match result {
                        Some(client_op) => Ok(ScheduleResult::ClientOp(client_op)),
                        None => Ok(ScheduleResult::RecordExecuted {
                            entry_pc: record_entry_pc,
                            origin_node: record_origin,
                            dest_node: record_dest,
                        }),
                    }
                }
                Runnable::ChannelSend {
                    channel, message, ..
                } => {
                    if let Some(mut chan) = state.channels.get(&channel).cloned() {
                        if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                            let mut r_node_env = state.nodes[reader.node.index].clone();
                            if let Err(e) = crate::simulator::core::eval::store(
                                &lhs,
                                message,
                                &mut reader.env,
                                &mut r_node_env,
                            ) {
                                log::warn!("Store failed in remote channel delivery: {}", e);
                            }
                            let node_index = reader.node.index;
                            state.nodes[node_index] = r_node_env;
                            state.push_to_local(node_index, Runnable::Record(reader));
                        } else {
                            chan.buffer.push_back(message);
                        }
                        state.channels.insert(channel, chan);
                    }
                    Ok(ScheduleResult::None)
                }
                _ => unreachable!(),
            }
        }
    }
}

fn crash_node<H: HashPolicy>(state: &mut State<H>, node_id: NodeId) {
    if state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is already crashed", node_id);
        return;
    }
    state.crash_info.currently_crashed.insert(node_id);

    // 1. Process local queue for crashed node: save external records, drop the rest
    let local = std::mem::take(&mut state.local_queues[node_id.index]);
    for task in local {
        if let Runnable::Record(record) = task {
            if record.origin_node != record.node {
                let mut record = record;
                record.reset();
                state
                    .crash_info
                    .queued_messages
                    .push_back((node_id, record));
            }
        }
    }

    // 2. Filter network queue: remove items targeting the crashed node
    let net = std::mem::take(&mut state.network_queue);
    for task in net {
        match &task {
            Runnable::Record(r) if r.node == node_id => {
                if r.origin_node != r.node {
                    let mut r = r.clone();
                    r.reset();
                    state.crash_info.queued_messages.push_back((node_id, r));
                }
            }
            Runnable::ChannelSend { target, .. } if *target == node_id => {}
            Runnable::Crash { node_id: nid, .. } | Runnable::Recover { node_id: nid, .. }
                if *nid == node_id => {}
            _ => state.network_queue.push_back(task),
        }
    }

    // 3. Filter timer queue: remove timers for the crashed node
    let timers = std::mem::take(&mut state.timer_queue);
    for task in timers {
        if let Runnable::Timer(ref t) = task {
            if t.node == node_id {
                continue;
            }
        }
        state.timer_queue.push_back(task);
    }
}

fn recover_crashed_node<H: HashPolicy, L: Logger>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    topology: &TopologyInfo,
    node_id: NodeId,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
) -> Result<(), RuntimeError> {
    if !state.crash_info.currently_crashed.contains(&node_id) {
        warn!("Node {} is not crashed", node_id);
        return Ok(());
    }
    state.crash_info.currently_crashed.remove(&node_id);

    state.nodes[node_id.index] = Env::<H>::default();
    reinit_node(
        topology,
        state,
        logger,
        program,
        node_id,
        global_state,
        global_snapshot,
        local_coverage,
        policy,
        purgatory_config,
    )?;

    let queued = std::mem::take(&mut state.crash_info.queued_messages);
    for (dest, record) in queued {
        if dest == node_id {
            state.push_runnable(Runnable::Record(record));
        } else {
            state.crash_info.queued_messages.push_back((dest, record));
        }
    }
    Ok(())
}

fn reinit_node<H: HashPolicy, L: Logger>(
    topology: &TopologyInfo,
    state: &mut State<H>,
    logger: &mut L,
    prog: &Program,
    node_id: NodeId,
    global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
) -> Result<(), RuntimeError> {
    use crate::compiler::cfg::{SELF_SLOT, VarSlot};

    let init_fn = prog
        .get_func_by_name("Node.BASE_NODE_INIT")
        .ok_or_else(|| RuntimeError::MissingRequiredFunction("Node.BASE_NODE_INIT".to_string()))?;

    if let VarSlot::Node(self_idx, _) = SELF_SLOT {
        state.nodes[node_id.index].set(self_idx, Value::<H>::node(node_id));
    }

    let node_env = &state.nodes[node_id.index];
    let mut env = make_local_env(
        init_fn,
        vec![],
        &Env::<H>::default(),
        node_env,
        &prog.id_to_name,
    );

    exec_sync_on_node(
        state,
        logger,
        prog,
        &mut env,
        node_id,
        init_fn.entry,
        global_snapshot,
        local_coverage,
        policy,
        purgatory_config,
    )?;

    recover_node(
        topology,
        state,
        logger,
        prog,
        node_id,
        global_state,
        global_snapshot,
        local_coverage,
        policy,
        purgatory_config,
    )
}

fn recover_node<H: HashPolicy, L: Logger>(
    topology: &TopologyInfo,
    state: &mut State<H>,
    logger: &mut L,
    prog: &Program,
    node_id: NodeId,
    _global_state: &GlobalState,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
) -> Result<(), RuntimeError> {
    let Some(recover_fn) = prog.get_func_by_name("Node.RecoverInit") else {
        return Ok(());
    };

    let actuals = match topology.topology {
        Topology::Full => vec![
            Value::<H>::int(node_id.index as i64),
            Value::<H>::list(
                (0..topology.num_servers)
                    .map(|j| {
                        Value::<H>::node(NodeId {
                            role: node_id.role,
                            index: j as usize,
                        })
                    })
                    .collect(),
            ),
        ],
    };

    let node_env = &state.nodes[node_id.index];
    let env = make_local_env(
        recover_fn,
        actuals,
        &Env::<H>::default(),
        node_env,
        &prog.id_to_name,
    );

    let mut rng = rand::rng();
    let record = Record {
        pc: recover_fn.entry,
        node: node_id,
        origin_node: node_id,
        continuation: Continuation::Recover,
        entry_pc: recover_fn.entry,
        initial_env: env.clone(),
        env,
        priority: policy.sample(&mut rng, RunnableCategory::Record),
        causal_operation_id: None,
        trace_id: None,
    };

    exec(
        state,
        logger,
        prog,
        record,
        global_snapshot,
        local_coverage,
        policy,
        purgatory_config,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::hash_utils::NoHashing;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn heal(priority: f64) -> Runnable<NoHashing> {
        Runnable::Heal { priority }
    }

    /// Score under default `score_runnable` parameters (no novelty signal,
    /// no quick-fire boost) is `0.25 + 0.75 * priority`.
    fn expected_score(priority: f64) -> f64 {
        0.25 + 0.75 * priority
    }

    #[test]
    fn proportional_selection_matches_expected_distribution() {
        let queue: Vector<Runnable<NoHashing>> = vec![
            heal(0.0), // score 0.25
            heal(0.5), // score 0.625
            heal(1.0), // score 1.00
        ]
        .into();
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let crashed = OrdSet::new();
        let selector = WithinQueueSelector::Proportional { exponent: 1.0 };

        let mut rng = StdRng::seed_from_u64(0xdeadbeef);
        let trials = 50_000usize;
        let mut counts = [0usize; 3];
        for _ in 0..trials {
            let idx = select_within_queue(
                &queue,
                &eligible,
                None,
                &crashed,
                1.0,
                &selector,
                &mut rng,
            );
            counts[idx] += 1;
        }

        let total_score: f64 = (0..3)
            .map(|i| expected_score(queue[i].priority()))
            .sum();
        for i in 0..3 {
            let expected = expected_score(queue[i].priority()) / total_score;
            let observed = counts[i] as f64 / trials as f64;
            // Binomial std error ≈ sqrt(p(1-p)/n); with n=50k and p ~0.5 that's
            // ~0.0022. Allow 0.015 (≈7σ) to keep the test robust.
            assert!(
                (observed - expected).abs() < 0.015,
                "bucket {}: expected ~{:.3}, observed {:.3} (n={})",
                i,
                expected,
                observed,
                trials,
            );
        }
    }

    #[test]
    fn proportional_with_zero_exponent_is_uniform() {
        let queue: Vector<Runnable<NoHashing>> = vec![heal(0.0), heal(0.5), heal(1.0)].into();
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let crashed = OrdSet::new();
        let selector = WithinQueueSelector::Proportional { exponent: 0.0 };

        let mut rng = StdRng::seed_from_u64(42);
        let trials = 30_000usize;
        let mut counts = [0usize; 3];
        for _ in 0..trials {
            let idx = select_within_queue(
                &queue,
                &eligible,
                None,
                &crashed,
                1.0,
                &selector,
                &mut rng,
            );
            counts[idx] += 1;
        }
        for i in 0..3 {
            let observed = counts[i] as f64 / trials as f64;
            assert!(
                (observed - 1.0 / 3.0).abs() < 0.02,
                "bucket {} should be ~uniform 0.333, got {:.3}",
                i,
                observed,
            );
        }
    }

    #[test]
    fn tournament_default_preserves_existing_behavior() {
        // Default selector is Tournament { k: 10 }. With sampling-with-replacement,
        // the top-scoring item should dominate but not deterministically.
        let queue: Vector<Runnable<NoHashing>> = vec![heal(0.1), heal(0.9)].into();
        let eligible: Vec<usize> = (0..queue.len()).collect();
        let crashed = OrdSet::new();
        let selector = WithinQueueSelector::default();
        assert!(matches!(selector, WithinQueueSelector::Tournament { k: 10 }));

        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = [0usize; 2];
        for _ in 0..4_000 {
            let idx = select_within_queue(
                &queue,
                &eligible,
                None,
                &crashed,
                1.0,
                &selector,
                &mut rng,
            );
            counts[idx] += 1;
        }
        // P(top wins) = 1 - (1/2)^k.min(2) = 0.75. Allow a wide margin.
        assert!(
            counts[1] > counts[0] * 2,
            "tournament should favor higher-score index 1: got {:?}",
            counts
        );
    }

    #[test]
    fn select_within_queue_handles_singleton() {
        let queue: Vector<Runnable<NoHashing>> = vec![heal(0.5)].into();
        let eligible = vec![0];
        let crashed = OrdSet::new();
        let mut rng = StdRng::seed_from_u64(1);

        let tournament = WithinQueueSelector::Tournament { k: 10 };
        let proportional = WithinQueueSelector::Proportional { exponent: 1.0 };

        for selector in [&tournament, &proportional] {
            let idx = select_within_queue(
                &queue,
                &eligible,
                None,
                &crashed,
                1.0,
                selector,
                &mut rng,
            );
            assert_eq!(idx, 0);
        }
    }
}
