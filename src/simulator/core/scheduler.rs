use crate::compiler::cfg::Program;
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::make_local_env;
use crate::simulator::core::exec::{exec, exec_sync_on_node};
use crate::simulator::core::state::{
    Continuation, Logger, NodeId, Record, Runnable, ScheduleResult, State, UpdatePolicy,
};
use crate::simulator::core::values::{Env, Value};
use crate::simulator::coverage::{GlobalState, LocalCoverage, VertexMap};
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::Topology;
use crate::simulator::path::TopologyInfo;
use log::warn;
use rand::Rng;

pub fn schedule_runnable<H: HashPolicy, L: Logger>(
    state: &mut State<H>,
    logger: &mut L,
    program: &Program,
    randomly_drop_msgs: bool,
    cut_tail_from_mid: bool,
    sever_all_but_mid: bool,
    partition_away_nodes: &[NodeId],
    randomly_delay_msgs: bool,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
    topology: &TopologyInfo,
    global_state: &GlobalState,
) -> Result<ScheduleResult<H>, RuntimeError> {
    let len = state.runnable_tasks.len();
    if len == 0 {
        return Ok(ScheduleResult::None); // Halt equivalent
    }

    let mut rng = rand::rng();

    // Select task index using either tournament selection or random
    let idx = if let Some(snapshot) = global_snapshot {
        if len > 1 {
            // Tournament selection with K=3
            const K: usize = 3;
            let mut best_idx = rng.random_range(0..len);
            let mut best_score = snapshot.novelty_score(state.runnable_tasks[best_idx].pc());

            for _ in 1..K.min(len) {
                let candidate_idx = rng.random_range(0..len);
                let score = snapshot.novelty_score(state.runnable_tasks[candidate_idx].pc());
                if score > best_score {
                    best_idx = candidate_idx;
                    best_score = score;
                }
            }
            best_idx
        } else {
            0
        }
    } else {
        rng.random_range(0..len)
    };

    let runnable = state.runnable_tasks.remove(idx);

    match runnable {
        Runnable::Crash { node_id } => {
            crash_node(state, node_id);
            Ok(ScheduleResult::Crash { node_id })
        }
        Runnable::Recover { node_id } => {
            recover_crashed_node(
                state,
                logger,
                program,
                topology,
                node_id,
                global_state,
                global_snapshot,
                local_coverage,
            )?;
            Ok(ScheduleResult::Recover { node_id })
        }
        Runnable::Timer(timer) => {
            // Drop timer if node is crashed
            if state.crash_info.currently_crashed.contains(&timer.node) {
                return Ok(ScheduleResult::None);
            }

            // Send a unit value to the timer's channel to signal completion
            if let Some(mut chan) = state.channels.get(&timer.channel).cloned() {
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    // There's a reader waiting - directly wake it
                    let mut r_node_env = state.nodes[reader.node.index].clone();
                    if let Err(e) = crate::simulator::core::eval::store(
                        &lhs,
                        Value::<H>::unit(),
                        &mut reader.env,
                        &mut r_node_env,
                    ) {
                        log::warn!("Store failed in timer completion: {}", e);
                    }
                    state.nodes[reader.node.index] = r_node_env;
                    state.runnable_tasks.push_back(Runnable::Record(reader));
                } else {
                    // No reader waiting - buffer the value
                    chan.buffer.push_back(Value::<H>::unit());
                }
                state.channels.insert(timer.channel, chan);
            }
            Ok(ScheduleResult::None)
        }
        mut other => {
            let (src_node, dest_node, x, policy) = match &other {
                Runnable::Record(r) => (r.origin_node, r.node, r.x, &r.policy),
                Runnable::ChannelSend {
                    origin_node,
                    target,
                    x,
                    policy,
                    ..
                } => (*origin_node, *target, *x, policy),
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

            // Network faults (drops / partitions)
            let is_remote = src_node != dest_node;
            if is_remote {
                let mut should_deliver = true;
                if randomly_drop_msgs && rng.random::<f64>() < 0.3 {
                    should_deliver = false;
                }

                if cut_tail_from_mid
                    && ((src_node.index == 2 && dest_node.index == 1)
                        || (dest_node.index == 2 && src_node.index == 1))
                {
                    should_deliver = false;
                }

                if sever_all_but_mid {
                    if dest_node.index == 2 && src_node.index != 1 {
                        should_deliver = false;
                    } else if src_node.index == 2 && dest_node.index != 1 {
                        should_deliver = false;
                    }
                }

                if partition_away_nodes.contains(&src_node)
                    || partition_away_nodes.contains(&dest_node)
                {
                    should_deliver = false;
                }

                if !should_deliver {
                    return Ok(ScheduleResult::None);
                }

                // Latency / delay
                if randomly_delay_msgs && rng.random::<f64>() < x {
                    let new_x = policy.update(x);
                    match &mut other {
                        Runnable::Record(r) => r.x = new_x,
                        Runnable::ChannelSend { x, .. } => *x = new_x,
                        _ => unreachable!(),
                    }
                    state.runnable_tasks.push_back(other);
                    return Ok(ScheduleResult::None);
                }
            }

            match other {
                Runnable::Record(r) => {
                    let result = exec(state, logger, program, r, global_snapshot, local_coverage)?;
                    match result {
                        Some(client_op) => Ok(ScheduleResult::ClientOp(client_op)),
                        None => Ok(ScheduleResult::None),
                    }
                }
                Runnable::ChannelSend {
                    channel, message, ..
                } => {
                    if let Some(mut chan) = state.channels.get(&channel).cloned() {
                        if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                            // Wakeup reader
                            let mut r_node_env = state.nodes[reader.node.index].clone();
                            if let Err(e) = crate::simulator::core::eval::store(
                                &lhs,
                                message,
                                &mut reader.env,
                                &mut r_node_env,
                            ) {
                                log::warn!("Store failed in remote channel delivery: {}", e);
                            }
                            state.nodes[reader.node.index] = r_node_env;
                            state.runnable_tasks.push_back(Runnable::Record(reader));
                        } else {
                            // Unbounded buffer
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

    let tasks = std::mem::take(&mut state.runnable_tasks);
    for task in tasks {
        match task {
            Runnable::Timer(timer) => {
                // Drop timers for crashed nodes
                if timer.node != node_id {
                    state.runnable_tasks.push_back(Runnable::Timer(timer));
                }
            }
            Runnable::Record(record) => {
                if record.node == node_id {
                    let is_external = record.origin_node != record.node;
                    let origin_alive = !state
                        .crash_info
                        .currently_crashed
                        .contains(&record.origin_node);
                    if is_external && origin_alive {
                        let mut record = record;
                        record.reset();
                        state
                            .crash_info
                            .queued_messages
                            .push_back((node_id, record));
                    }
                } else {
                    state.runnable_tasks.push_back(Runnable::Record(record));
                }
            }
            Runnable::ChannelSend { target, .. } => {
                // If target is the crashed node, drop it.
                // If target is another node, keep it.
                if target != node_id {
                    state.runnable_tasks.push_back(task);
                }
            }
            // Keep other crash/recover runnables for different nodes
            Runnable::Crash { node_id: nid } | Runnable::Recover { node_id: nid } => {
                if nid != node_id {
                    state.runnable_tasks.push_back(task);
                }
            }
        }
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
    )?;

    let queued = std::mem::take(&mut state.crash_info.queued_messages);
    for (dest, record) in queued {
        if dest == node_id {
            state.runnable_tasks.push_back(Runnable::Record(record));
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

    let record = Record {
        pc: recover_fn.entry,
        node: node_id,
        origin_node: node_id,
        continuation: Continuation::Recover,
        entry_pc: recover_fn.entry,
        initial_env: env.clone(),
        env,
        x: 0.0,
        policy: UpdatePolicy::Identity,
    };

    exec(state, logger, prog, record, global_snapshot, local_coverage)?;
    Ok(())
}
