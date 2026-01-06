use crate::compiler::cfg::Program;
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::exec::exec;
use crate::simulator::core::state::{ClientOpResult, Logger, Runnable, State};
use crate::simulator::core::values::Value;
use crate::simulator::coverage::{LocalCoverage, VertexMap};
use rand::Rng;

pub fn schedule_runnable<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    randomly_drop_msgs: bool,
    cut_tail_from_mid: bool,
    sever_all_but_mid: bool,
    partition_away_nodes: &[usize],
    randomly_delay_msgs: bool,
    global_snapshot: Option<&VertexMap>,
    local_coverage: &mut LocalCoverage,
) -> Result<Option<ClientOpResult>, RuntimeError> {
    let len = state.runnable_tasks.len();
    if len == 0 {
        return Ok(None); // Halt equivalent
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
        Runnable::Timer(timer) => {
            // Drop timer if node is crashed
            if state.crash_info.currently_crashed.contains(&timer.node) {
                return Ok(None);
            }

            // Send a unit value to the timer's channel to signal completion
            if let Some(mut chan) = state.channels.get(&timer.channel).cloned() {
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    // There's a reader waiting - directly wake it
                    let mut r_node_env = state.nodes[reader.node].clone();
                    if let Err(e) = crate::simulator::core::eval::store(
                        &lhs,
                        Value::unit(),
                        &mut reader.env,
                        &mut r_node_env,
                    ) {
                        log::warn!("Store failed in timer completion: {}", e);
                    }
                    state.nodes[reader.node] = r_node_env;
                    state.runnable_tasks.push_back(Runnable::Record(reader));
                } else {
                    // No reader waiting - buffer the value
                    chan.buffer.push_back(Value::unit());
                }
                state.channels.insert(timer.channel, chan);
            }
            Ok(None)
        }
        Runnable::Record(mut r) => {
            let src_node = r.origin_node;
            let dest_node = r.node;

            if src_node != dest_node {
                if state.crash_info.currently_crashed.contains(&dest_node) {
                    // Queue message
                    state.crash_info.queued_messages.push_back((dest_node, r));
                    return Ok(None);
                } else {
                    let mut should_execute = true;
                    if randomly_drop_msgs && rng.random::<f64>() < 0.3 {
                        should_execute = false;
                    }

                    if cut_tail_from_mid
                        && ((src_node == 2 && dest_node == 1) || (dest_node == 2 && src_node == 1))
                    {
                        should_execute = false;
                    }

                    if sever_all_but_mid {
                        if dest_node == 2 && src_node != 1 {
                            should_execute = false;
                        } else if src_node == 2 && dest_node != 1 {
                            should_execute = false;
                        }
                    }

                    if partition_away_nodes.contains(&src_node)
                        || partition_away_nodes.contains(&dest_node)
                    {
                        should_execute = false;
                    }

                    if randomly_delay_msgs && rng.random::<f64>() < r.x {
                        r.x = r.policy.update(r.x);
                        state.runnable_tasks.push_back(Runnable::Record(r));
                        return Ok(None);
                    }

                    if should_execute {
                        let result =
                            exec(state, logger, program, r, global_snapshot, local_coverage)?;
                        return Ok(result);
                    }
                }
            } else {
                if state.crash_info.currently_crashed.contains(&r.node) {
                    return Ok(None);
                }
                let result = exec(state, logger, program, r, global_snapshot, local_coverage)?;
                return Ok(result);
            }
            Ok(None)
        }
    }
}
