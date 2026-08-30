use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::steer_terms::ResolvedTerms;
use crate::simulator::core::{
    Continuation, Env, LogEntry, Logger, NodeId, OpKind, Operation, PurgatoryConfig,
    QueuePolicyConfig, Record, Reservation, Runnable, RunnableCategory,
    RuntimeError, SchedulePolicy, ScheduleResult, State, TraceEntry, Value, WithinQueueSelector,
    make_local_env, schedule_runnable,
};
use crate::simulator::coverage::{FaultCoverage, GlobalState};
use crate::simulator::feedback::Feedback;
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::plan::{
    ClientOpSpec, DeliverSpec, EventAction, ExecutionPlan, PlanEngine, PlannedEvent,
};
use crate::simulator::rng::{Stream, StreamRng};
use crate::simulator::util_stats::{self, DeliveryBias, RunEnd, RunExtension, RunTermination};
use ecow::EcoString;
use log::{info, warn};
use petgraph::graph::NodeIndex;
use rand::Rng;
use std::collections::{BTreeSet, HashMap, HashSet};

pub mod generator;
pub mod plan;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Full,
}

#[derive(Clone, Debug)]
pub struct TopologyInfo {
    pub topology: Topology,
    pub num_servers: i32,
}

/// Newtype wrapper for log and trace entries that implements Logger.
#[derive(Debug, Default)]
pub struct Logs {
    pub entries: Vec<LogEntry>,
    pub traces: Vec<TraceEntry>,
}

impl Logger for Logs {
    fn log(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }
    fn log_trace(&mut self, entry: TraceEntry) {
        self.traces.push(entry);
    }
}

/// A pool that dynamically creates client nodes on demand and recycles them.
#[derive(Debug)]
pub struct ClientPool {
    free_clients: Vec<NodeId>,
    client_role: NameId,
    node_slot_count: usize,
}

impl ClientPool {
    pub fn new(client_role: NameId, node_slot_count: usize) -> Self {
        Self {
            free_clients: Vec::new(),
            client_role,
            node_slot_count,
        }
    }

    /// Get a client node: reuses a free one or creates a new one.
    /// Returns (NodeId, bool) where the boolean is true if the node was newly created.
    pub fn get<H: HashPolicy>(&mut self, state: &mut State<H>) -> (NodeId, bool) {
        if let Some(node_id) = self.free_clients.pop() {
            (node_id, false)
        } else {
            (state.add_node(self.client_role, self.node_slot_count), true)
        }
    }

    /// Return a client node to the pool for reuse.
    pub fn release(&mut self, node_id: NodeId) {
        self.free_clients.push(node_id);
    }
}

/// Wrapper around State that adds path-execution tracking fields.
#[derive(Debug)]
pub struct PathState<H: HashPolicy, F: Feedback> {
    pub state: State<H>,
    pub feedback: F::Local,
    pub logs: Logs,
    pub history: Vec<Operation<H>>,
    pub client_pool: ClientPool,
}

impl<H: HashPolicy, F: Feedback> PathState<H, F> {
    pub fn new(
        role_node_counts: &[(NameId, usize)],
        node_slot_count: usize,
        client_role: NameId,
    ) -> Self {
        Self {
            state: State::<H>::new(role_node_counts, node_slot_count),
            feedback: F::Local::default(),
            logs: Logs::default(),
            history: Vec::new(),
            client_pool: ClientPool::new(client_role, node_slot_count),
        }
    }
}

fn schedule_client_op<H: HashPolicy>(
    state: &mut State<H>,
    history: &mut Vec<Operation<H>>,
    prog: &Program,
    op_id: i32,
    op_spec: &ClientOpSpec,
    client_node_id: NodeId,
    server_role: NameId,
    policy: &SchedulePolicy,
    rng: &mut impl StreamRng,
) -> Result<(), RuntimeError> {
    let client_id = client_node_id.index as i32;
    let (op_name, actuals) = match op_spec {
        ClientOpSpec::Write(target, key) => (
            "ClientInterface.Write",
            vec![
                Value::<H>::node(NodeId {
                    role: server_role,
                    index: *target as usize,
                }),
                Value::<H>::string(EcoString::from(key.as_str())),
                Value::<H>::int(op_id as i64),
            ],
        ),
        ClientOpSpec::Read(target, key) => (
            "ClientInterface.Read",
            vec![
                Value::<H>::node(NodeId {
                    role: server_role,
                    index: *target as usize,
                }),
                Value::<H>::string(EcoString::from(key.as_str())),
            ],
        ),
        ClientOpSpec::Rmw(target, key) => (
            "ClientInterface.RMW",
            vec![
                Value::<H>::node(NodeId {
                    role: server_role,
                    index: *target as usize,
                }),
                Value::<H>::string(EcoString::from(key.as_str())),
                Value::<H>::int(op_id as i64),
            ],
        ),
    };

    let op_func = prog
        .get_func_by_name(op_name)
        .ok_or_else(|| RuntimeError::MissingRequiredFunction(op_name.to_string()))?;
    let env = make_local_env(
        op_func,
        actuals.clone(),
        &Env::default(),
        &state.nodes[client_node_id.index],
        &prog.id_to_name,
    );

    history.push(Operation {
        client_id,
        op_action: op_name.to_string(),
        kind: OpKind::Invocation,
        payload: actuals,
        unique_id: op_id,
        step: state.crash_info.current_step,
    });

    let send_ordinal = state.next_send_ordinal(client_node_id);
    let receiver_token_at_send = state.node_state_token(client_node_id);
    state.push_runnable(Runnable::Record(Record {
        pc: op_func.entry,
        node: client_node_id,
        origin_node: client_node_id,
        continuation: Continuation::ClientOp {
            client_id,
            op_name: op_name.to_string(),
            unique_id: op_id,
        },
        entry_pc: op_func.entry,
        initial_env: env.clone(),
        env,
        priority: policy.sample(rng, RunnableCategory::Record),
        causal_operation_id: Some(op_id),
        trace_id: None,
        link_seq: None,
        origin_incarnation: state.incarnation(client_node_id),
        bias: DeliveryBias::NONE,
        timer_entry: None,
        send_ordinal,
        receiver_token_at_send,
    }));
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Completed { steps: i32 },
    Deadlock { step: i32, pending_ops: usize },
    /// The step budget ran out with planned events still outstanding.
    IterationsExhausted { outstanding_events: usize },
}

/// How a run spent its steps: how many released a runnable, how many offered
/// queued work the scheduler released none of, and how many had nothing queued
/// at all. `tail_without_release` is the run of steps up to the current one
/// that released nothing.
#[derive(Default)]
struct StepCensus {
    released: u64,
    blocked: u64,
    idle: u64,
    tail_without_release: u64,
}

impl StepCensus {
    fn released(&mut self) {
        self.released += 1;
        self.tail_without_release = 0;
    }

    fn blocked(&mut self) {
        self.blocked += 1;
        self.tail_without_release += 1;
    }

    fn idle(&mut self) {
        self.idle += 1;
        self.tail_without_release += 1;
    }
}

/// Record why one plan execution stopped, together with the work that was
/// still queued at that moment. Observation only.
fn record_termination<H: HashPolicy>(
    end: RunEnd,
    state: &State<H>,
    engine: &PlanEngine,
    steps_used: i32,
    step_budget: i32,
    recovered_nodes: usize,
    census: &StepCensus,
) {
    if !util_stats::enabled() {
        return;
    }
    let pending = state.total_runnable_count() + state.purgatory.len();
    let steps_used = steps_used.max(0) as u64;
    util_stats::record_run_termination(&RunTermination {
        end,
        steps_used,
        step_budget: step_budget.max(0) as u64,
        pending_work_at_exit: pending as u64,
        planned_events_outstanding: engine.outstanding_count() as u64,
        recovered_nodes,
    });
    util_stats::record_run_extension(&RunExtension {
        end,
        steps: steps_used,
        steps_released: census.released,
        steps_blocked: census.blocked,
        steps_idle: census.idle,
        tail_without_release: census.tail_without_release,
        pending_at_exit: pending as u64,
        recovered_nodes,
    });
}

/// How a node looks as a place to put a fault, as a small integer.
///
/// Built only from how many times the node has restarted, whether it has a send
/// nobody has received yet, and how many of its peers are down, each clamped, so
/// the tag names a situation any message-passing specification can be in and has
/// tens of values rather than one per state.
fn fault_context_tag<H: HashPolicy>(state: &State<H>, node: NodeId) -> u16 {
    let restarts = state.incarnation(node).min(2) as u16;
    let unreceived_send = state
        .send_ledger
        .get(node.index)
        .is_some_and(|l| l.in_flight > 0) as u16;
    let peers_down = state
        .crash_info
        .currently_crashed
        .iter()
        .filter(|n| n.index != node.index)
        .count()
        .min(2) as u16;
    restarts * 6 + unreceived_send * 3 + peers_down
}

/// Which node a crash lands on, given the contexts faults have already been
/// placed in.
///
/// The candidates are the server nodes that are up and have no crash
/// outstanding; among those, the ones whose context the table has seen least
/// often, so a fault goes where this arm has damaged the system least. `fallback`
/// is used when nothing is eligible, which is what the workload asked for.
///
/// The tie is broken with the one draw a uniform choice among the candidates
/// would have taken, so the fault stream advances by the same amount either way
/// and the pick replays from it.
fn place_crash<H: HashPolicy>(
    state: &State<H>,
    coverage: &FaultCoverage,
    server_role: NameId,
    num_servers: usize,
    taken: &HashSet<usize>,
    fallback: NodeId,
    rng: &mut impl StreamRng,
) -> NodeId {
    let candidates: Vec<NodeId> = (0..num_servers)
        .filter(|i| {
            !taken.contains(i)
                && !state
                    .crash_info
                    .currently_crashed
                    .iter()
                    .any(|n| n.index == *i)
        })
        .map(|index| NodeId {
            role: server_role,
            index,
        })
        .collect();

    rng.use_stream(Stream::FaultPriority);
    let draw: usize = rng.random_range(0..candidates.len().max(1));

    if candidates.is_empty() {
        util_stats::record_fault_placement(0, false, 0, 0);
        return fallback;
    }

    let visits: Vec<u64> = candidates
        .iter()
        .map(|&c| coverage.visits(fault_context_tag(state, c)))
        .collect();
    let fewest = visits.iter().copied().min().unwrap_or(0);
    let least_visited: Vec<NodeId> = candidates
        .iter()
        .zip(&visits)
        .filter(|&(_, &v)| v == fewest)
        .map(|(&c, _)| c)
        .collect();

    let chosen = least_visited[draw % least_visited.len()];
    let uniform = candidates[draw % candidates.len()];
    let (distinct_tags, max_visits) = coverage.visit(fault_context_tag(state, chosen));
    util_stats::record_fault_placement(
        candidates.len(),
        chosen.index != uniform.index,
        distinct_tags,
        max_visits,
    );
    chosen
}

pub fn exec_plan<H: HashPolicy, F: Feedback>(
    path_state: &mut PathState<H, F>,
    program: Program,
    plan: ExecutionPlan,
    max_iterations: i32,
    topology: TopologyInfo,
    global_state: &GlobalState<F>,
    snapshot: &F::Snapshot,
    run_id: i64,
    policy: &SchedulePolicy,
    strict_timers: bool,
    queue_policy: &QueuePolicyConfig,
    within_queue: &WithinQueueSelector,
    terms: &ResolvedTerms,
    purgatory_config: &PurgatoryConfig,
    coverage_guided_fault_placement: bool,
    rng: &mut impl StreamRng,
) -> Result<RunOutcome, RuntimeError> {
    util_stats::begin_run();
    let mut selector = queue_policy.to_selector();
    let mut op_id_counter = 0i32;
    let mut in_progress: HashMap<i32, NodeIndex> = HashMap::new();
    // Map from node_id index to the plan engine NodeIndex for pending crash/recover events
    let mut pending_crash_recover: HashMap<usize, NodeIndex> = HashMap::new();
    // Map from (node_index, label) to the plan engine NodeIndex for pending AllowTimer events
    let mut pending_allow_timer: HashMap<(usize, String), NodeIndex> = HashMap::new();
    let mut pending_partition: Option<NodeIndex> = None;
    let mut pending_heal: Option<NodeIndex> = None;

    // Build name-to-entry-pc map for resolving deliver specs.
    // Resolution happens once upfront, not per scheduler call.
    let name_to_entry: HashMap<&str, Vertex> = program
        .func_name_to_id
        .iter()
        .filter_map(|(name, name_id)| {
            program.rpc.get(name_id).map(|fi| (name.as_str(), fi.entry))
        })
        .collect();

    // Reverse map for matching RecordExecuted results back to function names
    let entry_to_name: HashMap<Vertex, &str> = name_to_entry
        .iter()
        .map(|(&name, &entry)| (entry, name))
        .collect();

    // Collect all deliver events from the plan DAG before PlanEngine::new consumes it.
    let all_delivers: HashMap<NodeIndex, DeliverSpec> = plan
        .node_indices()
        .filter_map(|idx| match &plan[idx].action {
            EventAction::Deliver(spec) => Some((idx, spec.clone())),
            _ => None,
        })
        .collect();

    // Track deliver states: ready (unlocked) vs completed. `ready_delivers` is a
    // BTreeSet so the `.iter().find(...)` match at the deliver site is
    // deterministic across runs (required for replay).
    let mut ready_delivers: BTreeSet<NodeIndex> = BTreeSet::new();
    let mut completed_delivers: HashSet<NodeIndex> = HashSet::new();

    let mut engine = PlanEngine::new(plan);

    // Nodes observed crashing, and the subset that later recovered.
    let mut crashed_nodes: HashSet<usize> = HashSet::new();
    let mut recovered_nodes: HashSet<usize> = HashSet::new();

    // Where a chosen fault placement was actually put, so the restart follows
    // its crash. Keyed by the node the workload named, which the generator
    // serializes one crash/restart pair at a time. `taken` holds the nodes with
    // a crash outstanding, which no other placement may choose.
    let mut placement: HashMap<i32, NodeId> = HashMap::new();
    let mut taken: HashSet<usize> = HashSet::new();

    let mut census = StepCensus::default();

    // Starvation detection: track consecutive no-progress iterations
    let mut no_progress_count: i32 = 0;
    const STARVATION_WARN_THRESHOLD: i32 = 500;

    // Look up role NameIds from the program
    let server_role = program
        .roles
        .iter()
        .find(|(_, name)| name == "Node")
        .map(|(id, _)| *id)
        .ok_or_else(|| RuntimeError::RoleNotFound("Node".to_string()))?;

    let validate_node = |state: &State<H>,
                         index: usize,
                         expected_role: NameId,
                         _role_name: &str|
     -> Result<NodeId, RuntimeError> {
        if index >= state.nodes.len() {
            return Err(RuntimeError::IndexOutOfBounds {
                index,
                len: state.nodes.len(),
            });
        }
        let node_val = state.nodes[index].get(0);
        let node_id = node_val.as_node()?;
        if node_id.role != expected_role {
            return Err(RuntimeError::TypeError {
                expected: "node with correct role",
                got: "node with incorrect role",
            });
        }
        Ok(node_id)
    };

    for step in 0..max_iterations {
        if engine.is_complete() {
            info!("Plan {} completed in {} steps", run_id, step);
            record_termination(
                RunEnd::PlanComplete,
                &path_state.state,
                &engine,
                step,
                max_iterations,
                recovered_nodes.len(),
                &census,
            );
            return Ok(RunOutcome::Completed { steps: step });
        }

        path_state.state.crash_info.current_step = step;

        // Release delayed messages whose time has come
        path_state.state.release_from_purgatory(step);

        // Dispatch ready events
        let ready_events: Vec<(NodeIndex, PlannedEvent)> = engine
            .get_ready_events()
            .into_iter()
            .map(|(idx, e)| (idx, e.clone()))
            .collect();

        if ready_events.is_empty()
            && path_state.state.all_queues_empty()
            && !in_progress.is_empty()
        {
            warn!(
                "Plan {} deadlocked at step {}: {} client op(s) will never complete",
                run_id, step, in_progress.len()
            );
            record_termination(
                RunEnd::Deadlock,
                &path_state.state,
                &engine,
                step,
                max_iterations,
                recovered_nodes.len(),
                &census,
            );
            return Ok(RunOutcome::Deadlock {
                step,
                pending_ops: in_progress.len(),
            });
        }

        for (node_idx, event) in ready_events {
            match &event.action {
                EventAction::ClientRequest(op_spec) => {
                    op_id_counter += 1;
                    in_progress.insert(op_id_counter, node_idx);
                    util_stats::record_client_op_invoked();

                    // Get a client node from the pool (creates one if needed)
                    let (client_node_id, is_new) =
                        path_state.client_pool.get(&mut path_state.state);

                    if is_new
                        && let Some(init_fn) =
                            program.get_func_by_name("ClientInterface.BASE_NODE_INIT")
                        {
                            let mut env = make_local_env(
                                init_fn,
                                vec![],
                                &Env::<H>::default(),
                                &path_state.state.nodes[client_node_id.index],
                                &program.id_to_name,
                            );
                            if let Err(e) = crate::simulator::core::exec_sync_on_node::<H, _, F>(
                                &mut path_state.state,
                                &mut path_state.logs,
                                &program,
                                &mut env,
                                client_node_id,
                                init_fn.entry,
                                snapshot,
                                &mut path_state.feedback,
                                policy,
                                purgatory_config,
                                rng,
                            ) {
                                log::warn!(
                                    "Failed to initialize dynamic client node {}: {}",
                                    client_node_id,
                                    e
                                );
                            }
                        }

                    // Validate target server in op_spec
                    let target_idx = match op_spec {
                        ClientOpSpec::Write(t, _) => *t as usize,
                        ClientOpSpec::Read(t, _) => *t as usize,
                        ClientOpSpec::Rmw(t, _) => *t as usize,
                    };
                    validate_node(&path_state.state, target_idx, server_role, "Node")?;

                    schedule_client_op(
                        &mut path_state.state,
                        &mut path_state.history,
                        &program,
                        op_id_counter,
                        op_spec,
                        client_node_id,
                        server_role,
                        policy,
                        rng,
                    )?;
                }
                EventAction::CrashNode(node_id) => {
                    let named =
                        validate_node(&path_state.state, *node_id as usize, server_role, "Node")?;
                    let nid = if coverage_guided_fault_placement {
                        let chosen = place_crash(
                            &path_state.state,
                            &global_state.fault_coverage,
                            server_role,
                            topology.num_servers as usize,
                            &taken,
                            named,
                            rng,
                        );
                        placement.insert(*node_id, chosen);
                        taken.insert(chosen.index);
                        chosen
                    } else {
                        named
                    };
                    path_state.state.push_runnable(Runnable::Crash {
                        node_id: nid,
                        priority: policy.sample(rng, RunnableCategory::Crash),
                    });
                    pending_crash_recover.insert(nid.index, node_idx);
                }
                EventAction::RecoverNode(node_id) => {
                    let nid = match placement.remove(node_id) {
                        Some(placed) => {
                            taken.remove(&placed.index);
                            placed
                        }
                        None => {
                            validate_node(&path_state.state, *node_id as usize, server_role, "Node")?
                        }
                    };
                    path_state.state.push_runnable(Runnable::Recover {
                        node_id: nid,
                        priority: policy.sample(rng, RunnableCategory::Recover),
                    });
                    pending_crash_recover.insert(nid.index, node_idx);
                }
                EventAction::AllowTimer(node_id, label) => {
                    let key = (*node_id as usize, label.clone());
                    path_state.state.allowed_timers.insert(key.clone());
                    pending_allow_timer.insert(key, node_idx);
                }
                EventAction::Partition(spec) => {
                    let partition_type = spec.to_partition_type(
                        server_role,
                        topology.num_servers,
                    );
                    path_state.state.push_runnable(Runnable::Partition {
                        partition_type,
                        priority: policy.sample(rng, RunnableCategory::Partition),
                    });
                    pending_partition = Some(node_idx);
                }
                EventAction::Heal => {
                    path_state.state.push_runnable(Runnable::Heal {
                        priority: policy.sample(rng, RunnableCategory::Heal),
                    });
                    pending_heal = Some(node_idx);
                }
                EventAction::Deliver(_) => {
                    // Deliver events are constraints, not actions.
                    // When ready, lift the reservation so the scheduler can pick the match.
                    ready_delivers.insert(node_idx);
                }
            }
        }

        // Build reservations from delivers that are NOT yet ready and NOT completed.
        // These constrain the scheduler from picking their matching runnables early.
        let reservations: Vec<Reservation> = all_delivers
            .iter()
            .filter(|(idx, _)| !ready_delivers.contains(idx) && !completed_delivers.contains(idx))
            .filter_map(|(_, spec)| {
                name_to_entry.get(spec.function.as_str()).map(|&entry_pc| Reservation {
                    entry_pc,
                    from: spec.from.map(|f| f as usize),
                    to: spec.to.map(|t| t as usize),
                })
            })
            .collect();

        let history_start_len = path_state.history.len();

        if path_state.state.all_queues_empty() {
            census.idle();
        } else {
            let result = schedule_runnable::<H, _, _, F>(
                &mut path_state.state,
                &mut path_state.logs,
                &program,
                snapshot,
                &mut path_state.feedback,
                &topology,
                global_state,
                policy,
                strict_timers,
                &mut selector,
                within_queue,
                terms,
                purgatory_config,
                &reservations,
                rng,
            )?;

            if matches!(result, ScheduleResult::None) {
                census.blocked();
            } else {
                census.released();
            }

            match result {
                ScheduleResult::None => {}
                ScheduleResult::ClientOp(result) => {
                    path_state.client_pool.release(NodeId {
                        role: path_state.client_pool.client_role,
                        index: result.client_id as usize,
                    });
                    path_state.history.push(Operation {
                        client_id: result.client_id,
                        op_action: result.op_name,
                        kind: OpKind::Response,
                        payload: vec![result.value],
                        unique_id: result.unique_id,
                        step: path_state.state.crash_info.current_step,
                    });
                }
                ScheduleResult::Crash { node_id } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Crash".to_string(),
                        kind: OpKind::Crash,
                        payload: vec![Value::<H>::node(node_id)],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    crashed_nodes.insert(node_id.index);
                    if let Some(plan_node) = pending_crash_recover.remove(&node_id.index) {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::Recover { node_id } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Recover".to_string(),
                        kind: OpKind::Recover,
                        payload: vec![Value::<H>::node(node_id)],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    if crashed_nodes.contains(&node_id.index) {
                        recovered_nodes.insert(node_id.index);
                    }
                    if let Some(plan_node) = pending_crash_recover.remove(&node_id.index) {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::TimerFired { node_id, label } => {
                    // Recorded beside crashes and recoveries so a consumer can
                    // order a timer against the deliveries and faults around
                    // it. The node goes in `client_id` and the label after the
                    // `/` in the action, so a reader can select the firings it
                    // wants by column without parsing the payload; the payload
                    // carries both as values. The label is the specification's
                    // own name for the timer.
                    let action = format!("System.TimerFired/{label}");
                    path_state.history.push(Operation {
                        client_id: node_id.index as i32,
                        op_action: action,
                        kind: OpKind::TimerFired,
                        payload: vec![
                            Value::<H>::node(node_id),
                            Value::<H>::string(label.as_str().into()),
                        ],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    let key = (node_id.index, label);
                    if let Some(plan_node) = pending_allow_timer.remove(&key) {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::Partition { partition_type: _ } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Partition".to_string(),
                        kind: OpKind::Partition,
                        payload: vec![],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    if let Some(plan_node) = pending_partition.take() {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::Heal => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Heal".to_string(),
                        kind: OpKind::Heal,
                        payload: vec![],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    if let Some(plan_node) = pending_heal.take() {
                        engine.mark_event_completed(plan_node);
                    }
                }
                ScheduleResult::RecordExecuted {
                    entry_pc,
                    origin_node,
                    dest_node,
                } => {
                    // Check if this record delivery matches any ready deliver event.
                    if let Some(&func_name) = entry_to_name.get(&entry_pc) {
                        let matched = ready_delivers
                            .iter()
                            .find(|idx| {
                                if let Some(spec) = all_delivers.get(idx) {
                                    spec.function == func_name
                                        && spec
                                            .to
                                            .is_none_or(|t| dest_node.index == t as usize)
                                        && spec
                                            .from
                                            .is_none_or(|f| origin_node.index == f as usize)
                                } else {
                                    false
                                }
                            })
                            .copied();

                        if let Some(idx) = matched {
                            ready_delivers.remove(&idx);
                            completed_delivers.insert(idx);
                            engine.mark_event_completed(idx);
                        }
                    }
                }
            }
        }

        // Only scan new history entries added during this step
        let completed: Vec<i32> = path_state.history[history_start_len..]
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Response))
            .filter_map(|op| {
                in_progress.get(&op.unique_id).map(|&node_idx| {
                    engine.mark_event_completed(node_idx);
                    op.unique_id
                })
            })
            .collect();

        for id in completed {
            in_progress.remove(&id);
        }

        // Starvation detection: if nothing happened this iteration, increment counter.
        // Helps catch typos in deliver function names.
        if path_state.history.len() == history_start_len {
            no_progress_count += 1;
            if no_progress_count == STARVATION_WARN_THRESHOLD {
                let pending_deliver_names: Vec<&str> = ready_delivers
                    .iter()
                    .filter_map(|idx| all_delivers.get(idx).map(|s| s.function.as_str()))
                    .collect();
                let blocked_deliver_names: Vec<&str> = all_delivers
                    .iter()
                    .filter(|(idx, _)| {
                        !ready_delivers.contains(idx) && !completed_delivers.contains(idx)
                    })
                    .map(|(_, s)| s.function.as_str())
                    .collect();
                warn!(
                    "Plan {} stalled for {} iterations. Ready delivers waiting: {:?}. Blocked delivers: {:?}",
                    run_id, no_progress_count, pending_deliver_names, blocked_deliver_names
                );
            }
        } else {
            no_progress_count = 0;
        }
    }

    warn!(
        "Hit max iterations ({}) before plan {} completion",
        max_iterations, run_id
    );
    record_termination(
        RunEnd::IterationsExhausted,
        &path_state.state,
        &engine,
        max_iterations,
        max_iterations,
        recovered_nodes.len(),
        &census,
    );
    Ok(RunOutcome::IterationsExhausted {
        outstanding_events: engine.outstanding_count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::hash_utils::NoHashing;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SERVER: NameId = NameId(0);

    fn server(index: usize) -> NodeId {
        NodeId {
            role: SERVER,
            index,
        }
    }

    fn state() -> State<NoHashing> {
        State::new(&[(SERVER, 3)], 2)
    }

    #[test]
    fn tag_clamps_restarts_and_peers_down() {
        let mut st = state();
        assert_eq!(fault_context_tag(&st, server(0)), 0);

        st.crash_info.currently_crashed.insert(server(1));
        st.crash_info.currently_crashed.insert(server(2));
        assert_eq!(fault_context_tag(&st, server(0)), 2, "both peers down");
        assert_eq!(fault_context_tag(&st, server(1)), 1, "does not count itself");

        st.incarnations[0] = 9;
        st.send_ledger[0].in_flight = 1;
        assert_eq!(fault_context_tag(&st, server(0)), 2 * 6 + 3 + 2);
    }

    #[test]
    fn placement_avoids_down_and_reserved_nodes() {
        let mut st = state();
        st.crash_info.currently_crashed.insert(server(0));
        let coverage = FaultCoverage::default();
        let taken: HashSet<usize> = [1].into_iter().collect();
        let mut rng = SmallRng::seed_from_u64(4);
        for _ in 0..16 {
            let chosen = place_crash(&st, &coverage, SERVER, 3, &taken, server(0), &mut rng);
            assert_eq!(chosen.index, 2, "only node 2 is up and unreserved");
        }
    }

    #[test]
    fn placement_falls_back_when_nothing_is_eligible() {
        let mut st = state();
        for i in 0..3 {
            st.crash_info.currently_crashed.insert(server(i));
        }
        let coverage = FaultCoverage::default();
        let mut rng = SmallRng::seed_from_u64(1);
        let chosen = place_crash(
            &st,
            &coverage,
            SERVER,
            3,
            &HashSet::new(),
            server(1),
            &mut rng,
        );
        assert_eq!(chosen.index, 1, "the workload's own node");
    }

    #[test]
    fn placement_prefers_the_context_seen_least() {
        let mut st = state();
        // Node 0 waits on a send nobody has received, which is a context the
        // other two are not in.
        st.send_ledger[0].in_flight = 1;
        for seed in 0..8u64 {
            let coverage = FaultCoverage::default();
            coverage.visit(fault_context_tag(&st, server(1)));
            let mut rng = SmallRng::seed_from_u64(seed);
            let chosen =
                place_crash(&st, &coverage, SERVER, 3, &HashSet::new(), server(1), &mut rng);
            assert_eq!(chosen.index, 0, "the only unvisited context, seed {}", seed);
        }
    }
}
