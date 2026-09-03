use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::steer_terms::ResolvedTerms;
use crate::simulator::core::{
    Continuation, Env, LogEntry, Logger, NodeId, OpKind, Operation, PurgatoryConfig,
    QueuePolicyConfig, Record, Reservation, Runnable, RunnableCategory,
    RuntimeError, SchedulePolicy, ScheduleResult, State, TraceEntry, Value, WithinQueueSelector,
    make_local_env, schedule_runnable,
};
use crate::simulator::client_anchor::{self, HoldQueue, Released};
use crate::simulator::crash_phase;
use crate::simulator::coverage::GlobalState;
use crate::simulator::feedback::Feedback;
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::plan::{
    ClientOpSpec, DeliverSpec, EventAction, ExecutionPlan, PlanEngine, PlannedEvent,
};
use crate::simulator::fault_timing;
use crate::simulator::pair_order as pair_order_split;
use crate::simulator::rng::StreamRng;
use crate::simulator::run_cap;
use crate::simulator::timer_context;
use crate::simulator::util_stats::{self, DeliveryBias, RunEnd, RunExtension, RunTermination};
use ecow::EcoString;
use log::{info, warn};
use petgraph::graph::NodeIndex;
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

fn validate_node<H: HashPolicy>(
    state: &State<H>,
    index: usize,
    expected_role: NameId,
) -> Result<NodeId, RuntimeError> {
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
}

/// Hand the planned client request at `plan_node` to a client node. The
/// invocation is recorded at the current step, so a request the run held
/// back is recorded when it is issued, not when the plan made it ready.
fn invoke_client_request<H: HashPolicy, F: Feedback>(
    path_state: &mut PathState<H, F>,
    program: &Program,
    snapshot: &F::Snapshot,
    policy: &SchedulePolicy,
    purgatory_config: &PurgatoryConfig,
    server_role: NameId,
    plan_node: NodeIndex,
    op_spec: &ClientOpSpec,
    in_progress: &mut HashMap<i32, NodeIndex>,
    op_id_counter: &mut i32,
    rng: &mut impl StreamRng,
) -> Result<(), RuntimeError> {
    *op_id_counter += 1;
    in_progress.insert(*op_id_counter, plan_node);
    util_stats::record_client_op_invoked();

    // Get a client node from the pool (creates one if needed)
    let (client_node_id, is_new) = path_state.client_pool.get(&mut path_state.state);

    if is_new && let Some(init_fn) = program.get_func_by_name("ClientInterface.BASE_NODE_INIT") {
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
            program,
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
    validate_node(&path_state.state, target_idx, server_role)?;

    schedule_client_op(
        &mut path_state.state,
        &mut path_state.history,
        program,
        *op_id_counter,
        op_spec,
        client_node_id,
        server_role,
        policy,
        rng,
    )
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Completed { steps: i32 },
    Deadlock { step: i32, pending_ops: usize },
    /// The step budget ran out with planned events still outstanding.
    IterationsExhausted { outstanding_events: usize },
    /// The run reached the learned step cap, short of the configured budget,
    /// with planned events still outstanding.
    LearnedCapReached { cap: i32, outstanding_events: usize },
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
    run_id: i64,
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
    util_stats::record_quiet_stretch(run_id, end);
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
    partial_fanout_crash_bias: f64,
    retarget_crashes: bool,
    fresh_first: bool,
    pair_order: bool,
    client_anchor: bool,
    rng: &mut impl StreamRng,
) -> Result<RunOutcome, RuntimeError> {
    util_stats::begin_run();
    path_state.state.retarget.enabled = retarget_crashes;
    path_state.state.fresh_first.enabled = fresh_first;
    path_state.state.pair_order.enabled = pair_order;
    path_state.state.client_anchor.enabled = client_anchor;
    let anchored = client_anchor;
    // Client requests that became ready after the run's first crash and are
    // waiting out their hold, and the last step at which a window opened.
    let mut held: HoldQueue<(NodeIndex, ClientOpSpec)> = HoldQueue::default();
    let mut last_window_step: Option<i32> = None;
    let census = util_stats::enabled() && pair_order_split::is_census_run(run_id);
    path_state.state.pair_order.census = census;
    if census {
        util_stats::record_pair_order_census_run(pair_order);
    }
    let backup = max_iterations;
    let is_probe = run_cap::is_probe(run_id);
    let effective_cap = if is_probe {
        backup
    } else {
        run_cap::effective_cap(backup)
    };
    let timer_ctx_mode = timer_context::run_mode(run_id);
    let mut selector = queue_policy.to_selector();
    let mut op_id_counter = 0i32;
    let mut in_progress: HashMap<i32, NodeIndex> = HashMap::new();
    // Plan engine NodeIndex of the queued crash, and of the queued recover,
    // per node index. The plan serializes a node's pairs, so each node has at
    // most one of each outstanding; a retargeted crash leaves its recover
    // keyed on the node the crash landed on, which `victim_remap` records
    // from the plan's victim until that recover is issued.
    let mut pending_crash: HashMap<usize, NodeIndex> = HashMap::new();
    let mut pending_recover: HashMap<usize, NodeIndex> = HashMap::new();
    let mut victim_remap: HashMap<usize, NodeId> = HashMap::new();
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

    for step in 0..effective_cap {
        if engine.is_complete() {
            info!("Plan {} completed in {} steps", run_id, step);
            util_stats::record_client_anchor_run_end(anchored, true, held.pending());
            record_termination(
                RunEnd::PlanComplete,
                run_id,
                &path_state.state,
                &engine,
                step,
                max_iterations,
                recovered_nodes.len(),
                &census,
            );
            if is_probe {
                run_cap::merge_probe(backup, run_cap::Outcome::Completed, step);
                fault_timing::merge_stock_probe(run_id, backup, run_cap::Outcome::Completed, step);
            }
            return Ok(RunOutcome::Completed { steps: step });
        }

        path_state.state.crash_info.current_step = step;
        util_stats::record_steer_step_total();

        // Release delayed messages whose time has come
        path_state.state.release_from_purgatory(step);

        // Dispatch ready events
        let ready_events: Vec<(NodeIndex, PlannedEvent)> = engine
            .get_ready_events()
            .into_iter()
            .map(|(idx, e)| (idx, e.clone()))
            .collect();

        // Requests whose wait ran past expiry. When nothing else in the run
        // can move, the earliest held request is issued now instead, so a
        // held request never leaves a run idle or reads as a deadlock.
        let mut due: Vec<Released<(NodeIndex, ClientOpSpec)>> = held.take_due(step);
        if ready_events.is_empty()
            && path_state.state.all_queues_empty()
            && due.is_empty()
            && let Some(dry) = held.take_dry()
        {
            due.push(dry);
        }

        if ready_events.is_empty()
            && path_state.state.all_queues_empty()
            && due.is_empty()
            && !in_progress.is_empty()
        {
            util_stats::record_client_anchor_run_end(anchored, false, held.pending());
            warn!(
                "Plan {} deadlocked at step {}: {} client op(s) will never complete",
                run_id, step, in_progress.len()
            );
            record_termination(
                RunEnd::Deadlock,
                run_id,
                &path_state.state,
                &engine,
                step,
                max_iterations,
                recovered_nodes.len(),
                &census,
            );
            if is_probe {
                run_cap::merge_probe(backup, run_cap::Outcome::Deadlocked, step);
                fault_timing::merge_stock_probe(run_id, backup, run_cap::Outcome::Deadlocked, step);
            }
            return Ok(RunOutcome::Deadlock {
                step,
                pending_ops: in_progress.len(),
            });
        }

        let in_window = last_window_step == Some(step - 1);
        for released in due {
            let (plan_node, op_spec) = released.item;
            util_stats::record_client_anchor_release(
                released.release,
                (step - released.ready_step).max(0) as u64,
            );
            util_stats::record_client_anchor_post_fault_invocation(anchored, in_window);
            invoke_client_request(
                path_state,
                &program,
                snapshot,
                policy,
                purgatory_config,
                server_role,
                plan_node,
                &op_spec,
                &mut in_progress,
                &mut op_id_counter,
                rng,
            )?;
        }

        for (node_idx, event) in ready_events {
            match &event.action {
                EventAction::ClientRequest(op_spec) => {
                    // A request that becomes ready once a crash has happened
                    // is the population the hold applies to; on the treated
                    // half it waits, on the control half it is issued now.
                    let post_fault = !crashed_nodes.is_empty();
                    if post_fault {
                        util_stats::record_client_anchor_post_fault_request(anchored);
                    }
                    if post_fault && anchored {
                        held.hold((node_idx, op_spec.clone()), step);
                        continue;
                    }
                    if post_fault {
                        util_stats::record_client_anchor_post_fault_invocation(anchored, in_window);
                    }
                    invoke_client_request(
                        path_state,
                        &program,
                        snapshot,
                        policy,
                        purgatory_config,
                        server_role,
                        node_idx,
                        op_spec,
                        &mut in_progress,
                        &mut op_id_counter,
                        rng,
                    )?;
                }
                EventAction::CrashNode(node_id) => {
                    let nid =
                        validate_node(&path_state.state, *node_id as usize, server_role)?;
                    path_state.state.push_runnable(Runnable::Crash {
                        node_id: nid,
                        priority: policy.sample(rng, RunnableCategory::Crash),
                    });
                    // Placed-posture runs hold the crash until a step drawn
                    // uniformly over the learned completed-run span; stock
                    // runs draw nothing and behave exactly as before.
                    if let Some(target) =
                        fault_timing::draw_hold(run_id, backup, effective_cap, step, rng)
                    {
                        path_state.state.crash_hold_drawn = true;
                        if let Some(hold) =
                            path_state.state.crash_hold_until.get_mut(nid.index)
                        {
                            *hold = target;
                        }
                        // On the anchored half of the placed runs the crash
                        // waits further, for a drawn phase of the victim's
                        // own fan-out, once that target step arrives.
                        if crash_phase::is_anchored(run_id) {
                            path_state.state.crash_phase.arm_node(
                                nid.index,
                                fault_timing::cap_reserve(effective_cap),
                            );
                        }
                    }
                    pending_crash.insert(nid.index, node_idx);
                }
                EventAction::RecoverNode(node_id) => {
                    let nid =
                        validate_node(&path_state.state, *node_id as usize, server_role)?;
                    let target = victim_remap.remove(&nid.index).unwrap_or(nid);
                    path_state.state.push_runnable(Runnable::Recover {
                        node_id: target,
                        priority: policy.sample(rng, RunnableCategory::Recover),
                    });
                    pending_recover.insert(target.index, node_idx);
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

        if retarget_crashes {
            let mut mask = 0u64;
            for &n in pending_crash.keys().chain(pending_recover.keys()) {
                if n < u64::BITS as usize {
                    mask |= 1u64 << n;
                }
            }
            path_state.state.retarget.pending_pair_mask = mask;
        }

        if path_state.state.all_queues_empty() {
            census.idle();
            util_stats::record_steer_reach(util_stats::SteerReach::NoScheduleAttempt);
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
                partial_fanout_crash_bias,
                timer_ctx_mode,
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
                ScheduleResult::Crash { node_id, planned } => {
                    path_state.history.push(Operation {
                        client_id: -1,
                        op_action: "System.Crash".to_string(),
                        kind: OpKind::Crash,
                        payload: vec![Value::<H>::node(node_id)],
                        unique_id: -1,
                        step: path_state.state.crash_info.current_step,
                    });
                    crashed_nodes.insert(node_id.index);
                    if let Some(plan_node) = pending_crash.remove(&planned.index) {
                        engine.mark_event_completed(plan_node);
                    }
                    if node_id != planned {
                        victim_remap.insert(planned.index, node_id);
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
                    if let Some(plan_node) = pending_recover.remove(&node_id.index) {
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

        // A window opens when this step's dispatch left a server that took an
        // acted fault-crossing delivery with its full fan-out still in the
        // air. It is counted on both halves and issues nothing; a treated run
        // records what it held at its first window.
        if client_anchor::fanout_window(
            &path_state.state.send_ledger,
            topology.num_servers.max(0) as usize,
            step,
        ) {
            util_stats::record_client_anchor_window(anchored);
            last_window_step = Some(step);
            if anchored {
                let firing = held.fire();
                if firing.first {
                    util_stats::record_client_anchor_first_window(firing.held);
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

    util_stats::record_client_anchor_run_end(anchored, false, held.pending());
    if effective_cap < backup {
        record_termination(
            RunEnd::LearnedCapReached,
            run_id,
            &path_state.state,
            &engine,
            effective_cap,
            backup,
            recovered_nodes.len(),
            &census,
        );
        return Ok(RunOutcome::LearnedCapReached {
            cap: effective_cap,
            outstanding_events: engine.outstanding_count(),
        });
    }
    warn!(
        "Hit max iterations ({}) before plan {} completion",
        max_iterations, run_id
    );
    record_termination(
        RunEnd::IterationsExhausted,
        run_id,
        &path_state.state,
        &engine,
        max_iterations,
        max_iterations,
        recovered_nodes.len(),
        &census,
    );
    if is_probe {
        run_cap::merge_probe(backup, run_cap::Outcome::Exhausted, backup);
        fault_timing::merge_stock_probe(run_id, backup, run_cap::Outcome::Exhausted, backup);
    }
    Ok(RunOutcome::IterationsExhausted {
        outstanding_events: engine.outstanding_count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::feedback::NoFeedback;
    use crate::simulator::hash_utils::NoHashing;
    use crate::simulator::rng::{LiveRng, RecRng, RngSource, StreamSet};

    const SPEC: &str = include_str!("../../tests/fixtures/canchor.spur");

    fn role(program: &Program, name: &str) -> NameId {
        program
            .roles
            .iter()
            .find(|(_, n)| n == name)
            .map(|(id, _)| *id)
            .expect("the fixture declares the role")
    }

    /// A request held at one step and issued at expiry gets its invocation
    /// row, operation id and client record only at the step it is issued;
    /// a window in between writes nothing.
    #[test]
    fn a_held_request_is_recorded_at_the_step_it_is_issued() {
        let program = crate::compiler::compile(SPEC, "canchor.spur")
            .into_program()
            .expect("the fixture compiles");
        let server_role = role(&program, "Node");
        let client_role = role(&program, "ClientInterface");
        let mut path_state = PathState::<NoHashing, NoFeedback>::new(
            &[(server_role, 3)],
            program.max_node_slots as usize,
            client_role,
        );
        let policy = SchedulePolicy::default();
        let purgatory = PurgatoryConfig::default();
        let mut inner = StreamSet::new(7, true);
        let mut tape = <LiveRng as RngSource>::new_tape(None);
        let mut rng = RecRng::<LiveRng> {
            tape: &mut tape,
            inner: &mut inner,
        };
        let mut in_progress: HashMap<i32, NodeIndex> = HashMap::new();
        let mut op_id = 0i32;
        let mut held: HoldQueue<(NodeIndex, ClientOpSpec)> = HoldQueue::default();

        path_state.state.crash_info.current_step = 3;
        held.hold((NodeIndex::new(4), ClientOpSpec::Write(1, "k".into())), 3);
        assert!(path_state.history.is_empty(), "holding writes no row");
        assert_eq!(path_state.state.total_runnable_count(), 0);

        path_state.state.crash_info.current_step = 5;
        assert!(held.take_due(5).is_empty());
        assert_eq!(held.fire(), client_anchor::Firing { first: true, held: 1 });
        assert!(held.take_due(6).is_empty(), "a window sets nothing aside");
        assert!(path_state.history.is_empty(), "the window itself writes no row");

        let issue_step = 3 + client_anchor::EXPIRY_STEPS + 1;
        assert!(held.take_due(issue_step - 1).is_empty());
        path_state.state.crash_info.current_step = issue_step;
        let due = held.take_due(issue_step);
        assert_eq!(due.len(), 1);
        for released in due {
            let (plan_node, spec) = released.item;
            assert_eq!(released.ready_step, 3);
            assert_eq!(released.release, client_anchor::Release::Expiry);
            invoke_client_request(
                &mut path_state,
                &program,
                &(),
                &policy,
                &purgatory,
                server_role,
                plan_node,
                &spec,
                &mut in_progress,
                &mut op_id,
                &mut rng,
            )
            .expect("the request is issued");
        }
        assert_eq!(path_state.history.len(), 1);
        let row = &path_state.history[0];
        assert_eq!(row.step, issue_step, "the invocation row carries the issue step");
        assert!(matches!(row.kind, OpKind::Invocation));
        assert_eq!(row.op_action, "ClientInterface.Write");
        assert_eq!(row.unique_id, 1);
        assert_eq!(in_progress.get(&1), Some(&NodeIndex::new(4)));
        assert_eq!(
            path_state.state.total_runnable_count(),
            1,
            "the client record is queued when the request is issued"
        );
        assert!(held.is_empty());
    }
}
