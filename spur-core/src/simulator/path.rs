use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Program, Vertex};
use crate::simulator::core::steer_terms::ResolvedTerms;
use crate::simulator::core::{
    Continuation, LogEntry, Logger, NodeId, OpKind, Operation, PurgatoryConfig,
    QueuePolicyConfig, Record, Reservation, Runnable, RunnableCategory,
    RuntimeError, SchedulePolicy, ScheduleResult, State, TraceEntry, Value, WithinQueueSelector,
    build_frame, schedule_runnable,
};
use crate::simulator::client_anchor::{self, HoldQueue, Released};
use crate::simulator::coverage::GlobalState;
use crate::simulator::feedback::Feedback;
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::path::plan::{
    ClientOpSpec, DeliverSpec, EventAction, ExecutionPlan, PlanEngine, PlannedEvent,
};
use crate::simulator::fault_timing;
use crate::simulator::pair_order as pair_order_split;
use crate::simulator::ghost_release;
use crate::simulator::rng::StreamRng;
use crate::simulator::run_cap;
use crate::simulator::run_variant::{ArmSet, CrashArm};
use crate::simulator::stall_cap::{self, Marks, RunClock, RunEnding};
use crate::simulator::stall_release;
use crate::simulator::history::{RunRows, serialize_history, serialize_logs, serialize_traces};
use crate::simulator::text_buffer::{TextBuffer, TextBuffers};
use crate::simulator::timer_context;
use crate::simulator::util_stats::{
    self, DeliveryBias, RunEnd, RunExtension, RunTermination, StallCapCell, StallReleaseCell,
    StallReleaseRun, StallReleaseSettlement,
};
use ecow::{EcoString, EcoVec};
use log::{info, warn};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeSet, HashMap, HashSet};

pub mod generator;
pub mod plan;

use crate::simulator::deploy::Deployment;

/// A run's log and trace rows and the text they point into.
#[derive(Debug, Default)]
pub struct Logs {
    pub entries: Vec<LogEntry>,
    pub traces: Vec<TraceEntry>,
    pub text: TextBuffers,
}

impl Logger for Logs {
    fn log_text(&mut self) -> &mut TextBuffer {
        &mut self.text.log_content
    }
    fn trace_text(&mut self) -> &mut TextBuffer {
        &mut self.text.trace_payload
    }
    fn log(&mut self, entry: LogEntry) {
        if self.entries.len() == self.entries.capacity() {
            util_stats::record_log_vec_grow();
        }
        self.entries.push(entry);
    }
    fn log_trace(&mut self, entry: TraceEntry) {
        if self.traces.len() == self.traces.capacity() {
            util_stats::record_trace_vec_grow();
        }
        self.traces.push(entry);
    }
}

/// Largest channel table a run starts with, so one long run cannot hand a
/// large sparse table to the next.
const CHANNEL_TABLE_HINT_CAP: usize = 4_096;
/// Largest log or trace row vector a run starts with.
const ROW_VEC_HINT_CAP: usize = 8_192;

/// The sizes the previous run on this thread ended with. Only capacities are
/// taken from them, so no run can observe another through these.
#[derive(Clone, Copy)]
struct RunBufferHints {
    channels: usize,
    log_rows: usize,
    trace_rows: usize,
}

thread_local! {
    static RUN_BUFFER_HINTS: std::cell::Cell<RunBufferHints> = const {
        std::cell::Cell::new(RunBufferHints { channels: 0, log_rows: 0, trace_rows: 0 })
    };
}

/// The channel table capacity a run starting on this thread begins with.
pub fn channel_table_hint() -> usize {
    RUN_BUFFER_HINTS.with(|h| h.get().channels)
}

impl Logs {
    /// Empty row vectors sized to what the previous run on this thread wrote,
    /// and text buffers taken from the free list.
    pub fn sized_from_previous_run() -> Self {
        let h = RUN_BUFFER_HINTS.with(|h| h.get());
        Self {
            entries: Vec::with_capacity(h.log_rows),
            traces: Vec::with_capacity(h.trace_rows),
            text: TextBuffers::take(),
        }
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
            logs: Logs::sized_from_previous_run(),
            history: Vec::new(),
            client_pool: ClientPool::new(client_role, node_slot_count),
        }
    }

    /// Takes the run's executions, log and trace rows with their text, and
    /// records the run's channel, log and trace counts as the starting
    /// capacities of the next run on this thread.
    pub fn take_run_rows(&mut self) -> RunRows {
        let hints = RunBufferHints {
            channels: self.state.channels.len().min(CHANNEL_TABLE_HINT_CAP),
            log_rows: self.logs.entries.len().min(ROW_VEC_HINT_CAP),
            trace_rows: self.logs.traces.len().min(ROW_VEC_HINT_CAP),
        };
        RUN_BUFFER_HINTS.with(|h| h.set(hints));
        let mut text = std::mem::take(&mut self.logs.text);
        let history = serialize_history(&self.history, &mut text);
        text.record_allocated();
        RunRows {
            history,
            logs: serialize_logs(std::mem::take(&mut self.logs.entries)),
            traces: serialize_traces(std::mem::take(&mut self.logs.traces)),
            text,
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
    policy: &SchedulePolicy,
    rng: &mut impl StreamRng,
) -> Result<(), RuntimeError> {
    let client_id = client_node_id.index as i32;
    let functions = prog.role_table.functions(client_node_id.role);
    let (op_name, op_func, dest, key, uid) = match op_spec {
        ClientOpSpec::Write(dest, key) => ("Client.Write", &functions.write, dest, key, Some(op_id)),
        ClientOpSpec::Read(dest, key) => ("Client.Read", &functions.read, dest, key, None),
        ClientOpSpec::Rmw(dest, key) => ("Client.RMW", &functions.rmw, dest, key, Some(op_id)),
    };
    let op_func = op_func
        .as_ref()
        .ok_or_else(|| RuntimeError::MissingRequiredFunction(op_name.to_string()))?;
    // The invocation row always has the shape [dest, key, uid] (Read has no
    // uid), with unit for an operation that takes no destination; the
    // function receives only its own parameters.
    let mut payload = vec![
        dest.map(Value::<H>::node).unwrap_or_else(Value::<H>::unit),
        Value::<H>::string(EcoString::from(key.as_str())),
    ];
    if let Some(uid) = uid {
        payload.push(Value::<H>::int(uid as i64));
    }
    let actuals = &payload[usize::from(dest.is_none())..];
    let initial_args: EcoVec<Value<H>> = actuals.iter().cloned().collect();
    let env = build_frame(op_func, &initial_args);

    history.push(Operation {
        client_id,
        op_action: op_name.to_string(),
        kind: OpKind::Invocation,
        payload,
        unique_id: op_id,
        step: state.crash_info.current_step,
    });

    let send_ordinal = state.next_send_ordinal(client_node_id);
    let receiver_token_at_send = state.node_state_token(client_node_id);
    let drawn_priority = policy.sample(rng, RunnableCategory::Record);
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
        initial_args,
        entry_func: op_func.name,
        env,
        priority: state.record_priority(Some(op_id), drawn_priority),
        causal_operation_id: Some(op_id),
        trace_id: None,
        trace_payload: None,
        link_seq: None,
        origin_incarnation: state.incarnation(client_node_id),
        bias: DeliveryBias::NONE,
        timer_entry: None,
        send_ordinal,
        receiver_token_at_send,
    }));
    Ok(())
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
    deployment: &Deployment,
    plan_node: NodeIndex,
    op_spec: &ClientOpSpec,
    post_fault: bool,
    in_progress: &mut HashMap<i32, NodeIndex>,
    op_id_counter: &mut i32,
    rng: &mut impl StreamRng,
) -> Result<(), RuntimeError> {
    *op_id_counter += 1;
    in_progress.insert(*op_id_counter, plan_node);
    util_stats::record_client_op_invoked();

    if post_fault {
        let op_id = *op_id_counter;
        let step = path_state.state.crash_info.current_step;
        path_state
            .state
            .client_anchor
            .first_post_fault_op
            .get_or_insert(op_id);
        if path_state.state.client_anchor.arm == client_anchor::Arm::Rush {
            path_state.state.client_anchor.rushed_ops.insert(op_id);
            util_stats::record_client_anchor_rush_op();
        }
        if util_stats::enabled() && op_id % client_anchor::DISTANCE_STRIDE == 0 {
            path_state
                .state
                .client_anchor
                .awaiting_delivery
                .insert(op_id, step);
        }
    }

    // Get a client node from the pool (creates one if needed)
    let (client_node_id, is_new) = path_state.client_pool.get(&mut path_state.state);

    if is_new {
        path_state
            .state
            .set_context(client_node_id.index, deployment.root_value::<H>());
    }
    if is_new && let Some(init_fn) = program.role_table.functions(client_node_id.role).base_init.as_ref() {
        let mut env = build_frame::<H>(init_fn, &[]);
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

    schedule_client_op(
        &mut path_state.state,
        &mut path_state.history,
        program,
        *op_id_counter,
        op_spec,
        client_node_id,
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
    /// The run went `cap` steps past its last progress mark, short of its
    /// step cap, with planned events still outstanding; `step` is the steps
    /// it ran.
    StallCapReached {
        cap: i32,
        step: i32,
        outstanding_events: usize,
    },
}

/// Plan engine work of one run, counted when the run's plan loop is left by
/// any path, an error return included.
#[derive(Default)]
struct PlanWork {
    scans: u64,
    scans_skipped: u64,
    scans_empty: u64,
    events_released: u64,
    deliver_lookups: u64,
    deliver_lookups_skipped: u64,
}

impl Drop for PlanWork {
    fn drop(&mut self) {
        util_stats::record_plan_work(
            self.scans,
            self.scans_skipped,
            self.scans_empty,
            self.events_released,
            self.deliver_lookups,
            self.deliver_lookups_skipped,
        );
    }
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

/// Settle every client operation still in progress for the plan's
/// dependency purposes: its plan node completes, so the events ordered
/// behind it become ready, while the operation itself stays in progress so
/// its real response, if one arrives, is recorded as any other. Operations
/// are settled in id order so the run stays a function of its seed.
fn settle_in_progress(
    engine: &mut PlanEngine,
    in_progress: &HashMap<i32, NodeIndex>,
    settled: &mut HashSet<i32>,
) -> StallReleaseSettlement {
    let mut settlement = StallReleaseSettlement {
        ops_settled: 0,
        dependents_client: 0,
        dependents_fault: 0,
        dependents_other: 0,
    };
    let mut ids: Vec<i32> = in_progress.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        settled.insert(id);
        settlement.ops_settled += 1;
        for child in engine.mark_event_completed(in_progress[&id]) {
            match engine.event(child).action {
                EventAction::ClientRequest(_) => settlement.dependents_client += 1,
                EventAction::CrashNode(_)
                | EventAction::RecoverNode(_)
                | EventAction::Partition(_)
                | EventAction::Heal => settlement.dependents_fault += 1,
                EventAction::AllowTimer(..) | EventAction::Deliver(_) => {
                    settlement.dependents_other += 1
                }
            }
        }
    }
    settlement
}

/// One run ended, reported to the stall-release counters with the steps it
/// ran, whether its plan completed, whether the stall cap ended it, and the
/// client requests it issued.
fn finish_release_run(
    cell: StallReleaseCell,
    release_step: Option<i32>,
    steps: i32,
    completed: bool,
    stalled: bool,
    invocations: i32,
) {
    util_stats::record_stall_release_run(&StallReleaseRun {
        cell,
        release_step: release_step.map(|s| s.max(0) as u64),
        steps: steps.max(0) as u64,
        completed,
        stalled,
        invocations: invocations.max(0) as u64,
    });
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
    util_stats::record_ghost_release_run(state.ghost_release.cell, steps_used);
    util_stats::record_channels_created(state.channels.len() as u64);
    let (lookups, misses) = state.channels.lookup_counts();
    util_stats::record_channel_table(lookups, misses, state.channels.len() as u64);
    util_stats::flush_frame_stats();
}

pub fn exec_plan<H: HashPolicy, F: Feedback>(
    path_state: &mut PathState<H, F>,
    program: &Program,
    plan: ExecutionPlan,
    max_iterations: i32,
    deployment: &Deployment,
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
    arms: &ArmSet,
    rng: &mut impl StreamRng,
) -> Result<RunOutcome, RuntimeError> {
    util_stats::begin_run();
    util_stats::record_program_clone_avoided();
    path_state.state.retarget.enabled = arms.retarget;
    // The bound is read once here so the scheduler needs no learner access
    // of its own.
    path_state.state.ghost_release = ghost_release::RunState::at_run_start(
        run_id,
        max_iterations,
        arms.placed(),
        arms.crash == CrashArm::PlacedPhase,
    );
    path_state.state.fresh_first.enabled = arms.fresh_first;
    path_state.state.pair_order.enabled = arms.pair_order;
    path_state.state.client_anchor.arm = arms.request;
    let pair_order = arms.pair_order;
    util_stats::record_client_anchor_arm_run(arms.request);
    let anchored = path_state.state.client_anchor.holds();
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
    // The stall cap is frozen at run start like the step cap. Every cell
    // keeps the clock; only the treated cell is cut by it.
    let stall_cell = stall_cap::cell(run_id);
    let stall_treated = stall_cell == StallCapCell::Treated;
    let stall_cap_standing = stall_cap::effective_cap(backup);
    let mut stall_clock = RunClock::default();
    // A released run settles its in-progress operations once, at its first
    // stall; `settled` holds their ids so a later real response completes
    // no plan node twice.
    let release_cell = stall_release::cell(run_id);
    let mut release_step: Option<i32> = None;
    let mut settled: HashSet<i32> = HashSet::new();
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
    let mut plan_work = PlanWork::default();

    // Starvation detection: track consecutive no-progress iterations
    let mut no_progress_count: i32 = 0;
    const STARVATION_WARN_THRESHOLD: i32 = 500;


    // Eligible runnables per local queue, rewritten by every scheduling step.
    let mut local_queue_sizes: Vec<usize> = Vec::new();

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
            stall_cap::finish_run(
                stall_cell,
                &stall_clock,
                stall_cap_standing,
                RunEnding {
                    run_id,
                    backup,
                    completed: true,
                    steps_saved: None,
                },
            );
            finish_release_run(release_cell, release_step, step, true, false, op_id_counter);
            if is_probe {
                run_cap::merge_probe(backup, run_cap::Outcome::Completed, step);
                fault_timing::merge_stock_probe(run_id, backup, run_cap::Outcome::Completed, step);
            }
            return Ok(RunOutcome::Completed { steps: step });
        }

        path_state.state.crash_info.current_step = step;
        util_stats::record_steer_step_total();
        let step_rows_start = path_state.history.len();
        let mut marks = Marks::default();

        // Release delayed messages whose time has come
        path_state.state.release_from_purgatory(step);

        // Dispatch ready events
        let ready_events: Vec<(NodeIndex, PlannedEvent)> = if engine.has_ready() {
            let released: Vec<(NodeIndex, PlannedEvent)> = engine
                .get_ready_events()
                .into_iter()
                .map(|(idx, e)| (idx, e.clone()))
                .collect();
            plan_work.scans += 1;
            plan_work.scans_empty += released.is_empty() as u64;
            plan_work.events_released += released.len() as u64;
            released
        } else {
            plan_work.scans_skipped += 1;
            Vec::new()
        };

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
        marks.release = !ready_events.is_empty() || !due.is_empty();

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
            stall_cap::finish_run(
                stall_cell,
                &stall_clock,
                stall_cap_standing,
                RunEnding {
                    run_id,
                    backup,
                    completed: false,
                    steps_saved: None,
                },
            );
            finish_release_run(release_cell, release_step, step, false, false, op_id_counter);
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
                program,
                snapshot,
                policy,
                purgatory_config,
                deployment,
                plan_node,
                &op_spec,
                true,
                &mut in_progress,
                &mut op_id_counter,
                rng,
            )?;
        }

        for (node_idx, event) in ready_events {
            match &event.action {
                EventAction::ClientRequest(op_spec) => {
                    // A request that becomes ready once a crash has happened
                    // is the population every direction of the axis acts on:
                    // a holding run makes it wait, and the others issue it
                    // now.
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
                        program,
                        snapshot,
                        policy,
                        purgatory_config,
                        deployment,
                        node_idx,
                        op_spec,
                        post_fault,
                        &mut in_progress,
                        &mut op_id_counter,
                        rng,
                    )?;
                }
                EventAction::CrashNode(node_id) => {
                    let nid = *node_id;
                    path_state.state.push_runnable(Runnable::Crash {
                        node_id: nid,
                        priority: policy.sample(rng, RunnableCategory::Crash),
                    });
                    // Placed runs hold the crash until a step drawn
                    // uniformly over the learned completed-run span; stock
                    // runs draw nothing.
                    if let Some(target) =
                        fault_timing::draw_hold(arms.placed(), backup, effective_cap, step, rng)
                    {
                        path_state.state.crash_hold_drawn = true;
                        if let Some(hold) =
                            path_state.state.crash_hold_until.get_mut(nid.index)
                        {
                            *hold = target;
                        }
                        // On the phase arm the crash waits further, for a
                        // drawn phase of the victim's own fan-out, once that
                        // target step arrives.
                        if arms.crash == CrashArm::PlacedPhase {
                            path_state.state.crash_phase.arm_node(
                                nid.index,
                                fault_timing::cap_reserve(effective_cap),
                            );
                        }
                    }
                    pending_crash.insert(nid.index, node_idx);
                }
                EventAction::RecoverNode(node_id) => {
                    let nid = *node_id;
                    let target = victim_remap.remove(&nid.index).unwrap_or(nid);
                    path_state.state.push_runnable(Runnable::Recover {
                        node_id: target,
                        priority: policy.sample(rng, RunnableCategory::Recover),
                    });
                    pending_recover.insert(target.index, node_idx);
                }
                EventAction::AllowTimer(node_id, label) => {
                    let key = (node_id.index, label.clone());
                    path_state.state.allowed_timers.insert(key.clone());
                    pending_allow_timer.insert(key, node_idx);
                }
                EventAction::Partition(spec) => {
                    let partition_type = spec.to_partition_type();
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
        let reservations: Vec<Reservation> = if all_delivers.is_empty() {
            Vec::new()
        } else {
            all_delivers
                .iter()
                .filter(|(idx, _)| {
                    !ready_delivers.contains(idx) && !completed_delivers.contains(idx)
                })
                .filter_map(|(_, spec)| {
                    name_to_entry.get(spec.function.as_str()).map(|&entry_pc| Reservation {
                        entry_pc,
                        from: spec.from.map(|n| n.index),
                        to: spec.to.map(|n| n.index),
                    })
                })
                .collect()
        };

        let history_start_len = path_state.history.len();

        if arms.retarget {
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
                program,
                snapshot,
                &mut path_state.feedback,
                deployment,
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
                &mut local_queue_sizes,
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
                ScheduleResult::TimerFired {
                    node_id,
                    label,
                    acted,
                } => {
                    if acted {
                        marks.acted_timer = true;
                    }
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
                    acted,
                    timer_entry,
                } => {
                    if acted && timer_entry {
                        marks.acted_timer = true;
                    } else if acted {
                        marks.acted_delivery = true;
                    }
                    // Check if this record delivery matches any ready deliver event.
                    let deliver_ready = !ready_delivers.is_empty();
                    plan_work.deliver_lookups += deliver_ready as u64;
                    plan_work.deliver_lookups_skipped += !deliver_ready as u64;
                    if deliver_ready
                        && let Some(&func_name) = entry_to_name.get(&entry_pc)
                    {
                        let matched = ready_delivers
                            .iter()
                            .find(|idx| {
                                if let Some(spec) = all_delivers.get(idx) {
                                    spec.function == func_name
                                        && spec
                                            .to
                                            .is_none_or(|t| dest_node.index == t.index)
                                        && spec
                                            .from
                                            .is_none_or(|f| origin_node.index == f.index)
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
            &deployment.fanout_width,
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
                    if settled.remove(&op.unique_id) {
                        util_stats::record_stall_release_late_response();
                    } else {
                        engine.mark_event_completed(node_idx);
                    }
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

        // The clock holds while the run waits on a release the scheduler
        // itself owns and will grant: a planned crash withheld by its
        // placement hold or its phase wait, a held client request, or a
        // delayed message. Each of those is bounded, so a run waiting on one
        // is not stalled.
        marks.row = path_state.history[step_rows_start..]
            .iter()
            .any(|op| !matches!(op.kind, OpKind::TimerFired));
        let crash_withheld = pending_crash.keys().any(|&n| {
            path_state
                .state
                .crash_hold_until
                .get(n)
                .is_some_and(|&until| step < until)
                || path_state.state.crash_phase.awaits_release(n)
        });
        let suspended =
            crash_withheld || !held.is_empty() || !path_state.state.purgatory.is_empty();
        let gap = stall_clock.step(marks, suspended);
        if stall_treated
            && let Some(cap) = stall_cap_standing
            && gap > cap
            && !engine.is_complete()
        {
            let stop_step = step + 1;
            if release_cell == StallReleaseCell::Release && release_step.is_none() {
                if in_progress.is_empty() {
                    util_stats::record_stall_release_without_ops();
                } else {
                    let settlement = settle_in_progress(&mut engine, &in_progress, &mut settled);
                    stall_clock.release();
                    release_step = Some(stop_step);
                    util_stats::record_stall_release(&settlement);
                    continue;
                }
            }
            util_stats::record_client_anchor_run_end(anchored, false, held.pending());
            record_termination(
                RunEnd::StallCapReached,
                run_id,
                &path_state.state,
                &engine,
                stop_step,
                max_iterations,
                recovered_nodes.len(),
                &census,
            );
            stall_cap::finish_run(
                stall_cell,
                &stall_clock,
                stall_cap_standing,
                RunEnding {
                    run_id,
                    backup,
                    completed: false,
                    steps_saved: Some((effective_cap - stop_step).max(0) as u64),
                },
            );
            finish_release_run(release_cell, release_step, stop_step, false, true, op_id_counter);
            return Ok(RunOutcome::StallCapReached {
                cap,
                step: stop_step,
                outstanding_events: engine.outstanding_count(),
            });
        }
    }

    util_stats::record_client_anchor_run_end(anchored, false, held.pending());
    stall_cap::finish_run(
        stall_cell,
        &stall_clock,
        stall_cap_standing,
        RunEnding {
            run_id,
            backup,
            completed: false,
            steps_saved: None,
        },
    );
    finish_release_run(release_cell, release_step, effective_cap, false, false, op_id_counter);
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
        // The path state this builds does not fit a default test thread
        // stack in an unoptimized build.
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(held_request_recording)
            .expect("the test thread starts")
            .join()
            .expect("the test thread completes");
    }

    fn held_request_recording() {
        let program = crate::compiler::compile(SPEC, "canchor.spur")
            .into_program()
            .expect("the fixture compiles");
        let server_role = role(&program, "Node");
        let client_role = role(&program, "KVClient");
        let deploy = crate::simulator::deploy::select_deploy(&program, None).expect("one deploy");
        let deployment = crate::simulator::deploy::evaluate_deploy(&program, deploy, &serde_json::json!({"n": 3}))
            .expect("the deploy evaluates")
            .expect("n = 3 is accepted");
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
        let dest = NodeId { role: server_role, index: 1 };
        held.hold((NodeIndex::new(4), ClientOpSpec::Write(Some(dest), "k".into())), 3);
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
                &deployment,
                plan_node,
                &spec,
                true,
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
        assert_eq!(row.op_action, "Client.Write");
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
