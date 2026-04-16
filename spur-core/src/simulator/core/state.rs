use crate::analysis::resolver::NameId;
use crate::analysis::type_id::TypeId;
use crate::compiler::cfg::{Lhs, Vertex};
use crate::simulator::core::eval::store;
use crate::simulator::core::partition::{PartitionInfo, PartitionType};
use crate::simulator::core::values::{ChannelId, Env, Value};
use crate::simulator::hash_utils::{HashPolicy, compute_hash};
use imbl::{HashMap as ImHashMap, OrdSet, Vector};
use rand::Rng;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Defines the priority band for a category of runnable.
#[derive(Debug, Clone, Deserialize)]
pub struct PriorityBand {
    pub center: f64,
    pub width: f64,
}

impl PriorityBand {
    pub fn fixed(value: f64) -> Self {
        Self {
            center: value,
            width: 0.0,
        }
    }
}

/// Which category of runnable is being sampled.
#[derive(Debug, Clone, Copy)]
pub enum RunnableCategory {
    Record,
    Timer,
    ChannelSend,
    Crash,
    Recover,
    Partition,
    Heal,
}

/// Configures how base priorities are sampled for new runnables.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum SchedulePolicy {
    /// Fixed priorities per category (legacy behavior).
    Fixed,
    /// Sample from Beta(α, β) mapped into per-category bands.
    Shaped {
        alpha: f64,
        beta: f64,
        record: PriorityBand,
        timer: PriorityBand,
        channel_send: PriorityBand,
        crash: PriorityBand,
        recover: PriorityBand,
        #[serde(default = "default_partition_band")]
        partition: PriorityBand,
        #[serde(default = "default_heal_band")]
        heal: PriorityBand,
    },
}

impl Default for SchedulePolicy {
    fn default() -> Self {
        SchedulePolicy::Shaped {
            alpha: 0.5,
            beta: 0.5, // Arcsine distribution — favors tails
            record: PriorityBand {
                center: 0.5,
                width: 0.15,
            },
            timer: PriorityBand {
                center: 0.25,
                width: 0.10,
            },
            channel_send: PriorityBand {
                center: 0.5,
                width: 0.15,
            },
            crash: PriorityBand {
                center: 1.0,
                width: 0.05,
            },
            recover: PriorityBand {
                center: 1.0,
                width: 0.05,
            },
            partition: PriorityBand {
                center: 1.0,
                width: 0.05,
            },
            heal: PriorityBand {
                center: 1.0,
                width: 0.05,
            },
        }
    }
}

fn default_partition_band() -> PriorityBand {
    PriorityBand {
        center: 1.0,
        width: 0.05,
    }
}

fn default_heal_band() -> PriorityBand {
    PriorityBand {
        center: 1.0,
        width: 0.05,
    }
}

/// Configures probabilistic message delays ("purgatory").
#[derive(Debug, Clone, Deserialize)]
pub struct PurgatoryConfig {
    /// Probability that a remote ChannelSend is delayed. 0.0 = disabled.
    #[serde(default)]
    pub delay_probability: f64,
    /// (min_steps, max_steps) for log-uniform delay sampling.
    #[serde(default = "default_delay_range")]
    pub delay_duration_range: (i32, i32),
}

impl Default for PurgatoryConfig {
    fn default() -> Self {
        Self {
            delay_probability: 0.0,
            delay_duration_range: (5, 50),
        }
    }
}

fn default_delay_range() -> (i32, i32) {
    (5, 50)
}

impl SchedulePolicy {
    /// Sample a priority value for the given runnable category.
    pub fn sample(&self, rng: &mut impl Rng, cat: RunnableCategory) -> f64 {
        match self {
            SchedulePolicy::Fixed => match cat {
                RunnableCategory::Record => 0.5,
                RunnableCategory::Timer => 0.25,
                RunnableCategory::ChannelSend => 0.5,
                RunnableCategory::Crash | RunnableCategory::Recover => 1.0,
                RunnableCategory::Partition | RunnableCategory::Heal => 1.0,
            },
            SchedulePolicy::Shaped {
                alpha,
                beta,
                record,
                timer,
                channel_send,
                crash,
                recover,
                partition,
                heal,
            } => {
                let band = match cat {
                    RunnableCategory::Record => record,
                    RunnableCategory::Timer => timer,
                    RunnableCategory::ChannelSend => channel_send,
                    RunnableCategory::Crash => crash,
                    RunnableCategory::Recover => recover,
                    RunnableCategory::Partition => partition,
                    RunnableCategory::Heal => heal,
                };
                let dist = Beta::new(*alpha, *beta).unwrap();
                let sample = dist.sample(rng);
                (band.center + band.width * (2.0 * sample - 1.0)).clamp(0.0, 1.0)
            }
        }
    }
}

/// A node identifier pairing the role's NameId with a positional index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct NodeId {
    pub role: NameId,
    pub index: usize,
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}#{}", self.role, self.index)
    }
}

#[derive(Clone, Debug)]
pub struct LogEntry {
    pub node: NodeId,
    pub content: String,
    pub step: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TraceKind {
    Dispatch,
    Enter,
    Exit,
}

#[derive(Clone, Debug)]
pub struct TraceEntry {
    pub node: NodeId,
    pub function_name: String,
    pub kind: TraceKind,
    pub payload: Vec<String>,
    pub schedulable_count: usize,
    pub step: i32,
    pub trace_id: i64,
    pub causal_operation_id: Option<i64>,
}

/// Trait for handling Print statement output during execution.
pub trait Logger {
    fn log(&mut self, entry: LogEntry);
    fn log_trace(&mut self, _entry: TraceEntry) {}
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record<H: HashPolicy> {
    pub pc: Vertex,
    pub node: NodeId,
    pub origin_node: NodeId,
    pub continuation: Continuation<H>,
    pub env: Env<H>, // Just local env, node env is in State
    /// Original entry point for crash re-delivery.
    pub entry_pc: Vertex,
    /// Original local env for crash re-delivery.
    pub initial_env: Env<H>,
    pub priority: f64,
    /// Links this record (and its traces) back to the client operation that caused it.
    pub causal_operation_id: Option<i32>,
    pub trace_id: Option<i64>,
}

impl<H: HashPolicy> Record<H> {
    /// Reset pc and env to their initial values for crash re-delivery.
    pub fn reset(&mut self) {
        self.pc = self.entry_pc;
        self.env = self.initial_env.clone();
    }
}

impl<H: HashPolicy> Hash for Record<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        self.pc.hash(state);
        self.node.hash(state);
        self.origin_node.hash(state);
        self.continuation.hash(state);
        self.env.hash(state);
        self.entry_pc.hash(state);
        self.initial_env.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct CrashInfo<H: HashPolicy> {
    pub currently_crashed: OrdSet<NodeId>,
    pub queued_messages: Vector<(NodeId, Record<H>)>, // (dest_node, record)
    pub current_step: i32,
}

impl<H: HashPolicy> Hash for CrashInfo<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        // OrdSet has deterministic ordering, collect to Vec for safety
        let crashed_vec: Vec<_> = self.currently_crashed.iter().copied().collect();
        crashed_vec.hash(state);
        self.queued_messages.hash(state);
        self.current_step.hash(state);
    }
}

#[derive(Debug, Clone, Hash)]
pub struct ChannelState<H: HashPolicy> {
    pub buffer: Vector<Value<H>>,
    // We move Record out of Runnable and into Waiting.
    pub waiting_readers: Vector<(Record<H>, Lhs)>,
}

impl<H: HashPolicy> ChannelState<H> {
    pub fn new() -> Self {
        Self {
            buffer: Vector::new(),
            waiting_readers: Vector::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    Invocation,
    Response,
    Crash,
    Recover,
    Partition,
    Heal,
}

#[derive(Clone, Debug)]
pub struct Operation<H: HashPolicy> {
    pub client_id: i32,
    pub op_action: String,
    pub kind: OpKind,
    pub payload: Vec<Value<H>>,
    pub unique_id: i32,
    pub step: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Timer {
    pub pc: Vertex,
    pub node: NodeId,
    pub channel: ChannelId,
    pub priority: f64,
    pub label: Option<String>,
}

impl Hash for Timer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pc.hash(state);
        self.node.hash(state);
        self.channel.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Runnable<H: HashPolicy> {
    Timer(Timer),
    Record(Record<H>),
    ChannelSend {
        target: NodeId,
        channel: ChannelId,
        message: Value<H>,
        origin_node: NodeId,
        pc: Vertex,
        priority: f64,
    },
    Crash {
        node_id: NodeId,
        priority: f64,
    },
    Recover {
        node_id: NodeId,
        priority: f64,
    },
    Partition {
        partition_type: PartitionType,
        priority: f64,
    },
    Heal {
        priority: f64,
    },
}

impl<H: HashPolicy> Hash for Runnable<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        match self {
            Runnable::Timer(t) => {
                0u8.hash(state);
                t.hash(state);
            }
            Runnable::Record(r) => {
                1u8.hash(state);
                r.hash(state);
            }
            Runnable::ChannelSend {
                target,
                channel,
                message,
                origin_node,
                pc,
                ..
            } => {
                2u8.hash(state);
                target.hash(state);
                channel.hash(state);
                message.hash(state);
                origin_node.hash(state);
                pc.hash(state);
            }
            Runnable::Crash { node_id, .. } => {
                3u8.hash(state);
                node_id.hash(state);
            }
            Runnable::Recover { node_id, .. } => {
                4u8.hash(state);
                node_id.hash(state);
            }
            Runnable::Partition { partition_type, .. } => {
                5u8.hash(state);
                partition_type.hash(state);
            }
            Runnable::Heal { .. } => {
                6u8.hash(state);
            }
        }
    }
}

impl<H: HashPolicy> Runnable<H> {
    /// Get the node this runnable belongs to, if applicable.
    pub fn node(&self) -> Option<NodeId> {
        match self {
            Runnable::Timer(t) => Some(t.node),
            Runnable::Record(r) => Some(r.node),
            Runnable::ChannelSend { target, .. } => Some(*target),
            Runnable::Crash { node_id, .. } | Runnable::Recover { node_id, .. } => Some(*node_id),
            Runnable::Partition { .. } | Runnable::Heal { .. } => None,
        }
    }

    /// Get the PC (program counter vertex) for this runnable.
    pub fn pc(&self) -> Vertex {
        match self {
            Runnable::Timer(t) => t.pc,
            Runnable::Record(r) => r.pc,
            Runnable::ChannelSend { pc, .. } => *pc,
            Runnable::Crash { .. } | Runnable::Recover { .. } => usize::MAX,
            Runnable::Partition { .. } | Runnable::Heal { .. } => usize::MAX,
        }
    }

    /// Get the scheduling priority for this runnable.
    pub fn priority(&self) -> f64 {
        match self {
            Runnable::Record(r) => r.priority,
            Runnable::Timer(t) => t.priority,
            Runnable::ChannelSend { priority, .. } => *priority,
            Runnable::Crash { priority, .. } => *priority,
            Runnable::Recover { priority, .. } => *priority,
            Runnable::Partition { priority, .. } => *priority,
            Runnable::Heal { priority, .. } => *priority,
        }
    }
}

/// Result from scheduling a single runnable item.
#[derive(Debug)]
pub enum ScheduleResult<H: HashPolicy> {
    /// Nothing notable happened.
    None,
    /// A client operation completed.
    ClientOp(ClientOpResult<H>),
    /// A crash was executed on the given node.
    Crash { node_id: NodeId },
    /// A recovery was executed on the given node.
    Recover { node_id: NodeId },
    /// A labeled timer fired.
    TimerFired { node_id: NodeId, label: String },
    /// A network partition was activated.
    Partition { partition_type: PartitionType },
    /// A network partition was healed.
    Heal,
    /// A non-client Record runnable was executed (internal RPC delivery).
    RecordExecuted {
        entry_pc: Vertex,
        origin_node: NodeId,
        dest_node: NodeId,
    },
}

#[derive(Debug, Clone)]
pub struct State<H: HashPolicy> {
    pub nodes: Vector<Env<H>>, // Index is node_id.index
    pub local_queues: Vec<Vector<Runnable<H>>>,
    pub network_queue: Vector<Runnable<H>>,
    pub timer_queue: Vector<Runnable<H>>,
    /// Delayed runnables not yet schedulable. (release_step, runnable)
    pub purgatory: Vec<(i32, Runnable<H>)>,
    pub channels: ImHashMap<ChannelId, ChannelState<H>>,
    pub crash_info: CrashInfo<H>,
    pub partition_info: PartitionInfo<H>,
    /// Per-node durable storage that survives crashes. Keyed by node index.
    pub persisted_data: ImHashMap<usize, (TypeId, Value<H>)>,
    /// Set of (node_index, label) pairs for labeled timers that are allowed to fire.
    /// Only used when strict_timers is enabled in a plan execution.
    pub allowed_timers: HashSet<(usize, String)>,
    next_channel_id: usize,
    next_unique_id: usize,
}

impl<H: HashPolicy> State<H> {
    /// Create a new state. `role_node_counts` is a list of (role NameId, count) pairs.
    /// Nodes are laid out sequentially: all nodes of the first role, then all of the second, etc.
    pub fn new(role_node_counts: &[(NameId, usize)], node_slot_count: usize) -> Self {
        let mut nodes = Vector::new();
        let mut global_index = 0usize;
        for &(role, count) in role_node_counts {
            for _ in 0..count {
                let node_id = NodeId {
                    role,
                    index: global_index,
                };
                let mut env = Env::<H>::with_slots(node_slot_count);
                env.set(0, Value::<H>::node(node_id)); // Slot 0 = self
                nodes.push_back(env);
                global_index += 1;
            }
        }
        let num_nodes = nodes.len();
        Self {
            nodes,
            local_queues: (0..num_nodes).map(|_| Vector::new()).collect(),
            network_queue: Vector::new(),
            timer_queue: Vector::new(),
            purgatory: Vec::new(),
            channels: ImHashMap::new(),
            crash_info: CrashInfo {
                currently_crashed: OrdSet::new(),
                queued_messages: Vector::new(),
                current_step: 0,
            },
            partition_info: PartitionInfo::new(),
            persisted_data: ImHashMap::new(),
            allowed_timers: HashSet::new(),
            next_channel_id: 0,
            next_unique_id: 0,
        }
    }

    /// Dynamically add a new node with the given role.
    /// Returns the NodeId of the newly created node.
    pub fn add_node(&mut self, role: NameId, node_slot_count: usize) -> NodeId {
        let index = self.nodes.len();
        let node_id = NodeId { role, index };
        let mut env = Env::<H>::with_slots(node_slot_count);
        env.set(0, Value::<H>::node(node_id)); // Slot 0 = self
        self.nodes.push_back(env);
        self.local_queues.push(Vector::new());
        node_id
    }

    pub fn alloc_channel_id(&mut self) -> usize {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        id
    }

    pub fn alloc_unique_id(&mut self) -> usize {
        let id = self.next_unique_id;
        self.next_unique_id += 1;
        id
    }

    /// Auto-route a runnable to the correct queue.
    /// Records use origin_node == node to decide local vs network.
    /// Use `push_to_local()` for continuations/wakeups where the record
    /// is already delivered and being re-enqueued.
    pub fn push_runnable(&mut self, runnable: Runnable<H>) {
        match &runnable {
            Runnable::Timer(_) => self.timer_queue.push_back(runnable),
            Runnable::ChannelSend { .. } => self.network_queue.push_back(runnable),
            Runnable::Partition { .. } | Runnable::Heal { .. } => {
                self.network_queue.push_back(runnable)
            }
            Runnable::Crash { node_id, .. } | Runnable::Recover { node_id, .. } => {
                let idx = node_id.index;
                self.local_queues[idx].push_back(runnable);
            }
            Runnable::Record(r) => {
                if r.origin_node == r.node {
                    self.local_queues[r.node.index].push_back(runnable);
                } else {
                    self.network_queue.push_back(runnable);
                }
            }
        }
    }

    /// Force a runnable into a specific node's local queue.
    pub fn push_to_local(&mut self, node_index: usize, runnable: Runnable<H>) {
        self.local_queues[node_index].push_back(runnable);
    }

    /// True when all queue groups and purgatory are empty.
    pub fn all_queues_empty(&self) -> bool {
        self.network_queue.is_empty()
            && self.timer_queue.is_empty()
            && self.local_queues.iter().all(|q| q.is_empty())
            && self.purgatory.is_empty()
    }

    /// Move a runnable into purgatory, delaying it until `release_step`.
    pub fn delay_runnable(&mut self, release_step: i32, runnable: Runnable<H>) {
        self.purgatory.push((release_step, runnable));
    }

    /// Release purgatory items whose release_step <= current_step into their normal queues.
    pub fn release_from_purgatory(&mut self, current_step: i32) {
        let mut i = 0;
        while i < self.purgatory.len() {
            if self.purgatory[i].0 <= current_step {
                let (_, runnable) = self.purgatory.swap_remove(i);
                self.push_runnable(runnable);
                // Don't increment i — swap_remove moved the last element here
            } else {
                i += 1;
            }
        }
    }

    /// Total number of runnables across all queues.
    pub fn total_runnable_count(&self) -> usize {
        self.local_queues.iter().map(|q| q.len()).sum::<usize>()
            + self.network_queue.len()
            + self.timer_queue.len()
    }

    /// Compute state signature by aggregating component signatures.
    pub fn signature(&self) -> u64 {
        let mut h: u64 = 0;

        // Nodes: XOR of positioned Env signatures
        for (i, env) in self.nodes.iter().enumerate() {
            h ^= H::mix(env.sig, i as u32);
        }

        // Local queues
        let mut idx = 1000usize;
        for queue in &self.local_queues {
            for task in queue.iter() {
                h ^= H::mix(compute_hash(task), idx as u32);
                idx += 1;
            }
        }

        // Network queue
        for task in self.network_queue.iter() {
            h ^= H::mix(compute_hash(task), idx as u32);
            idx += 1;
        }

        // Timer queue
        for task in self.timer_queue.iter() {
            h ^= H::mix(compute_hash(task), idx as u32);
            idx += 1;
        }

        // Channels: Order-independent XOR
        for (chan_id, chan_state) in self.channels.iter() {
            let chan_hash = compute_hash(&(chan_id, chan_state));
            h ^= chan_hash;
        }

        // crash_info
        h ^= compute_hash(&self.crash_info);

        // partition_info
        h ^= compute_hash(&self.partition_info);

        // Persisted data: Order-independent XOR
        for (&node_idx, (tid, val)) in self.persisted_data.iter() {
            h ^= H::mix(compute_hash(&(node_idx, tid.0)), 2000) ^ H::mix(val.sig, 2001);
        }

        // Purgatory
        let mut purg_idx = 3000usize;
        for (release_step, task) in &self.purgatory {
            h ^= H::mix(compute_hash(&(release_step, task)), purg_idx as u32);
            purg_idx += 1;
        }

        h
    }
}

impl<H: HashPolicy> Hash for State<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        // Use precomputed signature for O(1) hashing
        self.signature().hash(state);
    }
}

/// Continuation representing what to do when an execution completes.
#[derive(Debug, Clone, PartialEq)]
pub enum Continuation<H: HashPolicy> {
    /// Node recovery continuation
    Recover,
    /// Async message delivery continuation
    Async {
        chan_id: ChannelId,
    },
    /// Client operation completion - returns data for caller to handle
    ClientOp {
        client_id: i32,
        op_name: String,
        unique_id: i32,
    },
    _Phantom(std::marker::PhantomData<H>),
}

impl<H: HashPolicy> Hash for Continuation<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        match self {
            Continuation::Recover => 0u8.hash(state),
            Continuation::Async { chan_id } => {
                1u8.hash(state);
                chan_id.hash(state);
            }
            Continuation::ClientOp {
                client_id,
                op_name,
                unique_id,
            } => {
                2u8.hash(state);
                client_id.hash(state);
                op_name.hash(state);
                unique_id.hash(state);
            }
            Continuation::_Phantom(_) => 3u8.hash(state),
        }
    }
}

/// Result returned when a ClientOp continuation completes.
#[derive(Debug, Clone)]
pub struct ClientOpResult<H: HashPolicy> {
    pub client_id: i32,
    pub op_name: String,
    pub unique_id: i32,
    pub value: Value<H>,
}

impl<H: HashPolicy> Continuation<H> {
    /// Execute the continuation and return any client operation result.
    pub fn call(self, state: &mut State<H>, val: Value<H>) -> Option<ClientOpResult<H>> {
        match self {
            Continuation::Recover => None,
            Continuation::Async { chan_id } => {
                let mut chan = match state.channels.get(&chan_id) {
                    Some(c) => c.clone(),
                    None => {
                        log::error!("Channel not found in async continuation: {}", chan_id.id);
                        return None;
                    }
                };
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    let mut node_env = state.nodes[reader.node.index].clone();
                    if let Err(e) = store(&lhs, val, &mut reader.env, &mut node_env) {
                        log::warn!("Store failed in async continuation: {}", e);
                    }
                    let node_index = reader.node.index;
                    state.nodes[node_index] = node_env;
                    state.push_to_local(node_index, Runnable::Record(reader));
                } else {
                    chan.buffer.push_back(val);
                }
                state.channels.insert(chan_id, chan);
                None
            }
            Continuation::ClientOp {
                client_id,
                op_name,
                unique_id,
            } => Some(ClientOpResult {
                client_id,
                op_name,
                unique_id,
                value: val,
            }),
            Continuation::_Phantom(_) => {
                unreachable!("_Phantom variant should never be constructed")
            }
        }
    }
}
