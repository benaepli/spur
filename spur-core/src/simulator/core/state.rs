use crate::analysis::resolver::NameId;
use crate::analysis::type_id::TypeId;
use crate::compiler::cfg::{Lhs, Vertex};
use crate::simulator::core::eval::store;
use crate::simulator::core::partition::{PartitionInfo, PartitionType};
use crate::simulator::core::steer_terms::Term;
use crate::simulator::core::values::{ChannelId, Env, LinkId, Value};
use crate::simulator::hash_utils::{HashPolicy, compute_hash};
use crate::simulator::rng::{Stream, StreamRng};
use crate::simulator::util_stats::DeliveryBias;
use imbl::{HashMap as ImHashMap, OrdSet, Vector};
use rand_distr::{Beta, Distribution};
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;

/// Channel storage. The hasher carries no per-process seed, so a session at
/// one seed replays the same schedule; the only place the map is iterated
/// combines entries with XOR, so iteration order is not observable either way.
pub type ChannelMap<H> = HashMap<ChannelId, ChannelState<H>, BuildHasherDefault<FxHasher>>;

/// Defines the priority band for a category of runnable.
#[derive(Debug, Clone, Deserialize, Serialize)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PurgatoryConfig {
    /// Probability that a remote ChannelSend is delayed. 0.0 = disabled.
    #[serde(default)]
    pub delay_probability: f64,
    /// (min_steps, max_steps) for log-uniform delay sampling.
    #[serde(default = "default_delay_range")]
    pub delay_duration_range: (i32, i32),
    /// When false, a send selected for a delay is enqueued undelayed if its
    /// destination node is currently crashed. The selection roll and the
    /// duration draw still happen, so the sends that stay eligible for a hold
    /// see the same random stream either way.
    #[serde(default = "default_hold_down_receivers")]
    pub hold_down_receivers: bool,
}

impl Default for PurgatoryConfig {
    fn default() -> Self {
        Self {
            delay_probability: 0.0,
            delay_duration_range: (5, 50),
            hold_down_receivers: true,
        }
    }
}

fn default_delay_range() -> (i32, i32) {
    (5, 50)
}

fn default_hold_down_receivers() -> bool {
    true
}

impl SchedulePolicy {
    /// Sample a priority value for the given runnable category.
    pub fn sample(&self, rng: &mut impl StreamRng, cat: RunnableCategory) -> f64 {
        rng.use_stream(match cat {
            RunnableCategory::Record | RunnableCategory::ChannelSend => Stream::MessagePriority,
            RunnableCategory::Timer => Stream::TimerPriority,
            RunnableCategory::Crash | RunnableCategory::Recover => Stream::FaultPriority,
            RunnableCategory::Partition | RunnableCategory::Heal => Stream::PartitionPriority,
        });
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
    /// FIFO link tag: `Some((link_id, seq))` if this RPC was sent through a
    /// FIFO link. Delivery is gated on `seq == link_deliver_seq[link_id]`.
    pub link_seq: Option<(LinkId, u32)>,
    /// Incarnation of `origin_node` at send time. Comparing it against the
    /// origin's incarnation at delivery says whether the sender restarted in
    /// between. Observation only.
    pub origin_incarnation: u32,
    /// Perturbations this message accumulated on its way to the receiver.
    /// Observation only, and deliberately excluded from `Hash` so state
    /// deduplication is unaffected.
    pub bias: DeliveryBias,
    /// Set when a timer firing woke this record: the vertex it resumes at,
    /// so the effect of the firing is measured on that first segment only.
    /// Observation only, excluded from `Hash`.
    pub timer_entry: Option<Vertex>,
    /// Position of this send among the origin's sends, so a record can be
    /// told apart from sends issued before the origin's last handler entry.
    /// Excluded from `Hash`: it describes when the record was made, not
    /// what it does.
    pub send_ordinal: u32,
    /// The receiver's state token when the record was sent, so a delivery
    /// can tell whether the receiver moved on in between. Excluded from
    /// `Hash` for the same reason.
    pub receiver_token_at_send: u64,
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
        self.link_seq.hash(state);
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

/// A reader blocked on a channel, paired with the destination the delivered
/// value is stored into.
///
/// Held behind a shared pointer because the persistent vector sizes its
/// storage by the element: with `Record` inline the element is hundreds of
/// bytes, so a channel holding a single blocked reader owns a multi-kilobyte
/// heap chunk that is reallocated and copied on every copy-on-write. At one
/// word the same readers fit in the vector's inline storage.
pub type WaitingReader<H> = Arc<(Record<H>, Lhs)>;

#[derive(Debug, Clone, Hash)]
pub struct ChannelState<H: HashPolicy> {
    pub buffer: Vector<Value<H>>,
    // We move Record out of Runnable and into Waiting.
    pub waiting_readers: Vector<WaitingReader<H>>,
}

impl<H: HashPolicy> ChannelState<H> {
    pub fn new() -> Self {
        Self {
            buffer: Vector::new(),
            waiting_readers: Vector::new(),
        }
    }

    pub fn push_waiting_reader(&mut self, record: Record<H>, lhs: Lhs) {
        self.waiting_readers.push_back(Arc::new((record, lhs)));
    }

    /// Takes the longest-waiting reader, copying it only when the channel
    /// state it came from is still shared.
    pub fn pop_waiting_reader(&mut self) -> Option<(Record<H>, Lhs)> {
        self.waiting_readers
            .pop_front()
            .map(|entry| Arc::try_unwrap(entry).unwrap_or_else(|shared| (*shared).clone()))
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
    /// A node's timer fired. The payload carries the node and the label;
    /// the row's client id is the node and its action ends in `/label`.
    TimerFired,
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
    #[allow(dead_code)]
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
    Partition {
        #[allow(dead_code)]
        partition_type: PartitionType,
    },
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
    // `nodes` and the three run queues are owned exclusively by the run that
    // created them; a `State` is never cloned or snapshotted, so plain `Vec`s
    // are the right representation.
    pub nodes: Vec<Env<H>>, // Index is node_id.index
    /// How many times each node has come back from a crash, indexed like
    /// `nodes`. Observation only: excluded from `signature()` so it cannot
    /// change deduplication or scheduling.
    pub incarnations: Vec<u32>,
    pub local_queues: Vec<Vec<Runnable<H>>>,
    pub network_queue: Vec<Runnable<H>>,
    pub timer_queue: Vec<Runnable<H>>,
    /// Delayed runnables not yet schedulable. (release_step, runnable)
    pub purgatory: Vec<(i32, Runnable<H>)>,
    /// Owned outright like `nodes` and the run queues, so a channel is updated
    /// through the map rather than copied out and put back.
    pub channels: ChannelMap<H>,
    pub crash_info: CrashInfo<H>,
    pub partition_info: PartitionInfo<H>,
    /// Per-node durable storage that survives crashes. Keyed by node index.
    pub persisted_data: ImHashMap<usize, (TypeId, Value<H>)>,
    /// Set of (node_index, label) pairs for labeled timers that are allowed to fire.
    /// Only used when strict_timers is enabled in a plan execution.
    pub allowed_timers: HashSet<(usize, String)>,
    next_channel_id: usize,
    next_unique_id: usize,
    next_link_id: usize,
    /// Per-link metadata: (sender, receiver). Used for debugging/visualisation.
    pub link_meta: ImHashMap<LinkId, (NodeId, NodeId)>,
    /// Per-link sender-side counter: the next sequence number to assign on send.
    pub link_send_seq: ImHashMap<LinkId, u32>,
    /// Per-link receiver-side counter: the next sequence number eligible for delivery.
    /// A FIFO-tagged runnable with `(link_id, seq)` is only deliverable when
    /// `seq == link_deliver_seq[link_id]`. Bumped on successful delivery.
    pub link_deliver_seq: ImHashMap<LinkId, u32>,
    /// Timer firings this run and what they changed. Observation only,
    /// excluded from `signature()`.
    pub timer_stats: TimerRunStats,
    /// Consecutive inert timer firings per (node index, resume vertex),
    /// reset when a firing at that vertex changes the node's state.
    timer_inert_streaks: Vec<(usize, Vertex, u32)>,
    /// Per-node send bookkeeping the score's predicates read; indexed like
    /// `nodes`. Kept exact by the queue hooks below and excluded from
    /// `signature()`, so it cannot change deduplication.
    pub send_ledger: Vec<SendLedger>,
    /// Remote records in the network queue whose origin has restarted since
    /// sending them: the sum over nodes of `net_records - net_fresh`.
    pub net_stale_records: u32,
    /// Remote records in the network queue whose origin has another role
    /// than the receiver, which is how a client request looks to a server.
    pub net_requests: u32,
}

/// What woke the handler a node ran last.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HandlerTrigger {
    #[default]
    None,
    Timer,
    Delivery,
}

/// One node's send bookkeeping. `issued` numbers the node's sends; `floor`
/// is `issued` at the node's last handler entry, so a record with
/// `send_ordinal >= floor` was sent by that handler. `in_flight` counts the
/// node's remote records and channel sends in the network queue or held by
/// a delay; `recent` the subset sent by the last handler. `net_records` and
/// `net_fresh` count the node's remote records in the network queue and the
/// ones whose origin incarnation is still current. `crash_pending` counts
/// the crashes of this node waiting in its local queue. `entries` counts the
/// handler entries the node has taken in this run and `entries_at_restart` its
/// value when the node last came back from a crash, so their difference is how
/// far the node has moved past that restart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SendLedger {
    pub issued: u32,
    pub floor: u32,
    pub trigger: HandlerTrigger,
    pub in_flight: u32,
    pub recent: u32,
    pub net_records: u32,
    pub net_fresh: u32,
    pub crash_pending: u32,
    pub entries: u32,
    pub entries_at_restart: u32,
}

/// Per-run totals of timer firings that woke a waiting record, split by
/// whether a delivery to the node was pending when it fired, and whether the
/// woken segment changed the node's state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimerRunStats {
    pub fired: u32,
    pub acted: u32,
    pub inflight_fired: u32,
    pub inflight_acted: u32,
    pub idle_fired: u32,
    pub idle_acted: u32,
    pub max_inert_streak: u32,
}

impl<H: HashPolicy> State<H> {
    /// Create a new state. `role_node_counts` is a list of (role NameId, count) pairs.
    /// Nodes are laid out sequentially: all nodes of the first role, then all of the second, etc.
    pub fn new(role_node_counts: &[(NameId, usize)], node_slot_count: usize) -> Self {
        let mut nodes = Vec::new();
        let mut global_index = 0usize;
        for &(role, count) in role_node_counts {
            for _ in 0..count {
                let node_id = NodeId {
                    role,
                    index: global_index,
                };
                let mut env = Env::<H>::with_slots(node_slot_count);
                env.set(0, Value::<H>::node(node_id)); // Slot 0 = self
                nodes.push(env);
                global_index += 1;
            }
        }
        let num_nodes = nodes.len();
        Self {
            nodes,
            incarnations: vec![0; num_nodes],
            local_queues: (0..num_nodes).map(|_| Vec::new()).collect(),
            network_queue: Vec::new(),
            timer_queue: Vec::new(),
            purgatory: Vec::new(),
            channels: ChannelMap::default(),
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
            next_link_id: 0,
            link_meta: ImHashMap::new(),
            link_send_seq: ImHashMap::new(),
            link_deliver_seq: ImHashMap::new(),
            timer_stats: TimerRunStats::default(),
            timer_inert_streaks: Vec::new(),
            send_ledger: vec![SendLedger::default(); num_nodes],
            net_stale_records: 0,
            net_requests: 0,
        }
    }

    /// Messages from other nodes addressed to `node` that are queued or held
    /// by a delay and not yet delivered.
    pub fn pending_deliveries_to(&self, node: NodeId) -> usize {
        let to_node = |r: &Runnable<H>| match r {
            Runnable::Record(rec) => rec.node == node && rec.origin_node != node,
            Runnable::ChannelSend {
                origin_node, target, ..
            } => *target == node && *origin_node != node,
            _ => false,
        };
        self.network_queue.iter().filter(|r| to_node(r)).count()
            + self.purgatory.iter().filter(|(_, r)| to_node(r)).count()
    }

    /// Consecutive inert timer firings at `pc` on node `node` so far in
    /// this run.
    pub fn timer_inert_streak(&self, node: usize, pc: Vertex) -> u32 {
        self.timer_inert_streaks
            .iter()
            .find(|(n, p, _)| *n == node && *p == pc)
            .map(|(_, _, s)| *s)
            .unwrap_or(0)
    }

    /// Account one timer firing that woke a record at `pc` on node `node`.
    pub fn note_timer_effect(&mut self, node: usize, pc: Vertex, inflight: bool, acted: bool) {
        let t = &mut self.timer_stats;
        t.fired += 1;
        if acted {
            t.acted += 1;
        }
        if inflight {
            t.inflight_fired += 1;
            if acted {
                t.inflight_acted += 1;
            }
        } else {
            t.idle_fired += 1;
            if acted {
                t.idle_acted += 1;
            }
        }
        let entry = match self
            .timer_inert_streaks
            .iter_mut()
            .find(|(n, p, _)| *n == node && *p == pc)
        {
            Some(e) => e,
            None => {
                self.timer_inert_streaks.push((node, pc, 0));
                self.timer_inert_streaks.last_mut().expect("just pushed")
            }
        };
        entry.2 = if acted { 0 } else { entry.2 + 1 };
        if entry.2 > t.max_inert_streak {
            t.max_inert_streak = entry.2;
        }
    }

    pub fn alloc_link_id(&mut self) -> LinkId {
        let id = LinkId(self.next_link_id);
        self.next_link_id += 1;
        id
    }

    /// Dynamically add a new node with the given role.
    /// Returns the NodeId of the newly created node.
    pub fn add_node(&mut self, role: NameId, node_slot_count: usize) -> NodeId {
        let index = self.nodes.len();
        let node_id = NodeId { role, index };
        let mut env = Env::<H>::with_slots(node_slot_count);
        env.set(0, Value::<H>::node(node_id)); // Slot 0 = self
        self.nodes.push(env);
        self.incarnations.push(0);
        self.local_queues.push(Vec::new());
        self.send_ledger.push(SendLedger::default());
        node_id
    }

    /// How many times `node` has recovered from a crash so far.
    #[inline]
    pub fn incarnation(&self, node: NodeId) -> u32 {
        self.incarnations.get(node.index).copied().unwrap_or(0)
    }

    /// A value that changes whenever `node`'s persistent state is written.
    /// Slot storage is copy-on-write and the caller's copy is shared with this
    /// one, so the first write after a read moves the allocation.
    #[inline]
    /// Observation handle for "was this node written". Compare two values
    /// taken around a step: unequal means the node's env was written. A raw
    /// slots pointer cannot answer this, since `EcoVec` shares its buffer
    /// across clones and reallocates only when copy-on-write fires.
    pub fn node_state_token(&self, node: NodeId) -> u64 {
        self.nodes[node.index].writes
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
            Runnable::Timer(_) => self.timer_queue.push(runnable),
            Runnable::ChannelSend { .. } => {
                self.flight_enter(&runnable);
                self.network_queue.push(runnable)
            }
            Runnable::Partition { .. } | Runnable::Heal { .. } => {
                self.network_queue.push(runnable)
            }
            Runnable::Crash { node_id, .. } => {
                let idx = node_id.index;
                if let Some(l) = self.send_ledger.get_mut(idx) {
                    l.crash_pending += 1;
                }
                self.local_queues[idx].push(runnable);
            }
            Runnable::Recover { node_id, .. } => {
                let idx = node_id.index;
                self.local_queues[idx].push(runnable);
            }
            Runnable::Record(r) => {
                if r.origin_node == r.node {
                    self.local_queues[r.node.index].push(runnable);
                } else {
                    self.flight_enter(&runnable);
                    self.net_enter(&runnable);
                    self.network_queue.push(runnable);
                }
            }
        }
    }

    /// Remove and return the network queue entry at `idx`, keeping the
    /// ledger exact.
    pub fn take_network(&mut self, idx: usize) -> Runnable<H> {
        let r = self.network_queue.remove(idx);
        self.flight_leave(&r);
        self.net_leave(&r);
        r
    }

    /// Remove and return the entry at `idx` of `node`'s local queue, keeping
    /// the ledger exact.
    pub fn take_local(&mut self, node: usize, idx: usize) -> Runnable<H> {
        let r = self.local_queues[node].remove(idx);
        if let Runnable::Crash { node_id, .. } = &r
            && let Some(l) = self.send_ledger.get_mut(node_id.index)
        {
            l.crash_pending = l.crash_pending.saturating_sub(1);
        }
        r
    }

    /// The origin of a remote message, or `None` for anything else.
    fn remote_origin(r: &Runnable<H>) -> Option<usize> {
        match r {
            Runnable::Record(rec) if rec.origin_node != rec.node => Some(rec.origin_node.index),
            Runnable::ChannelSend {
                origin_node, target, ..
            } if origin_node != target => Some(origin_node.index),
            _ => None,
        }
    }

    /// A remote message became undelivered: queued or held by a delay.
    pub fn flight_enter(&mut self, r: &Runnable<H>) {
        let Some(origin) = Self::remote_origin(r) else { return };
        let Some(l) = self.send_ledger.get_mut(origin) else { return };
        l.in_flight += 1;
        if let Runnable::Record(rec) = r
            && rec.send_ordinal >= l.floor
        {
            l.recent += 1;
        }
    }

    /// A remote message stopped being undelivered: delivered, dropped, or
    /// set aside by a crash or a partition.
    pub fn flight_leave(&mut self, r: &Runnable<H>) {
        let Some(origin) = Self::remote_origin(r) else { return };
        let Some(l) = self.send_ledger.get_mut(origin) else { return };
        l.in_flight = l.in_flight.saturating_sub(1);
        if let Runnable::Record(rec) = r
            && rec.send_ordinal >= l.floor
        {
            l.recent = l.recent.saturating_sub(1);
        }
    }

    /// A remote record entered the network queue.
    pub fn net_enter(&mut self, r: &Runnable<H>) {
        let Runnable::Record(rec) = r else { return };
        if rec.origin_node == rec.node {
            return;
        }
        let fresh = rec.origin_incarnation == self.incarnation(rec.origin_node);
        let Some(l) = self.send_ledger.get_mut(rec.origin_node.index) else { return };
        l.net_records += 1;
        if fresh {
            l.net_fresh += 1;
        } else {
            self.net_stale_records += 1;
        }
        if rec.origin_node.role != rec.node.role {
            self.net_requests += 1;
        }
    }

    /// A remote record left the network queue.
    pub fn net_leave(&mut self, r: &Runnable<H>) {
        let Runnable::Record(rec) = r else { return };
        if rec.origin_node == rec.node {
            return;
        }
        let fresh = rec.origin_incarnation == self.incarnation(rec.origin_node);
        let Some(l) = self.send_ledger.get_mut(rec.origin_node.index) else { return };
        l.net_records = l.net_records.saturating_sub(1);
        if fresh {
            l.net_fresh = l.net_fresh.saturating_sub(1);
        } else {
            self.net_stale_records = self.net_stale_records.saturating_sub(1);
        }
        if rec.origin_node.role != rec.node.role {
            self.net_requests = self.net_requests.saturating_sub(1);
        }
    }

    /// Number the next send of `node`.
    pub fn next_send_ordinal(&mut self, node: NodeId) -> u32 {
        match self.send_ledger.get_mut(node.index) {
            Some(l) => {
                let o = l.issued;
                l.issued += 1;
                o
            }
            None => 0,
        }
    }

    /// `node` is entering a handler woken by `trigger`: the sends it issues
    /// from here on are the ones a crash of the node would strand.
    pub fn note_handler_entry(&mut self, node: usize, trigger: HandlerTrigger) {
        if let Some(l) = self.send_ledger.get_mut(node) {
            l.floor = l.issued;
            l.recent = 0;
            l.trigger = trigger;
            if trigger != HandlerTrigger::None {
                l.entries = l.entries.saturating_add(1);
            }
        }
    }

    /// `node` came back from a crash: every remote record it sent before
    /// now carries an incarnation that no longer exists.
    pub fn note_incarnation_bump(&mut self, node: usize) {
        if let Some(l) = self.send_ledger.get_mut(node) {
            self.net_stale_records += l.net_fresh;
            l.net_fresh = 0;
            l.entries_at_restart = l.entries;
        }
    }

    /// Handler entries `node` has taken since it last came back from a crash,
    /// or since the start of the run if it never crashed.
    pub fn entries_since_restart(&self, node: usize) -> u32 {
        match self.send_ledger.get(node) {
            Some(l) => l.entries.saturating_sub(l.entries_at_restart),
            None => 0,
        }
    }

    /// Which crash term a crash of `node` satisfies right now, if any.
    pub fn crash_after_sends_term(&self, node: usize) -> Option<Term> {
        let l = self.send_ledger.get(node)?;
        if l.recent == 0 {
            return None;
        }
        match l.trigger {
            HandlerTrigger::Timer => Some(Term::CrashAfterTimerSends),
            HandlerTrigger::Delivery => Some(Term::CrashAfterDeliverySends),
            HandlerTrigger::None => None,
        }
    }

    /// The record's origin restarted since it was sent and its receiver has
    /// written state since.
    pub fn stale_late(&self, rec: &Record<H>) -> bool {
        rec.origin_node != rec.node
            && rec.origin_incarnation != self.incarnation(rec.origin_node)
            && self.node_state_token(rec.node) != rec.receiver_token_at_send
    }

    /// The record is a request across roles while some record from a
    /// restarted origin is still undelivered.
    pub fn request_before_stale(&self, rec: &Record<H>) -> bool {
        rec.origin_node != rec.node
            && rec.origin_node.role != rec.node.role
            && self.net_stale_records > 0
    }

    /// The predicates true of `r`, one bit per `Term::index`.
    pub fn term_mask(&self, r: &Runnable<H>) -> u8 {
        match r {
            Runnable::Crash { node_id, .. } => match self.crash_after_sends_term(node_id.index) {
                Some(t) => 1 << t.index(),
                None => 0,
            },
            Runnable::Record(rec) => {
                let mut m = 0;
                if self.stale_late(rec) {
                    m |= 1 << Term::StaleLate.index();
                }
                if self.request_before_stale(rec) {
                    m |= 1 << Term::RequestBeforeStale.index();
                }
                m
            }
            _ => 0,
        }
    }

    /// Force a runnable into a specific node's local queue.
    pub fn push_to_local(&mut self, node_index: usize, runnable: Runnable<H>) {
        self.local_queues[node_index].push(runnable);
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
        self.flight_enter(&runnable);
        self.purgatory.push((release_step, runnable));
    }

    /// Release purgatory items whose release_step <= current_step into their normal queues.
    pub fn release_from_purgatory(&mut self, current_step: i32) {
        let mut i = 0;
        while i < self.purgatory.len() {
            if self.purgatory[i].0 <= current_step {
                let (_, runnable) = self.purgatory.swap_remove(i);
                self.flight_leave(&runnable);
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

        // FIFO link state: order-independent XOR
        for (link_id, (sender, receiver)) in self.link_meta.iter() {
            h ^= compute_hash(&(link_id.0, sender, receiver));
        }
        for (link_id, &seq) in self.link_send_seq.iter() {
            h ^= H::mix(compute_hash(&(link_id.0, seq)), 3000);
        }
        for (link_id, &seq) in self.link_deliver_seq.iter() {
            h ^= H::mix(compute_hash(&(link_id.0, seq)), 3001);
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
                let chan = match state.channels.get_mut(&chan_id) {
                    Some(c) => c,
                    None => {
                        log::error!("Channel not found in async continuation: {}", chan_id.id);
                        return None;
                    }
                };
                match chan.pop_waiting_reader() {
                    None => chan.buffer.push_back(val),
                    Some((mut reader, lhs)) => {
                        let node_index = reader.node.index;
                        let mut node_env = state.nodes[node_index].clone();
                        if let Err(e) = store(&lhs, val, &mut reader.env, &mut node_env) {
                            log::warn!("Store failed in async continuation: {}", e);
                        }
                        state.nodes[node_index] = node_env;
                        state.push_to_local(node_index, Runnable::Record(reader));
                    }
                }
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

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::simulator::hash_utils::WithHashing;

    /// The persistent vector holding a channel's blocked readers sizes its
    /// heap chunk by the element type, so a wide element costs an allocation
    /// and a copy per copy-on-write even when one reader is waiting. Keeping
    /// the element one word wide keeps those readers in inline storage; the
    /// `Record` assertion is the tripwire that says why the indirection is
    /// there.
    #[test]
    fn waiting_reader_stays_narrow() {
        assert_eq!(
            std::mem::size_of::<WaitingReader<WithHashing>>(),
            std::mem::size_of::<usize>()
        );
        assert!(
            std::mem::size_of::<(Record<WithHashing>, Lhs)>()
                > 8 * std::mem::size_of::<WaitingReader<WithHashing>>(),
            "a reader narrow enough to sit in the vector inline no longer needs the indirection"
        );
    }

    /// A hash-array-mapped trie allocates one node per branch at the full
    /// branching factor of 32 entries, whatever the occupancy, so a wide entry
    /// puts every node above the allocator's 1032-byte fast path. Channels are
    /// updated in place through an owned map instead; the assertion is the
    /// tripwire that says why a persistent map is the wrong container here.
    #[test]
    fn channel_entry_is_too_wide_for_a_trie_node() {
        const TRIE_BRANCHING_FACTOR: usize = 32;
        const SMALL_ALLOCATION_LIMIT: usize = 1032;
        assert!(
            TRIE_BRANCHING_FACTOR
                * std::mem::size_of::<(ChannelId, ChannelState<WithHashing>)>()
                > SMALL_ALLOCATION_LIMIT,
            "a channel entry narrow enough to fit a trie node no longer forces a large allocation"
        );
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::*;
    use crate::analysis::resolver::NameId;
    use crate::simulator::hash_utils::NoHashing;

    const SERVER: NameId = NameId(0);
    const CLIENT: NameId = NameId(1);

    fn node(role: NameId, index: usize) -> NodeId {
        NodeId { role, index }
    }

    /// Three servers and one client; the client is node 3.
    fn state() -> State<NoHashing> {
        State::new(&[(SERVER, 3), (CLIENT, 1)], 2)
    }

    fn record(state: &mut State<NoHashing>, origin: NodeId, dest: NodeId) -> Record<NoHashing> {
        let env = Env::<NoHashing>::with_slots(1);
        Record {
            pc: 0,
            node: dest,
            origin_node: origin,
            continuation: Continuation::Recover,
            entry_pc: 0,
            initial_env: env.clone(),
            env,
            priority: 0.5,
            causal_operation_id: None,
            trace_id: None,
            link_seq: None,
            origin_incarnation: state.incarnation(origin),
            bias: DeliveryBias::NONE,
            timer_entry: None,
            send_ordinal: state.next_send_ordinal(origin),
            receiver_token_at_send: state.node_state_token(dest),
        }
    }

    /// The ledger recomputed from the queues, the oracle the hooks are held to.
    fn recount(state: &State<NoHashing>) -> (Vec<SendLedger>, u32, u32) {
        let mut ledgers: Vec<SendLedger> = state
            .send_ledger
            .iter()
            .map(|l| SendLedger {
                in_flight: 0,
                recent: 0,
                net_records: 0,
                net_fresh: 0,
                crash_pending: 0,
                ..*l
            })
            .collect();
        let mut stale = 0;
        let mut requests = 0;
        let mut count = |r: &Runnable<NoHashing>, in_network: bool| match r {
            Runnable::Record(rec) if rec.origin_node != rec.node => {
                let l = &mut ledgers[rec.origin_node.index];
                l.in_flight += 1;
                if rec.send_ordinal >= l.floor {
                    l.recent += 1;
                }
                if in_network {
                    l.net_records += 1;
                    if rec.origin_incarnation == state.incarnation(rec.origin_node) {
                        l.net_fresh += 1;
                    } else {
                        stale += 1;
                    }
                    if rec.origin_node.role != rec.node.role {
                        requests += 1;
                    }
                }
            }
            Runnable::ChannelSend {
                origin_node, target, ..
            } if origin_node != target => ledgers[origin_node.index].in_flight += 1,
            _ => {}
        };
        for r in &state.network_queue {
            count(r, true);
        }
        for (_, r) in &state.purgatory {
            count(r, false);
        }
        for (n, q) in state.local_queues.iter().enumerate() {
            ledgers[n].crash_pending =
                q.iter().filter(|r| matches!(r, Runnable::Crash { .. })).count() as u32;
        }
        (ledgers, stale, requests)
    }

    fn assert_exact(state: &State<NoHashing>, when: &str) {
        let (ledgers, stale, requests) = recount(state);
        assert_eq!(state.send_ledger, ledgers, "ledger drifted {when}");
        assert_eq!(state.net_stale_records, stale, "stale count drifted {when}");
        assert_eq!(state.net_requests, requests, "request count drifted {when}");
    }

    #[test]
    fn crash_after_sends_keys_on_last_trigger() {
        let mut st = state();
        let a = node(SERVER, 0);
        let b = node(SERVER, 1);
        assert_eq!(st.crash_after_sends_term(0), None);
        st.note_handler_entry(0, HandlerTrigger::Timer);
        let r1 = record(&mut st, a, b);
        let r2 = record(&mut st, a, b);
        st.push_runnable(Runnable::Record(r1));
        st.push_runnable(Runnable::Record(r2));
        assert_eq!(st.send_ledger[0].recent, 2);
        assert_eq!(st.crash_after_sends_term(0), Some(Term::CrashAfterTimerSends));
        st.take_network(0);
        assert_eq!(st.send_ledger[0].recent, 1);
        st.note_handler_entry(0, HandlerTrigger::Delivery);
        assert_eq!(st.send_ledger[0].recent, 0);
        assert_eq!(st.crash_after_sends_term(0), None);
        let r3 = record(&mut st, a, b);
        st.push_runnable(Runnable::Record(r3));
        assert_eq!(st.crash_after_sends_term(0), Some(Term::CrashAfterDeliverySends));
        st.note_handler_entry(0, HandlerTrigger::None);
        let r4 = record(&mut st, a, b);
        st.push_runnable(Runnable::Record(r4));
        assert_eq!(st.crash_after_sends_term(0), None, "sends with no trigger name no term");
        assert_exact(&st, "after the trigger sequence");
    }

    #[test]
    fn stale_late_needs_both_conjuncts() {
        let mut st = state();
        let a = node(SERVER, 0);
        let b = node(SERVER, 1);
        let r = record(&mut st, a, b);
        assert!(!st.stale_late(&r), "fresh origin, receiver unchanged");
        st.incarnations[0] += 1;
        assert!(!st.stale_late(&r), "stale origin but the receiver has not moved on");
        let mut env = st.nodes[1].clone();
        env.set(0, Value::<NoHashing>::int(7));
        st.nodes[1] = env;
        assert!(st.stale_late(&r), "stale origin and a receiver that wrote state");
        st.incarnations[0] -= 1;
        assert!(!st.stale_late(&r), "receiver moved on but the origin is current");
        let own = record(&mut st, b, b);
        st.incarnations[1] += 1;
        assert!(!st.stale_late(&own), "a record to itself is never a stale delivery");
    }

    #[test]
    fn request_before_stale_reads_the_global_count() {
        let mut st = state();
        let client = node(CLIENT, 3);
        let a = node(SERVER, 0);
        let b = node(SERVER, 1);
        let req = record(&mut st, client, a);
        assert!(!st.request_before_stale(&req));
        let msg = record(&mut st, a, b);
        st.incarnations[0] += 1;
        st.push_runnable(Runnable::Record(msg));
        assert_eq!(st.net_stale_records, 1);
        assert!(st.request_before_stale(&req));
        let peer = record(&mut st, a, b);
        assert!(!st.request_before_stale(&peer), "a message between servers is not a request");
        st.take_network(0);
        assert!(!st.request_before_stale(&req));
    }

    #[test]
    fn term_mask_names_each_predicate_bit() {
        let mut st = state();
        let a = node(SERVER, 0);
        let b = node(SERVER, 1);
        let crash = Runnable::<NoHashing>::Crash {
            node_id: a,
            priority: 1.0,
        };
        assert_eq!(st.term_mask(&crash), 0);
        st.note_handler_entry(0, HandlerTrigger::Timer);
        let r = record(&mut st, a, b);
        st.push_runnable(Runnable::Record(r));
        assert_eq!(st.term_mask(&crash), 1 << Term::CrashAfterTimerSends.index());
        let held = record(&mut st, a, b);
        st.incarnations[0] += 1;
        let mut env = st.nodes[1].clone();
        env.set(0, Value::<NoHashing>::int(1));
        st.nodes[1] = env;
        assert_eq!(
            st.term_mask(&Runnable::Record(held)),
            1 << Term::StaleLate.index()
        );
    }

    #[test]
    fn stale_accounting_is_exact_under_events() {
        let mut st = state();
        let a = node(SERVER, 0);
        let b = node(SERVER, 1);
        let c = node(SERVER, 2);
        let client = node(CLIENT, 3);
        st.note_handler_entry(0, HandlerTrigger::Delivery);
        for _ in 0..3 {
            let r = record(&mut st, a, b);
            st.push_runnable(Runnable::Record(r));
        }
        let req = record(&mut st, client, a);
        st.push_runnable(Runnable::Record(req));
        let delayed = record(&mut st, b, c);
        st.delay_runnable(5, Runnable::Record(delayed));
        st.push_runnable(Runnable::ChannelSend {
            target: c,
            channel: ChannelId { node: c, id: 0 },
            message: Value::<NoHashing>::unit(),
            origin_node: a,
            pc: 0,
            priority: 0.5,
        });
        st.push_runnable(Runnable::Crash {
            node_id: a,
            priority: 1.0,
        });
        assert_exact(&st, "after pushes");
        assert_eq!(st.send_ledger[0].recent, 3);
        assert_eq!(st.net_requests, 1);

        let taken = st.take_network(1);
        assert!(matches!(taken, Runnable::Record(_)));
        assert_exact(&st, "after a delivery");

        st.incarnations[0] += 1;
        st.note_incarnation_bump(0);
        assert_exact(&st, "after the origin restarted");
        assert_eq!(st.net_stale_records, 2);

        st.release_from_purgatory(5);
        assert_exact(&st, "after a purgatory release");

        let crash = st.take_local(0, 0);
        assert!(matches!(crash, Runnable::Crash { .. }));
        assert_eq!(st.send_ledger[0].crash_pending, 0);
        assert_exact(&st, "after the crash was taken");

        st.push_runnable(Runnable::Crash {
            node_id: b,
            priority: 1.0,
        });
        st.push_runnable(Runnable::Crash {
            node_id: b,
            priority: 1.0,
        });
        assert_eq!(st.send_ledger[1].crash_pending, 2);
        assert_exact(&st, "with two crashes pending");
    }

    #[test]
    fn send_ordinals_count_from_the_handler_floor() {
        let mut st = state();
        let a = node(SERVER, 0);
        let b = node(SERVER, 1);
        let early = record(&mut st, a, b);
        st.note_handler_entry(0, HandlerTrigger::Timer);
        let late = record(&mut st, a, b);
        st.push_runnable(Runnable::Record(early));
        st.push_runnable(Runnable::Record(late));
        assert_eq!(st.send_ledger[0].in_flight, 2);
        assert_eq!(st.send_ledger[0].recent, 1, "only the send after the entry counts");
        st.take_network(0);
        assert_eq!(st.send_ledger[0].recent, 1, "removing the early send leaves recent alone");
        assert_exact(&st, "after removing the early send");
    }
}
