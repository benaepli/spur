use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Lhs, Vertex};
use crate::simulator::core::eval::store;
use crate::simulator::core::values::{ChannelId, Env, Value};
use crate::simulator::hash_utils::{HashPolicy, compute_hash};
use imbl::{HashMap as ImHashMap, OrdSet, Vector};
use std::hash::{Hash, Hasher};

use serde::Serialize;

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

/// Trait for handling Print statement output during execution.
pub trait Logger {
    fn log(&mut self, entry: LogEntry);
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record<H: HashPolicy> {
    pub pc: Vertex,
    pub node: NodeId,
    pub origin_node: NodeId,
    pub continuation: Continuation<H>,
    pub env: Env<H>, // Just local env, node env is in State
    pub x: f64,
    pub policy: UpdatePolicy,
}

impl<H: HashPolicy> Hash for Record<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        self.pc.hash(state);
        self.node.hash(state);
        self.origin_node.hash(state);
        self.continuation.hash(state);
        self.env.hash(state);
        self.policy.hash(state);
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum UpdatePolicy {
    Identity,
    Halve,
}

impl UpdatePolicy {
    pub fn update(&self, x: f64) -> f64 {
        match self {
            UpdatePolicy::Identity => x,
            UpdatePolicy::Halve => x / 2.0,
        }
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
}

#[derive(Clone, Debug)]
pub struct Operation<H: HashPolicy> {
    pub client_id: i32,
    pub op_action: String,
    pub kind: OpKind,
    pub payload: Vec<Value<H>>,
    pub unique_id: i32,
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Timer {
    pub pc: Vertex,
    pub node: NodeId,
    pub channel: ChannelId,
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
        x: f64,
        policy: UpdatePolicy,
        pc: Vertex,
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
                x,
                policy,
                pc,
            } => {
                2u8.hash(state);
                target.hash(state);
                channel.hash(state);
                message.hash(state);
                origin_node.hash(state);
                x.to_bits().hash(state); // f64 hash via bits
                policy.hash(state);
                pc.hash(state);
            }
        }
    }
}

impl<H: HashPolicy> Runnable<H> {
    /// Get the node this runnable belongs to.
    pub fn node(&self) -> NodeId {
        match self {
            Runnable::Timer(t) => t.node,
            Runnable::Record(r) => r.node,
            Runnable::ChannelSend { target, .. } => *target,
        }
    }

    /// Get the PC (program counter vertex) for this runnable.
    pub fn pc(&self) -> Vertex {
        match self {
            Runnable::Timer(t) => t.pc,
            Runnable::Record(r) => r.pc,
            Runnable::ChannelSend { pc, .. } => *pc,
        }
    }
}

#[derive(Debug, Clone)]
pub struct State<H: HashPolicy> {
    pub nodes: Vector<Env<H>>, // Index is node_id.index
    pub runnable_tasks: Vector<Runnable<H>>,
    pub channels: ImHashMap<ChannelId, ChannelState<H>>,
    pub crash_info: CrashInfo<H>,
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
        Self {
            nodes,
            runnable_tasks: Vector::new(),
            channels: ImHashMap::new(),
            crash_info: CrashInfo {
                currently_crashed: OrdSet::new(),
                queued_messages: Vector::new(),
                current_step: 0,
            },
            next_channel_id: 0,
            next_unique_id: 0,
        }
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

    /// Compute state signature by aggregating component signatures.
    pub fn signature(&self) -> u64 {
        let mut h: u64 = 0;

        // Nodes: XOR of positioned Env signatures
        for (i, env) in self.nodes.iter().enumerate() {
            h ^= H::mix(env.sig, i as u32);
        }

        // Runnable tasks: Hash each task and mix
        for (i, task) in self.runnable_tasks.iter().enumerate() {
            h ^= H::mix(compute_hash(task), (1000 + i) as u32);
        }

        // Channels: Order-independent XOR
        for (chan_id, chan_state) in self.channels.iter() {
            let chan_hash = compute_hash(&(chan_id, chan_state));
            h ^= chan_hash;
        }

        // crash_info
        h ^= compute_hash(&self.crash_info);

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
                    // Note: store errors in continuations are logged but ignored (fire-and-forget)
                    if let Err(e) = store(&lhs, val, &mut reader.env, &mut node_env) {
                        log::warn!("Store failed in async continuation: {}", e);
                    }
                    state.nodes[reader.node.index] = node_env;
                    state.runnable_tasks.push_back(Runnable::Record(reader));
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
