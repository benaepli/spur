use crate::compiler::cfg::{Lhs, Vertex};
use crate::simulator::core::eval::store;
use crate::simulator::core::values::{ChannelId, Env, Value};
use crate::simulator::hash_utils::{compute_hash, mix};
use imbl::{HashMap as ImHashMap, OrdSet, Vector};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct LogEntry {
    pub node: usize,
    pub content: String,
    pub step: i32,
}

/// Trait for handling Print statement output during execution.
pub trait Logger {
    fn log(&mut self, entry: LogEntry);
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub pc: Vertex,
    pub node: usize,
    pub origin_node: usize,
    pub continuation: Continuation,
    pub env: Env, // Just local env, node env is in State
    pub x: f64,
    pub policy: UpdatePolicy,
}

impl Hash for Record {
    fn hash<H: Hasher>(&self, state: &mut H) {
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

#[derive(Debug, Clone)]
pub struct CrashInfo {
    pub currently_crashed: OrdSet<usize>,
    pub queued_messages: Vector<(usize, Record)>, // (dest_node, record)
    pub current_step: i32,
}

impl Hash for CrashInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // OrdSet has deterministic ordering, collect to Vec for safety
        let crashed_vec: Vec<_> = self.currently_crashed.iter().copied().collect();
        crashed_vec.hash(state);
        self.queued_messages.hash(state);
        self.current_step.hash(state);
    }
}

#[derive(Debug, Clone, Hash)]
pub struct ChannelState {
    pub capacity: i32,
    pub buffer: Vector<Value>,
    // We move Record out of Runnable and into Waiting.
    pub waiting_readers: Vector<(Record, Lhs)>,
    pub waiting_writers: Vector<(Record, Value)>,
}

impl ChannelState {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity,
            buffer: Vector::new(),
            waiting_readers: Vector::new(),
            waiting_writers: Vector::new(),
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
pub struct Operation {
    pub client_id: i32,
    pub op_action: String,
    pub kind: OpKind,
    pub payload: Vec<Value>,
    pub unique_id: i32,
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Timer {
    pub pc: Vertex,
    pub node: usize,
    pub channel: ChannelId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Runnable {
    Timer(Timer),
    Record(Record),
}

impl Hash for Runnable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Runnable::Timer(t) => {
                0u8.hash(state);
                t.hash(state);
            }
            Runnable::Record(r) => {
                1u8.hash(state);
                r.hash(state);
            }
        }
    }
}

impl Runnable {
    /// Get the node this runnable belongs to.
    pub fn node(&self) -> usize {
        match self {
            Runnable::Timer(t) => t.node,
            Runnable::Record(r) => r.node,
        }
    }

    /// Get the PC (program counter vertex) for this runnable.
    pub fn pc(&self) -> Vertex {
        match self {
            Runnable::Timer(t) => t.pc,
            Runnable::Record(r) => r.pc,
        }
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub nodes: Vector<Env>, // Index is node_id
    pub runnable_tasks: Vector<Runnable>,
    pub channels: ImHashMap<ChannelId, ChannelState>,
    pub crash_info: CrashInfo,
    next_channel_id: usize,
    next_unique_id: usize,
}

impl State {
    pub fn new(node_count: usize, node_slot_count: usize) -> Self {
        Self {
            nodes: (0..node_count)
                .map(|i| {
                    let mut env = Env::with_slots(node_slot_count);
                    env.set(0, Value::node(i)); // Slot 0 = self
                    env
                })
                .collect(),
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
            h ^= mix(env.sig, i as u32);
        }

        // Runnable tasks: Hash each task and mix
        for (i, task) in self.runnable_tasks.iter().enumerate() {
            h ^= mix(compute_hash(task), (1000 + i) as u32);
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

impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use precomputed signature for O(1) hashing
        self.signature().hash(state);
    }
}

/// Continuation representing what to do when an execution completes.
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Continuation {
    /// Node recovery continuation
    Recover,
    /// Async message delivery continuation
    Async { chan_id: ChannelId },
    /// Client operation completion - returns data for caller to handle
    ClientOp {
        client_id: i32,
        op_name: String,
        unique_id: i32,
    },
}

/// Result returned when a ClientOp continuation completes.
#[derive(Debug, Clone)]
pub struct ClientOpResult {
    pub client_id: i32,
    pub op_name: String,
    pub unique_id: i32,
    pub value: Value,
}

impl Continuation {
    /// Execute the continuation and return any client operation result.
    pub fn call(self, state: &mut State, val: Value) -> Option<ClientOpResult> {
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
                    let mut node_env = state.nodes[reader.node].clone();
                    // Note: store errors in continuations are logged but ignored (fire-and-forget)
                    if let Err(e) = store(&lhs, val, &mut reader.env, &mut node_env) {
                        log::warn!("Store failed in async continuation: {}", e);
                    }
                    state.nodes[reader.node] = node_env;
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
        }
    }
}
