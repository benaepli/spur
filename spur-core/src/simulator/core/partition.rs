use crate::compiler::cfg::Vertex;
use crate::simulator::core::state::{NodeId, Record, Runnable};
use crate::simulator::core::values::{ChannelId, Value};
use crate::simulator::hash_utils::HashPolicy;
use imbl::{OrdSet, Vector};
use log::warn;
use std::hash::{Hash, Hasher};

/// Which partition shape is active.
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionType {
    /// One node isolated from all others.
    IsolateOne(NodeId),
    /// Two explicit groups — cross-group messages blocked.
    Halves {
        side_a: OrdSet<NodeId>,
        side_b: OrdSet<NodeId>,
    },
    /// Overlapping majorities in a ring — each node can reach floor(n/2)+1
    /// nearest neighbors (including itself). No global quorum exists.
    MajoritiesRing { num_nodes: usize },
    /// Two halves connected only through one bridge node.
    Bridge {
        bridge: NodeId,
        side_a: OrdSet<NodeId>,
        side_b: OrdSet<NodeId>,
    },
}

impl Hash for PartitionType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            PartitionType::IsolateOne(node) => {
                0u8.hash(state);
                node.hash(state);
            }
            PartitionType::Halves { side_a, side_b } => {
                1u8.hash(state);
                let a: Vec<_> = side_a.iter().copied().collect();
                a.hash(state);
                let b: Vec<_> = side_b.iter().copied().collect();
                b.hash(state);
            }
            PartitionType::MajoritiesRing { num_nodes } => {
                2u8.hash(state);
                num_nodes.hash(state);
            }
            PartitionType::Bridge {
                bridge,
                side_a,
                side_b,
            } => {
                3u8.hash(state);
                bridge.hash(state);
                let a: Vec<_> = side_a.iter().copied().collect();
                a.hash(state);
                let b: Vec<_> = side_b.iter().copied().collect();
                b.hash(state);
            }
        }
    }
}

impl PartitionType {
    /// Returns true if src can send a message to dest under this partition.
    pub fn can_communicate(&self, src: NodeId, dest: NodeId) -> bool {
        if src == dest {
            return true;
        }
        match self {
            PartitionType::IsolateOne(isolated) => {
                // The isolated node cannot communicate with anyone else
                src != *isolated && dest != *isolated
            }
            PartitionType::Halves { side_a, side_b } => {
                // Cross-group blocked
                let src_in_a = side_a.contains(&src);
                let dest_in_a = side_a.contains(&dest);
                let src_in_b = side_b.contains(&src);
                let dest_in_b = side_b.contains(&dest);
                (src_in_a && dest_in_a) || (src_in_b && dest_in_b)
            }
            PartitionType::MajoritiesRing { num_nodes } => {
                let n = *num_nodes;
                if n <= 1 {
                    return true;
                }
                // Each node can reach floor(n/2)+1 nearest nodes (including itself).
                // Distance is min(|i-j|, n-|i-j|).
                let reach = n / 2; // floor(n/2) — plus self gives floor(n/2)+1 total
                let i = src.index;
                let j = dest.index;
                let dist = {
                    let d = if i > j { i - j } else { j - i };
                    d.min(n - d)
                };
                dist <= reach
            }
            PartitionType::Bridge {
                bridge,
                side_a,
                side_b,
            } => {
                // Bridge can communicate with everyone
                if src == *bridge || dest == *bridge {
                    return true;
                }
                // Non-bridge nodes can only reach nodes in their own half
                let same_side = (side_a.contains(&src) && side_a.contains(&dest))
                    || (side_b.contains(&src) && side_b.contains(&dest));
                same_side
            }
        }
    }
}

/// A network message buffered due to partition.
#[derive(Debug, Clone)]
pub enum QueuedMessage<H: HashPolicy> {
    Record {
        dest: NodeId,
        record: Record<H>,
    },
    ChannelSend {
        dest: NodeId,
        channel: ChannelId,
        message: Value<H>,
        origin_node: NodeId,
        pc: Vertex,
        priority: f64,
    },
}

impl<H: HashPolicy> Hash for QueuedMessage<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        match self {
            QueuedMessage::Record { dest, record } => {
                0u8.hash(state);
                dest.hash(state);
                record.hash(state);
            }
            QueuedMessage::ChannelSend {
                dest,
                channel,
                message,
                origin_node,
                pc,
                ..
            } => {
                1u8.hash(state);
                dest.hash(state);
                channel.hash(state);
                message.hash(state);
                origin_node.hash(state);
                pc.hash(state);
            }
        }
    }
}

/// Partition state tracked in simulator State.
#[derive(Debug, Clone)]
pub struct PartitionInfo<H: HashPolicy> {
    pub active: Option<PartitionType>,
    /// Messages blocked by the partition. Separate from crash queue.
    pub queued_messages: Vector<QueuedMessage<H>>,
}

impl<H: HashPolicy> Hash for PartitionInfo<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        self.active.is_some().hash(state);
        if let Some(ref pt) = self.active {
            pt.hash(state);
        }
        self.queued_messages.hash(state);
    }
}

impl<H: HashPolicy> PartitionInfo<H> {
    pub fn new() -> Self {
        Self {
            active: None,
            queued_messages: Vector::new(),
        }
    }

    /// Check if a message between src→dest is blocked by the active partition.
    pub fn is_blocked(&self, src: NodeId, dest: NodeId) -> bool {
        match &self.active {
            Some(pt) => !pt.can_communicate(src, dest),
            None => false,
        }
    }

    /// Buffer a Record that's blocked by the partition.
    pub fn buffer_record(&mut self, dest: NodeId, record: Record<H>) {
        self.queued_messages
            .push_back(QueuedMessage::Record { dest, record });
    }

    /// Buffer a ChannelSend that's blocked by the partition.
    pub fn buffer_channel_send(
        &mut self,
        dest: NodeId,
        channel: ChannelId,
        message: Value<H>,
        origin_node: NodeId,
        pc: Vertex,
        priority: f64,
    ) {
        self.queued_messages
            .push_back(QueuedMessage::ChannelSend {
                dest,
                channel,
                message,
                origin_node,
                pc,
                priority,
            });
    }
}

/// Activate a partition. No-op with warning if one is already active.
/// Only scans network_queue since local and timer items cannot be cross-node.
pub fn activate_partition<H: HashPolicy>(
    state: &mut crate::simulator::core::state::State<H>,
    partition: PartitionType,
) {
    if state.partition_info.active.is_some() {
        warn!("Partition already active, ignoring new partition");
        return;
    }
    state.partition_info.active = Some(partition);
    let tasks = std::mem::take(&mut state.network_queue);
    for task in tasks {
        match &task {
            Runnable::Record(r)
                if r.origin_node != r.node
                    && state.partition_info.is_blocked(r.origin_node, r.node) =>
            {
                let mut r = r.clone();
                r.reset();
                state.partition_info.buffer_record(r.node, r);
            }
            Runnable::ChannelSend {
                origin_node,
                target,
                channel,
                message,
                pc,
                priority,
            } if state.partition_info.is_blocked(*origin_node, *target) => {
                state.partition_info.buffer_channel_send(
                    *target,
                    *channel,
                    message.clone(),
                    *origin_node,
                    *pc,
                    *priority,
                );
            }
            _ => state.network_queue.push_back(task),
        }
    }
}

/// Heal the active partition. Drains the partition queue with crash-awareness:
/// - Messages to crashed nodes: Records move to crash queue, ChannelSends are dropped.
/// - Messages to alive nodes: converted back to runnables via push_runnable.
pub fn heal_partition<H: HashPolicy>(state: &mut crate::simulator::core::state::State<H>) {
    if state.partition_info.active.is_none() {
        warn!("No active partition to heal");
        return;
    }
    state.partition_info.active = None;
    let queued = std::mem::take(&mut state.partition_info.queued_messages);
    for msg in queued {
        match msg {
            QueuedMessage::Record { dest, record } => {
                if state.crash_info.currently_crashed.contains(&dest) {
                    state.crash_info.queued_messages.push_back((dest, record));
                } else {
                    state.push_runnable(Runnable::Record(record));
                }
            }
            QueuedMessage::ChannelSend {
                dest,
                channel,
                message,
                origin_node,
                pc,
                priority,
            } => {
                if !state.crash_info.currently_crashed.contains(&dest) {
                    state.push_runnable(Runnable::ChannelSend {
                        target: dest,
                        channel,
                        message,
                        origin_node,
                        pc,
                        priority,
                    });
                }
            }
        }
    }
}
