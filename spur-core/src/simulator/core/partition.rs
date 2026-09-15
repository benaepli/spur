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
    /// A group split in two: a message between members on different sides is
    /// blocked. `side_a` and `side_b` together are the group.
    Halves {
        side_a: OrdSet<NodeId>,
        side_b: OrdSet<NodeId>,
    },
    /// Overlapping majorities in a ring — each node can reach floor(n/2)+1
    /// nearest neighbors (including itself). No global quorum exists.
    MajoritiesRing { ring: Vec<NodeId> },
    /// A group split in two halves that reach each other only through the
    /// bridge. The bridge and both sides together are the group.
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
            PartitionType::MajoritiesRing { ring } => {
                2u8.hash(state);
                ring.hash(state);
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

/// 0 for a member of `side_a`, 1 for a member of `side_b`, `None` otherwise.
fn side_of(node: NodeId, side_a: &OrdSet<NodeId>, side_b: &OrdSet<NodeId>) -> Option<u8> {
    if side_a.contains(&node) {
        Some(0)
    } else if side_b.contains(&node) {
        Some(1)
    } else {
        None
    }
}

impl PartitionType {
    /// Returns true if src can send a message to dest under this partition.
    /// A group-shaped partition constrains a message only when both endpoints
    /// are members of its group; membership compares whole node ids.
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
                match (side_of(src, side_a, side_b), side_of(dest, side_a, side_b)) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                }
            }
            PartitionType::MajoritiesRing { ring } => {
                let position = |node: NodeId| ring.iter().position(|&m| m == node);
                let (Some(i), Some(j)) = (position(src), position(dest)) else {
                    return true;
                };
                let n = ring.len();
                // Distance is taken over ring positions: min(|i-j|, n-|i-j|).
                let reach = n / 2;
                let d = i.abs_diff(j);
                d.min(n - d) <= reach
            }
            PartitionType::Bridge {
                bridge,
                side_a,
                side_b,
            } => {
                if src == *bridge || dest == *bridge {
                    return true;
                }
                match (side_of(src, side_a, side_b), side_of(dest, side_a, side_b)) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                }
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
    program: &crate::compiler::cfg::Program,
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
                state.flight_leave(&task);
                state.net_leave(&task);
                let mut r = r.clone();
                r.reset(program);
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
                state.flight_leave(&task);
                state.partition_info.buffer_channel_send(
                    *target,
                    *channel,
                    message.clone(),
                    *origin_node,
                    *pc,
                    *priority,
                );
            }
            _ => state.network_queue.push(task),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::resolver::NameId;
    use crate::simulator::path::plan::PartitionAction;
    use crate::compiler::cfg::Program;
    use crate::simulator::core::state::{Continuation, State};
    use crate::simulator::core::values::Env;
    use crate::simulator::hash_utils::NoHashing;
    use crate::simulator::util_stats::DeliveryBias;

    fn node(role: usize, index: usize) -> NodeId {
        NodeId { role: NameId(role), index }
    }

    /// Shard A holds nodes 0, 2, 4 and 8 of role 0, shard B holds 1, 3 and 5 of
    /// the same role, 6 is a router of role 1 and 20 is a client.
    fn shard_a() -> Vec<NodeId> {
        vec![node(0, 0), node(0, 2), node(0, 4), node(0, 8)]
    }

    fn outsiders() -> Vec<NodeId> {
        vec![node(0, 1), node(0, 3), node(0, 5), node(1, 6), node(2, 20)]
    }

    fn assert_outsiders_unblocked(p: &PartitionType) {
        let group = shard_a();
        let outside = outsiders();
        for &o in &outside {
            for &m in &group {
                assert!(p.can_communicate(o, m), "{o:?} -> {m:?}");
                assert!(p.can_communicate(m, o), "{m:?} -> {o:?}");
            }
            for &o2 in &outside {
                assert!(p.can_communicate(o, o2), "{o:?} -> {o2:?}");
            }
        }
        for &n in group.iter().chain(&outside) {
            assert!(p.can_communicate(n, n), "self {n:?}");
        }
    }

    #[test]
    fn halves_blocks_only_member_pairs_on_different_sides() {
        let group = shard_a();
        let p = PartitionAction::Halves { group: group.clone(), side_a: vec![0, 3] }.to_partition_type();
        let (a0, a1, a2, a3) = (group[0], group[1], group[2], group[3]);
        for (s, d, delivered) in [(a0, a3, true), (a1, a2, true), (a0, a1, false), (a3, a2, false), (a2, a0, false)] {
            assert_eq!(p.can_communicate(s, d), delivered, "{s:?} -> {d:?}");
            assert_eq!(p.can_communicate(d, s), delivered, "{d:?} -> {s:?}");
        }
        assert_outsiders_unblocked(&p);
    }

    #[test]
    fn bridge_members_meet_only_through_the_bridge() {
        let group = shard_a();
        let p = PartitionAction::Bridge { group: group.clone(), bridge: 1 }.to_partition_type();
        let (a0, bridge, a2, a3) = (group[0], group[1], group[2], group[3]);
        for (s, d, delivered) in [(a0, bridge, true), (a2, bridge, true), (a3, bridge, true), (a2, a3, true), (a0, a2, false), (a0, a3, false)] {
            assert_eq!(p.can_communicate(s, d), delivered, "{s:?} -> {d:?}");
            assert_eq!(p.can_communicate(d, s), delivered, "{d:?} -> {s:?}");
        }
        assert_outsiders_unblocked(&p);
    }

    #[test]
    fn a_ring_constrains_members_by_ring_position_only() {
        let p = PartitionAction::MajoritiesRing { group: shard_a() }.to_partition_type();
        assert_outsiders_unblocked(&p);
        for high in [node(2, 9), node(2, 100), node(0, usize::MAX)] {
            for &m in &shard_a() {
                assert!(p.can_communicate(high, m) && p.can_communicate(m, high));
            }
        }
    }

    fn record(origin: NodeId, dest: NodeId) -> Record<NoHashing> {
        Record {
            pc: 0,
            node: dest,
            origin_node: origin,
            continuation: Continuation::Recover,
            entry_pc: 0,
            initial_args: ecow::EcoVec::new(),
            entry_func: NameId(0),
            env: Env::<NoHashing>::with_slots(2),
            priority: 0.5,
            causal_operation_id: None,
            trace_id: None,
            trace_payload: None,
            link_seq: None,
            origin_incarnation: 0,
            send_ordinal: 0,
            receiver_token_at_send: 0,
            bias: DeliveryBias::NONE,
            timer_entry: None,
        }
    }

    fn send(origin: NodeId, dest: NodeId) -> Runnable<NoHashing> {
        Runnable::ChannelSend {
            target: dest,
            channel: ChannelId { node: dest, id: 0 },
            message: Value::<NoHashing>::unit(),
            origin_node: origin,
            pc: 0,
            priority: 0.5,
        }
    }

    fn held_pairs(state: &State<NoHashing>) -> Vec<(NodeId, NodeId, &'static str)> {
        let mut pairs: Vec<_> = state
            .partition_info
            .queued_messages
            .iter()
            .map(|m| match m {
                QueuedMessage::Record { dest, record } => (record.origin_node, *dest, "rpc"),
                QueuedMessage::ChannelSend { dest, origin_node, .. } => (*origin_node, *dest, "send"),
            })
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn activation_holds_only_blocked_member_traffic_until_the_heal() {
        let mut state = State::<NoHashing>::new(&[(NameId(0), 6)], 2);
        let client = state.add_node(NameId(2), 2);
        let program = Program::default();
        let group = [node(0, 0), node(0, 2), node(0, 4)];
        state.push_runnable(Runnable::Record(record(group[0], group[1])));
        state.push_runnable(Runnable::Record(record(client, group[0])));
        state.push_runnable(Runnable::Record(record(node(0, 1), node(0, 3))));
        state.push_runnable(send(group[2], group[0]));
        state.push_runnable(send(group[1], group[2]));
        state.push_runnable(send(node(0, 5), group[1]));

        let halves = PartitionAction::Halves { group: group.to_vec(), side_a: vec![0] };
        activate_partition(&mut state, &program, halves.to_partition_type());
        assert_eq!(
            held_pairs(&state),
            vec![(group[0], group[1], "rpc"), (group[2], group[0], "send")],
            "only cross-side member traffic waits"
        );
        assert_eq!(state.network_queue.len(), 4, "client, other-shard and same-side traffic stays schedulable");
        assert!(state.partition_info.is_blocked(group[1], group[0]));
        assert!(!state.partition_info.is_blocked(client, group[1]));
        assert!(!state.partition_info.is_blocked(node(0, 1), node(0, 5)));

        heal_partition(&mut state);
        assert!(state.partition_info.queued_messages.is_empty());
        assert!(state.partition_info.active.is_none());
        assert_eq!(state.network_queue.len(), 6, "the heal releases every held message");
    }

    #[test]
    fn isolate_one_blocks_every_pair_with_one_isolated_endpoint() {
        let isolated = node(0, 2);
        let p = PartitionType::IsolateOne(isolated);
        for other in shard_a().into_iter().chain(outsiders()) {
            let delivered = other == isolated;
            assert_eq!(p.can_communicate(isolated, other), delivered);
            assert_eq!(p.can_communicate(other, isolated), delivered);
        }
        assert!(p.can_communicate(node(0, 1), node(2, 20)));
    }
}
