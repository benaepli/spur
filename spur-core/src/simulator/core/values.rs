use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::state::NodeId;
use crate::simulator::hash_utils::{HashPolicy, mix};
use crate::simulator::util_stats;
use ecow::{EcoString, EcoVec};
use std::cmp::Ordering;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId {
    pub node: NodeId,
    pub id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LinkId(pub usize);

/// Map storage of a spec value. The hasher carries no per-process seed, so
/// the order a spec sees when it iterates or takes the head of a map is a
/// function of the entries alone, and two sessions at one seed replay the
/// same schedule.
pub type ValueMap<H> =
    imbl::GenericHashMap<Value<H>, Value<H>, BuildHasherDefault<FxHasher>, imbl::shared_ptr::DefaultSharedPtr>;

/// Sequence storage of a spec value.
///
/// `EcoVec` is a copy-on-write vector behind one allocation sized to the
/// element count, so a short list costs bytes proportional to its length
/// rather than a fixed branching-factor block, and it is two words wide so it
/// does not set the size of `Value`. Cloning is a refcount bump; the first
/// write after a clone copies `len` elements. Element order is the sequence
/// order the spec wrote, so nothing a spec observes depends on the layout.
pub type ValueSeq<H> = EcoVec<Value<H>>;

/// The inner representation of a value, without cached signature.
#[derive(Clone, Debug)]
pub enum ValueKind<H: HashPolicy> {
    Int(i64),
    Bool(bool),
    Map(ValueMap<H>),
    List(ValueSeq<H>),
    Option(Option<Arc<Value<H>>>),
    Channel(ChannelId),
    Node(NodeId),
    /// FIFO RPC link: `(link_id, peer)`. Pair the per-link sequence identity
    /// with the peer node so RPCs through the link route deterministically
    /// even after sender crash + recovery.
    FifoLink(LinkId, NodeId),
    String(EcoString),
    Unit,
    Tuple(ValueSeq<H>),
    Variant(u32, EcoString, Option<Arc<Value<H>>>), // (enum_id, variant_name, payload)
}

#[derive(Clone, Debug)]
pub struct Value<H: HashPolicy> {
    pub kind: ValueKind<H>,
    pub sig: u64,
    _marker: PhantomData<H>,
}

/// Helper to securely combine K and V without needing map position.
/// Uses mix with different tags (0 for key, 1 for value) to bind them tightly
/// and avoid collisions from key/value swaps.
#[inline]
pub fn hash_map_entry(k_sig: u64, v_sig: u64) -> u64 {
    mix(k_sig, 0) ^ mix(v_sig, 1)
}

impl<H: HashPolicy> Value<H> {
    /// Create a new Value. Under an eager policy the signature is computed
    /// here; otherwise it is stored as 0 and `Hash` derives the leaf
    /// signature from the kind when the value is hashed.
    #[inline]
    pub fn new(kind: ValueKind<H>) -> Self {
        let sig = if H::EAGER { Self::compute_sig(&kind) } else { 0 };
        Self {
            kind,
            sig,
            _marker: PhantomData,
        }
    }

    /// Signature for leaf variants only; composites get sig = 0.
    /// Used under `NoHashing` so map-keyable leaves (Int/Bool/Node/String/Channel)
    /// still discriminate for `imbl::HashMap` bucketing, while composites skip work.
    #[inline]
    fn compute_sig_leaf_only(kind: &ValueKind<H>) -> u64 {
        match kind {
            ValueKind::Int(_)
            | ValueKind::Bool(_)
            | ValueKind::String(_)
            | ValueKind::Node(_)
            | ValueKind::Channel(_)
            | ValueKind::FifoLink(_, _)
            | ValueKind::Unit => Self::compute_sig(kind),
            ValueKind::Option(_)
            | ValueKind::Tuple(_)
            | ValueKind::List(_)
            | ValueKind::Map(_)
            | ValueKind::Variant(_, _, _) => 0,
        }
    }

    /// Compute signature for a ValueKind.
    fn compute_sig(kind: &ValueKind<H>) -> u64 {
        use crate::simulator::hash_utils::HASH_PRIME;
        match kind {
            // Leaves: direct mix, no hasher allocation. The wrapping_mul by
            // HASH_PRIME gives avalanche so small bit-level differences don't
            // cancel under XOR in parent composites.
            ValueKind::Int(i) => mix((*i as u64).wrapping_mul(HASH_PRIME), 0),
            ValueKind::Bool(b) => mix((*b as u64).wrapping_mul(HASH_PRIME), 1),
            ValueKind::Unit => mix(0, 4),
            ValueKind::Node(n) => {
                mix((n.role.0 as u64).wrapping_mul(HASH_PRIME), 3)
                    ^ mix((n.index as u64).wrapping_mul(HASH_PRIME), 103)
            }
            ValueKind::Channel(c) => {
                mix((c.node.role.0 as u64).wrapping_mul(HASH_PRIME), 5)
                    ^ mix((c.node.index as u64).wrapping_mul(HASH_PRIME), 105)
                    ^ mix((c.id as u64).wrapping_mul(HASH_PRIME), 205)
            }
            ValueKind::FifoLink(link_id, peer) => {
                mix((link_id.0 as u64).wrapping_mul(HASH_PRIME), 11)
                    ^ mix((peer.role.0 as u64).wrapping_mul(HASH_PRIME), 111)
                    ^ mix((peer.index as u64).wrapping_mul(HASH_PRIME), 211)
            }
            ValueKind::String(s) => {
                let mut h = FxHasher::default();
                s.hash(&mut h);
                mix(h.finish(), 2)
            }
            ValueKind::Option(o) => {
                let mut h = FxHasher::default();
                6u8.hash(&mut h);
                match o {
                    None => 0u8.hash(&mut h),
                    Some(v) => {
                        1u8.hash(&mut h);
                        v.sig.hash(&mut h);
                    }
                }
                h.finish()
            }
            ValueKind::Tuple(v) => {
                // Position-dependent XOR for order sensitivity
                let mut sig = 0u64;
                let mut h = FxHasher::default();
                7u8.hash(&mut h);
                v.len().hash(&mut h);
                sig ^= h.finish();
                for (i, val) in v.iter().enumerate() {
                    sig ^= mix(val.sig, i as u32);
                }
                sig
            }
            ValueKind::List(v) => {
                // Position-dependent XOR for order sensitivity
                let mut sig = 0u64;
                let mut h = FxHasher::default();
                8u8.hash(&mut h);
                v.len().hash(&mut h);
                sig ^= h.finish();
                for (i, val) in v.iter().enumerate() {
                    sig ^= mix(val.sig, i as u32);
                }
                sig
            }
            ValueKind::Map(m) => {
                // Order-independent XOR for maps
                let mut sig = 0u64;
                let mut h = FxHasher::default();
                9u8.hash(&mut h);
                m.len().hash(&mut h);
                sig ^= h.finish();
                for (k, v) in m.iter() {
                    // Use hash_map_entry for consistent entry hashing
                    sig ^= hash_map_entry(k.sig, v.sig);
                }
                sig
            }
            ValueKind::Variant(enum_id, name, payload) => {
                let mut h = FxHasher::default();
                10u8.hash(&mut h); // discriminant
                enum_id.hash(&mut h);
                name.hash(&mut h);
                match payload {
                    None => 0u8.hash(&mut h),
                    Some(v) => {
                        1u8.hash(&mut h);
                        v.sig.hash(&mut h);
                    }
                }
                h.finish()
            }
        }
    }

    // Convenience constructors
    #[inline]
    pub fn int(i: i64) -> Self {
        Self::new(ValueKind::Int(i))
    }

    #[inline]
    pub fn bool(b: bool) -> Self {
        Self::new(ValueKind::Bool(b))
    }

    #[inline]
    pub fn string(s: EcoString) -> Self {
        Self::new(ValueKind::String(s))
    }

    #[inline]
    pub fn node(n: NodeId) -> Self {
        Self::new(ValueKind::Node(n))
    }

    #[inline]
    pub fn unit() -> Self {
        Self::new(ValueKind::Unit)
    }

    #[inline]
    pub fn channel(c: ChannelId) -> Self {
        Self::new(ValueKind::Channel(c))
    }

    #[inline]
    pub fn fifo_link(link_id: LinkId, peer: NodeId) -> Self {
        Self::new(ValueKind::FifoLink(link_id, peer))
    }

    #[inline]
    pub fn option(o: Option<Arc<Value<H>>>) -> Self {
        Self::new(ValueKind::Option(o))
    }

    #[inline]
    pub fn option_some(v: Value<H>) -> Self {
        Self::new(ValueKind::Option(Some(Arc::new(v))))
    }

    #[inline]
    pub fn option_none() -> Self {
        Self::new(ValueKind::Option(None))
    }

    #[inline]
    pub fn tuple(v: ValueSeq<H>) -> Self {
        Self::new(ValueKind::Tuple(v))
    }

    #[inline]
    pub fn list(v: ValueSeq<H>) -> Self {
        Self::new(ValueKind::List(v))
    }

    #[inline]
    pub fn map(m: ValueMap<H>) -> Self {
        Self::new(ValueKind::Map(m))
    }

    #[inline]
    pub fn variant(enum_id: u32, name: EcoString, payload: Option<Arc<Value<H>>>) -> Self {
        Self::new(ValueKind::Variant(enum_id, name, payload))
    }

    /// Create a Value with a pre-computed signature (for incremental updates)
    #[inline]
    pub fn with_sig(kind: ValueKind<H>, sig: u64) -> Self {
        Self {
            kind,
            sig,
            _marker: PhantomData,
        }
    }
}

impl<H: HashPolicy> PartialEq for Value<H> {
    fn eq(&self, other: &Self) -> bool {
        use ValueKind::*;
        match (&self.kind, &other.kind) {
            (Int(a), Int(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (String(a), String(b)) => a == b,
            (Node(a), Node(b)) => a == b,
            (Unit, Unit) => true,
            (Option(a), Option(b)) => a == b,
            (Tuple(a), Tuple(b)) => a == b,
            (List(a), List(b)) => a == b,
            (Map(a), Map(b)) => a == b,
            (Channel(a), Channel(b)) => a == b,
            (FifoLink(la, pa), FifoLink(lb, pb)) => la == lb && pa == pb,
            (Variant(id_a, name_a, p_a), Variant(id_b, name_b, p_b)) => {
                id_a == id_b && name_a == name_b && p_a == p_b
            }
            _ => false,
        }
    }
}
impl<H: HashPolicy> Eq for Value<H> {}
impl<H: HashPolicy> PartialOrd for Value<H> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<H: HashPolicy> Ord for Value<H> {
    fn cmp(&self, other: &Self) -> Ordering {
        use ValueKind::*;
        match (&self.kind, &other.kind) {
            (Int(a), Int(b)) => a.cmp(b),
            (Bool(a), Bool(b)) => a.cmp(b),
            (String(a), String(b)) => a.cmp(b),
            (Node(a), Node(b)) => a.cmp(b),
            (Unit, Unit) => Ordering::Equal,
            (Option(a), Option(b)) => a.cmp(b),
            (Tuple(a), Tuple(b)) => a.cmp(b),
            (List(a), List(b)) => a.cmp(b),
            (Map(a), Map(b)) => {
                // Compare maps by converting to sorted vectors
                let mut a_vec: Vec<_> = a.iter().collect();
                let mut b_vec: Vec<_> = b.iter().collect();
                a_vec.sort_by(|x, y| x.0.cmp(y.0));
                b_vec.sort_by(|x, y| x.0.cmp(y.0));
                a_vec.cmp(&b_vec)
            }
            (Channel(a), Channel(b)) => a.cmp(b),
            (FifoLink(la, pa), FifoLink(lb, pb)) => (la, pa).cmp(&(lb, pb)),
            (Variant(id_a, name_a, p_a), Variant(id_b, name_b, p_b)) => {
                (id_a, name_a, p_a).cmp(&(id_b, name_b, p_b))
            }
            // Cross-type comparisons (simple deterministic ordering)
            (Int(_), _) => Ordering::Less,
            (_, Int(_)) => Ordering::Greater,
            (Bool(_), _) => Ordering::Less,
            (_, Bool(_)) => Ordering::Greater,
            (String(_), _) => Ordering::Less,
            (_, String(_)) => Ordering::Greater,
            (Node(_), _) => Ordering::Less,
            (_, Node(_)) => Ordering::Greater,
            (Unit, _) => Ordering::Less,
            (_, Unit) => Ordering::Greater,
            (Option(_), _) => Ordering::Less,
            (_, Option(_)) => Ordering::Greater,
            (Tuple(_), _) => Ordering::Less,
            (_, Tuple(_)) => Ordering::Greater,
            (List(_), _) => Ordering::Less,
            (_, List(_)) => Ordering::Greater,
            (Map(_), _) => Ordering::Less,
            (_, Map(_)) => Ordering::Greater,
            (Channel(_), _) => Ordering::Less,
            (_, Channel(_)) => Ordering::Greater,
            (FifoLink(_, _), _) => Ordering::Less,
            (_, FifoLink(_, _)) => Ordering::Greater,
        }
    }
}

impl<H: HashPolicy> Hash for Value<H> {
    /// Under a lazy policy the hashed signature must equal the one an eager
    /// leaf-only construction would have stored, because map layout and
    /// iteration order follow these bits.
    #[inline]
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        if H::EAGER {
            self.sig.hash(state);
        } else {
            let sig = Self::compute_sig_leaf_only(&self.kind);
            if !self.kind.is_composite() {
                util_stats::record_leaf_hash_deferred();
            }
            sig.hash(state);
        }
    }
}

impl<H: HashPolicy> ValueKind<H> {
    /// Whether the kind holds other values, so a leaf-only signature is 0.
    #[inline]
    pub fn is_composite(&self) -> bool {
        matches!(
            self,
            ValueKind::Option(_)
                | ValueKind::Tuple(_)
                | ValueKind::List(_)
                | ValueKind::Map(_)
                | ValueKind::Variant(_, _, _)
        )
    }
}

/// Decimal text of an integer held in a stack buffer, so it can be appended
/// to a string without formatting machinery or a temporary allocation.
pub struct Decimal {
    buf: [u8; 20],
    start: usize,
}

impl Decimal {
    pub fn of_i64(n: i64) -> Self {
        let mut d = Self::of_u64(n.unsigned_abs());
        if n < 0 {
            d.start -= 1;
            d.buf[d.start] = b'-';
        }
        d
    }

    pub fn of_u64(mut n: u64) -> Self {
        let mut buf = [0u8; 20];
        let mut start = buf.len();
        loop {
            start -= 1;
            buf[start] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        Self { buf, start }
    }

    pub fn as_str(&self) -> &str {
        // The buffer holds only ASCII digits and an optional leading sign.
        std::str::from_utf8(&self.buf[self.start..]).unwrap_or("")
    }
}

fn write_node_id<W: std::fmt::Write + ?Sized>(out: &mut W, n: &NodeId) -> std::fmt::Result {
    out.write_str("NameId(")?;
    out.write_str(Decimal::of_u64(n.role.0 as u64).as_str())?;
    out.write_str(")#")?;
    out.write_str(Decimal::of_u64(n.index as u64).as_str())
}

fn write_seq<H: HashPolicy, W: std::fmt::Write + ?Sized>(
    out: &mut W,
    open: &str,
    items: &ValueSeq<H>,
    close: &str,
) -> std::fmt::Result {
    out.write_str(open)?;
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.write_str(", ")?;
        }
        item.write_to(out)?;
    }
    out.write_str(close)
}

impl<H: HashPolicy> Value<H> {
    /// Appends the textual form of the value to `out`. This is the single
    /// definition of that text; `Display` forwards here, so appending to a
    /// `String` and formatting produce the same bytes.
    pub fn write_to<W: std::fmt::Write + ?Sized>(&self, out: &mut W) -> std::fmt::Result {
        use ValueKind::*;
        match &self.kind {
            Int(n) => out.write_str(Decimal::of_i64(*n).as_str()),
            Bool(true) => out.write_str("true"),
            Bool(false) => out.write_str("false"),
            String(s) => {
                out.write_char('"')?;
                out.write_str(s)?;
                out.write_char('"')
            }
            Node(n) => {
                out.write_str("node(")?;
                write_node_id(out, n)?;
                out.write_char(')')
            }
            Unit => out.write_str("()"),
            Option(None) => out.write_str("None"),
            Option(Some(v)) => {
                out.write_str("Some(")?;
                v.write_to(out)?;
                out.write_char(')')
            }
            Channel(ch) => {
                out.write_str("channel(")?;
                write_node_id(out, &ch.node)?;
                out.write_str(", ")?;
                out.write_str(Decimal::of_u64(ch.id as u64).as_str())?;
                out.write_char(')')
            }
            FifoLink(link_id, peer) => {
                out.write_str("fifo_link(")?;
                out.write_str(Decimal::of_u64(link_id.0 as u64).as_str())?;
                out.write_str(", ")?;
                write_node_id(out, peer)?;
                out.write_char(')')
            }
            Tuple(items) => write_seq(out, "(", items, ")"),
            List(items) => write_seq(out, "[", items, "]"),
            Map(map) => {
                out.write_str("{ ")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.write_str(", ")?;
                    }
                    k.write_to(out)?;
                    out.write_str(": ")?;
                    v.write_to(out)?;
                }
                out.write_str(" }")
            }
            Variant(_, name, None) => out.write_str(name),
            Variant(_, name, Some(payload)) => {
                out.write_str(name)?;
                out.write_char('(')?;
                payload.write_to(out)?;
                out.write_char(')')
            }
        }
    }
}

impl<H: HashPolicy> std::fmt::Display for Value<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_to(f)
    }
}

impl<H: HashPolicy> Value<H> {
    pub fn type_name(&self) -> &'static str {
        use ValueKind::*;
        match &self.kind {
            Int(_) => "int",
            Bool(_) => "bool",
            Map(_) => "map",
            List(_) => "list",
            Option(_) => "option",
            Channel(_) => "channel",
            FifoLink(_, _) => "fifo_link",
            Node(_) => "node",
            String(_) => "string",
            Unit => "unit",
            Tuple(_) => "tuple",
            Variant(_, _, _) => "variant",
        }
    }

    pub fn as_int(&self) -> Result<i64, RuntimeError> {
        if let ValueKind::Int(i) = &self.kind {
            Ok(*i)
        } else {
            Err(RuntimeError::TypeError {
                expected: "int",
                got: self.type_name(),
            })
        }
    }
    pub fn as_bool(&self) -> Result<bool, RuntimeError> {
        if let ValueKind::Bool(b) = &self.kind {
            Ok(*b)
        } else {
            Err(RuntimeError::TypeError {
                expected: "bool",
                got: self.type_name(),
            })
        }
    }
    pub fn as_node(&self) -> Result<NodeId, RuntimeError> {
        if let ValueKind::Node(n) = &self.kind {
            Ok(*n)
        } else {
            Err(RuntimeError::TypeError {
                expected: "node",
                got: self.type_name(),
            })
        }
    }
    /// Resolve an RPC target. Either a bare `Node` (no FIFO discipline) or a
    /// `FifoLink(link_id, peer)` (FIFO ordering on that link). Returns the
    /// destination node and optional link identity.
    pub fn as_rpc_target(&self) -> Result<(NodeId, Option<LinkId>), RuntimeError> {
        match &self.kind {
            ValueKind::Node(n) => Ok((*n, None)),
            ValueKind::FifoLink(link_id, peer) => Ok((*peer, Some(*link_id))),
            _ => Err(RuntimeError::TypeError {
                expected: "node or fifo_link",
                got: self.type_name(),
            }),
        }
    }
    pub fn as_map(&self) -> Result<&ValueMap<H>, RuntimeError> {
        if let ValueKind::Map(m) = &self.kind {
            Ok(m)
        } else {
            Err(RuntimeError::TypeError {
                expected: "map",
                got: self.type_name(),
            })
        }
    }
    pub fn as_list(&self) -> Result<&ValueSeq<H>, RuntimeError> {
        if let ValueKind::List(l) = &self.kind {
            Ok(l)
        } else {
            Err(RuntimeError::TypeError {
                expected: "list",
                got: self.type_name(),
            })
        }
    }
    pub fn as_channel(&self) -> Result<ChannelId, RuntimeError> {
        if let ValueKind::Channel(c) = &self.kind {
            Ok(*c)
        } else {
            Err(RuntimeError::TypeError {
                expected: "channel",
                got: self.type_name(),
            })
        }
    }

    pub fn as_variant(&self) -> Result<(u32, &EcoString, Option<&Arc<Value<H>>>), RuntimeError> {
        if let ValueKind::Variant(id, name, payload) = &self.kind {
            Ok((*id, name, payload.as_ref()))
        } else {
            Err(RuntimeError::TypeError {
                expected: "variant",
                got: self.type_name(),
            })
        }
    }
}

impl<H: HashPolicy> PartialEq for Env<H> {
    fn eq(&self, other: &Self) -> bool {
        self.slots == other.slots
    }
}

impl<H: HashPolicy> Eq for Env<H> {}

/// Slot storage for an environment.
///
/// `EcoVec` is a copy-on-write vector behind a single allocation whose length is
/// exactly the slot count. Cloning an `Env` (which happens on every record
/// enqueue, node-env read-modify-write and crash re-delivery) is a refcount
/// bump, and the first write after a clone copies only `len` slots.
type Slots<H> = EcoVec<Value<H>>;

#[derive(Clone, Debug)]
pub struct Env<H: HashPolicy> {
    pub slots: Slots<H>,
    pub sig: u64,
    /// Counts calls to `set`, so a caller can tell whether this env was
    /// written between two observations. Monotone and never reset, so wrap
    /// is the only way two observations can collide. Observation only:
    /// excluded from `Hash` and from `sig`, so deduplication and scheduling
    /// cannot see it.
    pub writes: u64,
    _marker: PhantomData<H>,
}

impl<H: HashPolicy> Hash for Env<H> {
    fn hash<Ha: Hasher>(&self, state: &mut Ha) {
        self.sig.hash(state);
    }
}

impl<H: HashPolicy> Default for Env<H> {
    fn default() -> Self {
        Self {
            slots: Slots::new(),
            sig: 0,
            writes: 0,
            _marker: PhantomData,
        }
    }
}

impl<H: HashPolicy> Env<H> {
    /// Wrap slots that are already laid out, with the signature they carry.
    /// The signature must be the one `set` would have produced for them.
    pub fn from_slots(slots: Slots<H>, sig: u64) -> Self {
        Self {
            slots,
            sig,
            writes: 0,
            _marker: PhantomData,
        }
    }

    /// Create an environment with `n` slots, all initialized to Unit
    pub fn with_slots(n: usize) -> Self {
        // Single exact-size allocation; no per-element push/grow.
        let slots: Slots<H> = EcoVec::from_elem(Value::<H>::unit(), n);

        let mut sig = 0u64;
        if H::EAGER {
            let unit_sig = Value::<H>::unit().sig;
            for i in 0..n {
                sig ^= H::mix(unit_sig, i as u32);
            }
        }

        Self {
            slots,
            sig,
            writes: 0,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn get(&self, slot: u32) -> &Value<H> {
        &self.slots[slot as usize]
    }

    #[inline(always)]
    pub fn set(&mut self, slot: u32, value: Value<H>) {
        self.writes = self.writes.wrapping_add(1);
        let idx = slot as usize;
        if idx >= self.slots.len() {
            // Extending requires recomputing signature
            let old_len = self.slots.len();
            self.slots.reserve(idx + 1 - old_len);
            while self.slots.len() < idx {
                self.slots.push(Value::<H>::unit());
            }
            if H::EAGER {
                let unit_sig = Value::<H>::unit().sig;
                for i in old_len..idx {
                    self.sig ^= H::mix(unit_sig, i as u32);
                }
                self.sig ^= H::mix(value.sig, slot);
            }
            self.slots.push(value);
        } else {
            if H::EAGER {
                let old_sig = self.slots[idx].sig;
                self.sig = H::update_env_sig(self.sig, old_sig, value.sig, slot);
            }
            // Copy-on-write: clones exactly `len` slots iff this env is shared.
            self.slots.make_mut()[idx] = value;
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Ensure we have at least `n` slots
    #[allow(dead_code)]
    pub fn ensure_slots(&mut self, n: usize) {
        if self.slots.len() < n {
            let old_len = self.slots.len();
            self.slots.reserve(n - old_len);
            while self.slots.len() < n {
                self.slots.push(Value::<H>::unit());
            }
            if H::EAGER {
                let unit_sig = Value::<H>::unit().sig;
                for i in old_len..n {
                    self.sig ^= H::mix(unit_sig, i as u32);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Independent, formatter-based rendering of a value, kept as the
    /// reference the appending writer must match byte for byte.
    struct Reference<'a, H: HashPolicy>(&'a Value<H>);

    impl<H: HashPolicy> std::fmt::Display for Reference<'_, H> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use ValueKind::*;
            match &self.0.kind {
                Int(n) => write!(f, "{}", n),
                Bool(b) => write!(f, "{}", b),
                String(s) => write!(f, "\"{}\"", s),
                Node(n) => write!(f, "node({})", n),
                Unit => write!(f, "()"),
                Option(None) => write!(f, "None"),
                Option(Some(v)) => write!(f, "Some({})", Reference(v)),
                Channel(ch) => write!(f, "channel({}, {})", ch.node, ch.id),
                FifoLink(link_id, peer) => write!(f, "fifo_link({}, {})", link_id.0, peer),
                Tuple(items) => {
                    write!(f, "(")?;
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", Reference(item))?;
                    }
                    write!(f, ")")
                }
                List(items) => {
                    write!(f, "[")?;
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", Reference(item))?;
                    }
                    write!(f, "]")
                }
                Map(map) => {
                    write!(f, "{{ ")?;
                    for (i, (k, v)) in map.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", Reference(k), Reference(v))?;
                    }
                    write!(f, " }}")
                }
                Variant(_, name, None) => write!(f, "{}", name),
                Variant(_, name, Some(payload)) => write!(f, "{}({})", name, Reference(payload)),
            }
        }
    }

    fn assert_text_matches_reference(v: &Value<WithHashing>) {
        let expected = Reference(v).to_string();
        let mut appended = String::from("prefix|");
        v.write_to(&mut appended).unwrap();
        assert_eq!(appended, format!("prefix|{expected}"), "{v:?}");
        assert_eq!(v.to_string(), expected, "{v:?}");
        assert_eq!(format!("<{v}>"), format!("<{expected}>"), "{v:?}");
    }

    #[test]
    fn decimal_matches_std_formatting() {
        for n in [
            i64::MIN,
            i64::MIN + 1,
            -1_000_000_000_000,
            -10,
            -9,
            -1,
            0,
            1,
            9,
            10,
            99,
            100,
            i64::MAX - 1,
            i64::MAX,
        ] {
            assert_eq!(Decimal::of_i64(n).as_str(), n.to_string());
        }
        for n in [0u64, 1, 9, 10, 12345, u64::MAX, usize::MAX as u64] {
            assert_eq!(Decimal::of_u64(n).as_str(), n.to_string());
        }
    }

    #[test]
    fn write_to_matches_reference_on_enumerated_values() {
        type V = Value<WithHashing>;
        let node = |role: usize, index: usize| NodeId {
            role: NameId(role),
            index,
        };
        let s = |t: &str| V::string(EcoString::from(t));
        let seq = |items: Vec<V>| ValueSeq::from(items);
        let map = |entries: Vec<(V, V)>| {
            let mut m = ValueMap::<WithHashing>::default();
            for (k, v) in entries {
                m.insert(k, v);
            }
            V::map(m)
        };
        let leaves: Vec<V> = vec![
            V::int(i64::MIN),
            V::int(i64::MAX),
            V::int(0),
            V::int(-1),
            V::int(-42),
            V::int(7),
            V::bool(true),
            V::bool(false),
            V::unit(),
            s(""),
            s("plain"),
            s("with \"quotes\""),
            s("back\\slash"),
            s("new\nline\ttab\r"),
            s("control \u{0}\u{1}\u{1f}\u{7f}"),
            s("unicode \u{e9}\u{4e2d}\u{1f600}"),
            V::node(node(0, 0)),
            V::node(node(usize::MAX, 17)),
            V::channel(ChannelId {
                node: node(3, 1),
                id: 9,
            }),
            V::channel(ChannelId {
                node: node(0, 0),
                id: usize::MAX,
            }),
            V::fifo_link(LinkId(0), node(2, 5)),
            V::fifo_link(LinkId(usize::MAX), node(1, 0)),
            V::option_none(),
            V::variant(0, EcoString::from("Empty"), None),
            V::variant(1, EcoString::from("Prepare"), Some(Arc::new(V::int(3)))),
        ];
        let mut cases: Vec<V> = leaves.clone();
        cases.push(V::option_some(V::option_some(s("x"))));
        cases.push(V::option_some(V::option_none()));
        cases.push(V::tuple(seq(vec![])));
        cases.push(V::tuple(seq(vec![V::int(1)])));
        cases.push(V::tuple(seq(vec![
            V::int(1),
            V::tuple(seq(vec![s("a"), V::bool(false)])),
            V::unit(),
        ])));
        cases.push(V::list(seq(vec![])));
        cases.push(V::list(seq(vec![V::list(seq(vec![]))])));
        cases.push(V::list(seq(leaves.clone())));
        cases.push(map(vec![]));
        cases.push(map(vec![(s("k"), V::int(1))]));
        cases.push(map(vec![
            (s("a"), map(vec![])),
            (V::int(2), V::list(seq(vec![V::int(3), V::int(4)]))),
            (V::tuple(seq(vec![s("t"), V::int(0)])), V::option_some(s("v"))),
        ]));
        cases.push(V::variant(
            2,
            EcoString::from("Commit"),
            Some(Arc::new(V::tuple(seq(vec![
                V::int(1),
                map(vec![(s("k"), V::list(seq(leaves.clone())))]),
            ])))),
        ));
        cases.push(V::variant(
            3,
            EcoString::from("Wrap"),
            Some(Arc::new(V::variant(0, EcoString::from("Empty"), None))),
        ));
        for v in &cases {
            assert_text_matches_reference(v);
        }
    }

    proptest! {
        #[test]
        fn write_to_matches_reference_on_arbitrary_values(v in arb_value()) {
            assert_text_matches_reference(&v);
        }
    }
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use crate::simulator::hash_utils::WithHashing;

    use crate::analysis::resolver::NameId;

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    // Strategy for NodeId
    prop_compose! {
        fn arb_node_id()(role in any::<usize>(), index in any::<usize>()) -> NodeId {
            NodeId { role: NameId(role), index }
        }
    }

    // Strategy for ChannelId
    prop_compose! {
        fn arb_channel_id()(node in arb_node_id(), id in any::<usize>()) -> ChannelId {
            ChannelId { node, id }
        }
    }

    // Strategy for Value
    fn arb_value() -> impl Strategy<Value = Value<WithHashing>> {
        let leaf = prop_oneof![
            any::<i64>().prop_map(Value::<WithHashing>::int),
            any::<bool>().prop_map(Value::<WithHashing>::bool),
            any::<String>().prop_map(|s| Value::<WithHashing>::string(EcoString::from(s))),
            arb_node_id().prop_map(Value::<WithHashing>::node),
            Just(Value::<WithHashing>::unit()),
            arb_channel_id().prop_map(Value::<WithHashing>::channel),
        ];

        leaf.prop_recursive(
            4,  // 4 levels deep
            64, // max size 64 nodes
            10, // 10 items per collection
            |inner| {
                prop_oneof![
                    // Option
                    prop::option::of(inner.clone()).prop_map(|opt| {
                        match opt {
                            Some(v) => Value::<WithHashing>::option_some(v),
                            None => Value::<WithHashing>::option_none(),
                        }
                    }),
                    // List
                    prop::collection::vec(inner.clone(), 0..5)
                        .prop_map(|v| Value::<WithHashing>::list(v.into())),
                    // Tuple
                    prop::collection::vec(inner.clone(), 0..5)
                        .prop_map(|v| Value::<WithHashing>::tuple(v.into())),
                    // Map
                    prop::collection::hash_map(inner.clone(), inner.clone(), 0..5)
                        .prop_map(|m| Value::<WithHashing>::map(m.into())),
                ]
            },
        )
    }

    proptest! {
        #[test]
        fn test_value_hashing_consistency(v in arb_value()) {
            let v_clone = v.clone();
            prop_assert_eq!(&v, &v_clone);
            prop_assert_eq!(calculate_hash(&v), calculate_hash(&v_clone));
            prop_assert_eq!(v.sig, v_clone.sig);
        }

        #[test]
        fn test_value_structural_equality(v1 in arb_value()) {
            let v_clone = v1.clone();
            prop_assert_eq!(v1.sig, v_clone.sig);
        }

        #[test]
        fn test_env_hashing_consistency(v in prop::collection::vec(arb_value(), 0..10)) {
            let mut env1 = Env::<WithHashing>::with_slots(v.len());
            let mut env2 = Env::<WithHashing>::with_slots(v.len());

            for (i, val) in v.iter().enumerate() {
                env1.set(i as u32, val.clone());
                env2.set(i as u32, val.clone());
            }

            prop_assert_eq!(&env1, &env2);
            prop_assert_eq!(calculate_hash(&env1), calculate_hash(&env2));
            prop_assert_eq!(env1.sig, env2.sig);
        }

        #[test]
        fn test_env_hashing_different(v in prop::collection::vec(arb_value(), 1..10), extra in arb_value()) {
             let mut env1 = Env::<WithHashing>::with_slots(v.len());
             for (i, val) in v.iter().enumerate() {
                env1.set(i as u32, val.clone());
             }

             let mut env2 = env1.clone();
             env2.set(0, extra);

             if env1 != env2 {
                 prop_assert_ne!(calculate_hash(&env1), calculate_hash(&env2));
                 prop_assert_ne!(env1.sig, env2.sig);
             }
        }
    }
}

#[cfg(test)]
mod lazy_signature_tests {
    use super::*;
    use crate::analysis::resolver::NameId;
    use crate::simulator::hash_utils::{NoHashing, WithHashing};

    fn fx_hash<T: Hash + ?Sized>(t: &T) -> u64 {
        let mut h = FxHasher::default();
        t.hash(&mut h);
        h.finish()
    }

    /// One value of every kind, strings on both sides of the inline length,
    /// and composites nested inside each other.
    fn value_set<H: HashPolicy>() -> Vec<Value<H>> {
        let node = |role: usize, index: usize| NodeId {
            role: NameId(role),
            index,
        };
        let s = |t: &str| Value::<H>::string(EcoString::from(t));
        let mut inner = ValueMap::<H>::default();
        inner.insert(s("k"), Value::<H>::int(1));
        inner.insert(Value::<H>::int(2), s("a string longer than fifteen bytes"));
        let mut outer = ValueMap::<H>::default();
        outer.insert(s("nested"), Value::<H>::map(inner.clone()));
        outer.insert(Value::<H>::node(node(0, 1)), Value::<H>::list(EcoVec::new()));
        vec![
            Value::<H>::int(0),
            Value::<H>::int(-7),
            Value::<H>::int(i64::MAX),
            Value::<H>::bool(true),
            Value::<H>::bool(false),
            Value::<H>::unit(),
            s(""),
            s("short"),
            s("exactly15bytes!"),
            s("sixteen bytes!!!"),
            s("a string well past the inline limit of the small string"),
            Value::<H>::node(node(0, 0)),
            Value::<H>::node(node(1, 5)),
            Value::<H>::channel(ChannelId {
                node: node(0, 2),
                id: 9,
            }),
            Value::<H>::channel(ChannelId {
                node: node(1, 0),
                id: 0,
            }),
            Value::<H>::fifo_link(LinkId(3), node(0, 1)),
            Value::<H>::fifo_link(LinkId(0), node(1, 2)),
            Value::<H>::option_none(),
            Value::<H>::option_some(Value::<H>::int(4)),
            Value::<H>::option_some(s("a string longer than fifteen bytes")),
            Value::<H>::variant(0, EcoString::from("Empty"), None),
            Value::<H>::variant(1, EcoString::from("Prepare"), Some(Arc::new(Value::<H>::int(3)))),
            Value::<H>::tuple(EcoVec::from([Value::<H>::int(1), s("t")])),
            Value::<H>::list(EcoVec::from([Value::<H>::unit(), Value::<H>::map(inner.clone())])),
            Value::<H>::map(inner),
            Value::<H>::map(outer),
        ]
    }

    #[test]
    fn hash_matches_the_signature_stored_at_construction() {
        let lazy = value_set::<NoHashing>();
        let eager = value_set::<WithHashing>();
        assert_eq!(lazy.len(), eager.len());
        for (l, e) in lazy.iter().zip(&eager) {
            // A leaf-only signature stored at construction was the leaf's
            // full signature, which does not depend on any other value.
            let stored = if l.kind.is_composite() { 0 } else { e.sig };
            assert_eq!(stored, Value::<NoHashing>::compute_sig_leaf_only(&l.kind), "{l:?}");
            assert_eq!(fx_hash(l), fx_hash(&stored), "{l:?}");
            assert_eq!(l.sig, 0, "{l:?}");
            assert_eq!(e.sig, Value::<WithHashing>::compute_sig(&e.kind), "{e:?}");
            assert_eq!(fx_hash(e), fx_hash(&e.sig), "{e:?}");
        }
    }

    /// A map key hashed with the signature the value stored at construction.
    #[derive(Clone, PartialEq, Eq)]
    struct StoredSigKey(Value<NoHashing>, u64);

    impl Hash for StoredSigKey {
        fn hash<Ha: Hasher>(&self, state: &mut Ha) {
            self.1.hash(state);
        }
    }

    #[test]
    fn map_iteration_order_matches_hashing_the_stored_signature() {
        let keys = value_set::<NoHashing>();
        let mut current = ValueMap::<NoHashing>::default();
        let mut reference: imbl::GenericHashMap<
            StoredSigKey,
            usize,
            BuildHasherDefault<FxHasher>,
            imbl::shared_ptr::DefaultSharedPtr,
        > = imbl::GenericHashMap::default();
        for round in 0..3usize {
            for (i, k) in keys.iter().enumerate() {
                let stored = Value::<NoHashing>::compute_sig_leaf_only(&k.kind);
                if (i + round) % 4 == 3 {
                    current.remove(k);
                    reference.remove(&StoredSigKey(k.clone(), stored));
                } else {
                    current.insert(k.clone(), Value::<NoHashing>::int((i * 10 + round) as i64));
                    reference.insert(StoredSigKey(k.clone(), stored), i * 10 + round);
                }
            }
            let a: Vec<(Value<NoHashing>, i64)> = current
                .iter()
                .map(|(k, v)| (k.clone(), v.as_int().unwrap()))
                .collect();
            let b: Vec<(Value<NoHashing>, i64)> = reference
                .iter()
                .map(|(k, v)| (k.0.clone(), *v as i64))
                .collect();
            assert_eq!(a, b, "round {round}");
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::simulator::hash_utils::WithHashing;

    /// `Value` is copied and cloned on every slot read, record enqueue and
    /// collection update, so its width sets the memory traffic of the whole
    /// interpreter. Every variant must stay behind a pointer-sized handle;
    /// storing a collection inline widens all of them at once.
    #[test]
    fn value_stays_narrow() {
        assert_eq!(std::mem::size_of::<Value<WithHashing>>(), 40);
        assert_eq!(std::mem::size_of::<ValueSeq<WithHashing>>(), 16);
    }
}

#[cfg(test)]
mod acted_token_tests {
    use super::*;
    use crate::simulator::hash_utils::WithHashing;

    #[test]
    fn write_counter_moves_only_on_write() {
        let mut env = Env::<WithHashing>::with_slots(4);
        let before = env.writes;
        let _ = env.get(0);
        assert_eq!(env.writes, before, "reads must not count as writes");
        env.set(1, Value::<WithHashing>::unit());
        assert_ne!(env.writes, before, "a write must move the token");
    }

    #[test]
    fn write_counter_catches_what_a_slots_pointer_misses() {
        // An unshared env mutates in place, so the slots pointer is unchanged
        // across a real write. Anything comparing pointers to decide whether a
        // node was written reports "not written" for this case.
        let mut env = Env::<WithHashing>::with_slots(4);
        let ptr_before = env.slots.as_ptr();
        let writes_before = env.writes;
        env.set(2, Value::<WithHashing>::unit());
        assert_eq!(env.slots.as_ptr(), ptr_before, "in-place write, same buffer");
        assert_ne!(env.writes, writes_before, "the counter must still move");
    }

    #[test]
    fn same_value_rewrite_still_counts() {
        // The counter reports that a write happened, not that the value
        // differs; callers that need value equality must compare values.
        let mut env = Env::<WithHashing>::with_slots(2);
        env.set(0, Value::<WithHashing>::unit());
        let after_first = env.writes;
        env.set(0, Value::<WithHashing>::unit());
        assert_eq!(env.writes, after_first + 1);
    }
}
