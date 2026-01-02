use crate::simulator::core::error::RuntimeError;
use crate::simulator::hash_utils::mix;
use ecow::EcoString;
use imbl::{HashMap as ImHashMap, Vector};
use std::cmp::Ordering;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId {
    pub node: usize,
    pub id: usize,
}

/// The inner representation of a value, without cached signature.
#[derive(Clone, Debug)]
pub enum ValueKind {
    Int(i64),
    Bool(bool),
    Map(ImHashMap<Value, Value>),
    List(Vector<Value>),
    Option(Option<Arc<Value>>),
    Channel(ChannelId),
    Node(usize),
    String(EcoString),
    Unit,
    Tuple(Vector<Value>),
}

#[derive(Clone, Debug)]
pub struct Value {
    pub kind: ValueKind,
    pub sig: u64,
}

/// Helper to securely combine K and V without needing map position.
/// Uses mix with different tags (0 for key, 1 for value) to bind them tightly
/// and avoid collisions from key/value swaps.
#[inline]
pub fn hash_map_entry(k_sig: u64, v_sig: u64) -> u64 {
    mix(k_sig, 0) ^ mix(v_sig, 1)
}

impl Value {
    /// Create a new Value with computed signature.
    pub fn new(kind: ValueKind) -> Self {
        let sig = Self::compute_sig(&kind);
        Self { kind, sig }
    }

    /// Compute signature for a ValueKind.
    fn compute_sig(kind: &ValueKind) -> u64 {
        match kind {
            ValueKind::Int(i) => {
                let mut h = DefaultHasher::new();
                0u8.hash(&mut h); // discriminant
                i.hash(&mut h);
                h.finish()
            }
            ValueKind::Bool(b) => {
                let mut h = DefaultHasher::new();
                1u8.hash(&mut h);
                b.hash(&mut h);
                h.finish()
            }
            ValueKind::String(s) => {
                let mut h = DefaultHasher::new();
                2u8.hash(&mut h);
                s.hash(&mut h);
                h.finish()
            }
            ValueKind::Node(n) => {
                let mut h = DefaultHasher::new();
                3u8.hash(&mut h);
                n.hash(&mut h);
                h.finish()
            }
            ValueKind::Unit => {
                let mut h = DefaultHasher::new();
                4u8.hash(&mut h);
                h.finish()
            }
            ValueKind::Channel(c) => {
                let mut h = DefaultHasher::new();
                5u8.hash(&mut h);
                c.hash(&mut h);
                h.finish()
            }
            ValueKind::Option(o) => {
                let mut h = DefaultHasher::new();
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
                let mut h = DefaultHasher::new();
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
                let mut h = DefaultHasher::new();
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
                let mut h = DefaultHasher::new();
                9u8.hash(&mut h);
                m.len().hash(&mut h);
                sig ^= h.finish();
                for (k, v) in m.iter() {
                    // Use hash_map_entry for consistent entry hashing
                    sig ^= hash_map_entry(k.sig, v.sig);
                }
                sig
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
    pub fn node(n: usize) -> Self {
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
    pub fn option(o: Option<Arc<Value>>) -> Self {
        Self::new(ValueKind::Option(o))
    }

    #[inline]
    pub fn option_some(v: Value) -> Self {
        Self::new(ValueKind::Option(Some(Arc::new(v))))
    }

    #[inline]
    pub fn option_none() -> Self {
        Self::new(ValueKind::Option(None))
    }

    #[inline]
    pub fn tuple(v: Vector<Value>) -> Self {
        Self::new(ValueKind::Tuple(v))
    }

    #[inline]
    pub fn list(v: Vector<Value>) -> Self {
        Self::new(ValueKind::List(v))
    }

    #[inline]
    pub fn map(m: ImHashMap<Value, Value>) -> Self {
        Self::new(ValueKind::Map(m))
    }

    /// Create a Value with a pre-computed signature (for incremental updates)
    #[inline]
    pub fn with_sig(kind: ValueKind, sig: u64) -> Self {
        Self { kind, sig }
    }
}

impl PartialEq for Value {
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
            _ => false,
        }
    }
}
impl Eq for Value {}
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
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
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sig.hash(state);
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ValueKind::*;
        match &self.kind {
            Int(n) => write!(f, "{}", n),
            Bool(b) => write!(f, "{}", b),
            String(s) => write!(f, "\"{}\"", s),
            Node(n) => write!(f, "node({})", n),
            Unit => write!(f, "()") ,
            Option(None) => write!(f, "None") ,
            Option(Some(v)) => write!(f, "Some({})", v),
            Channel(ch) => write!(f, "channel({}, {})", ch.node, ch.id),
            Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Map(map) => {
                write!(f, "{{ ")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, " }}")
            }
        }
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        use ValueKind::*;
        match &self.kind {
            Int(_) => "int",
            Bool(_) => "bool",
            Map(_) => "map",
            List(_) => "list",
            Option(_) => "option",
            Channel(_) => "channel",
            Node(_) => "node",
            String(_) => "string",
            Unit => "unit",
            Tuple(_) => "tuple",
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
    pub fn as_node(&self) -> Result<usize, RuntimeError> {
        if let ValueKind::Node(n) = &self.kind {
            Ok(*n)
        } else {
            Err(RuntimeError::TypeError {
                expected: "node",
                got: self.type_name(),
            })
        }
    }
    pub fn as_map(&self) -> Result<&ImHashMap<Value, Value>, RuntimeError> {
        if let ValueKind::Map(m) = &self.kind {
            Ok(m)
        } else {
            Err(RuntimeError::TypeError {
                expected: "map",
                got: self.type_name(),
            })
        }
    }
    pub fn as_list(&self) -> Result<&Vector<Value>, RuntimeError> {
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
}

impl PartialEq for Env {
    fn eq(&self, other: &Self) -> bool {
        self.slots == other.slots
    }
}

impl Eq for Env {}

#[derive(Clone, Debug, Default)]
pub struct Env {
    pub slots: Vec<Value>,
    pub sig: u64,
}

impl Hash for Env {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sig.hash(state);
    }
}

impl Env {
    /// Create an environment with `n` slots, all initialized to Unit
    pub fn with_slots(n: usize) -> Self {
        let unit_val = Value::unit();
        let unit_sig = unit_val.sig;
        let slots: Vec<Value> = (0..n).map(|_| Value::unit()).collect();

        let mut sig = 0u64;
        for i in 0..n {
            sig ^= mix(unit_sig, i as u32);
        }

        Self { slots, sig }
    }

    #[inline(always)]
    pub fn get(&self, slot: u32) -> &Value {
        &self.slots[slot as usize]
    }

    #[inline(always)]
    pub fn set(&mut self, slot: u32, value: Value) {
        let idx = slot as usize;
        if idx >= self.slots.len() {
            // Extending requires recomputing signature
            let old_len = self.slots.len();
            self.slots.resize(idx + 1, Value::unit());
            let unit_sig = Value::unit().sig;
            for i in old_len..idx {
                self.sig ^= mix(unit_sig, i as u32);
            }
            self.sig ^= mix(value.sig, slot);
            self.slots[idx] = value;
        } else {
            let old_sig = self.slots[idx].sig;
            self.sig ^= mix(old_sig, slot) ^ mix(value.sig, slot);
            self.slots[idx] = value;
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Ensure we have at least `n` slots
    pub fn ensure_slots(&mut self, n: usize) {
        if self.slots.len() < n {
            let old_len = self.slots.len();
            self.slots.resize(n, Value::unit());
            let unit_sig = Value::unit().sig;
            for i in old_len..n {
                self.sig ^= mix(unit_sig, i as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    // Strategy for ChannelId
    prop_compose! {
        fn arb_channel_id()(node in any::<usize>(), id in any::<usize>()) -> ChannelId {
            ChannelId { node, id }
        }
    }

    // Strategy for Value
    fn arb_value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            any::<i64>().prop_map(Value::int),
            any::<bool>().prop_map(Value::bool),
            any::<String>().prop_map(|s| Value::string(EcoString::from(s))),
            any::<usize>().prop_map(Value::node),
            Just(Value::unit()),
            arb_channel_id().prop_map(Value::channel),
        ];

        leaf.prop_recursive(
            4, // 4 levels deep
            64, // max size 64 nodes
            10, // 10 items per collection
            |inner| {
                prop_oneof![
                    // Option
                    prop::option::of(inner.clone()).prop_map(|opt| {
                         match opt {
                             Some(v) => Value::option_some(v),
                             None => Value::option_none(),
                         }
                    }),
                    // List
                    prop::collection::vec(inner.clone(), 0..5).prop_map(|v| Value::list(v.into())),
                    // Tuple
                    prop::collection::vec(inner.clone(), 0..5).prop_map(|v| Value::tuple(v.into())),
                    // Map
                    prop::collection::hash_map(inner.clone(), inner.clone(), 0..5).prop_map(|m| Value::map(m.into())),
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
            let mut env1 = Env::with_slots(v.len());
            let mut env2 = Env::with_slots(v.len());

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
             let mut env1 = Env::with_slots(v.len());
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
