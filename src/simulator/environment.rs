use crate::analysis::resolver::NameId;
use imbl::{OrdMap, Vector};

use super::core::Value;

pub trait Environment: Clone {
    fn empty() -> Self;
    fn get(&self, sym: NameId) -> Option<&Value>;
    fn insert(&self, sym: NameId, val: Value) -> Self;
}

/// Environment backed by an ordered map. O(log n) get and insert.
#[derive(Clone, Debug)]
pub struct OrdMapEnv(OrdMap<NameId, Value>);

impl Environment for OrdMapEnv {
    fn empty() -> Self {
        OrdMapEnv(OrdMap::new())
    }

    fn get(&self, sym: NameId) -> Option<&Value> {
        self.0.get(&sym)
    }

    fn insert(&self, sym: NameId, val: Value) -> Self {
        let mut map = self.0.clone();
        map.insert(sym, val);
        OrdMapEnv(map)
    }
}

/// Environment backed by a vector with linear search. O(n) get, O(1) insert.
#[derive(Clone, Debug)]
pub struct VectorEnv(Vector<(NameId, Value)>);

impl Environment for VectorEnv {
    fn empty() -> Self {
        VectorEnv(Vector::new())
    }

    fn get(&self, sym: NameId) -> Option<&Value> {
        self.0.iter().find(|(k, _)| *k == sym).map(|(_, v)| v)
    }

    fn insert(&self, sym: NameId, val: Value) -> Self {
        let mut vec = self.0.clone();
        vec.push_front((sym, val));
        VectorEnv(vec)
    }
}
