use serde::Serialize;
use std::str::FromStr;

/// A unique identifier for every named entity (variable, function, type, role).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct NameId(pub usize);

impl std::fmt::Display for NameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NameId({})", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFn {
    Println,
    IntToString,
    BoolToString,
    RoleToString,
    UniqueId,
}

impl FromStr for BuiltinFn {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "println" => Ok(BuiltinFn::Println),
            "int_to_string" => Ok(BuiltinFn::IntToString),
            "bool_to_string" => Ok(BuiltinFn::BoolToString),
            "role_to_string" => Ok(BuiltinFn::RoleToString),
            "unique_id" => Ok(BuiltinFn::UniqueId),
            _ => Err(()),
        }
    }
}
