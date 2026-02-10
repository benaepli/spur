#[cfg(test)]
mod test;

use crate::analysis::resolver::NameId;
use crate::analysis::types::{Type, TypedProgram};
use std::collections::HashMap;

/// Whether a type can be trivially duplicated without ownership concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriviallyCopyable {
    Trivial,
    NonTrivial,
}

/// Maps user-defined type NameIds to their trivial copyability.
pub type TriviallyCopyableMap = HashMap<NameId, TriviallyCopyable>;

/// Compute the trivial copyability of all user-defined types in a typed program.
///
/// Uses fixed-point iteration: initially assume all types are trivially copyable,
/// then repeatedly check until no changes occur. This correctly handles mutually
/// recursive type definitions.
pub fn compute_trivially_copyable(program: &TypedProgram) -> TriviallyCopyableMap {
    let mut map = TriviallyCopyableMap::new();

    for name_id in program.struct_defs.keys() {
        map.insert(*name_id, TriviallyCopyable::Trivial);
    }
    for name_id in program.enum_defs.keys() {
        map.insert(*name_id, TriviallyCopyable::Trivial);
    }

    loop {
        let mut changed = false;

        for (name_id, fields) in &program.struct_defs {
            if map[name_id] == TriviallyCopyable::NonTrivial {
                continue; // Already non-trivial, can't change.
            }
            for (_field_name, field_ty) in fields {
                if !is_trivially_copyable(field_ty, &map) {
                    map.insert(*name_id, TriviallyCopyable::NonTrivial);
                    changed = true;
                    break;
                }
            }
        }

        for (name_id, variants) in &program.enum_defs {
            if map[name_id] == TriviallyCopyable::NonTrivial {
                continue;
            }
            for (_variant_name, payload_ty) in variants {
                if let Some(ty) = payload_ty {
                    if !is_trivially_copyable(ty, &map) {
                        map.insert(*name_id, TriviallyCopyable::NonTrivial);
                        changed = true;
                        break;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    map
}

/// Check whether a given type is trivially copyable.
///
/// Primitive types are always trivially copyable. Channels are never trivially copyable.
/// Compound types are trivially copyable iff all their components are.
/// User-defined types (structs/enums) are looked up in the map.
pub fn is_trivially_copyable(ty: &Type, map: &TriviallyCopyableMap) -> bool {
    match ty {
        Type::Int | Type::String | Type::Bool => true,

        Type::Chan(_) => false,

        Type::List(elem) => is_trivially_copyable(elem, map),
        Type::Map(key, val) => is_trivially_copyable(key, map) && is_trivially_copyable(val, map),
        Type::Tuple(elems) => elems.iter().all(|e| is_trivially_copyable(e, map)),
        Type::Optional(inner) => is_trivially_copyable(inner, map),

        Type::Struct(id, _) | Type::Enum(id, _) => {
            map.get(id)
                .map(|c| *c == TriviallyCopyable::Trivial)
                .unwrap_or(true) // Unknown types default to trivial.
        }

        // Roles are just identifiers.
        Type::Role(_, _) => true,

        Type::EmptyList | Type::EmptyMap | Type::UnknownChannel => true,
        Type::Nil => true,
        Type::Never => true,
    }
}
