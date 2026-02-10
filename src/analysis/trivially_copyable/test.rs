use super::*;
use crate::analysis::resolver::NameId;
use crate::analysis::types::{Type, TypedProgram};
use std::collections::HashMap;

fn id(i: usize) -> NameId {
    NameId(i)
}

fn empty_program() -> TypedProgram {
    TypedProgram {
        top_level_defs: Vec::new(),
        next_name_id: 100,
        id_to_name: HashMap::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
    }
}

#[test]
fn test_primitives_are_trivially_copyable() {
    let map = TriviallyCopyableMap::new();
    assert!(is_trivially_copyable(&Type::Int, &map));
    assert!(is_trivially_copyable(&Type::String, &map));
    assert!(is_trivially_copyable(&Type::Bool, &map));
    assert!(is_trivially_copyable(&Type::Nil, &map));
    assert!(is_trivially_copyable(&Type::Never, &map));
}

#[test]
fn test_chan_is_not_trivially_copyable() {
    let map = TriviallyCopyableMap::new();
    assert!(!is_trivially_copyable(
        &Type::Chan(Box::new(Type::Int)),
        &map
    ));
}

#[test]
fn test_struct_all_primitives_is_trivial() {
    let mut program = empty_program();
    let struct_id = id(10);
    program.struct_defs.insert(
        struct_id,
        vec![
            ("x".to_string(), Type::Int),
            ("y".to_string(), Type::String),
        ],
    );

    let map = compute_trivially_copyable(&program);
    assert_eq!(map[&struct_id], TriviallyCopyable::Trivial);
}

#[test]
fn test_struct_with_chan_field_is_non_trivial() {
    let mut program = empty_program();
    let struct_id = id(10);
    program.struct_defs.insert(
        struct_id,
        vec![
            ("x".to_string(), Type::Int),
            ("ch".to_string(), Type::Chan(Box::new(Type::Int))),
        ],
    );

    let map = compute_trivially_copyable(&program);
    assert_eq!(map[&struct_id], TriviallyCopyable::NonTrivial);
}

#[test]
fn test_struct_containing_non_trivial_struct_is_non_trivial() {
    let mut program = empty_program();

    // Inner struct has a channel field.
    let inner_id = id(10);
    program.struct_defs.insert(
        inner_id,
        vec![("ch".to_string(), Type::Chan(Box::new(Type::Int)))],
    );

    // Outer struct contains the inner struct.
    let outer_id = id(11);
    program.struct_defs.insert(
        outer_id,
        vec![(
            "inner".to_string(),
            Type::Struct(inner_id, "Inner".to_string()),
        )],
    );

    let map = compute_trivially_copyable(&program);
    assert_eq!(map[&inner_id], TriviallyCopyable::NonTrivial);
    assert_eq!(map[&outer_id], TriviallyCopyable::NonTrivial);
}

#[test]
fn test_enum_with_chan_payload_is_non_trivial() {
    let mut program = empty_program();
    let enum_id = id(10);
    program.enum_defs.insert(
        enum_id,
        vec![
            ("None".to_string(), None),
            ("HasChan".to_string(), Some(Type::Chan(Box::new(Type::Int)))),
        ],
    );

    let map = compute_trivially_copyable(&program);
    assert_eq!(map[&enum_id], TriviallyCopyable::NonTrivial);
}

#[test]
fn test_enum_without_chan_is_trivial() {
    let mut program = empty_program();
    let enum_id = id(10);
    program.enum_defs.insert(
        enum_id,
        vec![("A".to_string(), None), ("B".to_string(), Some(Type::Int))],
    );

    let map = compute_trivially_copyable(&program);
    assert_eq!(map[&enum_id], TriviallyCopyable::Trivial);
}

#[test]
fn test_list_of_chan_is_non_trivial() {
    let map = TriviallyCopyableMap::new();
    assert!(!is_trivially_copyable(
        &Type::List(Box::new(Type::Chan(Box::new(Type::Int)))),
        &map
    ));
}

#[test]
fn test_optional_chan_is_non_trivial() {
    let map = TriviallyCopyableMap::new();
    assert!(!is_trivially_copyable(
        &Type::Optional(Box::new(Type::Chan(Box::new(Type::Int)))),
        &map
    ));
}

#[test]
fn test_map_with_chan_value_is_non_trivial() {
    let map = TriviallyCopyableMap::new();
    assert!(!is_trivially_copyable(
        &Type::Map(
            Box::new(Type::Int),
            Box::new(Type::Chan(Box::new(Type::String)))
        ),
        &map
    ));
}

#[test]
fn test_role_is_trivially_copyable() {
    let map = TriviallyCopyableMap::new();
    assert!(is_trivially_copyable(
        &Type::Role(id(5), "MyRole".to_string()),
        &map
    ));
}
