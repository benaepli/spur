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
fn test_primitives_get_distinct_ids() {
    let mut program = empty_program();
    // Put some types into struct fields so they get discovered.
    program.struct_defs.insert(
        id(1),
        vec![
            ("a".to_string(), Type::Int),
            ("b".to_string(), Type::String),
            ("c".to_string(), Type::Bool),
        ],
    );

    let map = assign_type_ids(&program);
    assert_ne!(map[&Type::Int], map[&Type::String]);
    assert_ne!(map[&Type::Int], map[&Type::Bool]);
    assert_ne!(map[&Type::String], map[&Type::Bool]);
}

#[test]
fn test_equal_types_get_same_id() {
    let mut program = empty_program();
    let list_int = Type::List(Box::new(Type::Int));
    // Two fields with the same type.
    program.struct_defs.insert(
        id(1),
        vec![
            ("a".to_string(), list_int.clone()),
            ("b".to_string(), list_int.clone()),
        ],
    );

    let map = assign_type_ids(&program);
    // Should only have 2 entries: List<Int> and Int.
    assert_eq!(map.len(), 2);
    assert!(map.contains_key(&list_int));
    assert!(map.contains_key(&Type::Int));
}

#[test]
fn test_compound_types_register_components() {
    let mut program = empty_program();
    let map_type = Type::Map(Box::new(Type::String), Box::new(Type::Int));
    program
        .struct_defs
        .insert(id(1), vec![("m".to_string(), map_type.clone())]);

    let map = assign_type_ids(&program);
    assert!(map.contains_key(&map_type));
    assert!(map.contains_key(&Type::String));
    assert!(map.contains_key(&Type::Int));
}

#[test]
fn test_tuple_registers_element_types() {
    let mut program = empty_program();
    let tuple_type = Type::Tuple(vec![Type::Int, Type::Bool]);
    program
        .struct_defs
        .insert(id(1), vec![("t".to_string(), tuple_type.clone())]);

    let map = assign_type_ids(&program);
    assert!(map.contains_key(&tuple_type));
    assert!(map.contains_key(&Type::Int));
    assert!(map.contains_key(&Type::Bool));
}

#[test]
fn test_optional_registers_inner() {
    let mut program = empty_program();
    let opt = Type::Optional(Box::new(Type::String));
    program
        .struct_defs
        .insert(id(1), vec![("o".to_string(), opt.clone())]);

    let map = assign_type_ids(&program);
    assert!(map.contains_key(&opt));
    assert!(map.contains_key(&Type::String));
}

#[test]
fn test_different_types_get_different_ids() {
    let mut program = empty_program();
    let list_int = Type::List(Box::new(Type::Int));
    let list_string = Type::List(Box::new(Type::String));
    program.struct_defs.insert(
        id(1),
        vec![
            ("a".to_string(), list_int.clone()),
            ("b".to_string(), list_string.clone()),
        ],
    );

    let map = assign_type_ids(&program);
    assert_ne!(map[&list_int], map[&list_string]);
}

#[test]
fn test_enum_payloads_registered() {
    let mut program = empty_program();
    program.enum_defs.insert(
        id(1),
        vec![
            ("None".to_string(), None),
            ("Some".to_string(), Some(Type::Int)),
        ],
    );

    let map = assign_type_ids(&program);
    assert!(map.contains_key(&Type::Int));
}

#[test]
fn test_empty_program_produces_empty_map() {
    let program = empty_program();
    let map = assign_type_ids(&program);
    assert!(map.is_empty());
}
