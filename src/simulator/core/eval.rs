use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Expr, FunctionInfo, Lhs, VarSlot};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::state::NodeId;
use crate::simulator::core::values::{Env, Value, ValueKind, hash_map_entry};
use crate::simulator::hash_utils::HashPolicy;
use ecow::EcoString;
use imbl::{HashMap as ImHashMap, Vector};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

#[inline(always)]
pub fn load<H: HashPolicy>(slot: VarSlot, local_env: &Env<H>, node_env: &Env<H>) -> Value<H> {
    match slot {
        VarSlot::Local(idx, _) => local_env.get(idx).clone(),
        VarSlot::Node(idx, _) => node_env.get(idx).clone(),
    }
}

#[inline(always)]
pub fn store_slot<H: HashPolicy>(
    slot: VarSlot,
    val: Value<H>,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
) {
    match slot {
        VarSlot::Local(idx, _) => local_env.set(idx, val),
        VarSlot::Node(idx, _) => node_env.set(idx, val),
    }
}

pub fn store<H: HashPolicy>(
    lhs: &Lhs,
    val: Value<H>,
    local_env: &mut Env<H>,
    node_env: &mut Env<H>,
) -> Result<(), RuntimeError> {
    match lhs {
        Lhs::Var(slot) => {
            store_slot(*slot, val, local_env, node_env);
            Ok(())
        }
        Lhs::Tuple(slots) => {
            if let ValueKind::Tuple(vals) = val.kind {
                if slots.len() != vals.len() {
                    return Err(RuntimeError::TupleLengthMismatch {
                        expected: slots.len(),
                        got: vals.len(),
                    });
                }
                for (slot, v) in slots.iter().zip(vals.into_iter()) {
                    store_slot(*slot, v, local_env, node_env);
                }
                Ok(())
            } else {
                Err(RuntimeError::NonTupleAssignment {
                    got: val.type_name(),
                })
            }
        }
    }
}

/// Create a fresh local environment for calling a function
pub fn make_local_env<H: HashPolicy>(
    func: &FunctionInfo,
    args: Vec<Value<H>>,
    local_env: &Env<H>,
    node_env: &Env<H>,
    role_names: &HashMap<NameId, String>,
) -> Env<H> {
    let mut env = Env::<H>::with_slots(func.local_slot_count as usize);

    // Set arguments in parameter slots
    for (i, arg) in args.into_iter().enumerate() {
        env.set(i as u32, arg);
    }

    // Initialize other locals to their defaults
    for (i, default_expr) in func.local_defaults.iter().enumerate() {
        let slot = func.param_count + i as u32;
        if let Ok(val) = eval(local_env, node_env, default_expr, role_names) {
            env.set(slot, val);
        }
    }

    env
}

fn update_collection<H: HashPolicy>(
    col: Value<H>,
    key: Value<H>,
    val: Value<H>,
) -> Result<Value<H>, RuntimeError> {
    use ValueKind::*;
    match col.kind {
        Map(m) => {
            let mut new_sig = col.sig; // Start with existing signature

            // Remove the old entry's contribution (if it exists)
            if let Some(old_val) = m.get(&key) {
                let old_entry_hash = hash_map_entry(key.sig, old_val.sig);
                new_sig ^= old_entry_hash; // XOR removes it
            // Also remove old length contribution and add adjusted length
            // (length stays the same, so no change needed)
            } else {
                // Key doesn't exist, length will increase by 1
                // Remove old length hash, add new length hash
                let mut h = DefaultHasher::new();
                9u8.hash(&mut h);
                m.len().hash(&mut h);
                new_sig ^= h.finish();

                let mut h = DefaultHasher::new();
                9u8.hash(&mut h);
                (m.len() + 1).hash(&mut h);
                new_sig ^= h.finish();
            }

            // Add the new entry's contribution
            let new_entry_hash = hash_map_entry(key.sig, val.sig);
            new_sig ^= new_entry_hash;

            let new_map = m.update(key, val);

            Ok(Value::<H>::with_sig(ValueKind::Map(new_map), new_sig))
        }
        List(l) => {
            let idx = key.as_int()? as usize;
            if idx >= l.len() {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: idx,
                    len: l.len(),
                });
            }
            Ok(Value::<H>::list(l.update(idx, val)))
        }
        _ => Err(RuntimeError::NotACollection {
            got: col.type_name(),
        }),
    }
}

pub fn eval<H: HashPolicy>(
    local_env: &Env<H>,
    node_env: &Env<H>,
    expr: &Expr,
    role_names: &HashMap<NameId, String>,
) -> Result<Value<H>, RuntimeError> {
    match expr {
        Expr::Int(i) => Ok(Value::<H>::int(*i)),
        Expr::Bool(b) => Ok(Value::<H>::bool(*b)),
        Expr::String(s) => Ok(Value::<H>::string(s.clone())),
        Expr::Unit => Ok(Value::<H>::unit()),
        Expr::Nil => Ok(Value::<H>::option_none()),
        Expr::Var(s) => Ok(load(*s, local_env, node_env)),
        Expr::Plus(e1, e2) => {
            let v1 = eval(local_env, node_env, e1, role_names)?;
            let v2 = eval(local_env, node_env, e2, role_names)?;

            match (&v1.kind, &v2.kind) {
                (ValueKind::Int(i1), ValueKind::Int(i2)) => Ok(Value::<H>::int(i1 + i2)),
                (ValueKind::String(s1), ValueKind::String(s2)) => {
                    let mut result = String::new();
                    result.push_str(s1.as_str());
                    result.push_str(s2.as_str());
                    Ok(Value::<H>::string(EcoString::from(result)))
                }
                _ => Err(RuntimeError::TypeError {
                    expected: "int or string",
                    got: v1.type_name(),
                }),
            }
        }
        Expr::Minus(e1, e2) => Ok(Value::<H>::int(
            eval(local_env, node_env, e1, role_names)?.as_int()?
                - eval(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::Times(e1, e2) => Ok(Value::<H>::int(
            eval(local_env, node_env, e1, role_names)?.as_int()?
                * eval(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::Div(e1, e2) => Ok(Value::<H>::int(
            eval(local_env, node_env, e1, role_names)?.as_int()?
                / eval(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::Mod(e1, e2) => Ok(Value::<H>::int(
            eval(local_env, node_env, e1, role_names)?.as_int()?
                % eval(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::LessThan(e1, e2) => Ok(Value::<H>::bool(
            eval(local_env, node_env, e1, role_names)? < eval(local_env, node_env, e2, role_names)?,
        )),
        Expr::EqualsEquals(e1, e2) => Ok(Value::<H>::bool(
            eval(local_env, node_env, e1, role_names)?
                == eval(local_env, node_env, e2, role_names)?,
        )),
        Expr::Not(e) => Ok(Value::<H>::bool(
            !eval(local_env, node_env, e, role_names)?.as_bool()?,
        )),
        Expr::And(e1, e2) => Ok(Value::<H>::bool(
            eval(local_env, node_env, e1, role_names)?.as_bool()?
                && eval(local_env, node_env, e2, role_names)?.as_bool()?,
        )),
        Expr::Or(e1, e2) => Ok(Value::<H>::bool(
            eval(local_env, node_env, e1, role_names)?.as_bool()?
                || eval(local_env, node_env, e2, role_names)?.as_bool()?,
        )),
        Expr::Some(e) => Ok(Value::<H>::option_some(eval(
            local_env, node_env, e, role_names,
        )?)),
        Expr::Tuple(es) => {
            let vals: Result<Vector<_>, _> = es
                .iter()
                .map(|e| eval(local_env, node_env, e, role_names))
                .collect();
            Ok(Value::<H>::tuple(vals?))
        }
        Expr::List(es) => {
            let vals: Result<Vector<_>, _> = es
                .iter()
                .map(|e| eval(local_env, node_env, e, role_names))
                .collect();
            Ok(Value::<H>::list(vals?))
        }
        Expr::Map(kv) => {
            let mut m = ImHashMap::new();
            for (k, v) in kv {
                m.insert(
                    eval(local_env, node_env, k, role_names)?,
                    eval(local_env, node_env, v, role_names)?,
                );
            }
            Ok(Value::<H>::map(m))
        }
        Expr::Find(col, key) => {
            let col_val = eval(local_env, node_env, col, role_names)?;
            match &col_val.kind {
                ValueKind::Map(m) => {
                    let k = eval(local_env, node_env, key, role_names)?;
                    m.get(&k).cloned().ok_or(RuntimeError::KeyNotFound)
                }
                ValueKind::List(l) => {
                    let idx = eval(local_env, node_env, key, role_names)?.as_int()? as usize;
                    l.get(idx).cloned().ok_or(RuntimeError::IndexOutOfBounds {
                        index: idx,
                        len: l.len(),
                    })
                }
                _ => Err(RuntimeError::NotACollection {
                    got: col_val.type_name(),
                }),
            }
        }
        Expr::ListPrepend(head, tail) => {
            let h = eval(local_env, node_env, head, role_names)?;
            let t = eval(local_env, node_env, tail, role_names)?
                .as_list()?
                .clone();
            let mut new_list = Vector::new();
            new_list.push_back(h);
            new_list.append(t);
            Ok(Value::<H>::list(new_list))
        }
        Expr::ListAppend(list, item) => {
            let mut l = eval(local_env, node_env, list, role_names)?
                .as_list()?
                .clone();
            let i = eval(local_env, node_env, item, role_names)?;
            l.push_back(i);
            Ok(Value::<H>::list(l))
        }
        Expr::ListSubsequence(list, start, end) => {
            let l = eval(local_env, node_env, list, role_names)?;
            let s = eval(local_env, node_env, start, role_names)?.as_int()? as usize;
            let e = eval(local_env, node_env, end, role_names)?.as_int()? as usize;
            let vec = l.as_list()?;
            if s > vec.len() || e > vec.len() || s > e {
                return Err(RuntimeError::SubsequenceOutOfBounds {
                    start: s,
                    end: e,
                    len: vec.len(),
                });
            }
            Ok(Value::<H>::list(vec.clone().slice(s..e)))
        }
        Expr::LessThanEquals(e1, e2) => Ok(Value::<H>::bool(
            eval(local_env, node_env, e1, role_names)?
                <= eval(local_env, node_env, e2, role_names)?,
        )),
        Expr::GreaterThan(e1, e2) => Ok(Value::<H>::bool(
            eval(local_env, node_env, e1, role_names)? > eval(local_env, node_env, e2, role_names)?,
        )),
        Expr::GreaterThanEquals(e1, e2) => Ok(Value::<H>::bool(
            eval(local_env, node_env, e1, role_names)?
                >= eval(local_env, node_env, e2, role_names)?,
        )),
        Expr::KeyExists(key, map) => {
            let k = eval(local_env, node_env, key, role_names)?;
            let m = eval(local_env, node_env, map, role_names)?;
            Ok(Value::<H>::bool(m.as_map()?.contains_key(&k)))
        }
        Expr::MapErase(key, map) => {
            let k = eval(local_env, node_env, key, role_names)?;
            let m = eval(local_env, node_env, map, role_names)?
                .as_map()?
                .clone();
            Ok(Value::<H>::map(m.without(&k)))
        }
        Expr::ListLen(list) => {
            let list_val = eval(local_env, node_env, list, role_names)?;
            match &list_val.kind {
                ValueKind::List(l) => Ok(Value::<H>::int(l.len() as i64)),
                ValueKind::Map(m) => Ok(Value::<H>::int(m.len() as i64)),
                _ => Err(RuntimeError::NotACollection {
                    got: list_val.type_name(),
                }),
            }
        }
        Expr::ListAccess(list, idx) => {
            let l = eval(local_env, node_env, list, role_names)?;
            let vec = l.as_list()?;
            let i = *idx;
            if i >= vec.len() {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: i,
                    len: vec.len(),
                });
            }
            Ok(vec[i].clone())
        }
        Expr::Min(e1, e2) => {
            let v1 = eval(local_env, node_env, e1, role_names)?.as_int()?;
            let v2 = eval(local_env, node_env, e2, role_names)?.as_int()?;
            Ok(Value::<H>::int(v1.min(v2)))
        }
        Expr::TupleAccess(tuple, idx) => {
            let t = eval(local_env, node_env, tuple, role_names)?;
            if let ValueKind::Tuple(vec) = &t.kind {
                if *idx >= vec.len() {
                    return Err(RuntimeError::IndexOutOfBounds {
                        index: *idx,
                        len: vec.len(),
                    });
                }
                Ok(vec[*idx].clone())
            } else {
                Err(RuntimeError::TypeError {
                    expected: "tuple",
                    got: t.type_name(),
                })
            }
        }
        Expr::Unwrap(e) => {
            let val = eval(local_env, node_env, e, role_names)?;
            match &val.kind {
                ValueKind::Option(Some(v)) => Ok(Arc::unwrap_or_clone(v.clone())),
                ValueKind::Option(None) => Err(RuntimeError::UnwrapNone),
                _ => Err(RuntimeError::TypeError {
                    expected: "option",
                    got: val.type_name(),
                }),
            }
        }
        Expr::Coalesce(opt, default) => {
            let val = eval(local_env, node_env, opt, role_names)?;
            match &val.kind {
                ValueKind::Option(Some(v)) => Ok(Arc::unwrap_or_clone(v.clone())),
                ValueKind::Option(None) => eval(local_env, node_env, default, role_names),
                _ => Err(RuntimeError::CoalesceNonOption {
                    got: val.type_name(),
                }),
            }
        }
        Expr::IntToString(e) => Ok(Value::<H>::string(EcoString::from(
            eval(local_env, node_env, e, role_names)?
                .as_int()?
                .to_string(),
        ))),
        Expr::BoolToString(e) => Ok(Value::<H>::string(EcoString::from(
            eval(local_env, node_env, e, role_names)?
                .as_bool()?
                .to_string(),
        ))),
        Expr::NodeToString(e) => {
            let node_id = eval(local_env, node_env, e, role_names)?.as_node()?;
            let role_name = role_names
                .get(&node_id.role)
                .map(|s| s.as_str())
                .unwrap_or("Unknown");
            Ok(Value::<H>::string(EcoString::from(format!(
                "{}[{}]",
                role_name, node_id.index
            ))))
        }
        Expr::Store(col, key, val) => update_collection(
            eval(local_env, node_env, col, role_names)?,
            eval(local_env, node_env, key, role_names)?,
            eval(local_env, node_env, val, role_names)?,
        ),
        Expr::Variant(enum_id, name, payload) => {
            let payload_val = payload
                .as_ref()
                .map(|p| eval(local_env, node_env, p, role_names))
                .transpose()?
                .map(Arc::new);
            Ok(Value::<H>::variant(*enum_id, name.clone(), payload_val))
        }
        Expr::IsVariant(expr, name) => {
            let val = eval(local_env, node_env, expr, role_names)?;
            match &val.kind {
                ValueKind::Variant(_, variant_name, _) => {
                    Ok(Value::<H>::bool(variant_name == name))
                }
                _ => Ok(Value::<H>::bool(false)),
            }
        }
        Expr::VariantPayload(expr) => {
            let val = eval(local_env, node_env, expr, role_names)?;
            match &val.kind {
                ValueKind::Variant(_, _, Some(payload)) => Ok((**payload).clone()),
                ValueKind::Variant(_, _, None) => Err(RuntimeError::VariantHasNoPayload),
                _ => Err(RuntimeError::TypeError {
                    expected: "variant",
                    got: val.type_name(),
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::resolver::NameId;
    use crate::simulator::core::state::NodeId;
    use crate::simulator::hash_utils::WithHashing;

    fn dummy_slot(idx: u32) -> VarSlot {
        VarSlot::Local(idx, NameId(0))
    }

    fn node_slot(idx: u32) -> VarSlot {
        VarSlot::Node(idx, NameId(0))
    }

    #[test]
    fn test_eval_literals() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        assert_eq!(
            eval(&env, &env, &Expr::Int(42), &roles).unwrap(),
            Value::<WithHashing>::int(42)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Bool(true), &roles).unwrap(),
            Value::<WithHashing>::bool(true)
        );
        assert_eq!(
            eval(&env, &env, &Expr::String("hello".into()), &roles).unwrap(),
            Value::<WithHashing>::string("hello".into())
        );
        assert_eq!(
            eval(&env, &env, &Expr::Unit, &roles).unwrap(),
            Value::<WithHashing>::unit()
        );
        assert_eq!(
            eval(&env, &env, &Expr::Nil, &roles).unwrap(),
            Value::<WithHashing>::option_none()
        );
    }

    #[test]
    fn test_eval_arithmetic() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        let e1 = Box::new(Expr::Int(10));
        let e2 = Box::new(Expr::Int(3));

        assert_eq!(
            eval(&env, &env, &Expr::Plus(e1.clone(), e2.clone()), &roles).unwrap(),
            Value::<WithHashing>::int(13)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Minus(e1.clone(), e2.clone()), &roles).unwrap(),
            Value::<WithHashing>::int(7)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Times(e1.clone(), e2.clone()), &roles).unwrap(),
            Value::<WithHashing>::int(30)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Div(e1.clone(), e2.clone()), &roles).unwrap(),
            Value::<WithHashing>::int(3)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Mod(e1.clone(), e2.clone()), &roles).unwrap(),
            Value::<WithHashing>::int(1)
        );
    }

    #[test]
    fn test_eval_string_concat() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();

        // Basic concatenation
        let s1 = Box::new(Expr::String(EcoString::from("hello")));
        let s2 = Box::new(Expr::String(EcoString::from("world")));
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s1, s2), &roles).unwrap(),
            Value::<WithHashing>::string(EcoString::from("helloworld"))
        );

        // Concatenation with spaces
        let s3 = Box::new(Expr::String(EcoString::from("hello ")));
        let s4 = Box::new(Expr::String(EcoString::from("world")));
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s3, s4), &roles).unwrap(),
            Value::<WithHashing>::string(EcoString::from("hello world"))
        );

        // Empty strings
        let s5 = Box::new(Expr::String(EcoString::from("")));
        let s6 = Box::new(Expr::String(EcoString::from("test")));
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s5.clone(), s6.clone()), &roles).unwrap(),
            Value::<WithHashing>::string(EcoString::from("test"))
        );
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s6, s5), &roles).unwrap(),
            Value::<WithHashing>::string(EcoString::from("test"))
        );
    }

    #[test]
    fn test_eval_logical() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        let t = Box::new(Expr::Bool(true));
        let f = Box::new(Expr::Bool(false));

        assert_eq!(
            eval(&env, &env, &Expr::Not(t.clone()), &roles).unwrap(),
            Value::<WithHashing>::bool(false)
        );
        assert_eq!(
            eval(&env, &env, &Expr::And(t.clone(), f.clone()), &roles).unwrap(),
            Value::<WithHashing>::bool(false)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Or(t.clone(), f.clone()), &roles).unwrap(),
            Value::<WithHashing>::bool(true)
        );
    }

    #[test]
    fn test_eval_comparison() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        let e1 = Box::new(Expr::Int(10));
        let e2 = Box::new(Expr::Int(20));

        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::EqualsEquals(e1.clone(), e1.clone()),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::bool(true)
        );
        assert_eq!(
            eval(&env, &env, &Expr::LessThan(e1.clone(), e2.clone()), &roles).unwrap(),
            Value::<WithHashing>::bool(true)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::LessThanEquals(e1.clone(), e1.clone()),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::bool(true)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::GreaterThan(e2.clone(), e1.clone()),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::bool(true)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::GreaterThanEquals(e1.clone(), e1.clone()),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::bool(true)
        );
    }

    #[test]
    fn test_eval_vars() {
        let mut local_env = Env::<WithHashing>::with_slots(2);
        let roles = HashMap::new();
        local_env.set(0, Value::<WithHashing>::int(100));
        let mut node_env = Env::<WithHashing>::with_slots(1);
        node_env.set(0, Value::<WithHashing>::int(200));

        assert_eq!(
            eval(&local_env, &node_env, &Expr::Var(dummy_slot(0)), &roles).unwrap(),
            Value::<WithHashing>::int(100)
        );
        assert_eq!(
            eval(&local_env, &node_env, &Expr::Var(node_slot(0)), &roles).unwrap(),
            Value::<WithHashing>::int(200)
        );
    }

    #[test]
    fn test_eval_collections() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();

        // Tuple
        let tuple_expr = Expr::Tuple(vec![Expr::Int(1), Expr::Bool(true)]);
        let _tuple_val = eval(&env, &env, &tuple_expr, &roles).unwrap();
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::TupleAccess(Box::new(tuple_expr), 1),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::bool(true)
        );

        // List
        let list_expr = Expr::List(vec![Expr::Int(1), Expr::Int(2)]);
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::ListLen(Box::new(list_expr.clone())),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::int(2)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::ListAccess(Box::new(list_expr.clone()), 0),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::int(1)
        );

        // List operations
        let append_expr = Expr::ListAppend(Box::new(list_expr.clone()), Box::new(Expr::Int(3)));
        let appended_val = eval(&env, &env, &append_expr, &roles).unwrap();
        assert_eq!(appended_val.as_list().unwrap().len(), 3);

        let prepend_expr = Expr::ListPrepend(Box::new(Expr::Int(0)), Box::new(list_expr.clone()));
        let prepended_val = eval(&env, &env, &prepend_expr, &roles).unwrap();
        assert_eq!(
            prepended_val.as_list().unwrap()[0],
            Value::<WithHashing>::int(0)
        );

        let sub_expr = Expr::ListSubsequence(
            Box::new(list_expr),
            Box::new(Expr::Int(0)),
            Box::new(Expr::Int(1)),
        );
        let sub_val = eval(&env, &env, &sub_expr, &roles).unwrap();
        assert_eq!(sub_val.as_list().unwrap().len(), 1);

        // Map
        let map_expr = Expr::Map(vec![(Expr::String("key".into()), Expr::Int(42))]);
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Find(
                    Box::new(map_expr.clone()),
                    Box::new(Expr::String("key".into()))
                ),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::int(42)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::KeyExists(
                    Box::new(Expr::String("key".into())),
                    Box::new(map_expr.clone())
                ),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::bool(true)
        );

        let erase_expr = Expr::MapErase(Box::new(Expr::String("key".into())), Box::new(map_expr));
        let erased_val = eval(&env, &env, &erase_expr, &roles).unwrap();
        assert_eq!(erased_val.as_map().unwrap().len(), 0);
    }

    #[test]
    fn test_eval_options() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        let some_expr = Expr::Some(Box::new(Expr::Int(42)));
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Unwrap(Box::new(some_expr.clone())),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::int(42)
        );

        let nil_expr = Expr::Nil;
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Coalesce(Box::new(nil_expr), Box::new(Expr::Int(100))),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::int(100)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Coalesce(Box::new(some_expr), Box::new(Expr::Int(100))),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::int(42)
        );
    }

    #[test]
    fn test_eval_misc() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Min(Box::new(Expr::Int(10)), Box::new(Expr::Int(20))),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::int(10)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::IntToString(Box::new(Expr::Int(123))),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::string("123".into())
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::BoolToString(Box::new(Expr::Bool(true))),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::string("true".into())
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::BoolToString(Box::new(Expr::Bool(false))),
                &roles
            )
            .unwrap(),
            Value::<WithHashing>::string("false".into())
        );
    }

    #[test]
    fn test_eval_store_update() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        let list_expr = Expr::List(vec![Expr::Int(1)]);
        let store_expr = Expr::Store(
            Box::new(list_expr),
            Box::new(Expr::Int(0)),
            Box::new(Expr::Int(42)),
        );
        let updated_list = eval(&env, &env, &store_expr, &roles).unwrap();
        assert_eq!(
            updated_list.as_list().unwrap()[0],
            Value::<WithHashing>::int(42)
        );
    }

    #[test]
    fn test_eval_errors() {
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        let res = eval(
            &env,
            &env,
            &Expr::Plus(Box::new(Expr::Bool(true)), Box::new(Expr::Int(1))),
            &roles,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_store() {
        let mut local_env = Env::<WithHashing>::with_slots(1);
        let mut node_env = Env::<WithHashing>::with_slots(1);

        // Simple var store
        store(
            &Lhs::Var(dummy_slot(0)),
            Value::<WithHashing>::int(42),
            &mut local_env,
            &mut node_env,
        )
        .unwrap();
        assert_eq!(local_env.get(0), &Value::<WithHashing>::int(42));

        // Tuple store
        let lhs = Lhs::Tuple(vec![dummy_slot(0), node_slot(0)]);
        let val = Value::<WithHashing>::tuple(Vector::from(vec![
            Value::<WithHashing>::int(1),
            Value::<WithHashing>::int(2),
        ]));
        store(&lhs, val, &mut local_env, &mut node_env).unwrap();
        assert_eq!(local_env.get(0), &Value::<WithHashing>::int(1));
        assert_eq!(node_env.get(0), &Value::<WithHashing>::int(2));

        // Mismatched tuple length
        let val_bad = Value::<WithHashing>::tuple(Vector::from(vec![Value::<WithHashing>::int(1)]));
        assert!(store(&lhs, val_bad, &mut local_env, &mut node_env).is_err());
    }

    #[test]
    fn test_make_local_env() {
        let func = FunctionInfo {
            entry: 0,
            name: NameId(0),
            param_count: 1,
            local_slot_count: 2,
            local_defaults: vec![Expr::Int(10)],
            is_sync: true,
            debug_slot_names: vec!["a".into(), "b".into()],
        };
        let args = vec![Value::<WithHashing>::int(5)];
        let env = Env::<WithHashing>::with_slots(0);
        let roles = HashMap::new();
        let local = make_local_env(&func, args, &env, &env, &roles);
        assert_eq!(local.get(0), &Value::<WithHashing>::int(5));
        assert_eq!(local.get(1), &Value::<WithHashing>::int(10));
    }
    #[test]
    fn test_role_to_string() {
        let mut local_env = Env::<WithHashing>::with_slots(1);
        let node_env = Env::<WithHashing>::with_slots(0);
        let mut roles = HashMap::new();
        roles.insert(NameId(0), "Server".to_string());
        roles.insert(NameId(1), "Client".to_string());

        // Test known role
        let server_node = Value::<WithHashing>::node(NodeId {
            role: NameId(0),
            index: 1,
        });
        local_env.set(0, server_node);

        let expr = Expr::NodeToString(Box::new(Expr::Var(crate::compiler::cfg::VarSlot::Local(
            0,
            NameId(0),
        ))));

        let result = eval(&local_env, &node_env, &expr, &roles).unwrap();
        assert_eq!(result, Value::<WithHashing>::string("Server[1]".into()));

        // Test unknown role
        let unknown_node = Value::<WithHashing>::node(NodeId {
            role: NameId(99),
            index: 0,
        });
        local_env.set(0, unknown_node);
        let result = eval(&local_env, &node_env, &expr, &roles).unwrap();
        assert_eq!(result, Value::<WithHashing>::string("Unknown[0]".into()));
    }
}
