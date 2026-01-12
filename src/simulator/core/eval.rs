use crate::compiler::cfg::{Expr, FunctionInfo, Lhs, VarSlot};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::values::{Env, Value, ValueKind, hash_map_entry};
use ecow::EcoString;
use imbl::{HashMap as ImHashMap, Vector};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

#[inline(always)]
pub fn load(slot: VarSlot, local_env: &Env, node_env: &Env) -> Value {
    match slot {
        VarSlot::Local(idx, _) => local_env.get(idx).clone(),
        VarSlot::Node(idx, _) => node_env.get(idx).clone(),
    }
}

#[inline(always)]
pub fn store_slot(slot: VarSlot, val: Value, local_env: &mut Env, node_env: &mut Env) {
    match slot {
        VarSlot::Local(idx, _) => local_env.set(idx, val),
        VarSlot::Node(idx, _) => node_env.set(idx, val),
    }
}

pub fn store(
    lhs: &Lhs,
    val: Value,
    local_env: &mut Env,
    node_env: &mut Env,
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
pub fn make_local_env(
    func: &FunctionInfo,
    args: Vec<Value>,
    local_env: &Env,
    node_env: &Env,
) -> Env {
    let mut env = Env::with_slots(func.local_slot_count as usize);

    // Set arguments in parameter slots
    for (i, arg) in args.into_iter().enumerate() {
        env.set(i as u32, arg);
    }

    // Initialize other locals to their defaults
    for (i, default_expr) in func.local_defaults.iter().enumerate() {
        let slot = func.param_count + i as u32;
        if let Ok(val) = eval(local_env, node_env, default_expr) {
            env.set(slot, val);
        }
    }

    env
}

fn update_collection(col: Value, key: Value, val: Value) -> Result<Value, RuntimeError> {
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

            Ok(Value::with_sig(ValueKind::Map(new_map), new_sig))
        }
        List(l) => {
            let idx = key.as_int()? as usize;
            if idx >= l.len() {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: idx,
                    len: l.len(),
                });
            }
            Ok(Value::list(l.update(idx, val)))
        }
        _ => Err(RuntimeError::NotACollection {
            got: col.type_name(),
        }),
    }
}

pub fn eval(local_env: &Env, node_env: &Env, expr: &Expr) -> Result<Value, RuntimeError> {
    match expr {
        Expr::Int(i) => Ok(Value::int(*i)),
        Expr::Bool(b) => Ok(Value::bool(*b)),
        Expr::String(s) => Ok(Value::string(s.clone())),
        Expr::Unit => Ok(Value::unit()),
        Expr::Nil => Ok(Value::option_none()),
        Expr::Var(s) => Ok(load(*s, local_env, node_env)),
        Expr::Plus(e1, e2) => {
            let v1 = eval(local_env, node_env, e1)?;
            let v2 = eval(local_env, node_env, e2)?;

            match (&v1.kind, &v2.kind) {
                (ValueKind::Int(i1), ValueKind::Int(i2)) => Ok(Value::int(i1 + i2)),
                (ValueKind::String(s1), ValueKind::String(s2)) => {
                    let mut result = String::new();
                    result.push_str(s1.as_str());
                    result.push_str(s2.as_str());
                    Ok(Value::string(EcoString::from(result)))
                }
                _ => Err(RuntimeError::TypeError {
                    expected: "int or string",
                    got: v1.type_name(),
                }),
            }
        }
        Expr::Minus(e1, e2) => Ok(Value::int(
            eval(local_env, node_env, e1)?.as_int()? - eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::Times(e1, e2) => Ok(Value::int(
            eval(local_env, node_env, e1)?.as_int()? * eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::Div(e1, e2) => Ok(Value::int(
            eval(local_env, node_env, e1)?.as_int()? / eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::Mod(e1, e2) => Ok(Value::int(
            eval(local_env, node_env, e1)?.as_int()? % eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::LessThan(e1, e2) => Ok(Value::bool(
            eval(local_env, node_env, e1)? < eval(local_env, node_env, e2)?,
        )),
        Expr::EqualsEquals(e1, e2) => Ok(Value::bool(
            eval(local_env, node_env, e1)? == eval(local_env, node_env, e2)?,
        )),
        Expr::Not(e) => Ok(Value::bool(!eval(local_env, node_env, e)?.as_bool()?)),
        Expr::And(e1, e2) => Ok(Value::bool(
            eval(local_env, node_env, e1)?.as_bool()?
                && eval(local_env, node_env, e2)?.as_bool()?,
        )),
        Expr::Or(e1, e2) => Ok(Value::bool(
            eval(local_env, node_env, e1)?.as_bool()?
                || eval(local_env, node_env, e2)?.as_bool()?,
        )),
        Expr::Some(e) => Ok(Value::option_some(eval(local_env, node_env, e)?)),
        Expr::Tuple(es) => {
            let vals: Result<Vector<_>, _> =
                es.iter().map(|e| eval(local_env, node_env, e)).collect();
            Ok(Value::tuple(vals?))
        }
        Expr::List(es) => {
            let vals: Result<Vector<_>, _> =
                es.iter().map(|e| eval(local_env, node_env, e)).collect();
            Ok(Value::list(vals?))
        }
        Expr::Map(kv) => {
            let mut m = ImHashMap::new();
            for (k, v) in kv {
                m.insert(eval(local_env, node_env, k)?, eval(local_env, node_env, v)?);
            }
            Ok(Value::map(m))
        }
        Expr::Find(col, key) => {
            let col_val = eval(local_env, node_env, col)?;
            match &col_val.kind {
                ValueKind::Map(m) => {
                    let k = eval(local_env, node_env, key)?;
                    m.get(&k).cloned().ok_or(RuntimeError::KeyNotFound)
                }
                ValueKind::List(l) => {
                    let idx = eval(local_env, node_env, key)?.as_int()? as usize;
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
            let h = eval(local_env, node_env, head)?;
            let t = eval(local_env, node_env, tail)?.as_list()?.clone();
            let mut new_list = Vector::new();
            new_list.push_back(h);
            new_list.append(t);
            Ok(Value::list(new_list))
        }
        Expr::ListAppend(list, item) => {
            let mut l = eval(local_env, node_env, list)?.as_list()?.clone();
            let i = eval(local_env, node_env, item)?;
            l.push_back(i);
            Ok(Value::list(l))
        }
        Expr::ListSubsequence(list, start, end) => {
            let l = eval(local_env, node_env, list)?;
            let s = eval(local_env, node_env, start)?.as_int()? as usize;
            let e = eval(local_env, node_env, end)?.as_int()? as usize;
            let vec = l.as_list()?;
            if s > vec.len() || e > vec.len() || s > e {
                return Err(RuntimeError::SubsequenceOutOfBounds {
                    start: s,
                    end: e,
                    len: vec.len(),
                });
            }
            Ok(Value::list(vec.clone().slice(s..e)))
        }
        Expr::LessThanEquals(e1, e2) => Ok(Value::bool(
            eval(local_env, node_env, e1)? <= eval(local_env, node_env, e2)?,
        )),
        Expr::GreaterThan(e1, e2) => Ok(Value::bool(
            eval(local_env, node_env, e1)? > eval(local_env, node_env, e2)?,
        )),
        Expr::GreaterThanEquals(e1, e2) => Ok(Value::bool(
            eval(local_env, node_env, e1)? >= eval(local_env, node_env, e2)?,
        )),
        Expr::KeyExists(key, map) => {
            let k = eval(local_env, node_env, key)?;
            let m = eval(local_env, node_env, map)?;
            Ok(Value::bool(m.as_map()?.contains_key(&k)))
        }
        Expr::MapErase(key, map) => {
            let k = eval(local_env, node_env, key)?;
            let m = eval(local_env, node_env, map)?.as_map()?.clone();
            Ok(Value::map(m.without(&k)))
        }
        Expr::ListLen(list) => {
            let list_val = eval(local_env, node_env, list)?;
            match &list_val.kind {
                ValueKind::List(l) => Ok(Value::int(l.len() as i64)),
                ValueKind::Map(m) => Ok(Value::int(m.len() as i64)),
                _ => Err(RuntimeError::NotACollection {
                    got: list_val.type_name(),
                }),
            }
        }
        Expr::ListAccess(list, idx) => {
            let l = eval(local_env, node_env, list)?;
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
            let v1 = eval(local_env, node_env, e1)?.as_int()?;
            let v2 = eval(local_env, node_env, e2)?.as_int()?;
            Ok(Value::int(v1.min(v2)))
        }
        Expr::TupleAccess(tuple, idx) => {
            let t = eval(local_env, node_env, tuple)?;
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
            let val = eval(local_env, node_env, e)?;
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
            let val = eval(local_env, node_env, opt)?;
            match &val.kind {
                ValueKind::Option(Some(v)) => Ok(Arc::unwrap_or_clone(v.clone())),
                ValueKind::Option(None) => eval(local_env, node_env, default),
                _ => Err(RuntimeError::CoalesceNonOption {
                    got: val.type_name(),
                }),
            }
        }
        Expr::IntToString(e) => Ok(Value::string(EcoString::from(
            eval(local_env, node_env, e)?.as_int()?.to_string(),
        ))),
        Expr::BoolToString(e) => Ok(Value::string(EcoString::from(
            eval(local_env, node_env, e)?.as_bool()?.to_string(),
        ))),
        Expr::Store(col, key, val) => update_collection(
            eval(local_env, node_env, col)?,
            eval(local_env, node_env, key)?,
            eval(local_env, node_env, val)?,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::resolver::NameId;

    fn dummy_slot(idx: u32) -> VarSlot {
        VarSlot::Local(idx, NameId(0))
    }

    fn node_slot(idx: u32) -> VarSlot {
        VarSlot::Node(idx, NameId(0))
    }

    #[test]
    fn test_eval_literals() {
        let env = Env::with_slots(0);
        assert_eq!(eval(&env, &env, &Expr::Int(42)).unwrap(), Value::int(42));
        assert_eq!(
            eval(&env, &env, &Expr::Bool(true)).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            eval(&env, &env, &Expr::String("hello".into())).unwrap(),
            Value::string("hello".into())
        );
        assert_eq!(eval(&env, &env, &Expr::Unit).unwrap(), Value::unit());
        assert_eq!(eval(&env, &env, &Expr::Nil).unwrap(), Value::option_none());
    }

    #[test]
    fn test_eval_arithmetic() {
        let env = Env::with_slots(0);
        let e1 = Box::new(Expr::Int(10));
        let e2 = Box::new(Expr::Int(3));

        assert_eq!(
            eval(&env, &env, &Expr::Plus(e1.clone(), e2.clone())).unwrap(),
            Value::int(13)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Minus(e1.clone(), e2.clone())).unwrap(),
            Value::int(7)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Times(e1.clone(), e2.clone())).unwrap(),
            Value::int(30)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Div(e1.clone(), e2.clone())).unwrap(),
            Value::int(3)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Mod(e1.clone(), e2.clone())).unwrap(),
            Value::int(1)
        );
    }

    #[test]
    fn test_eval_string_concat() {
        let env = Env::with_slots(0);

        // Basic concatenation
        let s1 = Box::new(Expr::String(EcoString::from("hello")));
        let s2 = Box::new(Expr::String(EcoString::from("world")));
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s1, s2)).unwrap(),
            Value::string(EcoString::from("helloworld"))
        );

        // Concatenation with spaces
        let s3 = Box::new(Expr::String(EcoString::from("hello ")));
        let s4 = Box::new(Expr::String(EcoString::from("world")));
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s3, s4)).unwrap(),
            Value::string(EcoString::from("hello world"))
        );

        // Empty strings
        let s5 = Box::new(Expr::String(EcoString::from("")));
        let s6 = Box::new(Expr::String(EcoString::from("test")));
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s5.clone(), s6.clone())).unwrap(),
            Value::string(EcoString::from("test"))
        );
        assert_eq!(
            eval(&env, &env, &Expr::Plus(s6, s5)).unwrap(),
            Value::string(EcoString::from("test"))
        );
    }

    #[test]
    fn test_eval_logical() {
        let env = Env::with_slots(0);
        let t = Box::new(Expr::Bool(true));
        let f = Box::new(Expr::Bool(false));

        assert_eq!(
            eval(&env, &env, &Expr::Not(t.clone())).unwrap(),
            Value::bool(false)
        );
        assert_eq!(
            eval(&env, &env, &Expr::And(t.clone(), f.clone())).unwrap(),
            Value::bool(false)
        );
        assert_eq!(
            eval(&env, &env, &Expr::Or(t.clone(), f.clone())).unwrap(),
            Value::bool(true)
        );
    }

    #[test]
    fn test_eval_comparison() {
        let env = Env::with_slots(0);
        let e1 = Box::new(Expr::Int(10));
        let e2 = Box::new(Expr::Int(20));

        assert_eq!(
            eval(&env, &env, &Expr::EqualsEquals(e1.clone(), e1.clone())).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            eval(&env, &env, &Expr::LessThan(e1.clone(), e2.clone())).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            eval(&env, &env, &Expr::LessThanEquals(e1.clone(), e1.clone())).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            eval(&env, &env, &Expr::GreaterThan(e2.clone(), e1.clone())).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            eval(&env, &env, &Expr::GreaterThanEquals(e1.clone(), e1.clone())).unwrap(),
            Value::bool(true)
        );
    }

    #[test]
    fn test_eval_vars() {
        let mut local_env = Env::with_slots(2);
        local_env.set(0, Value::int(100));
        let mut node_env = Env::with_slots(1);
        node_env.set(0, Value::int(200));

        assert_eq!(
            eval(&local_env, &node_env, &Expr::Var(dummy_slot(0))).unwrap(),
            Value::int(100)
        );
        assert_eq!(
            eval(&local_env, &node_env, &Expr::Var(node_slot(0))).unwrap(),
            Value::int(200)
        );
    }

    #[test]
    fn test_eval_collections() {
        let env = Env::with_slots(0);

        // Tuple
        let tuple_expr = Expr::Tuple(vec![Expr::Int(1), Expr::Bool(true)]);
        let _tuple_val = eval(&env, &env, &tuple_expr).unwrap();
        assert_eq!(
            eval(&env, &env, &Expr::TupleAccess(Box::new(tuple_expr), 1)).unwrap(),
            Value::bool(true)
        );

        // List
        let list_expr = Expr::List(vec![Expr::Int(1), Expr::Int(2)]);
        assert_eq!(
            eval(&env, &env, &Expr::ListLen(Box::new(list_expr.clone()))).unwrap(),
            Value::int(2)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::ListAccess(Box::new(list_expr.clone()), 0)
            )
            .unwrap(),
            Value::int(1)
        );

        // List operations
        let append_expr = Expr::ListAppend(Box::new(list_expr.clone()), Box::new(Expr::Int(3)));
        let appended_val = eval(&env, &env, &append_expr).unwrap();
        assert_eq!(appended_val.as_list().unwrap().len(), 3);

        let prepend_expr = Expr::ListPrepend(Box::new(Expr::Int(0)), Box::new(list_expr.clone()));
        let prepended_val = eval(&env, &env, &prepend_expr).unwrap();
        assert_eq!(prepended_val.as_list().unwrap()[0], Value::int(0));

        let sub_expr = Expr::ListSubsequence(
            Box::new(list_expr),
            Box::new(Expr::Int(0)),
            Box::new(Expr::Int(1)),
        );
        let sub_val = eval(&env, &env, &sub_expr).unwrap();
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
                )
            )
            .unwrap(),
            Value::int(42)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::KeyExists(
                    Box::new(Expr::String("key".into())),
                    Box::new(map_expr.clone())
                )
            )
            .unwrap(),
            Value::bool(true)
        );

        let erase_expr = Expr::MapErase(Box::new(Expr::String("key".into())), Box::new(map_expr));
        let erased_val = eval(&env, &env, &erase_expr).unwrap();
        assert_eq!(erased_val.as_map().unwrap().len(), 0);
    }

    #[test]
    fn test_eval_options() {
        let env = Env::with_slots(0);
        let some_expr = Expr::Some(Box::new(Expr::Int(42)));
        assert_eq!(
            eval(&env, &env, &Expr::Unwrap(Box::new(some_expr.clone()))).unwrap(),
            Value::int(42)
        );

        let nil_expr = Expr::Nil;
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Coalesce(Box::new(nil_expr), Box::new(Expr::Int(100)))
            )
            .unwrap(),
            Value::int(100)
        );
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Coalesce(Box::new(some_expr), Box::new(Expr::Int(100)))
            )
            .unwrap(),
            Value::int(42)
        );
    }

    #[test]
    fn test_eval_misc() {
        let env = Env::with_slots(0);
        assert_eq!(
            eval(
                &env,
                &env,
                &Expr::Min(Box::new(Expr::Int(10)), Box::new(Expr::Int(20)))
            )
            .unwrap(),
            Value::int(10)
        );
        assert_eq!(
            eval(&env, &env, &Expr::IntToString(Box::new(Expr::Int(123)))).unwrap(),
            Value::string("123".into())
        );
        assert_eq!(
            eval(&env, &env, &Expr::BoolToString(Box::new(Expr::Bool(true)))).unwrap(),
            Value::string("true".into())
        );
        assert_eq!(
            eval(&env, &env, &Expr::BoolToString(Box::new(Expr::Bool(false)))).unwrap(),
            Value::string("false".into())
        );
    }

    #[test]
    fn test_eval_store_update() {
        let env = Env::with_slots(0);
        let list_expr = Expr::List(vec![Expr::Int(1)]);
        let store_expr = Expr::Store(
            Box::new(list_expr),
            Box::new(Expr::Int(0)),
            Box::new(Expr::Int(42)),
        );
        let updated_list = eval(&env, &env, &store_expr).unwrap();
        assert_eq!(updated_list.as_list().unwrap()[0], Value::int(42));
    }

    #[test]
    fn test_eval_errors() {
        let env = Env::with_slots(0);
        let res = eval(
            &env,
            &env,
            &Expr::Plus(Box::new(Expr::Bool(true)), Box::new(Expr::Int(1))),
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_store() {
        let mut local_env = Env::with_slots(1);
        let mut node_env = Env::with_slots(1);

        // Simple var store
        store(
            &Lhs::Var(dummy_slot(0)),
            Value::int(42),
            &mut local_env,
            &mut node_env,
        )
        .unwrap();
        assert_eq!(local_env.get(0), &Value::int(42));

        // Tuple store
        let lhs = Lhs::Tuple(vec![dummy_slot(0), node_slot(0)]);
        let val = Value::tuple(Vector::from(vec![Value::int(1), Value::int(2)]));
        store(&lhs, val, &mut local_env, &mut node_env).unwrap();
        assert_eq!(local_env.get(0), &Value::int(1));
        assert_eq!(node_env.get(0), &Value::int(2));

        // Mismatched tuple length
        let val_bad = Value::tuple(Vector::from(vec![Value::int(1)]));
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
        let args = vec![Value::int(5)];
        let env = Env::with_slots(0);
        let local = make_local_env(&func, args, &env, &env);
        assert_eq!(local.get(0), &Value::int(5));
        assert_eq!(local.get(1), &Value::int(10));
    }
}
