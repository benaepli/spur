//! Evaluator for decoded expressions. Every arm evaluates like the matching
//! arm of `eval`: the same operand order, the same errors, and the same
//! borrowed-operand events, so a decoded program and its graph produce the
//! same values and the same counters.

use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{CExpr, Opnd};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::eval::{Operand, update_collection};
use crate::simulator::core::values::{
    Decimal, Env, Value, ValueKind, ValueMap, ValueSeq, struct_get, struct_to_map,
};
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::util_stats::{self, InterpreterTally};
use ecow::EcoString;
use std::collections::HashMap;
use std::sync::Arc;

/// Reads a slot in place, recording the same event `eval_operand` records
/// for a variable.
#[inline(always)]
pub(crate) fn borrowed<'a, H: HashPolicy>(v: &'a Value<H>, t: &mut InterpreterTally) -> Operand<'a, H> {
    util_stats::record_operand_borrowed(
        v.kind.is_composite() || matches!(v.kind, ValueKind::String(_)),
    );
    t.leaf_operands_inline += 1;
    Operand::Borrowed(v)
}

/// Evaluates a position whose value is only read, as `eval_operand` does.
#[inline(always)]
pub fn coperand<'a, H: HashPolicy>(
    local_env: &'a Env<H>,
    node_env: &'a Env<H>,
    o: &Opnd,
    role_names: &HashMap<NameId, String>,
    t: &mut InterpreterTally,
) -> Result<Operand<'a, H>, RuntimeError> {
    match o {
        Opnd::Local(idx) => Ok(borrowed(local_env.get(*idx), t)),
        Opnd::Node(idx) => Ok(borrowed(node_env.get(*idx), t)),
        Opnd::Int(i) => {
            t.leaf_operands_inline += 1;
            Ok(Operand::Owned(Value::<H>::int(*i)))
        }
        Opnd::Str(s) => {
            t.leaf_operands_inline += 1;
            Ok(Operand::Owned(Value::<H>::string(s.clone())))
        }
        Opnd::Bool(b) => {
            t.leaf_operands_inline += 1;
            Ok(Operand::Owned(Value::<H>::bool(*b)))
        }
        Opnd::Unit => {
            t.leaf_operands_inline += 1;
            Ok(Operand::Owned(Value::<H>::unit()))
        }
        Opnd::Nil => {
            t.leaf_operands_inline += 1;
            Ok(Operand::Owned(Value::<H>::option_none()))
        }
        Opnd::Tree(e) => ceval(local_env, node_env, e, role_names, t).map(Operand::Owned),
    }
}

/// Evaluates a position whose value is kept, as `eval` does.
#[inline(always)]
pub fn cvalue<H: HashPolicy>(
    local_env: &Env<H>,
    node_env: &Env<H>,
    o: &Opnd,
    role_names: &HashMap<NameId, String>,
    t: &mut InterpreterTally,
) -> Result<Value<H>, RuntimeError> {
    match o {
        Opnd::Local(idx) => {
            t.leaf_operands_inline += 1;
            Ok(local_env.get(*idx).clone())
        }
        Opnd::Node(idx) => {
            t.leaf_operands_inline += 1;
            Ok(node_env.get(*idx).clone())
        }
        Opnd::Int(i) => {
            t.leaf_operands_inline += 1;
            Ok(Value::<H>::int(*i))
        }
        Opnd::Str(s) => {
            t.leaf_operands_inline += 1;
            Ok(Value::<H>::string(s.clone()))
        }
        Opnd::Bool(b) => {
            t.leaf_operands_inline += 1;
            Ok(Value::<H>::bool(*b))
        }
        Opnd::Unit => {
            t.leaf_operands_inline += 1;
            Ok(Value::<H>::unit())
        }
        Opnd::Nil => {
            t.leaf_operands_inline += 1;
            Ok(Value::<H>::option_none())
        }
        Opnd::Tree(e) => ceval(local_env, node_env, e, role_names, t),
    }
}

/// Evaluates a decoded expression tree.
pub fn ceval<H: HashPolicy>(
    local_env: &Env<H>,
    node_env: &Env<H>,
    expr: &CExpr,
    role_names: &HashMap<NameId, String>,
    t: &mut InterpreterTally,
) -> Result<Value<H>, RuntimeError> {
    t.tree_evals += 1;
    let l = local_env;
    let n = node_env;
    let r = role_names;
    match expr {
        CExpr::Plus(e1, e2) => {
            let v1 = coperand(l, n, e1, r, t)?;
            let v2 = coperand(l, n, e2, r, t)?;
            match (&v1.kind, &v2.kind) {
                (ValueKind::Int(i1), ValueKind::Int(i2)) => Ok(Value::<H>::int(i1 + i2)),
                (ValueKind::String(s1), ValueKind::String(s2)) => {
                    let mut result = EcoString::with_capacity(s1.len() + s2.len());
                    result.push_str(s1.as_str());
                    result.push_str(s2.as_str());
                    Ok(Value::<H>::string(result))
                }
                _ => Err(RuntimeError::TypeError {
                    expected: "int or string",
                    got: v1.type_name(),
                }),
            }
        }
        CExpr::Minus(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?.as_int()?;
            let b = coperand(l, n, e2, r, t)?.as_int()?;
            Ok(Value::<H>::int(a - b))
        }
        CExpr::Times(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?.as_int()?;
            let b = coperand(l, n, e2, r, t)?.as_int()?;
            Ok(Value::<H>::int(a * b))
        }
        CExpr::Div(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?.as_int()?;
            let b = coperand(l, n, e2, r, t)?.as_int()?;
            Ok(Value::<H>::int(a / b))
        }
        CExpr::Mod(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?.as_int()?;
            let b = coperand(l, n, e2, r, t)?.as_int()?;
            Ok(Value::<H>::int(a % b))
        }
        CExpr::LessThan(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?;
            let b = coperand(l, n, e2, r, t)?;
            Ok(Value::<H>::bool(*a < *b))
        }
        CExpr::LessThanEquals(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?;
            let b = coperand(l, n, e2, r, t)?;
            Ok(Value::<H>::bool(*a <= *b))
        }
        CExpr::GreaterThan(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?;
            let b = coperand(l, n, e2, r, t)?;
            Ok(Value::<H>::bool(*a > *b))
        }
        CExpr::GreaterThanEquals(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?;
            let b = coperand(l, n, e2, r, t)?;
            Ok(Value::<H>::bool(*a >= *b))
        }
        CExpr::EqualsEquals(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?;
            let b = coperand(l, n, e2, r, t)?;
            Ok(Value::<H>::bool(*a == *b))
        }
        CExpr::NotEquals(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?;
            let b = coperand(l, n, e2, r, t)?;
            Ok(Value::<H>::bool(!(*a == *b)))
        }
        CExpr::Not(e) => Ok(Value::<H>::bool(!coperand(l, n, e, r, t)?.as_bool()?)),
        CExpr::And(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?.as_bool()?;
            Ok(Value::<H>::bool(a && coperand(l, n, e2, r, t)?.as_bool()?))
        }
        CExpr::Or(e1, e2) => {
            let a = coperand(l, n, e1, r, t)?.as_bool()?;
            Ok(Value::<H>::bool(a || coperand(l, n, e2, r, t)?.as_bool()?))
        }
        CExpr::Some(e) => Ok(Value::<H>::option_some(cvalue(l, n, e, r, t)?)),
        CExpr::Tuple(es) => {
            let vals: Result<ValueSeq<H>, _> = es.iter().map(|e| cvalue(l, n, e, r, t)).collect();
            Ok(Value::<H>::tuple(vals?))
        }
        CExpr::List(es) => {
            let vals: Result<ValueSeq<H>, _> = es.iter().map(|e| cvalue(l, n, e, r, t)).collect();
            Ok(Value::<H>::list(vals?))
        }
        CExpr::Map(kv) => {
            let mut m = ValueMap::<H>::new();
            for (k, v) in kv {
                m.insert(cvalue(l, n, k, r, t)?, cvalue(l, n, v, r, t)?);
            }
            Ok(Value::<H>::map(m))
        }
        CExpr::StructLit(shape, fields) => {
            let mut vals = ValueSeq::<H>::from_elem(Value::<H>::unit(), fields.len());
            let slots = vals.make_mut();
            for (done, (pos, v)) in fields.iter().enumerate() {
                t.leaf_operands_inline += 1;
                match cvalue(l, n, v, r, t) {
                    Ok(val) => slots[*pos] = val,
                    Err(e) => {
                        if !H::EAGER {
                            util_stats::record_struct_literal_failed(done as u64);
                        }
                        return Err(e);
                    }
                }
            }
            util_stats::record_struct_literal(if H::EAGER { 0 } else { fields.len() as u64 });
            Ok(Value::<H>::struct_of(shape, vals))
        }
        CExpr::Find(col, key) => {
            let col_val = coperand(l, n, col, r, t)?;
            match &col_val.kind {
                ValueKind::Map(m) => {
                    let k = coperand(l, n, key, r, t)?;
                    m.get(&*k).cloned().ok_or(RuntimeError::KeyNotFound)
                }
                ValueKind::Struct(shape, fields) => {
                    let k = coperand(l, n, key, r, t)?;
                    struct_get(shape, fields, &k, true)
                        .cloned()
                        .ok_or(RuntimeError::KeyNotFound)
                }
                ValueKind::List(list) => {
                    let idx = coperand(l, n, key, r, t)?.as_int()? as usize;
                    list.get(idx).cloned().ok_or(RuntimeError::IndexOutOfBounds {
                        index: idx,
                        len: list.len(),
                    })
                }
                _ => Err(RuntimeError::NotACollection {
                    got: col_val.type_name(),
                }),
            }
        }
        CExpr::FieldGet(col, name) => {
            let col_val = coperand(l, n, col, r, t)?;
            match &col_val.kind {
                ValueKind::Map(m) => {
                    t.leaf_operands_inline += 1;
                    let k = Value::<H>::string(name.clone());
                    m.get(&k).cloned().ok_or(RuntimeError::KeyNotFound)
                }
                ValueKind::Struct(shape, fields) => {
                    t.leaf_operands_inline += 1;
                    if !H::EAGER {
                        util_stats::record_struct_field_read();
                    }
                    shape
                        .position(name)
                        .map(|i| fields[i].clone())
                        .ok_or(RuntimeError::KeyNotFound)
                }
                ValueKind::List(list) => {
                    t.leaf_operands_inline += 1;
                    let idx = Value::<H>::string(name.clone()).as_int()? as usize;
                    list.get(idx).cloned().ok_or(RuntimeError::IndexOutOfBounds {
                        index: idx,
                        len: list.len(),
                    })
                }
                _ => Err(RuntimeError::NotACollection {
                    got: col_val.type_name(),
                }),
            }
        }
        CExpr::ListPrepend(head, tail) => {
            let h = cvalue(l, n, head, r, t)?;
            let tail_val = coperand(l, n, tail, r, t)?;
            let tl = tail_val.as_list()?;
            let mut new_list = ValueSeq::<H>::with_capacity(tl.len() + 1);
            new_list.push(h);
            new_list.extend_from_slice(tl);
            Ok(Value::<H>::list(new_list))
        }
        CExpr::ListAppend(list, item) => {
            let list_val = coperand(l, n, list, r, t)?;
            let mut out = list_val.as_list()?.clone();
            let i = cvalue(l, n, item, r, t)?;
            out.push(i);
            Ok(Value::<H>::list(out))
        }
        CExpr::ListSubsequence(list, start, end) => {
            let lv = coperand(l, n, list, r, t)?;
            let s = coperand(l, n, start, r, t)?.as_int()? as usize;
            let e = coperand(l, n, end, r, t)?.as_int()? as usize;
            let vec = lv.as_list()?;
            if s > vec.len() || e > vec.len() || s > e {
                return Err(RuntimeError::SubsequenceOutOfBounds {
                    start: s,
                    end: e,
                    len: vec.len(),
                });
            }
            Ok(Value::<H>::list(ValueSeq::<H>::from(&vec[s..e])))
        }
        CExpr::KeyExists(key, map) => {
            let k = coperand(l, n, key, r, t)?;
            let m = coperand(l, n, map, r, t)?;
            if let ValueKind::Struct(shape, fields) = &m.kind {
                return Ok(Value::<H>::bool(struct_get(shape, fields, &k, false).is_some()));
            }
            Ok(Value::<H>::bool(m.as_map()?.contains_key(&*k)))
        }
        CExpr::MapErase(key, map) => {
            let k = coperand(l, n, key, r, t)?;
            let m = coperand(l, n, map, r, t)?;
            if let ValueKind::Struct(shape, fields) = &m.kind {
                return Ok(Value::<H>::map(struct_to_map(shape, fields).without(&*k)));
            }
            Ok(Value::<H>::map(m.as_map()?.without(&*k)))
        }
        CExpr::ListLen(list) => {
            let list_val = coperand(l, n, list, r, t)?;
            match &list_val.kind {
                ValueKind::List(v) => Ok(Value::<H>::int(v.len() as i64)),
                ValueKind::Map(m) => Ok(Value::<H>::int(m.len() as i64)),
                ValueKind::Struct(shape, _) => Ok(Value::<H>::int(shape.len() as i64)),
                _ => Err(RuntimeError::NotACollection {
                    got: list_val.type_name(),
                }),
            }
        }
        CExpr::ListAccess(list, idx) => {
            let lv = coperand(l, n, list, r, t)?;
            let vec = lv.as_list()?;
            let i = *idx;
            if i >= vec.len() {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: i,
                    len: vec.len(),
                });
            }
            Ok(vec[i].clone())
        }
        CExpr::Min(e1, e2) => {
            let v1 = coperand(l, n, e1, r, t)?.as_int()?;
            let v2 = coperand(l, n, e2, r, t)?.as_int()?;
            Ok(Value::<H>::int(v1.min(v2)))
        }
        CExpr::TupleAccess(tuple, idx) => {
            let tv = coperand(l, n, tuple, r, t)?;
            if let ValueKind::Tuple(vec) = &tv.kind {
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
                    got: tv.type_name(),
                })
            }
        }
        CExpr::Unwrap(e) => {
            let val = coperand(l, n, e, r, t)?;
            match &val.kind {
                ValueKind::Option(Some(v)) => Ok((**v).clone()),
                ValueKind::Option(None) => Err(RuntimeError::UnwrapNone),
                _ => Err(RuntimeError::TypeError {
                    expected: "option",
                    got: val.type_name(),
                }),
            }
        }
        CExpr::Coalesce(opt, default) => {
            let val = coperand(l, n, opt, r, t)?;
            match &val.kind {
                ValueKind::Option(Some(v)) => Ok((**v).clone()),
                ValueKind::Option(None) => cvalue(l, n, default, r, t),
                _ => Err(RuntimeError::CoalesceNonOption {
                    got: val.type_name(),
                }),
            }
        }
        CExpr::IntToString(e) => {
            let i = coperand(l, n, e, r, t)?.as_int()?;
            Ok(Value::<H>::string(EcoString::from(Decimal::of_i64(i).as_str())))
        }
        CExpr::BoolToString(e) => Ok(Value::<H>::string(EcoString::from(
            coperand(l, n, e, r, t)?.as_bool()?.to_string(),
        ))),
        CExpr::NodeToString(e) => {
            let node_id = coperand(l, n, e, r, t)?.as_node()?;
            let role_name = r
                .get(&node_id.role)
                .map(|s| s.as_str())
                .unwrap_or("Unknown");
            Ok(Value::<H>::string(EcoString::from(format!(
                "{}[{}]",
                role_name, node_id.index
            ))))
        }
        CExpr::Store(col, key, val) => {
            let c = cvalue(l, n, col, r, t)?;
            let k = coperand(l, n, key, r, t)?;
            let v = cvalue(l, n, val, r, t)?;
            update_collection(c, k, v)
        }
        CExpr::Variant(enum_id, name, payload) => {
            let payload_val = payload
                .as_ref()
                .map(|p| cvalue(l, n, p, r, t))
                .transpose()?
                .map(Arc::new);
            Ok(Value::<H>::variant(*enum_id, name.clone(), payload_val))
        }
        CExpr::IsVariant(e, name) => {
            let val = coperand(l, n, e, r, t)?;
            match &val.kind {
                ValueKind::Variant(_, variant_name, _) => Ok(Value::<H>::bool(variant_name == name)),
                _ => Ok(Value::<H>::bool(false)),
            }
        }
        CExpr::VariantPayload(e) => {
            let val = coperand(l, n, e, r, t)?;
            match &val.kind {
                ValueKind::Variant(_, _, Some(payload)) => Ok((**payload).clone()),
                ValueKind::Variant(_, _, None) => Err(RuntimeError::VariantHasNoPayload),
                _ => Err(RuntimeError::TypeError {
                    expected: "variant",
                    got: val.type_name(),
                }),
            }
        }
        CExpr::SafeFind(col, key) => {
            let col_val = coperand(l, n, col, r, t)?;
            match &col_val.kind {
                ValueKind::Option(None) => Ok(Value::<H>::option_none()),
                ValueKind::Option(Some(inner)) => {
                    let inner_val: &Value<H> = inner;
                    let key_val = coperand(l, n, key, r, t)?;
                    match &inner_val.kind {
                        ValueKind::Map(m) => {
                            let result =
                                m.get(&*key_val).cloned().ok_or(RuntimeError::KeyNotFound)?;
                            Ok(Value::<H>::option_some(result))
                        }
                        ValueKind::Struct(shape, fields) => {
                            let result = struct_get(shape, fields, &key_val, false)
                                .cloned()
                                .ok_or(RuntimeError::KeyNotFound)?;
                            Ok(Value::<H>::option_some(result))
                        }
                        ValueKind::List(list) => {
                            let idx = key_val.as_int()? as usize;
                            let result =
                                list.get(idx).cloned().ok_or(RuntimeError::IndexOutOfBounds {
                                    index: idx,
                                    len: list.len(),
                                })?;
                            Ok(Value::<H>::option_some(result))
                        }
                        _ => Err(RuntimeError::NotACollection {
                            got: inner_val.type_name(),
                        }),
                    }
                }
                _ => Err(RuntimeError::TypeError {
                    expected: "option",
                    got: col_val.type_name(),
                }),
            }
        }
        CExpr::SafeTupleAccess(tuple, idx) => {
            let tv = coperand(l, n, tuple, r, t)?;
            match &tv.kind {
                ValueKind::Option(None) => Ok(Value::<H>::option_none()),
                ValueKind::Option(Some(inner)) => {
                    let inner_val: &Value<H> = inner;
                    if let ValueKind::Tuple(vec) = &inner_val.kind {
                        if *idx >= vec.len() {
                            return Err(RuntimeError::IndexOutOfBounds {
                                index: *idx,
                                len: vec.len(),
                            });
                        }
                        Ok(Value::<H>::option_some(vec[*idx].clone()))
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "tuple",
                            got: inner_val.type_name(),
                        })
                    }
                }
                _ => Err(RuntimeError::TypeError {
                    expected: "option",
                    got: tv.type_name(),
                }),
            }
        }
    }
}

#[cfg(test)]
mod test;
