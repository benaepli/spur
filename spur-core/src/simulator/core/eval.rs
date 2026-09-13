use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Expr, FunctionInfo, Lhs, SlotDefault, VarSlot};
use crate::simulator::core::error::RuntimeError;
use crate::simulator::core::values::{
    Decimal, Env, Value, ValueKind, ValueMap, ValueSeq, hash_map_entry,
};
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::util_stats;
use ecow::{EcoString, EcoVec};
use std::collections::HashMap;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[inline(always)]
pub fn load<H: HashPolicy>(slot: VarSlot, local_env: &Env<H>, node_env: &Env<H>) -> Value<H> {
    match slot {
        VarSlot::Local(idx, _) => local_env.get(idx).clone(),
        VarSlot::Node(idx, _) => node_env.get(idx).clone(),
    }
}

/// A value the evaluator only reads: a variable's slot in place, or the value
/// any other expression produced.
pub enum Operand<'a, H: HashPolicy> {
    Borrowed(&'a Value<H>),
    Owned(Value<H>),
}

impl<H: HashPolicy> std::ops::Deref for Operand<'_, H> {
    type Target = Value<H>;

    #[inline(always)]
    fn deref(&self) -> &Value<H> {
        match self {
            Operand::Borrowed(v) => v,
            Operand::Owned(v) => v,
        }
    }
}

impl<H: HashPolicy> Operand<'_, H> {
    #[inline]
    pub fn into_owned(self) -> Value<H> {
        match self {
            Operand::Borrowed(v) => v.clone(),
            Operand::Owned(v) => v,
        }
    }
}

/// Evaluates an expression whose value is only read. A variable is returned
/// in place, so its slot value is neither copied nor dropped. The result
/// borrows the environments, so no slot can be written while it is alive.
#[inline]
pub fn eval_operand<'a, H: HashPolicy>(
    local_env: &'a Env<H>,
    node_env: &'a Env<H>,
    expr: &Expr,
    role_names: &HashMap<NameId, String>,
) -> Result<Operand<'a, H>, RuntimeError> {
    match expr {
        Expr::Var(slot) => {
            let v = match *slot {
                VarSlot::Local(idx, _) => local_env.get(idx),
                VarSlot::Node(idx, _) => node_env.get(idx),
            };
            util_stats::record_operand_borrowed(
                v.kind.is_composite() || matches!(v.kind, ValueKind::String(_)),
            );
            Ok(Operand::Borrowed(v))
        }
        _ => eval(local_env, node_env, expr, role_names).map(Operand::Owned),
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
        VarSlot::Local(idx, _) => set_local(local_env, idx, val),
        VarSlot::Node(idx, _) => node_env.set(idx, val),
    }
}

/// Write one slot of a local call frame. A frame that is still shared when
/// it is written is copied by the write, which is counted here.
#[inline(always)]
pub fn set_local<H: HashPolicy>(local_env: &mut Env<H>, slot: u32, val: Value<H>) {
    if !local_env.slots.is_unique() {
        util_stats::record_entry_frame_copy();
    }
    local_env.set(slot, val);
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
    }
}

/// Builds a local call frame slot by slot: the caller pushes the argument
/// values, then `finish` appends the declared starting value of every
/// remaining slot. The buffer is sized once and each slot written once.
pub struct FrameBuilder<H: HashPolicy> {
    slots: EcoVec<Value<H>>,
    sig: u64,
}

impl<H: HashPolicy> FrameBuilder<H> {
    pub fn new(func: &FunctionInfo) -> Self {
        Self {
            slots: EcoVec::with_capacity(func.local_slot_count as usize),
            sig: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, value: Value<H>) {
        if H::EAGER {
            self.sig ^= H::mix(value.sig, self.slots.len() as u32);
        }
        self.slots.push(value);
    }

    /// Fills every slot from the current length up to the slot count: Unit
    /// for parameters the caller did not push, then the declared defaults in
    /// order, then Unit. Each slot value is constructed where it is written.
    pub fn finish(mut self, func: &FunctionInfo) -> Env<H> {
        let total = func.local_slot_count as usize;
        let start = self.slots.len();
        let defaults_start = start.max((func.param_count as usize).min(total));
        let defaults_end = defaults_start
            + func
                .local_defaults
                .len()
                .min(total.saturating_sub(defaults_start));
        if start < total {
            let (unit_sig, nil_sig) = if H::EAGER {
                (Value::<H>::unit().sig, Value::<H>::option_none().sig)
            } else {
                (0, 0)
            };
            let mut sig = self.sig;
            let defaults = &func.local_defaults;
            let fill = (start..total).map(|i| {
                let value = if i < defaults_start || i >= defaults_end {
                    Value::<H>::with_sig(ValueKind::Unit, unit_sig)
                } else {
                    match defaults[i - defaults_start] {
                        SlotDefault::Unit => Value::<H>::with_sig(ValueKind::Unit, unit_sig),
                        SlotDefault::Nil => {
                            Value::<H>::with_sig(ValueKind::Option(None), nil_sig)
                        }
                    }
                };
                if H::EAGER {
                    sig ^= H::mix(value.sig, i as u32);
                }
                value
            });
            // SAFETY: a mapped range reports its exact length.
            unsafe { self.slots.extend_from_trusted(fill) };
            self.sig = sig;
        }
        util_stats::record_frame_build(total as u64, start.min(total) as u64);
        Env::from_slots(self.slots, self.sig)
    }
}

/// The local call frame for `func` with `args` in its parameter slots.
pub fn build_frame<H: HashPolicy>(func: &FunctionInfo, args: &[Value<H>]) -> Env<H> {
    let mut builder = FrameBuilder::<H>::new(func);
    let params = (func.param_count as usize)
        .min(func.local_slot_count as usize)
        .min(args.len());
    let mut sig = builder.sig;
    let copied = args[..params].iter().enumerate().map(|(i, arg)| {
        if H::EAGER {
            sig ^= H::mix(arg.sig, i as u32);
        }
        arg.clone()
    });
    // SAFETY: an enumerated slice iterator reports its exact length.
    unsafe { builder.slots.extend_from_trusted(copied) };
    builder.sig = sig;
    builder.finish(func)
}

fn update_collection<H: HashPolicy>(
    col: Value<H>,
    key: Operand<'_, H>,
    val: Value<H>,
) -> Result<Value<H>, RuntimeError> {
    use ValueKind::*;
    match col.kind {
        Map(m) => {
            let new_sig = if H::EAGER {
                let mut s = col.sig;

                // Remove the old entry's contribution (if it exists)
                if let Some(old_val) = m.get(&key) {
                    let old_entry_hash = hash_map_entry(key.sig, old_val.sig);
                    s ^= old_entry_hash; // XOR removes it
                // length stays the same when replacing, so no change needed
                } else {
                    // Key doesn't exist, length will increase by 1
                    // Remove old length hash, add new length hash
                    let mut h = FxHasher::default();
                    9u8.hash(&mut h);
                    m.len().hash(&mut h);
                    s ^= h.finish();

                    let mut h = FxHasher::default();
                    9u8.hash(&mut h);
                    (m.len() + 1).hash(&mut h);
                    s ^= h.finish();
                }

                // Add the new entry's contribution
                let new_entry_hash = hash_map_entry(key.sig, val.sig);
                s ^= new_entry_hash;
                s
            } else {
                0
            };

            let new_map = m.update(key.into_owned(), val);

            Ok(Value::<H>::with_sig(ValueKind::Map(new_map), new_sig))
        }
        List(mut l) => {
            let idx = key.as_int()? as usize;
            if idx >= l.len() {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: idx,
                    len: l.len(),
                });
            }
            l.make_mut()[idx] = val;
            Ok(Value::<H>::list(l))
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
            let v1 = eval_operand(local_env, node_env, e1, role_names)?;
            let v2 = eval_operand(local_env, node_env, e2, role_names)?;

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
        Expr::Minus(e1, e2) => Ok(Value::<H>::int(
            eval_operand(local_env, node_env, e1, role_names)?.as_int()?
                - eval_operand(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::Times(e1, e2) => Ok(Value::<H>::int(
            eval_operand(local_env, node_env, e1, role_names)?.as_int()?
                * eval_operand(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::Div(e1, e2) => Ok(Value::<H>::int(
            eval_operand(local_env, node_env, e1, role_names)?.as_int()?
                / eval_operand(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::Mod(e1, e2) => Ok(Value::<H>::int(
            eval_operand(local_env, node_env, e1, role_names)?.as_int()?
                % eval_operand(local_env, node_env, e2, role_names)?.as_int()?,
        )),
        Expr::LessThan(e1, e2) => Ok(Value::<H>::bool(
            *eval_operand(local_env, node_env, e1, role_names)?
                < *eval_operand(local_env, node_env, e2, role_names)?,
        )),
        Expr::EqualsEquals(e1, e2) => Ok(Value::<H>::bool(
            *eval_operand(local_env, node_env, e1, role_names)?
                == *eval_operand(local_env, node_env, e2, role_names)?,
        )),
        Expr::Not(e) => Ok(Value::<H>::bool(
            !eval_operand(local_env, node_env, e, role_names)?.as_bool()?,
        )),
        Expr::And(e1, e2) => Ok(Value::<H>::bool(
            eval_operand(local_env, node_env, e1, role_names)?.as_bool()?
                && eval_operand(local_env, node_env, e2, role_names)?.as_bool()?,
        )),
        Expr::Or(e1, e2) => Ok(Value::<H>::bool(
            eval_operand(local_env, node_env, e1, role_names)?.as_bool()?
                || eval_operand(local_env, node_env, e2, role_names)?.as_bool()?,
        )),
        Expr::Some(e) => Ok(Value::<H>::option_some(eval(
            local_env, node_env, e, role_names,
        )?)),
        Expr::Tuple(es) => {
            let vals: Result<ValueSeq<H>, _> = es
                .iter()
                .map(|e| eval(local_env, node_env, e, role_names))
                .collect();
            Ok(Value::<H>::tuple(vals?))
        }
        Expr::List(es) => {
            let vals: Result<ValueSeq<H>, _> = es
                .iter()
                .map(|e| eval(local_env, node_env, e, role_names))
                .collect();
            Ok(Value::<H>::list(vals?))
        }
        Expr::Map(kv) => {
            let mut m = ValueMap::<H>::new();
            for (k, v) in kv {
                m.insert(
                    eval(local_env, node_env, k, role_names)?,
                    eval(local_env, node_env, v, role_names)?,
                );
            }
            Ok(Value::<H>::map(m))
        }
        Expr::Find(col, key) => {
            let col_val = eval_operand(local_env, node_env, col, role_names)?;
            match &col_val.kind {
                ValueKind::Map(m) => {
                    let k = eval_operand(local_env, node_env, key, role_names)?;
                    m.get(&*k).cloned().ok_or(RuntimeError::KeyNotFound)
                }
                ValueKind::List(l) => {
                    let idx =
                        eval_operand(local_env, node_env, key, role_names)?.as_int()? as usize;
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
            let tail_val = eval_operand(local_env, node_env, tail, role_names)?;
            let t = tail_val.as_list()?;
            let mut new_list = ValueSeq::<H>::with_capacity(t.len() + 1);
            new_list.push(h);
            new_list.extend_from_slice(t);
            Ok(Value::<H>::list(new_list))
        }
        Expr::ListAppend(list, item) => {
            let list_val = eval_operand(local_env, node_env, list, role_names)?;
            let mut l = list_val.as_list()?.clone();
            let i = eval(local_env, node_env, item, role_names)?;
            l.push(i);
            Ok(Value::<H>::list(l))
        }
        Expr::ListSubsequence(list, start, end) => {
            let l = eval_operand(local_env, node_env, list, role_names)?;
            let s = eval_operand(local_env, node_env, start, role_names)?.as_int()? as usize;
            let e = eval_operand(local_env, node_env, end, role_names)?.as_int()? as usize;
            let vec = l.as_list()?;
            if s > vec.len() || e > vec.len() || s > e {
                return Err(RuntimeError::SubsequenceOutOfBounds {
                    start: s,
                    end: e,
                    len: vec.len(),
                });
            }
            Ok(Value::<H>::list(ValueSeq::<H>::from(&vec[s..e])))
        }
        Expr::LessThanEquals(e1, e2) => Ok(Value::<H>::bool(
            *eval_operand(local_env, node_env, e1, role_names)?
                <= *eval_operand(local_env, node_env, e2, role_names)?,
        )),
        Expr::GreaterThan(e1, e2) => Ok(Value::<H>::bool(
            *eval_operand(local_env, node_env, e1, role_names)?
                > *eval_operand(local_env, node_env, e2, role_names)?,
        )),
        Expr::GreaterThanEquals(e1, e2) => Ok(Value::<H>::bool(
            *eval_operand(local_env, node_env, e1, role_names)?
                >= *eval_operand(local_env, node_env, e2, role_names)?,
        )),
        Expr::KeyExists(key, map) => {
            let k = eval_operand(local_env, node_env, key, role_names)?;
            let m = eval_operand(local_env, node_env, map, role_names)?;
            Ok(Value::<H>::bool(m.as_map()?.contains_key(&*k)))
        }
        Expr::MapErase(key, map) => {
            let k = eval_operand(local_env, node_env, key, role_names)?;
            let m = eval_operand(local_env, node_env, map, role_names)?;
            Ok(Value::<H>::map(m.as_map()?.without(&*k)))
        }
        Expr::ListLen(list) => {
            let list_val = eval_operand(local_env, node_env, list, role_names)?;
            match &list_val.kind {
                ValueKind::List(l) => Ok(Value::<H>::int(l.len() as i64)),
                ValueKind::Map(m) => Ok(Value::<H>::int(m.len() as i64)),
                _ => Err(RuntimeError::NotACollection {
                    got: list_val.type_name(),
                }),
            }
        }
        Expr::ListAccess(list, idx) => {
            let l = eval_operand(local_env, node_env, list, role_names)?;
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
            let v1 = eval_operand(local_env, node_env, e1, role_names)?.as_int()?;
            let v2 = eval_operand(local_env, node_env, e2, role_names)?.as_int()?;
            Ok(Value::<H>::int(v1.min(v2)))
        }
        Expr::TupleAccess(tuple, idx) => {
            let t = eval_operand(local_env, node_env, tuple, role_names)?;
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
            let val = eval_operand(local_env, node_env, e, role_names)?;
            match &val.kind {
                ValueKind::Option(Some(v)) => Ok((**v).clone()),
                ValueKind::Option(None) => Err(RuntimeError::UnwrapNone),
                _ => Err(RuntimeError::TypeError {
                    expected: "option",
                    got: val.type_name(),
                }),
            }
        }
        Expr::Coalesce(opt, default) => {
            let val = eval_operand(local_env, node_env, opt, role_names)?;
            match &val.kind {
                ValueKind::Option(Some(v)) => Ok((**v).clone()),
                ValueKind::Option(None) => eval(local_env, node_env, default, role_names),
                _ => Err(RuntimeError::CoalesceNonOption {
                    got: val.type_name(),
                }),
            }
        }
        Expr::IntToString(e) => {
            let n = eval_operand(local_env, node_env, e, role_names)?.as_int()?;
            Ok(Value::<H>::string(EcoString::from(Decimal::of_i64(n).as_str())))
        }
        Expr::BoolToString(e) => Ok(Value::<H>::string(EcoString::from(
            eval_operand(local_env, node_env, e, role_names)?
                .as_bool()?
                .to_string(),
        ))),
        Expr::NodeToString(e) => {
            let node_id = eval_operand(local_env, node_env, e, role_names)?.as_node()?;
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
            eval_operand(local_env, node_env, key, role_names)?,
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
            let val = eval_operand(local_env, node_env, expr, role_names)?;
            match &val.kind {
                ValueKind::Variant(_, variant_name, _) => {
                    Ok(Value::<H>::bool(variant_name == name))
                }
                _ => Ok(Value::<H>::bool(false)),
            }
        }
        Expr::VariantPayload(expr) => {
            let val = eval_operand(local_env, node_env, expr, role_names)?;
            match &val.kind {
                ValueKind::Variant(_, _, Some(payload)) => Ok((**payload).clone()),
                ValueKind::Variant(_, _, None) => Err(RuntimeError::VariantHasNoPayload),
                _ => Err(RuntimeError::TypeError {
                    expected: "variant",
                    got: val.type_name(),
                }),
            }
        }
        Expr::SafeFind(col, key) => {
            let col_val = eval_operand(local_env, node_env, col, role_names)?;
            match &col_val.kind {
                ValueKind::Option(None) => Ok(Value::<H>::option_none()),
                ValueKind::Option(Some(inner)) => {
                    let inner_val: &Value<H> = inner;
                    let key_val = eval_operand(local_env, node_env, key, role_names)?;
                    match &inner_val.kind {
                        ValueKind::Map(m) => {
                            let result = m.get(&*key_val).cloned().ok_or(RuntimeError::KeyNotFound)?;
                            Ok(Value::<H>::option_some(result))
                        }
                        ValueKind::List(l) => {
                            let idx = key_val.as_int()? as usize;
                            let result = l.get(idx).cloned().ok_or(RuntimeError::IndexOutOfBounds {
                                index: idx,
                                len: l.len(),
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
        Expr::SafeTupleAccess(tuple, idx) => {
            let t = eval_operand(local_env, node_env, tuple, role_names)?;
            match &t.kind {
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
                    got: t.type_name(),
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
    }

    #[test]
    fn test_build_frame() {
        let func = FunctionInfo {
            entry: 0,
            name: NameId(0),
            param_count: 1,
            local_slot_count: 2,
            local_defaults: vec![SlotDefault::Nil],
            is_sync: true,
            debug_slot_names: vec!["a".into(), "b".into()],
        };
        let args = [Value::<WithHashing>::int(5)];
        let local = build_frame(&func, &args);
        assert_eq!(local.get(0), &Value::<WithHashing>::int(5));
        assert_eq!(local.get(1), &Value::<WithHashing>::option_none());
        let mut reference = Env::<WithHashing>::with_slots(2);
        reference.set(0, Value::<WithHashing>::int(5));
        reference.set(1, Value::<WithHashing>::option_none());
        assert_eq!(
            local.sig, reference.sig,
            "the frame carries the signature a slot-by-slot build would give it"
        );
        assert_eq!(local, reference, "the frame holds the same slots");
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
