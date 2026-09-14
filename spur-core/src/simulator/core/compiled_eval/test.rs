//! Differential check of the decoded evaluator against `eval` and
//! `eval_operand`: every expression kind, over operands that succeed and
//! operands that fail, must give the same value (signature included), the
//! same error text and the same evaluator events.

use super::*;
use crate::compiler::cfg::compiled::opnd;
use crate::compiler::cfg::{Expr, VarSlot};
use crate::simulator::core::eval::{eval, eval_operand};
use crate::simulator::core::state::NodeId;
use crate::simulator::core::values::ChannelId;
use crate::simulator::hash_utils::{NoHashing, WithHashing};
use std::collections::BTreeSet;

const EXPR_KINDS: usize = 44;
const LEAF_KINDS: [usize; 6] = [0, 2, 3, 13, 31, 32];

/// Position of the expression's kind in the declaration order of `Expr`.
fn kind(e: &Expr) -> usize {
    match e {
        Expr::Var(_) => 0,
        Expr::Find(_, _) => 1,
        Expr::Int(_) => 2,
        Expr::Bool(_) => 3,
        Expr::Not(_) => 4,
        Expr::And(_, _) => 5,
        Expr::Or(_, _) => 6,
        Expr::EqualsEquals(_, _) => 7,
        Expr::Map(_) => 8,
        Expr::List(_) => 9,
        Expr::ListPrepend(_, _) => 10,
        Expr::ListAppend(_, _) => 11,
        Expr::ListSubsequence(_, _, _) => 12,
        Expr::String(_) => 13,
        Expr::LessThan(_, _) => 14,
        Expr::LessThanEquals(_, _) => 15,
        Expr::GreaterThan(_, _) => 16,
        Expr::GreaterThanEquals(_, _) => 17,
        Expr::KeyExists(_, _) => 18,
        Expr::MapErase(_, _) => 19,
        Expr::Store(_, _, _) => 20,
        Expr::ListLen(_) => 21,
        Expr::ListAccess(_, _) => 22,
        Expr::Plus(_, _) => 23,
        Expr::Minus(_, _) => 24,
        Expr::Times(_, _) => 25,
        Expr::Div(_, _) => 26,
        Expr::Mod(_, _) => 27,
        Expr::Min(_, _) => 28,
        Expr::Tuple(_) => 29,
        Expr::TupleAccess(_, _) => 30,
        Expr::Unit => 31,
        Expr::Nil => 32,
        Expr::Unwrap(_) => 33,
        Expr::Coalesce(_, _) => 34,
        Expr::Some(_) => 35,
        Expr::IntToString(_) => 36,
        Expr::BoolToString(_) => 37,
        Expr::NodeToString(_) => 38,
        Expr::Variant(_, _, _) => 39,
        Expr::IsVariant(_, _) => 40,
        Expr::VariantPayload(_) => 41,
        Expr::SafeFind(_, _) => 42,
        Expr::SafeTupleAccess(_, _) => 43,
    }
}

fn b(e: &Expr) -> Box<Expr> {
    Box::new(e.clone())
}

fn local(i: u32) -> Expr {
    Expr::Var(VarSlot::Local(i, NameId(0)))
}

fn node(i: u32) -> Expr {
    Expr::Var(VarSlot::Node(i, NameId(0)))
}

fn s(text: &str) -> Expr {
    Expr::String(text.into())
}

fn int_list<H: HashPolicy>(items: &[i64]) -> Value<H> {
    Value::<H>::list(items.iter().map(|i| Value::<H>::int(*i)).collect())
}

fn string_map<H: HashPolicy>(items: &[(&str, i64)]) -> Value<H> {
    let mut m = ValueMap::<H>::new();
    for (k, v) in items {
        m.insert(Value::<H>::string((*k).into()), Value::<H>::int(*v));
    }
    Value::<H>::map(m)
}

const LOCAL_SLOTS: u32 = 21;

fn local_env<H: HashPolicy>() -> Env<H> {
    let server = NodeId {
        role: NameId(0),
        index: 1,
    };
    let stranger = NodeId {
        role: NameId(99),
        index: 4,
    };
    let mut int_keyed = ValueMap::<H>::new();
    int_keyed.insert(Value::<H>::int(1), Value::<H>::string("one".into()));
    let tuple = Value::<H>::tuple(
        [Value::<H>::int(1), Value::<H>::string("x".into())]
            .into_iter()
            .collect(),
    );
    let values: Vec<Value<H>> = vec![
        Value::<H>::int(7),
        Value::<H>::int(0),
        Value::<H>::bool(true),
        Value::<H>::bool(false),
        Value::<H>::string("ab".into()),
        int_list(&[1, 2, 3]),
        string_map(&[("a", 1), ("b", 2)]),
        tuple.clone(),
        Value::<H>::option_some(Value::<H>::int(5)),
        Value::<H>::option_none(),
        Value::<H>::variant(0, "V".into(), Some(Arc::new(Value::<H>::int(9)))),
        Value::<H>::variant(0, "W".into(), None),
        Value::<H>::node(server),
        Value::<H>::unit(),
        Value::<H>::channel(ChannelId {
            node: server,
            id: 3,
        }),
        Value::<H>::map(int_keyed),
        Value::<H>::option_some(string_map(&[("a", 1)])),
        Value::<H>::option_some(int_list(&[4, 5])),
        Value::<H>::option_some(tuple),
        int_list(&[]),
        Value::<H>::node(stranger),
    ];
    assert_eq!(values.len() as u32, LOCAL_SLOTS);
    let mut env = Env::<H>::with_slots(values.len());
    for (i, v) in values.into_iter().enumerate() {
        env.set(i as u32, v);
    }
    env
}

fn node_env<H: HashPolicy>() -> Env<H> {
    let mut env = Env::<H>::with_slots(3);
    env.set(0, Value::<H>::int(100));
    let mut m = ValueMap::<H>::new();
    m.insert(Value::<H>::string("k".into()), int_list(&[8, 9]));
    env.set(1, Value::<H>::map(m));
    env.set(2, Value::<H>::string("node".into()));
    env
}

/// Operands for every child position: each slot, literals of every kind,
/// subtrees of several kinds, and a subtree that fails.
fn pool() -> Vec<Expr> {
    let mut p: Vec<Expr> = (0..LOCAL_SLOTS).map(local).collect();
    p.extend((0..3).map(node));
    p.extend([
        Expr::Int(2),
        Expr::Int(0),
        Expr::Int(-1),
        Expr::Int(1),
        Expr::Bool(true),
        Expr::Bool(false),
        s("a"),
        s("ab"),
        Expr::Unit,
        Expr::Nil,
    ]);
    p.extend([
        Expr::Plus(b(&local(0)), b(&Expr::Int(1))),
        Expr::Plus(b(&Expr::Bool(true)), b(&Expr::Int(1))),
        Expr::Some(b(&Expr::Int(3))),
        Expr::List(vec![Expr::Int(1), s("a")]),
        Expr::Map(vec![(s("a"), Expr::Int(1))]),
        Expr::Tuple(vec![Expr::Int(1), Expr::Int(2)]),
        Expr::Variant(0, "V".into(), Some(b(&Expr::Int(9)))),
        Expr::EqualsEquals(b(&local(0)), b(&Expr::Int(7))),
    ]);
    p
}

/// A small operand set for the third position of a three-child kind.
fn short_pool() -> Vec<Expr> {
    vec![
        local(0),
        local(1),
        local(4),
        Expr::Int(1),
        Expr::Int(2),
        Expr::Int(9),
        s("a"),
        Expr::Plus(b(&Expr::Bool(true)), b(&Expr::Int(1))),
        Expr::List(vec![Expr::Int(3)]),
    ]
}

fn expressions() -> Vec<Expr> {
    let p = pool();
    let q = short_pool();
    let mut out = Vec::new();
    for x in &p {
        out.push(x.clone());
        out.push(Expr::Not(b(x)));
        out.push(Expr::ListLen(b(x)));
        out.push(Expr::Unwrap(b(x)));
        out.push(Expr::Some(b(x)));
        out.push(Expr::IntToString(b(x)));
        out.push(Expr::BoolToString(b(x)));
        out.push(Expr::NodeToString(b(x)));
        out.push(Expr::VariantPayload(b(x)));
        out.push(Expr::List(vec![x.clone()]));
        out.push(Expr::Tuple(vec![x.clone()]));
        out.push(Expr::Map(vec![(x.clone(), Expr::Int(1))]));
        out.push(Expr::Map(vec![(s("k"), x.clone())]));
        out.push(Expr::Variant(2, "P".into(), Some(b(x))));
        for name in ["V", "X"] {
            out.push(Expr::IsVariant(b(x), name.into()));
        }
        for idx in [0usize, 1, 5] {
            out.push(Expr::ListAccess(b(x), idx));
            out.push(Expr::TupleAccess(b(x), idx));
            out.push(Expr::SafeTupleAccess(b(x), idx));
        }
        for y in &p {
            let (bx, by) = (b(x), b(y));
            out.push(Expr::Find(bx.clone(), by.clone()));
            out.push(Expr::And(bx.clone(), by.clone()));
            out.push(Expr::Or(bx.clone(), by.clone()));
            out.push(Expr::EqualsEquals(bx.clone(), by.clone()));
            out.push(Expr::Not(Box::new(Expr::EqualsEquals(bx.clone(), by.clone()))));
            out.push(Expr::Map(vec![(x.clone(), y.clone())]));
            out.push(Expr::List(vec![x.clone(), y.clone()]));
            out.push(Expr::Tuple(vec![x.clone(), y.clone()]));
            out.push(Expr::ListPrepend(bx.clone(), by.clone()));
            out.push(Expr::ListAppend(bx.clone(), by.clone()));
            out.push(Expr::LessThan(bx.clone(), by.clone()));
            out.push(Expr::LessThanEquals(bx.clone(), by.clone()));
            out.push(Expr::GreaterThan(bx.clone(), by.clone()));
            out.push(Expr::GreaterThanEquals(bx.clone(), by.clone()));
            out.push(Expr::KeyExists(bx.clone(), by.clone()));
            out.push(Expr::MapErase(bx.clone(), by.clone()));
            out.push(Expr::Plus(bx.clone(), by.clone()));
            out.push(Expr::Minus(bx.clone(), by.clone()));
            out.push(Expr::Times(bx.clone(), by.clone()));
            out.push(Expr::Div(bx.clone(), by.clone()));
            out.push(Expr::Mod(bx.clone(), by.clone()));
            out.push(Expr::Min(bx.clone(), by.clone()));
            out.push(Expr::Coalesce(bx.clone(), by.clone()));
            out.push(Expr::SafeFind(bx.clone(), by.clone()));
            for z in &q {
                out.push(Expr::ListSubsequence(bx.clone(), by.clone(), b(z)));
                out.push(Expr::Store(bx.clone(), by.clone(), b(z)));
            }
        }
    }
    out.push(Expr::Map(vec![]));
    out.push(Expr::List(vec![]));
    out.push(Expr::Tuple(vec![]));
    out.push(Expr::Variant(1, "E".into(), None));
    out
}

/// Whether evaluating `e` would divide by zero, which panics on both paths.
fn divides_by_zero<H: HashPolicy>(
    e: &Expr,
    l: &Env<H>,
    n: &Env<H>,
    roles: &HashMap<NameId, String>,
) -> bool {
    match e {
        Expr::Div(_, d) | Expr::Mod(_, d) => {
            matches!(eval(l, n, d, roles), Ok(v) if matches!(v.kind, ValueKind::Int(0)))
        }
        _ => false,
    }
}

fn outcome<H: HashPolicy>(r: &Result<Value<H>, RuntimeError>) -> String {
    match r {
        Ok(v) => format!("ok {v:?}"),
        Err(e) => format!("err {e} / {e:?}"),
    }
}

#[derive(Default)]
struct Coverage {
    succeeded: BTreeSet<usize>,
    failed: BTreeSet<usize>,
}

fn check_all<H: HashPolicy>() -> (usize, Coverage) {
    let l = local_env::<H>();
    let n = node_env::<H>();
    let mut roles = HashMap::new();
    roles.insert(NameId(0), "Server".to_string());
    let mut coverage = Coverage::default();
    let mut checked = 0;
    for e in expressions() {
        if divides_by_zero(&e, &l, &n, &roles) {
            continue;
        }
        let decoded = opnd(&e);

        let before = util_stats::pending_evaluator_events();
        let legacy = eval(&l, &n, &e, &roles);
        let mid = util_stats::pending_evaluator_events();
        let mut t = InterpreterTally::new();
        let compiled = cvalue(&l, &n, &decoded, &roles, &mut t);
        let after = util_stats::pending_evaluator_events();
        assert_eq!(outcome(&legacy), outcome(&compiled), "kept value of {e:?}");
        if let (Ok(a), Ok(c)) = (&legacy, &compiled) {
            assert!(a == c && a.sig == c.sig, "kept value of {e:?}");
        }
        let legacy_events: Vec<u64> = (0..3).map(|i| mid[i] - before[i]).collect();
        let compiled_events: Vec<u64> = (0..3).map(|i| after[i] - mid[i]).collect();
        assert_eq!(legacy_events, compiled_events, "events of kept {e:?}");

        let before = util_stats::pending_evaluator_events();
        let legacy = eval_operand(&l, &n, &e, &roles);
        let mid = util_stats::pending_evaluator_events();
        let mut t = InterpreterTally::new();
        let compiled = coperand(&l, &n, &decoded, &roles, &mut t);
        let after = util_stats::pending_evaluator_events();
        match (&legacy, &compiled) {
            (Ok(a), Ok(c)) => {
                assert_eq!(format!("{:?}", &**a), format!("{:?}", &**c), "read value of {e:?}");
                assert_eq!(
                    matches!(a, Operand::Borrowed(_)),
                    matches!(c, Operand::Borrowed(_)),
                    "read position of {e:?}"
                );
            }
            (Err(a), Err(c)) => {
                assert_eq!(format!("{a} / {a:?}"), format!("{c} / {c:?}"), "read error of {e:?}")
            }
            _ => panic!("read of {e:?} succeeded on one path only"),
        }
        let legacy_events: Vec<u64> = (0..3).map(|i| mid[i] - before[i]).collect();
        let compiled_events: Vec<u64> = (0..3).map(|i| after[i] - mid[i]).collect();
        assert_eq!(legacy_events, compiled_events, "events of read {e:?}");

        if legacy.is_ok() {
            coverage.succeeded.insert(kind(&e));
        } else {
            coverage.failed.insert(kind(&e));
        }
        checked += 1;
    }
    (checked, coverage)
}

fn assert_covered(coverage: &Coverage) {
    let all: BTreeSet<usize> = (0..EXPR_KINDS).collect();
    assert_eq!(coverage.succeeded, all, "every kind has a succeeding case");
    let with_children: BTreeSet<usize> = all
        .iter()
        .copied()
        .filter(|k| !LEAF_KINDS.contains(k))
        .collect();
    assert!(
        with_children.is_subset(&coverage.failed),
        "every kind with a child has a failing case; missing {:?}",
        with_children.difference(&coverage.failed).collect::<Vec<_>>()
    );
}

#[test]
fn decoded_evaluation_matches_eval_with_eager_signatures() {
    let (checked, coverage) = check_all::<WithHashing>();
    assert!(checked > 50_000);
    assert_covered(&coverage);
}

#[test]
fn decoded_evaluation_matches_eval_with_deferred_signatures() {
    let (checked, coverage) = check_all::<NoHashing>();
    assert!(checked > 50_000);
    assert_covered(&coverage);
}

#[test]
fn not_equals_and_string_field_reads_decode_to_their_fused_forms() {
    let ne = opnd(&Expr::Not(b(&Expr::EqualsEquals(b(&local(0)), b(&Expr::Int(1))))));
    assert!(matches!(ne, Opnd::Tree(ref t) if matches!(**t, CExpr::NotEquals(Opnd::Local(0), Opnd::Int(1)))));
    let field = opnd(&Expr::Find(b(&node(1)), b(&s("k"))));
    assert!(matches!(field, Opnd::Tree(ref t) if matches!(**t, CExpr::FieldGet(Opnd::Node(1), _))));
}

fn events_between(a: [u64; 3], b: [u64; 3]) -> Vec<u64> {
    (0..3).map(|i| b[i].wrapping_sub(a[i])).collect()
}

fn vr_literal(set: &[&str]) -> Expr {
    Expr::Map(
        set.iter()
            .enumerate()
            .map(|(i, k)| {
                let v = if i % 2 == 0 { local(i as u32) } else { Expr::Int(i as i64) };
                (s(k), v)
            })
            .collect(),
    )
}

fn is_struct_literal(e: &Expr) -> bool {
    matches!(opnd(e), Opnd::Tree(ref t) if matches!(**t, CExpr::StructLit(_, _)))
}

fn struct_literals_evaluate_like_eval<H: HashPolicy>() {
    use crate::simulator::core::values::struct_tests::VR_KEY_SETS;
    let l = local_env::<H>();
    let n = node_env::<H>();
    let roles = HashMap::new();
    for set in VR_KEY_SETS {
        let lit = vr_literal(set);
        let read = Expr::Find(b(&lit), b(&s(set[set.len() - 1])));
        let missing = Expr::Find(b(&lit), b(&s("missing")));
        for e in [lit, read, missing] {
            let decoded = opnd(&e);
            let before = util_stats::pending_evaluator_events();
            let legacy = eval(&l, &n, &e, &roles);
            let mid = util_stats::pending_evaluator_events();
            let mut t = InterpreterTally::new();
            let compiled = cvalue(&l, &n, &decoded, &roles, &mut t);
            let after = util_stats::pending_evaluator_events();
            assert_eq!(outcome(&legacy), outcome(&compiled), "{e:?}");
            if let (Ok(a), Ok(c)) = (&legacy, &compiled) {
                assert!(a == c && a.sig == c.sig, "{e:?}");
            }
            assert_eq!(events_between(before, mid), events_between(mid, after), "{e:?}");
        }
    }
}

#[test]
fn struct_shaped_map_literals_decode_to_struct_literals() {
    use crate::simulator::core::values::struct_shape;
    use crate::simulator::core::values::struct_tests::VR_KEY_SETS;
    for set in VR_KEY_SETS {
        assert!(is_struct_literal(&vr_literal(set)), "{set:?}");
    }

    let names: Vec<String> = (0..64).map(|i| format!("k{i}")).collect();
    let (i, j) = (1..names.len())
        .find_map(|j| {
            (0..j)
                .find(|&i| struct_shape(&[names[i].as_str().into(), names[j].as_str().into()]).is_none())
                .map(|i| (i, j))
        })
        .expect("two of 64 names share one of 32 home slots");
    let (first, second) = (names[i].as_str(), names[j].as_str());
    let order = |x: &str, y: &str| {
        let mut m = ValueMap::<NoHashing>::new();
        m.insert(Value::string(x.into()), Value::unit());
        m.insert(Value::string(y.into()), Value::unit());
        m.iter().map(|(k, _)| k.to_string()).collect::<Vec<_>>()
    };
    assert_ne!(order(first, second), order(second, first), "the key set is order sensitive");
    let shared_slot = vr_literal(&[first, second]);
    assert!(matches!(opnd(&shared_slot), Opnd::Tree(ref t) if matches!(**t, CExpr::Map(_))));

    assert!(!is_struct_literal(&vr_literal(&["x", "x"])));
    let sixteen: Vec<String> = (0..16).map(|i| format!("wide_{i}")).collect();
    let sixteen: Vec<&str> = sixteen.iter().map(|k| k.as_str()).collect();
    assert!(!is_struct_literal(&vr_literal(&sixteen)));
    assert!(!is_struct_literal(&Expr::Map(vec![])));
    assert!(!is_struct_literal(&Expr::Map(vec![(Expr::Int(1), Expr::Int(2))])));

    struct_literals_evaluate_like_eval::<WithHashing>();
    struct_literals_evaluate_like_eval::<NoHashing>();
}

fn struct_literal_positions(e: &Opnd) -> Vec<usize> {
    match e {
        Opnd::Tree(t) => match &**t {
            CExpr::StructLit(_, fields) => fields.iter().map(|(pos, _)| *pos).collect(),
            _ => panic!("not a struct literal"),
        },
        _ => panic!("not a struct literal"),
    }
}

fn struct_literals_count_their_field_order<H: HashPolicy>() {
    use crate::simulator::core::values::struct_shape;
    use crate::simulator::core::values::struct_tests::VR_KEY_SETS;
    let l = local_env::<H>();
    let n = node_env::<H>();
    let roles = HashMap::new();
    let (mut in_order_seen, mut permuted_seen) = (0, 0);
    for set in VR_KEY_SETS {
        let names: Vec<EcoString> = set.iter().map(|k| EcoString::from(*k)).collect();
        let shape = struct_shape(&names).expect("a VR key set is a struct shape");
        let mut sorted: Vec<&str> = set.to_vec();
        sorted.sort_by_key(|k| shape.position(k));
        let mut reversed = sorted.clone();
        reversed.reverse();
        for keys in [set.to_vec(), sorted, reversed] {
            let lit = vr_literal(&keys);
            let decoded = opnd(&lit);
            let positions = struct_literal_positions(&decoded);
            let in_order = positions.iter().enumerate().all(|(i, pos)| i == *pos);
            let before = util_stats::pending_struct_literal_counts();
            let legacy = eval(&l, &n, &lit, &roles);
            let mut t = InterpreterTally::new();
            let compiled = cvalue(&l, &n, &decoded, &roles, &mut t);
            let after = util_stats::pending_struct_literal_counts();
            let a = legacy.expect("the literal evaluates");
            let c = compiled.expect("the literal evaluates");
            assert!(a == c && a.sig == c.sig, "{keys:?}");
            assert_eq!(
                [after[0] - before[0], after[1] - before[1], after[2] - before[2]],
                [1, in_order as u64, !in_order as u64],
                "{keys:?}"
            );
            if in_order {
                in_order_seen += 1;
            } else {
                permuted_seen += 1;
            }
        }
    }
    assert!(in_order_seen > 0 && permuted_seen > 0);
}

fn struct_literals_fail_at_the_failing_field<H: HashPolicy>() {
    use crate::simulator::core::values::struct_tests::VR_KEY_SETS;
    let l = local_env::<H>();
    let n = node_env::<H>();
    let roles = HashMap::new();
    let failing = Expr::Plus(b(&Expr::Bool(true)), b(&Expr::Int(1)));
    let mut checked = 0;
    for set in VR_KEY_SETS.iter().filter(|set| set.len() >= 3) {
        for fail_at in 0..set.len() {
            let lit = Expr::Map(
                set.iter()
                    .enumerate()
                    .map(|(i, k)| {
                        let v = if i == fail_at {
                            failing.clone()
                        } else {
                            Expr::List(vec![Expr::Int(i as i64), s(k)])
                        };
                        (s(k), v)
                    })
                    .collect(),
            );
            assert!(is_struct_literal(&lit), "{set:?}");
            let decoded = opnd(&lit);
            let literals_before = util_stats::pending_struct_literal_counts();
            let before = util_stats::pending_evaluator_events();
            let legacy = eval(&l, &n, &lit, &roles);
            let mid = util_stats::pending_evaluator_events();
            let mut t = InterpreterTally::new();
            let compiled = cvalue(&l, &n, &decoded, &roles, &mut t);
            let after = util_stats::pending_evaluator_events();
            assert!(compiled.is_err(), "{set:?} failing at {fail_at}");
            assert_eq!(outcome(&legacy), outcome(&compiled), "{set:?} failing at {fail_at}");
            assert_eq!(events_between(before, mid), events_between(mid, after));
            if !H::EAGER {
                assert_eq!(
                    after[2].wrapping_sub(mid[2]),
                    fail_at as u64,
                    "the fields before the failing one count as literal entries"
                );
            }
            assert_eq!(
                util_stats::pending_struct_literal_counts(),
                literals_before,
                "a failed literal is not counted as a literal"
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

#[test]
fn struct_literals_place_fields_and_count_their_order_under_both_policies() {
    struct_literals_count_their_field_order::<WithHashing>();
    struct_literals_count_their_field_order::<NoHashing>();
    struct_literals_fail_at_the_failing_field::<WithHashing>();
    struct_literals_fail_at_the_failing_field::<NoHashing>();
}

#[test]
fn slot_and_literal_operands_are_read_without_entering_the_tree_evaluator() {
    let l = local_env::<NoHashing>();
    let n = node_env::<NoHashing>();
    let roles = HashMap::new();
    let e = opnd(&Expr::Plus(b(&local(0)), b(&Expr::Int(1))));
    let mut t = InterpreterTally::new();
    let v = cvalue(&l, &n, &e, &roles, &mut t).unwrap();
    assert_eq!(v, Value::<NoHashing>::int(8));
    assert_eq!(t.tree_evals, 1);
    assert_eq!(t.leaf_operands_inline, 2);
}
