//! Checks decode-time rewrites by a method separate from the liveness the
//! decoder uses: a forward analysis over the decoded operations alone,
//! tracking the local slots whose value may differ from the value the same
//! slot holds when every vertex executes as its label does. A read of such a
//! slot is a failure.

use super::{CExpr, Dest, Op, Opnd};
use crate::compiler::cfg::{Expr, Instr, Label, Lhs, Program, VarSlot};
use std::collections::BTreeSet;

/// Per vertex, the slots whose value may differ on entry, or `None` when no
/// entry reaches the vertex; and every read of such a slot.
pub(crate) struct Analysis {
    pub differing_in: Vec<Option<BTreeSet<u32>>>,
    pub failures: Vec<String>,
}

fn opnd_reads(o: &Opnd, out: &mut Vec<u32>) {
    match o {
        Opnd::Local(s) => out.push(*s),
        Opnd::Node(_) | Opnd::Int(_) | Opnd::Str(_) | Opnd::Bool(_) | Opnd::Unit | Opnd::Nil => {}
        Opnd::Tree(e) => tree_reads(e, out),
    }
}

fn tree_reads(e: &CExpr, out: &mut Vec<u32>) {
    match e {
        CExpr::Not(a)
        | CExpr::ListLen(a)
        | CExpr::ListAccess(a, _)
        | CExpr::TupleAccess(a, _)
        | CExpr::Unwrap(a)
        | CExpr::Some(a)
        | CExpr::IntToString(a)
        | CExpr::BoolToString(a)
        | CExpr::NodeToString(a)
        | CExpr::IsVariant(a, _)
        | CExpr::VariantPayload(a)
        | CExpr::SafeTupleAccess(a, _)
        | CExpr::FieldGet(a, _) => opnd_reads(a, out),
        CExpr::Find(a, b)
        | CExpr::And(a, b)
        | CExpr::Or(a, b)
        | CExpr::EqualsEquals(a, b)
        | CExpr::NotEquals(a, b)
        | CExpr::ListPrepend(a, b)
        | CExpr::ListAppend(a, b)
        | CExpr::LessThan(a, b)
        | CExpr::LessThanEquals(a, b)
        | CExpr::GreaterThan(a, b)
        | CExpr::GreaterThanEquals(a, b)
        | CExpr::KeyExists(a, b)
        | CExpr::MapErase(a, b)
        | CExpr::Plus(a, b)
        | CExpr::Minus(a, b)
        | CExpr::Times(a, b)
        | CExpr::Div(a, b)
        | CExpr::Mod(a, b)
        | CExpr::Min(a, b)
        | CExpr::Coalesce(a, b)
        | CExpr::SafeFind(a, b) => {
            opnd_reads(a, out);
            opnd_reads(b, out);
        }
        CExpr::ListSubsequence(a, b, c) | CExpr::Store(a, b, c) => {
            opnd_reads(a, out);
            opnd_reads(b, out);
            opnd_reads(c, out);
        }
        CExpr::Map(pairs) => {
            for (k, v) in pairs {
                opnd_reads(k, out);
                opnd_reads(v, out);
            }
        }
        CExpr::StructLit(_, fields) => {
            for (_, v) in fields {
                opnd_reads(v, out);
            }
        }
        CExpr::List(items) | CExpr::Tuple(items) => {
            for item in items {
                opnd_reads(item, out);
            }
        }
        CExpr::Variant(_, _, payload) => {
            if let Some(p) = payload {
                opnd_reads(p, out);
            }
        }
    }
}

fn local(dest: Dest) -> Option<u32> {
    match dest {
        Dest::Local(s) => Some(s),
        Dest::Node(_) => None,
    }
}

/// One outgoing edge: its target, the slots written on the way, and the
/// slots whose value may differ after it because the vertex's store was
/// decoded away.
struct Edge {
    target: usize,
    written: Vec<u32>,
    differs: Vec<u32>,
}

fn edge(target: u32, written: Option<u32>) -> Edge {
    Edge {
        target: target as usize,
        written: written.into_iter().collect(),
        differs: Vec::new(),
    }
}

/// What one vertex does to local slots.
struct Step {
    /// Slots read, each checked against the slots differing on entry, less
    /// those in `written_first`.
    reads: Vec<u32>,
    /// Slots the vertex writes before its reads.
    written_first: Vec<u32>,
    edges: Vec<Edge>,
}

/// The local slot store of the label at `v`, with its source expression.
fn label_store(program: &Program, v: usize) -> Option<(u32, &Expr)> {
    match program.cfg.graph.get(v)? {
        Label::Instr(Instr::Assign(Lhs::Var(VarSlot::Local(s, _)), rhs), _)
        | Label::Instr(Instr::Copy(Lhs::Var(VarSlot::Local(s, _)), rhs), _) => Some((*s, rhs)),
        _ => None,
    }
}

fn step(program: &Program, ops: &[Op], v: usize, failures: &mut Vec<String>) -> Step {
    let mut reads = Vec::new();
    let mut written_first = Vec::new();
    let edges = match &ops[v] {
        Op::AssignLocal { slot, next, rhs } => {
            opnd_reads(rhs, &mut reads);
            vec![edge(*next, Some(*slot))]
        }
        Op::AssignNode { next, rhs, .. } => {
            opnd_reads(rhs, &mut reads);
            vec![edge(*next, None)]
        }
        Op::SyncCall(call) => {
            for a in &call.args {
                opnd_reads(a, &mut reads);
            }
            vec![edge(call.next, local(call.dest))]
        }
        Op::Async(call) => {
            opnd_reads(&call.target, &mut reads);
            for a in &call.args {
                opnd_reads(a, &mut reads);
            }
            vec![edge(call.next, local(call.dest))]
        }
        Op::MakeChannel { dest, next } => vec![edge(*next, local(*dest))],
        Op::MakeFifoLink { dest, next, peer } => {
            opnd_reads(peer, &mut reads);
            vec![edge(*next, local(*dest))]
        }
        Op::SetTimer { dest, next, .. } => vec![edge(*next, local(*dest))],
        Op::UniqueId { dest, next } => vec![edge(*next, local(*dest))],
        Op::CondLocal { slot, then, els } => {
            reads.push(*slot);
            vec![edge(*then, None), edge(*els, None)]
        }
        Op::CondNode { then, els, .. } => vec![edge(*then, None), edge(*els, None)],
        Op::Cond { cond, then, els } => {
            opnd_reads(cond, &mut reads);
            vec![edge(*then, None), edge(*els, None)]
        }
        Op::Return(value) => {
            opnd_reads(value, &mut reads);
            vec![]
        }
        Op::Print { value, next } => {
            opnd_reads(value, &mut reads);
            vec![edge(*next, None)]
        }
        Op::Goto(target) => vec![edge(*target, None)],
        Op::PersistData { next, value, .. } => {
            opnd_reads(value, &mut reads);
            vec![edge(*next, None)]
        }
        Op::RetrieveData { dest, next, .. } => vec![edge(*next, local(*dest))],
        Op::DiscardData(next) => vec![edge(*next, None)],
        Op::ForLoopIn(fl) => {
            if let Some(it) = fl.iter_slot {
                reads.push(it);
            }
            opnd_reads(&fl.collection, &mut reads);
            let mut body = edge(fl.body, fl.iter_slot);
            body.written.extend(local(fl.dest));
            vec![body, edge(fl.next, fl.iter_slot)]
        }
        Op::TraceDispatch(td) => {
            for p in &td.params {
                opnd_reads(p, &mut reads);
            }
            vec![edge(td.next, None)]
        }
        Op::TraceEnter(te) => {
            written_first.extend(local(te.dest));
            for p in &te.params {
                opnd_reads(p, &mut reads);
            }
            vec![edge(te.next, local(te.dest))]
        }
        Op::TraceExit(tx) => {
            opnd_reads(&tx.trace_id, &mut reads);
            opnd_reads(&tx.return_value, &mut reads);
            vec![edge(tx.next, None)]
        }
        Op::Send(send) => {
            opnd_reads(&send.chan, &mut reads);
            opnd_reads(&send.value, &mut reads);
            vec![edge(send.next, None)]
        }
        Op::Recv(recv) => {
            opnd_reads(&recv.chan, &mut reads);
            vec![edge(recv.next, local(recv.dest))]
        }
        Op::Pause(next) => vec![edge(*next, None)],
        Op::SpinAwait { cond, next } => {
            opnd_reads(cond, &mut reads);
            vec![edge(*next, None), edge(v as u32, None)]
        }
        Op::StoreSkipped(next) => {
            let mut e = edge(*next, None);
            match label_store(program, v) {
                Some((s, Expr::Var(VarSlot::Local(src, _)))) if *src == s => {}
                Some((
                    s,
                    Expr::Var(_)
                    | Expr::Int(_)
                    | Expr::Bool(_)
                    | Expr::String(_)
                    | Expr::Unit
                    | Expr::Nil,
                )) => e.differs.push(s),
                Some((s, _)) => {
                    failures.push(format!("vertex {v}: skipped store of slot {s} has a subtree source"));
                    e.differs.push(s);
                }
                None => failures.push(format!("vertex {v}: skipped store on a label that is not a local store")),
            }
            vec![e]
        }
    };
    Step {
        reads,
        written_first,
        edges,
    }
}

/// The local slots vertex `v` of `ops` reads, and each successor with the
/// local slots written on the way to it.
pub(crate) fn accesses(program: &Program, ops: &[Op], v: usize) -> (Vec<u32>, Vec<(usize, Vec<u32>)>) {
    let s = step(program, ops, v, &mut Vec::new());
    (s.reads, s.edges.into_iter().map(|e| (e.target, e.written)).collect())
}

/// Runs the analysis over `ops`, the decoded form of `program`, starting from
/// every vertex in `entries` with no slot differing.
pub(crate) fn analyze(program: &Program, ops: &[Op], entries: &[usize]) -> Analysis {
    let n = ops.len();
    let mut failures = Vec::new();
    let steps: Vec<Step> = (0..n).map(|v| step(program, ops, v, &mut failures)).collect();
    let mut differing_in: Vec<Option<BTreeSet<u32>>> = vec![None; n];
    let mut work: Vec<usize> = Vec::new();
    for &e in entries {
        if e < n && differing_in[e].is_none() {
            differing_in[e] = Some(BTreeSet::new());
            work.push(e);
        }
    }
    while let Some(v) = work.pop() {
        let Some(inn) = differing_in[v].clone() else { continue };
        for e in &steps[v].edges {
            if e.target >= n {
                failures.push(format!("vertex {v}: successor {} outside the graph", e.target));
                continue;
            }
            let mut out = inn.clone();
            for s in &e.written {
                out.remove(s);
            }
            out.extend(e.differs.iter().copied());
            let merged = match &differing_in[e.target] {
                None => out,
                Some(old) => {
                    if out.is_subset(old) {
                        continue;
                    }
                    old.union(&out).copied().collect()
                }
            };
            differing_in[e.target] = Some(merged);
            work.push(e.target);
        }
    }
    for (v, s) in steps.iter().enumerate() {
        let Some(inn) = &differing_in[v] else { continue };
        for r in &s.reads {
            if inn.contains(r) && !s.written_first.contains(r) {
                failures.push(format!("vertex {v}: reads slot {r}, whose value may differ"));
            }
        }
    }
    Analysis {
        differing_in,
        failures,
    }
}

/// Every function entry of `program`.
pub(crate) fn entries(program: &Program) -> Vec<usize> {
    let mut out: Vec<usize> = program.rpc.values().map(|f| f.entry).collect();
    out.sort_unstable();
    out
}
