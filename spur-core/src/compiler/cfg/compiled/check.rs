//! Checks decode-time rewrites by a method separate from the liveness the
//! decoder uses: a forward analysis over the decoded operations alone,
//! tracking the local slots whose value may differ from the value the same
//! slot holds when every vertex executes as its label does. A read of such a
//! slot is a failure.

use super::{CExpr, Dest, Op, Opnd, PieceSource, PrintPart};
use crate::compiler::cfg::{Expr, Instr, Label, Lhs, Program, VarSlot};
use std::collections::{BTreeSet, HashMap};

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
        | CExpr::IndexOf(a, b)
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
        Op::Spawn(s) => { opnd_reads(&s.count, &mut reads); vec![edge(s.next, local(s.dest))] }
        Op::Provide(p) | Op::ProvideAll(p) => { opnd_reads(&p.handle, &mut reads); opnd_reads(&p.value, &mut reads); vec![edge(p.next, None)] }
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
        Op::StoreFolded(next) => {
            let mut e = edge(*next, None);
            match label_store(program, v) {
                Some((s, _)) => e.differs.push(s),
                None => failures.push(format!("vertex {v}: folded store on a label that is not a local store")),
            }
            vec![e]
        }
        Op::PrintParts { next, .. } => {
            let mut e = edge(*next, None);
            match program.cfg.graph.get(v) {
                Some(Label::Print(Expr::Var(VarSlot::Local(target, _)), _)) => e.differs.push(*target),
                _ => failures.push(format!("vertex {v}: folded print on a label that is not a print of a local slot")),
            }
            vec![e]
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
    let mut steps: Vec<Step> = (0..n).map(|v| step(program, ops, v, &mut failures)).collect();
    let piece_reads = check_folded_prints(program, ops, entries, &mut steps, &mut failures);
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
    for (head, p, slot) in piece_reads {
        if differing_in[head].as_ref().is_some_and(|inn| inn.contains(&slot)) {
            failures.push(format!(
                "vertex {p}: folded print reads slot {slot}, whose value may differ at chain head {head}"
            ));
        }
    }
    Analysis {
        differing_in,
        failures,
    }
}

/// A value the labels of a folded chain compute.
#[derive(Clone)]
enum ChainValue {
    /// The value a slot held at the chain head.
    Head(PieceSource),
    /// A string: pieces in text order, each slot piece with the index of the
    /// check the labels make for it, in evaluation order.
    Text(Vec<(PrintPart, Option<u32>)>),
    /// Written by a store the decoded form does not execute.
    Unknown,
}

#[derive(Default)]
struct ChainLabels {
    env: HashMap<u32, ChainValue>,
    checks: u32,
    trees: u32,
}

impl ChainLabels {
    fn next_check(&mut self) -> u32 {
        self.checks += 1;
        self.checks - 1
    }

    /// Evaluates `e` as the label evaluator does: operands left to right,
    /// each check after its operands.
    fn eval(&mut self, e: &Expr) -> Result<ChainValue, String> {
        match e {
            Expr::Var(VarSlot::Local(s, _)) => match self.env.get(s) {
                None => Ok(ChainValue::Head(PieceSource::Local(*s))),
                Some(ChainValue::Unknown) => Err(format!("reads slot {s} after a store that does not run")),
                Some(v) => Ok(v.clone()),
            },
            Expr::Var(VarSlot::Node(s, _)) => Ok(ChainValue::Head(PieceSource::Node(*s))),
            Expr::String(s) => Ok(ChainValue::Text(vec![(PrintPart::Lit(s.clone()), None)])),
            Expr::IntToString(a) => {
                self.trees += 1;
                match self.eval(a)? {
                    ChainValue::Head(src) => {
                        Ok(ChainValue::Text(vec![(PrintPart::Int(src), Some(self.next_check()))]))
                    }
                    _ => Err("converts a value that is not a slot to decimal".to_string()),
                }
            }
            Expr::Plus(a, b) => {
                self.trees += 1;
                let (a, b) = (self.eval(a)?, self.eval(b)?);
                match (a, b) {
                    (ChainValue::Head(src), ChainValue::Text(rest)) => {
                        let mut out = vec![(PrintPart::Str { src, own_type: true }, Some(self.next_check()))];
                        out.extend(rest);
                        Ok(ChainValue::Text(out))
                    }
                    (ChainValue::Text(mut out), ChainValue::Head(src)) => {
                        out.push((PrintPart::Str { src, own_type: false }, Some(self.next_check())));
                        Ok(ChainValue::Text(out))
                    }
                    (ChainValue::Text(mut out), ChainValue::Text(rest)) => {
                        out.extend(rest);
                        Ok(ChainValue::Text(out))
                    }
                    _ => Err("adds values not known to be a string".to_string()),
                }
            }
            _ => Err("stores a value that is not a concatenation of strings".to_string()),
        }
    }
}

/// The pieces and tree count the labels of `chain` build for the print at
/// `p`, given that each folded store is not executed and each skipped store
/// other than a self-copy leaves its slot unknown.
fn chain_pieces(program: &Program, ops: &[Op], chain: &[usize], p: usize) -> Result<(Vec<PrintPart>, u32), String> {
    let Some(Label::Print(Expr::Var(VarSlot::Local(target, _)), _)) = program.cfg.graph.get(p) else {
        return Err("the label is not a print of a local slot".to_string());
    };
    let mut labels = ChainLabels::default();
    for &u in chain {
        let Some((s, rhs)) = label_store(program, u) else {
            return Err(format!("chain vertex {u} is not a local store"));
        };
        match &ops[u] {
            Op::StoreFolded(_) => {
                let value = labels.eval(rhs).map_err(|m| format!("vertex {u} {m}"))?;
                labels.env.insert(s, value);
            }
            _ => {
                if !matches!(rhs, Expr::Var(VarSlot::Local(src, _)) if *src == s) {
                    labels.env.insert(s, ChainValue::Unknown);
                }
            }
        }
    }
    let Some(ChainValue::Text(pieces)) = labels.env.remove(target) else {
        return Err(format!("the chain does not build a string in slot {target}"));
    };
    let checks: Vec<u32> = pieces.iter().filter_map(|(_, c)| *c).collect();
    if checks != (0..labels.checks).collect::<Vec<_>>() {
        return Err(format!("checks in text order {checks:?} are not the {} checks in evaluation order", labels.checks));
    }
    if labels.trees == 0 {
        return Err("the chain folds no tree".to_string());
    }
    Ok((pieces.into_iter().map(|(part, _)| part).collect(), labels.trees))
}

fn local_piece(part: &PrintPart) -> Option<u32> {
    match part {
        PrintPart::Str { src: PieceSource::Local(s), .. } | PrintPart::Int(PieceSource::Local(s)) => Some(*s),
        _ => None,
    }
}

/// Checks every folded print against the labels of its chain: the run of
/// skipped and folded stores before it in which every vertex after the
/// first has one way in, counting graph edges and `entries`, starting at the
/// first folded store. Poisons the folded slots after the print, fails a
/// folded store no print's chain holds, and returns each local slot a print
/// reads with the chain head and the print, for checking at the head.
fn check_folded_prints(
    program: &Program,
    ops: &[Op],
    entries: &[usize],
    steps: &mut [Step],
    failures: &mut Vec<String>,
) -> Vec<(usize, usize, u32)> {
    let n = ops.len();
    let mut preds = vec![0u32; n];
    let mut edge_from = vec![None; n];
    for (v, s) in steps.iter().enumerate() {
        for e in &s.edges {
            if e.target < n {
                preds[e.target] += 1;
                edge_from[e.target] = Some(v);
            }
        }
    }
    for &e in entries {
        if e < n {
            preds[e] += 1;
        }
    }
    let mut in_chain = vec![false; n];
    let mut piece_reads = Vec::new();
    for p in 0..n {
        let Op::PrintParts { parts, trees_folded, .. } = &ops[p] else { continue };
        let mut run = Vec::new();
        let mut cur = p;
        while preds[cur] == 1 {
            let Some(u) = edge_from[cur] else { break };
            if !matches!(ops[u], Op::StoreFolded(_) | Op::StoreSkipped(_)) {
                break;
            }
            run.push(u);
            cur = u;
        }
        run.reverse();
        let Some(first) = run.iter().position(|&u| matches!(ops[u], Op::StoreFolded(_))) else {
            failures.push(format!("vertex {p}: folded print with no folded store reaching it alone"));
            continue;
        };
        let chain = &run[first..];
        let head = chain[0];
        let mut written = Vec::new();
        for &u in chain {
            let slot = label_store(program, u).map(|(s, _)| s);
            written.extend(slot);
            if matches!(ops[u], Op::StoreFolded(_)) {
                in_chain[u] = true;
                steps[p].edges[0].differs.extend(slot);
            }
        }
        match chain_pieces(program, ops, chain, p) {
            Ok((pieces, trees)) => {
                if pieces[..] != parts[..] {
                    failures.push(format!(
                        "vertex {p}: folded print pieces {parts:?} differ from the pieces its labels build {pieces:?}"
                    ));
                }
                if trees != *trees_folded {
                    failures.push(format!(
                        "vertex {p}: folded print counts {trees_folded} trees, its labels hold {trees}"
                    ));
                }
            }
            Err(m) => failures.push(format!("vertex {p}: folded print chain from {head}: {m}")),
        }
        for s in parts.iter().filter_map(local_piece) {
            if written.contains(&s) {
                failures.push(format!("vertex {p}: piece slot {s} is written between chain head {head} and the print"));
            }
            piece_reads.push((head, p, s));
        }
    }
    for (v, op) in ops.iter().enumerate() {
        if matches!(op, Op::StoreFolded(_)) && !in_chain[v] {
            failures.push(format!("vertex {v}: folded store in no folded print's chain"));
        }
    }
    piece_reads
}

/// Every function entry of `program`.
pub(crate) fn entries(program: &Program) -> Vec<usize> {
    let mut out: Vec<usize> = program.rpc.values().map(|f| f.entry).collect();
    out.sort_unstable();
    out
}
