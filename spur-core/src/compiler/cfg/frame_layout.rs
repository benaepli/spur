//! Local slot layout of one function. Slots whose values are never needed at
//! the same time share one frame position, so a call frame holds fewer
//! values without any read seeing a different value.
//!
//! A slot is live at a point when some path from that point reads it before
//! writing it. Two slots may share a position only when neither is ever
//! written while the other is live, and not both are live when the frame is
//! built. Parameters keep their positions because callers fill them by
//! position, and a slot read before any write keeps its declared starting
//! value.

use super::ir::{Expr, FunctionInfo, Instr, Label, Lhs, SlotDefault, VarSlot, Vertex};

/// How a label touches one local slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Access {
    Read,
    /// Written before the label moves to any successor.
    Write,
    /// Written only on the way to the label's first successor.
    WriteFirstEdge,
}

fn visit_lhs(lhs: &mut Lhs, access: Access, f: &mut impl FnMut(Access, &mut u32)) {
    match lhs {
        Lhs::Var(slot) => visit_slot(slot, access, f),
    }
}

fn visit_slot(slot: &mut VarSlot, access: Access, f: &mut impl FnMut(Access, &mut u32)) {
    match slot {
        VarSlot::Local(idx, _) => f(access, idx),
        VarSlot::Node(_, _) => {}
    }
}

fn visit_exprs(exprs: &mut [Expr], f: &mut impl FnMut(Access, &mut u32)) {
    for e in exprs {
        visit_expr(e, f);
    }
}

/// Every local slot an expression names is a read, including those behind a
/// short-circuit, which makes the reads a superset of what evaluation touches.
fn visit_expr(expr: &mut Expr, f: &mut impl FnMut(Access, &mut u32)) {
    match expr {
        Expr::Var(slot) => visit_slot(slot, Access::Read, f),
        Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Unit | Expr::Nil => {}
        Expr::Not(a)
        | Expr::ListLen(a)
        | Expr::ListAccess(a, _)
        | Expr::TupleAccess(a, _)
        | Expr::Unwrap(a)
        | Expr::Some(a)
        | Expr::IntToString(a)
        | Expr::BoolToString(a)
        | Expr::NodeToString(a)
        | Expr::IsVariant(a, _)
        | Expr::VariantPayload(a)
        | Expr::SafeTupleAccess(a, _) => visit_expr(a, f),
        Expr::Find(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::EqualsEquals(a, b)
        | Expr::ListPrepend(a, b)
        | Expr::ListAppend(a, b)
        | Expr::LessThan(a, b)
        | Expr::LessThanEquals(a, b)
        | Expr::GreaterThan(a, b)
        | Expr::GreaterThanEquals(a, b)
        | Expr::KeyExists(a, b)
        | Expr::MapErase(a, b)
        | Expr::Plus(a, b)
        | Expr::Minus(a, b)
        | Expr::Times(a, b)
        | Expr::Div(a, b)
        | Expr::Mod(a, b)
        | Expr::IndexOf(a, b)
        | Expr::Min(a, b)
        | Expr::Coalesce(a, b)
        | Expr::SafeFind(a, b) => {
            visit_expr(a, f);
            visit_expr(b, f);
        }
        Expr::ListSubsequence(a, b, c) | Expr::Store(a, b, c) => {
            visit_expr(a, f);
            visit_expr(b, f);
            visit_expr(c, f);
        }
        Expr::Map(pairs) => {
            for (k, v) in pairs {
                visit_expr(k, f);
                visit_expr(v, f);
            }
        }
        Expr::List(items) | Expr::Tuple(items) => visit_exprs(items, f),
        Expr::Variant(_, _, payload) => {
            if let Some(p) = payload {
                visit_expr(p, f);
            }
        }
    }
}

/// Calls `f` for every local slot the label touches, in the order its
/// execution touches them.
fn visit_label(label: &mut Label, f: &mut impl FnMut(Access, &mut u32)) {
    match label {
        Label::Instr(instr, _) => match instr {
            Instr::Assign(lhs, rhs) | Instr::Copy(lhs, rhs) => {
                visit_expr(rhs, f);
                visit_lhs(lhs, Access::Write, f);
            }
            Instr::Async(lhs, target, _, args) => {
                visit_expr(target, f);
                visit_exprs(args, f);
                visit_lhs(lhs, Access::Write, f);
            }
            Instr::SyncCall(lhs, _, args) => {
                visit_exprs(args, f);
                visit_lhs(lhs, Access::Write, f);
            }
        },
        Label::Pause(_) => {}
        Label::MakeChannel(lhs, _, _) => visit_lhs(lhs, Access::Write, f),
        Label::SetTimer(lhs, _, _) => visit_lhs(lhs, Access::Write, f),
        Label::MakeFifoLink(lhs, peer, _) => {
            visit_expr(peer, f);
            visit_lhs(lhs, Access::Write, f);
        }
        Label::Spawn(_, count, lhs, _, _) => { visit_expr(count, f); visit_lhs(lhs, Access::Write, f); }
        Label::Provide(handle, value, _, _) | Label::ProvideAll(handle, value, _, _) => { visit_expr(handle, f); visit_expr(value, f); }
        Label::UniqueId(lhs, _) => visit_lhs(lhs, Access::Write, f),
        Label::Send(chan, value, _) => {
            visit_expr(chan, f);
            visit_expr(value, f);
        }
        // A blocked receive stores into the same slot when it wakes, before
        // moving on.
        Label::Recv(lhs, chan, _) => {
            visit_expr(chan, f);
            visit_lhs(lhs, Access::Write, f);
        }
        Label::SpinAwait(cond, _) => visit_expr(cond, f),
        Label::Return(e) => visit_expr(e, f),
        Label::Cond(cond, _, _) => visit_expr(cond, f),
        // The iterator slot is read, then written on both exits (the rest of
        // the collection toward the body, Unit toward the exit); the binding
        // is written only toward the body.
        Label::ForLoopIn(lhs, collection, iter_slot, _, _) => {
            visit_slot(iter_slot, Access::Read, f);
            visit_expr(collection, f);
            visit_slot(iter_slot, Access::Write, f);
            visit_lhs(lhs, Access::WriteFirstEdge, f);
        }
        Label::Print(e, _) => visit_expr(e, f),
        Label::PersistData(_, e, _) => visit_expr(e, f),
        Label::RetrieveData(_, lhs, _) => visit_lhs(lhs, Access::Write, f),
        Label::DiscardData(_) => {}
        Label::Break(_) => {}
        Label::Continue(_) => {}
        Label::TraceDispatch(_, params, _) => visit_exprs(params, f),
        // The trace id is stored before the parameters are formatted.
        Label::TraceEnter(_, params, lhs, _) => {
            visit_lhs(lhs, Access::Write, f);
            visit_exprs(params, f);
        }
        Label::TraceExit(_, trace_id, return_value, _) => {
            visit_expr(trace_id, f);
            visit_expr(return_value, f);
        }
    }
}

/// The label's successors, the first being the one `WriteFirstEdge` applies
/// to. A spin-await that re-runs itself writes nothing, so its self edge
/// changes no liveness and is left out.
pub(super) fn successors(label: &Label) -> [Option<Vertex>; 2] {
    match label {
        Label::Instr(_, n)
        | Label::Pause(n)
        | Label::MakeChannel(_, _, n)
        | Label::SetTimer(_, n, _)
        | Label::MakeFifoLink(_, _, n)
        | Label::Spawn(_, _, _, n, _)
        | Label::Provide(_, _, n, _)
        | Label::ProvideAll(_, _, n, _)
        | Label::UniqueId(_, n)
        | Label::Send(_, _, n)
        | Label::Recv(_, _, n)
        | Label::SpinAwait(_, n)
        | Label::Print(_, n)
        | Label::PersistData(_, _, n)
        | Label::RetrieveData(_, _, n)
        | Label::DiscardData(n)
        | Label::Break(n)
        | Label::Continue(n)
        | Label::TraceDispatch(_, _, n)
        | Label::TraceEnter(_, _, _, n)
        | Label::TraceExit(_, _, _, n) => [Some(*n), None],
        Label::Return(_) => [None, None],
        Label::Cond(_, then, els) => [Some(*then), Some(*els)],
        Label::ForLoopIn(_, _, _, body, next) => [Some(*body), Some(*next)],
    }
}

/// Fixed-width bit rows over the function's slots.
struct Bits {
    words: usize,
    data: Vec<u64>,
}

impl Bits {
    fn new(rows: usize, bits: usize) -> Self {
        let words = bits.div_ceil(64).max(1);
        Self {
            words,
            data: vec![0; rows * words],
        }
    }

    fn row(&self, r: usize) -> &[u64] {
        &self.data[r * self.words..(r + 1) * self.words]
    }

    fn set(&mut self, r: usize, bit: usize) {
        self.data[r * self.words + bit / 64] |= 1 << (bit % 64);
    }

    fn get(&self, r: usize, bit: usize) -> bool {
        self.data[r * self.words + bit / 64] & (1 << (bit % 64)) != 0
    }
}

fn each_bit(row: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in row.iter().enumerate() {
        let mut rest = word;
        while rest != 0 {
            let b = rest.trailing_zeros() as usize;
            f(w * 64 + b);
            rest &= rest - 1;
        }
    }
}

/// The value slot `slot` starts with when the frame is built.
fn declared_default(info: &FunctionInfo, slot: usize) -> SlotDefault {
    slot.checked_sub(info.param_count as usize)
        .and_then(|i| info.local_defaults.get(i))
        .copied()
        .unwrap_or(SlotDefault::Unit)
}

/// Renumbers the local slots of the function whose labels are
/// `graph[first..]`, rewriting every label in that range and the function's
/// slot count, starting values and slot names. The function is left as it is
/// when a label names a slot or a successor outside the function.
pub(super) fn compact_function_slots(graph: &mut [Label], first: Vertex, info: &mut FunctionInfo) {
    let n = info.local_slot_count as usize;
    let params = (info.param_count as usize).min(n);
    if first > graph.len() {
        return;
    }
    let labels = &mut graph[first..];
    let m = labels.len();
    if n == 0 || info.entry < first || info.entry >= first + m {
        return;
    }
    let entry = info.entry - first;

    let mut accesses: Vec<Vec<(Access, usize)>> = Vec::with_capacity(m);
    let mut succ: Vec<[Option<usize>; 2]> = Vec::with_capacity(m);
    let mut reads = Bits::new(m, n);
    let mut kill_all = Bits::new(m, n);
    let mut kill_first = Bits::new(m, n);
    for (v, label) in labels.iter_mut().enumerate() {
        let mut touched = Vec::new();
        let mut in_range = true;
        visit_label(label, &mut |access, idx| {
            let idx = *idx as usize;
            if idx >= n {
                in_range = false;
                return;
            }
            touched.push((access, idx));
        });
        if !in_range {
            return;
        }
        for &(access, idx) in &touched {
            match access {
                Access::Read => reads.set(v, idx),
                Access::Write => kill_all.set(v, idx),
                Access::WriteFirstEdge => kill_first.set(v, idx),
            }
        }
        let mut edges = [None, None];
        for (e, target) in successors(label).into_iter().enumerate() {
            if let Some(t) = target {
                if t < first || t >= first + m {
                    return;
                }
                edges[e] = Some(t - first);
            }
        }
        accesses.push(touched);
        succ.push(edges);
    }
    for p in 0..params {
        reads.set(entry, p);
    }

    let words = reads.words;
    let mut live_in = Bits::new(m, n);
    let mut scratch = vec![0u64; words];
    let mut changed = true;
    while changed {
        changed = false;
        for v in (0..m).rev() {
            scratch.copy_from_slice(reads.row(v));
            for (e, target) in succ[v].iter().enumerate() {
                let Some(t) = *target else { continue };
                for w in 0..words {
                    let mut kill = kill_all.row(v)[w];
                    if e == 0 {
                        kill |= kill_first.row(v)[w];
                    }
                    scratch[w] |= live_in.row(t)[w] & !kill;
                }
            }
            let row = &mut live_in.data[v * words..(v + 1) * words];
            if row != scratch.as_slice() {
                row.copy_from_slice(&scratch);
                changed = true;
            }
        }
    }

    // A slot written at a label interferes with every slot live after the
    // label on the edges the write reaches, and with every slot the label
    // reads after the write.
    let mut interferes = Bits::new(n, n);
    let mut against = vec![0u64; words];
    let mut read_after = vec![0u64; words];
    for v in 0..m {
        read_after.iter_mut().for_each(|w| *w = 0);
        let mut written: Vec<usize> = Vec::new();
        for &(access, idx) in accesses[v].iter().rev() {
            match access {
                Access::Read => read_after[idx / 64] |= 1 << (idx % 64),
                Access::Write | Access::WriteFirstEdge => {
                    against.copy_from_slice(&read_after);
                    for (e, target) in succ[v].iter().enumerate() {
                        let Some(t) = *target else { continue };
                        if e == 0 || access == Access::Write {
                            for w in 0..words {
                                against[w] |= live_in.row(t)[w];
                            }
                        }
                    }
                    each_bit(&against, |x| {
                        if x != idx {
                            interferes.set(idx, x);
                            interferes.set(x, idx);
                        }
                    });
                    written.push(idx);
                }
            }
        }
        for &a in &written {
            for &b in &written {
                if a != b {
                    interferes.set(a, b);
                }
            }
        }
    }
    let entry_live: Vec<usize> = {
        let mut out = Vec::new();
        each_bit(live_in.row(entry), |x| out.push(x));
        out
    };
    for &a in &entry_live {
        for &b in &entry_live {
            if a != b {
                interferes.set(a, b);
            }
        }
    }

    let mut color = vec![usize::MAX; n];
    for (p, c) in color.iter_mut().enumerate().take(params) {
        *c = p;
    }
    let mut taken = Vec::new();
    for s in params..n {
        taken.clear();
        each_bit(interferes.row(s), |x| {
            if color[x] != usize::MAX {
                taken.push(color[x]);
            }
        });
        taken.sort_unstable();
        let mut c = 0;
        for &t in &taken {
            if t == c {
                c += 1;
            } else if t > c {
                break;
            }
        }
        color[s] = c;
    }

    let count = color.iter().copied().max().map_or(0, |c| c + 1).max(params);
    let mut defaults: Vec<Option<SlotDefault>> = vec![None; count];
    let mut names: Vec<String> = vec![String::new(); count];
    for s in 0..n {
        let c = color[s];
        if s >= params && (defaults[c].is_none() || live_in.get(entry, s)) {
            defaults[c] = Some(declared_default(info, s));
        }
        if let Some(name) = info.debug_slot_names.get(s) {
            if !names[c].is_empty() {
                names[c].push('|');
            }
            names[c].push_str(name);
        }
    }

    // A label may visit one field twice (the iterator slot of a for-in loop),
    // so each visit takes its new index from the value the field had before
    // the label was rewritten.
    for (label, touched) in labels.iter_mut().zip(&accesses) {
        let mut k = 0;
        visit_label(label, &mut |_, idx| {
            *idx = color[touched[k].1] as u32;
            k += 1;
        });
    }
    info.local_slot_count = count as u32;
    info.local_defaults = defaults[params..]
        .iter()
        .map(|d| d.unwrap_or(SlotDefault::Unit))
        .collect();
    info.debug_slot_names = names;
}

/// Local slot liveness over a whole graph, one bit per slot index below 128.
/// Every successor edge stays inside one function, so a pass over the whole
/// graph equals the union of passes over each function.
pub(super) struct LocalLiveness {
    /// Slots some path from a successor of the vertex reads before writing,
    /// with no write of the vertex itself taken out.
    live_out: Vec<u128>,
}

impl LocalLiveness {
    /// Whether no path after `v` reads `slot` before writing it. A slot at or
    /// above 128 is never reported dead.
    pub(super) fn dead_after(&self, v: usize, slot: u32) -> bool {
        slot < 128
            && self
                .live_out
                .get(v)
                .is_some_and(|out| out & (1u128 << slot) == 0)
    }
}

/// Computes liveness for every vertex of `graph`. A read of a slot at or
/// above 128 marks every slot live, and a write of one kills nothing.
pub(super) fn local_liveness(graph: &[Label]) -> LocalLiveness {
    let m = graph.len();
    let mut reads = vec![0u128; m];
    let mut kill_all = vec![0u128; m];
    let mut kill_first = vec![0u128; m];
    let mut succ: Vec<[Option<Vertex>; 2]> = Vec::with_capacity(m);
    for (v, label) in graph.iter().enumerate() {
        let mut label = label.clone();
        visit_label(&mut label, &mut |access, idx| {
            let bit = if *idx < 128 { 1u128 << *idx } else { 0 };
            match access {
                Access::Read => reads[v] |= if bit == 0 { u128::MAX } else { bit },
                Access::Write => kill_all[v] |= bit,
                Access::WriteFirstEdge => kill_first[v] |= bit,
            }
        });
        succ.push(successors(&label));
    }
    let mut live_in = vec![0u128; m];
    let mut live_out = vec![0u128; m];
    let mut changed = true;
    while changed {
        changed = false;
        for v in (0..m).rev() {
            let mut out = 0u128;
            let mut through = 0u128;
            for (e, target) in succ[v].iter().enumerate() {
                let Some(t) = *target else { continue };
                let after = if t < m { live_in[t] } else { u128::MAX };
                out |= after;
                let kill = if e == 0 { kill_all[v] | kill_first[v] } else { kill_all[v] };
                through |= after & !kill;
            }
            let inn = reads[v] | through;
            if inn != live_in[v] || out != live_out[v] {
                live_in[v] = inn;
                live_out[v] = out;
                changed = true;
            }
        }
    }
    LocalLiveness { live_out }
}

#[cfg(test)]
mod test;
