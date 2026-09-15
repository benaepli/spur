//! A form of the control-flow graph decoded once per program for the
//! simulator: one operation per vertex, with every expression child
//! classified as a slot, a literal or a subtree. The labels and expressions
//! it is built from stay the program's reference form; this form must
//! execute each vertex exactly as its label does.

use super::ir::{Expr, FunctionInfo, Instr, Label, Lhs, Program, VarSlot, Vertex};
use crate::analysis::resolver::NameId;
use crate::analysis::type_id::TypeId;
use crate::simulator::{StructShape, struct_shape};
use ecow::EcoString;

#[cfg(test)]
pub(crate) mod check;
#[cfg(test)]
mod test;

/// A child expression position. Slots and literals are read in place by the
/// parent; only `Tree` enters the tree evaluator.
#[derive(Debug, Clone, PartialEq)]
pub enum Opnd {
    Local(u32),
    Node(u32),
    Int(i64),
    Str(EcoString),
    Bool(bool),
    Unit,
    Nil,
    Tree(Box<CExpr>),
}

/// An expression whose children are operand positions. Each variant
/// evaluates like the `Expr` variant of the same name, in the same order,
/// with the same errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CExpr {
    Find(Opnd, Opnd),
    /// `Find` whose key is a string literal.
    FieldGet(Opnd, EcoString),
    Not(Opnd),
    And(Opnd, Opnd),
    Or(Opnd, Opnd),
    EqualsEquals(Opnd, Opnd),
    /// `Not(EqualsEquals(a, b))`.
    NotEquals(Opnd, Opnd),
    Map(Vec<(Opnd, Opnd)>),
    /// `Map` whose keys are the distinct string literals of a struct shape:
    /// each value with the field position it fills, in source order.
    StructLit(&'static StructShape, Vec<(usize, Opnd)>),
    List(Vec<Opnd>),
    ListPrepend(Opnd, Opnd),
    ListAppend(Opnd, Opnd),
    ListSubsequence(Opnd, Opnd, Opnd),
    LessThan(Opnd, Opnd),
    LessThanEquals(Opnd, Opnd),
    GreaterThan(Opnd, Opnd),
    GreaterThanEquals(Opnd, Opnd),
    KeyExists(Opnd, Opnd),
    MapErase(Opnd, Opnd),
    Store(Opnd, Opnd, Opnd),
    ListLen(Opnd),
    ListAccess(Opnd, usize),
    Plus(Opnd, Opnd),
    Minus(Opnd, Opnd),
    Times(Opnd, Opnd),
    Div(Opnd, Opnd),
    Mod(Opnd, Opnd),
    Min(Opnd, Opnd),
    IndexOf(Opnd, Opnd),
    Tuple(Vec<Opnd>),
    TupleAccess(Opnd, usize),
    Unwrap(Opnd),
    Coalesce(Opnd, Opnd),
    Some(Opnd),
    IntToString(Opnd),
    BoolToString(Opnd),
    NodeToString(Opnd),
    Variant(u32, EcoString, Option<Opnd>),
    IsVariant(Opnd, EcoString),
    VariantPayload(Opnd),
    SafeFind(Opnd, Opnd),
    SafeTupleAccess(Opnd, usize),
}

/// Where a label stores its result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dest {
    Local(u32),
    Node(u32),
}

/// Callee index of a call whose name did not resolve when the program was
/// decoded; such a call resolves by name when it runs.
pub const UNRESOLVED_CALLEE: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct SyncCallOp {
    pub dest: Dest,
    pub callee: u32,
    pub name: String,
    pub args: Vec<Opnd>,
    pub next: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsyncOp {
    pub dest: Dest,
    pub target: Opnd,
    pub callee: u32,
    pub name: String,
    pub args: Vec<Opnd>,
    pub next: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForLoopInOp {
    pub dest: Dest,
    pub collection: Opnd,
    /// `None` when the iterator state slot is a node slot, which is an error
    /// when the label runs.
    pub iter_slot: Option<u32>,
    pub body: u32,
    pub next: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceDispatchOp {
    pub func_name: &'static str,
    pub params: Vec<Opnd>,
    pub next: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceEnterOp {
    pub func_name: &'static str,
    pub params: Vec<Opnd>,
    pub dest: Dest,
    pub next: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceExitOp {
    pub func_name: &'static str,
    pub trace_id: Opnd,
    pub return_value: Opnd,
    pub next: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SendOp {
    pub chan: Opnd,
    pub value: Opnd,
    pub next: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecvOp {
    /// Kept whole because a blocked reader carries it until a value arrives.
    pub lhs: Lhs,
    pub dest: Dest,
    pub chan: Opnd,
    pub next: u32,
}

/// One vertex decoded. Vertex ids are the label's own targets.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Spawn(Box<SpawnOp>),
    Provide(Box<ProvideOp>),
    ProvideAll(Box<ProvideOp>),
    AssignLocal { slot: u32, next: u32, rhs: Opnd },
    AssignNode { slot: u32, next: u32, rhs: Opnd },
    SyncCall(Box<SyncCallOp>),
    Async(Box<AsyncOp>),
    MakeChannel { dest: Dest, next: u32 },
    MakeFifoLink { dest: Dest, next: u32, peer: Opnd },
    SetTimer { dest: Dest, next: u32, label: Option<String> },
    UniqueId { dest: Dest, next: u32 },
    CondLocal { slot: u32, then: u32, els: u32 },
    CondNode { slot: u32, then: u32, els: u32 },
    Cond { cond: Opnd, then: u32, els: u32 },
    Return(Opnd),
    Print { value: Opnd, next: u32 },
    Goto(u32),
    PersistData { type_id: TypeId, next: u32, value: Opnd },
    RetrieveData { type_id: TypeId, dest: Dest, next: u32 },
    DiscardData(u32),
    ForLoopIn(Box<ForLoopInOp>),
    TraceDispatch(Box<TraceDispatchOp>),
    TraceEnter(Box<TraceEnterOp>),
    TraceExit(Box<TraceExitOp>),
    Send(Box<SendOp>),
    Recv(Box<RecvOp>),
    Pause(u32),
    SpinAwait { cond: Opnd, next: u32 },
    /// A local store whose value no later read can see: a slot copied onto
    /// itself, or a slot or literal stored where every path writes the slot
    /// again before reading it. Moves to its successor like the store would.
    StoreSkipped(u32),
    /// A local store whose string value is written by the print its chain
    /// ends in, and whose slot no path reads after that print. Moves to its
    /// successor like the store would.
    StoreFolded(u32),
    /// A print of a local slot whose string the stores before it build,
    /// written piece by piece. Checks each slot piece in order, raising the
    /// error the first failing store would, then writes the same text the
    /// print would. `trees_folded` counts the expression trees of the folded
    /// stores.
    PrintParts { parts: Box<[PrintPart]>, trees_folded: u32, next: u32 },
}

/// A slot read by a piece of a folded print.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PieceSource {
    Local(u32),
    Node(u32),
}

/// One piece of a folded print, in text order.
#[derive(Debug, Clone, PartialEq)]
pub enum PrintPart {
    Lit(EcoString),
    /// A slot that must hold a string. A failed check reports the slot's own
    /// type when `own_type`, and "string" otherwise, as the concatenation it
    /// stands for does.
    Str { src: PieceSource, own_type: bool },
    /// A slot that must hold an integer, written in decimal.
    Int(PieceSource),
}

/// Which decode-time rewrites a build applies. Every rewrite keeps vertex
/// ids, transitions and every value a later read sees; only values no read
/// can see, and when handles are dropped, may differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rewrites {
    /// Each vertex decodes exactly as its label executes.
    Off,
    On,
}

/// The decoded form of a whole program.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompiledProgram {
    /// One operation per vertex of `cfg.graph`, in vertex order.
    pub ops: Vec<Op>,
    /// Callees of the call labels, in first-seen vertex order, each equal to
    /// the program's entry for that name.
    pub call_functions: Vec<FunctionInfo>,
}

impl CompiledProgram {
    /// Decodes every vertex of `program` with every rewrite applied. Reads
    /// the program's labels, call maps and roles, which must not change
    /// afterwards.
    pub fn build(program: &Program) -> Self {
        Self::build_with(program, Rewrites::On)
    }

    /// Decodes every vertex of `program`, applying `rewrites`. Execution may
    /// start only at a function entry of `program.rpc` or at a successor of
    /// an executed vertex.
    pub fn build_with(program: &Program, rewrites: Rewrites) -> Self {
        let liveness = match rewrites {
            Rewrites::Off => None,
            Rewrites::On => Some(super::frame_layout::local_liveness(&program.cfg.graph)),
        };
        let mut builder = Builder {
            program,
            call_functions: Vec::new(),
            callee_index: std::collections::HashMap::new(),
        };
        let mut ops: Vec<Op> = program
            .cfg
            .graph
            .iter()
            .enumerate()
            .map(|(v, label)| match &liveness {
                Some(liveness) if store_is_unread(label, v, liveness) => {
                    let Label::Instr(_, next) = label else {
                        unreachable!("only a store is skipped")
                    };
                    Op::StoreSkipped(vertex(*next))
                }
                _ => builder.op(label),
            })
            .collect();
        if let Some(liveness) = &liveness {
            fold_print_chains(program, &mut ops, liveness);
        }
        CompiledProgram {
            ops,
            call_functions: builder.call_functions,
        }
    }

    /// Whether this form was built for a graph of this size. A program
    /// assembled without decoding runs its labels directly.
    #[inline]
    pub fn covers(&self, program_len: usize) -> bool {
        !self.ops.is_empty() && self.ops.len() == program_len
    }
}

struct Builder<'a> {
    program: &'a Program,
    call_functions: Vec<FunctionInfo>,
    callee_index: std::collections::HashMap<NameId, u32>,
}

fn vertex(v: Vertex) -> u32 {
    u32::try_from(v).expect("a vertex id fits in 32 bits")
}

fn dest(lhs: &Lhs) -> Dest {
    match lhs {
        Lhs::Var(VarSlot::Local(idx, _)) => Dest::Local(*idx),
        Lhs::Var(VarSlot::Node(idx, _)) => Dest::Node(*idx),
    }
}

/// Whether `label` at vertex `v` stores into a local slot a value no later
/// read can see: the slot's own value, or a slot or literal where the slot is
/// not live afterwards. A store whose value comes from a subtree still runs,
/// since evaluating the subtree can fail.
fn store_is_unread(label: &Label, v: usize, liveness: &super::frame_layout::LocalLiveness) -> bool {
    let Label::Instr(Instr::Assign(lhs, rhs) | Instr::Copy(lhs, rhs), _) = label else {
        return false;
    };
    let Dest::Local(slot) = dest(lhs) else {
        return false;
    };
    match rhs {
        Expr::Var(VarSlot::Local(src, _)) if *src == slot => true,
        Expr::Var(_) | Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Unit | Expr::Nil => {
            liveness.dead_after(v, slot)
        }
        _ => false,
    }
}

/// The local slot a store label writes.
fn local_store_slot(label: &Label) -> Option<u32> {
    match label {
        Label::Instr(Instr::Assign(lhs, _) | Instr::Copy(lhs, _), _) => match dest(lhs) {
            Dest::Local(slot) => Some(slot),
            Dest::Node(_) => None,
        },
        _ => None,
    }
}

/// Per vertex, the number of ways execution can arrive: one per graph edge,
/// including a spin-await running itself again, and one per function entry;
/// with the source of the last edge seen.
fn predecessors(program: &Program) -> (Vec<u32>, Vec<Option<usize>>) {
    let graph = &program.cfg.graph;
    let n = graph.len();
    let mut count = vec![0u32; n];
    let mut edge_from = vec![None; n];
    for (v, label) in graph.iter().enumerate() {
        let rerun = matches!(label, Label::SpinAwait(..)).then_some(v);
        for t in super::frame_layout::successors(label).into_iter().flatten().chain(rerun) {
            if t < n {
                count[t] += 1;
                edge_from[t] = Some(v);
            }
        }
    }
    for info in program.rpc.values() {
        if info.entry < n {
            count[info.entry] += 1;
        }
    }
    (count, edge_from)
}

/// The most pieces a folded print may hold. A chain that concatenates a
/// string with itself doubles its pieces per store.
const MAX_PRINT_PARTS: usize = 256;

/// A value a chain of stores computes, as far as decoding can tell.
#[derive(Clone)]
enum Sym {
    /// The value of a slot as it was before the chain.
    Copy(PieceSource),
    /// A string: its pieces in text order, each slot piece with the index of
    /// its check among the checks the stores make, in evaluation order.
    Text(Vec<(PrintPart, Option<u32>)>),
}

#[derive(Default)]
struct ChainFold {
    env: std::collections::HashMap<u32, Sym>,
    checks: u32,
    trees: u32,
}

impl ChainFold {
    fn check(&mut self) -> u32 {
        self.checks += 1;
        self.checks - 1
    }

    /// The value of `o`, evaluating trees operand by operand, left before
    /// right, and each check after its operands, as the tree evaluator does.
    /// `None` when the value is not a string built from literals, slots and
    /// integer slots by concatenation.
    fn eval(&mut self, o: &Opnd) -> Option<Sym> {
        match o {
            Opnd::Local(i) => Some(
                self.env
                    .get(i)
                    .cloned()
                    .unwrap_or(Sym::Copy(PieceSource::Local(*i))),
            ),
            Opnd::Node(i) => Some(Sym::Copy(PieceSource::Node(*i))),
            Opnd::Str(s) => Some(Sym::Text(vec![(PrintPart::Lit(s.clone()), None)])),
            Opnd::Tree(e) => {
                self.trees += 1;
                match &**e {
                    CExpr::IntToString(a) => match self.eval(a)? {
                        Sym::Copy(src) => Some(Sym::Text(vec![(PrintPart::Int(src), Some(self.check()))])),
                        Sym::Text(_) => None,
                    },
                    CExpr::Plus(a, b) => {
                        let a = self.eval(a)?;
                        let b = self.eval(b)?;
                        let out = match (a, b) {
                            (Sym::Copy(_), Sym::Copy(_)) => return None,
                            (Sym::Copy(src), Sym::Text(rest)) => {
                                let mut out = vec![(PrintPart::Str { src, own_type: true }, Some(self.check()))];
                                out.extend(rest);
                                out
                            }
                            (Sym::Text(mut out), Sym::Copy(src)) => {
                                out.push((PrintPart::Str { src, own_type: false }, Some(self.check())));
                                out
                            }
                            (Sym::Text(mut out), Sym::Text(rest)) => {
                                out.extend(rest);
                                out
                            }
                        };
                        (out.len() <= MAX_PRINT_PARTS).then_some(Sym::Text(out))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Folds the stores `suffix` into the print of local slot `target` at `p`,
/// returning the pieces and the trees folded. Refused unless every store is
/// a concatenation of literals, slots and integer slots; every folded slot
/// and the target are dead after the print; the target is such a string
/// and some store holds a tree; every check the stores make appears once, in
/// evaluation order, as the pieces are ordered; and no label of the suffix
/// writes a slot a piece reads.
fn fold_suffix(
    program: &Program,
    ops: &[Op],
    suffix: &[usize],
    p: usize,
    target: u32,
    liveness: &super::frame_layout::LocalLiveness,
) -> Option<(Vec<PrintPart>, u32)> {
    let mut fold = ChainFold::default();
    let mut written = Vec::new();
    for &u in suffix {
        match &ops[u] {
            Op::AssignLocal { slot, rhs, .. } => {
                if !liveness.dead_after(p, *slot) {
                    return None;
                }
                let sym = fold.eval(rhs)?;
                fold.env.insert(*slot, sym);
                written.push(*slot);
            }
            Op::StoreSkipped(_) => written.extend(local_store_slot(&program.cfg.graph[u])),
            _ => return None,
        }
    }
    if !liveness.dead_after(p, target) || fold.trees == 0 {
        return None;
    }
    let Some(Sym::Text(pieces)) = fold.env.remove(&target) else {
        return None;
    };
    let mut next_check = 0;
    for (part, check) in &pieces {
        if let Some(k) = check {
            if *k != next_check {
                return None;
            }
            next_check += 1;
        }
        if let PrintPart::Str { src: PieceSource::Local(s), .. } | PrintPart::Int(PieceSource::Local(s)) = part {
            if written.contains(s) {
                return None;
            }
        }
    }
    if next_check != fold.checks {
        return None;
    }
    Some((pieces.into_iter().map(|(part, _)| part).collect(), fold.trees))
}

/// Rewrites each print of a local slot whose string a chain of stores builds
/// into a folded print, and the chain's stores into folded stores. The chain
/// is the longest run of stores ending at the print in which every vertex
/// after the first, the print included, is reached only from the vertex
/// before it, so entering anywhere past the first store is impossible.
fn fold_print_chains(program: &Program, ops: &mut [Op], liveness: &super::frame_layout::LocalLiveness) {
    let (preds, edge_from) = predecessors(program);
    for p in 0..ops.len() {
        let Op::Print { value: Opnd::Local(target), next } = &ops[p] else {
            continue;
        };
        let (target, next) = (*target, *next);
        let mut chain = Vec::new();
        let mut cur = p;
        while preds[cur] == 1 {
            let Some(u) = edge_from[cur] else { break };
            if !matches!(ops[u], Op::AssignLocal { .. } | Op::StoreSkipped(_)) {
                break;
            }
            chain.push(u);
            cur = u;
        }
        chain.reverse();
        for start in 0..chain.len() {
            if !matches!(ops[chain[start]], Op::AssignLocal { .. }) {
                continue;
            }
            let suffix = &chain[start..];
            let Some((parts, trees_folded)) = fold_suffix(program, ops, suffix, p, target, liveness) else {
                continue;
            };
            for &u in suffix {
                if let Op::AssignLocal { next, .. } = ops[u] {
                    ops[u] = Op::StoreFolded(next);
                }
            }
            ops[p] = Op::PrintParts {
                parts: parts.into_boxed_slice(),
                trees_folded,
                next,
            };
            break;
        }
    }
}

impl Builder<'_> {
    fn callee(&mut self, name: &str) -> u32 {
        let Some(id) = self.program.func_name_to_id.get(name) else {
            return UNRESOLVED_CALLEE;
        };
        let Some(info) = self.program.rpc.get(id) else {
            return UNRESOLVED_CALLEE;
        };
        if let Some(&idx) = self.callee_index.get(id) {
            return idx;
        }
        let idx = u32::try_from(self.call_functions.len()).expect("callee count fits in 32 bits");
        self.call_functions.push(info.clone());
        self.callee_index.insert(*id, idx);
        idx
    }

    fn op(&mut self, label: &Label) -> Op {
        match label {
            Label::Instr(instr, next) => {
                let next = vertex(*next);
                match instr {
                    Instr::Assign(lhs, rhs) | Instr::Copy(lhs, rhs) => match dest(lhs) {
                        Dest::Local(slot) => Op::AssignLocal {
                            slot,
                            next,
                            rhs: opnd(rhs),
                        },
                        Dest::Node(slot) => Op::AssignNode {
                            slot,
                            next,
                            rhs: opnd(rhs),
                        },
                    },
                    Instr::SyncCall(lhs, name, args) => Op::SyncCall(Box::new(SyncCallOp {
                        dest: dest(lhs),
                        callee: self.callee(name),
                        name: name.clone(),
                        args: args.iter().map(opnd).collect(),
                        next,
                    })),
                    Instr::Async(lhs, target, name, args) => Op::Async(Box::new(AsyncOp {
                        dest: dest(lhs),
                        target: opnd(target),
                        callee: self.callee(name),
                        name: name.clone(),
                        args: args.iter().map(opnd).collect(),
                        next,
                    })),
                }
            }
            Label::Pause(next) => Op::Pause(vertex(*next)),
            Label::MakeChannel(lhs, _, next) => Op::MakeChannel {
                dest: dest(lhs),
                next: vertex(*next),
            },
            Label::SetTimer(lhs, next, label) => Op::SetTimer {
                dest: dest(lhs),
                next: vertex(*next),
                label: label.clone(),
            },
            Label::MakeFifoLink(lhs, peer, next) => Op::MakeFifoLink {
                dest: dest(lhs),
                next: vertex(*next),
                peer: opnd(peer),
            },
            Label::Spawn(role, count, lhs, next, span) => Op::Spawn(Box::new(SpawnOp { role: *role, count: opnd(count), dest: dest(lhs), next: vertex(*next), span: *span })),
            Label::Provide(handle, value, next, span) => Op::Provide(Box::new(ProvideOp { handle: opnd(handle), value: opnd(value), next: vertex(*next), span: *span })),
            Label::ProvideAll(handle, value, next, span) => Op::ProvideAll(Box::new(ProvideOp { handle: opnd(handle), value: opnd(value), next: vertex(*next), span: *span })),
            Label::UniqueId(lhs, next) => Op::UniqueId {
                dest: dest(lhs),
                next: vertex(*next),
            },
            Label::Send(chan, value, next) => Op::Send(Box::new(SendOp {
                chan: opnd(chan),
                value: opnd(value),
                next: vertex(*next),
            })),
            Label::Recv(lhs, chan, next) => Op::Recv(Box::new(RecvOp {
                lhs: lhs.clone(),
                dest: dest(lhs),
                chan: opnd(chan),
                next: vertex(*next),
            })),
            Label::SpinAwait(cond, next) => Op::SpinAwait {
                cond: opnd(cond),
                next: vertex(*next),
            },
            Label::Return(expr) => Op::Return(opnd(expr)),
            Label::Cond(cond, then, els) => {
                let (then, els) = (vertex(*then), vertex(*els));
                match cond {
                    Expr::Var(VarSlot::Local(slot, _)) => Op::CondLocal {
                        slot: *slot,
                        then,
                        els,
                    },
                    Expr::Var(VarSlot::Node(slot, _)) => Op::CondNode {
                        slot: *slot,
                        then,
                        els,
                    },
                    _ => Op::Cond {
                        cond: opnd(cond),
                        then,
                        els,
                    },
                }
            }
            Label::ForLoopIn(lhs, collection, iter_slot, body, next) => {
                Op::ForLoopIn(Box::new(ForLoopInOp {
                    dest: dest(lhs),
                    collection: opnd(collection),
                    iter_slot: match iter_slot {
                        VarSlot::Local(idx, _) => Some(*idx),
                        VarSlot::Node(_, _) => None,
                    },
                    body: vertex(*body),
                    next: vertex(*next),
                }))
            }
            Label::Print(expr, next) => Op::Print {
                value: opnd(expr),
                next: vertex(*next),
            },
            Label::PersistData(type_id, expr, next) => Op::PersistData {
                type_id: *type_id,
                next: vertex(*next),
                value: opnd(expr),
            },
            Label::RetrieveData(type_id, lhs, next) => Op::RetrieveData {
                type_id: *type_id,
                dest: dest(lhs),
                next: vertex(*next),
            },
            Label::DiscardData(next) => Op::DiscardData(vertex(*next)),
            Label::Break(target) | Label::Continue(target) => Op::Goto(vertex(*target)),
            Label::TraceDispatch(func_name, params, next) => {
                Op::TraceDispatch(Box::new(TraceDispatchOp {
                    func_name: *func_name,
                    params: params.iter().map(opnd).collect(),
                    next: vertex(*next),
                }))
            }
            Label::TraceEnter(func_name, params, lhs, next) => {
                Op::TraceEnter(Box::new(TraceEnterOp {
                    func_name: *func_name,
                    params: params.iter().map(opnd).collect(),
                    dest: dest(lhs),
                    next: vertex(*next),
                }))
            }
            Label::TraceExit(func_name, trace_id, return_value, next) => {
                Op::TraceExit(Box::new(TraceExitOp {
                    func_name: *func_name,
                    trace_id: opnd(trace_id),
                    return_value: opnd(return_value),
                    next: vertex(*next),
                }))
            }
        }
    }
}

/// Classifies one expression position.
pub fn opnd(expr: &Expr) -> Opnd {
    match expr {
        Expr::Var(VarSlot::Local(idx, _)) => Opnd::Local(*idx),
        Expr::Var(VarSlot::Node(idx, _)) => Opnd::Node(*idx),
        Expr::Int(i) => Opnd::Int(*i),
        Expr::Bool(b) => Opnd::Bool(*b),
        Expr::String(s) => Opnd::Str(s.clone()),
        Expr::Unit => Opnd::Unit,
        Expr::Nil => Opnd::Nil,
        _ => Opnd::Tree(Box::new(tree(expr))),
    }
}

fn o(expr: &Expr) -> Opnd {
    opnd(expr)
}

/// A map literal whose keys are all string literals and form a struct shape.
fn struct_literal(kv: &[(Expr, Expr)]) -> Option<CExpr> {
    let names = kv
        .iter()
        .map(|(k, _)| match k {
            Expr::String(s) => Some(s.clone()),
            _ => None,
        })
        .collect::<Option<Vec<EcoString>>>()?;
    let shape = struct_shape(&names)?;
    let fields = kv
        .iter()
        .zip(&names)
        .map(|((_, v), name)| (shape.position(name).expect("a shape holds its own keys"), o(v)))
        .collect();
    Some(CExpr::StructLit(shape, fields))
}

/// Decodes an expression that is not a slot or a literal.
fn tree(expr: &Expr) -> CExpr {
    match expr {
        Expr::Var(_) | Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Unit | Expr::Nil => {
            unreachable!("slots and literals are operand positions, not trees")
        }
        Expr::Find(col, key) => match &**key {
            Expr::String(s) => CExpr::FieldGet(o(col), s.clone()),
            _ => CExpr::Find(o(col), o(key)),
        },
        Expr::Not(inner) => match &**inner {
            Expr::EqualsEquals(a, b) => CExpr::NotEquals(o(a), o(b)),
            _ => CExpr::Not(o(inner)),
        },
        Expr::And(a, b) => CExpr::And(o(a), o(b)),
        Expr::Or(a, b) => CExpr::Or(o(a), o(b)),
        Expr::EqualsEquals(a, b) => CExpr::EqualsEquals(o(a), o(b)),
        Expr::Map(kv) => struct_literal(kv)
            .unwrap_or_else(|| CExpr::Map(kv.iter().map(|(k, v)| (o(k), o(v))).collect())),
        Expr::List(es) => CExpr::List(es.iter().map(o).collect()),
        Expr::ListPrepend(a, b) => CExpr::ListPrepend(o(a), o(b)),
        Expr::ListAppend(a, b) => CExpr::ListAppend(o(a), o(b)),
        Expr::ListSubsequence(a, b, c) => CExpr::ListSubsequence(o(a), o(b), o(c)),
        Expr::LessThan(a, b) => CExpr::LessThan(o(a), o(b)),
        Expr::LessThanEquals(a, b) => CExpr::LessThanEquals(o(a), o(b)),
        Expr::GreaterThan(a, b) => CExpr::GreaterThan(o(a), o(b)),
        Expr::GreaterThanEquals(a, b) => CExpr::GreaterThanEquals(o(a), o(b)),
        Expr::KeyExists(a, b) => CExpr::KeyExists(o(a), o(b)),
        Expr::MapErase(a, b) => CExpr::MapErase(o(a), o(b)),
        Expr::Store(a, b, c) => CExpr::Store(o(a), o(b), o(c)),
        Expr::ListLen(a) => CExpr::ListLen(o(a)),
        Expr::ListAccess(a, i) => CExpr::ListAccess(o(a), *i),
        Expr::Plus(a, b) => CExpr::Plus(o(a), o(b)),
        Expr::Minus(a, b) => CExpr::Minus(o(a), o(b)),
        Expr::Times(a, b) => CExpr::Times(o(a), o(b)),
        Expr::Div(a, b) => CExpr::Div(o(a), o(b)),
        Expr::Mod(a, b) => CExpr::Mod(o(a), o(b)),
        Expr::IndexOf(a, b) => CExpr::IndexOf(o(a), o(b)),
        Expr::Min(a, b) => CExpr::Min(o(a), o(b)),
        Expr::Tuple(es) => CExpr::Tuple(es.iter().map(o).collect()),
        Expr::TupleAccess(a, i) => CExpr::TupleAccess(o(a), *i),
        Expr::Unwrap(a) => CExpr::Unwrap(o(a)),
        Expr::Coalesce(a, b) => CExpr::Coalesce(o(a), o(b)),
        Expr::Some(a) => CExpr::Some(o(a)),
        Expr::IntToString(a) => CExpr::IntToString(o(a)),
        Expr::BoolToString(a) => CExpr::BoolToString(o(a)),
        Expr::NodeToString(a) => CExpr::NodeToString(o(a)),
        Expr::Variant(enum_id, name, payload) => {
            CExpr::Variant(*enum_id, name.clone(), payload.as_deref().map(o))
        }
        Expr::IsVariant(a, name) => CExpr::IsVariant(o(a), name.clone()),
        Expr::VariantPayload(a) => CExpr::VariantPayload(o(a)),
        Expr::SafeFind(a, b) => CExpr::SafeFind(o(a), o(b)),
        Expr::SafeTupleAccess(a, i) => CExpr::SafeTupleAccess(o(a), *i),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpawnOp {
    pub role: NameId,
    pub count: Opnd,
    pub dest: Dest,
    pub next: u32,
    pub span: crate::parser::Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProvideOp {
    pub handle: Opnd,
    pub value: Opnd,
    pub next: u32,
    pub span: crate::parser::Span,
}
