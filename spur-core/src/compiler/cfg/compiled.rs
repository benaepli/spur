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
}

/// The decoded form of a whole program.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompiledProgram {
    /// One operation per vertex of `cfg.graph`, in vertex order.
    pub ops: Vec<Op>,
    /// Callees of the call labels, in first-seen vertex order, each equal to
    /// the program's entry for that name.
    pub call_functions: Vec<FunctionInfo>,
    /// The role named "Node", if the program declares one.
    pub server_role: Option<NameId>,
}

impl CompiledProgram {
    /// Decodes every vertex of `program`. Reads the program's labels, call
    /// maps and roles, which must not change afterwards.
    pub fn build(program: &Program) -> Self {
        let mut builder = Builder {
            program,
            call_functions: Vec::new(),
            callee_index: std::collections::HashMap::new(),
        };
        let ops = program
            .cfg
            .graph
            .iter()
            .map(|label| builder.op(label))
            .collect();
        CompiledProgram {
            ops,
            call_functions: builder.call_functions,
            server_role: program
                .roles
                .iter()
                .find(|(_, n)| n == "Node")
                .map(|(id, _)| *id),
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
