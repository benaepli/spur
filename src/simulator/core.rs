use crate::compiler::cfg::{Expr, Instr, Label, Lhs, Program, Vertex};
use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    hash::{Hash, Hasher},
    rc::Rc,
};
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum RuntimeError {
    #[error("type error: expected {expected}, got {got}")]
    TypeError {
        expected: &'static str,
        got: &'static str,
    },

    #[error("variable not found: {0}")]
    VariableNotFound(String),

    #[error("function not found: {0}")]
    FunctionNotFound(String),

    #[error("index out of bounds: index {index}, length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("map key not found")]
    KeyNotFound,

    #[error("tuple length mismatch: expected {expected}, got {got}")]
    TupleLengthMismatch { expected: usize, got: usize },

    #[error("cannot assign non-tuple to tuple LHS, got {got}")]
    NonTupleAssignment { got: &'static str },

    #[error("unwrap called on None")]
    UnwrapNone,

    #[error("coalesce called on non-option, got {got}")]
    CoalesceNonOption { got: &'static str },

    #[error("sync function tried to call non-sync function: {0}")]
    SyncCallToAsyncFunction(String),

    #[error("instruction not allowed in sync function: {0}")]
    UnsupportedSyncInstruction(String),

    #[error("for-loop collection must be a variable")]
    ForLoopCollectionNotVariable,

    #[error("for-loop over map expects tuple LHS")]
    ForLoopMapExpectsTupleLhs,

    #[error("for-loop requires collection, got {got}")]
    ForLoopNotCollection { got: &'static str },

    #[error("attempted to unlock an already-unlocked lock")]
    UnlockAlreadyUnlocked,

    #[error("cannot read remote channel directly")]
    RemoteChannelRead,

    #[error("networked channels are unsupported")]
    NetworkedChannelUnsupported,

    #[error("cannot index into non-collection, got {got}")]
    NotACollection { got: &'static str },

    #[error("list subsequence out of bounds: start {start}, end {end}, length {len}")]
    SubsequenceOutOfBounds { start: usize, end: usize, len: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId {
    pub node: usize,
    pub id: usize,
}

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Map(BTreeMap<Value, Value>),
    List(Vec<Value>),
    Option(Option<Box<Value>>),
    Channel(ChannelId),
    Lock(Rc<RefCell<bool>>),
    Node(usize),
    String(String),
    Unit,
    Tuple(Vec<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Node(a), Value::Node(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            (Value::Option(a), Value::Option(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => a == b,
            (Value::Channel(a), Value::Channel(b)) => a == b,
            (Value::Lock(a), Value::Lock(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}
impl Eq for Value {}
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        use Value::*;
        match (self, other) {
            (Int(a), Int(b)) => a.cmp(b),
            (Bool(a), Bool(b)) => a.cmp(b),
            (String(a), String(b)) => a.cmp(b),
            (Node(a), Node(b)) => a.cmp(b),
            (Unit, Unit) => Ordering::Equal,
            (Option(a), Option(b)) => a.cmp(b),
            (Tuple(a), Tuple(b)) => a.cmp(b),
            (List(a), List(b)) => a.cmp(b),
            (Map(a), Map(b)) => a.cmp(b),
            (Channel(a), Channel(b)) => a.cmp(b),
            (Lock(a), Lock(b)) => {
                if Rc::ptr_eq(a, b) {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            } // Arbitrary
            // Cross-type comparisons (simple deterministic ordering)
            (Int(_), _) => Ordering::Less,
            (_, Int(_)) => Ordering::Greater,
            (Bool(_), _) => Ordering::Less,
            (_, Bool(_)) => Ordering::Greater,
            (String(_), _) => Ordering::Less,
            (_, String(_)) => Ordering::Greater,
            (Node(_), _) => Ordering::Less,
            (_, Node(_)) => Ordering::Greater,
            (Unit, _) => Ordering::Less,
            (_, Unit) => Ordering::Greater,
            (Option(_), _) => Ordering::Less,
            (_, Option(_)) => Ordering::Greater,
            (Tuple(_), _) => Ordering::Less,
            (_, Tuple(_)) => Ordering::Greater,
            (List(_), _) => Ordering::Less,
            (_, List(_)) => Ordering::Greater,
            (Map(_), _) => Ordering::Less,
            (_, Map(_)) => Ordering::Greater,
            (Channel(_), _) => Ordering::Less,
            (_, Channel(_)) => Ordering::Greater,
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Int(i) => i.hash(state),
            Value::Bool(b) => b.hash(state),
            Value::String(s) => s.hash(state),
            Value::Node(n) => n.hash(state),
            Value::Unit => 0.hash(state),
            Value::Option(o) => o.hash(state),
            Value::Tuple(v) => v.hash(state),
            Value::List(v) => v.hash(state),
            Value::Map(m) => m.hash(state),
            Value::Channel(c) => c.hash(state),
            Value::Lock(l) => (Rc::as_ptr(l) as usize).hash(state),
        }
    }
}

impl Value {
    fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Bool(_) => "bool",
            Value::Map(_) => "map",
            Value::List(_) => "list",
            Value::Option(_) => "option",
            Value::Channel(_) => "channel",
            Value::Lock(_) => "lock",
            Value::Node(_) => "node",
            Value::String(_) => "string",
            Value::Unit => "unit",
            Value::Tuple(_) => "tuple",
        }
    }

    fn as_int(&self) -> Result<i64, RuntimeError> {
        if let Value::Int(i) = self {
            Ok(*i)
        } else {
            Err(RuntimeError::TypeError {
                expected: "int",
                got: self.type_name(),
            })
        }
    }
    fn as_bool(&self) -> Result<bool, RuntimeError> {
        if let Value::Bool(b) = self {
            Ok(*b)
        } else {
            Err(RuntimeError::TypeError {
                expected: "bool",
                got: self.type_name(),
            })
        }
    }
    fn as_node(&self) -> Result<usize, RuntimeError> {
        match self {
            Value::Node(n) => Ok(*n),
            Value::Int(n) => Ok(*n as usize),
            _ => Err(RuntimeError::TypeError {
                expected: "node",
                got: self.type_name(),
            }),
        }
    }
    fn as_map(&self) -> Result<&BTreeMap<Value, Value>, RuntimeError> {
        if let Value::Map(m) = self {
            Ok(m)
        } else {
            Err(RuntimeError::TypeError {
                expected: "map",
                got: self.type_name(),
            })
        }
    }
    fn as_list(&self) -> Result<&Vec<Value>, RuntimeError> {
        if let Value::List(l) = self {
            Ok(l)
        } else {
            Err(RuntimeError::TypeError {
                expected: "list",
                got: self.type_name(),
            })
        }
    }
    fn as_channel(&self) -> Result<ChannelId, RuntimeError> {
        if let Value::Channel(c) = self {
            Ok(*c)
        } else {
            Err(RuntimeError::TypeError {
                expected: "channel",
                got: self.type_name(),
            })
        }
    }
    fn as_lock(&self) -> Result<&Rc<RefCell<bool>>, RuntimeError> {
        if let Value::Lock(l) = self {
            Ok(l)
        } else {
            Err(RuntimeError::TypeError {
                expected: "lock",
                got: self.type_name(),
            })
        }
    }
}

#[derive(Debug)]
pub struct Record {
    pub pc: Vertex,
    pub node: usize,
    pub origin_node: usize,
    pub continuation: Rc<dyn Continuation>,
    pub env: Env, // Just local env, node env is in State
    pub id: i32,
    pub x: f64,
    pub policy: UpdatePolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UpdatePolicy {
    Identity,
    Halve,
}

impl UpdatePolicy {
    pub fn update(&self, x: f64) -> f64 {
        match self {
            UpdatePolicy::Identity => x,
            UpdatePolicy::Halve => x / 2.0,
        }
    }
}

pub struct CrashInfo {
    pub currently_crashed: HashSet<usize>,
    pub queued_messages: Vec<(usize, Record)>, // (dest_node, record)
    pub current_step: i32,
}

#[derive(Debug)]
pub struct ChannelState {
    pub capacity: i32,
    pub buffer: VecDeque<Value>,
    // We move Record out of Runnable and into Waiting.
    pub waiting_readers: VecDeque<(Record, Lhs)>,
    pub waiting_writers: VecDeque<(Record, Value)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    Invocation,
    Response,
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub client_id: i32,
    pub op_action: String,
    pub kind: OpKind,
    pub payload: Vec<Value>,
    pub unique_id: i32,
}

pub struct State {
    pub nodes: Vec<Rc<RefCell<Env>>>, // Index is node_id
    pub runnable_records: Vec<Record>,
    pub waiting_records: Vec<Record>, // Generic waiting (timeouts etc, logic simplified here)
    pub channels: HashMap<ChannelId, ChannelState>,
    pub history: Vec<Operation>,
    pub free_clients: Vec<i32>,
    pub crash_info: CrashInfo,
}

impl State {
    pub fn new(node_count: usize) -> Self {
        Self {
            nodes: (0..node_count)
                .map(|_| Rc::new(RefCell::new(HashMap::new())))
                .collect(),
            runnable_records: Vec::new(),
            waiting_records: Vec::new(),
            channels: HashMap::new(),
            history: Vec::new(),
            free_clients: Vec::new(),
            crash_info: CrashInfo {
                currently_crashed: HashSet::new(),
                queued_messages: Vec::new(),
                current_step: 0,
            },
        }
    }
}

pub type Env = HashMap<String, Value>;

pub trait Continuation: Debug {
    fn call(&self, state: &mut State, val: Value);
}

#[derive(Debug)]
pub struct NoOpContinuation;
impl Continuation for NoOpContinuation {
    fn call(&self, _state: &mut State, _val: Value) {}
}

fn load(var: &str, local_env: &Env, node_env: &Env) -> Result<Value, RuntimeError> {
    if let Some(v) = local_env.get(var) {
        Ok(v.clone())
    } else if let Some(v) = node_env.get(var) {
        Ok(v.clone())
    } else {
        Err(RuntimeError::VariableNotFound(var.to_string()))
    }
}

fn store(
    lhs: &Lhs,
    val: Value,
    local_env: &mut Env,
    node_env: &mut Env,
) -> Result<(), RuntimeError> {
    match lhs {
        Lhs::Var(name) => {
            if local_env.contains_key(name) {
                local_env.insert(name.clone(), val);
            } else {
                // Local preference for new variables.
                if node_env.contains_key(name) {
                    node_env.insert(name.clone(), val);
                } else {
                    local_env.insert(name.clone(), val);
                }
            }
            Ok(())
        }
        Lhs::Tuple(vars) => {
            if let Value::Tuple(vals) = val {
                if vars.len() != vals.len() {
                    return Err(RuntimeError::TupleLengthMismatch {
                        expected: vars.len(),
                        got: vals.len(),
                    });
                }
                for (name, v) in vars.iter().zip(vals.into_iter()) {
                    store(&Lhs::Var(name.clone()), v, local_env, node_env)?;
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

fn update_collection(col: Value, key: Value, val: Value) -> Result<Value, RuntimeError> {
    match col {
        Value::Map(mut m) => {
            m.insert(key, val);
            Ok(Value::Map(m))
        }
        Value::List(mut l) => {
            let idx = key.as_int()? as usize;
            if idx >= l.len() {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: idx,
                    len: l.len(),
                });
            }
            l[idx] = val;
            Ok(Value::List(l))
        }
        _ => Err(RuntimeError::NotACollection {
            got: col.type_name(),
        }),
    }
}

pub fn eval(local_env: &Env, node_env: &Env, expr: &Expr) -> Result<Value, RuntimeError> {
    match expr {
        Expr::Int(i) => Ok(Value::Int(*i)),
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::String(s) => Ok(Value::String(s.clone())),
        Expr::Unit => Ok(Value::Unit),
        Expr::Nil => Ok(Value::Option(None)),
        Expr::Var(s) => load(s, local_env, node_env),
        Expr::Plus(e1, e2) => Ok(Value::Int(
            eval(local_env, node_env, e1)?.as_int()? + eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::Minus(e1, e2) => Ok(Value::Int(
            eval(local_env, node_env, e1)?.as_int()? - eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::Times(e1, e2) => Ok(Value::Int(
            eval(local_env, node_env, e1)?.as_int()? * eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::Div(e1, e2) => Ok(Value::Int(
            eval(local_env, node_env, e1)?.as_int()? / eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::Mod(e1, e2) => Ok(Value::Int(
            eval(local_env, node_env, e1)?.as_int()? % eval(local_env, node_env, e2)?.as_int()?,
        )),
        Expr::LessThan(e1, e2) => {
            Ok(Value::Bool(eval(local_env, node_env, e1)? < eval(local_env, node_env, e2)?))
        }
        Expr::EqualsEquals(e1, e2) => {
            Ok(Value::Bool(eval(local_env, node_env, e1)? == eval(local_env, node_env, e2)?))
        }
        Expr::Not(e) => Ok(Value::Bool(!eval(local_env, node_env, e)?.as_bool()?)),
        Expr::And(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)?.as_bool()? && eval(local_env, node_env, e2)?.as_bool()?,
        )),
        Expr::Or(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)?.as_bool()? || eval(local_env, node_env, e2)?.as_bool()?,
        )),
        Expr::Some(e) => Ok(Value::Option(Some(Box::new(eval(local_env, node_env, e)?)))),
        Expr::Tuple(es) => {
            let vals: Result<Vec<_>, _> = es.iter().map(|e| eval(local_env, node_env, e)).collect();
            Ok(Value::Tuple(vals?))
        }
        Expr::List(es) => {
            let vals: Result<Vec<_>, _> = es.iter().map(|e| eval(local_env, node_env, e)).collect();
            Ok(Value::List(vals?))
        }
        Expr::Map(kv) => {
            let mut m = BTreeMap::new();
            for (k, v) in kv {
                m.insert(eval(local_env, node_env, k)?, eval(local_env, node_env, v)?);
            }
            Ok(Value::Map(m))
        }
        Expr::Find(col, key) => match eval(local_env, node_env, col)? {
            Value::Map(m) => {
                let k = eval(local_env, node_env, key)?;
                m.get(&k).cloned().ok_or(RuntimeError::KeyNotFound)
            }
            Value::List(l) => {
                let idx = eval(local_env, node_env, key)?.as_int()? as usize;
                l.get(idx).cloned().ok_or(RuntimeError::IndexOutOfBounds {
                    index: idx,
                    len: l.len(),
                })
            }
            other => Err(RuntimeError::NotACollection {
                got: other.type_name(),
            }),
        },
        Expr::CreateLock => Ok(Value::Lock(Rc::new(RefCell::new(false)))),
        Expr::ListPrepend(head, tail) => {
            let h = eval(local_env, node_env, head)?;
            let mut t = eval(local_env, node_env, tail)?.as_list()?.clone();
            t.insert(0, h);
            Ok(Value::List(t))
        }
        Expr::ListAppend(list, item) => {
            let mut l = eval(local_env, node_env, list)?.as_list()?.clone();
            let i = eval(local_env, node_env, item)?;
            l.push(i);
            Ok(Value::List(l))
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
            Ok(Value::List(vec[s..e].to_vec()))
        }
        Expr::LessThanEquals(e1, e2) => {
            Ok(Value::Bool(eval(local_env, node_env, e1)? <= eval(local_env, node_env, e2)?))
        }
        Expr::GreaterThan(e1, e2) => {
            Ok(Value::Bool(eval(local_env, node_env, e1)? > eval(local_env, node_env, e2)?))
        }
        Expr::GreaterThanEquals(e1, e2) => {
            Ok(Value::Bool(eval(local_env, node_env, e1)? >= eval(local_env, node_env, e2)?))
        }
        Expr::KeyExists(key, map) => {
            let k = eval(local_env, node_env, key)?;
            let m = eval(local_env, node_env, map)?;
            Ok(Value::Bool(m.as_map()?.contains_key(&k)))
        }
        Expr::MapErase(key, map) => {
            let k = eval(local_env, node_env, key)?;
            let mut m = eval(local_env, node_env, map)?.as_map()?.clone();
            m.remove(&k);
            Ok(Value::Map(m))
        }
        Expr::ListLen(list) => match eval(local_env, node_env, list)? {
            Value::List(l) => Ok(Value::Int(l.len() as i64)),
            Value::Map(m) => Ok(Value::Int(m.len() as i64)),
            other => Err(RuntimeError::NotACollection {
                got: other.type_name(),
            }),
        },
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
            Ok(Value::Int(v1.min(v2)))
        }
        Expr::TupleAccess(tuple, idx) => {
            let t = eval(local_env, node_env, tuple)?;
            if let Value::Tuple(vec) = t {
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
        Expr::Unwrap(e) => match eval(local_env, node_env, e)? {
            Value::Option(Some(v)) => Ok(*v),
            Value::Option(None) => Err(RuntimeError::UnwrapNone),
            other => Err(RuntimeError::TypeError {
                expected: "option",
                got: other.type_name(),
            }),
        },
        Expr::Coalesce(opt, default) => match eval(local_env, node_env, opt)? {
            Value::Option(Some(v)) => Ok(*v),
            Value::Option(None) => eval(local_env, node_env, default),
            other => Err(RuntimeError::CoalesceNonOption {
                got: other.type_name(),
            }),
        },
        Expr::IntToString(e) => Ok(Value::String(eval(local_env, node_env, e)?.as_int()?.to_string())),
        Expr::Store(col, key, val) => update_collection(
            eval(local_env, node_env, col)?,
            eval(local_env, node_env, key)?,
            eval(local_env, node_env, val)?,
        ),
    }
}

pub fn exec_sync_on_node(
    state: &mut State,
    program: &Program,
    local_env: &mut Env,
    node_id: usize,
    start_pc: usize,
) -> Result<Value, RuntimeError> {
    let node_env = Rc::clone(&state.nodes[node_id]);
    exec_sync_inner(
        state,
        program,
        local_env,
        &mut node_env.borrow_mut(),
        start_pc,
    )
}

fn exec_sync_inner(
    state: &mut State,
    program: &Program,
    local_env: &mut Env,
    node_env: &mut Env,
    start_pc: usize,
) -> Result<Value, RuntimeError> {
    let mut pc = start_pc;
    loop {
        let label = program.cfg.get_label(pc).clone();
        match label {
            Label::Instr(instr, next) => {
                pc = next;
                match instr {
                    Instr::Assign(lhs, rhs) => {
                        let v = eval(local_env, node_env, &rhs)?;
                        store(&lhs, v, local_env, node_env)?;
                    }
                    Instr::Copy(lhs, rhs) => {
                        let v = eval(local_env, node_env, &rhs)?;
                        store(&lhs, v, local_env, node_env)?;
                    }
                    Instr::SyncCall(lhs, func_name, args) => {
                        let arg_vals: Result<Vec<Value>, _> =
                            args.iter().map(|a| eval(local_env, node_env, a)).collect();
                        let arg_vals = arg_vals?;
                        let func_info = program
                            .rpc
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;

                        if !func_info.is_sync {
                            return Err(RuntimeError::SyncCallToAsyncFunction(func_name.clone()));
                        }

                        let mut callee_local = HashMap::new();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_local.insert(name.clone(), arg_vals[i].clone());
                        }
                        for (name, expr) in &func_info.locals {
                            callee_local.insert(name.clone(), eval(local_env, node_env, expr)?);
                        }

                        if let Some(s) = local_env.get("self") {
                            callee_local.insert("self".to_string(), s.clone());
                        }

                        // Pass node_env directly (it's already mutable borrowed from caller)
                        let val = exec_sync_inner(
                            state,
                            program,
                            &mut callee_local,
                            node_env,
                            func_info.entry,
                        )?;

                        store(&lhs, val, local_env, node_env)?;
                    }
                    Instr::Async(lhs, node_expr, func_name, args) => {
                        // Similar to exec Async but we are inside sync.
                        let target_node = eval(local_env, node_env, &node_expr)?.as_node()?;
                        let arg_vals: Result<Vec<Value>, _> =
                            args.iter().map(|a| eval(local_env, node_env, a)).collect();
                        let arg_vals = arg_vals?;

                        let chan_id = ChannelId {
                            node: target_node,
                            // So Channel attached to target node.
                            id: state.channels.len(),
                        };

                        state.channels.insert(
                            chan_id,
                            ChannelState {
                                capacity: 1,
                                buffer: VecDeque::new(),
                                waiting_readers: VecDeque::new(),
                                waiting_writers: VecDeque::new(),
                            },
                        );
                        store(&lhs, Value::Channel(chan_id), local_env, node_env)?;

                        let func_info = program
                            .rpc
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let mut callee_locals = HashMap::new();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_locals.insert(name.clone(), arg_vals[i].clone());
                        }
                        for (name, expr) in &func_info.locals {
                            callee_locals.insert(name.clone(), eval(local_env, node_env, expr)?);
                        }

                        #[derive(Debug)]
                        struct AsyncContinuation {
                            chan_id: ChannelId,
                        }
                        impl Continuation for AsyncContinuation {
                            fn call(&self, state: &mut State, val: Value) {
                                let chan = state.channels.get_mut(&self.chan_id).unwrap();
                                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                                    let node_env = Rc::clone(&state.nodes[reader.node]);
                                    // Note: store errors in continuations are ignored (fire-and-forget)
                                    let _ =
                                        store(&lhs, val, &mut reader.env, &mut node_env.borrow_mut());
                                    state.runnable_records.push(reader);
                                } else {
                                    chan.buffer.push_back(val);
                                }
                            }
                        }

                        let origin_node = local_env
                            .get("self")
                            .ok_or_else(|| RuntimeError::VariableNotFound("self".to_string()))?
                            .as_node()?;

                        let new_record = Record {
                            pc: func_info.entry,
                            node: target_node,
                            origin_node,
                            continuation: Rc::new(AsyncContinuation { chan_id }),
                            env: callee_locals,
                            id: -1,
                            x: 0.5,
                            policy: UpdatePolicy::Identity,
                        };

                        if state.crash_info.currently_crashed.contains(&target_node) {
                            state
                                .crash_info
                                .queued_messages
                                .push((target_node, new_record));
                        } else {
                            state.runnable_records.push(new_record);
                        }
                    }
                }
            }
            Label::MakeChannel(lhs, cap, next) => {
                let origin_node = local_env
                    .get("self")
                    .ok_or_else(|| RuntimeError::VariableNotFound("self".to_string()))?
                    .as_node()?;
                let id = state.channels.len();
                let cid = ChannelId {
                    node: origin_node,
                    id,
                };
                state.channels.insert(
                    cid,
                    ChannelState {
                        capacity: cap as i32,
                        buffer: VecDeque::new(),
                        waiting_readers: VecDeque::new(),
                        waiting_writers: VecDeque::new(),
                    },
                );
                store(&lhs, Value::Channel(cid), local_env, node_env)?;
                pc = next;
            }
            Label::Cond(cond, bthen, belse) => {
                if eval(local_env, node_env, &cond)?.as_bool()? {
                    pc = bthen;
                } else {
                    pc = belse;
                }
            }
            Label::Return(expr) => {
                return eval(local_env, node_env, &expr);
            }
            Label::Print(expr, next) => {
                // log_app!("{:?}", eval(local_env, node_env, &expr)?);
                let _ = eval(local_env, node_env, &expr)?;
                pc = next;
            }
            Label::Break(target) => {
                pc = target;
            }
            Label::ForLoopIn(lhs, expr, body, next) => {
                // Simplified iteration: we need to handle list/map iteration
                // Since we don't have easy mutable iterator in this loop structure
                // We rely on "eval" fetching the current state of collection
                let col_val = eval(local_env, node_env, &expr)?;

                match col_val {
                    Value::Map(m) => {
                        if m.is_empty() {
                            pc = next;
                        } else {
                            let (k, v) = m.iter().next().unwrap();
                            let k = k.clone();
                            let v = v.clone();
                            let mut new_m = m.clone();
                            new_m.remove(&k);

                            if let Expr::Var(vname) = expr {
                                let updated = Value::Map(new_m);
                                store(&Lhs::Var(vname), updated, local_env, node_env)?;

                                match lhs {
                                    Lhs::Tuple(vars) if vars.len() == 2 => {
                                        store(&Lhs::Var(vars[0].clone()), k, local_env, node_env)?;
                                        store(&Lhs::Var(vars[1].clone()), v, local_env, node_env)?;
                                        pc = body;
                                    }
                                    _ => return Err(RuntimeError::ForLoopMapExpectsTupleLhs),
                                }
                            } else {
                                return Err(RuntimeError::ForLoopCollectionNotVariable);
                            }
                        }
                    }
                    Value::List(l) => {
                        if l.is_empty() {
                            pc = next;
                        } else {
                            let item = l[0].clone();
                            // Update collection
                            if let Expr::Var(vname) = expr {
                                let new_l = Value::List(l[1..].to_vec());
                                store(&Lhs::Var(vname), new_l, local_env, node_env)?;

                                store(&lhs, item, local_env, node_env)?;
                                pc = body;
                            } else {
                                return Err(RuntimeError::ForLoopCollectionNotVariable);
                            }
                        }
                    }
                    other => {
                        return Err(RuntimeError::ForLoopNotCollection {
                            got: other.type_name(),
                        })
                    }
                }
            }
            other => {
                return Err(RuntimeError::UnsupportedSyncInstruction(format!(
                    "{:?}",
                    other
                )))
            }
        }
    }
}

pub fn exec(
    state: &mut State,
    program: &mut Program,
    mut record: Record,
) -> Result<(), RuntimeError> {
    let mut local_env = record.env;
    let node_env = Rc::clone(&state.nodes[record.node]);

    local_env.insert("self".to_string(), Value::Node(record.node));

    loop {
        let label = program.cfg.get_label(record.pc).clone();

        match label {
            Label::Instr(instr, next) => {
                record.pc = next;
                match instr {
                    Instr::Assign(lhs, rhs) => {
                        let v = eval(&local_env, &node_env.borrow(), &rhs)?;
                        store(&lhs, v, &mut local_env, &mut node_env.borrow_mut())?;
                    }
                    Instr::Copy(lhs, rhs) => {
                        let v = eval(&local_env, &node_env.borrow(), &rhs)?;
                        store(&lhs, v, &mut local_env, &mut node_env.borrow_mut())?;
                    }
                    Instr::SyncCall(lhs, func_name, args) => {
                        let arg_vals: Result<Vec<Value>, _> = args
                            .iter()
                            .map(|a| eval(&local_env, &node_env.borrow(), a))
                            .collect();
                        let arg_vals = arg_vals?;
                        let func_info = program
                            .rpc
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;

                        if !func_info.is_sync {
                            return Err(RuntimeError::SyncCallToAsyncFunction(func_name.clone()));
                        }

                        let mut callee_local = HashMap::new();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_local.insert(name.clone(), arg_vals[i].clone());
                        }
                        for (name, expr) in &func_info.locals {
                            callee_local
                                .insert(name.clone(), eval(&local_env, &node_env.borrow(), expr)?);
                        }

                        if let Some(s) = local_env.get("self") {
                            callee_local.insert("self".to_string(), s.clone());
                        }

                        // Pass node_env directly
                        let ret_val = exec_sync_inner(
                            state,
                            program,
                            &mut callee_local,
                            &mut node_env.borrow_mut(),
                            func_info.entry,
                        )?;

                        store(&lhs, ret_val, &mut local_env, &mut node_env.borrow_mut())?;
                    }
                    Instr::Async(lhs, node_expr, func_name, args) => {
                        // Async logic
                        let target_node =
                            eval(&local_env, &node_env.borrow(), &node_expr)?.as_node()?;
                        let arg_vals: Result<Vec<Value>, _> = args
                            .iter()
                            .map(|a| eval(&local_env, &node_env.borrow(), a))
                            .collect();
                        let arg_vals = arg_vals?;

                        // Make channel
                        let chan_id = ChannelId {
                            node: record.node,
                            id: state.channels.len(),
                        };
                        state.channels.insert(
                            chan_id,
                            ChannelState {
                                capacity: 1,
                                buffer: VecDeque::new(),
                                waiting_readers: VecDeque::new(),
                                waiting_writers: VecDeque::new(),
                            },
                        );
                        store(
                            &lhs,
                            Value::Channel(chan_id),
                            &mut local_env,
                            &mut node_env.borrow_mut(),
                        )?;

                        // Setup Callee Record
                        let func_info = program
                            .rpc
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let mut callee_locals = HashMap::new();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_locals.insert(name.clone(), arg_vals[i].clone());
                        }
                        // Default locals would go here

                        // Continuation for the async call
                        #[derive(Debug)]
                        struct AsyncContinuation {
                            chan_id: ChannelId,
                        }
                        impl Continuation for AsyncContinuation {
                            fn call(&self, state: &mut State, val: Value) {
                                let chan = state.channels.get_mut(&self.chan_id).unwrap();
                                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                                    // Direct handoff
                                    let node_env = Rc::clone(&state.nodes[reader.node]);
                                    // Note: store errors in continuations are ignored (fire-and-forget)
                                    let _ =
                                        store(&lhs, val, &mut reader.env, &mut node_env.borrow_mut());
                                    state.runnable_records.push(reader);
                                } else {
                                    chan.buffer.push_back(val);
                                }
                            }
                        }

                        let new_record = Record {
                            pc: func_info.entry,
                            node: target_node,
                            origin_node: record.node,
                            continuation: Rc::new(AsyncContinuation { chan_id }),
                            env: callee_locals,
                            id: -1,
                            x: 0.5,
                            policy: UpdatePolicy::Identity,
                        };

                        if state.crash_info.currently_crashed.contains(&target_node) {
                            state
                                .crash_info
                                .queued_messages
                                .push((target_node, new_record));
                        } else {
                            state.runnable_records.push(new_record);
                        }
                    }
                }
            }
            Label::MakeChannel(lhs, cap, next) => {
                let id = state.channels.len();
                let cid = ChannelId {
                    node: record.node,
                    id,
                };
                state.channels.insert(
                    cid,
                    ChannelState {
                        capacity: cap as i32,
                        buffer: VecDeque::new(),
                        waiting_readers: VecDeque::new(),
                        waiting_writers: VecDeque::new(),
                    },
                );
                store(
                    &lhs,
                    Value::Channel(cid),
                    &mut local_env,
                    &mut node_env.borrow_mut(),
                )?;
                record.pc = next;
            }
            Label::Send(chan_expr, val_expr, next) => {
                let cid = eval(&local_env, &node_env.borrow(), &chan_expr)?.as_channel()?;
                let val = eval(&local_env, &node_env.borrow(), &val_expr)?;
                if cid.node != record.node {
                    return Err(RuntimeError::NetworkedChannelUnsupported);
                }
                // Local Send
                let chan = state.channels.get_mut(&cid).unwrap();
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    // Wakeup reader
                    let r_node = Rc::clone(&state.nodes[reader.node]);
                    store(&lhs, val, &mut reader.env, &mut r_node.borrow_mut())?;
                    state.runnable_records.push(reader);
                    record.pc = next;
                } else if (chan.buffer.len() as i32) < chan.capacity {
                    chan.buffer.push_back(val);
                    record.pc = next;
                } else {
                    record.env = local_env;
                    record.pc = next;
                    chan.waiting_writers.push_back((record, val));
                    return Ok(());
                }
            }
            Label::Recv(lhs, chan_expr, next) => {
                let cid = eval(&local_env, &node_env.borrow(), &chan_expr)?.as_channel()?;
                if cid.node != record.node {
                    return Err(RuntimeError::RemoteChannelRead);
                }

                let chan = state.channels.get_mut(&cid).unwrap();

                if let Some(val) = chan.buffer.pop_front() {
                    store(&lhs, val, &mut local_env, &mut node_env.borrow_mut())?;
                    // Wake writer if any
                    if let Some((writer, w_val)) = chan.waiting_writers.pop_front() {
                        chan.buffer.push_back(w_val);
                        state.runnable_records.push(writer);
                    }
                    record.pc = next;
                } else if let Some((writer, w_val)) = chan.waiting_writers.pop_front() {
                    store(&lhs, w_val, &mut local_env, &mut node_env.borrow_mut())?;
                    state.runnable_records.push(writer);
                    record.pc = next;
                } else {
                    // Block Reader
                    record.env = local_env;
                    record.pc = next; // When woke, proceed to next
                    chan.waiting_readers.push_back((record, lhs));
                    return Ok(()); // Stop execution
                }
            }
            Label::Return(expr) => {
                let val = eval(&local_env, &node_env.borrow(), &expr)?;
                record.env = local_env;
                // Call continuation
                record.continuation.call(state, val);
                return Ok(()); // End thread
            }
            Label::Pause(next) => {
                record.env = local_env;
                record.pc = next;
                state.runnable_records.push(record);
                return Ok(()); // Yield
            }
            Label::Cond(cond, bthen, belse) => {
                if eval(&local_env, &node_env.borrow(), &cond)?.as_bool()? {
                    record.pc = bthen;
                } else {
                    record.pc = belse;
                }
            }
            Label::Print(expr, next) => {
                // log_app!("{:?}", eval(&local_env, &node_env.borrow(), &expr)?);
                let _ = eval(&local_env, &node_env.borrow(), &expr)?;
                record.pc = next;
            }
            Label::SpinAwait(expr, next) => {
                if eval(&local_env, &node_env.borrow(), &expr)?.as_bool()? {
                    record.pc = next;
                } else {
                    record.env = local_env;
                    state.runnable_records.push(record);
                    return Ok(()); // Yield
                }
            }
            Label::ForLoopIn(lhs, expr, body_pc, next_pc) => {
                let col_val = eval(&local_env, &node_env.borrow(), &expr)?;
                match col_val {
                    Value::Map(m) => {
                        if m.is_empty() {
                            record.pc = next_pc;
                        } else {
                            let (k, v) = m.iter().next().unwrap();
                            let k = k.clone();
                            let v = v.clone();
                            let mut new_m = m.clone();
                            new_m.remove(&k);

                            if let Expr::Var(vname) = expr {
                                let updated = Value::Map(new_m);
                                store(
                                    &Lhs::Var(vname),
                                    updated,
                                    &mut local_env,
                                    &mut node_env.borrow_mut(),
                                )?;

                                match lhs {
                                    Lhs::Tuple(vars) if vars.len() == 2 => {
                                        store(
                                            &Lhs::Var(vars[0].clone()),
                                            k,
                                            &mut local_env,
                                            &mut node_env.borrow_mut(),
                                        )?;
                                        store(
                                            &Lhs::Var(vars[1].clone()),
                                            v,
                                            &mut local_env,
                                            &mut node_env.borrow_mut(),
                                        )?;
                                        record.pc = body_pc;
                                    }
                                    _ => return Err(RuntimeError::ForLoopMapExpectsTupleLhs),
                                }
                            } else {
                                return Err(RuntimeError::ForLoopCollectionNotVariable);
                            }
                        }
                    }
                    Value::List(l) => {
                        if l.is_empty() {
                            record.pc = next_pc;
                        } else {
                            let item = l[0].clone();
                            if let Expr::Var(vname) = expr {
                                let new_l = Value::List(l[1..].to_vec());
                                store(
                                    &Lhs::Var(vname),
                                    new_l,
                                    &mut local_env,
                                    &mut node_env.borrow_mut(),
                                )?;

                                store(&lhs, item, &mut local_env, &mut node_env.borrow_mut())?;
                                record.pc = body_pc;
                            } else {
                                return Err(RuntimeError::ForLoopCollectionNotVariable);
                            }
                        }
                    }
                    other => {
                        return Err(RuntimeError::ForLoopNotCollection {
                            got: other.type_name(),
                        })
                    }
                }
            }
            Label::Lock(lock_expr, next) => {
                let lock_val = eval(&local_env, &node_env.borrow(), &lock_expr)?;
                let lock = lock_val.as_lock()?;
                if *lock.borrow() == false {
                    *lock.borrow_mut() = true;
                    record.pc = next;
                } else {
                    record.env = local_env;
                    state.runnable_records.push(record);
                    return Ok(()); // Yield
                }
            }
            Label::Unlock(lock_expr, next) => {
                let lock_val = eval(&local_env, &node_env.borrow(), &lock_expr)?;
                let lock = lock_val.as_lock()?;
                if *lock.borrow() == true {
                    *lock.borrow_mut() = false;
                    record.pc = next;
                } else {
                    return Err(RuntimeError::UnlockAlreadyUnlocked);
                }
            }
            Label::Break(target) => {
                record.pc = target;
            }
        }
    }
}

pub fn schedule_record(
    state: &mut State,
    program: &mut Program,
    randomly_drop_msgs: bool,
    cut_tail_from_mid: bool,
    sever_all_but_mid: bool,
    partition_away_nodes: &[usize],
    randomly_delay_msgs: bool,
) -> Result<(), RuntimeError> {
    let len = state.runnable_records.len();
    if len == 0 {
        return Ok(()); // Halt equivalent
    }

    use rand::Rng;
    let mut rng = rand::rng();
    let idx = rng.random_range(0..len);

    let mut r = state.runnable_records.swap_remove(idx);

    if state.crash_info.currently_crashed.contains(&r.node) {
        // log_err!("Failure: source {} for dst {}", r.origin_node, r.node);
        return Ok(());
    }

    let src_node = r.origin_node;
    let dest_node = r.node;

    if src_node != dest_node {
        if state.crash_info.currently_crashed.contains(&dest_node) {
            // Queue message
            state.crash_info.queued_messages.push((dest_node, r));
            return Ok(());
        } else {
            let mut should_execute = true;
            if randomly_drop_msgs && rng.random::<f64>() < 0.3 {
                should_execute = false;
            }

            if cut_tail_from_mid
                && ((src_node == 2 && dest_node == 1) || (dest_node == 2 && src_node == 1))
            {
                should_execute = false;
            }

            if sever_all_but_mid {
                if dest_node == 2 && src_node != 1 {
                    should_execute = false;
                } else if src_node == 2 && dest_node != 1 {
                    should_execute = false;
                }
            }

            if partition_away_nodes.contains(&src_node) || partition_away_nodes.contains(&dest_node)
            {
                should_execute = false;
            }

            if randomly_delay_msgs {
                if rng.random::<f64>() < r.x {
                    r.x = r.policy.update(r.x);
                    should_execute = false;
                    state.runnable_records.push(r);
                    return Ok(());
                }
            }

            if should_execute {
                exec(state, program, r)?;
            }
        }
    } else {
        exec(state, program, r)?;
    }
    Ok(())
}
