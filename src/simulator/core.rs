use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Expr, Instr, Label, Lhs, Program, SELF_NAME, Vertex};
use crate::simulator::coverage::{GlobalCoverage, LocalCoverage};
use ecow::EcoString;
use imbl::{HashMap as ImHashMap, Vector};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::rc::Rc;
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
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
    VariableNotFound(NameId),

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
    SubsequenceOutOfBounds {
        start: usize,
        end: usize,
        len: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId {
    pub node: usize,
    pub id: usize,
}

#[derive(Clone, Debug)]
pub struct LogEntry {
    pub node: usize,
    pub content: String,
    pub step: i32,
}

/// Trait for handling Print statement output during execution.
pub trait Logger {
    fn log(&mut self, entry: LogEntry);
}

/// A no-op logger that discards all log entries.
#[derive(Debug, Default)]
pub struct NoOpLogger;

impl Logger for NoOpLogger {
    fn log(&mut self, _entry: LogEntry) {}
}

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Map(ImHashMap<Value, Value>),
    List(Vector<Value>),
    Option(Option<Arc<Value>>),
    Channel(ChannelId),
    Lock(Arc<Mutex<bool>>),
    Node(usize),
    String(EcoString),
    Unit,
    Tuple(Vector<Value>),
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
            (Value::Lock(a), Value::Lock(b)) => Arc::ptr_eq(a, b),
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
            (Map(a), Map(b)) => {
                // Compare maps by converting to sorted vectors
                let mut a_vec: Vec<_> = a.iter().collect();
                let mut b_vec: Vec<_> = b.iter().collect();
                a_vec.sort_by(|x, y| x.0.cmp(y.0));
                b_vec.sort_by(|x, y| x.0.cmp(y.0));
                a_vec.cmp(&b_vec)
            }
            (Channel(a), Channel(b)) => a.cmp(b),
            (Lock(a), Lock(b)) => {
                if Arc::ptr_eq(a, b) {
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
            Value::Map(m) => {
                // Hash map by hashing sorted entries
                let mut entries: Vec<_> = m.iter().collect();
                entries.sort_by(|x, y| x.0.cmp(y.0));
                entries.len().hash(state);
                for (k, v) in entries {
                    k.hash(state);
                    v.hash(state);
                }
            }
            Value::Channel(c) => c.hash(state),
            Value::Lock(l) => (Arc::as_ptr(l) as usize).hash(state),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Node(n) => write!(f, "node({})", n),
            Value::Unit => write!(f, "()"),
            Value::Option(None) => write!(f, "None"),
            Value::Option(Some(v)) => write!(f, "Some({})", v),
            Value::Channel(ch) => write!(f, "channel({}, {})", ch.node, ch.id),
            Value::Lock(_) => write!(f, "<lock>"),
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Map(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
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
        match self {
            Value::Int(i) => Ok(*i),
            _ => Err(RuntimeError::TypeError {
                expected: "int",
                got: self.type_name(),
            }),
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
            _ => Err(RuntimeError::TypeError {
                expected: "node",
                got: self.type_name(),
            }),
        }
    }
    fn as_map(&self) -> Result<&ImHashMap<Value, Value>, RuntimeError> {
        if let Value::Map(m) = self {
            Ok(m)
        } else {
            Err(RuntimeError::TypeError {
                expected: "map",
                got: self.type_name(),
            })
        }
    }
    fn as_list(&self) -> Result<&Vector<Value>, RuntimeError> {
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
    fn as_lock(&self) -> Result<&Arc<Mutex<bool>>, RuntimeError> {
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
    pub continuation: Continuation,
    pub env: Env, // Just local env, node env is in State
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

#[derive(Debug)]
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
    Crash,
    Recover,
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub client_id: i32,
    pub op_action: String,
    pub kind: OpKind,
    pub payload: Vec<Value>,
    pub unique_id: i32,
}

#[derive(Debug)]
pub struct State {
    pub nodes: Vec<Rc<RefCell<Env>>>, // Index is node_id
    pub runnable_records: Vec<Record>,
    pub channels: HashMap<ChannelId, ChannelState>,
    pub crash_info: CrashInfo,
    pub coverage: LocalCoverage,
    next_channel_id: usize,
}

impl State {
    pub fn new(node_count: usize) -> Self {
        Self {
            nodes: (0..node_count)
                .map(|_| Rc::new(RefCell::new(Env::default())))
                .collect(),
            runnable_records: Vec::new(),
            channels: HashMap::new(),
            crash_info: CrashInfo {
                currently_crashed: HashSet::new(),
                queued_messages: Vec::new(),
                current_step: 0,
            },
            coverage: LocalCoverage::new(),
            next_channel_id: 0,
        }
    }

    pub fn alloc_channel_id(&mut self) -> usize {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        id
    }
}

/// Creates a special NameId for for-loop iterator state based on the PC.
/// Uses high bit range to avoid collision with normal IDs.
fn iter_name_id(pc: Vertex) -> NameId {
    // Use a high range that won't collide with normal IDs or SELF_NAME_ID
    NameId(usize::MAX - 1 - pc)
}

pub type Env = FxHashMap<NameId, Value>;

/// Continuation representing what to do when an execution completes.
#[derive(Debug, Clone)]
pub enum Continuation {
    /// No action needed
    NoOp,
    /// Node recovery continuation
    Recover,
    /// Async message delivery continuation
    Async { chan_id: ChannelId },
    /// Client operation completion - returns data for caller to handle
    ClientOp {
        client_id: i32,
        op_name: String,
        unique_id: i32,
    },
}

/// Result returned when a ClientOp continuation completes.
#[derive(Debug, Clone)]
pub struct ClientOpResult {
    pub client_id: i32,
    pub op_name: String,
    pub unique_id: i32,
    pub value: Value,
}

fn load(var: NameId, local_env: &Env, node_env: &Env) -> Result<Value, RuntimeError> {
    if let Some(v) = local_env.get(&var) {
        Ok(v.clone())
    } else if let Some(v) = node_env.get(&var) {
        Ok(v.clone())
    } else {
        Err(RuntimeError::VariableNotFound(var))
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
                local_env.insert(*name, val);
            } else {
                // Node preference for new variables.
                if local_env.contains_key(name) {
                    local_env.insert(*name, val);
                } else {
                    node_env.insert(*name, val);
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
                    store(&Lhs::Var(*name), v, local_env, node_env)?;
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

impl Continuation {
    /// Execute the continuation and return any client operation result.
    pub fn call(self, state: &mut State, val: Value) -> Option<ClientOpResult> {
        match self {
            Continuation::NoOp | Continuation::Recover => None,
            Continuation::Async { chan_id } => {
                let chan = state.channels.get_mut(&chan_id).unwrap();
                if let Some((mut reader, lhs)) = chan.waiting_readers.pop_front() {
                    let node_env: Rc<RefCell<Env>> = Rc::clone(&state.nodes[reader.node]);
                    // Note: store errors in continuations are ignored (fire-and-forget)
                    let _ = store(&lhs, val, &mut reader.env, &mut node_env.borrow_mut());
                    state.runnable_records.push(reader);
                } else {
                    chan.buffer.push_back(val);
                }
                None
            }
            Continuation::ClientOp {
                client_id,
                op_name,
                unique_id,
            } => Some(ClientOpResult {
                client_id,
                op_name,
                unique_id,
                value: val,
            }),
        }
    }
}

fn update_collection(col: Value, key: Value, val: Value) -> Result<Value, RuntimeError> {
    match col {
        Value::Map(m) => Ok(Value::Map(m.update(key, val))),
        Value::List(l) => {
            let idx = key.as_int()? as usize;
            if idx >= l.len() {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: idx,
                    len: l.len(),
                });
            }
            Ok(Value::List(l.update(idx, val)))
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
        Expr::Var(s) => load(*s, local_env, node_env),
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
        Expr::LessThan(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)? < eval(local_env, node_env, e2)?,
        )),
        Expr::EqualsEquals(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)? == eval(local_env, node_env, e2)?,
        )),
        Expr::Not(e) => Ok(Value::Bool(!eval(local_env, node_env, e)?.as_bool()?)),
        Expr::And(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)?.as_bool()?
                && eval(local_env, node_env, e2)?.as_bool()?,
        )),
        Expr::Or(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)?.as_bool()?
                || eval(local_env, node_env, e2)?.as_bool()?,
        )),
        Expr::Some(e) => Ok(Value::Option(Some(Arc::new(eval(local_env, node_env, e)?)))),
        Expr::Tuple(es) => {
            let vals: Result<Vector<_>, _> =
                es.iter().map(|e| eval(local_env, node_env, e)).collect();
            Ok(Value::Tuple(vals?))
        }
        Expr::List(es) => {
            let vals: Result<Vector<_>, _> =
                es.iter().map(|e| eval(local_env, node_env, e)).collect();
            Ok(Value::List(vals?))
        }
        Expr::Map(kv) => {
            let mut m = ImHashMap::new();
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
        Expr::CreateLock => Ok(Value::Lock(Arc::new(Mutex::new(false)))),
        Expr::ListPrepend(head, tail) => {
            let h = eval(local_env, node_env, head)?;
            let t = eval(local_env, node_env, tail)?.as_list()?.clone();
            let mut new_list = Vector::new();
            new_list.push_back(h);
            new_list.append(t);
            Ok(Value::List(new_list))
        }
        Expr::ListAppend(list, item) => {
            let mut l = eval(local_env, node_env, list)?.as_list()?.clone();
            let i = eval(local_env, node_env, item)?;
            l.push_back(i);
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
            Ok(Value::List(vec.clone().slice(s..e)))
        }
        Expr::LessThanEquals(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)? <= eval(local_env, node_env, e2)?,
        )),
        Expr::GreaterThan(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)? > eval(local_env, node_env, e2)?,
        )),
        Expr::GreaterThanEquals(e1, e2) => Ok(Value::Bool(
            eval(local_env, node_env, e1)? >= eval(local_env, node_env, e2)?,
        )),
        Expr::KeyExists(key, map) => {
            let k = eval(local_env, node_env, key)?;
            let m = eval(local_env, node_env, map)?;
            Ok(Value::Bool(m.as_map()?.contains_key(&k)))
        }
        Expr::MapErase(key, map) => {
            let k = eval(local_env, node_env, key)?;
            let m = eval(local_env, node_env, map)?.as_map()?.clone();
            Ok(Value::Map(m.without(&k)))
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
            Value::Option(Some(v)) => Ok(Arc::unwrap_or_clone(v)),
            Value::Option(None) => Err(RuntimeError::UnwrapNone),
            other => Err(RuntimeError::TypeError {
                expected: "option",
                got: other.type_name(),
            }),
        },
        Expr::Coalesce(opt, default) => match eval(local_env, node_env, opt)? {
            Value::Option(Some(v)) => Ok(Arc::unwrap_or_clone(v)),
            Value::Option(None) => eval(local_env, node_env, default),
            other => Err(RuntimeError::CoalesceNonOption {
                got: other.type_name(),
            }),
        },
        Expr::IntToString(e) => Ok(Value::String(EcoString::from(
            eval(local_env, node_env, e)?.as_int()?.to_string(),
        ))),
        Expr::Store(col, key, val) => update_collection(
            eval(local_env, node_env, col)?,
            eval(local_env, node_env, key)?,
            eval(local_env, node_env, val)?,
        ),
        Expr::SetTimer => todo!(),
    }
}

pub fn exec_sync_on_node<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env,
    node_id: usize,
    start_pc: usize,
) -> Result<Value, RuntimeError> {
    let node_env = Rc::clone(&state.nodes[node_id]);
    exec_sync_inner(
        state,
        logger,
        program,
        local_env,
        &mut node_env.borrow_mut(),
        start_pc,
        node_id,
        None,
    )
}

fn exec_sync_inner<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    local_env: &mut Env,
    node_env: &mut Env,
    start_pc: usize,
    node_id: usize,
    global_coverage: Option<&GlobalCoverage>,
) -> Result<Value, RuntimeError> {
    let mut pc = start_pc;
    let mut prev_pc = pc;
    loop {
        if pc != prev_pc {
            let rarity = global_coverage
                .map(|gc| gc.novelty_score(pc))
                .unwrap_or(1.0);
            state.coverage.record_with_rarity(prev_pc, pc, rarity);
            prev_pc = pc;
        }

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
                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;

                        if !func_info.is_sync {
                            return Err(RuntimeError::SyncCallToAsyncFunction(func_name.clone()));
                        }

                        let mut callee_local = Env::default();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_local.insert(*name, arg_vals[i].clone());
                        }
                        for (name, expr) in &func_info.locals {
                            callee_local.insert(*name, eval(local_env, node_env, expr)?);
                        }

                        if let Some(s) = local_env.get(&SELF_NAME) {
                            callee_local.insert(SELF_NAME, s.clone());
                        }

                        // Pass node_env directly (it's already mutable borrowed from caller)
                        let val = exec_sync_inner(
                            state,
                            logger,
                            program,
                            &mut callee_local,
                            node_env,
                            func_info.entry,
                            node_id,
                            global_coverage,
                        )?;

                        store(&lhs, val, local_env, node_env)?;
                    }
                    Instr::Async(lhs, node_expr, func_name, args) => {
                        // Similar to exec Async but we are inside sync.
                        let target_node = eval(local_env, node_env, &node_expr)?.as_node()?;
                        let arg_vals: Result<Vec<Value>, _> =
                            args.iter().map(|a| eval(local_env, node_env, a)).collect();
                        let arg_vals = arg_vals?;

                        let origin_node = local_env
                            .get(&SELF_NAME)
                            .ok_or_else(|| RuntimeError::VariableNotFound(SELF_NAME))?
                            .as_node()?;

                        let chan_id = ChannelId {
                            node: origin_node,
                            id: state.alloc_channel_id(),
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

                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let mut callee_locals = Env::default();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_locals.insert(*name, arg_vals[i].clone());
                        }
                        for (name, expr) in &func_info.locals {
                            callee_locals.insert(*name, eval(local_env, node_env, expr)?);
                        }

                        let new_record = Record {
                            pc: func_info.entry,
                            node: target_node,
                            origin_node,
                            continuation: Continuation::Async { chan_id },
                            env: callee_locals,
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
                    .get(&SELF_NAME)
                    .ok_or_else(|| RuntimeError::VariableNotFound(SELF_NAME))?
                    .as_node()?;
                let cid = ChannelId {
                    node: origin_node,
                    id: state.alloc_channel_id(),
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
                let val = eval(local_env, node_env, &expr)?;
                logger.log(LogEntry {
                    node: node_id,
                    content: val.to_string(),
                    step: state.crash_info.current_step,
                });
                pc = next;
            }
            Label::Break(target) => {
                pc = target;
            }
            Label::ForLoopIn(lhs, expr, body, next) => {
                let iter_key = iter_name_id(pc);

                if !local_env.contains_key(&iter_key) {
                    let original_collection = eval(local_env, node_env, &expr)?;
                    local_env.insert(iter_key, original_collection);
                }

                let col_val = local_env
                    .get(&iter_key)
                    .ok_or(RuntimeError::KeyNotFound)?
                    .clone();

                match col_val {
                    Value::List(l) => {
                        if l.is_empty() {
                            local_env.remove(&iter_key);
                            pc = next;
                        } else {
                            let item = l.head().unwrap().clone();
                            let new_l = Value::List(l.skip(1));
                            local_env.insert(iter_key, new_l);

                            store(&lhs, item, local_env, node_env)?;
                            pc = body;
                        }
                    }
                    Value::Map(m) => {
                        if m.is_empty() {
                            local_env.remove(&iter_key);
                            pc = next;
                        } else {
                            let (k, v) = m.iter().next().unwrap();
                            let k = k.clone();
                            let v = v.clone();

                            let new_m = m.without(&k);
                            local_env.insert(iter_key, Value::Map(new_m));

                            match lhs {
                                Lhs::Tuple(vars) if vars.len() == 2 => {
                                    store(&Lhs::Var(vars[0]), k, local_env, node_env)?;
                                    store(&Lhs::Var(vars[1]), v, local_env, node_env)?;
                                    pc = body;
                                }
                                _ => return Err(RuntimeError::ForLoopMapExpectsTupleLhs),
                            }
                        }
                    }
                    other => {
                        return Err(RuntimeError::ForLoopNotCollection {
                            got: other.type_name(),
                        });
                    }
                }
            }
            other => {
                return Err(RuntimeError::UnsupportedSyncInstruction(format!(
                    "{:?}",
                    other
                )));
            }
        }
    }
}

pub fn exec<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    mut record: Record,
    global_coverage: Option<&GlobalCoverage>,
) -> Result<Option<ClientOpResult>, RuntimeError> {
    let mut local_env = record.env;
    let node_env = Rc::clone(&state.nodes[record.node]);

    local_env.insert(SELF_NAME, Value::Node(record.node));

    let mut prev_pc = record.pc;

    loop {
        let current_pc = record.pc;
        if current_pc != prev_pc {
            let rarity = global_coverage
                .map(|gc| gc.novelty_score(current_pc))
                .unwrap_or(1.0);
            state
                .coverage
                .record_with_rarity(prev_pc, current_pc, rarity);
            prev_pc = current_pc;
        }

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
                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;

                        if !func_info.is_sync {
                            return Err(RuntimeError::SyncCallToAsyncFunction(func_name.clone()));
                        }

                        let mut callee_local = Env::default();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_local.insert(*name, arg_vals[i].clone());
                        }
                        for (name, expr) in &func_info.locals {
                            callee_local.insert(*name, eval(&local_env, &node_env.borrow(), expr)?);
                        }

                        if let Some(s) = local_env.get(&SELF_NAME) {
                            callee_local.insert(SELF_NAME, s.clone());
                        }

                        // Pass node_env directly
                        let ret_val = exec_sync_inner(
                            state,
                            logger,
                            program,
                            &mut callee_local,
                            &mut node_env.borrow_mut(),
                            func_info.entry,
                            record.node,
                            global_coverage,
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
                            id: state.alloc_channel_id(),
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
                        let func_name_id = program
                            .func_name_to_id
                            .get(&func_name)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let func_info = program
                            .rpc
                            .get(func_name_id)
                            .ok_or_else(|| RuntimeError::FunctionNotFound(func_name.clone()))?;
                        let mut callee_locals = Env::default();
                        for (i, name) in func_info.formals.iter().enumerate() {
                            callee_locals.insert(*name, arg_vals[i].clone());
                        }
                        for (name, expr) in &func_info.locals {
                            callee_locals
                                .insert(*name, eval(&local_env, &node_env.borrow(), expr)?);
                        }

                        let new_record = Record {
                            pc: func_info.entry,
                            node: target_node,
                            origin_node: record.node,
                            continuation: Continuation::Async { chan_id },
                            env: callee_locals,
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
                let cid = ChannelId {
                    node: record.node,
                    id: state.alloc_channel_id(),
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
                    return Ok(None);
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
                    return Ok(None); // Stop execution
                }
            }
            Label::Return(expr) => {
                let val = eval(&local_env, &node_env.borrow(), &expr)?;
                // Call continuation and return the result
                let result = record.continuation.call(state, val);
                return Ok(result);
            }
            Label::Pause(next) => {
                record.env = local_env;
                record.pc = next;
                state.runnable_records.push(record);
                return Ok(None); // Yield
            }
            Label::Cond(cond, bthen, belse) => {
                if eval(&local_env, &node_env.borrow(), &cond)?.as_bool()? {
                    record.pc = bthen;
                } else {
                    record.pc = belse;
                }
            }
            Label::Print(expr, next) => {
                let val = eval(&local_env, &node_env.borrow(), &expr)?;
                logger.log(LogEntry {
                    node: record.node,
                    content: val.to_string(),
                    step: state.crash_info.current_step,
                });
                record.pc = next;
            }
            Label::SpinAwait(expr, next) => {
                if eval(&local_env, &node_env.borrow(), &expr)?.as_bool()? {
                    record.pc = next;
                } else {
                    record.env = local_env;
                    state.runnable_records.push(record);
                    return Ok(None); // Yield
                }
            }
            Label::ForLoopIn(lhs, expr, body_pc, next_pc) => {
                let iter_key = iter_name_id(record.pc);

                if !local_env.contains_key(&iter_key) {
                    let original_collection = eval(&local_env, &node_env.borrow(), &expr)?;
                    local_env.insert(iter_key, original_collection);
                }

                let col_val = local_env
                    .get(&iter_key)
                    .ok_or(RuntimeError::KeyNotFound)?
                    .clone();

                match col_val {
                    Value::List(l) => {
                        if l.is_empty() {
                            local_env.remove(&iter_key);
                            record.pc = next_pc;
                        } else {
                            let item = l.head().unwrap().clone();
                            let new_l = Value::List(l.skip(1));
                            local_env.insert(iter_key, new_l);

                            store(&lhs, item, &mut local_env, &mut node_env.borrow_mut())?;
                            record.pc = body_pc;
                        }
                    }
                    Value::Map(m) => {
                        if m.is_empty() {
                            local_env.remove(&iter_key);
                            record.pc = next_pc;
                        } else {
                            let (k, v) = m.iter().next().unwrap();
                            let k = k.clone();
                            let v = v.clone();

                            let new_m = m.without(&k);
                            local_env.insert(iter_key, Value::Map(new_m));

                            match lhs {
                                Lhs::Tuple(vars) if vars.len() == 2 => {
                                    store(
                                        &Lhs::Var(vars[0]),
                                        k,
                                        &mut local_env,
                                        &mut node_env.borrow_mut(),
                                    )?;
                                    store(
                                        &Lhs::Var(vars[1]),
                                        v,
                                        &mut local_env,
                                        &mut node_env.borrow_mut(),
                                    )?;
                                    record.pc = body_pc;
                                }
                                _ => return Err(RuntimeError::ForLoopMapExpectsTupleLhs),
                            }
                        }
                    }
                    other => {
                        return Err(RuntimeError::ForLoopNotCollection {
                            got: other.type_name(),
                        });
                    }
                }
            }
            Label::Lock(lock_expr, next) => {
                let lock_val = eval(&local_env, &node_env.borrow(), &lock_expr)?;
                let lock = lock_val.as_lock()?;
                if *lock.lock().unwrap() == false {
                    *lock.lock().unwrap() = true;
                    record.pc = next;
                } else {
                    record.env = local_env;
                    state.runnable_records.push(record);
                    return Ok(None); // Yield
                }
            }
            Label::Unlock(lock_expr, next) => {
                let lock_val = eval(&local_env, &node_env.borrow(), &lock_expr)?;
                let lock = lock_val.as_lock()?;
                if *lock.lock().unwrap() == true {
                    *lock.lock().unwrap() = false;
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

pub fn schedule_record<L: Logger>(
    state: &mut State,
    logger: &mut L,
    program: &Program,
    randomly_drop_msgs: bool,
    cut_tail_from_mid: bool,
    sever_all_but_mid: bool,
    partition_away_nodes: &[usize],
    randomly_delay_msgs: bool,
    global_coverage: Option<&GlobalCoverage>,
) -> Result<Option<ClientOpResult>, RuntimeError> {
    let len = state.runnable_records.len();
    if len == 0 {
        return Ok(None); // Halt equivalent
    }

    use rand::Rng;
    let mut rng = rand::rng();

    // Select record index using either tournament selection or random
    let idx = if let Some(coverage) = global_coverage {
        if len > 1 {
            // Tournament selection with K=3
            const K: usize = 3;
            let mut best_idx = rng.random_range(0..len);
            let mut best_score = coverage.novelty_score(state.runnable_records[best_idx].pc);

            for _ in 1..K.min(len) {
                let candidate_idx = rng.random_range(0..len);
                let score = coverage.novelty_score(state.runnable_records[candidate_idx].pc);
                if score > best_score {
                    best_idx = candidate_idx;
                    best_score = score;
                }
            }
            best_idx
        } else {
            0
        }
    } else {
        rng.random_range(0..len)
    };

    let mut r = state.runnable_records.swap_remove(idx);

    let src_node = r.origin_node;
    let dest_node = r.node;

    if src_node != dest_node {
        if state.crash_info.currently_crashed.contains(&dest_node) {
            // Queue message
            state.crash_info.queued_messages.push((dest_node, r));
            return Ok(None);
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
                    return Ok(None);
                }
            }

            if should_execute {
                let result = exec(state, logger, program, r, global_coverage)?;
                return Ok(result);
            }
        }
    } else {
        if state.crash_info.currently_crashed.contains(&r.node) {
            return Ok(None);
        }
        let result = exec(state, logger, program, r, global_coverage)?;
        return Ok(result);
    }
    Ok(None)
}
