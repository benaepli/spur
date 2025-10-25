use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use crate::analysis::resolver::{ResolvedClientDef, ResolvedProgram, ResolvedRoleDef, ResolvedTopLevelDef};
use crate::compiler::cfg;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
    EVar(String),
    EFind(Box<Expr>, Box<Expr>),
    EInt(i64),
    EBool(bool),
    ENot(Box<Expr>),
    EAnd(Box<Expr>, Box<Expr>),
    EOr(Box<Expr>, Box<Expr>),
    EEqualsEquals(Box<Expr>, Box<Expr>),
    EMap(Vec<(Expr, Expr)>),
    EList(Vec<Expr>),
    EListPrepend(Box<Expr>, Box<Expr>),
    EListAppend(Box<Expr>, Box<Expr>),
    EListSubsequence(Box<Expr>, Box<Expr>, Box<Expr>),
    EString(String),
    ELessThan(Box<Expr>, Box<Expr>),
    ELessThanEquals(Box<Expr>, Box<Expr>),
    EGreaterThan(Box<Expr>, Box<Expr>),
    EGreaterThanEquals(Box<Expr>, Box<Expr>),
    EKeyExists(Box<Expr>, Box<Expr>),
    EListLen(Box<Expr>),
    EListAccess(Box<Expr>, usize),
    EPlus(Box<Expr>, Box<Expr>),
    EMinus(Box<Expr>, Box<Expr>),
    ETimes(Box<Expr>, Box<Expr>),
    EDiv(Box<Expr>, Box<Expr>),
    EMod(Box<Expr>, Box<Expr>),
    EPollForResps(Box<Expr>, Box<Expr>),
    EPollForAnyResp(Box<Expr>),
    ENextResp(Box<Expr>),
    EMin(Box<Expr>, Box<Expr>),
    ETuple(Vec<Expr>),
    ETupleAccess(Box<Expr>, usize),
    EUnit,
    ENil,
    EUnwrap(Box<Expr>),
    ECoalesce(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Lhs {
    Var(String),
    Access(Expr, Expr),
    Tuple(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Instr {
    Assign(Lhs, Expr),
    Async(Lhs, Expr, String, Vec<Expr>),
    Copy(Lhs, Expr),
}

pub type Vertex = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionInfo {
    pub entry: Vertex,
    pub name: String,
    pub formals: Vec<String>,
    pub locals: Vec<(String, Expr)>, // (name, default_value)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Label {
    Instr(Instr, Vertex /* next_vertex */),
    Pause(Vertex /* next_vertex */),
    Await(Lhs, Expr, Vertex /* next_vertex */),
    Return(Expr),
    Cond(
        Expr,
        Vertex, /* then_vertex */
        Vertex, /* else_vertex */
    ),
    ForLoopIn(
        Lhs,
        Expr,
        Vertex, /* body_vertex */
        Vertex, /* next_vertex */
    ),
    Print(Expr, Vertex /* next_vertex */),
    Break(Vertex /* break_target_vertex */),
}

#[derive(Debug, Clone)]
pub enum Value {
    VInt(i64),
    VBool(bool),
    VString(String),
    VNode(i64),
    VUnit,
    VOption(Option<Box<Value>>),
    VTuple(Vec<Value>),
    VList(Rc<RefCell<Vec<Value>>>),
    VFuture(Rc<RefCell<Option<Value>>>),
    VMap(Rc<RefCell<HashMap<Value, Value>>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    // The CFG is just a list of all vertices.
    // The `Vertex` indices in `Label` and `FunctionInfo`
    // are indices into this Vec.
    pub cfg: Vec<Label>,

    // Map of function_name -> FunctionInfo
    pub rpc: HashMap<String, FunctionInfo>,
    pub client_ops: HashMap<String, FunctionInfo>,
}

pub struct Compiler {
    /// The CFG being built. New labels are pushed here.
    cfg: Vec<Label>,
    /// Counter for generating fresh temporary variable names.
    temp_counter: usize,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            cfg: Vec::new(),
            temp_counter: 0,
        }
    }

    /// Allocates a new, unique temporary variable name.
    fn new_temp_var(&mut self) -> String {
        let name = format!("_tmp{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    /// Adds a new label to the CFG and returns its Vertex index.
    fn add_label(&mut self, label: cfg::Label) -> cfg::Vertex {
        let vertex = self.cfg.len();
        self.cfg.push(label);
        vertex
    }

    /// The main entry point.
    pub fn compile_program(mut self, program: ResolvedProgram) -> cfg::Program {
        let mut rpc_map = HashMap::new();
        let mut client_ops_map = HashMap::new();

        // Compile all top-level definitions
        for def in program.top_level_defs {
            match def {
                ResolvedTopLevelDef::Role(role) => {
                    let (init_fn, rpc_fns) = self.compile_role_def(role);
                    // The role's init function is also an "RPC"
                    rpc_map.insert(init_fn.name.clone(), init_fn);
                    for func in rpc_fns {
                        rpc_map.insert(func.name.clone(), func);
                    }
                }
                ResolvedTopLevelDef::Type(_) => {
                    // Type definitions have no CFG representation
                }
            }
        }

        // Compile client definition
        let (init_fn, client_fns) = self.compile_client_def(program.client_def);
        client_ops_map.insert(init_fn.name.clone(), init_fn);
        for func in client_fns {
            client_ops_map.insert(func.name.clone(), func);
        }

        cfg::Program {
            cfg: self.cfg,
            rpc: rpc_map,
            client_ops: client_ops_map,
        }
    }

    fn compile_role_def(
        &mut self,
        role: ResolvedRoleDef,
    ) -> (cfg::FunctionInfo, Vec<cfg::FunctionInfo>) {
        // TODO:
        // 1. Compile `role.var_inits` into a single "BASE_NODE_INIT" function.
        // 2. Compile each func in `role.func_defs` using `compile_func_def`.
        // 3. Return (init_fn, other_fns)
        unimplemented!()
    }

    fn compile_client_def(
        &mut self,
        client: ResolvedClientDef,
    ) -> (cfg::FunctionInfo, Vec<cfg::FunctionInfo>) {
        // TODO:
        // 1. Compile `client.var_inits` into a "BASE_CLIENT_INIT" function.
        // 2. Compile each func in `client.func_defs` using `compile_func_def`.
        // 3. Return (init_fn, other_fns)
        unimplemented!()
    }
}