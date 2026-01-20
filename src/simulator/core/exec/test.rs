use super::*;
use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Cfg, Expr, Instr, Label, Lhs, Program, VarSlot};
use crate::simulator::core::state::{Continuation, LogEntry, Logger, Record, State, UpdatePolicy};
use crate::simulator::core::values::{Env, Value};
use crate::simulator::coverage::LocalCoverage;
use crate::simulator::hash_utils::WithHashing;
use std::collections::HashMap;

fn slot(idx: u32) -> VarSlot {
    VarSlot::Local(idx, NameId(idx as usize))
}

fn node_slot(idx: u32) -> VarSlot {
    VarSlot::Node(idx, NameId(1000 + idx as usize))
}

struct TestProgramBuilder {
    labels: Vec<Label>,
}

impl TestProgramBuilder {
    fn new() -> Self {
        Self { labels: Vec::new() }
    }

    fn add(&mut self, label: Label) -> usize {
        let idx = self.labels.len();
        self.labels.push(label);
        idx
    }

    fn build(self) -> Program {
        Program {
            cfg: Cfg { graph: self.labels },
            rpc: HashMap::new(),
            func_name_to_id: HashMap::new(),
            id_to_name: HashMap::new(),
            next_name_id: 0,
            vertex_to_span: HashMap::new(),
            max_node_slots: 2,
        }
    }
}

struct TestLogger {
    entries: Vec<LogEntry>,
}

impl TestLogger {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl Logger for TestLogger {
    fn log(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }
}

fn make_record(pc: usize, local_slots: usize) -> Record<WithHashing> {
    make_record_with_cont(pc, local_slots, Continuation::Recover)
}

fn make_record_with_cont(
    pc: usize,
    local_slots: usize,
    continuation: Continuation<WithHashing>,
) -> Record<WithHashing> {
    Record {
        pc,
        node: 0,
        origin_node: 0,
        continuation,
        env: Env::<WithHashing>::with_slots(local_slots),
        x: 0.5,
        policy: UpdatePolicy::Identity,
    }
}

#[test]
fn test_assign_then_cond_true() {
    let mut builder = TestProgramBuilder::new();
    let ret_true = builder.add(Label::Return(Expr::Bool(true)));
    let ret_false = builder.add(Label::Return(Expr::Bool(false)));
    let cond = builder.add(Label::Cond(Expr::Var(slot(0)), ret_true, ret_false));
    let assign = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(0)), Expr::Bool(true)),
        cond,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
}

#[test]
fn test_assign_then_cond_false() {
    let mut builder = TestProgramBuilder::new();
    let ret_true = builder.add(Label::Return(Expr::Int(1)));
    let ret_false = builder.add(Label::Return(Expr::Int(0)));
    let cond = builder.add(Label::Cond(Expr::Var(slot(0)), ret_true, ret_false));
    let assign = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(0)), Expr::Bool(false)),
        cond,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
}

#[test]
fn test_arithmetic_assignment() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Var(slot(0))));
    let assign = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(0)),
            Expr::Plus(Box::new(Expr::Int(10)), Box::new(Expr::Int(32))),
        ),
        ret,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
}

#[test]
fn test_multiple_assigns() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Plus(
        Box::new(Expr::Var(slot(0))),
        Box::new(Expr::Var(slot(1))),
    )));
    let assign2 = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(1)), Expr::Int(20)),
        ret,
    ));
    let assign1 = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(0)), Expr::Int(10)),
        assign2,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign1, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
}

#[test]
fn test_node_slot_assignment() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Var(node_slot(1))));
    let assign = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(node_slot(1)), Expr::Int(42)),
        ret,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
    assert_eq!(state.nodes[0].get(1), &Value::<WithHashing>::int(42));
}

#[test]
fn test_copy_instruction() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Var(slot(1))));
    let copy = builder.add(Label::Instr(
        Instr::Copy(Lhs::Var(slot(1)), Expr::Var(slot(0))),
        ret,
    ));
    let assign = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(0)), Expr::Int(99)),
        copy,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
}

#[test]
fn test_print_instruction() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Unit));
    let print = builder.add(Label::Print(Expr::Int(123), ret));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(print, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
    assert_eq!(logger.entries.len(), 1);
    assert_eq!(logger.entries[0].content, "123");
}

#[test]
fn test_for_loop_in_list() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Var(slot(0))));
    let for_loop = builder.add(Label::Return(Expr::Unit));
    let body = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(0)),
            Expr::Plus(Box::new(Expr::Var(slot(0))), Box::new(Expr::Var(slot(1)))),
        ),
        for_loop,
    ));
    builder.labels[for_loop] = Label::ForLoopIn(
        Lhs::Var(slot(1)),
        Expr::List(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]),
        slot(2),
        body,
        ret,
    );
    let init = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(0)), Expr::Int(0)),
        for_loop,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(init, 4);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
}

#[test]
fn test_pause_yields() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Int(42)));
    let pause = builder.add(Label::Pause(ret));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(pause, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
    assert_eq!(state.runnable_tasks.len(), 1);
    match &state.runnable_tasks[0] {
        crate::simulator::core::state::Runnable::Record(r) => assert_eq!(r.pc, ret),
        _ => panic!("Expected Record"),
    }
}

#[test]
fn test_coverage_records_transitions() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Unit));
    let assign2 = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(1)), Expr::Int(2)),
        ret,
    ));
    let assign1 = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(0)), Expr::Int(1)),
        assign2,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign1, 2);
    let mut coverage = LocalCoverage::new();

    let _ = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );
    assert!(coverage.unique_edges() >= 2);
}

#[test]
fn test_channel_send_recv() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Var(slot(1))));
    let recv = builder.add(Label::Recv(Lhs::Var(slot(1)), Expr::Var(slot(0)), ret));
    let send = builder.add(Label::Send(Expr::Var(slot(0)), Expr::Int(99), recv));
    let make = builder.add(Label::MakeChannel(Lhs::Var(slot(0)), 1, send));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record_with_cont(
        make,
        2,
        Continuation::ClientOp {
            client_id: 0,
            op_name: "test".to_string(),
            unique_id: 0,
        },
    );
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );

    let op = result.expect("exec failed").expect("should complete");
    assert_eq!(op.value, Value::<WithHashing>::int(99));
}

#[test]
fn test_recv_blocks_on_empty() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Unit));
    let recv = builder.add(Label::Recv(Lhs::Var(slot(1)), Expr::Var(slot(0)), ret));
    let make = builder.add(Label::MakeChannel(Lhs::Var(slot(0)), 1, recv));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(make, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );

    assert!(result.is_ok());
    assert!(result.unwrap().is_none());

    let chan_id = state.channels.keys().next().unwrap();
    let chan = state.channels.get(chan_id).unwrap();
    assert_eq!(chan.waiting_readers.len(), 1);
}

#[test]
fn test_for_loop_map_destructuring() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Var(slot(3))));
    let loop_start = builder.add(Label::Return(Expr::Unit));

    let body = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(3)),
            Expr::Plus(
                Box::new(Expr::Var(slot(3))),
                Box::new(Expr::Plus(
                    Box::new(Expr::Var(slot(0))),
                    Box::new(Expr::Var(slot(1))),
                )),
            ),
        ),
        loop_start,
    ));

    builder.labels[loop_start] = Label::ForLoopIn(
        Lhs::Tuple(vec![slot(0), slot(1)]),
        Expr::Map(vec![
            (Expr::Int(1), Expr::Int(10)),
            (Expr::Int(2), Expr::Int(20)),
        ]),
        slot(2),
        body,
        ret,
    );

    let init = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(3)), Expr::Int(0)),
        loop_start,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 4);
    let mut logger = TestLogger::new();
    let record = make_record_with_cont(
        init,
        4,
        Continuation::ClientOp {
            client_id: 0,
            op_name: "test".to_string(),
            unique_id: 0,
        },
    );
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );

    let op = result.expect("exec failed").expect("should complete");
    assert_eq!(op.value, Value::<WithHashing>::int(33));
}

#[test]
fn test_tuple_assignment() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Plus(
        Box::new(Expr::Var(slot(0))),
        Box::new(Expr::Var(slot(1))),
    )));

    let assign = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Tuple(vec![slot(0), slot(1)]),
            Expr::Tuple(vec![Expr::Int(10), Expr::Int(20)]),
        ),
        ret,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record_with_cont(
        assign,
        2,
        Continuation::ClientOp {
            client_id: 0,
            op_name: "test".to_string(),
            unique_id: 0,
        },
    );
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );

    let op = result.expect("exec failed").expect("should complete");
    assert_eq!(op.value, Value::<WithHashing>::int(30));
}

#[test]
fn test_runtime_type_error() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Unit));

    let assign = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(0)),
            Expr::Plus(Box::new(Expr::Bool(true)), Box::new(Expr::Int(5))),
        ),
        ret,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(1, 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger>(
        &mut state,
        &mut logger,
        &program,
        record,
        None,
        &mut coverage,
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        crate::simulator::core::RuntimeError::TypeError { .. } => (),
        e => panic!("Expected TypeError, got {:?}", e),
    }
}
