use super::*;
use crate::analysis::resolver::NameId;
use crate::compiler::cfg::{Cfg, Expr, Instr, Label, Lhs, Program, VarSlot};
use crate::simulator::core::state::{
    Continuation, LogEntry, Logger, NodeId, Record, SchedulePolicy, State,
};
use crate::simulator::core::values::{Env, Value};
use crate::simulator::coverage::{LocalCoverage, VertexMap};
use crate::simulator::feedback::CfgFeedback;
use crate::simulator::hash_utils::WithHashing;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::collections::HashMap;

fn slot(idx: u32) -> VarSlot {
    VarSlot::Local(idx, NameId(idx as usize))
}

#[test]
fn json_string_array_matches_serde_vec_of_strings() {
    let cases: Vec<Vec<&str>> = vec![
        vec![],
        vec![""],
        vec!["", ""],
        vec!["0"],
        vec!["\"quoted\"", "back\\slash"],
        vec!["new\nline", "tab\t", "cr\r", "bell\u{7}", "nul\0", "esc\u{1b}", "del\u{7f}"],
        vec!["unicode \u{e9}\u{1f600} \u{2028}", "</script>", "a/b"],
        vec!["[node(NameId(10)#0), node(NameId(10)#1)]", "{  }", "Some(\"x\")", "()"],
    ];
    for items in cases {
        let mut text = String::new();
        let mut ends = Vec::new();
        for item in &items {
            text.push_str(item);
            ends.push(text.len());
        }
        let expected = serde_json::to_string(
            &items
                .iter()
                .map(|s| serde_json::Value::String(s.to_string()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(json_string_array(&text, &ends), expected, "items {items:?}");
    }
}

#[test]
fn trace_payload_formats_values_and_errors() {
    let values: Vec<Result<Value<WithHashing>, RuntimeError>> = vec![
        Ok(Value::int(-7)),
        Err(RuntimeError::TypeError {
            expected: "int",
            got: "bool",
        }),
        Ok(Value::string("q\"s".into())),
    ];
    assert_eq!(
        trace_payload(values.into_iter()),
        "[\"-7\",\"<error>\",\"\\\"q\\\"s\\\"\"]"
    );
    let none: Vec<Result<Value<WithHashing>, RuntimeError>> = vec![];
    assert_eq!(trace_payload(none.into_iter()), "[]");
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
        let mut program = self.build_undecoded();
        program.decode();
        program
    }

    fn build_undecoded(self) -> Program {
        Program {
            cfg: Cfg { graph: self.labels },
            rpc: HashMap::new(),
            func_name_to_id: HashMap::new(),
            id_to_name: HashMap::new(),
            next_name_id: 0,
            vertex_to_span: HashMap::new(),
            max_node_slots: 2,
            roles: vec![],
            type_ids: HashMap::new(),
            compiled: Default::default(),
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
    let nid = NodeId {
        role: NameId(0),
        index: 0,
    };
    let env = Env::<WithHashing>::with_slots(local_slots);
    Record {
        pc,
        node: nid,
        origin_node: nid,
        continuation,
        entry_pc: pc,
        initial_args: ecow::EcoVec::new(),
        entry_func: NameId(0),
        env,
        priority: 0.5,
        causal_operation_id: None,
        trace_id: None,
        trace_payload: None,
        link_seq: None,
        origin_incarnation: 0,
        send_ordinal: 0,
        receiver_token_at_send: 0,
        bias: crate::simulator::util_stats::DeliveryBias::NONE,
        timer_entry: None,
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign1, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );
    assert!(result.is_ok());
}

#[test]
fn test_print_instruction() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Unit));
    let print = builder.add(Label::Print(Expr::Int(123), ret));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(print, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(init, 4);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );
    assert!(result.is_ok());
}

#[test]
fn test_pause_yields() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Int(42)));
    let pause = builder.add(Label::Pause(ret));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(pause, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
    assert_eq!(state.local_queues[0].len(), 1);
    match &state.local_queues[0][0] {
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign1, 2);
    let mut coverage = LocalCoverage::new();

    let _ = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );
    assert!(coverage.unique_edges() >= 2);
}

#[test]
fn test_channel_send_recv() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Var(slot(1))));
    let recv = builder.add(Label::Recv(Lhs::Var(slot(1)), Expr::Var(slot(0)), ret));
    let send = builder.add(Label::Send(Expr::Var(slot(0)), Expr::Int(99), recv));
    let make = builder.add(Label::MakeChannel(Lhs::Var(slot(0)), Some(1), send));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
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

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );

    let op = result.expect("exec failed").expect("should complete");
    assert_eq!(op.value, Value::<WithHashing>::int(99));
}

#[test]
fn test_recv_blocks_on_empty() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Unit));
    let recv = builder.add(Label::Recv(Lhs::Var(slot(1)), Expr::Var(slot(0)), ret));
    let make = builder.add(Label::MakeChannel(Lhs::Var(slot(0)), Some(1), recv));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(make, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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

    // Body: sum += k + v
    let accumulate = builder.add(Label::Instr(
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

    // Extract value to slot(1) from pair in slot(4)
    let extract_v = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(1)),
            Expr::TupleAccess(Box::new(Expr::Var(slot(4))), 1),
        ),
        accumulate,
    ));

    // Extract key to slot(0) from pair in slot(4)
    let extract_k = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(0)),
            Expr::TupleAccess(Box::new(Expr::Var(slot(4))), 0),
        ),
        extract_v,
    ));

    builder.labels[loop_start] = Label::ForLoopIn(
        Lhs::Var(slot(4)),
        Expr::Map(vec![
            (Expr::Int(1), Expr::Int(10)),
            (Expr::Int(2), Expr::Int(20)),
        ]),
        slot(2),
        extract_k,
        ret,
    );

    let init = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(3)), Expr::Int(0)),
        loop_start,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 5);
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

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
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

    // Extract slot(1) = tmp.1
    let extract_1 = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(1)),
            Expr::TupleAccess(Box::new(Expr::Var(slot(2))), 1),
        ),
        ret,
    ));

    // Extract slot(0) = tmp.0
    let extract_0 = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(0)),
            Expr::TupleAccess(Box::new(Expr::Var(slot(2))), 0),
        ),
        extract_1,
    ));

    // Assign tuple to tmp slot(2)
    let assign = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(2)),
            Expr::Tuple(vec![Expr::Int(10), Expr::Int(20)]),
        ),
        extract_0,
    ));

    let program = builder.build();
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 3);
    let mut logger = TestLogger::new();
    let record = make_record_with_cont(
        assign,
        3,
        Continuation::ClientOp {
            client_id: 0,
            op_name: "test".to_string(),
            unique_id: 0,
        },
    );
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );

    let op = result.expect("exec failed").expect("should complete");
    assert_eq!(op.value, Value::<WithHashing>::int(30));
}

/// What one execution left behind, for comparing the decoded loop with the
/// label loop.
struct Observed {
    result: String,
    logs: Vec<String>,
    state: String,
    edges: usize,
    events: [u64; 3],
    tally: crate::simulator::util_stats::InterpreterTally,
}

fn observe(program: &Program, start: usize, slots: usize) -> Observed {
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record_with_cont(
        start,
        slots,
        Continuation::ClientOp {
            client_id: 0,
            op_name: "test".to_string(),
            unique_id: 0,
        },
    );
    let mut coverage = LocalCoverage::new();
    let events_before = util_stats::pending_evaluator_events();
    let tally_before = util_stats::pending_interpreter_tally();
    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );
    let events_after = util_stats::pending_evaluator_events();
    let tally_after = util_stats::pending_interpreter_tally();
    let result = match result {
        Ok(Some(op)) => format!("value {:?}", op.value),
        Ok(None) => "yielded".to_string(),
        Err(e) => format!("error {e} / {e:?}"),
    };
    let t = crate::simulator::util_stats::InterpreterTally {
        label_execs: tally_after.label_execs - tally_before.label_execs,
        legacy_labels: tally_after.legacy_labels - tally_before.legacy_labels,
        leaf_operands_inline: tally_after.leaf_operands_inline - tally_before.leaf_operands_inline,
        tree_evals: tally_after.tree_evals - tally_before.tree_evals,
        legacy_evals: tally_after.legacy_evals - tally_before.legacy_evals,
        call_targets_indexed: tally_after.call_targets_indexed - tally_before.call_targets_indexed,
        call_targets_fallback: tally_after.call_targets_fallback
            - tally_before.call_targets_fallback,
    };
    Observed {
        result,
        logs: logger
            .entries
            .iter()
            .map(|e| format!("{:?} {} {}", e.node, e.step, e.content))
            .collect(),
        state: format!("{state:?}"),
        edges: coverage.unique_edges(),
        events: [
            events_after[0] - events_before[0],
            events_after[1] - events_before[1],
            events_after[2] - events_before[2],
        ],
        tally: t,
    }
}

/// Runs `program` through the decoded loop and through the label loop and
/// requires the same result, output, state and evaluator events.
fn assert_loops_agree(program: Program, start: usize, slots: usize) -> (Observed, Observed) {
    assert!(program.compiled.covers(program.cfg.graph.len()));
    let mut undecoded = program.clone();
    undecoded.compiled = Default::default();
    let decoded = observe(&program, start, slots);
    let legacy = observe(&undecoded, start, slots);
    assert_eq!(decoded.result, legacy.result);
    assert_eq!(decoded.logs, legacy.logs);
    assert_eq!(decoded.state, legacy.state);
    assert_eq!(decoded.edges, legacy.edges);
    assert_eq!(decoded.events, legacy.events);
    assert_eq!(decoded.tally.legacy_labels, 0);
    assert_eq!(decoded.tally.legacy_evals, 0);
    assert_eq!(legacy.tally.label_execs, 0);
    assert_eq!(decoded.tally.label_execs, legacy.tally.legacy_labels);
    (decoded, legacy)
}

fn function(entry: usize, name: usize, params: u32, slots: u32, is_sync: bool) -> FunctionInfo {
    FunctionInfo {
        entry,
        name: NameId(name),
        param_count: params,
        local_slot_count: slots,
        local_defaults: vec![],
        is_sync,
        debug_slot_names: vec![],
    }
}

fn register(program: &mut Program, qualified: &str, info: FunctionInfo) {
    program
        .func_name_to_id
        .insert(qualified.to_string(), info.name);
    program.rpc.insert(info.name, info);
}

#[test]
fn decoded_loop_matches_label_loop_across_calls_loops_and_channels() {
    let mut builder = TestProgramBuilder::new();
    let f_entry = builder.add(Label::Return(Expr::Plus(
        Box::new(Expr::Var(slot(0))),
        Box::new(Expr::Int(1)),
    )));
    let h_entry = builder.add(Label::Return(Expr::Var(slot(0))));

    let ret_else = builder.add(Label::Return(Expr::Unit));
    let ret_then = builder.add(Label::Return(Expr::Tuple(vec![
        Expr::Var(slot(0)),
        Expr::Var(slot(6)),
    ])));
    let spawn = builder.add(Label::Instr(
        Instr::Async(
            Lhs::Var(slot(7)),
            Expr::Var(node_slot(0)),
            "h".to_string(),
            vec![Expr::Var(slot(6))],
        ),
        ret_then,
    ));
    let recv = builder.add(Label::Recv(Lhs::Var(slot(6)), Expr::Var(slot(5)), spawn));
    let send = builder.add(Label::Send(Expr::Var(slot(5)), Expr::Int(42), recv));
    let make = builder.add(Label::MakeChannel(Lhs::Var(slot(5)), None, send));
    let to_node = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(node_slot(1)), Expr::Var(slot(0))),
        make,
    ));
    let cond = builder.add(Label::Cond(Expr::Var(slot(4)), to_node, ret_else));
    let compare = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(4)),
            Expr::Not(Box::new(Expr::EqualsEquals(
                Box::new(Expr::Var(slot(0))),
                Box::new(Expr::Int(0)),
            ))),
        ),
        cond,
    ));

    let loop_head = builder.add(Label::Return(Expr::Unit));
    let back = builder.add(Label::Continue(loop_head));
    let print = builder.add(Label::Print(Expr::Var(slot(0)), back));
    let accumulate = builder.add(Label::Instr(
        Instr::Assign(
            Lhs::Var(slot(0)),
            Expr::Plus(Box::new(Expr::Var(slot(0))), Box::new(Expr::Var(slot(3)))),
        ),
        print,
    ));
    let call = builder.add(Label::Instr(
        Instr::SyncCall(Lhs::Var(slot(3)), "f".to_string(), vec![Expr::Var(slot(1))]),
        accumulate,
    ));
    builder.labels[loop_head] = Label::ForLoopIn(
        Lhs::Var(slot(1)),
        Expr::List(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]),
        slot(2),
        call,
        compare,
    );
    let init = builder.add(Label::Instr(
        Instr::Assign(Lhs::Var(slot(0)), Expr::Int(0)),
        loop_head,
    ));

    let mut program = builder.build_undecoded();
    register(&mut program, "f", function(f_entry, 5, 1, 1, true));
    register(&mut program, "h", function(h_entry, 6, 1, 1, false));
    program.decode();

    let (decoded, _) = assert_loops_agree(program, init, 8);
    assert!(decoded.result.contains("Tuple([Value { kind: Int(9)"));
    assert!(decoded.result.contains("kind: Int(42)"));
    assert_eq!(decoded.logs.len(), 3);
    assert_eq!(decoded.tally.call_targets_indexed, 4);
    assert_eq!(decoded.tally.call_targets_fallback, 0);
    assert!(decoded.tally.leaf_operands_inline > 0);
}

#[test]
fn unresolved_and_async_callees_fail_alike_on_both_loops() {
    let mut builder = TestProgramBuilder::new();
    let h_entry = builder.add(Label::Return(Expr::Unit));
    let ret = builder.add(Label::Return(Expr::Unit));
    let missing = builder.add(Label::Instr(
        Instr::SyncCall(Lhs::Var(slot(0)), "missing".to_string(), vec![]),
        ret,
    ));
    let to_async = builder.add(Label::Instr(
        Instr::SyncCall(Lhs::Var(slot(0)), "h".to_string(), vec![]),
        ret,
    ));
    let mut program = builder.build_undecoded();
    register(&mut program, "h", function(h_entry, 6, 0, 0, false));
    program.decode();

    let (decoded, _) = assert_loops_agree(program.clone(), missing, 2);
    assert!(decoded.result.contains("function not found: missing"));
    assert_eq!(decoded.tally.call_targets_fallback, 1);

    let (decoded, _) = assert_loops_agree(program, to_async, 2);
    assert!(decoded.result.contains("sync function tried to call non-sync function: h"));
    assert_eq!(decoded.tally.call_targets_indexed, 1);
}

#[test]
fn a_channel_operation_in_a_sync_function_fails_alike_on_both_loops() {
    let mut builder = TestProgramBuilder::new();
    let ret = builder.add(Label::Return(Expr::Unit));
    let send = builder.add(Label::Send(Expr::Var(slot(0)), Expr::Int(1), ret));
    let program = builder.build();
    let mut undecoded = program.clone();
    undecoded.compiled = Default::default();
    let run = |program: &Program| {
        let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
        let mut logger = TestLogger::new();
        let mut env = Env::<WithHashing>::with_slots(1);
        let mut coverage = LocalCoverage::new();
        exec_sync_on_node::<WithHashing, TestLogger, CfgFeedback>(
            &mut state,
            &mut logger,
            program,
            &mut env,
            NodeId {
                role: NameId(0),
                index: 0,
            },
            send,
            &VertexMap::new(),
            &mut coverage,
            &SchedulePolicy::Fixed,
            &PurgatoryConfig::default(),
            &mut SmallRng::seed_from_u64(0),
        )
        .unwrap_err()
        .to_string()
    };
    let decoded = run(&program);
    assert_eq!(decoded, run(&undecoded));
    assert!(decoded.starts_with("instruction not allowed in sync function: Send("));
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
    let mut state = State::<WithHashing>::new(&[(NameId(0), 1)], 2);
    let mut logger = TestLogger::new();
    let record = make_record(assign, 2);
    let mut coverage = LocalCoverage::new();

    let result = exec::<WithHashing, TestLogger, CfgFeedback>(
        &mut state,
        &mut logger,
        &program,
        record,
        &VertexMap::new(),
        &mut coverage,
        &SchedulePolicy::Fixed,
        &PurgatoryConfig::default(),
        &mut SmallRng::seed_from_u64(0),
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        crate::simulator::core::RuntimeError::TypeError { .. } => (),
        e => panic!("Expected TypeError, got {:?}", e),
    }
}
