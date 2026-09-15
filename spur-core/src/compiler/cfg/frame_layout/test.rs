//! Checks a slot layout against the declared one by a separate method: a
//! forward reaching-values analysis over the laid-out program, with every
//! store tagged by the declared slot it wrote. A read of declared slot `s` may
//! only ever see a value stored to `s`, or the value the frame was built
//! with when that is what `s` would hold.

use crate::compiler::cfg::{
    Compiler, Expr, FunctionInfo, Instr, Label, Lhs, Program, SlotDefault, VarSlot, Vertex,
};
use crate::compiler::compile_with;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    Always,
    FirstSuccessor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Read(u32),
    Store(u32, Reach),
}

fn subexpressions(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Var(_) | Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Unit | Expr::Nil => {
            vec![]
        }
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
        | Expr::SafeFind(a, b) => vec![&**a, &**b],
        Expr::ListSubsequence(a, b, c) | Expr::Store(a, b, c) => vec![&**a, &**b, &**c],
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
        | Expr::SafeTupleAccess(a, _) => vec![&**a],
        Expr::Map(pairs) => pairs.iter().flat_map(|(k, v)| [k, v]).collect(),
        Expr::List(items) | Expr::Tuple(items) => items.iter().collect(),
        Expr::Variant(_, _, payload) => payload.iter().map(|p| &**p).collect(),
    }
}

fn reads_of(e: &Expr, out: &mut Vec<Event>) {
    let mut stack = vec![e];
    while let Some(e) = stack.pop() {
        if let Expr::Var(VarSlot::Local(i, _)) = e {
            out.push(Event::Read(*i));
        }
        stack.extend(subexpressions(e));
    }
}

fn store_of(lhs: &Lhs, reach: Reach, out: &mut Vec<Event>) {
    let Lhs::Var(slot) = lhs;
    if let VarSlot::Local(i, _) = slot {
        out.push(Event::Store(*i, reach));
    }
}

/// The label's local slot events in execution order and its successors.
fn events(label: &Label, at: Vertex) -> (Vec<Event>, Vec<Vertex>) {
    let mut ev = Vec::new();
    let succ = match label {
        Label::Instr(Instr::Assign(lhs, rhs), n) | Label::Instr(Instr::Copy(lhs, rhs), n) => {
            reads_of(rhs, &mut ev);
            store_of(lhs, Reach::Always, &mut ev);
            vec![*n]
        }
        Label::Instr(Instr::Async(lhs, target, _, args), n) => {
            reads_of(target, &mut ev);
            args.iter().for_each(|a| reads_of(a, &mut ev));
            store_of(lhs, Reach::Always, &mut ev);
            vec![*n]
        }
        Label::Instr(Instr::SyncCall(lhs, _, args), n) => {
            args.iter().for_each(|a| reads_of(a, &mut ev));
            store_of(lhs, Reach::Always, &mut ev);
            vec![*n]
        }
        Label::Pause(n) | Label::DiscardData(n) | Label::Break(n) | Label::Continue(n) => vec![*n],
        Label::MakeChannel(lhs, _, n) | Label::SetTimer(lhs, n, _) | Label::UniqueId(lhs, n) => {
            store_of(lhs, Reach::Always, &mut ev);
            vec![*n]
        }
        Label::Spawn(_, count, lhs, n, _) => { reads_of(count, &mut ev); store_of(lhs, Reach::Always, &mut ev); vec![*n] }
        Label::Provide(h, v, n, _) | Label::ProvideAll(h, v, n, _) => { reads_of(h, &mut ev); reads_of(v, &mut ev); vec![*n] }
        Label::RetrieveData(_, lhs, n) => {
            store_of(lhs, Reach::Always, &mut ev);
            vec![*n]
        }
        Label::MakeFifoLink(lhs, peer, n) => {
            reads_of(peer, &mut ev);
            store_of(lhs, Reach::Always, &mut ev);
            vec![*n]
        }
        Label::Send(chan, value, n) => {
            reads_of(chan, &mut ev);
            reads_of(value, &mut ev);
            vec![*n]
        }
        Label::Recv(lhs, chan, n) => {
            reads_of(chan, &mut ev);
            store_of(lhs, Reach::Always, &mut ev);
            vec![*n]
        }
        Label::SpinAwait(cond, n) => {
            reads_of(cond, &mut ev);
            vec![*n, at]
        }
        Label::Return(e) => {
            reads_of(e, &mut ev);
            vec![]
        }
        Label::Cond(cond, then, els) => {
            reads_of(cond, &mut ev);
            vec![*then, *els]
        }
        Label::ForLoopIn(lhs, collection, iter_slot, body, next) => {
            if let VarSlot::Local(i, _) = iter_slot {
                ev.push(Event::Read(*i));
            }
            reads_of(collection, &mut ev);
            if let VarSlot::Local(i, _) = iter_slot {
                ev.push(Event::Store(*i, Reach::Always));
            }
            store_of(lhs, Reach::FirstSuccessor, &mut ev);
            vec![*body, *next]
        }
        Label::Print(e, n) | Label::PersistData(_, e, n) => {
            reads_of(e, &mut ev);
            vec![*n]
        }
        Label::TraceDispatch(_, params, n) => {
            params.iter().for_each(|p| reads_of(p, &mut ev));
            vec![*n]
        }
        Label::TraceEnter(_, params, lhs, n) => {
            store_of(lhs, Reach::Always, &mut ev);
            params.iter().for_each(|p| reads_of(p, &mut ev));
            vec![*n]
        }
        Label::TraceExit(_, trace_id, return_value, n) => {
            reads_of(trace_id, &mut ev);
            reads_of(return_value, &mut ev);
            vec![*n]
        }
    };
    (ev, succ)
}

/// The label's text with every local slot index and every type id blanked.
/// Type ids are numbered in hash map order, so two compilations of one source
/// can number the same type differently.
fn shape(label: &Label) -> String {
    let mut text = format!("{label:?}");
    for marker in ["Local(", "TypeId("] {
        let mut out = String::with_capacity(text.len());
        let mut rest = text.as_str();
        while let Some(pos) = rest.find(marker) {
            out.push_str(&rest[..pos + marker.len()]);
            rest = rest[pos + marker.len()..].trim_start_matches(|c: char| c.is_ascii_digit());
            out.push('_');
        }
        out.push_str(rest);
        text = out;
    }
    text
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Held {
    FrameStart,
    StoredTo(u32),
}

fn default_of(info: &FunctionInfo, slot: u32) -> SlotDefault {
    (slot as usize)
        .checked_sub(info.param_count as usize)
        .and_then(|i| info.local_defaults.get(i))
        .copied()
        .unwrap_or(SlotDefault::Unit)
}

/// Accepts the layout of `laid` when every read in it sees only values its
/// declared slot in `declared` could hold.
fn check_layout(
    declared_prog: &Program,
    declared: &FunctionInfo,
    laid_prog: &Program,
    laid: &FunctionInfo,
) -> Result<(), String> {
    if declared.entry != laid.entry || declared.param_count != laid.param_count {
        return Err("entry or parameter count differs".into());
    }
    let pc = laid.param_count;
    let colors = laid.local_slot_count as usize;
    if laid.local_defaults.len() + pc as usize != colors {
        return Err(format!(
            "{} starting values for {} slots and {} parameters",
            laid.local_defaults.len(),
            colors,
            pc
        ));
    }

    let mut paired: HashMap<Vertex, (Vec<(Event, u32)>, Vec<Vertex>)> = HashMap::new();
    let mut queue = VecDeque::from([laid.entry]);
    while let Some(v) = queue.pop_front() {
        if paired.contains_key(&v) {
            continue;
        }
        let (d_label, l_label) = (&declared_prog.cfg.graph[v], &laid_prog.cfg.graph[v]);
        if shape(d_label) != shape(l_label) {
            return Err(format!("vertex {v}: labels differ beyond slot numbers"));
        }
        let (d_ev, d_succ) = events(d_label, v);
        let (l_ev, l_succ) = events(l_label, v);
        if d_succ != l_succ || d_ev.len() != l_ev.len() {
            return Err(format!("vertex {v}: successors or events differ"));
        }
        let mut pairs = Vec::with_capacity(l_ev.len());
        for (d, l) in d_ev.into_iter().zip(l_ev) {
            let (d_slot, l_slot) = match (d, l) {
                (Event::Read(a), Event::Read(b)) => (a, b),
                (Event::Store(a, ra), Event::Store(b, rb)) if ra == rb => (a, b),
                _ => return Err(format!("vertex {v}: event kinds differ")),
            };
            if d_slot >= declared.local_slot_count || l_slot as usize >= colors {
                return Err(format!("vertex {v}: slot out of range"));
            }
            pairs.push((d, l_slot));
        }
        queue.extend(l_succ.iter().copied());
        paired.insert(v, (pairs, l_succ));
    }

    let start: Vec<BTreeSet<Held>> = vec![BTreeSet::from([Held::FrameStart]); colors];
    let mut state: HashMap<Vertex, Vec<BTreeSet<Held>>> = HashMap::from([(laid.entry, start)]);
    let mut work = VecDeque::from([laid.entry]);
    while let Some(v) = work.pop_front() {
        let (pairs, succ) = &paired[&v];
        for (edge, &target) in succ.iter().enumerate() {
            let mut out = state[&v].clone();
            for &(declared_event, color) in pairs {
                if let Event::Store(slot, reach) = declared_event {
                    if reach == Reach::Always || edge == 0 {
                        out[color as usize] = BTreeSet::from([Held::StoredTo(slot)]);
                    }
                }
            }
            let entry = state.entry(target).or_insert_with(|| vec![BTreeSet::new(); colors]);
            let mut grew = false;
            for (have, add) in entry.iter_mut().zip(out) {
                for h in add {
                    grew |= have.insert(h);
                }
            }
            if grew {
                work.push_back(target);
            }
        }
    }

    for (&v, (pairs, _)) in &paired {
        let mut held = state[&v].clone();
        for &(declared_event, color) in pairs {
            match declared_event {
                Event::Store(slot, _) => {
                    held[color as usize] = BTreeSet::from([Held::StoredTo(slot)]);
                }
                Event::Read(slot) => {
                    for h in &held[color as usize] {
                        let fine = match *h {
                            Held::StoredTo(s) => s == slot,
                            Held::FrameStart => {
                                if slot < pc {
                                    color == slot
                                } else {
                                    color >= pc && default_of(laid, color) == default_of(declared, slot)
                                }
                            }
                        };
                        if !fine {
                            return Err(format!(
                                "vertex {v}: a read of declared slot {slot} at position {color} can see {h:?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn spec_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("the spec directory is readable") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            spec_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "spur") {
            out.push(path);
        }
    }
}

/// The program with declared slots and the program laid out, or `None` when
/// the source compiles under neither. The layout runs after every check that
/// can reject a program, so compiling under one and not the other is a
/// failure.
fn compile_both(source: &str, name: &str) -> Option<(Program, Program)> {
    let declared =
        compile_with(source, name, Compiler::new().without_frame_compaction()).into_program();
    let laid = compile_with(source, name, Compiler::new()).into_program();
    match (declared, laid) {
        (Ok(declared), Ok(laid)) => {
            assert_eq!(declared.cfg.graph.len(), laid.cfg.graph.len(), "{name}");
            Some((declared, laid))
        }
        (Err(_), Err(_)) => None,
        (declared, laid) => panic!(
            "{name} compiles with declared slots: {}, laid out: {}",
            declared.is_ok(),
            laid.is_ok()
        ),
    }
}

fn on_big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("the test thread starts")
        .join()
        .expect("the test thread finishes");
}

#[test]
fn every_spec_function_layout_keeps_every_read() {
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bin/spur");
        let mut files = Vec::new();
        spec_files(&root, &mut files);
        files.sort();
        assert!(!files.is_empty(), "no spec under {}", root.display());
        let mut failures = Vec::new();
        let mut skipped = 0;
        for file in &files {
            let name = file.display().to_string();
            let source = std::fs::read_to_string(file).expect("the spec is readable");
            let Some((declared, laid)) = compile_both(&source, &name) else {
                eprintln!("{name}: compiles under neither layout, skipped");
                skipped += 1;
                continue;
            };
            let (mut before, mut after) = (0u64, 0u64);
            let mut names: Vec<_> = declared.func_name_to_id.iter().collect();
            names.sort();
            for (func, id) in names {
                let d = &declared.rpc[id];
                let l = &laid.rpc[id];
                before += u64::from(d.local_slot_count);
                after += u64::from(l.local_slot_count);
                assert!(l.local_slot_count <= d.local_slot_count, "{name} {func}");
                if let Err(e) = check_layout(&declared, d, &laid, l) {
                    failures.push(format!("{name} {func}: {e}"));
                }
            }
            assert_eq!(declared.rpc.len(), laid.rpc.len());
            eprintln!("{name}: {} functions, slots {before} -> {after}", declared.rpc.len());
        }
        assert!(skipped < files.len(), "no spec compiled");
        assert!(failures.is_empty(), "layouts that change a read:\n{}", failures.join("\n"));
    });
}

const LOOP_SPEC: &str = r#"
role Node(unused: int) {
    var total: int = 0;

    fn Count(n: int): int {
        for var i: int = 0; i < n; i = i + 1 {
            total = total + i;
        }
        return total;
    }
}
"#;

/// Moves the dead store at the loop's condition entry onto the position of
/// the loop variable in `program`.
fn clobber_loop_variable(declared: &Program, program: &mut Program) {
    let info = declared.get_func_by_name("Node.Count").expect("the function exists");
    let laid_info = program.get_func_by_name("Node.Count").expect("the function exists").clone();
    let i_declared = info
        .debug_slot_names
        .iter()
        .position(|n| n == "i")
        .expect("the loop variable has a slot") as u32;

    let mut reachable = BTreeSet::new();
    let mut preds: HashMap<Vertex, usize> = HashMap::new();
    let mut queue = VecDeque::from([info.entry]);
    while let Some(v) = queue.pop_front() {
        if !reachable.insert(v) {
            continue;
        }
        for s in events(&declared.cfg.graph[v], v).1 {
            if s != v {
                *preds.entry(s).or_default() += 1;
                queue.push_back(s);
            }
        }
    }
    let read_slots: BTreeSet<u32> = reachable
        .iter()
        .flat_map(|&v| events(&declared.cfg.graph[v], v).0)
        .filter_map(|e| match e {
            Event::Read(s) => Some(s),
            Event::Store(..) => None,
        })
        .collect();
    let trampolines: Vec<Vertex> = reachable
        .iter()
        .copied()
        .filter(|v| preds.get(v).copied().unwrap_or(0) >= 2)
        .filter(|&v| {
            matches!(
                &declared.cfg.graph[v],
                Label::Instr(Instr::Assign(Lhs::Var(VarSlot::Local(d, _)), Expr::Unit), _)
                    if !read_slots.contains(d)
            )
        })
        .collect();
    assert_eq!(trampolines.len(), 1, "one dead store joins the loop's two entries");

    let i_position = {
        let mut found = None;
        for &v in &reachable {
            let d = events(&declared.cfg.graph[v], v).0;
            let l = events(&program.cfg.graph[v], v).0;
            for (a, b) in d.into_iter().zip(l) {
                if let (Event::Read(x), Event::Read(y)) = (a, b) {
                    if x == i_declared {
                        found = Some(y);
                    }
                }
            }
        }
        found.expect("the loop variable is read")
    };
    assert!(i_position < laid_info.local_slot_count);
    match &mut program.cfg.graph[trampolines[0]] {
        Label::Instr(Instr::Assign(Lhs::Var(VarSlot::Local(d, _)), Expr::Unit), _) => *d = i_position,
        other => panic!("unexpected label {other:?}"),
    }
}

#[test]
fn the_checker_rejects_a_dead_store_on_a_live_loop_variable() {
    on_big_stack(|| {
        let (declared, laid) = compile_both(LOOP_SPEC, "loop.spur").expect("the loop spec compiles");
        let info = declared.get_func_by_name("Node.Count").unwrap();
        let laid_info = laid.get_func_by_name("Node.Count").unwrap();
        assert_eq!(check_layout(&declared, info, &declared, info), Ok(()), "the declared layout");
        assert_eq!(check_layout(&declared, info, &laid, laid_info), Ok(()), "the computed layout");

        let mut clobbered = declared.clone();
        clobber_loop_variable(&declared, &mut clobbered);
        let err = check_layout(&declared, info, &clobbered, info);
        assert!(err.is_err(), "a declared layout with the dead store moved is accepted");

        let mut clobbered = laid.clone();
        clobber_loop_variable(&declared, &mut clobbered);
        let err = check_layout(&declared, info, &clobbered, laid_info);
        assert!(err.is_err(), "a computed layout with the dead store moved is accepted");
    });
}
