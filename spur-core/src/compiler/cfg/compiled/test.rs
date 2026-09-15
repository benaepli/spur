use super::check::{accesses, analyze, entries};
use super::*;
use crate::compiler::cfg::Compiler;
use crate::compiler::compile_with;
use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

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

fn spec_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bin/spur")
}

fn compile_spec(path: &Path) -> Option<Program> {
    let name = path.display().to_string();
    let source = std::fs::read_to_string(path).expect("the spec is readable");
    compile_with(&source, &name, Compiler::new()).into_program().ok()
}

fn on_big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("the test thread starts")
        .join()
        .expect("the test thread finishes");
}

/// Whether `rewritten` may stand in for `plain` at one vertex.
fn rewrite_allowed(plain: &Op, rewritten: &Op) -> bool {
    plain == rewritten
        || match (plain, rewritten) {
            (Op::AssignLocal { next, .. }, Op::StoreSkipped(to) | Op::StoreFolded(to)) => next == to,
            (Op::Print { value: Opnd::Local(_), next }, Op::PrintParts { next: to, .. }) => next == to,
            _ => false,
        }
}

#[test]
fn no_rewritten_value_is_read_in_any_spec() {
    on_big_stack(|| {
        let mut files = Vec::new();
        spec_files(&spec_root(), &mut files);
        files.sort();
        assert!(!files.is_empty());
        let mut failures = Vec::new();
        let (mut compiled, mut skipped_stores, mut folded_prints) = (0, 0, 0);
        for file in &files {
            let name = file.display().to_string();
            let Some(program) = compile_spec(file) else {
                eprintln!("{name}: does not compile, skipped");
                continue;
            };
            compiled += 1;
            let plain = CompiledProgram::build_with(&program, Rewrites::Off);
            assert_eq!(program.compiled.ops.len(), plain.ops.len(), "{name}");
            for (v, (p, r)) in plain.ops.iter().zip(&program.compiled.ops).enumerate() {
                assert!(rewrite_allowed(p, r), "{name} vertex {v}: {p:?} decoded as {r:?}");
                skipped_stores += usize::from(matches!(r, Op::StoreSkipped(_)));
                folded_prints += usize::from(matches!(r, Op::PrintParts { .. }));
            }
            let analysis = analyze(&program, &program.compiled.ops, &entries(&program));
            failures.extend(analysis.failures.iter().map(|f| format!("{name} {f}")));
            eprintln!("{name}: {} vertices checked", program.compiled.ops.len());
        }
        assert!(compiled > 0, "no spec compiled");
        assert!(skipped_stores > 0, "no store was skipped in any spec");
        assert!(folded_prints > 0, "no print was folded in any spec");
        assert!(failures.is_empty(), "rewritten values read:\n{}", failures.join("\n"));
    });
}

/// Edges that close a loop: from a vertex to one still on the depth-first
/// search stack from a function entry.
fn back_edges(program: &Program, ops: &[Op]) -> BTreeSet<(usize, usize)> {
    let n = ops.len();
    let succ: Vec<Vec<usize>> = (0..n)
        .map(|v| accesses(program, ops, v).1.into_iter().map(|(t, _)| t).filter(|t| *t < n).collect())
        .collect();
    let mut state = vec![0u8; n];
    let mut back = BTreeSet::new();
    for entry in entries(program) {
        if state[entry] != 0 {
            continue;
        }
        state[entry] = 1;
        let mut stack = vec![(entry, 0usize)];
        while let Some((u, i)) = stack.last_mut() {
            let u = *u;
            if let Some(&t) = succ[u].get(*i) {
                *i += 1;
                match state[t] {
                    0 => {
                        state[t] = 1;
                        stack.push((t, 0));
                    }
                    1 => {
                        back.insert((u, t));
                    }
                    _ => {}
                }
            } else {
                state[u] = 2;
                stack.pop();
            }
        }
    }
    back
}

/// A store at `v` to a slot that some path reads after crossing a loop's
/// back edge, with no write to the slot in between.
fn loop_variable_store(program: &Program, ops: &[Op]) -> Option<(usize, u32, u32)> {
    let back = back_edges(program, ops);
    for (v, op) in ops.iter().enumerate() {
        let Op::AssignLocal { slot, next, .. } = op else { continue };
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([(*next as usize, false)]);
        while let Some((u, crossed)) = queue.pop_front() {
            if u >= ops.len() || !seen.insert((u, crossed)) {
                continue;
            }
            let (reads, edges) = accesses(program, ops, u);
            if crossed && reads.contains(slot) {
                return Some((v, *slot, *next));
            }
            for (target, written) in edges {
                if !written.contains(slot) {
                    queue.push_back((target, crossed || back.contains(&(u, target))));
                }
            }
        }
    }
    None
}

#[test]
fn a_skipped_store_to_a_live_loop_variable_is_rejected() {
    on_big_stack(|| {
        let program = compile_spec(&spec_root().join("VR.spur")).expect("VR compiles");
        let entries = entries(&program);
        assert!(analyze(&program, &program.compiled.ops, &entries).failures.is_empty());
        let (v, slot, next) =
            loop_variable_store(&program, &program.compiled.ops).expect("VR has a loop variable store");
        let mut ops = program.compiled.ops.clone();
        ops[v] = Op::StoreSkipped(next);
        let failures = analyze(&program, &ops, &entries).failures;
        assert!(
            failures.iter().any(|f| f.contains(&format!("reads slot {slot},"))),
            "skipping the store at {v} was accepted: {failures:?}"
        );
    });
}
