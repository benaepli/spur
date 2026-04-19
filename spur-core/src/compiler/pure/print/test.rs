//! Snapshot-style tests for the Pure IR pretty-printer.
//!
//! Each test builds a small `PProgram` directly (bypassing the threaded ->
//! pure lowering) and asserts the exact rendered output. Building the AST
//! by hand keeps tests independent of the lowerer.

use std::collections::HashMap;

use super::print_program;
use crate::analysis::resolver::NameId;
use crate::analysis::types::Type;
use crate::liquid::pure::ast::*;
use crate::parser::{BinOp, Span};

fn dummy_span() -> Span {
    Span::default()
}

fn nid(i: usize) -> NameId {
    NameId(i)
}

fn pvar(i: usize, name: &str) -> PAtomic {
    PAtomic::Var(nid(i), name.to_string())
}

fn pexpr(kind: PExprKind, ty: Type) -> PExpr {
    PExpr {
        kind,
        ty,
        span: dummy_span(),
    }
}

fn plet(name_id: usize, name: &str, ty: Type, value: PExpr) -> PStatement {
    PStatement {
        kind: PStatementKind::LetAtom(PLetAtom {
            name: nid(name_id),
            original_name: name.to_string(),
            ty,
            value,
            span: dummy_span(),
        }),
        span: dummy_span(),
    }
}

fn preturn(a: PAtomic) -> PStatement {
    PStatement {
        kind: PStatementKind::Return(a),
        span: dummy_span(),
    }
}

fn pblock(stmts: Vec<PStatement>, tail: Option<PAtomic>, ty: Type) -> PBlock {
    PBlock {
        statements: stmts,
        tail_expr: tail,
        ty,
        span: dummy_span(),
    }
}

fn pparam(name_id: usize, name: &str, ty: Type) -> PFuncParam {
    PFuncParam {
        name: nid(name_id),
        original_name: name.to_string(),
        ty,
        span: dummy_span(),
    }
}

fn pfunc(
    name_id: usize,
    name: &str,
    kind: PFuncKind,
    params: Vec<PFuncParam>,
    return_type: Type,
    body: PBlock,
) -> PFuncDef {
    PFuncDef {
        name: nid(name_id),
        original_name: name.to_string(),
        kind,
        is_traced: false,
        params,
        return_type,
        body,
        span: dummy_span(),
    }
}

fn empty_program(top: Vec<PTopLevelDef>) -> PProgram {
    PProgram {
        top_level_defs: top,
        next_name_id: 100,
        id_to_name: HashMap::new(),
        struct_defs: HashMap::new(),
        enum_defs: HashMap::new(),
    }
}

#[test]
fn prints_trivial_passthrough() {
    let body = pblock(vec![], Some(pvar(0, "a")), Type::Int);
    let func = pfunc(
        10,
        "id_int",
        PFuncKind::Sync,
        vec![pparam(0, "a", Type::Int)],
        Type::Int,
        body,
    );
    let prog = empty_program(vec![PTopLevelDef::FreeFunc(func)]);

    let expected = "\
# Pure IR / SSA

fn id_int(a: int) -> int {
  a
}

";
    assert_eq!(print_program(&prog), expected);
}

#[test]
fn prints_let_with_binop_and_return() {
    // fn add(x: int, y: int) -> int {
    //   let __t#5: int = x + y
    //   return __t#5
    // }
    let value = pexpr(
        PExprKind::BinOp(BinOp::Add, pvar(0, "x"), pvar(1, "y")),
        Type::Int,
    );
    let body = pblock(
        vec![
            plet(5, "__t", Type::Int, value),
            preturn(pvar(5, "__t")),
        ],
        None,
        Type::Int,
    );
    let func = pfunc(
        20,
        "add",
        PFuncKind::Sync,
        vec![pparam(0, "x", Type::Int), pparam(1, "y", Type::Int)],
        Type::Int,
        body,
    );
    let prog = empty_program(vec![PTopLevelDef::FreeFunc(func)]);

    let expected = "\
# Pure IR / SSA

fn add(x: int, y: int) -> int {
  let __t: int = x + y;
  return __t;
}

";
    assert_eq!(print_program(&prog), expected);
}

#[test]
fn prints_conditional_in_let() {
    // fn pick(c: bool) -> int {
    //   let r#5: int = if c { 1 } else { 2 }
    //   return r#5
    // }
    let if_body = pblock(vec![], Some(PAtomic::IntLit(1)), Type::Int);
    let else_body = pblock(vec![], Some(PAtomic::IntLit(2)), Type::Int);
    let cond = PCondExpr {
        if_branch: PIfBranch {
            condition: pvar(0, "c"),
            body: if_body,
            span: dummy_span(),
        },
        elseif_branches: vec![],
        else_branch: Some(else_body),
        span: dummy_span(),
    };
    let value = pexpr(PExprKind::Conditional(Box::new(cond)), Type::Int);
    let body = pblock(
        vec![plet(5, "r", Type::Int, value), preturn(pvar(5, "r"))],
        None,
        Type::Int,
    );
    let func = pfunc(
        30,
        "pick",
        PFuncKind::Sync,
        vec![pparam(0, "c", Type::Bool)],
        Type::Int,
        body,
    );
    let prog = empty_program(vec![PTopLevelDef::FreeFunc(func)]);

    let expected = "\
# Pure IR / SSA

fn pick(c: bool) -> int {
  let r: int =
  if c {
    1
  } else {
    2
  };
  return r;
}

";
    assert_eq!(print_program(&prog), expected);
}

#[test]
fn prints_role_with_async_and_loop_funcs() {
    let f_sync = pfunc(
        10,
        "init",
        PFuncKind::Sync,
        vec![],
        Type::Nil,
        pblock(vec![], None, Type::Nil),
    );
    let f_async = pfunc(
        11,
        "tick",
        PFuncKind::Async,
        vec![],
        Type::Nil,
        pblock(vec![], None, Type::Nil),
    );
    let f_loop = pfunc(
        12,
        "loop_helper",
        PFuncKind::LoopConverted,
        vec![pparam(0, "n", Type::Int)],
        Type::Int,
        pblock(vec![preturn(pvar(0, "n"))], None, Type::Int),
    );

    let role = PRoleDef {
        name: nid(100),
        original_name: "Server".to_string(),
        func_defs: vec![f_sync, f_async],
        span: dummy_span(),
    };

    let prog = empty_program(vec![
        PTopLevelDef::Role(role),
        PTopLevelDef::FreeFunc(f_loop),
    ]);

    let expected = "\
# Pure IR / SSA

role Server {
  fn init() -> nil {
  }

  [async] fn tick() -> nil {
  }
}

[loop] fn loop_helper(n: int) -> int {
  return n;
}

";
    assert_eq!(print_program(&prog), expected);
}

#[test]
fn prints_struct_and_enum_defs_sorted() {
    // Two structs, two enums, inserted out of order in the source map but
    // emitted in NameId-sorted order in the output.
    let mut struct_defs = HashMap::new();
    struct_defs.insert(
        nid(7),
        vec![
            ("ok".to_string(), Type::Bool),
            ("id".to_string(), Type::Int),
        ],
    );
    struct_defs.insert(nid(3), vec![("x".to_string(), Type::Int)]);

    let mut enum_defs = HashMap::new();
    enum_defs.insert(
        nid(9),
        vec![
            ("Some".to_string(), Some(Type::Int)),
            ("None".to_string(), None),
        ],
    );
    enum_defs.insert(nid(5), vec![("Red".to_string(), None)]);

    let prog = PProgram {
        top_level_defs: vec![],
        next_name_id: 100,
        id_to_name: HashMap::new(),
        struct_defs,
        enum_defs,
    };

    let expected = "\
# Pure IR / SSA

struct Struct#3 { x: int }
struct Struct#7 { ok: bool; id: int }

enum Enum#5 { Red }
enum Enum#9 { Some(int), None }

";
    assert_eq!(print_program(&prog), expected);
}

#[test]
fn prints_struct_and_enum_defs_with_names() {
    // Same shape as `prints_struct_and_enum_defs_sorted` but with id_to_name
    // populated, exercising the `name_lookup` hit path.
    let mut struct_defs = HashMap::new();
    struct_defs.insert(
        nid(7),
        vec![
            ("ok".to_string(), Type::Bool),
            ("body".to_string(), Type::Int),
        ],
    );
    struct_defs.insert(nid(3), vec![("x".to_string(), Type::Int)]);

    let mut enum_defs = HashMap::new();
    enum_defs.insert(
        nid(9),
        vec![
            ("Some".to_string(), Some(Type::Int)),
            ("None".to_string(), None),
        ],
    );
    enum_defs.insert(nid(5), vec![("Happy".to_string(), None)]);

    let mut id_to_name = HashMap::new();
    id_to_name.insert(nid(3), "Color".to_string());
    id_to_name.insert(nid(7), "Response".to_string());
    id_to_name.insert(nid(5), "Mood".to_string());
    id_to_name.insert(nid(9), "Optional".to_string());

    let prog = PProgram {
        top_level_defs: vec![],
        next_name_id: 100,
        id_to_name,
        struct_defs,
        enum_defs,
    };

    let expected = "\
# Pure IR / SSA

struct Color { x: int }
struct Response { ok: bool; body: int }

enum Mood { Happy }
enum Optional { Some(int), None }

";
    assert_eq!(print_program(&prog), expected);
}

#[test]
fn prints_collision_keeps_ids() {
    // fn f(uniq: int) -> int {
    //   let __upd: int = uniq + 1   // NameId 100
    //   let __upd: int = __upd + 2   // NameId 101 (same original name!)
    //   return __upd                 // refers to NameId 101
    // }
    //
    // The two `__upd` binders share the original name with distinct ids, so
    // both must keep their #id suffix; `uniq` is unique and stays bare.
    let v1 = pexpr(
        PExprKind::BinOp(BinOp::Add, pvar(0, "uniq"), PAtomic::IntLit(1)),
        Type::Int,
    );
    let v2 = pexpr(
        PExprKind::BinOp(BinOp::Add, pvar(100, "__upd"), PAtomic::IntLit(2)),
        Type::Int,
    );
    let body = pblock(
        vec![
            plet(100, "__upd", Type::Int, v1),
            plet(101, "__upd", Type::Int, v2),
            preturn(pvar(101, "__upd")),
        ],
        None,
        Type::Int,
    );
    let func = pfunc(
        50,
        "f",
        PFuncKind::Sync,
        vec![pparam(0, "uniq", Type::Int)],
        Type::Int,
        body,
    );
    let prog = empty_program(vec![PTopLevelDef::FreeFunc(func)]);

    let expected = "\
# Pure IR / SSA

fn f(uniq: int) -> int {
  let __upd#100: int = uniq + 1;
  let __upd#101: int = __upd#100 + 2;
  return __upd#101;
}

";
    assert_eq!(print_program(&prog), expected);
}
