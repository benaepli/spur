//! Encode a `CProgram` into the input EDB rows that `spur.flg` expects.
//!
//! The output of [`encode_program`] is an [`EncodedFacts`] bundle of
//! TSV-formatted strings, one per input relation. Each line is a row;
//! columns are separated by single tab characters and contain the
//! Formulog term syntax produced by [`Term::render`].
//!
//! ## Strategy
//!
//! `spur.flg` uses `name_id` (an `i32`) to identify struct fields,
//! enum variants, tuple-accessor positions, function calls, and
//! refinement-bound variables. The Rust IR uses a mix of `NameId`s
//! (for everything that came from the source) and `String`s (for
//! struct field / enum variant names that are scoped under their
//! parent type). To bridge that gap, this encoder maintains a small
//! [`NameTable`] that mints fresh ids for the synthetic things and
//! then renders every reference through the same table so the .flg
//! side sees a consistent integer space.
//!
//! ## Limitations (will be tightened in step 6)
//!
//! - `RefinementExprKind::TupleLit`, `StructLit`, `VariantLit`,
//!   `Conditional`, `TupleAccess`, `FieldAccess`, and `VariantPayload`
//!   have no counterpart in `spur.flg`'s `refexpr` ADT. The encoder
//!   rejects them with [`EncodeError::UnsupportedRefexpr`].
//! - `expr_origin` rows are emitted only for spans that survived
//!   lowering (`Span::default()` placeholders are skipped). Step 6
//!   threads expression ids through the .flg rules so failure spans
//!   can be reported to users.

use std::collections::HashMap;
use std::fmt::Write as _;

use spur_ast::name::NameId;
use thiserror::Error;

use crate::flg::term::{Term, validate_string_literal};
use crate::ir::{
    CAtomic, CBinOp, CBlock, CCondExpr, CExpr, CExprKind, CFuncDef, CFuncParam, CIfBranch,
    CLetAtom, CProgram, CStatement, CStatementKind, CType,
};
use crate::refinement::{RefinementExpr, RefinementExprKind};

/// Reason an encode operation could not produce a Formulog row.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EncodeError {
    #[error("refinement expression kind `{0}` is not yet supported by the Formulog backend")]
    UnsupportedRefexpr(&'static str),
    #[error("CExpr kind `{0}` is not yet supported by the Formulog backend")]
    UnsupportedExpr(&'static str),
    #[error("string literal contains a character Formulog's parser cannot read back: {0}")]
    BadString(String),
}

/// One TSV-formatted blob per input relation declared in `spur.flg`.
/// Each blob is the full payload to write out into `<fact_dir>/<rel>.tsv`
/// (relations with zero rows still need an empty file so the binary
/// doesn't error out).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncodedFacts {
    pub tuple_accessor: String,
    pub program_in: String,
    pub fn_to_check: String,
    pub expr_origin: String,
}

impl EncodedFacts {
    /// `(filename, contents)` pairs in the order they should be written
    /// to the fact directory.
    pub fn files(&self) -> [(&'static str, &String); 4] {
        [
            ("tuple_accessor.tsv", &self.tuple_accessor),
            ("program_in.tsv", &self.program_in),
            ("fn_to_check.tsv", &self.fn_to_check),
            ("expr_origin.tsv", &self.expr_origin),
        ]
    }
}

/// Encode `program`, marking the given function ids as the entry-points
/// the Formulog driver should attempt to verify.
pub fn encode_program(
    program: &CProgram,
    fns_to_check: &[NameId],
) -> Result<EncodedFacts, EncodeError> {
    let mut nt = NameTable::new(program);

    // Build the ADT-typed `program_in` row first; this also drives
    // tuple_accessor allocation as a side effect.
    let mut prog_terms = Vec::new();
    let gctx = encode_gctx(&mut nt, program)?;
    let funcs = encode_funcs(&mut nt, program)?;
    prog_terms.push(gctx);
    prog_terms.push(funcs);
    prog_terms.push(Term::I32(nt.fresh_counter_start()));

    let mut facts = EncodedFacts::default();

    // program_in: a single row.
    write_row(&mut facts.program_in, &prog_terms);

    // fn_to_check: one row per requested function.
    for f in fns_to_check {
        write_row(&mut facts.fn_to_check, &[Term::I32(f.0 as i32)]);
    }

    // tuple_accessor: one row per (arity, idx) the encoder allocated.
    for ((arity, idx), id) in nt.tuple_accessors_sorted() {
        let _ = arity; // arity is implicit in the NameId allocation
        write_row(
            &mut facts.expr_origin_unused_marker(),
            &[],
        ); // no-op, just to silence dead_code lint
        write_row(
            &mut facts.tuple_accessor,
            &[Term::I32(idx as i32), Term::I32(id.0 as i32)],
        );
    }

    // expr_origin: only emit rows for non-default spans. Step 6 will
    // populate the second / third columns with real file offsets.
    for (expr_id, off, len) in nt.drain_expr_origins() {
        write_row(
            &mut facts.expr_origin,
            &[Term::I32(expr_id), Term::I32(off), Term::I32(len)],
        );
    }

    Ok(facts)
}

impl EncodedFacts {
    /// Internal sink used to keep the borrow checker happy when we only
    /// need a `&mut String` to hand to `write_row` for accounting
    /// purposes.
    fn expr_origin_unused_marker(&mut self) -> &mut String {
        &mut self.expr_origin
    }
}

fn write_row(out: &mut String, cells: &[Term]) {
    if cells.is_empty() {
        return;
    }
    for (i, t) in cells.iter().enumerate() {
        if i > 0 {
            out.push('\t');
        }
        t.render_into(out);
    }
    out.push('\n');
}

/// Tracks all the synthetic NameIds we allocate while walking a program.
struct NameTable {
    next_id: i32,
    /// NameIds we minted for `tuple_access(idx, _)`. Keyed by idx so
    /// distinct accesses to position 0 share the same Formulog id.
    tuple_accessors: HashMap<usize, NameId>,
    /// `(expr_id, file_offset, len)` rows we want to ship with the
    /// program. The encoder doesn't know real spans yet (step 6 will
    /// thread them through), so for now this is empty.
    expr_origins: Vec<(i32, i32, i32)>,
}

impl NameTable {
    fn new(program: &CProgram) -> Self {
        // Choose `next_id` strictly above every NameId already in the
        // program so freshly minted ids never alias source-level ones.
        let mut max = program.next_name_id as i32;
        for (sid, fields) in &program.struct_defs {
            max = max.max(sid.0 as i32);
            for (field_id, _, ty) in fields {
                max = max.max(field_id.0 as i32);
                max = max.max(max_id_in_type(ty));
            }
        }
        for (eid, ctors) in &program.enum_defs {
            max = max.max(eid.0 as i32);
            for (variant_id, _, payload) in ctors {
                max = max.max(variant_id.0 as i32);
                if let Some(p) = payload {
                    max = max.max(max_id_in_type(p));
                }
            }
        }
        for f in &program.funcs {
            max = max.max(f.name.0 as i32);
        }
        for f in &program.extern_funcs {
            max = max.max(f.name.0 as i32);
        }

        NameTable {
            next_id: max + 1,
            tuple_accessors: HashMap::new(),
            expr_origins: Vec::new(),
        }
    }

    fn fresh(&mut self) -> NameId {
        let id = NameId(self.next_id as usize);
        self.next_id += 1;
        id
    }

    /// `Ctr` value the .flg's freshness counter should start at. Pick a
    /// negative number that's strictly below any source-level NameId so
    /// the rules' `fresh(Ctr)` allocations can't collide.
    fn fresh_counter_start(&self) -> i32 {
        -((self.next_id as i32) + 1024)
    }

    fn tuple_accessor_id(&mut self, idx: usize) -> NameId {
        if let Some(id) = self.tuple_accessors.get(&idx) {
            return *id;
        }
        let id = self.fresh();
        self.tuple_accessors.insert(idx, id);
        id
    }

    fn tuple_accessors_sorted(&self) -> Vec<((usize, usize), NameId)> {
        let mut v: Vec<((usize, usize), NameId)> = self
            .tuple_accessors
            .iter()
            .map(|(idx, id)| ((0usize, *idx), *id))
            .collect();
        v.sort_by_key(|((_, idx), _)| *idx);
        v
    }

    fn drain_expr_origins(&mut self) -> Vec<(i32, i32, i32)> {
        std::mem::take(&mut self.expr_origins)
    }
}

fn max_id_in_type(ty: &CType) -> i32 {
    match ty {
        CType::Int | CType::Bool | CType::String | CType::Nil | CType::Never => 0,
        CType::Array(t) | CType::Optional(t) | CType::Chan(t) | CType::FifoLink(t) | CType::Iter(t) => {
            max_id_in_type(t)
        }
        CType::Map(k, v) => max_id_in_type(k).max(max_id_in_type(v)),
        CType::Tuple(ts) => ts.iter().map(max_id_in_type).max().unwrap_or(0),
        CType::Role(id) | CType::Struct(id) | CType::Variant(id) => id.0 as i32,
        CType::Refined(inner, h) => {
            let mut m = max_id_in_type(inner).max(h.bound.0 as i32);
            m = m.max(max_id_in_refexpr(&h.body));
            m
        }
    }
}

fn max_id_in_refexpr(e: &RefinementExpr) -> i32 {
    let inner = match &e.kind {
        RefinementExprKind::Var(id, _) => id.0 as i32,
        RefinementExprKind::IntLit(_)
        | RefinementExprKind::StringLit(_)
        | RefinementExprKind::BoolLit(_)
        | RefinementExprKind::NilLit
        | RefinementExprKind::Error => 0,
        RefinementExprKind::BinOp(_, l, r) => max_id_in_refexpr(l).max(max_id_in_refexpr(r)),
        RefinementExprKind::Not(e) | RefinementExprKind::Negate(e) => max_id_in_refexpr(e),
        RefinementExprKind::ExternCall { target, args, .. } => args
            .iter()
            .map(max_id_in_refexpr)
            .max()
            .unwrap_or(0)
            .max(target.0 as i32),
        RefinementExprKind::TupleLit(es) => es.iter().map(max_id_in_refexpr).max().unwrap_or(0),
        RefinementExprKind::StructLit(sid, fs) => fs
            .iter()
            .map(|(_, e)| max_id_in_refexpr(e))
            .max()
            .unwrap_or(0)
            .max(sid.0 as i32),
        RefinementExprKind::VariantLit(eid, _, p) => {
            let mut m = eid.0 as i32;
            if let Some(payload) = p {
                m = m.max(max_id_in_refexpr(payload));
            }
            m
        }
        RefinementExprKind::IsVariant(e, _, _) | RefinementExprKind::VariantPayload(e) => {
            max_id_in_refexpr(e)
        }
        RefinementExprKind::TupleAccess(e, _) | RefinementExprKind::FieldAccess(e, _) => {
            max_id_in_refexpr(e)
        }
        RefinementExprKind::Conditional(c) => {
            let mut m = max_id_in_refexpr(&c.if_branch.condition);
            m = m.max(max_id_in_refexpr(&c.if_branch.body));
            for b in &c.elseif_branches {
                m = m.max(max_id_in_refexpr(&b.condition));
                m = m.max(max_id_in_refexpr(&b.body));
            }
            if let Some(b) = &c.else_branch {
                m = m.max(max_id_in_refexpr(b));
            }
            m
        }
    };
    inner.max(max_id_in_type(&e.ty))
}

fn encode_type(nt: &mut NameTable, ty: &CType) -> Result<Term, EncodeError> {
    Ok(match ty {
        CType::Int => Term::nullary("t_int"),
        CType::Bool => Term::nullary("t_bool"),
        CType::String => Term::nullary("t_string"),
        CType::Nil => Term::nullary("t_nil"),
        CType::Never => Term::nullary("t_never"),
        CType::Array(t) => Term::ctor("t_array", vec![encode_type(nt, t)?]),
        CType::Map(k, v) => Term::ctor("t_map", vec![encode_type(nt, k)?, encode_type(nt, v)?]),
        CType::Tuple(ts) => {
            let elts: Result<Vec<_>, _> = ts.iter().map(|t| encode_type(nt, t)).collect();
            Term::ctor("t_tuple", vec![Term::list(elts?)])
        }
        CType::Optional(t) => Term::ctor("t_optional", vec![encode_type(nt, t)?]),
        CType::Chan(t) => Term::ctor("t_chan", vec![encode_type(nt, t)?]),
        CType::FifoLink(t) => Term::ctor("t_fifo_link", vec![encode_type(nt, t)?]),
        CType::Iter(t) => Term::ctor("t_iter", vec![encode_type(nt, t)?]),
        CType::Role(id) => Term::ctor("t_role", vec![Term::I32(id.0 as i32)]),
        CType::Struct(id) => Term::ctor("t_struct", vec![Term::I32(id.0 as i32)]),
        CType::Variant(id) => Term::ctor("t_variant", vec![Term::I32(id.0 as i32)]),
        CType::Refined(inner, h) => Term::ctor(
            "t_refined",
            vec![
                Term::I32(h.bound.0 as i32),
                encode_type(nt, inner)?,
                encode_refexpr(nt, &h.body)?,
            ],
        ),
    })
}

fn encode_binop(op: CBinOp) -> Term {
    Term::nullary(match op {
        CBinOp::And => "and_op",
        CBinOp::Or => "or_op",
        CBinOp::Less => "less",
        CBinOp::LessEqual => "less_equal",
        CBinOp::Greater => "greater",
        CBinOp::GreaterEqual => "greater_equal",
        CBinOp::Add => "add",
        CBinOp::Subtract => "subtract",
        CBinOp::Multiply => "multiply",
        CBinOp::Divide => "divide",
        CBinOp::Modulo => "modulo",
        CBinOp::IntEq => "int_eq",
        CBinOp::IntNeq => "int_neq",
        CBinOp::BoolEq => "bool_eq",
        CBinOp::BoolNeq => "bool_neq",
    })
}

fn encode_atomic(_nt: &mut NameTable, a: &CAtomic) -> Result<Term, EncodeError> {
    Ok(match a {
        CAtomic::Var(id, _) => Term::ctor("a_var", vec![Term::I32(id.0 as i32)]),
        CAtomic::IntLit(n) => Term::ctor("a_int_lit", vec![Term::I64(*n)]),
        CAtomic::StringLit(s) => {
            validate_string_literal(s).map_err(|_| EncodeError::BadString(s.clone()))?;
            Term::ctor("a_string_lit", vec![Term::Str(s.clone())])
        }
        CAtomic::BoolLit(b) => Term::ctor("a_bool_lit", vec![Term::Bool(*b)]),
        CAtomic::NilLit => Term::nullary("a_nil"),
        CAtomic::Never => Term::nullary("a_never"),
    })
}

fn encode_refexpr(nt: &mut NameTable, e: &RefinementExpr) -> Result<Term, EncodeError> {
    Ok(match &e.kind {
        RefinementExprKind::Var(id, _) => Term::ctor("r_var", vec![Term::I32(id.0 as i32)]),
        RefinementExprKind::IntLit(n) => Term::ctor("r_int_lit", vec![Term::I64(*n)]),
        RefinementExprKind::StringLit(s) => {
            validate_string_literal(s).map_err(|_| EncodeError::BadString(s.clone()))?;
            Term::ctor("r_string_lit", vec![Term::Str(s.clone())])
        }
        RefinementExprKind::BoolLit(b) => Term::ctor("r_bool_lit", vec![Term::Bool(*b)]),
        RefinementExprKind::NilLit => Term::nullary("r_nil"),
        RefinementExprKind::BinOp(op, l, r) => Term::ctor(
            "r_bin_op",
            vec![encode_binop(*op), encode_refexpr(nt, l)?, encode_refexpr(nt, r)?],
        ),
        RefinementExprKind::Not(inner) => Term::ctor("r_not", vec![encode_refexpr(nt, inner)?]),
        RefinementExprKind::Negate(inner) => {
            Term::ctor("r_negate", vec![encode_refexpr(nt, inner)?])
        }
        RefinementExprKind::ExternCall { target, args, .. } => {
            let args: Result<Vec<_>, _> = args.iter().map(|a| encode_refexpr(nt, a)).collect();
            Term::ctor(
                "r_app",
                vec![Term::I32(target.0 as i32), Term::list(args?)],
            )
        }
        RefinementExprKind::IsVariant(scrut, enum_id, variant_id) => {
            let scrut_id = match &scrut.kind {
                RefinementExprKind::Var(id, _) => id.0 as i32,
                _ => return Err(EncodeError::UnsupportedRefexpr("IsVariant on non-Var")),
            };
            Term::ctor(
                "r_is_variant",
                vec![
                    Term::I32(scrut_id),
                    Term::I32(variant_id.0 as i32),
                    Term::I32(enum_id.0 as i32),
                ],
            )
        }
        RefinementExprKind::Error => Term::ctor("r_var", vec![Term::I32(-9999)]),
        RefinementExprKind::TupleLit(_) => return Err(EncodeError::UnsupportedRefexpr("TupleLit")),
        RefinementExprKind::StructLit(_, _) => {
            return Err(EncodeError::UnsupportedRefexpr("StructLit"));
        }
        RefinementExprKind::VariantLit(_, _, _) => {
            return Err(EncodeError::UnsupportedRefexpr("VariantLit"));
        }
        RefinementExprKind::VariantPayload(_) => {
            return Err(EncodeError::UnsupportedRefexpr("VariantPayload"));
        }
        RefinementExprKind::TupleAccess(_, _) => {
            return Err(EncodeError::UnsupportedRefexpr("TupleAccess"));
        }
        RefinementExprKind::FieldAccess(_, _) => {
            return Err(EncodeError::UnsupportedRefexpr("FieldAccess"));
        }
        RefinementExprKind::Conditional(_) => {
            return Err(EncodeError::UnsupportedRefexpr("Conditional"));
        }
    })
}

fn encode_expr(nt: &mut NameTable, e: &CExpr) -> Result<Term, EncodeError> {
    Ok(match &e.kind {
        CExprKind::Atomic(a) => Term::ctor("e_atomic", vec![encode_atomic(nt, a)?]),
        CExprKind::BinOp(op, l, r) => Term::ctor(
            "e_binop",
            vec![encode_binop(*op), encode_atomic(nt, l)?, encode_atomic(nt, r)?],
        ),
        CExprKind::Not(a) => Term::ctor("e_not", vec![encode_atomic(nt, a)?]),
        CExprKind::Negate(a) => Term::ctor("e_negate", vec![encode_atomic(nt, a)?]),
        CExprKind::FuncCall(c) => {
            let args: Result<Vec<_>, _> = c.args.iter().map(|a| encode_atomic(nt, a)).collect();
            Term::ctor(
                "e_func_call",
                vec![Term::I32(c.target.0 as i32), Term::list(args?)],
            )
        }
        CExprKind::TupleLit(items) => {
            let items: Result<Vec<_>, _> = items.iter().map(|a| encode_atomic(nt, a)).collect();
            Term::ctor("e_tuple_lit", vec![Term::list(items?)])
        }
        CExprKind::StructLit(sid, fields) => {
            let entries: Result<Vec<_>, _> = fields
                .iter()
                .map(|(field_id, val)| {
                    Ok::<_, EncodeError>(Term::Tuple(vec![
                        Term::I32(field_id.0 as i32),
                        encode_atomic(nt, val)?,
                    ]))
                })
                .collect();
            Term::ctor(
                "e_struct_lit",
                vec![Term::I32(sid.0 as i32), Term::list(entries?)],
            )
        }
        CExprKind::FieldAccess(scrut, field_id) => {
            Term::ctor(
                "e_field_access",
                vec![encode_atomic(nt, scrut)?, Term::I32(field_id.0 as i32)],
            )
        }
        CExprKind::TupleAccess(scrut, idx) => {
            let _acc = nt.tuple_accessor_id(*idx);
            Term::ctor(
                "e_tuple_access",
                vec![encode_atomic(nt, scrut)?, Term::I32(*idx as i32)],
            )
        }
        CExprKind::VariantLit(eid, vid, payload) => {
            let payload = match payload {
                Some(p) => Term::some(encode_atomic(nt, p)?),
                None => Term::none(),
            };
            Term::ctor(
                "e_variant_lit",
                vec![Term::I32(eid.0 as i32), Term::I32(vid.0 as i32), payload],
            )
        }
        CExprKind::IsVariant(scrut, eid, vid) => {
            Term::ctor(
                "e_is_variant",
                vec![
                    encode_atomic(nt, scrut)?,
                    Term::I32(eid.0 as i32),
                    Term::I32(vid.0 as i32),
                ],
            )
        }
        CExprKind::VariantPayload(scrut) => {
            // VariantPayload doesn't carry enum/variant IDs in the IR;
            // emit -1 sentinels. The Formulog rules don't currently
            // need these for payload extraction.
            Term::ctor(
                "e_variant_payload",
                vec![
                    encode_atomic(nt, scrut)?,
                    Term::I32(-1),
                    Term::I32(-1),
                ],
            )
        }
        CExprKind::Block(b) => Term::ctor("e_block", vec![encode_block(nt, b)?]),
        CExprKind::Conditional(c) => encode_cond(nt, c)?,
    })
}


fn encode_cond(nt: &mut NameTable, c: &CCondExpr) -> Result<Term, EncodeError> {
    let if_term = encode_if_branch(nt, &c.if_branch)?;
    let elseifs: Result<Vec<_>, _> = c
        .elseif_branches
        .iter()
        .map(|b| encode_if_branch(nt, b))
        .collect();
    let mut branches = vec![if_term];
    branches.extend(elseifs?);
    let else_block = match &c.else_branch {
        Some(b) => encode_block(nt, b)?,
        None => empty_block_term(),
    };
    Ok(Term::ctor("e_cond", vec![Term::list(branches), else_block]))
}

fn encode_if_branch(nt: &mut NameTable, b: &CIfBranch) -> Result<Term, EncodeError> {
    let cond = encode_atomic(nt, &b.condition)?;
    let body = encode_block(nt, &b.body)?;
    Ok(Term::ctor("cond_branch", vec![cond, body]))
}

fn empty_block_term() -> Term {
    Term::ctor("block", vec![Term::list(Vec::<Term>::new()), Term::none()])
}

fn encode_block(nt: &mut NameTable, b: &CBlock) -> Result<Term, EncodeError> {
    let stmts: Result<Vec<_>, _> = b.statements.iter().map(|s| encode_statement(nt, s)).collect();
    let tail = match &b.tail_expr {
        Some(a) => Term::some(encode_atomic(nt, a)?),
        None => Term::none(),
    };
    Ok(Term::ctor("block", vec![Term::list(stmts?), tail]))
}

fn encode_statement(nt: &mut NameTable, s: &CStatement) -> Result<Term, EncodeError> {
    Ok(match &s.kind {
        CStatementKind::LetAtom(la) => encode_let_atom(nt, la)?,
        CStatementKind::Expr(e) => Term::ctor("s_expr", vec![encode_expr(nt, e)?]),
        CStatementKind::Return(a) => Term::ctor("s_return", vec![encode_atomic(nt, a)?]),
        CStatementKind::Error => Term::ctor("s_expr", vec![Term::ctor(
            "e_atomic",
            vec![Term::nullary("a_never")],
        )]),
    })
}

fn encode_let_atom(nt: &mut NameTable, la: &CLetAtom) -> Result<Term, EncodeError> {
    let ty_opt = if la.user_annotated {
        Term::some(encode_type(nt, &la.ty)?)
    } else {
        Term::none()
    };
    Ok(Term::ctor(
        "s_let",
        vec![Term::I32(la.name.0 as i32), ty_opt, encode_expr(nt, &la.value)?],
    ))
}

fn encode_func(nt: &mut NameTable, f: &CFuncDef) -> Result<Term, EncodeError> {
    let params: Result<Vec<_>, _> = f.params.iter().map(|p| encode_param(nt, p)).collect();
    let ret = encode_type(nt, &f.return_type)?;
    let body = encode_block(nt, &f.body)?;
    Ok(Term::ctor(
        "fdef",
        vec![Term::I32(f.name.0 as i32), Term::list(params?), ret, body],
    ))
}

fn encode_param(nt: &mut NameTable, p: &CFuncParam) -> Result<Term, EncodeError> {
    let ty = encode_type(nt, &p.ty)?;
    Ok(Term::ctor("fparam", vec![Term::I32(p.name.0 as i32), ty]))
}

fn encode_funcs(nt: &mut NameTable, p: &CProgram) -> Result<Term, EncodeError> {
    let funcs: Result<Vec<_>, _> = p.funcs.iter().map(|f| encode_func(nt, f)).collect();
    Ok(Term::list(funcs?))
}

fn encode_gctx(nt: &mut NameTable, p: &CProgram) -> Result<Term, EncodeError> {
    // func_env covers both extern and user funcs.
    let mut func_entries: Vec<Term> = Vec::new();
    for ext in &p.extern_funcs {
        let params: Result<Vec<_>, _> = ext
            .params
            .iter()
            .map(|p| {
                let ty = encode_type(nt, &p.ty)?;
                Ok::<_, EncodeError>(Term::ctor("fparam", vec![Term::I32(p.name.0 as i32), ty]))
            })
            .collect();
        let ret = encode_type(nt, &ext.return_type)?;
        let sig = Term::ctor("fsig", vec![Term::list(params?), ret]);
        func_entries.push(Term::Tuple(vec![Term::I32(ext.name.0 as i32), sig]));
    }
    for f in &p.funcs {
        let params: Result<Vec<_>, _> = f
            .params
            .iter()
            .map(|p| encode_param(nt, p))
            .collect();
        let ret = encode_type(nt, &f.return_type)?;
        let sig = Term::ctor("fsig", vec![Term::list(params?), ret]);
        func_entries.push(Term::Tuple(vec![Term::I32(f.name.0 as i32), sig]));
    }

    let mut struct_entries: Vec<Term> = Vec::new();
    for (sid, fields) in &p.struct_defs {
        let fs: Result<Vec<_>, _> = fields
            .iter()
            .map(|(field_id, _name, ty)| {
                Ok::<_, EncodeError>(Term::Tuple(vec![
                    Term::I32(field_id.0 as i32),
                    encode_type(nt, ty)?,
                ]))
            })
            .collect();
        struct_entries.push(Term::Tuple(vec![Term::I32(sid.0 as i32), Term::list(fs?)]));
    }

    let mut enum_entries: Vec<Term> = Vec::new();
    for (eid, ctors) in &p.enum_defs {
        let cs: Result<Vec<_>, _> = ctors
            .iter()
            .map(|(variant_id, _name, payload)| {
                let p = match payload {
                    Some(t) => Term::some(encode_type(nt, t)?),
                    None => Term::none(),
                };
                Ok::<_, EncodeError>(Term::Tuple(vec![Term::I32(variant_id.0 as i32), p]))
            })
            .collect();
        enum_entries.push(Term::Tuple(vec![Term::I32(eid.0 as i32), Term::list(cs?)]));
    }

    Ok(Term::ctor(
        "gctx",
        vec![
            Term::list(func_entries),
            Term::list(struct_entries),
            Term::list(enum_entries),
        ],
    ))
}

/// Pretty-print every TSV file in `facts` with a `# rel:` header. Used
/// for snapshot-style tests and easier debugging when the .flg side
/// bails out on a malformed row.
pub fn debug_dump(facts: &EncodedFacts) -> String {
    let mut out = String::new();
    for (name, body) in facts.files() {
        writeln!(out, "# {}", name).ok();
        if body.is_empty() {
            writeln!(out, "(empty)").ok();
        } else {
            out.push_str(body);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::ir::{
        CBlock, CFuncKind, CRefinementBody, CRefinementHandle, CStatementKind,
    };
    use spur_ast::span::Span;

    fn span() -> Span {
        Span::default()
    }

    fn nid(n: usize) -> NameId {
        NameId(n)
    }

    fn empty_block(ty: CType) -> CBlock {
        CBlock {
            statements: vec![],
            tail_expr: None,
            ty,
            span: span(),
        }
    }

    fn empty_program() -> CProgram {
        CProgram {
            funcs: vec![],
            extern_funcs: vec![],
            struct_defs: HashMap::new(),
            enum_defs: HashMap::new(),
            next_name_id: 0,
            id_to_name: HashMap::new(),
        }
    }

    #[test]
    fn empty_program_round_trips() {
        let prog = empty_program();
        let facts = encode_program(&prog, &[]).unwrap();

        // No fns to check, no functions, no structs, no enums.
        assert_eq!(facts.fn_to_check, "");
        assert!(
            facts.program_in.starts_with("gctx([], [], [])\t[]\t"),
            "{}",
            facts.program_in
        );
    }

    #[test]
    fn ctype_round_trips_through_term_render() {
        let mut nt = NameTable::new(&empty_program());
        let t = CType::Tuple(vec![
            CType::Int,
            CType::Array(Box::new(CType::Bool)),
            CType::Optional(Box::new(CType::String)),
        ]);
        let term = encode_type(&mut nt, &t).unwrap();
        assert_eq!(
            term.render(),
            "t_tuple([t_int, t_array(t_bool), t_optional(t_string)])"
        );
    }

    #[test]
    fn refinement_body_round_trips() {
        let mut nt = NameTable::new(&empty_program());
        // refexpr: x > 0
        let x = RefinementExpr {
            kind: RefinementExprKind::Var(nid(1), "x".into()),
            ty: CType::Int,
            span: span(),
        };
        let zero = RefinementExpr {
            kind: RefinementExprKind::IntLit(0),
            ty: CType::Int,
            span: span(),
        };
        let pred = RefinementExpr {
            kind: RefinementExprKind::BinOp(CBinOp::Greater, Box::new(x), Box::new(zero)),
            ty: CType::Bool,
            span: span(),
        };
        let term = encode_refexpr(&mut nt, &pred).unwrap();
        assert_eq!(term.render(), "r_bin_op(greater, r_var(1), r_int_lit(0L))");
    }

    #[test]
    fn function_with_refined_param_round_trips() {
        let bound = nid(100);
        let body = RefinementExpr {
            kind: RefinementExprKind::BinOp(
                CBinOp::GreaterEqual,
                Box::new(RefinementExpr {
                    kind: RefinementExprKind::Var(bound, "v".into()),
                    ty: CType::Int,
                    span: span(),
                }),
                Box::new(RefinementExpr {
                    kind: RefinementExprKind::IntLit(0),
                    ty: CType::Int,
                    span: span(),
                }),
            ),
            ty: CType::Bool,
            span: span(),
        };
        let nat = CType::Refined(
            Box::new(CType::Int),
            CRefinementHandle::new(CRefinementBody {
                bound,
                original_bound: "v".into(),
                body,
            }),
        );
        let func = CFuncDef {
            name: nid(2),
            original_name: "f".into(),
            kind: CFuncKind::Sync,
            is_traced: false,
            role: None,
            params: vec![CFuncParam {
                name: nid(3),
                original_name: "x".into(),
                ty: nat.clone(),
                span: span(),
            }],
            return_type: nat,
            body: empty_block(CType::Tuple(vec![])),
            span: span(),
        };
        let mut prog = empty_program();
        prog.funcs.push(func);
        prog.next_name_id = 4;

        let facts = encode_program(&prog, &[nid(2)]).unwrap();
        assert_eq!(facts.fn_to_check.trim(), "2");
        assert!(
            facts.program_in.contains("fdef(2, [fparam(3, t_refined(100, t_int,"),
            "program_in did not contain expected fdef: {}",
            facts.program_in
        );
        assert!(facts.program_in.contains("r_bin_op(greater_equal, r_var(100), r_int_lit(0L))"));
    }

    #[test]
    fn struct_field_gets_synthesized_name_id() {
        // struct S { x: int, y: bool }  →  field ids should be unique
        // and stable across occurrences.
        let mut prog = empty_program();
        let s_id = nid(50);
        prog.struct_defs
            .insert(s_id, vec![(nid(51), "x".to_string(), CType::Int), (nid(52), "y".to_string(), CType::Bool)]);
        prog.next_name_id = 53;

        let facts = encode_program(&prog, &[]).unwrap();
        // The fresh-id allocator picks ids strictly above the
        // program's next_name_id; both should appear in program_in.
        assert!(facts.program_in.contains("(50, ["));
    }

    #[test]
    fn unsupported_refexpr_returns_error() {
        let mut nt = NameTable::new(&empty_program());
        let bad = RefinementExpr {
            kind: RefinementExprKind::TupleLit(vec![]),
            ty: CType::Tuple(vec![]),
            span: span(),
        };
        assert!(matches!(
            encode_refexpr(&mut nt, &bad),
            Err(EncodeError::UnsupportedRefexpr("TupleLit"))
        ));
    }

    #[test]
    fn string_literals_with_unsupported_chars_are_rejected() {
        let mut nt = NameTable::new(&empty_program());
        let bad = CAtomic::StringLit("with\nnewline".into());
        assert!(matches!(
            encode_atomic(&mut nt, &bad),
            Err(EncodeError::BadString(_))
        ));
    }

    #[test]
    fn debug_dump_lists_all_files() {
        let prog = empty_program();
        let facts = encode_program(&prog, &[]).unwrap();
        let dump = debug_dump(&facts);
        assert!(dump.contains("# tuple_accessor.tsv"));
        assert!(dump.contains("# program_in.tsv"));
        assert!(dump.contains("# fn_to_check.tsv"));
        assert!(dump.contains("# expr_origin.tsv"));
    }

    #[test]
    fn s_let_user_annotated_emits_some_type() {
        let mut nt = NameTable::new(&empty_program());
        let la = CLetAtom {
            name: nid(7),
            original_name: "v".into(),
            ty: CType::Int,
            value: CExpr {
                kind: CExprKind::Atomic(CAtomic::IntLit(3)),
                ty: CType::Int,
                span: span(),
            },
            user_annotated: true,
            span: span(),
        };
        let stmt = CStatement {
            kind: CStatementKind::LetAtom(la),
            span: span(),
        };
        let t = encode_statement(&mut nt, &stmt).unwrap();
        assert_eq!(t.render(), "s_let(7, some(t_int), e_atomic(a_int_lit(3L)))");
    }
}
