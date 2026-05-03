//! Post-lowering validation pass over [`CProgram`] for refinement bodies.

use std::collections::HashSet;

use crate::ir::{
    CBinOp, CBlock, CCondExpr, CExpr, CExprKind, CProgram, CRefinementBody, CRefinementHandle,
    CStatement, CStatementKind, CType,
};
use crate::lower::{RefinementValidationError, RefinementValidationErrorKind};
use crate::refinement::{
    RefinementCond, RefinementExpr, RefinementExprKind, is_constant_int,
};

/// Run all refinement-body theory checks against `program` and return any
/// errors found. Each unique refinement body is checked exactly once even if
/// it appears in many type slots.
pub fn validate_refinements(program: &CProgram) -> Vec<RefinementValidationError> {
    let mut collector = HandleCollector {
        seen: HashSet::new(),
        bodies: Vec::new(),
    };
    collector.walk_program(program);

    let mut errors = Vec::new();
    for handle in &collector.bodies {
        check_linearity(&handle.body, &mut errors);
    }
    errors
}

/// Collects every unique [`CRefinementHandle`] reachable from a [`CProgram`]
/// by walking every type slot and recursively descending into types that
/// nest others.
struct HandleCollector {
    seen: HashSet<*const CRefinementBody>,
    bodies: Vec<CRefinementHandle>,
}

impl HandleCollector {
    fn walk_program(&mut self, p: &CProgram) {
        for f in &p.funcs {
            for param in &f.params {
                self.walk_type(&param.ty);
            }
            self.walk_type(&f.return_type);
            self.walk_block(&f.body);
        }
        for ext in &p.extern_funcs {
            for param in &ext.params {
                self.walk_type(&param.ty);
            }
            self.walk_type(&ext.return_type);
        }
        for fields in p.struct_defs.values() {
            for (_, _, ty) in fields {
                self.walk_type(ty);
            }
        }
        for variants in p.enum_defs.values() {
            for (_, _, payload) in variants {
                if let Some(ty) = payload {
                    self.walk_type(ty);
                }
            }
        }
    }

    fn walk_type(&mut self, ty: &CType) {
        match ty {
            CType::Refined(inner, h) => {
                let first_time = self.seen.insert(h.as_ptr());
                if first_time {
                    self.bodies.push(h.clone());
                }
                self.walk_type(inner);
                if first_time {
                    self.walk_refinement_expr(&h.body);
                }
            }
            CType::Array(t)
            | CType::Optional(t)
            | CType::Chan(t)
            | CType::FifoLink(t)
            | CType::Iter(t) => self.walk_type(t),
            CType::Map(k, v) => {
                self.walk_type(k);
                self.walk_type(v);
            }
            CType::Tuple(ts) => {
                for t in ts {
                    self.walk_type(t);
                }
            }
            CType::Int
            | CType::Bool
            | CType::String
            | CType::Nil
            | CType::Never
            | CType::Role(_)
            | CType::Struct(_)
            | CType::Variant(_) => {}
        }
    }

    fn walk_block(&mut self, b: &CBlock) {
        self.walk_type(&b.ty);
        for s in &b.statements {
            self.walk_statement(s);
        }
    }

    fn walk_statement(&mut self, s: &CStatement) {
        match &s.kind {
            CStatementKind::LetAtom(la) => {
                self.walk_type(&la.ty);
                self.walk_expr(&la.value);
            }
            CStatementKind::Expr(e) => self.walk_expr(e),
            CStatementKind::Return(_) | CStatementKind::Error => {}
        }
    }

    fn walk_expr(&mut self, e: &CExpr) {
        self.walk_type(&e.ty);
        match &e.kind {
            CExprKind::FuncCall(c) => self.walk_type(&c.return_type),
            CExprKind::Conditional(c) => self.walk_cond(c),
            CExprKind::Block(b) => self.walk_block(b),
            CExprKind::Atomic(_)
            | CExprKind::BinOp(_, _, _)
            | CExprKind::Not(_)
            | CExprKind::Negate(_)
            | CExprKind::TupleLit(_)
            | CExprKind::StructLit(_, _)
            | CExprKind::VariantLit(_, _, _)
            | CExprKind::IsVariant(_, _, _)
            | CExprKind::VariantPayload(_)
            | CExprKind::TupleAccess(_, _)
            | CExprKind::FieldAccess(_, _) => {}
        }
    }

    fn walk_cond(&mut self, c: &CCondExpr) {
        self.walk_block(&c.if_branch.body);
        for b in &c.elseif_branches {
            self.walk_block(&b.body);
        }
        if let Some(b) = &c.else_branch {
            self.walk_block(b);
        }
    }

    /// A refinement body's nodes carry [`CType`]s of their own; if any of
    /// those reference further [`CRefinementHandle`]s, collect those too so
    /// we don't miss nested refinements.
    fn walk_refinement_expr(&mut self, e: &RefinementExpr) {
        self.walk_type(&e.ty);
        match &e.kind {
            RefinementExprKind::BinOp(_, l, r) => {
                self.walk_refinement_expr(l);
                self.walk_refinement_expr(r);
            }
            RefinementExprKind::Not(e) | RefinementExprKind::Negate(e) => {
                self.walk_refinement_expr(e);
            }
            RefinementExprKind::ExternCall {
                args, return_type, ..
            } => {
                self.walk_type(return_type);
                for a in args {
                    self.walk_refinement_expr(a);
                }
            }
            RefinementExprKind::TupleLit(es) => {
                for x in es {
                    self.walk_refinement_expr(x);
                }
            }
            RefinementExprKind::StructLit(_, fields) => {
                for (_, x) in fields {
                    self.walk_refinement_expr(x);
                }
            }
            RefinementExprKind::VariantLit(_, _, payload) => {
                if let Some(x) = payload {
                    self.walk_refinement_expr(x);
                }
            }
            RefinementExprKind::IsVariant(e, _, _) | RefinementExprKind::VariantPayload(e) => {
                self.walk_refinement_expr(e);
            }
            RefinementExprKind::TupleAccess(e, _) | RefinementExprKind::FieldAccess(e, _) => {
                self.walk_refinement_expr(e);
            }
            RefinementExprKind::Conditional(c) => self.walk_refinement_cond(c),
            RefinementExprKind::Var(_, _)
            | RefinementExprKind::IntLit(_)
            | RefinementExprKind::StringLit(_)
            | RefinementExprKind::BoolLit(_)
            | RefinementExprKind::NilLit
            | RefinementExprKind::Error => {}
        }
    }

    fn walk_refinement_cond(&mut self, c: &RefinementCond) {
        self.walk_refinement_expr(&c.if_branch.condition);
        self.walk_refinement_expr(&c.if_branch.body);
        for b in &c.elseif_branches {
            self.walk_refinement_expr(&b.condition);
            self.walk_refinement_expr(&b.body);
        }
        if let Some(e) = &c.else_branch {
            self.walk_refinement_expr(e);
        }
    }
}

/// Walk a single refinement body, flagging every `*` / `/` / `%` whose
/// operands are both non-constant.
fn check_linearity(body: &RefinementExpr, errors: &mut Vec<RefinementValidationError>) {
    walk_check(body, errors);
}

fn walk_check(e: &RefinementExpr, errs: &mut Vec<RefinementValidationError>) {
    if let RefinementExprKind::BinOp(op, l, r) = &e.kind
        && matches!(op, CBinOp::Multiply | CBinOp::Divide | CBinOp::Modulo)
        && !is_constant_int(l)
        && !is_constant_int(r)
    {
        errs.push(RefinementValidationError {
            kind: RefinementValidationErrorKind::NonLinearArithmetic { op: *op },
            span: e.span,
        });
    }
    match &e.kind {
        RefinementExprKind::BinOp(_, l, r) => {
            walk_check(l, errs);
            walk_check(r, errs);
        }
        RefinementExprKind::Not(e) | RefinementExprKind::Negate(e) => walk_check(e, errs),
        RefinementExprKind::ExternCall { args, .. } => {
            for a in args {
                walk_check(a, errs);
            }
        }
        RefinementExprKind::TupleLit(es) => {
            for x in es {
                walk_check(x, errs);
            }
        }
        RefinementExprKind::StructLit(_, fields) => {
            for (_, x) in fields {
                walk_check(x, errs);
            }
        }
        RefinementExprKind::VariantLit(_, _, payload) => {
            if let Some(x) = payload {
                walk_check(x, errs);
            }
        }
        RefinementExprKind::IsVariant(e, _, _) | RefinementExprKind::VariantPayload(e) => {
            walk_check(e, errs);
        }
        RefinementExprKind::TupleAccess(e, _) | RefinementExprKind::FieldAccess(e, _) => {
            walk_check(e, errs);
        }
        RefinementExprKind::Conditional(c) => {
            walk_check(&c.if_branch.condition, errs);
            walk_check(&c.if_branch.body, errs);
            for b in &c.elseif_branches {
                walk_check(&b.condition, errs);
                walk_check(&b.body, errs);
            }
            if let Some(e) = &c.else_branch {
                walk_check(e, errs);
            }
        }
        RefinementExprKind::Var(_, _)
        | RefinementExprKind::IntLit(_)
        | RefinementExprKind::StringLit(_)
        | RefinementExprKind::BoolLit(_)
        | RefinementExprKind::NilLit
        | RefinementExprKind::Error => {}
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::ir::{CBinOp, CRefinementBody, CRefinementHandle, CType};
    use crate::refinement::{
        RefinementCond, RefinementExpr, RefinementExprKind, RefinementIfBranch,
    };
    use spur_ast::name::NameId;
    use spur_ast::span::Span;

    fn span() -> Span {
        Span::default()
    }

    fn nid(n: usize) -> NameId {
        NameId(n)
    }

    fn int_lit(v: i64) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::IntLit(v),
            ty: CType::Int,
            span: span(),
        }
    }

    fn var(id: usize, name: &str) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::Var(nid(id), name.to_string()),
            ty: CType::Int,
            span: span(),
        }
    }

    fn bool_lit(b: bool) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::BoolLit(b),
            ty: CType::Bool,
            span: span(),
        }
    }

    fn binop(op: CBinOp, l: RefinementExpr, r: RefinementExpr, ty: CType) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::BinOp(op, Box::new(l), Box::new(r)),
            ty,
            span: span(),
        }
    }

    fn negate(e: RefinementExpr) -> RefinementExpr {
        RefinementExpr {
            kind: RefinementExprKind::Negate(Box::new(e)),
            ty: CType::Int,
            span: span(),
        }
    }

    /// Build a `CProgram` whose only refinement is one struct field carrying
    /// `Refined(int, { x | <body> })`.
    fn program_with_body(body: RefinementExpr) -> CProgram {
        let handle = CRefinementHandle::new(CRefinementBody {
            bound: nid(1),
            original_bound: "x".to_string(),
            body,
        });
        let refined = CType::Refined(Box::new(CType::Int), handle);
        let mut struct_defs = HashMap::new();
        struct_defs.insert(nid(2), vec![(nid(99), "f".to_string(), refined)]);
        CProgram {
            funcs: vec![],
            extern_funcs: vec![],
            struct_defs,
            enum_defs: HashMap::new(),
            next_name_id: 3,
            id_to_name: HashMap::new(),
        }
    }

    #[test]
    fn accepts_x_times_const() {
        // x * 2
        let body = binop(CBinOp::Multiply, var(1, "x"), int_lit(2), CType::Int);
        let errors = validate_refinements(&program_with_body(body));
        assert!(errors.is_empty(), "unexpected: {:?}", errors);
    }

    #[test]
    fn accepts_const_times_x() {
        // 2 * x
        let body = binop(CBinOp::Multiply, int_lit(2), var(1, "x"), CType::Int);
        let errors = validate_refinements(&program_with_body(body));
        assert!(errors.is_empty(), "unexpected: {:?}", errors);
    }

    #[test]
    fn accepts_compound_constant_times_x() {
        // (2 + 3) * x — accepted because the LHS is recognized as a constant
        // expression. No constant folding is performed; the IR stays as a
        // BinOp(Add, IntLit, IntLit) on the LHS.
        let lhs = binop(CBinOp::Add, int_lit(2), int_lit(3), CType::Int);
        let body = binop(CBinOp::Multiply, lhs, var(1, "x"), CType::Int);
        let errors = validate_refinements(&program_with_body(body));
        assert!(errors.is_empty(), "unexpected: {:?}", errors);
    }

    #[test]
    fn accepts_negated_constant_times_x() {
        // -1 * x
        let body = binop(CBinOp::Multiply, negate(int_lit(1)), var(1, "x"), CType::Int);
        let errors = validate_refinements(&program_with_body(body));
        assert!(errors.is_empty(), "unexpected: {:?}", errors);
    }

    #[test]
    fn accepts_x_div_const_and_mod_const() {
        let div = binop(CBinOp::Divide, var(1, "x"), int_lit(2), CType::Int);
        let modulo = binop(CBinOp::Modulo, var(1, "x"), int_lit(3), CType::Int);
        for body in [div, modulo] {
            let errors = validate_refinements(&program_with_body(body));
            assert!(errors.is_empty(), "unexpected: {:?}", errors);
        }
    }

    #[test]
    fn rejects_x_times_y() {
        let body = binop(CBinOp::Multiply, var(1, "x"), var(2, "y"), CType::Int);
        let errors = validate_refinements(&program_with_body(body));
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0].kind,
            RefinementValidationErrorKind::NonLinearArithmetic {
                op: CBinOp::Multiply
            }
        ));
    }

    #[test]
    fn rejects_n_div_x() {
        // n / x where both are variables.
        let body = binop(CBinOp::Divide, var(2, "n"), var(1, "x"), CType::Int);
        let errors = validate_refinements(&program_with_body(body));
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0].kind,
            RefinementValidationErrorKind::NonLinearArithmetic {
                op: CBinOp::Divide
            }
        ));
    }

    #[test]
    fn rejects_x_mod_y() {
        let body = binop(CBinOp::Modulo, var(1, "x"), var(2, "y"), CType::Int);
        let errors = validate_refinements(&program_with_body(body));
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0].kind,
            RefinementValidationErrorKind::NonLinearArithmetic {
                op: CBinOp::Modulo
            }
        ));
    }

    #[test]
    fn rejects_non_linear_inside_conditional() {
        // if x > 0 then x * y else 0
        let cond = binop(CBinOp::Greater, var(1, "x"), int_lit(0), CType::Bool);
        let then_branch = binop(CBinOp::Multiply, var(1, "x"), var(2, "y"), CType::Int);
        let else_branch = int_lit(0);
        let cond_expr = RefinementExpr {
            kind: RefinementExprKind::Conditional(Box::new(RefinementCond {
                if_branch: RefinementIfBranch {
                    condition: cond,
                    body: then_branch,
                    span: span(),
                },
                elseif_branches: vec![],
                else_branch: Some(else_branch),
                span: span(),
            })),
            ty: CType::Int,
            span: span(),
        };
        // Wrap so the body type matches `bool`-shaped predicate convention,
        // though the validator doesn't care about predicate shape.
        let body = binop(CBinOp::Greater, cond_expr, int_lit(0), CType::Bool);
        let errors = validate_refinements(&program_with_body(body));
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0].kind,
            RefinementValidationErrorKind::NonLinearArithmetic {
                op: CBinOp::Multiply
            }
        ));
    }

    #[test]
    fn dedups_handle_reachable_from_multiple_slots() {
        // A single CRefinementHandle reused in a struct field, an enum
        // payload, a function param, a function return type, and as the
        // element type of an array. The validator must visit its body
        // exactly once.
        let body = binop(CBinOp::Multiply, var(1, "x"), var(2, "y"), CType::Int);
        let handle = CRefinementHandle::new(CRefinementBody {
            bound: nid(1),
            original_bound: "x".to_string(),
            body,
        });
        let refined = CType::Refined(Box::new(CType::Int), handle);

        use crate::ir::{
            CBlock, CFuncDef, CFuncKind, CFuncParam,
        };
        let func = CFuncDef {
            name: nid(10),
            original_name: "f".to_string(),
            kind: CFuncKind::Sync,
            is_traced: false,
            role: None,
            params: vec![CFuncParam {
                name: nid(11),
                original_name: "p".to_string(),
                ty: CType::Array(Box::new(refined.clone())),
                span: span(),
            }],
            return_type: refined.clone(),
            body: CBlock {
                statements: vec![],
                tail_expr: None,
                ty: CType::Tuple(vec![]),
                span: span(),
            },
            span: span(),
        };

        let mut struct_defs = HashMap::new();
        struct_defs.insert(nid(20), vec![(nid(100), "field".to_string(), refined.clone())]);
        let mut enum_defs = HashMap::new();
        enum_defs.insert(nid(21), vec![(nid(101), "V".to_string(), Some(refined))]);

        let prog = CProgram {
            funcs: vec![func],
            extern_funcs: vec![],
            struct_defs,
            enum_defs,
            next_name_id: 30,
            id_to_name: HashMap::new(),
        };

        let errors = validate_refinements(&prog);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error after dedup, got {:?}",
            errors
        );
    }

    #[test]
    fn empty_program_yields_no_errors() {
        let prog = CProgram {
            funcs: vec![],
            extern_funcs: vec![],
            struct_defs: HashMap::new(),
            enum_defs: HashMap::new(),
            next_name_id: 0,
            id_to_name: HashMap::new(),
        };
        assert!(validate_refinements(&prog).is_empty());
    }

    #[test]
    fn linear_body_in_predicate_position_passes() {
        // Realistic "predicate-shaped" body: `x * 2 > 0`.
        let lhs = binop(CBinOp::Multiply, var(1, "x"), int_lit(2), CType::Int);
        let body = binop(CBinOp::Greater, lhs, int_lit(0), CType::Bool);
        let _ = bool_lit(true); // keep the helper alive for future tests
        let errors = validate_refinements(&program_with_body(body));
        assert!(errors.is_empty(), "unexpected: {:?}", errors);
    }
}
