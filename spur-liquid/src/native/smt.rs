use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use spur_ast::name::NameId;

use crate::ir::{CBinOp, CRefinementHandle, CType};
use crate::refinement::{RefinementExpr, RefinementExprKind};

use super::context::{Env, env_lookup};
use super::subst::subst_refexpr;

pub struct SmtSolver {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    declared_fns: HashSet<NameId>,
}

impl SmtSolver {
    pub fn new() -> Result<Self, std::io::Error> {
        let mut child = Command::new("z3")
            .arg("-in")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        let mut solver = SmtSolver {
            child,
            stdin,
            stdout,
            declared_fns: HashSet::new(),
        };

        solver.emit("(set-logic UFLIA)\n");
        solver.emit("(declare-fun smt_is_variant (Int Int Int) Bool)\n");

        Ok(solver)
    }

    pub fn check_implication(
        &mut self,
        env: &Env,
        h1: &CRefinementHandle,
        h2: &CRefinementHandle,
    ) -> bool {
        let mut buf = String::new();
        let mut vars = HashSet::new();

        let env_formula = self.env_to_smt(env, &mut vars);
        let r1_renamed = subst_refexpr(
            &RefinementExpr {
                kind: RefinementExprKind::Var(h1.bound, h1.original_bound.clone()),
                ty: CType::Int,
                span: spur_ast::span::Span::default(),
            },
            h1.bound,
            &h1.body,
        );
        let r1_formula = self.refexpr_to_smt(&r1_renamed, SmtSort::Bool, &mut vars);

        let r2_renamed = subst_refexpr(
            &RefinementExpr {
                kind: RefinementExprKind::Var(h1.bound, h1.original_bound.clone()),
                ty: CType::Int,
                span: spur_ast::span::Span::default(),
            },
            h2.bound,
            &h2.body,
        );
        let r2_formula = self.refexpr_to_smt(&r2_renamed, SmtSort::Bool, &mut vars);

        buf.push_str("(push)\n");
        for var in &vars {
            let sort = self.var_sort(*var, env);
            buf.push_str(&format!("(declare-const v_{} {})\n", var.0 as i32, sort));
        }

        let antecedent = if env_formula == "true" {
            r1_formula.clone()
        } else {
            format!("(and {} {})", env_formula, r1_formula)
        };
        buf.push_str(&format!(
            "(assert (not (=> {} {})))\n",
            antecedent, r2_formula
        ));
        buf.push_str("(check-sat)\n");

        self.emit(&buf);
        let result = self.read_response();

        self.emit("(pop)\n");

        result == "unsat"
    }

    fn env_to_smt(&mut self, env: &Env, vars: &mut HashSet<NameId>) -> String {
        let mut conjuncts = Vec::new();
        for (id, ty) in env {
            if let CType::Refined(_, handle) = ty {
                let renamed = subst_refexpr(
                    &RefinementExpr {
                        kind: RefinementExprKind::Var(*id, String::new()),
                        ty: CType::Int,
                        span: spur_ast::span::Span::default(),
                    },
                    handle.bound,
                    &handle.body,
                );
                let smt = self.refexpr_to_smt(&renamed, SmtSort::Bool, vars);
                conjuncts.push(smt);
            }
        }
        match conjuncts.len() {
            0 => "true".to_string(),
            1 => conjuncts.into_iter().next().unwrap(),
            _ => format!("(and {})", conjuncts.join(" ")),
        }
    }

    fn refexpr_to_smt(
        &mut self,
        expr: &RefinementExpr,
        expected: SmtSort,
        vars: &mut HashSet<NameId>,
    ) -> String {
        match &expr.kind {
            RefinementExprKind::Var(id, _) => {
                vars.insert(*id);
                format!("v_{}", id.0 as i32)
            }
            RefinementExprKind::IntLit(n) => format!("{}", n),
            RefinementExprKind::BoolLit(b) => format!("{}", b),
            RefinementExprKind::NilLit => "0".to_string(),

            RefinementExprKind::BinOp(op, lhs, rhs) => {
                self.binop_to_smt(*op, lhs, rhs, vars)
            }

            RefinementExprKind::Not(inner) => {
                let s = self.refexpr_to_smt(inner, SmtSort::Bool, vars);
                format!("(not {})", s)
            }
            RefinementExprKind::Negate(inner) => {
                let s = self.refexpr_to_smt(inner, SmtSort::Int, vars);
                format!("(- 0 {})", s)
            }

            RefinementExprKind::ExternCall {
                target,
                args,
                return_type,
            } => {
                self.ensure_func_declared(*target, args.len(), return_type);
                let arg_strs: Vec<String> = args
                    .iter()
                    .map(|a| self.refexpr_to_smt(a, SmtSort::Int, vars))
                    .collect();
                if arg_strs.is_empty() {
                    format!("f_{}", target.0)
                } else {
                    format!("(f_{} {})", target.0, arg_strs.join(" "))
                }
            }

            RefinementExprKind::IsVariant(scrutinee, eid, vid) => {
                let s = self.refexpr_to_smt(scrutinee, SmtSort::Int, vars);
                format!("(smt_is_variant {} {} {})", s, eid.0, vid.0)
            }

            RefinementExprKind::TupleAccess(inner, _idx) => {
                let s = self.refexpr_to_smt(inner, SmtSort::Int, vars);
                format!("{}", s)
            }
            RefinementExprKind::FieldAccess(inner, _field) => {
                let s = self.refexpr_to_smt(inner, SmtSort::Int, vars);
                format!("{}", s)
            }

            _ => match expected {
                SmtSort::Bool => "true".to_string(),
                SmtSort::Int => "0".to_string(),
            },
        }
    }

    fn binop_to_smt(
        &mut self,
        op: CBinOp,
        lhs: &RefinementExpr,
        rhs: &RefinementExpr,
        vars: &mut HashSet<NameId>,
    ) -> String {
        match op {
            CBinOp::Add => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(+ {} {})", l, r)
            }
            CBinOp::Subtract => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(- {} {})", l, r)
            }
            CBinOp::Multiply => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(* {} {})", l, r)
            }
            CBinOp::Divide => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(div {} {})", l, r)
            }
            CBinOp::Modulo => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(mod {} {})", l, r)
            }
            CBinOp::Less => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(< {} {})", l, r)
            }
            CBinOp::LessEqual => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(<= {} {})", l, r)
            }
            CBinOp::Greater => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(> {} {})", l, r)
            }
            CBinOp::GreaterEqual => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(>= {} {})", l, r)
            }
            CBinOp::And => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Bool, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Bool, vars);
                format!("(and {} {})", l, r)
            }
            CBinOp::Or => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Bool, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Bool, vars);
                format!("(or {} {})", l, r)
            }
            CBinOp::IntEq => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(= {} {})", l, r)
            }
            CBinOp::IntNeq => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Int, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Int, vars);
                format!("(not (= {} {}))", l, r)
            }
            CBinOp::BoolEq => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Bool, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Bool, vars);
                format!("(= {} {})", l, r)
            }
            CBinOp::BoolNeq => {
                let l = self.refexpr_to_smt(lhs, SmtSort::Bool, vars);
                let r = self.refexpr_to_smt(rhs, SmtSort::Bool, vars);
                format!("(not (= {} {}))", l, r)
            }
        }
    }

    fn ensure_func_declared(&mut self, name: NameId, arity: usize, return_type: &CType) {
        if self.declared_fns.contains(&name) {
            return;
        }
        self.declared_fns.insert(name);
        let ret_sort = if matches!(base_type_of(return_type), CType::Bool) {
            "Bool"
        } else {
            "Int"
        };
        let param_sorts = vec!["Int"; arity].join(" ");
        let decl = format!("(declare-fun f_{} ({}) {})\n", name.0, param_sorts, ret_sort);
        self.emit(&decl);
    }

    fn var_sort(&self, var: NameId, env: &Env) -> &'static str {
        if let Some(ty) = env_lookup(var, env) {
            if matches!(base_type_of(ty), CType::Bool) {
                return "Bool";
            }
        }
        "Int"
    }

    fn emit(&mut self, s: &str) {
        let _ = self.stdin.write_all(s.as_bytes());
        let _ = self.stdin.flush();
    }

    fn read_response(&mut self) -> String {
        let mut line = String::new();
        let _ = self.stdout.read_line(&mut line);
        line.trim().to_string()
    }
}

impl Drop for SmtSolver {
    fn drop(&mut self) {
        let _ = self.emit("(exit)\n");
        let _ = self.child.wait();
    }
}

fn base_type_of(ty: &CType) -> &CType {
    match ty {
        CType::Refined(base, _) => base_type_of(base),
        _ => ty,
    }
}

#[derive(Clone, Copy)]
enum SmtSort {
    Bool,
    Int,
}
