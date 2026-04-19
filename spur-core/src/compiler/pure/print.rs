//! Pure IR pretty-printer.
//!
//! Renders a [`PProgram`] into a stable, human-readable text form suitable for
//! writing to a `.pir` file. The output is intentionally one-way: there is no
//! corresponding parser. The format is designed for diffing across runs, so
//! all `HashMap` iteration is sorted by `NameId`.

use std::collections::{HashMap, HashSet};

use crate::analysis::resolver::{BuiltinFn, NameId};
use crate::analysis::types::Type;
use crate::liquid::pure::ast::*;

const INDENT: &str = "  ";

/// Render `program` to a `.pir` text representation.
pub fn print_program(program: &PProgram) -> String {
    let mut printer = Printer::new(&program.id_to_name);
    printer.write_program(program);
    printer.buf
}

struct Printer<'a> {
    buf: String,
    indent: usize,
    /// Resolves struct/enum/role NameIds to their user-facing names.
    name_lookup: &'a HashMap<NameId, String>,
    /// Per-function disambiguation context. Names whose `NameId` is in
    /// `disamb.needs_id` are printed as `<name>#<id>` to distinguish from
    /// other ids that share the same `original_name`. Outside a function
    /// (e.g. top-level role and free-function names) this stays empty so
    /// program-scope names are rendered without ids.
    disamb: NameDisamb,
}

impl<'a> Printer<'a> {
    fn new(name_lookup: &'a HashMap<NameId, String>) -> Self {
        Self {
            buf: String::new(),
            indent: 0,
            name_lookup,
            disamb: NameDisamb::empty(),
        }
    }

    fn push_indent(&mut self) {
        for _ in 0..self.indent {
            self.buf.push_str(INDENT);
        }
    }

    fn writeln(&mut self, s: &str) {
        self.push_indent();
        self.buf.push_str(s);
        self.buf.push('\n');
    }

    fn blank_line(&mut self) {
        self.buf.push('\n');
    }

    /// Look up the user-facing name for a struct/enum/role NameId. Falls back
    /// to `<fallback>#<id>` if the lookup misses (defensive — shouldn't
    /// happen for well-formed programs).
    fn type_name(&self, nid: NameId, fallback: &str) -> String {
        self.name_lookup
            .get(&nid)
            .cloned()
            .unwrap_or_else(|| format!("{}#{}", fallback, nid.0))
    }

    /// Render a name that may need disambiguation. Inside a function this
    /// consults `self.disamb`; for program-scope names the default (empty)
    /// disamb means we always emit just `name`.
    fn fmt_name(&self, name: &str, id: NameId) -> String {
        if self.disamb.needs_id.contains(&id) {
            format!("{}#{}", name, id.0)
        } else {
            name.to_string()
        }
    }

    fn write_program(&mut self, p: &PProgram) {
        self.writeln("# Pure IR / SSA");
        self.blank_line();

        self.write_struct_defs(&p.struct_defs);
        self.write_enum_defs(&p.enum_defs);

        for def in &p.top_level_defs {
            self.write_top_level(def);
            self.blank_line();
        }
    }

    fn write_struct_defs(&mut self, defs: &HashMap<NameId, Vec<(String, Type)>>) {
        if defs.is_empty() {
            return;
        }
        let mut entries: Vec<(&NameId, &Vec<(String, Type)>)> = defs.iter().collect();
        entries.sort_by_key(|(nid, _)| nid.0);
        for (nid, fields) in entries {
            let name = self.type_name(*nid, "Struct");
            let fields_str = fields
                .iter()
                .map(|(n, t)| format!("{}: {}", n, t))
                .collect::<Vec<_>>()
                .join(", ");
            self.writeln(&format!("struct {} {{ {} }}", name, fields_str));
        }
        self.blank_line();
    }

    fn write_enum_defs(&mut self, defs: &HashMap<NameId, Vec<(String, Option<Type>)>>) {
        if defs.is_empty() {
            return;
        }
        let mut entries: Vec<(&NameId, &Vec<(String, Option<Type>)>)> = defs.iter().collect();
        entries.sort_by_key(|(nid, _)| nid.0);
        for (nid, variants) in entries {
            let name = self.type_name(*nid, "Enum");
            let variants_str = variants
                .iter()
                .map(|(vname, payload)| match payload {
                    Some(t) => format!("{}({})", vname, t),
                    None => vname.clone(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            self.writeln(&format!("enum {} {{ {} }}", name, variants_str));
        }
        self.blank_line();
    }

    fn write_top_level(&mut self, def: &PTopLevelDef) {
        match def {
            PTopLevelDef::FreeFunc(f) => self.write_func(f),
            PTopLevelDef::Role(r) => self.write_role(r),
        }
    }

    fn write_role(&mut self, r: &PRoleDef) {
        // Role name lives at program scope (empty disamb), so just use the
        // type-name lookup or the original_name directly.
        self.writeln(&format!("role {} {{", self.fmt_name(&r.original_name, r.name)));
        self.indent += 1;
        for (i, f) in r.func_defs.iter().enumerate() {
            if i > 0 {
                self.blank_line();
            }
            self.write_func(f);
        }
        self.indent -= 1;
        self.writeln("}");
    }

    fn write_func(&mut self, f: &PFuncDef) {
        // Install per-function disambiguation, restoring the previous one on
        // exit. Functions don't nest in Pure IR, so the previous disamb is
        // always the empty (program-scope) one — but the swap pattern is
        // robust regardless.
        let prev = std::mem::replace(&mut self.disamb, NameDisamb::for_func(f));

        let mut tags: Vec<&str> = Vec::new();
        match f.kind {
            PFuncKind::Sync => {}
            PFuncKind::Async => tags.push("async"),
            PFuncKind::LoopConverted => tags.push("loop"),
        }
        if f.is_traced {
            tags.push("traced");
        }
        let tag_str = if tags.is_empty() {
            String::new()
        } else {
            format!("[{}] ", tags.join(", "))
        };

        let params = f
            .params
            .iter()
            .map(|p| format!("{}: {}", self.fmt_name(&p.original_name, p.name), p.ty))
            .collect::<Vec<_>>()
            .join(", ");

        self.writeln(&format!(
            "{}fn {}({}) -> {} {{",
            tag_str,
            self.fmt_name(&f.original_name, f.name),
            params,
            f.return_type,
        ));
        self.indent += 1;
        self.write_block_body(&f.body);
        self.indent -= 1;
        self.writeln("}");

        self.disamb = prev;
    }

    /// Write the contents of a block (statements + optional tail expr) without
    /// the surrounding braces.
    fn write_block_body(&mut self, b: &PBlock) {
        for stmt in &b.statements {
            self.write_statement(stmt);
        }
        if let Some(tail) = &b.tail_expr {
            let s = self.fmt_atomic(tail);
            self.writeln(&s);
        }
    }

    fn write_statement(&mut self, s: &PStatement) {
        match &s.kind {
            PStatementKind::LetAtom(la) => self.write_let(la),
            PStatementKind::Expr(e) => self.write_expr_stmt(e),
            PStatementKind::Return(a) => {
                let s = format!("return {}", self.fmt_atomic(a));
                self.writeln(&s);
            }
            PStatementKind::Error => self.writeln("error"),
        }
    }

    fn write_let(&mut self, la: &PLetAtom) {
        let header = format!(
            "let {}: {} =",
            self.fmt_name(&la.original_name, la.name),
            la.ty
        );
        self.write_expr_with_header(&header, &la.value);
    }

    fn write_expr_stmt(&mut self, e: &PExpr) {
        // Statement-position expression: no binding, no `=`.
        self.write_expr_with_header("", e);
    }

    /// Write an expression that may either fit on one line (after `header`) or
    /// expand into a multi-line block / conditional. `header` ends without a
    /// trailing space; we add one when the expression body is non-empty.
    fn write_expr_with_header(&mut self, header: &str, e: &PExpr) {
        match &e.kind {
            PExprKind::Conditional(cond) => {
                if header.is_empty() {
                    self.write_conditional(cond);
                } else {
                    self.writeln(header);
                    self.write_conditional(cond);
                }
            }
            PExprKind::Block(b) => {
                if header.is_empty() {
                    self.writeln("{");
                } else {
                    self.writeln(&format!("{} {{", header));
                }
                self.indent += 1;
                self.write_block_body(b);
                self.indent -= 1;
                self.writeln("}");
            }
            _ => {
                let body = self.fmt_simple_expr(&e.kind);
                if header.is_empty() {
                    self.writeln(&body);
                } else {
                    self.writeln(&format!("{} {}", header, body));
                }
            }
        }
    }

    fn write_conditional(&mut self, cond: &PCondExpr) {
        let cond_str = self.fmt_atomic(&cond.if_branch.condition);
        self.writeln(&format!("if {} {{", cond_str));
        self.indent += 1;
        self.write_block_body(&cond.if_branch.body);
        self.indent -= 1;

        for branch in &cond.elseif_branches {
            let cs = self.fmt_atomic(&branch.condition);
            self.writeln(&format!("}} else if {} {{", cs));
            self.indent += 1;
            self.write_block_body(&branch.body);
            self.indent -= 1;
        }

        if let Some(else_b) = &cond.else_branch {
            self.writeln("} else {");
            self.indent += 1;
            self.write_block_body(else_b);
            self.indent -= 1;
        }

        self.writeln("}");
    }

    // ===== expression / atomic formatters =====

    fn fmt_atomic(&self, a: &PAtomic) -> String {
        match a {
            PAtomic::Var(id, name) => self.fmt_name(name, *id),
            PAtomic::IntLit(v) => v.to_string(),
            PAtomic::StringLit(v) => format!("{:?}", v),
            PAtomic::BoolLit(v) => v.to_string(),
            PAtomic::NilLit => "nil".to_string(),
            PAtomic::Never => "!".to_string(),
        }
    }

    fn fmt_atomic_list(&self, atoms: &[PAtomic]) -> String {
        atoms
            .iter()
            .map(|a| self.fmt_atomic(a))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn fmt_user_call(&self, call: &PUserFuncCall) -> String {
        format!(
            "{}({})",
            self.fmt_name(&call.original_name, call.name),
            self.fmt_atomic_list(&call.args),
        )
    }

    fn fmt_func_call(&self, call: &PFuncCall) -> String {
        match call {
            PFuncCall::User(u) => self.fmt_user_call(u),
            PFuncCall::Builtin(b, args, _) => {
                format!("@{}({})", fmt_builtin(*b), self.fmt_atomic_list(args))
            }
        }
    }

    /// Render an expression kind that does not contain a Conditional or Block.
    fn fmt_simple_expr(&self, kind: &PExprKind) -> String {
        match kind {
            PExprKind::Atomic(a) => self.fmt_atomic(a),
            PExprKind::BinOp(op, a, b) => {
                format!("{} {} {}", self.fmt_atomic(a), op, self.fmt_atomic(b))
            }
            PExprKind::Not(a) => format!("!{}", self.fmt_atomic(a)),
            PExprKind::Negate(a) => format!("-{}", self.fmt_atomic(a)),

            PExprKind::FuncCall(c) => self.fmt_func_call(c),

            PExprKind::MapLit(pairs) => {
                let items = pairs
                    .iter()
                    .map(|(k, v)| format!("{}: {}", self.fmt_atomic(k), self.fmt_atomic(v)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{}}}", items)
            }
            PExprKind::ListLit(items) => format!("[{}]", self.fmt_atomic_list(items)),
            PExprKind::TupleLit(items) => format!("({})", self.fmt_atomic_list(items)),

            PExprKind::Append(a, b) => {
                format!("@append({}, {})", self.fmt_atomic(a), self.fmt_atomic(b))
            }
            PExprKind::Prepend(a, b) => {
                format!("@prepend({}, {})", self.fmt_atomic(a), self.fmt_atomic(b))
            }
            PExprKind::Min(a, b) => {
                format!("@min({}, {})", self.fmt_atomic(a), self.fmt_atomic(b))
            }
            PExprKind::Exists(a, b) => {
                format!("@exists({}, {})", self.fmt_atomic(a), self.fmt_atomic(b))
            }
            PExprKind::Erase(a, b) => {
                format!("@erase({}, {})", self.fmt_atomic(a), self.fmt_atomic(b))
            }
            PExprKind::Store(a, b, c) => format!(
                "@store({}, {}, {})",
                self.fmt_atomic(a),
                self.fmt_atomic(b),
                self.fmt_atomic(c)
            ),
            PExprKind::Head(a) => format!("@head({})", self.fmt_atomic(a)),
            PExprKind::Tail(a) => format!("@tail({})", self.fmt_atomic(a)),
            PExprKind::Len(a) => format!("@len({})", self.fmt_atomic(a)),

            PExprKind::RpcCall(target, call) => {
                format!("{}->{}", self.fmt_atomic(target), self.fmt_user_call(call))
            }

            PExprKind::Conditional(_) | PExprKind::Block(_) => {
                // Should be unreachable in this helper; the caller routes
                // these through write_conditional / block writer. Emit a
                // sentinel so a stray call is debuggable.
                "<block-or-cond>".to_string()
            }

            PExprKind::VariantLit(enum_id, variant, payload) => {
                let enum_name = self.type_name(*enum_id, "Enum");
                match payload {
                    Some(p) => format!("{}::{}({})", enum_name, variant, self.fmt_atomic(p)),
                    None => format!("{}::{}", enum_name, variant),
                }
            }
            PExprKind::IsVariant(a, name) => format!("{} is {}", self.fmt_atomic(a), name),
            PExprKind::VariantPayload(a) => format!("@payload({})", self.fmt_atomic(a)),

            PExprKind::UnwrapOptional(a) => format!("{}!", self.fmt_atomic(a)),

            PExprKind::MakeIter(a) => format!("@iter({})", self.fmt_atomic(a)),
            PExprKind::IterIsDone(a) => format!("@iter_done({})", self.fmt_atomic(a)),
            PExprKind::IterNext(a) => format!("@iter_next({})", self.fmt_atomic(a)),

            PExprKind::MakeChannel => "@chan()".to_string(),
            PExprKind::Send(s, c, v) => format!(
                "@send({}, {}, {})",
                self.fmt_atomic(s),
                self.fmt_atomic(c),
                self.fmt_atomic(v)
            ),
            PExprKind::Recv(s, c) => {
                format!("@recv({}, {})", self.fmt_atomic(s), self.fmt_atomic(c))
            }

            PExprKind::SetTimer(label) => match label {
                Some(l) => format!("@set_timer({:?})", l),
                None => "@set_timer()".to_string(),
            },

            PExprKind::Index(a, b) => {
                format!("{}[{}]", self.fmt_atomic(a), self.fmt_atomic(b))
            }
            PExprKind::Slice(a, b, c) => format!(
                "{}[{}..{}]",
                self.fmt_atomic(a),
                self.fmt_atomic(b),
                self.fmt_atomic(c)
            ),
            PExprKind::TupleAccess(a, i) => format!("{}.{}", self.fmt_atomic(a), i),
            PExprKind::FieldAccess(a, name) => format!("{}.{}", self.fmt_atomic(a), name),

            PExprKind::SafeFieldAccess(a, name) => format!("{}?.{}", self.fmt_atomic(a), name),
            PExprKind::SafeIndex(a, b) => {
                format!("{}?[{}]", self.fmt_atomic(a), self.fmt_atomic(b))
            }
            PExprKind::SafeTupleAccess(a, i) => format!("{}?.{}", self.fmt_atomic(a), i),

            PExprKind::StructLit(struct_id, fields) => {
                let struct_name = self.type_name(*struct_id, "Struct");
                let items = fields
                    .iter()
                    .map(|(n, a)| format!("{}: {}", n, self.fmt_atomic(a)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} {{ {} }}", struct_name, items)
            }

            PExprKind::WrapInOptional(a) => format!("@some({})", self.fmt_atomic(a)),
            PExprKind::PersistData(a) => format!("@persist({})", self.fmt_atomic(a)),
            PExprKind::RetrieveData(t) => format!("@retrieve::<{}>()", t),
            PExprKind::DiscardData => "@discard()".to_string(),
        }
    }
}

fn fmt_builtin(b: BuiltinFn) -> &'static str {
    match b {
        BuiltinFn::Println => "println",
        BuiltinFn::IntToString => "int_to_string",
        BuiltinFn::BoolToString => "bool_to_string",
        BuiltinFn::RoleToString => "role_to_string",
        BuiltinFn::UniqueId => "unique_id",
    }
}

/// Records, per function, which `NameId`s share an `original_name` with at
/// least one other `NameId`. Names whose ids appear in `needs_id` are printed
/// as `<name>#<id>` to keep them distinct; everything else prints as bare
/// `<name>`.
struct NameDisamb {
    needs_id: HashSet<NameId>,
}

impl NameDisamb {
    fn empty() -> Self {
        Self {
            needs_id: HashSet::new(),
        }
    }

    fn for_func(f: &PFuncDef) -> Self {
        let mut visitor = DisambVisitor {
            by_name: HashMap::new(),
            needs_id: HashSet::new(),
        };
        visitor.visit_func(f);
        Self {
            needs_id: visitor.needs_id,
        }
    }
}

struct DisambVisitor {
    by_name: HashMap<String, NameId>,
    needs_id: HashSet<NameId>,
}

impl DisambVisitor {
    fn visit(&mut self, id: NameId, name: &str) {
        match self.by_name.get(name) {
            None => {
                self.by_name.insert(name.to_string(), id);
            }
            Some(prev_id) => {
                if *prev_id != id {
                    self.needs_id.insert(*prev_id);
                    self.needs_id.insert(id);
                }
            }
        }
    }

    fn visit_func(&mut self, f: &PFuncDef) {
        self.visit(f.name, &f.original_name);
        for p in &f.params {
            self.visit(p.name, &p.original_name);
        }
        self.visit_block(&f.body);
    }

    fn visit_block(&mut self, b: &PBlock) {
        for s in &b.statements {
            self.visit_stmt(s);
        }
        if let Some(t) = &b.tail_expr {
            self.visit_atomic(t);
        }
    }

    fn visit_stmt(&mut self, s: &PStatement) {
        match &s.kind {
            PStatementKind::LetAtom(la) => {
                self.visit(la.name, &la.original_name);
                self.visit_expr(&la.value);
            }
            PStatementKind::Expr(e) => self.visit_expr(e),
            PStatementKind::Return(a) => self.visit_atomic(a),
            PStatementKind::Error => {}
        }
    }

    fn visit_expr(&mut self, e: &PExpr) {
        match &e.kind {
            PExprKind::Atomic(a) => self.visit_atomic(a),
            PExprKind::BinOp(_, a, b) => {
                self.visit_atomic(a);
                self.visit_atomic(b);
            }
            PExprKind::Not(a) | PExprKind::Negate(a) => self.visit_atomic(a),

            PExprKind::FuncCall(c) => self.visit_func_call(c),

            PExprKind::MapLit(pairs) => {
                for (k, v) in pairs {
                    self.visit_atomic(k);
                    self.visit_atomic(v);
                }
            }
            PExprKind::ListLit(items) | PExprKind::TupleLit(items) => {
                for i in items {
                    self.visit_atomic(i);
                }
            }

            PExprKind::Append(a, b)
            | PExprKind::Prepend(a, b)
            | PExprKind::Min(a, b)
            | PExprKind::Exists(a, b)
            | PExprKind::Erase(a, b) => {
                self.visit_atomic(a);
                self.visit_atomic(b);
            }
            PExprKind::Store(a, b, c) => {
                self.visit_atomic(a);
                self.visit_atomic(b);
                self.visit_atomic(c);
            }
            PExprKind::Head(a) | PExprKind::Tail(a) | PExprKind::Len(a) => self.visit_atomic(a),

            PExprKind::RpcCall(target, call) => {
                self.visit_atomic(target);
                self.visit_user_call(call);
            }

            PExprKind::Conditional(cond) => {
                self.visit_atomic(&cond.if_branch.condition);
                self.visit_block(&cond.if_branch.body);
                for branch in &cond.elseif_branches {
                    self.visit_atomic(&branch.condition);
                    self.visit_block(&branch.body);
                }
                if let Some(else_b) = &cond.else_branch {
                    self.visit_block(else_b);
                }
            }
            PExprKind::Block(b) => self.visit_block(b),

            PExprKind::VariantLit(_, _, payload) => {
                if let Some(p) = payload {
                    self.visit_atomic(p);
                }
            }
            PExprKind::IsVariant(a, _) | PExprKind::VariantPayload(a) => self.visit_atomic(a),

            PExprKind::UnwrapOptional(a) => self.visit_atomic(a),

            PExprKind::MakeIter(a) | PExprKind::IterIsDone(a) | PExprKind::IterNext(a) => {
                self.visit_atomic(a)
            }

            PExprKind::MakeChannel => {}
            PExprKind::Send(s, c, v) => {
                self.visit_atomic(s);
                self.visit_atomic(c);
                self.visit_atomic(v);
            }
            PExprKind::Recv(s, c) => {
                self.visit_atomic(s);
                self.visit_atomic(c);
            }

            PExprKind::SetTimer(_) => {}

            PExprKind::Index(a, b) => {
                self.visit_atomic(a);
                self.visit_atomic(b);
            }
            PExprKind::Slice(a, b, c) => {
                self.visit_atomic(a);
                self.visit_atomic(b);
                self.visit_atomic(c);
            }
            PExprKind::TupleAccess(a, _)
            | PExprKind::FieldAccess(a, _)
            | PExprKind::SafeFieldAccess(a, _)
            | PExprKind::SafeTupleAccess(a, _) => self.visit_atomic(a),
            PExprKind::SafeIndex(a, b) => {
                self.visit_atomic(a);
                self.visit_atomic(b);
            }

            PExprKind::StructLit(_, fields) => {
                for (_, a) in fields {
                    self.visit_atomic(a);
                }
            }

            PExprKind::WrapInOptional(a) | PExprKind::PersistData(a) => self.visit_atomic(a),
            PExprKind::RetrieveData(_) | PExprKind::DiscardData => {}
        }
    }

    fn visit_func_call(&mut self, c: &PFuncCall) {
        match c {
            PFuncCall::User(u) => self.visit_user_call(u),
            PFuncCall::Builtin(_, args, _) => {
                for a in args {
                    self.visit_atomic(a);
                }
            }
        }
    }

    fn visit_user_call(&mut self, u: &PUserFuncCall) {
        self.visit(u.name, &u.original_name);
        for a in &u.args {
            self.visit_atomic(a);
        }
    }

    fn visit_atomic(&mut self, a: &PAtomic) {
        if let PAtomic::Var(id, name) = a {
            self.visit(*id, name);
        }
    }
}

#[cfg(test)]
mod test;
