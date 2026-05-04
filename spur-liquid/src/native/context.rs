use std::collections::HashMap;

use spur_ast::name::NameId;

use crate::ir::{CProgram, CType};

pub type Env = Vec<(NameId, CType)>;

pub struct FuncSig {
    pub params: Vec<(NameId, CType)>,
    pub return_type: CType,
}

pub struct GlobalCtx<'p> {
    pub func_env: HashMap<NameId, FuncSig>,
    pub struct_env: &'p HashMap<NameId, Vec<(NameId, String, CType)>>,
    pub enum_env: &'p HashMap<NameId, Vec<(NameId, String, Option<CType>)>>,
    pub tuple_accessors: HashMap<usize, NameId>,
    pub fresh_start: i32,
}

impl<'p> GlobalCtx<'p> {
    pub fn from_program(program: &'p CProgram) -> Self {
        let mut func_env = HashMap::new();

        for f in &program.extern_funcs {
            func_env.insert(
                f.name,
                FuncSig {
                    params: f.params.iter().map(|p| (p.name, p.ty.clone())).collect(),
                    return_type: f.return_type.clone(),
                },
            );
        }

        for f in &program.funcs {
            func_env.insert(
                f.name,
                FuncSig {
                    params: f.params.iter().map(|p| (p.name, p.ty.clone())).collect(),
                    return_type: f.return_type.clone(),
                },
            );
        }

        let mut next_id = program.next_name_id as i32;
        for (sid, fields) in &program.struct_defs {
            next_id = next_id.max(sid.0 as i32 + 1);
            for (fid, _, _) in fields {
                next_id = next_id.max(fid.0 as i32 + 1);
            }
        }
        for (eid, ctors) in &program.enum_defs {
            next_id = next_id.max(eid.0 as i32 + 1);
            for (vid, _, _) in ctors {
                next_id = next_id.max(vid.0 as i32 + 1);
            }
        }
        for f in &program.funcs {
            next_id = next_id.max(f.name.0 as i32 + 1);
        }
        for f in &program.extern_funcs {
            next_id = next_id.max(f.name.0 as i32 + 1);
        }

        let mut tuple_accessors = HashMap::new();
        let mut alloc_id = next_id;
        for f in &program.funcs {
            collect_tuple_indices(&f.body, &mut tuple_accessors, &mut alloc_id);
        }

        let fresh_start = -(alloc_id + 1024);

        GlobalCtx {
            func_env,
            struct_env: &program.struct_defs,
            enum_env: &program.enum_defs,
            tuple_accessors,
            fresh_start,
        }
    }

    pub fn lookup_func(&self, name: NameId) -> Option<&FuncSig> {
        self.func_env.get(&name)
    }

    pub fn lookup_struct_fields(&self, name: NameId) -> Option<&[(NameId, String, CType)]> {
        self.struct_env.get(&name).map(|v| v.as_slice())
    }

    pub fn lookup_enum_variants(&self, name: NameId) -> Option<&[(NameId, String, Option<CType>)]> {
        self.enum_env.get(&name).map(|v| v.as_slice())
    }

    pub fn tuple_accessor_id(&self, idx: usize) -> Option<NameId> {
        self.tuple_accessors.get(&idx).copied()
    }
}

fn collect_tuple_indices(
    block: &crate::ir::CBlock,
    accessors: &mut HashMap<usize, NameId>,
    next_id: &mut i32,
) {
    use crate::ir::CStatementKind;

    for stmt in &block.statements {
        match &stmt.kind {
            CStatementKind::LetAtom(let_atom) => {
                collect_tuple_indices_expr(&let_atom.value, accessors, next_id);
            }
            CStatementKind::Expr(e) => {
                collect_tuple_indices_expr(e, accessors, next_id);
            }
            _ => {}
        }
    }
}

fn collect_tuple_indices_expr(
    expr: &crate::ir::CExpr,
    accessors: &mut HashMap<usize, NameId>,
    next_id: &mut i32,
) {
    use crate::ir::CExprKind;

    match &expr.kind {
        CExprKind::TupleAccess(_, idx) => {
            accessors.entry(*idx).or_insert_with(|| {
                let id = NameId(*next_id as usize);
                *next_id += 1;
                id
            });
        }
        CExprKind::Block(b) => collect_tuple_indices(b, accessors, next_id),
        CExprKind::Conditional(c) => {
            collect_tuple_indices(&c.if_branch.body, accessors, next_id);
            for branch in &c.elseif_branches {
                collect_tuple_indices(&branch.body, accessors, next_id);
            }
            if let Some(else_b) = &c.else_branch {
                collect_tuple_indices(else_b, accessors, next_id);
            }
        }
        _ => {}
    }
}

pub fn env_lookup(x: NameId, env: &Env) -> Option<&CType> {
    env.iter().find(|(id, _)| *id == x).map(|(_, ty)| ty)
}

pub struct Counter(i32);

impl Counter {
    pub fn new(start: i32) -> Self {
        Counter(start)
    }

    pub fn fresh(&mut self) -> NameId {
        let id = NameId(self.0 as usize);
        self.0 -= 1;
        id
    }
}
