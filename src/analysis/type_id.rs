#[cfg(test)]
mod test;

use crate::analysis::types::{
    Type, TypedExpr, TypedExprKind, TypedFuncCall, TypedMatchArm, TypedPattern, TypedPatternKind,
    TypedProgram, TypedStatement, TypedStatementKind, TypedTopLevelDef, TypedVarInit,
    TypedVarTarget,
};
use serde::Serialize;
use std::collections::HashMap;

/// A unique identifier for a structurally distinct type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct TypeId(pub u32);

/// Maps each structurally distinct `Type` to its unique `TypeId`.
pub type TypeIdMap = HashMap<Type, TypeId>;

/// Walk the entire typed program, collect every distinct `Type` encountered,
/// and assign each a unique `TypeId`.
pub fn assign_type_ids(program: &TypedProgram) -> TypeIdMap {
    let mut map = TypeIdMap::new();
    let mut next_id: u32 = 0;

    let register = |ty: &Type, map: &mut TypeIdMap, next_id: &mut u32| {
        register_type(ty, map, next_id);
    };

    // Register types from struct definitions.
    for (_name_id, fields) in &program.struct_defs {
        for (_field_name, field_ty) in fields {
            register(field_ty, &mut map, &mut next_id);
        }
    }

    // Register types from enum definitions.
    for (_name_id, variants) in &program.enum_defs {
        for (_variant_name, payload_ty) in variants {
            if let Some(ty) = payload_ty {
                register(ty, &mut map, &mut next_id);
            }
        }
    }

    // Walk all top-level definitions (roles, functions, expressions, etc.).
    for def in &program.top_level_defs {
        match def {
            TypedTopLevelDef::Role(role) => {
                for init in &role.var_inits {
                    register_var_init(init, &mut map, &mut next_id);
                }
                for func in &role.func_defs {
                    // Register parameter types.
                    for param in &func.params {
                        register_type(&param.ty, &mut map, &mut next_id);
                    }
                    // Register return type.
                    register_type(&func.return_type, &mut map, &mut next_id);
                    // Walk the function body.
                    register_body(&func.body, &mut map, &mut next_id);
                }
            }
        }
    }

    map
}

/// Register a type and all of its component types.
fn register_type(ty: &Type, map: &mut TypeIdMap, next_id: &mut u32) {
    if map.contains_key(ty) {
        return;
    }

    let id = TypeId(*next_id);
    *next_id += 1;
    map.insert(ty.clone(), id);

    match ty {
        Type::List(inner) | Type::Optional(inner) | Type::Chan(inner) => {
            register_type(inner, map, next_id);
        }
        Type::Map(key, val) => {
            register_type(key, map, next_id);
            register_type(val, map, next_id);
        }
        Type::Tuple(elems) => {
            for elem in elems {
                register_type(elem, map, next_id);
            }
        }
        Type::Int
        | Type::String
        | Type::Bool
        | Type::Struct(_, _)
        | Type::Enum(_, _)
        | Type::Role(_, _)
        | Type::EmptyList
        | Type::EmptyMap
        | Type::UnknownChannel
        | Type::Nil
        | Type::Never
        | Type::Error => {}
    }
}

fn register_var_init(init: &TypedVarInit, map: &mut TypeIdMap, next_id: &mut u32) {
    register_type(&init.type_def, map, next_id);
    match &init.target {
        TypedVarTarget::Name(_, _) => {}
        TypedVarTarget::Tuple(elements) => {
            for (_, _, ty) in elements {
                register_type(ty, map, next_id);
            }
        }
    }
    register_expr(&init.value, map, next_id);
}

fn register_body(body: &[TypedStatement], map: &mut TypeIdMap, next_id: &mut u32) {
    for stmt in body {
        register_stmt(stmt, map, next_id);
    }
}

fn register_stmt(stmt: &TypedStatement, map: &mut TypeIdMap, next_id: &mut u32) {
    match &stmt.kind {
        TypedStatementKind::VarInit(init) => {
            register_var_init(init, map, next_id);
        }
        TypedStatementKind::Assignment(assign) => {
            register_expr(&assign.target, map, next_id);
            register_expr(&assign.value, map, next_id);
        }
        TypedStatementKind::Expr(expr) => {
            register_expr(expr, map, next_id);
        }
        TypedStatementKind::Return(expr) => {
            register_expr(expr, map, next_id);
        }
        TypedStatementKind::Conditional(cond) => {
            register_expr(&cond.if_branch.condition, map, next_id);
            register_body(&cond.if_branch.body, map, next_id);
            for branch in &cond.elseif_branches {
                register_expr(&branch.condition, map, next_id);
                register_body(&branch.body, map, next_id);
            }
            if let Some(body) = &cond.else_branch {
                register_body(body, map, next_id);
            }
        }
        TypedStatementKind::ForLoop(fl) => {
            match &fl.init {
                Some(crate::analysis::types::TypedForLoopInit::VarInit(init)) => {
                    register_var_init(init, map, next_id);
                }
                Some(crate::analysis::types::TypedForLoopInit::Assignment(assign)) => {
                    register_expr(&assign.target, map, next_id);
                    register_expr(&assign.value, map, next_id);
                }
                None => {}
            }
            if let Some(cond) = &fl.condition {
                register_expr(cond, map, next_id);
            }
            if let Some(inc) = &fl.increment {
                register_expr(&inc.target, map, next_id);
                register_expr(&inc.value, map, next_id);
            }
            register_body(&fl.body, map, next_id);
        }
        TypedStatementKind::ForInLoop(fl) => {
            register_pattern(&fl.pattern, map, next_id);
            register_expr(&fl.iterable, map, next_id);
            register_body(&fl.body, map, next_id);
        }
        TypedStatementKind::Break | TypedStatementKind::Continue => {}
        TypedStatementKind::Error => {}
    }
}

fn register_expr(expr: &TypedExpr, map: &mut TypeIdMap, next_id: &mut u32) {
    // Register the expression's own type.
    register_type(&expr.ty, map, next_id);

    match &expr.kind {
        TypedExprKind::BinOp(_, l, r)
        | TypedExprKind::Append(l, r)
        | TypedExprKind::Prepend(l, r)
        | TypedExprKind::Min(l, r)
        | TypedExprKind::Exists(l, r)
        | TypedExprKind::Erase(l, r)
        | TypedExprKind::Send(l, r)
        | TypedExprKind::Index(l, r)
        | TypedExprKind::SafeIndex(l, r) => {
            register_expr(l, map, next_id);
            register_expr(r, map, next_id);
        }
        TypedExprKind::Store(a, b, c) | TypedExprKind::Slice(a, b, c) => {
            register_expr(a, map, next_id);
            register_expr(b, map, next_id);
            register_expr(c, map, next_id);
        }
        TypedExprKind::Not(e)
        | TypedExprKind::Negate(e)
        | TypedExprKind::Head(e)
        | TypedExprKind::Tail(e)
        | TypedExprKind::Len(e)
        | TypedExprKind::UnwrapOptional(e)
        | TypedExprKind::Recv(e)
        | TypedExprKind::TupleAccess(e, _)
        | TypedExprKind::FieldAccess(e, _)
        | TypedExprKind::SafeFieldAccess(e, _)
        | TypedExprKind::SafeTupleAccess(e, _)
        | TypedExprKind::WrapInOptional(e)
        | TypedExprKind::PersistData(e) => {
            register_expr(e, map, next_id);
        }
        TypedExprKind::FuncCall(call) => match call {
            TypedFuncCall::User(u) => {
                register_type(&u.return_type, map, next_id);
                for arg in &u.args {
                    register_expr(arg, map, next_id);
                }
            }
            TypedFuncCall::Builtin(_, args, ret_ty) => {
                register_type(ret_ty, map, next_id);
                for arg in args {
                    register_expr(arg, map, next_id);
                }
            }
        },
        TypedExprKind::MapLit(pairs) => {
            for (k, v) in pairs {
                register_expr(k, map, next_id);
                register_expr(v, map, next_id);
            }
        }
        TypedExprKind::ListLit(exprs) | TypedExprKind::TupleLit(exprs) => {
            for e in exprs {
                register_expr(e, map, next_id);
            }
        }
        TypedExprKind::RpcCall(target, call) => {
            register_expr(target, map, next_id);
            register_type(&call.return_type, map, next_id);
            for arg in &call.args {
                register_expr(arg, map, next_id);
            }
        }
        TypedExprKind::Match(scrutinee, arms) => {
            register_expr(scrutinee, map, next_id);
            for arm in arms {
                register_match_arm(arm, map, next_id);
            }
        }
        TypedExprKind::VariantLit(_, _, payload) => {
            if let Some(p) = payload {
                register_expr(p, map, next_id);
            }
        }
        TypedExprKind::StructLit(_, fields) => {
            for (_, e) in fields {
                register_expr(e, map, next_id);
            }
        }
        TypedExprKind::RetrieveData(ty) => {
            register_type(ty, map, next_id);
        }
        TypedExprKind::Var(_, _)
        | TypedExprKind::IntLit(_)
        | TypedExprKind::StringLit(_)
        | TypedExprKind::BoolLit(_)
        | TypedExprKind::NilLit
        | TypedExprKind::MakeChannel
        | TypedExprKind::SetTimer
        | TypedExprKind::DiscardData
        | TypedExprKind::Error => {}
    }
}

fn register_match_arm(arm: &TypedMatchArm, map: &mut TypeIdMap, next_id: &mut u32) {
    register_pattern(&arm.pattern, map, next_id);
    register_body(&arm.body, map, next_id);
}

fn register_pattern(pat: &TypedPattern, map: &mut TypeIdMap, next_id: &mut u32) {
    register_type(&pat.ty, map, next_id);
    match &pat.kind {
        TypedPatternKind::Var(_, _) | TypedPatternKind::Wildcard => {}
        TypedPatternKind::Tuple(pats) => {
            for p in pats {
                register_pattern(p, map, next_id);
            }
        }
        TypedPatternKind::Variant(_, _, payload) => {
            if let Some(p) = payload {
                register_pattern(p, map, next_id);
            }
        }
        TypedPatternKind::Error => {}
    }
}
