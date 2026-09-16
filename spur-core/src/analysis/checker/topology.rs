use super::*;
use spur_ast::types::*;

impl TypeChecker {
    pub(super) fn check_topology_builtin(&mut self, builtin: BuiltinFn, args: Vec<ResolvedExpr>, span: Span) -> TypedExpr {
        if args.len() != 2 {
            self.emit(TypeError::WrongNumberOfArgs { expected: 2, got: args.len(), span });
            return self.error_expr(span);
        }
        let mut args = args.into_iter();
        let target = self.infer_expr(args.next().unwrap());
        let value = args.next().unwrap();
        let (expected, result) = if builtin == BuiltinFn::IndexOf {
            let Type::List(element) = &target.ty else {
                self.emit(TypeError::NotAList { ty: target.ty, span });
                return self.error_expr(span);
            };
            ((**element).clone(), Type::Optional(Box::new(Type::Int)))
        } else {
            let role_ty = if builtin == BuiltinFn::ProvideAll {
                match &target.ty { Type::List(t) => t.as_ref(), _ => &Type::Error }
            } else { &target.ty };
            let Type::Role(id, _) = role_ty else {
                self.emit(TypeError::ProvideTargetNotRole { message: "provide requires a role handle; provide_all requires a list of role handles".into(), span });
                return self.error_expr(span);
            };
            let Some((RoleKind::Role, parameter)) = self.role_params.get(id) else {
                self.emit(TypeError::ProvideTargetNotRole { message: "clients cannot receive provided parameters".into(), span });
                return self.error_expr(span);
            };
            (parameter.clone(), Type::Tuple(vec![]))
        };
        let first_error = self.errors.len();
        let value = self.check_expr(value, &expected);
        if builtin != BuiltinFn::IndexOf {
            for error in &mut self.errors[first_error..] {
                if matches!(error, TypeError::Mismatch { .. }) {
                    *error = TypeError::ProvideValueType { message: format!("expected role parameter {expected}"), span };
                }
            }
        }
        TypedExpr { kind: TypedExprKind::FuncCall(TypedFuncCall::Builtin(builtin, vec![target, value], result.clone())), ty: result, span }
    }

    fn validate_annotations(&mut self, tags: &[Annotation], allowed: &[&str]) {
        let mut seen = HashSet::new();
        for tag in tags {
            let span = tag.span;
            if !seen.insert(&tag.name) {
                self.emit(TypeError::DuplicateTag { message: tag.name.clone(), span });
            }
            if !["trace", "deploy", "scale", "choice", "quorum"].contains(&tag.name.as_str()) {
                self.emit(TypeError::UnknownTag { message: tag.name.clone(), span });
            } else if !allowed.contains(&tag.name.as_str()) {
                self.emit(TypeError::TagPlacement { message: format!("@{} is not allowed here", tag.name), span });
            }
            if tag.name != "deploy" && !tag.args.is_empty() {
                self.emit(TypeError::TagPlacement { message: format!("@{} takes no arguments", tag.name), span });
            }
        }
    }

    pub(super) fn check_topology(&mut self, tops: &[TypedTopLevelDef], structs: &HashMap<NameId, Vec<(NameId, String, Type)>>, enums: &HashMap<NameId, Vec<(NameId, String, Option<Type>)>>, tags: HashMap<NameId, Vec<Annotation>>) -> TopologyMetadata {
        let mut meta = TopologyMetadata { structs: structs.clone(), enums: enums.clone(), tags, ..Default::default() };
        let role_defs: Vec<_> = tops.iter().filter_map(|d| match d { TypedTopLevelDef::Role(r) => Some(r), _ => None }).collect();
        for role in &role_defs {
            // A client's parameter always occupies a node slot; a role's does
            // only when a function other than Init and RecoverInit reads it.
            let param_in_env = role.kind == RoleKind::Client
                || role.func_defs.iter().any(|f| {
                    !["Init", "RecoverInit"].contains(&f.original_name.as_str()) && reads(&f.body, role.param.name)
                });
            meta.roles.insert(role.name, RoleMetadata { kind: role.kind, parameter: role.param.ty.clone(), parameter_id: role.param.name, param_in_env, destinations: HashMap::new() });
        }
        let mut functions = Vec::new();
        let mut node_vars = HashSet::from([crate::compiler::cfg::SELF_NAME]);
        for role in &role_defs {
            node_vars.insert(role.param.name);
            for var in &role.var_inits {
                match &var.target { TypedVarTarget::Name(id, _) => { node_vars.insert(*id); }, TypedVarTarget::Tuple(vs) => node_vars.extend(vs.iter().map(|(id, _, _)| *id)) }
            }
            if !deployable(&role.param.ty, &meta, &mut HashSet::new()) {
                self.emit(TypeError::RoleParamNotDeployable { message: role.original_name.clone(), span: role.param.span });
            }
            for func in &role.func_defs {
                if func.annotations.iter().any(|a| a.name == "deploy") {
                    self.emit(TypeError::DeployNotFree { message: "deploy must annotate a free function".into(), span: func.span });
                }
                self.validate_annotations(&func.annotations, &["trace"]);
                if ["Init", "RecoverInit"].contains(&func.original_name.as_str()) && (!func.params.is_empty() || func.return_type != Type::Tuple(vec![])) {
                    self.emit(TypeError::InitSignature { message: "Init and RecoverInit take no parameters and return unit".into(), span: func.span });
                }
                functions.push(func);
            }
            if role.kind == RoleKind::Client {
                for name in ["Write", "Read", "RMW"] {
                    let Some(op) = role.func_defs.iter().find(|f| f.original_name == name) else {
                        if name != "RMW" { self.emit(TypeError::ClientOpMissing { message: name.into(), span: role.span }); }
                        continue;
                    };
                    if op.is_sync { self.emit(TypeError::ClientOpSync { message: name.into(), span: op.span }); }
                    let dest = op.params.first().and_then(|p| match &p.ty { Type::Role(id, _) if meta.roles.get(id).is_some_and(|r| r.kind == RoleKind::Role) => Some(*id), _ => None });
                    let params = &op.params[usize::from(dest.is_some())..];
                    let expected = if name == "Read" { vec![Type::String] } else { vec![Type::String, Type::Int] };
                    let ret = if name == "Write" { Type::Tuple(vec![]) } else { Type::List(Box::new(Type::Int)) };
                    if params.iter().map(|p| &p.ty).ne(expected.iter()) || op.return_type != ret {
                        self.emit(TypeError::ClientOpSignature { message: name.into(), span: op.span });
                    }
                    meta.roles.get_mut(&role.name).unwrap().destinations.insert(name.into(), dest);
                }
            }
        }
        for fields in structs.values() {
            for (id, _, ty) in fields {
                let annotations = meta.tags.get(id).cloned().unwrap_or_default();
                self.validate_annotations(&annotations, &["scale", "choice", "quorum"]);
                for a in &annotations {
                    match a.name.as_str() {
                        "scale" if *ty != Type::Int => self.emit(TypeError::ScaleType { message: "@scale requires int".into(), span: a.span }),
                        "choice" if !choice_type(ty, enums) => self.emit(TypeError::ChoiceType { message: "@choice requires int, bool, string, or a payload-free enum".into(), span: a.span }),
                        "quorum" if !matches!(ty, Type::List(t) if matches!(t.as_ref(), Type::Role(id, _) if meta.roles.get(id).is_some_and(|r| r.kind == RoleKind::Role))) => self.emit(TypeError::QuorumType { message: "@quorum requires a list of role handles".into(), span: a.span }),
                        _ => {}
                    }
                }
            }
        }
        for top in tops {
            if let TypedTopLevelDef::FreeFunc(func) = top { functions.push(func); }
        }
        let mut effects = HashMap::new();
        for f in &functions {
            let mut effect = Effect::default();
            visit_block(&f.body, &node_vars, None, &mut effect);
            effects.insert(f.name, effect);
        }
        loop {
            let old: HashSet<_> = effects.iter().filter(|(_, e)| e.node_bound).map(|(id, _)| *id).collect();
            let mut changed = false;
            for effect in effects.values_mut() {
                if !effect.node_bound && effect.calls.iter().any(|id| old.contains(id)) { effect.node_bound = true; changed = true; }
            }
            if !changed { break; }
        }
        let mut consumed_params = HashSet::new();
        let mut reachable_types = HashSet::new();
        for top in tops {
            let TypedTopLevelDef::FreeFunc(func) = top else { continue; };
            self.validate_annotations(&func.annotations, &["deploy"]);
            let Some(tag) = func.annotations.iter().find(|a| a.name == "deploy") else { continue; };
            let span = func.span;
            let client_args: Vec<_> = tag.args.iter().filter(|(k, _)| k == "client").collect();
            if client_args.is_empty() { self.emit(TypeError::DeployClientMissing { message: "@deploy requires client = C".into(), span }); continue; }
            if tag.args.len() != 1 { self.emit(TypeError::TagPlacement { message: "@deploy accepts exactly one client argument".into(), span }); }
            let Some(client) = role_defs.iter().find(|r| r.kind == RoleKind::Client && r.original_name == client_args[0].1) else {
                self.emit(TypeError::DeployClientUnknown { message: client_args[0].1.clone(), span }); continue;
            };
            if effects[&func.name].node_bound { self.emit(TypeError::DeployNodeBound { message: "deploy calls an operation requiring a running node".into(), span }); }
            let root = match &func.return_type { Type::Optional(t) if !matches!(t.as_ref(), Type::Optional(_)) && deployable(t, &meta, &mut HashSet::new()) => t.as_ref().clone(), _ => { self.emit(TypeError::DeployReturnType { message: "deploy must return a deployable T?".into(), span }); continue; } };
            if client.param.ty != root { self.emit(TypeError::ClientParamMismatch { message: "client parameter must equal the deploy root type".into(), span }); }
            reachable_structs(&root, &meta, &mut reachable_types);
            let mut fields = Vec::new();
            if func.params.len() > 1 { self.emit(TypeError::DeployParamNotStruct { message: "deploy takes zero or one parameter".into(), span }); }
            if let Some(param) = func.params.first() {
                if let Type::Struct(id, _) = &param.ty {
                    consumed_params.insert(*id);
                    for (fid, name, ty) in structs.get(id).into_iter().flatten() {
                        let tags = meta.tags.get(fid).map(Vec::as_slice).unwrap_or(&[]);
                        let scale = tags.iter().any(|t| t.name == "scale");
                        let choice = tags.iter().any(|t| t.name == "choice");
                        if !scale && !choice { self.emit(TypeError::ParamFieldUntagged { message: name.clone(), span }); }
                        if scale && choice { self.emit(TypeError::ParamFieldTwoTags { message: name.clone(), span }); }
                        fields.push(ParamField { name: name.clone(), ty: ty.clone(), tag: if scale { ParamTag::Scale } else { ParamTag::Choice } });
                    }
                } else { self.emit(TypeError::DeployParamNotStruct { message: "deploy parameter must be a struct".into(), span }); }
            }
            meta.deploys.push(DeployMetadata { id: func.name, name: func.original_name.clone(), client: client.name, parameter: func.params.first().map(|p| p.ty.clone()), fields, root });
        }
        for (id, fields) in structs {
            for (fid, name, _) in fields {
                for tag in meta.tags.get(fid).into_iter().flatten() {
                    if (["scale", "choice"].contains(&tag.name.as_str()) && !consumed_params.contains(id)) || (tag.name == "quorum" && !reachable_types.contains(id)) {
                        meta.warnings.push(format!("TagWithoutConsumer: @{} on {name}", tag.name));
                    }
                }
            }
        }
        meta
    }
}

fn choice_type(ty: &Type, enums: &HashMap<NameId, Vec<(NameId, String, Option<Type>)>>) -> bool {
    matches!(ty, Type::Int | Type::Bool | Type::String) || matches!(ty, Type::Enum(id, _) if enums.get(id).is_some_and(|v| v.iter().all(|(_, _, p)| p.is_none())))
}

fn deployable(ty: &Type, meta: &TopologyMetadata, seen: &mut HashSet<NameId>) -> bool {
    match ty {
        Type::Int | Type::Bool | Type::String => true,
        Type::Role(id, _) => meta.roles.get(id).is_some_and(|r| r.kind == RoleKind::Role),
        Type::List(t) | Type::Optional(t) => deployable(t, meta, seen),
        Type::Map(k, v) => deployable(k, meta, seen) && deployable(v, meta, seen),
        Type::Tuple(ts) => ts.iter().all(|t| deployable(t, meta, seen)),
        Type::Struct(id, _) => !seen.insert(*id) || meta.structs.get(id).is_some_and(|fs| fs.iter().all(|(_, _, t)| deployable(t, meta, seen))),
        Type::Enum(id, _) => !seen.insert(*id) || meta.enums.get(id).is_some_and(|vs| vs.iter().all(|(_, _, t)| t.as_ref().is_none_or(|t| deployable(t, meta, seen)))),
        _ => false,
    }
}

fn reachable_structs(ty: &Type, meta: &TopologyMetadata, seen: &mut HashSet<NameId>) {
    match ty {
        Type::Struct(id, _) if seen.insert(*id) => { for (_, _, t) in &meta.structs[id] { reachable_structs(t, meta, seen); } }
        Type::Enum(id, _) if seen.insert(*id) => { for (_, _, t) in &meta.enums[id] { if let Some(t) = t { reachable_structs(t, meta, seen); } } }
        Type::List(t) | Type::Optional(t) => reachable_structs(t, meta, seen),
        Type::Map(k, v) => { reachable_structs(k, meta, seen); reachable_structs(v, meta, seen); }
        Type::Tuple(ts) => { for t in ts { reachable_structs(t, meta, seen); } }
        _ => {}
    }
}

#[derive(Default)]
struct Effect { node_bound: bool, reads_target: bool, calls: HashSet<NameId> }

/// Whether `block` reads `target`.
fn reads(block: &TypedBlock, target: NameId) -> bool {
    let mut effect = Effect::default();
    visit_block(block, &HashSet::new(), Some(target), &mut effect);
    effect.reads_target
}

fn visit_block(block: &TypedBlock, vars: &HashSet<NameId>, probe: Option<NameId>, effect: &mut Effect) {
    for s in &block.statements { visit_statement(s, vars, probe, effect); }
    if let Some(t) = &block.tail_expr { visit_expr(t, vars, probe, effect); }
}
fn visit_statement(stmt: &TypedStatement, vars: &HashSet<NameId>, probe: Option<NameId>, effect: &mut Effect) {
    match &stmt.kind {
        TypedStatementKind::Assignment(a) => visit_expr(&a.value, vars, probe, effect),
        TypedStatementKind::Expr(e) => visit_expr(e, vars, probe, effect),
        TypedStatementKind::ForLoop(l) => {
            if let Some(a) = &l.init { visit_expr(&a.value, vars, probe, effect); }
            if let Some(c) = &l.condition { visit_expr(c, vars, probe, effect); }
            if let Some(a) = &l.increment { visit_expr(&a.value, vars, probe, effect); }
            for s in &l.body { visit_statement(s, vars, probe, effect); }
        }
        TypedStatementKind::ForInLoop(l) => { visit_expr(&l.iterable, vars, probe, effect); for s in &l.body { visit_statement(s, vars, probe, effect); } }
        TypedStatementKind::Error => {}
    }
}
fn visit_expr(expr: &TypedExpr, vars: &HashSet<NameId>, probe: Option<NameId>, effect: &mut Effect) {
    use TypedExprKind::*;
    if matches!(&expr.kind, Var(id, _) if Some(*id) == probe) { effect.reads_target = true; }
    if matches!(&expr.kind, Var(id, _) if vars.contains(id)) || matches!(&expr.kind, MakeChannel | Send(..) | Recv(..) | SetTimer(..) | Fifo(..) | PersistData(..) | RetrieveData(..) | DiscardData | RpcCall(..)) { effect.node_bound = true; }
    match &expr.kind {
        FuncCall(TypedFuncCall::User(call)) | RpcCall(_, call) => {
            effect.calls.insert(call.name);
            for a in &call.args { visit_expr(a, vars, probe, effect); }
            if let RpcCall(target, _) = &expr.kind { visit_expr(target, vars, probe, effect); }
        }
        FuncCall(TypedFuncCall::Builtin(b, args, _)) => { if *b == BuiltinFn::UniqueId { effect.node_bound = true; } for a in args { visit_expr(a, vars, probe, effect); } }
        BinOp(_, a, b) | Append(a, b) | Prepend(a, b) | Min(a, b) | Exists(a, b) | Erase(a, b) | Send(a, b) | Index(a, b) | SafeIndex(a, b) => { visit_expr(a, vars, probe, effect); visit_expr(b, vars, probe, effect); }
        Store(a, b, c) | Slice(a, b, c) => { visit_expr(a, vars, probe, effect); visit_expr(b, vars, probe, effect); visit_expr(c, vars, probe, effect); }
        Not(a) | Negate(a) | Head(a) | Tail(a) | Len(a) | UnwrapOptional(a) | Recv(a) | Fifo(a) | TupleAccess(a, _) | FieldAccess(a, _, _) | SafeFieldAccess(a, _, _) | SafeTupleAccess(a, _) | WrapInOptional(a) | PersistData(a) | Return(a) => visit_expr(a, vars, probe, effect),
        MapLit(pairs) => { for (a, b) in pairs { visit_expr(a, vars, probe, effect); visit_expr(b, vars, probe, effect); } }
        ListLit(xs) | TupleLit(xs) => { for x in xs { visit_expr(x, vars, probe, effect); } }
        StructLit(_, fields) => { for (_, _, v) in fields { visit_expr(v, vars, probe, effect); } }
        Match(e, arms) => { visit_expr(e, vars, probe, effect); for arm in arms { visit_block(&arm.body, vars, probe, effect); } }
        Conditional(c) => {
            for b in std::iter::once(&c.if_branch).chain(&c.elseif_branches) { visit_expr(&b.condition, vars, probe, effect); visit_block(&b.body, vars, probe, effect); }
            if let Some(b) = &c.else_branch { visit_block(b, vars, probe, effect); }
        }
        Block(b) => visit_block(b, vars, probe, effect),
        VariantLit(_, _, _, Some(p)) => visit_expr(p, vars, probe, effect),
        _ => {}
    }
}
