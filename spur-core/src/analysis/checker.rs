use crate::analysis::resolver::{
    BuiltinFn, NameId, PrepopulatedTypes, ResolvedAssignItem, ResolvedAssignment, ResolvedBlock,
    ResolvedCondExpr, ResolvedExpr, ResolvedExprKind, ResolvedForInLoop, ResolvedForLoop,
    ResolvedFuncCall, ResolvedFuncDef, ResolvedMatchArm, ResolvedPattern, ResolvedPatternKind,
    ResolvedProgram, ResolvedRoleDef, ResolvedStatement, ResolvedStatementKind,
    ResolvedTopLevelDef, ResolvedTypeDef, ResolvedTypeDefStmtKind, ResolvedUserFuncCall,
    ResolvedVarInit,
};
use crate::analysis::trivially_copyable::{
    TriviallyCopyableMap, compute_trivially_copyable, is_trivially_copyable,
};
use crate::analysis::types::{
    Type, TypedAssignItem, TypedAssignment, TypedBlock, TypedCondExpr, TypedExpr, TypedExprKind,
    TypedForInLoop, TypedForLoop, TypedFuncCall, TypedFuncDef, TypedFuncParam, TypedIfBranch,
    TypedMatchArm, TypedPattern, TypedPatternKind, TypedProgram, TypedRoleDef, TypedStatement,
    TypedStatementKind, TypedTopLevelDef, TypedUserFuncCall, TypedVarInit, TypedVarTarget,
};
use crate::parser::{BinOp, Span};
use chumsky::span::Span as ChumskySpan;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Clone)]
pub enum TypeError {
    #[error("Type Mismatch: Expected `{expected}`, but found `{found}`")]
    Mismatch {
        expected: Type,
        found: Type,
        span: Span,
    },
    #[error("Undefined type name used")]
    UndefinedType(Span),
    #[error("Operator `{op}` cannot be applied to type `{ty}`")]
    InvalidUnaryOp {
        op: &'static str,
        ty: Type,
        span: Span,
    },
    #[error("Operator `{op}` cannot be applied to types `{left}` and `{right}`")]
    InvalidBinOp {
        op: BinOp,
        left: Type,
        right: Type,
        span: Span,
    },
    #[error("Function call has wrong number of arguments. Expected {expected}, got {got}")]
    WrongNumberOfArgs {
        expected: usize,
        got: usize,
        span: Span,
    },
    #[error("Attempted to access field `{field_name}` on a non-struct type `{ty}`")]
    NotAStruct {
        ty: Type,
        field_name: String,
        span: Span,
    },
    #[error("Field `{field_name}` not found on struct")]
    FieldNotFound { field_name: String, span: Span },
    #[error("Attempted to index a non-indexable type `{ty}`")]
    NotIndexable { ty: Type, span: Span },
    #[error("Invalid index type `{ty}`. Expected `int`")]
    InvalidIndexType { ty: Type, span: Span },
    #[error("Invalid map key type. Expected `{expected}`, but found `{found}`")]
    InvalidMapKeyType {
        expected: Type,
        found: Type,
        span: Span,
    },
    #[error("Invalid struct key type. Expected `string`, but found `{found}`")]
    InvalidStructKeyType { found: Type, span: Span },
    #[error("store requires a list or map, found `{ty}`")]
    StoreOnInvalidType { ty: Type, span: Span },
    #[error("Invalid assignment target")]
    InvalidAssignmentTarget(Span),
    #[error("A function that is not `unit` must have a return statement on all paths")]
    MissingReturn(Span),
    #[error("Return statement found outside of a function")]
    ReturnOutsideFunction(Span),
    #[error("Break statement found outside of a loop")]
    BreakOutsideLoop(Span),
    #[error("Continue statement found outside of a loop")]
    ContinueOutsideLoop(Span),
    #[error("Cannot iterate over type `{ty}`")]
    NotIterable { ty: Type, span: Span },
    #[error("Pattern does not match iterable type")]
    PatternMismatch {
        pattern_ty: String,
        iterable_ty: Type,
        span: Span,
    },
    #[error("Tuple access index {index} out of bounds for tuple of size {size}")]
    TupleIndexOutOfBounds {
        index: usize,
        size: usize,
        span: Span,
    },
    #[error("Cannot access tuple index on non-tuple type `{ty}`")]
    NotATuple { ty: Type, span: Span },
    #[error("Struct literal field `{field_name}` is not defined in the struct")]
    UndefinedStructField { field_name: String, span: Span },
    #[error("Struct literal is missing field `{field_name}`")]
    MissingStructField { field_name: String, span: Span },
    #[error("RPC call target must be a role type, found `{ty}`")]
    RpcCallTargetNotRole { ty: Type, span: Span },
    #[error("List operation requires a list type, found `{ty}`")]
    NotAList { ty: Type, span: Span },
    #[error("Operator `!` can only be used on an optional, found `{ty}`")]
    UnwrapOnNonOptional { ty: Type, span: Span },
    #[error("Operation requires a channel, found `{ty}`")]
    NotAChannel { ty: Type, span: Span },
    #[error("recv cannot be used inside a `sync` function")]
    RecvInSyncFunc { span: Span },
    #[error("send cannot be used inside a `sync` function")]
    SendInSyncFunc { span: Span },
    #[error("RPC call targets a `sync` function `{func_name}`, but RPC calls must be async")]
    RpcCallToSyncFunc { func_name: String, span: Span },
    #[error("Enum `{name}` not found")]
    EnumNotFound { name: String, span: Span },
    #[error("Variant `{variant}` not found on enum `{enum_name}`")]
    VariantNotFound {
        enum_name: String,
        variant: String,
        span: Span,
    },
    #[error("Variant `{variant}` expects payload of type `{expected}`, but found `{found}`")]
    VariantPayloadMismatch {
        variant: String,
        expected: Type,
        found: Type,
        span: Span,
    },
    #[error("Variant `{variant}` expects no payload")]
    VariantExpectsNoPayload { variant: String, span: Span },
    #[error("Variant `{variant}` expects a payload")]
    VariantExpectsPayload { variant: String, span: Span },
    #[error("Match scrutinee must be an enum, found `{ty}`")]
    MatchScrutineeNotEnum { ty: Type, span: Span },
    #[error("Match arms have inconsistent types. Expected `{expected}`, found `{found}`")]
    MatchArmTypeMismatch {
        expected: Type,
        found: Type,
        span: Span,
    },

    #[error("Type `{ty}` is not trivially copyable, but it is required to be")]
    NonTriviallyCopyable { ty: Type, span: Span },

    #[error("Safe navigation `?.` requires an optional type, found `{ty}`")]
    SafeNavOnNonOptional { ty: Type, span: Span },

    #[error("Internal error: {message}")]
    InternalError { message: String, span: Span },
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    params: Vec<Type>,
    return_type: Type,
    is_sync: bool,
}

#[derive(Debug, Clone)]
enum TypeDefinition {
    Builtin(Type),
    UserDefined(ResolvedTypeDefStmtKind),
}

pub struct TypeChecker {
    scopes: Vec<HashMap<NameId, Type>>,
    type_defs: HashMap<NameId, TypeDefinition>,
    func_signatures: HashMap<NameId, FunctionSignature>,
    builtin_signatures: HashMap<BuiltinFn, FunctionSignature>,
    role_defs: HashMap<NameId, String>,
    role_func_signatures: HashMap<NameId, HashMap<String, (NameId, FunctionSignature)>>,
    role_func_name_ids: HashSet<NameId>,
    struct_names: HashMap<NameId, String>,
    enum_names: HashMap<NameId, String>,
    current_return_type: Option<Type>,
    current_func_is_sync: bool,
    loop_depth: usize,
    trivially_copyable: TriviallyCopyableMap,
    errors: Vec<TypeError>,
}

impl TypeChecker {
    pub fn new(prepopulated_types: PrepopulatedTypes) -> Self {
        let mut predefined = HashMap::new();
        predefined.insert(prepopulated_types.int, TypeDefinition::Builtin(Type::Int));
        predefined.insert(
            prepopulated_types.string,
            TypeDefinition::Builtin(Type::String),
        );
        predefined.insert(prepopulated_types.bool, TypeDefinition::Builtin(Type::Bool));
        let unit_type = Type::Tuple(vec![]);
        predefined.insert(
            prepopulated_types.unit,
            TypeDefinition::Builtin(unit_type.clone()),
        );

        let mut builtin_signatures = HashMap::new();
        builtin_signatures.insert(
            BuiltinFn::Println,
            FunctionSignature {
                params: vec![Type::String],
                return_type: unit_type.clone(),
                is_sync: true,
            },
        );
        builtin_signatures.insert(
            BuiltinFn::IntToString,
            FunctionSignature {
                params: vec![Type::Int],
                return_type: Type::String,
                is_sync: true,
            },
        );
        builtin_signatures.insert(
            BuiltinFn::BoolToString,
            FunctionSignature {
                params: vec![Type::Bool],
                return_type: Type::String,
                is_sync: true,
            },
        );
        builtin_signatures.insert(
            BuiltinFn::UniqueId,
            FunctionSignature {
                params: vec![],
                return_type: Type::Int,
                is_sync: true,
            },
        );

        Self {
            scopes: vec![HashMap::new()],
            type_defs: predefined,
            func_signatures: HashMap::new(),
            builtin_signatures,
            role_defs: HashMap::new(),
            role_func_signatures: HashMap::new(),
            role_func_name_ids: HashSet::new(),
            struct_names: HashMap::new(),
            enum_names: HashMap::new(),
            current_return_type: None,
            current_func_is_sync: false,
            loop_depth: 0,
            trivially_copyable: TriviallyCopyableMap::new(),
            errors: Vec::new(),
        }
    }

    fn emit(&mut self, err: TypeError) {
        self.errors.push(err);
    }

    fn error_expr(&self, span: Span) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Error,
            ty: Type::Error,
            span,
        }
    }

    fn error_stmt(&self, span: Span) -> (TypedStatement, bool) {
        (
            TypedStatement {
                kind: TypedStatementKind::Error,
                span,
            },
            false,
        )
    }

    pub fn check_program(&mut self, program: ResolvedProgram) -> (TypedProgram, Vec<TypeError>) {
        self.collect_definitions(&program);

        let next_name_id = program.next_name_id;
        let id_to_name = program.id_to_name;
        let mut typed_top_levels = Vec::new();
        for top_level_def in program.top_level_defs {
            match top_level_def {
                ResolvedTopLevelDef::Role(role) => {
                    let typed_role = self.check_role_def(role);
                    typed_top_levels.push(TypedTopLevelDef::Role(typed_role));
                }
                ResolvedTopLevelDef::FreeFunc(func) => {
                    let typed_func = self.check_func_def(func);
                    typed_top_levels.push(TypedTopLevelDef::FreeFunc(typed_func));
                }
                ResolvedTopLevelDef::Type(_) => {}
            }
        }

        // Resolve type definitions for downstream passes.
        let mut struct_defs = HashMap::new();
        let mut enum_defs = HashMap::new();
        for (name_id, type_def) in &self.type_defs {
            match type_def {
                TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Struct(fields)) => {
                    let resolved_fields: Vec<(String, Type)> = fields
                        .iter()
                        .filter_map(|f| match self.resolve_type(&f.type_def) {
                            Ok(ty) => Some((f.name.clone(), ty)),
                            Err(_) => None,
                        })
                        .collect();
                    struct_defs.insert(*name_id, resolved_fields);
                }
                TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Enum(variants)) => {
                    let resolved_variants: Vec<(String, Option<Type>)> = variants
                        .iter()
                        .filter_map(|v| {
                            let payload_ty = match v.payload.as_ref().map(|p| self.resolve_type(p))
                            {
                                Some(Ok(ty)) => Some(ty),
                                Some(Err(_)) => return None,
                                None => None,
                            };
                            Some((v.name.clone(), payload_ty))
                        })
                        .collect();
                    enum_defs.insert(*name_id, resolved_variants);
                }
                _ => {}
            }
        }

        self.trivially_copyable = compute_trivially_copyable(&struct_defs, &enum_defs);

        let errors = std::mem::take(&mut self.errors);
        (
            TypedProgram {
                top_level_defs: typed_top_levels,
                next_name_id,
                id_to_name,
                struct_defs,
                enum_defs,
            },
            errors,
        )
    }

    fn collect_definitions(&mut self, program: &ResolvedProgram) {
        for top_level_def in &program.top_level_defs {
            match top_level_def {
                ResolvedTopLevelDef::Type(type_def) => {
                    self.type_defs.insert(
                        type_def.name,
                        TypeDefinition::UserDefined(type_def.def.clone()),
                    );
                    match &type_def.def {
                        ResolvedTypeDefStmtKind::Struct(_) => {
                            self.struct_names
                                .insert(type_def.name, type_def.original_name.clone());
                        }
                        ResolvedTypeDefStmtKind::Enum(_) => {
                            self.enum_names
                                .insert(type_def.name, type_def.original_name.clone());
                        }
                        _ => {}
                    }
                }
                ResolvedTopLevelDef::Role(role) => {
                    self.role_defs.insert(role.name, role.original_name.clone());
                    self.role_func_signatures.insert(role.name, HashMap::new());
                }
                ResolvedTopLevelDef::FreeFunc(_) => {}
            }
        }

        for top_level_def in &program.top_level_defs {
            match top_level_def {
                ResolvedTopLevelDef::Role(role) => {
                    for func in &role.func_defs {
                        let sig = match self.build_function_signature(func) {
                            Ok(sig) => sig,
                            Err(e) => {
                                self.emit(e);
                                continue;
                            }
                        };
                        let role_funcs = match self.role_func_signatures.get_mut(&role.name) {
                            Some(funcs) => funcs,
                            None => {
                                self.emit(TypeError::InternalError {
                                    message: format!(
                                        "Role '{}' not found in signature map",
                                        role.original_name
                                    ),
                                    span: role.span,
                                });
                                continue;
                            }
                        };

                        self.func_signatures.insert(func.name, sig.clone());
                        self.role_func_name_ids.insert(func.name);
                        role_funcs.insert(func.original_name.clone(), (func.name, sig));
                    }
                }
                ResolvedTopLevelDef::FreeFunc(func) => {
                    let sig = match self.build_function_signature(func) {
                        Ok(sig) => sig,
                        Err(e) => {
                            self.emit(e);
                            continue;
                        }
                    };
                    self.func_signatures.insert(func.name, sig);
                }
                ResolvedTopLevelDef::Type(_) => {}
            }
        }
    }

    fn build_function_signature(
        &self,
        func: &ResolvedFuncDef,
    ) -> Result<FunctionSignature, TypeError> {
        let params = func
            .params
            .iter()
            .map(|p| self.resolve_type(&p.type_def))
            .collect::<Result<Vec<_>, _>>()?;

        let return_type = if let Some(ret_type_def) = &func.return_type {
            self.resolve_type(ret_type_def)?
        } else {
            Type::Tuple(vec![])
        };

        Ok(FunctionSignature {
            params,
            return_type,
            is_sync: func.is_sync,
        })
    }

    fn check_role_def(&mut self, role: ResolvedRoleDef) -> TypedRoleDef {
        self.enter_scope();
        self.current_func_is_sync = true;

        let mut typed_var_inits = Vec::new();
        for var_init in role.var_inits {
            typed_var_inits.push(self.check_var_init(var_init));
        }

        self.current_func_is_sync = false;

        let mut typed_func_defs = Vec::new();
        for func in role.func_defs {
            typed_func_defs.push(self.check_func_def(func));
        }

        self.exit_scope();
        TypedRoleDef {
            name: role.name,
            original_name: role.original_name,
            var_inits: typed_var_inits,
            func_defs: typed_func_defs,
            span: role.span,
        }
    }

    fn check_func_def(&mut self, func: ResolvedFuncDef) -> TypedFuncDef {
        let sig = match self.func_signatures.get(&func.name) {
            Some(sig) => sig.clone(),
            None => {
                self.emit(TypeError::InternalError {
                    message: format!("Function signature not found for '{}'", func.original_name),
                    span: func.span,
                });
                return TypedFuncDef {
                    name: func.name,
                    original_name: func.original_name,
                    is_sync: false,
                    is_traced: func.is_traced,
                    params: vec![],
                    return_type: Type::Error,
                    body: TypedBlock {
                        statements: vec![],
                        tail_expr: None,
                        ty: Type::Error,
                        span: func.span,
                    },
                    span: func.span,
                };
            }
        };

        self.enter_scope();
        self.current_return_type = Some(sig.return_type.clone());
        self.current_func_is_sync = sig.is_sync;

        let mut typed_params = Vec::new();
        for (param, param_type) in func.params.into_iter().zip(sig.params.iter()) {
            self.add_var(param.name, param_type.clone());
            typed_params.push(TypedFuncParam {
                name: param.name,
                original_name: param.original_name,
                ty: param_type.clone(),
                span: param.span,
            });
        }

        let mut body = func.body;
        if let Some(tail) = body.tail_expr.take() {
            let span = tail.span;
            body.statements.push(ResolvedStatement {
                kind: ResolvedStatementKind::Expr(ResolvedExpr {
                    kind: ResolvedExprKind::Return(Box::new(*tail)),
                    span,
                }),
                span,
            });
        }

        let typed_body = self.check_block(body);
        let body_type = typed_body.ty.clone();

        // Non-unit return type requires divergence (body_type == Never)
        if sig.return_type != Type::Tuple(vec![]) && body_type != Type::Never {
            self.emit(TypeError::MissingReturn(func.span));
        }

        self.current_return_type = None;
        self.current_func_is_sync = false;
        self.exit_scope();
        TypedFuncDef {
            name: func.name,
            original_name: func.original_name,
            is_sync: sig.is_sync,
            is_traced: func.is_traced,
            params: typed_params,
            return_type: sig.return_type,
            body: typed_body,
            span: func.span,
        }
    }

    fn check_var_init(&mut self, var_init: ResolvedVarInit) -> TypedVarInit {
        let span = var_init.span;
        let (typed_value, type_def) = if let Some(def) = var_init.type_def {
            match self.resolve_type(&def) {
                Ok(expected_ty) => {
                    let typed_val = self.check_expr(var_init.value, &expected_ty);
                    (typed_val, expected_ty)
                }
                Err(e) => {
                    self.emit(e);
                    let typed_val = self.infer_expr(var_init.value);
                    let ty = typed_val.ty.clone();
                    (typed_val, ty)
                }
            }
        } else {
            let typed_val = self.infer_expr(var_init.value);
            let inferred_ty = typed_val.ty.clone();
            (typed_val, inferred_ty)
        };

        let target = match var_init.target {
            crate::analysis::resolver::ResolvedVarTarget::Name(name_id, name) => {
                self.add_var(name_id, type_def.clone());
                TypedVarTarget::Name(name_id, name)
            }
            crate::analysis::resolver::ResolvedVarTarget::Tuple(names) => match &type_def {
                Type::Tuple(types) => {
                    if names.len() != types.len() {
                        self.emit(TypeError::PatternMismatch {
                            pattern_ty: format!("tuple of {} elements", names.len()),
                            iterable_ty: type_def.clone(),
                            span,
                        });
                        // Still bind names with Error type so downstream code doesn't fail
                        let mut elements = Vec::new();
                        for (name_id, name) in names.into_iter() {
                            self.add_var(name_id, Type::Error);
                            elements.push((name_id, name, Type::Error));
                        }
                        TypedVarTarget::Tuple(elements)
                    } else {
                        let mut elements = Vec::new();
                        for ((name_id, name), ty) in names.into_iter().zip(types.iter()) {
                            self.add_var(name_id, ty.clone());
                            elements.push((name_id, name, ty.clone()));
                        }
                        TypedVarTarget::Tuple(elements)
                    }
                }
                _ => {
                    self.emit(TypeError::NotATuple {
                        ty: type_def.clone(),
                        span,
                    });
                    // Bind names with Error type
                    let mut elements = Vec::new();
                    for (name_id, name) in names.into_iter() {
                        self.add_var(name_id, Type::Error);
                        elements.push((name_id, name, Type::Error));
                    }
                    TypedVarTarget::Tuple(elements)
                }
            },
        };

        TypedVarInit {
            target,
            type_def,
            value: typed_value,
            span,
        }
    }

    fn check_statement(&mut self, stmt: ResolvedStatement) -> (TypedStatement, bool) {
        let span = stmt.span;
        match stmt.kind {
            ResolvedStatementKind::Assignment(assign) => {
                let typed_assign = self.check_assignment(assign);
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::Assignment(typed_assign),
                    span,
                };
                (typed_stmt, false)
            }
            ResolvedStatementKind::Expr(expr) => {
                let typed_expr = self.infer_expr(expr);
                let diverges = typed_expr.ty == Type::Never;
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::Expr(typed_expr),
                    span,
                };
                (typed_stmt, diverges)
            }
            ResolvedStatementKind::ForLoop(for_loop) => {
                let typed_loop = self.check_for_loop(for_loop);
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::ForLoop(typed_loop),
                    span,
                };
                (typed_stmt, false)
            }
            ResolvedStatementKind::ForInLoop(for_in_loop) => {
                let typed_loop = self.check_for_in_loop(for_in_loop);
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::ForInLoop(typed_loop),
                    span,
                };
                (typed_stmt, false)
            }
            ResolvedStatementKind::Error => (
                TypedStatement {
                    kind: TypedStatementKind::Error,
                    span,
                },
                false,
            ),
        }
    }

    fn check_tailless_block(
        &mut self,
        block: Vec<ResolvedStatement>,
    ) -> (Vec<TypedStatement>, bool) {
        self.enter_scope();
        let mut typed_stmts = Vec::new();
        let mut has_return = false;
        for stmt in block {
            let (typed_stmt, stmt_returns) = self.check_statement(stmt);
            typed_stmts.push(typed_stmt);
            if stmt_returns {
                has_return = true;
            }
        }
        self.exit_scope();
        (typed_stmts, has_return)
    }

    fn check_block(&mut self, block: ResolvedBlock) -> TypedBlock {
        self.enter_scope();
        let mut typed_stmts = Vec::new();
        let mut block_diverges = false;

        for stmt in block.statements {
            let (typed_stmt, diverges) = self.check_statement(stmt);
            if diverges {
                block_diverges = true;
            }
            typed_stmts.push(typed_stmt);
        }

        let (tail_expr, block_ty) = if block_diverges {
            // A statement already diverged; block type is Never regardless of tail.
            // Still type-check tail for error reporting, but ignore its type.
            let tail = block.tail_expr.map(|t| Box::new(self.infer_expr(*t)));
            (tail, Type::Never)
        } else if let Some(tail) = block.tail_expr {
            let typed_tail = self.infer_expr(*tail);
            let ty = typed_tail.ty.clone();
            (Some(Box::new(typed_tail)), ty)
        } else {
            (None, Type::Tuple(vec![]))
        };
        self.exit_scope();
        TypedBlock {
            statements: typed_stmts,
            tail_expr,
            ty: block_ty,
            span: block.span,
        }
    }

    fn unify_branch_types(&self, branch_types: &[(Type, Span)], has_else: bool) -> Type {
        if !has_else {
            return Type::Tuple(vec![]);
        }

        let mut is_optional = false;
        let mut concrete = None;

        for (ty, _) in branch_types {
            match ty {
                Type::Never | Type::Error => continue,
                Type::Nil => is_optional = true,
                Type::Optional(inner) => {
                    is_optional = true;
                    if concrete.is_none() {
                        concrete = Some((**inner).clone());
                    }
                }
                _ => {
                    if concrete.is_none() {
                        concrete = Some(ty.clone());
                    }
                }
            }
        }

        if let Some(c) = concrete {
            if is_optional {
                Type::Optional(Box::new(c))
            } else {
                c
            }
        } else if is_optional {
            Type::Nil
        } else {
            Type::Never
        }
    }

    fn wrap_block_tail(&mut self, block: &mut TypedBlock, target_ty: &Type) {
        if let Some(tail) = block.tail_expr.take() {
            match self.check_types_match(*tail, target_ty) {
                Ok(wrapped) => {
                    block.tail_expr = Some(Box::new(wrapped));
                    block.ty = target_ty.clone();
                }
                Err(e) => {
                    self.emit(e);
                }
            }
        }
    }

    fn check_conditional(&mut self, cond: ResolvedCondExpr, span: Span) -> TypedExpr {
        let typed_if_cond = self.check_expr(cond.if_branch.condition, &Type::Bool);
        let typed_if_body = self.check_block(cond.if_branch.body);
        let if_ty = typed_if_body.ty.clone();
        let mut typed_if_branch = TypedIfBranch {
            condition: typed_if_cond,
            body: typed_if_body,
            span: cond.if_branch.span,
        };

        let mut typed_elseif_branches = Vec::new();
        for elseif in cond.elseif_branches {
            let typed_elseif_cond = self.check_expr(elseif.condition, &Type::Bool);
            let typed_elseif_body = self.check_block(elseif.body);
            typed_elseif_branches.push(TypedIfBranch {
                condition: typed_elseif_cond,
                body: typed_elseif_body,
                span: elseif.span,
            });
        }

        let mut typed_else_branch = cond.else_branch.map(|b| self.check_block(b));

        // Collect all branch types
        let mut branch_types: Vec<(Type, Span)> = vec![];
        branch_types.push((if_ty.clone(), cond.if_branch.span));
        for elseif in &typed_elseif_branches {
            branch_types.push((elseif.body.ty.clone(), elseif.span));
        }
        if let Some(else_b) = &typed_else_branch {
            branch_types.push((else_b.ty.clone(), else_b.span));
        }

        let target_ty = self.unify_branch_types(&branch_types, typed_else_branch.is_some());

        self.wrap_block_tail(&mut typed_if_branch.body, &target_ty);
        for elseif in &mut typed_elseif_branches {
            self.wrap_block_tail(&mut elseif.body, &target_ty);
        }
        if let Some(else_b) = &mut typed_else_branch {
            self.wrap_block_tail(else_b, &target_ty);
        }

        TypedExpr {
            kind: TypedExprKind::Conditional(Box::new(TypedCondExpr {
                if_branch: typed_if_branch,
                elseif_branches: typed_elseif_branches,
                else_branch: typed_else_branch,
                span: cond.span,
            })),
            ty: target_ty,
            span,
        }
    }

    fn check_for_loop(&mut self, for_loop: ResolvedForLoop) -> TypedForLoop {
        self.enter_scope();
        self.loop_depth += 1;

        let typed_init = for_loop.init.map(|a| self.check_assignment(a));
        let typed_condition = for_loop.condition.map(|c| self.check_expr(c, &Type::Bool));
        let typed_increment = for_loop.increment.map(|a| self.check_assignment(a));

        let mut typed_body = Vec::new();
        for stmt in for_loop.body {
            let (typed_stmt, _) = self.check_statement(stmt);
            typed_body.push(typed_stmt);
        }

        self.loop_depth -= 1;
        self.exit_scope();
        TypedForLoop {
            init: typed_init,
            condition: typed_condition,
            increment: typed_increment,
            body: typed_body,
            span: for_loop.span,
        }
    }

    fn check_for_in_loop(&mut self, for_in_loop: ResolvedForInLoop) -> TypedForInLoop {
        let typed_iterable = self.infer_expr(for_in_loop.iterable);

        let element_type = match &typed_iterable.ty {
            Type::List(elem_ty) => (**elem_ty).clone(),
            Type::Map(key_ty, val_ty) => Type::Tuple(vec![(**key_ty).clone(), (**val_ty).clone()]),
            _ => {
                self.emit(TypeError::NotIterable {
                    ty: typed_iterable.ty.clone(),
                    span: for_in_loop.span,
                });
                Type::Error
            }
        };

        self.enter_scope();
        self.loop_depth += 1;

        let typed_pattern = self.check_pattern(for_in_loop.pattern, &element_type);

        let mut typed_body = Vec::new();
        for stmt in for_in_loop.body {
            let (typed_stmt, _) = self.check_statement(stmt);
            typed_body.push(typed_stmt);
        }

        self.loop_depth -= 1;
        self.exit_scope();
        TypedForInLoop {
            pattern: typed_pattern,
            iterable: typed_iterable,
            body: typed_body,
            span: for_in_loop.span,
        }
    }

    fn check_pattern(&mut self, pattern: ResolvedPattern, expected_type: &Type) -> TypedPattern {
        let error_pattern = |span| TypedPattern {
            kind: TypedPatternKind::Error,
            ty: expected_type.clone(),
            span,
        };

        match pattern.kind {
            ResolvedPatternKind::Var(name_id, name) => {
                self.add_var(name_id, expected_type.clone());
                TypedPattern {
                    kind: TypedPatternKind::Var(name_id, name),
                    ty: expected_type.clone(),
                    span: pattern.span,
                }
            }
            ResolvedPatternKind::Wildcard => TypedPattern {
                kind: TypedPatternKind::Wildcard,
                ty: expected_type.clone(),
                span: pattern.span,
            },
            ResolvedPatternKind::Tuple(patterns) => {
                if let Type::Tuple(types) = expected_type {
                    if patterns.len() != types.len() {
                        self.emit(TypeError::PatternMismatch {
                            pattern_ty: format!("tuple of {} elements", patterns.len()),
                            iterable_ty: expected_type.clone(),
                            span: pattern.span,
                        });
                        return error_pattern(pattern.span);
                    }
                    let mut typed_patterns = Vec::new();
                    for (pat, ty) in patterns.into_iter().zip(types.iter()) {
                        typed_patterns.push(self.check_pattern(pat, ty));
                    }
                    TypedPattern {
                        kind: TypedPatternKind::Tuple(typed_patterns),
                        ty: expected_type.clone(),
                        span: pattern.span,
                    }
                } else {
                    self.emit(TypeError::PatternMismatch {
                        pattern_ty: "tuple".to_string(),
                        iterable_ty: expected_type.clone(),
                        span: pattern.span,
                    });
                    error_pattern(pattern.span)
                }
            }
            ResolvedPatternKind::Variant(enum_id, variant_name, payload_pat) => {
                if let Type::Enum(expected_enum_id, _) = expected_type {
                    if enum_id != *expected_enum_id {
                        self.emit(TypeError::Mismatch {
                            expected: expected_type.clone(),
                            found: Type::Enum(enum_id, "".to_string()),
                            span: pattern.span,
                        });
                        return error_pattern(pattern.span);
                    }

                    let (resolved_id, _enum_name, variant_def) = match self
                        .get_resolved_enum_variant(enum_id, &variant_name, pattern.span)
                    {
                        Ok(v) => v,
                        Err(e) => {
                            self.emit(e);
                            return error_pattern(pattern.span);
                        }
                    };

                    debug_assert_eq!(resolved_id, enum_id, "enum ID mismatch after lookup");

                    let typed_payload = match (&variant_def.payload, payload_pat) {
                        (Some(payload_ty_def), Some(pat)) => {
                            match self.resolve_type(payload_ty_def) {
                                Ok(payload_ty) => {
                                    Some(Box::new(self.check_pattern(*pat, &payload_ty)))
                                }
                                Err(e) => {
                                    self.emit(e);
                                    None
                                }
                            }
                        }
                        (None, None) => None,
                        (Some(_), None) => {
                            self.emit(TypeError::VariantExpectsPayload {
                                variant: variant_name.clone(),
                                span: pattern.span,
                            });
                            None
                        }
                        (None, Some(_)) => {
                            self.emit(TypeError::VariantExpectsNoPayload {
                                variant: variant_name.clone(),
                                span: pattern.span,
                            });
                            None
                        }
                    };

                    TypedPattern {
                        kind: TypedPatternKind::Variant(enum_id, variant_name, typed_payload),
                        ty: expected_type.clone(),
                        span: pattern.span,
                    }
                } else {
                    self.emit(TypeError::PatternMismatch {
                        pattern_ty: "enum variant".to_string(),
                        iterable_ty: expected_type.clone(),
                        span: pattern.span,
                    });
                    error_pattern(pattern.span)
                }
            }
            ResolvedPatternKind::Error => TypedPattern {
                kind: TypedPatternKind::Error,
                ty: expected_type.clone(),
                span: pattern.span,
            },
        }
    }

    fn check_assignment(&mut self, assign: ResolvedAssignment) -> TypedAssignment {
        let span = assign.span;

        // Resolve the overall type: annotation or infer from RHS
        let (typed_value, overall_ty) = if let Some(def) = assign.type_def {
            match self.resolve_type(&def) {
                Ok(expected_ty) => {
                    let typed_val = self.check_expr(assign.value, &expected_ty);
                    (typed_val, expected_ty)
                }
                Err(e) => {
                    self.emit(e);
                    let typed_val = self.infer_expr(assign.value);
                    let ty = typed_val.ty.clone();
                    (typed_val, ty)
                }
            }
        } else if let Some(expected_ty) = self.target_expected_type(&assign.targets, span) {
            let typed_val = self.check_expr(assign.value, &expected_ty);
            (typed_val, expected_ty)
        } else {
            let typed_val = self.infer_expr(assign.value);
            let ty = typed_val.ty.clone();
            (typed_val, ty)
        };

        // Single target: match against overall type directly
        // Multi target: expect a tuple type and match element-wise
        let targets = if assign.targets.len() == 1 {
            vec![self.check_assign_item(assign.targets.into_iter().next().unwrap(), &overall_ty, span)]
        } else {
            match &overall_ty {
                Type::Tuple(types) => {
                    if assign.targets.len() != types.len() {
                        self.emit(TypeError::PatternMismatch {
                            pattern_ty: format!("tuple of {} elements", assign.targets.len()),
                            iterable_ty: overall_ty.clone(),
                            span,
                        });
                        assign.targets
                            .into_iter()
                            .map(|item| self.check_assign_item(item, &Type::Error, span))
                            .collect()
                    } else {
                        assign.targets
                            .into_iter()
                            .zip(types.iter())
                            .map(|(item, ty)| self.check_assign_item(item, ty, span))
                            .collect()
                    }
                }
                Type::Error => {
                    assign.targets
                        .into_iter()
                        .map(|item| self.check_assign_item(item, &Type::Error, span))
                        .collect()
                }
                _ => {
                    self.emit(TypeError::NotATuple {
                        ty: overall_ty.clone(),
                        span,
                    });
                    assign.targets
                        .into_iter()
                        .map(|item| self.check_assign_item(item, &Type::Error, span))
                        .collect()
                }
            }
        };

        TypedAssignment {
            targets,
            value: typed_value,
            span,
        }
    }

    fn target_expected_type(&self, targets: &[ResolvedAssignItem], span: Span) -> Option<Type> {
        if targets.len() == 1 {
            match &targets[0] {
                ResolvedAssignItem::Existing(id, _) => {
                    self.get_var_type(*id, span).ok()
                }
                _ => None,
            }
        } else {
            let mut types = Vec::new();
            for target in targets {
                match target {
                    ResolvedAssignItem::Existing(id, _) => {
                        match self.get_var_type(*id, span) {
                            Ok(ty) => types.push(ty),
                            Err(_) => return None,
                        }
                    }
                    _ => return None,
                }
            }
            Some(Type::Tuple(types))
        }
    }

    fn check_assign_item(
        &mut self,
        item: ResolvedAssignItem,
        expected_ty: &Type,
        span: Span,
    ) -> TypedAssignItem {
        match item {
            ResolvedAssignItem::Existing(id, name) => {
                // Look up the existing variable's type and check it matches
                let var_ty = self.get_var_type(id, span).unwrap_or(Type::Error);
                if var_ty != Type::Error && *expected_ty != Type::Error && var_ty != *expected_ty {
                    self.emit(TypeError::Mismatch {
                        expected: var_ty.clone(),
                        found: expected_ty.clone(),
                        span,
                    });
                }
                TypedAssignItem::Existing(id, name, var_ty)
            }
            ResolvedAssignItem::Declare(id, name) => {
                self.add_var(id, expected_ty.clone());
                TypedAssignItem::Declare(id, name, expected_ty.clone())
            }
            ResolvedAssignItem::Wildcard => {
                TypedAssignItem::Wildcard(expected_ty.clone())
            }
            ResolvedAssignItem::Nested(items) => {
                let typed_items = match expected_ty {
                    Type::Tuple(types) => {
                        if items.len() != types.len() {
                            self.emit(TypeError::PatternMismatch {
                                pattern_ty: format!("tuple of {} elements", items.len()),
                                iterable_ty: expected_ty.clone(),
                                span,
                            });
                            items
                                .into_iter()
                                .map(|i| self.check_assign_item(i, &Type::Error, span))
                                .collect()
                        } else {
                            items
                                .into_iter()
                                .zip(types.iter())
                                .map(|(i, ty)| self.check_assign_item(i, ty, span))
                                .collect()
                        }
                    }
                    Type::Error => {
                        items
                            .into_iter()
                            .map(|i| self.check_assign_item(i, &Type::Error, span))
                            .collect()
                    }
                    _ => {
                        self.emit(TypeError::NotATuple {
                            ty: expected_ty.clone(),
                            span,
                        });
                        items
                            .into_iter()
                            .map(|i| self.check_assign_item(i, &Type::Error, span))
                            .collect()
                    }
                };
                TypedAssignItem::Nested(typed_items, expected_ty.clone())
            }
        }
    }

    fn resolve_struct_definition(
        &self,
        start_id: NameId,
        span: Span,
        field_context: &str,
    ) -> Result<(NameId, &Vec<crate::analysis::resolver::ResolvedFieldDef>), TypeError> {
        let mut current_id = start_id;

        loop {
            let type_def = self
                .type_defs
                .get(&current_id)
                .ok_or(TypeError::UndefinedType(span))?;

            match type_def {
                TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Struct(fields)) => {
                    return Ok((current_id, fields));
                }
                TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Alias(
                    ResolvedTypeDef::Named(next_id),
                )) => {
                    current_id = *next_id;
                }
                _ => {
                    let struct_name = self
                        .struct_names
                        .get(&current_id)
                        .cloned()
                        .unwrap_or_else(|| format!("type_{}", current_id.0));

                    return Err(TypeError::NotAStruct {
                        ty: Type::Struct(current_id, struct_name),
                        field_name: field_context.to_string(),
                        span,
                    });
                }
            }
        }
    }

    fn resolve_enum_definition(
        &self,
        start_id: NameId,
        span: Span,
        _variant_context: &str,
    ) -> Result<(NameId, &Vec<crate::analysis::resolver::ResolvedEnumVariant>), TypeError> {
        let mut current_id = start_id;

        loop {
            let type_def = self
                .type_defs
                .get(&current_id)
                .ok_or(TypeError::UndefinedType(span))?;

            match type_def {
                TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Enum(variants)) => {
                    return Ok((current_id, variants));
                }
                TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Alias(
                    ResolvedTypeDef::Named(next_id),
                )) => {
                    current_id = *next_id;
                }
                _ => {
                    let enum_name = self
                        .enum_names
                        .get(&current_id)
                        .cloned()
                        .unwrap_or_else(|| format!("type_{}", current_id.0));

                    return Err(TypeError::EnumNotFound {
                        name: enum_name,
                        span,
                    });
                }
            }
        }
    }

    fn check_variant_lit(
        &mut self,
        enum_id: NameId,
        variant_name: String,
        payload: Option<Box<ResolvedExpr>>,
        span: Span,
    ) -> TypedExpr {
        let (resolved_id, enum_name, variant_def) =
            match self.get_resolved_enum_variant(enum_id, &variant_name, span) {
                Ok(v) => v,
                Err(e) => {
                    self.emit(e);
                    return self.error_expr(span);
                }
            };

        let typed_payload = match (&variant_def.payload, payload) {
            (Some(payload_ty_def), Some(expr)) => {
                match self.resolve_type(payload_ty_def) {
                    Ok(payload_ty) => Some(Box::new(self.check_expr(*expr, &payload_ty))),
                    Err(e) => {
                        self.emit(e);
                        // Still type-check the payload expr for further errors
                        let _ = self.infer_expr(*expr);
                        None
                    }
                }
            }
            (None, None) => None,
            (Some(expected), None) => {
                match self.resolve_type(expected) {
                    Ok(expected_ty) => {
                        self.emit(TypeError::VariantPayloadMismatch {
                            variant: variant_name.clone(),
                            expected: expected_ty,
                            found: Type::Tuple(vec![]),
                            span,
                        });
                    }
                    Err(e) => self.emit(e),
                }
                None
            }
            (None, Some(expr)) => {
                let _ = self.infer_expr(*expr);
                self.emit(TypeError::VariantExpectsNoPayload {
                    variant: variant_name.clone(),
                    span,
                });
                None
            }
        };

        TypedExpr {
            kind: TypedExprKind::VariantLit(resolved_id, variant_name, typed_payload),
            ty: Type::Enum(resolved_id, enum_name),
            span,
        }
    }

    fn check_match(
        &mut self,
        scrutinee: ResolvedExpr,
        arms: Vec<ResolvedMatchArm>,
        span: Span,
    ) -> TypedExpr {
        let typed_scrutinee = self.infer_expr(scrutinee);
        let scrutinee_ty = typed_scrutinee.ty.clone();

        if !matches!(scrutinee_ty, Type::Enum(_, _) | Type::Error) {
            self.emit(TypeError::MatchScrutineeNotEnum {
                ty: scrutinee_ty.clone(),
                span: typed_scrutinee.span,
            });
        }

        let mut typed_arms = Vec::new();
        let mut match_ty: Option<Type> = None;

        for arm in arms {
            self.enter_scope();
            let typed_pattern = self.check_pattern(arm.pattern, &scrutinee_ty);
            let typed_body = self.check_block(arm.body);
            let arm_ty = typed_body.ty.clone();
            self.exit_scope();

            // Skip Never and Error arms when determining match type
            if arm_ty != Type::Never && arm_ty != Type::Error {
                if let Some(prev_ty) = &match_ty {
                    if *prev_ty != arm_ty && !matches!(prev_ty, Type::Error) {
                        self.emit(TypeError::MatchArmTypeMismatch {
                            expected: prev_ty.clone(),
                            found: arm_ty,
                            span: arm.span,
                        });
                    }
                } else {
                    match_ty = Some(arm_ty);
                }
            }

            typed_arms.push(TypedMatchArm {
                pattern: typed_pattern,
                body: typed_body,
                span: arm.span,
            });
        }

        let final_ty = match_ty.unwrap_or(Type::Never);

        TypedExpr {
            kind: TypedExprKind::Match(Box::new(typed_scrutinee), typed_arms),
            ty: final_ty,
            span,
        }
    }

    fn check_expr(&mut self, expr: ResolvedExpr, expected: &Type) -> TypedExpr {
        let span = expr.span;
        match (&expr.kind, expected) {
            (ResolvedExprKind::NilLit, Type::Optional(_)) => TypedExpr {
                kind: TypedExprKind::NilLit,
                ty: expected.clone(),
                span,
            },

            (ResolvedExprKind::ListLit(items), Type::List(elem_ty)) => {
                let mut typed_items = Vec::new();
                for item in items.clone() {
                    typed_items.push(self.check_expr(item, elem_ty));
                }
                TypedExpr {
                    kind: TypedExprKind::ListLit(typed_items),
                    ty: expected.clone(),
                    span,
                }
            }

            (ResolvedExprKind::MapLit(pairs), Type::Map(key_ty, val_ty)) => {
                let mut typed_pairs = Vec::new();
                for (key, val) in pairs.clone() {
                    let typed_key = self.check_expr(key, key_ty);
                    let typed_val = self.check_expr(val, val_ty);
                    typed_pairs.push((typed_key, typed_val));
                }
                TypedExpr {
                    kind: TypedExprKind::MapLit(typed_pairs),
                    ty: expected.clone(),
                    span,
                }
            }

            (ResolvedExprKind::StructLit(struct_id, fields), Type::Struct(expected_id, _))
                if *struct_id == *expected_id =>
            {
                self.check_struct_literal(*struct_id, fields.clone(), span)
            }

            (ResolvedExprKind::MakeChannel, Type::Chan(_)) => TypedExpr {
                kind: TypedExprKind::MakeChannel,
                ty: expected.clone(),
                span,
            },

            _ => {
                let inferred = self.infer_expr(expr);
                match self.check_types_match(inferred, expected) {
                    Ok(typed) => typed,
                    Err(e) => {
                        self.emit(e);
                        self.error_expr(span)
                    }
                }
            }
        }
    }

    fn infer_expr(&mut self, expr: ResolvedExpr) -> TypedExpr {
        let span = expr.span;
        match expr.kind {
            ResolvedExprKind::Var(name_id, name) => match self.get_var_type(name_id, span) {
                Ok(ty) => TypedExpr {
                    kind: TypedExprKind::Var(name_id, name),
                    ty,
                    span,
                },
                Err(e) => {
                    self.emit(e);
                    self.error_expr(span)
                }
            },
            ResolvedExprKind::IntLit(i) => TypedExpr {
                kind: TypedExprKind::IntLit(i),
                ty: Type::Int,
                span,
            },
            ResolvedExprKind::StringLit(s) => TypedExpr {
                kind: TypedExprKind::StringLit(s),
                ty: Type::String,
                span,
            },
            ResolvedExprKind::FString(exprs) => {
                let mut curr_expr: Option<TypedExpr> = None;
                for e in exprs {
                    let mut typed_e = self.infer_expr(e);

                    if typed_e.ty == Type::Int {
                        typed_e = TypedExpr {
                            span: typed_e.span,
                            ty: Type::String,
                            kind: TypedExprKind::FuncCall(TypedFuncCall::Builtin(
                                crate::analysis::resolver::BuiltinFn::IntToString,
                                vec![typed_e],
                                Type::String,
                            )),
                        };
                    } else if typed_e.ty == Type::Bool {
                        typed_e = TypedExpr {
                            span: typed_e.span,
                            ty: Type::String,
                            kind: TypedExprKind::FuncCall(TypedFuncCall::Builtin(
                                crate::analysis::resolver::BuiltinFn::BoolToString,
                                vec![typed_e],
                                Type::String,
                            )),
                        };
                    } else if typed_e.ty != Type::String && typed_e.ty != Type::Error {
                        self.emit(TypeError::Mismatch {
                            expected: Type::String,
                            found: typed_e.ty.clone(),
                            span: typed_e.span,
                        });
                    }

                    if let Some(prev) = curr_expr {
                        let combined_span = prev.span.union(typed_e.span);
                        curr_expr = Some(TypedExpr {
                            kind: TypedExprKind::BinOp(
                                BinOp::Add,
                                Box::new(prev),
                                Box::new(typed_e),
                            ),
                            ty: Type::String,
                            span: combined_span,
                        });
                    } else {
                        curr_expr = Some(typed_e);
                    }
                }

                match curr_expr {
                    Some(e) => e,
                    None => {
                        self.emit(TypeError::Mismatch {
                            expected: Type::String,
                            found: Type::Tuple(vec![]),
                            span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::PersistData(expr) => {
                let typed_e = self.infer_expr(*expr);
                if !is_trivially_copyable(&typed_e.ty, &self.trivially_copyable) {
                    self.emit(TypeError::NonTriviallyCopyable {
                        ty: typed_e.ty.clone(),
                        span,
                    });
                }
                TypedExpr {
                    kind: TypedExprKind::PersistData(Box::new(typed_e)),
                    ty: Type::Tuple(vec![]),
                    span,
                }
            }
            ResolvedExprKind::RetrieveData(type_def) => match self.resolve_type(&type_def) {
                Ok(inner_type) => {
                    if !is_trivially_copyable(&inner_type, &self.trivially_copyable) {
                        self.emit(TypeError::NonTriviallyCopyable {
                            ty: inner_type.clone(),
                            span,
                        });
                    }
                    TypedExpr {
                        kind: TypedExprKind::RetrieveData(inner_type.clone()),
                        ty: Type::Optional(Box::new(inner_type)),
                        span,
                    }
                }
                Err(e) => {
                    self.emit(e);
                    self.error_expr(span)
                }
            },
            ResolvedExprKind::DiscardData => TypedExpr {
                kind: TypedExprKind::DiscardData,
                ty: Type::Tuple(vec![]),
                span,
            },
            ResolvedExprKind::BoolLit(b) => TypedExpr {
                kind: TypedExprKind::BoolLit(b),
                ty: Type::Bool,
                span,
            },
            ResolvedExprKind::NilLit => TypedExpr {
                kind: TypedExprKind::NilLit,
                ty: Type::Nil,
                span,
            },
            ResolvedExprKind::BinOp(op, left, right) => self.infer_binop(op, *left, *right, span),
            ResolvedExprKind::Not(e) => {
                let typed_e = self.check_expr(*e, &Type::Bool);
                TypedExpr {
                    kind: TypedExprKind::Not(Box::new(typed_e)),
                    ty: Type::Bool,
                    span,
                }
            }
            ResolvedExprKind::Negate(e) => {
                let typed_e = self.check_expr(*e, &Type::Int);
                TypedExpr {
                    kind: TypedExprKind::Negate(Box::new(typed_e)),
                    ty: Type::Int,
                    span,
                }
            }
            ResolvedExprKind::VariantLit(enum_id, variant_name, payload) => {
                self.check_variant_lit(enum_id, variant_name, payload, span)
            }
            ResolvedExprKind::Match(scrutinee, arms) => self.check_match(*scrutinee, arms, span),
            ResolvedExprKind::Conditional(cond) => self.check_conditional(*cond, span),
            ResolvedExprKind::FuncCall(call) => match call {
                ResolvedFuncCall::User(user_call) => {
                    let sig = match self.func_signatures.get(&user_call.name) {
                        Some(sig) => sig.clone(),
                        None => {
                            self.emit(TypeError::UndefinedType(user_call.span));
                            return self.error_expr(span);
                        }
                    };

                    let is_free = !self.role_func_name_ids.contains(&user_call.name);
                    let typed_call = self.check_user_func_call(user_call, &sig, is_free);
                    let return_ty = if sig.is_sync {
                        sig.return_type
                    } else {
                        Type::Chan(Box::new(sig.return_type.clone()))
                    };

                    TypedExpr {
                        kind: TypedExprKind::FuncCall(TypedFuncCall::User(typed_call)),
                        ty: return_ty,
                        span,
                    }
                }
                ResolvedFuncCall::Builtin(builtin, args, span) => {
                    self.check_builtin_call(builtin, args, span)
                }
            },

            ResolvedExprKind::MapLit(pairs) => self.infer_map_literal(pairs, span),
            ResolvedExprKind::ListLit(items) => self.infer_list_literal(items, span),
            ResolvedExprKind::TupleLit(items) => {
                let mut typed_items = Vec::new();
                let mut types = Vec::new();
                for item in items {
                    let typed_item = self.infer_expr(item);
                    types.push(typed_item.ty.clone());
                    typed_items.push(typed_item);
                }
                TypedExpr {
                    kind: TypedExprKind::TupleLit(typed_items),
                    ty: Type::Tuple(types),
                    span,
                }
            }
            ResolvedExprKind::Append(list_expr, item_expr) => {
                let typed_list = self.infer_expr(*list_expr);
                let typed_item = self.infer_expr(*item_expr);

                let (list_ty, coerced_item) = match typed_list.ty.clone() {
                    Type::List(elem_ty) => match self.check_types_match(typed_item, &elem_ty) {
                        Ok(coerced) => (Type::List(elem_ty), coerced),
                        Err(e) => {
                            self.emit(e);
                            return self.error_expr(span);
                        }
                    },
                    Type::EmptyList => (Type::List(Box::new(typed_item.ty.clone())), typed_item),
                    Type::Error => return self.error_expr(span),
                    _ => {
                        self.emit(TypeError::NotAList {
                            ty: typed_list.ty,
                            span,
                        });
                        return self.error_expr(span);
                    }
                };

                TypedExpr {
                    kind: TypedExprKind::Append(Box::new(typed_list), Box::new(coerced_item)),
                    ty: list_ty,
                    span,
                }
            }
            ResolvedExprKind::Prepend(item_expr, list_expr) => {
                let typed_item = self.infer_expr(*item_expr);
                let typed_list = self.infer_expr(*list_expr);

                let (list_ty, coerced_item) = match typed_list.ty.clone() {
                    Type::List(elem_ty) => match self.check_types_match(typed_item, &elem_ty) {
                        Ok(coerced) => (Type::List(elem_ty), coerced),
                        Err(e) => {
                            self.emit(e);
                            return self.error_expr(span);
                        }
                    },
                    Type::EmptyList => (Type::List(Box::new(typed_item.ty.clone())), typed_item),
                    Type::Error => return self.error_expr(span),
                    _ => {
                        self.emit(TypeError::NotAList {
                            ty: typed_list.ty,
                            span,
                        });
                        return self.error_expr(span);
                    }
                };

                TypedExpr {
                    kind: TypedExprKind::Prepend(Box::new(coerced_item), Box::new(typed_list)),
                    ty: list_ty,
                    span,
                }
            }
            ResolvedExprKind::Store(collection, key, value) => {
                let typed_collection = self.infer_expr(*collection);

                match typed_collection.ty.clone() {
                    Type::List(elem_ty) => {
                        let typed_key = self.infer_expr(*key);
                        if let Err(e) =
                            self.check_type_compatibility(&Type::Int, &typed_key.ty, span)
                        {
                            self.emit(e);
                        }

                        let typed_value = self.check_expr(*value, &elem_ty);

                        TypedExpr {
                            kind: TypedExprKind::Store(
                                Box::new(typed_collection.clone()),
                                Box::new(typed_key),
                                Box::new(typed_value),
                            ),
                            ty: typed_collection.ty,
                            span,
                        }
                    }
                    Type::Map(key_ty, val_ty) => {
                        let typed_key = self.check_expr(*key, &key_ty);
                        let typed_value = self.check_expr(*value, &val_ty);

                        TypedExpr {
                            kind: TypedExprKind::Store(
                                Box::new(typed_collection.clone()),
                                Box::new(typed_key),
                                Box::new(typed_value),
                            ),
                            ty: typed_collection.ty,
                            span,
                        }
                    }
                    Type::Struct(struct_id, _) => {
                        let key_span = key.span;
                        let field_name = if let ResolvedExprKind::StringLit(s) = &key.kind {
                            s.clone()
                        } else {
                            let typed_key = self.infer_expr(*key);
                            self.emit(TypeError::InvalidStructKeyType {
                                found: typed_key.ty,
                                span: key_span,
                            });
                            let typed_value = self.infer_expr(*value);
                            let _ = typed_value;
                            return self.error_expr(span);
                        };

                        let field_ty = match self.get_field_type(struct_id, &field_name, span) {
                            Ok(ty) => ty,
                            Err(e) => {
                                self.emit(e);
                                let _ = self.infer_expr(*value);
                                return self.error_expr(span);
                            }
                        };
                        let typed_value = self.check_expr(*value, &field_ty);

                        let typed_key = TypedExpr {
                            kind: TypedExprKind::StringLit(field_name),
                            ty: Type::String,
                            span: key_span,
                        };

                        TypedExpr {
                            kind: TypedExprKind::Store(
                                Box::new(typed_collection.clone()),
                                Box::new(typed_key),
                                Box::new(typed_value),
                            ),
                            ty: typed_collection.ty,
                            span,
                        }
                    }
                    Type::Error => self.error_expr(span),
                    _ => {
                        self.emit(TypeError::StoreOnInvalidType {
                            ty: typed_collection.ty,
                            span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::Head(list_expr) => {
                let typed_list = self.infer_expr(*list_expr);
                if let Type::List(elem_ty) = typed_list.ty.clone() {
                    TypedExpr {
                        kind: TypedExprKind::Head(Box::new(typed_list)),
                        ty: *elem_ty,
                        span,
                    }
                } else if matches!(typed_list.ty, Type::Error) {
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::NotAList {
                        ty: typed_list.ty,
                        span,
                    });
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::Tail(list_expr) => {
                let typed_list = self.infer_expr(*list_expr);
                if let Type::List(_) = &typed_list.ty {
                    TypedExpr {
                        kind: TypedExprKind::Tail(Box::new(typed_list.clone())),
                        ty: typed_list.ty,
                        span,
                    }
                } else if matches!(typed_list.ty, Type::Error) {
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::NotAList {
                        ty: typed_list.ty,
                        span,
                    });
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::Len(expr) => {
                let typed_e = self.infer_expr(*expr);
                match typed_e.ty {
                    Type::List(_) | Type::Map(_, _) | Type::String | Type::Error => TypedExpr {
                        kind: TypedExprKind::Len(Box::new(typed_e)),
                        ty: Type::Int,
                        span,
                    },
                    _ => {
                        self.emit(TypeError::InvalidUnaryOp {
                            op: "len",
                            ty: typed_e.ty.clone(),
                            span: typed_e.span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::Min(left, right) => {
                let typed_left = self.check_expr(*left, &Type::Int);
                let typed_right = self.check_expr(*right, &Type::Int);
                TypedExpr {
                    kind: TypedExprKind::Min(Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Int,
                    span,
                }
            }
            ResolvedExprKind::Exists(map_expr, key_expr) => {
                let typed_map = self.infer_expr(*map_expr);
                let typed_key = self.infer_expr(*key_expr);

                if let Type::Map(expected_key_ty, _) = &typed_map.ty {
                    if let Err(e) =
                        self.check_type_compatibility(expected_key_ty, &typed_key.ty, span)
                    {
                        self.emit(e);
                    }
                    TypedExpr {
                        kind: TypedExprKind::Exists(Box::new(typed_map), Box::new(typed_key)),
                        ty: Type::Bool,
                        span,
                    }
                } else if matches!(typed_map.ty, Type::Error) {
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::InvalidUnaryOp {
                        op: "exists",
                        ty: typed_map.ty,
                        span,
                    });
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::Erase(map_expr, key_expr) => {
                let typed_map = self.infer_expr(*map_expr);
                let typed_key = self.infer_expr(*key_expr);

                if let Type::Map(expected_key_ty, _) = &typed_map.ty {
                    if let Err(e) =
                        self.check_type_compatibility(expected_key_ty, &typed_key.ty, span)
                    {
                        self.emit(e);
                    }
                    TypedExpr {
                        kind: TypedExprKind::Erase(
                            Box::new(typed_map.clone()),
                            Box::new(typed_key),
                        ),
                        ty: typed_map.ty,
                        span,
                    }
                } else if matches!(typed_map.ty, Type::Error) {
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::InvalidUnaryOp {
                        op: "erase",
                        ty: typed_map.ty,
                        span,
                    });
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::MakeChannel => TypedExpr {
                kind: TypedExprKind::MakeChannel,
                ty: Type::UnknownChannel,
                span,
            },
            ResolvedExprKind::Send(chan_expr, val_expr) => {
                if self.current_func_is_sync {
                    self.emit(TypeError::SendInSyncFunc { span });
                    let _ = self.infer_expr(*chan_expr);
                    let _ = self.infer_expr(*val_expr);
                    return self.error_expr(span);
                }

                let typed_chan = self.infer_expr(*chan_expr);

                if let Type::Chan(inner_ty) = &typed_chan.ty {
                    let typed_val = self.check_expr(*val_expr, inner_ty);
                    TypedExpr {
                        kind: TypedExprKind::Send(Box::new(typed_chan), Box::new(typed_val)),
                        ty: Type::Tuple(vec![]),
                        span,
                    }
                } else if matches!(typed_chan.ty, Type::Error) {
                    let _ = self.infer_expr(*val_expr);
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::NotAChannel {
                        ty: typed_chan.ty,
                        span,
                    });
                    let _ = self.infer_expr(*val_expr);
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::Recv(chan_expr) => {
                if self.current_func_is_sync {
                    self.emit(TypeError::RecvInSyncFunc { span });
                    let _ = self.infer_expr(*chan_expr);
                    return self.error_expr(span);
                }

                let typed_chan = self.infer_expr(*chan_expr);

                if let Type::Chan(inner_ty) = typed_chan.ty.clone() {
                    TypedExpr {
                        kind: TypedExprKind::Recv(Box::new(typed_chan)),
                        ty: *inner_ty,
                        span,
                    }
                } else if matches!(typed_chan.ty, Type::Error) {
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::NotAChannel {
                        ty: typed_chan.ty,
                        span,
                    });
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::SetTimer(label) => TypedExpr {
                kind: TypedExprKind::SetTimer(label),
                ty: Type::Chan(Box::new(Type::Tuple(vec![]))),
                span,
            },
            ResolvedExprKind::Index(target, index) => self.infer_index(*target, *index, span),
            ResolvedExprKind::Slice(target, start, end) => {
                let typed_target = self.infer_expr(*target);
                let typed_start = self.check_expr(*start, &Type::Int);
                let typed_end = self.check_expr(*end, &Type::Int);

                match typed_target.ty.clone() {
                    Type::List(_) | Type::String => TypedExpr {
                        kind: TypedExprKind::Slice(
                            Box::new(typed_target.clone()),
                            Box::new(typed_start),
                            Box::new(typed_end),
                        ),
                        ty: typed_target.ty,
                        span,
                    },
                    Type::Error => self.error_expr(span),
                    _ => {
                        self.emit(TypeError::NotIndexable {
                            ty: typed_target.ty,
                            span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::TupleAccess(tuple_expr, index) => {
                let typed_tuple = self.infer_expr(*tuple_expr);
                if let Type::Tuple(types) = typed_tuple.ty.clone() {
                    if index < types.len() {
                        TypedExpr {
                            kind: TypedExprKind::TupleAccess(Box::new(typed_tuple), index),
                            ty: types[index].clone(),
                            span,
                        }
                    } else {
                        self.emit(TypeError::TupleIndexOutOfBounds {
                            index,
                            size: types.len(),
                            span,
                        });
                        self.error_expr(span)
                    }
                } else if matches!(typed_tuple.ty, Type::Error) {
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::NotATuple {
                        ty: typed_tuple.ty,
                        span,
                    });
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::FieldAccess(struct_expr, field_name) => {
                let typed_struct = self.infer_expr(*struct_expr);
                if matches!(typed_struct.ty, Type::Error) {
                    return self.error_expr(span);
                }
                if matches!(typed_struct.ty, Type::Optional(_)) {
                    self.emit(TypeError::NotAStruct {
                        ty: typed_struct.ty,
                        field_name: field_name.clone(),
                        span,
                    });
                    return self.error_expr(span);
                }
                if let Type::Struct(struct_id, _) = typed_struct.ty {
                    match self.get_field_type(struct_id, &field_name, span) {
                        Ok(field_ty) => TypedExpr {
                            kind: TypedExprKind::FieldAccess(Box::new(typed_struct), field_name),
                            ty: field_ty,
                            span,
                        },
                        Err(e) => {
                            self.emit(e);
                            self.error_expr(span)
                        }
                    }
                } else {
                    self.emit(TypeError::NotAStruct {
                        ty: typed_struct.ty,
                        field_name: field_name.clone(),
                        span,
                    });
                    self.error_expr(span)
                }
            }
            ResolvedExprKind::SafeFieldAccess(struct_expr, field_name) => {
                let typed_struct = self.infer_expr(*struct_expr);
                if matches!(typed_struct.ty, Type::Error) {
                    return self.error_expr(span);
                }
                match &typed_struct.ty {
                    Type::Optional(inner) => {
                        if let Type::Struct(struct_id, _) = inner.as_ref() {
                            match self.get_field_type(*struct_id, &field_name, span) {
                                Ok(field_ty) => TypedExpr {
                                    kind: TypedExprKind::SafeFieldAccess(
                                        Box::new(typed_struct),
                                        field_name,
                                    ),
                                    ty: Type::Optional(Box::new(field_ty)),
                                    span,
                                },
                                Err(e) => {
                                    self.emit(e);
                                    self.error_expr(span)
                                }
                            }
                        } else {
                            self.emit(TypeError::NotAStruct {
                                ty: *inner.clone(),
                                field_name: field_name.clone(),
                                span,
                            });
                            self.error_expr(span)
                        }
                    }
                    _ => {
                        self.emit(TypeError::SafeNavOnNonOptional {
                            ty: typed_struct.ty,
                            span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::SafeIndex(target, index) => {
                let typed_target = self.infer_expr(*target);
                if matches!(typed_target.ty, Type::Error) {
                    return self.error_expr(span);
                }
                let target_ty = typed_target.ty.clone();
                match &target_ty {
                    Type::Optional(inner) => {
                        let typed_index = self.infer_expr(*index);
                        if matches!(typed_index.ty, Type::Error) {
                            return self.error_expr(span);
                        }
                        match inner.as_ref() {
                            Type::List(elem_ty) => {
                                if let Err(e) =
                                    self.check_type_compatibility(&Type::Int, &typed_index.ty, span)
                                {
                                    self.emit(e);
                                    return self.error_expr(span);
                                }
                                TypedExpr {
                                    kind: TypedExprKind::SafeIndex(
                                        Box::new(typed_target),
                                        Box::new(typed_index),
                                    ),
                                    ty: Type::Optional(elem_ty.clone()),
                                    span,
                                }
                            }
                            Type::Map(key_ty, val_ty) => {
                                if let Err(e) =
                                    self.check_type_compatibility(key_ty, &typed_index.ty, span)
                                {
                                    self.emit(e);
                                    return self.error_expr(span);
                                }
                                TypedExpr {
                                    kind: TypedExprKind::SafeIndex(
                                        Box::new(typed_target),
                                        Box::new(typed_index),
                                    ),
                                    ty: Type::Optional(val_ty.clone()),
                                    span,
                                }
                            }
                            Type::String => {
                                if let Err(e) =
                                    self.check_type_compatibility(&Type::Int, &typed_index.ty, span)
                                {
                                    self.emit(e);
                                    return self.error_expr(span);
                                }
                                TypedExpr {
                                    kind: TypedExprKind::SafeIndex(
                                        Box::new(typed_target),
                                        Box::new(typed_index),
                                    ),
                                    ty: Type::Optional(Box::new(Type::String)),
                                    span,
                                }
                            }
                            _ => {
                                self.emit(TypeError::NotIndexable {
                                    ty: *inner.clone(),
                                    span,
                                });
                                self.error_expr(span)
                            }
                        }
                    }
                    _ => {
                        self.emit(TypeError::SafeNavOnNonOptional {
                            ty: typed_target.ty,
                            span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::SafeTupleAccess(tuple_expr, index) => {
                let typed_tuple = self.infer_expr(*tuple_expr);
                if matches!(typed_tuple.ty, Type::Error) {
                    return self.error_expr(span);
                }
                let tuple_ty = typed_tuple.ty.clone();
                match &tuple_ty {
                    Type::Optional(inner) => {
                        if let Type::Tuple(types) = inner.as_ref() {
                            if index < types.len() {
                                TypedExpr {
                                    kind: TypedExprKind::SafeTupleAccess(
                                        Box::new(typed_tuple),
                                        index,
                                    ),
                                    ty: Type::Optional(Box::new(types[index].clone())),
                                    span,
                                }
                            } else {
                                self.emit(TypeError::TupleIndexOutOfBounds {
                                    index,
                                    size: types.len(),
                                    span,
                                });
                                self.error_expr(span)
                            }
                        } else {
                            self.emit(TypeError::NotATuple {
                                ty: *inner.clone(),
                                span,
                            });
                            self.error_expr(span)
                        }
                    }
                    _ => {
                        self.emit(TypeError::SafeNavOnNonOptional {
                            ty: typed_tuple.ty,
                            span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::Unwrap(e) => {
                let typed_e = self.infer_expr(*e);
                let inner_ty = typed_e.ty.clone();

                match inner_ty {
                    Type::Optional(t) => TypedExpr {
                        kind: TypedExprKind::UnwrapOptional(Box::new(typed_e)),
                        ty: *t,
                        span,
                    },
                    Type::Error => self.error_expr(span),
                    _ => {
                        self.emit(TypeError::UnwrapOnNonOptional { ty: inner_ty, span });
                        self.error_expr(span)
                    }
                }
            }
            ResolvedExprKind::StructLit(struct_id, fields) => {
                self.check_struct_literal(struct_id, fields, span)
            }
            ResolvedExprKind::Error => TypedExpr {
                kind: TypedExprKind::Error,
                ty: Type::Error,
                span,
            },
            ResolvedExprKind::RpcCall(target, call) => {
                let typed_target = self.infer_expr(*target);
                let role_id = match &typed_target.ty {
                    Type::Role(id, _) => *id,
                    Type::Error => return self.error_expr(span),
                    _ => {
                        self.emit(TypeError::RpcCallTargetNotRole {
                            ty: typed_target.ty.clone(),
                            span: typed_target.span,
                        });
                        return self.error_expr(span);
                    }
                };

                let role_funcs = match self.role_func_signatures.get(&role_id) {
                    Some(funcs) => funcs,
                    None => {
                        self.emit(TypeError::UndefinedType(typed_target.span));
                        return self.error_expr(span);
                    }
                };

                let (func_id, sig) = match role_funcs.get(&call.original_name) {
                    Some(v) => v,
                    None => {
                        self.emit(TypeError::FieldNotFound {
                            field_name: call.original_name.clone(),
                            span: call.span,
                        });
                        return self.error_expr(span);
                    }
                };
                if sig.is_sync {
                    self.emit(TypeError::RpcCallToSyncFunc {
                        func_name: call.original_name.clone(),
                        span,
                    });
                    return self.error_expr(span);
                }

                // Clone data before mutable borrow
                let func_id = *func_id;
                let params = sig.params.clone();
                let return_type = sig.return_type.clone();

                let typed_args = self.check_call_args(&params, call.args, call.span);
                let typed_call = TypedUserFuncCall {
                    name: func_id,
                    original_name: call.original_name,
                    args: typed_args,
                    return_type: return_type.clone(),
                    is_free: false,
                    span: call.span,
                };

                let return_ty = Type::Chan(Box::new(return_type));
                TypedExpr {
                    kind: TypedExprKind::RpcCall(Box::new(typed_target), typed_call),
                    ty: return_ty,
                    span,
                }
            }

            ResolvedExprKind::Return(inner) => {
                if let Some(expected_return_type) = self.current_return_type.clone() {
                    let typed_inner = self.check_expr(*inner, &expected_return_type);
                    TypedExpr {
                        kind: TypedExprKind::Return(Box::new(typed_inner)),
                        ty: Type::Never,
                        span,
                    }
                } else {
                    self.emit(TypeError::ReturnOutsideFunction(span));
                    TypedExpr {
                        kind: TypedExprKind::Error,
                        ty: Type::Error,
                        span,
                    }
                }
            }
            ResolvedExprKind::Break => {
                if self.loop_depth == 0 {
                    self.emit(TypeError::BreakOutsideLoop(span));
                }
                TypedExpr {
                    kind: TypedExprKind::Break,
                    ty: Type::Never,
                    span,
                }
            }
            ResolvedExprKind::Continue => {
                if self.loop_depth == 0 {
                    self.emit(TypeError::ContinueOutsideLoop(span));
                }
                TypedExpr {
                    kind: TypedExprKind::Continue,
                    ty: Type::Never,
                    span,
                }
            }
        }
    }

    fn check_types_match(
        &self,
        mut typed_expr: TypedExpr,
        expected: &Type,
    ) -> Result<TypedExpr, TypeError> {
        let actual = &typed_expr.ty;

        // Error types are compatible with anything — suppress cascading errors
        if matches!(expected, Type::Error) || matches!(actual, Type::Error) {
            return Ok(typed_expr);
        }

        if actual == expected {
            return Ok(typed_expr);
        }

        if let (Type::List(e), Type::EmptyList) = (expected, actual) {
            typed_expr.ty = Type::List(e.clone());
            return Ok(typed_expr);
        }
        if let (Type::Map(k, v), Type::EmptyMap) = (expected, actual) {
            typed_expr.ty = Type::Map(k.clone(), v.clone());
            return Ok(typed_expr);
        }
        if let (Type::Chan(e), Type::UnknownChannel) = (expected, actual) {
            typed_expr.ty = Type::Chan(e.clone());
            return Ok(typed_expr);
        }

        if let Type::Optional(expected_inner) = expected {
            if *actual == Type::Nil {
                typed_expr.ty = expected.clone();
                return Ok(typed_expr);
            }

            if let Type::Optional(actual_inner) = actual {
                return self
                    .check_type_compatibility(expected_inner, actual_inner, typed_expr.span)
                    .map(|_| typed_expr);
            }

            let span = typed_expr.span;
            if self
                .check_type_compatibility(expected_inner, actual, span)
                .is_ok()
            {
                return Ok(TypedExpr {
                    kind: TypedExprKind::WrapInOptional(Box::new(typed_expr)),
                    ty: expected.clone(),
                    span,
                });
            }
        }

        self.check_type_compatibility(expected, actual, typed_expr.span)?;
        Ok(typed_expr)
    }

    fn infer_binop(
        &mut self,
        op: BinOp,
        left: ResolvedExpr,
        right: ResolvedExpr,
        span: Span,
    ) -> TypedExpr {
        match op {
            BinOp::Add => {
                // Infer types of both operands
                let typed_left = self.infer_expr(left);
                let typed_right = self.infer_expr(right);

                // Both must be the same type (either Int or String)
                match (&typed_left.ty, &typed_right.ty) {
                    (Type::Int, Type::Int) => TypedExpr {
                        kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                        ty: Type::Int,
                        span,
                    },
                    (Type::String, Type::String) => TypedExpr {
                        kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                        ty: Type::String,
                        span,
                    },
                    _ if matches!(typed_left.ty, Type::Error)
                        || matches!(typed_right.ty, Type::Error) =>
                    {
                        self.error_expr(span)
                    }
                    _ => {
                        self.emit(TypeError::InvalidBinOp {
                            op: op.clone(),
                            left: typed_left.ty.clone(),
                            right: typed_right.ty.clone(),
                            span,
                        });
                        self.error_expr(span)
                    }
                }
            }
            BinOp::Subtract | BinOp::Multiply | BinOp::Divide | BinOp::Modulo => {
                let typed_left = self.check_expr(left, &Type::Int);
                let typed_right = self.check_expr(right, &Type::Int);
                TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Int,
                    span,
                }
            }
            BinOp::Equal | BinOp::NotEqual => {
                let typed_left = self.infer_expr(left);
                let typed_right = self.infer_expr(right);
                let (coerced_left, coerced_right) = match self.check_and_coerce_symmetric(
                    typed_left,
                    typed_right,
                    op.clone(),
                    span,
                ) {
                    Ok(pair) => pair,
                    Err(e) => {
                        self.emit(e);
                        return self.error_expr(span);
                    }
                };
                TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(coerced_left), Box::new(coerced_right)),
                    ty: Type::Bool,
                    span,
                }
            }
            BinOp::Less | BinOp::LessEqual | BinOp::Greater | BinOp::GreaterEqual => {
                let typed_left = self.check_expr(left, &Type::Int);
                let typed_right = self.check_expr(right, &Type::Int);
                TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Bool,
                    span,
                }
            }
            BinOp::And | BinOp::Or => {
                let typed_left = self.check_expr(left, &Type::Bool);
                let typed_right = self.check_expr(right, &Type::Bool);
                TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Bool,
                    span,
                }
            }
            BinOp::Coalesce => {
                let typed_left = self.infer_expr(left);
                if let Type::Optional(inner_ty) = typed_left.ty.clone() {
                    let typed_right = self.check_expr(right, &inner_ty);
                    TypedExpr {
                        kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                        ty: *inner_ty,
                        span,
                    }
                } else if matches!(typed_left.ty, Type::Error) {
                    let _ = self.infer_expr(right);
                    self.error_expr(span)
                } else {
                    self.emit(TypeError::InvalidBinOp {
                        op: BinOp::Coalesce,
                        left: typed_left.ty,
                        right: Type::Nil,
                        span,
                    });
                    let _ = self.infer_expr(right);
                    self.error_expr(span)
                }
            }
        }
    }

    fn check_and_coerce_symmetric(
        &self,
        typed_left: TypedExpr,
        typed_right: TypedExpr,
        op: BinOp,
        span: Span,
    ) -> Result<(TypedExpr, TypedExpr), TypeError> {
        let left_ty = typed_left.ty.clone();
        let right_ty = typed_right.ty.clone();

        match self.check_types_match(typed_right.clone(), &left_ty) {
            Ok(coerced_right) => Ok((typed_left, coerced_right)),
            Err(_) => match self.check_types_match(typed_left.clone(), &right_ty) {
                Ok(coerced_left) => Ok((coerced_left, typed_right)),
                Err(_) => Err(TypeError::InvalidBinOp {
                    op,
                    left: left_ty,
                    right: right_ty,
                    span,
                }),
            },
        }
    }

    fn check_call_args(
        &mut self,
        expected_params: &Vec<Type>,
        args: Vec<ResolvedExpr>,
        call_span: Span,
    ) -> Vec<TypedExpr> {
        if args.len() != expected_params.len() {
            self.emit(TypeError::WrongNumberOfArgs {
                expected: expected_params.len(),
                got: args.len(),
                span: call_span,
            });
            // Still type-check as many args as we can
            let mut typed_args = Vec::new();
            for (arg, expected_ty) in args.into_iter().zip(expected_params.iter()) {
                typed_args.push(self.check_expr(arg, expected_ty));
            }
            return typed_args;
        }

        let mut typed_args = Vec::new();
        for (arg, expected_ty) in args.into_iter().zip(expected_params.iter()) {
            let typed_arg = self.check_expr(arg, expected_ty);
            typed_args.push(typed_arg);
        }
        typed_args
    }

    fn check_builtin_call(
        &mut self,
        builtin: BuiltinFn,
        args: Vec<ResolvedExpr>,
        span: Span,
    ) -> TypedExpr {
        // Special case: role_to_string accepts any role type
        if builtin == BuiltinFn::RoleToString {
            if args.len() != 1 {
                self.emit(TypeError::WrongNumberOfArgs {
                    expected: 1,
                    got: args.len(),
                    span,
                });
                return self.error_expr(span);
            }
            let typed_arg =
                self.infer_expr(args.into_iter().next().expect("length already checked"));
            if !matches!(typed_arg.ty, Type::Role(_, _) | Type::Error) {
                self.emit(TypeError::Mismatch {
                    expected: Type::Role(NameId(0), "role".to_string()),
                    found: typed_arg.ty.clone(),
                    span,
                });
            }
            return TypedExpr {
                kind: TypedExprKind::FuncCall(TypedFuncCall::Builtin(
                    builtin,
                    vec![typed_arg],
                    Type::String,
                )),
                ty: Type::String,
                span,
            };
        }

        let sig = match self.builtin_signatures.get(&builtin) {
            Some(sig) => sig,
            None => {
                self.emit(TypeError::InternalError {
                    message: format!("Missing signature for builtin function '{:?}'", builtin),
                    span,
                });
                return self.error_expr(span);
            }
        };

        // Clone data before mutable borrow
        let params = sig.params.clone();
        let return_type = sig.return_type.clone();

        let typed_args = self.check_call_args(&params, args, span);

        TypedExpr {
            kind: TypedExprKind::FuncCall(TypedFuncCall::Builtin(
                builtin,
                typed_args,
                return_type.clone(),
            )),
            ty: return_type,
            span,
        }
    }

    fn check_user_func_call(
        &mut self,
        call: ResolvedUserFuncCall,
        sig: &FunctionSignature,
        is_free: bool,
    ) -> TypedUserFuncCall {
        let typed_args = self.check_call_args(&sig.params, call.args, call.span);

        TypedUserFuncCall {
            name: call.name,
            original_name: call.original_name,
            args: typed_args,
            return_type: sig.return_type.clone(),
            is_free,
            span: call.span,
        }
    }

    fn infer_index(&mut self, target: ResolvedExpr, index: ResolvedExpr, span: Span) -> TypedExpr {
        let typed_target = self.infer_expr(target);
        let typed_index = self.infer_expr(index);

        let target_ty = typed_target.ty.clone();
        let index_ty = typed_index.ty.clone();

        match target_ty {
            Type::List(elem_ty) => {
                if let Err(e) = self.check_type_compatibility(&Type::Int, &index_ty, span) {
                    self.emit(e);
                }
                TypedExpr {
                    kind: TypedExprKind::Index(Box::new(typed_target), Box::new(typed_index)),
                    ty: *elem_ty,
                    span,
                }
            }
            Type::Map(key_ty, val_ty) => {
                if let Err(e) = self.check_type_compatibility(&key_ty, &index_ty, span) {
                    self.emit(e);
                }
                TypedExpr {
                    kind: TypedExprKind::Index(Box::new(typed_target), Box::new(typed_index)),
                    ty: *val_ty,
                    span,
                }
            }
            Type::String => {
                if let Err(e) = self.check_type_compatibility(&Type::Int, &index_ty, span) {
                    self.emit(e);
                }
                TypedExpr {
                    kind: TypedExprKind::Index(Box::new(typed_target), Box::new(typed_index)),
                    ty: Type::String,
                    span,
                }
            }
            Type::Error => self.error_expr(span),
            _ => {
                self.emit(TypeError::NotIndexable {
                    ty: typed_target.ty,
                    span,
                });
                self.error_expr(span)
            }
        }
    }

    fn infer_list_literal(&mut self, mut items: Vec<ResolvedExpr>, span: Span) -> TypedExpr {
        if items.is_empty() {
            return TypedExpr {
                kind: TypedExprKind::ListLit(vec![]),
                ty: Type::EmptyList,
                span,
            };
        }

        let mut typed_items = Vec::new();
        let first_typed = self.infer_expr(items.remove(0));
        let first_ty = first_typed.ty.clone();
        typed_items.push(first_typed);

        for item in items {
            let typed_item = self.check_expr(item, &first_ty);
            typed_items.push(typed_item);
        }

        TypedExpr {
            kind: TypedExprKind::ListLit(typed_items),
            ty: Type::List(Box::new(first_ty)),
            span,
        }
    }

    fn infer_map_literal(
        &mut self,
        mut pairs: Vec<(ResolvedExpr, ResolvedExpr)>,
        span: Span,
    ) -> TypedExpr {
        if pairs.is_empty() {
            return TypedExpr {
                kind: TypedExprKind::MapLit(vec![]),
                ty: Type::EmptyMap,
                span,
            };
        }

        let (first_key, first_val) = pairs.remove(0);
        let typed_first_key = self.infer_expr(first_key);
        let typed_first_val = self.infer_expr(first_val);
        let key_ty = typed_first_key.ty.clone();
        let val_ty = typed_first_val.ty.clone();

        let mut typed_pairs = vec![(typed_first_key, typed_first_val)];

        for (key, val) in pairs {
            let typed_key = self.check_expr(key, &key_ty);
            let typed_val = self.check_expr(val, &val_ty);
            typed_pairs.push((typed_key, typed_val));
        }

        TypedExpr {
            kind: TypedExprKind::MapLit(typed_pairs),
            ty: Type::Map(Box::new(key_ty), Box::new(val_ty)),
            span,
        }
    }

    fn get_resolved_struct_fields(
        &self,
        struct_id: NameId,
        span: Span,
        context: &str,
    ) -> Result<(NameId, Vec<crate::analysis::resolver::ResolvedFieldDef>), TypeError> {
        let (resolved_id, fields) = self.resolve_struct_definition(struct_id, span, context)?;
        Ok((resolved_id, fields.clone()))
    }

    fn get_resolved_enum_variant(
        &self,
        enum_id: NameId,
        variant_name: &str,
        span: Span,
    ) -> Result<
        (
            NameId,
            String,
            crate::analysis::resolver::ResolvedEnumVariant,
        ),
        TypeError,
    > {
        let (resolved_id, variants) = self.resolve_enum_definition(enum_id, span, variant_name)?;

        let variant = variants.iter().find(|v| v.name == variant_name).cloned();

        let enum_name = self
            .enum_names
            .get(&resolved_id)
            .cloned()
            .unwrap_or_else(|| format!("enum_{}", resolved_id.0));

        match variant {
            Some(v) => Ok((resolved_id, enum_name, v)),
            None => Err(TypeError::VariantNotFound {
                enum_name,
                variant: variant_name.to_string(),
                span,
            }),
        }
    }

    fn check_struct_literal(
        &mut self,
        struct_id: NameId,
        fields: Vec<(String, ResolvedExpr)>,
        span: Span,
    ) -> TypedExpr {
        let (resolved_id, field_defs) = match self.get_resolved_struct_fields(struct_id, span, "") {
            Ok(v) => v,
            Err(e) => {
                self.emit(e);
                // Still type-check field expressions for further errors
                for (_, field_expr) in fields {
                    let _ = self.infer_expr(field_expr);
                }
                return self.error_expr(span);
            }
        };

        let mut typed_fields = Vec::new();
        for (field_name, field_expr) in fields {
            let field_def = field_defs.iter().find(|f| f.name == field_name);

            match field_def {
                Some(fd) => match self.resolve_type(&fd.type_def) {
                    Ok(expected_ty) => {
                        let typed_field_expr = self.check_expr(field_expr, &expected_ty);
                        typed_fields.push((field_name, typed_field_expr));
                    }
                    Err(e) => {
                        self.emit(e);
                        let _ = self.infer_expr(field_expr);
                    }
                },
                None => {
                    self.emit(TypeError::UndefinedStructField {
                        field_name: field_name.clone(),
                        span,
                    });
                    let _ = self.infer_expr(field_expr);
                }
            }
        }

        for field_def in field_defs {
            if !typed_fields.iter().any(|(name, _)| name == &field_def.name) {
                self.emit(TypeError::MissingStructField {
                    field_name: field_def.name.clone(),
                    span,
                });
            }
        }

        let struct_name = self
            .struct_names
            .get(&resolved_id)
            .cloned()
            .unwrap_or_else(|| format!("struct_{}", resolved_id.0));

        let ty = Type::Struct(resolved_id, struct_name);
        TypedExpr {
            kind: TypedExprKind::StructLit(resolved_id, typed_fields),
            ty,
            span,
        }
    }

    fn get_field_type(
        &self,
        struct_id: NameId,
        field_name: &str,
        span: Span,
    ) -> Result<Type, TypeError> {
        let (_, field_defs) = self.resolve_struct_definition(struct_id, span, field_name)?;

        let field_def = field_defs
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| TypeError::FieldNotFound {
                field_name: field_name.to_string(),
                span,
            })?;

        self.resolve_type(&field_def.type_def)
    }

    fn resolve_type(&self, type_def: &ResolvedTypeDef) -> Result<Type, TypeError> {
        match type_def {
            ResolvedTypeDef::Named(name_id) => {
                if let Some(def) = self.type_defs.get(name_id) {
                    match def {
                        TypeDefinition::Builtin(ty) => Ok(ty.clone()),
                        TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Struct(_)) => {
                            let name = self
                                .struct_names
                                .get(name_id)
                                .cloned()
                                .unwrap_or_else(|| format!("struct_{}", name_id.0));
                            Ok(Type::Struct(*name_id, name))
                        }
                        TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Enum(_)) => {
                            let name = self
                                .enum_names
                                .get(name_id)
                                .cloned()
                                .unwrap_or_else(|| format!("enum_{}", name_id.0));
                            Ok(Type::Enum(*name_id, name))
                        }
                        TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Alias(
                            aliased_type,
                        )) => self.resolve_type(aliased_type),
                    }
                } else if let Some(role_name) = self.role_defs.get(name_id) {
                    Ok(Type::Role(*name_id, role_name.clone()))
                } else {
                    let name = self
                        .struct_names
                        .get(name_id)
                        .cloned()
                        .unwrap_or_else(|| format!("unknown_{}", name_id.0));
                    Ok(Type::Struct(*name_id, name))
                }
            }
            ResolvedTypeDef::Map(key, val) => {
                let key_ty = self.resolve_type(key)?;
                let val_ty = self.resolve_type(val)?;
                Ok(Type::Map(Box::new(key_ty), Box::new(val_ty)))
            }
            ResolvedTypeDef::List(elem) => {
                let elem_ty = self.resolve_type(elem)?;
                Ok(Type::List(Box::new(elem_ty)))
            }
            ResolvedTypeDef::Tuple(types) => {
                let resolved_types = types
                    .iter()
                    .map(|t| self.resolve_type(t))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Type::Tuple(resolved_types))
            }
            ResolvedTypeDef::Optional(t) => {
                let base_type = self.resolve_type(t)?;
                if matches!(base_type, Type::Optional(_)) {
                    Ok(base_type)
                } else {
                    Ok(Type::Optional(Box::new(base_type)))
                }
            }
            ResolvedTypeDef::Chan(t) => {
                let base_type = self.resolve_type(t)?;
                Ok(Type::Chan(Box::new(base_type)))
            }
            ResolvedTypeDef::Error => Ok(Type::Error),
        }
    }

    fn check_type_compatibility(
        &self,
        expected: &Type,
        actual: &Type,
        span: Span,
    ) -> Result<(), TypeError> {
        // Error types are compatible with anything — suppress cascading errors
        if matches!(expected, Type::Error) || matches!(actual, Type::Error) {
            return Ok(());
        }

        if expected == actual {
            return Ok(());
        }

        // Never is the bottom type and coerces to any type
        if *actual == Type::Never {
            return Ok(());
        }

        if let (Type::List(_), Type::EmptyList) = (expected, actual) {
            return Ok(());
        }
        if let (Type::Map(_, _), Type::EmptyMap) = (expected, actual) {
            return Ok(());
        }
        if let (Type::Chan(_), Type::UnknownChannel) = (expected, actual) {
            return Ok(());
        }

        if let Type::Optional(expected_inner) = expected {
            if *actual == Type::Nil {
                return Ok(());
            }
            if let Type::Optional(actual_inner) = actual {
                return self.check_type_compatibility(expected_inner, actual_inner, span);
            }

            if self
                .check_type_compatibility(expected_inner, actual, span)
                .is_ok()
            {
                return Ok(());
            }
        }
        if let (Type::Chan(expected_inner), Type::Chan(actual_inner)) = (expected, actual) {
            return self.check_type_compatibility(expected_inner, actual_inner, span);
        }

        Err(TypeError::Mismatch {
            expected: expected.clone(),
            found: actual.clone(),
            span,
        })
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn add_var(&mut self, name_id: NameId, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name_id, ty);
        }
    }

    fn get_var_type(&self, name_id: NameId, span: Span) -> Result<Type, TypeError> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(&name_id) {
                return Ok(ty.clone());
            }
        }
        Err(TypeError::UndefinedType(span))
    }
}

#[cfg(test)]
mod test;
