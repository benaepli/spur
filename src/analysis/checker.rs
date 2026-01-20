use crate::analysis::resolver::{
    BuiltinFn, NameId, PrepopulatedTypes, ResolvedAssignment, ResolvedCondStmts, ResolvedExpr,
    ResolvedExprKind, ResolvedForInLoop, ResolvedForLoop, ResolvedFuncCall, ResolvedFuncDef,
    ResolvedPattern, ResolvedPatternKind, ResolvedProgram, ResolvedRoleDef, ResolvedStatement,
    ResolvedStatementKind, ResolvedTopLevelDef, ResolvedTypeDef, ResolvedTypeDefStmtKind,
    ResolvedUserFuncCall, ResolvedVarInit,
};
use crate::analysis::types::{
    Type, TypedAssignment, TypedCondStmts, TypedExpr, TypedExprKind, TypedForInLoop, TypedForLoop,
    TypedForLoopInit, TypedFuncCall, TypedFuncDef, TypedFuncParam, TypedIfBranch, TypedMatchArm,
    TypedPattern, TypedPatternKind, TypedProgram, TypedRoleDef, TypedStatement, TypedStatementKind,
    TypedTopLevelDef, TypedUserFuncCall, TypedVarInit,
};
use crate::parser::{BinOp, Span};
use std::collections::HashMap;
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
    struct_names: HashMap<NameId, String>,
    enum_names: HashMap<NameId, String>,
    current_return_type: Option<Type>,
    current_func_is_sync: bool,
    loop_depth: usize,
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
            struct_names: HashMap::new(),
            enum_names: HashMap::new(),
            current_return_type: None,
            current_func_is_sync: false,
            loop_depth: 0,
        }
    }

    pub fn check_program(&mut self, program: ResolvedProgram) -> Result<TypedProgram, TypeError> {
        self.collect_definitions(&program)?;

        let next_name_id = program.next_name_id;
        let id_to_name = program.id_to_name;
        let mut typed_top_levels = Vec::new();
        for top_level_def in program.top_level_defs {
            match top_level_def {
                ResolvedTopLevelDef::Role(role) => {
                    let typed_role = self.check_role_def(role)?;
                    typed_top_levels.push(TypedTopLevelDef::Role(typed_role));
                }
                ResolvedTopLevelDef::Type(_) => {}
            }
        }

        Ok(TypedProgram {
            top_level_defs: typed_top_levels,
            next_name_id,
            id_to_name,
        })
    }

    fn collect_definitions(&mut self, program: &ResolvedProgram) -> Result<(), TypeError> {
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
            }
        }

        for top_level_def in &program.top_level_defs {
            if let ResolvedTopLevelDef::Role(role) = top_level_def {
                for func in &role.func_defs {
                    let sig = self.build_function_signature(func)?;
                    let role_funcs = self
                        .role_func_signatures
                        .get_mut(&role.name)
                        .unwrap_or_else(|| {
                            panic!(
                                "Role '{}' not found in signature map. \
                                 This should have been initialized in the first pass.",
                                role.original_name
                            )
                        });

                    self.func_signatures.insert(func.name, sig.clone());
                    role_funcs.insert(func.original_name.clone(), (func.name, sig));
                }
            }
        }

        Ok(())
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

    fn check_role_def(&mut self, role: ResolvedRoleDef) -> Result<TypedRoleDef, TypeError> {
        self.enter_scope();
        self.current_func_is_sync = true;

        let mut typed_var_inits = Vec::new();
        for var_init in role.var_inits {
            typed_var_inits.push(self.check_var_init(var_init)?);
        }

        self.current_func_is_sync = false;

        let mut typed_func_defs = Vec::new();
        for func in role.func_defs {
            typed_func_defs.push(self.check_func_def(func)?);
        }

        self.exit_scope();
        Ok(TypedRoleDef {
            name: role.name,
            original_name: role.original_name,
            var_inits: typed_var_inits,
            func_defs: typed_func_defs,
            span: role.span,
        })
    }

    fn check_func_def(&mut self, func: ResolvedFuncDef) -> Result<TypedFuncDef, TypeError> {
        let sig = self
            .func_signatures
            .get(&func.name)
            .unwrap_or_else(|| {
                panic!(
                    "Function signature not found for '{}'. \
                     This should have been collected in the signature collection pass.",
                    func.original_name
                )
            })
            .clone();

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

        let mut typed_body = Vec::new();
        let mut has_return = false;
        for stmt in func.body {
            let (typed_stmt, stmt_returns) = self.check_statement(stmt)?;
            typed_body.push(typed_stmt);
            if stmt_returns {
                has_return = true;
            }
        }

        if sig.return_type != Type::Tuple(vec![]) && !has_return {
            return Err(TypeError::MissingReturn(func.span));
        }

        self.current_return_type = None;
        self.current_func_is_sync = false;
        self.exit_scope();
        Ok(TypedFuncDef {
            name: func.name,
            original_name: func.original_name,
            is_sync: sig.is_sync,
            params: typed_params,
            return_type: sig.return_type,
            body: typed_body,
            span: func.span,
        })
    }

    fn check_var_init(&mut self, var_init: ResolvedVarInit) -> Result<TypedVarInit, TypeError> {
        let (typed_value, type_def) = if let Some(def) = var_init.type_def {
            let expected_ty = self.resolve_type(&def)?;
            let typed_val = self.check_expr(var_init.value, &expected_ty)?;
            (typed_val, expected_ty)
        } else {
            let typed_val = self.infer_expr(var_init.value)?;
            let inferred_ty = typed_val.ty.clone();
            (typed_val, inferred_ty)
        };

        self.add_var(var_init.name, type_def.clone());
        Ok(TypedVarInit {
            name: var_init.name,
            original_name: var_init.original_name,
            type_def,
            value: typed_value,
            span: var_init.span,
        })
    }

    fn check_statement(
        &mut self,
        stmt: ResolvedStatement,
    ) -> Result<(TypedStatement, bool), TypeError> {
        let span = stmt.span;
        match stmt.kind {
            ResolvedStatementKind::Conditional(cond) => {
                let (typed_cond, returns) = self.check_conditional(cond)?;
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::Conditional(typed_cond),
                    span,
                };
                Ok((typed_stmt, returns))
            }
            ResolvedStatementKind::VarInit(var_init) => {
                let typed_var_init = self.check_var_init(var_init)?;
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::VarInit(typed_var_init),
                    span,
                };
                Ok((typed_stmt, false))
            }
            ResolvedStatementKind::Assignment(assign) => {
                let typed_assign = self.check_assignment(assign)?;
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::Assignment(typed_assign),
                    span,
                };
                Ok((typed_stmt, false))
            }
            ResolvedStatementKind::Expr(expr) => {
                let typed_expr = self.infer_expr(expr)?;
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::Expr(typed_expr),
                    span,
                };
                Ok((typed_stmt, false))
            }
            ResolvedStatementKind::Return(expr) => {
                if let Some(expected_return_type) = self.current_return_type.clone() {
                    let typed_expr = self.check_expr(expr, &expected_return_type)?;
                    let typed_stmt = TypedStatement {
                        kind: TypedStatementKind::Return(typed_expr),
                        span,
                    };
                    Ok((typed_stmt, true))
                } else {
                    Err(TypeError::ReturnOutsideFunction(span))
                }
            }
            ResolvedStatementKind::ForLoop(for_loop) => {
                let typed_loop = self.check_for_loop(for_loop)?;
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::ForLoop(typed_loop),
                    span,
                };
                Ok((typed_stmt, false))
            }
            ResolvedStatementKind::ForInLoop(for_in_loop) => {
                let typed_loop = self.check_for_in_loop(for_in_loop)?;
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::ForInLoop(typed_loop),
                    span,
                };
                Ok((typed_stmt, false))
            }
            ResolvedStatementKind::Break => {
                if self.loop_depth == 0 {
                    return Err(TypeError::BreakOutsideLoop(span));
                }
                let typed_stmt = TypedStatement {
                    kind: TypedStatementKind::Break,
                    span,
                };
                Ok((typed_stmt, false))
            }
        }
    }

    fn check_block(
        &mut self,
        block: Vec<ResolvedStatement>,
    ) -> Result<(Vec<TypedStatement>, bool), TypeError> {
        self.enter_scope();
        let mut typed_stmts = Vec::new();
        let mut has_return = false;
        for stmt in block {
            let (typed_stmt, stmt_returns) = self.check_statement(stmt)?;
            typed_stmts.push(typed_stmt);
            if stmt_returns {
                has_return = true;
            }
        }
        self.exit_scope();
        Ok((typed_stmts, has_return))
    }

    fn check_conditional(
        &mut self,
        cond: ResolvedCondStmts,
    ) -> Result<(TypedCondStmts, bool), TypeError> {
        let typed_if_cond = self.check_expr(cond.if_branch.condition, &Type::Bool)?;
        let (typed_if_body, if_returns) = self.check_block(cond.if_branch.body)?;
        let typed_if_branch = TypedIfBranch {
            condition: typed_if_cond,
            body: typed_if_body,
            span: cond.if_branch.span,
        };

        let mut all_branches_return = if_returns;
        let mut typed_elseif_branches = Vec::new();
        for elseif in cond.elseif_branches {
            let typed_elseif_cond = self.check_expr(elseif.condition, &Type::Bool)?;
            let (typed_elseif_body, elseif_returns) = self.check_block(elseif.body)?;

            typed_elseif_branches.push(TypedIfBranch {
                condition: typed_elseif_cond,
                body: typed_elseif_body,
                span: elseif.span,
            });
            all_branches_return = all_branches_return && elseif_returns;
        }

        let typed_else_branch;
        if let Some(else_body) = cond.else_branch {
            let (typed_body, else_returns) = self.check_block(else_body)?;
            typed_else_branch = Some(typed_body);
            all_branches_return = all_branches_return && else_returns;
        } else {
            typed_else_branch = None;
            all_branches_return = false;
        }

        Ok((
            TypedCondStmts {
                if_branch: typed_if_branch,
                elseif_branches: typed_elseif_branches,
                else_branch: typed_else_branch,
                span: cond.span,
            },
            all_branches_return,
        ))
    }

    fn check_for_loop(&mut self, for_loop: ResolvedForLoop) -> Result<TypedForLoop, TypeError> {
        self.enter_scope();
        self.loop_depth += 1;

        let typed_init = if let Some(init) = for_loop.init {
            match init {
                crate::analysis::resolver::ResolvedForLoopInit::VarInit(vi) => {
                    Some(TypedForLoopInit::VarInit(self.check_var_init(vi)?))
                }
                crate::analysis::resolver::ResolvedForLoopInit::Assignment(a) => {
                    Some(TypedForLoopInit::Assignment(self.check_assignment(a)?))
                }
            }
        } else {
            None
        };

        let typed_condition = if let Some(condition) = for_loop.condition {
            Some(self.check_expr(condition, &Type::Bool)?)
        } else {
            None
        };

        let typed_increment = if let Some(increment) = for_loop.increment {
            Some(self.check_assignment(increment)?)
        } else {
            None
        };

        let mut typed_body = Vec::new();
        for stmt in for_loop.body {
            let (typed_stmt, _) = self.check_statement(stmt)?;
            typed_body.push(typed_stmt);
        }

        self.loop_depth -= 1;
        self.exit_scope();
        Ok(TypedForLoop {
            init: typed_init,
            condition: typed_condition,
            increment: typed_increment,
            body: typed_body,
            span: for_loop.span,
        })
    }

    fn check_for_in_loop(
        &mut self,
        for_in_loop: ResolvedForInLoop,
    ) -> Result<TypedForInLoop, TypeError> {
        let typed_iterable = self.infer_expr(for_in_loop.iterable)?;

        let element_type = match &typed_iterable.ty {
            Type::List(elem_ty) => (**elem_ty).clone(),
            Type::Map(key_ty, val_ty) => Type::Tuple(vec![(**key_ty).clone(), (**val_ty).clone()]),
            _ => {
                return Err(TypeError::NotIterable {
                    ty: typed_iterable.ty,
                    span: for_in_loop.span,
                });
            }
        };

        self.enter_scope();
        self.loop_depth += 1;

        let typed_pattern = self.check_pattern(for_in_loop.pattern, &element_type)?;

        let mut typed_body = Vec::new();
        for stmt in for_in_loop.body {
            let (typed_stmt, _) = self.check_statement(stmt)?;
            typed_body.push(typed_stmt);
        }

        self.loop_depth -= 1;
        self.exit_scope();
        Ok(TypedForInLoop {
            pattern: typed_pattern,
            iterable: typed_iterable,
            body: typed_body,
            span: for_in_loop.span,
        })
    }

    fn check_pattern(
        &mut self,
        pattern: ResolvedPattern,
        expected_type: &Type,
    ) -> Result<TypedPattern, TypeError> {
        match pattern.kind {
            ResolvedPatternKind::Var(name_id, name) => {
                self.add_var(name_id, expected_type.clone());
                Ok(TypedPattern {
                    kind: TypedPatternKind::Var(name_id, name),
                    ty: expected_type.clone(),
                    span: pattern.span,
                })
            }
            ResolvedPatternKind::Wildcard => Ok(TypedPattern {
                kind: TypedPatternKind::Wildcard,
                ty: expected_type.clone(),
                span: pattern.span,
            }),
            ResolvedPatternKind::Tuple(patterns) => {
                if let Type::Tuple(types) = expected_type {
                    if patterns.len() != types.len() {
                        return Err(TypeError::PatternMismatch {
                            pattern_ty: format!("tuple of {} elements", patterns.len()),
                            iterable_ty: expected_type.clone(),
                            span: pattern.span,
                        });
                    }
                    let mut typed_patterns = Vec::new();
                    for (pat, ty) in patterns.into_iter().zip(types.iter()) {
                        typed_patterns.push(self.check_pattern(pat, ty)?);
                    }
                    Ok(TypedPattern {
                        kind: TypedPatternKind::Tuple(typed_patterns),
                        ty: expected_type.clone(),
                        span: pattern.span,
                    })
                } else {
                    Err(TypeError::PatternMismatch {
                        pattern_ty: "tuple".to_string(),
                        iterable_ty: expected_type.clone(),
                        span: pattern.span,
                    })
                }
            }
            ResolvedPatternKind::Variant(enum_id, variant_name, payload_pat) => {
                if let Type::Enum(expected_enum_id, _) = expected_type {
                    if enum_id != *expected_enum_id {
                        return Err(TypeError::Mismatch {
                            expected: expected_type.clone(),
                            found: Type::Enum(enum_id, "".to_string()),
                            span: pattern.span,
                        });
                    }

                    let (resolved_id, _enum_name, variant_def) =
                        self.get_resolved_enum_variant(enum_id, &variant_name, pattern.span)?;

                    debug_assert_eq!(resolved_id, enum_id, "enum ID mismatch after lookup");

                    let typed_payload = match (&variant_def.payload, payload_pat) {
                        (Some(payload_ty_def), Some(pat)) => {
                            let payload_ty = self.resolve_type(payload_ty_def)?;
                            Some(Box::new(self.check_pattern(*pat, &payload_ty)?))
                        }
                        (None, None) => None,
                        (Some(_), None) => {
                            return Err(TypeError::VariantExpectsPayload {
                                variant: variant_name,
                                span: pattern.span,
                            });
                        }
                        (None, Some(_)) => {
                            return Err(TypeError::VariantExpectsNoPayload {
                                variant: variant_name,
                                span: pattern.span,
                            });
                        }
                    };

                    Ok(TypedPattern {
                        kind: TypedPatternKind::Variant(enum_id, variant_name, typed_payload),
                        ty: expected_type.clone(),
                        span: pattern.span,
                    })
                } else {
                    Err(TypeError::PatternMismatch {
                        pattern_ty: "enum variant".to_string(),
                        iterable_ty: expected_type.clone(),
                        span: pattern.span,
                    })
                }
            }
        }
    }

    fn check_assignment(
        &mut self,
        assign: ResolvedAssignment,
    ) -> Result<TypedAssignment, TypeError> {
        if !self.is_valid_lvalue(&assign.target.kind) {
            return Err(TypeError::InvalidAssignmentTarget(assign.span));
        }

        let typed_target = self.infer_expr(assign.target)?;
        let typed_value = self.check_expr(assign.value, &typed_target.ty)?;

        Ok(TypedAssignment {
            target: typed_target,
            value: typed_value,
            span: assign.span,
        })
    }

    fn is_valid_lvalue(&self, expr_kind: &ResolvedExprKind) -> bool {
        match expr_kind {
            ResolvedExprKind::Var(_, _) => true,
            _ => false,
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
    ) -> Result<TypedExpr, TypeError> {
        let (resolved_id, enum_name, variant_def) =
            self.get_resolved_enum_variant(enum_id, &variant_name, span)?;

        let typed_payload = match (&variant_def.payload, payload) {
            (Some(payload_ty_def), Some(expr)) => {
                let payload_ty = self.resolve_type(payload_ty_def)?;
                Some(Box::new(self.check_expr(*expr, &payload_ty)?))
            }
            (None, None) => None,
            (Some(expected), None) => {
                let expected_ty = self.resolve_type(expected)?;
                return Err(TypeError::VariantPayloadMismatch {
                    variant: variant_name,
                    expected: expected_ty,
                    found: Type::Tuple(vec![]), // Assumed unit/missing
                    span,
                });
            }
            (None, Some(expr)) => {
                let _ = self.infer_expr(*expr)?;
                return Err(TypeError::VariantExpectsNoPayload {
                    variant: variant_name,
                    span,
                });
            }
        };

        Ok(TypedExpr {
            kind: TypedExprKind::VariantLit(resolved_id, variant_name, typed_payload),
            ty: Type::Enum(resolved_id, enum_name),
            span,
        })
    }

    fn check_match(
        &mut self,
        scrutinee: ResolvedExpr,
        arms: Vec<crate::analysis::resolver::ResolvedMatchArm>,
        span: Span,
    ) -> Result<TypedExpr, TypeError> {
        let typed_scrutinee = self.infer_expr(scrutinee)?;
        let scrutinee_ty = typed_scrutinee.ty.clone();

        if !matches!(scrutinee_ty, Type::Enum(_, _)) {
            return Err(TypeError::MatchScrutineeNotEnum {
                ty: scrutinee_ty,
                span: typed_scrutinee.span,
            });
        }

        let mut typed_arms = Vec::new();
        let mut match_ty: Option<Type> = None;

        for arm in arms {
            self.enter_scope();
            let typed_pattern = self.check_pattern(arm.pattern, &scrutinee_ty)?;

            let mut typed_body = Vec::new();
            let mut arm_diverges = false;

            for stmt in arm.body {
                let (typed_stmt, stmt_returns) = self.check_statement(stmt)?;
                typed_body.push(typed_stmt);
                if stmt_returns {
                    arm_diverges = true;
                }
            }

            // Determine type of the arm if it doesn't diverge
            let arm_ty = if arm_diverges {
                None
            } else if let Some(last_stmt) = typed_body.last() {
                match &last_stmt.kind {
                    TypedStatementKind::Expr(e) => Some(e.ty.clone()),
                    _ => Some(Type::Tuple(vec![])),
                }
            } else {
                Some(Type::Tuple(vec![]))
            };

            self.exit_scope();

            if let Some(ty) = arm_ty {
                if let Some(prev_ty) = &match_ty {
                    if *prev_ty != ty {
                        return Err(TypeError::MatchArmTypeMismatch {
                            expected: prev_ty.clone(),
                            found: ty,
                            span: arm.span,
                        });
                    }
                } else {
                    match_ty = Some(ty);
                }
            }

            typed_arms.push(TypedMatchArm {
                pattern: typed_pattern,
                body: typed_body,
                span: arm.span,
            });
        }

        // If all arms diverge, the match type is Never (the bottom type)
        let final_ty = match_ty.unwrap_or(Type::Never);

        Ok(TypedExpr {
            kind: TypedExprKind::Match(Box::new(typed_scrutinee), typed_arms),
            ty: final_ty,
            span,
        })
    }

    fn check_expr(&mut self, expr: ResolvedExpr, expected: &Type) -> Result<TypedExpr, TypeError> {
        let span = expr.span;
        match (&expr.kind, expected) {
            (ResolvedExprKind::NilLit, Type::Optional(_)) => Ok(TypedExpr {
                kind: TypedExprKind::NilLit,
                ty: expected.clone(),
                span,
            }),

            (ResolvedExprKind::ListLit(items), Type::List(elem_ty)) => {
                let mut typed_items = Vec::new();
                for item in items.clone() {
                    typed_items.push(self.check_expr(item, elem_ty)?);
                }
                Ok(TypedExpr {
                    kind: TypedExprKind::ListLit(typed_items),
                    ty: expected.clone(),
                    span,
                })
            }

            (ResolvedExprKind::MapLit(pairs), Type::Map(key_ty, val_ty)) => {
                let mut typed_pairs = Vec::new();
                for (key, val) in pairs.clone() {
                    let typed_key = self.check_expr(key, key_ty)?;
                    let typed_val = self.check_expr(val, val_ty)?;
                    typed_pairs.push((typed_key, typed_val));
                }
                Ok(TypedExpr {
                    kind: TypedExprKind::MapLit(typed_pairs),
                    ty: expected.clone(),
                    span,
                })
            }

            (ResolvedExprKind::StructLit(struct_id, fields), Type::Struct(expected_id, _))
                if *struct_id == *expected_id =>
            {
                self.check_struct_literal(*struct_id, fields.clone(), span)
            }

            (ResolvedExprKind::MakeChannel(cap), Type::Chan(_)) => {
                let typed_cap = self.check_expr(*cap.clone(), &Type::Int)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::MakeChannel(Box::new(typed_cap)),
                    ty: expected.clone(),
                    span,
                })
            }

            _ => {
                let inferred = self.infer_expr(expr)?;
                self.check_types_match(inferred, expected)
            }
        }
    }

    fn infer_expr(&mut self, expr: ResolvedExpr) -> Result<TypedExpr, TypeError> {
        let span = expr.span;
        match expr.kind {
            ResolvedExprKind::Var(name_id, name) => {
                let ty = self.get_var_type(name_id, span)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Var(name_id, name),
                    ty,
                    span,
                })
            }
            ResolvedExprKind::IntLit(i) => Ok(TypedExpr {
                kind: TypedExprKind::IntLit(i),
                ty: Type::Int,
                span,
            }),
            ResolvedExprKind::StringLit(s) => Ok(TypedExpr {
                kind: TypedExprKind::StringLit(s),
                ty: Type::String,
                span,
            }),
            ResolvedExprKind::BoolLit(b) => Ok(TypedExpr {
                kind: TypedExprKind::BoolLit(b),
                ty: Type::Bool,
                span,
            }),
            ResolvedExprKind::NilLit => Ok(TypedExpr {
                kind: TypedExprKind::NilLit,
                ty: Type::Nil,
                span,
            }),
            ResolvedExprKind::BinOp(op, left, right) => self.infer_binop(op, *left, *right, span),
            ResolvedExprKind::Not(e) => {
                let typed_e = self.check_expr(*e, &Type::Bool)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Not(Box::new(typed_e)),
                    ty: Type::Bool,
                    span,
                })
            }
            ResolvedExprKind::Negate(e) => {
                let typed_e = self.check_expr(*e, &Type::Int)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Negate(Box::new(typed_e)),
                    ty: Type::Int,
                    span,
                })
            }
            ResolvedExprKind::VariantLit(enum_id, variant_name, payload) => {
                self.check_variant_lit(enum_id, variant_name, payload, span)
            }
            ResolvedExprKind::Match(scrutinee, arms) => self.check_match(*scrutinee, arms, span),
            ResolvedExprKind::FuncCall(call) => match call {
                ResolvedFuncCall::User(user_call) => {
                    let sig = self
                        .func_signatures
                        .get(&user_call.name)
                        .ok_or(TypeError::UndefinedType(user_call.span))?
                        .clone();

                    let typed_call = self.check_user_func_call(user_call, &sig)?;
                    let return_ty = if sig.is_sync {
                        sig.return_type
                    } else {
                        Type::Chan(Box::new(sig.return_type.clone()))
                    };

                    Ok(TypedExpr {
                        kind: TypedExprKind::FuncCall(TypedFuncCall::User(typed_call)),
                        ty: return_ty,
                        span,
                    })
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
                    let typed_item = self.infer_expr(item)?;
                    types.push(typed_item.ty.clone());
                    typed_items.push(typed_item);
                }
                Ok(TypedExpr {
                    kind: TypedExprKind::TupleLit(typed_items),
                    ty: Type::Tuple(types),
                    span,
                })
            }
            ResolvedExprKind::Append(list_expr, item_expr) => {
                let typed_list = self.infer_expr(*list_expr)?;
                let typed_item = self.infer_expr(*item_expr)?;

                let (list_ty, coerced_item) = match typed_list.ty.clone() {
                    Type::List(elem_ty) => {
                        let coerced = self.check_types_match(typed_item, &elem_ty)?;
                        (Type::List(elem_ty), coerced)
                    }
                    Type::EmptyList => (Type::List(Box::new(typed_item.ty.clone())), typed_item),
                    _ => {
                        return Err(TypeError::NotAList {
                            ty: typed_list.ty,
                            span,
                        });
                    }
                };

                Ok(TypedExpr {
                    kind: TypedExprKind::Append(Box::new(typed_list), Box::new(coerced_item)),
                    ty: list_ty,
                    span,
                })
            }
            ResolvedExprKind::Prepend(item_expr, list_expr) => {
                let typed_item = self.infer_expr(*item_expr)?;
                let typed_list = self.infer_expr(*list_expr)?;

                let (list_ty, coerced_item) = match typed_list.ty.clone() {
                    Type::List(elem_ty) => {
                        let coerced = self.check_types_match(typed_item, &elem_ty)?;
                        (Type::List(elem_ty), coerced)
                    }
                    Type::EmptyList => (Type::List(Box::new(typed_item.ty.clone())), typed_item),
                    _ => {
                        return Err(TypeError::NotAList {
                            ty: typed_list.ty,
                            span,
                        });
                    }
                };

                Ok(TypedExpr {
                    kind: TypedExprKind::Prepend(Box::new(coerced_item), Box::new(typed_list)),
                    ty: list_ty,
                    span,
                })
            }
            ResolvedExprKind::Store(collection, key, value) => {
                let typed_collection = self.infer_expr(*collection)?;

                match typed_collection.ty.clone() {
                    Type::List(elem_ty) => {
                        let typed_key = self.infer_expr(*key)?;
                        self.check_type_compatibility(&Type::Int, &typed_key.ty, span)?;

                        let typed_value = self.check_expr(*value, &elem_ty)?;

                        Ok(TypedExpr {
                            kind: TypedExprKind::Store(
                                Box::new(typed_collection.clone()),
                                Box::new(typed_key),
                                Box::new(typed_value),
                            ),
                            ty: typed_collection.ty,
                            span,
                        })
                    }
                    Type::Map(key_ty, val_ty) => {
                        let typed_key = self.check_expr(*key, &key_ty)?;
                        let typed_value = self.check_expr(*value, &val_ty)?;

                        Ok(TypedExpr {
                            kind: TypedExprKind::Store(
                                Box::new(typed_collection.clone()),
                                Box::new(typed_key),
                                Box::new(typed_value),
                            ),
                            ty: typed_collection.ty,
                            span,
                        })
                    }
                    Type::Struct(struct_id, _) => {
                        let key_span = key.span;
                        let field_name = if let ResolvedExprKind::StringLit(s) = &key.kind {
                            s.clone()
                        } else {
                            return Err(TypeError::InvalidStructKeyType {
                                found: self.infer_expr(*key)?.ty,
                                span: key_span,
                            });
                        };

                        let field_ty = self.get_field_type(struct_id, &field_name, span)?;
                        let typed_value = self.check_expr(*value, &field_ty)?;

                        let typed_key = TypedExpr {
                            kind: TypedExprKind::StringLit(field_name),
                            ty: Type::String,
                            span: key_span,
                        };

                        Ok(TypedExpr {
                            kind: TypedExprKind::Store(
                                Box::new(typed_collection.clone()),
                                Box::new(typed_key),
                                Box::new(typed_value),
                            ),
                            ty: typed_collection.ty,
                            span,
                        })
                    }
                    _ => Err(TypeError::StoreOnInvalidType {
                        ty: typed_collection.ty,
                        span,
                    }),
                }
            }
            ResolvedExprKind::Head(list_expr) => {
                let typed_list = self.infer_expr(*list_expr)?;
                if let Type::List(elem_ty) = typed_list.ty.clone() {
                    Ok(TypedExpr {
                        kind: TypedExprKind::Head(Box::new(typed_list)),
                        ty: *elem_ty,
                        span,
                    })
                } else {
                    Err(TypeError::NotAList {
                        ty: typed_list.ty,
                        span,
                    })
                }
            }
            ResolvedExprKind::Tail(list_expr) => {
                let typed_list = self.infer_expr(*list_expr)?;
                if let Type::List(_) = &typed_list.ty {
                    Ok(TypedExpr {
                        kind: TypedExprKind::Tail(Box::new(typed_list.clone())),
                        ty: typed_list.ty,
                        span,
                    })
                } else {
                    Err(TypeError::NotAList {
                        ty: typed_list.ty,
                        span,
                    })
                }
            }
            ResolvedExprKind::Len(expr) => {
                let typed_e = self.infer_expr(*expr)?;
                match typed_e.ty {
                    Type::List(_) | Type::Map(_, _) | Type::String => Ok(TypedExpr {
                        kind: TypedExprKind::Len(Box::new(typed_e)),
                        ty: Type::Int,
                        span,
                    }),
                    _ => Err(TypeError::InvalidUnaryOp {
                        op: "len",
                        ty: typed_e.ty,
                        span: typed_e.span,
                    }),
                }
            }
            ResolvedExprKind::Min(left, right) => {
                let typed_left = self.check_expr(*left, &Type::Int)?;
                let typed_right = self.check_expr(*right, &Type::Int)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Min(Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Int,
                    span,
                })
            }
            ResolvedExprKind::Exists(map_expr, key_expr) => {
                let typed_map = self.infer_expr(*map_expr)?;
                let typed_key = self.infer_expr(*key_expr)?;

                if let Type::Map(expected_key_ty, _) = &typed_map.ty {
                    self.check_type_compatibility(expected_key_ty, &typed_key.ty, span)?;
                    Ok(TypedExpr {
                        kind: TypedExprKind::Exists(Box::new(typed_map), Box::new(typed_key)),
                        ty: Type::Bool,
                        span,
                    })
                } else {
                    Err(TypeError::InvalidUnaryOp {
                        op: "exists",
                        ty: typed_map.ty,
                        span,
                    })
                }
            }
            ResolvedExprKind::Erase(map_expr, key_expr) => {
                let typed_map = self.infer_expr(*map_expr)?;
                let typed_key = self.infer_expr(*key_expr)?;

                if let Type::Map(expected_key_ty, _) = &typed_map.ty {
                    self.check_type_compatibility(expected_key_ty, &typed_key.ty, span)?;
                    Ok(TypedExpr {
                        kind: TypedExprKind::Erase(
                            Box::new(typed_map.clone()),
                            Box::new(typed_key),
                        ),
                        ty: typed_map.ty,
                        span,
                    })
                } else {
                    Err(TypeError::InvalidUnaryOp {
                        op: "erase",
                        ty: typed_map.ty,
                        span,
                    })
                }
            }
            ResolvedExprKind::MakeChannel(cap) => {
                let typed_cap = self.check_expr(*cap, &Type::Int)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::MakeChannel(Box::new(typed_cap)),
                    ty: Type::UnknownChannel,
                    span,
                })
            }
            ResolvedExprKind::Send(chan_expr, val_expr) => {
                if self.current_func_is_sync {
                    return Err(TypeError::SendInSyncFunc { span });
                }

                let typed_chan = self.infer_expr(*chan_expr)?;

                if let Type::Chan(inner_ty) = &typed_chan.ty {
                    let typed_val = self.check_expr(*val_expr, inner_ty)?;
                    Ok(TypedExpr {
                        kind: TypedExprKind::Send(Box::new(typed_chan), Box::new(typed_val)),
                        ty: Type::Tuple(vec![]),
                        span,
                    })
                } else {
                    Err(TypeError::NotAChannel {
                        ty: typed_chan.ty,
                        span,
                    })
                }
            }
            ResolvedExprKind::Recv(chan_expr) => {
                if self.current_func_is_sync {
                    return Err(TypeError::RecvInSyncFunc { span });
                }

                let typed_chan = self.infer_expr(*chan_expr)?;

                if let Type::Chan(inner_ty) = typed_chan.ty.clone() {
                    Ok(TypedExpr {
                        kind: TypedExprKind::Recv(Box::new(typed_chan)),
                        ty: *inner_ty,
                        span,
                    })
                } else {
                    Err(TypeError::NotAChannel {
                        ty: typed_chan.ty,
                        span,
                    })
                }
            }
            ResolvedExprKind::SetTimer => Ok(TypedExpr {
                kind: TypedExprKind::SetTimer,
                ty: Type::Chan(Box::new(Type::Tuple(vec![]))),
                span,
            }),
            ResolvedExprKind::Index(target, index) => self.infer_index(*target, *index, span),
            ResolvedExprKind::Slice(target, start, end) => {
                let typed_target = self.infer_expr(*target)?;
                let typed_start = self.check_expr(*start, &Type::Int)?;
                let typed_end = self.check_expr(*end, &Type::Int)?;

                match typed_target.ty.clone() {
                    Type::List(_) | Type::String => Ok(TypedExpr {
                        kind: TypedExprKind::Slice(
                            Box::new(typed_target.clone()),
                            Box::new(typed_start),
                            Box::new(typed_end),
                        ),
                        ty: typed_target.ty,
                        span,
                    }),
                    _ => Err(TypeError::NotIndexable {
                        ty: typed_target.ty,
                        span,
                    }),
                }
            }
            ResolvedExprKind::TupleAccess(tuple_expr, index) => {
                let typed_tuple = self.infer_expr(*tuple_expr)?;
                if let Type::Tuple(types) = typed_tuple.ty.clone() {
                    if index < types.len() {
                        Ok(TypedExpr {
                            kind: TypedExprKind::TupleAccess(Box::new(typed_tuple), index),
                            ty: types[index].clone(),
                            span,
                        })
                    } else {
                        Err(TypeError::TupleIndexOutOfBounds {
                            index,
                            size: types.len(),
                            span,
                        })
                    }
                } else {
                    Err(TypeError::NotATuple {
                        ty: typed_tuple.ty,
                        span,
                    })
                }
            }
            ResolvedExprKind::FieldAccess(struct_expr, field_name) => {
                let typed_struct = self.infer_expr(*struct_expr)?;
                if matches!(typed_struct.ty, Type::Optional(_)) {
                    return Err(TypeError::NotAStruct {
                        ty: typed_struct.ty,
                        field_name: field_name.clone(),
                        span,
                    });
                }
                if let Type::Struct(struct_id, _) = typed_struct.ty {
                    let field_ty = self.get_field_type(struct_id, &field_name, span)?;
                    Ok(TypedExpr {
                        kind: TypedExprKind::FieldAccess(Box::new(typed_struct), field_name),
                        ty: field_ty,
                        span,
                    })
                } else {
                    Err(TypeError::NotAStruct {
                        ty: typed_struct.ty,
                        field_name: field_name.clone(),
                        span,
                    })
                }
            }
            ResolvedExprKind::Unwrap(e) => {
                let typed_e = self.infer_expr(*e)?;
                let inner_ty = typed_e.ty.clone();

                match inner_ty {
                    Type::Optional(t) => Ok(TypedExpr {
                        kind: TypedExprKind::UnwrapOptional(Box::new(typed_e)),
                        ty: *t,
                        span,
                    }),
                    _ => Err(TypeError::UnwrapOnNonOptional { ty: inner_ty, span }),
                }
            }
            ResolvedExprKind::StructLit(struct_id, fields) => {
                self.check_struct_literal(struct_id, fields, span)
            }
            ResolvedExprKind::RpcCall(target, call) => {
                let typed_target = self.infer_expr(*target)?;
                let role_id = match &typed_target.ty {
                    Type::Role(id, _) => *id,
                    _ => {
                        return Err(TypeError::RpcCallTargetNotRole {
                            ty: typed_target.ty,
                            span: typed_target.span,
                        });
                    }
                };

                let role_funcs = self
                    .role_func_signatures
                    .get(&role_id)
                    .ok_or(TypeError::UndefinedType(typed_target.span))?;

                let (func_id, sig) = role_funcs.get(&call.original_name).ok_or_else(|| {
                    TypeError::FieldNotFound {
                        field_name: call.original_name.clone(),
                        span: call.span,
                    }
                })?;
                if sig.is_sync {
                    return Err(TypeError::RpcCallToSyncFunc {
                        func_name: call.original_name.clone(),
                        span,
                    });
                }

                // Clone data before mutable borrow
                let func_id = *func_id;
                let params = sig.params.clone();
                let return_type = sig.return_type.clone();

                let typed_args = self.check_call_args(&params, call.args, call.span)?;
                let typed_call = TypedUserFuncCall {
                    name: func_id,
                    original_name: call.original_name,
                    args: typed_args,
                    return_type: return_type.clone(),
                    span: call.span,
                };

                let return_ty = Type::Chan(Box::new(return_type));
                Ok(TypedExpr {
                    kind: TypedExprKind::RpcCall(Box::new(typed_target), typed_call),
                    ty: return_ty,
                    span,
                })
            }
        }
    }

    fn check_types_match(
        &self,
        mut typed_expr: TypedExpr,
        expected: &Type,
    ) -> Result<TypedExpr, TypeError> {
        let actual = &typed_expr.ty;

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
    ) -> Result<TypedExpr, TypeError> {
        match op {
            BinOp::Add => {
                // Infer types of both operands
                let typed_left = self.infer_expr(left)?;
                let typed_right = self.infer_expr(right)?;

                // Both must be the same type (either Int or String)
                match (&typed_left.ty, &typed_right.ty) {
                    (Type::Int, Type::Int) => Ok(TypedExpr {
                        kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                        ty: Type::Int,
                        span,
                    }),
                    (Type::String, Type::String) => Ok(TypedExpr {
                        kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                        ty: Type::String,
                        span,
                    }),
                    _ => Err(TypeError::InvalidBinOp {
                        op: op.clone(),
                        left: typed_left.ty.clone(),
                        right: typed_right.ty.clone(),
                        span,
                    }),
                }
            }
            BinOp::Subtract | BinOp::Multiply | BinOp::Divide | BinOp::Modulo => {
                let typed_left = self.check_expr(left, &Type::Int)?;
                let typed_right = self.check_expr(right, &Type::Int)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Int,
                    span,
                })
            }
            BinOp::Equal | BinOp::NotEqual => {
                let typed_left = self.infer_expr(left)?;
                let typed_right = self.infer_expr(right)?;
                let (coerced_left, coerced_right) =
                    self.check_and_coerce_symmetric(typed_left, typed_right, op.clone(), span)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(coerced_left), Box::new(coerced_right)),
                    ty: Type::Bool,
                    span,
                })
            }
            BinOp::Less | BinOp::LessEqual | BinOp::Greater | BinOp::GreaterEqual => {
                let typed_left = self.check_expr(left, &Type::Int)?;
                let typed_right = self.check_expr(right, &Type::Int)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Bool,
                    span,
                })
            }
            BinOp::And | BinOp::Or => {
                let typed_left = self.check_expr(left, &Type::Bool)?;
                let typed_right = self.check_expr(right, &Type::Bool)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                    ty: Type::Bool,
                    span,
                })
            }
            BinOp::Coalesce => {
                let typed_left = self.infer_expr(left)?;
                if let Type::Optional(inner_ty) = typed_left.ty.clone() {
                    let typed_right = self.check_expr(right, &inner_ty)?;
                    Ok(TypedExpr {
                        kind: TypedExprKind::BinOp(op, Box::new(typed_left), Box::new(typed_right)),
                        ty: *inner_ty,
                        span,
                    })
                } else {
                    Err(TypeError::InvalidBinOp {
                        op: BinOp::Coalesce,
                        left: typed_left.ty,
                        right: Type::Nil,
                        span,
                    })
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
    ) -> Result<Vec<TypedExpr>, TypeError> {
        if args.len() != expected_params.len() {
            return Err(TypeError::WrongNumberOfArgs {
                expected: expected_params.len(),
                got: args.len(),
                span: call_span,
            });
        }

        let mut typed_args = Vec::new();
        for (arg, expected_ty) in args.into_iter().zip(expected_params.iter()) {
            let typed_arg = self.check_expr(arg, expected_ty)?;
            typed_args.push(typed_arg);
        }
        Ok(typed_args)
    }

    fn check_builtin_call(
        &mut self,
        builtin: BuiltinFn,
        args: Vec<ResolvedExpr>,
        span: Span,
    ) -> Result<TypedExpr, TypeError> {
        let sig = self.builtin_signatures.get(&builtin).unwrap_or_else(|| {
            panic!(
                "Missing signature for builtin function '{:?}'. \
                     This should have been initialized when the type checker was created.",
                builtin
            )
        });

        // Clone data before mutable borrow
        let params = sig.params.clone();
        let return_type = sig.return_type.clone();

        let typed_args = self.check_call_args(&params, args, span)?;

        Ok(TypedExpr {
            kind: TypedExprKind::FuncCall(TypedFuncCall::Builtin(
                builtin,
                typed_args,
                return_type.clone(),
            )),
            ty: return_type,
            span,
        })
    }

    fn check_user_func_call(
        &mut self,
        call: ResolvedUserFuncCall,
        sig: &FunctionSignature,
    ) -> Result<TypedUserFuncCall, TypeError> {
        let typed_args = self.check_call_args(&sig.params, call.args, call.span)?;

        Ok(TypedUserFuncCall {
            name: call.name,
            original_name: call.original_name,
            args: typed_args,
            return_type: sig.return_type.clone(),
            span: call.span,
        })
    }

    fn infer_index(
        &mut self,
        target: ResolvedExpr,
        index: ResolvedExpr,
        span: Span,
    ) -> Result<TypedExpr, TypeError> {
        let typed_target = self.infer_expr(target)?;
        let typed_index = self.infer_expr(index)?;

        let target_ty = typed_target.ty.clone();
        let index_ty = typed_index.ty.clone();

        match target_ty {
            Type::List(elem_ty) => {
                self.check_type_compatibility(&Type::Int, &index_ty, span)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Index(Box::new(typed_target), Box::new(typed_index)),
                    ty: *elem_ty,
                    span,
                })
            }
            Type::Map(key_ty, val_ty) => {
                self.check_type_compatibility(&key_ty, &index_ty, span)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Index(Box::new(typed_target), Box::new(typed_index)),
                    ty: *val_ty,
                    span,
                })
            }
            Type::String => {
                self.check_type_compatibility(&Type::Int, &index_ty, span)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Index(Box::new(typed_target), Box::new(typed_index)),
                    ty: Type::String,
                    span,
                })
            }
            _ => Err(TypeError::NotIndexable {
                ty: typed_target.ty,
                span,
            }),
        }
    }

    fn infer_list_literal(
        &mut self,
        mut items: Vec<ResolvedExpr>,
        span: Span,
    ) -> Result<TypedExpr, TypeError> {
        if items.is_empty() {
            return Ok(TypedExpr {
                kind: TypedExprKind::ListLit(vec![]),
                ty: Type::EmptyList,
                span,
            });
        }

        let mut typed_items = Vec::new();
        let first_typed = self.infer_expr(items.remove(0))?;
        let first_ty = first_typed.ty.clone();
        typed_items.push(first_typed);

        for item in items {
            let typed_item = self.check_expr(item, &first_ty)?;
            typed_items.push(typed_item);
        }

        Ok(TypedExpr {
            kind: TypedExprKind::ListLit(typed_items),
            ty: Type::List(Box::new(first_ty)),
            span,
        })
    }

    fn infer_map_literal(
        &mut self,
        mut pairs: Vec<(ResolvedExpr, ResolvedExpr)>,
        span: Span,
    ) -> Result<TypedExpr, TypeError> {
        if pairs.is_empty() {
            return Ok(TypedExpr {
                kind: TypedExprKind::MapLit(vec![]),
                ty: Type::EmptyMap,
                span,
            });
        }

        let (first_key, first_val) = pairs.remove(0);
        let typed_first_key = self.infer_expr(first_key)?;
        let typed_first_val = self.infer_expr(first_val)?;
        let key_ty = typed_first_key.ty.clone();
        let val_ty = typed_first_val.ty.clone();

        let mut typed_pairs = vec![(typed_first_key, typed_first_val)];

        for (key, val) in pairs {
            let typed_key = self.check_expr(key, &key_ty)?;
            let typed_val = self.check_expr(val, &val_ty)?;
            typed_pairs.push((typed_key, typed_val));
        }

        Ok(TypedExpr {
            kind: TypedExprKind::MapLit(typed_pairs),
            ty: Type::Map(Box::new(key_ty), Box::new(val_ty)),
            span,
        })
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
    ) -> Result<TypedExpr, TypeError> {
        let (resolved_id, field_defs) = self.get_resolved_struct_fields(struct_id, span, "")?;

        let mut typed_fields = Vec::new();
        for (field_name, field_expr) in fields {
            let field_def = field_defs
                .iter()
                .find(|f| f.name == field_name)
                .ok_or_else(|| TypeError::UndefinedStructField {
                    field_name: field_name.clone(),
                    span,
                })?;

            let expected_ty = self.resolve_type(&field_def.type_def)?;
            let typed_field_expr = self.check_expr(field_expr, &expected_ty)?;
            typed_fields.push((field_name, typed_field_expr));
        }

        for field_def in field_defs {
            if !typed_fields.iter().any(|(name, _)| name == &field_def.name) {
                return Err(TypeError::MissingStructField {
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
        Ok(TypedExpr {
            kind: TypedExprKind::StructLit(resolved_id, typed_fields),
            ty,
            span,
        })
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
        }
    }

    fn check_type_compatibility(
        &self,
        expected: &Type,
        actual: &Type,
        span: Span,
    ) -> Result<(), TypeError> {
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
