use crate::analysis::resolver::{
    NameId, PrepopulatedTypes, ResolvedAssignment, ResolvedClientDef, ResolvedCondStmts,
    ResolvedExpr, ResolvedExprKind, ResolvedForInLoop, ResolvedForLoop, ResolvedFuncCall,
    ResolvedFuncDef, ResolvedPattern, ResolvedPatternKind, ResolvedProgram, ResolvedRoleDef,
    ResolvedStatement, ResolvedStatementKind, ResolvedTopLevelDef, ResolvedTypeDef,
    ResolvedTypeDefStmtKind, ResolvedVarInit,
};
use crate::parser::{BinOp, Span};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Tuple(Vec<Type>),
    Struct(NameId, String),
    Role(NameId, String),
    Optional(Box<Type>),
    Future(Box<Type>),
    Promise(Box<Type>),
    Lock,

    // Placeholder types.
    EmptyList,
    EmptyMap,
    EmptyPromise,
    Nil,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::String => write!(f, "string"),
            Type::Bool => write!(f, "bool"),
            Type::List(t) => write!(f, "list<{}>", t),
            Type::Map(k, v) => write!(f, "map<{}, {}>", k, v),
            Type::Tuple(ts) => {
                let inner = ts
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "({})", inner)
            }
            Type::Struct(_, name) => write!(f, "{}", name),
            Type::Role(_, name) => write!(f, "{}", name),
            Type::Optional(t) => write!(f, "{}?", t),
            Type::Future(t) => write!(f, "future<{}>", t),
            Type::Promise(t) => write!(f, "promise<{}>", t),
            Type::Lock => write!(f, "lock"),
            Type::EmptyList => write!(f, "empty list"),
            Type::EmptyMap => write!(f, "empty map"),
            Type::EmptyPromise => write!(f, "empty promise"),
            Type::Nil => write!(f, "nil"),
        }
    }
}

#[derive(Error, Debug, PartialEq, Clone)]
pub enum TypeError {
    #[error("Type Mismatch: Expected `{expected}`, but found `{found}`")]
    Mismatch {
        expected: Type,
        found: Type,
        span: Span,
    },
    #[error("Undefined type name used (this should be caught by resolver)")]
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
    #[error("Invalid map value type. Expected `{expected}`, but found `{found}`")]
    InvalidMapValueType {
        expected: Type,
        found: Type,
        span: Span,
    },
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
    #[error("Cannot apply operation to role type `{ty}`")]
    InvalidRoleOperation { ty: Type, span: Span },
    #[error("Cannot force-unwrap a non-optional type `{ty}`")]
    UnwrapOnNonOptional { ty: Type, span: Span },
    #[error("await can only be used on a future, found `{ty}`")]
    AwaitOnNonFuture { ty: Type, span: Span },
    #[error("Operation requires a promise, found `{ty}`")]
    NotAPromise { ty: Type, span: Span },
    #[error("await lock can only be used on a Lock, found `{ty}`")]
    NotALock { ty: Type, span: Span },
    #[error("Polling operation requires a collection of futures or bools, found `{ty}`")]
    PollingOnInvalidType { ty: Type, span: Span },
    #[error("next_resp requires a map of futures, found `{ty}`")]
    NextRespOnInvalidType { ty: Type, span: Span },
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    params: Vec<Type>,
    return_type: Type,
}

#[derive(Debug, Clone)]
enum TypeDefinition {
    /// A built-in type (int, string, bool, unit)
    Builtin(Type),
    /// A user-defined type (struct or alias)
    UserDefined(ResolvedTypeDefStmtKind),
}

pub struct TypeChecker {
    /// A stack of scopes, mapping a variable's `NameId` to its `Type`.
    scopes: Vec<HashMap<NameId, Type>>,
    /// Maps a type's `NameId` to its definition. Includes both built-in and user-defined types.
    type_defs: HashMap<NameId, TypeDefinition>,
    /// Maps a function's `NameId` to its signature. Populated in the first pass.
    func_signatures: HashMap<NameId, FunctionSignature>,
    /// Maps role `NameId` to role name
    role_defs: HashMap<NameId, String>,
    /// Maps struct `NameId` to struct name
    struct_names: HashMap<NameId, String>,
    /// The expected return type of the current function being checked.
    current_return_type: Option<Type>,
    /// Track if we're inside a loop (for break statements)
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
        predefined.insert(
            prepopulated_types.unit,
            TypeDefinition::Builtin(Type::Tuple(vec![])),
        );
        predefined.insert(prepopulated_types.lock, TypeDefinition::Builtin(Type::Lock));

        Self {
            scopes: vec![HashMap::new()], // Global scope
            type_defs: predefined,
            func_signatures: HashMap::new(),
            role_defs: HashMap::new(),
            struct_names: HashMap::new(),
            current_return_type: None,
            loop_depth: 0,
        }
    }

    /// Main entry point for type checking
    pub fn check_program(&mut self, program: ResolvedProgram) -> Result<(), TypeError> {
        // First pass: collect all type definitions and function signatures
        self.collect_definitions(&program)?;

        // Second pass: type check all function bodies
        for top_level_def in &program.top_level_defs {
            match top_level_def {
                ResolvedTopLevelDef::Role(role) => {
                    self.check_role_def(role)?;
                }
                ResolvedTopLevelDef::Type(_) => {
                    // Already processed in first pass
                }
            }
        }

        // Check client definition
        self.check_client_def(&program.client_def)?;

        Ok(())
    }

    /// First pass: collect all type and function definitions
    fn collect_definitions(&mut self, program: &ResolvedProgram) -> Result<(), TypeError> {
        // Collect type definitions
        for top_level_def in &program.top_level_defs {
            match top_level_def {
                ResolvedTopLevelDef::Type(type_def) => {
                    self.type_defs.insert(
                        type_def.name,
                        TypeDefinition::UserDefined(type_def.def.clone()),
                    );
                    // Store the struct name for later use
                    self.struct_names
                        .insert(type_def.name, type_def.original_name.clone());
                }
                ResolvedTopLevelDef::Role(role) => {
                    self.role_defs.insert(role.name, role.original_name.clone());

                    // Collect function signatures from this role
                    for func in &role.func_defs {
                        let sig = self.build_function_signature(func)?;
                        self.func_signatures.insert(func.name, sig);
                    }
                }
            }
        }

        // Collect client function signatures
        for func in &program.client_def.func_defs {
            let sig = self.build_function_signature(func)?;
            self.func_signatures.insert(func.name, sig);
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
        })
    }

    fn check_role_def(&mut self, role: &ResolvedRoleDef) -> Result<(), TypeError> {
        self.enter_scope();

        // Check role variable initializations
        for var_init in &role.var_inits {
            self.check_var_init(var_init)?;
        }

        // Check all functions in this role
        for func in &role.func_defs {
            self.check_func_def(func)?;
        }

        self.exit_scope();
        Ok(())
    }

    fn check_client_def(&mut self, client: &ResolvedClientDef) -> Result<(), TypeError> {
        self.enter_scope();

        // Check client variable initializations
        for var_init in &client.var_inits {
            self.check_var_init(var_init)?;
        }

        // Check all functions in the client
        for func in &client.func_defs {
            self.check_func_def(func)?;
        }

        self.exit_scope();
        Ok(())
    }

    fn check_func_def(&mut self, func: &ResolvedFuncDef) -> Result<(), TypeError> {
        let sig = self
            .func_signatures
            .get(&func.name)
            .expect("Function signature should be collected")
            .clone();

        self.enter_scope();
        self.current_return_type = Some(sig.return_type.clone());

        // Add parameters to scope
        for (param, param_type) in func.params.iter().zip(sig.params.iter()) {
            self.add_var(param.name, param_type.clone());
        }

        // Check function body
        let mut has_return = false;
        for stmt in &func.body {
            if self.check_statement(stmt)? {
                has_return = true;
            }
        }

        // Check that non-unit functions have a return statement
        if sig.return_type != Type::Tuple(vec![]) && !has_return {
            return Err(TypeError::MissingReturn(func.span));
        }

        self.current_return_type = None;
        self.exit_scope();
        Ok(())
    }

    fn check_var_init(&mut self, var_init: &ResolvedVarInit) -> Result<(), TypeError> {
        let expected_type = self.resolve_type(&var_init.type_def)?;
        let value_type = self.check_expr(&var_init.value)?;

        self.check_type_compatibility(&expected_type, &value_type, var_init.span)?;
        self.add_var(var_init.name, expected_type);
        Ok(())
    }

    /// Check a statement. Returns true if the statement definitely returns.
    fn check_statement(&mut self, stmt: &ResolvedStatement) -> Result<bool, TypeError> {
        match &stmt.kind {
            ResolvedStatementKind::Conditional(cond) => self.check_conditional(cond),
            ResolvedStatementKind::VarInit(var_init) => {
                self.check_var_init(var_init)?;
                Ok(false)
            }
            ResolvedStatementKind::Assignment(assign) => {
                self.check_assignment(assign)?;
                Ok(false)
            }
            ResolvedStatementKind::Expr(expr) => {
                self.check_expr(expr)?;
                Ok(false)
            }
            ResolvedStatementKind::Return(expr) => {
                let expr_type = self.check_expr(expr)?;
                if let Some(expected_return_type) = &self.current_return_type {
                    self.check_type_compatibility(expected_return_type, &expr_type, stmt.span)?;
                } else {
                    return Err(TypeError::ReturnOutsideFunction(stmt.span));
                }
                Ok(true)
            }
            ResolvedStatementKind::ForLoop(for_loop) => {
                self.check_for_loop(for_loop)?;
                Ok(false)
            }
            ResolvedStatementKind::ForInLoop(for_in_loop) => {
                self.check_for_in_loop(for_in_loop)?;
                Ok(false)
            }
            ResolvedStatementKind::Print(expr) => {
                self.check_expr(expr)?;
                Ok(false)
            }
            ResolvedStatementKind::Break => {
                if self.loop_depth == 0 {
                    return Err(TypeError::BreakOutsideLoop(stmt.span));
                }
                Ok(false)
            }
            ResolvedStatementKind::Lock(lock_expr, body) => {
                let lock_ty = self.check_expr(lock_expr)?;
                if lock_ty != Type::Lock {
                    return Err(TypeError::NotALock {
                        ty: lock_ty,
                        span: lock_expr.span,
                    });
                }

                let mut body_returns = false;
                for stmt in body {
                    if self.check_statement(stmt)? {
                        body_returns = true;
                    }
                }
                Ok(body_returns)
            }
        }
    }

    fn check_conditional(&mut self, cond: &ResolvedCondStmts) -> Result<bool, TypeError> {
        // Check if condition
        let cond_type = self.check_expr(&cond.if_branch.condition)?;
        self.check_type_compatibility(&Type::Bool, &cond_type, cond.if_branch.span)?;

        let mut if_returns = false;
        for stmt in &cond.if_branch.body {
            if self.check_statement(stmt)? {
                if_returns = true;
            }
        }

        // Check elseif branches
        let mut all_branches_return = if_returns;
        for elseif in &cond.elseif_branches {
            let elseif_cond_type = self.check_expr(&elseif.condition)?;
            self.check_type_compatibility(&Type::Bool, &elseif_cond_type, elseif.span)?;

            let mut elseif_returns = false;
            for stmt in &elseif.body {
                if self.check_statement(stmt)? {
                    elseif_returns = true;
                }
            }
            all_branches_return = all_branches_return && elseif_returns;
        }

        // Check else branch
        if let Some(else_body) = &cond.else_branch {
            let mut else_returns = false;
            for stmt in else_body {
                if self.check_statement(stmt)? {
                    else_returns = true;
                }
            }
            all_branches_return = all_branches_return && else_returns;
        } else {
            // No else branch means not all paths return
            all_branches_return = false;
        }

        Ok(all_branches_return)
    }

    fn check_for_loop(&mut self, for_loop: &ResolvedForLoop) -> Result<(), TypeError> {
        self.enter_scope();
        self.loop_depth += 1;

        // Check initialization
        if let Some(init) = &for_loop.init {
            match init {
                crate::analysis::resolver::ResolvedForLoopInit::VarInit(vi) => {
                    self.check_var_init(vi)?;
                }
                crate::analysis::resolver::ResolvedForLoopInit::Assignment(a) => {
                    self.check_assignment(a)?;
                }
            }
        }

        // Check condition
        if let Some(condition) = &for_loop.condition {
            let cond_type = self.check_expr(condition)?;
            self.check_type_compatibility(&Type::Bool, &cond_type, for_loop.span)?;
        }

        // Check increment
        if let Some(increment) = &for_loop.increment {
            self.check_assignment(increment)?;
        }

        // Check body
        for stmt in &for_loop.body {
            self.check_statement(stmt)?;
        }

        self.loop_depth -= 1;
        self.exit_scope();
        Ok(())
    }

    fn check_for_in_loop(&mut self, for_in_loop: &ResolvedForInLoop) -> Result<(), TypeError> {
        // Check the iterable expression
        let iterable_type = self.check_expr(&for_in_loop.iterable)?;

        // Determine the element type based on the iterable
        let element_type = match &iterable_type {
            Type::List(elem_ty) => (**elem_ty).clone(),
            Type::Map(key_ty, val_ty) => Type::Tuple(vec![(**key_ty).clone(), (**val_ty).clone()]),
            _ => {
                return Err(TypeError::NotIterable {
                    ty: iterable_type,
                    span: for_in_loop.span,
                });
            }
        };

        self.enter_scope();
        self.loop_depth += 1;

        // Check and bind pattern
        self.check_pattern(&for_in_loop.pattern, &element_type)?;

        // Check body
        for stmt in &for_in_loop.body {
            self.check_statement(stmt)?;
        }

        self.loop_depth -= 1;
        self.exit_scope();
        Ok(())
    }

    fn check_pattern(
        &mut self,
        pattern: &ResolvedPattern,
        expected_type: &Type,
    ) -> Result<(), TypeError> {
        match &pattern.kind {
            ResolvedPatternKind::Var(name_id, _) => {
                self.add_var(*name_id, expected_type.clone());
                Ok(())
            }
            ResolvedPatternKind::Wildcard => Ok(()),
            ResolvedPatternKind::Tuple(patterns) => {
                if let Type::Tuple(types) = expected_type {
                    if patterns.len() != types.len() {
                        return Err(TypeError::PatternMismatch {
                            pattern_ty: format!("tuple of {} elements", patterns.len()),
                            iterable_ty: expected_type.clone(),
                            span: pattern.span,
                        });
                    }
                    for (pat, ty) in patterns.iter().zip(types.iter()) {
                        self.check_pattern(pat, ty)?;
                    }
                    Ok(())
                } else {
                    Err(TypeError::PatternMismatch {
                        pattern_ty: "tuple".to_string(),
                        iterable_ty: expected_type.clone(),
                        span: pattern.span,
                    })
                }
            }
        }
    }

    fn check_assignment(&mut self, assign: &ResolvedAssignment) -> Result<(), TypeError> {
        // Check if target is a valid lvalue
        if !self.is_valid_lvalue(&assign.target.kind) {
            return Err(TypeError::InvalidAssignmentTarget(assign.span));
        }

        let target_type = self.check_expr(&assign.target)?;
        let value_type = self.check_expr(&assign.value)?;

        self.check_type_compatibility(&target_type, &value_type, assign.span)?;
        Ok(())
    }

    fn is_valid_lvalue(&self, expr_kind: &ResolvedExprKind) -> bool {
        match expr_kind {
            ResolvedExprKind::Var(_, _) => true,
            ResolvedExprKind::FieldAccess(_, _) => true,
            ResolvedExprKind::Index(_, _) => true,
            _ => false,
        }
    }

    fn check_expr(&mut self, expr: &ResolvedExpr) -> Result<Type, TypeError> {
        match &expr.kind {
            ResolvedExprKind::Var(name_id, _) => self.get_var_type(*name_id, expr.span),
            ResolvedExprKind::IntLit(_) => Ok(Type::Int),
            ResolvedExprKind::StringLit(_) => Ok(Type::String),
            ResolvedExprKind::BoolLit(_) => Ok(Type::Bool),
            ResolvedExprKind::NilLit => Ok(Type::Nil),
            ResolvedExprKind::BinOp(op, left, right) => {
                self.check_binop(op.clone(), left, right, expr.span)
            }
            ResolvedExprKind::Not(e) => {
                let ty = self.check_expr(e)?;
                self.check_type_compatibility(&Type::Bool, &ty, expr.span)?;
                Ok(Type::Bool)
            }
            ResolvedExprKind::Negate(e) => {
                let ty = self.check_expr(e)?;
                self.check_type_compatibility(&Type::Int, &ty, expr.span)?;
                Ok(Type::Int)
            }
            ResolvedExprKind::FuncCall(call) => {
                let return_type = self.check_func_call(call)?;
                Ok(Type::Future(Box::new(return_type)))
            }
            ResolvedExprKind::MapLit(pairs) => self.check_map_literal(pairs, expr.span),
            ResolvedExprKind::ListLit(items) => self.check_list_literal(items, expr.span),
            ResolvedExprKind::TupleLit(items) => {
                let types = items
                    .iter()
                    .map(|item| self.check_expr(item))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Type::Tuple(types))
            }
            ResolvedExprKind::Append(list_expr, item_expr) => {
                let list_type = self.check_expr(list_expr)?;
                let item_type = self.check_expr(item_expr)?;

                match list_type {
                    Type::List(elem_ty) => {
                        self.check_type_compatibility(elem_ty.as_ref(), &item_type, expr.span)?;
                        Ok(Type::List(elem_ty))
                    }
                    Type::EmptyList => Ok(Type::List(Box::new(item_type))),
                    _ => Err(TypeError::NotAList {
                        ty: list_type,
                        span: expr.span,
                    }),
                }
            }
            ResolvedExprKind::Prepend(item_expr, list_expr) => {
                let item_type = self.check_expr(item_expr)?;
                let list_type = self.check_expr(list_expr)?;

                match list_type {
                    Type::List(elem_ty) => {
                        self.check_type_compatibility(elem_ty.as_ref(), &item_type, expr.span)?;
                        Ok(Type::List(elem_ty))
                    }
                    Type::EmptyList => {
                        // `prepend(item, [])` creates a new `List<typeof(item)>`
                        Ok(Type::List(Box::new(item_type)))
                    }
                    _ => Err(TypeError::NotAList {
                        ty: list_type,
                        span: expr.span,
                    }),
                }
            }
            ResolvedExprKind::Head(list_expr) => {
                let list_type = self.check_expr(list_expr)?;
                if let Type::List(elem_ty) = list_type {
                    Ok(*elem_ty)
                } else {
                    Err(TypeError::NotAList {
                        ty: list_type,
                        span: expr.span,
                    })
                }
            }
            ResolvedExprKind::Tail(list_expr) => {
                let list_type = self.check_expr(list_expr)?;
                if let Type::List(_) = &list_type {
                    Ok(list_type)
                } else {
                    Err(TypeError::NotAList {
                        ty: list_type,
                        span: expr.span,
                    })
                }
            }
            ResolvedExprKind::Len(expr) => {
                let ty = self.check_expr(expr)?;
                match ty {
                    Type::List(_) | Type::Map(_, _) | Type::String => Ok(Type::Int),
                    _ => Err(TypeError::InvalidUnaryOp {
                        op: "len",
                        ty,
                        span: expr.span,
                    }),
                }
            }
            ResolvedExprKind::Min(left, right) => {
                let left_ty = self.check_expr(left)?;
                let right_ty = self.check_expr(right)?;
                self.check_type_compatibility(&Type::Int, &left_ty, expr.span)?;
                self.check_type_compatibility(&Type::Int, &right_ty, expr.span)?;
                Ok(Type::Int)
            }
            ResolvedExprKind::Exists(map_expr, key_expr) => {
                let map_ty = self.check_expr(map_expr)?;
                let key_ty = self.check_expr(key_expr)?;

                if let Type::Map(expected_key_ty, _) = &map_ty {
                    self.check_type_compatibility(expected_key_ty, &key_ty, expr.span)?;
                    Ok(Type::Bool)
                } else {
                    Err(TypeError::InvalidUnaryOp {
                        op: "exists",
                        ty: map_ty,
                        span: expr.span,
                    })
                }
            }
            ResolvedExprKind::Erase(map_expr, key_expr) => {
                // NEW
                let map_ty = self.check_expr(map_expr)?;
                let key_ty = self.check_expr(key_expr)?;

                if let Type::Map(expected_key_ty, value_ty) = &map_ty {
                    self.check_type_compatibility(expected_key_ty, &key_ty, expr.span)?;
                    Ok(Type::Map(expected_key_ty.clone(), value_ty.clone()))
                } else {
                    Err(TypeError::InvalidUnaryOp {
                        op: "erase",
                        ty: map_ty,
                        span: expr.span,
                    })
                }
            }
            ResolvedExprKind::CreatePromise => Ok(Type::EmptyPromise),
            ResolvedExprKind::CreateFuture(promise_expr) => {
                let promise_type = self.check_expr(promise_expr)?;
                match promise_type {
                    Type::Promise(inner_ty) => Ok(Type::Future(inner_ty)),
                    _ => Err(TypeError::NotAPromise {
                        ty: promise_type,
                        span: promise_expr.span,
                    }),
                }
            }
            ResolvedExprKind::ResolvePromise(promise_expr, value_expr) => {
                let promise_type = self.check_expr(promise_expr)?;
                let value_type = self.check_expr(value_expr)?;

                match promise_type {
                    Type::Promise(inner_ty) => {
                        // resolve_promise(promise<T>, T)
                        self.check_type_compatibility(
                            inner_ty.as_ref(),
                            &value_type,
                            value_expr.span,
                        )?;
                        // resolve_promise returns unit
                        Ok(Type::Tuple(vec![]))
                    }
                    _ => Err(TypeError::NotAPromise {
                        ty: promise_type,
                        span: promise_expr.span,
                    }),
                }
            }
            ResolvedExprKind::CreateLock => Ok(Type::Lock),
            ResolvedExprKind::Index(target, index) => self.check_index(target, index, expr.span),
            ResolvedExprKind::Slice(target, start, end) => {
                let target_ty = self.check_expr(target)?;
                let start_ty = self.check_expr(start)?;
                let end_ty = self.check_expr(end)?;

                self.check_type_compatibility(&Type::Int, &start_ty, expr.span)?;
                self.check_type_compatibility(&Type::Int, &end_ty, expr.span)?;

                match target_ty {
                    Type::List(_) | Type::String => Ok(target_ty),
                    _ => Err(TypeError::NotIndexable {
                        ty: target_ty,
                        span: expr.span,
                    }),
                }
            }
            ResolvedExprKind::TupleAccess(tuple_expr, index) => {
                let tuple_ty = self.check_expr(tuple_expr)?;
                if let Type::Tuple(types) = tuple_ty {
                    if *index < types.len() {
                        Ok(types[*index].clone())
                    } else {
                        Err(TypeError::TupleIndexOutOfBounds {
                            index: *index,
                            size: types.len(),
                            span: expr.span,
                        })
                    }
                } else {
                    Err(TypeError::NotATuple {
                        ty: tuple_ty,
                        span: expr.span,
                    })
                }
            }
            ResolvedExprKind::FieldAccess(struct_expr, field_name) => {
                let struct_ty = self.check_expr(struct_expr)?;
                if matches!(struct_ty, Type::Optional(_)) {
                    return Err(TypeError::NotAStruct {
                        ty: struct_ty,
                        field_name: field_name.clone(),
                        span: expr.span,
                    });
                }
                if let Type::Struct(struct_id, _) = struct_ty {
                    self.get_field_type(struct_id, field_name, expr.span)
                } else {
                    Err(TypeError::NotAStruct {
                        ty: struct_ty,
                        field_name: field_name.clone(),
                        span: expr.span,
                    })
                }
            }
            ResolvedExprKind::Unwrap(e) => {
                let inner_ty = self.check_expr(e)?;
                match inner_ty {
                    Type::Optional(t) => Ok(*t), // T? -> T
                    _ => Err(TypeError::UnwrapOnNonOptional {
                        ty: inner_ty,
                        span: expr.span,
                    }),
                }
            }
            ResolvedExprKind::StructLit(struct_id, fields) => {
                self.check_struct_literal(*struct_id, fields, expr.span)
            }
            ResolvedExprKind::RpcCall(target, call) => {
                let target_ty = self.check_expr(target)?;
                if !matches!(target_ty, Type::Role(..)) {
                    return Err(TypeError::RpcCallTargetNotRole {
                        ty: target_ty,
                        span: expr.span,
                    });
                }
                // Asynchronous call, returns future<T>
                let return_type = self.check_func_call(call)?;
                Ok(Type::Future(Box::new(return_type)))
            }

            ResolvedExprKind::Await(e) => {
                let future_ty = self.check_expr(e)?;
                match future_ty {
                    Type::Future(inner_ty) => Ok(*inner_ty), // await future<T> returns T
                    _ => Err(TypeError::AwaitOnNonFuture {
                        ty: future_ty,
                        span: expr.span,
                    }),
                }
            }

            ResolvedExprKind::Spawn(e) => {
                let inner_ty = self.check_expr(e)?;
                // spawn T returns future<T>
                Ok(Type::Future(Box::new(inner_ty)))
            }

            ResolvedExprKind::PollForResps(collection, _value) => {
                let col_ty = self.check_expr(collection)?;
                match &col_ty {
                    Type::List(elem_ty) => {
                        if !matches!(**elem_ty, Type::Future(_) | Type::Bool) {
                            return Err(TypeError::PollingOnInvalidType {
                                ty: col_ty,
                                span: collection.span,
                            });
                        }
                    }
                    Type::Map(_, val_ty) => {
                        if !matches!(**val_ty, Type::Future(_) | Type::Bool) {
                            return Err(TypeError::PollingOnInvalidType {
                                ty: col_ty,
                                span: collection.span,
                            });
                        }
                    }
                    _ => {
                        return Err(TypeError::PollingOnInvalidType {
                            ty: col_ty,
                            span: collection.span,
                        });
                    }
                }
                Ok(Type::Int)
            }

            ResolvedExprKind::PollForAnyResp(collection) => {
                let col_ty = self.check_expr(collection)?;
                match &col_ty {
                    Type::List(elem_ty) => {
                        if !matches!(**elem_ty, Type::Future(_) | Type::Bool) {
                            return Err(TypeError::PollingOnInvalidType {
                                ty: col_ty,
                                span: collection.span,
                            });
                        }
                    }
                    Type::Map(_, val_ty) => {
                        if !matches!(**val_ty, Type::Future(_) | Type::Bool) {
                            return Err(TypeError::PollingOnInvalidType {
                                ty: col_ty,
                                span: collection.span,
                            });
                        }
                    }
                    _ => {
                        return Err(TypeError::PollingOnInvalidType {
                            ty: col_ty,
                            span: collection.span,
                        });
                    }
                }
                Ok(Type::Bool)
            }

            ResolvedExprKind::NextResp(collection) => {
                let col_ty = self.check_expr(collection)?;
                match col_ty {
                    Type::Map(key_ty, val_ty) => {
                        if let Type::Future(inner_ty) = *val_ty {
                            // Returns the value inside the future
                            Ok(*inner_ty)
                        } else {
                            Err(TypeError::NextRespOnInvalidType {
                                ty: Type::Map(key_ty, val_ty),
                                span: collection.span,
                            })
                        }
                    }
                    _ => Err(TypeError::NextRespOnInvalidType {
                        ty: col_ty,
                        span: collection.span,
                    }),
                }
            }
        }
    }

    fn check_binop(
        &mut self,
        op: BinOp,
        left: &ResolvedExpr,
        right: &ResolvedExpr,
        span: Span,
    ) -> Result<Type, TypeError> {
        let left_ty = self.check_expr(left)?;
        let right_ty = self.check_expr(right)?;

        match op {
            BinOp::Add | BinOp::Subtract | BinOp::Multiply | BinOp::Divide | BinOp::Modulo => {
                self.check_type_compatibility(&Type::Int, &left_ty, left.span)?;
                self.check_type_compatibility(&Type::Int, &right_ty, right.span)?;
                Ok(Type::Int)
            }
            BinOp::Equal | BinOp::NotEqual => {
                if (matches!(left_ty, Type::Optional(_)) && right_ty == Type::Nil)
                    || (left_ty == Type::Nil && matches!(right_ty, Type::Optional(_)))
                {
                    return Ok(Type::Bool);
                }
                self.check_type_compatibility(&left_ty, &right_ty, right.span)?;
                Ok(Type::Bool)
            }
            BinOp::Less | BinOp::LessEqual | BinOp::Greater | BinOp::GreaterEqual => {
                self.check_type_compatibility(&Type::Int, &left_ty, left.span)?;
                self.check_type_compatibility(&Type::Int, &right_ty, right.span)?;
                Ok(Type::Bool)
            }
            BinOp::And | BinOp::Or => {
                self.check_type_compatibility(&Type::Bool, &left_ty, left.span)?;
                self.check_type_compatibility(&Type::Bool, &right_ty, right.span)?;
                Ok(Type::Bool)
            }
            BinOp::Coalesce => {
                let left_ty = self.check_expr(left)?;
                let right_ty = self.check_expr(right)?;

                if let Type::Optional(inner_ty) = left_ty {
                    self.check_type_compatibility(inner_ty.as_ref(), &right_ty, span)?;
                    Ok(*inner_ty)
                } else {
                    Err(TypeError::InvalidBinOp {
                        //
                        op: BinOp::Coalesce,
                        left: left_ty,
                        right: right_ty,
                        span,
                    })
                }
            }
        }
    }

    fn check_func_call(&mut self, call: &ResolvedFuncCall) -> Result<Type, TypeError> {
        let sig = self
            .func_signatures
            .get(&call.name)
            .ok_or_else(|| TypeError::UndefinedType(call.span))?
            .clone();

        if call.args.len() != sig.params.len() {
            return Err(TypeError::WrongNumberOfArgs {
                expected: sig.params.len(),
                got: call.args.len(),
                span: call.span,
            });
        }

        for (arg, param_ty) in call.args.iter().zip(sig.params.iter()) {
            let arg_ty = self.check_expr(arg)?;
            self.check_type_compatibility(param_ty, &arg_ty, arg.span)?;
        }

        Ok(sig.return_type)
    }

    fn check_index(
        &mut self,
        target: &ResolvedExpr,
        index: &ResolvedExpr,
        span: Span,
    ) -> Result<Type, TypeError> {
        let target_ty = self.check_expr(target)?;
        let index_ty = self.check_expr(index)?;

        match target_ty {
            Type::List(elem_ty) => {
                self.check_type_compatibility(&Type::Int, &index_ty, span)?;
                Ok(*elem_ty)
            }
            Type::Map(key_ty, val_ty) => {
                self.check_type_compatibility(&key_ty, &index_ty, span)?;
                Ok(*val_ty)
            }
            Type::String => {
                self.check_type_compatibility(&Type::Int, &index_ty, span)?;
                Ok(Type::String)
            }
            _ => Err(TypeError::NotIndexable {
                ty: target_ty,
                span,
            }),
        }
    }

    fn check_list_literal(&mut self, items: &[ResolvedExpr], _: Span) -> Result<Type, TypeError> {
        if items.is_empty() {
            return Ok(Type::EmptyList);
        }

        let first_ty = self.check_expr(&items[0])?;
        for item in &items[1..] {
            let item_ty = self.check_expr(item)?;
            self.check_type_compatibility(&first_ty, &item_ty, item.span)?;
        }

        Ok(Type::List(Box::new(first_ty)))
    }

    fn check_map_literal(
        &mut self,
        pairs: &[(ResolvedExpr, ResolvedExpr)],
        _: Span,
    ) -> Result<Type, TypeError> {
        if pairs.is_empty() {
            return Ok(Type::EmptyMap);
        }

        let (first_key, first_val) = &pairs[0];
        let key_ty = self.check_expr(first_key)?;
        let val_ty = self.check_expr(first_val)?;

        for (key, val) in &pairs[1..] {
            let k_ty = self.check_expr(key)?;
            let v_ty = self.check_expr(val)?;

            self.check_type_compatibility(&key_ty, &k_ty, key.span)?;
            self.check_type_compatibility(&val_ty, &v_ty, val.span)?;
        }

        Ok(Type::Map(Box::new(key_ty), Box::new(val_ty)))
    }
    fn check_struct_literal(
        &mut self,
        struct_id: NameId,
        fields: &[(String, ResolvedExpr)],
        span: Span,
    ) -> Result<Type, TypeError> {
        let type_def = self
            .type_defs
            .get(&struct_id)
            .ok_or(TypeError::UndefinedType(span))?
            .clone();

        let field_defs = match type_def {
            TypeDefinition::Builtin(_) => {
                let struct_name = self
                    .struct_names
                    .get(&struct_id)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| format!("struct_{}", struct_id.0));
                return Err(TypeError::NotAStruct {
                    ty: Type::Struct(struct_id, struct_name),
                    field_name: "".to_string(),
                    span,
                });
            }
            TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Struct(fields)) => fields,
            TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Alias(_)) => {
                let struct_name = self
                    .struct_names
                    .get(&struct_id)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| format!("struct_{}", struct_id.0));
                return Err(TypeError::NotAStruct {
                    ty: Type::Struct(struct_id, struct_name),
                    field_name: "".to_string(),
                    span,
                });
            }
        };

        // Check all provided fields exist and have correct types
        for (field_name, field_expr) in fields {
            let field_def = field_defs
                .iter()
                .find(|f| &f.name == field_name)
                .ok_or_else(|| TypeError::UndefinedStructField {
                    field_name: field_name.clone(),
                    span,
                })?;

            let expected_ty = self.resolve_type(&field_def.type_def)?;
            let actual_ty = self.check_expr(field_expr)?;
            self.check_type_compatibility(&expected_ty, &actual_ty, field_expr.span)?;
        }

        // Check all required fields are provided
        for field_def in &field_defs {
            if !fields.iter().any(|(name, _)| name == &field_def.name) {
                return Err(TypeError::MissingStructField {
                    field_name: field_def.name.clone(),
                    span,
                });
            }
        }

        let struct_name = self
            .struct_names
            .get(&struct_id)
            .map(|s| s.clone())
            .unwrap_or_else(|| format!("struct_{}", struct_id.0));
        Ok(Type::Struct(struct_id, struct_name))
    }

    fn get_field_type(
        &self,
        struct_id: NameId,
        field_name: &str,
        span: Span,
    ) -> Result<Type, TypeError> {
        let type_def = self
            .type_defs
            .get(&struct_id)
            .ok_or(TypeError::UndefinedType(span))?;

        let field_defs = match type_def {
            TypeDefinition::Builtin(_) => {
                let struct_name = self
                    .struct_names
                    .get(&struct_id)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| format!("struct_{}", struct_id.0));
                return Err(TypeError::NotAStruct {
                    ty: Type::Struct(struct_id, struct_name),
                    field_name: field_name.to_string(),
                    span,
                });
            }
            TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Struct(fields)) => fields,
            TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Alias(_)) => {
                let struct_name = self
                    .struct_names
                    .get(&struct_id)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| format!("struct_{}", struct_id.0));
                return Err(TypeError::NotAStruct {
                    ty: Type::Struct(struct_id, struct_name),
                    field_name: field_name.to_string(),
                    span,
                });
            }
        };

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
                // Check if it's in type_defs (built-in or user-defined)
                if let Some(def) = self.type_defs.get(name_id) {
                    match def {
                        TypeDefinition::Builtin(ty) => Ok(ty.clone()),
                        TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Struct(_)) => {
                            let name = self
                                .struct_names
                                .get(name_id)
                                .map(|s| s.clone())
                                .unwrap_or_else(|| format!("struct_{}", name_id.0));
                            Ok(Type::Struct(*name_id, name))
                        }
                        TypeDefinition::UserDefined(ResolvedTypeDefStmtKind::Alias(
                            aliased_type,
                        )) => self.resolve_type(aliased_type),
                    }
                } else if let Some(role_name) = self.role_defs.get(name_id) {
                    Ok(Type::Role(*name_id, role_name.clone()))
                } else {
                    // Type not found - create with a fallback name
                    let name = self
                        .struct_names
                        .get(name_id)
                        .map(|s| s.clone())
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
                // "Collapse" nested optionals: T?? becomes T?
                if matches!(base_type, Type::Optional(_)) {
                    Ok(base_type)
                } else {
                    Ok(Type::Optional(Box::new(base_type)))
                }
            }
            ResolvedTypeDef::Future(t) => {
                let base_type = self.resolve_type(t)?;
                Ok(Type::Future(Box::new(base_type)))
            }
            ResolvedTypeDef::Promise(t) => {
                let base_type = self.resolve_type(t)?;
                Ok(Type::Promise(Box::new(base_type)))
            }
            ResolvedTypeDef::Lock => Ok(Type::Lock),
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

        // Check for "Empty" placeholders
        if let (Type::List(_), Type::EmptyList) = (expected, actual) {
            return Ok(());
        }
        if let (Type::Map(_, _), Type::EmptyMap) = (expected, actual) {
            return Ok(());
        }
        if let (Type::Promise(_), Type::EmptyPromise) = (expected, actual) {
            return Ok(());
        }

        // Check for optional "widening"
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
        // Check for future "widening" (e.g., if expected is Future<T> and actual is Future<T>)
        // This is primarily for placeholder types like EmptyList/Map when used with futures.
        if let (Type::Future(expected_inner), Type::Future(actual_inner)) = (expected, actual) {
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
