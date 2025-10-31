use crate::parser::*;
use std::collections::HashMap;
use thiserror::Error;

// A unique identifier for every named entity (variable, function, type, role).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NameId(pub usize);

#[derive(Error, Debug, PartialEq, Clone)]
pub enum ResolutionError {
    #[error("Name `{0}` not found in this scope")]
    NameNotFound(String, Span),
    #[error("Name `{0}` is already defined in this scope")]
    DuplicateName(String, Span),
    #[error("Direct call to function `{0}` is not allowed from another role")]
    InvalidCrossRoleCall(String, Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedProgram {
    pub top_level_defs: Vec<ResolvedTopLevelDef>,
    pub client_def: ResolvedClientDef,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedTopLevelDef {
    Role(ResolvedRoleDef),
    Type(ResolvedTypeDefStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedRoleDef {
    pub name: NameId,
    pub original_name: String,
    pub var_inits: Vec<ResolvedVarInit>,
    pub func_defs: Vec<ResolvedFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedClientDef {
    pub var_inits: Vec<ResolvedVarInit>,
    pub func_defs: Vec<ResolvedFuncDef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFuncDef {
    pub name: NameId,
    pub original_name: String,
    pub params: Vec<ResolvedFuncParam>,
    pub return_type: Option<ResolvedTypeDef>,
    pub body: Vec<ResolvedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFuncParam {
    pub name: NameId,
    pub original_name: String,
    pub type_def: ResolvedTypeDef,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedVarInit {
    pub name: NameId,
    pub original_name: String,
    pub type_def: ResolvedTypeDef,
    pub value: ResolvedExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTypeDefStmt {
    pub name: NameId,
    pub original_name: String,
    pub def: ResolvedTypeDefStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedTypeDefStmtKind {
    Struct(Vec<ResolvedFieldDef>),
    Alias(ResolvedTypeDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFieldDef {
    pub name: String, // Field names don't need resolution as they live in a struct's namespace
    pub type_def: ResolvedTypeDef,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedTypeDef {
    Named(NameId),
    Map(Box<ResolvedTypeDef>, Box<ResolvedTypeDef>),
    List(Box<ResolvedTypeDef>),
    Tuple(Vec<ResolvedTypeDef>),
    Optional(Box<ResolvedTypeDef>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedStatement {
    pub kind: ResolvedStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedStatementKind {
    Conditional(ResolvedCondStmts),
    VarInit(ResolvedVarInit),
    Assignment(ResolvedAssignment),
    Expr(ResolvedExpr),
    Return(ResolvedExpr),
    ForLoop(ResolvedForLoop),
    ForInLoop(ResolvedForInLoop),
    Print(ResolvedExpr),
    Break,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCondStmts {
    pub if_branch: ResolvedIfBranch,
    pub elseif_branches: Vec<ResolvedIfBranch>,
    pub else_branch: Option<Vec<ResolvedStatement>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedIfBranch {
    pub condition: ResolvedExpr,
    pub body: Vec<ResolvedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedForLoopInit {
    VarInit(ResolvedVarInit),
    Assignment(ResolvedAssignment),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedForLoop {
    pub init: Option<ResolvedForLoopInit>,
    pub condition: Option<ResolvedExpr>,
    pub increment: Option<ResolvedAssignment>,
    pub body: Vec<ResolvedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedForInLoop {
    pub pattern: ResolvedPattern,
    pub iterable: ResolvedExpr,
    pub body: Vec<ResolvedStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAssignment {
    pub target: ResolvedExpr,
    pub value: ResolvedExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPattern {
    pub kind: ResolvedPatternKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedPatternKind {
    Var(NameId, String),
    Wildcard,
    Tuple(Vec<ResolvedPattern>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedExpr {
    pub kind: ResolvedExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedExprKind {
    Var(NameId, String),
    IntLit(i64),
    StringLit(String),
    BoolLit(bool),
    NilLit,

    BinOp(BinOp, Box<ResolvedExpr>, Box<ResolvedExpr>),
    Not(Box<ResolvedExpr>),
    Negate(Box<ResolvedExpr>),
    FuncCall(ResolvedFuncCall),
    MapLit(Vec<(ResolvedExpr, ResolvedExpr)>),
    ListLit(Vec<ResolvedExpr>),
    TupleLit(Vec<ResolvedExpr>),
    Append(Box<ResolvedExpr>, Box<ResolvedExpr>),
    Prepend(Box<ResolvedExpr>, Box<ResolvedExpr>),
    PollForResps(Box<ResolvedExpr>, Box<ResolvedExpr>),
    PollForAnyResp(Box<ResolvedExpr>),
    NextResp(Box<ResolvedExpr>),
    Min(Box<ResolvedExpr>, Box<ResolvedExpr>),
    Exists(Box<ResolvedExpr>, Box<ResolvedExpr>),
    Erase(Box<ResolvedExpr>, Box<ResolvedExpr>),
    Head(Box<ResolvedExpr>),
    Tail(Box<ResolvedExpr>),
    Len(Box<ResolvedExpr>),
    RpcCall(Box<ResolvedExpr>, ResolvedFuncCall),
    RpcAsyncCall(Box<ResolvedExpr>, ResolvedFuncCall),
    Await(Box<ResolvedExpr>),
    Spawn(Box<ResolvedExpr>),
    Index(Box<ResolvedExpr>, Box<ResolvedExpr>),
    Slice(Box<ResolvedExpr>, Box<ResolvedExpr>, Box<ResolvedExpr>),
    TupleAccess(Box<ResolvedExpr>, usize),
    FieldAccess(Box<ResolvedExpr>, String),
    Unwrap(Box<ResolvedExpr>),
    StructLit(NameId, Vec<(String, ResolvedExpr)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFuncCall {
    pub name: NameId,
    pub original_name: String,
    pub args: Vec<ResolvedExpr>,
    pub span: Span,
}

type Scope<T> = HashMap<String, T>;

pub struct Resolver {
    var_scopes: Vec<Scope<NameId>>,
    func_scope: Scope<NameId>,
    type_scopes: Vec<Scope<NameId>>,
    role_scope: Scope<NameId>, // Roles are global and not nested
    next_id: usize,
    func_to_role_map: HashMap<NameId, String>,
    current_role: Option<String>,

    pre_populated_types: PrepopulatedTypes,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrepopulatedTypes {
    pub int: NameId,
    pub string: NameId,
    pub bool: NameId,
    pub unit: NameId,
}

impl Resolver {
    pub fn new() -> Self {
        let mut next_id = 0;
        let mut new_name_id = || {
            let id = NameId(next_id);
            next_id += 1;
            id
        };

        // Pre-populate the global scope with built-in types.
        let mut type_scope = HashMap::new();

        let int_type = new_name_id();
        let string_type = new_name_id();
        let bool_type = new_name_id();
        let unit_type = new_name_id();

        type_scope.insert("int".to_string(), int_type);
        type_scope.insert("string".to_string(), string_type);
        type_scope.insert("bool".to_string(), bool_type);
        type_scope.insert("unit".to_string(), unit_type);

        Resolver {
            var_scopes: vec![HashMap::new()],
            func_scope: HashMap::new(),
            type_scopes: vec![type_scope.clone()],
            role_scope: HashMap::new(),
            next_id,
            func_to_role_map: HashMap::new(),
            current_role: None,
            pre_populated_types: PrepopulatedTypes {
                int: int_type,
                string: string_type,
                bool: bool_type,
                unit: unit_type,
            },
        }
    }

    pub fn get_pre_populated_types(&self) -> &PrepopulatedTypes {
        &self.pre_populated_types
    }

    fn new_name_id(&mut self) -> NameId {
        let id = NameId(self.next_id);
        self.next_id += 1;
        id
    }

    fn enter_scope(&mut self) {
        self.var_scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.var_scopes.pop();
    }

    fn declare_in_scope<T>(
        scope: &mut Scope<T>,
        name: &str,
        value: T,
        span: Span,
    ) -> Result<(), ResolutionError> {
        if scope.contains_key(name) {
            Err(ResolutionError::DuplicateName(name.to_string(), span))
        } else {
            scope.insert(name.to_string(), value);
            Ok(())
        }
    }

    fn declare_var(&mut self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        let id = self.new_name_id();
        Self::declare_in_scope(self.var_scopes.last_mut().unwrap(), name, id, span)?;
        Ok(id)
    }

    fn declare_func(&mut self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        let id = self.new_name_id();
        // All functions are declared in the single global function scope
        Self::declare_in_scope(&mut self.func_scope, name, id, span)?;
        Ok(id)
    }

    fn declare_type(&mut self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        let id = self.new_name_id();
        Self::declare_in_scope(self.type_scopes.last_mut().unwrap(), name, id, span)?;
        Ok(id)
    }

    fn declare_role(&mut self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        let id = self.new_name_id();
        Self::declare_in_scope(&mut self.role_scope, name, id, span)?;
        Self::declare_in_scope(self.type_scopes.last_mut().unwrap(), name, id, span)?;
        Ok(id)
    }

    fn lookup_in_scopes<'a, T: Copy + 'a, I>(
        scopes: I,
        name: &str,
        span: Span,
    ) -> Result<T, ResolutionError>
    where
        I: IntoIterator<Item = &'a Scope<T>>,
    {
        for scope in scopes {
            if let Some(id) = scope.get(name) {
                return Ok(*id);
            }
        }
        Err(ResolutionError::NameNotFound(name.to_string(), span))
    }

    fn lookup_var(&self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        Self::lookup_in_scopes(&self.var_scopes, name, span)
    }

    fn lookup_func(&self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        Self::lookup_in_scopes(std::iter::once(&self.func_scope), name, span)
    }

    fn lookup_type(&self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        match Self::lookup_in_scopes(&self.type_scopes, name, span) {
            Ok(id) => Ok(id), // Found it as a type
            Err(ResolutionError::NameNotFound(_, _)) => {
                // Not found as a type, fall back to checking if it's a role name.
                // This allows syntax like `let x: MyRole = ...` to work.
                self.lookup_role(name, span)
            }
            Err(e) => Err(e), // Propagate any other error
        }
    }

    fn lookup_role(&self, name: &str, span: Span) -> Result<NameId, ResolutionError> {
        self.role_scope
            .get(name)
            .copied()
            .ok_or_else(|| ResolutionError::NameNotFound(name.to_string(), span))
    }

    pub fn resolve_program(mut self, program: Program) -> Result<ResolvedProgram, ResolutionError> {
        // == PASS 1: DECLARE ALL TOP-LEVEL NAMES AND MAP FUNCTIONS TO ROLES ==
        for def in &program.top_level_defs {
            if let TopLevelDef::Type(type_def) = def {
                self.declare_type(&type_def.name, type_def.span)?;
            }
        }
        for def in &program.top_level_defs {
            if let TopLevelDef::Role(role_def) = def {
                self.declare_role(&role_def.name, role_def.span)?;
            }
        }
        for def in &program.top_level_defs {
            if let TopLevelDef::Role(role_def) = def {
                for func in &role_def.func_defs {
                    self.declare_func(&func.name, func.span)?;
                }
            }
        }
        for func in &program.client_def.func_defs {
            let id = self.declare_func(&func.name, func.span)?;
            self.func_to_role_map
                .insert(id, "ClientInterface".to_string());
        }

        // == PASS 2: RESOLVE ALL BODIES ==
        let resolved_top_levels = program
            .top_level_defs
            .into_iter()
            .map(|def| self.resolve_top_level_def(def))
            .collect::<Result<_, _>>()?;

        let resolved_client = self.resolve_client_def(program.client_def)?;

        Ok(ResolvedProgram {
            top_level_defs: resolved_top_levels,
            client_def: resolved_client,
        })
    }

    fn resolve_top_level_def(
        &mut self,
        def: TopLevelDef,
    ) -> Result<ResolvedTopLevelDef, ResolutionError> {
        match def {
            TopLevelDef::Role(role) => Ok(ResolvedTopLevelDef::Role(self.resolve_role_def(role)?)),
            TopLevelDef::Type(stmt) => {
                Ok(ResolvedTopLevelDef::Type(self.resolve_type_def_stmt(stmt)?))
            }
        }
    }

    fn resolve_role_def(&mut self, role: RoleDef) -> Result<ResolvedRoleDef, ResolutionError> {
        let prev_role = self.current_role.clone();
        self.current_role = Some(role.name.clone());

        let name_id = self.lookup_role(&role.name, role.span)?;
        self.enter_scope();
        let var_inits = role
            .var_inits
            .into_iter()
            .map(|v| self.resolve_var_init(v))
            .collect::<Result<_, _>>()?;
        let func_defs = role
            .func_defs
            .into_iter()
            .map(|f| self.resolve_func_def(f))
            .collect::<Result<_, _>>()?;
        self.exit_scope();

        self.current_role = prev_role;
        Ok(ResolvedRoleDef {
            name: name_id,
            original_name: role.name,
            var_inits,
            func_defs,
            span: role.span,
        })
    }

    fn resolve_client_def(
        &mut self,
        client: ClientDef,
    ) -> Result<ResolvedClientDef, ResolutionError> {
        let prev_role = self.current_role.clone();
        self.current_role = Some("ClientInterface".to_string());

        self.enter_scope();
        let var_inits = client
            .var_inits
            .into_iter()
            .map(|v| self.resolve_var_init(v))
            .collect::<Result<_, _>>()?;
        let func_defs = client
            .func_defs
            .into_iter()
            .map(|f| self.resolve_func_def(f))
            .collect::<Result<_, _>>()?;
        self.exit_scope();

        self.current_role = prev_role;
        Ok(ResolvedClientDef {
            var_inits,
            func_defs,
            span: client.span,
        })
    }

    fn resolve_func_def(&mut self, func: FuncDef) -> Result<ResolvedFuncDef, ResolutionError> {
        let name_id = self.lookup_func(&func.name, func.span)?;
        self.enter_scope();
        let params = func
            .params
            .into_iter()
            .map(|p| self.resolve_func_param(p))
            .collect::<Result<_, _>>()?;
        let return_type = func
            .return_type
            .map(|t| self.resolve_type_def(t))
            .transpose()?;
        let body = func
            .body
            .into_iter()
            .map(|s| self.resolve_statement(s))
            .collect::<Result<_, _>>()?;
        self.exit_scope();
        Ok(ResolvedFuncDef {
            name: name_id,
            original_name: func.name,
            params,
            return_type,
            body,
            span: func.span,
        })
    }

    fn resolve_func_param(
        &mut self,
        param: FuncParam,
    ) -> Result<ResolvedFuncParam, ResolutionError> {
        let type_def = self.resolve_type_def(param.type_def)?;
        let name_id = self.declare_var(&param.name, param.span)?;
        Ok(ResolvedFuncParam {
            name: name_id,
            original_name: param.name,
            type_def,
            span: param.span,
        })
    }

    fn resolve_var_init(&mut self, init: VarInit) -> Result<ResolvedVarInit, ResolutionError> {
        let type_def = self.resolve_type_def(init.type_def)?;
        let value = self.resolve_expr(init.value)?;
        let name_id = self.declare_var(&init.name, init.span)?;
        Ok(ResolvedVarInit {
            name: name_id,
            original_name: init.name,
            type_def,
            value,
            span: init.span,
        })
    }

    fn resolve_type_def_stmt(
        &mut self,
        stmt: TypeDefStmt,
    ) -> Result<ResolvedTypeDefStmt, ResolutionError> {
        let name_id = self.lookup_type(&stmt.name, stmt.span)?;
        let def = match stmt.def {
            TypeDefStmtKind::Struct(fields) => {
                let resolved_fields = fields
                    .into_iter()
                    .map(|f| self.resolve_field_def(f))
                    .collect::<Result<_, _>>()?;
                ResolvedTypeDefStmtKind::Struct(resolved_fields)
            }
            TypeDefStmtKind::Alias(td) => {
                ResolvedTypeDefStmtKind::Alias(self.resolve_type_def(td)?)
            }
        };
        Ok(ResolvedTypeDefStmt {
            name: name_id,
            original_name: stmt.name,
            def,
            span: stmt.span,
        })
    }

    fn resolve_field_def(&mut self, field: FieldDef) -> Result<ResolvedFieldDef, ResolutionError> {
        Ok(ResolvedFieldDef {
            name: field.name,
            type_def: self.resolve_type_def(field.type_def)?,
            span: field.span,
        })
    }

    fn resolve_type_def(&mut self, td: TypeDef) -> Result<ResolvedTypeDef, ResolutionError> {
        let span = td.span;

        match td.kind {
            TypeDefKind::Named(name) => Ok(ResolvedTypeDef::Named(self.lookup_type(&name, span)?)),
            TypeDefKind::Map(k, v) => Ok(ResolvedTypeDef::Map(
                Box::new(self.resolve_type_def(*k)?),
                Box::new(self.resolve_type_def(*v)?),
            )),
            TypeDefKind::List(t) => Ok(ResolvedTypeDef::List(Box::new(self.resolve_type_def(*t)?))),
            TypeDefKind::Tuple(ts) => {
                let resolved_ts = ts
                    .into_iter()
                    .map(|t| self.resolve_type_def(t))
                    .collect::<Result<_, _>>()?;
                Ok(ResolvedTypeDef::Tuple(resolved_ts))
            }
            TypeDefKind::Optional(t) => Ok(ResolvedTypeDef::Optional(Box::new(
                self.resolve_type_def(*t)?,
            ))),
        }
    }

    fn resolve_statement(&mut self, stmt: Statement) -> Result<ResolvedStatement, ResolutionError> {
        let span = stmt.span;
        let kind = match stmt.kind {
            StatementKind::Conditional(c) => {
                ResolvedStatementKind::Conditional(self.resolve_cond_stmts(c)?)
            }
            StatementKind::VarInit(vi) => {
                ResolvedStatementKind::VarInit(self.resolve_var_init(vi)?)
            }
            StatementKind::Assignment(a) => {
                ResolvedStatementKind::Assignment(self.resolve_assignment(a)?)
            }
            StatementKind::Expr(e) => ResolvedStatementKind::Expr(self.resolve_expr(e)?),
            StatementKind::Return(e) => ResolvedStatementKind::Return(self.resolve_expr(e)?),
            StatementKind::ForLoop(fl) => {
                ResolvedStatementKind::ForLoop(self.resolve_for_loop(fl)?)
            }
            StatementKind::ForInLoop(fil) => {
                ResolvedStatementKind::ForInLoop(self.resolve_for_in_loop(fil)?)
            }
            StatementKind::Print(e) => ResolvedStatementKind::Print(self.resolve_expr(e)?),
            StatementKind::Break => ResolvedStatementKind::Break,
        };
        Ok(ResolvedStatement { kind, span })
    }

    fn resolve_block(
        &mut self,
        block: Vec<Statement>,
    ) -> Result<Vec<ResolvedStatement>, ResolutionError> {
        self.enter_scope();
        let resolved_block = block
            .into_iter()
            .map(|s| self.resolve_statement(s))
            .collect::<Result<_, _>>()?;
        self.exit_scope();
        Ok(resolved_block)
    }

    fn resolve_cond_stmts(
        &mut self,
        cond: CondStmts,
    ) -> Result<ResolvedCondStmts, ResolutionError> {
        Ok(ResolvedCondStmts {
            if_branch: self.resolve_if_branch(cond.if_branch)?,
            elseif_branches: cond
                .elseif_branches
                .into_iter()
                .map(|b| self.resolve_if_branch(b))
                .collect::<Result<_, _>>()?,
            else_branch: cond
                .else_branch
                .map(|b| self.resolve_block(b))
                .transpose()?,
            span: cond.span,
        })
    }

    fn resolve_if_branch(&mut self, branch: IfBranch) -> Result<ResolvedIfBranch, ResolutionError> {
        Ok(ResolvedIfBranch {
            condition: self.resolve_expr(branch.condition)?,
            body: self.resolve_block(branch.body)?,
            span: branch.span,
        })
    }

    fn resolve_for_loop(&mut self, loop_stmt: ForLoop) -> Result<ResolvedForLoop, ResolutionError> {
        self.enter_scope();

        let init = loop_stmt
            .init
            .map(|init_kind| match init_kind {
                ForLoopInit::VarInit(vi) => {
                    self.resolve_var_init(vi).map(ResolvedForLoopInit::VarInit)
                }
                ForLoopInit::Assignment(a) => self
                    .resolve_assignment(a)
                    .map(ResolvedForLoopInit::Assignment),
            })
            .transpose()?;

        let condition = loop_stmt
            .condition
            .map(|e| self.resolve_expr(e))
            .transpose()?;

        let increment = loop_stmt
            .increment
            .map(|a| self.resolve_assignment(a))
            .transpose()?;

        let body = loop_stmt
            .body
            .into_iter()
            .map(|s| self.resolve_statement(s))
            .collect::<Result<_, _>>()?;

        self.exit_scope();

        Ok(ResolvedForLoop {
            init,
            condition,
            increment,
            body,
            span: loop_stmt.span,
        })
    }

    fn resolve_for_in_loop(
        &mut self,
        loop_stmt: ForInLoop,
    ) -> Result<ResolvedForInLoop, ResolutionError> {
        let iterable = self.resolve_expr(loop_stmt.iterable)?;
        self.enter_scope();
        let pattern = self.resolve_pattern(loop_stmt.pattern)?;
        let body = loop_stmt
            .body
            .into_iter()
            .map(|s| self.resolve_statement(s))
            .collect::<Result<_, _>>()?;
        self.exit_scope();
        Ok(ResolvedForInLoop {
            pattern,
            iterable,
            body,
            span: loop_stmt.span,
        })
    }

    fn resolve_assignment(
        &mut self,
        assign: Assignment,
    ) -> Result<ResolvedAssignment, ResolutionError> {
        Ok(ResolvedAssignment {
            target: self.resolve_expr(assign.target)?,
            value: self.resolve_expr(assign.value)?,
            span: assign.span,
        })
    }

    fn resolve_pattern(&mut self, pat: Pattern) -> Result<ResolvedPattern, ResolutionError> {
        let span = pat.span;
        let kind = match pat.kind {
            PatternKind::Var(name) => {
                let id = self.declare_var(&name, span)?;
                ResolvedPatternKind::Var(id, name)
            }
            PatternKind::Wildcard => ResolvedPatternKind::Wildcard,
            PatternKind::Unit => ResolvedPatternKind::Tuple(vec![]),
            PatternKind::Tuple(pats) => {
                let resolved_pats = pats
                    .into_iter()
                    .map(|p| self.resolve_pattern(p))
                    .collect::<Result<_, _>>()?;
                ResolvedPatternKind::Tuple(resolved_pats)
            }
        };
        Ok(ResolvedPattern { kind, span })
    }

    fn resolve_expr(&mut self, expr: Expr) -> Result<ResolvedExpr, ResolutionError> {
        let span = expr.span;
        let kind = match expr.kind {
            ExprKind::Var(name) => ResolvedExprKind::Var(self.lookup_var(&name, span)?, name),
            ExprKind::IntLit(i) => ResolvedExprKind::IntLit(i),
            ExprKind::StringLit(s) => ResolvedExprKind::StringLit(s),
            ExprKind::BoolLit(b) => ResolvedExprKind::BoolLit(b),
            ExprKind::NilLit => ResolvedExprKind::NilLit,
            ExprKind::BinOp(op, l, r) => ResolvedExprKind::BinOp(
                op,
                Box::new(self.resolve_expr(*l)?),
                Box::new(self.resolve_expr(*r)?),
            ),
            ExprKind::Not(e) => ResolvedExprKind::Not(Box::new(self.resolve_expr(*e)?)),
            ExprKind::Negate(e) => ResolvedExprKind::Negate(Box::new(self.resolve_expr(*e)?)),
            ExprKind::FuncCall(fc) => {
                let func_name = fc.name.clone();
                let resolved_call = self.resolve_func_call(fc)?;

                if let Some(current_role_name) = &self.current_role {
                    if let Some(target_func_role) = self.func_to_role_map.get(&resolved_call.name) {
                        if current_role_name != target_func_role {
                            return Err(ResolutionError::InvalidCrossRoleCall(func_name, span));
                        }
                    }
                }
                ResolvedExprKind::FuncCall(resolved_call)
            }
            ExprKind::MapLit(pairs) => {
                let resolved_pairs = pairs
                    .into_iter()
                    .map(|(k, v)| Ok((self.resolve_expr(k)?, self.resolve_expr(v)?)))
                    .collect::<Result<_, _>>()?;
                ResolvedExprKind::MapLit(resolved_pairs)
            }
            ExprKind::ListLit(items) => {
                let resolved_items = items
                    .into_iter()
                    .map(|i| self.resolve_expr(i))
                    .collect::<Result<_, _>>()?;
                ResolvedExprKind::ListLit(resolved_items)
            }
            ExprKind::TupleLit(items) => {
                let resolved_items = items
                    .into_iter()
                    .map(|i| self.resolve_expr(i))
                    .collect::<Result<_, _>>()?;
                ResolvedExprKind::TupleLit(resolved_items)
            }
            ExprKind::Append(l, r) => ResolvedExprKind::Append(
                Box::new(self.resolve_expr(*l)?),
                Box::new(self.resolve_expr(*r)?),
            ),
            ExprKind::Prepend(l, r) => ResolvedExprKind::Prepend(
                Box::new(self.resolve_expr(*l)?),
                Box::new(self.resolve_expr(*r)?),
            ),
            ExprKind::PollForResps(e1, e2) => ResolvedExprKind::PollForResps(
                Box::new(self.resolve_expr(*e1)?),
                Box::new(self.resolve_expr(*e2)?),
            ),
            ExprKind::PollForAnyResp(e) => {
                ResolvedExprKind::PollForAnyResp(Box::new(self.resolve_expr(*e)?))
            }
            ExprKind::NextResp(e) => ResolvedExprKind::NextResp(Box::new(self.resolve_expr(*e)?)),
            ExprKind::Min(e1, e2) => ResolvedExprKind::Min(
                Box::new(self.resolve_expr(*e1)?),
                Box::new(self.resolve_expr(*e2)?),
            ),
            ExprKind::Exists(e1, e2) => ResolvedExprKind::Exists(
                Box::new(self.resolve_expr(*e1)?),
                Box::new(self.resolve_expr(*e2)?),
            ),
            ExprKind::Erase(e1, e2) => ResolvedExprKind::Erase(
                Box::new(self.resolve_expr(*e1)?),
                Box::new(self.resolve_expr(*e2)?),
            ),
            ExprKind::Head(e) => ResolvedExprKind::Head(Box::new(self.resolve_expr(*e)?)),
            ExprKind::Tail(e) => ResolvedExprKind::Tail(Box::new(self.resolve_expr(*e)?)),
            ExprKind::Len(e) => ResolvedExprKind::Len(Box::new(self.resolve_expr(*e)?)),
            ExprKind::RpcCall(target, call) => ResolvedExprKind::RpcCall(
                Box::new(self.resolve_expr(*target)?),
                self.resolve_func_call(call)?,
            ),
            ExprKind::RpcAsyncCall(target, call) => ResolvedExprKind::RpcAsyncCall(
                Box::new(self.resolve_expr(*target)?),
                self.resolve_func_call(call)?,
            ),
            ExprKind::Await(e) => ResolvedExprKind::Await(Box::new(self.resolve_expr(*e)?)),
            ExprKind::Spawn(e) => ResolvedExprKind::Spawn(Box::new(self.resolve_expr(*e)?)),
            ExprKind::Index(e, i) => ResolvedExprKind::Index(
                Box::new(self.resolve_expr(*e)?),
                Box::new(self.resolve_expr(*i)?),
            ),
            ExprKind::Slice(e, s, f) => ResolvedExprKind::Slice(
                Box::new(self.resolve_expr(*e)?),
                Box::new(self.resolve_expr(*s)?),
                Box::new(self.resolve_expr(*f)?),
            ),
            ExprKind::TupleAccess(e, i) => {
                ResolvedExprKind::TupleAccess(Box::new(self.resolve_expr(*e)?), i)
            }
            ExprKind::FieldAccess(e, name) => {
                ResolvedExprKind::FieldAccess(Box::new(self.resolve_expr(*e)?), name)
            }
            ExprKind::Unwrap(e) => ResolvedExprKind::Unwrap(Box::new(self.resolve_expr(*e)?)),
            ExprKind::StructLit(name, fields) => {
                let type_id = self.lookup_type(&name, span)?;
                let resolved_fields = fields
                    .into_iter()
                    .map(|(field_name, field_expr)| {
                        Ok((field_name, self.resolve_expr(field_expr)?))
                    })
                    .collect::<Result<Vec<_>, ResolutionError>>()?;
                ResolvedExprKind::StructLit(type_id, resolved_fields)
            }
        };
        Ok(ResolvedExpr { kind, span })
    }

    fn resolve_func_call(&mut self, call: FuncCall) -> Result<ResolvedFuncCall, ResolutionError> {
        let name_id = self.lookup_func(&call.name, call.span)?;
        let args = call
            .args
            .into_iter()
            .map(|arg| self.resolve_expr(arg))
            .collect::<Result<_, _>>()?;
        Ok(ResolvedFuncCall {
            name: name_id,
            original_name: call.name,
            args,
            span: call.span,
        })
    }
}
