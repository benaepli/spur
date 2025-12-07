# Language Design

Spur is an imperative, C-style language with modern features.

## EBNF Grammar

```ebnf
program ::= top_level_def* client_def EOF

top_level_def ::=
role_def
| type_def_stmt

role_def ::= 'role' ID '{' var_inits func_defs '}'
client_def ::= 'ClientInterface' '{' var_inits func_defs '}'

func_defs ::= ( func_def )*
func_def ::= ( 'sync' )? 'func' ID '(' func_params? ')' ( '->' type_def )? '{' statements '}'

func_call ::= ID '(' args? ')'
args ::= expr ( ',' expr )* ','?

var_inits ::= ( var_init ';' )*
var_init ::= 'var' ID ':' type_def '=' expr

func_params ::= ID ':' type_def ( ',' ID ':' type_def )* ','?

type_def_list ::= type_def ( ',' type_def )* ','?

type_def_stmt ::= 'type' ID ( struct_body | type_alias ) ';'
struct_body ::= '{' field_defs? '}'
field_defs ::= field_def ( ';' field_def )* ';'?
field_def ::= ID ':' type_def

type_alias ::= '=' type_def

type_def ::= base_type ( '?' )?

base_type ::=
ID
| 'map' '<' type_def ',' type_def '>'
| 'list' '<' type_def '>'
| 'promise' '<' type_def '>'
| 'future' '<' type_def '>'
| '(' type_def_list? ')'

statements ::= statement*
statement ::=
cond_stmts
| var_init ';'
| assignment ';'
| expr ';'
| 'return' expr ';'
| for_loop
| for_in_loop
| 'break' ';'

cond_stmts ::= if_stmt ( elseif_stmt )* ( else_stmt )?
if_stmt ::= 'if' '(' expr ')' '{' statements '}'
elseif_stmt ::= 'elseif' '(' expr ')' '{' statements '}'
else_stmt ::= 'else' '{' statements '}'

for_loop ::= 'for' '(' ( var_init | assignment )? ';' expr? ';' assignment? ')' '{' statements '}'
for_in_loop ::= 'for' '(' pattern 'in' expr ')' '{' statements '}'

assignment ::= ID '=' expr

pattern ::=
ID
| '_'
| '(' ')'
| '(' pattern_list ')'

pattern_list ::= pattern ( ',' pattern )* ','?

expr ::= coalescing_expr

coalescing_expr ::= boolean_or_expr ( '??' boolean_or_expr )*

boolean_or_expr ::= boolean_and_expr ( 'or' boolean_and_expr )*

boolean_and_expr ::= comparison_expr ( 'and' comparison_expr )*

comparison_expr ::= additive_expr ( ('==' | '!=' | '<' | '<=' | '>' | '>=') additive_expr )?

additive_expr ::= multiplicative_expr ( ( '+' | '-' ) multiplicative_expr )*

multiplicative_expr ::= unary_expr ( ( '*' | '/' | '%' ) unary_expr )*

unary_expr ::= ( '!' | '-' | 'await' | 'spin_await' | '@' ) unary_expr | primary_expr

primary_expr ::= primary_base postfix_op*

primary_base ::= 
ID
| 'true' | 'false' | 'nil'
| literals
| func_call
| struct_literal
| collection
| rpc_call
| list_ops
| 'append' '(' expr ',' expr ')'
| 'prepend' '(' expr ',' expr ')'     
| 'store' '(' expr ',' expr ',' expr ')'
| 'min' '(' expr ',' expr ')'
| 'exists' '(' expr ',' expr ')'
| 'erase' '(' expr ',' expr ')'
| 'create_promise' '(' ')'
| 'create_future' '(' expr ')'
| 'resolve_promise' '(' expr ',' expr ')'
| '(' expr ')'

postfix_op ::=
'[' expr ']'
| '[' expr ':' expr ']'
| '.' INT
| '.' ID
| '!'

literals ::= STRING | INT

collection ::= '{' kv_pairs? '}' | list_lit | tuple_lit
kv_pairs ::= ( expr ':' expr ) ( ',' expr ':' expr )* ','?

tuple_lit ::=
'(' ')'
| '(' expr ',' items? ')'

items ::= expr ( ',' expr )* ','?
list_lit ::= '[' items? ']'

struct_literal ::= ID '{' field_inits? '}'
field_inits ::= field_init ( ',' field_init )* ','?
field_init ::= ID ':' expr

list_ops ::= ( 'head' | 'tail' | 'len' ) '(' expr ')'

rpc_call ::= ( 'rpc_call' ) '('
```

## Typing

Spur is a strongly and statically typed language.

The type system is composed of:

- Primitives: `int`, `string`, `bool`
- Tuples and the unit type: `()`, `(T)`, `(T, U)`, etc.
- Collections: `list<T>`, `map<K, V>`
- Concurrency: `future<T>`, `promise<T>`, and `lock`
- Optional types: `T?`, which can be either `nil` or a value of type `T`

## Reference and Value Semantics

Primitives and tuples are passed by value.
All other types are passed by reference.

## Concurrency

Spur supports two function types for concurrency: `async` (default) and `sync`.

By default, all functions are asynchronous. 
Calling a standard func does not block execution and immediately returns a `future<T>`. 
To get the actual return value, you must await the future, which will pause the current task.

A function can be explicitly marked as synchronous with the `sync` keyword:
```
sync func my_sync_call() -> int {
  return 10;
}
```
A `sync func` is a blocking, atomic call. They are restricted and cannot contain `await`, `rpc_call`, `spin_await`,
or call other `async` functions.

This async-first model, where most operations are non-blocking, can result in issues around synchronizing asynchronous code.\
To address this, we provide a `lock` type.

### RPCs

This asynchronous model extends seamlessly to remote procedure calls (RPCs). The syntax is nearly identical,
but it requires a target role instance:

```
var f: future<string> = rpc_call(other_role, some_func(1, 2));
```

### Spin Await

For simpler, busy-wait synchronization, the language also provides `spin_await`.
```
spin_await <bool_expr>
```
This expression will pause the current task and poll until `bool_expr` evaluates to true. 
Once true, it unblocks and evaluates to ().

Note: This is a highly inefficient operation and is generally unrealistic for production systems. 
It is included primarily to map directly towards design, which is the focus of this language.
For faster execution and schedule generation/consistency checking, use promises when possible. 


## Additional Operators

### Await Shorthand

The @ operator is a prefix shorthand for the await keyword.
- `@f` is equivalent to `await f` for `f: future<T>`

### Unwrap

The unwrap `!` operator is used for unwrapping an optional. 
In other words, `o!` either retrieves the value or panics if the optional is `nil`.

## Built-in Functions
We also have a variety of built-in functions.
Right now, this includes:
- `println: string -> ()`
- `int_to_string: int -> string`