# Language Design

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
var_init ::= 'var' ID ( ':' type_def )? '=' expr

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
| 'chan' '<' type_def '>'
| 'lock'
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
| lock_stmt
| 'break' ';'

cond_stmts ::= if_stmt ( else_if_stmt )* ( else_stmt )?
if_stmt ::= 'if' '(' expr ')' '{' statements '}'
else_if_stmt ::= 'else' 'if' '(' expr ')' '{' statements '}'
else_stmt ::= 'else' '{' statements '}'

for_loop ::= 'for' ( ( var_init | assignment )? ';' expr? ';' assignment? | expr | ) '{' statements '}'
for_in_loop ::= 'for' pattern 'in' expr '{' statements '}'
lock_stmt ::= 'lock' '(' expr ')' '{' statements '}'

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

unary_expr ::= ( '!' | '-' | '<-' ) unary_expr | primary_expr

primary_expr ::= primary_base postfix_op*

primary_base ::=
ID
| 'true' | 'false' | 'nil'
| literals
| func_call
| struct_literal
| collection
| list_ops
| 'append' '(' expr ',' expr ')'
| 'prepend' '(' expr ',' expr ')'
| 'store' '(' expr ',' expr ',' expr ')'
| 'min' '(' expr ',' expr ')'
| 'exists' '(' expr ',' expr ')'
| 'erase' '(' expr ',' expr ')'
| 'make' '(' expr ')'
| 'send' '(' expr ',' expr ')'
| 'recv' '(' expr ')'
| 'create_lock' '(' ')'
| '(' expr ')'

postfix_op ::=
'[' expr ']'
| '[' expr ':' expr ']'
| '.' INT
| '.' ID
| '!'
| '->' func_call
| ':=' expr

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
```

## Typing

Spur is a strongly and statically typed language.

The type system is composed of:

- Primitives: `int`, `string`, `bool`
- Tuples and the unit type: `()`, `(T)`, `(T, U)`, etc.
- Collections: `list<T>`, `map<K, V>`
- Concurrency: `chan<T>` and `lock`
- Optional types: `T?`, which can be either `nil` or a value of type `T`

## Reference and Value Semantics

Primitives and tuples are passed by value.
All other types are passed by reference.
This is likely irrelevant, since all maps, structs, and lists are immutable.

## Struct Updates

Spur provides immutable update syntax for structs, maps, and lists using the `:=` operator.
This allows you to create a modified copy without mutating the original.

### Field Update Syntax

You can update struct fields using the `.field := value` syntax:

```
var updated = record.address.city := "New York";
```

This creates a copy of `record` with the nested field `address.city` updated to `"New York"`.
All intermediate structures are copied to preserve immutability.

### Index Update Syntax

You can update map or list elements using the `[key] := value` syntax:

```
var updated = record["age"] := 30;
```

This creates a copy of `record` with the `"age"` key updated to `30`.

### Nested Updates

Both syntaxes can be chained for deeply nested updates:

```
var updated = record.address.zip := 12345;
var updated2 = my_map["outer"]["inner"] := value;
```

### Desugaring

Update expressions are syntactic sugar for the `store` built-in function:
- `x.field := value` desugars to `store(x, "field", value)`
- `x[key] := value` desugars to `store(x, key, value)`
- Nested updates like `x.a.b := v` desugar to `store(x, "a", store(x.a, "b", v))`

## Concurrency

Spur supports two function types for concurrency: `async` (default) and `sync`.

By default, all functions are asynchronous. 
Calling a standard func does not block execution and immediately returns a `chan<T>`. 
To get the actual return value, you must receive from the channel, which will pause the current task.

A function can be explicitly marked as synchronous with the `sync` keyword:
```
sync func my_sync_call() -> int {
  return 10;
}
```
A `sync func` is a blocking, atomic call. They are restricted and cannot contain:
- RPC calls
- Calls to other `async` functions
- Channel operations (`send` and `recv`)

This async-first model, where most operations are non-blocking, can result in issues around synchronizing asynchronous code.\
To address this, we provide a `lock` type.

### RPCs

This asynchronous model extends seamlessly to remote procedure calls (RPCs). RPCs use the `->` arrow operator
to call a function on a target role instance:

```
var f: chan<string> = other_role->some_func(1, 2);
```

### Channels

Channels provide a typed communication mechanism for passing values between concurrent tasks.

#### Channel Type

Channels have type `chan<T>` where `T` is the type of values sent through the channel:
```
var my_chan: chan<int>;
var msg_chan: chan<string>;
```

#### Creating Channels

Use the `make()` function to create a new channel with a specified buffer size:
```
var ch = make(10);  // Creates a channel with buffer size 10
```

#### Sending Values

Send values to a channel using the `send()` function:
```
send(ch, 42);       // Send the value 42 to channel ch
send(msg_chan, "hello");  // Send a string to msg_chan
```

#### Receiving Values

Receive values from a channel using either `recv()` or the `<-` operator:
```
var value = recv(ch);      // Explicit recv call
var value = <- ch;          // Syntactic sugar using <- operator
```

Both forms block until a value is available on the channel.

#### Restrictions

Channel operations (`send` and `recv`) cannot be used inside `sync` functions. Attempting to do so will result in a compile-time error. This ensures that synchronous functions remain non-blocking and atomic.

## Additional Operators

### Unwrap

The unwrap `!` operator is used for unwrapping an optional. 
In other words, `o!` either retrieves the value or panics if the optional is `nil`.

## Built-in Functions
We also have a variety of built-in functions.
Right now, this includes:
- `println: string -> ()`
- `int_to_string: int -> string`