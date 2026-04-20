# Language Design

## EBNF Grammar

```ebnf
program ::= top_level_def* EOF

top_level_def ::=
  role_def
  | client_def
  | type_def_stmt
  | func_def

role_def ::= 'role' ID '{' var_inits func_defs '}'
client_def ::= 'ClientInterface' '{' var_inits func_defs '}'

func_defs ::= ( func_def )*
func_def ::= ( '@trace' )? ( 'async' )? 'fn' ID '(' func_params? ')' ( ':' type_def )? block

func_call ::= ID '(' args? ')'
args ::= expr ( ',' expr )* ','?

var_inits ::= ( var_init ';'? )*
var_init ::= 'var' var_target ( ':' type_def )? '=' expr
var_target ::= ID | '(' ID ( ',' ID )* ','? ')'

func_params ::= ID ':' type_def ( ',' ID ':' type_def )* ','?

type_def_list ::= type_def ( ',' type_def )* ','?

type_def_stmt ::= 'type' ID ( struct_body | enum_def | type_alias ) ';'?
struct_body ::= '{' field_defs? '}'
field_defs ::= field_def ( ';' field_def )* ';'?
field_def ::= ID ':' type_def

enum_def ::= 'enum' '{' enum_variants '}'
enum_variant ::= ID ( '(' type_def ')' )?
enum_variants ::= enum_variant ( ',' enum_variant )* ','?

type_alias ::= '=' type_def

type_def ::= base_type ( '?' )? ( '{' ID '|' expr '}' )?

base_type ::=
  ID
  | 'map' '<' type_def ',' type_def '>'
  | 'list' '<' type_def '>'
  | 'chan' '<' type_def '>'
  | '(' type_def_list? ')'

(* If the last item is an expression without a
   trailing semicolon, it becomes the block's tail expression / implicit
   return value. *)
block ::= '{' block_item* '}'
block_item ::= statement | expr ';'?

statements ::= ( statement | expr ';'? )*
statement ::=
  for_loop
  | for_in_loop
  | simple_stmt ';'?

simple_stmt ::= assignment

for_loop ::= 'for' ( assignment? ';' expr? ';' assignment? | expr | ) '{' statements '}'
for_in_loop ::= 'for' pattern 'in' expr '{' statements '}'

assignment ::= assign_target ( ':' type_def )? '=' expr

assign_target ::= assign_item ( ',' assign_item )* ','?

assign_item ::=
  'var' ID
  | '_'
  | ID
  | '(' assign_target ')'

pattern ::=
  ID '.' ID ( '(' pattern ')' )?
  | ID
  | '_'
  | '(' ')'
  | '(' pattern_list ')'

pattern_list ::= pattern ( ',' pattern )* ','?

expr ::= control_flow_expr

control_flow_expr ::=
  'return' expr?
  | 'break'
  | 'continue'
  | send_expr

send_expr ::= coalescing_expr ( '>-' coalescing_expr )?

coalescing_expr ::= boolean_or_expr ( '??' boolean_or_expr )*

boolean_or_expr ::= boolean_and_expr ( 'or' boolean_and_expr )*

boolean_and_expr ::= comparison_expr ( 'and' comparison_expr )*

comparison_expr ::= additive_expr ( ('==' | '!=' | '<' | '<=' | '>' | '>=') additive_expr )?

additive_expr ::= multiplicative_expr ( ( '+' | '-' ) multiplicative_expr )*

multiplicative_expr ::= unary_expr ( ( '*' | '/' | '%' ) unary_expr )*

unary_expr ::= ( '!' | '-' | '<-' ) unary_expr | primary_expr

primary_expr ::= primary_base postfix_op*

primary_base ::=
  'true' | 'false' | 'nil'
  | literals
  | func_call
  | match_expr
  | cond_expr
  | named_dot_access
  | struct_literal
  | collection
  | list_ops
  | 'append' '(' expr ',' expr ')'
  | 'prepend' '(' expr ',' expr ')'
  | 'store' '(' expr ',' expr ',' expr ')'
  | 'min' '(' expr ',' expr ')'
  | 'exists' '(' expr ',' expr ')'
  | 'erase' '(' expr ',' expr ')'
  | 'persist_data' '(' expr ')'
  | 'retrieve_data' '<' type_def '>' '(' ')'
  | 'discard_data' '(' ')'
  | 'make' '(' ')'
  | 'send' '(' expr ',' expr ')'
  | 'recv' '(' expr ')'
  | 'set_timer' '(' string_literal? ')'
  | ID
  | '(' expr ')'

postfix_op ::=
  '[' expr ']'
  | '[' expr ':' expr ']'
  | '.' INT
  | '.' ID
  | '?.' INT
  | '?.' ID
  | '?' '[' expr ']'
  | '!'
  | '->' func_call
  | ':=' expr
  | 'with' '{' with_entries '}'

with_entries ::= with_entry ( ',' with_entry )* ','?
with_entry ::= ID ':' expr | '[' expr ']' ':' expr

cond_expr ::= 'if' expr block ( 'else' 'if' expr block )* ( 'else' block )?

match_expr ::= 'match' expr '{' match_arms '}'
match_arms ::= match_arm ( ',' match_arm )* ','?
match_arm ::= pattern '=>' ( block | expr )

named_dot_access ::= ID '.' ID ( '(' expr ')' )?

literals ::= STRING | INT | fstring
fstring ::= FSTRING_START ( expr FSTRING_PART )* expr FSTRING_END

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
- Concurrency: `chan<T>`
- Optional types: `T?`, which can be either `nil` or a value of type `T`
- Refinement types: `T { x | expr }`, where `x` binds a value of type `T` and `expr` must be of type `bool`

### Refinements

A refinement type `T { x | expr }` attaches a boolean predicate to an existing type. The bound variable `x` has type `T` and is in scope within `expr`. Example:

```
type Positive = int { x | x > 0 };
var n: int { v | v > 0 } = 42;
```

- Refinements can appear anywhere a type is expected: variable declarations, function parameters, return types, struct fields, type aliases, and nested inside generic types like `list<int { y | y > 0 }>`.
- The body must be a side-effect-free expression of type `bool`. Only built-in function calls are allowed in the body (user function calls are rejected).
- Dependent parameter refinements are supported: in `fn f(n: int, v: list<int> { xs | len(xs) == n })`, the refinement body can reference earlier parameters.

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

### Bulk Updates with `with`

The `with` keyword provides a postfix syntax for applying multiple updates at once.
It desugars to nested `store()` calls, applied left-to-right.

#### Struct Fields

```
var updated = record with { age: 31, active: true };
// desugars to: store(store(record, "age", 31), "active", true)
```

#### Collection Keys

```
var updated_map = my_map with { ["key1"]: "val1", ["key2"]: "val2" };
var updated_list = my_list with { [0]: first_item };
```

#### Nested Updates

For deep updates, compose `with` expressions manually:

```
var updated = record with { address: record.address with { zip: 12345 } };
```

#### Restrictions

`with` inherits the same restrictions as `:=`.

## Concurrency

Spur supports two function types for concurrency: `sync` (default) and `async`.

By default, all functions are synchronous. A sync function is a blocking, atomic call that:

- Cannot use channel operations (`send` and `recv`)

A function can be explicitly marked as asynchronous with the `async` keyword:

```
async fn my_async_call(): int {
  return 10;
}
```

Calling an async function does not block execution and immediately returns a `chan<T>`.
To get the actual return value, you must receive from the channel, which will pause the current task.

This sync-first model, where most operations are blocking, can result in issues around concurrent processing.

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

Use the `make()` function to create a new channel:

```
var ch = make();
```

#### Sending Values

Send values to a channel using the `>-` operator or the `send()` function:

```
42 >- ch;                // Send the value 42 to channel ch
"hello" >- msg_chan;      // Send a string to msg_chan
send(ch, 42);            // Equivalent builtin form
```

#### Receiving Values

Receive values from a channel using either `recv()` or the `<-` operator:

```
var value = recv(ch);      // Explicit recv call
var value = <- ch;          // Syntactic sugar using <- operator
```

Both forms block until a value is available on the channel.

#### Restrictions

Channel operations (`>-`, `send`, `<-`, and `recv`) cannot be used inside sync functions (non-async). Attempting to do so will result in a compile-time error. This ensures that synchronous functions remain non-blocking and atomic.

#### Channel Behavior During Crashes

Channel operations are resilient, but are implicitly affected by node crashes:

- When a node crashes, any channels awaiting `recv` will indefinitely pause execution until recovery.
- If a channel tries to resolve an asynchronous continuation during a crashed state, the runtime will raise a simulator error: `"Channel not found in async continuation"`.
- On recovery, pending continuations begin processing incoming records immediately after the _first yield point_ of `RecoverInit`. Be cautious when structuring asynchronous logic around potential crash points.

## Safe Navigation

The `?.` and `?[]` operators provide safe navigation on optional types, short-circuiting to `nil` if the receiver is `nil`.

### Safe Field Access

```
var name: string? = person?.name;
```

If `person` is `nil`, the whole expression evaluates to `nil`. If `person` is non-nil, the field is accessed normally. The result type is always `T?` where `T` is the field type.

### Safe Index Access

```
var val: int? = my_map?["key"];
var elem: int? = my_list?[0];
```

Same nil-guarding behavior for map and list indexing. The receiver must be an optional collection type.

### Safe Tuple Access

```
var first: int? = my_tuple?.0;
```

### Chaining

Safe navigation operators compose with each other and with `??`:

```
var city: string? = person?.address?.city;
var city_or_default: string = person?.address?.city ?? "unknown";
```

### Note

Safe navigation is read-only. It cannot be combined with `:=` update syntax.

## Additional Operators

### Unwrap

The unwrap `!` operator is used for unwrapping an optional.
In other words, `o!` either retrieves the value or panics if the optional is `nil`.

## Built-in Functions

We also have a variety of built-in functions.
Right now, this includes:

- `println: string -> ()`
- `int_to_string: int -> string`

## Syntactic Notes

- Variable declarations are illegal in for loop increments.
