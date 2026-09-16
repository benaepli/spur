# Language Design

## EBNF Grammar

```ebnf
program ::= top_level_def* EOF

top_level_def ::=
  role_def
  | client_def
  | type_def_stmt
  | func_def

(* A role and a client each declare exactly one parameter. *)
role_def ::= 'role' ID role_param '{' var_inits func_defs '}'
client_def ::= 'client' ID role_param '{' var_inits func_defs '}'
role_param ::= '(' ID ':' type_def ')'

func_defs ::= ( func_def )*
func_def ::= annotation* ( 'async' )? 'fn' ID '(' func_params? ')' ( ':' type_def )? block

annotation ::= '@' ID ( '(' annotation_args? ')' )?
annotation_args ::= annotation_arg ( ',' annotation_arg )* ','?
annotation_arg ::= ID '=' qualified_name
qualified_name ::= ID ( '.' ID )*

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
field_def ::= annotation* ID ':' type_def

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
  | 'FifoLink' '<' type_def '>'
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
  | 'fifo' '(' expr ')'
  | 'self'
  | 'spawn' '<' type_def '>' '(' expr ')'
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

## Roles and Clients

A program declares server roles with `role` and the operations a workload issues
with `client`. Each declaration takes exactly one parameter, the node's context.
The parameter is read-only and is in scope in the block's variable initializers
and in every one of its functions.

```
type Cluster {
    @quorum nodes: list<Node>;
};

role Node(cluster: Cluster) {
    var me: int = index_of(cluster.nodes, self)!;
    var replicas: list<Node> = cluster.nodes;

    fn Init() { ... }
    async fn RecoverInit() { ... }

    @trace
    async fn AppendEntries(...) { ... }
}
```

- `self` is a keyword. In `role R` it has type `R` and in `client C` it has type
  `C`: it is a handle to the node the code runs on. Outside a role or a client
  it is an error.
- A node's integer identity is an ordinary role variable. The convention is
  `var me: int = index_of(cluster.nodes, self)!;`.
- `Init` and `RecoverInit` take no parameters and return unit. Both are
  optional, and either may be `async`.
- Variable initializers run before `Init` at startup and again on recovery
  before `RecoverInit`. They may read the parameter and `self`.
- Assigning to the parameter is an error. `ctx.f := v` builds a new value and
  does not rebind `ctx`.
- The parameter type must be deployable (see Deployments). A role that needs no
  context declares `role Witness(unused: ())`.

A role parameter is supplied by a deploy function. A client's parameter is the
deployment root, so it always sees the whole deployment.

### Client operations

```
client KVClient(sys: Cluster) {
    async fn Write(dest: Node, key: string, uid: int) { ... }
    async fn Read(dest: Node, key: string): list<int> { ... }
    async fn RMW(dest: Node, key: string, uid: int): list<int> { ... }
}
```

- `Write` and `Read` are required, `RMW` is optional. All three are `async` and
  must have these parameter and return types.
- `dest` is optional per operation. When present it is the first parameter and
  its type is a role, never a client. Different operations may name different
  roles. An operation without `dest` routes itself from the client parameter.
- Other functions in a client are helpers. The simulator calls only these three.
- The linearizability model follows from the client: `kv_rmw` when `RMW` is
  declared, `kv` otherwise.

## Deployments

A deploy function builds the nodes of a run and the value each of them receives
as its parameter. It is a free function marked `@deploy(client = C)`, where `C`
names the client that issues operations against it.

```
type ClusterParams {
    @scale n: int;
};

fn cluster(n: int): Cluster {
    var nodes: list<Node> = spawn<Node>(n);
    var c: Cluster = Cluster { nodes: nodes };
    provide_all(nodes, c);
    c
}

@deploy(client = KVClient)
fn Main(p: ClusterParams): Cluster? {
    if (p.n < 1) {
        return nil;
    }
    cluster(p.n)
}
```

### Allocation builtins

| Builtin | Type | Meaning |
| --- | --- | --- |
| `spawn<R>(k)` | `int -> list<R>`, `R` a role | Allocates `k` fresh handles of role `R`. No node runs yet. |
| `provide(h, v)` | `(R, P) -> ()`, `P` the parameter type of `R` | Binds `v` as the parameter of node `h`. |
| `provide_all(hs, v)` | `(list<R>, P) -> ()` | `provide(h, v)` for each `h`, in list order. |
| `index_of(xs, x)` | `(list<T>, T) -> int?` | Position of the first element equal to `x`, or `nil`. |

`spawn` names a role, never a client: clients are created by the simulator, not
by a deploy. Every spawned handle must be provided exactly once by the time the
deploy returns; a missing or repeated `provide` stops the session.

These three allocation builtins do something only while a deploy function runs,
so an ordinary function that calls them is a library builder for deploys.
Calling one from role or client code raises a runtime error. `index_of` is an
ordinary builtin and may be called anywhere.

`spawn` calls take global node indices in call order: the first call gets
`0..k`, the next continues from there. That index is the node's identity in
every output table and in every payload. Nodes of one role are also numbered by
an ordinal within the role, used in reports and debug output as `Role[ordinal]`.

### `@deploy`

- The function is free and is not allowed to do anything that needs a running
  node: no `self`, no role variable, no RPC, no channel operation, no timer, no
  `fifo`, no `unique_id`, no persistence builtin. The restriction follows calls,
  so it also covers every function a deploy calls.
- It takes one parameter of struct type, or no parameter at all.
- It returns `T?`, where `T` is a deployable type that is not itself optional.
  Returning `nil` rejects the parameter tuple: the explorer skips it rather than
  treating it as an error. A deploy may coerce its parameters instead, for
  example rounding a replica count up to odd.
- `client = C` is required and names a `client` declaration whose parameter type
  is exactly `T`.
- A program may declare several deploys. Their names are distinct, and a config
  or the `--deploy` flag selects one.
- A deploy function may also be called as an ordinary function.

### Parameter structs

A deploy's parameter is a struct whose every field carries exactly one of
`@scale` or `@choice`. The explorer varies those fields; nothing else about a
deployment varies.

| Tag | Field type | Config supplies |
| --- | --- | --- |
| `@scale` | `int` | `{ "min", "max", "step" }`; the explorer visits small values first |
| `@choice` | `int`, `bool`, `string`, or an enum whose variants carry no payload | a JSON array of values |

### `@quorum`

`@quorum` marks a field of type `list<R>` as a group from which a majority is
drawn. Generated `majorities_ring` and `bridge` partitions prefer quorum groups.
It has no other effect.

### Deployable types

A type is deployable when it is built only from `int`, `string`, `bool`, `()`,
tuples, `list<T>`, `map<K, V>`, `T?`, structs and enums of deployable types, and
role handle types. Channels, FIFO links and client handle types are not
deployable, because a deployment exists before any node runs. Role parameter
types, deploy parameter structs and deploy root types must all be deployable.
Handles are plain values, so roles may refer to each other's types.

### Groups and paths

The simulator walks the returned root value, guided by its static type, and
records:

- a canonical path for every handle, such as `nodes[2]` or
  `shards["east"].nodes[0]`, written from the root; `$` names the root itself
- every value of type `list<R>` as a group, deduplicated by role and member
  sequence, with the first path canonical and later ones aliases; a group is a
  quorum group when any of its occurrences sits under a `@quorum` field

Walk order is struct fields in declaration order, list and tuple elements by
index, and map entries in ascending key order. `nil` contributes nothing, and
enum payloads are walked but are not addressable by a path. A spawned handle
that the root does not reach still runs and can still crash, but it has no path,
so configs cannot name it.

## Typing

Spur is a strongly and statically typed language.

The type system is composed of:

- Primitives: `int`, `string`, `bool`
- Tuples and the unit type: `()`, `(T)`, `(T, U)`, etc.
- Collections: `list<T>`, `map<K, V>`
- Concurrency: `chan<T>`
- Optional types: `T?`, which can be either `nil` or a value of type `T`
- Role handles: `R` for a role or client `R`, the type of `self` and of every value `spawn<R>` returns
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

### FIFO RPC links

By default, Spur's simulator makes no ordering guarantee between RPCs — two
RPCs from node A to node B can be delivered in any order. For protocols that
assume a TCP-like FIFO link (original VR, some Paxos variants), use an
explicit `FifoLink<T>`:

```
var link: FifoLink<Node> = fifo(peer);
var ch1: chan<Response> = link->Handler(args1);
var ch2: chan<Response> = link->Handler(args2);
```

RPCs sent through the same link are guaranteed to be delivered to the
receiver in send order. Direct `peer->Handler(...)` calls remain unordered.
Multiple `fifo(peer)` calls to the same peer create independent links with
no ordering relationship between them.

`FifoLink<T>` values live in a node's env and are lost on crash unless
explicitly persisted. The simulator owns the per-link sequence state, so:

- **Receiver crash**: messages already enqueued through a link are buffered
  across the crash and delivered in original send order once the receiver
  recovers (same as non-FIFO messages, just with ordering preserved).
- **Sender crash (link not persisted)**: the link value is lost (normal env
  cleanup). Messages sent before the crash retain their sequence tags and
  still drain in order. After recovery, a fresh `fifo(peer)` returns a new
  link with its own independent sequence — pre- and post-crash sends are
  not ordered.
- **Sender crash (link persisted)**: if the `FifoLink` was saved via
  `persist_data` and recovered with `retrieve_data`, the same link ID
  resumes with its existing sequence counter. Post-recovery sends are
  guaranteed to arrive after any pre-crash in-flight messages — cross-crash
  FIFO ordering is preserved.

FIFO ordering applies to the request direction only; responses come back
on per-RPC `chan<T>` and carry no cross-response ordering. Handlers at the
receiver still run concurrently — FIFO orders *delivery* (the order in
which handler tasks are spawned), not handler execution.

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
- `index_of: (list<T>, T) -> int?`
- `spawn<R>: int -> list<R>`, `provide: (R, P) -> ()`, `provide_all: (list<R>, P) -> ()` (deploy functions only)

## Syntactic Notes

- Variable declarations are illegal in for loop increments.
