# Spur

Spur is a domain-specific language for specifying distributed protocols.

This repository provides a compiler frontend in Rust that is used in [Turnpike](https://github.com/benaepli/turnpike/)
for consistency (linearizability, etc.) refutation.

## Features

- Role-based architecture: define distributed nodes with persistent state and message handlers
- First-class RPCs: native support for both asynchronous and synchronous RPCs
- Static, strong type system: comprehensive type checking with primitives, structs, and optional types

The language design is detailed in [language.md](design/language.md). On the compiler side, we most importantly achieve:

- Compilation to a CFG (control-flow graph, see [cfg.rs](src/compiler/cfg.rs))
- Helpful error messages throughout all compilation phases

## Examples

See the [specs](specs) directory. For brevity, we provide a short example here:

```
type Command = int;

role Node {
  var db: map<string, Command> = {};

  fn Write(key: string, value: Command): bool {
      db = db[key] := value;
      return true;
  }
}

ClientInterface {
  fn Write(node: Node, key: string, value: Command) {
      <-node->Write(key, value);
  }
}
```

## Usage

### Building

This project uses Rust and cargo. To build the compiler:

```bash
cargo build --release
```

### Compiling Specifications

The compiler takes a Spur specification file and writes a directory containing
`program.json` (CFG IR), `cfg.svg` (CFG visualization), and `program.pir` (Pure / SSA IR).

**Example:**

```bash
# Compile the simple key-value store
cargo run -- compile specs/simple.spur --output-dir output -y
```

## Resources

- There is a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=baepli.spur) for syntax highlighting.
