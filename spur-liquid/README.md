# spur-liquid

Refinement-type checking for spur. Hosts:

- The post-lowering "core" refinement IR (`CProgram`) and the
  `PProgram -> CProgram` lowering pass.
- A structural validator for refinement bodies (linearity, no user calls,
  etc.).
- (Behind the `formulog` Cargo feature) a Formulog-driven SMT-discharge
  pipeline. We ship `formulog.jar` plus a Datalog program (`spur.flg`)
  that implements bidirectional refinement type checking, and at build
  time we generate, compile, and link a native `flg` binary that the
  Rust crate drives over a fact-directory protocol.

## Building without Formulog (default)

```
cargo build -p spur-liquid
```

This produces the lowering and validator only and has no host-toolchain
dependencies beyond the standard Rust pipeline.

## Building with Formulog (refinement checking)

```
cargo build -p spur-liquid --features formulog
```

The build script invokes Formulog's compile mode (`java -jar
formulog.jar -c spur.flg --codegen-dir <OUT_DIR>/codegen`), then runs
`cmake` against the generated CMake project to produce the `flg`
binary. The path is published to the crate via
`cargo:rustc-env=SPUR_FLG_BIN=...`.

### Host dependencies

- Java (any modern JDK; tested on OpenJDK 21).
- CMake 3.21 or newer.
- A C++17 compiler (GCC or clang) with OpenMP support.
- Boost ~1.81 (`boost-devel` on Fedora, `libboost-all-dev` on Debian).
- oneTBB ~2021.x (`tbb-devel`, `libtbb-dev`).
- Souffle 2.3+ (`souffle`).

### Faster local iteration

The Formulog codegen + CMake build takes ~90 seconds from scratch. To
avoid paying that on every `cargo build`, build the binary once via the
xtask helper and point at it:

```
cargo xtask formulog-codegen
export SPUR_FLG_BIN="$(pwd)/spur-liquid/target/flg-codegen/build/flg"
```

The build script honours `SPUR_FLG_BIN` and skips the codegen + cmake
steps when it is set.

## Driving from the CLI

When the spur CLI is built with the same feature flag, the
`--refinements` switch on `spur check` and `spur compile` will run the
SMT-backed checker after the regular type-checking pipeline:

```
cargo build --release --features formulog -p spur-cli
./target/release/spur check my.spur --refinements
```

`--refinements-timeout <secs>` (default 60) puts an upper bound on the
SMT solver's wall-clock budget.

## LSP integration

The language server reads `spur.refinements.onSave` from
`workspace/configuration` (or from `initializationOptions`). When set
to `true` and the LSP itself was built with `--features formulog`, the
debounced analysis loop runs `compile_with_refinements` instead of
`compile_lsp` so refinement diagnostics show up alongside the regular
ones.
