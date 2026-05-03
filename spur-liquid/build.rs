//! Build script for spur-liquid.
//!
//! When the `formulog` feature is enabled this script:
//!   1. invokes `java -jar formulog.jar -c spur.flg --codegen-dir <OUT>`
//!      to produce a CMake project in `$OUT_DIR/codegen/`,
//!   2. drives `cmake` to build the resulting C++ project, producing the
//!      `flg` binary at `$OUT_DIR/codegen/build/flg`,
//!   3. exposes the binary path to the crate via the
//!      `cargo:rustc-env=SPUR_FLG_BIN=...` declaration so call sites can
//!      use `option_env!("SPUR_FLG_BIN")` at runtime.
//!
//! Setting the `SPUR_FLG_BIN` environment variable before running cargo
//! short-circuits the codegen + cmake build and uses the supplied binary
//! verbatim. This is the path most developers want to take while
//! iterating on `spur.flg` (see `cargo xtask formulog-codegen`).
//!
//! When the `formulog` feature is *not* enabled the script does nothing
//! beyond wiring up rerun-if-changed dependencies, so a default
//! `cargo build` does not need any of the heavyweight system toolchain.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=spur.flg");
    println!("cargo:rerun-if-changed=formulog.jar");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SPUR_FLG_BIN");

    if env::var_os("CARGO_FEATURE_FORMULOG").is_none() {
        return;
    }

    if let Some(prebuilt) = env::var_os("SPUR_FLG_BIN") {
        let path = PathBuf::from(prebuilt);
        println!("cargo:rustc-env=SPUR_FLG_BIN={}", path.display());
        return;
    }

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let codegen_dir = out_dir.join("codegen");

    let formulog_jar = crate_dir.join("formulog.jar");
    let flg_src = crate_dir.join("spur.flg");

    if !formulog_jar.exists() {
        panic!(
            "formulog.jar not found at {}; expected it next to spur-liquid/Cargo.toml",
            formulog_jar.display()
        );
    }
    if !flg_src.exists() {
        panic!(
            "spur.flg not found at {}; expected it next to spur-liquid/Cargo.toml",
            flg_src.display()
        );
    }

    let _ = std::fs::remove_dir_all(&codegen_dir);

    let codegen_status = Command::new("java")
        .arg("-jar")
        .arg(&formulog_jar)
        .arg("-c")
        .arg("--codegen-dir")
        .arg(&codegen_dir)
        .arg(&flg_src)
        .status()
        .expect("failed to spawn `java` (is the JDK on PATH?)");
    if !codegen_status.success() {
        panic!(
            "formulog codegen failed (exit {:?}). Run\n  java -jar {} -c --codegen-dir {} {}\nto reproduce.",
            codegen_status.code(),
            formulog_jar.display(),
            codegen_dir.display(),
            flg_src.display()
        );
    }

    if !codegen_dir.join("CMakeLists.txt").exists() {
        panic!(
            "formulog codegen ran but produced no CMakeLists.txt at {}",
            codegen_dir.display()
        );
    }

    // `cmake::Config::build` returns the install prefix, but we want the
    // raw build directory so we can pick up `flg` from `build/flg`. We
    // disable cmake's default `--install` step by setting the install
    // target to the build directory itself; the `cmake` crate runs
    // `cmake --build <build> --target install` by default.
    let dst = cmake::Config::new(&codegen_dir)
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("flg")
        .very_verbose(false)
        .build();

    let flg_bin = dst.join("build").join("flg");
    if !flg_bin.exists() {
        panic!(
            "expected flg binary at {} after cmake build, but it is missing",
            flg_bin.display()
        );
    }

    println!("cargo:rustc-env=SPUR_FLG_BIN={}", flg_bin.display());
}
