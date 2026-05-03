use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let task = args.first().map(|s| s.as_str()).unwrap_or("");

    match task {
        "install-lsp" => install_lsp(),
        "formulog-codegen" => formulog_codegen(),
        _ => {
            eprintln!("Usage: cargo xtask <TASK>");
            eprintln!();
            eprintln!("Tasks:");
            eprintln!("  install-lsp        Build spur-lsp and copy it into editors/code/server/");
            eprintln!("  formulog-codegen   Run formulog.jar -c spur.flg and cmake-build the flg binary");
            std::process::exit(1);
        }
    }
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside workspace")
        .to_path_buf()
}

fn install_lsp() {
    let root = project_root();

    // Build spur-lsp in release mode.
    let status = Command::new("cargo")
        .args(["build", "--release", "-p", "spur-lsp"])
        .current_dir(&root)
        .status()
        .expect("failed to run cargo build");

    if !status.success() {
        eprintln!("cargo build failed");
        std::process::exit(1);
    }

    // Determine source and destination paths.
    let src = root.join("target").join("release").join("spur-lsp");
    let dest_dir = root.join("editors").join("code").join("server");
    let dest = dest_dir.join("spur-lsp");

    fs::create_dir_all(&dest_dir).expect("failed to create server directory");
    fs::copy(&src, &dest).expect("failed to copy spur-lsp binary");

    println!("Installed spur-lsp to {}", dest.display());
}

/// Run the Formulog codegen + cmake build out-of-tree, into
/// `spur-liquid/target/flg-codegen/`. The resulting `flg` binary lands at
/// `spur-liquid/target/flg-codegen/build/flg` and the path is printed at
/// the end so users can `export SPUR_FLG_BIN=...` and skip rebuilding it
/// from `spur-liquid/build.rs`.
fn formulog_codegen() {
    let root = project_root();
    let liquid_dir = root.join("spur-liquid");
    let codegen_dir = liquid_dir.join("target").join("flg-codegen");
    let _ = fs::remove_dir_all(&codegen_dir);
    fs::create_dir_all(&codegen_dir).expect("failed to create codegen dir");

    let formulog_jar = liquid_dir.join("formulog.jar");
    let flg_src = liquid_dir.join("spur.flg");

    println!("==> running formulog codegen into {}", codegen_dir.display());
    let status = Command::new("java")
        .arg("-jar")
        .arg(&formulog_jar)
        .arg("-c")
        .arg("--codegen-dir")
        .arg(&codegen_dir)
        .arg(&flg_src)
        .status()
        .expect("failed to spawn java");
    if !status.success() {
        eprintln!("formulog codegen failed (exit {:?})", status.code());
        std::process::exit(1);
    }

    println!("==> configuring cmake");
    let build_dir = codegen_dir.join("build");
    fs::create_dir_all(&build_dir).expect("failed to create build dir");
    let status = Command::new("cmake")
        .arg("-B")
        .arg(&build_dir)
        .arg("-S")
        .arg(&codegen_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .status()
        .expect("failed to spawn cmake");
    if !status.success() {
        eprintln!("cmake configure failed (exit {:?})", status.code());
        std::process::exit(1);
    }

    println!("==> building flg");
    let status = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("-j")
        .status()
        .expect("failed to spawn cmake --build");
    if !status.success() {
        eprintln!("cmake build failed (exit {:?})", status.code());
        std::process::exit(1);
    }

    let flg_bin = build_dir.join("flg");
    println!("==> built {}", flg_bin.display());
    println!();
    println!("To use this binary in cargo builds without rerunning codegen:");
    println!("  export SPUR_FLG_BIN={}", flg_bin.display());
}
