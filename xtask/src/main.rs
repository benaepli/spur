use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let task = args.first().map(|s| s.as_str()).unwrap_or("");

    match task {
        "install-lsp" => install_lsp(),
        _ => {
            eprintln!("Usage: cargo xtask <TASK>");
            eprintln!();
            eprintln!("Tasks:");
            eprintln!("  install-lsp    Build spur-lsp and copy it into editors/code/server/");
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
