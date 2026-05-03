//! Smoke test: when the `formulog` feature is on, the build script must
//! have produced a runnable `flg` binary, and `flg --help` must exit
//! cleanly.

#![cfg(feature = "formulog")]

use std::process::Command;

#[test]
fn flg_binary_exists_and_runs_help() {
    let bin = spur_liquid::flg_binary_path()
        .expect("SPUR_FLG_BIN was not set by build.rs (or via env override)");
    assert!(
        bin.exists(),
        "flg binary should exist at {}",
        bin.display()
    );

    let output = Command::new(&bin)
        .arg("--help")
        .output()
        .expect("failed to spawn flg --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stdout.contains("--help") || stderr.contains("--help"),
        "flg --help did not mention --help; stdout=\n{}\nstderr=\n{}",
        stdout,
        stderr
    );
}
