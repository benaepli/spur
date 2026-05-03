use anyhow::{Context, bail};
use include_dir::{Dir, include_dir};

// Embed the specs directory at compile time
static SPECS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/specs");

/// Compiling real specs through `compile -> liquid_lower_program`
/// recursively walks deeply-nested types (e.g. refined list-of-list
/// invariants), which overflows the default 2 MiB test-thread stack.
/// We bump it to 32 MiB so the integration suite mirrors what the CLI
/// gets when run with `RUST_MIN_STACK` unset.
fn run_test() -> anyhow::Result<()> {
    let spec_files: Vec<_> = SPECS_DIR
        .files()
        .filter(|file| file.path().extension().and_then(|s| s.to_str()) == Some("spur"))
        .collect();

    if spec_files.is_empty() {
        bail!("No .spur files found in specs/ directory");
    }

    for file in spec_files {
        let file_name = file
            .path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let source = file
            .contents_utf8()
            .context(format!("Failed to read {} as UTF-8", file_name))?;

        spur_core::compiler::compile(source, file_name)
            .into_program()
            .context(format!("Failed to compile {}", file_name))?;
    }

    Ok(())
}

#[test]
fn test_all_spec_files_compile() -> anyhow::Result<()> {
    let join = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(run_test)
        .expect("failed to spawn worker thread");
    join.join().expect("worker thread panicked")
}
