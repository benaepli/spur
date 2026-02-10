use anyhow::{Context, bail};
use include_dir::{Dir, include_dir};

// Embed the specs directory at compile time
static SPECS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/specs");

#[test]
fn test_all_spec_files_compile() -> anyhow::Result<()> {
    // Find all .spur files in the embedded directory
    let spec_files: Vec<_> = SPECS_DIR
        .files()
        .filter(|file| file.path().extension().and_then(|s| s.to_str()) == Some("spur"))
        .collect();

    // Ensure we found at least one spec file
    if spec_files.is_empty() {
        bail!("No .spur files found in specs/ directory");
    }

    // Test each spec file
    for file in spec_files {
        let file_name = file
            .path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let source = file
            .contents_utf8()
            .context(format!("Failed to read {} as UTF-8", file_name))?;

        spur::compiler::compile(source, file_name)
            .context(format!("Failed to compile {}", file_name))?;
    }

    Ok(())
}
