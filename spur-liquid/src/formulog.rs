//! Driver glue for the Formulog-generated `flg` binary.
//!
//! The pipeline at runtime is:
//!
//! 1. Encode the `CProgram` into TSV input facts via [`crate::flg`].
//! 2. Materialize the facts in a fresh temp directory.
//! 3. Spawn the `flg` binary with `--fact-dir <tempdir>
//!    --out-dir <tempdir>` and a wall-clock deadline.
//! 4. Read back `func_ok.tsv` / `func_failed.tsv` and translate them
//!    into [`RefinementCheckError`] diagnostics.
//!
//! The Formulog-generated binary writes one row per derived tuple. We
//! treat any function id in `fn_to_check` for which `func_ok` is not
//! derived (or `func_failed` *is* derived) as a verification failure
//! and surface it to higher layers.

use std::fs;
use std::io;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use spur_ast::name::NameId;
use thiserror::Error;

use crate::cache::{Cache, CacheKey};
use crate::flg::{EncodeError, EncodedFacts, encode_program};
use crate::ir::CProgram;

/// One refinement check failure reported back from Formulog.
///
/// Currently the only kind that can arise is a function-level failure
/// (`func_failed/2` derived, or `func_ok/1` not derived). The optional
/// `expr_id` will be populated once the .flg rules learn to thread
/// failing-expression ids through (`expr_origin` step 6.b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementCheckError {
    pub function: NameId,
    pub kind: RefinementCheckErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefinementCheckErrorKind {
    /// The Formulog driver could not derive `func_ok(F)` for this
    /// function, so its body did not type-check against its signature.
    /// `expr_id` is the synthetic id of the offending expression when
    /// available; once `expr_origin` is plumbed end-to-end it will be
    /// translatable back to a [`spur_ast::Span`].
    FunctionFailed { expr_id: Option<i32> },
}

#[derive(Debug, Error)]
pub enum FormulogError {
    #[error("could not locate the `flg` binary; build with `--features formulog` or set SPUR_FLG_BIN")]
    BinaryNotFound,
    #[error("could not spawn `flg`: {0}")]
    Spawn(#[from] io::Error),
    #[error("`flg` exited with non-zero status {0}")]
    NonZeroExit(i32),
    #[error("`flg` did not finish within {0:?}")]
    Timeout(Duration),
    #[error("could not encode the program for Formulog: {0}")]
    Encode(#[from] EncodeError),
    #[error("could not parse Formulog output: {0}")]
    BadOutput(String),
}

/// Run the refinement checker against `program` for the given set of
/// entry-point function ids. Returns one [`RefinementCheckError`] per
/// failed function (an empty vec means "all checks succeeded").
///
/// `flg_bin` should be the absolute path to the Formulog-compiled
/// driver; in normal Cargo builds [`crate::flg_binary_path`] returns
/// the right value.
///
/// `timeout` puts an upper bound on how long we wait for `flg` to
/// finish; on expiry we kill the child and return [`FormulogError::Timeout`].
pub fn check_with_formulog(
    program: &CProgram,
    fns_to_check: &[NameId],
    flg_bin: &Path,
    timeout: Duration,
) -> Result<Vec<RefinementCheckError>, FormulogError> {
    if fns_to_check.is_empty() {
        return Ok(Vec::new());
    }

    let facts = encode_program(program, fns_to_check)?;

    let key = CacheKey::from_facts(&facts, fns_to_check);
    if let Some(cached) = result_cache().get(&key) {
        return Ok(cached);
    }

    let result = run_flg(&facts, fns_to_check, flg_bin, timeout)?;
    result_cache().insert(key, result.clone());
    Ok(result)
}

/// Process-wide blake3-keyed cache. Lazy-initialized so test harnesses
/// that exercise [`check_with_formulog`] under different programs hit
/// the cache only when they should.
fn result_cache() -> &'static Cache<Vec<RefinementCheckError>> {
    static CACHE: OnceLock<Cache<Vec<RefinementCheckError>>> = OnceLock::new();
    CACHE.get_or_init(Cache::new)
}

fn run_flg(
    facts: &EncodedFacts,
    fns_to_check: &[NameId],
    flg_bin: &Path,
    timeout: Duration,
) -> Result<Vec<RefinementCheckError>, FormulogError> {
    let keep_tmp = std::env::var_os("SPUR_FORMULOG_KEEP_TMPDIR").is_some();

    let tempdir = tempfile::Builder::new()
        .prefix("spur-formulog-")
        .tempdir()?;
    let fact_dir = tempdir.path().to_path_buf();
    write_fact_files(&fact_dir, facts)?;

    let out_dir = tempdir.path().to_path_buf();

    if keep_tmp {
        eprintln!("[spur-formulog] keeping tempdir: {}", fact_dir.display());
    }

    let child = Command::new(flg_bin)
        .arg("--fact-dir")
        .arg(&fact_dir)
        .arg("--out-dir")
        .arg(&out_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let (exit_status, stdout, stderr) = wait_with_timeout(child, timeout)?;

    if keep_tmp {
        eprintln!("[spur-formulog] flg stdout:\n{}", stdout);
        eprintln!("[spur-formulog] flg stderr:\n{}", stderr);
    }

    if !exit_status.success() {
        return Err(FormulogError::NonZeroExit(exit_status.code().unwrap_or(-1)));
    }

    let result = parse_outputs(&out_dir, fns_to_check);

    if keep_tmp {
        let _ = tempdir.into_path();
    }

    result
}

/// Write every TSV blob in `facts` into `dir`. Empty relations still
/// get a 0-byte file because the `flg` runtime fails fast if any
/// declared input EDB has no backing file.
pub(crate) fn write_fact_files(dir: &Path, facts: &EncodedFacts) -> io::Result<()> {
    for (name, body) in facts.files() {
        let path = dir.join(name);
        fs::write(&path, body)?;
    }
    Ok(())
}

fn wait_with_timeout(
    mut child: Child,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, String, String), FormulogError> {
    let start = Instant::now();
    let poll = Duration::from_millis(25);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut o) = child.stdout.take() {
                    use std::io::Read as _;
                    let _ = o.read_to_string(&mut stdout);
                }
                if let Some(mut e) = child.stderr.take() {
                    use std::io::Read as _;
                    let _ = e.read_to_string(&mut stderr);
                }
                return Ok((status, stdout, stderr));
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(FormulogError::Timeout(timeout));
                }
                thread::sleep(poll);
            }
            Err(e) => return Err(FormulogError::Spawn(e)),
        }
    }
}

/// Parse `func_ok.tsv` and `func_failed.tsv` out of `out_dir` and
/// produce a list of [`RefinementCheckError`]s for every requested
/// function that did not pass.
fn parse_outputs(
    out_dir: &Path,
    fns_to_check: &[NameId],
) -> Result<Vec<RefinementCheckError>, FormulogError> {
    let ok = read_func_ok(out_dir)?;
    let failed = read_func_failed(out_dir)?;

    let mut errors = Vec::new();
    for f in fns_to_check {
        let id = f.0 as i32;
        if ok.contains(&id) {
            continue;
        }
        let expr_id = failed.iter().find(|(fid, _)| *fid == id).map(|(_, eid)| *eid);
        errors.push(RefinementCheckError {
            function: *f,
            kind: RefinementCheckErrorKind::FunctionFailed { expr_id },
        });
    }
    Ok(errors)
}

fn read_func_ok(dir: &Path) -> Result<Vec<i32>, FormulogError> {
    read_int_column(&dir.join("func_ok.tsv"))
}

fn read_func_failed(dir: &Path) -> Result<Vec<(i32, i32)>, FormulogError> {
    let path = dir.join("func_failed.tsv");
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(FormulogError::Spawn(e)),
    };
    let mut rows = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let f: i32 = parse_or_bad(cols.next(), &path, lineno)?;
        let e: i32 = parse_or_bad(cols.next(), &path, lineno)?;
        rows.push((f, e));
    }
    Ok(rows)
}

fn read_int_column(path: &Path) -> Result<Vec<i32>, FormulogError> {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(FormulogError::Spawn(e)),
    };
    let mut rows = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let first = line.split('\t').next();
        rows.push(parse_or_bad(first, path, lineno)?);
    }
    Ok(rows)
}

fn parse_or_bad(
    cell: Option<&str>,
    path: &Path,
    lineno: usize,
) -> Result<i32, FormulogError> {
    let s = cell.ok_or_else(|| {
        FormulogError::BadOutput(format!(
            "{}:{}: missing column",
            path.display(),
            lineno + 1
        ))
    })?;
    s.trim().parse::<i32>().map_err(|e| {
        FormulogError::BadOutput(format!(
            "{}:{}: {} (got `{}`)",
            path.display(),
            lineno + 1,
            e,
            s
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_func_ok_with_no_file_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let v = read_func_ok(dir.path()).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn parses_func_ok_one_per_line() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("func_ok.tsv"), "1\n2\n3\n").unwrap();
        let v = read_func_ok(dir.path()).unwrap();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn parses_func_failed_two_columns() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("func_failed.tsv"), "5\t-1\n7\t42\n").unwrap();
        let v = read_func_failed(dir.path()).unwrap();
        assert_eq!(v, vec![(5, -1), (7, 42)]);
    }

    #[test]
    fn parse_outputs_splits_pass_fail() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("func_ok.tsv"), "1\n").unwrap();
        fs::write(dir.path().join("func_failed.tsv"), "2\t-1\n").unwrap();
        let errs = parse_outputs(dir.path(), &[NameId(1), NameId(2), NameId(3)]).unwrap();
        // Function 1 is ok; functions 2 and 3 both lack a `func_ok`
        // tuple. Function 2 has an explicit `func_failed`, function 3
        // is implicit.
        assert_eq!(errs.len(), 2);
        assert_eq!(errs[0].function, NameId(2));
        assert!(matches!(
            errs[0].kind,
            RefinementCheckErrorKind::FunctionFailed { expr_id: Some(-1) }
        ));
        assert_eq!(errs[1].function, NameId(3));
        assert!(matches!(
            errs[1].kind,
            RefinementCheckErrorKind::FunctionFailed { expr_id: None }
        ));
    }
}
