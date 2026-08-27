//! Field-level overrides applied to an explorer/plan config after it is read
//! from disk and before it is parsed.
//!
//! A runner loads exactly the config file it was pointed at, so varying a
//! single scalar knob across runs otherwise requires one sibling config file
//! per value - and a file no runner loads has no effect on anything. An
//! override expresses the same variation against the loaded file:
//!
//! ```text
//! SPUR_CONFIG_SET='purgatory.hold_range.max=300' spur explore ...
//! spur explore ... --set num_crashes_range.max=3 --set stats=true
//! ```
//!
//! The left side is a dot-separated path into the config object; missing
//! intermediate objects are created, so a knob that relies on its serde
//! default and is absent from the file can still be set. The right side is
//! parsed as JSON when it parses and taken as a string otherwise, so
//! `true`, `12`, `[1,2]` and `parquet` all mean what they look like. Overrides
//! are applied before strict key checking, so a misspelled top-level knob is
//! rejected by the same rule that rejects it in a file.

use serde_json::{Map, Value};
use std::sync::Mutex;

/// Environment variable holding `;`- or newline-separated `path=value`
/// assignments.
pub const OVERRIDE_ENV: &str = "SPUR_CONFIG_SET";

static EXTRA_OVERRIDES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Registers assignments supplied outside the environment (a command-line
/// flag). Applied after the environment ones, so a flag wins a conflict.
pub fn set_extra_overrides(assignments: Vec<String>) {
    let mut slot = EXTRA_OVERRIDES.lock().expect("override list poisoned");
    *slot = assignments;
}

/// Every assignment that will be applied, environment first.
pub fn active_overrides() -> Vec<String> {
    let mut all = match std::env::var(OVERRIDE_ENV) {
        Ok(raw) => split_env(&raw),
        Err(_) => Vec::new(),
    };
    all.extend(
        EXTRA_OVERRIDES
            .lock()
            .expect("override list poisoned")
            .iter()
            .cloned(),
    );
    all
}

fn split_env(raw: &str) -> Vec<String> {
    raw.split([';', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Reads a config file and applies the active overrides to its text.
pub fn load_config_text(path: &str) -> Result<String, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read config {path}: {e}"))?;
    let assignments = active_overrides();
    if assignments.is_empty() {
        return Ok(text);
    }
    log::info!("config overrides: {}", assignments.join(", "));
    apply_assignments(&text, &assignments)
}

/// Applies `path=value` assignments to a JSON object, returning the new text.
pub fn apply_assignments(config_text: &str, assignments: &[String]) -> Result<String, String> {
    let mut root: Value = serde_json::from_str(config_text)
        .map_err(|e| format!("config is not valid JSON: {}", e))?;
    if !root.is_object() {
        return Err("config must be a JSON object".to_string());
    }
    for assignment in assignments {
        let (path, value) = parse_assignment(assignment)?;
        set_path(&mut root, &path, value)
            .map_err(|e| format!("override `{}`: {}", assignment, e))?;
    }
    serde_json::to_string_pretty(&root).map_err(|e| format!("failed to re-encode config: {}", e))
}

fn parse_assignment(assignment: &str) -> Result<(Vec<String>, Value), String> {
    let (raw_path, raw_value) = assignment
        .split_once('=')
        .ok_or_else(|| format!("override `{}` is not of the form path=value", assignment))?;
    let path: Vec<String> = raw_path.trim().split('.').map(str::to_string).collect();
    if path.iter().any(|seg| seg.is_empty()) {
        return Err(format!("override `{}` has an empty path segment", assignment));
    }
    let raw_value = raw_value.trim();
    let value = serde_json::from_str::<Value>(raw_value)
        .unwrap_or_else(|_| Value::String(raw_value.to_string()));
    Ok((path, value))
}

fn set_path(root: &mut Value, path: &[String], value: Value) -> Result<(), String> {
    let mut cursor: &mut Map<String, Value> = root
        .as_object_mut()
        .ok_or_else(|| "config root is not an object".to_string())?;
    let (last, parents) = path.split_last().expect("path has at least one segment");
    for segment in parents {
        let child = cursor
            .entry(segment.clone())
            .or_insert_with(|| Value::Object(Map::new()));
        cursor = child
            .as_object_mut()
            .ok_or_else(|| format!("`{}` is not an object", segment))?;
    }
    cursor.insert(last.clone(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(text: &str, assignments: &[&str]) -> Value {
        let owned: Vec<String> = assignments.iter().map(|s| s.to_string()).collect();
        let out = apply_assignments(text, &owned).expect("applies");
        serde_json::from_str(&out).expect("valid JSON")
    }

    #[test]
    fn sets_existing_scalar() {
        let v = apply(r#"{"num_runs": 10}"#, &["num_runs=250"]);
        assert_eq!(v["num_runs"], Value::from(250));
    }

    #[test]
    fn sets_nested_and_creates_missing_parents() {
        let v = apply(r#"{"a": {"b": 1}}"#, &["a.b=2", "c.d.e=true"]);
        assert_eq!(v["a"]["b"], Value::from(2));
        assert_eq!(v["c"]["d"]["e"], Value::Bool(true));
    }

    #[test]
    fn unquoted_value_is_a_string() {
        let v = apply(r#"{}"#, &["queue_policy=fifo"]);
        assert_eq!(v["queue_policy"], Value::String("fifo".to_string()));
    }

    #[test]
    fn json_values_keep_their_type() {
        let v = apply(r#"{}"#, &["r=[1,2]", "s={\"k\":1}", "t=1.5"]);
        assert!(v["r"].is_array());
        assert_eq!(v["s"]["k"], Value::from(1));
        assert_eq!(v["t"], Value::from(1.5));
    }

    #[test]
    fn later_assignment_wins() {
        let v = apply(r#"{"n": 1}"#, &["n=2", "n=3"]);
        assert_eq!(v["n"], Value::from(3));
    }

    #[test]
    fn rejects_malformed_assignment() {
        assert!(apply_assignments(r#"{}"#, &["no_equals_sign".to_string()]).is_err());
        assert!(apply_assignments(r#"{}"#, &["a..b=1".to_string()]).is_err());
    }

    #[test]
    fn rejects_descent_through_a_scalar() {
        assert!(apply_assignments(r#"{"a": 1}"#, &["a.b=2".to_string()]).is_err());
    }

    #[test]
    fn loads_a_file_and_applies_overrides_into_a_usable_config() {
        use crate::simulator::explorer::{
            EXPLORER_CONFIG_KEYS, ExplorerConfig, check_top_level_keys,
        };

        let path = std::env::temp_dir().join("spur_config_override_load_test.json");
        std::fs::write(
            &path,
            r#"{
                 "num_servers": {"min": 3, "max": 3, "step": 1},
                 "num_crashes": {"min": 1, "max": 3, "step": 1},
                 "purgatory": {"delay_probability": 0.15},
                 "num_runs_per_config": 100,
                 "strict_config_keys": true
               }"#,
        )
        .expect("writes fixture");

        set_extra_overrides(vec![
            "num_runs_per_config=3".to_string(),
            "purgatory.delay_probability=0.4".to_string(),
            r#"num_write_ops={"min":2,"max":4,"step":1}"#.to_string(),
            r#"num_read_ops={"min":4,"max":8,"step":2}"#.to_string(),
            r#"num_keys={"min":1,"max":1,"step":1}"#.to_string(),
            "dependency_density=[0.0,0.3]".to_string(),
            "max_iterations=6000".to_string(),
        ]);
        let text = load_config_text(path.to_str().expect("utf-8 path")).expect("loads");
        set_extra_overrides(Vec::new());

        check_top_level_keys(&text, &[EXPLORER_CONFIG_KEYS]).expect("no unknown keys");
        let config: ExplorerConfig = serde_json::from_str(&text).expect("parses as a config");
        assert_eq!(config.num_runs_per_config, 3);
        assert!((config.purgatory.delay_probability - 0.4).abs() < 1e-9);
        assert_eq!(config.num_crashes_range.max, 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn env_splitting_ignores_blanks() {
        assert_eq!(
            split_env(" a=1 ; b=2 \n\n c=3 ;"),
            vec!["a=1".to_string(), "b=2".to_string(), "c=3".to_string()]
        );
    }
}
