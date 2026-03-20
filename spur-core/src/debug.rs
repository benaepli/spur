use duckdb::{Connection, Result, params};
use std::path::Path;

/// Opens a DuckDB connection for the given path.
///
/// - If `path` is a `.duckdb` file, opens it directly.
/// - If `path` is a directory (Parquet backend), opens an in-memory DuckDB
///   and uses DuckDB's native `read_parquet()` to query the `.parquet` files
///   inside the directory. This is transparent to all query methods.
fn open_connection(path: &Path) -> Result<Connection> {
    if path.extension().and_then(|e| e.to_str()) == Some("duckdb") {
        Connection::open(path)
    } else {
        // Parquet directory: open in-memory DuckDB
        Connection::open_in_memory()
    }
}

pub struct SimulatorDebugger {
    conn: Connection,
    /// None for DuckDB mode; Some(dir) for Parquet mode.
    parquet_dir: Option<std::path::PathBuf>,
}

impl SimulatorDebugger {
    /// Connects to an existing simulation database or Parquet directory.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let conn = open_connection(path)?;
        let parquet_dir = if path.is_dir() {
            Some(path.to_path_buf())
        } else {
            None
        };
        Ok(Self { conn, parquet_dir })
    }

    /// Returns the SQL table expression to use for `logs`.
    fn logs_source(&self) -> String {
        match &self.parquet_dir {
            None => "logs".to_string(),
            Some(dir) => {
                let p = dir.join("logs").join("*.parquet");
                format!("read_parquet('{}', union_by_name=true)", p.display())
            }
        }
    }

    /// Returns the SQL table expression to use for `executions`.
    fn executions_source(&self) -> String {
        match &self.parquet_dir {
            None => "executions".to_string(),
            Some(dir) => {
                let p = dir.join("executions").join("*.parquet");
                format!("read_parquet('{}', union_by_name=true)", p.display())
            }
        }
    }

    /// Returns the SQL table expression to use for `traces`.
    fn traces_source(&self) -> String {
        match &self.parquet_dir {
            None => "traces".to_string(),
            Some(dir) => {
                let p = dir.join("traces").join("*.parquet");
                format!("read_parquet('{}', union_by_name=true)", p.display())
            }
        }
    }

    /// Fetches all logs for a specific node, ordered by simulation step.
    pub fn get_node_timeline(&self, run_id: i64, node_id: i64) -> Result<Vec<(i32, String)>> {
        let src = self.logs_source();
        let query = format!(
            "SELECT step, content FROM {src}
             WHERE run_id = ?1 AND node_id = ?2
             ORDER BY step ASC"
        );
        let mut stmt = self.conn.prepare(&query)?;

        let rows = stmt.query_map(params![run_id, node_id], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        rows.collect()
    }

    /// Fetches all logs for a specific run, ordered by simulation step.
    pub fn get_all_logs(&self, run_id: i64) -> Result<Vec<(i32, Option<i64>, String)>> {
        let src = self.logs_source();
        let query = format!(
            "SELECT step, node_id, content FROM {src}
             WHERE run_id = ?1
             ORDER BY step ASC"
        );
        let mut stmt = self.conn.prepare(&query)?;

        let rows = stmt.query_map(params![run_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect()
    }

    /// Returns a summary of a run (count of invocations, crashes, etc.)
    pub fn get_run_summary(&self, run_id: i64) -> Result<std::collections::HashMap<String, i64>> {
        let src = self.executions_source();
        let query = format!("SELECT kind, COUNT(*) FROM {src} WHERE run_id = ?1 GROUP BY kind");
        let mut stmt = self.conn.prepare(&query)?;

        let rows = stmt.query_map(params![run_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;

        let mut summary = std::collections::HashMap::new();
        for row in rows {
            let (kind, count) = row?;
            summary.insert(kind, count);
        }
        Ok(summary)
    }

    /// Fetches a combined, interleaved timeline of executions, logs, and traces
    /// for a given run, ordered by simulation step.
    pub fn get_combined_timeline(&self, run_id: i64) -> Result<Vec<CombinedEvent>> {
        let exec_src = self.executions_source();
        let logs_src = self.logs_source();
        let traces_src = self.traces_source();

        let query = format!(
            "SELECT step, source, node_id, description FROM (
                SELECT step, 'Execution' AS source, client_id AS node_id, seq_num,
                       kind || ': ' || action AS description
                FROM {exec_src} WHERE run_id = ?1
              UNION ALL
                SELECT step, 'Log' AS source, node_id, seq_num,
                       content AS description
                FROM {logs_src} WHERE run_id = ?1
              UNION ALL
                SELECT step, 'Trace' AS source, node_id, seq_num,
                       trace_kind || ' ' || function_name 
                         || ' [tid=' || trace_id || ']'
                         || ' [sched=' || schedulable_count || ']'
                         || CASE WHEN causal_operation_id IS NOT NULL THEN ' [cop=' || causal_operation_id || ']' ELSE '' END
                       AS description
                FROM {traces_src} WHERE run_id = ?1
            ) ORDER BY step ASC, source ASC, seq_num ASC"
        );

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(CombinedEvent {
                step: row.get(0)?,
                source: row.get(1)?,
                node_id: row.get(2)?,
                description: row.get(3)?,
            })
        })?;
        rows.collect()
    }
}

pub struct CombinedEvent {
    pub step: i32,
    pub source: String,
    pub node_id: Option<i64>,
    pub description: String,
}

pub struct TraceEvent {
    pub step: i32,
    pub node_id: i64,
    pub trace_id: i64,
    pub function_name: String,
    pub trace_kind: String,
    pub payload: String,
    pub schedulable_count: i64,
    pub causal_operation_id: Option<i64>,
}

impl SimulatorDebugger {
    /// Fetches all traces for a specific run, and optionally a specific node.
    pub fn get_traces(&self, run_id: i64, node_id: Option<i64>) -> Result<Vec<TraceEvent>> {
        let src = self.traces_source();

        if let Some(n_id) = node_id {
            let q = format!(
                "SELECT step, node_id, trace_id, function_name, trace_kind, payload, schedulable_count, causal_operation_id
                 FROM {src}
                 WHERE run_id = ?1 AND node_id = ?2
                 ORDER BY step ASC, seq_num ASC"
            );
            let mut stmt = self.conn.prepare(&q)?;
            let rows = stmt.query_map(params![run_id, n_id], |row| {
                Ok(TraceEvent {
                    step: row.get(0)?,
                    node_id: row.get(1)?,
                    trace_id: row.get(2)?,
                    function_name: row.get(3)?,
                    trace_kind: row.get(4)?,
                    payload: row.get(5)?,
                    schedulable_count: row.get(6)?,
                    causal_operation_id: row.get(7)?,
                })
            })?;
            rows.collect()
        } else {
            let q = format!(
                "SELECT step, node_id, trace_id, function_name, trace_kind, payload, schedulable_count, causal_operation_id
                 FROM {src}
                 WHERE run_id = ?1
                 ORDER BY step ASC, seq_num ASC"
            );
            let mut stmt = self.conn.prepare(&q)?;
            let rows = stmt.query_map(params![run_id], |row| {
                Ok(TraceEvent {
                    step: row.get(0)?,
                    node_id: row.get(1)?,
                    trace_id: row.get(2)?,
                    function_name: row.get(3)?,
                    trace_kind: row.get(4)?,
                    payload: row.get(5)?,
                    schedulable_count: row.get(6)?,
                    causal_operation_id: row.get(7)?,
                })
            })?;
            rows.collect()
        }
    }
}
