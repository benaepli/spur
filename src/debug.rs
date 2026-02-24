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
}
