use duckdb::{Connection, Result, params};
use std::path::Path;

pub struct SimulatorDebugger {
    conn: Connection,
}

impl SimulatorDebugger {
    /// Connects to an existing simulation database.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        Ok(Self { conn })
    }

    /// Fetches all logs for a specific node, ordered by simulation step.
    pub fn get_node_timeline(&self, run_id: i64, node_id: i64) -> Result<Vec<(i32, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT step, content FROM logs
             WHERE run_id = ?1 AND node_id = ?2
             ORDER BY step ASC",
        )?;

        let rows = stmt.query_map(params![run_id, node_id], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        rows.collect()
    }

    /// Fetches all logs for a specific run, ordered by simulation step.
    pub fn get_all_logs(&self, run_id: i64) -> Result<Vec<(i32, Option<i64>, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT step, node_id, content FROM logs
             WHERE run_id = ?1
             ORDER BY step ASC",
        )?;

        let rows = stmt.query_map(params![run_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect()
    }

    /// Returns a summary of a run (count of invocations, crashes, etc.)
    pub fn get_run_summary(&self, run_id: i64) -> Result<std::collections::HashMap<String, i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT kind, COUNT(*) FROM executions WHERE run_id = ?1 GROUP BY kind")?;

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
