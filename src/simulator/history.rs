use crate::simulator::core::{ChannelId, LogEntry, OpKind, Operation, Value};
use log::error;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use serde_json::{Value as JsonValue, json};
use std::error::Error;
use std::path::Path;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

/// A pre-serialized operation ready for database insertion.
/// JSON serialization is done by worker threads before sending to the writer.
pub struct PersistableOp {
    pub unique_id: i64,
    pub client_id: i64,
    pub kind: &'static str,
    pub action: String,
    pub payload_json: String,
}

pub struct PersistableLog {
    pub node_id: i64,
    pub content: String,
    pub step: i32,
}

pub fn serialize_logs(logs: &[LogEntry]) -> Vec<PersistableLog> {
    logs.par_iter()
        .map(|l| PersistableLog {
            node_id: l.node as i64,
            content: l.content.clone(),
            step: l.step,
        })
        .collect()
}

fn json_of_value(v: &Value) -> JsonValue {
    match v {
        Value::Int(i) => json!({
        "type": "VInt",
        "value": i
        }),
        Value::Bool(b) => json!({
        "type": "VBool",
        "value": b
        }),
        Value::String(s) => json!({
        "type": "VString",
        "value": s
        }),
        Value::Node(n) => json!({
        "type": "VNode",
        "value": n
        }),
        Value::Channel(ChannelId { node, id }) => json!({
        "type": "VChannel",
        "value": { "node": node, "id": id }
        }),
        Value::Map(m) => {
            let json_pairs: Vec<JsonValue> = m
                .iter()
                .map(|(k, v)| json!([json_of_value(k), json_of_value(v)]))
                .collect();
            json!({
            "type": "VMap",
            "value": json_pairs
            })
        }
        Value::Option(opt) => {
            let value_json = match opt {
                Some(inner) => json_of_value(inner),
                None => JsonValue::Null, //
            };
            json!({
            "type": "VOption",
            "value": value_json
            })
        }
        Value::List(l) => {
            let items: Vec<JsonValue> = l.iter().map(json_of_value).collect();
            json!({
            "type": "VList",
            "value": items
            })
        }
        Value::Unit => json!({
        "type": "VUnit",
        "value": null
        }),
        Value::Tuple(t) => {
            let items: Vec<JsonValue> = t.iter().map(json_of_value).collect();
            json!({
            "type": "VTuple",
            "value": items
            })
        }
    }
}

fn payload_to_json_string(payload: &[Value]) -> String {
    let json_list: Vec<JsonValue> = payload.iter().map(json_of_value).collect();
    serde_json::to_string(&json_list).unwrap_or_else(|_| "[]".to_string())
}

/// Serializes a list of Operations into PersistableOps.
/// This should be called from worker threads to distribute CPU work.
pub fn serialize_history(history: &[Operation]) -> Vec<PersistableOp> {
    history
        .par_iter()
        .map(|op| PersistableOp {
            unique_id: op.unique_id as i64,
            client_id: op.client_id as i64,
            kind: match op.kind {
                OpKind::Response => "Response",
                OpKind::Invocation => "Invocation",
                OpKind::Crash => "Crash",
                OpKind::Recover => "Recover",
            },
            action: op.op_action.clone(),
            payload_json: payload_to_json_string(&op.payload),
        })
        .collect()
}

/// Saves the simulation history to a CSV file.
pub fn save_history_to_csv<P: AsRef<Path>>(
    history: &[Operation],
    filename: P,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(filename)?;

    wtr.write_record(&["UniqueID", "ClientID", "Kind", "Action", "Payload"])?;

    for op in history {
        let kind = match op.kind {
            OpKind::Response => "Response",
            OpKind::Invocation => "Invocation",
            OpKind::Crash => "Crash",
            OpKind::Recover => "Recover",
        };

        let payload_str = payload_to_json_string(&op.payload);
        wtr.write_record(&[
            op.unique_id.to_string(),
            op.client_id.to_string(),
            kind.to_string(),
            op.op_action.clone(),
            payload_str,
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Initialize the SQLite database with the required tables.
pub fn init_sqlite(conn: &Connection) -> Result<(), rusqlite::Error> {
    // Performance: less durability, but should be OK
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;",
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs (
run_id INTEGER PRIMARY KEY,
start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
meta_info TEXT
);",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS executions (
run_id INTEGER REFERENCES runs(run_id),
seq_num INTEGER,
unique_id INTEGER,
client_id INTEGER,
kind TEXT,
action TEXT,
payload JSON,
PRIMARY KEY (run_id, seq_num)
);",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_run_execution ON executions(run_id, seq_num);",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS logs (
            run_id INTEGER REFERENCES runs(run_id),
            seq_num INTEGER,
            node_id INTEGER,
            step INTEGER,
            content TEXT,
            PRIMARY KEY (run_id, seq_num)
        );",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_run_logs ON logs(run_id);",
        [],
    )?;

    Ok(())
}

/// Saves pre-serialized history to the SQLite database.
/// This is a fast I/O-only operation since serialization is already done.
fn save_history_sqlite(
    conn: &mut Connection,
    run_id: i64,
    history: &[PersistableOp],
) -> Result<(), Box<dyn Error>> {
    conn.execute("INSERT INTO runs (run_id) VALUES (?1)", params![run_id])?;
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO executions (run_id, seq_num, unique_id, client_id, kind, action, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;

        for (seq_num, op) in history.iter().enumerate() {
            stmt.execute(params![
                run_id,
                seq_num as i64,
                op.unique_id,
                op.client_id,
                op.kind,
                op.action,
                op.payload_json
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn save_logs_sqlite(
    conn: &mut Connection,
    run_id: i64,
    logs: &[PersistableLog],
) -> Result<(), Box<dyn Error>> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO logs (run_id, seq_num, node_id, step, content)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;

        for (seq_num, log) in logs.iter().enumerate() {
            stmt.execute(params![
                run_id,
                seq_num as i64,
                log.node_id,
                log.step,
                &log.content
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Command sent to the background history writer thread.
pub enum HistoryCommand {
    Write {
        run_id: i64,
        history: Vec<PersistableOp>,
        logs: Vec<PersistableLog>,
    },
    Shutdown,
}

/// A background writer that serializes all SQLite writes to a single thread.
pub struct HistoryWriter {
    sender: Sender<HistoryCommand>,
    handle: Option<JoinHandle<()>>,
}

impl HistoryWriter {
    /// Creates a new HistoryWriter, opening and initializing the SQLite database.
    /// The connection is then moved to a background thread that processes write requests.
    pub fn new(db_path: &str) -> Result<Self, Box<dyn Error>> {
        let mut conn = Connection::open(db_path)?;
        init_sqlite(&conn)?;

        let (sender, receiver) = mpsc::channel::<HistoryCommand>();

        let handle = thread::spawn(move || {
            while let Ok(cmd) = receiver.recv() {
                match cmd {
                    HistoryCommand::Write {
                        run_id,
                        history,
                        logs,
                    } => {
                        if let Err(e) = save_history_sqlite(&mut conn, run_id, &history) {
                            error!("failed to save history for run {}: {}", run_id, e);
                        }
                        if let Err(e) = save_logs_sqlite(&mut conn, run_id, &logs) {
                            error!("failed to save logs for run {}: {}", run_id, e);
                        }
                    }
                    HistoryCommand::Shutdown => break,
                }
            }
        });

        Ok(Self {
            sender,
            handle: Some(handle),
        })
    }

    /// Sends a pre-serialized history and logs write request to the background thread.
    pub fn write(&self, run_id: i64, history: Vec<PersistableOp>, logs: Vec<PersistableLog>) {
        let _ = self.sender.send(HistoryCommand::Write {
            run_id,
            history,
            logs,
        });
    }

    /// Shuts down the background writer, waiting for all pending writes to complete.
    pub fn shutdown(mut self) {
        let _ = self.sender.send(HistoryCommand::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
