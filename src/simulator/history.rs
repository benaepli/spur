use crate::simulator::core::{ChannelId, LogEntry, OpKind, Operation, Value, ValueKind};
use duckdb::{Connection, params};
use log::error;
use rayon::prelude::*;
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

fn json_of_value<H: crate::simulator::hash_utils::HashPolicy>(v: &Value<H>) -> JsonValue {
    match &v.kind {
        ValueKind::Int(i) => json!({
        "type": "VInt",
        "value": i
        }),
        ValueKind::Bool(b) => json!({
        "type": "VBool",
        "value": b
        }),
        ValueKind::String(s) => json!({
        "type": "VString",
        "value": s
        }),
        ValueKind::Node(n) => json!({
        "type": "VNode",
        "value": n
        }),
        ValueKind::Channel(ChannelId { node, id }) => json!({
        "type": "VChannel",
        "value": { "node": node, "id": id }
        }),
        ValueKind::Map(m) => {
            let json_pairs: Vec<JsonValue> = m
                .iter()
                .map(|(k, v)| json!([json_of_value(k), json_of_value(v)]))
                .collect();
            json!({
            "type": "VMap",
            "value": json_pairs
            })
        }
        ValueKind::Option(opt) => {
            let value_json = match opt {
                Some(inner) => json_of_value(inner),
                None => JsonValue::Null,
            };
            json!({
            "type": "VOption",
            "value": value_json
            })
        }
        ValueKind::List(l) => {
            let items: Vec<JsonValue> = l.iter().map(json_of_value).collect();
            json!({
            "type": "VList",
            "value": items
            })
        }
        ValueKind::Unit => json!({
        "type": "VUnit",
        "value": null
        }),
        ValueKind::Tuple(t) => {
            let items: Vec<JsonValue> = t.iter().map(json_of_value).collect();
            json!({
            "type": "VTuple",
            "value": items
            })
        }
    }
}

fn payload_to_json_string<H: crate::simulator::hash_utils::HashPolicy>(payload: &[Value<H>]) -> String {
    let json_list: Vec<JsonValue> = payload.iter().map(json_of_value::<H>).collect();
    serde_json::to_string(&json_list).unwrap_or_else(|_| "[]".to_string())
}

/// Serializes a list of Operations into PersistableOps.
/// This should be called from worker threads to distribute CPU work.
pub fn serialize_history<H: crate::simulator::hash_utils::HashPolicy>(history: &[Operation<H>]) -> Vec<PersistableOp> {
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
            payload_json: payload_to_json_string::<H>(&op.payload),
        })
        .collect()
}

/// Saves the simulation history to a CSV file.
pub fn save_history_to_csv<H: crate::simulator::hash_utils::HashPolicy, P: AsRef<Path>>(
    history: &[Operation<H>],
    filename: P,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(filename)?;

    wtr.write_record(["UniqueID", "ClientID", "Kind", "Action", "Payload"])?;

    for op in history {
        let kind = match op.kind {
            OpKind::Response => "Response",
            OpKind::Invocation => "Invocation",
            OpKind::Crash => "Crash",
            OpKind::Recover => "Recover",
        };

        let payload_str = payload_to_json_string::<H>(&op.payload);
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

/// Initialize the DuckDB database with the required tables.
pub fn init_db(conn: &Connection) -> Result<(), duckdb::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            meta_info TEXT
        );

        CREATE TABLE IF NOT EXISTS executions (
            run_id INTEGER REFERENCES runs(run_id),
            seq_num INTEGER,
            unique_id INTEGER,
            client_id INTEGER,
            kind TEXT,
            action TEXT,
            payload TEXT,
            PRIMARY KEY (run_id, seq_num)
        );

        CREATE INDEX IF NOT EXISTS idx_run_execution ON executions(run_id, seq_num);

        CREATE TABLE IF NOT EXISTS logs (
            run_id INTEGER REFERENCES runs(run_id),
            seq_num INTEGER,
            node_id INTEGER,
            step INTEGER,
            content TEXT,
            PRIMARY KEY (run_id, seq_num)
        );

        CREATE INDEX IF NOT EXISTS idx_run_logs ON logs(run_id);",
    )?;

    Ok(())
}

/// Saves pre-serialized history to the DuckDB database using bulk Appender API.
fn save_history_db(
    conn: &Connection,
    run_id: i64,
    history: &[PersistableOp],
) -> Result<(), Box<dyn Error>> {
    conn.execute("INSERT INTO runs (run_id) VALUES (?1)", params![run_id])?;

    // Use DuckDB's Appender for fast bulk inserts
    let mut appender = conn.appender("executions")?;

    for (seq_num, op) in history.iter().enumerate() {
        appender.append_row(params![
            run_id,
            seq_num as i64,
            op.unique_id,
            op.client_id,
            op.kind,
            &op.action,
            &op.payload_json
        ])?;
    }

    appender.flush()?;
    Ok(())
}

fn save_logs_db(
    conn: &Connection,
    run_id: i64,
    logs: &[PersistableLog],
) -> Result<(), Box<dyn Error>> {
    // Use DuckDB's Appender for fast bulk inserts
    let mut appender = conn.appender("logs")?;

    for (seq_num, log) in logs.iter().enumerate() {
        appender.append_row(params![
            run_id,
            seq_num as i64,
            log.node_id,
            log.step,
            &log.content
        ])?;
    }

    appender.flush()?;
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

/// A background writer that serializes all DuckDB writes to a single thread.
pub struct HistoryWriter {
    sender: Sender<HistoryCommand>,
    handle: Option<JoinHandle<()>>,
}

impl HistoryWriter {
    /// Creates a new HistoryWriter, opening and initializing the DuckDB database.
    /// The connection is then moved to a background thread that processes write requests.
    pub fn new(db_path: &str) -> Result<Self, Box<dyn Error>> {
        let conn = Connection::open(db_path)?;
        init_db(&conn)?;

        let (sender, receiver) = mpsc::channel::<HistoryCommand>();

        let handle = thread::spawn(move || {
            while let Ok(cmd) = receiver.recv() {
                match cmd {
                    HistoryCommand::Write {
                        run_id,
                        history,
                        logs,
                    } => {
                        if let Err(e) = save_history_db(&conn, run_id, &history) {
                            error!("failed to save history for run {}: {}", run_id, e);
                        }
                        if let Err(e) = save_logs_db(&conn, run_id, &logs) {
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
        if let Err(e) = self.sender.send(HistoryCommand::Write {
            run_id,
            history,
            logs,
        }) {
            log::error!(
                "Failed to send history write command for run {}: {}",
                run_id,
                e
            );
        }
    }

    /// Shuts down the background writer, waiting for all pending writes to complete.
    pub fn shutdown(mut self) {
        if let Err(e) = self.sender.send(HistoryCommand::Shutdown) {
            log::error!("Failed to send shutdown command to history writer: {}", e);
        }
        if let Some(h) = self.handle.take() {
            if let Err(e) = h.join() {
                log::error!("History writer thread panicked during shutdown: {:?}", e);
            }
        }
    }
}
