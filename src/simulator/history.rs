use crate::simulator::core::{ChannelId, OpKind, Operation, Value};
use rusqlite::{Connection, params};
use serde_json::{Value as JsonValue, json};
use std::error::Error;
use std::path::Path;

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
        Value::Lock(l) => {
            let val = *l.borrow();
            json!({
            "type": "VLock",
            "value": val
            })
        }
    }
}

fn payload_to_json_string(payload: &[Value]) -> String {
    let json_list: Vec<JsonValue> = payload.iter().map(json_of_value).collect();
    serde_json::to_string(&json_list).unwrap_or_else(|_| "[]".to_string())
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
/// Matches `init_sqlite` .
pub fn init_sqlite(conn: &Connection) -> Result<(), Box<dyn Error>> {
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

    Ok(())
}

/// Saves the simulation history to the SQLite database.
/// Matches `save_history` .
pub fn save_history_sqlite(
    conn: &mut Connection,
    run_id: i64,
    history: &[Operation],
) -> Result<(), Box<dyn Error>> {
    conn.execute("INSERT INTO runs (run_id) VALUES (?1)", params![run_id])?;

    let tx = conn.transaction()?;

    {
        let mut stmt = tx.prepare(
            "INSERT INTO executions
(run_id, seq_num, unique_id, client_id, kind, action, payload)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;

        for (seq_num, op) in history.iter().enumerate() {
            let kind = match op.kind {
                OpKind::Response => "Response",
                OpKind::Invocation => "Invocation",
            };

            let payload_str = payload_to_json_string(&op.payload);

            stmt.execute(params![
                run_id,
                seq_num as i64,
                op.unique_id as i64,
                op.client_id as i64,
                kind,
                op.op_action,
                payload_str
            ])?;
        }
    }

    tx.commit()?;
    Ok(())
}
