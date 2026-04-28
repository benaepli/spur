use crate::simulator::core::{
    ChannelId, LogEntry, OpKind, Operation, TraceEntry, TraceKind, Value, ValueKind,
};
use arrow::array::{Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use log::error;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use serde_json::{Value as JsonValue, json};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

/// A pre-serialized operation ready for database / file insertion.
/// JSON serialization is done by worker threads before sending to the writer.
pub struct PersistableOp {
    pub unique_id: i64,
    pub client_id: i64,
    pub kind: &'static str,
    pub action: String,
    pub payload_json: String,
    pub step: i32,
}

pub struct PersistableLog {
    pub node_id: i64,
    pub content: String,
    pub step: i32,
}

pub fn serialize_logs(logs: &[LogEntry]) -> Vec<PersistableLog> {
    logs.par_iter()
        .map(|l| PersistableLog {
            node_id: l.node.index as i64,
            content: l.content.clone(),
            step: l.step,
        })
        .collect()
}

pub struct PersistableTrace {
    pub node_id: i64,
    pub step: i32,
    pub function_name: String,
    pub trace_kind: &'static str,
    pub payload: String,
    pub schedulable_count: i64,
    pub trace_id: i64,
    pub causal_operation_id: Option<i64>,
}

pub fn serialize_traces(traces: &[TraceEntry]) -> Vec<PersistableTrace> {
    traces
        .par_iter()
        .map(|t| {
            let payload = if t.payload.is_empty() {
                "[]".to_string()
            } else {
                let items: Vec<JsonValue> = t
                    .payload
                    .iter()
                    .map(|s| JsonValue::String(s.clone()))
                    .collect();
                serde_json::to_string(&items).unwrap_or_else(|_| "[]".to_string())
            };
            PersistableTrace {
                node_id: t.node.index as i64,
                step: t.step,
                function_name: t.function_name.clone(),
                trace_kind: match t.kind {
                    TraceKind::Dispatch => "Dispatch",
                    TraceKind::Enter => "Enter",
                    TraceKind::Exit => "Exit",
                },
                payload,
                schedulable_count: t.schedulable_count as i64,
                trace_id: t.trace_id,
                causal_operation_id: t.causal_operation_id,
            }
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
        ValueKind::FifoLink(link_id, peer) => json!({
        "type": "VFifoLink",
        "value": { "link_id": link_id.0, "peer": peer }
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
        ValueKind::Variant(enum_id, name, payload) => {
            let payload_json = match payload {
                Some(inner) => json_of_value(inner),
                None => JsonValue::Null,
            };
            json!({
                "type": "VVariant",
                "value": {
                    "enum_id": enum_id,
                    "name": name.as_str(),
                    "payload": payload_json
                }
            })
        }
    }
}

fn payload_to_json_string<H: crate::simulator::hash_utils::HashPolicy>(
    payload: &[Value<H>],
) -> String {
    let json_list: Vec<JsonValue> = payload.iter().map(json_of_value::<H>).collect();
    serde_json::to_string(&json_list).unwrap_or_else(|_| "[]".to_string())
}

/// Serializes a list of Operations into PersistableOps.
/// This should be called from worker threads to distribute CPU work.
pub fn serialize_history<H: crate::simulator::hash_utils::HashPolicy>(
    history: &[Operation<H>],
) -> Vec<PersistableOp> {
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
                OpKind::Partition => "Partition",
                OpKind::Heal => "Heal",
            },
            action: op.op_action.clone(),
            payload_json: payload_to_json_string::<H>(&op.payload),
            step: op.step,
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
            OpKind::Partition => "Partition",
            OpKind::Heal => "Heal",
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

// ─── HistoryWriter trait ──────────────────────────────────────────────────────

/// Command sent to the background history writer thread.
pub enum HistoryCommand {
    Write {
        run_id: i64,
        history: Vec<PersistableOp>,
        logs: Vec<PersistableLog>,
        traces: Vec<PersistableTrace>,
    },
    Shutdown,
}

/// The abstract interface for logging simulation history.
/// Implementations must be Send + Sync so they can be wrapped in `Arc<dyn HistoryWriter>`.
pub trait HistoryWriter: Send + Sync {
    /// Sends a pre-serialized history, logs, and traces write request to the background thread.
    fn write(
        &self,
        run_id: i64,
        history: Vec<PersistableOp>,
        logs: Vec<PersistableLog>,
        traces: Vec<PersistableTrace>,
    );

    /// Shuts down the background writer, waiting for all pending writes to complete.
    fn shutdown(&self);
}

// ─── Parquet backend ──────────────────────────────────────────────────────────

fn executions_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Int64, false),
        Field::new("seq_num", DataType::Int64, false),
        Field::new("unique_id", DataType::Int64, false),
        Field::new("client_id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("action", DataType::Utf8, false),
        Field::new("payload", DataType::Utf8, false),
        Field::new("step", DataType::Int32, false),
    ]))
}

fn logs_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Int64, false),
        Field::new("seq_num", DataType::Int64, false),
        Field::new("node_id", DataType::Int64, false),
        Field::new("step", DataType::Int32, false),
        Field::new("content", DataType::Utf8, false),
    ]))
}

/// Writes a batch of PersistableOps into an open ArrowWriter for the executions table.
fn append_executions_batch(
    writer: &mut ArrowWriter<File>,
    run_id: i64,
    ops: &[PersistableOp],
) -> Result<(), Box<dyn Error>> {
    let n = ops.len();
    let run_ids = Int64Array::from(vec![run_id; n]);
    let seq_nums: Int64Array = (0..n as i64).collect::<Vec<_>>().into();
    let unique_ids: Int64Array = ops.iter().map(|o| o.unique_id).collect::<Vec<_>>().into();
    let client_ids: Int64Array = ops.iter().map(|o| o.client_id).collect::<Vec<_>>().into();
    let kinds: StringArray = ops.iter().map(|o| o.kind).collect::<Vec<_>>().into();
    let actions: StringArray = ops
        .iter()
        .map(|o| o.action.as_str())
        .collect::<Vec<_>>()
        .into();
    let payloads: StringArray = ops
        .iter()
        .map(|o| o.payload_json.as_str())
        .collect::<Vec<_>>()
        .into();
    let steps: Int32Array = ops.iter().map(|o| o.step).collect::<Vec<_>>().into();

    let batch = RecordBatch::try_new(
        executions_schema(),
        vec![
            Arc::new(run_ids),
            Arc::new(seq_nums),
            Arc::new(unique_ids),
            Arc::new(client_ids),
            Arc::new(kinds),
            Arc::new(actions),
            Arc::new(payloads),
            Arc::new(steps),
        ],
    )?;
    writer.write(&batch)?;
    Ok(())
}

/// Writes a batch of PersistableLogs into an open ArrowWriter for the logs table.
fn append_logs_batch(
    writer: &mut ArrowWriter<File>,
    run_id: i64,
    logs: &[PersistableLog],
) -> Result<(), Box<dyn Error>> {
    let n = logs.len();
    let run_ids = Int64Array::from(vec![run_id; n]);
    let seq_nums: Int64Array = (0..n as i64).collect::<Vec<_>>().into();
    let node_ids: Int64Array = logs.iter().map(|l| l.node_id).collect::<Vec<_>>().into();
    let steps: Int32Array = logs.iter().map(|l| l.step).collect::<Vec<_>>().into();
    let contents: StringArray = logs
        .iter()
        .map(|l| l.content.as_str())
        .collect::<Vec<_>>()
        .into();

    let batch = RecordBatch::try_new(
        logs_schema(),
        vec![
            Arc::new(run_ids),
            Arc::new(seq_nums),
            Arc::new(node_ids),
            Arc::new(steps),
            Arc::new(contents),
        ],
    )?;
    writer.write(&batch)?;
    Ok(())
}

fn traces_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Int64, false),
        Field::new("seq_num", DataType::Int64, false),
        Field::new("node_id", DataType::Int64, false),
        Field::new("step", DataType::Int32, false),
        Field::new("function_name", DataType::Utf8, false),
        Field::new("trace_kind", DataType::Utf8, false),
        Field::new("payload", DataType::Utf8, false),
        Field::new("schedulable_count", DataType::Int64, false),
        Field::new("trace_id", DataType::Int64, false),
        Field::new("causal_operation_id", DataType::Int64, true),
    ]))
}

/// Writes a batch of PersistableTraces into an open ArrowWriter for the traces table.
fn append_traces_batch(
    writer: &mut ArrowWriter<File>,
    run_id: i64,
    traces: &[PersistableTrace],
) -> Result<(), Box<dyn Error>> {
    let n = traces.len();
    let run_ids = Int64Array::from(vec![run_id; n]);
    let seq_nums: Int64Array = (0..n as i64).collect::<Vec<_>>().into();
    let node_ids: Int64Array = traces.iter().map(|t| t.node_id).collect::<Vec<_>>().into();
    let steps: Int32Array = traces.iter().map(|t| t.step).collect::<Vec<_>>().into();
    let func_names: StringArray = traces
        .iter()
        .map(|t| t.function_name.as_str())
        .collect::<Vec<_>>()
        .into();
    let kinds: StringArray = traces
        .iter()
        .map(|t| t.trace_kind)
        .collect::<Vec<_>>()
        .into();
    let payloads: StringArray = traces
        .iter()
        .map(|t| t.payload.as_str())
        .collect::<Vec<_>>()
        .into();
    let sched_counts: Int64Array = traces
        .iter()
        .map(|t| t.schedulable_count)
        .collect::<Vec<_>>()
        .into();
    let trace_ids: Int64Array = traces.iter().map(|t| t.trace_id).collect::<Vec<_>>().into();
    let causal_op_ids: Int64Array = traces
        .iter()
        .map(|t| t.causal_operation_id)
        .collect::<Vec<Option<i64>>>()
        .into();

    let batch = RecordBatch::try_new(
        traces_schema(),
        vec![
            Arc::new(run_ids),
            Arc::new(seq_nums),
            Arc::new(node_ids),
            Arc::new(steps),
            Arc::new(func_names),
            Arc::new(kinds),
            Arc::new(payloads),
            Arc::new(sched_counts),
            Arc::new(trace_ids),
            Arc::new(causal_op_ids),
        ],
    )?;
    writer.write(&batch)?;
    Ok(())
}

/// Number of writes between file rotations. Each batch is finalized (footer
/// written) before a new file is opened, so all completed batches survive
/// process termination.
const PARQUET_ROTATION_INTERVAL: usize = 25_000;

/// A background writer that serializes all Parquet writes to a single thread.
/// Outputs batched files into `executions/` and `logs/` subdirectories.
pub struct ParquetWriter {
    sender: Sender<HistoryCommand>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

/// Helper: creates a new ArrowWriter for the given path and schema.
fn open_parquet_writer(
    path: &Path,
    schema: Arc<Schema>,
) -> Result<ArrowWriter<File>, Box<dyn Error>> {
    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    Ok(ArrowWriter::try_new(file, schema, Some(props))?)
}

/// Helper: formats a batch file name like `batch_0001.parquet`.
fn batch_filename(batch_num: usize) -> String {
    format!("batch_{:04}.parquet", batch_num)
}

impl ParquetWriter {
    /// Creates a new ParquetWriter.
    /// `output_dir` is the base directory. Files are written into
    /// `output_dir/executions/batch_NNNN.parquet` and `output_dir/logs/batch_NNNN.parquet`.
    pub fn new(output_dir: &Path) -> Result<Self, Box<dyn Error>> {
        let exec_dir = output_dir.join("executions");
        let logs_dir = output_dir.join("logs");
        let traces_dir = output_dir.join("traces");
        std::fs::create_dir_all(&exec_dir)?;
        std::fs::create_dir_all(&logs_dir)?;
        std::fs::create_dir_all(&traces_dir)?;

        let exec_schema = executions_schema();
        let logs_schema_arc = logs_schema();
        let traces_schema_arc = traces_schema();

        let mut exec_writer =
            open_parquet_writer(&exec_dir.join(batch_filename(1)), exec_schema.clone())?;
        let mut log_writer =
            open_parquet_writer(&logs_dir.join(batch_filename(1)), logs_schema_arc.clone())?;
        let mut trace_writer = open_parquet_writer(
            &traces_dir.join(batch_filename(1)),
            traces_schema_arc.clone(),
        )?;

        let (sender, receiver) = mpsc::channel::<HistoryCommand>();

        let handle = thread::spawn(move || {
            let mut writes_in_batch: usize = 0;
            let mut batch_num: usize = 1;

            while let Ok(cmd) = receiver.recv() {
                match cmd {
                    HistoryCommand::Write {
                        run_id,
                        history,
                        logs,
                        traces,
                    } => {
                        if !history.is_empty() {
                            if let Err(e) =
                                append_executions_batch(&mut exec_writer, run_id, &history)
                            {
                                error!(
                                    "failed to save executions parquet for run {}: {}",
                                    run_id, e
                                );
                            }
                        }
                        if !logs.is_empty() {
                            if let Err(e) = append_logs_batch(&mut log_writer, run_id, &logs) {
                                error!("failed to save logs parquet for run {}: {}", run_id, e);
                            }
                        }
                        if !traces.is_empty() {
                            if let Err(e) = append_traces_batch(&mut trace_writer, run_id, &traces)
                            {
                                error!("failed to save traces parquet for run {}: {}", run_id, e);
                            }
                        }

                        writes_in_batch += 1;

                        // Rotate: finalize current files and open new ones
                        if writes_in_batch >= PARQUET_ROTATION_INTERVAL {
                            if let Err(e) = exec_writer.finish() {
                                error!("failed to finalize executions batch {}: {}", batch_num, e);
                            }
                            if let Err(e) = log_writer.finish() {
                                error!("failed to finalize logs batch {}: {}", batch_num, e);
                            }
                            if let Err(e) = trace_writer.finish() {
                                error!("failed to finalize traces batch {}: {}", batch_num, e);
                            }

                            batch_num += 1;
                            writes_in_batch = 0;

                            match open_parquet_writer(
                                &exec_dir.join(batch_filename(batch_num)),
                                exec_schema.clone(),
                            ) {
                                Ok(w) => exec_writer = w,
                                Err(e) => {
                                    error!(
                                        "failed to open new executions batch {}: {}",
                                        batch_num, e
                                    );
                                    break;
                                }
                            }
                            match open_parquet_writer(
                                &logs_dir.join(batch_filename(batch_num)),
                                logs_schema_arc.clone(),
                            ) {
                                Ok(w) => log_writer = w,
                                Err(e) => {
                                    error!("failed to open new logs batch {}: {}", batch_num, e);
                                    break;
                                }
                            }
                            match open_parquet_writer(
                                &traces_dir.join(batch_filename(batch_num)),
                                traces_schema_arc.clone(),
                            ) {
                                Ok(w) => trace_writer = w,
                                Err(e) => {
                                    error!("failed to open new traces batch {}: {}", batch_num, e);
                                    break;
                                }
                            }
                        }
                    }
                    HistoryCommand::Shutdown => {
                        if let Err(e) = exec_writer.finish() {
                            error!("failed to finalize executions batch {}: {}", batch_num, e);
                        }
                        if let Err(e) = log_writer.finish() {
                            error!("failed to finalize logs batch {}: {}", batch_num, e);
                        }
                        if let Err(e) = trace_writer.finish() {
                            error!("failed to finalize traces batch {}: {}", batch_num, e);
                        }
                        break;
                    }
                }
            }
        });

        Ok(Self {
            sender,
            handle: Mutex::new(Some(handle)),
        })
    }
}

impl HistoryWriter for ParquetWriter {
    fn write(
        &self,
        run_id: i64,
        history: Vec<PersistableOp>,
        logs: Vec<PersistableLog>,
        traces: Vec<PersistableTrace>,
    ) {
        if let Err(e) = self.sender.send(HistoryCommand::Write {
            run_id,
            history,
            logs,
            traces,
        }) {
            log::error!(
                "Failed to send parquet write command for run {}: {}",
                run_id,
                e
            );
        }
    }

    fn shutdown(&self) {
        if let Err(e) = self.sender.send(HistoryCommand::Shutdown) {
            log::error!("Failed to send shutdown command to parquet writer: {}", e);
        }
        if let Ok(mut guard) = self.handle.lock() {
            if let Some(h) = guard.take() {
                if let Err(e) = h.join() {
                    log::error!("Parquet writer thread panicked: {:?}", e);
                }
            }
        }
    }
}

/// Which storage backend to use for logging history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogBackend {
    Parquet,
}

impl Default for LogBackend {
    fn default() -> Self {
        LogBackend::Parquet
    }
}

/// Creates the appropriate HistoryWriter for the given backend.
pub fn create_writer(
    backend: LogBackend,
    output_path: &str,
) -> Result<Box<dyn HistoryWriter>, Box<dyn Error>> {
    match backend {
        LogBackend::Parquet => {
            let dir = PathBuf::from(output_path);
            std::fs::create_dir_all(&dir)?;
            Ok(Box::new(ParquetWriter::new(&dir)?))
        }
    }
}
