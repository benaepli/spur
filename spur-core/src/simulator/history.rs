use crate::simulator::core::{
    ChannelId, LogEntry, OpKind, Operation, TraceEntry, TraceKind, Value, ValueKind,
};
use arrow::array::{Int32Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use crossbeam::channel::{self, Receiver, Sender};
use log::error;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use serde_json::{Value as JsonValue, json};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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

/// One row of the `runs` table: what a run was and what it cost, so a
/// consumer can attribute every run to the strategy that issued it and
/// normalise by the time it took. A run that failed before producing a
/// history has no row.
pub struct PersistableRun {
    pub run_id: i64,
    /// Name of the strategy that issued the run; the explorer mode for a
    /// single-strategy session.
    pub arm: String,
    /// Position of the strategy in a multi-strategy session, -1 otherwise.
    pub arm_index: i32,
    /// Index into the expanded grid, -1 when the run was not a grid point.
    pub config_index: i32,
    pub workload_seed: u64,
    pub schedule_seed: u64,
    pub steps_used: i32,
    /// Active time the run took, on a monotonic clock.
    pub wall_us: i64,
    pub end_reason: &'static str,
    /// Active time from the session's start to the run's end.
    pub session_offset_ms: i64,
    /// Timer firings that woke a waiting record, and how many of those
    /// segments changed the node's state, split by whether a delivery to
    /// the node was pending at the firing.
    pub timers_fired: i32,
    pub timers_acted: i32,
    pub timers_inflight_fired: i32,
    pub timers_inflight_acted: i32,
    pub timers_idle_fired: i32,
    pub timers_idle_acted: i32,
    /// Longest run of inert firings at one resume point on one node.
    pub max_inert_streak: i32,
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
                OpKind::TimerFired => "TimerFired",
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
            OpKind::TimerFired => "TimerFired",
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
    Run(PersistableRun),
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

    /// Records the run's row in the `runs` table. A backend without that
    /// table may ignore it.
    fn write_run(&self, _run: PersistableRun) {}

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

fn runs_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Int64, false),
        Field::new("arm", DataType::Utf8, false),
        Field::new("arm_index", DataType::Int32, false),
        Field::new("config_index", DataType::Int32, false),
        Field::new("workload_seed", DataType::UInt64, false),
        Field::new("schedule_seed", DataType::UInt64, false),
        Field::new("steps_used", DataType::Int32, false),
        Field::new("wall_us", DataType::Int64, false),
        Field::new("end_reason", DataType::Utf8, false),
        Field::new("session_offset_ms", DataType::Int64, false),
        Field::new("timers_fired", DataType::Int32, false),
        Field::new("timers_acted", DataType::Int32, false),
        Field::new("timers_inflight_fired", DataType::Int32, false),
        Field::new("timers_inflight_acted", DataType::Int32, false),
        Field::new("timers_idle_fired", DataType::Int32, false),
        Field::new("timers_idle_acted", DataType::Int32, false),
        Field::new("max_inert_streak", DataType::Int32, false),
    ]))
}

/// Writes buffered run rows into an open ArrowWriter for the runs table.
fn append_runs_batch(
    writer: &mut ArrowWriter<File>,
    runs: &[PersistableRun],
) -> Result<(), Box<dyn Error>> {
    let run_ids: Int64Array = runs.iter().map(|r| r.run_id).collect::<Vec<_>>().into();
    let arms: StringArray = runs.iter().map(|r| r.arm.as_str()).collect::<Vec<_>>().into();
    let arm_indices: Int32Array = runs.iter().map(|r| r.arm_index).collect::<Vec<_>>().into();
    let config_indices: Int32Array = runs.iter().map(|r| r.config_index).collect::<Vec<_>>().into();
    let workload_seeds: UInt64Array = runs.iter().map(|r| r.workload_seed).collect::<Vec<_>>().into();
    let schedule_seeds: UInt64Array = runs.iter().map(|r| r.schedule_seed).collect::<Vec<_>>().into();
    let steps: Int32Array = runs.iter().map(|r| r.steps_used).collect::<Vec<_>>().into();
    let walls: Int64Array = runs.iter().map(|r| r.wall_us).collect::<Vec<_>>().into();
    let reasons: StringArray = runs.iter().map(|r| r.end_reason).collect::<Vec<_>>().into();
    let offsets: Int64Array = runs.iter().map(|r| r.session_offset_ms).collect::<Vec<_>>().into();
    let timer_col = |f: fn(&PersistableRun) -> i32| -> Int32Array { runs.iter().map(f).collect::<Vec<_>>().into() };
    let timers_fired = timer_col(|r| r.timers_fired);
    let timers_acted = timer_col(|r| r.timers_acted);
    let timers_inflight_fired = timer_col(|r| r.timers_inflight_fired);
    let timers_inflight_acted = timer_col(|r| r.timers_inflight_acted);
    let timers_idle_fired = timer_col(|r| r.timers_idle_fired);
    let timers_idle_acted = timer_col(|r| r.timers_idle_acted);
    let max_inert_streak = timer_col(|r| r.max_inert_streak);
    let batch = RecordBatch::try_new(
        runs_schema(),
        vec![
            Arc::new(run_ids),
            Arc::new(arms),
            Arc::new(arm_indices),
            Arc::new(config_indices),
            Arc::new(workload_seeds),
            Arc::new(schedule_seeds),
            Arc::new(steps),
            Arc::new(walls),
            Arc::new(reasons),
            Arc::new(offsets),
            Arc::new(timers_fired),
            Arc::new(timers_acted),
            Arc::new(timers_inflight_fired),
            Arc::new(timers_inflight_acted),
            Arc::new(timers_idle_fired),
            Arc::new(timers_idle_acted),
            Arc::new(max_inert_streak),
        ],
    )?;
    writer.write(&batch)?;
    Ok(())
}

/// Run rows are small and arrive one per run, so they are buffered and
/// written in groups rather than one row group each.
const RUNS_FLUSH_ROWS: usize = 256;

/// Number of writes between file rotations. Each batch is finalized (footer
/// written) before a new file is opened, so all completed batches survive
/// process termination.
const PARQUET_ROTATION_INTERVAL: usize = 25_000;

// Runs finish on every simulation thread faster than the writers encode
// parquet, so the queue in front of them must be bounded or it holds every
// unwritten run's history, logs and traces in memory. In-flight runs are
// bounded by this capacity plus one per writer thread; a full queue blocks
// the simulation thread until a writer catches up.
const HISTORY_QUEUE_CAPACITY: usize = 256;

/// Persists run histories as parquet. `executions/`, `logs/` and `traces/`
/// each hold a series of `batch_NNNN.parquet` files. Writer threads take
/// finished runs from one bounded queue; each owns an open file per table
/// and numbers its files so that no two writers name the same file.
pub struct ParquetWriter {
    sender: Sender<HistoryCommand>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    writer_count: usize,
}

// One writer thread encodes roughly 600 runs/s of log-heavy output while
// eight simulation threads produce about that many, so the writer count
// follows the simulation thread count at that ratio.
fn writer_thread_count() -> usize {
    rayon::current_num_threads().div_ceil(8).max(1)
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

/// Batch numbers are shared across writers: writer `index` owns
/// `index + 1`, `index + 1 + count`, `index + 1 + 2 * count`, ...
fn batch_number(writer_index: usize, series: usize, writer_count: usize) -> usize {
    writer_index + 1 + series * writer_count
}

/// Helper: formats a batch file name like `batch_0001.parquet`.
fn batch_filename(batch_num: usize) -> String {
    format!("batch_{:04}.parquet", batch_num)
}

#[derive(Clone)]
struct TableDirs {
    executions: PathBuf,
    logs: PathBuf,
    traces: PathBuf,
    runs: PathBuf,
}

/// The files a writer has open for the batch it is filling, and the run rows
/// waiting to be written into it.
struct OpenBatch {
    number: usize,
    executions: ArrowWriter<File>,
    logs: ArrowWriter<File>,
    traces: ArrowWriter<File>,
    runs: ArrowWriter<File>,
    pending_runs: Vec<PersistableRun>,
}

impl OpenBatch {
    fn open(dirs: &TableDirs, number: usize) -> Result<Self, Box<dyn Error>> {
        let name = batch_filename(number);
        Ok(Self {
            number,
            executions: open_parquet_writer(&dirs.executions.join(&name), executions_schema())?,
            logs: open_parquet_writer(&dirs.logs.join(&name), logs_schema())?,
            traces: open_parquet_writer(&dirs.traces.join(&name), traces_schema())?,
            runs: open_parquet_writer(&dirs.runs.join(&name), runs_schema())?,
            pending_runs: Vec::with_capacity(RUNS_FLUSH_ROWS),
        })
    }

    fn flush_runs(&mut self) {
        if self.pending_runs.is_empty() {
            return;
        }
        if let Err(e) = append_runs_batch(&mut self.runs, &self.pending_runs) {
            error!("failed to save runs parquet in batch {}: {}", self.number, e);
        }
        self.pending_runs.clear();
    }

    fn finish(mut self) {
        self.flush_runs();
        if let Err(e) = self.executions.finish() {
            error!("failed to finalize executions batch {}: {}", self.number, e);
        }
        if let Err(e) = self.logs.finish() {
            error!("failed to finalize logs batch {}: {}", self.number, e);
        }
        if let Err(e) = self.traces.finish() {
            error!("failed to finalize traces batch {}: {}", self.number, e);
        }
        if let Err(e) = self.runs.finish() {
            error!("failed to finalize runs batch {}: {}", self.number, e);
        }
    }
}

fn writer_loop(
    receiver: Receiver<HistoryCommand>,
    dirs: TableDirs,
    writer_index: usize,
    writer_count: usize,
    rotation_interval: usize,
    mut batch: OpenBatch,
) {
    let mut writes_in_batch: usize = 0;
    let mut series: usize = 0;
    while let Ok(cmd) = receiver.recv() {
        match cmd {
            HistoryCommand::Write {
                run_id,
                history,
                logs,
                traces,
            } => {
                if !history.is_empty()
                    && let Err(e) = append_executions_batch(&mut batch.executions, run_id, &history)
                {
                    error!("failed to save executions parquet for run {}: {}", run_id, e);
                }
                if !logs.is_empty()
                    && let Err(e) = append_logs_batch(&mut batch.logs, run_id, &logs)
                {
                    error!("failed to save logs parquet for run {}: {}", run_id, e);
                }
                if !traces.is_empty()
                    && let Err(e) = append_traces_batch(&mut batch.traces, run_id, &traces)
                {
                    error!("failed to save traces parquet for run {}: {}", run_id, e);
                }
                writes_in_batch += 1;
                if writes_in_batch >= rotation_interval {
                    batch.finish();
                    series += 1;
                    let number = batch_number(writer_index, series, writer_count);
                    batch = match OpenBatch::open(&dirs, number) {
                        Ok(b) => b,
                        Err(e) => {
                            error!("failed to open batch {}: {}", number, e);
                            return;
                        }
                    };
                    writes_in_batch = 0;
                }
            }
            HistoryCommand::Run(run) => {
                batch.pending_runs.push(run);
                if batch.pending_runs.len() >= RUNS_FLUSH_ROWS {
                    batch.flush_runs();
                }
            }
            HistoryCommand::Shutdown => break,
        }
    }
    batch.finish();
}

impl ParquetWriter {
    /// Creates a new ParquetWriter.
    /// `output_dir` is the base directory. Files are written into
    /// `output_dir/executions/batch_NNNN.parquet` and `output_dir/logs/batch_NNNN.parquet`.
    pub fn new(output_dir: &Path) -> Result<Self, Box<dyn Error>> {
        Self::spawn(output_dir, writer_thread_count(), PARQUET_ROTATION_INTERVAL)
    }

    /// `rotation_interval` counts runs across all writers; each writer
    /// rotates at its share, so finished files appear at the same cadence
    /// whatever the writer count.
    fn spawn(
        output_dir: &Path,
        writer_count: usize,
        rotation_interval: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let dirs = TableDirs {
            executions: output_dir.join("executions"),
            logs: output_dir.join("logs"),
            traces: output_dir.join("traces"),
            runs: output_dir.join("runs"),
        };
        for dir in [&dirs.executions, &dirs.logs, &dirs.traces, &dirs.runs] {
            std::fs::create_dir_all(dir)?;
        }
        let (sender, receiver) = channel::bounded::<HistoryCommand>(HISTORY_QUEUE_CAPACITY);
        let per_writer_rotation = rotation_interval.div_ceil(writer_count).max(1);
        let mut handles = Vec::with_capacity(writer_count);
        for index in 0..writer_count {
            let batch = OpenBatch::open(&dirs, batch_number(index, 0, writer_count))?;
            let receiver = receiver.clone();
            let dirs = dirs.clone();
            let handle = thread::Builder::new()
                .name(format!("parquet-writer-{index}"))
                .spawn(move || {
                    writer_loop(receiver, dirs, index, writer_count, per_writer_rotation, batch)
                })?;
            handles.push(handle);
        }
        Ok(Self {
            sender,
            handles: Mutex::new(handles),
            writer_count,
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

    fn write_run(&self, run: PersistableRun) {
        let run_id = run.run_id;
        if let Err(e) = self.sender.send(HistoryCommand::Run(run)) {
            log::error!("Failed to send run row for run {}: {}", run_id, e);
        }
    }

    fn shutdown(&self) {
        // Every writer consumes exactly one shutdown, after it has drained
        // the writes queued ahead of it.
        for _ in 0..self.writer_count {
            if let Err(e) = self.sender.send(HistoryCommand::Shutdown) {
                log::error!("Failed to send shutdown command to parquet writer: {}", e);
                break;
            }
        }
        if let Ok(mut guard) = self.handles.lock() {
            for handle in guard.drain(..) {
                if let Err(e) = handle.join() {
                    log::error!("Parquet writer thread panicked: {:?}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod parquet_writer_tests {
    use super::*;
    use arrow::array::AsArray;
    use arrow::datatypes::Int64Type;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::collections::{HashMap, HashSet};

    /// Every `(file, run_id, seq_num)` row of one table directory.
    fn read_table(dir: &Path) -> Vec<(String, i64, i64)> {
        let mut rows = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
                .unwrap()
                .build()
                .unwrap();
            for batch in reader {
                let batch = batch.unwrap();
                let run_ids = batch.column_by_name("run_id").unwrap().as_primitive::<Int64Type>();
                let seq_nums = batch.column_by_name("seq_num").unwrap().as_primitive::<Int64Type>();
                for i in 0..batch.num_rows() {
                    rows.push((name.clone(), run_ids.value(i), seq_nums.value(i)));
                }
            }
        }
        rows
    }

    fn rows_for(run_id: i64) -> usize {
        (run_id % 7 + 1) as usize
    }

    #[test]
    fn each_run_lands_in_one_file_with_contiguous_seq_nums() {
        let dir = std::env::temp_dir().join(format!("spur-history-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let writer = ParquetWriter::spawn(&dir, 3, 20).unwrap();
        let runs: i64 = 100;
        for run_id in 0..runs {
            let n = rows_for(run_id);
            let history = (0..n)
                .map(|i| PersistableOp {
                    unique_id: i as i64,
                    client_id: 0,
                    kind: "Invocation",
                    action: "Write".to_string(),
                    payload_json: "{}".to_string(),
                    step: i as i32,
                })
                .collect();
            let logs = (0..n)
                .map(|i| PersistableLog {
                    node_id: 0,
                    content: format!("line {i}"),
                    step: i as i32,
                })
                .collect();
            let traces = (0..n)
                .map(|i| PersistableTrace {
                    node_id: 0,
                    step: i as i32,
                    function_name: "f".to_string(),
                    trace_kind: "enter",
                    payload: "{}".to_string(),
                    schedulable_count: 0,
                    trace_id: i as i64,
                    causal_operation_id: None,
                })
                .collect();
            writer.write(run_id, history, logs, traces);
            writer.write_run(PersistableRun {
                run_id,
                arm: "test".to_string(),
                arm_index: -1,
                config_index: (run_id % 3) as i32,
                workload_seed: run_id as u64,
                schedule_seed: run_id as u64 + 1,
                steps_used: n as i32,
                wall_us: 10,
                end_reason: "plan_complete",
                session_offset_ms: run_id,
                timers_fired: 0,
                timers_acted: 0,
                timers_inflight_fired: 0,
                timers_inflight_acted: 0,
                timers_idle_fired: 0,
                timers_idle_acted: 0,
                max_inert_streak: 0,
            });
        }
        writer.shutdown();

        let mut run_rows: Vec<i64> = Vec::new();
        for entry in std::fs::read_dir(dir.join("runs")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }
            let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
                .unwrap()
                .build()
                .unwrap();
            for batch in reader {
                let batch = batch.unwrap();
                let ids = batch.column_by_name("run_id").unwrap().as_primitive::<Int64Type>();
                for i in 0..batch.num_rows() {
                    run_rows.push(ids.value(i));
                }
            }
        }
        run_rows.sort_unstable();
        assert_eq!(run_rows, (0..runs).collect::<Vec<i64>>(), "runs: one row per run");

        for table in ["executions", "logs", "traces"] {
            let mut per_run: HashMap<i64, (HashSet<String>, Vec<i64>)> = HashMap::new();
            for (file, run_id, seq) in read_table(&dir.join(table)) {
                let entry = per_run.entry(run_id).or_default();
                entry.0.insert(file);
                entry.1.push(seq);
            }
            assert_eq!(per_run.len(), runs as usize, "{table}: every run is present");
            for (run_id, (files, mut seqs)) in per_run {
                assert_eq!(files.len(), 1, "{table}: run {run_id} spans {files:?}");
                seqs.sort_unstable();
                let expected: Vec<i64> = (0..rows_for(run_id) as i64).collect();
                assert_eq!(seqs, expected, "{table}: run {run_id} seq_nums");
            }
            let files = std::fs::read_dir(dir.join(table)).unwrap().count();
            assert!(files > 3, "{table}: rotation produced only {files} files");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Which storage backend to use for logging history.
#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum LogBackend {
    #[default]
    Parquet,
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
