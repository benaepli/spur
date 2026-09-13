use crate::simulator::core::{
    ChannelId, LogEntry, OpKind, Operation, TraceEntry, TraceKind, Value, ValueKind,
};
use crate::simulator::util_stats;
use arrow::array::{Int32Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use crossbeam::channel::{self, Receiver, Sender};
use log::error;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use crate::simulator::core::state::NodeId;
use crate::simulator::core::values::ValueMap;
use crate::simulator::hash_utils::HashPolicy;
use serde::ser::{Serialize, SerializeMap, Serializer};
#[cfg(test)]
use serde_json::{Value as JsonValue, json};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

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

/// Rows keep the order of `logs`; the text moves into each row unchanged.
pub fn serialize_logs(logs: Vec<LogEntry>) -> Vec<PersistableLog> {
    logs.into_iter()
        .map(|l| PersistableLog {
            node_id: l.node.index as i64,
            content: l.content,
            step: l.step,
        })
        .collect()
}

pub struct PersistableTrace {
    pub node_id: i64,
    pub step: i32,
    pub function_name: Arc<str>,
    pub trace_kind: &'static str,
    /// JSON array of the parameter texts.
    pub payload: String,
    pub schedulable_count: i64,
    pub trace_id: i64,
    pub causal_operation_id: Option<i64>,
}

/// Rows keep the order of `traces`; the name and the already serialized
/// payload move into each row unchanged.
pub fn serialize_traces(traces: Vec<TraceEntry>) -> Vec<PersistableTrace> {
    traces
        .into_iter()
        .map(|t| PersistableTrace {
            node_id: t.node.index as i64,
            step: t.step,
            function_name: t.function_name,
            trace_kind: match t.kind {
                TraceKind::Dispatch => "Dispatch",
                TraceKind::Enter => "Enter",
                TraceKind::Exit => "Exit",
            },
            payload: t.payload,
            schedulable_count: t.schedulable_count as i64,
            trace_id: t.trace_id,
            causal_operation_id: t.causal_operation_id,
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
    /// Bitfield naming the session-global mechanisms that selected this run
    /// and whether placed crashes acted; see `run_variant`.
    pub variant: i32,
}

#[cfg(test)]
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

/// A spec value written as the JSON object `{"type":..,"value":..}`. Every
/// object is written with its keys in sorted order, the order a
/// `serde_json::Value` object holds them in, so the text is byte for byte
/// what serializing the equivalent `serde_json::Value` produces.
struct ValueJson<'a, H: HashPolicy>(&'a Value<H>);

/// A node written as `{"index":..,"role":..}`. The derived `Serialize` of
/// `NodeId` writes its fields in declaration order, which is not sorted, so
/// it must not be used here.
struct NodeJson(NodeId);

struct SeqJson<'a, H: HashPolicy>(&'a [Value<H>]);

struct MapJson<'a, H: HashPolicy>(&'a ValueMap<H>);

struct ChannelJson(ChannelId);

struct FifoLinkJson {
    link_id: usize,
    peer: NodeId,
}

struct VariantJson<'a, H: HashPolicy> {
    enum_id: u32,
    name: &'a str,
    payload: Option<&'a Value<H>>,
}

impl Serialize for NodeJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut m = serializer.serialize_map(Some(2))?;
        m.serialize_entry("index", &self.0.index)?;
        m.serialize_entry("role", &self.0.role)?;
        m.end()
    }
}

impl<H: HashPolicy> Serialize for SeqJson<'_, H> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(ValueJson))
    }
}

impl<H: HashPolicy> Serialize for MapJson<'_, H> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(|(k, v)| (ValueJson(k), ValueJson(v))))
    }
}

impl Serialize for ChannelJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut m = serializer.serialize_map(Some(2))?;
        m.serialize_entry("id", &self.0.id)?;
        m.serialize_entry("node", &NodeJson(self.0.node))?;
        m.end()
    }
}

impl Serialize for FifoLinkJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut m = serializer.serialize_map(Some(2))?;
        m.serialize_entry("link_id", &self.link_id)?;
        m.serialize_entry("peer", &NodeJson(self.peer))?;
        m.end()
    }
}

impl<H: HashPolicy> Serialize for VariantJson<'_, H> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut m = serializer.serialize_map(Some(3))?;
        m.serialize_entry("enum_id", &self.enum_id)?;
        m.serialize_entry("name", self.name)?;
        m.serialize_entry("payload", &self.payload.map(ValueJson))?;
        m.end()
    }
}

impl<H: HashPolicy> Serialize for ValueJson<'_, H> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut m = serializer.serialize_map(Some(2))?;
        match &self.0.kind {
            ValueKind::Int(i) => {
                m.serialize_entry("type", "VInt")?;
                m.serialize_entry("value", i)?;
            }
            ValueKind::Bool(b) => {
                m.serialize_entry("type", "VBool")?;
                m.serialize_entry("value", b)?;
            }
            ValueKind::String(s) => {
                m.serialize_entry("type", "VString")?;
                m.serialize_entry("value", s.as_str())?;
            }
            ValueKind::Node(n) => {
                m.serialize_entry("type", "VNode")?;
                m.serialize_entry("value", &NodeJson(*n))?;
            }
            ValueKind::Channel(c) => {
                m.serialize_entry("type", "VChannel")?;
                m.serialize_entry("value", &ChannelJson(*c))?;
            }
            ValueKind::FifoLink(link_id, peer) => {
                m.serialize_entry("type", "VFifoLink")?;
                m.serialize_entry(
                    "value",
                    &FifoLinkJson {
                        link_id: link_id.0,
                        peer: *peer,
                    },
                )?;
            }
            ValueKind::Map(map) => {
                m.serialize_entry("type", "VMap")?;
                m.serialize_entry("value", &MapJson(map))?;
            }
            ValueKind::Option(opt) => {
                m.serialize_entry("type", "VOption")?;
                m.serialize_entry("value", &opt.as_deref().map(ValueJson))?;
            }
            ValueKind::List(l) => {
                m.serialize_entry("type", "VList")?;
                m.serialize_entry("value", &SeqJson(l.as_slice()))?;
            }
            ValueKind::Unit => {
                m.serialize_entry("type", "VUnit")?;
                m.serialize_entry("value", &())?;
            }
            ValueKind::Tuple(t) => {
                m.serialize_entry("type", "VTuple")?;
                m.serialize_entry("value", &SeqJson(t.as_slice()))?;
            }
            ValueKind::Variant(enum_id, name, payload) => {
                m.serialize_entry("type", "VVariant")?;
                m.serialize_entry(
                    "value",
                    &VariantJson {
                        enum_id: *enum_id,
                        name: name.as_str(),
                        payload: payload.as_deref(),
                    },
                )?;
            }
        }
        m.end()
    }
}

fn payload_to_json_string<H: HashPolicy>(payload: &[Value<H>]) -> String {
    serde_json::to_string(&SeqJson(payload)).unwrap_or_else(|_| "[]".to_string())
}

/// Serializes a list of Operations into PersistableOps, in order.
pub fn serialize_history<H: HashPolicy>(history: &[Operation<H>]) -> Vec<PersistableOp> {
    util_stats::record_history_ops_streamed(history.len() as u64);
    history
        .iter()
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
        .map(|t| &*t.function_name)
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
        Field::new("variant", DataType::Int32, false),
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
    let variants = timer_col(|r| r.variant);
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
            Arc::new(variants),
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
        let started = util_stats::enabled().then(Instant::now);
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
        record_writer_busy(started);
    }
    let started = util_stats::enabled().then(Instant::now);
    batch.finish();
    record_writer_busy(started);
}

fn record_writer_busy(started: Option<Instant>) {
    if let Some(started) = started {
        util_stats::record_history_writer_busy(started.elapsed().as_nanos() as u64);
    }
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

impl ParquetWriter {
    /// Queues a command, waiting for room when the queue is full. Only a send
    /// that finds the queue full reads the clock, so simulation threads pay
    /// nothing extra while the writers keep up.
    fn send(&self, cmd: HistoryCommand) -> Result<(), channel::SendError<HistoryCommand>> {
        match self.sender.try_send(cmd) {
            Ok(()) => Ok(()),
            Err(channel::TrySendError::Full(cmd)) => {
                let started = Instant::now();
                let sent = self.sender.send(cmd);
                util_stats::record_history_writer_blocked(started.elapsed().as_nanos() as u64);
                sent
            }
            Err(channel::TrySendError::Disconnected(cmd)) => Err(channel::SendError(cmd)),
        }
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
        if let Err(e) = self.send(HistoryCommand::Write {
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
        if let Err(e) = self.send(HistoryCommand::Run(run)) {
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
                    function_name: Arc::from("f"),
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
                variant: 0,
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

    #[test]
    fn writer_busy_time_and_full_queue_waits_are_counted() {
        let _serial = crate::simulator::config_override::exclusive_session();
        util_stats::set_enabled(true);
        let before = util_stats::snapshot().history_writer;

        let dir = std::env::temp_dir().join(format!("spur-history-busy-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let writer = ParquetWriter::spawn(&dir, 1, 20).unwrap();
        let history = (0..5)
            .map(|i| PersistableOp {
                unique_id: i,
                client_id: 0,
                kind: "Invocation",
                action: "Write".to_string(),
                payload_json: "{}".to_string(),
                step: i as i32,
            })
            .collect();
        writer.write(0, history, Vec::new(), Vec::new());
        writer.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
        let after_write = util_stats::snapshot().history_writer;
        assert!(after_write.busy_ns > before.busy_ns, "writer work was not timed");
        assert_eq!(after_write.queue_full_sends, before.queue_full_sends, "a queue with room counted as full");

        // The drain waits long enough that the second send finds the queue full.
        let (sender, receiver) = channel::bounded::<HistoryCommand>(1);
        let full = ParquetWriter { sender, handles: Mutex::new(Vec::new()), writer_count: 0 };
        full.send(HistoryCommand::Shutdown).unwrap();
        let drain = thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(100));
            while receiver.recv().is_ok() {}
        });
        full.send(HistoryCommand::Shutdown).unwrap();
        drop(full);
        drain.join().unwrap();
        let after_full = util_stats::snapshot().history_writer;
        assert_eq!(after_full.queue_full_sends, after_write.queue_full_sends + 1);
        assert!(after_full.blocked_ns > after_write.blocked_ns, "the wait for room was not timed");

        util_stats::set_enabled(false);
        util_stats::record_history_writer_blocked(1);
        util_stats::record_history_writer_busy(1);
        let disabled = util_stats::snapshot().history_writer;
        assert_eq!(disabled.queue_full_sends, after_full.queue_full_sends, "a disabled session counted");
        assert_eq!(disabled.busy_ns, after_full.busy_ns, "a disabled session counted");
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

#[cfg(test)]
mod payload_json_tests {
    use super::*;
    use crate::analysis::resolver::NameId;
    use crate::simulator::core::values::{LinkId, ValueSeq};
    use crate::simulator::hash_utils::{NoHashing, WithHashing};

    fn node(role: usize, index: usize) -> NodeId {
        NodeId {
            role: NameId(role),
            index,
        }
    }

    fn tree_text<H: HashPolicy>(payload: &[Value<H>]) -> String {
        let list: Vec<JsonValue> = payload.iter().map(json_of_value::<H>).collect();
        serde_json::to_string(&list).unwrap()
    }

    /// One value of every kind, alone and nested inside every container kind,
    /// with strings that need escaping and nodes whose role and index differ.
    fn every_kind<H: HashPolicy>() -> Vec<Value<H>> {
        let chan = Value::<H>::channel(ChannelId {
            node: node(3, 11),
            id: 42,
        });
        let link = Value::<H>::fifo_link(LinkId(9), node(1, 4));
        let leaves = vec![
            Value::int(0),
            Value::int(-17),
            Value::int(i64::MIN),
            Value::int(i64::MAX),
            Value::bool(true),
            Value::bool(false),
            Value::string("".into()),
            Value::string("quote \" backslash \\ newline \n tab \t ctrl \u{1} e-acute \u{e9}".into()),
            Value::node(node(7, 2)),
            chan.clone(),
            link.clone(),
            Value::unit(),
            Value::option_none(),
            Value::option_some(Value::int(5)),
            Value::list(ValueSeq::new()),
            Value::tuple(ValueSeq::new()),
            Value::map(ValueMap::<H>::default()),
            Value::variant(3, "Empty".into(), None),
        ];
        let mut map = ValueMap::<H>::default();
        for (i, leaf) in leaves.iter().enumerate() {
            map.insert(Value::int(i as i64), leaf.clone());
        }
        map.insert(Value::string("k\"ey".into()), Value::node(node(0, 5)));
        map.insert(Value::node(node(2, 1)), chan.clone());
        map.insert(link.clone(), Value::option_some(link.clone()));
        let list = Value::list(leaves.iter().cloned().collect());
        let tuple = Value::tuple(ValueSeq::from([
            Value::node(node(4, 0)),
            link.clone(),
            Value::unit(),
            list.clone(),
        ]));
        let nested_map = Value::map(map);
        let variant = Value::variant(
            12,
            "With\"Payload".into(),
            Some(std::sync::Arc::new(Value::tuple(ValueSeq::from([
                nested_map.clone(),
                Value::option_some(Value::option_some(chan.clone())),
            ])))),
        );
        let mut out = leaves;
        out.push(list);
        out.push(tuple);
        out.push(nested_map);
        out.push(variant.clone());
        out.push(Value::option_some(variant.clone()));
        out.push(Value::list(ValueSeq::from([variant, Value::option_none()])));
        out
    }

    fn check<H: HashPolicy>() {
        let values = every_kind::<H>();
        for v in &values {
            let one = std::slice::from_ref(v);
            assert_eq!(payload_to_json_string(one), tree_text(one));
        }
        assert_eq!(payload_to_json_string(&values), tree_text(&values));
        assert_eq!(payload_to_json_string::<H>(&[]), tree_text::<H>(&[]));
        let node_text = payload_to_json_string::<H>(&[Value::node(node(7, 2))]);
        assert_eq!(node_text, r#"[{"type":"VNode","value":{"index":2,"role":7}}]"#);
    }

    #[test]
    fn streamed_payload_is_byte_identical_to_the_json_tree_text() {
        check::<NoHashing>();
        check::<WithHashing>();
    }
}
