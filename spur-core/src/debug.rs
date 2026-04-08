use anyhow::{Result, anyhow};
use arrow::array::{Array, AsArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs;
use std::path::{Path, PathBuf};

pub struct SimulatorDebugger {
    /// Only Parquet mode is supported now.
    parquet_dir: PathBuf,
}

impl SimulatorDebugger {
    /// Connects to a Parquet directory setup.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Err(anyhow!(
                "Debug path must be a directory containing parquet files"
            ));
        }
        Ok(Self {
            parquet_dir: path.to_path_buf(),
        })
    }

    /// Fetches all logs for a specific node, ordered by simulation step.
    pub fn get_node_timeline(&self, run_id: i64, node_id: i64) -> Result<Vec<(i32, String)>> {
        let dir = self.parquet_dir.join("logs");
        let batches = read_all_batches(&dir)?;

        let mut results = Vec::new();
        for batch in batches {
            let run_id_arr = batch
                .column_by_name("run_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let node_id_arr = batch
                .column_by_name("node_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let step_arr = batch
                .column_by_name("step")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int32Type>();
            let content_arr = batch.column_by_name("content").unwrap().as_string::<i32>();

            for i in 0..batch.num_rows() {
                if run_id_arr.value(i) == run_id && node_id_arr.value(i) == node_id {
                    results.push((step_arr.value(i), content_arr.value(i).to_string()));
                }
            }
        }
        results.sort_by_key(|(step, _)| *step);
        Ok(results)
    }

    /// Fetches all logs for a specific run, ordered by simulation step.
    pub fn get_all_logs(&self, run_id: i64) -> Result<Vec<(i32, Option<i64>, String)>> {
        let dir = self.parquet_dir.join("logs");
        let batches = read_all_batches(&dir)?;

        let mut results = Vec::new();
        for batch in batches {
            let run_id_arr = batch
                .column_by_name("run_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let node_id_arr = batch
                .column_by_name("node_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let step_arr = batch
                .column_by_name("step")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int32Type>();
            let content_arr = batch.column_by_name("content").unwrap().as_string::<i32>();

            for i in 0..batch.num_rows() {
                if run_id_arr.value(i) == run_id {
                    let node_id_val = if node_id_arr.is_valid(i) {
                        Some(node_id_arr.value(i))
                    } else {
                        None
                    };
                    results.push((
                        step_arr.value(i),
                        node_id_val,
                        content_arr.value(i).to_string(),
                    ));
                }
            }
        }
        results.sort_by_key(|(step, _, _)| *step);
        Ok(results)
    }

    /// Returns a summary of a run (count of invocations, crashes, etc.)
    pub fn get_run_summary(&self, run_id: i64) -> Result<std::collections::HashMap<String, i64>> {
        let dir = self.parquet_dir.join("executions");
        let batches = read_all_batches(&dir)?;

        let mut summary = std::collections::HashMap::new();
        for batch in batches {
            let run_id_arr = batch
                .column_by_name("run_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let kind_arr = batch.column_by_name("kind").unwrap().as_string::<i32>();

            for i in 0..batch.num_rows() {
                if run_id_arr.value(i) == run_id {
                    let kind = kind_arr.value(i).to_string();
                    *summary.entry(kind).or_insert(0) += 1;
                }
            }
        }
        Ok(summary)
    }

    /// Fetches a combined, interleaved timeline of executions, logs, and traces
    /// for a given run, ordered by simulation step.
    pub fn get_combined_timeline(&self, run_id: i64) -> Result<Vec<CombinedEvent>> {
        let mut temp = Vec::new();

        // executions
        let batches = read_all_batches(&self.parquet_dir.join("executions"))?;
        for batch in batches {
            let run_id_arr = batch
                .column_by_name("run_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let step_arr = batch
                .column_by_name("step")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int32Type>();
            let client_id_arr = batch
                .column_by_name("client_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let seq_num_arr = batch
                .column_by_name("seq_num")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let kind_arr = batch.column_by_name("kind").unwrap().as_string::<i32>();
            let action_arr = batch.column_by_name("action").unwrap().as_string::<i32>();

            for i in 0..batch.num_rows() {
                if run_id_arr.value(i) == run_id {
                    temp.push(TempEvent {
                        step: step_arr.value(i),
                        source: "Execution".to_string(),
                        node_id: Some(client_id_arr.value(i)),
                        description: format!("{}: {}", kind_arr.value(i), action_arr.value(i)),
                        seq_num: seq_num_arr.value(i),
                    });
                }
            }
        }

        // logs
        let batches = read_all_batches(&self.parquet_dir.join("logs"))?;
        for batch in batches {
            let run_id_arr = batch
                .column_by_name("run_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let step_arr = batch
                .column_by_name("step")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int32Type>();
            let node_id_arr = batch
                .column_by_name("node_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let seq_num_arr = batch
                .column_by_name("seq_num")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let content_arr = batch.column_by_name("content").unwrap().as_string::<i32>();

            for i in 0..batch.num_rows() {
                if run_id_arr.value(i) == run_id {
                    temp.push(TempEvent {
                        step: step_arr.value(i),
                        source: "Log".to_string(),
                        node_id: Some(node_id_arr.value(i)),
                        description: content_arr.value(i).to_string(),
                        seq_num: seq_num_arr.value(i),
                    });
                }
            }
        }

        // traces
        let batches = read_all_batches(&self.parquet_dir.join("traces"))?;
        for batch in batches {
            let run_id_arr = batch
                .column_by_name("run_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let step_arr = batch
                .column_by_name("step")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int32Type>();
            let node_id_arr = batch
                .column_by_name("node_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let seq_num_arr = batch
                .column_by_name("seq_num")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();

            let function_name_arr = batch
                .column_by_name("function_name")
                .unwrap()
                .as_string::<i32>();
            let trace_kind_arr = batch
                .column_by_name("trace_kind")
                .unwrap()
                .as_string::<i32>();
            let trace_id_arr = batch
                .column_by_name("trace_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let schedulable_count_arr = batch
                .column_by_name("schedulable_count")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let causal_operation_id_arr = batch
                .column_by_name("causal_operation_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();

            for i in 0..batch.num_rows() {
                if run_id_arr.value(i) == run_id {
                    let cop = if causal_operation_id_arr.is_valid(i) {
                        format!(" [cop={}]", causal_operation_id_arr.value(i))
                    } else {
                        "".to_string()
                    };

                    let desc = format!(
                        "{} {} [tid={}] [sched={}]{}",
                        trace_kind_arr.value(i),
                        function_name_arr.value(i),
                        trace_id_arr.value(i),
                        schedulable_count_arr.value(i),
                        cop
                    );

                    temp.push(TempEvent {
                        step: step_arr.value(i),
                        source: "Trace".to_string(),
                        node_id: Some(node_id_arr.value(i)),
                        description: desc,
                        seq_num: seq_num_arr.value(i),
                    });
                }
            }
        }

        temp.sort_unstable_by(|a, b| {
            a.step
                .cmp(&b.step)
                .then(a.source.cmp(&b.source))
                .then(a.seq_num.cmp(&b.seq_num))
        });

        Ok(temp
            .into_iter()
            .map(|e| CombinedEvent {
                step: e.step,
                source: e.source,
                node_id: e.node_id,
                description: e.description,
            })
            .collect())
    }

    /// Fetches all traces for a specific run, and optionally a specific node.
    pub fn get_traces(&self, run_id: i64, node_id: Option<i64>) -> Result<Vec<TraceEvent>> {
        let dir = self.parquet_dir.join("traces");
        let batches = read_all_batches(&dir)?;

        struct TempTrace {
            event: TraceEvent,
            seq_num: i64,
        }

        let mut results = Vec::new();
        for batch in batches {
            let run_id_arr = batch
                .column_by_name("run_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let node_id_arr = batch
                .column_by_name("node_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let seq_num_arr = batch
                .column_by_name("seq_num")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();

            let step_arr = batch
                .column_by_name("step")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int32Type>();
            let function_name_arr = batch
                .column_by_name("function_name")
                .unwrap()
                .as_string::<i32>();
            let trace_kind_arr = batch
                .column_by_name("trace_kind")
                .unwrap()
                .as_string::<i32>();
            let trace_id_arr = batch
                .column_by_name("trace_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let payload_arr = batch.column_by_name("payload").unwrap().as_string::<i32>();
            let schedulable_count_arr = batch
                .column_by_name("schedulable_count")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();
            let causal_operation_id_arr = batch
                .column_by_name("causal_operation_id")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>();

            for i in 0..batch.num_rows() {
                if run_id_arr.value(i) == run_id {
                    let n_id = node_id_arr.value(i);
                    if let Some(target_node) = node_id {
                        if n_id != target_node {
                            continue;
                        }
                    }

                    let cop = if causal_operation_id_arr.is_valid(i) {
                        Some(causal_operation_id_arr.value(i))
                    } else {
                        None
                    };

                    results.push(TempTrace {
                        event: TraceEvent {
                            step: step_arr.value(i),
                            node_id: n_id,
                            trace_id: trace_id_arr.value(i),
                            function_name: function_name_arr.value(i).to_string(),
                            trace_kind: trace_kind_arr.value(i).to_string(),
                            payload: payload_arr.value(i).to_string(),
                            schedulable_count: schedulable_count_arr.value(i),
                            causal_operation_id: cop,
                        },
                        seq_num: seq_num_arr.value(i),
                    });
                }
            }
        }

        results.sort_unstable_by(|a, b| {
            a.event
                .step
                .cmp(&b.event.step)
                .then(a.seq_num.cmp(&b.seq_num))
        });

        Ok(results.into_iter().map(|t| t.event).collect())
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

struct TempEvent {
    step: i32,
    source: String,
    node_id: Option<i64>,
    description: String,
    seq_num: i64,
}

fn read_all_batches(dir: &Path) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    if !dir.exists() {
        return Ok(batches);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("parquet") {
            let file = fs::File::open(&path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let reader = builder.build()?;
            for batch_result in reader {
                batches.push(batch_result?);
            }
        }
    }
    Ok(batches)
}
