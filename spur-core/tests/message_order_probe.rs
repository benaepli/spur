//! Reads a trace directory written by `spur explore` and reports whether the
//! simulator ever delivers two messages between the same ordered pair of nodes
//! in an order other than the one they were sent in.
//!
//! The trace already pairs a send with its delivery: a `Dispatch` row is
//! emitted on the sending node immediately before the message is queued, and
//! the handler that eventually runs it emits an `Enter` row carrying the same
//! `trace_id` on the receiving node. Within a run, `seq_num` is the emission
//! index, so it totally orders both events.
//!
//! Point `SPUR_TRACE_DIR` at the exploration output directory. Without it the
//! probe looks beside the repository for a previous run and reports nothing
//! when there is none, so the suite stays runnable with no exploration on disk.

#![cfg(feature = "simulator")]

use arrow::array::AsArray;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_TRACE_DIR: &str = "../../tmp/loop/fifo-probe";

/// One message, assembled from the two rows that mention its `trace_id`.
#[derive(Default)]
struct Message {
    sender: Option<i64>,
    send_order: Option<i64>,
    receiver: Option<i64>,
    deliver_order: Option<i64>,
    /// Deliveries beyond the first. A record can be handed to a node more than
    /// once when the node crashes mid-handler.
    extra_deliveries: u32,
}

#[derive(Default)]
struct Totals {
    runs: usize,
    dispatches: usize,
    remote_deliveries: usize,
    redelivered: usize,
    undelivered: usize,
    comparable_pairs: u64,
    inverted_pairs: u64,
    /// Consecutively sent messages on one channel, and how many of those pairs
    /// arrived swapped. Unlike the all-pairs count this does not shrink as a
    /// channel gets longer.
    adjacent_pairs: u64,
    adjacent_inversions: u64,
    /// Deliveries that arrived before a message sent earlier on the same
    /// channel.
    overtaking_deliveries: u64,
    channels: usize,
    channels_with_inversion: usize,
    runs_with_inversion: usize,
}

fn read_batches(dir: &Path) -> Vec<RecordBatch> {
    let mut batches = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return batches;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        let file = fs::File::open(&path).expect("open parquet file");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("read parquet footer")
            .build()
            .expect("build parquet reader");
        for batch in reader {
            batches.push(batch.expect("decode parquet batch"));
        }
    }
    batches
}

/// Groups every message of every run by `trace_id`.
fn collect_messages(batches: &[RecordBatch]) -> HashMap<i64, HashMap<i64, Message>> {
    let mut runs: HashMap<i64, HashMap<i64, Message>> = HashMap::new();
    for batch in batches {
        let run_ids = batch
            .column_by_name("run_id")
            .expect("run_id column")
            .as_primitive::<arrow::datatypes::Int64Type>();
        let seq_nums = batch
            .column_by_name("seq_num")
            .expect("seq_num column")
            .as_primitive::<arrow::datatypes::Int64Type>();
        let node_ids = batch
            .column_by_name("node_id")
            .expect("node_id column")
            .as_primitive::<arrow::datatypes::Int64Type>();
        let trace_ids = batch
            .column_by_name("trace_id")
            .expect("trace_id column")
            .as_primitive::<arrow::datatypes::Int64Type>();
        let kinds = batch
            .column_by_name("trace_kind")
            .expect("trace_kind column")
            .as_string::<i32>();

        for i in 0..batch.num_rows() {
            let kind = kinds.value(i);
            if kind != "Dispatch" && kind != "Enter" {
                continue;
            }
            let message = runs
                .entry(run_ids.value(i))
                .or_default()
                .entry(trace_ids.value(i))
                .or_default();
            let node = node_ids.value(i);
            let seq = seq_nums.value(i);
            if kind == "Dispatch" {
                message.sender = Some(node);
                message.send_order = Some(seq);
            } else if message.deliver_order.is_none() {
                message.receiver = Some(node);
                message.deliver_order = Some(seq);
            } else {
                message.extra_deliveries += 1;
            }
        }
    }
    runs
}

fn tally(runs: &HashMap<i64, HashMap<i64, Message>>) -> Totals {
    let mut totals = Totals {
        runs: runs.len(),
        ..Totals::default()
    };

    for messages in runs.values() {
        // Send/deliver index pairs, keyed by the ordered node pair they crossed.
        let mut channels: HashMap<(i64, i64), Vec<(i64, i64)>> = HashMap::new();
        for message in messages.values() {
            let Some(send_order) = message.send_order else {
                continue;
            };
            totals.dispatches += 1;
            totals.redelivered += message.extra_deliveries as usize;
            let (Some(sender), Some(receiver), Some(deliver_order)) =
                (message.sender, message.receiver, message.deliver_order)
            else {
                totals.undelivered += 1;
                continue;
            };
            if sender == receiver {
                continue;
            }
            totals.remote_deliveries += 1;
            channels
                .entry((sender, receiver))
                .or_default()
                .push((send_order, deliver_order));
        }

        let mut run_inversions = 0u64;
        for mut pairs in channels.into_values() {
            if pairs.len() < 2 {
                continue;
            }
            totals.channels += 1;
            pairs.sort_by_key(|(send_order, _)| *send_order);
            let mut inversions = 0u64;
            for a in 0..pairs.len() {
                for b in (a + 1)..pairs.len() {
                    totals.comparable_pairs += 1;
                    if pairs[a].1 > pairs[b].1 {
                        inversions += 1;
                    }
                }
            }
            totals.adjacent_pairs += pairs.len() as u64 - 1;
            let mut latest_delivery = i64::MIN;
            for window in pairs.windows(2) {
                if window[0].1 > window[1].1 {
                    totals.adjacent_inversions += 1;
                }
            }
            for (_, deliver_order) in &pairs {
                if *deliver_order < latest_delivery {
                    totals.overtaking_deliveries += 1;
                } else {
                    latest_delivery = *deliver_order;
                }
            }
            if inversions > 0 {
                totals.channels_with_inversion += 1;
            }
            run_inversions += inversions;
        }
        totals.inverted_pairs += run_inversions;
        if run_inversions > 0 {
            totals.runs_with_inversion += 1;
        }
    }
    totals
}

fn percent(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        100.0 * numerator as f64 / denominator as f64
    }
}

#[test]
fn report_out_of_order_delivery_between_node_pairs() {
    let dir = std::env::var("SPUR_TRACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_TRACE_DIR));
    let traces = dir.join("traces");
    if !traces.is_dir() {
        println!("message-order probe: no traces at {}", traces.display());
        return;
    }

    let batches = read_batches(&traces);
    let mut rows = 0usize;
    let mut by_kind: HashMap<String, usize> = HashMap::new();
    let mut by_function: HashMap<String, usize> = HashMap::new();
    for batch in &batches {
        rows += batch.num_rows();
        let kinds = batch
            .column_by_name("trace_kind")
            .expect("trace_kind column")
            .as_string::<i32>();
        let functions = batch
            .column_by_name("function_name")
            .expect("function_name column")
            .as_string::<i32>();
        for i in 0..batch.num_rows() {
            *by_kind.entry(kinds.value(i).to_string()).or_default() += 1;
            *by_function
                .entry(format!("{} {}", kinds.value(i), functions.value(i)))
                .or_default() += 1;
        }
    }
    let runs = collect_messages(&batches);
    let t = tally(&runs);

    println!("message-order probe over {}", traces.display());
    println!("  trace rows               {}", rows);
    let mut kind_counts: Vec<_> = by_kind.into_iter().collect();
    kind_counts.sort();
    for (kind, count) in kind_counts {
        println!("    {kind:<10} {count}");
    }
    let mut function_counts: Vec<_> = by_function.into_iter().collect();
    function_counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (name, count) in function_counts.iter().take(24) {
        println!("    {name:<44} {count}");
    }
    println!("  runs                     {}", t.runs);
    println!("  messages sent            {}", t.dispatches);
    println!("  delivered to another node {}", t.remote_deliveries);
    println!("  never delivered          {}", t.undelivered);
    println!("  delivered more than once {}", t.redelivered);
    println!(
        "  node pairs carrying >=2 messages in one run {}",
        t.channels
    );
    println!(
        "  of those, at least one out-of-order pair    {} ({:.3}%)",
        t.channels_with_inversion,
        percent(t.channels_with_inversion as u64, t.channels as u64)
    );
    println!(
        "  message pairs comparable {}, out of order {} ({:.3}%)",
        t.comparable_pairs,
        t.inverted_pairs,
        percent(t.inverted_pairs, t.comparable_pairs)
    );
    println!(
        "  consecutively sent pairs {}, arrived swapped {} ({:.3}%)",
        t.adjacent_pairs,
        t.adjacent_inversions,
        percent(t.adjacent_inversions, t.adjacent_pairs)
    );
    println!(
        "  deliveries that overtook an earlier send {} ({:.3}% of deliveries)",
        t.overtaking_deliveries,
        percent(t.overtaking_deliveries, t.remote_deliveries as u64)
    );
    println!(
        "  runs with >=1 out-of-order pair {} ({:.3}%)",
        t.runs_with_inversion,
        percent(t.runs_with_inversion as u64, t.runs as u64)
    );
}
