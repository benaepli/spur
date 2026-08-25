//! Opt-in, per-run census of scheduling choices.
//!
//! For every scheduling step the census records which event class was taken and
//! which classes had at least one eligible item but were not taken. Enabled per
//! explorer session by the CLI and dumped to `<output_dir>/depth_census.json`,
//! keyed by run id so it can be joined against any per-run grade computed
//! elsewhere. Recording is observation-only: it never affects scheduling,
//! scoring, or RNG consumption, and when disabled every probe is a single
//! relaxed atomic load.

use crate::simulator::core::RunnableCategory;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const CLASS_COUNT: usize = 7;

/// Index order for every per-class array in this module.
pub const CLASS_NAMES: [&str; CLASS_COUNT] = [
    "record",
    "timer",
    "channel_send",
    "crash",
    "recover",
    "partition",
    "heal",
];

/// Steps of a run that take a class other than `record` are listed
/// individually; beyond this many the count in `chosen` is the only record.
const CONTROL_EVENT_CAP: usize = 64;

/// Runs beyond this many are counted in `runs_dropped` instead of stored.
const RUN_CAP: usize = 50_000;

fn class_index(category: RunnableCategory) -> usize {
    match category {
        RunnableCategory::Record => 0,
        RunnableCategory::Timer => 1,
        RunnableCategory::ChannelSend => 2,
        RunnableCategory::Crash => 3,
        RunnableCategory::Recover => 4,
        RunnableCategory::Partition => 5,
        RunnableCategory::Heal => 6,
    }
}

/// The set of event classes with at least one eligible item at a step.
#[derive(Clone, Copy, Default)]
pub struct ClassSet(u8);

impl ClassSet {
    pub fn insert(&mut self, category: RunnableCategory) {
        self.0 |= 1 << class_index(category);
    }

    fn contains(self, index: usize) -> bool {
        self.0 & (1 << index) != 0
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static RUNS_DROPPED: AtomicU64 = AtomicU64::new(0);
static RUNS: Mutex<Vec<RunCensus>> = Mutex::new(Vec::new());

thread_local! {
    static CURRENT: RefCell<Option<RunCensus>> = const { RefCell::new(None) };
}

#[derive(Clone, Serialize)]
pub struct ControlEvent {
    pub step: u32,
    pub class: &'static str,
}

#[derive(Clone)]
struct RunCensus {
    run_id: i64,
    steps: u32,
    outcome: &'static str,
    hit_iteration_cap: bool,
    chosen: [u32; CLASS_COUNT],
    offered: [u32; CLASS_COUNT],
    offered_unchosen: [u32; CLASS_COUNT],
    /// Step at which a class first became eligible; `u32::MAX` means never.
    first_offered_step: [u32; CLASS_COUNT],
    control_events: Vec<ControlEvent>,
}

impl RunCensus {
    fn new(run_id: i64) -> Self {
        Self {
            run_id,
            steps: 0,
            outcome: "unfinished",
            hit_iteration_cap: false,
            chosen: [0; CLASS_COUNT],
            offered: [0; CLASS_COUNT],
            offered_unchosen: [0; CLASS_COUNT],
            first_offered_step: [u32::MAX; CLASS_COUNT],
            control_events: Vec::new(),
        }
    }
}

/// Enable or disable recording for this explorer session. Enabling clears
/// anything a previous session in the same process collected.
pub fn set_enabled(on: bool) {
    if on {
        RUNS_DROPPED.store(0, Ordering::Relaxed);
        if let Ok(mut runs) = RUNS.lock() {
            runs.clear();
        }
    }
    ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Start collecting for `run_id` on this thread. A run executes entirely on one
/// thread, so the in-progress census is thread-local.
pub fn begin_run(run_id: i64) {
    if !enabled() {
        return;
    }
    CURRENT.with(|c| *c.borrow_mut() = Some(RunCensus::new(run_id)));
}

/// Record one scheduling step: `chosen` was taken, `offered` is the set of
/// classes that had an eligible item at that step.
pub fn record_step(chosen: RunnableCategory, offered: ClassSet) {
    if !enabled() {
        return;
    }
    let chosen_index = class_index(chosen);
    CURRENT.with(|c| {
        let mut slot = c.borrow_mut();
        let Some(run) = slot.as_mut() else {
            return;
        };
        let step = run.steps;
        run.steps += 1;
        run.chosen[chosen_index] += 1;
        for index in 0..CLASS_COUNT {
            if !offered.contains(index) {
                continue;
            }
            run.offered[index] += 1;
            if run.first_offered_step[index] == u32::MAX {
                run.first_offered_step[index] = step;
            }
            if index != chosen_index {
                run.offered_unchosen[index] += 1;
            }
        }
        if chosen_index != 0 && run.control_events.len() < CONTROL_EVENT_CAP {
            run.control_events.push(ControlEvent {
                step,
                class: CLASS_NAMES[chosen_index],
            });
        }
    });
}

/// Mark the run in progress on this thread as having run out of iterations
/// before its plan finished.
pub fn note_iteration_cap() {
    if !enabled() {
        return;
    }
    CURRENT.with(|c| {
        if let Some(run) = c.borrow_mut().as_mut() {
            run.hit_iteration_cap = true;
        }
    });
}

/// Close the run in progress on this thread and file it under `outcome`
/// ("completed", "deadlock", or "failed").
pub fn finish_run(outcome: &'static str) {
    if !enabled() {
        return;
    }
    let Some(mut run) = CURRENT.with(|c| c.borrow_mut().take()) else {
        return;
    };
    run.outcome = outcome;
    let Ok(mut runs) = RUNS.lock() else {
        return;
    };
    if runs.len() >= RUN_CAP {
        RUNS_DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    runs.push(run);
}

#[derive(Serialize)]
pub struct RunReport {
    pub run_id: i64,
    pub steps: u32,
    pub outcome: &'static str,
    pub hit_iteration_cap: bool,
    pub chosen: BTreeMap<&'static str, u32>,
    pub offered: BTreeMap<&'static str, u32>,
    pub offered_unchosen: BTreeMap<&'static str, u32>,
    pub first_offered_step: BTreeMap<&'static str, u32>,
    pub control_events: Vec<ControlEvent>,
}

#[derive(Serialize)]
pub struct CensusSnapshot {
    pub runs_recorded: usize,
    pub runs_dropped: u64,
    pub runs_hit_iteration_cap: usize,
    pub run_cap: usize,
    pub control_event_cap: usize,
    pub totals_chosen: BTreeMap<&'static str, u64>,
    pub totals_offered: BTreeMap<&'static str, u64>,
    pub totals_offered_unchosen: BTreeMap<&'static str, u64>,
    pub outcomes: BTreeMap<&'static str, u64>,
    pub per_run: Vec<RunReport>,
}

fn nonzero_map(counts: &[u32; CLASS_COUNT]) -> BTreeMap<&'static str, u32> {
    CLASS_NAMES
        .iter()
        .zip(counts.iter())
        .filter(|&(_, &count)| count > 0)
        .map(|(&name, &count)| (name, count))
        .collect()
}

pub fn snapshot() -> CensusSnapshot {
    let runs = match RUNS.lock() {
        Ok(runs) => runs.clone(),
        Err(_) => Vec::new(),
    };
    let mut totals_chosen = BTreeMap::new();
    let mut totals_offered = BTreeMap::new();
    let mut totals_offered_unchosen = BTreeMap::new();
    let mut outcomes: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut per_run = Vec::with_capacity(runs.len());

    for run in &runs {
        for (index, &name) in CLASS_NAMES.iter().enumerate() {
            *totals_chosen.entry(name).or_insert(0u64) += run.chosen[index] as u64;
            *totals_offered.entry(name).or_insert(0u64) += run.offered[index] as u64;
            *totals_offered_unchosen.entry(name).or_insert(0u64) +=
                run.offered_unchosen[index] as u64;
        }
        *outcomes.entry(run.outcome).or_insert(0) += 1;
        let first_offered = CLASS_NAMES
            .iter()
            .zip(run.first_offered_step.iter())
            .filter(|&(_, &step)| step != u32::MAX)
            .map(|(&name, &step)| (name, step))
            .collect();
        per_run.push(RunReport {
            run_id: run.run_id,
            steps: run.steps,
            outcome: run.outcome,
            hit_iteration_cap: run.hit_iteration_cap,
            chosen: nonzero_map(&run.chosen),
            offered: nonzero_map(&run.offered),
            offered_unchosen: nonzero_map(&run.offered_unchosen),
            first_offered_step: first_offered,
            control_events: run.control_events.clone(),
        });
    }
    per_run.sort_by_key(|report| report.run_id);

    CensusSnapshot {
        runs_recorded: runs.len(),
        runs_dropped: RUNS_DROPPED.load(Ordering::Relaxed),
        runs_hit_iteration_cap: runs.iter().filter(|run| run.hit_iteration_cap).count(),
        run_cap: RUN_CAP,
        control_event_cap: CONTROL_EVENT_CAP,
        totals_chosen,
        totals_offered,
        totals_offered_unchosen,
        outcomes,
        per_run,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test for the whole module: the enable flag is process-wide, so
    /// separate test functions would race each other.
    #[test]
    fn records_only_when_enabled_and_counts_unchosen_classes() {
        set_enabled(false);
        begin_run(1);
        let mut offered = ClassSet::default();
        offered.insert(RunnableCategory::Record);
        record_step(RunnableCategory::Record, offered);
        finish_run("completed");
        assert_eq!(snapshot().runs_recorded, 0);

        set_enabled(true);
        begin_run(7);
        let mut offered = ClassSet::default();
        offered.insert(RunnableCategory::Record);
        offered.insert(RunnableCategory::Crash);
        record_step(RunnableCategory::Record, offered);
        finish_run("completed");
        let snap = snapshot();
        set_enabled(false);
        assert_eq!(snap.runs_recorded, 1);
        assert_eq!(snap.totals_chosen.get("record"), Some(&1));
        assert_eq!(snap.totals_offered.get("crash"), Some(&1));
        assert_eq!(snap.totals_offered_unchosen.get("crash"), Some(&1));
        assert_eq!(snap.totals_offered_unchosen.get("record"), Some(&0));
    }
}
