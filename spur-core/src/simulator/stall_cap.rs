//! A stall cap for runs, learned across a session and shared by every
//! worker. A run makes progress at a step when the step appends a client or
//! fault row to the history, when a dispatched handler or timer leaves its
//! node's state changed, or when the plan releases something; the steps
//! between two such marks are a quiet gap. Every completed run-cap probe
//! reports its longest quiet gap, bucketed by the run's configured step
//! budget, and a treated run ends once its open gap exceeds a high quantile
//! of those gaps with headroom. The learner has the shape of the step cap's:
//! the cap is recomputed only when a scope's completed count crosses a
//! doubling checkpoint (200, 400, 800, ...) and is constant in between, so it
//! is a deterministic function of the sample sequence.
//!
//! Three of four runs outside the probe streams are treated. Run-cap probes
//! and timer-context probes are never cut, so every learner sees exactly the
//! runs it would see without the cap. An untreated run keeps the same clock
//! and reports its longest gap against the standing cap, so whether the cap
//! would cut a run that still progresses is readable off the untreated
//! quarter without cutting it.

use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::timer_context;
use crate::simulator::util_stats::{self, StallCapCell, StallCapMarks, StallCapRun};
use dashmap::DashMap;
use std::sync::{LazyLock, Mutex};

/// Domain salt for the treated split, so it is independent of every other
/// salted split and of the probe phases.
pub const STALL_CAP_SALT: u64 = 0x_5354_414C_4C43_4150; // "STALLCAP"

/// Phases the treated split is drawn over; phase zero is the untreated
/// quarter.
const TREATED_PERIOD: i64 = 4;

/// Quantile of the completed-probe gap distribution the cap is read from.
const QUANTILE: f64 = 0.99;

/// Multiplier applied to the quantile so a gap slightly longer than the
/// quantile still passes.
const HEADROOM: f64 = 1.5;

/// Completed probes a scope must accumulate before its cap takes effect.
/// Below the floor no run is cut.
const MIN_COMPLETED_SAMPLES: u64 = 200;

/// Histogram cells per scope. Gaps are bucketed by a per-scope width so the
/// full budget fits; a gap is never longer than the run, so the last cell
/// never saturates.
const HIST_CELLS: usize = 256;

/// Per-run rows kept for the untreated quarter; rows past the cap are
/// counted as dropped.
const ROW_CAP: usize = 1_000_000;

struct ScopeAccum {
    /// Steps per histogram cell, fixed when the scope is created.
    bucket_width: u32,
    /// Cell `c` counts the completed probes whose longest gap fell in
    /// `[c * bucket_width, (c + 1) * bucket_width)`.
    hist: [u32; HIST_CELLS],
    /// Completed probes folded into the histogram.
    completed: u64,
    /// The cap set at the last checkpoint, governing every run until the
    /// next one; None until the first checkpoint is crossed.
    current: Option<i32>,
    /// Completed count at which the cap is next recomputed; doubles after
    /// each recompute.
    next_checkpoint: u64,
}

impl ScopeAccum {
    fn new(backup: i32) -> Self {
        Self {
            bucket_width: ((backup.max(1) + HIST_CELLS as i32 - 1) / HIST_CELLS as i32).max(1)
                as u32,
            hist: [0; HIST_CELLS],
            completed: 0,
            current: None,
            next_checkpoint: MIN_COMPLETED_SAMPLES,
        }
    }

    /// The cap the histogram supports right now, or None while it is below
    /// the sample floor. The quantile is Laplace-smoothed and read at the
    /// winning cell's upper edge, so bucketing only ever rounds the cap up.
    fn estimate(&self, backup: i32) -> Option<i32> {
        if self.completed < MIN_COMPLETED_SAMPLES {
            return None;
        }
        let samples: u64 = self.hist.iter().map(|&n| n as u64).sum();
        if samples == 0 {
            return None;
        }
        let denom = (samples + 2) as f64;
        let mut cum: u64 = 0;
        for (c, &n) in self.hist.iter().enumerate() {
            cum += n as u64;
            if (cum + 1) as f64 / denom >= QUANTILE {
                let upper = (c as i64 + 1) * self.bucket_width as i64 - 1;
                let capped = (HEADROOM * upper as f64).ceil() as i64;
                return Some(capped.min(backup as i64) as i32);
            }
        }
        None
    }
}

static TABLE: LazyLock<DashMap<i32, ScopeAccum>> = LazyLock::new(DashMap::new);

/// One row per untreated run: the gap the mechanism measured and the cap it
/// would have been held to, keyed by the run id a measurement taken outside
/// the simulator is joined on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunRow {
    pub run_id: i64,
    pub longest_quiet_gap: i32,
    /// The cap standing for the run's scope when it ended, or zero while the
    /// scope was below its sample floor.
    pub stall_cap_standing: i32,
}

static ROWS: Mutex<Vec<RunRow>> = Mutex::new(Vec::new());

/// The cell a run id lands in. Both probe streams are exempt so every
/// learner they feed sees the runs it sees without the cap.
pub fn cell(run_id: i64) -> StallCapCell {
    if run_cap::is_probe(run_id)
        || timer_context::run_mode(run_id) == timer_context::RunMode::Probe
    {
        StallCapCell::Probe
    } else if run_phase::salted_phase(run_id, STALL_CAP_SALT, TREATED_PERIOD) != 0 {
        StallCapCell::Treated
    } else {
        StallCapCell::Untreated
    }
}

/// Whether the run is cut once its clock exceeds the cap.
pub fn is_treated(run_id: i64) -> bool {
    cell(run_id) == StallCapCell::Treated
}

/// The stall cap for runs whose configured budget is `backup`, or None while
/// the scope has crossed no checkpoint. Read once at run start so the bound
/// is frozen for the whole run.
pub fn effective_cap(backup: i32) -> Option<i32> {
    TABLE.get(&backup).and_then(|acc| acc.current)
}

/// Fold one completed run-cap probe's longest quiet gap into its scope. The
/// cap it is checked against is the one in effect before it merges, so a
/// completion the cap would have cut is visible.
pub fn merge_probe(backup: i32, longest_gap: i32) {
    {
        let mut acc = TABLE.entry(backup).or_insert_with(|| ScopeAccum::new(backup));
        let over_cap = acc
            .current
            .is_some_and(|cap| cap < backup && longest_gap > cap);
        util_stats::record_stall_cap_probe(over_cap);
        let cell = ((longest_gap.max(0) as u32) / acc.bucket_width).min(HIST_CELLS as u32 - 1);
        acc.hist[cell as usize] = acc.hist[cell as usize].saturating_add(1);
        acc.completed += 1;
        if acc.completed >= acc.next_checkpoint {
            acc.current = acc.estimate(backup);
            acc.next_checkpoint = acc.next_checkpoint.saturating_mul(2);
        }
    }
    publish_gauges();
}

/// Scale every scope's mass by `factor`, dropping scopes that reach zero,
/// so stale phases of a long exploration lose their vote. The cap and the
/// next checkpoint are left alone: shrinking the completed count delays the
/// next crossing, and the recompute there reads the decay-weighted
/// histogram, so the cap leans toward recent phases.
pub fn decay(factor: f64) {
    let factor = factor.clamp(0.0, 1.0);
    TABLE.retain(|_, acc| {
        for n in acc.hist.iter_mut() {
            *n = ((*n as f64) * factor).floor() as u32;
        }
        acc.completed = ((acc.completed as f64) * factor).floor() as u64;
        acc.current.is_some() || acc.completed > 0 || acc.hist.iter().any(|&n| n > 0)
    });
    publish_gauges();
}

/// Clear the table and the rows so explorer sessions in one process do not
/// share caps.
pub fn reset() {
    TABLE.clear();
    if let Ok(mut rows) = ROWS.lock() {
        rows.clear();
    }
    util_stats::set_stall_cap_learned(0, 0);
}

fn publish_gauges() {
    let mut learned: u64 = 0;
    let mut max_scope: Option<(i32, i32)> = None;
    for e in TABLE.iter() {
        let backup = *e.key();
        if let Some(cap) = e.value().current {
            learned += 1;
            if max_scope.map_or(true, |(k, _)| backup > k) {
                max_scope = Some((backup, cap));
            }
        }
    }
    util_stats::set_stall_cap_learned(learned, max_scope.map_or(0, |(_, c)| c.max(0) as u64));
}

/// What one step did that counts as progress. A step can carry several
/// kinds at once; each is counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Marks {
    /// A history row other than a timer firing was appended.
    pub row: bool,
    /// A dispatched delivery left its node's state changed.
    pub acted_delivery: bool,
    /// A timer firing, or the segment it woke, left its node's state
    /// changed.
    pub acted_timer: bool,
    /// The plan released an event or a held client request was issued.
    pub release: bool,
}

impl Marks {
    pub fn any(&self) -> bool {
        self.row || self.acted_delivery || self.acted_timer || self.release
    }
}

/// The per-run clock: steps since the last progress mark, and the longest
/// such gap the run has had. A suspended step is one the run spends waiting
/// on a bounded release the scheduler itself owns; it counts toward neither.
#[derive(Clone, Debug, Default)]
pub struct RunClock {
    gap: i32,
    longest: i32,
    suspended_steps: u64,
    marks: StallCapMarks,
}

impl RunClock {
    /// Fold one step and return the open gap after it. A marked step closes
    /// the gap whether or not it was suspended.
    pub fn step(&mut self, marks: Marks, suspended: bool) -> i32 {
        if marks.any() {
            self.longest = self.longest.max(self.gap);
            self.gap = 0;
            self.marks.rows += marks.row as u64;
            self.marks.acted_deliveries += marks.acted_delivery as u64;
            self.marks.acted_timers += marks.acted_timer as u64;
            self.marks.releases += marks.release as u64;
        } else if suspended {
            self.suspended_steps += 1;
        } else {
            self.gap += 1;
        }
        self.gap
    }

    /// The longest gap so far, counting the open segment.
    pub fn longest(&self) -> i32 {
        self.longest.max(self.gap)
    }

    pub fn suspended_steps(&self) -> u64 {
        self.suspended_steps
    }

    pub fn marks(&self) -> StallCapMarks {
        self.marks.clone()
    }
}

/// How one run ended, as far as the stall cap is concerned.
pub struct RunEnding {
    pub run_id: i64,
    pub backup: i32,
    /// Every planned event completed, so a probe's gap can feed the learner.
    pub completed: bool,
    /// Steps the stall stop saved against the run's frozen step cap, on a
    /// stopped run.
    pub steps_saved: Option<u64>,
}

/// One run finished, of any cell and outcome. A completed run-cap probe
/// feeds the learner, so its checkpoints are the step cap's; a timer-context
/// probe is exempt from the cut but feeds nothing. An untreated run leaves a
/// row and its gap against the standing cap; every run reports its marks.
pub fn finish_run(cell: StallCapCell, clock: &RunClock, standing: Option<i32>, end: RunEnding) {
    let longest = clock.longest();
    let mut row_dropped = false;
    if cell == StallCapCell::Untreated
        && util_stats::enabled()
        && let Ok(mut rows) = ROWS.lock()
    {
        if rows.len() < ROW_CAP {
            rows.push(RunRow {
                run_id: end.run_id,
                longest_quiet_gap: longest,
                stall_cap_standing: standing.unwrap_or(0),
            });
        } else {
            row_dropped = true;
        }
    }
    util_stats::record_stall_cap_run(&StallCapRun {
        cell,
        marks: clock.marks(),
        suspended_steps: clock.suspended_steps(),
        steps_saved: end.steps_saved,
        longest_gap: longest.max(0) as u32,
        standing_cap: standing.map(|c| c.max(0) as u32),
        row_dropped,
    });
    if cell == StallCapCell::Probe && end.completed && run_cap::is_probe(end.run_id) {
        merge_probe(end.backup, longest);
    }
}

/// The untreated rows as CSV with a header line, or None when no row was
/// recorded.
pub fn render_run_rows() -> Option<String> {
    let rows = ROWS.lock().ok()?;
    if rows.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(rows.len() * 24 + 48);
    out.push_str("run_id,longest_quiet_gap,stall_cap_standing\n");
    for r in rows.iter() {
        out.push_str(&format!(
            "{},{},{}\n",
            r.run_id, r.longest_quiet_gap, r.stall_cap_standing
        ));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;

    fn feed(backup: i32, n: usize, gap: i32) {
        for _ in 0..n {
            merge_probe(backup, gap);
        }
    }

    #[test]
    fn identity_below_the_sample_floor() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(6000, 199, 400);
        assert_eq!(effective_cap(6000), None, "199 samples are under the floor");
        merge_probe(6000, 400);
        assert!(effective_cap(6000).is_some(), "the 200th sample engages the cap");
        reset();
    }

    #[test]
    fn cap_is_the_quantile_upper_edge_with_headroom_and_scopes_are_separate() {
        let _serial = config_override::exclusive_session();
        reset();
        feed(6000, 200, 400);
        // Width 24, so 400 lands in cell 16 with upper edge 407; the cap is
        // ceil(1.5 * 407), the same arithmetic the step cap uses.
        assert_eq!(effective_cap(6000), Some(611));
        assert_eq!(effective_cap(1500), None, "another scope stays identity");
        reset();
    }

    #[test]
    fn cap_clamps_to_the_budget_and_stays_constant_between_checkpoints() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        feed(6000, 200, 400);
        assert_eq!(effective_cap(6000), Some(611));
        let before = util_stats::snapshot().stall_cap;
        feed(6000, 150, 4000);
        let after = util_stats::snapshot().stall_cap;
        assert_eq!(effective_cap(6000), Some(611), "350 completions sit between checkpoints");
        assert_eq!(
            after.probe_over_cap_completions,
            before.probe_over_cap_completions + 150,
            "completions above the standing cap are counted while it holds"
        );
        assert_eq!(after.probes_keyed, before.probes_keyed + 150);
        feed(6000, 50, 4000);
        util_stats::set_enabled(false);
        // The p99 of the mixture lands in the 4000-gap cell, whose upper
        // edge with headroom exceeds the budget, so the cap clamps to it.
        assert_eq!(effective_cap(6000), Some(6000));
        reset();
    }

    #[test]
    fn decay_delays_the_next_checkpoint_and_reset_empties() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        feed(6000, 200, 100);
        let engaged = effective_cap(6000);
        assert!(engaged.is_some());
        decay(0.5);
        assert_eq!(effective_cap(6000), engaged, "the cap survives decay unchanged");
        assert!(!TABLE.is_empty(), "an engaged scope is retained through decay");
        feed(6000, 300, 400);
        assert_eq!(effective_cap(6000), Some(611), "the recompute follows the fresher gaps");
        let s = util_stats::snapshot().stall_cap;
        assert_eq!(s.scopes_learned, 1);
        assert_eq!(s.cap_max_scope, 611);
        reset();
        assert!(TABLE.is_empty());
        assert_eq!(effective_cap(6000), None);
        let s = util_stats::snapshot().stall_cap;
        util_stats::set_enabled(false);
        assert_eq!((s.scopes_learned, s.cap_max_scope), (0, 0));
    }

    #[test]
    fn the_treated_cell_is_three_quarters_of_the_runs_outside_both_probe_streams() {
        let n = 64_000i64;
        let mut treated = 0i64;
        let mut eligible = 0i64;
        for id in -n / 2..n / 2 {
            let c = cell(id);
            assert_eq!(is_treated(id), c == StallCapCell::Treated);
            let probe = run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe;
            assert_eq!(c == StallCapCell::Probe, probe, "run {id}: the probe exemption");
            if !probe {
                eligible += 1;
                treated += (c == StallCapCell::Treated) as i64;
            }
        }
        let share = treated as f64 / eligible as f64;
        assert!((share - 0.75).abs() < 0.02, "the treated cell takes {share}");
    }

    #[test]
    fn the_clock_counts_steps_between_marks_and_holds_while_suspended() {
        let mut c = RunClock::default();
        let quiet = Marks::default();
        assert_eq!(c.step(quiet, false), 1);
        assert_eq!(c.step(quiet, false), 2);
        assert_eq!(c.step(quiet, true), 2, "a suspended step does not advance");
        assert_eq!(c.step(quiet, false), 3);
        assert_eq!(c.longest(), 3, "the open segment counts");
        let row = Marks { row: true, ..Marks::default() };
        assert_eq!(c.step(row, false), 0, "a mark closes the gap");
        assert_eq!(c.step(quiet, false), 1);
        assert_eq!(c.longest(), 3);
        let both = Marks { acted_timer: true, release: true, ..Marks::default() };
        assert_eq!(c.step(both, true), 0, "a marked step closes the gap even when suspended");
        for _ in 0..5 {
            c.step(quiet, false);
        }
        assert_eq!(c.longest(), 5);
        assert_eq!(c.step(Marks { acted_delivery: true, ..Marks::default() }, false), 0);
        assert_eq!(c.suspended_steps(), 1);
        let m = c.marks();
        assert_eq!((m.rows, m.acted_deliveries, m.acted_timers, m.releases), (1, 1, 1, 1));
    }

    #[test]
    fn an_untreated_run_leaves_a_row_and_its_gap_against_the_standing_cap() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        let mut clock = RunClock::default();
        for _ in 0..20 {
            clock.step(Marks::default(), false);
        }
        clock.step(Marks { row: true, ..Marks::default() }, false);
        let before = util_stats::snapshot().stall_cap;
        finish_run(
            StallCapCell::Untreated,
            &clock,
            Some(15),
            RunEnding { run_id: 41, backup: 256, completed: false, steps_saved: None },
        );
        finish_run(
            StallCapCell::Untreated,
            &clock,
            None,
            RunEnding { run_id: 42, backup: 256, completed: true, steps_saved: None },
        );
        finish_run(
            StallCapCell::Treated,
            &clock,
            Some(15),
            RunEnding { run_id: 43, backup: 256, completed: false, steps_saved: Some(200) },
        );
        let after = util_stats::snapshot().stall_cap;
        util_stats::set_enabled(false);
        assert_eq!(after.untreated_runs, before.untreated_runs + 2);
        assert_eq!(after.untreated_runs_capped, before.untreated_runs_capped + 1);
        assert_eq!(after.untreated_over_cap_runs, before.untreated_over_cap_runs + 1);
        assert_eq!(after.treated_runs, before.treated_runs + 1);
        assert_eq!(after.stops, before.stops + 1);
        assert_eq!(after.steps_saved_sum, before.steps_saved_sum + 200);
        assert_eq!(after.marks.rows, before.marks.rows + 3);
        assert_eq!(after.probes_keyed, before.probes_keyed, "no probe was folded");
        let csv = render_run_rows().expect("two rows were recorded");
        assert_eq!(
            csv,
            "run_id,longest_quiet_gap,stall_cap_standing\n41,20,15\n42,20,0\n"
        );
        reset();
        assert!(render_run_rows().is_none(), "reset clears the rows");
    }

    #[test]
    fn only_a_completed_run_cap_probe_feeds_the_learner() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        let mut clock = RunClock::default();
        for _ in 0..10 {
            clock.step(Marks::default(), false);
        }
        let cap_probe = (0..10_000i64).find(|&id| run_cap::is_probe(id)).expect("a probe id");
        let timer_probe = (0..10_000i64)
            .find(|&id| {
                !run_cap::is_probe(id)
                    && timer_context::run_mode(id) == timer_context::RunMode::Probe
            })
            .expect("a timer-context probe id");
        assert_eq!(cell(timer_probe), StallCapCell::Probe);
        let before = util_stats::snapshot().stall_cap;
        finish_run(
            StallCapCell::Probe,
            &clock,
            None,
            RunEnding { run_id: cap_probe, backup: 256, completed: false, steps_saved: None },
        );
        assert_eq!(util_stats::snapshot().stall_cap.probes_keyed, before.probes_keyed);
        assert!(TABLE.is_empty(), "an exhausted probe leaves no sample");
        finish_run(
            StallCapCell::Probe,
            &clock,
            None,
            RunEnding { run_id: timer_probe, backup: 256, completed: true, steps_saved: None },
        );
        assert_eq!(util_stats::snapshot().stall_cap.probes_keyed, before.probes_keyed);
        assert!(TABLE.is_empty(), "a completed timer-context probe leaves no sample");
        finish_run(
            StallCapCell::Probe,
            &clock,
            None,
            RunEnding { run_id: cap_probe, backup: 256, completed: true, steps_saved: None },
        );
        let after = util_stats::snapshot().stall_cap;
        util_stats::set_enabled(false);
        assert_eq!(after.probes_keyed, before.probes_keyed + 1);
        assert_eq!(after.untreated_runs, before.untreated_runs, "a probe is not an untreated run");
        assert!(render_run_rows().is_none(), "a probe leaves no row");
        reset();
    }
}
