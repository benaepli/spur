//! Compile-time-selectable feedback strategies.
//!
//! Feedback (CFG coverage, timeline coverage, genetic scoring, within-run
//! steering) is modeled as a `Feedback` trait whose implementations are
//! zero-sized strategy types, mirroring the `HashPolicy` pattern. A single
//! runtime `match` at each explorer entry point selects the monomorphized
//! instantiation, so disabled feedback collapses to no-op methods that the
//! optimizer eliminates entirely rather than runtime branches.

use crate::compiler::cfg::Vertex;
use crate::simulator::core::{NodeId, Runnable};
use crate::simulator::coverage::{GlobalCoverage, LocalCoverage, VertexMap};
use crate::simulator::hash_utils::HashPolicy;
use crate::simulator::util_stats;
use dashmap::DashMap;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

fn default_timeline_weight() -> f64 {
    0.8
}
fn default_cfg_weight() -> f64 {
    0.2
}
fn default_novel_scale() -> f64 {
    5.0
}

/// Weights for the blended genetic fitness and the timeline saturation curve.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CoverageConfig {
    #[serde(default = "default_timeline_weight")]
    pub timeline_weight: f64,
    #[serde(default = "default_cfg_weight")]
    pub cfg_weight: f64,
    #[serde(default = "default_novel_scale")]
    pub novel_scale: f64,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            timeline_weight: default_timeline_weight(),
            cfg_weight: default_cfg_weight(),
            novel_scale: default_novel_scale(),
        }
    }
}

/// Which feedback strategy to monomorphize for an exploration session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FeedbackMode {
    /// No feedback: zero cost, genetic loop degenerates to random search.
    /// This is the default so no feedback work is paid for unless requested.
    #[default]
    None,
    /// CFG-edge coverage only (the historical behavior).
    Cfg,
    /// Abstract Lamport timeline coverage only.
    Timeline,
    /// Both CFG and timeline coverage, blended.
    Both,
}

/// How many fields distinguish one timeline coverage key from another.
/// Lower resolution collapses more orderings into the same key, so a run has
/// fewer ways to look novel; higher resolution splits them further apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TimelineKeyGranularity {
    /// Handler pair only: the same ordering on two different nodes is one key.
    Coarse,
    /// Handler pair per destination node.
    #[default]
    Default,
    /// Handler pair per destination node per restart generation of that node,
    /// so an ordering seen before a crash is distinct from the same ordering
    /// seen after recovery.
    Fine,
}

impl TimelineKeyGranularity {
    /// Builds the coverage key for "handler `a` ran before handler `b` on
    /// `dest`, while `dest` was on its `generation`-th incarnation".
    pub fn key(self, dest: NodeId, a: Vertex, b: Vertex, generation: u32) -> TimelineTuple {
        match self {
            Self::Coarse => TimelineTuple {
                dest: None,
                a,
                b,
                generation: 0,
            },
            Self::Default => TimelineTuple {
                dest: Some(dest),
                a,
                b,
                generation: 0,
            },
            Self::Fine => TimelineTuple {
                dest: Some(dest),
                a,
                b,
                generation,
            },
        }
    }

    fn tracks_generation(self) -> bool {
        matches!(self, Self::Fine)
    }
}

/// Session-level feedback selection, deserialized from the explorer config.
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct FeedbackConfig {
    #[serde(default)]
    pub mode: FeedbackMode,
    /// Within-run delivery steering toward novel timeline tuples.
    /// Only meaningful for `Timeline`/`Both`.
    #[serde(default)]
    pub steer: bool,
    #[serde(default)]
    pub weights: CoverageConfig,
    /// Resolution of the timeline coverage key.
    /// Only meaningful for `Timeline`/`Both`.
    #[serde(default)]
    pub timeline_key_granularity: TimelineKeyGranularity,
    /// Standardize the novelty term and the priority term against their running
    /// session scale before combining them in the within-queue score, instead of
    /// combining the raw values with fixed coefficients.
    #[serde(default)]
    pub normalize_scores: bool,
}

impl FeedbackConfig {
    pub fn validate(&self) -> Result<(), String> {
        let w = &self.weights;
        if w.timeline_weight < 0.0 || w.cfg_weight < 0.0 {
            return Err(format!(
                "feedback weights must be >= 0 (timeline={}, cfg={})",
                w.timeline_weight, w.cfg_weight
            ));
        }
        if w.timeline_weight + w.cfg_weight <= 0.0 {
            return Err("feedback weights must sum to > 0".to_string());
        }
        Ok(())
    }
}

/// A direction-aware happens-before tuple: on `dest`, handler `a` ran before
/// handler `b`. `(dest, a, b)` is distinct from `(dest, b, a)`. Which fields
/// are populated is decided by `TimelineKeyGranularity`; build keys with
/// `TimelineKeyGranularity::key` rather than constructing them directly.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct TimelineTuple {
    pub dest: Option<NodeId>,
    pub a: Vertex,
    pub b: Vertex,
    pub generation: u32,
}

/// Per-run timeline accumulator.
#[derive(Debug, Default, Clone)]
pub struct LocalTimeline {
    pub tuples: HashSet<TimelineTuple>,
    pub per_dest_seen: HashMap<NodeId, HashSet<Vertex>>,
    granularity: TimelineKeyGranularity,
    generations: HashMap<NodeId, u32>,
}

impl LocalTimeline {
    pub fn set_granularity(&mut self, granularity: TimelineKeyGranularity) {
        self.granularity = granularity;
    }

    fn generation(&self, dest: NodeId) -> u32 {
        self.generations.get(&dest).copied().unwrap_or(0)
    }

    /// Records that `handler` was delivered to `dest`, forming a happens-before
    /// tuple against every distinct prior handler on that node.
    ///
    /// Pairing against the distinct prior set (rather than the full ordered
    /// sequence) gives the same tuple set, since the `HashSet` already
    /// collapses duplicates, but scales with the number of distinct handlers.
    pub fn note_delivery(&mut self, dest: NodeId, handler: Vertex) {
        let generation = self.generation(dest);
        let granularity = self.granularity;
        if let Some(priors) = self.per_dest_seen.get(&dest) {
            for &prior in priors {
                self.tuples
                    .insert(granularity.key(dest, prior, handler, generation));
            }
        }
        self.per_dest_seen.entry(dest).or_default().insert(handler);
    }

    /// Records that `node` came back up, starting a new incarnation. Prior
    /// handlers stay in the pairing set so orderings still span the restart.
    pub fn note_recovery(&mut self, node: NodeId) {
        *self.generations.entry(node).or_default() += 1;
    }

    /// Number of this run's tuples not present in the global snapshot.
    pub fn novel_count(&self, snap: &TimelineSnap) -> usize {
        self.tuples
            .iter()
            .filter(|t| !snap.counts.contains_key(*t))
            .count()
    }
}

/// Read-only view of global timeline coverage, taken once per run.
#[derive(Debug, Default, Clone)]
pub struct TimelineSnap {
    pub counts: HashMap<TimelineTuple, u64>,
    pub total: u64,
}

impl TimelineSnap {
    /// Rarity of a tuple: 1.0 if never seen, otherwise `1 - count/total`.
    pub fn novelty_score(&self, tuple: &TimelineTuple) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        let c = self.counts.get(tuple).copied().unwrap_or(0) as f64;
        1.0 - (c / self.total as f64)
    }
}

/// Global timeline store, shared across runs.
#[derive(Debug, Default)]
pub struct GlobalTimeline {
    counts: DashMap<TimelineTuple, u64>,
    total: AtomicU64,
}

impl GlobalTimeline {
    pub fn snapshot(&self) -> TimelineSnap {
        let counts: HashMap<TimelineTuple, u64> =
            self.counts.iter().map(|e| (*e.key(), *e.value())).collect();
        TimelineSnap {
            counts,
            total: self.total.load(Ordering::Relaxed),
        }
    }

    pub fn merge(&self, local: &LocalTimeline) {
        for t in &local.tuples {
            *self.counts.entry(*t).or_default() += 1;
        }
        let _ = self
            .total
            .fetch_add(local.tuples.len() as u64, Ordering::Relaxed);
    }

    /// Scale all tuple counts by factor.
    pub fn decay(&self, factor: f64) {
        let factor = factor.clamp(0.0, 1.0);
        self.counts.retain(|_, c| {
            *c = ((*c as f64) * factor).floor() as u64;
            *c > 0
        });
        let new_total: u64 = self.counts.iter().map(|e| *e.value()).sum();
        self.total.store(new_total, Ordering::Relaxed);
    }
}

/// Prospective within-run steering bias for delivering `r` next: the mean
/// global rarity of the tuples it would form against priors on its node.
/// Returns 1.0 (neutral) for anything that is not a first-entry delivery.
fn timeline_steer_bias<H: HashPolicy>(
    tl: &LocalTimeline,
    r: &Runnable<H>,
    snap: &TimelineSnap,
) -> f64 {
    let Runnable::Record(rec) = r else {
        return 1.0;
    };
    if rec.pc != rec.entry_pc {
        return 1.0;
    }
    let generation = tl.generation(rec.node);
    match tl.per_dest_seen.get(&rec.node) {
        Some(priors) if !priors.is_empty() => {
            let sum: f64 = priors
                .iter()
                .map(|&a| {
                    snap.novelty_score(&tl.granularity.key(
                        rec.node,
                        a,
                        rec.entry_pc,
                        generation,
                    ))
                })
                .sum();
            sum / priors.len() as f64
        }
        _ => 1.0,
    }
}

/// Saturating timeline fitness: rewards runs that add globally-new tuples with
/// diminishing returns, staying in [0, 1).
fn timeline_plan_score(local: &LocalTimeline, snap: &TimelineSnap, w: &CoverageConfig) -> f64 {
    let novel = local.novel_count(snap) as f64;
    novel / (novel + w.novel_scale.max(1e-9))
}

/// A compile-time feedback strategy. Default methods are no-ops so that the
/// `NoFeedback` strategy collapses to nothing and every call site is uniform
/// (no `if enabled` branches anywhere in the hot path).
pub trait Feedback: 'static + Send + Sync + std::fmt::Debug {
    /// Per-run accumulator, threaded as `&mut`.
    type Local: Default + std::fmt::Debug + Send;
    /// Shared store living on `GlobalState`.
    type Global: Default + std::fmt::Debug + Send + Sync;
    /// Read-only global view, taken once per run.
    type Snapshot;

    /// Whether the first-entry delivery capture block runs. Const-folded out
    /// for non-timeline strategies.
    const CAPTURES_TIMELINE: bool = false;

    fn snapshot(global: &Self::Global) -> Self::Snapshot;
    fn merge(global: &Self::Global, local: &Self::Local);

    /// Record a CFG transition `prev -> pc`.
    #[inline]
    fn record_transition(
        _local: &mut Self::Local,
        _prev: Vertex,
        _pc: Vertex,
        _snap: &Self::Snapshot,
    ) {
    }

    /// Capture a first-entry handler delivery on a server node.
    #[inline]
    fn note_delivery(_local: &mut Self::Local, _dest: NodeId, _handler: Vertex) {}

    /// Capture a node coming back up after a crash.
    #[inline]
    fn note_recovery(_local: &mut Self::Local, _node: NodeId) {}

    /// Apply the session's timeline key resolution to a fresh run accumulator.
    #[inline]
    fn set_key_granularity(_local: &mut Self::Local, _granularity: TimelineKeyGranularity) {}

    /// Novelty term in [0, 1] used by the within-queue selector. Defaults to
    /// 1.0 (uniform), which reproduces the "no global snapshot" behavior.
    #[inline]
    fn runnable_novelty<H: HashPolicy>(
        _local: &Self::Local,
        _r: &Runnable<H>,
        _snap: &Self::Snapshot,
    ) -> f64 {
        1.0
    }

    /// Genetic fitness for a completed run. Defaults to 0.0 (random search).
    #[inline]
    fn plan_score(_local: &Self::Local, _snap: &Self::Snapshot, _w: &CoverageConfig) -> f64 {
        0.0
    }

    fn decay(_global: &Self::Global, _factor: f64) {}

    /// Per-vertex CFG hit counts for the CLI heatmap, if this strategy tracks
    /// CFG coverage.
    fn vertex_coverage(_global: &Self::Global) -> Option<HashMap<usize, u64>> {
        None
    }

    /// The run's timeline happens-before tuples, if this strategy tracks them.
    /// Used by the AOS controller to compute within-scenario credit. Defaults
    /// to `None` for non-timeline strategies.
    fn timeline_tuples(_local: &Self::Local) -> Option<&HashSet<TimelineTuple>> {
        None
    }
}

/// No feedback. Zero cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoFeedback;

impl Feedback for NoFeedback {
    type Local = ();
    type Global = ();
    type Snapshot = ();

    #[inline]
    fn snapshot(_global: &()) {}
    #[inline]
    fn merge(_global: &(), _local: &()) {}
}

/// CFG-edge coverage only. Reproduces the historical behavior exactly.
#[derive(Debug, Clone, Copy, Default)]
pub struct CfgFeedback;

impl Feedback for CfgFeedback {
    type Local = LocalCoverage;
    type Global = GlobalCoverage;
    type Snapshot = VertexMap;

    fn snapshot(global: &GlobalCoverage) -> VertexMap {
        global.snapshot()
    }
    fn merge(global: &GlobalCoverage, local: &LocalCoverage) {
        global.merge(local);
    }

    #[inline]
    fn record_transition(local: &mut LocalCoverage, prev: Vertex, pc: Vertex, snap: &VertexMap) {
        let rarity = snap.novelty_score(pc);
        local.record_with_rarity(prev, pc, rarity);
    }

    #[inline]
    fn runnable_novelty<H: HashPolicy>(
        _local: &LocalCoverage,
        r: &Runnable<H>,
        snap: &VertexMap,
    ) -> f64 {
        snap.novelty_score(r.pc())
    }

    #[inline]
    fn plan_score(local: &LocalCoverage, _snap: &VertexMap, _w: &CoverageConfig) -> f64 {
        let cfg_score = local.plan_score();
        util_stats::record_feedback_scores(None, Some(cfg_score));
        cfg_score
    }

    fn decay(global: &GlobalCoverage, factor: f64) {
        global.decay(factor);
    }

    fn vertex_coverage(global: &GlobalCoverage) -> Option<HashMap<usize, u64>> {
        Some(global.vertices_snapshot().into_iter().collect())
    }
}

/// Abstract Lamport timeline coverage. `STEER` toggles within-run steering at
/// compile time (dead-code-eliminated when false).
#[derive(Debug, Clone, Copy, Default)]
pub struct TimelineFeedback<const STEER: bool>;

impl<const STEER: bool> Feedback for TimelineFeedback<STEER> {
    type Local = LocalTimeline;
    type Global = GlobalTimeline;
    type Snapshot = TimelineSnap;

    const CAPTURES_TIMELINE: bool = true;

    fn snapshot(global: &GlobalTimeline) -> TimelineSnap {
        global.snapshot()
    }
    fn merge(global: &GlobalTimeline, local: &LocalTimeline) {
        global.merge(local);
    }

    #[inline]
    fn note_delivery(local: &mut LocalTimeline, dest: NodeId, handler: Vertex) {
        local.note_delivery(dest, handler);
    }

    #[inline]
    fn note_recovery(local: &mut LocalTimeline, node: NodeId) {
        if local.granularity.tracks_generation() {
            local.note_recovery(node);
        }
    }

    #[inline]
    fn set_key_granularity(local: &mut LocalTimeline, granularity: TimelineKeyGranularity) {
        local.set_granularity(granularity);
    }

    #[inline]
    fn runnable_novelty<H: HashPolicy>(
        local: &LocalTimeline,
        r: &Runnable<H>,
        snap: &TimelineSnap,
    ) -> f64 {
        if STEER {
            timeline_steer_bias(local, r, snap)
        } else {
            1.0
        }
    }

    #[inline]
    fn plan_score(local: &LocalTimeline, snap: &TimelineSnap, w: &CoverageConfig) -> f64 {
        let tl_score = timeline_plan_score(local, snap, w);
        util_stats::record_feedback_scores(Some(tl_score), None);
        tl_score
    }

    fn decay(global: &GlobalTimeline, factor: f64) {
        global.decay(factor);
    }

    fn timeline_tuples(local: &LocalTimeline) -> Option<&HashSet<TimelineTuple>> {
        Some(&local.tuples)
    }
}

/// CFG + timeline coverage, blended. `STEER` toggles within-run steering.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullFeedback<const STEER: bool>;

impl<const STEER: bool> Feedback for FullFeedback<STEER> {
    type Local = (LocalCoverage, LocalTimeline);
    type Global = (GlobalCoverage, GlobalTimeline);
    type Snapshot = (VertexMap, TimelineSnap);

    const CAPTURES_TIMELINE: bool = true;

    fn snapshot(global: &Self::Global) -> Self::Snapshot {
        (global.0.snapshot(), global.1.snapshot())
    }
    fn merge(global: &Self::Global, local: &Self::Local) {
        global.0.merge(&local.0);
        global.1.merge(&local.1);
    }

    #[inline]
    fn record_transition(local: &mut Self::Local, prev: Vertex, pc: Vertex, snap: &Self::Snapshot) {
        let rarity = snap.0.novelty_score(pc);
        local.0.record_with_rarity(prev, pc, rarity);
    }

    #[inline]
    fn note_delivery(local: &mut Self::Local, dest: NodeId, handler: Vertex) {
        local.1.note_delivery(dest, handler);
    }

    #[inline]
    fn note_recovery(local: &mut Self::Local, node: NodeId) {
        if local.1.granularity.tracks_generation() {
            local.1.note_recovery(node);
        }
    }

    #[inline]
    fn set_key_granularity(local: &mut Self::Local, granularity: TimelineKeyGranularity) {
        local.1.set_granularity(granularity);
    }

    #[inline]
    fn runnable_novelty<H: HashPolicy>(
        local: &Self::Local,
        r: &Runnable<H>,
        snap: &Self::Snapshot,
    ) -> f64 {
        let cfg = snap.0.novelty_score(r.pc());
        if STEER {
            0.5 * cfg + 0.5 * timeline_steer_bias(&local.1, r, &snap.1)
        } else {
            cfg
        }
    }

    #[inline]
    fn plan_score(local: &Self::Local, snap: &Self::Snapshot, w: &CoverageConfig) -> f64 {
        let cfg_score = local.0.plan_score();
        let tl_score = timeline_plan_score(&local.1, &snap.1, w);
        util_stats::record_feedback_scores(Some(tl_score), Some(cfg_score));
        let denom = w.timeline_weight + w.cfg_weight;
        if denom <= 0.0 {
            return cfg_score;
        }
        (w.timeline_weight * tl_score + w.cfg_weight * cfg_score) / denom
    }

    fn decay(global: &Self::Global, factor: f64) {
        global.0.decay(factor);
        global.1.decay(factor);
    }

    fn vertex_coverage(global: &Self::Global) -> Option<HashMap<usize, u64>> {
        Some(global.0.vertices_snapshot().into_iter().collect())
    }

    fn timeline_tuples(local: &Self::Local) -> Option<&HashSet<TimelineTuple>> {
        Some(&local.1.tuples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(index: usize) -> NodeId {
        NodeId {
            role: crate::analysis::resolver::NameId(0),
            index,
        }
    }

    #[test]
    fn distinct_prior_pairing_equals_all_pairs() {
        // Ordered delivery sequence [a, b, a, c] on one node.
        let d = node(0);
        let (a, b, c) = (10usize, 20usize, 30usize);

        let mut tl = LocalTimeline::default();
        for h in [a, b, a, c] {
            tl.note_delivery(d, h);
        }

        // Reference all-pairs ToTimeline over the ordered sequence.
        let seq = [a, b, a, c];
        let mut expected: HashSet<TimelineTuple> = HashSet::new();
        for i in 0..seq.len() {
            for j in (i + 1)..seq.len() {
                expected.insert(TimelineKeyGranularity::Default.key(d, seq[i], seq[j], 0));
            }
        }

        assert_eq!(tl.tuples, expected);
    }

    #[test]
    fn timeline_merge_and_snapshot() {
        let g = GlobalTimeline::default();
        let mut tl = LocalTimeline::default();
        let d = node(1);
        tl.note_delivery(d, 1);
        tl.note_delivery(d, 2); // forms (d,1,2)
        assert_eq!(tl.tuples.len(), 1);

        let snap0 = g.snapshot();
        assert_eq!(tl.novel_count(&snap0), 1);

        g.merge(&tl);
        let snap1 = g.snapshot();
        assert_eq!(snap1.counts.len(), 1);
        assert_eq!(tl.novel_count(&snap1), 0);
    }

    #[test]
    fn timeline_decay_scales_counts_and_drops_zeros() {
        let g = GlobalTimeline::default();
        let d = node(0);
        // Tuple (d,1,2) accumulates a count of 4; (d,3,4) a count of 1.
        for _ in 0..4 {
            let mut tl = LocalTimeline::default();
            tl.note_delivery(d, 1);
            tl.note_delivery(d, 2);
            g.merge(&tl);
        }
        let mut tl = LocalTimeline::default();
        tl.note_delivery(d, 3);
        tl.note_delivery(d, 4);
        g.merge(&tl);
        assert_eq!(g.snapshot().total, 5);

        g.decay(0.5);
        let after = g.snapshot();
        assert_eq!(after.counts.len(), 1, "the count-1 tuple should be dropped");
        assert_eq!(after.total, 2, "total recomputed from surviving counts");
        let surviving = TimelineKeyGranularity::Default.key(d, 1, 2, 0);
        assert_eq!(after.counts.get(&surviving), Some(&2));
        assert!((0.0..=1.0).contains(&after.novelty_score(&surviving)));
    }

    #[test]
    fn full_plan_score_in_range_and_blends() {
        let mut cov = LocalCoverage::new();
        cov.record_with_rarity(0, 1, 1.0);
        let mut tl = LocalTimeline::default();
        let d = node(0);
        tl.note_delivery(d, 1);
        tl.note_delivery(d, 2);

        let local = (cov, tl);
        let snap = (VertexMap::new(), TimelineSnap::default());
        let w = CoverageConfig::default();
        let s = FullFeedback::<false>::plan_score(&local, &snap, &w);
        assert!((0.0..=1.0).contains(&s), "score out of range: {}", s);

        // denom <= 0 guard returns the cfg component.
        let w0 = CoverageConfig {
            timeline_weight: 0.0,
            cfg_weight: 0.0,
            novel_scale: 5.0,
        };
        let s0 = FullFeedback::<false>::plan_score(&local, &snap, &w0);
        assert_eq!(s0, local.0.plan_score());
    }

    #[test]
    fn steer_bias_prefers_novel_orderings() {
        // A common tuple (low novelty) should yield a lower bias than an unseen one.
        let d = node(0);
        let mut snap = TimelineSnap::default();
        snap.total = 10;
        snap.counts
            .insert(TimelineKeyGranularity::Default.key(d, 1, 2, 0), 10);

        let mut tl = LocalTimeline::default();
        tl.per_dest_seen.entry(d).or_default().insert(1);

        // Candidate handler 2 forms the common tuple (d,1,2): low novelty.
        let common = snap.novelty_score(&TimelineKeyGranularity::Default.key(d, 1, 2, 0));
        // Candidate handler 3 forms an unseen tuple: novelty 1.0.
        let novel = snap.novelty_score(&TimelineKeyGranularity::Default.key(d, 1, 3, 0));
        assert!(novel > common);
    }

    #[test]
    fn granularity_parses_from_config_json() {
        let cfg: FeedbackConfig = serde_json::from_str(
            r#"{"mode": "timeline", "steer": true, "timeline_key_granularity": "fine"}"#,
        )
        .expect("config should parse");
        assert_eq!(cfg.timeline_key_granularity, TimelineKeyGranularity::Fine);

        let defaulted: FeedbackConfig = serde_json::from_str(r#"{"mode": "timeline"}"#).unwrap();
        assert_eq!(
            defaulted.timeline_key_granularity,
            TimelineKeyGranularity::Default
        );
    }

    #[test]
    fn granularity_changes_key_resolution() {
        let (d0, d1) = (node(0), node(1));
        assert_eq!(
            TimelineKeyGranularity::Coarse.key(d0, 1, 2, 0),
            TimelineKeyGranularity::Coarse.key(d1, 1, 2, 3),
            "coarse keys forget the destination node and its generation"
        );
        assert_ne!(
            TimelineKeyGranularity::Default.key(d0, 1, 2, 0),
            TimelineKeyGranularity::Default.key(d1, 1, 2, 0)
        );
        assert_eq!(
            TimelineKeyGranularity::Default.key(d0, 1, 2, 0),
            TimelineKeyGranularity::Default.key(d0, 1, 2, 7),
            "default keys ignore the restart generation"
        );
        assert_ne!(
            TimelineKeyGranularity::Fine.key(d0, 1, 2, 0),
            TimelineKeyGranularity::Fine.key(d0, 1, 2, 1)
        );
    }

    #[test]
    fn fine_granularity_separates_orderings_across_a_restart() {
        let d = node(0);
        let mut fine = LocalTimeline::default();
        fine.set_granularity(TimelineKeyGranularity::Fine);
        fine.note_delivery(d, 1);
        fine.note_delivery(d, 2);
        fine.note_recovery(d);
        fine.note_delivery(d, 2);
        assert_eq!(fine.tuples.len(), 3, "(d,1,2) counted once per incarnation");

        let mut default = LocalTimeline::default();
        default.note_delivery(d, 1);
        default.note_delivery(d, 2);
        default.note_recovery(d);
        default.note_delivery(d, 2);
        assert_eq!(default.tuples.len(), 2);
    }
}
