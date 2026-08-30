use crate::compiler::cfg::Vertex;
use crate::simulator::feedback::Feedback;
use dashmap::DashMap;
use imbl::HashMap as ImMap;
use imbl::shared_ptr::ArcK;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub struct VertexMap {
    vertices: ImMap<Vertex, u64>,
    total: u64,
}

impl Default for VertexMap {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexMap {
    pub fn new() -> Self {
        Self {
            vertices: ImMap::new(),
            total: 0,
        }
    }

    pub fn novelty_score(&self, vertex: Vertex) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        let count = self.vertices.get(&vertex).copied().unwrap_or(0) as f64;
        1.0 - (count / self.total as f64)
    }

    pub fn get(&self, vertex: &Vertex) -> Option<u64> {
        self.vertices.get(vertex).copied()
    }

    pub fn merge_from(&mut self, other: &HashMap<Vertex, u64>) {
        for (v, count) in other {
            self.vertices
                .entry(*v)
                .and_modify(|e| *e += count)
                .or_insert(*count);
            self.total += count;
        }
    }

    /// Scale every count by `factor` (clamped to `[0, 1]`), dropping entries
    /// that round to zero, and recompute `total`. Used by the continuous
    /// explorer to bound memory and let saturated vertices regain novelty
    /// over a long session.
    pub fn decay(&mut self, factor: f64) {
        let factor = factor.clamp(0.0, 1.0);
        let mut scaled = ImMap::new();
        let mut total = 0u64;
        for (v, count) in self.vertices.iter() {
            let s = ((*count as f64) * factor).floor() as u64;
            if s > 0 {
                scaled.insert(*v, s);
                total += s;
            }
        }
        self.vertices = scaled;
        self.total = total;
    }
}

impl IntoIterator for VertexMap {
    type Item = (Vertex, u64);
    type IntoIter = imbl::hashmap::ConsumingIter<(Vertex, u64), ArcK>;

    fn into_iter(self) -> Self::IntoIter {
        self.vertices.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct LocalCoverage {
    edges: HashMap<(Vertex, Vertex), u64>,
    vertices: HashMap<Vertex, u64>,
    total: u64,
    novelty_score: f64,
}

impl Default for LocalCoverage {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalCoverage {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            vertices: HashMap::new(),
            total: 0,
            novelty_score: 0.0,
        }
    }

    /// Records a transition and accumulates novelty score using rarity^3.
    /// `rarity` should come from GlobalCoverage::novelty_score().
    pub fn record_with_rarity(&mut self, from: Vertex, to: Vertex, rarity: f64) -> bool {
        self.novelty_score += rarity.powi(3);
        self.record(from, to)
    }

    /// Returns the normalized plan score: mean novelty per step.
    /// Returns a value in [0, 1] where 1 means every transition was maximally novel.
    pub fn plan_score(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.novelty_score / self.total as f64
    }

    /// Returns the raw novelty score.
    pub fn novelty_score(&self) -> f64 {
        self.novelty_score
    }

    /// Records a transition between two vertices in the CFG.
    /// Returns true if this is the first time this edge has been seen.
    pub fn record(&mut self, from: Vertex, to: Vertex) -> bool {
        self.total += 1;

        let count = self.edges.entry((from, to)).or_insert(0);
        *count += 1;
        {
            let count = self.vertices.entry(to).or_insert(0);
            *count += 1;
        }
        *count == 1
    }

    /// Returns the total number of unique edges visited.
    pub fn unique_edges(&self) -> usize {
        self.edges.len()
    }

    /// Access to edges for merging into GlobalCoverage
    pub fn edges(&self) -> &HashMap<(Vertex, Vertex), u64> {
        &self.edges
    }

    /// Access to vertices for merging into GlobalCoverage
    pub fn vertices(&self) -> &HashMap<Vertex, u64> {
        &self.vertices
    }
}

#[derive(Debug)]
pub struct GlobalCoverage {
    edges: DashMap<(usize, usize), u64>,
    vertices: RwLock<VertexMap>,
    total: AtomicU64,
}

impl Default for GlobalCoverage {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalCoverage {
    pub fn new() -> Self {
        Self {
            edges: DashMap::new(),
            vertices: RwLock::new(VertexMap::new()),
            total: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> VertexMap {
        self.vertices
            .read()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock")
            .clone()
    }

    pub fn merge(&self, local: &LocalCoverage) {
        let _ = self.total.fetch_add(local.total, Ordering::Relaxed);
        for ((from, to), count) in local.edges() {
            *self.edges.entry((*from, *to)).or_default() += count;
        }

        let mut vertices = self
            .vertices
            .write()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock");
        vertices.merge_from(local.vertices());
    }

    /// Returns the total number of vertex visits across all merged runs.
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Scale all edge/vertex counts by factor.
    pub fn decay(&self, factor: f64) {
        let factor = factor.clamp(0.0, 1.0);
        self.edges.retain(|_, c| {
            *c = ((*c as f64) * factor).floor() as u64;
            *c > 0
        });
        let mut vertices = self
            .vertices
            .write()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock");
        vertices.decay(factor);
        // Keep the cheap `== 0` guard's denominator in step with the vertices.
        self.total.store(vertices.total, Ordering::Relaxed);
    }

    /// Calculates the novelty score for a vertex using global stats.
    /// Returns 1.0 if never seen, otherwise 1.0 - (count/total).
    pub fn novelty_score(&self, vertex: Vertex) -> f64 {
        let total = self.total();
        if total == 0 {
            return 1.0;
        }
        self.vertices
            .read()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock")
            .novelty_score(vertex)
    }

    /// Access to vertices for coverage visualization.
    pub fn vertices_snapshot(&self) -> VertexMap {
        self.vertices
            .read()
            .expect("RwLock poisoned - this indicates a panic occurred while holding the lock")
            .clone()
    }
}

/// How often a fault has been placed in each fault context, shared by the runs
/// that share a `GlobalState`.
///
/// The context is a small tag with tens of possible values rather than a
/// per-state key, so the counts still separate contexts late in a session
/// instead of every value having been seen.
#[derive(Debug, Default)]
pub struct FaultCoverage {
    visits: DashMap<u16, u64>,
    max_visits: AtomicU64,
}

impl FaultCoverage {
    pub fn visits(&self, tag: u16) -> u64 {
        self.visits.get(&tag).map(|v| *v).unwrap_or(0)
    }

    /// Count one fault placed in `tag`'s context. Returns how many distinct
    /// contexts the table holds and the largest count in it afterwards.
    pub fn visit(&self, tag: u16) -> (u64, u64) {
        let count = {
            let mut entry = self.visits.entry(tag).or_insert(0);
            *entry += 1;
            *entry
        };
        let max = self
            .max_visits
            .fetch_max(count, Ordering::Relaxed)
            .max(count);
        (self.visits.len() as u64, max)
    }
}

/// Global feedback state shared across all simulation runs.
///
/// Generic over the feedback strategy `F`: the per-session feedback store lives
/// in `feedback`.
pub struct GlobalState<F: Feedback> {
    pub feedback: F::Global,
    pub fault_coverage: FaultCoverage,
}

impl<F: Feedback> Default for GlobalState<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Feedback> std::fmt::Debug for GlobalState<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalState").finish_non_exhaustive()
    }
}

impl<F: Feedback> GlobalState<F> {
    pub fn new() -> Self {
        Self {
            feedback: F::Global::default(),
            fault_coverage: FaultCoverage::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_map_decay_scales_drops_zeros_and_stays_in_range() {
        let mut m = VertexMap::new();
        let mut src: HashMap<Vertex, u64> = HashMap::new();
        src.insert(1, 4);
        src.insert(2, 1);
        m.merge_from(&src);
        assert_eq!(m.total, 5);

        m.decay(0.5);
        assert_eq!(m.get(&1), Some(2)); // 4 -> 2
        assert_eq!(m.get(&2), None); // 1 -> 0, dropped
        assert_eq!(m.total, 2); // total recomputed
        assert!((0.0..=1.0).contains(&m.novelty_score(1)));
        assert!((0.0..=1.0).contains(&m.novelty_score(99)));
    }

    #[test]
    fn global_coverage_decay_keeps_total_consistent() {
        let g = GlobalCoverage::new();
        let mut local = LocalCoverage::new();
        for _ in 0..4 {
            local.record(0, 1);
        }
        local.record(2, 3);
        g.merge(&local);
        g.decay(0.5);
        // Vertex `1` was hit 4x -> 2; vertex `3` once -> dropped. Total tracks it.
        assert_eq!(g.total(), 2);
        assert!((0.0..=1.0).contains(&g.novelty_score(1)));
    }

    #[test]
    fn test_plan_score_empty_coverage() {
        let coverage = LocalCoverage::new();
        assert_eq!(coverage.plan_score(), 0.0);
    }

    #[test]
    fn test_plan_score_normalized_range() {
        let mut coverage = LocalCoverage::new();
        // Record transitions with maximum rarity (1.0)
        coverage.record_with_rarity(0, 1, 1.0);
        coverage.record_with_rarity(1, 2, 1.0);
        coverage.record_with_rarity(2, 3, 0.5);

        let score = coverage.plan_score();
        assert!(score >= 0.0, "plan_score should be >= 0, got {}", score);
        assert!(score <= 1.0, "plan_score should be <= 1, got {}", score);
    }

    #[test]
    fn test_plan_score_independent_of_length() {
        // Two runs with identical per-step rarity but different lengths
        let mut short = LocalCoverage::new();
        short.record_with_rarity(0, 1, 0.8);
        short.record_with_rarity(1, 2, 0.8);

        let mut long = LocalCoverage::new();
        for i in 0..100 {
            long.record_with_rarity(i, i + 1, 0.8);
        }

        let diff = (short.plan_score() - long.plan_score()).abs();
        assert!(
            diff < 1e-10,
            "Same per-step rarity should give same plan_score: short={}, long={}, diff={}",
            short.plan_score(),
            long.plan_score(),
            diff
        );
    }

    #[test]
    fn test_plan_score_higher_for_rarer_transitions() {
        let mut novel = LocalCoverage::new();
        novel.record_with_rarity(0, 1, 1.0);
        novel.record_with_rarity(1, 2, 1.0);

        let mut common = LocalCoverage::new();
        common.record_with_rarity(0, 1, 0.1);
        common.record_with_rarity(1, 2, 0.1);

        assert!(
            novel.plan_score() > common.plan_score(),
            "Novel transitions should score higher: novel={}, common={}",
            novel.plan_score(),
            common.plan_score()
        );
    }
}
