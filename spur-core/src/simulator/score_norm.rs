//! Online standardization of the two terms in the within-queue score.
//!
//! The novelty term and the priority term are produced on unrelated numeric
//! scales, so combining them with fixed coefficients lets whichever term
//! happens to have the wider spread decide every pick. Standardizing each term
//! against its own running mean and standard deviation puts both on one scale,
//! giving them comparable authority without a hand-set coefficient.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Standardized values are clamped to this many standard deviations, so a term
/// whose spread collapses toward zero cannot monopolize the combined score.
const Z_CLAMP: f64 = 4.0;

/// A term with fewer observations than this, or with a spread below
/// `MIN_STDDEV`, has no usable scale yet and contributes nothing.
const MIN_SAMPLES: f64 = 8.0;
const MIN_STDDEV: f64 = 1e-9;

/// Count, sum and sum of squares of one term.
#[derive(Debug, Default, Clone, Copy)]
pub struct Moments {
    count: f64,
    sum: f64,
    sum_sq: f64,
}

impl Moments {
    fn observe(&mut self, x: f64) {
        self.count += 1.0;
        self.sum += x;
        self.sum_sq += x * x;
    }

    fn plus(self, other: Self) -> Self {
        Self {
            count: self.count + other.count,
            sum: self.sum + other.sum,
            sum_sq: self.sum_sq + other.sum_sq,
        }
    }

    /// Distance of `x` from the mean in standard deviations, clamped; 0.0 when
    /// there is not yet a usable scale.
    fn standardize(&self, x: f64) -> f64 {
        if self.count < MIN_SAMPLES {
            return 0.0;
        }
        let mean = self.sum / self.count;
        let variance = (self.sum_sq / self.count - mean * mean).max(0.0);
        let stddev = variance.sqrt();
        if stddev < MIN_STDDEV {
            return 0.0;
        }
        ((x - mean) / stddev).clamp(-Z_CLAMP, Z_CLAMP)
    }
}

fn add_f64(cell: &AtomicU64, v: f64) {
    let mut cur = cell.load(Ordering::Relaxed);
    loop {
        let next = (f64::from_bits(cur) + v).to_bits();
        match cell.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// `Moments` held as f64 bit patterns so they can be shared across runs.
#[derive(Debug, Default)]
struct AtomicMoments {
    count: AtomicU64,
    sum: AtomicU64,
    sum_sq: AtomicU64,
}

impl AtomicMoments {
    fn load(&self) -> Moments {
        Moments {
            count: f64::from_bits(self.count.load(Ordering::Relaxed)),
            sum: f64::from_bits(self.sum.load(Ordering::Relaxed)),
            sum_sq: f64::from_bits(self.sum_sq.load(Ordering::Relaxed)),
        }
    }

    fn merge(&self, m: Moments) {
        add_f64(&self.count, m.count);
        add_f64(&self.sum, m.sum);
        add_f64(&self.sum_sq, m.sum_sq);
    }
}

/// Session-wide moments of both scoring terms, shared by every run.
#[derive(Debug, Default)]
pub struct GlobalScoreNorm {
    novelty: AtomicMoments,
    priority: AtomicMoments,
}

/// One run's view of the standardization: the session moments read once at run
/// start plus what this run has observed since. Scoring happens behind a shared
/// reference (the queue it reads is borrowed from the same state), and a run is
/// single-threaded, so the run-local part uses `Cell`.
#[derive(Debug)]
pub struct ScoreNorm {
    enabled: bool,
    prior_novelty: Moments,
    prior_priority: Moments,
    local_novelty: Cell<Moments>,
    local_priority: Cell<Moments>,
}

impl ScoreNorm {
    /// Combining stays off: callers keep their fixed-coefficient blend.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            prior_novelty: Moments::default(),
            prior_priority: Moments::default(),
            local_novelty: Cell::new(Moments::default()),
            local_priority: Cell::new(Moments::default()),
        }
    }

    pub fn new(enabled: bool, global: &GlobalScoreNorm) -> Self {
        if !enabled {
            return Self::disabled();
        }
        Self {
            enabled,
            prior_novelty: global.novelty.load(),
            prior_priority: global.priority.load(),
            local_novelty: Cell::new(Moments::default()),
            local_priority: Cell::new(Moments::default()),
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Fold one scored candidate into this run's moments.
    #[inline]
    pub fn observe(&self, novelty: f64, priority: f64) {
        let mut n = self.local_novelty.get();
        n.observe(novelty);
        self.local_novelty.set(n);
        let mut p = self.local_priority.get();
        p.observe(priority);
        self.local_priority.set(p);
    }

    /// Standardized combination of the two terms, squashed into (0, 1) so it
    /// stays a positive weight for proportional selection and keeps the same
    /// range as an unstandardized score. `priority_weight` scales priority
    /// relative to novelty; 1.0 gives them equal authority.
    pub fn combine(&self, novelty: f64, priority: f64, priority_weight: f64) -> f64 {
        let z_novelty = self
            .prior_novelty
            .plus(self.local_novelty.get())
            .standardize(novelty);
        let z_priority = self
            .prior_priority
            .plus(self.local_priority.get())
            .standardize(priority);
        let w = priority_weight.max(0.0);
        let z = (z_novelty + w * z_priority) / (1.0 + w);
        1.0 / (1.0 + (-z).exp())
    }

    /// Publish this run's moments to the session so later runs start from them.
    pub fn merge_into(&self, global: &GlobalScoreNorm) {
        if !self.enabled {
            return;
        }
        global.novelty.merge(self.local_novelty.get());
        global.priority.merge(self.local_priority.get());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(norm: &ScoreNorm, samples: &[(f64, f64)]) {
        for &(n, p) in samples {
            norm.observe(n, p);
        }
    }

    #[test]
    fn disabled_norm_reports_disabled_and_merges_nothing() {
        let global = GlobalScoreNorm::default();
        let norm = ScoreNorm::new(false, &global);
        assert!(!norm.enabled());
        norm.observe(0.5, 0.5);
        norm.merge_into(&global);
        assert_eq!(global.novelty.load().count, 0.0);
    }

    #[test]
    fn a_term_without_enough_samples_is_neutral() {
        let global = GlobalScoreNorm::default();
        let norm = ScoreNorm::new(true, &global);
        assert_eq!(norm.combine(0.9, 0.1, 1.0), 0.5);
    }

    #[test]
    fn tiny_novelty_spread_still_orders_candidates() {
        let global = GlobalScoreNorm::default();
        let norm = ScoreNorm::new(true, &global);
        // Novelty varies by 1e-4 while priority varies by 1.0.
        let samples: Vec<(f64, f64)> = (0..40)
            .map(|i| (1.0 - (i % 2) as f64 * 1e-4, (i % 4) as f64 / 3.0))
            .collect();
        feed(&norm, &samples);

        let common = norm.combine(1.0 - 1e-4, 0.5, 1.0);
        let novel = norm.combine(1.0, 0.5, 1.0);
        assert!(
            novel > common,
            "novelty should separate candidates: {} vs {}",
            novel,
            common
        );
        // The separation is on the same order as a priority difference.
        let higher_priority = norm.combine(1.0 - 1e-4, 1.0, 1.0);
        assert!((novel - common).abs() > 0.1 * (higher_priority - common).abs());
    }

    #[test]
    fn combined_score_stays_a_positive_weight() {
        let global = GlobalScoreNorm::default();
        let norm = ScoreNorm::new(true, &global);
        feed(
            &norm,
            &(0..40)
                .map(|i| ((i % 3) as f64 / 2.0, (i % 5) as f64 / 4.0))
                .collect::<Vec<_>>(),
        );
        for &(n, p) in &[(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)] {
            let s = norm.combine(n, p, 5.0);
            assert!(s > 0.0 && s < 1.0, "score out of range: {}", s);
        }
    }

    #[test]
    fn priority_weight_shifts_authority_toward_priority() {
        let global = GlobalScoreNorm::default();
        let norm = ScoreNorm::new(true, &global);
        feed(
            &norm,
            &(0..40)
                .map(|i| ((i % 2) as f64, (i % 2) as f64))
                .collect::<Vec<_>>(),
        );
        let balanced = norm.combine(0.0, 1.0, 1.0);
        let priority_led = norm.combine(0.0, 1.0, 8.0);
        assert!(priority_led > balanced);
    }

    #[test]
    fn run_moments_accumulate_into_the_session() {
        let global = GlobalScoreNorm::default();
        for _ in 0..3 {
            let norm = ScoreNorm::new(true, &global);
            feed(&norm, &[(1.0, 0.5), (0.0, 0.25), (1.0, 0.5), (0.0, 0.25)]);
            norm.merge_into(&global);
        }
        assert_eq!(global.novelty.load().count, 12.0);
        assert_eq!(global.priority.load().sum, 4.5);

        // A later run sees the accumulated scale.
        let norm = ScoreNorm::new(true, &global);
        assert!(norm.prior_novelty.count >= MIN_SAMPLES);
        assert!(norm.combine(1.0, 0.5, 1.0) > norm.combine(0.0, 0.5, 1.0));
    }
}
