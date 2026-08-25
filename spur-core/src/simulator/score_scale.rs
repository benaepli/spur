//! Online standardization of the two terms that make up a runnable's score.
//!
//! The novelty term and the priority term live on unrelated numeric scales: a
//! fixed linear blend of the two lets whichever term happens to spread wider
//! decide every comparison. Standardizing each term by its own running mean
//! and standard deviation puts both on the same scale at the point of choice,
//! so the blend weight expresses intent rather than absorbing a scale mismatch.
//!
//! Estimates are per-thread and reset when a session enables the mechanism.
//! When disabled, `standardized_blend` is a single relaxed atomic load.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Bumped by `set_enabled` so per-thread estimates from an earlier session are
/// discarded the first time a thread scores in the current one.
static SESSION: AtomicU64 = AtomicU64::new(0);

/// Below this many observations the running spread is too noisy to divide by.
const MIN_SAMPLES: u64 = 64;
/// A term that never varies carries no information; treat it as unstandardizable.
const MIN_VARIANCE: f64 = 1e-12;

/// Welford accumulator: count, mean, and sum of squared deviations.
#[derive(Clone, Copy)]
struct Moments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    const EMPTY: Self = Self {
        count: 0,
        mean: 0.0,
        m2: 0.0,
    };

    fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    /// Deviations from the mean in units of the standard deviation, or None
    /// while the estimate is too thin or degenerate to divide by.
    fn z(&self, x: f64) -> Option<f64> {
        if self.count < MIN_SAMPLES {
            return None;
        }
        let var = self.m2 / self.count as f64;
        if var < MIN_VARIANCE {
            return None;
        }
        Some((x - self.mean) / var.sqrt())
    }
}

#[derive(Clone, Copy)]
struct ScaleState {
    session: u64,
    novelty: Moments,
    priority: Moments,
}

thread_local! {
    static SCALE: Cell<ScaleState> = const {
        Cell::new(ScaleState {
            session: 0,
            novelty: Moments::EMPTY,
            priority: Moments::EMPTY,
        })
    };
}

/// Enable or disable standardized scoring for this explorer session.
pub fn set_enabled(on: bool) {
    SESSION.fetch_add(1, Ordering::Relaxed);
    ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

fn with_current<R>(f: impl FnOnce(&mut ScaleState) -> R) -> R {
    let session = SESSION.load(Ordering::Relaxed);
    SCALE.with(|cell| {
        let mut state = cell.get();
        if state.session != session {
            state = ScaleState {
                session,
                novelty: Moments::EMPTY,
                priority: Moments::EMPTY,
            };
        }
        let out = f(&mut state);
        cell.set(state);
        out
    })
}

/// Feed one candidate's raw terms into the running estimates.
#[inline]
pub fn observe(novelty: f64, priority: f64) {
    if !enabled() {
        return;
    }
    with_current(|state| {
        state.novelty.push(novelty);
        state.priority.push(priority);
    });
}

/// Blend the standardized terms into a score in (0, 1), weighting priority by
/// `priority_weight` relative to novelty. Returns None when standardization is
/// off or the estimates are not yet usable, leaving the caller on its raw blend.
#[inline]
pub fn standardized_blend(novelty: f64, priority: f64, priority_weight: f64) -> Option<f64> {
    if !enabled() {
        return None;
    }
    let (zn, zp) = with_current(|state| {
        Some((state.novelty.z(novelty)?, state.priority.z(priority)?))
    })?;
    let z = (zn + priority_weight * zp) / (1.0 + priority_weight);
    Some(1.0 / (1.0 + (-z).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_leaves_scoring_alone() {
        set_enabled(false);
        assert!(standardized_blend(0.5, 0.5, 3.0).is_none());
    }

    #[test]
    fn spread_terms_separate_after_warmup() {
        set_enabled(true);
        for i in 0..MIN_SAMPLES {
            let x = i as f64 / MIN_SAMPLES as f64;
            observe(x * 1e-3, x);
        }
        let low = standardized_blend(0.0, 0.5, 1.0).unwrap();
        let high = standardized_blend(1e-3, 0.5, 1.0).unwrap();
        // A novelty gap of 1e-3 is decisive once both terms share a scale.
        assert!(high - low > 0.3, "low={} high={}", low, high);
        set_enabled(false);
    }

    #[test]
    fn constant_term_is_not_standardized() {
        set_enabled(true);
        for _ in 0..(MIN_SAMPLES * 2) {
            observe(1.0, 0.5);
        }
        assert!(standardized_blend(1.0, 0.5, 1.0).is_none());
        set_enabled(false);
    }
}
