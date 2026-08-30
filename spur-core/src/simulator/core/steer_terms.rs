//! The named terms of the scheduling score.
//!
//! A runnable's score is a weighted sum of terms normalised to [0, 1]:
//! novelty and priority, which every runnable carries, plus one term per
//! predicate over the runnable and the run's state. A predicate's weight is
//! its switch: zero removes the term, and the defaults reproduce the fixed
//! blend the scheduler used before terms had names, bit for bit.

use serde::{Deserialize, Serialize};

/// Predicates that can carry a weight, in the order their counters and
/// weights are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    /// A crash of a node whose most recent handler, woken by a timer, has
    /// sends still undelivered.
    CrashAfterTimerSends,
    /// The same crash where the handler was woken by a delivery.
    CrashAfterDeliverySends,
    /// A delivery from an incarnation that has since restarted, arriving
    /// after the receiver's state moved on.
    StaleLate,
    /// A client request delivered while a delivery from a restarted sender
    /// is still undelivered somewhere.
    RequestBeforeStale,
}

pub const TERMS: usize = 4;

impl Term {
    pub const ALL: [Term; TERMS] = [
        Term::CrashAfterTimerSends,
        Term::CrashAfterDeliverySends,
        Term::StaleLate,
        Term::RequestBeforeStale,
    ];

    #[inline]
    pub fn index(self) -> usize {
        match self {
            Term::CrashAfterTimerSends => 0,
            Term::CrashAfterDeliverySends => 1,
            Term::StaleLate => 2,
            Term::RequestBeforeStale => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Term::CrashAfterTimerSends => "crash_after_timer_sends",
            Term::CrashAfterDeliverySends => "crash_after_delivery_sends",
            Term::StaleLate => "stale_late",
            Term::RequestBeforeStale => "request_before_stale",
        }
    }
}

/// The score terms as they appear in a config. `recover_crashed` left unset
/// reads the legacy `quick_fire_multiplier` key, so a config written before
/// this block existed keeps its meaning.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SteerTerms {
    #[serde(default = "default_novelty")]
    pub novelty: f64,
    #[serde(default = "default_priority")]
    pub priority: f64,
    /// Multiplies the priority weight of a recover for a node that is down.
    #[serde(default)]
    pub recover_crashed: Option<f64>,
    #[serde(default)]
    pub crash_after_timer_sends: f64,
    #[serde(default)]
    pub crash_after_delivery_sends: f64,
    #[serde(default)]
    pub stale_late: f64,
    #[serde(default)]
    pub request_before_stale: f64,
}

fn default_novelty() -> f64 {
    0.25
}

fn default_priority() -> f64 {
    0.75
}

/// The value the legacy key defaults to; a legacy key at this value is taken
/// as unset when the terms block names the multiplier itself.
pub const LEGACY_RECOVER_CRASHED: f64 = 5.0;

impl Default for SteerTerms {
    fn default() -> Self {
        Self {
            novelty: default_novelty(),
            priority: default_priority(),
            recover_crashed: None,
            crash_after_timer_sends: 0.0,
            crash_after_delivery_sends: 0.0,
            stale_late: 0.0,
            request_before_stale: 0.0,
        }
    }
}

impl SteerTerms {
    /// Fix every weight for a session. `legacy_quick_fire` is the value of the
    /// `quick_fire_multiplier` key, which `recover_crashed` supersedes; the two
    /// may not disagree when both are set.
    pub fn resolve(&self, legacy_quick_fire: f64) -> Result<ResolvedTerms, String> {
        let weights = [
            self.crash_after_timer_sends,
            self.crash_after_delivery_sends,
            self.stale_late,
            self.request_before_stale,
        ];
        for (t, w) in Term::ALL.iter().zip(weights.iter()) {
            if !w.is_finite() || *w < 0.0 {
                return Err(format!(
                    "steer_terms.{} must be a finite number >= 0 (got {})",
                    t.name(),
                    w
                ));
            }
        }
        for (name, w) in [("novelty", self.novelty), ("priority", self.priority)] {
            if !w.is_finite() || w < 0.0 {
                return Err(format!(
                    "steer_terms.{} must be a finite number >= 0 (got {})",
                    name, w
                ));
            }
        }
        if self.novelty + self.priority <= 0.0 {
            return Err("steer_terms: novelty and priority may not both be 0".to_string());
        }
        if !legacy_quick_fire.is_finite() || legacy_quick_fire < 0.0 {
            return Err(format!(
                "quick_fire_multiplier must be a finite number >= 0 (got {})",
                legacy_quick_fire
            ));
        }
        let recover_crashed = match self.recover_crashed {
            Some(x) if !x.is_finite() || x < 0.0 => {
                return Err(format!(
                    "steer_terms.recover_crashed must be a finite number >= 0 (got {})",
                    x
                ));
            }
            Some(x) if legacy_quick_fire != LEGACY_RECOVER_CRASHED && legacy_quick_fire != x => {
                return Err(format!(
                    "steer_terms.recover_crashed ({}) and quick_fire_multiplier ({}) disagree; set one",
                    x, legacy_quick_fire
                ));
            }
            Some(x) => x,
            None => legacy_quick_fire,
        };
        Ok(ResolvedTerms {
            novelty: self.novelty,
            priority: self.priority,
            recover_crashed,
            source: bind(weights),
        })
    }
}

/// The predicate weights when at least one of them can change a score, and
/// nothing otherwise.
fn bind(weights: [f64; TERMS]) -> Option<[f64; TERMS]> {
    weights.iter().any(|&w| w > 0.0).then_some(weights)
}

/// The weights a run scores with. Copied into every run configuration, so
/// nothing at scoring time consults an `Option`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedTerms {
    pub novelty: f64,
    pub priority: f64,
    pub recover_crashed: f64,
    /// The predicate weights, present only when one of them carries weight.
    /// Decided once when the terms are resolved, so a scheduling decision
    /// answers "is there anything to prefer" without reading the weights and
    /// cannot be handed a set of weights that disagrees with the answer.
    source: Option<[f64; TERMS]>,
}

impl Default for ResolvedTerms {
    fn default() -> Self {
        Self {
            novelty: default_novelty(),
            priority: default_priority(),
            recover_crashed: LEGACY_RECOVER_CRASHED,
            source: None,
        }
    }
}

impl ResolvedTerms {
    /// Whether any predicate carries weight, which is the only case in which
    /// scoring consults the run's state at all.
    #[inline]
    pub fn any_predicate(&self) -> bool {
        self.source.is_some()
    }

    #[inline]
    pub fn weight(&self, t: Term) -> f64 {
        match &self.source {
            Some(w) => w[t.index()],
            None => 0.0,
        }
    }

    /// The same terms with another recover multiplier.
    pub fn with_recover_crashed(self, recover_crashed: f64) -> Self {
        Self {
            recover_crashed,
            ..self
        }
    }

    /// The same novelty, priority and recover weighting with no predicate
    /// carrying weight, i.e. the score a run would produce if the terms block
    /// were left out.
    pub fn without_predicates(self) -> Self {
        Self {
            source: None,
            ..self
        }
    }
}

/// Probability that a record with predicate weight `w` beats the best of
/// `competitors` records without it inside one queue, where every priority
/// is drawn from the default record band (centre 0.5, half-width 0.15,
/// Beta(0.5, 0.5)) and novelty is constant. A record with bonus `w` scores
/// `(0.25 + 0.75 p + w) / (1 + w)` against `0.25 + 0.75 p`; from
/// `w >= 0.857` the lowest draw beats the highest and the win is certain.
/// The table in PARAMETERS.md is this function at a few points.
#[cfg(test)]
pub fn within_queue_win_probability(w: f64, competitors: usize, samples: usize, seed: u64) -> f64 {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Beta, Distribution};
    let mut rng = StdRng::seed_from_u64(seed);
    let band = Beta::new(0.5, 0.5).expect("a valid Beta");
    let mut draw = |rng: &mut StdRng| 0.5 + 0.15 * (2.0 * band.sample(rng) - 1.0);
    let mut wins = 0usize;
    for _ in 0..samples {
        let mine = (0.25 + 0.75 * draw(&mut rng) + w) / (1.0 + w);
        let best = (0..competitors)
            .map(|_| 0.25 + 0.75 * draw(&mut rng))
            .fold(f64::NEG_INFINITY, f64::max);
        if mine > best {
            wins += 1;
        }
    }
    wins as f64 / samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The within-queue win table the weight derivation rests on.
    #[test]
    fn within_queue_win_table_reproduces() {
        let table: [(f64, [f64; 4]); 5] = [
            (0.0, [0.500, 0.334, 0.201, 0.109]),
            (0.25, [0.752, 0.612, 0.484, 0.405]),
            (0.5, [0.881, 0.795, 0.692, 0.607]),
            (0.75, [0.969, 0.943, 0.901, 0.847]),
            (1.0, [1.0, 1.0, 1.0, 1.0]),
        ];
        for (w, row) in table {
            for (m, expected) in [1usize, 2, 4, 8].into_iter().zip(row) {
                let p = within_queue_win_probability(w, m, 40_000, 17);
                assert!(
                    (p - expected).abs() < 0.02,
                    "w {w} against {m}: expected {expected:.3}, got {p:.3}"
                );
            }
        }
    }

    #[test]
    fn defaults_reproduce_the_legacy_blend_weights() {
        let r = SteerTerms::default().resolve(LEGACY_RECOVER_CRASHED).unwrap();
        assert_eq!(r, ResolvedTerms::default());
        assert!(!r.any_predicate());
    }

    #[test]
    fn the_terms_block_supersedes_the_legacy_key() {
        let t = SteerTerms {
            recover_crashed: Some(3.0),
            ..SteerTerms::default()
        };
        assert_eq!(t.resolve(LEGACY_RECOVER_CRASHED).unwrap().recover_crashed, 3.0);
        assert_eq!(t.resolve(3.0).unwrap().recover_crashed, 3.0);
        assert!(t.resolve(4.0).unwrap_err().contains("disagree"));
        assert_eq!(SteerTerms::default().resolve(2.0).unwrap().recover_crashed, 2.0);
    }

    #[test]
    fn weights_are_validated() {
        let mut t = SteerTerms {
            stale_late: -1.0,
            ..SteerTerms::default()
        };
        assert!(t.resolve(5.0).unwrap_err().contains("stale_late"));
        t.stale_late = f64::NAN;
        assert!(t.resolve(5.0).is_err());
        t.stale_late = 2.33;
        assert!(t.resolve(5.0).unwrap().any_predicate());
        assert_eq!(t.resolve(5.0).unwrap().weight(Term::StaleLate), 2.33);
        let unbound = t.resolve(5.0).unwrap().without_predicates();
        assert!(!unbound.any_predicate());
        assert_eq!(unbound.weight(Term::StaleLate), 0.0);
        let z = SteerTerms {
            novelty: 0.0,
            priority: 0.0,
            ..SteerTerms::default()
        };
        assert!(z.resolve(5.0).is_err());
    }

    #[test]
    fn a_config_without_the_block_parses_to_the_defaults() {
        let t: SteerTerms = serde_json::from_str("{}").unwrap();
        assert_eq!(t.resolve(5.0).unwrap(), ResolvedTerms::default());
        let t: SteerTerms = serde_json::from_str(r#"{"stale_late": 2.33}"#).unwrap();
        assert_eq!(t.resolve(5.0).unwrap().weight(Term::StaleLate), 2.33);
    }
}
