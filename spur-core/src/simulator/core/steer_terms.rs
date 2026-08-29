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
            weights,
        })
    }
}

/// The weights a run scores with. Copied into every run configuration, so
/// nothing at scoring time consults an `Option`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedTerms {
    pub novelty: f64,
    pub priority: f64,
    pub recover_crashed: f64,
    pub weights: [f64; TERMS],
}

impl Default for ResolvedTerms {
    fn default() -> Self {
        Self {
            novelty: default_novelty(),
            priority: default_priority(),
            recover_crashed: LEGACY_RECOVER_CRASHED,
            weights: [0.0; TERMS],
        }
    }
}

impl ResolvedTerms {
    /// Whether any predicate carries weight, which is the only case in which
    /// scoring consults the run's state at all.
    #[inline]
    pub fn any_predicate(&self) -> bool {
        self.weights.iter().any(|&w| w > 0.0)
    }

    #[inline]
    pub fn weight(&self, t: Term) -> f64 {
        self.weights[t.index()]
    }

    /// The same terms with another recover multiplier.
    pub fn with_recover_crashed(self, recover_crashed: f64) -> Self {
        Self {
            recover_crashed,
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
