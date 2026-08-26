use rand::Rng;
use serde::Deserialize;

#[derive(Debug, Clone, Copy)]
pub enum QueueSelection {
    Local(usize),
    Network,
    Timer,
}

#[derive(Debug)]
pub struct QueueInfo {
    pub local_queue_sizes: Vec<usize>,
    pub network_queue_size: usize,
    pub timer_queue_size: usize,
    #[allow(dead_code)]
    pub step: i32,
}

impl QueueInfo {
    fn total_local(&self) -> usize {
        self.local_queue_sizes.iter().sum()
    }

    fn total(&self) -> usize {
        self.total_local() + self.network_queue_size + self.timer_queue_size
    }
}

pub trait QueueSelector {
    fn select(&mut self, info: &QueueInfo, rng: &mut impl Rng) -> Option<QueueSelection>;
}

/// Pick a non-empty local queue index, weighted by queue size.
fn pick_local(info: &QueueInfo, rng: &mut impl Rng) -> Option<QueueSelection> {
    let total: usize = info.local_queue_sizes.iter().sum();
    if total == 0 {
        return None;
    }
    let mut target = rng.random_range(0..total);
    for (i, &size) in info.local_queue_sizes.iter().enumerate() {
        if target < size {
            return Some(QueueSelection::Local(i));
        }
        target -= size;
    }
    unreachable!()
}

#[derive(Debug, Clone)]
pub struct ProbabilisticSelector {
    pub p_local: f64,
    pub p_timer: f64,
}

impl ProbabilisticSelector {
    /// Try to select from a specific queue category, falling back to others.
    fn try_select(
        &self,
        primary: usize,
        info: &QueueInfo,
        rng: &mut impl Rng,
    ) -> Option<QueueSelection> {
        let order: [usize; 3] = match primary {
            0 => [0, 1, 2],
            1 => [1, 0, 2],
            _ => [2, 0, 1],
        };
        for &cat in &order {
            match cat {
                0 => {
                    if let Some(sel) = pick_local(info, rng) {
                        return Some(sel);
                    }
                }
                1 => {
                    if info.network_queue_size > 0 {
                        return Some(QueueSelection::Network);
                    }
                }
                2 => {
                    if info.timer_queue_size > 0 {
                        return Some(QueueSelection::Timer);
                    }
                }
                _ => unreachable!(),
            }
        }
        None
    }
}

impl QueueSelector for ProbabilisticSelector {
    fn select(&mut self, info: &QueueInfo, rng: &mut impl Rng) -> Option<QueueSelection> {
        if info.total() == 0 {
            return None;
        }
        let roll: f64 = rng.random();
        let primary = if roll < self.p_local {
            0 // local
        } else if roll < self.p_local + self.p_timer {
            2 // timer
        } else {
            1 // network
        };
        self.try_select(primary, info, rng)
    }
}

#[derive(Debug, Clone)]
pub struct PreemptiveSelector {
    pub p_timer: f64,
    pub preempt_interval: i32,
    active_node: Option<usize>,
    steps_since_network_pull: i32,
}

impl QueueSelector for PreemptiveSelector {
    fn select(&mut self, info: &QueueInfo, rng: &mut impl Rng) -> Option<QueueSelection> {
        if info.total() == 0 {
            return None;
        }

        if info.timer_queue_size > 0 && rng.random::<f64>() < self.p_timer {
            return Some(QueueSelection::Timer);
        }

        if self.steps_since_network_pull >= self.preempt_interval && info.network_queue_size > 0 {
            self.steps_since_network_pull = 0;
            self.active_node = None;
            return Some(QueueSelection::Network);
        }

        if let Some(node) = self.active_node {
            if info.local_queue_sizes.get(node).copied().unwrap_or(0) > 0 {
                self.steps_since_network_pull += 1;
                return Some(QueueSelection::Local(node));
            }
            // Active node drained, clear it
            self.active_node = None;
        }

        if let Some(sel) = pick_local(info, rng) {
            if let QueueSelection::Local(node) = sel {
                self.active_node = Some(node);
            }
            self.steps_since_network_pull += 1;
            return Some(sel);
        }

        if info.network_queue_size > 0 {
            self.steps_since_network_pull = 0;
            return Some(QueueSelection::Network);
        }
        if info.timer_queue_size > 0 {
            return Some(QueueSelection::Timer);
        }
        None
    }
}

#[derive(Debug, Clone)]
pub enum AnySelector {
    Probabilistic(ProbabilisticSelector),
    Preemptive(PreemptiveSelector),
}

impl QueueSelector for AnySelector {
    fn select(&mut self, info: &QueueInfo, rng: &mut impl Rng) -> Option<QueueSelection> {
        match self {
            AnySelector::Probabilistic(s) => s.select(info, rng),
            AnySelector::Preemptive(s) => s.select(info, rng),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum QueuePolicyConfig {
    Probabilistic {
        p_local: f64,
        p_timer: f64,
        /// Multiplier applied to `p_timer`, letting the eagerness of timer
        /// firing be tuned without restating the rest of the policy. A weight
        /// of 1.0 uses `p_timer` as written; 0.0 never prefers the timer queue,
        /// so timers only fire when every other queue is empty.
        #[serde(default = "default_timer_weight")]
        timer_weight: f64,
    },
    Preemptive {
        p_timer: f64,
        preempt_interval: i32,
        #[serde(default = "default_timer_weight")]
        timer_weight: f64,
    },
}

fn default_timer_weight() -> f64 {
    1.0
}

impl Default for QueuePolicyConfig {
    fn default() -> Self {
        QueuePolicyConfig::Probabilistic {
            p_local: 0.80,
            p_timer: 0.03,
            timer_weight: default_timer_weight(),
        }
    }
}

impl QueuePolicyConfig {
    pub fn timer_weight(&self) -> f64 {
        match self {
            QueuePolicyConfig::Probabilistic { timer_weight, .. }
            | QueuePolicyConfig::Preemptive { timer_weight, .. } => *timer_weight,
        }
    }

    pub fn to_selector(&self) -> AnySelector {
        match self {
            QueuePolicyConfig::Probabilistic {
                p_local,
                p_timer,
                timer_weight,
            } => AnySelector::Probabilistic(ProbabilisticSelector {
                p_local: *p_local,
                p_timer: weighted_p_timer(*p_timer, *timer_weight),
            }),
            QueuePolicyConfig::Preemptive {
                p_timer,
                preempt_interval,
                timer_weight,
            } => AnySelector::Preemptive(PreemptiveSelector {
                p_timer: weighted_p_timer(*p_timer, *timer_weight),
                preempt_interval: *preempt_interval,
                active_node: None,
                steps_since_network_pull: 0,
            }),
        }
    }
}

fn weighted_p_timer(p_timer: f64, timer_weight: f64) -> f64 {
    (p_timer * timer_weight.max(0.0)).clamp(0.0, 1.0)
}

/// Within-queue selection method. Decides which runnable, among the eligible
/// items in a single queue, gets executed next. Orthogonal to `QueuePolicyConfig`,
/// which decides *which* queue to draw from.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum WithinQueueSelector {
    /// K-tournament: sample `k` indices uniformly, take the highest score.
    /// Near-greedy for typical k. This is the historical behavior.
    Tournament {
        #[serde(default = "default_tournament_k")]
        k: usize,
    },
    /// Proportional lottery (Waldspurger-style): selection probability is
    /// proportional to `score^exponent`. `exponent = 1.0` is plain proportional;
    /// `exponent = 0.0` is uniform; large `exponent` approaches greedy.
    Proportional {
        #[serde(default = "default_proportional_exponent")]
        exponent: f64,
    },
}

fn default_tournament_k() -> usize {
    10
}

fn default_proportional_exponent() -> f64 {
    1.0
}

impl Default for WithinQueueSelector {
    fn default() -> Self {
        WithinQueueSelector::Tournament {
            k: default_tournament_k(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn p_timer_of(selector: &AnySelector) -> f64 {
        match selector {
            AnySelector::Probabilistic(s) => s.p_timer,
            AnySelector::Preemptive(s) => s.p_timer,
        }
    }

    #[test]
    fn omitted_timer_weight_leaves_p_timer_untouched() {
        let policy: QueuePolicyConfig =
            serde_json::from_str(r#"{"type":"Probabilistic","p_local":0.8,"p_timer":0.03}"#)
                .unwrap();
        assert_eq!(policy.timer_weight(), 1.0);
        assert!((p_timer_of(&policy.to_selector()) - 0.03).abs() < 1e-12);
    }

    #[test]
    fn timer_weight_scales_p_timer_and_clamps() {
        let scaled = QueuePolicyConfig::Probabilistic {
            p_local: 0.8,
            p_timer: 0.03,
            timer_weight: 4.0,
        };
        assert!((p_timer_of(&scaled.to_selector()) - 0.12).abs() < 1e-12);

        let zero = QueuePolicyConfig::Preemptive {
            p_timer: 0.03,
            preempt_interval: 50,
            timer_weight: 0.0,
        };
        assert_eq!(p_timer_of(&zero.to_selector()), 0.0);

        let over = QueuePolicyConfig::Probabilistic {
            p_local: 0.8,
            p_timer: 0.5,
            timer_weight: 100.0,
        };
        assert_eq!(p_timer_of(&over.to_selector()), 1.0);
    }

    #[test]
    fn zero_timer_weight_still_fires_a_lone_timer() {
        let mut selector = QueuePolicyConfig::Probabilistic {
            p_local: 0.8,
            p_timer: 0.03,
            timer_weight: 0.0,
        }
        .to_selector();
        let info = QueueInfo {
            local_queue_sizes: vec![0, 0, 0],
            network_queue_size: 0,
            timer_queue_size: 1,
            step: 0,
        };
        let mut rng = SmallRng::seed_from_u64(7);
        assert!(matches!(
            selector.select(&info, &mut rng),
            Some(QueueSelection::Timer)
        ));
    }
}
