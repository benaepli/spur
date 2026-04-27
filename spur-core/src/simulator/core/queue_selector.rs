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
    Probabilistic { p_local: f64, p_timer: f64 },
    Preemptive { p_timer: f64, preempt_interval: i32 },
}

impl Default for QueuePolicyConfig {
    fn default() -> Self {
        QueuePolicyConfig::Probabilistic {
            p_local: 0.80,
            p_timer: 0.03,
        }
    }
}

impl QueuePolicyConfig {
    pub fn into_selector(&self) -> AnySelector {
        match self {
            QueuePolicyConfig::Probabilistic { p_local, p_timer } => {
                AnySelector::Probabilistic(ProbabilisticSelector {
                    p_local: *p_local,
                    p_timer: *p_timer,
                })
            }
            QueuePolicyConfig::Preemptive {
                p_timer,
                preempt_interval,
            } => AnySelector::Preemptive(PreemptiveSelector {
                p_timer: *p_timer,
                preempt_interval: *preempt_interval,
                active_node: None,
                steps_since_network_pull: 0,
            }),
        }
    }
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
