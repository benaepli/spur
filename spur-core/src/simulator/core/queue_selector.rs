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

#[derive(Debug, Clone)]
pub struct ProbabilisticSelector {
    pub p_local: f64,
    pub p_timer: f64,
}

impl ProbabilisticSelector {
    /// Pick a non-empty local queue index, weighted by queue size.
    fn pick_local(&self, info: &QueueInfo, rng: &mut impl Rng) -> Option<QueueSelection> {
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

    /// Try to select from a specific queue category, falling back to others.
    fn try_select(
        &self,
        primary: usize,
        info: &QueueInfo,
        rng: &mut impl Rng,
    ) -> Option<QueueSelection> {
        let order = match primary {
            0 => [0, 1, 2],
            1 => [1, 0, 2],
            _ => [2, 0, 1],
        };
        for &cat in &order {
            match cat {
                0 => {
                    if let Some(sel) = self.pick_local(info, rng) {
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
pub enum AnySelector {
    Probabilistic(ProbabilisticSelector),
}

impl QueueSelector for AnySelector {
    fn select(&mut self, info: &QueueInfo, rng: &mut impl Rng) -> Option<QueueSelection> {
        match self {
            AnySelector::Probabilistic(s) => s.select(info, rng),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum QueuePolicyConfig {
    Probabilistic { p_local: f64, p_timer: f64 },
}

impl Default for QueuePolicyConfig {
    fn default() -> Self {
        QueuePolicyConfig::Probabilistic {
            p_local: 0.34,
            p_timer: 0.33,
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
        }
    }
}
