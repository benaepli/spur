use rand::Rng;
use serde::{Deserialize, Serialize};

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

    /// Whether `select_timer_biased` applies its bias rather than ignoring it.
    fn supports_timer_bias(&self) -> bool {
        false
    }

    /// Select with the timer's share of the roll multiplied by `timer_bias`.
    /// A selector that does not support the bias makes the stock selection,
    /// drawing exactly what `select` would draw.
    fn select_timer_biased(
        &mut self,
        info: &QueueInfo,
        _timer_bias: f64,
        rng: &mut impl Rng,
    ) -> Option<QueueSelection> {
        self.select(info, rng)
    }
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

    fn supports_timer_bias(&self) -> bool {
        true
    }

    fn select_timer_biased(
        &mut self,
        info: &QueueInfo,
        timer_bias: f64,
        rng: &mut impl Rng,
    ) -> Option<QueueSelection> {
        if info.total() == 0 {
            return None;
        }
        // The bias trades timer mass against the network share only: the
        // local share is untouched, and the cap keeps the effective timer
        // probability inside the unit interval when p_local is large.
        let p_timer_eff = (self.p_timer * timer_bias).min((1.0 - self.p_local).max(0.0));
        let roll: f64 = rng.random();
        let primary = if roll < self.p_local {
            0 // local
        } else if roll < self.p_local + p_timer_eff {
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

    fn supports_timer_bias(&self) -> bool {
        matches!(self, AnySelector::Probabilistic(_))
    }

    fn select_timer_biased(
        &mut self,
        info: &QueueInfo,
        timer_bias: f64,
        rng: &mut impl Rng,
    ) -> Option<QueueSelection> {
        match self {
            AnySelector::Probabilistic(s) => s.select_timer_biased(info, timer_bias, rng),
            AnySelector::Preemptive(s) => s.select(info, rng),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    pub fn to_selector(&self) -> AnySelector {
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
#[derive(Debug, Clone, Deserialize, Serialize)]
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn contested_info() -> QueueInfo {
        QueueInfo {
            local_queue_sizes: vec![1],
            network_queue_size: 1,
            timer_queue_size: 1,
            step: 0,
        }
    }

    fn frequency(selector: &mut ProbabilisticSelector, bias: Option<f64>, draws: u32) -> f64 {
        let mut rng = StdRng::seed_from_u64(7);
        let info = contested_info();
        let mut timers = 0u32;
        for _ in 0..draws {
            let sel = match bias {
                Some(b) => selector.select_timer_biased(&info, b, &mut rng),
                None => selector.select(&info, &mut rng),
            };
            if matches!(sel, Some(QueueSelection::Timer)) {
                timers += 1;
            }
        }
        f64::from(timers) / f64::from(draws)
    }

    #[test]
    fn identity_bias_matches_the_stock_roll_and_draw_counts_agree() {
        let mut a = ProbabilisticSelector { p_local: 0.80, p_timer: 0.03 };
        let mut b = a.clone();
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let info = contested_info();
        for _ in 0..10_000 {
            let sa = a.select(&info, &mut rng_a);
            let sb = b.select_timer_biased(&info, 1.0, &mut rng_b);
            assert_eq!(format!("{sa:?}"), format!("{sb:?}"));
        }
        // Equal draws so far imply the next draw agrees; a path that drew
        // more or less would desynchronize here.
        assert_eq!(rng_a.random::<u64>(), rng_b.random::<u64>());
    }

    #[test]
    fn draw_count_parity_holds_for_a_non_identity_bias() {
        let mut s = ProbabilisticSelector { p_local: 0.80, p_timer: 0.03 };
        let mut rng_a = StdRng::seed_from_u64(9);
        let mut rng_b = StdRng::seed_from_u64(9);
        let info = contested_info();
        for _ in 0..10_000 {
            s.select(&info, &mut rng_a);
            s.select_timer_biased(&info, 4.0, &mut rng_b);
        }
        assert_eq!(rng_a.random::<u64>(), rng_b.random::<u64>());
    }

    #[test]
    fn bias_scales_the_timer_share_of_the_roll() {
        let mut s = ProbabilisticSelector { p_local: 0.80, p_timer: 0.03 };
        let stock = frequency(&mut s, None, 200_000);
        assert!((stock - 0.03).abs() < 0.005, "stock timer share was {stock}");
        let promoted = frequency(&mut s, Some(4.0), 200_000);
        assert!((promoted - 0.12).abs() < 0.005, "promoted share was {promoted}");
        let suppressed = frequency(&mut s, Some(0.25), 200_000);
        assert!((suppressed - 0.0075).abs() < 0.002, "suppressed share was {suppressed}");
    }

    #[test]
    fn effective_timer_probability_is_capped_by_the_non_local_mass() {
        let mut s = ProbabilisticSelector { p_local: 0.95, p_timer: 0.2 };
        let f = frequency(&mut s, Some(4.0), 200_000);
        assert!((f - 0.05).abs() < 0.005, "capped share was {f}");
    }
}
