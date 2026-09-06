//! Per-cell selection of the mechanism arm set, learned from a per-run
//! reward read off the finished run.
//!
//! Every run carries one direction on each of five mechanism axes (the
//! `ArmSet`). Each run that is not a probe of either kind is assigned by
//! its id, under a salt of its own, to one of three learners. A learner
//! keeps one discounted Beta posterior per direction for every cell, a
//! cell being one configuration of the grid, shared by every campaign arm
//! that walks the grid, or one arm without a configuration. The assigned learner
//! decides from its cell's state whether the run explores or exploits: its
//! exploration share is the posterior odds against the leader on the
//! cell's most decided axis, and a unit draw from the run id under a
//! further salt below that share makes the run coin-drawn. A coin-drawn
//! run takes the directions the run id's coins name and is the control
//! sample every learner observes. Any other run is a learner run: it picks
//! each axis independently, a direction with probability proportional to
//! the coin's share of that direction times the posterior probability that
//! the direction leads its axis, which is uniform when nothing separates
//! the directions and tends to one for a separated leader. Nothing is
//! drawn from the posteriors: the coin-drawn runs supply the exploration,
//! and their share falls as the cell's leader separates. The probability
//! of leading is computed in closed form, each pairwise term being the
//! normal approximation to the difference of two Beta variables. Each
//! direction's prior is shrunk toward the cell's own discounted reward rate
//! with the warmup count as its weight, so a direction the cell has rarely
//! carried sits near the cell's rate, not near one half. Below the warmup
//! count the exploration share is one, so every run of a young cell is
//! coin-drawn.
//!
//! Each learner has a reward of its own, a per-run bool the scheduler sets
//! when the run reached the shape the learner is after. A learner observes
//! every coin-drawn run, whichever learner drew it, and its own learner
//! runs, credited to the five directions the run carried, and never another
//! learner's runs, so the three learners are read against one common
//! control. Further rewards are read on the coin-drawn runs for calibration
//! and steer nothing.
//!
//! The selector's categorical draws come from a generator seeded from the
//! run's schedule seed under a salt of its own, so the run's schedule
//! stream is untouched and a coin-drawn run draws exactly what it would
//! without the selector. Whether a run is coin-drawn, and a learner run's
//! arm set, depend on the learner's state at draw time, so neither is a
//! function of the run id alone; the run's tag records the arms it ran
//! under and which learner, if any, drew them.

use crate::simulator::rng::derive_seed;
use crate::simulator::run_cap;
use crate::simulator::run_phase;
use crate::simulator::run_variant::{AXES, AXIS_START, ArmSet, DIRECTIONS};
use crate::simulator::timer_context;
use crate::simulator::util_stats::{self, Reward};
use dashmap::DashMap;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::sync::LazyLock;

/// Salt for the assignment of a run to a learner. Distinct from every other
/// split of a session.
pub const SELECTOR_SALT: u64 = 0x_4152_4D53_454C_4354; // "ARMSELCT"

/// Salt for the selector's own generator, derived per run from the schedule
/// seed so the draw is reproducible given the learner's state.
const DRAW_SALT: u64 = 0x_4152_4D44_5241_5753; // "ARMDRAWS"

/// Salt for the unit draw that decides whether an assigned run explores
/// under its coins or takes its learner's pick. Independent of the learner
/// assignment.
pub const EXPLORE_SALT: u64 = 0x_4152_4D45_5850_4C52; // "ARMEXPLR"

/// Per-observation discount on the posteriors of the directions a run
/// carried, so a cell follows its recent reward rate rather than its whole
/// history. Every reward is rare, so the window is about five hundred
/// observations per cell.
pub const DISCOUNT: f64 = 0.998;

/// Observations a cell needs before its learner reads an exploration share
/// from the posteriors; below it every run of the cell is coin-drawn.
pub const WARMUP_OBSERVATIONS: u64 = 24;

/// A cell key: a campaign arm index and a configuration index. A run at a
/// configuration of the grid is keyed by the configuration alone, under
/// `POOLED_ARM`, so every campaign arm walking the grid (and standard mode,
/// whose arm index is -1) reads and credits one learner state per
/// configuration; a run without a configuration keeps its arm's own cell.
pub type Cell = (i32, i32);

/// The arm slot of every configuration-keyed cell. Below every arm index,
/// so it never collides with an arm's own cell.
pub const POOLED_ARM: i32 = i32::MIN;

/// The cell a run is read from and credited to: `(POOLED_ARM, config_index)`
/// when the run has a configuration, `(arm_index, -1)` otherwise.
pub fn cell(arm_index: i32, config_index: i32) -> Cell {
    if config_index >= 0 {
        (POOLED_ARM, config_index)
    } else {
        (arm_index, -1)
    }
}

/// Whether a cell is shared by every arm at its configuration.
pub fn is_pooled(cell: Cell) -> bool {
    cell.0 == POOLED_ARM
}

/// The three learners, each assigned one third of the runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Learner {
    /// Rewarded when a message from a dead incarnation of a restarted sender
    /// lands on a restarted receiver behind the sender's newer traffic and
    /// the handler writes state.
    OvertakenGhost,
    /// Rewarded when a node crashes while marked as having acted on a
    /// fault-crossing delivery, recovers, and then takes a message from
    /// another restarted node's current incarnation.
    AbsorberCycle,
    /// Rewarded when the absorber cycle closed before the first message
    /// entry caused by a client operation invoked after the run's first
    /// crash reached a server.
    CycleBeforeRequest,
}

impl Learner {
    pub const ALL: [Learner; 3] = [
        Learner::OvertakenGhost,
        Learner::AbsorberCycle,
        Learner::CycleBeforeRequest,
    ];

    /// The learner's position in `ALL`.
    pub fn index(self) -> usize {
        match self {
            Learner::OvertakenGhost => 0,
            Learner::AbsorberCycle => 1,
            Learner::CycleBeforeRequest => 2,
        }
    }

    /// The reward this learner is trained on.
    pub fn reward(self) -> Reward {
        match self {
            Learner::OvertakenGhost => Reward::OvertakenGhost,
            Learner::AbsorberCycle => Reward::AbsorberCycle,
            Learner::CycleBeforeRequest => Reward::CycleBeforeRequest,
        }
    }
}

/// The learner a run is assigned to, or None for a probe of either kind:
/// run-cap probes feed the length learners and timer-context probes must
/// run unsteered, so neither is steered nor observed. A pure function of
/// the run id; whether the assigned learner steers the run is decided in
/// `choose`.
pub fn learner(run_id: i64) -> Option<Learner> {
    if run_cap::is_probe(run_id) || timer_context::run_mode(run_id) == timer_context::RunMode::Probe
    {
        return None;
    }
    Some(match run_phase::salted_phase(run_id, SELECTOR_SALT, 3) {
        0 => Learner::OvertakenGhost,
        1 => Learner::AbsorberCycle,
        _ => Learner::CycleBeforeRequest,
    })
}

/// What the selector decided for a run: the arm set it runs under and, on
/// a learner run, the learner whose pick it took. A coin-drawn run and a
/// probe carry no learner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    pub arms: ArmSet,
    pub learner: Option<Learner>,
}

/// The standard normal distribution function, by a rational approximation
/// of the error function accurate to about 1e-7. Exactly one half at zero,
/// so directions with one posterior get one weight.
fn normal_cdf(z: f64) -> f64 {
    if z == 0.0 {
        return 0.5;
    }
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-x * x).exp();
    if z >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
}

/// The per-run facts the rewards are read from.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rewards {
    pub overtaken_ghost: bool,
    pub absorber_cycle: bool,
    pub mutual_absorber_cycle: bool,
    /// Whether the run's ghost signal fired, or None on a run whose prefix
    /// replays a recorded tape: the signal is then the parent's, not the
    /// arms', and the run is not read for it.
    pub ghost_signal: Option<bool>,
    pub cycle_before_request: bool,
    pub exchange_before_request: bool,
    /// The step of the first message entry at a server caused by a
    /// post-fault client operation, when the run had one.
    pub first_post_fault_request_entry_step: Option<i32>,
}

/// One cell's learner state.
struct CellLearner {
    /// Discounted reward and non-reward mass per direction.
    alpha: [f64; DIRECTIONS],
    beta: [f64; DIRECTIONS],
    /// Discounted reward mass and observation mass over every run the cell
    /// has taken, whichever directions it carried; their ratio is the
    /// cell's running reward rate.
    reward_mass: f64,
    mass: f64,
    /// Runs observed in this cell.
    observations: u64,
}

impl CellLearner {
    fn new() -> Self {
        Self {
            alpha: [0.0; DIRECTIONS],
            beta: [0.0; DIRECTIONS],
            reward_mass: 0.0,
            mass: 0.0,
            observations: 0,
        }
    }

    /// The cell's discounted running reward rate; one half before the
    /// first observation, when nothing draws from the cell.
    fn mu(&self) -> f64 {
        if self.mass > 0.0 { self.reward_mass / self.mass } else { 0.5 }
    }

    /// The Beta parameters of a direction: a unit flat prior, a prior of
    /// `WARMUP_OBSERVATIONS` pseudo-counts placed at the cell's running
    /// reward rate, and the direction's own discounted counts. A direction
    /// with no data of its own therefore sits near the cell's rate rather
    /// than at one half.
    fn posterior(&self, d: usize) -> (f64, f64) {
        let m = WARMUP_OBSERVATIONS as f64;
        let mu = self.mu();
        (1.0 + m * mu + self.alpha[d], 1.0 + m * (1.0 - mu) + self.beta[d])
    }

    /// The posterior mean of a direction.
    fn mean(&self, d: usize) -> f64 {
        let (a, b) = self.posterior(d);
        a / (a + b)
    }

    /// The posterior variance of a direction.
    fn variance(&self, d: usize) -> f64 {
        let (a, b) = self.posterior(d);
        a * b / ((a + b) * (a + b) * (a + b + 1.0))
    }

    /// The posterior probability that direction `d` has a higher reward
    /// rate than direction `j`, treating the difference of the two Beta
    /// variables as normal with their means and variances. One half when
    /// the two posteriors coincide.
    fn leads_pairwise(&self, d: usize, j: usize) -> f64 {
        let spread = (self.variance(d) + self.variance(j)).sqrt();
        normal_cdf((self.mean(d) - self.mean(j)) / spread)
    }

    /// For each direction of axis `a`, the product over the axis's other
    /// directions of the pairwise probability of leading them. Slots past
    /// the axis's width are zero. On a binary axis this is the probability
    /// of being the better direction; on a three-way axis the products are
    /// used as relative weights without normalization.
    fn leading_probability(&self, a: usize) -> [f64; 3] {
        let range = AXIS_START[a]..AXIS_START[a + 1];
        let mut out = [0.0f64; 3];
        for (slot, d) in range.clone().enumerate() {
            out[slot] = range
                .clone()
                .filter(|&j| j != d)
                .map(|j| self.leads_pairwise(d, j))
                .product();
        }
        out
    }

    /// The pairwise probability that the direction of axis `a` with the
    /// highest posterior mean leads the direction with the next highest.
    /// Never below one half, since the leader's mean is the higher one.
    fn leader_margin(&self, a: usize) -> f64 {
        let lead = self.leader(a);
        let runner_up = (AXIS_START[a]..AXIS_START[a + 1])
            .filter(|&d| d != lead)
            .max_by(|&x, &y| self.mean(x).total_cmp(&self.mean(y)))
            .expect("every axis has at least two directions");
        self.leads_pairwise(lead, runner_up)
    }

    /// The direction of axis `a` with the highest posterior mean; the
    /// lowest index wins a tie.
    fn leader(&self, a: usize) -> usize {
        let mut best = AXIS_START[a];
        for d in AXIS_START[a] + 1..AXIS_START[a + 1] {
            if self.mean(d) > self.mean(best) {
                best = d;
            }
        }
        best
    }

    /// The share of the cell's runs the learner leaves to the coins, with
    /// the leader margin of every axis: the posterior odds against the
    /// leader, `(1 - m) / m` for margin `m`, on the axis where the margin
    /// is highest. One when no axis is separated; a separated leader on a
    /// single axis lowers it by itself, since an axis whose directions are
    /// equal within resolution has odds of one and never holds the share
    /// up.
    fn explore_share(&self) -> (f64, [f64; AXES]) {
        let mut margins = [0.0f64; AXES];
        let mut share = 1.0f64;
        for (a, m) in margins.iter_mut().enumerate() {
            *m = self.leader_margin(a);
            share = share.min((1.0 - *m) / *m);
        }
        (share, margins)
    }

    /// Pick one direction of axis `a` with probability proportional to the
    /// coin share times the posterior probability of leading the axis. No
    /// value is drawn from the posterior: the coin-drawn runs supply the
    /// exploration, so a learner run exploits what the cell has learned,
    /// and the generator serves the categorical pick only.
    fn sample_axis_leading(&self, a: usize, rng: &mut SmallRng) -> usize {
        let range = AXIS_START[a]..AXIS_START[a + 1];
        let mut weights = self.leading_probability(a);
        for (slot, d) in range.enumerate() {
            weights[slot] *= ArmSet::coin_probability(d);
        }
        Self::pick(a, &weights, rng)
    }

    /// One categorical draw over the directions of axis `a` with the given
    /// weights, slot by slot from the axis's first direction. The first
    /// direction when no weight is positive.
    fn pick(a: usize, weights: &[f64; 3], rng: &mut SmallRng) -> usize {
        let range = AXIS_START[a]..AXIS_START[a + 1];
        let total: f64 = weights[..range.len()].iter().sum();
        if total <= 0.0 {
            return AXIS_START[a];
        }
        let mut u = rng.random::<f64>() * total;
        for (slot, d) in range.clone().enumerate() {
            u -= weights[slot];
            if u < 0.0 {
                return d;
            }
        }
        AXIS_START[a + 1] - 1
    }

    /// Fold one run's reward into the directions it carried.
    fn credit(&mut self, arms: &ArmSet, reward: bool) {
        self.observations += 1;
        let r = reward as u8 as f64;
        self.reward_mass = DISCOUNT * self.reward_mass + r;
        self.mass = DISCOUNT * self.mass + 1.0;
        for d in arms.directions() {
            self.alpha[d] = DISCOUNT * self.alpha[d] + r;
            self.beta[d] = DISCOUNT * self.beta[d] + (1.0 - r);
        }
    }
}

static CELLS: [LazyLock<DashMap<Cell, CellLearner>>; 3] = [
    LazyLock::new(DashMap::new),
    LazyLock::new(DashMap::new),
    LazyLock::new(DashMap::new),
];

/// The arm set a run takes and the learner that drew it. A probe takes its
/// coins. Any other run is assigned to a learner; the run is coin-drawn
/// when its unit draw under `EXPLORE_SALT` falls below that learner's
/// exploration share in the cell, which is one below warmup, and takes its
/// learner's pick on every axis otherwise. The draw reads nothing from the
/// run's schedule stream. `arm_index` is the campaign arm the run ran
/// under, for the per-arm counters; `cell` is where its learner state
/// lives, which several arms may share.
pub fn choose(run_id: i64, schedule_seed: u64, arm_index: i32, cell: Cell) -> Choice {
    let coins = ArmSet::coins(run_id);
    let Some(learner) = learner(run_id) else {
        return Choice { arms: coins, learner: None };
    };
    let slot = learner.index();
    let pooled = is_pooled(cell);
    let warmed = CELLS[slot]
        .get(&cell)
        .filter(|state| state.observations >= WARMUP_OBSERVATIONS);
    let Some(state) = warmed else {
        util_stats::record_arm_selector_draw(slot, arm_index, pooled, 1.0, None, true);
        return Choice { arms: coins, learner: None };
    };
    let (share, margins) = state.explore_share();
    let coin = run_phase::salted_unit(run_id, EXPLORE_SALT) < share;
    let top_margin = margins.iter().copied().fold(0.0, f64::max);
    util_stats::record_arm_selector_draw(slot, arm_index, pooled, share, Some(top_margin), coin);
    if coin {
        return Choice { arms: coins, learner: None };
    }
    let mut rng = SmallRng::seed_from_u64(derive_seed(schedule_seed, run_id, DRAW_SALT));
    let mut directions = [0usize; AXES];
    let mut agreements = 0u64;
    for (a, d) in directions.iter_mut().enumerate() {
        *d = state.sample_axis_leading(a, &mut rng);
        agreements += (*d == state.leader(a)) as u64;
    }
    let arms = ArmSet::from_directions(directions);
    util_stats::record_arm_selector_learner_run(learner.reward(), &arms, &coins, agreements, &margins);
    Choice {
        arms,
        learner: Some(learner),
    }
}

fn credit(learner: Learner, cell: Cell, arms: &ArmSet, reward: bool) {
    let pooled = is_pooled(cell);
    CELLS[learner.index()]
        .entry(cell)
        .or_insert_with(|| {
            util_stats::record_arm_selector_cell_created(learner.reward(), pooled);
            CellLearner::new()
        })
        .credit(arms, reward);
    if pooled {
        util_stats::record_arm_selector_pooled_credit(learner.index());
    }
}

/// Fold one finished run into the learners that may see it: a coin-drawn
/// run into every learner, a learner run into its learner only, a probe
/// into none. The coin-drawn runs are also where every reward's rate per
/// direction, and the step of the first request-caused entry, are read.
/// `arm_index` is the campaign arm the run ran under, for the per-arm
/// counters; `cell` is where the credit lands.
pub fn observe(arm_index: i32, cell: Cell, run_id: i64, choice: &Choice, rewards: &Rewards) {
    if learner(run_id).is_none() {
        return;
    }
    let arms = &choice.arms;
    let value = |reward: Reward| match reward {
        Reward::OvertakenGhost => Some(rewards.overtaken_ghost),
        Reward::AbsorberCycle => Some(rewards.absorber_cycle),
        Reward::MutualAbsorberCycle => Some(rewards.mutual_absorber_cycle),
        Reward::GhostSignal => rewards.ghost_signal,
        Reward::EitherShape => Some(rewards.overtaken_ghost || rewards.absorber_cycle),
        Reward::CycleBeforeRequest => Some(rewards.cycle_before_request),
        Reward::ExchangeBeforeRequest => Some(rewards.exchange_before_request),
    };
    util_stats::record_arm_selector_run_observed(arm_index, rewards.overtaken_ghost);
    match choice.learner {
        None => {
            for learner in Learner::ALL {
                credit(learner, cell, arms, rewards_of(learner, rewards));
            }
            for reward in Reward::ALL {
                if let Some(r) = value(reward) {
                    util_stats::record_arm_selector_observation(reward, false, arm_index, arms, r);
                }
            }
            if let Some(step) = rewards.first_post_fault_request_entry_step {
                util_stats::record_client_anchor_first_post_fault_entry(arms, step);
            }
        }
        Some(learner) => {
            let r = rewards_of(learner, rewards);
            credit(learner, cell, arms, r);
            util_stats::record_arm_selector_observation(learner.reward(), true, arm_index, arms, r);
        }
    }
}

fn rewards_of(learner: Learner, rewards: &Rewards) -> bool {
    match learner {
        Learner::OvertakenGhost => rewards.overtaken_ghost,
        Learner::AbsorberCycle => rewards.absorber_cycle,
        Learner::CycleBeforeRequest => rewards.cycle_before_request,
    }
}

/// Clear every cell so explorer sessions in one process do not share
/// learners.
pub fn reset() {
    for cells in &CELLS {
        cells.clear();
    }
    util_stats::reset_arm_selector_gauges();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::config_override;
    use crate::simulator::fault_timing;
    use crate::simulator::rng::StreamRng;
    use crate::simulator::run_variant::{COMBINATIONS, CrashArm};
    use rand::RngCore;
    use rand_distr::{Beta, Distribution};

    /// Counts draws so a test can assert a path consumed none.
    struct CountingRng {
        inner: SmallRng,
        draws: u64,
    }

    impl RngCore for CountingRng {
        fn next_u32(&mut self) -> u32 {
            self.draws += 1;
            self.inner.next_u32()
        }
        fn next_u64(&mut self) -> u64 {
            self.draws += 1;
            self.inner.next_u64()
        }
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            self.draws += 1;
            self.inner.fill_bytes(dst)
        }
    }

    impl StreamRng for CountingRng {}

    fn id_where(pred: impl Fn(i64) -> bool) -> i64 {
        (0..1_000_000i64).find(|&id| pred(id)).expect("every kind of run is reachable")
    }

    fn unprobed_id() -> i64 {
        id_where(|id| learner(id).is_some())
    }

    fn assigned_id(l: Learner) -> i64 {
        id_where(|id| learner(id) == Some(l))
    }

    fn coin(arms: ArmSet) -> Choice {
        Choice {
            arms,
            learner: None,
        }
    }

    fn learned(arms: ArmSet, l: Learner) -> Choice {
        Choice {
            arms,
            learner: Some(l),
        }
    }

    fn rewarded(overtaken_ghost: bool, absorber_cycle: bool) -> Rewards {
        Rewards {
            overtaken_ghost,
            absorber_cycle,
            ghost_signal: Some(false),
            ..Rewards::default()
        }
    }

    /// Feed `n` coin-drawn observations of `arms` with every learner's
    /// reward set to `reward`.
    fn feed(cell: Cell, n: usize, arms: &ArmSet, reward: bool) {
        let id = unprobed_id();
        for _ in 0..n {
            observe(
                cell.0,
                cell,
                id,
                &coin(*arms),
                &Rewards {
                    cycle_before_request: reward,
                    ..rewarded(reward, reward)
                },
            );
        }
    }

    /// Five hundred coin-drawn runs on each direction of the retarget axis,
    /// interleaved, rewarded for every learner at one in fifty without the
    /// retarget and one in seventeen with it.
    fn separate_retarget(cell: Cell) {
        let id = unprobed_id();
        for i in 0..1000usize {
            let retarget = i % 2 == 1;
            let k = i / 2;
            let reward = if retarget { k % 17 == 0 } else { k % 50 == 0 };
            let arms = ArmSet {
                retarget,
                ..ArmSet::default()
            };
            observe(
                cell.0,
                cell,
                id,
                &coin(arms),
                &Rewards {
                    cycle_before_request: reward,
                    ..rewarded(reward, reward)
                },
            );
        }
    }

    fn share_of(l: Learner, cell: Cell) -> f64 {
        CELLS[l.index()].get(&cell).expect("the cell exists").explore_share().0
    }

    #[test]
    fn the_thirds_partition_the_unprobed_ids_and_spare_every_probe() {
        let n = 64_000i64;
        let mut counts = [0i64; 3];
        for id in 0..n {
            let probe = run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe;
            match learner(id) {
                None => assert!(probe, "run {id}: an unprobed run has no learner"),
                Some(l) => {
                    assert!(!probe, "run {id}: a probe has a learner");
                    counts[l.index()] += 1;
                }
            }
        }
        let unprobed: i64 = counts.iter().sum();
        assert!(unprobed > 50_000, "unprobed {unprobed} of {n}");
        for (i, c) in counts.iter().enumerate() {
            let share = *c as f64 / unprobed as f64;
            assert!((share - 1.0 / 3.0).abs() < 0.02, "learner {i} holds {share}");
        }
    }

    #[test]
    fn a_run_with_a_configuration_is_keyed_by_it_and_a_run_without_one_by_its_arm() {
        assert!(POOLED_ARM < -1, "the shared slot sits below every arm index");
        for k in 0..54 {
            for arm in [-1, 0, 1, 2, 3, 7] {
                assert_eq!(cell(arm, k), (POOLED_ARM, k), "arm {arm} at configuration {k}");
                assert!(is_pooled(cell(arm, k)));
            }
        }
        assert_ne!(cell(0, 0), cell(0, 1), "configurations keep separate cells");
        assert_eq!(cell(4, -1), (4, -1));
        assert_eq!(cell(-1, -1), (-1, -1));
        assert_ne!(cell(4, -1), cell(5, -1), "arms without a configuration keep their own cells");
        assert!(!is_pooled(cell(4, -1)));
    }

    #[test]
    fn arms_at_one_configuration_share_a_cell_and_an_arm_without_one_keeps_its_own() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        util_stats::set_enabled(true);
        reset();
        let before = util_stats::snapshot().arm_selector_axis;
        let shared = cell(0, 5);
        assert_eq!(shared, cell(1, 5));
        let own = cell(4, -1);
        let arms = ArmSet::default();
        let id = unprobed_id();
        // Half the warmup through one arm, the rest through another: the
        // one cell warms up and every learner holds exactly one cell.
        let half = WARMUP_OBSERVATIONS / 2;
        for _ in 0..half {
            observe(0, shared, id, &coin(arms), &rewarded(false, false));
        }
        for _ in half..WARMUP_OBSERVATIONS {
            observe(1, shared, id, &coin(arms), &rewarded(false, false));
        }
        for l in Learner::ALL {
            assert_eq!(CELLS[l.index()].len(), 1, "{l:?} holds more than the shared cell");
            let state = CELLS[l.index()].get(&shared).expect("the shared cell exists");
            assert_eq!(state.observations, WARMUP_OBSERVATIONS, "{l:?}");
        }
        // The arm without a configuration lands in a cell of its own.
        observe(4, own, id, &coin(arms), &rewarded(false, false));
        for l in Learner::ALL {
            assert_eq!(CELLS[l.index()].len(), 2, "{l:?}");
            assert_eq!(CELLS[l.index()].get(&own).expect("the arm's own cell exists").observations, 1);
        }
        // A third arm at the configuration reads the warmed shared cell and
        // steers some of its runs; the young own cell steers none.
        let assigned: Vec<i64> = (0..20_000i64).filter(|&r| learner(r).is_some()).collect();
        let steered = assigned.iter().filter(|&&r| choose(r, 11, 2, shared).learner.is_some()).count();
        assert!(steered > 0, "no run is steered from the shared cell past warmup");
        for &r in &assigned {
            assert_eq!(choose(r, 11, 4, own), coin(ArmSet::coins(r)), "run {r}: steered below warmup");
        }
        let after = util_stats::snapshot().arm_selector_axis;
        util_stats::set_enabled(false);
        assert_eq!(after.pooled.cells, 1);
        assert_eq!(after.pooled.cells_by_learner, vec![1, 1, 1]);
        assert_eq!(after.cells, 2);
        assert_eq!(after.overtaken_ghost.cells, 2);
        assert_eq!(after.cycle_before_request.cells, 2);
        for l in 0..3 {
            assert_eq!(
                after.pooled.observations_by_learner[l] - before.pooled.observations_by_learner[l],
                WARMUP_OBSERVATIONS,
                "learner {l}: credits into the shared cell"
            );
        }
        let draws = assigned.len() as u64;
        assert_eq!(after.pooled.draws - before.pooled.draws, draws);
        assert_eq!(after.explore.draws - before.explore.draws, 2 * draws);
        assert_eq!(after.explore.draws_by_arm[3] - before.explore.draws_by_arm[3], draws, "arm 2");
        assert_eq!(after.explore.draws_by_arm[5] - before.explore.draws_by_arm[5], draws, "arm 4");
        assert_eq!(after.explore.warmup_coin_runs - before.explore.warmup_coin_runs, draws);
        reset();
        let cleared = util_stats::snapshot().arm_selector_axis;
        assert_eq!(cleared.pooled.cells, 0);
        assert_eq!(cleared.pooled.cells_by_learner, vec![0, 0, 0]);
        assert_eq!(cleared.cells, 0);
    }

    #[test]
    fn no_run_touches_the_schedule_stream_and_the_choice_is_reproducible() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (7, 3);
        let probe = id_where(run_cap::is_probe);
        let assigned = Learner::ALL.map(assigned_id);
        let schedule = CountingRng {
            inner: SmallRng::seed_from_u64(1),
            draws: 0,
        };
        assert_eq!(choose(probe, 5, cell.0, cell), coin(ArmSet::coins(probe)));
        for id in assigned {
            assert_eq!(choose(id, 5, cell.0, cell), coin(ArmSet::coins(id)), "below warmup: coin-drawn");
        }
        feed(cell, WARMUP_OBSERVATIONS as usize, &ArmSet::default(), false);
        for id in assigned {
            let first = choose(id, 5, cell.0, cell);
            assert_eq!(choose(id, 5, cell.0, cell), first, "the same id and state choose alike");
        }
        assert_eq!(choose(probe, 5, cell.0, cell), coin(ArmSet::coins(probe)));
        assert_eq!(schedule.draws, 0, "the selector must not read the run's stream");
        reset();
    }

    #[test]
    fn every_run_of_a_cell_below_warmup_is_coin_drawn() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (9, 9);
        feed(cell, WARMUP_OBSERVATIONS as usize - 1, &ArmSet::default(), false);
        for l in Learner::ALL {
            let state = CELLS[l.index()].get(&cell).expect("the cell exists");
            assert!(state.observations < WARMUP_OBSERVATIONS);
        }
        for id in 0..20_000i64 {
            let choice = choose(id, 11, cell.0, cell);
            assert_eq!(choice, coin(ArmSet::coins(id)), "run {id}: steered below warmup");
        }
        // The observation that completes the warmup lets the learners read
        // a share below one, since the coin-drawn runs carried the stock
        // directions only and the others sit at the cell's rate.
        feed(cell, 1, &ArmSet::default(), false);
        let steered = (0..20_000i64)
            .filter(|&id| choose(id, 11, cell.0, cell).learner.is_some())
            .count();
        assert!(steered > 0, "no run is steered once the warmup is complete");
        reset();
    }

    #[test]
    fn each_learner_sees_the_coin_drawn_runs_and_its_own_runs_only() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (4, 4);
        let arms = ArmSet::default();
        let observations =
            |l: Learner| CELLS[l.index()].get(&cell).map_or(0, |state| state.observations);
        // A learner's runs reach its own cell and no other's.
        for l in Learner::ALL {
            let id = assigned_id(l);
            for _ in 0..WARMUP_OBSERVATIONS {
                observe(cell.0, cell, id, &learned(arms, l), &rewarded(true, true));
            }
            for other in Learner::ALL {
                let want = if other.index() <= l.index() { WARMUP_OBSERVATIONS } else { 0 };
                assert_eq!(observations(other), want, "{other:?} after {l:?}'s runs");
            }
        }
        // A coin-drawn run reaches every learner, whichever third drew it.
        observe(cell.0, cell, assigned_id(Learner::AbsorberCycle), &coin(arms), &rewarded(true, false));
        for l in Learner::ALL {
            assert_eq!(observations(l), WARMUP_OBSERVATIONS + 1, "{l:?} after a coin-drawn run");
        }
        // Each learner is credited with its own reward: the coin-drawn run
        // above rewarded A and neither B nor C.
        let d = arms.directions()[0];
        let alpha = |l: Learner| CELLS[l.index()].get(&cell).unwrap().alpha[d];
        assert!(alpha(Learner::OvertakenGhost) > alpha(Learner::AbsorberCycle));
        assert!(alpha(Learner::OvertakenGhost) > alpha(Learner::CycleBeforeRequest));
        // A probe is observed by none.
        let probe = id_where(run_cap::is_probe);
        observe(cell.0, cell, probe, &coin(arms), &rewarded(true, true));
        for l in Learner::ALL {
            assert_eq!(observations(l), WARMUP_OBSERVATIONS + 1, "{l:?} after a probe");
        }
        reset();
    }

    #[test]
    fn flat_posteriors_leave_every_run_to_the_coins_and_evidence_steers_the_learner_runs() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (1, 1);
        // Every combination four times, rewarded on one pass, so every
        // direction has the same counts at the same rate: the posteriors
        // within an axis are equal, every leader margin is exactly one
        // half, the odds against every leader are one, and every run is
        // coin-drawn.
        for pass in 0..4 {
            for i in 0..COMBINATIONS {
                observe(
                    cell.0,
                    cell,
                    unprobed_id(),
                    &coin(ArmSet::from_index(i)),
                    &rewarded(pass == 0, pass == 0),
                );
            }
        }
        for l in Learner::ALL {
            assert_eq!(share_of(l, cell), 1.0, "{l:?}: a flat cell explores below one");
        }
        for id in 0..20_000i64 {
            if learner(id).is_none() {
                continue;
            }
            assert_eq!(choose(id, 11, cell.0, cell), coin(ArmSet::coins(id)), "run {id}: steered on a flat cell");
        }

        // Reward only runs that took stock crashes and the rush: learner A's
        // posteriors then separate those directions from the coin. Learner
        // A is rewarded on every run of the stock-and-rush set and never on
        // the mixed runs; learner B on half of the mixed runs, whatever
        // their arms, and never on the stock-and-rush set.
        reset();
        let rewarded_arms = ArmSet {
            crash: CrashArm::Stock,
            request: crate::simulator::client_anchor::Arm::Rush,
            ..ArmSet::default()
        };
        for i in 0..400 {
            let arms = if i % 2 == 0 {
                rewarded_arms
            } else {
                ArmSet::from_index(i % COMBINATIONS)
            };
            observe(cell.0, cell, unprobed_id(), &coin(arms), &rewarded(i % 2 == 0, i % 4 == 1));
        }
        let mut coin_runs = [0usize; 3];
        let mut stock = [0usize; 3];
        let mut rush = [0usize; 3];
        let mut m = [0usize; 3];
        let mut n = [0usize; 3];
        for id in 0..100_000i64 {
            let Some(l) = learner(id) else { continue };
            let choice = choose(id, 11, cell.0, cell);
            n[l.index()] += 1;
            let Some(drew) = choice.learner else {
                assert_eq!(choice.arms, ArmSet::coins(id), "run {id}: coin-drawn off its coins");
                coin_runs[l.index()] += 1;
                continue;
            };
            assert_eq!(drew, l, "run {id}: drawn by another learner");
            stock[l.index()] += (choice.arms.crash == CrashArm::Stock) as usize;
            rush[l.index()] += (choice.arms.request == crate::simulator::client_anchor::Arm::Rush)
                as usize;
            m[l.index()] += 1;
        }
        // Learner A's crash axis is decided, so its share of coin-drawn
        // runs is small and its learner runs take the rewarded directions.
        let a = Learner::OvertakenGhost.index();
        let coin_share_a = coin_runs[a] as f64 / n[a] as f64;
        assert!(coin_share_a < 0.1, "learner A leaves {coin_share_a} to the coins");
        let stock_share = stock[a] as f64 / m[a] as f64;
        let rush_share = rush[a] as f64 / m[a] as f64;
        assert!(stock_share > ArmSet::coin_probability(0) + 0.1, "stock share {stock_share}");
        assert!(rush_share > 0.35, "rush share {rush_share}");
        // Learner B's reward never landed on the stock-and-rush set, whose
        // runs all carry the off direction of the retarget, fresh-first and
        // pair-order axes: the on directions lead those axes decisively, so
        // B is decided too, and its learner runs pull stock and rush under
        // their coins while the other learner's reward moved nothing.
        let b = Learner::AbsorberCycle.index();
        let coin_share_b = coin_runs[b] as f64 / n[b] as f64;
        assert!(coin_share_b < 0.1, "learner B leaves {coin_share_b} to the coins");
        let stock_share_b = stock[b] as f64 / m[b] as f64;
        let rush_share_b = rush[b] as f64 / m[b] as f64;
        assert!(stock_share_b < ArmSet::coin_probability(0), "learner B stock share {stock_share_b}");
        assert!(rush_share_b < 0.2, "learner B rush share {rush_share_b}");
        eprintln!(
            "coin shares: A {coin_share_a} B {coin_share_b}; A stock {stock_share} rush {rush_share}; B stock {stock_share_b} rush {rush_share_b}"
        );
        reset();
    }

    #[test]
    fn a_direction_without_data_draws_around_the_cells_reward_rate() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (3, 3);
        // Two hundred coin-drawn runs, all on the all-stock set, one in
        // twenty rewarded: the cell's rate is near 0.05 and no run carried
        // the phase direction.
        let arms = ArmSet::default();
        let id = unprobed_id();
        for i in 0..200 {
            observe(cell.0, cell, id, &coin(arms), &rewarded(i % 20 == 0, false));
        }
        let state = CELLS[0].get(&cell).unwrap();
        let mu = state.mu();
        assert!((mu - 0.05).abs() < 0.015, "cell rate {mu}");
        let phase = AXIS_START[0] + 2;
        assert_eq!(state.alpha[phase] + state.beta[phase], 0.0, "the phase direction has no data");
        let (a, b) = state.posterior(phase);
        // The unit flat prior keeps two pseudo-counts at one half, so the
        // direction sits a little above the cell's rate, never near one
        // half.
        let want = (1.0 + WARMUP_OBSERVATIONS as f64 * mu) / (2.0 + WARMUP_OBSERVATIONS as f64);
        assert!((state.mean(phase) - want).abs() < 1e-9);
        let mut rng = SmallRng::seed_from_u64(3);
        let dist = Beta::new(a, b).unwrap();
        let draws = 20_000;
        let sampled = (0..draws).map(|_| dist.sample(&mut rng)).sum::<f64>() / draws as f64;
        assert!(
            (sampled - want).abs() < 0.01 && sampled < 0.1,
            "a direction without data sampled a mean of {sampled} in a cell at {mu}"
        );
        // The stock direction carries the cell's evidence and sits at the
        // rate too, so nothing separates the two beyond the draws.
        assert!((state.mean(AXIS_START[0]) - mu).abs() < 0.02);
        drop(state);
        reset();
    }

    #[test]
    fn the_normal_distribution_function_matches_tabled_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.96) - 0.975_002).abs() < 1e-5);
        assert!((normal_cdf(-1.0) - 0.158_655).abs() < 1e-5);
        assert!((normal_cdf(3.0) - 0.998_650).abs() < 1e-5);
        assert!((normal_cdf(-3.0) - 0.001_350).abs() < 1e-5);
        assert!(normal_cdf(9.0) > 0.999_999);
        assert!(normal_cdf(-9.0) < 1e-6);
    }

    #[test]
    fn the_pairwise_leading_probability_matches_a_monte_carlo_of_the_betas() {
        let mut state = CellLearner::new();
        state.mass = 1000.0;
        state.reward_mass = 40.0;
        let (lo, hi) = (AXIS_START[1], AXIS_START[1] + 1);
        state.alpha[lo] = 6.0;
        state.beta[lo] = 300.0;
        state.alpha[hi] = 15.0;
        state.beta[hi] = 290.0;
        let (a_lo, b_lo) = state.posterior(lo);
        let (a_hi, b_hi) = state.posterior(hi);
        let mut rng = SmallRng::seed_from_u64(9);
        let d_lo = Beta::new(a_lo, b_lo).unwrap();
        let d_hi = Beta::new(a_hi, b_hi).unwrap();
        let draws = 200_000;
        let hi_leads = (0..draws)
            .filter(|_| d_hi.sample(&mut rng) > d_lo.sample(&mut rng))
            .count() as f64
            / draws as f64;
        let closed = state.leads_pairwise(hi, lo);
        assert!(
            (closed - hi_leads).abs() < 0.02,
            "closed form {closed} against the Monte Carlo {hi_leads}"
        );
        assert!(closed > 0.9, "a separated leader reads {closed}");
        assert!((state.leads_pairwise(lo, hi) - (1.0 - closed)).abs() < 1e-9);
        // Identical posteriors: exactly one half, whatever the width.
        state.alpha[hi] = state.alpha[lo];
        state.beta[hi] = state.beta[lo];
        assert_eq!(state.leads_pairwise(hi, lo), 0.5);
        let lead = state.leading_probability(1);
        assert_eq!(lead[0], 0.5);
        assert_eq!(lead[1], 0.5);
        assert_eq!(lead[2], 0.0, "a binary axis has no third slot");
        // A three-way axis with one posterior per direction: every product
        // is one quarter.
        let lead = state.leading_probability(0);
        for slot in 0..3 {
            assert_eq!(lead[slot], 0.25);
        }
        eprintln!("pairwise closed form {closed} against the Monte Carlo {hi_leads}");
    }

    #[test]
    fn the_exploration_share_is_one_on_a_flat_cell_and_the_odds_on_the_most_decided_axis() {
        // One posterior per direction on every axis: every margin is
        // exactly one half and the share is exactly one.
        let mut state = CellLearner::new();
        state.observations = WARMUP_OBSERVATIONS;
        state.mass = 500.0;
        state.reward_mass = 25.0;
        for d in 0..DIRECTIONS {
            state.alpha[d] = 10.0;
            state.beta[d] = 190.0;
        }
        let (share, margins) = state.explore_share();
        assert_eq!(share, 1.0);
        assert_eq!(margins, [0.5; AXES]);
        // Separate the retarget axis: the share is the odds against its
        // leader, and the flat axes, at odds of one, do not hold it up.
        let (lo, hi) = (AXIS_START[1], AXIS_START[1] + 1);
        state.alpha[lo] = 6.0;
        state.beta[lo] = 300.0;
        state.alpha[hi] = 15.0;
        state.beta[hi] = 290.0;
        let m = state.leads_pairwise(hi, lo);
        assert!(m > 0.9, "the separated axis reads {m}");
        let (share, margins) = state.explore_share();
        assert!((share - (1.0 - m) / m).abs() < 1e-12, "share {share} against odds of {m}");
        assert_eq!(margins[1], m);
        for a in [0, 2, 3, 4] {
            assert_eq!(margins[a], 0.5, "axis {a} is not flat");
        }
        // A second axis separated less than the first leaves the share
        // where the first put it; separating the first further lowers it.
        let (lo3, hi3) = (AXIS_START[3], AXIS_START[3] + 1);
        state.alpha[lo3] = 9.0;
        state.alpha[hi3] = 12.0;
        let m3 = state.leads_pairwise(hi3, lo3);
        assert!(m3 > 0.5 && m3 < m, "the second axis reads {m3} against {m}");
        let (still, _) = state.explore_share();
        assert_eq!(still, share);
        state.alpha[hi] = 30.0;
        let (lower, margins) = state.explore_share();
        assert!(margins[1] > m && lower < share, "more separation raised the share to {lower}");
        eprintln!("share {share} at margin {m}; {lower} at margin {}", margins[1]);
    }

    #[test]
    fn a_separated_leader_takes_its_axis_on_the_learner_runs_and_sets_the_coin_share() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (6, 6);
        separate_retarget(cell);
        let want = Learner::ALL.map(|l| share_of(l, cell));
        let mut runs = [0usize; 3];
        let mut coin_runs = [0usize; 3];
        let mut retarget = [0usize; 3];
        for run in 0..120_000i64 {
            let Some(l) = learner(run) else { continue };
            let choice = choose(run, 11, cell.0, cell);
            runs[l.index()] += 1;
            match choice.learner {
                None => {
                    assert_eq!(choice.arms, ArmSet::coins(run), "run {run}: coin-drawn off its coins");
                    coin_runs[l.index()] += 1;
                }
                Some(drew) => {
                    assert_eq!(drew, l, "run {run}: drawn by another learner");
                    retarget[l.index()] += choice.arms.retarget as usize;
                }
            }
        }
        for l in Learner::ALL {
            let i = l.index();
            let coin_share = coin_runs[i] as f64 / runs[i] as f64;
            let leader = retarget[i] as f64 / (runs[i] - coin_runs[i]) as f64;
            assert!(want[i] < 0.2, "{l:?}: a separated cell explores at {}", want[i]);
            assert!(
                (coin_share - want[i]).abs() < 0.02,
                "{l:?}: coin share {coin_share} against the odds {}",
                want[i]
            );
            assert!(leader > 0.9, "{l:?}: the learner runs give the leader {leader}");
            eprintln!("{l:?}: odds {} coin share {coin_share} leader share {leader}", want[i]);
        }
        reset();
    }

    #[test]
    fn the_draws_are_counted() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        util_stats::set_enabled(true);
        reset();
        let cell = (8, 8);
        let before = util_stats::snapshot().arm_selector_axis;
        let probe = id_where(run_cap::is_probe);
        let a = assigned_id(Learner::OvertakenGhost);
        // A probe draws nothing; a run below warmup is a coin-drawn draw at
        // share one, in the arm's slot and the learner's.
        choose(probe, 3, cell.0, cell);
        choose(a, 3, cell.0, cell);
        let after = util_stats::snapshot().arm_selector_axis;
        let d = |x: u64, y: u64| x - y;
        let e = &after.explore;
        let e0 = &before.explore;
        assert_eq!(d(e.draws, e0.draws), 1);
        assert_eq!(d(e.coin_runs, e0.coin_runs), 1);
        assert_eq!(d(e.warmup_coin_runs, e0.warmup_coin_runs), 1);
        assert_eq!(d(e.share_micro, e0.share_micro), 1_000_000);
        assert_eq!(d(e.margin_micro, e0.margin_micro), 0);
        assert_eq!(d(e.share_hist[9], e0.share_hist[9]), 1);
        assert_eq!(d(e.draws_by_learner[0], e0.draws_by_learner[0]), 1);
        assert_eq!(d(e.coin_runs_by_learner[0], e0.coin_runs_by_learner[0]), 1);
        assert_eq!(d(e.share_micro_by_learner[0], e0.share_micro_by_learner[0]), 1_000_000);
        assert_eq!(d(e.draws_by_arm[7], e0.draws_by_arm[7]), 1, "arm 8 lands in the last slot");
        assert_eq!(d(e.coin_runs_by_arm[7], e0.coin_runs_by_arm[7]), 1);
        assert_eq!(d(after.chosen_runs, before.chosen_runs), 0);
        // A separated cell: every draw is counted, the coin-drawn ones
        // against the share, the learner runs with their directions.
        separate_retarget(cell);
        let share = share_of(Learner::OvertakenGhost, cell);
        let ids: Vec<i64> = (0..1_000_000i64)
            .filter(|&id| learner(id) == Some(Learner::OvertakenGhost))
            .take(300)
            .collect();
        let mut coins = 0u64;
        for &id in &ids {
            coins += choose(id, 4, cell.0, cell).learner.is_none() as u64;
        }
        let after = util_stats::snapshot().arm_selector_axis;
        util_stats::set_enabled(false);
        let e = &after.explore;
        assert_eq!(d(e.draws, e0.draws), 301);
        assert_eq!(d(e.coin_runs, e0.coin_runs), 1 + coins);
        assert_eq!(d(e.warmup_coin_runs, e0.warmup_coin_runs), 1);
        let micro = d(e.share_micro, e0.share_micro) - 1_000_000;
        let want = (300.0 * share * 1_000_000.0).round() as u64;
        assert!(micro.abs_diff(want) <= 300, "share micro {micro} against {want}");
        assert_eq!(micro, d(e.share_micro_by_learner[0], e0.share_micro_by_learner[0]) - 1_000_000);
        let margin = d(e.margin_micro, e0.margin_micro);
        assert!(margin >= 300 * 900_000 && margin <= 300 * 1_000_000, "margin micro {margin}");
        assert_eq!(d(e.margin_micro_by_arm[7], e0.margin_micro_by_arm[7]), margin);
        assert_eq!(d(e.margin_micro_by_arm[0], e0.margin_micro_by_arm[0]), 0);
        let hist: u64 = (0..10).map(|b| d(e.share_hist[b], e0.share_hist[b])).sum();
        assert_eq!(hist, 301);
        let bin = ((share * 10.0) as usize).min(9);
        assert_eq!(d(e.share_hist[bin], e0.share_hist[bin]), 300);
        assert_eq!(d(e.draws_by_learner[0], e0.draws_by_learner[0]), 301);
        assert_eq!(d(e.draws_by_learner[1], e0.draws_by_learner[1]), 0);
        assert_eq!(d(e.coin_runs_by_learner[0], e0.coin_runs_by_learner[0]), 1 + coins);
        assert_eq!(d(e.draws_by_arm[7], e0.draws_by_arm[7]), 301);
        let learner_runs = 300 - coins;
        assert!(learner_runs > 250, "only {learner_runs} learner runs at share {share}");
        assert_eq!(d(after.chosen_runs, before.chosen_runs), learner_runs);
        let sum = |v: &[u64], w: &[u64]| v.iter().zip(w).map(|(x, y)| x - y).sum::<u64>();
        assert_eq!(sum(&after.chosen_by_direction, &before.chosen_by_direction), 5 * learner_runs);
        assert_eq!(d(after.axis_draws, before.axis_draws), 5 * learner_runs);
        for a in 0..AXES {
            let micro = after.leader_margin_micro[a] - before.leader_margin_micro[a];
            assert!(
                micro >= learner_runs * 400_000 && micro <= learner_runs * 1_000_000,
                "axis {a} margin {micro}"
            );
        }
        assert!(after.leader_margin_micro[1] - before.leader_margin_micro[1] > learner_runs * 900_000);
        // The other learners' sections are untouched, as is the mirror.
        assert_eq!(after.overtaken_ghost.chosen_runs, after.chosen_runs);
        assert_eq!(after.absorber_cycle.chosen_runs, before.absorber_cycle.chosen_runs);
        reset();
    }

    #[test]
    fn the_rewards_are_counted_on_the_runs_they_came_from() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        let cell = (2, 2);
        let before = util_stats::snapshot();
        let arms = ArmSet::from_index(COMBINATIONS - 1);
        let a = Learner::OvertakenGhost;
        let b = Learner::AbsorberCycle;
        let c = Learner::CycleBeforeRequest;
        let coin_run = coin(arms);
        observe(cell.0, cell, assigned_id(a), &coin_run, &rewarded(true, false));
        observe(
            cell.0,
            cell,
            assigned_id(b),
            &coin_run,
            &Rewards {
                ghost_signal: Some(true),
                mutual_absorber_cycle: true,
                cycle_before_request: true,
                first_post_fault_request_entry_step: Some(40),
                ..rewarded(false, true)
            },
        );
        // A replay child is not read for the ghost signal.
        observe(
            cell.0,
            cell,
            assigned_id(c),
            &coin_run,
            &Rewards {
                ghost_signal: None,
                exchange_before_request: true,
                first_post_fault_request_entry_step: Some(10),
                ..rewarded(false, false)
            },
        );
        observe(cell.0, cell, assigned_id(a), &learned(arms, a), &rewarded(true, true));
        observe(cell.0, cell, assigned_id(b), &learned(arms, b), &rewarded(true, true));
        observe(cell.0, cell, assigned_id(b), &learned(arms, b), &rewarded(false, false));
        // A learner run's request entry step is not read.
        observe(
            cell.0,
            cell,
            assigned_id(c),
            &learned(arms, c),
            &Rewards {
                mutual_absorber_cycle: true,
                cycle_before_request: true,
                first_post_fault_request_entry_step: Some(99),
                ..rewarded(true, true)
            },
        );
        observe(cell.0, cell, assigned_id(c), &learned(arms, c), &rewarded(true, true));
        let snapshot = util_stats::snapshot();
        util_stats::set_enabled(false);
        let after = snapshot.arm_selector_axis;
        let entry = &snapshot.client_anchor.first_post_fault_entry;
        let entry0 = &before.client_anchor.first_post_fault_entry;
        let before = before.arm_selector_axis;
        for d in arms.directions() {
            assert_eq!(entry.runs[d] - entry0.runs[d], 2);
            assert_eq!(entry.steps_sum[d] - entry0.steps_sum[d], 50);
        }
        let cr = &after.cycle_before_request;
        let cr0 = &before.cycle_before_request;
        assert_eq!(cr.reward_runs_control - cr0.reward_runs_control, 3);
        assert_eq!(cr.reward_positive_control - cr0.reward_positive_control, 1);
        assert_eq!(cr.reward_runs_treated - cr0.reward_runs_treated, 2);
        assert_eq!(cr.reward_positive_treated - cr0.reward_positive_treated, 1);
        assert_eq!(cr.cells, 1);
        let mc = &after.mutual_absorber_cycle;
        let mc0 = &before.mutual_absorber_cycle;
        assert_eq!(mc.reward_runs_control - mc0.reward_runs_control, 3);
        assert_eq!(mc.reward_positive_control - mc0.reward_positive_control, 1);
        let er = &after.exchange_before_request;
        let er0 = &before.exchange_before_request;
        assert_eq!(er.reward_runs_control - er0.reward_runs_control, 3);
        assert_eq!(er.reward_positive_control - er0.reward_positive_control, 1);
        assert_eq!(after.observations - before.observations, 8);
        // Learner A at the top level and under its reward name.
        assert_eq!(after.reward_runs_control - before.reward_runs_control, 3);
        assert_eq!(after.reward_positive_control - before.reward_positive_control, 1);
        assert_eq!(after.reward_runs_treated - before.reward_runs_treated, 1);
        assert_eq!(after.reward_positive_treated - before.reward_positive_treated, 1);
        let og = &after.overtaken_ghost;
        let og0 = &before.overtaken_ghost;
        assert_eq!(og.reward_runs_control - og0.reward_runs_control, 3);
        assert_eq!(og.reward_positive_control - og0.reward_positive_control, 1);
        assert_eq!(og.reward_runs_treated - og0.reward_runs_treated, 1);
        assert_eq!(og.reward_positive_treated - og0.reward_positive_treated, 1);
        let ac = &after.absorber_cycle;
        let ac0 = &before.absorber_cycle;
        assert_eq!(ac.reward_runs_control - ac0.reward_runs_control, 3);
        assert_eq!(ac.reward_positive_control - ac0.reward_positive_control, 1);
        assert_eq!(ac.reward_runs_treated - ac0.reward_runs_treated, 2);
        assert_eq!(ac.reward_positive_treated - ac0.reward_positive_treated, 1);
        let gs = &after.ghost_signal;
        let gs0 = &before.ghost_signal;
        assert_eq!(gs.reward_runs_control - gs0.reward_runs_control, 2);
        assert_eq!(gs.reward_positive_control - gs0.reward_positive_control, 1);
        let es = &after.either_shape;
        let es0 = &before.either_shape;
        assert_eq!(es.reward_runs_control - es0.reward_runs_control, 3);
        assert_eq!(es.reward_positive_control - es0.reward_positive_control, 2);
        assert_eq!(after.cells, 1);
        assert_eq!(after.absorber_cycle.cells, 1);
        assert_eq!(after.reward_runs_by_arm[3] - before.reward_runs_by_arm[3], 8);
        assert_eq!(after.reward_positive_by_arm[3] - before.reward_positive_by_arm[3], 5);
        assert_eq!(
            after.control_runs_by_combination[COMBINATIONS - 1]
                - before.control_runs_by_combination[COMBINATIONS - 1],
            3
        );
        assert_eq!(
            ac.control_reward_positive_by_combination[COMBINATIONS - 1]
                - ac0.control_reward_positive_by_combination[COMBINATIONS - 1],
            1
        );
        for d in arms.directions() {
            assert_eq!(
                after.control_runs_by_direction[d] - before.control_runs_by_direction[d],
                3
            );
            assert_eq!(
                gs.control_runs_by_direction[d] - gs0.control_runs_by_direction[d],
                2
            );
            // The per-arm table: the cell's arm is 2, row 3 of eight,
            // twelve directions per row; every other row is untouched.
            let row = 3 * DIRECTIONS + d;
            let other = d;
            assert_eq!(
                after.control_runs_by_arm_direction[row] - before.control_runs_by_arm_direction[row],
                3
            );
            assert_eq!(
                after.control_reward_positive_by_arm_direction[row]
                    - before.control_reward_positive_by_arm_direction[row],
                1
            );
            assert_eq!(
                cr.control_runs_by_arm_direction[row] - cr0.control_runs_by_arm_direction[row],
                3
            );
            assert_eq!(
                cr.control_reward_positive_by_arm_direction[row]
                    - cr0.control_reward_positive_by_arm_direction[row],
                1
            );
            assert_eq!(
                gs.control_runs_by_arm_direction[row] - gs0.control_runs_by_arm_direction[row],
                2
            );
            assert_eq!(
                after.control_runs_by_arm_direction[other] - before.control_runs_by_arm_direction[other],
                0
            );
        }
        assert_eq!(after.control_runs_by_arm_direction.len(), 8 * DIRECTIONS);
        reset();
        assert_eq!(util_stats::snapshot().arm_selector_axis.cells, 0);
        assert_eq!(util_stats::snapshot().arm_selector_axis.absorber_cycle.cells, 0);
        assert_eq!(util_stats::snapshot().arm_selector_axis.cycle_before_request.cells, 0);
    }
}
