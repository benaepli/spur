//! Per-cell selection of the mechanism arm set, learned from a per-run
//! reward read off the finished run.
//!
//! Every run carries one direction on each of five mechanism axes (the
//! `ArmSet`). The runs that are not probes of either kind are split by id
//! under a salt of their own into four quarters. The coin quarter takes the
//! directions the run id's coins name. Each of the other three quarters is
//! steered by one learner: one discounted Beta posterior per direction is
//! kept for every cell, a cell being one campaign arm at one configuration,
//! and the treated run picks each axis independently, a direction with
//! probability proportional to the coin's share of that direction times
//! its posterior mean, so with equal posteriors the expected share of every
//! direction is the coin's, and a learner departs from the coins only on
//! evidence. Nothing is drawn from the posteriors: the coin quarter, which
//! every learner observes, supplies the exploration. Each direction's prior
//! is shrunk toward the cell's own discounted reward rate with the warmup
//! count as its weight, so a direction the cell has rarely carried sits
//! near the cell's rate, not near one half. A cell below its warmup count
//! hands the treated run its coin arm set.
//!
//! Each learner has a reward of its own, a per-run bool the scheduler sets
//! when the run reached the shape the learner is after. A learner observes
//! the coin quarter and its own quarter, credited to the five directions
//! the run carried, and never another learner's quarter, so the three
//! learners are read against one common control. Further rewards are read
//! on the coin quarter for calibration and steer nothing.
//!
//! The selector's own draws come from a generator seeded from the run's
//! schedule seed under a salt of its own, so the run's schedule stream is
//! untouched and a coin-quarter run draws exactly what it would without the
//! selector. A treated run's arm set depends on the learner's state at draw
//! time, so it is not a function of the run id alone; the run's tag records
//! the arms it ran under and which learner drew them.

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

/// Salt for the four quarters. Distinct from every other split of a session.
pub const SELECTOR_SALT: u64 = 0x_4152_4D53_454C_4354; // "ARMSELCT"

/// Salt for the selector's own generator, derived per run from the schedule
/// seed so the draw is reproducible given the learner's state.
const DRAW_SALT: u64 = 0x_4152_4D44_5241_5753; // "ARMDRAWS"

/// Per-observation discount on the posteriors of the directions a run
/// carried, so a cell follows its recent reward rate rather than its whole
/// history. Every reward is rare, so the window is about five hundred
/// observations per cell.
pub const DISCOUNT: f64 = 0.998;

/// Observations a cell needs before a treated run draws from its
/// posteriors; below it the treated run takes its coin arm set.
pub const WARMUP_OBSERVATIONS: u64 = 24;

/// A cell: the campaign arm index and the configuration index the run is
/// attributed to.
pub type Cell = (i32, i32);

/// The three learners, each steering one quarter of the runs.
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

    fn index(self) -> usize {
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

/// Which quarter a run falls in: the coin quarter, or the quarter one
/// learner steers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quarter {
    Coin,
    Learned(Learner),
}

/// The run's quarter, or None for a probe of either kind: run-cap probes
/// feed the length learners and timer-context probes must run unsteered,
/// so neither is steered nor observed.
pub fn quarter(run_id: i64) -> Option<Quarter> {
    if run_cap::is_probe(run_id) || timer_context::run_mode(run_id) == timer_context::RunMode::Probe
    {
        return None;
    }
    Some(match run_phase::salted_phase(run_id, SELECTOR_SALT, 4) {
        0 => Quarter::Coin,
        1 => Quarter::Learned(Learner::OvertakenGhost),
        2 => Quarter::Learned(Learner::AbsorberCycle),
        _ => Quarter::Learned(Learner::CycleBeforeRequest),
    })
}

/// The learner that steers this run, if any.
pub fn learner(run_id: i64) -> Option<Learner> {
    match quarter(run_id) {
        Some(Quarter::Learned(l)) => Some(l),
        _ => None,
    }
}

/// Whether this run takes a learned arm set.
pub fn is_treated(run_id: i64) -> bool {
    learner(run_id).is_some()
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

    /// Pick one direction of axis `a` with probability proportional to the
    /// coin share times the posterior mean. No value is drawn from the
    /// posterior: the coin quarter supplies the exploration, so a treated run
    /// exploits what the cell has learned, and the generator serves the
    /// categorical pick only.
    fn sample_axis(&self, a: usize, rng: &mut SmallRng) -> usize {
        let range = AXIS_START[a]..AXIS_START[a + 1];
        let mut weights = [0.0f64; 3];
        let mut total = 0.0;
        for (slot, d) in range.clone().enumerate() {
            weights[slot] = ArmSet::coin_probability(d) * self.mean(d);
            total += weights[slot];
        }
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

/// The arm set a run takes. A coin-quarter run or a probe takes its coins; a
/// treated run draws from its learner's cell once the cell is past warmup.
/// The draw reads nothing from the run's schedule stream.
pub fn choose(run_id: i64, schedule_seed: u64, cell: Cell) -> ArmSet {
    let coins = ArmSet::coins(run_id);
    let Some(learner) = learner(run_id) else {
        return coins;
    };
    let drawn = CELLS[learner.index()].get(&cell).and_then(|state| {
        if state.observations < WARMUP_OBSERVATIONS {
            return None;
        }
        let mut rng = SmallRng::seed_from_u64(derive_seed(schedule_seed, run_id, DRAW_SALT));
        let mut directions = [0usize; AXES];
        let mut agreements = 0u64;
        for (a, slot) in directions.iter_mut().enumerate() {
            *slot = state.sample_axis(a, &mut rng);
            agreements += (*slot == state.leader(a)) as u64;
        }
        Some((ArmSet::from_directions(directions), agreements))
    });
    let reward = learner.reward();
    match drawn {
        Some((arms, agreements)) => {
            util_stats::record_arm_selector_treated_run(reward, Some(&arms), &coins, agreements);
            arms
        }
        None => {
            util_stats::record_arm_selector_treated_run(reward, None, &coins, 0);
            coins
        }
    }
}

fn credit(learner: Learner, cell: Cell, arms: &ArmSet, reward: bool) {
    CELLS[learner.index()]
        .entry(cell)
        .or_insert_with(|| {
            util_stats::record_arm_selector_cell_created(learner.reward());
            CellLearner::new()
        })
        .credit(arms, reward);
}

/// Fold one finished run into the learners that may see it: a coin-quarter
/// run into every learner, a treated run into its own learner only, a
/// probe into none. The coin quarter is also where every reward's rate per
/// direction, and the step of the first request-caused entry, are read.
pub fn observe(cell: Cell, run_id: i64, arms: &ArmSet, rewards: &Rewards) {
    let Some(quarter) = quarter(run_id) else {
        return;
    };
    let value = |reward: Reward| match reward {
        Reward::OvertakenGhost => Some(rewards.overtaken_ghost),
        Reward::AbsorberCycle => Some(rewards.absorber_cycle),
        Reward::MutualAbsorberCycle => Some(rewards.mutual_absorber_cycle),
        Reward::GhostSignal => rewards.ghost_signal,
        Reward::EitherShape => Some(rewards.overtaken_ghost || rewards.absorber_cycle),
        Reward::CycleBeforeRequest => Some(rewards.cycle_before_request),
        Reward::ExchangeBeforeRequest => Some(rewards.exchange_before_request),
    };
    util_stats::record_arm_selector_run_observed(cell.0, rewards.overtaken_ghost);
    match quarter {
        Quarter::Coin => {
            for learner in Learner::ALL {
                credit(learner, cell, arms, rewards_of(learner, rewards));
            }
            for reward in Reward::ALL {
                if let Some(r) = value(reward) {
                    util_stats::record_arm_selector_observation(reward, false, arms, r);
                }
            }
            if let Some(step) = rewards.first_post_fault_request_entry_step {
                util_stats::record_client_anchor_first_post_fault_entry(arms, step);
            }
        }
        Quarter::Learned(learner) => {
            let r = rewards_of(learner, rewards);
            credit(learner, cell, arms, r);
            util_stats::record_arm_selector_observation(learner.reward(), true, arms, r);
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
        (0..1_000_000i64).find(|&id| pred(id)).expect("every quarter is reachable")
    }

    fn coin_id() -> i64 {
        id_where(|id| quarter(id) == Some(Quarter::Coin))
    }

    fn learned_id(l: Learner) -> i64 {
        id_where(|id| learner(id) == Some(l))
    }

    fn rewarded(overtaken_ghost: bool, absorber_cycle: bool) -> Rewards {
        Rewards {
            overtaken_ghost,
            absorber_cycle,
            ghost_signal: Some(false),
            ..Rewards::default()
        }
    }

    /// Feed `n` coin-quarter observations of `arms` with every learner's
    /// reward set to `reward`.
    fn feed(cell: Cell, n: usize, arms: &ArmSet, reward: bool) {
        let id = coin_id();
        for _ in 0..n {
            observe(
                cell,
                id,
                arms,
                &Rewards {
                    cycle_before_request: reward,
                    ..rewarded(reward, reward)
                },
            );
        }
    }

    #[test]
    fn the_quarters_partition_the_unprobed_ids_and_spare_every_probe() {
        let n = 64_000i64;
        let mut counts = [0i64; 4];
        for id in 0..n {
            let probe = run_cap::is_probe(id)
                || timer_context::run_mode(id) == timer_context::RunMode::Probe;
            match quarter(id) {
                None => assert!(probe, "run {id}: an unprobed run has no quarter"),
                Some(q) => {
                    assert!(!probe, "run {id}: a probe has a quarter");
                    counts[match q {
                        Quarter::Coin => 0,
                        Quarter::Learned(l) => 1 + l.index(),
                    }] += 1;
                }
            }
            assert_eq!(is_treated(id), learner(id).is_some());
            for l in Learner::ALL {
                assert_eq!(learner(id) == Some(l), quarter(id) == Some(Quarter::Learned(l)));
            }
        }
        let unprobed: i64 = counts.iter().sum();
        assert!(unprobed > 50_000, "unprobed {unprobed} of {n}");
        for (i, c) in counts.iter().enumerate() {
            let share = *c as f64 / unprobed as f64;
            assert!((share - 0.25).abs() < 0.02, "quarter {i} holds {share}");
        }
        let treated = counts[1] + counts[2] + counts[3];
        assert!(treated > 30_000 && treated < 56_000, "treated {treated} of {n}");
    }

    #[test]
    fn no_quarter_touches_the_schedule_stream_and_the_draw_is_reproducible() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (7, 3);
        let coin = coin_id();
        let a = learned_id(Learner::OvertakenGhost);
        let b = learned_id(Learner::AbsorberCycle);
        let c = learned_id(Learner::CycleBeforeRequest);
        let schedule = CountingRng {
            inner: SmallRng::seed_from_u64(1),
            draws: 0,
        };
        assert_eq!(choose(coin, 5, cell), ArmSet::coins(coin));
        assert_eq!(choose(a, 5, cell), ArmSet::coins(a), "below warmup: coins");
        assert_eq!(choose(b, 5, cell), ArmSet::coins(b), "below warmup: coins");
        assert_eq!(choose(c, 5, cell), ArmSet::coins(c), "below warmup: coins");
        feed(cell, WARMUP_OBSERVATIONS as usize, &ArmSet::default(), false);
        let first = choose(a, 5, cell);
        assert_eq!(choose(a, 5, cell), first, "the same seed and state draw alike");
        let first_b = choose(b, 5, cell);
        assert_eq!(choose(b, 5, cell), first_b, "the same seed and state draw alike");
        let first_c = choose(c, 5, cell);
        assert_eq!(choose(c, 5, cell), first_c, "the same seed and state draw alike");
        assert_eq!(choose(coin, 5, cell), ArmSet::coins(coin));
        assert_eq!(schedule.draws, 0, "the selector must not read the run's stream");
        reset();
    }

    #[test]
    fn each_learner_sees_the_coin_quarter_and_its_own_quarter_only() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (4, 4);
        let arms = ArmSet::default();
        let observations =
            |l: Learner| CELLS[l.index()].get(&cell).map_or(0, |state| state.observations);
        // A learner's treated runs reach its own cell and no other's.
        for l in Learner::ALL {
            let id = learned_id(l);
            for _ in 0..WARMUP_OBSERVATIONS {
                observe(cell, id, &arms, &rewarded(true, true));
            }
            for other in Learner::ALL {
                let want = if other.index() <= l.index() { WARMUP_OBSERVATIONS } else { 0 };
                assert_eq!(observations(other), want, "{other:?} after {l:?}'s quarter");
            }
        }
        // A coin-quarter run reaches every learner.
        observe(cell, coin_id(), &arms, &rewarded(true, false));
        for l in Learner::ALL {
            assert_eq!(observations(l), WARMUP_OBSERVATIONS + 1, "{l:?} after a coin run");
        }
        // Each learner is credited with its own reward: the coin run above
        // rewarded A and neither B nor C.
        let d = arms.directions()[0];
        let alpha = |l: Learner| CELLS[l.index()].get(&cell).unwrap().alpha[d];
        assert!(alpha(Learner::OvertakenGhost) > alpha(Learner::AbsorberCycle));
        assert!(alpha(Learner::OvertakenGhost) > alpha(Learner::CycleBeforeRequest));
        // A probe is observed by none.
        let probe = id_where(run_cap::is_probe);
        observe(cell, probe, &arms, &rewarded(true, true));
        for l in Learner::ALL {
            assert_eq!(observations(l), WARMUP_OBSERVATIONS + 1, "{l:?} after a probe");
        }
        reset();
    }

    #[test]
    fn flat_posteriors_draw_the_coin_shares_and_evidence_moves_them() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (1, 1);
        // Every combination four times, rewarded on one pass, so every
        // direction has the same counts at the same rate: the posteriors
        // within an axis are equal and concentrated, and the draw can only
        // follow the coins.
        for pass in 0..4 {
            for i in 0..COMBINATIONS {
                observe(cell, coin_id(), &ArmSet::from_index(i), &rewarded(pass == 0, pass == 0));
            }
        }
        let mut placed = 0usize;
        let mut stock = 0usize;
        let mut hold = 0usize;
        let mut n = 0usize;
        for id in 0..200_000i64 {
            if !is_treated(id) {
                continue;
            }
            let arms = choose(id, 11, cell);
            placed += arms.placed() as usize;
            stock += (arms.crash == CrashArm::Stock) as usize;
            hold += (arms.request == crate::simulator::client_anchor::Arm::Hold) as usize;
            n += 1;
        }
        // The pick is proportional to the coin times the posterior mean,
        // so with equal posteriors the expected share is exactly the
        // coin's, up to the noise of the categorical pick.
        let want_placed = ArmSet::coin_probability(1) + ArmSet::coin_probability(2);
        let got_placed = placed as f64 / n as f64;
        let got_stock = stock as f64 / n as f64;
        assert!(
            (got_placed - want_placed).abs() < 0.02,
            "equal posteriors placed {got_placed} against the coin {want_placed}"
        );
        assert!(
            (got_stock - ArmSet::coin_probability(0)).abs() < 0.02,
            "equal posteriors stock {got_stock}"
        );
        let got_hold = hold as f64 / n as f64;
        assert!((got_hold - 0.5).abs() < 0.03, "equal posteriors hold {got_hold}");
        eprintln!("equal posteriors: placed {got_placed} (coin {want_placed}), hold {got_hold}");

        // Reward only runs that took stock crashes and the rush: the cell's
        // posteriors then favour those directions over the coin. Learner A
        // is rewarded on its own reward only, so learner B's quarter stays
        // at the coins.
        reset();
        let rewarded_arms = ArmSet {
            crash: CrashArm::Stock,
            request: crate::simulator::client_anchor::Arm::Rush,
            ..ArmSet::default()
        };
        // Learner A is rewarded on every run of the stock-and-rush set and
        // never on the mixed runs; learner B on half of the mixed runs,
        // whatever their arms, and never on the stock-and-rush set.
        for i in 0..400 {
            let arms = if i % 2 == 0 {
                rewarded_arms
            } else {
                ArmSet::from_index(i % COMBINATIONS)
            };
            observe(cell, coin_id(), &arms, &rewarded(i % 2 == 0, i % 4 == 1));
        }
        let mut stock = [0usize; 3];
        let mut rush = [0usize; 3];
        let mut m = [0usize; 3];
        for id in 0..100_000i64 {
            let Some(l) = learner(id) else { continue };
            let arms = choose(id, 11, cell);
            stock[l.index()] += (arms.crash == CrashArm::Stock) as usize;
            rush[l.index()] += (arms.request == crate::simulator::client_anchor::Arm::Rush) as usize;
            m[l.index()] += 1;
        }
        let stock_share = stock[0] as f64 / m[0] as f64;
        let rush_share = rush[0] as f64 / m[0] as f64;
        assert!(stock_share > ArmSet::coin_probability(0) + 0.1, "stock share {stock_share}");
        assert!(rush_share > 0.35, "rush share {rush_share}");
        // Learner B's reward never landed on the stock-and-rush set, so its
        // stock and rush posteriors sit below its others: the other
        // learner's reward moved nothing here, and B's own pulled the two
        // directions under their coins.
        let stock_share_b = stock[1] as f64 / m[1] as f64;
        let rush_share_b = rush[1] as f64 / m[1] as f64;
        assert!(stock_share_b < ArmSet::coin_probability(0), "learner B stock share {stock_share_b}");
        assert!(rush_share_b < 0.2, "learner B rush share {rush_share_b}");
        reset();
    }

    #[test]
    fn a_direction_without_data_draws_around_the_cells_reward_rate() {
        let _serial = config_override::exclusive_session();
        fault_timing::reset();
        reset();
        let cell = (3, 3);
        // Two hundred coin runs, all on the all-stock set, one in twenty
        // rewarded: the cell's rate is near 0.05 and no run carried the
        // phase direction.
        let arms = ArmSet::default();
        let id = coin_id();
        for i in 0..200 {
            observe(cell, id, &arms, &rewarded(i % 20 == 0, false));
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
    fn the_rewards_are_counted_on_the_quarter_they_came_from() {
        let _serial = config_override::exclusive_session();
        util_stats::set_enabled(true);
        reset();
        let cell = (2, 2);
        let before = util_stats::snapshot();
        let arms = ArmSet::from_index(COMBINATIONS - 1);
        let coin = coin_id();
        let a = learned_id(Learner::OvertakenGhost);
        let b = learned_id(Learner::AbsorberCycle);
        let c = learned_id(Learner::CycleBeforeRequest);
        observe(cell, coin, &arms, &rewarded(true, false));
        observe(
            cell,
            coin,
            &arms,
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
            cell,
            coin,
            &arms,
            &Rewards {
                ghost_signal: None,
                exchange_before_request: true,
                first_post_fault_request_entry_step: Some(10),
                ..rewarded(false, false)
            },
        );
        observe(cell, a, &arms, &rewarded(true, true));
        observe(cell, b, &arms, &rewarded(true, true));
        observe(cell, b, &arms, &rewarded(false, false));
        // A treated run's request entry step is not read.
        observe(
            cell,
            c,
            &arms,
            &Rewards {
                mutual_absorber_cycle: true,
                cycle_before_request: true,
                first_post_fault_request_entry_step: Some(99),
                ..rewarded(true, true)
            },
        );
        observe(cell, c, &arms, &rewarded(true, true));
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
        }
        reset();
        assert_eq!(util_stats::snapshot().arm_selector_axis.cells, 0);
        assert_eq!(util_stats::snapshot().arm_selector_axis.absorber_cycle.cells, 0);
        assert_eq!(util_stats::snapshot().arm_selector_axis.cycle_before_request.cells, 0);
    }
}
