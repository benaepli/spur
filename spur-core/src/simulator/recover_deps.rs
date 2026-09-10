//! Whether a generated plan lets a node restart wait on client work.
//!
//! The plan generator's probabilistic pass adds an edge between any two
//! events with only a cycle guard, so a restart can come to depend on a
//! client request. A request addressed to the node that is down cannot
//! finish before the node comes back, and the restart cannot be released
//! before the request finishes: the run stalls with the node down until
//! the step budget ends it. Each run takes one of two cells: stock keeps
//! the generator as it is, and exempt never lets a restart wait on a
//! probabilistic edge.
//!
//! The cell is a pure function of the workload seed, which is the value a
//! plan is generated from. A child run that replays its parent's plan
//! inherits the parent's workload seed and so lands in the parent's cell;
//! no run ever executes a plan generated under another cell. It is not a
//! function of the run id, so a run's tag has to be read from the run
//! record and never recomputed from the id.

use crate::simulator::run_phase;

/// Domain salt for the cell draw, so it is independent of every other
/// split drawn over the same value.
pub const DEPS_SALT: u64 = 0x_5245_434F_5644_4550; // "RECOVDEP"

/// The cell a run's plan is generated under.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecoverDeps {
    /// The generator's pass runs unchanged.
    #[default]
    Stock,
    /// A restart takes no probabilistic edge from any event; only the
    /// mandatory edges order it.
    Exempt,
}

impl RecoverDeps {
    /// The cell the workload seed names: half the seeds exempt, half stock.
    pub fn of_workload_seed(workload_seed: u64) -> Self {
        match run_phase::salted_phase(workload_seed as i64, DEPS_SALT, 2) {
            1 => RecoverDeps::Exempt,
            _ => RecoverDeps::Stock,
        }
    }

    /// The index this cell occupies in per-cell tallies.
    pub fn index(self) -> usize {
        match self {
            RecoverDeps::Stock => 0,
            RecoverDeps::Exempt => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::rng::{WORKLOAD_SALT, derive_seed};

    #[test]
    fn the_cell_is_a_function_of_the_workload_seed_alone() {
        for seed in 0..10_000u64 {
            let a = RecoverDeps::of_workload_seed(seed);
            assert_eq!(a, RecoverDeps::of_workload_seed(seed), "seed {seed} varies");
        }
        assert_eq!(RecoverDeps::Stock.index(), 0);
        assert_eq!(RecoverDeps::Exempt.index(), 1);
    }

    #[test]
    fn each_cell_gets_close_to_half_over_derived_seeds() {
        let n = 40_000i64;
        let mut exempt = 0u32;
        for run_id in 0..n {
            let seed = derive_seed(7, run_id, WORKLOAD_SALT);
            exempt += (RecoverDeps::of_workload_seed(seed) == RecoverDeps::Exempt) as u32;
        }
        let share = exempt as f64 / n as f64;
        assert!((share - 0.5).abs() < 0.02, "exempt takes {share}");
    }

    #[test]
    fn the_cell_follows_the_seed_and_not_the_id_that_issued_the_run() {
        // A replay child is issued under a fresh run id and its parent's
        // workload seed. The seed's cell must agree with the id's own phase
        // under the same salt only at chance, or the cell would be readable
        // off the id after all.
        let n = 20_000i64;
        let mut agree = 0i64;
        for run_id in 0..n {
            let seed = derive_seed(11, run_id, WORKLOAD_SALT);
            let by_seed = RecoverDeps::of_workload_seed(seed);
            let by_id = match run_phase::salted_phase(run_id, DEPS_SALT, 2) {
                1 => RecoverDeps::Exempt,
                _ => RecoverDeps::Stock,
            };
            agree += (by_seed == by_id) as i64;
        }
        let share = agree as f64 / n as f64;
        assert!((share - 0.5).abs() < 0.02, "the seed's cell follows the id on {share}");
    }
}
