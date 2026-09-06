//! The phase a run occupies in the session-global mechanisms that treat
//! only part of a session's runs.
//!
//! A phase read straight off the run id shares that id's factors with the
//! configuration grid's width, because a grid is walked in order and a run's
//! configuration is a deterministic function of its id. At a 32-run probe
//! period against a 54-configuration grid the common factor is two, and the
//! probe stream reaches only the 27 even configurations - so a learner fed
//! by probes never sees half the grid, and applies what it learned there to
//! the half it never saw. Mixing the id first makes the phase independent
//! of every grid width, and costs three multiplies per run.
//!
//! Phases of nested periods keep their relationships: both are the same
//! mixed value reduced, so a run at phase 0 of 64 is at phase 0 of 32.
//! Every phase remains a pure function of the run id, so a session is as
//! reproducible as it was.

/// The run's phase in `[0, period)`. `period` must be positive.
pub fn phase(run_id: i64, period: i64) -> i64 {
    debug_assert!(period > 0, "a phase period must be positive");
    (mix(run_id as u64) % period as u64) as i64
}

/// The run's phase under a domain salt. Two mechanisms that split a session
/// at the same period must not select the same runs, so each passes its own
/// salt and gets a split independent of every other one.
pub fn salted_phase(run_id: i64, salt: u64, period: i64) -> i64 {
    debug_assert!(period > 0, "a phase period must be positive");
    (mix(run_id as u64 ^ salt) % period as u64) as i64
}

/// A uniform value in `[0, 1)` under a domain salt, a pure function of the
/// run id: the top 53 bits of the mixed id over 2^53, so every double in
/// the range is reachable and no two salts agree beyond chance.
pub fn salted_unit(run_id: i64, salt: u64) -> f64 {
    (mix(run_id as u64 ^ salt) >> 11) as f64 / (1u64 << 53) as f64
}

/// SplitMix64's finalizer over the id. Chosen for avalanche: every input
/// bit reaches every output bit, which is what breaks the correlation with
/// the grid width.
fn mix(run_id: u64) -> u64 {
    let mut z = run_id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phase_is_in_range_and_is_a_function_of_the_id() {
        for id in -1000..1000i64 {
            let p = phase(id, 32);
            assert!((0..32).contains(&p), "id {id} gave phase {p}");
            assert_eq!(p, phase(id, 32), "a phase must not vary between reads");
        }
    }

    #[test]
    fn nested_periods_keep_their_relationship() {
        // The mechanisms rely on this: a run that feeds a learner at phase 0
        // of the posture period must also be a probe at phase 0 of the probe
        // period, and a run placed at phase 32 of 64 must be a probe when it
        // is at phase 0 of 32.
        for id in 0..20_000i64 {
            if phase(id, 64) == 0 {
                assert_eq!(phase(id, 32), 0, "id {id} feeds the learner but is no probe");
            }
            assert_eq!(phase(id, 64) % 32, phase(id, 32), "id {id} breaks nesting");
        }
    }

    #[test]
    fn each_phase_gets_close_to_its_share() {
        let mut counts = [0u32; 32];
        for id in 0..32_000i64 {
            counts[phase(id, 32) as usize] += 1;
        }
        for (p, &n) in counts.iter().enumerate() {
            assert!((900..1100).contains(&n), "phase {p} drew {n} of an expected 1000");
        }
    }

    #[test]
    fn a_salt_gives_an_even_split_independent_of_the_unsalted_one() {
        const A: u64 = 0x_1122_3344_5566_7788;
        const B: u64 = 0x_99AA_BBCC_DDEE_FF00;
        let n = 20_000i64;
        let mut ones = 0i64;
        let mut agree_other_salt = 0i64;
        let mut agree_unsalted = 0i64;
        for id in 0..n {
            let a = salted_phase(id, A, 2);
            assert!((0..2).contains(&a), "id {id} gave phase {a}");
            assert_eq!(a, salted_phase(id, A, 2), "a salted phase must not vary");
            ones += a;
            if a == salted_phase(id, B, 2) {
                agree_other_salt += 1;
            }
            if a == phase(id, 2) {
                agree_unsalted += 1;
            }
        }
        let share = ones as f64 / n as f64;
        assert!((share - 0.5).abs() < 0.02, "one salt takes {share} of the runs");
        for (what, agree) in [("another salt", agree_other_salt), ("no salt", agree_unsalted)] {
            let overlap = agree as f64 / n as f64;
            assert!((overlap - 0.5).abs() < 0.02, "the split agrees with {what} on {overlap}");
        }
    }

    #[test]
    fn a_salted_unit_is_uniform_and_independent_of_a_phase_under_another_salt() {
        const A: u64 = 0x_1122_3344_5566_7788;
        const B: u64 = 0x_99AA_BBCC_DDEE_FF00;
        let n = 20_000i64;
        let mut sum = 0.0;
        let mut below_half_on_phase_one = 0i64;
        let mut phase_one = 0i64;
        for id in -n / 2..n / 2 {
            let u = salted_unit(id, A);
            assert!((0.0..1.0).contains(&u), "id {id} gave {u}");
            assert_eq!(u, salted_unit(id, A), "a salted unit must not vary");
            sum += u;
            if salted_phase(id, B, 2) == 1 {
                phase_one += 1;
                below_half_on_phase_one += (u < 0.5) as i64;
            }
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "the units average {mean}");
        let share = below_half_on_phase_one as f64 / phase_one as f64;
        assert!((share - 0.5).abs() < 0.02, "the unit follows another salt's phase on {share}");
    }

    #[test]
    fn a_phase_does_not_align_with_a_grid_width() {
        // The defect this module exists for: with the phase read straight
        // off the id, `id % 32 == 0` selects only even ids, and against a
        // 54-wide grid walked in order that is only ever an even
        // configuration. Every width below is checked, so a grid resized
        // later cannot quietly reintroduce it.
        for width in 2..=64usize {
            let mut seen = vec![false; width];
            for id in 0..200_000i64 {
                if phase(id, 32) == 0 {
                    seen[(id as usize) % width] = true;
                }
            }
            let missed = seen.iter().filter(|s| !**s).count();
            assert_eq!(missed, 0, "probes reached no run at {missed} of {width} grid positions");
        }
    }
}
