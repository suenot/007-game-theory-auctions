//! Colonel Blotto game for capital allocation across battlefields.
//!
//! Two players each have a fixed budget. Each must allocate their entire
//! budget across `K` battlefields (e.g., assets in a portfolio competition).
//! On each battlefield the higher allocation wins one point. The player
//! with more points wins the overall game. The classical analysis (Borel,
//! Gross & Wagner) gives a symmetric mixed-strategy equilibrium when both
//! budgets are equal: each player's allocation on each battlefield is
//! drawn from a uniform distribution on `[0, 2 * budget / K]`.
//!
//! For asymmetric budgets the equilibrium is more involved; we expose the
//! symmetric closed form and a generic Monte-Carlo evaluator.

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{ensure_finite_slice, GameError, Result};

#[derive(Debug, Clone)]
pub struct ColonelBlotto {
    pub budget_a: f64,
    pub budget_b: f64,
    pub battlefields: usize,
}

impl ColonelBlotto {
    pub fn new(budget_a: f64, budget_b: f64, battlefields: usize) -> Result<Self> {
        if !budget_a.is_finite() || !budget_b.is_finite() || budget_a <= 0.0 || budget_b <= 0.0 {
            return Err(GameError::InvalidParameter(
                "budgets must be finite and positive",
            ));
        }
        if battlefields < 2 {
            return Err(GameError::InvalidParameter("need at least 2 battlefields"));
        }
        Ok(Self {
            budget_a,
            budget_b,
            battlefields,
        })
    }

    /// Symmetric mean allocation per battlefield assuming equal budgets.
    /// Returns `Err` if budgets are not equal.
    pub fn symmetric_uniform_upper(&self) -> Result<f64> {
        if (self.budget_a - self.budget_b).abs() > 1e-12 {
            return Err(GameError::InvalidParameter(
                "symmetric equilibrium requires equal budgets",
            ));
        }
        Ok(2.0 * self.budget_a / self.battlefields as f64)
    }

    /// Sample a single allocation from the symmetric mixed-strategy
    /// equilibrium. The classical result: draw `K` independent uniforms on
    /// `[0, 2B/K]` and rescale to sum to `B`.
    pub fn symmetric_equilibrium_sample(&self, rng: &mut StdRng) -> Result<Vec<f64>> {
        let upper = self.symmetric_uniform_upper()?;
        let mut allocation: Vec<f64> = (0..self.battlefields)
            .map(|_| rng.gen_range(0.0..upper))
            .collect();
        let total: f64 = allocation.iter().sum();
        if total > 0.0 {
            for value in &mut allocation {
                *value *= self.budget_a / total;
            }
        }
        Ok(allocation)
    }

    /// Expected payoff for player A given specific allocations on both sides.
    /// Each won battlefield contributes 1; ties contribute 0.5.
    pub fn expected_payoff(&self, allocation_a: &[f64], allocation_b: &[f64]) -> Result<f64> {
        if allocation_a.len() != self.battlefields || allocation_b.len() != self.battlefields {
            return Err(GameError::DimensionMismatch {
                expected: self.battlefields,
                actual: allocation_a.len(),
            });
        }
        ensure_finite_slice("allocation_a", allocation_a)?;
        ensure_finite_slice("allocation_b", allocation_b)?;

        let sum_a: f64 = allocation_a.iter().sum();
        let sum_b: f64 = allocation_b.iter().sum();
        if (sum_a - self.budget_a).abs() > 1e-6 {
            return Err(GameError::InvalidParameter(
                "allocation_a does not match budget",
            ));
        }
        if (sum_b - self.budget_b).abs() > 1e-6 {
            return Err(GameError::InvalidParameter(
                "allocation_b does not match budget",
            ));
        }

        let mut score = 0.0;
        for k in 0..self.battlefields {
            let a = allocation_a[k];
            let b = allocation_b[k];
            if a > b {
                score += 1.0;
            } else if a < b {
                score -= 0.0;
            } else {
                score += 0.5;
            }
        }
        Ok(score)
    }

    /// Monte-Carlo estimate of the symmetric equilibrium expected payoff to
    /// player A. With equal budgets the value should converge to
    /// `battlefields / 2`.
    pub fn monte_carlo_value(&self, samples: usize, seed: u64) -> Result<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total = 0.0;
        for _ in 0..samples {
            let allocation_a = self.symmetric_equilibrium_sample(&mut rng)?;
            let allocation_b = self.symmetric_equilibrium_sample(&mut rng)?;
            total += self.expected_payoff(&allocation_a, &allocation_b)?;
        }
        Ok(total / samples as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_budgets_symmetric_value() {
        let game = ColonelBlotto::new(10.0, 10.0, 4).unwrap();
        let value = game.monte_carlo_value(2000, 42).unwrap();
        // Symmetric game has expected value = battlefields / 2.
        assert!((value - 2.0).abs() < 0.2);
    }

    #[test]
    fn rejects_unequal_budgets_for_symmetric_helper() {
        let game = ColonelBlotto::new(10.0, 12.0, 4).unwrap();
        assert!(game.symmetric_uniform_upper().is_err());
    }

    #[test]
    fn higher_allocation_wins_battlefield() {
        let game = ColonelBlotto::new(10.0, 10.0, 2).unwrap();
        let payoff = game
            .expected_payoff(&[7.0, 3.0], &[3.0, 7.0])
            .unwrap();
        assert!((payoff - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ties_contribute_half() {
        let game = ColonelBlotto::new(10.0, 10.0, 2).unwrap();
        let payoff = game
            .expected_payoff(&[5.0, 5.0], &[5.0, 5.0])
            .unwrap();
        assert!((payoff - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(ColonelBlotto::new(-1.0, 1.0, 2).is_err());
        assert!(ColonelBlotto::new(1.0, 1.0, 1).is_err());
    }
}
