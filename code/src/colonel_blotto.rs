use crate::{ModelError, Result};

/// Colonel Blotto allocation game for distributing liquidity across venues.
#[derive(Debug, Clone, PartialEq)]
pub struct ColonelBlotto {
    pub players: usize,
    pub battlefields: usize,
    pub budget: Vec<f64>,
}

impl ColonelBlotto {
    pub fn new(players: usize, battlefields: usize, budget: Vec<f64>) -> Result<Self> {
        if players == 0 || battlefields == 0 || budget.len() != players {
            return Err(ModelError::DimensionMismatch);
        }
        if budget
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModelError::NonFiniteValue);
        }
        Ok(Self {
            players,
            battlefields,
            budget,
        })
    }

    /// Symmetric benchmark allocation: split the average budget equally.
    pub fn symmetric_equilibrium(&self) -> Vec<f64> {
        let average_budget = self.budget.iter().sum::<f64>() / self.players as f64;
        vec![average_budget / self.battlefields as f64; self.battlefields]
    }

    /// Scores an allocation against an opponent: win = 1, tie = 0.5, loss = 0.
    pub fn expected_payoff(&self, allocation: &[f64], opponent: &[f64]) -> Result<f64> {
        if allocation.len() != self.battlefields || opponent.len() != self.battlefields {
            return Err(ModelError::DimensionMismatch);
        }
        if allocation
            .iter()
            .chain(opponent.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModelError::NonFiniteValue);
        }

        Ok(allocation
            .iter()
            .zip(opponent.iter())
            .map(|(left, right)| {
                if left > right {
                    1.0
                } else if (left - right).abs() <= 1e-10 {
                    0.5
                } else {
                    0.0
                }
            })
            .sum())
    }

    /// Normalizes any non-negative allocation to a target budget.
    pub fn normalize_allocation(allocation: &[f64], budget: f64) -> Result<Vec<f64>> {
        if !budget.is_finite() || budget < 0.0 {
            return Err(ModelError::NonFiniteValue);
        }
        if allocation
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModelError::NonFiniteValue);
        }
        let total: f64 = allocation.iter().sum();
        if total == 0.0 {
            return Ok(vec![0.0; allocation.len()]);
        }
        Ok(allocation
            .iter()
            .map(|value| value * budget / total)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_budget_splits_evenly() {
        let game = ColonelBlotto::new(2, 3, vec![9.0, 9.0]).unwrap();

        assert_eq!(game.symmetric_equilibrium(), vec![3.0, 3.0, 3.0]);
    }
}
