//! Stackelberg leader/follower games.
//!
//! In a Stackelberg game, the leader commits to a (possibly mixed) strategy
//! first, and the follower observes it before choosing its own response. We
//! model the discrete bilevel problem
//!
//! ```text
//! max_x leader_payoff(x, BR(x))
//! BR(x) = argmax_y follower_payoff(x, y)
//! ```
//!
//! Both players have finite pure-strategy spaces. The follower's best
//! response is computed by exhaustive search over its strategies; the
//! leader's optimization is also exhaustive over its pure strategies. This
//! is sufficient for small teaching examples and matches the chapter's
//! expectations.

use nalgebra::DMatrix;

use crate::{GameError, Result};

#[derive(Debug, Clone)]
pub struct StackelbergGame {
    leader_payoff: DMatrix<f64>,
    follower_payoff: DMatrix<f64>,
}

impl StackelbergGame {
    /// Build the bimatrix game from leader and follower payoff matrices.
    /// Both matrices are indexed `[leader_action][follower_action]`.
    pub fn new(leader_payoff: DMatrix<f64>, follower_payoff: DMatrix<f64>) -> Result<Self> {
        if leader_payoff.shape() != follower_payoff.shape() {
            return Err(GameError::DimensionMismatch {
                expected: leader_payoff.nrows() * leader_payoff.ncols(),
                actual: follower_payoff.nrows() * follower_payoff.ncols(),
            });
        }
        if leader_payoff.nrows() == 0 || leader_payoff.ncols() == 0 {
            return Err(GameError::EmptyInput);
        }
        if leader_payoff.iter().any(|value| !value.is_finite())
            || follower_payoff.iter().any(|value| !value.is_finite())
        {
            return Err(GameError::NonFiniteInput("payoff_matrix"));
        }

        Ok(Self {
            leader_payoff,
            follower_payoff,
        })
    }

    /// Convenience constructor from row vectors.
    pub fn from_rows(leader: Vec<Vec<f64>>, follower: Vec<Vec<f64>>) -> Result<Self> {
        if leader.is_empty() || follower.is_empty() {
            return Err(GameError::EmptyInput);
        }
        let n = leader.len();
        let m = leader[0].len();
        let l = matrix_from_rows(leader, n, m)?;
        let f = matrix_from_rows(follower, n, m)?;
        Self::new(l, f)
    }

    /// Number of leader actions.
    pub fn leader_actions(&self) -> usize {
        self.leader_payoff.nrows()
    }

    /// Number of follower actions.
    pub fn follower_actions(&self) -> usize {
        self.leader_payoff.ncols()
    }

    /// Best-response of the follower to a (possibly mixed) leader strategy.
    /// Returns a pure strategy distribution that puts mass 1 on the
    /// follower's best deterministic response. Ties are broken by lowest
    /// index.
    pub fn follower_response(&self, leader_strategy: &[f64]) -> Result<Vec<f64>> {
        if leader_strategy.len() != self.leader_actions() {
            return Err(GameError::DimensionMismatch {
                expected: self.leader_actions(),
                actual: leader_strategy.len(),
            });
        }
        if leader_strategy.iter().any(|value| !value.is_finite()) {
            return Err(GameError::NonFiniteInput("leader_strategy"));
        }
        let total: f64 = leader_strategy.iter().sum();
        if total <= 0.0 {
            return Err(GameError::InvalidParameter("leader strategy must sum to > 0"));
        }

        let mut best_index = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for j in 0..self.follower_actions() {
            let mut value = 0.0;
            for i in 0..self.leader_actions() {
                value += leader_strategy[i] * self.follower_payoff[(i, j)];
            }
            value /= total;
            if value > best_value {
                best_value = value;
                best_index = j;
            }
        }

        let mut response = vec![0.0; self.follower_actions()];
        response[best_index] = 1.0;
        Ok(response)
    }

    /// Optimal pure leader strategy given the follower will best-respond.
    /// Returns the leader's pure strategy as a one-hot vector.
    pub fn leader_optimal(&self) -> Result<Vec<f64>> {
        let mut best_leader = 0usize;
        let mut best_payoff = f64::NEG_INFINITY;

        for i in 0..self.leader_actions() {
            let mut leader_strategy = vec![0.0; self.leader_actions()];
            leader_strategy[i] = 1.0;
            let response = self.follower_response(&leader_strategy)?;

            let mut payoff = 0.0;
            for j in 0..self.follower_actions() {
                payoff += response[j] * self.leader_payoff[(i, j)];
            }
            if payoff > best_payoff {
                best_payoff = payoff;
                best_leader = i;
            }
        }

        let mut leader_strategy = vec![0.0; self.leader_actions()];
        leader_strategy[best_leader] = 1.0;
        Ok(leader_strategy)
    }

    /// Equilibrium payoffs (leader, follower) under the Stackelberg solution.
    pub fn equilibrium_payoffs(&self) -> Result<(f64, f64)> {
        let leader_strategy = self.leader_optimal()?;
        let response = self.follower_response(&leader_strategy)?;

        let mut leader_value = 0.0;
        let mut follower_value = 0.0;
        for i in 0..self.leader_actions() {
            for j in 0..self.follower_actions() {
                leader_value += leader_strategy[i] * response[j] * self.leader_payoff[(i, j)];
                follower_value += leader_strategy[i] * response[j] * self.follower_payoff[(i, j)];
            }
        }
        Ok((leader_value, follower_value))
    }
}

fn matrix_from_rows(rows: Vec<Vec<f64>>, n: usize, m: usize) -> Result<DMatrix<f64>> {
    for row in &rows {
        if row.len() != m {
            return Err(GameError::DimensionMismatch {
                expected: m,
                actual: row.len(),
            });
        }
    }
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(DMatrix::from_row_slice(n, m, &flat))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follower_picks_best_response() {
        // Leader plays first. Follower's payoffs reward column 1 most when
        // leader plays row 0.
        let leader = vec![vec![5.0, 1.0], vec![0.0, 4.0]];
        let follower = vec![vec![0.0, 3.0], vec![4.0, 1.0]];
        let game = StackelbergGame::from_rows(leader, follower).unwrap();

        let response = game.follower_response(&[1.0, 0.0]).unwrap();
        assert_eq!(response, vec![0.0, 1.0]);

        let response = game.follower_response(&[0.0, 1.0]).unwrap();
        assert_eq!(response, vec![1.0, 0.0]);
    }

    #[test]
    fn leader_optimizes_against_best_response() {
        // Leader gains 5 if follower plays column 1, gains 4 if follower
        // plays column 0. Since follower will best-respond, leader has to
        // anticipate.
        let leader = vec![vec![5.0, 1.0], vec![0.0, 4.0]];
        let follower = vec![vec![0.0, 3.0], vec![4.0, 1.0]];
        let game = StackelbergGame::from_rows(leader, follower).unwrap();

        let optimal = game.leader_optimal().unwrap();
        // Picking row 0 induces follower to pick column 1 (payoff 1 to leader).
        // Picking row 1 induces follower to pick column 0 (payoff 0 to leader).
        // So leader picks row 0.
        assert_eq!(optimal, vec![1.0, 0.0]);

        let (leader_value, follower_value) = game.equilibrium_payoffs().unwrap();
        assert!((leader_value - 1.0).abs() < 1e-12);
        assert!((follower_value - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let leader = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let follower = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matches!(
            StackelbergGame::new(leader, follower),
            Err(GameError::DimensionMismatch { .. })
        ));
    }
}
