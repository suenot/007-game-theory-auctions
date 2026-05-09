use nalgebra::DMatrix;

use crate::{validate_probability_vector, validate_rows, ModelError, Result};

/// Finite Stackelberg game with a row leader and a column follower.
#[derive(Debug, Clone, PartialEq)]
pub struct StackelbergGame {
    leader_payoff: DMatrix<f64>,
    follower_payoff: DMatrix<f64>,
}

impl StackelbergGame {
    /// Creates a game from leader and follower payoff matrices.
    pub fn from_rows(leader_rows: Vec<Vec<f64>>, follower_rows: Vec<Vec<f64>>) -> Result<Self> {
        let (leader_row_count, leader_col_count) = validate_rows(&leader_rows)?;
        let (follower_row_count, follower_col_count) = validate_rows(&follower_rows)?;
        if leader_row_count != follower_row_count || leader_col_count != follower_col_count {
            return Err(ModelError::DimensionMismatch);
        }

        Ok(Self {
            leader_payoff: DMatrix::from_row_slice(
                leader_row_count,
                leader_col_count,
                &leader_rows.into_iter().flatten().collect::<Vec<_>>(),
            ),
            follower_payoff: DMatrix::from_row_slice(
                follower_row_count,
                follower_col_count,
                &follower_rows.into_iter().flatten().collect::<Vec<_>>(),
            ),
        })
    }

    /// Number of leader actions.
    pub fn leader_actions(&self) -> usize {
        self.leader_payoff.nrows()
    }

    /// Number of follower actions.
    pub fn follower_actions(&self) -> usize {
        self.leader_payoff.ncols()
    }

    /// Returns the leader's optimal pure commitment as a one-hot strategy.
    pub fn leader_optimal(&self) -> Vec<f64> {
        let mut best_row = 0;
        let mut best_payoff = f64::NEG_INFINITY;

        for row in 0..self.leader_actions() {
            let follower_col = self.follower_best_response_to_row(row);
            let payoff = self.leader_payoff[(row, follower_col)];
            if payoff > best_payoff {
                best_payoff = payoff;
                best_row = row;
            }
        }

        one_hot(self.leader_actions(), best_row)
    }

    /// Returns the follower's best response to a leader mixed strategy.
    pub fn follower_response(&self, leader_strategy: &[f64]) -> Result<Vec<f64>> {
        validate_probability_vector(leader_strategy, self.leader_actions())?;

        let mut best_payoff = f64::NEG_INFINITY;
        let mut best_cols = Vec::new();
        for col in 0..self.follower_actions() {
            let payoff = (0..self.leader_actions())
                .map(|row| leader_strategy[row] * self.follower_payoff[(row, col)])
                .sum::<f64>();
            if payoff > best_payoff + 1e-10 {
                best_payoff = payoff;
                best_cols.clear();
                best_cols.push(col);
            } else if (payoff - best_payoff).abs() <= 1e-10 {
                best_cols.push(col);
            }
        }

        let probability = 1.0 / best_cols.len() as f64;
        let mut response = vec![0.0; self.follower_actions()];
        for col in best_cols {
            response[col] = probability;
        }
        Ok(response)
    }

    /// Expected payoffs under mixed strategies.
    pub fn expected_payoffs(
        &self,
        leader_strategy: &[f64],
        follower_strategy: &[f64],
    ) -> Result<(f64, f64)> {
        validate_probability_vector(leader_strategy, self.leader_actions())?;
        validate_probability_vector(follower_strategy, self.follower_actions())?;

        let mut leader_value = 0.0;
        let mut follower_value = 0.0;
        for (row, leader_probability) in leader_strategy.iter().enumerate() {
            for (col, follower_probability) in follower_strategy.iter().enumerate() {
                let probability = leader_probability * follower_probability;
                leader_value += probability * self.leader_payoff[(row, col)];
                follower_value += probability * self.follower_payoff[(row, col)];
            }
        }
        Ok((leader_value, follower_value))
    }

    fn follower_best_response_to_row(&self, row: usize) -> usize {
        (0..self.follower_actions())
            .max_by(|&left, &right| {
                self.follower_payoff[(row, left)]
                    .partial_cmp(&self.follower_payoff[(row, right)])
                    .unwrap()
                    .then_with(|| {
                        self.leader_payoff[(row, left)]
                            .partial_cmp(&self.leader_payoff[(row, right)])
                            .unwrap()
                    })
            })
            .unwrap()
    }
}

fn one_hot(len: usize, index: usize) -> Vec<f64> {
    let mut strategy = vec![0.0; len];
    strategy[index] = 1.0;
    strategy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leader_accounts_for_follower_reaction() {
        let game = StackelbergGame::from_rows(
            vec![vec![1.0, 4.0], vec![3.0, 2.0]],
            vec![vec![3.0, 1.0], vec![1.0, 4.0]],
        )
        .unwrap();

        assert_eq!(game.leader_optimal(), vec![0.0, 1.0]);
        assert_eq!(game.follower_response(&[0.0, 1.0]).unwrap(), vec![0.0, 1.0]);
    }
}
