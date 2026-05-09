use nalgebra::DMatrix;

use crate::{validate_rows, ModelError, Result};

const EPS: f64 = 1e-10;

/// Player role in a two-player zero-sum normal-form game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    /// Row player maximizes the payoff matrix entries.
    Row,
    /// Column player minimizes the row player's payoff.
    Column,
}

/// Finite zero-sum game represented by the row player's payoff matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroSumGame {
    payoff_matrix: DMatrix<f64>,
}

impl ZeroSumGame {
    /// Builds a validated game from row-major payoff values.
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self> {
        let (row_count, col_count) = validate_rows(&rows)?;
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Ok(Self {
            payoff_matrix: DMatrix::from_row_slice(row_count, col_count, &flat),
        })
    }

    /// Returns the number of row-player pure strategies.
    pub fn rows(&self) -> usize {
        self.payoff_matrix.nrows()
    }

    /// Returns the number of column-player pure strategies.
    pub fn cols(&self) -> usize {
        self.payoff_matrix.ncols()
    }

    /// Returns the payoff at `(row, col)`.
    pub fn payoff(&self, row: usize, col: usize) -> Option<f64> {
        self.payoff_matrix.get((row, col)).copied()
    }

    /// Returns the row-player mixed Nash equilibrium.
    ///
    /// Two-by-two games are solved analytically. Larger matrices use
    /// deterministic fictitious play, which is sufficient for the educational
    /// simulations in this chapter and keeps the crate dependency-light.
    pub fn nash_equilibrium(&self) -> Vec<f64> {
        if self.rows() == 2 && self.cols() == 2 {
            return self.two_by_two_row_strategy();
        }
        self.fictitious_play_row_strategy(20_000)
    }

    /// Returns the column-player mixed Nash equilibrium.
    pub fn column_equilibrium(&self) -> Vec<f64> {
        if self.rows() == 2 && self.cols() == 2 {
            return self.two_by_two_column_strategy();
        }
        self.fictitious_play_column_strategy(20_000)
    }

    /// Computes the value of the game under the returned mixed equilibria.
    pub fn game_value(&self) -> f64 {
        let row_strategy = self.nash_equilibrium();
        let col_strategy = self.column_equilibrium();
        self.expected_payoff(&row_strategy, &col_strategy)
            .expect("internal equilibrium dimensions are valid")
    }

    /// Computes expected row-player payoff for mixed strategies.
    pub fn expected_payoff(&self, row_strategy: &[f64], col_strategy: &[f64]) -> Result<f64> {
        crate::validate_probability_vector(row_strategy, self.rows())?;
        crate::validate_probability_vector(col_strategy, self.cols())?;

        let mut payoff = 0.0;
        for (row, row_probability) in row_strategy.iter().enumerate() {
            for (col, col_probability) in col_strategy.iter().enumerate() {
                payoff += row_probability * col_probability * self.payoff_matrix[(row, col)];
            }
        }
        Ok(payoff)
    }

    /// Returns a pure dominant strategy index if one exists.
    pub fn dominant_strategy(&self, player: Player) -> Option<usize> {
        let strategy_count = match player {
            Player::Row => self.rows(),
            Player::Column => self.cols(),
        };

        (0..strategy_count).find(|&candidate| {
            (0..strategy_count)
                .filter(|&other| other != candidate)
                .all(|other| self.strategy_dominates(candidate, other, player))
        })
    }

    /// Enumerates pure-strategy Nash equilibria as `(row, col)` pairs.
    pub fn pure_nash_equilibria(&self) -> Vec<(usize, usize)> {
        let mut equilibria = Vec::new();
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let value = self.payoff_matrix[(row, col)];
                let row_best_response = (0..self.rows())
                    .all(|candidate| value + EPS >= self.payoff_matrix[(candidate, col)]);
                let col_best_response = (0..self.cols())
                    .all(|candidate| value <= self.payoff_matrix[(row, candidate)] + EPS);

                if row_best_response && col_best_response {
                    equilibria.push((row, col));
                }
            }
        }
        equilibria
    }

    fn strategy_dominates(&self, candidate: usize, other: usize, player: Player) -> bool {
        let mut strict = false;
        match player {
            Player::Row => {
                for col in 0..self.cols() {
                    let lhs = self.payoff_matrix[(candidate, col)];
                    let rhs = self.payoff_matrix[(other, col)];
                    if lhs + EPS < rhs {
                        return false;
                    }
                    strict |= lhs > rhs + EPS;
                }
            }
            Player::Column => {
                for row in 0..self.rows() {
                    let lhs = self.payoff_matrix[(row, candidate)];
                    let rhs = self.payoff_matrix[(row, other)];
                    if lhs > rhs + EPS {
                        return false;
                    }
                    strict |= lhs + EPS < rhs;
                }
            }
        }
        strict
    }

    fn two_by_two_row_strategy(&self) -> Vec<f64> {
        let a = self.payoff_matrix[(0, 0)];
        let b = self.payoff_matrix[(0, 1)];
        let c = self.payoff_matrix[(1, 0)];
        let d = self.payoff_matrix[(1, 1)];
        let denom = a - b - c + d;
        if denom.abs() <= EPS {
            return self.row_maximin_strategy();
        }

        let p = ((d - c) / denom).clamp(0.0, 1.0);
        if p <= EPS || p >= 1.0 - EPS {
            self.row_maximin_strategy()
        } else {
            vec![p, 1.0 - p]
        }
    }

    fn two_by_two_column_strategy(&self) -> Vec<f64> {
        let a = self.payoff_matrix[(0, 0)];
        let b = self.payoff_matrix[(0, 1)];
        let c = self.payoff_matrix[(1, 0)];
        let d = self.payoff_matrix[(1, 1)];
        let denom = a - b - c + d;
        if denom.abs() <= EPS {
            return self.column_minimax_strategy();
        }

        let q = ((d - b) / denom).clamp(0.0, 1.0);
        if q <= EPS || q >= 1.0 - EPS {
            self.column_minimax_strategy()
        } else {
            vec![q, 1.0 - q]
        }
    }

    fn row_maximin_strategy(&self) -> Vec<f64> {
        let mut best_row = 0;
        let mut best_floor = f64::NEG_INFINITY;
        for row in 0..self.rows() {
            let floor = (0..self.cols())
                .map(|col| self.payoff_matrix[(row, col)])
                .fold(f64::INFINITY, f64::min);
            if floor > best_floor {
                best_floor = floor;
                best_row = row;
            }
        }
        one_hot(self.rows(), best_row)
    }

    fn column_minimax_strategy(&self) -> Vec<f64> {
        let mut best_col = 0;
        let mut best_ceiling = f64::INFINITY;
        for col in 0..self.cols() {
            let ceiling = (0..self.rows())
                .map(|row| self.payoff_matrix[(row, col)])
                .fold(f64::NEG_INFINITY, f64::max);
            if ceiling < best_ceiling {
                best_ceiling = ceiling;
                best_col = col;
            }
        }
        one_hot(self.cols(), best_col)
    }

    fn fictitious_play_row_strategy(&self, iterations: usize) -> Vec<f64> {
        let (row_counts, _) = self.fictitious_play_counts(iterations);
        normalize_counts(&row_counts)
    }

    fn fictitious_play_column_strategy(&self, iterations: usize) -> Vec<f64> {
        let (_, col_counts) = self.fictitious_play_counts(iterations);
        normalize_counts(&col_counts)
    }

    fn fictitious_play_counts(&self, iterations: usize) -> (Vec<f64>, Vec<f64>) {
        let mut row_counts = vec![1.0; self.rows()];
        let mut col_counts = vec![1.0; self.cols()];

        for _ in 0..iterations {
            let col_strategy = normalize_counts(&col_counts);
            let row_action = (0..self.rows())
                .max_by(|&left, &right| {
                    expected_row_payoff(self, left, &col_strategy)
                        .partial_cmp(&expected_row_payoff(self, right, &col_strategy))
                        .unwrap()
                })
                .unwrap();
            row_counts[row_action] += 1.0;

            let row_strategy = normalize_counts(&row_counts);
            let col_action = (0..self.cols())
                .min_by(|&left, &right| {
                    expected_col_payoff(self, &row_strategy, left)
                        .partial_cmp(&expected_col_payoff(self, &row_strategy, right))
                        .unwrap()
                })
                .unwrap();
            col_counts[col_action] += 1.0;
        }

        (row_counts, col_counts)
    }
}

impl TryFrom<Vec<Vec<f64>>> for ZeroSumGame {
    type Error = ModelError;

    fn try_from(value: Vec<Vec<f64>>) -> Result<Self> {
        Self::from_rows(value)
    }
}

fn expected_row_payoff(game: &ZeroSumGame, row: usize, col_strategy: &[f64]) -> f64 {
    (0..game.cols())
        .map(|col| col_strategy[col] * game.payoff_matrix[(row, col)])
        .sum()
}

fn expected_col_payoff(game: &ZeroSumGame, row_strategy: &[f64], col: usize) -> f64 {
    (0..game.rows())
        .map(|row| row_strategy[row] * game.payoff_matrix[(row, col)])
        .sum()
}

fn normalize_counts(counts: &[f64]) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    counts.iter().map(|count| count / total).collect()
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
    fn identifies_pure_saddle_point() {
        let game = ZeroSumGame::from_rows(vec![vec![3.0, 1.0], vec![2.0, 0.0]]).unwrap();

        assert_eq!(game.pure_nash_equilibria(), vec![(0, 1)]);
        assert_eq!(game.nash_equilibrium(), vec![1.0, 0.0]);
        assert_eq!(game.column_equilibrium(), vec![0.0, 1.0]);
    }

    #[test]
    fn detects_row_dominance() {
        let game = ZeroSumGame::from_rows(vec![vec![2.0, 2.0], vec![1.0, 0.0]]).unwrap();

        assert_eq!(game.dominant_strategy(Player::Row), Some(0));
    }
}
