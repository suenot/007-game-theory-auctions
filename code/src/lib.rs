//! Game-theoretic and auction models for algorithmic trading examples.

pub mod all_pay_auction;
pub mod auctions;
pub mod colonel_blotto;
pub mod hft_arms_race;
pub mod kyle_model;
pub mod stackelberg;
pub mod zero_sum_games;

use thiserror::Error;

/// Shared result type for model constructors and checked calculations.
pub type Result<T> = std::result::Result<T, ModelError>;

/// Errors returned when model inputs violate mathematical assumptions.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ModelError {
    #[error("matrix must contain at least one row and one column")]
    EmptyMatrix,
    #[error("all matrix rows must have the same length")]
    RaggedMatrix,
    #[error("input contains a non-finite value")]
    NonFiniteValue,
    #[error("dimensions do not match")]
    DimensionMismatch,
    #[error("probabilities must be finite, non-negative, and sum to one")]
    InvalidProbabilityVector,
    #[error("input must be strictly positive")]
    NonPositiveInput,
    #[error("at least one participant is required")]
    EmptyParticipants,
}

pub(crate) fn validate_rows(rows: &[Vec<f64>]) -> Result<(usize, usize)> {
    let row_count = rows.len();
    let col_count = rows.first().map_or(0, Vec::len);
    if row_count == 0 || col_count == 0 {
        return Err(ModelError::EmptyMatrix);
    }
    if rows.iter().any(|row| row.len() != col_count) {
        return Err(ModelError::RaggedMatrix);
    }
    if rows.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteValue);
    }
    Ok((row_count, col_count))
}

pub(crate) fn validate_probability_vector(values: &[f64], expected_len: usize) -> Result<()> {
    if values.len() != expected_len {
        return Err(ModelError::DimensionMismatch);
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ModelError::InvalidProbabilityVector);
    }
    let total: f64 = values.iter().sum();
    if (total - 1.0).abs() > 1e-8 {
        return Err(ModelError::InvalidProbabilityVector);
    }
    Ok(())
}
