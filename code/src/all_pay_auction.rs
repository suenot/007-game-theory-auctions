use crate::{ModelError, Result};

/// All-pay auction where every bidder pays their bid and the highest bid wins.
#[derive(Debug, Clone, PartialEq)]
pub struct AllPayAuction {
    pub valuations: Vec<f64>,
    pub num_bidders: usize,
}

impl AllPayAuction {
    pub fn new(valuations: Vec<f64>) -> Result<Self> {
        if valuations.is_empty() {
            return Err(ModelError::EmptyParticipants);
        }
        if valuations
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModelError::NonFiniteValue);
        }
        Ok(Self {
            num_bidders: valuations.len(),
            valuations,
        })
    }

    /// Symmetric equilibrium bid curve for uniform private values on `[0, max(v)]`.
    pub fn equilibrium_bids(&self) -> Vec<f64> {
        let max_value = self.valuations.iter().copied().fold(0.0, f64::max);
        if max_value == 0.0 {
            return vec![0.0; self.valuations.len()];
        }

        let n = self.num_bidders as i32;
        let coefficient = (self.num_bidders as f64 - 1.0) / self.num_bidders as f64;
        self.valuations
            .iter()
            .map(|value| {
                let normalized = (value / max_value).clamp(0.0, 1.0);
                max_value * coefficient * normalized.powi(n)
            })
            .collect()
    }

    /// Total payments collected in expectation under the bid curve.
    pub fn expected_revenue(&self) -> f64 {
        self.equilibrium_bids().iter().sum()
    }

    /// Ex-post payoff for a bidder after all-pay bids are submitted.
    pub fn bidder_payoff(&self, bidder_index: usize, bids: &[f64]) -> Result<f64> {
        if bidder_index >= self.valuations.len() || bids.len() != self.valuations.len() {
            return Err(ModelError::DimensionMismatch);
        }
        if bids.iter().any(|bid| !bid.is_finite() || *bid < 0.0) {
            return Err(ModelError::NonFiniteValue);
        }
        let winning_bid = bids.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let wins = (bids[bidder_index] - winning_bid).abs() <= 1e-10;
        Ok(if wins {
            self.valuations[bidder_index] - bids[bidder_index]
        } else {
            -bids[bidder_index]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_bids_are_monotone() {
        let auction = AllPayAuction::new(vec![0.25, 0.5, 1.0]).unwrap();
        let bids = auction.equilibrium_bids();

        assert!(bids[0] < bids[1]);
        assert!(bids[1] < bids[2]);
    }
}
