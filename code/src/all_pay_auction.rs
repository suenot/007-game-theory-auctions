//! All-pay auction with complete information and heterogeneous valuations.
//!
//! In an all-pay auction every bidder pays its bid regardless of who wins,
//! and the highest bid wins. With `n` symmetric bidders and uniform private
//! values on `[0, 1]`, the symmetric equilibrium bid for value `v` is
//!
//! ```text
//! b(v) = (n - 1) / n * v^n.
//! ```
//!
//! For two heterogeneous bidders with valuations `v_1 >= v_2`, the
//! complete-information equilibrium bid distributions are
//!
//! ```text
//! F_1(b) = b / v_2,                  on [0, v_2]
//! F_2(b) = 1 - v_2/v_1 + b / v_1,    on [0, v_2]
//! ```
//!
//! and the expected revenue (sum of all bids) is `v_2 / 2 + v_2^2 / (2 v_1)`.
//! These results follow from Baye, Kovenock and de Vries (1996).

use crate::{ensure_finite_slice, GameError, Result};

#[derive(Debug, Clone)]
pub struct AllPayAuction {
    pub valuations: Vec<f64>,
}

impl AllPayAuction {
    pub fn new(valuations: Vec<f64>) -> Result<Self> {
        if valuations.len() < 2 {
            return Err(GameError::InvalidParameter("need at least 2 bidders"));
        }
        ensure_finite_slice("valuations", &valuations)?;
        if valuations.iter().any(|&value| value <= 0.0) {
            return Err(GameError::InvalidParameter(
                "valuations must be strictly positive",
            ));
        }
        Ok(Self { valuations })
    }

    pub fn num_bidders(&self) -> usize {
        self.valuations.len()
    }

    /// Symmetric equilibrium bid for a bidder with valuation `v` in an
    /// auction with `n` symmetric bidders and uniform `[0, 1]` priors.
    pub fn symmetric_equilibrium_bid(&self, valuation: f64) -> Result<f64> {
        if !valuation.is_finite() || !(0.0..=1.0).contains(&valuation) {
            return Err(GameError::InvalidParameter("valuation must be in [0, 1]"));
        }
        let n = self.num_bidders() as f64;
        Ok((n - 1.0) / n * valuation.powf(n))
    }

    /// Expected revenue under the symmetric uniform model.
    /// Closed form: `(n - 1) / (n + 1)`.
    pub fn symmetric_expected_revenue(&self) -> f64 {
        let n = self.num_bidders() as f64;
        (n - 1.0) / (n + 1.0)
    }

    /// Expected total revenue for the two-bidder asymmetric case.
    /// Returns `None` if the auction has more than 2 bidders.
    pub fn asymmetric_expected_revenue(&self) -> Result<f64> {
        if self.num_bidders() != 2 {
            return Err(GameError::InvalidParameter(
                "asymmetric closed-form is only defined for 2 bidders",
            ));
        }
        let mut sorted = self.valuations.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let v1 = sorted[0];
        let v2 = sorted[1];
        Ok(v2 / 2.0 + v2 * v2 / (2.0 * v1))
    }

    /// Expected payoff to a bidder with valuation `v` in the symmetric model.
    /// The expected payoff to the highest type is the prize minus expected
    /// payment; lower types earn zero in expectation.
    pub fn expected_payoff(&self, valuation: f64) -> Result<f64> {
        if !valuation.is_finite() || !(0.0..=1.0).contains(&valuation) {
            return Err(GameError::InvalidParameter("valuation must be in [0, 1]"));
        }
        let n = self.num_bidders() as f64;
        let bid = self.symmetric_equilibrium_bid(valuation)?;
        let win_prob = valuation.powf(n - 1.0);
        Ok(valuation * win_prob - bid)
    }

    /// Equilibrium bids for an explicit list of valuations under the
    /// symmetric uniform model (returns one bid per bidder).
    pub fn equilibrium_bids(&self) -> Result<Vec<f64>> {
        self.valuations
            .iter()
            .map(|&v| self.symmetric_equilibrium_bid(v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_bid_matches_formula_for_two_bidders() {
        let auction = AllPayAuction::new(vec![0.5, 0.7]).unwrap();
        let bid = auction.symmetric_equilibrium_bid(0.5).unwrap();
        // (1/2) * (0.5^2) = 0.125.
        assert!((bid - 0.125).abs() < 1e-12);
    }

    #[test]
    fn revenue_increases_with_n() {
        let r2 = AllPayAuction::new(vec![0.5; 2])
            .unwrap()
            .symmetric_expected_revenue();
        let r3 = AllPayAuction::new(vec![0.5; 3])
            .unwrap()
            .symmetric_expected_revenue();
        let r10 = AllPayAuction::new(vec![0.5; 10])
            .unwrap()
            .symmetric_expected_revenue();
        assert!(r2 < r3);
        assert!(r3 < r10);
        assert!(r10 < 1.0);
    }

    #[test]
    fn asymmetric_revenue_matches_formula() {
        let auction = AllPayAuction::new(vec![1.0, 0.5]).unwrap();
        let revenue = auction.asymmetric_expected_revenue().unwrap();
        // v2 / 2 + v2^2 / (2 v1) = 0.25 + 0.125 = 0.375.
        assert!((revenue - 0.375).abs() < 1e-12);
    }

    #[test]
    fn expected_payoff_at_max_valuation_is_positive() {
        let auction = AllPayAuction::new(vec![1.0; 3]).unwrap();
        let payoff = auction.expected_payoff(1.0).unwrap();
        // E[pi | v = 1] = 1 - (n-1)/n = 1/n.
        assert!((payoff - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(AllPayAuction::new(vec![1.0]).is_err());
        assert!(AllPayAuction::new(vec![1.0, 0.0]).is_err());
        assert!(AllPayAuction::new(vec![1.0, f64::NAN]).is_err());
    }
}
