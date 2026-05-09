//! Kyle (1985) sequential auction model with one informed trader, noise
//! traders, and a competitive market maker.
//!
//! In the single-period version,
//!
//! ```text
//! v ~ N(0, sigma_v^2)            (asset value)
//! u ~ N(0, sigma_u^2)            (noise volume)
//! x = beta * v                   (informed trader's signed volume)
//! y = x + u                      (total order flow seen by maker)
//! p = lambda * y                 (maker's pricing rule)
//! ```
//!
//! In equilibrium
//!
//! ```text
//! beta = sigma_u / sigma_v
//! lambda = sigma_v / (2 * sigma_u)
//! ```
//!
//! and the informed trader's expected profit is
//!
//! ```text
//! pi = sigma_v * sigma_u / 2.
//! ```

use crate::{ensure_positive, GameError, Result};

#[derive(Debug, Clone, Copy)]
pub struct KyleModel {
    pub sigma_u: f64,
    pub sigma_v: f64,
}

impl KyleModel {
    pub fn new(sigma_u: f64, sigma_v: f64) -> Result<Self> {
        ensure_positive("sigma_u", sigma_u)?;
        ensure_positive("sigma_v", sigma_v)?;
        Ok(Self { sigma_u, sigma_v })
    }

    /// Equilibrium informed-trader trading intensity `beta = sigma_u / sigma_v`.
    pub fn informed_intensity(&self) -> f64 {
        self.sigma_u / self.sigma_v
    }

    /// Equilibrium price impact `lambda = sigma_v / (2 sigma_u)`.
    pub fn equilibrium_lambda(&self) -> f64 {
        self.sigma_v / (2.0 * self.sigma_u)
    }

    /// Optimal informed-trader signed volume given a private value signal.
    pub fn informed_strategy(&self, private_info: f64) -> Result<f64> {
        if !private_info.is_finite() {
            return Err(GameError::NonFiniteInput("private_info"));
        }
        Ok(self.informed_intensity() * private_info)
    }

    /// Market-maker pricing rule given total observed order flow `y`.
    pub fn pricing_rule(&self, total_order: f64) -> Result<f64> {
        if !total_order.is_finite() {
            return Err(GameError::NonFiniteInput("total_order"));
        }
        Ok(self.equilibrium_lambda() * total_order)
    }

    /// Expected profit of the informed trader,
    /// `E[pi] = sigma_v * sigma_u / 2`.
    pub fn expected_informed_profit(&self) -> f64 {
        self.sigma_v * self.sigma_u / 2.0
    }

    /// Probability of informed trading PIN-style: with normal volumes the
    /// fraction of variance in order flow attributable to the informed
    /// trader is `beta^2 sigma_v^2 / (beta^2 sigma_v^2 + sigma_u^2)`. Plug
    /// in the equilibrium beta and the expression simplifies to 1/2.
    pub fn informed_volume_share(&self) -> f64 {
        let beta = self.informed_intensity();
        let informed_var = beta * beta * self.sigma_v * self.sigma_v;
        let noise_var = self.sigma_u * self.sigma_u;
        informed_var / (informed_var + noise_var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_lambda_matches_formula() {
        let model = KyleModel::new(1.0, 2.0).unwrap();
        let lambda = model.equilibrium_lambda();
        assert!((lambda - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pricing_rule_is_linear_in_order_flow() {
        let model = KyleModel::new(2.0, 4.0).unwrap();
        let p1 = model.pricing_rule(1.0).unwrap();
        let p2 = model.pricing_rule(2.0).unwrap();
        let p3 = model.pricing_rule(3.0).unwrap();
        assert!((p2 - 2.0 * p1).abs() < 1e-12);
        assert!((p3 - 3.0 * p1).abs() < 1e-12);
    }

    #[test]
    fn informed_strategy_scales_with_signal() {
        let model = KyleModel::new(1.0, 1.0).unwrap();
        let x = model.informed_strategy(2.0).unwrap();
        assert!((x - 2.0).abs() < 1e-12);
    }

    #[test]
    fn informed_volume_share_is_one_half_in_equilibrium() {
        let model = KyleModel::new(0.7, 1.3).unwrap();
        let share = model.informed_volume_share();
        assert!((share - 0.5).abs() < 1e-12);
    }

    #[test]
    fn expected_profit_matches_formula() {
        let model = KyleModel::new(0.5, 4.0).unwrap();
        let profit = model.expected_informed_profit();
        assert!((profit - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(KyleModel::new(0.0, 1.0).is_err());
        assert!(KyleModel::new(1.0, -1.0).is_err());
    }
}
