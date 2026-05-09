use crate::{ModelError, Result};

/// Kyle (1985) single-period informed-trading model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KyleModel {
    pub sigma_u: f64,
    pub sigma_v: f64,
    pub lambda: f64,
}

impl KyleModel {
    /// Creates the equilibrium model from noise-trader volume and asset-value
    /// standard deviations.
    pub fn new(sigma_u: f64, sigma_v: f64) -> Result<Self> {
        if !sigma_u.is_finite() || !sigma_v.is_finite() || sigma_u <= 0.0 || sigma_v <= 0.0 {
            return Err(ModelError::NonPositiveInput);
        }
        Ok(Self {
            sigma_u,
            sigma_v,
            lambda: sigma_v / (2.0 * sigma_u),
        })
    }

    /// Creates a model with an externally calibrated price-impact coefficient.
    pub fn with_lambda(sigma_u: f64, sigma_v: f64, lambda: f64) -> Result<Self> {
        if [sigma_u, sigma_v, lambda]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ModelError::NonPositiveInput);
        }
        Ok(Self {
            sigma_u,
            sigma_v,
            lambda,
        })
    }

    /// Equilibrium price impact, `lambda = sigma_v / (2 sigma_u)`.
    pub fn equilibrium_lambda(&self) -> f64 {
        self.lambda
    }

    /// Informed trader order size for private value signal `v`.
    pub fn informed_strategy(&self, private_info: f64) -> f64 {
        private_info / (2.0 * self.lambda)
    }

    /// Competitive market-maker pricing rule for total order flow.
    pub fn pricing_rule(&self, total_order: f64) -> f64 {
        self.lambda * total_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_positive_volatility() {
        assert_eq!(
            KyleModel::new(0.0, 1.0).unwrap_err(),
            ModelError::NonPositiveInput
        );
    }
}
