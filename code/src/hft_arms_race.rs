use crate::{ModelError, Result};

/// High-frequency trading participant with a private latency-arbitrage value.
#[derive(Debug, Clone, PartialEq)]
pub struct HFT {
    pub id: String,
    pub private_value: f64,
}

impl HFT {
    pub fn new(id: impl Into<String>, private_value: f64) -> Self {
        Self {
            id: id.into(),
            private_value,
        }
    }
}

/// All-pay-style speed investment model.
#[derive(Debug, Clone, PartialEq)]
pub struct HFTArmsRace {
    pub participants: Vec<HFT>,
    pub speed_costs: Vec<f64>,
}

impl HFTArmsRace {
    pub fn new(participants: Vec<HFT>, speed_costs: Vec<f64>) -> Result<Self> {
        if participants.is_empty() {
            return Err(ModelError::EmptyParticipants);
        }
        if participants.len() != speed_costs.len() {
            return Err(ModelError::DimensionMismatch);
        }
        if participants.iter().any(|participant| {
            !participant.private_value.is_finite() || participant.private_value < 0.0
        }) || speed_costs
            .iter()
            .any(|cost| !cost.is_finite() || *cost <= 0.0)
        {
            return Err(ModelError::NonPositiveInput);
        }

        Ok(Self {
            participants,
            speed_costs,
        })
    }

    /// Symmetric all-pay approximation of equilibrium speed investment.
    pub fn equilibrium_speeds(&self) -> Vec<f64> {
        let n = self.participants.len() as f64;
        let shading = if n <= 1.0 { 1.0 } else { (n - 1.0) / n };
        self.participants
            .iter()
            .zip(self.speed_costs.iter())
            .map(|(participant, cost)| participant.private_value * shading / cost)
            .collect()
    }

    /// Regulator's one-winner benchmark: fund the highest value/cost speed only.
    pub fn socially_optimal_speed(&self) -> f64 {
        self.participants
            .iter()
            .zip(self.speed_costs.iter())
            .map(|(participant, cost)| participant.private_value / cost)
            .fold(0.0, f64::max)
    }

    /// Excess private investment over the one-winner social benchmark.
    pub fn deadweight_speed_loss(&self) -> f64 {
        (self.equilibrium_speeds().iter().sum::<f64>() - self.socially_optimal_speed()).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_race_has_positive_deadweight_loss_with_many_players() {
        let race = HFTArmsRace::new(
            vec![
                HFT::new("a", 10.0),
                HFT::new("b", 10.0),
                HFT::new("c", 10.0),
            ],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();

        assert!(race.deadweight_speed_loss() > 0.0);
    }
}
