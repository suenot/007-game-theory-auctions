//! HFT speed arms race modeled as a Tullock contest (rent-seeking game).
//!
//! Each of `n` HFT firms invests in latency reduction. Speed `s_i >= 0`
//! costs `c_i * s_i` and the firm with relatively higher speed wins a
//! larger share of the per-period prize. The win probability follows the
//! standard Tullock / contest-success function
//!
//! ```text
//! p_i = s_i^r / sum_j s_j^r           (with r > 0, here r = 1)
//! ```
//!
//! With `r = 1` and symmetric costs `c`, the unique symmetric Nash
//! equilibrium is
//!
//! ```text
//! s* = V * (n - 1) / (n^2 * c)
//! ```
//!
//! and total rent dissipation is `V * (n - 1) / n`. As `n -> infinity` all
//! rents are dissipated, recovering the all-pay auction limit.
//!
//! For asymmetric costs we expose a best-response iteration. The socially
//! optimal investment is zero: pure rent-seeking is a deadweight loss in
//! this stylized model.

use crate::{ensure_finite_slice, ensure_positive, GameError, Result};

/// One HFT participant.
#[derive(Debug, Clone, Copy)]
pub struct HFT {
    /// Marginal cost of one unit of speed.
    pub speed_cost: f64,
}

impl HFT {
    pub fn new(speed_cost: f64) -> Result<Self> {
        ensure_positive("speed_cost", speed_cost)?;
        Ok(Self { speed_cost })
    }
}

#[derive(Debug, Clone)]
pub struct HFTArmsRace {
    pub participants: Vec<HFT>,
    pub prize: f64,
}

impl HFTArmsRace {
    pub fn new(participants: Vec<HFT>, prize: f64) -> Result<Self> {
        if participants.len() < 2 {
            return Err(GameError::InvalidParameter("need at least 2 participants"));
        }
        ensure_positive("prize", prize)?;
        Ok(Self {
            participants,
            prize,
        })
    }

    pub fn n(&self) -> usize {
        self.participants.len()
    }

    /// Win probability of player `i` under the Tullock success function.
    /// Returns `1/n` when all speeds are zero.
    pub fn win_probabilities(&self, speeds: &[f64]) -> Result<Vec<f64>> {
        if speeds.len() != self.n() {
            return Err(GameError::DimensionMismatch {
                expected: self.n(),
                actual: speeds.len(),
            });
        }
        ensure_finite_slice("speeds", speeds)?;
        let total: f64 = speeds.iter().sum();
        if total <= 0.0 {
            return Ok(vec![1.0 / self.n() as f64; self.n()]);
        }
        Ok(speeds.iter().map(|&s| s / total).collect())
    }

    /// Expected profit of player `i` given everyone's speeds.
    pub fn expected_profit(&self, speeds: &[f64], i: usize) -> Result<f64> {
        let probs = self.win_probabilities(speeds)?;
        let cost = self.participants[i].speed_cost * speeds[i];
        Ok(self.prize * probs[i] - cost)
    }

    /// Closed-form symmetric Nash equilibrium speed for symmetric costs.
    /// `s* = V * (n - 1) / (n^2 * c)`.
    pub fn symmetric_equilibrium_speed(&self) -> Result<f64> {
        let c = self.participants[0].speed_cost;
        for participant in &self.participants {
            if (participant.speed_cost - c).abs() > 1e-12 {
                return Err(GameError::InvalidParameter("costs are not symmetric"));
            }
        }
        let n = self.n() as f64;
        Ok(self.prize * (n - 1.0) / (n * n * c))
    }

    /// Equilibrium speeds for arbitrary costs. For the Tullock contest
    /// with linear costs and `p_i = s_i / S`, the FOC for an active player
    /// is
    ///
    /// ```text
    /// V * (S - s_i) / S^2 = c_i,
    /// ```
    ///
    /// summed over active players gives `S = V * (k - 1) / sum_active c_i`,
    /// so we identify the active set by iteratively dropping the most
    /// expensive player whose closed-form `s_i` would be non-positive. The
    /// closed-form expression for the survivor speeds is then
    ///
    /// ```text
    /// s_i = S * (1 - c_i * (k - 1) / sum_active c_j),
    /// ```
    ///
    /// which we return here (with zero for inactive players).
    pub fn equilibrium_speeds(&self) -> Result<Vec<f64>> {
        let costs: Vec<f64> = self.participants.iter().map(|p| p.speed_cost).collect();
        let n = costs.len();

        let mut active: Vec<bool> = vec![true; n];
        loop {
            let active_indices: Vec<usize> =
                (0..n).filter(|&i| active[i]).collect();
            let k = active_indices.len();
            if k < 2 {
                break;
            }
            let sum_c: f64 = active_indices.iter().map(|&i| costs[i]).sum();
            let mut worst: Option<usize> = None;
            let mut worst_value = f64::NEG_INFINITY;
            for &i in &active_indices {
                let share = costs[i] * (k as f64 - 1.0) / sum_c;
                if share >= 1.0 && costs[i] > worst_value {
                    worst_value = costs[i];
                    worst = Some(i);
                }
            }
            match worst {
                Some(i) => active[i] = false,
                None => break,
            }
        }

        let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        let k = active_indices.len();
        let mut speeds = vec![0.0; n];
        if k < 2 {
            return Ok(speeds);
        }
        let sum_c: f64 = active_indices.iter().map(|&i| costs[i]).sum();
        let total = self.prize * (k as f64 - 1.0) / sum_c;

        for &i in &active_indices {
            let s = total * (1.0 - costs[i] * (k as f64 - 1.0) / sum_c);
            speeds[i] = s.max(0.0);
        }

        Ok(speeds)
    }

    /// Socially optimal speed: speed investments are pure rent dissipation
    /// in this stylized model, so welfare-maximizing speed is zero for
    /// each participant.
    pub fn socially_optimal_speed(&self) -> f64 {
        0.0
    }

    /// Total rent dissipation (deadweight loss) at the equilibrium.
    pub fn deadweight_loss(&self) -> Result<f64> {
        let speeds = self.equilibrium_speeds()?;
        let total_cost: f64 = speeds
            .iter()
            .zip(self.participants.iter())
            .map(|(speed, hft)| speed * hft.speed_cost)
            .sum();
        Ok(total_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn symmetric_equilibrium_matches_closed_form() {
        let race =
            HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 4], 16.0).unwrap();
        let s = race.symmetric_equilibrium_speed().unwrap();
        let expected = 16.0 * (4.0 - 1.0) / (4.0 * 4.0 * 1.0);
        assert!(approx(s, expected, 1e-12));
    }

    #[test]
    fn iterated_equilibrium_matches_closed_form_for_symmetric_costs() {
        let race =
            HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 3], 9.0).unwrap();
        let closed_form = race.symmetric_equilibrium_speed().unwrap();
        let speeds = race.equilibrium_speeds().unwrap();
        for s in &speeds {
            assert!(approx(*s, closed_form, 1e-6));
        }
    }

    #[test]
    fn cheaper_player_invests_more() {
        let race = HFTArmsRace::new(
            vec![HFT::new(1.0).unwrap(), HFT::new(2.0).unwrap()],
            10.0,
        )
        .unwrap();
        let speeds = race.equilibrium_speeds().unwrap();
        assert!(speeds[0] > speeds[1]);
    }

    #[test]
    fn socially_optimal_speed_is_zero() {
        let race =
            HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 3], 5.0).unwrap();
        assert_eq!(race.socially_optimal_speed(), 0.0);
    }

    #[test]
    fn rent_dissipation_grows_toward_full_value_with_n() {
        let race_2 =
            HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 2], 10.0).unwrap();
        let race_10 =
            HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 10], 10.0).unwrap();
        let dissipation_2 = race_2.deadweight_loss().unwrap();
        let dissipation_10 = race_10.deadweight_loss().unwrap();
        // Tullock dissipation = V * (n - 1) / n, monotone in n.
        assert!(dissipation_10 > dissipation_2);
        assert!(dissipation_2 < 10.0);
        assert!(dissipation_10 < 10.0);
    }

    #[test]
    fn equilibrium_profit_is_non_negative_for_symmetric_costs() {
        let race =
            HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 4], 8.0).unwrap();
        let speeds = race.equilibrium_speeds().unwrap();
        for i in 0..speeds.len() {
            let profit = race.expected_profit(&speeds, i).unwrap();
            // At the Tullock equilibrium each symmetric player earns V/n^2.
            assert!(profit > -1e-9);
            assert!(approx(profit, 8.0 / 16.0, 1e-6));
        }
    }

    #[test]
    fn rejects_too_few_participants() {
        let result = HFTArmsRace::new(vec![HFT::new(1.0).unwrap()], 5.0);
        assert!(matches!(result, Err(GameError::InvalidParameter(_))));
    }
}
