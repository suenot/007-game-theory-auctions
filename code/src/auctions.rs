use crate::{ModelError, Result};

/// Auction mechanisms used in market design and execution examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuctionType {
    FirstPrice,
    SecondPrice,
    Dutch,
    English,
    Combinatorial,
}

/// A bidder with a private valuation and submitted bid.
#[derive(Debug, Clone, PartialEq)]
pub struct Participant {
    pub id: String,
    pub valuation: f64,
    pub bid: f64,
}

impl Participant {
    pub fn new(id: impl Into<String>, valuation: f64, bid: f64) -> Self {
        Self {
            id: id.into(),
            valuation,
            bid,
        }
    }
}

/// Beliefs needed by closed-form optimal bidding examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeliefDistribution {
    bidders: usize,
}

impl BeliefDistribution {
    pub fn new(bidders: usize) -> Result<Self> {
        if bidders == 0 {
            return Err(ModelError::EmptyParticipants);
        }
        Ok(Self { bidders })
    }

    pub fn bidders(&self) -> usize {
        self.bidders
    }
}

/// Single-lot auction configured with a reserve price.
#[derive(Debug, Clone, PartialEq)]
pub struct Auction {
    pub auction_type: AuctionType,
    pub participants: Vec<Participant>,
    pub reserve_price: f64,
}

/// Outcome returned after running an auction.
#[derive(Debug, Clone, PartialEq)]
pub struct AuctionResult {
    pub winner: Option<String>,
    pub clearing_price: f64,
    pub revenue: f64,
    pub winning_bid: Option<f64>,
}

impl Auction {
    pub fn new(
        auction_type: AuctionType,
        participants: Vec<Participant>,
        reserve_price: f64,
    ) -> Result<Self> {
        if participants.is_empty() {
            return Err(ModelError::EmptyParticipants);
        }
        if !reserve_price.is_finite() || reserve_price < 0.0 {
            return Err(ModelError::NonFiniteValue);
        }
        if participants
            .iter()
            .any(|participant| !participant.valuation.is_finite() || !participant.bid.is_finite())
        {
            return Err(ModelError::NonFiniteValue);
        }

        Ok(Self {
            auction_type,
            participants,
            reserve_price,
        })
    }

    /// Runs the auction and returns a deterministic, tie-stable outcome.
    pub fn run(&mut self) -> Result<AuctionResult> {
        let mut ranked: Vec<&Participant> = self
            .participants
            .iter()
            .filter(|participant| participant.bid >= self.reserve_price)
            .collect();

        ranked.sort_by(|left, right| {
            right
                .bid
                .partial_cmp(&left.bid)
                .unwrap()
                .then_with(|| left.id.cmp(&right.id))
        });

        let Some(winner) = ranked.first() else {
            return Ok(AuctionResult {
                winner: None,
                clearing_price: 0.0,
                revenue: 0.0,
                winning_bid: None,
            });
        };

        let second_bid = ranked.get(1).map_or(self.reserve_price, |participant| {
            participant.bid.max(self.reserve_price)
        });
        let clearing_price = match self.auction_type {
            AuctionType::FirstPrice | AuctionType::Dutch | AuctionType::Combinatorial => winner.bid,
            AuctionType::SecondPrice => second_bid,
            AuctionType::English => second_bid.min(winner.bid),
        };

        Ok(AuctionResult {
            winner: Some(winner.id.clone()),
            clearing_price,
            revenue: clearing_price,
            winning_bid: Some(winner.bid),
        })
    }

    /// Closed-form optimal bid under independent uniform private values.
    pub fn optimal_bid(&self, valuation: f64, beliefs: &BeliefDistribution) -> f64 {
        if !valuation.is_finite() || valuation < self.reserve_price {
            return 0.0;
        }

        match self.auction_type {
            AuctionType::SecondPrice | AuctionType::English => valuation,
            AuctionType::FirstPrice | AuctionType::Dutch | AuctionType::Combinatorial => {
                if beliefs.bidders <= 1 {
                    valuation
                } else {
                    valuation * (beliefs.bidders as f64 - 1.0) / beliefs.bidders as f64
                }
            }
        }
    }

    /// Expected revenue proxy for the current participant set.
    pub fn expected_revenue(&self) -> Result<f64> {
        let mut cloned = self.clone();
        Ok(cloned.run()?.revenue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_price_winner_pays_own_bid() {
        let mut auction = Auction::new(
            AuctionType::FirstPrice,
            vec![
                Participant::new("a", 10.0, 7.0),
                Participant::new("b", 11.0, 8.0),
            ],
            1.0,
        )
        .unwrap();

        let result = auction.run().unwrap();

        assert_eq!(result.winner.as_deref(), Some("b"));
        assert_eq!(result.clearing_price, 8.0);
    }

    #[test]
    fn reserve_blocks_low_bids() {
        let mut auction = Auction::new(
            AuctionType::SecondPrice,
            vec![Participant::new("a", 2.0, 1.0)],
            3.0,
        )
        .unwrap();

        assert_eq!(auction.run().unwrap().winner, None);
    }
}
