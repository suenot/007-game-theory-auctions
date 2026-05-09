//! Auction mechanisms used in financial markets.
//!
//! Five auction formats are supported:
//!
//! - First-price sealed-bid: highest bidder wins and pays its own bid.
//! - Second-price sealed-bid (Vickrey): highest bidder wins and pays the
//!   second-highest bid.
//! - Dutch: clock descends from the seller's start price; first bidder to
//!   accept wins and pays the clock price at acceptance. We simulate the
//!   clock-strike with a small price grid.
//! - English: clock ascends; auction ends when only one bidder remains
//!   active. The price equals the second-highest valuation plus a small
//!   bid increment, identical (up to the increment) to a Vickrey outcome.
//! - Combinatorial: bidders submit XOR bids over bundles of items, and the
//!   auctioneer chooses the welfare-maximizing assignment by exhaustive
//!   search. This is exponential in the number of bundles, so the
//!   implementation is intended for small numbers of items (<=10) typical
//!   in textbook examples.
//!
//! For sealed-bid auctions we expose the closed-form optimal bid under the
//! standard private-value model with uniform `[0, 1]` valuations.

use crate::{
    ensure_finite_value, ensure_non_negative, ensure_positive, GameError, Result,
};

/// Auction mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuctionType {
    FirstPrice,
    SecondPrice,
    Dutch,
    English,
    Combinatorial,
}

/// Belief distribution about competing valuations. The chapter uses the
/// classic uniform `[0, 1]` benchmark, scaled by `upper`. We keep the API
/// minimal and only support uniform priors for the closed-form bidding
/// formulas; callers can simulate other distributions externally.
#[derive(Debug, Clone, Copy)]
pub struct BeliefDistribution {
    pub n: usize,
    pub upper: f64,
}

impl BeliefDistribution {
    pub fn uniform(n: usize, upper: f64) -> Result<Self> {
        if n < 2 {
            return Err(GameError::InvalidParameter("n must be >= 2"));
        }
        ensure_positive("upper", upper)?;
        Ok(Self { n, upper })
    }
}

/// One bidder.
#[derive(Debug, Clone)]
pub struct Participant {
    pub name: String,
    pub valuation: f64,
    pub bid: f64,
}

impl Participant {
    pub fn new(name: impl Into<String>, valuation: f64, bid: f64) -> Self {
        Self {
            name: name.into(),
            valuation,
            bid,
        }
    }
}

/// Result of running an auction.
#[derive(Debug, Clone)]
pub struct AuctionResult {
    pub winner: Option<String>,
    pub price: f64,
    pub revenue: f64,
}

/// Auction definition.
#[derive(Debug, Clone)]
pub struct Auction {
    pub auction_type: AuctionType,
    pub participants: Vec<Participant>,
    pub reserve_price: f64,
}

impl Auction {
    pub fn new(
        auction_type: AuctionType,
        participants: Vec<Participant>,
        reserve_price: f64,
    ) -> Result<Self> {
        if participants.is_empty() {
            return Err(GameError::EmptyInput);
        }
        ensure_non_negative("reserve_price", reserve_price)?;
        for p in &participants {
            ensure_non_negative("valuation", p.valuation)?;
            ensure_non_negative("bid", p.bid)?;
        }
        Ok(Self {
            auction_type,
            participants,
            reserve_price,
        })
    }

    /// Run the auction and produce the result.
    pub fn run(&self) -> Result<AuctionResult> {
        match self.auction_type {
            AuctionType::FirstPrice => Ok(self.run_first_price()),
            AuctionType::SecondPrice | AuctionType::English => Ok(self.run_second_price()),
            AuctionType::Dutch => Ok(self.run_dutch()),
            AuctionType::Combinatorial => Err(GameError::InvalidParameter(
                "use run_combinatorial for combinatorial auctions",
            )),
        }
    }

    fn run_first_price(&self) -> AuctionResult {
        let (winner_idx, winning_bid) = self.highest_bidder();
        if winning_bid < self.reserve_price {
            return AuctionResult {
                winner: None,
                price: 0.0,
                revenue: 0.0,
            };
        }
        AuctionResult {
            winner: Some(self.participants[winner_idx].name.clone()),
            price: winning_bid,
            revenue: winning_bid,
        }
    }

    fn run_second_price(&self) -> AuctionResult {
        let (winner_idx, winning_bid) = self.highest_bidder();
        if winning_bid < self.reserve_price {
            return AuctionResult {
                winner: None,
                price: 0.0,
                revenue: 0.0,
            };
        }
        let second = self
            .participants
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != winner_idx)
            .map(|(_, p)| p.bid)
            .fold(0.0_f64, f64::max);
        let price = second.max(self.reserve_price);
        AuctionResult {
            winner: Some(self.participants[winner_idx].name.clone()),
            price,
            revenue: price,
        }
    }

    fn run_dutch(&self) -> AuctionResult {
        // Strategically equivalent to first-price sealed-bid: the bidder
        // with the highest bid would jump first, paying its own bid.
        self.run_first_price()
    }

    fn highest_bidder(&self) -> (usize, f64) {
        let mut best = 0usize;
        let mut best_bid = self.participants[0].bid;
        for (i, p) in self.participants.iter().enumerate().skip(1) {
            if p.bid > best_bid {
                best_bid = p.bid;
                best = i;
            }
        }
        (best, best_bid)
    }

    /// Closed-form optimal sealed-bid for symmetric IID uniform priors.
    /// For first-price the equilibrium bid is `b(v) = (n-1)/n * v`. For
    /// second-price truthful bidding `b(v) = v` is dominant.
    pub fn optimal_bid(&self, valuation: f64, beliefs: &BeliefDistribution) -> Result<f64> {
        ensure_non_negative("valuation", valuation)?;
        if valuation > beliefs.upper {
            return Err(GameError::InvalidParameter(
                "valuation exceeds upper bound of belief distribution",
            ));
        }
        let n = beliefs.n as f64;
        match self.auction_type {
            AuctionType::FirstPrice | AuctionType::Dutch => Ok((n - 1.0) / n * valuation),
            AuctionType::SecondPrice | AuctionType::English => Ok(valuation),
            AuctionType::Combinatorial => Err(GameError::InvalidParameter(
                "no closed-form optimal bid for combinatorial auctions",
            )),
        }
    }

    /// Expected revenue under symmetric IID uniform priors. Both first- and
    /// second-price auctions yield the same expected revenue
    /// `(n-1)/(n+1) * upper` (revenue equivalence theorem).
    pub fn expected_revenue(&self, beliefs: &BeliefDistribution) -> Result<f64> {
        match self.auction_type {
            AuctionType::FirstPrice
            | AuctionType::SecondPrice
            | AuctionType::Dutch
            | AuctionType::English => {
                let n = beliefs.n as f64;
                Ok((n - 1.0) / (n + 1.0) * beliefs.upper)
            }
            AuctionType::Combinatorial => Err(GameError::InvalidParameter(
                "no closed-form expected revenue for combinatorial auctions",
            )),
        }
    }
}

/// One XOR bid in a combinatorial auction. The bidder is willing to pay
/// `price` for any one of `bundles`, and at most one bundle in total.
#[derive(Debug, Clone)]
pub struct CombinatorialBid {
    pub bidder: String,
    pub bundles: Vec<Vec<usize>>, // indices of items
    pub price: f64,
}

impl CombinatorialBid {
    pub fn new(bidder: impl Into<String>, bundles: Vec<Vec<usize>>, price: f64) -> Result<Self> {
        ensure_finite_value("price", price)?;
        Ok(Self {
            bidder: bidder.into(),
            bundles,
            price,
        })
    }
}

/// Combinatorial auction outcome.
#[derive(Debug, Clone)]
pub struct CombinatorialOutcome {
    /// `(bidder, bundle, price)` for each winning allocation.
    pub winners: Vec<(String, Vec<usize>, f64)>,
    pub total_revenue: f64,
}

/// Solve a small combinatorial XOR auction by exhaustive search over
/// bundle assignments. Each bidder takes at most one of its bundles, and
/// items cannot be allocated twice. Returns the welfare-maximizing
/// allocation.
pub fn run_combinatorial(num_items: usize, bids: &[CombinatorialBid]) -> Result<CombinatorialOutcome> {
    if bids.is_empty() {
        return Err(GameError::EmptyInput);
    }
    if num_items == 0 {
        return Err(GameError::InvalidParameter("num_items must be > 0"));
    }
    for bid in bids {
        for bundle in &bid.bundles {
            for &item in bundle {
                if item >= num_items {
                    return Err(GameError::InvalidParameter("item index out of range"));
                }
            }
        }
    }

    // Each bidder chooses either "no allocation" or one of its bundles.
    // Encode choices as base-(k+1) digits and check feasibility.
    let mut best_revenue = 0.0;
    let mut best_assignment: Vec<(String, Vec<usize>, f64)> = Vec::new();

    let max_choices: Vec<usize> = bids.iter().map(|bid| bid.bundles.len() + 1).collect();
    let total_assignments: usize = max_choices.iter().product();

    for assignment in 0..total_assignments {
        let mut remaining = assignment;
        let mut used = vec![false; num_items];
        let mut feasible = true;
        let mut winners: Vec<(String, Vec<usize>, f64)> = Vec::new();
        let mut revenue = 0.0;

        for (b, bid) in bids.iter().enumerate() {
            let choice = remaining % max_choices[b];
            remaining /= max_choices[b];
            if choice == 0 {
                continue;
            }
            let bundle = &bid.bundles[choice - 1];
            for &item in bundle {
                if used[item] {
                    feasible = false;
                    break;
                }
                used[item] = true;
            }
            if !feasible {
                break;
            }
            winners.push((bid.bidder.clone(), bundle.clone(), bid.price));
            revenue += bid.price;
        }

        if feasible && revenue > best_revenue {
            best_revenue = revenue;
            best_assignment = winners;
        }
    }

    Ok(CombinatorialOutcome {
        winners: best_assignment,
        total_revenue: best_revenue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_price_winner_pays_its_bid() {
        let auction = Auction::new(
            AuctionType::FirstPrice,
            vec![
                Participant::new("A", 100.0, 80.0),
                Participant::new("B", 100.0, 70.0),
                Participant::new("C", 100.0, 90.0),
            ],
            0.0,
        )
        .unwrap();
        let result = auction.run().unwrap();
        assert_eq!(result.winner.as_deref(), Some("C"));
        assert!((result.price - 90.0).abs() < 1e-12);
    }

    #[test]
    fn second_price_winner_pays_second_highest() {
        let auction = Auction::new(
            AuctionType::SecondPrice,
            vec![
                Participant::new("A", 100.0, 90.0),
                Participant::new("B", 100.0, 80.0),
                Participant::new("C", 100.0, 70.0),
            ],
            0.0,
        )
        .unwrap();
        let result = auction.run().unwrap();
        assert_eq!(result.winner.as_deref(), Some("A"));
        assert!((result.price - 80.0).abs() < 1e-12);
    }

    #[test]
    fn reserve_price_blocks_low_bids() {
        let auction = Auction::new(
            AuctionType::SecondPrice,
            vec![
                Participant::new("A", 10.0, 8.0),
                Participant::new("B", 10.0, 6.0),
            ],
            50.0,
        )
        .unwrap();
        let result = auction.run().unwrap();
        assert!(result.winner.is_none());
        assert_eq!(result.price, 0.0);
    }

    #[test]
    fn first_price_optimal_bid_matches_formula() {
        let auction = Auction::new(
            AuctionType::FirstPrice,
            vec![Participant::new("A", 0.5, 0.0)],
            0.0,
        )
        .unwrap();
        let beliefs = BeliefDistribution::uniform(4, 1.0).unwrap();
        let bid = auction.optimal_bid(0.5, &beliefs).unwrap();
        assert!((bid - 0.375).abs() < 1e-12);
    }

    #[test]
    fn second_price_truthful_bid() {
        let auction = Auction::new(
            AuctionType::SecondPrice,
            vec![Participant::new("A", 0.5, 0.0)],
            0.0,
        )
        .unwrap();
        let beliefs = BeliefDistribution::uniform(4, 1.0).unwrap();
        let bid = auction.optimal_bid(0.5, &beliefs).unwrap();
        assert!((bid - 0.5).abs() < 1e-12);
    }

    #[test]
    fn revenue_equivalence_holds_for_uniform_iid() {
        let beliefs = BeliefDistribution::uniform(5, 1.0).unwrap();
        let first = Auction::new(
            AuctionType::FirstPrice,
            vec![Participant::new("A", 0.0, 0.0)],
            0.0,
        )
        .unwrap();
        let second = Auction::new(
            AuctionType::SecondPrice,
            vec![Participant::new("A", 0.0, 0.0)],
            0.0,
        )
        .unwrap();
        let r1 = first.expected_revenue(&beliefs).unwrap();
        let r2 = second.expected_revenue(&beliefs).unwrap();
        assert!((r1 - r2).abs() < 1e-12);
        assert!((r1 - 4.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn combinatorial_picks_highest_welfare() {
        let bids = vec![
            CombinatorialBid::new("A", vec![vec![0]], 10.0).unwrap(),
            CombinatorialBid::new("B", vec![vec![1]], 8.0).unwrap(),
            CombinatorialBid::new("C", vec![vec![0, 1]], 15.0).unwrap(),
        ];
        let outcome = run_combinatorial(2, &bids).unwrap();
        // A + B = 18 > C alone = 15, so A and B win.
        assert!((outcome.total_revenue - 18.0).abs() < 1e-12);
        let names: Vec<&str> = outcome.winners.iter().map(|(name, _, _)| name.as_str()).collect();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
    }

    #[test]
    fn dutch_matches_first_price_outcome() {
        let dutch = Auction::new(
            AuctionType::Dutch,
            vec![
                Participant::new("A", 100.0, 80.0),
                Participant::new("B", 100.0, 90.0),
            ],
            0.0,
        )
        .unwrap();
        let result = dutch.run().unwrap();
        assert_eq!(result.winner.as_deref(), Some("B"));
        assert!((result.price - 90.0).abs() < 1e-12);
    }
}
