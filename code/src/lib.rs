//! # Game Theory and Auctions for Algorithmic Trading
//!
//! Compact, well-documented implementations of the core quantitative tools
//! discussed in Chapter 7:
//!
//! - [`zero_sum_games`] - matrix games and mixed-strategy Nash equilibria
//! - [`stackelberg`] - leader/follower games for institutional vs. HFT
//! - [`auctions`] - first-price, second-price, Dutch, English, and combinatorial auctions
//! - [`kyle_model`] - informed-trader model with endogenous price impact
//! - [`hft_arms_race`] - all-pay auction for latency investments
//! - [`colonel_blotto`] - resource allocation across multiple battlefields
//! - [`all_pay_auction`] - all-pay auction with heterogeneous valuations
//!
//! ## Quick example
//!
//! ```rust
//! use game_theory_auctions::auctions::{Auction, AuctionType, Participant};
//!
//! let auction = Auction::new(
//!     AuctionType::SecondPrice,
//!     vec![
//!         Participant::new("A", 100.0, 90.0),
//!         Participant::new("B", 100.0, 80.0),
//!         Participant::new("C", 100.0, 70.0),
//!     ],
//!     0.0,
//! ).unwrap();
//! let result = auction.run().unwrap();
//! assert_eq!(result.winner.as_deref(), Some("A"));
//! assert!((result.price - 80.0).abs() < 1e-9);
//! ```

pub mod all_pay_auction;
pub mod auctions;
pub mod colonel_blotto;
pub mod hft_arms_race;
pub mod kyle_model;
pub mod stackelberg;
pub mod zero_sum_games;

pub use all_pay_auction::AllPayAuction;
pub use auctions::{Auction, AuctionResult, AuctionType, Participant};
pub use colonel_blotto::ColonelBlotto;
pub use hft_arms_race::HFTArmsRace;
pub use kyle_model::KyleModel;
pub use stackelberg::StackelbergGame;
pub use zero_sum_games::{Player, ZeroSumGame};

/// Crate-level result type.
pub type Result<T> = std::result::Result<T, GameError>;

/// Errors returned by numerical routines when input is malformed or the
/// requested computation is not well-defined.
#[derive(Debug, Clone, PartialEq)]
pub enum GameError {
    EmptyInput,
    InsufficientData(&'static str),
    DimensionMismatch { expected: usize, actual: usize },
    NonFiniteInput(&'static str),
    InvalidParameter(&'static str),
    NoEquilibrium,
    SingularMatrix,
}

impl std::fmt::Display for GameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input must not be empty"),
            Self::InsufficientData(msg) => write!(f, "insufficient data: {msg}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NonFiniteInput(name) => write!(f, "{name} contains NaN or infinite values"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::NoEquilibrium => write!(f, "no equilibrium could be computed"),
            Self::SingularMatrix => write!(f, "matrix is singular or ill-conditioned"),
        }
    }
}

impl std::error::Error for GameError {}

pub(crate) fn ensure_finite_value(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GameError::NonFiniteInput(name))
    }
}

pub(crate) fn ensure_finite_slice(name: &'static str, data: &[f64]) -> Result<()> {
    if data.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(GameError::NonFiniteInput(name))
    }
}

pub(crate) fn ensure_positive(name: &'static str, value: f64) -> Result<()> {
    ensure_finite_value(name, value)?;
    if value <= 0.0 {
        Err(GameError::InvalidParameter("must be strictly positive"))
    } else {
        Ok(())
    }
}

pub(crate) fn ensure_non_negative(name: &'static str, value: f64) -> Result<()> {
    ensure_finite_value(name, value)?;
    if value < 0.0 {
        Err(GameError::InvalidParameter("must be non-negative"))
    } else {
        Ok(())
    }
}
