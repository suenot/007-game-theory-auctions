//! Monte-Carlo comparison of first-price vs. second-price auction revenue.
//!
//! Bidders draw private valuations from a uniform `[0, 1]` distribution and
//! bid according to the symmetric Bayesian Nash equilibrium of each
//! auction format. Empirical revenue is averaged over many independent
//! auctions and compared against the closed-form revenue equivalence
//! prediction `(n - 1) / (n + 1)`.

use game_theory_auctions::auctions::{Auction, AuctionType, BeliefDistribution, Participant};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let n_bidders = 5usize;
    let n_runs = 50_000;
    let beliefs = BeliefDistribution::uniform(n_bidders, 1.0).unwrap();
    let mut rng = StdRng::seed_from_u64(20260509);

    let mut first_revenue = 0.0;
    let mut second_revenue = 0.0;
    for _ in 0..n_runs {
        let valuations: Vec<f64> = (0..n_bidders).map(|_| rng.gen_range(0.0..1.0)).collect();

        // First-price equilibrium bid: (n - 1) / n * v.
        let first_bids: Vec<f64> = valuations
            .iter()
            .map(|&v| (n_bidders as f64 - 1.0) / n_bidders as f64 * v)
            .collect();
        let first_participants: Vec<Participant> = valuations
            .iter()
            .zip(first_bids.iter())
            .enumerate()
            .map(|(i, (v, b))| Participant::new(format!("p{i}"), *v, *b))
            .collect();
        let first = Auction::new(AuctionType::FirstPrice, first_participants, 0.0).unwrap();
        first_revenue += first.run().unwrap().revenue;

        // Second-price equilibrium bid: truthful.
        let second_participants: Vec<Participant> = valuations
            .iter()
            .enumerate()
            .map(|(i, &v)| Participant::new(format!("p{i}"), v, v))
            .collect();
        let second = Auction::new(AuctionType::SecondPrice, second_participants, 0.0).unwrap();
        second_revenue += second.run().unwrap().revenue;
    }

    let theoretical = (n_bidders as f64 - 1.0) / (n_bidders as f64 + 1.0);
    println!("Number of bidders: {n_bidders}");
    println!("Runs: {n_runs}");
    println!("Mean first-price revenue:  {:.4}", first_revenue / n_runs as f64);
    println!("Mean second-price revenue: {:.4}", second_revenue / n_runs as f64);
    println!("Theoretical (revenue equivalence): {theoretical:.4}");
    println!(
        "Closed-form expected revenue (auction module): {:.4}",
        Auction::new(AuctionType::FirstPrice, vec![Participant::new("a", 0.0, 0.0)], 0.0)
            .unwrap()
            .expected_revenue(&beliefs)
            .unwrap()
    );
}
