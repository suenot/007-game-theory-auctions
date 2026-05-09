//! Demonstration CLI for the Chapter 7 implementations.

use game_theory_auctions::{
    auctions::{Auction, AuctionType, BeliefDistribution, Participant},
    AllPayAuction, ColonelBlotto, HFTArmsRace, KyleModel, StackelbergGame, ZeroSumGame,
};
use game_theory_auctions::hft_arms_race::HFT;

fn main() {
    println!("=== Chapter 7: Game Theory and Auctions ===\n");
    zero_sum_demo();
    stackelberg_demo();
    auction_demo();
    kyle_demo();
    hft_demo();
    blotto_demo();
    all_pay_demo();
}

fn zero_sum_demo() {
    println!("--- Zero-Sum Game (Matching Pennies) ---");
    let game = ZeroSumGame::from_rows(vec![vec![1.0, -1.0], vec![-1.0, 1.0]]).unwrap();
    let (p, q, value) = game.nash_equilibrium().unwrap();
    println!("Row strategy: {:?}", p);
    println!("Column strategy: {:?}", q);
    println!("Value of game: {:.4}\n", value);
}

fn stackelberg_demo() {
    println!("--- Stackelberg Game (Institutional Leader vs. HFT Follower) ---");
    let leader = vec![vec![5.0, 1.0], vec![0.0, 4.0]];
    let follower = vec![vec![0.0, 3.0], vec![4.0, 1.0]];
    let game = StackelbergGame::from_rows(leader, follower).unwrap();
    let leader_strategy = game.leader_optimal().unwrap();
    let (leader_value, follower_value) = game.equilibrium_payoffs().unwrap();
    println!("Leader strategy: {:?}", leader_strategy);
    println!("Leader payoff: {:.4}", leader_value);
    println!("Follower payoff: {:.4}\n", follower_value);
}

fn auction_demo() {
    println!("--- Auctions ---");
    let participants = vec![
        Participant::new("A", 100.0, 90.0),
        Participant::new("B", 100.0, 80.0),
        Participant::new("C", 100.0, 70.0),
    ];
    let first = Auction::new(AuctionType::FirstPrice, participants.clone(), 0.0).unwrap();
    let second = Auction::new(AuctionType::SecondPrice, participants, 0.0).unwrap();
    println!("First-price result: {:?}", first.run().unwrap());
    println!("Second-price result: {:?}", second.run().unwrap());

    let beliefs = BeliefDistribution::uniform(5, 1.0).unwrap();
    println!(
        "Optimal first-price bid for v=0.6 against 5 bidders: {:.4}",
        first.optimal_bid(0.6, &beliefs).unwrap()
    );
    println!(
        "Expected revenue (uniform IID, n=5): {:.4}\n",
        first.expected_revenue(&beliefs).unwrap()
    );
}

fn kyle_demo() {
    println!("--- Kyle Model ---");
    let model = KyleModel::new(2.0, 1.0).unwrap();
    println!("beta = {:.4}", model.informed_intensity());
    println!("lambda = {:.4}", model.equilibrium_lambda());
    println!("Pricing rule for y=2: {:.4}", model.pricing_rule(2.0).unwrap());
    println!(
        "Expected informed profit: {:.4}\n",
        model.expected_informed_profit()
    );
}

fn hft_demo() {
    println!("--- HFT Arms Race (Tullock contest) ---");
    let race = HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 3], 9.0).unwrap();
    let symmetric = race.symmetric_equilibrium_speed().unwrap();
    let speeds = race.equilibrium_speeds().unwrap();
    println!("Closed-form symmetric speed: {:.4}", symmetric);
    println!("Iterated equilibrium speeds: {:?}", speeds);
    println!("Total rent dissipation: {:.4}\n", race.deadweight_loss().unwrap());
}

fn blotto_demo() {
    println!("--- Colonel Blotto ---");
    let game = ColonelBlotto::new(10.0, 10.0, 4).unwrap();
    let value = game.monte_carlo_value(2_000, 7).unwrap();
    println!("Monte-Carlo value (4 battlefields, equal budgets): {:.4}\n", value);
}

fn all_pay_demo() {
    println!("--- All-Pay Auction ---");
    let auction = AllPayAuction::new(vec![1.0, 0.7, 0.4]).unwrap();
    println!("Equilibrium bids: {:?}", auction.equilibrium_bids().unwrap());
    println!("Symmetric expected revenue: {:.4}", auction.symmetric_expected_revenue());
}
