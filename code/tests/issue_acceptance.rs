use game_theory_auctions::auctions::{Auction, AuctionType, BeliefDistribution, Participant};
use game_theory_auctions::kyle_model::KyleModel;
use game_theory_auctions::zero_sum_games::ZeroSumGame;

fn approx_eq(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
}

#[test]
fn matching_pennies_has_symmetric_mixed_equilibrium() {
    let game = ZeroSumGame::from_rows(vec![vec![1.0, -1.0], vec![-1.0, 1.0]]).unwrap();

    let row_strategy = game.nash_equilibrium();

    approx_eq(row_strategy[0], 0.5);
    approx_eq(row_strategy[1], 0.5);
    approx_eq(game.game_value(), 0.0);
}

#[test]
fn second_price_auction_charges_second_highest_bid() {
    let mut auction = Auction::new(
        AuctionType::SecondPrice,
        vec![
            Participant::new("slow-mm", 10.0, 8.0),
            Participant::new("fast-hft", 12.0, 11.0),
            Participant::new("patient-fund", 9.0, 9.0),
        ],
        5.0,
    )
    .unwrap();

    let result = auction.run().unwrap();

    assert_eq!(result.winner.as_deref(), Some("fast-hft"));
    approx_eq(result.clearing_price, 9.0);
    approx_eq(
        auction.optimal_bid(12.0, &BeliefDistribution::new(3).unwrap()),
        12.0,
    );
}

#[test]
fn kyle_equilibrium_links_lambda_and_informed_order_size() {
    let model = KyleModel::new(2.0, 4.0).unwrap();

    approx_eq(model.equilibrium_lambda(), 1.0);
    approx_eq(model.informed_strategy(3.0), 1.5);
    approx_eq(model.pricing_rule(2.5), 2.5);
}
