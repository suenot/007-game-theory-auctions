use game_theory_auctions::auctions::{Auction, AuctionType, BeliefDistribution, Participant};
use game_theory_auctions::kyle_model::KyleModel;
use game_theory_auctions::zero_sum_games::ZeroSumGame;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let game = ZeroSumGame::from_rows(vec![vec![1.0, -1.0], vec![-1.0, 1.0]])?;
    println!(
        "matching-pennies row strategy: {:?}",
        game.nash_equilibrium()
    );
    println!("matching-pennies value: {:.4}", game.game_value());

    let mut auction = Auction::new(
        AuctionType::SecondPrice,
        vec![
            Participant::new("market-maker-a", 10.0, 9.0),
            Participant::new("hft-b", 12.0, 11.0),
            Participant::new("fund-c", 8.5, 8.5),
        ],
        5.0,
    )?;
    let result = auction.run()?;
    let truthful_bid = auction.optimal_bid(12.0, &BeliefDistribution::new(3)?);
    println!("auction result: {:?}", result);
    println!("second-price optimal bid for value 12: {truthful_bid:.2}");

    let kyle = KyleModel::new(2.0, 4.0)?;
    println!(
        "Kyle lambda={:.4}, order for signal 3={:.4}",
        kyle.equilibrium_lambda(),
        kyle.informed_strategy(3.0)
    );

    Ok(())
}
