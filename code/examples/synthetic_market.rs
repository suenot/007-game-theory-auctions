use game_theory_auctions::all_pay_auction::AllPayAuction;
use game_theory_auctions::colonel_blotto::ColonelBlotto;
use game_theory_auctions::hft_arms_race::{HFTArmsRace, HFT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let latency = AllPayAuction::new(vec![0.6, 0.8, 1.0])?;
    println!("latency all-pay bids: {:?}", latency.equilibrium_bids());
    println!("latency auction revenue: {:.4}", latency.expected_revenue());

    let blotto = ColonelBlotto::new(2, 4, vec![1_000_000.0, 1_000_000.0])?;
    println!(
        "liquidity allocation benchmark: {:?}",
        blotto.symmetric_equilibrium()
    );

    let race = HFTArmsRace::new(
        vec![
            HFT::new("alpha", 10.0),
            HFT::new("beta", 9.0),
            HFT::new("gamma", 7.0),
        ],
        vec![1.0, 1.2, 1.5],
    )?;
    println!("speed investments: {:?}", race.equilibrium_speeds());
    println!("deadweight speed loss: {:.4}", race.deadweight_speed_loss());

    Ok(())
}
