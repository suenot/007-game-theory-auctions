use criterion::{black_box, criterion_group, criterion_main, Criterion};
use game_theory_auctions::auctions::{Auction, AuctionType, BeliefDistribution, Participant};

fn auction_benchmark(c: &mut Criterion) {
    c.bench_function("second_price_auction_100_bidders", |b| {
        b.iter(|| {
            let participants = (0..100)
                .map(|index| {
                    let value = 50.0 + index as f64;
                    Participant::new(format!("bidder-{index}"), value, value * 0.95)
                })
                .collect();
            let mut auction = Auction::new(AuctionType::SecondPrice, participants, 10.0).unwrap();
            black_box(auction.run().unwrap());
        });
    });

    c.bench_function("first_price_optimal_bid", |b| {
        let auction = Auction::new(
            AuctionType::FirstPrice,
            vec![Participant::new("bidder", 100.0, 80.0)],
            0.0,
        )
        .unwrap();
        let beliefs = BeliefDistribution::new(20).unwrap();
        b.iter(|| black_box(auction.optimal_bid(100.0, &beliefs)));
    });
}

criterion_group!(benches, auction_benchmark);
criterion_main!(benches);
