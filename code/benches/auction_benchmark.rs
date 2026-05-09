//! Benchmarks for game-theoretic and auction primitives.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use game_theory_auctions::{
    auctions::{Auction, AuctionType, BeliefDistribution, Participant},
    hft_arms_race::HFT,
    AllPayAuction, ColonelBlotto, HFTArmsRace, KyleModel, ZeroSumGame,
};

fn benchmark_zero_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("nash_equilibrium");
    for n in [3usize, 5, 7] {
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push(
                (0..n)
                    .map(|j| ((i + 1) as f64).sin() - ((j + 1) as f64).cos())
                    .collect(),
            );
        }
        let game = ZeroSumGame::from_rows(rows).unwrap();
        group.bench_with_input(BenchmarkId::new("solve", n), &game, |b, game| {
            b.iter(|| game.nash_equilibrium().unwrap());
        });
    }
    group.finish();
}

fn benchmark_auctions(c: &mut Criterion) {
    let beliefs = BeliefDistribution::uniform(10, 1.0).unwrap();
    let auction = Auction::new(
        AuctionType::FirstPrice,
        (0..10)
            .map(|i| Participant::new(format!("p{i}"), 1.0, 0.5 + 0.05 * i as f64))
            .collect(),
        0.0,
    )
    .unwrap();
    c.bench_function("first_price_run", |b| b.iter(|| auction.run().unwrap()));
    c.bench_function("first_price_optimal_bid", |b| {
        b.iter(|| auction.optimal_bid(0.7, &beliefs).unwrap())
    });
}

fn benchmark_kyle(c: &mut Criterion) {
    let model = KyleModel::new(1.0, 2.0).unwrap();
    c.bench_function("kyle_pricing_rule", |b| {
        b.iter(|| model.pricing_rule(2.5).unwrap())
    });
}

fn benchmark_hft(c: &mut Criterion) {
    let race = HFTArmsRace::new(vec![HFT::new(1.0).unwrap(); 5], 10.0).unwrap();
    c.bench_function("hft_equilibrium_speeds", |b| {
        b.iter(|| race.equilibrium_speeds().unwrap())
    });
}

fn benchmark_blotto(c: &mut Criterion) {
    let game = ColonelBlotto::new(10.0, 10.0, 4).unwrap();
    c.bench_function("blotto_monte_carlo_500", |b| {
        b.iter(|| game.monte_carlo_value(500, 12).unwrap())
    });
}

fn benchmark_all_pay(c: &mut Criterion) {
    let auction = AllPayAuction::new(vec![0.3, 0.5, 0.7, 0.9]).unwrap();
    c.bench_function("all_pay_equilibrium_bids", |b| {
        b.iter(|| auction.equilibrium_bids().unwrap())
    });
}

criterion_group!(
    benches,
    benchmark_zero_sum,
    benchmark_auctions,
    benchmark_kyle,
    benchmark_hft,
    benchmark_blotto,
    benchmark_all_pay
);
criterion_main!(benches);
