# Chapter 7, Simple Version: Game Theory and Auctions

## The Short Idea

Trading is like choosing a move while other people are choosing moves at the same time. A strategy that looks smart alone can become bad once everyone reacts to it. Game theory helps describe those reactions. Auction theory helps describe rules for choosing winners, prices, and incentives.

## Everyday Analogies

### Zero-Sum Game

Imagine two people splitting a fixed prize. Every extra dollar one person gets is a dollar the other person loses. Some trading situations feel like this: if one trader buys before a price move, another trader may sell too cheaply.

The important question is not "what do I want?" It is "what will the other side do after seeing what I usually do?"

### Nash Equilibrium

Think about two drivers choosing lanes in traffic. If both know the traffic pattern and neither can improve by switching lanes alone, they are in an equilibrium. It may still be slow for everyone, but no single driver has an easy solo fix.

In markets, a Nash equilibrium means each participant's strategy makes sense given the others' strategies.

### Stackelberg Game

This is like a large moving truck turning first while smaller cars adjust around it. The truck is the leader because it commits to a path. The cars are followers because they react.

A large fund executing a big order can be the truck. If its schedule is predictable, faster traders and market makers can react to it.

### First-Price Auction

In a first-price auction, you win by bidding the most and you pay your own bid. If you bid exactly what the item is worth to you, winning gives no profit. So bidders usually shade their bids below value.

In simple form:

$$
b(v) = \frac{n-1}{n}v.
$$

### Second-Price Auction

In a second-price auction, the highest bidder wins but pays the second-highest bid. The best simple strategy is to bid your true value:

$$
b(v) = v.
$$

If you bid too high, you can win something you should not want. If you bid too low, you can lose something profitable.

### HFT Speed Race

Imagine several people paying for faster shoes to press the same button first. Everyone pays for the shoes, but only one person wins each race. This is why latency competition can waste money: the private winner benefits, but the group may spend too much.

Batch auctions change the game by saying: "Everyone who arrives in this short time window competes on price, not on the exact microsecond."

### Colonel Blotto

Imagine a coach distributing players across several fields. Putting everyone on one field wins that field but loses the rest. Liquidity allocation has the same shape: a trader must decide how much order flow to place across venues, assets, or time buckets.

## How the Rust Code Maps to the Ideas

- `zero_sum_games.rs`: fixed-prize strategic games and mixed strategies.
- `stackelberg.rs`: leader-follower commitment.
- `auctions.rs`: first-price, second-price, Dutch, English, and package-style auction examples.
- `kyle_model.rs`: how market makers infer information from order flow.
- `hft_arms_race.rs`: private speed investment versus social waste.
- `colonel_blotto.rs`: spreading liquidity across competing venues.
- `all_pay_auction.rs`: everyone pays to compete, even losers.

## What to Remember

1. Trading strategies interact.
2. Predictable execution can invite strategic response.
3. Auction rules decide whether traders compete on price, speed, or information.
4. A market can be in equilibrium and still waste resources.
5. Good backtests should simulate other agents, not just historical prices.
