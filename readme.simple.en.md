# Chapter 7 — Game Theory and Auctions, the Simple Story

## What this chapter is about

If you trade in a market, you are not playing against the universe — you are
playing against other traders. A market maker, a hedge fund, a high-frequency
firm, and a retail investor all show up at the same order book with different
goals. Game theory is the math for thinking about that.

This chapter teaches you how to reason about other traders the way a chess
player reasons about an opponent: not just "what will the price do?" but
"what will *they* do once they see what I do?"

## Three real-life analogies

### 1. The lemonade-stand block (Stackelberg game)

You and your friend run lemonade stands on the same street. You set up first
and pick the busier corner. Your friend, seeing you, picks the other corner —
the best response to your move. By committing first, you got the better spot.

In trading: a big institution announces a VWAP execution at noon. A
high-frequency firm sees the announcement and prices the rest of the day
around it. The institution lost some flexibility, but locked in predictable
execution. That is a Stackelberg game.

### 2. The eBay auction (first vs. second price)

Two ways to sell a baseball card on eBay:

* **First price:** the highest bidder pays what they bid. So you should
  shade — bid below what the card is worth to you, hoping to still win.
* **Second price (Vickrey):** the highest bidder pays the *second-highest*
  bid. Your best move is to bid your true value — there's no reason to lie.

The amazing fact is that on average, both formats raise the same money for
the seller. That is the "revenue equivalence theorem". Open and close call
auctions on stock exchanges run on this idea.

### 3. The arms race (HFT speed)

Every shop on a busy street puts a sign in the window. If one shop makes its
sign brighter, the others follow. After everybody has invested in giant LED
displays, the customers haven't changed — but each shop now pays a huge
electricity bill. Most of the marketing money has been *wasted* against the
other shops.

That is an HFT arms race. Each firm spends millions on faster cables and
microwave links. The market does not move faster as a result of any single
investment — only the *competitive standing* changes. The math says: with
enough firms, almost all of the prize ends up paid out as cost. Rent
dissipates.

## Topics in plain language

### Zero-sum games and "matching pennies"

A zero-sum game is one where someone's gain is exactly somebody else's loss.
If a game has no winning *fixed* strategy (like rock-paper-scissors), the
right answer is to randomize. Markets full of pattern-spotters reach the same
conclusion: deterministic strategies get exploited, so smart algorithms
randomize their child orders.

### The Stackelberg leader

When you can credibly *commit* to a plan before others act, you can pick the
best plan among everyone's best responses. Trading example: pre-announced
TWAP schedules trade off transparency for predictable execution.

### Auctions everywhere

Stock exchanges open and close with auctions. Treasuries are sold by auction.
Ad slots online are auctions. Even the limit-order book is a continuous
double auction. Knowing the rules of these auctions, and knowing how others
are bidding, is most of the job for execution algorithms.

### Kyle's model

If you know something the market does not, how aggressively should you
trade? Albert Kyle answered this in 1985: *trade at an intensity proportional
to the amount of noise in the market*. Trade harder than that, and you give
yourself away. Trade lighter, and you leave money on the table. Half the
volume is informed and half is noise — that's the equilibrium.

### Colonel Blotto and where to put your liquidity

You and a competitor each have $1M of capital and need to spread it across,
say, 10 trading pairs. Where do you put it? If you concentrate, the
competitor diversifies and beats you on the other 9. If you diversify, the
competitor concentrates and beats you on a few but loses the rest. The right
answer is *random*: spread randomly with a uniform distribution and trust
the average.

### All-pay auctions and lobbying

In an all-pay auction, you pay your bid even if you lose. Lobbying contests,
patent races, and HFT speed investment all look like this. The math says you
should bid less than your value, but pay (in expectation) almost the entire
prize. Society pays the price of the contest.

## What's in the code

The companion crate `code/` has working Rust implementations of:

* zero-sum game solvers (multiplicative-weights learning),
* Stackelberg games (backward induction),
* first-price, second-price, Dutch, English and combinatorial auctions,
* Kyle's model with closed-form trading intensity and price impact,
* HFT arms race as a Tullock contest with arbitrary cost asymmetry,
* Colonel Blotto with Monte Carlo verification,
* all-pay auctions, symmetric and asymmetric.

Everything has unit tests; benchmarks in `code/benches/` measure speed; an
example in `code/examples/auction_simulation.rs` runs 50 000 random
auctions and verifies revenue equivalence numerically.

## Why you should care

* If you're writing an execution algorithm, the price impact and the
  optimal bid in this chapter are not theory — they are first-line tools.
* If you're designing a market or a token sale, the auction-design results
  here tell you which formats raise the most revenue and which are robust
  to manipulation.
* If you're building HFT infrastructure, the arms-race math tells you when
  the next \$10M cable is *not* worth it.

You don't have to be a game theorist to use any of this. But once you start
thinking of the order book as a *game* and not a *signal*, your strategies
get more robust very quickly.
