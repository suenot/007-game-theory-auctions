# Chapter 7. Game Theory and Auctions in Algorithmic Trading

## Metadata

- Difficulty: advanced.
- Prerequisites: chapters 1-6. The chapter assumes stochastic calculus for uncertainty, microstructure for order-book mechanics, portfolio optimization for objective functions, machine learning for strategy estimation, low-latency systems for speed constraints, and information theory for adverse selection.
- Main implementation language: Rust.
- Companion implementation: `code/`, especially the reusable library modules and `code/examples/synthetic_market.rs`.

## 7.0 Why This Chapter Matters

Algorithmic trading is not only a problem of prediction. Every submitted order changes the opportunity set of other traders. A large buy program can reveal information, a market maker can shade quotes when toxic flow is expected, and a high-frequency firm can spend capital on speed even when the social value of the extra microsecond is close to zero. Game theory is the language for these strategic feedback loops.

An execution algorithm from chapter 3 might start with a single-agent objective:

$$
\min_x \mathbb{E}[C(x)] + \lambda \operatorname{Var}(C(x)).
$$

In a strategic market the cost function depends on other agents' responses:

$$
C_i = C_i(x_i, x_{-i}, s, \theta),
$$

where $x_i$ is our action, $x_{-i}$ are other actions, $s$ is the market state, and $\theta$ contains latent information such as inventory pressure or private value. The modeling shift is simple but important: an optimal action is no longer evaluated in isolation. It must be a best response to a predicted response.

The chapter uses four recurring questions:

1. What is each participant optimizing?
2. What information does each participant observe?
3. What commitment or timing advantage exists?
4. What market design turns private incentives into good or bad welfare outcomes?

## 7.1 Game Theory for Traders

### 7.1.1 Normal-Form and Zero-Sum Games

A finite normal-form game is a set of players, a finite action set for each player, and a payoff function. For two players:

$$
G = (A_1, A_2, u_1, u_2),
$$

where $A_i$ is player $i$'s action set and $u_i(a_1, a_2)$ is the payoff. A zero-sum game satisfies:

$$
u_1(a_1, a_2) + u_2(a_1, a_2) = 0.
$$

Many trading interactions are not literally zero-sum after fees, risk transfer, and information production. Still, zero-sum models are useful when one participant's execution advantage appears as another participant's cost. A simplified aggressive/passive execution matrix can be written as:

|                 | Trader B aggressive | Trader B passive |
|-----------------|---------------------|------------------|
| Trader A aggressive | $(0, 0)$           | $(3, -3)$        |
| Trader A passive    | $(-3, 3)$          | $(1, 1)$         |

The row player's payoff matrix is:

$$
A =
\begin{bmatrix}
0 & 3 \\
-3 & 1
\end{bmatrix}.
$$

A mixed strategy for the row player is a probability vector $p$ over rows. A mixed strategy for the column player is a probability vector $q$ over columns. The row player's expected payoff is:

$$
V(p, q) = p^\top A q.
$$

The minimax equilibrium solves:

$$
\max_p \min_q p^\top A q
=
\min_q \max_p p^\top A q.
$$

The Rust module `zero_sum_games.rs` implements this representation. Two-by-two games are solved analytically, while larger matrices use deterministic fictitious play for educational simulations.

### 7.1.2 Nash Equilibrium

A strategy profile $s^\*$ is a Nash equilibrium when no player can improve by deviating alone:

$$
u_i(s_i^\*, s_{-i}^\*) \ge u_i(s_i, s_{-i}^\*) \quad \forall i,\forall s_i.
$$

In markets, Nash equilibrium is a consistency condition. It says that once market makers, liquidity takers, and speed-sensitive traders all understand the strategy profile, no single participant wants to change only its own action. This does not mean the equilibrium is fair, efficient, or stable under rule changes.

The matching-pennies payoff matrix:

$$
A =
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}
$$

has no pure equilibrium. The mixed equilibrium is:

$$
p^\* = q^\* = (1/2, 1/2), \quad V = 0.
$$

This is implemented and tested in `code/tests/issue_acceptance.rs`.

### 7.1.3 Dominance and Best Responses

A row action $r$ weakly dominates row action $r'$ if:

$$
A_{r,c} \ge A_{r',c} \quad \forall c,
$$

with strict inequality for at least one column. Dominance is useful in execution systems because dominated actions can be removed before simulation. If a child-order schedule is worse in every liquidity state, it should not remain in the action set merely because the optimizer can technically choose it.

Best-response maps are more operational:

$$
BR_i(s_{-i}) = \arg\max_{s_i} u_i(s_i, s_{-i}).
$$

They describe how an adaptive agent reacts to observed behavior. A market maker's best response to toxic flow may be wider spreads; a broker's best response to wider spreads may be more patient execution; an HFT firm's best response to batch auctions may be price competition instead of speed competition.

### 7.1.4 Stackelberg Games

Stackelberg games model sequential commitment. A leader moves first, a follower observes or infers the commitment, and the follower best-responds. If the leader has actions $a_L$ and the follower has actions $a_F$, the leader solves:

$$
\max_{a_L} u_L(a_L, BR_F(a_L)).
$$

In trading, a large institutional trader can become the leader when its schedule is predictable. HFT and market-making agents become followers that condition on the footprint. The leader's problem is not "what schedule is cheap against a static market?" but:

$$
\min_x C_L(x, BR_F(x)).
$$

The `stackelberg.rs` implementation accepts separate payoff matrices for leader and follower. It returns the leader's optimal pure commitment and the follower's best response to a mixed leader strategy.

### 7.1.5 Repeated Games

Repeated interaction changes incentives. A one-shot market-making game may reward sniping stale quotes. A repeated venue relationship can reward quoting reliability if participants can condition future behavior on past conduct.

Let $\delta \in (0, 1)$ be the discount factor. A repeated payoff can be written as:

$$
U_i = \sum_{t=0}^{\infty} \delta^t u_i(a_t).
$$

Cooperation can be sustained when the present value of future punishment exceeds the one-shot gain from deviation:

$$
G_{\text{deviate}} \le
\frac{\delta}{1-\delta}(U_{\text{cooperate}} - U_{\text{punish}}).
$$

In market terms, a small immediate advantage from toxic routing may be unattractive if it causes future counterparties to widen spreads, reduce fill probability, or avoid the venue.

## 7.2 Auctions in Financial Markets

### 7.2.1 First-Price Sealed-Bid Auctions

In a first-price auction, the highest bidder wins and pays its own bid. With $n$ risk-neutral bidders and independent private values uniformly distributed on $[0, 1]$, the symmetric equilibrium bid is:

$$
b(v) = \frac{n-1}{n}v.
$$

The bid is shaded below value because winning at value leaves no surplus. In execution, this appears when a trader improves price just enough to win queue priority or access scarce liquidity while preserving expected edge.

### 7.2.2 Second-Price Auctions

In a second-price sealed-bid auction, the highest bidder wins and pays the second-highest bid. The dominant strategy is truthful bidding:

$$
b(v) = v.
$$

Truthfulness follows because a bidder's bid determines whether it wins, but not the price paid conditional on winning, except at the threshold. Overbidding can create negative surplus; underbidding can lose positive-surplus opportunities.

The `auctions.rs` module models this directly. The integration test verifies that a bidder with bid $11$ wins and pays the second-highest eligible bid $9$.

### 7.2.3 English, Dutch, Double, and Combinatorial Auctions

An English auction raises price until only one bidder remains. Under private values and standard assumptions, it is strategically close to second-price bidding. A Dutch auction lowers price until a bidder accepts; it is strategically close to first-price bidding because the winner controls the accepted price.

Financial markets also use double auctions, where buyers and sellers submit demand and supply. Continuous limit order books are continuous double auctions. Opening and closing auctions are call auctions: orders accumulate, then a clearing price maximizes executable volume, often with imbalance and price-continuity rules.

Combinatorial auctions allow package bids. In portfolio trading, a participant may value a basket differently from the sum of individual legs because the residual risk of partial fills is costly. A package bid can be represented as:

$$
b(S) \ne \sum_{i \in S} b_i,
$$

where $S$ is a set of instruments.

### 7.2.4 VCG and Mechanism Design

The Vickrey-Clarke-Groves mechanism charges each winner the externality imposed on others. For allocation $x^\*$ and participant $i$, a VCG payment can be written as:

$$
p_i =
\max_{x} \sum_{j \ne i} v_j(x_j)
-
\sum_{j \ne i} v_j(x_j^\*).
$$

The mechanism is useful as a benchmark because truthful reporting is incentive-compatible under quasilinear preferences. It is not a drop-in replacement for every trading venue because budgets, latency, collusion, and multi-dimensional preferences complicate implementation.

### 7.2.5 Revenue Equivalence and Its Limits

Under standard assumptions, common auction formats generate the same expected revenue. The important assumptions include risk neutrality, independent private values, symmetric bidders, and efficient allocation. Trading environments often violate each assumption. Market makers may be inventory constrained, values may be affiliated through common signals, and bidders may face latency asymmetry.

This matters for design. If values are affiliated, revealing more information before an auction can increase revenue and reduce adverse selection. If bidders are asymmetric, first-price shading can favor stronger or more informed participants.

## 7.3 Strategic Interaction in Markets

### 7.3.1 Market Making as a Game

A market maker chooses bid and ask quotes while facing informed and uninformed flow. A simplified spread objective is:

$$
\max_s \; \mathbb{E}[\text{spread capture}(s)]
- \mathbb{E}[\text{adverse selection}(s)]
- \mathbb{E}[\text{inventory cost}(s)].
$$

The best response to higher informed-flow probability is usually wider spreads or lower displayed size. The best response of liquidity takers to wider spreads is either delay, venue substitution, or hidden liquidity search. The resulting equilibrium connects chapter 2 microstructure with chapter 4 prediction: better toxicity estimates change the game.

### 7.3.2 Payment for Order Flow and Routing

Payment for order flow can be modeled as a game among brokers, wholesalers, exchanges, and customers. The broker's routing objective may include price improvement, execution quality, rebates, and contractual payments:

$$
U_{\text{broker}} =
\alpha \cdot Q_{\text{execution}}
+ \beta \cdot R_{\text{rebate}}
- \gamma \cdot C_{\text{customer harm}}.
$$

The policy question is whether private broker incentives align with customer welfare. A mechanism can look efficient for the broker and still transfer surplus away from uninformed customers if measurement is weak.

### 7.3.3 HFT Arms Race

Speed investment often has all-pay-auction structure. Every participant pays for latency reduction, but only the fastest participant captures a given race. If bidder value is $v$ and normalized private value is drawn from $[0,1]$, the all-pay bid curve used in the Rust example is:

$$
b(v) = \frac{n-1}{n}v^n.
$$

Private incentives can produce overinvestment:

$$
\sum_i c_i(s_i^{\text{private}}) >
c(s^{\text{social}}).
$$

Frequent batch auctions change the payoff function. Instead of rewarding the first message by microsecond priority, they group orders over an interval $\tau$ and clear by price. A latency advantage $\delta$ matters only on a fraction approximately $\delta/\tau$ of event times, which sharply reduces the value of tiny speed improvements.

The `hft_arms_race.rs` and `all_pay_auction.rs` modules provide simple simulations for this incentive pattern.

### 7.3.4 Kyle Model

The Kyle model links informed trading, noise trading, and market-maker pricing. Let $v$ be asset value, $u$ be noise-trader order flow, and $x$ be informed-trader order flow. Total order flow is:

$$
y = x + u.
$$

The market maker sets price:

$$
p = \lambda y.
$$

In the single-period normal model:

$$
\lambda = \frac{\sigma_v}{2\sigma_u},
\quad
x(v) = \frac{v}{2\lambda}.
$$

This makes price impact endogenous. Impact is not only a mechanical cost from walking the book; it is the market maker's rational response to inference from order flow. The Rust `KyleModel` encodes these equilibrium relationships and tests the link between $\lambda$ and informed order size.

## 7.4 Practical Scenarios

### 7.4.1 Colonel Blotto for Liquidity Allocation

Colonel Blotto games model budget allocation across battlefields. A trader can treat venues, time buckets, or correlated assets as battlefields. If trader $i$ allocates $x_{ik}$ to battlefield $k$, then:

$$
\sum_{k=1}^{K} x_{ik} \le B_i.
$$

A simple winner-take-all payoff is:

$$
u_i(x_i, x_j) =
\sum_{k=1}^{K}
\mathbf{1}\{x_{ik} > x_{jk}\}
+ \frac{1}{2}\mathbf{1}\{x_{ik} = x_{jk}\}.
$$

For liquidity allocation, the lesson is diversification under competition. Concentrating all liquidity in one venue can win that venue but leave other venues uncontested. Equal allocation is not always optimal, but it is a transparent symmetric benchmark.

### 7.4.2 All-Pay Auction for Latency

Latency races are all-pay-like because infrastructure costs are paid regardless of whether the next race is won. A strategy team should measure both private alpha and social waste:

$$
\text{private ROI} =
\frac{\mathbb{E}[\text{race profit}] - \text{speed cost}}
{\text{speed cost}},
$$

while a regulator or venue designer asks:

$$
\text{welfare} =
\text{investor surplus} + \text{liquidity provider surplus}
- \text{duplicated speed expenditure}.
$$

The same investment can be privately rational and socially wasteful.

### 7.4.3 Double Auctions and Walrasian Clearing

A double auction aggregates buy and sell interest. If demand $D(p)$ and supply $S(p)$ are continuous, the Walrasian clearing price solves:

$$
D(p^\*) = S(p^\*).
$$

In discrete order books, the clearing rule usually maximizes executable volume and then applies tie-breakers. Opening and closing auctions use this logic to concentrate liquidity and reduce noise from serial order arrival.

### 7.4.4 Optimal Execution Under Strategic Response

A non-strategic execution model may choose a schedule $x_t$ from expected volume and volatility. A strategic model includes the response:

$$
\min_{x_t}
\mathbb{E}\left[
\sum_t P_t x_t + I_t(x_t, BR_t(x_t))
\right]
+ \lambda \operatorname{Var}(C).
$$

The practical workflow is:

1. Estimate the passive market response from historical microstructure data.
2. Simulate adversarial response under stress assumptions.
3. Restrict strategies that are dominated across normal and stress regimes.
4. Use randomized or mixed schedules when predictability creates a measurable cost.

This connects chapter 5 low-latency constraints with chapter 6 information leakage: a fast system can still lose if it is too predictable.

## 7.5 Simulations and Backtesting

### 7.5.1 Monte Carlo Auction Simulation

A Monte Carlo auction simulator samples private values, applies a bidding rule, runs the mechanism, and records allocation, surplus, and revenue:

$$
\hat{R} = \frac{1}{M}\sum_{m=1}^{M} R^{(m)}.
$$

Core metrics include expected seller revenue, bidder surplus, allocative efficiency, reserve-price sensitivity, and tail outcomes. For execution, replace seller revenue with implementation shortfall and fill risk.

### 7.5.2 Evolutionary Strategy Dynamics

Repeated market interaction can be simulated with strategy weights:

$$
w_{i,t+1} = w_{i,t}\exp(\eta u_{i,t}),
$$

followed by normalization. Profitable strategies receive more capital or more order flow. This is a practical way to test whether a market design encourages liquidity provision, toxic flow, or excessive cancellation.

### 7.5.3 Agent-Based Market Model

A minimal agent-based model for this chapter includes:

- Market makers with inventory-sensitive spreads.
- Informed traders with private signals.
- Noise traders with exogenous demand.
- HFT agents with latency and sniping logic.
- A venue rule: continuous order book, call auction, or frequent batch auction.

The model should report spreads, depth, realized adverse selection, welfare, and duplicated speed spending. These outputs are more useful than raw PnL alone because they show whether a rule change moves surplus or creates surplus.

### 7.5.4 Validation Checklist

Before using a strategic simulator for research or production, check:

- Conservation: every transfer has a source and destination.
- Incentives: agents optimize the payoff actually coded.
- Equilibrium plausibility: obvious dominated strategies disappear.
- Sensitivity: conclusions do not depend on a single arbitrary parameter.
- Reproducibility: seeds, parameters, and outputs are versioned.
- Risk: tail losses and manipulation cases are tested separately from average behavior.

## Implementation Guide

Run the Rust checks:

```bash
cd code
cargo test
cargo run
cargo run --example synthetic_market
cargo bench --bench auction_benchmark
```

The crate is intentionally small and auditable:

- `zero_sum_games.rs`: payoff matrices, dominant strategies, pure equilibria, mixed equilibrium for two-by-two games, and fictitious play for larger games.
- `stackelberg.rs`: leader commitment and follower best response.
- `auctions.rs`: first-price, second-price, Dutch, English, and combinatorial placeholders for single-lot simulations.
- `kyle_model.rs`: equilibrium price impact and informed order size.
- `hft_arms_race.rs`: private speed investment and deadweight speed loss.
- `colonel_blotto.rs`: liquidity allocation across venues or assets.
- `all_pay_auction.rs`: latency-race all-pay bidding.

## Key Takeaways

1. Prediction is not enough in trading. Other agents react.
2. Nash equilibrium is a consistency condition, not a welfare guarantee.
3. Stackelberg models are natural for predictable institutional execution.
4. Auction design changes incentives by changing what agents compete on: price, speed, information, or package value.
5. HFT speed races can be privately rational and socially wasteful at the same time.
6. Batch auctions reduce the value of tiny latency advantages by discretizing time.
7. Strategic backtests should report welfare, adverse selection, and robustness, not just PnL.

## References

- John Nash, "Equilibrium Points in N-Person Games", PNAS, 1950: https://pmc.ncbi.nlm.nih.gov/articles/PMC1063129/
- William Vickrey, "Counterspeculation, Auctions, and Competitive Sealed Tenders", Journal of Finance, 1961: https://ideas.repec.org/a/bla/jfinan/v16y1961i1p8-37.html
- Roger Myerson, "Optimal Auction Design", Mathematics of Operations Research, 1981: https://pubsonline.informs.org/doi/abs/10.1287/moor.6.1.58
- Paul Milgrom and Robert Weber, "A Theory of Auctions and Competitive Bidding", Econometrica, 1982: https://www.scholars.northwestern.edu/en/publications/a-theory-of-auctions-and-competitive-bidding
- Eric Budish, Peter Cramton, and John Shim, "The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response", Quarterly Journal of Economics, 2015: https://academic.oup.com/qje/article/130/4/1547/1916146
- Songzi Du and Haoxiang Zhu, "What is the Optimal Trading Frequency in Financial Markets?", Review of Economic Studies, 2017: https://academic.oup.com/restud/article-pdf/84/4/1606/20386520/rdx006.pdf
- Thierry Foucault, Marco Pagano, and Ailsa Roell, "Market Liquidity: Theory, Evidence, and Policy", second edition, Oxford University Press, 2023: https://academic.oup.com/book/55158
