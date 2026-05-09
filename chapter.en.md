# Chapter 7: Game Theory and Auctions in Algorithmic Trading

## Introduction

The previous six chapters built a stack: stochastic processes for prices,
order-book microstructure for execution, portfolio optimization for capital
allocation, machine learning for forecasts, low-latency systems for delivery,
and information theory together with Kelly sizing for the connection between
edge and bet size. Each of those tools assumes the trader is interacting with
"the market" as if the market were nature: a passive source of returns whose
distribution is fixed.

That assumption breaks as soon as another participant has a goal that conflicts
with ours. A market maker who quotes too tightly invites adverse selection.
A large institutional buyer who reveals intent invites front-running. A
high-frequency firm that invests in a faster cable does so only because
competitors are doing the same. Markets are not nature — they are populations
of strategic agents whose payoffs depend on what other agents do.

This chapter introduces the mathematical machinery for analyzing strategic
interaction on financial markets:

1. Game theory as the language for "I think that you think that I think".
2. Auctions as the canonical mechanism design problem and the engine of
   modern electronic markets.
3. Kyle's model and HFT arms races as concrete examples where strategic
   feedback shapes prices and welfare.
4. Practical simulation tools for evaluating market designs and execution
   algorithms when behaviour is endogenous.

The unifying message is that the right primitive in finance is not the
price process but the *equilibrium* of a game: a self-consistent set of
strategies in which no participant has a strict incentive to deviate.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $A$ | payoff matrix for the row player |
| $p, q$ | mixed strategies of the row and column players |
| $v$ | value of a zero-sum game |
| $b(v)$ | optimal bid given valuation $v$ |
| $n$ | number of bidders / players |
| $V$ | prize value in a contest |
| $c_i$ | speed cost of HFT $i$ |
| $\sigma_u, \sigma_v$ | volatilities of noise volume and asset value (Kyle) |
| $\lambda$ | Kyle's lambda (price impact) |
| $\beta$ | informed-trader trading intensity |
| $S$ | total speed in a contest, $\sum_i s_i$ |
| $W$ | welfare or expected revenue |

We write $\Delta(X)$ for the simplex over a finite set $X$, the set of
probability distributions over $X$.

---

## 7.1 Foundations of Game Theory for Traders

### 7.1.1 Zero-Sum Matrix Games

A two-player zero-sum game is a matrix $A \in \mathbb{R}^{m \times n}$. The row
player picks a strategy $p \in \Delta(\{1,\dots,m\})$ and the column player
picks $q \in \Delta(\{1,\dots,n\})$. The expected payoff to the row player is
$p^\top A q$; to the column player it is $-p^\top A q$.

Von Neumann's minimax theorem states that

$$
\max_{p \in \Delta} \min_{q \in \Delta} p^\top A q
= \min_{q \in \Delta} \max_{p \in \Delta} p^\top A q
= v.
$$

The common value $v$ is called the **value of the game**, and any pair
$(p^*, q^*)$ achieving the equality is a **mixed-strategy Nash equilibrium**.
The same equilibrium can be computed by solving the linear program

$$
\max_{p, v} v \quad\text{s.t.}\quad p^\top A \ge v \mathbf{1}, \; \mathbf{1}^\top p = 1, \; p \ge 0,
$$

or by no-regret learning algorithms such as multiplicative weights, which is
the route taken by the implementation in `code/src/zero_sum_games.rs`.

#### Example: aggressive vs. passive traders

The chapter's recurring example is the 2x2 game in which two traders choose
between aggressive (market) and passive (limit) order types. A symmetric
zero-sum payoff matrix is

| | Trader B aggressive | Trader B passive |
|---|---|---|
| Trader A aggressive | $0$ | $+3$ |
| Trader A passive | $-3$ | $+1$ |

(only the row player's payoff is shown; the column player gets the negative).
The unique pure-strategy Nash equilibrium for the row player is *aggressive*
because it weakly dominates *passive*. The equilibrium value is $0$ and the
column player should also play aggressive.

This is too clean for real markets — payoffs depend on liquidity, queue
position, and information. In practice a richer Bayesian or repeated-game
formulation is needed, but the matrix game already captures the qualitative
trade-off: *aggressive trading taxes patient traders, but if everyone is
aggressive, no one captures spread*.

#### Why mixed strategies matter

Pure strategies are insufficient for many natural games. *Matching pennies*
(payoffs $\pm 1$ on the diagonal) has no pure-strategy equilibrium, but a
unique mixed equilibrium $(0.5, 0.5)$ for both players. Markets full of
identifiable patterns invite the same outcome: any deterministic strategy is
predictable and exploitable, so the equilibrium response is to randomize.

Real-world examples include order-slicing strategies (TWAP-style schedules
randomized over a window), child-order placement (random offsets within a
spread), and even physical placement of cross-connect cables (randomized
length to avoid timing arbitrage).

### 7.1.2 Stackelberg Games

A Stackelberg game has a strict order of moves. The **leader** commits to a
(possibly mixed) strategy $x$ first, the **follower** observes $x$ and picks
its best response $y(x) \in \arg\max_y u_F(x, y)$, and the leader anticipates
this best-response correspondence and chooses

$$
x^* = \arg\max_{x} u_L(x, y(x)).
$$

The leader's commitment value can be strictly higher than the simultaneous
Nash equilibrium because the leader can credibly choose a strategy that
shapes the follower's response. The commitment is what gives the model bite:
without commitment, the leader would deviate after the follower chose.

#### Application: institutional execution against HFT

A large institutional trader who must liquidate $X$ shares over a horizon $T$
moves first. HFTs observe the visible portion of the schedule (the marginal
demand on the book) and decide whether to provide liquidity, follow the trade,
or trade against it. Modeling the HFT response as a best-response function
of the institutional schedule turns an Almgren–Chriss-style optimization into
a Stackelberg problem.

The chapter implementation `code/src/stackelberg.rs` solves the discrete
bilevel problem by exhaustive search over the leader's pure strategies and
exact best-response computation for the follower. For small toy problems
this is sufficient and avoids the pitfalls of bilevel optimization in
continuous spaces.

### 7.1.3 Repeated Games and Cooperation

A one-shot prisoner's dilemma has *defect* as the unique Nash equilibrium.
Repeated $T$ times with $T$ known and finite, backward induction unravels
cooperation. With $T = \infty$ or with stochastic ending and a high enough
discount factor $\delta$, the **Folk theorem** says that any individually
rational payoff profile can be sustained as a subgame-perfect equilibrium
using trigger strategies.

In trading, repeated interaction on lit venues gives market makers a strong
incentive to cooperate on quoting wide enough spreads to cover adverse
selection. Tit-for-tat-like behaviour is rational: respond aggressively to
short-term liquidity demand, then return to a cooperative wide-quote regime.

The empirical signature is well known: spreads widen during the first minute
of a session and after macro releases, then compress as market makers infer
that order flow is uninformative.

### 7.1.4 Dominance and Iterated Deletion

A strategy $s_i$ **strictly dominates** $s_i'$ if for every opponent profile
$s_{-i}$, $u_i(s_i, s_{-i}) > u_i(s_i', s_{-i})$. Rational players never use
strictly dominated strategies, so removing them yields a smaller game whose
equilibria are equilibria of the original. Iterating this procedure until
no further dominated strategies exist is called **iterated elimination of
strictly dominated strategies (IESDS)**. The order does not matter.

In the auctions section we use IESDS to derive the second-price auction's
truthful-bidding result: bidding $b > v$ is weakly dominated by bidding $v$,
and so is $b < v$.

---

## 7.2 Auctions in Financial Markets

### 7.2.1 Auction Formats

Four single-item auction formats account for almost all activity in financial
markets:

| Format | Winner | Pays |
|---|---|---|
| First-price sealed-bid | highest bid | own bid |
| Second-price sealed-bid (Vickrey) | highest bid | second-highest bid |
| Dutch (descending clock) | first to accept | clock price at acceptance |
| English (ascending clock) | last active | second-highest valuation $+\epsilon$ |

The two sealed-bid formats are equivalent to the two clock formats under
private values: Dutch $\equiv$ first-price, English $\equiv$ second-price.

A fifth format, the **combinatorial auction**, generalizes single-item
auctions to bundles. Bidders submit XOR bids over bundles of items and the
auctioneer picks a welfare-maximizing assignment. Combinatorial auctions are
relevant to portfolio trading (e.g. ETF creation/redemption) and to spectrum
allocation; in financial markets they appear as *winner-determination
problems* on multi-leg trades.

### 7.2.2 Optimal Bidding Under Symmetric IID Priors

Assume $n$ symmetric bidders with private valuations drawn IID from a uniform
$[0, 1]$ distribution. The Bayesian Nash equilibrium bidding strategies are:

- First-price: $b(v) = \frac{n-1}{n} v$.
- Second-price: $b(v) = v$ (truthful bidding is dominant).
- Dutch: same as first-price by strategic equivalence.
- English: same as second-price by strategic equivalence.

The expected revenue under any of these formats is

$$
\mathbb{E}[R] = \frac{n-1}{n+1},
$$

a result known as the **revenue equivalence theorem** (Myerson 1981, Vickrey
1961). The intuition is that any standard auction with IID private values that
allocates to the highest-value bidder yields the same expected revenue.

The corresponding implementations in `code/src/auctions.rs` use these closed
forms for `optimal_bid` and `expected_revenue`. The Monte-Carlo example
`examples/auction_simulation.rs` verifies revenue equivalence empirically:
both first- and second-price runs produce mean revenue near $4/6 \approx
0.6667$ for $n = 5$.

### 7.2.3 Reserve Prices and the Optimal Auction

Myerson (1981) showed that the **revenue-maximizing auction** with IID
private values awards the item to the bidder with the highest *virtual
valuation* $\psi(v) = v - \frac{1 - F(v)}{f(v)}$ provided $\psi(v) \ge 0$, and
sets the price to the second-highest virtual valuation. For uniform priors,
$\psi(v) = 2v - 1$, and the optimal reserve price is $r^* = 1/2$.

A reserve price strictly increases revenue when the realization of the
highest valuation is below the reserve, at the cost of sometimes leaving
the item unsold. The chapter's implementation supports a reserve price
parameter; setting it to $0$ recovers the no-reserve baseline.

### 7.2.4 Frequent Batch Auctions

Continuous limit-order books are vulnerable to a latency arms race: if the
exchange clears in continuous time, the first firm to receive new
information can race to the front of the queue and adversely select stale
quotes. Budish, Cramton, and Shim (2015, 2024 update) propose **frequent
batch auctions (FBA)** as a market-design response: collect all orders
arriving within a discrete interval (e.g. 100 ms) and clear them in a
single uniform-price call auction.

In the FBA, the first-mover advantage collapses to noise on the millisecond
scale because arrival within the batch is irrelevant. The trade-off is a
small price-discovery delay — bounded by the batch length — in exchange for
removing the rent that pays for the latency arms race.

### 7.2.5 Manipulation and Sniping

The English auction's susceptibility to **sniping** (last-millisecond bids
that prevent further responses) generalizes to opening and closing auctions
on financial exchanges. Empirical work documents elevated trading volume
and price impact in the final seconds of the closing auction. Mechanism
designers respond with random ending times, guaranteed extension intervals,
or cross-period imbalance broadcasts.

---

## 7.3 Strategic Interaction on Markets

### 7.3.1 Market Making as a Game

A market maker quoting a two-sided price $\{b, a\}$ chooses a spread $a - b$
that trades off:

- *Profit per trade*: $\propto a - b$.
- *Adverse selection*: probability that an arriving order is informed.
- *Inventory risk*: penalty for accumulating one-sided inventory.

If multiple market makers compete on price-time priority, equilibrium spread
compression follows. The result is a Bertrand-style outcome with spreads
that cover adverse selection but earn no rent. Empirical evidence shows
that bid-ask spreads on liquid US equities and major crypto pairs are very
close to this competitive lower bound during normal hours.

Adverse selection is captured by the **Glosten–Milgrom** (1985) model: with
informed traders arriving with probability $\pi$ and uninformed with
probability $1-\pi$, a competitive market maker's bid and ask are the
conditional expectations of the asset value given a sell or buy order. Even
with risk-neutral makers and zero inventory cost, the spread is positive.

### 7.3.2 The Kyle (1985) Model

Kyle's single-period model is the cleanest formal treatment of informed
trading. Three agents:

- **Informed trader**: privately observes the asset's terminal value $v \sim
  \mathcal{N}(0, \sigma_v^2)$ and chooses a signed volume $x$.
- **Noise traders**: submit a random aggregate volume $u \sim
  \mathcal{N}(0, \sigma_u^2)$.
- **Market maker**: observes only the total order flow $y = x + u$ and quotes
  a price $p$ that earns zero expected profit.

In equilibrium:

$$
x = \beta v, \quad p = \lambda y, \quad \beta = \frac{\sigma_u}{\sigma_v},
\quad \lambda = \frac{\sigma_v}{2 \sigma_u}.
$$

The informed trader's expected profit is

$$
\mathbb{E}[\pi] = \frac{\sigma_v \sigma_u}{2}.
$$

Kyle's $\lambda$ is the canonical measure of **price impact**: a one-unit
order moves the price by $\lambda$. The variance of the order flow is split
half-and-half between informed and noise components in equilibrium, a fact
exploited by the implementation's `informed_volume_share` method.

The model generalizes to dynamic settings (Kyle 1985 multi-period; Back
1992; Foster and Viswanathan 1996), with informed traders front-loading or
back-loading depending on the dynamics of $\sigma_u^2(t)$ and the rate at
which information leaks. The intuition that *price impact is the price of
camouflage* survives intact.

### 7.3.3 The HFT Speed Arms Race

Following Budish, Cramton, and Shim, model HFTs as players who choose a
non-negative speed investment $s_i$ at cost $c_i s_i$. The probability of
winning a per-period rent $V$ is given by a Tullock contest-success function

$$
p_i = \frac{s_i}{\sum_j s_j}, \qquad s_i \ge 0.
$$

Each HFT solves $\max_{s_i \ge 0} V p_i - c_i s_i$. With symmetric costs
$c_i = c$, the unique symmetric Nash equilibrium is

$$
s^* = \frac{V (n-1)}{n^2 c}, \qquad
\text{rent dissipation } = V \frac{n-1}{n}.
$$

As $n \to \infty$ the dissipation tends to $V$: the entire rent is paid out
in cumulative speed investments. With asymmetric costs the equilibrium is
characterized by an *active set* of low-cost players, identified by the
condition $c_i (k-1) / \sum_{j \in \text{active}} c_j < 1$. The
implementation in `code/src/hft_arms_race.rs` solves this directly.

The **socially optimal** speed in this model is zero: speed investments
produce no public benefit and only redistribute the rent, so any positive
investment is deadweight loss. This is the formal justification for
discrete-time market designs (FBAs) that destroy the rent at its source.

### 7.3.4 Payment for Order Flow (PFOF)

PFOF is a transfer between a wholesaler (off-exchange market maker) and a
retail broker in exchange for routing the broker's flow. Game-theoretically,
PFOF is a *bid for a stream of orders*: wholesalers bid against each other
for the flow of a broker, and the broker chooses the highest bid net of
execution-quality concessions.

Welfare analyses are subtle. Critics argue that PFOF lets wholesalers
internalize uninformed flow and leaves only adversely-selected flow on
public exchanges, widening lit spreads. Proponents argue that retail flow
would not interact with the lit book even without PFOF and that the
wholesaler's quoted improvement is a real benefit to retail. Both views can
be modeled with a Glosten–Milgrom-style adverse-selection mechanism and
either supports or rejects PFOF depending on the calibration of $\pi$.

---

## 7.4 Concrete Scenarios

### 7.4.1 Colonel Blotto for Liquidity Allocation

In the Colonel Blotto game, each of two players allocates a fixed budget
across $K$ battlefields; the higher allocation on each battlefield wins one
point, and the player with more points wins the game. With equal budgets,
the symmetric mixed-strategy equilibrium independently draws each
battlefield's allocation from a uniform $[0, 2 B / K]$ distribution and
rescales to satisfy the budget constraint.

Trading interpretation: $K$ assets, $B$ units of capital, and the player
with more capital on each asset captures the rent (e.g. in winner-takes-all
allocation games such as proof-of-stake validation rewards or token-launch
auctions). The same machinery applies to *attention allocation*: a research
desk distributes analyst-hours across stocks, and analysts with more hours
on a stock are more likely to identify a profitable trade.

The implementation `code/src/colonel_blotto.rs` provides a Monte-Carlo
evaluator. Empirically, with equal budgets and $K$ battlefields the expected
score is $K / 2$, and the variance scales as $K / 12$ — a useful diagnostic
for confirming that an allocation rule has the right second moment.

### 7.4.2 All-Pay Auction for Latency

The all-pay auction is the natural normal-form representation of an arms
race: every player pays its bid regardless of outcome, and the highest
bidder wins. With $n$ symmetric bidders and uniform $[0, 1]$ private
valuations, the symmetric equilibrium bid is

$$
b(v) = \frac{n-1}{n} v^n,
$$

and the expected revenue is $(n-1)/(n+1)$, identical to first- and
second-price auctions by revenue equivalence.

For two heterogeneous bidders with valuations $v_1 \ge v_2$, the
complete-information equilibrium has bid distributions

$$
F_1(b) = \frac{b}{v_2}, \quad b \in [0, v_2],
$$

$$
F_2(b) = 1 - \frac{v_2}{v_1} + \frac{b}{v_1}, \quad b \in [0, v_2],
$$

(Baye, Kovenock, and de Vries 1996), and the expected total bid is
$v_2/2 + v_2^2/(2 v_1)$. This last expression captures the welfare cost of
arms races: as the underdog's valuation $v_2$ approaches the favourite's
$v_1$, total dissipation approaches $v_1$ — full rent dissipation.

### 7.4.3 Double Auction (Walrasian Clearing)

The double auction is the workhorse of financial markets: $m$ buyers and $n$
sellers submit bid and ask schedules, the auctioneer finds the price at
which aggregate demand equals aggregate supply, and trades execute at that
price. With private values drawn from continuous distributions, the unique
ex-post efficient mechanism is the **uniform-price double auction**, but its
incentive properties are weaker than the second-price auction's: bidders
have a small but positive incentive to shade their bids.

In practice, exchange-traded markets implement a continuous version of the
double auction via the limit order book. Frequent batch auctions are simply
periodic uniform-price double auctions. The choice between continuous and
batched clearing is a fundamental market-design dial that the chapter
returns to repeatedly.

---

## 7.5 Simulation, Backtesting, and Empirical Tests

### 7.5.1 Monte-Carlo for Auctions

The simplest empirical test of any auction model is a Monte-Carlo simulation
that draws private valuations from a parametric distribution and runs the
auction. The example
`examples/auction_simulation.rs` does this for first- and second-price
auctions with $n = 5$ uniform $[0, 1]$ bidders, returning empirical mean
revenue within standard error of the theoretical $(n - 1)/(n + 1)$.

Generalizations are straightforward:

- Replace the uniform prior with a calibrated empirical distribution.
- Add reserve prices and observe the mass of unsold items.
- Add bidder asymmetries by drawing types from different distributions.
- Add common-value components by correlating the private signals.

### 7.5.2 Evolutionary Dynamics

Replicator dynamics provide a learning-theoretic foundation for Nash
equilibrium. The replicator equation

$$
\dot{p}_i = p_i \left( (Aq)_i - p^\top A q \right)
$$

evolves a population playing strategy $i$ with frequency $p_i$. Under mild
conditions, fixed points of the replicator dynamics are Nash equilibria;
asymptotically stable fixed points are *evolutionarily stable strategies*
(ESS). For market making, replicator dynamics describe the gradual
crowd-out of strategies whose realized profit is below the population
average. In algorithmic trading research, replicator dynamics provide a
useful smoke test: if a candidate strategy's fitness is below the median in
agent-based simulation, it is unlikely to survive deployment.

### 7.5.3 Agent-Based Models

Agent-based models populate the order book with heterogeneous agents
(market makers, momentum traders, mean-reverters, noise traders) and
simulate price formation as the equilibrium of their interactions. The
Lux–Marchesi (1999) model and the Brock–Hommes (1998) model are canonical
references. Modern extensions add:

- Realistic limit-order-book microstructure.
- Latency heterogeneity for HFT participants.
- Calibration against empirical stylized facts such as fat tails, volatility
  clustering, and trade-size autocorrelation.

The modules in `code/src/` are intentionally small and composable, so they
can be embedded in a larger agent-based model: the `KyleModel` provides
informed-trader behaviour, `Auction` provides clearing, and `HFTArmsRace`
provides the latency-investment subroutine.

### 7.5.4 Comparing Market Designs

A common research pipeline is:

1. Simulate trading on a continuous limit-order book under realistic
   latency parameters.
2. Simulate the same trader population on a frequent batch auction with a
   chosen batch length $\Delta$.
3. Compare welfare metrics: realized spread, total HFT profit, and
   uninformed-trader cost.

The HFT arms race module supplies the comparison's quantitative backbone:
the equilibrium speed investment is zero under FBA (no rent to capture)
and $V (n-1)/n$ under continuous clearing, providing an upper bound on the
welfare gain from switching to FBA.

---

## 7.6 Connections to Earlier Chapters

- **Chapter 1 (Stochastic Calculus)**: Kyle's model and Glosten–Milgrom both
  rely on conditional expectations of normally distributed variables. The
  martingale representation theorem underpins the maker's pricing rule.
- **Chapter 2 (Microstructure)**: Auctions, double auctions, and limit
  order books are alternative implementations of the same economic
  primitive. The choice of clearing mechanism shapes price discovery.
- **Chapter 3 (Portfolio Optimization)**: Colonel Blotto generalizes
  portfolio choice to *competitive* allocation. Asset weights become
  battlefield investments; correlation between assets becomes correlation
  between battlefield outcomes.
- **Chapter 4 (ML for Time Series)**: Strategic interaction makes feature
  distributions endogenous. A signal that becomes profitable also becomes
  crowded, decaying its information coefficient. Game theory tells us *why*
  information coefficients decay.
- **Chapter 5 (Low-Latency Systems)**: The HFT arms race is the formal
  reason latency engineering is a strategic, not merely operational,
  problem. The deadweight loss of speed investments is a real cost that
  market design can eliminate.
- **Chapter 6 (Information and Kelly Sizing)**: Kelly tells the trader how
  much to bet when an edge is known. Game theory tells the trader why an
  edge erodes through repeated interaction with adversarial counterparties,
  and how to build sizing rules that anticipate the erosion.

---

## 7.7 Worked Implementations

The companion crate `game-theory-auctions` collects the chapter's algorithms
in `code/`. Each module is small and individually testable:

| Module | Topic | Public API |
|---|---|---|
| `zero_sum_games.rs` | matrix games | `ZeroSumGame::nash_equilibrium`, `dominant_strategy` |
| `stackelberg.rs` | leader/follower | `StackelbergGame::leader_optimal`, `follower_response` |
| `auctions.rs` | sealed-bid and clock auctions | `Auction::run`, `optimal_bid`, `expected_revenue` |
| `kyle_model.rs` | informed trading | `KyleModel::informed_strategy`, `pricing_rule`, `equilibrium_lambda` |
| `hft_arms_race.rs` | Tullock contest | `HFTArmsRace::equilibrium_speeds`, `deadweight_loss` |
| `colonel_blotto.rs` | resource allocation | `ColonelBlotto::monte_carlo_value`, `expected_payoff` |
| `all_pay_auction.rs` | all-pay equilibrium | `AllPayAuction::symmetric_equilibrium_bid`, `asymmetric_expected_revenue` |

The `benches/auction_benchmark.rs` file provides Criterion benchmarks for
each module; `examples/auction_simulation.rs` runs a 50,000-trial
revenue-equivalence Monte Carlo. All numerical results in this chapter are
reproducible with `cargo run` and `cargo test`.

### 7.7.1 Verifying Revenue Equivalence

```text
$ cargo run --example auction_simulation
Number of bidders: 5
Runs: 50000
Mean first-price revenue:  0.6669
Mean second-price revenue: 0.6654
Theoretical (revenue equivalence): 0.6667
```

The two formats agree with the theoretical $0.6667$ to within Monte-Carlo
error.

### 7.7.2 Verifying the HFT Equilibrium

For $n = 3$ symmetric HFTs, prize $V = 9$, cost $c = 1$:

```text
Closed-form symmetric speed: s* = V (n - 1)/(n^2 c) = 9 * 2 / 9 = 2
Iterated equilibrium speeds: [2.0, 2.0, 2.0]
Total rent dissipation: V (n - 1)/n = 6
```

The iterative solver matches the closed form. For asymmetric costs, the
solver correctly identifies the active set and assigns zero speed to
sufficiently expensive players.

---

## 7.8 Open Problems and Further Reading

1. **Endogenous information acquisition**: Most models above take
   informational asymmetries as given. A richer model has bidders or
   traders choosing how much to learn before participating; this connects
   game theory to Chapter 4 and Chapter 6 through the *value of information*
   framework.
2. **Mechanism design with budget constraints**: Real bidders have
   liquidity constraints. The optimal auction with budget-constrained
   bidders (Pai and Vohra 2014, Che and Gale 2000) is an active research
   area.
3. **Algorithmic collusion**: Q-learning bots have been shown to learn
   tacit collusion in Bertrand competition without explicit signalling
   (Calvano et al. 2020). Whether this happens in financial market making
   is an open empirical question with major regulatory implications.
4. **Fairness and auction design**: New batch-auction proposals and
   rollup-based MEV auctions raise questions about how fairness, latency,
   and welfare interact in practice. The chapter's tooling is a starting
   point for evaluating these proposals quantitatively.
5. **Reinforcement learning in games**: Self-play and policy-gradient
   methods have produced superhuman play in many games. Adapting them to
   the partial-information, partially-observable setting of order-book
   trading is an active research direction.

A short reading list for the energetic student:

- Fudenberg, D., and J. Tirole. *Game Theory*. MIT Press, 1991.
- Krishna, V. *Auction Theory*. Academic Press, 2nd ed., 2009.
- Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
  *Econometrica* 53 (6): 1315–1335.
- Budish, E., P. Cramton, and J. Shim (2015). "The High-Frequency Trading
  Arms Race: Frequent Batch Auctions as a Market Design Response."
  *Quarterly Journal of Economics* 130 (4): 1547–1621.
- Roughgarden, T. *Twenty Lectures on Algorithmic Game Theory*. Cambridge,
  2016.

---

## Summary

Strategic interaction is not a finishing flourish on top of probability
and optimization — it is the bedrock on which financial markets stand.
Every order interacts with traders who will respond, every quote competes
with quotes that will adjust, and every speed investment is matched by
rivals' investments. This chapter introduced the formal vocabulary —
zero-sum games, Stackelberg leadership, auctions, the Kyle model, the
Tullock contest, Colonel Blotto, and all-pay auctions — and provided
production-ready Rust implementations with tests and benchmarks.

The strongest practical takeaway is that *equilibrium is the right level
of abstraction*. A pricing model that ignores how counterparties will
react is incomplete; a strategy that does not anticipate adverse selection
is fragile. The exercises and code in this chapter are designed to make
strategic reasoning a default mode, not an afterthought.
