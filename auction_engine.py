"""
Core game-theory engine for the Auction Bidding Strategy Analyzer.

All simulation is vectorised with NumPy: `values` and `bids` are arrays of
shape (runs, n_bidders), so thousands of auctions run in milliseconds.
The seller's own value for the item is assumed to be 0.
"""
import numpy as np

FIRST = "First-price"
SECOND = "Second-price (Vickrey)"
MECHANISMS = [FIRST, SECOND]

STRATEGIES = [
    "Truthful",
    "Shade (bid below value)",
    "Overbid (bid above value)",
    "Equilibrium shading",
]
DISTRIBUTIONS = ["Uniform (0-100)", "Normal (mean 50, sd 15)", "Exponential (mean 30)"]


def bid_factor(strategy: str, pct: float, n: int) -> float:
    """Bid = valuation * factor."""
    if strategy == "Truthful":
        return 1.0
    if strategy.startswith("Shade"):
        return max(0.0, 1 - pct / 100)
    if strategy.startswith("Overbid"):
        return 1 + pct / 100
    if strategy == "Equilibrium shading":
        # Bayes-Nash equilibrium of the first-price auction with i.i.d. uniform values
        return (n - 1) / n
    raise ValueError(f"Unknown strategy: {strategy}")


def draw_values(rng, runs: int, n: int, dist: str) -> np.ndarray:
    if dist.startswith("Uniform"):
        return rng.uniform(0, 100, (runs, n))
    if dist.startswith("Normal"):
        return np.clip(rng.normal(50, 15, (runs, n)), 0, None)
    if dist.startswith("Exponential"):
        return rng.exponential(30, (runs, n))
    raise ValueError(dist)


def run_auctions(values, bids, mechanism, reserve, rng):
    """Resolve many auctions at once. Ties in bids are broken at random."""
    values = np.atleast_2d(values).astype(float)
    bids = np.atleast_2d(bids).astype(float)
    runs, n = bids.shape
    rows = np.arange(runs)

    order = np.argsort(-(bids + rng.random(bids.shape) * 1e-9), axis=1)
    w = order[:, 0]
    top = bids[rows, w]
    second = bids[rows, order[:, 1]]

    sold = top >= reserve
    if mechanism == FIRST:
        price = top
    else:
        price = np.maximum(second, reserve)
    price = np.where(sold, price, 0.0)

    win_val = values[rows, w]
    profits = np.zeros_like(values)
    profits[rows, w] = np.where(sold, win_val - price, 0.0)

    return {
        "winner": np.where(sold, w, -1),
        "price": price,
        "revenue": price,
        "profits": profits,
        "sold": sold,
        "efficient": sold & (w == values.argmax(axis=1)),
        "welfare": np.where(sold, win_val, 0.0),
    }


def summarize(res):
    return {
        "Avg seller revenue": res["revenue"].mean(),
        "Avg winner profit": res["profits"].sum(axis=1).mean(),
        "Avg total welfare": res["welfare"].mean(),
        "Allocation efficiency %": 100 * res["efficient"].mean(),
        "Item sold %": 100 * res["sold"].mean(),
    }


def monte_carlo(values, factors, mechanism, reserve, seed=0):
    """`factors` is a scalar or per-bidder array of bid/valuation ratios."""
    rng = np.random.default_rng(seed)
    bids = values * np.asarray(factors, dtype=float)
    return run_auctions(values, bids, mechanism, reserve, rng)


def focal_sweep(values, factor_grid, others_factor, mechanism, reserve, seed=0):
    """Bidder 0 tries each bid factor; everyone else bids `others_factor` * value."""
    n = values.shape[1]
    out = []
    for f in factor_grid:
        factors = np.full(n, others_factor)
        factors[0] = f
        res = monte_carlo(values, factors, mechanism, reserve, seed)
        out.append((f, res["profits"][:, 0].mean(), (res["winner"] == 0).mean()))
    return out
