"""Synthetic demonstration: shrinkage-induced stability is not evidence of a good target.

Why this exists (Editor condition E2)
-------------------------------------
The empirical study reported that anchoring reduced inter-seed dispersion, replicated across
four runs with a dose-response (Spearman rho = +0.585) and a sign test at p = 5.2e-04. A
scale-matched permuted control dissolved it. That is one case; a referee is entitled to ask
whether the artefact is general.

Here it is isolated with ground truth known. Everything is simulated, the optimal parameter
vector is available in closed form, and the "informative" and "uninformative" anchors are
constructed to be identical in scale and different only in whether they point at the truth.

The mechanism, stated analytically
----------------------------------
Fit theta by descending  L(theta) = pinball(y - X theta) + w * ||theta - a||^2  from a random
initialisation, for a finite number of steps. The gradient of the penalty is 2w(theta - a), so
each step contracts the iterate toward the FIXED point ``a`` by a factor (1 - 2*lr*w) in
addition to whatever the data term does. Two runs differing only in initialisation therefore
have their separation multiplied by (1 - 2*lr*w) every step:

    spread_T  ~  spread_0 * (1 - 2*lr*w)^T

This contraction depends on ``w``, ``lr`` and ``T``. **It does not depend on ``a``.** Shrinking
toward a perfect target and shrinking toward nonsense reduce inter-seed dispersion by the same
factor. Any study that reports "our regulariser makes the estimator more stable" and stops
there has reported this identity.

Pure numpy: no torch, no market data, runs in seconds.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "simulate", "optimal_theta", "fit_anchored", "seed_dispersion",
    "predicted_contraction", "run_demo", "default_weight_grid", "paired_comparison",
]


def simulate(n: int = 4000, d: int = 3, alpha: float = 0.05, seed: int = 0):
    """Linear location model with Gaussian noise, so the optimal linear quantile is exact.

    y = X beta + eps,  eps ~ N(0, 1)  =>  the alpha-quantile of y | X is  X beta + z_alpha.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta = rng.standard_normal(d)
    y = X @ beta + rng.standard_normal(n)
    X1 = np.hstack([X, np.ones((n, 1))])                 # intercept carries z_alpha
    theta_star = np.append(beta, stats.norm.ppf(alpha))
    return X1, y, theta_star


def optimal_theta(theta_star: np.ndarray) -> np.ndarray:
    """Ground truth: the population-optimal linear quantile parameters."""
    return theta_star.copy()


def _pinball_subgrad(X, y, theta, alpha):
    r = y - X @ theta
    g = np.where(r < 0, 1.0 - alpha, -alpha)             # d/dtheta of pinball wrt residual
    return X.T @ g / len(y)


def fit_anchored(X, y, alpha, anchor, w, seed, steps=400, lr=0.05, init_scale=3.0):
    """Descend the anchored pinball objective from a random start, for a FIXED budget.

    A finite budget is essential: with unlimited steps a convex problem converges to the same
    optimum from every initialisation and inter-seed dispersion is zero regardless of w. The
    real study had dispersion precisely because its budget was finite (early stopping), so the
    demonstration reproduces that condition rather than assuming it away.
    """
    rng = np.random.default_rng(1000 + seed)
    theta = rng.standard_normal(X.shape[1]) * init_scale
    for _ in range(steps):
        g = _pinball_subgrad(X, y, theta, alpha)
        if w:
            g = g + 2.0 * w * (theta - anchor)
        theta = theta - lr * g
    return theta


def pinball_loss(X, y, theta, alpha):
    r = y - X @ theta
    return float(np.mean(r * (alpha - (r < 0))))


def seed_dispersion(X, y, Xte, yte, alpha, anchor, w, n_seeds=20, **kw):
    """Inter-seed IQR of out-of-sample loss, and of the fitted parameters."""
    thetas = [fit_anchored(X, y, alpha, anchor, w, s, **kw) for s in range(n_seeds)]
    losses = np.array([pinball_loss(Xte, yte, t, alpha) for t in thetas])
    q25, q50, q75 = np.percentile(losses, [25, 50, 75])
    theta_spread = float(np.mean(np.std(np.array(thetas), axis=0)))
    return dict(median=float(q50), iqr=float(q75 - q25), theta_spread=theta_spread)


def predicted_contraction(w: float, lr: float = 0.05, steps: int = 400) -> float:
    """Analytical spread contraction from the penalty term alone: (1 - 2*lr*w)^steps.

    Depends on w, lr and steps. Independent of the anchor's value - which is the whole point.
    """
    factor = 1.0 - 2.0 * lr * w
    if factor <= 0:
        return 0.0
    return float(factor ** steps)


def default_weight_grid(n: int = 10, lo: float = 5e-4, hi: float = 0.1):
    """Log-spaced weights plus zero.

    Editor condition E7: the first version used four non-zero weights and the paper quoted the
    single most extreme cell (nonsense stabilising 2.5x more than the truth at w = 0.05). Four
    points also make a bootstrap interval useless - it saturates at the observed range. A denser
    grid removes both problems, and this demonstration is pure numpy, so density is free.
    """
    return (0.0,) + tuple(np.geomspace(lo, hi, n))


def paired_comparison(rows) -> dict:
    """Truth vs nonsense **at the same weight**, paired across the grid.

    The correct statistic for the demonstration's claim. Pairing uses every weight instead of
    inviting a choice among them, so there is no most-favourable cell to quote.
    Returns the per-weight ratios and a sign test that nonsense stabilises at least as much.
    """
    import pandas as pd
    from scipy import stats as st

    d = pd.DataFrame(rows)
    p = d[d.anchor.isin(["informative (truth)", "nonsense (scale-matched)"]) & (d.weight > 0)]
    p = p.pivot_table(index="weight", columns="anchor", values="iqr_ratio")
    rel = (p["nonsense (scale-matched)"] / p["informative (truth)"]).dropna()
    n = int(rel.size)
    k = int((rel >= 1.0).sum())
    return {
        "n_weights": n,
        "nonsense_at_least_as_stabilising": k,
        "sign_test_p": float(st.binomtest(k, n, 0.5).pvalue) if n else float("nan"),
        "median_relative": float(rel.median()),
        "min_relative": float(rel.min()),
        "max_relative": float(rel.max()),
        "per_weight": {float(w): float(v) for w, v in rel.items()},
    }


def run_demo(weights=None, n_seeds: int = 20,
             alpha: float = 0.05, seed: int = 0):
    """Compare an informative anchor against scale-matched nonsense across a weight grid."""
    if weights is None:
        weights = default_weight_grid()
    X, y, theta_star = simulate(n=4000, alpha=alpha, seed=seed)
    Xte, yte, _ = simulate(n=4000, alpha=alpha, seed=seed + 500)
    theta_opt = optimal_theta(theta_star)

    rng = np.random.default_rng(999)
    rand_dir = rng.standard_normal(theta_opt.shape)
    # scale-matched nonsense: same norm as the truth, pointing somewhere else entirely
    theta_nonsense = rand_dir / np.linalg.norm(rand_dir) * np.linalg.norm(theta_opt)

    anchors = {
        "informative (truth)": theta_opt,
        "nonsense (scale-matched)": theta_nonsense,
        "zero": np.zeros_like(theta_opt),
    }

    rows = []
    base = {k: seed_dispersion(X, y, Xte, yte, alpha, a, 0.0, n_seeds)["iqr"]
            for k, a in anchors.items()}
    for name, a in anchors.items():
        for w in weights:
            r = seed_dispersion(X, y, Xte, yte, alpha, a, w, n_seeds)
            rows.append(dict(anchor=name, weight=w, iqr=r["iqr"], median=r["median"],
                             theta_spread=r["theta_spread"],
                             iqr_ratio=base[name] / r["iqr"] if r["iqr"] else np.inf,
                             predicted=predicted_contraction(w)))
    return rows
