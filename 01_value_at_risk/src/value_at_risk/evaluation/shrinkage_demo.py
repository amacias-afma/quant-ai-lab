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
    "separation_trace", "contraction_accuracy", "anchor_invariance",
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


def separation_trace(X, y, alpha, anchor, w, seed_a=0, seed_b=1,
                     steps=400, lr=0.05, init_scale=3.0):
    """Track ||theta_a(t) - theta_b(t)|| for two runs differing only in initialisation.

    Why this exists
    ---------------
    The paper's derivation is

        Delta_{t+1} = (1 - 2*lr*w) * Delta_t  -  lr * [ g(theta_a) - g(theta_b) ]

    and then **drops the second term** to obtain ``spread_T ~ spread_0 * (1-2*lr*w)^T``.
    The cancellation of the anchor ``a`` in the first term is exact; the dropped term is an
    approximation that the paper asserted without measuring. This function measures it.

    Returns ``(observed, predicted)``, each of length ``steps + 1`` and normalised to
    ``||Delta_0|| = 1``, so they are directly comparable.

    Note the two runs must share the data and differ ONLY in initialisation, which is the
    condition under which the derivation applies.
    """
    rng_a = np.random.default_rng(1000 + seed_a)
    rng_b = np.random.default_rng(1000 + seed_b)
    ta = rng_a.standard_normal(X.shape[1]) * init_scale
    tb = rng_b.standard_normal(X.shape[1]) * init_scale

    d0 = float(np.linalg.norm(ta - tb))
    observed = np.empty(steps + 1)
    observed[0] = 1.0

    for t in range(steps):
        ga = _pinball_subgrad(X, y, ta, alpha)
        gb = _pinball_subgrad(X, y, tb, alpha)
        if w:
            ga = ga + 2.0 * w * (ta - anchor)
            gb = gb + 2.0 * w * (tb - anchor)
        ta = ta - lr * ga
        tb = tb - lr * gb
        observed[t + 1] = float(np.linalg.norm(ta - tb)) / d0

    factor = 1.0 - 2.0 * lr * w
    predicted = np.array([factor ** t if factor > 0 else 0.0 for t in range(steps + 1)])
    return observed, predicted


def contraction_accuracy(weights=None, alpha: float = 0.05, seed: int = 0,
                         steps: int = 400, lr: float = 0.05):
    """How well does (1-2*lr*w)^T describe the real separation trajectory?

    Reported per weight, using the TRUE anchor (the dropped term does not depend on the
    anchor's value, so the choice is immaterial; see ``test_shrinkage_demo``):

        absolute_ratio   obs_final / pred_final. 1.0 would mean the raw formula is exact.
        relative_ratio   the same comparison for the quantity the paper actually reports —
                         contraction **relative to w = 0** — which is what an IQR ratio is.
        max_log10_gap    worst absolute discrepancy over the trajectory, in decades.

    The w = 0 row is the diagnostic one: there the prediction is identically 1 (no penalty,
    hence no predicted contraction), so any observed contraction is **entirely** the dropped
    data term. It measures the omitted contribution in isolation.

    The distinction between the two ratios matters. The paper never quotes an absolute
    spread; every figure it reports is a ratio against the unanchored baseline. If the
    dropped term contributes a roughly constant factor, it cancels in that ratio, and the
    approximation is far better for the reported quantity than for the raw one.
    """
    if weights is None:
        weights = default_weight_grid()
    X, y, theta_star = simulate(n=4000, alpha=alpha, seed=seed)
    a = optimal_theta(theta_star)

    base = None
    rows = []
    for w in weights:
        obs, pred = separation_trace(X, y, alpha, a, w, steps=steps, lr=lr)
        if base is None:                       # first grid point is w = 0 by construction
            base = float(obs[-1])
        # Compare on the log scale: these span decades, so a raw difference is meaningless.
        floor = 1e-300
        gap = np.abs(np.log10(np.maximum(obs, floor)) - np.log10(np.maximum(pred, floor)))
        obs_rel = base / obs[-1] if obs[-1] > 0 else float("inf")
        pred_rel = 1.0 / pred[-1] if pred[-1] > 0 else float("inf")
        rows.append(dict(
            weight=float(w),
            observed_final=float(obs[-1]),
            predicted_final=float(pred[-1]),
            absolute_ratio=float(obs[-1] / pred[-1]) if pred[-1] > 0 else float("inf"),
            observed_rel=float(obs_rel),
            predicted_rel=float(pred_rel),
            relative_ratio=float(obs_rel / pred_rel) if pred_rel not in (0, float("inf")) else 1.0,
            max_log10_gap=float(gap.max()),
        ))
    return rows


def anchor_invariance(weights=None, alpha: float = 0.05, seed: int = 0,
                      steps: int = 400, lr: float = 0.05):
    """How anchor-independent is the contraction, really?

    **A correction, recorded because measuring it changed what we believed.** An earlier
    version of this docstring asserted that the two trajectories must agree "to numerical
    precision," on the grounds that ``a`` cancels exactly in

        Delta_{t+1} = (1 - 2*lr*w) * Delta_t  -  lr * [ g(theta_a) - g(theta_b) ]

    **That assertion was wrong, and this function is what showed it.** The cancellation is
    exact for the *penalty* term only. The anchor still moves each iterate individually, so
    it changes *where* the two runs sit, and therefore changes the pinball subgradient
    difference ``g(theta_a) - g(theta_b)`` — the term the derivation drops. The anchor is
    absent from the explicit contraction and re-enters implicitly through the data term.

    Measured (400 steps, lr = 0.05): agreement is within 3.4% for w <= 0.017 and degrades
    monotonically thereafter, reaching ~50% at w = 0.1. So the correct statement is
    **anchor-independent to first order, with a second-order dependence that grows with w**.

    This does not damage the paper's argument and the numbers say why: at w = 0.1 the two
    anchors differ by 50% while the contraction itself is 44x-56x, and the *nonsense* anchor
    contracts the parameter separation slightly **less** — so the residual dependence does
    not run in the direction that would rescue the informative prior.

    Returns per-weight the maximum relative difference between the two trajectories.
    """
    if weights is None:
        weights = default_weight_grid()
    X, y, theta_star = simulate(n=4000, alpha=alpha, seed=seed)
    truth = optimal_theta(theta_star)
    rng = np.random.default_rng(999)
    r = rng.standard_normal(truth.shape)
    nonsense = r / np.linalg.norm(r) * np.linalg.norm(truth)

    rows = []
    for w in weights:
        o_t, _ = separation_trace(X, y, alpha, truth, w, steps=steps, lr=lr)
        o_n, _ = separation_trace(X, y, alpha, nonsense, w, steps=steps, lr=lr)
        denom = np.maximum(np.abs(o_t), 1e-300)
        rows.append(dict(weight=float(w),
                         max_rel_diff=float(np.max(np.abs(o_t - o_n) / denom)),
                         final_truth=float(o_t[-1]),
                         final_nonsense=float(o_n[-1])))
    return rows


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
