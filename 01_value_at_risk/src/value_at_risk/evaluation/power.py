"""Power and minimum detectable effect - what this design could ever have found.

Why this exists
---------------
This project reported nulls across four selection configurations, a capacity arm and an MCS
that separated nothing, and at no point did anyone ask what effect size the design was able to
detect. A null from an underpowered design is not evidence of absence; it is absence of
evidence. The Risk review (R2) made this a condition of the G4 sign-off.

Two families of null are used in the study, so two power calculations are provided:

- **Proportion nulls** - "the selection error is indistinguishable from a coin flip",
  "the anchor reduces seed IQR in k of n comparisons". Exact binomial.
- **Loss-differential nulls** - "the anchored model does not beat the unanchored one",
  tested by Diebold-Mariano. Normal approximation using the observed HAC standard error, so
  the minimum detectable effect comes out in pinball-loss units and can be compared directly
  against the effect sizes actually observed.

Everything here is deliberately conservative: no continuity corrections that flatter power, and
the binomial routines are exact rather than normal-approximated at these small n.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "binomial_power", "binomial_mde", "binomial_required_n",
    "dm_standard_error", "dm_mde", "power_report",
    "bootstrap_ci", "ratio_report",
]


def bootstrap_ci(values, statistic=np.median, B: int = 10000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap interval for a statistic of ``values``.

    Returns ``(point, lo, hi)``.

    Note on what is being resampled. The magnitudes this project quotes - "median IQR ratio
    13.5x" - are medians **across comparisons**, so the resampling unit is the comparison. That
    is the interval for the quantity actually reported. A seed-level interval would require the
    per-seed losses, which the pipeline did not persist (see ``ratio_report``); the two answer
    different questions and must not be confused.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(B, v.size), replace=True)
    stats_ = np.apply_along_axis(statistic, 1, draws)
    a = (1.0 - level) / 2.0
    return (float(statistic(v)), float(np.quantile(stats_, a)),
            float(np.quantile(stats_, 1.0 - a)))


def ratio_report(values, label: str = "", B: int = 10000, level: float = 0.95,
                 seed: int = 0) -> dict:
    """A magnitude with its interval, formatted for direct quotation in the paper.

    Risk condition F7 / Editor E6: no ratio is quoted as a bare point estimate. A paper whose
    thesis is that people report numbers without their uncertainty cannot do so itself.
    """
    point, lo, hi = bootstrap_ci(values, np.median, B=B, level=level, seed=seed)
    v = np.asarray(list(values), dtype=float)
    return {
        "label": label, "n": int(np.isfinite(v).sum()),
        "median": point, "ci_low": lo, "ci_high": hi, "level": level,
        "text": f"{point:.1f}x (95% CI {lo:.1f}-{hi:.1f}, n={int(np.isfinite(v).sum())})",
    }


def _two_sided_rejection_region(n: int, p_null: float, alpha: float):
    """Exact critical values k_lo, k_hi such that rejecting outside [k_lo, k_hi] has size
    <= alpha under the null."""
    lo = stats.binom.ppf(alpha / 2, n, p_null) - 1
    hi = stats.binom.isf(alpha / 2, n, p_null) + 1
    return int(max(lo, -1)), int(min(hi, n + 1))


def binomial_power(n: int, p_true: float, p_null: float = 0.5,
                   alpha: float = 0.05) -> float:
    """Exact power of a two-sided binomial test of ``p = p_null`` when the truth is ``p_true``."""
    if n <= 0:
        return 0.0
    k_lo, k_hi = _two_sided_rejection_region(n, p_null, alpha)
    # reject when k <= k_lo or k >= k_hi
    return float(stats.binom.cdf(k_lo, n, p_true) + stats.binom.sf(k_hi - 1, n, p_true))


def binomial_mde(n: int, power: float = 0.8, p_null: float = 0.5,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Smallest departures from ``p_null`` detectable at the requested power.

    Returns ``(p_low, p_high)``: proportions at or beyond these values are detectable.
    ``nan`` on a side means no value on that side reaches the requested power at this n.
    """
    grid = np.linspace(0.0, 1.0, 1001)
    powers = np.array([binomial_power(n, p, p_null, alpha) for p in grid])
    ok = powers >= power
    low = grid[ok & (grid < p_null)]
    high = grid[ok & (grid > p_null)]
    return (float(low.max()) if low.size else float("nan"),
            float(high.min()) if high.size else float("nan"))


def binomial_required_n(p_true: float, power: float = 0.8, p_null: float = 0.5,
                        alpha: float = 0.05, n_max: int = 5000) -> int:
    """Smallest n reaching ``power`` against ``p_true``. Returns ``n_max`` if unreachable."""
    for n in range(5, n_max + 1):
        if binomial_power(n, p_true, p_null, alpha) >= power:
            return n
    return n_max


def dm_standard_error(loss_a, loss_b, lag: int = 5) -> float:
    """HAC (Newey-West, Bartlett) standard error of the mean loss differential.

    Mirrors the estimator used by ``scoring.diebold_mariano`` so the power statement refers to
    the test actually run, not an idealised one.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        lrv = float(np.var(d, ddof=1))
    return float(np.sqrt(max(lrv, 0.0) / n))


def dm_mde(loss_a, loss_b, lag: int = 5, power: float = 0.8,
           alpha: float = 0.05, one_sided: bool = True) -> float:
    """Minimum detectable mean loss differential, in loss units.

    An effect smaller than this could not have been detected at the requested power, so a
    non-rejection at or below it carries no information.
    """
    se = dm_standard_error(loss_a, loss_b, lag=lag)
    z_a = stats.norm.isf(alpha) if one_sided else stats.norm.isf(alpha / 2)
    z_b = stats.norm.isf(1 - power)
    return float((z_a + z_b) * se)


def power_report(n: int, k_observed: int, p_null: float = 0.5, alpha: float = 0.05,
                 power: float = 0.8) -> dict:
    """Everything needed to report a proportion null honestly."""
    p_hat = k_observed / n if n else float("nan")
    ci = stats.binomtest(k_observed, n, p_null).proportion_ci(1 - alpha)
    lo, hi = binomial_mde(n, power, p_null, alpha)
    return {
        "n": n,
        "k": k_observed,
        "p_hat": p_hat,
        "p_value": float(stats.binomtest(k_observed, n, p_null).pvalue),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "mde_low": lo,
        "mde_high": hi,
        "power_at_observed": binomial_power(n, p_hat, p_null, alpha),
        "n_for_80pct_at_observed": binomial_required_n(p_hat, power, p_null, alpha),
        "informative": bool(p_hat <= lo or p_hat >= hi) if not (np.isnan(lo) and np.isnan(hi))
        else False,
    }
