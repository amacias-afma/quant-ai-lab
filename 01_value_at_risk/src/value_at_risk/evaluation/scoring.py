"""Strictly-consistent scoring and honest model comparison for VaR forecasts.

Design notes (read before changing conventions):

- A VaR forecast is a *quantile* forecast of the next-period return at level ``alpha``
  (small, e.g. 0.01 for the 99% VaR). It is expressed on the return scale, so it is a
  negative number for a loss threshold.
- **Breach convention (the sign that silently inverts every results table):**
  a breach happens when the realised return falls *below* the forecast, i.e.
  ``realised < var``. This module fixes that convention in one place.
- Rank models with a **strictly consistent** loss (pinball). Never rank by breach-rate
  pass/fail or by "capital reserved" — those are not consistent for the quantile and
  reward a model for being less conservative.
- Report a **loss differential with a test** (Diebold-Mariano), never a bare ranking.

All functions are pure ``numpy``/``scipy`` — no torch, no pandas required — so they are
cheap to unit-test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "pinball_loss",
    "pinball_loss_series",
    "breaches",
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_cc",
    "diebold_mariano",
]


def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a


def pinball_loss_series(realised, var, alpha: float) -> np.ndarray:
    """Per-observation pinball (quantile) loss for the ``alpha``-quantile.

    L_t = (realised - var) * (alpha - 1{realised < var})

    which expands to ``alpha * e`` when the return is above the forecast and
    ``(1 - alpha) * (-e)`` when it is a breach. Non-negative, and uniquely minimised
    in expectation by the true ``alpha``-quantile.
    """
    realised = _as_1d(realised)
    var = _as_1d(var)
    if realised.shape != var.shape:
        raise ValueError(f"shape mismatch: realised {realised.shape} vs var {var.shape}")
    e = realised - var
    below = (realised < var).astype(float)
    return e * (alpha - below)


def pinball_loss(realised, var, alpha: float) -> float:
    """Mean pinball loss. The number you rank models by."""
    return float(np.mean(pinball_loss_series(realised, var, alpha)))


def breaches(realised, var) -> np.ndarray:
    """Breach indicator series. Breach == realised return strictly below the VaR."""
    realised = _as_1d(realised)
    var = _as_1d(var)
    return (realised < var).astype(int)


def kupiec_pof(realised, var, alpha: float):
    """Kupiec unconditional-coverage (proportion-of-failures) test.

    H0: breach probability == alpha. Returns (statistic, p_value, n_breaches, n_obs).
    Large p => breach rate is consistent with the nominal level.
    """
    b = breaches(realised, var)
    n = b.size
    x = int(b.sum())
    pi = x / n if n else 0.0
    if x == 0:
        lr = -2.0 * (n * np.log(1 - alpha))
    elif x == n:
        lr = -2.0 * (n * np.log(alpha))
    else:
        ll_null = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
        ll_alt = x * np.log(pi) + (n - x) * np.log(1 - pi)
        lr = -2.0 * (ll_null - ll_alt)
    p = float(stats.chi2.sf(lr, df=1))
    return float(lr), p, x, n


def christoffersen_independence(realised, var):
    """Christoffersen independence test — do breaches cluster?

    H0: a breach today is independent of a breach yesterday. Returns
    (statistic, p_value). Small p => clustered exceptions (the failure that hurts
    in a crisis, and the one Kupiec cannot see).
    """
    b = breaches(realised, var)
    b0 = b[:-1]
    b1 = b[1:]
    n00 = int(np.sum((b0 == 0) & (b1 == 0)))
    n01 = int(np.sum((b0 == 0) & (b1 == 1)))
    n10 = int(np.sum((b0 == 1) & (b1 == 0)))
    n11 = int(np.sum((b0 == 1) & (b1 == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) else 0.0

    def _pow(base, exp):
        # 0**0 == 1 here, and log side handled by guarding below
        return base ** exp

    # Degenerate cases: no transitions of a type -> statistic is 0 (cannot reject).
    if pi in (0.0, 1.0) or (pi01 in (0.0,) and n01 == 0 and n11 == 0):
        return 0.0, 1.0
    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt = 0.0
    ll_alt += n00 * np.log(1 - pi01) if (1 - pi01) > 0 else 0.0
    ll_alt += n01 * np.log(pi01) if pi01 > 0 else 0.0
    ll_alt += n10 * np.log(1 - pi11) if (1 - pi11) > 0 else 0.0
    ll_alt += n11 * np.log(pi11) if pi11 > 0 else 0.0
    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    p = float(stats.chi2.sf(lr, df=1))
    return float(lr), p


def christoffersen_cc(realised, var, alpha: float):
    """Christoffersen conditional-coverage test = Kupiec + independence (df=2).

    H0: correct unconditional coverage AND independence. This is the gate a VaR model
    must pass, not just the breach count. Returns (statistic, p_value).
    """
    lr_pof, _, _, _ = kupiec_pof(realised, var, alpha)
    lr_ind, _ = christoffersen_independence(realised, var)
    lr = lr_pof + lr_ind
    p = float(stats.chi2.sf(lr, df=2))
    return float(lr), p


def _newey_west_var(d: np.ndarray, lag: int) -> float:
    """Long-run variance of series d via Bartlett (Newey-West) kernel."""
    d = d - d.mean()
    n = d.size
    gamma0 = np.dot(d, d) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = np.dot(d[k:], d[:-k]) / n
        lrv += 2.0 * w * gamma_k
    return float(lrv)


def diebold_mariano(loss_a, loss_b, lag: int = 5, alternative: str = "a_better"):
    """Diebold-Mariano test on two per-observation loss series.

    Loss differential d_t = loss_a - loss_b. HAC (Newey-West) standard error with the
    given Bartlett ``lag`` (bw=5 by default, matching the workspace convention). Harvey-
    Leybourne-Newbold small-sample correction applied.

    ``alternative``:
      - ``"a_better"``  H1: model A has lower loss  (one-sided, the usual claim)
      - ``"b_better"``  H1: model B has lower loss
      - ``"two_sided"`` H1: the losses differ

    Returns (dm_statistic, p_value). Negative statistic favours A.
    """
    la = _as_1d(loss_a)
    lb = _as_1d(loss_b)
    if la.shape != lb.shape:
        raise ValueError("loss series must be the same length")
    d = la - lb
    n = d.size

    # Degenerate case: the two models produced IDENTICAL forecasts, so the loss differential
    # is exactly zero. This is not an error — it is the correct answer "no difference", and it
    # happens legitimately whenever a selected hyper-parameter switches a component off (e.g.
    # an anchor weight of 0 makes the anchored model the unanchored one). Raising here would
    # discard exactly the most informative outcome.
    if not np.any(d):
        return 0.0, 0.5

    lrv = _newey_west_var(d, lag)
    if lrv <= 0:
        # The Bartlett kernel is positive semi-definite, so this is a numerical edge case
        # rather than a real negative variance. Fall back to the iid (lag-0) estimator, which
        # is non-negative by construction, instead of failing the whole run.
        lrv = float(np.var(d, ddof=1))
        if lrv <= 0:
            return 0.0, 0.5
    dm = d.mean() / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold correction
    corr = np.sqrt((n + 1 - 2 * lag + lag * (lag - 1) / n) / n)
    dm *= corr
    t = stats.t(df=n - 1)
    if alternative == "a_better":
        p = float(t.cdf(dm))            # reject when dm very negative
    elif alternative == "b_better":
        p = float(t.sf(dm))
    elif alternative == "two_sided":
        p = float(2 * t.sf(abs(dm)))
    else:
        raise ValueError("alternative must be 'a_better', 'b_better' or 'two_sided'")
    return float(dm), p
