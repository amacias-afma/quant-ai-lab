"""Assemble the ranked ladder summary: pinball + Diebold-Mariano vs baseline + MCS + coverage.

This is what replaces the old pass/fail ``batch_summary.md``. It takes a dict of named
``Forecast`` objects (classical benchmarks + NN specs), aligns them on their common TEST dates,
and produces one ranked table per (ticker, alpha).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from value_at_risk.evaluation import scoring
from value_at_risk.evaluation.mcs import model_confidence_set

__all__ = ["align_forecasts", "ladder_summary"]


def align_forecasts(named: dict):
    """Inner-join forecasts on common dates.

    Returns (names, dates, realised_vec, var_matrix, loss_matrix). ``realised`` is shared
    across models (the market's next-day return); we verify the columns agree.
    """
    names = list(named.keys())
    alpha = next(iter(named.values())).alpha
    var_cols, real_cols = {}, {}
    for nm, f in named.items():
        idx = pd.to_datetime(np.asarray(f.dates))
        var_cols[nm] = pd.Series(np.asarray(f.var), index=idx)
        real_cols[nm] = pd.Series(np.asarray(f.realised), index=idx)
    var_df = pd.DataFrame(var_cols).dropna()
    dates = var_df.index
    real_df = pd.DataFrame(real_cols).loc[dates]
    # The realised next-day return must be identical across models on shared dates. If it is
    # not, some model is scored against a different target and the whole table is meaningless.
    # Tolerance allows float32 (torch) vs float64 (pandas) round-trips.
    ref = real_df.iloc[:, [0]].to_numpy()
    diffs = np.abs(real_df.to_numpy() - ref)
    if not np.allclose(real_df.to_numpy(), ref, rtol=1e-5, atol=1e-8, equal_nan=True):
        worst_col = int(np.nanargmax(np.nanmax(diffs, axis=0)))
        worst_row = int(np.nanargmax(diffs[:, worst_col]))
        raise ValueError(
            "realised returns differ across models on common dates — misaligned targets.\n"
            f"  reference model : {names[0]}\n"
            f"  worst mismatch  : {names[worst_col]} (max |diff| = "
            f"{np.nanmax(diffs[:, worst_col]):.3e})\n"
            f"  first bad date  : {pd.Timestamp(dates[worst_row]).date()} "
            f"({names[0]}={ref[worst_row, 0]:.6f} vs "
            f"{names[worst_col]}={real_df.to_numpy()[worst_row, worst_col]:.6f})\n"
            "  Usual cause: one path forgot the shift(-1), so it is scored against r_t "
            "instead of r_{t+1}."
        )
    realised = real_df.iloc[:, 0].to_numpy()
    V = var_df[names].to_numpy()
    L = np.column_stack([scoring.pinball_loss_series(realised, V[:, i], alpha)
                         for i in range(len(names))])
    return names, dates.to_numpy(), realised, V, L


def ladder_summary(
    named: dict, baseline_name: str, ticker: str = "", alpha_level: float | None = None,
    mcs_alpha: float = 0.10, dm_lag: int = 5, B: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """Ranked table over the full ladder. One row per model, sorted by mean pinball."""
    names, dates, realised, V, L = align_forecasts(named)
    alpha = alpha_level if alpha_level is not None else next(iter(named.values())).alpha
    mcs = model_confidence_set(L, names=names, alpha=mcs_alpha, B=B, seed=seed)
    base = names.index(baseline_name) if baseline_name in names else None

    rows = []
    for i, nm in enumerate(names):
        r, v = realised, V[:, i]
        _, kp, x, n = scoring.kupiec_pof(r, v, alpha)
        _, cip = scoring.christoffersen_independence(r, v)
        _, ccp = scoring.christoffersen_cc(r, v, alpha)
        if base is not None and i != base:
            dm, dmp = scoring.diebold_mariano(L[:, i], L[:, base], lag=dm_lag,
                                              alternative="a_better")
        else:
            dm, dmp = np.nan, np.nan
        rows.append({
            "ticker": ticker, "alpha": alpha, "model": nm,
            "pinball": float(L[:, i].mean()),
            "dm_vs_baseline": dm, "dm_p_better_than_baseline": dmp,
            "in_mcs": mcs.in_mcs[nm], "mcs_pvalue": mcs.mcs_pvalue[nm],
            "breach_rate": x / n if n else np.nan, "kupiec_p": kp,
            "christoffersen_ind_p": cip, "christoffersen_cc_p": ccp,
            "passes_gate": bool(kp > 0.05 and cip > 0.05),
            "n_test": int(n),
        })
    df = pd.DataFrame(rows).sort_values("pinball").reset_index(drop=True)
    df.attrs["baseline"] = baseline_name
    df.attrs["common_dates"] = int(len(dates))
    return df
