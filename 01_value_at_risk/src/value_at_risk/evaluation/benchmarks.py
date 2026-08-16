"""Classical VaR benchmarks as TEST-aligned Forecast objects, for the ladder / MCS.

Each function returns a ``harness.Forecast`` over the TEST block (dates strictly after
``val_end``). The VaR for day t+1 uses information available at close of t; the realised
target is the next-day return. No look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from value_at_risk.evaluation.harness import Forecast

__all__ = ["parametric_forecast", "historical_forecast", "garch_forecast"]


def _test_forecast(df: pd.DataFrame, var: pd.Series, alpha: float, val_end) -> Forecast:
    realised = df["log_ret"].shift(-1)                    # next-day return
    out = pd.DataFrame({"var": var, "realised": realised}).dropna()
    out = out[out.index > pd.Timestamp(val_end)]
    return Forecast(out.index.to_numpy(), out["realised"].to_numpy(),
                    out["var"].to_numpy(), alpha)


def parametric_forecast(df, alpha, rolling=132, val_end=None) -> Forecast:
    """Rolling Normal VaR: mu - z_alpha * sigma over a ``rolling``-day window."""
    z = abs(stats.norm.ppf(alpha))
    mu = df["log_ret"].rolling(rolling).mean()
    sd = df["log_ret"].rolling(rolling).std()
    return _test_forecast(df, mu - z * sd, alpha, val_end)


def historical_forecast(df, alpha, window=252, val_end=None) -> Forecast:
    """Rolling Historical Simulation VaR: empirical alpha-quantile over ``window`` days."""
    var = df["log_ret"].rolling(window).quantile(alpha)
    return _test_forecast(df, var, alpha, val_end)


def garch_forecast(df, alpha, val_end=None) -> Forecast:
    """GARCH(1,1)-t VaR (fixed quantile scaling). Imports arch lazily."""
    from value_at_risk.models.garch_model import calculate_garch_var
    var = calculate_garch_var(df["log_ret"], split_date=str(pd.Timestamp(val_end).date()),
                              alpha=alpha)
    var = var.reindex(df.index)
    return _test_forecast(df, var, alpha, val_end)
