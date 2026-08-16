"""Tests for the pure-pandas classical benchmarks (parametric, historical)."""
import numpy as np
import pandas as pd

from value_at_risk.evaluation.benchmarks import parametric_forecast, historical_forecast


def _df(n=2500, seed=0):
    idx = pd.bdate_range("2016-06-30", periods=n)
    r = np.random.default_rng(seed).standard_normal(n) * 0.01
    return pd.DataFrame({"log_ret": r}, index=idx)


VAL_END = "2023-06-30"


def test_parametric_is_test_only_and_finite():
    df = _df()
    f = parametric_forecast(df, alpha=0.05, rolling=132, val_end=VAL_END)
    assert len(f.dates) > 0
    assert (pd.to_datetime(f.dates) > pd.Timestamp(VAL_END)).all()
    assert np.isfinite(f.var).all()
    assert np.isfinite(f.realised).all()


def test_historical_no_lookahead_alignment():
    df = _df()
    f = historical_forecast(df, alpha=0.05, window=252, val_end=VAL_END)
    # VaR is negative (a loss threshold) and realised is the *next* day return.
    assert (f.var < 0).mean() > 0.9
    # breach rate for a well-specified 5% VaR should be in a sane ballpark
    breach = (f.realised < f.var).mean()
    assert 0.0 < breach < 0.15


def test_parametric_more_conservative_at_99_than_95():
    df = _df()
    f95 = parametric_forecast(df, alpha=0.05, rolling=132, val_end=VAL_END)
    f99 = parametric_forecast(df, alpha=0.01, rolling=132, val_end=VAL_END)
    assert f99.var.mean() < f95.var.mean()      # 99% VaR is deeper
