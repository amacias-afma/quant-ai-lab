"""Tests for the GARCH Student-t quantile fix.

The bug: arch fits a *standardized* (unit-variance) Student-t, but the code divided the raw
scipy t-quantile by sqrt((nu-2)/nu) instead of multiplying, inflating the tail by nu/(nu-2)
and producing absurd VaR (e.g. BTC -443% reserved). These tests pin the correct scaling.
"""
import numpy as np
import pytest
from scipy import stats

from value_at_risk.models.garch_model import standardized_t_quantile


def test_unit_variance_by_monte_carlo():
    # A standardized-t sample must have variance ~1; then its empirical 1% quantile should
    # match standardized_t_quantile(0.01, nu).
    nu = 6.0
    rng = np.random.default_rng(0)
    raw = rng.standard_t(nu, size=2_000_000)
    std_sample = raw * np.sqrt((nu - 2) / nu)
    assert abs(np.var(std_sample) - 1.0) < 0.02
    emp_q = np.quantile(std_sample, 0.01)
    assert abs(emp_q - standardized_t_quantile(0.01, nu)) < 0.05


def test_normal_limit():
    # As nu -> infinity the standardized t -> standard normal.
    for alpha in (0.05, 0.01):
        assert abs(standardized_t_quantile(alpha, 1e6) - stats.norm.ppf(alpha)) < 1e-3


def test_fix_direction_matches_multiply_not_divide():
    # The correct value multiplies by the scale factor (< 1), so its magnitude is SMALLER
    # than the raw t-quantile. The old buggy code (divide) produced a LARGER magnitude.
    nu, alpha = 4.0, 0.01
    scale = np.sqrt((nu - 2) / nu)
    correct = standardized_t_quantile(alpha, nu)
    raw = stats.t.ppf(alpha, df=nu)
    buggy = raw / scale
    assert np.isclose(correct, raw * scale)
    assert abs(correct) < abs(raw)          # standardization compresses toward unit variance
    assert abs(buggy) > abs(correct)        # the old path blew the quantile up
    # the inflation factor the bug introduced is exactly nu/(nu-2)
    assert np.isclose(abs(buggy) / abs(correct), nu / (nu - 2))


def test_more_extreme_tail_is_larger():
    nu = 5.0
    assert abs(standardized_t_quantile(0.01, nu)) > abs(standardized_t_quantile(0.05, nu))


def test_nu_near_two_is_guarded():
    # nu <= 2 has undefined variance; the helper must not divide-by-zero or return nan.
    q = standardized_t_quantile(0.01, 2.0)
    assert np.isfinite(q)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("arch") is None,
    reason="arch not installed; run locally for the end-to-end GARCH check",
)
def test_garch_var_is_sane_on_fat_tailed_series():
    # Regression guard for the blow-up: a fat-tailed return series must not yield a 99% VaR
    # deeper than -100% (i.e. reserving more than the whole position).
    import pandas as pd
    from value_at_risk.models.garch_model import calculate_garch_var

    rng = np.random.default_rng(1)
    nu = 3.5
    rets = pd.Series(
        rng.standard_t(nu, size=1500) * 0.02,
        index=pd.date_range("2015-01-01", periods=1500, freq="B"),
    )
    var = calculate_garch_var(rets, split_date="2019-01-01", alpha=0.01)
    assert var.notna().any()
    assert var.min() > -1.0                 # never reserve more than 100%
