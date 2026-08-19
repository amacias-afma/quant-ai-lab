"""Tests for the power / minimum-detectable-effect module (Risk condition R2)."""
import numpy as np
import pytest

from value_at_risk.evaluation.power import (
    binomial_power, binomial_mde, binomial_required_n,
    dm_standard_error, dm_mde, power_report,
)


def test_power_rises_with_n():
    p = [binomial_power(n, 0.3, 0.5) for n in (10, 20, 50, 100, 200)]
    assert all(b >= a for a, b in zip(p, p[1:]))
    assert p[-1] > 0.95


def test_power_rises_with_effect_size():
    p = [binomial_power(40, x, 0.5) for x in (0.50, 0.45, 0.40, 0.30, 0.20)]
    assert all(b >= a for a, b in zip(p, p[1:]))


def test_power_at_the_null_is_about_alpha():
    # Exact binomial is conservative at small n, so power at the null must not EXCEED alpha.
    assert binomial_power(50, 0.5, 0.5, alpha=0.05) <= 0.05 + 1e-9


def test_small_n_cannot_detect_a_modest_effect():
    # The project's actual situation: ~16 comparisons, true error 35% vs a 50% null.
    assert binomial_power(16, 0.35, 0.5) < 0.35


def test_required_n_matches_power_curve():
    n = binomial_required_n(0.3, power=0.8, p_null=0.5)
    assert binomial_power(n, 0.3, 0.5) >= 0.8
    assert binomial_power(n - 1, 0.3, 0.5) < 0.8


def test_mde_brackets_the_null():
    lo, hi = binomial_mde(60, power=0.8, p_null=0.5)
    assert lo < 0.5 < hi
    assert binomial_power(60, lo, 0.5) >= 0.8
    assert binomial_power(60, hi, 0.5) >= 0.8


def test_mde_is_unreachable_at_tiny_n():
    # With n = 5 no proportion is detectable at 80% power against a 0.5 null.
    lo, hi = binomial_mde(5, power=0.8, p_null=0.5)
    assert np.isnan(lo) or lo <= 0.0
    assert np.isnan(hi) or hi >= 1.0


def test_dm_se_matches_scoring_convention():
    # Independent series: HAC se should be close to the iid se.
    rng = np.random.default_rng(0)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    se = dm_standard_error(a, b, lag=5)
    iid = np.std(a - b, ddof=1) / np.sqrt(2000)
    assert 0.5 * iid < se < 2.0 * iid


def test_dm_mde_shrinks_with_sample_size():
    rng = np.random.default_rng(1)
    small = dm_mde(rng.standard_normal(200), rng.standard_normal(200))
    rng = np.random.default_rng(1)
    large = dm_mde(rng.standard_normal(4000), rng.standard_normal(4000))
    assert large < small


def test_power_report_flags_uninformative_null():
    # 6 of 16 (37.5%) against a 0.5 null: not far enough out to be detectable at n = 16.
    r = power_report(16, 6)
    assert r["p_value"] > 0.05
    assert r["informative"] is False
    assert r["n_for_80pct_at_observed"] > 16


def test_power_report_flags_informative_result():
    r = power_report(200, 60)          # 30% vs 50% at n = 200
    assert r["p_value"] < 0.01
    assert r["informative"] is True
