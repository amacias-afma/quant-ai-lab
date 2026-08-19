"""Tests for the anchor priors, including the falsification controls (Risk round 2).

The controls only work as controls if they are (a) genuinely uninformative and (b) IDENTICAL
across model seeds. If the shrinkage target varied by seed, the test would compare a moving
target against a fixed one and prove nothing.
"""
import numpy as np
import pandas as pd
import pytest

from value_at_risk.models.deep_var.parametric_model import (
    build_anchor_prior, INFORMATIVE_PRIORS, CONTROL_PRIORS,
)


def _df(n=800, seed=0):
    idx = pd.bdate_range("2016-06-30", periods=n)
    r = np.random.default_rng(seed).standard_normal(n) * 0.012
    return pd.DataFrame({"log_ret": r}, index=idx)


def _dates(df, skip=300):
    return df.index[skip:]


def test_informative_priors_are_finite_and_negative():
    df = _df()
    d = _dates(df)
    for p in INFORMATIVE_PRIORS:
        v = build_anchor_prior(df, d, rolling=22, alpha=0.05, anchor_type=p)
        assert np.isfinite(v).all()
        assert (v < 0).mean() > 0.9        # a VaR prior is a loss threshold


def test_zero_control_is_zero():
    df = _df(); d = _dates(df)
    v = build_anchor_prior(df, d, anchor_type="zero")
    assert np.allclose(v, 0.0)


def test_constmean_control_is_constant_and_on_scale():
    df = _df(); d = _dates(df)
    real = build_anchor_prior(df, d, anchor_type="param")
    v = build_anchor_prior(df, d, anchor_type="constmean")
    assert len(np.unique(v)) == 1                      # no time variation at all
    assert np.isclose(v[0], np.nanmean(real))          # right order of magnitude


def test_shuffled_control_keeps_distribution_but_destroys_alignment():
    df = _df(); d = _dates(df)
    real = build_anchor_prior(df, d, anchor_type="param")
    shuf = build_anchor_prior(df, d, anchor_type="shuffled")
    # same multiset of values -> same scale and marginal distribution
    assert np.allclose(np.sort(real), np.sort(shuf))
    # but not the same series -> time alignment destroyed
    assert not np.allclose(real, shuf)
    # and essentially uncorrelated with the real prior
    assert abs(np.corrcoef(real, shuf)[0, 1]) < 0.3


def test_controls_are_identical_across_calls():
    # The shrinkage target MUST be the same vector for every model seed. If this ever
    # becomes seed-dependent the falsification test is invalid.
    df = _df(); d = _dates(df)
    a = build_anchor_prior(df, d, anchor_type="shuffled")
    b = build_anchor_prior(df, d, anchor_type="shuffled")
    assert np.array_equal(a, b)


def test_unknown_prior_raises_with_options():
    df = _df(); d = _dates(df)
    with pytest.raises(ValueError, match="unknown anchor type"):
        build_anchor_prior(df, d, anchor_type="nope")


def test_control_priors_are_not_silently_treated_as_informative():
    assert set(INFORMATIVE_PRIORS).isdisjoint(CONTROL_PRIORS)
    assert "shuffled" in CONTROL_PRIORS and "param" in INFORMATIVE_PRIORS


def test_shuffled_carries_no_information_about_the_next_return():
    # The whole premise of the control: it must not predict tomorrow's tail.
    df = _df(n=1200, seed=3)
    d = _dates(df, 300)
    real = build_anchor_prior(df, d, anchor_type="param")
    shuf = build_anchor_prior(df, d, anchor_type="shuffled")
    nxt = df["log_ret"].shift(-1).loc[d].to_numpy()
    ok = np.isfinite(nxt)
    # the real prior tracks conditional scale; the shuffled one should not
    c_real = abs(np.corrcoef(real[ok], np.abs(nxt[ok]))[0, 1])
    c_shuf = abs(np.corrcoef(shuf[ok], np.abs(nxt[ok]))[0, 1])
    assert c_shuf < max(c_real, 0.05)
