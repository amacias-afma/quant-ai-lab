"""Tests for the bootstrap intervals (Risk F7 / Editor E6)."""
import numpy as np
import pytest

from value_at_risk.evaluation.power import bootstrap_ci, ratio_report


def test_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    v = rng.lognormal(1.0, 0.5, 200)
    point, lo, hi = bootstrap_ci(v)
    assert lo <= point <= hi


def test_interval_narrows_with_sample_size():
    rng = np.random.default_rng(1)
    small = bootstrap_ci(rng.lognormal(1, 0.6, 12))
    rng = np.random.default_rng(1)
    large = bootstrap_ci(rng.lognormal(1, 0.6, 400))
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_covers_the_truth_at_roughly_the_nominal_rate():
    # A crude coverage check: the interval for the median should contain the true median
    # in most replications. Loose bound - this is a sanity test, not a proof.
    true_median = np.exp(1.0)
    hits = 0
    for s in range(60):
        rng = np.random.default_rng(100 + s)
        v = rng.lognormal(1.0, 0.5, 80)
        _, lo, hi = bootstrap_ci(v, B=800, seed=s)
        hits += lo <= true_median <= hi
    assert hits >= 48          # >= 80% of 60


def test_degenerate_inputs_do_not_crash():
    assert all(np.isnan(x) for x in bootstrap_ci([]))
    p, lo, hi = bootstrap_ci([3.0])
    assert p == lo == hi == 3.0
    p2, lo2, hi2 = bootstrap_ci([2.0, np.nan, np.inf])
    assert np.isfinite(p2)


def test_ratio_report_is_quotable_and_never_bare():
    r = ratio_report([1.2, 3.4, 13.5, 20.1, 5.5], label="anchored vs unanchored")
    for k in ("median", "ci_low", "ci_high", "n", "text"):
        assert k in r
    assert "95% CI" in r["text"]
    assert r["ci_low"] <= r["median"] <= r["ci_high"]


def test_reproducible_given_a_seed():
    rng = np.random.default_rng(3)
    v = rng.lognormal(1.0, 0.5, 60)
    # same seed -> identical interval
    assert bootstrap_ci(v, seed=7) == bootstrap_ci(v, seed=7)
    # different seed -> a slightly different interval, since resampling differs
    assert bootstrap_ci(v, seed=7) != bootstrap_ci(v, seed=8)


def test_small_samples_saturate_at_the_observed_range():
    # With n = 5 the percentile interval for the median cannot extend beyond the observed
    # min and max, so it is seed-independent. This is correct behaviour, not a bug - and it
    # is a warning that bootstrap intervals from tiny samples carry little information.
    v = [1.0, 2.0, 5.0, 9.0, 14.0]
    a, b = bootstrap_ci(v, seed=7), bootstrap_ci(v, seed=8)
    assert a == b
    assert a[1] >= min(v) and a[2] <= max(v)
