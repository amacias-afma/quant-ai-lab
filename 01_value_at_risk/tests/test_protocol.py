"""Tests for the split + multi-seed discipline."""
import numpy as np
import pytest

from value_at_risk.evaluation.protocol import (
    chronological_split,
    aggregate_seeds,
    run_multi_seed,
    SeedSummary,
    MIN_SEEDS,
)


def _dates(n, start="2015-01-01"):
    return np.array(np.datetime64(start) + np.arange(n), dtype="datetime64[ns]")


def test_split_is_chronological_and_partitions():
    d = _dates(1000)  # daily 2015-01-01 .. ~2017-09
    s = chronological_split(d, train_end="2016-06-30", val_end="2017-03-31")
    # No overlap, union covers everything, order preserved.
    assert s.train.max() < s.val.min() < s.val.max() < s.test.min()
    assert s.train.size + s.val.size + s.test.size == 1000
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert np.array_equal(np.sort(all_idx), np.arange(1000))


def test_split_rejects_unsorted_dates():
    d = _dates(100)
    d = d[::-1].copy()  # descending
    with pytest.raises(ValueError):
        chronological_split(d, "2015-02-01", "2015-03-01")


def test_split_rejects_bad_boundaries():
    d = _dates(100)
    with pytest.raises(ValueError):
        chronological_split(d, train_end="2015-03-01", val_end="2015-02-01")


def test_aggregate_requires_min_seeds():
    with pytest.raises(ValueError):
        aggregate_seeds([0.1, 0.2, 0.3])  # < MIN_SEEDS
    ok = aggregate_seeds([0.1] * MIN_SEEDS)
    assert isinstance(ok, SeedSummary)


def test_aggregate_median_iqr():
    vals = list(range(1, 12))  # 1..11, MIN_SEEDS+1 values, median 6
    s = aggregate_seeds(vals)
    assert s.median == 6.0
    assert s.q25 == 3.5 and s.q75 == 8.5
    assert np.isclose(s.iqr, 5.0)


def test_dominates_requires_whole_iqr_below_benchmark():
    s = aggregate_seeds([0.8, 0.85, 0.9, 0.82, 0.88, 0.79, 0.81, 0.86, 0.83, 0.84])
    assert s.dominates(1.0) is True       # whole IQR below benchmark -> real edge
    assert s.dominates(0.84) is False     # benchmark inside the seed spread -> noise


def test_run_multi_seed_calls_each_seed():
    seen = []

    def metric(seed):
        seen.append(seed)
        return 0.5 + 0.001 * seed

    summary = run_multi_seed(metric, seeds=range(MIN_SEEDS))
    assert seen == list(range(MIN_SEEDS))
    assert summary.n_seeds == MIN_SEEDS
