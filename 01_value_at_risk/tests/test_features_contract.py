"""The frozen feature contract (docs/features.md) pinned against the code.

If one of these fails, either the code drifted from the pre-registered spec, or the spec
changed and needs a dated amendment in hypotheses.md plus a bump to the
'specifications evaluated' disclosure integer. Do not "fix" a test here to match new code
without doing that paperwork.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None

FROZEN_FEATURES = ("log_ret", "std", "mean")
FROZEN_WINDOW = 22


def _prices(n=400, seed=0):
    idx = pd.bdate_range("2016-06-30", periods=n)
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    df = pd.DataFrame({"price": price}, index=idx)
    df["log_ret"] = np.log(df["price"]).diff()
    return df.dropna()


def test_runner_uses_the_frozen_feature_set():
    # The runner must not quietly enable extra features.
    import run_experiment
    src = open(run_experiment.__file__, encoding="utf-8").read()
    assert 'features = ("log_ret", "std", "mean")' in src
    import run_batch_anchored
    src_b = open(run_batch_anchored.__file__, encoding="utf-8").read()
    assert 'FEATURES = ("log_ret", "std", "mean")' in src_b


@pytest.mark.skipif(not HAS_TORCH, reason="create_features builds torch tensors")
def test_feature_columns_count_and_target_alignment():
    from value_at_risk.models.deep_var.features import create_features

    df = _prices()
    data = create_features(df, alpha=0.05, rolling=FROZEN_WINDOW,
                           features=list(FROZEN_FEATURES))
    # exactly three input columns, per the frozen spec
    assert data["X"].shape[1] == 3
    assert data["X"].shape[0] == data["y"].shape[0] == len(data["dates"])

    # target is the NEXT day's return: y at row i equals log_ret at date i+1
    dates = pd.to_datetime(pd.Series(np.asarray(data["dates"])))
    y = data["y"].numpy().ravel()
    d0 = dates.iloc[0]
    pos = df.index.get_loc(d0)
    assert np.isclose(y[0], df["log_ret"].iloc[pos + 1], atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="create_features builds torch tensors")
def test_no_lookahead_first_feature_is_current_return():
    from value_at_risk.models.deep_var.features import create_features

    df = _prices()
    data = create_features(df, alpha=0.05, rolling=FROZEN_WINDOW,
                           features=list(FROZEN_FEATURES))
    dates = pd.to_datetime(pd.Series(np.asarray(data["dates"])))
    X = data["X"].numpy()
    pos = df.index.get_loc(dates.iloc[0])
    # feature 1 is the CURRENT day's return, not a future one
    assert np.isclose(X[0, 0], df["log_ret"].iloc[pos], atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="create_features builds torch tensors")
def test_warmup_rows_dropped():
    from value_at_risk.models.deep_var.features import create_features

    df = _prices(n=400)
    data = create_features(df, alpha=0.05, rolling=FROZEN_WINDOW,
                           features=list(FROZEN_FEATURES))
    # rolling warm-up (w-1) plus the final row (no next-day target) are removed
    assert len(data["dates"]) <= len(df) - FROZEN_WINDOW
    assert np.isfinite(data["X"].numpy()).all()
