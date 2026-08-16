"""Tests for the experiment harness — the split/VAL-selection/seed/scoring plumbing.

A deterministic stub replaces the NN so the orchestration is verified without torch.
The stub's VAL pinball is minimised at anchor weight = BEST_W, so weight selection has a
known right answer.
"""
import numpy as np
import pandas as pd
import pytest

from value_at_risk.evaluation.harness import (
    Spec, run_spec, run_study, compare_to_baseline, results_frame,
)
from value_at_risk.evaluation.protocol import chronological_split

BEST_W = 3.0
TRAIN_END = "2020-12-31"
VAL_END = "2022-06-30"


def make_data(n=2300, start="2015-01-01", seed=0):
    dates = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(seed)
    y = (rng.standard_normal((n, 1)) * 0.01).astype("float32")
    X = np.hstack([y, np.abs(y)]).astype("float32")
    return {"dates": dates.to_numpy(), "X": X, "y": y}


def stub_fit_one(data, spec, seed, weight, split_date, anchor_df):
    """Predict a constant VaR = train alpha-quantile, pushed away from optimal by a
    penalty that is zero at weight == BEST_W. Tiny seed jitter gives a nonzero IQR."""
    dates = pd.Series(pd.to_datetime(np.asarray(data["dates"])))
    after = (dates > pd.Timestamp(split_date)).to_numpy()
    upto = (dates <= pd.Timestamp(split_date)).to_numpy()
    d = np.asarray(data["dates"])[after]
    realised = np.asarray(data["y"]).ravel()[after]
    train_y = np.asarray(data["y"]).ravel()[upto]
    q = float(np.quantile(train_y, spec.alpha))
    penalty = 0.002 * (weight - BEST_W) ** 2          # 0 at the best weight
    var = np.full(d.shape[0], q + penalty) + 1e-7 * seed
    return d, realised, var


def _specs():
    unanchored = Spec(name="Unanchored NN", anchor=None, weight_grid=(0.0,), alpha=0.01)
    anchored = Spec(name="Anchor NN", anchor="param",
                    weight_grid=(0.0, 1.0, 3.0, 6.0), alpha=0.01)
    return unanchored, anchored


def test_split_sizes_partition():
    data = make_data()
    d = pd.to_datetime(pd.Series(data["dates"])).to_numpy()
    s = chronological_split(d, TRAIN_END, VAL_END)
    assert sum(s.sizes) == len(data["dates"])
    assert s.train.max() < s.val.min() < s.val.max() < s.test.min()


def test_weight_selected_on_val_is_best():
    data = make_data()
    _, anchored = _specs()
    seeds = range(10)
    res = run_spec(data, anchored, split=None, fit_one=stub_fit_one,
                   seeds=seeds, train_end=TRAIN_END, val_end=VAL_END)
    assert res.chosen_weight == BEST_W          # the harness found the VAL optimum


def test_seed_distribution_reported():
    data = make_data()
    unanchored, _ = _specs()
    res = run_spec(data, unanchored, split=None, fit_one=stub_fit_one,
                   seeds=range(10), train_end=TRAIN_END, val_end=VAL_END)
    assert res.test_summary.n_seeds == 10
    assert res.test_summary.iqr >= 0
    # TEST forecasts must be aligned to the post-VAL block only.
    fdates = pd.to_datetime(res.forecasts[0].dates)
    assert (fdates > pd.Timestamp(VAL_END)).all()


def test_anchored_beats_unanchored_via_dm():
    data = make_data()
    unanchored, anchored = _specs()
    r_un = run_spec(data, unanchored, None, stub_fit_one, seeds=range(10),
                    train_end=TRAIN_END, val_end=VAL_END)
    r_an = run_spec(data, anchored, None, stub_fit_one, seeds=range(10),
                    train_end=TRAIN_END, val_end=VAL_END)
    # By construction the anchored (best-weight) spec sits on the train quantile -> lower loss.
    assert r_an.test_summary.median < r_un.test_summary.median
    cmp = compare_to_baseline(r_an, r_un)
    assert cmp["dm_stat"] < 0
    assert cmp["dm_p_anchored_better"] < 0.05
    assert cmp["edge_exceeds_seed_iqr"] is True


def test_run_study_writes_csv_and_discloses_integers(tmp_path):
    data = make_data()
    unanchored, anchored = _specs()
    out = tmp_path / "anchored_var_results.csv"
    frame, results = run_study(
        data, [unanchored, anchored], TRAIN_END, VAL_END, stub_fit_one,
        seeds=range(10), out_csv=str(out),
    )
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert len(reloaded) == 2
    # Ranked ascending by pinball; anchored should rank first.
    assert frame.iloc[0]["spec"] == "Anchor NN"
    # Disclosure integers: unanchored 1*10 + anchored 4*10 = 50 specs; 2*10 = 20 test evals.
    assert frame.attrs["specifications_evaluated"] == 50
    assert frame.attrs["test_set_evaluations"] == 20
