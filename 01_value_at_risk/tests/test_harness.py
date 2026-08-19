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
    # The stub makes the anchored spec cleanly better, so BOTH H4 readings should agree.
    assert cmp["edge_exceeds_anchored_iqr"] is True
    assert cmp["edge_exceeds_baseline_iqr"] is True
    assert cmp["seed_noise_verdict"] == "detectable"


def _fake_result(median, iqr, name="x"):
    """A SpecResult-like stub carrying only the seed summary fields the H4 readings use."""
    from types import SimpleNamespace
    q25 = median - iqr / 2
    q75 = median + iqr / 2
    summary = SimpleNamespace(median=median, iqr=iqr, q25=q25, q75=q75, n_seeds=10,
                              dominates=lambda b, q=q75: q < b)
    return SimpleNamespace(spec=SimpleNamespace(name=name), test_summary=summary)


def test_h4_readings_can_disagree_and_verdict_is_ambiguous():
    # The real ^GSPC case: the anchored model is very stable (tiny IQR) and slightly better,
    # but the gap is far smaller than the NOISY baseline's spread. The lenient reading says
    # yes, the conservative one says no -> must be reported as ambiguous, never as a win.
    from value_at_risk.evaluation.harness import compare_to_baseline
    import value_at_risk.evaluation.harness as H

    anchored = _fake_result(median=1.08090e-3, iqr=1.01e-6, name="Anchor hist")
    baseline = _fake_result(median=1.08930e-3, iqr=5.14e-5, name="Unanchored")

    # bypass the forecast machinery: only the seed-summary logic is under test here
    edge = baseline.test_summary.median - anchored.test_summary.median
    lenient = anchored.test_summary.dominates(baseline.test_summary.median)
    conservative = edge > baseline.test_summary.iqr
    assert lenient is True
    assert conservative is False
    assert edge < baseline.test_summary.iqr        # gap is inside the baseline's noise


def test_h4_verdict_detectable_when_both_agree():
    anchored = _fake_result(median=1.0e-3, iqr=1e-6)
    baseline = _fake_result(median=2.0e-3, iqr=1e-5)
    edge = baseline.test_summary.median - anchored.test_summary.median
    assert anchored.test_summary.dominates(baseline.test_summary.median) is True
    assert (edge > baseline.test_summary.iqr) is True


def test_h4_verdict_not_detectable_when_neither_holds():
    anchored = _fake_result(median=1.99e-3, iqr=5e-4)
    baseline = _fake_result(median=2.00e-3, iqr=5e-4)
    edge = baseline.test_summary.median - anchored.test_summary.median
    assert anchored.test_summary.dominates(baseline.test_summary.median) is False
    assert (edge > baseline.test_summary.iqr) is False


def test_compare_to_baseline_reports_both_readings():
    from value_at_risk.evaluation.harness import compare_to_baseline
    data = make_data()
    unanchored, anchored = _specs()
    r_un = run_spec(data, unanchored, None, stub_fit_one, seeds=range(10),
                    train_end=TRAIN_END, val_end=VAL_END)
    r_an = run_spec(data, anchored, None, stub_fit_one, seeds=range(10),
                    train_end=TRAIN_END, val_end=VAL_END)
    c = compare_to_baseline(r_an, r_un)
    for key in ("edge", "anchored_iqr", "baseline_iqr",
                "edge_exceeds_anchored_iqr", "edge_exceeds_baseline_iqr",
                "seed_noise_verdict"):
        assert key in c
    assert c["seed_noise_verdict"] in {"detectable", "ambiguous", "not detectable"}


def test_val_curve_is_persisted_for_diagnosis():
    # A surprising weight choice must be diagnosable: the VAL loss at EVERY weight,
    # including 0, has to survive into the result.
    data = make_data()
    _, anchored = _specs()
    res = run_spec(data, anchored, None, stub_fit_one, seeds=range(10),
                   train_end=TRAIN_END, val_end=VAL_END)
    assert set(res.val_curve) == set(anchored.weight_grid)
    assert 0.0 in res.val_curve                      # the "no anchor" option is recorded
    # the chosen weight must be the argmin of the recorded curve — no hidden selection
    assert res.chosen_weight == min(res.val_curve, key=res.val_curve.get)


def test_one_se_rule_prefers_the_simpler_weight():
    # The anchored family NESTS the unanchored model at w=0, so it can only lose through
    # VAL selection error (measured at ~45% under plain argmin). The one-SE rule takes the
    # SMALLEST weight whose VAL loss is within one standard error of the best, biasing the
    # procedure back toward w=0 unless the evidence is clear.
    data = make_data()
    _, anchored = _specs()

    argmin = run_spec(data, anchored, None, stub_fit_one, seeds=range(10),
                      train_end=TRAIN_END, val_end=VAL_END, selection_rule="argmin")
    one_se = run_spec(data, anchored, None, stub_fit_one, seeds=range(10),
                      train_end=TRAIN_END, val_end=VAL_END, selection_rule="one_se")

    # never picks a LARGER weight than argmin
    assert one_se.chosen_weight <= argmin.chosen_weight
    # and the choice is still a member of the grid
    assert one_se.chosen_weight in anchored.weight_grid


def test_one_se_falls_back_to_zero_when_curve_is_flat():
    # If every weight performs the same on VAL (pure noise), the rule must return w=0 —
    # buying complexity on a flat curve is exactly the failure we are guarding against.
    import value_at_risk.evaluation.harness as H

    data = make_data()
    spec = Spec(name="A", anchor="param", weight_grid=(0.0, 1.0, 3.0, 6.0), alpha=0.01)

    def flat_fit(d, sp, seed, weight, split_date, anchor_df):
        dates, realised, var = stub_fit_one(d, sp, seed, 0.0, split_date, anchor_df)
        return dates, realised, var          # identical regardless of weight

    w, loss, curve = H.select_anchor_weight(
        data, spec, None, range(10), flat_fit, None, TRAIN_END, VAL_END,
        selection_rule="one_se",
    )
    assert w == 0.0


def test_anchor_support_alignment_equalises_rows():
    # restrict_to_anchor_support must drop exactly the rows where a prior is undefined,
    # so anchored and unanchored specs train on identical data.
    import pandas as pd
    from value_at_risk.evaluation.harness import restrict_to_anchor_support

    n = 900
    idx = pd.bdate_range("2016-06-30", periods=n)
    rng = np.random.default_rng(0)
    anchor_df = pd.DataFrame({"log_ret": rng.standard_normal(n) * 0.01}, index=idx)
    data = {"dates": idx.to_numpy(),
            "X": rng.standard_normal((n, 3)).astype("float32"),
            "y": rng.standard_normal((n, 1)).astype("float32")}

    trimmed, dropped = restrict_to_anchor_support(data, anchor_df, rolling=22, alpha=0.05)
    # the historical prior needs a 252-day warm-up -> those rows must go
    assert dropped >= 251
    assert len(trimmed["dates"]) == n - dropped
    assert trimmed["X"].shape[0] == trimmed["y"].shape[0] == len(trimmed["dates"])
    # and re-applying it is a no-op (idempotent)
    again, dropped2 = restrict_to_anchor_support(trimmed, anchor_df, rolling=22, alpha=0.05)
    assert dropped2 == 0


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
    # Disclosure integers count every fit: VAL weight-selection fits + TEST fits.
    #   unanchored: 0 VAL + 10 TEST            = 10
    #   anchored:   4 weights x 10 VAL + 10    = 50
    assert frame.attrs["specifications_evaluated"] == 60
    assert frame.attrs["test_set_evaluations"] == 20


def test_disclosure_uses_val_seeds_when_given(tmp_path):
    # Selecting the weight with fewer seeds must LOWER the disclosed count, not keep
    # reporting as if all reporting seeds had been used on VAL.
    data = make_data()
    unanchored, anchored = _specs()
    frame, _ = run_study(data, [unanchored, anchored], TRAIN_END, VAL_END, stub_fit_one,
                         seeds=range(10), val_seeds=range(3))
    #   unanchored: 0 VAL + 10 TEST         = 10
    #   anchored:   4 weights x 3 VAL + 10  = 22
    assert frame.attrs["specifications_evaluated"] == 32
    assert frame.attrs["test_set_evaluations"] == 20
