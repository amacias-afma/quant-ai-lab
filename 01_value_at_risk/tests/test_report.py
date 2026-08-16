"""Tests for the ladder summary assembly (no torch/arch)."""
import numpy as np
import pandas as pd
import pytest

from value_at_risk.evaluation.harness import Forecast
from value_at_risk.evaluation.report import ladder_summary, align_forecasts


def _forecast(realised, var, dates, alpha=0.05):
    return Forecast(np.asarray(dates), np.asarray(realised), np.asarray(var), alpha)


def _panel(seed=0, n=800):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-07-03", periods=n).to_numpy()
    realised = rng.standard_normal(n) * 0.01
    alpha = 0.05
    q = np.quantile(realised, alpha)
    good = np.full(n, q)                      # sits on the true quantile -> low loss
    loose = np.full(n, q - 0.02)              # over-reserves -> higher loss
    tight = np.full(n, q + 0.015)             # under-reserves -> higher loss + breaches
    named = {
        "Good": _forecast(realised, good, dates, alpha),
        "Loose": _forecast(realised, loose, dates, alpha),
        "Tight": _forecast(realised, tight, dates, alpha),
    }
    return named


def test_align_requires_consistent_realised():
    dates = pd.bdate_range("2023-07-03", periods=10).to_numpy()
    a = _forecast(np.zeros(10), np.full(10, -0.01), dates)
    b = _forecast(np.ones(10), np.full(10, -0.01), dates)  # different realised
    with pytest.raises(ValueError):
        align_forecasts({"a": a, "b": b})


def test_summary_ranks_and_flags_mcs():
    named = _panel()
    df = ladder_summary(named, baseline_name="Loose", ticker="TEST", B=400, seed=1)
    # Ranked ascending by pinball -> the on-quantile model is first.
    assert df.iloc[0]["model"] == "Good"
    # The best model must be in the MCS.
    assert bool(df[df.model == "Good"]["in_mcs"].iloc[0]) is True
    # DM column present, and the baseline row has NaN DM.
    assert np.isnan(df[df.model == "Loose"]["dm_p_better_than_baseline"].iloc[0])
    # 'Good' should beat the 'Loose' baseline (one-sided DM small p).
    assert df[df.model == "Good"]["dm_p_better_than_baseline"].iloc[0] < 0.10


def test_summary_columns_present():
    named = _panel()
    df = ladder_summary(named, baseline_name="Loose", B=200)
    for col in ["ticker", "alpha", "model", "pinball", "in_mcs", "mcs_pvalue",
                "breach_rate", "kupiec_p", "christoffersen_ind_p", "passes_gate", "n_test"]:
        assert col in df.columns
