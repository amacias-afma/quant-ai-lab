"""Tests for the Model Confidence Set."""
import numpy as np

from value_at_risk.evaluation.mcs import model_confidence_set


def test_dominant_model_survives_worst_excluded():
    # model 0 clearly best (low loss), model 3 clearly worst; two middling.
    rng = np.random.default_rng(0)
    n = 800
    base = rng.random((n, 1)) * 0.5
    L = np.hstack([
        base + 0.00,    # best
        base + 0.20,
        base + 0.25,
        base + 0.60,    # worst
    ])
    res = model_confidence_set(L, names=["best", "b", "c", "worst"], alpha=0.10, B=500, seed=1)
    assert res.in_mcs["best"] is True
    assert res.in_mcs["worst"] is False
    # worst has the smallest MCS p-value
    assert res.mcs_pvalue["worst"] == min(res.mcs_pvalue.values())


def test_all_equal_models_all_survive():
    # identical loss up to tiny noise -> none can be eliminated.
    rng = np.random.default_rng(2)
    n = 600
    base = rng.random((n, 1))
    L = np.hstack([base + 1e-9 * rng.standard_normal((n, 1)) for _ in range(4)])
    res = model_confidence_set(L, alpha=0.10, B=500, seed=3)
    assert set(res.surviving) == set(res.names)


def test_single_model_trivially_in_set():
    L = np.random.default_rng(4).random((100, 1))
    res = model_confidence_set(L, names=["only"], alpha=0.10)
    assert res.surviving == ["only"]
    assert res.mcs_pvalue["only"] == 1.0


def test_mean_loss_reported():
    L = np.array([[0.0, 1.0], [0.0, 3.0]])
    res = model_confidence_set(L, names=["a", "b"], B=100)
    assert np.isclose(res.mean_loss["a"], 0.0)
    assert np.isclose(res.mean_loss["b"], 2.0)


def test_mcs_pvalues_between_zero_and_one():
    rng = np.random.default_rng(5)
    L = rng.random((300, 5)) + np.arange(5) * 0.1
    res = model_confidence_set(L, alpha=0.10, B=300, seed=6)
    for p in res.mcs_pvalue.values():
        assert 0.0 <= p <= 1.0
