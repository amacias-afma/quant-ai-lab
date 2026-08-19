"""Golden tests for evaluation.scoring — the conventions that silently invert results.

Run:  PYTHONPATH=src python -m pytest tests/test_scoring.py -q
"""
import numpy as np
import pytest

from value_at_risk.evaluation.scoring import (
    pinball_loss,
    pinball_loss_series,
    breaches,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_cc,
    diebold_mariano,
)


def test_pinball_nonnegative():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    v = np.full_like(y, -1.6)
    assert np.all(pinball_loss_series(y, v, 0.05) >= 0)


def test_true_quantile_minimises_pinball():
    # The alpha-quantile of the sample minimises mean pinball loss. A forecast above or
    # below it must score worse. This is the property that makes pinball a valid ranking loss.
    rng = np.random.default_rng(1)
    y = rng.standard_normal(200_000)
    alpha = 0.05
    q_star = np.quantile(y, alpha)
    loss_star = pinball_loss(y, np.full_like(y, q_star), alpha)
    loss_hi = pinball_loss(y, np.full_like(y, q_star + 0.25), alpha)
    loss_lo = pinball_loss(y, np.full_like(y, q_star - 0.25), alpha)
    assert loss_star < loss_hi
    assert loss_star < loss_lo


def test_breach_sign_convention():
    # Breach == realised strictly below the VaR threshold. If this ever flips, every
    # coverage number inverts.
    realised = np.array([-0.10, -0.02, 0.01, -0.05])
    var = np.array([-0.03, -0.03, -0.03, -0.03])
    assert list(breaches(realised, var)) == [1, 0, 0, 1]


def test_kupiec_accepts_correct_rate_rejects_wrong_rate():
    rng = np.random.default_rng(2)
    n = 4000
    alpha = 0.05
    var = np.full(n, -1.6448536)          # 5% normal quantile
    y = rng.standard_normal(n)             # exactly 5% should fall below by construction
    _, p_ok, x, _ = kupiec_pof(y, var, alpha)
    assert abs(x / n - alpha) < 0.02
    assert p_ok > 0.05                     # cannot reject a correctly-specified model

    # A far-too-loose VaR (breaches ~ 0.5%) must be rejected against a 5% claim.
    var_loose = np.full(n, -2.576)
    _, p_bad, _, _ = kupiec_pof(y, var_loose, alpha)
    assert p_bad < 0.01


def test_clustered_exceptions_fool_kupiec_not_christoffersen():
    # 500 obs, 25 breaches (=5%, the nominal rate) but all consecutive. Kupiec is happy
    # about the count; Christoffersen independence must reject the clustering.
    n = 500
    alpha = 0.05
    var = np.full(n, -1.0)
    y = np.zeros(n)                        # above VaR everywhere by default (no breach)
    y[100:125] = -2.0                      # 25 consecutive breaches
    _, p_kupiec, x, _ = kupiec_pof(y, var, alpha)
    _, p_ind = christoffersen_independence(y, var)
    _, p_cc = christoffersen_cc(y, var, alpha)
    assert x == 25
    assert p_kupiec > 0.10                 # count looks fine
    assert p_ind < 0.01                    # but the clustering is caught
    assert p_cc < 0.05                     # joint gate fails


def test_dm_detects_better_model():
    # Model A has strictly lower loss every day -> 'a_better' rejects, 'b_better' does not.
    rng = np.random.default_rng(3)
    loss_b = rng.random(600) + 1.0
    loss_a = loss_b - 0.1
    dm, p_a = diebold_mariano(loss_a, loss_b, lag=5, alternative="a_better")
    _, p_b = diebold_mariano(loss_a, loss_b, lag=5, alternative="b_better")
    assert dm < 0
    assert p_a < 0.01
    assert p_b > 0.99


def test_dm_symmetric_and_twosided():
    rng = np.random.default_rng(4)
    loss_a = rng.random(500) + 1.0
    loss_b = loss_a + 0.05
    dm_ab, _ = diebold_mariano(loss_a, loss_b, alternative="a_better")
    dm_ba, _ = diebold_mariano(loss_b, loss_a, alternative="a_better")
    assert np.isclose(dm_ab, -dm_ba)
    _, p_two = diebold_mariano(loss_a, loss_b, alternative="two_sided")
    assert p_two < 0.05


def test_identical_models_give_no_difference_not_an_error():
    # Two identical forecasts -> loss differential is exactly zero. The right answer is
    # "no difference" (dm=0, p=0.5), NOT an exception. This case arises legitimately whenever
    # a selected weight switches a component off, and raising here killed half a real panel.
    rng = np.random.default_rng(11)
    loss = rng.random(500) + 1.0
    dm, p = diebold_mariano(loss, loss.copy(), lag=5, alternative="a_better")
    assert dm == 0.0
    assert p == 0.5
    for alt in ("b_better", "two_sided"):
        dm2, p2 = diebold_mariano(loss, loss.copy(), alternative=alt)
        assert dm2 == 0.0 and p2 == 0.5


def test_constant_loss_series_do_not_raise():
    # Zero-variance differential that is not identically zero.
    a = np.full(300, 2.0)
    b = np.full(300, 1.0)
    dm, p = diebold_mariano(a, b, lag=5, alternative="a_better")
    assert np.isfinite(dm)
    assert 0.0 <= p <= 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pinball_loss_series(np.zeros(3), np.zeros(4), 0.05)
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(3), np.zeros(4))
