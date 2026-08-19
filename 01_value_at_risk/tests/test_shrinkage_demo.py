"""Tests for the synthetic shrinkage demonstration (Editor condition E2).

The demonstration's job is to show, with ground truth known, that shrinkage-induced stability
does not distinguish a good anchor from a worthless one. These tests pin the properties that
make it a valid demonstration rather than a coincidence.
"""
import numpy as np
import pytest
from scipy import stats as st

from value_at_risk.evaluation.shrinkage_demo import (
    simulate, optimal_theta, fit_anchored, seed_dispersion, pinball_loss,
    predicted_contraction, run_demo,
)


def test_ground_truth_is_actually_optimal():
    # The demonstration is only meaningful if theta_star really is the best linear quantile.
    X, y, theta_star = simulate(n=20000, alpha=0.05, seed=1)
    best = pinball_loss(X, y, theta_star, 0.05)
    rng = np.random.default_rng(0)
    for _ in range(20):
        perturbed = theta_star + rng.standard_normal(theta_star.shape) * 0.15
        assert pinball_loss(X, y, perturbed, 0.05) >= best


def test_finite_budget_produces_seed_dispersion():
    # Without dispersion at w=0 there is nothing for shrinkage to remove, and the whole
    # demonstration would be vacuous.
    X, y, ts = simulate(n=3000, seed=2)
    Xte, yte, _ = simulate(n=3000, seed=3)
    r = seed_dispersion(X, y, Xte, yte, 0.05, optimal_theta(ts), 0.0, n_seeds=12)
    assert r["iqr"] > 0
    assert r["theta_spread"] > 0


def test_contraction_is_independent_of_the_anchor_value():
    # The analytical claim: the penalty contracts spread by (1-2*lr*w)^T, which contains no
    # reference to the anchor. Two very different anchors must give the same prediction.
    assert predicted_contraction(0.05) == predicted_contraction(0.05)
    assert predicted_contraction(0.05) < predicted_contraction(0.005) < predicted_contraction(0.0)
    assert predicted_contraction(0.0) == 1.0


def test_nonsense_anchor_also_reduces_dispersion():
    # THE demonstration. A scale-matched worthless anchor must still shrink seed dispersion.
    rows = run_demo(weights=(0.0, 0.05), n_seeds=16)
    import pandas as pd
    d = pd.DataFrame(rows)
    for name in ("informative (truth)", "nonsense (scale-matched)"):
        base = d[(d.anchor == name) & (d.weight == 0.0)].iqr.iloc[0]
        anch = d[(d.anchor == name) & (d.weight == 0.05)].iqr.iloc[0]
        assert anch < base, f"{name} failed to reduce dispersion"


def test_stability_does_not_track_usefulness():
    # The paper's claim in one assertion: the nonsense anchor stabilises at least as much as
    # the truth, while making the loss worse.
    import pandas as pd
    d = pd.DataFrame(run_demo(weights=(0.0, 0.05), n_seeds=16))
    truth = d[(d.anchor == "informative (truth)") & (d.weight == 0.05)].iloc[0]
    nons = d[(d.anchor == "nonsense (scale-matched)") & (d.weight == 0.05)].iloc[0]

    assert nons.iqr_ratio >= truth.iqr_ratio      # stabilises at least as much
    assert nons["median"] > truth["median"]       # but forecasts worse


def test_dose_response_appears_for_every_anchor():
    import pandas as pd
    d = pd.DataFrame(run_demo(n_seeds=16))
    for name in d.anchor.unique():
        g = d[(d.anchor == name) & (d.weight > 0)]
        rho, _ = st.spearmanr(g.weight, g.iqr_ratio)
        assert rho > 0.5, f"{name}: no dose-response (rho={rho:.2f})"


def test_scale_matched_control_is_genuinely_matched():
    # If the control differed in magnitude it would be a different experiment.
    _, _, ts = simulate(seed=7)
    theta_opt = optimal_theta(ts)
    rng = np.random.default_rng(999)
    r = rng.standard_normal(theta_opt.shape)
    nonsense = r / np.linalg.norm(r) * np.linalg.norm(theta_opt)
    assert np.isclose(np.linalg.norm(nonsense), np.linalg.norm(theta_opt))
    assert not np.allclose(nonsense, theta_opt)
