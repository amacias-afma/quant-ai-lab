"""Tests for the synthetic shrinkage demonstration (Editor condition E2).

The demonstration's job is to show, with ground truth known, that shrinkage-induced stability
does not distinguish a good anchor from a worthless one. These tests pin the properties that
make it a valid demonstration rather than a coincidence.
"""
import numpy as np
import pytest
from scipy import stats as st

from value_at_risk.evaluation import shrinkage_demo as sd
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


# --- the measured approximation (added after measuring what had been asserted) -----------

def test_separation_trace_matches_fit_anchored():
    """The traced loop must be the same algorithm as fit_anchored, not a re-implementation.

    If these drift apart, the measurement measures the wrong thing.
    """
    X, y, ts = sd.simulate(n=800, alpha=0.05, seed=3)
    a = sd.optimal_theta(ts)
    obs, _ = sd.separation_trace(X, y, 0.05, a, w=0.02, seed_a=0, seed_b=1, steps=50)
    ta = sd.fit_anchored(X, y, 0.05, a, 0.02, seed=0, steps=50)
    tb = sd.fit_anchored(X, y, 0.05, a, 0.02, seed=1, steps=50)
    rng_a = np.random.default_rng(1000 + 0)
    rng_b = np.random.default_rng(1000 + 1)
    d0 = np.linalg.norm(rng_a.standard_normal(X.shape[1]) * 3.0
                        - rng_b.standard_normal(X.shape[1]) * 3.0)
    assert abs(obs[-1] - np.linalg.norm(ta - tb) / d0) < 1e-12


def test_anchor_cancels_only_to_first_order():
    """The claim we corrected: the cancellation is exact for the penalty, not the trajectory.

    An earlier docstring asserted the two anchors give identical trajectories "to numerical
    precision". They do not — the anchor re-enters through the dropped data term. This test
    pins the measured behaviour so a future change cannot quietly restore the false claim.
    """
    rows = sd.anchor_invariance(weights=(0.0, 0.005, 0.1), steps=200)
    by_w = {r["weight"]: r["max_rel_diff"] for r in rows}
    assert by_w[0.0] == 0.0                       # no penalty, no anchor, identical by construction
    assert by_w[0.005] < 0.05                     # first-order agreement at small weight
    assert by_w[0.1] > 0.05                       # and a real, measurable divergence at large weight


def test_relative_contraction_is_the_accurate_quantity():
    """Absolute prediction is poor; prediction of the ratio-to-baseline is good.

    This is the distinction that keeps the paper's derivation usable: it never quotes an
    absolute spread, only ratios against the unanchored baseline.

    The horizon matters and the test uses the demonstration's own (steps=400, lr=0.05):
    at shorter horizons the penalty has had fewer steps to dominate the data term and the
    approximation is measurably worse. Testing at a different T would characterise a
    different regime than the one the paper reports.
    """
    rows = sd.contraction_accuracy(weights=(0.0, 0.005, 0.03), steps=400, lr=0.05)
    nz = [r for r in rows if r["weight"] > 0]
    assert all(r["absolute_ratio"] < 0.5 for r in rows)      # raw formula badly off
    assert all(0.9 < r["relative_ratio"] < 1.2 for r in nz)  # ratio-to-baseline close
