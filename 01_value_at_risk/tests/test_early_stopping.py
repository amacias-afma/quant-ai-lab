"""Why the early-stopping rule had to change.

The old rule stopped as soon as two CONSECUTIVE epochs differed by < 1e-6 in relative terms.
The pinball objective of a linear model is piecewise linear, so it genuinely plateaus between
steps and the rule fired almost immediately. Adding the L2 anchor makes the objective strictly
convex and smooth, so the anchored model did NOT trip the rule and trained much longer.

Net effect: the anchor was buying training epochs, not statistical information. These tests
pin the difference between the two rules on a synthetic loss trace, with no torch needed.
"""
import numpy as np


def old_rule_stop_epoch(losses, tol=1e-6):
    """Epoch index where the consecutive-relative-change rule fires."""
    prev = np.inf
    for i, loss in enumerate(losses):
        if abs(1 - loss / (prev + 1e-8)) < tol:
            return i
        prev = loss
    return len(losses)


def new_rule_stop_epoch(losses, min_epochs=100, patience=50, tol=1e-9):
    """Epoch index where patience-on-best-loss fires (with a min_epochs floor)."""
    best = np.inf
    since = 0
    for i, loss in enumerate(losses):
        if loss < best - tol:
            best = loss
            since = 0
        else:
            since += 1
        if i + 1 >= min_epochs and since >= patience:
            return i
    return len(losses)


def piecewise_linear_trace(n=500, plateau_at=12):
    """Pinball-like: decreases, then sits on an exact plateau for a few steps, then improves."""
    out = []
    val = 1.0
    for i in range(n):
        if plateau_at <= i < plateau_at + 5:
            pass                      # exact plateau — the piecewise-linear kink
        else:
            val -= 0.001
        out.append(val)
    return out


def smooth_trace(n=500):
    """L2-anchored: strictly convex, smooth geometric decay — never exactly flat early."""
    return [1.0 * (0.99 ** i) + 1e-4 for i in range(n)]


def test_old_rule_stops_early_on_piecewise_linear_loss():
    losses = piecewise_linear_trace()
    stop = old_rule_stop_epoch(losses)
    assert stop < 20, f"expected a premature stop, got {stop}"


def test_old_rule_does_not_stop_early_on_smooth_loss():
    # The same rule lets the anchored (smooth) objective run far longer — this asymmetry is
    # exactly the bug: the anchor bought epochs.
    losses = smooth_trace()
    assert old_rule_stop_epoch(losses) > 100


def test_old_rule_asymmetry_is_large():
    piecewise = old_rule_stop_epoch(piecewise_linear_trace())
    smooth = old_rule_stop_epoch(smooth_trace())
    assert smooth > 5 * piecewise


def test_new_rule_treats_both_objectives_alike():
    # Both traces are still improving, so neither should stop before the floor.
    for losses in (piecewise_linear_trace(), smooth_trace()):
        assert new_rule_stop_epoch(losses) >= 100


def test_new_rule_respects_min_epochs():
    flat = [1.0] * 500                       # no improvement at all
    assert new_rule_stop_epoch(flat, min_epochs=100, patience=50) >= 99


def test_new_rule_still_stops_when_converged():
    converged = [1.0] * 300
    stop = new_rule_stop_epoch(converged, min_epochs=100, patience=50)
    assert stop < 300                        # it does eventually stop
