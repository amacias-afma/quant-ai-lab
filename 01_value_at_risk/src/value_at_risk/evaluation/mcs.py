"""Hansen-Lunde-Nachman Model Confidence Set (2011).

Given a matrix of per-period losses (one column per model), the MCS is the subset of models
that cannot be distinguished from the best at a given confidence level. It is the honest way
to compare more than two forecasts: instead of declaring a single winner, you report the set
that survives, with an MCS p-value per model.

Implementation: the elimination procedure with the range/`T_max` statistic and a moving-block
bootstrap (to respect serial dependence in the loss differentials). Pure numpy — testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ModelConfidenceSet", "model_confidence_set"]


def _moving_block_indices(n: int, B: int, block: int, rng) -> np.ndarray:
    """(B, n) resampled time indices using overlapping blocks of length ``block``."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(B, -1)[:, :n]
    return idx


@dataclass
class ModelConfidenceSet:
    names: list[str]
    mcs_pvalue: dict[str, float]
    in_mcs: dict[str, bool]
    mean_loss: dict[str, float]
    alpha: float

    @property
    def surviving(self) -> list[str]:
        return [m for m in self.names if self.in_mcs[m]]


def model_confidence_set(
    losses, names=None, alpha: float = 0.10, B: int = 1000, block: int | None = None,
    seed: int = 0,
) -> ModelConfidenceSet:
    """Compute the MCS at confidence ``1 - alpha``.

    ``losses``: (n_obs, n_models) per-period losses (e.g. pinball). Lower is better.
    Returns per-model MCS p-values and membership. A model is in the set iff its MCS p-value
    exceeds ``alpha``.
    """
    L = np.asarray(losses, dtype=float)
    if L.ndim != 2:
        raise ValueError("losses must be 2-D (n_obs, n_models)")
    n, m = L.shape
    if names is None:
        names = [f"m{i}" for i in range(m)]
    if len(names) != m:
        raise ValueError("names length must match number of model columns")
    mean_loss = {names[i]: float(L[:, i].mean()) for i in range(m)}

    if m == 1:
        return ModelConfidenceSet(list(names), {names[0]: 1.0}, {names[0]: True}, mean_loss, alpha)

    if block is None:
        block = max(1, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    boot_idx = _moving_block_indices(n, B, block, rng)  # (B, n)

    active = list(range(m))
    elim_p: dict[int, float] = {}
    running = 0.0

    while len(active) > 1:
        sub = L[:, active]                                  # n x k
        k = len(active)
        d = sub - sub.mean(axis=1, keepdims=True)           # per-period relative loss
        dbar = d.mean(axis=0)                               # k
        boot_means = d[boot_idx].mean(axis=1)               # B x k
        var = boot_means.var(axis=0, ddof=1)                # k
        sd = np.sqrt(var + 1e-24)
        t = dbar / sd                                       # k
        boot_t = (boot_means - dbar) / sd                   # B x k (centered)
        t_max = t.max()
        boot_tmax = boot_t.max(axis=1)
        p = float(np.mean(boot_tmax >= t_max))

        running = max(running, p)                           # MCS p-values are monotone
        worst_local = int(np.argmax(t))                     # highest relative loss = worst
        worst = active[worst_local]
        elim_p[worst] = running

        if p >= alpha:
            # cannot reject equal predictive ability -> everything left survives
            break
        active.remove(worst)

    # survivors (and the final round if it broke) take the last running p-value
    final_p = max(running, alpha if len(active) > 0 else running)
    mcs_p: dict[str, float] = {}
    for i in range(m):
        if i in elim_p and i not in active:
            mcs_p[names[i]] = elim_p[i]
        else:
            mcs_p[names[i]] = max(final_p, running)
    in_mcs = {nm: mcs_p[nm] >= alpha for nm in names}
    return ModelConfidenceSet(list(names), mcs_p, in_mcs, mean_loss, alpha)
