"""Honest evaluation protocol: chronological splits and multi-seed aggregation.

Two rules from the workspace standards, enforced here in code so the pipeline cannot
quietly break them:

1. **Chronological three-way split.** Hyperparameters and the anchor weight are chosen on
   VALIDATION only; TEST is scored once, after the design is frozen. No shuffling, ever.
2. **Neural results are a distribution over seeds.** Report median and IQR across >= MIN_SEEDS
   seeds, never the best seed. If the inter-seed IQR swamps the model-vs-benchmark gap, the
   honest conclusion is "no detectable difference".

Pure numpy — no torch — so the discipline is testable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

MIN_SEEDS = 10
DEFAULT_SEEDS = tuple(range(MIN_SEEDS))

__all__ = [
    "Split",
    "chronological_split",
    "SeedSummary",
    "aggregate_seeds",
    "run_multi_seed",
]


@dataclass(frozen=True)
class Split:
    """Index arrays for a chronological train/val/test partition."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        # Contiguity + ordering guarantee: train entirely before val entirely before test.
        if self.train.size and self.val.size and self.train.max() >= self.val.min():
            raise ValueError("train overlaps or postdates val — split is not chronological")
        if self.val.size and self.test.size and self.val.max() >= self.test.min():
            raise ValueError("val overlaps or postdates test — split is not chronological")
        if self.train.size and self.test.size and self.train.max() >= self.test.min():
            raise ValueError("train overlaps or postdates test — split is not chronological")

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.train.size, self.val.size, self.test.size


def chronological_split(dates, train_end, val_end) -> Split:
    """Split by calendar date. Everything <= ``train_end`` is TRAIN, everything in
    ``(train_end, val_end]`` is VAL, everything after ``val_end`` is TEST.

    ``dates`` must already be sorted ascending (a time series). Returns integer indices.
    """
    d = np.asarray(dates, dtype="datetime64[ns]")
    if np.any(np.diff(d).astype("int64") < 0):
        raise ValueError("dates must be sorted ascending; refusing to split unsorted data")
    train_end = np.datetime64(train_end)
    val_end = np.datetime64(val_end)
    if not (train_end < val_end):
        raise ValueError("need train_end < val_end")
    idx = np.arange(d.size)
    train = idx[d <= train_end]
    val = idx[(d > train_end) & (d <= val_end)]
    test = idx[d > val_end]
    return Split(train=train, val=val, test=test)


@dataclass(frozen=True)
class SeedSummary:
    """Median + IQR of a scalar metric across seeds, with the raw values kept."""
    median: float
    iqr: float
    q25: float
    q75: float
    n_seeds: int
    values: np.ndarray

    def dominates(self, benchmark: float) -> bool:
        """True only if the *whole* IQR beats the benchmark — i.e. the edge is not
        indistinguishable from seed noise. Lower loss is better."""
        return self.q75 < benchmark


def aggregate_seeds(values: Sequence[float], enforce_min: bool = True) -> SeedSummary:
    """Summarise a per-seed metric as median + IQR. Refuses fewer than MIN_SEEDS unless
    explicitly told otherwise (e.g. a smoke test)."""
    v = np.asarray(list(values), dtype=float)
    if enforce_min and v.size < MIN_SEEDS:
        raise ValueError(f"need >= {MIN_SEEDS} seeds for an honest NN result, got {v.size}")
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return SeedSummary(
        median=float(q50), iqr=float(q75 - q25), q25=float(q25), q75=float(q75),
        n_seeds=int(v.size), values=v,
    )


def run_multi_seed(
    metric_fn: Callable[[int], float],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    enforce_min: bool = True,
) -> SeedSummary:
    """Call ``metric_fn(seed)`` for each seed and aggregate the returned scalar metric.

    ``metric_fn`` should train the model under the given seed and return a single
    out-of-sample score (e.g. mean pinball on TEST). Kept deliberately generic so it wraps
    the existing ``train_model`` without depending on torch here.
    """
    scores = [float(metric_fn(int(s))) for s in seeds]
    return aggregate_seeds(scores, enforce_min=enforce_min)
