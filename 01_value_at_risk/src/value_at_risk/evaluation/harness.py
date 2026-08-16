"""Experiment harness: run the anchored-NN VaR study honestly, end to end.

What this enforces (so the pipeline cannot quietly cheat):

- **Chronological split.** The anchor weight is selected on VALIDATION only; TEST is scored
  once. Uses ``protocol.chronological_split``.
- **Seed distribution.** Every NN spec is run over >= 10 seeds; the reported TEST score is the
  median with IQR (``protocol.aggregate_seeds``). Never the best seed.
- **Consistent scoring.** Ranking by pinball loss; pairwise comparison by Diebold-Mariano;
  coverage gate by Kupiec + Christoffersen (``scoring``).
- **Numbers come from a CSV**, written by ``run_study`` — never typed into a draft.

The model fit is injected as a ``fit_one`` callable so the orchestration is testable with a
deterministic stub (no torch). The real adapter, ``torch_fit_one``, wires the existing
``train_model`` and imports torch lazily, so this module imports fine without it.

fit_one contract
----------------
``fit_one(data, spec, seed, weight, split_date, anchor_df) -> (dates, realised, var)``
    Fit on the rows of ``data`` up to ``split_date`` and walk-forward predict every row after
    it. Returns three aligned 1-D arrays: forecast ``dates`` (datetime64), ``realised`` next-day
    returns, and the ``var`` forecast (return scale, negative).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from value_at_risk.evaluation import scoring
from value_at_risk.evaluation.protocol import (
    chronological_split,
    aggregate_seeds,
    SeedSummary,
    DEFAULT_SEEDS,
)

FitOne = Callable[[dict, "Spec", int, float, object, object],
                  tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Spec:
    """One model configuration under test."""
    name: str
    model_type: str = "SimpleQuantileNeuron"
    alpha: float = 0.01
    features: tuple = ("log_ret", "std", "mean")
    rolling: int = 22
    epochs: int = 500
    lr: float = 0.01
    hidden_size: int = 64
    num_layers: int = 1
    anchor: str | None = None                 # None | "param" | "hist"
    weight_grid: tuple = (0.0,)               # anchor-weight candidates, selected on VAL

    @property
    def is_anchored(self) -> bool:
        return self.anchor is not None and any(w > 0 for w in self.weight_grid)


@dataclass
class Forecast:
    """Aligned TEST-block forecast for one spec at one seed."""
    dates: np.ndarray
    realised: np.ndarray
    var: np.ndarray
    alpha: float

    def loss_series(self) -> np.ndarray:
        return scoring.pinball_loss_series(self.realised, self.var, self.alpha)

    def pinball(self) -> float:
        return scoring.pinball_loss(self.realised, self.var, self.alpha)

    def coverage(self) -> dict:
        _, p_kupiec, x, n = scoring.kupiec_pof(self.realised, self.var, self.alpha)
        _, p_ind = scoring.christoffersen_independence(self.realised, self.var)
        _, p_cc = scoring.christoffersen_cc(self.realised, self.var, self.alpha)
        return {
            "breaches": x, "n": n, "breach_rate": x / n if n else float("nan"),
            "kupiec_p": p_kupiec, "christoffersen_ind_p": p_ind, "christoffersen_cc_p": p_cc,
            "passes_gate": bool(p_kupiec > 0.05 and p_ind > 0.05),
        }


@dataclass
class SpecResult:
    spec: Spec
    test_summary: SeedSummary                 # over TEST pinball across seeds
    forecasts: list[Forecast]                 # one per seed (TEST)
    chosen_weight: float
    val_pinball: float

    @property
    def median_loss_series(self) -> np.ndarray:
        """Per-day loss series of the median-loss seed (for DM comparisons)."""
        losses = [f.pinball() for f in self.forecasts]
        med_idx = int(np.argsort(losses)[len(losses) // 2])
        return self.forecasts[med_idx].loss_series()

    @property
    def median_forecast(self) -> Forecast:
        losses = [f.pinball() for f in self.forecasts]
        med_idx = int(np.argsort(losses)[len(losses) // 2])
        return self.forecasts[med_idx]


def _slice_upto(data: dict, upto) -> dict:
    """Return a copy of the data dict keeping only rows with date <= upto.

    Uses positional integer indexing so it works for numpy arrays (tests) and torch
    tensors (real path) alike, and preserves a reset-index pandas Series for ``dates`` so
    the downstream ``train_model`` positional slicing and ``.loc`` anchor alignment still hold.
    """
    dates = pd.Series(pd.to_datetime(np.asarray(data["dates"])))
    idx = np.nonzero((dates <= pd.Timestamp(upto)).to_numpy())[0]
    out = dict(data)
    d = data["dates"]
    if isinstance(d, pd.Series):
        out["dates"] = d.iloc[idx].reset_index(drop=True)
    else:
        out["dates"] = np.asarray(d)[idx]
    out["X"] = data["X"][idx]
    out["y"] = data["y"][idx]
    return out


def select_anchor_weight(
    data: dict, spec: Spec, split, seeds: Sequence[int], fit_one: FitOne, anchor_df,
    train_end, val_end,
) -> tuple[float, float]:
    """Pick the anchor weight that minimises median VALIDATION pinball across seeds.

    Returns (chosen_weight, val_pinball_at_choice). Only VAL is touched here.
    """
    if not spec.anchor or spec.weight_grid == (0.0,):
        return 0.0, float("nan")

    val_data = _slice_upto(data, val_end)          # so walk-forward covers VAL only
    best_w, best_loss = 0.0, np.inf
    for w in spec.weight_grid:
        seed_losses = []
        for s in seeds:
            dates, realised, var = fit_one(val_data, spec, s, w, train_end, anchor_df)
            seed_losses.append(scoring.pinball_loss(realised, var, spec.alpha))
        med = float(np.median(seed_losses))
        if med < best_loss:
            best_w, best_loss = float(w), med
    return best_w, best_loss


def run_spec(
    data: dict, spec: Spec, split, fit_one: FitOne, anchor_df=None,
    seeds: Sequence[int] = DEFAULT_SEEDS, train_end=None, val_end=None,
    enforce_min_seeds: bool = True,
) -> SpecResult:
    """Select the weight on VAL, then produce the seed distribution of TEST forecasts."""
    chosen_w, val_loss = select_anchor_weight(
        data, spec, split, seeds, fit_one, anchor_df, train_end, val_end
    )
    forecasts: list[Forecast] = []
    for s in seeds:
        dates, realised, var = fit_one(data, spec, s, chosen_w, val_end, anchor_df)
        forecasts.append(Forecast(np.asarray(dates), np.asarray(realised),
                                  np.asarray(var), spec.alpha))
    summary = aggregate_seeds([f.pinball() for f in forecasts], enforce_min=enforce_min_seeds)
    return SpecResult(spec=spec, test_summary=summary, forecasts=forecasts,
                      chosen_weight=chosen_w, val_pinball=val_loss)


def compare_to_baseline(anchored: SpecResult, baseline: SpecResult, lag: int = 5) -> dict:
    """Diebold-Mariano: is the anchored spec's median-seed loss below the baseline's?

    Uses common dates only. Returns the DM statistic and one-sided p (H1: anchored better).
    """
    fa, fb = anchored.median_forecast, baseline.median_forecast
    da = pd.Series(fa.loss_series(), index=pd.to_datetime(fa.dates))
    db = pd.Series(fb.loss_series(), index=pd.to_datetime(fb.dates))
    common = da.index.intersection(db.index)
    dm, p = scoring.diebold_mariano(da.loc[common].to_numpy(), db.loc[common].to_numpy(),
                                    lag=lag, alternative="a_better")
    return {
        "anchored": anchored.spec.name, "baseline": baseline.spec.name,
        "anchored_median_pinball": anchored.test_summary.median,
        "baseline_median_pinball": baseline.test_summary.median,
        "dm_stat": dm, "dm_p_anchored_better": p, "n_common": int(len(common)),
        "edge_exceeds_seed_iqr": bool(anchored.test_summary.dominates(baseline.test_summary.median)),
    }


def results_frame(results: Sequence[SpecResult]) -> pd.DataFrame:
    """Long table of every spec's TEST outcome. This is what gets written to CSV."""
    rows = []
    for r in results:
        cov = r.median_forecast.coverage()
        rows.append({
            "spec": r.spec.name, "model": r.spec.model_type, "alpha": r.spec.alpha,
            "anchor": r.spec.anchor or "none", "chosen_weight": r.chosen_weight,
            "n_seeds": r.test_summary.n_seeds,
            "pinball_median": r.test_summary.median,
            "pinball_iqr": r.test_summary.iqr,
            "pinball_q25": r.test_summary.q25, "pinball_q75": r.test_summary.q75,
            "breach_rate": cov["breach_rate"], "kupiec_p": cov["kupiec_p"],
            "christoffersen_ind_p": cov["christoffersen_ind_p"],
            "christoffersen_cc_p": cov["christoffersen_cc_p"],
            "passes_gate": cov["passes_gate"],
        })
    return pd.DataFrame(rows).sort_values("pinball_median").reset_index(drop=True)


def run_study(
    data: dict, specs: Sequence[Spec], train_end, val_end, fit_one: FitOne,
    anchor_df=None, seeds: Sequence[int] = DEFAULT_SEEDS,
    out_csv: str | None = None, enforce_min_seeds: bool = True,
) -> tuple[pd.DataFrame, list[SpecResult]]:
    """Full study: split -> per-spec seed distributions -> ranked table (+ optional CSV).

    Disclosure integers are attached to the frame via ``.attrs``.
    """
    dates = np.asarray(data["dates"])
    split = chronological_split(pd.to_datetime(pd.Series(dates)).to_numpy(), train_end, val_end)
    results = [
        run_spec(data, spec, split, fit_one, anchor_df=anchor_df, seeds=seeds,
                 train_end=train_end, val_end=val_end, enforce_min_seeds=enforce_min_seeds)
        for spec in specs
    ]
    frame = results_frame(results)

    # Two disclosure integers (validation-protocol §7).
    n_specs_evaluated = sum(max(1, len(s.weight_grid) if s.anchor else 1) * len(seeds)
                            for s in specs)
    n_test_evaluations = len(specs) * len(seeds)
    frame.attrs["specifications_evaluated"] = int(n_specs_evaluated)
    frame.attrs["test_set_evaluations"] = int(n_test_evaluations)
    frame.attrs["train_end"] = str(train_end)
    frame.attrs["val_end"] = str(val_end)
    frame.attrs["split_sizes"] = split.sizes

    if out_csv:
        frame.to_csv(out_csv, index=False)
    return frame, results


# --------------------------------------------------------------------------- #
# Real model adapter — wires the existing train_model. torch imported lazily.  #
# --------------------------------------------------------------------------- #
def torch_fit_one(data, spec: Spec, seed: int, weight: float, split_date, anchor_df):
    """Adapter: run ``train_model`` for one (spec, seed, weight) and return aligned
    (dates, realised, var) over the walk-forward block after ``split_date``.
    """
    from value_at_risk.models.deep_var.train import train_model  # lazy: torch lives here

    reg = None
    if spec.anchor and weight > 0:
        reg = {"weight": float(weight), "df": anchor_df, "type": spec.anchor}

    _, _, _, (X_test_aux, y_test_aux, preds_test_aux, dates_test_aux) = train_model(
        data,
        model_type=spec.model_type,
        alpha=spec.alpha,
        epochs=spec.epochs,
        lr=spec.lr,
        rolling=spec.rolling,
        split_type={"date": pd.Timestamp(split_date).strftime("%Y-%m-%d")},
        regularization_pm=reg,
        hidden_size=spec.hidden_size,
        num_layers=spec.num_layers,
        silent=True,
        seed=seed,
    )
    var = np.concatenate([p.detach().cpu().numpy().ravel() for p in preds_test_aux])
    realised = np.concatenate([y.detach().cpu().numpy().ravel() for y in y_test_aux])
    dates = np.concatenate([np.asarray(pd.to_datetime(pd.Series(d)).to_numpy()) for d in dates_test_aux])
    return dates, realised, var


def sha256_of_frame(df: pd.DataFrame) -> str:
    """Stable hash of the input data, so a run is reproducible and the snapshot is pinned."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
