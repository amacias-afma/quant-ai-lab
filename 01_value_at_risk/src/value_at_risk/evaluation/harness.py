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
import time
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
    # VAL median pinball at every weight in the grid. Persisted so a surprising selection
    # (e.g. a boundary weight winning) can be diagnosed instead of guessed at.
    val_curve: dict = field(default_factory=dict)

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


def restrict_to_anchor_support(data: dict, anchor_df, rolling: int, alpha: float):
    """Trim the dataset to rows where BOTH anchor priors are defined.

    Why this exists: ``train_model`` drops rows whose anchor prior is NaN, but only when an
    anchor is actually in use. That made ``weight = 0`` (no anchor) train on ~252 more rows
    than ``weight > 0`` (historical prior needs a 252-day warm-up) — so the weight grid was
    not comparing like with like, and the unanchored spec silently got more training data
    than the anchored ones. That breaks the ladder's "identical data across rungs" rule and
    biases the anchored-vs-unanchored ablation.

    Applying this once, up front, makes every rung see exactly the same rows.
    Returns ``(trimmed_data, n_dropped)``.
    """
    from value_at_risk.models.deep_var.parametric_model import priori_value_at_risk

    dates = pd.Series(pd.to_datetime(np.asarray(data["dates"])))
    prior = priori_value_at_risk(anchor_df, rolling=rolling, alpha=alpha)
    cols = ["value_at_risk_param", "value_at_risk_hist"]
    sub = prior.loc[dates.to_numpy(), cols]
    valid = np.isfinite(sub.to_numpy()).all(axis=1)

    n_dropped = int((~valid).sum())
    if n_dropped == 0:
        return data, 0
    idx = np.nonzero(valid)[0]
    out = dict(data)
    d = data["dates"]
    out["dates"] = d.iloc[idx].reset_index(drop=True) if isinstance(d, pd.Series) else np.asarray(d)[idx]
    out["X"] = data["X"][idx]
    out["y"] = data["y"][idx]
    return out, n_dropped


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
    train_end, val_end, verbose: bool = False, selection_rule: str = "argmin",
) -> tuple[float, float, dict]:
    """Pick the anchor weight that minimises median VALIDATION pinball across seeds.

    Returns (chosen_weight, val_pinball_at_choice). Only VAL is touched here.
    """
    if not spec.anchor or spec.weight_grid == (0.0,):
        return 0.0, float("nan"), {}

    val_data = _slice_upto(data, val_end)          # so walk-forward covers VAL only
    best_w, best_loss = 0.0, np.inf
    curve: dict[float, float] = {}
    spread: dict[float, float] = {}
    for w in spec.weight_grid:
        t0 = time.time()
        seed_losses = []
        for s in seeds:
            dates, realised, var = fit_one(val_data, spec, s, w, train_end, anchor_df)
            seed_losses.append(scoring.pinball_loss(realised, var, spec.alpha))
        med = float(np.median(seed_losses))
        curve[float(w)] = med
        # standard error of the VAL loss across seeds, for the one-SE rule
        spread[float(w)] = float(np.std(seed_losses, ddof=1) / np.sqrt(len(seed_losses))) \
            if len(seed_losses) > 1 else 0.0
        if verbose:
            print(f"    [VAL] weight={w:<6g} median pinball={med:.6e}  "
                  f"({len(list(seeds))} seeds, {time.time() - t0:.1f}s)", flush=True)
        if med < best_loss:
            best_w, best_loss = float(w), med

    if selection_rule == "one_se" and curve:
        # One-standard-error rule: among weights whose VAL loss is within one SE of the best,
        # take the SMALLEST. The anchored family nests the unanchored model at w=0, so the
        # simpler model should win ties — otherwise a noisy VAL signal buys complexity for
        # nothing. Measured selection-error rate under plain argmin was ~45%.
        threshold = best_loss + spread.get(best_w, 0.0)
        eligible = [w for w, v in curve.items() if v <= threshold]
        if eligible:
            chosen = float(min(eligible))
            if verbose and chosen != best_w:
                print(f"    [VAL] one-SE rule: argmin was {best_w:g}, within 1 SE "
                      f"({threshold:.6e}) -> choosing simpler {chosen:g}", flush=True)
            best_w, best_loss = chosen, curve[chosen]

    if verbose:
        print(f"    [VAL] -> chosen weight={best_w:g}  (rule={selection_rule})", flush=True)
    return best_w, best_loss, curve


def run_spec(
    data: dict, spec: Spec, split, fit_one: FitOne, anchor_df=None,
    seeds: Sequence[int] = DEFAULT_SEEDS, train_end=None, val_end=None,
    enforce_min_seeds: bool = True, val_seeds: Sequence[int] | None = None,
    verbose: bool = False, selection_rule: str = "argmin",
) -> SpecResult:
    """Select the weight on VAL, then produce the seed distribution of TEST forecasts."""
    chosen_w, val_loss, val_curve = select_anchor_weight(
        data, spec, split, val_seeds if val_seeds is not None else seeds,
        fit_one, anchor_df, train_end, val_end, verbose=verbose,
        selection_rule=selection_rule,
    )
    forecasts: list[Forecast] = []
    t0 = time.time()
    for i, s in enumerate(seeds, 1):
        dates, realised, var = fit_one(data, spec, s, chosen_w, val_end, anchor_df)
        forecasts.append(Forecast(np.asarray(dates), np.asarray(realised),
                                  np.asarray(var), spec.alpha))
        if verbose:
            print(f"    [TEST] seed {i}/{len(list(seeds))} "
                  f"pinball={forecasts[-1].pinball():.6e}  "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)
    summary = aggregate_seeds([f.pinball() for f in forecasts], enforce_min=enforce_min_seeds)
    return SpecResult(spec=spec, test_summary=summary, forecasts=forecasts,
                      chosen_weight=chosen_w, val_pinball=val_loss, val_curve=val_curve)


def compare_to_baseline(anchored: SpecResult, baseline: SpecResult, lag: int = 5) -> dict:
    """Diebold-Mariano plus BOTH seed-noise readings of H4.

    H4 asks whether the model-vs-baseline gap exceeds "the inter-seed IQR". That phrase is
    ambiguous when the two models have very different seed dispersion, and the two readings
    can disagree — so this reports both rather than silently picking the flattering one:

    - ``edge_exceeds_anchored_iqr`` (lenient): the anchored spec's own q75 sits below the
      baseline's median. Answers "is the model under test reliably better?"
    - ``edge_exceeds_baseline_iqr`` (conservative): the gap is larger than the BASELINE's
      inter-seed IQR. Answers "is the gap bigger than the noise of the thing we compare to?"

    ``seed_noise_verdict`` is "detectable" only when both agree, "ambiguous" when they
    disagree, "not detectable" when neither holds. Report the verdict, not one reading.
    """
    fa, fb = anchored.median_forecast, baseline.median_forecast
    da = pd.Series(fa.loss_series(), index=pd.to_datetime(fa.dates))
    db = pd.Series(fb.loss_series(), index=pd.to_datetime(fb.dates))
    common = da.index.intersection(db.index)
    dm, p = scoring.diebold_mariano(da.loc[common].to_numpy(), db.loc[common].to_numpy(),
                                    lag=lag, alternative="a_better")

    edge = baseline.test_summary.median - anchored.test_summary.median   # >0 => anchored better

    # VAL can legitimately switch the anchor off (weight 0). Then the "anchored" spec IS the
    # unanchored one and the comparison is vacuous by construction — a finding in its own
    # right ("validation selected no anchoring"), not a result to average in with the rest.
    anchor_disabled = bool(anchored.chosen_weight == 0.0)
    identical = bool(np.allclose(
        anchored.median_forecast.var, baseline.median_forecast.var, rtol=0, atol=0
    )) if len(anchored.median_forecast.var) == len(baseline.median_forecast.var) else False

    lenient = bool(anchored.test_summary.dominates(baseline.test_summary.median))
    conservative = bool(edge > baseline.test_summary.iqr)
    if anchor_disabled or identical:
        verdict = "anchor disabled by VAL"
    elif lenient and conservative:
        verdict = "detectable"
    elif lenient or conservative:
        verdict = "ambiguous"
    else:
        verdict = "not detectable"

    return {
        "anchored": anchored.spec.name, "baseline": baseline.spec.name,
        "anchored_median_pinball": anchored.test_summary.median,
        "baseline_median_pinball": baseline.test_summary.median,
        "edge": edge,
        "anchored_iqr": anchored.test_summary.iqr,
        "baseline_iqr": baseline.test_summary.iqr,
        "dm_stat": dm, "dm_p_anchored_better": p, "n_common": int(len(common)),
        # Both readings of H4, always reported together.
        "edge_exceeds_anchored_iqr": lenient,
        "edge_exceeds_baseline_iqr": conservative,
        "seed_noise_verdict": verdict,
        "anchor_disabled_by_val": anchor_disabled,
        "identical_forecasts": identical,
    }


def results_frame(results: Sequence[SpecResult]) -> pd.DataFrame:
    """Long table of every spec's TEST outcome. This is what gets written to CSV."""
    rows = []
    for r in results:
        cov = r.median_forecast.coverage()
        rows.append({
            "spec": r.spec.name, "model": r.spec.model_type, "alpha": r.spec.alpha,
            "anchor": r.spec.anchor or "none", "chosen_weight": r.chosen_weight,
            "val_pinball_at_choice": r.val_pinball,
            "val_pinball_at_zero": r.val_curve.get(0.0, float("nan")),
            "val_curve": ";".join(f"{w:g}:{v:.6e}" for w, v in sorted(r.val_curve.items())),
            "n_seeds": r.test_summary.n_seeds,
            # Per-seed losses, persisted so a seed-level bootstrap is possible from the CSV
            # alone. Earlier runs stored only quartiles, which made the interval Risk F7
            # required impossible to compute without re-running the whole study.
            "pinball_per_seed": ";".join(f"{f.pinball():.9e}" for f in r.forecasts),
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
    val_seeds: Sequence[int] | None = None, verbose: bool = False,
    selection_rule: str = "argmin",
) -> tuple[pd.DataFrame, list[SpecResult]]:
    """Full study: split -> per-spec seed distributions -> ranked table (+ optional CSV).

    Disclosure integers are attached to the frame via ``.attrs``.
    """
    dates = np.asarray(data["dates"])
    split = chronological_split(pd.to_datetime(pd.Series(dates)).to_numpy(), train_end, val_end)
    if verbose:
        n_val = len(list(val_seeds if val_seeds is not None else seeds))
        n_fits = sum((len(s.weight_grid) * n_val if s.anchor else 0) + len(list(seeds))
                     for s in specs)
        print(f"[study] {len(specs)} specs · {len(list(seeds))} seeds · "
              f"~{n_fits} model fits · split {split.sizes} (train/val/test)", flush=True)

    results = []
    t_start = time.time()
    for i, spec in enumerate(specs, 1):
        if verbose:
            print(f"\n[{i}/{len(specs)}] {spec.name}  "
                  f"(model={spec.model_type}, anchor={spec.anchor or 'none'})", flush=True)
        results.append(
            run_spec(data, spec, split, fit_one, anchor_df=anchor_df, seeds=seeds,
                     train_end=train_end, val_end=val_end,
                     enforce_min_seeds=enforce_min_seeds, val_seeds=val_seeds,
                     verbose=verbose, selection_rule=selection_rule)
        )
    if verbose:
        print(f"\n[study] done in {time.time() - t_start:.1f}s", flush=True)
    frame = results_frame(results)

    # Two disclosure integers (validation-protocol §7).
    # NOTE: weight selection runs on VAL with val_seeds, which may differ from the reporting
    # seeds. Counting it with len(seeds) overstated the figure whenever --val-seeds was used.
    n_val_seeds = len(list(val_seeds if val_seeds is not None else seeds))
    n_specs_evaluated = sum(
        (len(s.weight_grid) * n_val_seeds if s.anchor else 0) + len(list(seeds))
        for s in specs
    )
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
