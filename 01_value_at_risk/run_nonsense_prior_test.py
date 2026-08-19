"""Falsification test: is the seed-variance reduction a finding, or just shrinkage?

Risk round 2 (docs/risk-review-G4-round2.md) established that the IQR ratio rises with the
selected anchor weight (Spearman rho = +0.585, p < 1e-4, n = 85), exactly as a pull toward a
FIXED target predicts. If the reduction is mechanical, it must appear just as strongly when the
target carries no risk information at all.

Priors compared, all at identical weights:

    param      real rolling Normal VaR                  (informative)
    hist       real rolling historical quantile         (informative)
    shuffled   param prior, time-permuted               (same scale, NO information)  <- key control
    constmean  a single constant at the prior's mean    (right magnitude, no variation)
    zero       shrink toward 0                          (no information, wrong scale)

Decision rule, fixed before running:

    If `shuffled` produces an IQR reduction of comparable magnitude to `param`, the variance
    reduction is a property of shrinkage and every variance claim in the paper is withdrawn.
    Only a clear separation (informative priors reducing dispersion materially more than the
    uninformative ones) leaves the claim alive.

**VALIDATION ONLY.** This script never reads TEST and costs zero test-set evaluations.

    python run_nonsense_prior_test.py --tickers ^GSPC,NVDA,SQM --alpha 0.01 --seeds 10
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from run_experiment import prepare
from value_at_risk.data.snapshot import DEFAULT_SNAPSHOT_DIR
from value_at_risk.evaluation import scoring
from value_at_risk.evaluation.harness import Spec, _slice_upto, torch_fit_one

FEATURES = ("log_ret", "std", "mean")
INFORMATIVE = ("param", "hist")
CONTROLS = ("shuffled", "constmean", "zero")


def val_dispersion(data, df, spec, weight, seeds, train_end, val_end):
    """Per-seed VAL pinball for one (spec, weight). Returns (median, iqr)."""
    val_data = _slice_upto(data, val_end)
    losses = []
    for s in seeds:
        _, realised, var = torch_fit_one(val_data, spec, s, weight, train_end, df)
        losses.append(scoring.pinball_loss(realised, var, spec.alpha))
    v = np.asarray(losses, dtype=float)
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return float(q50), float(q75 - q25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="^GSPC,NVDA,SQM")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--train-end", default="2021-12-31")
    ap.add_argument("--val-end", default="2023-06-30")
    ap.add_argument("--rolling", type=int, default=22)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--weights", default="0.5,1", help="anchor weights (w=0 baseline is automatic)")
    ap.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    ap.add_argument("--outdir", default="outputs/nonsense_prior_test")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    weights = [float(w) for w in args.weights.split(",")]
    seeds = list(range(args.seeds))
    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, f"nonsense_prior_a{args.alpha}.csv")

    print("=" * 78)
    print("FALSIFICATION TEST - is the variance reduction shrinkage or a finding?")
    print("VALIDATION ONLY. No TEST data is read; zero test-set evaluations.")
    print(f"alpha={args.alpha} seeds={args.seeds} weights={weights}")
    print("=" * 78)

    rows = []
    t0 = time.time()
    for ticker in tickers:
        df, data = prepare(ticker, args.end_date, args.alpha, args.rolling, FEATURES,
                           snapshot_dir=args.snapshot_dir)
        print(f"\n### {ticker}")

        base_spec = Spec(name="unanchored", model_type="SimpleQuantileNeuron", alpha=args.alpha,
                         features=FEATURES, rolling=args.rolling, epochs=args.epochs,
                         lr=args.lr, anchor=None, weight_grid=(0.0,))
        med0, iqr0 = val_dispersion(data, df, base_spec, 0.0, seeds,
                                    args.train_end, args.val_end)
        print(f"  {'w=0 (unanchored)':28s} VAL median={med0:.6e}  IQR={iqr0:.3e}")
        rows.append(dict(ticker=ticker, prior="none", weight=0.0, informative=None,
                         val_median=med0, val_iqr=iqr0, iqr_ratio=1.0))

        for prior in INFORMATIVE + CONTROLS:
            for w in weights:
                spec = Spec(name=f"{prior}-w{w}", model_type="SimpleQuantileNeuron",
                            alpha=args.alpha, features=FEATURES, rolling=args.rolling,
                            epochs=args.epochs, lr=args.lr, anchor=prior, weight_grid=(w,))
                med, iqr = val_dispersion(data, df, spec, w, seeds,
                                          args.train_end, args.val_end)
                ratio = iqr0 / iqr if iqr else np.inf
                rows.append(dict(ticker=ticker, prior=prior, weight=w,
                                 informative=prior in INFORMATIVE,
                                 val_median=med, val_iqr=iqr, iqr_ratio=ratio))
                tag = "INFO " if prior in INFORMATIVE else "CTRL "
                print(f"  {tag}{prior:10s} w={w:<5g} VAL median={med:.6e}  "
                      f"IQR={iqr:.3e}  ratio={ratio:6.2f}x", flush=True)
                pd.DataFrame(rows).to_csv(csv, index=False)

    out = pd.DataFrame(rows)
    anchored = out[out.prior != "none"]

    print("\n" + "=" * 78)
    print("RESULT - median IQR reduction ratio by prior")
    print("=" * 78)
    summ = anchored.groupby(["prior", "weight"]).iqr_ratio.median().unstack()
    print(summ.round(2).to_string())

    # NOTE: the column is object dtype (the baseline row carries None), so `~series`
    # would do a BITWISE complement and silently produce garbage indices. Cast first.
    is_info = anchored["informative"].astype(bool)
    info = anchored.loc[is_info, "iqr_ratio"].median()
    ctrl = anchored.loc[~is_info, "iqr_ratio"].median()
    print(f"\n  informative priors : median ratio {info:.2f}x")
    print(f"  control priors     : median ratio {ctrl:.2f}x")

    print("\n" + "=" * 78)
    if ctrl >= 0.5 * info:
        print("VERDICT: the uninformative controls shrink dispersion comparably.")
        print("  -> The variance reduction is a property of SHRINKAGE, not of the anchor.")
        print("  -> N2 is tautological. Withdraw every variance claim from the paper.")
        code = 10
    else:
        print("VERDICT: informative priors reduce dispersion materially more than controls.")
        print("  -> The reduction is not purely mechanical. N2 partially survives.")
        code = 0
    print("=" * 78)
    print(f"\nWrote {csv}   ({(time.time() - t0) / 60:.1f} min)")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
