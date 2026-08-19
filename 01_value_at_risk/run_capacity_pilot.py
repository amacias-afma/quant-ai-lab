"""VALIDATION-ONLY capacity pilot - does model capacity buy seed instability, and does the
anchor absorb it?

Why this script exists
----------------------
The hypothesis is: as capacity grows, a quantile network becomes unstable across random
initialisations (many parameters, little tail data), and the anchor's role is to stabilise it.
That hypothesis has a PREMISE that can be falsified before touching the test set:

    unanchored seed dispersion must INCREASE with capacity.

If it does not, there is nothing for the anchor to stabilise and the hypothesis is dead - no
test evaluation needed. Seed dispersion is a property of the fitting procedure, so it is
measurable on VALIDATION alone.

**This script never touches TEST.** It walk-forwards only over the VAL block (split at
``--train-end``, data truncated at ``--val-end``), exactly as ``select_anchor_weight`` does.
Running it therefore costs zero test-set evaluations and adds nothing to that disclosure
integer. Its output is a go/no-go signal, not a result to report.

    python run_capacity_pilot.py --tickers ^GSPC,NVDA --alpha 0.01 --seeds 10
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

# Capacity ladder: name, architecture, hidden_size, num_layers.
# Parameter counts for 3 input features: 4 -> 41 -> 1217 -> 8641.
CAPACITY_LADDER = [
    ("Linear",    "SimpleQuantileNeuron",  0,  0),
    ("MLP 8x1",   "QuantileMLP",           8,  1),
    ("MLP 32x2",  "QuantileMLP",          32,  2),
    ("MLP 64x3",  "QuantileMLP",          64,  3),
]


def n_params(arch: str, hidden: int, layers: int, n_features: int = 3) -> int:
    if arch == "SimpleQuantileNeuron":
        return n_features + 1
    total, d_in = 0, n_features
    for _ in range(max(1, layers)):
        total += d_in * hidden + hidden
        d_in = hidden
    return total + d_in + 1


def val_seed_dispersion(data, df, spec, weight, seeds, train_end, val_end):
    """Per-seed VAL pinball for one (spec, weight). Returns (median, iqr, values)."""
    val_data = _slice_upto(data, val_end)
    losses = []
    for s in seeds:
        _, realised, var = torch_fit_one(val_data, spec, s, weight, train_end, df)
        losses.append(scoring.pinball_loss(realised, var, spec.alpha))
    v = np.asarray(losses, dtype=float)
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return float(q50), float(q75 - q25), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="^GSPC,NVDA")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--train-end", default="2021-12-31")
    ap.add_argument("--val-end", default="2023-06-30")
    ap.add_argument("--rolling", type=int, default=22)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--weights", default="0,0.5",
                    help="anchor weights to probe; 0 is required as the reference")
    ap.add_argument("--anchor", default="param", choices=["param", "hist"])
    ap.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    ap.add_argument("--outdir", default="outputs/pilot_capacity")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    weights = [float(w) for w in args.weights.split(",")]
    seeds = list(range(args.seeds))
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("VALIDATION-ONLY capacity pilot - no TEST data is read by this script.")
    print(f"alpha={args.alpha}  seeds={args.seeds}  weights={weights}  anchor={args.anchor}")
    print("=" * 78)

    rows = []
    t0 = time.time()
    for ticker in tickers:
        df, data = prepare(ticker, args.end_date, args.alpha, args.rolling, FEATURES,
                           snapshot_dir=args.snapshot_dir)
        print(f"\n### {ticker}")
        for name, arch, hidden, layers in CAPACITY_LADDER:
            npar = n_params(arch, hidden, layers)
            for w in weights:
                spec = Spec(name=f"{name} w={w}", model_type=arch, alpha=args.alpha,
                            features=FEATURES, rolling=args.rolling, epochs=args.epochs,
                            lr=args.lr, hidden_size=max(1, hidden),
                            num_layers=max(1, layers),
                            anchor=args.anchor if w > 0 else None,
                            weight_grid=(w,))
                med, iqr, vals = val_seed_dispersion(
                    data, df, spec, w, seeds, args.train_end, args.val_end)
                rows.append(dict(ticker=ticker, capacity=name, n_params=npar, weight=w,
                                 val_pinball_median=med, val_pinball_iqr=iqr,
                                 rel_iqr=iqr / med if med else np.nan))
                print(f"  {name:10s} ({npar:5d} par) w={w:<5g}  "
                      f"VAL median={med:.6e}  IQR={iqr:.3e}  relIQR={iqr / med:.4f}",
                      flush=True)

                # Write after EVERY row. A long pilot must never be all-or-nothing: the
                # first version only saved at the end, so interrupting it would have thrown
                # away hours of completed fits and left no way to stop the run safely.
                pd.DataFrame(rows).to_csv(
                    os.path.join(args.outdir, f"capacity_pilot_a{args.alpha}.csv"),
                    index=False)

    out = pd.DataFrame(rows)
    csv = os.path.join(args.outdir, f"capacity_pilot_a{args.alpha}.csv")
    out.to_csv(csv, index=False)

    # ---- the go/no-go check -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PREMISE CHECK - does unanchored seed dispersion grow with capacity?")
    print("=" * 78)
    base = out[out.weight == 0].groupby("capacity", sort=False).rel_iqr.median()
    order = [c[0] for c in CAPACITY_LADDER if c[0] in base.index]
    vals = [base[c] for c in order]
    for c, v in zip(order, vals):
        print(f"  {c:10s} relative IQR (w=0): {v:.4f}")
    monotone = all(b >= a for a, b in zip(vals, vals[1:]))
    grew = vals[-1] > vals[0] if vals else False
    print(f"\n  monotonically increasing: {monotone}   (H5 strong form)")
    print(f"  largest > smallest      : {grew}   (H5a gate)")
    if not grew:
        print("\n  >>> PREMISE FAILS. Bigger models are not more seed-unstable here, so the")
        print("  >>> anchor has nothing to stabilise. Do NOT run the full panel; report this.")
    else:
        print("\n  >>> Premise holds. Now check whether the anchor absorbs the extra dispersion.")
        if not monotone:
            print("  >>> NOTE: the ordering is NOT monotone. H5 must be reported as the weaker")
            print("  >>> two-point claim (linear vs largest), not as a capacity ladder.")

    if len(weights) > 1:
        print("\n" + "=" * 78)
        print("DOES THE ANCHOR ABSORB IT? (IQR ratio w=0 / w>0, higher = more stabilising)")
        print("=" * 78)
        piv = out.pivot_table(index="capacity", columns="weight", values="val_pinball_iqr",
                              aggfunc="median")
        for w in weights[1:]:
            if w in piv.columns and 0.0 in piv.columns:
                for c in order:
                    r = piv.loc[c, 0.0] / piv.loc[c, w] if piv.loc[c, w] else np.nan
                    print(f"  {c:10s} w={w:<5g} IQR ratio = {r:.2f}x")
        print("\n  Prediction under the hypothesis: this ratio should GROW with capacity.")

    print(f"\nWrote {csv}   ({(time.time() - t0) / 60:.1f} min)")

    # Make the pre-registered gate executable: exit non-zero when the premise fails, so a
    # chained stage-B run cannot start by accident (e.g. overnight, unattended).
    # Exit codes are a SCIENTIFIC signal and must not be confused with a crash:
    #   0  -> gate passed, stage B may run
    #   10 -> gate failed on the evidence (premise false) -- a finding, report it
    #   anything else (1, 2, ...) -> the script broke; nothing was concluded
    # Using 1 for "premise failed" once let a UnicodeEncodeError masquerade as a result.
    if not grew:
        print("\nEXIT 10 - H5a gate FAILED ON THE EVIDENCE. Stage B must not run.")
        raise SystemExit(10)
    print("\nEXIT 0 - H5a gate passed. Stage B may proceed.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
