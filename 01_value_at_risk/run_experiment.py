"""Run the anchored-NN VaR study end to end and write the reported numbers to CSV.

    python run_experiment.py --ticker ^GSPC --alpha 0.01 \
        --train-end 2022-12-31 --val-end 2024-06-30 --seeds 10 --epochs 500

Every number in the paper comes from the CSV this writes - never typed by hand. NN specs are
run over >=10 seeds and reported as median + IQR. The anchor weight is chosen on VALIDATION
only; TEST is scored once.

Needs torch / arch / yfinance (they are in requirements.txt). The scoring and orchestration are
unit-tested separately in tests/ and do not need those packages.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from value_at_risk.data.snapshot import DEFAULT_SNAPSHOT_DIR
from value_at_risk.models.registry import available_models, get_model_info
from value_at_risk.evaluation.ledger import record_touch
from value_at_risk.evaluation.harness import (
    Spec, run_study, compare_to_baseline, torch_fit_one, sha256_of_frame,
    restrict_to_anchor_support,
)


def prepare(ticker: str, end_date: str, alpha: float, rolling: int, features: tuple,
            snapshot_dir: str = DEFAULT_SNAPSHOT_DIR):
    """Load the FROZEN price snapshot, compute log returns, build tensors and the anchor df.

    Reads the on-disk snapshot (downloading once if it does not exist yet) and verifies its
    sha256, so a rerun cannot silently train on different data.
    """
    # Imported lazily so spec-building stays usable without yfinance/torch installed.
    from value_at_risk.data.snapshot import get_prices
    from value_at_risk.models.deep_var.features import create_features

    df = get_prices(ticker, end_date, snapshot_dir=snapshot_dir)
    df = df.copy()
    df["log_ret"] = np.log(df["price"]).diff()
    df = df.dropna()
    data = create_features(df, alpha=alpha, rolling=rolling, features=list(features))

    # Every rung of the ladder must train on identical rows. The historical anchor needs a
    # 252-day warm-up, so without this the unanchored spec would silently get ~252 extra
    # training rows and the weight grid would not be comparing like with like.
    data, n_dropped = restrict_to_anchor_support(data, df, rolling=rolling, alpha=alpha)
    if n_dropped:
        print(f"[data] dropped {n_dropped} rows without anchor support "
              f"-> all specs now train on identical rows ({len(data['dates'])} left)")
    return df, data


def build_specs(alpha: float, features: tuple, rolling: int, epochs: int, lr: float,
                weight_grid: tuple, architectures: tuple = ("SimpleQuantileNeuron",),
                hidden_size: int = 32, num_layers: int = 2) -> list[Spec]:
    """Ladder for each requested architecture: unanchored ablation + the two anchored variants.

    Architectures come from ``models.registry``; adding one there makes it runnable here with
    no code change. Running more than one architecture also gives the nonlinearity ablation
    (e.g. QuantileMLP vs SimpleQuantileNeuron on identical features).
    """
    specs: list[Spec] = []
    for arch in architectures:
        get_model_info(arch)                      # fail fast on an unknown name
        tag = "" if len(architectures) == 1 else f" [{arch}]"
        common = dict(model_type=arch, alpha=alpha, features=features, rolling=rolling,
                      epochs=epochs, lr=lr, hidden_size=hidden_size, num_layers=num_layers)
        specs += [
            Spec(name=f"Unanchored{tag}", anchor=None, weight_grid=(0.0,), **common),
            Spec(name=f"Anchor param{tag}", anchor="param", weight_grid=weight_grid, **common),
            Spec(name=f"Anchor hist{tag}", anchor="hist", weight_grid=weight_grid, **common),
        ]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--end-date", default="2026-06-30", help="data snapshot end (10y back)")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--train-end", default="2021-12-31")  # pre-reg amendment 2026-08-13
    ap.add_argument("--val-end", default="2023-06-30")
    ap.add_argument("--rolling", type=int, default=22)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--weights", default="0,1,5,10", help="anchor-weight grid (VAL-selected)")
    ap.add_argument("--models", default="SimpleQuantileNeuron",
                    help=f"comma-separated architectures. available: {', '.join(available_models())}")
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR,
                    help="frozen price snapshots (hash-verified on load)")
    ap.add_argument("--val-seeds", type=int, default=None,
                    help="seeds used for anchor-weight selection on VAL "
                         "(default: same as --seeds; lower is much faster)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-fit progress")
    ap.add_argument("--selection-rule", choices=["argmin", "one_se"], default="argmin",
                    help="VAL weight rule. one_se takes the SMALLEST weight within one "
                         "standard error of the best (biases toward w=0, the nested "
                         "unanchored model)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 seeds, 50 epochs, 2 weights. Validates the pipeline "
                         "in ~1 min. NOT a reportable result.")
    ap.add_argument("--run-label", dest="run_label", default="run_experiment",
                    help="label recorded in the TEST-touch ledger")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.epochs, args.weights, args.val_seeds = 2, 50, "0,1", 2
        print("[quick] smoke-test settings - results are NOT reportable\n")

    features = ("log_ret", "std", "mean")
    weight_grid = tuple(float(w) for w in args.weights.split(","))
    architectures = tuple(m.strip() for m in args.models.split(",") if m.strip())
    seeds = list(range(args.seeds))
    os.makedirs(args.outdir, exist_ok=True)

    df, data = prepare(args.ticker, args.end_date, args.alpha, args.rolling, features,
                       snapshot_dir=args.snapshot_dir)
    data_hash = sha256_of_frame(df[["price", "log_ret"]])

    specs = build_specs(args.alpha, features, args.rolling, args.epochs, args.lr, weight_grid,
                        architectures=architectures, hidden_size=args.hidden_size,
                        num_layers=args.num_layers)

    def fit_one(d, spec, seed, weight, split_date, anchor_df):
        return torch_fit_one(d, spec, seed, weight, split_date, anchor_df)

    # NOTE: named ticker_tag, not tag - the architecture loop below uses its own arch_tag.
    # These collided once and silently dropped the ticker from the meta filename.
    ticker_tag = args.ticker.replace("^", "").replace("=", "")
    out_csv = os.path.join(args.outdir, f"{ticker_tag}_anchored_var_a{args.alpha}.csv")
    frame, results = run_study(
        data, specs, args.train_end, args.val_end, fit_one,
        anchor_df=df, seeds=seeds, out_csv=out_csv,
        val_seeds=list(range(args.val_seeds)) if args.val_seeds else None,
        enforce_min_seeds=not args.quick,
        verbose=not args.quiet, selection_rule=args.selection_rule,
    )

    record_touch(args.ticker, args.alpha, [r.spec.name for r in results],
                 n_seeds=args.seeds, run_label=args.run_label, purpose="exploratory")

    # H1: each anchored spec against the unanchored ablation of the SAME architecture.
    by_name = {r.spec.name: r for r in results}
    comparisons = []
    for arch in architectures:
        arch_tag = "" if len(architectures) == 1 else f" [{arch}]"
        base = by_name.get(f"Unanchored{arch_tag}")
        if base is None:
            continue
        for anch in (f"Anchor param{arch_tag}", f"Anchor hist{arch_tag}"):
            if anch in by_name:
                comparisons.append(compare_to_baseline(by_name[anch], base))

    meta = {
        "ticker": args.ticker, "alpha": args.alpha, "data_hash_sha256": data_hash,
        "train_end": args.train_end, "val_end": args.val_end,
        "split_sizes_train_val_test": frame.attrs["split_sizes"],
        "specifications_evaluated": frame.attrs["specifications_evaluated"],
        "test_set_evaluations": frame.attrs["test_set_evaluations"],
        "primary_hypothesis_H1": comparisons,
    }
    meta_path = os.path.join(args.outdir, f"{ticker_tag}_anchored_var_a{args.alpha}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n=== Ranked by TEST pinball (median over seeds) ===")
    print(frame.to_string(index=False))
    print(f"\nData hash: {data_hash[:16]}...  split (train/val/test) = {frame.attrs['split_sizes']}")
    print(f"Specifications evaluated: {frame.attrs['specifications_evaluated']}   "
          f"Test-set evaluations: {frame.attrs['test_set_evaluations']}")
    print("\n=== H1 - anchored vs unanchored (Diebold-Mariano) ===")
    for c in comparisons:
        print(f"  {c['anchored']} vs {c['baseline']}")
        print(f"    DM={c['dm_stat']:+.3f}  p(anchored better)={c['dm_p_anchored_better']:.4f}"
              f"  -> {'rejects' if c['dm_p_anchored_better'] < 0.05 else 'DOES NOT reject'} at 5%")
        print(f"    edge={c['edge']:+.3e}   anchored IQR={c['anchored_iqr']:.2e}   "
              f"baseline IQR={c['baseline_iqr']:.2e}")
        print(f"    H4 lenient  (edge > anchored IQR): {c['edge_exceeds_anchored_iqr']}")
        print(f"    H4 conserv. (edge > baseline IQR): {c['edge_exceeds_baseline_iqr']}")
        print(f"    -> seed-noise verdict: {c['seed_noise_verdict'].upper()}")


if __name__ == "__main__":
    main()
