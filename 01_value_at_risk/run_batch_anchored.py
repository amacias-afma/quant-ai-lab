"""Rebuild the batch summary the honest way: full ladder, ranked by pinball, with
Diebold-Mariano vs the unanchored NN and Model Confidence Set membership.

Replaces the old pass/fail ``outputs/batch_summary.md``. Runs locally (needs torch / arch /
yfinance). The scoring, MCS and assembly are unit-tested separately in tests/.

    python run_batch_anchored.py --tickers ^GSPC,BTC-USD,TSLA --alphas 0.05,0.01 --seeds 10
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from run_experiment import prepare, build_specs
from value_at_risk.models.registry import available_models
from value_at_risk.evaluation.harness import run_study, torch_fit_one, sha256_of_frame
from value_at_risk.evaluation import benchmarks, report

FEATURES = ("log_ret", "std", "mean")


def run_one(ticker: str, alpha: float, args, weight_grid, architectures):
    df, data = prepare(ticker, args.end_date, alpha, args.rolling, FEATURES)
    data_hash = sha256_of_frame(df[["price", "log_ret"]])

    # DM baseline: the unanchored ablation of the first architecture.
    baseline = "Unanchored" if len(architectures) == 1 else f"Unanchored [{architectures[0]}]"

    specs = build_specs(alpha, FEATURES, args.rolling, args.epochs, args.lr, weight_grid,
                        architectures=architectures, hidden_size=args.hidden_size,
                        num_layers=args.num_layers)
    frame_nn, results = run_study(
        data, specs, args.train_end, args.val_end, torch_fit_one,
        anchor_df=df, seeds=list(range(args.seeds)),
    )
    named = {r.spec.name: r.median_forecast for r in results}

    named["Parametric-Normal"] = benchmarks.parametric_forecast(df, alpha, args.rolling, args.val_end)
    named["Historical"] = benchmarks.historical_forecast(df, alpha, args.hist_window, args.val_end)
    try:
        named["GARCH(1,1)-t"] = benchmarks.garch_forecast(df, alpha, args.val_end)
    except Exception as exc:                     # keep the panel going if one GARCH fit fails
        print(f"  [warn] GARCH failed for {ticker} a={alpha}: {exc}")

    summary = report.ladder_summary(
        named, baseline_name=baseline, ticker=ticker, alpha_level=alpha,
        B=args.bootstrap, seed=0,
    )
    summary.attrs["baseline"] = baseline
    n_classical = sum(k in named for k in ("Parametric-Normal", "Historical", "GARCH(1,1)-t"))
    meta = {
        "ticker": ticker, "alpha": alpha, "data_hash_sha256": data_hash,
        "split_sizes_train_val_test": frame_nn.attrs["split_sizes"],
        "specifications_evaluated": int(frame_nn.attrs["specifications_evaluated"] + n_classical),
        "test_set_evaluations": int(frame_nn.attrs["test_set_evaluations"] + n_classical),
    }
    return summary, meta


def _fmt(x, nd=6):
    if isinstance(x, float) and np.isnan(x):
        return "—"
    if isinstance(x, bool):
        return "✓" if x else "·"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def to_markdown(df: pd.DataFrame, baseline: str = "unanchored") -> str:
    cols = ["model", "pinball", "dm_p_better_than_baseline", "in_mcs", "mcs_pvalue",
            "breach_rate", "kupiec_p", "christoffersen_ind_p", "passes_gate"]
    out = []
    for (ticker, alpha), g in df.groupby(["ticker", "alpha"]):
        out.append(f"\n### {ticker} — α = {alpha}  (DM baseline: {baseline})\n")
        out.append("| " + " | ".join(cols) + " |")
        out.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in g.iterrows():
            out.append("| " + " | ".join(_fmt(r[c], 5) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="^GSPC,BTC-USD,TSLA,NVDA,SQM,CLP=X,HG=F,CL=F")
    ap.add_argument("--alphas", default="0.05,0.01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--train-end", default="2021-12-31")
    ap.add_argument("--val-end", default="2023-06-30")
    ap.add_argument("--rolling", type=int, default=22)
    ap.add_argument("--hist-window", type=int, default=252)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--weights", default="0,1,5,10")
    ap.add_argument("--models", default="SimpleQuantileNeuron",
                    help=f"comma-separated architectures. available: {', '.join(available_models())}")
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    alphas = [float(a) for a in args.alphas.split(",")]
    weight_grid = tuple(float(w) for w in args.weights.split(","))
    architectures = tuple(m.strip() for m in args.models.split(",") if m.strip())
    baseline_label = ("Unanchored" if len(architectures) == 1
                      else f"Unanchored [{architectures[0]}]")
    os.makedirs(args.outdir, exist_ok=True)

    all_summaries, all_meta = [], []
    for ticker in tickers:
        for alpha in alphas:
            print(f"=== {ticker}  α={alpha} ===")
            try:
                summary, meta = run_one(ticker, alpha, args, weight_grid, architectures)
            except Exception as exc:
                print(f"  [error] skipped {ticker} a={alpha}: {exc}")
                continue
            all_summaries.append(summary)
            all_meta.append(meta)

    if not all_summaries:
        print("No results produced.")
        return

    master = pd.concat(all_summaries, ignore_index=True)
    csv_path = os.path.join(args.outdir, "anchored_batch_summary.csv")
    md_path = os.path.join(args.outdir, "anchored_batch_summary.md")
    master.to_csv(csv_path, index=False)

    tot_specs = int(sum(m["specifications_evaluated"] for m in all_meta))
    tot_tests = int(sum(m["test_set_evaluations"] for m in all_meta))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Anchored-NN VaR — batch summary (ranked by pinball; DM vs unanchored; MCS)\n")
        f.write(f"\nSplit: TRAIN ≤ {args.train_end} · VAL ≤ {args.val_end} · TEST after.\n")
        f.write(f"\nArchitectures: {', '.join(architectures)}.\n")
        f.write(f"\n**Disclosure —** specifications evaluated: **{tot_specs}**, "
                f"test-set evaluations: **{tot_tests}**.\n")
        f.write(to_markdown(master, baseline_label))
    with open(os.path.join(args.outdir, "anchored_batch_meta.json"), "w") as f:
        json.dump({"runs": all_meta, "specifications_evaluated": tot_specs,
                   "test_set_evaluations": tot_tests}, f, indent=2, default=str)

    print(f"\nWrote {csv_path}\n      {md_path}")
    print(f"Specifications evaluated: {tot_specs}   Test-set evaluations: {tot_tests}")


if __name__ == "__main__":
    main()
