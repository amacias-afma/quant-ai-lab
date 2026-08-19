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
import time
import traceback

import numpy as np
import pandas as pd

from run_experiment import prepare, build_specs
from value_at_risk.data.snapshot import DEFAULT_SNAPSHOT_DIR
from value_at_risk.models.registry import available_models
from value_at_risk.evaluation.harness import (
    run_study, torch_fit_one, sha256_of_frame, compare_to_baseline,
)
from value_at_risk.evaluation import benchmarks, report
from value_at_risk.evaluation.ledger import record_touch

FEATURES = ("log_ret", "std", "mean")


def run_one(ticker: str, alpha: float, args, weight_grid, architectures):
    df, data = prepare(ticker, args.end_date, alpha, args.rolling, FEATURES,
                       snapshot_dir=args.snapshot_dir)
    data_hash = sha256_of_frame(df[["price", "log_ret"]])

    # DM baseline: the unanchored ablation of the first architecture.
    baseline = "Unanchored" if len(architectures) == 1 else f"Unanchored [{architectures[0]}]"

    specs = build_specs(alpha, FEATURES, args.rolling, args.epochs, args.lr, weight_grid,
                        architectures=architectures, hidden_size=args.hidden_size,
                        num_layers=args.num_layers)
    frame_nn, results = run_study(
        data, specs, args.train_end, args.val_end, torch_fit_one,
        anchor_df=df, seeds=list(range(args.seeds)),
        val_seeds=list(range(args.val_seeds)) if args.val_seeds else None,
        verbose=not args.quiet, selection_rule=args.selection_rule,
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

    # H1 / H4: each anchored spec against the unanchored ablation of the SAME architecture,
    # carrying BOTH seed-noise readings (pre-registration amendment 2026-08-16).
    by_name = {r.spec.name: r for r in results}
    verdicts = []
    for arch in architectures:
        arch_tag = "" if len(architectures) == 1 else f" [{arch}]"
        base_res = by_name.get(f"Unanchored{arch_tag}")
        if base_res is None:
            continue
        for anch in (f"Anchor param{arch_tag}", f"Anchor hist{arch_tag}"):
            if anch not in by_name:
                continue
            c = compare_to_baseline(by_name[anch], base_res)
            c.update({"ticker": ticker, "alpha": alpha, "architecture": arch})
            # A claim of improvement needs BOTH: a DM rejection and a 'detectable' verdict.
            # An anchor switched off by VAL can never support the claim, by construction.
            c["claim_supported"] = bool(c["dm_p_anchored_better"] < 0.05
                                        and c["seed_noise_verdict"] == "detectable"
                                        and not c["anchor_disabled_by_val"])
            # Flag when VAL picked a grid endpoint (optimum lies outside the grid).
            curve = by_name[anch].val_curve
            c["val_weight_at_grid_edge"] = bool(
                curve and by_name[anch].chosen_weight == max(curve)
            )
            c["chosen_weight"] = by_name[anch].chosen_weight
            verdicts.append(c)

    # Record the TEST touch BEFORE assembling results. On a protected (holdout) cell this
    # raises, so a second scoring pass cannot happen by accident or by forgetting.
    record_touch(ticker, alpha, [r.spec.name for r in results], n_seeds=args.seeds,
                 run_label=getattr(args, "run_label", os.path.basename(args.outdir)),
                 purpose=getattr(args, "purpose", "exploratory"))
    n_classical = sum(k in named for k in ("Parametric-Normal", "Historical", "GARCH(1,1)-t"))
    meta = {
        "ticker": ticker, "alpha": alpha, "data_hash_sha256": data_hash,
        "split_sizes_train_val_test": frame_nn.attrs["split_sizes"],
        "specifications_evaluated": int(frame_nn.attrs["specifications_evaluated"] + n_classical),
        "test_set_evaluations": int(frame_nn.attrs["test_set_evaluations"] + n_classical),
        "h1_h4_verdicts": verdicts,
    }
    return summary, meta, verdicts


def _fmt(x, nd=6):
    if isinstance(x, float) and np.isnan(x):
        return "-"
    if isinstance(x, bool):
        return "OK" if x else "."
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def to_markdown(df: pd.DataFrame, baseline: str = "unanchored") -> str:
    cols = ["model", "pinball", "dm_p_better_than_baseline", "in_mcs", "mcs_pvalue",
            "breach_rate", "kupiec_p", "christoffersen_ind_p", "passes_gate"]
    out = []
    for (ticker, alpha), g in df.groupby(["ticker", "alpha"]):
        out.append(f"\n### {ticker} - alpha = {alpha}  (DM baseline: {baseline})\n")
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
    ap.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR,
                    help="frozen price snapshots (hash-verified on load)")
    ap.add_argument("--val-seeds", type=int, default=None,
                    help="seeds for anchor-weight selection on VAL (default: same as --seeds)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-fit progress")
    ap.add_argument("--selection-rule", choices=["argmin", "one_se"], default="argmin",
                    help="VAL weight rule (argmin or one-standard-error)")
    ap.add_argument("--run-label", dest="run_label", default=None,
                    help="label recorded in the TEST-touch ledger (defaults to outdir name)")
    ap.add_argument("--purpose", default="exploratory",
                    choices=["exploratory", "confirmatory"],
                    help="recorded in the ledger; confirmatory runs are the ones the paper cites")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    if args.run_label is None:
        args.run_label = os.path.basename(os.path.normpath(args.outdir))

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    alphas = [float(a) for a in args.alphas.split(",")]
    weight_grid = tuple(float(w) for w in args.weights.split(","))
    architectures = tuple(m.strip() for m in args.models.split(",") if m.strip())
    baseline_label = ("Unanchored" if len(architectures) == 1
                      else f"Unanchored [{architectures[0]}]")
    os.makedirs(args.outdir, exist_ok=True)

    csv_path = os.path.join(args.outdir, "anchored_batch_summary.csv")
    verdict_path = os.path.join(args.outdir, "anchored_batch_verdicts.csv")

    fail_log = os.path.join(args.outdir, "anchored_batch_failures.log")
    fail_csv = os.path.join(args.outdir, "anchored_batch_failures.csv")
    for stale in (fail_log, fail_csv):
        if os.path.exists(stale):
            os.remove(stale)

    all_summaries, all_meta, all_verdicts, failures = [], [], [], []
    combos = [(t, a) for t in tickers for a in alphas]
    t_batch = time.time()
    for n, (ticker, alpha) in enumerate(combos, 1):
        elapsed = time.time() - t_batch
        eta = f", ETA ~{elapsed / (n - 1) * (len(combos) - n + 1) / 60:.0f} min" if n > 1 else ""
        print(f"\n=== [{n}/{len(combos)}] {ticker}  alpha={alpha}  "
              f"(elapsed {elapsed / 60:.1f} min{eta}) ===", flush=True)
        try:
            summary, meta, verdicts = run_one(ticker, alpha, args, weight_grid, architectures)
        except Exception as exc:
            # A silently dropped combination biases the panel. Record the full traceback so
            # the failure is diagnosable after the run, not just a line lost in the console.
            tb = traceback.format_exc()
            failures.append({"ticker": ticker, "alpha": alpha,
                             "error_type": type(exc).__name__, "error": str(exc),
                             "traceback": tb})
            print(f"  [ERROR] {ticker} a={alpha} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            with open(fail_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n{'=' * 70}\n{ticker}  alpha={alpha}\n{'=' * 70}\n{tb}\n")
            pd.DataFrame(failures).drop(columns=["traceback"]).to_csv(fail_csv, index=False)
            continue
        all_summaries.append(summary)
        all_meta.append(meta)
        all_verdicts.extend(verdicts)

        # Write after every combination: a 30-90 min batch must not lose everything on a
        # late failure, and partial results stay inspectable while it runs.
        pd.concat(all_summaries, ignore_index=True).to_csv(csv_path, index=False)
        if all_verdicts:
            pd.DataFrame(all_verdicts).to_csv(verdict_path, index=False)

    if failures:
        print("\n" + "!" * 70)
        print(f"!! {len(failures)} of {len(combos)} combinations FAILED - the panel is "
              f"INCOMPLETE.")
        print("!! Results below are NOT a panel result until these are fixed or the")
        print("!! exclusions are justified and disclosed. Surviving combinations may be a")
        print("!! biased subset (e.g. if failures track volatility).")
        for f in failures:
            print(f"!!   {f['ticker']:8s} a={f['alpha']:<5g} {f['error_type']}: {f['error']}")
        print(f"!! Full tracebacks: {fail_log}")
        print("!" * 70)

    if not all_summaries:
        print("No results produced.")
        return

    master = pd.concat(all_summaries, ignore_index=True)
    md_path = os.path.join(args.outdir, "anchored_batch_summary.md")
    master.to_csv(csv_path, index=False)
    vdf = pd.DataFrame(all_verdicts) if all_verdicts else pd.DataFrame()
    if not vdf.empty:
        vdf.to_csv(verdict_path, index=False)

    tot_specs = int(sum(m["specifications_evaluated"] for m in all_meta))
    tot_tests = int(sum(m["test_set_evaluations"] for m in all_meta))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Anchored-NN VaR - batch summary (ranked by pinball; DM vs unanchored; MCS)\n")
        if failures:
            f.write(f"\n> ! **INCOMPLETE PANEL - {len(failures)} of {len(combos)} "
                    f"ticker-alpha combinations failed and are missing below.** "
                    f"Do not read this as a panel result: the surviving subset may be biased. "
                    f"Failures: "
                    + ", ".join(f"{f['ticker']} alpha={f['alpha']}" for f in failures) + ".\n")
        f.write(f"\nSplit: TRAIN <= {args.train_end} . VAL <= {args.val_end} . TEST after.\n")
        f.write(f"\nArchitectures: {', '.join(architectures)}.\n")
        f.write(f"\n**Disclosure -** specifications evaluated: **{tot_specs}**, "
                f"test-set evaluations: **{tot_tests}**.\n")

        if not vdf.empty:
            f.write("\n## H1 / H4 - anchored vs unanchored\n")
            f.write("\nA claim of improvement requires BOTH a Diebold-Mariano rejection at 5% "
                    "and a `detectable` seed-noise verdict (pre-registration amendment "
                    "2026-08-16). Both H4 readings are reported; neither is used alone.\n\n")
            cols = ["ticker", "alpha", "anchored", "chosen_weight", "edge",
                    "dm_p_anchored_better", "edge_exceeds_anchored_iqr",
                    "edge_exceeds_baseline_iqr", "seed_noise_verdict", "claim_supported"]
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, r in vdf.iterrows():
                f.write("| " + " | ".join(_fmt(r[c], 5) for c in cols) + " |\n")

            n_ok = int(vdf["claim_supported"].sum())
            f.write(f"\n**Panel result:** the anchoring claim is supported in "
                    f"**{n_ok} of {len(vdf)}** ticker-alpha comparisons.\n")
            if vdf["val_weight_at_grid_edge"].any():
                edge_hits = vdf.loc[vdf["val_weight_at_grid_edge"],
                                    ["ticker", "alpha", "anchored"]].to_dict("records")
                f.write(f"\n**Limitation -** VAL selected a grid endpoint in "
                        f"{len(edge_hits)} case(s), so the VAL optimum lies outside the "
                        f"pre-registered weight grid. The grid was deliberately NOT widened "
                        f"after seeing results.\n")

        f.write(to_markdown(master, baseline_label))
    with open(os.path.join(args.outdir, "anchored_batch_meta.json"), "w") as f:
        json.dump({"runs": all_meta, "specifications_evaluated": tot_specs,
                   "test_set_evaluations": tot_tests}, f, indent=2, default=str)

    print(f"\nWrote {csv_path}\n      {md_path}")
    if not vdf.empty:
        print(f"      {verdict_path}")
    print(f"Specifications evaluated: {tot_specs}   Test-set evaluations: {tot_tests}")

    if not vdf.empty:
        print("\n=== H1 / H4 - panel verdict ===")
        for _, r in vdf.iterrows():
            print(f"  {r['ticker']:8s} a={r['alpha']:<5g} {r['anchored']:22s} "
                  f"w={r['chosen_weight']:<5g} DM p={r['dm_p_anchored_better']:.3f}  "
                  f"{r['seed_noise_verdict']:>14s}  "
                  f"claim={'YES' if r['claim_supported'] else 'no'}")
        n_ok = int(vdf["claim_supported"].sum())
        print(f"\n  anchoring claim supported in {n_ok}/{len(vdf)} comparisons")
        print(f"  total batch time: {(time.time() - t_batch) / 60:.1f} min")


if __name__ == "__main__":
    main()
