"""Build the paper's two figures. Every value is read from a result file.

    python scripts/make_figures.py

Figure 1  The dose-response the study took as corroboration.
          Selected anchor weight against inter-seed IQR reduction, pooled over all four runs
          (n = 85 cells with a non-zero selected weight). Spearman rho = +0.585.
          The figure is included precisely because it looks convincing: it is the evidence
          that persuaded us, and it is an artefact.

Figure 2  The two panels that dissolved it.
          (a) Real prior against a time-permuted prior of identical scale, on the same cells.
          (b) The synthetic demonstration with ground truth known, overlaid with the
              analytical contraction (1 - 2*lr*w)^T -- a curve containing no reference to
              the anchor, which both the true and the nonsense anchor track.

Design note: panel (b) carries the argument. If the reader takes one thing from the figures
it should be that the analytical curve, which cannot see the anchor, predicts both series.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTDIR = os.path.join("paper", "figures")
DEMO_CSV = os.path.join("outputs", "shrinkage_demo", "synthetic_shrinkage.csv")
NONSENSE = os.path.join("outputs", "nonsense_prior_test", "nonsense_prior_a0.01.csv")

TRUTH = "informative (truth)"
NONSENSE_ANCHOR = "nonsense (scale-matched)"

# Matches evaluation/shrinkage_demo.fit_anchored defaults; the analytical curve is only
# meaningful if it uses the same lr and step budget the demonstration actually ran.
DEMO_LR, DEMO_STEPS = 0.05, 400

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

C_REAL, C_FAKE, C_PRED = "#1f4e79", "#c0392b", "#7f8c8d"


def load_pooled_verdicts() -> pd.DataFrame:
    """Every anchored-vs-unanchored comparison across all four runs."""
    files = ["outputs/anchored_batch_verdicts.csv"] + sorted(
        glob.glob("outputs/stage*/anchored_batch_verdicts.csv"))
    # The first run lives at the top of outputs/, so its directory name is "outputs" rather
    # than a stage label. Rename it for the legend instead of letting the path leak in.
    labels = {"outputs": "stage 1", "stage2": "stage 2", "stage2b": "stage 2b",
              "stageB_mlp": "stage B (MLP)"}
    frames = []
    for f in files:
        if os.path.exists(f):
            d = pd.read_csv(f)
            raw = os.path.basename(os.path.dirname(f))
            d["run"] = labels.get(raw, raw)
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def figure1(outdir: str) -> dict:
    a = load_pooled_verdicts()
    d = a.dropna(subset=["chosen_weight", "anchored_iqr", "baseline_iqr"])
    d = d[(d.chosen_weight > 0) & (d.anchored_iqr > 0)].copy()
    d["ratio"] = d.baseline_iqr / d.anchored_iqr
    rho, p = stats.spearmanr(d.chosen_weight, d.ratio)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for run, g in d.groupby("run"):
        ax.scatter(g.chosen_weight, g.ratio, s=26, alpha=0.75, label=run, edgecolor="none")

    ax.axhline(1.0, color="k", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("anchor weight selected on validation  (log)")
    ax.set_ylabel("inter-seed IQR reduction\nunanchored / anchored  (log)")
    ax.set_title("Figure 1  The dose-response that convinced us", loc="left", fontsize=10)
    ax.annotate(f"Spearman $\\rho$ = +{rho:.3f}\n$p$ = {p:.1e},  $n$ = {len(d)}",
                xy=(0.03, 0.95), xycoords="axes fraction", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.6))
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), columnspacing=1.2, handletextpad=0.3)
    fig.text(0.01, -0.10,
             "Stronger shrinkage, more stability. The relationship is real and it is "
             "mechanical:\nan L2 penalty contracts every seed toward the same fixed point "
             "whatever that point is.",
             fontsize=7.5, color="0.35", va="top")
    fig.tight_layout()
    path = os.path.join(outdir, "figure1_dose_response.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": path, "rho": float(rho), "p": float(p), "n": int(len(d))}


def figure2(outdir: str) -> dict:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # ---- (a) real vs scale-matched permuted prior, empirical -------------------------
    n = pd.read_csv(NONSENSE)
    emp = n[n.prior.isin(["param", "shuffled"])].copy()
    piv = emp.pivot_table(index=["ticker", "weight"], columns="prior", values="iqr_ratio")
    piv = piv.dropna().reset_index()
    lab = [f"{t}\nw={w:g}" for t, w in zip(piv.ticker, piv.weight)]
    x = np.arange(len(piv))

    axL.bar(x - 0.19, piv["param"], 0.38, label="real prior", color=C_REAL)
    axL.bar(x + 0.19, piv["shuffled"], 0.38, label="permuted prior (same scale)", color=C_FAKE)
    axL.axhline(1.0, color="k", lw=0.8, ls=":", zorder=0)
    axL.set_xticks(x)
    axL.set_xticklabels(lab, fontsize=6.5)
    axL.set_ylabel("inter-seed IQR reduction")
    axL.set_title("(a) empirical: information stripped, effect intact", loc="left", fontsize=9)
    axL.legend(fontsize=7, frameon=False)

    wins = int((piv["shuffled"] >= piv["param"]).sum())
    try:
        wil_p = float(stats.wilcoxon(piv["param"], piv["shuffled"]).pvalue)
        wil_txt = f"Wilcoxon $p$ = {wil_p:.3f}"
    except ValueError:                                   # too few pairs to run the test
        wil_p = float("nan")
        wil_txt = "Wilcoxon not computable at this n"
    axL.annotate(f"permuted >= real in {wins} of {len(piv)}\n{wil_txt}",
                 xy=(0.03, 0.95), xycoords="axes fraction", va="top", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.6))

    # ---- (b) synthetic, with the analytical curve that cannot see the anchor ---------
    d = pd.read_csv(DEMO_CSV)
    t = d[(d.anchor == TRUTH) & (d.weight > 0)].sort_values("weight")
    f = d[(d.anchor == NONSENSE_ANCHOR) & (d.weight > 0)].sort_values("weight")

    axR.plot(t.weight, t.iqr_ratio, "o-", color=C_REAL, ms=4, lw=1.4,
             label="anchor = the true optimum")
    axR.plot(f.weight, f.iqr_ratio, "s-", color=C_FAKE, ms=4, lw=1.4,
             label="anchor = scale-matched nonsense")

    # 1 / (1 - 2*lr*w)^T : predicted IQR RATIO (baseline / anchored) from the penalty alone.
    w = np.asarray(t.weight, dtype=float)
    contraction = np.clip(1.0 - 2.0 * DEMO_LR * w, 1e-12, None) ** DEMO_STEPS
    axR.plot(w, 1.0 / contraction, ls="--", color=C_PRED, lw=1.3,
             label=r"analytical  $(1-2\eta w)^{-T}$")

    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("penalty weight $w$  (log)")
    axR.set_ylabel("inter-seed IQR reduction  (log)")
    axR.set_title("(b) synthetic: the curve does not know the target", loc="left", fontsize=9)
    axR.legend(fontsize=7, frameon=False, loc="upper left")

    with open(os.path.join("outputs", "shrinkage_demo", "paired_comparison.json"),
              encoding="utf-8") as fh:
        paired = json.load(fh)["paired"]
    axR.annotate(
        f"nonsense stabilises >= truth\nat {paired['nonsense_at_least_as_stabilising']} of "
        f"{paired['n_weights']} weights\nsign test $p$ = {paired['sign_test_p']:.3f}",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.6))

    fig.text(0.01, -0.03,
             "The dashed curve is derived from the penalty gradient alone and contains no "
             "reference to the anchor's value.\nBoth anchors track it. Stability measures the "
             "penalty; only forecast loss measures whether the target was any good.",
             fontsize=7.5, color="0.35", va="top")
    fig.tight_layout()
    path = os.path.join(outdir, "figure2_control.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": path, "paired_cells": int(len(piv)), "permuted_wins": wins,
            "wilcoxon_p": wil_p}


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    f1 = figure1(OUTDIR)
    f2 = figure2(OUTDIR)
    print(f"Figure 1  {f1['path']}")
    print(f"          rho=+{f1['rho']:.4f}  p={f1['p']:.3e}  n={f1['n']}")
    print(f"Figure 2  {f2['path']}")
    print(f"          {f2['paired_cells']} paired cells, permuted >= real in "
          f"{f2['permuted_wins']}, Wilcoxon p={f2['wilcoxon_p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
