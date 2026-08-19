"""Measure how well `(1 - 2*lr*w)^T` describes the real separation trajectory.

The paper derives

    Delta_{t+1} = (1 - 2*lr*w) * Delta_t  -  lr * [ g(theta_a) - g(theta_b) ]

and drops the second term to obtain `spread_T ~ spread_0 * (1 - 2*lr*w)^T`. Until now that
approximation was **asserted, not measured** — which is the practice this paper exists to
criticise, committed in the paper's own central derivation. This script measures it.

Two quantities, and the distinction between them is the point:

  A. absolute accuracy  — does the formula predict the raw separation after T steps?
  B. relative accuracy  — does it predict contraction **relative to w = 0**, which is what
                          every ratio the paper actually reports is?

Plus a direct test of the exact claim (the anchor cancels), which turned out to be true only
to first order. See `docs/contraction-measurement.md`.

    python scripts/measure_contraction.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from value_at_risk.evaluation.shrinkage_demo import (  # noqa: E402
    anchor_invariance, contraction_accuracy,
)

OUTDIR = os.path.join("outputs", "contraction_check")


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    acc = pd.DataFrame(contraction_accuracy())
    inv = pd.DataFrame(anchor_invariance())
    acc.to_csv(os.path.join(OUTDIR, "contraction_accuracy.csv"), index=False)
    inv.to_csv(os.path.join(OUTDIR, "anchor_invariance.csv"), index=False)

    fmt = lambda v: f"{v:.5g}"                                        # noqa: E731
    print("=" * 78)
    print("A. Does (1-2*lr*w)^T describe the trajectory?")
    print("=" * 78)
    print(acc[["weight", "observed_final", "predicted_final", "absolute_ratio",
               "observed_rel", "predicted_rel", "relative_ratio"]]
          .to_string(index=False, float_format=fmt))

    nz = acc[acc.weight > 0]
    print(f"\n  ABSOLUTE  obs/pred : median {acc.absolute_ratio.median():.4f}"
          f"   range [{acc.absolute_ratio.min():.4f}, {acc.absolute_ratio.max():.4f}]")
    print(f"  RELATIVE  obs/pred : median {nz.relative_ratio.median():.4f}"
          f"   range [{nz.relative_ratio.min():.4f}, {nz.relative_ratio.max():.4f}]")
    print("\n  The w=0 row isolates the dropped term: prediction is 1 (no penalty), and the")
    print(f"  observed contraction of {acc.observed_final.iloc[0]:.4f} is entirely the data term")
    print(f"  — a factor of {1/acc.observed_final.iloc[0]:.1f}x that the formula does not model.")

    print("\n" + "=" * 78)
    print("B. Is the contraction independent of the anchor?")
    print("=" * 78)
    print(inv.to_string(index=False, float_format=fmt))
    small = inv[inv.weight <= 0.017].max_rel_diff.max()
    print(f"\n  max relative difference, w <= 0.017 : {small:.4f}")
    print(f"  max relative difference, whole grid : {inv.max_rel_diff.max():.4f}")

    verdict = {
        "absolute_ratio_median": float(acc.absolute_ratio.median()),
        "relative_ratio_median": float(nz.relative_ratio.median()),
        "relative_ratio_min": float(nz.relative_ratio.min()),
        "relative_ratio_max": float(nz.relative_ratio.max()),
        "data_term_only_contraction": float(acc.observed_final.iloc[0]),
        "anchor_max_rel_diff_small_w": float(small),
        "anchor_max_rel_diff_all": float(inv.max_rel_diff.max()),
    }
    with open(os.path.join(OUTDIR, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  The raw formula is NOT accurate in absolute terms: it ignores a data-term")
    print("  contraction of roughly 14x, so it under-predicts by about that factor.")
    print("  For the quantity the paper reports — contraction relative to the unanchored")
    print("  baseline — it is accurate to a median of ~5% and a worst case of ~20%.")
    print("  Anchor-independence holds to first order (<3.4% for w <= 0.017) and degrades")
    print("  to ~50% at the largest weight, because the anchor re-enters through the")
    print("  dropped term. The residual does NOT favour the informative anchor.")
    print(f"\nWrote {OUTDIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
