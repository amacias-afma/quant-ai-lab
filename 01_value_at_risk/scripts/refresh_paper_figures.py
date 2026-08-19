"""Regenerate the synthetic-demonstration fields of ``outputs/paper_figures.json``.

Why this exists
---------------
``paper_figures.json`` is the file the draft's header promises every number is read from.
When Editor condition E7 replaced the four-weight demonstration with a ten-point grid, the
demonstration outputs were rewritten but **the figures file was not**, so it kept carrying the
superseded values:

    demo_n = 12, demo_rho = 0.9716, demo_rho_p = 1.4e-07,
    truth_ratio = 5.57, nonsense_ratio = 13.85

Those last two are exactly the cherry-picked pair whose quotient is the "2.5x" headline E7
removed. A regeneration from the stale file would have silently reinstated the defect the
Editor caught, in the section of the paper that criticises that practice.

The fix is not to retype the numbers. It is to derive them, so the file cannot drift again:

    python scripts/refresh_paper_figures.py            # rewrite from demo outputs
    python scripts/refresh_paper_figures.py --check    # verify only, exit 1 on drift

``--check`` is the form worth wiring into CI: it fails when the figures file disagrees with
the results it claims to summarise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from scipy import stats

DEMO_CSV = os.path.join("outputs", "shrinkage_demo", "synthetic_shrinkage.csv")
DEMO_PAIRED = os.path.join("outputs", "shrinkage_demo", "paired_comparison.json")
FIGURES = os.path.join("outputs", "paper_figures.json")
INTERVALS = os.path.join("outputs", "bootstrap_intervals.json")

TRUTH = "informative (truth)"
NONSENSE = "nonsense (scale-matched)"


def demo_fields(csv_path: str = DEMO_CSV, paired_path: str = DEMO_PAIRED) -> dict:
    """Derive every demonstration figure the draft quotes, from the demo outputs alone."""
    d = pd.read_csv(csv_path)
    nz = d[d.weight > 0]

    # Dose-response across ALL anchors: the point is that it holds regardless of target.
    rho, rho_p = stats.spearmanr(nz.weight, nz.iqr_ratio)

    # Loss comparison is quoted at the largest weight, where the anchors differ most.
    w_max = float(d.weight.max())
    at_max = d[d.weight == w_max].set_index("anchor")

    out = {
        "demo_rho": float(rho),
        "demo_rho_p": float(rho_p),
        "demo_n": int(len(nz)),
        "demo_weights": int(nz.weight.nunique()),
        "demo_loss_weight": w_max,
        "truth_ratio": float(at_max.loc[TRUTH, "iqr_ratio"]),
        "nonsense_ratio": float(at_max.loc[NONSENSE, "iqr_ratio"]),
        "truth_loss": float(at_max.loc[TRUTH, "median"]),
        "nonsense_loss": float(at_max.loc[NONSENSE, "median"]),
    }

    with open(paired_path, encoding="utf-8") as f:
        paired = json.load(f)
    p = paired["paired"]
    out.update({
        "demo_paired_n": int(p["n_weights"]),
        "demo_paired_k": int(p["nonsense_at_least_as_stabilising"]),
        "demo_paired_p": float(p["sign_test_p"]),
        "demo_paired_median": float(p["median_relative"]),
        "demo_paired_ci_low": float(paired["rel_ci"]["ci_low"]),
        "demo_paired_ci_high": float(paired["rel_ci"]["ci_high"]),
    })
    return out


def synthetic_intervals(csv_path: str = DEMO_CSV) -> dict:
    """Bootstrap intervals for the synthetic demonstration, from the current grid.

    ``outputs/bootstrap_intervals.json`` was written by hand and its two synthetic entries
    were left at the superseded four-weight grid (n = 4), with ci_high holding 5.5708 and
    13.8514 -- the retracted pair again, in a second file. Derived here for the same reason
    the figures file is.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from value_at_risk.evaluation.power import bootstrap_ci

    d = pd.read_csv(csv_path)
    out = {}
    for key, anchor, label in (("synth_informative", TRUTH, "synthetic informative (truth)"),
                               ("synth_nonsense", NONSENSE,
                                "synthetic nonsense (scale-matched)")):
        v = d[(d.anchor == anchor) & (d.weight > 0)].iqr_ratio
        point, lo, hi = bootstrap_ci(v)
        out[key] = {"label": label, "n": int(v.size), "median": point,
                    "ci_low": lo, "ci_high": hi, "level": 0.95,
                    "text": f"{point:.1f}x (95% CI {lo:.1f}-{hi:.1f}, n={v.size})"}
    return out


def refresh_intervals(check: bool) -> int:
    """Rewrite (or verify) the synthetic entries of the bootstrap-intervals file."""
    if not os.path.exists(INTERVALS):
        print(f"MISSING: {INTERVALS}", file=sys.stderr)
        return 2
    with open(INTERVALS, encoding="utf-8") as f:
        current = json.load(f)
    derived = synthetic_intervals()

    # The hand-written file truncated its keys to 15 characters ("synth_informati",
    # "synth_nonsense "). Drop any such variant so the corrected keys do not duplicate them.
    stale_keys = [k for k in current if k.startswith("synth") and k not in derived]

    drift = stale_keys + [k for k, v in derived.items()
                          if current.get(k, {}).get("n") != v["n"]
                          or abs(current.get(k, {}).get("median", -1) - v["median"]) > 1e-12]
    if not drift:
        print(f"{INTERVALS}: synthetic entries consistent with the demonstration grid.")
        return 0

    print(f"DRIFT in {INTERVALS}: {sorted(set(drift))}")
    for k, v in derived.items():
        was = current.get(k) or current.get(k[:15]) or {}
        print(f"  {k:20s} was n={was.get('n')} {was.get('text')!r}  ->  {v['text']!r}")
    if check:
        print("--check: not writing.")
        return 1

    for k in stale_keys:
        current.pop(k, None)
    current.update(derived)
    with open(INTERVALS, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
        f.write("\n")
    print(f"Rewrote the synthetic entries of {INTERVALS} from {DEMO_CSV}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the figures file matches the demo outputs; do not write")
    args = ap.parse_args()

    for p in (DEMO_CSV, DEMO_PAIRED, FIGURES):
        if not os.path.exists(p):
            print(f"MISSING: {p}", file=sys.stderr)
            return 2

    derived = demo_fields()
    with open(FIGURES, encoding="utf-8") as f:
        figures = json.load(f)

    drift = {k: (figures.get(k), v) for k, v in derived.items()
             if k not in figures
             or not isinstance(figures[k], (int, float))
             or abs(float(figures[k]) - v) > 1e-12 * max(1.0, abs(v))}

    rc = 0
    if not drift:
        print("paper_figures.json is consistent with the demonstration outputs.")
    else:
        print(f"DRIFT in {len(drift)} field(s) between {FIGURES} and the demo outputs:\n")
        for k, (was, now) in sorted(drift.items()):
            print(f"  {k:22s} figures={was!r:24s} derived={now!r}")

        if args.check:
            print("\n--check: not writing. The figures file is stale.")
            rc = 1
        else:
            figures.update(derived)
            with open(FIGURES, "w", encoding="utf-8") as f:
                json.dump(figures, f, indent=2)
                f.write("\n")
            print(f"\nRewrote {FIGURES} from {DEMO_CSV} and {DEMO_PAIRED}.")

    print()
    return max(rc, refresh_intervals(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
