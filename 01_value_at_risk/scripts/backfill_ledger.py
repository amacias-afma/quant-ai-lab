"""Backfill the TEST-touch ledger from the run manifests that already exist.

The ledger was built after the fact. Starting it empty would imply the test block is
untouched, which is the opposite of the truth: stages 1, 2, 2b and B scored it four times.
This reconstructs that history from `anchored_batch_meta.json` files so the disclosure
integers in the paper come from the ledger rather than from a hand count.

Run once:  python scripts/backfill_ledger.py
Idempotent: refuses to append a run_label that is already present.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from value_at_risk.evaluation.ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH, load_ledger, record_touch, summarise,
)

# (manifest path, run_label). Order is chronological.
SOURCES = [
    ("outputs/anchored_batch_meta.json",            "stage1"),
    ("outputs/stage2/anchored_batch_meta.json",     "stage2_fine_onese"),
    ("outputs/stage2b/anchored_batch_meta.json",    "stage2b_fine_argmin"),
    ("outputs/stageB_mlp/anchored_batch_meta.json", "stageB_mlp"),
]

# Pre-panel debugging runs: single-ticker passes that also scored TEST.
SINGLE_RUNS = [
    ("^GSPC",   0.05, "prepanel_gspc_a005"),
    ("BTC-USD", 0.05, "prepanel_btc_a005"),
]


def main():
    existing = {r["run_label"] for r in load_ledger(DEFAULT_LEDGER_PATH)}
    written = 0

    for path, label in SOURCES:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        if label in existing:
            print(f"  [skip] {label} already in ledger")
            continue
        meta = json.load(open(path, encoding="utf-8"))
        for run in meta.get("runs", []):
            specs = [v["anchored"] for v in run.get("h1_h4_verdicts", [])]
            nn_names = specs + ["Unanchored"] if specs else ["(unrecorded NN specs)"]
            written += record_touch(
                run["ticker"], run["alpha"], nn_names, n_seeds=10,
                run_label=label, purpose="exploratory (backfilled)", enforce=False,
            )
            # Classical rungs are scored on TEST too (one deterministic pass each). Omitting
            # them made the ledger undercount by 149 versus the manifests; the ledger is the
            # source of the disclosure integer, so it must reconcile exactly.
            n_classical = run["test_set_evaluations"] - 10 * len(nn_names)
            if n_classical > 0:
                written += record_touch(
                    run["ticker"], run["alpha"],
                    [f"classical_rung_{i+1}" for i in range(n_classical)], n_seeds=1,
                    run_label=label, purpose="exploratory (backfilled)", enforce=False,
                )
        print(f"  [ok]   {label}: {len(meta.get('runs', []))} cells")

    for ticker, alpha, label in SINGLE_RUNS:
        if label in existing:
            print(f"  [skip] {label} already in ledger")
            continue
        written += record_touch(ticker, alpha, ["Unanchored", "Anchor param", "Anchor hist"],
                                n_seeds=10, run_label=label,
                                purpose="debugging (backfilled)", enforce=False)
        print(f"  [ok]   {label}")

    print(f"\nrows written: {written}")
    print(json.dumps(summarise(), indent=2)[:1200])


if __name__ == "__main__":
    main()
