"""TEST-touch ledger - an append-only record of every scoring pass over the test block.

Why this exists
---------------
Across stages 1, 2, 2b and B this project accumulated **1 899 test-set evaluations** and scored
the same block four times, with design decisions between passes informed by the previous
pass's outcomes. Nothing counted them, so nobody noticed until the Risk review reconstructed
it from run manifests after the fact. `standards/validation-protocol.md` §1 allows the test
block to be scored **once**.

The ledger makes that control mechanical rather than aspirational:

- every scoring pass appends one line to ``outputs/test_touch_ledger.jsonl``; the file is
  never rewritten, so history cannot be quietly lost;
- each line records the code hash, so a touch made with different code is distinguishable;
- cells can be declared **protected** (a holdout). Touching a protected cell that already has
  a recorded touch raises, rather than warning - the single-pass rule is enforced in code.

Design note: the ledger records what happened, it does not judge. A second touch is not
forbidden in exploratory work; it must simply be *visible* and *counted*, so the disclosure
integer in the paper is derived from the record instead of reconstructed later.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

__all__ = [
    "TouchRecord", "DEFAULT_LEDGER_PATH", "code_fingerprint", "record_touch",
    "load_ledger", "count_touches", "check_touch_allowed", "summarise",
    "protected_cells", "protect_cells", "ProtectedCellError",
]

DEFAULT_LEDGER_PATH = os.path.join("outputs", "test_touch_ledger.jsonl")
PROTECTED_PATH = os.path.join("outputs", "test_protected_cells.json")

# Modules whose contents change what a TEST score means.
_FINGERPRINT_MODULES = (
    "src/value_at_risk/evaluation/scoring.py",
    "src/value_at_risk/evaluation/harness.py",
    "src/value_at_risk/evaluation/protocol.py",
    "src/value_at_risk/models/deep_var/train.py",
    "src/value_at_risk/models/deep_var/losses.py",
)


class ProtectedCellError(RuntimeError):
    """Raised when a protected (holdout) cell would be scored a second time."""


@dataclass(frozen=True)
class TouchRecord:
    ticker: str
    alpha: float
    spec: str
    n_seeds: int
    run_label: str
    purpose: str                 # "exploratory" | "confirmatory" | "backfill"
    code_fingerprint: str
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def cell_key(ticker: str, alpha: float) -> str:
    return f"{ticker}@{float(alpha):g}"


def code_fingerprint(root: str = ".") -> str:
    """sha256 over the modules that determine what a TEST score means.

    Missing files are recorded as such rather than skipped, so an incomplete checkout cannot
    silently produce the same fingerprint as a complete one.
    """
    h = hashlib.sha256()
    for rel in _FINGERPRINT_MODULES:
        p = os.path.join(root, rel)
        h.update(rel.encode())
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
        else:
            h.update(b"<MISSING>")
    return h.hexdigest()[:16]


def load_ledger(path: str = DEFAULT_LEDGER_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def count_touches(ticker: str, alpha: float, path: str = DEFAULT_LEDGER_PATH) -> int:
    """How many scoring passes this cell has already received (spec-level rows collapse to
    passes by (run_label, cell))."""
    key = cell_key(ticker, alpha)
    seen = {(r["run_label"], cell_key(r["ticker"], r["alpha"]))
            for r in load_ledger(path)}
    return sum(1 for _, k in seen if k == key)


def protected_cells(path: str | None = None) -> set[str]:
    # Resolved at CALL time, not def time: a control whose location cannot be redirected
    # is untestable, and an untestable control is not a control.
    path = PROTECTED_PATH if path is None else path
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f).get("cells", []))


def protect_cells(cells: list[tuple[str, float]], reason: str,
                  path: str | None = None) -> None:
    """Declare cells as holdout. Additive: protection can be granted, never revoked here."""
    path = PROTECTED_PATH if path is None else path
    existing = protected_cells(path)
    new = existing | {cell_key(t, a) for t, a in cells}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cells": sorted(new), "reason": reason,
                   "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2)


def check_touch_allowed(ticker: str, alpha: float, path: str | None = None,
                        protected_path: str | None = None) -> None:
    """Raise if scoring this cell would violate the single-pass rule on a protected cell."""
    path = DEFAULT_LEDGER_PATH if path is None else path
    protected_path = PROTECTED_PATH if protected_path is None else protected_path
    key = cell_key(ticker, alpha)
    if key not in protected_cells(protected_path):
        return
    n = count_touches(ticker, alpha, path)
    if n >= 1:
        raise ProtectedCellError(
            f"{key} is a PROTECTED holdout cell and has already been scored {n} time(s).\n"
            f"The single-pass rule (validation-protocol s1) forbids a second pass.\n"
            f"If this is deliberate, the cell stops being a holdout and every result derived "
            f"from it must be relabelled as validation."
        )


def record_touch(ticker: str, alpha: float, specs, n_seeds: int, run_label: str,
                 purpose: str = "exploratory", path: str = DEFAULT_LEDGER_PATH,
                 root: str = ".", enforce: bool = True) -> int:
    """Append one row per spec scored on TEST for this cell. Returns rows written."""
    if enforce:
        check_touch_allowed(ticker, alpha, path)
    fp = code_fingerprint(root)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:            # append-only, never truncated
        for spec in specs:
            rec = TouchRecord(ticker=ticker, alpha=float(alpha), spec=str(spec),
                              n_seeds=int(n_seeds), run_label=run_label, purpose=purpose,
                              code_fingerprint=fp)
            f.write(json.dumps(asdict(rec)) + "\n")
            n += 1
    return n


def summarise(path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Totals the paper's disclosure integers are read from - never recomputed by hand."""
    rows = load_ledger(path)
    cells: dict[str, set] = {}
    for r in rows:
        cells.setdefault(cell_key(r["ticker"], r["alpha"]), set()).add(r["run_label"])
    evaluations = sum(r["n_seeds"] for r in rows)
    return {
        "rows": len(rows),
        "test_set_evaluations": evaluations,
        "cells": len(cells),
        "passes_per_cell": {k: len(v) for k, v in sorted(cells.items())},
        "cells_touched_once": sorted(k for k, v in cells.items() if len(v) == 1),
        "cells_touched_more_than_once": sorted(k for k, v in cells.items() if len(v) > 1),
        "run_labels": sorted({r["run_label"] for r in rows}),
        "code_fingerprints": sorted({r["code_fingerprint"] for r in rows}),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the TEST-touch ledger.")
    ap.add_argument("--path", default=DEFAULT_LEDGER_PATH)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--check", nargs=2, metavar=("TICKER", "ALPHA"))
    args = ap.parse_args()

    if args.check:
        t, a = args.check[0], float(args.check[1])
        n = count_touches(t, a, args.path)
        prot = cell_key(t, a) in protected_cells()
        print(f"{cell_key(t, a)}: {n} pass(es), protected={prot}")
        try:
            check_touch_allowed(t, a, args.path)
            print("  -> a further scoring pass is ALLOWED")
        except ProtectedCellError as exc:
            print(f"  -> BLOCKED: {exc}")
            raise SystemExit(1)
        return

    s = summarise(args.path)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
