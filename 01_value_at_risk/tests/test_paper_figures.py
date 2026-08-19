"""The figures file the paper is generated from must agree with the result files.

This is the control for the near-miss recorded in draft §3.5: after the E7 correction the
demonstration outputs were rewritten and `paper_figures.json` was not, so it kept the exact
pair of numbers whose quotient was the retracted "2.5x" headline. Prose was correct; the
machine-readable summary was stale. These tests make the derivation testable and the drift
detectable.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("pandas")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from refresh_paper_figures import DEMO_CSV, DEMO_PAIRED, FIGURES, demo_fields  # noqa: E402

_HAVE_OUTPUTS = all(os.path.exists(p) for p in (DEMO_CSV, DEMO_PAIRED, FIGURES))
needs_outputs = pytest.mark.skipif(not _HAVE_OUTPUTS, reason="demo outputs not present")


@needs_outputs
def test_figures_file_matches_demo_outputs():
    """The regression that actually happened: figures file drifting from its sources."""
    derived = demo_fields()
    with open(FIGURES, encoding="utf-8") as f:
        figures = json.load(f)
    stale = [k for k, v in derived.items()
             if k not in figures or abs(float(figures[k]) - v) > 1e-12 * max(1.0, abs(v))]
    assert not stale, (
        f"paper_figures.json is stale in {stale}. "
        f"Run `python scripts/refresh_paper_figures.py`."
    )


@needs_outputs
def test_retracted_cherry_picked_pair_is_absent():
    """Guard the specific values E7 removed, so they cannot reappear unnoticed.

    5.570818 / 13.851437 = 2.49, the 'nonsense stabilises 2.5x more' figure the Editor
    rejected as the largest of four cells. Named explicitly because a generic staleness
    check would pass on any self-consistent-but-wrong file.
    """
    with open(FIGURES, encoding="utf-8") as f:
        figures = json.load(f)
    for key, retracted in (("truth_ratio", 5.570817881915514),
                           ("nonsense_ratio", 13.851437014782304)):
        assert abs(figures[key] - retracted) > 1e-9, (
            f"{key} holds the retracted four-weight value {retracted}; "
            f"the E7 correction has been reverted."
        )


@needs_outputs
def test_paired_claim_is_the_one_the_paper_makes():
    """The draft claims 9 of 10 weights and p = 0.021. Both must come from the outputs."""
    d = demo_fields()
    assert d["demo_paired_n"] == 10
    assert d["demo_paired_k"] == 9
    assert round(d["demo_paired_p"], 3) == 0.021
    # The claim is 'at least as much', not 'more' - the median must be near parity, not large.
    assert 1.0 <= d["demo_paired_median"] < 1.2


@needs_outputs
def test_dose_response_spans_every_anchor():
    """The demonstration's point is that the dose-response holds for ANY target.

    30 = 10 non-zero weights x 3 anchors. If this collapses to 10 or 20, the correlation is
    being computed within a single anchor and no longer supports the claim made.
    """
    d = demo_fields()
    assert d["demo_n"] == 30
    assert d["demo_weights"] == 10
    assert d["demo_rho"] > 0.9
