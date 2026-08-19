"""Tests for the TEST-touch ledger.

The control this enforces is the one whose absence let the project reach 1 899 test-set
evaluations across four passes over the same block.
"""
import json
import os

import pytest

from value_at_risk.evaluation.ledger import (
    record_touch, load_ledger, count_touches, summarise, cell_key,
    protect_cells, protected_cells, check_touch_allowed, ProtectedCellError,
    code_fingerprint,
)


@pytest.fixture
def paths(tmp_path):
    return (str(tmp_path / "ledger.jsonl"), str(tmp_path / "protected.json"))


def test_record_and_count(paths):
    led, _ = paths
    record_touch("^GSPC", 0.01, ["Unanchored", "Anchor param"], 10, "stage1", path=led,
                 enforce=False)
    assert len(load_ledger(led)) == 2
    assert count_touches("^GSPC", 0.01, led) == 1          # one PASS, two specs


def test_second_pass_is_counted_not_hidden(paths):
    led, _ = paths
    for label in ("stage1", "stage2", "stage2b"):
        record_touch("^GSPC", 0.01, ["a", "b"], 10, label, path=led, enforce=False)
    assert count_touches("^GSPC", 0.01, led) == 3
    s = summarise(led)
    assert s["passes_per_cell"][cell_key("^GSPC", 0.01)] == 3
    assert cell_key("^GSPC", 0.01) in s["cells_touched_more_than_once"]


def test_ledger_is_append_only(paths):
    led, _ = paths
    record_touch("A", 0.05, ["s1"], 10, "run1", path=led, enforce=False)
    first = open(led, encoding="utf-8").read()
    record_touch("B", 0.05, ["s1"], 10, "run2", path=led, enforce=False)
    second = open(led, encoding="utf-8").read()
    # earlier content must still be a prefix: history cannot be rewritten
    assert second.startswith(first)


def test_protected_cell_blocks_second_pass(paths, monkeypatch):
    led, prot = paths
    import value_at_risk.evaluation.ledger as L
    monkeypatch.setattr(L, "PROTECTED_PATH", prot)

    protect_cells([("NEW1", 0.01)], reason="holdout", path=prot)
    assert cell_key("NEW1", 0.01) in protected_cells(prot)

    # first pass allowed
    record_touch("NEW1", 0.01, ["frozen_spec"], 10, "holdout_run", path=led)
    # second pass must raise, not warn
    with pytest.raises(ProtectedCellError, match="already been scored"):
        record_touch("NEW1", 0.01, ["frozen_spec"], 10, "holdout_run_2", path=led)


def test_unprotected_cell_allows_repeats(paths, monkeypatch):
    led, prot = paths
    import value_at_risk.evaluation.ledger as L
    monkeypatch.setattr(L, "PROTECTED_PATH", prot)
    record_touch("EXPL", 0.01, ["s"], 10, "r1", path=led)
    record_touch("EXPL", 0.01, ["s"], 10, "r2", path=led)     # allowed, but counted
    assert count_touches("EXPL", 0.01, led) == 2


def test_disclosure_integer_comes_from_the_ledger(paths):
    led, _ = paths
    record_touch("A", 0.01, ["s1", "s2", "s3"], 10, "r1", path=led, enforce=False)
    record_touch("B", 0.01, ["s1", "s2"], 10, "r1", path=led, enforce=False)
    # 5 spec-rows x 10 seeds
    assert summarise(led)["test_set_evaluations"] == 50


def test_code_fingerprint_changes_with_scoring_code(tmp_path):
    root = tmp_path / "repo"
    (root / "src/value_at_risk/evaluation").mkdir(parents=True)
    (root / "src/value_at_risk/models/deep_var").mkdir(parents=True)
    f = root / "src/value_at_risk/evaluation/scoring.py"
    f.write_text("alpha = 1\n", encoding="utf-8")
    a = code_fingerprint(str(root))
    f.write_text("alpha = 2\n", encoding="utf-8")
    b = code_fingerprint(str(root))
    assert a != b, "a change to scoring must change the fingerprint"


def test_fingerprint_recorded_on_every_row(paths):
    led, _ = paths
    record_touch("A", 0.01, ["s"], 10, "r1", path=led, enforce=False)
    row = load_ledger(led)[0]
    for k in ("ticker", "alpha", "spec", "n_seeds", "run_label", "purpose",
              "code_fingerprint", "timestamp_utc"):
        assert k in row
