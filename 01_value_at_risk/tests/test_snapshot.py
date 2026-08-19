"""Tests for frozen data snapshots. A fake downloader stands in for yfinance."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from value_at_risk.data.snapshot import (
    get_prices, verify_snapshots, load_manifest, snapshot_filename, snapshot_key,
    sha256_file, MANIFEST_NAME,
)


class FakeSource:
    """Counts calls so we can prove the second read does NOT hit the network."""

    def __init__(self, n=300, seed=0):
        self.calls = 0
        self.n = n
        self.seed = seed

    def __call__(self, ticker, end_date):
        self.calls += 1
        idx = pd.bdate_range("2016-06-30", periods=self.n)
        rng = np.random.default_rng(self.seed)
        price = 100 * np.exp(np.cumsum(rng.standard_normal(self.n) * 0.01))
        return pd.DataFrame({"price": price}, index=idx)


def test_first_call_downloads_second_call_reads_snapshot(tmp_path):
    src = FakeSource()
    d = str(tmp_path)
    df1 = get_prices("^GSPC", "2026-06-30", d, downloader=src)
    assert src.calls == 1
    df2 = get_prices("^GSPC", "2026-06-30", d, downloader=src)
    assert src.calls == 1                       # no second download
    pd.testing.assert_frame_equal(df1, df2)


def test_manifest_records_hash_and_range(tmp_path):
    src = FakeSource()
    d = str(tmp_path)
    df = get_prices("^GSPC", "2026-06-30", d, downloader=src)
    man = load_manifest(d)
    key = snapshot_key("^GSPC", "2026-06-30")
    assert key in man
    entry = man[key]
    assert entry["rows"] == len(df)
    assert entry["first_date"] == str(df.index[0].date())
    assert entry["last_date"] == str(df.index[-1].date())
    # hash in the manifest matches the file on disk
    assert entry["sha256"] == sha256_file(os.path.join(d, entry["filename"]))
    assert len(entry["sha256"]) == 64


def test_filename_is_filesystem_safe():
    assert snapshot_filename("^GSPC", "2026-06-30") == "GSPC@2026-06-30.csv"
    assert snapshot_filename("CLP=X", "2026-06-30") == "CLP_X@2026-06-30.csv"
    assert snapshot_filename("BTC-USD", "2026-06-30") == "BTC-USD@2026-06-30.csv"


def test_tampered_snapshot_is_rejected(tmp_path):
    src = FakeSource()
    d = str(tmp_path)
    get_prices("^GSPC", "2026-06-30", d, downloader=src)
    path = os.path.join(d, snapshot_filename("^GSPC", "2026-06-30"))
    with open(path, "a", encoding="utf-8") as f:
        f.write("2030-01-01,999.0\n")           # silent corruption
    with pytest.raises(ValueError, match="hash mismatch"):
        get_prices("^GSPC", "2026-06-30", d, downloader=src)
    assert verify_snapshots(d)                   # verifier also reports it


def test_verify_clean_when_untouched(tmp_path):
    src = FakeSource()
    d = str(tmp_path)
    get_prices("^GSPC", "2026-06-30", d, downloader=src)
    get_prices("BTC-USD", "2026-06-30", d, downloader=src)
    assert verify_snapshots(d) == []


def test_force_download_refreezes(tmp_path):
    d = str(tmp_path)
    src_a = FakeSource(seed=1)
    get_prices("^GSPC", "2026-06-30", d, downloader=src_a)
    old = load_manifest(d)[snapshot_key("^GSPC", "2026-06-30")]["sha256"]
    src_b = FakeSource(seed=2)                   # different data
    get_prices("^GSPC", "2026-06-30", d, downloader=src_b, force_download=True)
    new = load_manifest(d)[snapshot_key("^GSPC", "2026-06-30")]["sha256"]
    assert new != old                            # re-freezing is explicit and visible
    assert verify_snapshots(d) == []


def test_orphan_snapshot_without_manifest_entry_raises(tmp_path):
    src = FakeSource()
    d = str(tmp_path)
    get_prices("^GSPC", "2026-06-30", d, downloader=src)
    os.remove(os.path.join(d, MANIFEST_NAME))    # manifest lost, file remains
    with pytest.raises(FileNotFoundError, match="no manifest entry"):
        get_prices("^GSPC", "2026-06-30", d, downloader=src)


def test_empty_download_raises(tmp_path):
    def empty(ticker, end_date):
        return pd.DataFrame()
    with pytest.raises(ValueError, match="no rows"):
        get_prices("^GSPC", "2026-06-30", str(tmp_path), downloader=empty)
