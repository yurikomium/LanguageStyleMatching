# exploration/Transcript/lsm/tests/test_run_lsm.py
import sys
import os
import types
import pandas as pd
import numpy as np
import pytest

import lsm.runner as run_lsm

def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--dic", "dummy.dic"])
    args = run_lsm.parse_args()
    # Check only the presence of key defaults (avoid strict comparisons for env-dependent paths)
    assert args.round == [1, 2]
    assert isinstance(args.window, float)
    assert isinstance(args.procs, int)
    assert args.temporal in (True, False)

def test_run_lsm_for_round_parallel_writes_csv(tmp_path, monkeypatch):
    # --- Stub: DataAnalyzer ---
    class FakeAnalyzer:
        def get_conversation_files(self, round_number):
            assert round_number == 1
            return ["fileA", "fileB"]
    monkeypatch.setattr(run_lsm, "DataAnalyzer", FakeAnalyzer)

    # --- Stub: init_worker (no-op) ---
    monkeypatch.setattr(run_lsm, "init_worker", lambda *a, **k: None)

    # --- Stub: process_file (return two dummy results) ---
    cats = run_lsm.FUNCTION_WORD_CATEGORIES
    def fake_process_file(path):
        base = {
            "female_id": "F001",
            "male_id": "M002",
            "file_path": path,
            "female_total_words": 100,
            "male_total_words": 120,
            "lsm_mean": 0.8,
            "lsm_std": 0.1,
        }
        for c in cats:
            base[f"lsm_{c}"] = 0.8
        return base
    monkeypatch.setattr(run_lsm, "process_file", fake_process_file)

    # --- Stub: get_ctx().Pool (make imap_unordered a simple iterator) ---
    class FakePool:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
        def imap_unordered(self, func, iterable, chunksize=1):
            for x in iterable:
                yield func(x)

    class FakeCtx:
        def Pool(self, *a, **k): return FakePool()

    monkeypatch.setattr(run_lsm, "get_ctx", lambda: FakeCtx())

    # --- Run ---
    outdir = tmp_path / "results"
    df = run_lsm.run_lsm_for_round_parallel(
        round_number=1,
        dic_path=str(tmp_path / "mini.dic"),
        results_dir=str(outdir),
        procs=1,
        chunksize=1,
        temporal=False,
        window_min=2.5,
    )

    # DataFrame return value: 2 rows, with round_label populated
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["round_label"]) == {"1st_round"}

    # CSV files are written
    csv1 = outdir / "lsm_results_1st.csv"
    csv2 = outdir / "lsm_results_1st_full.csv"
    assert csv1.exists() and csv2.exists()

    # Content consistency (mean is 0.8)
    df1 = pd.read_csv(csv1)
    assert pytest.approx(df1["lsm_mean"].mean(), rel=1e-9) == 0.8
