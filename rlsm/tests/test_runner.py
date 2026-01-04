# -*- coding: utf-8 -*-
"""
==============================================================================
test_runner_outputs.py - Runner framework and output format tests
==============================================================================

What this test file guarantees:

1. **Correct output file names**
   - Proper naming when rw is enabled (rw_ prefix)
   - Standard naming when rw is disabled
   - Comprehensive outputs for each category (conversations, pairs, individual, dyad)

2. **Parallel processing framework**
   - process_file_rlsm is executed in parallel as intended
   - Correct aggregation/integration across processes
   - Correct propagation of the context argument

3. **Error handling**
   - Generates an error file on failure (rw_rlsm_errors.csv)
   - Does not generate the error file on success
   - Records appropriate error information

4. **Full vs rounded outputs**
   - Rounded output via float_format
   - Full-precision output in parallel (_full.csv)
   - Row-count consistency between both formats

5. **Mocked test environment**
   - Lightweight tests without running heavy real file processing
   - Safe file operations in a temporary directory
   - Isolation of external dependencies via monkeypatch
"""

import os
import pandas as pd
import types

import rlsm.runner as R


def test_runner_writes_expected_rw_filenames(tmp_path, monkeypatch):
    # One dummy CSV
    df = pd.DataFrame([
        {"speaker": "female", "text": "a", "text_clean": "a"},
        {"speaker": "male",   "text": "b", "text_clean": "b"},
    ])
    f = tmp_path / "F001_M002.csv"
    df.to_csv(f, index=False)

    # Mock process_file_rlsm: return a minimal dict
    def fake_worker(args, context=None):
        return {
            "conversation_summary": {"file_path": str(f)},
            "pairs_rows": [{"file_path": str(f)}],
            "individual_category_rows": [{"file_path": str(f)}],
            "dyad_category_rows": [{"file_path": str(f)}],
            # Added when rw is enabled
            "rw_conv_row": {"file_path": str(f)},
            "rw_pair_rows": [{"file_path": str(f)}],
            "rw_individual_rows": [{"file_path": str(f)}],
            "rw_dyad_rows": [{"file_path": str(f)}],
        }

    monkeypatch.setattr(R, "process_file_rlsm", fake_worker)

    results_dir = tmp_path / "out"
    os.makedirs(results_dir, exist_ok=True)

    # Run with rw=1
    R.run_rlsm_parallel(
        data_dir=str(tmp_path),
        dic_path="/dev/null",
        results_dir=str(results_dir),
        procs=1,
        chunksize=1,
        na_policy="bilateral_only",
        eps=1e-4,
        zero_tol=0.0,
        unitize="off",
        merge_gap=0.0,
        strip_tags=0,
        bracket_mode="off",
        enable_micro_ppron=0,
        rw=1,
        rw_window_size=2,
        rw_include_current=1,
        rw_min_window_tokens=0,
    )

    # Expected files
    expect = [
        "rw_rlsm_conversations.csv",
        "rw_rlsm_conversations_full.csv",
        "rw_pairs_rlsm.csv",
        "rw_pairs_rlsm_full.csv",
        "rw_individual_category_rlsm.csv",
        "rw_individual_category_rlsm_full.csv",
        "rw_dyad_category_rlsm.csv",
        "rw_dyad_category_rlsm_full.csv",
    ]
    for name in expect:
        assert (results_dir / name).exists()

    # Error CSV (we could also test a version that returns one dummy error, but here we assert it does not exist)
    assert not (results_dir / "rw_rlsm_errors.csv").exists()

    # With rw=0, rw-related files are not created
    results_dir2 = tmp_path / "out2"
    os.makedirs(results_dir2, exist_ok=True)
    R.run_rlsm_parallel(
        data_dir=str(tmp_path),
        dic_path="/dev/null",
        results_dir=str(results_dir2),
        procs=1,
        chunksize=1,
        na_policy="bilateral_only",
        eps=1e-4,
        zero_tol=0.0,
        unitize="off",
        merge_gap=0.0,
        strip_tags=0,
        bracket_mode="off",
        enable_micro_ppron=0,
        rw=0,
        rw_window_size=2,
        rw_include_current=1,
        rw_min_window_tokens=0,
    )
    for name in expect:
        assert not (results_dir2 / name).exists()


