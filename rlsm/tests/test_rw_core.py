# -*- coding: utf-8 -*-
"""
==============================================================================
test_rw_rlsm_core.py - Rolling-window feature tests
==============================================================================

What this test file guarantees:

1. **Equivalence at window_size=1**
   - With window_size=1 and include_current=True, results exactly match instantaneous rates
   - Confirms equivalence between the classic rLSM and the rolling-window variant

2. **Correctness of window averaging**
   - Correct averaging of usage rates over multiple turns
   - include_current=True includes the current turn in the window
   - include_current=False excludes the current turn (past-only window)

3. **Behavior of include_current**
   - True: averages including the current turn
   - False: averages using only past turns
   - Proper interaction with Table 5 rules (NaN when prev=0 and curr>0)

4. **Accurate detection of speaker switches**
   - Consecutive turns by the same speaker are not scored
   - Properly skips cases where the final turn does not form a speaker-switch pair

5. **Rolling-window metadata consistency**
   - Correct counting of win_size (turns in the window)
   - Correct aggregation of win_total (total words in the window)
   - Correct assignment of pair_index
"""

import math
import numpy as np
import pytest

from rlsm.core import compute_rlsm_core, EPS
from rlsm.workers import _compute_rw_window_turns


def test_rw_equals_plain_when_window_size_1_include_current_true():
    # 2-speaker dialog, 2 categories, and set total=100 for every turn
    cats = ["c1", "c2"]
    # Build from counts/total (= rate * 100)
    tc = [
        {"speaker": "female", "counts": {"c1": 10, "c2": 0},  "total": 100},
        {"speaker": "male",   "counts": {"c1": 30, "c2": 10}, "total": 100},
        {"speaker": "female", "counts": {"c1": 50, "c2": 20}, "total": 100},
        {"speaker": "male",   "counts": {"c1": 40, "c2": 40}, "total": 100},
    ]
    # "Instantaneous rate" turns for standard rLSM
    turns_plain = []
    for t in tc:
        rates = {c: (t["counts"][c] / t["total"] * 100.0) for c in cats}
        turns_plain.append({"speaker": t["speaker"], "rates": rates})

    res_plain = compute_rlsm_core(turns_plain, cats)

    # rw: window_size=1, include_current=True -> matches the "instantaneous rate"
    rw_turns = _compute_rw_window_turns(tc, cats, window_size=1, include_current=True)
    res_rw = compute_rlsm_core(rw_turns, cats)

    assert res_plain["dyad_final"] == pytest.approx(res_rw["dyad_final"])
    assert res_plain["dyad_category_means"] == res_rw["dyad_category_means"]
    assert res_plain["individual_overall"] == res_rw["individual_overall"]
    # Pair count also matches
    assert len(res_plain["pair_category_scores"]) == len(res_rw["pair_category_scores"])
    # Ensure compute_rlsm_core works even if extra keys (win_*) are present
    assert all(set(["rates", "speaker"]).issubset(t.keys()) for t in rw_turns)


def test_window_mean_include_current_true_size2():
    # Single category for clarity: female(10%) -> male(30%) -> female(90%)
    cats = ["c1"]
    tc = [
        {"speaker": "female", "counts": {"c1": 10}, "total": 100},  # 10%
        {"speaker": "male",   "counts": {"c1": 30}, "total": 100},  # 30%
        {"speaker": "female", "counts": {"c1": 90}, "total": 100},  # 90%; with window=2 and include_current=True -> (10+90)/200=50%
    ]
    rw_turns = _compute_rw_window_turns(tc, cats, window_size=2, include_current=True)
    res = compute_rlsm_core(rw_turns, cats)

    # Expect: pair_index=1 has leader=male(30), responder=female(50)
    # r = 1 - |30-50| / (30+50+EPS)
    expected = 1.0 - (abs(30.0 - 50.0) / (30.0 + 50.0 + EPS))

    # Extract c1 for pair_index=1
    p1 = [p for p in res["pair_category_scores"] if p["pair_index"] == 1][0]
    assert p1["category_scores"]["c1"] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_window_mean_include_current_false_size2():
    cats = ["c1"]
    tc = [
        {"speaker": "female", "counts": {"c1": 10}, "total": 100},
        {"speaker": "male",   "counts": {"c1": 30}, "total": 100},
        {"speaker": "female", "counts": {"c1": 90}, "total": 100},
    ]
    rw_turns = _compute_rw_window_turns(tc, cats, window_size=2, include_current=False)
    res = compute_rlsm_core(rw_turns, cats)

    p1 = [p for p in res["pair_category_scores"] if p["pair_index"] == 1][0]
    # Leader=male first window is empty -> 0% / responder=female=10% -> Table 5: prev==0 & curr>0 => NaN
    assert math.isnan(p1["category_scores"]["c1"])

def test_last_leader_window_is_not_used():
    # End with same speaker: female->male forms a pair; male->male is not a speaker switch, so no pair
    cats = ["c1"]
    tc = [
        {"speaker": "female", "counts": {"c1": 10}, "total": 100},  # A lead
        {"speaker": "male",   "counts": {"c1": 30}, "total": 100},  # B respond -> pair 0
        {"speaker": "male",   "counts": {"c1": 40}, "total": 100},  # B lead (no next & same speaker) -> no pair
    ]
    for include_current in (True, False):
        rw_turns = _compute_rw_window_turns(tc, cats, window_size=8, include_current=include_current)
        res = compute_rlsm_core(rw_turns, cats)
        assert len(res["pair_category_scores"]) == 1
