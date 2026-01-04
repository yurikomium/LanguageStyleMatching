# -*- coding: utf-8 -*-
"""
==============================================================================
test_rlsm_core.py - Core behavior tests for the rLSM algorithm
==============================================================================

What this test file guarantees:

1. **Table 5 rule compliance**
   - prev=0, curr=0 -> NaN (excluded as missing)
   - prev=0, curr>0 -> NaN (excluded as missing)
   - prev>0, curr=0 -> computed normally (low rLSM)

2. **Compute only speaker-switch pairs**
   - Consecutive turns by the same speaker are not scored
   - rLSM is computed only for adjacent pairs where the speaker changes

3. **Attribution to the responder (later turn)**
   - rLSM scores are attributed to the responder (the later turn in the pair)
   - Individual aggregation collects only the responder-attributed scores

4. **Correct hierarchical aggregation**
   - Individual x category means are computed correctly
   - Dyad x category means (average over both speakers) are computed correctly
   - Final dyad score (equal-weight mean over categories) is computed correctly

5. **Edge cases: single speaker / insufficient data**
   - Returns NaN when only one speaker is present
   - NaN-handling in category averaging behaves as intended
"""

import math
import numpy as np
import pytest

# Test target
from rlsm.core import (
    rlsm_per_category_from_rates,
    compute_pair_category_rlsm,
    aggregate_individual_category_means,
    aggregate_dyad_category_means,
    mean_over_categories,
    compute_rlsm_core,
    EPS,
)

CATS = ["ppron", "ipron"]  # Validate with the minimal 2 categories (in practice, passing 7 categories is OK)


# ==============
# Table 5 missingness rules (single-exchange level)
# ==============
def test_table5_case_both_zero_is_nan():
    prev = {"ppron": 0.0, "ipron": 3.0}
    curr = {"ppron": 0.0, "ipron": 3.0}
    out = rlsm_per_category_from_rates(prev, curr, CATS)
    assert math.isnan(out["ppron"]), "prev=0 and curr=0 should be NaN (excluded)"
    # ipron is computed (validated separately in a later case)


def test_table5_case_prev_zero_curr_positive_is_nan():
    prev = {"ppron": 0.0}
    curr = {"ppron": 5.0}
    out = rlsm_per_category_from_rates(prev, curr, ["ppron"])
    assert math.isnan(out["ppron"]), "prev=0 and curr>0 should also be NaN (excluded)"


def test_table5_case_prev_positive_curr_zero_is_low_value():
    prev = {"ppron": 10.0}
    curr = {"ppron": 0.0}
    out = rlsm_per_category_from_rates(prev, curr, ["ppron"])
    # Expected: 1 - 10/(10+0+EPS) = 1 - 10/10.0001 ≈ 1e-5
    expected = 1.0 - (10.0 / (10.0 + EPS))
    assert out["ppron"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ==============
# Compute only speaker-switch pairs & attribute to the responder (the later turn)
# ==============
def test_only_speaker_changes_are_counted_and_scores_belong_to_responder():
    turns = [
        {"speaker": "A", "rates": {"ppron": 10.0, "ipron": 0.0}},
        {"speaker": "A", "rates": {"ppron": 9.0,  "ipron": 0.0}},  # Same speaker -> skip
        {"speaker": "B", "rates": {"ppron": 10.0, "ipron": 5.0}},  # ← speaker switch happens here (A→B)
        {"speaker": "A", "rates": {"ppron": 0.0,  "ipron": 5.0}},  # ← speaker switch happens here (B→A)
    ]
    pair_scores = compute_pair_category_rlsm(turns, CATS)

    # Only 2 switches
    assert len(pair_scores) == 2

    # First switch (A→B): responder is B
    assert pair_scores[0]["leader"] == "A"
    assert pair_scores[0]["responder"] == "B"

    # Second switch (B→A): responder is A
    assert pair_scores[1]["leader"] == "B"
    assert pair_scores[1]["responder"] == "A"


def test_individual_category_means_are_for_responder_only():
    # Keep the example simple so we can track values precisely
    turns = [
        {"speaker": "A", "rates": {"ppron": 10.0, "ipron": 0.0}},   # lead
        {"speaker": "B", "rates": {"ppron": 10.0, "ipron": 5.0}},   # respond(B)
        {"speaker": "A", "rates": {"ppron": 0.0,  "ipron": 5.0}},   # respond(A)
    ]
    pair_scores = compute_pair_category_rlsm(turns, CATS)
    indiv = aggregate_individual_category_means(pair_scores, CATS)

    # B's ppron is 1.0 (10 vs 10)
    assert indiv["B"]["ppron"] == pytest.approx(1.0)

    # B's ipron has prev=0, curr=5 -> NaN (not included for responder B)
    assert math.isnan(indiv["B"]["ipron"])

    # A's ppron has prev=10, curr=0 -> small value (~1e-5)
    expected_ppron_A = 1.0 - (10.0 / (10.0 + EPS))
    assert indiv["A"]["ppron"] == pytest.approx(expected_ppron_A, rel=1e-9, abs=1e-12)

    # A's ipron is 5 vs 5 -> 1.0
    assert indiv["A"]["ipron"] == pytest.approx(1.0)


# ==============
# Category-level dyad -> final dyad (mean over categories)
# ==============
def test_dyad_category_and_final_under_bilateral_only():
    # Continuation of the case above (A/B each responds once)
    turns = [
        {"speaker": "A", "rates": {"ppron": 10.0, "ipron": 0.0}},
        {"speaker": "B", "rates": {"ppron": 10.0, "ipron": 5.0}},  # For B: ppron=1.0, ipron=NaN
        {"speaker": "A", "rates": {"ppron": 0.0,  "ipron": 5.0}},  # For A: ppron≈1e-5, ipron=1.0
    ]
    res = compute_rlsm_core(turns, CATS, na_policy="bilateral_only")
    indiv = res["individual_category_means"]
    dyad_cat = res["dyad_category_means"]
    final = res["dyad_final"]

    # Dyad by category:
    # ppron = mean( B=1.0, A≈1e-5 ) ≈ 0.500005
    expected_ppron_A = 1.0 - (10.0 / (10.0 + EPS))
    expected_dyad_ppron = (1.0 + expected_ppron_A) / 2.0
    assert dyad_cat["ppron"] == pytest.approx(expected_dyad_ppron, rel=1e-9, abs=1e-12)

    # ipron: B is NaN, A is 1.0 -> bilateral_only => NaN
    assert math.isnan(dyad_cat["ipron"])

    # Final: mean over valid categories only (= ppron only)
    assert final == pytest.approx(expected_dyad_ppron, rel=1e-9, abs=1e-12)

    # Individual overall (derived): mean of (individual x category means) over categories
    # A: mean( ppron≈1e-5, ipron=1.0 ) ≈ 0.500005
    expected_A_overall = np.nanmean([expected_ppron_A, 1.0])
    assert res["individual_overall"]["A"] == pytest.approx(expected_A_overall, rel=1e-9, abs=1e-12)
    # B: mean( ppron=1.0, ipron=NaN ) = 1.0
    assert res["individual_overall"]["B"] == pytest.approx(1.0)


def test_dyad_category_and_final_under_nanmean():
    turns = [
        {"speaker": "A", "rates": {"ppron": 10.0, "ipron": 0.0}},
        {"speaker": "B", "rates": {"ppron": 10.0, "ipron": 5.0}},
        {"speaker": "A", "rates": {"ppron": 0.0,  "ipron": 5.0}},
    ]
    res = compute_rlsm_core(turns, CATS, na_policy="nanmean")
    dyad_cat = res["dyad_category_means"]
    final = res["dyad_final"]

    # ppron is the same as above
    expected_ppron_A = 1.0 - (10.0 / (10.0 + EPS))
    expected_dyad_ppron = (1.0 + expected_ppron_A) / 2.0
    assert dyad_cat["ppron"] == pytest.approx(expected_dyad_ppron, rel=1e-9, abs=1e-12)

    # ipron: with nanmean, mean(1.0, NaN) = 1.0
    assert dyad_cat["ipron"] == pytest.approx(1.0)

    # Final is the average over 2 categories -> (0.500005.. + 1.0)/2 ≈ 0.750005..
    expected_final = np.nanmean([expected_dyad_ppron, 1.0])
    assert final == pytest.approx(expected_final, rel=1e-9, abs=1e-12)


# ==============
# When only one speaker is present
# ==============
def test_single_speaker_returns_nans():
    turns = [
        {"speaker": "A", "rates": {"ppron": 5.0, "ipron": 2.0}},
        {"speaker": "A", "rates": {"ppron": 6.0, "ipron": 1.0}},
    ]
    res = compute_rlsm_core(turns, CATS)
    assert res["pair_category_scores"] == []
    assert all(math.isnan(v) for v in res["dyad_category_means"].values())
    assert math.isnan(res["dyad_final"])
    assert res["individual_category_means"] == {}
    assert res["individual_overall"] == {}


# ==============
# Basic properties of mean_over_categories
# ==============
def test_mean_over_categories_ignores_nans():
    vals = {"ppron": 1.0, "ipron": np.nan}
    m = mean_over_categories(vals)
    assert m == pytest.approx(1.0)

    vals = {"ppron": np.nan, "ipron": np.nan}
    m = mean_over_categories(vals)
    assert math.isnan(m)
