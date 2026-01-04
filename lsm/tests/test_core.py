# exploration/Transcript/lsm/tests/test_liwc_lsm.py
import math
import pytest

from lsm.core import (
    compute_lsm,
    match_token_to_categories,
)

def test_compute_lsm_exact_values_basic():
    # Use 2 categories (ppron/ipron). Build p from total words and category counts.
    # Case A: same rates -> LSM=1
    target = ["ppron", "ipron"]
    c1 = {"ppron": 10, "ipron": 5};  tot1 = 100
    c2 = {"ppron": 20, "ipron": 10}; tot2 = 200  # Same proportions (0.10, 0.05)
    scores = compute_lsm(c1, c2, target, tot1, tot2, rounding=None)
    assert scores["ppron"] == pytest.approx(1.0)
    assert scores["ipron"]  == pytest.approx(1.0)

    # Case B: only one side > 0 -> LSM=0
    c1 = {"ppron": 0}; tot1 = 100
    c2 = {"ppron": 10}; tot2 = 100
    scores = compute_lsm(c1, c2, ["ppron"], tot1, tot2, rounding=None)
    assert scores["ppron"] == pytest.approx(0.0)

    # Case C: p1=0.04, p2=0.06 -> 1 - 0.02/0.10 = 0.8
    c1 = {"ppron": 4};  tot1 = 100
    c2 = {"ppron": 6};  tot2 = 100
    scores = compute_lsm(c1, c2, ["ppron"], tot1, tot2, rounding=None)
    assert scores["ppron"] == pytest.approx(0.8, rel=1e-9)

def test_compute_lsm_zero_category_excluded_from_mean():
    # Categories that are 0% for both speakers are omitted from the dict (= excluded from mean)
    target = ["ppron", "ipron"]
    c1 = {"ppron": 10, "ipron": 0}; tot1 = 100
    c2 = {"ppron": 20, "ipron": 0}; tot2 = 200
    scores = compute_lsm(c1, c2, target, tot1, tot2, rounding=None)
    assert "ipron" not in scores            # Both 0% -> excluded
    assert scores["ppron"] == pytest.approx(1.0)

def test_compute_lsm_bounds_and_symmetry():
    target = ["ppron"]
    c1 = {"ppron": 3}; tot1 = 50
    c2 = {"ppron": 10}; tot2 = 100
    s12 = compute_lsm(c1, c2, target, tot1, tot2, rounding=None)["ppron"]
    s21 = compute_lsm(c2, c1, target, tot2, tot1, rounding=None)["ppron"]
    assert 0.0 <= s12 <= 1.0
    assert s12 == pytest.approx(s21)        # Symmetry

def test_match_token_to_categories_wildcard(compiled_patterns):
    # Verify wildcard token matching and normal token matching using the Japanese examples below
    cats1 = match_token_to_categories("私", compiled_patterns, cache={})
    cats2 = match_token_to_categories("私たち", compiled_patterns, cache={})
    cats3 = match_token_to_categories("自分", compiled_patterns, cache={})
    cats4 = match_token_to_categories("何か", compiled_patterns, cache={})

    assert "ppron" in cats1 and "ppron" in cats2
    assert "ppron" in cats3
    assert "ipron" in cats4
