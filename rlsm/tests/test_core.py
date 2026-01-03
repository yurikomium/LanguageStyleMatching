# -*- coding: utf-8 -*-
"""
==============================================================================
test_rlsm_core.py - RLSMコアアルゴリズムの基本動作テスト
==============================================================================

このテストファイルが担保する機能：

1. **Table 5規則の実装担保**
   - 前回値=0, 今回値=0 → NaN（欠損として除外）
   - 前回値=0, 今回値>0 → NaN（欠損として除外）  
   - 前回値>0, 今回値=0 → 通常計算（低いrLSM値）

2. **話者交代ペアのみの計算**
   - 同一話者の連続発話は計算対象外であること
   - 話者が交代した隣接ペアのみでrLSMを計算すること

3. **応答側（後手）への帰属**
   - rLSMスコアが応答者（発話の後手）に正しく帰属されること
   - 個人別集計で応答側のスコアのみが集計されること

4. **階層的集計の正確性**
   - 個人×カテゴリ平均が正しく計算されること
   - ダイアド×カテゴリ平均（両話者の平均）が正しく算出されること
   - 最終ダイアドスコア（カテゴリ等重み平均）が正しく計算されること

5. **単一話者・データ不足時の処理**
   - 1人しか話者がいない場合にNaNを返すこと
   - カテゴリ平均でのNaN無視処理が正しく動作すること
"""

import math
import numpy as np
import pytest

# テスト対象
from rlsm.core import (
    rlsm_per_category_from_rates,
    compute_pair_category_rlsm,
    aggregate_individual_category_means,
    aggregate_dyad_category_means,
    mean_over_categories,
    compute_rlsm_core,
    EPS,
)

CATS = ["ppron", "ipron"]  # 最小2カテゴリで検証（実際は7カテゴリを渡せばOK）


# ==============
# Table 5 の欠損規則（1往復レベル）
# ==============
def test_table5_case_both_zero_is_nan():
    prev = {"ppron": 0.0, "ipron": 3.0}
    curr = {"ppron": 0.0, "ipron": 3.0}
    out = rlsm_per_category_from_rates(prev, curr, CATS)
    assert math.isnan(out["ppron"]), "前=0,今=0 は NaN（除外）"
    # ipron は計算される（後のケースで別途検証）


def test_table5_case_prev_zero_curr_positive_is_nan():
    prev = {"ppron": 0.0}
    curr = {"ppron": 5.0}
    out = rlsm_per_category_from_rates(prev, curr, ["ppron"])
    assert math.isnan(out["ppron"]), "前=0,今>0 も NaN（除外）"


def test_table5_case_prev_positive_curr_zero_is_low_value():
    prev = {"ppron": 10.0}
    curr = {"ppron": 0.0}
    out = rlsm_per_category_from_rates(prev, curr, ["ppron"])
    # 期待値: 1 - 10/(10+0+EPS) = 1 - 10/10.0001 ≈ 1e-5
    expected = 1.0 - (10.0 / (10.0 + EPS))
    assert out["ppron"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ==============
# 話者交代ペアのみ計算 & 応答側（後手）への帰属
# ==============
def test_only_speaker_changes_are_counted_and_scores_belong_to_responder():
    turns = [
        {"speaker": "A", "rates": {"ppron": 10.0, "ipron": 0.0}},
        {"speaker": "A", "rates": {"ppron": 9.0,  "ipron": 0.0}},  # 同一話者→スキップ
        {"speaker": "B", "rates": {"ppron": 10.0, "ipron": 5.0}},  # ← ここで交代(A→B)
        {"speaker": "A", "rates": {"ppron": 0.0,  "ipron": 5.0}},  # ← ここで交代(B→A)
    ]
    pair_scores = compute_pair_category_rlsm(turns, CATS)

    # 交代は2回だけ
    assert len(pair_scores) == 2

    # 1つめの交代(A→B)：応答はB
    assert pair_scores[0]["leader"] == "A"
    assert pair_scores[0]["responder"] == "B"

    # 2つめの交代(B→A)：応答はA
    assert pair_scores[1]["leader"] == "B"
    assert pair_scores[1]["responder"] == "A"


def test_individual_category_means_are_for_responder_only():
    # 例をシンプルに、数値を厳密に追えるように設定
    turns = [
        {"speaker": "A", "rates": {"ppron": 10.0, "ipron": 0.0}},   # lead
        {"speaker": "B", "rates": {"ppron": 10.0, "ipron": 5.0}},   # respond(B)
        {"speaker": "A", "rates": {"ppron": 0.0,  "ipron": 5.0}},   # respond(A)
    ]
    pair_scores = compute_pair_category_rlsm(turns, CATS)
    indiv = aggregate_individual_category_means(pair_scores, CATS)

    # Bのppronは1.0（10 vs 10）
    assert indiv["B"]["ppron"] == pytest.approx(1.0)

    # Bのipronは前=0,今=5 → NaN（応答側Bには入らない）
    assert math.isnan(indiv["B"]["ipron"])

    # Aのppronは前=10,今=0 → 低値（約1e-5）
    expected_ppron_A = 1.0 - (10.0 / (10.0 + EPS))
    assert indiv["A"]["ppron"] == pytest.approx(expected_ppron_A, rel=1e-9, abs=1e-12)

    # Aのipronは5 vs 5 → 1.0
    assert indiv["A"]["ipron"] == pytest.approx(1.0)


# ==============
# カテゴリ別ダイアド → 最終ダイアド（カテゴリ平均）
# ==============
def test_dyad_category_and_final_under_bilateral_only():
    # 上のケースの続き（A/Bそれぞれ1回ずつ応答）
    turns = [
        {"speaker": "A", "rates": {"ppron": 10.0, "ipron": 0.0}},
        {"speaker": "B", "rates": {"ppron": 10.0, "ipron": 5.0}},  # Bにppron=1.0, ipron=NaN
        {"speaker": "A", "rates": {"ppron": 0.0,  "ipron": 5.0}},  # Aにppron≈1e-5, ipron=1.0
    ]
    res = compute_rlsm_core(turns, CATS, na_policy="bilateral_only")
    indiv = res["individual_category_means"]
    dyad_cat = res["dyad_category_means"]
    final = res["dyad_final"]

    # カテゴリ別ダイアド:
    # ppron = mean( B=1.0, A≈1e-5 ) ≈ 0.500005
    expected_ppron_A = 1.0 - (10.0 / (10.0 + EPS))
    expected_dyad_ppron = (1.0 + expected_ppron_A) / 2.0
    assert dyad_cat["ppron"] == pytest.approx(expected_dyad_ppron, rel=1e-9, abs=1e-12)

    # ipron: BがNaN, Aが1.0 → bilateral_onlyではNaN
    assert math.isnan(dyad_cat["ipron"])

    # 最終：カテゴリ平均（有効カテゴリのみ= ppronのみ）
    assert final == pytest.approx(expected_dyad_ppron, rel=1e-9, abs=1e-12)

    # 個人総合（派生）：個人×カテゴリ平均のカテゴリ平均
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

    # ppron は上と同じ
    expected_ppron_A = 1.0 - (10.0 / (10.0 + EPS))
    expected_dyad_ppron = (1.0 + expected_ppron_A) / 2.0
    assert dyad_cat["ppron"] == pytest.approx(expected_dyad_ppron, rel=1e-9, abs=1e-12)

    # ipron: nanmean では mean(1.0, NaN) = 1.0
    assert dyad_cat["ipron"] == pytest.approx(1.0)

    # 最終は 2カテゴリの平均 → (0.500005.. + 1.0)/2 ≈ 0.750005..
    expected_final = np.nanmean([expected_dyad_ppron, 1.0])
    assert final == pytest.approx(expected_final, rel=1e-9, abs=1e-12)


# ==============
# 片方しか話者がいない場合
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
# mean_over_categories の基本性質
# ==============
def test_mean_over_categories_ignores_nans():
    vals = {"ppron": 1.0, "ipron": np.nan}
    m = mean_over_categories(vals)
    assert m == pytest.approx(1.0)

    vals = {"ppron": np.nan, "ipron": np.nan}
    m = mean_over_categories(vals)
    assert math.isnan(m)
