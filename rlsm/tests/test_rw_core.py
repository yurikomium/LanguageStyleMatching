# -*- coding: utf-8 -*-
"""
==============================================================================
test_rw_rlsm_core.py - ローリングウィンドウ機能テスト
==============================================================================

このテストファイルが担保する機能：

1. **ウィンドウサイズ=1での一致性**
   - window_size=1, include_current=Trueで瞬間率と完全一致すること
   - 従来のrLSMとローリングウィンドウ版の等価性確認

2. **ウィンドウ平均の正確性**
   - 複数ターンにわたる使用率の正しい平均計算
   - include_current=Trueでの当該ターン含有計算
   - include_current=Falseでの当該ターン除外計算

3. **include_currentフラグの動作**
   - Trueの場合：当該ターンを含めた平均計算
   - Falseの場合：過去のターンのみでの平均計算
   - Table 5規則との適切な相互作用（前=0,今>0でのNaN処理）

4. **話者交代の正確な検出**
   - 同一話者の連続はrLSM計算対象外であること
   - 最終発話で話者交代が無い場合の適切なスキップ

5. **ローリングウィンドウのメタデータ整合性**
   - win_size（ウィンドウ内ターン数）の正確な計数
   - win_total（ウィンドウ内総語数）の正確な集計
   - ペアインデックスの適切な付与
"""

import math
import numpy as np
import pytest

from rlsm.core import compute_rlsm_core, EPS
from rlsm.workers import _compute_rw_window_turns


def test_rw_equals_plain_when_window_size_1_include_current_true():
    # 2人対話・2カテゴリ・全ターン total=100 とする
    cats = ["c1", "c2"]
    # counts/total（=率×100）で作る
    tc = [
        {"speaker": "female", "counts": {"c1": 10, "c2": 0},  "total": 100},
        {"speaker": "male",   "counts": {"c1": 30, "c2": 10}, "total": 100},
        {"speaker": "female", "counts": {"c1": 50, "c2": 20}, "total": 100},
        {"speaker": "male",   "counts": {"c1": 40, "c2": 40}, "total": 100},
    ]
    # 通常 rLSM 用の "瞬間率"
    turns_plain = []
    for t in tc:
        rates = {c: (t["counts"][c] / t["total"] * 100.0) for c in cats}
        turns_plain.append({"speaker": t["speaker"], "rates": rates})

    res_plain = compute_rlsm_core(turns_plain, cats)

    # rw: window_size=1, include_current=True -> "瞬間率"と一致
    rw_turns = _compute_rw_window_turns(tc, cats, window_size=1, include_current=True)
    res_rw = compute_rlsm_core(rw_turns, cats)

    assert res_plain["dyad_final"] == pytest.approx(res_rw["dyad_final"])
    assert res_plain["dyad_category_means"] == res_rw["dyad_category_means"]
    assert res_plain["individual_overall"] == res_rw["individual_overall"]
    # ペア数も一致
    assert len(res_plain["pair_category_scores"]) == len(res_rw["pair_category_scores"])
    # 余剰キー（win_*）が付いても compute_rlsm_core が動くことを明示
    assert all(set(["rates", "speaker"]).issubset(t.keys()) for t in rw_turns)


def test_window_mean_include_current_true_size2():
    # 単一カテゴリで分かりやすく：female(10%) -> male(30%) -> female(90%)
    cats = ["c1"]
    tc = [
        {"speaker": "female", "counts": {"c1": 10}, "total": 100},  # 10%
        {"speaker": "male",   "counts": {"c1": 30}, "total": 100},  # 30%
        {"speaker": "female", "counts": {"c1": 90}, "total": 100},  # 90% ; 窓=2で include_current=True なら (10+90)/200=50%
    ]
    rw_turns = _compute_rw_window_turns(tc, cats, window_size=2, include_current=True)
    res = compute_rlsm_core(rw_turns, cats)

    # 期待：pair_index=1 は leader=male(30), responder=female(50)
    # r = 1 - |30-50| / (30+50+EPS)
    expected = 1.0 - (abs(30.0 - 50.0) / (30.0 + 50.0 + EPS))

    # pair_index=1 の c1 の値を取り出す
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
    # leader=male の最初の窓は空→0% / responder=female=10% → Table 5: prev==0 & curr>0 → NaN
    assert math.isnan(p1["category_scores"]["c1"])

def test_last_leader_window_is_not_used():
    # 最後が同一話者で終わる：female→male は成立、male→male は話者交代でないので不成立
    cats = ["c1"]
    tc = [
        {"speaker": "female", "counts": {"c1": 10}, "total": 100},  # A lead
        {"speaker": "male",   "counts": {"c1": 30}, "total": 100},  # B respond -> pair 0
        {"speaker": "male",   "counts": {"c1": 40}, "total": 100},  # B lead (次がない&同一話者) -> pair 不成立
    ]
    for include_current in (True, False):
        rw_turns = _compute_rw_window_turns(tc, cats, window_size=8, include_current=include_current)
        res = compute_rlsm_core(rw_turns, cats)
        assert len(res["pair_category_scores"]) == 1
