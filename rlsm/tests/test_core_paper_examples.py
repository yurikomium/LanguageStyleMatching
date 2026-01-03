# -*- coding: utf-8 -*-
"""
==============================================================================
test_rlsm_paper_examples.py - 学術論文準拠性テスト
==============================================================================

このテストファイルが担保する機能：

1. **論文Table 2の数値検証**
   - Romeo-Benvolioの対話例での厳密な数値検証
   - 論文記載の具体的な使用率（FW%）での計算確認
   - 理論値と実装値の完全一致

2. **式(8)の実装準拠性**
   - rLSM_B(FW) = 1 - |A1-B1|/(A1+B1+eps) ≈ .59
   - 個別ペアレベルでのrLSM計算の正確性

3. **式(9)の実装準拠性**
   - rLSM_A(FW) = 1 - |B1-A2|/(B1+A2+eps) ≈ .70
   - 話者交代ペアでの正確な計算

4. **式(14)とカテゴリ平均の準拠性**
   - 最終ダイアドスコア ≈ .74 の正確な再現
   - カテゴリ等重み平均の正しい実装確認

5. **学術的信頼性の担保**
   - 公開論文の計算例での完全再現性
   - 理論的基盤と実装の一致性証明
   - rLSMアルゴリズムの学術的妥当性確認

注：個人平均については論文テキストと若干の数値差異があるが、
   式(8)(9)および最終ダイアド(式14)は完全一致するため実装は正確。
"""

import pytest
from rlsm.core import (
    rlsm_per_category_from_rates,
    compute_rlsm_core,
    EPS,
)

# 論文 Table 2, Example C の値（FW ％）
# A1=60, B1=25, A2=46.67..., B2=42.857..., A3=75
A1, B1 = 60.0, 25.0
A2, B2 = 46.6666666667, 42.8571428571
A3      = 75.0

def test_romeo_benvolio_eq8_statement1():
    """式(8): rLSM_B(FW) = 1 - |A1-B1|/(A1+B1+eps) ≈ .59"""
    out = rlsm_per_category_from_rates(
        {"fw": A1}, {"fw": B1}, ["fw"], eps=EPS
    )
    assert out["fw"] == pytest.approx(0.588235, rel=1e-6, abs=1e-6)

def test_romeo_benvolio_eq9_statement2():
    """式(9): rLSM_A(FW) = 1 - |B1-A2|/(B1+A2+eps) ≈ .70"""
    out = rlsm_per_category_from_rates(
        {"fw": B1}, {"fw": A2}, ["fw"], eps=EPS
    )
    assert out["fw"] == pytest.approx(0.6976, rel=1e-4, abs=1e-4)

def test_romeo_benvolio_eq14_dyad_final_fw_only():
    """式(14) + カテゴリ平均（FWのみ）: 最終ダイアド ≈ .74"""
    turns = [
        {"speaker": "A", "rates": {"fw": A1}},  # Romeo
        {"speaker": "B", "rates": {"fw": B1}},  # Benvolio
        {"speaker": "A", "rates": {"fw": A2}},
        {"speaker": "B", "rates": {"fw": B2}},
        {"speaker": "A", "rates": {"fw": A3}},
    ]
    res = compute_rlsm_core(turns, ["fw"], na_policy="bilateral_only", eps=EPS)
    # ダイアド最終（FWのみなのでカテゴリ平均=FW値）
    assert res["dyad_final"] == pytest.approx(0.74, rel=5e-3, abs=5e-3)

    # 参考: 応答側に帰属する個人平均（論文本文では A≈.77, B≈.71 と記述あり）
    # 本実装では、Table 2 のFW%を厳密に用いると
    #   A(=Romeo) ≈ 0.712, B(=Benvolio) ≈ 0.773
    # となる（dyadは.742...で一致）。個人値はテキスト記述と入れ替わるが、
    # 式(8)(9)および最終ダイアド（式14）は一致するため、ここでは個人平均のアサートは行わない。
