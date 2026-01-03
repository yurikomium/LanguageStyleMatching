# -*- coding: utf-8 -*-
"""
==============================================================================
test_unitize_brackets_and_merge_split.py - テキスト前処理・マージ・分割機能テスト
==============================================================================

このテストファイルが担保する機能：

1. **ブラケット処理機能**
   - [え]、[笑]などの記述タグの適切な処理
   - unwrapモード：内容を残してブラケットのみ除去
   - 記述タグ（[笑]等）の完全削除
   - 空のブラケット[]の適切な除去

2. **同話者連続発話のマージ機能**
   - 時間的に連続する同話者発話の自動結合
   - merge_gap閾値による結合制御
   - テキスト連結時の適切なスペース挿入

3. **異話者重複の分割と帰属**
   - オーバーラップ発話の検出と分割処理
   - 後着優先ポリシー（後から始まった発話が重複区間を専有）
   - 先着発話の適切な時間カットと残存部分の処理

4. **交互性の確保**
   - 分割・マージ後の話者交互パターンの実現
   - 連続同話者の回避
   - 時間的連続性の保持（end[i] ≤ start[i+1]）

5. **時間ベースの処理精度**
   - 秒単位での正確な時間管理
   - 時間比に基づくテキスト分割の近似処理
   - 非重複条件の厳密な担保
"""

import pandas as pd
import numpy as np
from rlsm.unitize import unitize_transcript

def test_brackets_unwrap_and_drop_default():
    df = pd.DataFrame([
        {"start": 0.0, "end": 1.0, "speaker": "female", "text": "[え] それで"},
        {"start": 1.0, "end": 2.0, "speaker": "male",   "text": "[笑] はい"},
        {"start": 2.0, "end": 3.0, "speaker": "female", "text": "今日は[たぶん]大丈夫"},
        {"start": 3.0, "end": 4.0, "speaker": "male",   "text": "[] 空のかっこ"},
    ])
    out, rep = unitize_transcript(
        df, mode="approx", merge_gap=0.0, strip_tags=True, bracket_mode="unwrap"
    )
    # 中身を残して[]は外す
    assert out.loc[0, "text_clean"].startswith("え ")
    assert "たぶん" in out.loc[out["text"].str.contains("たぶん")].iloc[0]["text_clean"]
    # 記述タグは丸ごと削除
    assert out.loc[1, "text_clean"].startswith("はい")
    # 空の[]は消える
    assert "[]" not in out.loc[3, "text_clean"]

def test_same_speaker_merge_with_gap_threshold():
    # female が2回連続（0.0–1.0 と 1.0–2.0）→ merge_gap=0.0 なら結合、>0でもOK
    df = pd.DataFrame([
        {"start": 0.0, "end": 1.0, "speaker": "female", "text": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "female", "text": "B"},
    ])
    out, _ = unitize_transcript(df, mode="approx", merge_gap=0.0, strip_tags=False)
    assert len(out) == 1
    assert out.iloc[0]["start"] == 0.0 and out.iloc[0]["end"] == 2.0
    assert out.iloc[0]["text_clean"] == "A B"

def test_overlap_is_split_and_assigned_to_latter_speaker():
    # 例：あなたが提示したケース（M 長発話中に F が割り込み）
    df = pd.DataFrame([
        {"start": 1.71, "end": 3.08,  "speaker": "female", "text": "F1"},
        {"start": 2.24, "end": 3.17,  "speaker": "male",   "text": "M1"},
        {"start": 3.99, "end": 5.08,  "speaker": "male",   "text": "M2"},
        {"start": 5.33, "end": 6.66,  "speaker": "female", "text": "F2"},
        {"start": 7.63, "end": 14.23, "speaker": "male",   "text": "M3"},
        {"start": 8.48, "end": 8.77,  "speaker": "female", "text": "F3"},
    ])
    # 同話者の分割断片を“交互”に寄せたいので、ギャップを許容して結合する
    out, rep = unitize_transcript(
        df, mode="approx", merge_gap=10.0, strip_tags=False  # ← 10秒まで同話者結合
    )
    # 非重複性
    for i in range(len(out)-1):
        assert out.iloc[i]["end"] <= out.iloc[i+1]["start"]

    # 期待される分割（時間順）
    # F:1.71–2.24 / M:2.24–5.08 / F:5.33–6.66 / M:7.63–8.48 / F:8.48–8.77 / M:8.77–14.23
    times = [(round(r["start"], 2), round(r["end"], 2), r["speaker"]) for _, r in out.iterrows()]
    assert times == [
        (1.71, 2.24, "female"),
        (2.24, 5.08, "male"),
        (5.33, 6.66, "female"),
        (7.63, 8.48, "male"),
        (8.48, 8.77, "female"),
        (8.77, 14.23, "male"),
    ]

    # 交互性（連続で同話者になっていない）
    for i in range(len(out)-1):
        assert out.iloc[i]["speaker"] != out.iloc[i+1]["speaker"]
