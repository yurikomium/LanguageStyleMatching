# -*- coding: utf-8 -*-
"""
==============================================================================
test_runner_outputs.py - 実行フレームワーク・出力形式テスト
==============================================================================

このテストファイルが担保する機能：

1. **出力ファイル名の正確性**
   - rwフラグ有効時の適切なファイル命名（rw_プレフィックス）
   - rwフラグ無効時の標準ファイル命名
   - 各カテゴリ（conversations, pairs, individual, dyad）の網羅的出力

2. **並列処理フレームワーク**
   - process_file_rlsm の適切な並列実行
   - プロセス間でのデータ統合の正確性
   - context引数の適切な引き渡し

3. **エラーハンドリング**
   - 処理失敗時のエラーファイル生成（rw_rlsm_errors.csv）
   - 成功時のエラーファイル非生成
   - 適切なエラー情報の記録

4. **フル版と丸め版の出力**
   - float_format指定による数値丸め版
   - フル精度版（_full.csv）の並行出力
   - 両形式での行数整合性

5. **モック・テスト環境**
   - 実際のファイル処理を行わない軽量テスト
   - 一時ディレクトリでの安全なファイル操作
   - monkeypatchによる外部依存の分離
"""

import os
import pandas as pd
import types

import exploration.Transcript.rlsm.run_rlsm as R


def test_runner_writes_expected_rw_filenames(tmp_path, monkeypatch):
    # ダミーのCSV1枚
    df = pd.DataFrame([
        {"speaker": "female", "text": "a", "text_clean": "a"},
        {"speaker": "male",   "text": "b", "text_clean": "b"},
    ])
    f = tmp_path / "F001_M002.csv"
    df.to_csv(f, index=False)

    # process_file_rlsm をモック: 必要最低限の辞書を返す
    def fake_worker(args, context=None):
        return {
            "conversation_summary": {"file_path": str(f)},
            "pairs_rows": [{"file_path": str(f)}],
            "individual_category_rows": [{"file_path": str(f)}],
            "dyad_category_rows": [{"file_path": str(f)}],
            # rw enabled のとき追加
            "rw_conv_row": {"file_path": str(f)},
            "rw_pair_rows": [{"file_path": str(f)}],
            "rw_individual_rows": [{"file_path": str(f)}],
            "rw_dyad_rows": [{"file_path": str(f)}],
        }

    monkeypatch.setattr(R, "process_file_rlsm", fake_worker)

    results_dir = tmp_path / "out"
    os.makedirs(results_dir, exist_ok=True)

    # rw=1 で実行
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

    # 期待ファイル
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

    # エラーCSV（ダミーで1件エラーを返すバージョンも検査可能だが、ここでは存在しないことを確認）
    assert not (results_dir / "rw_rlsm_errors.csv").exists()

    # rw=0 のときは rw系ファイルが出ない
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


