# test_rw_short_utterance_filter.py
import pandas as pd
import math
from rlsm.workers import process_file_rlsm

def test_short_utterances_removed_only_for_rw(monkeypatch, tmp_path):
    import rlsm.workers as W
    # 1語/2語/3語のダミー計数を吐くモック
    def fake_rates(text):
        n = int(text)  # "1","2","3" を total にする
        counts = {"ppron": n}  # 全部ppronに入れる（率は常に100%）
        rates = {"ppron": 100.0 if n>0 else 0.0}
        return (rates, n, counts)
    monkeypatch.setattr(W, "_rates_for_text", fake_rates)

    df = pd.DataFrame([
        {"speaker":"female","text":"1"},  # ← ≤2語でrwからは落ちる
        {"speaker":"male",  "text":"2"},
        {"speaker":"female","text":"3"},  # ← 残る
        {"speaker":"male",  "text":"3"},
    ])
    f = tmp_path/"F001_M002.csv"; df.to_csv(f, index=False)

    args = (str(f), "bilateral_only", 1e-4, 0.0, ["ppron"], "off", 0.0, 0, "off", False, [])
    ctx  = {"rw_enabled": True, "rw_window_size":1, "rw_include_current":1, "rw_min_window_tokens":0}
    out  = process_file_rlsm(args, context=ctx)

    # 通常rLSM（瞬間率）は4ターン分のペアが出る前提（話者交代3回）
    assert out["pair_category_scores"], "通常rLSMのペアは存在する"

    # rwは ≤2語を除去 → 実質 '3','3' のみ → 話者交代1回
    assert out["rw_conv_row"]["n_pair_turns"] == 1
