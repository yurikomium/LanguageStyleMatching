# test_rw_short_utterance_filter.py
import pandas as pd
import math
from rlsm.workers import process_file_rlsm

def test_short_utterances_removed_only_for_rw(monkeypatch, tmp_path):
    import rlsm.workers as W
    # Mock that emits dummy counts for 1/2/3-word utterances
    def fake_rates(text):
        n = int(text)  # Use "1","2","3" as the total token count
        counts = {"ppron": n}  # Put everything into ppron (rate is always 100%)
        rates = {"ppron": 100.0 if n>0 else 0.0}
        return (rates, n, counts)
    monkeypatch.setattr(W, "_rates_for_text", fake_rates)

    df = pd.DataFrame([
        {"speaker":"female","text":"1"},  # ← ≤2 words will be filtered out for rw
        {"speaker":"male",  "text":"2"},
        {"speaker":"female","text":"3"},  # ← remains
        {"speaker":"male",  "text":"3"},
    ])
    f = tmp_path/"F001_M002.csv"; df.to_csv(f, index=False)

    args = (str(f), "bilateral_only", 1e-4, 0.0, ["ppron"], "off", 0.0, 0, "off", False, [])
    ctx  = {"rw_enabled": True, "rw_window_size":1, "rw_include_current":1, "rw_min_window_tokens":0}
    out  = process_file_rlsm(args, context=ctx)

    # Standard rLSM ("instantaneous rate") should produce pairs for 4 turns (3 speaker switches)
    assert out["pair_category_scores"], "Standard rLSM pairs should exist"

    # rw removes ≤2-word turns -> effectively only '3','3' remain -> 1 speaker switch
    assert out["rw_conv_row"]["n_pair_turns"] == 1
