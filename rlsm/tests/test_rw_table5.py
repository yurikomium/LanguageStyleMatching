# test_rw_pair_rows_table5.py
import math
from rlsm.workers import _compute_rw_window_turns, _build_rw_pair_rows

def test_rw_pair_rows_respect_table5_nan_when_prev_zero_curr_pos():
    cats = ["c"]
    # leader窓=0%, responder窓=10% になるように総語数で調整
    turns_raw = [
        {"speaker":"A","counts":{"c":0},"total":10,"turn_index":0},  # A
        {"speaker":"B","counts":{"c":1},"total":10,"turn_index":1},  # B
    ]
    rw_turns = _compute_rw_window_turns(turns_raw, cats, window_size=1, include_current=True)
    rows, _ = _build_rw_pair_rows(rw_turns, cats, {}, {"rw_window_size":1,"rw_include_current":1,"rw_min_window_tokens":0})
    r = rows[0]
    assert math.isnan(r["rw_rlsm"]), "前=0 & 今>0 は NaN（Table 5B）"
