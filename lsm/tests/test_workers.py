# exploration/Transcript/lsm/tests/test_lsm_workers.py
import pandas as pd
import numpy as np

from lsm.workers import (
    _window_index,
    split_conversation_by_time,
)

def test_window_index_basic():
    assert _window_index(0.0, 150) == 0
    assert _window_index(149.9, 150) == 0
    assert _window_index(150.0, 150) == 1
    assert _window_index(301.0, 150) == 2
    assert _window_index(float("nan"), 150) == 0

def test_split_conversation_by_time_2p5min():
    df = pd.DataFrame({
        "start":   [0.0, 10.0, 149.9, 150.0, 299.9, 300.0],
        "speaker": ["F","M","F","M","F","M"],
        "text":    ["a","b","c","d","e","f"],
    })
    # 2.5 minutes -> 150 seconds
    windows = split_conversation_by_time(df, window_size_minutes=2.5)

    # Expect: win0=[0,150), win1=[150,300), win2=[300,450)
    assert [w["window_index"] for w in windows] == [0, 1, 2]
    w0 = windows[0]["data"]; w1 = windows[1]["data"]; w2 = windows[2]["data"]
    assert list(w0["text"]) == ["a","b","c"]
    assert list(w1["text"]) == ["d","e"]
    assert list(w2["text"]) == ["f"]

def test_split_conversation_by_time_edge_and_empty():
    # Empty DataFrame
    assert split_conversation_by_time(pd.DataFrame(), 2.5) == []
    # Missing 'start' column
    assert split_conversation_by_time(pd.DataFrame({"text":["x"]}), 2.5) == []
    # Non-positive window size
    df = pd.DataFrame({"start":[0.0], "speaker":["F"], "text":["x"]})
    assert split_conversation_by_time(df, 0) == []
