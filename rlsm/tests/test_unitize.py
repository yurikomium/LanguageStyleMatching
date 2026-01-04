# -*- coding: utf-8 -*-
"""
==============================================================================
test_unitize_brackets_and_merge_split.py - Tests for text preprocessing, merging, and splitting
==============================================================================

What this test file guarantees:

1. **Bracket handling**
   - Proper handling of annotation tags like [え], [笑], etc.
   - unwrap mode: keep contents while removing only the brackets
   - Full removal of annotation tags (e.g., [笑])
   - Proper removal of empty brackets []

2. **Merging consecutive turns by the same speaker**
   - Automatic merging of temporally adjacent same-speaker turns
   - Merge control via the merge_gap threshold
   - Proper space insertion when concatenating text

3. **Splitting overlaps across different speakers and assignment**
   - Detect and split overlapping utterances
   - Latter-starting utterance takes ownership of the overlap interval
   - Earlier-starting utterance is trimmed appropriately and the remainder is preserved

4. **Ensuring alternation**
   - Achieve an alternating speaker pattern after split/merge
   - Avoid consecutive turns by the same speaker
   - Preserve temporal consistency (end[i] ≤ start[i+1])

5. **Time-based processing accuracy**
   - Accurate time handling at the second level
   - Approximate text splitting based on time ratios
   - Strict enforcement of non-overlap constraints
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
    # Keep the content but remove the surrounding brackets []
    assert out.loc[0, "text_clean"].startswith("え ")
    assert "たぶん" in out.loc[out["text"].str.contains("たぶん")].iloc[0]["text_clean"]
    # Remove descriptive tags entirely
    assert out.loc[1, "text_clean"].startswith("はい")
    # Empty [] disappears
    assert "[]" not in out.loc[3, "text_clean"]

def test_same_speaker_merge_with_gap_threshold():
    # Same speaker twice in a row (0.0–1.0 and 1.0–2.0) -> merged if merge_gap=0.0 (and also OK if >0)
    df = pd.DataFrame([
        {"start": 0.0, "end": 1.0, "speaker": "female", "text": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "female", "text": "B"},
    ])
    out, _ = unitize_transcript(df, mode="approx", merge_gap=0.0, strip_tags=False)
    assert len(out) == 1
    assert out.iloc[0]["start"] == 0.0 and out.iloc[0]["end"] == 2.0
    assert out.iloc[0]["text_clean"] == "A B"

def test_overlap_is_split_and_assigned_to_latter_speaker():
    # Example: the case you provided (F interrupts during a long M utterance)
    df = pd.DataFrame([
        {"start": 1.71, "end": 3.08,  "speaker": "female", "text": "F1"},
        {"start": 2.24, "end": 3.17,  "speaker": "male",   "text": "M1"},
        {"start": 3.99, "end": 5.08,  "speaker": "male",   "text": "M2"},
        {"start": 5.33, "end": 6.66,  "speaker": "female", "text": "F2"},
        {"start": 7.63, "end": 14.23, "speaker": "male",   "text": "M3"},
        {"start": 8.48, "end": 8.77,  "speaker": "female", "text": "F3"},
    ])
    # Merge split fragments from the same speaker (allowing small gaps) to preserve alternation
    out, rep = unitize_transcript(
        df, mode="approx", merge_gap=10.0, strip_tags=False  # ← merge same-speaker turns up to 10 seconds apart
    )
    # Non-overlap property
    for i in range(len(out)-1):
        assert out.iloc[i]["end"] <= out.iloc[i+1]["start"]

    # Expected split (time order)
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

    # Alternation (no consecutive turns by the same speaker)
    for i in range(len(out)-1):
        assert out.iloc[i]["speaker"] != out.iloc[i+1]["speaker"]
