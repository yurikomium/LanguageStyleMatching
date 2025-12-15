# exploration/Transcript/rlsm/unitize.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import re

# ===== Default dictionary for bracket processing =====
# Words included here (ignoring case/full-width/half-width/leading-trailing spaces) are considered "description tags" and deleted entirely
DROP_TAG_WORDS_DEFAULT = {
    "笑", "拍手", "音楽", "bgm", "se", "効果音", "咳", "咳払い",
    "ため息", "ノイズ", "雑音", "聞き取れず", "不明", "沈黙", "無音",
    "laugh", "applause", "music", "noise", "cough", "sigh", "inaudible", "silence"
}

_BRACKET_PATTERNS = [
    (re.compile(r"\[(.*?)\]"), "[", "]"),     # Half-width
    (re.compile(r"［(.*?)］"), "［", "］"),   # Full-width
    (re.compile(r"【(.*?)】"), "【", "】"),   # Alternative notation
]

_STRAY_BRACKETS = re.compile(r"[\[\]［］【】]")

_WS = re.compile(r"\s+")

def _normalize_text(s: str) -> str:
    return s.lower().strip()

def _is_drop_tag(content: str, drop_words: set) -> bool:
    # Determine if description tag (alphanumeric/kana/full-width/half-width handled roughly by lower + strip)
    return _normalize_text(content) in drop_words

def _strip_brackets(
    text: str,
    *,
    mode: str = "unwrap",     # unwrap | drop_all | off
    drop_words: Optional[set] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    unwrap: By default keep contents and remove [] only. But if matches drop_words, delete with contents
    drop_all: Always delete [] with contents
    off: Do nothing
    """
    stats = {"unwrap_kept": 0, "drop_deleted": 0}
    if not text or mode == "off":
        return text, stats

    drop_words = drop_words or DROP_TAG_WORDS_DEFAULT
    out = text

    for pat, lch, rch in _BRACKET_PATTERNS:
        def repl(m):
            inner = m.group(1)
            if mode == "drop_all":
                stats["drop_deleted"] += 1
                return ""
            # unwrap:
            if _is_drop_tag(inner, drop_words):
                stats["drop_deleted"] += 1
                return ""
            stats["unwrap_kept"] += 1
            return inner

        out = pat.sub(repl, out)

    # Remove stray brackets
    out = _STRAY_BRACKETS.sub("", out)
    # Normalize whitespace
    out = _WS.sub(" ", out).strip()
    return out, stats

def _normalize_speaker(spk: str) -> str:
    s = (spk or "").strip().lower()
    if s in {"f", "female", "woman", "girl"}:
        return "female"
    if s in {"m", "male", "man", "boy"}:
        return "male"
    # Assume already female/male. Return others as-is (validated at upper level)
    return s

@dataclass
class UnitizeConfig:
    mode: str = "approx"          # off | approx | strict
    merge_gap: float = 0.0        # Allowable gap for merging consecutive same-speaker turns (seconds)
    strip_tags: bool = True       # Enable [] processing
    bracket_mode: str = "unwrap"  # unwrap | drop_all | off
    drop_words: Optional[set] = None

def unitize_transcript(
    df: pd.DataFrame,
    *,
    mode: str = "approx",
    merge_gap: float = 0.0,
    strip_tags: bool = True,
    bracket_mode: str = "unwrap",
    drop_words: Optional[set] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Input: Raw CSV (columns: start, end, text, speaker) ※ start/end assumed in seconds (float)
    Output: Non-overlapping & alternating (same-speaker merged as much as possible) talk-turn list and report
    """
    report = {"merged": 0, "split": 0, "approx_splits": 0, "dropped_zero": 0,
              "unwrap_kept": 0, "drop_deleted": 0}

    required = {"text", "speaker"}
    if not required.issubset(df.columns):
        # Pass through if required columns missing
        return df.copy(), report

    has_time = {"start", "end"}.issubset(df.columns)

    df = df.copy()

    # --- Normalize speaker ---
    df["speaker"] = df["speaker"].map(_normalize_speaker)

    # --- Bracket processing (stored in text_clean column) ---
    if strip_tags:
        cleaned = []
        for t in df["text"].astype(str).tolist():
            out, st = _strip_brackets(t, mode=bracket_mode,
                                      drop_words=(drop_words or DROP_TAG_WORDS_DEFAULT))
            cleaned.append(out)
            report["unwrap_kept"] += st["unwrap_kept"]
            report["drop_deleted"] += st["drop_deleted"]
        df["text_clean"] = cleaned
    else:
        df["text_clean"] = df["text"].astype(str)

    # If no time columns, cannot unitize further → return as is
    if not has_time or mode == "off":
        return df, report

    # --- Basic sorting ---
    df = df.sort_values(["start", "end"], kind="mergesort").reset_index(drop=True)

    # --- Merge consecutive same-speaker (only when no other speaker in between) ---
    merged_rows: List[Dict] = []
    def _append_row(acc, row):
        if row["end"] <= row["start"]:
            report["dropped_zero"] += 1
            return
        acc.append(row)

    for _, row in df.iterrows():
        row = dict(row)
        if not merged_rows:
            _append_row(merged_rows, row); continue
        last = merged_rows[-1]
        if (last["speaker"] == row["speaker"]) and (row["start"] - last["end"] <= merge_gap) and (row["start"] >= last["end"]):
            # Adjacent or small gap → merge
            last["end"] = max(last["end"], row["end"])
            # Concatenate text (clean side)
            last["text"] = f"{last['text']} {row['text']}".strip()
            last["text_clean"] = f"{last['text_clean']} {row['text_clean']}".strip()
            report["merged"] += 1
        else:
            _append_row(merged_rows, row)

    segs = merged_rows

    # --- Resolve different-speaker overlaps ---
    # Policy: Later arrival (later start) takes overlap region. Earlier one cut to [start, next.start).
    # Further, if earlier one extends beyond later's end, create new segment [next.end, prev.end) as "same-speaker second half".
    out: List[Dict] = []
    i = 0
    skip_append_cur_once = False   # If nxt already output in previous loop, suppress append of cur(=nxt) once
    while i < len(segs):
        cur = dict(segs[i])
        if i == len(segs) - 1:
            # Final
            if cur["end"] > cur["start"]:
                out.append(cur)
            i += 1
            break
        nxt = dict(segs[i + 1])

        if nxt["start"] >= cur["end"]:
            # No overlap
            if cur["end"] > cur["start"]:
                if not skip_append_cur_once:       # If nxt already output in previous overlap, skip cur append this time
                    out.append(cur)
                else:
                    skip_append_cur_once = False   # Release after skipping once
            i += 1
            continue

        # Overlap starts here
        if cur["speaker"] == nxt["speaker"]:
            # For same-speaker overlap, simply merge (absorb into longer one)
            # However, don't distribute text by time ratio, just concatenate later arrival's text at end (rare in practice)
            cur["end"] = max(cur["end"], nxt["end"])
            cur["text"] = f"{cur['text']} {nxt['text']}".strip()
            cur["text_clean"] = f"{cur['text_clean']} {nxt['text_clean']}".strip()
            report["merged"] += 1
            # Remove segs[i+1] to skip nxt and continue with cur as segs[i]
            segs.pop(i + 1)
            continue

        # Different-speaker overlap → later arrival (nxt) takes overlap region
        # cur_pre = [cur.start, nxt.start]
        cur_pre = dict(cur)
        cur_pre["end"] = max(cur["start"], min(cur["end"], nxt["start"]))

        # cur_post = [nxt.end, cur.end] (if exists)
        cur_post = None
        if nxt["end"] < cur["end"]:
            cur_post = dict(cur)
            cur_post["start"] = max(nxt["end"], cur["start"])
            # String split (approximate): distribute by time ratio
            # pre ratio, post ratio
            total = max(cur["end"] - cur["start"], 1e-9)
            pre_ratio = max(cur_pre["end"] - cur_pre["start"], 0.0) / total
            post_ratio = max(cur["end"] - cur_post["start"], 0.0) / total

            def _split_by_ratio(s: str, pre_r: float) -> Tuple[str, str]:
                # Japanese has few spaces, so approximate with character-based split without tokenization
                if not s:
                    return "", ""
                n = len(s)
                k = int(round(n * pre_r))
                return s[:k].strip(), s[k:].strip()

            t_pre, t_post = _split_by_ratio(cur["text"], pre_ratio)
            tc_pre, tc_post = _split_by_ratio(cur["text_clean"], pre_ratio)
            # Replace cur_pre text
            cur_pre["text"], cur_pre["text_clean"] = t_pre, tc_pre
            # Replace cur_post text
            cur_post["text"], cur_post["text_clean"] = t_post, tc_post
            report["approx_splits"] += 1

        else:
            # If nxt completely covers cur, no cur_post
            total = max(cur["end"] - cur["start"], 1e-9)
            pre_ratio = max(cur_pre["end"] - cur_pre["start"], 0.0) / total

            def _split_by_ratio(s: str, pre_r: float) -> Tuple[str, str]:
                if not s:
                    return "", ""
                n = len(s)
                k = int(round(n * pre_r))
                return s[:k].strip(), s[k:].strip()

            t_pre, _ = _split_by_ratio(cur["text"], pre_ratio)
            tc_pre, _ = _split_by_ratio(cur["text_clean"], pre_ratio)
            cur_pre["text"], cur_pre["text_clean"] = t_pre, tc_pre
            report["approx_splits"] += 1

        # Drop 0-length pre
        if cur_pre["end"] > cur_pre["start"]:
            out.append(cur_pre)
        else:
            report["dropped_zero"] += 1

        # Keep nxt as-is (later arrival takes overlap region)
        out.append(nxt)
        skip_append_cur_once = True   # Don't duplicate append cur(=nxt just output) in next iteration

        # If post exists, insert before segs[i+1] to become next comparison target
        if cur_post and (cur_post["end"] > cur_post["start"]):
            segs.insert(i + 2, cur_post)
        report["split"] += 1

        # Here increment i by +1 (to compare nxt with next element)
        i += 1

    # Rescue last item if not in out (non-overlapping single end case)
    if out and out[-1] is not segs[-1]:
        last = segs[-1]
        if last["end"] > last["start"]:
            if not out or (out[-1]["start"] != last["start"] or out[-1]["end"] != last["end"]):
                out.append(last)

    # --- Finishing: Re-merge if same-speaker adjacent (strengthen alternation) ---
    final_rows: List[Dict] = []
    for row in out:
        if not final_rows:
            final_rows.append(row)
            continue
        last = final_rows[-1]
        # Merge perfectly connected or small gap (same-speaker only)
        if (last.get("speaker") == row.get("speaker")
            and "start" in row and "end" in row and "end" in last
            and row["start"] >= last["end"]
            and (row["start"] - last["end"] <= merge_gap)):
            last["end"] = max(last["end"], row["end"])
            last["text"] = f"{last.get('text','')} {row.get('text','')}".strip()
            last["text_clean"] = f"{last.get('text_clean','')} {row.get('text_clean','')}".strip()
            report["merged"] += 1
        else:
            final_rows.append(row)

    # ---- From here, guarantee invariants of returned DF "at implementation side" ----
    out_df = pd.DataFrame(final_rows)

    # If text_clean missing/NaN, fill from text
    if "text_clean" not in out_df.columns:
        out_df["text_clean"] = out_df.get("text", "").astype(str)
    else:
        mask = out_df["text_clean"].isna() | (out_df["text_clean"].astype(str).str.len() == 0)
        if "text" in out_df.columns:
            out_df.loc[mask, "text_clean"] = out_df.loc[mask, "text"].astype(str)

    # Remove column duplicates (especially eliminate duplicate text_clean generation)
    out_df = out_df.loc[:, ~out_df.columns.duplicated()]

    # Normalize column order (required columns first)
    order = [c for c in ["start", "end", "speaker", "text", "text_clean"] if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in order]
    out_df = out_df[order + rest]

    # Normalize types: start/end to float (prevent string sort accidents)
    for c in ("start", "end"):
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    # Non-overlapping, stable sort, index formatting
    out_df = out_df.dropna(subset=["start", "end"])
    out_df = out_df[out_df["end"] > out_df["start"]]                 # Exclude zero/negative length
    out_df = out_df.sort_values(["start", "end"], kind="mergesort")  # Stable sort
    out_df = out_df.reset_index(drop=True)

    # Up to here:
    # - Column names are unique
    # - start/end are float
    # - Rows sorted ascending by start,end, zero-length removed, non-overlapping condition (end[i] <= start[i+1]) satisfied
    return out_df, report
