# -*- coding: utf-8 -*-
"""
rLSM child process: Heavy initialization (spaCy, LIWC dictionary) and single file processing
- Calculate LIWC category "usage rate" for each turn (%)
- rLSM (paper-compliant core) for pair x category → individual x category → dyad x category → final
- Expand to CSV rows (conversation summary / pairs / individual category / dyad category)
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple
from exploration.Transcript.rlsm.unitize import unitize_transcript
from collections import deque
from typing import Deque
from exploration.Transcript.rlsm.rlsm_core import rlsm_per_category_from_rates, EPS as CORE_EPS

# Prevent parallel x BLAS runaway (child process insurance)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import spacy

# Add /workspace to path
import sys
sys.path.append("/workspace/")

# Reuse existing LIWC utilities
from exploration.Transcript.lsm.liwc_lsm import (  # type: ignore
    load_liwc_dic,
    build_compiled_patterns,
    count_liwc_categories,
    count_total_words,
)

# rLSM core (paper-compliant)
from exploration.Transcript.rlsm.rlsm_core import compute_rlsm_core  # type: ignore
from collections import defaultdict

# ---- Heavy resources shared within child process (process-wide global) ----
_WORKER_NLP = None
_WORKER_COMPILED = None
_WORKER_ALL_CATS: List[str] = []   # ← Hold main ∪ micro

def init_worker(dic_path: str, function_labels: List[str], micro_labels: List[str]):
    """
    Called once at each process startup. Load heavy resources here.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    global _WORKER_NLP, _WORKER_COMPILED, _WORKER_ALL_CATS

    # spaCy Japanese model (with fallback)
    try:
        _WORKER_NLP = spacy.load("ja_ginza", disable=["ner", "parser", "attribute_ruler"])
    except Exception:
        _WORKER_NLP = spacy.load("ja_ginza_electra", disable=["ner", "parser", "attribute_ruler"])

    # LIWC dictionary
    cat_map, word_map = load_liwc_dic(dic_path)

    # Available categories (robustly from both keys/values)
    available = set(cat_map) | set(cat_map.values())

    # Filter main categories
    main_ok = [c for c in function_labels if c in available]
    if not main_ok:
        raise RuntimeError("Specified main categories do not exist in dictionary.")

    # Micro categories (adopt if exist)
    micro_ok = [c for c in (micro_labels or []) if c in available]

    # Compile "main ∪ micro" union only once
    _WORKER_ALL_CATS = list(dict.fromkeys(main_ok + micro_ok))  # Deduplicate preserving order
    _WORKER_COMPILED = build_compiled_patterns(word_map, cat_map, set(_WORKER_ALL_CATS))

def _normalize_speaker(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("female", "f", "woman", "girl", "女性"):
        return "female"
    if s in ("male", "m", "man", "boy", "男性"):
        return "male"
    return s or "unknown"

def _extract_ids_from_filename(base: str) -> Tuple[Optional[int], Optional[int]]:
    f_m = re.search(r'F0*(\d+)', base, flags=re.I)
    m_m = re.search(r'M0*(\d+)', base, flags=re.I)
    female_id = int(f_m.group(1)) if f_m else None
    male_id = int(m_m.group(1)) if m_m else None
    return female_id, male_id


def _extract_round_from_filename(base: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract round information from filename.
    Return: (round_label, round_num)
      - round_label: "1st_round" | "2nd_round" | None
      - round_num  : 1 | 2 | None
    Supported examples: "...round1...", "...round2...", "...1st...", "...2nd...", "...first...", "...second..."
    """
    s = (base or "").lower()
    num: Optional[int] = None

    m = re.search(r"round[_\- ]*([12])\b", s)
    if m:
        try:
            num = int(m.group(1))
        except Exception:
            num = None
    else:
        if re.search(r"\b1st\b|\bfirst\b", s):
            num = 1
        elif re.search(r"\b2nd\b|\bsecond\b", s):
            num = 2

    if num == 1:
        return "1st_round", 1
    if num == 2:
        return "2nd_round", 2
    return None, None

def _rates_for_text(text: str) -> Tuple[Dict[str, float], int, Dict[str, int]]:
    """
    Return LIWC category usage rate (%) for one text. Keys are _WORKER_ALL_CATS.
    returns: (rates, total_words, counts)
    """
    if not text or not text.strip():
        empty_counts = {c: 0 for c in _WORKER_ALL_CATS}
        return {c: 0.0 for c in _WORKER_ALL_CATS}, 0, empty_counts
    doc = _WORKER_NLP(text)
    counts = count_liwc_categories(doc, _WORKER_COMPILED)
    total = count_total_words(doc)
    if total <= 0:
        empty_counts = {c: 0 for c in _WORKER_ALL_CATS}
        return {c: 0.0 for c in _WORKER_ALL_CATS}, 0, empty_counts
    rates = {c: (counts.get(c, 0) / float(total) * 100.0) for c in _WORKER_ALL_CATS}
    counts_dict = {c: counts.get(c, 0) for c in _WORKER_ALL_CATS}
    return rates, int(total), counts_dict

def _lsm(a: float, b: float, eps: float = 1e-9) -> float:
    den = a + b
    if den <= eps:
        return float("nan")
    return 1.0 - abs(a - b) / den

def _build_rw_pair_rows(
    turns_rw_meta: List[Dict[str, Any]],
    categories: List[str],
    base_meta: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    pair_index = 0
    min_tok = int(ctx.get("rw_min_window_tokens", 0))
    # eps は ctx → rlsm_core.EPS → 1e-4 の優先で解決
    eps = float(ctx.get("eps", ctx.get("rlsm_eps", CORE_EPS if 'CORE_EPS' in globals() else 1e-4)))
    zero_tol = float(ctx.get("zero_tol", 0.0))

    for i in range(len(turns_rw_meta) - 1):
        A, B = turns_rw_meta[i], turns_rw_meta[i + 1]
        if A["speaker"] == B["speaker"]:
            continue  # Not alternating
        pair_index += 1
        leader, responder = A, B
        for cat in categories:
            l_rate = float(leader["rates"][cat]); r_rate = float(responder["rates"][cat])
            leader_ok = (leader["win_total"] >= min_tok) if min_tok > 0 else True
            responder_ok = (responder["win_total"] >= min_tok) if min_tok > 0 else True
            if leader_ok and responder_ok:
                # Apply Table 5 rules to window rates (prev=0 & curr>0 excluded as NaN)
                _res = rlsm_per_category_from_rates(
                    prev_rates={cat: l_rate},
                    curr_rates={cat: r_rate},
                    categories=[cat],
                    eps=eps,
                    zero_tol=zero_tol,
                )
                val = _res[cat]
            else:
                val = float("nan")

            row = {
                **base_meta,  # female_id / male_id / file_path
                "pair_index": pair_index,
                "leader": leader["speaker"],
                "responder": responder["speaker"],
                "responder_gender": responder["speaker"],
                "category": cat,
                "rw_rlsm": val,
                "leader_window_size": int(leader["win_size"]),
                "responder_window_size": int(responder["win_size"]),
                "leader_window_total_words": int(leader["win_total"]),
                "responder_window_total_words": int(responder["win_total"]),
                "window_size_param": int(ctx["rw_window_size"]),
                "include_current": bool(ctx["rw_include_current"]),
                "leader_window_ok": bool(leader_ok),
                "responder_window_ok": bool(responder_ok),
            }
            rows.append(row)

    # Return total number of alternating turns (max value of pair_index)
    return rows, pair_index

# Replace entire existing _compute_rw_window_turns
from collections import deque
from typing import Deque

def _compute_rw_window_turns(
    turns_raw: List[Dict[str, Any]],
    categories: List[str],
    window_size: int,
    include_current: bool = True,
    eps: float = 1e-9,
) -> List[Dict[str, Any]]:
    """
    turns_raw: [{"speaker":"female"/"male","counts":{cat:int},"total":int,"turn_index":int}, ...]
    return   : [{"speaker":..., "rates":{cat:%}, "win_size":int, "win_total":int, "turn_index":int}, ...]
    """
    # 話者ラベルは動的に確保（'female'/'male'固定にしない）
    dq: Dict[str, Deque[Dict[str, Any]]] = {}
    sums: Dict[str, Dict[str, int]] = {}
    sums_total: Dict[str, int] = {}

    def ensure(spk: str):
        if spk not in dq:
            dq[spk] = deque()
            sums[spk] = {c: 0 for c in categories}
            sums_total[spk] = 0
    out_turns: List[Dict[str, Any]] = []

    def make_row(spk: str, t_idx: int):
        ensure(spk)
        tot = sums_total[spk]
        rates = {c: (100.0 * (sums[spk][c] / max(tot, eps))) if tot > 0 else 0.0 for c in categories}
        return {
            "speaker": spk,
            "rates": rates,
            "win_size": len(dq[spk]),
            "win_total": int(tot),
            "turn_index": t_idx,
        }

    for t in turns_raw:
        spk = t["speaker"]
        t_idx = int(t.get("turn_index", -1))
        ensure(spk)
        if not include_current:
            out_turns.append(make_row(spk, t_idx))  # 追加前の窓

        dq[spk].append(t)
        for c in categories:
            sums[spk][c] += int(t["counts"][c])
        sums_total[spk] += int(t["total"])
        while len(dq[spk]) > window_size:
            old = dq[spk].popleft()
            for c in categories:
                sums[spk][c] -= int(old["counts"][c])
            sums_total[spk] -= int(old["total"])

        if include_current:
            out_turns.append(make_row(spk, t_idx))  # 追加後の窓

    return out_turns

def _expand_pair_category_rows(base_meta: Dict[str, Any], pair_scores: List[Dict[str, Any]], prefix: str = "") -> List[Dict[str, Any]]:
    """
    Expand pair x category scores into rows
    """
    rows = []
    for rec in pair_scores:
        leader = rec["leader"]
        responder = rec["responder"]
        responder_gender = responder if responder in ("female", "male") else (
            "female" if responder == "FEMALE" else "male" if responder == "MALE" else responder
        )
        for c, v in rec["category_scores"].items():
            rows.append({
                **base_meta,
                "pair_index": int(rec["pair_index"]),
                "leader": leader,
                "responder": responder,
                "responder_gender": responder_gender,
                "category": c,
                f"{prefix}rlsm": (float(v) if (v is not None and not np.isnan(v)) else np.nan),
            })
    return rows

def _expand_individual_category_rows(base_meta: Dict[str, Any], indiv_cat_means: Dict[str, Dict[str, Any]], prefix: str = "") -> List[Dict[str, Any]]:
    """
    Expand individual x category scores into rows
    """
    rows = []
    for gender in ("female", "male"):
        cmap = indiv_cat_means.get(gender, {})
        for c, v in cmap.items():
            rows.append({
                **base_meta,
                "speaker_gender": gender,
                "category": c,
                f"{prefix}individual_rlsm": float(v) if (v is not None and not np.isnan(v)) else np.nan,
            })
    return rows

def _expand_dyad_category_rows(base_meta: Dict[str, Any], dyad_cat_means: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    """
    Expand dyad x category scores into rows
    """
    rows = []
    for c, v in dyad_cat_means.items():
        rows.append({
            **base_meta,
            "category": c,
            f"{prefix}dyad_rlsm": float(v) if (v is not None and not np.isnan(v)) else np.nan,
        })
    return rows

def _build_turns(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """
    Convert CSV -> rLSM core input (turns).
    returns: (turns, total_words_by_gender, turns_raw)
    """
    turns: List[Dict[str, Any]] = []
    turns_raw: List[Dict[str, Any]] = []
    tot_by_gender = {"female": 0, "male": 0}
    for i, (_, row) in enumerate(df.iterrows()):
        speaker = _normalize_speaker(str(row.get("speaker", "")))
        # Prioritize text_clean, but ignore NaN or blanks and fall back to text
        tc = row.get("text_clean", None)
        if pd.isna(tc) or str(tc).strip() == "":
            text = str(row.get("text", "") or "")
        else:
            text = str(tc)
        rates, tot, counts = _rates_for_text(text)
        if speaker in tot_by_gender:
            tot_by_gender[speaker] += tot
        turns.append({
            "speaker": speaker,
            "rates": rates,
        })
        turns_raw.append({
            "speaker": speaker,
            "counts": counts,
            "total": int(tot),
            "turn_index": int(i),
        })
    return turns, tot_by_gender, turns_raw

def process_file_rlsm(args: Tuple[str, str, float, float, List[str], str, float, int, str,
                                  bool, List[str]], context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Changes:
      Added (enable_micro: bool, micro_categories: List[str]) at the end
    """
    try:
        (file_path, na_policy, eps, zero_tol, categories,  # Main categories
         unitize_mode, merge_gap, strip_tags, bracket_mode,
         enable_micro, micro_categories) = args
        # Heavy initialization (spaCy/LIWC) assumed done in init_worker, but
        # _rates_for_text may be replaced in mocking or test environments,
        # so proceed without forced assertion here.
        
        # Set default values for context
        if context is None:
            context = {}
        context = {
            "rw_enabled": True,
            "rw_window_size": 3,
            "rw_include_current": 1,
            "enable_micro_ppron": False,
            **context  # Overwrite with provided context
        }

        # Load (check required columns)
        df = pd.read_csv(file_path)
        if df is None or df.empty or ("text" not in df.columns) or ("speaker" not in df.columns):
            return None

        base = os.path.basename(file_path)
        female_id, male_id = _extract_ids_from_filename(base)

        # Additional metadata (basename / dyad ID / round)
        file_basename = base
        dyad_id = (f"F{female_id:03d}_M{male_id:03d}"
                   if (female_id is not None and male_id is not None) else None)
        round_label, round_num = _extract_round_from_filename(base)
        extra_meta = {
            "file_basename": file_basename,
            "dyad_id": dyad_id,
            "round_label": round_label,
            "round_num": (int(round_num) if round_num is not None else None),
        }

        # ★ Execute preprocessing (unitize) here ★
        try:
            df_unit, unit_report = unitize_transcript(
                df,
                mode=unitize_mode,
                merge_gap=float(merge_gap),
                strip_tags=bool(int(strip_tags)) if isinstance(strip_tags, str) else bool(strip_tags),
                bracket_mode=bracket_mode,
            )
        except Exception as e:
            # Use original df if preprocessing fails (return log)
            df_unit, unit_report = df, {"error": str(e)}

        # Guarantee complete text column passthrough when unitization is off (ensure priority use of text_clean)
        if unitize_mode == "off":
            if ("text_clean" in df.columns) and ("text_clean" not in df_unit.columns):
                try:
                    # Assume same row order/count (off)
                    df_unit = df_unit.copy()
                    df_unit["text_clean"] = df["text_clean"]
                except Exception:
                    # Ignore on failure just in case (falls back to text)
                    pass

        # Build turns (hold all category rates)
        turns, tot_by_gender, turns_raw = _build_turns(df_unit)
        # rw.rLSM paper compliance: Exclude short utterances (≤2 words)
        turns_raw = [t for t in turns_raw if int(t.get("total", 0)) >= 3]

        # ===== Main rLSM =====
        res = compute_rlsm_core(
            turns=turns,
            categories=categories,    # Specify main categories only
            eps=eps,
            zero_tol=zero_tol,
            na_policy=na_policy,
        )

        # --- Added: Per-category pair case statistics (for understanding structural NaN) ---
        pair_case = {c: defaultdict(int) for c in categories}
        # Look only at speaker changes among adjacent pairs
        for i in range(len(turns) - 1):
            if turns[i]["speaker"] == turns[i+1]["speaker"]:
                continue
            prev_rates = turns[i]["rates"]; curr_rates = turns[i+1]["rates"]
            for c in categories:
                a = float(prev_rates.get(c, 0.0)); b = float(curr_rates.get(c, 0.0))
                a0 = (abs(a) <= float(zero_tol)); b0 = (abs(b) <= float(zero_tol))
                if a0 and b0:
                    pair_case[c]["both_zero"] += 1
                elif a0 and not b0:
                    pair_case[c]["leader_zero"] += 1   # prev=0, curr>0
                elif (not a0) and b0:
                    pair_case[c]["responder_zero"] += 1  # prev>0, curr=0
                else:
                    pair_case[c]["valid"] += 1           # prev>0 & curr>0

        # --- Conversation summary (including final, individual overall, category-wise dyad) ---
        dyad_cat = res["dyad_category_means"]  # {cat: val}
        indiv_cat = res["individual_category_means"]  # {"female":{cat:...}, "male":{...}} etc
        indiv_overall = res["individual_overall"]     # {"female": val, "male": val}

        # Speaker check, word count check (as existing)
        uniq_speakers = sorted(set(t["speaker"] for t in turns if t["speaker"]))
        if set(uniq_speakers) != {"female", "male"}:
            return {
                "error": (
                    f"Invalid speakers in file: {file_path}. "
                    f"Found speakers={uniq_speakers}. Expected exactly ['female','male']."
                ),
                "file_path": file_path,
            }
        if tot_by_gender.get("female", 0) <= 0 or tot_by_gender.get("male", 0) <= 0:
            return {
                "error": (
                    f"No tokens for one gender in file: {file_path}. "
                    f"female_total={tot_by_gender.get('female',0)}, male_total={tot_by_gender.get('male',0)}."
                ),
                "file_path": file_path,
            }

        # Conversation summary row
        conv_row: Dict[str, Any] = {
            "female_id": female_id,
            "male_id": male_id,
            "file_path": file_path,
            "female_total_words": int(tot_by_gender.get("female", 0)),
            "male_total_words": int(tot_by_gender.get("male", 0)),
            "dyad_final_rlsm": float(res["dyad_final"]),
            "n_pair_turns": int(len(res["pair_category_scores"])),
            "na_policy": na_policy,
        }
        conv_row.update(extra_meta)
        # Individual overall
        conv_row["female_overall_rlsm"] = float(indiv_overall.get("female", np.nan))
        conv_row["male_overall_rlsm"] = float(indiv_overall.get("male", np.nan))
        # Individual category
        for c in categories:
            conv_row[f"female_rlsm_{c}"] = float(indiv_cat.get("female", {}).get(c, np.nan))
            conv_row[f"male_rlsm_{c}"] = float(indiv_cat.get("male", {}).get(c, np.nan))
        # Dyad category
        for c, v in dyad_cat.items():
            conv_row[f"dyad_rlsm_{c}"] = float(v) if (v is not None and not np.isnan(v)) else np.nan

        # Added: Embed coverage statistics in conversation summary (useful for auditing & learning features too)
        for c in categories:
            pcs = pair_case[c]
            total_pairs = int(sum(pcs.values()))
            conv_row[f"rlsm_{c}_pairs_total"] = total_pairs
            conv_row[f"rlsm_{c}_pairs_valid"] = int(pcs.get("valid", 0))
            conv_row[f"rlsm_{c}_pairs_both_zero"] = int(pcs.get("both_zero", 0))
            conv_row[f"rlsm_{c}_pairs_leader_zero"] = int(pcs.get("leader_zero", 0))
            conv_row[f"rlsm_{c}_pairs_responder_zero"] = int(pcs.get("responder_zero", 0))
            # Coverage ratio (proportion where both > 0)
            conv_row[f"rlsm_{c}_coverage_ratio"] = (
                (pcs.get("valid", 0) / total_pairs) if total_pairs > 0 else float("nan")
            )

        # --- Added: Imputed version (fill structural unobserved or NaN with 0.5, add flag) ---
        import numpy as _np
        IMP = 0.5
        # Individual x category
        for c in categories:
            total = int(conv_row.get(f"rlsm_{c}_pairs_total", 0) or 0)
            valid = int(conv_row.get(f"rlsm_{c}_pairs_valid", 0) or 0)
            f_val = float(indiv_cat.get("female", {}).get(c, _np.nan))
            m_val = float(indiv_cat.get("male", {}).get(c, _np.nan))
            d_val = float(dyad_cat.get(c, _np.nan))

            # Target for filling: "structurally unobserved (valid==0 or total==0)" or "value is NaN"
            f_need = (valid == 0 or total == 0 or _np.isnan(f_val))
            m_need = (valid == 0 or total == 0 or _np.isnan(m_val))
            d_need = (valid == 0 or total == 0 or _np.isnan(d_val))

            conv_row[f"female_rlsm_{c}_imputed"] = (IMP if f_need else f_val)
            conv_row[f"male_rlsm_{c}_imputed"]   = (IMP if m_need else m_val)
            conv_row[f"dyad_rlsm_{c}_imputed"]   = (IMP if d_need else d_val)

            # Flag (effective for improving learning stability)
            conv_row[f"female_rlsm_{c}_was_imputed"] = int(f_need)
            conv_row[f"male_rlsm_{c}_was_imputed"]   = int(m_need)
            conv_row[f"dyad_rlsm_{c}_was_imputed"]   = int(d_need)

        # Fill individual overall and final dyad similarly + flag
        def _imp_and_flag(name_src, name_dst, name_flag):
            val = conv_row.get(name_src, _np.nan)
            need = (val is None) or _np.isnan(float(val))
            conv_row[name_dst]  = (IMP if need else float(val))
            conv_row[name_flag] = int(need)
        _imp_and_flag("female_overall_rlsm", "female_overall_rlsm_imputed", "female_overall_rlsm_was_imputed")
        _imp_and_flag("male_overall_rlsm",   "male_overall_rlsm_imputed",   "male_overall_rlsm_was_imputed")
        _imp_and_flag("dyad_final_rlsm",     "dyad_final_rlsm_imputed",     "dyad_final_rlsm_was_imputed")

        # --- Pair x category rows (attributed to responder) ---
        pair_rows: List[Dict[str, Any]] = []
        for rec in res["pair_category_scores"]:
            leader = rec["leader"]
            responder = rec["responder"]
            responder_gender = responder if responder in ("female", "male") else ("female" if responder == "FEMALE" else "male" if responder == "MALE" else responder)
            for c, v in rec["category_scores"].items():
                pair_rows.append({
                    "female_id": female_id,
                    "male_id": male_id,
                    "file_path": file_path,
                    "pair_index": int(rec["pair_index"]),
                    "leader": leader,
                    "responder": responder,
                    "responder_gender": responder_gender,
                    "category": c,
                    "rlsm": (float(v) if (v is not None and not np.isnan(v)) else np.nan),
                })

        # --- Individual x category rows ---
        individual_category_rows: List[Dict[str, Any]] = []
        for gender in ("female", "male"):
            cmap = indiv_cat.get(gender, {})
            for c in categories:
                individual_category_rows.append({
                    "female_id": female_id,
                    "male_id": male_id,
                    "file_path": file_path,
                    "speaker_gender": gender,
                    "category": c,
                    "individual_rlsm": float(cmap.get(c, np.nan)),
                })

        # --- Dyad x category rows ---
        dyad_category_rows: List[Dict[str, Any]] = []
        for c in categories:
            dyad_category_rows.append({
                "female_id": female_id,
                "male_id": male_id,
                "file_path": file_path,
                "category": c,
                "dyad_rlsm": float(dyad_cat.get(c, np.nan)),
            })

        # ===== Calculate micro-ppron optionally =====
        micro_pair_rows: List[Dict[str, Any]] = []
        micro_individual_rows: List[Dict[str, Any]] = []
        micro_dyad_rows: List[Dict[str, Any]] = []

        if enable_micro and micro_categories:
            res_micro = compute_rlsm_core(
                turns=turns,
                categories=micro_categories,  # Micro categories only
                eps=eps,
                zero_tol=zero_tol,
                na_policy=na_policy,
            )

            # Pair x micro category (attributed to responder)
            for rec in res_micro["pair_category_scores"]:
                leader = rec["leader"]
                responder = rec["responder"]
                responder_gender = responder if responder in ("female", "male") else \
                    ("female" if responder == "FEMALE" else "male" if responder == "MALE" else responder)
                for c, v in rec["category_scores"].items():
                    micro_pair_rows.append({
                        "female_id": female_id,
                        "male_id": male_id,
                        "file_path": file_path,
                        "pair_index": int(rec["pair_index"]),
                        "leader": leader,
                        "responder": responder,
                        "responder_gender": responder_gender,
                        "micro_cat": c,
                        "micro_rlsm": (float(v) if (v is not None and not np.isnan(v)) else np.nan),
                    })

            # Individual x micro category
            indiv_cat_micro = res_micro["individual_category_means"]
            for gender in ("female", "male"):
                cmap = indiv_cat_micro.get(gender, {})
                for c in micro_categories:
                    micro_individual_rows.append({
                        "female_id": female_id,
                        "male_id": male_id,
                        "file_path": file_path,
                        "speaker_gender": gender,
                        "micro_cat": c,
                        "individual_micro_rlsm": float(cmap.get(c, np.nan)),
                    })

            # Dyad x micro category
            dyad_micro = res_micro["dyad_category_means"]
            for c in micro_categories:
                micro_dyad_rows.append({
                    "female_id": female_id,
                    "male_id": male_id,
                    "file_path": file_path,
                    "micro_cat": c,
                    "dyad_micro_rlsm": float(dyad_micro.get(c, np.nan)),
                })

        # Add additional metadata to each row (unify join keys for learning)
        for _row in pair_rows:
            _row.update(extra_meta)
        for _row in individual_category_rows:
            _row.update(extra_meta)
        for _row in dyad_category_rows:
            _row.update(extra_meta)
        for _row in micro_pair_rows:
            _row.update(extra_meta)
        for _row in micro_individual_rows:
            _row.update(extra_meta)
        for _row in micro_dyad_rows:
            _row.update(extra_meta)

        # ===== From here rw.rLSM =====
        base_meta = {
            "female_id": female_id,
            "male_id": male_id,
            "file_path": file_path,
            "file_basename": file_basename,
            "dyad_id": dyad_id,
            "round_label": round_label,
            "round_num": (int(round_num) if round_num is not None else None),
        }

        out = {
            "conversation_summary": conv_row,
            "pair_category_scores": res["pair_category_scores"],
            "pairs_rows": pair_rows,
            "individual_category_rows": individual_category_rows,
            "dyad_category_rows": dyad_category_rows,
            "unitize_report": unit_report,
            "micro_pair_rows": micro_pair_rows,
            "micro_individual_rows": micro_individual_rows,
            "micro_dyad_rows": micro_dyad_rows,
        }

        if context.get("rw_enabled", True):
            # 1) Rolling window turns (rates + window metadata)
            rw_turns = _compute_rw_window_turns(
                turns_raw=turns_raw,
                categories=categories,
                window_size=int(context.get("rw_window_size", 3)),
                include_current=bool(context.get("rw_include_current", 1)),
                eps=eps,
            )

            # 2) Pair table (with spec column names/metadata) and total alternations
            rw_pair_rows, total_pair_turns = _build_rw_pair_rows(
                turns_rw_meta=rw_turns,
                categories=categories,
                base_meta=base_meta,
                ctx=context,
            )

            # Number of non-NaN pairs (those meeting min_window_tokens condition)
            valid_pair_indices = set()
            for r in rw_pair_rows:
                val = r.get("rw_rlsm")
                if val == val:  # not NaN
                    valid_pair_indices.add(int(r["pair_index"]))
            n_pair_turns = len(valid_pair_indices)

            # 3) Calculate individual/dyad averages for conversation summary with existing core
            rw_res = compute_rlsm_core(
                turns=rw_turns,
                categories=categories,
                eps=eps,
                zero_tol=zero_tol,
                na_policy=na_policy,
            )

            # 4) Conversation summary (spec column names)
            rw_indiv_overall = rw_res["individual_overall"]
            rw_indiv_cat     = rw_res["individual_category_means"]  # {"female":{cat:..},"male":{..}}
            rw_dyad_cat      = rw_res["dyad_category_means"]        # {cat:..}

            rw_conv_row = {
                **base_meta,
                "female_total_words": int(tot_by_gender.get("female", 0)),
                "male_total_words": int(tot_by_gender.get("male", 0)),
                "rw_dyad_final_rlsm": float(rw_res["dyad_final"]),
                "n_pair_turns": int(n_pair_turns),
                "na_policy": na_policy,
                "window_size_param": int(context.get("rw_window_size", 3)),
                "include_current": bool(context.get("rw_include_current", 1)),
                "min_window_tokens": int(context.get("rw_min_window_tokens", 0)),
                "rw_female_overall_rlsm": float(rw_indiv_overall.get("female", np.nan)),
                "rw_male_overall_rlsm": float(rw_indiv_overall.get("male", np.nan)),
                # Individual x category
                **{f"rw_female_rlsm_{k}": float(v) if v is not None else np.nan
                   for k, v in rw_indiv_cat.get("female", {}).items()},
                **{f"rw_male_rlsm_{k}": float(v) if v is not None else np.nan
                   for k, v in rw_indiv_cat.get("male", {}).items()},
                # Dyad x category
                **{f"rw_dyad_rlsm_{k}": float(v) if v is not None else np.nan
                   for k, v in rw_dyad_cat.items()},
            }

            # 5) Individual x category (spec column names)
            rw_individual_rows: List[Dict[str, Any]] = []
            for spk in ("female", "male"):
                for c in categories:
                    rw_individual_rows.append({
                        **base_meta,
                        "speaker_gender": spk,
                        "category": c,
                        "rw_individual_rlsm": float(rw_indiv_cat.get(spk, {}).get(c, np.nan)),
                        "window_size_param": int(context.get("rw_window_size", 3)),
                        "include_current": bool(context.get("rw_include_current", 1)),
                    })

            # 6) Dyad x category (spec column names)
            rw_dyad_rows: List[Dict[str, Any]] = []
            for c in categories:
                rw_dyad_rows.append({
                    **base_meta,
                    "category": c,
                    "rw_dyad_rlsm": float(rw_dyad_cat.get(c, np.nan)),
                    "window_size_param": int(context.get("rw_window_size", 3)),
                    "na_policy": na_policy,
                    "include_current": bool(context.get("rw_include_current", 1)),
                })

            # 7) Add to return value (run_rlsm.py will pick)
            out["rw_conv_row"] = rw_conv_row
            out["rw_pair_rows"] = rw_pair_rows
            out["rw_individual_rows"] = rw_individual_rows
            out["rw_dyad_rows"] = rw_dyad_rows
        
        return out

    except Exception:
        # Return None on failure (parent skips silently)
        return None
