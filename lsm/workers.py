# -*- coding: utf-8 -*-
"""
Heavy initialization for child processes (spaCy, LIWC dictionary), and single file processing.
Intended to be imported from parent process (run_lsm.py).

◎ Roles
- Initialize each child process (load spaCy Japanese model and LIWC dictionary, preprocess patterns)
- Calculate LSM per file (read DataFrame → combine text by speaker → morphological analysis → category counting → LSM calculation → dictify)
"""

import os
import re
from typing import Dict, Any, List, Optional
from collections import Counter

# Prevent parallel x BLAS runaway (child process insurance)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import spacy

# Use utilities under /workspace
import sys
sys.path.append("/workspace/")
from utils import DataAnalyzer
try:
    from .liwc_lsm import load_liwc_dic, build_compiled_patterns, count_liwc_categories, count_total_words, compute_lsm
except ImportError:
    from liwc_lsm import load_liwc_dic, build_compiled_patterns, count_liwc_categories, count_total_words, compute_lsm

# ---- Heavy resources shared within child process (process-wide global) ----
_WORKER_NLP = None
_WORKER_COMPILED = None
_WORKER_TARGET_CATS: List[str] = []
_WORKER_DIC_BASENAME = None
_WORKER_SPACY_MODEL = "ja_ginza"


def split_conversation_by_time(df, window_size_minutes: float):
    """
    Helper to split utterances by half-open time windows.
    - Input df should have at least columns: ["start", "end", "speaker", "text"] (end can be unused)
    - Window width in minutes. e.g. 2.5 -> 150 seconds
    - Rule: Assign to [k*W, (k+1)*W) based on start seconds (half-open interval).
    Return: List of dicts for each window. {"window_index", "start", "end", "data"(df for that window)}
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return []
    if "start" not in df.columns:
        return []

    window_sec = int(round(float(window_size_minutes) * 60))
    if window_sec <= 0:
        return []

    # Copy only needed columns to avoid side effects
    use_cols = [c for c in ["start", "speaker", "text", "end"] if c in df.columns]
    sdf = df[use_cols].copy()
    # Convert start to numeric
    sdf["start"] = sdf["start"].astype(float)

    # Assign to half-open interval [kW, (k+1)W): floor(start / W)
    sdf["_win"] = (sdf["start"] // window_sec).astype(int)

    results = []
    if sdf["_win"].empty:
        return results
    for w in sorted(sdf["_win"].unique()):
        part = sdf[sdf["_win"] == w].drop(columns=["_win"]) if "_win" in sdf.columns else sdf
        results.append({
            "window_index": int(w),
            "start": int(w * window_sec),
            "end": int((w + 1) * window_sec),
            "data": part.reset_index(drop=True),
        })
    return results


def compute_temporal_lsm_for_window(window_rec, nlp, compiled_patterns):
    """
    Compute and return LSM from female/male text for one time window record.
    Minimum fields in input window_rec:
      - female_id, male_id, file_path, window_index, start_time, end_time
      - female_text, male_text
    Return dict:
      { female_id, male_id, file_path, window_index, start_time, end_time,
        female_word_count, male_word_count,
        lsm_scores: {cat: score, ...}, lsm_mean }
    """
    try:
        female_text = str(window_rec.get("female_text", "") or "")
        male_text = str(window_rec.get("male_text", "") or "")

        doc_f = nlp(female_text)
        doc_m = nlp(male_text)

        cnt_f = count_liwc_categories(doc_f, compiled_patterns)
        cnt_m = count_liwc_categories(doc_m, compiled_patterns)
        tot_f = int(count_total_words(doc_f))
        tot_m = int(count_total_words(doc_m))

        # Extract target categories from compiled_patterns
        target_categories = sorted({lab for _, labs in compiled_patterns for lab in labs})
        try:
            lsm_scores = compute_lsm(cnt_f, cnt_m, target_categories, tot_f, tot_m, rounding=None)
        except TypeError:
            lsm_scores = compute_lsm(cnt_f, cnt_m, target_categories, tot_f, tot_m)

        vals = list(lsm_scores.values())
        lsm_mean = float("nan") if not vals else float(np.mean(vals))

        return {
            "female_id": window_rec.get("female_id"),
            "male_id": window_rec.get("male_id"),
            "file_path": window_rec.get("file_path"),
            "window_index": int(window_rec.get("window_index", 0)),
            "start_time": window_rec.get("start_time"),
            "end_time": window_rec.get("end_time"),
            "female_word_count": tot_f,
            "male_word_count": tot_m,
            "lsm_scores": lsm_scores,
            "lsm_mean": lsm_mean,
        }
    except Exception:
        return None

def init_worker(dic_path: str, function_labels: List[str]):
    """
    Called once at each process startup. Load heavy resources here.
    """
    # Fix thread count here as well just in case
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    global _WORKER_NLP, _WORKER_COMPILED, _WORKER_TARGET_CATS

    # spaCy Japanese model (for POS and lemma)
    _WORKER_NLP = spacy.load("ja_ginza", disable=["ner", "parser", "attribute_ruler"])
    global _WORKER_DIC_BASENAME
    _WORKER_DIC_BASENAME = os.path.basename(dic_path)

    # LIWC dictionary
    cat_map, word_map = load_liwc_dic(dic_path)

    # Categories to use (maintain specified order)
    available = set(cat_map.values())
    _WORKER_TARGET_CATS = [c for c in function_labels if c in available]
    if not _WORKER_TARGET_CATS:
        raise RuntimeError("Specified function word categories do not exist in dictionary. Please check category names and dictionary.")

    # Pre-compile patterns
    _WORKER_COMPILED = build_compiled_patterns(word_map, cat_map, set(_WORKER_TARGET_CATS))


def process_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Calculate LSM from a single file and return as dict (exceptions suppressed to None).
    """
    try:
        assert _WORKER_NLP is not None and _WORKER_COMPILED is not None

        analyzer = DataAnalyzer()
        df = analyzer.get_single_conversation(file_path)
        if df is None or df.empty or ("text" not in df.columns) or ("speaker" not in df.columns):
            return None

        base = os.path.basename(file_path)
        f_m = re.search(r'F0*(\d+)', base)
        m_m = re.search(r'M0*(\d+)', base)
        if not (f_m and m_m):
            return None
        female_id = int(f_m.group(1))
        male_id = int(m_m.group(1))

        # Combine speaker text
        female_text = " ".join([str(t) for t in df[df["speaker"].str.lower() == "female"]["text"].dropna()])
        male_text = " ".join([str(t) for t in df[df["speaker"].str.lower() == "male"]["text"].dropna()])

        # Morphological analysis
        doc_f = _WORKER_NLP(female_text)
        doc_m = _WORKER_NLP(male_text)

        # Count & total words
        cnt_f = count_liwc_categories(doc_f, _WORKER_COMPILED)
        cnt_m = count_liwc_categories(doc_m, _WORKER_COMPILED)
        tot_f = count_total_words(doc_f)
        tot_m = count_total_words(doc_m)

        # LSM score
        try:
            lsm_scores = compute_lsm(cnt_f, cnt_m, _WORKER_TARGET_CATS, tot_f, tot_m, rounding=None)
        except TypeError:
            lsm_scores = compute_lsm(cnt_f, cnt_m, _WORKER_TARGET_CATS, tot_f, tot_m)

        vals = list(lsm_scores.values())
        lsm_mean = float(np.nan) if not vals else float(np.mean(vals))
        lsm_std = float(np.nan) if not vals else (0.0 if len(vals) == 1 else float(np.std(vals, ddof=0)))

        rec: Dict[str, Any] = {
            "female_id": female_id,
            "male_id": male_id,
            "file_path": file_path,
            "round_label": None,  # Assigned by parent
            "female_total_words": int(tot_f),
            "male_total_words": int(tot_m),
            "lsm_mean": lsm_mean,
            "lsm_std": lsm_std,
        }
        for cat in _WORKER_TARGET_CATS:
            rec[f"lsm_{cat}"] = lsm_scores.get(cat, np.nan)
        # --- Additional metadata ---
        rec["dyad_id"] = f"F{female_id:03d}_M{male_id:03d}"
        rec["file_basename"] = os.path.basename(file_path)
        rec["lsm_n_active_cats"] = int(len(lsm_scores))
        rec["categories_order"] = "|".join(_WORKER_TARGET_CATS)
        rec["dic_filename"] = _WORKER_DIC_BASENAME
        rec["spacy_model"] = _WORKER_SPACY_MODEL
        rec["has_timestamps"] = bool("start" in df.columns)

        return rec

    except Exception:
        # Return None on failure (parent skips silently)
        return None


def _window_index(t_start_sec: float, window_sec: int) -> int:
    if t_start_sec is None or np.isnan(t_start_sec):
        return 0
    return int(t_start_sec // window_sec)


def process_file_temporal(args) -> Optional[list]:
    """
    Calculate temporal LSM (e.g. 2.5 min) and return array of records for each time window.
    args: (file_path, window_sec)
    """
    try:
        file_path, window_sec = args
        assert _WORKER_NLP is not None and _WORKER_COMPILED is not None

        analyzer = DataAnalyzer()
        df = analyzer.get_single_conversation(file_path)
        if df is None or df.empty or ("text" not in df.columns) or ("speaker" not in df.columns):
            return []

        # Skip temporal if no time column
        if "start" not in df.columns:
            return []

        base = os.path.basename(file_path)
        f_m = re.search(r'F0*(\d+)', base)
        m_m = re.search(r'M0*(\d+)', base)
        if not (f_m and m_m):
            return []
        female_id = int(f_m.group(1))
        male_id = int(m_m.group(1))

        # Extract and format only needed columns
        df = df[["speaker", "text", "start"]].dropna(subset=["text"])
        df["speaker"] = df["speaker"].str.lower()

        # Batch analyze utterance texts
        texts = [str(t) for t in df["text"].tolist()]
        docs = list(_WORKER_NLP.pipe(texts, batch_size=256, n_process=1))  # n_process=1 OK since already multiprocess

        # Counters per window
        cnt_f_by = {}
        cnt_m_by = {}
        tot_f_by = {}
        tot_m_by = {}

        for (speaker, start), doc in zip(df[["speaker", "start"]].itertuples(index=False, name=None), docs):
            w = _window_index(float(start), window_sec)
            cats = count_liwc_categories(doc, _WORKER_COMPILED)
            tot = count_total_words(doc)

            if speaker == "female":
                cnt_f_by[w] = cnt_f_by.get(w, Counter()); cnt_f_by[w].update(cats)
                tot_f_by[w] = tot_f_by.get(w, 0) + tot
            elif speaker == "male":
                cnt_m_by[w] = cnt_m_by.get(w, Counter()); cnt_m_by[w].update(cats)
                tot_m_by[w] = tot_m_by.get(w, 0) + tot
            else:
                # Ignore other speaker labels
                continue

        # Calculate LSM per window
        results = []
        if not cnt_f_by and not cnt_m_by:
            return results

        max_w = max(list(cnt_f_by.keys() or [0]) + list(cnt_m_by.keys() or [0]))
        for w in range(max_w + 1):
            cf = cnt_f_by.get(w, Counter())
            cm = cnt_m_by.get(w, Counter())
            tf = int(tot_f_by.get(w, 0))
            tm = int(tot_m_by.get(w, 0))

            try:
                lsm_scores = compute_lsm(cf, cm, _WORKER_TARGET_CATS, tf, tm, rounding=None)
            except TypeError:
                lsm_scores = compute_lsm(cf, cm, _WORKER_TARGET_CATS, tf, tm)

            vals = list(lsm_scores.values())
            lsm_mean = float(np.nan) if not vals else float(np.mean(vals))
            lsm_std  = float(np.nan) if not vals else (0.0 if len(vals) == 1 else float(np.std(vals, ddof=0)))

            rec = {
                "female_id": female_id,
                "male_id": male_id,
                "file_path": file_path,
                "round_label": None,  # Assigned by parent
                "window_index": int(w),
                "start_sec": int(w * window_sec),
                "end_sec": int((w + 1) * window_sec),
                "female_total_words_window": tf,
                "male_total_words_window": tm,
                "lsm_mean": lsm_mean,
                "lsm_std": lsm_std,
            }
            for cat in _WORKER_TARGET_CATS:
                rec[f"lsm_{cat}"] = lsm_scores.get(cat, np.nan)
            # --- Additional metadata (window) ---
            rec["dyad_id"] = f"F{female_id:03d}_M{male_id:03d}"
            rec["file_basename"] = os.path.basename(file_path)
            rec["lsm_n_active_cats"] = int(len(lsm_scores))
            rec["window_size_sec"] = int(window_sec)
            rec["categories_order"] = "|".join(_WORKER_TARGET_CATS)
            rec["dic_filename"] = _WORKER_DIC_BASENAME
            rec["spacy_model"] = _WORKER_SPACY_MODEL
            results.append(rec)

        return results

    except Exception:
        return []
