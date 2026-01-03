# -*- coding: utf-8 -*-
"""
rLSM execution script (parallel, CSV save: pairs/individual category/dyad category/final/individual overall)
Example execution:
python -m rlsm.runner \
  --data ./sample_data \
  --dic path/to/Japanese_Dictionary.dic \
  --results ./results/rlsm \
  --procs 8 --chunksize 8 --na_policy bilateral_only
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import platform
import multiprocessing as mp
import pandas as pd
from functools import partial

# --- Prevent parallel x BLAS runaway (always first in parent process) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Child process side processing
from .workers import init_worker, process_file_rlsm

# ---- Settings: Function word categories (maintain specified order) ----
FUNCTION_WORD_CATEGORIES: List[str] = [
    "ppron",     # Personal pronouns
    "ipron",     # Indefinite pronouns
    "casepart",  # Case particles
    "auxverb",   # Auxiliary verbs
    "adverb",    # Adverbs
    "conj",      # Conjunctions
    "negate"     # Negation
]

# Add near the top category definitions
MICRO_PPRON_CATEGORIES: List[str] = ["i", "we", "you", "shehe", "they"]

def get_ctx():
    """Determine start method based on platform. Use fork for devcontainer(Linux)."""
    return mp.get_context("fork" if platform.system() == "Linux" else "spawn")

def list_csv_files(data_dir: str) -> List[str]:
    p = Path(data_dir)
    return sorted([str(f) for f in p.glob("*.csv") if f.is_file()])

def _parse_round_from_filename(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Helper to add round information in parent script.
    Expected filename style: Like "02_1_5_D_F001_M020_concat.csv",
    where second number in underscore-separated parts is round number.
    Return: (round_label, round_num)
    Example: 1 -> ("1st_round", 1), 2 -> ("2nd_round", 2), otherwise/unknown -> (None, None)
    """
    try:
        base = os.path.basename(file_path)
        m = re.match(r"^(\d+)_([0-9]+)_", base)
        if not m:
            parts = base.split("_")
            # Fallback: Use if 2nd element is numeric
            if len(parts) >= 2 and parts[1].isdigit():
                num = int(parts[1])
            else:
                return None, None
        else:
            num = int(m.group(2))
        if num == 1:
            return "1st_round", 1
        if num == 2:
            return "2nd_round", 2
        return None, None
    except Exception:
        return None, None

def run_rlsm_parallel(data_dir: str,
                      dic_path: str,
                      results_dir: str,
                      procs: int,
                      chunksize: int,
                      na_policy: str,
                      eps: float,
                      zero_tol: float,
                      unitize: str,
                      merge_gap: float,
                      strip_tags: int,
                      bracket_mode: str,
                      enable_micro_ppron: int,
                      rw: int,
                      rw_window_size: int,
                      rw_include_current: int,
                      rw_min_window_tokens: int) -> Optional[Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]]:

    files = list_csv_files(data_dir)
    if not files:
        print(f"[INFO] no CSV files under: {data_dir}")
        return None

    print(f"[INFO] files: {len(files)} | procs={procs} chunksize={chunksize}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    ctx = get_ctx()
    conv_rows: List[Dict[str, Any]] = []          # Conversation unit (final, individual overall, etc.)
    pairs_rows: List[Dict[str, Any]] = []         # Pair x category
    indiv_cat_rows: List[Dict[str, Any]] = []     # Individual x category
    dyad_cat_rows: List[Dict[str, Any]] = []      # Dyad x category
    unitize_reports: List[Dict[str, Any]] = []    # Unitize process reports

    # Add flag and micro categories at the end when constructing task_args
    task_args = [(fp, na_policy, eps, zero_tol, FUNCTION_WORD_CATEGORIES,
                  unitize, merge_gap, strip_tags, bracket_mode,
                  bool(enable_micro_ppron), MICRO_PPRON_CATEGORIES) for fp in files]

    # Build context
    context = {
        "rw_enabled": bool(rw),
        "rw_window_size": rw_window_size,
        "rw_include_current": rw_include_current,
        "rw_min_window_tokens": rw_min_window_tokens,
        "enable_micro_ppron": bool(enable_micro_ppron),
        "micro_categories": MICRO_PPRON_CATEGORIES if enable_micro_ppron else [],
    }

    # Bind context to process_file_rlsm
    worker_func = partial(process_file_rlsm, context=context)

    errors = []
    # For collection (micro in addition to existing)
    micro_pair_rows: List[Dict[str, Any]] = []
    micro_indiv_rows: List[Dict[str, Any]] = []
    micro_dyad_rows: List[Dict[str, Any]] = []
    
    # For collection (rw rolling window)
    rw_conv_rows: List[Dict[str, Any]] = []
    rw_pair_rows: List[Dict[str, Any]] = []
    rw_indiv_rows: List[Dict[str, Any]] = []
    rw_dyad_rows: List[Dict[str, Any]] = []
    rw_micro_pair_rows: List[Dict[str, Any]] = []
    rw_micro_indiv_rows: List[Dict[str, Any]] = []
    rw_micro_dyad_rows: List[Dict[str, Any]] = []

    if procs == 1:
        # Sequential execution (useful for mocking/debugging, OK with non-picklable functions)
        it = (worker_func(args) for args in task_args)
        enum_it = enumerate(it, 1)
    else:
        pool = ctx.Pool(
            processes=procs,
            initializer=init_worker,
            initargs=(dic_path, FUNCTION_WORD_CATEGORIES, MICRO_PPRON_CATEGORIES if enable_micro_ppron else []),
        )
        enum_it = enumerate(pool.imap_unordered(worker_func, task_args, chunksize=chunksize), 1)

    for i, rec in enum_it:
            if not rec:
                continue
            # If error, record and skip
            if "error" in rec:
                msg = rec["error"]
                fpath = rec.get("file_path", "")
                print(f"[ERROR] {msg}")
                errors.append({"file_path": fpath, "error": msg})
                continue

            # Existing collection
            # Add round_label/round_num on parent side (align with LSM)
            conv = rec["conversation_summary"]
            file_path = conv.get("file_path")
            round_label, round_num = _parse_round_from_filename(file_path)
            if round_label is not None:
                conv["round_label"] = round_label
            if round_num is not None:
                conv["round_num"] = int(round_num)

            def _apply_round(rows: List[Dict[str, Any]]):
                if not rows:
                    return rows
                for r in rows:
                    if round_label is not None:
                        r["round_label"] = round_label
                    if round_num is not None:
                        r["round_num"] = int(round_num)
                return rows

            conv_rows.append(conv)
            pairs_rows.extend(_apply_round(rec["pairs_rows"]))
            indiv_cat_rows.extend(_apply_round(rec["individual_category_rows"]))
            dyad_cat_rows.extend(_apply_round(rec["dyad_category_rows"]))
            if "unitize_report" in rec:
                report = rec["unitize_report"].copy()
                report["file_path"] = rec["conversation_summary"]["file_path"]
                unitize_reports.append(report)

            # Collect micro
            if bool(enable_micro_ppron):
                micro_pair_rows.extend(_apply_round(rec.get("micro_pair_rows", [])))
                micro_indiv_rows.extend(_apply_round(rec.get("micro_individual_rows", [])))
                micro_dyad_rows.extend(_apply_round(rec.get("micro_dyad_rows", [])))

            # Collect rw rolling window
            if bool(rw):
                if "rw_conv_row" in rec:
                    rw_conv = rec["rw_conv_row"]
                    if round_label is not None:
                        rw_conv["round_label"] = round_label
                    if round_num is not None:
                        rw_conv["round_num"] = int(round_num)
                    rw_conv_rows.append(rw_conv)
                rw_pair_rows.extend(_apply_round(rec.get("rw_pair_rows", [])))
                rw_indiv_rows.extend(_apply_round(rec.get("rw_individual_rows", [])))
                rw_dyad_rows.extend(_apply_round(rec.get("rw_dyad_rows", [])))
                
                # Collect rw micro
                if bool(enable_micro_ppron):
                    rw_micro_pair_rows.extend(_apply_round(rec.get("rw_micro_pair_rows", [])))
                    rw_micro_indiv_rows.extend(_apply_round(rec.get("rw_micro_individual_rows", [])))
                    rw_micro_dyad_rows.extend(_apply_round(rec.get("rw_micro_dyad_rows", [])))

            if i % 25 == 0 or i == len(files):
                print(f"[INFO] processed {i}/{len(files)}")

    if procs != 1:
        pool.close()
        pool.join()

    if not conv_rows:
        print("[INFO] no successful conversations.")
        return None

    # ---- Convert to DataFrame ----
    df_conv = pd.DataFrame(conv_rows)
    sort_keys = [k for k in ["female_id", "male_id"] if k in df_conv.columns]
    if sort_keys:
        df_conv = df_conv.sort_values(sort_keys, na_position="first").reset_index(drop=True)
    df_pairs = pd.DataFrame(pairs_rows) if pairs_rows else pd.DataFrame()
    df_indiv_cat = pd.DataFrame(indiv_cat_rows) if indiv_cat_rows else pd.DataFrame()
    df_dyad_cat = pd.DataFrame(dyad_cat_rows) if dyad_cat_rows else pd.DataFrame()

    # Convert to DataFrame (micro in addition to existing)
    df_micro_pairs = pd.DataFrame(micro_pair_rows) if micro_pair_rows else pd.DataFrame()
    df_micro_indiv = pd.DataFrame(micro_indiv_rows) if micro_indiv_rows else pd.DataFrame()
    df_micro_dyad  = pd.DataFrame(micro_dyad_rows)  if micro_dyad_rows  else pd.DataFrame()

    # Convert to DataFrame (rw rolling window)
    df_rw_conv = pd.DataFrame(rw_conv_rows) if rw_conv_rows else pd.DataFrame()
    df_rw_pairs = pd.DataFrame(rw_pair_rows) if rw_pair_rows else pd.DataFrame()
    df_rw_indiv = pd.DataFrame(rw_indiv_rows) if rw_indiv_rows else pd.DataFrame()
    df_rw_dyad = pd.DataFrame(rw_dyad_rows) if rw_dyad_rows else pd.DataFrame()
    df_rw_micro_pairs = pd.DataFrame(rw_micro_pair_rows) if rw_micro_pair_rows else pd.DataFrame()
    df_rw_micro_indiv = pd.DataFrame(rw_micro_indiv_rows) if rw_micro_indiv_rows else pd.DataFrame()
    df_rw_micro_dyad = pd.DataFrame(rw_micro_dyad_rows) if rw_micro_dyad_rows else pd.DataFrame()

    # ---- Save (rounded and full versions) ----
    def _save(df: pd.DataFrame, name: str):
        if df is None or df.empty:
            return
        p1 = os.path.join(results_dir, f"{name}.csv")
        p2 = os.path.join(results_dir, f"{name}_full.csv")
        df.to_csv(p1, index=False, float_format="%.6f")
        df.to_csv(p2, index=False)
        print(f"[OK] saved: {p1} ({len(df)} rows)")
        print(f"[OK] saved: {p2} ({len(df)} rows)")

    # Save (reuse existing _save)
    _save(df_conv, "rlsm_conversations")             # Conversation unit summary (includes final/individual overall/category-wise dyad)
    _save(df_pairs, "pairs_rlsm")                    # Pair x category (attributed to responder)
    _save(df_indiv_cat, "individual_category_rlsm")  # Individual x category
    _save(df_dyad_cat, "dyad_category_rlsm")         # Dyad x category

    if bool(enable_micro_ppron):
        _save(df_micro_pairs, "pairs_micro_rlsm")
        _save(df_micro_indiv, "individual_micro_rlsm")
        _save(df_micro_dyad,  "dyad_micro_rlsm")

    if bool(rw):
        _save(df_rw_conv, "rw_rlsm_conversations")
        _save(df_rw_pairs, "rw_pairs_rlsm")
        _save(df_rw_indiv, "rw_individual_category_rlsm")
        _save(df_rw_dyad, "rw_dyad_category_rlsm")
        
        if bool(enable_micro_ppron):
            _save(df_rw_micro_pairs, "pairs_rw_micro_rlsm")
            _save(df_rw_micro_indiv, "individual_rw_micro_rlsm")
            _save(df_rw_micro_dyad, "dyad_rw_micro_rlsm")

    # Save error list (after DataFrame save processing)
    if errors:
        df_err = pd.DataFrame(errors).sort_values("file_path")
        err_csv = os.path.join(results_dir, "rlsm_errors.csv")
        df_err.to_csv(err_csv, index=False)
        print(f"[WARN] {len(errors)} files failed validation. See: {err_csv}")
        # Save same content under rw name too (spec #5)
        if bool(rw):
            err_csv2 = os.path.join(results_dir, "rw_rlsm_errors.csv")
            df_err.to_csv(err_csv2, index=False)
            print(f"[WARN] (rw) See: {err_csv2}")
    
    # Save unitize reports
    if unitize_reports:
        df_unitize = pd.DataFrame(unitize_reports)
        unitize_csv = os.path.join(results_dir, "unitize_reports.csv")
        df_unitize.to_csv(unitize_csv, index=False)
        print(f"[INFO] Unitize reports saved: {unitize_csv} ({len(unitize_reports)} files)")

    return df_conv, {
        "pairs": df_pairs,
        "individual_category": df_indiv_cat,
        "dyad_category": df_dyad_cat,
        "unitize_reports": pd.DataFrame(unitize_reports) if unitize_reports else pd.DataFrame(),
        # Additional return (optional)
        "micro_pairs": df_micro_pairs,
        "micro_individual": df_micro_indiv,
        "micro_dyad": df_micro_dyad,
        # Additional return for rw rolling window
        "rw_conv": df_rw_conv,
        "rw_pairs": df_rw_pairs,
        "rw_individual": df_rw_indiv,
        "rw_dyad": df_rw_dyad,
        "rw_micro_pairs": df_rw_micro_pairs,
        "rw_micro_individual": df_rw_micro_indiv,
        "rw_micro_dyad": df_rw_micro_dyad,
    }

def parse_args():
    ap = argparse.ArgumentParser(description="rLSM batch runner (paper-compliant)")
    ap.add_argument("--data", type=str, default="./sample_data",
                    help="Input CSV directory (*.csv directly under it)")
    ap.add_argument("--dic", type=str, required=True,
                    help="Path to LIWC dictionary (.dic)")
    ap.add_argument("--results", type=str, default="./results/rlsm",
                    help="CSV output directory")
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Number of parallel processes")
    ap.add_argument("--chunksize", type=int, default=8,
                    help="chunksize for imap_unordered")
    ap.add_argument("--na_policy", type=str, default="bilateral_only", choices=["bilateral_only", "nanmean"],
                    help="NA policy for dyad category averaging")
    ap.add_argument("--eps", type=float, default=1e-4, help="EPS for rLSM calculation (default=1e-4)")
    ap.add_argument("--zero_tol", type=float, default=0.0, help="Threshold to consider as 0 (default=0.0)")
    ap.add_argument("--unitize", choices=["off", "approx", "strict"], default="approx",
                    help="Preprocess transcript into non-overlapping alternating talk-turns.")
    ap.add_argument("--merge_gap", type=float, default=0.0,
                    help="Seconds allowed to merge consecutive same-speaker turns.")
    ap.add_argument("--strip_tags", type=int, choices=[0,1], default=1,
                    help="Strip []-brackets per policy (1:on, 0:off).")
    ap.add_argument("--bracket_mode", choices=["unwrap", "drop_all", "off"], default="unwrap",
                    help="How to treat [ ... ]: unwrap (keep inner), drop_all, or off.")
    ap.add_argument("--enable_micro_ppron", type=int, choices=[0,1], default=0,
                    help="Enable micro-ppron rLSM (i,we,you,shehe,they) as auxiliary outputs.")
    ap.add_argument("--rw", dest="rw", type=int, choices=[0,1], default=1, 
                    help="Calculate rw.rLSM simultaneously (1=yes, 0=no)")
    ap.add_argument("--rw_window_size", type=int, default=8,
                    help="Number of utterances in rolling window (paper default is 8=past 7 + current)")
    ap.add_argument("--rw_include_current", type=int, choices=[0,1], default=1,
                    help="Include current turn in window average (1=include, 0=exclude)")
    ap.add_argument("--rw_min_window_tokens", type=int, default=0,
                    help="Small window filter threshold (0=disabled. Output to conversation summary and pair table)")
    return ap.parse_args()

def main():
    args = parse_args()
    dic_path = str(Path(args.dic).resolve())
    data_dir = str(Path(args.data).resolve())
    results_dir = str(Path(args.results).resolve())

    print(f"[INFO] target categories: {', '.join(FUNCTION_WORD_CATEGORIES)}")
    print(f"[INFO] dictionary      : {dic_path}")
    print(f"[INFO] data_dir        : {data_dir}")
    print(f"[INFO] results_dir     : {results_dir}")
    print(f"[INFO] na_policy={args.na_policy}, eps={args.eps}, zero_tol={args.zero_tol}")
    print(f"[INFO] unitize={args.unitize}, merge_gap={args.merge_gap}, strip_tags={args.strip_tags}, bracket_mode={args.bracket_mode}")
    print(f"[INFO] enable_micro_ppron={args.enable_micro_ppron}")
    print(f"[INFO] rw={args.rw}, rw_window_size={args.rw_window_size}, rw_include_current={args.rw_include_current}")
    print(f"[INFO] rw_min_window_tokens={args.rw_min_window_tokens}")

    _ = run_rlsm_parallel(
        data_dir=data_dir,
        dic_path=dic_path,
        results_dir=results_dir,
        procs=args.procs,
        chunksize=args.chunksize,
        na_policy=args.na_policy,
        eps=args.eps,
        zero_tol=args.zero_tol,
        unitize=args.unitize,
        merge_gap=args.merge_gap,
        strip_tags=args.strip_tags,
        bracket_mode=args.bracket_mode,
        enable_micro_ppron=args.enable_micro_ppron,
        rw=args.rw,
        rw_window_size=args.rw_window_size,
        rw_include_current=args.rw_include_current,
        rw_min_window_tokens=args.rw_min_window_tokens,
    )

if __name__ == "__main__":
    main()
