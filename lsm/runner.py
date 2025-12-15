# -*- coding: utf-8 -*-
"""
Language Style Matching execution script (parallel, CSV save only)
Example execution:
cd /workspace/exploration/Transcript
python -m lsm.run_lsm \
  --round 1 2 \
  --dic /workspace/exploration/Transcript/Japanese_Dictionary.dic \
  --results /workspace/results/Transcript/2025-08-1-LanguageStyleMatching \
  --procs 8 --chunksize 8
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import platform
import multiprocessing as mp
import pandas as pd
from . import lsm_workers as mod

# --- Prevent parallel x BLAS runaway (always first in parent process) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Use utilities under /workspace
sys.path.append("/workspace/")
from utils import DataAnalyzer  # noqa: E402

# Child process side processing
from .lsm_workers import init_worker, process_file, process_file_temporal


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


def get_ctx():
    """Determine start method based on platform. Use fork for devcontainer(Linux)."""
    return mp.get_context("fork" if platform.system() == "Linux" else "spawn")


def run_lsm_for_round_parallel(round_number: int,
                               dic_path: str,
                               results_dir: str,
                               procs: int,
                               chunksize: int,
                               temporal: bool = False,
                               window_min: float = 5.0) -> Optional[pd.DataFrame]:
    analyzer = DataAnalyzer()
    files = analyzer.get_conversation_files(round_number=round_number)
    if not files:
        print(f"[INFO] round {round_number}: No conversation files")
        return None

    print(f"[INFO] round {round_number}: {len(files)} files / {procs} processes (chunksize={chunksize})")

    ctx = get_ctx()
    rows: List[Dict[str, Any]] = []

    with ctx.Pool(
        processes=procs,
        initializer=init_worker,
        initargs=(dic_path, FUNCTION_WORD_CATEGORIES),
    ) as pool:
        for i, rec in enumerate(pool.imap_unordered(process_file, files, chunksize=chunksize), 1):
            if rec:
                rec["round_label"] = "1st_round" if round_number == 1 else "2nd_round"
                rows.append(rec)
            if i % 50 == 0:
                print(f"[{round_number}] processed {i}/{len(files)}")

        if not rows:
            print(f"[INFO] round {round_number}: No successful records")
            return None

        df = pd.DataFrame(rows)

        # Save
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        tag = "1st" if round_number == 1 else "2nd"
        csv_rounded = os.path.join(results_dir, f"lsm_results_{tag}.csv")
        csv_full = os.path.join(results_dir, f"lsm_results_{tag}_full.csv")

        df["round_num"] = df["round_label"].map({"1st_round": 1, "2nd_round": 2}).astype("Int64")
        df.to_csv(csv_rounded, index=False, float_format="%.6f")
        df.to_csv(csv_full, index=False)
        print(f"[OK] saved: {csv_rounded} ({len(df)} rows)")
        print(f"[OK] saved: {csv_full}    ({len(df)} rows)")

        # Optional temporal analysis
        if temporal:
            print(f"[INFO] temporal windows: {window_min} min")
            wsec = int(round(window_min * 60))
            trows = []
            for trecs in pool.imap_unordered(process_file_temporal, [(f, wsec) for f in files], chunksize=chunksize):
                if trecs:
                    for r in trecs:
                        r["round_label"] = "1st_round" if round_number == 1 else "2nd_round"
                    trows.extend(trecs)
            if trows:
                tdf = pd.DataFrame(trows)
                tdf["round_num"] = tdf["round_label"].map({"1st_round": 1, "2nd_round": 2}).astype("Int64")
                tag = "1st" if round_number == 1 else "2nd"
                csv_temporal = os.path.join(results_dir, f"lsm_temporal_{tag}.csv")
                tdf.to_csv(csv_temporal, index=False, float_format="%.6f")
                print(f"[OK] saved: {csv_temporal} ({len(tdf)} rows)")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Language Style Matching batch runner")
    parser.add_argument("--round", nargs="+", type=int, default=[1, 2],
                        help="Target rounds (e.g. --round 1 2)")
    parser.add_argument("--dic", type=str,
                        default="/workspace/exploration/Transcript/Japanese_Dictionary.dic",
                        help="Absolute path to LIWC dictionary file (.dic)")
    parser.add_argument("--results", type=str,
                        default="/workspace/results/Transcript/2025-08-1-LanguageStyleMatching",
                        help="CSV output directory")
    parser.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 1),
                        help="Number of parallel processes")
    parser.add_argument("--chunksize", type=int, default=8,
                        help="chunksize for imap_unordered")
    parser.add_argument("--temporal", action="store_true",
                        help="Execute temporal analysis")
    parser.add_argument("--window", type=float, default=5.0,
                        help="Window size for temporal analysis (minutes)")
    return parser.parse_args()


def main():
    args = parse_args()

    dic_path = str(Path(args.dic).resolve())
    results_dir = str(Path(args.results).resolve())

    print(f"[INFO] target categories: {', '.join(FUNCTION_WORD_CATEGORIES)}")
    print(f"[INFO] dictionary      : {dic_path}")
    print(f"[INFO] results_dir     : {results_dir}")

    for r in args.round:
        _ = run_lsm_for_round_parallel(
            round_number=r,
            dic_path=dic_path,
            results_dir=results_dir,
            procs=args.procs,
            chunksize=args.chunksize,
            temporal=args.temporal,
            window_min=args.window,
        )


if __name__ == "__main__":
    main()
