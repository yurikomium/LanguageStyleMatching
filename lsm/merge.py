"""
=============================================================================
lsm_conversations.csv (Long format summary of static LSM)

Purpose:
- Unify LSM (1st/2nd round) in "long" format instead of "wide".
- Make it easy to uniquely join with rLSM / basic info by (dyad_id, female_id, male_id, round_num).
- Unify column names with lsm_ prefix (also add lsm_ to file_path etc. to avoid collisions).
- Auto-calculate lsm_total_words / lsm_f_talk_ratio / lsm_n_active_cats if missing.

Example command:
python lsm/merge.py \
  --lsm_1st ./results/lsm/data/lsm_results_1st_full.csv \
  --lsm_2nd ./results/lsm/data/lsm_results_2nd_full.csv \
  --out ./data/processed/lsm/lsm_conversations.csv
=============================================================================
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Output unique key
KEY_COLS = ["dyad_id", "female_id", "male_id", "round_num"]

# LSM categories (aligned with same granularity as rLSM)
LSM_CATS = ["adverb", "auxverb", "casepart", "conj", "ipron", "ppron", "negate"]


def _ensure_dyad_id(df: pd.DataFrame) -> pd.DataFrame:
    """Generate dyad_id from female_id / male_id if missing (following your existing function)."""
    df = df.copy()
    if "dyad_id" not in df.columns:
        if not {"female_id", "male_id"} <= set(df.columns):
            raise ValueError("dyad_id is missing and cannot be generated from female_id / male_id.")
        def _fmt(v, tag):
            try:
                i = int(str(v).strip().lstrip("FfMm0"))
                return f"{tag}{i:03d}"
            except Exception:
                return f"{tag}{str(v).strip()}"
        df["dyad_id"] = [
            f"{_fmt(f, 'F')}_{_fmt(m, 'M')}"
            for f, m in zip(df["female_id"], df["male_id"])
        ]
    return df


def _safe_rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Safely rename only existing columns."""
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=present)


def _compute_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing derived columns (lsm_total_words / lsm_f_talk_ratio / lsm_n_active_cats)."""
    df = df.copy()

    # lsm_total_words
    if "lsm_total_words" not in df.columns:
        f_col = "lsm_female_total_words"
        m_col = "lsm_male_total_words"
        if f_col in df.columns and m_col in df.columns:
            df["lsm_total_words"] = df[f_col].fillna(0) + df[m_col].fillna(0)

    # lsm_f_talk_ratio = female / total
    if "lsm_f_talk_ratio" not in df.columns:
        f_col, t_col = "lsm_female_total_words", "lsm_total_words"
        if f_col in df.columns and t_col in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["lsm_f_talk_ratio"] = np.where(
                    df[t_col] > 0, df[f_col] / df[t_col], np.nan
                )

    # lsm_n_active_cats = number of non-NaN lsm_* categories
    if "lsm_n_active_cats" not in df.columns:
        cand = [f"lsm_{c}" for c in LSM_CATS if f"lsm_{c}" in df.columns]
        if cand:
            df["lsm_n_active_cats"] = df[cand].notna().sum(axis=1)

    return df

def _standardize_one_round(df: pd.DataFrame, round_num: int) -> pd.DataFrame:
    """
    Format one round of LSM CSV to standard columns for long format.
    - Unify column name prefix to lsm_ (including non-collision)
    - Add round_num / round_label
    """
    df = df.copy()

    # Ensure dyad_id
    df = _ensure_dyad_id(df)

    # Keep columns already starting with lsm_ as is, rename others per convention
    base_rename = {
        "categories_order": "lsm_categories_order",
        "dic_filename": "lsm_dic_filename",
        "female_total_words": "lsm_female_total_words",
        "male_total_words": "lsm_male_total_words",
        "total_words": "lsm_total_words",
        "file_path": "lsm_file_path",
        "file_basename": "lsm_file_basename",
        "has_timestamps": "lsm_has_timestamps",
        "spacy_model": "lsm_spacy_model",
        "f_talk_ratio": "lsm_f_talk_ratio",
        # Keep lsm_mean / lsm_std / lsm_* categories as is if present
    }
    df = _safe_rename(df, base_rename)

    # Drop any excess round info (will reassign in this function)
    df = df.drop(columns=[c for c in ["round_label", "round_num"] if c in df.columns])

    # Add round
    df["round_num"] = int(round_num)
    # Don't keep round_label in LSM long since basic info manages it centrally

    # Calculate derived columns if missing
    df = _compute_derivatives(df)

    # Column order (for readability)
    key_order = ["dyad_id", "female_id", "male_id", "round_num"]
    # Delegate path/filename to basic info side (avoid duplication)
    df = df.drop(columns=[c for c in ["lsm_file_path", "lsm_file_basename"] if c in df.columns])
    lsm_cols = sorted(
        [c for c in df.columns if c.startswith("lsm_")]
    )
    others = [c for c in df.columns if c not in key_order + lsm_cols]
    df = df[key_order + lsm_cols + others]

    return df


def _ensure_unique(df: pd.DataFrame) -> None:
    """Check that (dyad_id, female_id, male_id, round_num) is unique."""
    dup = (
        df.groupby(KEY_COLS, dropna=False)
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    if not dup.empty:
        raise ValueError(
            f"Duplicate keys exist ({len(dup)} cases). Please resolve duplicates in input CSV first.\n"
            f"Example: {dup.head(5).to_dict(orient='records')}"
        )


def build_lsm_long(lsm_first_csv: Path, lsm_second_csv: Path) -> pd.DataFrame:
    """Combine 1st and 2nd round LSM result CSVs in long format and return."""
    df1_raw = pd.read_csv(lsm_first_csv)
    df2_raw = pd.read_csv(lsm_second_csv)

    df1 = _standardize_one_round(df1_raw, round_num=1)
    df2 = _standardize_one_round(df2_raw, round_num=2)

    merged_long = pd.concat([df1, df2], ignore_index=True)

    # Align ID dtypes with basic info
    # female_id/male_id are int, dyad_id is "F%03d_M%03d" string
    if "female_id" in merged_long: merged_long["female_id"] = merged_long["female_id"].astype(int)
    if "male_id" in merged_long:   merged_long["male_id"]   = merged_long["male_id"].astype(int)
    if "dyad_id" in merged_long:
        # Reformat in case not already formatted
        merged_long["dyad_id"] = [
            f"F{int(f):03d}_M{int(m):03d}" for f, m in zip(merged_long["female_id"], merged_long["male_id"])
        ]

    _ensure_unique(merged_long)
    return merged_long


def main():
    ap = argparse.ArgumentParser(description="LSM(1st/2nd) long format merge script")
    ap.add_argument("--lsm_1st", required=True, help="1st round LSM CSV (e.g. lsm_results_1st_full.csv)")
    ap.add_argument("--lsm_2nd", required=True, help="2nd round LSM CSV (e.g. lsm_results_2nd_full.csv)")
    ap.add_argument(
        "--out",
        default="lsm_conversations_long.csv",
        help="Output CSV path (default: lsm_conversations_long.csv)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    df = build_lsm_long(Path(args.lsm_1st), Path(args.lsm_2nd))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] LSM long -> {out_path}  rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    main()
