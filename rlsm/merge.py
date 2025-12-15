"""
=============================================================================
rlsm_conversations.csv (static rLSM) + rw_rlsm_conversations.csv (rolling rLSM)
→ De-duplication & clean merge version

Usage example:
python merge_rlsm.py \
  --rlsm /workspace/results/Transcript/2025-08-2-相互LSM/data/rlsm_conversations_full.csv \
  --rw   /workspace/results/Transcript/2025-08-2-相互LSM/data/rw_rlsm_conversations_full.csv \
  --out  /workspace/data/processed/rlsm/rlsm_conversations.csv \
  --param-window 8 --include-current true --min-window-tokens 0 --na-policy any
=============================================================================
"""

import argparse, os
import pandas as pd
import numpy as np

KEYS = ["dyad_id", "female_id", "male_id", "round_num"]
PARAM_COLS = ["window_size_param", "include_current", "min_window_tokens", "na_policy"]

def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return np.nan


def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Align types
    for c in ["female_id", "male_id", "round_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Create dyad_id if missing
    if "dyad_id" not in df.columns and {"female_id","male_id"} <= set(df.columns):
        df["dyad_id"] = [
            f"F{int(f):03d}_M{int(m):03d}" if pd.notna(f) and pd.notna(m) else None
            for f, m in zip(df["female_id"], df["male_id"])
        ]
    return df


def _drop_rlsm_side_meta(rlsm: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rLSM-side metadata (delegated to basic info).
    User request: Remove file_path, file_basename.
    """
    drop_cols = ["file_path", "file_basename"]
    existing = [c for c in drop_cols if c in rlsm.columns]
    if existing:
        rlsm = rlsm.drop(columns=existing)
        print(f"[info] dropped from rlsm: {existing}")
    return rlsm


def _pick_join_keys(rlsm_cols, rw_cols):
    # Highest priority is KEYS (dyad_id, female_id, male_id, round_num)
    if all(k in rlsm_cols and k in rw_cols for k in KEYS):
        return KEYS
    # Fallback: Use file_basename temporarily if in both
    candidates = [
        KEYS + ["file_basename"],
        ["female_id", "male_id", "round_num"],
        ["female_id", "male_id"],
    ]
    for keys in candidates:
        if all(k in rlsm_cols and k in rw_cols for k in keys):
            return keys
    raise ValueError("Common keys not found. At least female_id, male_id, round_num required.")


def _select_param_combo_mode(rw: pd.DataFrame):
    if rw.empty:
        return None
    if not set(PARAM_COLS) <= set(rw.columns):
        return None
    vc = (
        rw[PARAM_COLS]
        .assign(include_current=lambda d: d["include_current"].map(_coerce_bool))
        .value_counts(dropna=False)
    )
    if len(vc) == 0:
        return None
    combo = vc.index[0]
    return dict(zip(PARAM_COLS, combo))


def _filter_by_params(rw: pd.DataFrame, param_window: int|None,
                      include_current: bool|None, min_window_tokens: int|None,
                      na_policy: str|None) -> pd.DataFrame:
    rw = rw.copy()
    if param_window is not None and "window_size_param" in rw.columns:
        rw = rw[rw["window_size_param"] == param_window]
    if include_current is not None and "include_current" in rw.columns:
        rw = rw[rw["include_current"].map(_coerce_bool) == include_current]
    if min_window_tokens is not None and "min_window_tokens" in rw.columns:
        rw = rw[rw["min_window_tokens"] == min_window_tokens]
    if na_policy is not None and "na_policy" in rw.columns:
        rw = rw[rw["na_policy"].astype(str) == str(na_policy)]
    return rw


def _dedup_rw_with_preferences(rw: pd.DataFrame, keys, prefer_window: int|None):
    """
    Last resort when multiple rows remain for same key: Select representative row.
    Priority:
      1) include_current=True
      2) window_size_param close to prefer_window (default=8)
      3) min_window_tokens large
      4) rw_n_pair_turns large
    """
    rw = rw.copy()

    # Normalize
    if "include_current" in rw.columns:
        rw["__incl"] = rw["include_current"].map(_coerce_bool).fillna(False).astype(int)
    else:
        rw["__incl"] = 0

    if "window_size_param" in rw.columns:
        target = prefer_window if prefer_window is not None else 8
        rw["__w_dist"] = (rw["window_size_param"] - target).abs()
    else:
        rw["__w_dist"] = 9999

    if "min_window_tokens" in rw.columns:
        rw["__min_tok"] = rw["min_window_tokens"].fillna(-1)
    else:
        rw["__min_tok"] = -1

    if "rw_n_pair_turns" in rw.columns:
        rw["__nwin"] = rw["rw_n_pair_turns"].fillna(-1)
    else:
        rw["__nwin"] = -1

    sort_cols = ["__incl", "__w_dist", "__min_tok", "__nwin"]
    asc = [False, True, False, False]  # incl desc, distance asc, tokens desc, windows desc

    rw = rw.sort_values(sort_cols, ascending=asc, kind="mergesort")
    rw = rw.drop_duplicates(subset=keys, keep="first").drop(columns=sort_cols)
    return rw


def _prefix_rw_columns(rw: pd.DataFrame, keys):
    rw = rw.copy()
    keep = set(keys)  # Keep keys as-is

    # Make existing parameter columns explicit with rw_ (to distinguish from static side na_policy)
    param_cols = ["window_size_param", "include_current", "min_window_tokens", "na_policy"]
    for p in param_cols:
        if p in rw.columns:
            rw = rw.rename(columns={p: f"rw_{p}"})
    keep |= {f"rw_{p}" for p in param_cols}

    # Leave columns already rw_, add rw_ only to others
    rename = {}
    for c in rw.columns:
        if c in keep:
            continue
        if c.startswith("rw_"):
            continue  # Prevent double prefixing
        rename[c] = f"rw_{c}"
    rw = rw.rename(columns=rename)

    # Drop meta unnecessary outside auditing (delegate to basic info)
    for c in ["rw_file_path", "rw_file_basename", "rw_round_label"]:
        if c in rw.columns:
            rw = rw.drop(columns=[c])

    return rw

def main(rlsm_path, rw_path, out_path,
         prefer_window: int|None, include_current: bool|None,
         min_window_tokens: int|None, na_policy: str|None):
    rlsm = pd.read_csv(rlsm_path)
    rw   = pd.read_csv(rw_path)

    rlsm = _ensure_ids(rlsm)
    # ★ Delete unnecessary metadata columns from rLSM side (file_path / file_basename)
    rlsm = _drop_rlsm_side_meta(rlsm)
    rw   = _ensure_ids(rw)

    # Normalize parameter columns (avoid type mismatches)
    if "include_current" in rw.columns:
        rw["include_current"] = rw["include_current"].map(_coerce_bool)

    keys = _pick_join_keys(rlsm.columns, rw.columns)
    print(f"[info] join keys = {keys}")

    # 1) User-specified or 2) Most frequent combo (mode) or 3) Representative by priority rules
    rw_original_len = len(rw)

    # 1) User-specified filter
    rw_f = _filter_by_params(rw, prefer_window, include_current, min_window_tokens, na_policy)

    # 2) If unspecified and nothing filtered, select most frequent combo
    if len(rw_f) == rw_original_len or rw_f.empty:
        combo = _select_param_combo_mode(rw)
        if combo:
            print(f"[info] auto-select most frequent params: {combo}")
            rw_f = _filter_by_params(rw, combo.get("window_size_param"),
                                     combo.get("include_current"),
                                     combo.get("min_window_tokens"),
                                     combo.get("na_policy"))
        else:
            rw_f = rw.copy()

    # 3) If duplicates still remain, select representative row
    dup = (
        rw_f.groupby(keys, dropna=False).size().reset_index(name="n").query("n>1")
    )
    if len(dup):
        print(f"[warn] duplicates after filtering: {len(dup)} key-groups → picking representatives")
        rw_f = _dedup_rw_with_preferences(rw_f, keys, prefer_window)

    # Final check
    assert (
        rw_f.groupby(keys, dropna=False).size().le(1).all()
    ), "internal error: deduplication failed"

    # To avoid column collisions, prefix rw_ to rw side except keys/parameters
    rw_pref = _prefix_rw_columns(rw_f, keys)

    # Collision (just in case): Assume no same names as rw_ on rlsm side, but drop/rename if any
    collision = sorted(set(rlsm.columns) & set(rw_pref.columns) - set(keys))
    if collision:
        print(f"[warn] dropping colliding columns from rw side: {collision}")
        rw_pref = rw_pref.drop(columns=collision)

    # Clean m:1 join
    merged = pd.merge(rlsm, rw_pref, on=keys, how="left", validate="m:1")

    # Check unmatched
    rw_value_cols = [c for c in rw_pref.columns if c not in set(keys) | set(PARAM_COLS)]
    unmatched = merged[rw_value_cols].isna().all(axis=1).sum()
    print(f"[info] rows={len(merged)}  rw_unmatched_left_join_rows={unmatched}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[ok] saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlsm", required=True, help="path to rlsm_conversations.csv")
    ap.add_argument("--rw", required=True, help="path to rw_rlsm_conversations.csv")
    ap.add_argument("--out", required=True, help="path to output merged csv")

    # Parameter preference (auto-select if unspecified → priority rules on duplication)
    ap.add_argument("--param-window", type=int, default=None,
                    help="Specify window_size_param (e.g. 8)")
    ap.add_argument("--include-current", type=str, default=None,
                    help="Specify as true/false (e.g. true)")
    ap.add_argument("--min-window-tokens", type=int, default=None,
                    help="Specify min_window_tokens (e.g. 0, 10, 20)")
    ap.add_argument("--na-policy", type=str, default=None,
                    help="Specify na_policy (e.g. any, strict, etc.)")

    args = ap.parse_args()
    include_current = None
    if args.include_current is not None:
        include_current = _coerce_bool(args.include_current)

    main(args.rlsm, args.rw, args.out,
         args.param_window, include_current,
         args.min_window_tokens, args.na_policy)
