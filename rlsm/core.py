"""
=============================================================================
rLSM (Reciprocal LSM) Function Definitions
=============================================================================
"""

from typing import List, Dict, Any, Tuple
import numpy as np

EPS = 1e-4  # Avoid division by zero

__all__ = [
    "EPS",
    "rlsm_per_category_from_rates",
    "compute_pair_category_rlsm",
    "aggregate_individual_category_means",
    "aggregate_dyad_category_means",
    "mean_over_categories",
    "compute_rlsm_core",
]

def rlsm_per_category_from_rates(
    prev_rates: Dict[str, float],
    curr_rates: Dict[str, float],
    categories: List[str],
    eps: float = EPS,
    zero_tol: float = 0.0,
) -> Dict[str, float]:
    """
    Category-wise rLSM for one exchange (prev=previous, curr=response) (Table 5 rules).
      - prev==0 & curr==0 -> NaN (exclude)
      - prev==0 & curr>0  -> NaN (exclude)
      - prev>0  & curr==0 -> Normal calculation (low rLSM)
    rLSM_c = 1 - |prev - curr| / (prev + curr + eps)
    """
    out = {}
    for c in categories:
        a = float(prev_rates.get(c, 0.0))
        b = float(curr_rates.get(c, 0.0))
        a_is_zero = abs(a) <= zero_tol
        b_is_zero = abs(b) <= zero_tol
        if a_is_zero and b_is_zero:
            out[c] = np.nan
        elif a_is_zero and not b_is_zero:
            out[c] = np.nan
        else:
            out[c] = 1.0 - (abs(a - b) / (a + b + eps))
    return out


def compute_pair_category_rlsm(
    turns: List[Dict[str, Any]],
    categories: List[str],
    eps: float = EPS,
    zero_tol: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Calculate category-wise rLSM only at adjacent pairs (i→i+1) where "speaker changes".
    turns: [{"speaker": any ID, "rates": {cat: usage_rate, ...}}, ...]
    Output attributed to responder (second speaker).
    """
    out = []
    for i in range(len(turns) - 1):
        s_prev = turns[i]["speaker"]
        s_curr = turns[i + 1]["speaker"]
        if s_prev == s_curr:
            continue
        prev_rates = turns[i]["rates"]
        curr_rates = turns[i + 1]["rates"]
        cat_scores = rlsm_per_category_from_rates(prev_rates, curr_rates, categories, eps, zero_tol)
        out.append({
            "pair_index": i,
            "leader": s_prev,
            "responder": s_curr,
            "category_scores": cat_scores,
        })
    return out


def aggregate_individual_category_means(
    pair_scores: List[Dict[str, Any]],
    categories: List[str],
) -> Dict[Any, Dict[str, float]]:
    """
    Individual x category mean (Eq. 12, 13): Aggregate only rLSMs calculated as responder to each person.
    """
    buckets: Dict[Any, Dict[str, List[float]]] = {}
    for rec in pair_scores:
        res = rec["responder"]
        if res not in buckets:
            buckets[res] = {c: [] for c in categories}
        for c, v in rec["category_scores"].items():
            if not np.isnan(v):
                buckets[res][c].append(float(v))

    means: Dict[Any, Dict[str, float]] = {}
    for spk, cat_lists in buckets.items():
        means[spk] = {}
        for c, vals in cat_lists.items():
            means[spk][c] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    return means


def aggregate_dyad_category_means(
    individual_cat_means: Dict[Any, Dict[str, float]],
    speakers: Tuple[Any, Any],
    categories: List[str],
    na_policy: str = "bilateral_only",
) -> Dict[str, float]:
    """
    Category-wise dyad (Eq. 14): dyad_c = mean( r_A,c , r_B,c )
    na_policy:
      - 'bilateral_only' : Drop category if either side is NaN
      - 'nanmean'        : Take mean even if one side is NaN
    """
    a, b = speakers
    dyad = {}
    for c in categories:
        a_val = individual_cat_means.get(a, {}).get(c, np.nan)
        b_val = individual_cat_means.get(b, {}).get(c, np.nan)
        if na_policy == "bilateral_only":
            dyad[c] = np.nan if (np.isnan(a_val) or np.isnan(b_val)) else (a_val + b_val) / 2.0
        elif na_policy == "nanmean":
            dyad[c] = float(np.nanmean([a_val, b_val]))
        else:
            raise ValueError("na_policy must be 'bilateral_only' or 'nanmean'")
    return dyad


def mean_over_categories(values_by_cat: Dict[str, float]) -> float:
    """Equal-weighted average over categories (ignore NaN. Return NaN if all NaN)."""
    vals = np.array([v for v in values_by_cat.values()], dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def compute_rlsm_core(
    turns: List[Dict[str, Any]],
    categories: List[str],
    *,
    eps: float = EPS,
    zero_tol: float = 0.0,
    na_policy: str = "bilateral_only",
) -> Dict[str, Any]:
    """
    rLSM core (Eq. (8)–(14), Table 5 compliant):
      1) Pair x category (speaker change) → Attribute to responder
      2) Individual x category mean (Eq. 12, 13)
      3) Category-wise dyad (Eq. 14)
      4) Final dyad (category average)
      5) Individual overall (category average of individual x category)
    """
    speakers_seq = [t["speaker"] for t in turns]
    uniq = []
    for s in speakers_seq:
        if s not in uniq:
            uniq.append(s)
        if len(uniq) == 2:
            break
    if len(uniq) < 2:
        return {
            "pair_category_scores": [],
            "individual_category_means": {},
            "dyad_category_means": {c: np.nan for c in categories},
            "dyad_final": np.nan,
            "individual_overall": {},
        }

    pair_scores = compute_pair_category_rlsm(turns, categories, eps=eps, zero_tol=zero_tol)
    indiv_cat = aggregate_individual_category_means(pair_scores, categories)
    dyad_cat = aggregate_dyad_category_means(indiv_cat, (uniq[0], uniq[1]), categories, na_policy=na_policy)
    dyad_final = mean_over_categories(dyad_cat)
    indiv_overall = {spk: mean_over_categories(cmap) for spk, cmap in indiv_cat.items()}

    return {
        "pair_category_scores": pair_scores,
        "individual_category_means": indiv_cat,
        "dyad_category_means": dyad_cat,
        "dyad_final": dyad_final,
        "individual_overall": indiv_overall,
    }
