from collections import Counter
import re

"""
=============================================================================
LSM Function Definitions
=============================================================================
"""


# --- Library imports ---
import spacy
import re
from collections import Counter
import pandas as pd

# --- Load J-LIWC dictionary ---
def load_liwc_dic(path):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    cat_map = {}
    word_map = []
    mode = "none"

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
            if not line:
                continue
        if line.startswith('%'):
            mode = "category" if mode == "none" else "words"
            continue
        if mode == "category":
            num, label = line.split(None, 1)
            cat_map[num] = label.strip()
        elif mode == "words":
            parts = line.split()
            word = parts[0]
            cats = parts[1:]
            word_map.append((word, cats))
    return cat_map, word_map


# --- Pre-compile regex + category filter (#3, #5) ---
def build_compiled_patterns(word_map, cat_map, allowed_labels):
    """
    word_map: [(pattern, [cat_ids...]), ...]
    allowed_labels: set of category names (labels) to use
    return: [(compiled_regex, set(label_names)), ...]
    """
    compiled = []
    for pattern, cats in word_map:
        # Convert cat_id -> label
        labels = [cat_map[c] for c in cats if c in cat_map]
        # Keep only desired categories
        labels = [lab for lab in labels if lab in allowed_labels]
        if not labels:
            continue

        # Safe regex: escape all except "*" which becomes wildcard (#5)
        regex = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
        try:
            compiled.append((re.compile(regex), set(labels)))
        except re.error:
            # Skip broken patterns
            pass
    return compiled


# --- Token -> Category (deduplication + full match + cache) (#4, #5) ---
def match_token_to_categories(token_lemma, compiled_patterns, cache):
    if token_lemma in cache:
        return cache[token_lemma]
    matched = set()
    for rx, labels in compiled_patterns:
        if rx.fullmatch(token_lemma):  # fullmatch is explicit since ^...$
            matched.update(labels)
    cache[token_lemma] = matched
    return matched


# --- Count category occurrences (no double-counting of same category) (#4, #6, #7) ---
def count_liwc_categories(doc, compiled_patterns):
    counter = Counter()
    cache = {}
    for token in doc:
        # Exclude punctuation, whitespace, symbols to match total word count definition (#7)
        if token.is_space or token.is_punct or token.pos_ in {"PUNCT", "SYM"}:
            continue
        lemma = token.lemma_ or token.text  # Fallback when lemmatizer not provided (#6)
        cats = match_token_to_categories(lemma, compiled_patterns, cache)
        # cats is a set: prevents duplicate counting of same token x same category (#4)
        counter.update(cats)
    return counter


# --- Total word count (denominator = total words correction) (#1, #7) ---
def count_total_words(doc):
    return sum(
        1
        for t in doc
        if not t.is_space and not t.is_punct and t.pos_ not in {"PUNCT", "SYM"}
    )


# --- Compute LSM score (exclude categories where both are 0%) (#1, #2) ---
def compute_lsm(count1, count2, target_categories, total_words1, total_words2, rounding=None):
    """
    total_wordsX is "total word count".
    Categories where p1+p2==0 are excluded from average (not added to dict).
    """
    lsm_scores = {}
    if total_words1 == 0 or total_words2 == 0:
        return lsm_scores

    for cat in target_categories:
        p1 = count1.get(cat, 0) / total_words1
        p2 = count2.get(cat, 0) / total_words2
        if p1 + p2 == 0:
            # Exclude when both are 0% (#2)
            continue
        lsm = 1 - abs(p1 - p2) / (p1 + p2)
        lsm_scores[cat] = (round(lsm, rounding) if rounding is not None else lsm)
    return lsm_scores