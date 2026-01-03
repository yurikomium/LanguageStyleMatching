#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic LSM (Language Style Matching) Example

This script demonstrates how to calculate LSM scores for a Japanese conversation
using the sample data and dictionary included in this repository.

LSM measures linguistic coordination between speakers by comparing their use
of function words (pronouns, articles, conjunctions, etc.).
"""

import sys
from pathlib import Path

# Add parent directory to path to import lsm module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from lsm.core import (
    load_liwc_dic,
    build_compiled_patterns,
    count_liwc_categories,
    count_total_words,
    compute_lsm
)
import spacy

def main():
    print("=" * 70)
    print("Language Style Matching (LSM) - Basic Example")
    print("=" * 70)

    # Step 1: Load the sample LIWC dictionary
    print("\n[Step 1] Loading LIWC dictionary...")
    dic_path = "./sample_dictionary/sample_liwc.dic"
    cat_map, word_map = load_liwc_dic(dic_path)
    print(f"  ✓ Loaded {len(cat_map)} categories")

    # Step 2: Build compiled regex patterns for matching
    print("\n[Step 2] Building pattern matchers...")
    allowed_categories = {"ppron", "ipron", "casepart", "auxverb", "adverb", "conj", "negate"}
    patterns = build_compiled_patterns(word_map, cat_map, allowed_categories)
    print(f"  ✓ Built patterns for {len(patterns)} categories")

    # Step 3: Load spaCy model for Japanese tokenization
    print("\n[Step 3] Loading Japanese NLP model...")
    nlp = spacy.load("ja_ginza")
    print("  ✓ Loaded ja_ginza model")

    # Step 4: Load sample conversation data
    print("\n[Step 4] Loading sample conversation...")
    conversation_file = "./sample_data/01_1_1_A_F001_M001_concat.csv"
    df = pd.read_csv(conversation_file)
    print(f"  ✓ Loaded conversation with {len(df)} utterances")
    print(f"    - Female utterances: {(df['speaker'] == 'female').sum()}")
    print(f"    - Male utterances: {(df['speaker'] == 'male').sum()}")

    # Step 5: Separate utterances by speaker
    print("\n[Step 5] Processing conversation...")
    female_text = " ".join(df[df['speaker'] == 'female']['text'].tolist())
    male_text = " ".join(df[df['speaker'] == 'male']['text'].tolist())

    # Tokenize (keep spaCy Token objects, not just text)
    female_doc = nlp(female_text)
    male_doc = nlp(male_text)

    print(f"  ✓ Female tokens: {len(female_doc)}")
    print(f"  ✓ Male tokens: {len(male_doc)}")

    # Step 6: Count function word usage by category
    print("\n[Step 6] Counting function word usage...")
    female_counts = count_liwc_categories(female_doc, patterns)
    male_counts = count_liwc_categories(male_doc, patterns)

    female_total = count_total_words(female_doc)
    male_total = count_total_words(male_doc)

    print(f"  ✓ Female total words: {female_total}")
    print(f"  ✓ Male total words: {male_total}")

    # Step 7: Compute LSM score
    print("\n[Step 7] Computing LSM score...")
    lsm_scores = compute_lsm(
        female_counts, male_counts,
        allowed_categories,
        female_total, male_total
    )

    # Calculate mean and std
    import numpy as np
    if lsm_scores:
        lsm_values = list(lsm_scores.values())
        lsm_mean = np.mean(lsm_values)
        lsm_std = np.std(lsm_values)
        print(f"  ✓ LSM Mean: {lsm_mean:.4f}")
        print(f"  ✓ LSM Std:  {lsm_std:.4f}")
    else:
        lsm_mean = lsm_std = 0.0
        print("  ⚠ No categories with matching usage found")

    # Step 8: Show per-category scores
    print("\n[Step 8] Per-category LSM scores:")
    print("  " + "-" * 40)
    for category in sorted(allowed_categories):
        if category in lsm_scores:
            score = lsm_scores[category]
            print(f"  {category:12s}: {score:.4f}")
        else:
            print(f"  {category:12s}: N/A (not used by both)")
    print("  " + "-" * 40)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Conversation: {conversation_file}")
    if lsm_scores:
        print(f"  LSM Score: {lsm_mean:.4f} (±{lsm_std:.4f})")
        print(f"  Interpretation: Higher scores (closer to 1.0) indicate greater")
        print(f"                  linguistic coordination between speakers.")
    else:
        print(f"  LSM Score: N/A (no matching categories found)")
    print("=" * 70)
    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("  - Try with your own conversation data")
    print("  - Use a full LIWC dictionary for research (see sample_dictionary/README.md)")
    print("  - Explore rLSM for temporal dynamics (see examples/basic_rlsm_example.py)")
    print()

if __name__ == "__main__":
    main()
