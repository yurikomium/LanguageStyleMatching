#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic rLSM (reciprocal Language Style Matching) Example

This script demonstrates how to calculate rLSM scores for a Japanese conversation.

rLSM extends LSM by measuring temporal reciprocity - how speakers dynamically
adjust their language style in response to each other over the course of a conversation.

Reference: Müller-Frommeyer et al. (2019) "Introducing rLSM: An Integrated Metric
Assessing Temporal Reciprocity in Language Style Matching"
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rlsm.core import compute_rlsm_core
from lsm.core import (
    load_liwc_dic,
    build_compiled_patterns,
    count_liwc_categories,
    count_total_words,
)
import spacy

def main():
    print("=" * 70)
    print("Reciprocal Language Style Matching (rLSM) - Basic Example")
    print("=" * 70)

    # Step 1: Load the sample LIWC dictionary
    print("\n[Step 1] Loading LIWC dictionary...")
    dic_path = "./sample_dictionary/sample_liwc.dic"
    cat_map, word_map = load_liwc_dic(dic_path)
    print(f"  ✓ Loaded {len(cat_map)} categories")

    # Step 2: Build compiled regex patterns
    print("\n[Step 2] Building pattern matchers...")
    allowed_categories = {"ppron", "ipron", "casepart", "auxverb", "adverb", "conj", "negate"}
    patterns = build_compiled_patterns(word_map, cat_map, allowed_categories)
    print(f"  ✓ Built patterns for {len(patterns)} categories")

    # Step 3: Load spaCy model
    print("\n[Step 3] Loading Japanese NLP model...")
    nlp = spacy.load("ja_ginza")
    print("  ✓ Loaded ja_ginza model")

    # Step 4: Load sample conversation data
    print("\n[Step 4] Loading sample conversation...")
    conversation_file = "./sample_data/01_1_1_A_F001_M001_concat.csv"
    df = pd.read_csv(conversation_file)
    print(f"  ✓ Loaded conversation with {len(df)} utterances")

    # Step 5: Prepare turn-by-turn data
    print("\n[Step 5] Processing turn-by-turn conversation...")

    # Convert to turn-by-turn format with speakers and LIWC category usage rates (%)
    turns = []
    for _, row in df.iterrows():
        doc = nlp(row["text"])
        counts = count_liwc_categories(doc, patterns)
        total = count_total_words(doc)
        if total <= 0:
            rates = {c: 0.0 for c in allowed_categories}
        else:
            rates = {c: (counts.get(c, 0) / float(total) * 100.0) for c in allowed_categories}
        speaker = 0 if row['speaker'] == 'female' else 1
        turns.append({
            'speaker': speaker,
            'rates': rates,
        })

    print(f"  ✓ Processed {len(turns)} turns")

    # Step 6: Compute rLSM
    print("\n[Step 6] Computing rLSM scores...")
    print("  (This measures how each speaker adjusts to the other's language style)")

    result = compute_rlsm_core(
        turns=turns,
        categories=sorted(allowed_categories),
        na_policy='bilateral_only'  # Only count when both speakers use a category
    )

    # Step 7: Display results
    print("\n[Step 7] Results:")
    print("  " + "-" * 60)

    # Overall rLSM score (dyad-level, averaged across categories)
    print(f"\n  Overall rLSM Score: {result['dyad_final']:.4f}")
    print(f"  (Range: 0.0 to 1.0, higher = more reciprocal coordination)")

    # Per-speaker scores
    female_to_male = result["individual_overall"].get(1, float("nan"))
    male_to_female = result["individual_overall"].get(0, float("nan"))
    print(f"\n  Female → Male responsiveness: {female_to_male:.4f}")
    print(f"  Male → Female responsiveness: {male_to_female:.4f}")

    # Per-category dyad scores
    print("\n  Per-category dyad rLSM scores:")
    for cat in sorted(allowed_categories):
        if cat in result["dyad_category_means"]:
            score = result["dyad_category_means"][cat]
            if pd.notna(score):
                print(f"    {cat:12s}: {score:.4f}")
            else:
                print(f"    {cat:12s}: N/A (not used by both)")

    print("  " + "-" * 60)

    # Summary and interpretation
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Conversation: {conversation_file}")
    print(f"  Overall rLSM: {result['dyad_final']:.4f}")
    print()
    print("  Interpretation:")
    print("    - rLSM measures turn-by-turn linguistic reciprocity")
    print("    - Higher scores indicate speakers are more responsive to")
    print("      each other's language style changes")
    print("    - Unlike static LSM, rLSM captures temporal dynamics")
    print("=" * 70)

    print("\n✓ Example completed successfully!")
    print("\nKey differences from LSM:")
    print("  1. LSM: Overall similarity across entire conversation")
    print("  2. rLSM: Turn-by-turn responsiveness and adaptation")
    print()
    print("Next steps:")
    print("  - Compare rLSM scores across different conversation rounds")
    print("  - Analyze individual speaker responsiveness patterns")
    print("  - See the paper for interpretation guidelines")
    print()

if __name__ == "__main__":
    main()
