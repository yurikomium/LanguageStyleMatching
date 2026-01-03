#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample dictionary validation script
"""
from lsm.core import load_liwc_dic, build_compiled_patterns

def main():
    dic_path = "./sample_dictionary/sample_liwc.dic"

    print(f"Loading dictionary: {dic_path}")
    cat_map, word_map = load_liwc_dic(dic_path)

    print("\n=== Category Map ===")
    for cat_id, cat_name in sorted(cat_map.items()):
        print(f"  {cat_id}: {cat_name}")

    print(f"\n=== Word Map (total: {len(word_map)} entries) ===")
    for word, cat_ids in sorted(word_map.items())[:10]:
        cat_names = [cat_map[cid] for cid in cat_ids]
        print(f"  {word} -> {cat_names}")
    print("  ...")

    print("\n=== Building Compiled Patterns ===")
    allowed = {"ppron", "ipron", "casepart", "auxverb", "adverb", "conj", "negate"}
    patterns = build_compiled_patterns(word_map, cat_map, allowed)

    print(f"\nSuccessfully compiled patterns for {len(patterns)} categories:")
    for cat_name in sorted(patterns.keys()):
        print(f"  - {cat_name}")

    print("\n✓ Sample dictionary is valid!")

if __name__ == "__main__":
    main()
