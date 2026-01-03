# Sample LIWC Dictionary

This directory contains a simplified LIWC dictionary for **demonstration and testing purposes** of the Language Style Matching (LSM) algorithm.

For how dictionaries are used in this repository (examples, batch runners), see the top-level `README.md` and `README_ja.md`.

## Important Notice

⚠️ **This sample dictionary is NOT suitable for research purposes**

- This dictionary is a **synthetic demonstration dataset**, not derived from or extracted from the official J-LIWC dictionary
- Created solely for code verification and demonstration purposes
- For actual research and analysis, please use an official Japanese LIWC dictionary
- This sample contains only limited vocabulary and does not guarantee linguistic validity

## About Official LIWC Dictionaries

For serious research, we recommend using the official Japanese LIWC dictionary:

- **J-LIWC 2015** - Japanese version of the Linguistic Inquiry and Word Count Dictionary 2015
  - Reference: Igarashi, Tasuku, Shimpei Okuda, and Kazutoshi Sasahara. 2022. "Development of the Japanese Version of the Linguistic Inquiry and Word Count Dictionary 2015." *Frontiers in Psychology* 13 (March): 841534.
  - Availability: Please refer to the [official LIWC website](https://www.liwc.app/)
  - Licensing: Separate license required for commercial use

⚠️ **Due to copyright restrictions, we cannot include the official J-LIWC dictionary in this repository**

## File Format

Format of `sample_liwc.dic`:

```
%
<CategoryID>	<CategoryName>
...
%
<WordPattern>	<CategoryID> [<CategoryID2> ...]
...
```

- `%` lines separate sections (category definitions → vocabulary)
- Category definition: `ID <TAB> LabelName` (e.g., `1	ppron`)
- Vocabulary: `Pattern <TAB> ID...` (multiple category assignments possible)
- Wildcard `*` supported (for matching conjugated forms)

## Category List

This sample dictionary includes the following categories:

| ID | Category | Description | Examples (Japanese) |
|----|----------|-------------|---------------------|
| 1 | ppron | Personal pronouns | 私 (watashi/I), 僕 (boku/I), あなた (anata/you), 彼 (kare/he) |
| 2 | ipron | Impersonal pronouns | 何 (nani/what), これ (kore/this), それ (sore/that), どこ (doko/where) |
| 3 | casepart | Case particles | が (ga), を (wo), に (ni), で (de) |
| 4 | auxverb | Auxiliary verbs | です (desu), ます (masu), た (ta), れる (reru) |
| 5 | adverb | Adverbs | とても (totemo/very), すごく (sugoku/extremely), きっと (kitto/surely) |
| 6 | conj | Conjunctions | そして (soshite/and), しかし (shikashi/however), でも (demo/but) |
| 7 | negate | Negation | ない (nai/not), ず (zu/not), ぬ (nu/not), まい (mai/not) |

## Usage

Specify with the `--dic` option when running LSM scripts:

```bash
python -m lsm.runner \
  --round 1 \
  --dic ./sample_dictionary/sample_liwc.dic \
  --results ./results/lsm
```

## License

This sample dictionary file is released under the same MIT License as this repository.
However, official LIWC dictionaries have their own separate licenses.
