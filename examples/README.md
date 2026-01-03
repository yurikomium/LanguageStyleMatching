# Usage Examples

This directory contains example scripts demonstrating how to use the LSM and rLSM implementations.

For installation details, paper-aligned definitions, and Japanese-specific design choices, see:
- `../README.md` (English)
- `../README_ja.md` (日本語)

## Quick Start

### Prerequisites

- Run commands from the repository root (so paths like `./sample_data/...` work).
- If you haven't installed dependencies yet, follow the steps in `../README.md` / `../README_ja.md`.

### Running the Examples

#### 1. Basic LSM Example

Calculates static Language Style Matching score for a conversation:

```bash
python examples/basic_lsm_example.py
```

**What it does:**
- Loads the sample LIWC dictionary
- Analyzes a sample conversation between two speakers
- Computes LSM scores for function word categories
- Prints a mean score across categories

**Expected output:**
```
LSM Mean: 0.XXXX (±0.XXXX)
Higher scores indicate greater linguistic coordination (matching)
```

#### 2. Basic rLSM Example

Calculates reciprocal LSM scores measuring temporal dynamics:

```bash
python examples/basic_rlsm_example.py
```

**What it does:**
- Analyzes turn-by-turn conversation dynamics
- Measures how speakers respond to each other's language style
- Computes individual responsiveness scores
- Shows dyad-level reciprocity

**Expected output:**
```
Overall rLSM Score: 0.XXXX
Female → Male responsiveness: 0.XXXX
Male → Female responsiveness: 0.XXXX
```

## Understanding the Output

### LSM (Language Style Matching)
- **Range:** 0.0 to 1.0
- **Interpretation:**
  - 1.0 = Perfect matching (identical function word usage)
  - 0.5 = Moderate matching
  - 0.0 = No matching (completely different usage patterns)

### rLSM (reciprocal Language Style Matching)
- **Range:** 0.0 to 1.0
- **Interpretation:**
  - Higher scores = Greater temporal reciprocity
  - Measures turn-by-turn adaptation, not just overall similarity
  - Can reveal asymmetric patterns (one speaker more responsive than the other)

## Using Your Own Data

### Data Format Requirements

Conversations should be in CSV format with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `speaker` | string | Speaker identifier (e.g., "female", "male", "A", "B") |
| `text` | string | Utterance text in Japanese |
| `start` | float | (Optional) Utterance start time in seconds |

Example:
```csv
speaker,text,start
female,こんにちは、今日はいい天気ですね。,0.0
male,そうですね。散歩に行きたいです。,3.5
female,いいアイデアですね！,7.2
```

### Using a Full LIWC Dictionary

The examples use a simplified sample dictionary. For actual research:

1. Obtain the official J-LIWC 2015 dictionary (see `sample_dictionary/README.md`)
2. Replace the dictionary path in the scripts:
   ```python
   dic_path = "./path/to/your/Japanese_LIWC_Dictionary.dic"
   ```

## Customization

### Changing Function Word Categories

Edit the `allowed_categories` set in the scripts:

```python
allowed_categories = {
    "ppron",    # Personal pronouns
    "ipron",    # Indefinite pronouns
    "casepart", # Case particles
    # Add or remove categories as needed
}
```

### Adjusting rLSM Parameters

Modify the `na_policy` parameter (how to average dyad scores when one side is missing):

```python
result = compute_rlsm_core(
    turns=turns,
    categories=sorted(allowed_categories),
    na_policy='bilateral_only'  # or 'nanmean' for different handling
)
```

## References

- **LSM**: Gonzales et al. (2010) "Language Style Matching as a Predictor of Social Dynamics in Small Groups"
- **rLSM**: Müller-Frommeyer et al. (2019) "Introducing rLSM: An Integrated Metric Assessing Temporal Reciprocity in Language Style Matching"
- **J-LIWC**: Igarashi et al. (2022) "Development of the Japanese Version of the LIWC Dictionary 2015"

## Troubleshooting

**Error: "No module named 'spacy'"**
- Install dependencies: `pip install -r requirements.txt`

**Error: "Can't find model 'ja_ginza'"**
- The ja-ginza model should install automatically with requirements.txt
- If not, install manually: `pip install ja-ginza`

**Different results than expected?**
- Ensure you're using a proper LIWC dictionary (sample dictionary is for demonstration only)
- Check that conversation data is properly formatted
- Verify speaker labels are consistent

## Next Steps

1. Run the examples with sample data to understand the output
2. Prepare your own conversation data in the required CSV format
3. Obtain and use a full LIWC dictionary for actual research
4. Run tests to verify your installation: `pytest -v`
5. See the main README.md for more details on the implementation
