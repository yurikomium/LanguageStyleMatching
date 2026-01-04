# Language Style Matching for Japanese

This repository implements **Language Style Matching (LSM)**, **reciprocal / directional LSM (rLSM)**, and **rolling-window rLSM (rw.rLSM)** for Japanese conversational text analysis.

- Japanese README: `README_ja.md`
- Minimal runnable examples: `examples/`
- Sample data: `sample_data/`
- Sample dictionary (demo only): `sample_dictionary/`

For directory-specific details:

- Examples (usage, data format, customization): `examples/README.md`
- Sample data format / naming: `sample_data/README.md`
- Sample dictionary notes / licensing: `sample_dictionary/README.md`

## Project Overview

- **Goal**: Quantify linguistic style coordination (mimicry) focusing on **function words** (style) rather than content words.
- **What’s included**:
  - **LSM**: Conversation-level style similarity (non-directional)
  - **rLSM**: Turn-by-turn responsiveness to the previous speaker (directional)
  - **rw.rLSM**: Rolling-window rLSM to smooth noisy short turns

## Definitions (conceptual summary; see original papers for formulas)

This README summarizes the intended interpretation and implementation scope. Please refer to the original papers for full mathematical definitions.

## Validation & Scope (how we justify “paper-aligned”)

We make the implementation choices explicit and validate key behaviors with tests:

- **rLSM core rules (Table 5-style missingness handling)**: see `rlsm/tests/test_core.py`
- **Paper example checks (where applicable)**: see `rlsm/tests/test_core_paper_examples.py`
- **rw.rLSM (rolling-window) behavior**: see `rlsm/tests/test_rw_core.py` and `rlsm/tests/test_rw_table5.py`

To run the most relevant tests:

```bash
pytest rlsm/tests/test_core.py -v
pytest rlsm/tests/test_core_paper_examples.py -v
```

Scope/limitations to keep in mind:

- **Category mapping is language-specific** (we use a 7-category Japanese mapping; see below). This differs from the original English LIWC setup.
- Results depend on **tokenization/lemmatization** (spaCy + GiNZA) and on the **dictionary** you provide.

### 1) LSM (Language Style Matching; conversation-level similarity)

- **Definition**: Measures how similar two speakers’ **function-word category usage rates** are in a conversation.
- **Unit of analysis**: Typically **conversation-level** (aggregate per speaker, then compare).
- **Range**: Designed to be in **0–1** (higher = more matching).
- **Property**: **Non-directional** (does not indicate “who matched whom”).

### 2) rLSM (reciprocal / directional LSM; immediate-turn responsiveness)

- **Definition**: Addresses the limitation of conversation-level LSM by measuring, turn-by-turn, **how much the responder matches the immediately preceding speaker**.
- **Requirement**: Compute dictionary-based category rates **per turn**.
- **Key property**: rLSM evaluates matching relative to **categories present in the previous turn** (i.e., “followable” categories).
- **Missingness matters**: Turn-level zeros are frequent; treating zeros naively can inflate matching. Müller-Frommeyer et al. (2019) discusses cases where **zeros should be treated as missing** (e.g., 0–0 should not automatically imply perfect match).
- **Aggregation**: Category-wise rLSM can be aggregated into **speaker-level** and **dyad-level** scores.

### 3) rw.rLSM (rolling-window rLSM; smoothed responsiveness)

- **Definition**: Rolling-window variant of rLSM to reduce variance from short turns and capture sustained responsiveness.
- **Typical implementation (paper example)**:
  - Merge adjacent same-speaker turns before computing dictionary features (reduce segmentation noise).
  - Use an **8-utterance window** (previous 7 + current) per speaker to compute window-level rates and then rLSM.

## Japanese implementation choices (category design + preprocessing)

### A) Mapping LIWC function-word categories to Japanese (7 categories)

Original English LIWC uses 9 function-word categories; for Japanese we use the following **7 categories** (default in this repo, based on J-LIWC-style categories):

- `ppron` (personal pronouns)
- `ipron` (indefinite pronouns)
- `casepart` (case particles) — used as an analogue to English prepositions
- `auxverb` (auxiliary verbs)
- `adverb` (adverbs)
- `conj` (conjunctions)
- `negate` (negations)

Rationale:

- Articles / quantifiers are excluded due to weak functional correspondence in Japanese.
- J-LIWC is used only for category identification, not for psychological interpretation.

### B) Tokenization and dictionary matching (reproducibility assumptions)

- We use **spaCy + GiNZA (`ja_ginza`)** and match dictionary entries on **lemmas**.
- For rLSM / rw.rLSM processing, we include a fallback to **`ja_ginza_electra`** if `ja_ginza` is unavailable.

## Installation

### Option 1: Using Dev Container (Recommended)

The easiest way to get started is using VS Code with Dev Containers. This provides a consistent, reproducible environment.

**Prerequisites:**

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**Setup:**

1. Open this repository in VS Code
2. Click "Reopen in Container" when prompted (or use Command Palette: "Dev Containers: Reopen in Container")
3. Wait for the container to build (first time only, ~5-10 minutes)
4. All dependencies are automatically installed!

**What's included:**

- Python 3.10
- All dependencies from `requirements.txt`
- Git for version control
- VS Code extensions (Python, Pylance, Black, Ruff)
- pytest configured and ready to run

**Verifying the setup:**
Once inside the container, run tests to verify everything works:

```bash
pytest -v  # Should show 31 passed tests
```

**Alternative: Using Docker CLI directly**
If you prefer not to use VS Code, you can use Docker directly:

```bash
# Build the image
docker build -f .devcontainer/Dockerfile -t lsm .

# Run tests
docker run --rm -v "$(pwd):/workspace" lsm pytest -v

# Interactive shell
docker run --rm -it -v "$(pwd):/workspace" lsm bash
```

### Option 2: Local Installation

**Prerequisites:**

- Python 3.8 or higher
- pip (Python package installer)

**Setup:**

1. Clone this repository:

```bash
git clone https://github.com/yurikomium/LanguageStyleMatching
cd LanguageStyleMatching
```

2. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

This will install all required packages including:

- pandas, numpy (data processing)
- spacy, ja-ginza (Japanese NLP)
- pytest, pytest-cov, pytest-mock (testing)

## Reproducibility (verified setup + lock file)

### Verified environment (example)

The following environment was used to verify the examples and test suite:

- OS: macOS-26.1-arm64-arm-64bit
- Python: 3.10.2
- Key packages: pandas 2.3.3, numpy 2.2.6, spacy 3.8.11, ginza 5.2.0, ja-ginza 5.2.0, pytest 9.0.2
- Tests: `pytest -q` → `31 passed`

### Dependency lock file (optional but recommended)

`requirements.txt` uses version ranges. For stricter reproducibility, this repository also provides `requirements.lock`
(generated via `pip freeze` in a verified environment).

```bash
pip install -r requirements.lock
```

## Usage

### Quickstart (run minimal examples)

```bash
python examples/basic_lsm_example.py
python examples/basic_rlsm_example.py
```

### Input data format (CSV)

At minimum, a conversation CSV is expected to contain:

- `speaker`: speaker ID (e.g., `female` / `male`, or `A` / `B`)
- `text`: Japanese utterance text
- `start`: (optional) start time in seconds (or similar)

See `examples/README.md` for more details.

### Batch runners (research use)

For processing multiple conversation files and saving CSV outputs:

- LSM:
  - `python -m lsm.runner --dic <dic_path> --results <out_dir> --round 1 2`
- rLSM:
  - `python -m rlsm.runner --data <csv_dir> --dic <dic_path> --results <out_dir>`

Use `--help` to see all options.

## Testing

This repository includes comprehensive test suites for both LSM and rLSM implementations.

### Running All Tests

```bash
pytest -v
```

### Running Specific Test Modules

```bash
# LSM tests only
pytest lsm/tests/ -v

# rLSM tests only
pytest rlsm/tests/ -v

# Specific test file
pytest lsm/tests/test_core.py -v
```

### Test Coverage

To generate a coverage report:

```bash
pytest --cov=lsm --cov=rlsm --cov-report=html
```

### Expected Results

All 31 tests should pass:

- LSM core functionality tests
- rLSM core functionality tests
- Paper example validations (Müller-Frommeyer et al. 2019)
- Worker and runner tests

## Dictionary Notes

- `sample_dictionary/sample_liwc.dic` is a **simplified demo dictionary** for examples.
- For actual research, obtain a full Japanese LIWC-style dictionary (e.g., J-LIWC 2015) and replace the dictionary path (`--dic` or scripts).
- See `sample_dictionary/README.md` for notes on dictionary handling.

## Citation

If you use this repository as part of academic work, you can cite the software metadata in:

- `CITATION.cff`
- `CITATION.bib`

Note: This is an independent implementation and does not claim official equivalence to any external/official implementation.

## References

[1] Gonzales, Amy L., Jeffrey T. Hancock, and James W. Pennebaker. 2010. "Language Style Matching as a Predictor of Social Dynamics in Small Groups." Communication Research 37 (1): 3–19.

[2] Müller-Frommeyer, Lena C., Niels A. M. Frommeyer, and Simone Kauffeld. 2019. "Introducing rLSM: An Integrated Metric Assessing Temporal Reciprocity in Language Style Matching." Behavior Research Methods 51 (3): 1343–59.

[3] Khaleghzadegan, Salar, Michael Rosen, Anne Links, Alya Ahmad, Molly Kilcullen, Emily Boss, Mary Catherine Beach, and Somnath Saha. 2024. "Validating Computer-Generated Measures of Linguistic Style Matching and Accommodation in Patient-Clinician Communication." Patient Education and Counseling 119 (108074): 108074.

[4] Igarashi, Tasuku, Shimpei Okuda, and Kazutoshi Sasahara. 2022. "Development of the Japanese Version of the Linguistic Inquiry and Word Count Dictionary 2015." Frontiers in Psychology 13 (March): 841534.
